"""Reliability auditing against a reference: error stats, Bland-Altman, Lin's CCC,
calibration drift, and disagreement windows.

No `mcp` imports allowed in this package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import AnomalyWindow, ReliabilityMetrics
from .profiler import epoch_to_utc


def evaluate_reliability(
    t,
    reference,
    candidate,
    is_dropout=None,
    *,
    candidate_id: str = "candidate",
    reference_id: str = "reference",
    alpha: float = 0.05,
    anomaly_threshold: float | None = None,
) -> tuple[ReliabilityMetrics, list[AnomalyWindow]]:
    """Score paired candidate/reference samples (already matched in time).

    NaN candidate entries and dropout-flagged entries are excluded from all
    statistics. Disagreement windows are contiguous runs of paired points whose
    |error| exceeds ``anomaly_threshold`` (defaults to the 1.96-SD Bland-Altman
    half-width). Returns ``(metrics, windows_sorted_by_severity)``.
    """
    t = np.asarray(t, dtype="float64")
    ref = np.asarray(reference, dtype="float64")
    cand = np.asarray(candidate, dtype="float64")
    n_ref_total = int(len(t))

    mask = np.isfinite(ref) & np.isfinite(cand) & np.isfinite(t)
    if is_dropout is not None:
        mask &= ~np.asarray(is_dropout, dtype=bool)
    order = np.argsort(t[mask], kind="mergesort")
    t, ref, cand = t[mask][order], ref[mask][order], cand[mask][order]
    n = int(len(t))

    metrics = ReliabilityMetrics(
        candidate_id=candidate_id,
        reference_id=reference_id,
        n_matched=n,
        n_reference_points=n_ref_total,
        coverage_pct=round(100.0 * n / n_ref_total, 3) if n_ref_total else 0.0,
    )
    windows: list[AnomalyWindow] = []
    if n < 2:
        return metrics, windows

    err = cand - ref
    metrics.mae = round(float(np.mean(np.abs(err))), 6)
    metrics.rmse = round(float(np.sqrt(np.mean(err**2))), 6)

    sd_ref = float(np.std(ref))
    sd_cand = float(np.std(cand))
    if sd_ref > 0 and sd_cand > 0:
        r = float(np.corrcoef(ref, cand)[0, 1])
        cov = float(np.mean((ref - ref.mean()) * (cand - cand.mean())))
        ccc = 2.0 * cov / (sd_ref**2 + sd_cand**2 + (ref.mean() - cand.mean()) ** 2)
        metrics.pearson_r = round(r, 6)
        metrics.lins_ccc = round(ccc, 6)
    elif np.allclose(ref, cand):
        metrics.pearson_r = 1.0
        metrics.lins_ccc = 1.0

    bias = float(np.mean(err))
    sd = float(np.std(err, ddof=1))
    metrics.mean_bias = round(bias, 6)
    metrics.sd_diff = round(sd, 6)
    metrics.loa_lower = round(bias - 1.96 * sd, 6)
    metrics.loa_upper = round(bias + 1.96 * sd, 6)
    if sd > 0:
        metrics.pct_outside_loa = round(float(100.0 * np.mean(np.abs(err - bias) > 1.96 * sd)), 3)
    else:
        metrics.pct_outside_loa = 0.0

    hours = (t - t[0]) / 3600.0
    if n >= 3 and np.ptp(hours) > 0 and np.std(err) > 0:
        lr = stats.linregress(hours, err)
        metrics.drift_per_hour = round(float(lr.slope), 6)
        metrics.drift_stderr_per_hour = round(float(lr.stderr), 6)
        p = float(lr.pvalue) if np.isfinite(lr.pvalue) else 1.0
        metrics.drift_p_value = round(p, 8)
        metrics.drift_significant = bool(p < alpha)
    else:
        metrics.drift_per_hour = 0.0
        metrics.drift_stderr_per_hour = 0.0
        metrics.drift_p_value = 1.0
        metrics.drift_significant = False

    threshold = anomaly_threshold
    if threshold is None and sd > 0:
        threshold = 1.96 * sd
    if threshold is not None and threshold > 0:
        flags = np.abs(err) > threshold
        i = 0
        while i < n:
            if not flags[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            seg_err = np.abs(err[i : j + 1])
            windows.append(
                AnomalyWindow(
                    candidate_id=candidate_id,
                    start_utc=epoch_to_utc(t[i]),
                    end_utc=epoch_to_utc(t[j]),
                    duration_seconds=round(float(t[j] - t[i]), 3),
                    n_points=int(j - i + 1),
                    max_abs_error=round(float(seg_err.max()), 6),
                    mean_abs_error=round(float(seg_err.mean()), 6),
                )
            )
            i = j + 1
    windows.sort(key=lambda w: w.max_abs_error, reverse=True)
    return metrics, windows


def rolling_error(t, err, window_points: int | None = None) -> np.ndarray:
    """Centered rolling mean of signed paired error, for the drift plot."""
    err = np.asarray(err, dtype="float64")
    n = len(err)
    if n == 0:
        return err
    w = int(window_points or max(3, n // 10))
    if w < 3:
        w = 3
    s = pd.Series(err)
    return s.rolling(w, center=True, min_periods=max(2, w // 2)).mean().to_numpy()
