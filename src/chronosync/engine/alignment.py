"""Clock-lag estimation, gap-guarded resampling, and asof reference pairing.

No `mcp` imports allowed in this package.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .config import SyncConfig


def _clean_arrays(t, v) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype="float64")
    v = np.asarray(v, dtype="float64")
    keep = np.isfinite(t) & np.isfinite(v)
    t, v = t[keep], v[keep]
    order = np.argsort(t, kind="mergesort")
    t, v = t[order], v[order]
    _, first = np.unique(t, return_index=True)
    first = np.sort(first)
    return t[first], v[first]


def build_target_grid(start: float, end: float, step_seconds: float) -> np.ndarray:
    n = int(np.floor((end - start) / step_seconds))
    return start + step_seconds * np.arange(n + 1, dtype="float64")


def resample_gap_guarded(
    t, v, grid, method: str = "linear", max_gap_seconds: float = 30.0
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate onto ``grid`` but never across a gap longer than
    ``max_gap_seconds``, and never outside the sampled range.

    Returns ``(values, is_dropout)``. Values at grid points bracketed by a real
    hardware gap > max_gap are NaN and flagged ``is_dropout=True``; grid points
    outside the sampled range are NaN (not extrapolated) and not flagged.
    """
    t, v = _clean_arrays(t, v)
    grid = np.asarray(grid, dtype="float64")
    min_pts = 4 if method == "cubic" else 2
    if len(t) < min_pts or len(grid) == 0:
        return np.full(grid.shape, np.nan), np.zeros(grid.shape, dtype=bool)

    n = len(t)
    idx = np.searchsorted(t, grid, side="left")
    left_i = np.clip(idx - 1, 0, n - 1)
    right_i = np.clip(idx, 0, n - 1)

    exact = (idx > 0) & (grid == t[left_i])
    exact |= (idx < n) & (grid == t[right_i])
    inside = (idx > 0) & (idx < n) & (grid > t[0]) & (grid < t[-1])
    bracket_gap = t[right_i] - t[left_i]

    if method == "cubic":
        interp_vals = np.asarray(PchipInterpolator(t, v)(grid), dtype="float64")
    else:
        interp_vals = np.interp(grid, t, v)

    valid = exact | (inside & (bracket_gap <= max_gap_seconds))
    interp_vals[~valid] = np.nan
    is_dropout = inside & (bracket_gap > max_gap_seconds) & ~exact
    return interp_vals, is_dropout


def _smooth_nan_aware(values: np.ndarray, window: int) -> np.ndarray:
    if window < 3:
        return values.astype("float64").copy()
    s = pd.Series(values, dtype="float64")
    return s.rolling(window, center=True, min_periods=max(2, window // 2)).mean().to_numpy()


def estimate_clock_lag(
    ref_t,
    ref_v,
    cand_t,
    cand_v,
    *,
    max_lag_seconds: float = 90.0,
    resolution_seconds: float = 0.05,
    max_gap_seconds: float = 30.0,
    smooth_seconds: float = 2.0,
    min_pairs: int = 8,
) -> dict:
    """Estimate the constant clock offset of a candidate stream against a reference.

    Computes r(lag) = pearson( reference(t), candidate(t + lag) ) over a lag
    grid, evaluating the candidate from its own samples (light smoothing, no
    values from inside hardware dropouts). A positive ``lag_seconds`` means the
    candidate's timestamps run late by that amount; correcting it means
    subtracting the lag from the candidate's timestamps.

    Returns ``{"lag_seconds", "peak_correlation", "n_pairs", "lag_grid",
    "correlation"}`` — the last two downsampled for reporting.
    """
    ref_t, ref_v = _clean_arrays(ref_t, ref_v)
    cand_t, cand_v = _clean_arrays(cand_t, cand_v)
    empty = {
        "lag_seconds": None,
        "peak_correlation": None,
        "n_pairs": 0,
        "lag_grid": [],
        "correlation": [],
    }
    if len(ref_t) < max(min_pairs, 4) or len(cand_t) < 4:
        return empty

    dt = np.diff(cand_t)
    pos = dt[dt > 1e-9]
    hz = min(2.0, 1.0 / float(np.median(pos))) if len(pos) else 1.0
    step_c = 1.0 / max(hz, 1e-6)
    grid_c = build_target_grid(cand_t[0], cand_t[-1], step_c)
    vals_c, _ = resample_gap_guarded(cand_t, cand_v, grid_c, "linear", max_gap_seconds)
    win = int(round((smooth_seconds or 0.0) / step_c))
    if win >= 3:
        vals_c = _smooth_nan_aware(vals_c, win if win % 2 else win + 1)

    lags = np.arange(-max_lag_seconds, max_lag_seconds + resolution_seconds / 2, resolution_seconds)
    corr = np.full(lags.shape, np.nan)
    lo_c, hi_c = grid_c[0], grid_c[-1]

    for i, lag in enumerate(lags):
        inst = ref_t + lag
        mask = (inst >= lo_c) & (inst <= hi_c)
        if mask.sum() < min_pairs:
            continue
        cv = np.interp(inst[mask], grid_c, vals_c)
        rv = ref_v[mask]
        ok = np.isfinite(cv)
        if ok.sum() < min_pairs:
            continue
        x = rv[ok] - rv[ok].mean()
        y = cv[ok] - cv[ok].mean()
        den = np.sqrt(float((x * x).sum()) * float((y * y).sum()))
        if den <= 0:
            continue
        corr[i] = float((x * y).sum() / den)

    finite = np.isfinite(corr)
    if not finite.any():
        return empty
    i = int(np.argmax(np.where(finite, corr, -np.inf)))
    lag_best = float(lags[i])
    peak = float(corr[i])

    if 0 < i < len(lags) - 1 and np.isfinite(corr[i - 1]) and np.isfinite(corr[i + 1]):
        y0, y1, y2 = corr[i - 1], corr[i], corr[i + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom < 0:
            delta = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
            lag_best = float(
                np.clip(lags[i] + delta * resolution_seconds, lags[i - 1], lags[i + 1])
            )

    inst_best = ref_t + lag_best
    mask = (inst_best >= lo_c) & (inst_best <= hi_c)
    n_pairs = int(np.isfinite(np.interp(inst_best[mask], grid_c, vals_c)).sum()) if mask.any() else 0

    stride = max(1, len(lags) // 201)
    curve = [None if not np.isfinite(c) else round(float(c), 6) for c in corr[::stride]]
    return {
        "lag_seconds": round(lag_best, 6),
        "peak_correlation": round(peak, 6),
        "n_pairs": n_pairs,
        "lag_grid": [round(float(x), 4) for x in lags[::stride]],
        "correlation": curve,
    }


def asof_reference_pairs(
    reference_t,
    cand_t,
    cand_v,
    *,
    tolerance_seconds: float = 5.0,
    max_gap_seconds: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge-asof semantics: for each reference instant take the nearest actual
    candidate sample, within ``tolerance_seconds``. No interpolation is faked at
    reference points. Returns ``(values, is_dropout)`` where a pair is flagged
    as inside a hardware dropout when the candidate's neighbouring samples
    around that instant span a gap > ``max_gap_seconds``."""
    t, v = _clean_arrays(cand_t, cand_v)
    g = np.asarray(reference_t, dtype="float64")
    values = np.full(g.shape, np.nan)
    is_dropout = np.zeros(g.shape, dtype=bool)
    if len(t) == 0 or len(g) == 0:
        return values, is_dropout

    n = len(t)
    idx = np.searchsorted(t, g, side="left")
    left_i = np.clip(idx - 1, 0, n - 1)
    right_i = np.clip(idx, 0, n - 1)
    has_left = idx > 0
    has_right = idx < n
    d_left = np.where(has_left, g - t[left_i], np.inf)
    d_right = np.where(has_right, t[right_i] - g, np.inf)
    nearest = np.minimum(d_left, d_right)
    use_left = d_left <= d_right
    paired_i = np.where(use_left, left_i, right_i)
    ok = nearest <= tolerance_seconds
    values[ok] = v[paired_i[ok]]

    bracket_gap = t[right_i] - t[left_i]
    inside = has_left & has_right
    is_dropout = ok & inside & (bracket_gap > max_gap_seconds)
    return values, is_dropout


def harmonize_wide(
    streams: dict[str, tuple[np.ndarray, np.ndarray]],
    reference_id: str,
    config: SyncConfig,
) -> tuple[pd.DataFrame, float, float]:
    """Resample every stream onto one shared regular grid.

    The reference stream is never interpolated: grid points coincide with real
    reference samples or stay NaN (everything else would be hallucinated gold).
    Candidate streams use gap-guarded interpolation per ``config``.
    """
    starts = [np.asarray(t, float)[0] for t, _ in streams.values() if len(t)]
    ends = [np.asarray(t, float)[-1] for t, _ in streams.values() if len(t)]
    if not starts:
        raise ValueError("no usable streams to harmonize")
    g0, g1 = max(starts), min(ends)
    if g1 <= g0:
        raise ValueError("streams do not overlap in time; cannot harmonize")

    grid = build_target_grid(g0, g1, config.target_freq_seconds)
    wide = {
        "t": grid,
        "t_utc": [
            datetime.fromtimestamp(float(x), tz=timezone.utc).isoformat(timespec="milliseconds")
            for x in grid
        ],
    }
    for sid, (t, v) in streams.items():
        t = np.asarray(t, float)
        v = np.asarray(v, float)
        if sid == reference_id:
            values = np.full(grid.shape, np.nan)
            if len(t):
                idx = np.searchsorted(t, grid, side="left")
                n = len(t)
                li = np.clip(idx - 1, 0, n - 1)
                ri = np.clip(idx, 0, n - 1)
                hit_left = (idx > 0) & (np.abs(grid - t[li]) <= 1e-6)
                hit_right = (idx < n) & (np.abs(grid - t[ri]) <= 1e-6)
                values[hit_left] = v[li[hit_left]]
                values[hit_right] = v[ri[hit_right]]
            inside = (grid > t[0]) & (grid < t[-1]) if len(t) else np.zeros_like(grid, bool)
            is_dropout = inside & ~np.isfinite(values)
        else:
            values, is_dropout = resample_gap_guarded(
                t, v, grid, config.interp_method, config.max_gap_seconds
            )
        wide[sid] = values
        wide[f"{sid}_is_dropout"] = is_dropout
    return pd.DataFrame(wide), g0, g1
