"""Reliability auditing: error stats, Bland-Altman, CCC, OLS drift detection,
and disagreement-window detection against synthetic streams with injected truth."""

import numpy as np
import pytest

from chronosync.engine.reliability import evaluate_reliability


def _dataset(seed=11, drift_per_hour=0.3, bias=1.5, noise=0.2):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 7201.0, 60.0)
    ref = 100.0 + 8.0 * np.sin(2 * np.pi * t / 1800.0)
    cand = ref + bias + drift_per_hour * (t / 3600.0) + rng.normal(0.0, noise, t.size)
    return t, ref, cand


def test_error_stats_and_bland_altman_track_injected_truth():
    t, ref, cand = _dataset()
    m, _ = evaluate_reliability(t, ref, cand)

    assert m.n_matched == t.size
    assert m.n_reference_points == t.size
    assert m.coverage_pct == pytest.approx(100.0)

    # true mean error = bias + mean drift over the 2 h window = 1.5 + 0.3
    assert m.mean_bias == pytest.approx(1.8, abs=0.1)
    assert 0.2 < m.sd_diff < 0.4
    # metrics are rounded to 6 decimals for the JSON artifact; identity holds
    # to within that output precision
    assert m.loa_upper - m.loa_lower == pytest.approx(2 * 1.96 * m.sd_diff, abs=2e-6)
    assert m.pct_outside_loa < 10.0

    assert 1.6 < m.mae < 2.0
    assert 1.6 < m.rmse < 2.1
    assert m.rmse >= m.mae

    # noise ceiling: var_ref/(var_ref+0.04) bounds r ~0.9989 for this synthetic
    assert m.pearson_r > 0.998
    assert m.lins_ccc > 0.93


def test_identical_streams_prove_perfect_agreement():
    t, ref, _ = _dataset()
    m, windows = evaluate_reliability(t, ref, ref.copy())
    assert m.mean_bias == pytest.approx(0.0)
    assert m.sd_diff == pytest.approx(0.0)
    assert m.mae == pytest.approx(0.0)
    assert m.pearson_r == pytest.approx(1.0)
    assert m.lins_ccc == pytest.approx(1.0)
    assert m.pct_outside_loa == 0.0
    assert not m.drift_significant
    assert windows == []


def test_significant_calibration_drift_is_detected():
    t, ref, cand = _dataset(drift_per_hour=0.3)
    m, _ = evaluate_reliability(t, ref, cand)
    assert m.drift_per_hour == pytest.approx(0.3, abs=0.08)
    assert m.drift_p_value < 1e-6
    assert m.drift_significant is True


def test_pure_bias_without_drift_is_not_flagged_as_drift():
    t, ref, cand = _dataset(drift_per_hour=0.0)
    m, _ = evaluate_reliability(t, ref, cand)
    assert abs(m.drift_per_hour) < 0.08
    assert m.drift_significant is False


def test_localized_disagreement_window_is_detected():
    t, ref, cand = _dataset(drift_per_hour=0.0)
    cand = cand.copy()
    block = (t >= 3600.0) & (t < 3960.0)
    cand[block] += 6.0

    _, windows = evaluate_reliability(t, ref, cand)
    assert len(windows) == 1
    w = windows[0]
    assert w.n_points == 6
    assert w.duration_seconds == pytest.approx(300.0)
    assert 6.0 < w.max_abs_error < 9.0
    assert w.mean_abs_error > 6.0

    _, windows2 = evaluate_reliability(t, ref, cand, anomaly_threshold=3.0)
    assert len(windows2) == 1
    assert windows2[0].n_points == 6


def test_dropout_flagged_pairs_are_excluded_from_statistics():
    t, ref, cand = _dataset(drift_per_hour=0.0)
    corrupted = cand.copy()
    block = (t >= 3600.0) & (t < 3960.0)
    corrupted[block] += 50.0
    flags = np.zeros(t.size, dtype=bool)
    flags[block] = True

    m_clean, windows_clean = evaluate_reliability(t, ref, corrupted, flags,
                                                  anomaly_threshold=3.0)
    assert m_clean.n_matched == t.size - block.sum()
    assert m_clean.mean_bias == pytest.approx(1.5, abs=0.1)
    assert windows_clean == []

    m_dirty, windows_dirty = evaluate_reliability(t, ref, corrupted,
                                                 anomaly_threshold=3.0)
    assert m_dirty.n_matched == t.size
    assert len(windows_dirty) == 1
