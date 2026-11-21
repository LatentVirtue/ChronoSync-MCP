"""Gap-guarded resampling: long hardware dropouts stay NaN and flagged; short
gaps interpolate; no extrapolation past sampled bounds; asof pairing semantics."""

import numpy as np

from chronosync.engine.alignment import (
    asof_reference_pairs,
    build_target_grid,
    resample_gap_guarded,
)


def _stream_with_120s_dropout():
    t = np.arange(0.0, 601.0, 1.0)
    v = 50.0 + 3.0 * np.sin(2 * np.pi * t / 120.0)
    keep = ~((t > 200.0) & (t < 320.0))
    return t[keep], v[keep]


def test_long_dropout_stays_nan_and_is_flagged():
    t, v = _stream_with_120s_dropout()
    grid = build_target_grid(0.0, 600.0, 1.0)
    vals, is_dropout = resample_gap_guarded(t, v, grid, "linear", max_gap_seconds=30.0)

    interior = (grid > 200.5) & (grid < 319.5)
    assert np.all(np.isnan(vals[interior]))
    assert np.all(is_dropout[interior])

    edges = np.isin(grid, [200.0, 320.0])
    assert np.all(np.isfinite(vals[edges]))
    assert not np.any(is_dropout[edges])

    adjacent = (grid >= 150.0) & (grid <= 199.0)
    assert np.all(np.isfinite(vals[adjacent]))
    assert not np.any(is_dropout[adjacent])


def test_short_gap_interpolates_and_is_not_flagged():
    t = np.arange(0.0, 601.0, 1.0)
    v = np.full(t.shape, 42.0)
    keep = ~((t > 200.0) & (t < 220.0))  # 20 s gap
    t, v = t[keep], v[keep]
    grid = build_target_grid(0.0, 600.0, 1.0)

    vals, is_dropout = resample_gap_guarded(t, v, grid, "linear", max_gap_seconds=30.0)
    at_gap = np.isclose(grid, 210.0)
    assert np.allclose(vals[at_gap], 42.0)
    assert not np.any(is_dropout[at_gap])

    vals2, is_dropout2 = resample_gap_guarded(t, v, grid, "linear", max_gap_seconds=10.0)
    assert np.all(np.isnan(vals2[at_gap]))
    assert np.all(is_dropout2[at_gap])


def test_no_extrapolation_outside_sampled_range():
    t, v = _stream_with_120s_dropout()
    grid = build_target_grid(-5.0, 650.0, 1.0)
    vals, is_dropout = resample_gap_guarded(t, v, grid, "linear", max_gap_seconds=30.0)
    out_of_range = (grid > 600.5) | (grid < -0.5)
    assert np.all(np.isnan(vals[out_of_range]))
    assert not np.any(is_dropout[out_of_range])


def test_cubic_respects_identical_guards_and_tracks_the_signal():
    t, v = _stream_with_120s_dropout()
    grid = build_target_grid(0.0, 600.0, 0.5)
    lin, drop_lin = resample_gap_guarded(t, v, grid, "linear", max_gap_seconds=30.0)
    cub, drop_cub = resample_gap_guarded(t, v, grid, "cubic", max_gap_seconds=30.0)

    assert np.array_equal(np.isnan(lin), np.isnan(cub))
    assert np.array_equal(drop_lin, drop_cub)

    truth = 50.0 + 3.0 * np.sin(2 * np.pi * grid / 120.0)
    good = np.isfinite(cub)
    assert np.max(np.abs(cub[good] - truth[good])) < 0.01


def test_asof_matches_nearest_sample_within_tolerance_without_interpolating():
    t = np.arange(0.0, 101.0, 1.0)
    v = t * 2.0
    refs = np.array([10.2, 99.4, 106.0, 250.0])
    vals, is_dropout = asof_reference_pairs(
        refs, t, v, tolerance_seconds=5.0, max_gap_seconds=30.0
    )
    assert np.isclose(vals[0], v[10])  # nearest sample to 10.2
    assert np.isclose(vals[1], v[99])  # nearest sample to 99.4
    assert np.isnan(vals[2])  # nearest sample is 6 s away > 5 s tolerance
    assert np.isnan(vals[3])


def test_asof_flags_pairs_forced_across_a_hardware_gap():
    t = np.array([0.0, 20.0, 80.0, 100.0])
    v = np.array([0.0, 1.0, 2.0, 3.0])
    vals, is_dropout = asof_reference_pairs(
        np.array([50.0]), t, v, tolerance_seconds=40.0, max_gap_seconds=30.0
    )
    assert np.isclose(vals[0], 1.0)  # nearest sample (tie breaks left)
    assert bool(is_dropout[0])  # bracketing samples span a 60 s gap
