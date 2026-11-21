"""Three-stream synthetic benchmark with known ground truth.

Streams:
- ``wearable_pat``: ~10 Hz, jittered sampling, two long dropouts (150 s, 90 s),
  a +14 s clock lag, constant bias 0.75, linear calibration drift 0.35/h, noise.
- ``aux_hri``: 1 Hz, +3 s clock lag, bias 1.5, no drift, no dropouts.
- ``reference_arms``: sparse gold standard every 60 s, near noise-free, no lag.

All three share one deterministic underlying "true" signal, so injected lags,
dropouts, bias, and drift are exactly recoverable by the pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

WEARABLE_ID = "wearable_pat"
AUX_ID = "aux_hri"
REFERENCE_ID = "reference_arms"

BASE_UTC_EPOCH = datetime(2026, 11, 21, tzinfo=timezone.utc).timestamp()

TRUE_LAG_SECONDS = {WEARABLE_ID: 14.0, AUX_ID: 3.0, REFERENCE_ID: 0.0}
WEARABLE_DROPOUTS_TRUE = ((2000.0, 2150.0), (4600.0, 4690.0))  # true-time bounds
WEARABLE_BIAS = 0.75
WEARABLE_DRIFT_PER_HOUR = 0.35
WEARABLE_NOISE = 0.35
AUX_BIAS = 1.5
AUX_NOISE = 1.0
REFERENCE_NOISE = 0.05
DURATION_SECONDS = 7200.0


def _true_signal(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype="float64")
    return (
        100.0
        + 6.0 * np.sin(2 * np.pi * t / 430.0 + 0.8)
        + 4.2 * np.sin(2 * np.pi * t / 233.0 + 1.7)
        + 2.6 * np.sin(2 * np.pi * t / 131.0 + 2.6)
        + 1.1 * np.sin(2 * np.pi * t / 71.0 + 0.35)
    )


def _iso(clock_seconds: np.ndarray) -> np.ndarray:
    return [
        datetime.fromtimestamp(BASE_UTC_EPOCH + float(c), tz=timezone.utc).isoformat(
            timespec="microseconds"
        )
        for c in clock_seconds
    ]


def _messify(frame: pd.DataFrame, rng: np.random.Generator, swaps: int, duplicates: int) -> pd.DataFrame:
    """Introduce out-of-order rows (row swaps) and duplicate retransmissions."""
    n = len(frame)
    df = frame.reset_index(drop=True)
    positions = rng.choice(np.arange(1, n - 2), size=min(swaps, max(1, n // 40)), replace=False)
    for i in positions:
        df.loc[[i, i + 1]] = df.loc[[i + 1, i]].values
    dup_rows = df.iloc[rng.choice(n, size=min(duplicates, n), replace=False)]
    return pd.concat([df, dup_rows], ignore_index=True)


def generate_benchmark(seed: int = 42) -> tuple[dict[str, pd.DataFrame], dict]:
    """Return (frames, truth). Frames are CSV-shaped DataFrames with columns
    ``timestamp`` (ISO-8601 UTC, device clock) and ``value``."""
    rng = np.random.default_rng(seed)

    # --- wearable, ~10 Hz, jittered, two dropouts, +14 s clock lag --------------
    t_true = np.arange(200.0, DURATION_SECONDS + 400.0, 0.1)
    jitter = np.clip(rng.normal(0.0, 0.008, t_true.size), -0.03, 0.03)
    t_true = t_true + jitter
    drop_mask = np.zeros(t_true.size, dtype=bool)
    for lo, hi in WEARABLE_DROPOUTS_TRUE:
        drop_mask |= (t_true > lo) & (t_true < hi)
    t_true = t_true[~drop_mask]
    n_w = t_true.size
    hours_w = (t_true - 200.0) / 3600.0
    v_w = (
        _true_signal(t_true)
        + WEARABLE_BIAS
        + WEARABLE_DRIFT_PER_HOUR * hours_w
        + rng.normal(0.0, WEARABLE_NOISE, n_w)
    )
    clock_w = t_true + TRUE_LAG_SECONDS[WEARABLE_ID]
    frame_w = pd.DataFrame({"timestamp": _iso(clock_w), "value": v_w})

    # --- auxiliary, 1 Hz, +3 s clock lag -----------------------------------------
    t_aux = np.arange(100.0, DURATION_SECONDS + 500.0, 1.0)
    t_aux = t_aux + np.clip(rng.normal(0.0, 0.05, t_aux.size), -0.2, 0.2)
    v_aux = _true_signal(t_aux) + AUX_BIAS + rng.normal(0.0, AUX_NOISE, t_aux.size)
    clock_aux = t_aux + TRUE_LAG_SECONDS[AUX_ID]
    frame_aux = pd.DataFrame({"timestamp": _iso(clock_aux), "value": v_aux})

    # --- sparse gold-standard reference, every 60 s ------------------------------
    t_ref = np.arange(0.0, DURATION_SECONDS + 420.0, 60.0)
    v_ref = _true_signal(t_ref) + rng.normal(0.0, REFERENCE_NOISE, t_ref.size)
    frame_ref = pd.DataFrame({"timestamp": _iso(t_ref), "value": v_ref})

    frame_w = _messify(frame_w, rng, swaps=5, duplicates=3)
    frame_aux = _messify(frame_aux, rng, swaps=5, duplicates=3)
    frame_ref = _messify(frame_ref, rng, swaps=2, duplicates=1)

    truth = {
        "seed": seed,
        "stream_ids": {"wearable": WEARABLE_ID, "aux": AUX_ID, "reference": REFERENCE_ID},
        "base_utc_epoch": BASE_UTC_EPOCH,
        "true_lag_seconds": dict(TRUE_LAG_SECONDS),
        "true_dropouts_clock_seconds": [
            {
                "start_epoch": BASE_UTC_EPOCH + lo + TRUE_LAG_SECONDS[WEARABLE_ID],
                "end_epoch": BASE_UTC_EPOCH + hi + TRUE_LAG_SECONDS[WEARABLE_ID],
                "duration_seconds": hi - lo,
            }
            for lo, hi in WEARABLE_DROPOUTS_TRUE
        ],
        "wearable_bias": WEARABLE_BIAS,
        "wearable_drift_per_hour": WEARABLE_DRIFT_PER_HOUR,
        "wearable_noise_sigma": WEARABLE_NOISE,
        "aux_bias": AUX_BIAS,
        "reference_noise_sigma": REFERENCE_NOISE,
        "wearable_nominal_hz": 10.0,
        "aux_nominal_hz": 1.0,
        "reference_period_seconds": 60.0,
    }
    return {WEARABLE_ID: frame_w, AUX_ID: frame_aux, REFERENCE_ID: frame_ref}, truth


def write_benchmark_csvs(out_dir, seed: int = 42) -> tuple[dict[str, Path], dict]:
    """Write the benchmark to ``<out_dir>/<stream_id>.csv``. Returns paths+truth."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames, truth = generate_benchmark(seed)
    paths: dict[str, Path] = {}
    for sid, frame in frames.items():
        p = out / f"{sid}.csv"
        frame.to_csv(p, index=False)
        paths[sid] = p
    return paths, truth
