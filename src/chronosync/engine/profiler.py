"""Stream loading and profiling: sampling stats, data-quality defects, dropout inventory.

No `mcp` imports allowed in this package.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DropoutWindow, StreamProfileReport


def _to_epoch_seconds(dt: pd.Series) -> np.ndarray:
    """Timedelta-based conversion is resolution-independent: pandas 2.x infers
    datetime64[us]/[s] from some ISO inputs, where astype(int64)/1e9 is wrong."""
    dt = pd.to_datetime(dt, utc=True, errors="coerce")
    secs = (dt - pd.Timestamp("1970-01-01", tz="utc")) / pd.Timedelta(seconds=1)
    return np.asarray(secs, dtype="float64")


def parse_timestamp_series(series: pd.Series) -> np.ndarray:
    """Return epoch seconds for an ISO-8601 or numeric epoch column.

    Numeric columns are auto-scaled (s / ms / us / ns heuristics). Unparseable
    entries become NaN.
    """
    if len(series) == 0:
        return np.asarray([], dtype=float)
    if pd.api.types.is_datetime64_any_dtype(series):
        return _to_epoch_seconds(series)

    num = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    finite = np.isfinite(num)
    if finite.mean() >= 0.9:
        med = float(np.median(np.abs(num[finite])))
        scale = 1.0
        if med > 1e17:
            scale = 1e-9
        elif med > 1e14:
            scale = 1e-6
        elif med > 1e11:
            scale = 1e-3
        return num * scale

    return _to_epoch_seconds(series.astype(str))


def load_stream(path, timestamp_col: str = "timestamp", value_col: str = "value") -> pd.DataFrame:
    """Read a CSV/Parquet stream and return it with added float columns ``t``
    (epoch seconds) and ``value``. Original columns are preserved."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"stream file not found: {p}")
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    for col, kind in ((timestamp_col, "timestamp"), (value_col, "value")):
        if col not in df.columns:
            raise ValueError(
                f"{p.name}: {kind} column {col!r} not found; columns available: {list(df.columns)}"
            )
    out = df.copy()
    out["t"] = parse_timestamp_series(df[timestamp_col])
    out["value"] = pd.to_numeric(df[value_col], errors="coerce").astype("float64")
    return out


def clean_stream(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with non-finite time/value, sort by time, drop duplicate timestamps."""
    t = df["t"].to_numpy(dtype="float64")
    v = df["value"].to_numpy(dtype="float64")
    keep = np.isfinite(t) & np.isfinite(v)
    d = df.loc[keep].sort_values("t", kind="mergesort")
    return d.drop_duplicates("t", keep="first").reset_index(drop=True)


def epoch_to_utc(epoch: float) -> str:
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat(timespec="milliseconds")


def profile_stream(
    df: pd.DataFrame,
    stream_id: str,
    path: str,
    role: str = "candidate",
    dropout_min_seconds: float = 30.0,
) -> StreamProfileReport:
    """Profile one loaded stream (as returned by :func:`load_stream`).

    Counts data-quality defects in the raw order, then computes sampling-rate
    stats, jitter, and dropout windows: gaps longer than
    ``max(dropout_min_seconds, 5 x median period)`` so a sparse stream's own
    cadence is never mistaken for a dropout.
    """
    t_raw = df["t"].to_numpy(dtype="float64")
    v_raw = df["value"].to_numpy(dtype="float64")
    n_rows_raw = len(t_raw)

    finite_row = np.isfinite(t_raw) & np.isfinite(v_raw)
    n_invalid_rows = int(n_rows_raw - finite_row.sum())
    t_finite = t_raw[finite_row]
    n_duplicates = int(len(t_finite) - np.unique(t_finite).size) if len(t_finite) else 0
    n_out_of_order = int(np.sum(np.diff(t_raw) < 0))

    clean = clean_stream(df)
    t = clean["t"].to_numpy(dtype="float64")
    report = StreamProfileReport(
        stream_id=stream_id,
        path=str(path),
        role=role,
        n_rows_raw=n_rows_raw,
        n_rows_clean=len(t),
        n_duplicate_timestamps=n_duplicates,
        n_out_of_order_rows=n_out_of_order,
        n_invalid_rows=n_invalid_rows,
    )
    if len(t) == 0:
        return report

    report.first_utc = epoch_to_utc(t[0])
    report.last_utc = epoch_to_utc(t[-1])

    if len(t) >= 2:
        dt = np.diff(t)
        pos = dt[dt > 1e-9]
        median_dt = float(np.median(pos)) if len(pos) else 0.0
        if median_dt > 0.0:
            report.median_dt_seconds = median_dt
            report.median_hz = 1.0 / median_dt
            report.jitter_iqr_seconds = float(np.quantile(dt, 0.75) - np.quantile(dt, 0.25))
        # A dropout must be long in absolute terms AND relative to the stream's
        # own cadence: a 60 s period sparse reference has no dropout at a 60 s
        # interval; a 10 Hz wearable has one at 150 s.
        dropout_min = max(dropout_min_seconds, 5.0 * median_dt)
        gaps = np.nonzero(dt > dropout_min)[0]
        for i in gaps:
            start, end = float(t[i]), float(t[i + 1])
            report.dropouts.append(
                DropoutWindow(
                    start_epoch=start, end_epoch=end, duration_seconds=end - start
                )
            )
        report.dropout_seconds_total = float(sum(d.duration_seconds for d in report.dropouts))
    return report
