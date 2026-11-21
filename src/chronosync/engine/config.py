"""Pydantic v2 models: configuration inputs, profiles, metrics, and the audit manifest.

This module must never import `mcp` (engine layer purity).
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

_FREQ_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h)?\s*$", re.IGNORECASE)
_UNIT_SECONDS = {"ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0}


def parse_freq_seconds(value: Union[str, int, float]) -> float:
    """Parse a target-grid frequency such as ``500ms``, ``1s``, ``10s``, ``1m``
    into seconds between grid samples. Plain numbers are taken as seconds."""
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds <= 0:
            raise ValueError("target frequency must be > 0")
        return seconds
    m = _FREQ_RE.match(str(value))
    if not m:
        raise ValueError(
            f"cannot parse frequency {value!r}; expected e.g. '500ms', '1s', '10s', '5m'"
        )
    seconds = float(m.group(1)) * _UNIT_SECONDS[(m.group(2) or "s").lower()]
    if seconds <= 0:
        raise ValueError("target frequency must be > 0")
    return seconds


class StreamSpec(BaseModel):
    """Declares one input stream: where it is and which columns hold time/value."""

    model_config = ConfigDict(extra="forbid")

    stream_id: str
    path: str
    timestamp_col: str = "timestamp"
    value_col: str = "value"
    role: Literal["candidate", "reference"] = "candidate"


class SyncConfig(BaseModel):
    """Harmonization and audit parameters for one run."""

    model_config = ConfigDict(extra="forbid")

    target_freq_seconds: float = Field(1.0, gt=0, description="seconds between target-grid samples")
    max_gap_seconds: float = Field(
        30.0, gt=0, description="gaps longer than this stay NaN and are flagged as dropouts"
    )
    interp_method: Literal["linear", "cubic"] = "linear"
    auto_correct_lag: bool = True
    max_lag_seconds: float = Field(90.0, gt=0, description="clock-lag search radius")
    lag_resolution_seconds: float = Field(0.05, gt=0)
    reference_tolerance_seconds: float = Field(
        5.0, gt=0, description="max asof match distance when pairing against reference points"
    )
    smooth_seconds: float = Field(2.0, ge=0, description="pre-smoothing window for lag estimation")
    drift_alpha: float = Field(0.05, ge=0.0, le=1.0)
    anomaly_top_k_manifest: int = Field(5, ge=1, le=50)

    @classmethod
    def from_target_freq(cls, target_freq: Union[str, float], **overrides: Any) -> "SyncConfig":
        kwargs: dict[str, Any] = {"target_freq_seconds": parse_freq_seconds(target_freq)}
        kwargs.update(overrides)
        return cls(**kwargs)


class DropoutWindow(BaseModel):
    start_epoch: float
    end_epoch: float
    duration_seconds: float


class StreamProfileReport(BaseModel):
    """Profiler output for one stream; lag fields filled by the pipeline."""

    stream_id: str
    path: str
    role: str = "candidate"
    n_rows_raw: int
    n_rows_clean: int
    n_duplicate_timestamps: int = 0
    n_out_of_order_rows: int = 0
    n_invalid_rows: int = 0
    first_utc: Optional[str] = None
    last_utc: Optional[str] = None
    median_hz: Optional[float] = None
    median_dt_seconds: Optional[float] = None
    jitter_iqr_seconds: Optional[float] = None
    dropouts: list[DropoutWindow] = Field(default_factory=list)
    dropout_seconds_total: float = 0.0
    recovered_lag_seconds: Optional[float] = None
    lag_peak_correlation: Optional[float] = None


class ReliabilityMetrics(BaseModel):
    """Agreement scorecard for one candidate stream against the reference."""

    candidate_id: str
    reference_id: str
    n_matched: int
    n_reference_points: int
    coverage_pct: float
    mae: Optional[float] = None
    rmse: Optional[float] = None
    pearson_r: Optional[float] = None
    lins_ccc: Optional[float] = None
    mean_bias: Optional[float] = None
    sd_diff: Optional[float] = None
    loa_lower: Optional[float] = None
    loa_upper: Optional[float] = None
    pct_outside_loa: Optional[float] = None
    drift_per_hour: Optional[float] = None
    drift_stderr_per_hour: Optional[float] = None
    drift_p_value: Optional[float] = None
    drift_significant: bool = False


class AnomalyWindow(BaseModel):
    candidate_id: str
    start_utc: str
    end_utc: str
    duration_seconds: float
    n_points: int
    max_abs_error: float
    mean_abs_error: float


class AuditManifest(BaseModel):
    """Compact (<2,000-token) JSON digest intended for LLM agents."""

    schema_version: str = "1.0"
    run_id: str
    created_utc: str
    config: dict[str, Any] = Field(default_factory=dict)
    overlap_utc: dict[str, Optional[str]] = Field(default_factory=dict)
    streams: list[StreamProfileReport] = Field(default_factory=list)
    metrics: list[ReliabilityMetrics] = Field(default_factory=list)
    top_anomaly_windows: list[AnomalyWindow] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
