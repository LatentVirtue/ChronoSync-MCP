"""Build, write, and read the compact (<2,000-token) JSON audit manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..engine.config import (
    AnomalyWindow,
    AuditManifest,
    ReliabilityMetrics,
    StreamProfileReport,
    SyncConfig,
)
from ..engine.profiler import epoch_to_utc


def _r(value: Optional[float], ndigits: int) -> Optional[float]:
    return None if value is None else round(float(value), ndigits)


def _round_profile(p: StreamProfileReport) -> StreamProfileReport:
    q = p.model_copy(deep=True)
    q.median_hz = _r(q.median_hz, 3)
    q.median_dt_seconds = _r(q.median_dt_seconds, 5)
    q.jitter_iqr_seconds = _r(q.jitter_iqr_seconds, 5)
    q.recovered_lag_seconds = _r(q.recovered_lag_seconds, 3)
    q.lag_peak_correlation = _r(q.lag_peak_correlation, 4)
    q.dropout_seconds_total = _r(q.dropout_seconds_total, 1)
    for d in q.dropouts:
        d.start_epoch = _r(d.start_epoch, 1)
        d.end_epoch = _r(d.end_epoch, 1)
        d.duration_seconds = _r(d.duration_seconds, 1)
    return q


def build_manifest(
    *,
    run_id: str,
    created_utc: str,
    config: SyncConfig,
    profiles: list[StreamProfileReport],
    metrics: list[ReliabilityMetrics],
    anomaly_windows: list[AnomalyWindow],
    artifacts: dict[str, str],
    overlap_utc: Optional[tuple[Optional[float], Optional[float]]] = None,
) -> AuditManifest:
    top = sorted(anomaly_windows, key=lambda w: w.max_abs_error, reverse=True)[
        : config.anomaly_top_k_manifest
    ]
    overlap = {}
    if overlap_utc is not None:
        overlap = {
            "start_utc": epoch_to_utc(overlap_utc[0]) if overlap_utc[0] is not None else None,
            "end_utc": epoch_to_utc(overlap_utc[1]) if overlap_utc[1] is not None else None,
        }
    return AuditManifest(
        run_id=run_id,
        created_utc=created_utc,
        config=config.model_dump(),
        overlap_utc=overlap,
        streams=[_round_profile(p) for p in profiles],
        metrics=metrics,
        top_anomaly_windows=top,
        artifacts={k: str(v) for k, v in artifacts.items()},
    )


def manifest_to_json(model: AuditManifest) -> str:
    return json.dumps(
        model.model_dump(mode="json", exclude_none=True), indent=1, ensure_ascii=False
    )


def write_manifest(model: AuditManifest, path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(manifest_to_json(model), encoding="utf-8")
    return p


def read_manifest(path) -> AuditManifest:
    return AuditManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
