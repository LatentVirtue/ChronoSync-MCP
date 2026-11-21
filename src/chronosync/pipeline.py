"""End-to-end run orchestration shared by the CLI and the MCP server.

Profile -> lag-correct -> gap-guarded harmonize -> pair against reference ->
audit -> dual artifacts (manifest.json + report.html + parquet datasets).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from .engine import alignment, profiler, reliability
from .engine.config import SyncConfig
from .reporting import html_report, manifest as manifest_mod


def _unique_stream_id(stem: str, taken: set[str]) -> str:
    candidate = stem or "stream"
    i = 2
    while candidate in taken:
        candidate = f"{stem}_{i}"
        i += 1
    taken.add(candidate)
    return candidate


def run_pipeline(
    candidate_paths: Sequence[str],
    reference_path: str,
    *,
    target_freq: Any = "1s",
    max_gap_seconds: float = 30.0,
    interp_method: str = "linear",
    auto_correct_lag: bool = True,
    max_lag_seconds: float = 90.0,
    lag_resolution_seconds: float = 0.05,
    reference_tolerance_seconds: float = 5.0,
    anomaly_threshold: Optional[float] = None,
    output_dir: str = "runs",
    run_id: Optional[str] = None,
    write_report: bool = True,
) -> dict:
    """Run one full audit; artifacts land in ``<output_dir>/<run_id>/``.

    Returns ``{"run_id", "run_dir", "artifacts", "manifest"}`` — the manifest is
    the compact JSON-safe digest itself.
    """
    config = SyncConfig.from_target_freq(
        target_freq,
        max_gap_seconds=max_gap_seconds,
        interp_method=interp_method,
        auto_correct_lag=auto_correct_lag,
        max_lag_seconds=max_lag_seconds,
        lag_resolution_seconds=lag_resolution_seconds,
        reference_tolerance_seconds=reference_tolerance_seconds,
    )
    run_id = run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = Path(output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    created_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    ref_path = Path(reference_path)
    cand_paths = [Path(p) for p in candidate_paths]
    if not cand_paths:
        raise ValueError("at least one candidate stream is required")
    if any(p == ref_path for p in cand_paths):
        raise ValueError("the reference stream must not also be listed as a candidate")

    taken: set[str] = set()
    ref_id = _unique_stream_id(ref_path.stem, taken)
    cand_ids: list[str] = []
    paths: dict[str, Path] = {}
    for p in cand_paths:
        cid = _unique_stream_id(p.stem, taken)
        cand_ids.append(cid)
        paths[cid] = p
    paths[ref_id] = ref_path

    raw = {sid: profiler.load_stream(paths[sid]) for sid in paths}
    profiles = {
        sid: profiler.profile_stream(
            raw[sid], sid, str(paths[sid]),
            role="reference" if sid == ref_id else "candidate",
            dropout_min_seconds=config.max_gap_seconds,
        )
        for sid in paths
    }
    cleaned = {sid: profiler.clean_stream(raw[sid]) for sid in paths}
    if len(cleaned[ref_id]) < 2:
        raise ValueError(f"reference stream {ref_id!r} has fewer than 2 usable rows")

    working = {
        sid: (
            cleaned[sid]["t"].to_numpy(dtype="float64"),
            cleaned[sid]["value"].to_numpy(dtype="float64"),
        )
        for sid in paths
    }

    lag_results: dict[str, dict] = {}
    if config.auto_correct_lag:
        ref_t, ref_v = working[ref_id]
        for cid in cand_ids:
            est = alignment.estimate_clock_lag(
                ref_t, ref_v, working[cid][0], working[cid][1],
                max_lag_seconds=config.max_lag_seconds,
                resolution_seconds=config.lag_resolution_seconds,
                max_gap_seconds=config.max_gap_seconds,
                smooth_seconds=config.smooth_seconds,
            )
            lag_results[cid] = est
            if est["lag_seconds"] is not None:
                t, v = working[cid]
                working[cid] = (t - est["lag_seconds"], v)
                profiles[cid].recovered_lag_seconds = est["lag_seconds"]
                profiles[cid].lag_peak_correlation = est["peak_correlation"]

    wide, overlap_start, overlap_end = alignment.harmonize_wide(working, ref_id, config)
    harmonized_path = out_dir / "harmonized.parquet"
    wide.to_parquet(harmonized_path, index=False)

    ref_t, ref_v = working[ref_id]
    paired = pd.DataFrame(
        {
            "t": ref_t,
            "t_utc": [
                datetime.fromtimestamp(float(x), tz=timezone.utc).isoformat(timespec="milliseconds")
                for x in ref_t
            ],
            "reference_value": ref_v,
        }
    )
    for cid in cand_ids:
        vals, drop = alignment.asof_reference_pairs(
            ref_t, working[cid][0], working[cid][1],
            tolerance_seconds=config.reference_tolerance_seconds,
            max_gap_seconds=config.max_gap_seconds,
        )
        paired[cid] = vals
        paired[f"{cid}_is_dropout"] = drop
    paired_path = out_dir / "paired_reference.parquet"
    paired.to_parquet(paired_path, index=False)

    metrics_all = []
    windows_all = []
    for cid in cand_ids:
        m, w = reliability.evaluate_reliability(
            paired["t"], paired["reference_value"], paired[cid],
            paired[f"{cid}_is_dropout"],
            candidate_id=cid, reference_id=ref_id,
            alpha=config.drift_alpha, anomaly_threshold=anomaly_threshold,
        )
        metrics_all.append(m)
        windows_all.extend(w)

    artifact_paths = {
        "harmonized_parquet": str(harmonized_path),
        "paired_reference_parquet": str(paired_path),
        "manifest_json": str(out_dir / "manifest.json"),
    }
    if write_report:
        artifact_paths["report_html"] = str(out_dir / "report.html")

    profile_order = [profiles[ref_id]] + [profiles[c] for c in cand_ids]
    model = manifest_mod.build_manifest(
        run_id=run_id,
        created_utc=created_utc,
        config=config,
        profiles=profile_order,
        metrics=metrics_all,
        anomaly_windows=windows_all,
        artifacts=artifact_paths,
        overlap_utc=(overlap_start, overlap_end),
    )
    manifest_mod.write_manifest(model, out_dir / "manifest.json")

    if write_report:
        html_report.render_report(
            out_dir / "report.html",
            run_id=run_id,
            created_utc=created_utc,
            config=config.model_dump(),
            profiles=profiles,
            lag_results=lag_results,
            wide=wide,
            paired=paired,
            reference_id=ref_id,
            candidate_ids=cand_ids,
            metrics=metrics_all,
            anomaly_windows=[w.model_dump() for w in
                            sorted(windows_all, key=lambda w: w.max_abs_error, reverse=True)[:10]],
        )

    return {
        "run_id": run_id,
        "run_dir": str(out_dir),
        "artifacts": artifact_paths,
        "manifest": model.model_dump(mode="json", exclude_none=True),
    }
