"""FastMCP server exposing ChronoSync tools and a run-manifest resource.

This is the only module in the package allowed to import `mcp`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import pipeline
from .engine import profiler

try:  # mcp SDK 2.x renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ModuleNotFoundError:  # mcp SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server

server = _Server("chronosync")


def _runs_root() -> Path:
    return Path(os.environ.get("CHRONOSYNC_RUNS_ROOT", "runs"))


@server.tool()
def inspect_sensor_streams(stream_paths: list[str], timestamp_col: str = "timestamp") -> dict:
    """Profile sensor stream files (sampling rate, jitter, duplicates, dropouts)."""
    reports = []
    for path in stream_paths:
        df = profiler.load_stream(path, timestamp_col=timestamp_col)
        report = profiler.profile_stream(df, Path(path).stem, path)
        reports.append(report.model_dump(mode="json"))
    return {"streams": reports}


@server.tool()
def harmonize_and_audit_streams(
    candidate_paths: list[str],
    reference_path: str,
    target_freq: str = "1s",
    max_gap_seconds: float = 30.0,
    auto_correct_lag: bool = True,
    output_dir: str = "runs",
) -> dict:
    """Lag-correct, gap-guarded harmonize, and audit candidates against a
    gold-standard reference. Writes harmonized.parquet, paired_reference.parquet,
    manifest.json, and report.html under ``<output_dir>/<run_id>/`` and returns
    the compact audit manifest."""
    return pipeline.run_pipeline(
        candidate_paths,
        reference_path,
        target_freq=target_freq,
        max_gap_seconds=max_gap_seconds,
        auto_correct_lag=auto_correct_lag,
        output_dir=output_dir,
    )


@server.tool()
def query_disagreement_windows(
    run_id: str, top_k: int = 10, output_dir: str = "runs"
) -> dict:
    """Return the worst candidate/reference disagreement windows for a past run."""
    manifest_path = Path(output_dir) / run_id / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"no run {run_id!r} at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    windows = sorted(
        manifest.get("top_anomaly_windows", []),
        key=lambda w: w.get("max_abs_error", 0.0),
        reverse=True,
    )[: max(1, top_k)]
    return {
        "run_id": run_id,
        "requested_top_k": top_k,
        "note": "manifest stores the top configured windows (default 5)",
        "windows": windows,
    }


@server.resource("chronosync://runs/{run_id}/manifest")
def read_run_manifest(run_id: str) -> str:
    """Read the manifest JSON of a previous run from the runs root
    (env CHRONOSYNC_RUNS_ROOT, default ./runs)."""
    path = _runs_root() / run_id / "manifest.json"
    if not path.exists():
        raise ValueError(f"no run {run_id!r} under {_runs_root().resolve()}")
    return path.read_text(encoding="utf-8")


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
