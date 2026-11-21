"""Typer CLI: ``chronosync inspect|harmonize|generate-demo|query-anomalies``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import yaml

from .engine import synthetic
from .pipeline import run_pipeline
from .reporting.manifest import read_manifest

app = typer.Typer(
    help="Synchronize, audit, and validate multi-device sensor streams against a reference.",
    no_args_is_help=True,
)


def _print_json(payload) -> None:
    typer.echo(json.dumps(payload, indent=1, ensure_ascii=False))


@app.command()
def inspect(
    streams: list[str] = typer.Argument(..., help="CSV/Parquet stream files"),
    timestamp_col: str = typer.Option("timestamp", "--timestamp-col"),
    value_col: str = typer.Option("value", "--value-col"),
    dropout_threshold: float = typer.Option(30.0, "--dropout-threshold",
                                           help="seconds above which a sample gap is a dropout"),
) -> None:
    """Profile streams: sampling rate, jitter, duplicates, dropout windows."""
    from .engine import profiler

    reports = []
    for path in streams:
        df = profiler.load_stream(path, timestamp_col=timestamp_col, value_col=value_col)
        report = profiler.profile_stream(df, Path(path).stem, path,
                                        role="candidate", dropout_min_seconds=dropout_threshold)
        reports.append(report.model_dump(mode="json"))
    _print_json({"streams": reports})


@app.command("generate-demo")
def generate_demo(
    out: Path = typer.Option(Path("examples/demo_run/streams"), "--out",
                             help="directory for generated stream CSVs"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Write the synthetic 3-stream benchmark (10 Hz wearable, 1 Hz aux, sparse reference)."""
    paths, truth = synthetic.write_benchmark_csvs(out, seed=seed)
    _print_json({
        "written": {sid: str(p) for sid, p in paths.items()},
        "ground_truth": truth,
        "next": (
            f"chronosync harmonize -c {paths[synthetic.WEARABLE_ID]} "
            f"-c {paths[synthetic.AUX_ID]} -r {paths[synthetic.REFERENCE_ID]} "
            f"--output-dir {out.parent / 'runs'}"
        ),
    })


@app.command()
def harmonize(
    candidates: list[str] = typer.Option(..., "--candidate", "-c",
                                         help="candidate stream file (repeatable)"),
    reference: str = typer.Option(..., "--reference", "-r",
                                  help="gold-standard reference stream file"),
    target_freq: str = typer.Option("1s", "--target-freq", help="e.g. 1s, 10s, 5m"),
    max_gap_seconds: float = typer.Option(30.0, "--max-gap-seconds"),
    interp: str = typer.Option("linear", "--interp", help="linear | cubic"),
    auto_correct_lag: bool = typer.Option(True, "--auto-correct-lag/--no-auto-correct-lag"),
    anomaly_threshold: Optional[float] = typer.Option(
        None, "--anomaly-threshold", help="fixed |error| threshold; default = 1.96*SD LoA"),
    output_dir: Path = typer.Option(Path("runs"), "--output-dir"),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
    config: Optional[Path] = typer.Option(
        None, "--config", help="YAML file with SyncConfig overrides"),
) -> None:
    """Full run: lag-correct, gap-guarded harmonize, audit, write dual artifacts."""
    overrides: dict = {}
    if config is not None:
        loaded = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise typer.BadParameter("config YAML must be a mapping of SyncConfig fields")
        overrides.update(loaded)
    overrides.update(
        max_gap_seconds=max_gap_seconds,
        interp_method=interp,
        auto_correct_lag=auto_correct_lag,
    )
    try:
        summary = run_pipeline(
            candidates, reference,
            target_freq=target_freq, anomaly_threshold=anomaly_threshold,
            output_dir=str(output_dir), run_id=run_id, **overrides,
        )
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_json(summary)


@app.command("query-anomalies")
def query_anomalies(
    run_id: str = typer.Argument(...),
    output_dir: Path = typer.Option(Path("runs"), "--output-dir"),
    top_k: int = typer.Option(10, "--top-k"),
) -> None:
    """Print the worst disagreement windows stored in a run manifest."""
    manifest_path = output_dir / run_id / "manifest.json"
    if not manifest_path.exists():
        typer.echo(f"error: no run {run_id!r} at {manifest_path}", err=True)
        raise typer.Exit(code=1)
    model = read_manifest(manifest_path)
    windows = model.top_anomaly_windows[: max(1, top_k)]
    _print_json({
        "run_id": run_id,
        "requested_top_k": top_k,
        "note": "manifest stores the top configured windows (default 5)",
        "windows": [w.model_dump(mode="json") for w in windows],
    })


if __name__ == "__main__":
    app()
