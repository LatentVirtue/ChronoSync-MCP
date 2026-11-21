# ChronoSync-MCP

Synchronizes asynchronous multi-device sensor streams onto a common timestamp grid, then audits each device against a gold-standard reference and reports reliability scores (Bland-Altman limits of agreement, Lin's concordance correlation, MAE/RMSE, calibration drift, disagreement windows). Ships as an MCP server and a CLI over the same engine.

The motivating case: patient-side wearables (10 Hz, drifting clocks, hardware dropouts), auxiliary clinical sensors, and a research-grade arm reference sampled sparsely (every 60 s). Raw streams share no clock, no sample grid, and no duty cycle.

## Install

Python 3.11+.

```bash
uv venv .venv && source .venv/activate
uv pip install -e ".[dev]"
```

## Quick demo

The benchmark generator writes three synthetic streams with known injected defects: a 10 Hz wearable with jitter, two long dropouts (150 s, 90 s), a +14 s clock lag, bias, and +0.35/h calibration drift; a 1 Hz auxiliary with a +3 s lag; and a sparse 60 s reference.

```bash
chronosync generate-demo --out examples/demo_run/streams
chronosync harmonize \
  -c examples/demo_run/streams/wearable_pat.csv \
  -c examples/demo_run/streams/aux_hri.csv \
  -r examples/demo_run/streams/reference_arms.csv \
  --output-dir examples/demo_run/runs
chronosync query-anomalies <run_id>
```

The run recovers the injected 14.0 s lag to within ±0.5 s and flags the drift as significant. Open `examples/demo_run/runs/<run_id>/report.html` in a browser.

## CLI

```text
chronosync inspect STREAM...            sampling rate, jitter, duplicates, dropout windows
chronosync harmonize -c CAND -r REF     lag-correct, harmonize, audit, write artifacts
chronosync generate-demo                synthetic benchmark with known ground truth
chronosync query-anomalies RUN_ID       worst disagreement windows from a manifest
```

Key `harmonize` flags: `--target-freq` (1s, 10s, 5m...), `--max-gap-seconds` (default 30; interpolation never crosses a longer gap), `--interp linear|cubic`, `--auto-correct-lag/--no-auto-correct-lag`, `--anomaly-threshold`, `--config config.yaml`.

## MCP server

Run with any MCP client (`chronosync-mcp`, stdio transport). Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "chronosync": {
      "command": "path/to/.venv/bin/chronosync-mcp"
    }
  }
}
```

For the resource below, set `CHRONOSYNC_RUNS_ROOT` to the directory holding past runs (default `./runs`).

### Tools

| Tool | Arguments | Returns |
|---|---|---|
| `inspect_sensor_streams` | `stream_paths: list[str]`, `timestamp_col: str` | per-stream profiles: median Hz, jitter IQR, duplicates, dropout windows |
| `harmonize_and_audit_streams` | `candidate_paths: list[str]`, `reference_path: str`, `target_freq: str`, `max_gap_seconds: float`, `auto_correct_lag: bool`, `output_dir: str` | compact audit manifest; writes the full artifacts to disk |
| `query_disagreement_windows` | `run_id: str`, `top_k: int`, `output_dir: str` | worst disagreement windows from a stored manifest |

### Resources

`chronosync://runs/{run_id}/manifest` — the run manifest JSON from a previous invocation.

## Outputs

Each run writes to `runs/<run_id>/` (override with `--output-dir`):

- `harmonized.parquet` — long-format multi-track dataset: one common time grid, one value column per stream, plus boolean `*_is_dropout` columns. Points inside hardware dropouts are `NaN` and flagged; they are never bridged by interpolation.
- `paired_reference.parquet` — every reference timestamp plus the nearest candidate sample within `reference_tolerance_seconds` (default 5 s), nearest-sample semantics, no interpolation between candidate samples.
- `manifest.json` — audit digest under 2,000 tokens: profiles, recovered lags, all reliability metrics, top-5 anomaly windows, config used. This is what the MCP tools return.
- `report.html` — standalone interactive Plotly dashboard: multi-track overlay of synchronized streams, Bland-Altman plots, cross-correlation lag curves, rolling error drift, scorecard table.

## Design notes

- **Gap guards are the point.** A dropout is missing hardware, not missing data. `resample_gap_guarded` interpolates freely up to `max_gap_seconds`, then emits `NaN` + mask. `evaluate_reliability` excludes dropout-flagged pairs from all statistics.
- **Clock lag** is recovered by cross-correlating a candidate against the reference across offsets on a reference-aligned grid (`estimate_clock_lag`); the returned curve is embedded in the report.
- **Agreement, not just correlation.** Two devices can correlate at r=0.99 and disagree by a clinically useless margin; Bland-Altman intervals and CCC are reported instead of trusting r.
- The `engine/` package never imports `mcp`; CLI, tests, and plain Python use the same entry point as the server (`pipeline.run_pipeline`).
- Timestamps parse as ISO-8601 or epoch (auto-scaled s/ms/µs/ns); values coerce to float64.

## Tests

```bash
.venv/Scripts/python.exe -m pytest -q        # Windows
.venv/bin/python -m pytest -q                # POSIX
```

The suites check lag recovery against the injected 14 s ground truth (±0.5 s), dropout handling (NaN + flagged, never bridged), Bland-Altman/drift math against synthetic streams with known bias and drift, and the MCP tool surface end-to-end including artifact contents.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
