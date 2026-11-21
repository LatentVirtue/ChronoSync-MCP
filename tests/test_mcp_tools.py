"""MCP surface: tool registration, engine purity, end-to-end audit on the
synthetic benchmark, manifest budget, dual artifacts, and resource reads."""

import asyncio
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronosync import mcp_server
from chronosync.engine import synthetic

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_engine_layer_contains_no_mcp_imports():
    offenders = []
    for path in (REPO_ROOT / "src" / "chronosync" / "engine").glob("*.py"):
        src = path.read_text(encoding="utf-8")
        if re.search(r"^\s*(import mcp|from mcp(\.|\s))", src, re.MULTILINE):
            offenders.append(path.name)
    assert not offenders


def test_server_registers_the_three_tools_and_the_manifest_resource():
    tools = asyncio.run(mcp_server.server.list_tools())
    names = {t.name for t in tools}
    assert names >= {
        "inspect_sensor_streams",
        "harmonize_and_audit_streams",
        "query_disagreement_windows",
    }
    listed = []
    for method in ("list_resource_templates", "list_resources"):
        fn = getattr(mcp_server.server, method, None)
        if fn is not None:
            listed.extend(asyncio.run(fn()))
    uris = [
        str(getattr(r, "uri_template", None) or getattr(r, "uri", None)
            or getattr(r, "name", ""))
        for r in listed
    ]
    assert any("runs/{run_id}/manifest" in u for u in uris), uris


@pytest.fixture(scope="module")
def demo_streams(tmp_path_factory):
    out = tmp_path_factory.mktemp("streams")
    paths, truth = synthetic.write_benchmark_csvs(out, seed=5)
    return paths, truth


def test_inspect_profiles_synthetic_streams(demo_streams):
    paths, _ = demo_streams
    result = mcp_server.inspect_sensor_streams([str(p) for p in paths.values()])
    by = {s["stream_id"]: s for s in result["streams"]}

    w = by["wearable_pat"]
    assert w["n_rows_raw"] > 60_000
    assert w["median_hz"] == pytest.approx(10.0, abs=1.0)
    assert w["n_duplicate_timestamps"] >= 3
    assert w["n_out_of_order_rows"] >= 3
    assert len(w["dropouts"]) == 2
    durations = sorted(d["duration_seconds"] for d in w["dropouts"])
    assert durations == [pytest.approx(90.0, abs=1.5), pytest.approx(150.0, abs=1.5)]

    ref = by["reference_arms"]
    assert ref["median_hz"] == pytest.approx(1.0 / 60.0, abs=0.002)
    assert ref["n_duplicate_timestamps"] >= 1  # the generator injects one retransmission
    assert ref["dropouts"] == []  # 60 s cadence is not a dropout


def test_full_audit_recovers_every_injected_defect(tmp_path, demo_streams):
    paths, truth = demo_streams
    runs = tmp_path / "runs"
    res = mcp_server.harmonize_and_audit_streams(
        [str(paths["wearable_pat"]), str(paths["aux_hri"])],
        str(paths["reference_arms"]),
        target_freq="1s",
        max_gap_seconds=30.0,
        auto_correct_lag=True,
        output_dir=str(runs),
    )
    run_dir = runs / res["run_id"]

    for name in ("harmonized.parquet", "paired_reference.parquet", "manifest.json",
                 "report.html"):
        assert (run_dir / name).exists(), name

    assert len(json.dumps(res["manifest"])) < 8_000  # < 2,000-token budget

    streams = {s["stream_id"]: s for s in res["manifest"]["streams"]}
    assert streams["wearable_pat"]["recovered_lag_seconds"] == pytest.approx(14.0, abs=0.5)
    # 1 Hz auxiliary: broad correlation peak bounds precision to ~1-2 s (see
    # test_clock_lag_recovery); spec's +/-0.5 s requirement is for the wearable.
    assert streams["aux_hri"]["recovered_lag_seconds"] == pytest.approx(3.0, abs=2.0)

    metrics = {m["candidate_id"]: m for m in res["manifest"]["metrics"]}
    wear = metrics["wearable_pat"]
    assert wear["n_matched"] > 100
    assert wear["lins_ccc"] > 0.9
    assert wear["drift_significant"] is True
    assert wear["drift_per_hour"] == pytest.approx(0.35, abs=0.12)
    # base bias 0.75 plus mean drift over ~2 h of matched points
    assert 0.9 < wear["mean_bias"] < 1.4

    # harmonized artifact: 150 s dropout must be NaN + flagged in corrected time
    wide = pd.read_parquet(run_dir / "harmonized.parquet")
    base = truth["base_utc_epoch"]
    inside = wide[(wide["t"] > base + 2010) & (wide["t"] < base + 2140)]
    assert len(inside) > 100
    assert np.all(np.isnan(inside["wearable_pat"].to_numpy()))
    assert inside["wearable_pat_is_dropout"].all()
    outside = wide[(wide["t"] > base + 1900) & (wide["t"] < base + 1990)]
    assert np.all(np.isfinite(outside["wearable_pat"].to_numpy()))
    assert not outside["wearable_pat_is_dropout"].any()

    paired = pd.read_parquet(run_dir / "paired_reference.parquet")
    assert {"t_utc", "reference_value", "wearable_pat", "wearable_pat_is_dropout"} <= set(
        paired.columns
    )

    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "plotly" in html.lower()
    assert "Bland" in html
    assert len(html) > 50_000


def test_query_disagreement_windows_and_manifest_resource(tmp_path, demo_streams, monkeypatch):
    paths, _ = demo_streams
    runs = tmp_path / "runs"
    res = mcp_server.harmonize_and_audit_streams(
        [str(paths["wearable_pat"])],
        str(paths["reference_arms"]),
        target_freq="5s",
        max_gap_seconds=60.0,
        auto_correct_lag=True,
        output_dir=str(runs),
    )

    q = mcp_server.query_disagreement_windows(res["run_id"], top_k=3, output_dir=str(runs))
    assert q["run_id"] == res["run_id"]
    assert 0 < len(q["windows"]) <= 3
    errs = [w["max_abs_error"] for w in q["windows"]]
    assert errs == sorted(errs, reverse=True)

    monkeypatch.setenv("CHRONOSYNC_RUNS_ROOT", str(runs))
    text = mcp_server.read_run_manifest(res["run_id"])
    manifest = json.loads(text)
    assert manifest["run_id"] == res["run_id"]

    with pytest.raises(ValueError):
        mcp_server.read_run_manifest("definitely-not-a-run")
    with pytest.raises(ValueError):
        mcp_server.query_disagreement_windows("definitely-not-a-run", output_dir=str(runs))


def test_inspect_accepts_numeric_epoch_columns(tmp_path):
    t = 1_766_000_000 + np.arange(0.0, 30.0, 0.5)
    df = pd.DataFrame({"timestamp": t, "value": np.sin(t / 10.0)})
    path = tmp_path / "epoch_stream.csv"
    df.to_csv(path, index=False)
    result = mcp_server.inspect_sensor_streams([str(path)])
    s = result["streams"][0]
    assert s["n_rows_raw"] == t.size
    assert s["median_hz"] == pytest.approx(2.0, abs=0.1)
