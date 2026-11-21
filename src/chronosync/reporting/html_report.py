"""Standalone single-file interactive Plotly HTML audit dashboard."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Template
from plotly.subplots import make_subplots
from scipy import stats

import plotly.graph_objects as go

from ..engine.config import ReliabilityMetrics, StreamProfileReport
from ..engine.reliability import rolling_error


def _ts(epoch: float) -> datetime:
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc)


def _fmt(value, nd: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (int,)):
        return str(value)
    return f"{float(value):.{nd}f}"


def _streams_figure(
    wide: pd.DataFrame,
    profiles_by_id: dict[str, StreamProfileReport],
    reference_id: str,
    candidate_ids: list[str],
) -> go.Figure:
    order = [reference_id] + list(candidate_ids)
    n = len(order)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                       subplot_titles=order)
    x = [_ts(t) for t in wide["t"].to_numpy(dtype="float64")]
    for row, sid in enumerate(order, 1):
        y = wide[sid].to_numpy(dtype="float64")
        if sid == reference_id:
            fig.add_trace(
                go.Scatter(x=x, y=y, mode="markers", name=f"{sid} (gold reference samples)",
                           marker=dict(size=5, color="#2563eb")),
                row=row, col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(x=x, y=y, mode="lines", connectgaps=False,
                           name=f"{sid} (harmonized grid)",
                           line=dict(width=1.5, color="#0f766e")),
                row=row, col=1,
            )
        prof = profiles_by_id.get(sid)
        if prof is not None:
            lag = prof.recovered_lag_seconds or 0.0
            for d in prof.dropouts:
                fig.add_vrect(x0=_ts(d.start_epoch - lag), x1=_ts(d.end_epoch - lag),
                              row=row, col=1, fillcolor="rgba(220,38,38,0.18)", line_width=0)
    fig.update_yaxes(title_text="value", title_font_size=11)
    fig.update_xaxes(title_text="time (UTC)", row=n, col=1)
    fig.update_layout(
        height=180 * n + 160,
        margin=dict(l=64, r=24, t=70),
        legend=dict(orientation="h", yanchor="top", y=-0.10 - 0.12 * n, x=0),
        title="Synchronized multi-track streams (red bands = hardware dropouts kept as NaN)",
    )
    return fig


def _lag_figure(candidate_id: str, est: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=est["lag_grid"], y=est["correlation"], mode="lines",
                             name=f"{candidate_id} vs reference", line=dict(color="#2563eb")))
    if est.get("lag_seconds") is not None:
        fig.add_vline(x=est["lag_seconds"], line_dash="dash", line_color="crimson",
                      annotation_text=f"{est['lag_seconds']:+.2f} s",
                      annotation_position="top right")
    fig.update_layout(
        height=320, margin=dict(l=64, r=24, t=60),
        title=f"Cross-correlation clock-lag curve — {candidate_id}",
        xaxis_title="candidate clock offset applied to reference instants (s)",
        yaxis_title="Pearson r",
    )
    return fig


def _ba_figure(candidate_id: str, paired: pd.DataFrame, m: ReliabilityMetrics) -> go.Figure:
    ok = np.isfinite(paired["reference_value"]) & np.isfinite(paired[candidate_id])
    x = paired.loc[ok, "reference_value"].to_numpy(dtype="float64")
    y = paired.loc[ok, candidate_id].to_numpy(dtype="float64") - x
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=f"{candidate_id} − reference",
                             marker=dict(size=6, color="#0f766e", opacity=0.65)))
    hlines = [
        (m.mean_bias, "crimson", "dash", f"bias {m.mean_bias:+.3f}"),
        (m.loa_upper, "#555", "dashdot", f"+1.96 SD {m.loa_upper:+.3f}"),
        (m.loa_lower, "#555", "dashdot", f"−1.96 SD {m.loa_lower:+.3f}"),
    ]
    for value, color, dash, label in hlines:
        if value is not None:
            fig.add_hline(y=value, line_dash=dash, line_color=color,
                         annotation_text=label, annotation_position="top left")
    fig.update_layout(
        height=420, margin=dict(l=64, r=24, t=60),
        title=f"Bland–Altman agreement — {candidate_id}",
        xaxis_title="reference value", yaxis_title="difference (candidate − reference)",
    )
    return fig


def _drift_figure(candidate_id: str, paired: pd.DataFrame) -> go.Figure:
    ok = np.isfinite(paired["reference_value"]) & np.isfinite(paired[candidate_id])
    t = paired.loc[ok, "t"].to_numpy(dtype="float64")
    err = paired.loc[ok, candidate_id].to_numpy(dtype="float64") - paired.loc[ok, "reference_value"].to_numpy(dtype="float64")
    roll = rolling_error(t, err)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[_ts(v) for v in t], y=roll, mode="lines",
                             name="rolling mean error", line=dict(color="#b45309", width=2)))
    if len(t) >= 3 and np.ptp(t) > 0:
        hours = (t - t[0]) / 3600.0
        lr = stats.linregress(hours, err)
        fig.add_trace(go.Scatter(
            x=[_ts(t[0]), _ts(t[-1])],
            y=[lr.intercept, lr.intercept + lr.slope * hours[-1]],
            mode="lines", name=f"OLS drift {lr.slope:+.3f}/h",
            line=dict(color="#6b7280", dash="dash", width=1.5),
        ))
    fig.update_layout(
        height=380, margin=dict(l=64, r=24, t=60),
        title=f"Temporal error drift — {candidate_id}",
        xaxis_title="time (UTC)", yaxis_title="error vs reference",
    )
    return fig


_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ChronoSync audit — {{ run_id }}</title>
<style>
 body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 0;
        background: #f6f7f9; color: #111827; }
 header { background: #111827; color: #f9fafb; padding: 18px 32px; }
 header h1 { margin: 0; font-size: 20px; }
 header .meta { color: #9ca3af; font-size: 13px; margin-top: 4px; }
 main { padding: 24px 32px; max-width: 1200px; margin: 0 auto; }
 section { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
           padding: 16px; margin-bottom: 24px; }
 h2 { font-size: 16px; margin: 0 0 12px; }
 table { border-collapse: collapse; width: 100%; font-size: 13px; }
 th, td { border: 1px solid #e5e7eb; padding: 6px 10px; text-align: right; }
 th:first-child, td:first-child { text-align: left; }
 th { background: #f3f4f6; }
 td.neg { color: #b91c1c; }
 .cfg { font-family: ui-monospace, monospace; font-size: 12px; white-space: pre-wrap; }
</style>
</head>
<body>
<header>
  <h1>ChronoSync audit report</h1>
  <div class="meta">run {{ run_id }} · created {{ created_utc }} · target grid {{ target_freq_s }}s
    · max gap {{ max_gap_s }}s · interpolation {{ interp }} · lag correction {{ lag_mode }}</div>
</header>
<main>
<section>
  <h2>Scorecard</h2>
  <table>
    <tr><th>candidate</th><th>n / ref pts</th><th>coverage %</th><th>MAE</th><th>RMSE</th>
        <th>PEARSON r</th><th>CCC</th><th>bias</th><th>LoA low / high</th>
        <th>% out LoA</th><th>drift/h</th><th>p</th></tr>
    {% for m in metrics %}
    <tr>
      <td>{{ m.candidate_id }}</td>
      <td>{{ m.n_matched }} / {{ m.n_reference_points }}</td>
      <td>{{ '%.1f'|format(m.coverage_pct) }}</td>
      <td>{{ fmt(m.mae) }}</td><td>{{ fmt(m.rmse) }}</td>
      <td>{{ fmt(m.pearson_r, 4) }}</td><td>{{ fmt(m.lins_ccc, 4) }}</td>
      <td class="{{ 'neg' if m.mean_bias is not none and m.mean_bias|abs > 0 }}">{{ fmt(m.mean_bias) }}</td>
      <td>{{ fmt(m.loa_lower) }} / {{ fmt(m.loa_upper) }}</td>
      <td>{{ fmt(m.pct_outside_loa, 1) }}</td>
      <td>{{ fmt(m.drift_per_hour) }}{{ ' *' if m.drift_significant }}</td>
      <td>{{ fmt(m.drift_p_value, 1, 'sci') }}</td>
    </tr>
    {% endfor %}
  </table>
  <p class="cfg">* = calibration drift statistically significant at α={{ alpha }}</p>
</section>

<section>
  <h2>Synchronized streams</h2>
  {{ streams_div }}
</section>

{% for c in candidates %}
<section>
  <h2>{{ c.id }}</h2>
  {{ c.lag_div }}
  {{ c.ba_div }}
  {{ c.drift_div }}
</section>
{% endfor %}

<section>
  <h2>Anomaly windows ({{ windows|length }})</h2>
  <table>
    <tr><th>candidate</th><th>start (UTC)</th><th>end (UTC)</th><th>duration s</th>
        <th>points</th><th>max |err|</th><th>mean |err|</th></tr>
    {% for w in windows %}
    <tr>
      <td>{{ w.candidate_id }}</td><td>{{ w.start_utc }}</td><td>{{ w.end_utc }}</td>
      <td>{{ '%.0f'|format(w.duration_seconds) }}</td><td>{{ w.n_points }}</td>
      <td>{{ '%.3f'|format(w.max_abs_error) }}</td><td>{{ '%.3f'|format(w.mean_abs_error) }}</td>
    </tr>
    {% endfor %}
  </table>
</section>

<section>
  <h2>Run configuration</h2>
  <pre class="cfg">{{ config_json }}</pre>
</section>
</main>
</body>
</html>
"""
)


def _fmt_filter(value, nd=3, mode=None):
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return "—"
    if mode == "sci":
        return f"{v:.1e}"
    return f"{v:.{int(nd)}f}"


def render_report(
    path,
    *,
    run_id: str,
    created_utc: str,
    config: dict,
    profiles: dict[str, StreamProfileReport],
    lag_results: dict[str, dict],
    wide: pd.DataFrame,
    paired: pd.DataFrame,
    reference_id: str,
    candidate_ids: list[str],
    metrics: list[ReliabilityMetrics],
    anomaly_windows: list[dict],
) -> Path:
    streams_fig = _streams_figure(wide, profiles, reference_id, candidate_ids)
    streams_div = streams_fig.to_html(full_html=False, include_plotlyjs=True,
                                      include_mathjax=False, config={"responsive": True})

    candidates = []
    for cid in candidate_ids:
        m = next(mm for mm in metrics if mm.candidate_id == cid)
        blocks = []
        lag_html = "<p>no usable lag correlation found; timestamps used as-is.</p>"
        if cid in lag_results and lag_results[cid].get("lag_grid"):
            lag_fig = _lag_figure(cid, lag_results[cid])
            lag_html = lag_fig.to_html(full_html=False, include_plotlyjs=False,
                                       config={"responsive": True})
        ba_html = _ba_figure(cid, paired, m).to_html(full_html=False, include_plotlyjs=False,
                                                     config={"responsive": True})
        drift_html = _drift_figure(cid, paired).to_html(full_html=False, include_plotlyjs=False,
                                                        config={"responsive": True})
        blocks.extend([lag_html, ba_html, drift_html])
        candidates.append({"id": cid, "lag_div": blocks[0], "ba_div": blocks[1],
                           "drift_div": blocks[2]})

    html = _TEMPLATE.render(
        run_id=run_id,
        created_utc=created_utc,
        target_freq_s=config.get("target_freq_seconds"),
        max_gap_s=config.get("max_gap_seconds"),
        interp=config.get("interp_method"),
        lag_mode="on" if config.get("auto_correct_lag") else "off",
        alpha=config.get("drift_alpha"),
        metrics=[m.model_dump() for m in metrics],
        streams_div=streams_div,
        candidates=candidates,
        windows=anomaly_windows,
        config_json=json.dumps(config, indent=1),
        fmt=_fmt_filter,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
