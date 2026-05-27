"""
reporting.py — HTML Report Generator
======================================
Generates a self-contained, styled HTML report from experiment results.
No external dependencies beyond what's already in requirements.txt.

Usage
-----
From Python:
    from src.reporting import generate_html_report
    generate_html_report(config, results, data, output_path="report.html")

From CLI:
    python -m src.experiment --data data/sample_experiment.csv \\
        --metric revenue --report reports/my_experiment.html
"""

from __future__ import annotations

import base64
import io
import os
from datetime import datetime, timezone
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_html_report(
    config,
    results,
    data: pd.DataFrame,
    output_path: str = "experiment_report.html",
    variant_col: str = "variant",
) -> str:
    """Generate a self-contained HTML report and write it to *output_path*.

    Parameters
    ----------
    config : ExperimentConfig
        The experiment configuration.
    results : ExperimentResults
        The results object from Experiment.run().
    data : pd.DataFrame
        The raw experiment data (used for plots).
    output_path : str
        File path to write the HTML report.
    variant_col : str
        Column identifying control vs treatment.

    Returns
    -------
    str
        The generated HTML string.
    """
    logger.info("Generating HTML report", extra={"output_path": output_path})

    generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build chart images (embedded as base64 so the file is self-contained)
    dist_chart   = _distribution_chart(data, variant_col, config.metric)
    power_chart  = _power_curve_chart(results, config)
    guard_chart  = _guardrail_chart(data, variant_col, config.guardrail_metrics) if config.guardrail_metrics else ""

    html = _render_html(config, results, generated_at, dist_chart, power_chart, guard_chart)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    logger.info("HTML report written", extra={"output_path": output_path, "size_kb": round(len(html) / 1024, 1)})
    return html


# ---------------------------------------------------------------------------
# Chart helpers (return base64-encoded PNG strings)
# ---------------------------------------------------------------------------

def _b64_png(fig) -> str:
    """Convert a matplotlib figure to a base64 PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _distribution_chart(data: pd.DataFrame, variant_col: str, metric: str) -> str:
    """Overlapping distribution of metric for control vs treatment."""
    ctrl = data[data[variant_col] == "control"][metric].dropna()
    trt  = data[data[variant_col] != "control"][metric].dropna()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ctrl, bins=40, alpha=0.55, color="#4C8BF5", label=f"Control (n={len(ctrl):,})", density=True)
    ax.hist(trt,  bins=40, alpha=0.55, color="#F5844C", label=f"Treatment (n={len(trt):,})", density=True)
    ax.axvline(ctrl.mean(), color="#4C8BF5", linestyle="--", linewidth=1.5, label=f"Control mean = {ctrl.mean():.3f}")
    ax.axvline(trt.mean(),  color="#F5844C", linestyle="--", linewidth=1.5, label=f"Treatment mean = {trt.mean():.3f}")
    ax.set_xlabel(metric, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Metric Distribution: {metric}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _b64_png(fig)


def _power_curve_chart(results, config) -> str:
    """Power curve showing the MDE and achieved power region."""
    from src.stats.power import power_curve, minimum_detectable_effect

    pooled_std = results.mde / 2.8  # back-calculate approx std from MDE
    if pooled_std <= 0:
        pooled_std = 1.0

    n     = min(results.n_control, results.n_treatment)
    deltas = np.linspace(0, results.mde * 3, 200)
    powers = power_curve(deltas.tolist(), n=n, std=pooled_std, alpha=config.alpha)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(deltas, powers, color="#4C8BF5", linewidth=2.5)
    ax.axhline(0.80, color="grey", linestyle="--", linewidth=1, label="80% power")
    ax.axvline(results.mde, color="#E74C3C", linestyle="--", linewidth=1.5,
               label=f"MDE = {results.mde:.3f}")
    ax.axvline(abs(results.lift), color="#2ECC71", linestyle="-", linewidth=1.5,
               label=f"Observed lift = {results.lift:+.3f}")
    ax.fill_between(deltas, powers, where=[d >= results.mde for d in deltas],
                    alpha=0.12, color="#2ECC71", label="Detectable region")
    ax.set_xlabel("Effect Size (absolute)", fontsize=11)
    ax.set_ylabel("Statistical Power", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Power Curve", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _b64_png(fig)


def _guardrail_chart(data: pd.DataFrame, variant_col: str, guardrail_metrics: list) -> str:
    """Bar chart comparing control vs treatment means for guardrail metrics."""
    if not guardrail_metrics:
        return ""

    ctrl = data[data[variant_col] == "control"]
    trt  = data[data[variant_col] != "control"]

    ctrl_means = [ctrl[m].mean() for m in guardrail_metrics]
    trt_means  = [trt[m].mean()  for m in guardrail_metrics]

    x     = np.arange(len(guardrail_metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(guardrail_metrics) * 2), 4))
    ax.bar(x - width / 2, ctrl_means, width, label="Control",   color="#4C8BF5", alpha=0.8)
    ax.bar(x + width / 2, trt_means,  width, label="Treatment", color="#F5844C", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(guardrail_metrics, fontsize=10)
    ax.set_ylabel("Mean Value", fontsize=11)
    ax.set_title("Guardrail Metrics: Control vs Treatment", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _b64_png(fig)


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

def _verdict_badge(results, config) -> str:
    guardrails_ok = all(results.guardrail_status.values()) if results.guardrail_status else True
    if results.significant and guardrails_ok and not results.novelty_detected:
        return '<span class="badge badge-ship">🚀 SHIP IT</span>'
    elif results.significant and not guardrails_ok:
        return '<span class="badge badge-warn">🔧 FIX GUARDRAILS</span>'
    elif results.novelty_detected:
        return '<span class="badge badge-warn">⚠️ NOVELTY EFFECT</span>'
    else:
        return '<span class="badge badge-no">⏳ NOT SIGNIFICANT</span>'


def _render_html(config, results, generated_at: str, dist_chart: str, power_chart: str, guard_chart: str) -> str:
    guardrail_rows = ""
    for metric, passed in results.guardrail_status.items():
        icon   = "✅" if passed else "🚨"
        status = "PASS" if passed else "FAIL"
        color  = "#27AE60" if passed else "#E74C3C"
        guardrail_rows += f"""
        <tr>
          <td>{metric}</td>
          <td style="color:{color}; font-weight:600">{icon} {status}</td>
        </tr>"""

    guard_section = ""
    if results.guardrail_status:
        guard_section = f"""
        <div class="section">
          <h2>Guardrail Metrics</h2>
          <table>
            <thead><tr><th>Metric</th><th>Status</th></tr></thead>
            <tbody>{guardrail_rows}</tbody>
          </table>
          {"<img src='" + guard_chart + "' style='max-width:100%;margin-top:20px'/>" if guard_chart else ""}
        </div>"""

    novelty_section = ""
    if results.novelty_detected:
        novelty_section = """
        <div class="section alert-warn">
          <h2>⚠️ Novelty Effect Detected</h2>
          <p>The treatment effect appears stronger in the early cohort than the late cohort.
          This may indicate users are responding to novelty rather than genuine value.
          Consider extending the experiment and re-evaluating with late-cohort data only.</p>
        </div>"""

    sig_color = "#27AE60" if results.significant else "#E74C3C"
    sig_label = "YES" if results.significant else "NO"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Experiment Report — {config.name}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #F7F8FA;
      color: #1A1A2E;
      line-height: 1.6;
    }}
    header {{
      background: linear-gradient(135deg, #1A1A2E 0%, #16213E 60%, #0F3460 100%);
      color: white;
      padding: 40px 48px 32px;
    }}
    header h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 6px; }}
    header p  {{ opacity: 0.65; font-size: 13px; }}
    .badge {{
      display: inline-block;
      padding: 6px 18px;
      border-radius: 20px;
      font-size: 14px;
      font-weight: 700;
      margin-top: 16px;
      letter-spacing: 0.5px;
    }}
    .badge-ship {{ background: #27AE60; color: white; }}
    .badge-warn {{ background: #F39C12; color: white; }}
    .badge-no   {{ background: #E74C3C; color: white; }}
    .container  {{ max-width: 960px; margin: 32px auto; padding: 0 24px; }}
    .section    {{
      background: white;
      border-radius: 12px;
      padding: 28px 32px;
      margin-bottom: 24px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }}
    .section h2 {{
      font-size: 17px;
      font-weight: 600;
      margin-bottom: 18px;
      padding-bottom: 10px;
      border-bottom: 2px solid #F0F1F3;
      color: #16213E;
    }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
    }}
    .metric-card {{
      background: #F7F8FA;
      border-radius: 10px;
      padding: 16px 20px;
      border-left: 4px solid #4C8BF5;
    }}
    .metric-card .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: #888; }}
    .metric-card .value {{ font-size: 22px; font-weight: 700; color: #1A1A2E; margin-top: 4px; }}
    .metric-card .value.positive {{ color: #27AE60; }}
    .metric-card .value.negative {{ color: #E74C3C; }}
    .metric-card .value.sig {{ color: {sig_color}; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    thead th {{
      background: #F7F8FA;
      padding: 10px 14px;
      text-align: left;
      font-weight: 600;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: #888;
    }}
    tbody td {{ padding: 10px 14px; border-bottom: 1px solid #F0F1F3; }}
    tbody tr:last-child td {{ border-bottom: none; }}
    .alert-warn {{
      border-left: 4px solid #F39C12;
      background: #FFFBF2;
    }}
    .alert-warn h2 {{ color: #E67E22; border-bottom-color: #FDEBD0; }}
    .alert-warn p {{ color: #7D6608; font-size: 14px; }}
    img {{ border-radius: 8px; }}
    footer {{
      text-align: center;
      font-size: 12px;
      color: #AAA;
      padding: 24px;
      margin-top: 8px;
    }}
  </style>
</head>
<body>

<header>
  <h1>{config.name}</h1>
  <p>Generated {generated_at}</p>
  {_verdict_badge(results, config)}
</header>

<div class="container">

  <!-- Key Metrics -->
  <div class="section">
    <h2>Primary Metric — {config.metric}</h2>
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="label">Control Mean</div>
        <div class="value">{results.mean_control:.4f}</div>
      </div>
      <div class="metric-card">
        <div class="label">Treatment Mean</div>
        <div class="value">{results.mean_treatment:.4f}</div>
      </div>
      <div class="metric-card">
        <div class="label">Absolute Lift</div>
        <div class="value {'positive' if results.lift >= 0 else 'negative'}">{results.lift:+.4f}</div>
      </div>
      <div class="metric-card">
        <div class="label">Relative Lift</div>
        <div class="value {'positive' if results.lift >= 0 else 'negative'}">{results.relative_lift:+.2%}</div>
      </div>
      <div class="metric-card">
        <div class="label">p-value</div>
        <div class="value">{results.p_value:.6f}</div>
      </div>
      <div class="metric-card">
        <div class="label">Significant (α={config.alpha})</div>
        <div class="value sig">{sig_label}</div>
      </div>
      <div class="metric-card">
        <div class="label">MDE</div>
        <div class="value">{results.mde:.4f}</div>
      </div>
    </div>
  </div>

  <!-- Experiment Config -->
  <div class="section">
    <h2>Experiment Configuration</h2>
    <table>
      <tbody>
        <tr><td>N (Control)</td><td><strong>{results.n_control:,}</strong></td></tr>
        <tr><td>N (Treatment)</td><td><strong>{results.n_treatment:,}</strong></td></tr>
        <tr><td>Alpha (α)</td><td><strong>{config.alpha}</strong></td></tr>
        <tr><td>Target Power</td><td><strong>{config.power}</strong></td></tr>
        <tr><td>CUPED Applied</td><td><strong>{"Yes" if results.cuped_applied else "No"}</strong></td></tr>
        <tr><td>Test Method</td><td><strong>{"mSPRT (Sequential)" if config.use_sequential else "Welch t-test (Fixed Horizon)"}</strong></td></tr>
        <tr><td>Run Duration</td><td><strong>{results.duration_seconds}s</strong></td></tr>
      </tbody>
    </table>
  </div>

  <!-- Distribution Chart -->
  <div class="section">
    <h2>Metric Distribution</h2>
    <img src="{dist_chart}" style="max-width:100%"/>
  </div>

  <!-- Power Curve -->
  <div class="section">
    <h2>Power Curve</h2>
    <img src="{power_chart}" style="max-width:100%"/>
  </div>

  {guard_section}
  {novelty_section}

</div>

<footer>A/B Testing Framework · {generated_at}</footer>

</body>
</html>"""
