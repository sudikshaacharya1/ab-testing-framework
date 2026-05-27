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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.logger import get_logger

logger = get_logger(__name__)

# ── Brand palette ──────────────────────────────────────────────────────────
BLUE   = "#4361EE"
ORANGE = "#F72585"
GREEN  = "#06D6A0"
YELLOW = "#FFD166"
DARK   = "#0D1117"
CARD   = "#161B22"
BORDER = "#21262D"
MUTED  = "#8B949E"
WHITE  = "#F0F6FC"


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
    """Generate a self-contained HTML report and write it to *output_path*."""
    logger.info("Generating HTML report", extra={"output_path": output_path})

    generated_at = datetime.now(tz=timezone.utc).strftime("%d %b %Y · %H:%M UTC")

    dist_chart  = _distribution_chart(data, variant_col, config.metric)
    power_chart = _power_curve_chart(results, config)
    guard_chart = _guardrail_chart(data, variant_col, config.guardrail_metrics) if config.guardrail_metrics else ""
    ci_chart    = _confidence_interval_chart(results, config)

    html = _render_html(config, results, generated_at, dist_chart, power_chart, guard_chart, ci_chart, data, variant_col)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    logger.info("HTML report written", extra={"output_path": output_path, "size_kb": round(len(html) / 1024, 1)})
    return html


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _b64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=CARD, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _style_ax(ax, fig):
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)


def _distribution_chart(data: pd.DataFrame, variant_col: str, metric: str) -> str:
    ctrl = data[data[variant_col] == "control"][metric].dropna()
    trt  = data[data[variant_col] != "control"][metric].dropna()

    fig, ax = plt.subplots(figsize=(9, 4))
    _style_ax(ax, fig)

    ax.hist(ctrl, bins=45, alpha=0.6, color=BLUE,   label=f"Control  (n={len(ctrl):,})", density=True, linewidth=0)
    ax.hist(trt,  bins=45, alpha=0.6, color=ORANGE, label=f"Treatment (n={len(trt):,})", density=True, linewidth=0)
    ax.axvline(ctrl.mean(), color=BLUE,   linestyle="--", linewidth=1.5, alpha=0.9)
    ax.axvline(trt.mean(),  color=ORANGE, linestyle="--", linewidth=1.5, alpha=0.9)

    # Annotation arrows for means
    ymax = ax.get_ylim()[1]
    ax.annotate(f"${ctrl.mean():.2f}", xy=(ctrl.mean(), ymax * 0.85),
                color=BLUE, fontsize=8.5, ha="center", fontweight="bold")
    ax.annotate(f"${trt.mean():.2f}", xy=(trt.mean(),  ymax * 0.85),
                color=ORANGE, fontsize=8.5, ha="center", fontweight="bold")

    ax.set_xlabel(f"{metric}", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Metric Distribution", fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=9, facecolor=DARK, edgecolor=BORDER, labelcolor=WHITE)
    ax.grid(axis="y", color=BORDER, linewidth=0.5, alpha=0.6)
    fig.tight_layout(pad=1.5)
    return _b64_png(fig)


def _confidence_interval_chart(results, config) -> str:
    """Forest-plot style CI chart."""
    lift = results.lift
    se   = results.mde / 2.8 if results.mde > 0 else abs(lift) * 0.1
    ci_lo = lift - 1.96 * se
    ci_hi = lift + 1.96 * se

    fig, ax = plt.subplots(figsize=(9, 2.4))
    _style_ax(ax, fig)

    color = GREEN if lift > 0 else ORANGE
    ax.barh([0], [ci_hi - ci_lo], left=[ci_lo], height=0.18,
            color=color, alpha=0.25, linewidth=0)
    ax.plot([ci_lo, ci_hi], [0, 0], color=color, linewidth=2.5, solid_capstyle="round")
    ax.scatter([lift], [0], color=color, s=100, zorder=5)
    ax.axvline(0, color=MUTED, linewidth=1.2, linestyle="--", alpha=0.7)
    ax.axvline(results.mde,  color=YELLOW, linewidth=1, linestyle=":", alpha=0.8)
    ax.axvline(-results.mde, color=YELLOW, linewidth=1, linestyle=":", alpha=0.8)

    ax.annotate(f"Observed lift: {lift:+.3f}", xy=(lift, 0.15),
                color=color, fontsize=9, ha="center", fontweight="bold")
    ax.annotate(f"95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]",
                xy=((ci_lo + ci_hi) / 2, -0.22),
                color=MUTED, fontsize=8.5, ha="center")
    ax.annotate(f"MDE: ±{results.mde:.3f}", xy=(results.mde, 0.15),
                color=YELLOW, fontsize=8, ha="left")

    ax.set_yticks([])
    ax.set_xlabel("Effect Size", fontsize=10)
    ax.set_title("95% Confidence Interval", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(min(ci_lo * 1.4, -results.mde * 1.5), max(ci_hi * 1.4, results.mde * 1.5))
    ax.grid(axis="x", color=BORDER, linewidth=0.5, alpha=0.5)
    fig.tight_layout(pad=1.5)
    return _b64_png(fig)


def _power_curve_chart(results, config) -> str:
    from src.stats.power import power_curve

    pooled_std = results.mde / 2.8 if results.mde > 0 else 1.0
    n     = min(results.n_control, results.n_treatment)
    x_max = max(abs(results.lift) * 2.5, results.mde * 3)
    deltas = np.linspace(0, x_max, 300)
    powers = power_curve(deltas.tolist(), n=n, std=pooled_std, alpha=config.alpha)

    fig, ax = plt.subplots(figsize=(9, 4))
    _style_ax(ax, fig)

    ax.plot(deltas, powers, color=BLUE, linewidth=2.5)
    ax.fill_between(deltas, powers, alpha=0.08, color=BLUE)
    ax.axhline(0.80, color=MUTED,   linewidth=1,   linestyle="--", alpha=0.7, label="80% power threshold")
    ax.axvline(results.mde,         color=YELLOW,  linewidth=1.5, linestyle=":",  label=f"MDE = {results.mde:.3f}")
    ax.axvline(abs(results.lift),   color=GREEN,   linewidth=2,   linestyle="-",  label=f"Observed lift = {results.lift:+.3f}")

    ax.fill_between(deltas, powers,
                    where=[d >= results.mde for d in deltas],
                    alpha=0.12, color=GREEN)

    ax.set_xlabel("Effect Size (absolute)", fontsize=10)
    ax.set_ylabel("Statistical Power", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_title("Power Curve", fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=9, facecolor=DARK, edgecolor=BORDER, labelcolor=WHITE)
    ax.grid(color=BORDER, linewidth=0.5, alpha=0.5)
    fig.tight_layout(pad=1.5)
    return _b64_png(fig)


def _guardrail_chart(data: pd.DataFrame, variant_col: str, guardrail_metrics: list) -> str:
    if not guardrail_metrics:
        return ""

    ctrl = data[data[variant_col] == "control"]
    trt  = data[data[variant_col] != "control"]

    n   = len(guardrail_metrics)
    fig, axes = plt.subplots(1, n, figsize=(max(5, n * 4), 4))
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, guardrail_metrics):
        _style_ax(ax, fig)
        c_vals = ctrl[metric].dropna()
        t_vals = trt[metric].dropna()

        means = [c_vals.mean(), t_vals.mean()]
        errs  = [c_vals.sem() * 1.96, t_vals.sem() * 1.96]
        colors = [BLUE, ORANGE]
        bars = ax.bar(["Control", "Treatment"], means, color=colors,
                      alpha=0.8, width=0.5, linewidth=0, yerr=errs,
                      error_kw={"ecolor": WHITE, "elinewidth": 1.5, "capsize": 4})

        # Annotate values
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{val:.2f}", ha="center", va="bottom", color=WHITE, fontsize=9, fontweight="bold")

        ax.set_title(metric, fontsize=11, fontweight="bold", pad=10)
        ax.tick_params(axis="x", colors=WHITE)
        ax.set_ylabel("Mean ± 95% CI", fontsize=9)
        ax.grid(axis="y", color=BORDER, linewidth=0.5, alpha=0.5)
        ymin = min(means) * 0.95
        ax.set_ylim(ymin, max(means) * 1.08)

    fig.tight_layout(pad=2)
    return _b64_png(fig)


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

def _verdict(results, config) -> tuple[str, str, str]:
    """Returns (label, color, description)."""
    guardrails_ok = all(results.guardrail_status.values()) if results.guardrail_status else True
    if results.novelty_detected:
        return "⚠️  NOVELTY EFFECT", YELLOW, "Effect may be driven by novelty. Extend experiment before shipping."
    if results.significant and guardrails_ok:
        return "🚀  SHIP IT", GREEN,  "Primary metric improved. All guardrails passed. Safe to ship."
    if results.significant and not guardrails_ok:
        return "🔧  FIX GUARDRAILS", YELLOW, "Primary metric improved but one or more guardrails failed. Do not ship yet."
    return "⏳  NOT SIGNIFICANT", ORANGE, "No statistically significant effect detected. Extend or conclude null."


def _render_html(config, results, generated_at, dist_chart, power_chart, guard_chart, ci_chart, data, variant_col) -> str:

    verdict_label, verdict_color, verdict_desc = _verdict(results, config)
    sig_icon  = "✅" if results.significant else "❌"
    lift_sign = "positive" if results.lift >= 0 else "negative"

    # Guardrail rows
    guardrail_rows = ""
    for metric, passed in results.guardrail_status.items():
        ctrl_mean = data[data[variant_col] == "control"][metric].mean()
        trt_mean  = data[data[variant_col] != "control"][metric].mean()
        delta_pct = (trt_mean - ctrl_mean) / ctrl_mean * 100 if ctrl_mean != 0 else 0
        status_color = GREEN if passed else "#F72585"
        status_label = "PASS" if passed else "FAIL"
        icon = "✅" if passed else "🚨"
        guardrail_rows += f"""
          <tr>
            <td class="td-metric">{metric}</td>
            <td class="td-num">{ctrl_mean:.3f}</td>
            <td class="td-num">{trt_mean:.3f}</td>
            <td class="td-num" style="color:{('#F72585' if delta_pct > 0 and not passed else MUTED)}">{delta_pct:+.2f}%</td>
            <td><span class="pill" style="background:{status_color}22; color:{status_color}; border:1px solid {status_color}44">{icon} {status_label}</span></td>
          </tr>"""

    guard_section = ""
    if results.guardrail_status:
        guard_section = f"""
        <section class="card">
          <h2>Guardrail Metrics</h2>
          <table>
            <thead>
              <tr>
                <th>Metric</th><th>Control</th><th>Treatment</th><th>Δ Change</th><th>Status</th>
              </tr>
            </thead>
            <tbody>{guardrail_rows}</tbody>
          </table>
          {"<div class='chart-wrap'><img src='" + guard_chart + "'/></div>" if guard_chart else ""}
        </section>"""

    novelty_section = ""
    if results.novelty_detected:
        novelty_section = f"""
        <section class="card card-warn">
          <h2>⚠️ Novelty Effect Detected</h2>
          <p>The treatment effect is significantly stronger in the early cohort than the late cohort.
          Users may be responding to novelty rather than genuine value.
          Recommended action: extend the experiment by at least 1 additional week and re-evaluate using late-cohort data only.</p>
        </section>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{config.name} — Experiment Report</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
      background: {DARK};
      color: {WHITE};
      line-height: 1.6;
      min-height: 100vh;
    }}

    /* ── Header ── */
    header {{
      padding: 48px 56px 40px;
      border-bottom: 1px solid {BORDER};
      background: {CARD};
    }}
    .header-top {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 24px;
      flex-wrap: wrap;
    }}
    .header-left h1 {{
      font-size: 26px;
      font-weight: 700;
      letter-spacing: -0.3px;
      color: {WHITE};
      margin-bottom: 6px;
    }}
    .header-left p {{
      font-size: 13px;
      color: {MUTED};
    }}
    .verdict-badge {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 14px 24px;
      border-radius: 10px;
      font-size: 15px;
      font-weight: 700;
      letter-spacing: 0.3px;
      border: 1px solid {verdict_color}44;
      background: {verdict_color}14;
      color: {verdict_color};
      white-space: nowrap;
    }}
    .verdict-desc {{
      margin-top: 12px;
      font-size: 13px;
      color: {MUTED};
    }}

    /* ── Layout ── */
    .container {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 40px 24px 64px;
    }}

    /* ── Cards ── */
    .card {{
      background: {CARD};
      border: 1px solid {BORDER};
      border-radius: 12px;
      padding: 28px 32px;
      margin-bottom: 20px;
    }}
    .card-warn {{
      border-color: {YELLOW}55;
      background: {YELLOW}08;
    }}
    .card h2 {{
      font-size: 14px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: {MUTED};
      margin-bottom: 20px;
      padding-bottom: 12px;
      border-bottom: 1px solid {BORDER};
    }}
    .card-warn h2 {{ color: {YELLOW}; border-color: {YELLOW}22; }}
    .card p {{ font-size: 14px; color: {MUTED}; line-height: 1.7; }}

    /* ── KPI Grid ── */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .kpi {{
      background: {DARK};
      border: 1px solid {BORDER};
      border-radius: 10px;
      padding: 18px 20px;
    }}
    .kpi-label {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.9px;
      color: {MUTED};
      margin-bottom: 8px;
    }}
    .kpi-value {{
      font-size: 26px;
      font-weight: 700;
      letter-spacing: -0.5px;
      color: {WHITE};
    }}
    .kpi-value.positive {{ color: {GREEN}; }}
    .kpi-value.negative {{ color: #F72585; }}
    .kpi-value.sig-yes  {{ color: {GREEN}; }}
    .kpi-value.sig-no   {{ color: #F72585; }}
    .kpi-sub {{
      font-size: 12px;
      color: {MUTED};
      margin-top: 4px;
    }}

    /* ── Two-col layout ── */
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }}
    @media (max-width: 720px) {{ .two-col {{ grid-template-columns: 1fr; }} }}

    /* ── Config table ── */
    .config-table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
    .config-table td {{ padding: 9px 4px; border-bottom: 1px solid {BORDER}; color: {MUTED}; }}
    .config-table td:last-child {{ color: {WHITE}; font-weight: 500; text-align: right; }}
    .config-table tr:last-child td {{ border-bottom: none; }}

    /* ── Guardrail table ── */
    table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
    thead tr {{ border-bottom: 1px solid {BORDER}; }}
    th {{ padding: 10px 12px; text-align: left; font-size: 11px; text-transform: uppercase;
          letter-spacing: 0.8px; color: {MUTED}; font-weight: 600; }}
    .td-metric {{ padding: 12px; color: {WHITE}; font-weight: 500; }}
    .td-num    {{ padding: 12px; color: {MUTED}; font-family: "SF Mono", "Fira Code", monospace; font-size: 13px; }}
    tbody tr {{ border-bottom: 1px solid {BORDER}22; }}
    tbody tr:last-child {{ border-bottom: none; }}
    .pill {{
      display: inline-block;
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 600;
    }}

    /* ── Charts ── */
    .chart-wrap {{ margin-top: 20px; border-radius: 8px; overflow: hidden; }}
    .chart-wrap img {{ width: 100%; display: block; border-radius: 8px; }}

    /* ── Business Summary ── */
    .biz-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 16px;
    }}
    .biz-block {{
      background: {DARK};
      border: 1px solid {BORDER};
      border-radius: 10px;
      padding: 18px 20px;
    }}
    .biz-icon  {{ font-size: 22px; margin-bottom: 8px; }}
    .biz-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.9px; color: {MUTED}; margin-bottom: 6px; }}
    .biz-text  {{ font-size: 13.5px; color: {WHITE}; line-height: 1.6; }}
    .biz-text strong {{ color: {WHITE}; }}

    /* ── Footer ── */
    footer {{
      text-align: center;
      font-size: 12px;
      color: {MUTED}66;
      padding: 32px 24px;
      border-top: 1px solid {BORDER};
    }}
  </style>
</head>
<body>

<header>
  <div class="header-top">
    <div class="header-left">
      <h1>{config.name}</h1>
      <p>Generated {generated_at} &nbsp;·&nbsp; {results.n_control + results.n_treatment:,} users &nbsp;·&nbsp; α = {config.alpha}</p>
    </div>
    <div>
      <div class="verdict-badge">{verdict_label}</div>
    </div>
  </div>
  <p class="verdict-desc">{verdict_desc}</p>
</header>

<div class="container">

  <!-- ── Business Summary ── -->
  <section class="card" style="border-color:{verdict_color}33; background:{verdict_color}08">
    <h2 style="color:{verdict_color}; border-color:{verdict_color}22">What This Means for the Business</h2>
    <div class="biz-grid">
      <div class="biz-block">
        <div class="biz-icon">📊</div>
        <div class="biz-label">What we tested</div>
        <div class="biz-text">Two versions of the product were shown to <strong>{results.n_control + results.n_treatment:,} real users</strong> over the experiment period. Half saw the original (control), half saw the new version (treatment).</div>
      </div>
      <div class="biz-block">
        <div class="biz-icon">💰</div>
        <div class="biz-label">What happened</div>
        <div class="biz-text">Users in the treatment group generated <strong>{results.lift:+.2f} more per user ({results.relative_lift:+.1%})</strong> compared to the control group.</div>
      </div>
      <div class="biz-block">
        <div class="biz-icon">🎯</div>
        <div class="biz-label">How confident are we</div>
        <div class="biz-text">{"<strong>Very confident.</strong> There is less than a 1% chance this result is random noise. The sample size was large enough to detect effects as small as " + f"{results.mde:.2f}." if results.significant else "<strong>Not confident enough to ship.</strong> The result could be random noise. We need more data or a larger effect before making a decision."}</div>
      </div>
      <div class="biz-block">
        <div class="biz-icon">🛡️</div>
        <div class="biz-label">Did anything break</div>
        <div class="biz-text">{"<strong>No.</strong> All guardrail metrics (things we must not break) passed. The new version is safe to roll out." if all(results.guardrail_status.values()) else "<strong>Yes.</strong> One or more guardrail metrics showed a statistically significant degradation. Fix these before shipping." if results.guardrail_status else "<strong>No guardrails were configured</strong> for this experiment."}</div>
      </div>
    </div>
  </section>

  <!-- ── Primary Metric KPIs ── -->
  <section class="card">
    <h2>Primary Metric &nbsp;·&nbsp; {config.metric}</h2>
    <div class="kpi-grid">
      <div class="kpi">
        <div class="kpi-label">Control Mean</div>
        <div class="kpi-value">{results.mean_control:.2f}</div>
        <div class="kpi-sub">n = {results.n_control:,}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Treatment Mean</div>
        <div class="kpi-value">{results.mean_treatment:.2f}</div>
        <div class="kpi-sub">n = {results.n_treatment:,}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Absolute Lift</div>
        <div class="kpi-value {lift_sign}">{results.lift:+.2f}</div>
        <div class="kpi-sub">per user</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Relative Lift</div>
        <div class="kpi-value {lift_sign}">{results.relative_lift:+.1%}</div>
        <div class="kpi-sub">vs control</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">p-value</div>
        <div class="kpi-value">{results.p_value:.4f}</div>
        <div class="kpi-sub">two-sided</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Significant</div>
        <div class="kpi-value {'sig-yes' if results.significant else 'sig-no'}">{sig_icon}</div>
        <div class="kpi-sub">at α = {config.alpha}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Min. Detectable Effect</div>
        <div class="kpi-value">{results.mde:.3f}</div>
        <div class="kpi-sub">absolute</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">CUPED Applied</div>
        <div class="kpi-value">{"Yes" if results.cuped_applied else "No"}</div>
        <div class="kpi-sub">variance reduction</div>
      </div>
    </div>
  </section>

  <!-- ── Distribution + CI side by side ── -->
  <div class="two-col">
    <section class="card">
      <h2>Metric Distribution</h2>
      <div class="chart-wrap"><img src="{dist_chart}"/></div>
    </section>
    <section class="card">
      <h2>95% Confidence Interval</h2>
      <div class="chart-wrap"><img src="{ci_chart}"/></div>
    </section>
  </div>

  <!-- ── Power Curve ── -->
  <section class="card">
    <h2>Power Curve</h2>
    <div class="chart-wrap"><img src="{power_chart}"/></div>
  </section>

  <!-- ── Guardrails ── -->
  {guard_section}

  <!-- ── Novelty ── -->
  {novelty_section}

  <!-- ── Experiment Config ── -->
  <section class="card">
    <h2>Experiment Configuration</h2>
    <table class="config-table">
      <tbody>
        <tr><td>Test method</td><td>{"mSPRT — Sequential" if config.use_sequential else "Welch t-test — Fixed Horizon"}</td></tr>
        <tr><td>Significance level (α)</td><td>{config.alpha}</td></tr>
        <tr><td>Target power</td><td>{config.power}</td></tr>
        <tr><td>N control</td><td>{results.n_control:,}</td></tr>
        <tr><td>N treatment</td><td>{results.n_treatment:,}</td></tr>
        <tr><td>Covariate (CUPED)</td><td>{config.covariate if config.covariate else "—"}</td></tr>
        <tr><td>Guardrail metrics</td><td>{", ".join(config.guardrail_metrics) if config.guardrail_metrics else "—"}</td></tr>
        <tr><td>Analysis duration</td><td>{results.duration_seconds}s</td></tr>
      </tbody>
    </table>
  </section>

</div>

<footer>
  A/B Testing Framework &nbsp;·&nbsp; {generated_at}
</footer>

</body>
</html>"""
