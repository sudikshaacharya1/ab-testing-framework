"""
compare.py — Multi-Experiment Comparison Dashboard
====================================================
Run multiple experiments and generate a single HTML dashboard
comparing all results side by side.

Usage
-----
    from src.compare import ExperimentComparison
    from src.experiment import Experiment, ExperimentConfig
    import pandas as pd

    runs = [
        (ExperimentConfig(name="Test A", metric="revenue"), pd.read_csv("a.csv")),
        (ExperimentConfig(name="Test B", metric="revenue"), pd.read_csv("b.csv")),
    ]
    comp = ExperimentComparison(runs)
    comp.run_all()
    comp.save_report("reports/comparison.html")
"""

from __future__ import annotations

import base64
import io
import os
from datetime import datetime, timezone
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiment import Experiment, ExperimentConfig, ExperimentResults
from src.logger import get_logger

logger = get_logger(__name__)

# ── Palette ────────────────────────────────────────────────────────────────
DARK   = "#0D1117"
CARD   = "#161B22"
BORDER = "#21262D"
MUTED  = "#8B949E"
WHITE  = "#F0F6FC"
BLUE   = "#4361EE"
ORANGE = "#F72585"
GREEN  = "#06D6A0"
YELLOW = "#FFD166"
PURPLE = "#7B2FBE"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ExperimentComparison:
    """Run and compare multiple experiments in one dashboard.

    Parameters
    ----------
    runs : list of (ExperimentConfig, pd.DataFrame)
        Each tuple is one experiment to run.
    variant_col : str
        Column identifying control vs treatment.
    """

    def __init__(
        self,
        runs: List[Tuple[ExperimentConfig, pd.DataFrame]],
        variant_col: str = "variant",
    ) -> None:
        self.runs        = runs
        self.variant_col = variant_col
        self.results: List[Tuple[ExperimentConfig, ExperimentResults, pd.DataFrame]] = []

    def run_all(self) -> None:
        """Execute every experiment and store results."""
        logger.info("Starting comparison run", extra={"n_experiments": len(self.runs)})
        for config, data in self.runs:
            exp    = Experiment(config)
            result = exp.run(data, variant_col=self.variant_col)
            self.results.append((config, result, data))
            logger.info(
                "Experiment complete",
                extra={
                    "experiment_name": config.name,
                    "lift":            round(result.lift, 4),
                    "p_value":         round(result.p_value, 6),
                    "significant":     result.significant,
                },
            )
        logger.info("All experiments complete", extra={"total": len(self.results)})

    def save_report(self, output_path: str = "reports/comparison.html") -> None:
        """Generate and save the comparison HTML dashboard."""
        if not self.results:
            raise RuntimeError("Call run_all() before save_report().")

        logger.info("Building comparison report", extra={"output_path": output_path})
        html = self._render(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        logger.info("Comparison report saved", extra={"output_path": output_path})
        print(f"📊 Comparison report saved → {output_path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _verdict(self, config, result) -> tuple:
        guardrails_ok = all(result.guardrail_status.values()) if result.guardrail_status else True
        if result.novelty_detected:
            return "⚠️ NOVELTY",    YELLOW, "Novelty effect"
        if result.significant and guardrails_ok:
            return "🚀 SHIP",       GREEN,  "Significant · All guardrails pass"
        if result.significant and not guardrails_ok:
            return "🔧 FIX FIRST",  YELLOW, "Significant · Guardrail failure"
        return "⏳ EXTEND",         ORANGE, "Not significant"

    def _b64_png(self, fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                    facecolor=CARD, edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    def _lift_comparison_chart(self) -> str:
        """Horizontal bar chart comparing relative lifts across experiments."""
        names  = [cfg.name for cfg, _, _ in self.results]
        lifts  = [res.relative_lift * 100 for _, res, _ in self.results]
        sigs   = [res.significant for _, res, _ in self.results]
        colors = [GREEN if (s and l > 0) else (ORANGE if not s else YELLOW)
                  for s, l in zip(sigs, lifts)]

        fig, ax = plt.subplots(figsize=(10, max(3, len(names) * 1.1)))
        fig.patch.set_facecolor(CARD)
        ax.set_facecolor(CARD)

        bars = ax.barh(names, lifts, color=colors, alpha=0.85,
                       height=0.5, linewidth=0)
        ax.axvline(0, color=MUTED, linewidth=1.2, linestyle="--", alpha=0.6)

        for bar, lift, sig in zip(bars, lifts, sigs):
            label = f"{lift:+.1f}%{'  ✓' if sig else '  ~'}"
            ax.text(
                bar.get_width() + (0.3 if lift >= 0 else -0.3),
                bar.get_y() + bar.get_height() / 2,
                label, va="center",
                ha="left" if lift >= 0 else "right",
                color=WHITE, fontsize=10, fontweight="600",
            )

        ax.set_xlabel("Relative Lift (%)", fontsize=10, color=MUTED)
        ax.set_title("Relative Lift by Experiment", fontsize=13,
                     fontweight="bold", color=WHITE, pad=14)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(MUTED)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(axis="x", color=BORDER, linewidth=0.5, alpha=0.5)
        ax.invert_yaxis()
        fig.tight_layout(pad=1.5)
        return self._b64_png(fig)

    def _pvalue_chart(self) -> str:
        """Dot plot of p-values with alpha threshold line."""
        names   = [cfg.name for cfg, _, _ in self.results]
        pvals   = [min(res.p_value, 0.5) for _, res, _ in self.results]
        colors  = [GREEN if res.significant else ORANGE for _, res, _ in self.results]

        fig, ax = plt.subplots(figsize=(10, max(3, len(names) * 1.1)))
        fig.patch.set_facecolor(CARD)
        ax.set_facecolor(CARD)

        ax.scatter(pvals, names, color=colors, s=120, zorder=5)
        ax.axvline(0.05, color=YELLOW, linewidth=1.5, linestyle="--",
                   alpha=0.8, label="α = 0.05")
        ax.fill_betweenx([-0.5, len(names) - 0.5], 0, 0.05,
                         alpha=0.06, color=GREEN)

        for pv, name in zip(pvals, names):
            ax.annotate(f"p={pv:.4f}", xy=(pv, name),
                        xytext=(8, 0), textcoords="offset points",
                        color=MUTED, fontsize=8.5, va="center")

        ax.set_xlabel("p-value", fontsize=10, color=MUTED)
        ax.set_title("p-values (green region = significant)", fontsize=13,
                     fontweight="bold", color=WHITE, pad=14)
        ax.set_xlim(-0.01, 0.55)
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(axis="x", color=BORDER, linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=9, facecolor=DARK, edgecolor=BORDER, labelcolor=WHITE)
        ax.invert_yaxis()
        fig.tight_layout(pad=1.5)
        return self._b64_png(fig)

    def _render(self, output_path: str) -> str:
        generated_at  = datetime.now(tz=timezone.utc).strftime("%d %b %Y · %H:%M UTC")
        lift_chart    = self._lift_comparison_chart()
        pvalue_chart  = self._pvalue_chart()
        exp_cards     = self._experiment_cards()
        summary_table = self._summary_table()

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Experiment Comparison Dashboard</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: {DARK};
      color: {WHITE};
      line-height: 1.6;
    }}

    header {{
      padding: 44px 56px 36px;
      border-bottom: 1px solid {BORDER};
      background: {CARD};
    }}
    header h1 {{ font-size: 26px; font-weight: 700; letter-spacing: -0.3px; margin-bottom: 6px; }}
    header p  {{ font-size: 13px; color: {MUTED}; }}

    .container {{ max-width: 1200px; margin: 0 auto; padding: 36px 24px 64px; }}

    .card {{
      background: {CARD};
      border: 1px solid {BORDER};
      border-radius: 12px;
      padding: 28px 32px;
      margin-bottom: 20px;
    }}
    .card h2 {{
      font-size: 13px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: {MUTED};
      margin-bottom: 20px;
      padding-bottom: 12px;
      border-bottom: 1px solid {BORDER};
    }}

    /* ── Summary table ── */
    table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
    thead tr {{ border-bottom: 1px solid {BORDER}; }}
    th {{
      padding: 10px 14px;
      text-align: left;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: {MUTED};
      font-weight: 600;
    }}
    tbody td {{ padding: 13px 14px; border-bottom: 1px solid {BORDER}22; }}
    tbody tr:last-child td {{ border-bottom: none; }}
    tbody tr:hover {{ background: {BORDER}44; }}
    .td-name  {{ font-weight: 600; color: {WHITE}; }}
    .td-num   {{ font-family: "SF Mono","Fira Code",monospace; color: {MUTED}; }}
    .td-pos   {{ color: {GREEN};  font-weight: 600; font-family: monospace; }}
    .td-neg   {{ color: {ORANGE}; font-weight: 600; font-family: monospace; }}
    .td-neut  {{ color: {MUTED};  font-weight: 600; font-family: monospace; }}
    .pill {{
      display: inline-block;
      padding: 3px 11px;
      border-radius: 20px;
      font-size: 11.5px;
      font-weight: 700;
      white-space: nowrap;
    }}

    /* ── Experiment cards grid ── */
    .exp-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }}
    .exp-card {{
      background: {DARK};
      border: 1px solid {BORDER};
      border-radius: 12px;
      padding: 22px 24px;
      border-top: 3px solid var(--accent);
    }}
    .exp-card-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 14px;
    }}
    .exp-card-name  {{ font-size: 14px; font-weight: 600; color: {WHITE}; line-height: 1.4; }}
    .exp-card-badge {{
      font-size: 11px;
      font-weight: 700;
      padding: 3px 10px;
      border-radius: 20px;
      white-space: nowrap;
      margin-left: 10px;
    }}
    .exp-stat {{ display: flex; justify-content: space-between; padding: 7px 0;
                 border-bottom: 1px solid {BORDER}; font-size: 13px; }}
    .exp-stat:last-child {{ border-bottom: none; }}
    .exp-stat-label {{ color: {MUTED}; }}
    .exp-stat-value {{ font-weight: 600; font-family: monospace; }}

    /* ── Charts ── */
    .chart-wrap {{ border-radius: 8px; overflow: hidden; margin-top: 4px; }}
    .chart-wrap img {{ width: 100%; display: block; border-radius: 8px; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    @media (max-width: 800px) {{ .two-col {{ grid-template-columns: 1fr; }} }}

    footer {{
      text-align: center;
      font-size: 12px;
      color: {MUTED}55;
      padding: 28px 24px;
      border-top: 1px solid {BORDER};
    }}
  </style>
</head>
<body>

<header>
  <h1>📊 Experiment Comparison Dashboard</h1>
  <p>{len(self.results)} experiments &nbsp;·&nbsp; Generated {generated_at}</p>
</header>

<div class="container">

  <!-- ── Experiment Cards ── -->
  <section class="card">
    <h2>At a Glance</h2>
    <div class="exp-grid">
      {exp_cards}
    </div>
  </section>

  <!-- ── Summary Table ── -->
  <section class="card">
    <h2>Full Comparison Table</h2>
    {summary_table}
  </section>

  <!-- ── Charts ── -->
  <div class="two-col">
    <section class="card">
      <h2>Relative Lift by Experiment</h2>
      <div class="chart-wrap"><img src="{lift_chart}"/></div>
    </section>
    <section class="card">
      <h2>p-values</h2>
      <div class="chart-wrap"><img src="{pvalue_chart}"/></div>
    </section>
  </div>

</div>

<footer>A/B Testing Framework &nbsp;·&nbsp; {generated_at}</footer>
</body>
</html>"""

    def _experiment_cards(self) -> str:
        palette = [BLUE, GREEN, ORANGE, PURPLE, YELLOW]
        cards   = ""
        for i, (config, result, _) in enumerate(self.results):
            label, color, desc = self._verdict(config, result)
            accent = palette[i % len(palette)]
            lift_class = "td-pos" if result.lift > 0 else "td-neg"
            sig_color  = GREEN if result.significant else ORANGE

            guardrail_line = ""
            if result.guardrail_status:
                failed = sum(1 for v in result.guardrail_status.values() if not v)
                total  = len(result.guardrail_status)
                g_color = GREEN if failed == 0 else ORANGE
                guardrail_line = f"""
                <div class="exp-stat">
                  <span class="exp-stat-label">Guardrails</span>
                  <span class="exp-stat-value" style="color:{g_color}">
                    {total - failed}/{total} pass
                  </span>
                </div>"""

            cards += f"""
      <div class="exp-card" style="--accent:{accent}">
        <div class="exp-card-header">
          <div class="exp-card-name">{config.name}</div>
          <div class="exp-card-badge" style="background:{color}22;color:{color};border:1px solid {color}44">{label}</div>
        </div>
        <div class="exp-stat">
          <span class="exp-stat-label">Relative lift</span>
          <span class="exp-stat-value {lift_class}">{result.relative_lift:+.1%}</span>
        </div>
        <div class="exp-stat">
          <span class="exp-stat-label">Absolute lift</span>
          <span class="exp-stat-value {lift_class}">{result.lift:+.3f}</span>
        </div>
        <div class="exp-stat">
          <span class="exp-stat-label">p-value</span>
          <span class="exp-stat-value" style="color:{sig_color}">{result.p_value:.4f}</span>
        </div>
        <div class="exp-stat">
          <span class="exp-stat-label">Users</span>
          <span class="exp-stat-value" style="color:{WHITE}">{result.n_control + result.n_treatment:,}</span>
        </div>
        {guardrail_line}
        <div style="margin-top:12px;font-size:12px;color:{MUTED}">{desc}</div>
      </div>"""
        return cards

    def _summary_table(self) -> str:
        rows = ""
        for config, result, _ in self.results:
            label, color, _ = self._verdict(config, result)
            lift_class = "td-pos" if result.lift > 0 else "td-neg"
            sig_color  = GREEN if result.significant else ORANGE
            sig_label  = "Yes" if result.significant else "No"

            guard_cell = "—"
            if result.guardrail_status:
                failed = sum(1 for v in result.guardrail_status.values() if not v)
                total  = len(result.guardrail_status)
                g_color = GREEN if failed == 0 else ORANGE
                guard_cell = f'<span style="color:{g_color};font-weight:600">{total-failed}/{total} pass</span>'

            rows += f"""
          <tr>
            <td class="td-name">{config.name}</td>
            <td class="td-num">{config.metric}</td>
            <td class="{lift_class}">{result.relative_lift:+.1%}</td>
            <td class="{lift_class}">{result.lift:+.3f}</td>
            <td class="td-num" style="color:{sig_color}">{result.p_value:.4f}</td>
            <td style="color:{sig_color};font-weight:600">{sig_label}</td>
            <td class="td-num">{result.mde:.3f}</td>
            <td>{guard_cell}</td>
            <td><span class="pill" style="background:{color}18;color:{color};border:1px solid {color}33">{label}</span></td>
          </tr>"""

        return f"""
        <table>
          <thead>
            <tr>
              <th>Experiment</th>
              <th>Metric</th>
              <th>Rel. Lift</th>
              <th>Abs. Lift</th>
              <th>p-value</th>
              <th>Significant</th>
              <th>MDE</th>
              <th>Guardrails</th>
              <th>Verdict</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>"""
