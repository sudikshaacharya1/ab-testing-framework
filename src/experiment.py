"""
experiment.py — Core Experiment Class + CLI Entry Point
=======================================================
Central orchestrator for running an A/B experiment analysis end-to-end.

Usage (CLI):
    python -m src.experiment --data data/experiment.csv \
        --metric revenue --covariate pre_revenue --alpha 0.05

The Experiment class ties together:
  - CUPED variance reduction   (src/stats/cuped.py)
  - Sequential testing         (src/stats/sequential.py)
  - Power analysis             (src/stats/power.py)
  - Guardrail checks           (src/stats/guardrails.py)
  - Novelty effect detection   (src/novelty/correction.py)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.stats.cuped import cuped_adjust
from src.stats.sequential import msprt_test
from src.stats.power import minimum_detectable_effect, required_sample_size
from src.stats.guardrails import check_guardrails
from src.novelty.correction import detect_novelty_effect


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run.

    Attributes
    ----------
    name : str
        Human-readable experiment name.
    metric : str
        Column name of the primary success metric.
    covariate : str, optional
        Column name of the pre-experiment covariate for CUPED.
    alpha : float
        Significance level (Type I error rate). Defaults to 0.05.
    power : float
        Desired statistical power (1 - Type II error). Defaults to 0.80.
    guardrail_metrics : list of str
        Column names that must not be negatively impacted.
    use_cuped : bool
        Whether to apply CUPED variance reduction before analysis.
    use_sequential : bool
        Whether to use mSPRT sequential testing instead of fixed-horizon.
    check_novelty : bool
        Whether to run the novelty-effect detection module.
    """

    name: str = "Unnamed Experiment"
    metric: str = "metric"
    covariate: Optional[str] = None
    alpha: float = 0.05
    power: float = 0.80
    guardrail_metrics: List[str] = field(default_factory=list)
    use_cuped: bool = True
    use_sequential: bool = False
    check_novelty: bool = False


@dataclass
class ExperimentResults:
    """Results produced by an experiment analysis.

    Attributes
    ----------
    mean_control : float
        Mean of the metric in the control group.
    mean_treatment : float
        Mean of the metric in the treatment group.
    lift : float
        Absolute lift (treatment - control).
    relative_lift : float
        Relative lift as a fraction of the control mean.
    p_value : float
        Two-sided p-value from the chosen test.
    significant : bool
        Whether the result is statistically significant at alpha.
    guardrail_status : dict
        Pass/fail status for each guardrail metric.
    novelty_detected : bool
        Whether a novelty effect was flagged.
    mde : float
        Minimum detectable effect given the observed sample sizes.
    """

    mean_control: float = 0.0
    mean_treatment: float = 0.0
    lift: float = 0.0
    relative_lift: float = 0.0
    p_value: float = 1.0
    significant: bool = False
    guardrail_status: Dict[str, bool] = field(default_factory=dict)
    novelty_detected: bool = False
    mde: float = 0.0


# ---------------------------------------------------------------------------
# Core Experiment Class
# ---------------------------------------------------------------------------

class Experiment:
    """Run a full A/B experiment analysis pipeline.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment settings (metric, alpha, flags, etc.).

    Examples
    --------
    >>> import pandas as pd
    >>> from src.experiment import Experiment, ExperimentConfig
    >>> df = pd.DataFrame({
    ...     "variant": ["control"] * 500 + ["treatment"] * 500,
    ...     "revenue": [10.0] * 500 + [11.0] * 500,
    ...     "pre_revenue": [9.5] * 1000,
    ... })
    >>> cfg = ExperimentConfig(metric="revenue", covariate="pre_revenue")
    >>> exp = Experiment(cfg)
    >>> results = exp.run(df)
    >>> results.significant
    True
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, data: pd.DataFrame, variant_col: str = "variant") -> ExperimentResults:
        """Execute the full analysis pipeline on *data*.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain at least *variant_col* and ``config.metric``.
            Rows with ``variant == "control"`` form the control group;
            all other unique variant values are pooled as treatment.
        variant_col : str
            Column that identifies control vs treatment.

        Returns
        -------
        ExperimentResults
        """
        self._validate(data, variant_col)

        control = data[data[variant_col] == "control"].copy()
        treatment = data[data[variant_col] != "control"].copy()

        metric = self.config.metric
        covariate = self.config.covariate

        # --- 1. CUPED adjustment -------------------------------------------
        if self.config.use_cuped and covariate is not None:
            control_y, treatment_y = cuped_adjust(
                control[metric].values,
                treatment[metric].values,
                control[covariate].values,
                treatment[covariate].values,
            )
        else:
            control_y = control[metric].values
            treatment_y = treatment[metric].values

        # --- 2. Hypothesis test -------------------------------------------
        if self.config.use_sequential:
            p_value, _ = msprt_test(control_y, treatment_y, alpha=self.config.alpha)
        else:
            from scipy import stats as sp_stats
            _, p_value = sp_stats.ttest_ind(treatment_y, control_y)

        # --- 3. Guardrail checks ------------------------------------------
        guardrail_status: Dict[str, bool] = {}
        if self.config.guardrail_metrics:
            guardrail_status = check_guardrails(
                data, variant_col, self.config.guardrail_metrics, alpha=self.config.alpha
            )

        # --- 4. Novelty effect -------------------------------------------
        novelty_detected = False
        if self.config.check_novelty and "days_since_signup" in data.columns:
            novelty_detected = detect_novelty_effect(data, variant_col, metric)

        # --- 5. Summary statistics ----------------------------------------
        mean_ctrl = float(np.mean(control_y))
        mean_trt = float(np.mean(treatment_y))
        lift = mean_trt - mean_ctrl
        relative_lift = lift / mean_ctrl if mean_ctrl != 0 else float("nan")

        pooled_std = float(np.std(np.concatenate([control_y, treatment_y]), ddof=1))
        n = min(len(control_y), len(treatment_y))
        mde = minimum_detectable_effect(
            n, pooled_std, alpha=self.config.alpha, power=self.config.power
        )

        return ExperimentResults(
            mean_control=mean_ctrl,
            mean_treatment=mean_trt,
            lift=lift,
            relative_lift=relative_lift,
            p_value=float(p_value),
            significant=float(p_value) < self.config.alpha,
            guardrail_status=guardrail_status,
            novelty_detected=novelty_detected,
            mde=mde,
        )

    def print_summary(self, results: ExperimentResults) -> None:
        """Pretty-print a results summary to stdout."""
        cfg = self.config
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  Experiment: {cfg.name}")
        print(sep)
        print(f"  Metric            : {cfg.metric}")
        print(f"  Control mean      : {results.mean_control:.4f}")
        print(f"  Treatment mean    : {results.mean_treatment:.4f}")
        print(f"  Absolute lift     : {results.lift:+.4f}")
        print(f"  Relative lift     : {results.relative_lift:+.2%}")
        print(f"  p-value           : {results.p_value:.4f}")
        print(f"  Significant (α={cfg.alpha}) : {'✅ YES' if results.significant else '❌ NO'}")
        print(f"  MDE (given n)     : {results.mde:.4f}")

        if results.guardrail_status:
            print(f"\n  Guardrail Metrics:")
            for gm, passed in results.guardrail_status.items():
                icon = "✅" if passed else "🚨"
                print(f"    {icon} {gm}: {'PASS' if passed else 'FAIL'}")

        if results.novelty_detected:
            print("\n  ⚠️  Novelty effect detected — treat results with caution.")

        print(f"{sep}\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate(self, data: pd.DataFrame, variant_col: str) -> None:
        required = {variant_col, self.config.metric}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")
        if "control" not in data[variant_col].unique():
            raise ValueError(f"No rows with {variant_col}='control' found.")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.experiment",
        description="Run an A/B experiment analysis from a CSV file.",
    )
    parser.add_argument("--data", required=True, help="Path to experiment CSV.")
    parser.add_argument("--metric", required=True, help="Primary metric column name.")
    parser.add_argument("--covariate", default=None, help="Pre-experiment covariate for CUPED.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default 0.05).")
    parser.add_argument("--power", type=float, default=0.80, help="Desired power (default 0.80).")
    parser.add_argument("--variant-col", default="variant", help="Variant column name (default 'variant').")
    parser.add_argument("--guardrails", nargs="*", default=[], help="Guardrail metric column names.")
    parser.add_argument("--no-cuped", action="store_true", help="Disable CUPED variance reduction.")
    parser.add_argument("--sequential", action="store_true", help="Use mSPRT sequential testing.")
    parser.add_argument("--novelty", action="store_true", help="Run novelty effect detection.")
    parser.add_argument("--name", default="CLI Experiment", help="Experiment name for display.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        data = pd.read_csv(args.data)
    except FileNotFoundError:
        print(f"ERROR: File not found: {args.data}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR reading CSV: {exc}", file=sys.stderr)
        return 1

    config = ExperimentConfig(
        name=args.name,
        metric=args.metric,
        covariate=args.covariate,
        alpha=args.alpha,
        power=args.power,
        guardrail_metrics=args.guardrails,
        use_cuped=not args.no_cuped,
        use_sequential=args.sequential,
        check_novelty=args.novelty,
    )

    exp = Experiment(config)
    try:
        results = exp.run(data, variant_col=args.variant_col)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    exp.print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
