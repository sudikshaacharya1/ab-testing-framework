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
  - HTML report generation     (src/reporting.py)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.logger import configure_logging, get_logger
from src.stats.cuped import cuped_adjust
from src.stats.sequential import msprt_test
from src.stats.power import minimum_detectable_effect
from src.stats.guardrails import check_guardrails
from src.novelty.correction import detect_novelty_effect

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

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
    """Results produced by an experiment analysis."""

    mean_control: float = 0.0
    mean_treatment: float = 0.0
    lift: float = 0.0
    relative_lift: float = 0.0
    p_value: float = 1.0
    significant: bool = False
    guardrail_status: Dict[str, bool] = field(default_factory=dict)
    novelty_detected: bool = False
    mde: float = 0.0
    n_control: int = 0
    n_treatment: int = 0
    cuped_applied: bool = False
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core Experiment Class
# ---------------------------------------------------------------------------

class Experiment:
    """Run a full A/B experiment analysis pipeline."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        logger.info(
            "Experiment initialised",
            extra={
                "experiment_name": config.name,
                "metric": config.metric,
                "alpha": config.alpha,
                "power": config.power,
                "use_cuped": config.use_cuped,
                "use_sequential": config.use_sequential,
                "guardrail_metrics": config.guardrail_metrics,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, data: pd.DataFrame, variant_col: str = "variant") -> ExperimentResults:
        """Execute the full analysis pipeline on *data*."""
        start = time.perf_counter()

        logger.info(
            "Starting experiment run",
            extra={
                "experiment_name": self.config.name,
                "total_rows": len(data),
                "columns": list(data.columns),
            },
        )

        self._validate(data, variant_col)

        control   = data[data[variant_col] == "control"].copy()
        treatment = data[data[variant_col] != "control"].copy()
        n_ctrl, n_trt = len(control), len(treatment)

        logger.info(
            "Group sizes validated",
            extra={
                "n_control": n_ctrl,
                "n_treatment": n_trt,
                "split_ratio": round(n_ctrl / (n_ctrl + n_trt), 3),
            },
        )

        metric    = self.config.metric
        covariate = self.config.covariate
        cuped_applied = False

        # --- 1. CUPED adjustment ------------------------------------------
        if self.config.use_cuped and covariate is not None:
            logger.debug(
                "Applying CUPED variance reduction",
                extra={"covariate": covariate},
            )
            try:
                control_y, treatment_y = cuped_adjust(
                    control[metric].values,
                    treatment[metric].values,
                    control[covariate].values,
                    treatment[covariate].values,
                )
                cuped_applied = True
                var_before = float(np.var(np.concatenate([
                    control[metric].values, treatment[metric].values
                ]), ddof=1))
                var_after = float(np.var(np.concatenate([control_y, treatment_y]), ddof=1))
                logger.info(
                    "CUPED adjustment applied",
                    extra={
                        "variance_before": round(var_before, 4),
                        "variance_after":  round(var_after, 4),
                        "variance_reduction_pct": round((1 - var_after / var_before) * 100, 2),
                    },
                )
            except Exception as exc:
                logger.warning(
                    "CUPED adjustment failed — falling back to raw metric",
                    extra={"error": str(exc)},
                )
                control_y  = control[metric].values
                treatment_y = treatment[metric].values
        else:
            reason = "disabled by config" if not self.config.use_cuped else "no covariate provided"
            logger.info("CUPED skipped", extra={"reason": reason})
            control_y  = control[metric].values
            treatment_y = treatment[metric].values

        # --- 2. Hypothesis test -------------------------------------------
        test_method = "mSPRT" if self.config.use_sequential else "Welch t-test"
        logger.debug("Running hypothesis test", extra={"method": test_method})

        if self.config.use_sequential:
            p_value, lambda_t = msprt_test(control_y, treatment_y, alpha=self.config.alpha)
            logger.info(
                "mSPRT test complete",
                extra={"p_value": round(p_value, 6), "lambda_t": round(lambda_t, 4)},
            )
        else:
            from scipy import stats as sp_stats
            _, p_value = sp_stats.ttest_ind(treatment_y, control_y)
            logger.info(
                "Welch t-test complete",
                extra={"p_value": round(float(p_value), 6)},
            )

        # --- 3. Guardrail checks ------------------------------------------
        guardrail_status: Dict[str, bool] = {}
        if self.config.guardrail_metrics:
            logger.debug(
                "Checking guardrail metrics",
                extra={"guardrails": self.config.guardrail_metrics},
            )
            guardrail_status = check_guardrails(
                data, variant_col, self.config.guardrail_metrics, alpha=self.config.alpha
            )
            failed = [m for m, ok in guardrail_status.items() if not ok]
            if failed:
                logger.warning(
                    "Guardrail metrics FAILED",
                    extra={"failed_guardrails": failed},
                )
            else:
                logger.info(
                    "All guardrail metrics passed",
                    extra={"guardrails": list(guardrail_status.keys())},
                )

        # --- 4. Novelty effect -------------------------------------------
        novelty_detected = False
        if self.config.check_novelty and "days_since_signup" in data.columns:
            logger.debug("Running novelty effect detection")
            novelty_detected = detect_novelty_effect(data, variant_col, metric)
            if novelty_detected:
                logger.warning(
                    "Novelty effect detected",
                    extra={"experiment_name": self.config.name},
                )
            else:
                logger.info("No novelty effect detected")
        elif self.config.check_novelty:
            logger.warning(
                "Novelty check requested but 'days_since_signup' column not found",
                extra={"available_columns": list(data.columns)},
            )

        # --- 5. Summary statistics ----------------------------------------
        mean_ctrl = float(np.mean(control_y))
        mean_trt  = float(np.mean(treatment_y))
        lift      = mean_trt - mean_ctrl
        relative_lift = lift / mean_ctrl if mean_ctrl != 0 else float("nan")

        pooled_std = float(np.std(np.concatenate([control_y, treatment_y]), ddof=1))
        n = min(len(control_y), len(treatment_y))
        mde = minimum_detectable_effect(
            n, pooled_std, alpha=self.config.alpha, power=self.config.power
        )

        significant = float(p_value) < self.config.alpha
        duration = time.perf_counter() - start

        logger.info(
            "Experiment run complete",
            extra={
                "experiment_name":  self.config.name,
                "mean_control":     round(mean_ctrl, 4),
                "mean_treatment":   round(mean_trt, 4),
                "lift":             round(lift, 4),
                "relative_lift_pct": round(relative_lift * 100, 2),
                "p_value":          round(float(p_value), 6),
                "significant":      significant,
                "mde":              round(mde, 4),
                "duration_seconds": round(duration, 3),
            },
        )

        if significant:
            logger.info("🟢 Result: SIGNIFICANT — consider shipping")
        else:
            logger.info("🔴 Result: NOT SIGNIFICANT — do not ship based on this run")

        return ExperimentResults(
            mean_control=mean_ctrl,
            mean_treatment=mean_trt,
            lift=lift,
            relative_lift=relative_lift,
            p_value=float(p_value),
            significant=significant,
            guardrail_status=guardrail_status,
            novelty_detected=novelty_detected,
            mde=mde,
            n_control=n_ctrl,
            n_treatment=n_trt,
            cuped_applied=cuped_applied,
            duration_seconds=round(duration, 3),
        )

    def print_summary(self, results: ExperimentResults) -> None:
        """Pretty-print a results summary to stdout."""
        cfg = self.config
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  Experiment: {cfg.name}")
        print(sep)
        print(f"  Metric            : {cfg.metric}")
        print(f"  N (control)       : {results.n_control:,}")
        print(f"  N (treatment)     : {results.n_treatment:,}")
        print(f"  CUPED applied     : {'Yes' if results.cuped_applied else 'No'}")
        print(f"  Control mean      : {results.mean_control:.4f}")
        print(f"  Treatment mean    : {results.mean_treatment:.4f}")
        print(f"  Absolute lift     : {results.lift:+.4f}")
        print(f"  Relative lift     : {results.relative_lift:+.2%}")
        print(f"  p-value           : {results.p_value:.6f}")
        print(f"  Significant (α={cfg.alpha}) : {'✅ YES' if results.significant else '❌ NO'}")
        print(f"  MDE (given n)     : {results.mde:.4f}")

        if results.guardrail_status:
            print(f"\n  Guardrail Metrics:")
            for gm, passed in results.guardrail_status.items():
                icon = "✅" if passed else "🚨"
                print(f"    {icon} {gm}: {'PASS' if passed else 'FAIL'}")

        if results.novelty_detected:
            print("\n  ⚠️  Novelty effect detected — treat results with caution.")

        print(f"\n  Completed in {results.duration_seconds}s")
        print(f"{sep}\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate(self, data: pd.DataFrame, variant_col: str) -> None:
        logger.debug("Validating input data", extra={"shape": list(data.shape)})
        required = {variant_col, self.config.metric}
        missing  = required - set(data.columns)
        if missing:
            logger.error("Missing required columns", extra={"missing": list(missing)})
            raise ValueError(f"DataFrame is missing columns: {missing}")
        if "control" not in data[variant_col].unique():
            logger.error(
                "No control group found",
                extra={"variant_col": variant_col, "found_values": list(data[variant_col].unique())},
            )
            raise ValueError(f"No rows with {variant_col}='control' found.")
        null_counts = data[[variant_col, self.config.metric]].isnull().sum().to_dict()
        if any(v > 0 for v in null_counts.values()):
            logger.warning("Null values detected in key columns", extra={"null_counts": null_counts})


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.experiment",
        description="Run an A/B experiment analysis from a CSV file.",
    )
    parser.add_argument("--data",        required=True,  help="Path to experiment CSV.")
    parser.add_argument("--metric",      required=True,  help="Primary metric column name.")
    parser.add_argument("--covariate",   default=None,   help="Pre-experiment covariate for CUPED.")
    parser.add_argument("--alpha",       type=float, default=0.05)
    parser.add_argument("--power",       type=float, default=0.80)
    parser.add_argument("--variant-col", default="variant")
    parser.add_argument("--guardrails",  nargs="*", default=[])
    parser.add_argument("--no-cuped",    action="store_true")
    parser.add_argument("--sequential",  action="store_true")
    parser.add_argument("--novelty",     action="store_true")
    parser.add_argument("--name",        default="CLI Experiment")
    parser.add_argument("--report",      default=None, help="Path to save HTML report (e.g. report.html).")
    parser.add_argument("--log-level",   default=None, help="LOG_LEVEL override: DEBUG|INFO|WARNING|ERROR")
    parser.add_argument("--log-format",  default=None, help="LOG_FORMAT override: text|json")
    parser.add_argument("--log-file",    default=None, help="Path to write log file.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    configure_logging(
        level    = args.log_level,
        fmt      = args.log_format,
        log_file = args.log_file,
    )

    logger.info("CLI invoked", extra={"args": vars(args)})

    try:
        data = pd.read_csv(args.data)
        logger.info("Data loaded", extra={"path": args.data, "rows": len(data), "cols": len(data.columns)})
    except FileNotFoundError:
        logger.error("Data file not found", extra={"path": args.data})
        print(f"ERROR: File not found: {args.data}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Failed to read data file", extra={"path": args.data, "error": str(exc)})
        print(f"ERROR reading CSV: {exc}", file=sys.stderr)
        return 1

    config = ExperimentConfig(
        name             = args.name,
        metric           = args.metric,
        covariate        = args.covariate,
        alpha            = args.alpha,
        power            = args.power,
        guardrail_metrics = args.guardrails,
        use_cuped        = not args.no_cuped,
        use_sequential   = args.sequential,
        check_novelty    = args.novelty,
    )

    exp = Experiment(config)
    try:
        results = exp.run(data, variant_col=args.variant_col)
    except ValueError as exc:
        logger.error("Experiment run failed", extra={"error": str(exc)})
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    exp.print_summary(results)

    if args.report:
        from src.reporting import generate_html_report
        generate_html_report(config, results, data, output_path=args.report)
        logger.info("HTML report saved", extra={"path": args.report})
        print(f"📄 Report saved → {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
