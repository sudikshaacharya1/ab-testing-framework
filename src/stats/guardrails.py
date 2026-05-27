"""
guardrails.py — Guardrail Metric Checks
========================================
Guardrail metrics are key business metrics that an experiment must *not*
negatively impact, even if the primary metric improves.

Examples of guardrail metrics at real companies
-----------------------------------------------
- Page load time (must not increase)
- Crash rate (must not increase)
- Customer support contacts per user (must not increase)
- Revenue per user (global, not just the tested surface)

Statistical Approach
--------------------
Each guardrail metric is tested independently using a two-sided Welch's t-test.
A guardrail **fails** if:

    1. The treatment mean is worse than control, AND
    2. The difference is statistically significant at `alpha`.

This is a directional test: an *improvement* in a guardrail metric does not
cause failure. Only statistically significant degradations trigger an alert.

Multiple Testing
----------------
When many guardrail metrics are checked simultaneously, the family-wise error
rate (FWER) is inflated.  We apply the Holm-Bonferroni correction by default
to control the FWER at alpha.

Reference
---------
Dmitriev, P., Fabijan, A., & Hasson, U. (2016). A dirty dozen: Twelve common
metric interpretation pitfalls in online controlled experiments. KDD 2017.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.logger import get_logger

logger = get_logger(__name__)


def check_guardrails(
    data: pd.DataFrame,
    variant_col: str,
    guardrail_metrics: List[str],
    alpha: float = 0.05,
    correction: str = "holm",
    direction: str = "lower_is_worse",
) -> Dict[str, bool]:
    """Test each guardrail metric for statistically significant degradation.

    Parameters
    ----------
    data : pd.DataFrame
        Full experiment dataset with variant and metric columns.
    variant_col : str
        Column name identifying the variant (must contain 'control').
    guardrail_metrics : list of str
        Column names to test as guardrail metrics.
    alpha : float
        Significance level for the family-wise test.
    correction : {'holm', 'bonferroni', 'none'}
        Multiple testing correction method.
    direction : {'lower_is_worse', 'higher_is_worse'}
        Whether the guardrail fails when treatment mean is lower ('lower_is_worse')
        or higher ('higher_is_worse') than control.

    Returns
    -------
    dict
        Mapping of guardrail metric name → True (PASS) / False (FAIL).

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> df = pd.DataFrame({
    ...     "variant": ["control"]*n + ["treatment"]*n,
    ...     "latency": np.r_[rng.normal(100,10,n), rng.normal(115,10,n)],
    ... })
    >>> check_guardrails(df, "variant", ["latency"], direction="higher_is_worse")
    {'latency': False}
    """
    control_mask = data[variant_col] == "control"
    control_data = data[control_mask]
    treatment_data = data[~control_mask]

    raw_pvalues: Dict[str, float] = {}
    raw_directions: Dict[str, bool] = {}  # True if movement is "worse"

    for metric in guardrail_metrics:
        c_vals = control_data[metric].dropna().values
        t_vals = treatment_data[metric].dropna().values

        _, p = sp_stats.ttest_ind(t_vals, c_vals, equal_var=False)

        trt_mean = np.mean(t_vals)
        ctrl_mean = np.mean(c_vals)

        if direction == "lower_is_worse":
            is_worse = trt_mean < ctrl_mean
        else:  # higher_is_worse
            is_worse = trt_mean > ctrl_mean

        raw_pvalues[metric] = float(p)
        raw_directions[metric] = bool(is_worse)

    # Apply multiple testing correction
    adjusted = _apply_correction(raw_pvalues, alpha=alpha, method=correction)

    results: Dict[str, bool] = {}
    for metric in guardrail_metrics:
        degraded   = raw_directions[metric]
        significant = adjusted[metric] < alpha
        passed     = not (degraded and significant)
        results[metric] = passed
        logger.debug(
            "Guardrail checked",
            extra={
                "metric": metric,
                "raw_p_value": round(raw_pvalues[metric], 6),
                "adjusted_p_value": round(adjusted[metric], 6),
                "is_degraded": degraded,
                "is_significant": significant,
                "status": "PASS" if passed else "FAIL",
            },
        )

    return results


def guardrail_summary(
    data: pd.DataFrame,
    variant_col: str,
    guardrail_metrics: List[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Return a DataFrame with per-metric statistics and pass/fail status.

    Returns
    -------
    pd.DataFrame with columns:
        metric, ctrl_mean, trt_mean, delta, pct_change, p_value, status
    """
    control = data[data[variant_col] == "control"]
    treatment = data[data[variant_col] != "control"]

    rows = []
    for metric in guardrail_metrics:
        c = control[metric].dropna().values
        t = treatment[metric].dropna().values
        _, p = sp_stats.ttest_ind(t, c, equal_var=False)
        ctrl_mean = np.mean(c)
        trt_mean = np.mean(t)
        delta = trt_mean - ctrl_mean
        pct = delta / ctrl_mean * 100 if ctrl_mean != 0 else float("nan")
        rows.append({
            "metric": metric,
            "ctrl_mean": round(ctrl_mean, 4),
            "trt_mean": round(trt_mean, 4),
            "delta": round(delta, 4),
            "pct_change": round(pct, 2),
            "p_value": round(p, 4),
            "status": "FAIL" if (p < alpha and delta < 0) else "PASS",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_correction(
    pvalues: Dict[str, float],
    alpha: float,
    method: str,
) -> Dict[str, float]:
    """Return adjusted p-values using the chosen correction method."""
    if method == "none":
        return pvalues

    metrics = list(pvalues.keys())
    raw = np.array([pvalues[m] for m in metrics])

    if method == "bonferroni":
        adjusted = np.minimum(raw * len(raw), 1.0)
    elif method == "holm":
        # Holm-Bonferroni step-down procedure
        order = np.argsort(raw)
        adjusted = np.zeros_like(raw)
        for rank, idx in enumerate(order):
            adj = raw[idx] * (len(raw) - rank)
            adjusted[idx] = min(adj, 1.0)
        # Ensure monotonicity
        for i in range(1, len(order)):
            adjusted[order[i]] = max(adjusted[order[i]], adjusted[order[i - 1]])
    else:
        raise ValueError(f"Unknown correction method: {method!r}")

    return dict(zip(metrics, adjusted.tolist()))
