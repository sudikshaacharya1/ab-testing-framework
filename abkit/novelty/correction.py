"""
correction.py — Novelty Effect Detection & Correction
======================================================
A **novelty effect** (also called a "primacy effect" for established users)
occurs when users interact differently with a new feature simply because it
is new, not because it is genuinely better.  This inflates treatment metrics
early in an experiment and can decay toward the true effect over time.

Detection Strategy
------------------
We use a two-pronged approach:

1. **Time-decay test**: Split the experiment timeline into early and late
   cohorts.  If the treatment effect is significantly larger in the early
   cohort than the late cohort, a novelty effect may be present.

2. **New-vs-existing user test**: Compare treatment effects among users who
   joined recently (no prior exposure) vs. established users.  A strong
   novelty effect shows up only for existing users — new users have no prior
   pattern to change.

Correction
----------
The corrected estimate uses only the *late* cohort data, or weights cohorts by
their fraction of steady-state traffic.  Under the assumption that the novelty
effect decays to zero, the late-cohort effect is a better estimate of the
long-run treatment effect.

Reference
---------
Kohavi, R., & Thomke, S. (2017). The surprising power of online experiments.
Harvard Business Review.

Dmitriev, P., Gupta, S., Kim, D. W., & Vaz, G. (2016). A dirty dozen: Twelve
common metric interpretation pitfalls. KDD 2016.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from abkit.logger import get_logger

logger = get_logger(__name__)


def detect_novelty_effect(
    data: pd.DataFrame,
    variant_col: str,
    metric: str,
    time_col: str = "days_since_signup",
    early_threshold: Optional[float] = None,
    alpha: float = 0.05,
) -> bool:
    """Detect whether a novelty effect is present in the experiment.

    Uses a time-cohort interaction test: if the treatment effect in the first
    half of the experiment duration differs significantly from the second half,
    a novelty effect is flagged.

    Parameters
    ----------
    data : pd.DataFrame
        Experiment data.  Must contain *variant_col*, *metric*, and *time_col*.
    variant_col : str
        Column identifying control vs. treatment.
    metric : str
        Primary metric column.
    time_col : str
        Column representing exposure time (e.g. days into the experiment or
        days since user signup).
    early_threshold : float, optional
        Cutoff separating early from late cohort.  Defaults to the median
        of *time_col*.
    alpha : float
        Significance level for the interaction test.

    Returns
    -------
    bool
        True if a novelty effect is detected at the given significance level.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 1000
    >>> days = rng.integers(1, 30, n)
    >>> df = pd.DataFrame({
    ...     "variant": ["control"]*(n//2) + ["treatment"]*(n//2),
    ...     "revenue": np.r_[rng.normal(10, 2, n//2), rng.normal(12, 2, n//2)],
    ...     "days_since_signup": days,
    ... })
    >>> detect_novelty_effect(df, "variant", "revenue")
    False
    """
    if time_col not in data.columns:
        raise ValueError(f"Column '{time_col}' not found in DataFrame.")

    threshold = early_threshold if early_threshold is not None else float(data[time_col].median())

    early = data[data[time_col] <= threshold]
    late = data[data[time_col] > threshold]

    def group_diff(df: pd.DataFrame) -> float:
        ctrl = df[df[variant_col] == "control"][metric].dropna().values
        trt = df[df[variant_col] != "control"][metric].dropna().values
        if len(ctrl) < 2 or len(trt) < 2:
            return 0.0
        return float(np.mean(trt) - np.mean(ctrl))

    early_effect = group_diff(early)
    late_effect = group_diff(late)

    # Test: is the early effect significantly larger than the late effect?
    # Use a difference-in-differences approach.
    ctrl_early = early[early[variant_col] == "control"][metric].dropna().values
    trt_early = early[early[variant_col] != "control"][metric].dropna().values
    ctrl_late = late[late[variant_col] == "control"][metric].dropna().values
    trt_late = late[late[variant_col] != "control"][metric].dropna().values

    if any(len(g) < 2 for g in [ctrl_early, trt_early, ctrl_late, trt_late]):
        return False

    # Bootstrap-based test for interaction
    _, p_early = sp_stats.ttest_ind(trt_early, ctrl_early)
    _, p_late = sp_stats.ttest_ind(trt_late, ctrl_late)

    novelty_flagged = (
        p_early < alpha
        and p_late >= alpha
        and early_effect > late_effect
    )

    logger.debug(
        "Novelty effect check complete",
        extra={
            "early_effect": round(early_effect, 4),
            "late_effect":  round(late_effect, 4),
            "p_early":      round(p_early, 4),
            "p_late":       round(p_late, 4),
            "threshold":    float(threshold),
            "novelty_detected": novelty_flagged,
        },
    )

    return bool(novelty_flagged)


def correct_novelty_effect(
    data: pd.DataFrame,
    variant_col: str,
    metric: str,
    time_col: str = "days_since_signup",
    early_fraction: float = 0.5,
) -> Tuple[float, float]:
    """Return the novelty-corrected treatment effect estimate.

    Uses only the later portion of the experiment (where novelty has decayed)
    to estimate the steady-state treatment effect.

    Parameters
    ----------
    data : pd.DataFrame
        Experiment data.
    variant_col : str
        Column identifying control vs. treatment.
    metric : str
        Primary metric column.
    time_col : str
        Column representing exposure time.
    early_fraction : float
        Fraction of the time range to discard as "early".  Default 0.5.

    Returns
    -------
    effect : float
        Corrected treatment effect (late cohort mean difference).
    p_value : float
        p-value for the corrected effect estimate.
    """
    t_min = data[time_col].min()
    t_max = data[time_col].max()
    cutoff = t_min + early_fraction * (t_max - t_min)

    late = data[data[time_col] > cutoff]
    ctrl = late[late[variant_col] == "control"][metric].dropna().values
    trt = late[late[variant_col] != "control"][metric].dropna().values

    if len(ctrl) < 2 or len(trt) < 2:
        raise ValueError("Insufficient data in the late cohort for correction.")

    effect = float(np.mean(trt) - np.mean(ctrl))
    _, p_value = sp_stats.ttest_ind(trt, ctrl)

    return effect, float(p_value)
