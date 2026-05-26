"""
power.py — Power Analysis & Sample Size Calculator
===================================================
Tools for experiment planning: determine how many observations are needed to
detect a given effect, or what effect is detectable given a sample size.

Key Formulas
------------
For a two-sample z-test (or t-test asymptotically), the required per-group
sample size to detect an absolute effect δ with power (1-β) at significance α:

    n = (z_{α/2} + z_β)² * 2σ² / δ²

where z_{α/2} and z_β are the quantiles of the standard normal distribution.

The minimum detectable effect (MDE) for a given n is the inverse:

    MDE = (z_{α/2} + z_β) * σ * sqrt(2 / n)

Reference
---------
Cohen, J. (1988). Statistical power analysis for the behavioral sciences
(2nd ed.). Lawrence Erlbaum Associates.

Kohavi, R., Longbotham, R., Sommerfield, D., & Henne, R. L. (2009).
Controlled experiments on the web. Data Mining and Knowledge Discovery.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def required_sample_size(
    effect_size: float,
    std: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> int:
    """Compute the minimum per-group sample size for a two-sample t-test.

    Parameters
    ----------
    effect_size : float
        Absolute difference in means to detect (δ = μ_t - μ_c).
    std : float
        Pooled standard deviation of the metric.
    alpha : float
        Significance level (Type I error rate).
    power : float
        Desired statistical power (1 - Type II error).
    two_sided : bool
        Use a two-sided test (default True).

    Returns
    -------
    int
        Required observations *per group*.  Round up to the nearest integer.

    Raises
    ------
    ValueError
        If effect_size or std are non-positive.

    Examples
    --------
    >>> required_sample_size(0.5, 2.0)
    252
    """
    if effect_size <= 0:
        raise ValueError("effect_size must be positive.")
    if std <= 0:
        raise ValueError("std must be positive.")

    alpha_adj = alpha / 2 if two_sided else alpha
    z_alpha = sp_stats.norm.ppf(1 - alpha_adj)
    z_beta = sp_stats.norm.ppf(power)

    n = (z_alpha + z_beta) ** 2 * 2 * std ** 2 / effect_size ** 2
    return int(np.ceil(n))


def minimum_detectable_effect(
    n: int,
    std: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> float:
    """Compute the MDE given a per-group sample size.

    Parameters
    ----------
    n : int
        Per-group sample size.
    std : float
        Pooled standard deviation of the metric.
    alpha : float
        Significance level.
    power : float
        Desired statistical power.
    two_sided : bool
        Use a two-sided test (default True).

    Returns
    -------
    float
        Minimum detectable absolute effect size.

    Examples
    --------
    >>> minimum_detectable_effect(1000, 2.0)
    0.3934...
    """
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if std <= 0:
        raise ValueError("std must be positive.")

    alpha_adj = alpha / 2 if two_sided else alpha
    z_alpha = sp_stats.norm.ppf(1 - alpha_adj)
    z_beta = sp_stats.norm.ppf(power)

    return (z_alpha + z_beta) * std * np.sqrt(2.0 / n)


def relative_mde(
    n: int,
    baseline_mean: float,
    std: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Return the MDE as a percentage of the baseline mean.

    Parameters
    ----------
    n : int
        Per-group sample size.
    baseline_mean : float
        Control group mean.
    std : float
        Pooled standard deviation.
    alpha : float
        Significance level.
    power : float
        Desired statistical power.

    Returns
    -------
    float
        Relative MDE as a fraction (e.g. 0.05 means 5% lift detectable).
    """
    abs_mde = minimum_detectable_effect(n, std, alpha=alpha, power=power)
    if baseline_mean == 0:
        return float("inf")
    return abs_mde / abs(baseline_mean)


def power_curve(
    effect_sizes: list,
    n: int,
    std: float,
    alpha: float = 0.05,
) -> list:
    """Compute achieved power across a range of effect sizes.

    Parameters
    ----------
    effect_sizes : list of float
        Absolute effect sizes to evaluate.
    n : int
        Per-group sample size.
    std : float
        Pooled standard deviation.
    alpha : float
        Significance level.

    Returns
    -------
    list of float
        Power for each effect size.
    """
    se = std * np.sqrt(2.0 / n)
    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
    powers = []
    for delta in effect_sizes:
        nc = abs(delta) / se  # non-centrality parameter
        power = 1 - sp_stats.norm.cdf(z_alpha - nc) + sp_stats.norm.cdf(-z_alpha - nc)
        powers.append(float(power))
    return powers
