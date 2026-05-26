"""
cuped.py — CUPED Variance Reduction
=====================================
CUPED (Controlled-experiment Using Pre-Experiment Data) reduces the variance
of the treatment effect estimator by regressing out the effect of a pre-
experiment covariate that is correlated with the primary metric but independent
of the treatment assignment.

Theory
------
Let Y be the post-experiment metric and X be the pre-experiment covariate.
The CUPED-adjusted outcome is:

    Y_cuped = Y - θ * (X - E[X])

where θ is estimated via OLS across both groups pooled:

    θ = Cov(Y, X) / Var(X)

Since E[X] is constant, it does not change the treatment effect estimate, only
the variance.  Under the assumption that X ⊥ treatment assignment:

    Var(Y_cuped) = Var(Y) * (1 - ρ²)

where ρ = Corr(Y, X).  A covariate with ρ = 0.5 halves the required sample
size; ρ = 0.7 reduces it by ~50 %.

Reference
---------
Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). Improving the sensitivity
of online controlled experiments by utilizing pre-experiment data.
WSDM 2013. https://doi.org/10.1145/2433396.2433413
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def cuped_adjust(
    control_y: np.ndarray,
    treatment_y: np.ndarray,
    control_x: np.ndarray,
    treatment_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply CUPED adjustment to control and treatment outcomes.

    Parameters
    ----------
    control_y : np.ndarray, shape (n_c,)
        Post-experiment metric values for the control group.
    treatment_y : np.ndarray, shape (n_t,)
        Post-experiment metric values for the treatment group.
    control_x : np.ndarray, shape (n_c,)
        Pre-experiment covariate values for the control group.
    treatment_x : np.ndarray, shape (n_t,)
        Pre-experiment covariate values for the treatment group.

    Returns
    -------
    control_y_adj : np.ndarray
        CUPED-adjusted metric for the control group.
    treatment_y_adj : np.ndarray
        CUPED-adjusted metric for the treatment group.

    Raises
    ------
    ValueError
        If arrays have mismatched lengths within each group.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(10, 2, 1000)
    >>> y = x + rng.normal(0, 1, 1000)      # strong correlation with x
    >>> ctrl_adj, trt_adj = cuped_adjust(y[:500], y[500:], x[:500], x[500:])
    >>> # Variance should be lower after adjustment
    >>> assert np.var(ctrl_adj) < np.var(y[:500])
    """
    control_y = np.asarray(control_y, dtype=float)
    treatment_y = np.asarray(treatment_y, dtype=float)
    control_x = np.asarray(control_x, dtype=float)
    treatment_x = np.asarray(treatment_x, dtype=float)

    if len(control_y) != len(control_x):
        raise ValueError("control_y and control_x must have the same length.")
    if len(treatment_y) != len(treatment_x):
        raise ValueError("treatment_y and treatment_x must have the same length.")

    # Pool both groups to estimate theta
    all_y = np.concatenate([control_y, treatment_y])
    all_x = np.concatenate([control_x, treatment_x])

    theta = _estimate_theta(all_y, all_x)
    x_mean = np.mean(all_x)

    control_y_adj = control_y - theta * (control_x - x_mean)
    treatment_y_adj = treatment_y - theta * (treatment_x - x_mean)

    return control_y_adj, treatment_y_adj


def _estimate_theta(y: np.ndarray, x: np.ndarray) -> float:
    """OLS estimate of the covariate coefficient: θ = Cov(Y, X) / Var(X)."""
    x_centered = x - np.mean(x)
    var_x = np.dot(x_centered, x_centered) / (len(x) - 1)
    if var_x == 0:
        return 0.0
    cov_yx = np.dot(y - np.mean(y), x_centered) / (len(y) - 1)
    return cov_yx / var_x


def variance_reduction_factor(y: np.ndarray, x: np.ndarray) -> float:
    """Compute the theoretical variance reduction factor (1 - ρ²).

    A value of 0.5 means CUPED cuts variance in half, doubling sensitivity.

    Parameters
    ----------
    y : np.ndarray
        Post-experiment metric (combined or single group).
    x : np.ndarray
        Pre-experiment covariate.

    Returns
    -------
    float
        Expected ratio Var(Y_cuped) / Var(Y), in [0, 1].
    """
    corr = float(np.corrcoef(y, x)[0, 1])
    return 1.0 - corr ** 2
