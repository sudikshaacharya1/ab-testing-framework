"""
sequential.py — Sequential Testing with mSPRT
==============================================
The mixture Sequential Probability Ratio Test (mSPRT) allows continuous
monitoring of experiment results without inflating Type I error (the "peeking
problem" of naive repeated testing).

Theory
------
Standard hypothesis testing assumes a fixed sample size decided upfront.
Peeking at results early and stopping when p < α inflates the false positive
rate far above α.

mSPRT derives a test statistic — the *e-value* or *mixture likelihood ratio*
— that is a valid test martingale.  At any stopping time τ, the probability of
ever exceeding threshold 1/α under H₀ is bounded by α:

    P(∃t : Λ_t ≥ 1/α | H₀) ≤ α

For a two-sample normal test with unknown variance, the mixture is taken over
a Gaussian prior on the effect size δ ~ N(0, τ²).  Choosing τ² = σ²/n_0
(where n_0 is the expected final sample size) gives a well-calibrated test.

The always-valid p-value is:

    p_t = min(1, 1 / Λ_t)

Reference
---------
Johari, R., Koomen, P., Pekelis, L., & Walsh, D. (2017). Peeking at A/B tests:
Why it matters and what to do about it. KDD 2017.
https://doi.org/10.1145/3097983.3097992

Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2021).
Time-uniform, nonparametric, nonasymptotic confidence sequences.
The Annals of Statistics. https://doi.org/10.1214/20-AOS1991
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import stats as sp_stats

from abkit.logger import get_logger

logger = get_logger(__name__)


def msprt_test(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    tau_sq: float | None = None,
) -> Tuple[float, float]:
    """Run the mSPRT sequential test on accumulated observations.

    Parameters
    ----------
    control : np.ndarray
        All control observations collected so far.
    treatment : np.ndarray
        All treatment observations collected so far.
    alpha : float
        Significance level.  The test rejects when p-value < alpha.
    tau_sq : float, optional
        Prior variance τ² for the Gaussian mixture.  Defaults to
        pooled_var / max(n_c, n_t), which corresponds to expecting an
        effect of roughly one standard error.

    Returns
    -------
    p_value : float
        Always-valid p-value (safe to check at any time).
    lambda_t : float
        Current mixture likelihood ratio (reject when >= 1/alpha).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ctrl = rng.normal(0, 1, 500)
    >>> trt  = rng.normal(0.3, 1, 500)  # true effect of 0.3
    >>> p, lam = msprt_test(ctrl, trt)
    >>> p < 0.05
    True
    """
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)

    n_c, n_t = len(control), len(treatment)
    if n_c < 2 or n_t < 2:
        return 1.0, 0.0

    # Pooled variance estimate
    var_c = np.var(control, ddof=1)
    var_t = np.var(treatment, ddof=1)
    pooled_var = ((n_c - 1) * var_c + (n_t - 1) * var_t) / (n_c + n_t - 2)

    if tau_sq is None:
        tau_sq = pooled_var / max(n_c, n_t)

    # Standard error of the mean difference
    se_sq = pooled_var * (1.0 / n_c + 1.0 / n_t)
    se = np.sqrt(se_sq)

    if se == 0:
        return 1.0, 0.0

    delta_hat = np.mean(treatment) - np.mean(control)
    t_stat = delta_hat / se

    # mSPRT mixture likelihood ratio (Gaussian prior)
    # Λ_t = sqrt(σ²/(σ² + n*τ²)) * exp(n*τ²*z²/(2*(σ² + n*τ²)))
    # Simplified for the two-sample case with effective n:
    n_eff = (n_c * n_t) / (n_c + n_t)  # harmonic mean / 2

    numerator_var = pooled_var / n_eff
    kappa = tau_sq / (numerator_var + tau_sq)

    lambda_t = float(np.sqrt(numerator_var / (numerator_var + tau_sq))
                     * np.exp(0.5 * kappa * t_stat ** 2))

    p_value = min(1.0, 1.0 / lambda_t) if lambda_t > 0 else 1.0

    logger.debug(
        "mSPRT computed",
        extra={
            "n_control": n_c,
            "n_treatment": n_t,
            "delta_hat": round(float(delta_hat), 6),
            "lambda_t": round(float(lambda_t), 4),
            "p_value": round(float(p_value), 6),
        },
    )

    return p_value, lambda_t


def sequential_boundaries(
    n_observations: List[int],
    alpha: float = 0.05,
    tau_sq: float = 1.0,
) -> List[float]:
    """Compute the mSPRT rejection threshold expressed as a critical z-value
    at each interim look.

    Parameters
    ----------
    n_observations : list of int
        Cumulative sample sizes (per group) at each planned interim analysis.
    alpha : float
        Overall significance level.
    tau_sq : float
        Prior variance for the Gaussian mixture.

    Returns
    -------
    list of float
        Critical z-values.  Reject at look i if |z_i| >= boundary[i].
    """
    threshold = 1.0 / alpha
    boundaries = []
    for n in n_observations:
        # Invert the mixture ratio to find the critical z
        # threshold = sqrt(1/(1 + n*tau_sq)) * exp(n*tau_sq*z²/(2*(1+n*tau_sq)))
        # Solve numerically
        kappa = (n * tau_sq) / (1.0 + n * tau_sq)
        scale = np.sqrt(1.0 / (1.0 + n * tau_sq))
        # threshold / scale = exp(kappa * z^2 / 2)
        if threshold / scale <= 0:
            boundaries.append(float("inf"))
            continue
        z_crit = np.sqrt(2.0 * np.log(threshold / scale) / kappa) if kappa > 0 else float("inf")
        boundaries.append(float(z_crit))

    return boundaries
