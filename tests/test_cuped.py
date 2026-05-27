"""
Tests for CUPED variance reduction (src/stats/cuped.py).
"""

import numpy as np
import pytest
from abkit.stats.cuped import cuped_adjust, variance_reduction_factor, _estimate_theta


class TestEstimateTheta:
    """Unit tests for the OLS theta estimator."""

    def test_perfect_correlation(self):
        """θ should equal 1 when Y = X exactly."""
        x = np.arange(1.0, 101.0)
        y = x.copy()
        theta = _estimate_theta(y, x)
        assert abs(theta - 1.0) < 1e-10

    def test_zero_variance(self):
        """θ should be 0 when X is constant (no variance to exploit)."""
        x = np.ones(100)
        y = np.random.default_rng(0).normal(0, 1, 100)
        theta = _estimate_theta(y, x)
        assert theta == 0.0

    def test_no_correlation(self):
        """θ should be near 0 when Y and X are independent."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 10_000)
        y = rng.normal(0, 1, 10_000)  # independent
        theta = _estimate_theta(y, x)
        assert abs(theta) < 0.05  # should be close to 0


class TestCupedAdjust:
    """Integration tests for cuped_adjust."""

    def setup_method(self):
        rng = np.random.default_rng(99)
        n = 2000
        # Pre-experiment covariate
        self.x_ctrl = rng.normal(10, 2, n)
        self.x_trt = rng.normal(10, 2, n)
        # Post-experiment metric correlated with covariate
        noise = rng.normal(0, 0.5, n)
        self.y_ctrl = self.x_ctrl + noise
        self.y_trt = self.x_trt + rng.normal(0.5, 0.5, n)  # true effect of 0.5

    def test_returns_correct_shapes(self):
        y_c, y_t = cuped_adjust(self.y_ctrl, self.y_trt, self.x_ctrl, self.x_trt)
        assert y_c.shape == self.y_ctrl.shape
        assert y_t.shape == self.y_trt.shape

    def test_variance_is_reduced(self):
        """After CUPED, variance should be strictly lower than original."""
        y_c, y_t = cuped_adjust(self.y_ctrl, self.y_trt, self.x_ctrl, self.x_trt)
        assert np.var(y_c) < np.var(self.y_ctrl)
        assert np.var(y_t) < np.var(self.y_trt)

    def test_effect_preserved(self):
        """The mean difference should be approximately preserved.

        CUPED preserves the *expected* treatment effect. In finite samples,
        the adjusted estimator is still unbiased, but the sample estimate
        shifts by θ*(x̄_t - x̄_c).  With n=2000 and θ≈1 we allow a slack
        of 3σ ≈ 3 * sqrt(2 * Var(X)/n) ≈ 0.17.
        """
        orig_effect = np.mean(self.y_trt) - np.mean(self.y_ctrl)
        y_c, y_t = cuped_adjust(self.y_ctrl, self.y_trt, self.x_ctrl, self.x_trt)
        cuped_effect = np.mean(y_t) - np.mean(y_c)
        # Both estimators target the same true effect; allow for finite-sample slack
        assert abs(cuped_effect - orig_effect) < 0.20

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            cuped_adjust(
                self.y_ctrl[:100], self.y_trt,
                self.x_ctrl[:50], self.x_trt,
            )

    def test_no_correlation_no_change(self):
        """When covariate is noise, variance should barely change."""
        rng = np.random.default_rng(7)
        n = 1000
        y_c = rng.normal(10, 2, n)
        y_t = rng.normal(10.5, 2, n)
        x_c = rng.normal(0, 1, n)  # unrelated
        x_t = rng.normal(0, 1, n)

        adj_c, adj_t = cuped_adjust(y_c, y_t, x_c, x_t)
        # Variance reduction should be tiny (< 5%)
        reduction = 1 - np.var(adj_c) / np.var(y_c)
        assert abs(reduction) < 0.05


class TestVarianceReductionFactor:
    """Tests for the variance_reduction_factor utility."""

    def test_perfect_correlation(self):
        """With ρ=1, variance reduction factor = 0 (100% reduction)."""
        x = np.arange(1.0, 101.0)
        vrf = variance_reduction_factor(x, x)
        assert abs(vrf) < 1e-10

    def test_zero_correlation(self):
        """With ρ=0, variance reduction factor = 1 (no reduction)."""
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 10_000)
        x = rng.normal(0, 1, 10_000)
        vrf = variance_reduction_factor(y, x)
        assert abs(vrf - 1.0) < 0.05

    def test_moderate_correlation(self):
        """With ρ≈0.7, VRF ≈ 0.51."""
        rng = np.random.default_rng(5)
        x = rng.normal(0, 1, 100_000)
        y = 0.7 * x + np.sqrt(1 - 0.49) * rng.normal(0, 1, 100_000)
        vrf = variance_reduction_factor(y, x)
        assert 0.45 < vrf < 0.57
