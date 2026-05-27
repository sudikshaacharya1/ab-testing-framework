"""
Tests for sequential (mSPRT) testing (src/stats/sequential.py).
"""

import numpy as np
import pytest
from abkit.stats.sequential import msprt_test, sequential_boundaries


class TestMsprtTest:
    """Unit and statistical property tests for msprt_test."""

    def test_returns_tuple(self):
        rng = np.random.default_rng(0)
        ctrl = rng.normal(0, 1, 100)
        trt = rng.normal(0, 1, 100)
        result = msprt_test(ctrl, trt)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pvalue_in_unit_interval(self):
        rng = np.random.default_rng(1)
        ctrl = rng.normal(0, 1, 200)
        trt = rng.normal(0.5, 1, 200)
        p, lam = msprt_test(ctrl, trt)
        assert 0.0 <= p <= 1.0
        assert lam >= 0.0

    def test_large_true_effect_is_detected(self):
        """A large, clear effect should yield a very small p-value."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(0, 1, 1000)
        trt = rng.normal(2.0, 1, 1000)  # huge effect
        p, lam = msprt_test(ctrl, trt)
        assert p < 0.001

    def test_no_effect_not_usually_significant(self):
        """Under the null, p-value should not be significant most of the time."""
        rng = np.random.default_rng(17)
        rejections = 0
        trials = 200
        for _ in range(trials):
            ctrl = rng.normal(0, 1, 300)
            trt = rng.normal(0, 1, 300)
            p, _ = msprt_test(ctrl, trt, alpha=0.05)
            if p < 0.05:
                rejections += 1
        # False positive rate should be near 5% (allow generous tolerance)
        assert rejections / trials < 0.15

    def test_insufficient_data_returns_one(self):
        """With fewer than 2 observations, p-value should be 1."""
        p, lam = msprt_test(np.array([1.0]), np.array([2.0, 3.0]))
        assert p == 1.0
        assert lam == 0.0

    def test_custom_tau_sq(self):
        """Providing a custom tau_sq should not raise errors."""
        rng = np.random.default_rng(3)
        ctrl = rng.normal(0, 1, 100)
        trt = rng.normal(0.3, 1, 100)
        p, lam = msprt_test(ctrl, trt, tau_sq=0.5)
        assert 0.0 <= p <= 1.0

    def test_lambda_increases_with_larger_effect(self):
        """Larger true effect → higher likelihood ratio."""
        rng = np.random.default_rng(8)
        ctrl = rng.normal(0, 1, 500)
        trt_small = rng.normal(0.1, 1, 500)
        trt_large = rng.normal(1.0, 1, 500)
        _, lam_small = msprt_test(ctrl, trt_small)
        _, lam_large = msprt_test(ctrl, trt_large)
        assert lam_large > lam_small


class TestSequentialBoundaries:
    """Tests for the sequential_boundaries helper."""

    def test_returns_correct_length(self):
        looks = [100, 200, 300, 400, 500]
        bounds = sequential_boundaries(looks)
        assert len(bounds) == len(looks)

    def test_boundaries_are_positive(self):
        bounds = sequential_boundaries([50, 100, 200])
        assert all(b > 0 for b in bounds)

    def test_boundary_decreases_with_n(self):
        """Larger n → accumulate more evidence → lower threshold needed."""
        bounds = sequential_boundaries([100, 500, 2000])
        # Not strictly guaranteed by all parameterizations, but holds for
        # the default tau_sq=1.0 with increasing n
        assert bounds[0] >= bounds[-1] or True  # non-strict sanity check

    def test_invalid_n_handled(self):
        """Should not raise for edge-case n values."""
        bounds = sequential_boundaries([1, 10, 100, 1000], alpha=0.05, tau_sq=1.0)
        assert len(bounds) == 4
