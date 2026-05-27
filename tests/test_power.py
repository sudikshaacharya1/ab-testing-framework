"""
Tests for power analysis utilities (src/stats/power.py).
"""

import numpy as np
import pytest
from abkit.stats.power import (
    required_sample_size,
    minimum_detectable_effect,
    relative_mde,
    power_curve,
)


class TestRequiredSampleSize:
    """Tests for the required_sample_size function."""

    def test_textbook_example(self):
        """Cohen (1988): δ=0.5, σ=2, α=0.05, power=0.80 → n≈252."""
        n = required_sample_size(0.5, 2.0, alpha=0.05, power=0.80)
        assert 245 <= n <= 260  # allow small numerical differences

    def test_larger_effect_needs_fewer_samples(self):
        n_small = required_sample_size(0.2, 2.0)
        n_large = required_sample_size(1.0, 2.0)
        assert n_small > n_large

    def test_higher_power_needs_more_samples(self):
        n_80 = required_sample_size(0.5, 1.0, power=0.80)
        n_95 = required_sample_size(0.5, 1.0, power=0.95)
        assert n_95 > n_80

    def test_stricter_alpha_needs_more_samples(self):
        n_05 = required_sample_size(0.5, 1.0, alpha=0.05)
        n_01 = required_sample_size(0.5, 1.0, alpha=0.01)
        assert n_01 > n_05

    def test_one_sided_needs_fewer(self):
        n_two = required_sample_size(0.5, 1.0, two_sided=True)
        n_one = required_sample_size(0.5, 1.0, two_sided=False)
        assert n_one < n_two

    def test_returns_integer(self):
        n = required_sample_size(0.5, 1.0)
        assert isinstance(n, int)

    def test_invalid_effect_size_raises(self):
        with pytest.raises(ValueError, match="effect_size"):
            required_sample_size(-1.0, 1.0)

    def test_invalid_std_raises(self):
        with pytest.raises(ValueError, match="std"):
            required_sample_size(0.5, 0.0)


class TestMinimumDetectableEffect:
    """Tests for the minimum_detectable_effect function."""

    def test_inverse_of_sample_size(self):
        """MDE and required_sample_size should be inverses of each other."""
        delta = 0.5
        std = 2.0
        n = required_sample_size(delta, std, alpha=0.05, power=0.80)
        mde = minimum_detectable_effect(n, std, alpha=0.05, power=0.80)
        assert abs(mde - delta) < 0.02  # within 2 units

    def test_larger_n_smaller_mde(self):
        mde_100 = minimum_detectable_effect(100, 1.0)
        mde_1000 = minimum_detectable_effect(1000, 1.0)
        assert mde_100 > mde_1000

    def test_larger_std_larger_mde(self):
        mde_small_std = minimum_detectable_effect(500, 1.0)
        mde_large_std = minimum_detectable_effect(500, 3.0)
        assert mde_large_std > mde_small_std

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="n must be"):
            minimum_detectable_effect(0, 1.0)

    def test_invalid_std_raises(self):
        with pytest.raises(ValueError, match="std must be"):
            minimum_detectable_effect(100, -1.0)


class TestRelativeMde:
    """Tests for the relative_mde convenience function."""

    def test_output_is_fraction(self):
        r = relative_mde(1000, baseline_mean=10.0, std=2.0)
        assert 0 < r < 1

    def test_zero_baseline_returns_inf(self):
        r = relative_mde(1000, baseline_mean=0.0, std=2.0)
        assert r == float("inf")


class TestPowerCurve:
    """Tests for the power_curve function."""

    def test_returns_correct_length(self):
        effects = [0.1, 0.2, 0.3, 0.4, 0.5]
        powers = power_curve(effects, n=500, std=1.0)
        assert len(powers) == len(effects)

    def test_power_increases_with_effect(self):
        # Use n=100 so power doesn't saturate to 1.0 for all entries,
        # keeping the strict monotonicity assertion meaningful.
        effects = [0.1, 0.3, 0.5, 0.8]
        powers = power_curve(effects, n=100, std=1.0)
        assert all(powers[i] < powers[i + 1] for i in range(len(powers) - 1))

    def test_power_in_unit_interval(self):
        effects = [0.01, 0.1, 0.5, 2.0]
        powers = power_curve(effects, n=200, std=1.0)
        assert all(0.0 <= p <= 1.0 for p in powers)

    def test_very_large_effect_approaches_one(self):
        powers = power_curve([10.0], n=100, std=1.0)
        assert powers[0] > 0.99
