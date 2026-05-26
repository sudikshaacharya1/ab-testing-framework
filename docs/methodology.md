# Statistical Methodology

This document explains every statistical technique implemented in this framework,
including the assumptions, mathematical derivations, and implementation details.

---

## Table of Contents

1. [Classical A/B Testing Foundations](#1-classical-ab-testing-foundations)
2. [CUPED Variance Reduction](#2-cuped-variance-reduction)
3. [Sequential Testing (mSPRT)](#3-sequential-testing-msprt)
4. [Power Analysis](#4-power-analysis)
5. [Guardrail Metrics](#5-guardrail-metrics)
6. [Novelty Effect Detection](#6-novelty-effect-detection)
7. [Putting It All Together](#7-putting-it-all-together)

---

## 1. Classical A/B Testing Foundations

### The Hypothesis Testing Framework

A standard A/B experiment tests:

- **H₀ (null)**: μ_treatment = μ_control  (no effect)
- **H₁ (alternative)**: μ_treatment ≠ μ_control  (two-sided)

The **two-sample Welch's t-test** is used by default (it does not assume equal variances):

```
t = (ȳ_t - ȳ_c) / sqrt(s²_t/n_t + s²_c/n_c)
```

We reject H₀ when |t| > t_{α/2, ν}, where ν is the Welch-Satterthwaite degrees of freedom.

### Type I and Type II Errors

| Decision \ Truth | H₀ true | H₀ false |
|---|---|---|
| Reject H₀ | **Type I error (α)** | Correct (power) |
| Fail to reject | Correct | **Type II error (β)** |

- **α = 0.05**: 5% chance of detecting a non-existent effect
- **Power = 1 - β = 0.80**: 80% chance of detecting a real effect

---

## 2. CUPED Variance Reduction

### Motivation

Even with equal group sizes, the variance of `ȳ_t - ȳ_c` is `2σ²/n`. If we can reduce σ², 
we need fewer observations to detect the same effect.

### The Estimator

Let Y be the post-experiment metric and X be a pre-experiment covariate.  
Define the CUPED-adjusted outcome:

```
Y* = Y - θ(X - E[X])
```

where `θ = Cov(Y, X) / Var(X)` is estimated via OLS on pooled data.

**Key properties:**

1. **Unbiasedness**: `E[Y*_t - Y*_c] = E[Y_t - Y_c]` — the treatment effect is preserved.
2. **Variance reduction**: `Var(Y*) = Var(Y)(1 - ρ²)` where `ρ = Corr(Y, X)`.
3. **Independence requirement**: X must be independent of treatment assignment
   (guaranteed by pre-randomization measurement).

### Sample Size Savings

| ρ | Variance Reduction | Sample Size Reduction |
|---|---|---|
| 0.3 | 9% | 9% |
| 0.5 | 25% | 25% |
| 0.7 | 51% | 51% |
| 0.9 | 81% | 81% |

### Best Practices

- Use the **same metric from a prior period** as the covariate (highest ρ).
- Pool both groups to estimate θ — never estimate separately.
- Normalize X if it has outliers to prevent θ from being distorted.

**Reference**: Deng et al. (2013). *Improving the sensitivity of online controlled experiments by utilizing pre-experiment data.* WSDM.

---

## 3. Sequential Testing (mSPRT)

### The Peeking Problem

Checking a standard t-test p-value at n=100, 200, 300... inflates Type I error:

| Number of peeks | Effective FPR (α=0.05) |
|---|---|
| 1 | 5.0% |
| 5 | ~14% |
| 20 | ~25% |
| continuous | ~100% |

### The mSPRT Solution

The mixture Sequential Probability Ratio Test defines a **test martingale** Λ_t that satisfies:

```
P(∃t : Λ_t ≥ 1/α | H₀) ≤ α
```

This guarantees that the probability of ever exceeding the threshold under the null is bounded by α,
regardless of how many times you peek.

### The Statistic

For a two-sample normal test with a Gaussian prior δ ~ N(0, τ²) on the effect size:

```
Λ_t = sqrt(σ²/(σ² + n·τ²)) · exp(n·τ²·z²_t / (2(σ² + n·τ²)))
```

where z_t is the current z-statistic and n is the effective sample size.

The **always-valid p-value** is:

```
p_t = min(1, 1/Λ_t)
```

### Choosing τ²

- **τ² = σ²/n₀**: prior centered on detecting a one-standard-error effect by the planned sample size n₀.
- Larger τ² → more sensitive to large effects, slower for small effects.
- Default: `τ² = pooled_var / max(n_c, n_t)`.

### When to Use mSPRT

- When experiments need to ship quickly and waiting for the planned sample size is costly.
- When experiments have unpredictable traffic and need a flexible stopping rule.
- **Not** when the experiment has a hard fixed deadline (use classical testing then).

**Reference**: Johari et al. (2017). *Peeking at A/B tests: Why it matters and what to do about it.* KDD.

---

## 4. Power Analysis

### Required Sample Size

For a two-sided two-sample test detecting absolute effect δ with pooled std σ:

```
n = (z_{α/2} + z_β)² · 2σ² / δ²
```

Where:
- `z_{0.025} = 1.96` for α = 0.05
- `z_{0.20} = 0.842` for 80% power
- So `n ≈ (1.96 + 0.84)² · 2σ² / δ² = 15.68 σ²/δ²`

### Minimum Detectable Effect (MDE)

The MDE is the smallest effect that can be detected at the specified power:

```
MDE = (z_{α/2} + z_β) · σ · sqrt(2/n)
```

### Power Curve

Power is a function of the true effect size δ and the sample size n:

```
Power(δ) = 1 - Φ(z_{α/2} - δ/SE) + Φ(-z_{α/2} - δ/SE)
```

where `SE = σ·sqrt(2/n)` and Φ is the standard normal CDF.

### Practical Guidelines

- Target **80% power** for standard experiments; 90% for high-stakes decisions.
- Use **relative MDE** (MDE / baseline_mean) to communicate in business terms.
- When using CUPED, substitute `σ_cuped = σ·sqrt(1 - ρ²)` into the formula.

---

## 5. Guardrail Metrics

### Purpose

Guardrail metrics protect against unintended side-effects. An experiment can declare success 
on the primary metric only if **all guardrail metrics pass**.

### Statistical Test

Each guardrail metric is tested independently using Welch's t-test.  
A guardrail **fails** when:

1. The treatment mean is statistically significantly *worse* than control, AND  
2. The degradation is directionally consistent with harm.

Improvements in guardrail metrics do not cause failure.

### Multiple Testing Correction

When K guardrail metrics are tested simultaneously, the family-wise error rate (FWER)
under the null is approximately `1 - (1-α)^K`. For K=10, this is ~40% instead of 5%.

We use the **Holm-Bonferroni step-down procedure**:

1. Sort p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(K)
2. Reject H₀_(i) if p_(i) ≤ α / (K - i + 1)
3. Stop at first failure to reject; do not reject remaining

Holm is uniformly more powerful than Bonferroni while controlling FWER at α.

### Recommended Guardrails by Product Area

| Area | Guardrail Metrics |
|---|---|
| E-commerce | Cart abandonment rate, support contact rate |
| Content | Pages/session, bounce rate |
| Subscription | Churn rate, cancellation rate |
| Infrastructure | Latency (p50, p95), error rate |

---

## 6. Novelty Effect Detection

### What Is the Novelty Effect?

Users exposed to a new feature may engage with it more initially simply because it is *new*,
not because it is genuinely better.  This inflates treatment metrics early and decays over time.

The symmetric problem — **primacy effect** — occurs when existing users initially resist a UI
change, depressing metrics early before adapting.

### Detection Strategy

**Time-cohort test**: Split the experiment timeline into early and late cohorts.
If the treatment effect is significantly positive in the early cohort but non-significant (or
smaller) in the late cohort, a novelty effect is flagged.

**New-vs-existing user test**: Novelty effects manifest for existing users (who have a prior
pattern), not for new users (who have no prior expectation). If existing users drive all of the
treatment effect and new users show no effect, novelty is likely.

### Correction

The corrected estimate uses only the **late cohort** (after the novelty has decayed):

```
effect_corrected = ȳ_t(late) - ȳ_c(late)
```

This is a better estimate of the **long-run steady-state treatment effect**.

### Practical Guidance

- Run experiments for at least **2–4 weeks** to let novelty decay.
- If novelty is detected, extend the experiment duration before making a ship decision.
- New user cohorts are less susceptible; weight them appropriately for long-run projections.

---

## 7. Putting It All Together

### Recommended Experiment Analysis Checklist

```
☐ 1. Power analysis before launch (set n, MDE, α)
☐ 2. Sanity check: randomization unit balance, SRM test
☐ 3. If CUPED available: apply with pre-experiment covariate
☐ 4. If monitoring in real-time: use mSPRT, not standard t-test
☐ 5. Check all guardrail metrics (with Holm correction)
☐ 6. Check for novelty effect if experiment ran < 2 weeks
☐ 7. Report: p-value, effect size, CI, MDE, power
```

### Decision Matrix

| Primary sig. | Guardrails | Decision |
|---|---|---|
| ✅ Yes | ✅ All pass | 🚀 Ship |
| ✅ Yes | 🚨 Any fail | 🔧 Fix guardrail metric first |
| ❌ No | — | ⏳ Extend or conclude no effect |
| ❌ No (novelty) | — | ⚠️ Extend experiment; re-evaluate |
