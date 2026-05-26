# A/B Testing Framework

> Production-grade experimentation infrastructure with statistical guardrails — built for teams that want to move fast without breaking metrics.

---

## The Business Case

Every product decision is a bet. Most companies make those bets blind — shipping features based on intuition, HiPPO opinions, or anecdotal user feedback. The ones that win long-term treat their product as a living experiment.

**The cost of bad experimentation is real:**

- A feature ships that *looks* like it lifts revenue — but the effect was noise. You've now permanently changed the product for worse.
- You run an experiment for 3 days, see p < 0.05, call it a win. In reality you peeked 12 times and your false positive rate was 40%, not 5%.
- You ship a UI change. Engagement goes up. But customer support contacts go up 18% too — and nobody checked.

This framework exists to close that gap. It gives any team the same statistical infrastructure that powers experimentation at companies like Netflix, Airbnb, Booking.com, and Microsoft — without needing a dedicated data science team to run every test.

---

## What It Does

| Problem | Solution | Module |
|---|---|---|
| "We don't have enough traffic to detect small effects" | CUPED variance reduction — boost sensitivity without more users | `src/stats/cuped.py` |
| "We need to ship fast — can we stop the test early?" | mSPRT sequential testing — peek anytime, FPR stays controlled | `src/stats/sequential.py` |
| "How many users do we actually need?" | Power analysis & MDE calculator | `src/stats/power.py` |
| "The primary metric is up but did we break anything?" | Automated guardrail metric checks with Holm-Bonferroni correction | `src/stats/guardrails.py` |
| "Users are just excited because it's new" | Novelty effect detection and late-cohort correction | `src/novelty/correction.py` |

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/your-username/ab-testing-framework.git
cd ab-testing-framework
pip install -r requirements.txt

# 2. Run on your data
python -m src.experiment \
  --data your_experiment.csv \
  --metric revenue \
  --covariate pre_revenue \
  --guardrails latency_ms crash_rate \
  --alpha 0.05 \
  --name "Checkout CTA Test"
```

### Example Output

```
============================================================
  Experiment: Checkout CTA Test
============================================================
  Metric            : revenue
  Control mean      : 9.8198
  Treatment mean    : 10.6566
  Absolute lift     : +0.8368
  Relative lift     : +8.52%
  p-value           : 0.0000
  Significant (α=0.05) : ✅ YES
  MDE (given n)     : 0.2094

  Guardrail Metrics:
    ✅ latency_ms: PASS
    ✅ crash_rate: PASS
============================================================
```

Your CSV just needs a `variant` column (`"control"` or `"treatment"`) plus whatever metric columns you care about.

---

## Features

### CUPED Variance Reduction
Pre-experiment data is the most underused asset in experimentation. If your users had revenue last month, that number is a strong predictor of revenue this month — before you ever ran a test. CUPED exploits that correlation to shrink metric variance, which means you can detect the same effect with fewer users. A covariate with ρ = 0.7 cuts required sample size by ~51%.

### Sequential Testing (mSPRT)
The "peeking problem" is one of the most common statistical sins in industry. Checking a standard t-test p-value repeatedly inflates your false positive rate from 5% to 30%+. The mixture Sequential Probability Ratio Test (mSPRT) gives you an *always-valid* p-value: check it at any point, as many times as you want, without losing Type I error control. Stop the experiment the moment you have enough evidence — no wasted traffic, no inflated FPR.

### Power Analysis
Never start an experiment without knowing whether you can actually detect the effect you care about. This module answers two questions before you launch:
- *"How many users do I need to detect a 5% lift with 80% power?"*
- *"Given my current traffic, what's the smallest effect I can reliably detect?"*

### Guardrail Metrics
Primary metrics going up means nothing if your app gets slower or support volume spikes. The guardrails module tests a configurable set of metrics for statistically significant degradation, with Holm-Bonferroni correction to handle multiple comparisons. An experiment only gets a green light when *all* guardrails pass.

### Novelty Effect Detection
New UI, new feature — users engage with it more because it's *different*, not because it's *better*. The novelty correction module detects this by comparing early vs. late cohort treatment effects. If early-period excitement is masking a null long-run effect, you'll know before you ship.

---

## Project Structure

```
ab-testing-framework/
├── src/
│   ├── experiment.py        # Core orchestrator + CLI entry point
│   ├── stats/
│   │   ├── cuped.py         # CUPED variance reduction
│   │   ├── sequential.py    # mSPRT sequential testing
│   │   ├── power.py         # Power analysis & MDE
│   │   └── guardrails.py    # Guardrail metric checks
│   └── novelty/
│       └── correction.py    # Novelty effect detection & correction
├── tests/                   # 41 pytest tests, all passing
├── notebooks/
│   ├── 01_cuped_demo.ipynb            # CUPED power gain visualizations
│   └── 02_sequential_testing_demo.ipynb  # mSPRT vs peeking simulation
├── data/
│   └── sample_experiment.csv          # Try it immediately
└── docs/
    └── methodology.md       # Full statistical methodology with math
```

---

## Running Tests

```bash
pytest tests/ -v
# 41 passed in 0.86s
```

---

## Experiment Decision Framework

```
☐ 1. Power analysis before launch — set n, MDE, α
☐ 2. Randomization sanity check — verify group balance
☐ 3. Apply CUPED if pre-experiment data is available
☐ 4. Use mSPRT if you need to monitor in real-time
☐ 5. Check all guardrail metrics (Holm-Bonferroni corrected)
☐ 6. Check for novelty effect if experiment ran < 2 weeks
☐ 7. Report: effect size, CI, p-value, MDE, power
```

| Primary metric | Guardrails | Decision |
|---|---|---|
| ✅ Significant | ✅ All pass | 🚀 Ship |
| ✅ Significant | 🚨 Any fail | 🔧 Fix the regression first |
| ❌ Not significant | — | ⏳ Extend or conclude no effect |

---

## Statistical References

- Deng et al. (2013). *Improving the sensitivity of online controlled experiments by utilizing pre-experiment data.* WSDM. — **CUPED**
- Johari et al. (2017). *Peeking at A/B tests: Why it matters and what to do about it.* KDD. — **mSPRT**
- Howard et al. (2021). *Time-uniform, nonparametric, nonasymptotic confidence sequences.* Annals of Statistics. — **Sequential testing**
- Cohen, J. (1988). *Statistical power analysis for the behavioral sciences.* — **Power analysis**

---

## Requirements

Python 3.9+ — see `requirements.txt`.
