# Changelog

---

## v0.2.0 — 2026-05-27

### Added
- **`src/compare.py`** — Multi-experiment comparison dashboard. Run N experiments in one call and generate a unified HTML dashboard with lift chart, p-value dot plot, verdict cards, and full comparison table
- **`src/reporting.py`** — Fully redesigned HTML report (dark theme, business-first layout)
  - Business summary at the top in plain English — no statistics knowledge required
  - KPI metric cards, 95% Confidence Interval forest plot, distribution chart, power curve
  - Guardrail comparison table with % change per metric
  - Self-contained single HTML file, zero external dependencies
- **`src/logger.py`** — Production-grade structured logging
  - Text (development) and JSON (production) output formats
  - Environment-variable config: `LOG_LEVEL`, `LOG_FORMAT`, `LOG_FILE`
  - Rotating file handler (10 MB × 5 backups)
  - Logging wired through every module with structured `extra={}` fields
- **`--report`**, **`--log-level`**, **`--log-format`**, **`--log-file`** CLI flags added
- Four realistic business case datasets in `data/` covering every verdict outcome
  - `case1_checkout_cta.csv` — 🚀 Ship (significant, guardrails pass)
  - `case2_homepage_banner.csv` — ⏳ Extend (not significant)
  - `case3_pricing_page.csv` — 🔧 Fix guardrails (significant but page load fails)
  - `case4_recommendation_algo.csv` — ⚠️ Novelty effect detected

### Changed
- `src/experiment.py` — Full logging throughout pipeline; `ExperimentResults` now includes `n_control`, `n_treatment`, `cuped_applied`, `duration_seconds`
- `src/stats/cuped.py` — Debug logging for theta estimation and variance reduction %
- `src/stats/sequential.py` — Debug logging for lambda_t and p-value at each step
- `src/stats/guardrails.py` — Per-metric debug logs with raw and adjusted p-values
- `src/novelty/correction.py` — Debug logging for early/late cohort comparison
- `README.md` — Updated with logging, HTML report, and comparison dashboard sections

### Coming in v0.3.0
- Sample Ratio Mismatch (SRM) detection
- Confidence intervals in CLI output
- Bayesian A/B testing mode
- Multi-covariate CUPED
- Experiment registry (YAML config)

---

## v0.1.0 — 2026-05-26

### Added
- `src/experiment.py` — Core `Experiment` class and CLI entry point
- `src/stats/cuped.py` — CUPED variance reduction (Deng et al., 2013)
- `src/stats/sequential.py` — mSPRT sequential testing (Johari et al., 2017)
- `src/stats/power.py` — Power analysis, MDE, and sample size calculator
- `src/stats/guardrails.py` — Guardrail metric checks with Holm-Bonferroni correction
- `src/novelty/correction.py` — Novelty effect detection and late-cohort correction
- 41 passing pytest tests across all modules
- Jupyter notebooks: CUPED demo, sequential testing demo
- `docs/methodology.md` — Full statistical methodology with references
