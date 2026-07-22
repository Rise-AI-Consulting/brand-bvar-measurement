# Claim traceability

## Source/method claims

1. **Claim:** Classical MMM is useful for short-term marketing effects but can miss long-term brand effects when measured only against short-term sales.
   - **Support:** Cain paper framing in the public IJRM PDF, especially the two-step framework and commercial MMM discussion, plus the synthetic Stage 1/Stage 2 design documented in `analysis-contract.md` and implemented in `experiments/pymc_marketing_gp_baseline.py` and `experiments/brand_bvar_simulation.py`.

2. **Claim:** Cain proposes a two-step framework: short-term UCM/DLM-style path-to-purchase modeling and long-term VAR/CVAR brand-building modeling.
   - **Support:** Cain IJRM PDF states the two-step framework and use of a parsimonious VAR on extracted trend, mindset metrics, paid and earned media.

3. **Claim:** Base sales and attitudinal metrics are treated as brand-health signals, and observed sales can contaminate long-term movement with short-term effects.
   - **Support:** Cain IJRM PDF discussion of mindset metrics and observed sales contamination.

4. **Claim:** Cain's application found online paid media such as search/display was short-term while offline paid media, in-store marketing, and new product PR contributed to long-term brand loyalty.
   - **Support:** Cain IJRM PDF application summary and managerial implications sections.

5. **Claim:** Cain notes practical use in commercial MMM settings with two to three years of longitudinal time-series data.
   - **Support:** Cain IJRM PDF introduction and managerial implications sections.

6. **Claim:** Cain's paper used a cointegrated VAR / CVAR structure, not the simpler stationary BVARX in this post.
   - **Support:** Cain IJRM PDF VAR/CVAR model specification section.

## Implementation claims

7. **Claim:** A conceptual measurement stack separates short-term response modeling from base extraction before the BVARX brand-system layer.
   - **Support:** `brand-measurement-stack.png`; `figure-manifest.md`. This is a Gemini-generated conceptual infographic, not an empirical result.

8. **Claim:** The executable example includes a PyMC-Marketing first-stage MMM with `time_varying_intercept=True` and an HSGP prior via `intercept_tvp_config`.
   - **Support:** `experiments/pymc_marketing_gp_baseline.py`; PyMC-Marketing docs page `MMM with time-varying media baseline`, which documents HSGP time-varying components and the `HSGPKwargs` configuration pattern.

9. **Claim:** The first-stage simulated observed weekly sales includes short-term media, promotions, seasonality, noise, and an evolving true base; the HSGP baseline is a more realistic first-stage artifact than handing the BVAR a clean latent base. In the compact demo, the baseline median has about 0.95 correlation with the true synthetic base on the scaled target axis.
   - **Support:** `outputs/stage1_weekly_mmm_data.csv`, `outputs/stage1_gp_baseline_recovery.csv`, `figures/pymc-marketing-gp-baseline.png`, and `outputs/stage1_gp_baseline_diagnostics.json`.

10. **Claim:** The executable second-stage example fits a generic PyMC BVARX on base sales, awareness, consideration, and brand media.
   - **Support:** `analysis-contract.md`; `experiments/brand_bvar_simulation.py`, especially `BVARXSpec`, `build_lagged_design`, `make_direct_effect_mask`, and `fit_bvarx`.

11. **Claim:** The PyMC BVARX uses `y_t = c + A y_{t-1} + B x_t + epsilon_t` in the one-lag blog example, with a generic implementation that supports configurable variables, exogenous regressors, and lag order.
   - **Support:** `experiments/brand_bvar_simulation.py`, functions `BVARXSpec`, `coefficient_prior_arrays`, and `fit_bvarx`.

12. **Claim:** Brand media has no immediate direct effect on extracted base sales in the synthetic DGP/model; base sales moves through lagged brand dynamics.
   - **Support:** `experiments/brand_bvar_simulation.py`, where true `B = [[0.0], [0.18], [0.08]]` and the model uses `make_direct_effect_mask(... blocked_effects=[("base_sales", "brand_media")])`.

13. **Claim:** The BVARX likelihood uses correlated multivariate residuals rather than independent per-equation Normal errors.
   - **Support:** `experiments/brand_bvar_simulation.py`, function `fit_bvarx`, where `pm.LKJCholeskyCov` and `pm.MvNormal` define the likelihood.

14. **Claim:** The posterior impulse response uses `IRF_0 = B` and recursively propagates through the VAR lag matrices.
   - **Support:** `experiments/brand_bvar_simulation.py`, functions `posterior_irf`, `_irf_from_arrays`, and `true_irf`; outputs `outputs/posterior_irf.csv` and `outputs/true_irf.csv`.

15. **Claim:** MCMC used four chains, 800 tune draws, and 800 posterior draws per chain; maximum R-hat was 1.00 and minimum bulk ESS was 2,899.
   - **Support:** `outputs/diagnostics.json`; `outputs/run-log.json`.

16. **Claim:** All posterior companion-matrix draws were stable; the 95th percentile of the largest absolute eigenvalue was 0.89.
   - **Support:** `outputs/diagnostics.json`.

17. **Claim:** A Bayesian VECM / CVAR works in first differences and adds a long-run equilibrium error term, while the main BVARX example works in levels.
   - **Support:** `experiments/bvecm_cointegration_demo.py`, especially `fit_bvecm`, where `equilibrium_error = y_level_lag @ beta_vec` enters the mean for `dy_now` through `alpha`.

18. **Claim:** In the synthetic Bayesian VECM demo, the true cointegration relationship is `base = 0.65 * awareness + 0.35 * consideration`; posterior means recovered about `0.69` and `0.36`, with zero divergences, max R-hat 1.01, and minimum bulk ESS 1,897 across the core reported coefficients.
   - **Support:** `outputs/bvecm_synthetic_truth.json`, `outputs/bvecm_cointegration_summary.csv`, `outputs/bvecm_diagnostics.json`, and `outputs/bvecm-run-log.json`.

## Numerical blog claims

19. **Claim:** First-month base-sales lift is zero by design because direct media-to-base effect is fixed at zero.
   - **Support:** `outputs/long_term_uplift_summary.csv`, row `through_month=1`; `outputs/synthetic_truth.json`.

20. **Claim:** Through 12 months, mean cumulative base-sales lift was about 66,000.
   - **Support:** `outputs/long_term_uplift_summary.csv`, row `through_month=12`.

21. **Claim:** Through 36 months, mean cumulative base-sales lift was about 80,000 with 90% posterior interval from about 45,000 to 131,000.
   - **Support:** `outputs/long_term_uplift_summary.csv`, row `through_month=36`.

## Figure claims

22. **Claim:** Synthetic data figure shows brand media, extracted base sales, awareness, and consideration over time.
   - **Support:** `figures/synthetic-brand-system.png`; `outputs/synthetic_brand_data.csv`; `figure-manifest.md`.

23. **Claim:** True-vs-estimated IRF figure shows synthetic recovery against known truth.
   - **Support:** `figures/true-vs-estimated-irf.png`; `outputs/posterior_irf.csv`; `outputs/true_irf.csv`; `figure-manifest.md`.

24. **Claim:** IRF figure shows posterior impulse responses and 90% intervals.
   - **Support:** `figures/bvar-impulse-response.png`; `outputs/posterior_irf.csv`; `figure-manifest.md`.

25. **Claim:** Cumulative uplift figure shows long-term base-sales lift at selected horizons.
   - **Support:** `figures/long-term-base-uplift.png`; `outputs/long_term_uplift_summary.csv`; `figure-manifest.md`.

26. **Claim:** Bayesian VECM figure shows non-stationary synthetic brand states drifting together and the estimated equilibrium error tracking the true error.
   - **Support:** `figures/bvecm-cointegration-demo.png`, `outputs/bvecm_synthetic_data.csv`, and `outputs/bvecm_cointegration_summary.csv`.
