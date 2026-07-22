# Analysis contract

## Unit of analysis

Monthly brand-system observations after the first-stage MMM/UCM has extracted an evolving base-sales series.

## Immutable source artifacts

- Cain, P.M. "Modelling short-and long-term marketing effects in the consumer purchase journey," IJRM 39 (2022) 96-116, available online 24 June 2021. Public PDF: https://market.science/wp-content/uploads/Modelling-short-and-long-term-effects_IJRM-2021.pdf
- `experiments/pymc_marketing_gp_baseline.py`: synthetic PyMC-Marketing HSGP baseline extraction demonstration.
- `experiments/brand_bvar_simulation.py`: full reproducible simulation and PyMC BVARX fit.
- `outputs/*.csv`, `outputs/*.json`: generated model outputs.
- `figures/*.png`: generated figures.

## Modeling boundary

The executable example starts after base extraction. It assumes a short-term model has already removed price, promotion, seasonality, and direct media response from observed sales and produced an extracted base trend. This keeps the synthetic example focused on the long-term brand system.

## Decision-time information boundary

At month `t`, the BVAR uses only brand media at `t` and endogenous variables from `t-1`. No future brand metrics or sales outcomes are used as predictors.

## Metrics

- Posterior impulse response by month after a one-standard-deviation brand media shock.
- Cumulative base-sales lift through 1, 6, 12, 24, and 36 months.
- MCMC diagnostics: maximum R-hat and minimum bulk ESS for model parameters.

## Most likely failure modes and diagnostics

1. **Toy-overreach:** The blog may imply a full reproduction of Cain. Diagnostic: claims and blog text explicitly call this a synthetic PyMC BVAR illustration, not a reproduction.
2. **BVAR/VAR terminology error:** BVAR means Bayesian VAR / Bayesian Vector Autoregression. Diagnostic: prose uses that definition.
3. **Unstable VAR dynamics:** Impulse responses could explode if posterior autoregressive matrices imply roots outside the unit circle. Diagnostic: synthetic truth is stable and posterior IRFs decay in generated figure.
4. **Causal overclaiming:** Observational brand media shocks do not by themselves identify causal effects. Diagnostic: limitations section states the required identification assumptions.
5. **Traceability gap:** Numerical claims must map to outputs. Diagnostic: `blog/claim-traceability.md` maps every number to generated CSV/JSON or paper lines.
