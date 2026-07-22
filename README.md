# PyMC BVARX brand measurement example

This repository contains the reproducible code for Luca Fiaschi's blog post, "Measuring Brand Without Pretending It Converts This Week."

The example has two stages:

1. `experiments/pymc_marketing_gp_baseline.py` simulates weekly sales and fits a PyMC-Marketing MMM with a time-varying intercept using an HSGP prior.
2. `experiments/brand_bvar_simulation.py` fits a generic PyMC BVARX model for monthly brand-system dynamics among base sales, awareness, consideration, and brand media.

The BVARX example is intentionally synthetic. It is a teaching implementation, not a reproduction of Cain's full empirical CVAR/cointegration workflow.

## Run

```bash
cd experiments
uv run --project . brand_bvar_simulation.py
uv run --project . pymc_marketing_gp_baseline.py
uv run --project . bvar_generic_smoke_test.py
```

Outputs are written to `outputs/` and figures to `figures/`.

## Key model call

```python
spec = BVARXSpec(
    variable_names=["base_sales", "awareness", "consideration"],
    exog_names=["brand_media"],
    lags=1,
)

mask = make_direct_effect_mask(
    spec.variable_names,
    spec.exog_names,
    blocked_effects=[("base_sales", "brand_media")],
)

idata = fit_bvarx(y, x, spec, direct_effect_mask=mask)
```

The implementation behind `fit_bvarx` uses shrinkage priors for VAR coefficients, a masked exogenous coefficient matrix, and a multivariate normal likelihood with an LKJ prior on residual covariance.
