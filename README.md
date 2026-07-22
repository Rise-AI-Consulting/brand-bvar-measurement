# PyMC BVARX brand measurement example

This repository contains the reproducible code for Luca Fiaschi's blog post, "Measuring Brand Without Pretending It Converts This Week."

The example has two stages:

1. `experiments/pymc_marketing_gp_baseline.py` simulates weekly sales and fits a PyMC-Marketing MMM with a time-varying intercept using an HSGP prior.
2. `experiments/brand_bvar_simulation.py` fits a generic PyMC BVARX model for monthly brand-system dynamics among base sales, awareness, consideration, and brand media.
3. `experiments/bvecm_cointegration_demo.py` fits a small Bayesian VECM / Bayesian CVAR demo to show what changes when the model estimates a long-run equilibrium relationship instead of only stationary level dynamics.

The BVARX example is intentionally synthetic. It is a teaching implementation, not a reproduction of Cain's full empirical CVAR/cointegration workflow.

## Run

```bash
cd experiments
uv run --project . brand_bvar_simulation.py
uv run --project . pymc_marketing_gp_baseline.py
uv run --project . bvecm_cointegration_demo.py
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

## Bayesian VECM / CVAR contrast

The `bvecm_cointegration_demo.py` script demonstrates the model class that is closer to Cain's CVAR logic. A VECM works in changes and adds an error-correction term:

```python
# equilibrium error: base[t-1] - beta_awareness * awareness[t-1]
#                    - beta_consideration * consideration[t-1]
ect = pm.Deterministic("equilibrium_error", pt.dot(y_level_lag, beta_vec), dims="time")

alpha = pm.Normal("alpha", mu=np.array([-0.20, 0.03, 0.03]), sigma=0.18, dims="equation")
gamma = pm.Normal("gamma", 0.0, 0.12, dims=("equation", "lagged_delta_variable"))
media_effect = pm.Normal("media_effect", mu=np.array([0.0, 0.07, 0.04]), sigma=0.12, dims="equation")

mu = intercept + ect[:, None] * alpha[None, :] + pt.dot(dy_lag, gamma.T) + x_now * media_effect
```

In the synthetic run the true cointegration relationship is `base = 0.65 * awareness + 0.35 * consideration`. The Bayesian VECM posterior recovered approximately `0.69` and `0.36`, with max R-hat `1.00` and min bulk ESS `1,897`.
