# Figure manifest

## `brand-measurement-stack.png`

- **Type:** Conceptual infographic generated with Gemini.
- **Generating prompt:** `prompts/brand-measurement-stack-prompt.txt`; final image was converted to true PNG.
- **Supported claim:** The measurement workflow separates short-term response modeling from base extraction before the BVARX brand-system layer.
- **Caveat:** Conceptual illustration only. It is not empirical evidence.

## `figures/pymc-marketing-gp-baseline.png`

- **Type:** Empirical synthetic first-stage MMM figure.
- **Source data:** `outputs/stage1_weekly_mmm_data.csv` and `outputs/stage1_gp_baseline_recovery.csv`.
- **Generating script:** `experiments/pymc_marketing_gp_baseline.py`.
- **Command:** `cd experiments && uv run --project . pymc_marketing_gp_baseline.py` after running `brand_bvar_simulation.py`.
- **Supported claim:** PyMC-Marketing can fit a time-varying-intercept MMM with an HSGP prior whose baseline median tracks the slow movement in an evolving synthetic base while weekly observed sales includes short-term media, promotions, seasonality, and noise.
- **Caveat:** The main article figure intentionally omits the posterior interval because the first-stage diagnostic is meant to show slow base movement, not to validate a production MMM uncertainty model.

## `figures/synthetic-brand-system.png`

- **Type:** Empirical synthetic figure.
- **Source data:** `outputs/synthetic_brand_data.csv`.
- **Generating script:** `experiments/brand_bvar_simulation.py`, function `plot_inputs`.
- **Command:** `cd experiments && uv run --project . brand_bvar_simulation.py`.
- **Supported claim:** The synthetic example represents the second-stage brand system after an evolving base has been extracted: brand media, base sales, awareness, and consideration move together over monthly time.
- **Caveat:** Synthetic data only. This does not validate the Cain commercial case study.

## `figures/true-vs-estimated-irf.png`

- **Type:** Empirical synthetic recovery figure.
- **Source data:** `outputs/posterior_irf.csv` and `outputs/true_irf.csv`.
- **Generating script:** `experiments/brand_bvar_simulation.py`, function `plot_true_vs_estimated_irf`.
- **Command:** `cd experiments && uv run --project . brand_bvar_simulation.py`.
- **Supported claim:** Because the example is synthetic, the true impulse response is known. The posterior recovers the main shape of the DGP: media first moves awareness and consideration, then base sales accumulates through lagged brand dynamics.
- **Caveat:** Recovery under a known synthetic DGP does not establish causal identification in real observational marketing data.

## `figures/bvar-impulse-response.png`

- **Type:** Empirical model-output figure.
- **Source data:** `outputs/posterior_irf.csv`.
- **Generating script:** `experiments/brand_bvar_simulation.py`, function `plot_irf`.
- **Command:** `cd experiments && uv run --project . brand_bvar_simulation.py`.
- **Supported claim:** The generic PyMC BVARX builder converts a one-standard-deviation brand-media shock into posterior impulse responses for base sales, awareness, and consideration, with uncertainty intervals.
- **Caveat:** The impulse response is conditional on the simulated DGP, the model structure, the direct-effect mask, and exogeneity of the shock.

## `figures/long-term-base-uplift.png`

- **Type:** Empirical model-output figure.
- **Source data:** `outputs/long_term_uplift_summary.csv`.
- **Generating script:** `experiments/brand_bvar_simulation.py`, function `plot_cumulative`.
- **Command:** `cd experiments && uv run --project . brand_bvar_simulation.py`.
- **Supported claim:** The example estimates cumulative long-term base-sales lift through multiple horizons and shows that most of the synthetic effect accrues after the first month because there is no direct same-month media-to-base effect.
- **Caveat:** Dollar values are derived from the synthetic scaling of one standardized base-sales unit to 180,000 base-sales dollars.
