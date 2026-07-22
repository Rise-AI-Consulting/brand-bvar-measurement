# Claims and implementation scope

## Central source claim

Cain (2021/2022 IJRM) argues that marketing effectiveness needs a two-step structure: first isolate short-term transactional effects and a long-term base-sales component with unobserved-component style models, then model the long-term brand-building system using mindset metrics, earned media, paid media, and extracted base sales. The paper's core claim is that mindset metrics only become credible brand-building evidence when they are connected to persistent base-sales evolution, rather than merely correlated with observed short-term sales.

## Practitioner framing used in the post

A short-term sales MMM is useful, but it can undervalue brand marketing when the relevant effect is a slow change in base demand. The post uses that practitioner framing and then makes the mechanism executable: estimate an evolving base, feed that base into a brand-system model, and compute dynamic responses instead of forcing all media value into same-week sales attribution.

## What this implementation tests

The repository implements a simplified synthetic two-stage teaching example:

1. A PyMC-Marketing MMM with a time-varying intercept using an HSGP prior recovers an evolving weekly baseline from simulated observed sales.
2. A generic PyMC BVARX models monthly dynamics among extracted base sales, awareness, consideration, and exogenous brand media.
3. A separate Bayesian VECM demo simulates cointegrated brand states and estimates the long-run equilibrium relationship to show how CVAR logic differs from stationary BVARX logic.

The second-stage model estimates:

$$
y_t = c + A y_{t-1} + B x_t + \epsilon_t
$$

where `y_t` contains extracted base sales, awareness, and consideration, and `x_t` is exogenous brand media. The same-month media-to-base-sales coefficient is blocked by an explicit direct-effect mask, so base sales moves through lagged brand dynamics in the synthetic example.

## What it does not claim

- It does not reproduce Cain's full commercial case study.
- It does not implement Cain's full empirical cointegrated VAR / error-correction workflow.
- It does not prove brand media is causal from observational data.
- It does not implement Cain's full first-stage UCM/DLM. The Stage 1 script is a compact synthetic PyMC-Marketing HSGP baseline demonstration.
- It does not propagate first-stage posterior uncertainty into the second-stage BVARX. The stages are shown modularly for clarity.
- The Bayesian VECM demo is a small synthetic contrast, not a full Cain reproduction or a production cointegration testing workflow.
