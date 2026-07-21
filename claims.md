# Claims and implementation scope

## Central source claim

Cain (2021/2022 IJRM) argues that marketing effectiveness needs a two-step structure: first isolate short-term transactional effects and a long-term base-sales component with unobserved-component style models, then model the long-term brand-building system using mindset metrics, earned media, paid media, and extracted base sales. The paper's core claim is that mindset metrics only become credible brand-building evidence when they are connected to persistent base-sales evolution, rather than merely correlated with observed short-term sales.

## 1749 blog claim reused

The 1749 post makes the same practitioner point in simpler MMM language: a classical sales MMM usually measures short-term response and tends to undervalue brand marketing when the brand effect mainly appears through an evolving base. It proposes estimating an evolving base, then fitting a Bayesian VAR, meaning Bayesian Vector Autoregression, over the extracted base and brand metrics to compute impulse responses and long-term ROI.

## What this implementation tests

This post implements the second-stage idea with a PyMC Bayesian VAR on synthetic monthly data. The synthetic data represents the state after a first-stage MMM/UCM has already extracted base sales. The model estimates:

\[
y_t = c + A y_{t-1} + B x_t + \epsilon_t
\]

where `y_t` contains extracted base sales, awareness, and consideration, and `x_t` is exogenous brand media.

## What it does not claim

- It does not reproduce Cain's full commercial case study.
- It does not implement the paper's cointegrated VAR / error-correction system.
- It does not prove brand media is causal from observational data.
- It does not estimate the first-stage MMM/UCM. The blog explains where that stage sits and focuses the executable code on the PyMC BVAR stage.
