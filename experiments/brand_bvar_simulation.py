from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

RNG = np.random.default_rng(1749)
VARIABLES = ["base_sales", "awareness", "consideration"]
EXOG = ["brand_media"]
T = 84  # monthly observations, seven years
HORIZON = 36
BASE_SALES_SCALE = 180_000


@dataclass(frozen=True)
class BVARXSpec:
    """Configuration for a Bayesian VARX(p) model.

    The implementation is generic over the number of endogenous variables, number
    of exogenous regressors, lag order, and allowed direct exogenous effects. The
    blog example uses one exogenous regressor and masks its same-month effect on
    base sales, but that is a modeling choice, not a structural limitation of the
    BVARX builder.
    """

    variable_names: list[str]
    exog_names: list[str]
    lags: int = 1
    intercept_sigma: float = 0.30
    own_lag_mu: float = 0.55
    own_lag_sigma: float = 0.18
    cross_lag_sigma: float = 0.10
    lag_decay: float = 0.60
    exog_sigma: float = 0.20
    lkj_eta: float = 3.0


def build_lagged_design(y: np.ndarray, x: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create aligned VARX design arrays.

    Parameters
    ----------
    y
        Endogenous series with shape (time, variables), already on a comparable
        scale. In this example the variables are standardized synthetic states.
    x
        Exogenous regressors with shape (time, exog).
    lags
        VAR lag order p.

    Returns
    -------
    y_now
        Shape (time - p, variables).
    y_lags
        Shape (time - p, p, variables), where lag index 0 is t-1.
    x_now
        Shape (time - p, exog).
    """
    if y.ndim != 2:
        raise ValueError("y must have shape (time, variables)")
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("x must have shape (time, exog)")
    if len(y) != len(x):
        raise ValueError("y and x must have the same number of time points")
    if lags < 1:
        raise ValueError("lags must be >= 1")
    if len(y) <= lags:
        raise ValueError("need more observations than lags")

    y_now = y[lags:]
    y_lags = np.stack([y[lags - lag : len(y) - lag] for lag in range(1, lags + 1)], axis=1)
    x_now = x[lags:]
    return y_now, y_lags, x_now


def make_direct_effect_mask(
    variable_names: Iterable[str],
    exog_names: Iterable[str],
    blocked_effects: Iterable[tuple[str, str]] = (),
) -> np.ndarray:
    """Return an equation x exog mask, with zeros for blocked direct effects."""
    variables = list(variable_names)
    exogs = list(exog_names)
    mask = np.ones((len(variables), len(exogs)), dtype=float)
    for variable, exog in blocked_effects:
        mask[variables.index(variable), exogs.index(exog)] = 0.0
    return mask


def coefficient_prior_arrays(spec: BVARXSpec) -> tuple[np.ndarray, np.ndarray]:
    """Create Minnesota-style prior mean/sd arrays for A[lag, equation, variable]."""
    k = len(spec.variable_names)
    mu = np.zeros((spec.lags, k, k))
    sigma = np.zeros((spec.lags, k, k))
    for lag in range(spec.lags):
        decay = spec.lag_decay**lag
        for eq in range(k):
            for var in range(k):
                if lag == 0 and eq == var:
                    mu[lag, eq, var] = spec.own_lag_mu
                sigma[lag, eq, var] = (spec.own_lag_sigma if eq == var else spec.cross_lag_sigma) * decay
    return mu, sigma


def fit_bvarx(
    y: np.ndarray,
    x: np.ndarray,
    spec: BVARXSpec,
    direct_effect_mask: np.ndarray | None = None,
    *,
    draws: int = 800,
    tune: int = 800,
    chains: int = 4,
    random_seed: int = 20260721,
):
    """Fit a Bayesian VARX(p) with shrinkage priors and correlated residuals."""
    if x.ndim == 1:
        x = x[:, None]
    k = len(spec.variable_names)
    m = len(spec.exog_names)
    if y.shape[1] != k:
        raise ValueError(f"y has {y.shape[1]} columns but spec has {k} variables")
    if x.shape[1] != m:
        raise ValueError(f"x has {x.shape[1]} columns but spec has {m} exogenous regressors")
    if direct_effect_mask is None:
        direct_effect_mask = np.ones((k, m), dtype=float)
    direct_effect_mask = np.asarray(direct_effect_mask, dtype=float)
    if direct_effect_mask.shape != (k, m):
        raise ValueError(f"direct_effect_mask must have shape {(k, m)}")

    y_now, y_lags, x_now = build_lagged_design(y, x, spec.lags)
    prior_mean_A, prior_sd_A = coefficient_prior_arrays(spec)
    coords = {
        "equation": spec.variable_names,
        "lagged_variable": spec.variable_names,
        "lag": np.arange(1, spec.lags + 1),
        "exog": spec.exog_names,
        "time": np.arange(len(y_now)),
    }

    with pm.Model(coords=coords) as model:
        intercept = pm.Normal("intercept", 0, spec.intercept_sigma, dims="equation")
        A = pm.Normal(
            "A",
            mu=prior_mean_A,
            sigma=prior_sd_A,
            dims=("lag", "equation", "lagged_variable"),
        )
        B_raw = pm.Normal("B_raw", 0, spec.exog_sigma, dims=("equation", "exog"))
        B = pm.Deterministic("B", B_raw * direct_effect_mask, dims=("equation", "exog"))
        chol, corr, sigma = pm.LKJCholeskyCov(
            "chol_cov",
            n=k,
            eta=spec.lkj_eta,
            sd_dist=pm.Exponential.dist(4.0, shape=k),
            compute_corr=True,
        )
        pm.Deterministic("residual_corr", corr, dims=("equation", "lagged_variable"))
        pm.Deterministic("residual_sigma", sigma, dims="equation")

        # y_lags: obs x lag x lagged_variable
        # A: lag x equation x lagged_variable
        lagged_mu = pt.einsum("tlj,lej->te", y_lags, A)
        exog_mu = pt.dot(x_now, B.T)
        mu = intercept + lagged_mu + exog_mu
        pm.MvNormal("obs", mu=mu, chol=chol, observed=y_now, dims=("time", "equation"))
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.92,
            random_seed=random_seed,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
    return idata


def simulate_brand_system() -> pd.DataFrame:
    """Simulate a long-run brand system after a first-stage MMM has extracted base sales."""
    t = np.arange(T)
    media_raw = (
        0.45 * np.sin(2 * np.pi * t / 12)
        + 0.25 * np.sin(2 * np.pi * t / 48)
        + RNG.normal(0, 0.45, size=T)
    )
    for start, amp in [(8, 1.2), (28, 1.6), (55, 1.1), (68, 1.5)]:
        media_raw[start : start + 3] += amp
    media = (media_raw - media_raw.mean()) / media_raw.std()

    A = np.array(
        [
            [0.76, 0.16, 0.08],  # base is persistent and reacts to lagged brand metrics
            [0.05, 0.62, 0.05],  # awareness persists
            [0.04, 0.19, 0.58],  # consideration follows awareness
        ]
    )
    B = np.array([[0.0], [0.18], [0.08]])  # no same-month media-to-base effect
    c = np.array([0.03, 0.0, 0.0])
    cov = np.array(
        [
            [0.12**2, 0.004, 0.003],
            [0.004, 0.18**2, 0.010],
            [0.003, 0.010, 0.16**2],
        ]
    )
    y = np.zeros((T, len(VARIABLES)))
    y[0] = RNG.multivariate_normal(np.zeros(len(VARIABLES)), cov)
    for i in range(1, T):
        y[i] = c + A @ y[i - 1] + B[:, 0] * media[i] + RNG.multivariate_normal(np.zeros(len(VARIABLES)), cov)

    df = pd.DataFrame(
        {
            "month": pd.date_range("2018-01-01", periods=T, freq="MS"),
            "brand_media_std": media,
            "base_sales": 1_800_000 + BASE_SALES_SCALE * y[:, 0],
            "awareness": 54 + 6.5 * y[:, 1],
            "consideration": 32 + 5.0 * y[:, 2],
            "base_sales_std": y[:, 0],
            "awareness_std": y[:, 1],
            "consideration_std": y[:, 2],
        }
    )
    truth = {"A": A.tolist(), "B": B.tolist(), "c": c.tolist(), "cov": cov.tolist()}
    (OUT / "synthetic_truth.json").write_text(json.dumps(truth, indent=2))
    return df


def _posterior_dataset(idata):
    posterior_tree = idata["posterior"] if not hasattr(idata, "posterior") else idata.posterior
    return posterior_tree.ds if hasattr(posterior_tree, "ds") else posterior_tree


def _irf_from_arrays(A: np.ndarray, B: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    """Return IRFs for A/B arrays shaped (samples, lags, K, K) and (samples, K, M)."""
    n, p, k, _ = A.shape
    m = B.shape[2]
    responses = np.zeros((n, horizon + 1, k, m))
    for h in range(horizon + 1):
        if h == 0:
            response = B
        else:
            response = np.zeros((n, k, m))
            for lag in range(1, min(p, h) + 1):
                response += np.einsum("nej,njm->nem", A[:, lag - 1, :, :], responses[:, h - lag, :, :])
        responses[:, h, :, :] = response
    return responses


def true_irf() -> pd.DataFrame:
    truth = json.loads((OUT / "synthetic_truth.json").read_text())
    A = np.array(truth["A"])[None, None, :, :]
    B = np.array(truth["B"])[None, :, :]
    irf = _irf_from_arrays(A, B)[0]
    rows = []
    for h in range(HORIZON + 1):
        for j, var in enumerate(VARIABLES):
            rows.append({"horizon_month": h, "variable": var, "exog": EXOG[0], "true": float(irf[h, j, 0])})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "true_irf.csv", index=False)
    return df


def posterior_irf(idata) -> tuple[pd.DataFrame, pd.DataFrame]:
    posterior = _posterior_dataset(idata)
    stacked = posterior.stack(sample=("chain", "draw"))
    A = stacked["A"].transpose("sample", "lag", "equation", "lagged_variable").values
    B = stacked["B"].transpose("sample", "equation", "exog").values
    irf = _irf_from_arrays(A, B)

    rows = []
    for h in range(HORIZON + 1):
        for j, var in enumerate(VARIABLES):
            draws = irf[:, h, j, 0]
            rows.append(
                {
                    "horizon_month": h,
                    "variable": var,
                    "exog": EXOG[0],
                    "mean": float(draws.mean()),
                    "q05": float(np.quantile(draws, 0.05)),
                    "q50": float(np.quantile(draws, 0.50)),
                    "q95": float(np.quantile(draws, 0.95)),
                }
            )
    irf_df = pd.DataFrame(rows)

    cum_base = irf[:, :, VARIABLES.index("base_sales"), 0].cumsum(axis=1)
    summary = []
    for h in [0, 5, 11, 23, 35]:
        draws = cum_base[:, h]
        summary.append(
            {
                "through_month": h + 1,
                "cumulative_base_std_mean": float(draws.mean()),
                "cumulative_base_std_q05": float(np.quantile(draws, 0.05)),
                "cumulative_base_std_q50": float(np.quantile(draws, 0.50)),
                "cumulative_base_std_q95": float(np.quantile(draws, 0.95)),
                "incremental_base_sales_mean": float(draws.mean() * BASE_SALES_SCALE),
                "incremental_base_sales_q05": float(np.quantile(draws, 0.05) * BASE_SALES_SCALE),
                "incremental_base_sales_q95": float(np.quantile(draws, 0.95) * BASE_SALES_SCALE),
            }
        )
    cum_df = pd.DataFrame(summary)
    return irf_df, cum_df


def save_diagnostics(idata, irf_df: pd.DataFrame, cum_df: pd.DataFrame, spec: BVARXSpec):
    summary = az.summary(idata, var_names=["intercept", "A", "B", "residual_sigma", "residual_corr"])
    summary.to_csv(OUT / "posterior_summary.csv")
    posterior = _posterior_dataset(idata).stack(sample=("chain", "draw"))
    A = posterior["A"].transpose("sample", "lag", "equation", "lagged_variable").values
    companion = companion_matrices(A)
    max_abs_eig = np.array([np.max(np.abs(np.linalg.eigvals(a))) for a in companion])
    stability = {
        "max_posterior_abs_eigenvalue_q50": float(np.quantile(max_abs_eig, 0.50)),
        "max_posterior_abs_eigenvalue_q95": float(np.quantile(max_abs_eig, 0.95)),
        "share_stable_draws": float(np.mean(max_abs_eig < 1.0)),
    }
    diagnostics = {
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "draws": 800,
        "tune": 800,
        "chains": 4,
        "variables": spec.variable_names,
        "exog": spec.exog_names,
        "lags": spec.lags,
        "residual_covariance": "LKJCholeskyCov",
        **stability,
    }
    (OUT / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    irf_df.to_csv(OUT / "posterior_irf.csv", index=False)
    cum_df.to_csv(OUT / "long_term_uplift_summary.csv", index=False)


def companion_matrices(A: np.ndarray) -> np.ndarray:
    """Build companion matrices for A shaped (samples, lags, K, K)."""
    n, p, k, _ = A.shape
    comp = np.zeros((n, k * p, k * p))
    comp[:, :k, : k * p] = A.reshape(n, p * k, k).transpose(0, 2, 1)
    if p > 1:
        comp[:, k:, :-k] = np.eye(k * (p - 1))[None, :, :]
    return comp


def plot_inputs(df: pd.DataFrame):
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(df["month"], df["brand_media_std"], color="#d95f02")
    axes[0].set_ylabel("Brand\nmedia")
    axes[1].plot(df["month"], df["base_sales"] / 1_000_000, color="#1b9e77")
    axes[1].set_ylabel("Base sales\n($M)")
    axes[2].plot(df["month"], df["awareness"], color="#7570b3")
    axes[2].set_ylabel("Awareness")
    axes[3].plot(df["month"], df["consideration"], color="#e7298a")
    axes[3].set_ylabel("Consideration")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Month")
    fig.suptitle("Synthetic brand system after extracting a long-term base trend", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG / "synthetic-brand-system.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_irf(irf_df: pd.DataFrame):
    labels = {"base_sales": "Base sales", "awareness": "Awareness", "consideration": "Consideration"}
    colors = {"base_sales": "#1b9e77", "awareness": "#7570b3", "consideration": "#e7298a"}
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for var in VARIABLES:
        sub = irf_df[irf_df["variable"] == var]
        x = sub["horizon_month"].to_numpy()
        mean = sub["mean"].to_numpy()
        q05 = sub["q05"].to_numpy()
        q95 = sub["q95"].to_numpy()
        ax.plot(x, mean, label=labels[var], color=colors[var])
        ax.fill_between(x, q05, q95, color=colors[var], alpha=0.16)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Months after a one-standard-deviation brand media shock")
    ax.set_ylabel("Posterior response, standardized units")
    ax.set_title("PyMC BVARX impulse responses with 90% posterior intervals")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "bvar-impulse-response.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative(cum_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    months = cum_df["through_month"].to_numpy()
    mean = cum_df["incremental_base_sales_mean"].to_numpy() / 1_000
    q05 = cum_df["incremental_base_sales_q05"].to_numpy() / 1_000
    q95 = cum_df["incremental_base_sales_q95"].to_numpy() / 1_000
    ax.plot(months, mean, marker="o", color="#1b9e77")
    ax.fill_between(months, q05, q95, color="#1b9e77", alpha=0.18)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cumulative window")
    ax.set_ylabel("Incremental base sales, thousands")
    ax.set_title("Estimated long-term base-sales lift from one brand media shock")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "long-term-base-uplift.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_true_vs_estimated_irf(irf_df: pd.DataFrame, truth_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    colors = {"base_sales": "#1b9e77", "awareness": "#7570b3", "consideration": "#e7298a"}
    labels = {"base_sales": "Base sales", "awareness": "Awareness", "consideration": "Consideration"}
    for ax, var in zip(axes, VARIABLES, strict=True):
        est = irf_df[irf_df["variable"] == var]
        tru = truth_df[truth_df["variable"] == var]
        x = est["horizon_month"].to_numpy()
        ax.fill_between(x, est["q05"], est["q95"], color=colors[var], alpha=0.18, label="90% posterior")
        ax.plot(x, est["mean"], color=colors[var], label="Posterior mean")
        ax.plot(x, tru["true"], color="black", linestyle="--", linewidth=1.4, label="Synthetic truth")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(labels[var])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Response, standardized units")
    for ax in axes:
        ax.set_xlabel("Months")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("True vs estimated impulse responses in the synthetic BVARX example", y=1.04)
    fig.tight_layout()
    fig.savefig(FIG / "true-vs-estimated-irf.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    df = simulate_brand_system()
    df.to_csv(OUT / "synthetic_brand_data.csv", index=False)
    spec = BVARXSpec(variable_names=VARIABLES, exog_names=EXOG, lags=1)
    mask = make_direct_effect_mask(VARIABLES, EXOG, blocked_effects=[("base_sales", "brand_media")])
    y = df[[f"{v}_std" for v in VARIABLES]].to_numpy()
    x = df[["brand_media_std"]].to_numpy()
    idata = fit_bvarx(y, x, spec, mask)
    idata.to_netcdf(OUT / "brand_bvar_idata.nc")
    truth_df = true_irf()
    irf_df, cum_df = posterior_irf(idata)
    save_diagnostics(idata, irf_df, cum_df, spec)
    plot_inputs(df)
    plot_irf(irf_df)
    plot_cumulative(cum_df)
    plot_true_vs_estimated_irf(irf_df, truth_df)
    diagnostics = json.loads((OUT / "diagnostics.json").read_text())
    print(json.dumps({"diagnostics": diagnostics, "cumulative_base_lift": cum_df.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
