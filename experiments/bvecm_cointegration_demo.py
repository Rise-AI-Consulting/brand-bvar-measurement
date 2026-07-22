from __future__ import annotations

import json
from pathlib import Path

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

RNG = np.random.default_rng(20260722)
VARIABLES = ["base_sales", "awareness", "consideration"]
T = 120


def simulate_cointegrated_brand_system() -> pd.DataFrame:
    """Simulate non-stationary brand states tied by one long-run equilibrium.

    Awareness and consideration are persistent random-walk-like states. Base sales
    is allowed to drift with them, but the system corrects when base sales gets
    too high or too low relative to the long-run brand-health relationship.
    """
    k = len(VARIABLES)
    y = np.zeros((T, k))
    media = RNG.normal(0, 0.5, size=T)
    for start, amp in [(12, 1.2), (36, 1.0), (72, 1.4), (96, 1.1)]:
        media[start : start + 4] += amp
    media = (media - media.mean()) / media.std()

    # beta defines the equilibrium error: base - 0.65*awareness - 0.35*consideration.
    beta = np.array([1.0, -0.65, -0.35])
    alpha = np.array([-0.28, 0.06, 0.04])
    gamma = np.array(
        [
            [0.18, 0.05, 0.02],
            [0.03, 0.12, 0.04],
            [0.02, 0.06, 0.10],
        ]
    )
    b_media = np.array([0.00, 0.09, 0.05])
    chol = np.array(
        [
            [0.055, 0.000, 0.000],
            [0.020, 0.075, 0.000],
            [0.012, 0.025, 0.065],
        ]
    )

    dy_prev = np.zeros(k)
    for t in range(1, T):
        ect = y[t - 1] @ beta
        noise = chol @ RNG.normal(size=k)
        dy = alpha * ect + gamma @ dy_prev + b_media * media[t] + noise
        y[t] = y[t - 1] + dy
        dy_prev = dy

    df = pd.DataFrame(
        {
            "month": pd.date_range("2015-01-01", periods=T, freq="MS"),
            "brand_media_std": media,
            "base_sales_std": y[:, 0],
            "awareness_std": y[:, 1],
            "consideration_std": y[:, 2],
            "equilibrium_error_true": y @ beta,
        }
    )
    truth = {
        "beta": beta.tolist(),
        "alpha": alpha.tolist(),
        "gamma": gamma.tolist(),
        "b_media": b_media.tolist(),
    }
    (OUT / "bvecm_synthetic_truth.json").write_text(json.dumps(truth, indent=2))
    df.to_csv(OUT / "bvecm_synthetic_data.csv", index=False)
    return df


def fit_bvecm(y: np.ndarray, x: np.ndarray, *, draws: int = 600, tune: int = 600, chains: int = 4):
    """Fit a Bayesian VECM with one normalized cointegration relationship.

    The cointegration vector is normalized on base sales:

        ect[t-1] = base[t-1] - beta_awareness*awareness[t-1]
                   - beta_consideration*consideration[t-1]

    The model estimates how changes in each variable react to that disequilibrium.
    """
    dy = np.diff(y, axis=0)
    dy_now = dy[1:]
    dy_lag = dy[:-1]
    y_level_lag = y[1:-1]
    x_now = x[2:, None] if x.ndim == 1 else x[2:]
    k = y.shape[1]

    coords = {
        "equation": VARIABLES,
        "lagged_delta_variable": VARIABLES,
        "brand_variable": ["awareness", "consideration"],
        "time": np.arange(len(dy_now)),
    }

    with pm.Model(coords=coords) as model:
        intercept = pm.Normal("intercept", 0.0, 0.05, dims="equation")
        beta_brand = pm.Normal(
            "beta_brand",
            mu=np.array([0.5, 0.5]),
            sigma=np.array([0.35, 0.35]),
            dims="brand_variable",
        )
        beta_vec = pt.concatenate([pt.ones((1,)), -beta_brand])
        ect = pm.Deterministic("equilibrium_error", pt.dot(y_level_lag, beta_vec), dims="time")

        alpha = pm.Normal("alpha", mu=np.array([-0.20, 0.03, 0.03]), sigma=0.18, dims="equation")
        gamma = pm.Normal("gamma", 0.0, 0.12, dims=("equation", "lagged_delta_variable"))
        media_effect = pm.Normal("media_effect", mu=np.array([0.0, 0.07, 0.04]), sigma=0.12, dims="equation")

        chol, corr, sigma = pm.LKJCholeskyCov(
            "chol_cov",
            n=k,
            eta=3.0,
            sd_dist=pm.Exponential.dist(12.0, shape=k),
            compute_corr=True,
        )
        pm.Deterministic("residual_corr", corr, dims=("equation", "lagged_delta_variable"))
        pm.Deterministic("residual_sigma", sigma, dims="equation")

        mu = intercept + ect[:, None] * alpha[None, :] + pt.dot(dy_lag, gamma.T) + x_now * media_effect
        pm.MvNormal("obs", mu=mu, chol=chol, observed=dy_now, dims=("time", "equation"))

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.93,
            random_seed=20260722,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
    return idata


def summarize(idata) -> dict:
    core_var_names = ["intercept", "beta_brand", "alpha", "gamma", "media_effect", "residual_sigma"]
    summary = az.summary(idata, var_names=core_var_names)
    summary.to_csv(OUT / "bvecm_posterior_summary.csv")
    divergences = int(idata.sample_stats["diverging"].sum().item())
    diagnostics = {
        "diagnostic_scope": "intercept, beta_brand, alpha, gamma, media_effect, residual_sigma",
        "divergences": divergences,
        "max_r_hat": float(summary["r_hat"].max()),
        "min_ess_bulk": int(summary["ess_bulk"].min()),
        "draws": 600,
        "tune": 600,
        "chains": 4,
        "beta_awareness_mean": float(summary.loc["beta_brand[awareness]", "mean"]),
        "beta_consideration_mean": float(summary.loc["beta_brand[consideration]", "mean"]),
        "alpha_base_sales_mean": float(summary.loc["alpha[base_sales]", "mean"]),
    }
    (OUT / "bvecm_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    return diagnostics


def plot_results(df: pd.DataFrame, idata) -> None:
    posterior = idata.posterior if hasattr(idata, "posterior") else idata["posterior"].ds
    beta_aw = posterior["beta_brand"].sel(brand_variable="awareness").values.reshape(-1)
    beta_cons = posterior["beta_brand"].sel(brand_variable="consideration").values.reshape(-1)
    beta_aw_mean = beta_aw.mean()
    beta_cons_mean = beta_cons.mean()
    estimated_ect = (
        df["base_sales_std"].to_numpy()
        - beta_aw_mean * df["awareness_std"].to_numpy()
        - beta_cons_mean * df["consideration_std"].to_numpy()
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(df["month"], df["base_sales_std"], label="base sales", linewidth=1.5)
    axes[0].plot(df["month"], df["awareness_std"], label="awareness", linewidth=1.2)
    axes[0].plot(df["month"], df["consideration_std"], label="consideration", linewidth=1.2)
    axes[0].set_title("Cointegrated synthetic brand states can drift together")
    axes[0].set_ylabel("standardized level")
    axes[0].legend(frameon=False, ncol=3)

    axes[1].plot(df["month"], df["equilibrium_error_true"], label="true equilibrium error", linewidth=1.5)
    axes[1].plot(df["month"], estimated_ect, label="posterior-mean equilibrium error", linewidth=1.2, linestyle="--")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("The VECM models correction toward this long-run relationship")
    axes[1].set_ylabel("base - beta' brand health")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "bvecm-cointegration-demo.png", dpi=180)
    plt.close(fig)

    rows = []
    for name, draws in [("awareness", beta_aw), ("consideration", beta_cons)]:
        rows.append(
            {
                "coefficient": f"beta_{name}",
                "mean": draws.mean(),
                "q05": np.quantile(draws, 0.05),
                "q50": np.quantile(draws, 0.50),
                "q95": np.quantile(draws, 0.95),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "bvecm_cointegration_summary.csv", index=False)


def main() -> None:
    df = simulate_cointegrated_brand_system()
    y = df[["base_sales_std", "awareness_std", "consideration_std"]].to_numpy()
    x = df["brand_media_std"].to_numpy()
    idata = fit_bvecm(y, x)
    diagnostics = summarize(idata)
    plot_results(df, idata)
    print(json.dumps(diagnostics))


if __name__ == "__main__":
    main()
