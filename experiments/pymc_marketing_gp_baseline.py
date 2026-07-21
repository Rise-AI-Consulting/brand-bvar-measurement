from __future__ import annotations

import json
import warnings
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymc_marketing.hsgp_kwargs import HSGPKwargs
from pymc_marketing.mmm import GeometricAdstock, MichaelisMentenSaturation
from pymc_marketing.mmm.mmm import MMM

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

RNG = np.random.default_rng(20260721)


def load_monthly_base() -> pd.DataFrame:
    path = OUT / "synthetic_brand_data.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Run brand_bvar_simulation.py first so synthetic_brand_data.csv exists."
        )
    df = pd.read_csv(path, parse_dates=["month"])
    return df[["month", "base_sales"]]


def simulate_weekly_mmm_data(monthly_base: pd.DataFrame) -> pd.DataFrame:
    weeks = pd.date_range(monthly_base["month"].min(), monthly_base["month"].max(), freq="W-MON")
    combined_index = monthly_base["month"].dt.normalize().tolist() + list(weeks)
    combined_index = pd.DatetimeIndex(combined_index).sort_values().unique()
    # Work in thousands of sales units. PyMC-Marketing scales the target internally,
    # but keeping the raw numbers moderate avoids unnecessary sampler geometry issues
    # in this compact documentation example.
    base_weekly = (
        monthly_base.set_index("month")["base_sales"]
        .reindex(combined_index)
        .interpolate(method="time")
        .reindex(weeks)
        .ffill()
        .bfill()
        / 1_000
    )
    n = len(weeks)
    t = np.arange(n)
    performance_media = RNG.gamma(shape=2.2, scale=70.0, size=n)
    performance_media += 50.0 * (np.sin(2 * np.pi * t / 26) + 1)
    promo_event = np.zeros(n)
    promo_event[[18, 57, 103, 166, 249, 311]] = 1
    seasonal = 80.0 * np.sin(2 * np.pi * t / 52) + 35.0 * np.cos(2 * np.pi * t / 26)
    short_term_media = 13.0 * np.sqrt(performance_media)
    promo = 115.0 * promo_event
    noise = RNG.normal(0, 35.0, size=n)
    observed_sales = base_weekly.to_numpy() + seasonal + short_term_media + promo + noise
    df = pd.DataFrame(
        {
            "date_week": weeks,
            "performance_media": performance_media,
            "promo_event": promo_event,
            "observed_sales": observed_sales,
            "true_base_sales": base_weekly.to_numpy(),
        }
    )
    df.to_csv(OUT / "stage1_weekly_mmm_data.csv", index=False)
    return df


def fit_time_varying_intercept_mmm(df: pd.DataFrame):
    hsgp_kwargs = HSGPKwargs(m=60, L=None, ls_mu=20.0, ls_sigma=8.0)
    mmm = MMM(
        date_column="date_week",
        channel_columns=["performance_media"],
        control_columns=["promo_event"],
        yearly_seasonality=2,
        adstock=GeometricAdstock(l_max=6).set_dims_for_all_priors("channel"),
        saturation=MichaelisMentenSaturation().set_dims_for_all_priors("channel"),
        time_varying_intercept=True,
        model_config={"intercept_tvp_config": hsgp_kwargs},
    )
    X = df[["date_week", "performance_media", "promo_event"]]
    y = df["observed_sales"]
    idata = mmm.fit(
        X=X,
        y=y,
        draws=250,
        tune=500,
        chains=2,
        target_accept=0.95,
        random_seed=20260721,
        progressbar=False,
    )
    return mmm, idata


def summarize_and_plot(df: pd.DataFrame, mmm) -> None:
    intercept = mmm.fit_result["intercept"]
    q = intercept.quantile([0.05, 0.5, 0.95], dim=("chain", "draw"))
    target_scale = 1.0
    recovered = pd.DataFrame(
        {
            "date_week": pd.to_datetime(mmm.fit_result.coords["date"].values),
            "estimated_base_q05": q.sel(quantile=0.05).values * target_scale,
            "estimated_base_q50": q.sel(quantile=0.5).values * target_scale,
            "estimated_base_q95": q.sel(quantile=0.95).values * target_scale,
        }
    )
    # PyMC-Marketing's posterior contribution variables are on the scaled target axis.
    # For this demonstration we store the recovered baseline on that same axis and
    # use correlation/shape as the diagnostic rather than pretending this is a
    # production-grade dollar calibration.
    out = df.merge(recovered, on="date_week", how="left")
    observed_scale = float(out["observed_sales"].max())
    out["observed_sales_scaled"] = out["observed_sales"] / observed_scale
    out["true_base_sales_scaled"] = out["true_base_sales"] / observed_scale
    out.to_csv(OUT / "stage1_gp_baseline_recovery.csv", index=False)
    rmse = float(np.sqrt(np.mean((out["estimated_base_q50"] - out["true_base_sales_scaled"]) ** 2)))
    corr = float(np.corrcoef(out["estimated_base_q50"], out["true_base_sales_scaled"])[0, 1])
    summary = az.summary(mmm.fit_result, var_names=["intercept_baseline", "y_sigma"])
    diagnostics = {
        "stage1_base_rmse": rmse,
        "stage1_base_correlation": corr,
        "max_rhat_selected": float(summary["r_hat"].max()),
        "min_ess_bulk_selected": float(summary["ess_bulk"].min()),
        "draws": 250,
        "tune": 500,
        "chains": 2,
        "pymc_marketing_feature": "MMM(time_varying_intercept=True, model_config={'intercept_tvp_config': HSGPKwargs(...)})",
        "docs": "https://www.pymc-marketing.io/en/latest/notebooks/mmm/mmm_time_varying_media_example.html",
    }
    (OUT / "stage1_gp_baseline_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(out["date_week"], out["observed_sales_scaled"], color="0.78", linewidth=0.8, label="Observed weekly sales, scaled")
    ax.plot(out["date_week"], out["true_base_sales_scaled"], color="black", linestyle="--", linewidth=1.8, label="True synthetic base, scaled")
    ax.plot(out["date_week"], out["estimated_base_q50"], color="#1b9e77", linewidth=2.2, label="PyMC-Marketing HSGP baseline median")
    ax.set_title("Stage 1 diagnostic: estimated time-varying base tracks the synthetic base")
    ax.set_ylabel("Scaled sales")
    ax.set_xlabel("Week")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "pymc-marketing-gp-baseline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    monthly = load_monthly_base()
    weekly = simulate_weekly_mmm_data(monthly)
    mmm, _ = fit_time_varying_intercept_mmm(weekly)
    summarize_and_plot(weekly, mmm)
    print((OUT / "stage1_gp_baseline_diagnostics.json").read_text())


if __name__ == "__main__":
    main()
