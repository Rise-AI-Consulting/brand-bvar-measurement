from __future__ import annotations

import numpy as np

from brand_bvar_simulation import (
    BVARXSpec,
    _irf_from_arrays,
    build_lagged_design,
    coefficient_prior_arrays,
    make_direct_effect_mask,
)


def main() -> None:
    rng = np.random.default_rng(123)
    y = rng.normal(size=(20, 4))
    x = rng.normal(size=(20, 2))
    spec = BVARXSpec(
        variable_names=["base", "awareness", "consideration", "loyalty"],
        exog_names=["brand_media", "earned_media"],
        lags=2,
    )

    y_now, y_lags, x_now = build_lagged_design(y, x, spec.lags)
    assert y_now.shape == (18, 4)
    assert y_lags.shape == (18, 2, 4)
    assert x_now.shape == (18, 2)
    np.testing.assert_allclose(y_lags[0, 0], y[1])
    np.testing.assert_allclose(y_lags[0, 1], y[0])

    mask = make_direct_effect_mask(
        spec.variable_names,
        spec.exog_names,
        blocked_effects=[("base", "brand_media"), ("base", "earned_media")],
    )
    assert mask.shape == (4, 2)
    assert mask[0, 0] == 0
    assert mask[0, 1] == 0
    assert mask[1:, :].sum() == 6

    prior_mu, prior_sigma = coefficient_prior_arrays(spec)
    assert prior_mu.shape == (2, 4, 4)
    assert prior_sigma.shape == (2, 4, 4)
    assert prior_mu[0, 0, 0] == spec.own_lag_mu
    assert prior_mu[1].sum() == 0
    assert prior_sigma[1, 0, 0] < prior_sigma[0, 0, 0]

    A = np.zeros((1, 2, 4, 4))
    A[0, 0] = np.eye(4) * 0.5
    A[0, 1] = np.eye(4) * 0.2
    B = np.ones((1, 4, 2))
    irf = _irf_from_arrays(A, B, horizon=3)
    assert irf.shape == (1, 4, 4, 2)
    np.testing.assert_allclose(irf[0, 0], B[0])
    np.testing.assert_allclose(irf[0, 1], 0.5 * B[0])
    np.testing.assert_allclose(irf[0, 2], 0.5 * irf[0, 1] + 0.2 * irf[0, 0])

    print("generic BVARX smoke test passed")


if __name__ == "__main__":
    main()
