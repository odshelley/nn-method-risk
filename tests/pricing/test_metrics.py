import numpy as np
import pytest

from neural_particle_method.market.ssvi import SSVIParams, implied_vol_ssvi
from neural_particle_method.pricing.metrics import iv_metrics, target_ivs
from neural_particle_method.pricing.reprice import RepriceConfig, snap_times

P = SSVIParams(sigma0=0.2, eta=1.0, gamma=0.4, rho=-0.6)


def test_target_ivs_uses_snapped_times():
    k = np.log(np.array([0.9, 1.0, 1.1]))
    mats = [0.25, 0.5]
    got = target_ivs(P, k, mats, n_steps=25)
    ts = snap_times(mats, 25)
    exp = np.stack([implied_vol_ssvi(P, k, t) for t in ts])
    np.testing.assert_array_equal(got, exp)


def test_iv_metrics_shapes():
    k = np.log(np.array([0.7, 1.0, 1.5]))
    target = np.full((2, 3), 0.2)
    model = target + np.array([[0.001, 0.0, -0.002], [0.0, 0.0, np.nan]])
    m = iv_metrics(model, target, k, [0.5, 1.0])
    assert m["n_failed"] == 1
    assert m["pooled_max_bp"] == 20.0
    assert m["wings_rmse_bp"] > 0
    assert len(m["per_maturity"]) == 2


def test_liquid_mae_is_near_money_and_long_dated_only():
    k = np.array([-0.5, -0.1, 0.0, 0.1, 0.5])
    target = np.full((2, 5), 0.2)
    err_bp = np.array([[10.0, 20.0, 30.0, 40.0, 50.0],       # T = 0.25: never liquid
                       [1.0, 2.0, -4.0, 6.0, 100.0]])        # T = 1.0: |k| <= 0.25 counts
    model = target + err_bp / 1e4
    m = iv_metrics(model, target, k, [0.25, 1.0])
    assert m["liquid_mae_bp"] == pytest.approx((2.0 + 4.0 + 6.0) / 3)
    # additive only: every other key is what a call without the new argument returns
    ref = iv_metrics(model, target, k, [0.25, 1.0], 0.25)
    assert set(m) == set(ref) == {
        "pooled_rmse_bp", "pooled_max_bp", "wings_rmse_bp", "wings_max_bp", "n_failed",
        "per_maturity", "pooled_mae_bp", "wings_mae_bp", "mae_per_maturity", "liquid_mae_bp"}
    assert {key: v for key, v in m.items() if key != "liquid_mae_bp"} == \
           {key: v for key, v in ref.items() if key != "liquid_mae_bp"}
    assert m["pooled_mae_bp"] == pytest.approx(263.0 / 10)
    assert m["wings_mae_bp"] == pytest.approx((10.0 + 50.0 + 1.0 + 100.0) / 4)


def test_liquid_mae_is_nan_when_no_quote_qualifies():
    k = np.array([-0.5, 0.5])
    target = np.full((1, 2), 0.2)
    m = iv_metrics(target + 1e-4, target, k, [0.25])
    assert np.isnan(m["liquid_mae_bp"])


def test_reprice_config_defaults():
    c = RepriceConfig()
    assert (c.n_particles, c.n_steps) == (500_000, 200) and c.as_params() == {"n_particles": 500_000, "n_steps": 200}
