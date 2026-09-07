import numpy as np

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


def test_reprice_config_defaults():
    c = RepriceConfig()
    assert (c.n_particles, c.n_steps) == (500_000, 200) and c.as_params() == {"n_particles": 500_000, "n_steps": 200}
