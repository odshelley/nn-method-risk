import numpy as np
import pytest

from neural_particle_method.bench.scenarios import (
    HestonMarketSpec,
    full_registry,
    heston_registry,
    make_registry,
)
from neural_particle_method.calibrate.config import ExplicitConfig
from neural_particle_method.calibrate.explicit import calibrate_explicit
from neural_particle_method.estimators import make_estimator
from neural_particle_method.pricing.metrics import iv_metrics
from neural_particle_method.pricing.reprice import RepriceConfig, reprice_iv
from neural_particle_method.simulate.dynamics import HestonParams


def test_registry_members_and_params():
    reg = heston_registry()
    assert list(reg) == ["li_simple", "li_complex", "bayer"]
    li = reg["li_simple"]
    assert li.market == HestonParams(1.5768, 0.0484, 0.5751, -0.7, 0.1024)
    assert li.dynamics == HestonParams(1.5768, 0.0484, 0.5751, -0.5, 0.1024)
    assert reg["li_complex"].dynamics == HestonParams(1.0, 0.0144, 0.5751, 0.0, 0.0144)
    assert reg["bayer"].market == HestonParams(2.19, 0.17023, 1.04, -0.83, 0.0045)
    assert reg["bayer"].dynamics == HestonParams(1.0, 0.0144, 0.5751, -0.9, 0.0144)
    assert li.family == "heston" and make_registry()["s01"].family == "ssvi"
    assert set(heston_registry()) <= set(full_registry()) and len(full_registry()) == 29


def test_both_specs_expose_local_vol_and_targets():
    for sc in (make_registry()["s01"], heston_registry()["li_simple"]):
        lv = sc.local_vol()
        assert hasattr(lv, "sigma") and hasattr(lv, "T_grid")
        k = np.log(np.array([0.9, 1.0, 1.1]))
        ivs = sc.target_ivs(k, list(sc.maturities))
        assert ivs.shape == (len(sc.maturities), 3) and np.all(np.isfinite(ivs)) and np.all(ivs > 0)
        p = sc.as_params()
        assert p["scenario.family"] == sc.family and "scenario.dynamics.kappa" in p


def test_heston_market_targets_have_negative_skew():
    sc = heston_registry()["li_simple"]
    k = np.linspace(-0.3, 0.3, 7)
    iv = sc.target_ivs(k, [1.0])[0]
    assert iv[0] > iv[3]
    assert abs(iv[3] - np.sqrt(0.0484)) < 0.08


def _flat():
    p = HestonParams(1.5768, 0.0484, 1e-4, -0.7, 0.1024)
    return HestonMarketSpec("flat", market=p, dynamics=p)


def test_heston_market_flat_limit_gives_unit_leverage():
    flat = _flat()
    r = calibrate_explicit(flat.local_vol(), flat.dynamics, make_estimator("nw"),
                           ExplicitConfig(n_steps=10, n_particles=20_000, fit_subsample=10_000), s0=1.0, T=1.0, seed=0)
    for s in r.field[2:]:
        mid = np.abs(s.grid) < 0.25
        assert np.abs(s.L[mid] - 1.0).max() < 0.06, s.t


@pytest.mark.slow
def test_heston_market_reprices_market_within_15bp():
    flat = _flat()
    r = calibrate_explicit(flat.local_vol(), flat.dynamics, make_estimator("nw"),
                           ExplicitConfig(n_steps=50, n_particles=100_000), s0=1.0, T=1.0, seed=0)
    k = np.log(np.geomspace(0.7, 1.4, 9))
    ivs = reprice_iv(r.field, flat.dynamics, 1.0, [1.0], k, RepriceConfig(500_000, 200), seed=1)
    m = iv_metrics(ivs, flat.target_ivs(k, [1.0]), k, [1.0])
    assert m["pooled_rmse_bp"] < 15.0, m
