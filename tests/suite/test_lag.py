import dataclasses
import math

import numpy as np

from neural_particle_method.bench.scenarios import full_registry, make_registry, quote_k_grid
from neural_particle_method.calibrate.warm import bump
from neural_particle_method.market.dupire import DupireSurface
from neural_particle_method.market.heston import heston_iv
from neural_particle_method.market.local_vol import SSVILocalVol, dupire_local_vol
from neural_particle_method.market.ssvi import implied_vol_ssvi, w_and_derivs
from neural_particle_method.suite.lag import (
    LAGS,
    Lag,
    TranslatedSSVILocalVol,
    bump_heston,
    lagged_scenario,
)


def test_lags_are_the_two_agreed_ones():
    assert [l.kind for l in LAGS] == ["surface", "surface_spot"]
    assert LAGS[0].spot_move == 0.0 and math.isclose(LAGS[1].spot_move, math.log(1.02))


def test_surface_lag_reproduces_bump_on_an_admissible_scenario():
    sc = make_registry()["s01"]
    lsc = lagged_scenario(sc, Lag("surface", 0.0))
    assert lsc.ssvi == bump(sc.ssvi) and lsc.s0 == sc.s0 and lsc.s0_ref == sc.s0
    k = quote_k_grid()
    np.testing.assert_allclose(lsc.target_ivs(k, [0.5]), [implied_vol_ssvi(bump(sc.ssvi), k, 0.5)])


def test_spot_lag_local_vol_is_dupire_of_translated_surface():
    sc = make_registry()["s01"]
    d = math.log(1.02)
    lsc = lagged_scenario(sc, Lag("surface_spot", d))
    assert math.isclose(lsc.s0, sc.s0 * math.exp(d))
    k = quote_k_grid()
    np.testing.assert_allclose(lsc.target_ivs(k, [0.25, 1.0]),
                               np.stack([implied_vol_ssvi(lsc.ssvi, k + d, t)
                                         for t in (0.25, 1.0)]))

    lv = lsc.local_vol()
    s0_new = lsc.s0
    for t in (0.25, 1.0):
        for k_new in (-0.2, 0.0, 0.2):
            x = s0_new * math.exp(k_new)
            actual = float(lv.sigma(t, x, s0_new))
            w, dwdk, d2wdk2, dwdT = w_and_derivs(lsc.ssvi, k_new + d, T=t)
            expected = float(dupire_local_vol(w, dwdk, d2wdk2, dwdT, k_new))
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    # the test would have caught the defect: the old ("evaluate the old surface at the old,
    # absolute spot") behaviour differs by more than 1e-4 here
    old_buggy = float(SSVILocalVol(lsc.ssvi, 1.0, T_max=2.0).sigma(0.25, s0_new * math.exp(-0.2),
                                                                    1.0))
    new_val = float(lv.sigma(0.25, s0_new * math.exp(-0.2), s0_new))
    assert abs(new_val - old_buggy) > 1e-4


def test_translated_ssvi_local_vol_is_identity_when_delta_is_zero():
    sc = make_registry()["s02"]
    p = sc.ssvi
    x = np.exp(np.linspace(-0.3, 0.3, 5))
    translated = TranslatedSSVILocalVol(p, 1.0, 0.0, T_max=2.0)
    base = SSVILocalVol(p, 1.0, T_max=2.0)
    np.testing.assert_array_equal(translated.sigma(0.3, x, 1.0), base.sigma(0.3, x, 1.0))


def test_bump_heston_moves_vol_level_by_one_point_and_stays_valid():
    for sid in ("li_simple", "li_complex", "bayer"):
        m = full_registry()[sid].market
        b = bump_heston(m)
        assert math.isclose(math.sqrt(b.v0), math.sqrt(m.v0) + 0.01)
        assert math.isclose(math.sqrt(b.theta), math.sqrt(m.theta) + 0.01)
        assert b.rho == min(m.rho + 0.03, -0.05)
        assert math.isclose(b.xi, 0.95 * m.xi) and b.kappa == m.kappa
        assert b.xi > 0 and abs(b.rho) < 1 and b.v0 > 0 and b.theta > 0


def test_heston_lagged_scenario_targets_and_params():
    sc = full_registry()["li_simple"]
    d = math.log(1.02)
    lsc = lagged_scenario(sc, Lag("surface_spot", d))
    k = np.array([-0.2, 0.0, 0.2])
    np.testing.assert_allclose(lsc.target_ivs(k, [1.0]),
                               [heston_iv(k + d, 1.0, bump_heston(sc.market), sc.s0)])
    p = lsc.as_params()
    assert p["lag.kind"] == "surface_spot" and math.isclose(float(p["lag.spot_move"]), d)
    assert p["s0_ref"] == sc.s0 and "bumped.v0" in p and p["scenario.family"] == "heston"
    assert lsc.family == "heston" and lsc.T == sc.T and lsc.maturities == sc.maturities


def test_heston_spot_lag_local_vol_is_a_translated_dupire_surface():
    hsc = full_registry()["li_simple"]
    d = math.log(1.02)
    hlsc = lagged_scenario(hsc, Lag("surface_spot", d))
    lv = hlsc.local_vol()
    assert isinstance(lv, DupireSurface)
    s0_new = hlsc.s0
    for k in (-0.2, 0.0, 0.2):
        val = float(np.asarray(lv.sigma(1.0, s0_new * math.exp(k), s0_new)).item())
        assert math.isfinite(val) and val > 0

    # delta == 0 pins the un-lagged path: the surface lag reproduces HestonMarketSpec.local_vol()
    # on the bumped market exactly.
    base_lsc = lagged_scenario(hsc, Lag("surface", 0.0))
    bumped_spec = dataclasses.replace(hsc, market=bump_heston(hsc.market))
    x = hsc.s0 * np.exp(np.array([-0.2, 0.0, 0.2]))
    np.testing.assert_array_equal(base_lsc.local_vol().sigma(1.0, x, hsc.s0),
                                  bumped_spec.local_vol().sigma(1.0, x, hsc.s0))
