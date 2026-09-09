import math

import numpy as np

from neural_particle_method.bench.scenarios import full_registry, make_registry, quote_k_grid
from neural_particle_method.calibrate.warm import bump
from neural_particle_method.market.heston import heston_iv
from neural_particle_method.market.ssvi import implied_vol_ssvi
from neural_particle_method.suite.lag import (
    LAGS,
    Lag,
    ShiftedLocalVol,
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


def test_spot_lag_shifts_target_and_start_but_not_absolute_local_vol():
    sc = make_registry()["s01"]
    d = math.log(1.02)
    lsc = lagged_scenario(sc, Lag("surface_spot", d))
    assert math.isclose(lsc.s0, sc.s0 * math.exp(d))
    k = quote_k_grid()
    np.testing.assert_allclose(lsc.target_ivs(k, [0.25, 1.0]),
                               np.stack([implied_vol_ssvi(lsc.ssvi, k + d, t)
                                         for t in (0.25, 1.0)]))
    lv, base = lsc.local_vol(), lagged_scenario(sc, Lag("surface", 0.0)).local_vol()
    x = np.exp(np.linspace(-0.5, 0.5, 7))
    # the passed s0 is ignored: moneyness is always relative to s0_ref
    for s0 in (1.0, 1.02, 0.7):
        np.testing.assert_allclose(lv.sigma(0.5, x, s0), base.sigma(0.5, x, sc.s0))
    assert lv.t_min == base.t_min and np.array_equal(lv.T_grid, base.T_grid)


def test_shifted_local_vol_is_identity_when_ref_equals_passed_s0():
    sc = make_registry()["s02"]
    inner = sc.local_vol()
    sh = ShiftedLocalVol(inner, s0_ref=sc.s0)
    x = np.exp(np.linspace(-0.3, 0.3, 5))
    np.testing.assert_array_equal(sh.sigma(0.3, x, sc.s0), inner.sigma(0.3, x, sc.s0))


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
