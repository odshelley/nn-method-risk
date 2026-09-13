import numpy as np

from neural_particle_method.bench.scenarios import make_registry
from neural_particle_method.calibrate.importance import TiltDesign
from neural_particle_method.market.bs import bs_call
from neural_particle_method.pricing.metrics import far_wing_metrics
from neural_particle_method.pricing.reprice import (
    FAR_PRICE_FLOOR,
    RepriceConfig,
    far_k_grid,
    reprice_iv,
    reprice_iv_otm,
)
from neural_particle_method.simulate.leverage import DEFAULT_GRID, LeverageField, Slice


def _flat_field(n_steps, T):
    return LeverageField([Slice(k * T / n_steps, DEFAULT_GRID.copy(), np.ones_like(DEFAULT_GRID),
                                np.full_like(DEFAULT_GRID, 0.04)) for k in range(n_steps)])


def test_far_grid():
    k = far_k_grid()
    assert k.shape == (21,)
    assert abs(k[0] - np.log(0.45)) < 1e-12 and abs(k[-1] - np.log(2.2)) < 1e-12


def test_untilted_calls_are_bit_identical_to_reprice_iv():
    sc = make_registry()["s01"]
    cfg = RepriceConfig(4_000, 8)
    k = np.array([0.05, 0.15, 0.3])
    a = reprice_iv(_flat_field(8, sc.T), sc.dynamics, sc.s0, [0.5, 1.0], k, cfg, seed=5)
    b, prices, ess = reprice_iv_otm(_flat_field(8, sc.T), sc.dynamics, sc.s0, [0.5, 1.0], k, cfg,
                                    seed=5)
    np.testing.assert_array_equal(a, b)
    assert ess == 1.0 and prices.shape == (2, 3)


def test_put_side_inverts_through_parity():
    sc = make_registry()["s01"]
    cfg = RepriceConfig(50_000, 8)
    k = np.array([-0.3, -0.15, 0.15, 0.3])
    ivs, prices, _ = reprice_iv_otm(_flat_field(8, sc.T), sc.dynamics, sc.s0, [1.0], k, cfg, seed=1)
    # same seed and config, so both repricers see the same paths: only the inversion path differs,
    # calls at every strike here against OTM puts below the money carried across by parity
    ivs_call = reprice_iv(_flat_field(8, sc.T), sc.dynamics, sc.s0, [1.0], k, cfg, seed=1)
    assert np.all(np.isfinite(ivs))
    ok = np.isfinite(ivs) & np.isfinite(ivs_call)
    assert np.nanmax(np.abs(ivs[ok] - ivs_call[ok])) < 0.02       # put and call sides alike
    assert np.all(prices > 0) and prices[0, 0] < prices[0, 1]      # deeper put is cheaper


def test_tilted_reprice_agrees_with_untilted_and_reaches_the_wings():
    sc = make_registry()["s01"]
    cfg = RepriceConfig(200_000, 8)
    mix = TiltDesign("constant", 3.0).mixture(8, sc.T, sc.dynamics.rho)
    k = np.array([-0.6, -0.3, 0.0, 0.3, 0.6])
    u, _pu, _ = reprice_iv_otm(_flat_field(8, sc.T), sc.dynamics, sc.s0, [2.0], k, cfg, seed=2)
    t, _pt, ess = reprice_iv_otm(_flat_field(8, sc.T), sc.dynamics, sc.s0, [2.0], k, cfg, seed=2,
                                mixture=mix)
    assert 0.5 < ess < 1.0
    ok = np.isfinite(u) & np.isfinite(t)
    assert ok.sum() >= 3
    assert np.nanmax(np.abs(u[ok] - t[ok])) < 0.01          # within MC noise at 200k paths
    assert np.all(np.isfinite(t[0, [0, -1]]))                # the tilted cloud prices the far wings


def test_far_wing_metrics_drop_quotes_below_the_price_floor():
    k = far_k_grid()
    times = [0.25, 2.0]
    tgt = np.full((2, 21), 0.2)
    mod = tgt + 0.001                                          # 10 bp everywhere
    m, err = far_wing_metrics(mod, tgt, k, times, 1.0)
    K = np.exp(k)
    p = np.stack([bs_call(1.0, K, t, 0.2) for t in times])
    p = np.where(k[None, :] < 0, p - 1.0 + K[None, :], p)      # OTM price via parity
    kept = (p >= FAR_PRICE_FLOOR) & (np.abs(k)[None, :] > 0.25)
    assert m["far_wings_n"] == int(kept.sum()) and m["far_wings_n"] < 2 * 21
    assert abs(m["far_wings_mae_bp"] - 10.0) < 1e-6
    assert np.isnan(err[0, 0]) and abs(err[1, 10]) < 1e-9 + 10.0   # 3-month far put dropped
    assert m["far_n/T0.25"] < m["far_n/T2"]
    assert abs(m["far_all_mae_bp"] - 10.0) < 1e-6
