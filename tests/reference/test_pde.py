"""solve_leverage_pde: trivial-leverage limit, calibrated LSV repricing, pricing helpers."""
import numpy as np
import pytest

from neural_particle_method.bench.scenarios import make_registry, quote_k_grid
from neural_particle_method.market.bs import bs_call, implied_vol
from neural_particle_method.market.dupire import DupireSurface
from neural_particle_method.market.heston import heston_call
from neural_particle_method.market.local_vol import SSVILocalVol
from neural_particle_method.market.ssvi import implied_vol_ssvi
from neural_particle_method.pricing.reprice import snap_times
from neural_particle_method.reference.pde import (
    PDEResult,
    call_prices,
    default_v_grid,
    default_x_grid,
    implied_vols,
    solve_leverage_pde,
)
from neural_particle_method.simulate.dynamics import HestonParams
from neural_particle_method.simulate.leverage import LeverageField

LI = HestonParams(kappa=1.5768, theta=0.0484, xi=0.5751, rho=-0.7, v0=0.1024)
CAL = HestonParams(kappa=1.0, theta=0.06, xi=0.35, rho=-0.5, v0=0.08)
K_GRID = quote_k_grid()


class FlatVol:
    T_grid = np.array([0.0, 2.0])

    def __init__(self, s):
        self.s = s

    def sigma(self, t, x, s0=1.0):
        return np.full_like(np.asarray(x, dtype=float), self.s)


@pytest.fixture(scope="module")
def li_market():
    """Dupire surface of the Li Heston market and its exact T = 1 implied vols on the quote grid."""
    T = 1.0

    def price(K, t):
        return heston_call(K, t, LI.v0, LI.kappa, LI.theta, LI.xi, LI.rho)

    surf = DupireSurface.from_price_fn(price, 1.0, np.linspace(0.01, T, 100),
                                       np.linspace(np.log(0.25), np.log(3.0), 161))
    K = np.exp(K_GRID)
    iv = np.array([implied_vol(p, 1.0, kk, T) for p, kk in zip(price(K, T), K)])
    return surf, iv, T


def test_xi_to_zero_flat_surface_gives_unit_leverage():
    """(a) xi -> 0, v0 = theta, sigma_Dup = sqrt(theta): L = 1 up to the O(xi^2) tilt of E[V | X]."""
    p = HestonParams(kappa=1.0, theta=0.04, xi=0.01, rho=0.0, v0=0.04)
    res = solve_leverage_pde(FlatVol(0.2), p, T=1.0, n_steps=25, x_grid=default_x_grid(121, 2.0),
                             v_grid=default_v_grid(p, 1.0, 400, v_max=0.08, stretch=0.5))
    first = res.field[0]
    assert len(first.grid) == 1 and first.L[0] == 1.0 and first.f[0] == 0.04
    for s in res.field.slices[1:]:
        core = (s.grid > -0.7) & (s.grid < 0.6)
        assert np.abs(s.L[core] - 1.0).max() < 5e-3, s.t
        assert np.abs(s.L - 1.0).max() < 1e-2, s.t


@pytest.mark.slow
def test_calibrated_to_own_market_reprices_it(li_market):
    """(b) Market = Li Heston, calibrated dynamics = Li: the exact leverage is 1 and the PDE
    reprices the market. Coarse grid (nx=401 on [-3, 3], nv=100, 50 steps): measured 4.9 bp max."""
    surf, iv_mkt, T = li_market
    res = solve_leverage_pde(surf, LI, T=T, n_steps=50, x_grid=default_x_grid(401),
                             v_grid=default_v_grid(LI, T, 100))
    err_bp = (implied_vols(res.x_grid, res.density, K_GRID, T, dx=res.dx) - iv_mkt) * 1e4
    assert np.all(np.isfinite(err_bp))
    assert np.abs(err_bp).max() < 7.0, err_bp
    assert abs(res.forward - 1.0) < 1e-4 and 1.0 - res.mass < 1e-4
    for s in res.field.slices[1:]:
        if s.t >= 0.5:     # the quote range is inside the resolved density by then
            core = (s.grid > np.log(0.6)) & (s.grid < np.log(1.6))
            assert core.sum() > 50 and np.abs(s.L[core] - 1.0).max() < 0.05, s.t


@pytest.mark.slow
def test_calibrated_lsv_reprices_heston_market(li_market):
    """(b) Different calibrated dynamics: the frozen-leverage scheme's O(dt) bias dominates at
    50 steps (docs/pde_reference.md table 2). Small grid here (measured 16.5 bp max at
    nx=301, nv=80); the accuracy claim lives in the convergence tables."""
    surf, iv_mkt, T = li_market
    res = solve_leverage_pde(surf, CAL, T=T, n_steps=50, x_grid=default_x_grid(301),
                             v_grid=default_v_grid(CAL, T, 80))
    err_bp = (implied_vols(res.x_grid, res.density, K_GRID, T, dx=res.dx) - iv_mkt) * 1e4
    assert np.all(np.isfinite(err_bp))
    assert np.abs(err_bp).max() < 25.0, err_bp
    assert isinstance(res, PDEResult) and isinstance(res.field, LeverageField)
    assert len(res.field) == 50 and np.all(np.diff([s.t for s in res.field.slices]) > 0)
    widths = [s.grid[-1] - s.grid[0] for s in res.field.slices[1:]]
    assert np.all(np.diff(widths) >= 0) and widths[0] < 1.0 < widths[-1]    # resolved range grows
    for s in res.field.slices[1:]:
        assert np.isin(s.grid, res.x_grid).all() and len(s.grid) == len(s.L) == len(s.f)


@pytest.mark.slow
def test_ssvi_scenario_reprices_targets_at_snapped_maturities():
    """(b) s01 with the analytic Dupire local vol, T = 2, maturities snapped like reprice_iv.
    Short maturities carry the scheme's O(dt) bias (6 steps to T = 0.25); T >= 1 is at a few bp."""
    sc = make_registry()["s01"]
    mats = list(sc.maturities)
    n_steps = 50
    ts = snap_times(mats, n_steps, sc.T)
    res = solve_leverage_pde(SSVILocalVol(sc.ssvi), sc.dynamics, T=sc.T, n_steps=n_steps,
                             x_grid=default_x_grid(401), v_grid=default_v_grid(sc.dynamics, sc.T, 100),
                             snapshot_times=ts)
    assert set(res.snapshots) == set(ts)
    for m, t in zip(mats, ts):
        err_bp = (implied_vols(res.x_grid, res.snapshots[t], K_GRID, t, dx=res.dx)
                  - implied_vol_ssvi(sc.ssvi, K_GRID, t)) * 1e4
        assert np.all(np.isfinite(err_bp)), m
        if m >= 1.0:
            assert np.abs(err_bp).max() < 8.0, (m, err_bp)
    np.testing.assert_array_equal(res.snapshots[ts[-1]], res.density)


@pytest.mark.slow
def test_short_maturity_error_is_the_scheme_bias():
    """Refining n_steps alone shrinks the T = 0.25 error of s01 (scheme O(dt)) on the reference
    grid, where the spatial error no longer dominates (docs table 3: rms 44 -> 14 bp)."""
    sc = make_registry()["s01"]
    errs = []
    for n_steps in (6, 25):      # the steps that reach T = 0.25 at n_steps = 50 and 200 on T = 2
        res = solve_leverage_pde(SSVILocalVol(sc.ssvi), sc.dynamics, T=0.25, n_steps=n_steps,
                                 x_grid=default_x_grid(), v_grid=default_v_grid(sc.dynamics, 0.25))
        e = (implied_vols(res.x_grid, res.density, K_GRID, 0.25, dx=res.dx)
             - implied_vol_ssvi(sc.ssvi, K_GRID, 0.25)) * 1e4
        errs.append(np.sqrt(np.mean(e ** 2)))
    assert errs[1] < 0.5 * errs[0], errs


def test_call_prices_from_lognormal_density_match_black_scholes():
    sigma, T = 0.25, 1.0
    x = default_x_grid(1201, 3.0)
    h = x[1] - x[0]
    mu = -0.5 * sigma ** 2 * T
    dens = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2 * T)) / np.sqrt(2 * np.pi * sigma ** 2 * T)
    K = np.array([0.6, 0.9, 1.0, 1.3, 1.6])
    got = call_prices(x, dens, K, dx=np.full(len(x), h))
    np.testing.assert_allclose(got, bs_call(1.0, K, T, sigma), atol=5e-6)   # O(h^2) cell integration
    assert call_prices(x, dens, 1.0) == pytest.approx(bs_call(1.0, 1.0, T, sigma), abs=5e-6)


def test_default_grids():
    x = default_x_grid(201, 1.5, x0=0.3)
    assert len(x) == 201 and 0.3 in x and abs(x[0] - (0.3 - 1.5)) < 1e-12
    v = default_v_grid(LI, 1.0, 50)
    assert len(v) == 50 and v[0] > 0 and np.all(np.diff(v) > 0) and np.all(np.diff(np.diff(v)) > 0)
    with pytest.raises(ValueError):
        solve_leverage_pde(FlatVol(0.2), LI, T=1.0, n_steps=2, x_grid=x, v_grid=v[::-1])
