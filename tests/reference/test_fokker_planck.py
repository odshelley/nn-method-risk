"""Operator-level checks: conservation, the CIR marginal, and pure Heston against heston_call."""
import numpy as np
import pytest
from scipy.stats import ncx2

from neural_particle_method.market.bs import implied_vol
from neural_particle_method.market.heston import heston_call
from neural_particle_method.reference.fokker_planck import FokkerPlanck, bernoulli
from neural_particle_method.reference.pde import call_prices, default_v_grid, default_x_grid
from neural_particle_method.simulate.dynamics import HestonParams

LI = HestonParams(kappa=1.5768, theta=0.0484, xi=0.5751, rho=-0.7, v0=0.1024)   # Feller fails
FELLER = HestonParams(kappa=2.0, theta=0.06, xi=0.3, rho=-0.5, v0=0.05)          # Feller holds


def test_bernoulli_limits():
    assert bernoulli(0.0) == 1.0
    assert abs(bernoulli(1e-10) - 1.0) < 1e-9
    assert abs(bernoulli(-800.0) - 800.0) < 1e-9
    assert bernoulli(800.0) == 0.0
    w = np.array([-2.0, 0.5, 3.0])
    np.testing.assert_allclose(bernoulli(w), w / np.expm1(w))


def _run_fixed_L(params, L, T, n_steps, n_sub, nx=161, nv=80, scheme="cn"):
    fp = FokkerPlanck(default_x_grid(nx), default_v_grid(params, T, nv), params)
    m = fp.initial(0.0, params.v0)
    A = fp.operator(L)
    dt = T / n_steps
    for k in range(n_steps):
        m = fp.advance(m, A, dt, n_sub, first=(k == 0), scheme=scheme)
    return fp, m


@pytest.mark.parametrize("params", [LI, FELLER])
def test_operator_conservative_and_marginal_nonnegative(params):
    fp, m = _run_fixed_L(params, 1.0, T=1.0, n_steps=20, n_sub=2)
    A = fp.operator(np.linspace(0.5, 1.5, fp.nx))
    colsum = np.asarray(A.sum(axis=0)).reshape(fp.shape)
    assert np.abs(colsum[1:-1, :-1]).max() < 1e-9        # only outer faces let mass out
    assert 1.0 - fp.mass(m) < 2e-3                        # tail leakage through x_lo
    q = fp.marginal(m)
    assert q.min() > -1e-6 * q.max()
    assert abs(np.sum(q * fp.dx) - fp.mass(m)) < 1e-12


@pytest.mark.parametrize("params", [LI, FELLER])
def test_cir_marginal_matches_noncentral_chi_square(params):
    """With L = 0 the x-fluxes vanish and the v-marginal is the CIR transition law."""
    T = 1.0
    fp, m = _run_fixed_L(params, 0.0, T=T, n_steps=50, n_sub=2, nx=3, nv=400)
    k, th, xi, v0 = params.kappa, params.theta, params.xi, params.v0
    e = np.exp(-k * T)
    mean = th + (v0 - th) * e
    var = v0 * xi ** 2 / k * (e - e ** 2) + th * xi ** 2 / (2 * k) * (1 - e) ** 2
    mv = m.sum(axis=0)
    assert abs(mv @ fp.mean_v - mean) < 2e-3 * mean
    second = mv @ fp.mean_v2
    assert abs(second - mean ** 2 - var) < 1e-2 * var
    c = xi ** 2 * (1 - e) / (4 * k)
    df, nc = 4 * k * th / xi ** 2, v0 * e / c
    for jf in (10, 40, 120, 250):
        v_face = fp.v_faces[jf]
        exact = ncx2.cdf(v_face / c, df, nc)
        assert abs(mv[:jf].sum() - exact) < 3e-3, (v_face, mv[:jf].sum(), exact)


def _heston_iv_err_bp(params, T, nx, nv, n_steps=50, n_sub=2):
    fp, m = _run_fixed_L(params, 1.0, T=T, n_steps=n_steps, n_sub=n_sub, nx=nx, nv=nv)
    k_grid = np.log(np.geomspace(0.6, 1.6, 13))
    K = np.exp(k_grid)
    price = call_prices(fp.x, fp.marginal(m), K, dx=fp.dx)
    exact = heston_call(K, T, params.v0, params.kappa, params.theta, params.xi, params.rho)
    iv = np.array([implied_vol(p, 1.0, kk, T) for p, kk in zip(price, K)])
    iv_ex = np.array([implied_vol(p, 1.0, kk, T) for p, kk in zip(exact, K)])
    return (iv - iv_ex) * 1e4


@pytest.mark.slow
@pytest.mark.parametrize("params,tol_bp", [(LI, 8.0), (FELLER, 4.5)])
def test_pure_heston_reprices_heston_call_coarse(params, tol_bp):
    """Coarse grid (nx=401 on [-3, 3], nv=100); the reference grid is the slow test."""
    err_bp = _heston_iv_err_bp(params, 1.0, nx=401, nv=100)
    assert np.all(np.isfinite(err_bp))
    assert np.max(np.abs(err_bp)) < tol_bp, err_bp


@pytest.mark.slow
@pytest.mark.parametrize("params,tol_bp", [(LI, 2.5), (FELLER, 1.2)])
def test_pure_heston_reprices_heston_call_reference_grid(params, tol_bp):
    """Default reference grid (nx=801, nv=200)."""
    err_bp = _heston_iv_err_bp(params, 1.0, nx=801, nv=200)
    assert np.max(np.abs(err_bp)) < tol_bp, err_bp
