import numpy as np

from neural_particle_method.market.dupire import DupireSurface
from neural_particle_method.market.local_vol import SSVILocalVol, dupire_local_vol
from neural_particle_method.market.ssvi import SSVIParams, total_variance, w_and_derivs

P = SSVIParams(sigma0=0.2, eta=1.0, gamma=0.4, rho=-0.6)


def test_dupire_local_vol_flat_surface_is_flat():
    k = np.linspace(-0.5, 0.5, 11)
    w = np.full_like(k, 0.04)          # w = sigma^2 T with sigma = 0.2, T = 1
    sig = dupire_local_vol(w, np.zeros_like(k), np.zeros_like(k), np.full_like(k, 0.04), k)
    assert np.allclose(sig, 0.2)


def test_ssvi_local_vol_uses_shared_formula():
    lv = SSVILocalVol(P)
    k = np.array([-0.2, 0.0, 0.3])
    w, dwdk, d2wdk2, dwdT = w_and_derivs(P, k, 1.0)
    expected = dupire_local_vol(w, dwdk, d2wdk2, dwdT, k)
    np.testing.assert_array_equal(lv.sigma(1.0, np.exp(k)), expected)


def test_grid_surface_agrees_with_analytic_on_same_ssvi():
    T_grid = np.linspace(0.1, 1.5, 29)
    k_grid = np.linspace(-0.6, 0.6, 61)
    w = np.stack([total_variance(P, k_grid, T) for T in T_grid])
    grid = DupireSurface(T_grid, k_grid, w)
    analytic = SSVILocalVol(P)
    for t in (0.5, 1.0):
        for k in (-0.2, 0.0, 0.2):
            ours = grid.sigma(t, np.exp(k)).item()
            theirs = analytic.sigma(t, np.exp(k)).item()
            assert abs(ours - theirs) / theirs < 0.05, (t, k, ours, theirs)


def test_t_min_property():
    assert SSVILocalVol(P).t_min == SSVILocalVol(P).T_grid[0]
