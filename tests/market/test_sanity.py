"""Fast sanity checks for the pricing and Dupire pipeline."""
import numpy as np

from neural_particle_method.market.bs import bs_call, implied_vol
from neural_particle_method.market.dupire import DupireSurface
from neural_particle_method.market.heston import heston_call


def test_heston_bs_limit():
    K = np.array([0.8, 1.0, 1.25])
    hp = heston_call(K, 1.0, v0=0.04, kappa=1.0, theta=0.04, xi=1e-6, rho=0.0)
    bp = bs_call(1.0, K, 1.0, 0.2)
    assert np.allclose(hp, bp, atol=2e-6), (hp, bp)


def test_heston_deep_itm():
    p = heston_call(np.array([1e-4]), 1.0, 0.04, 2.0, 0.04, 0.3, -0.7)
    assert abs(p[0] - (1.0 - 1e-4)) < 1e-4


def test_implied_vol_roundtrip():
    p = bs_call(1.0, 1.1, 0.5, 0.25)
    assert abs(implied_vol(p, 1.0, 1.1, 0.5) - 0.25) < 1e-8


def test_dupire_flat():
    T_grid = np.linspace(0.05, 1.0, 20)
    k_grid = np.linspace(-0.5, 0.5, 33)
    w = 0.04 * T_grid[:, None] * np.ones((1, len(k_grid)))
    surf = DupireSurface(T_grid, k_grid, w)
    interior = surf.sigma_loc[2:-2, 4:-4]
    assert np.allclose(interior, 0.2, atol=1e-3), (interior.min(), interior.max())
