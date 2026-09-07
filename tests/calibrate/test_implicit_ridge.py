"""Overnight-body/intraday-head configuration: frozen implicit net + per-slice ridge readout."""
import numpy as np

from neural_particle_method.calibrate.config import ExplicitConfig, ImplicitConfig
from neural_particle_method.calibrate.explicit import calibrate_explicit
from neural_particle_method.calibrate.implicit import calibrate_implicit
from neural_particle_method.estimators.ridge import GlobalRidge

DYN = {"kappa": 2.0, "theta": 0.04, "xi": 0.05, "rho": -0.5, "v0": 0.04}


class FlatDupire:
    T_grid = np.array([0.004, 1.0])

    def sigma(self, t, x, s0=1.0):
        return np.full(np.shape(np.log(np.asarray(x))), 0.2)


def test_frozen_body_ridge_recovers_flat_leverage():
    lv = FlatDupire()
    icfg = ImplicitConfig(n_steps=6, n_particles=6_000, n_iters=2, pool_subsample=8_000, fit_steps=150)
    r = calibrate_implicit(lv, DYN, icfg, T=0.5, seed=1)
    head = GlobalRidge(r.net, T=0.5)
    ecfg = ExplicitConfig(n_steps=6, n_particles=8_000, fit_subsample=4_000)
    er = calibrate_explicit(lv, DYN, head, ecfg, T=0.5, seed=2)
    s = er.field[-1]
    mid = np.abs(s.grid) < 0.3
    assert np.all(np.isfinite(s.L))
    assert np.abs(s.L[mid] - 1.0).max() < 0.2
    assert head.w_prev is not None


def test_residual_shrinkage_carries_across_slices():
    lv = FlatDupire()
    icfg = ImplicitConfig(n_steps=6, n_particles=6_000, n_iters=2, pool_subsample=8_000, fit_steps=150)
    r = calibrate_implicit(lv, DYN, icfg, T=0.5, seed=1)
    head = GlobalRidge(r.net, T=0.5)
    rng = np.random.default_rng(0)
    lnx = rng.normal(0.0, 0.1, 3_000)
    v = np.full(3_000, 0.04) + rng.normal(0, 0.002, 3_000)
    g = np.linspace(-0.3, 0.3, 21)
    f1 = head.fit_predict(0.25, lnx, v, g)
    beta1 = head.w_prev.copy()
    f2 = head.fit_predict(0.3, lnx + 0.02, v, g)
    assert not np.allclose(head.w_prev, beta1)
    assert np.all(np.isfinite(f1)) and np.all(np.isfinite(f2))
    assert np.abs(f2 - 0.04).max() < 0.02
