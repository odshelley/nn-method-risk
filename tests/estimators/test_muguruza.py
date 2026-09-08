import numpy as np
import pytest

from neural_particle_method.calibrate.config import ExplicitConfig
from neural_particle_method.calibrate.explicit import StepContext, calibrate_explicit
from neural_particle_method.estimators import make_estimator
from neural_particle_method.estimators.muguruza import ConditionalMC
from neural_particle_method.simulate.dynamics import HestonParams
from neural_particle_method.simulate.stepper import heston_step

DYN = HestonParams(2.0, 0.04, 0.3, -0.5, 0.04)


def _ctx(lnx_prev, v_prev, L, zb, zp, dt=0.1, params=DYN, theta_p=None):
    return StepContext(np.asarray(lnx_prev, float), np.asarray(v_prev, float), np.asarray(L, float),
                       np.asarray(zb, float), np.asarray(zp, float), theta_p, dt, params)


def test_hand_computed_three_particles():
    ctx = _ctx([0.0, 0.1, -0.1], [0.04, 0.05, 0.03], [1.0, 1.2, 0.8], [0.5, -1.0, 0.2], [0.0, 0.0, 0.0])
    lnx_now, _ = heston_step(ctx.lnx_prev, ctx.v_prev, ctx.L_p, ctx.zb, ctx.zp, DYN, 0.1, np.sqrt(0.1))
    v_now = np.array([0.041, 0.052, 0.029])
    x = np.array([0.02])
    rho = DYN.rho
    mu = ctx.lnx_prev - 0.5 * ctx.L_p ** 2 * ctx.v_prev * 0.1 + ctx.L_p * np.sqrt(ctx.v_prev) * rho * ctx.zb * np.sqrt(0.1)
    s2 = (1 - rho ** 2) * ctx.L_p ** 2 * ctx.v_prev * 0.1
    phi = np.exp(-0.5 * (x[0] - mu) ** 2 / s2) / np.sqrt(s2)
    expected = (phi * v_now).sum() / phi.sum()
    got = ConditionalMC().fit_predict(0.1, lnx_now, v_now, x, ctx=ctx)
    assert abs(got[0] - expected) < 1e-12


def test_zero_variance_particles_are_dropped():
    ctx = _ctx([0.0, 0.0], [0.04, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0])
    f = ConditionalMC().fit_predict(0.1, np.zeros(2), np.array([0.05, 0.9]), np.array([0.0]), ctx=ctx)
    assert abs(f[0] - 0.05) < 1e-12


def test_weights_and_tilt_enter():
    ctx = _ctx([0.0, 0.0], [0.04, 0.04], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0])
    est = ConditionalMC()
    f = est.fit_predict(0.1, np.zeros(2), np.array([0.02, 0.06]), np.array([0.0]), weights=np.array([3.0, 1.0]), ctx=ctx)
    assert abs(f[0] - 0.03) < 1e-12
    tilted = _ctx([0.0, 0.0], [0.04, 0.04], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], theta_p=np.array([0.0, 5.0]))
    g = est.fit_predict(0.1, np.zeros(2), np.array([0.02, 0.06]), np.array([0.0]), ctx=tilted)
    assert g[0] < 0.04


def test_requires_ctx():
    with pytest.raises(ValueError, match="StepContext"):
        ConditionalMC().fit_predict(0.1, np.zeros(3), np.zeros(3), np.zeros(2))


class FlatDupire:
    T_grid = np.array([0.004, 1.0])

    def sigma(self, t, x, s0=1.0):
        return np.full(np.shape(np.log(np.asarray(x))), 0.2)


def test_end_to_end_flat_recovery_low_vov():
    dyn = HestonParams(2.0, 0.04, 0.05, -0.5, 0.04)
    r = calibrate_explicit(FlatDupire(), dyn, make_estimator("muguruza"),
                           ExplicitConfig(n_steps=8, n_particles=20_000, fit_subsample=20_000), T=0.5, seed=3)
    s = r.field[-1]
    mid = np.abs(s.grid) < 0.3
    assert np.abs(s.L[mid] - 1.0).max() < 0.15
