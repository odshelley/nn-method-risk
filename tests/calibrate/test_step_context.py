import numpy as np

from neural_particle_method.calibrate.config import ExplicitConfig
from neural_particle_method.calibrate.explicit import StepContext, calibrate_explicit
from neural_particle_method.simulate.dynamics import HestonParams
from neural_particle_method.simulate.stepper import heston_step

DYN = HestonParams(2.0, 0.04, 0.3, -0.5, 0.04)


class FlatDupire:
    T_grid = np.array([0.004, 1.0])

    def sigma(self, t, x, s0=1.0):
        return np.full(np.shape(np.log(np.asarray(x))), 0.2)


class Recorder:
    supports_weights = True
    needs_step_context = True

    def __init__(self):
        self.seen = []

    def fit_predict(self, t, lnx, v, grid, weights=None, ctx=None):
        self.seen.append((t, lnx.copy(), ctx))
        return np.full(len(grid), 0.04)


class Plain:
    supports_weights = True

    def __init__(self):
        self.calls = 0

    def fit_predict(self, t, lnx, v, grid, weights=None):   # no ctx parameter at all
        self.calls += 1
        return np.full(len(grid), 0.04)


def test_context_delivered_and_consistent():
    rec = Recorder()
    calibrate_explicit(FlatDupire(), DYN, rec, ExplicitConfig(n_steps=5, n_particles=3_000, fit_subsample=1_000), T=0.5, seed=1)
    assert len(rec.seen) == 4
    for t, lnx, ctx in rec.seen:
        assert isinstance(ctx, StepContext) and len(lnx) == 1_000
        for a in (ctx.lnx_prev, ctx.v_prev, ctx.L_p, ctx.zb, ctx.zp):
            assert a.shape == (1_000,)
        assert ctx.theta_p is None and np.isclose(ctx.dt, 0.1) and ctx.params == DYN
        assert np.all(ctx.v_prev >= 0)
        # the recorded previous state and normals reproduce the subsampled current cloud exactly
        lnx1, _ = heston_step(ctx.lnx_prev, ctx.v_prev, ctx.L_p, ctx.zb, ctx.zp, DYN, ctx.dt, np.sqrt(ctx.dt))
        np.testing.assert_array_equal(lnx1, lnx)


def test_plain_estimator_never_receives_ctx():
    est = Plain()
    calibrate_explicit(FlatDupire(), DYN, est, ExplicitConfig(n_steps=5, n_particles=3_000, fit_subsample=1_000), T=0.5, seed=1)
    assert est.calls == 4


def test_getitem_subsamples_arrays_only():
    c = StepContext(np.arange(5.0), np.arange(5.0), np.ones(5), np.zeros(5), np.zeros(5), None, 0.1, DYN)
    s = c[np.array([0, 2])]
    assert s.lnx_prev.tolist() == [0.0, 2.0] and s.dt == 0.1 and s.params is DYN and s.theta_p is None
