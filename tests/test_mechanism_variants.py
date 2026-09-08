"""Diagnostic variants used to probe why the implicit scheme beats the explicit one."""
from dataclasses import replace

import numpy as np

from neural_particle_method.bench.algos import ALGOS, MECHANISM_ALGOS, run_algo
from neural_particle_method.bench.scenarios import full_registry
from neural_particle_method.calibrate.config import ExplicitConfig, ImplicitConfig
from neural_particle_method.calibrate.explicit import calibrate_explicit
from neural_particle_method.estimators import make_estimator
from neural_particle_method.simulate.leverage import DEFAULT_GRID

TINY_E = ExplicitConfig(n_steps=4, n_particles=400, fit_subsample=200, first_steps=5, later_steps=2)
TINY_I = ImplicitConfig(n_steps=4, n_particles=400, n_iters=1, fit_steps=5, pool_subsample=300)


def test_mechanism_algos_are_not_in_registry():
    assert set(MECHANISM_ALGOS).isdisjoint(ALGOS)


def test_fixed_grid_puts_every_slice_on_default_grid():
    sc = full_registry()["s01"]
    est = make_estimator("nw", seed=0, first_steps=5, later_steps=2, local_vol=sc.local_vol(), s0=sc.s0)
    r = calibrate_explicit(sc.local_vol(), sc.dynamics, est, replace(TINY_E, grid="fixed"), s0=sc.s0, T=sc.T, seed=0)
    for s in r.field:
        assert np.array_equal(s.grid, DEFAULT_GRID)


def test_default_grid_is_quantile_and_unchanged():
    sc = full_registry()["s01"]
    est = make_estimator("nw", seed=0, first_steps=5, later_steps=2, local_vol=sc.local_vol(), s0=sc.s0)
    r = calibrate_explicit(sc.local_vol(), sc.dynamics, est, TINY_E, s0=sc.s0, T=sc.T, seed=0)
    assert len(r.field[0].grid) == 1 and len(r.field[1].grid) > 1


def test_implicit_one_runs_single_undamped_iteration():
    sc = full_registry()["s01"]
    res = run_algo("implicit_nn_1", sc, 400, 0, TINY_E, TINY_I)
    assert len(res.diagnostics["deltas"]) == 1


def test_crn_reuses_the_same_cloud_every_iteration(monkeypatch):
    from neural_particle_method.calibrate import implicit as mod
    seen = []
    real = mod.simulate_slices

    def spy(*a, **k):
        out = real(*a, **k)
        seen.append(out[-1][1].copy())
        return out
    monkeypatch.setattr(mod, "simulate_slices", spy)
    sc = full_registry()["s01"]
    cfg = replace(TINY_I, n_iters=3, fit_steps=0, alpha=1.0, crn=True)
    r = mod.calibrate_implicit(sc.local_vol(), sc.dynamics, cfg, s0=sc.s0, T=sc.T, seed=0)
    assert len(seen) == 3 and len(r.deltas_rms) == 3
    # alpha=1 and fit_steps=0: L1 = Phi = L2, so iterations 2 and 3 share the leverage; identical draws => identical cloud
    assert not np.array_equal(seen[0], seen[1]) and np.array_equal(seen[1], seen[2])


def test_average_last_returns_mean_of_final_iterates():
    from neural_particle_method.calibrate import implicit as mod
    sc = full_registry()["s01"]
    r3 = mod.calibrate_implicit(sc.local_vol(), sc.dynamics, replace(TINY_I, n_iters=3), s0=sc.s0, T=sc.T, seed=0)
    ra = mod.calibrate_implicit(sc.local_vol(), sc.dynamics, replace(TINY_I, n_iters=3, average_last=1), s0=sc.s0, T=sc.T, seed=0)
    for a, b in zip(r3.field, ra.field):
        assert np.array_equal(a.L, b.L)


def test_alpha_decay_and_reinit_run():
    from neural_particle_method.calibrate import implicit as mod
    sc = full_registry()["s01"]
    r = mod.calibrate_implicit(sc.local_vol(), sc.dynamics, replace(TINY_I, n_iters=2, alpha_decay=1.0, reinit_net=True),
                               s0=sc.s0, T=sc.T, seed=0)
    assert len(r.deltas) == 2


def test_per_slice_nw_implicit_runs_all_shifts():
    from neural_particle_method.calibrate import implicit as mod
    sc = full_registry()["s01"]
    for shift in ("causal", "next", "mid"):
        cfg = replace(TINY_I, n_iters=2, estimator="nw", nw_shift=shift, nw_subsample=200)
        r = mod.calibrate_implicit(sc.local_vol(), sc.dynamics, cfg, s0=sc.s0, T=sc.T, seed=0)
        assert len(r.deltas) == 2 and all(np.array_equal(s.grid, DEFAULT_GRID) for s in r.field)
        assert all(np.all(s.f > 0) for s in r.field)


def test_fit_v_floor_changes_the_fit_only_when_variance_goes_negative():
    sc = full_registry()["s11"]   # Feller-violating: raw Euler v goes negative
    est = make_estimator("nw", seed=0, first_steps=5, later_steps=2, local_vol=sc.local_vol(), s0=sc.s0)
    raw = calibrate_explicit(sc.local_vol(), sc.dynamics, est, replace(TINY_E, fit_v_floor=False),
                             s0=sc.s0, T=sc.T, seed=0)
    flo = calibrate_explicit(sc.local_vol(), sc.dynamics, est, replace(TINY_E, fit_v_floor=True), s0=sc.s0, T=sc.T, seed=0)
    assert np.array_equal(raw.field[0].f, flo.field[0].f)          # slice 0 is v0 in both
    assert np.all(flo.field[1].f >= raw.field[1].f - 1e-12) and not np.array_equal(raw.field[1].f, flo.field[1].f)
