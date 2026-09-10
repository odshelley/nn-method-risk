import numpy as np

from neural_particle_method.bench.scenarios import make_registry
from neural_particle_method.calibrate.config import ExplicitConfig
from neural_particle_method.calibrate.explicit import calibrate_explicit
from neural_particle_method.estimators.nadaraya_watson import nw_estimate
from neural_particle_method.estimators.nn import V_SCALE, NNRegressor
from neural_particle_method.simulate.leverage import DEFAULT_GRID
from neural_particle_method.suite.artifacts import SliceBank
from neural_particle_method.suite.heads import (
    FeatureRidgeHead,
    RKHSHead,
    SplineHead,
    online_sweep,
    stale_field,
)

E = ExplicitConfig(n_steps=4, n_particles=600, fit_subsample=300, first_steps=5, later_steps=2)


def _body():
    sc = make_registry()["s01"]
    est = NNRegressor(seed=0, first_steps=5, later_steps=2, keep_slice_weights=True)
    r = calibrate_explicit(sc.local_vol(), sc.dynamics, est, E, s0=sc.s0, T=sc.T, seed=0)
    return sc, SliceBank.from_regressor(est), r.field


def _f0(bank, v0):
    """Slice 0 is v0 (never fitted); later slices come from the bank. Same rule as online_sweep."""
    return lambda t, g: np.full(len(g), v0) if t == 0.0 else bank.f(t, g)


def test_rkhs_head_on_zero_residual_is_zero():
    _, bank, _ = _body()
    rng = np.random.default_rng(0)
    x = rng.normal(scale=0.1, size=300)
    f = bank.f(0.5, x)
    corr = RKHSHead().correction(0.5, x, f, f, bank, np.linspace(-0.2, 0.2, 9))
    assert np.abs(corr).max() < 1e-8


def test_rkhs_head_recovers_a_smooth_residual():
    _, bank, _ = _body()
    rng = np.random.default_rng(1)
    x = rng.normal(scale=0.1, size=2000)
    f = bank.f(0.5, x)
    grid = np.linspace(-0.15, 0.15, 7)
    corr = RKHSHead().correction(0.5, x, f + 0.01 * x, f, bank, grid)
    np.testing.assert_allclose(corr, 0.01 * grid, atol=3e-3)


def test_rkhs_head_guard_floors_the_denominator():
    """A residual that drags the denominator to zero is clipped at floor_frac * f_stale."""
    _, bank, _ = _body()
    rng = np.random.default_rng(3)
    x = rng.normal(scale=0.1, size=300)
    f = bank.f(0.5, x)
    grid = np.linspace(-0.2, 0.2, 9)
    corr = RKHSHead().correction(0.5, x, 0.0 * f, f, bank, grid)
    f_grid = bank.f(0.5, grid)
    denom = f_grid + corr
    assert np.all(denom >= 0.25 * f_grid - 1e-12)
    assert np.any(np.isclose(denom, 0.25 * f_grid, atol=1e-12))


def test_spline_head_zero_residual_and_smooth_residual():
    _, bank, _ = _body()
    rng = np.random.default_rng(0)
    x = rng.normal(scale=0.1, size=300)
    f = bank.f(0.5, x)
    corr = SplineHead().correction(0.5, x, f, f, bank, np.linspace(-0.2, 0.2, 9))
    assert np.abs(corr).max() < 1e-8

    rng = np.random.default_rng(1)
    x = rng.normal(scale=0.1, size=2000)
    f = bank.f(0.5, x)
    grid = np.linspace(-0.15, 0.15, 7)
    corr = SplineHead().correction(0.5, x, f + 0.01 * x, f, bank, grid)
    np.testing.assert_allclose(corr, 0.01 * grid, atol=3e-3)


def test_spline_head_guard_floors_the_denominator():
    _, bank, _ = _body()
    rng = np.random.default_rng(3)
    x = rng.normal(scale=0.1, size=300)
    f = bank.f(0.5, x)
    grid = np.linspace(-0.2, 0.2, 9)
    corr = SplineHead().correction(0.5, x, 0.0 * f, f, bank, grid)
    f_grid = bank.f(0.5, grid)
    denom = f_grid + corr
    assert np.all(denom >= 0.25 * f_grid - 1e-12)
    assert np.any(np.isclose(denom, 0.25 * f_grid, atol=1e-12))


def test_feature_ridge_head_reproduces_offline_when_target_is_offline():
    _, bank, _ = _body()
    rng = np.random.default_rng(2)
    x = rng.normal(scale=0.1, size=2000)
    f = bank.f(0.5, x)
    grid = np.linspace(-0.15, 0.15, 7)
    corr = FeatureRidgeHead().correction(0.5, x, f, f, bank, grid)
    # softplus output vs linear readout: small but not exactly zero
    assert np.abs(corr).max() < 5e-4


def test_feature_ridge_head_recovers_a_smooth_residual():
    _, bank, _ = _body()
    rng = np.random.default_rng(1)
    x = rng.normal(scale=0.1, size=2000)
    f = bank.f(0.5, x)
    grid = np.linspace(-0.15, 0.15, 7)
    corr = FeatureRidgeHead().correction(0.5, x, f + 0.01 * x, f, bank, grid)
    np.testing.assert_allclose(corr, 0.01 * grid, atol=3e-3)


def _softplus_pre_head(t, lnx, v_plus, model, grid, lam=1e-3):
    """The pre-Gauss-Newton head: softplus^{-1} of each particle target, then ridge on z."""
    A = model.features(t, lnx)
    w0 = model.readout(t)
    z = np.log(np.expm1(np.clip(np.asarray(v_plus, dtype=float) / V_SCALE, 1e-3, None)))
    n_lam = lam * len(lnx)
    w = np.linalg.solve(A.T @ A + n_lam * np.eye(A.shape[1]), A.T @ z + n_lam * w0)
    return np.logaddexp(0.0, model.features(t, grid) @ w) * V_SCALE - model.f(t, grid)


def test_feature_ridge_head_matches_the_local_mean_of_a_truncated_noisy_target():
    """The target has an atom at zero, so E[V|X] must be fitted on the raw scale.

    The per-particle softplus^{-1} head maps every zero to the clip floor and comes out well below
    the local mean; the converged raw-scale refit matches it.
    """
    _, bank, _ = _body()
    rng = np.random.default_rng(7)
    n = 20_000
    x = rng.normal(scale=0.1, size=n)
    v_plus = np.maximum(0.04 + 0.02 * x + 0.03 * rng.standard_normal(n), 0.0)
    assert (v_plus == 0.0).mean() > 0.05          # a visible atom at exactly zero
    grid = np.linspace(-0.15, 0.15, 7)
    want = nw_estimate(x, v_plus, grid, bandwidth=0.03)
    stale = bank.f(0.5, grid)
    new = stale + FeatureRidgeHead().correction(0.5, x, v_plus, bank.f(0.5, x), bank, grid)
    np.testing.assert_allclose(new, want, atol=3e-3)
    old = stale + _softplus_pre_head(0.5, x, v_plus, bank, grid)
    assert np.abs(old - want).min() > 1e-2        # the old head is far off at every grid point


def test_stale_field_matches_calibrate_explicit_when_f_is_the_bodys_own():
    sc, bank, field = _body()
    lv = sc.local_vol()
    sf = stale_field(_f0(bank, sc.dynamics.v0), lv, sc.s0, sc.T, E.n_steps, L_max=E.L_max)
    np.testing.assert_array_equal(sf[0].f, field[0].f)
    assert len(sf) == E.n_steps
    assert all(np.array_equal(s.grid, DEFAULT_GRID) for s in sf[1:])
    for k in range(1, E.n_steps):     # same denominators, same Dupire => same leverage on the grid
        s = field[k]
        inside = (DEFAULT_GRID >= s.grid.min()) & (DEFAULT_GRID <= s.grid.max())
        got, want = sf[k].L[inside], np.interp(DEFAULT_GRID[inside], s.grid, s.L)
        np.testing.assert_allclose(got, want, rtol=0.05)


def test_online_sweep_with_no_head_is_the_stale_field():
    sc, bank, _ = _body()
    lv = sc.local_vol()
    field, fit_s = online_sweep(bank, lv, sc.dynamics, sc.s0, sc.T, E, head=None, seed=5)
    ref = stale_field(_f0(bank, sc.dynamics.v0), lv, sc.s0, sc.T, E.n_steps, L_max=E.L_max)
    assert fit_s == 0.0 and len(field) == E.n_steps
    for a, b in zip(field, ref):
        np.testing.assert_array_equal(a.L, b.L)


def test_online_sweep_with_head_runs_and_changes_f():
    sc, bank, _ = _body()
    lv = sc.local_vol()
    head = RKHSHead(n_centres=20)
    field, fit_s = online_sweep(bank, lv, sc.dynamics, sc.s0, sc.T, E, head=head, seed=5)
    ref = stale_field(_f0(bank, sc.dynamics.v0), lv, sc.s0, sc.T, E.n_steps, L_max=E.L_max)
    assert fit_s > 0 and len(field) == E.n_steps
    assert np.array_equal(field[0].f, ref[0].f)                 # slice 0 is v0 in both
    assert any(not np.array_equal(field[k].f, ref[k].f) for k in range(1, E.n_steps))
