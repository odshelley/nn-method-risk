import numpy as np
import pytest

from neural_particle_method.bench.scenarios import make_registry
from neural_particle_method.calibrate.config import ExplicitConfig
from neural_particle_method.calibrate.explicit import calibrate_explicit
from neural_particle_method.calibrate.importance import (
    DESIGNS,
    TiltDesign,
    design_by_name,
    design_mixture,
)
from neural_particle_method.estimators import make_estimator


def _cost(thetas, dt, rho):
    th = np.asarray(thetas, dtype=float)
    if th.ndim == 1:
        th = th[:, None] * np.ones((3, int(round(1.0 / dt))))  # noqa: RUF046
    return float((th[2] ** 2).sum() * dt / (1 - rho ** 2))


@pytest.mark.parametrize("sid", ["s01", "s11"])
def test_constant_cost_3_reproduces_design_mixture(sid):
    sc = make_registry()[sid]
    ref = design_mixture(sc.dynamics.to_dict(), sc.T)
    got = TiltDesign("constant", 3.0).mixture(200, sc.T, sc.dynamics.rho)
    assert got.alphas == ref.alphas and got.rho == ref.rho
    np.testing.assert_allclose(np.asarray(got.thetas), np.asarray(ref.thetas), rtol=1e-12)


def test_inverse_sqrt_schedule_has_the_same_total_cost():
    n, T, rho = 200, 2.0, -0.6
    d = TiltDesign("inverse_sqrt", 3.0).mixture(n, T, rho)
    th = np.asarray(d.thetas)
    assert th.shape == (3, n) and np.all(th[1] == 0.0) and np.all(th[2] > 0) and np.all(th[0] < 0)
    assert th[2, 0] == th[2, 1]                     # max(t, dt) at k = 0 equals dt
    assert th[2, 1] > th[2, -1]                     # decays in t
    assert abs((th[2] ** 2).sum() * (T / n) / (1 - rho ** 2) - 3.0) < 1e-12
    c = TiltDesign("constant", 3.0).mixture(n, T, rho)
    assert abs(c.thetas[2] ** 2 * T / (1 - rho ** 2) - 3.0) < 1e-12


def test_names_and_lookup():
    assert [d.name for d in DESIGNS] == ["constant-1", "constant-3", "constant-9",
                                         "inverse_sqrt-1", "inverse_sqrt-3", "inverse_sqrt-9",
                                         "front-3", "front-8"]


def test_front_schedule_spends_the_whole_cost_before_t0():
    n, T, rho = 200, 2.0, -0.6
    d = TiltDesign("front", 8.0).mixture(n, T, rho)
    th = np.asarray(d.thetas)
    dt = T / n
    on = th[2] > 0
    assert on.sum() == 25 and not np.any(th[2][25:])           # 0.25 / 0.01 steps
    assert abs((th[2] ** 2).sum() * dt / (1 - rho ** 2) - 8.0) < 1e-12
    assert th[2][0] == np.sqrt(8.0 * (1 - rho ** 2) / 0.25)
    assert design_by_name("none") is None
    assert design_by_name("inverse_sqrt-9") == TiltDesign("inverse_sqrt", 9.0)
    with pytest.raises(KeyError):
        design_by_name("constant-2")
    with pytest.raises(ValueError):
        TiltDesign("linear", 1.0).mixture(4, 1.0, -0.5)


def test_scheduled_and_constant_mixtures_agree_when_the_schedule_is_flat():
    sc = make_registry()["s01"]
    cfg = ExplicitConfig(n_steps=6, n_particles=3_000, fit_subsample=1_000)
    const = TiltDesign("constant", 3.0).mixture(6, sc.T, sc.dynamics.rho)
    flat = type(const)(const.alphas, np.asarray(const.thetas)[:, None] * np.ones((3, 6)), const.rho)
    a = calibrate_explicit(sc.local_vol(), sc.dynamics, make_estimator("nw"), cfg, s0=sc.s0,
                           T=sc.T, seed=1, mixture=const)
    b = calibrate_explicit(sc.local_vol(), sc.dynamics, make_estimator("nw"), cfg, s0=sc.s0,
                           T=sc.T, seed=1, mixture=flat)
    np.testing.assert_array_equal(a.weights, b.weights)
    for sa, sb in zip(a.field, b.field):
        np.testing.assert_array_equal(sa.L, sb.L)
    assert 0 < a.is_diag["ess_min_slice"] <= 1.0
    assert a.is_diag["ess_min_slice"] <= a.is_diag["ess_frac"] + 1e-12


def test_bit_for_bit_against_design_mixture_on_a_tiny_run():
    sc = make_registry()["s01"]
    cfg = ExplicitConfig(n_steps=4, n_particles=2_000, fit_subsample=600)
    ref = design_mixture(sc.dynamics.to_dict(), sc.T)
    new = TiltDesign("constant", 3.0).mixture(4, sc.T, sc.dynamics.rho)
    a = calibrate_explicit(sc.local_vol(), sc.dynamics, make_estimator("nw"), cfg, s0=sc.s0,
                           T=sc.T, seed=0, mixture=ref)
    b = calibrate_explicit(sc.local_vol(), sc.dynamics, make_estimator("nw"), cfg, s0=sc.s0,
                           T=sc.T, seed=0, mixture=new)
    np.testing.assert_array_equal(a.weights, b.weights)
    for sa, sb in zip(a.field, b.field):
        np.testing.assert_array_equal(sa.f, sb.f)
