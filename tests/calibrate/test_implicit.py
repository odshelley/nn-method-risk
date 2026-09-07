import numpy as np

from neural_particle_method.calibrate.config import ExplicitConfig, ImplicitConfig
from neural_particle_method.calibrate.explicit import calibrate_explicit
from neural_particle_method.calibrate.implicit import calibrate_implicit
from neural_particle_method.estimators import make_estimator

# AMENDMENT (controller-approved): xi=0.05 instead of the brief's 0.4. At the
# brief's xi=0.4, the calibrated leverage L = 0.2/sqrt(E[V|X=x]) genuinely
# varies with x under correlation for a FLAT Dupire surface, so |L-1|<0.2 in
# the mid-band is false at that vol-of-vol. With xi=0.05, V stays close to
# v0=0.04 so E[V|X] ~= 0.04 and L ~= 1 genuinely holds in the mid-band.
DYN = {"kappa": 2.0, "theta": 0.04, "xi": 0.05, "rho": -0.5, "v0": 0.04}


class FlatDupire:
    T_grid = np.array([0.004, 1.0])

    def sigma(self, t, x, s0=1.0):
        return np.full(np.shape(np.log(np.asarray(x))), 0.2)


def test_damped_update_is_contraction_on_toy():
    # scalar analogue: Phi(L) = 0.2/sqrt(f(L)) with f pinned at 0.04 -> fixed point 1
    L, alpha = 3.0, 0.5
    for _ in range(20):
        L = (1 - alpha) * L + alpha * 1.0
    assert abs(L - 1.0) < 1e-5


def test_implicit_flat_recovery_smoke():
    cfg = ImplicitConfig(n_steps=6, n_particles=8_000, n_iters=3, pool_subsample=10_000, fit_steps=150)
    r = calibrate_implicit(FlatDupire(), DYN, cfg, T=0.5, seed=1)
    assert len(r.field) == 6
    s = r.field[3]
    mid = np.abs(s.grid) < 0.3
    assert np.abs(s.L[mid] - 1.0).max() < 0.2
    assert len(r.deltas) == 3 and r.deltas[-1] <= r.deltas[0] + 0.05


def test_implicit_warm_start_from_explicit():
    # L0 (including calibrate_explicit's single-point k=0 grid) is a
    # load-bearing warm-start path with no prior coverage; exercise it end to end.
    ecfg = ExplicitConfig(n_steps=6, n_particles=4_000)
    exp_r = calibrate_explicit(FlatDupire(), DYN, make_estimator("nw"), ecfg, T=0.5, seed=2)
    icfg = ImplicitConfig(n_steps=6, n_particles=8_000, n_iters=3, pool_subsample=10_000, fit_steps=150)
    r = calibrate_implicit(FlatDupire(), DYN, icfg, T=0.5, seed=1, L0=exp_r.field)
    assert len(r.field) == 6
    for s in r.field:
        assert np.all(np.isfinite(s.L))
    assert len(r.deltas) == 3


# v0 != theta so E[V|X] is genuinely time-varying (via deterministic mean
# reversion) while xi stays low so it is only weakly x-dependent; sigma is
# flat in both t and x, so all of the time signal comes from f, not sigma.
DYN_TV = {"kappa": 2.0, "theta": 0.03, "xi": 0.05, "rho": -0.5, "v0": 0.09}


class FlatSigma:
    T_grid = np.array([0.004, 1.0])

    def sigma(self, t, x, s0=1.0):
        return np.full(np.shape(np.log(np.asarray(x))), 0.2)


def test_implicit_same_time_pairing_regression():
    # Regression for a time-offset bug: the damped update paired sigma(t, .)
    # with the net evaluated at (t+dt)/T instead of t/T, so leverage was
    # computed from tomorrow's E[V|X] instead of today's. Under DYN_TV, V
    # decays deterministically from v0 to theta as V_det(t) = theta +
    # (v0-theta)*exp(-kappa*t), so the true leverage at record time t is
    # L(t) ~= 0.2 / sqrt(V_det(t)). A same-time evaluation should track this
    # target more closely, summed over interior slices, than a
    # one-step-ahead evaluation does.
    T, n_steps = 0.5, 6
    dt = T / n_steps
    cfg = ImplicitConfig(n_steps=n_steps, n_particles=8_000, n_iters=3, pool_subsample=10_000, fit_steps=150)
    r = calibrate_implicit(FlatSigma(), DYN_TV, cfg, T=T, seed=1)

    def target(t):
        v_det = DYN_TV["theta"] + (DYN_TV["v0"] - DYN_TV["theta"]) * np.exp(-DYN_TV["kappa"] * t)
        return 0.2 / np.sqrt(v_det)

    err_same, err_next = 0.0, 0.0
    for k in range(1, n_steps - 1):  # interior slices only
        s = r.field[k]
        mid = np.abs(s.grid) < 0.3
        Lk = s.L[mid].mean()
        err_same += abs(Lk - target(s.t))
        err_next += abs(Lk - target(s.t + dt))
    assert err_same < err_next
