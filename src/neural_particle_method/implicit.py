from .calibrate.config import ImplicitConfig
from .calibrate.implicit import GlobalNet, calibrate_implicit as _new, simulate_slices  # noqa: F401
from .estimators.ridge import GlobalRidge as GlobalRidgeHead  # noqa: F401


def _simulate(L_records, params, s0, T, n_steps, n_particles, rng):
    return simulate_slices(L_records, params, s0, T, n_steps, n_particles, rng)


def calibrate_implicit(dupire, params, s0=1.0, T=1.0, n_steps=50, n_particles=50_000,
                       alpha=0.5, n_iters=6, seed=0, L_max=4.0, fit_steps=300,
                       pool_subsample=60_000, L0_records=None):
    cfg = ImplicitConfig(n_steps=n_steps, n_particles=n_particles, alpha=alpha, n_iters=n_iters,
                         L_max=L_max, fit_steps=fit_steps, pool_subsample=pool_subsample)
    r = _new(dupire, params, cfg, s0=s0, T=T, seed=seed, L0=L0_records)
    return r.field.to_records(), {"deltas": r.deltas, "fit_s": r.fit_s, "net": r.net}
