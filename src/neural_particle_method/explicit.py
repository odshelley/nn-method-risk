"""Shim (removed in Task 12): old calibrate_explicit signature over the new implementation."""
import numpy as np

from .calibrate.config import ExplicitConfig
from .calibrate.explicit import calibrate_explicit as _new
from .estimators import make_estimator


def calibrate_explicit(dupire, params, s0=1.0, T=1.0, n_steps=50, n_particles=200_000,
                       method="nn", fit_subsample=30_000, seed=0, L_max=4.0,
                       first_steps=400, later_steps=120, snapshot_times=(), mixture=None, regressor=None):
    est = regressor if regressor is not None else make_estimator(method, seed=seed, first_steps=first_steps,
                                                                    later_steps=later_steps)
    cfg = ExplicitConfig(n_steps=n_steps, n_particles=n_particles, fit_subsample=fit_subsample,
                         L_max=L_max, first_steps=first_steps, later_steps=later_steps,
                         snapshot_times=tuple(snapshot_times))
    r = _new(dupire, params, est, cfg, s0=s0, T=T, seed=seed, mixture=mixture)
    info = {"L_records": r.field.to_records(), "snapshots": r.snapshots, "fit_s": r.fit_s}
    if mixture is not None:
        info.update(weights=r.weights, snapshot_weights=r.snapshot_weights, is_diag=r.is_diag)
    return r.lnx, info


def mc_smile(lnx_T, K_grid, s0=1.0, weights=None):
    x = np.exp(lnx_T)
    if weights is None:
        return np.array([np.mean(np.maximum(x - K, 0.0)) for K in K_grid])
    wn = weights / weights.sum()
    return np.array([np.sum(wn * np.maximum(x - K, 0.0)) for K in K_grid])
