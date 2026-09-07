"""Explicit (per-slice) neural particle calibration of a Heston-type LSV model."""
import time

import numpy as np

from .estimators import make_estimator
from .simulate.dynamics import HestonParams
from .simulate.stepper import heston_step


def calibrate_explicit(dupire, params, s0=1.0, T=1.0, n_steps=50, n_particles=200_000,
                       method="nn", fit_subsample=30_000, seed=0, L_max=4.0,
                       first_steps=400, later_steps=120, snapshot_times=(), mixture=None, regressor=None):
    """Single forward pass. Returns terminal lnX, diagnostics with L on per-slice grids,
    and lnX snapshots at requested times.

    If `mixture` is a MixtureDesign, particles are split across its components (Algorithm 4):
    the X-drift gains a component-dependent tilt on the orthogonal noise only, V is pathwise
    unchanged, and each particle accumulates a balance-heuristic importance weight that is fed
    to every conditional-expectation estimator. Under a mixture, the returned `lnx` and any
    `snapshots` are proposal- (not physical-) distributed: use `info["weights"]` /
    `info["snapshot_weights"]` to recover physical-measure statistics (e.g. via `mc_smile`).
    """
    hp = HestonParams.from_dict(params)
    v0 = hp.v0
    rng = np.random.default_rng(seed)
    dt, sdt = T / n_steps, np.sqrt(T / n_steps)

    lnx = np.full(n_particles, np.log(s0))
    v = np.full(n_particles, v0)
    if regressor is not None:
        est = regressor
    else:
        est = make_estimator(method, seed=seed, first_steps=first_steps, later_steps=later_steps)
    if mixture is not None and not est.supports_weights:
        raise ValueError(f"{type(est).__name__} estimator does not support importance weights")

    L_records, snapshots, snapshot_weights = [], {}, {}
    snap_steps = {int(round(t / dt)): t for t in snapshot_times}

    fit_s = 0.0
    if mixture is not None:
        comp = rng.choice(3, size=n_particles, p=list(mixture.alphas))
        theta_p = np.array(mixture.thetas)[comp]
        etas = np.array(mixture.etas)
        eta_p = etas[comp]
        ell = np.zeros((3, n_particles))
    w = None

    for k in range(n_steps):
        t = k * dt
        qs = np.linspace(0.001, 0.999, 101)
        grid = np.quantile(lnx, qs) if k > 0 else np.array([np.log(s0)])
        grid = np.unique(grid)
        if k == 0:
            f_grid = np.full(len(grid), v0)
        else:
            idx = rng.choice(n_particles, size=min(fit_subsample, n_particles), replace=False)
            wi = None if w is None else w[idx]
            t0 = time.perf_counter()
            f_grid = est.fit_predict(t, lnx[idx], v[idx], grid, weights=wi)
            fit_s += time.perf_counter() - t0
        f_grid = np.clip(f_grid, 1e-4, None)
        sig = dupire.sigma(max(t, dupire.T_grid[0]), np.exp(grid), s0)
        L_grid = np.clip(sig / np.sqrt(f_grid), 0.0, L_max)
        L_records.append((t, grid.copy(), L_grid.copy(), f_grid.copy()))

        L_p = np.interp(lnx, grid, L_grid) if len(grid) > 1 else np.full(n_particles, L_grid[0])
        zb = rng.standard_normal(n_particles)
        zp = rng.standard_normal(n_particles)
        lnx, v = heston_step(lnx, v, L_p, zb, zp, hp, dt, sdt,
                             theta_p=theta_p if mixture is not None else None)
        if mixture is not None:
            dbperp = zp * sdt + eta_p * dt
            ell += etas[:, None] * dbperp[None, :] - 0.5 * (etas ** 2)[:, None] * dt
            w = 1.0 / (np.array(mixture.alphas) @ np.exp(np.clip(ell, -60, 60)))
        if k + 1 in snap_steps:
            snapshots[snap_steps[k + 1]] = lnx.copy()
            if mixture is not None:
                snapshot_weights[snap_steps[k + 1]] = w.copy()

    info = {"L_records": L_records, "snapshots": snapshots, "fit_s": fit_s}
    if mixture is not None:
        info["weights"] = w
        info["snapshot_weights"] = snapshot_weights
        info["is_diag"] = {"max_w": float(w.max()),
                          "ess_frac": float(w.sum() ** 2 / (len(w) * (w ** 2).sum()))}
    return lnx, info


def mc_smile(lnx_T, K_grid, s0=1.0, weights=None):
    """MC call prices from terminal particles. Pass importance `weights` (e.g.
    `info["weights"]` from a mixture-tilted run) for a self-normalised weighted mean —
    required because under a mixture the particle cloud is proposal-, not
    physical-distributed, and a plain average is biased."""
    x = np.exp(lnx_T)
    if weights is None:
        return np.array([np.mean(np.maximum(x - K, 0.0)) for K in K_grid])
    wn = weights / weights.sum()
    return np.array([np.sum(wn * np.maximum(x - K, 0.0)) for K in K_grid])
