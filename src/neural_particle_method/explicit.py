"""Explicit (per-slice) neural particle calibration of a Heston-type LSV model."""
import numpy as np

from .condexp import NNRegressor, nw_estimate


def calibrate_explicit(dupire, params, s0=1.0, T=1.0, n_steps=50, n_particles=200_000,
                       method="nn", fit_subsample=30_000, seed=0, L_max=4.0,
                       first_steps=400, later_steps=120, snapshot_times=()):
    """Single forward pass. Returns terminal lnX, diagnostics with L on per-slice grids,
    and lnX snapshots at requested times."""
    kappa, theta, xi, rho, v0 = (params[k] for k in ("kappa", "theta", "xi", "rho", "v0"))
    rng = np.random.default_rng(seed)
    dt, sdt = T / n_steps, np.sqrt(T / n_steps)

    lnx = np.full(n_particles, np.log(s0))
    v = np.full(n_particles, v0)
    reg = NNRegressor(seed=seed) if method == "nn" else None

    L_records, snapshots = [], {}
    snap_steps = {int(round(t / dt)): t for t in snapshot_times}

    for k in range(n_steps):
        t = k * dt
        qs = np.linspace(0.001, 0.999, 101)
        grid = np.quantile(lnx, qs) if k > 0 else np.array([np.log(s0)])
        grid = np.unique(grid)
        if k == 0:
            f_grid = np.full(len(grid), v0)
        else:
            idx = rng.choice(n_particles, size=min(fit_subsample, n_particles), replace=False)
            if method == "nn":
                reg.fit(lnx[idx], v[idx], steps=first_steps if k == 1 else later_steps)
                f_grid = reg.predict(grid)
            else:
                f_grid = nw_estimate(lnx[idx], v[idx], grid)
        f_grid = np.clip(f_grid, 1e-4, None)
        sig = dupire.sigma(max(t, dupire.T_grid[0]), np.exp(grid), s0)
        L_grid = np.clip(sig / np.sqrt(f_grid), 0.0, L_max)
        L_records.append((t, grid.copy(), L_grid.copy(), f_grid.copy()))

        L_p = np.interp(lnx, grid, L_grid) if len(grid) > 1 else np.full(n_particles, L_grid[0])
        z1 = rng.standard_normal(n_particles)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n_particles)
        vp = np.maximum(v, 0.0)
        lnx = lnx + (-0.5 * L_p ** 2 * vp) * dt + L_p * np.sqrt(vp) * sdt * z1
        v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp) * sdt * z2
        if k + 1 in snap_steps:
            snapshots[snap_steps[k + 1]] = lnx.copy()

    return lnx, {"L_records": L_records, "snapshots": snapshots}


def mc_smile(lnx_T, K_grid, s0=1.0):
    """MC call prices from terminal particles."""
    x = np.exp(lnx_T)
    return np.array([np.mean(np.maximum(x - K, 0.0)) for K in K_grid])
