"""Explicit (per-slice) neural particle calibration of a Heston-type LSV model."""
import time
from dataclasses import dataclass

import numpy as np

from ..simulate.dynamics import HestonParams
from ..simulate.leverage import DEFAULT_GRID, LeverageField, Slice
from ..simulate.stepper import heston_step
from .config import ExplicitConfig

TAIL_DX = 0.5   # log-spot distance of the tail anchor points beyond the grid ends
TAILS = ("flat", "linear", "free")


def extend_tail(grid, f_grid, tail, estimator=None, dx=TAIL_DX):
    """Append one anchor point beyond each end of `grid` so np.interp continues the slice by the
    chosen rule. "flat" returns the inputs unchanged (np.interp already holds the end values)."""
    if tail not in TAILS:
        raise ValueError(f"tail must be one of {TAILS}, got {tail!r}")
    if tail == "flat" or len(grid) < 2:
        return grid, f_grid
    lo, hi = grid[0] - dx, grid[-1] + dx
    if tail == "linear":
        f_lo = f_grid[0] + (f_grid[0] - f_grid[1]) / (grid[1] - grid[0]) * dx
        f_hi = f_grid[-1] + (f_grid[-1] - f_grid[-2]) / (grid[-1] - grid[-2]) * dx
    else:
        f_lo, f_hi = (float(x) for x in estimator.predict(np.array([lo, hi])))
    return (np.concatenate([[lo], grid, [hi]]),
            np.concatenate([[f_lo], f_grid, [f_hi]]))


@dataclass(frozen=True)
class StepContext:
    """What an estimator may know about the step that produced the current cloud (see spec)."""
    lnx_prev: np.ndarray
    v_prev: np.ndarray
    L_p: np.ndarray
    zb: np.ndarray
    zp: np.ndarray
    theta_p: np.ndarray | None
    dt: float
    params: HestonParams

    def __getitem__(self, idx):
        return StepContext(self.lnx_prev[idx], self.v_prev[idx], self.L_p[idx], self.zb[idx], self.zp[idx],
                           None if self.theta_p is None else self.theta_p[idx], self.dt, self.params)


@dataclass
class ExplicitResult:
    lnx: np.ndarray
    field: LeverageField
    fit_s: float
    snapshots: dict
    snapshot_weights: dict
    weights: np.ndarray | None
    is_diag: dict | None


def calibrate_explicit(local_vol, params, estimator, cfg=ExplicitConfig(), *,
                       s0=1.0, T=1.0, seed=0, mixture=None):
    """Single forward pass. Under a mixture the returned lnx and snapshots are proposal-distributed;
    use `weights` / `snapshot_weights` to recover physical-measure statistics."""
    if mixture is not None and not estimator.supports_weights:
        raise ValueError(f"{type(estimator).__name__} estimator does not support importance weights")
    hp = HestonParams.from_dict(params)
    wants_ctx = bool(getattr(estimator, "needs_step_context", False))
    ctx = None
    v0 = hp.v0
    n_steps, n_particles, fit_subsample, L_max = cfg.n_steps, cfg.n_particles, cfg.fit_subsample, cfg.L_max
    rng = np.random.default_rng(seed)
    dt, sdt = T / n_steps, np.sqrt(T / n_steps)

    lnx = np.full(n_particles, np.log(s0))
    v = np.full(n_particles, v0)

    slices, snapshots, snapshot_weights = [], {}, {}
    snap_steps = {int(round(t / dt)): t for t in cfg.snapshot_times}  # noqa: RUF046

    fit_s = 0.0
    theta_p = None
    ess_min = 1.0
    if mixture is not None:
        comp = rng.choice(3, size=n_particles, p=list(mixture.alphas))
        scheduled = mixture.scheduled
        thetas_all, etas_all = np.asarray(mixture.thetas), np.asarray(mixture.etas)
        if not scheduled:
            theta_p = np.array(mixture.thetas)[comp]
            etas = np.array(mixture.etas)
            eta_p = etas[comp]
        ell = np.zeros((3, n_particles))
    w = None

    for k in range(n_steps):
        if mixture is not None and scheduled:
            theta_p = thetas_all[:, k][comp]
            etas = etas_all[:, k]
            eta_p = etas[comp]
        t = k * dt
        if cfg.grid == "fixed":
            grid = DEFAULT_GRID.copy()
        else:
            qs = np.linspace(0.001, 0.999, 101)
            grid = np.quantile(lnx, qs) if k > 0 else np.array([np.log(s0)])
            grid = np.unique(grid)
        if k == 0:
            f_grid = np.full(len(grid), v0)
        else:
            idx = rng.choice(n_particles, size=min(fit_subsample, n_particles), replace=False)
            wi = None if w is None else w[idx]
            v_fit = np.maximum(v[idx], 0.0) if cfg.fit_v_floor else v[idx]
            t0 = time.perf_counter()
            if wants_ctx:
                f_grid = estimator.fit_predict(t, lnx[idx], v_fit, grid, weights=wi, ctx=ctx[idx])
            else:
                f_grid = estimator.fit_predict(t, lnx[idx], v_fit, grid, weights=wi)
            fit_s += time.perf_counter() - t0
            grid, f_grid = extend_tail(grid, f_grid, cfg.tail, estimator)
        f_grid = np.clip(f_grid, 1e-4, None)
        sig = local_vol.sigma(max(t, local_vol.T_grid[0]), np.exp(grid), s0)
        L_grid = np.clip(sig / np.sqrt(f_grid), 0.0, L_max)
        slices.append(Slice(t, grid.copy(), L_grid.copy(), f_grid.copy()))

        L_p = np.interp(lnx, grid, L_grid) if len(grid) > 1 else np.full(n_particles, L_grid[0])
        zb = rng.standard_normal(n_particles)
        zp = rng.standard_normal(n_particles)
        if wants_ctx:
            ctx = StepContext(lnx, np.maximum(v, 0.0), L_p, zb, zp, theta_p, dt, hp)
        lnx, v = heston_step(lnx, v, L_p, zb, zp, hp, dt, sdt, theta_p=theta_p)
        if mixture is not None:
            dbperp = zp * sdt + eta_p * dt
            ell += etas[:, None] * dbperp[None, :] - 0.5 * (etas ** 2)[:, None] * dt
            w = 1.0 / (np.array(mixture.alphas) @ np.exp(np.clip(ell, -60, 60)))
            ess_min = min(ess_min, float(w.sum() ** 2 / (n_particles * (w ** 2).sum())))
        if k + 1 in snap_steps:
            snapshots[snap_steps[k + 1]] = lnx.copy()
            if mixture is not None:
                snapshot_weights[snap_steps[k + 1]] = w.copy()

    is_diag = None
    if mixture is not None:
        is_diag = {"max_w": float(w.max()),
                   "ess_frac": float(w.sum() ** 2 / (len(w) * (w ** 2).sum())),
                   "ess_min_slice": ess_min}
    return ExplicitResult(lnx, LeverageField(slices), fit_s, snapshots, snapshot_weights, w, is_diag)
