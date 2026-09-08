"""PDE reference calibration: leverage from the forward Kolmogorov density, and repricing."""
import time
from dataclasses import dataclass

import numpy as np

from ..market.bs import implied_vol
from ..simulate.dynamics import HestonParams
from ..simulate.leverage import LeverageField, Slice
from .fokker_planck import FokkerPlanck

X_LO, X_HI = np.log(0.4), np.log(2.2)


def default_x_grid(n=801, half_width=3.0, x0=0.0):
    """Uniform cell centres on [x0 - half_width, x0 + half_width] with x0 on a centre.

    Mass leaving through the outer faces is lost (absorbing boundary), and mass lost on the
    right at e^x ~ 20 costs 20 times its weight in every call price, so the domain must hold
    the whole right tail: half_width ~ 6 sigma_max sqrt(T). Check `PDEResult.forward`.
    """
    h = 2.0 * half_width / (n - 1)
    return x0 + (np.arange(n) - (n - 1) // 2) * h


def default_v_grid(params, T, n=200, v_max=None, stretch=2.0):
    """Cell centres on (0, v_max], sinh-stretched towards v = 0 (faces at v_max*sinh(c s)/sinh(c))."""
    hp = HestonParams.from_dict(params)
    if v_max is None:
        top = max(hp.v0, hp.theta)
        v_max = top + 6.0 * hp.xi * np.sqrt(top * T)
    s = np.linspace(0.0, 1.0, n + 1)
    faces = v_max * np.sinh(stretch * s) / np.sinh(stretch)
    return 0.5 * (faces[1:] + faces[:-1])


@dataclass
class PDEResult:
    field: LeverageField
    x_grid: np.ndarray
    density: np.ndarray
    snapshots: dict
    mass: float
    forward: float
    runtime_s: float
    dx: np.ndarray


def solve_leverage_pde(local_vol, params, *, s0=1.0, T, n_steps, x_grid, v_grid,
                       n_substeps=2, scheme="cn", L_max=4.0, snapshot_times=()):
    """Explicit-in-leverage calibration with the density evolved by the Fokker-Planck equation.

    Mirrors `calibrate_explicit`: at step k the leverage is built from sigma_Dup and E[V | X]
    at t = k dt and frozen over the step (an O(dt) scheme bias shared with the particle
    method). Step 0 uses the same one-point slice as the particle scheme; later slices live
    on the resolved part of `x_grid` (marginal density above 1e-6 of its peak, the analogue
    of the particle scheme's quantile grid) and are extrapolated as constants beyond it.
    """
    t0 = time.perf_counter()
    hp = HestonParams.from_dict(params)
    fp = FokkerPlanck(x_grid, v_grid, hp)
    dt = T / n_steps
    x0 = np.log(s0)
    m = fp.initial(x0, hp.v0)
    snap_steps = {int(round(t / dt)): t for t in snapshot_times}  # noqa: RUF046
    slices, snapshots = [], {}
    for k in range(n_steps):
        t = k * dt
        if k == 0:
            grid, f_grid = np.array([x0]), np.array([hp.v0])
        else:
            cells = fp.resolved(m)
            grid, f_grid = fp.x[cells], fp.cond_mean_v(m, cells)
        f_grid = np.clip(f_grid, 1e-4, None)
        sig = local_vol.sigma(max(t, local_vol.T_grid[0]), np.exp(grid), s0)
        L_grid = np.clip(np.asarray(sig, dtype=float) / np.sqrt(f_grid), 0.0, L_max)
        slices.append(Slice(t, grid.copy(), L_grid.copy(), f_grid.copy()))
        A = fp.operator(L_grid[0] if len(grid) == 1 else np.interp(fp.x, grid, L_grid))
        m = fp.advance(m, A, dt, n_substeps, first=(k == 0), scheme=scheme)
        if k + 1 in snap_steps:
            snapshots[snap_steps[k + 1]] = fp.marginal(m)
    density = fp.marginal(m)
    forward = float(np.sum(density * (np.exp(fp.x_faces[1:]) - np.exp(fp.x_faces[:-1]))))
    return PDEResult(LeverageField(slices), fp.x.copy(), density, snapshots, fp.mass(m), forward,
                     time.perf_counter() - t0, fp.dx.copy())


def call_prices(x_grid, density, K, s0=1.0, dx=None):
    """Calls from a cell-averaged marginal density of ln(S/s0): exact integral per cell."""
    x = np.asarray(x_grid, dtype=float)
    faces = np.empty(len(x) + 1)
    if dx is None:
        faces[1:-1] = 0.5 * (x[1:] + x[:-1])
        faces[0], faces[-1] = x[0] - (faces[1] - x[0]), x[-1] + (x[-1] - faces[-2])
    else:
        faces[0] = x[0] - 0.5 * dx[0]
        faces[1:] = faces[0] + np.cumsum(dx)
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))
    lnk = np.log(K_arr / s0)[:, None]
    lo = np.maximum(faces[None, :-1], lnk)
    hi = faces[None, 1:]
    seg = np.where(hi > lo, (np.exp(hi) - np.exp(lo)) - (K_arr[:, None] / s0) * (hi - lo), 0.0)
    price = s0 * (seg @ np.asarray(density, dtype=float))
    return price if np.ndim(K) else float(price[0])


def implied_vols(x_grid, density, k_grid, T, s0=1.0, dx=None):
    K = s0 * np.exp(np.asarray(k_grid, dtype=float))
    prices = call_prices(x_grid, density, K, s0, dx)
    return np.array([implied_vol(p, s0, kk, T) for p, kk in zip(prices, K)])
