"""Online heads on a frozen offline body, and the causal one-pass sweep that applies them.

Every head implements `correction(t, lnx, v_plus, f_stale, model, grid)` returning the additive
correction to the stale denominator on `grid`, fitted from the online cloud (`lnx`, `v_plus`) and
the stale values `f_stale = model.f(t, lnx)` at the same particles.
"""
import time

import numpy as np

from ..calibrate.config import ExplicitConfig
from ..estimators.nn import V_SCALE
from ..estimators.rkhs import RKHSRidge
from ..estimators.spline import PSpline
from ..simulate.dynamics import HestonParams
from ..simulate.leverage import DEFAULT_GRID, LeverageField, Slice
from ..simulate.stepper import heston_step


def _guarded(f_grid, corr, floor_frac):
    """Additive correction clipped so the corrected denominator stays >= floor_frac * f_grid."""
    return np.maximum(f_grid + corr, floor_frac * f_grid) - f_grid


class RKHSHead:
    """Gaussian kernel ridge on the residual v+ - f_stale, kernel width scaled to the cloud,
    guarded.

    variance = (width_scale * std(lnx))^2 unless a fixed `variance` is given (legacy behaviour);
    the corrected denominator is floored at floor_frac * f_stale so the head can never push the
    leverage to its cap.
    """

    def __init__(self, n_centres=100, lam=1e-3, width_scale=0.5, variance=None, floor_frac=0.25):
        self.n_centres, self.lam, self.width_scale = n_centres, lam, width_scale
        self.variance, self.floor_frac = variance, floor_frac
        self.params = {"head": "rkhs", "n_centres": n_centres, "lam": lam,
                       "width_scale": width_scale, "variance": variance, "floor_frac": floor_frac}

    def correction(self, t, lnx, v_plus, f_stale, model, grid):
        var = self.variance if self.variance is not None else (
            self.width_scale * max(float(np.std(lnx)), 1e-3)) ** 2
        est = RKHSRidge(n_centres=self.n_centres, lam=self.lam, variance=var)
        corr = est.fit_predict(t, lnx, v_plus - f_stale, grid)
        return _guarded(model.f(t, grid), corr, self.floor_frac)


class SplineHead:
    """Penalised cubic B-spline on the residual v+ - f_stale (same P-spline as the cold estimator),
    guarded."""

    def __init__(self, n_knots=25, lam=1.0, floor_frac=0.25):
        self.n_knots, self.lam, self.floor_frac = n_knots, lam, floor_frac
        self.params = {"head": "spline", "n_knots": n_knots, "lam": lam, "floor_frac": floor_frac}

    def correction(self, t, lnx, v_plus, f_stale, model, grid):
        est = PSpline(n_knots=self.n_knots, lam=self.lam)
        corr = est.fit_predict(t, lnx, v_plus - f_stale, grid)
        return _guarded(model.f(t, grid), corr, self.floor_frac)


class FeatureRidgeHead:
    """Per-slice refit of the body's last layer, shrunk toward the offline readout.

    Gauss-Newton to convergence on the raw variance scale from the offline last layer w0, with a
    penalty expressed as a fraction of the data curvature. When the online target equals the
    body's own f the first step is zero and the correction vanishes.
    """

    def __init__(self, lam=1e-3, max_iter=20, tol=1e-9):
        self.lam, self.max_iter, self.tol = lam, max_iter, tol
        self.params = {"head": "ridge", "lam": lam, "max_iter": max_iter}

    def correction(self, t, lnx, v_plus, f_stale, model, grid):
        """Ridge refit of the body's last layer on the raw variance scale.

        Minimises sum (softplus(A w) V_SCALE - v+)^2 + lam_eff ||w - w0||^2 by Gauss-Newton from
        the offline readout w0, where lam_eff = lam * mean diagonal of the Gauss-Newton curvature
        at w0 (so `lam` is a fraction of the data curvature, independent of V_SCALE and n).
        """
        A = model.features(t, lnx)
        w0 = model.readout(t)
        w = w0.copy()
        # `features` and `readout` expose the body and the head only, so a model carrying a
        # post-hoc output scale (a mean-matched SliceBank) needs it applied here too; without it
        # a converged refit would return f * (1 - scale) instead of zero. 1.0 for models with none.
        sc = getattr(model, "scale", None)
        sc = 1.0 if sc is None else float(sc(t))
        eye = np.eye(A.shape[1])
        lam_eff = None
        for _ in range(self.max_iter):
            z = A @ w
            s = 1.0 / (1.0 + np.exp(-z))
            f = np.logaddexp(0.0, z) * V_SCALE * sc
            B = (V_SCALE * sc * s)[:, None] * A
            H = B.T @ B
            if lam_eff is None:
                lam_eff = self.lam * np.trace(H) / A.shape[1]
            d = np.linalg.solve(H + lam_eff * eye, B.T @ (v_plus - f) - lam_eff * (w - w0))
            w = w + d
            if np.abs(d).max() < self.tol:
                break
        Ag = model.features(t, grid)
        return np.logaddexp(0.0, Ag @ w) * V_SCALE * sc - model.f(t, grid)


def stale_field(model_f, local_vol, s0, T, n_steps, L_max=4.0, grid=DEFAULT_GRID):
    """Stale denominators, fresh Dupire: no particles needed.

    Slice 0 is the one-point slice at log(s0); the caller's `model_f` must return v0 there (the
    explicit pass never fits slice 0). `online_sweep` wraps the body accordingly."""
    dt = T / n_steps
    slices = []
    for k in range(n_steps):
        t = k * dt
        if k == 0:
            g = np.array([np.log(s0)])
            f = np.clip(model_f(t, g), 1e-4, None)
        else:
            g = grid.copy()
            f = np.clip(model_f(t, g), 1e-4, None)
        sig = local_vol.sigma(max(t, local_vol.T_grid[0]), np.exp(g), s0)
        slices.append(Slice(t, g, np.clip(sig / np.sqrt(f), 0.0, L_max), f))
    return LeverageField(slices)


def online_sweep(model, local_vol, params, s0, T, cfg=ExplicitConfig(), head=None, seed=0):
    """calibrate_explicit with the estimator replaced by "stale + head correction".

    Returns (field, fit_s) where fit_s is the head time only. `head=None` is the stale-f,
    fresh-Dupire method and costs nothing: it returns `stale_field` on the fixed grid without
    simulating.
    """
    hp = HestonParams.from_dict(params)
    if head is None:
        def f0(t, g):
            return np.full(len(g), hp.v0) if t == 0.0 else model.f(t, g)
        return stale_field(f0, local_vol, s0, T, cfg.n_steps, L_max=cfg.L_max), 0.0
    n_steps, n_particles = cfg.n_steps, cfg.n_particles
    fit_subsample, L_max = cfg.fit_subsample, cfg.L_max
    rng = np.random.default_rng(seed)
    dt, sdt = T / n_steps, np.sqrt(T / n_steps)
    lnx = np.full(n_particles, np.log(s0))
    v = np.full(n_particles, hp.v0)
    slices, fit_s = [], 0.0
    for k in range(n_steps):
        t = k * dt
        if k == 0:
            grid, f_grid = np.array([np.log(s0)]), np.array([hp.v0])
        else:
            grid = np.unique(np.quantile(lnx, np.linspace(0.001, 0.999, 101)))
            idx = rng.choice(n_particles, size=min(fit_subsample, n_particles), replace=False)
            x_sub, v_sub = lnx[idx], np.maximum(v[idx], 0.0)
            t0 = time.perf_counter()
            f_stale = model.f(t, x_sub)
            corr = head.correction(t, x_sub, v_sub, f_stale, model, grid)
            f_grid = model.f(t, grid) + corr
            fit_s += time.perf_counter() - t0
        f_grid = np.clip(f_grid, 1e-4, None)
        sig = local_vol.sigma(max(t, local_vol.T_grid[0]), np.exp(grid), s0)
        L_grid = np.clip(sig / np.sqrt(f_grid), 0.0, L_max)
        slices.append(Slice(t, grid.copy(), L_grid.copy(), f_grid.copy()))
        L_p = np.interp(lnx, grid, L_grid) if len(grid) > 1 else np.full(n_particles, L_grid[0])
        zb, zp = rng.standard_normal(n_particles), rng.standard_normal(n_particles)
        lnx, v = heston_step(lnx, v, L_p, zb, zp, hp, dt, sdt)
    return LeverageField(slices), fit_s
