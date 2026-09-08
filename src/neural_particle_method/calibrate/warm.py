"""Warm-start helpers: surface bumps, body distillation, beta-space corrections, Kalman passes."""
import dataclasses
import time

import numpy as np

from ..calibrate.config import ExplicitConfig
from ..calibrate.explicit import calibrate_explicit
from ..calibrate.implicit import calibrate_implicit, simulate_slices
from ..estimators import make_estimator
from ..estimators.ridge import GlobalRidge
from ..market.ssvi import SSVIParams, no_arb_ok
from ..simulate.dynamics import HestonParams
from ..simulate.leverage import DEFAULT_GRID as GRID
from ..simulate.leverage import LeverageField, Slice

ALPHA = 0.5


def bump(p):
    q = SSVIParams(sigma0=p.sigma0 + 0.01, eta=0.95 * p.eta, gamma=p.gamma,
                   rho=min(p.rho + 0.03, -0.05))
    assert no_arb_ok(q)
    return q


def scaled_bump(p, s):
    for shrink in (1.0, 0.75, 0.5, 0.25):
        eff = s * shrink
        q = SSVIParams(sigma0=p.sigma0 + 0.01 * eff, eta=p.eta * max(1 - 0.05 * eff, 0.3),
                       gamma=p.gamma, rho=min(p.rho + 0.03 * eff, -0.05))
        if no_arb_ok(q):
            return q, eff
    raise RuntimeError("no-arb bump exhausted")


def seq_path(p, n, rng):
    """Cumulative random walk on the SSVI parameters, no-arb preserved by halving."""
    out, cur = [], p
    for _ in range(n):
        step = 1.0
        for _ in range(8):
            q = SSVIParams(sigma0=cur.sigma0 + step * rng.normal(0, 0.004),
                           eta=cur.eta * float(np.exp(step * rng.normal(0, 0.02))),
                           gamma=cur.gamma,
                           rho=float(np.clip(cur.rho + step * rng.normal(0, 0.015), -0.85, -0.05)))
            if no_arb_ok(q):
                break
            step *= 0.5
        else:
            q = cur
        out.append(q)
        cur = q
    return out


def dyn_variants(params):
    p = HestonParams.from_dict(params)
    return {
        "xi_up": HestonParams(p.kappa, p.theta, p.xi * 1.3, p.rho, p.v0),
        "rho_dn": HestonParams(p.kappa, p.theta, p.xi, max(p.rho - 0.15, -0.9), p.v0),
        "kappa_dn": HestonParams(p.kappa * 0.6, p.theta, p.xi, p.rho, p.v0),
    }


def _slice0(lv, s0, v0):
    sig0 = lv.sigma(max(0.0, lv.T_grid[0]), np.exp(GRID), s0)
    return Slice(0.0, GRID.copy(), np.clip(sig0 / np.sqrt(v0), 0.0, 4.0), np.full(len(GRID), v0))


def _slice_from_f(lv, t, f, s0):
    sig = lv.sigma(max(t, lv.T_grid[0]), np.exp(GRID), s0)
    return Slice(t, GRID.copy(), np.clip(sig / np.sqrt(f), 0.0, 4.0), f)


def distil(net, slices, lv, T, n_steps, s0, v0, sub, rng):
    """Per-slice ridge heads on a simulated cloud; same-time pairing throughout."""
    dt = T / n_steps
    head = GlobalRidge(net, T)
    betas, out = {}, [_slice0(lv, s0, v0)]
    for k in range(1, n_steps):
        t = k * dt
        _, lnx, v = slices[k - 1]
        idx = rng.choice(len(lnx), size=min(sub, len(lnx)), replace=False)
        f = np.clip(head.fit_predict(t, lnx[idx], v[idx], GRID), 1e-4, None)
        betas[k] = head.w_prev.copy()
        out.append(_slice_from_f(lv, t, f, s0))
    return betas, LeverageField(out)


def records_from_betas(net, betas, lv, T, n_steps, s0, v0):
    return records_from_head(GlobalRidge(net, T), betas, lv, T, n_steps, s0, v0)


def records_from_head(head, betas, lv, T, n_steps, s0, v0):
    """Rebuild the leverage field from a head's feature map and per-slice betas."""
    dt = T / n_steps
    out = [_slice0(lv, s0, v0)]
    for k in range(1, n_steps):
        t = k * dt
        f = np.clip(head.features(t, GRID) @ betas[k], 1e-4, None)
        out.append(_slice_from_f(lv, t, f, s0))
    return LeverageField(out)


def beta_correction(net, betas, lv, params, T, n_steps, N, s0, v0, sub, seed):
    """One damped correction in beta space: simulate under L(beta), refit heads, damp."""
    field = records_from_betas(net, betas, lv, T, n_steps, s0, v0)
    rng = np.random.default_rng(seed)
    slices = simulate_slices(field, params, s0, T, n_steps, N, rng)
    hat, _ = distil(net, slices, lv, T, n_steps, s0, v0, sub, rng)
    return {k: (1 - ALPHA) * betas[k] + ALPHA * hat[k] for k in betas}


def kalman_pass(head, betas, lv, params, T, n_steps, N, s0, v0, sub, seed):
    """Simulate under the current leverage, one Kalman measurement update per slice."""
    field = records_from_head(head, betas, lv, T, n_steps, s0, v0)
    rng = np.random.default_rng(seed)
    slices = simulate_slices(field, params, s0, T, n_steps, N, rng)
    for k in range(1, n_steps):
        t = k * (T / n_steps)
        _, lnx, v = slices[k - 1]
        idx = rng.choice(len(lnx), size=min(sub, len(lnx)), replace=False)
        head.update(k, t, lnx[idx], v[idx], GRID)
    return head.betas()


def full_resolve(lv, params, s0, T, cfg, seed, L0=None, n_iters=None):
    """Explicit warm start (unless L0 given) followed by the damped implicit solve. Returns (field, seconds)."""
    t0 = time.perf_counter()
    if L0 is None:
        ecfg = ExplicitConfig(n_steps=cfg.n_steps, n_particles=cfg.n_particles)
        L0 = calibrate_explicit(lv, params, make_estimator("nn", seed=seed, first_steps=ecfg.first_steps,
                                                            later_steps=ecfg.later_steps),
                                ecfg, s0=s0, T=T, seed=seed).field
    icfg = cfg if n_iters is None else dataclasses.replace(cfg, n_iters=n_iters)
    r = calibrate_implicit(lv, params, icfg, s0=s0, T=T, seed=seed, L0=L0)
    return r.field, time.perf_counter() - t0
