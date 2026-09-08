"""Implicit scheme: global network over (t, x) with damped leverage iteration."""
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..estimators import make_estimator
from ..estimators.nn import V_SCALE, Z_SCALE
from ..simulate.dynamics import HestonParams
from ..simulate.leverage import DEFAULT_GRID, LeverageField, Slice
from ..simulate.stepper import heston_step
from .config import ImplicitConfig


class GlobalNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, tz):
        return nn.functional.softplus(self.head(self.body(tz))) * V_SCALE


def simulate_slices(field, params, s0, T, n_steps, n_particles, rng):
    """Forward Euler under a fixed leverage; returns pooled (t, lnx, v) slices."""
    hp = HestonParams.from_dict(params)
    field = LeverageField.from_records(field)
    dt, sdt = T / n_steps, np.sqrt(T / n_steps)
    lnx = np.full(n_particles, np.log(s0))
    v = np.full(n_particles, hp.v0)
    out = []
    for k in range(n_steps):
        t = k * dt
        L_p = field.at(t, lnx)
        zb = rng.standard_normal(n_particles)
        zp = rng.standard_normal(n_particles)
        lnx, v = heston_step(lnx, v, L_p, zb, zp, hp, dt, sdt)
        out.append((t + dt, lnx.copy(), np.maximum(v, 0.0).copy()))
    return out


@dataclass
class ImplicitResult:
    field: LeverageField
    deltas: list
    fit_s: float
    net: GlobalNet
    deltas_rms: list = None


def _per_slice_update(field, slices, cfg, alpha_n, local_vol, s0, v0, L_max, rng):
    """Damped update with a per-slice kernel fit. The field slice at t_k is fitted on the cloud at t_k
    ("causal": Lemma 1, must reproduce the explicit scheme), at t_{k+1} ("next"), or their average ("mid")."""
    est = make_estimator("nw")
    clouds = {round(t, 12): (x, v) for t, x, v in slices}   # data at t_1 .. t_K

    def fit(t, grid):
        key = round(t, 12)
        if key not in clouds:
            return np.full(len(grid), v0)
        x, v = clouds[key]
        idx = rng.choice(len(x), size=min(cfg.nw_subsample, len(x)), replace=False)
        return est.fit_predict(t, x[idx], v[idx], grid)

    dt = slices[0][0] - (field[0].t if len(field) else 0.0)
    new_slices, sup, sq, cnt = [], 0.0, 0.0, 0
    for s in field:
        t, g, Lg = s.t, s.grid, s.L
        if cfg.nw_shift == "causal":
            f = fit(t, g)
        elif cfg.nw_shift == "next":
            f = fit(t + dt, g)
        else:
            f = 0.5 * (fit(t, g) + fit(t + dt, g))
        f = np.clip(f, 1e-4, None)
        sig = local_vol.sigma(max(t, local_vol.T_grid[0]), np.exp(g), s0)
        Phi = np.clip(sig / np.sqrt(f), 0.0, L_max)
        L_new = (1 - alpha_n) * Lg + alpha_n * Phi
        sup = max(sup, float(np.abs(L_new - Lg).max()))
        sq += float(((L_new - Lg) ** 2).sum()); cnt += len(g)
        new_slices.append(Slice(t, g, L_new, f))
    return LeverageField(new_slices), sup, float(np.sqrt(sq / cnt))


def calibrate_implicit(local_vol, params, cfg=ImplicitConfig(), *, s0=1.0, T=1.0, seed=0, L0=None):
    hp = HestonParams.from_dict(params)
    n_steps, n_particles, alpha, L_max = cfg.n_steps, cfg.n_particles, cfg.alpha, cfg.L_max
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    grid = DEFAULT_GRID.copy()
    dt = T / n_steps
    v0 = hp.v0
    if L0 is None:
        sig0 = local_vol.sigma(local_vol.T_grid[0], np.exp(grid), s0)
        L0g = np.clip(sig0 / np.sqrt(v0), 0.0, L_max)
        field = LeverageField([Slice(k * dt, grid.copy(), L0g.copy(), np.full(len(grid), v0))
                               for k in range(n_steps)])
    else:
        field = LeverageField.from_records(L0).resample(grid)
    net = GlobalNet()
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    deltas, deltas_rms, fit_s, history = [], [], 0.0, []
    for n_it in range(cfg.n_iters):
        alpha_n = alpha / (1.0 + cfg.alpha_decay * n_it)
        if cfg.reinit_net:
            torch.manual_seed(seed)
            net = GlobalNet()
            opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        it_rng = np.random.default_rng(seed + 1) if cfg.crn else rng
        slices = simulate_slices(field, hp, s0, T, n_steps, n_particles, it_rng)
        if cfg.estimator == "nw":
            field, sup, rms = _per_slice_update(field, slices, cfg, alpha_n, local_vol, s0, v0, L_max, it_rng)
            deltas.append(sup); deltas_rms.append(rms)
            if cfg.average_last:
                history.append(list(field))
            continue
        ts = np.concatenate([np.full(len(x), t) for t, x, _ in slices])
        xs = np.concatenate([x for _, x, _ in slices])
        vs = np.concatenate([v for _, _, v in slices])
        idx = it_rng.choice(len(ts), size=min(cfg.pool_subsample, len(ts)), replace=False)
        tz = torch.tensor(np.stack([ts[idx] / max(T, 1e-9), xs[idx] / Z_SCALE], axis=1),
                          dtype=torch.float32)
        tv = torch.tensor(vs[idx][:, None], dtype=torch.float32)
        t0 = time.perf_counter()
        for _ in range(cfg.fit_steps):
            opt.zero_grad()
            loss = ((net(tz) - tv) ** 2).mean()
            loss.backward()
            opt.step()
        fit_s += time.perf_counter() - t0
        new_slices, sup, sq, cnt = [], 0.0, 0.0, 0
        with torch.no_grad():
            for s in field:
                t, g, Lg = s.t, s.grid, s.L
                tzg = torch.tensor(np.stack([np.full(len(g), t / max(T, 1e-9)),
                                             g / Z_SCALE], axis=1), dtype=torch.float32)
                f = np.clip(net(tzg).numpy()[:, 0], 1e-4, None)
                sig = local_vol.sigma(max(t, local_vol.T_grid[0]), np.exp(g), s0)
                Phi = np.clip(sig / np.sqrt(f), 0.0, L_max)
                L_new = (1 - alpha_n) * Lg + alpha_n * Phi
                sup = max(sup, float(np.abs(L_new - Lg).max()))
                sq += float(((L_new - Lg) ** 2).sum()); cnt += len(g)
                new_slices.append(Slice(t, g, L_new, f))
        field = LeverageField(new_slices)
        deltas.append(sup)
        deltas_rms.append(float(np.sqrt(sq / cnt)))
        if cfg.average_last:
            history.append(new_slices)
    if cfg.average_last:
        last = history[-cfg.average_last:]
        field = LeverageField([Slice(sl[0].t, sl[0].grid, np.mean([x.L for x in sl], axis=0), np.mean([x.f for x in sl], axis=0))
                               for sl in zip(*last)])
    return ImplicitResult(field, deltas, fit_s, net, deltas_rms)
