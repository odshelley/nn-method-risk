"""Implicit scheme: global network over (t, x) with damped leverage iteration."""
import time

import numpy as np
import torch
import torch.nn as nn

from .condexp import V_SCALE, Z_SCALE
from .reprice import L_lookup


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


def _simulate(L_records, params, s0, T, n_steps, n_particles, rng):
    """Forward Euler under a fixed leverage; returns pooled (t, lnx, v) slices."""
    kappa, theta, xi, rho, v0 = (params[k] for k in ("kappa", "theta", "xi", "rho", "v0"))
    L = L_lookup(L_records)
    dt, sdt = T / n_steps, np.sqrt(T / n_steps)
    lnx = np.full(n_particles, np.log(s0))
    v = np.full(n_particles, v0)
    out = []
    for k in range(n_steps):
        t = k * dt
        L_p = L(t, lnx)
        zb = rng.standard_normal(n_particles)
        zp = rng.standard_normal(n_particles)
        z1 = rho * zb + np.sqrt(1 - rho ** 2) * zp
        vp = np.maximum(v, 0.0)
        lnx = lnx + (-0.5 * L_p ** 2 * vp) * dt + L_p * np.sqrt(vp) * sdt * z1
        v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp) * sdt * zb
        out.append((t + dt, lnx.copy(), np.maximum(v, 0.0).copy()))
    return out


def calibrate_implicit(dupire, params, s0=1.0, T=1.0, n_steps=50, n_particles=50_000,
                       alpha=0.5, n_iters=6, seed=0, L_max=4.0, fit_steps=300,
                       pool_subsample=60_000, L0_records=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    grid = np.linspace(np.log(0.4), np.log(2.2), 81)
    dt = T / n_steps
    v0 = params["v0"]
    if L0_records is None:
        sig0 = dupire.sigma(dupire.T_grid[0], np.exp(grid), s0)
        L0 = np.clip(sig0 / np.sqrt(v0), 0.0, L_max)
        L_records = [(k * dt, grid.copy(), L0.copy(), np.full(len(grid), v0)) for k in range(n_steps)]
    else:
        L_records = [(t, grid.copy(),
                      np.interp(grid, g, Lg) if len(g) > 1 else np.full(len(grid), Lg[0]),
                      np.interp(grid, g, fg) if len(g) > 1 else np.full(len(grid), fg[0]))
                     for (t, g, Lg, fg) in L0_records]
    net = GlobalNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    deltas, fit_s = [], 0.0
    for _ in range(n_iters):
        slices = _simulate(L_records, params, s0, T, n_steps, n_particles, rng)
        ts = np.concatenate([np.full(len(x), t) for t, x, _ in slices])
        xs = np.concatenate([x for _, x, _ in slices])
        vs = np.concatenate([v for _, _, v in slices])
        idx = rng.choice(len(ts), size=min(pool_subsample, len(ts)), replace=False)
        tz = torch.tensor(np.stack([ts[idx] / max(T, 1e-9), xs[idx] / Z_SCALE], axis=1),
                          dtype=torch.float32)
        tv = torch.tensor(vs[idx][:, None], dtype=torch.float32)
        t0 = time.perf_counter()
        for _ in range(fit_steps):
            opt.zero_grad()
            loss = ((net(tz) - tv) ** 2).mean()
            loss.backward()
            opt.step()
        fit_s += time.perf_counter() - t0
        new_records, sup = [], 0.0
        with torch.no_grad():
            for (t, g, Lg, _) in L_records:
                tzg = torch.tensor(np.stack([np.full(len(g), (t + dt) / max(T, 1e-9)),
                                             g / Z_SCALE], axis=1), dtype=torch.float32)
                f = np.clip(net(tzg).numpy()[:, 0], 1e-4, None)
                sig = dupire.sigma(max(t, dupire.T_grid[0]), np.exp(g), s0)
                Phi = np.clip(sig / np.sqrt(f), 0.0, L_max)
                L_new = (1 - alpha) * Lg + alpha * Phi
                sup = max(sup, float(np.abs(L_new - Lg).max()))
                new_records.append((t, g, L_new, f))
        L_records = new_records
        deltas.append(sup)
    return L_records, {"deltas": deltas, "fit_s": fit_s}
