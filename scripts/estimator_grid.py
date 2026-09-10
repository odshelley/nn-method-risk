"""Estimator-quality grid against the exact conditional expectation from the PDE reference.

For scenario in (s02, s11): simulate a 500k cloud under the PDE reference leverage; at slices
t in (0.25, 1.0, 2.0) fit each estimator on (x, max(v,0)) and measure |f_hat - f_ref| on the
reference grid, density-weighted, overall and in the wings (|x| > 0.25). Writes a CSV.
"""
import json
import os
import sys
import tempfile
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd
import torch
from torch import nn

torch.set_num_threads(2)
from neural_particle_method.bench.scenarios import full_registry
from neural_particle_method.estimators import make_estimator
from neural_particle_method.estimators.nn import V_SCALE, Z_SCALE
from neural_particle_method.simulate.dynamics import HestonParams
from neural_particle_method.simulate.leverage import LeverageField
from neural_particle_method.simulate.stepper import heston_step
from neural_particle_method.tracking.store import Store

OUT = "results/estimator_grid.csv"
SIDS = ("s02", "s11")
SLICES = (0.25, 1.0, 2.0)
N_CLOUD = 500_000
N_FIT = 100_000


def cloud_at(field, hp, s0, T, n_steps, n, rng, times):
    dt, sdt = T / n_steps, np.sqrt(T / n_steps)
    lnx, v = np.full(n, np.log(s0)), np.full(n, hp.v0)
    want = {int(round(t / dt)): t for t in times}
    out = {}
    for k in range(n_steps):
        if k in want:
            out[want[k]] = (lnx.copy(), np.maximum(v, 0.0))
        L = field.at(k * dt, lnx)
        lnx, v = heston_step(lnx, v, L, rng.standard_normal(n), rng.standard_normal(n), hp, dt, sdt)
    return out


class MLP(nn.Module):
    def __init__(self, hidden, depth):
        super().__init__()
        layers, d = [], 1
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)

    def forward(self, z):
        return nn.functional.softplus(self.head(self.body(z))) * V_SCALE


def fit_nn(x, v, grid, hidden, depth, steps, lr, batch, seed=0):
    torch.manual_seed(seed)
    net = MLP(hidden, depth)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    z = torch.tensor(x[:, None] / Z_SCALE, dtype=torch.float32)
    tv = torch.tensor(v[:, None], dtype=torch.float32)
    n = len(x)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        if batch and batch < n:
            idx = torch.randint(0, n, (batch,), generator=g)
            zb, tb = z[idx], tv[idx]
        else:
            zb, tb = z, tv
        opt.zero_grad()
        loss = ((net(zb) - tb) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return net(torch.tensor(grid[:, None] / Z_SCALE, dtype=torch.float32)).numpy()[:, 0]


def main():
    store = Store()
    rows = pd.read_csv(OUT).to_dict("records") if os.path.exists(OUT) else []
    done = {(r["sid"], float(r["t"]), r["estimator"]) for r in rows}
    for sid in SIDS:
        sc = full_registry()[sid]
        hp = HestonParams.from_dict(sc.dynamics)
        rid = store.find_finished("pde_reference", {"sid": sid, "n_steps": 200, "lag": "none"})
        with tempfile.TemporaryDirectory() as d:
            ref = LeverageField.from_json(json.loads(store.download(rid, "leverage.json", d).read_text()))
        rng = np.random.default_rng(123)
        clouds = cloud_at(ref, hp, sc.s0, sc.T, 200, N_CLOUD, rng, SLICES)
        for t, (x, v) in clouds.items():
            s = ref[int(round(t / (sc.T / 200)))]
            lo, hi = np.quantile(x, [0.001, 0.999])
            m = (s.grid >= lo) & (s.grid <= hi)
            grid, f_ref = s.grid[m], s.f[m]
            dens = np.histogram(x, bins=np.r_[grid[0] - 1e-9, 0.5 * (grid[1:] + grid[:-1]), grid[-1] + 1e-9])[0].astype(float)
            dens /= dens.sum()
            wing = np.abs(grid) > 0.25
            idx = rng.choice(N_CLOUD, N_FIT, replace=False)
            xs, vs = x[idx], v[idx]

            def rec(name, f_hat, secs, **kw):
                e = np.abs(f_hat - f_ref) / np.maximum(f_ref, 1e-4)
                done.add((sid, float(t), name))
                rows.append({"sid": sid, "t": t, "estimator": name, **kw,
                             "rel_err_weighted": float((dens * e).sum()),
                             "rel_err_wings": float(np.mean(e[wing])) if wing.any() else np.nan,
                             "rel_err_center": float(np.mean(e[~wing])), "secs": secs})
                print(f"{sid} t={t} {name:28s} weighted {rows[-1]['rel_err_weighted']:.3%} "
                      f"wings {rows[-1]['rel_err_wings']:.3%} center {rows[-1]['rel_err_center']:.3%} "
                      f"{secs:.1f}s", flush=True)
                pd.DataFrame(rows).to_csv(OUT, index=False)

            def todo(name):
                return (sid, float(t), name) not in done

            for n_sub, label in ((30_000, "nw_30k"), (100_000, "nw_100k"), (500_000, "nw_500k")):
                j = rng.choice(N_CLOUD, n_sub, replace=False)
                if not todo(label):
                    continue
                t0 = time.perf_counter()
                f = make_estimator("nw").fit_predict(t, x[j], v[j], grid)
                rec(label, f, time.perf_counter() - t0, n_fit=n_sub)
            for label, est in (("spline_100k", make_estimator("spline")), ("rkhs_100k", make_estimator("rkhs"))):
                if not todo(label):
                    continue
                t0 = time.perf_counter()
                rec(label, est.fit_predict(t, xs, vs, grid), time.perf_counter() - t0, n_fit=N_FIT)
            for hidden in (64, 256):
                for depth in (2, 3):
                    for steps, lr, batch in ((120, 1e-2, 0), (1000, 1e-2, 0), (4000, 1e-3, 0),
                                             (4000, 1e-3, 8192), (20000, 1e-3, 8192)):
                        name = f"nn_h{hidden}_d{depth}_s{steps}_lr{lr:g}_b{batch}"
                        if not todo(name):
                            continue
                        t0 = time.perf_counter()
                        f = fit_nn(xs, vs, grid, hidden, depth, steps, lr, batch)
                        rec(name, f, time.perf_counter() - t0,
                            n_fit=N_FIT, hidden=hidden, depth=depth, steps=steps, lr=lr, batch=batch)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
