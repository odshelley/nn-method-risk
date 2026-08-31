"""Exp B: same-Heston market => fixed point L* = 1; Exp C: mismatched model params.

Metrics: |L-1| stats (B only) and repricing error in implied-vol bp at T = 0.5, 1.0.
"""
import json
import sys
import time
import numpy as np
from neural_particle_method.heston import heston_call
from neural_particle_method.bs import implied_vol
from neural_particle_method.dupire import DupireSurface
from neural_particle_method.explicit import calibrate_explicit, mc_smile

MKT = dict(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
T_grid = np.linspace(0.05, 1.0, 20)
k_grid = np.linspace(np.log(0.6), np.log(1.6), 33)
price_fn = lambda K, T: heston_call(K, T, **{k: MKT[k] for k in ("v0", "kappa", "theta", "xi", "rho")})
surf = DupireSurface.from_price_fn(price_fn, 1.0, T_grid, k_grid)

K_test = np.exp(np.linspace(np.log(0.7), np.log(1.4), 15))

def reprice_errors(snapshots, lnx_T):
    errs = {}
    for T, lnx in [(0.5, snapshots[0.5]), (1.0, lnx_T)]:
        mc = mc_smile(lnx, K_test)
        ref = price_fn(K_test, T)
        iv_mc = np.array([implied_vol(p, 1.0, k, T) for p, k in zip(mc, K_test)])
        iv_ref = np.array([implied_vol(p, 1.0, k, T) for p, k in zip(ref, K_test)])
        ok = np.isfinite(iv_mc) & np.isfinite(iv_ref)
        errs[T] = {"max_bp": float(np.nanmax(np.abs(iv_mc - iv_ref)[ok]) * 1e4),
                   "rmse_bp": float(np.sqrt(np.nanmean(((iv_mc - iv_ref)[ok]) ** 2)) * 1e4)}
    return errs

out = {}
for tag, model, methods in [("B_same", MKT, ["nn", "nw"]),
                            ("C_mismatch", dict(MKT, xi=0.6, rho=-0.4), ["nn", "nw"])]:
    for method in methods:
        t0 = time.time()
        lnx_T, diag = calibrate_explicit(surf, model, method=method, n_steps=50,
                                         n_particles=200_000, seed=42,
                                         snapshot_times=(0.5,))
        run = {"runtime_s": round(time.time() - t0, 1),
               "reprice": reprice_errors(diag["snapshots"], lnx_T)}
        if tag == "B_same":
            devs = []
            for t, grid, L, f in diag["L_records"][1:]:
                lo, hi = np.quantile(grid, [0.05, 0.95])
                sel = (grid >= lo) & (grid <= hi)
                devs.append(np.abs(L[sel] - 1.0))
            alld = np.concatenate(devs)
            run["L_minus_1"] = {"mean": float(alld.mean()), "p95": float(np.quantile(alld, 0.95)),
                                "max": float(alld.max())}
        out[f"{tag}_{method}"] = run
        print(f"{tag}_{method}: {json.dumps(run)}", flush=True)

# uncalibrated contrast for C: L clamped to 1 via L_max trick is wrong; instead run pure model (L=1)
# by feeding a flat Dupire surface equal to sqrt(E[V]) is not meaningful; report model-C with L=1:
import numpy as np
from neural_particle_method.explicit import calibrate_explicit as _ce
class UnitSurface:
    T_grid = T_grid
    def sigma(self, t, x, s0=1.0):
        f = np.sqrt(0.04)
        return np.full(np.shape(np.atleast_1d(x)), 1.0) * f  # placeholder; see below
# Proper uncalibrated run: simulate model C with L identically 1
model_c = dict(MKT, xi=0.6, rho=-0.4)
rng = np.random.default_rng(7)
N, steps, T = 200_000, 50, 1.0
dt, sdt = T / steps, np.sqrt(T / steps)
lnx = np.zeros(N); v = np.full(N, model_c["v0"]); snap = None
for k in range(steps):
    z1 = rng.standard_normal(N)
    z2 = model_c["rho"] * z1 + np.sqrt(1 - model_c["rho"] ** 2) * rng.standard_normal(N)
    vp = np.maximum(v, 0.0)
    lnx = lnx + (-0.5 * vp) * dt + np.sqrt(vp) * sdt * z1
    v = v + model_c["kappa"] * (model_c["theta"] - vp) * dt + model_c["xi"] * np.sqrt(vp) * sdt * z2
    if k + 1 == steps // 2: snap = lnx.copy()
out["C_uncalibrated_L1"] = {"reprice": reprice_errors({0.5: snap}, lnx)}
print("C_uncalibrated_L1:", json.dumps(out["C_uncalibrated_L1"]))

json.dump(out, open("experiments/results/exp_bc.json", "w"), indent=2)
