"""Exp D: ridge head and spline head through the same battery (A: analytic estimator; B/C: calibration)."""
import json
import time
import numpy as np
from neural_particle_method.condexp import RidgeHead, spline_estimate
from neural_particle_method.heston import heston_call
from neural_particle_method.bs import implied_vol
from neural_particle_method.dupire import DupireSurface
from neural_particle_method.explicit import calibrate_explicit, mc_smile

out = {}

# ---- Part A: analytic estimator accuracy ----
sx, sv, rv, v0 = 0.25, 0.8, -0.7, 0.04
rng = np.random.default_rng(1)
for N in [2_000, 10_000, 50_000]:
    z = rng.standard_normal(N)
    wperp = rng.standard_normal(N)
    lnx = sx * z
    v = v0 * np.exp(rv * sv * z + np.sqrt(1 - rv ** 2) * sv * wperp - 0.5 * sv ** 2)
    grid = np.quantile(lnx, np.linspace(0.02, 0.98, 97))
    truth = v0 * np.exp(rv * sv * grid / sx - 0.5 * rv ** 2 * sv ** 2)
    rmse = lambda est: float(np.sqrt(np.mean((est - truth) ** 2)) / v0)

    rh = RidgeHead(seed=0)
    rh.train_body(lnx, v, steps=500)
    ridge = rh.fit_predict(lnx, v, grid)
    spl = spline_estimate(lnx, v, grid)
    out[f"A_{N}"] = {"ridge_rmse_rel": rmse(ridge), "spline_rmse_rel": rmse(spl)}
    print(f"A_{N}: {out[f'A_{N}']}", flush=True)

# ---- Parts B/C: calibration ----
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

for tag, model in [("B_same", MKT), ("C_mismatch", dict(MKT, xi=0.6, rho=-0.4))]:
    for method in ["ridge", "spline"]:
        t0 = time.time()
        lnx_T, diag = calibrate_explicit(surf, model, method=method, n_steps=50,
                                         n_particles=200_000, seed=42, snapshot_times=(0.5,))
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

json.dump(out, open("experiments/results/exp_d.json", "w"), indent=2)
