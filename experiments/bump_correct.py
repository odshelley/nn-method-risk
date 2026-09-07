"""Bump-and-correct: intraday strategies on a bumped surface from an overnight implicit body."""
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

from bench.scenarios import make_registry, quote_k_grid
from neural_particle_method.explicit import calibrate_explicit
from neural_particle_method.implicit import GlobalRidgeHead, calibrate_implicit, _simulate
from neural_particle_method.reprice import reprice_iv, iv_metrics, snap_times
from neural_particle_method.ssvi import SSVIParams, SSVILocalVol, implied_vol_ssvi, no_arb_ok

ALPHA = 0.5
GRID = np.linspace(np.log(0.4), np.log(2.2), 81)


def bump(p):
    q = SSVIParams(sigma0=p.sigma0 + 0.01, eta=0.95 * p.eta, gamma=p.gamma,
                   rho=min(p.rho + 0.03, -0.05))
    assert no_arb_ok(q)
    return q


def distil(net, slices, lv, T, n_steps, s0, v0, sub, rng):
    """Per-slice ridge heads on a simulated cloud; same-time pairing throughout."""
    dt = T / n_steps
    head = GlobalRidgeHead(net, T)
    betas, recs = {}, []
    sig0 = lv.sigma(max(0.0, lv.T_grid[0]), np.exp(GRID), s0)
    recs.append((0.0, GRID.copy(), np.clip(sig0 / np.sqrt(v0), 0.0, 4.0), np.full(len(GRID), v0)))
    for k in range(1, n_steps):
        t = k * dt
        _, lnx, v = slices[k - 1]
        idx = rng.choice(len(lnx), size=min(sub, len(lnx)), replace=False)
        f = np.clip(head.fit_predict(t, lnx[idx], v[idx], GRID), 1e-4, None)
        betas[k] = head.w_prev.copy()
        sig = lv.sigma(max(t, lv.T_grid[0]), np.exp(GRID), s0)
        recs.append((t, GRID.copy(), np.clip(sig / np.sqrt(f), 0.0, 4.0), f))
    return betas, recs


def records_from_betas(net, betas, lv, T, n_steps, s0, v0):
    dt = T / n_steps
    head = GlobalRidgeHead(net, T)
    sig0 = lv.sigma(max(0.0, lv.T_grid[0]), np.exp(GRID), s0)
    recs = [(0.0, GRID.copy(), np.clip(sig0 / np.sqrt(v0), 0.0, 4.0), np.full(len(GRID), v0))]
    for k in range(1, n_steps):
        t = k * dt
        f = np.clip(head._features(t, GRID) @ betas[k], 1e-4, None)
        sig = lv.sigma(max(t, lv.T_grid[0]), np.exp(GRID), s0)
        recs.append((t, GRID.copy(), np.clip(sig / np.sqrt(f), 0.0, 4.0), f))
    return recs


def beta_correction(net, betas, lv, dyn, T, n_steps, N, s0, v0, sub, seed):
    """One damped correction in beta space: simulate under L(beta), refit heads, damp."""
    recs = records_from_betas(net, betas, lv, T, n_steps, s0, v0)
    rng = np.random.default_rng(seed)
    slices = _simulate(recs, dyn, s0, T, n_steps, N, rng)
    hat, _ = distil(net, slices, lv, T, n_steps, s0, v0, sub, rng)
    return {k: (1 - ALPHA) * betas[k] + ALPHA * hat[k] for k in betas}


def score(recs, dyn, s0, mats, kq, ssvi_p, seed):
    ivs = reprice_iv(recs, dyn, s0, mats, kq, n_particles=300_000, n_steps=200, seed=seed)
    ts = snap_times(mats, 200)
    tgt = np.stack([implied_vol_ssvi(ssvi_p, kq, t) for t in ts])
    return iv_metrics(ivs, tgt, kq, mats)["pooled_rmse_bp"]


def run_pair(sc, seed, N, n_steps, sub, n_iters, fit_steps, out_dir):
    dyn, s0, T, mats, kq = sc.dynamics, sc.s0, sc.T, list(sc.maturities), quote_k_grid()
    v0 = dyn["v0"]
    lvS = SSVILocalVol(sc.ssvi, s0, T_max=T)
    pB = bump(sc.ssvi)
    lvB = SSVILocalVol(pB, s0, T_max=T)
    res, times = {}, {}

    t0 = time.perf_counter()
    _, winfo = calibrate_explicit(lvS, dyn, s0=s0, T=T, n_steps=n_steps, n_particles=N, seed=seed)
    recsS, iinfo = calibrate_implicit(lvS, dyn, s0=s0, T=T, n_steps=n_steps, n_particles=N,
                                      alpha=ALPHA, n_iters=n_iters, seed=seed,
                                      fit_steps=fit_steps, L0_records=winfo["L_records"])
    net = iinfo["net"]
    rng = np.random.default_rng(seed + 50)
    slices = _simulate(recsS, dyn, s0, T, n_steps, N, rng)
    betas0, recs_b0 = distil(net, slices, lvS, T, n_steps, s0, v0, sub, rng)
    times["overnight"] = time.perf_counter() - t0
    res["anchor_unbumped_implicit"] = score(recsS, dyn, s0, mats, kq, sc.ssvi, seed + 900)
    res["anchor_unbumped_distilled"] = score(recs_b0, dyn, s0, mats, kq, sc.ssvi, seed + 900)

    t0 = time.perf_counter()
    res["stale_L"] = score(recsS, dyn, s0, mats, kq, pB, seed + 901)
    times["stale_L"] = 0.0

    t0 = time.perf_counter()
    recs_js = records_from_betas(net, betas0, lvB, T, n_steps, s0, v0)
    times["stale_f_fresh_sigma"] = time.perf_counter() - t0
    res["stale_f_fresh_sigma"] = score(recs_js, dyn, s0, mats, kq, pB, seed + 902)

    t0 = time.perf_counter()
    head = GlobalRidgeHead(net, T)
    _, einfo = calibrate_explicit(lvB, dyn, s0=s0, T=T, n_steps=n_steps, n_particles=N,
                                  fit_subsample=sub, seed=seed + 1, regressor=head)
    times["causal_sweep"] = time.perf_counter() - t0
    res["causal_sweep"] = score(einfo["L_records"], dyn, s0, mats, kq, pB, seed + 903)

    b = dict(betas0)
    for i in (1, 2):
        t0 = time.perf_counter()
        b = beta_correction(net, b, lvB, dyn, T, n_steps, N, s0, v0, sub, seed + 10 + i)
        times[f"beta_corr_{i}"] = time.perf_counter() - t0 + (times.get(f"beta_corr_{i-1}", 0.0) if i > 1 else 0.0)
        res[f"beta_corr_{i}"] = score(records_from_betas(net, b, lvB, T, n_steps, s0, v0),
                                      dyn, s0, mats, kq, pB, seed + 904 + i)

    t0 = time.perf_counter()
    _, w2 = calibrate_explicit(lvB, dyn, s0=s0, T=T, n_steps=n_steps, n_particles=N, seed=seed + 2)
    recsB, _ = calibrate_implicit(lvB, dyn, s0=s0, T=T, n_steps=n_steps, n_particles=N,
                                  alpha=ALPHA, n_iters=n_iters, seed=seed + 2,
                                  fit_steps=fit_steps, L0_records=w2["L_records"])
    times["full_resolve"] = time.perf_counter() - t0
    res["full_resolve"] = score(recsB, dyn, s0, mats, kq, pB, seed + 907)

    doc = {"sid": sc.sid, "seed": seed, "rmse_bp": res, "build_s": {k: round(v, 2) for k, v in times.items()},
           "bumped": dataclasses.asdict(pB)}
    (out_dir / f"{sc.sid}_s{seed}.json").write_text(json.dumps(doc, indent=1))
    print(f"{sc.sid} s{seed}: " + " ".join(f"{k}={v:.0f}" for k, v in res.items()), flush=True)


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    reg = make_registry()
    out = Path("results/bump"); out.mkdir(parents=True, exist_ok=True)
    if smoke:
        run_pair(reg["s01"], 0, N=20_000, n_steps=12, sub=8_000, n_iters=2, fit_steps=120, out_dir=out)
    else:
        for sid in ("s01", "s02", "s03", "s04", "s05"):
            for seed in (0, 1):
                p = out / f"{sid}_s{seed}.json"
                if p.exists():
                    continue
                run_pair(reg[sid], seed, N=200_000, n_steps=50, sub=30_000,
                         n_iters=6, fit_steps=300, out_dir=out)
    print("done")
