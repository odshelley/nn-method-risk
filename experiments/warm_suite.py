"""Warm-start suite: dynamics bumps, sequential tracking, crossover sweep, feedback-norm measurement.

Arms
  dyn    dynamics-parameter bumps (xi, rho, kappa re-marks): do-nothing vs head refits vs re-solve.
         The surface is unchanged, so the free refresh is literally the do-nothing strategy and
         the staleness premise fails by construction (DF is hit directly).
  seq    a day of cumulative small surface bumps: refresh-only vs damped ridge refit vs Kalman
         head tracking (plain and spline-augmented) vs warm-started re-solve at every step.
  xover  the standard surface bump scaled by s: locate the crossover where corrections and the
         re-solve overtake the free refresh, as predicted near |dsigma| ~ floor/|A|.
  norm   power iteration on the feedback operator A = (L/2f) DF via common-random-number
         simulate-and-refit differences; reports |A| and the implied error fraction.

Resumable: one JSON per (arm, sid, seed) in results/warm/, existing files skipped.
Overnight artifacts (net, betas, implicit records) are cached per (sid, seed, size) in
results/warm/cache/ and shared across arms.
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

from bench.scenarios import make_registry, quote_k_grid
from bump_correct import ALPHA, GRID, beta_correction, bump, distil, records_from_betas
from neural_particle_method.explicit import calibrate_explicit
from neural_particle_method.implicit import GlobalNet, GlobalRidgeHead, calibrate_implicit, _simulate
from neural_particle_method.readouts import KalmanHead
from neural_particle_method.reprice import reprice_iv, iv_metrics, snap_times
from neural_particle_method.ssvi import SSVIParams, SSVILocalVol, implied_vol_ssvi, no_arb_ok

OUT = ROOT / "results" / "warm"
CACHE = OUT / "cache"


@dataclasses.dataclass(frozen=True)
class Cfg:
    N: int = 200_000
    n_steps: int = 50
    sub: int = 30_000
    n_iters: int = 6
    fit_steps: int = 300
    reprice_N: int = 300_000
    reprice_steps: int = 200
    seq_len: int = 6
    xover_scales: tuple = (0.5, 1.0, 2.0, 4.0)
    norm_iters: int = 4

FULL = Cfg()
SMOKE = Cfg(N=20_000, n_steps=12, sub=8_000, n_iters=2, fit_steps=120,
            reprice_N=60_000, reprice_steps=100, seq_len=3,
            xover_scales=(1.0, 4.0), norm_iters=2)


def write_atomic(path, doc):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(path)


def score(recs, dyn, s0, mats, kq, ssvi_p, seed, cfg):
    ivs = reprice_iv(recs, dyn, s0, mats, kq, n_particles=cfg.reprice_N,
                     n_steps=cfg.reprice_steps, seed=seed)
    ts = snap_times(mats, cfg.reprice_steps)
    tgt = np.stack([implied_vol_ssvi(ssvi_p, kq, t) for t in ts])
    return iv_metrics(ivs, tgt, kq, mats)["pooled_rmse_bp"]


def overnight(sc, seed, cfg):
    """Overnight solve, cached: implicit records, trained net, distilled per-slice betas."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{sc.sid}_s{seed}_N{cfg.N}_k{cfg.n_steps}.pt"
    lvS = SSVILocalVol(sc.ssvi, sc.s0, T_max=sc.T)
    if path.exists():
        # weights_only=False is fine here: the cache is written by this script, never downloaded
        blob = torch.load(path, weights_only=False)
        net = GlobalNet()
        net.load_state_dict(blob["net"])
        return net, blob["betas"], blob["recsS"], blob["overnight_s"], lvS
    dyn, s0, T = sc.dynamics, sc.s0, sc.T
    t0 = time.perf_counter()
    _, winfo = calibrate_explicit(lvS, dyn, s0=s0, T=T, n_steps=cfg.n_steps,
                                  n_particles=cfg.N, seed=seed)
    recsS, iinfo = calibrate_implicit(lvS, dyn, s0=s0, T=T, n_steps=cfg.n_steps,
                                      n_particles=cfg.N, alpha=ALPHA, n_iters=cfg.n_iters,
                                      seed=seed, fit_steps=cfg.fit_steps,
                                      L0_records=winfo["L_records"])
    net = iinfo["net"]
    rng = np.random.default_rng(seed + 50)
    slices = _simulate(recsS, dyn, s0, T, cfg.n_steps, cfg.N, rng)
    betas, _ = distil(net, slices, lvS, T, cfg.n_steps, s0, dyn["v0"], cfg.sub, rng)
    elapsed = time.perf_counter() - t0
    torch.save({"net": net.state_dict(), "betas": betas, "recsS": recsS,
                "overnight_s": elapsed}, path)
    return net, betas, recsS, elapsed, lvS


def records_from_head(head, betas, lv, T, n_steps, s0, v0):
    """Rebuild leverage records from a head's feature map and per-slice betas."""
    dt = T / n_steps
    sig0 = lv.sigma(max(0.0, lv.T_grid[0]), np.exp(GRID), s0)
    recs = [(0.0, GRID.copy(), np.clip(sig0 / np.sqrt(v0), 0.0, 4.0), np.full(len(GRID), v0))]
    for k in range(1, n_steps):
        t = k * dt
        f = np.clip(head._features(t, GRID) @ betas[k], 1e-4, None)
        sig = lv.sigma(max(t, lv.T_grid[0]), np.exp(GRID), s0)
        recs.append((t, GRID.copy(), np.clip(sig / np.sqrt(f), 0.0, 4.0), f))
    return recs


def kalman_pass(head, betas, lv, dyn, T, n_steps, N, s0, v0, sub, seed):
    """Simulate under the current leverage, one Kalman measurement update per slice."""
    recs = records_from_head(head, betas, lv, T, n_steps, s0, v0)
    rng = np.random.default_rng(seed)
    slices = _simulate(recs, dyn, s0, T, n_steps, N, rng)
    for k in range(1, n_steps):
        t = k * (T / n_steps)
        _, lnx, v = slices[k - 1]
        idx = rng.choice(len(lnx), size=min(sub, len(lnx)), replace=False)
        head.update(k, t, lnx[idx], v[idx], GRID)
    return head.betas()


def full_resolve(lv, dyn, s0, T, cfg, seed, L0_records=None, n_iters=None):
    t0 = time.perf_counter()
    if L0_records is None:
        _, winfo = calibrate_explicit(lv, dyn, s0=s0, T=T, n_steps=cfg.n_steps,
                                      n_particles=cfg.N, seed=seed)
        L0_records = winfo["L_records"]
    recs, _ = calibrate_implicit(lv, dyn, s0=s0, T=T, n_steps=cfg.n_steps,
                                 n_particles=cfg.N, alpha=ALPHA,
                                 n_iters=cfg.n_iters if n_iters is None else n_iters,
                                 seed=seed, fit_steps=cfg.fit_steps, L0_records=L0_records)
    return recs, time.perf_counter() - t0


# ---------------------------------------------------------------- arm: dyn

def dyn_variants(dyn):
    return {
        "xi_up": dict(dyn, xi=dyn["xi"] * 1.3),
        "rho_dn": dict(dyn, rho=max(dyn["rho"] - 0.15, -0.9)),
        "kappa_dn": dict(dyn, kappa=dyn["kappa"] * 0.6),
    }


def arm_dyn(sc, seed, cfg):
    net, betas0, recsS, on_s, lvS = overnight(sc, seed, cfg)
    dyn, s0, T, mats, kq = sc.dynamics, sc.s0, sc.T, list(sc.maturities), quote_k_grid()
    v0 = dyn["v0"]
    res, times = {}, {"overnight": round(on_s, 2)}
    recs0 = records_from_betas(net, betas0, lvS, T, cfg.n_steps, s0, v0)
    res["anchor_unbumped"] = score(recs0, dyn, s0, mats, kq, sc.ssvi, seed + 900, cfg)
    for name, dynB in dyn_variants(dyn).items():
        t0 = time.perf_counter()
        res[f"{name}/do_nothing"] = score(recs0, dynB, s0, mats, kq, sc.ssvi, seed + 910, cfg)
        times[f"{name}/do_nothing"] = 0.0

        b = dict(betas0)
        for i in (1, 2):
            t0 = time.perf_counter()
            b = beta_correction(net, b, lvS, dynB, T, cfg.n_steps, cfg.N, s0, v0,
                                cfg.sub, seed + 20 + i)
            times[f"{name}/ridge_{i}"] = round(time.perf_counter() - t0
                                               + times.get(f"{name}/ridge_{i-1}", 0.0), 2)
            res[f"{name}/ridge_{i}"] = score(records_from_betas(net, b, lvS, T, cfg.n_steps, s0, v0),
                                             dynB, s0, mats, kq, sc.ssvi, seed + 920 + i, cfg)

        head = KalmanHead(net, T)
        head.init_from(betas0)
        bk = dict(betas0)
        for i in (1, 2):
            t0 = time.perf_counter()
            bk = kalman_pass(head, bk, lvS, dynB, T, cfg.n_steps, cfg.N, s0, v0,
                             cfg.sub, seed + 30 + i)
            times[f"{name}/kalman_{i}"] = round(time.perf_counter() - t0
                                                + times.get(f"{name}/kalman_{i-1}", 0.0), 2)
            res[f"{name}/kalman_{i}"] = score(records_from_head(head, bk, lvS, T, cfg.n_steps, s0, v0),
                                              dynB, s0, mats, kq, sc.ssvi, seed + 930 + i, cfg)

        recsR, dt_s = full_resolve(lvS, dynB, s0, T, cfg, seed + 40)
        times[f"{name}/full_resolve"] = round(dt_s, 2)
        res[f"{name}/full_resolve"] = score(recsR, dynB, s0, mats, kq, sc.ssvi, seed + 940, cfg)
    return {"rmse_bp": res, "build_s": times}


# ---------------------------------------------------------------- arm: seq

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


def arm_seq(sc, seed, cfg):
    net, betas0, recsS, on_s, lvS = overnight(sc, seed, cfg)
    dyn, s0, T, mats, kq = sc.dynamics, sc.s0, sc.T, list(sc.maturities), quote_k_grid()
    v0 = dyn["v0"]
    path = seq_path(sc.ssvi, cfg.seq_len, np.random.default_rng(seed + 7000))
    headK = KalmanHead(net, T)
    headK.init_from(betas0)
    headS = KalmanHead(net, T, n_spline=10)
    headS.init_from(betas0)
    b_ridge, bK, bS = dict(betas0), headK.betas(), headS.betas()
    recsW = recsS
    res, times = {"steps": []}, {"overnight": round(on_s, 2)}
    for j, pj in enumerate(path):
        lvj = SSVILocalVol(pj, s0, T_max=T)
        row, trow = {}, {}

        t0 = time.perf_counter()
        recs = records_from_betas(net, betas0, lvj, T, cfg.n_steps, s0, v0)
        trow["refresh"] = round(time.perf_counter() - t0, 2)
        row["refresh"] = score(recs, dyn, s0, mats, kq, pj, seed + 800 + j, cfg)

        t0 = time.perf_counter()
        b_ridge = beta_correction(net, b_ridge, lvj, dyn, T, cfg.n_steps, cfg.N, s0, v0,
                                  cfg.sub, seed + 100 + j)
        trow["ridge"] = round(time.perf_counter() - t0, 2)
        row["ridge"] = score(records_from_betas(net, b_ridge, lvj, T, cfg.n_steps, s0, v0),
                             dyn, s0, mats, kq, pj, seed + 810 + j, cfg)

        t0 = time.perf_counter()
        bK = kalman_pass(headK, bK, lvj, dyn, T, cfg.n_steps, cfg.N, s0, v0,
                         cfg.sub, seed + 200 + j)
        trow["kalman"] = round(time.perf_counter() - t0, 2)
        row["kalman"] = score(records_from_head(headK, bK, lvj, T, cfg.n_steps, s0, v0),
                              dyn, s0, mats, kq, pj, seed + 820 + j, cfg)

        t0 = time.perf_counter()
        bS = kalman_pass(headS, bS, lvj, dyn, T, cfg.n_steps, cfg.N, s0, v0,
                         cfg.sub, seed + 300 + j)
        trow["kalman_spline"] = round(time.perf_counter() - t0, 2)
        row["kalman_spline"] = score(records_from_head(headS, bS, lvj, T, cfg.n_steps, s0, v0),
                                     dyn, s0, mats, kq, pj, seed + 830 + j, cfg)

        recsW, dt_s = full_resolve(lvj, dyn, s0, T, cfg, seed + 400 + j,
                                   L0_records=recsW, n_iters=3)
        trow["resolve_warm"] = round(dt_s, 2)
        row["resolve_warm"] = score(recsW, dyn, s0, mats, kq, pj, seed + 840 + j, cfg)

        res["steps"].append(row)
        times[f"step_{j}"] = trow

    recsC, dt_s = full_resolve(SSVILocalVol(path[-1], s0, T_max=T), dyn, s0, T, cfg, seed + 500)
    times["resolve_cold_final"] = round(dt_s, 2)
    res["resolve_cold_final"] = score(recsC, dyn, s0, mats, kq, path[-1], seed + 850, cfg)
    res["path"] = [dataclasses.asdict(pj) for pj in path]
    return {"rmse_bp": res, "build_s": times}


# ---------------------------------------------------------------- arm: xover

def scaled_bump(p, s):
    for shrink in (1.0, 0.75, 0.5, 0.25):
        eff = s * shrink
        q = SSVIParams(sigma0=p.sigma0 + 0.01 * eff, eta=p.eta * max(1 - 0.05 * eff, 0.3),
                       gamma=p.gamma, rho=min(p.rho + 0.03 * eff, -0.05))
        if no_arb_ok(q):
            return q, eff
    raise RuntimeError("no-arb bump exhausted")


def arm_xover(sc, seed, cfg):
    net, betas0, recsS, on_s, lvS = overnight(sc, seed, cfg)
    dyn, s0, T, mats, kq = sc.dynamics, sc.s0, sc.T, list(sc.maturities), quote_k_grid()
    v0 = dyn["v0"]
    res, times = {}, {"overnight": round(on_s, 2)}
    for s in cfg.xover_scales:
        pB, eff = scaled_bump(sc.ssvi, s)
        lvB = SSVILocalVol(pB, s0, T_max=T)
        tag = f"s{s:g}"
        res[f"{tag}/scale_eff"] = eff

        recs = records_from_betas(net, betas0, lvB, T, cfg.n_steps, s0, v0)
        res[f"{tag}/refresh"] = score(recs, dyn, s0, mats, kq, pB, seed + 600, cfg)
        times[f"{tag}/refresh"] = 0.0

        t0 = time.perf_counter()
        b = beta_correction(net, dict(betas0), lvB, dyn, T, cfg.n_steps, cfg.N, s0, v0,
                            cfg.sub, seed + 610)
        times[f"{tag}/ridge_1"] = round(time.perf_counter() - t0, 2)
        res[f"{tag}/ridge_1"] = score(records_from_betas(net, b, lvB, T, cfg.n_steps, s0, v0),
                                      dyn, s0, mats, kq, pB, seed + 620, cfg)

        recsR, dt_s = full_resolve(lvB, dyn, s0, T, cfg, seed + 630)
        times[f"{tag}/full_resolve"] = round(dt_s, 2)
        res[f"{tag}/full_resolve"] = score(recsR, dyn, s0, mats, kq, pB, seed + 640, cfg)
    return {"rmse_bp": res, "build_s": times}


# ---------------------------------------------------------------- arm: norm

def arm_norm(sc, seed, cfg, eps=0.02):
    net, betas0, recsS, on_s, lvS = overnight(sc, seed, cfg)
    dyn, s0, T = sc.dynamics, sc.s0, sc.T
    v0 = dyn["v0"]
    n_steps, dt = cfg.n_steps, T / cfg.n_steps
    base_recs = records_from_betas(net, betas0, lvS, T, n_steps, s0, v0)
    L_field = np.stack([r[2] for r in base_recs])

    def F_of(recs, sim_seed, fit_seed):
        rng = np.random.default_rng(sim_seed)
        slices = _simulate(recs, dyn, s0, T, n_steps, cfg.N, rng)
        fr = np.random.default_rng(fit_seed)
        head = GlobalRidgeHead(net, T)
        out = np.full((n_steps, len(GRID)), v0)
        for k in range(1, n_steps):
            t = k * dt
            _, lnx, v = slices[k - 1]
            idx = fr.choice(len(lnx), size=min(cfg.sub, len(lnx)), replace=False)
            out[k] = np.clip(head.fit_predict(t, lnx[idx], v[idx], GRID), 1e-4, None)
        return out

    f_base = F_of(base_recs, seed + 1, seed + 2)
    rng = np.random.default_rng(seed + 3)
    u = rng.standard_normal((n_steps, len(GRID)))
    u[0] = 0.0
    u /= np.abs(u).max()
    history = []
    for _ in range(cfg.norm_iters):
        recs_p = [(t, g, np.clip(Lg + eps * u[k], 0.0, 4.0), fg)
                  for k, (t, g, Lg, fg) in enumerate(base_recs)]
        f_pert = F_of(recs_p, seed + 1, seed + 2)
        Au = (L_field / (2.0 * f_base)) * (f_pert - f_base) / eps
        Au[0] = 0.0
        nrm = float(np.abs(Au).max())
        history.append(round(nrm, 4))
        if nrm < 1e-12:
            break
        u = Au / nrm
    a = history[-1]
    return {"A_norm_history": history, "A_norm": a,
            "error_fraction_pred": round(a / (1.0 + a), 4) if a < 20 else None,
            "eps": eps}


# ---------------------------------------------------------------- driver

ARMS = {"dyn": arm_dyn, "seq": arm_seq, "xover": arm_xover, "norm": arm_norm}
DEFAULT_SIDS = ("s01", "s02", "s03", "s04", "s05")


def job_list(arms, sids, dyn_seeds=(0, 1), other_seeds=(0,)):
    jobs = []
    for arm in arms:
        for sid in sids:
            for seed in (dyn_seeds if arm == "dyn" else other_seeds):
                jobs.append((arm, sid, seed))
    return jobs


def run(arms, sids, cfg, tag=""):
    reg = make_registry()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = job_list(arms, sids)
    for arm, sid, seed in jobs:
        path = OUT / f"{tag}{arm}_{sid}_s{seed}.json"
        if path.exists():
            continue
        t0 = time.perf_counter()
        doc = ARMS[arm](reg[sid], seed, cfg)
        doc.update({"arm": arm, "sid": sid, "seed": seed,
                    "wall_s": round(time.perf_counter() - t0, 1)})
        write_atomic(path, doc)
        print(f"{arm} {sid} s{seed}: done in {doc['wall_s']}s", flush=True)
    print("all jobs done", flush=True)


def aggregate(tag=""):
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob(f"{tag}*.json"))]
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    lines = ["# Warm-start suite summary", ""]
    for arm in ("dyn", "xover", "norm", "seq"):
        docs = by_arm.get(arm, [])
        if not docs:
            continue
        lines.append(f"## {arm} ({len(docs)} runs)")
        if arm == "seq":
            n_steps = min(len(d["rmse_bp"]["steps"]) for d in docs)
            strategies = docs[0]["rmse_bp"]["steps"][0].keys()
            lines.append("| step | " + " | ".join(strategies) + " |")
            lines.append("|" + "---|" * (len(list(strategies)) + 1))
            for j in range(n_steps):
                vals = [np.mean([d["rmse_bp"]["steps"][j][s] for d in docs])
                        for s in strategies]
                lines.append(f"| {j} | " + " | ".join(f"{v:.0f}" for v in vals) + " |")
            lines.append(f"cold re-solve at final step: "
                         f"{np.mean([d['rmse_bp']['resolve_cold_final'] for d in docs]):.0f} bp")
        elif arm == "norm":
            for d in docs:
                lines.append(f"- {d['sid']} s{d['seed']}: |A| = {d['A_norm']}, "
                             f"predicted error fraction {d['error_fraction_pred']}, "
                             f"history {d['A_norm_history']}")
        else:
            keys = sorted({k for d in docs for k in d["rmse_bp"]
                           if not k.endswith("scale_eff")})
            lines.append("| strategy | mean rmse bp | mean build s |")
            lines.append("|---|---|---|")
            for k in keys:
                vals = [d["rmse_bp"][k] for d in docs if k in d["rmse_bp"]]
                ts = [d["build_s"].get(k, "") for d in docs if k in d["rmse_bp"]]
                ts = [t for t in ts if t != ""]
                lines.append(f"| {k} | {np.mean(vals):.0f} | "
                             f"{np.mean(ts):.1f} |" if ts else
                             f"| {k} | {np.mean(vals):.0f} | |")
        lines.append("")
    md = "\n".join(lines)
    (OUT / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "smoke", "aggregate"])
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--sids", nargs="*", default=list(DEFAULT_SIDS))
    args = ap.parse_args()
    if args.cmd == "smoke":
        run(args.arms, ["s01"], SMOKE, tag="smoke_")
    elif args.cmd == "run":
        run(args.arms, args.sids, FULL)
    else:
        aggregate()
