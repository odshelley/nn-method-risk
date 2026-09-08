"""Warm-start suite: dynamics bumps, sequential tracking, crossover sweep, feedback-norm measurement."""
import dataclasses
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from ..bench.scenarios import make_registry, quote_k_grid
from ..calibrate.config import ExplicitConfig
from ..calibrate.explicit import calibrate_explicit
from ..calibrate.implicit import GlobalNet, calibrate_implicit, simulate_slices
from ..calibrate.warm import (
    beta_correction,
    distil,
    dyn_variants,
    full_resolve,
    kalman_pass,
    records_from_betas,
    records_from_head,
    scaled_bump,
    seq_path,
)
from ..estimators import make_estimator
from ..estimators.kalman import KalmanHead
from ..estimators.ridge import GlobalRidge
from ..market.local_vol import SSVILocalVol
from ..pricing.metrics import iv_metrics, target_ivs
from ..pricing.reprice import reprice_iv
from ..simulate.leverage import DEFAULT_GRID as GRID
from ..simulate.leverage import LeverageField, Slice
from ..tracking.store import flatten_metrics, git_hash, to_jsonable

OVERNIGHT_EXPERIMENT = "overnight"
WARM_EXPERIMENT = "warm"


def score(field, params, s0, mats, kq, ssvi_p, seed, reprice_cfg):
    ivs = reprice_iv(field, params, s0, mats, kq, reprice_cfg, seed=seed)
    tgt = target_ivs(ssvi_p, kq, mats, reprice_cfg.n_steps)
    return iv_metrics(ivs, tgt, kq, mats)["pooled_rmse_bp"]


def overnight_key(sc, seed, cfg):
    return {"sid": sc.sid, "seed": seed, "N": cfg.N, "n_steps": cfg.n_steps}


def overnight(store, sc, seed, cfg):
    """Overnight solve, cached as an `overnight` run: implicit field, trained net, distilled betas."""
    lvS = SSVILocalVol(sc.ssvi, sc.s0, T_max=sc.T)
    key = overnight_key(sc, seed, cfg)
    rid = store.find_finished(OVERNIGHT_EXPERIMENT, key)
    if rid is not None:
        with tempfile.TemporaryDirectory() as d:
            blob = torch.load(store.download(rid, "overnight.pt", d), weights_only=False)
        net = GlobalNet()
        net.load_state_dict(blob["net"])
        return net, blob["betas"], LeverageField.from_records(blob["recsS"]), blob["overnight_s"], lvS
    dyn, s0, T = sc.dynamics, sc.s0, sc.T
    with store.run(OVERNIGHT_EXPERIMENT, {**key, "git_hash": git_hash()}) as h:
        t0 = time.perf_counter()
        w = calibrate_explicit(lvS, dyn, make_estimator("nn", seed=seed),
                               ExplicitConfig(n_steps=cfg.n_steps, n_particles=cfg.N, fit_v_floor=cfg.fit_v_floor),
                               s0=s0, T=T, seed=seed)
        r = calibrate_implicit(lvS, dyn, cfg.implicit, s0=s0, T=T, seed=seed, L0=w.field)
        net = r.net
        rng = np.random.default_rng(seed + 50)
        slices = simulate_slices(r.field, dyn, s0, T, cfg.n_steps, cfg.N, rng)
        betas, _ = distil(net, slices, lvS, T, cfg.n_steps, s0, dyn.v0, cfg.sub, rng)
        elapsed = time.perf_counter() - t0
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "overnight.pt"
            torch.save({"net": net.state_dict(), "betas": betas, "recsS": r.field.to_records(),
                        "overnight_s": elapsed}, p)
            h.log_file(p)
        h.log_metrics({"overnight_s": elapsed})
    return net, betas, r.field, elapsed, lvS


# ---------------------------------------------------------------- arm: dyn

def arm_dyn(store, sc, seed, cfg):
    net, betas0, _, on_s, lvS = overnight(store, sc, seed, cfg)
    dyn, s0, T, mats, kq = sc.dynamics, sc.s0, sc.T, list(sc.maturities), quote_k_grid()
    v0 = dyn.v0
    res, times = {}, {"overnight": round(on_s, 2)}
    recs0 = records_from_betas(net, betas0, lvS, T, cfg.n_steps, s0, v0)
    res["anchor_unbumped"] = score(recs0, dyn, s0, mats, kq, sc.ssvi, seed + 900, cfg.reprice)
    for name, dynB in dyn_variants(dyn).items():
        # kept verbatim from the legacy script; the first t0 is unused
        t0 = time.perf_counter()
        res[f"{name}/do_nothing"] = score(recs0, dynB, s0, mats, kq, sc.ssvi, seed + 910, cfg.reprice)
        times[f"{name}/do_nothing"] = 0.0

        b = dict(betas0)
        for i in (1, 2):
            t0 = time.perf_counter()
            b = beta_correction(net, b, lvS, dynB, T, cfg.n_steps, cfg.N, s0, v0,
                                cfg.sub, seed + 20 + i)
            times[f"{name}/ridge_{i}"] = round(time.perf_counter() - t0
                                               + times.get(f"{name}/ridge_{i-1}", 0.0), 2)
            res[f"{name}/ridge_{i}"] = score(records_from_betas(net, b, lvS, T, cfg.n_steps, s0, v0),
                                             dynB, s0, mats, kq, sc.ssvi, seed + 920 + i, cfg.reprice)

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
                                              dynB, s0, mats, kq, sc.ssvi, seed + 930 + i, cfg.reprice)

        recsR, dt_s = full_resolve(lvS, dynB, s0, T, cfg.implicit, seed + 40, fit_v_floor=cfg.fit_v_floor)
        times[f"{name}/full_resolve"] = round(dt_s, 2)
        res[f"{name}/full_resolve"] = score(recsR, dynB, s0, mats, kq, sc.ssvi, seed + 940, cfg.reprice)
    return {"rmse_bp": res, "build_s": times}


# ---------------------------------------------------------------- arm: seq

def arm_seq(store, sc, seed, cfg):
    net, betas0, recsS, on_s, _ = overnight(store, sc, seed, cfg)
    dyn, s0, T, mats, kq = sc.dynamics, sc.s0, sc.T, list(sc.maturities), quote_k_grid()
    v0 = dyn.v0
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
        row["refresh"] = score(recs, dyn, s0, mats, kq, pj, seed + 800 + j, cfg.reprice)

        t0 = time.perf_counter()
        b_ridge = beta_correction(net, b_ridge, lvj, dyn, T, cfg.n_steps, cfg.N, s0, v0,
                                  cfg.sub, seed + 100 + j)
        trow["ridge"] = round(time.perf_counter() - t0, 2)
        row["ridge"] = score(records_from_betas(net, b_ridge, lvj, T, cfg.n_steps, s0, v0),
                             dyn, s0, mats, kq, pj, seed + 810 + j, cfg.reprice)

        t0 = time.perf_counter()
        bK = kalman_pass(headK, bK, lvj, dyn, T, cfg.n_steps, cfg.N, s0, v0,
                         cfg.sub, seed + 200 + j)
        trow["kalman"] = round(time.perf_counter() - t0, 2)
        row["kalman"] = score(records_from_head(headK, bK, lvj, T, cfg.n_steps, s0, v0),
                              dyn, s0, mats, kq, pj, seed + 820 + j, cfg.reprice)

        t0 = time.perf_counter()
        bS = kalman_pass(headS, bS, lvj, dyn, T, cfg.n_steps, cfg.N, s0, v0,
                         cfg.sub, seed + 300 + j)
        trow["kalman_spline"] = round(time.perf_counter() - t0, 2)
        row["kalman_spline"] = score(records_from_head(headS, bS, lvj, T, cfg.n_steps, s0, v0),
                                     dyn, s0, mats, kq, pj, seed + 830 + j, cfg.reprice)

        recsW, dt_s = full_resolve(lvj, dyn, s0, T, cfg.implicit, seed + 400 + j,
                                   L0=recsW, n_iters=3, fit_v_floor=cfg.fit_v_floor)
        trow["resolve_warm"] = round(dt_s, 2)
        row["resolve_warm"] = score(recsW, dyn, s0, mats, kq, pj, seed + 840 + j, cfg.reprice)

        res["steps"].append(row)
        times[f"step_{j}"] = trow

    recsC, dt_s = full_resolve(SSVILocalVol(path[-1], s0, T_max=T), dyn, s0, T, cfg.implicit, seed + 500,
                               fit_v_floor=cfg.fit_v_floor)
    times["resolve_cold_final"] = round(dt_s, 2)
    res["resolve_cold_final"] = score(recsC, dyn, s0, mats, kq, path[-1], seed + 850, cfg.reprice)
    res["path"] = [dataclasses.asdict(pj) for pj in path]
    return {"rmse_bp": res, "build_s": times}


# ---------------------------------------------------------------- arm: xover

def arm_xover(store, sc, seed, cfg):
    net, betas0, _, on_s, _ = overnight(store, sc, seed, cfg)
    dyn, s0, T, mats, kq = sc.dynamics, sc.s0, sc.T, list(sc.maturities), quote_k_grid()
    v0 = dyn.v0
    res, times = {}, {"overnight": round(on_s, 2)}
    for s in cfg.xover_scales:
        pB, eff = scaled_bump(sc.ssvi, s)
        lvB = SSVILocalVol(pB, s0, T_max=T)
        tag = f"s{s:g}"
        res[f"{tag}/scale_eff"] = eff

        recs = records_from_betas(net, betas0, lvB, T, cfg.n_steps, s0, v0)
        res[f"{tag}/refresh"] = score(recs, dyn, s0, mats, kq, pB, seed + 600, cfg.reprice)
        times[f"{tag}/refresh"] = 0.0

        t0 = time.perf_counter()
        b = beta_correction(net, dict(betas0), lvB, dyn, T, cfg.n_steps, cfg.N, s0, v0,
                            cfg.sub, seed + 610)
        times[f"{tag}/ridge_1"] = round(time.perf_counter() - t0, 2)
        res[f"{tag}/ridge_1"] = score(records_from_betas(net, b, lvB, T, cfg.n_steps, s0, v0),
                                      dyn, s0, mats, kq, pB, seed + 620, cfg.reprice)

        recsR, dt_s = full_resolve(lvB, dyn, s0, T, cfg.implicit, seed + 630, fit_v_floor=cfg.fit_v_floor)
        times[f"{tag}/full_resolve"] = round(dt_s, 2)
        res[f"{tag}/full_resolve"] = score(recsR, dyn, s0, mats, kq, pB, seed + 640, cfg.reprice)
    return {"rmse_bp": res, "build_s": times}


# ---------------------------------------------------------------- arm: norm

def arm_norm(store, sc, seed, cfg, eps=0.02):
    net, betas0, _, _, lvS = overnight(store, sc, seed, cfg)
    dyn, s0, T = sc.dynamics, sc.s0, sc.T
    v0 = dyn.v0
    n_steps, dt = cfg.n_steps, T / cfg.n_steps
    base_recs = records_from_betas(net, betas0, lvS, T, n_steps, s0, v0)
    L_field = base_recs.L_matrix()

    def F_of(recs, sim_seed, fit_seed):
        rng = np.random.default_rng(sim_seed)
        slices = simulate_slices(recs, dyn, s0, T, n_steps, cfg.N, rng)
        fr = np.random.default_rng(fit_seed)
        head = GlobalRidge(net, T)
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
        recs_p = LeverageField([Slice(s.t, s.grid, np.clip(s.L + eps * u[k], 0.0, 4.0), s.f)
                               for k, s in enumerate(base_recs)])
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


def log_warm_doc(h, arm, doc):
    if arm == "norm":
        m = {"A_norm": doc["A_norm"], "eps": doc["eps"]}
        if doc.get("error_fraction_pred") is not None:
            m["error_fraction_pred"] = doc["error_fraction_pred"]
        h.log_metrics(m)
        for i, a in enumerate(doc["A_norm_history"]):
            h.log_metrics({"A_norm_history": a}, step=i)
    elif arm == "seq":
        for j, row in enumerate(doc["rmse_bp"]["steps"]):
            h.log_metrics(flatten_metrics("rmse_bp", row), step=j)
            h.log_metrics(flatten_metrics("build_s", doc["build_s"][f"step_{j}"]), step=j)
        h.log_metrics({"rmse_bp/resolve_cold_final": doc["rmse_bp"]["resolve_cold_final"],
                       "build_s/resolve_cold_final": doc["build_s"]["resolve_cold_final"],
                       "build_s/overnight": doc["build_s"]["overnight"]})
    else:
        h.log_metrics(flatten_metrics("rmse_bp", doc["rmse_bp"]))
        h.log_metrics(flatten_metrics("build_s", doc["build_s"]))
    h.log_metrics({"wall_s": doc["wall_s"]})
    h.log_json("result.json", doc)


def run_warm(store, arms, sids, cfg):
    reg = make_registry()
    n = 0
    for arm, sid, seed in job_list(arms, sids):
        key = {"arm": arm, "sid": sid, "seed": seed}
        if store.find_finished(WARM_EXPERIMENT, key) is not None:
            continue
        with store.run(WARM_EXPERIMENT, {**key, **cfg.as_params(), "git_hash": git_hash()}) as h:
            t0 = time.perf_counter()
            doc = ARMS[arm](store, reg[sid], seed, cfg)
            doc.update({"arm": arm, "sid": sid, "seed": seed, "wall_s": round(time.perf_counter() - t0, 1)})
            log_warm_doc(h, arm, to_jsonable(doc))
        n += 1
        print(f"{arm} {sid} s{seed}: done in {doc['wall_s']}s", flush=True)
    return n


def summarise(store):
    """Markdown summary rebuilt from result.json artifacts (same layout as the legacy summary.md)."""
    df = store.search(WARM_EXPERIMENT, "attributes.status = 'FINISHED'")
    rows = []
    with tempfile.TemporaryDirectory() as d:
        for rid in df.get("run_id", []):
            rows.append(json.loads(store.download(rid, "result.json", Path(d) / rid).read_text()))
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
            strategies = list(docs[0]["rmse_bp"]["steps"][0].keys())
            lines.append("| step | " + " | ".join(strategies) + " |")
            lines.append("|" + "---|" * (len(strategies) + 1))
            for j in range(n_steps):
                vals = [np.mean([d["rmse_bp"]["steps"][j][s] for d in docs]) for s in strategies]
                lines.append(f"| {j} | " + " | ".join(f"{v:.0f}" for v in vals) + " |")
            lines.append(f"cold re-solve at final step: "
                         f"{np.mean([d['rmse_bp']['resolve_cold_final'] for d in docs]):.0f} bp")
        elif arm == "norm":
            for d in docs:
                lines.append(f"- {d['sid']} s{d['seed']}: |A| = {d['A_norm']}, "
                             f"predicted error fraction {d['error_fraction_pred']}, "
                             f"history {d['A_norm_history']}")
        else:
            keys = sorted({k for d in docs for k in d["rmse_bp"] if not k.endswith("scale_eff")})
            lines.append("| strategy | mean rmse bp | mean build s |")
            lines.append("|---|---|---|")
            for k in keys:
                vals = [d["rmse_bp"][k] for d in docs if k in d["rmse_bp"]]
                ts = [d["build_s"].get(k, "") for d in docs if k in d["rmse_bp"]]
                ts = [t for t in ts if t != ""]
                lines.append(f"| {k} | {np.mean(vals):.0f} | {np.mean(ts):.1f} |" if ts
                             else f"| {k} | {np.mean(vals):.0f} | |")
        lines.append("")
    return "\n".join(lines)
