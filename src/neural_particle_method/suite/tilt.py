"""Importance-sampling study: does the Girsanov tilt buy budget equivalence?

Layer 1 (`tilt_slices`): particles under the frozen PDE leverage, no feedback, per-slice
estimates at the quoted strikes against the PDE conditional expectation. Layer 2: cold
calibration (`tilt_cold`) and the frozen body + spline head (`tilt_online`) with and without the
tilt, scored like section 4. Spec: docs/superpowers/specs/2026-09-13-importance-sampling-design.md.
"""
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..bench.reference_runs import run_reference
from ..bench.runner import run_one
from ..bench.scenarios import full_registry, quote_k_grid
from ..calibrate.importance import DESIGNS, UNTILTED, design_by_name
from ..estimators.nadaraya_watson import nw_estimate
from ..estimators.recipes import load_recipe, recipe_hash, regressor_from_recipe
from ..pricing.reprice import snap_times
from ..simulate.dynamics import HestonParams
from ..simulate.leverage import LeverageField
from ..simulate.stepper import heston_step
from ..tracking.store import Store, git_hash
from .budget import run_budget_cell
from .cold import NN_ALGOS, cold_extra_key
from .config import FULL, SSVI_SIDS
from .grid import _drain
from .lag import LAGS

SLICE_SIDS = ("s01", "s02", "s05", "s09", "s11", "s16")
SLICE_PARTICLES = (10_000, 30_000, 80_000)
SLICE_SEEDS = (0, 1, 2, 3, 4)
COLD_PARTICLES = (10_000, 30_000, 80_000)
COLD_SEEDS = (0, 1)
COLD_ALGO_SIDS = {"nw": SSVI_SIDS, "explicit_nn_opt": SLICE_SIDS}
ONLINE_BUDGETS = (10_000, 80_000)
ONLINE_SEEDS = (0, 1)
ONLINE_METHOD = "explicit_opt_spline"
CLOUD_KEEP = 5_000
WING_CUT = 0.25


def mixture_for(design_name, n_steps, T, rho):
    """The MixtureDesign for a design name at this scenario's step count; None for "none"."""
    d = design_by_name(design_name)
    return None if d is None else d.mixture(n_steps, T, rho)


def run_tilt_cold(store, sid, algo, n_particles, seed, design_name, settings=FULL):
    """One cold calibration, tilted or not, scored as the cold suite is; keyed by `design`."""
    if algo not in COLD_ALGO_SIDS:
        raise ValueError(f"tilt cold rows are {tuple(COLD_ALGO_SIDS)}, got {algo!r}")
    sc = full_registry()[sid]
    mixture = mixture_for(design_name, settings.explicit.n_steps, sc.T, sc.dynamics.rho)
    knobs = {"keep_slice_weights": True} if algo in NN_ALGOS else None
    extra = {"design": design_name, **(cold_extra_key(algo) or {})}
    return run_one(store, sid, algo, int(n_particles), seed, settings.explicit, settings.implicit,
                   settings.reprice, experiment=settings.experiment("tilt_cold"), knobs=knobs,
                   extra_key=extra, mixture=mixture)


def _restrict(sids, wanted):
    return tuple(sids) if wanted is None else tuple(s for s in sids if s in set(wanted))


def cold_jobs(design_name, settings=FULL, sids=None, particles=COLD_PARTICLES, seeds=COLD_SEEDS):
    """Untilted and tilted cold cells, paired by seed, for both algorithms."""
    design_by_name(design_name)      # KeyError for an unknown design
    if design_name == UNTILTED:
        raise ValueError("the untilted arm is not a design for these stages; every cell would "
                         "be emitted twice")
    jobs = []
    for algo, algo_sids in COLD_ALGO_SIDS.items():
        for sid in _restrict(algo_sids, sids):
            for n in particles:
                for seed in seeds:
                    for d in (UNTILTED, design_name):
                        jobs.append(("cold", sid, algo, int(n), int(seed), d))
    return jobs


def run_tilt_online(store, sid, budget, lag, seed, design_name, settings=FULL, recipe=None):
    """Frozen searched body + weighted spline head on a tilted online cloud."""
    design_by_name(design_name)
    if design_name == UNTILTED:
        raise ValueError("the untilted online cells are the suite_budget_tuned cells")
    return run_budget_cell(store, sid, ONLINE_METHOD, int(budget), lag, int(seed),
                           body="explicit_opt", settings=settings, recipe=recipe,
                           design=design_name)


def online_jobs(design_name, settings=FULL, sids=None, budgets=ONLINE_BUDGETS, seeds=ONLINE_SEEDS):
    design_by_name(design_name)
    return [("online", sid, int(b), lag.kind, int(seed), design_name)
            for sid in _restrict(SSVI_SIDS, sids) for lag in LAGS
            for seed in seeds for b in budgets]


def simulate_frozen(field, params, s0, T, n_steps, n_particles, seed, mixture=None,
                    keep_times=()):
    """Particles under a frozen leverage field, tilted or not, with the calibrator's weights.

    Returns ({t: (lnx, v_plus, w)} at the requested (grid-snapped) times, [per-step ESS]).
    Clouds are taken at the start of the step whose time equals `t` (the same convention as the
    field's slices)."""
    hp = HestonParams.from_dict(params) if isinstance(params, dict) else params
    rng = np.random.default_rng(seed)
    dt, sdt = T / n_steps, np.sqrt(T / n_steps)
    lnx = np.full(n_particles, np.log(s0))
    v = np.full(n_particles, hp.v0)
    w = np.ones(n_particles)
    theta_p, ess_path = None, []
    if mixture is not None:
        comp = rng.choice(3, size=n_particles, p=list(mixture.alphas))
        thetas_all, etas_all = np.asarray(mixture.thetas), np.asarray(mixture.etas)
        alphas = np.array(mixture.alphas)
        ell = np.zeros((3, n_particles))
    wanted = {int(round(t / dt)): float(t) for t in keep_times}  # noqa: RUF046
    out = {}
    for k in range(n_steps + 1):
        if k in wanted:
            out[wanted[k]] = (lnx.copy(), np.maximum(v, 0.0), w.copy())
        if k == n_steps:
            break
        t = k * dt
        if mixture is not None:
            th = thetas_all[:, k] if mixture.scheduled else thetas_all
            etas = etas_all[:, k] if mixture.scheduled else etas_all
            theta_p, eta_p = th[comp], etas[comp]
        L_p = field.at(t, lnx)
        zb, zp = rng.standard_normal(n_particles), rng.standard_normal(n_particles)
        lnx, v = heston_step(lnx, v, L_p, zb, zp, hp, dt, sdt, theta_p=theta_p)
        if mixture is not None:
            dbperp = zp * sdt + eta_p * dt
            ell += etas[:, None] * dbperp[None, :] - 0.5 * (etas ** 2)[:, None] * dt
            w = 1.0 / (alphas @ np.exp(np.clip(ell, -60, 60)))
            ess_path.append(float(w.sum() ** 2 / (n_particles * (w ** 2).sum())))
    return out, ess_path


def _local_ess(lnx, w, k, bandwidth):
    ker = np.exp(-0.5 * ((k[:, None] - lnx[None, :]) / bandwidth) ** 2) * w[None, :]
    return (ker.sum(axis=1) ** 2) / np.clip((ker ** 2).sum(axis=1), 1e-300, None)


def slice_estimates(lnx, v, w, k, recipe, seed, monotone_sign):
    """NW and a fresh network fit on one weighted cloud, evaluated at the strikes `k`."""
    bw = 1.06 * np.std(lnx) * len(lnx) ** (-1 / 5)
    weights = None if np.all(w == 1.0) else w
    t0 = time.perf_counter()
    f_nw = nw_estimate(lnx, v, k, weights=weights, bandwidth=bw)
    s_nw = time.perf_counter() - t0
    net = regressor_from_recipe(recipe, seed=seed, monotone_sign=monotone_sign)
    t0 = time.perf_counter()
    f_net = net.fit_predict(0.0, lnx, v, k, weights=weights)
    s_net = time.perf_counter() - t0
    return {"nw": f_nw, "net": f_net, "ess_local": _local_ess(lnx, w, k, bw),
            "bandwidth": bw, "fit_s": {"nw": s_nw, "net": s_net}}


def _reference_field(store, sid, settings):
    rid = run_reference(store, sid, n_steps=settings.explicit.n_steps, n_x=settings.n_x,
                        n_v=settings.n_v)
    with tempfile.TemporaryDirectory() as d:
        raw = store.download(rid, "leverage.json", d).read_text()
    return rid, LeverageField.from_json(json.loads(raw))


def _slice_at(field, t):
    i = max(int(np.searchsorted(field.times, t + 1e-12)) - 1, 0)
    return field[i]


def run_slice_cell(store, sid, n_particles, design_name, seed, settings=FULL):
    """One layer-1 cell: a frozen-leverage cloud scored slice by slice against the PDE."""
    n_steps = settings.explicit.n_steps
    key = {"sid": sid, "n_particles": int(n_particles), "design": design_name, "seed": int(seed),
           "n_steps": int(n_steps)}
    exp = settings.experiment("tilt_slices")
    existing = store.find_finished(exp, key)
    if existing is not None:
        return existing
    sc = full_registry()[sid]
    recipe = load_recipe("explicit_opt")
    ref_rid, ref = _reference_field(store, sid, settings)
    mixture = mixture_for(design_name, n_steps, sc.T, sc.dynamics.rho)
    # a maturity that snaps to t = 0 is a one-point cloud with no bandwidth: skip it, and score
    # each snapped time once (coarse grids can map two maturities onto one slice)
    mats = list(dict.fromkeys(t for t in snap_times(list(sc.maturities), n_steps, T=sc.T)
                              if t > 0))
    k = quote_k_grid() + np.log(sc.s0)
    wings = np.abs(quote_k_grid()) > WING_CUT
    lv = sc.local_vol()
    params = {**key, "git_hash": git_hash(), "reference_run": ref_rid,
              "recipe_hash": recipe_hash(recipe), **sc.as_params()}
    with store.run(exp, params) as h:
        t0 = time.perf_counter()
        clouds, ess_path = simulate_frozen(ref, sc.dynamics, sc.s0, sc.T, n_steps,
                                           int(n_particles), seed, mixture=mixture,
                                           keep_times=mats)
        sim_s = time.perf_counter() - t0
        doc = {"maturities": mats, "k": k.tolist(), "f_ref": [], "L_ref": [],
               "f_hat": {"nw": [], "net": []}, "L_hat": {"nw": [], "net": []},
               "f_rel_err": {"nw": [], "net": []}, "lev_rel_err": {"nw": [], "net": []},
               "ess_local": [], "ess_slice": []}
        fit_s = {"nw": 0.0, "net": 0.0}
        rng = np.random.default_rng(seed + 77)
        for t in mats:
            lnx, v, w = clouds[t]
            s = _slice_at(ref, t)
            f_ref = np.interp(k, s.grid, s.f)
            L_ref = np.interp(k, s.grid, s.L)
            sig = lv.sigma(max(t, lv.T_grid[0]), np.exp(k), sc.s0)
            est = slice_estimates(lnx, v, w, k, recipe, seed,
                                  -float(np.sign(sc.dynamics.rho) or 1.0))
            doc["f_ref"].append(f_ref.tolist()); doc["L_ref"].append(L_ref.tolist())
            for name in ("nw", "net"):
                f_hat = np.clip(est[name], 1e-4, None)
                L_hat = np.clip(sig / np.sqrt(f_hat), 0.0, settings.explicit.L_max)
                doc["f_hat"][name].append(f_hat.tolist())
                doc["L_hat"][name].append(L_hat.tolist())
                doc["f_rel_err"][name].append(((f_hat - f_ref) / f_ref).tolist())
                doc["lev_rel_err"][name].append(((L_hat - L_ref) / L_ref).tolist())
                fit_s[name] += est["fit_s"][name]
            doc["ess_local"].append(est["ess_local"].tolist())
            doc["ess_slice"].append(float(w.sum() ** 2 / (len(w) * (w ** 2).sum())))
            keep = rng.choice(len(lnx), size=min(CLOUD_KEEP, len(lnx)), replace=False)
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / f"cloud_T{t:g}.npz"
                np.savez_compressed(p, lnx=lnx[keep].astype(np.float32),
                                    v=v[keep].astype(np.float32), w=w[keep].astype(np.float32))
                h.log_file(p)
        h.log_json("slice_scores.json", doc)
        w_last = clouds[mats[-1]][2]
        metrics = {"sim_s": sim_s, "fit_s/nw": fit_s["nw"], "fit_s/net": fit_s["net"],
                   "ess_final": float(w_last.sum() ** 2 / (len(w_last) * (w_last ** 2).sum())),
                   "ess_min_slice": float(min(ess_path)) if ess_path else 1.0,
                   "max_w": float(w_last.max())}
        for name in ("nw", "net"):
            e = np.abs(np.array(doc["f_rel_err"][name]))[:, wings]
            metrics[f"wings_abs_f_rel/{name}"] = float(e.mean())
        h.log_metrics(metrics)
        return h.run_id


def slice_jobs(settings=FULL, sids=None, particles=SLICE_PARTICLES, seeds=SLICE_SEEDS):
    """One cell per (sid, particle count, design incl. untilted, seed)."""
    names = [UNTILTED] + [d.name for d in DESIGNS]
    return [("slice", sid, int(n), d, int(seed)) for sid in _restrict(SLICE_SIDS, sids)
            for n in particles for d in names for seed in seeds]


SMOKE_GRID = {"sids": ("s01",), "seeds": (0,), "designs": (UNTILTED, "constant-3")}


def run_tilt_job(store, job, settings=FULL, recipe=None):
    kind = job[0]
    if kind == "slice":
        return run_slice_cell(store, job[1], job[2], job[3], job[4], settings)
    if kind == "cold":
        return run_tilt_cold(store, job[1], job[2], job[3], job[4], job[5], settings)
    if kind == "online":
        lag = {lg.kind: lg for lg in LAGS}[job[3]]
        return run_tilt_online(store, job[1], job[2], lag, job[4], job[5], settings, recipe=recipe)
    raise KeyError(kind)


def _is_finished(store, job, settings):
    kind, n_steps = job[0], int(settings.explicit.n_steps)
    if kind == "slice":
        key = {"sid": job[1], "n_particles": job[2], "design": job[3], "seed": job[4],
               "n_steps": n_steps}
        return store.find_finished(settings.experiment("tilt_slices"), key) is not None
    if kind == "cold":
        key = {"sid": job[1], "algo": job[2], "n_particles": job[3], "seed": job[4],
               "design": job[5], **(cold_extra_key(job[2]) or {})}
        return store.find_finished(settings.experiment("tilt_cold"), key) is not None
    if kind == "online":
        key = {"sid": job[1], "method": ONLINE_METHOD, "budget": job[2], "lag": job[3],
               "seed": job[4], "n_steps": n_steps, "design": job[5],
               "recipe_hash": recipe_hash(load_recipe("explicit_opt"))}
        return store.find_finished(settings.experiment("tilt_online"), key) is not None
    raise KeyError(kind)


def _worker(args):
    uri, root, job, settings, n_jobs = args
    if n_jobs > 1:
        import torch
        k = max(1, (os.cpu_count() or n_jobs) // n_jobs)
        os.environ.setdefault("OMP_NUM_THREADS", str(k))
        torch.set_num_threads(k)
    try:
        run_tilt_job(Store(uri, root), job, settings)
        return job, None
    except Exception as e:  # noqa: BLE001 - the run is already recorded FAILED by Store.run
        return job, repr(e)


def tilt_jobs(stage, settings=FULL, sids=None, design=None, particles=None, seeds=None,
              designs=None, budgets=None):
    if stage == "slices":
        kw = {k: v for k, v in (("particles", particles), ("seeds", seeds)) if v is not None}
        jobs = slice_jobs(settings, sids, **kw)
        if designs is not None:
            jobs = [j for j in jobs if j[3] in set(designs)]
        return jobs
    if design is None:
        raise ValueError(f"stage {stage!r} needs a design name (nparticle tilt winner)")
    if design == UNTILTED:
        raise ValueError(f"stage {stage!r}: the untilted arm is not a design for these stages")
    if stage == "cold":
        kw = {k: v for k, v in (("particles", particles), ("seeds", seeds)) if v is not None}
        return cold_jobs(design, settings, sids, **kw)
    if stage == "online":
        kw = {k: v for k, v in (("budgets", budgets), ("seeds", seeds)) if v is not None}
        return online_jobs(design, settings, sids, **kw)
    raise KeyError(stage)


def run_tilt_stage(store, stage, settings=FULL, n_jobs=1, sids=None, design=None, particles=None,
                   seeds=None, designs=None, budgets=None):
    """Run every unfinished cell of `stage`; returns (n_done, n_failed). Pre-creates the
    experiments (MLflow race) and, for online, the body experiment."""
    for name in ("tilt_slices", "tilt_cold", "tilt_online", "suite_offline", "suite_budget_tuned"):
        store.experiment_id(settings.experiment(name))
    store.experiment_id("pde_reference")
    jobs = tilt_jobs(stage, settings, sids, design, particles, seeds, designs, budgets)
    if stage == "slices":
        # the PDE reference of each scenario is computed once here, serially, so pooled workers
        # never race on the same (sid, n_steps) reference run
        for sid in sorted({j[1] for j in jobs}):
            run_reference(store, sid, n_steps=settings.explicit.n_steps, n_x=settings.n_x,
                          n_v=settings.n_v)
    jobs = [j for j in jobs if not _is_finished(store, j, settings)]
    print(f"tilt {stage}: {len(jobs)} jobs listed", flush=True)
    args = [(store.tracking_uri, store.artifact_root, j, settings, n_jobs) for j in jobs]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            return _drain(pool.map(_worker, args))
    return _drain(map(_worker, args))
