"""Run one (scenario, algo, N, seed) and write a self-describing result JSON."""
import dataclasses
import json
import subprocess
import traceback
from pathlib import Path

import numpy as np

from neural_particle_method.reprice import reprice_iv, iv_metrics
from neural_particle_method.ssvi import implied_vol_ssvi

from .algos import run_algo
from .scenarios import full_registry, quote_k_grid


def run_path(sid, algo, n_particles, seed, results_dir="results/runs"):
    return Path(results_dir) / sid / algo / f"n{n_particles}_s{seed}.json"


def _git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if not np.isfinite(f) else f
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def run_one(sid, algo, n_particles, seed, results_dir="results/runs", cfg=None,
            reprice_particles=500_000, reprice_steps=100):
    sc = full_registry()[sid]
    out = run_path(sid, algo, n_particles, seed, results_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": 1, "sid": sid, "algo": algo, "n_particles": n_particles,
           "seed": seed, "git_hash": _git_hash(),
           "scenario": {"ssvi": dataclasses.asdict(sc.ssvi), "dynamics": sc.dynamics,
                        "s0": sc.s0, "T": sc.T, "maturities": list(sc.maturities)}}
    try:
        res = run_algo(algo, sc, n_particles, seed, cfg)
        k = quote_k_grid()
        mats = list(sc.maturities)
        iv_model = reprice_iv(res.L_records, sc.dynamics, sc.s0, mats, k,
                              n_particles=reprice_particles, n_steps=reprice_steps,
                              seed=seed + 10_000)
        iv_target = np.stack([implied_vol_ssvi(sc.ssvi, k, m) for m in mats])
        doc.update(status="ok", timings=_jsonable(res.timings),
                   diagnostics=_jsonable(res.diagnostics),
                   metrics=_jsonable(iv_metrics(iv_model, iv_target, k, mats)),
                   iv_err_bp=_jsonable(((iv_model - iv_target) * 1e4).tolist()))
    except Exception:
        doc.update(status="failed", error=traceback.format_exc())
    out.write_text(json.dumps(doc, indent=1))
    return out
