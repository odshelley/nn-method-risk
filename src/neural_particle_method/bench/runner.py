"""Run one (scenario, algo, N, seed) and record it in the store."""
import json
import tempfile

import numpy as np

from ..calibrate.config import ExplicitConfig, ImplicitConfig
from ..pricing.metrics import iv_metrics
from ..pricing.reprice import RepriceConfig, reprice_iv, snap_times
from ..simulate.leverage import DEFAULT_GRID, LeverageField
from ..tracking.store import git_hash, to_jsonable
from .algos import run_algo
from .scenarios import full_registry, quote_k_grid

BENCH_EXPERIMENT = "bench"
PDE_EXPERIMENT = "pde_reference"
SCHEMA = 2


def run_key(sid, algo, n_particles, seed):
    return {"sid": sid, "algo": algo, "n_particles": int(n_particles), "seed": int(seed)}


def _prefixed(prefix, d):
    return {f"{prefix}.{k}": v for k, v in d.items()}


def _leverage_error(store, sid, field, n_steps, lag="none"):
    """RMSE against the PDE reference leverage for this scenario at the same step count and lag."""
    rid = store.find_finished(PDE_EXPERIMENT, {"sid": sid, "n_steps": int(n_steps), "lag": lag})
    if rid is None:
        return {}
    with tempfile.TemporaryDirectory() as d:
        ref = LeverageField.from_json(json.loads(store.download(rid, "leverage.json", d).read_text()))
    a, b = field.resample(DEFAULT_GRID), ref.resample(DEFAULT_GRID)
    n = min(len(a), len(b))
    if not np.allclose(a.times[:n], b.times[:n], atol=1e-9):
        print(f"warning: pde_reference for {sid} has a different time grid; lev_rmse skipped")
        return {}
    per = {f"lev_rmse/T{a[k].t:g}": float(np.sqrt(np.mean((a[k].L - b[k].L) ** 2))) for k in range(n)}
    pooled = float(np.sqrt(np.mean([(a[k].L - b[k].L) ** 2 for k in range(n)])))
    return {"lev_rmse": pooled, **per}


def run_one(store, sid, algo, n_particles, seed, explicit=ExplicitConfig(), implicit=ImplicitConfig(),
            reprice=RepriceConfig(), *, experiment=BENCH_EXPERIMENT, knobs=None, extra_key=None,
            save_model=None, mixture=None):
    key = {**run_key(sid, algo, n_particles, seed), **(extra_key or {})}
    existing = store.find_finished(experiment, key)
    if existing is not None:
        return existing
    sc = full_registry()[sid]
    params = {**key, "git_hash": git_hash(), "schema": SCHEMA, "budget": int(n_particles),
              **_prefixed("explicit", explicit.as_params()), **_prefixed("implicit", implicit.as_params()),
              **_prefixed("reprice", reprice.as_params()), **sc.as_params(),
              **_prefixed("estimator", knobs or {})}
    with store.run(experiment, params) as h:
        res = run_algo(algo, sc, n_particles, seed, explicit, implicit, knobs=knobs, mixture=mixture)
        k = quote_k_grid()
        mats = list(sc.maturities)
        iv_model = reprice_iv(res.field, sc.dynamics, sc.s0, mats, k, reprice, seed=seed + 10_000)
        iv_target = sc.target_ivs(k, snap_times(mats, reprice.n_steps))
        m = iv_metrics(iv_model, iv_target, k, mats)
        metrics = {c: m[c] for c in ("pooled_rmse_bp", "pooled_max_bp", "wings_rmse_bp", "wings_max_bp", "n_failed")}
        metrics.update({f"rmse_bp/T{r['T']:g}": r["rmse_bp"] for r in m["per_maturity"]})
        metrics.update({"pooled_mae_bp": m["pooled_mae_bp"], "wings_mae_bp": m["wings_mae_bp"],
                       "liquid_mae_bp": m["liquid_mae_bp"]})
        metrics.update({f"mae_bp/T{r['T']:g}": r["mae_bp"] for r in m["mae_per_maturity"]})
        metrics.update(res.timings)
        if "is_diag" in res.diagnostics:
            metrics.update({k: float(v) for k, v in res.diagnostics["is_diag"].items()})
        for c in ("intraday_s", "overnight_s"):
            if c in res.diagnostics:
                metrics[c] = res.diagnostics[c]
        metrics["budget"] = int(n_particles)
        if extra_key and "knob_value" in extra_key:
            metrics["knob_value"] = float(extra_key["knob_value"])
        metrics.update(_leverage_error(store, sid, res.field, explicit.n_steps))
        h.log_metrics(metrics)
        h.log_json("leverage.json", res.field.to_json())
        h.log_json("iv_err_bp.json", ((iv_model - iv_target) * 1e4).tolist())
        h.log_json("diagnostics.json", to_jsonable(res.diagnostics))
        if save_model is not None:
            save_model(h, res)
        return h.run_id
