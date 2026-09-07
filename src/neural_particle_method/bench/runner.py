"""Run one (scenario, algo, N, seed) and record it in the store."""
import dataclasses

from ..calibrate.config import ExplicitConfig, ImplicitConfig
from ..pricing.metrics import iv_metrics, target_ivs
from ..pricing.reprice import RepriceConfig, reprice_iv
from ..tracking.store import git_hash, to_jsonable
from .algos import run_algo
from .scenarios import full_registry, quote_k_grid

BENCH_EXPERIMENT = "bench"
SCHEMA = 2


def run_key(sid, algo, n_particles, seed):
    return {"sid": sid, "algo": algo, "n_particles": int(n_particles), "seed": int(seed)}


def _prefixed(prefix, d):
    return {f"{prefix}.{k}": v for k, v in d.items()}


def run_one(store, sid, algo, n_particles, seed, explicit=ExplicitConfig(), implicit=ImplicitConfig(),
            reprice=RepriceConfig()):
    key = run_key(sid, algo, n_particles, seed)
    existing = store.find_finished(BENCH_EXPERIMENT, key)
    if existing is not None:
        return existing
    sc = full_registry()[sid]
    params = {**key, "git_hash": git_hash(), "schema": SCHEMA,
              **_prefixed("explicit", explicit.as_params()), **_prefixed("implicit", implicit.as_params()),
              **_prefixed("reprice", reprice.as_params()),
              **_prefixed("scenario.ssvi", dataclasses.asdict(sc.ssvi)),
              **_prefixed("scenario.dynamics", sc.dynamics.to_dict()),
              "scenario.s0": sc.s0, "scenario.T": sc.T, "scenario.maturities": str(list(sc.maturities))}
    with store.run(BENCH_EXPERIMENT, params) as h:
        res = run_algo(algo, sc, n_particles, seed, explicit, implicit)
        k = quote_k_grid()
        mats = list(sc.maturities)
        iv_model = reprice_iv(res.field, sc.dynamics, sc.s0, mats, k, reprice, seed=seed + 10_000)
        iv_target = target_ivs(sc.ssvi, k, mats, reprice.n_steps)
        m = iv_metrics(iv_model, iv_target, k, mats)
        metrics = {c: m[c] for c in ("pooled_rmse_bp", "pooled_max_bp", "wings_rmse_bp", "wings_max_bp", "n_failed")}
        metrics.update({f"rmse_bp/T{r['T']:g}": r["rmse_bp"] for r in m["per_maturity"]})
        metrics.update(res.timings)
        for c in ("intraday_s", "overnight_s"):
            if c in res.diagnostics:
                metrics[c] = res.diagnostics[c]
        h.log_metrics(metrics)
        h.log_json("leverage.json", res.field.to_json())
        h.log_json("iv_err_bp.json", ((iv_model - iv_target) * 1e4).tolist())
        h.log_json("diagnostics.json", to_jsonable(res.diagnostics))
        return h.run_id
