"""Small-online-budget sweep: long-trained explicit body (500k) with online heads against NW
re-solved at the same budget, for online budgets of 10k, 30k and 80k particles, both lags, six
scenarios. Every fit uses the full online cloud. Bodies come from `suite_offline_conv`; results go
to `suite_budget`, one run per (sid, method, budget, lag, seed).

Usage: python scripts/budget_sweep.py [jobs]
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

from neural_particle_method.bench.scenarios import full_registry
from neural_particle_method.calibrate.explicit import calibrate_explicit
from neural_particle_method.estimators import make_estimator
from neural_particle_method.suite.artifacts import load_run
from neural_particle_method.suite.config import FULL
from neural_particle_method.suite.heads import FeatureRidgeHead, RKHSHead, SplineHead, online_sweep
from neural_particle_method.suite.lag import LAGS, lagged_scenario
from neural_particle_method.suite.reference import score_field
from neural_particle_method.tracking.store import Store, git_hash

EXPERIMENT = "suite_budget"
BODY_EXPERIMENT = "suite_offline_conv"
SIDS = ("s01", "s02", "s05", "s09", "s11", "s16")
BUDGETS = (10_000, 30_000, 80_000)
METHODS = ("nw_resolve", "explicit_stale", "explicit_rkhs", "explicit_spline", "explicit_ridge")
SEEDS = (0, 1)
N_BODY = 500_000


def _head(method):
    return {"explicit_rkhs": RKHSHead(), "explicit_spline": SplineHead(),
            "explicit_ridge": FeatureRidgeHead()}.get(method)


def run_cell(store, sid, method, budget, lag, seed):
    key = {"sid": sid, "method": method, "budget": int(budget), "lag": lag.kind, "seed": int(seed),
           "n_steps": 200}
    existing = store.find_finished(EXPERIMENT, key)
    if existing is not None:
        return existing
    sc = full_registry()[sid]
    lsc = lagged_scenario(sc, lag)
    lv = lsc.local_vol()
    ecfg = replace(FULL.explicit, n_particles=int(budget), fit_subsample=int(budget))
    body_key = {"sid": sid, "body": "explicit", "n_particles": N_BODY, "seed": 0, "n_steps": 200}
    body_rid = store.find_finished(BODY_EXPERIMENT, body_key)
    if body_rid is None:
        raise RuntimeError(f"no long-trained explicit body for {sid} in {BODY_EXPERIMENT}")
    body = load_run(store, body_rid)
    head = _head(method)
    params = {**key, "git_hash": git_hash(), "body_run": body_rid, **lsc.as_params(),
              **{f"explicit.{k}": v for k, v in ecfg.as_params().items()}}
    if head is not None:
        params.update({f"head.{k}": v for k, v in head.params.items()})
    with store.run(EXPERIMENT, params) as h:
        t0 = time.perf_counter()
        if method == "nw_resolve":
            est = make_estimator("nw", seed=seed + 1, local_vol=lv, s0=lsc.s0)
            field = calibrate_explicit(lv, lsc.dynamics, est, ecfg, s0=lsc.s0, T=lsc.T,
                                       seed=seed + 1).field
        else:
            field, _ = online_sweep(body.model, lv, lsc.dynamics, lsc.s0, lsc.T, ecfg, head=head,
                                    seed=seed + 1)
        online_s = time.perf_counter() - t0 if method != "explicit_stale" else 0.0
        metrics, err = score_field(field, lsc, seed, FULL.reprice)
        metrics.update({"online_s": online_s, "fit_s": online_s, "total_s": online_s})
        h.log_metrics(metrics)
        h.log_json("leverage.json", field.to_json())
        h.log_json("iv_err_bp.json", err)
        return h.run_id


def jobs():
    return [(sid, m, b, lag.kind, seed) for sid in SIDS for lag in LAGS for seed in SEEDS
            for b in BUDGETS for m in METHODS]


def _worker(args):
    uri, root, job = args
    lag = {lg.kind: lg for lg in LAGS}[job[3]]
    try:
        run_cell(Store(uri, root), job[0], job[1], job[2], lag, job[4])
        return job, None
    except Exception as e:  # noqa: BLE001
        return job, repr(e)


if __name__ == "__main__":
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    store = Store()
    store.experiment_id(EXPERIMENT)
    todo = jobs()
    print(f"{len(todo)} cells", flush=True)
    args = [(store.tracking_uri, store.artifact_root, j) for j in todo]
    done = failed = 0
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for job, err in ex.map(_worker, args):
            if err is None:
                done += 1
                print("done:", *job, flush=True)
            else:
                failed += 1
                print("FAILED:", *job, err, flush=True)
    print(f"budget: {done} done, {failed} failed", flush=True)
