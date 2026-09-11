"""Small-online-budget sweep: a frozen offline body (500k) with online heads against NW re-solved
at the same budget, for online budgets of 10k, 30k and 80k particles, both lags, two seeds. Every
fit uses the full online cloud.

The default body is the tuned per-slice network (`explicit_tuned`), taken from the suite's offline
stage and trained here if that stage has not produced it yet; its cells go to `suite_budget_tuned`.
`--body explicit` is the legacy path: the short-trained body from `suite_offline_conv`, cells in
`suite_budget`. One run per (sid, method, budget, lag, seed) either way.

Usage: python scripts/budget_sweep.py [--jobs N] [--body explicit_tuned|explicit] [--sids s01 ...]
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

from neural_particle_method.bench.scenarios import full_registry
from neural_particle_method.calibrate.explicit import calibrate_explicit
from neural_particle_method.estimators import make_estimator
from neural_particle_method.suite.artifacts import load_run
from neural_particle_method.suite.config import FULL, SSVI_SIDS
from neural_particle_method.suite.heads import FeatureRidgeHead, RKHSHead, SplineHead, online_sweep
from neural_particle_method.suite.lag import LAGS, lagged_scenario
from neural_particle_method.suite.offline import run_offline
from neural_particle_method.suite.reference import score_field
from neural_particle_method.tracking.store import Store, git_hash

EXPERIMENTS = {"explicit_tuned": "suite_budget_tuned", "explicit": "suite_budget"}
LEGACY_BODY_EXPERIMENT = "suite_offline_conv"
DEFAULT_BODY = "explicit_tuned"
SIDS = SSVI_SIDS
BUDGETS = (10_000, 30_000, 80_000)
SEEDS = (0, 1)
N_BODY = 500_000
N_STEPS = 200


def methods(body):
    return ("nw_resolve", f"{body}_stale", f"{body}_rkhs", f"{body}_spline", f"{body}_ridge")


def _head(method):
    """Heads key off the method suffix; `nw_resolve` and `*_stale` have none."""
    return {"rkhs": RKHSHead(), "spline": SplineHead(),
            "ridge": FeatureRidgeHead()}.get(method.rsplit("_", 1)[-1])


def _body_run(store, sid, body):
    if body == "explicit_tuned":
        # the offline stage runs first, so this normally finds the finished body and returns it
        return run_offline(store, sid, "explicit_tuned", N_BODY, FULL)
    key = {"sid": sid, "body": "explicit", "n_particles": N_BODY, "seed": 0, "n_steps": N_STEPS}
    rid = store.find_finished(LEGACY_BODY_EXPERIMENT, key)
    if rid is None:
        raise RuntimeError(f"no long-trained explicit body for {sid} in {LEGACY_BODY_EXPERIMENT}")
    return rid


def run_cell(store, sid, method, budget, lag, seed, body=DEFAULT_BODY):
    exp = EXPERIMENTS[body]
    key = {"sid": sid, "method": method, "budget": int(budget), "lag": lag.kind, "seed": int(seed),
           "n_steps": N_STEPS}
    existing = store.find_finished(exp, key)
    if existing is not None:
        return existing
    sc = full_registry()[sid]
    lsc = lagged_scenario(sc, lag)
    lv = lsc.local_vol()
    ecfg = replace(FULL.explicit, n_particles=int(budget), fit_subsample=int(budget))
    body_rid = _body_run(store, sid, body)
    loaded = load_run(store, body_rid)
    head = _head(method)
    params = {**key, "git_hash": git_hash(), "body_run": body_rid, "body": body, **lsc.as_params(),
              **{f"explicit.{k}": v for k, v in ecfg.as_params().items()}}
    if head is not None:
        params.update({f"head.{k}": v for k, v in head.params.items()})
    with store.run(exp, params) as h:
        t0 = time.perf_counter()
        if method == "nw_resolve":
            est = make_estimator("nw", seed=seed + 1, local_vol=lv, s0=lsc.s0)
            field = calibrate_explicit(lv, lsc.dynamics, est, ecfg, s0=lsc.s0, T=lsc.T,
                                       seed=seed + 1).field
        else:
            field, _ = online_sweep(loaded.model, lv, lsc.dynamics, lsc.s0, lsc.T, ecfg, head=head,
                                    seed=seed + 1)
        online_s = 0.0 if method.endswith("_stale") else time.perf_counter() - t0
        metrics, err = score_field(field, lsc, seed, FULL.reprice)
        metrics.update({"online_s": online_s, "fit_s": online_s, "total_s": online_s})
        h.log_metrics(metrics)
        h.log_json("leverage.json", field.to_json())
        h.log_json("iv_err_bp.json", err)
        return h.run_id


def jobs(sids=SIDS, body=DEFAULT_BODY):
    return [(sid, m, b, lag.kind, seed) for sid in sids for lag in LAGS for seed in SEEDS
            for b in BUDGETS for m in methods(body)]


def _worker(args):
    uri, root, job, body = args
    lag = {lg.kind: lg for lg in LAGS}[job[3]]
    try:
        run_cell(Store(uri, root), job[0], job[1], job[2], lag, job[4], body)
        return job, None
    except Exception as e:  # noqa: BLE001
        return job, repr(e)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="online-budget sweep on a frozen offline body")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--body", choices=sorted(EXPERIMENTS), default=DEFAULT_BODY)
    p.add_argument("--sids", nargs="+", default=list(SIDS))
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    store = Store()
    store.experiment_id(EXPERIMENTS[a.body])
    todo = jobs(tuple(a.sids), a.body)
    print(f"{len(todo)} cells on the {a.body} body", flush=True)
    args = [(store.tracking_uri, store.artifact_root, j, a.body) for j in todo]
    done = failed = 0
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for job, err in ex.map(_worker, args):
            if err is None:
                done += 1
                print("done:", *job, flush=True)
            else:
                failed += 1
                print("FAILED:", *job, err, flush=True)
    print(f"budget: {done} done, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
