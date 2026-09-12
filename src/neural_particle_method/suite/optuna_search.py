"""Optuna search over the per-slice estimator recipe on the tuning clouds, tracked in MLflow: one
parent run per study, one nested run per trial, the Optuna SQLite storage as an artifact."""
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import optuna
import torch

from ..calibrate.explicit import extend_tail
from ..estimators.recipes import RECIPE_TYPES, coerce_recipe, regressor_from_recipe
from ..tracking.store import Store, git_hash
from .config import FULL, TUNING_SIDS
from .optuna_clouds import SLICE_TIMES, ensure_cloud, heldout_loss, load_cloud, quantile_grid

EXPERIMENT = "optuna_offline"
STORAGE_DIR = Path("results/optuna")
TUNING_SEED = 5000
TRIAL_TIMEOUT_S = 900          # wall-clock cap on one trial; past it the trial is a TIMEOUT
FULL_BATCH_SUBSAMPLE = 100_000  # the fit sample a full-batch recipe is pinned to
FAST_RECIPE_ENV = "NPM_OPTUNA_FAST_RECIPE"
_FAST_RECIPE = {"hidden": 16, "depth": 2, "batch_size": 0, "first_steps": 5, "later_steps": 2,
                "weight_decay": 0.0, "fit_subsample": FULL_BATCH_SUBSAMPLE, "warm_start": True,
                "mean_match": False, "monotone": False, "monotone_penalty": 0.0, "tail": "flat",
                "hetero": False}


class TrialTimeout(optuna.TrialPruned):
    """The trial's wall-clock cap passed.

    A `TrialPruned` subclass, so Optuna records the trial as PRUNED and `optimize` does not
    re-raise it; the MLflow child is tagged TIMEOUT rather than PRUNED and keeps the partial
    score, which says how far the trial got before the cap.
    """

    def __init__(self, score_partial, fit_s):
        super().__init__(f"trial exceeded its wall-clock cap after {fit_s:.1f}s")
        self.score_partial, self.fit_s = score_partial, fit_s


def suggest_recipe(trial):
    """One trial's recipe.

    A full-batch fit (`batch_size == 0`) costs one forward and backward pass over the whole fit
    sample per step, so its price is linear in `fit_subsample` and the two knobs together span
    two orders of magnitude of trial cost. Full batch is therefore pinned to the smallest
    sample: `fit_subsample` is `FULL_BATCH_SUBSAMPLE` and is not suggested at all, so the
    sampler never spends trials on a dimension that cannot vary.

    `NPM_OPTUNA_FAST_RECIPE=1` returns a tiny fixed recipe with only `lr` sampled. It is there
    for the multi-worker test: workers are separate processes that re-import this module, so a
    monkeypatch in the parent never reaches them. Test-only; never set it for a real study.
    """
    if os.environ.get(FAST_RECIPE_ENV) == "1":
        return {**_FAST_RECIPE, "lr": trial.suggest_float("lr", 1e-3, 1e-2, log=True)}
    monotone = trial.suggest_categorical("monotone", [False, True])
    batch_size = trial.suggest_categorical("batch_size", [2048, 8192, 0])
    return {
        "hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
        "depth": trial.suggest_categorical("depth", [2, 3, 4]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": batch_size,
        "first_steps": trial.suggest_int("first_steps", 500, 4000, step=250),
        "later_steps": trial.suggest_int("later_steps", 100, 1500, step=50),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "fit_subsample": (FULL_BATCH_SUBSAMPLE if batch_size == 0 else
                          trial.suggest_categorical("fit_subsample",
                                                    [100_000, 250_000, 400_000])),
        "warm_start": trial.suggest_categorical("warm_start", [True, False]),
        "mean_match": trial.suggest_categorical("mean_match", [False, True]),
        "monotone": monotone,
        "monotone_penalty": (trial.suggest_float("monotone_penalty", 1e-5, 1e-1, log=True)
                             if monotone else 0.0),
        "tail": trial.suggest_categorical("tail", ["free", "flat", "linear"]),
        "hetero": trial.suggest_categorical("hetero", [False, True]),
    }


def fit_chain(recipe, cloud, seed):
    """Fit the cloud's slices in order with one regressor; log held-out loss ratios to NW."""
    r = coerce_recipe(recipe)
    est = regressor_from_recipe(r, seed=seed, monotone_sign=-float(np.sign(cloud.rho) or 1.0))
    rng = np.random.default_rng(seed)
    out = []
    for i, ((lf, vf), (lh, vh)) in enumerate(zip(cloud.fit, cloud.held)):
        idx = rng.choice(len(lf), size=min(r["fit_subsample"], len(lf)), replace=False)
        grid = quantile_grid(lf[idx])
        f = est.fit_predict(cloud.times[i], lf[idx], vf[idx], grid)
        grid, f = extend_tail(grid, f, r["tail"], est)
        f = np.clip(f, 1e-4, None)
        out.append(float(np.log(heldout_loss(grid, f, lh, vh) / cloud.nw_loss[i])))
    return out


def score_recipe(recipe, clouds, seed=0, report=None, seeds=None, deadline=None):
    """Mean log loss ratio to NW over the slices of every cloud.

    `seeds` is the fit seed of each cloud, defaulting to `seed + index in clouds`. The objective
    passes each cloud's index in the study's full cloud list instead, so a scenario is fitted
    with the same seed in every trial and trials differ only by their recipe.

    `deadline` is a `time.perf_counter()` value. It is checked after each scenario, so a trial
    overruns its cap by at most one scenario, and raises `TrialTimeout` carrying the score so
    far. Checking mid-scenario would need a callback into the fit loop for no useful precision.
    """
    detail, logs, t0 = {}, [], time.perf_counter()
    if seeds is None:
        seeds = [seed + i for i in range(len(clouds))]
    for i, cloud in enumerate(clouds):
        ratios = fit_chain(recipe, cloud, seeds[i])
        detail[cloud.sid] = dict(zip(cloud.times, ratios))
        logs += ratios
        if report is not None:
            report(i, float(np.mean(logs)))
        if deadline is not None and time.perf_counter() > deadline:
            raise TrialTimeout(float(np.mean(logs)), time.perf_counter() - t0)
    return float(np.mean(logs)), detail, time.perf_counter() - t0


def find_parent(store, study):
    """The study's parent run id, or None. Read-only: creates nothing, changes no status."""
    exp = store.client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        return None
    runs = store.client.search_runs(
        [exp.experiment_id], f"params.study = '{study}' and tags.`optuna.kind` = 'study'",
        max_results=1)
    return runs[0].info.run_id if runs else None


def study_parent(store, study):
    """Find-or-create the parent run and mark it RUNNING. Writes; for `run_study` only."""
    rid = find_parent(store, study)
    if rid is not None:
        store.client.update_run(rid, status="RUNNING")
        return rid
    r = store.client.create_run(store.experiment_id(EXPERIMENT), tags={"optuna.kind": "study"})
    for k, v in {"study": study, "tuning_seed": TUNING_SEED, "git_hash": git_hash()}.items():
        store.client.log_param(r.info.run_id, k, v)
    return r.info.run_id


def _storage(storage_dir, study):
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{Path(storage_dir) / f'{study}.db'}"


def _sampler_and_pruner(seed=0):
    """Optuna persists neither in storage, so every open of the study must rebuild both. `seed`
    is the worker index: concurrent workers must not draw the same startup trials."""
    return (optuna.samplers.TPESampler(seed=seed),
            optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))


def _load_study(study, storage, seed=0):
    sampler, pruner = _sampler_and_pruner(seed)
    return optuna.load_study(study_name=study, storage=storage, sampler=sampler, pruner=pruner)


def _child_state_counts(store, parent):
    """How many of `parent`'s trial runs ended in each `optuna.state`."""
    df = store.search(EXPERIMENT)
    needed = ("tags.mlflow.parentRunId", "tags.optuna.state")
    if any(c not in df.columns for c in needed):
        return {}
    kids = df[df["tags.mlflow.parentRunId"] == parent]
    return {k: int(v) for k, v in kids["tags.optuna.state"].value_counts().items()}


def _trial_worker(args):
    """One process: runs `n` trials of the shared study, logging each as a nested MLflow run."""
    if args["pin_threads"]:
        # one BLAS thread per worker; the pool is the parallelism. Only in a spawned worker, as
        # `grid._worker` does: `n_jobs == 1` runs this in the caller's own process, and changing
        # the global thread count there changes every later torch reduction in that process.
        torch.set_num_threads(1)
    store = Store(args["uri"], args["root"])
    cloud_runs = args["cloud_runs"]
    clouds = {rid: load_cloud(store, rid, cache_dir=_cache_for(args["cache_dir"], rid))
              for rid in cloud_runs}
    st = _load_study(args["study"], args["storage"], seed=args["widx"])
    exp = store.experiment_id(EXPERIMENT)
    seed, parent = args["seed"], args["parent"]

    def objective(trial):
        rng = np.random.default_rng(trial.number)
        n_pick = min(args["per_trial"], len(cloud_runs))
        picked = [cloud_runs[j] for j in rng.choice(len(cloud_runs), size=n_pick, replace=False)]
        recipe = suggest_recipe(trial)
        run = store.client.create_run(exp, tags={"mlflow.parentRunId": parent,
                                                 "optuna.kind": "trial"})
        rid = run.info.run_id
        timeout = args["trial_timeout_s"]
        deadline = None if not timeout else time.perf_counter() + timeout

        def report(i, running):
            trial.report(running, step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            # inside the try: a store that rejects a param must tag the child FAIL, not leave a
            # RUNNING orphan whose state no reader can interpret
            for k, v in {**recipe, "trial_number": trial.number}.items():
                store.client.log_param(rid, k, v)
            score, detail, fit_s = score_recipe(
                recipe, [clouds[r] for r in picked], seed=seed, report=report,
                seeds=[seed + cloud_runs.index(r) for r in picked], deadline=deadline)
        except TrialTimeout as e:
            store.client.log_metric(rid, "fit_s", e.fit_s)
            store.client.log_metric(rid, "score_partial", e.score_partial)
            store.client.set_tag(rid, "optuna.state", "TIMEOUT")
            store.client.set_terminated(rid, status="FINISHED")
            raise
        except optuna.TrialPruned:
            store.client.set_tag(rid, "optuna.state", "PRUNED")
            store.client.set_terminated(rid, status="FINISHED")
            raise
        except Exception:
            store.client.set_tag(rid, "optuna.state", "FAIL")
            store.client.set_terminated(rid, status="FAILED")
            raise
        store.client.log_metric(rid, "score", score)
        store.client.log_metric(rid, "fit_s", fit_s)
        for sid, per in detail.items():
            store.client.log_metric(rid, f"score/{sid}", float(np.mean(list(per.values()))))
            for t, lr_ in per.items():
                store.client.log_metric(rid, f"loss_ratio/{sid}/t{t:g}", lr_)
        store.client.set_tag(rid, "optuna.state", "COMPLETE")
        store.client.set_terminated(rid, status="FINISHED")
        return score

    st.optimize(objective, n_trials=args["n"], timeout=args["study_timeout_s"],
                catch=(Exception,))
    return args["n"]


def _cache_for(cache_dir, run_id):
    """The cache path for one cloud; the parent warms exactly the path the workers read."""
    return None if cache_dir is None else Path(cache_dir) / run_id


def run_study(store, study, n_trials, n_jobs=1, settings=FULL, sids=TUNING_SIDS, per_trial=8,
              times=SLICE_TIMES, registry=None, storage_dir=STORAGE_DIR, cache_dir=None, seed=0,
              trial_timeout_s=TRIAL_TIMEOUT_S, study_timeout_s=None):
    """Run `n_trials` more trials of `study` across `n_jobs` processes.

    `trial_timeout_s` caps one trial's wall clock (None or 0 for no cap); `study_timeout_s` caps
    each worker's `optimize` call, so `--timeout-hours` bounds the whole study. `n_trials=0` is a
    no-op that still warms the cloud cache and refreshes the parent run.
    """
    cloud_runs = [ensure_cloud(store, sid, settings, registry, times) for sid in sids]
    # warm the cache here, before the pool exists: workers otherwise race to download the same
    # `cloud.npz` on their first trial, and the loser reads a file that is still being written
    for rid in cloud_runs:
        load_cloud(store, rid, cache_dir=_cache_for(cache_dir, rid))
    storage = _storage(storage_dir, study)
    sampler, pruner = _sampler_and_pruner(seed=0)
    optuna.create_study(study_name=study, storage=storage, load_if_exists=True,
                        direction="minimize", sampler=sampler, pruner=pruner)
    parent = study_parent(store, study)
    per_proc = [n_trials // n_jobs + (1 if i < n_trials % n_jobs else 0) for i in range(n_jobs)]
    args = [{"uri": store.tracking_uri, "root": store.artifact_root, "study": study,
             "storage": storage, "parent": parent, "cloud_runs": cloud_runs,
             "per_trial": per_trial, "n": n, "cache_dir": cache_dir, "seed": seed, "widx": widx,
             "trial_timeout_s": trial_timeout_s, "study_timeout_s": study_timeout_s,
             "pin_threads": n_jobs > 1}
            for widx, n in enumerate(per_proc) if n > 0]
    if not args:
        pass                                  # n_trials == 0: nothing to run, nothing to spawn
    elif n_jobs == 1:
        _trial_worker(args[0])
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            list(ex.map(_trial_worker, args))
    st = _load_study(study, storage)
    done = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
    store.client.log_metric(parent, "n_trials", len(st.trials))
    counts = _child_state_counts(store, parent)
    n_failed, n_timeout = counts.get("FAIL", 0), counts.get("TIMEOUT", 0)
    store.client.log_metric(parent, "n_failed", n_failed)
    store.client.log_metric(parent, "n_timeout", n_timeout)
    print(f"study {study}: {len(st.trials)} trials, {len(done)} complete, "
          f"{n_failed} failed, {n_timeout} timed out", flush=True)
    if done:
        best = min(done, key=lambda t: t.value)
        store.client.log_metric(parent, "best_score", best.value)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "best.json"
            p.write_text(json.dumps({"trial_number": best.number, "score": best.value,
                                     "params": best.params}, indent=1))
            store.client.log_artifact(parent, str(p))
    store.client.log_artifact(parent, str(Path(storage_dir) / f"{study}.db"))
    store.client.set_terminated(parent, status="FINISHED")
    return parent


def top_recipes(store, study, k):
    """The k best COMPLETE trials of `study`, best first. Read-only; [] if there are none."""
    parent = find_parent(store, study)
    if parent is None:
        return []
    df = store.search(EXPERIMENT)
    needed = ("tags.mlflow.parentRunId", "tags.optuna.state", "metrics.score",
              "params.trial_number")
    if any(c not in df.columns for c in needed):
        return []
    kids = df[(df["tags.mlflow.parentRunId"] == parent) & (df["tags.optuna.state"] == "COMPLETE")]
    kids = kids.sort_values("metrics.score").head(k)
    out = []
    for _, r in kids.iterrows():
        recipe = coerce_recipe({key: r[f"params.{key}"] for key in RECIPE_TYPES
                                if f"params.{key}" in r and r[f"params.{key}"] is not None})
        out.append({"trial_number": int(r["params.trial_number"]),
                    "score": float(r["metrics.score"]), "recipe": recipe,
                    "run_id": r["run_id"]})
    return out
