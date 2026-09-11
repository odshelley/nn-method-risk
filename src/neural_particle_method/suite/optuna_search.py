"""Optuna search over the per-slice estimator recipe on the tuning clouds, tracked in MLflow: one
parent run per study, one nested run per trial, the Optuna SQLite storage as an artifact."""
import json
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import optuna

from ..calibrate.explicit import extend_tail
from ..estimators.recipes import RECIPE_TYPES, coerce_recipe, regressor_from_recipe
from ..tracking.store import Store, git_hash
from .config import FULL, TUNING_SIDS
from .optuna_clouds import SLICE_TIMES, ensure_cloud, heldout_loss, load_cloud, quantile_grid

EXPERIMENT = "optuna_offline"
STORAGE_DIR = Path("results/optuna")
TUNING_SEED = 5000


def suggest_recipe(trial):
    monotone = trial.suggest_categorical("monotone", [False, True])
    return {
        "hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
        "depth": trial.suggest_categorical("depth", [2, 3, 4]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [2048, 8192, 0]),
        "first_steps": trial.suggest_int("first_steps", 500, 4000, step=250),
        "later_steps": trial.suggest_int("later_steps", 100, 1500, step=50),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "fit_subsample": trial.suggest_categorical("fit_subsample", [100_000, 250_000, 400_000]),
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


def score_recipe(recipe, clouds, seed=0, report=None):
    detail, logs, t0 = {}, [], time.perf_counter()
    for i, cloud in enumerate(clouds):
        ratios = fit_chain(recipe, cloud, seed + i)
        detail[cloud.sid] = dict(zip(cloud.times, ratios))
        logs += ratios
        if report is not None:
            report(i, float(np.mean(logs)))
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


def _trial_worker(args):
    """One process: runs `n` trials of the shared study, logging each as a nested MLflow run."""
    uri, root, study, storage, parent, cloud_runs, per_trial, n, cache_dir, seed, widx = args
    store = Store(uri, root)
    clouds = {rid: load_cloud(store, rid, cache_dir=None if cache_dir is None
                              else Path(cache_dir) / rid) for rid in cloud_runs}
    st = _load_study(study, storage, seed=widx)
    exp = store.experiment_id(EXPERIMENT)

    def objective(trial):
        rng = np.random.default_rng(trial.number)
        n_pick = min(per_trial, len(cloud_runs))
        picked = [cloud_runs[j] for j in rng.choice(len(cloud_runs), size=n_pick, replace=False)]
        recipe = suggest_recipe(trial)
        run = store.client.create_run(exp, tags={"mlflow.parentRunId": parent,
                                                 "optuna.kind": "trial"})
        rid = run.info.run_id
        for k, v in {**recipe, "trial_number": trial.number}.items():
            store.client.log_param(rid, k, v)

        def report(i, running):
            trial.report(running, step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            score, detail, fit_s = score_recipe(recipe, [clouds[r] for r in picked], seed=seed,
                                                report=report)
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

    st.optimize(objective, n_trials=n, catch=(Exception,))
    return n


def run_study(store, study, n_trials, n_jobs=1, settings=FULL, sids=TUNING_SIDS, per_trial=8,
              times=SLICE_TIMES, registry=None, storage_dir=STORAGE_DIR, cache_dir=None, seed=0):
    cloud_runs = [ensure_cloud(store, sid, settings, registry, times) for sid in sids]
    storage = _storage(storage_dir, study)
    sampler, pruner = _sampler_and_pruner(seed=0)
    optuna.create_study(study_name=study, storage=storage, load_if_exists=True,
                        direction="minimize", sampler=sampler, pruner=pruner)
    parent = study_parent(store, study)
    per_proc = [n_trials // n_jobs + (1 if i < n_trials % n_jobs else 0) for i in range(n_jobs)]
    args = [(store.tracking_uri, store.artifact_root, study, storage, parent, cloud_runs,
             per_trial, n, cache_dir, seed, widx)
            for widx, n in enumerate(per_proc) if n > 0]
    if n_jobs == 1:
        _trial_worker(args[0])
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            list(ex.map(_trial_worker, args))
    st = _load_study(study, storage)
    done = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
    store.client.log_metric(parent, "n_trials", len(st.trials))
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
