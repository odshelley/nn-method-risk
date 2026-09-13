"""Small-online-budget cells: a frozen offline body with online heads against NW re-solved at the
same budget, for a handful of online budgets, both lags, two seeds. Every fit uses the full online
cloud.

The default body is the tuned per-slice network (`explicit_tuned`), taken from the suite's offline
stage and trained here if that stage has not produced it yet; its cells go to `suite_budget_tuned`.
`explicit` is the legacy path: the short-trained body from `suite_offline_conv`, cells in
`suite_budget`. `explicit_opt` is the recipe-driven body from the Optuna search; it shares the
tuned experiment and is separated from it by the `recipe_hash` in the run key.

`nw_resolve` is the exception: it re-solves from scratch and reads no body, so every body shares
one cell for it, logged under `explicit_tuned` with no recipe.
"""
import time
from dataclasses import replace

from ..bench.scenarios import full_registry
from ..calibrate.explicit import calibrate_explicit
from ..estimators import make_estimator
from ..estimators.recipes import (
    coerce_recipe,
    explicit_config_from_recipe,
    load_recipe,
    recipe_hash,
)
from ..tracking.store import git_hash
from .artifacts import load_run
from .config import FULL, SSVI_SIDS
from .heads import FeatureRidgeHead, RKHSHead, SplineHead, online_sweep
from .lag import LAGS, lagged_scenario
from .offline import run_offline
from .reference import score_field

EXPERIMENTS = {"explicit_tuned": "suite_budget_tuned", "explicit": "suite_budget",
               "explicit_opt": "suite_budget_tuned"}
LEGACY_BODY_EXPERIMENT = "suite_offline_conv"
DEFAULT_BODY = "explicit_tuned"
SIDS = SSVI_SIDS
BUDGETS = (10_000, 30_000, 80_000)
SEEDS = (0, 1)
N_BODY = 500_000
N_STEPS = 200


def budget_methods(body):
    return ("nw_resolve", f"{body}_stale", f"{body}_rkhs", f"{body}_spline", f"{body}_ridge")


def head_for(method):
    """Heads key off the method suffix; `nw_resolve` and `*_stale` have none."""
    return {"rkhs": RKHSHead(), "spline": SplineHead(),
            "ridge": FeatureRidgeHead()}.get(method.rsplit("_", 1)[-1])


def body_run(store, sid, body, settings=FULL, recipe=None):
    if body == "explicit_opt":
        return run_offline(store, sid, "explicit_opt", settings.offline_sizes[-1], settings,
                           recipe=recipe)
    if body == "explicit_tuned":
        # the offline stage runs first, so this normally finds the finished body and returns it
        return run_offline(store, sid, "explicit_tuned", settings.offline_sizes[-1], settings)
    key = {"sid": sid, "body": "explicit", "n_particles": N_BODY, "seed": 0, "n_steps": N_STEPS}
    rid = store.find_finished(LEGACY_BODY_EXPERIMENT, key)
    if rid is None:
        raise RuntimeError(f"no long-trained explicit body for {sid} in {LEGACY_BODY_EXPERIMENT}")
    return rid


def run_budget_cell(store, sid, method, budget, lag, seed, body=DEFAULT_BODY, settings=FULL,
                    recipe=None):
    # `nw_resolve` re-solves from scratch and never touches the body, so it is one cell per
    # (sid, budget, lag, seed) shared by every body: it stays in the tuned experiment with no
    # `recipe_hash`, which is the row the tuned sweep already logged. Without this an
    # `explicit_opt` re-solve would land beside the tuned one under the same method, and the
    # containment filter in `find_finished` would let a tuned query match it.
    if method == "nw_resolve":
        body, recipe = DEFAULT_BODY, None
    exp = settings.experiment(EXPERIMENTS[body])
    key = {"sid": sid, "method": method, "budget": int(budget), "lag": lag.kind, "seed": int(seed),
           "n_steps": int(settings.explicit.n_steps)}
    if body == "explicit_opt":
        recipe = coerce_recipe(recipe if recipe is not None else load_recipe("explicit_opt"))
        key["recipe_hash"] = recipe_hash(recipe)
    existing = store.find_finished(exp, key)
    if existing is not None:
        return existing
    sc = full_registry()[sid]
    lsc = lagged_scenario(sc, lag)
    lv = lsc.local_vol()
    ecfg = replace(settings.explicit, n_particles=int(budget), fit_subsample=int(budget))
    if body == "explicit_opt":
        # the recipe's `tail` rule belongs to the body and has to reach the online sweep; the
        # step counts it also sets are unused online, and `fit_subsample` stays the full cloud
        # because the recipe's is never below an online budget.
        ecfg = explicit_config_from_recipe(ecfg, recipe)
    # a body-free cell logs the same param names as the others, with an empty `body_run`
    body_rid, loaded = "", None
    if method != "nw_resolve":
        body_rid = body_run(store, sid, body, settings, recipe)
        loaded = load_run(store, body_rid)
    head = head_for(method)
    params = {**key, "git_hash": git_hash(), "body_run": body_rid, "body": body, **lsc.as_params(),
              **{f"explicit.{k}": v for k, v in ecfg.as_params().items()}}
    if head is not None:
        params.update({f"head.{k}": v for k, v in head.params.items()})
    if body == "explicit_opt":
        params.update({f"recipe.{k}": v for k, v in recipe.items()})
    with store.run(exp, params) as h:
        t0 = time.perf_counter()
        if method == "nw_resolve":
            est = make_estimator("nw", seed=seed + 1, local_vol=lv, s0=lsc.s0)
            field = calibrate_explicit(lv, lsc.dynamics, est, ecfg, s0=lsc.s0, T=lsc.T,
                                       seed=seed + 1).field
        else:
            field, _, _ = online_sweep(loaded.model, lv, lsc.dynamics, lsc.s0, lsc.T, ecfg,
                                       head=head, seed=seed + 1)
        online_s = 0.0 if method.endswith("_stale") else time.perf_counter() - t0
        metrics, err = score_field(field, lsc, seed, settings.reprice)
        metrics.update({"online_s": online_s, "fit_s": online_s, "total_s": online_s})
        h.log_metrics(metrics)
        h.log_json("leverage.json", field.to_json())
        h.log_json("iv_err_bp.json", err)
        return h.run_id


def budget_jobs(sids=SIDS, body=DEFAULT_BODY, budgets=BUDGETS, seeds=SEEDS):
    return [(sid, m, b, lag.kind, seed) for sid in sids for lag in LAGS for seed in seeds
            for b in budgets for m in budget_methods(body)]
