"""Validate the top recipes of a study as frozen bodies plus spline head at the practical online
budget, against NW re-solved at the same budget; promote the winner into the estimator recipes."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..estimators.recipes import coerce_recipe, recipe_hash, save_recipe
from ..tracking.store import Store
from .budget import EXPERIMENTS, run_budget_cell
from .config import FULL, SSVI_SIDS
from .lag import LAGS
from .offline import run_offline
from .optuna_search import STORAGE_DIR, top_recipes

LAG_BY_KIND = {lag.kind: lag for lag in LAGS}


def _cell(args):
    """One validation job in one process: the frozen body, or a budget cell on top of it."""
    uri, root, sid, method, budget, lag_kind, seed, settings, recipe = args
    store = Store(uri, root)
    if method == "body":
        return run_offline(store, sid, "explicit_opt", settings.offline_sizes[-1], settings,
                           recipe=recipe)
    # `nw_resolve` is body-free: `run_budget_cell` rewrites the body and drops the recipe itself.
    return run_budget_cell(store, sid, method, budget, LAG_BY_KIND[lag_kind], seed,
                           body="explicit_opt", settings=settings, recipe=recipe)


def _run_all(jobs, n_jobs):
    if n_jobs == 1:
        return [_cell(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(_cell, jobs))


def _mean_over_scenarios(sel, key):
    """Mean over seeds within a scenario, then over scenarios; NaN when nothing is selected."""
    if not len(sel):
        return np.nan
    per = sel.groupby("params.sid")[key].apply(lambda v: pd.to_numeric(v).mean())
    return float(per.mean()) if len(per) else np.nan


def comparison_table(store, recipes, sids, settings=FULL, budget=80_000):
    """Rows: one per recipe (index = trial_number) plus 'nw_resolve'; the two-stage mean of
    pooled and liquid MAE per lag and the median online seconds, from the budget experiment."""
    df = store.search(settings.experiment(EXPERIMENTS["explicit_opt"]))
    needed = ("params.sid", "params.budget", "params.method", "params.lag")
    if any(c not in df.columns for c in needed):
        df = df.iloc[:0]
    else:
        df = df[(df["status"] == "FINISHED") & df["params.sid"].isin(sids)
                & (pd.to_numeric(df["params.budget"]) == budget)]
    has_hash = "params.recipe_hash" in df.columns
    targets = [("nw_resolve", df[df["params.method"] == "nw_resolve"] if len(df) else df)]
    for r in recipes:
        sel = df[(df["params.method"] == "explicit_opt_spline")
                 & (df["params.recipe_hash"] == r["hash"])] if has_hash else df.iloc[:0]
        targets.append((r["trial_number"], sel))
    rows = {}
    for name, sel in targets:
        row = {}
        for lag in LAG_BY_KIND:
            x = sel[sel["params.lag"] == lag] if len(sel) else sel
            row[f"mae_{lag}"] = _mean_over_scenarios(x, "metrics.pooled_mae_bp")
            row[f"liquid_{lag}"] = _mean_over_scenarios(x, "metrics.liquid_mae_bp")
        row["online_s"] = (float(pd.to_numeric(sel["metrics.online_s"]).median()) if len(sel)
                           else np.nan)
        rows[name] = row
    return pd.DataFrame(rows).T


def validate(store, study, top=3, jobs=1, sids=SSVI_SIDS, settings=FULL, budget=80_000,
             seeds=(0, 1)):
    """Train each top recipe as a frozen body, run its spline head at `budget` against NW
    re-solved there, print the comparison and return it. Writes `<study>_top.json`."""
    best = top_recipes(store, study, top)
    out = Path(STORAGE_DIR) / f"{study}_top.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([{k: b[k] for k in ("trial_number", "score", "recipe")}
                               for b in best], indent=1))
    uri, root = store.tracking_uri, store.artifact_root
    for b in best:
        b["hash"] = recipe_hash(b["recipe"])
        _run_all([(uri, root, sid, "body", 0, "surface", 0, settings, b["recipe"])
                  for sid in sids], jobs)
        _run_all([(uri, root, sid, "explicit_opt_spline", budget, lag.kind, seed, settings,
                   b["recipe"]) for sid in sids for lag in LAGS for seed in seeds], jobs)
    _run_all([(uri, root, sid, "nw_resolve", budget, lag.kind, seed, settings, None)
              for sid in sids for lag in LAGS for seed in seeds], jobs)
    table = comparison_table(store, best, sids, settings, budget)
    print(table.round(1).to_string())
    return table


def promote(store, study, trial_number):
    """Write the trial's recipe to `estimators/recipes/explicit_opt.json`; returns the path."""
    for b in top_recipes(store, study, 10_000):
        if b["trial_number"] == trial_number:
            return save_recipe("explicit_opt", coerce_recipe(b["recipe"]))
    raise KeyError(f"trial {trial_number} is not a COMPLETE trial of study {study!r}")
