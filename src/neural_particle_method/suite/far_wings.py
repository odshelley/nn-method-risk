"""Far-wing re-scoring of stored leverage fields with the OTM tilted repricer (no recalibration).

Selected runs (all at the 80k budget unless told otherwise): the tilt study's cold rows in
`tilt_cold` and `tilt_cold_s400`, the PDE floors at 200 and 400 steps, the tilted online cells in
`tilt_online` and their untilted partners and the NW re-solve in `suite_budget_tuned`. Each run
gets `far_*` metrics and a `far_iv_err_bp.json` artifact; a run already carrying
`far_wings_mae_bp` is skipped unless forced.
"""
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from mlflow.entities import Metric

from ..bench.scenarios import full_registry
from ..calibrate.importance import TiltDesign
from ..estimators.recipes import load_recipe, recipe_exists, recipe_hash
from ..pricing.metrics import far_wing_metrics
from ..pricing.reprice import RepriceConfig, far_k_grid, reprice_iv_otm, snap_times
from ..simulate.leverage import LeverageField
from ..suite.lag import LAGS, lagged_scenario
from ..tracking.store import Store
from .config import FULL, SSVI_SIDS
from .grid import _drain

REPRICE_TILT = TiltDesign("constant", 3.0)
SEED_OFFSET = 20_000
LAG_BY_KIND = {lag.kind: lag for lag in LAGS}


def far_score(field, sc, seed, reprice_cfg):
    """Far-wing IV metrics of `field` on scenario `sc` with the tilted OTM repricer."""
    k = far_k_grid()
    mats = list(sc.maturities)
    times = snap_times(mats, reprice_cfg.n_steps, T=sc.T)
    mix = REPRICE_TILT.mixture(reprice_cfg.n_steps, sc.T, sc.dynamics.rho)
    ivs, _, ess = reprice_iv_otm(field, sc.dynamics, sc.s0, mats, k, reprice_cfg,
                                 seed=seed + SEED_OFFSET, mixture=mix)
    tgt = sc.target_ivs(k, times)
    m, err = far_wing_metrics(ivs, tgt, k, times, sc.s0)
    m["far_reprice_ess"] = ess
    return m, err


def _scenario(params):
    sc = full_registry()[params["sid"]]
    lag_kind = params.get("lag")
    if isinstance(lag_kind, str) and lag_kind in LAG_BY_KIND:
        sc = lagged_scenario(sc, LAG_BY_KIND[lag_kind])
    return sc


def rescore_run(store, run_id, force=False):
    run = store.client.get_run(run_id)
    if not force and "far_wings_mae_bp" in run.data.metrics:
        return None
    p = dict(run.data.params)
    sc = _scenario(p)
    cfg = RepriceConfig(int(p.get("reprice.n_particles", 500_000)),
                        int(p.get("reprice.n_steps", 200)))
    with tempfile.TemporaryDirectory() as d:
        field = LeverageField.from_json(json.loads(store.download(run_id, "leverage.json", d)
                                                   .read_text()))
        m, err = far_score(field, sc, int(p.get("seed", 0)), cfg)
        ts = int(time.time() * 1000)
        store.client.log_batch(run_id, metrics=[Metric(k, float(v), ts, 0) for k, v in m.items()])
        pth = Path(d) / "far_iv_err_bp.json"
        pth.write_text(json.dumps(np.where(np.isfinite(err), err, None).tolist()))
        store.client.log_artifact(run_id, str(pth))
    return m


def _finished(store, exp):
    if store.client.get_experiment_by_name(exp) is None:
        return None
    df = store.search(exp)
    return df[df.status == "FINISHED"] if len(df) else df


def select_runs(store, settings=FULL, budget=80_000):
    """(experiment, run_id) pairs to score, deduplicated, in a stable order."""
    out = []
    h = recipe_hash(load_recipe("explicit_opt")) if recipe_exists("explicit_opt") else None
    sids = set(SSVI_SIDS)

    def take(exp, df):
        for rid in df.run_id:
            out.append((exp, rid))

    for exp in (settings.experiment("tilt_cold"), settings.experiment("tilt_cold_s400")):
        df = _finished(store, exp)
        if df is not None and len(df):
            take(exp, df[(df["params.n_particles"] == str(budget)) & df["params.sid"].isin(sids)])
    for exp in (settings.experiment("suite_pde_floor"),
                settings.experiment("suite_pde_floor_s400")):
        df = _finished(store, exp)
        if df is not None and len(df):
            take(exp, df[df["params.sid"].isin(sids)])
    df = _finished(store, settings.experiment("tilt_online"))
    if df is not None and len(df):
        take(settings.experiment("tilt_online"), df[df["params.budget"] == str(budget)])
    df = _finished(store, settings.experiment("suite_budget_tuned"))
    if df is not None and len(df):
        sel = df[(df["params.budget"] == str(budget)) & df["params.sid"].isin(sids)]
        if "params.recipe_hash" in sel.columns:
            head = sel[(sel["params.method"] == "explicit_opt_spline")
                       & (sel["params.recipe_hash"] == h)]
        else:
            head = sel.iloc[:0]
        resolve = sel[sel["params.method"] == "nw_resolve"]
        take(settings.experiment("suite_budget_tuned"), head)
        take(settings.experiment("suite_budget_tuned"), resolve)
    seen, uniq = set(), []
    for item in out:
        if item[1] not in seen:
            seen.add(item[1]); uniq.append(item)
    return uniq


def _worker(args):
    uri, root, exp, rid, force, n_jobs = args
    if n_jobs > 1:
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or n_jobs) // n_jobs)))
    try:
        m = rescore_run(Store(uri, root), rid, force=force)
        return (exp, rid, "skip" if m is None else "scored"), None
    except Exception as e:  # noqa: BLE001
        return (exp, rid), repr(e)


def run_farwings(store, settings=FULL, budget=80_000, n_jobs=1, force=False):
    todo = select_runs(store, settings, budget)
    if not force:
        todo = [(e, r) for e, r in todo
                if "far_wings_mae_bp" not in store.client.get_run(r).data.metrics]
    print(f"farwings: {len(todo)} runs to score", flush=True)
    args = [(store.tracking_uri, store.artifact_root, e, r, force, n_jobs) for e, r in todo]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            return _drain(pool.map(_worker, args))
    return _drain(map(_worker, args))
