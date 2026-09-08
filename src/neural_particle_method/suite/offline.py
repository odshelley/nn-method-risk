"""Offline bodies: a full calibration on the overnight surface, saved once and reused by every
online cell."""
import time
from dataclasses import replace

from ..bench.scenarios import full_registry
from ..calibrate.explicit import calibrate_explicit
from ..calibrate.implicit import calibrate_implicit
from ..estimators.nn import NNRegressor
from ..tracking.store import git_hash
from .artifacts import save_model
from .config import FULL
from .reference import score_field

BODIES = ("explicit", "implicit")


def run_offline(store, sid, body, n_particles, settings=FULL, seed=0):
    if body not in BODIES:
        raise ValueError(f"body must be one of {BODIES}, got {body!r}")
    n_steps = settings.explicit.n_steps
    key = {"sid": sid, "body": body, "n_particles": int(n_particles), "seed": int(seed),
           "n_steps": int(n_steps)}
    exp = settings.experiment("suite_offline")
    existing = store.find_finished(exp, key)
    if existing is not None:
        return existing
    sc = full_registry()[sid]
    lv = sc.local_vol()
    ecfg = replace(settings.explicit, n_particles=int(n_particles))
    icfg = replace(settings.implicit, n_particles=int(n_particles))
    params = {**key, "git_hash": git_hash(), **sc.as_params(),
              **{f"explicit.{k}": v for k, v in ecfg.as_params().items()},
              **{f"implicit.{k}": v for k, v in icfg.as_params().items()},
              **{f"reprice.{k}": v for k, v in settings.reprice.as_params().items()}}
    with store.run(exp, params) as h:
        t0 = time.perf_counter()
        est = NNRegressor(seed=seed, first_steps=ecfg.first_steps, later_steps=ecfg.later_steps,
                          keep_slice_weights=(body == "explicit"))
        w = calibrate_explicit(lv, sc.dynamics, est, ecfg, s0=sc.s0, T=sc.T, seed=seed)
        if body == "explicit":
            field, model, fit_s = w.field, est, w.fit_s
        else:
            r = calibrate_implicit(lv, sc.dynamics, icfg, s0=sc.s0, T=sc.T, seed=seed, L0=w.field)
            field, model, fit_s = r.field, r.net, w.fit_s + r.fit_s
        total = time.perf_counter() - t0
        metrics, err = score_field(field, sc, seed, settings.reprice)
        metrics.update({"fit_s": fit_s, "total_s": total})
        h.log_metrics(metrics)
        h.log_json("leverage.json", field.to_json())
        h.log_json("iv_err_bp.json", err)
        save_model(h, model, {**key, "T": sc.T})
        return h.run_id
