"""Lagged suite, online stage: one run per (scenario, method, offline size, lag, seed)."""
import time
from dataclasses import replace

from ..bench.reference_runs import run_reference
from ..bench.runner import _leverage_error
from ..bench.scenarios import full_registry
from ..calibrate.explicit import calibrate_explicit
from ..estimators import make_estimator
from ..tracking.store import git_hash, to_jsonable
from .artifacts import load_run
from .config import FULL
from .heads import FeatureRidgeHead, RKHSHead, SplineHead, online_sweep
from .lag import lagged_scenario
from .offline import run_offline
from .reference import score_field

ONLINE_METHODS = {
    "explicit_stale": ("explicit", None), "implicit_stale": ("implicit", None),
    "explicit_rkhs": ("explicit", "rkhs"), "implicit_rkhs": ("implicit", "rkhs"),
    "explicit_ridge": ("explicit", "ridge"), "implicit_ridge": ("implicit", "ridge"),
    "explicit_spline": ("explicit", "spline"), "implicit_spline": ("implicit", "spline"),
    "stale_L": ("explicit", "stale_L"), "nw_resolve": (None, "nw_resolve"),
}


def _head(kind):
    return {"rkhs": RKHSHead(), "ridge": FeatureRidgeHead(), "spline": SplineHead()}[kind]


def ensure_lagged_reference(store, sid, lag, settings):
    """PDE reference for the lagged scenario (keyed by lag kind); None if the solve fails."""
    lsc = lagged_scenario(full_registry()[sid], lag)
    try:
        return run_reference(store, sid, n_steps=settings.explicit.n_steps, n_x=settings.n_x,
                             n_v=settings.n_v, scenario=lsc, lag=lag.kind)
    except Exception as e:  # noqa: BLE001 - a failed reference must not block the online run
        print(f"warning: lagged PDE reference failed for {sid}/{lag.kind}: {e}")
        return None


def run_online(store, sid, method, offline_n, lag, seed, settings=FULL):
    if method not in ONLINE_METHODS:
        raise KeyError(f"unknown online method {method!r}; choose from {list(ONLINE_METHODS)}")
    body, head_kind = ONLINE_METHODS[method]
    n_steps = settings.explicit.n_steps
    key = {"sid": sid, "method": method, "offline_n": int(offline_n if body else 0),
           "lag": lag.kind, "seed": int(seed), "n_steps": int(n_steps)}
    exp = settings.experiment("suite_lagged")
    existing = store.find_finished(exp, key)
    if existing is not None:
        return existing
    sc = full_registry()[sid]
    lsc = lagged_scenario(sc, lag)
    lv = lsc.local_vol()
    ecfg = replace(settings.explicit, n_particles=settings.n_online)
    offline = load_run(store, run_offline(store, sid, body, offline_n, settings)) if body else None
    ensure_lagged_reference(store, sid, lag, settings)
    params = {**key, "git_hash": git_hash(), **lsc.as_params(),
              "offline_run": offline.run_id if offline else "",
              **{f"explicit.{k}": v for k, v in ecfg.as_params().items()},
              **{f"reprice.{k}": v for k, v in settings.reprice.as_params().items()}}
    head = _head(head_kind) if head_kind in ("rkhs", "ridge", "spline") else None
    if head is not None:
        params.update({f"head.{k}": v for k, v in head.params.items()})
    with store.run(exp, params) as h:
        t0 = time.perf_counter()
        if head_kind == "nw_resolve":
            # seed + 1 is the head sweeps' cloud stream: no method shares a stream with
            # another method at a different seed.
            est = make_estimator("nw", seed=seed + 1, local_vol=lv, s0=lsc.s0)
            r = calibrate_explicit(lv, lsc.dynamics, est, ecfg, s0=lsc.s0, T=lsc.T,
                                   seed=seed + 1)
            field, online_s = r.field, time.perf_counter() - t0
        elif head_kind == "stale_L":
            field, online_s = offline.field, 0.0
        else:
            field, online_s = online_sweep(offline.model, lv, lsc.dynamics, lsc.s0, lsc.T, ecfg,
                                           head=head, seed=seed + 1)
            if head is not None:
                online_s = time.perf_counter() - t0
        metrics, err = score_field(field, lsc, seed, settings.reprice)
        metrics.update({"online_s": online_s, "fit_s": online_s, "total_s": online_s})
        metrics.update(_leverage_error(store, sid, field, n_steps, lag=lag.kind))
        h.log_metrics(metrics)
        h.log_json("leverage.json", field.to_json())
        h.log_json("iv_err_bp.json", err)
        h.log_json("diagnostics.json", to_jsonable({"offline_run": params["offline_run"],
                                                    "lag": lag.kind, "spot_move": lag.spot_move,
                                                    "head": head.params if head else None}))
        return h.run_id
