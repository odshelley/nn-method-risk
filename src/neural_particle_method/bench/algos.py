"""Algorithm registry: one calibrate interface over the seven contenders."""
import time
from dataclasses import dataclass, replace

from ..calibrate.config import ExplicitConfig, ImplicitConfig
from ..calibrate.explicit import calibrate_explicit
from ..calibrate.implicit import calibrate_implicit
from ..calibrate.importance import design_mixture
from ..estimators import make_estimator
from ..estimators.ridge import GlobalRidge
from ..simulate.leverage import LeverageField


@dataclass
class CalibResult:
    field: LeverageField
    timings: dict
    diagnostics: dict
    # trained object when there is one (estimator instance or GlobalNet); ignored by bench
    model: object = None


def _explicit(sc, n, seed, ecfg, method, mixture=None, estimator=None, knobs=None):
    lv = sc.local_vol()
    cfg = replace(ecfg, n_particles=n)
    est = estimator if estimator is not None else make_estimator(
        method, seed=seed, first_steps=cfg.first_steps, later_steps=cfg.later_steps,
        local_vol=lv, s0=sc.s0, **(knobs or {}))
    t0 = time.perf_counter()
    r = calibrate_explicit(lv, sc.dynamics, est, cfg, s0=sc.s0, T=sc.T, seed=seed, mixture=mixture)
    total = time.perf_counter() - t0
    diag = {"is_diag": r.is_diag} if r.is_diag is not None else {}
    return CalibResult(r.field, {"total_s": total, "fit_s": r.fit_s}, diag, model=est)


def _nw(sc, n, seed, e, i, knobs=None): return _explicit(sc, n, seed, e, "nw", knobs=knobs)
def _nn(sc, n, seed, e, i, knobs=None): return _explicit(sc, n, seed, e, "nn", knobs=knobs)
def _ridge(sc, n, seed, e, i, knobs=None): return _explicit(sc, n, seed, e, "ridge", knobs=knobs)
def _spline(sc, n, seed, e, i, knobs=None): return _explicit(sc, n, seed, e, "spline", knobs=knobs)


def _nn_is(sc, n, seed, e, i, knobs=None):
    return _explicit(sc, n, seed, e, "nn", mixture=design_mixture(sc.dynamics, sc.T), knobs=knobs)


def _implicit_core(sc, n, seed, e, i):
    lv = sc.local_vol()
    warm = _explicit(sc, n, seed, e, "nn")
    t0 = time.perf_counter()
    r = calibrate_implicit(lv, sc.dynamics, replace(i, n_particles=n), s0=sc.s0, T=sc.T,
                           seed=seed, L0=warm.field)
    total = time.perf_counter() - t0 + warm.timings["total_s"]
    return lv, warm, r, total


def _implicit(sc, n, seed, e, i, knobs=None):
    _, warm, r, total = _implicit_core(sc, n, seed, e, i)
    return CalibResult(r.field, {"total_s": total, "fit_s": r.fit_s + warm.timings["fit_s"]},
                       {"deltas": r.deltas, "deltas_rms": r.deltas_rms}, model=r.net)


def _implicit_ridge(sc, n, seed, e, i, knobs=None):
    lv, _, r, overnight = _implicit_core(sc, n, seed, e, i)
    head = GlobalRidge(r.net, sc.T)
    t1 = time.perf_counter()
    er = calibrate_explicit(lv, sc.dynamics, head, replace(e, n_particles=n), s0=sc.s0, T=sc.T, seed=seed + 1)
    intraday = time.perf_counter() - t1
    return CalibResult(er.field, {"total_s": overnight + intraday, "fit_s": er.fit_s},
                       {"deltas": r.deltas, "overnight_s": overnight, "intraday_s": intraday})


def _nw_ghl(sc, n, seed, e, i, knobs=None): return _explicit(sc, n, seed, e, "nw_ghl", knobs=knobs)
def _muguruza(sc, n, seed, e, i, knobs=None): return _explicit(sc, n, seed, e, "muguruza", knobs=knobs)
def _rkhs(sc, n, seed, e, i, knobs=None): return _explicit(sc, n, seed, e, "rkhs", knobs=knobs)
def _bins(sc, n, seed, e, i, knobs=None): return _explicit(sc, n, seed, e, "bins", knobs=knobs)
def _purbf(sc, n, seed, e, i, knobs=None): return _explicit(sc, n, seed, e, "purbf", knobs=knobs)

def _nn_wide(sc, n, seed, e, i, knobs=None):
    """Explicit network on the fixed wide grid (mechanism test: grid extent)."""
    return _explicit(sc, n, seed, replace(e, grid="fixed"), "nn", knobs=knobs)


def _nw_wide(sc, n, seed, e, i, knobs=None):
    """Nadaraya-Watson on the fixed wide grid (mechanism test: grid extent)."""
    return _explicit(sc, n, seed, replace(e, grid="fixed"), "nw", knobs=knobs)


def _implicit_one(sc, n, seed, e, i, knobs=None):
    """One undamped re-simulate-and-fit from the explicit output (mechanism test: is it the iteration?)."""
    _, warm, r, total = _implicit_core(sc, n, seed, e, replace(i, n_iters=1, alpha=1.0))
    return CalibResult(r.field, {"total_s": total, "fit_s": r.fit_s + warm.timings["fit_s"]},
                       {"deltas": r.deltas, "deltas_rms": r.deltas_rms})


PAPER_ALGOS = ("nw", "explicit_nn", "ridge", "explicit_nn_is", "implicit_nn", "spline", "implicit_ridge")
BASELINE_ALGOS = ("nw_ghl", "muguruza", "rkhs", "bins", "purbf")
ALGOS = {"nw": _nw, "explicit_nn": _nn, "ridge": _ridge, "explicit_nn_is": _nn_is,
         "implicit_nn": _implicit, "spline": _spline, "implicit_ridge": _implicit_ridge,
         "nw_ghl": _nw_ghl, "muguruza": _muguruza, "rkhs": _rkhs, "bins": _bins, "purbf": _purbf}
# Diagnostic variants, kept out of ALGOS so the paper/baseline grids and goldens are unchanged.
def _nw_vfloor(sc, n, seed, e, i, knobs=None):
    """Nadaraya-Watson fitted on max(v, 0) (mechanism test: raw-vs-truncated variance target)."""
    return _explicit(sc, n, seed, replace(e, fit_v_floor=True), "nw", knobs=knobs)


def _nn_vfloor(sc, n, seed, e, i, knobs=None):
    return _explicit(sc, n, seed, replace(e, fit_v_floor=True), "nn", knobs=knobs)


def _rkhs_vfloor(sc, n, seed, e, i, knobs=None):
    return _explicit(sc, n, seed, replace(e, fit_v_floor=True), "rkhs", knobs=knobs)


MECHANISM_ALGOS = {"explicit_nn_wide": _nn_wide, "nw_wide": _nw_wide, "implicit_nn_1": _implicit_one,
                   "nw_vfloor": _nw_vfloor, "explicit_nn_vfloor": _nn_vfloor, "rkhs_vfloor": _rkhs_vfloor}


def _nn_tuned(sc, n, seed, e, i, knobs=None):
    """Per-slice network with the studies' tuned architecture and minibatch schedule."""
    return _explicit(sc, n, seed, replace(e, first_steps=2000, later_steps=500), "nn",
                     knobs={"hidden": 64, "depth": 3, "batch_size": 8192, "lr": 1e-3,
                            **(knobs or {})})


# Suite-only algorithms, kept out of ALGOS so the paper/baseline grids and goldens are unchanged.
SUITE_ALGOS = {"explicit_nn_tuned": _nn_tuned}


def run_algo(name, scenario, n_particles, seed, explicit=ExplicitConfig(), implicit=ImplicitConfig(), knobs=None):
    return {**ALGOS, **MECHANISM_ALGOS, **SUITE_ALGOS}[name](
        scenario, n_particles, seed, explicit, implicit, knobs=knobs)
