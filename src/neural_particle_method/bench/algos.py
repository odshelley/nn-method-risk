"""Algorithm registry: one calibrate interface over the seven contenders."""
import time
from dataclasses import dataclass, replace

from ..calibrate.config import ExplicitConfig, ImplicitConfig
from ..calibrate.explicit import calibrate_explicit
from ..calibrate.implicit import calibrate_implicit
from ..calibrate.importance import design_mixture
from ..estimators import make_estimator
from ..estimators.ridge import GlobalRidge
from ..market.local_vol import SSVILocalVol
from ..simulate.leverage import LeverageField


@dataclass
class CalibResult:
    field: LeverageField
    timings: dict
    diagnostics: dict


def _explicit(sc, n, seed, ecfg, method, mixture=None, estimator=None):
    lv = SSVILocalVol(sc.ssvi, sc.s0, T_max=sc.T)
    cfg = replace(ecfg, n_particles=n)
    est = estimator if estimator is not None else make_estimator(
        method, seed=seed, first_steps=cfg.first_steps, later_steps=cfg.later_steps)
    t0 = time.perf_counter()
    r = calibrate_explicit(lv, sc.dynamics, est, cfg, s0=sc.s0, T=sc.T, seed=seed, mixture=mixture)
    total = time.perf_counter() - t0
    diag = {"is_diag": r.is_diag} if r.is_diag is not None else {}
    return CalibResult(r.field, {"total_s": total, "fit_s": r.fit_s}, diag)


def _nw(sc, n, seed, e, i): return _explicit(sc, n, seed, e, "nw")
def _nn(sc, n, seed, e, i): return _explicit(sc, n, seed, e, "nn")
def _ridge(sc, n, seed, e, i): return _explicit(sc, n, seed, e, "ridge")
def _spline(sc, n, seed, e, i): return _explicit(sc, n, seed, e, "spline")


def _nn_is(sc, n, seed, e, i):
    return _explicit(sc, n, seed, e, "nn", mixture=design_mixture(sc.dynamics, sc.T))


def _implicit_core(sc, n, seed, e, i):
    lv = SSVILocalVol(sc.ssvi, sc.s0, T_max=sc.T)
    warm = _explicit(sc, n, seed, e, "nn")
    t0 = time.perf_counter()
    r = calibrate_implicit(lv, sc.dynamics, replace(i, n_particles=n), s0=sc.s0, T=sc.T,
                           seed=seed, L0=warm.field)
    total = time.perf_counter() - t0 + warm.timings["total_s"]
    return lv, warm, r, total


def _implicit(sc, n, seed, e, i):
    _, warm, r, total = _implicit_core(sc, n, seed, e, i)
    return CalibResult(r.field, {"total_s": total, "fit_s": r.fit_s + warm.timings["fit_s"]},
                       {"deltas": r.deltas})


def _implicit_ridge(sc, n, seed, e, i):
    lv, warm, r, overnight = _implicit_core(sc, n, seed, e, i)
    head = GlobalRidge(r.net, sc.T)
    t1 = time.perf_counter()
    er = calibrate_explicit(lv, sc.dynamics, head, replace(e, n_particles=n), s0=sc.s0, T=sc.T, seed=seed + 1)
    intraday = time.perf_counter() - t1
    return CalibResult(er.field, {"total_s": overnight + intraday, "fit_s": er.fit_s},
                       {"deltas": r.deltas, "overnight_s": overnight, "intraday_s": intraday})


ALGOS = {"nw": _nw, "explicit_nn": _nn, "ridge": _ridge, "explicit_nn_is": _nn_is,
         "implicit_nn": _implicit, "spline": _spline, "implicit_ridge": _implicit_ridge}


def run_algo(name, scenario, n_particles, seed, explicit=ExplicitConfig(), implicit=ImplicitConfig()):
    return ALGOS[name](scenario, n_particles, seed, explicit, implicit)
