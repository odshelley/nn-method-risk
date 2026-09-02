"""Algorithm registry: one calibrate interface over the five contenders."""
import time
from dataclasses import dataclass

from neural_particle_method.explicit import calibrate_explicit
from neural_particle_method.implicit import GlobalRidgeHead, calibrate_implicit
from neural_particle_method.importance import design_mixture
from neural_particle_method.ssvi import SSVILocalVol


@dataclass
class CalibResult:
    L_records: list
    timings: dict
    diagnostics: dict


def _cfg(cfg, key, default):
    return default if cfg is None or key not in cfg else cfg[key]


def _explicit(scenario, n_particles, seed, cfg, method, mixture=None):
    lv = SSVILocalVol(scenario.ssvi, scenario.s0, T_max=scenario.T)
    t0 = time.perf_counter()
    _, info = calibrate_explicit(
        lv, scenario.dynamics, s0=scenario.s0, T=scenario.T,
        n_steps=_cfg(cfg, "n_steps", 50), n_particles=n_particles, method=method,
        fit_subsample=_cfg(cfg, "fit_subsample", 30_000), seed=seed,
        first_steps=_cfg(cfg, "first_steps", 400), later_steps=_cfg(cfg, "later_steps", 120),
        mixture=mixture)
    total = time.perf_counter() - t0
    diag = {"is_diag": info["is_diag"]} if "is_diag" in info else {}
    return CalibResult(info["L_records"], {"total_s": total, "fit_s": info["fit_s"]}, diag)


def _nw(sc, n, seed, cfg): return _explicit(sc, n, seed, cfg, "nw")
def _nn(sc, n, seed, cfg): return _explicit(sc, n, seed, cfg, "nn")
def _ridge(sc, n, seed, cfg): return _explicit(sc, n, seed, cfg, "ridge")


def _spline(sc, n, seed, cfg): return _explicit(sc, n, seed, cfg, "spline")


def _nn_is(sc, n, seed, cfg):
    mix = design_mixture(sc.dynamics, sc.T)
    return _explicit(sc, n, seed, cfg, "nn", mixture=mix)


def _implicit(sc, n, seed, cfg):
    lv = SSVILocalVol(sc.ssvi, sc.s0, T_max=sc.T)
    warm = _explicit(sc, n, seed, cfg, "nn")
    t0 = time.perf_counter()
    recs, info = calibrate_implicit(
        lv, sc.dynamics, s0=sc.s0, T=sc.T, n_steps=_cfg(cfg, "n_steps", 50),
        n_particles=n, alpha=_cfg(cfg, "alpha", 0.5), n_iters=_cfg(cfg, "n_iters", 6),
        seed=seed, fit_steps=_cfg(cfg, "fit_steps", 300), L0_records=warm.L_records)
    total = time.perf_counter() - t0 + warm.timings["total_s"]
    return CalibResult(recs, {"total_s": total, "fit_s": info["fit_s"] + warm.timings["fit_s"]},
                       {"deltas": info["deltas"]})


def _implicit_ridge(sc, n, seed, cfg):
    lv = SSVILocalVol(sc.ssvi, sc.s0, T_max=sc.T)
    warm = _explicit(sc, n, seed, cfg, "nn")
    t0 = time.perf_counter()
    recs, info = calibrate_implicit(
        lv, sc.dynamics, s0=sc.s0, T=sc.T, n_steps=_cfg(cfg, "n_steps", 50),
        n_particles=n, alpha=_cfg(cfg, "alpha", 0.5), n_iters=_cfg(cfg, "n_iters", 6),
        seed=seed, fit_steps=_cfg(cfg, "fit_steps", 300), L0_records=warm.L_records)
    overnight = time.perf_counter() - t0 + warm.timings["total_s"]
    head = GlobalRidgeHead(info["net"], sc.T)
    t1 = time.perf_counter()
    _, einfo = calibrate_explicit(
        lv, sc.dynamics, s0=sc.s0, T=sc.T, n_steps=_cfg(cfg, "n_steps", 50),
        n_particles=n, fit_subsample=_cfg(cfg, "fit_subsample", 30_000),
        seed=seed + 1, regressor=head)
    intraday = time.perf_counter() - t1
    return CalibResult(einfo["L_records"],
                       {"total_s": overnight + intraday, "fit_s": einfo["fit_s"]},
                       {"deltas": info["deltas"], "overnight_s": overnight,
                        "intraday_s": intraday})


ALGOS = {"nw": _nw, "explicit_nn": _nn, "ridge": _ridge,
         "explicit_nn_is": _nn_is, "implicit_nn": _implicit, "spline": _spline,
         "implicit_ridge": _implicit_ridge}


def run_algo(name, scenario, n_particles, seed, cfg=None):
    return ALGOS[name](scenario, n_particles, seed, cfg)
