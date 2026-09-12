"""Tuning clouds for the estimator search: one MLflow run per held-out scenario holding the
particle slices an offline pass sees, split into a fit pool and a held-out set, with NW's
held-out loss per slice as the yardstick."""
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..bench.scenarios import make_tuning_registry
from ..calibrate.explicit import calibrate_explicit
from ..estimators.nadaraya_watson import NadarayaWatson, nw_estimate
from ..tracking.store import git_hash
from .config import FULL

EXPERIMENT = "optuna_clouds"
CACHE_ROOT = "results/optuna/clouds"
SLICE_TIMES = (0.10, 0.50, 1.00, 1.75)
HELD_FRAC = 0.2


def quantile_grid(lnx):
    return np.unique(np.quantile(lnx, np.linspace(0.001, 0.999, 101)))


def heldout_loss(grid, f_grid, lnx_held, v_held):
    return float(np.mean((np.interp(lnx_held, grid, f_grid) - v_held) ** 2))


class RecordingNW(NadarayaWatson):
    """NW estimator that also keeps the (t, lnx, v) it was handed at the requested slice times."""

    def __init__(self, times, dt):
        super().__init__()
        self.dt = dt
        self.steps = {round(t / dt): float(t) for t in times}
        self.records = []

    def fit_predict(self, t, lnx, v, grid, weights=None):
        k = round(t / self.dt)
        if k in self.steps:
            self.records.append((self.steps[k], lnx.copy(), v.copy()))
        return super().fit_predict(t, lnx, v, grid, weights=weights)


@dataclass
class Cloud:
    sid: str
    rho: float
    times: list
    fit: list       # [(lnx, v_plus)] per slice, the fit pool
    held: list      # [(lnx, v_plus)] per slice, held out
    nw_loss: list   # NW held-out MSE per slice, fitted on the whole fit pool


def cloud_key(sid, settings=FULL):
    return {"sid": sid, "n_particles": int(settings.offline_sizes[-1]),
            "n_steps": int(settings.explicit.n_steps), "seed": 0}


def _split(lnx, v):
    n_held = int(HELD_FRAC * len(lnx))
    return (lnx[:-n_held], v[:-n_held]), (lnx[-n_held:], v[-n_held:])


def ensure_cloud(store, sid, settings=FULL, registry=None, times=SLICE_TIMES):
    key = cloud_key(sid, settings)
    existing = store.find_finished(EXPERIMENT, key)
    if existing is not None:
        return existing
    sc = (registry or make_tuning_registry())[sid]
    n = key["n_particles"]
    # every fit on the whole cloud; rng.choice without replacement hands the estimator a random
    # permutation, which is the split's randomness
    ecfg = replace(settings.explicit, n_particles=n, fit_subsample=n, fit_v_floor=True)
    est = RecordingNW(times, sc.T / ecfg.n_steps)
    params = {**key, "git_hash": git_hash(), "times": ",".join(f"{t:g}" for t in times),
              **sc.as_params()}
    with store.run(EXPERIMENT, params) as h:
        calibrate_explicit(sc.local_vol(), sc.dynamics, est, ecfg, s0=sc.s0, T=sc.T, seed=0)
        arrays = {"times": np.array([r[0] for r in est.records]), "rho": sc.dynamics.rho}
        metrics = {}
        for i, (t, lnx, v) in enumerate(est.records):
            (lf, vf), (lh, vh) = _split(lnx, v)
            grid = quantile_grid(lf)
            loss = heldout_loss(grid, nw_estimate(lf, vf, grid), lh, vh)
            arrays.update({f"lnx_fit_{i}": lf, f"v_fit_{i}": vf, f"lnx_held_{i}": lh,
                           f"v_held_{i}": vh})
            metrics[f"nw_loss/t{t:g}"] = loss
        h.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cloud.npz"
            np.savez_compressed(p, **arrays)
            h.log_file(p)
        return h.run_id


def cloud_cache(run_id, cache_dir=None):
    """The per-run cache directory a worker reads `cloud.npz` from."""
    return Path(cache_dir) if cache_dir is not None else Path(CACHE_ROOT) / run_id


def load_cloud(store, run_id, cache_dir=None):
    """The cloud, from the per-run cache, downloading it once if it is not there.

    The download is atomic: several workers may race on the same run id, and a half-written
    `cloud.npz` would be loaded as a truncated archive by whoever looked next. Arrays come back
    as float32; the fits run in float32 anyway and the full clouds are held in memory per worker.
    """
    cache = cloud_cache(run_id, cache_dir)
    p = cache / "cloud.npz"
    if not p.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache.parent) as d:
            os.replace(store.download(run_id, "cloud.npz", d), p)
    z = np.load(p)
    times = [float(t) for t in z["times"]]
    m = store.get_metrics(run_id)

    def pair(kind, i):
        return (z[f"lnx_{kind}_{i}"].astype(np.float32), z[f"v_{kind}_{i}"].astype(np.float32))

    return Cloud(sid=store.get_params(run_id)["sid"], rho=float(z["rho"]), times=times,
                 fit=[pair("fit", i) for i in range(len(times))],
                 held=[pair("held", i) for i in range(len(times))],
                 nw_loss=[m[f"nw_loss/t{t:g}"] for t in times])
