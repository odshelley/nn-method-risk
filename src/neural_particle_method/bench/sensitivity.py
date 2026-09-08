"""Knob-sensitivity sweeps: one knob per method over a range centred on its published default."""
from concurrent.futures import ProcessPoolExecutor

from ..calibrate.config import ExplicitConfig, ImplicitConfig
from ..pricing.reprice import RepriceConfig
from ..tracking.store import Store
from .runner import run_key, run_one

SENSITIVITY_EXPERIMENT = "sensitivity"
_SCALES = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
KNOBS = {
    "nw_ghl": ("c", _SCALES),
    "nw": ("bandwidth_scale", _SCALES),
    "rkhs": ("lam", tuple(10.0 ** -k for k in range(9, 1, -1))),
    "bins": ("n_bins", (5, 10, 20, 50, 100, 200, 500)),
    "purbf": ("prune", (0.5, 1.0, 2.0)),
    "explicit_nn": ("hidden", (8, 16, 32, 64, 128)),
    "ridge": ("lam", (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)),
    "spline": ("lam", (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)),
}


def sensitivity_grid(sids=("s01", "li_simple"), budgets=(10_000, 100_000), seeds=(0, 1, 2), algos=None):
    jobs = []
    for sid in sids:
        for algo in (algos or KNOBS):
            name, values = KNOBS[algo]
            for n in budgets:
                for seed in seeds:
                    for val in values:
                        jobs.append((sid, algo, n, seed, name, val))
    return jobs


def _key(job):
    sid, algo, n, seed, name, val = job
    return {**run_key(sid, algo, n, seed), "knob_name": name, "knob_value": val}


def _worker(args):
    uri, root, job, explicit, implicit, reprice = args
    sid, algo, n, seed, name, val = job
    store = Store(uri, root)
    run_one(store, sid, algo, n, seed, explicit=explicit, implicit=implicit, reprice=reprice,
            experiment=SENSITIVITY_EXPERIMENT, knobs={name: val}, extra_key={"knob_name": name, "knob_value": val})
    return job


def run_sensitivity(store, jobs, n_jobs=1, explicit=ExplicitConfig(), implicit=ImplicitConfig(), reprice=RepriceConfig()):
    todo = [j for j in jobs if store.find_finished(SENSITIVITY_EXPERIMENT, _key(j)) is None]
    print(f"{len(todo)} sensitivity runs to do", flush=True)
    args = [(store.tracking_uri, store.artifact_root, j, explicit, implicit, reprice) for j in todo]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for done in ex.map(_worker, args):
                print("done:", *done, flush=True)
    else:
        for a in args:
            print("done:", *_worker(a), flush=True)
    return len(todo)
