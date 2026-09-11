"""Job lists and the multiprocess driver for the four suite stages."""
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from ..bench.runner import run_key
from ..tracking.store import Store
from .cold import COLD_ALGOS, run_cold
from .config import FULL
from .lag import LAGS
from .offline import BODIES, run_offline
from .online import ONLINE_METHODS, ensure_lagged_reference, run_online
from .reference import run_pde_floor

STAGES = ("pde", "cold", "offline", "online")
EXPERIMENTS = {
    "pde": ("pde_reference", "suite_pde_floor"),
    "cold": ("suite_cold", "pde_reference"),
    "offline": ("suite_offline",),
    "online": ("suite_lagged", "suite_offline", "pde_reference"),
}


def _sids(settings, sids):
    if sids is None:
        return tuple(settings.sids)
    return tuple(s for s in settings.sids if s in set(sids))


def pde_jobs(settings=FULL, sids=None):
    """PDE floors for the registry scenarios, then one reference per (scenario, lag)."""
    sids = _sids(settings, sids)
    jobs = [("pde", sid, seed) for sid in sids for seed in settings.seeds]
    jobs += [("pde_lag", sid, lag.kind) for sid in sids for lag in LAGS]
    return jobs


def cold_jobs(settings=FULL, sids=None):
    return [("cold", sid, algo, seed) for sid in _sids(settings, sids) for algo in COLD_ALGOS
            for seed in settings.seeds]


def offline_jobs(settings=FULL, sids=None):
    return [("offline", sid, body, n) for sid in _sids(settings, sids) for body in BODIES
            for n in settings.offline_sizes]


def online_jobs(settings=FULL, sids=None):
    jobs = []
    for sid in _sids(settings, sids):
        for lag in LAGS:
            for seed in settings.seeds:
                for method, (body, _) in ONLINE_METHODS.items():
                    if body is None:
                        sizes = (0,)
                    elif body == "explicit_tuned":
                        # the head table quotes the tuned body at the largest size only
                        sizes = (max(settings.offline_sizes),)
                    elif method == "stale_L":
                        sizes = (settings.offline_sizes[0],)
                    else:
                        sizes = settings.offline_sizes
                    for n in sizes:
                        jobs.append(("online", sid, method, n, lag.kind, seed))
    return jobs


def run_job(store, job, settings=FULL):
    kind = job[0]
    if kind == "pde":
        return run_pde_floor(store, job[1], job[2], settings)
    if kind == "cold":
        return run_cold(store, job[1], job[2], job[3], settings)
    if kind == "offline":
        return run_offline(store, job[1], job[2], job[3], settings)
    if kind == "pde_lag":
        lag = {l.kind: l for l in LAGS}[job[2]]
        return ensure_lagged_reference(store, job[1], lag, settings)
    if kind == "online":
        lag = {l.kind: l for l in LAGS}[job[4]]
        return run_online(store, job[1], job[2], job[3], lag, job[5], settings)
    raise KeyError(kind)


def _worker(args):
    uri, root, job, settings, n_jobs = args
    if n_jobs > 1:
        import torch
        k = max(1, (os.cpu_count() or n_jobs) // n_jobs)
        os.environ.setdefault("OMP_NUM_THREADS", str(k))
        torch.set_num_threads(k)
    try:
        run_job(Store(uri, root), job, settings)
        return job, None
    except Exception as e:  # noqa: BLE001 - the run is already recorded FAILED by Store.run
        return job, repr(e)


JOBS = {"pde": pde_jobs, "cold": cold_jobs, "offline": offline_jobs, "online": online_jobs}


def _is_finished(store, job, settings):
    kind = job[0]
    n_steps = settings.explicit.n_steps
    if kind == "pde":
        _, sid, seed = job
        key = {"sid": sid, "seed": seed, "n_steps": n_steps}
        return store.find_finished(settings.experiment("suite_pde_floor"), key) is not None
    if kind == "pde_lag":
        _, sid, lag_kind = job
        key = {"sid": sid, "n_steps": n_steps, "lag": lag_kind}
        return store.find_finished("pde_reference", key) is not None
    if kind == "cold":
        _, sid, algo, seed = job
        key = run_key(sid, algo, settings.n_online, seed)
        return store.find_finished(settings.experiment("suite_cold"), key) is not None
    if kind == "offline":
        _, sid, body, n = job
        key = {"sid": sid, "body": body, "n_particles": int(n), "seed": 0, "n_steps": n_steps}
        return store.find_finished(settings.experiment("suite_offline"), key) is not None
    if kind == "online":
        _, sid, method, offline_n, lag_kind, seed = job
        body, _ = ONLINE_METHODS[method]
        key = {"sid": sid, "method": method, "offline_n": int(offline_n if body else 0),
               "lag": lag_kind, "seed": int(seed), "n_steps": n_steps}
        return store.find_finished(settings.experiment("suite_lagged"), key) is not None
    raise KeyError(kind)


def _drain(results):
    """Count results as they arrive; a broken pool ends the stage with one extra failure."""
    done = failed = 0
    try:
        for job, err in results:
            if err is None:
                done += 1
                print("done:", *job, flush=True)
            else:
                failed += 1
                print("FAILED:", *job, err, flush=True)
    except BrokenProcessPool as e:
        print(f"FAILED: pool broken: {e}", flush=True)
        failed += 1
    return done, failed


def run_stage(store, stage, settings=FULL, n_jobs=1, sids=None):
    """Run every unfinished job of `stage`; returns (n_done, n_failed).

    Pre-creates experiments (MLflow race).
    """
    if stage == "online":
        offline_key = {"seed": 0, "n_steps": settings.explicit.n_steps}
        missing = [j for j in offline_jobs(settings, sids)
                   if store.find_finished(settings.experiment("suite_offline"),
                                          {"sid": j[1], "body": j[2], "n_particles": j[3],
                                           **offline_key}) is None]
        if missing:
            raise RuntimeError(f"{len(missing)} offline bodies unfinished; "
                               "run --stage offline first")
    for name in EXPERIMENTS[stage]:
        store.experiment_id(settings.experiment(name) if name != "pde_reference" else name)
    jobs = JOBS[stage](settings, sids)
    jobs = [j for j in jobs if not _is_finished(store, j, settings)]
    print(f"{stage}: {len(jobs)} jobs listed", flush=True)
    args = [(store.tracking_uri, store.artifact_root, j, settings, n_jobs) for j in jobs]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            return _drain(pool.map(_worker, args))
    return _drain(map(_worker, args))
