"""Paper grid enumeration and a resumable, optionally parallel sweep."""
from concurrent.futures import ProcessPoolExecutor

from ..calibrate.config import ExplicitConfig, ImplicitConfig
from ..pricing.reprice import RepriceConfig
from ..tracking.store import Store
from .algos import ALGOS, PAPER_ALGOS
from .runner import BENCH_EXPERIMENT, run_key, run_one
from .scenarios import fig3_registry, make_registry

BUDGETS = (1_000, 10_000, 100_000)


def paper_grid():
    jobs = []
    for sid in make_registry():
        for algo in PAPER_ALGOS:
            for n in (50_000, 200_000):
                for seed in (0, 1, 2):
                    jobs.append((sid, algo, n, seed))
    for sid in fig3_registry():
        for algo in ("nw", "explicit_nn"):
            for seed in (0, 1, 2):
                jobs.append((sid, algo, 200_000, seed))
    return jobs


def baselines_grid():
    jobs = []
    for sid in make_registry():
        for algo in ALGOS:
            for n in BUDGETS:
                for seed in (0, 1, 2):
                    jobs.append((sid, algo, n, seed))
    for sid in fig3_registry():
        for algo in ALGOS:
            for n in BUDGETS[1:]:
                for seed in (0, 1, 2):
                    jobs.append((sid, algo, n, seed))
    return jobs


def _worker(args):
    uri, root, job, explicit, implicit, reprice, experiment = args
    store = Store(uri, root)
    run_one(store, *job, explicit=explicit, implicit=implicit, reprice=reprice, experiment=experiment)
    return job


def sweep(store, jobs, n_jobs=1, explicit=ExplicitConfig(), implicit=ImplicitConfig(), reprice=RepriceConfig(),
          experiment=BENCH_EXPERIMENT):
    todo = [j for j in jobs if store.find_finished(experiment, run_key(*j)) is None]
    print(f"{len(todo)} runs to do", flush=True)
    args = [(store.tracking_uri, store.artifact_root, j, explicit, implicit, reprice, experiment) for j in todo]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for done in ex.map(_worker, args):
                print("done:", *done, flush=True)
    else:
        for a in args:
            print("done:", *_worker(a), flush=True)
    return len(todo)
