import pytest

from neural_particle_method.bench.algos import ALGOS
from neural_particle_method.bench.runner import run_one
from neural_particle_method.bench.sweep import paper_grid, sweep
from neural_particle_method.pricing.reprice import RepriceConfig
from neural_particle_method.tracking.store import Store
from tests.conftest import TINY_EXPLICIT, TINY_IMPLICIT, TINY_N, TINY_REPRICE_N, TINY_REPRICE_STEPS

TINY_REPRICE = RepriceConfig(TINY_REPRICE_N, TINY_REPRICE_STEPS)


def test_paper_grid_size():
    jobs = paper_grid()
    assert len(jobs) == 20 * len(ALGOS) * 2 * 3 + 6 * 2 * 3
    assert jobs[0] == ("s01", "nw", 50_000, 0)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_sweep_skips_finished_runs(tmp_path, n_jobs):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    run_one(store, "s01", "nw", TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT, TINY_REPRICE)
    jobs = [("s01", "nw", TINY_N, 0), ("s01", "nw", TINY_N, 1)]
    done = sweep(store, jobs, n_jobs=n_jobs, explicit=TINY_EXPLICIT, implicit=TINY_IMPLICIT, reprice=TINY_REPRICE)
    assert done == 1 and len(store.search("bench")) == 2
