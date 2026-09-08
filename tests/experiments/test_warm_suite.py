import pytest

from neural_particle_method.experiments.bump_correct import run_pair
from neural_particle_method.experiments.config import BUMP_SMOKE, SMOKE
from neural_particle_method.experiments.warm_suite import ARMS, job_list, run_warm, summarise
from neural_particle_method.bench.scenarios import make_registry
from neural_particle_method.tracking.store import Store


def test_job_list_enumerates_expected_jobs():
    jobs = job_list(["dyn", "norm"], ["s01", "s02"])
    assert jobs == [("dyn", "s01", 0), ("dyn", "s01", 1), ("dyn", "s02", 0), ("dyn", "s02", 1),
                    ("norm", "s01", 0), ("norm", "s02", 0)]
    assert set(ARMS) == {"dyn", "seq", "xover", "norm"}


@pytest.mark.slow
@pytest.mark.parametrize("arm", sorted(ARMS))
def test_each_arm_smokes_through_store(tmp_path, arm):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    # "dyn" runs both of job_list's default dyn_seeds (0, 1) per sid; every other arm runs one.
    n_jobs = 2 if arm == "dyn" else 1
    assert run_warm(store, [arm], ["s01"], SMOKE) == n_jobs
    assert run_warm(store, [arm], ["s01"], SMOKE) == 0          # resumable
    assert len(store.search("overnight")) == n_jobs               # one cache per seed, logged once each
    md = summarise(store)
    assert arm in md


@pytest.mark.slow
def test_bump_smokes_through_store(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    rid = run_pair(store, make_registry()["s01"], 0, BUMP_SMOKE)
    m = store.get_metrics(rid)
    assert "rmse_bp/causal_sweep" in m and "build_s/full_resolve" in m
    assert run_pair(store, make_registry()["s01"], 0, BUMP_SMOKE) == rid
