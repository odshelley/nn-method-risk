import pytest

from neural_particle_method.suite.config import FULL, SuiteSettings
from neural_particle_method.suite.grid import (
    STAGES,
    cold_jobs,
    offline_jobs,
    online_jobs,
    pde_jobs,
    run_stage,
)
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()


def test_full_job_counts():
    assert len(pde_jobs(FULL)) == 23 * 2 + 23 * 2   # floors (2 seeds) + lagged references (2 lags)
    assert len(cold_jobs(FULL)) == 23 * 10 * 2
    assert len(offline_jobs(FULL)) == 23 * 2 * 2
    # 6 head/stale methods x 2 sizes, stale_L, nw_resolve; 2 lags; 2 seeds
    assert len(online_jobs(FULL)) == 23 * (8 * 2 + 1 + 1) * 2 * 2
    assert STAGES == ("pde", "cold", "offline", "online")


def test_sid_filter_and_online_refuses_without_bodies(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    assert {j[1] for j in cold_jobs(TINY, sids=["s01"])} == {"s01"}
    with pytest.raises(RuntimeError, match="offline"):
        run_stage(store, "online", TINY, sids=["s01"])


def test_run_stage_runs_and_skips_finished(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    small = SuiteSettings(**{**TINY.__dict__, "sids": ("s01",)})
    done, failed = run_stage(store, "pde", small)
    assert (done, failed) == (3, 0)      # one floor plus the two lagged references
    assert run_stage(store, "pde", small) == (0, 0)
