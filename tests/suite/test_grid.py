import pytest

from neural_particle_method.estimators import recipes as R
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
FAST = {"hidden": 16, "depth": 2, "lr": 1e-2, "batch_size": 0, "first_steps": 5,
        "later_steps": 2, "weight_decay": 0.0, "fit_subsample": 400, "warm_start": True,
        "mean_match": False, "monotone": False, "monotone_penalty": 0.0, "tail": "flat",
        "hetero": False}


def test_full_job_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RECIPE_DIR", tmp_path)      # counts before anything is promoted
    assert len(pde_jobs(FULL)) == 23 * 2 + 23 * 2   # floors (2 seeds) + lagged references (2 lags)
    assert len(cold_jobs(FULL)) == 23 * 10 * 2
    assert len(offline_jobs(FULL)) == 23 * 3 * 2
    # 8 head/stale methods x 2 sizes, 4 tuned-body methods at 500k only, stale_L, nw_resolve;
    # 2 lags; 2 seeds
    assert len(online_jobs(FULL)) == 23 * (8 * 2 + 4 * 1 + 1 + 1) * 2 * 2
    assert STAGES == ("pde", "cold", "offline", "online")


def test_job_lists_grow_when_a_recipe_is_promoted(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RECIPE_DIR", tmp_path)
    R.save_recipe("explicit_opt", FAST)
    assert len(cold_jobs(FULL)) == 23 * 11 * 2
    # the two searched-body methods join the four tuned-body ones at the largest size only
    assert len(online_jobs(FULL)) == 23 * (8 * 2 + 4 + 2 + 1 + 1) * 2 * 2


def test_tuned_methods_only_run_on_the_largest_body(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RECIPE_DIR", tmp_path)
    sizes = {j[3] for j in online_jobs(FULL) if j[2].startswith("explicit_tuned_")}
    assert sizes == {max(FULL.offline_sizes)}


def test_sid_filter_and_online_refuses_without_bodies(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RECIPE_DIR", tmp_path / "recipes")
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
