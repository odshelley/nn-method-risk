import pytest

from neural_particle_method.estimators import recipes as R
from neural_particle_method.suite.config import SuiteSettings
from neural_particle_method.suite.far_wings import (
    REPRICE_TILT,
    rescore_run,
    run_farwings,
    select_runs,
)
from neural_particle_method.suite.tilt import run_tilt_cold
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()
FAST = {"hidden": 16, "depth": 2, "lr": 1e-2, "batch_size": 0, "first_steps": 5,
        "later_steps": 2, "weight_decay": 0.0, "fit_subsample": 400, "warm_start": True,
        "mean_match": True, "monotone": False, "monotone_penalty": 0.0, "tail": "linear",
        "hetero": False}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RECIPE_DIR", tmp_path / "recipes")
    R.save_recipe("explicit_opt", FAST)
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def test_rescore_logs_far_metrics_once(store):
    rid = run_tilt_cold(store, "s01", "nw", TINY.n_online, 0, "none", TINY)
    m = rescore_run(store, rid)
    assert m is not None and "far_wings_mae_bp" in m and m["far_wings_n"] > 0
    got = store.get_metrics(rid)
    assert got["far_wings_n"] == m["far_wings_n"] and "far_all_mae_bp" in got
    names = {a.path for a in store.client.list_artifacts(rid)}
    assert "far_iv_err_bp.json" in names
    assert rescore_run(store, rid) is None            # already scored
    assert rescore_run(store, rid, force=True) is not None
    assert REPRICE_TILT.name == "constant-3"


def test_select_runs_targets_the_80k_rows_and_floors(store):
    a = run_tilt_cold(store, "s01", "nw", TINY.n_online, 0, "none", TINY)
    sel = select_runs(store, settings=TINY, budget=TINY.n_online)
    assert (TINY.experiment("tilt_cold"), a) in sel
    done, failed = run_farwings(store, settings=TINY, budget=TINY.n_online)
    assert (done, failed) == (1, 0)
    assert run_farwings(store, settings=TINY, budget=TINY.n_online) == (0, 0)
