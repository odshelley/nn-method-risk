import pytest

from neural_particle_method.estimators import recipes as R
from neural_particle_method.suite.config import SuiteSettings
from neural_particle_method.suite.tilt import (
    COLD_PARTICLES,
    SLICE_SIDS,
    cold_jobs,
    mixture_for,
    run_tilt_cold,
)
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()
FAST = {"hidden": 16, "depth": 2, "lr": 1e-2, "batch_size": 0, "first_steps": 5,
        "later_steps": 2, "weight_decay": 0.0, "fit_subsample": 400, "warm_start": True,
        "mean_match": True, "monotone": False, "monotone_penalty": 0.0, "tail": "linear",
        "hetero": False}


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


@pytest.fixture
def promoted(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RECIPE_DIR", tmp_path / "recipes")
    R.save_recipe("explicit_opt", FAST)


def test_cold_job_counts(promoted):
    jobs = cold_jobs("constant-3")
    assert len(jobs) == (20 + 6) * 3 * 2 * 2
    assert {j[5] for j in jobs} == {"none", "constant-3"}
    assert {j[3] for j in jobs} == set(COLD_PARTICLES)
    only_s02 = cold_jobs("constant-3", sids=("s02",))
    assert len(only_s02) == 2 * 3 * 2 * 2 and all(j[1] == "s02" for j in only_s02)
    assert SLICE_SIDS == ("s01", "s02", "s05", "s09", "s11", "s16")
    with pytest.raises(KeyError):
        cold_jobs("constant-2")


def test_mixture_for():
    assert mixture_for("none", 4, 1.0, -0.5) is None
    m = mixture_for("inverse_sqrt-1", 4, 1.0, -0.5)
    assert m.scheduled and m.thetas.shape == (3, 4)


def test_cold_cell_is_keyed_by_design_and_logs_diagnostics(store, promoted):
    a = run_tilt_cold(store, "s01", "nw", TINY.n_online, 0, "none", TINY)
    b = run_tilt_cold(store, "s01", "nw", TINY.n_online, 0, "constant-3", TINY)
    assert a != b
    assert run_tilt_cold(store, "s01", "nw", TINY.n_online, 0, "constant-3", TINY) == b
    pa, pb = store.get_params(a), store.get_params(b)
    assert pa["design"] == "none" and pb["design"] == "constant-3"
    ma, mb = store.get_metrics(a), store.get_metrics(b)
    assert "ess_min_slice" not in ma
    assert 0 < mb["ess_min_slice"] <= 1 and mb["max_w"] <= 2 + 1e-9
    assert mb["pooled_mae_bp"] >= 0 and "liquid_mae_bp" in mb
    assert len(store.search(TINY.experiment("tilt_cold"))) == 2


def test_cold_cell_on_the_searched_network_carries_the_recipe_hash(store, promoted):
    rid = run_tilt_cold(store, "s01", "explicit_nn_opt", TINY.n_online, 0, "constant-1", TINY)
    p = store.get_params(rid)
    assert p["recipe_hash"] == R.recipe_hash(FAST) and p["design"] == "constant-1"
    with pytest.raises(ValueError):
        run_tilt_cold(store, "s01", "spline", TINY.n_online, 0, "constant-1", TINY)
