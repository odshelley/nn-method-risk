import pytest

from neural_particle_method.estimators import recipes as R
from neural_particle_method.suite.artifacts import load_run
from neural_particle_method.suite.cold import COLD_ALGOS, cold_extra_key, run_cold
from neural_particle_method.suite.config import FULL, SMOKE, SuiteSettings, cold_algos
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def test_full_settings_match_the_spec(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RECIPE_DIR", tmp_path)      # a promoted recipe must not change this
    assert (FULL.explicit.n_steps, FULL.implicit.n_steps, FULL.reprice.n_steps) == (200, 200, 200)
    assert FULL.explicit.fit_v_floor is True and FULL.reprice.n_particles == 500_000
    assert FULL.explicit.fit_subsample == 100_000   # every per-slice fit uses the full online cloud
    assert (FULL.implicit.n_iters, FULL.implicit.alpha) == (30, 0.5)
    assert (FULL.n_online == 100_000 and FULL.offline_sizes == (200_000, 500_000)
           and FULL.seeds == (0, 1))
    assert len(FULL.sids) == 23 and FULL.sids[:2] == ("s01", "s02") and FULL.sids[-1] == "bayer"
    assert (FULL.experiment("suite_cold") == "suite_cold"
           and SMOKE.experiment("suite_cold") == "suite_cold_smoke")
    assert COLD_ALGOS == ("nw", "explicit_nn", "explicit_nn_tuned", "implicit_nn", "rkhs",
                          "spline", "nw_ghl", "bins", "muguruza", "purbf")
    # the searched network joins the cold rows only once a recipe has been promoted
    assert cold_algos() == COLD_ALGOS


@pytest.mark.parametrize("algo", ["nw", "explicit_nn", "implicit_nn"])
def test_run_cold_logs_and_saves_model(store, algo):
    rid = run_cold(store, "s01", algo, 0, TINY)
    p, m = store.get_params(rid), store.get_metrics(rid)
    assert (p["algo"] == algo and p["n_particles"] == str(TINY.n_online)
           and p["explicit.fit_v_floor"] == "True")
    assert m["pooled_mae_bp"] >= 0 and m["fit_s"] >= 0
    lr = load_run(store, rid)
    if algo == "nw":
        assert lr.model is None and lr.meta["kind"] == "field_only"
    else:
        assert lr.model is not None and lr.meta["kind"] == {
            "explicit_nn": "explicit_slices", "implicit_nn": "implicit_net"
        }[algo]
    assert run_cold(store, "s01", algo, 0, TINY) == rid
    assert len(store.search(TINY.experiment("suite_cold"))) == 1


def _fast_recipe():
    return {"hidden": 16, "depth": 2, "lr": 1e-2, "batch_size": 0, "first_steps": 3,
            "later_steps": 1, "weight_decay": 0.0, "fit_subsample": 400, "warm_start": True,
            "mean_match": False, "monotone": False, "monotone_penalty": 0.0, "tail": "flat",
            "hetero": False}


def test_searched_cold_run_is_keyed_by_the_promoted_recipe(store, tmp_path, monkeypatch):
    """A second promotion must not read back the first recipe's cold rows."""
    monkeypatch.setattr(R, "RECIPE_DIR", tmp_path)
    R.save_recipe("explicit_opt", _fast_recipe())
    want = R.recipe_hash(R.load_recipe("explicit_opt"))
    rid = run_cold(store, "s01", "explicit_nn_opt", 0, TINY)
    assert store.get_params(rid)["recipe_hash"] == want
    assert run_cold(store, "s01", "explicit_nn_opt", 0, TINY) == rid
    assert len(store.search(TINY.experiment("suite_cold"))) == 1
    R.save_recipe("explicit_opt", {**_fast_recipe(), "tail": "linear"})
    other = run_cold(store, "s01", "explicit_nn_opt", 0, TINY)
    assert other != rid and store.get_params(other)["recipe_hash"] != want


def test_only_the_searched_cold_algo_carries_a_recipe_hash(store, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RECIPE_DIR", tmp_path)
    R.save_recipe("explicit_opt", _fast_recipe())
    assert cold_extra_key("explicit_nn") is None and cold_extra_key("nw") is None
    assert set(cold_extra_key("explicit_nn_opt")) == {"recipe_hash"}
    rid = run_cold(store, "s01", "explicit_nn", 0, TINY)
    assert "recipe_hash" not in store.get_params(rid)


def test_grid_sees_the_searched_cold_job_as_finished_only_for_its_own_recipe(store, tmp_path,
                                                                             monkeypatch):
    from neural_particle_method.suite.grid import _is_finished
    monkeypatch.setattr(R, "RECIPE_DIR", tmp_path)
    R.save_recipe("explicit_opt", _fast_recipe())
    job = ("cold", "s01", "explicit_nn_opt", 0)
    assert not _is_finished(store, job, TINY)
    run_cold(store, "s01", "explicit_nn_opt", 0, TINY)
    assert _is_finished(store, job, TINY)
    R.save_recipe("explicit_opt", {**_fast_recipe(), "tail": "linear"})
    assert not _is_finished(store, job, TINY)
