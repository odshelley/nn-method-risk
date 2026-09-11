import pytest

from neural_particle_method.estimators import recipes as R
from neural_particle_method.suite.config import SuiteSettings
from neural_particle_method.suite.lag import LAGS
from neural_particle_method.suite.online import ONLINE_METHODS, run_online
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()
FAST = {"hidden": 16, "depth": 2, "lr": 1e-2, "batch_size": 0, "first_steps": 5,
        "later_steps": 2, "weight_decay": 0.0, "fit_subsample": 400, "warm_start": True,
        "mean_match": False, "monotone": False, "monotone_penalty": 0.0, "tail": "flat",
        "hetero": False}


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def test_method_table_is_the_agreed_one():
    assert list(ONLINE_METHODS) == ["explicit_stale", "implicit_stale", "explicit_rkhs",
                                    "implicit_rkhs", "explicit_ridge", "implicit_ridge",
                                    "explicit_spline", "implicit_spline",
                                    "explicit_tuned_stale", "explicit_tuned_rkhs",
                                    "explicit_tuned_ridge", "explicit_tuned_spline",
                                    "explicit_opt_stale", "explicit_opt_spline",
                                    "stale_L", "nw_resolve"]


@pytest.mark.parametrize("method", list(ONLINE_METHODS))
def test_every_method_runs_on_the_spot_lag(store, method, tmp_path, monkeypatch):
    if method.startswith("explicit_opt"):
        monkeypatch.setattr(R, "RECIPE_DIR", tmp_path)
        R.save_recipe("explicit_opt", FAST)
    n = TINY.offline_sizes[0]
    rid = run_online(store, "s01", method, n, LAGS[1], 0, TINY)
    p, m = store.get_params(rid), store.get_metrics(rid)
    assert p["method"] == method and p["lag"] == "surface_spot" and p["lag.kind"] == "surface_spot"
    assert float(p["scenario.s0"]) > 1.0 and "bumped.sigma0" in p
    assert m["pooled_mae_bp"] >= 0 and m["online_s"] >= 0
    if method in ("explicit_stale", "implicit_stale", "explicit_tuned_stale", "stale_L"):
        assert m["online_s"] == 0.0
    if method != "nw_resolve":
        assert p["offline_run"]
    if method.startswith("explicit_opt"):
        # the searched body is recipe-driven, so the run key names the promoted recipe
        assert p["recipe_hash"] == R.recipe_hash(R.coerce_recipe(FAST))
    else:
        assert "recipe_hash" not in p
    assert run_online(store, "s01", method, n, LAGS[1], 0, TINY) == rid


def test_offline_bodies_are_shared_across_lags_and_seeds(store):
    n = TINY.offline_sizes[0]
    a = run_online(store, "s01", "explicit_rkhs", n, LAGS[0], 0, TINY)
    b = run_online(store, "s01", "explicit_rkhs", n, LAGS[1], 0, TINY)
    assert a != b
    assert store.get_params(a)["offline_run"] == store.get_params(b)["offline_run"]
    assert len(store.search(TINY.experiment("suite_offline"))) == 1


def test_lagged_reference_gives_lev_rmse(store):
    n = TINY.offline_sizes[0]
    rid = run_online(store, "li_simple", "implicit_stale", n, LAGS[1], 0, TINY)
    assert "lev_rmse" in store.get_metrics(rid)
    refs = store.search("pde_reference")
    assert set(refs["params.lag"]) == {"surface_spot"}
