import pytest

from neural_particle_method.suite.budget import (
    EXPERIMENTS,
    budget_jobs,
    budget_methods,
    head_for,
    run_budget_cell,
)
from neural_particle_method.suite.config import SuiteSettings
from neural_particle_method.suite.heads import SplineHead
from neural_particle_method.suite.lag import LAGS
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()
FAST = {"hidden": 16, "depth": 2, "lr": 1e-2, "batch_size": 0, "first_steps": 5,
        "later_steps": 2, "weight_decay": 0.0, "fit_subsample": 400, "warm_start": True,
        "mean_match": True, "monotone": False, "monotone_penalty": 0.0, "tail": "linear",
        "hetero": False}


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def test_methods_and_heads():
    assert budget_methods("explicit_opt") == ("nw_resolve", "explicit_opt_stale",
                                              "explicit_opt_rkhs", "explicit_opt_spline",
                                              "explicit_opt_ridge")
    assert isinstance(head_for("explicit_opt_spline"), SplineHead)
    assert head_for("nw_resolve") is None and head_for("explicit_opt_stale") is None
    assert len(budget_jobs(("s01",), "explicit_tuned")) == 60
    assert EXPERIMENTS["explicit_opt"] == "suite_budget_tuned"


def test_opt_cell_runs_on_tiny_and_is_keyed_by_recipe(store):
    rid = run_budget_cell(store, "s01", "explicit_opt_spline", TINY.n_online, LAGS[0], 0,
                          body="explicit_opt", settings=TINY, recipe=FAST)
    p = store.get_params(rid)
    assert p["body"] == "explicit_opt" and len(p["recipe_hash"]) == 10
    assert p["recipe.tail"] == "linear"
    assert run_budget_cell(store, "s01", "explicit_opt_spline", TINY.n_online, LAGS[0], 0,
                           body="explicit_opt", settings=TINY, recipe=FAST) == rid
    other = run_budget_cell(store, "s01", "explicit_opt_spline", TINY.n_online, LAGS[0], 0,
                            body="explicit_opt", settings=TINY, recipe={**FAST, "tail": "flat"})
    assert other != rid
    assert store.get_metrics(rid)["liquid_mae_bp"] >= 0
    assert len(store.search(TINY.experiment("suite_budget_tuned"))) == 2


def test_nw_resolve_is_one_body_free_cell_shared_by_every_body(store):
    rid = run_budget_cell(store, "s01", "nw_resolve", TINY.n_online, LAGS[0], 0,
                          body="explicit_opt", settings=TINY, recipe=FAST)
    p = store.get_params(rid)
    assert "recipe_hash" not in p and not any(k.startswith("recipe.") for k in p)
    assert p["body"] == "explicit_tuned" and p["body_run"] == ""
    # no body of any kind was trained for it
    assert store.search(TINY.experiment("suite_offline")).empty
    assert run_budget_cell(store, "s01", "nw_resolve", TINY.n_online, LAGS[0], 0,
                           body="explicit_tuned", settings=TINY) == rid
    assert run_budget_cell(store, "s01", "nw_resolve", TINY.n_online, LAGS[0], 0,
                           body="explicit", settings=TINY) == rid
    assert len(store.search(TINY.experiment("suite_budget_tuned"))) == 1
