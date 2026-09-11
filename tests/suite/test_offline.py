import numpy as np
import pytest

from neural_particle_method.bench.algos import TUNED_KNOBS, TUNED_STEPS
from neural_particle_method.suite.artifacts import GlobalNetModel, SliceBank, load_run
from neural_particle_method.suite.config import SuiteSettings
from neural_particle_method.suite.offline import BODIES, STAGE_BODIES, run_offline
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


@pytest.mark.parametrize("body,cls", [("explicit", SliceBank), ("implicit", GlobalNetModel)])
def test_offline_body_is_cached_and_reloadable(store, body, cls):
    n = TINY.offline_sizes[0]
    rid = run_offline(store, "s01", body, n, TINY)
    p, m = store.get_params(rid), store.get_metrics(rid)
    assert p["body"] == body and p["n_particles"] == str(n) and p["seed"] == "0"
    assert m["fit_s"] >= 0 and m["pooled_mae_bp"] >= 0
    lr = load_run(store, rid)
    assert isinstance(lr.model, cls) and len(lr.field) == TINY.explicit.n_steps
    assert lr.meta["body"] == body and lr.meta["n_particles"] == n
    assert run_offline(store, "s01", body, n, TINY) == rid
    assert len(store.search(TINY.experiment("suite_offline"))) == 1


def test_stage_bodies_are_the_three_agreed_ones():
    assert STAGE_BODIES == ("explicit", "explicit_tuned", "implicit")


def test_opt_body_uses_the_recipe_and_is_keyed_by_its_hash(store):
    from neural_particle_method.estimators.recipes import recipe_hash
    from neural_particle_method.suite.offline import STAGE_BODIES
    recipe = {"hidden": 16, "depth": 2, "lr": 1e-2, "batch_size": 0, "first_steps": 5,
              "later_steps": 2, "weight_decay": 0.0, "fit_subsample": 400, "warm_start": True,
              "mean_match": False, "monotone": True, "monotone_penalty": 0.01, "tail": "free",
              "hetero": False}
    n = TINY.offline_sizes[0]
    rid = run_offline(store, "s01", "explicit_opt", n, TINY, recipe=recipe)
    p = store.get_params(rid)
    assert p["recipe_hash"] == recipe_hash(recipe) and p["explicit.tail"] == "free"
    assert p["explicit.first_steps"] == "5" and p["recipe.monotone"] == "True"
    lr = load_run(store, rid)
    assert isinstance(lr.model, SliceBank) and len(lr.field) == TINY.explicit.n_steps
    assert run_offline(store, "s01", "explicit_opt", n, TINY, recipe=recipe) == rid
    assert BODIES == ("explicit", "explicit_tuned", "implicit", "explicit_opt")
    assert STAGE_BODIES == ("explicit", "explicit_tuned", "implicit")


def test_tuned_body_reloads_as_a_depth_three_bank(store):
    n = TINY.offline_sizes[0]
    rid = run_offline(store, "s01", "explicit_tuned", n, TINY)
    p = store.get_params(rid)
    assert p["body"] == "explicit_tuned"
    assert p["explicit.first_steps"] == str(TUNED_STEPS["first_steps"])
    assert p["explicit.later_steps"] == str(TUNED_STEPS["later_steps"])
    lr = load_run(store, rid)
    assert isinstance(lr.model, SliceBank)
    assert lr.model.state()["depth"] == TUNED_KNOBS["depth"] == 3
    x = np.linspace(-0.2, 0.2, 9)
    assert np.all(np.isfinite(lr.model.f(float(lr.model.times[0]), x)))
    assert run_offline(store, "s01", "explicit_tuned", n, TINY) == rid


def test_unknown_body_rejected(store):
    with pytest.raises(ValueError):
        run_offline(store, "s01", "spline", 100, TINY)
