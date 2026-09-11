import json

import pytest

from neural_particle_method.calibrate.config import ExplicitConfig
from neural_particle_method.estimators import recipes as R

RECIPE = {"hidden": 32, "depth": 3, "lr": 1e-3, "batch_size": 0, "first_steps": 500,
          "later_steps": 100, "weight_decay": 1e-5, "fit_subsample": 250_000,
          "warm_start": False, "mean_match": True, "monotone": True, "monotone_penalty": 0.1,
          "tail": "linear", "hetero": True}


def test_coerce_from_mlflow_strings():
    raw = {k: str(v) for k, v in RECIPE.items()}
    assert R.coerce_recipe(raw) == RECIPE
    assert R.coerce_recipe({**raw, "monotone": "False"})["monotone_penalty"] == pytest.approx(0.1)
    without = {k: v for k, v in raw.items() if k != "monotone_penalty"}
    assert R.coerce_recipe(without)["monotone_penalty"] == 0.0


def test_hash_is_order_independent_and_short():
    h = R.recipe_hash(RECIPE)
    assert h == R.recipe_hash(dict(reversed(list(RECIPE.items())))) and len(h) == 10


def test_regressor_and_config_from_recipe():
    est = R.regressor_from_recipe(RECIPE, seed=3, monotone_sign=-1.0)
    assert (est.hidden, est.depth, est.lr, est.batch_size) == (32, 3, 1e-3, None)
    assert (est.first_steps, est.later_steps) == (500, 100)
    assert est.opt.param_groups[0]["weight_decay"] == pytest.approx(1e-5)
    assert est.warm_start is False and est.mean_match and est.hetero
    assert est.monotone_penalty == pytest.approx(0.1) and est.monotone_sign == -1.0
    off = R.regressor_from_recipe({**RECIPE, "monotone": False}, seed=3)
    assert off.monotone_penalty == 0.0
    cfg = R.explicit_config_from_recipe(ExplicitConfig(n_particles=100_000), RECIPE)
    assert (cfg.first_steps, cfg.later_steps, cfg.tail, cfg.fit_subsample) == (500, 100, "linear",
                                                                              100_000)


def test_save_load_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RECIPE_DIR", tmp_path)
    assert not R.recipe_exists("explicit_opt")
    p = R.save_recipe("explicit_opt", RECIPE)
    assert p == tmp_path / "explicit_opt.json" and json.loads(p.read_text()) == RECIPE
    assert R.recipe_exists("explicit_opt") and R.load_recipe("explicit_opt") == RECIPE
