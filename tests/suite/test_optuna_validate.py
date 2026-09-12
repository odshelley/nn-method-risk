import json

import pytest

from neural_particle_method.bench.scenarios import make_tuning_registry
from neural_particle_method.estimators import recipes as R
from neural_particle_method.suite.config import SuiteSettings
from neural_particle_method.suite.optuna_search import run_study
from neural_particle_method.suite.optuna_validate import promote, validate
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()
FAST = {"hidden": 16, "depth": 2, "lr": 1e-2, "batch_size": 0, "first_steps": 5,
        "later_steps": 2, "weight_decay": 0.0, "fit_subsample": 400, "warm_start": True,
        "mean_match": False, "monotone": False, "monotone_penalty": 0.0, "tail": "flat",
        "hetero": False}


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def test_validate_and_promote_on_tiny(store, tmp_path, monkeypatch):
    import neural_particle_method.suite.optuna_search as S
    monkeypatch.setattr(S, "suggest_recipe", lambda trial: {**FAST, "lr": trial.suggest_float(
        "lr", 1e-3, 1e-2, log=True)})
    monkeypatch.setattr(R, "RECIPE_DIR", tmp_path / "recipes")
    monkeypatch.chdir(tmp_path)
    run_study(store, "unit", n_trials=2, n_jobs=1, settings=TINY, sids=("t01",), per_trial=1,
              times=(0.5, 1.0), registry=make_tuning_registry(), storage_dir=tmp_path / "o",
              cache_dir=tmp_path / "c")
    df = validate(store, "unit", top=1, jobs=1, sids=("s01",), settings=TINY,
                  budget=TINY.n_online, seeds=(0,))
    assert set(df.columns) == {"mae_surface", "mae_surface_spot", "liquid_surface",
                               "liquid_surface_spot", "online_s"}
    assert "nw_resolve" in df.index and len(df) == 2
    top = json.loads((tmp_path / "results" / "optuna" / "unit_top.json").read_text())
    assert len(top) == 1 and set(top[0]) == {"trial_number", "score", "recipe"}
    bodies = store.search(TINY.experiment("suite_offline"))
    assert (bodies["params.body"] == "explicit_opt").sum() == 1
    cells = store.search(TINY.experiment("suite_budget_tuned"))
    assert set(cells["params.method"]) == {"explicit_opt_spline", "nw_resolve"}
    p = promote(store, "unit", top[0]["trial_number"])
    assert p == tmp_path / "recipes" / "explicit_opt.json"
    assert R.load_recipe("explicit_opt")["hidden"] == 16


def test_validate_refuses_a_study_with_no_complete_trials(store, tmp_path, monkeypatch):
    """Silently validating nothing prints a table with only the NW row, which reads as success."""
    import optuna

    import neural_particle_method.suite.optuna_search as S
    monkeypatch.setattr(S, "suggest_recipe", lambda trial: dict(FAST))
    monkeypatch.setattr(S, "score_recipe", lambda *a, **k: (_ for _ in ()).throw(
        optuna.TrialPruned()))
    monkeypatch.chdir(tmp_path)
    run_study(store, "empty", n_trials=1, n_jobs=1, settings=TINY, sids=("t01",), per_trial=1,
              times=(0.5, 1.0), registry=make_tuning_registry(), storage_dir=tmp_path / "o",
              cache_dir=tmp_path / "c")
    with pytest.raises(RuntimeError, match="study empty has no COMPLETE trials"):
        validate(store, "empty", top=1, jobs=1, sids=("s01",), settings=TINY,
                 budget=TINY.n_online, seeds=(0,))
    assert not (tmp_path / "results" / "optuna" / "empty_top.json").exists()
    assert store.search(TINY.experiment("suite_budget_tuned")).empty


def test_validate_refuses_a_study_that_was_never_run(store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="study ghost has no COMPLETE trials"):
        validate(store, "ghost", top=1, jobs=1, sids=("s01",), settings=TINY)


def test_only_the_spawned_validation_cell_pins_torch_threads(monkeypatch):
    """`_run_all(jobs, 1)` runs in this process; pinning BLAS there would change every later
    torch reduction the process makes, which is how a golden replay drifts."""
    import torch

    import neural_particle_method.suite.optuna_validate as V
    calls = []
    monkeypatch.setattr(V, "_cell", calls.append)
    before = torch.get_num_threads()
    V._run_all(["a", "b"], 1)
    assert calls == ["a", "b"] and torch.get_num_threads() == before
    try:
        V._pinned_cell("c")          # the pool's entry point is the one that pins
        assert calls == ["a", "b", "c"] and torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(before)
