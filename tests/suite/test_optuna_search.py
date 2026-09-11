import numpy as np
import optuna
import pytest

from neural_particle_method.bench.scenarios import make_tuning_registry
from neural_particle_method.estimators.nadaraya_watson import nw_estimate
from neural_particle_method.suite.config import SuiteSettings
from neural_particle_method.suite.optuna_clouds import Cloud, heldout_loss, quantile_grid
from neural_particle_method.suite.optuna_search import (
    EXPERIMENT,
    run_study,
    score_recipe,
    suggest_recipe,
    top_recipes,
)
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()
TIMES = (0.5, 1.0)
FAST = {"hidden": 16, "depth": 2, "lr": 1e-2, "batch_size": 0, "first_steps": 5,
        "later_steps": 2, "weight_decay": 0.0, "fit_subsample": 400, "warm_start": True,
        "mean_match": False, "monotone": False, "monotone_penalty": 0.0, "tail": "flat",
        "hetero": False}


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def _synthetic_cloud():
    rng = np.random.default_rng(1)
    fit, held, nw_loss = [], [], []
    for _ in range(2):
        lnx = rng.normal(0, 0.3, 2000)
        v = np.maximum(0.04 * np.exp(-lnx) + 0.01 * rng.normal(size=2000), 0)
        (lf, vf), (lh, vh) = (lnx[:1600], v[:1600]), (lnx[1600:], v[1600:])
        g = quantile_grid(lf)
        fit.append((lf, vf)); held.append((lh, vh))
        nw_loss.append(heldout_loss(g, nw_estimate(lf, vf, g), lh, vh))
    return Cloud("x", -0.5, [0.5, 1.0], fit, held, nw_loss)


def test_nw_itself_scores_zero(monkeypatch):
    import neural_particle_method.suite.optuna_search as S
    cloud = _synthetic_cloud()

    class NWStub:
        def __init__(self):
            self.i = 0

        def fit_predict(self, t, lnx, v, grid, weights=None):
            return nw_estimate(lnx, v, grid)

        def predict(self, x):
            return np.zeros_like(x)

    monkeypatch.setattr(S, "regressor_from_recipe", lambda *a, **k: NWStub())
    score, detail, fit_s = score_recipe({**FAST, "fit_subsample": 1600}, [cloud], seed=0)
    assert score == pytest.approx(0.0, abs=1e-12)
    assert set(detail) == {"x"} and list(detail["x"]) == [0.5, 1.0]
    assert detail["x"][0.5] == pytest.approx(0.0, abs=1e-12)
    assert detail["x"][1.0] == pytest.approx(0.0, abs=1e-12)
    assert fit_s >= 0


def test_score_is_finite_for_a_real_recipe_and_reports_for_pruning():
    cloud = _synthetic_cloud()
    seen = []
    score, detail, _ = score_recipe(FAST, [cloud, cloud], seed=0,
                                    report=lambda i, m: seen.append(i))
    assert np.isfinite(score) and seen == [0, 1] and set(detail) == {"x"}


def test_suggest_recipe_covers_the_search_space():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    keys = set()
    for _ in range(20):
        t = study.ask()
        r = suggest_recipe(t)
        study.tell(t, 0.0)
        keys |= set(r)
        assert r["hidden"] in (32, 64, 128) and r["depth"] in (2, 3, 4)
        assert 1e-4 <= r["lr"] <= 1e-2 and r["batch_size"] in (2048, 8192, 0)
        assert 500 <= r["first_steps"] <= 4000 and r["first_steps"] % 250 == 0
        assert 100 <= r["later_steps"] <= 1500 and r["later_steps"] % 50 == 0
        assert r["fit_subsample"] in (100_000, 250_000, 400_000) and r["tail"] in ("free", "flat",
                                                                                  "linear")
        assert (r["monotone_penalty"] == 0.0) == (not r["monotone"])
    assert keys == set(FAST)


def test_run_study_logs_parent_and_children_and_resumes(store, tmp_path, monkeypatch):
    import neural_particle_method.suite.optuna_search as S
    monkeypatch.setattr(S, "suggest_recipe", lambda trial: {**FAST, "lr": trial.suggest_float(
        "lr", 1e-3, 1e-2, log=True)})
    reg = make_tuning_registry()
    sids = ("t01", "t02")
    kw = {"settings": TINY, "sids": sids, "per_trial": 2, "times": TIMES, "registry": reg,
          "storage_dir": tmp_path / "optuna", "cache_dir": tmp_path / "cache"}
    parent = run_study(store, "unit", n_trials=2, n_jobs=1, **kw)
    runs = store.search(EXPERIMENT)
    kids = runs[runs["tags.mlflow.parentRunId"] == parent]
    assert len(kids) == 2 and set(kids["tags.optuna.state"]) <= {"COMPLETE", "PRUNED"}
    assert {"metrics.score", "params.lr", "params.trial_number"} <= set(kids.columns)
    assert run_study(store, "unit", n_trials=1, n_jobs=1, **kw) == parent
    storage = f"sqlite:///{tmp_path / 'optuna' / 'unit.db'}"
    assert len(optuna.load_study(study_name="unit", storage=storage).trials) == 3
    before = store.client.get_run(parent).info.status
    top = top_recipes(store, "unit", 2)
    assert len(top) >= 1 and top[0]["score"] <= top[-1]["score"]
    assert set(top[0]) == {"trial_number", "score", "recipe", "run_id"}
    assert top[0]["recipe"]["hidden"] == 16
    assert store.client.get_run(parent).info.status == before == "FINISHED"


def test_the_worker_loads_the_study_with_the_spec_sampler_and_pruner(store, tmp_path, monkeypatch):
    """Optuna persists neither sampler nor pruner, so the worker must rebuild both on load."""
    import neural_particle_method.suite.optuna_search as S
    sampler, pruner = S._sampler_and_pruner()
    assert isinstance(sampler, optuna.samplers.TPESampler)
    assert pruner._n_startup_trials == 10 and pruner._n_warmup_steps == 1

    monkeypatch.setattr(S, "suggest_recipe", lambda trial: dict(FAST))
    seen, real = [], S._load_study

    def spy(study, storage, seed=0):
        seen.append(real(study, storage, seed))
        return seen[-1]

    monkeypatch.setattr(S, "_load_study", spy)
    run_study(store, "cfg", n_trials=1, n_jobs=1, settings=TINY, sids=("t01",), per_trial=1,
              times=TIMES, registry=make_tuning_registry(), storage_dir=tmp_path / "optuna",
              cache_dir=tmp_path / "cache")
    assert seen, "the worker did not open the study through _load_study"
    for st in seen:
        assert isinstance(st.sampler, optuna.samplers.TPESampler)
        assert st.pruner._n_startup_trials == 10 and st.pruner._n_warmup_steps == 1


def test_each_worker_gets_its_own_sampler_seed():
    """Workers share one storage, so an identical sampler seed duplicates the startup trials."""
    import neural_particle_method.suite.optuna_search as S

    def draw(seed):
        return S._sampler_and_pruner(seed)[0]._rng.rng.random_sample(4).tolist()

    assert draw(0) == draw(0), "the same seed must be reproducible"
    assert S._sampler_and_pruner()[0]._rng.rng.random_sample(4).tolist() == draw(0)
    assert draw(3) != draw(0) and draw(1) != draw(0)


def test_top_recipes_is_empty_and_writes_nothing_for_an_unknown_study(store):
    assert top_recipes(store, "never-run", 3) == []
    assert len(store.search(EXPERIMENT)) == 0


def test_top_recipes_is_empty_when_every_trial_was_pruned(store, tmp_path, monkeypatch):
    import neural_particle_method.suite.optuna_search as S

    def pruned(*a, **k):
        raise optuna.TrialPruned()

    monkeypatch.setattr(S, "suggest_recipe", lambda trial: dict(FAST))
    monkeypatch.setattr(S, "score_recipe", pruned)
    parent = run_study(store, "allpruned", n_trials=1, n_jobs=1, settings=TINY, sids=("t01",),
                       per_trial=1, times=TIMES, registry=make_tuning_registry(),
                       storage_dir=tmp_path / "optuna", cache_dir=tmp_path / "cache")
    runs = store.search(EXPERIMENT)
    kids = runs[runs["tags.mlflow.parentRunId"] == parent]
    assert len(kids) == 1 and list(kids["tags.optuna.state"]) == ["PRUNED"]
    assert top_recipes(store, "allpruned", 3) == []
