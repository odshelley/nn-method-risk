import time

import numpy as np
import optuna
import pytest

import neural_particle_method.suite.optuna_search as S_
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


def test_full_batch_is_pinned_to_the_smallest_fit_sample():
    """Full batch times the largest sample is the trial cost the study cannot afford."""
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    full, mini = 0, 0
    for _ in range(60):
        t = study.ask()
        r = suggest_recipe(t)
        study.tell(t, 0.0)
        if r["batch_size"] == 0:
            full += 1
            assert r["fit_subsample"] == 100_000
            assert "fit_subsample" not in t.params      # not suggested at all, so TPE ignores it
        else:
            mini += 1
            assert r["fit_subsample"] in (100_000, 250_000, 400_000)
    assert full > 0 and mini > 0


def test_score_recipe_seeds_each_scenario_by_its_own_seed():
    """A scenario's fit seed must be the same in every trial, so trials differ only by recipe."""
    cloud = _synthetic_cloud()
    seen = []
    score, _, _ = score_recipe(FAST, [cloud, cloud], seeds=[5, 5],
                               report=lambda i, m: seen.append(m))
    assert seen[0] == pytest.approx(seen[1]) == pytest.approx(score)
    other = []
    score_recipe(FAST, [cloud, cloud], seeds=[5, 6], report=lambda i, m: other.append(m))
    assert other[0] != other[1]


def test_score_recipe_stops_at_its_deadline():
    cloud = _synthetic_cloud()
    with pytest.raises(S_.TrialTimeout) as e:
        score_recipe(FAST, [cloud, cloud, cloud], seed=0, deadline=time.perf_counter() - 1.0)
    assert np.isfinite(e.value.score_partial) and e.value.fit_s >= 0
    assert isinstance(e.value, optuna.TrialPruned)     # optimize must not re-raise it


def test_a_trial_past_its_wall_clock_cap_is_a_timeout(store, tmp_path, monkeypatch):
    """The cap must end the trial inside `run_study`, tagged TIMEOUT with its partial score."""
    monkeypatch.setattr(S_, "suggest_recipe", lambda trial: dict(FAST))

    def slow(recipe, cloud, seed):
        time.sleep(0.05)
        return [0.25, 0.25]

    monkeypatch.setattr(S_, "fit_chain", slow)
    parent = run_study(store, "slow", n_trials=1, n_jobs=1, settings=TINY, sids=("t01",),
                       per_trial=1, times=TIMES, registry=make_tuning_registry(),
                       storage_dir=tmp_path / "optuna", cache_dir=tmp_path / "cache",
                       trial_timeout_s=0.01)
    runs = store.search(EXPERIMENT)
    kids = runs[runs["tags.mlflow.parentRunId"] == parent]
    assert list(kids["tags.optuna.state"]) == ["TIMEOUT"]
    assert list(kids["status"]) == ["FINISHED"]
    assert float(kids["metrics.score_partial"].iloc[0]) == pytest.approx(0.25)
    assert float(kids["metrics.fit_s"].iloc[0]) > 0
    assert store.get_metrics(parent)["n_timeout"] == 1
    assert top_recipes(store, "slow", 3) == []


def test_a_failing_trial_tags_the_child_and_counts_on_the_parent(store, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise ValueError("no")

    monkeypatch.setattr(S_, "suggest_recipe", lambda trial: dict(FAST))
    monkeypatch.setattr(S_, "score_recipe", boom)
    parent = run_study(store, "bad", n_trials=1, n_jobs=1, settings=TINY, sids=("t01",),
                       per_trial=1, times=TIMES, registry=make_tuning_registry(),
                       storage_dir=tmp_path / "optuna", cache_dir=tmp_path / "cache")
    runs = store.search(EXPERIMENT)
    kids = runs[runs["tags.mlflow.parentRunId"] == parent]
    assert list(kids["tags.optuna.state"]) == ["FAIL"] and list(kids["status"]) == ["FAILED"]
    m = store.get_metrics(parent)
    assert m["n_failed"] == 1 and m["n_timeout"] == 0


def test_a_trial_whose_params_cannot_be_logged_is_a_fail_not_an_orphan(store, tmp_path,
                                                                      monkeypatch):
    """The child's params are written inside the try, so a store failure tags it FAIL."""
    from mlflow.tracking import MlflowClient

    from neural_particle_method.suite.optuna_clouds import ensure_cloud
    reg = make_tuning_registry()
    ensure_cloud(store, "t01", TINY, reg, TIMES)     # before the patch: the cloud logs params too
    monkeypatch.setattr(S_, "suggest_recipe", lambda trial: dict(FAST))
    real = MlflowClient.log_param

    def flaky(self, rid, key, value, *a, **kw):
        if key == "tail":
            raise RuntimeError("store says no")
        return real(self, rid, key, value, *a, **kw)

    monkeypatch.setattr(MlflowClient, "log_param", flaky)
    parent = run_study(store, "flaky", n_trials=1, n_jobs=1, settings=TINY, sids=("t01",),
                       per_trial=1, times=TIMES, registry=reg,
                       storage_dir=tmp_path / "optuna", cache_dir=tmp_path / "cache")
    runs = store.search(EXPERIMENT)
    kids = runs[runs["tags.mlflow.parentRunId"] == parent]
    assert list(kids["tags.optuna.state"]) == ["FAIL"] and list(kids["status"]) == ["FAILED"]


def test_no_trials_is_a_no_op_that_still_warms_the_cloud_cache(store, tmp_path):
    """Workers must never be the first to download a cloud: they would race on the same file."""
    cache = tmp_path / "cache"
    parent = run_study(store, "none", n_trials=0, n_jobs=4, settings=TINY, sids=("t01", "t02"),
                       per_trial=2, times=TIMES, registry=make_tuning_registry(),
                       storage_dir=tmp_path / "optuna", cache_dir=cache)
    cached = list(cache.glob("*/cloud.npz"))
    assert len(cached) == 2 and all(p.stat().st_size > 0 for p in cached)
    assert list(store.search(EXPERIMENT)["run_id"]) == [parent]     # the parent and nothing else
    assert store.get_metrics(parent)["n_trials"] == 0


def test_two_workers_share_the_study_without_tripping_over_each_other(store, tmp_path,
                                                                     monkeypatch):
    """Spawned workers re-import the module, so the recipe override has to be an env variable."""
    monkeypatch.setenv(S_.FAST_RECIPE_ENV, "1")
    parent = run_study(store, "par", n_trials=2, n_jobs=2, settings=TINY, sids=("t01",),
                       per_trial=1, times=TIMES, registry=make_tuning_registry(),
                       storage_dir=tmp_path / "optuna", cache_dir=tmp_path / "cache")
    runs = store.search(EXPERIMENT)
    kids = runs[runs["tags.mlflow.parentRunId"] == parent]
    assert len(kids) == 2 and set(kids["tags.optuna.state"]) == {"COMPLETE"}
    m = store.get_metrics(parent)
    assert m["n_trials"] == 2 and m["n_failed"] == 0 and m["n_timeout"] == 0


def test_a_single_job_study_does_not_repin_the_caller_torch_threads(store, tmp_path, monkeypatch):
    """`n_jobs=1` runs the worker in this process; pinning BLAS there changes every later
    reduction the process makes, which is how a golden replay drifts by 1e-2 bp."""
    import torch
    monkeypatch.setattr(S_, "suggest_recipe", lambda trial: dict(FAST))
    before = torch.get_num_threads()
    run_study(store, "threads", n_trials=1, n_jobs=1, settings=TINY, sids=("t01",), per_trial=1,
              times=TIMES, registry=make_tuning_registry(), storage_dir=tmp_path / "optuna",
              cache_dir=tmp_path / "cache")
    assert torch.get_num_threads() == before
