import json
import tempfile

import numpy as np
import pytest

from neural_particle_method.bench.reference_runs import run_reference
from neural_particle_method.estimators import recipes as R
from neural_particle_method.suite.config import SuiteSettings
from neural_particle_method.suite.lag import LAGS
from neural_particle_method.suite.tilt import (
    COLD_PARTICLES,
    SLICE_SIDS,
    cold_jobs,
    mixture_for,
    online_jobs,
    run_slice_cell,
    run_tilt_cold,
    run_tilt_online,
    run_tilt_stage,
    simulate_frozen,
    slice_jobs,
    tilt_jobs,
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
    # --algos splits the two-hour NW rows from the overnight network rows
    nw_only = cold_jobs("constant-3", algos=("nw",))
    assert len(nw_only) == 20 * 3 * 2 * 2 and {j[2] for j in nw_only} == {"nw"}
    assert len(cold_jobs("constant-3", algos=("explicit_nn_opt",))) == 6 * 3 * 2 * 2
    assert len(tilt_jobs("cold", design="constant-3", algos=("nw",))) == 240
    with pytest.raises(ValueError):
        cold_jobs("constant-3", algos=("spline",))
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


def test_online_job_count_and_shape(promoted):
    jobs = online_jobs("constant-3")
    assert len(jobs) == 20 * 2 * 2 * 2 and all(j[5] == "constant-3" for j in jobs)
    assert {j[2] for j in jobs} == {10_000, 80_000}
    assert len(online_jobs("constant-3", sids=("s01", "s02"))) == 16


def test_online_cell_is_tilted_keyed_and_weighted(store, promoted):
    rid = run_tilt_online(store, "s01", TINY.n_online, LAGS[1], 0, "inverse_sqrt-3", TINY,
                          recipe=FAST)
    p, m = store.get_params(rid), store.get_metrics(rid)
    assert p["design"] == "inverse_sqrt-3" and p["method"] == "explicit_opt_spline"
    assert p["recipe_hash"] == R.recipe_hash(FAST)
    assert 0 < m["ess_min_slice"] <= 1 and m["online_s"] > 0
    assert run_tilt_online(store, "s01", TINY.n_online, LAGS[1], 0, "inverse_sqrt-3", TINY,
                           recipe=FAST) == rid
    assert len(store.search(TINY.experiment("tilt_online"))) == 1
    # the untilted partner lives in the budget experiment, not here
    assert store.search(TINY.experiment("suite_budget_tuned")).empty


def test_slice_job_count():
    jobs = slice_jobs()
    assert len(jobs) == 6 * 3 * 7 * 5
    assert len({j[3] for j in jobs}) == 7 and "none" in {j[3] for j in jobs}
    assert len(slice_jobs(sids=("s11",))) == 3 * 7 * 5


def test_simulate_frozen_untilted_has_unit_weights_and_tilted_has_bounded_weights():
    from neural_particle_method.bench.scenarios import make_registry
    from neural_particle_method.simulate.leverage import DEFAULT_GRID, LeverageField, Slice
    sc = make_registry()["s01"]
    n_steps = 8
    field = LeverageField([Slice(k * sc.T / n_steps, DEFAULT_GRID.copy(),
                                 np.ones_like(DEFAULT_GRID), np.full_like(DEFAULT_GRID, 0.04))
                           for k in range(n_steps)])
    clouds, ess = simulate_frozen(field, sc.dynamics, sc.s0, sc.T, n_steps, 2_000, 0,
                                  keep_times=(0.5, 1.0))
    assert set(clouds) == {0.5, 1.0} and ess == []
    lnx, v, w = clouds[1.0]
    assert lnx.shape == v.shape == w.shape == (2_000,) and np.all(w == 1.0)
    mix = mixture_for("constant-3", n_steps, sc.T, sc.dynamics.rho)
    clouds_t, ess_t = simulate_frozen(field, sc.dynamics, sc.s0, sc.T, n_steps, 2_000, 0,
                                      mixture=mix, keep_times=(0.5, 1.0))
    _, _, wt = clouds_t[1.0]
    assert wt.max() <= 2 + 1e-9 and abs(wt.mean() - 1) < 0.1 and len(ess_t) == n_steps
    assert np.std(clouds_t[1.0][0]) > np.std(lnx)     # the tilt spreads the cloud


def test_slice_cell_logs_scores_and_clouds(store, promoted):
    run_reference(store, "s01", n_steps=TINY.explicit.n_steps, n_x=TINY.n_x, n_v=TINY.n_v)
    rid = run_slice_cell(store, "s01", 800, "constant-3", 0, TINY)
    assert run_slice_cell(store, "s01", 800, "constant-3", 0, TINY) == rid
    m = store.get_metrics(rid)
    for k in ("wings_abs_f_rel/nw", "wings_abs_f_rel/net", "ess_final", "ess_min_slice", "max_w",
              "sim_s", "fit_s/nw", "fit_s/net"):
        assert k in m
    with tempfile.TemporaryDirectory() as d:
        doc = json.loads(store.download(rid, "slice_scores.json", d).read_text())
        names = {a.path for a in store.client.list_artifacts(rid)}
    n_mat, n_k = len(doc["maturities"]), 13
    assert np.array(doc["f_ref"]).shape == (n_mat, n_k)
    assert np.array(doc["f_hat"]["nw"]).shape == (n_mat, n_k)
    assert np.array(doc["ess_local"]).shape == (n_mat, n_k) and len(doc["ess_slice"]) == n_mat
    assert len(doc["bandwidth"]) == n_mat and all(b > 0 for b in doc["bandwidth"])
    assert len(doc["ess_path"]) == TINY.explicit.n_steps     # the whole per-step ESS path
    assert all(f"cloud_T{t:g}.npz" in names for t in doc["maturities"])
    p = store.get_params(rid)
    assert p["design"] == "constant-3" and p["n_particles"] == "800"
    untilted = run_slice_cell(store, "s01", 800, "none", 0, TINY)
    assert store.get_metrics(untilted)["ess_final"] == 1.0
    with tempfile.TemporaryDirectory() as d:
        u = json.loads(store.download(untilted, "slice_scores.json", d).read_text())
    assert u["ess_path"] == []                               # no weights, no path


def test_stage_runner_skips_finished_cells(store, promoted):
    run_reference(store, "s01", n_steps=TINY.explicit.n_steps, n_x=TINY.n_x, n_v=TINY.n_v)
    done, failed = run_tilt_stage(store, "slices", TINY, sids=("s01",), particles=(800,),
                                  seeds=(0,), designs=("none", "constant-3"))
    assert (done, failed) == (2, 0)
    done, failed = run_tilt_stage(store, "slices", TINY, sids=("s01",), particles=(800,),
                                  seeds=(0,), designs=("none", "constant-3"))
    assert (done, failed) == (0, 0)
    done, failed = run_tilt_stage(store, "cold", TINY, sids=("s01",), design="constant-3",
                                  particles=(TINY.n_online,), seeds=(0,))
    assert (done, failed) == (4, 0)          # nw + explicit_nn_opt, untilted + tilted


def test_untilted_is_rejected_as_a_design_for_the_cold_and_online_stages():
    # a winner() of "none" must not reach the stages: cold would emit every cell twice and
    # every online cell would die in run_tilt_online
    with pytest.raises(ValueError):
        cold_jobs("none")
    with pytest.raises(ValueError):
        tilt_jobs("cold", design="none")
    with pytest.raises(ValueError):
        tilt_jobs("online", design="none")
    assert len(tilt_jobs("cold", sids=("s02",), design="constant-3")) == 2 * 3 * 2 * 2
