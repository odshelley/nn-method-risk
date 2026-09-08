import json

import pytest

from neural_particle_method.bench.algos import ALGOS, BASELINE_ALGOS, PAPER_ALGOS, run_algo
from neural_particle_method.bench.runner import run_one
from neural_particle_method.bench.scenarios import heston_registry, make_registry
from neural_particle_method.bench.sweep import BUDGETS, baselines_grid, sweep
from neural_particle_method.pricing.reprice import RepriceConfig
from neural_particle_method.simulate.leverage import LeverageField
from neural_particle_method.tracking.store import Store
from tests.conftest import TINY_EXPLICIT, TINY_IMPLICIT, TINY_N, TINY_REPRICE_N, TINY_REPRICE_STEPS

TINY_REPRICE = RepriceConfig(TINY_REPRICE_N, TINY_REPRICE_STEPS)


def test_registries():
    assert set(PAPER_ALGOS) | set(BASELINE_ALGOS) == set(ALGOS) and len(ALGOS) == 12
    assert BUDGETS == (1_000, 10_000, 100_000)
    jobs = baselines_grid()
    assert len(jobs) == 20 * 12 * 3 * 3 + 6 * 12 * 2 * 3
    assert jobs[0] == ("s01", "nw", 1_000, 0)


@pytest.mark.parametrize("name", BASELINE_ALGOS)
def test_each_baseline_runs_on_both_families(name):
    for sc in (make_registry()["s01"], heston_registry()["li_simple"]):
        res = run_algo(name, sc, 3_000, 0, TINY_EXPLICIT, TINY_IMPLICIT)
        assert len(res.field) == 6 and res.timings["total_s"] > 0


def test_knobs_reach_the_estimator_and_are_logged(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    a = run_one(store, "s01", "bins", TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT, TINY_REPRICE,
                experiment="sensitivity", knobs={"n_bins": 5}, extra_key={"knob_name": "n_bins", "knob_value": 5})
    b = run_one(store, "s01", "bins", TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT, TINY_REPRICE,
                experiment="sensitivity", knobs={"n_bins": 50}, extra_key={"knob_name": "n_bins", "knob_value": 50})
    assert a != b
    p = store.get_params(a)
    assert p["estimator.n_bins"] == "5" and p["knob_value"] == "5" and p["budget"] == str(TINY_N)
    assert store.get_metrics(a)["budget"] == TINY_N
    assert len(store.search("sensitivity")) == 2 and len(store.search("bench")) == 0


def test_lev_rmse_logged_when_pde_reference_exists(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    rid = run_one(store, "s01", "nw", TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT, TINY_REPRICE)
    lev = json.loads(store.download(rid, "leverage.json", tmp_path / "d").read_text())
    with store.run("pde_reference", {"sid": "s01"}) as h:      # a fake reference equal to the nw field
        h.log_json("leverage.json", lev)
    rid2 = run_one(store, "s01", "nw", TINY_N, 1, TINY_EXPLICIT, TINY_IMPLICIT, TINY_REPRICE)
    m = store.get_metrics(rid2)
    assert "lev_rmse" in m and m["lev_rmse"] >= 0.0
    ref = LeverageField.from_json(lev)
    assert any(k.startswith("lev_rmse/T") for k in m) and len([k for k in m if k.startswith("lev_rmse/T")]) == len(ref)


def test_lev_rmse_skipped_when_time_grids_differ(tmp_path, capsys):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    rid = run_one(store, "s01", "nw", TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT, TINY_REPRICE)
    lev = json.loads(store.download(rid, "leverage.json", tmp_path / "d").read_text())
    field = LeverageField.from_json(lev)
    coarse = LeverageField(list(field)[::2])   # every other slice: a coarser, misaligned time grid
    with store.run("pde_reference", {"sid": "s01", "n_steps": 3}) as h:
        h.log_json("leverage.json", coarse.to_json())
    rid2 = run_one(store, "s01", "nw", TINY_N, 1, TINY_EXPLICIT, TINY_IMPLICIT, TINY_REPRICE)
    m = store.get_metrics(rid2)
    assert "lev_rmse" not in m and not any(k.startswith("lev_rmse/T") for k in m)
    assert "different time grid" in capsys.readouterr().out


def test_sweep_into_baselines_experiment(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    jobs = [("s01", "bins", TINY_N, 0), ("s01", "nw_ghl", TINY_N, 0)]
    assert sweep(store, jobs, explicit=TINY_EXPLICIT, implicit=TINY_IMPLICIT, reprice=TINY_REPRICE, experiment="baselines") == 2
    assert len(store.search("baselines")) == 2 and len(store.search("bench")) == 0
