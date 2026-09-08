from neural_particle_method.bench.sensitivity import KNOBS, run_sensitivity, sensitivity_grid
from neural_particle_method.pricing.reprice import RepriceConfig
from neural_particle_method.tracking.store import Store
from tests.conftest import TINY_EXPLICIT, TINY_IMPLICIT, TINY_N, TINY_REPRICE_N, TINY_REPRICE_STEPS

TINY_REPRICE = RepriceConfig(TINY_REPRICE_N, TINY_REPRICE_STEPS)


def test_knob_table_matches_spec():
    assert KNOBS["nw_ghl"] == ("c", (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0))
    assert KNOBS["rkhs"][0] == "lam" and KNOBS["rkhs"][1] == tuple(10.0 ** -k for k in range(9, 1, -1))
    assert KNOBS["bins"] == ("n_bins", (5, 10, 20, 50, 100, 200, 500))
    assert KNOBS["purbf"] == ("prune", (0.5, 1.0, 2.0))
    assert KNOBS["explicit_nn"] == ("hidden", (8, 16, 32, 64, 128))
    assert KNOBS["ridge"] == ("lam", (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1))
    assert KNOBS["spline"] == ("lam", (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0))
    assert KNOBS["nw"] == ("bandwidth_scale", (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0))
    assert "muguruza" not in KNOBS


def test_grid_enumeration():
    jobs = sensitivity_grid(sids=("s01",), budgets=(1_000,), seeds=(0,), algos=("bins", "purbf"))
    assert len(jobs) == 7 + 3 and jobs[0] == ("s01", "bins", 1_000, 0, "n_bins", 5)


def test_run_is_resumable_and_logs_knob(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    jobs = sensitivity_grid(sids=("s01",), budgets=(TINY_N,), seeds=(0,), algos=("bins",))[:2]
    assert run_sensitivity(store, jobs, explicit=TINY_EXPLICIT, implicit=TINY_IMPLICIT, reprice=TINY_REPRICE) == 2
    assert run_sensitivity(store, jobs, explicit=TINY_EXPLICIT, implicit=TINY_IMPLICIT, reprice=TINY_REPRICE) == 0
    df = store.search("sensitivity")
    assert sorted(df["params.knob_value"].astype(int)) == [5, 10] and set(df["metrics.knob_value"]) == {5.0, 10.0}
