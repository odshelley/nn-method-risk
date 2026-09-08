import json
from dataclasses import replace

from neural_particle_method.bench.reference_runs import run_reference
from neural_particle_method.bench.runner import run_one
from neural_particle_method.pricing.reprice import RepriceConfig
from neural_particle_method.simulate.leverage import LeverageField
from neural_particle_method.tracking.store import Store
from tests.conftest import TINY_EXPLICIT, TINY_IMPLICIT, TINY_N, TINY_REPRICE_N, TINY_REPRICE_STEPS

TINY_REPRICE = RepriceConfig(TINY_REPRICE_N, TINY_REPRICE_STEPS)


def test_run_reference_populates_store_and_is_idempotent(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    rid = run_reference(store, "li_simple", n_steps=6, n_x=41, n_v=20)
    assert rid

    field = LeverageField.from_json(json.loads(store.download(rid, "leverage.json", tmp_path / "d").read_text()))
    assert len(field) == 6

    rid2 = run_reference(store, "li_simple", n_steps=6, n_x=41, n_v=20)
    assert rid2 == rid
    assert len(store.search("pde_reference")) == 1


def test_run_one_picks_up_the_reference_and_logs_lev_rmse(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    run_reference(store, "li_simple", n_steps=6, n_x=41, n_v=20)

    explicit = replace(TINY_EXPLICIT, n_steps=6)
    rid = run_one(store, "li_simple", "nw", TINY_N, 0, explicit, TINY_IMPLICIT, TINY_REPRICE)
    m = store.get_metrics(rid)
    assert "lev_rmse" in m and m["lev_rmse"] >= 0.0


def test_reference_is_keyed_by_n_steps_and_lag(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    a = run_reference(store, "li_simple", n_steps=6, n_x=41, n_v=20)
    b = run_reference(store, "li_simple", n_steps=4, n_x=41, n_v=20)
    assert a != b
    assert store.get_params(a)["lag"] == "none"
    explicit = replace(TINY_EXPLICIT, n_steps=4)
    rid = run_one(store, "li_simple", "nw", TINY_N, 0, explicit, TINY_IMPLICIT, TINY_REPRICE)
    m = store.get_metrics(rid)
    assert "lev_rmse" in m            # matched the 4-step reference, not the 6-step one


def test_reference_accepts_an_explicit_scenario(tmp_path):
    from dataclasses import replace as dc_replace

    from neural_particle_method.bench.scenarios import full_registry
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    sc = dc_replace(full_registry()["li_simple"], s0=1.02)
    rid = run_reference(store, "li_simple", n_steps=6, n_x=41, n_v=20, scenario=sc, lag="surface_spot")
    p = store.get_params(rid)
    assert p["lag"] == "surface_spot" and p["s0"] == "1.02"
    assert run_reference(store, "li_simple", n_steps=6, n_x=41, n_v=20) != rid
