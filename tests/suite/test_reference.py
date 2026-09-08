from neural_particle_method.suite.config import SuiteSettings
from neural_particle_method.suite.reference import run_pde_floor
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()


def test_pde_floor_reprices_the_reference_and_is_idempotent(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    rid = run_pde_floor(store, "li_simple", 0, TINY)
    p, m = store.get_params(rid), store.get_metrics(rid)
    assert (p["sid"] == "li_simple" and p["n_steps"] == str(TINY.explicit.n_steps)
           and p["seed"] == "0")
    assert m["pooled_mae_bp"] >= 0 and m["fit_s"] == 0.0 and m["lev_rmse"] == 0.0
    assert run_pde_floor(store, "li_simple", 0, TINY) == rid
    assert len(store.search("pde_reference")) == 1
    assert run_pde_floor(store, "li_simple", 1, TINY) != rid
