import json

import pytest

from neural_particle_method.bench import runner
from neural_particle_method.bench.runner import run_key, run_one
from neural_particle_method.pricing.reprice import RepriceConfig
from neural_particle_method.tracking.store import Store
from tests.conftest import TINY_EXPLICIT, TINY_IMPLICIT, TINY_N, TINY_REPRICE_N, TINY_REPRICE_STEPS

TINY_REPRICE = RepriceConfig(TINY_REPRICE_N, TINY_REPRICE_STEPS)


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def test_run_one_logs_params_metrics_artifacts(store, tmp_path):
    rid = run_one(store, "s01", "nw", TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT, TINY_REPRICE)
    p, m = store.get_params(rid), store.get_metrics(rid)
    assert p["sid"] == "s01" and p["explicit.n_steps"] == "6" and p["schema"] == "2" and p["git_hash"]
    assert m["pooled_rmse_bp"] >= 0 and m["rmse_bp/T0.25"] >= 0 and m["total_s"] > 0
    err = json.loads(store.download(rid, "iv_err_bp.json", tmp_path / "d").read_text())
    assert len(err) == 4 and len(err[0]) == 13
    lev = json.loads(store.download(rid, "leverage.json", tmp_path / "d").read_text())
    assert len(lev["slices"]) == 6


def test_run_one_is_resumable(store):
    a = run_one(store, "s01", "nw", TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT, TINY_REPRICE)
    b = run_one(store, "s01", "nw", TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT, TINY_REPRICE)
    assert a == b and len(store.search("bench")) == 1


def test_failure_recorded_then_raised(store, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(runner, "run_algo", boom)
    with pytest.raises(RuntimeError):
        run_one(store, "s01", "nw", TINY_N, 1, TINY_EXPLICIT, TINY_IMPLICIT, TINY_REPRICE)
    df = store.search("bench")
    assert df.loc[0, "status"] == "FAILED"
    assert run_key("s01", "nw", TINY_N, 1) == {"sid": "s01", "algo": "nw", "n_particles": TINY_N, "seed": 1}
