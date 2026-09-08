import pytest

from neural_particle_method.bench.runner import run_one
from neural_particle_method.pricing.reprice import RepriceConfig
from neural_particle_method.tracking.store import Store
from tests.conftest import TINY_EXPLICIT, TINY_IMPLICIT, TINY_N, TINY_REPRICE_N, TINY_REPRICE_STEPS

TINY_REPRICE = RepriceConfig(TINY_REPRICE_N, TINY_REPRICE_STEPS)


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def test_save_model_hook_is_called_with_handle_and_result(store):
    seen = {}

    def hook(h, res):
        seen["run_id"], seen["model"] = h.run_id, res.model
        h.log_json("model_meta.json", {"ok": True})

    rid = run_one(store, "s01", "nw", TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT, TINY_REPRICE,
                  experiment="hooked", save_model=hook)
    assert seen["run_id"] == rid and seen["model"] is not None
    assert "model_meta.json" in [a.path for a in store.client.list_artifacts(rid)]
