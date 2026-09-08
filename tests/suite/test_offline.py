import pytest

from neural_particle_method.suite.artifacts import GlobalNetModel, SliceBank, load_run
from neural_particle_method.suite.config import SuiteSettings
from neural_particle_method.suite.offline import run_offline
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


@pytest.mark.parametrize("body,cls", [("explicit", SliceBank), ("implicit", GlobalNetModel)])
def test_offline_body_is_cached_and_reloadable(store, body, cls):
    n = TINY.offline_sizes[0]
    rid = run_offline(store, "s01", body, n, TINY)
    p, m = store.get_params(rid), store.get_metrics(rid)
    assert p["body"] == body and p["n_particles"] == str(n) and p["seed"] == "0"
    assert m["fit_s"] >= 0 and m["pooled_mae_bp"] >= 0
    lr = load_run(store, rid)
    assert isinstance(lr.model, cls) and len(lr.field) == TINY.explicit.n_steps
    assert lr.meta["body"] == body and lr.meta["n_particles"] == n
    assert run_offline(store, "s01", body, n, TINY) == rid
    assert len(store.search(TINY.experiment("suite_offline"))) == 1


def test_unknown_body_rejected(store):
    with pytest.raises(ValueError):
        run_offline(store, "s01", "spline", 100, TINY)
