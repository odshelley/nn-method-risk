import json

import numpy as np
import pytest

from neural_particle_method.tracking.store import Store, flatten_metrics, to_jsonable


@pytest.fixture
def store(tmp_path):
    return Store(tracking_uri=f"sqlite:///{tmp_path / 'mlruns.db'}", artifact_root=str(tmp_path / "art"))


def test_log_and_query_round_trip(store, tmp_path):
    with store.run("bench", {"sid": "s01", "algo": "nw", "n_particles": 1000, "seed": 0}) as h:
        h.log_metrics({"pooled_rmse_bp": 12.5, "rmse_bp/T0.25": 3.0})
        h.log_json("leverage.json", {"slices": [{"t": 0.0}]})
        rid = h.run_id
    df = store.search("bench")
    assert len(df) == 1 and df.loc[0, "status"] == "FINISHED"
    assert df.loc[0, "params.algo"] == "nw" and df.loc[0, "metrics.pooled_rmse_bp"] == 12.5
    assert store.get_metrics(rid)["rmse_bp/T0.25"] == 3.0
    p = store.download(rid, "leverage.json", tmp_path / "dl")
    assert json.loads(p.read_text())["slices"][0]["t"] == 0.0


def test_find_finished_hits_on_identical_params_only(store):
    key = {"sid": "s01", "algo": "nw", "n_particles": 1000, "seed": 0}
    assert store.find_finished("bench", key) is None
    with store.run("bench", key) as h:
        rid = h.run_id
    assert store.find_finished("bench", key) == rid
    assert store.find_finished("bench", {**key, "seed": 1}) is None


def test_failed_run_is_recorded_and_does_not_block(store):
    key = {"sid": "s01", "algo": "nw", "n_particles": 1000, "seed": 0}
    with pytest.raises(RuntimeError, match="boom"), store.run("bench", key):
        raise RuntimeError("boom")
    df = store.search("bench")
    assert df.loc[0, "status"] == "FAILED"
    assert store.find_finished("bench", key) is None


def test_metrics_with_step(store):
    with store.run("warm", {"arm": "seq", "sid": "s01", "seed": 0}) as h:
        for j in range(3):
            h.log_metrics({"rmse_bp/refresh": float(j)}, step=j)
        rid = h.run_id
    assert store.get_metrics(rid)["rmse_bp/refresh"] == 2.0     # latest value


def test_to_jsonable_and_flatten():
    assert to_jsonable({"a": np.float64(1.5), "b": [np.int64(2)], "c": float("nan")}) == {"a": 1.5, "b": [2], "c": None}
    assert flatten_metrics("rmse_bp", {"x": {"y": 1.0}, "z": 2, "s": "skip"}) == {"rmse_bp/x/y": 1.0, "rmse_bp/z": 2.0}
