from pathlib import Path

import numpy as np
import pytest

from neural_particle_method.bench.scenarios import make_tuning_registry
from neural_particle_method.suite.config import TUNING_SIDS, SuiteSettings
from neural_particle_method.suite.optuna_clouds import (
    EXPERIMENT,
    Cloud,
    cloud_cache,
    ensure_cloud,
    heldout_loss,
    load_cloud,
    quantile_grid,
)
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()
TIMES = (0.5, 1.0)             # inside the tiny horizon (T=2, 4 steps), on its dt=0.5 grid


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def test_tuning_sids_are_the_registry():
    assert TUNING_SIDS == tuple(make_tuning_registry())


def test_heldout_loss_is_interpolated_mse():
    grid, f = np.array([0.0, 1.0]), np.array([0.0, 2.0])
    assert heldout_loss(grid, f, np.array([0.5, 2.0]), np.array([1.0, 3.0])) == pytest.approx(0.5)


def test_quantile_grid_is_sorted_unique_101_or_fewer():
    g = quantile_grid(np.random.default_rng(0).normal(size=5000))
    assert len(g) <= 101 and np.all(np.diff(g) > 0)


def test_ensure_cloud_is_idempotent_and_records_the_slices(store, tmp_path):
    reg = make_tuning_registry()
    rid = ensure_cloud(store, "t01", TINY, reg, times=TIMES)
    assert ensure_cloud(store, "t01", TINY, reg, times=TIMES) == rid
    assert len(store.search(EXPERIMENT)) == 1
    m = store.get_metrics(rid)
    assert {k for k in m if k.startswith("nw_loss/")} == {"nw_loss/t0.5", "nw_loss/t1"}
    c = load_cloud(store, rid, cache_dir=tmp_path / "cache")
    assert isinstance(c, Cloud) and c.sid == "t01" and c.times == [0.5, 1.0]
    n = TINY.offline_sizes[-1]
    for (lf, vf), (lh, vh) in zip(c.fit, c.held):
        assert len(lf) == len(vf) == n - int(0.2 * n) and len(lh) == len(vh) == int(0.2 * n)
        assert (vf >= 0).all() and (vh >= 0).all()
    assert c.rho == reg["t01"].dynamics.rho
    assert all(loss > 0 for loss in c.nw_loss)


def test_load_cloud_downloads_once_and_lands_the_file_atomically(store, tmp_path, monkeypatch):
    """Workers race on the same run id: a half-written `cloud.npz` must never be visible."""
    rid = ensure_cloud(store, "t01", TINY, make_tuning_registry(), times=TIMES)
    cache = tmp_path / "cache" / rid
    calls, real = [], store.download
    seen_partial = []

    def spy(run_id, path, dst):
        calls.append(run_id)
        got = real(run_id, path, dst)
        # while the download is in flight the destination is a temporary directory, not the cache
        seen_partial.append((cache / "cloud.npz").exists())
        return got

    monkeypatch.setattr(store, "download", spy)
    a = load_cloud(store, rid, cache_dir=cache)
    b = load_cloud(store, rid, cache_dir=cache)
    assert calls == [rid] and seen_partial == [False]
    assert (cache / "cloud.npz").exists() and not list(cache.parent.glob("tmp*"))
    assert a.times == b.times and a.sid == b.sid


def test_load_cloud_returns_float32(store, tmp_path):
    """The clouds are the worker's resident memory; float32 halves it and the fits cast anyway."""
    rid = ensure_cloud(store, "t01", TINY, make_tuning_registry(), times=TIMES)
    c = load_cloud(store, rid, cache_dir=tmp_path / "cache")
    for (lf, vf), (lh, vh) in zip(c.fit, c.held):
        assert lf.dtype == vf.dtype == lh.dtype == vh.dtype == np.float32


def test_cloud_cache_is_the_path_the_workers_read(tmp_path):
    assert cloud_cache("abc", tmp_path / "c") == tmp_path / "c"
    assert cloud_cache("abc") == Path("results/optuna/clouds/abc")
