import pytest

from neural_particle_method.bench.aggregate import COLUMNS, aggregate
from neural_particle_method.tracking.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def _fake_run(store, algo, rmse, fail=False, n_particles=1000):
    key = {"sid": "s01", "algo": algo, "n_particles": n_particles, "seed": 0}
    if fail:
        with pytest.raises(RuntimeError), store.run("bench", {**key, "git_hash": "abc"}):
            raise RuntimeError("x")
        return
    with store.run("bench", {**key, "git_hash": "abc"}) as h:
        h.log_metrics({"pooled_rmse_bp": rmse, "pooled_max_bp": 30.0, "wings_rmse_bp": 20.0,
                       "wings_max_bp": 30.0, "n_failed": 0, "total_s": 1.0, "fit_s": 0.2})


def test_aggregate_writes_csv_and_digest(store, tmp_path):
    _fake_run(store, "nw", 12.0)
    _fake_run(store, "ridge", 8.0)
    _fake_run(store, "spline", 0.0, fail=True)
    df = aggregate(store, tmp_path / "s.csv", tmp_path / "d.md")
    assert list(df.columns) == list(COLUMNS) and len(df) == 3
    assert set(df.status) == {"ok", "failed"}
    md = (tmp_path / "d.md").read_text()
    assert "1 failed" in md and "s01: best = ridge" in md
    csv_text = (tmp_path / "s.csv").read_text()
    assert ",0.0," not in csv_text
    lines = {ln.split(",")[1]: ln for ln in csv_text.splitlines()[1:]}
    assert lines["nw"].split(",")[9] == "0"
    assert lines["spline"].split(",")[9] == ""


def test_aggregate_sorts_n_particles_numerically(store, tmp_path):
    _fake_run(store, "nw", 12.0, n_particles=200_000)
    _fake_run(store, "nw", 8.0, n_particles=50_000)
    df = aggregate(store, tmp_path / "s.csv", tmp_path / "d.md")
    assert list(df.n_particles) == [50_000, 200_000]
    rows = [ln for ln in (tmp_path / "s.csv").read_text().splitlines()[1:] if ln]
    assert [int(ln.split(",")[2]) for ln in rows] == [50_000, 200_000]


def test_aggregate_empty(store, tmp_path):
    df = aggregate(store, tmp_path / "s.csv", tmp_path / "d.md")
    assert len(df) == 0 and list(df.columns) == list(COLUMNS)
    assert "0 runs, 0 failed" in (tmp_path / "d.md").read_text()


def test_aggregate_running_run_is_not_counted_as_failed(store, tmp_path):
    _fake_run(store, "nw", 12.0)
    # a run left RUNNING (e.g. an interrupted sweep), never terminated
    store.client.create_run(store.experiment_id("bench"), tags={})
    df = aggregate(store, tmp_path / "s.csv", tmp_path / "d.md")
    assert "running" in set(df.status)
    assert (df.status == "failed").sum() == 0
    md = (tmp_path / "d.md").read_text()
    assert "0 failed" in md


def test_aggregate_other_experiment_and_new_columns(store, tmp_path):
    with store.run("baselines", {"sid": "s01", "algo": "bins", "n_particles": 1000, "seed": 0, "budget": 1000}) as h:
        h.log_metrics({"pooled_rmse_bp": 1.0, "budget": 1000})
    df = aggregate(store, tmp_path / "b.csv", tmp_path / "b.md", experiment="baselines")
    assert len(df) == 1 and df.loc[0, "budget"] == 1000 and "knob_name" in df.columns
    assert len(aggregate(store, tmp_path / "s.csv", tmp_path / "d.md")) == 0
