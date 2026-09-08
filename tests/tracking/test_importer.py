import json

import pytest

from neural_particle_method.tracking.importer import _rel, import_all, import_bench, import_warm
from neural_particle_method.tracking.store import Store, repo_root

OK = {"schema": 1, "sid": "s01", "algo": "nw", "n_particles": 1000, "seed": 0, "git_hash": "abc",
      "scenario": {"ssvi": {"sigma0": 0.2, "eta": 1.0, "gamma": 0.4, "rho": -0.6},
                   "dynamics": {"kappa": 1.0, "theta": 0.04, "xi": 0.3, "rho": -0.7, "v0": 0.04},
                   "s0": 1.0, "T": 2.0, "maturities": [0.25, 0.5, 1.0, 2.0]},
      "status": "ok", "timings": {"total_s": 1.0, "fit_s": 0.2}, "diagnostics": {"intraday_s": 0.5},
      "metrics": {"pooled_rmse_bp": 12.0, "pooled_max_bp": 30.0, "wings_rmse_bp": 20.0, "wings_max_bp": 30.0,
                  "n_failed": 0, "per_maturity": [{"T": 0.25, "rmse_bp": 3.0, "max_bp": 5.0}]},
      "iv_err_bp": [[1.0, None]]}
BAD = {"schema": 1, "sid": "s01", "algo": "ridge", "n_particles": 1000, "seed": 0, "git_hash": "abc",
       "status": "failed", "error": "Traceback: boom"}
SEQ = {"arm": "seq", "sid": "s01", "seed": 0, "wall_s": 9.5,
       "rmse_bp": {"steps": [{"refresh": 10.0, "ridge": 8.0}, {"refresh": 11.0, "ridge": 7.0}],
                   "resolve_cold_final": 5.0, "path": []},
       "build_s": {"overnight": 100.0, "step_0": {"refresh": 0.1, "ridge": 2.0},
                   "step_1": {"refresh": 0.1, "ridge": 2.1}, "resolve_cold_final": 50.0}}


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def _bench_dir(tmp_path):
    d = tmp_path / "runs" / "s01" / "nw"; d.mkdir(parents=True)
    (d / "n1000_s0.json").write_text(json.dumps(OK))
    d2 = tmp_path / "runs" / "s01" / "ridge"; d2.mkdir(parents=True)
    (d2 / "n1000_s0.json").write_text(json.dumps(BAD))
    return tmp_path / "runs"


def test_import_bench_creates_runs_and_is_idempotent(store, tmp_path):
    runs = _bench_dir(tmp_path)
    assert import_bench(store, runs) == 2
    assert import_bench(store, runs) == 0
    df = store.search("bench")
    assert set(df.status) == {"FINISHED", "FAILED"} and set(df["tags.source"]) == {"legacy_json"}
    ok = df[df.status == "FINISHED"].iloc[0]
    assert ok["metrics.pooled_rmse_bp"] == 12.0 and ok["metrics.rmse_bp/T0.25"] == 3.0
    assert ok["metrics.intraday_s"] == 0.5 and ok["params.git_hash"] == "abc"
    rid = ok["run_id"]
    err = json.loads(store.download(rid, "iv_err_bp.json", tmp_path / "d").read_text())
    assert err == [[1.0, None]]


def test_import_warm_seq_logs_steps(store, tmp_path):
    d = tmp_path / "warm"; d.mkdir()
    (d / "seq_s01_s0.json").write_text(json.dumps(SEQ))
    assert import_warm(store, d) == 1
    df = store.search("warm")
    assert df.loc[0, "params.arm"] == "seq"
    assert df.loc[0, "metrics.rmse_bp/refresh"] == 11.0          # last step wins as "latest"
    assert df.loc[0, "metrics.rmse_bp/resolve_cold_final"] == 5.0 and df.loc[0, "metrics.wall_s"] == 9.5


def test_import_all_reports_counts(store, tmp_path):
    _bench_dir(tmp_path)
    (tmp_path / "bump").mkdir(); (tmp_path / "warm").mkdir()
    counts = import_all(store, tmp_path)
    assert counts == {"bench": 2, "bump": 0, "warm": 0, "overnight": 0}


def test_import_warm_routes_smoke_files_to_separate_experiment(store, tmp_path):
    d = tmp_path / "warm"; d.mkdir()
    (d / "seq_s01_s0.json").write_text(json.dumps(SEQ))
    (d / "smoke_seq_s01_s0.json").write_text(json.dumps(SEQ))
    assert import_warm(store, d) == 2

    warm = store.search("warm")
    smoke = store.search("warm_smoke")
    assert len(warm) == 1
    assert len(smoke) == 1
    assert smoke.loc[0, "tags.smoke"] == "true"

    assert import_warm(store, d) == 0


def test_rel_is_independent_of_cwd(monkeypatch, tmp_path):
    p = repo_root() / "results" / "runs" / "s01" / "nw" / "n1000_s0.json"
    expected = _rel(p)
    monkeypatch.chdir(tmp_path)
    assert _rel(p) == expected
