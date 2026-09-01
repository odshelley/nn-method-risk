import json
from bench.aggregate import aggregate

OK = {"schema": 1, "sid": "s01", "algo": "nw", "n_particles": 1000, "seed": 0,
      "git_hash": "abc", "status": "ok", "timings": {"total_s": 1.0, "fit_s": 0.2},
      "diagnostics": {}, "metrics": {"pooled_rmse_bp": 12.0, "pooled_max_bp": 30.0,
      "wings_rmse_bp": 20.0, "wings_max_bp": 30.0, "n_failed": 0, "per_maturity": []}}
BAD = {"schema": 1, "sid": "s01", "algo": "ridge", "n_particles": 1000, "seed": 0,
       "git_hash": "abc", "status": "failed", "error": "boom"}

def test_aggregate(tmp_path):
    d = tmp_path / "s01" / "nw"; d.mkdir(parents=True)
    (d / "n1000_s0.json").write_text(json.dumps(OK))
    d2 = tmp_path / "s01" / "ridge"; d2.mkdir(parents=True)
    (d2 / "n1000_s0.json").write_text(json.dumps(BAD))
    df = aggregate(tmp_path, tmp_path / "summary.csv", tmp_path / "digest.md")
    assert len(df) == 2 and set(df.status) == {"ok", "failed"}
    assert (tmp_path / "summary.csv").exists()
    md = (tmp_path / "digest.md").read_text()
    assert "1 failed" in md and "s01" in md
