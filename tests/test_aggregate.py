import json
from bench.aggregate import aggregate, COLUMNS

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


def test_aggregate_empty_runs_dir(tmp_path):
    empty = tmp_path / "empty"; empty.mkdir()
    df = aggregate(empty, tmp_path / "summary_empty.csv", tmp_path / "digest_empty.md")
    assert len(df) == 0
    assert list(df.columns) == list(COLUMNS)
    assert (tmp_path / "summary_empty.csv").exists()
    md = (tmp_path / "digest_empty.md").read_text()
    assert "0 runs, 0 failed" in md


def test_aggregate_skips_unreadable_json(tmp_path):
    d = tmp_path / "s01" / "nw"; d.mkdir(parents=True)
    (d / "n1000_s0.json").write_text(json.dumps(OK))
    (d / "n1000_s1.json.tmp").write_text("")  # not *.json, shouldn't even be picked up
    (d / "n2000_s0.json").write_text("{truncated garbage, not valid json")
    df = aggregate(tmp_path, tmp_path / "summary.csv", tmp_path / "digest.md")
    assert len(df) == 1
    assert (tmp_path / "summary.csv").exists()
    md = (tmp_path / "digest.md").read_text()
    assert "1 unreadable" in md


def test_aggregate_out_md_dir_is_created(tmp_path, monkeypatch):
    # out_csv and out_md land under two different, not-yet-existing directories;
    # aggregate() must mkdir both parents rather than relying on the cwd having results/.
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    d = runs / "s01" / "nw"; d.mkdir(parents=True)
    (d / "n1000_s0.json").write_text(json.dumps(OK))
    csv_dir = tmp_path / "csv_out"
    md_dir = tmp_path / "md_out"
    assert not csv_dir.exists() and not md_dir.exists()
    df = aggregate(runs, csv_dir / "summary.csv", md_dir / "digest.md")
    assert len(df) == 1
    assert (csv_dir / "summary.csv").exists()
    assert (md_dir / "digest.md").exists()
