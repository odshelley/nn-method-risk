import json
import numpy as np
from bench import runner
from bench.runner import run_one, run_path
from tests.conftest import TINY_EXPLICIT, TINY_IMPLICIT

def test_run_one_writes_valid_json(tmp_path):
    p = run_one("s01", "nw", 4_000, 0, results_dir=tmp_path,
                explicit=TINY_EXPLICIT, implicit=TINY_IMPLICIT,
                reprice_particles=8_000, reprice_steps=8)
    assert p == run_path("s01", "nw", 4_000, 0, tmp_path)
    d = json.loads(p.read_text())
    assert d["status"] == "ok" and d["schema"] == 1
    assert d["metrics"]["pooled_rmse_bp"] >= 0
    assert len(d["iv_err_bp"]) == 4 and len(d["iv_err_bp"][0]) == 13
    assert d["git_hash"]

def test_failure_writes_failed_json(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(runner, "run_algo", boom)
    p = run_one("s01", "nw", 4_000, 1, results_dir=tmp_path,
                explicit=TINY_EXPLICIT, implicit=TINY_IMPLICIT)
    d = json.loads(p.read_text())
    assert d["status"] == "failed" and "synthetic failure" in d["error"]
