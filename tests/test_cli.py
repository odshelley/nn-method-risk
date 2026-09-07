import subprocess
import json

def test_bench_list_runs():
    r = subprocess.run(["uv", "run", "bench", "list"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "scenarios" in r.stdout.lower()


def test_bench_run_and_sweep_skip(tmp_path):
    args = ["uv", "run", "bench", "run", "--scenario", "s01", "--algo", "nw",
            "--n", "3000", "--seed", "0", "--n-steps", "6",
            "--reprice-n", "5000", "--reprice-steps", "8",
            "--results-dir", str(tmp_path)]
    r = subprocess.run(args, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = tmp_path / "s01" / "nw" / "n3000_s0.json"
    assert json.loads(out.read_text())["status"] == "ok"
    r2 = subprocess.run(args, capture_output=True, text=True)
    assert "skip" in (r2.stdout + r2.stderr).lower()
