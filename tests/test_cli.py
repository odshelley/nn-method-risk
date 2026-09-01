import subprocess, sys

def test_bench_list_runs():
    r = subprocess.run(["uv", "run", "bench", "list"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "scenarios" in r.stdout.lower()
