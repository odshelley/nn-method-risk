"""Generate tests/golden/warm_bump_tiny.json and warm_dyn_tiny.json from the PRE-REFACTOR code.

This script imports the old flat-module API (`bench.scenarios`, `experiments.bump_correct.run_pair`,
`experiments.warm_suite.Cfg`/`arm_dyn`) as it existed at commit de16b9b. It will NOT import in the
current (refactored) tree, where those modules live under `neural_particle_method.*` with a different
shape. That is expected: this file is a record of exactly how the goldens were produced, not something
meant to run here.

To regenerate:
    git worktree add <path> de16b9b
    cd <path> && uv sync
    uv run python <this file, copied into the worktree root> <out_dir>

Usage inside the worktree:
    uv run python make_warm_goldens.py <out_dir>
"""
import dataclasses
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "experiments"))

from bench.scenarios import make_registry
from bump_correct import run_pair
from warm_suite import Cfg, arm_dyn

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

reg = make_registry()

with tempfile.TemporaryDirectory() as bump_out:
    bump_out = Path(bump_out)
    run_pair(reg["s01"], 0, N=2000, n_steps=6, sub=1000, n_iters=2, fit_steps=40, out_dir=bump_out)
    bump_doc = json.loads((bump_out / "s01_s0.json").read_text())

bump_golden = {
    "params": {"sid": "s01", "seed": 0, "N": 2000, "n_steps": 6, "sub": 1000, "n_iters": 2, "fit_steps": 40},
    "rmse_bp": bump_doc["rmse_bp"],
    "bumped": bump_doc["bumped"],
}
(OUT / "warm_bump_tiny.json").write_text(json.dumps(bump_golden, indent=1))
print("wrote warm_bump_tiny.json")

cfg = Cfg(N=2000, n_steps=6, sub=1000, n_iters=2, fit_steps=40, reprice_N=20000, reprice_steps=20)
dyn_doc = arm_dyn(reg["s01"], 0, cfg)

dyn_golden = {
    "params": {"sid": "s01", "seed": 0, **{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}},
    "rmse_bp": dyn_doc["rmse_bp"],
}
(OUT / "warm_dyn_tiny.json").write_text(json.dumps(dyn_golden, indent=1))
print("wrote warm_dyn_tiny.json")
