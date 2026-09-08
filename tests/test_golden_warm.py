"""Bit-for-bit replay of the warm-path (bump-correct and dyn-arm) goldens generated from the
pre-refactor code at commit de16b9b (see tests/golden/make_warm_goldens.py)."""
import json
from pathlib import Path

import pytest

from neural_particle_method.bench.scenarios import make_registry
from neural_particle_method.experiments.bump_correct import run_pair
from neural_particle_method.experiments.config import BumpConfig, WarmConfig
from neural_particle_method.experiments.warm_suite import arm_dyn
from neural_particle_method.tracking.store import Store

GOLDEN = Path(__file__).parent / "golden"


@pytest.mark.golden
def test_bump_run_pair_matches_pre_refactor_golden(tmp_path):
    doc = json.loads((GOLDEN / "warm_bump_tiny.json").read_text())
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    cfg = BumpConfig(N=2000, n_steps=6, sub=1000, n_iters=2, fit_steps=40, fit_v_floor=False)
    rid = run_pair(store, make_registry()["s01"], 0, cfg)
    result = json.loads(store.download(rid, "result.json", tmp_path / "dl").read_text())
    for key, ref in doc["rmse_bp"].items():
        assert result["rmse_bp"][key] == ref, key
    for key, ref in doc["bumped"].items():
        assert result["bumped"][key] == ref, key


@pytest.mark.golden
def test_arm_dyn_matches_pre_refactor_golden(tmp_path):
    doc = json.loads((GOLDEN / "warm_dyn_tiny.json").read_text())
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    cfg = WarmConfig(N=2000, n_steps=6, sub=1000, n_iters=2, fit_steps=40,
                     reprice_N=20000, reprice_steps=20, fit_v_floor=False)
    result = arm_dyn(store, make_registry()["s01"], 0, cfg)
    for key, ref in doc["rmse_bp"].items():
        assert result["rmse_bp"][key] == ref, key
