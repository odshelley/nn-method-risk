"""Bit-for-bit replay of four archived paper runs. Slow: run with `uv run pytest -m golden`."""
import json
from pathlib import Path

import numpy as np
import pytest

from neural_particle_method.bench.algos import run_algo
from neural_particle_method.bench.scenarios import make_registry, quote_k_grid
from neural_particle_method.pricing.metrics import iv_metrics, target_ivs
from neural_particle_method.pricing.reprice import RepriceConfig, reprice_iv

RUNS = Path(__file__).resolve().parents[1] / "results" / "runs"
CASES = ["nw", "explicit_nn", "explicit_nn_is", "implicit_ridge"]


def replay(algo, n_particles=50_000, seed=0, sid="s01"):
    sc = make_registry()[sid]
    k = quote_k_grid()
    mats = list(sc.maturities)
    res = run_algo(algo, sc, n_particles, seed)
    ivs = reprice_iv(res.field.to_records(), sc.dynamics, sc.s0, mats, k,
                     cfg=RepriceConfig(500_000, 200), seed=seed + 10_000)
    tgt = target_ivs(sc.ssvi, k, mats, 200)
    return iv_metrics(ivs, tgt, k, mats), (ivs - tgt) * 1e4


@pytest.mark.golden
@pytest.mark.parametrize("algo", CASES)
def test_archived_run_replays_bit_for_bit(algo):
    doc = json.loads((RUNS / "s01" / algo / "n50000_s0.json").read_text())
    assert doc["status"] == "ok"
    metrics, err = replay(algo)
    for key in ("pooled_rmse_bp", "pooled_max_bp", "wings_rmse_bp", "wings_max_bp", "n_failed"):
        assert metrics[key] == doc["metrics"][key], key
    ref = np.array([[np.nan if v is None else v for v in row] for row in doc["iv_err_bp"]])
    np.testing.assert_array_equal(err, ref)
