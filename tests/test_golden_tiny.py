"""Bit-for-bit replay of tiny goldens generated from the pre-refactor code."""
import json
from pathlib import Path

import numpy as np
import pytest

from bench.algos import ALGOS, run_algo
from tests.conftest import TINY, TINY_N, TINY_REPRICE_N, TINY_REPRICE_STEPS
from bench.scenarios import make_registry, quote_k_grid
from neural_particle_method.reprice import iv_metrics, reprice_iv, snap_times
from neural_particle_method.ssvi import implied_vol_ssvi

GOLDEN = Path(__file__).parent / "golden"


def run_tiny(algo):
    """Run one algorithm at the tiny config. Later tasks update this helper as the API moves;
    the assertions below never change."""
    sc = make_registry()["s01"]
    k = quote_k_grid()
    mats = list(sc.maturities)
    res = run_algo(algo, sc, TINY_N, 0, TINY)
    ivs = reprice_iv(res.L_records, sc.dynamics, sc.s0, mats, k,
                     n_particles=TINY_REPRICE_N, n_steps=TINY_REPRICE_STEPS, seed=10_000)
    ts = snap_times(mats, TINY_REPRICE_STEPS)
    tgt = np.stack([implied_vol_ssvi(sc.ssvi, k, t) for t in ts])
    records = [(float(t), np.asarray(g), np.asarray(L), np.asarray(f)) for (t, g, L, f) in res.L_records]
    return records, iv_metrics(ivs, tgt, k, mats), (ivs - tgt) * 1e4


@pytest.mark.parametrize("algo", sorted(ALGOS))
def test_tiny_golden_bit_for_bit(algo):
    doc = json.loads((GOLDEN / f"tiny_{algo}.json").read_text())
    records, metrics, err = run_tiny(algo)
    assert len(records) == len(doc["L_records"])
    for (t, g, L, f), ref in zip(records, doc["L_records"]):
        assert t == ref["t"]
        np.testing.assert_array_equal(g, np.array(ref["grid"]))
        np.testing.assert_array_equal(L, np.array(ref["L"]))
        np.testing.assert_array_equal(f, np.array(ref["f"]))
    for key in ("pooled_rmse_bp", "pooled_max_bp", "wings_rmse_bp", "wings_max_bp", "n_failed"):
        assert metrics[key] == doc["metrics"][key], key
    for got, ref in zip(metrics["per_maturity"], doc["metrics"]["per_maturity"]):
        assert got == ref
    np.testing.assert_array_equal(err, np.array(doc["iv_err_bp"], dtype=float))
