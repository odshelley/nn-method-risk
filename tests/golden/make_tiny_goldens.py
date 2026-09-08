"""Generate tests/golden/tiny_<algo>.json from the CURRENT code. Run once before the refactor.

Usage: uv run python tests/golden/make_tiny_goldens.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neural_particle_method.bench.algos import ALGOS, run_algo
from neural_particle_method.bench.scenarios import make_registry, quote_k_grid
from neural_particle_method.pricing.metrics import iv_metrics, target_ivs
from neural_particle_method.pricing.reprice import RepriceConfig, reprice_iv
from tests.conftest import (
    TINY,
    TINY_EXPLICIT,
    TINY_IMPLICIT,
    TINY_N,
    TINY_REPRICE_N,
    TINY_REPRICE_STEPS,
)

OUT = Path(__file__).parent


def main():
    sc = make_registry()["s01"]
    k = quote_k_grid()
    mats = list(sc.maturities)
    for algo in ALGOS:
        res = run_algo(algo, sc, TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT)
        ivs = reprice_iv(res.field.to_records(), sc.dynamics, sc.s0, mats, k,
                         cfg=RepriceConfig(TINY_REPRICE_N, TINY_REPRICE_STEPS), seed=10_000)
        tgt = target_ivs(sc.ssvi, k, mats, TINY_REPRICE_STEPS)
        doc = {
            "algo": algo, "n_particles": TINY_N, "seed": 0, "cfg": TINY,
            "L_records": [{"t": float(t), "grid": g.tolist(), "L": L.tolist(), "f": f.tolist()}
                          for (t, g, L, f) in res.field.to_records()],
            "metrics": iv_metrics(ivs, tgt, k, mats),
            "iv_err_bp": ((ivs - tgt) * 1e4).tolist(),
        }
        (OUT / f"tiny_{algo}.json").write_text(json.dumps(doc))
        print("wrote", algo)


if __name__ == "__main__":
    main()
