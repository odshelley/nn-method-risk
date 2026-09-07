"""Generate tests/golden/tiny_<algo>.json from the CURRENT code. Run once before the refactor.

Usage: uv run python tests/golden/make_tiny_goldens.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.conftest import TINY, TINY_N, TINY_REPRICE_N, TINY_REPRICE_STEPS  # noqa: E402

from bench.algos import ALGOS, run_algo  # noqa: E402
from bench.scenarios import make_registry, quote_k_grid  # noqa: E402
from neural_particle_method.reprice import iv_metrics, reprice_iv, snap_times  # noqa: E402
from neural_particle_method.ssvi import implied_vol_ssvi  # noqa: E402

OUT = Path(__file__).parent


def main():
    sc = make_registry()["s01"]
    k = quote_k_grid()
    mats = list(sc.maturities)
    for algo in ALGOS:
        res = run_algo(algo, sc, TINY_N, 0, TINY)
        ivs = reprice_iv(res.L_records, sc.dynamics, sc.s0, mats, k,
                         n_particles=TINY_REPRICE_N, n_steps=TINY_REPRICE_STEPS, seed=10_000)
        ts = snap_times(mats, TINY_REPRICE_STEPS)
        tgt = np.stack([implied_vol_ssvi(sc.ssvi, k, t) for t in ts])
        doc = {
            "algo": algo, "n_particles": TINY_N, "seed": 0, "cfg": TINY,
            "L_records": [{"t": float(t), "grid": g.tolist(), "L": L.tolist(), "f": f.tolist()}
                          for (t, g, L, f) in res.L_records],
            "metrics": iv_metrics(ivs, tgt, k, mats),
            "iv_err_bp": ((ivs - tgt) * 1e4).tolist(),
        }
        (OUT / f"tiny_{algo}.json").write_text(json.dumps(doc))
        print("wrote", algo)


if __name__ == "__main__":
    main()
