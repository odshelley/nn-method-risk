import numpy as np
import pytest
from bench.algos import ALGOS, run_algo
from bench.scenarios import make_registry

TINY = {"n_steps": 6, "fit_subsample": 4_000, "n_iters": 2, "first_steps": 80, "later_steps": 30, "fit_steps": 80}

@pytest.mark.parametrize("name", list(ALGOS))
def test_each_algo_runs_and_returns_records(name):
    sc = make_registry()["s01"]
    res = run_algo(name, sc, n_particles=6_000, seed=0, cfg=TINY)
    assert len(res.L_records) == 6
    t, grid, Lg, fg = res.L_records[-1]
    assert np.all(np.isfinite(Lg)) and Lg.min() >= 0.0 and Lg.max() <= 4.0
    assert res.timings["total_s"] > 0
    if name == "explicit_nn_is":
        assert res.diagnostics["is_diag"]["max_w"] <= 2.0 + 1e-9
