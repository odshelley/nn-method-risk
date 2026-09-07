import numpy as np
import pytest
from bench.algos import ALGOS, run_algo
from bench.scenarios import make_registry
from tests.conftest import TINY_EXPLICIT, TINY_IMPLICIT

@pytest.mark.parametrize("name", list(ALGOS))
def test_each_algo_runs_and_returns_records(name):
    sc = make_registry()["s01"]
    res = run_algo(name, sc, n_particles=6_000, seed=0, explicit=TINY_EXPLICIT, implicit=TINY_IMPLICIT)
    assert len(res.field) == 6
    s = res.field[-1]
    assert np.all(np.isfinite(s.L)) and s.L.min() >= 0.0 and s.L.max() <= 4.0
    assert res.timings["total_s"] > 0
    if name == "explicit_nn_is":
        assert res.diagnostics["is_diag"]["max_w"] <= 2.0 + 1e-9
