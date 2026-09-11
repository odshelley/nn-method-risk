import numpy as np
import pytest

from neural_particle_method.calibrate.config import ExplicitConfig
from neural_particle_method.calibrate.explicit import TAIL_DX, extend_tail

GRID = np.array([-0.2, 0.0, 0.2])
F = np.array([0.06, 0.04, 0.03])


def test_flat_is_a_no_op():
    g, f = extend_tail(GRID, F, "flat")
    np.testing.assert_array_equal(g, GRID)
    np.testing.assert_array_equal(f, F)


def test_linear_continues_with_the_end_slopes():
    g, f = extend_tail(GRID, F, "linear")
    assert g[0] == pytest.approx(-0.2 - TAIL_DX) and g[-1] == pytest.approx(0.2 + TAIL_DX)
    assert f[0] == pytest.approx(0.06 + (0.06 - 0.04) / 0.2 * TAIL_DX)
    assert f[-1] == pytest.approx(0.03 + (0.03 - 0.04) / 0.2 * TAIL_DX)
    np.testing.assert_array_equal(f[1:-1], F)


def test_free_asks_the_estimator():
    class Est:
        def predict(self, x):
            return 0.01 + 0.0 * x
    _g, f = extend_tail(GRID, F, "free", Est())
    assert f[0] == pytest.approx(0.01) and f[-1] == pytest.approx(0.01)


def test_unknown_tail_and_default():
    assert ExplicitConfig().tail == "flat"
    with pytest.raises(ValueError):
        extend_tail(GRID, F, "quadratic")
