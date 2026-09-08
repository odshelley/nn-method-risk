import numpy as np
import pytest
import torch

from neural_particle_method.calibrate.implicit import GlobalNet
from neural_particle_method.estimators import make_estimator
from neural_particle_method.estimators.ridge import GlobalRidge

RNG = np.random.default_rng(0)
LNX = RNG.normal(0.0, 0.2, 3_000)
V = 0.04 + 0.01 * LNX + RNG.normal(0.0, 0.002, 3_000)
GRID = np.linspace(-0.4, 0.4, 17)


def _global_net():
    torch.manual_seed(0)
    return GlobalNet()


def _all():
    return [make_estimator("nn", seed=0, first_steps=40, later_steps=20),
            make_estimator("nw"),
            make_estimator("nw_ghl"),
            make_estimator("ridge", seed=0, first_steps=40),
            make_estimator("spline"),
            make_estimator("bins"),
            make_estimator("rkhs"),
            GlobalRidge(_global_net(), T=1.0)]


@pytest.mark.parametrize("est", _all(), ids=lambda e: type(e).__name__)
def test_shape_finite_positive(est):
    f = est.fit_predict(0.5, LNX, V, GRID)
    assert f.shape == GRID.shape and np.all(np.isfinite(f))
    if type(est).__name__ in ("NNRegressor", "NadarayaWatson"):   # softplus output / positive data
        assert np.all(f > 0)


@pytest.mark.parametrize("est", [e for e in _all() if e.supports_weights], ids=lambda e: type(e).__name__)
def test_unit_weights_equal_no_weights(est):
    a = est.fit_predict(0.5, LNX, V, GRID)
    # a fresh copy of the same estimator, same seed, weighted with ones
    est2 = type(est)(**_ctor_kwargs(est))
    b = est2.fit_predict(0.5, LNX, V, GRID, weights=np.ones(len(LNX)))
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9)


def _ctor_kwargs(est):
    name = type(est).__name__
    if name == "NNRegressor":
        return {"seed": 0, "first_steps": 40, "later_steps": 20}
    if name == "SliceRidge":
        return {"seed": 0, "body_steps": 40}
    if name == "GlobalRidge":
        return {"net": _global_net(), "T": 1.0}
    return {}


def test_spline_rejects_weights():
    with pytest.raises(ValueError, match="importance weights"):
        make_estimator("spline").fit_predict(0.5, LNX, V, GRID, weights=np.ones(len(LNX)))
