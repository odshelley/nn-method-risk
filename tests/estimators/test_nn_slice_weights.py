import numpy as np
import torch

from neural_particle_method.estimators.nn import NNRegressor


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return x, 0.04 + 0.01 * x**2


def test_default_keeps_nothing_and_is_numerically_unchanged():
    x, v = _data()
    a = NNRegressor(seed=0, first_steps=5, later_steps=2)
    b = NNRegressor(seed=0, first_steps=5, later_steps=2, keep_slice_weights=True)
    grid = np.linspace(-1, 1, 5)
    pa = a.fit_predict(0.1, x, v, grid)
    pb = b.fit_predict(0.1, x, v, grid)
    np.testing.assert_array_equal(pa, pb)
    assert a.slice_weights == [] and len(b.slice_weights) == 1


def test_snapshots_are_copies_taken_per_slice():
    x, v = _data()
    est = NNRegressor(seed=0, first_steps=5, later_steps=2, keep_slice_weights=True)
    grid = np.linspace(-1, 1, 5)
    est.fit_predict(0.1, x, v, grid)
    w0 = {k: t.clone() for k, t in est.slice_weights[0][1].items()}
    est.fit_predict(0.2, x, v + 0.01, grid)
    assert [t for t, _ in est.slice_weights] == [0.1, 0.2]
    for k, t in w0.items():
        # first snapshot untouched by later training
        assert torch.equal(t, est.slice_weights[0][1][k])
    later = est.slice_weights[0][1].items()
    assert any(not torch.equal(t, est.slice_weights[1][1][k]) for k, t in later)
