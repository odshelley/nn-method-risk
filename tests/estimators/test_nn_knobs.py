import numpy as np

from neural_particle_method.estimators.nn import NNRegressor


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return x, 0.04 + 0.01 * x**2


def test_depth_three_body_has_three_linear_layers():
    est = NNRegressor(seed=0, first_steps=1, later_steps=1, depth=3)
    linears = [m for m in est.net.body if hasattr(m, "weight")]
    assert len(linears) == 3
    assert [m.in_features for m in linears] == [1, 64, 64]
    assert [m.out_features for m in linears] == [64, 64, 64]


def test_batch_size_none_is_bit_for_bit_with_todays_defaults():
    x, v = _data()
    grid = np.linspace(-1, 1, 5)
    a = NNRegressor(seed=0, first_steps=5, later_steps=2)
    b = NNRegressor(seed=0, first_steps=5, later_steps=2, batch_size=None, lr=1e-2, depth=2)
    pa = a.fit_predict(0.1, x, v, grid)
    pb = b.fit_predict(0.1, x, v, grid)
    np.testing.assert_array_equal(pa, pb)


def test_minibatch_training_runs_and_gives_finite_output():
    x, v = _data(n=200)
    grid = np.linspace(-1, 1, 5)
    est = NNRegressor(seed=0, first_steps=5, later_steps=2, batch_size=64)
    pred = est.fit_predict(0.1, x, v, grid)
    assert np.all(np.isfinite(pred))


def test_lr_is_applied_to_the_optimiser():
    est = NNRegressor(seed=0, lr=1e-3)
    assert est.opt.param_groups[0]["lr"] == 1e-3
