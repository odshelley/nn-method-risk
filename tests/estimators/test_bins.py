import numpy as np

from neural_particle_method.estimators import make_estimator
from neural_particle_method.estimators.bins import Bins

RNG = np.random.default_rng(7)


def test_recovers_bucket_means_exactly():
    # 4 buckets of 3 particles each; the estimate at a grid point is its bucket's mean
    lnx = np.array([0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2])
    v = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0])
    est = Bins(n_bins=4)
    f = est.fit_predict(0.5, lnx, v, np.array([0.1, 1.1, 2.1, 3.1]))
    np.testing.assert_allclose(f, [1.0, 2.0, 3.0, 4.0])


def test_grid_outside_cloud_clamps_to_edge_buckets():
    lnx = np.linspace(-1.0, 1.0, 100)
    v = np.linspace(0.01, 0.09, 100)
    f = Bins(n_bins=10).fit_predict(0.5, lnx, v, np.array([-5.0, 5.0]))
    assert f[0] < f[1]
    assert np.all(np.isfinite(f))


def test_monotone_signal_recovered_monotone():
    lnx = RNG.normal(0.0, 0.3, 5_000)
    v = 0.04 + 0.02 * lnx + RNG.normal(0.0, 0.001, 5_000)
    grid = np.linspace(-0.5, 0.5, 21)
    f = make_estimator("bins").fit_predict(0.5, lnx, v, grid)
    assert np.all(np.diff(f) >= -1e-4)


def test_weights_shift_bucket_means():
    lnx = np.array([0.0, 0.0, 1.0, 1.0])
    v = np.array([1.0, 3.0, 1.0, 3.0])
    w = np.array([1.0, 0.0, 0.0, 1.0])
    f = Bins(n_bins=2).fit_predict(0.5, lnx, v, np.array([0.0, 1.0]))
    fw = Bins(n_bins=2).fit_predict(0.5, lnx, v, np.array([0.0, 1.0]), weights=w)
    np.testing.assert_allclose(f, [2.0, 2.0])
    np.testing.assert_allclose(fw, [1.0, 3.0])
