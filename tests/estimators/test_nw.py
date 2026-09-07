import numpy as np

from neural_particle_method.estimators.nadaraya_watson import NadarayaWatson, nw_estimate


def test_constant_data_gives_constant_estimate():
    lnx = np.random.default_rng(1).normal(0, 0.3, 500)
    v = np.full(500, 0.05)
    f = NadarayaWatson().fit_predict(0.1, lnx, v, np.linspace(-0.5, 0.5, 7))
    assert np.allclose(f, 0.05)


def test_class_matches_function():
    rng = np.random.default_rng(2)
    lnx, v, g = rng.normal(0, 0.3, 400), rng.uniform(0.01, 0.09, 400), np.linspace(-0.3, 0.3, 5)
    np.testing.assert_array_equal(NadarayaWatson().fit_predict(0.0, lnx, v, g), nw_estimate(lnx, v, g))
