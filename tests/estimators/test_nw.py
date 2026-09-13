import numpy as np

from neural_particle_method.estimators.nadaraya_watson import (
    NadarayaWatson,
    cloud_std,
    nw_estimate,
)


def test_constant_data_gives_constant_estimate():
    lnx = np.random.default_rng(1).normal(0, 0.3, 500)
    v = np.full(500, 0.05)
    f = NadarayaWatson().fit_predict(0.1, lnx, v, np.linspace(-0.5, 0.5, 7))
    assert np.allclose(f, 0.05)


def test_class_matches_function():
    rng = np.random.default_rng(2)
    lnx, v, g = rng.normal(0, 0.3, 400), rng.uniform(0.01, 0.09, 400), np.linspace(-0.3, 0.3, 5)
    np.testing.assert_array_equal(NadarayaWatson().fit_predict(0.0, lnx, v, g), nw_estimate(lnx, v, g))


def test_unit_weights_are_the_unweighted_fit_exactly():
    """The weighted bandwidth rule must collapse onto the unweighted one, so the untilted arm of
    the importance-sampling study stays bit-for-bit."""
    rng = np.random.default_rng(3)
    lnx, v = rng.normal(0.02, 0.3, 600), rng.uniform(0.01, 0.09, 600)
    g = np.linspace(-0.5, 0.5, 13)
    a = NadarayaWatson().fit_predict(0.5, lnx, v, g, weights=np.ones(600))
    np.testing.assert_array_equal(a, NadarayaWatson().fit_predict(0.5, lnx, v, g))


def test_weights_on_a_narrow_subset_shrink_the_bandwidth():
    """Weights say which cloud the sample represents: concentrate them on a narrow slab and the
    bandwidth is the slab's, not the full sample's."""
    rng = np.random.default_rng(4)
    lnx, v = rng.normal(0.0, 0.4, 800), rng.uniform(0.01, 0.09, 800)
    g = np.linspace(-0.2, 0.2, 9)
    w = np.where(np.abs(lnx) < 0.05, 1.0, 1e-12)
    narrow = cloud_std(lnx, w)
    assert narrow < 0.5 * cloud_std(lnx)
    f = NadarayaWatson().fit_predict(0.5, lnx, v, g, weights=w)
    expected = nw_estimate(lnx, v, g, weights=w,
                           bandwidth=1.06 * narrow * len(lnx) ** (-1 / 5))
    np.testing.assert_allclose(f, expected, rtol=0, atol=0)
