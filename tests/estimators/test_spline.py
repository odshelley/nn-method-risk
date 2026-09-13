import numpy as np

from neural_particle_method.estimators.spline import PSpline


def test_recovers_quadratic():
    rng = np.random.default_rng(3)
    lnx = rng.uniform(-0.5, 0.5, 5_000)
    v = 0.04 + 0.1 * lnx ** 2
    g = np.linspace(-0.3, 0.3, 13)
    f = PSpline(lam=1e-3).fit_predict(0.0, lnx, v, g)
    assert np.abs(f - (0.04 + 0.1 * g ** 2)).max() < 1e-3


import pytest

from neural_particle_method.estimators.spline import spline_estimate


def test_unit_weights_equal_the_unweighted_fit():
    rng = np.random.default_rng(4)
    lnx = rng.uniform(-0.5, 0.5, 2_000)
    v = 0.04 + 0.1 * lnx ** 2 + rng.normal(scale=0.01, size=lnx.size)
    g = np.linspace(-0.3, 0.3, 13)
    a = spline_estimate(lnx, v, g)
    b = spline_estimate(lnx, v, g, weights=np.ones_like(v))
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


def test_weight_two_equals_a_duplicated_point():
    rng = np.random.default_rng(5)
    lnx = rng.uniform(-0.5, 0.5, 500)
    v = 0.04 + 0.1 * lnx ** 2 + rng.normal(scale=0.01, size=lnx.size)
    g = np.linspace(-0.3, 0.3, 13)
    w = np.ones_like(v); w[:50] = 2.0
    # knots are quantiles of the sample; pin them so both fits share the same basis
    q = np.quantile(lnx, np.linspace(0.005, 0.995, 25))
    a = spline_estimate(lnx, v, g, weights=w, knots=q)
    b = spline_estimate(np.concatenate([lnx, lnx[:50]]), np.concatenate([v, v[:50]]), g, knots=q)
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_pspline_accepts_weights():
    rng = np.random.default_rng(6)
    lnx = rng.uniform(-0.5, 0.5, 1_000)
    v = 0.04 + 0.1 * lnx ** 2
    g = np.linspace(-0.3, 0.3, 7)
    est = PSpline(lam=1e-3)
    assert est.supports_weights
    f = est.fit_predict(0.0, lnx, v, g, weights=np.ones_like(v))
    assert np.abs(f - (0.04 + 0.1 * g ** 2)).max() < 1e-3
    with pytest.raises(ValueError):
        est.fit_predict(0.0, lnx, v, g, weights=-np.ones_like(v))
