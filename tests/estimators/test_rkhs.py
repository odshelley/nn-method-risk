import numpy as np

from neural_particle_method.estimators import make_estimator
from neural_particle_method.estimators.rkhs import RKHSRidge

RNG = np.random.default_rng(11)


def test_recovers_smooth_conditional_mean():
    lnx = RNG.normal(0.0, 0.3, 8_000)
    m = 0.04 + 0.02 * np.tanh(3.0 * lnx)
    v = m + RNG.normal(0.0, 0.002, 8_000)
    grid = np.linspace(-0.5, 0.5, 21)
    f = make_estimator("rkhs").fit_predict(0.5, lnx, v, grid)
    truth = 0.04 + 0.02 * np.tanh(3.0 * grid)
    assert np.max(np.abs(f - truth)) < 5e-3


def test_matches_normal_equations_on_tiny_problem():
    lnx = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    v = np.array([0.05, 0.048, 0.045, 0.043, 0.041, 0.04])
    grid = np.array([-0.15, 0.15])
    est = RKHSRidge(n_centres=3, lam=1e-6, variance=0.1)
    f = est.fit_predict(0.5, lnx, v, grid)

    q = np.arange(1, 4) * (100.0 / 4)
    c = np.percentile(lnx, q)
    K = np.exp(-0.5 * (lnx[:, None] - c[None, :]) ** 2 / 0.1)
    R = np.exp(-0.5 * (c[:, None] - c[None, :]) ** 2 / 0.1)
    beta = np.linalg.solve(K.T @ K + 6 * 1e-6 * R + 1e-12 * np.eye(3), K.T @ v)
    expect = np.exp(-0.5 * (grid[:, None] - c[None, :]) ** 2 / 0.1) @ beta
    np.testing.assert_allclose(f, expect, rtol=1e-10)


def test_large_lambda_flattens_towards_zero_function():
    lnx = RNG.normal(0.0, 0.3, 2_000)
    v = 0.04 + 0.02 * lnx + RNG.normal(0.0, 0.002, 2_000)
    grid = np.linspace(-0.4, 0.4, 9)
    f_small = RKHSRidge(lam=1e-9).fit_predict(0.5, lnx, v, grid)
    f_big = RKHSRidge(lam=1e3).fit_predict(0.5, lnx, v, grid)
    assert np.ptp(f_big) < np.ptp(f_small)
    assert np.max(np.abs(f_big)) < np.max(np.abs(f_small))


def test_stateless_across_slices():
    lnx = RNG.normal(0.0, 0.3, 2_000)
    v = 0.04 + 0.02 * lnx
    grid = np.linspace(-0.4, 0.4, 9)
    est = make_estimator("rkhs")
    a = est.fit_predict(0.1, lnx, v, grid)
    b = est.fit_predict(0.9, lnx, v, grid)
    np.testing.assert_allclose(a, b)
