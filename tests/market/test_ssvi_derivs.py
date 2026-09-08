import numpy as np

from neural_particle_method.market.ssvi import SSVIParams, w_and_derivs

P = SSVIParams(sigma0=0.25, eta=0.8, gamma=0.45, rho=-0.5)


def _fd(fn, x, h=1e-5):
    return (fn(x + h) - fn(x - h)) / (2 * h)


def test_k_derivatives_match_finite_differences():
    k = np.array([-0.4, -0.1, 0.0, 0.2, 0.5])
    T = 0.8
    _, dwdk, d2wdk2, _ = w_and_derivs(P, k, T)
    fd1 = _fd(lambda kk: w_and_derivs(P, kk, T)[0], k)
    fd2 = _fd(lambda kk: w_and_derivs(P, kk, T)[1], k)
    assert np.allclose(dwdk, fd1, rtol=1e-6, atol=1e-9)
    assert np.allclose(d2wdk2, fd2, rtol=1e-5, atol=1e-8)


def test_T_derivative_matches_finite_difference():
    k = np.array([-0.3, 0.0, 0.3])
    T = 1.2
    _, _, _, dwdT = w_and_derivs(P, k, T)
    fd = _fd(lambda TT: w_and_derivs(P, k, TT)[0], T)
    assert np.allclose(dwdT, fd, rtol=1e-6, atol=1e-9)
