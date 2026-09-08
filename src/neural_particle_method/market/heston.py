"""Semi-analytic Heston call pricing via the trap-free characteristic function (r = q = 0)."""
import numpy as np


def heston_cf(u, T, v0, kappa, theta, xi, rho, s0=1.0):
    """Characteristic function of ln S_T. Trap-free (Albrecher) formulation, vectorised in u."""
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    beta = kappa - rho * xi * iu
    d = np.sqrt(beta ** 2 + xi ** 2 * (iu + u ** 2))
    g = (beta - d) / (beta + d)
    e = np.exp(-d * T)
    C = (kappa * theta / xi ** 2) * ((beta - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    D = ((beta - d) / xi ** 2) * (1.0 - e) / (1.0 - g * e)
    return np.exp(C + D * v0 + iu * np.log(s0))


def heston_call(K, T, v0, kappa, theta, xi, rho, s0=1.0, n_quad=256, u_max=200.0):
    """Call prices via the P1/P2 representation with Gauss-Legendre quadrature on [0, u_max]."""
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))
    x, w = np.polynomial.legendre.leggauss(n_quad)
    u = 0.5 * u_max * (x + 1.0)
    w = w * 0.5 * u_max

    phi2 = heston_cf(u, T, v0, kappa, theta, xi, rho, s0)
    s_fwd = heston_cf(np.array([-1j]), T, v0, kappa, theta, xi, rho, s0)[0].real
    phi1 = heston_cf(u - 1j, T, v0, kappa, theta, xi, rho, s0) / s_fwd

    lnK = np.log(K_arr)[:, None]
    ker = np.exp(-1j * u[None, :] * lnK) / (1j * u[None, :])
    P1 = 0.5 + (np.real(ker * phi1[None, :]) @ w) / np.pi
    P2 = 0.5 + (np.real(ker * phi2[None, :]) @ w) / np.pi

    price = s0 * P1 - K_arr * P2
    price = np.maximum(price, np.maximum(s0 - K_arr, 0.0) + 1e-14)
    return price if np.ndim(K) else float(price[0])


def heston_iv(k_grid, T, params, s0=1.0):
    """Implied vol of the semi-analytic Heston call at log-moneyness k_grid and maturity T."""
    from .bs import implied_vol

    K = s0 * np.exp(np.asarray(k_grid, dtype=float))
    prices = np.atleast_1d(heston_call(K, T, params.v0, params.kappa, params.theta, params.xi, params.rho, s0))
    return np.array([implied_vol(float(p), s0, float(kk), T) for p, kk in zip(prices, K)])
