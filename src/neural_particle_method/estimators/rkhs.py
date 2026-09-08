"""RKHS ridge conditional-expectation estimator (Bayer-Belomestny-Butkovsky-Schoenmakers, arXiv:2203.01160)."""
import numpy as np


class RKHSRidge:
    """Kernel ridge with L quantile centres: solve (K'WK + n*lam*R) beta = K'W v, their (4.3)/(5.9).

    Gaussian kernel k(x, y) = exp(-(x - y)^2 / (2 * variance)) on the log-spot cloud.
    Centres sit at the j*100/(L+1) percentiles of the current slice's cloud; no state across slices.
    """
    supports_weights = True

    def __init__(self, n_centres=100, lam=1e-9, variance=0.1):
        self.n_centres = n_centres
        self.lam = lam
        self.variance = variance

    def _kernel(self, a, b):
        d = a[:, None] - b[None, :]
        return np.exp(-0.5 * d**2 / self.variance)

    def fit_predict(self, t, lnx, v, grid, weights=None):
        n = len(lnx)
        L = min(self.n_centres, n)
        q = np.arange(1, L + 1) * (100.0 / (L + 1))
        centres = np.percentile(lnx, q)
        K = self._kernel(lnx, centres)
        R = self._kernel(centres, centres)
        wv = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
        lhs = K.T @ (wv[:, None] * K) + n * self.lam * R + 1e-12 * np.eye(L)
        rhs = K.T @ (wv * v)
        beta = np.linalg.solve(lhs, rhs)
        return self._kernel(grid, centres) @ beta
