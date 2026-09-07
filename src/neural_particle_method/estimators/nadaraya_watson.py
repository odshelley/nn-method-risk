"""Gaussian Nadaraya-Watson conditional-expectation estimator."""
import numpy as np


def nw_estimate(lnx, v, lnx_grid, weights=None, bandwidth=None):
    """Gaussian Nadaraya-Watson estimate of E[V | ln X = g] on a grid (optionally weighted)."""
    if bandwidth is None:
        bandwidth = 1.06 * np.std(lnx) * len(lnx) ** (-1 / 5)
    d = (lnx_grid[:, None] - lnx[None, :]) / bandwidth
    ker = np.exp(-0.5 * d ** 2)
    if weights is not None:
        ker = ker * weights[None, :]
    num = ker @ v
    den = ker.sum(axis=1)
    return num / np.clip(den, 1e-300, None)


class NadarayaWatson:
    supports_weights = True

    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth

    def fit_predict(self, t, lnx, v, grid, weights=None):
        return nw_estimate(lnx, v, grid, weights=weights, bandwidth=self.bandwidth)
