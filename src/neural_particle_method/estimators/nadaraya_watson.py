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

    def __init__(self, bandwidth=None, bandwidth_scale=1.0):
        self.bandwidth = bandwidth
        self.bandwidth_scale = bandwidth_scale

    def fit_predict(self, t, lnx, v, grid, weights=None):
        bandwidth = self.bandwidth
        if bandwidth is None:
            bandwidth = 1.06 * np.std(lnx) * len(lnx) ** (-1 / 5) * self.bandwidth_scale
        return nw_estimate(lnx, v, grid, weights=weights, bandwidth=bandwidth)


def quartic_kernel(u):
    return np.where(np.abs(u) <= 1.0, 15.0 / 16.0 * (1.0 - u ** 2) ** 2, 0.0)


class GHLKernel:
    """Nadaraya-Watson in spot with the GHL rule-of-thumb bandwidth as stated by Cozma et al. (2019):
    h(t) = c * 1.5 * S0 * sigma_LV(S0, t) * sqrt(max(t, 0.25)) * N^(-1/5). Quartic kernel by default.
    Without a local-vol object the ATM vol is proxied by sqrt(mean(v)) of the cloud."""
    supports_weights = True

    def __init__(self, c=1.0, kernel="quartic", local_vol=None, s0=1.0):
        if kernel not in ("quartic", "gaussian"):
            raise ValueError(f"unknown kernel {kernel!r}")
        self.c, self.kernel, self.local_vol, self.s0 = c, kernel, local_vol, s0

    def sigma_atm(self, t, v):
        if self.local_vol is not None:
            s = self.local_vol.sigma(max(t, self.local_vol.T_grid[0]), self.s0, self.s0)
            return float(np.asarray(s, dtype=float).reshape(-1)[0])
        return float(np.sqrt(np.mean(v)))

    def bandwidth(self, t, n, v):
        return self.c * 1.5 * self.s0 * self.sigma_atm(t, v) * np.sqrt(max(t, 0.25)) * n ** (-0.2)

    def fit_predict(self, t, lnx, v, grid, weights=None, ctx=None):
        S, K = np.exp(lnx), np.exp(grid)
        h = self.bandwidth(t, len(lnx), v)
        u = (K[:, None] - S[None, :]) / h
        ker = quartic_kernel(u) if self.kernel == "quartic" else np.exp(-0.5 * u ** 2)
        if weights is not None:
            ker = ker * weights[None, :]
        den, num = ker.sum(axis=1), ker @ v
        ok = den > 0.0
        f = np.empty(len(grid))
        f[ok] = num[ok] / den[ok]
        if not ok.all():
            if not ok.any():
                f[:] = np.average(v, weights=weights)
            else:
                f[~ok] = np.interp(grid[~ok], grid[ok], f[ok])
        return f
