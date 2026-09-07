"""Dupire local volatility surface from a call-price function, via total implied variance."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .bs import implied_vol
from .local_vol import dupire_local_vol


class DupireSurface:
    """sigma_Dup(t, x) built on a (T, k) grid, k = log(K/s0). Inputs clamped to the grid."""

    def __init__(self, T_grid, k_grid, w):
        self.T_grid, self.k_grid = T_grid, k_grid
        K = k_grid[None, :]
        W = w
        dwdT = np.gradient(W, T_grid, axis=0)
        dwdk = np.gradient(W, k_grid, axis=1)
        d2wdk2 = np.gradient(dwdk, k_grid, axis=1)
        self.sigma_loc = dupire_local_vol(W, dwdk, d2wdk2, dwdT, K)
        self._interp = RegularGridInterpolator(
            (T_grid, k_grid), self.sigma_loc, bounds_error=False, fill_value=None)

    @property
    def t_min(self):
        return float(self.T_grid[0])

    @classmethod
    def from_price_fn(cls, price_fn, s0, T_grid, k_grid):
        w = np.empty((len(T_grid), len(k_grid)))
        for i, T in enumerate(T_grid):
            K = s0 * np.exp(k_grid)
            prices = price_fn(K, T)
            for j, (p, k) in enumerate(zip(prices, K)):
                iv = implied_vol(p, s0, k, T)
                w[i, j] = (iv ** 2) * T if np.isfinite(iv) else np.nan
        # fill any failed inversions by nearest neighbour along strike
        for i in range(w.shape[0]):
            row = w[i]
            if np.isnan(row).any():
                idx = np.arange(len(row))
                good = ~np.isnan(row)
                w[i] = np.interp(idx, idx[good], row[good])
        return cls(T_grid, k_grid, w)

    def sigma(self, t, x, s0=1.0):
        t = np.clip(t, self.T_grid[0], self.T_grid[-1])
        k = np.clip(np.log(np.asarray(x) / s0), self.k_grid[0], self.k_grid[-1])
        pts = np.stack([np.full_like(k, t, dtype=float), k], axis=-1)
        return self._interp(pts)
