"""SSVI (power-law) implied variance surface with analytic Dupire local volatility."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SSVIParams:
    sigma0: float
    eta: float
    gamma: float
    rho: float


def no_arb_ok(p):
    """Gatheral-Jacquier sufficient conditions for the power-law parameterisation."""
    return (p.sigma0 > 0 and 0 < p.gamma <= 0.5 and abs(p.rho) < 1
            and p.eta * (1 + abs(p.rho)) <= 2.0)


def _w_and_derivs(p, k, T):
    """w, dw/dk, d2w/dk2, dw/dT at (k, T); all closed form."""
    k = np.asarray(k, dtype=float)
    th = p.sigma0 ** 2 * T
    phi = p.eta * th ** (-p.gamma)
    dphi = -p.gamma * p.eta * th ** (-p.gamma - 1)
    u = phi * k + p.rho
    R = np.sqrt(u ** 2 + 1 - p.rho ** 2)
    w = 0.5 * th * (1 + p.rho * phi * k + R)
    dwdk = 0.5 * th * phi * (p.rho + u / R)
    d2wdk2 = 0.5 * th * phi ** 2 * (1 - p.rho ** 2) / R ** 3
    dwdth = w / th + 0.5 * th * dphi * k * (p.rho + u / R)
    dwdT = dwdth * p.sigma0 ** 2
    return w, dwdk, d2wdk2, dwdT


def total_variance(p, k, T):
    return _w_and_derivs(p, k, T)[0]


def implied_vol_ssvi(p, k, T):
    return np.sqrt(total_variance(p, k, T) / T)


class SSVILocalVol:
    """Analytic Dupire local vol from an SSVI surface; duck-types DupireSurface for calibrate_explicit."""

    def __init__(self, p, s0=1.0, t_min=0.004, T_max=2.0):
        self.p, self.s0 = p, s0
        self.T_grid = np.array([t_min, T_max])

    def sigma(self, t, x, s0=1.0):
        t = float(np.clip(t, self.T_grid[0], self.T_grid[-1]))
        k = np.log(np.asarray(x, dtype=float) / s0)
        w, dwdk, d2wdk2, dwdT = _w_and_derivs(self.p, k, T=t)
        denom = (1.0 - k / w * dwdk
                 + 0.25 * (-0.25 - 1.0 / w + (k / w) ** 2) * dwdk ** 2
                 + 0.5 * d2wdk2)
        v_loc = np.clip(dwdT, 1e-8, None) / np.clip(denom, 0.05, None)
        return np.sqrt(np.clip(v_loc, 1e-8, 9.0))
