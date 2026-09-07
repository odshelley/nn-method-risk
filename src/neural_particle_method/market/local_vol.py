"""Local volatility: the Dupire formula (written once) and the LocalVol protocol."""
from typing import Protocol

import numpy as np

from .ssvi import w_and_derivs


class LocalVol(Protocol):
    T_grid: np.ndarray

    @property
    def t_min(self) -> float: ...

    def sigma(self, t, x, s0=1.0): ...


def dupire_local_vol(w, dwdk, d2wdk2, dwdT, k):
    """Dupire local vol from total implied variance w(k, T) and its derivatives.

    Clipping constants are part of the calibrated behaviour: denominator floored at 0.05,
    local variance in [1e-8, 9]. Do not change them.
    """
    denom = (1.0 - k / w * dwdk
             + 0.25 * (-0.25 - 1.0 / w + (k / w) ** 2) * dwdk ** 2
             + 0.5 * d2wdk2)
    v_loc = np.clip(dwdT, 1e-8, None) / np.clip(denom, 0.05, None)
    return np.sqrt(np.clip(v_loc, 1e-8, 9.0))


class SSVILocalVol:
    """Analytic Dupire local vol from an SSVI surface."""

    def __init__(self, p, s0=1.0, t_min=0.004, T_max=2.0):
        self.p, self.s0 = p, s0
        self.T_grid = np.array([t_min, T_max])

    @property
    def t_min(self):
        return float(self.T_grid[0])

    def sigma(self, t, x, s0=1.0):
        t = float(np.clip(t, self.T_grid[0], self.T_grid[-1]))
        k = np.log(np.asarray(x, dtype=float) / s0)
        w, dwdk, d2wdk2, dwdT = w_and_derivs(self.p, k, T=t)
        return dupire_local_vol(w, dwdk, d2wdk2, dwdT, k)
