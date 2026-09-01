"""Defensive-mixture importance sampling: offline design targeting the optimal marginal."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MixtureDesign:
    alphas: tuple
    thetas: tuple
    rho: float

    @property
    def etas(self):
        r = np.sqrt(1 - self.rho ** 2)
        return tuple(th / r for th in self.thetas)


def design_mixture(dynamics, T, k_target=0.47, alpha0=0.5, ess_cost_cap=3.0):
    """Moment-matched wing tilts with a defensive untilted component and a kernel-cost cap."""
    rho, v0, theta_bar = dynamics["rho"], dynamics["v0"], dynamics["theta"]
    sig_bar = np.sqrt(0.5 * (v0 + theta_bar))
    th = k_target / (sig_bar * T)
    th_cap = np.sqrt(ess_cost_cap * (1 - rho ** 2) / T)
    th = min(th, th_cap)
    a_wing = 0.5 * (1 - alpha0)
    return MixtureDesign(alphas=(a_wing, alpha0, a_wing), thetas=(-th, 0.0, th), rho=rho)
