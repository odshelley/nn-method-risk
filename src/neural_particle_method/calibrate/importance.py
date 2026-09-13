"""Defensive-mixture importance sampling: offline design targeting the optimal marginal."""
from dataclasses import dataclass

import numpy as np

from ..simulate.dynamics import HestonParams

SCHEDULES = ("constant", "inverse_sqrt", "front")
FRONT_T0 = 0.25   # the front-loaded schedule tilts on [0, FRONT_T0) at full strength, then stops


@dataclass(frozen=True)
class MixtureDesign:
    """Three-component defensive mixture: tilts (-theta, 0, +theta) with weights `alphas`.

    `thetas` is either three floats (a constant tilt, the original design) or a `(3, n_steps)`
    array (a schedule in time); `etas` are the tilts on the orthogonal Brownian direction."""
    alphas: tuple
    thetas: object
    rho: float

    @property
    def etas(self):
        r = np.sqrt(1 - self.rho ** 2)
        th = np.asarray(self.thetas, dtype=float)
        if th.ndim == 2:
            return th / r
        return tuple(t / r for t in self.thetas)

    @property
    def scheduled(self):
        return np.asarray(self.thetas).ndim == 2


def design_mixture(dynamics, T, k_target=0.47, alpha0=0.5, ess_cost_cap=3.0):
    """Moment-matched wing tilts with a defensive untilted component and a kernel-cost cap."""
    p = HestonParams.from_dict(dynamics)
    rho, v0, theta_bar = p.rho, p.v0, p.theta
    sig_bar = np.sqrt(0.5 * (v0 + theta_bar))
    th = k_target / (sig_bar * T)
    th_cap = np.sqrt(ess_cost_cap * (1 - rho ** 2) / T)
    th = min(th, th_cap)
    a_wing = 0.5 * (1 - alpha0)
    return MixtureDesign(alphas=(a_wing, alpha0, a_wing), thetas=(-th, 0.0, th), rho=rho)


@dataclass(frozen=True)
class TiltDesign:
    """A tilt schedule and its cost cap: `cost` bounds sum_k theta_k^2 dt / (1 - rho^2) over [0, T]
    (the log-weight variance of a pure tilt; twice its relative entropy)."""
    schedule: str
    cost: float
    alpha0: float = 0.5

    @property
    def name(self):
        return f"{self.schedule}-{self.cost:g}"

    def mixture(self, n_steps, T, rho):
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}, got {self.schedule!r}")
        a_wing = 0.5 * (1 - self.alpha0)
        alphas = (a_wing, self.alpha0, a_wing)
        r2 = 1 - rho ** 2
        if self.schedule == "constant":
            th = float(np.sqrt(self.cost * r2 / T))
            return MixtureDesign(alphas=alphas, thetas=(-th, 0.0, th), rho=rho)
        dt = T / n_steps
        t = np.arange(n_steps) * dt
        if self.schedule == "front":
            # the whole cost spent on [0, t0): the weights stop moving after t0, so the slice
            # ESS is frozen at its t0 value for the rest of the horizon
            t0 = min(FRONT_T0, T)
            th = np.where(t < t0 - 1e-12, np.sqrt(self.cost * r2 / t0), 0.0)
            return MixtureDesign(alphas=alphas, thetas=np.stack([-th, np.zeros(n_steps), th]),
                                 rho=rho)
        harmonic = 1.0 + np.sum(1.0 / np.arange(1, n_steps))     # 1 + H_{n-1}
        c = np.sqrt(self.cost * r2 / harmonic)
        th = c / np.sqrt(np.maximum(t, dt))
        thetas = np.stack([-th, np.zeros(n_steps), th])
        return MixtureDesign(alphas=alphas, thetas=thetas, rho=rho)


DESIGNS = ([TiltDesign(s, float(c)) for s in ("constant", "inverse_sqrt") for c in (1, 3, 9)]
           + [TiltDesign("front", 3.0), TiltDesign("front", 8.0)])
UNTILTED = "none"


def design_by_name(name):
    """`"none"` is the untilted arm (None); anything else must be one of DESIGNS."""
    if name == UNTILTED:
        return None
    for d in DESIGNS:
        if d.name == name:
            return d
    raise KeyError(f"unknown tilt design {name!r}; choose from {[d.name for d in DESIGNS]}")
