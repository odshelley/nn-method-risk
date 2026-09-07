"""Scenario registry: SSVI surface draws paired with Heston dynamics."""
import dataclasses
from dataclasses import dataclass

import numpy as np

from ..market.ssvi import SSVIParams, no_arb_ok
from ..simulate.dynamics import HestonParams

XIS, RHOS, KAPPAS = (0.3, 0.6, 1.0), (-0.7, -0.3), (1.0, 2.0, 3.0)


@dataclass(frozen=True)
class ScenarioSpec:
    sid: str
    ssvi: SSVIParams
    dynamics: HestonParams
    s0: float = 1.0
    T: float = 2.0
    maturities: tuple = (0.25, 0.5, 1.0, 2.0)


def quote_k_grid():
    return np.log(np.geomspace(0.6, 1.6, 13))


def _draw_ssvi(rng):
    for _ in range(100):
        p = SSVIParams(sigma0=rng.uniform(0.12, 0.35), eta=rng.uniform(0.5, 2.0),
                       gamma=rng.uniform(0.3, 0.5), rho=rng.uniform(-0.85, -0.1))
        if no_arb_ok(p):
            return p
    raise RuntimeError("no-arb resampling exhausted")


def make_registry():
    reg = {}
    for i in range(1, 21):
        rng = np.random.default_rng(1000 + i)
        p = _draw_ssvi(rng)
        dyn = HestonParams(kappa=float(rng.choice(KAPPAS)), theta=p.sigma0 ** 2,
                           xi=float(rng.choice(XIS)), rho=float(rng.choice(RHOS)),
                           v0=p.sigma0 ** 2)
        reg[f"s{i:02d}"] = ScenarioSpec(f"s{i:02d}", p, dyn)
    return reg


def fig3_registry():
    base = make_registry()["s01"]
    reg = {}
    for xi in XIS:
        for rho in RHOS:
            sid = f"f_xi{xi}_rho{rho}"
            dyn = dataclasses.replace(base.dynamics, xi=xi, rho=rho)
            reg[sid] = ScenarioSpec(sid, base.ssvi, dyn)
    return reg


def full_registry():
    return {**make_registry(), **fig3_registry()}
