"""What a lag is: a no-arbitrage surface bump, optionally with a sticky-strike spot move.

The particle cloud lives in absolute log-spot (`lnx` starts at log(s0)) and
`local_vol.sigma(t, x, s0)` takes moneyness relative to the `s0` it is passed. Under
sticky-strike the implied vol at each absolute strike is unchanged, but call prices at that
fixed strike change with the spot, so the Dupire local volatility is that of the translated
surface, `w~(k) = w(k + delta)` with `k` measured from the new forward; the particles also
start at the new spot. `TranslatedSSVILocalVol` (analytic SSVI surface) and the translated
`DupireSurface` construction below (grid-based Heston surface) both build this; `delta = 0`
reproduces the un-lagged surface exactly, so the surface-lag rows already in the store are
unaffected.
"""
import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from ..bench.scenarios import HestonMarketSpec, ScenarioSpec
from ..calibrate.warm import scaled_bump
from ..market.dupire import DupireSurface
from ..market.heston import heston_call, heston_iv
from ..market.local_vol import SSVILocalVol, dupire_local_vol
from ..market.ssvi import implied_vol_ssvi, w_and_derivs
from ..simulate.dynamics import HestonParams


@dataclass(frozen=True)
class Lag:
    kind: str          # "surface" | "surface_spot"
    spot_move: float   # log spot return; 0 for a pure surface lag


LAGS = (Lag("surface", 0.0), Lag("surface_spot", math.log(1.02)))


def bump_heston(m):
    """Heston-market analogue of the SSVI bump: one vol point on level, +0.03 on rho, -5% on xi."""
    b = HestonParams(kappa=m.kappa, theta=(math.sqrt(m.theta) + 0.01) ** 2, xi=0.95 * m.xi,
                     rho=min(m.rho + 0.03, -0.05), v0=(math.sqrt(m.v0) + 0.01) ** 2)
    assert b.xi > 0 and abs(b.rho) < 1 and b.v0 > 0 and b.theta > 0
    return b


class TranslatedSSVILocalVol:
    """Dupire local vol of the sticky-strike-translated SSVI surface w~(k) = w(k + delta),
    with moneyness k measured from the new spot. delta = 0 reproduces SSVILocalVol exactly."""

    def __init__(self, p, s0_new, delta, t_min=0.004, T_max=2.0):
        self.p, self.s0_new, self.delta = p, float(s0_new), float(delta)
        self.T_grid = np.array([t_min, T_max])

    @property
    def t_min(self):
        return float(self.T_grid[0])

    def sigma(self, t, x, s0=1.0):
        t = float(np.clip(t, self.T_grid[0], self.T_grid[-1]))
        k = np.log(np.asarray(x, dtype=float) / self.s0_new)
        w, dwdk, d2wdk2, dwdT = w_and_derivs(self.p, k + self.delta, T=t)
        return dupire_local_vol(w, dwdk, d2wdk2, dwdT, k)


@dataclass(frozen=True)
class LaggedScenario:
    sid: str
    base: object
    lag: Lag
    s0_ref: float
    s0: float
    T: float
    maturities: tuple
    dynamics: HestonParams
    family: str
    ssvi: object = None      # bumped SSVIParams for family "ssvi"
    market: object = None    # bumped HestonParams for family "heston"

    def _inner_local_vol(self):
        """The un-lagged (delta=0) surface, referenced to the old spot s0_ref."""
        if self.family == "ssvi":
            return SSVILocalVol(self.ssvi, self.s0_ref, T_max=self.T)
        b = self.base
        m = self.market
        T_grid = np.linspace(b.t_lo, b.T, b.n_t)
        k_grid = np.linspace(b.k_lo, b.k_hi, b.n_k)

        def price_fn(K, T):
            return heston_call(K, T, m.v0, m.kappa, m.theta, m.xi, m.rho, self.s0_ref)

        return DupireSurface.from_price_fn(price_fn, self.s0_ref, T_grid, k_grid)

    def local_vol(self):
        d = self.lag.spot_move
        if d == 0.0:
            return self._inner_local_vol()
        if self.family == "ssvi":
            return TranslatedSSVILocalVol(self.ssvi, self.s0, d, T_max=self.T)
        b = self.base
        m = self.market
        T_grid = np.linspace(b.t_lo, b.T, b.n_t)
        k_grid = np.linspace(b.k_lo, b.k_hi, b.n_k)
        w = np.stack([heston_iv(k_grid + d, T, m, self.s0_ref) ** 2 * T for T in T_grid])
        # fill any failed inversions by nearest neighbour along strike, as from_price_fn does
        for i in range(w.shape[0]):
            row = w[i]
            if np.isnan(row).any():
                idx = np.arange(len(row))
                good = ~np.isnan(row)
                w[i] = np.interp(idx, idx[good], row[good])
        return DupireSurface(T_grid, k_grid, w)

    def target_ivs(self, k_grid, maturities):
        k = np.asarray(k_grid, dtype=float) + self.lag.spot_move
        if self.family == "ssvi":
            return np.stack([implied_vol_ssvi(self.ssvi, k, t) for t in maturities])
        return np.stack([heston_iv(k, t, self.market, self.s0_ref) for t in maturities])

    def as_params(self):
        p = dict(self.base.as_params())
        p["scenario.s0"] = self.s0
        bumped = dataclasses.asdict(self.ssvi) if self.family == "ssvi" else self.market.to_dict()
        p.update({f"bumped.{k}": v for k, v in bumped.items()})
        p.update({"lag.kind": self.lag.kind, "lag.spot_move": self.lag.spot_move,
                  "s0_ref": self.s0_ref})
        return p


def lagged_scenario(sc, lag):
    """The intraday market S1 for registry scenario `sc` under `lag`."""
    s0_new = sc.s0 * math.exp(lag.spot_move)
    common = {"sid": sc.sid, "base": sc, "lag": lag, "s0_ref": sc.s0, "s0": s0_new, "T": sc.T,
              "maturities": tuple(sc.maturities), "dynamics": sc.dynamics, "family": sc.family}
    if isinstance(sc, ScenarioSpec):
        q, _ = scaled_bump(sc.ssvi, 1.0)
        return LaggedScenario(ssvi=q, **common)
    if isinstance(sc, HestonMarketSpec):
        return LaggedScenario(market=bump_heston(sc.market), **common)
    raise TypeError(f"unsupported scenario type {type(sc).__name__}")
