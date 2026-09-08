"""Fresh-seed repricing of vanillas under a calibrated leverage field, and MC smiles from particles."""
from dataclasses import asdict, dataclass

import numpy as np

from ..market.bs import implied_vol
from ..simulate.dynamics import HestonParams
from ..simulate.leverage import LeverageField
from ..simulate.stepper import heston_step


@dataclass(frozen=True)
class RepriceConfig:
    n_particles: int = 500_000
    n_steps: int = 200

    def as_params(self):
        return asdict(self)


def snap_times(maturities, n_steps, T=None):
    """Grid-snapped time for each requested maturity, mirroring reprice_iv's dt exactly."""
    T = max(maturities) if T is None else T
    dt = T / n_steps
    return [int(round(m / dt)) * dt for m in maturities]  # noqa: RUF046


def reprice_iv(field, params, s0, maturities, k_grid, cfg=RepriceConfig(), seed=10_000):
    """Simulate fresh paths under the field, price calls at each maturity, invert to IV at the
    SNAPPED time (the time the simulated cloud actually reached)."""
    hp = HestonParams.from_dict(params)
    field = LeverageField.from_records(field)
    n_particles, n_steps = cfg.n_particles, cfg.n_steps
    rng = np.random.default_rng(seed)
    T = max(maturities)
    dt, sdt = T / n_steps, np.sqrt(T / n_steps)
    t_snap = dict(zip(maturities, snap_times(maturities, n_steps, T)))
    snap = {}
    for m in maturities:
        snap.setdefault(int(round(m / dt)), []).append(m)  # noqa: RUF046
    lnx = np.full(n_particles, np.log(s0))
    v = np.full(n_particles, hp.v0)
    ivs = np.full((len(maturities), len(k_grid)), np.nan)
    mat_idx = {m: i for i, m in enumerate(maturities)}
    for step in range(n_steps):
        t = step * dt
        L_p = field.at(t, lnx)
        zb = rng.standard_normal(n_particles)
        zp = rng.standard_normal(n_particles)
        lnx, v = heston_step(lnx, v, L_p, zb, zp, hp, dt, sdt)
        if step + 1 in snap:
            x = np.exp(lnx)
            for m in snap[step + 1]:
                for j, k in enumerate(k_grid):
                    K = s0 * np.exp(k)
                    price = float(np.mean(np.maximum(x - K, 0.0)))
                    ivs[mat_idx[m], j] = implied_vol(price, s0, K, t_snap[m])
    return ivs


def mc_smile(lnx_T, K_grid, s0=1.0, weights=None):
    """MC call prices from terminal particles; pass importance weights for a self-normalised mean."""
    x = np.exp(lnx_T)
    if weights is None:
        return np.array([np.mean(np.maximum(x - K, 0.0)) for K in K_grid])
    wn = weights / weights.sum()
    return np.array([np.sum(wn * np.maximum(x - K, 0.0)) for K in K_grid])
