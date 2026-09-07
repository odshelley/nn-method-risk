"""Fresh-seed repricing of vanillas under a calibrated leverage, and IV error metrics."""
import numpy as np

from .bs import implied_vol
from .simulate.dynamics import HestonParams
from .simulate.stepper import heston_step


def L_lookup(L_records):
    """Piecewise-constant-in-t, interp-in-lnx leverage function from calibration records."""
    from .simulate.leverage import LeverageField
    return LeverageField.from_records(L_records).at


def snap_times(maturities, n_steps, T=None):
    """Grid-snapped time for each requested maturity, mirroring reprice_iv's dt exactly."""
    T = max(maturities) if T is None else T
    dt = T / n_steps
    return [int(round(m / dt)) * dt for m in maturities]


def reprice_iv(L_records, dynamics, s0, maturities, k_grid,
               n_particles=500_000, n_steps=200, seed=10_000):
    """Simulate fresh paths under L, price calls at each maturity, invert to IV.

    Each requested maturity snaps to the nearest simulation step; the price is
    inverted at that SNAPPED time, not the requested maturity, since that is the
    time the simulated cloud actually reached.
    """
    hp = HestonParams.from_dict(dynamics)
    v0 = hp.v0
    L = L_lookup(L_records)
    rng = np.random.default_rng(seed)
    T = max(maturities)
    dt, sdt = T / n_steps, np.sqrt(T / n_steps)
    t_snap = dict(zip(maturities, snap_times(maturities, n_steps, T)))
    snap = {}
    for m in maturities:
        snap.setdefault(int(round(m / dt)), []).append(m)
    lnx = np.full(n_particles, np.log(s0))
    v = np.full(n_particles, v0)
    ivs = np.full((len(maturities), len(k_grid)), np.nan)
    mat_idx = {m: i for i, m in enumerate(maturities)}
    for step in range(n_steps):
        t = step * dt
        L_p = L(t, lnx)
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


def iv_metrics(iv_model, iv_target, k_grid, maturities, wing_cut=0.25):
    """RMSE/max IV errors in vol bp: pooled, wings-only, per maturity; NaNs counted and excluded."""
    err = (np.asarray(iv_model) - np.asarray(iv_target)) * 1e4
    ok = np.isfinite(err)
    wings = np.abs(np.asarray(k_grid)) > wing_cut

    def stats(mask):
        e = err[mask & ok]
        if e.size == 0:
            return float("nan"), float("nan")
        # round away float64 cancellation noise (~1e-13 bp), well below any real vol-bp difference
        return round(float(np.sqrt(np.mean(e ** 2))), 9), round(float(np.max(np.abs(e))), 9)

    pooled = stats(np.ones_like(ok, dtype=bool))
    wing_stats = stats(np.tile(wings, (err.shape[0], 1)))
    per = []
    for i, T in enumerate(maturities):
        row = np.zeros_like(ok, dtype=bool)
        row[i] = True
        r, mx = stats(row)
        per.append({"T": float(T), "rmse_bp": r, "max_bp": mx})
    return {"pooled_rmse_bp": pooled[0], "pooled_max_bp": pooled[1],
            "wings_rmse_bp": wing_stats[0], "wings_max_bp": wing_stats[1],
            "n_failed": int((~ok).sum()), "per_maturity": per}
