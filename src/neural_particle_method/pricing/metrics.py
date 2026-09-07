"""IV error metrics, and exact SSVI target IVs at the times reprice_iv actually reaches."""
import numpy as np

from ..market.ssvi import implied_vol_ssvi
from .reprice import snap_times


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


def target_ivs(ssvi_params, k_grid, maturities, n_steps):
    """Exact SSVI implied vols at the times reprice_iv actually reaches."""
    ts = snap_times(maturities, n_steps)
    return np.stack([implied_vol_ssvi(ssvi_params, k_grid, t) for t in ts])
