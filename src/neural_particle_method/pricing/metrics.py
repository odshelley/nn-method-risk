"""IV error metrics, and exact SSVI target IVs at the times reprice_iv actually reaches."""
import numpy as np

from ..market.bs import bs_call
from ..market.ssvi import implied_vol_ssvi
from .reprice import FAR_PRICE_FLOOR, snap_times


def iv_metrics(iv_model, iv_target, k_grid, maturities, wing_cut=0.25, liquid_T=0.5):
    """RMSE/max IV errors in vol bp: pooled, wings-only, per maturity; NaNs counted and excluded.

    `liquid_mae_bp` is the MAE over the liquid quotes only: |k| <= `wing_cut` and T >= `liquid_T`.
    It is additive; every other key is what it was before the liquid mask existed.
    """
    err = (np.asarray(iv_model) - np.asarray(iv_target)) * 1e4
    ok = np.isfinite(err)
    wings = np.abs(np.asarray(k_grid)) > wing_cut

    def stats(mask):
        e = err[mask & ok]
        if e.size == 0:
            return float("nan"), float("nan")
        # round away float64 cancellation noise (~1e-13 bp), well below any real vol-bp difference
        return round(float(np.sqrt(np.mean(e ** 2))), 9), round(float(np.max(np.abs(e))), 9)

    def mae(mask):
        e = err[mask & ok]
        return float("nan") if e.size == 0 else round(float(np.mean(np.abs(e))), 9)

    pooled = stats(np.ones_like(ok, dtype=bool))
    wing_stats = stats(np.tile(wings, (err.shape[0], 1)))
    per = []
    for i, T in enumerate(maturities):
        row = np.zeros_like(ok, dtype=bool)
        row[i] = True
        r, mx = stats(row)
        per.append({"T": float(T), "rmse_bp": r, "max_bp": mx})
    all_mask = np.ones_like(ok, dtype=bool)
    wing_mask = np.tile(wings, (err.shape[0], 1))
    mae_per = []
    for i, T in enumerate(maturities):
        row = np.zeros_like(ok, dtype=bool)
        row[i] = True
        mae_per.append({"T": float(T), "mae_bp": mae(row)})
    liquid = np.zeros_like(ok, dtype=bool)
    for i, T in enumerate(maturities):
        if float(T) >= liquid_T:
            liquid[i] = ~wings
    # `per_maturity` is kept verbatim (golden replays compare it); MAE lives in its own keys.
    return {"pooled_rmse_bp": pooled[0], "pooled_max_bp": pooled[1],
            "wings_rmse_bp": wing_stats[0], "wings_max_bp": wing_stats[1],
            "n_failed": int((~ok).sum()), "per_maturity": per,
            "pooled_mae_bp": mae(all_mask), "wings_mae_bp": mae(wing_mask),
            "mae_per_maturity": mae_per, "liquid_mae_bp": mae(liquid)}


def far_wing_metrics(iv_model, iv_target, k_grid, times, s0, wing_cut=0.25,
                     price_floor=FAR_PRICE_FLOOR):
    """MAE in vol bp over the far-wing quotes that are priceable: |k| > wing_cut and target OTM
    price >= price_floor * s0. `times` are the snapped maturities the IVs were inverted at.

    Returns (metrics, err_bp) with err_bp NaN wherever a quote is dropped or failed."""
    k = np.asarray(k_grid, dtype=float)
    K = s0 * np.exp(k)
    tgt = np.asarray(iv_target, dtype=float)
    call = np.stack([bs_call(s0, K, float(t), tgt[i]) for i, t in enumerate(times)])
    otm = np.where(k[None, :] < 0, call - s0 + K[None, :], call)
    err = (np.asarray(iv_model, dtype=float) - tgt) * 1e4
    keep = (otm >= price_floor * s0) & np.isfinite(err)
    wings = np.tile(np.abs(k) > wing_cut, (len(times), 1))
    err = np.where(keep, err, np.nan)

    def mae(mask):
        e = err[mask & keep]
        return float("nan") if e.size == 0 else float(np.mean(np.abs(e)))

    out = {"far_wings_mae_bp": mae(wings), "far_wings_n": int((wings & keep).sum()),
           "far_all_mae_bp": mae(np.ones_like(keep))}
    for i, t in enumerate(times):
        row = np.zeros_like(keep)
        row[i] = True
        out[f"far_mae_bp/T{float(t):g}"] = mae(row & wings)
        out[f"far_n/T{float(t):g}"] = int((row & wings & keep).sum())
    return {kk: v for kk, v in out.items() if not (isinstance(v, float) and np.isnan(v))}, err


def target_ivs(ssvi_params, k_grid, maturities, n_steps):
    """Exact SSVI implied vols at the times reprice_iv actually reaches."""
    ts = snap_times(maturities, n_steps)
    return np.stack([implied_vol_ssvi(ssvi_params, k_grid, t) for t in ts])
