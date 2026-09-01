"""Fresh-seed repricing of vanillas under a calibrated leverage, and IV error metrics."""
import numpy as np

from .bs import implied_vol


def L_lookup(L_records):
    """Piecewise-constant-in-t, interp-in-lnx leverage function from calibration records."""
    ts = np.array([r[0] for r in L_records])

    def L(t, lnx):
        i = max(int(np.searchsorted(ts, t + 1e-12)) - 1, 0)
        _, grid, Lg, _ = L_records[i]
        if len(grid) == 1:
            return np.full_like(lnx, Lg[0])
        return np.interp(lnx, grid, Lg)

    return L


def reprice_iv(L_records, dynamics, s0, maturities, k_grid,
               n_particles=500_000, n_steps=100, seed=10_000):
    """Simulate fresh paths under L, price calls at each maturity, invert to IV."""
    kappa, theta, xi, rho, v0 = (dynamics[k] for k in ("kappa", "theta", "xi", "rho", "v0"))
    L = L_lookup(L_records)
    rng = np.random.default_rng(seed)
    T = max(maturities)
    dt, sdt = T / n_steps, np.sqrt(T / n_steps)
    snap = {int(round(m / dt)): m for m in maturities}
    lnx = np.full(n_particles, np.log(s0))
    v = np.full(n_particles, v0)
    ivs = np.full((len(maturities), len(k_grid)), np.nan)
    mat_idx = {m: i for i, m in enumerate(maturities)}
    for step in range(n_steps):
        t = step * dt
        L_p = L(t, lnx)
        zb = rng.standard_normal(n_particles)
        zp = rng.standard_normal(n_particles)
        z1 = rho * zb + np.sqrt(1 - rho ** 2) * zp   # W increment
        vp = np.maximum(v, 0.0)
        lnx = lnx + (-0.5 * L_p ** 2 * vp) * dt + L_p * np.sqrt(vp) * sdt * z1
        v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp) * sdt * zb
        if step + 1 in snap:
            m = snap[step + 1]
            x = np.exp(lnx)
            for j, k in enumerate(k_grid):
                K = s0 * np.exp(k)
                price = float(np.mean(np.maximum(x - K, 0.0)))
                ivs[mat_idx[m], j] = implied_vol(price, s0, K, m)
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
