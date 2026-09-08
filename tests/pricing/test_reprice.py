import numpy as np

from neural_particle_method.pricing.reprice import RepriceConfig, reprice_iv, snap_times
from neural_particle_method.simulate.leverage import LeverageField

FLAT_DYN = {"kappa": 0.0, "theta": 0.04, "xi": 0.0, "rho": 0.0, "v0": 0.04}

def const_records(L0, T=1.0, n=4):
    g = np.linspace(-1.0, 1.0, 5)
    return LeverageField.from_records(
        [(i * T / n, g, np.full(5, L0), np.full(5, 0.04)) for i in range(n)])

def test_L_lookup_piecewise():
    field = const_records(1.5)
    assert np.allclose(field.at(0.3, np.array([0.0, 0.2])), 1.5)

def test_gbm_limit_recovers_flat_iv():
    # xi=0, kappa=0: V frozen at v0; L=1 gives GBM with vol sqrt(v0)=0.2
    ivs = reprice_iv(const_records(1.0), FLAT_DYN, s0=1.0, maturities=[0.5, 1.0],
                     k_grid=np.log(np.array([0.8, 1.0, 1.25])),
                     cfg=RepriceConfig(20_000, 12), seed=7)
    assert np.nanmax(np.abs(ivs - 0.2)) < 0.01

def test_reprice_iv_handles_snapshot_collision():
    # 0.48 and 0.52 both round to step 6 when T=max(maturities)=1.0, n_steps=12 (dt=1/12).
    # A naive step->single-maturity snap map drops one of them; both must come back finite.
    ivs = reprice_iv(const_records(1.0), FLAT_DYN, s0=1.0, maturities=[0.48, 0.52, 1.0],
                     k_grid=np.log(np.array([0.9, 1.0, 1.1])),
                     cfg=RepriceConfig(5_000, 12), seed=3)
    assert np.all(np.isfinite(ivs[0]))
    assert np.all(np.isfinite(ivs[1]))
    assert np.all(np.isfinite(ivs[2]))

def test_snap_times_matches_reprice_iv_grid():
    # T=0.5, n_steps=25 -> dt=0.02; 0.25/0.02=12.5 is deliberately off-grid.
    ts = snap_times([0.25, 0.5], n_steps=25, T=0.5)
    assert np.isclose(ts[0], 12 * 0.02)   # round(12.5) -> 12 (banker's rounding), not 0.25
    assert np.isclose(ts[1], 25 * 0.02)   # exactly on-grid already

def test_reprice_iv_inverts_at_snapped_time_not_requested_maturity():
    # Flat GBM (xi=0, kappa=0, v0=0.04 -> vol sqrt(v0)=0.2): since vol is constant,
    # the true IV is 0.2 at ANY evaluation time, so this isolates the inversion-time
    # bug cleanly. maturities=[0.25, 0.5] with n_steps=25 over T=max(maturities)=0.5
    # gives dt=0.02; 0.25/0.02=12.5 is deliberately off-grid (snaps to step 12,
    # t=0.24) while 0.5 lands exactly on step 25. Inverting the off-grid row's
    # price at the requested m=0.25 instead of the snapped t=0.24 mixes a price
    # simulated to 0.24 with a 0.25-maturity BS formula, producing a large biased
    # IV error; inverting at the snapped time removes it.
    ivs = reprice_iv(const_records(1.0, T=0.5, n=4), FLAT_DYN, s0=1.0,
                     maturities=[0.25, 0.5], k_grid=np.log(np.array([0.95, 1.0, 1.05])),
                     cfg=RepriceConfig(500_000, 25), seed=11)
    row_mean_err_bp = np.abs(ivs - 0.2).mean(axis=1) * 1e4
    assert row_mean_err_bp[0] < 15.0, f"misaligned row (m=0.25): {row_mean_err_bp[0]:.1f} bp"
    assert row_mean_err_bp[1] < 15.0, f"aligned row (m=0.5): {row_mean_err_bp[1]:.1f} bp"
