"""Shim (removed in Task 12): old reprice_iv keyword signature over the new pricing subpackage."""
from .pricing.metrics import iv_metrics  # noqa: F401
from .pricing.reprice import RepriceConfig, mc_smile, snap_times  # noqa: F401
from .pricing.reprice import reprice_iv as _new
from .simulate.leverage import LeverageField


def L_lookup(L_records):
    return LeverageField.from_records(L_records).at


def reprice_iv(L_records, dynamics, s0, maturities, k_grid, n_particles=500_000, n_steps=200, seed=10_000):
    return _new(L_records, dynamics, s0, maturities, k_grid,
                RepriceConfig(n_particles=n_particles, n_steps=n_steps), seed=seed)
