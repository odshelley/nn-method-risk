"""Populates the `pde_reference` experiment with Fokker-Planck reference leverage fields.

Nothing else in the package writes to this experiment; `bench/runner.py::_leverage_error` reads
from it (by `sid`, `n_steps`, and `lag`) to compute `lev_rmse` for a bench/acceptance/sensitivity
run, when a reference happens to be present. `run_reference` is how a reference gets there.
"""
import numpy as np

from ..reference.pde import default_v_grid, default_x_grid, solve_leverage_pde
from ..tracking.store import git_hash
from .scenarios import full_registry

REFERENCE_EXPERIMENT = "pde_reference"


def run_reference(store, sid, n_steps=50, n_x=801, n_v=200, scenario=None, lag="none"):
    """Returns the run id of a finished pde_reference run for (sid, n_steps, lag), computing one
    if needed.

    `scenario` overrides the registry entry (a lagged scenario, say); `lag` is part of the key so
    references for the same sid under different lags never collide. Legacy runs without a `lag`
    param are not matched: they were all computed on the registry scenario at 50 steps and are
    superseded by re-running."""
    key = {"sid": sid, "n_steps": int(n_steps), "lag": lag}
    existing = store.find_finished(REFERENCE_EXPERIMENT, key)
    if existing is not None:
        return existing
    sc = full_registry()[sid] if scenario is None else scenario
    x_grid = default_x_grid(n=n_x, x0=float(np.log(sc.s0)))
    v_grid = default_v_grid(sc.dynamics, sc.T, n=n_v)
    result = solve_leverage_pde(sc.local_vol(), sc.dynamics, s0=sc.s0, T=sc.T, n_steps=n_steps,
                                x_grid=x_grid, v_grid=v_grid)
    params = {**key, "n_x": int(n_x), "n_v": int(n_v), "s0": sc.s0, "git_hash": git_hash()}
    with store.run(REFERENCE_EXPERIMENT, params) as h:
        h.log_json("leverage.json", result.field.to_json())
        h.log_metrics({"runtime_s": result.runtime_s, "mass": result.mass,
                       "forward": result.forward})
        return h.run_id
