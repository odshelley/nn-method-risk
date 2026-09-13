"""Importance-sampling study: does the Girsanov tilt buy budget equivalence?

Layer 1 (`tilt_slices`): particles under the frozen PDE leverage, no feedback, per-slice
estimates at the quoted strikes against the PDE conditional expectation. Layer 2: cold
calibration (`tilt_cold`) and the frozen body + spline head (`tilt_online`) with and without the
tilt, scored like section 4. Spec: docs/superpowers/specs/2026-09-13-importance-sampling-design.md.
"""
from ..bench.runner import run_one
from ..bench.scenarios import full_registry
from ..calibrate.importance import UNTILTED, design_by_name
from .budget import run_budget_cell
from .cold import NN_ALGOS, cold_extra_key
from .config import FULL, SSVI_SIDS
from .lag import LAGS

SLICE_SIDS = ("s01", "s02", "s05", "s09", "s11", "s16")
SLICE_PARTICLES = (10_000, 30_000, 80_000)
SLICE_SEEDS = (0, 1, 2, 3, 4)
COLD_PARTICLES = (10_000, 30_000, 80_000)
COLD_SEEDS = (0, 1)
COLD_ALGO_SIDS = {"nw": SSVI_SIDS, "explicit_nn_opt": SLICE_SIDS}
ONLINE_BUDGETS = (10_000, 80_000)
ONLINE_SEEDS = (0, 1)
ONLINE_METHOD = "explicit_opt_spline"


def mixture_for(design_name, n_steps, T, rho):
    """The MixtureDesign for a design name at this scenario's step count; None for "none"."""
    d = design_by_name(design_name)
    return None if d is None else d.mixture(n_steps, T, rho)


def run_tilt_cold(store, sid, algo, n_particles, seed, design_name, settings=FULL):
    """One cold calibration, tilted or not, scored as the cold suite is; keyed by `design`."""
    if algo not in COLD_ALGO_SIDS:
        raise ValueError(f"tilt cold rows are {tuple(COLD_ALGO_SIDS)}, got {algo!r}")
    sc = full_registry()[sid]
    mixture = mixture_for(design_name, settings.explicit.n_steps, sc.T, sc.dynamics.rho)
    knobs = {"keep_slice_weights": True} if algo in NN_ALGOS else None
    extra = {"design": design_name, **(cold_extra_key(algo) or {})}
    return run_one(store, sid, algo, int(n_particles), seed, settings.explicit, settings.implicit,
                   settings.reprice, experiment=settings.experiment("tilt_cold"), knobs=knobs,
                   extra_key=extra, mixture=mixture)


def _restrict(sids, wanted):
    return tuple(sids) if wanted is None else tuple(s for s in sids if s in set(wanted))


def cold_jobs(design_name, settings=FULL, sids=None, particles=COLD_PARTICLES, seeds=COLD_SEEDS):
    """Untilted and tilted cold cells, paired by seed, for both algorithms."""
    design_by_name(design_name)      # KeyError for an unknown design
    jobs = []
    for algo, algo_sids in COLD_ALGO_SIDS.items():
        for sid in _restrict(algo_sids, sids):
            for n in particles:
                for seed in seeds:
                    for d in (UNTILTED, design_name):
                        jobs.append(("cold", sid, algo, int(n), int(seed), d))
    return jobs


def run_tilt_online(store, sid, budget, lag, seed, design_name, settings=FULL, recipe=None):
    """Frozen searched body + weighted spline head on a tilted online cloud."""
    design_by_name(design_name)
    if design_name == UNTILTED:
        raise ValueError("the untilted online cells are the suite_budget_tuned cells")
    return run_budget_cell(store, sid, ONLINE_METHOD, int(budget), lag, int(seed),
                           body="explicit_opt", settings=settings, recipe=recipe,
                           design=design_name)


def online_jobs(design_name, settings=FULL, sids=None, budgets=ONLINE_BUDGETS, seeds=ONLINE_SEEDS):
    design_by_name(design_name)
    return [("online", sid, int(b), lag.kind, int(seed), design_name)
            for sid in _restrict(SSVI_SIDS, sids) for lag in LAGS
            for seed in seeds for b in budgets]
