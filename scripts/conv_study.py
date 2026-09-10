"""Body-convergence study: full-cloud fits, paper vs converged offline budgets, six scenarios.

Experiments (suffix per arm; pde_reference is shared and already populated at 200 steps):
  suite_cold_full      NW and spline cold, fit on the full 100k cloud
  suite_offline_full / suite_lagged_full   paper training budget, full-cloud online sweeps
  suite_offline_conv / suite_lagged_conv   converged bodies, full-cloud online sweeps
"""
import sys
from dataclasses import replace

from neural_particle_method.calibrate.config import ExplicitConfig, ImplicitConfig
from neural_particle_method.suite.cold import run_cold
from neural_particle_method.suite.config import FULL, SuiteSettings
from neural_particle_method.suite.grid import run_stage
from neural_particle_method.tracking.store import Store

SIDS = ("s01", "s02", "s05", "s09", "s11", "s16")
N_ONLINE = 100_000
JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 4

explicit_full = ExplicitConfig(n_steps=200, n_particles=N_ONLINE, fit_subsample=N_ONLINE,
                               fit_v_floor=True)
explicit_conv = replace(explicit_full, first_steps=2000, later_steps=1000)
implicit_full = ImplicitConfig(n_steps=200, n_particles=N_ONLINE, n_iters=30, alpha=0.5)
implicit_conv = replace(implicit_full, fit_steps=1000, pool_subsample=200_000)

base = dict(reprice=FULL.reprice, n_online=N_ONLINE, offline_sizes=(500_000,), seeds=(0, 1),
            sids=SIDS, n_x=FULL.n_x, n_v=FULL.n_v)
S_FULL = SuiteSettings(explicit=explicit_full, implicit=implicit_full, suffix="_full", **base)
S_CONV = SuiteSettings(explicit=explicit_conv, implicit=implicit_conv, suffix="_conv", **base)

if __name__ == "__main__":
    store = Store()
    for name in ("suite_cold_full", "suite_offline_full", "suite_lagged_full",
                 "suite_offline_conv", "suite_lagged_conv"):
        store.experiment_id(name)
    print("== cold full-cloud NW and spline", flush=True)
    for sid in SIDS:
        for algo in ("nw", "spline"):
            for seed in (0, 1):
                run_cold(store, sid, algo, seed, S_FULL)
                print("done: cold_full", sid, algo, seed, flush=True)
    for label, settings in (("full", S_FULL), ("conv", S_CONV)):
        for stage in ("offline", "online"):
            print(f"== {stage} {label}", flush=True)
            done, failed = run_stage(store, stage, settings, n_jobs=JOBS)
            print(f"{stage}_{label}: {done} done, {failed} failed", flush=True)
