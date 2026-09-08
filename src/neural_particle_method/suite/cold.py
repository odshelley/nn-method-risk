"""Cold suite: one calibration from scratch per (scenario, algorithm, seed) at the online budget."""
from ..bench.runner import run_one
from ..bench.scenarios import full_registry
from .artifacts import save_model
from .config import COLD_ALGOS, FULL

__all__ = ["COLD_ALGOS", "run_cold"]


def run_cold(store, sid, algo, seed, settings=FULL):
    sc = full_registry()[sid]
    knobs = {"keep_slice_weights": True} if algo == "explicit_nn" else None

    def save(h, res):
        save_model(h, res.model, {"sid": sid, "algo": algo, "seed": seed, "T": sc.T,
                                  "n_steps": settings.explicit.n_steps,
                                  "n_particles": settings.n_online})

    return run_one(store, sid, algo, settings.n_online, seed, settings.explicit, settings.implicit,
                   settings.reprice, experiment=settings.experiment("suite_cold"), knobs=knobs,
                   save_model=save)
