"""Cold suite: one calibration from scratch per (scenario, algorithm, seed) at the online budget."""
from ..bench.runner import run_one
from ..bench.scenarios import full_registry
from ..estimators.recipes import load_recipe, recipe_hash
from .artifacts import save_model
from .config import COLD_ALGOS, FULL

__all__ = ["COLD_ALGOS", "run_cold"]

NN_ALGOS = ("explicit_nn", "explicit_nn_tuned", "explicit_nn_opt")
RECIPE_ALGO = "explicit_nn_opt"


def cold_extra_key(algo):
    """The searched network is recipe-driven, so its cold runs carry the promoted recipe's hash.

    Without it `find_finished` matches by containment and a second promotion would read back the
    first recipe's runs, exactly as for the offline bodies and the online cells.
    """
    if algo != RECIPE_ALGO:
        return None
    return {"recipe_hash": recipe_hash(load_recipe("explicit_opt"))}


def run_cold(store, sid, algo, seed, settings=FULL):
    sc = full_registry()[sid]
    knobs = {"keep_slice_weights": True} if algo in NN_ALGOS else None

    def save(h, res):
        save_model(h, res.model, {"sid": sid, "algo": algo, "seed": seed, "T": sc.T,
                                  "n_steps": settings.explicit.n_steps,
                                  "n_particles": settings.n_online})

    return run_one(store, sid, algo, settings.n_online, seed, settings.explicit, settings.implicit,
                   settings.reprice, experiment=settings.experiment("suite_cold"), knobs=knobs,
                   extra_key=cold_extra_key(algo), save_model=save)
