"""A recipe is the dict of per-slice estimator knobs an Optuna trial samples. This module is the
only place a recipe becomes a regressor or an ExplicitConfig, so the search, the validation
bodies and the promoted algorithm cannot drift apart."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from .nn import NNRegressor

RECIPE_TYPES = {"hidden": int, "depth": int, "lr": float, "batch_size": int, "first_steps": int,
                "later_steps": int, "weight_decay": float, "fit_subsample": int,
                "warm_start": bool, "mean_match": bool, "monotone": bool,
                "monotone_penalty": float, "tail": str, "hetero": bool}
RECIPE_DIR = Path(__file__).parent / "recipes"


def _cast(typ, value):
    if typ is bool and isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return typ(value)


def coerce_recipe(d):
    """Cast a recipe read back from MLflow params (all strings) to its typed form."""
    out = {k: _cast(typ, d[k]) for k, typ in RECIPE_TYPES.items() if k in d}
    out.setdefault("monotone_penalty", 0.0)
    missing = set(RECIPE_TYPES) - set(out)
    if missing:
        raise KeyError(f"recipe is missing {sorted(missing)}")
    return out


def recipe_hash(recipe):
    return hashlib.sha1(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:10]


def regressor_from_recipe(recipe, seed=0, keep_slice_weights=False, monotone_sign=1.0):
    r = coerce_recipe(recipe)
    return NNRegressor(seed=seed, first_steps=r["first_steps"], later_steps=r["later_steps"],
                       hidden=r["hidden"], depth=r["depth"], lr=r["lr"],
                       batch_size=r["batch_size"] or None, weight_decay=r["weight_decay"],
                       warm_start=r["warm_start"], mean_match=r["mean_match"],
                       monotone_penalty=r["monotone_penalty"] if r["monotone"] else 0.0,
                       monotone_sign=monotone_sign, hetero=r["hetero"],
                       keep_slice_weights=keep_slice_weights)


def explicit_config_from_recipe(cfg, recipe):
    r = coerce_recipe(recipe)
    return replace(cfg, first_steps=r["first_steps"], later_steps=r["later_steps"],
                   tail=r["tail"], fit_subsample=min(r["fit_subsample"], cfg.n_particles))


def _path(name):
    return RECIPE_DIR / f"{name}.json"


def recipe_exists(name):
    return _path(name).exists()


def load_recipe(name):
    return coerce_recipe(json.loads(_path(name).read_text()))


def save_recipe(name, recipe):
    RECIPE_DIR.mkdir(parents=True, exist_ok=True)
    p = _path(name)
    p.write_text(json.dumps(coerce_recipe(recipe), indent=1, sort_keys=True) + "\n")
    return p
