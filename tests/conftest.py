"""Shared tiny configurations. Small enough that every algorithm runs in about a second."""
from neural_particle_method.calibrate.config import ExplicitConfig, ImplicitConfig

TINY = {"n_steps": 6, "fit_subsample": 4_000, "n_iters": 2,
        "first_steps": 80, "later_steps": 30, "fit_steps": 80}
TINY_EXPLICIT = ExplicitConfig(n_steps=6, fit_subsample=4_000, first_steps=80, later_steps=30)
TINY_IMPLICIT = ImplicitConfig(n_steps=6, n_iters=2, fit_steps=80)
TINY_N = 4_000
TINY_REPRICE_N = 8_000
TINY_REPRICE_STEPS = 8
