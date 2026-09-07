"""Frozen configurations. Defaults are the pre-refactor defaults; do not change them."""
from dataclasses import asdict, dataclass


def _flat(d):
    return {k: (str(v) if isinstance(v, (tuple, list)) else v) for k, v in d.items()}


@dataclass(frozen=True)
class ExplicitConfig:
    n_steps: int = 50
    n_particles: int = 200_000
    fit_subsample: int = 30_000
    L_max: float = 4.0
    first_steps: int = 400
    later_steps: int = 120
    snapshot_times: tuple = ()

    def as_params(self):
        return _flat(asdict(self))


@dataclass(frozen=True)
class ImplicitConfig:
    n_steps: int = 50
    n_particles: int = 50_000
    alpha: float = 0.5
    n_iters: int = 6
    L_max: float = 4.0
    fit_steps: int = 300
    pool_subsample: int = 60_000

    def as_params(self):
        return _flat(asdict(self))
