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
    grid: str = "quantile"  # "quantile": per-slice cloud quantiles (paper); "fixed": DEFAULT_GRID
    fit_v_floor: bool = False  # fit the estimator on max(v, 0) (what the dynamics use) instead of the raw Euler v

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
    crn: bool = False   # common random numbers: re-seed the cloud and subsample identically every outer iteration
    lr: float = 1e-2
    alpha_decay: float = 0.0   # alpha_n = alpha / (1 + alpha_decay * n)
    average_last: int = 0      # return the mean of the last k leverage iterates (0: last iterate)
    reinit_net: bool = False   # fresh network and optimiser every outer iteration (deterministic map under crn)
    estimator: str = "net"    # "net": global network; "nw": per-slice Nadaraya-Watson (causal map, Lemma 1 check)
    nw_shift: str = "causal"  # per-slice data placement: "causal" (cloud at t_k), "next" (t_{k+1}), "mid" (average)
    nw_subsample: int = 30_000

    def as_params(self):
        return _flat(asdict(self))
