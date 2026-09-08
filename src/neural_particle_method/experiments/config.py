"""Experiment configurations (values verbatim from the legacy scripts)."""
from dataclasses import asdict, dataclass

from ..calibrate.config import ImplicitConfig
from ..pricing.reprice import RepriceConfig


def _flat(d):
    return {k: (str(v) if isinstance(v, (tuple, list)) else v) for k, v in d.items()}


@dataclass(frozen=True)
class WarmConfig:
    N: int = 200_000
    n_steps: int = 50
    sub: int = 30_000
    n_iters: int = 6
    fit_steps: int = 300
    reprice_N: int = 300_000
    reprice_steps: int = 200
    seq_len: int = 6
    xover_scales: tuple = (0.5, 1.0, 2.0, 4.0)
    norm_iters: int = 4

    def as_params(self):
        return _flat(asdict(self))

    @property
    def implicit(self):
        return ImplicitConfig(n_steps=self.n_steps, n_particles=self.N, alpha=0.5,
                              n_iters=self.n_iters, fit_steps=self.fit_steps)

    @property
    def reprice(self):
        return RepriceConfig(self.reprice_N, self.reprice_steps)


FULL = WarmConfig()
SMOKE = WarmConfig(N=20_000, n_steps=12, sub=8_000, n_iters=2, fit_steps=120,
                   reprice_N=60_000, reprice_steps=100, seq_len=3,
                   xover_scales=(1.0, 4.0), norm_iters=2)


@dataclass(frozen=True)
class BumpConfig:
    N: int = 200_000
    n_steps: int = 50
    sub: int = 30_000
    n_iters: int = 6
    fit_steps: int = 300
    fit_v_floor: bool = False   # fit explicit estimators on max(v, 0); False reproduces the paper runs
    explicit_est: str = "nw"    # estimator for the explicit-overnight strategies

    def as_params(self):
        return _flat(asdict(self))

    @property
    def implicit(self):
        return ImplicitConfig(n_steps=self.n_steps, n_particles=self.N, alpha=0.5,
                              n_iters=self.n_iters, fit_steps=self.fit_steps)

    @property
    def reprice(self):
        return RepriceConfig(300_000, 200)


BUMP_FULL = BumpConfig()
BUMP_SMOKE = BumpConfig(20_000, 12, 8_000, 2, 120)
