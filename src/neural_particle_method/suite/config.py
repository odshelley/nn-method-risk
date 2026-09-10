"""Suite settings: the agreed configuration (FULL), a two-minute SMOKE, and a test-sized tiny()."""
from dataclasses import dataclass

from ..bench.scenarios import heston_registry, make_registry
from ..calibrate.config import ExplicitConfig, ImplicitConfig
from ..pricing.reprice import RepriceConfig

SSVI_SIDS = tuple(make_registry())
HESTON_SIDS = tuple(heston_registry())
COLD_ALGOS = ("nw", "explicit_nn", "implicit_nn", "rkhs", "spline", "nw_ghl", "bins", "muguruza",
             "purbf")


@dataclass(frozen=True)
class SuiteSettings:
    explicit: ExplicitConfig
    implicit: ImplicitConfig
    reprice: RepriceConfig
    n_online: int
    offline_sizes: tuple
    seeds: tuple
    sids: tuple
    suffix: str = ""
    n_x: int = 801       # PDE reference grid
    n_v: int = 200

    def experiment(self, name):
        return name + self.suffix

    @classmethod
    def tiny(cls):
        """Seconds per stage; for unit tests."""
        return cls(ExplicitConfig(n_steps=4, n_particles=600, fit_subsample=300, first_steps=5,
                                  later_steps=2),
                   ImplicitConfig(n_steps=4, n_particles=600, n_iters=1, fit_steps=5,
                                  pool_subsample=300),
                   RepriceConfig(2_000, 8), n_online=600, offline_sizes=(800,), seeds=(0,),
                   sids=("s01", "li_simple"), suffix="_tiny", n_x=41, n_v=20)


FULL = SuiteSettings(
    ExplicitConfig(n_steps=200, n_particles=100_000, fit_subsample=100_000, fit_v_floor=True),
    ImplicitConfig(n_steps=200, n_particles=100_000, n_iters=30, alpha=0.5),
    RepriceConfig(500_000, 200),
    n_online=100_000, offline_sizes=(200_000, 500_000), seeds=(0, 1), sids=SSVI_SIDS + HESTON_SIDS)

SMOKE = SuiteSettings(
    ExplicitConfig(n_steps=10, n_particles=2_000, fit_subsample=1_000, first_steps=20,
                   later_steps=5, fit_v_floor=True),
    ImplicitConfig(n_steps=10, n_particles=2_000, n_iters=2, fit_steps=20, pool_subsample=2_000,
                   alpha=0.5),
    RepriceConfig(20_000, 10),
    n_online=2_000, offline_sizes=(4_000,), seeds=(0,), sids=("s01", "li_simple"), suffix="_smoke",
    n_x=201, n_v=60)
