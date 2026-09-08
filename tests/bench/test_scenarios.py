from neural_particle_method.bench.scenarios import (
    fig3_registry,
    full_registry,
    make_registry,
    quote_k_grid,
)
from neural_particle_method.market.ssvi import no_arb_ok
from neural_particle_method.simulate.dynamics import HestonParams


def test_registry_deterministic_and_arb_free():
    a, b = make_registry(), make_registry()
    assert list(a) == [f"s{i:02d}" for i in range(1, 21)]
    assert all(a[k].ssvi == b[k].ssvi for k in a)
    assert all(no_arb_ok(s.ssvi) for s in a.values())
    assert isinstance(next(iter(a.values())).dynamics, HestonParams)


def test_fig3_cross():
    f = fig3_registry()
    assert len(f) == 6
    assert len({(s.dynamics.xi, s.dynamics.rho) for s in f.values()}) == 6


def test_grids():
    k = quote_k_grid()
    assert len(k) == 13 and k[0] < 0 < k[-1]
    assert len(full_registry()) == 26
