import pytest

from neural_particle_method.bench.scenarios import feller_ratio, full_registry, make_tuning_registry


def test_tuning_registry_is_stratified_deterministic_and_disjoint():
    reg = make_tuning_registry()
    assert list(reg) == [f"t{i:02d}" for i in range(1, 25)]
    fr = [feller_ratio(s.dynamics) for s in reg.values()]
    assert sum(f < 0.3 for f in fr) == 8 and sum(f > 0.7 for f in fr) == 8
    assert all(0.3 <= f <= 0.7 for f in fr if not (f < 0.3 or f > 0.7))
    again = make_tuning_registry()
    assert all(reg[k].ssvi == again[k].ssvi and reg[k].dynamics == again[k].dynamics for k in reg)
    assert not set(reg) & set(full_registry())


def test_feller_ratio():
    from neural_particle_method.simulate.dynamics import HestonParams
    # 2*kappa*theta/xi**2 == 1.0 in exact arithmetic; 0.4**2 is not exactly 0.16 in
    # float64, so the result differs from 1.0 by ~2e-16 regardless of operand order.
    ratio = feller_ratio(HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.5, v0=0.04))
    assert ratio == pytest.approx(1.0)
