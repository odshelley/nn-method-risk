from neural_particle_method.simulate.dynamics import HestonParams


def test_round_trip_dict():
    d = {"kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7, "v0": 0.05}
    p = HestonParams.from_dict(d)
    assert p.kappa == 2.0 and p.v0 == 0.05
    assert p.to_dict() == d


def test_from_params_is_identity():
    p = HestonParams(1.0, 0.04, 0.3, -0.5, 0.04)
    assert HestonParams.from_dict(p) is p
