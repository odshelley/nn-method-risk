import dataclasses

from neural_particle_method.calibrate.config import ExplicitConfig, ImplicitConfig


def test_defaults_match_pre_refactor_values():
    e, i = ExplicitConfig(), ImplicitConfig()
    assert (e.n_steps, e.n_particles, e.fit_subsample, e.L_max, e.first_steps, e.later_steps) == \
        (50, 200_000, 30_000, 4.0, 400, 120)
    assert (i.n_steps, i.n_particles, i.alpha, i.n_iters, i.L_max, i.fit_steps, i.pool_subsample) == \
        (50, 50_000, 0.5, 6, 4.0, 300, 60_000)


def test_frozen_and_replaceable():
    e = dataclasses.replace(ExplicitConfig(), n_steps=6)
    assert e.n_steps == 6
    try:
        e.n_steps = 7
        assert False
    except dataclasses.FrozenInstanceError:
        pass


def test_as_params_is_flat_and_stringifiable():
    p = ExplicitConfig(snapshot_times=(0.25, 0.5)).as_params()
    assert p["snapshot_times"] == "(0.25, 0.5)" and p["n_steps"] == 50
    assert all(isinstance(v, (str, int, float)) for v in p.values())
