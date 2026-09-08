from neural_particle_method.calibrate.config import ExplicitConfig
from neural_particle_method.experiments.config import BumpConfig, WarmConfig


def test_variance_floor_is_the_default_everywhere():
    assert ExplicitConfig().fit_v_floor is True
    assert BumpConfig().fit_v_floor is True
    assert WarmConfig().fit_v_floor is True
    assert WarmConfig(fit_v_floor=False).fit_v_floor is False
