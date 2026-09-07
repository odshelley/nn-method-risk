import numpy as np

from neural_particle_method.estimators.spline import PSpline


def test_recovers_quadratic():
    rng = np.random.default_rng(3)
    lnx = rng.uniform(-0.5, 0.5, 5_000)
    v = 0.04 + 0.1 * lnx ** 2
    g = np.linspace(-0.3, 0.3, 13)
    f = PSpline(lam=1e-3).fit_predict(0.0, lnx, v, g)
    assert np.abs(f - (0.04 + 0.1 * g ** 2)).max() < 1e-3
