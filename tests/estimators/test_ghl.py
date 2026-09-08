import numpy as np
import pytest

from neural_particle_method.estimators import NAMES, make_estimator
from neural_particle_method.estimators.nadaraya_watson import GHLKernel


class FlatLV:
    T_grid = np.array([0.004, 1.0])

    def sigma(self, t, x, s0=1.0):
        return np.full(np.shape(np.asarray(x, dtype=float)), 0.2)


def _cloud(n=5_000, seed=0):
    rng = np.random.default_rng(seed)
    lnx = rng.normal(0.0, 0.2, n)
    return lnx, 0.04 + 0.02 * lnx


def test_registered_with_knobs_and_context():
    assert "nw_ghl" in NAMES
    est = make_estimator("nw_ghl", local_vol=FlatLV(), s0=1.0, c=2.0)
    assert isinstance(est, GHLKernel) and est.c == 2.0 and est.supports_weights


def test_bandwidth_rule():
    est = GHLKernel(local_vol=FlatLV(), s0=1.0)
    n = 10_000
    assert np.isclose(est.bandwidth(0.5, n, v=None), 1.5 * 0.2 * np.sqrt(0.5) * n ** (-0.2))
    assert np.isclose(est.bandwidth(0.1, n, v=None), 1.5 * 0.2 * np.sqrt(0.25) * n ** (-0.2))
    proxy = GHLKernel()
    assert np.isclose(proxy.bandwidth(0.5, n, v=np.full(n, 0.04)), 1.5 * 0.2 * np.sqrt(0.5) * n ** (-0.2))
    assert GHLKernel(fixed_scale=1.0).bandwidth(0.5, 10_000, None) == 10_000 ** -0.2


def test_constant_and_linear_recovery():
    lnx, v = _cloud()
    grid = np.linspace(-0.3, 0.3, 13)
    est = GHLKernel(local_vol=FlatLV())
    assert np.allclose(est.fit_predict(0.5, lnx, np.full_like(v, 0.05), grid), 0.05)
    f = est.fit_predict(0.5, lnx, v, grid)
    assert np.abs(f - (0.04 + 0.02 * grid)).max() < 2e-3


def test_empty_support_is_interpolated_not_nan():
    lnx, v = _cloud(n=200)
    grid = np.linspace(-2.0, 2.0, 21)
    f = GHLKernel(c=0.2, local_vol=FlatLV()).fit_predict(0.5, lnx, v, grid)
    assert np.all(np.isfinite(f))


def test_weights_are_applied():
    lnx, v = _cloud()
    est = GHLKernel(local_vol=FlatLV())
    w = (lnx > 0).astype(float)
    hi = est.fit_predict(0.5, lnx, v, np.array([0.0]), weights=w)[0]
    lo = est.fit_predict(0.5, lnx, v, np.array([0.0]), weights=1.0 - w)[0]
    assert hi > lo


def test_gaussian_variant_and_bad_kernel():
    lnx, v = _cloud()
    g = GHLKernel(kernel="gaussian", local_vol=FlatLV()).fit_predict(0.5, lnx, v, np.array([0.0, 0.1]))
    q = GHLKernel(kernel="quartic", local_vol=FlatLV()).fit_predict(0.5, lnx, v, np.array([0.0, 0.1]))
    assert np.all(np.isfinite(g)) and not np.allclose(g, q)
    with pytest.raises(ValueError):
        GHLKernel(kernel="triangle")


def test_nn_hidden_knob():
    est = make_estimator("nn", seed=0, hidden=16)
    assert est.net.head.in_features == 16
