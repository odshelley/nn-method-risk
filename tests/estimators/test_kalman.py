import numpy as np
import torch

from neural_particle_method.estimators.kalman import KalmanHead, bspline_basis


class _QuadBody(torch.nn.Module):
    """Well-conditioned stand-in for a trained body: features (x, x^2) of the scaled log-spot."""
    def forward(self, tz):
        x = tz[:, 1:2]
        return torch.cat([x, x ** 2], dim=1)


class _FakeNet:
    body = _QuadBody()


def _head(**kw):
    return KalmanHead(_FakeNet(), T=1.0, **kw)


def test_zero_prior_reproduces_least_squares():
    rng = np.random.default_rng(0)
    lnx = rng.normal(0, 0.2, 2_000)
    v = 0.04 + 0.01 * lnx + rng.normal(0, 1e-3, 2_000)
    head = _head(n_prior=0, n_eff=1_000)
    g = np.linspace(-0.2, 0.2, 5)
    f = head.update(1, 0.5, lnx, v, g)
    A = head.features(0.5, lnx)
    w = np.linalg.lstsq(A, v, rcond=None)[0]
    np.testing.assert_allclose(f, head.features(0.5, g) @ w, rtol=1e-3, atol=1e-5)


def test_infinite_prior_leaves_beta_unchanged():
    rng = np.random.default_rng(1)
    lnx = rng.normal(0, 0.2, 1_000)
    v = np.full(1_000, 0.05)
    head = _head(n_prior=1e12, n_eff=1)
    beta0 = np.linspace(0.1, 0.2, head.features(0.5, lnx).shape[1])
    head.init_from({1: beta0})
    head.update(1, 0.5, lnx, v, lnx[:2])
    np.testing.assert_allclose(head.betas()[1], beta0, rtol=1e-8)


def test_bspline_basis_partition_of_unity():
    B = bspline_basis(np.linspace(-0.5, 0.5, 50), -0.6, 0.6, 10)
    assert B.shape == (50, 10) and np.allclose(B.sum(axis=1), 1.0)
