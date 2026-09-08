import numpy as np

from neural_particle_method.estimators import NAMES, make_estimator
from neural_particle_method.estimators.purbf import PURBF


def _cloud(n=4_000, seed=0):
    rng = np.random.default_rng(seed)
    lnx = rng.uniform(-0.6, 0.6, n)
    return lnx, 0.04 + 0.1 * lnx ** 2


def test_registered_defaults():
    assert "purbf" in NAMES
    est = make_estimator("purbf")
    assert isinstance(est, PURBF)
    assert (est.n_centres, est.n_neighbours, est.lam, est.prune) == (40, 5, 0.2, 1.0)


def test_centres_include_extremes_and_pruning_is_monotone():
    lnx, _ = _cloud()
    c, h = PURBF(n_centres=40, prune=1.0, seed=1).centres_and_widths(lnx)
    assert c.min() == lnx.min() and c.max() == lnx.max()
    assert 2 <= len(c) <= 40 and len(h) == len(c) and np.all(h > 0) and np.all(np.diff(c) > 0)
    n_loose = len(PURBF(n_centres=40, prune=0.5, seed=1).centres_and_widths(lnx)[0])
    n_tight = len(PURBF(n_centres=40, prune=2.0, seed=1).centres_and_widths(lnx)[0])
    assert n_tight <= len(c) <= n_loose


def test_recovers_quadratic_without_seams():
    lnx, v = _cloud()
    grid = np.linspace(-0.5, 0.5, 101)
    f = PURBF(lam=1e-6, seed=0).fit_predict(0.3, lnx, v, grid)
    assert np.abs(f - (0.04 + 0.1 * grid ** 2)).max() < 5e-3     # blended local constants: O(h^2) bias
    assert np.abs(np.diff(f, 2)).max() < 2e-3                    # no kink at centre boundaries


def test_constant_recovery_with_default_lambda():
    lnx, _ = _cloud()
    f = PURBF().fit_predict(0.3, lnx, np.full_like(lnx, 0.05), np.linspace(-0.5, 0.5, 11))
    assert np.abs(f - 0.05).max() < 5e-4


def test_weights_and_determinism():
    lnx, v = _cloud()
    grid = np.array([0.0])
    a = PURBF(seed=3).fit_predict(0.3, lnx, v, grid)
    b = PURBF(seed=3).fit_predict(0.3, lnx, v, grid)
    assert a == b
    w = (lnx > 0).astype(float)
    hi = PURBF(seed=3).fit_predict(0.3, lnx, v + 0.5 * (lnx > 0), grid, weights=w)[0]
    lo = PURBF(seed=3).fit_predict(0.3, lnx, v + 0.5 * (lnx > 0), grid, weights=1.0 - w)[0]
    assert hi > lo
