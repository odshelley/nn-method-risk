import numpy as np

from neural_particle_method.estimators.ridge import RidgeHead


def _feat(t, lnx):
    return np.stack([lnx, lnx ** 2, np.ones_like(lnx)], axis=1)


def test_ridge_solution_matches_closed_form_without_residual():
    rng = np.random.default_rng(0)
    lnx = rng.normal(0, 0.3, 2_000)
    v = 0.04 + 0.02 * lnx + rng.normal(0, 1e-3, 2_000)
    head = RidgeHead(_feat, lam=1e-3, residual=False)
    g = np.linspace(-0.2, 0.2, 5)
    f = head.fit_predict(0.0, lnx, v, g)
    A = _feat(0.0, lnx)
    lam = 1e-3 * len(lnx)
    w = np.linalg.solve(A.T @ A + lam * np.eye(3), A.T @ v)
    np.testing.assert_allclose(f, _feat(0.0, g) @ w, rtol=1e-10)
    np.testing.assert_allclose(head.w_prev, w, rtol=1e-10)


def test_residual_centres_on_previous_coefficients():
    rng = np.random.default_rng(1)
    lnx = rng.normal(0, 0.3, 1_000)
    v = np.full(1_000, 0.05)
    head = RidgeHead(_feat, lam=10.0, residual=True)
    head.fit_predict(0.0, lnx, v, lnx[:3])
    w1 = head.w_prev.copy()
    head.fit_predict(0.1, lnx, v + 0.01, lnx[:3])
    # with a huge lam the second solve barely moves from w1 (shrinks toward it, not toward 0)
    assert np.abs(head.w_prev - w1).max() < 0.02
