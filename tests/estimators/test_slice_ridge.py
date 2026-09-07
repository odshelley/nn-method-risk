import numpy as np
from neural_particle_method.estimators.ridge import SliceRidge as RidgeHead


def test_ridgehead_reuses_body_standardisation_across_slices():
    """train_body fixes (mu, sd) once; fit_predict must not recompute them per
    slice, or w_prev's coefficients end up in a different standardised frame
    than the current slice's features and residual shrinkage is corrupted."""
    rng = np.random.default_rng(0)
    reg = RidgeHead(seed=0)

    lnx0 = rng.normal(0.0, 0.1, size=2_000)
    v0 = np.full_like(lnx0, 0.04) + rng.normal(0.0, 1e-3, size=2_000)
    reg.train_body(lnx0, v0, steps=50)
    mu0, sd0 = reg.mu, reg.sd
    grid = np.linspace(-0.2, 0.2, 21)
    f0 = reg.fit_predict(0.0, lnx0, v0, grid)

    # A later slice's cloud has shifted mean/spread; without the fix, fit_predict
    # would silently recompute (mu, sd) from this cloud instead of reusing the body's.
    lnx1 = rng.normal(0.3, 0.4, size=2_000)
    v1 = np.full_like(lnx1, 0.04) + rng.normal(0.0, 1e-3, size=2_000)
    w_prev_before = reg.w_prev.copy()
    f1 = reg.fit_predict(0.0, lnx1, v1, grid)

    # Reproduce fit_predict's ridge solve by hand, forcing (mu0, sd0). If fit_predict
    # instead standardised on lnx1's own mean/std (the bug this test guards against),
    # its A matrix -- and hence f1 -- would differ from this manual computation.
    A1 = reg.features(0.0, lnx1)
    lam = reg.lam * len(lnx1)
    lhs = A1.T @ A1 + lam * np.eye(A1.shape[1])
    rhs = A1.T @ v1 + lam * w_prev_before
    w1_manual = np.linalg.solve(lhs, rhs)
    f1_manual = reg.features(0.0, grid) @ w1_manual

    assert reg.mu == mu0 and reg.sd == sd0
    assert np.all(np.isfinite(f0)) and np.all(np.isfinite(f1))
    assert np.all(f1 > 0)
    assert np.allclose(f1, f1_manual, rtol=1e-6, atol=1e-8)
