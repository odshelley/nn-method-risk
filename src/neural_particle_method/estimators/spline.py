"""Penalised cubic B-spline (P-spline) conditional-expectation estimator."""
import numpy as np


def spline_estimate(lnx, v, lnx_grid, n_knots=25, lam=1.0, degree=3, weights=None, knots=None):
    """Penalised cubic B-spline (P-spline) estimate of E[V | ln X = .] on a grid.

    `weights` (optional, positive) are importance weights: weighted least squares on the data
    term, penalty unchanged. `knots` overrides the quantile knots (tests)."""
    from scipy.interpolate import BSpline
    q = np.quantile(lnx, np.linspace(0.005, 0.995, n_knots)) if knots is None else np.asarray(knots)
    q = np.unique(q)
    t = np.concatenate([[q[0]] * degree, q, [q[-1]] * degree])
    lo, hi = t[degree], t[-degree - 1]
    x = np.clip(lnx, lo, hi - 1e-12)
    B = BSpline.design_matrix(x, t, degree).toarray()
    p = B.shape[1]
    D2 = np.diff(np.eye(p), n=2, axis=0)
    if weights is None:
        gram, rhs = B.T @ B, B.T @ v
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != np.shape(v) or np.any(w < 0):
            raise ValueError("spline weights must be non-negative and one per particle")
        gram, rhs = B.T @ (w[:, None] * B), B.T @ (w * v)
    c = np.linalg.solve(gram + lam * (D2.T @ D2), rhs)
    Bg = BSpline.design_matrix(np.clip(lnx_grid, lo, hi - 1e-12), t, degree).toarray()
    return Bg @ c


class PSpline:
    supports_weights = True

    def __init__(self, n_knots=25, lam=1.0, degree=3):
        self.n_knots, self.lam, self.degree = n_knots, lam, degree

    def fit_predict(self, t, lnx, v, grid, weights=None):
        return spline_estimate(lnx, v, grid, n_knots=self.n_knots, lam=self.lam,
                               degree=self.degree, weights=weights)
