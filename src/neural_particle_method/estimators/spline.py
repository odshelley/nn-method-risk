"""Penalised cubic B-spline (P-spline) conditional-expectation estimator."""
import numpy as np


def spline_estimate(lnx, v, lnx_grid, n_knots=25, lam=1.0, degree=3):
    """Penalised cubic B-spline (P-spline) estimate of E[V | ln X = .] on a grid."""
    from scipy.interpolate import BSpline
    q = np.quantile(lnx, np.linspace(0.005, 0.995, n_knots))
    q = np.unique(q)
    t = np.concatenate([[q[0]] * degree, q, [q[-1]] * degree])
    lo, hi = t[degree], t[-degree - 1]
    x = np.clip(lnx, lo, hi - 1e-12)
    B = BSpline.design_matrix(x, t, degree).toarray()
    p = B.shape[1]
    D2 = np.diff(np.eye(p), n=2, axis=0)
    c = np.linalg.solve(B.T @ B + lam * (D2.T @ D2), B.T @ v)
    Bg = BSpline.design_matrix(np.clip(lnx_grid, lo, hi - 1e-12), t, degree).toarray()
    return Bg @ c


class PSpline:
    supports_weights = False

    def __init__(self, n_knots=25, lam=1.0, degree=3):
        self.n_knots, self.lam, self.degree = n_knots, lam, degree

    def fit_predict(self, t, lnx, v, grid, weights=None):
        if weights is not None:
            raise ValueError("spline estimator does not support importance weights")
        return spline_estimate(lnx, v, grid, n_knots=self.n_knots, lam=self.lam, degree=self.degree)
