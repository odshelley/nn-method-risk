"""Sequential readouts on a frozen implicit-scheme body: Kalman/RLS head with optional spline block."""
import numpy as np
import torch

from .nn import Z_SCALE


def bspline_basis(x, lo, hi, n_bases, degree=3):
    from scipy.interpolate import BSpline
    inner = np.linspace(lo, hi, n_bases - degree + 1)
    knots = np.concatenate([np.full(degree, lo), inner, np.full(degree, hi)])
    return BSpline.design_matrix(np.clip(x, lo, hi + 0.0), knots, degree,
                                 extrapolate=False).toarray()


class KalmanHead:
    """Frozen body as a feature map with a per-slice RLS/Kalman readout, information form.

    State per slice k is (beta_k, Lambda_k) with Lambda the posterior precision.
    Everything is expressed in effective-sample units of the batch's own per-sample
    information I1 = A'A/(n r), which makes the balance scale-free: the overnight
    prior is worth n_prior effective samples, each batch contributes n_eff (particles
    within a batch share simulation error, so n_eff is far below the particle count),
    and the forgetting factor inflates the covariance between intraday steps so the
    head can track drift instead of freezing as evidence accumulates.
    """

    def __init__(self, net, T, n_eff=2000, n_prior=4000, forget=0.85, n_spline=0,
                 x_lo=np.log(0.4), x_hi=np.log(2.2)):
        self.net, self.T = net, T
        self.n_eff, self.n_prior, self.forget, self.n_spline = n_eff, n_prior, forget, n_spline
        self.x_lo, self.x_hi = x_lo, x_hi
        self.state = {}

    def features(self, t, lnx):
        with torch.no_grad():
            tz = torch.tensor(np.stack([np.full(len(lnx), t / max(self.T, 1e-9)),
                                        lnx / Z_SCALE], axis=1), dtype=torch.float32)
            phi = self.net.body(tz).numpy()
        blocks = [phi, np.ones((len(lnx), 1))]
        if self.n_spline:
            blocks.append(bspline_basis(lnx, self.x_lo, self.x_hi, self.n_spline))
        return np.concatenate(blocks, axis=1)

    _features = features

    def init_from(self, betas):
        for k, b in betas.items():
            beta = np.concatenate([b, np.zeros(self.n_spline)]) if self.n_spline else b.copy()
            self.state[k] = (beta, None)

    def update(self, k, t, lnx, v, lnx_grid):
        A = self.features(t, lnx)
        d = A.shape[1]
        beta, Lam = self.state.get(k, (np.zeros(d), None))
        n = len(v)
        r = max(float(np.mean((v - A @ beta) ** 2)), 1e-8)
        I1 = (A.T @ A) / (n * r)
        if Lam is None:
            Lam = self.n_prior * I1 + 1e-8 * np.eye(d)
        else:
            Lam = self.forget * Lam
        lhs = Lam + self.n_eff * I1
        beta = np.linalg.solve(lhs, Lam @ beta + (self.n_eff / (n * r)) * (A.T @ v))
        self.state[k] = (beta, lhs)
        return self.features(t, lnx_grid) @ beta

    def betas(self):
        return {k: b.copy() for k, (b, _) in self.state.items()}
