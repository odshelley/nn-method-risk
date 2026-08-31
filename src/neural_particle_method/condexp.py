"""Conditional-expectation estimators on a particle cloud: NN with warm start, and Nadaraya-Watson."""
import numpy as np
import torch
import torch.nn as nn

V_SCALE = 0.04  # rough variance scale for output conditioning
Z_SCALE = 0.3   # rough log-spot scale for input conditioning


class SliceNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, z):
        return nn.functional.softplus(self.head(self.body(z))) * V_SCALE


class NNRegressor:
    """Warm-startable per-slice regressor for E[V | ln X = .]."""

    def __init__(self, seed=0):
        torch.manual_seed(seed)
        self.net = SliceNet()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-2)

    def fit(self, lnx, v, steps=120, weights=None):
        z = torch.tensor(lnx[:, None] / Z_SCALE, dtype=torch.float32)
        tv = torch.tensor(v[:, None], dtype=torch.float32)
        tw = None if weights is None else torch.tensor(weights[:, None], dtype=torch.float32)
        for _ in range(steps):
            self.opt.zero_grad()
            resid = (self.net(z) - tv) ** 2
            loss = (resid if tw is None else tw * resid).mean()
            loss.backward()
            self.opt.step()
        return float(loss.detach())

    def predict(self, lnx_grid):
        with torch.no_grad():
            z = torch.tensor(lnx_grid[:, None] / Z_SCALE, dtype=torch.float32)
            return self.net(z).numpy()[:, 0]


def nw_estimate(lnx, v, lnx_grid, weights=None, bandwidth=None):
    """Gaussian Nadaraya-Watson estimate of E[V | ln X = g] on a grid (optionally weighted)."""
    if bandwidth is None:
        bandwidth = 1.06 * np.std(lnx) * len(lnx) ** (-1 / 5)
    d = (lnx_grid[:, None] - lnx[None, :]) / bandwidth
    ker = np.exp(-0.5 * d ** 2)
    if weights is not None:
        ker = ker * weights[None, :]
    num = ker @ v
    den = ker.sum(axis=1)
    return num / np.clip(den, 1e-300, None)


class RidgeHead:
    """Frozen learned-feature body + per-slice ridge readout, optionally centred on the previous slice's coefficients (residual ridge)."""

    def __init__(self, seed=0, lam=1e-3, residual=True):
        torch.manual_seed(seed)
        self.net = SliceNet()
        self.lam, self.residual = lam, residual
        self.w_prev = None
        self.trained = False

    def train_body(self, lnx, v, steps=400):
        mu, sd = lnx.mean(), max(lnx.std(), 1e-6)
        z = torch.tensor(((lnx - mu) / sd)[:, None], dtype=torch.float32)
        tv = torch.tensor(v[:, None], dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-2)
        for _ in range(steps):
            opt.zero_grad()
            loss = ((self.net(z) - tv) ** 2).mean()
            loss.backward()
            opt.step()
        self.trained = True

    def _features(self, lnx, mu, sd):
        with torch.no_grad():
            z = torch.tensor(((lnx - mu) / sd)[:, None], dtype=torch.float32)
            phi = self.net.body(z).numpy()
        return np.concatenate([phi, np.ones((len(lnx), 1))], axis=1)

    def fit_predict(self, lnx, v, lnx_grid):
        mu, sd = lnx.mean(), max(lnx.std(), 1e-6)
        A = self._features(lnx, mu, sd)
        lam = self.lam * len(lnx)
        lhs = A.T @ A + lam * np.eye(A.shape[1])
        rhs = A.T @ v
        if self.residual and self.w_prev is not None:
            rhs = rhs + lam * self.w_prev
        w = np.linalg.solve(lhs, rhs)
        self.w_prev = w
        return self._features(lnx_grid, mu, sd) @ w


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
