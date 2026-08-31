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
