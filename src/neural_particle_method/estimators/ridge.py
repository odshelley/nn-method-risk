"""Ridge readouts on frozen learned features, optionally centred on the previous slice's coefficients."""
import numpy as np
import torch

from .nn import Z_SCALE, SliceNet


class RidgeHead:
    """Per-slice ridge on features A = features_fn(t, lnx). Residual mode shrinks toward w_prev."""
    supports_weights = True

    def __init__(self, features_fn, lam=1e-3, residual=True):
        self._features_fn = features_fn
        self.lam, self.residual = lam, residual
        self.w_prev = None

    def features(self, t, lnx):
        return self._features_fn(t, lnx)

    def fit_predict(self, t, lnx, v, grid, weights=None):
        A = self.features(t, lnx)
        lam = self.lam * len(lnx)
        wv = np.ones(len(lnx)) if weights is None else weights
        lhs = A.T @ (wv[:, None] * A) + lam * np.eye(A.shape[1])
        rhs = A.T @ (wv * v)
        if self.residual and self.w_prev is not None:
            rhs = rhs + lam * self.w_prev
        w = np.linalg.solve(lhs, rhs)
        self.w_prev = w
        return self.features(t, grid) @ w


class SliceRidge(RidgeHead):
    """Frozen per-slice body (trained once on the first slice) + ridge readout. Ignores t."""

    def __init__(self, seed=0, lam=1e-3, residual=True, body_steps=400):
        torch.manual_seed(seed)
        self.net = SliceNet()
        self.body_steps = body_steps
        self.trained = False
        self.mu, self.sd = None, None
        super().__init__(self._slice_features, lam=lam, residual=residual)

    def train_body(self, lnx, v, steps=400):
        self.mu, self.sd = lnx.mean(), max(lnx.std(), 1e-6)
        z = torch.tensor(((lnx - self.mu) / self.sd)[:, None], dtype=torch.float32)
        tv = torch.tensor(v[:, None], dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-2)
        for _ in range(steps):
            opt.zero_grad()
            loss = ((self.net(z) - tv) ** 2).mean()
            loss.backward()
            opt.step()
        self.trained = True

    def _slice_features(self, t, lnx):
        with torch.no_grad():
            z = torch.tensor(((lnx - self.mu) / self.sd)[:, None], dtype=torch.float32)
            phi = self.net.body(z).numpy()
        return np.concatenate([phi, np.ones((len(lnx), 1))], axis=1)

    def fit_predict(self, t, lnx, v, grid, weights=None):
        if not self.trained:
            self.train_body(lnx, v, steps=self.body_steps)
        return super().fit_predict(t, lnx, v, grid, weights)


class GlobalRidge(RidgeHead):
    """Frozen implicit-scheme body as a (t, x) feature extractor with a per-slice ridge readout."""

    def __init__(self, net, T, lam=1e-3, residual=True):
        self.net, self.T = net, T
        super().__init__(self._global_features, lam=lam, residual=residual)

    def _global_features(self, t, lnx):
        with torch.no_grad():
            tz = torch.tensor(np.stack([np.full(len(lnx), t / max(self.T, 1e-9)),
                                        lnx / Z_SCALE], axis=1), dtype=torch.float32)
            phi = self.net.body(tz).numpy()
        return np.concatenate([phi, np.ones((len(lnx), 1))], axis=1)
