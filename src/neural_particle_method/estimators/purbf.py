"""Partition-of-unity radial basis functions (Hakala 2019, Frontiers in AI 2:4).

f(x) = sum_j w_j psi_j(x),  psi_j(x) = K_{h_j}(x - c_j) / sum_k K_{h_k}(x - c_k),  K_h(r) = exp(-r^2 / (2 h^2)).
Centres: cloud min and max plus a random subset, pruned so no centre lies within prune * h_j of an
already kept one; h_j = mean distance to the n_neighbours nearest centres. Weights from ridge
normal equations (A^T W A + lam I) w = A^T W y with A_ij = psi_j(x_i). Defaults are his preferred
configuration: 40 centres, 5-NN widths, regulariser 0.2; the pruning constant is never stated in
the paper and is a free knob here. His printed solution has -lambda in the ridge normal equations,
which is inconsistent with his own least-squares loss; we implement +lambda.
"""
import numpy as np


class PURBF:
    supports_weights = True

    def __init__(self, n_centres=40, n_neighbours=5, lam=0.2, prune=1.0, seed=0):
        self.n_centres, self.n_neighbours, self.lam, self.prune, self.seed = n_centres, n_neighbours, lam, prune, seed

    def centres_and_widths(self, lnx):
        rng = np.random.default_rng(self.seed)
        lo, hi = float(lnx.min()), float(lnx.max())
        n_rand = max(self.n_centres - 2, 0)
        cand = np.concatenate([[lo, hi], rng.choice(lnx, size=min(n_rand, len(lnx)), replace=False)])
        cand = np.unique(cand)                       # sorted; lo first, hi last
        # widths from the candidate set, then greedy pruning in sorted order; the extremes always stay
        h_all = self._widths(cand)
        kept, kept_h = [], []
        for c, h in zip(cand, h_all):
            if c in (lo, hi) or not kept or np.abs(np.array(kept) - c).min() / h > self.prune:
                kept.append(c); kept_h.append(h)
        c, h = np.array(kept), np.array(kept_h)
        if len(c) <= self.n_neighbours:
            h = np.full(len(c), self._global_width(lnx))
        return c, h

    def _widths(self, c):
        if len(c) <= 1:
            return np.ones(len(c))
        d = np.abs(c[:, None] - c[None, :])
        d[np.eye(len(c), dtype=bool)] = np.inf
        k = min(self.n_neighbours, len(c) - 1)
        return np.sort(d, axis=1)[:, :k].mean(axis=1)

    @staticmethod
    def _global_width(lnx):
        return (4.0 * np.std(lnx) ** 5 / (3.0 * len(lnx))) ** 0.2

    def _basis(self, x, c, h):
        K = np.exp(-0.5 * ((x[:, None] - c[None, :]) / h[None, :]) ** 2)
        return K / np.clip(K.sum(axis=1, keepdims=True), 1e-300, None)

    def fit_predict(self, t, lnx, v, grid, weights=None, ctx=None):
        c, h = self.centres_and_widths(lnx)
        A = self._basis(lnx, c, h)
        wv = np.ones(len(lnx)) if weights is None else weights
        lhs = A.T @ (wv[:, None] * A) + self.lam * np.eye(len(c))
        rhs = A.T @ (wv * v)
        w = np.linalg.solve(lhs, rhs)
        return self._basis(grid, c, h) @ w
