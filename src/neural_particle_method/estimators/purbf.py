"""Partition-of-unity radial basis functions (Hakala 2019, Frontiers in AI 2:4).

f(x) = sum_j w_j psi_j(x),  psi_j(x) = K_{h_j}(x - c_j) / sum_k K_{h_k}(x - c_k),  K_h(r) = exp(-r^2 / (2 h^2)).
Centres: cloud min and max plus a random subset, pruned so no centre lies within prune * h_j of an
already kept one (pruning itself uses the candidate-set widths, since that is how the greedy pass
is defined); h_j is then recomputed on the surviving (kept) centres as the mean distance to the
n_neighbours nearest *kept* centres, so a centre's final width reflects its actual neighbours after
pruning rather than the denser candidate set. Weights from ridge normal equations
(A^T W A + lam I) w = A^T W y with A_ij = psi_j(x_i). Defaults are his preferred configuration: 40
centres, 5-NN widths, regulariser 0.2; the pruning constant is never stated in the paper and is a
free knob here. His printed solution has -lambda in the ridge normal equations, which is
inconsistent with his own least-squares loss; we implement +lambda.

Regulariser scale: the ridge term is `lam * I` on the normal equations of the unnormalised
squared loss sum_i (y_i - f(x_i))^2. Hakala's printed loss carries a 1/(2N) factor, under which
his lambda = 0.2 would correspond to 2 * N * lam here. The source is internally inconsistent on
this (see BASELINES.md Card 5), so this implementation keeps lam as the literal 0.2 on the
unnormalised loss and reports it as such.
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
        # greedy pruning in sorted order using the candidate-set widths; the extremes always stay
        h_cand = self._widths(cand)
        kept = []
        for c, h in zip(cand, h_cand):
            if c in (lo, hi) or not kept or np.abs(np.array(kept) - c).min() / h > self.prune:
                kept.append(c)
        c = np.array(kept)
        # widths are then recomputed on the surviving centres, so they reflect the post-pruning spacing
        h = self._widths(c) if len(c) > self.n_neighbours else np.full(len(c), self._global_width(lnx))
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
