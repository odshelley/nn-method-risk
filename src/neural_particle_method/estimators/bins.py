"""Equal-frequency bin conditional-expectation estimator (van der Stoep et al. via Li 2023)."""
import numpy as np


class Bins:
    """Piecewise-constant E[V | ln X] on n_bins equal-frequency buckets of the cloud."""
    supports_weights = True

    def __init__(self, n_bins=20):
        self.n_bins = n_bins

    def fit_predict(self, t, lnx, v, grid, weights=None):
        n = len(lnx)
        n_bins = min(self.n_bins, n)
        order = np.argsort(lnx, kind="stable")
        wv = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
        # equal-frequency edges from the sorted cloud; interior edges only
        counts = np.full(n_bins, n // n_bins)
        counts[: n % n_bins] += 1
        stops = np.cumsum(counts)
        vs, ws = v[order], wv[order]
        starts = np.concatenate([[0], stops[:-1]])
        num = np.add.reduceat(ws * vs, starts)
        den = np.add.reduceat(ws, starts)
        means = num / np.clip(den, 1e-300, None)
        edges = lnx[order][stops[:-1] - 1]  # right edge of each interior bucket, inclusive
        idx = np.searchsorted(edges, grid, side="left")
        return means[idx]
