"""Leverage field L(t, x) on per-slice grids, with the value E[V | ln X] used to build it."""
from dataclasses import dataclass

import numpy as np

DEFAULT_GRID = np.linspace(np.log(0.4), np.log(2.2), 81)


@dataclass(frozen=True)
class Slice:
    t: float
    grid: np.ndarray
    L: np.ndarray
    f: np.ndarray


class LeverageField:
    """Piecewise-constant in t, linear in ln x; a one-point grid means a constant slice."""

    def __init__(self, slices):
        self.slices = list(slices)
        self._times = np.array([s.t for s in self.slices])

    @property
    def times(self):
        return self._times

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, k):
        return self.slices[k]

    def __iter__(self):
        return iter(self.slices)

    def at(self, t, lnx):
        i = max(int(np.searchsorted(self._times, t + 1e-12)) - 1, 0)
        s = self.slices[i]
        if len(s.grid) == 1:
            return np.full_like(lnx, s.L[0])
        return np.interp(lnx, s.grid, s.L)

    def resample(self, grid):
        out = []
        for s in self.slices:
            if len(s.grid) > 1:
                L, f = np.interp(grid, s.grid, s.L), np.interp(grid, s.grid, s.f)
            else:
                L, f = np.full(len(grid), s.L[0]), np.full(len(grid), s.f[0])
            out.append(Slice(s.t, grid.copy(), L, f))
        return LeverageField(out)

    def L_matrix(self):
        return np.stack([s.L for s in self.slices])

    def to_records(self):
        return [(s.t, s.grid, s.L, s.f) for s in self.slices]

    @classmethod
    def from_records(cls, records):
        if isinstance(records, cls):
            return records
        return cls(Slice(float(t), np.asarray(g, dtype=float), np.asarray(L, dtype=float),
                         np.asarray(f, dtype=float)) for (t, g, L, f) in records)

    def to_json(self):
        return {"slices": [{"t": float(s.t), "grid": s.grid.tolist(), "L": s.L.tolist(),
                            "f": s.f.tolist()} for s in self.slices]}

    @classmethod
    def from_json(cls, d):
        return cls(Slice(float(s["t"]), np.array(s["grid"], dtype=float),
                         np.array(s["L"], dtype=float), np.array(s["f"], dtype=float))
                   for s in d["slices"])
