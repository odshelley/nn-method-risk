"""Conditional-expectation estimator interface: E[V | ln X = .] on a grid, one slice at a time."""
from typing import Protocol

import numpy as np


class Estimator(Protocol):
    supports_weights: bool

    def fit_predict(self, t: float, lnx: np.ndarray, v: np.ndarray, grid: np.ndarray,
                    weights: np.ndarray | None = None) -> np.ndarray: ...
