"""Conditional-expectation estimators sharing the Estimator interface."""
from .base import Estimator
from .bins import Bins
from .kalman import KalmanHead
from .nadaraya_watson import NadarayaWatson
from .nn import NNRegressor
from .ridge import GlobalRidge, RidgeHead, SliceRidge
from .rkhs import RKHSRidge
from .spline import PSpline

NAMES = ("nn", "nw", "ridge", "spline", "bins", "rkhs")


def make_estimator(name, seed=0, first_steps=400, later_steps=120):
    if name == "nn":
        return NNRegressor(seed=seed, first_steps=first_steps, later_steps=later_steps)
    if name == "nw":
        return NadarayaWatson()
    if name == "ridge":
        return SliceRidge(seed=seed, body_steps=first_steps)
    if name == "spline":
        return PSpline()
    if name == "bins":
        return Bins()
    if name == "rkhs":
        return RKHSRidge()
    raise KeyError(f"unknown estimator {name!r}; choose from {NAMES}")


__all__ = [
    "NAMES",
    "Bins",
    "Estimator",
    "GlobalRidge",
    "KalmanHead",
    "NNRegressor",
    "NadarayaWatson",
    "PSpline",
    "RKHSRidge",
    "RidgeHead",
    "SliceRidge",
    "make_estimator",
]
