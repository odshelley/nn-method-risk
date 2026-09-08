"""Conditional-expectation estimators sharing the Estimator interface."""
from .base import Estimator
from .bins import Bins
from .kalman import KalmanHead
from .muguruza import ConditionalMC
from .nadaraya_watson import GHLKernel, NadarayaWatson
from .nn import NNRegressor
from .ridge import GlobalRidge, RidgeHead, SliceRidge
from .rkhs import RKHSRidge
from .spline import PSpline

NAMES = ("nn", "nw", "nw_ghl", "ridge", "spline", "bins", "rkhs", "muguruza")


def make_estimator(name, seed=0, first_steps=400, later_steps=120, local_vol=None, s0=1.0, **knobs):
    """Build an estimator by registry name. `knobs` go to the constructor (sensitivity sweeps and
    acceptance overrides); `local_vol` and `s0` are given to estimators that need the target surface."""
    if name == "nn":
        return NNRegressor(seed=seed, first_steps=first_steps, later_steps=later_steps, **knobs)
    if name == "nw":
        return NadarayaWatson(**knobs)
    if name == "nw_ghl":
        return GHLKernel(local_vol=local_vol, s0=s0, **knobs)
    if name == "ridge":
        return SliceRidge(seed=seed, body_steps=first_steps, **knobs)
    if name == "spline":
        return PSpline(**knobs)
    if name == "bins":
        return Bins(**knobs)
    if name == "rkhs":
        return RKHSRidge(**knobs)
    if name == "muguruza":
        return ConditionalMC(**knobs)
    raise KeyError(f"unknown estimator {name!r}; choose from {NAMES}")


__all__ = [
    "NAMES",
    "Bins",
    "ConditionalMC",
    "Estimator",
    "GHLKernel",
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
