"""Black-Scholes pricing and implied volatility (r = q = 0)."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr


def bs_call(s0, K, T, sigma):
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    st = np.maximum(sigma * np.sqrt(T), 1e-12)
    d1 = (np.log(s0 / K) + 0.5 * st ** 2) / st
    return s0 * ndtr(d1) - K * ndtr(d1 - st)


def implied_vol(price, s0, K, T, lo=1e-4, hi=4.0):
    intrinsic = max(s0 - K, 0.0)
    if price <= intrinsic + 1e-12 or price >= s0:
        return np.nan
    try:
        return brentq(lambda sig: bs_call(s0, K, T, sig) - price, lo, hi, xtol=1e-10)
    except ValueError:
        return np.nan
