"""One Euler step of (ln X, V) under a Heston variance and a leverage value per particle.

Never draws random numbers: callers pass the two standard normals so RNG order is theirs.
The expressions are verbatim from the pre-refactor loops; keep the operation order.
"""
import numpy as np


def heston_step(lnx, v, L_p, zb, zp, params, dt, sdt, theta_p=None):
    """Return (lnx_new, v_new). `theta_p` is the optional per-particle mixture tilt."""
    rho, kappa, theta, xi = params.rho, params.kappa, params.theta, params.xi
    z1 = rho * zb + np.sqrt(1 - rho ** 2) * zp
    vp = np.maximum(v, 0.0)
    drift_x = -0.5 * L_p ** 2 * vp
    if theta_p is not None:
        drift_x = drift_x + L_p * np.sqrt(vp) * theta_p
    lnx_new = lnx + drift_x * dt + L_p * np.sqrt(vp) * sdt * z1
    v_new = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp) * sdt * zb
    return lnx_new, v_new
