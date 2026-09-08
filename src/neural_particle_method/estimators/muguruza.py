"""Conditional Monte Carlo (Muguruza 2019, Corollary 4.1): kernel-free and bandwidth-free.

Each particle contributes with its own one-step conditional density of log-spot given the vol
path, which under the harness's Euler step with frozen leverage is Gaussian with
    mean     = lnx_prev + (-1/2 L^2 v + L sqrt(v) theta_p) dt + L sqrt(v) rho zb sqrt(dt)
    variance = (1 - rho^2) L^2 v dt.
"""
import numpy as np


class ConditionalMC:
    supports_weights = True
    needs_step_context = True

    def fit_predict(self, t, lnx, v, grid, weights=None, ctx=None):
        if ctx is None:
            raise ValueError("ConditionalMC needs a StepContext (delivered by calibrate_explicit)")
        rho, dt = ctx.params.rho, ctx.dt
        vol = ctx.L_p * np.sqrt(ctx.v_prev)
        drift = -0.5 * ctx.L_p ** 2 * ctx.v_prev
        if ctx.theta_p is not None:
            drift = drift + vol * ctx.theta_p
        mu = ctx.lnx_prev + drift * dt + vol * rho * ctx.zb * np.sqrt(dt)
        s2 = (1.0 - rho ** 2) * vol ** 2 * dt
        keep = s2 > 1e-18
        mu, s2, vv = mu[keep], s2[keep], v[keep]
        w = np.ones(len(vv)) if weights is None else weights[keep]
        phi = np.exp(-0.5 * (grid[:, None] - mu[None, :]) ** 2 / s2[None, :]) / np.sqrt(s2)[None, :] * w[None, :]
        den, num = phi.sum(axis=1), phi @ vv
        fallback = float(np.average(vv, weights=w))
        return np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), fallback)
