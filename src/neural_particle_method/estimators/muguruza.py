"""Conditional Monte Carlo (Muguruza 2019, Corollary 4.1): kernel-free and bandwidth-free.

Each particle contributes with its own one-step conditional density of log-spot given the vol
path, which under the harness's Euler step with frozen leverage is Gaussian with
    mean     = lnx_prev + (-1/2 L^2 v + L sqrt(v) theta_p) dt + L sqrt(v) rho zb sqrt(dt)
    variance = (1 - rho^2) L^2 v dt.

A particle with s2 <= 1e-18 is degenerate (rho^2 -> 1 or the step vol -> 0): its one-step
conditional law of log-spot collapses to a point mass at its conditional mean mu_i (Corollary
4.1). Such a particle is kept and contributes its weight w_i to the denominator and w_i * V_i
to the numerator of the grid point nearest mu_i, rather than being dropped; the Gaussian
contribution from non-degenerate particles is unchanged.
"""
import numpy as np


class ConditionalMC:
    supports_weights = True
    needs_step_context = True

    def fit_predict(self, t, lnx, v, grid, weights=None, ctx=None):
        if ctx is None:
            raise ValueError("ConditionalMC needs a StepContext (delivered by calibrate_explicit)")
        grid = np.asarray(grid, dtype=float)
        if len(grid) == 0:
            return np.array([])
        rho, dt = ctx.params.rho, ctx.dt
        vol = ctx.L_p * np.sqrt(ctx.v_prev)
        drift = -0.5 * ctx.L_p ** 2 * ctx.v_prev
        if ctx.theta_p is not None:
            drift = drift + vol * ctx.theta_p
        mu = ctx.lnx_prev + drift * dt + vol * rho * ctx.zb * np.sqrt(dt)
        s2 = (1.0 - rho ** 2) * vol ** 2 * dt
        w_all = np.ones(len(v)) if weights is None else weights

        den, num = np.zeros(len(grid)), np.zeros(len(grid))

        degenerate = s2 <= 1e-18
        if degenerate.any():
            mu_d, v_d, w_d = mu[degenerate], v[degenerate], w_all[degenerate]
            idx = np.clip(np.searchsorted(grid, mu_d), 0, len(grid) - 1)
            left = np.clip(idx - 1, 0, len(grid) - 1)
            idx = np.where(np.abs(grid[left] - mu_d) < np.abs(grid[idx] - mu_d), left, idx)
            np.add.at(den, idx, w_d)
            np.add.at(num, idx, w_d * v_d)

        keep = ~degenerate
        if keep.any():
            mu_k, s2_k, v_k, w_k = mu[keep], s2[keep], v[keep], w_all[keep]
            phi = np.exp(-0.5 * (grid[:, None] - mu_k[None, :]) ** 2 / s2_k[None, :]) / np.sqrt(s2_k)[None, :] * w_k[None, :]
            den = den + phi.sum(axis=1)
            num = num + phi @ v_k

        fallback = float(np.average(v, weights=w_all))
        return np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), fallback)
