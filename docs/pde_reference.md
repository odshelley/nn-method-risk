# PDE reference solution for the Heston-type LSV calibration

Module: `neural_particle_method.reference` (`pde.py`, `fokker_planck.py`, `convergence.py`).
Design: `docs/superpowers/specs/2026-09-08-pde-reference-design.md`.
Entry point:

```python
from neural_particle_method.reference.pde import solve_leverage_pde, default_x_grid, default_v_grid, implied_vols

res = solve_leverage_pde(local_vol, params, s0=1.0, T=1.0, n_steps=50,
                         x_grid=default_x_grid(), v_grid=default_v_grid(params, T=1.0),
                         snapshot_times=snap_times(maturities, n_steps, T))
res.field                                   # LeverageField: per-slice grid, L, f = E[V | X]
iv = implied_vols(res.x_grid, res.density, k_grid, T, dx=res.dx)      # terminal marginal
iv_t = implied_vols(res.x_grid, res.snapshots[t], k_grid, t, dx=res.dx)
res.forward, res.mass                       # E[e^X_T] and mass left on the grid (boundary loss)
```

`local_vol` is any object with `.sigma(t, x, s0)` and `.T_grid` (`market/local_vol.py::LocalVol`);
`params` is `simulate/dynamics.py::HestonParams` or a dict.

## What it computes

The same calibration scheme as `calibrate_explicit`, with the particle cloud replaced by the
joint density of (X, V) = (ln S, V). At step k (t = k dt) the leverage
L(t, x) = sigma_Dup(t, x) / sqrt(E[V_t | X_t = x]) is built from the current density and
frozen over [t, t + dt); step 0 uses the one-point slice L = sigma(t_min, s0) / sqrt(v0)
exactly as the particle scheme does; clipping (f >= 1e-4, 0 <= L <= L_max = 4) is the
same. The density is evolved by the forward Kolmogorov equation of the model

    dX = -1/2 L^2 V dt + L sqrt(V) dW,   dV = kappa (theta - V) dt + xi sqrt(V) dB,   d<W,B> = rho dt

and E[V | X = x] is the ratio of the two v-integrals of the density, computed exactly on the
grid. So the reference carries the scheme's own time-discretisation bias (leverage frozen
per step, sigma_Dup sampled at the left endpoint) but none of the estimation error of the
particle methods. That bias is O(dt) and is quantified below (tables 2 and 3): it is what
the particle harness would converge to with infinitely many particles and a perfect
estimator at the same `n_steps`.

## Equation, boundary treatment, scaled variable

In conservation form, d phi/dt + d F_x/dx + d F_v/dv = 0 with

    F_x = -1/2 L^2 v phi - 1/2 d/dx(L^2 v phi) - 1/2 d/dv(rho xi L v phi)
    F_v = kappa (theta - v) phi - 1/2 d/dv(xi^2 v phi) - 1/2 d/dx(rho xi L v phi)

**v = 0.** Cozma, Mariapragassam and Reisinger (arXiv:1701.06001, Theorem 4, eq. 4.1)
impose `xi^2/2 d(v phi)/dv - kappa (theta - v) phi + rho xi v d(x alpha phi)/dx = 0` at
v = 0, which is F_v = 0: no probability flux through the boundary, total mass conserved.
This is the continuous-time counterpart of full truncation in `heston_step`: the truncated
Euler scheme converges to the CIR process, which is reflected at 0 when the Feller
condition fails (Li market: 2 kappa theta / xi^2 = 0.46). We use this zero-flux condition.
Vanishing conditions at the outer x faces and at v_max are implemented as absorbing
boundaries; what they absorb is reported in `PDEResult.mass` and `PDEResult.forward`.

**Scaled variable.** With beta = 2 kappa theta / xi^2 - 1 the density behaves like v^beta at
v = 0 (beta = -0.54 for the Li market, so phi is unbounded there). Following their
Corollary 5 we solve for p = phi v^(-beta), which is bounded; the fluxes become

    F_v = -v^(beta+1) [ kappa p + xi^2/2 dp/dv ] - 1/2 rho xi v^(beta+1) d(L p)/dx
    F_x = -1/2 v^(beta+1) [ L^2 p + d(L^2 p)/dx ] - 1/2 rho xi d(L v^(beta+1) p)/dv

so the zero-flux condition at v = 0 holds identically and both drift-diffusion parts have
constant coefficients.

## Discretisation

- **Finite volume, cell-centred**, unknowns are cell masses; p is constant per cell, and
  all v^beta weights enter through O(1) ratios of incomplete power integrals, so any beta
  (including the xi -> 0 limit) is numerically safe.
- **Scharfetter-Gummel** (exponentially fitted) fluxes for the drift-diffusion parts:
  exact for the CIR Gamma equilibrium in v and for the e^{-x} equilibrium in x,
  positivity preserving, mass conserving to round-off (interior column sums of the
  operator are zero to 1e-13), and the discrete forward E[e^X] is conserved exactly by
  construction (verified: forward - 1 is at the 1e-5 level, the cell-quadrature offset).
- **Mixed derivative** by the conservative corner-average stencil (each corner value the
  mean of the four adjacent cells; cross fluxes vanish at v = 0 exactly).
- **Time stepping**: per leverage step the 9-point operator is assembled once and
  factorised once (`scipy.sparse.linalg.splu`); Crank-Nicolson with a Rannacher start
  (backward Euler at half steps over the first leverage step, because the initial
  condition is a Dirac mass split bilinearly over four cells). `n_substeps=2` per leverage
  step; the time error is already negligible at n_steps = 50 (table 1). Backward Euler
  throughout (`scheme="be"`) is available and is clearly worse.
- **Grids**: `default_x_grid()` is uniform on [-3, 3] (801 cells, h = 0.0075) with ln s0
  on a centre; `default_v_grid(params, T)` has 200 cells, sinh-stretched towards 0, on
  [0, v_max], v_max = max(v0, theta) + 6 xi sqrt(max(v0, theta) T). The x domain must
  contain the whole right tail: mass absorbed at e^x = 20 costs 20 times its weight in
  every call price. On [ln 0.4 - 1.2, ln 2.2 + 0.6] the SSVI scenario s01 lost 2.5e-4 of
  mass by T = 2 and priced 30 bp low; on [-3, 3] the loss is 1e-5 and the error 2 bp.
- **Leverage slices** are defined on the resolved x range (marginal density above 1e-6 of
  its peak), the analogue of the particle scheme's quantile grid, and extrapolated as
  constants outside it. Below that level the mixed-derivative stencil leaves
  sign-alternating masses (1e-8 relative) that are harmless for pricing but not for a ratio.
- **Repricing** integrates (e^x - K)^+ exactly against the piecewise-constant marginal, so
  the payoff kink costs nothing; the cell-average quadrature error is O(h^2) (2e-6 in price
  at h = 0.005 against Black-Scholes).

## Validation

Tests in `tests/reference/`. The default suite keeps only the cheap ones (about 6 s: operator
conservation, CIR law, xi -> 0, pricing helpers); the repricing checks below run with
`uv run pytest tests/reference -m slow` (about 2 min):

- Operator: interior conservation to 1e-13; marginal non-negative; with L = 0 the
  v-marginal reproduces the CIR transition law (mean, variance, and the non-central
  chi-square CDF at four points to 3e-3) for both a Feller-satisfying and the Li set.
- Pure Heston (L = 1) reprices `heston_call` (table 1).
- (a) xi = 0.01, rho = 0, v0 = theta, flat sigma_Dup: |L - 1| < 5e-3 on the quote range in
  every slice (the residual is the physical O(xi^2) tilt of E[V | X]; with rho != 0 the
  tilt is O(rho xi x / theta) and reaches 10% at xi = 0.02, so rho = 0 is required).
- (b) Li Heston market -> `DupireSurface.from_price_fn(heston_call)` -> calibrated LSV
  reprices the market implied vols (table 2); the SSVI scenario s01 with the analytic
  local vol reprices the SSVI targets at the snapped maturities (table 3).
- (c) Grid refinement: `uv run python -m neural_particle_method.reference.convergence`
  regenerates the tables below.

## Accuracy and runtime (Apple Silicon, single core, `uv run`)

### 1. Pure Heston, Li parameters, T = 1: PDE error alone (rms | max, vol bp)

| n_x (h) | n_v | n_steps x n_sub | rms | max | runtime s |
|---|---|---|---|---|---|
| 201 (0.0300) | 50 | 50 x 2 |  9.88 | 27.34 | 1 |
| 401 (0.0150) | 100 | 50 x 2 |  2.72 |  6.49 | 5 |
| 801 (0.0075) | 200 | 50 x 2 |  0.82 |  1.71 | 27 |
| 801 (0.0075) | 200 | 100 x 2 |  0.82 |  1.74 | 52 |
| 801 (0.0075) | 200 | 50 x 8 |  0.82 |  1.74 | 30 |
| 1601 (0.0037) | 400 | 50 x 2 |  0.27 |  0.43 | 184 |

### 2. LSV calibrated to the Li Heston market, T = 1 (rms | max, vol bp)

Dynamics `cal`: kappa=1, theta=0.06, xi=0.35, rho=-0.5, v0=0.08. `Li`: calibrated model = market, so the exact leverage is 1.

| dynamics | n_x | n_v | n_steps | rms | max | forward - 1 | runtime s |
|---|---|---|---|---|---|---|---|
| cal | 401 | 100 | 50 |  9.13 | 13.18 | +6.1e-06 | 7 |
| cal | 801 | 200 | 50 |  8.26 | 10.14 | -1.2e-06 | 34 |
| cal | 801 | 200 | 100 |  4.15 |  5.15 | -1.2e-06 | 68 |
| cal | 801 | 200 | 200 |  2.12 |  2.90 | -1.2e-06 | 137 |
| cal | 801 | 200 | 400 |  1.17 |  1.87 | -1.2e-06 | 273 |
| Li | 401 | 100 | 50 |  2.01 |  4.89 | +8.2e-06 | 7 |
| Li | 801 | 200 | 50 |  0.97 |  1.25 | +1.1e-06 | 34 |
| Li | 801 | 200 | 100 |  0.69 |  0.95 | +1.1e-06 | 68 |
| Li | 801 | 200 | 200 |  0.61 |  0.88 | +1.1e-06 | 137 |
| Li | 801 | 200 | 400 |  0.58 |  0.88 | +1.1e-06 | 273 |

### 3. SSVI scenario s01 (analytic Dupire local vol), T = 2, per snapped maturity (rms | max, vol bp)

| n_x | n_v | n_steps | T=0.25 rms | max | T=0.5 rms | max | T=1 rms | max | T=2 rms | max | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 401 | 100 | 50 | 69.88 | 204.79 | 10.10 | 27.66 |  2.95 |  4.47 |  2.45 |  3.66 | 7 |
| 801 | 200 | 50 | 43.78 | 122.53 |  7.60 | 18.30 |  3.36 |  4.65 |  2.63 |  4.11 | 34 |
| 801 | 200 | 200 | 14.45 | 46.17 |  1.73 |  4.80 |  0.66 |  1.01 |  0.59 |  0.90 | 136 |
| 801 | 200 | 800 | 11.33 | 37.87 |  1.14 |  3.26 |  0.19 |  0.40 |  0.12 |  0.16 | 546 |

## Reading the tables

- **PDE error alone** (table 1, pure Heston at the Li parameters, Feller violated): the
  spatial error is second order when x and v are refined together; the reference grid
  (801 x 200) is at 1.7 bp max (0.8 rms) on the quote grid, 0.4 bp at twice the resolution, and the time stepping is
  converged at 50 steps.
- **Scheme bias** (tables 2 and 3): with the calibrated model different from the market, the
  IV error halves when n_steps doubles at fixed grid. This is the frozen-leverage
  left-endpoint bias of the explicit scheme, common to the particle method, and it is the
  floor any particle estimator can reach at that `n_steps`. It is largest at short
  maturities on the T = 2 scenarios (6 steps to T = 0.25 at n_steps = 50). When the
  calibrated model equals the market (`Li` rows) the exact leverage is 1, the bias
  vanishes, and the residual is the PDE error (0.9 bp max).
- **Short maturities on T = 2 scenarios** converge slowly in both dt and h: at T = 0.25 the
  quote grid reaches 3.6 standard deviations out, where the implied vol is extremely
  sensitive to the price, and the density there is a few cells wide at the first steps.
  Treat the reference as reliable to a few bp for T >= 0.5, and to about 10 bp rms /
  40 bp max at T = 0.25 even with 800 steps.
- For the comparison set, run the reference at the harness's `n_steps` to anchor the
  particle methods against the same scheme, and at 8x more steps to see the scheme bias.
