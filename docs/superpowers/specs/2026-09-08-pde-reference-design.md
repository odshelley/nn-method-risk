# PDE reference solution for the Heston-type LSV calibration — design

Date: 2026-09-08. Status: brief received from the neural-particle-code session on Osian's
behalf; design choices below were made in this session and are flagged for review.
Branch: `pde-reference` off `refactor-mlflow`.

## Purpose

An accuracy anchor for the baseline-comparison set. The particle harness estimates
E[V_t | X_t = x] from a cloud; the reference evolves the joint density of (X, V) with the
forward Kolmogorov equation on a grid and computes the conditional expectation by
quadrature. The leverage is updated once per time step and frozen over the step, exactly
as in `calibrate_explicit`, so the two methods discretise the same calibration scheme
and differ only in how E[V | X] is estimated.

## Model and equation

Log-spot x = ln S, r = q = 0:

    dX = -1/2 L(t,X)^2 V dt + L(t,X) sqrt(V) dW,   dV = kappa (theta - V) dt + xi sqrt(V) dB,
    d<W,B> = rho dt,   L(t,x) = sigma_Dup(t,x) / sqrt(E[V_t | X_t = x]).

Forward Kolmogorov equation for the density phi(t, x, v), in conservation form
d phi/dt + d F_x/dx + d F_v/dv = 0 with

    F_x = -1/2 L^2 v phi - 1/2 d/dx(L^2 v phi) - 1/2 d/dv(rho xi L v phi)
    F_v = kappa (theta - v) phi - 1/2 d/dv(xi^2 v phi) - 1/2 d/dx(rho xi L v phi)

(the mixed derivative is split half and half between the two fluxes).

**Boundary at v = 0.** Cozma, Mariapragassam and Reisinger (arXiv:1701.06001, Theorem 4,
eq. 4.1) impose the non-Dirichlet condition
`xi^2/2 d(v phi)/dv - kappa (theta - v) phi + rho xi v d(x alpha phi)/dx = 0` at v = 0,
which is F_v = 0: zero probability flux through v = 0, so mass is conserved. This is the
continuous-time counterpart of full truncation in the particle scheme: the truncated Euler
scheme converges to the CIR process, which is instantaneously reflected at 0 when the
Feller condition fails. We use the same condition. Vanishing (Dirichlet) conditions at
the far x boundaries and at v_max; mass leaking through them is reported.

**Scaled variable.** With beta = 2 kappa theta / xi^2 - 1, the density behaves like
v^beta near v = 0 (singular when Feller fails: the Li market has beta = -0.54). Following
Cozma et al. (Corollary 5) we write phi = v^beta p with p bounded. Then

    F_v = -v^(beta+1) [ kappa p + 1/2 xi^2 dp/dv ]          - 1/2 rho xi v^(beta+1) d(L p)/dx
    F_x = -1/2 v^(beta+1) [ L^2 p + d(L^2 p)/dx ]          - 1/2 rho xi d(L v^(beta+1) p)/dv

The zero-flux condition at v = 0 is now automatic (v^(beta+1) -> 0), and both
drift-diffusion parts have constant coefficients in the scaled variable.

## Discretisation

**Finite volume, cell-centred.** `x_grid` and `v_grid` are cell centres. Faces are
midpoints between centres; the first v face is at v = 0, the outer faces mirror the last
spacing. Unknowns are cell masses m_ij = integral of phi over the cell. The in-cell
reconstruction is p constant per cell, so m_ij = dx_i W_j p_ij with
W_j = integral of v^beta over cell j. Everything is expressed through the ratios
V_{j+1}^(beta+1) / W_j, V_{j+1}^(beta+1) / W_{j+1} and W'_j / W_j
(W'_j = integral of v^(beta+1)), which are O(1) for any beta; v^beta itself is never formed,
so the xi -> 0 limit (beta -> infinity) is numerically safe.

**Fluxes.** Drift-diffusion parts use the Scharfetter-Gummel (exponentially fitted)
flux `F = (D/h) [B(-w) u_left - B(w) u_right]`, `B(w) = w / (e^w - 1)`, `w = a h / D`, which is
exact for the exponential equilibrium of each direction (in v: p ~ exp(-2 kappa v / xi^2),
the Gamma stationary law of CIR; in x: u = L^2 p ~ e^{-x}), positivity preserving, and
reduces to upwinding where diffusion vanishes. The mixed term is discretised by the
conservative corner-average stencil: the integral of a v-derivative across a cell is the
difference of corner values, each corner value being the mean of the four adjacent cells.
Cross fluxes vanish at v = 0 exactly. Ghost cells with p = 0 implement the outer
Dirichlet conditions.

**Time stepping.** Within one leverage step of length dt = T / n_steps the operator A(L)
is fixed; it is assembled once (9-point stencil, scipy sparse) and factorised once with
`splu`. `n_substeps` theta-steps are taken per leverage step. `scheme="cn"` (default) is
Crank-Nicolson with a Rannacher start: the first leverage step is taken with backward
Euler at half the substep size (the initial condition is a Dirac mass, which
Crank-Nicolson alone would turn into oscillations). `scheme="be"` is backward Euler
throughout, for the convergence study.

**Initial condition.** Unit mass placed on the four cells around (ln s0, v0) with bilinear
weights so that the discrete mean is exact.

**Leverage update.** At step k, t = k dt: q_i = sum_j m_ij / dx_i is the marginal,
f_i = sum_j m_ij (W'_j / W_j) / sum_j m_ij is E[V | X = x_i]. Cells whose marginal mass is
below 1e-12 of the maximum take the nearest well-defined value. Then
L_i = clip(sigma_Dup(max(t, t_min), e^{x_i}) / sqrt(max(f_i, 1e-4)), 0, L_max), the same
clipping as `calibrate_explicit`. At k = 0 the density is a point mass, and the slice is the
same one-point slice the particle scheme uses: grid [ln s0], f = v0, L = sigma(t_min, s0)/sqrt(v0).
Later slices are on `x_grid`. Slices are collected into a `LeverageField`.

**Repricing.** Call prices from the terminal marginal treat the density as piecewise
constant per cell and integrate (e^x - K)^+ exactly over each cell, so the payoff kink
costs no accuracy. Implied vols via `market/bs.py::implied_vol`. The forward E[e^{X_T}]
and the remaining mass are reported as consistency diagnostics.

## Package layout

```
src/neural_particle_method/reference/
  __init__.py
  fokker_planck.py   FokkerPlanck(x_grid, v_grid, params): grid geometry, beta weights,
                     initial(x0, v0), operator(L) -> sparse A, advance(m, A, dt, n_sub, first),
                     marginal(m), cond_mean_v(m), mass(m)
  pde.py             solve_leverage_pde(...) -> PDEResult; default_x_grid, default_v_grid;
                     call_prices(x_grid, density, K, s0), implied_vols(x_grid, density, k_grid, T, s0)
  convergence.py     python -m ... : grid-refinement table for docs/pde_reference.md
tests/reference/
  test_fokker_planck.py, test_pde.py
docs/pde_reference.md
```

No existing module is modified. Dependencies: numpy, scipy only.

## API

```python
solve_leverage_pde(local_vol, params, *, s0=1.0, T, n_steps, x_grid, v_grid,
                   n_substeps=2, scheme="cn", L_max=4.0, snapshot_times=()) -> PDEResult

@dataclass
class PDEResult:
    field: LeverageField      # per-slice grid, L, f
    x_grid: np.ndarray        # cell centres
    density: np.ndarray       # terminal marginal density of X (cell average, per unit x)
    snapshots: dict           # snapped time -> marginal density, for multi-maturity repricing
    mass: float               # mass remaining at T
    forward: float            # E[e^{X_T}] from the discrete density
    runtime_s: float
```

`local_vol` is anything with `.sigma(t, x, s0)` and `.T_grid`; `params` is `HestonParams`
or a dict. Defaults: `default_x_grid(n=301)` is uniform on [ln 0.4 - 0.6, ln 2.2 + 0.6];
`default_v_grid(params, T, n=100)` is sinh-stretched towards 0 on [0, v_max] with
v_max = max(v0, theta) + 6 xi sqrt(max(v0, theta) T).

## Validation

- **Operator.** Mass is conserved to round-off when the outer boundaries carry no mass.
  With L = 0 the x-fluxes vanish and the v-marginal is the CIR transition law: mean and
  variance match the closed forms and P(V_T <= c) matches `scipy.stats.ncx2` for both a
  Feller-satisfying and a Feller-violating parameter set.
- **Pure Heston.** With L fixed to 1 the terminal marginal reprices `heston_call` at the
  Li parameters (Feller violated) on the quote grid; the tolerance in vol bp is fixed
  from the convergence study.
- **(a) Trivial leverage.** xi = 0.01, v0 = theta, flat sigma_Dup = sqrt(theta): every
  slice has L = 1 up to discretisation.
- **(b) Calibrated LSV.** Heston market (Li parameters) -> `DupireSurface.from_price_fn`
  -> calibrated model with different dynamics -> PDE-implied IVs at T = 1 match the
  Heston market IVs on the quote grid to a stated tolerance in bp. Same for the SSVI
  scenario s01 with the analytic Dupire local vol, at each of its maturities.
- **(c) Convergence.** `convergence.py` refines (n_x, n_v, n_substeps, n_steps)
  independently and prints the IV error table that goes into `docs/pde_reference.md`,
  together with runtimes.

Heavy configurations are marked `slow`; the default suite stays fast.

## Out of scope

- ADI or finite-element schemes; the sparse direct solve is fast enough for a reference.
- Integration into `bench/` (the neural-particle-code session does that).
- Stochastic rates, dividends, or the 4-factor model of the reference paper.
