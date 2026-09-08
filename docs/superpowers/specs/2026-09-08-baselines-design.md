# Baseline comparison set for conditional-expectation estimators — design

Date: 2026-09-08. Status: approved by Osian (five sections presented and accepted in session).
Builds on the refactored package (spec `2026-09-07-refactor-mlflow-design.md`, branch `refactor-mlflow`).

## Purpose

Add the published competitors for the conditional-expectation step of particle LSV calibration
to the benchmark harness, one card per method, run on identical terms, so that the paper's
Figures 1 and 3 and a new sensitivity figure compare the network methods against kernel
Nadaraya-Watson (GHL), Muguruza's conditional Monte Carlo, RKHS ridge (Bayer et al.),
bin Monte Carlo, and partition-of-unity RBF (Hakala). Every card carries source equations,
its tuning knob and protocol, the failure mode it is meant to probe, and an acceptance
criterion reproducing a number from its source.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Architecture | Option C: each baseline is an `Estimator` in the existing explicit pipeline; constructor knobs expose the source's settings; defaults are the method's published tuning rule, never a value tuned on our scenarios. The paper comparison calls every estimator with no arguments. |
| Ground truth | The SSVI grid stays for the paper figures. A new Heston-market scenario family (surface from the semi-analytic Heston pricer through the grid Dupire class) hosts the acceptance tests. |
| Budgets | 1e3, 1e4, 1e5 particles; reprice budget unchanged at 500 000 paths, 200 steps. |
| Muguruza | Corollary 4.1 conditional formula only. |
| PDE anchor | Implemented by a separate session (PDE-method) on its own branch as a `LeverageField` on the harness time grid; the leverage-error metric is wired in when it lands. Not a blocker. |
| PURBF | Included; the Hakala paper is obtained and ingested first; the card is written from the source. |
| Control variates | Excluded this round; related-work mention on the kernel card only. |
| Work split | Papers session: `BASELINES.md` cards, `rkhs`, `bins`, Hakala sourcing. This session: harness (Heston family, step-context hook, budgets, sweeps, sensitivity, figures), `nw_ghl`, `muguruza`, `purbf`. |

## Harness contract

For every run the only thing that varies between methods is the `estimator` object passed to
`calibrate_explicit`. Fixed by the scenario and budget: target surface and its Dupire local vol,
calibrated-model Heston dynamics, particle count, 50 Euler steps in log-spot with full truncation,
per-slice quantile grid, seed, reprice protocol (fresh 500 000 paths, exact target implied vols at
snapped times). Numpy draws happen in `calibrate_explicit`, not in estimators, so two methods at
the same seed share Brownian increments until their leverages diverge (common random numbers).

### Budgets

`BudgetSpec = (1_000, 10_000, 100_000)`. `ExplicitConfig.fit_subsample` is capped at the
particle count inside `calibrate_explicit` (it already is, via `min(fit_subsample, n_particles)`).

### Heston market family

`bench/scenarios.py` gains

```python
@dataclass(frozen=True)
class HestonMarketSpec:
    sid: str
    market: HestonParams        # generates the target surface
    dynamics: HestonParams      # calibrated model
    s0: float = 1.0
    T: float = 1.0
    maturities: tuple = (1.0,)
    k_grid_market: np.ndarray = field(default_factory=lambda: np.linspace(-0.9, 0.9, 121))
    T_grid_market: np.ndarray = field(default_factory=lambda: np.linspace(0.02, 1.0, 50))
```

with `local_vol()` returning `DupireSurface.from_price_fn(lambda K, T: heston_call(K, T, **market), s0, T_grid_market, k_grid_market)` and
`target_ivs(k, maturities)` inverting `heston_call` exactly. Members:

- `li_simple`: market kappa 1.5768, theta 0.0484, xi 0.5751, rho -0.7, v0 0.1024; dynamics same but rho -0.5 (thesis eq. 3.3.2 "simple LSVM").
- `li_complex`: same market; dynamics kappa 1, theta 0.0144, xi 0.5751, rho 0, v0 0.0144 (thesis "complex LSVM").
- `bayer`: market kappa 2.19, theta 0.17023, xi 1.04, rho -0.83, v0 0.0045; dynamics kappa 1, theta 0.0144, xi 0.5751, rho -0.9, v0 0.0144 (Bayer et al. section 5). Bayer et al. floor their CIR at 1e-3; the harness uses full truncation at 0 for every method and the `rkhs` card records this difference as a known deviation.

`bench/algos.py` accepts either scenario type: it calls `sc.local_vol()` and `sc.target_ivs(...)`;
`ScenarioSpec` gets the same two methods wrapping `SSVILocalVol` and `implied_vol_ssvi`, and
`runner.run_one` uses them instead of constructing `SSVILocalVol` itself. Registry: `heston_registry()`;
`full_registry()` includes it. Run params log `scenario.family` as `ssvi` or `heston`.

Grid Dupire caveat: finite differences in T and k make the surface approximate near the first
maturity. `T_grid_market` starts at 0.02 and has 50 points; the test `test_heston_market_reprices_market`
requires the local-vol model with `dynamics = market` and xi -> 0 (leverage one) to reprice the market
calls within 15 bp of implied vol at T = 1 at 500 000 paths. Acceptance tolerances (below) include this.

## Step-context hook

`calibrate/explicit.py` builds, after each Euler step k -> k+1, a frozen

```python
@dataclass(frozen=True)
class StepContext:
    lnx_prev: np.ndarray   # log-spot at t_k
    v_prev: np.ndarray     # truncated variance max(v_k, 0)
    L_p: np.ndarray        # leverage applied to each particle over the step
    zb: np.ndarray         # variance-driver normal
    zp: np.ndarray         # orthogonal normal
    theta_p: np.ndarray | None   # mixture tilt per particle, or None
    dt: float
    params: HestonParams
```

and passes `ctx=StepContext(...)[idx]` (all arrays subsampled with the same `idx` as `lnx`, `v`)
to `estimator.fit_predict(t, lnx[idx], v[idx], grid, weights=wi, ctx=...)` only when
`getattr(estimator, "needs_step_context", False)` is true. Estimators that do not declare it are
called exactly as today, so the call path and the numpy draw order for existing algorithms are
unchanged; the tiny and archived goldens must replay bit for bit after this change. Building the
context copies no arrays (views of the pre-step state kept for one iteration).

Under a mixture the tilt enters the conditional mean; the Muguruza card marks weighted use as
untested against the source.

## Estimators

All live in `src/neural_particle_method/estimators/`, registered in `make_estimator`, added to
the contract test's `_all()`. Defaults are the published rule. `supports_weights` as stated.

### `nw_ghl` — kernel Nadaraya-Watson (Guyon and Henry-Labordere 2012)

File `nadaraya_watson.py`, class `GHLKernel(c=1.0, kernel="quartic")`. Estimate on spot
S = e^x: f(K) = sum_i V_i K_h(S_i - K) / sum_i K_h(S_i - K), quartic kernel
K(u) = 15/16 (1 - u^2)^2 on |u| <= 1, bandwidth
h(t) = c * 1.5 * S0 * sigma_target(S0, t) * sqrt(max(t, 0.25)) * N^(-1/5)
(Cozma, Mariapragassam, Reisinger 2019 eq. for h_N(T), stated as GHL's rule; the card quotes the
GHL book chapter 11 once confirmed). `sigma_target(S0, t)` is the scenario's local vol at the
money, available from the local-vol object. Knob: `c`. `supports_weights = True` (weights multiply
the kernel). Failure to probe: O(h^2) bias at large c, variance explosion at small c and N = 1e3
(Muguruza 2019 section 3.2), and the Silverman-rule degradation at 1e5 to 2e6 particles reported by
Bain, Mariapragassam, Reisinger 2019. Acceptance: Li thesis Table 2.2 (Heston market, N = 1e5)
average absolute IV error within 0.3 percentage points, with override `h = S0 N^(-1/5)`.
The existing `nw` (Gaussian kernel, Silverman 1.06 std N^(-1/5) in log-spot) remains a separate row.

### `muguruza` — conditional Monte Carlo (Muguruza 2019, Corollary 4.1)

File `muguruza.py`, class `ConditionalMC()`, `needs_step_context = True`, no knob. For each
particle in the subsample, with L = ctx.L_p, v = ctx.v_prev, rho = params.rho, dt = ctx.dt:

```
mu_i    = ctx.lnx_prev + (-0.5 L^2 v + (L sqrt(v) theta_p if tilt)) dt + L sqrt(v) rho zb sqrt(dt)
s2_i    = (1 - rho^2) L^2 v dt
phi_i(x)= exp(-0.5 (x - mu_i)^2 / s2_i) / sqrt(s2_i)
f(x)    = sum_i w_i V_i phi_i(x) / sum_i w_i phi_i(x)        (V_i = current-slice v, w_i weights or 1)
```

Particles with s2_i below 1e-18 (variance truncated to zero) are dropped from both sums.
This is Corollary 4.1 with the discount factor D = 1 and lambda frozen over the step (Assumption
3.1), evaluated in log-spot. `supports_weights = True` but weighted use is untested (card note).
Failure to probe: effective bandwidth is the physical one-step spread, so error grows as dt
shrinks or N falls; source reports non-convergence at high vol-of-vol (its section 8), probed on
the (xi, rho) plane. Acceptance: on `li_simple` at N = 5e4 the pooled IV RMSE is within 10 bp of
`nw_ghl` at the same seed (source claim: "performs at the level of the particle method").

### `rkhs` — kernel ridge in an RKHS (Bayer, Belomestny, Butkovsky, Schoenmakers 2022)

File `rkhs.py`, class `RKHSRidge(lam=1e-9, n_centres=100, variance=0.1)`. Centres Z_j at the
j*100/(L+1) percentiles of the subsample's log-spot (source eq. 5.9); Gaussian kernel
k(x, y) = exp(-(x - y)^2 / (2 variance)); solve (K^T K + N lam R) beta = K^T v with
K_ij = k(x_i, Z_j), R_jk = k(Z_j, Z_k) (source eq. 4.3); f(x) = sum_j beta_j k(Z_j, x) (eq. 4.4).
Knob: `lam`. `supports_weights = True` (row weights in the normal equations). Failure to probe:
thesis section 3.4: strong spot-vol correlation breaks it at lam >= 1e-6 with N = 1e5.
Acceptance: thesis Table 3.2 within 0.3 points with overrides `n_centres=40, variance=5, lam=1e-7`;
Bayer Figure 5 plateau: ATM error flat for lam in [1e-9, 1e-5] on `bayer`.
Implemented by the papers session.

### `bins` — bin Monte Carlo (van der Stoep et al. 2014; Li thesis chapter 4)

File `bins.py`, class `BinMC(n_bins=20, kind="frequency")`. Equal-frequency bins on the
subsample's log-spot; f is the (weighted) mean of v within the bin containing each grid point
(thesis eq. 4.0.1). Knob: `n_bins`. `supports_weights = True`. Failure to probe: too many bins per
particle gives single-particle noise (thesis section 4.3). Acceptance: thesis Table 4.2, average
absolute IV error 0.91% (simple) and 1.01% (complex) at N = 1e5, l = 20, within 0.3 points.
Implemented by the papers session.

### `purbf` — partition-of-unity RBF (Hakala 2019)

File `purbf.py`, class `PURBF(n_patches, overlap, rbf)`; the three defaults are deliberately unspecified here and are fixed by the card once the source is ingested. Overlapping patches over the
log-spot range, local RBF least-squares fit per patch, blended by a smooth partition of unity.
Defaults, knob (`n_patches`), failure mode, and acceptance number are taken from the Hakala paper
once obtained and ingested; the card is written from the source before implementation. Until the
paper is in, this estimator is a stub raising `NotImplementedError` and is excluded from sweeps.

### Ours (unchanged)

`explicit_nn`, `ridge`, `spline`, `explicit_nn_is`, `implicit_nn`, `implicit_ridge` with frozen
defaults. Sensitivity knobs: network hidden width (`SliceNet(hidden=...)` exposed through
`NNRegressor(hidden=64)`), ridge `lam`, spline `lam`.

## Experiment protocol and tracking

Three new MLflow experiments; the paper's `bench` experiment is untouched.

- `baselines`: SSVI s01..s20 x every estimator (existing seven plus `nw_ghl`, `muguruza`,
  `rkhs`, `bins`, `purbf` when available) x budgets (1e3, 1e4, 1e5) x seeds (0, 1, 2), plus the
  six fig3 stress scenarios x every estimator x 1e4 and 1e5 x seeds. `nparticle sweep --preset baselines`.
  Runs log everything `run_one` logs today plus `scenario.family`, `budget`, and `estimator.<knob>`
  params. Aggregation reuses `aggregate` with an `--experiment` flag.
- `sensitivity`: scenarios `s01` and `li_simple`, budgets 1e4 and 1e5, seeds (0, 1, 2), one knob per
  method over a geometric range centred on its default: `nw_ghl` c in {0.1, 0.2, 0.5, 1, 2, 5, 10};
  `nw` bandwidth multiplier same set; `rkhs` lam in {1e-9, 1e-8, ..., 1e-2}; `bins` n_bins in
  {5, 10, 20, 50, 100, 200, 500}; `purbf` n_patches per its card; `explicit_nn` hidden in
  {8, 16, 32, 64, 128}; `ridge` lam in {1e-6, ..., 1e-1}; `spline` lam in {1e-3, ..., 1e2}.
  `nparticle sensitivity [--sids] [--budgets]`. Params: `knob_name`, `knob_value`; metric `knob_value`
  duplicated for plotting.
- `acceptance`: one run per card on its Heston market with the source budget and overrides,
  driven by `tests/acceptance/test_cards.py` (marker `slow`) which also logs to the store and
  asserts the criterion. `nparticle acceptance` runs the same code outside pytest.

Metrics per run: existing pooled and wings IV RMSE and max, per-maturity RMSE, `fit_s`, `total_s`;
new `budget` (int), `knob_value`; and, when a PDE reference `LeverageField` is available for the
scenario (looked up in experiment `pde_reference` by sid), `lev_rmse` per slice and pooled.

Figures: `fig1_accuracy` becomes a grouped bar chart per estimator at 1e4 and 1e5 from the
`baselines` experiment; `fig3_plane` becomes a small-multiples panel, one heatmap per estimator;
new `fig5_sensitivity` plots each method's curve with a horizontal line for `explicit_nn` at its
default. Figure 2 and Figure 4 are unchanged.

`BASELINES.md` at the repo root: one card per method with source and equations, knob and
protocol, failure mode probed, acceptance criterion, achieved number, and the MLflow run id.
Drafted by the papers session; achieved numbers filled in from the `acceptance` experiment.

## Testing and fidelity

- Goldens remain the gate: after the step-context hook lands, `tests/test_golden_tiny.py` and
  `uv run pytest -m golden` must be green unchanged. Each new estimator gets a tiny golden of its
  own (generated once, then frozen) so its numbers are pinned.
- Contract test: every new estimator is added to `tests/estimators/test_contract.py::_all()`.
  `ConditionalMC` gets its own contract test that constructs a synthetic `StepContext`.
- Analytic checks: `GHLKernel` and `BinMC` recover a constant E[V|X] exactly and a linear one
  within tolerance; `RKHSRidge` with lam = 1e-12 interpolates a smooth function at the centres;
  `PURBF` reproduces a quadratic across a patch boundary with no seam (tolerance 1e-6); `ConditionalMC`
  on one hand-built step (three particles) matches a hand-computed weighted average to 1e-12.
- Heston market: `test_heston_market_flat_limit` (xi -> 0 and dynamics = market gives leverage one
  within 0.02 on the mid-band); `test_heston_market_reprices_market` (15 bp at T = 1).
- Acceptance tests under `tests/acceptance/`, marker `slow`, one per card.
- Drivers: `sweep --preset baselines` job enumeration count; `sensitivity` enumeration and knob
  logging; both resumable, tested on a temporary store like the existing sweep test.
- Default suite stays under about 30 s.

## Out of scope

- Control variates (Cozma et al.), Table 1 (real FX surface), the exact-distribution variant of
  Muguruza, the PDE solver itself (separate session), and any change to the paper's existing
  `bench` results.
