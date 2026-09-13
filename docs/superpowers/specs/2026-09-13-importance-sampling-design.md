# Importance sampling by change of measure: does the tilt pay?

Date: 2026-09-13. Branch: `refactor-mlflow`. Builds on the benchmark suite
(`docs/superpowers/specs/2026-09-08-benchmark-suite-design.md`), the budget cells of
`suite/budget.py`, and the Girsanov tilt already implemented in `calibrate/importance.py` and
`calibrate/explicit.py` (the `mixture` argument), whose theory is section "Importance sampling
by change of measure" of `paper/notes.tex`.

## Goal

Decide whether the volatility-preserving Girsanov tilt (a defensive three-component mixture
of drifts on log-spot, with exact discrete-time weights) buys accuracy in the regime the
notes predicted for it: small particle budgets and short-dated far wings. The claim to test
is **budget equivalence**: a tilted calibration at a small budget matches an untilted one at a
larger budget, on wing error and on leverage error against the PDE reference. If the tilt
does not pay, the study must say why, from measurements rather than conjecture. This is a
second aspect of the paper, with its own section, tables and figures.

## Decisions taken with Osian (13 Sept)

- Success claim: budget equivalence (tilted 10k or 30k particles against untilted 80k).
- Two layers. Layer 1 isolates the estimator: particles simulated under the frozen PDE
  leverage, no feedback, so the tilt's effect on per-slice estimation variance is measured
  alone. Layer 2 is end-to-end calibration scored against the floor exactly as section 4.
- A small tilt-design sweep (two schedules, three strengths) in layer 1 only; the winner
  goes into layer 2.
- Layer 2 rows: NW cold at 10k/30k/80k on all 20 SSVI scenarios; the promoted network
  recipe cold at 10k/30k/80k on the six study scenarios; the promoted body plus spline head
  online at 10k/80k on all 20 scenarios, both lags. Repricing stays untilted.
- Visual diagnostics: particle clouds at several slice times with the PDE conditional
  expectation, our estimate and the quoted wing strikes, tilted beside untilted.
- Structure: a new suite stage `nparticle tilt` with MLflow experiments `tilt_slices`,
  `tilt_cold`, `tilt_online` (approach A of the brainstorm).

## 1. Tilt design

`calibrate/importance.py` gains

```python
@dataclass(frozen=True)
class TiltDesign:
    schedule: str          # "constant" | "inverse_sqrt"
    cost: float            # cap on  sum_k theta_k^2 dt / (1 - rho^2)  over [0, T]
    alpha0: float = 0.5    # defensive (untilted) mixture weight

    @property
    def name(self) -> str: ...            # f"{schedule}-{cost:g}", e.g. "constant-3"

    def mixture(self, n_steps, T, rho) -> MixtureDesign: ...
```

`MixtureDesign.thetas` may now be a tuple of three floats (today) or a NumPy array of
shape `(3, n_steps)` (a schedule); `etas` divides by `sqrt(1 - rho^2)` in both cases.

- `constant`: `theta = sqrt(cost * (1 - rho**2) / T)` for the two wing components, `0` for
  the defensive one. This is exactly today's `th_cap`; `TiltDesign("constant", 3).mixture(...)`
  must give the thetas `design_mixture` gives whenever the cap binds in `design_mixture`
  (it does for every suite scenario; a test asserts equality on s01 and s11).
- `inverse_sqrt`: `theta_k = c / sqrt(max(t_k, dt))`, `t_k = k dt`, with `c` fixed by the same
  total cost: `c = sqrt(cost * (1 - rho**2) / (1 + H_{n-1}))`, `H_m` the harmonic number.
- The moment-matching target `k_target` of `design_mixture` is not used by `TiltDesign`;
  `design_mixture` itself is untouched (it is golden-replayed through `explicit_nn_is`).
- Sweep: `DESIGNS = [TiltDesign(s, c) for s in ("constant", "inverse_sqrt") for c in (1, 3, 9)]`
  plus `None` (untilted, name `"none"`).

`calibrate/explicit.py::calibrate_explicit` handles a scheduled mixture: at step `k` the
per-particle tilt is `thetas[:, k][comp]` and the weight recursion uses `etas[:, k]`. With
1-D thetas the code path and arithmetic are unchanged. `is_diag` gains
`"ess_min_slice"`: the minimum over steps of the per-step ESS fraction
`w.sum()**2 / (N * (w**2).sum())`.

## 2. Layer 1: the slice study (`tilt_slices`)

`suite/tilt.py::run_slice_cell(store, sid, n_particles, design, seed, settings=FULL)`.
Key `{sid, n_particles, design: name, seed, n_steps: 200}`. One cell:

1. Load the PDE reference leverage field for `sid` at 200 steps
   (`bench/reference_runs.run_reference(store, sid, n_steps=200, n_x=settings.n_x,
   n_v=settings.n_v)`, artifact `leverage.json`, `LeverageField.from_json`). Its slices carry
   the PDE conditional expectation `f` and leverage `L` on `DEFAULT_GRID`.
2. Simulate `n_particles` particles for 200 steps under that frozen field (`L_p =
   np.interp(lnx, slice.grid, slice.L)` per step, `heston_step` with `theta_p` from the
   mixture when tilted), accumulating the weights exactly as `calibrate_explicit` does. No
   estimator feeds back into the dynamics.
3. At each quoted maturity `T_j` in {0.25, 0.5, 1, 2} (snapped to the grid as `snap_times`
   does), fit each estimator on the whole cloud with its weights and evaluate on the 13
   quoted log-strikes `k = log(quote_k_grid())` (`bench/scenarios.quote_k_grid`):
   - `nw`: `NadarayaWatson` as the suite's cold NW uses it (Silverman bandwidth, floored V);
   - `net`: `regressor_from_recipe(load_recipe("explicit_opt"))`, fitted per slice with
     `first_steps` (one cold slice per maturity, `warm_start` irrelevant), weights passed.
   Targets: the reference slice's `f` and `L` interpolated at the strikes.
4. Local ESS at each strike: with the NW kernel weights `K_i = K_h(X_i - k)` at that slice's
   bandwidth, `ess_local = (sum K_i w_i)**2 / sum (K_i w_i)**2` (w = 1 untilted).

Logged: artifact `slice_scores.json` with, per maturity, per strike, per estimator:
`f_rel_err = (f_hat - f_ref) / f_ref`, `lev_rel_err = (L_hat - L_ref) / L_ref`, `ess_local`;
per maturity `ess_slice`; metrics `wings_abs_f_rel/{nw,net}` (mean |f_rel_err| over the six
wing strikes and four maturities), `ess_final`, `max_w`, `sim_s`, `fit_s/{nw,net}`.

Grid: `SLICE_SIDS = ("s01", "s02", "s05", "s09", "s11", "s16")`, `n_particles` in
`(10_000, 30_000, 80_000)`, seven designs, seeds 0..4: 630 cells (both estimators inside
one cell). Under two hours at `--jobs 4`.

Scoring (in `tilt tables`): for each (sid, N, design, estimator, maturity, strike), over the
five seeds, `std = std(f_rel_err)` and `bias = mean(f_rel_err)`. Wing summaries average
over the six wing strikes, then over scenarios. NW's Silverman bandwidth is computed from the
*weighted* spread of the cloud, so the tilted and untilted arms smooth on the same scale; the
per-maturity bandwidth is stored in `slice_scores.json`. The **winning design** is the one with
the smallest wing std at N = 30k for NW, averaged over the six scenarios and four maturities,
among the admissible designs. A design is admissible only if the bias it *adds* is small next to
the variance it removes: `|bias_d - bias_u| <= 1.0 + 0.5 * max(std_u - std_d, 0)` in percentage
points of f, so one point of bias is free and half a point more is bought per point of std
removed (the total bias is dominated by the shared Euler-versus-Fokker-Planck discretisation
error, which no tilt can move). A design whose (sid, T, k) coverage does not match the untilted
arm's, or with a cell of fewer than two seeds, is skipped with a warning. A quoted maturity that snaps to
t = 0 is skipped (a one-point cloud has no bandwidth). `nparticle tilt winner` prints it; `tilt cold` and
`tilt online` default `--design` to it.

## 3. Layer 2: end-to-end (`tilt_cold`, `tilt_online`)

### Cold

`suite/tilt.py::run_tilt_cold(store, sid, algo, n_particles, seed, design, settings=FULL)`
wraps `bench/runner.run_one` with `experiment=settings.experiment("tilt_cold")`,
`extra_key={"design": name, **(cold_extra_key(algo) or {})}` and a new keyword
`mixture=` that `run_one` and `run_algo` forward to `_nw` and `_nn_opt`, which pass it to
`_explicit` (a `ValueError` for any other algo). `knobs={"keep_slice_weights": True}`
for the network as in `run_cold`. `run_one` already logs `is_diag`; `ess_min_slice` rides
along. Scoring is `run_one`'s: 500k repricing paths, `lev_rmse` against the PDE reference,
`liquid_mae_bp` (native in `pricing/metrics.iv_metrics`).

Grid: `nw` on `SSVI_SIDS` (20), `explicit_nn_opt` on `SLICE_SIDS` (6); `n_particles` in
`(10_000, 30_000, 80_000)`; seeds `(0, 1)`; designs `("none", winner)`. Counts 240 + 72 =
312 cells. `tilt_jobs_cold(design)` returns them; the untilted cells are run in the same
experiment so every comparison is paired by seed.

### Online

`suite/budget.py::run_budget_cell` gains `design=None`; when set, the experiment is
`settings.experiment("tilt_online")` and the key gains `"design": name`. The cell passes
`mixture=design.mixture(n_steps, T, rho)` to `online_sweep`.

`suite/heads.py::online_sweep(..., mixture=None)`: when given, the sweep draws the component
per particle, applies `theta_p` in `heston_step`, accumulates `ell` and `w` exactly as
`calibrate_explicit` does, and calls `head.correction(t, x_sub, v_sub, f_stale, model, grid,
weights=w[idx])`. Every head's `correction` gains `weights=None`; `SplineHead` forwards it,
`RKHSHead` and `FeatureRidgeHead` raise `ValueError` when it is not `None`. `online_sweep`
returns `is_diag` as a third value (`None` untilted); the cell logs it.

`estimators/spline.py::spline_estimate(..., weights=None)`: weighted least squares,
`c = solve(B.T @ (w[:, None] * B) + lam * D2.T @ D2, B.T @ (w * v))`; `PSpline.supports_weights
= True`. With `weights=None` the arithmetic is unchanged.

Grid: `SSVI_SIDS`, both `LAGS`, budgets `(10_000, 80_000)`, seeds `(0, 1)`, method
`explicit_opt_spline`, design = winner: 160 cells. The untilted partners are the existing
cells in `suite_budget_tuned` keyed by the promoted `recipe_hash` (same key without `design`),
where `scripts/budget_sweep.py --body explicit_opt` wrote them; `tilt tables` reads both.

## 4. Figures (`nparticle tilt figures`)

`suite/tilt_figures.py`, from layer-1 cells of s02 and s11 at 30k, seed 0, designs `none`
and the winner (the cells store `cloud_T{t}.npz` with a 5 000-particle subsample of
`(lnx, v, w)` at each quoted maturity for this purpose):

- **A, clouds** (`figures/out/tilt_cloud_{sid}.png`): 2 x 2, rows t = 0.25 and t = 1,
  columns untilted and tilted; scatter of the subsample with marker area proportional to
  weight; PDE `f` solid, NW estimate dashed, network estimate dotted; vertical lines at the
  six wing strikes.
- **B, leverage** (`figures/out/tilt_leverage_{sid}.png`): same panels, `L` from the PDE and
  from the two estimates, with the cloud's weighted marginal density as a faint histogram.
- **C, weights** (`figures/out/tilt_ess_{sid}.png`): per-slice ESS fraction against time for
  each of the seven designs, and the local ESS at the outermost quoted strikes at 30k.

A and B enter the notes.

## 5. Tables and notes

`suite/tilt_tables.py`, invoked by `nparticle tilt tables`, writes bare tabulars with the
section-4 conventions (`_render_rows` bold-best per column, `--` for missing):

- `paper/tables/tilt_slices.tex`: rows estimator x N; columns per design (untilted first):
  wing std at T = 0.25, wing std pooled over maturities, wing bias pooled; values in percent
  of `f`. Winner marked in the caption.
- `paper/tables/tilt_cold.tex`: rows method x N x {untilted, tilted}, floor last; columns
  pooled MAE, wings, T = 0.25 MAE, liquid, `lev_rmse`, `ess_min_slice`, median seconds.
- `paper/tables/tilt_online.tex`: rows budget x lag x {untilted, tilted}; columns pooled,
  wings, liquid, online seconds.

`paper/notes_experiments.tex` gains a section "Importance sampling" after section 4: the
design (schedules, cost, mixture), layer 1 with `tilt_slices.tex` and figures A and B, layer
2 with the two tables, a results paragraph written after the runs, and the diagnosis
paragraph: variance versus bias from layer 1, local ESS at the strikes, and leverage error
versus implied-vol error in layer 2 (a leverage gain without an implied-vol gain means the
200-step scoring floor is binding in the wings). Section 4 is unchanged.

## 6. CLI

`nparticle tilt {slices,cold,online,winner,tables,figures}`, flags `--jobs`, `--sids`,
`--design` (name; default the winner for cold/online), `--budgets`, `--seeds`, `--smoke`, and
on `cold` an `--algos` filter (`nw`, `explicit_nn_opt`; both by default) so the cheap NW rows
and the overnight network rows can be run as separate stages.
Smoke: one scenario, 2 000 particles, seeds (0,), designs (`none`, `constant-3`), every
command runs end to end in under three minutes. Heavy stages never overlap.

## 7. Testing

- Bit-for-bit: the tiny goldens (including `explicit_nn_is`) replay unchanged; a test runs
  `calibrate_explicit` on a tiny scenario with `design_mixture(dynamics, T)` (whose cap binds
  there) and with `TiltDesign("constant", cost=3).mixture(n_steps, T, rho)` and asserts
  identical weights and field.
- `TiltDesign`: constant thetas equal `design_mixture`'s on s01 and s11; inverse-sqrt
  schedule's total cost equals `cost` to 1e-12; `name` round-trips through the CLI.
- Weighted spline: weights of one equal the unweighted fit; a duplicated point with weight
  two equals the fit with the point duplicated.
- `online_sweep` with a mixture: weights are positive, bounded by `1/alpha0`, ESS in (0, 1];
  with `mixture=None` bit-for-bit against today.
- Layer-1 scoring: std/bias split and winner selection on a synthetic `slice_scores` set
  with a known answer; a design that adds more bias than its std saving allows is disqualified,
  and a design missing a scenario's cells is skipped.
- Job counts: slices 630, cold 312, online 160 (full); smoke counts.
- Tables: fixtures render all three files with expected bold cells.

## 8. Run order and cost

1. `nparticle tilt slices --jobs 4` (under 2 h).
2. `nparticle tilt winner`; `nparticle tilt figures`.
3. `nparticle tilt cold --jobs 4` NW rows (about 2 h) then network rows (about 16 h, overnight).
4. `nparticle tilt online --jobs 4` (about 2 h).
5. `nparticle tilt tables`; notes section; PDF; commit; push.
