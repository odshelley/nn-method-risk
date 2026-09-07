# Codebase refactor with unit tests and MLflow tracking — design

Date: 2026-09-07. Status: approved by Osian (four sections presented and accepted in session).

## Purpose

Restructure the repository into one layered, testable Python package; remove the
duplicated simulation, readout, and Dupire code; replace the hand-rolled JSON result
stores, caches, and aggregators with MLflow; and raise unit-test coverage. The
refactor is behaviour-preserving: every archived result must reproduce bit for bit.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Numerical fidelity | Bit-for-bit with the committed results. Golden replay tests guard it. If a golden fails because the old code was wrong, stop and report; never adjust the expected value silently. |
| MLflow role | Source of truth for every benchmark run and experiment arm. Aggregation and figures read from it. JSON result tree and torch cache retire. |
| Legacy results | One-shot idempotent importer loads existing JSONs into MLflow with their original git hash. No re-run required. |
| Tracking backend | Local SQLite `mlruns.db` plus `mlartifacts/`, both gitignored. `MLFLOW_TRACKING_URI` overrides when set. |
| Configuration | Frozen dataclasses. No Hydra, pydantic, or plugin registries. |
| Structure | Option A: one package under `src/`, subpackages by responsibility, one CLI. |

## Package layout

```
src/neural_particle_method/
  market/
    bs.py            Black-Scholes price and implied vol (r = q = 0)
    heston.py        characteristic function and semi-analytic call
    ssvi.py          SSVIParams, no-arb check, w and its derivatives, implied vol
    dupire.py        grid-based DupireSurface built from a price function
    local_vol.py     LocalVol protocol; the Dupire formula from (w, dw/dk, d2w/dk2, dw/dT)
                     written once and used by both DupireSurface and SSVILocalVol
  simulate/
    dynamics.py      HestonParams frozen dataclass (kappa, theta, xi, rho, v0)
    stepper.py       one Euler step of (lnx, v) under a leverage value and two normals
    leverage.py      LeverageField: typed replacement for the (t, grid, L, f) tuple list
  estimators/
    base.py          Estimator protocol: fit_predict(t, lnx, v, grid, weights=None) -> f_grid
    nn.py            SliceNet, warm-started NNRegressor
    nadaraya_watson.py
    spline.py        P-spline estimator
    ridge.py         frozen-body ridge readout (one class serving both the per-slice body
                     and the global (t, x) body)
    kalman.py        KalmanHead RLS readout with optional B-spline block
  calibrate/
    explicit.py      calibrate_explicit(local_vol, params, estimator, cfg, mixture=None)
    implicit.py      calibrate_implicit(local_vol, params, cfg, L0=None); GlobalNet
    importance.py    MixtureDesign, design_mixture
    warm.py          bump, distil, records_from_betas, beta_correction, kalman_pass,
                     full_resolve (moved out of the experiment scripts)
  pricing/
    reprice.py       reprice_iv, snap_times
    metrics.py       iv_metrics
  tracking/
    store.py         the only module that imports mlflow: start/finish run, log params,
                     metrics, artifacts, find finished run by params, download artifact
    importer.py      one-shot import of results/runs, results/bump, results/warm
  bench/
    scenarios.py     ScenarioSpec, registries, quote_k_grid
    algos.py         algorithm registry over the seven contenders
    runner.py        run_one(...) -> logs a run to the store
    sweep.py         paper grid enumeration, resumable parallel sweep
    aggregate.py     query store -> results/summary.csv + results/digest.md
  experiments/
    bump_correct.py  bump study as a function of (scenario, seed, cfg)
    warm_suite.py    the four arms (dyn, seq, xover, norm) as functions; job list
  figures/
    fig1_accuracy.py, fig2_wings.py, fig3_plane.py, fig4_latency.py
  cli.py             `nparticle` entry point (not `npm`, which would shadow Node's npm)
```

Removed after migration: top-level `bench/`, `figures/`, `experiments/*.py`, and the
flat modules in `src/neural_particle_method/`. The old modules stay as import shims
until the last migration step so tests keep passing throughout.

### Unit contracts

**LocalVol** (protocol): `sigma(t, x, s0=1.0) -> ndarray` and `t_min: float`.
`DupireSurface` and `SSVILocalVol` implement it. The Dupire denominator and clipping
constants live in `local_vol.dupire_local_vol(w, dwdk, d2wdk2, dwdT, k)`.

**HestonParams**: frozen dataclass with `from_dict` and `to_dict`, because scenarios and
legacy JSON store dynamics as dicts.

**stepper.heston_step(lnx, v, L_p, zb, zp, params, dt, sdt, tilt=None)**: returns
`(lnx_new, v_new)`. `tilt` is the optional mixture drift term `L_p * sqrt(v+) * theta_p`.
It never draws random numbers. Correlation is applied inside: `z1 = rho*zb + sqrt(1-rho^2)*zp`.

**LeverageField**: holds `times`, and per slice `grid`, `L`, `f`. Methods: `at(t, lnx)`
(piecewise-constant in t, linear interpolation in lnx, constant when the grid has one
point), `resample(grid)`, `to_records()`, `from_records()`, `to_json()`, `from_json()`.
`reprice.L_lookup` becomes `LeverageField.at`.

**Estimator** (protocol): `fit_predict(t, lnx, v, grid, weights=None) -> f_grid`. Per-slice estimators ignore `t`; the global-body ridge uses it. `KalmanHead` is a readout with its own `update(k, t, ...)` interface used only by the warm arms and is not an Estimator. Stateful
estimators keep warm-start state between calls. Estimators that do not support weights
raise `ValueError` when given any. The per-slice step budget (`first_steps` on the first
fitted slice, `later_steps` after) moves into `NNRegressor` so `calibrate_explicit` does
not need to know about it.

**Ridge readout**: one `RidgeHead(features_fn, lam, residual)` where `features_fn(t, lnx)`
returns the design matrix. The per-slice body ignores `t`; the global body uses it.
This merges `condexp.RidgeHead`, `implicit.GlobalRidgeHead`, and the private
`_features` helpers. `train_body` for the per-slice variant stays a method on the
per-slice feature provider.

**Configs** (frozen dataclasses with today's defaults):
`ExplicitConfig(n_steps=50, n_particles=200_000, fit_subsample=30_000, L_max=4.0, first_steps=400, later_steps=120, snapshot_times=())`,
`ImplicitConfig(n_steps=50, n_particles=50_000, alpha=0.5, n_iters=6, L_max=4.0, fit_steps=300, pool_subsample=60_000)`,
`RepriceConfig(n_particles=500_000, n_steps=200)`. The bench `cfg` dict becomes
`dataclasses.replace` on these.

## MLflow tracking model

| MLflow experiment | One run per | Params | Metrics | Artifacts |
|---|---|---|---|---|
| `bench` | (sid, algo, n_particles, seed) | sid, algo, n_particles, seed, every config field, git hash, schema | pooled/wings rmse and max, n_failed, total_s, fit_s, intraday_s, `rmse_bp_T<m>` per maturity | `leverage.json`, `iv_err_bp.json`, `diagnostics.json` |
| `overnight` | (sid, seed, N, n_steps) | those four plus git hash | overnight_s | `overnight.pt` (net state, betas, implicit records) |
| `bump` | (sid, seed) | sid, seed, config fields | `rmse_bp/<strategy>`, `build_s/<strategy>` | per-strategy leverage JSON, full result JSON |
| `warm` | (arm, sid, seed) | arm, sid, seed, config fields | flattened `rmse_bp/<key>` and `build_s/<key>`; the seq arm logs per-step metrics with `step=j`; norm logs `A_norm` and `error_fraction_pred` | full result JSON |

Rules:

- `tracking/store.py` is the only importer of mlflow. Library code never logs.
  Drivers (runner, sweep, experiment arms) call the store. Tests use a temporary
  SQLite URI or a fake store exposing the same methods.
- Resumability: a driver calls `store.find_finished(experiment, params)` before
  starting work and skips if a FINISHED run with identical identifying params exists.
  Identifying params are the "one run per" key columns above. Failed runs do not block
  a retry.
- Failures are logged as runs with status FAILED and `traceback.txt` as an artifact.
- Metric keys use `/` as separator; MLflow accepts `/` in metric names.
- Aggregation is a query over the `bench` experiment producing the same
  `results/summary.csv` columns as today, which stays committed as the paper snapshot.
  Figures keep reading the CSV, except fig2, which today reads run JSONs for the
  IV-error grid and will instead download the `iv_err_bp.json` artifact.
- The importer creates one run per legacy JSON, sets the tag `source=legacy_json` and
  the param `legacy_path`, and skips any file whose `legacy_path` already exists in the
  store. Legacy JSON directories stay on disk untouched; nothing reads them afterwards.
- Tracking URI resolution: `MLFLOW_TRACKING_URI` if set, else `sqlite:///<repo>/mlruns.db`
  with artifact root `<repo>/mlartifacts`. Both paths gitignored.

## Bit-for-bit safety

### Golden replay tests

Four archived run JSONs from `results/runs/` at N=50 000, seed 0, scenario s01, for
`nw`, `explicit_nn`, `explicit_nn_is`, and `implicit_ridge`. A test behind
`@pytest.mark.golden` re-runs each through the public entry point and asserts:

- the leverage field equals the archived `L_records` array-for-array (`np.array_equal`), and
- every metric in `metrics` equals the archived value exactly.

These are slow (minutes) and excluded from the default `pytest` run. They run before
each commit of the migration.

One fast golden at tiny size (`n_steps=6`, 4 000 particles, reprice 8 000 particles and
8 steps, the `TINY` config already used in tests) for every algorithm is generated once
from the current code into `tests/golden/` before any refactor commit and runs in the
default suite.

### RNG draw order (must not change)

- `calibrate_explicit`: `rng = default_rng(seed)`. If a mixture is present, one
  `rng.choice(3, n_particles, p=alphas)` before the loop. Per step k: for k > 0 one
  `rng.choice(n_particles, size=min(fit_subsample, n_particles), replace=False)`; then
  `zb = rng.standard_normal(n)`, then `zp = rng.standard_normal(n)`.
- `implicit._simulate`: per step `zb` then `zp`, same shapes.
- `calibrate_implicit`: `rng = default_rng(seed)`; `torch.manual_seed(seed)`; per
  iteration: simulate (draws as above), then one `rng.choice(len(pool), size=..., replace=False)`.
- `reprice_iv`: `rng = default_rng(seed)`; per step `zb` then `zp`.
- Torch seeding: `NNRegressor.__init__` and `RidgeHead.__init__` (per-slice) call
  `torch.manual_seed(seed)` on construction; `calibrate_implicit` seeds before building
  `GlobalNet`. Construction order relative to numpy draws is unchanged.
- Experiment arms derive seeds as today (`seed + 900`, `seed + 20 + i`, ...). The
  constants move into the arm functions unchanged.

The stepper receives `zb` and `zp` as arguments and never draws, so extracting it cannot
reorder draws.

### Migration order

Each step lands with the full default suite green and the golden suite green.

1. Generate the tiny goldens from current code; add the golden replay tests. Commit.
2. `market/`: move bs, heston, ssvi, dupire; introduce `local_vol.py`; shims in old paths.
3. `simulate/`: `HestonParams`, `heston_step`, `LeverageField`. Switch the three loops to
   the stepper and `LeverageField.at`.
4. `estimators/`: protocol, moves, merged `RidgeHead`, Kalman move.
5. `calibrate/`: configs, estimator injection into `calibrate_explicit`, `warm.py` from
   the experiment helpers.
6. `pricing/`.
7. `tracking/store.py` with tests on a temporary SQLite store.
8. `bench/` moved into the package; runner logs to the store; sweep resumes from the
   store; aggregate queries the store.
9. `experiments/` moved into the package; overnight cache becomes an `overnight` run.
10. `tracking/importer.py`; run it on the archived results.
11. `figures/` moved; fig2 reads the artifact.
12. `cli.py` as `nparticle`; delete shims, old top-level packages, and root `experiments/*.py`.
    Update README and `pyproject.toml` (`[project.scripts] nparticle = ...`, wheel packages,
    `mlflow>=2.14` dependency, `pytest-cov` dev dependency, golden marker).

## Testing

Default suite target: under 30 seconds, no golden runs.

- **market**: Heston to Black-Scholes in the small vol-of-vol limit (kept); deep ITM
  (kept); implied vol round trip (kept); Dupire flat surface (kept); SSVI analytic
  `dw/dk`, `d2w/dk2`, `dw/dT` against central finite differences; `SSVILocalVol` and
  `DupireSurface.from_price_fn` on the same SSVI surface agree to grid tolerance.
- **simulate**: one `heston_step` against a hand-computed value; variance floor at zero;
  `LeverageField.at` piecewise-constant in t and interpolating in lnx; single-point grid;
  JSON round trip exact.
- **estimators** (contract test parametrised over the five Estimators: nn, nw, spline, per-slice ridge, global ridge): output shape equals grid
  length, finite, positive; weights of all ones give the same output as no weights for
  estimators that accept weights; spline raises on weights. Specific: Kalman update with
  infinite prior information leaves beta unchanged and with zero prior reproduces
  weighted least squares; spline recovers a quadratic to 1e-3; NW on constant data is
  constant.
- **calibrate**: existing explicit and implicit tests kept; importance weights average
  to one within MC tolerance and ESS fraction in (0, 1]; `warm.records_from_betas` and
  `warm.distil` round trip on a tiny body.
- **pricing**: existing reprice and metrics tests kept; snapped-time inversion.
- **tracking**: temporary SQLite store: log params, metrics, artifact; `find_finished`
  hits on identical params and misses on a differing one; a FAILED run does not block;
  importer on a two-file fixture directory creates two runs and is a no-op on rerun.
- **bench**: existing algo, scenario, runner, aggregate, cli tests adapted to the store.
- **experiments**: smoke test of each arm at the SMOKE config through the store, marked
  `slow` and excluded by default; a fast test that `job_list` enumerates the expected jobs.
- **figures**: existing tests kept, run against a summary CSV fixture and a temporary
  store for fig2.
- **tooling**: `ruff check` clean under the existing config; `pytest-cov` reporting with
  no threshold; markers `golden` and `slow` registered in `pyproject.toml`, both
  deselected by default via `addopts = "-m 'not golden and not slow'"`.

## Out of scope

- Any change to algorithms, defaults, or numerics.
- Hydra, pydantic, or plugin configuration frameworks.
- A remote tracking server or artifact store.
- Changes to `paper/`.
