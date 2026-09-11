# Optuna search of the offline body's per-slice estimator

Date: 2026-09-11. Branch: `refactor-mlflow`. Builds on the benchmark suite
(`docs/superpowers/specs/2026-09-08-benchmark-suite-design.md`) and the tuned per-slice
network of commits bf022fd (F13) and 1af782e (F14a).

## Goal

Find a per-slice network recipe for the offline body that beats Nadaraya–Watson (NW) as an
estimator of E[V⁺ | X], and show that a frozen body trained with it, plus the spline head,
beats NW re-solved from scratch at the practical online budget of 80 000 particles on the 20
reported SSVI scenarios. The hand-tuned recipe of F13 (64×3, minibatch 8192, lr 1e-3,
2000/500 steps) is a draw with NW cold and 1.5 bp ahead with a head at 80k; the search is
meant to widen that.

## Decisions taken with Osian (11 Sept)

- Two-stage: an Optuna search on a cheap proxy objective, then full validation of the top
  recipes. The proxy is the held-out particle loss of the per-slice fit, the criterion the
  true conditional expectation minimises, measured without the 6.6 bp Monte Carlo noise of
  the repricing score.
- The search tunes on a fresh, held-out set of SSVI scenarios; the 20 reported scenarios are
  touched only at validation.
- The search space is the network and optimiser knobs plus switchable structural priors on
  the conditional expectation.
- The fits see the offline cloud (500 000 particles; the fit sample size is a knob).
  Validation is body + spline head at 80 000 online particles against NW at 80 000.
- Budget: overnight, about 200 trials, then validation of the top three recipes.

## 1. Tuning set and clouds

`bench/scenarios.py::make_tuning_registry(n=24, seed=5000)` draws SSVI and CIR parameters
exactly as `make_registry` does (`_draw_ssvi`, `KAPPAS`, `XIS`, `RHOS`), with
`np.random.default_rng(seed + i)`, and keeps drawing until the set has at least 8 scenarios
with Feller ratio 2κθ/ξ² below 0.3 and at least 8 above 0.7. Scenario ids are `t01`…`t24`.
The registry is deterministic and is not added to `full_registry()`.

`suite/optuna_clouds.py::ensure_cloud(store, sid, settings)` produces, once per tuning
scenario, one MLflow run in experiment `optuna_clouds` keyed by
`{sid, n_particles: 500000, n_steps: 200, seed: 0}` with artifact `cloud.npz` holding, for
each of the four slice times t ∈ {0.10, 0.50, 1.00, 1.75} (the nearest grid step), the
arrays `lnx` and `v_plus = max(v, 0)` of the whole cloud, plus the scenario parameters.
The cloud is simulated by `calibrate_explicit` with the NW estimator at the suite's FULL
explicit settings (500 000 particles, 200 steps, full-cloud fits), i.e. under the
NW-calibrated leverage, so the slices are what an offline pass actually sees. A fixed
permutation (seed 1) splits each slice into the first 400 000 particles (fit pool) and the
last 100 000 (held-out). The NW held-out loss per slice, fitted on the full 400 000 fit pool,
is stored as metrics `nw_loss/t<t>` on the same run.

The held-out loss of a predictor f on a slice is
`mean over held-out particles of (f(lnx) − v_plus)²`, evaluated by linear interpolation of
the fitted values on the slice's quantile grid (`np.quantile(lnx_fit, linspace(0.001, 0.999,
101))`), with the tail rule of the trial applied beyond the grid (see §2). This is the same
grid `calibrate_explicit` uses, so the proxy scores exactly the object the pass consumes.

## 2. Search space and new estimator options

A trial is a dict `recipe` with these keys (Optuna distributions in brackets):

| key | values |
|---|---|
| `hidden` | categorical {32, 64, 128} |
| `depth` | categorical {2, 3, 4} |
| `lr` | log-uniform [1e-4, 1e-2] |
| `batch_size` | categorical {2048, 8192, 0} (0 = full batch) |
| `first_steps` | int [500, 4000], step 250 |
| `later_steps` | int [100, 1500], step 50 |
| `weight_decay` | categorical {0, 1e-6, 1e-5, 1e-4, 1e-3} (Adam `weight_decay`) |
| `fit_subsample` | categorical {100000, 250000, 400000} |
| `warm_start` | categorical {True, False}; False calls `reset()` and re-initialises the net before every slice |
| `mean_match` | categorical {False, True}: after fitting a slice, multiply the fitted grid values by `mean(w·v_plus) / mean(w·f(lnx))` over the fit sample, so the fitted slice reproduces the cloud's own mean variance exactly. The sample mean is used rather than the analytic CIR mean because the Euler cloud's mean differs from the continuous one by the truncation bias, and the pass must be consistent with the cloud it simulates |
| `monotone` / `monotone_penalty` | `monotone` categorical {False, True}; when True, `monotone_penalty` log-uniform [1e-4, 1]: adds `λ · mean(relu(sign(ρ) · df/dz))²` on the batch, penalising slope of the wrong sign. With ρ < 0 the target decreases in log-spot |
| `tail` | categorical {"free", "flat", "linear"}: beyond the grid's outermost quantiles the slice is continued by the network ("free", today's behaviour), by the boundary value ("flat"), or linearly with the slope of the last two grid points ("linear"). Applied in `calibrate_explicit`'s interpolation (`np.interp` already gives "flat"; "linear" and "free" are new) |
| `hetero` | categorical {False, True}: heteroscedastic loss, weights `1 / (var_local + 1e-8)` where `var_local` is the NW local variance estimate of v_plus given lnx on the fit sample (bandwidth as NW), normalised to mean one, multiplied into `weights` |

`NNRegressor` gains `weight_decay=0.0`, `mean_match=False`, `monotone_penalty=0.0` (0 is
off) with `monotone_sign=-1.0` (the sign of the slope to penalise, `sign(ρ)`),
`hetero=False`, `warm_start=True`. `calibrate_explicit`/`Slice` gain `tail` ("flat" default keeps
`np.interp`'s behaviour bit-for-bit; "free" evaluates the network directly on the particles
outside the grid, which requires the estimator to expose `predict`; "linear" extrapolates).
Every default reproduces today's numbers bit for bit; the goldens are the check.

The recipe is applied through one function, `estimators/nn.py::regressor_from_recipe(recipe,
seed)`, used by the search, by the validation bodies, and by the promoted algorithm, so the
three cannot drift apart.

## 3. Objective

`suite/optuna_search.py::score_recipe(recipe, clouds, seed)`:

1. For each tuning scenario in `clouds` (a list of loaded `cloud.npz`), build one regressor
   from the recipe and fit the four slices in order, warm-started unless `warm_start` is
   False, each on a fresh random subsample of `fit_subsample` particles of the fit pool
   (rng seeded by `seed` and the scenario index), with `fit_v_floor` semantics (v_plus).
2. Score each slice by its held-out loss (§1), divided by the stored NW held-out loss of
   that slice; take the log.
3. The trial's score is the mean of these logs over slices and scenarios. Zero is "as good as
   NW", negative is better; −0.1 means 10 % lower held-out squared error.

Per trial, 8 scenarios are drawn from the 24 with `rng(trial.number)` so that every trial
sees a different but reproducible mix; a median pruner reports the running mean after each
scenario and prunes below the median of completed trials at the same scenario count once 10
trials have completed. Fit seconds per slice are recorded.

## 4. MLflow linkage

Experiment `optuna_offline`. One parent run per study, params `{study, n_trials_requested,
tuning_seed, git_hash}`, tag `optuna.study`, artifacts `optuna.db` (the Optuna SQLite
storage, logged at the end and after every 10 trials) and `best.json`. One nested child run
per trial (tag `mlflow.parentRunId`), params = the recipe plus `trial_number`, metrics
`score`, `score/<sid>` per scenario seen, `loss_ratio/<sid>/t<t>` per slice, `fit_s`
(total), tag `optuna.state` ∈ {COMPLETE, PRUNED, FAIL}. The Optuna storage lives at
`results/optuna/<study>.db`; `nparticle optuna run --study <name> --trials N --jobs J`
loads or creates it (`optuna.create_study(storage=..., load_if_exists=True,
direction="minimize", sampler=TPESampler(seed=0), pruner=MedianPruner(n_startup_trials=10))`)
and runs N more trials with J worker processes (`study.optimize(n_jobs=1)` per process, J
processes sharing the SQLite storage, the standard Optuna multi-process pattern). The parent
run is found by `study` name and reused, so a crash or a second invocation continues the
same study. The best recipes are read back from MLflow (`search("optuna_offline")`, state
COMPLETE, sorted by `score`), never from the Optuna file, so the notes and tables have a
single source.

`optuna` is added to the project dependencies.

## 5. Validation and promotion

`nparticle optuna validate --study <name> --top 3 --jobs 4`:

1. Reads the top-k COMPLETE trials by score from MLflow and writes their recipes to
   `results/optuna/<study>_top.json`.
2. For each recipe k, trains the offline body on the 20 SSVI scenarios at 500 000 particles
   (experiment `suite_offline`, body kind `explicit_opt`, extra key param `recipe_hash` so
   different recipes do not collide), through `run_offline` with the regressor from
   `regressor_from_recipe`.
3. Runs the spline head at 80 000 online particles under both lags and two online seeds
   (experiment `suite_budget_tuned`, method `explicit_opt_spline`, budget 80000,
   `recipe_hash` in the key) against the NW re-solve rows already in that experiment.
4. Prints, per recipe, the mean pooled MAE, liquid MAE and online seconds at 80k under each
   lag next to NW's, and marks the winner (lowest mean pooled MAE over both lags).

Promotion: `nparticle optuna promote --study <name> --trial <n>` writes the recipe to
`src/neural_particle_method/estimators/recipes/explicit_opt.json` (checked in). Body kind
`explicit_opt` and cold algorithm `explicit_nn_opt` read that file; `BODY_ROWS`/`COLD_ROWS`
gain a row "Explicit NN, searched" and `HEAD_ROWS` gains "Searched body + spline head",
rendered `--` until the runs exist. The tables then carry the hand-tuned and the searched
recipe side by side.

## 6. Testing

- Tuning registry: 24 ids, deterministic across two calls, at least 8 below Feller 0.3 and 8
  above 0.7, no id in `full_registry()`.
- Clouds (tiny settings): `ensure_cloud` is idempotent (second call returns the same run id),
  `cloud.npz` has the four slices with the expected sizes, `nw_loss/t<t>` metrics present.
- Estimator options, each on a 300-point synthetic slice: `mean_match` reproduces the
  sample mean of v_plus to 1e-6; `monotone` penalty is exactly zero on a fit that is already
  decreasing and positive on an increasing one; `hetero` weights are positive with mean one;
  `warm_start=False` gives the same prediction from two consecutive fits on identical data
  as a fresh regressor; every default reproduces today's predictions bit for bit
  (`assert_array_equal` against the pre-change path, and the goldens untouched).
- Tail rule: "flat" equals `np.interp`; "linear" continues with the end slope; "free" calls
  the regressor's `predict` outside the grid.
- Objective: on a fixture cloud whose held-out loss is computed for NW itself, `score_recipe`
  with a stub regressor that returns NW's predictions scores exactly 0.
- Study (tiny): two trials produce a parent run with two nested children carrying `score`
  and the recipe params; a third invocation with `--trials 1` adds one child to the same
  parent and the Optuna storage reports three trials.
- Validation (tiny): `validate --top 1` on the tiny settings creates one `explicit_opt` body
  and one `explicit_opt_spline` budget run and prints the comparison line.
- Promotion: `promote` writes the JSON and `regressor_from_recipe` of the written file builds
  a regressor with the written knobs.

## 7. Run plan (after the code lands)

1. `nparticle optuna clouds --jobs 4` (24 clouds, about 5 minutes each).
2. `nparticle optuna run --study offline_v1 --trials 200 --jobs 4` overnight.
3. `nparticle optuna validate --study offline_v1 --top 3 --jobs 4` (about 3 hours).
4. Promote the winner, run the cold row (`suite run --stage cold --sids <SSVI>`), refresh the
   tables and the notes with one more row per table and a short subsection in section 5
   describing the study, the winning recipe, and what the priors bought.

## Out of scope

Searching the implicit loop; searching the head; changing the suite's 100k cold budget;
Heston-market scenarios.
