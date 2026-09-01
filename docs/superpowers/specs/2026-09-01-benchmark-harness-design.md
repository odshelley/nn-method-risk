# Benchmark harness for LSV calibration algorithms — design

Date: 2026-09-01. Status: approved by Osian (sections presented and accepted in session).

## Purpose

A structured, CLI-driven benchmark inside this repository that tests the paper's four algorithms plus a kernel baseline on synthetic surfaces, with implied vol from repricing vanillas as the metric. Output feeds the Risk paper's Figures 1-4.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Ground truth | SSVI surface generator (arbitrage-free draws); target IVs are exact SSVI values |
| Contenders | nw, explicit_nn, ridge, explicit_nn_is (defensive mixture), implicit_nn |
| Metric | Fresh-seed MC reprice, IV error in vol bp vs SSVI target; RMSE + max, per maturity, pooled, and wings-only |
| Scale | 20 SSVI scenarios, each paired with one dynamics setting sampled from the (xi, rho) grid; N in {50k, 200k}, 3 seeds. Fig 3 additionally runs one representative surface against the full (xi, rho) cross |
| Structure | Option A: benchmark package inside this repo, CLI-driven, resumable runs |
| Dependencies | QuantLib permitted and used (implied vol, LocalVolSurface cross-check, Heston proxy moments) |

## Repository layout

```
src/neural_particle_method/    # library; existing modules kept
  ssvi.py        # SSVI: params -> total variance w(k,T); Gatheral-Jacquier no-arb checks;
                 #   analytic Dupire (exact k-derivatives, FD in T); QuantLib cross-check
  implicit.py    # damped global scheme (paper alg 2)
  importance.py  # defensive-mixture IS (paper alg 4): offline design (alpha, theta from q*),
                 #   tilted simulation drift, balance-heuristic weights
  reprice.py     # fresh-seed MC pricer + implied vol inversion (QuantLib)
bench/
  scenarios.py   # ScenarioSpec dataclass + registry (s01..s20 x dynamics grid); seeded draws
  algos.py       # AlgoSpec registry wrapping the five contenders behind one interface
  runner.py      # one (scenario, algo, N, seed) -> results/runs/<scenario>/<algo>/n<N>_s<seed>.json
  aggregate.py   # run JSONs -> results/summary.csv + markdown digest
  cli.py         # entry point (uv run bench)
figures/         # fig1..fig4 scripts; read summary.csv (fig2 also reads wing-detail JSONs)
results/runs/    # gitignored; summary.csv and figures committed
tests/
```

The three legacy `experiments/exp_*.py` scripts retire once their coverage is reproduced.

## CLI

Registered via `[project.scripts]`:

- `bench list` — scenarios and algorithms with one-line descriptions
- `bench run --scenario s07 --algo ridge --n 200000 --seed 1`
- `bench sweep --preset paper [--jobs K]` — enumerates the grid, skips runs whose JSON exists (interruption-proof), optional process parallelism
- `bench aggregate` — deterministic, no simulation
- `bench figures` — regenerates figures byte-identically from committed summary.csv

## Scenario generation

Each scenario = SSVI surface + Heston dynamics. SSVI in Gatheral-Jacquier power-law form, phi(theta) = eta / theta^gamma; parameters (ATM term structure, eta, gamma, rho_ssvi) drawn from realistic ranges under a per-scenario seed. Butterfly and calendar no-arb conditions checked at generation; violating draws resampled with a bounded retry count. Dupire local vol from analytic SSVI k-derivatives (FD only in T), validated against QuantLib LocalVolSurface in tests. Heston dynamics (kappa, theta-bar, xi, rho, v0) from a small explicit grid; xi and rho are the stress axes swept by Fig 3.

## Algorithm interface

`calibrate(dupire, dynamics, n_particles, seed, cfg) -> CalibResult` with leverage on a fixed (t, x) grid, simulate/fit timing split, and per-algo diagnostics. Contenders:

- `nw` — Silverman-bandwidth Nadaraya-Watson (classical reference)
- `explicit_nn` — per-slice network, warm-started (paper alg 1)
- `ridge` — frozen body + per-slice ridge head (paper alg 3)
- `explicit_nn_is` — explicit NN under the defensive mixture (paper alg 4); ESS per bucket and realised weight bound logged; offline design (alpha, theta from q*) computed from the Heston proxy in importance.py
- `implicit_nn` — damped iteration (paper alg 2), initialised from an explicit pass; alpha and iteration budget in cfg

## Metric protocol

Fresh-seed MC (independent RNG stream, N_reprice = 500k, same Euler grid) prices vanillas under the calibrated leverage; QuantLib inverts to IV. Error = model IV - SSVI target IV in vol bp on moneyness 0.6-1.6 (13 log-spaced strikes) x maturities {0.25, 0.5, 1, 2}. Reported: RMSE and max per maturity, pooled, and wings-only (|log-moneyness| > 0.25). Failed inversions become NaN, excluded from statistics, count reported. Timings: simulate, fit, total wall-clock.

## Run artifacts and aggregation

Every run writes one self-describing JSON: full config echo, git hash, seed, timings, metrics, diagnostics, `status: ok|failed` (failed carries the traceback). `bench aggregate` flattens all runs into one tidy summary.csv row per (scenario, algo, N, seed) plus a markdown digest (best algo per scenario, failure counts).

## Figures

Fig 1: pooled accuracy across algorithms at fixed budget. Fig 2: wing IV error with and without the tilt, plus ESS (the central figure). Fig 3: (xi, rho)-plane heatmap, kernel vs network. Fig 4: latency-accuracy frontier (ridge/refresh settings).

## Error handling

Failed runs never kill a sweep. No-arb violations resample with bounded retries. Leverage clamps ([0, 4]) counted and logged per run; heavy clamping flags a broken calibration.

## Testing

Under one minute via `uv run pytest`: SSVI no-arb checker rejects known-bad params; analytic SSVI-Dupire vs QuantLib LocalVolSurface on a skewed surface; flat-surface end-to-end recovery for every algo at small N; mixture weight bound w <= 1/alpha_0 and E_Q[w] approx 1; implicit iteration contracts on a toy case; CLI smoke test (tiny run end to end, JSON schema validated). Existing four sanity tests kept.

## Out of scope

Real market surfaces (Risk paper Table 1), PURBF and bin-MC baselines, PDE reference solver, multi-asset LSV. Each can be added later as a new AlgoSpec or ScenarioSpec without structural change.
