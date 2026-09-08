# neural_particle_method

Neural L2 calibration of LSV models (Risk paper workspace).

## Layout

- `src/neural_particle_method/` — the package; `cli.py` exposes the `nparticle` console script
  - `market` — Black-Scholes, Heston, SSVI, Dupire local vol
  - `simulate` — Heston-plus-leverage Euler step and the `LeverageField`
  - `estimators` — conditional-expectation estimators behind one `fit_predict` interface
  - `calibrate` — explicit and implicit schemes, mixture importance sampling, warm-start helpers
  - `pricing` — fresh-seed repricing and IV metrics
  - `tracking` — MLflow store wrapper and the legacy importer
  - `bench` — the paper's benchmark sweep, including `bench/sensitivity.py` (knob sweeps) and
    `bench/acceptance.py` (BASELINES.md card criteria); scenarios include the `HestonMarketSpec`
    family (`li_simple`, `li_complex`, `bayer`)
  - `experiments` — the bump-and-correct and warm-start suites
  - `figures` — figures 1-4
  - `reference` — Fokker-Planck reference leverage solver, the accuracy anchor; see `docs/pde_reference.md`
- Paper notes: `paper/`; specs and plans under `docs/superpowers/`.

## Tracking

All runs go to MLflow. Default store: `sqlite:///mlruns.db` and `mlartifacts/` in the working directory
(both gitignored). Set `MLFLOW_TRACKING_URI` to use a server. Browse with `uv run mlflow ui --backend-store-uri sqlite:///mlruns.db`.

## Commands

- `uv run nparticle list`
- `uv run nparticle run --scenario s01 --algo ridge --n 200000 --seed 1`
- `uv run nparticle sweep --preset paper --jobs 4` (resumable: finished runs are skipped)
- `uv run nparticle sweep --preset baselines --jobs 4` — every estimator at N in {1e3, 1e4, 1e5} on the SSVI grid (resumable; about 2 000 runs)
- `uv run nparticle sensitivity --jobs 4` — one knob per method around its published default on `s01` and `li_simple`
- `uv run nparticle acceptance` — the BASELINES.md card criteria on the Heston market family
- `uv run nparticle aggregate` -> `results/summary.csv`, `results/digest.md` (committed snapshots)
- `uv run nparticle figures` -> `figures/out/`; now also writes `fig1_baselines.pdf`, `fig3_baselines.pdf`, `fig5_sensitivity.pdf`
- `uv run nparticle experiment bump|warm [--smoke]`; `uv run nparticle warm-summary`
- `uv run nparticle import-legacy` (one-shot import of the archived JSON results under `results/`)

## Tests

- `uv run pytest` — fast suite (about 30 s)
- `uv run pytest -m golden` — bit-for-bit replay of four archived runs (minutes)
- `uv run pytest -m slow` — smoke runs of every experiment arm
- `uv run pytest -m slow tests/acceptance` — full acceptance-card runs for BASELINES.md
- `uv run pytest --cov=neural_particle_method` — coverage report
- The golden replays read archived run JSONs under `results/runs/` (gitignored). A fresh clone or
  worktree needs a copy of `results/runs/` from a checkout that has them before `uv run pytest -m golden` can pass.
