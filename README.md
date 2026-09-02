# neural_particle_method

Neural L2 calibration of LSV models (Risk paper workspace).

- Paper notes: `paper/notes.tex`; spec and plans under `docs/superpowers/`.
- Benchmark: `uv run bench list | run | sweep --preset paper --jobs 4 | aggregate | figures`.
- One JSON per (scenario, algo, N, seed) under `results/runs/` (gitignored); `results/summary.csv` and `figures/out/` are the committed artifacts.
- Tests: `uv run pytest` (fast, no large simulations).
- Legacy exploratory results are archived in `experiments/results/`.
