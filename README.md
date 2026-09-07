# neural_particle_method

Neural L2 calibration of LSV models: benchmark harness and experiment suite.

- Benchmark CLI: `uv run bench list | run | sweep --preset paper --jobs 4 | aggregate | figures`.
- One JSON per (scenario, algo, N, seed) lands under `results/runs/`; aggregate builds the summary tables.
- Warm-start suite: `uv run python experiments/warm_suite.py smoke | run | aggregate`.
- Tests: `uv run pytest` (fast, no large simulations).
