# Benchmark results — main sweep (2026-09-02)

756 runs: 20 SSVI scenarios x 6 algorithms x N in {50k, 200k} x 3 seeds, plus the (xi, rho) stress cross. Metric: fresh-seed IV repricing error vs the SSVI target, vol basis points.

## Table 1 — pooled accuracy and latency by algorithm (mean over scenarios and seeds)

| Algorithm | N | Pooled RMSE (bp) | Pooled max (bp) | Wings RMSE (bp) | Calib. time (s) |
|---|---|---|---|---|---|
| implicit_nn | 50k | 154.0 | 559.7 | 196.2 | 149.5 |
| implicit_nn | 200k | 154.0 | 550.7 | 196.5 | 154.9 |
| implicit_ridge | 50k | 330.1 | 827.7 | 367.8 | 139.6 |
| implicit_ridge | 200k | 334.3 | 835.9 | 371.4 | 143.0 |
| nw | 50k | 335.3 | 861.8 | 375.8 | 0.9 |
| ridge | 50k | 339.6 | 862.9 | 379.4 | 6.2 |
| nw | 200k | 339.8 | 861.2 | 380.4 | 1.5 |
| ridge | 200k | 346.3 | 875.8 | 387.1 | 6.9 |
| explicit_nn | 50k | 379.1 | 944.2 | 426.4 | 96.4 |
| explicit_nn_is | 50k | 380.6 | 937.8 | 427.0 | 105.4 |
| explicit_nn_is | 200k | 381.1 | 942.6 | 427.1 | 102.0 |
| explicit_nn | 200k | 381.3 | 943.5 | 428.2 | 89.8 |

## Table 2 — RMSE by maturity (bp, N=200k)

| T | implicit_nn | implicit_ridge | nw | ridge | explicit_nn | explicit_nn_is |
|---|---|---|---|---|---|---|
| 0.25 | 264 | 396 | 410 | 420 | 453 | 451 |
| 0.5 | 140 | 272 | 277 | 280 | 330 | 330 |
| 1.0 | 70 | 276 | 280 | 283 | 320 | 321 |
| 2.0 | 43 | 324 | 321 | 325 | 352 | 353 |

## Table 3 — vol-of-vol/correlation stress cross (pooled RMSE bp, N=200k)

| Scenario | explicit_nn | nw |
|---|---|---|
| f_xi0.3_rho-0.3 | 53.0 | 52.1 |
| f_xi0.3_rho-0.7 | 55.5 | 50.9 |
| f_xi0.6_rho-0.3 | 87.4 | 87.5 |
| f_xi0.6_rho-0.7 | 164.0 | 154.1 |
| f_xi1.0_rho-0.3 | 232.0 | 228.2 |
| f_xi1.0_rho-0.7 | 734.1 | 648.4 |

## Table 4 — implicit_ridge overnight/intraday split (N=200k)

Mean overnight (implicit solve + distillation): 142 s. Mean intraday (causal ridge sweep): 1.34 s.

Notes: 50k vs 200k differ by <2 percent everywhere (bias-dominated regime); importance sampling matches explicit_nn within noise on this quote grid (see analysis in session log); spline algorithm added after this sweep, results pending.
