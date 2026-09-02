# Benchmark results — main sweep (2026-09-02)

756 runs: 20 SSVI scenarios x 6 algorithms x N in {50k, 200k} x 3 seeds, plus the (xi, rho) stress cross. Metric: fresh-seed IV repricing error vs the SSVI target, vol basis points.

## Table 1 — pooled accuracy and latency by algorithm (mean over scenarios and seeds)

| Algorithm | N | Pooled RMSE (bp) | Pooled max (bp) | Wings RMSE (bp) | Intraday calib. time (s) |
|---|---|---|---|---|---|
| implicit_nn | 50k | 154.0 | 559.7 | 196.2 | 149.5 |
| implicit_nn | 200k | 154.0 | 550.7 | 196.5 | 154.9 |
| implicit_ridge | 50k | 330.1 | 827.7 | 367.8 | 0.8 |
| implicit_ridge | 200k | 334.3 | 835.9 | 371.4 | 1.3 |
| spline | 50k | 335.0 | 841.8 | 372.7 | 0.4 |
| nw | 50k | 335.3 | 861.8 | 375.8 | 0.9 |
| spline | 200k | 338.6 | 840.7 | 376.1 | 1.0 |
| ridge | 50k | 339.6 | 862.9 | 379.4 | 6.2 |
| nw | 200k | 339.8 | 861.2 | 380.4 | 1.5 |
| ridge | 200k | 346.3 | 875.8 | 387.1 | 6.9 |
| explicit_nn | 50k | 379.1 | 944.2 | 426.4 | 96.4 |
| explicit_nn_is | 50k | 380.6 | 937.8 | 427.0 | 105.4 |
| explicit_nn_is | 200k | 381.1 | 942.6 | 427.1 | 102.0 |
| explicit_nn | 200k | 381.3 | 943.5 | 428.2 | 89.8 |

implicit_ridge's time is the intraday ridge sweep only; its implicit solve (~142 s mean) is offline cost, reported in Table 4. All other algorithms calibrate fully in-line.

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

Notes: 50k vs 200k differ by <2 percent everywhere (bias-dominated regime); importance sampling matches explicit_nn within noise on this quote grid (see analysis in session log); spline (P-spline head) runs at ridge-class latency and is statistically indistinguishable from NW (paired +1.2 bp, t=0.5), slightly ahead of the frozen-feature ridge (paired +7.7 bp, t=2.6).

## Table 5 — warm intraday recalibration on a bumped surface (10 pairs: 5 scenarios x 2 seeds, N=200k)

Protocol: overnight implicit solve + head distillation on surface S; surface bumped to S' (sigma0 +0.01, eta x0.95, rho +0.03); strategies reprice against S'. Cold-calibration Table 1 has no overnight state, so these rows are not comparable to it row-by-row.

| Intraday strategy | Pooled RMSE (bp) | Intraday cost (s) |
|---|---|---|
| stale f, fresh Dupire (free Jacobi step) | 142.1 | 0.00 |
| 1 damped beta-correction | 169.3 | 0.64 |
| 2 damped beta-corrections | 155.9 | 1.26 |
| stale L (no action) | 192.9 | 0.00 |
| full re-solve (warm explicit + implicit) | 186.4 | 52.99 |
| causal ridge sweep | 327.0 | 0.97 |

Overnight cost (implicit solve + distillation): 53 s mean. The free Jacobi step beats the full re-solve by 44 bp paired (t = 2.4) because the conditional expectation is second-order stable under surface bumps while re-solving re-incurs the full calibration noise. Beta-corrections match it within noise here; they are expected to matter on larger moves or dynamics changes (untested).
