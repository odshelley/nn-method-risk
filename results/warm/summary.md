# Warm-start suite summary

## dyn (12 runs)
| strategy | mean rmse bp | mean build s |
|---|---|---|
| anchor_unbumped | 287 | |
| kappa_dn/do_nothing | 199 | 0.0 |
| kappa_dn/full_resolve | 242 | 45.5 |
| kappa_dn/kalman_1 | 226 | 0.5 |
| kappa_dn/kalman_2 | 245 | 1.0 |
| kappa_dn/ridge_1 | 180 | 0.5 |
| kappa_dn/ridge_2 | 178 | 1.1 |
| rho_dn/do_nothing | 170 | 0.0 |
| rho_dn/full_resolve | 229 | 45.4 |
| rho_dn/kalman_1 | 221 | 0.5 |
| rho_dn/kalman_2 | 236 | 1.0 |
| rho_dn/ridge_1 | 156 | 0.5 |
| rho_dn/ridge_2 | 161 | 1.1 |
| xi_up/do_nothing | 219 | 0.0 |
| xi_up/full_resolve | 268 | 45.5 |
| xi_up/kalman_1 | 249 | 0.5 |
| xi_up/kalman_2 | 250 | 1.0 |
| xi_up/ridge_1 | 199 | 0.5 |
| xi_up/ridge_2 | 199 | 1.0 |

## xover (6 runs)
| strategy | mean rmse bp | mean build s |
|---|---|---|
| s0.5/full_resolve | 212 | 53.2 |
| s0.5/refresh | 143 | 0.0 |
| s0.5/ridge_1 | 128 | 0.7 |
| s1/full_resolve | 196 | 45.4 |
| s1/refresh | 171 | 0.0 |
| s1/ridge_1 | 179 | 0.5 |
| s2/full_resolve | 196 | 52.9 |
| s2/refresh | 133 | 0.0 |
| s2/ridge_1 | 122 | 0.7 |
| s4/full_resolve | 171 | 45.4 |
| s4/refresh | 173 | 0.0 |
| s4/ridge_1 | 174 | 0.6 |

## norm (6 runs)
- s01 s0: |A| = 0.1092, predicted error fraction 0.0984, history [0.5608, 0.0599, 0.0488, 0.1092]
- s02 s0: |A| = 0.0478, predicted error fraction 0.0456, history [14.8692, 0.4537, 0.316, 0.0478]
- s03 s0: |A| = 0.5094, predicted error fraction 0.3375, history [50.491, 0.3464, 0.0875, 0.5094]
- s04 s0: |A| = 0.3486, predicted error fraction 0.2585, history [0.6088, 0.0496, 0.275, 0.3486]
- s05 s0: |A| = 0.2541, predicted error fraction 0.2026, history [0.9123, 0.0236, 0.2075, 0.2541]
- s01 s0: |A| = 0.0279, predicted error fraction 0.0271, history [1.3206, 0.0279]

## seq (6 runs)
| step | refresh | ridge | kalman | kalman_spline | resolve_warm |
|---|---|---|---|---|---|
| 0 | 188 | 233 | 295 | 277 | 209 |
| 1 | 170 | 185 | 249 | 259 | 171 |
| 2 | 145 | 203 | 223 | 261 | 213 |
cold re-solve at final step: 168 bp
