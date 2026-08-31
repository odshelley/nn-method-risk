"""Exp A: estimator accuracy against an analytic conditional expectation.

X = exp(sx*Z), V = v0*exp(rv*sv*Z + sqrt(1-rv^2)*sv*W - sv^2/2)
=> E[V | Z=z] = v0*exp(rv*sv*z - rv^2*sv^2/2), and lnX = sx*Z so conditioning on X is
conditioning on Z. Truth is analytic; measure RMSE of NW and NN on interior quantiles.
"""
import json
import numpy as np
from neural_particle_method.condexp import NNRegressor, nw_estimate

sx, sv, rv, v0 = 0.25, 0.8, -0.7, 0.04
rng = np.random.default_rng(1)
results = {}
for N in [2_000, 10_000, 50_000]:
    z = rng.standard_normal(N)
    wperp = rng.standard_normal(N)
    lnx = sx * z
    v = v0 * np.exp(rv * sv * z + np.sqrt(1 - rv ** 2) * sv * wperp - 0.5 * sv ** 2)
    grid = np.quantile(lnx, np.linspace(0.02, 0.98, 97))
    truth = v0 * np.exp(rv * sv * grid / sx - 0.5 * rv ** 2 * sv ** 2)

    nw = nw_estimate(lnx, v, grid)
    reg = NNRegressor(seed=0)
    reg.fit(lnx, v, steps=500)
    nn = reg.predict(grid)

    rmse = lambda est: float(np.sqrt(np.mean((est - truth) ** 2)) / v0)
    results[N] = {"nw_rmse_rel": rmse(nw), "nn_rmse_rel": rmse(nn)}
    print(N, results[N])

json.dump(results, open("experiments/results/exp_a.json", "w"), indent=2)
