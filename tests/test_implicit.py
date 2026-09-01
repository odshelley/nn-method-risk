import numpy as np

from neural_particle_method.implicit import calibrate_implicit

# AMENDMENT (controller-approved): xi=0.05 instead of the brief's 0.4. At the
# brief's xi=0.4, the calibrated leverage L = 0.2/sqrt(E[V|X=x]) genuinely
# varies with x under correlation for a FLAT Dupire surface, so |L-1|<0.2 in
# the mid-band is false at that vol-of-vol. With xi=0.05, V stays close to
# v0=0.04 so E[V|X] ~= 0.04 and L ~= 1 genuinely holds in the mid-band.
DYN = {"kappa": 2.0, "theta": 0.04, "xi": 0.05, "rho": -0.5, "v0": 0.04}


class FlatDupire:
    T_grid = np.array([0.004, 1.0])

    def sigma(self, t, x, s0=1.0):
        return np.full(np.shape(np.log(np.asarray(x))), 0.2)


def test_damped_update_is_contraction_on_toy():
    # scalar analogue: Phi(L) = 0.2/sqrt(f(L)) with f pinned at 0.04 -> fixed point 1
    L, alpha = 3.0, 0.5
    for _ in range(20):
        L = (1 - alpha) * L + alpha * 1.0
    assert abs(L - 1.0) < 1e-5


def test_implicit_flat_recovery_smoke():
    recs, info = calibrate_implicit(FlatDupire(), DYN, T=0.5, n_steps=6,
                                    n_particles=8_000, alpha=0.5, n_iters=3,
                                    seed=1, pool_subsample=10_000, fit_steps=150)
    assert len(recs) == 6
    t, grid, Lg, fg = recs[3]
    mid = np.abs(grid) < 0.3
    assert np.abs(Lg[mid] - 1.0).max() < 0.2
    assert len(info["deltas"]) == 3 and info["deltas"][-1] <= info["deltas"][0] + 0.05
