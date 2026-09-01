import numpy as np
from neural_particle_method.importance import MixtureDesign, design_mixture
from neural_particle_method.explicit import calibrate_explicit

DYN = {"kappa": 2.0, "theta": 0.04, "xi": 0.5, "rho": -0.6, "v0": 0.04}
# Flat Dupire implies L ~= 1 only when E[V|X] ~= v0 uniformly, i.e. at low vol-of-vol.
# At DYN's xi=0.5, rho=-0.6, E[V|X] genuinely varies in X (down-moves associate with
# high V), so the calibrated L is correctly far from 1 in the wings; that's not a defect.
DYN_LOWVOV = dict(DYN, xi=0.05)

class FlatDupire:
    T_grid = np.array([0.004, 1.0])
    def sigma(self, t, x, s0=1.0):
        return np.full(np.shape(np.log(np.asarray(x))), 0.2)

def test_design_caps_cost():
    d = design_mixture(DYN, T=1.0)
    assert d.alphas[1] == 0.5 and d.thetas[1] == 0.0
    assert d.thetas[2] > 0 > d.thetas[0]
    cost = d.thetas[2] ** 2 * 1.0 / (1 - DYN["rho"] ** 2)
    assert cost <= 3.0 + 1e-9

def test_weights_bounded_and_normalised():
    d = design_mixture(DYN, T=0.5)
    _, info = calibrate_explicit(FlatDupire(), DYN, T=0.5, n_steps=8,
                                 n_particles=20_000, method="nw",
                                 fit_subsample=5_000, seed=3, mixture=d)
    w = info["weights"]
    assert w.max() <= 1 / d.alphas[1] + 1e-9
    assert abs(w.mean() - 1.0) < 0.05

def test_flat_recovery_with_mixture():
    # At low vol-of-vol, E[V|X] ~= v0 uniformly, so flat Dupire has a known ground
    # truth L ~= 1; this isolates whether tilting introduces bias.
    d = design_mixture(DYN_LOWVOV, T=0.5)
    _, info = calibrate_explicit(FlatDupire(), DYN_LOWVOV, T=0.5, n_steps=8,
                                 n_particles=20_000, method="nw",
                                 fit_subsample=5_000, seed=3, mixture=d)
    t, grid, Lg, _ = info["L_records"][-1]
    mid = np.abs(grid) < 0.3
    assert np.abs(Lg[mid] - 1.0).max() < 0.15
    assert "fit_s" in info and info["is_diag"]["max_w"] <= 2.0 + 1e-9

def test_flat_recovery_untilted_control():
    # Control: the untilted run at the same low vol-of-vol dynamics must also recover
    # L ~= 1, so test_flat_recovery_with_mixture demonstrably isolates the tilt's effect
    # rather than relying on properties only the mixture run happens to have.
    _, info = calibrate_explicit(FlatDupire(), DYN_LOWVOV, T=0.5, n_steps=8,
                                 n_particles=20_000, method="nw",
                                 fit_subsample=5_000, seed=3, mixture=None)
    t, grid, Lg, _ = info["L_records"][-1]
    mid = np.abs(grid) < 0.3
    assert np.abs(Lg[mid] - 1.0).max() < 0.15
