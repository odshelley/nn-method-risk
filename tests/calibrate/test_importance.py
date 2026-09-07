import numpy as np
import pytest

from neural_particle_method.calibrate.config import ExplicitConfig
from neural_particle_method.calibrate.explicit import calibrate_explicit
from neural_particle_method.calibrate.importance import MixtureDesign, design_mixture
from neural_particle_method.estimators import make_estimator
from neural_particle_method.pricing.reprice import mc_smile

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
    cfg = ExplicitConfig(n_steps=8, n_particles=20_000, fit_subsample=5_000)
    r = calibrate_explicit(FlatDupire(), DYN, make_estimator("nw"), cfg, T=0.5, seed=3, mixture=d)
    w = r.weights
    assert w.max() <= 1 / d.alphas[1] + 1e-9
    assert abs(w.mean() - 1.0) < 0.05

def test_flat_recovery_with_mixture():
    # At low vol-of-vol, E[V|X] ~= v0 uniformly, so flat Dupire has a known ground
    # truth L ~= 1; this isolates whether tilting introduces bias.
    d = design_mixture(DYN_LOWVOV, T=0.5)
    cfg = ExplicitConfig(n_steps=8, n_particles=20_000, fit_subsample=5_000)
    r = calibrate_explicit(FlatDupire(), DYN_LOWVOV, make_estimator("nw"), cfg, T=0.5, seed=3, mixture=d)
    s = r.field[-1]
    mid = np.abs(s.grid) < 0.3
    assert np.abs(s.L[mid] - 1.0).max() < 0.15
    assert r.fit_s is not None and r.is_diag["max_w"] <= 2.0 + 1e-9

def test_mixture_weighted_pricing_matches_untilted_reference():
    # DYN_LOWVOV's L-recovery test can't detect a broken weighting scheme: at low
    # vol-of-vol the tilt barely moves the leverage estimate whether or not weights are
    # applied correctly (or even at all). Discriminate on terminal pricing at the
    # original DYN (xi=0.5) instead: the WEIGHTED average over the tilted cloud must
    # recover the untilted reference smile, while the UNWEIGHTED average over the same
    # tilted cloud is grossly biased (the tilt pushes mass into the wings without
    # correcting for it).
    d = design_mixture(DYN, T=0.5)
    cfg = ExplicitConfig(n_steps=8, n_particles=20_000, fit_subsample=5_000)
    r_ref = calibrate_explicit(FlatDupire(), DYN, make_estimator("nw"), cfg, T=0.5, seed=3, mixture=None)
    r_mix = calibrate_explicit(FlatDupire(), DYN, make_estimator("nw"), cfg, T=0.5, seed=3, mixture=d)

    K_grid = [0.8, 1.0, 1.2]
    ref = mc_smile(r_ref.lnx, K_grid)
    weighted = mc_smile(r_mix.lnx, K_grid, weights=r_mix.weights)
    unweighted = mc_smile(r_mix.lnx, K_grid)

    rel_weighted = np.abs(weighted - ref) / ref
    rel_unweighted = np.abs(unweighted - ref) / ref
    assert rel_weighted.max() < 0.1, rel_weighted
    # 1.2 wing: unweighted mixture-cloud pricing is grossly wrong (empirically ~270%).
    assert rel_unweighted[-1] > 0.5, rel_unweighted

def test_flat_recovery_untilted_control():
    # Control: the untilted run at the same low vol-of-vol dynamics must also recover
    # L ~= 1, so test_flat_recovery_with_mixture demonstrably isolates the tilt's effect
    # rather than relying on properties only the mixture run happens to have.
    cfg = ExplicitConfig(n_steps=8, n_particles=20_000, fit_subsample=5_000)
    r = calibrate_explicit(FlatDupire(), DYN_LOWVOV, make_estimator("nw"), cfg, T=0.5, seed=3, mixture=None)
    s = r.field[-1]
    mid = np.abs(s.grid) < 0.3
    assert np.abs(s.L[mid] - 1.0).max() < 0.15

def test_spline_rejects_mixture():
    d = design_mixture(DYN, T=0.5)
    cfg = ExplicitConfig(n_steps=4, n_particles=2_000, fit_subsample=500)
    with pytest.raises(ValueError, match="importance weights"):
        calibrate_explicit(FlatDupire(), DYN, make_estimator("spline"), cfg, T=0.5, seed=3, mixture=d)

def test_snapshot_weights_recorded_under_mixture():
    d = design_mixture(DYN, T=0.5)
    cfg = ExplicitConfig(n_steps=8, n_particles=2_000, fit_subsample=500, snapshot_times=(0.25,))
    r = calibrate_explicit(FlatDupire(), DYN, make_estimator("nw"), cfg, T=0.5, seed=3, mixture=d)
    assert set(r.snapshot_weights) == set(r.snapshots)
    sw = r.snapshot_weights[0.25]
    assert sw.shape == r.snapshots[0.25].shape
    assert np.all(sw > 0) and sw.max() <= 1 / d.alphas[1] + 1e-9
