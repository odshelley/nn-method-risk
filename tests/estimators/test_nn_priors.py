import numpy as np
import pytest
import torch

from neural_particle_method.estimators.nadaraya_watson import nw_local_variance
from neural_particle_method.estimators.nn import HETERO_W_MAX, NNRegressor, SliceNet

rng = np.random.default_rng(0)
LNX = rng.normal(0.0, 0.3, 300)
V = 0.04 * np.exp(-2.0 * LNX) * (1 + 0.3 * rng.normal(size=300))   # decreasing target
GRID = np.linspace(-0.8, 0.8, 41)


def test_defaults_are_bit_for_bit():
    a = NNRegressor(seed=0, first_steps=5, later_steps=2).fit_predict(0.5, LNX, V, GRID)
    b = NNRegressor(seed=0, first_steps=5, later_steps=2, weight_decay=0.0, warm_start=True,
                    mean_match=False, monotone_penalty=0.0, hetero=False).fit_predict(
        0.5, LNX, V, GRID)
    np.testing.assert_array_equal(a, b)


def test_log_scale_buffer_defaults_to_identity_and_loads_old_state():
    net = SliceNet()
    assert "log_scale" in net.state_dict() and float(net.log_scale) == 0.0
    old = {k: v for k, v in net.state_dict().items() if k != "log_scale"}
    fresh = SliceNet()
    fresh.load_state_dict(old, strict=False)
    z = torch.zeros(3, 1)
    torch.testing.assert_close(fresh(z), net(z))


def test_mean_match_reproduces_the_sample_mean():
    est = NNRegressor(seed=0, first_steps=20, later_steps=5, mean_match=True)
    est.fit_predict(0.5, LNX, V, GRID)
    assert abs(est.predict(LNX).mean() - V.mean()) < 1e-6 * V.mean()


def test_mean_match_is_weighted():
    w = rng.uniform(0.5, 1.5, 300)
    est = NNRegressor(seed=0, first_steps=20, later_steps=5, mean_match=True)
    est.fit_predict(0.5, LNX, V, GRID, weights=w)
    assert abs(np.average(est.predict(LNX), weights=w) - np.average(V, weights=w)) < 1e-6 * V.mean()


def test_monotone_penalty_is_zero_on_a_decreasing_fit_and_positive_on_an_increasing_one():
    # lambda = 0.01, not 1.0: the penalty is normalised by V_SCALE ** 2 while the data loss is on
    # the raw V_SCALE ** 2 scale, so lambda = 1.0 weights the prior ~600x the data, flattens the
    # fit outright and leaves a violation of ~1e-6 at every step budget from 100 to 1600. At 0.01
    # the prior and the data actually trade off, which is what the penalty has to register.
    dec = NNRegressor(seed=0, first_steps=200, later_steps=5, monotone_penalty=0.01,
                      monotone_sign=1.0)
    dec.fit_predict(0.5, LNX, V, GRID)
    assert dec.last_penalty < 1e-6
    inc = NNRegressor(seed=0, first_steps=200, later_steps=5, monotone_penalty=0.01,
                      monotone_sign=-1.0)
    inc.fit_predict(0.5, LNX, V, GRID)
    assert inc.last_penalty > 1e-4


def test_warm_start_off_refits_from_scratch():
    cold = NNRegressor(seed=0, first_steps=5, later_steps=2, warm_start=False)
    a = cold.fit_predict(0.5, LNX, V, GRID)
    b = cold.fit_predict(0.6, LNX, V, GRID)
    fresh = NNRegressor(seed=0, first_steps=5, later_steps=2).fit_predict(0.5, LNX, V, GRID)
    np.testing.assert_array_equal(a, fresh)
    assert not np.array_equal(a, b)        # re-initialised with a different seed offset
    assert cold._n_fits == 2


def test_hetero_weights_are_positive_with_mean_one():
    var = nw_local_variance(LNX, V)
    assert var.shape == (300,) and (var > 0).all()
    est = NNRegressor(seed=0, first_steps=3, later_steps=1, hetero=True)
    est.fit_predict(0.5, LNX, V, GRID)
    w = est.last_weights
    assert w.shape == (300,) and (w > 0).all() and abs(w.mean() - 1.0) < 1e-9


def test_weight_decay_reaches_the_optimiser():
    est = NNRegressor(weight_decay=1e-4)
    assert est.opt.param_groups[0]["weight_decay"] == pytest.approx(1e-4)


def test_mean_match_is_skipped_when_the_weighted_target_mean_is_not_positive():
    """A non-positive target mean has no scale to match; the fit must be left alone."""
    zero = np.zeros_like(V)
    est = NNRegressor(seed=0, first_steps=5, later_steps=2, mean_match=True)
    got = est.fit_predict(0.5, LNX, zero, GRID)
    plain = NNRegressor(seed=0, first_steps=5, later_steps=2).fit_predict(0.5, LNX, zero, GRID)
    assert float(est.net.log_scale) == 0.0
    np.testing.assert_array_equal(got, plain)


def test_hetero_weights_are_capped(monkeypatch):
    """A near-noiseless region sends 1 / var to the floor's reciprocal; the cap holds it down."""
    import neural_particle_method.estimators.nn as N
    n = 2000
    x = rng.normal(0.0, 0.3, n)
    v = 0.04 * np.exp(-2.0 * x)
    var = np.full(n, 0.01)
    var[0] = 1e-8                                     # a near-zero-variance region
    monkeypatch.setattr(N, "nw_local_variance", lambda *a, **k: var)
    raw = (1.0 / var) / (1.0 / var).mean()
    assert raw.max() > HETERO_W_MAX, "the fixture must actually exercise the cap"
    est = NNRegressor(seed=0, first_steps=3, later_steps=1, hetero=True)
    est.fit_predict(0.5, x, v, GRID)
    assert est.last_weights.max() <= HETERO_W_MAX
    assert (est.last_weights[1:] > 0).all()


def test_design_weights_equalise_a_sparse_wing():
    """A dense bulk plus a sparse rising wing: plain least squares ignores the wing; the
    density-equalised weights recover it, and both stay unbiased where the data are dense."""
    import numpy as np

    from neural_particle_method.estimators.nn import NNRegressor, design_weights

    rng = np.random.default_rng(0)
    bulk = rng.normal(-0.1, 0.08, 20_000)
    wing = rng.uniform(0.15, 0.5, 300)
    x = np.concatenate([bulk, wing])
    f_true = np.where(x < 0.05, 0.2 * np.clip(0.05 - x, 0, None), 0.15 * (x - 0.05))
    v = f_true + rng.normal(0, 0.01, x.size)
    g = design_weights(x)
    assert abs(g.mean() - 1) < 1e-12 and g.min() > 0
    assert g[x > 0.15].mean() > 10 * g[x < 0].mean()          # the wing is up-weighted
    grid = np.array([-0.2, 0.0, 0.3, 0.45])
    plain = NNRegressor(seed=0, first_steps=600, hidden=32, depth=3, lr=3e-3, batch_size=4096)
    dw = NNRegressor(seed=0, first_steps=600, hidden=32, depth=3, lr=3e-3, batch_size=4096,
                     design_weight=True)
    fp = plain.fit_predict(0.0, x, v, grid)
    fd = dw.fit_predict(0.0, x, v, grid)
    truth = np.where(grid < 0.05, 0.2 * np.clip(0.05 - grid, 0, None), 0.15 * (grid - 0.05))
    assert np.abs(fd[2:] - truth[2:]).max() < 0.015                 # wing recovered
    assert np.abs(fd[:2] - truth[:2]).max() < 0.01                  # bulk still right
    assert np.all(np.isfinite(fp))                                  # plain fit still runs
