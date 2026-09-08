import numpy as np
import pytest
import torch

from neural_particle_method.bench.scenarios import make_registry
from neural_particle_method.calibrate.config import ExplicitConfig, ImplicitConfig
from neural_particle_method.calibrate.explicit import calibrate_explicit
from neural_particle_method.calibrate.implicit import calibrate_implicit
from neural_particle_method.estimators.nn import NNRegressor
from neural_particle_method.suite.artifacts import (
    GlobalNetModel,
    SliceBank,
    load_run,
    model_kind,
    save_model,
)
from neural_particle_method.tracking.store import Store

E = ExplicitConfig(n_steps=4, n_particles=500, fit_subsample=300, first_steps=5, later_steps=2)
I = ImplicitConfig(n_steps=4, n_particles=500, n_iters=1, fit_steps=5, pool_subsample=300)


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def _explicit():
    sc = make_registry()["s01"]
    est = NNRegressor(seed=0, first_steps=5, later_steps=2, keep_slice_weights=True)
    r = calibrate_explicit(sc.local_vol(), sc.dynamics, est, E, s0=sc.s0, T=sc.T, seed=0)
    return sc, est, r


def test_slice_bank_reproduces_the_regressor_slice_by_slice():
    _sc, est, r = _explicit()
    bank = SliceBank.from_regressor(est)
    assert len(bank.nets) == E.n_steps - 1 and bank.times[0] == r.field[1].t
    x = np.linspace(-0.2, 0.2, 9)
    est.net.load_state_dict(est.slice_weights[0][1])
    np.testing.assert_allclose(bank.f(r.field[1].t, x), est.predict(x), rtol=1e-6)
    assert bank.features(0.3, x).shape == (9, 65) and np.all(bank.features(0.3, x)[:, -1] == 1.0)
    assert bank.readout(0.3).shape == (65,)
    assert model_kind(est) == "explicit_slices"


def test_slice_bank_time_rule_matches_leverage_field():
    sc, est, _r = _explicit()
    bank = SliceBank.from_regressor(est)
    x = np.zeros(3)
    dt = sc.T / E.n_steps
    np.testing.assert_allclose(bank.f(dt * 1.5, x), bank.f(dt, x))          # last time <= t
    # before the first slice: first slice
    np.testing.assert_allclose(bank.f(0.0, x), bank.f(dt, x))


def test_global_net_model_matches_the_network():
    sc = make_registry()["s01"]
    r = calibrate_implicit(sc.local_vol(), sc.dynamics, I, s0=sc.s0, T=sc.T, seed=0)
    gm = GlobalNetModel(r.net, sc.T)
    x = np.linspace(-0.3, 0.3, 5)
    with torch.no_grad():
        tz = torch.tensor(np.stack([np.full(5, 0.5 / sc.T), x / 0.3], axis=1), dtype=torch.float32)
        ref = r.net(tz).numpy()[:, 0]
    np.testing.assert_allclose(gm.f(0.5, x), ref, rtol=1e-6)
    assert gm.features(0.5, x).shape == (5, 65) and gm.readout(0.5).shape == (65,)
    assert model_kind(r.net) == "implicit_net"


def test_save_and_load_round_trip(store, tmp_path):
    _sc, est, r = _explicit()
    with store.run("suite_offline", {"sid": "s01", "body": "explicit"}) as h:
        h.log_json("leverage.json", r.field.to_json())
        kind = save_model(h, est, {"body": "explicit", "n_steps": 4})
        rid = h.run_id
    assert kind == "explicit_slices"
    lr = load_run(store, rid)
    assert lr.meta["body"] == "explicit" and lr.meta["kind"] == "explicit_slices"
    assert isinstance(lr.model, SliceBank) and lr.params["sid"] == "s01"
    x = np.linspace(-0.2, 0.2, 9)
    np.testing.assert_allclose(
        lr.model.f(0.3, x), SliceBank.from_regressor(est).f(0.3, x), rtol=1e-6
    )
    for a, b in zip(lr.field, r.field):
        np.testing.assert_array_equal(a.L, b.L)
        np.testing.assert_array_equal(a.f, b.f)


def test_field_only_models_load_with_model_none(store):
    _sc, _est, r = _explicit()
    from neural_particle_method.estimators import make_estimator
    with store.run("suite_cold", {"sid": "s01", "algo": "nw"}) as h:
        h.log_json("leverage.json", r.field.to_json())
        assert save_model(h, make_estimator("nw"), {"body": "nw"}) == "field_only"
        rid = h.run_id
    lr = load_run(store, rid)
    assert lr.model is None and len(lr.field) == 4
