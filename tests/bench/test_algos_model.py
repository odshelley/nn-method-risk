from dataclasses import replace

from neural_particle_method.bench.algos import CalibResult, run_algo
from neural_particle_method.bench.scenarios import make_registry
from neural_particle_method.calibrate.implicit import GlobalNet
from neural_particle_method.estimators.nn import NNRegressor
from tests.conftest import TINY_EXPLICIT, TINY_IMPLICIT, TINY_N


def test_calibresult_model_defaults_to_none():
    assert CalibResult(None, {}, {}).model is None


def test_explicit_nn_and_implicit_expose_their_models():
    sc = make_registry()["s01"]
    knobs = {"keep_slice_weights": True}
    ex = run_algo("explicit_nn", sc, TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT, knobs=knobs)
    assert isinstance(ex.model, NNRegressor)
    assert len(ex.model.slice_weights) == TINY_EXPLICIT.n_steps - 1
    im = run_algo("implicit_nn", sc, TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT)
    assert isinstance(im.model, GlobalNet)
    nw = run_algo("nw", sc, TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT)
    assert nw.model is not None  # the NW estimator instance


def test_explicit_nn_tuned_runs_with_the_tiny_step_count():
    sc = make_registry()["s01"]
    e = replace(TINY_EXPLICIT, first_steps=5, later_steps=2)
    res = run_algo("explicit_nn_tuned", sc, TINY_N, 0, e, TINY_IMPLICIT,
                   knobs={"batch_size": 512})
    assert len(res.field) == TINY_EXPLICIT.n_steps
