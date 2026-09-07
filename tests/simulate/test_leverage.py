import json

import numpy as np

from neural_particle_method.simulate.leverage import DEFAULT_GRID, LeverageField, Slice


def _field():
    g = np.linspace(-1.0, 1.0, 5)
    return LeverageField([Slice(i * 0.25, g, np.full(5, 1.0 + i), np.full(5, 0.04)) for i in range(4)])


def test_at_is_piecewise_constant_in_t_and_interpolates_in_x():
    f = _field()
    assert np.allclose(f.at(0.3, np.array([0.0, 0.2])), 2.0)      # slice index 1
    assert np.allclose(f.at(0.0, np.array([0.0])), 1.0)
    assert np.allclose(f.at(5.0, np.array([0.0])), 4.0)             # clamps to last slice
    g = np.array([-1.0, 1.0])
    f2 = LeverageField([Slice(0.0, g, np.array([1.0, 3.0]), np.array([0.04, 0.04]))])
    assert f2.at(0.0, np.array([0.0]))[0] == 2.0


def test_single_point_grid_is_constant():
    f = LeverageField([Slice(0.0, np.array([0.0]), np.array([1.7]), np.array([0.04]))])
    np.testing.assert_array_equal(f.at(0.0, np.array([-2.0, 0.5])), np.array([1.7, 1.7]))


def test_resample_matches_np_interp_and_single_point_rule():
    g = np.array([-1.0, 1.0])
    single = Slice(0.0, np.array([0.0]), np.array([1.7]), np.array([0.05]))
    two = Slice(0.5, g, np.array([1.0, 3.0]), np.array([0.04, 0.06]))
    r = LeverageField([single, two]).resample(DEFAULT_GRID)
    np.testing.assert_array_equal(r[0].L, np.full(len(DEFAULT_GRID), 1.7))
    np.testing.assert_array_equal(r[0].f, np.full(len(DEFAULT_GRID), 0.05))
    np.testing.assert_array_equal(r[1].L, np.interp(DEFAULT_GRID, g, two.L))
    np.testing.assert_array_equal(r[1].f, np.interp(DEFAULT_GRID, g, two.f))
    assert r.times.tolist() == [0.0, 0.5]


def test_records_and_json_round_trip_exact():
    f = _field()
    recs = f.to_records()
    assert isinstance(recs[0], tuple) and len(recs[0]) == 4
    g = LeverageField.from_records(recs)
    h = LeverageField.from_json(json.loads(json.dumps(f.to_json())))
    for a, b, c in zip(f, g, h):
        assert a.t == b.t == c.t
        np.testing.assert_array_equal(a.L, b.L); np.testing.assert_array_equal(a.L, c.L)
        np.testing.assert_array_equal(a.f, c.f); np.testing.assert_array_equal(a.grid, c.grid)


def test_L_matrix_stacks_slices():
    m = _field().L_matrix()
    assert m.shape == (4, 5) and m[2, 0] == 3.0
