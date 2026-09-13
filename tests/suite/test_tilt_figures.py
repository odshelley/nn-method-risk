import pytest

from neural_particle_method.bench.reference_runs import run_reference
from neural_particle_method.estimators import recipes as R
from neural_particle_method.suite.config import SuiteSettings
from neural_particle_method.suite.tilt import run_slice_cell
from neural_particle_method.suite.tilt_figures import make_figures
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()
FAST = {"hidden": 16, "depth": 2, "lr": 1e-2, "batch_size": 0, "first_steps": 5,
        "later_steps": 2, "weight_decay": 0.0, "fit_subsample": 400, "warm_start": True,
        "mean_match": True, "monotone": False, "monotone_penalty": 0.0, "tail": "linear",
        "hetero": False}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RECIPE_DIR", tmp_path / "recipes")
    R.save_recipe("explicit_opt", FAST)
    s = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    run_reference(s, "s01", n_steps=TINY.explicit.n_steps, n_x=TINY.n_x, n_v=TINY.n_v)
    for design in ("none", "constant-3", "inverse_sqrt-1"):
        run_slice_cell(s, "s01", 800, design, 0, TINY)
    return s


def test_figures_are_written(store, tmp_path):
    paths = make_figures(store, "constant-3", out_dir=tmp_path, settings=TINY, sids=("s01",),
                         n_particles=800, seed=0, times=(0.5, 1.0))
    names = sorted(p.name for p in paths)
    assert names == ["tilt_cloud_s01.png", "tilt_ess_s01.png", "tilt_leverage_s01.png"]
    assert all(p.stat().st_size > 1_000 for p in paths)


def test_missing_cells_raise_a_clear_error(store, tmp_path):
    with pytest.raises(RuntimeError, match="tilt_slices"):
        make_figures(store, "constant-9", out_dir=tmp_path, settings=TINY, sids=("s01",),
                     n_particles=800, seed=0, times=(0.5, 1.0))
