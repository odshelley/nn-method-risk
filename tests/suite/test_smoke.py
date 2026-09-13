import pytest

from neural_particle_method.cli import main
from neural_particle_method.estimators import recipes as R
from neural_particle_method.suite.config import SMOKE
from neural_particle_method.tracking.store import Store


@pytest.mark.slow
def test_smoke_all_stages_end_to_end(tmp_path, monkeypatch):
    # the counts below are the pre-promotion suite; a promoted recipe would add a cold row and
    # two online methods, so pin the recipe directory empty rather than track the store's state
    monkeypatch.setattr(R, "RECIPE_DIR", tmp_path / "recipes")
    uri, root = f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art")
    assert main(["--tracking-uri", uri, "--artifact-root", root,
                "suite", "run", "--stage", "all", "--smoke"]) == 0
    store = Store(uri, root)
    assert len(store.search(SMOKE.experiment("suite_cold"))) == 2 * 10
    assert len(store.search(SMOKE.experiment("suite_pde_floor"))) == 2
    assert len(store.search(SMOKE.experiment("suite_offline"))) == 2 * 3
    assert len(store.search(SMOKE.experiment("suite_lagged"))) == 2 * 14 * 2
    assert (store.search(SMOKE.experiment("suite_lagged"))["status"] == "FINISHED").all()


@pytest.mark.slow
def test_smoke_tilt_end_to_end(tmp_path, monkeypatch):
    from neural_particle_method.estimators.recipes import save_recipe
    monkeypatch.setattr(R, "RECIPE_DIR", tmp_path / "recipes")
    save_recipe("explicit_opt", {"hidden": 16, "depth": 2, "lr": 1e-2, "batch_size": 0,
                                 "first_steps": 20, "later_steps": 5, "weight_decay": 0.0,
                                 "fit_subsample": 1_000, "warm_start": True, "mean_match": True,
                                 "monotone": False, "monotone_penalty": 0.0, "tail": "linear",
                                 "hetero": False})
    uri, root = f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art")
    base = ["--tracking-uri", uri, "--artifact-root", root, "tilt"]
    assert main(base + ["slices", "--smoke"]) == 0
    assert main(base + ["winner", "--smoke"]) == 0
    assert main(base + ["cold", "--smoke", "--design", "constant-3"]) == 0
    assert main(base + ["online", "--smoke", "--design", "constant-3"]) == 0
    assert main(base + ["tables", "--smoke", "--design", "constant-3",
                        "--out", str(tmp_path / "tables")]) == 0
    assert main(base + ["figures", "--smoke", "--design", "constant-3",
                        "--out", str(tmp_path / "figs")]) == 0
    store = Store(uri, root)
    assert len(store.search(SMOKE.experiment("tilt_slices"))) == 1 * 1 * 2 * 1
    assert len(store.search(SMOKE.experiment("tilt_cold"))) == 2 * 1 * 1 * 2
    assert len(store.search(SMOKE.experiment("tilt_online"))) == 1 * 2 * 1 * 1
    assert (tmp_path / "tables" / "tilt_cold.tex").exists()
    assert (tmp_path / "figs" / "tilt_cloud_s01.png").exists()
