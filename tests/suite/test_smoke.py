import pytest

from neural_particle_method.cli import main
from neural_particle_method.suite.config import SMOKE
from neural_particle_method.tracking.store import Store


@pytest.mark.slow
def test_smoke_all_stages_end_to_end(tmp_path):
    uri, root = f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art")
    assert main(["--tracking-uri", uri, "--artifact-root", root,
                "suite", "run", "--stage", "all", "--smoke"]) == 0
    store = Store(uri, root)
    assert len(store.search(SMOKE.experiment("suite_cold"))) == 2 * 9
    assert len(store.search(SMOKE.experiment("suite_pde_floor"))) == 2
    assert len(store.search(SMOKE.experiment("suite_offline"))) == 2 * 2
    assert len(store.search(SMOKE.experiment("suite_lagged"))) == 2 * 10 * 2
    assert (store.search(SMOKE.experiment("suite_lagged"))["status"] == "FINISHED").all()
