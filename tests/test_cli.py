from neural_particle_method.cli import main
from tests.conftest import TINY_N, TINY_REPRICE_N, TINY_REPRICE_STEPS


def test_list(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out.lower()
    assert "scenarios" in out and "implicit_ridge" in out


def test_run_then_skip_then_aggregate(tmp_path, capsys):
    uri = ["--tracking-uri", f"sqlite:///{tmp_path / 'db'}", "--artifact-root", str(tmp_path / "art")]
    args = uri + ["run", "--scenario", "s01", "--algo", "nw", "--n", str(TINY_N), "--seed", "0",
                  "--n-steps", "6", "--reprice-n", str(TINY_REPRICE_N), "--reprice-steps", str(TINY_REPRICE_STEPS)]
    assert main(args) == 0
    assert "run_id" in capsys.readouterr().out
    assert main(args) == 0
    assert "skip" in capsys.readouterr().out.lower()
    assert main(uri + ["aggregate", "--out", str(tmp_path / "s.csv"), "--digest", str(tmp_path / "d.md")]) == 0
    assert (tmp_path / "s.csv").exists()


def test_import_legacy_on_empty_root(tmp_path):
    uri = ["--tracking-uri", f"sqlite:///{tmp_path / 'db'}", "--artifact-root", str(tmp_path / "art")]
    (tmp_path / "results").mkdir()
    assert main(uri + ["import-legacy", "--root", str(tmp_path / "results")]) == 0
