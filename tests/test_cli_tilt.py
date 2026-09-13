from neural_particle_method.cli import _parser


def test_tilt_subcommands_parse():
    a = _parser().parse_args(["tilt", "slices", "--jobs", "4", "--sids", "s01", "s02"])
    assert (a.cmd, a.tilt_cmd, a.jobs, a.sids, a.smoke) == ("tilt", "slices", 4,
                                                            ["s01", "s02"], False)
    b = _parser().parse_args(["tilt", "cold", "--design", "constant-3", "--smoke"])
    assert (b.tilt_cmd, b.design, b.smoke, b.jobs) == ("cold", "constant-3", True, 1)
    assert b.algos == ["nw", "explicit_nn_opt"]              # both rows by default
    assert _parser().parse_args(["tilt", "cold", "--algos", "nw"]).algos == ["nw"]
    assert not hasattr(_parser().parse_args(["tilt", "slices"]), "algos")
    c = _parser().parse_args(["tilt", "online", "--budgets", "10000", "--seeds", "0"])
    assert (c.tilt_cmd, c.budgets, c.seeds, c.design) == ("online", [10_000], [0], None)
    d = _parser().parse_args(["tilt", "tables", "--out", "x"])
    assert (d.tilt_cmd, d.out) == ("tables", "x")
    assert _parser().parse_args(["tilt", "winner"]).tilt_cmd == "winner"
    assert _parser().parse_args(["tilt", "figures", "--design", "none"]).design == "none"


def test_cold_exits_1_when_no_design_won_the_slice_layer(tmp_path, monkeypatch):
    """A winner() of "none" is the documented null result; the stages must not run on it."""
    import neural_particle_method.suite.tilt as T
    import neural_particle_method.suite.tilt_tables as TT
    from neural_particle_method.cli import main

    def boom(*a, **k):
        raise AssertionError("the stage must not run when no design won")

    monkeypatch.setattr(TT, "slice_frame", lambda *a, **k: None)
    monkeypatch.setattr(TT, "winner", lambda *a, **k: "none")
    monkeypatch.setattr(T, "run_tilt_stage", boom)
    uri, root = f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art")
    base = ["--tracking-uri", uri, "--artifact-root", root, "tilt"]
    assert main(base + ["cold", "--smoke"]) == 1
    assert main(base + ["online", "--smoke"]) == 1
    assert main(base + ["tables", "--out", str(tmp_path / "t")]) == 1
    assert main(base + ["figures", "--out", str(tmp_path / "f")]) == 1
    assert not (tmp_path / "t").exists() and not (tmp_path / "f").exists()
    # an explicit --design still runs, and "winner" still reports the null result
    assert main(base + ["winner"]) == 0
