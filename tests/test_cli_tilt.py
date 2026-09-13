from neural_particle_method.cli import _parser


def test_tilt_subcommands_parse():
    a = _parser().parse_args(["tilt", "slices", "--jobs", "4", "--sids", "s01", "s02"])
    assert (a.cmd, a.tilt_cmd, a.jobs, a.sids, a.smoke) == ("tilt", "slices", 4,
                                                            ["s01", "s02"], False)
    b = _parser().parse_args(["tilt", "cold", "--design", "constant-3", "--smoke"])
    assert (b.tilt_cmd, b.design, b.smoke, b.jobs) == ("cold", "constant-3", True, 1)
    c = _parser().parse_args(["tilt", "online", "--budgets", "10000", "--seeds", "0"])
    assert (c.tilt_cmd, c.budgets, c.seeds, c.design) == ("online", [10_000], [0], None)
    d = _parser().parse_args(["tilt", "tables", "--out", "x"])
    assert (d.tilt_cmd, d.out) == ("tables", "x")
    assert _parser().parse_args(["tilt", "winner"]).tilt_cmd == "winner"
    assert _parser().parse_args(["tilt", "figures", "--design", "none"]).design == "none"
