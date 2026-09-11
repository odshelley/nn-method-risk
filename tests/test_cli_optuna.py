from neural_particle_method.cli import _parser


def test_optuna_run_parses_with_the_default_per_trial():
    args = _parser().parse_args(["optuna", "run", "--study", "x", "--trials", "3"])
    assert (args.cmd, args.optuna_cmd, args.study, args.trials) == ("optuna", "run", "x", 3)
    assert args.per_trial == 8 and args.jobs == 1


def test_optuna_validate_and_promote_parse():
    a = _parser().parse_args(["optuna", "validate", "--study", "s"])
    assert (a.optuna_cmd, a.top, a.jobs, a.budget) == ("validate", 3, 1, 80_000)
    b = _parser().parse_args(["optuna", "promote", "--study", "s", "--trial", "7"])
    assert (b.optuna_cmd, b.study, b.trial) == ("promote", "s", 7)
    c = _parser().parse_args(["optuna", "clouds", "--jobs", "4"])
    assert (c.optuna_cmd, c.jobs, c.sids) == ("clouds", 4, None)
