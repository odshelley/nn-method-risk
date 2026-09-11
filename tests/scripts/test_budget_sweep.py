from neural_particle_method.suite.budget import BUDGETS, budget_jobs, head_for
from neural_particle_method.suite.config import SSVI_SIDS
from neural_particle_method.suite.heads import FeatureRidgeHead, RKHSHead, SplineHead
from tests.scripts.helpers import load_script

bs = load_script("budget_sweep")


def test_tuned_job_list_is_one_cell_per_method_budget_lag_and_seed():
    jobs = budget_jobs(("s01",), "explicit_tuned")
    assert len(jobs) == 1 * 2 * 2 * 3 * 5 == 60
    assert {j[1] for j in jobs} == {"nw_resolve", "explicit_tuned_stale", "explicit_tuned_rkhs",
                                    "explicit_tuned_spline", "explicit_tuned_ridge"}
    assert {j[2] for j in jobs} == set(BUDGETS)
    assert {j[0] for j in jobs} == {"s01"}


def test_legacy_body_keeps_its_own_method_names_and_experiment():
    jobs = budget_jobs(("s01",), "explicit")
    assert {j[1] for j in jobs} == {"nw_resolve", "explicit_stale", "explicit_rkhs",
                                    "explicit_spline", "explicit_ridge"}
    assert bs.EXPERIMENTS == {"explicit_tuned": "suite_budget_tuned", "explicit": "suite_budget",
                              "explicit_opt": "suite_budget_tuned"}


def test_head_keys_off_the_method_suffix():
    assert isinstance(head_for("explicit_tuned_spline"), SplineHead)
    assert isinstance(head_for("explicit_tuned_rkhs"), RKHSHead)
    assert isinstance(head_for("explicit_tuned_ridge"), FeatureRidgeHead)
    assert head_for("nw_resolve") is None
    assert head_for("explicit_tuned_stale") is None


def test_defaults_are_the_tuned_body_over_every_ssvi_scenario():
    assert bs.DEFAULT_BODY == "explicit_tuned"
    assert bs.SIDS == SSVI_SIDS
    assert len(budget_jobs()) == len(SSVI_SIDS) * 2 * 2 * 3 * 5


def test_cli_takes_a_body_and_a_recipe():
    a = bs.parse_args(["--body", "explicit_opt", "--recipe", "explicit_opt", "--sids", "s01"])
    assert a.body == "explicit_opt" and a.recipe == "explicit_opt" and a.sids == ["s01"]
    assert bs.parse_args([]).recipe == "explicit_opt"
