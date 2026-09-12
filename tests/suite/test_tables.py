import re

import numpy as np
import pytest

from neural_particle_method.estimators import recipes as R
from neural_particle_method.suite.config import FULL
from neural_particle_method.suite.tables import (
    BODY_ROWS,
    COLD_ROWS,
    HEAD_ROWS,
    _render_rows,
    bodies_frame,
    cold_frame,
    heads_frame,
    section4_tables,
)
from neural_particle_method.tracking.store import Store

COLD = {  # (sid, seed): pooled, liquid, wings, price
    ("s01", 0): (10.0, 4.0, 20.0, 2.0),
    ("s01", 1): (11.0, 6.0, 22.0, 2.2),
    ("s02", 0): (30.0, 13.0, 61.0, 3.1),
}
PDE_FLOOR = (3.0, 1.5, 5.0, 1.0)
PDE_SOLVE_S = 150.0
BODIES = {  # (body, n_particles): pooled, liquid, wings, price, total_s
    ("explicit_tuned", 200_000): (55.0, 20.0, 80.0, 5.0, 3000.0),
    ("explicit_tuned", 500_000): (50.0, 18.0, 72.0, 4.5, 9000.0),
    ("explicit", 500_000): (64.0, 25.0, 93.0, 6.0, 346.0),
}
LAGGED = {  # (method, seed): pooled, liquid, online_s
    ("explicit_tuned_spline", 0): (48.0, 15.0, 0.9),
    ("explicit_tuned_spline", 1): (50.0, 17.0, 1.1),
    ("nw_resolve", 0): (49.0, 16.0, 1.0),
    ("nw_resolve", 1): (49.0, 16.0, 1.0),
}
BUDGET = {"explicit_tuned_spline": (51.0, 0.4), "nw_resolve": (57.0, 0.5)}
FAST = {"hidden": 16, "depth": 2, "lr": 1e-2, "batch_size": 0, "first_steps": 5,
        "later_steps": 2, "weight_decay": 0.0, "fit_subsample": 400, "warm_start": True,
        "mean_match": False, "monotone": False, "monotone_penalty": 0.0, "tail": "flat",
        "hetero": False}
PROMOTED_MAE, REJECTED_MAE = 33.0, 99.0


@pytest.fixture
def store(tmp_path):
    s = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    for (sid, seed), (mae, liquid, wings, price) in COLD.items():
        params = {"sid": sid, "algo": "nw", "seed": seed, "n_particles": 100_000}
        with s.run("suite_cold", params) as h:
            h.log_metrics({"pooled_mae_bp": mae, "liquid_mae_bp": liquid, "wings_mae_bp": wings,
                           "price_bp/pooled": price, "fit_s": 1.0})
    mae, liquid, wings, price = PDE_FLOOR
    with s.run("suite_pde_floor", {"sid": "s01", "seed": 0, "n_steps": 200}) as h:
        h.log_metrics({"pooled_mae_bp": mae, "liquid_mae_bp": liquid, "wings_mae_bp": wings,
                       "price_bp/pooled": price, "fit_s": 0.0})
    with s.run("pde_reference", {"sid": "s01", "n_steps": 200, "lag": "none"}) as h:
        h.log_metrics({"runtime_s": PDE_SOLVE_S, "mass": 1.0, "forward": 1.0})
    for (body, n), (mae, liquid, wings, price, total_s) in BODIES.items():
        params = {"sid": "s01", "body": body, "n_particles": n, "seed": 0, "n_steps": 200}
        with s.run("suite_offline", params) as h:
            h.log_metrics({"pooled_mae_bp": mae, "liquid_mae_bp": liquid, "wings_mae_bp": wings,
                           "price_bp/pooled": price, "total_s": total_s, "fit_s": total_s})
    for (method, seed), (mae, liquid, online_s) in LAGGED.items():
        offline_n = 0 if method == "nw_resolve" else 500_000
        params = {"sid": "s01", "method": method, "offline_n": offline_n, "lag": "surface",
                  "seed": seed, "n_steps": 200}
        with s.run("suite_lagged", params) as h:
            h.log_metrics({"pooled_mae_bp": mae, "liquid_mae_bp": liquid, "online_s": online_s})
    for method, (mae, online_s) in BUDGET.items():
        params = {"sid": "s01", "method": method, "budget": 10_000, "lag": "surface", "seed": 0,
                  "n_steps": 200}
        with s.run("suite_budget_tuned", params) as h:
            h.log_metrics({"pooled_mae_bp": mae, "online_s": online_s})
    return s


def _log_searched_runs(store, promoted_hash, rejected_hash):
    """One searched body and one searched budget cell per recipe, as `validate --top 2` leaves."""
    for h, mae in ((promoted_hash, PROMOTED_MAE), (rejected_hash, REJECTED_MAE)):
        cell = {"sid": "s01", "method": "explicit_opt_spline", "budget": 10_000,
                "lag": "surface", "seed": 0, "n_steps": 200, "recipe_hash": h}
        with store.run("suite_budget_tuned", cell) as handle:
            handle.log_metrics({"pooled_mae_bp": mae, "online_s": 0.3})
        body = {"sid": "s01", "body": "explicit_opt", "n_particles": 500_000, "seed": 0,
                "n_steps": 200, "recipe_hash": h}
        with store.run("suite_offline", body) as handle:
            handle.log_metrics({"pooled_mae_bp": mae, "liquid_mae_bp": mae / 2,
                                "wings_mae_bp": mae * 2, "price_bp/pooled": 1.0,
                                "total_s": 20.0, "fit_s": 20.0})
        cold = {"sid": "s01", "algo": "explicit_nn_opt", "seed": 0, "n_particles": 100_000,
                "recipe_hash": h}
        with store.run("suite_cold", cold) as handle:
            handle.log_metrics({"pooled_mae_bp": mae, "liquid_mae_bp": mae / 2,
                                "wings_mae_bp": mae * 2, "price_bp/pooled": 1.0, "fit_s": 2.0})


def test_searched_rows_quote_only_the_promoted_recipe(store, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RECIPE_DIR", tmp_path)
    R.save_recipe("explicit_opt", FAST)
    _log_searched_runs(store, R.recipe_hash(R.coerce_recipe(FAST)),
                       R.recipe_hash(R.coerce_recipe({**FAST, "hidden": 64})))
    heads = heads_frame(store, FULL)
    assert heads.loc["Searched body + spline head", "mae_surface_10k"] == pytest.approx(
        PROMOTED_MAE)
    bodies = bodies_frame(store, FULL)
    assert bodies.loc[("Explicit NN, searched", "500k"), "mae_mean"] == pytest.approx(
        PROMOTED_MAE)
    assert bodies.loc[("Explicit NN, searched", "200k")].isna().all()
    cold = cold_frame(store, FULL)
    assert cold.loc["Explicit NN, searched", "mae_mean"] == pytest.approx(PROMOTED_MAE)


def test_searched_rows_are_blank_until_a_recipe_is_promoted(store, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RECIPE_DIR", tmp_path / "empty")
    _log_searched_runs(store, "aaaaaaaaaa", "bbbbbbbbbb")
    assert np.isnan(heads_frame(store, FULL).loc["Searched body + spline head",
                                                 "mae_surface_10k"])
    assert bodies_frame(store, FULL).loc[("Explicit NN, searched", "500k")].isna().all()
    assert cold_frame(store, FULL).loc["Explicit NN, searched"].isna().all()


def _two_stage(index):
    """Per-scenario mean over seeds, then mean over scenarios, of one COLD column."""
    s01 = (COLD[("s01", 0)][index] + COLD[("s01", 1)][index]) / 2
    return (s01 + COLD[("s02", 0)][index]) / 2


def _cells(text, label):
    """The cells of the row with this label, `\\\\` stripped."""
    line = next(ln for ln in text.splitlines() if ln.startswith(label + " &"))
    return [c.strip() for c in line.removesuffix("\\\\").split("&")]


def test_cold_frame_aggregates_mean_and_median_over_scenarios_and_seeds(store):
    df = cold_frame(store, FULL)
    nw = df.loc["NW"]
    assert nw["mae_mean"] == pytest.approx(_two_stage(0))
    assert nw["mae_mean"] != pytest.approx(sum(v[0] for v in COLD.values()) / len(COLD))
    assert nw["mae_median"] == pytest.approx(_two_stage(0))
    assert nw["liquid_mean"] == pytest.approx(_two_stage(1))
    assert nw["wings_mean"] == pytest.approx(_two_stage(2))
    assert nw["p_pooled_mean"] == pytest.approx(_two_stage(3))
    assert nw["lat_median"] == 1.0
    pde = df.loc["PDE (attainable floor)"]
    assert pde["mae_mean"] == PDE_FLOOR[0] and pde["liquid_mean"] == PDE_FLOOR[1]
    assert pde["lat_median"] == pytest.approx(PDE_SOLVE_S)
    assert df.loc["Explicit NN, short training"].isna().all()
    assert list(df.index) == [r[1] for r in COLD_ROWS]
    assert ("explicit_nn_opt", "Explicit NN, searched") in COLD_ROWS


def test_bodies_frame_rows_per_body_and_size_with_the_pde_floor(store):
    df = bodies_frame(store, FULL)
    assert ("Explicit NN, tuned", "200k") in df.index
    assert ("Explicit NN, tuned", "500k") in df.index
    assert ("Explicit NN, short training", "500k") in df.index
    tuned = df.loc[("Explicit NN, tuned", "500k")]
    mae, liquid, wings, price, total_s = BODIES[("explicit_tuned", 500_000)]
    assert tuned["mae_mean"] == mae and tuned["liquid_mean"] == liquid
    assert tuned["wings_mean"] == wings and tuned["p_pooled_mean"] == price
    assert tuned["train_s"] == pytest.approx(total_s)
    short = df.loc[("Explicit NN, short training", "500k")]
    assert short["mae_mean"] == BODIES[("explicit", 500_000)][0]
    for size in ("200k", "500k"):
        assert df.loc[("Implicit NN", size)].isna().all()
    floor = df.loc[("PDE (attainable floor)", "--")]
    assert floor["mae_mean"] == PDE_FLOOR[0]
    assert floor["train_s"] == pytest.approx(PDE_SOLVE_S)
    assert len(BODY_ROWS) == 4
    assert ("explicit_opt", "Explicit NN, searched") in BODY_ROWS


def test_heads_frame_joins_the_budget_sweep_to_the_lagged_suite(store):
    df = heads_frame(store, FULL)
    spline = df.loc["Tuned body + spline head"]
    seed_mean = (LAGGED[("explicit_tuned_spline", 0)][0]
                 + LAGGED[("explicit_tuned_spline", 1)][0]) / 2
    assert spline["mae_surface_100k"] == pytest.approx(seed_mean)
    assert spline["mae_surface_10k"] == pytest.approx(BUDGET["explicit_tuned_spline"][0])
    assert np.isnan(spline["mae_surface_spot_10k"])
    assert np.isnan(spline["mae_surface_30k"]) and np.isnan(spline["mae_surface_80k"])
    liquid_mean = (LAGGED[("explicit_tuned_spline", 0)][1]
                   + LAGGED[("explicit_tuned_spline", 1)][1]) / 2
    assert spline["liquid_surface_100k"] == pytest.approx(liquid_mean)
    assert spline["s_10k"] == pytest.approx(BUDGET["explicit_tuned_spline"][1])
    assert spline["s_100k"] == pytest.approx(1.0)
    nw = df.loc["NW re-solve on $S_1$"]
    assert nw["mae_surface_10k"] == pytest.approx(BUDGET["nw_resolve"][0])
    assert list(df.index) == [r[1] for r in HEAD_ROWS]
    assert ("explicit_opt_spline", "Searched body + spline head") in HEAD_ROWS


def test_section4_tables_writes_three_files(store, tmp_path):
    paths = section4_tables(store, tmp_path / "tables", FULL)
    assert sorted(p.name for p in paths) == [
        "suite_bodies.tex", "suite_cold.tex", "suite_heads.tex"]


def _plain(text):
    """Strip the bold wrappers so the number layout can be compared on its own."""
    return re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)


def test_render_rows_bolds_the_best_and_skips_the_floor_and_zero_seconds():
    f = "{:.1f}"
    rows = [("PDE (attainable floor)", [(1.0, f), (5.0, f)]),
            ("stale", [(3.0, f), (0.0, f)]),
            ("head", [(2.0, f), (2.0, f)]),
            ("nw", [(2.04, f), (4.0, f)])]
    lines = _render_rows(rows, secs_cols={1})
    assert lines[0] == "PDE (attainable floor) & 1.0 & 5.0\\\\"
    assert lines[1] == "stale & 3.0 & 0.0\\\\"
    assert lines[2] == "head & \\textbf{2.0} & \\textbf{2.0}\\\\"
    assert lines[3] == "nw & \\textbf{2.0} & 4.0\\\\"   # a tie at the displayed precision


def test_cold_tex_columns_and_formats(store, tmp_path):
    section4_tables(store, tmp_path / "tables", FULL)
    raw = (tmp_path / "tables" / "suite_cold.tex").read_text()
    cold = _plain(raw)
    nw_line = next(ln for ln in raw.splitlines() if ln.startswith("NW &"))
    assert nw_line.count("\\textbf{") == 6      # the only method row with numbers
    pde_line = next(ln for ln in raw.splitlines() if ln.startswith("PDE"))
    assert "\\textbf" not in pde_line
    expected = (f"NW & {_two_stage(0):.0f} & {_two_stage(0):.0f} & {_two_stage(1):.0f} & "
                f"{_two_stage(2):.0f} & {_two_stage(3):.1f} & 1.0")
    assert f"{expected}\\\\" in cold
    assert "& mean & median & liquid & wings & pooled & median\\\\" in cold
    assert "\\begin{tabular}{l rrrrr r}" in cold
    assert "Explicit NN, short training & -- & -- & -- & -- & -- & --\\\\" in cold
    assert "Explicit NN, searched & -- & -- & -- & -- & -- & --\\\\" in cold
    assert f"PDE (attainable floor) & 3 & 3 & 2 & 5 & 1.0 & {PDE_SOLVE_S:.1f}\\\\" in cold


def test_bodies_tex_has_a_size_column_and_the_floor_row(store, tmp_path):
    section4_tables(store, tmp_path / "tables", FULL)
    raw = (tmp_path / "tables" / "suite_bodies.tex").read_text()
    bodies = _plain(raw)
    assert "\\textbf{" in raw
    assert "\\textbf" not in next(ln for ln in raw.splitlines() if ln.startswith("PDE"))
    assert "body & particles & pooled & liquid & wings & price, bp & s\\\\" in bodies
    assert "Explicit NN, tuned & 500k & 50 & 18 & 72 & 4.5 & 9000.0\\\\" in bodies
    assert "Explicit NN, short training & 200k & -- & -- & -- & -- & --\\\\" in bodies
    assert f"PDE (attainable floor) & -- & 3 & 2 & 5 & 1.0 & {PDE_SOLVE_S:.1f}\\\\" in bodies


def test_heads_tex_renders_missing_cells_as_dashes(store, tmp_path):
    section4_tables(store, tmp_path / "tables", FULL)
    heads = _plain((tmp_path / "tables" / "suite_heads.tex").read_text())
    assert "\\begin{tabular}{l rrrrr rrrrr rr}" in heads
    assert ("& 10k & 30k & 80k & 100k & liquid & 10k & 30k & 80k & 100k & liquid & "
            "10k & 100k\\\\") in heads
    assert "\\multicolumn{2}{c}{online s}" in heads
    for _, label in HEAD_ROWS:
        # cells 6..10 are the surface-plus-spot block: no such run in the fixture
        assert _cells(heads, label)[6:11] == ["--"] * 5
    assert ("Tuned body + spline head & 51 & -- & -- & 49 & 16 & -- & -- & -- & -- & -- & "
            "0.4 & 1.0\\\\") in heads
