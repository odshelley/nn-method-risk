import pytest

from neural_particle_method.suite.config import FULL
from neural_particle_method.suite.tables import (
    COLD_ROWS,
    LAGGED_ROWS,
    cold_frame,
    lagged_frame,
    section4_tables,
)
from neural_particle_method.tracking.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    common = {"n_particles": 100_000}
    for sid, mae, rmse in (("s01", 10.0, 12.0), ("s02", 30.0, 40.0)):
        for seed in (0, 1):
            params = {"sid": sid, "algo": "nw", "seed": seed, **common}
            with s.run("suite_cold", params) as h:
                h.log_metrics({
                    "pooled_mae_bp": mae + seed, "pooled_rmse_bp": rmse, "wings_mae_bp": mae * 2,
                    "lev_rmse": 0.05, "fit_s": 1.0, "mae_bp/T0.25": mae,
                })
    # an unbalanced design: s02 has a third seed, so a pooled mean would differ from the
    # two-stage (per-scenario then across-scenario) mean the tables report.
    with s.run("suite_cold", {"sid": "s02", "algo": "nw", "seed": 2, **common}) as h:
        h.log_metrics({
            "pooled_mae_bp": 60.0, "pooled_rmse_bp": 40.0, "wings_mae_bp": 60.0,
            "lev_rmse": 0.05, "fit_s": 1.0, "mae_bp/T0.25": 30.0,
        })
    with s.run("suite_pde_floor", {"sid": "s01", "seed": 0, "n_steps": 200}) as h:
        h.log_metrics({
            "pooled_mae_bp": 3.0, "pooled_rmse_bp": 4.0, "wings_mae_bp": 5.0, "fit_s": 0.0,
            "lev_rmse": 0.0,
        })
    lagged_params = {
        "sid": "s01", "method": "explicit_rkhs", "offline_n": 200_000, "lag": "surface",
        "seed": 0, "n_steps": 200,
    }
    with s.run("suite_lagged", lagged_params) as h:
        h.log_metrics({
            "pooled_mae_bp": 20.0, "pooled_rmse_bp": 25.0, "wings_mae_bp": 30.0,
            "mae_bp/T0.25": 35.0, "online_s": 2.5,
        })
    heston_cold_params = {"sid": "li_simple", "algo": "nw", "seed": 0, **common}
    with s.run("suite_cold", heston_cold_params) as h:
        h.log_metrics({
            "pooled_mae_bp": 15.0, "pooled_rmse_bp": 18.0, "wings_mae_bp": 22.0,
            "lev_rmse": 0.05, "fit_s": 1.0, "mae_bp/T0.25": 15.0,
        })
    heston_lagged_params = {
        "sid": "li_simple", "method": "explicit_rkhs", "offline_n": 200_000, "lag": "surface",
        "seed": 0, "n_steps": 200,
    }
    with s.run("suite_lagged", heston_lagged_params) as h:
        h.log_metrics({
            "pooled_mae_bp": 22.0, "pooled_rmse_bp": 27.0, "wings_mae_bp": 33.0,
            "mae_bp/T0.25": 37.0, "online_s": 2.7,
        })
    return s


def test_cold_frame_aggregates_mean_and_median_over_scenarios_and_seeds(store):
    df = cold_frame(store, FULL, family="ssvi")
    nw = df.loc["NW"]
    # per-sid first: s01 = 10.5, s02 = (30 + 31 + 60) / 3; a pooled mean would be 28.4
    two_stage = (10.5 + (30 + 31 + 60) / 3) / 2
    assert nw["mae_mean"] == pytest.approx(two_stage)
    assert nw["mae_median"] == pytest.approx(two_stage)
    assert nw["mae_mean"] != pytest.approx((10 + 11 + 30 + 31 + 60) / 5)
    assert nw["lat_median"] == 1.0 and nw["lev_median"] == 0.05
    assert df.loc["PDE (attainable floor)"]["mae_mean"] == 3.0
    assert df.loc["Explicit NN"].isna().all()
    assert list(df.index) == [r[1] for r in COLD_ROWS]


def test_lagged_frame_and_files(store, tmp_path):
    df = lagged_frame(store, FULL, family="ssvi")
    row = df.loc[("surface", "Explicit NN + RKHS head", "200k")]
    assert row["mae_mean"] == 20.0 and row["t025_mean"] == 35.0 and row["lat_median"] == 2.5
    assert ("surface_spot", "NW re-solve on $S_1$", "--") in df.index
    assert len(LAGGED_ROWS) == 10
    paths = section4_tables(store, tmp_path / "tables", FULL)
    names = sorted(p.name for p in paths)
    assert names == [
        "suite_appendix.tex", "suite_cold_heston.tex", "suite_cold_ssvi.tex",
        "suite_lagged_heston.tex", "suite_lagged_ssvi.tex",
    ]
    cold = (tmp_path / "tables" / "suite_cold_ssvi.tex").read_text()
    assert "\\begin{tabular}" in cold and "NW & 25 & 25 &" in cold
    assert "Explicit NN & -- & --" in cold
    assert "PDE (attainable floor) & 3 & 3 &" in cold


def test_heston_lagged_table_drops_the_t025_column(store, tmp_path):
    section4_tables(store, tmp_path / "tables", FULL)
    ssvi = (tmp_path / "tables" / "suite_lagged_ssvi.tex").read_text()
    heston = (tmp_path / "tables" / "suite_lagged_heston.tex").read_text()
    assert "$T=0.25$ MAE" in ssvi and "\\multicolumn{7}" in ssvi
    assert "$T=0.25$ MAE" not in heston and "\\multicolumn{6}" in heston
    assert "Explicit NN + RKHS head & 200k & 22 & 27 & 33 & 2.7" in heston


def test_appendix_has_one_resized_block_per_experiment_family_and_lag(store, tmp_path):
    section4_tables(store, tmp_path / "tables", FULL)
    appendix = (tmp_path / "tables" / "suite_appendix.tex").read_text()
    for title in ("Cold suite, SSVI scenarios, pooled MAE (bp)",
                  "Cold suite, Heston scenarios, pooled MAE (bp)",
                  "Lagged suite, surface lag, SSVI scenarios, pooled MAE (bp)",
                  "Lagged suite, surface lag, Heston scenarios, pooled MAE (bp)"):
        assert f"\\paragraph{{{title}}}" in appendix
    assert appendix.count("\\resizebox{\\textwidth}{!}{%") == 4
    # method/size only: the lag is the block, not a column
    assert "sid & explicit\\_rkhs/200000" in appendix


def test_appendix_escapes_underscore_scenario_ids(store, tmp_path):
    section4_tables(store, tmp_path / "tables", FULL)
    appendix = (tmp_path / "tables" / "suite_appendix.tex").read_text()
    assert "li\\_simple &" in appendix
    assert "li_simple &" not in appendix
