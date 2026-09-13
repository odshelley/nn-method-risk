import numpy as np
import pytest

from neural_particle_method.suite.tilt_tables import (
    cold_frame,
    online_frame,
    slice_frame,
    tilt_tables,
    winner,
)
from neural_particle_method.tracking.store import Store

K = np.log(np.geomspace(0.6, 1.6, 13)).tolist()
MATS = [0.25, 0.5, 1.0, 2.0]


def _doc(scale, bias, seed):
    # seeds 0, 1, 2 give bias - scale, bias, bias + scale at every strike: mean = bias and
    # std (ddof=1) = scale exactly, so the expected frame values are deterministic
    err = np.full((4, 13), bias + scale * (seed - 1)).tolist()
    return {"maturities": MATS, "k": K, "f_ref": np.ones((4, 13)).tolist(),
            "L_ref": np.ones((4, 13)).tolist(),
            "f_hat": {"nw": (1 + np.array(err)).tolist(), "net": (1 + np.array(err)).tolist()},
            "L_hat": {"nw": np.ones((4, 13)).tolist(), "net": np.ones((4, 13)).tolist()},
            "f_rel_err": {"nw": err, "net": err}, "lev_rel_err": {"nw": err, "net": err},
            "ess_local": np.ones((4, 13)).tolist(), "ess_slice": [1.0] * 4}


def _log_slices(s, spec, sids=("s01", "s02"), ns=(10_000, 30_000), seeds=3):
    for sid in sids:
        for n in ns:
            for design, (scale, bias) in spec.items():
                for seed in range(seeds):
                    params = {"sid": sid, "n_particles": n, "design": design, "seed": seed,
                              "n_steps": 200}
                    with s.run("tilt_slices", params) as h:
                        h.log_json("slice_scores.json", _doc(scale, bias, seed))
                        h.log_metrics({"ess_final": 1.0})


@pytest.fixture
def store(tmp_path):
    s = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    # layer 1: untilted noisy, constant-3 half the noise, inverse_sqrt-9 quarter noise but biased
    _log_slices(s, {"none": (0.10, 0.0), "constant-3": (0.05, 0.0),
                    "inverse_sqrt-9": (0.025, 0.4)})
    # layer 2 cold: nw untilted vs tilted at two budgets, one seed
    for n, (mae_u, mae_t) in {10_000: (57.0, 50.0), 80_000: (44.0, 43.0)}.items():
        for design, mae in (("none", mae_u), ("constant-3", mae_t)):
            params = {"sid": "s01", "algo": "nw", "n_particles": n, "seed": 0, "design": design}
            with s.run("tilt_cold", params) as h:
                m = {"pooled_mae_bp": mae, "wings_mae_bp": mae + 20, "mae_bp/T0.25": mae + 40,
                     "liquid_mae_bp": mae - 30, "lev_rmse": 0.1, "fit_s": 3.0}
                if design != "none":
                    m["ess_min_slice"] = 0.6
                h.log_metrics(m)
    with s.run("suite_pde_floor", {"sid": "s01", "seed": 0, "n_steps": 200}) as h:
        h.log_metrics({"pooled_mae_bp": 42.0, "wings_mae_bp": 62.0, "mae_bp/T0.25": 100.0,
                       "liquid_mae_bp": 13.0, "lev_rmse": 0.0, "fit_s": 0.0})
    # layer 2 online: untilted in suite_budget_tuned, tilted in tilt_online
    for budget, (mae_u, mae_t) in {10_000: (46.0, 44.0), 80_000: (42.4, 42.0)}.items():
        base = {"sid": "s01", "method": "explicit_opt_spline", "budget": budget,
                "lag": "surface", "seed": 0, "n_steps": 200, "recipe_hash": "abc"}
        with s.run("suite_budget_tuned", base) as h:
            h.log_metrics({"pooled_mae_bp": mae_u, "wings_mae_bp": mae_u + 20,
                           "liquid_mae_bp": 16.0, "online_s": 7.0})
        with s.run("tilt_online", {**base, "design": "constant-3"}) as h:
            h.log_metrics({"pooled_mae_bp": mae_t, "wings_mae_bp": mae_t + 20,
                           "liquid_mae_bp": 15.0, "online_s": 7.2, "ess_min_slice": 0.5})
    return s


def test_slice_frame_splits_std_and_bias(store):
    df = slice_frame(store)
    u = df.loc[("nw", 30_000, "none")]
    c = df.loc[("nw", 30_000, "constant-3")]
    b = df.loc[("nw", 30_000, "inverse_sqrt-9")]
    np.testing.assert_allclose([u["std_pooled"], c["std_pooled"], b["std_pooled"]],
                               [10.0, 5.0, 2.5], rtol=1e-9)              # percent of f
    np.testing.assert_allclose([u["bias_pooled"], b["bias_pooled"]], [0.0, 40.0], atol=1e-9)
    np.testing.assert_allclose(u["std_T0.25"], u["std_pooled"], rtol=1e-9)
    assert set(df.index.get_level_values(0)) == {"nw", "net"}
    # 2 scenarios x 4 maturities x 7 wing strikes, three seeds in every one of them
    assert u["n_cells"] == c["n_cells"] == b["n_cells"] == 56.0
    assert u["n_seeds_min"] == 3.0


def test_winner_prefers_lowest_std_but_disqualifies_bias(store):
    df = slice_frame(store)
    assert winner(df) == "constant-3"
    # with the biased design's bias removed it would win on variance alone
    fixed = df.copy()
    fixed.loc[("nw", 30_000, "inverse_sqrt-9"), "bias_pooled"] = 0.0
    assert winner(fixed) == "inverse_sqrt-9"
    # nothing beats untilted -> "none"
    worse = df.copy()
    worse.loc[("nw", 30_000, "constant-3"), "std_pooled"] = 1e3
    worse.loc[("nw", 30_000, "inverse_sqrt-9"), "std_pooled"] = 1e3
    assert winner(worse) == "none"


def test_winner_allows_bias_in_proportion_to_the_std_removed(store):
    """The guard is on the bias the tilt *adds*, scaled to the standard deviation it buys:
    1 point free plus half of what it removes. Untilted here is std 10, bias 0."""
    df = slice_frame(store)
    b, c = ("nw", 30_000, "inverse_sqrt-9"), ("nw", 30_000, "constant-3")
    base = df.copy()
    base.loc[c, "std_pooled"] = 1e3               # take the other design out of the running
    base.loc[b, "std_pooled"] = 7.0               # removes 3 points of std
    ok = base.copy()
    ok.loc[b, "bias_pooled"] = 2.0                # 2 <= 1 + 0.5 x 3
    assert winner(ok) == "inverse_sqrt-9"
    bad = base.copy()
    bad.loc[b, "bias_pooled"] = 3.0               # 3 > 2.5
    assert winner(bad) == "none"
    # the guard is on the difference, so a bias shared with the untilted arm costs nothing
    shared = ok.copy()
    shared.loc[("nw", 30_000, "none"), "bias_pooled"] = 20.0
    shared.loc[b, "bias_pooled"] = 21.5
    assert winner(shared) == "inverse_sqrt-9"


def test_winner_skips_a_design_that_is_missing_cells(tmp_path, capsys):
    """An unbalanced design averages over a different set of cells, so its std is not comparable
    even when it is the smallest."""
    s = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    _log_slices(s, {"none": (0.10, 0.0), "constant-3": (0.05, 0.0)}, ns=(30_000,))
    _log_slices(s, {"inverse_sqrt-9": (0.01, 0.0)}, sids=("s01",), ns=(30_000,))
    df = slice_frame(s)
    assert df.loc[("nw", 30_000, "inverse_sqrt-9"), "std_pooled"] == pytest.approx(1.0)
    assert df.loc[("nw", 30_000, "inverse_sqrt-9"), "n_cells"] == 28.0
    assert df.loc[("nw", 30_000, "none"), "n_cells"] == 56.0
    assert winner(df) == "constant-3"
    assert "skipping inverse_sqrt-9" in capsys.readouterr().out


def test_winner_skips_a_design_with_a_single_seed_cell(store):
    df = slice_frame(store)
    thin = df.copy()
    thin.loc[("nw", 30_000, "inverse_sqrt-9"), "bias_pooled"] = 0.0
    thin.loc[("nw", 30_000, "inverse_sqrt-9"), "n_seeds_min"] = 1.0
    assert winner(thin) == "constant-3"


def test_cold_and_online_frames_pair_tilted_with_untilted(store, monkeypatch):
    import neural_particle_method.suite.tilt_tables as T
    monkeypatch.setattr(T, "_promoted_hash", lambda: "abc")
    c = cold_frame(store, "constant-3")
    assert c.loc[("NW", 10_000, "tilted"), "mae"] == 50.0
    assert c.loc[("NW", 10_000, "untilted"), "mae"] == 57.0
    assert np.isnan(c.loc[("NW", 10_000, "untilted"), "ess_min"])
    assert c.loc[("NW", 10_000, "tilted"), "ess_min"] == 0.6
    assert c.loc[("PDE (attainable floor)", 0, ""), "mae"] == 42.0
    o = online_frame(store, "constant-3")
    assert o.loc[(10_000, "surface", "tilted"), "mae"] == 44.0
    assert o.loc[(10_000, "surface", "untilted"), "mae"] == 46.0
    assert np.isnan(o.loc[(80_000, "surface_spot", "tilted"), "mae"])


def test_tables_are_written_with_bold_best(store, tmp_path, monkeypatch):
    import neural_particle_method.suite.tilt_tables as T
    monkeypatch.setattr(T, "_promoted_hash", lambda: "abc")
    paths = tilt_tables(store, "constant-3", out_dir=tmp_path)
    assert [p.name for p in paths] == ["tilt_slices.tex", "tilt_cold.tex", "tilt_online.tex"]
    cold = (tmp_path / "tilt_cold.tex").read_text()
    # bold is the column minimum over every non-floor row: 43 at 80k tilted, not 50 at 10k
    assert "NW, 10k, tilted & 50 &" in cold
    assert "NW, 80k, tilted & \\textbf{43} &" in cold
    assert "PDE (attainable floor) & 42 &" in cold
    assert "& 0.60 &" in cold and "& -- &" in cold          # ess_min: value / untilted
    sl = (tmp_path / "tilt_slices.tex").read_text()
    assert "constant-3" in sl and "NW, 30k &" in sl and "\\textbf{2.5}" in sl
    # the bias columns are signed, so they are never bolded: 0.0 and 40.0 are plain
    assert "\\textbf{0.0}" not in sl and "\\textbf{40.0}" not in sl
    assert "& 40.0 &" in sl
    on = (tmp_path / "tilt_online.tex").read_text()
    assert "10k, surface, tilted & 44 &" in on
    assert "80k, surface, tilted & \\textbf{42} &" in on
    assert "80k, surface, untilted & \\textbf{42} &" in on   # ties at displayed precision
