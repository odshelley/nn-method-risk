from neural_particle_method.bench.acceptance import CARDS, Card, check, run_acceptance
from neural_particle_method.pricing.reprice import RepriceConfig
from neural_particle_method.tracking.store import Store
from tests.conftest import TINY_EXPLICIT, TINY_N, TINY_REPRICE_N, TINY_REPRICE_STEPS

TINY_REPRICE = RepriceConfig(TINY_REPRICE_N, TINY_REPRICE_STEPS)


def test_cards_table():
    names = [c.name for c in CARDS]
    assert names == ["nw_ghl_li_simple", "nw_ghl_li_complex", "muguruza_li_simple", "rkhs_li_simple",
                     "rkhs_li_complex", "bins_li_simple", "bins_li_complex", "purbf_li_simple"]
    by = {c.name: c for c in CARDS}
    assert by["bins_li_simple"].source_value == 0.91 and by["bins_li_complex"].source_value == 1.01
    assert by["nw_ghl_li_simple"].source_value == 1.44 and by["nw_ghl_li_complex"].source_value == 1.18
    assert by["nw_ghl_li_simple"].knobs == {"fixed_scale": 1.0}
    assert by["rkhs_li_simple"].knobs == {"n_centres": 40, "variance": 5.0, "lam": 1e-7}
    assert by["muguruza_li_simple"].metric == "delta_vs_nw_ghl_bp" and by["muguruza_li_simple"].tolerance == 10.0
    assert all(c.n_particles == 100_000 for c in CARDS if c.algo != "muguruza" and c.algo != "purbf")
    assert by["muguruza_li_simple"].n_particles == 50_000 and by["purbf_li_simple"].n_particles == 2_048


def test_check_runs_and_evaluates(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    tiny = Card("tiny_bins", "li_simple", "bins", TINY_N, 0, {}, source_value=100.0, tolerance=0.3, metric="avg_abs_pct")
    status, achieved, rid = check(store, tiny, explicit=TINY_EXPLICIT, reprice=TINY_REPRICE)
    assert status == "BETTER" and achieved >= 0.0 and store.get_params(rid)["card"] == "tiny_bins"
    assert len(store.search("acceptance")) == 1


def test_run_acceptance_filters_by_name(tmp_path, monkeypatch):
    import neural_particle_method.bench.acceptance as acc
    tiny = Card("tiny_bins", "li_simple", "bins", TINY_N, 0, {}, 100.0, 0.3, "avg_abs_pct")
    monkeypatch.setattr(acc, "CARDS", (tiny,))
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    out = run_acceptance(store, names=["tiny_bins"], explicit=TINY_EXPLICIT, reprice=TINY_REPRICE)
    assert out == [("tiny_bins", "BETTER", out[0][2], out[0][3])]
