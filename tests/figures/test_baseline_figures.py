from pathlib import Path

from neural_particle_method.figures import ALL, fig1_baselines, fig3_baselines, fig5_sensitivity
from neural_particle_method.tracking.store import Store


def _store(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    for sid in ("s01", "s02", "f_xi0.3_rho-0.7", "f_xi0.6_rho-0.3"):
        for algo in ("nw_ghl", "bins", "explicit_nn"):
            for n in (10_000, 100_000):
                with store.run("baselines", {"sid": sid, "algo": algo, "n_particles": n, "seed": 0, "budget": n}) as h:
                    h.log_metrics({"pooled_rmse_bp": 10.0 + n / 1e5, "budget": n})
    for algo, name, vals in (("bins", "n_bins", (5, 20, 100)), ("explicit_nn", "hidden", (16, 64))):
        for v in vals:
            with store.run("sensitivity", {"sid": "s01", "algo": algo, "n_particles": 10_000, "seed": 0,
                                           "knob_name": name, "knob_value": v}) as h:
                h.log_metrics({"pooled_rmse_bp": 5.0 + v / 100, "knob_value": v})
    return store


def test_modules_registered():
    assert fig1_baselines in ALL and fig3_baselines in ALL and fig5_sensitivity in ALL


def test_each_baseline_figure_writes_pdf(tmp_path):
    store = _store(tmp_path)
    for mod in (fig1_baselines, fig3_baselines, fig5_sensitivity):
        out = mod.make("unused.csv", store, str(tmp_path / "out"))
        assert Path(out).exists() and out.endswith(".pdf") and b"/CreationDate" not in Path(out).read_bytes()


def test_figures_survive_empty_store(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    for mod in (fig1_baselines, fig3_baselines, fig5_sensitivity):
        assert Path(mod.make("unused.csv", store, str(tmp_path / "out"))).exists()
