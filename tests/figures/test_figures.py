from pathlib import Path

import pandas as pd
import pytest

from neural_particle_method.figures import ALL, fig1_accuracy as f1, fig2_wings as f2
from neural_particle_method.tracking.store import Store


def _summary(tmp_path):
    rows = []
    for sid in ("s01", "s02", "f_xi0.3_rho-0.7", "f_xi0.6_rho-0.7"):
        for algo in ("nw", "explicit_nn", "ridge", "explicit_nn_is", "implicit_nn"):
            rows.append(dict(sid=sid, algo=algo, n_particles=50_000, seed=0, status="ok", git_hash="x",
                             pooled_rmse_bp=10.0, pooled_max_bp=20.0, wings_rmse_bp=15.0, wings_max_bp=25.0,
                             n_failed=0, total_s=5.0, fit_s=1.0))
    p = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _store(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    for sid in ("s01", "f_xi0.3_rho-0.7"):
        for algo in ("nw", "explicit_nn", "explicit_nn_is"):
            with store.run("bench", {"sid": sid, "algo": algo, "n_particles": 200_000, "seed": 0}) as h:
                h.log_json("iv_err_bp.json", [[float(i - 6) for i in range(13)]] * 4)
    return store


@pytest.mark.parametrize("mod", ALL)
def test_each_figure_writes_pdf(tmp_path, mod):
    out = mod.make(str(_summary(tmp_path)), _store(tmp_path), str(tmp_path / "out"))
    assert Path(out).exists() and out.endswith(".pdf")


def test_fig2_profile_excludes_stress_scenarios(tmp_path):
    prof = f2.profile(_store(tmp_path), "explicit_nn")
    assert prof.shape == (13,) and prof[0] == 6.0


def test_fig1_is_byte_identical_across_runs(tmp_path):
    summary, store = _summary(tmp_path), _store(tmp_path)
    a = Path(f1.make(str(summary), store, str(tmp_path / "a"))).read_bytes()
    b = Path(f1.make(str(summary), store, str(tmp_path / "b"))).read_bytes()
    assert a == b and b"/CreationDate" not in a
