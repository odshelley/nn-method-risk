import json
from pathlib import Path
import pandas as pd
import pytest
import figures.fig1_accuracy as f1
import figures.fig2_wings as f2
import figures.fig3_plane as f3
import figures.fig4_latency as f4

def _summary(tmp_path):
    rows = []
    for sid in ("s01", "s02", "f_xi0.3_rho-0.7", "f_xi0.6_rho-0.7"):
        for algo in ("nw", "explicit_nn", "ridge", "explicit_nn_is", "implicit_nn"):
            rows.append(dict(sid=sid, algo=algo, n_particles=50_000, seed=0, status="ok",
                             git_hash="x", pooled_rmse_bp=10.0, pooled_max_bp=20.0,
                             wings_rmse_bp=15.0, wings_max_bp=25.0, n_failed=0,
                             total_s=5.0, fit_s=1.0))
    p = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p

def _runs(tmp_path):
    for sid in ("s01", "f_xi0.3_rho-0.7"):
        for algo in ("nw", "explicit_nn", "explicit_nn_is"):
            d = tmp_path / "runs" / sid / algo
            d.mkdir(parents=True, exist_ok=True)
            doc = {"schema": 1, "sid": sid, "algo": algo, "n_particles": 50_000, "seed": 0,
                   "status": "ok", "iv_err_bp": [[float(i - 6) for i in range(13)]] * 4,
                   "diagnostics": {"is_diag": {"max_w": 1.8, "ess_frac": 0.6}} if "is" in algo else {},
                   "scenario": {"maturities": [0.25, 0.5, 1.0, 2.0]}}
            (d / "n50000_s0.json").write_text(json.dumps(doc))
    return tmp_path / "runs"

@pytest.mark.parametrize("mod", [f1, f2, f3, f4])
def test_each_figure_writes_pdf(tmp_path, mod):
    out = mod.make(str(_summary(tmp_path)), str(_runs(tmp_path)), str(tmp_path / "out"))
    assert Path(out).exists() and out.endswith(".pdf")


def test_fig1_is_byte_identical_across_runs(tmp_path):
    summary = _summary(tmp_path)
    runs = _runs(tmp_path)
    out_a = f1.make(str(summary), str(runs), str(tmp_path / "out_a"))
    out_b = f1.make(str(summary), str(runs), str(tmp_path / "out_b"))
    bytes_a, bytes_b = Path(out_a).read_bytes(), Path(out_b).read_bytes()
    assert bytes_a == bytes_b
    # The byte-equality check above can pass by luck if both calls land in the
    # same wall-clock second (PDF /CreationDate has 1s resolution). Assert the
    # actual root cause is fixed: no embedded timestamp at all.
    assert b"/CreationDate" not in bytes_a
