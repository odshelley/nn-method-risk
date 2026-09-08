"""Fig 4: latency-accuracy frontier, one point per (algo, N)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def make(summary_csv, store, outdir):
    df = pd.read_csv(summary_csv)
    ok = df[(df.status == "ok") & (~df.sid.str.startswith("f_"))]
    deploy = ok["intraday_s"].fillna(ok["total_s"]) if "intraday_s" in ok.columns else ok["total_s"]
    ok = ok.assign(deploy_s=deploy)
    g = ok.groupby(["algo", "n_particles"]).agg(
        rmse=("pooled_rmse_bp", "mean"), t=("deploy_s", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for algo, sub in g.groupby("algo"):
        ax.plot(sub.t, sub.rmse, marker="o", label=algo)
    ax.set_xlabel("intraday calibration wall-clock (s)"); ax.set_ylabel("pooled IV RMSE (bp)")
    ax.set_xscale("log"); ax.legend(fontsize=8)
    out = Path(outdir) / "fig4_latency.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, metadata={"CreationDate": None}); plt.close(fig)
    return str(out)
