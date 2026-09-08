"""Fig 1: pooled IV RMSE per algorithm at fixed budget (bars, mean over scenarios/seeds)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def make(summary_csv, store, outdir):
    df = pd.read_csv(summary_csv)
    ok = df[(df.status == "ok") & (~df.sid.str.startswith("f_"))]
    g = ok.groupby("algo").pooled_rmse_bp.agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(g.index, g["mean"], yerr=g["std"].fillna(0.0), capsize=3)
    ax.set_ylabel("pooled IV RMSE (bp)")
    ax.tick_params(axis="x", rotation=20)
    out = Path(outdir) / "fig1_accuracy.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, metadata={"CreationDate": None}); plt.close(fig)
    return str(out)
