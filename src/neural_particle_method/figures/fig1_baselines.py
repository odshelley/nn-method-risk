"""Fig 1 (baselines): pooled IV RMSE per estimator at 1e4 and 1e5 particles, SSVI scenarios."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _frame(store):
    df = store.search("baselines", "attributes.status = 'FINISHED'")
    if len(df) == 0 or "params.sid" not in df:
        return None
    df = df[~df["params.sid"].str.startswith("f_")].copy()
    df["budget"] = df["params.budget"].astype(int)
    df["rmse"] = df["metrics.pooled_rmse_bp"].astype(float)
    return df


def make(summary_csv, store, outdir):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    df = _frame(store)
    if df is not None and len(df):
        g = df.groupby(["params.algo", "budget"]).rmse.agg(["mean", "std"]).reset_index()
        algos = sorted(g["params.algo"].unique(), key=lambda a: g[(g["params.algo"] == a)]["mean"].min())
        width, x = 0.38, np.arange(len(algos))
        for j, n in enumerate((10_000, 100_000)):
            sub = g[g.budget == n].set_index("params.algo").reindex(algos)
            ax.bar(x + (j - 0.5) * width, sub["mean"], width, yerr=sub["std"].fillna(0.0), capsize=2, label=f"N = {n:,}")
        ax.set_xticks(x, algos); ax.tick_params(axis="x", rotation=30); ax.legend(fontsize=8)
    ax.set_ylabel("pooled IV RMSE (bp)")
    out = Path(outdir) / "fig1_baselines.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, metadata={"CreationDate": None}); plt.close(fig)
    return str(out)
