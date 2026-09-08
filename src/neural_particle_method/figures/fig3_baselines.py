"""Fig 3 (baselines): (xi, rho)-plane pooled RMSE heatmaps per estimator, budget = 1e5."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

XIS, RHOS = (0.3, 0.6, 1.0), (-0.7, -0.3)
NCOLS = 4


def _frame(store):
    df = store.search("baselines", "attributes.status = 'FINISHED'")
    if len(df) == 0 or "params.sid" not in df:
        return None
    df = df[df["params.sid"].str.startswith("f_")].copy()
    df["budget"] = df["params.budget"].astype(int)
    df = df[df.budget == 100_000]
    if len(df) == 0:
        return None
    df["rmse"] = df["metrics.pooled_rmse_bp"].astype(float)
    return df


def make(summary_csv, store, outdir, return_fig=False):
    df = _frame(store)
    algos = sorted(df["params.algo"].unique()) if df is not None else []
    nrows = max(1, -(-len(algos) // NCOLS)) if algos else 1
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(3.2 * NCOLS, 3 * nrows), sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(algos):]:
        ax.axis("off")
    if algos:
        vmin, vmax, im = df.rmse.min(), df.rmse.max(), None
        for i, algo in enumerate(algos):
            ax = axes[i]
            grid = [[df[(df["params.algo"] == algo) & (df["params.sid"] == f"f_xi{x}_rho{r}")].rmse.mean()
                     for x in XIS] for r in RHOS]
            im = ax.imshow(grid, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(XIS)), XIS); ax.set_yticks(range(len(RHOS)), RHOS)
            ax.set_xlabel("xi"); ax.set_title(algo)
            if i % NCOLS == 0:
                ax.set_ylabel("rho")
        fig.colorbar(im, ax=list(axes[:len(algos)]), label="pooled IV RMSE (bp)")
    else:
        axes[0].set_xlabel("xi")
        axes[0].set_ylabel("rho")
    if return_fig:
        return fig
    out = Path(outdir) / "fig3_baselines.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", metadata={"CreationDate": None}); plt.close(fig)
    return str(out)
