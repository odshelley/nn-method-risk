"""Fig 5: knob-sensitivity curves, one subplot per estimator, dashed line at the explicit_nn default."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NCOLS = 4


def _frame(store, experiment):
    df = store.search(experiment, "attributes.status = 'FINISHED'")
    if len(df) == 0 or "params.algo" not in df:
        return None
    return df


def _default_rmse(store, budget):
    base = _frame(store, "baselines")
    if base is None or "params.budget" not in base or budget is None:
        return None
    sub = base[(base["params.algo"] == "explicit_nn") & (base["params.budget"].astype(int) == budget)]
    return sub["metrics.pooled_rmse_bp"].astype(float).mean() if len(sub) else None


def make(summary_csv, store, outdir):
    df = _frame(store, "sensitivity")
    algos = sorted(df["params.algo"].unique()) if df is not None else []
    fig, axes = plt.subplots(1, NCOLS, figsize=(3.2 * NCOLS, 3), sharey=True)
    for ax in axes[len(algos):]:
        ax.axis("off")
    for ax, algo in zip(axes, algos):
        sub = df[df["params.algo"] == algo].copy()
        sub["knob_value"] = sub["params.knob_value"].astype(float)
        sub["rmse"] = sub["metrics.pooled_rmse_bp"].astype(float)
        g = sub.groupby("knob_value").rmse.mean().sort_index()
        ax.plot(g.index, g.values, marker="o")
        ax.set_xscale("log"); ax.set_xlabel("knob value"); ax.set_title(algo)
        budget = int(sub["params.n_particles"].iloc[0]) if "params.n_particles" in sub else None
        default = _default_rmse(store, budget)
        if default is not None:
            ax.axhline(default, linestyle="--", color="gray")
    if not algos:
        axes[0].set_xlabel("knob value")
    axes[0].set_ylabel("pooled IV RMSE (bp)")
    out = Path(outdir) / "fig5_sensitivity.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, metadata={"CreationDate": None}); plt.close(fig)
    return str(out)
