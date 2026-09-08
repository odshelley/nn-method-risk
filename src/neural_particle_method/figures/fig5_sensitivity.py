"""Fig 5: knob-sensitivity curves, one subplot per estimator, dashed line at the explicit_nn default.

Faceted by particle budget: one row-block per distinct `params.n_particles`, with item 1's
column-wrapping grid logic applied inside each block (so a block with more estimators than
`NCOLS` spills onto extra rows). The dashed `explicit_nn` reference line for a row-block uses
that block's own budget.
"""
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


def make(summary_csv, store, outdir, return_fig=False):
    df = _frame(store, "sensitivity")
    algos = sorted(df["params.algo"].unique()) if df is not None else []
    if df is not None and algos and "params.n_particles" in df:
        budgets = sorted(df["params.n_particles"].astype(int).unique())
    else:
        budgets = [None]
    rows_per_budget = max(1, -(-len(algos) // NCOLS)) if algos else 1
    nrows = rows_per_budget * len(budgets)
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(3.2 * NCOLS, 3 * nrows), sharey=True, squeeze=False)

    if not algos:
        for ax in axes.ravel():
            ax.axis("off")
        axes.ravel()[0].axis("on")
        axes.ravel()[0].set_xlabel("knob value")
        axes.ravel()[0].set_ylabel("pooled IV RMSE (bp)")
        if return_fig:
            return fig
        out = Path(outdir) / "fig5_sensitivity.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out, metadata={"CreationDate": None}); plt.close(fig)
        return str(out)

    for b_i, budget in enumerate(budgets):
        block = axes[b_i * rows_per_budget:(b_i + 1) * rows_per_budget].ravel()
        sub_budget = df if budget is None else df[df["params.n_particles"].astype(int) == budget]
        default = _default_rmse(store, budget)
        for i, algo in enumerate(algos):
            ax = block[i]
            sub = sub_budget[sub_budget["params.algo"] == algo].copy()
            if len(sub):
                sub["knob_value"] = sub["params.knob_value"].astype(float)
                sub["rmse"] = sub["metrics.pooled_rmse_bp"].astype(float)
                g = sub.groupby("knob_value").rmse.mean().sort_index()
                ax.plot(g.index, g.values, marker="o")
            ax.set_xscale("log"); ax.set_xlabel("knob value")
            ax.set_title(f"{algo} (N={budget:,})" if budget is not None else algo)
            if default is not None:
                ax.axhline(default, linestyle="--", color="gray")
            if i % NCOLS == 0:
                ax.set_ylabel("pooled IV RMSE (bp)")
        for ax in block[len(algos):]:
            ax.axis("off")

    if return_fig:
        return fig
    out = Path(outdir) / "fig5_sensitivity.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, metadata={"CreationDate": None}); plt.close(fig)
    return str(out)
