"""Fig 3: (xi, rho)-plane pooled RMSE heatmaps, kernel vs network."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

XIS, RHOS = (0.3, 0.6, 1.0), (-0.7, -0.3)


def make(summary_csv, runs_dir, outdir):
    df = pd.read_csv(summary_csv)
    ok = df[(df.status == "ok") & (df.sid.str.startswith("f_"))]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    for ax, algo in zip(axes, ("nw", "explicit_nn")):
        grid = [[ok[(ok.algo == algo) & (ok.sid == f"f_xi{x}_rho{r}")].pooled_rmse_bp.mean()
                 for x in XIS] for r in RHOS]
        im = ax.imshow(grid, aspect="auto", origin="lower")
        ax.set_xticks(range(len(XIS)), XIS); ax.set_yticks(range(len(RHOS)), RHOS)
        ax.set_xlabel("xi"); ax.set_title(algo)
    axes[0].set_ylabel("rho")
    fig.colorbar(im, ax=axes, label="pooled IV RMSE (bp)")
    out = Path(outdir) / "fig3_plane.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", metadata={"CreationDate": None}); plt.close(fig)
    return str(out)
