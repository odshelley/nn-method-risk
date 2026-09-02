"""Fig 2: wing IV error with and without the tilt, from per-run iv_err_bp."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _profile(runs_dir, algo, n_particles=200_000):
    # explicit_nn_is never runs on the f_* stress scenarios, so pooling all runs of
    # an algo would compare explicit_nn (with f_* included) against explicit_nn_is
    # (without) over asymmetric run sets; skip f_* sids and pin one particle budget
    # so both curves are averaged over the same scenarios and the same n.
    errs = []
    for p in Path(runs_dir).rglob("*.json"):
        d = json.loads(p.read_text())
        if d.get("sid", "").startswith("f_"):
            continue
        if d.get("n_particles") != n_particles:
            continue
        if d.get("algo") == algo and d.get("status") == "ok" and d.get("iv_err_bp"):
            e = np.array(d["iv_err_bp"], dtype=float)
            errs.append(np.abs(e[-1]))  # longest maturity row
    return np.nanmean(np.stack(errs), axis=0) if errs else None


def make(summary_csv, runs_dir, outdir, n_particles=200_000):
    k = np.log(np.geomspace(0.6, 1.6, 13))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for algo, label in (("explicit_nn", "no tilt"), ("explicit_nn_is", "defensive mixture")):
        prof = _profile(runs_dir, algo, n_particles)
        if prof is not None:
            ax.plot(k, prof, marker="o", label=label)
    ax.set_xlabel("log-moneyness"); ax.set_ylabel("|IV error| (bp), longest maturity")
    ax.legend()
    out = Path(outdir) / "fig2_wings.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, metadata={"CreationDate": None}); plt.close(fig)
    return str(out)
