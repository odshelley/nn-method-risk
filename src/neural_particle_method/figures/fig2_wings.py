"""Fig 2: wing IV error with and without the tilt, from per-run iv_err_bp artifacts."""
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def profile(store, algo, n_particles=200_000):
    """Mean |IV error| on the longest maturity over non-stress scenarios at one particle budget."""
    df = store.search("bench", f"params.algo = '{algo}' and params.n_particles = '{n_particles}' "
                               f"and attributes.status = 'FINISHED'")
    errs = []
    with tempfile.TemporaryDirectory() as d:
        for _, r in df.iterrows():
            if str(r.get("params.sid", "")).startswith("f_"):
                continue
            p = store.download(r["run_id"], "iv_err_bp.json", Path(d) / r["run_id"])
            e = np.array(json.loads(p.read_text()), dtype=float)
            errs.append(np.abs(e[-1]))
    return np.nanmean(np.stack(errs), axis=0) if errs else None


def make(summary_csv, store, outdir, n_particles=200_000):
    k = np.log(np.geomspace(0.6, 1.6, 13))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for algo, label in (("explicit_nn", "no tilt"), ("explicit_nn_is", "defensive mixture")):
        prof = profile(store, algo, n_particles)
        if prof is not None:
            ax.plot(k, prof, marker="o", label=label)
    ax.set_xlabel("log-moneyness"); ax.set_ylabel("|IV error| (bp), longest maturity")
    ax.legend()
    out = Path(outdir) / "fig2_wings.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, metadata={"CreationDate": None}); plt.close(fig)
    return str(out)
