"""Figures of the importance-sampling study: clouds, leverage and weights, tilted vs untilted."""
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..bench.scenarios import full_registry, quote_k_grid
from ..calibrate.importance import DESIGNS, UNTILTED
from ..pricing.reprice import snap_times
from .config import FULL
from .tilt import WING_CUT


def _cell(store, settings, sid, n, design, seed):
    key = {"sid": sid, "n_particles": int(n), "design": design, "seed": int(seed),
           "n_steps": int(settings.explicit.n_steps)}
    rid = store.find_finished(settings.experiment("tilt_slices"), key)
    if rid is None:
        raise RuntimeError(f"no finished tilt_slices cell for {key}")
    return rid


def _load(store, rid, times):
    with tempfile.TemporaryDirectory() as d:
        doc = json.loads(store.download(rid, "slice_scores.json", d).read_text())
        clouds = {}
        for t in times:
            z = np.load(store.download(rid, f"cloud_T{t:g}.npz", d))
            clouds[t] = (z["lnx"], z["v"], z["w"])
    return doc, clouds


def _row_index(doc, t):
    return int(np.argmin(np.abs(np.array(doc["maturities"]) - t)))


def _panels(times, arms):
    fig, axes = plt.subplots(len(times), len(arms), figsize=(5.2 * len(arms), 3.4 * len(times)),
                             squeeze=False, sharex="row")
    return fig, axes


def _cloud_figure(sid, docs, clouds, times, arms, out):
    k = quote_k_grid()
    fig, axes = _panels(times, arms)
    for i, t in enumerate(times):
        for j, arm in enumerate(arms):
            ax = axes[i, j]
            doc, (lnx, v, w) = docs[arm], clouds[arm][t]
            r = _row_index(doc, t)
            ax.scatter(lnx, v, s=6 * w, alpha=0.15, color="0.5", linewidths=0)
            kk = np.array(doc["k"])
            ax.plot(kk, doc["f_ref"][r], "k-", lw=1.5, label="PDE $E[V|X]$")
            ax.plot(kk, doc["f_hat"]["nw"][r], "C0--", lw=1.2, label="NW")
            ax.plot(kk, doc["f_hat"]["net"][r], "C3:", lw=1.4, label="net")
            for x in kk[np.abs(k) > WING_CUT]:
                ax.axvline(x, color="C1", lw=0.6, alpha=0.6)
            ax.set_ylim(0, np.percentile(v, 99.5) * 1.1)
            ax.set_title(f"{sid}, t = {t:g}, {arm}")
            ax.set_xlabel("log-spot"); ax.set_ylabel("V")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def _leverage_figure(sid, docs, clouds, times, arms, out):
    k = quote_k_grid()
    fig, axes = _panels(times, arms)
    for i, t in enumerate(times):
        for j, arm in enumerate(arms):
            ax = axes[i, j]
            doc, (lnx, _, w) = docs[arm], clouds[arm][t]
            r = _row_index(doc, t)
            kk = np.array(doc["k"])
            ax.plot(kk, doc["L_ref"][r], "k-", lw=1.5, label="PDE $L$")
            ax.plot(kk, doc["L_hat"]["nw"][r], "C0--", lw=1.2, label="NW")
            ax.plot(kk, doc["L_hat"]["net"][r], "C3:", lw=1.4, label="net")
            for x in kk[np.abs(k) > WING_CUT]:
                ax.axvline(x, color="C1", lw=0.6, alpha=0.6)
            ax2 = ax.twinx()
            ax2.hist(lnx, bins=60, weights=w, density=True, color="0.7", alpha=0.35)
            ax2.set_yticks([])
            ax.set_title(f"{sid}, t = {t:g}, {arm}")
            ax.set_xlabel("log-spot"); ax.set_ylabel("leverage")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def _ess_figure(sid, store, settings, n, seed, out):
    """Panel A: the ESS fraction of the whole cloud against time, the full per-step path where
    the cell stored one (`ess_path`, one entry per step) and otherwise the four quoted
    maturities. Panel B: the local ESS at the outermost strikes."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    k = quote_k_grid()
    outer = [0, len(k) - 1]
    n_steps = int(settings.explicit.n_steps)
    T = full_registry()[sid].T
    for name in [UNTILTED] + [d.name for d in DESIGNS]:
        key = {"sid": sid, "n_particles": int(n), "design": name, "seed": int(seed),
               "n_steps": n_steps}
        rid = store.find_finished(settings.experiment("tilt_slices"), key)
        if rid is None:
            continue
        with tempfile.TemporaryDirectory() as d:
            doc = json.loads(store.download(rid, "slice_scores.json", d).read_text())
        path = doc.get("ess_path") or []
        if path:
            ts = (np.arange(len(path)) + 1) * T / n_steps      # ESS after each step
            a.plot(ts, path, lw=1.0, label=name)
        else:
            a.plot(doc["maturities"], doc["ess_slice"], marker="o", label=name)
        loc = np.array(doc["ess_local"])[:, outer].mean(axis=1)
        b.plot(doc["maturities"], loc, marker="o", label=name)
    a.set_xlabel("t"); a.set_ylabel("ESS fraction (slice)"); a.set_ylim(0, 1.05)
    b.set_xlabel("t"); b.set_ylabel("local ESS at the outermost strikes")
    a.legend(fontsize=7); a.set_title(sid)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def make_figures(store, design, out_dir="figures/out", settings=FULL, sids=("s02", "s11"),
                 n_particles=30_000, seed=0, times=(0.25, 1.0)):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arms = {"untilted": UNTILTED, "tilted": design}
    paths = []
    for sid in sids:
        T = full_registry()[sid].T
        snapped = snap_times(list(times), settings.explicit.n_steps, T=T)
        docs, clouds = {}, {}
        for arm, name in arms.items():
            rid = _cell(store, settings, sid, n_particles, name, seed)
            docs[arm], clouds[arm] = _load(store, rid, snapped)
        p = out / f"tilt_cloud_{sid}.png"
        _cloud_figure(sid, docs, clouds, snapped, list(arms), p); paths.append(p)
        p = out / f"tilt_leverage_{sid}.png"
        _leverage_figure(sid, docs, clouds, snapped, list(arms), p); paths.append(p)
        p = out / f"tilt_ess_{sid}.png"
        _ess_figure(sid, store, settings, n_particles, seed, p); paths.append(p)
    return paths
