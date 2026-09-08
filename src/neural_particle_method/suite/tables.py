"""Section-4 tables of paper/notes_experiments.tex from the suite experiments. Bare tabulars."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FULL, HESTON_SIDS, SSVI_SIDS

COLD_ROWS = (
    ("nw", "NW"), ("explicit_nn", "Explicit NN"), ("implicit_nn", "Implicit NN"), ("rkhs", "RKHS"),
    ("spline", "Spline"), ("nw_ghl", "GHL kernel"), ("bins", "Bins"), ("muguruza", "Muguruza"),
    ("purbf", "PU-RBF"), ("pde", "PDE (attainable floor)"),
)
LAGGED_ROWS = (
    ("stale_L", "Stale leverage, no action"),
    ("nw_resolve", "NW re-solve on $S_1$"),
    ("explicit_stale", "Explicit NN, stale $f$, fresh Dupire"),
    ("implicit_stale", "Implicit NN, stale $f$, fresh Dupire"),
    ("explicit_rkhs", "Explicit NN + RKHS head"),
    ("implicit_rkhs", "Implicit NN + RKHS head"),
    ("explicit_ridge", "Explicit NN + ridge head"),
    ("implicit_ridge", "Implicit NN + ridge head"),
)
LAG_TITLES = {
    "surface": "Surface lag",
    "surface_spot": "Surface plus $2\\%$ sticky-strike spot lag",
}


def _sids(family):
    return set(SSVI_SIDS if family == "ssvi" else HESTON_SIDS)


def _finished(store, experiment):
    df = store.search(experiment)
    return df[df["status"] == "FINISHED"] if len(df) else df


def _num(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _agg(g):
    """Per-scenario means over seeds, then mean/median over scenarios."""
    per_sid = g.groupby("params.sid").agg(
        mae=("mae", "mean"), rmse=("rmse", "mean"), wings=("wings", "mean"),
        t025=("t025", "mean"), lev=("lev", "median"), lat=("lat", "median"),
    )
    return pd.Series({
        "mae_mean": per_sid.mae.mean(), "mae_median": per_sid.mae.median(),
        "rmse_mean": per_sid.rmse.mean(), "rmse_median": per_sid.rmse.median(),
        "wings_mean": per_sid.wings.mean(), "wings_median": per_sid.wings.median(),
        "t025_mean": per_sid.t025.mean(), "lev_median": per_sid.lev.median(),
        "lat_median": per_sid.lat.median(), "n_sids": len(per_sid),
    })


def _prep(df):
    out = pd.DataFrame(index=df.index)
    out["params.sid"] = df["params.sid"]
    out["mae"] = _num(df, "metrics.pooled_mae_bp")
    out["rmse"] = _num(df, "metrics.pooled_rmse_bp")
    out["wings"] = _num(df, "metrics.wings_mae_bp")
    out["t025"] = _num(df, "metrics.mae_bp/T0.25")
    out["lev"] = _num(df, "metrics.lev_rmse")
    if "metrics.online_s" in df.columns:
        out["lat"] = _num(df, "metrics.online_s")
    else:
        out["lat"] = _num(df, "metrics.fit_s")
    return out


EMPTY = pd.Series({k: np.nan for k in (
    "mae_mean", "mae_median", "rmse_mean", "rmse_median", "wings_mean",
    "wings_median", "t025_mean", "lev_median", "lat_median", "n_sids",
)})


def cold_frame(store, settings=FULL, family="ssvi"):
    cold = _finished(store, settings.experiment("suite_cold"))
    pde = _finished(store, settings.experiment("suite_pde_floor"))
    rows = {}
    for algo, label in COLD_ROWS:
        src = pde if algo == "pde" else cold
        if len(src) == 0:
            rows[label] = EMPTY.copy()
            continue
        sel = src[src["params.sid"].isin(_sids(family))]
        if algo != "pde":
            sel = sel[sel["params.algo"] == algo]
        rows[label] = _agg(_prep(sel)) if len(sel) else EMPTY.copy()
    return pd.DataFrame(rows).T


def lagged_frame(store, settings=FULL, family="ssvi"):
    lagged = _finished(store, settings.experiment("suite_lagged"))
    rows = {}
    for lag in ("surface", "surface_spot"):
        for method, label in LAGGED_ROWS:
            if method in ("stale_L", "nw_resolve"):
                sizes = ("--",)
            else:
                sizes = tuple(f"{n // 1000}k" for n in settings.offline_sizes)
            for size in sizes:
                key = (lag, label, size)
                if len(lagged) == 0:
                    rows[key] = EMPTY.copy()
                    continue
                sel = lagged[
                    (lagged["params.sid"].isin(_sids(family)))
                    & (lagged["params.method"] == method)
                    & (lagged["params.lag"] == lag)
                ]
                if size != "--":
                    sel = sel[pd.to_numeric(sel["params.offline_n"]) == int(size[:-1]) * 1000]
                rows[key] = _agg(_prep(sel)) if len(sel) else EMPTY.copy()
    df = pd.DataFrame(rows).T
    df.index = pd.MultiIndex.from_tuples(df.index, names=["lag", "method", "size"])
    return df


def _cell(x, fmt="{:.0f}"):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)


def _cold_tex(df):
    lines = [
        "\\begin{tabular}{l rr rr rr r r}", "\\toprule",
        (" & \\multicolumn{2}{c}{pooled MAE (bp)} & \\multicolumn{2}{c}{pooled RMSE (bp)} & "
         "\\multicolumn{2}{c}{wings MAE (bp)} & lev.\\ RMSE & latency (s)\\\\"),
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "method & mean & median & mean & median & mean & median & median & median\\\\", "\\midrule",
    ]
    for label, r in df.iterrows():
        cells = [
            _cell(r.mae_mean), _cell(r.mae_median), _cell(r.rmse_mean), _cell(r.rmse_median),
            _cell(r.wings_mean), _cell(r.wings_median), _cell(r.lev_median, "{:.3f}"),
            _cell(r.lat_median, "{:.1f}"),
        ]
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _lagged_tex(df):
    lines = [
        "\\begin{tabular}{l l rr rr r}", "\\toprule",
        ("method & offline $N$ & pooled MAE & pooled RMSE & wings MAE & $T=0.25$ MAE & "
         "online latency (s)\\\\"),
        "\\midrule",
    ]
    for lag in ("surface", "surface_spot"):
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{LAG_TITLES[lag]}}}}}\\\\")
        for (lg, label, size), r in df.iterrows():
            if lg != lag:
                continue
            cells = [
                _cell(r.mae_mean), _cell(r.rmse_mean), _cell(r.wings_mean), _cell(r.t025_mean),
                _cell(r.lat_median, "{:.1f}"),
            ]
            lines.append(f"{label} & {size} & " + " & ".join(cells) + "\\\\")
        if lag == "surface":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _appendix_tex(store, settings):
    """Per-scenario pooled MAE (seeds averaged): one block per experiment, scenarios as rows."""
    blocks = []
    cold = _finished(store, settings.experiment("suite_cold"))
    if len(cold):
        p = _prep(cold)
        p["algo"] = cold["params.algo"]
        piv = p.groupby(["params.sid", "algo"]).mae.mean().unstack("algo")
        cols = [a for a, _ in COLD_ROWS if a in piv.columns]
        piv = piv.reindex(columns=cols)
        head = " & ".join(dict(COLD_ROWS)[a] for a in cols)
        lines = [
            "\\begin{tabular}{l " + "r" * len(cols) + "}", "\\toprule", f"sid & {head}\\\\",
            "\\midrule",
        ]
        for sid, r in piv.iterrows():
            row_sid = str(sid).replace("_", "\\_")
            lines.append(f"{row_sid} & " + " & ".join(_cell(v) for v in r.values) + "\\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        blocks.append("% cold suite, pooled MAE bp\n" + "\n".join(lines))
    lagged = _finished(store, settings.experiment("suite_lagged"))
    if len(lagged):
        p = _prep(lagged)
        p["col"] = (
            lagged["params.method"] + "/" + lagged["params.offline_n"].astype(str)
            + "/" + lagged["params.lag"]
        )
        piv = p.groupby(["params.sid", "col"]).mae.mean().unstack("col")
        lines = [
            "\\begin{tabular}{l " + "r" * len(piv.columns) + "}", "\\toprule",
            "sid & " + " & ".join(c.replace("_", "\\_") for c in piv.columns) + "\\\\",
            "\\midrule",
        ]
        for sid, r in piv.iterrows():
            row_sid = str(sid).replace("_", "\\_")
            lines.append(f"{row_sid} & " + " & ".join(_cell(v) for v in r.values) + "\\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        blocks.append("% lagged suite, pooled MAE bp\n" + "\n".join(lines))
    return "\n\n".join(blocks) + "\n" if blocks else "% no finished suite runs\n"


def section4_tables(store, out_dir="paper/tables", settings=FULL):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "suite_cold_ssvi.tex": _cold_tex(cold_frame(store, settings, "ssvi")),
        "suite_cold_heston.tex": _cold_tex(cold_frame(store, settings, "heston")),
        "suite_lagged_ssvi.tex": _lagged_tex(lagged_frame(store, settings, "ssvi")),
        "suite_lagged_heston.tex": _lagged_tex(lagged_frame(store, settings, "heston")),
        "suite_appendix.tex": _appendix_tex(store, settings),
    }
    paths = []
    for name, text in files.items():
        (out / name).write_text(text)
        paths.append(out / name)
    return paths
