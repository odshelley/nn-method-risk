"""Tables of the importance-sampling study (paper/tables/tilt_*.tex) and the design winner."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..calibrate.importance import DESIGNS, UNTILTED
from ..estimators.recipes import load_recipe, recipe_exists, recipe_hash
from .config import FULL, SSVI_SIDS
from .tables import FLOOR_LABEL, _cell, _finished, _num, _render_rows
from .tilt import COLD_PARTICLES, ONLINE_BUDGETS, ONLINE_METHOD, SLICE_SIDS, WING_CUT

LAGS = ("surface", "surface_spot")
BIAS_TOLERANCE = 1.5     # a design may carry at most this multiple of the untilted wing bias...
BIAS_FLOOR = 1.0         # ...plus one percentage point, so a near-zero untilted bias is not a wall
WINNER_N = 30_000
ALGO_LABELS = {"nw": "NW", "explicit_nn_opt": "Explicit NN, searched"}
EST_LABELS = {"nw": "NW", "net": "Net"}


def _promoted_hash():
    return recipe_hash(load_recipe("explicit_opt")) if recipe_exists("explicit_opt") else None


def _label(n):
    return f"{int(n) // 1000}k"


def _slice_docs(store, settings):
    """One row per (sid, n, design, seed, estimator, maturity, strike) of wing relative error."""
    df = _finished(store, settings.experiment("tilt_slices"))
    rows = []
    if len(df) == 0:
        return pd.DataFrame(rows, columns=["sid", "n", "design", "seed", "est", "T", "k", "err"])
    df = df[df["params.sid"].isin(SLICE_SIDS)]
    with tempfile.TemporaryDirectory() as d:
        for _, r in df.iterrows():
            doc = json.loads(store.download(r["run_id"], "slice_scores.json",
                                            Path(d) / r["run_id"]).read_text())
            k = np.array(doc["k"])                 # log-strike; s0 = 1 on the SSVI scenarios
            wings = np.abs(k) > WING_CUT
            for est in ("nw", "net"):
                err = np.array(doc["f_rel_err"][est])
                for i, t in enumerate(doc["maturities"]):
                    for j in np.flatnonzero(wings):
                        rows.append((r["params.sid"], int(r["params.n_particles"]),
                                     r["params.design"], int(r["params.seed"]), est, float(t),
                                     float(k[j]), float(err[i, j])))
    return pd.DataFrame(rows, columns=["sid", "n", "design", "seed", "est", "T", "k", "err"])


def slice_frame(store, settings=FULL):
    """Wing std and bias (percent of f) per (estimator, n, design): std and mean over seeds at
    each (sid, T, k), then averaged over strikes, scenarios and (for pooled) maturities."""
    long = _slice_docs(store, settings)
    if len(long) == 0:
        return pd.DataFrame(columns=["std_T0.25", "std_pooled", "bias_pooled"])
    g = long.groupby(["est", "n", "design", "sid", "T", "k"])["err"]
    per = pd.DataFrame({"std": g.std(ddof=1), "bias": g.mean()}).reset_index()
    out = {}
    for (est, n, design), d in per.groupby(["est", "n", "design"]):
        first = d[np.isclose(d["T"], d["T"].min())]
        out[(est, int(n), design)] = pd.Series({
            "std_T0.25": 100 * first["std"].mean(),
            "std_pooled": 100 * d["std"].mean(),
            "bias_pooled": 100 * d["bias"].mean(),
        })
    df = pd.DataFrame(out).T
    df.index = pd.MultiIndex.from_tuples(df.index, names=["est", "n", "design"])
    return df.sort_index()


def winner(frame, n=WINNER_N, est="nw"):
    """The design with the smallest NW wing std at `n`, unless its wing bias exceeds
    BIAS_TOLERANCE x the untilted bias plus BIAS_FLOOR points; "none" when no admissible design
    beats untilted (a NaN std, e.g. a single seed, never beats anything)."""
    if (est, n, UNTILTED) not in frame.index:
        return UNTILTED
    base = frame.loc[(est, n, UNTILTED)]
    best, best_std = UNTILTED, base["std_pooled"]
    for d in DESIGNS:
        if (est, n, d.name) not in frame.index:
            continue
        row = frame.loc[(est, n, d.name)]
        if abs(row["bias_pooled"]) > BIAS_TOLERANCE * abs(base["bias_pooled"]) + BIAS_FLOOR:
            continue
        if row["std_pooled"] < best_std:
            best, best_std = d.name, row["std_pooled"]
    return best


def _cold_metrics(sel):
    if len(sel) == 0:
        return pd.Series({c: np.nan for c in ("mae", "wings", "t025", "liquid", "lev",
                                              "ess_min", "secs")})
    per = pd.DataFrame({
        "sid": sel["params.sid"], "mae": _num(sel, "metrics.pooled_mae_bp"),
        "wings": _num(sel, "metrics.wings_mae_bp"), "t025": _num(sel, "metrics.mae_bp/T0.25"),
        "liquid": _num(sel, "metrics.liquid_mae_bp"), "lev": _num(sel, "metrics.lev_rmse"),
        "ess_min": _num(sel, "metrics.ess_min_slice"), "secs": _num(sel, "metrics.fit_s"),
    }).groupby("sid").mean(numeric_only=True)
    return pd.Series({"mae": per.mae.mean(), "wings": per.wings.mean(), "t025": per.t025.mean(),
                      "liquid": per.liquid.mean(), "lev": per.lev.median(),
                      "ess_min": per.ess_min.mean(), "secs": per.secs.median()})


def cold_frame(store, design, settings=FULL):
    df = _finished(store, settings.experiment("tilt_cold"))
    rows = {}
    for algo, label in ALGO_LABELS.items():
        for n in COLD_PARTICLES:
            for arm, dname in (("untilted", UNTILTED), ("tilted", design)):
                sel = df.iloc[:0] if len(df) == 0 else df[
                    (df["params.algo"] == algo) & (_num(df, "params.n_particles") == n)
                    & (df["params.design"] == dname) & df["params.sid"].isin(SSVI_SIDS)]
                if algo == "explicit_nn_opt" and len(sel):
                    sel = (sel[sel["params.recipe_hash"] == _promoted_hash()]
                           if "params.recipe_hash" in sel.columns else sel.iloc[:0])
                rows[(label, int(n), arm)] = _cold_metrics(sel)
    floor = _finished(store, settings.experiment("suite_pde_floor"))
    floor = floor[floor["params.sid"].isin(SSVI_SIDS)] if len(floor) else floor
    rows[(FLOOR_LABEL, 0, "")] = _cold_metrics(floor)
    out = pd.DataFrame(rows).T
    out.index = pd.MultiIndex.from_tuples(out.index, names=["method", "n", "arm"])
    return out


def online_frame(store, design, settings=FULL):
    tilted = _finished(store, settings.experiment("tilt_online"))
    base = _finished(store, "suite_budget_tuned")
    h = _promoted_hash()
    rows = {}
    for b in ONLINE_BUDGETS:
        for lag in LAGS:
            for arm, df, dname in (("untilted", base, None), ("tilted", tilted, design)):
                if len(df) == 0:
                    sel = df
                else:
                    sel = df[(df["params.method"] == ONLINE_METHOD)
                             & (_num(df, "params.budget") == b) & (df["params.lag"] == lag)
                             & df["params.sid"].isin(SSVI_SIDS)]
                    if "params.recipe_hash" in sel.columns:
                        sel = sel[sel["params.recipe_hash"] == h]
                    if dname is not None:
                        sel = sel[sel["params.design"] == dname]
                if len(sel) == 0:
                    rows[(b, lag, arm)] = pd.Series({"mae": np.nan, "wings": np.nan,
                                                     "liquid": np.nan, "secs": np.nan})
                    continue
                per = pd.DataFrame({"sid": sel["params.sid"],
                                    "mae": _num(sel, "metrics.pooled_mae_bp"),
                                    "wings": _num(sel, "metrics.wings_mae_bp"),
                                    "liquid": _num(sel, "metrics.liquid_mae_bp"),
                                    "secs": _num(sel, "metrics.online_s")}).groupby("sid").mean()
                rows[(b, lag, arm)] = pd.Series({"mae": per.mae.mean(), "wings": per.wings.mean(),
                                                 "liquid": per.liquid.mean(),
                                                 "secs": per.secs.median()})
    out = pd.DataFrame(rows).T
    out.index = pd.MultiIndex.from_tuples(out.index, names=["budget", "lag", "arm"])
    return out


def _slices_tex(df, design):
    names = [UNTILTED] + [d.name for d in DESIGNS]
    cols = " ".join("rrr" for _ in names)
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{n}}}" for n in names)
    sub = " & ".join("std $T{=}0.25$ & std & bias" for _ in names)
    lines = [f"\\begin{{tabular}}{{l {cols}}}", "\\toprule", f"estimator, $N$ & {head}\\\\",
             "& " + sub + "\\\\", "\\midrule"]
    rows = []
    for est, elabel in EST_LABELS.items():
        for n in sorted({i[1] for i in df.index}) if len(df) else []:
            cells = []
            for name in names:
                r = df.loc[(est, n, name)] if (est, n, name) in df.index else None
                for c in ("std_T0.25", "std_pooled", "bias_pooled"):
                    cells.append((np.nan if r is None else float(r[c]), "{:.1f}"))
            rows.append((f"{elabel}, {_label(n)}", cells))
    lines += _render_rows(rows)
    lines += ["\\bottomrule", "\\end{tabular}", f"% winner: {design}"]
    return "\n".join(lines) + "\n"


def _cold_tex(df):
    """Errors and seconds go through `_render_rows` (bold = column minimum); the ESS column is
    "bigger is better" and is appended unbolded, `--` for the untilted rows and the floor."""
    header = ("method, $N$, arm & pooled & wings & $T{=}0.25$ & liquid & lev.\\ RMSE"
              " & min ESS & s\\\\")
    lines = ["\\begin{tabular}{l rrrr r r r}", "\\toprule", header, "\\midrule"]
    rows, ess = [], []
    for (label, n, arm), r in df.iterrows():
        prefix = label if label == FLOOR_LABEL else f"{label}, {_label(n)}, {arm}"
        cells = [(r["mae"], "{:.0f}"), (r["wings"], "{:.0f}"), (r["t025"], "{:.0f}"),
                 (r["liquid"], "{:.0f}"), (r["lev"], "{:.3f}"), (r["secs"], "{:.1f}")]
        rows.append((prefix, cells))
        ess.append(_cell(r["ess_min"], "{:.2f}"))
    rendered = _render_rows(rows, secs_cols={5})
    for line, e in zip(rendered, ess):
        head, secs = line[: -len("\\\\")].rsplit(" & ", 1)
        lines.append(f"{head} & {e} & {secs}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _online_tex(df):
    lines = ["\\begin{tabular}{l rrr r}", "\\toprule",
             "budget, lag, arm & pooled & wings & liquid & online s\\\\", "\\midrule"]
    rows = [(f"{_label(b)}, {lag}, {arm}",
             [(r["mae"], "{:.0f}"), (r["wings"], "{:.0f}"), (r["liquid"], "{:.0f}"),
              (r["secs"], "{:.1f}")]) for (b, lag, arm), r in df.iterrows()]
    lines += _render_rows(rows, secs_cols={3})
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def tilt_tables(store, design, out_dir="paper/tables", settings=FULL):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "tilt_slices.tex": _slices_tex(slice_frame(store, settings), design),
        "tilt_cold.tex": _cold_tex(cold_frame(store, design, settings)),
        "tilt_online.tex": _online_tex(online_frame(store, design, settings)),
    }
    paths = []
    for name, text in files.items():
        (out / name).write_text(text)
        paths.append(out / name)
    return paths
