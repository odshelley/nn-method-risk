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
BIAS_FLOOR = 1.0         # percentage points of f a design may add to the bias for free...
BIAS_PER_STD = 0.5       # ...plus this share of the wing standard deviation it removes
MIN_SEEDS = 2            # a (sid, T, k) cell needs this many seeds to carry a standard deviation
WINNER_N = 30_000
ALGO_LABELS = {"nw": "NW (20 scenarios)",
               "explicit_nn_opt": "Explicit NN, searched (six study scenarios)",
               "explicit_nn_opt_dw": "Explicit NN, density-weighted (six study scenarios)"}
EST_LABELS = {"nw": "NW", "net": "Net"}


def _promoted_hash():
    return recipe_hash(load_recipe("explicit_opt")) if recipe_exists("explicit_opt") else None


def _label(n):
    return f"{int(n) // 1000}k"


def _tex(name):
    """Design and lag names carry underscores; they are set in text mode."""
    return str(name).replace("_", r"\_")


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
        return pd.DataFrame(columns=["std_T0.25", "std_pooled", "bias_pooled", "n_cells",
                                     "n_seeds_min"])
    g = long.groupby(["est", "n", "design", "sid", "T", "k"])["err"]
    per = pd.DataFrame({"std": g.std(ddof=1), "bias": g.mean(), "seeds": g.count()}).reset_index()
    out = {}
    for (est, n, design), d in per.groupby(["est", "n", "design"]):
        first = d[np.isclose(d["T"], d["T"].min())]
        scored = d[d["seeds"] >= MIN_SEEDS]     # cells that carry a standard deviation at all
        out[(est, int(n), design)] = pd.Series({
            "std_T0.25": 100 * first["std"].mean(),
            "std_pooled": 100 * d["std"].mean(),
            "bias_pooled": 100 * d["bias"].mean(),
            "n_cells": float(len(scored)),
            "n_seeds_min": float(scored["seeds"].min()) if len(scored) else 0.0,
        })
    df = pd.DataFrame(out).T
    df.index = pd.MultiIndex.from_tuples(df.index, names=["est", "n", "design"])
    return df.sort_index()


def winner(frame, n=WINNER_N, est="nw"):
    """The design with the smallest NW wing std at `n`; "none" when no admissible design beats
    untilted (a NaN std, e.g. a single seed, never beats anything).

    A design is admissible only if the bias it *adds* to the untilted arm is small relative to
    the standard deviation it removes: `|bias_d - bias_u| <= BIAS_FLOOR + BIAS_PER_STD x
    max(std_u - std_d, 0)`. The total bias is dominated by the Euler-versus-Fokker-Planck
    discretisation error, which is common to both arms and which no tilt can touch, so only the
    difference is the tilt's doing. A design whose (sid, T, k) coverage does not match the
    untilted arm's, or which has a cell with fewer than MIN_SEEDS seeds, is skipped with a
    warning: its averages are taken over a different set of cells and are not comparable."""
    if (est, n, UNTILTED) not in frame.index:
        return UNTILTED
    base = frame.loc[(est, n, UNTILTED)]
    best, best_std = UNTILTED, base["std_pooled"]
    for d in DESIGNS:
        if (est, n, d.name) not in frame.index:
            continue
        row = frame.loc[(est, n, d.name)]
        if row["n_cells"] != base["n_cells"] or row["n_seeds_min"] < MIN_SEEDS:
            print(f"tilt winner: skipping {d.name}: {int(row['n_cells'])} scored cells against "
                  f"the untilted arm's {int(base['n_cells'])}, fewest seeds in a cell "
                  f"{int(row['n_seeds_min'])}", flush=True)
            continue
        bought = max(base["std_pooled"] - row["std_pooled"], 0.0)
        if abs(row["bias_pooled"] - base["bias_pooled"]) > BIAS_FLOOR + BIAS_PER_STD * bought:
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
                # the network's cold rows were run on the six study scenarios at every budget;
                # the 80k budget also exists on the other 14 (for the far-wing table), so pin the
                # set here to keep the three budgets comparable within this table
                sids = SLICE_SIDS if algo.startswith("explicit_nn_opt") else SSVI_SIDS
                sel = df.iloc[:0] if len(df) == 0 else df[
                    (df["params.algo"] == algo) & (_num(df, "params.n_particles") == n)
                    & (df["params.design"] == dname) & df["params.sid"].isin(sids)]
                if algo.startswith("explicit_nn_opt") and len(sel):
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


FAR_COLS = ("far", "t05", "t1", "t2", "n", "pooled")
FAR_ROWS = (
    "NW, 200 steps, untilted", "NW, 200 steps, tilted",
    "NW, 400 steps, untilted", "NW, 400 steps, tilted",
    "Explicit NN, searched, untilted", "Explicit NN, searched, tilted",
    "Explicit NN, density-weighted, untilted", "Explicit NN, density-weighted, tilted",
    "NW re-solve (online)",
    "Searched body + spline head, untilted", "Searched body + spline head, tilted",
    "PDE floor, 200 steps", "PDE floor, 400 steps",
)


def _far_metrics(sel):
    if len(sel) == 0:
        return pd.Series({c: np.nan for c in FAR_COLS})
    per = pd.DataFrame({
        "sid": sel["params.sid"], "far": _num(sel, "metrics.far_wings_mae_bp"),
        "t05": _num(sel, "metrics.far_mae_bp/T0.5"), "t1": _num(sel, "metrics.far_mae_bp/T1"),
        "t2": _num(sel, "metrics.far_mae_bp/T2"), "n": _num(sel, "metrics.far_wings_n"),
        "pooled": _num(sel, "metrics.pooled_mae_bp"),
    }).groupby("sid").mean(numeric_only=True)
    return per.mean()


def far_frame(store, design, settings=FULL, budget=80_000):
    """Far-wing scores per row of \\cref{tab:tilt-far}: per-scenario mean over seeds, then mean
    over scenarios, of each metric; missing runs give NaN."""
    cold = _finished(store, settings.experiment("tilt_cold"))
    cold_s400 = _finished(store, settings.experiment("tilt_cold_s400"))
    online = _finished(store, settings.experiment("tilt_online"))
    base = _finished(store, "suite_budget_tuned")
    floor = _finished(store, settings.experiment("suite_pde_floor"))
    floor_s400 = _finished(store, settings.experiment("suite_pde_floor_s400"))

    def cold_sel(df, algo, dname):
        if len(df) == 0:
            return df
        sel = df[(df["params.algo"] == algo) & (_num(df, "params.n_particles") == budget)
                 & (df["params.design"] == dname) & df["params.sid"].isin(SSVI_SIDS)]
        if algo.startswith("explicit_nn_opt") and len(sel):
            sel = (sel[sel["params.recipe_hash"] == _promoted_hash()]
                   if "params.recipe_hash" in sel.columns else sel.iloc[:0])
        return sel

    def resolve_sel():
        if len(base) == 0:
            return base
        return base[(base["params.method"] == "nw_resolve")
                    & (_num(base, "params.budget") == budget)
                    & (base["params.lag"] == "surface") & base["params.sid"].isin(SSVI_SIDS)]

    def head_sel(df, dname):
        if len(df) == 0:
            return df
        sel = df[(df["params.method"] == "explicit_opt_spline")
                 & (_num(df, "params.budget") == budget) & (df["params.lag"] == "surface")
                 & df["params.sid"].isin(SSVI_SIDS)]
        if dname is None:
            sel = (sel[sel["params.recipe_hash"] == _promoted_hash()]
                   if "params.recipe_hash" in sel.columns else sel.iloc[:0])
        else:
            sel = sel[sel["params.design"] == dname]
        return sel

    def floor_sel(df, n_steps=None):
        if len(df) == 0:
            return df
        sel = df[df["params.sid"].isin(SSVI_SIDS)]
        if n_steps is not None and "params.n_steps" in sel.columns:
            sel = sel[_num(sel, "params.n_steps") == n_steps]
        return sel

    rows = {
        "NW, 200 steps, untilted": _far_metrics(cold_sel(cold, "nw", UNTILTED)),
        "NW, 200 steps, tilted": _far_metrics(cold_sel(cold, "nw", design)),
        "NW, 400 steps, untilted": _far_metrics(cold_sel(cold_s400, "nw", UNTILTED)),
        "NW, 400 steps, tilted": _far_metrics(cold_sel(cold_s400, "nw", design)),
        "Explicit NN, searched, untilted": _far_metrics(
            cold_sel(cold, "explicit_nn_opt", UNTILTED)),
        "Explicit NN, searched, tilted": _far_metrics(cold_sel(cold, "explicit_nn_opt", design)),
        "Explicit NN, density-weighted, untilted": _far_metrics(
            cold_sel(cold, "explicit_nn_opt_dw", UNTILTED)),
        "Explicit NN, density-weighted, tilted": _far_metrics(
            cold_sel(cold, "explicit_nn_opt_dw", design)),
        "NW re-solve (online)": _far_metrics(resolve_sel()),
        "Searched body + spline head, untilted": _far_metrics(head_sel(base, None)),
        "Searched body + spline head, tilted": _far_metrics(head_sel(online, design)),
        "PDE floor, 200 steps": _far_metrics(floor_sel(floor, n_steps=200)),
        "PDE floor, 400 steps": _far_metrics(floor_sel(floor_s400)),
    }
    return pd.DataFrame({label: rows[label] for label in FAR_ROWS}).T


def _slices_tex(df, design):
    names = [UNTILTED] + [d.name for d in DESIGNS]
    cols = " ".join("rrr" for _ in names)
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{_tex(n)}}}" for n in names)
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
    # the bias is signed: its column minimum is the most negative bias, not the best one
    lines += _render_rows(rows, plain_cols={3 * i + 2 for i in range(len(names))})
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
    rows = [(f"{_label(b)}, {_tex(lag)}, {arm}",
             [(r["mae"], "{:.0f}"), (r["wings"], "{:.0f}"), (r["liquid"], "{:.0f}"),
              (r["secs"], "{:.1f}")]) for (b, lag, arm), r in df.iterrows()]
    lines += _render_rows(rows, secs_cols={3})
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _far_tex(df):
    """Non-floor rows go through `_render_rows` (bold = column minimum); the two floor rows are
    appended unbolded via `_cell`, since the floor is a reference, not a contender."""
    header = ("row & far wings & $T{=}0.5$ & $T{=}1$ & $T{=}2$ & quotes"
              " & pooled (13-strike)\\\\")
    lines = ["\\begin{tabular}{l rrrrrr}", "\\toprule", header, "\\midrule"]
    floors = [label for label in df.index if label.startswith("PDE floor")]
    body = [label for label in df.index if label not in floors]
    fmt = "{:.0f}"
    rows = [(label, [(df.loc[label, c], fmt) for c in FAR_COLS]) for label in body]
    lines += _render_rows(rows, plain_cols={4})      # the quote count is not a score
    for label in floors:
        cells = " & ".join(_cell(df.loc[label, c], fmt) for c in FAR_COLS)
        lines.append(f"{label} & {cells}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def tilt_tables(store, design, out_dir="paper/tables", settings=FULL):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "tilt_slices.tex": _slices_tex(slice_frame(store, settings), design),
        "tilt_cold.tex": _cold_tex(cold_frame(store, design, settings)),
        "tilt_online.tex": _online_tex(online_frame(store, design, settings)),
        "tilt_far.tex": _far_tex(far_frame(store, design, settings)),
    }
    paths = []
    for name, text in files.items():
        (out / name).write_text(text)
        paths.append(out / name)
    return paths
