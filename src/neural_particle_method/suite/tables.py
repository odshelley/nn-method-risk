"""Section-4 tables of paper/notes_experiments.tex from the suite experiments. Bare tabulars.

Three tables: the cold suite, the offline bodies, and the frozen tuned body with an online head
across online particle budgets. SSVI scenarios only.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ..estimators.recipes import load_recipe, recipe_exists, recipe_hash
from .config import FULL, SSVI_SIDS

COLD_ROWS = (
    ("nw", "NW"), ("explicit_nn", "Explicit NN, short training"),
    ("explicit_nn_tuned", "Explicit NN, tuned"),
    ("explicit_nn_opt", "Explicit NN, searched"),
    ("implicit_nn", "Implicit NN"), ("rkhs", "RKHS"),
    ("spline", "Spline"), ("nw_ghl", "GHL kernel"), ("bins", "Bins"), ("muguruza", "Muguruza"),
    ("purbf", "PU-RBF"), ("pde", "PDE (attainable floor)"),
)
BODY_ROWS = (
    ("explicit", "Explicit NN, short training"),
    ("explicit_tuned", "Explicit NN, tuned"),
    ("explicit_opt", "Explicit NN, searched"),
    ("implicit", "Implicit NN"),
)
HEAD_ROWS = (
    ("nw_resolve", "NW re-solve on $S_1$"),
    ("explicit_tuned_stale", "Tuned body, stale $f$, fresh Dupire"),
    ("explicit_tuned_spline", "Tuned body + spline head"),
    ("explicit_opt_spline", "Searched body + spline head"),
    ("explicit_tuned_rkhs", "Tuned body + RKHS head"),
    ("explicit_tuned_ridge", "Tuned body + ridge head"),
)
LAGS = ("surface", "surface_spot")
LAG_GROUPS = {"surface": "surface lag", "surface_spot": "surface plus $2\\%$ spot lag"}
FLOOR_LABEL = "PDE (attainable floor)"
BUDGET_EXPERIMENT = "suite_budget_tuned"
BODY_OFFLINE_N = "500000"


def _sids():
    return set(SSVI_SIDS)


def _only_promoted_recipe(sel):
    """Keep the rows written with the recipe that is currently promoted.

    `validate --top k` leaves searched bodies and budget cells for every candidate in the store,
    so a row that quoted them all would average the winner with the recipes it beat. Nothing
    promoted, or runs written before the hash was logged, means the row has no numbers.
    """
    if not recipe_exists("explicit_opt") or "params.recipe_hash" not in sel.columns:
        return sel.iloc[:0]
    return sel[sel["params.recipe_hash"] == recipe_hash(load_recipe("explicit_opt"))]


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
        mae=("mae", "mean"), wings=("wings", "mean"), liquid=("liquid", "mean"),
        lev=("lev", "median"), lat=("lat", "median"), total=("total", "median"),
        p_pooled=("p_pooled", "mean"),
    )
    return pd.Series({
        "mae_mean": per_sid.mae.mean(), "mae_median": per_sid.mae.median(),
        "wings_mean": per_sid.wings.mean(), "wings_median": per_sid.wings.median(),
        "liquid_mean": per_sid.liquid.mean(), "lev_median": per_sid.lev.median(),
        "lat_median": per_sid.lat.median(), "total_median": per_sid.total.median(),
        "p_pooled_mean": per_sid.p_pooled.mean(), "p_pooled_median": per_sid.p_pooled.median(),
    })


def _prep(df):
    out = pd.DataFrame(index=df.index)
    out["params.sid"] = df["params.sid"]
    out["mae"] = _num(df, "metrics.pooled_mae_bp")
    out["wings"] = _num(df, "metrics.wings_mae_bp")
    out["liquid"] = _num(df, "metrics.liquid_mae_bp")
    out["lev"] = _num(df, "metrics.lev_rmse")
    if "metrics.online_s" in df.columns:
        out["lat"] = _num(df, "metrics.online_s")
    else:
        out["lat"] = _num(df, "metrics.fit_s")
    out["total"] = _num(df, "metrics.total_s")
    out["p_pooled"] = _num(df, "metrics.price_bp/pooled")
    return out


EMPTY = pd.Series({k: np.nan for k in (
    "mae_mean", "mae_median", "wings_mean", "wings_median", "liquid_mean", "lev_median",
    "lat_median", "total_median", "p_pooled_mean", "p_pooled_median",
)})


def _pde_solve_seconds(store, settings):
    """Median, over the SSVI scenarios, of the Fokker-Planck solve time (`runtime_s` on the
    `pde_reference` run at the suite's step count, unlagged)."""
    df = _finished(store, "pde_reference")
    if len(df) == 0:
        return np.nan
    sel = df[df["params.sid"].isin(_sids())
             & (df["params.n_steps"] == str(settings.explicit.n_steps))
             & (df["params.lag"] == "none")]
    if len(sel) == 0:
        return np.nan
    per_sid = _num(sel, "metrics.runtime_s").groupby(sel["params.sid"]).median()
    return per_sid.median()


def cold_frame(store, settings=FULL):
    cold = _finished(store, settings.experiment("suite_cold"))
    pde = _finished(store, settings.experiment("suite_pde_floor"))
    rows = {}
    for algo, label in COLD_ROWS:
        src = pde if algo == "pde" else cold
        if len(src) == 0:
            rows[label] = EMPTY.copy()
            continue
        sel = src[src["params.sid"].isin(_sids())]
        if algo != "pde":
            sel = sel[sel["params.algo"] == algo]
        row = _agg(_prep(sel)) if len(sel) else EMPTY.copy()
        if algo == "pde":
            row = row.copy()
            row["lat_median"] = _pde_solve_seconds(store, settings)
        rows[label] = row
    return pd.DataFrame(rows).T


def bodies_frame(store, settings=FULL):
    """One row per (offline body, particle count), plus the PDE floor of the same scoring."""
    off = _finished(store, settings.experiment("suite_offline"))
    rows = {}
    for body, label in BODY_ROWS:
        for n in settings.offline_sizes:
            key = (label, f"{n // 1000}k")
            if len(off) == 0:
                rows[key] = EMPTY.copy()
                continue
            sel = off[off["params.sid"].isin(_sids())
                      & (off["params.body"] == body)
                      & (_num(off, "params.n_particles") == n)]
            if body.startswith("explicit_opt"):
                sel = _only_promoted_recipe(sel)
            rows[key] = _agg(_prep(sel)) if len(sel) else EMPTY.copy()
    floor = EMPTY.copy()
    pde = cold_frame(store, settings).loc[FLOOR_LABEL]
    for k in ("mae_mean", "mae_median", "wings_mean", "liquid_mean", "p_pooled_mean"):
        floor[k] = pde[k]
    floor["total_median"] = pde["lat_median"]
    rows[(FLOOR_LABEL, "--")] = floor
    df = pd.DataFrame(rows).T.rename(columns={"total_median": "train_s"})
    df.index = pd.MultiIndex.from_tuples(df.index, names=["body", "size"])
    return df


def _label(budget):
    return f"{budget // 1000}k"


def _head_cells(store, settings, method, budget, lag=None):
    """The finished runs for one head method at one online budget, optionally one lag.

    The suite's own online budget lives in `suite_lagged` (keyed by offline body size); the
    smaller budgets are the sweep of `scripts/budget_sweep.py` in `suite_budget_tuned`.
    """
    if budget == settings.n_online:
        df = _finished(store, settings.experiment("suite_lagged"))
        if len(df) == 0:
            return df
        offline_n = "0" if method == "nw_resolve" else BODY_OFFLINE_N
        sel = df[(df["params.method"] == method) & (df["params.offline_n"] == offline_n)]
    else:
        df = _finished(store, BUDGET_EXPERIMENT)
        if len(df) == 0:
            return df
        sel = df[(df["params.method"] == method)
                 & (_num(df, "params.budget") == budget)]
    if method.startswith("explicit_opt"):
        sel = _only_promoted_recipe(sel)
    sel = sel[sel["params.sid"].isin(_sids())]
    return sel if lag is None else sel[sel["params.lag"] == lag]


def heads_frame(store, settings=FULL, budgets=(10_000, 30_000, 80_000, 100_000)):
    """Pooled MAE per (head method, lag, online budget), with the liquid error and the online
    seconds at the extreme budgets."""
    rows = {}
    for method, label in HEAD_ROWS:
        row = {}
        for lag in LAGS:
            for b in budgets:
                sel = _head_cells(store, settings, method, b, lag)
                agg = _agg(_prep(sel)) if len(sel) else EMPTY
                row[f"mae_{lag}_{_label(b)}"] = agg["mae_mean"]
                if b == budgets[-1]:
                    row[f"liquid_{lag}_{_label(b)}"] = agg["liquid_mean"]
        for b in (budgets[0], budgets[-1]):
            sel = _head_cells(store, settings, method, b)
            row[f"s_{_label(b)}"] = (_num(sel, "metrics.online_s").median() if len(sel)
                                     else np.nan)
        rows[label] = pd.Series(row)
    return pd.DataFrame(rows).T


def _cell(x, fmt="{:.0f}"):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)


FLOOR_LABEL = "PDE (attainable floor)"


def _render_rows(rows, secs_cols=()):
    """Render table rows, bolding the best value in every column.

    `rows` is a list of (prefix, cells) with cells a list of (value, fmt). The best is the
    smallest value among the rows whose prefix is not the PDE floor (the reference, never a
    contender); in the `secs_cols` columns a zero (no online work) does not count. Ties at the
    displayed precision are all bold.
    """
    n = max((len(cells) for _, cells in rows), default=0)
    best = []
    for j in range(n):
        vals = [cells[j][0] for prefix, cells in rows
                if not prefix.startswith(FLOOR_LABEL) and cells[j][0] is not None
                and np.isfinite(cells[j][0]) and not (j in secs_cols and cells[j][0] <= 0)]
        best.append(min(vals) if vals else None)
    lines = []
    for prefix, cells in rows:
        out = []
        for j, (v, fmt) in enumerate(cells):
            txt = _cell(v, fmt)
            if (best[j] is not None and not prefix.startswith(FLOOR_LABEL) and txt != "--"
                    and txt == fmt.format(best[j])):
                txt = f"\\textbf{{{txt}}}"
            out.append(txt)
        lines.append(f"{prefix} & " + " & ".join(out) + "\\\\")
    return lines


def _cold_tex(df):
    lines = [
        "\\begin{tabular}{l rrrrr r}", "\\toprule",
        "method & \\multicolumn{4}{c}{implied vol, bp} & price, bp & s\\\\",
        "\\cmidrule(lr){2-5}",
        "& mean & median & liquid & wings & pooled & median\\\\", "\\midrule",
    ]
    f0, f1 = "{:.0f}", "{:.1f}"
    rows = [(label, [(r.mae_mean, f0), (r.mae_median, f0), (r.liquid_mean, f0),
                     (r.wings_mean, f0), (r.p_pooled_mean, f1), (r.lat_median, f1)])
            for label, r in df.iterrows()]
    lines += _render_rows(rows, secs_cols={5})
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _bodies_tex(df):
    lines = [
        "\\begin{tabular}{l l rrr r r}", "\\toprule",
        "body & particles & pooled & liquid & wings & price, bp & s\\\\", "\\midrule",
    ]
    f0, f1 = "{:.0f}", "{:.1f}"
    rows = [(f"{label} & {size}", [(r.mae_mean, f0), (r.liquid_mean, f0), (r.wings_mean, f0),
                                   (r.p_pooled_mean, f1), (r.train_s, f1)])
            for (label, size), r in df.iterrows()]
    lines += _render_rows(rows, secs_cols={4})
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _heads_tex(df, budgets=(10_000, 30_000, 80_000, 100_000)):
    groups = " & ".join(f"\\multicolumn{{5}}{{c}}{{{LAG_GROUPS[lag]}}}" for lag in LAGS)
    sub = []
    for _ in LAGS:
        sub += [_label(b) for b in budgets] + ["liquid"]
    sub += [_label(budgets[0]), _label(budgets[-1])]
    lines = [
        "\\begin{tabular}{l rrrrr rrrrr rr}", "\\toprule",
        f"method & {groups} & \\multicolumn{{2}}{{c}}{{online s}}\\\\",
        "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\\cmidrule(lr){12-13}",
        "& " + " & ".join(sub) + "\\\\", "\\midrule",
    ]
    f0, f1 = "{:.0f}", "{:.1f}"
    rows = []
    for label, r in df.iterrows():
        cells = []
        for lag in LAGS:
            cells += [(r[f"mae_{lag}_{_label(b)}"], f0) for b in budgets]
            cells.append((r[f"liquid_{lag}_{_label(budgets[-1])}"], f0))
        cells += [(r[f"s_{_label(b)}"], f1) for b in (budgets[0], budgets[-1])]
        rows.append((label, cells))
    n_vol = 5 * len(LAGS)
    lines += _render_rows(rows, secs_cols={n_vol, n_vol + 1})
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def section4_tables(store, out_dir="paper/tables", settings=FULL):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    budgets = (10_000, 30_000, 80_000, settings.n_online)
    files = {
        "suite_cold.tex": _cold_tex(cold_frame(store, settings)),
        "suite_bodies.tex": _bodies_tex(bodies_frame(store, settings)),
        "suite_heads.tex": _heads_tex(heads_frame(store, settings, budgets), budgets),
    }
    paths = []
    for name, text in files.items():
        (out / name).write_text(text)
        paths.append(out / name)
    return paths
