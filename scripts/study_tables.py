"""Study tables: the estimator grid, the body-training comparison and the small-budget sweep.
Reads results/estimator_grid.csv and the study experiments in the store, writes
paper/tables/study_grid.tex, study_bodies.tex, study_arms.tex, study_budget.tex (bare tabulars).

Section 5 of paper/notes_experiments.tex quotes these numbers in prose and no longer inputs the
files; run this script when the studies are rerun and the prose has to be checked against them.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from neural_particle_method.tracking.store import Store

OUT = Path("paper/tables")
SIDS = ["s01", "s02", "s05", "s09", "s11", "s16"]
METHOD_LABELS = {
    "nw_resolve": "NW re-solve",
    "explicit_stale": "Explicit body, stale $f$, fresh Dupire",
    "explicit_rkhs": "Explicit body + RKHS head",
    "explicit_spline": "Explicit body + spline head",
    "explicit_ridge": "Explicit body + ridge head",
}
GRID_ROWS = [("nw_30k", "NW, 30k particles"), ("nw_100k", "NW, 100k"), ("nw_500k", "NW, 500k"),
             ("spline_100k", "Spline, 100k"), ("rkhs_100k", "RKHS, 100k"),
             ("nn_h64_d2_s120_lr0.01_b0", "Net $64\\times2$, 120 full-batch steps (paper)"),
             ("nn_h64_d2_s1000_lr0.01_b0", "Net $64\\times2$, 1000 full-batch steps"),
             ("nn_h64_d2_s20000_lr0.001_b8192", "Net $64\\times2$, 20000 minibatch steps"),
             ("nn_h64_d3_s120_lr0.01_b0", "Net $64\\times3$, 120 full-batch steps"),
             ("nn_h64_d3_s20000_lr0.001_b8192", "Net $64\\times3$, 20000 minibatch steps"),
             ("nn_h256_d2_s20000_lr0.001_b8192", "Net $256\\times2$, 20000 minibatch steps")]


def cell(x, fmt="{:.1f}"):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)


def num(df, c):
    return (pd.to_numeric(df[c], errors="coerce") if c in df.columns
            else pd.Series(np.nan, index=df.index))


def _pooled_mae(g):
    return num(g, "metrics.pooled_mae_bp").mean()


def grid_table():
    df = pd.read_csv("results/estimator_grid.csv")
    lines = ["\\begin{tabular}{l rr rr r}", "\\toprule",
             " & \\multicolumn{2}{c}{s02 wings} & \\multicolumn{2}{c}{s11 wings} & s11 centre\\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "estimator & $t=0.25$ & $t=1$ & $t=0.25$ & $t=1$ & mean & s / fit\\\\", "\\midrule"]
    lines[0] = "\\begin{tabular}{l rr rr r r}"
    for key, label in GRID_ROWS:
        d = df[df.estimator == key]

        def pick(sid, t, col="rel_err_wings", d=d):
            x = d[(d.sid == sid) & (np.isclose(d.t, t))][col]
            return float(x.iloc[0]) * 100 if len(x) else np.nan

        c11 = d[d.sid == "s11"]["rel_err_center"].mean() * 100 if len(d[d.sid == "s11"]) else np.nan
        secs = d.secs.median() if len(d) else np.nan
        cells = [
            cell(pick("s02", 0.25)),
            cell(pick("s02", 1.0)),
            cell(pick("s11", 0.25)),
            cell(pick("s11", 1.0)),
            cell(c11),
            cell(secs, "{:.0f}"),
        ]
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def bodies_table(store):
    rows = [
        ("suite_offline_sub30k", "Short training, 30k fits"),
        ("suite_offline_full", "Short training, 100k fits"),
        ("suite_offline_conv", "Long training, 100k fits"),
    ]
    lines = ["\\begin{tabular}{l l r r r r}", "\\toprule",
             "training & body & pooled MAE & wings MAE & $T=2$ MAE & train s\\\\", "\\midrule"]
    for exp, label in rows:
        d = store.search(exp)
        d = d[
            (d.status == "FINISHED")
            & d["params.sid"].isin(SIDS)
            & (d["params.n_particles"] == "500000")
        ]
        for body in ("explicit", "implicit"):
            g = d[d["params.body"] == body]
            lines.append(
                f"{label} & {body} & "
                f"{cell(num(g, 'metrics.pooled_mae_bp').mean(), '{:.0f}')} & "
                f"{cell(num(g, 'metrics.wings_mae_bp').mean(), '{:.0f}')} & "
                f"{cell(num(g, 'metrics.mae_bp/T2').mean(), '{:.0f}')} & "
                f"{cell(num(g, 'metrics.total_s').median(), '{:.0f}')}\\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def arms_table(store):
    arms = [
        ("suite_lagged_sub30k", "30k"),
        ("suite_lagged_full", "100k"),
        ("suite_lagged_conv", "conv"),
    ]
    order = ["nw_resolve", "explicit_stale", "explicit_rkhs", "explicit_spline", "explicit_ridge"]
    data = {}
    for exp, label in arms:
        d = store.search(exp)
        d = d[
            (d.status == "FINISHED")
            & d["params.sid"].isin(SIDS)
            & d["params.offline_n"].isin(["500000", "0"])
        ]
        for lagk in ("surface", "surface_spot"):
            x = d[d["params.lag"] == lagk]
            per = x.groupby(["params.method", "params.sid"]).apply(_pooled_mae)
            data[(label, lagk)] = per.groupby(level=0).mean()
    lines = ["\\begin{tabular}{l rrr rrr}", "\\toprule",
             " & \\multicolumn{3}{c}{surface lag} & \\multicolumn{3}{c}{surface plus spot lag}\\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             ("online method & short, 30k fits & short, 100k fits & long, 100k fits "
              "& short, 30k fits & short, 100k fits & long, 100k fits\\\\"), "\\midrule"]
    for m in order:
        cells = [
            cell(data[(a, lagk)].get(m, np.nan), "{:.0f}")
            for lagk in ("surface", "surface_spot")
            for a in ("30k", "100k", "conv")
        ]
        lines.append(f"{METHOD_LABELS[m]} & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def budget_table(store):
    d = store.search("suite_budget")
    d = d[d.status == "FINISHED"]
    d["budget"] = pd.to_numeric(d["params.budget"])
    c = store.search("suite_lagged_conv")
    c = c[(c.status == "FINISHED") & c["params.method"].isin(METHOD_LABELS)]
    c["budget"] = 100_000
    d = pd.concat([d, c], ignore_index=True)
    order = ["nw_resolve", "explicit_stale", "explicit_spline", "explicit_rkhs", "explicit_ridge"]
    budgets = [10_000, 30_000, 80_000, 100_000]
    lines = [
        "\\begin{tabular}{l rrrr rrrr r}", "\\toprule",
        (" & \\multicolumn{4}{c}{surface lag, pooled MAE} & "
         "\\multicolumn{4}{c}{surface plus spot lag} & \\\\"),
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
        ("online method & 10k & 30k & 80k & 100k & 10k & 30k & 80k & 100k & "
         "s at 10k\\\\"),
        "\\midrule",
    ]
    for m in order:
        cells = []
        for lagk in ("surface", "surface_spot"):
            for b in budgets:
                x = d[(d["params.method"] == m) & (d["params.lag"] == lagk) & (d["budget"] == b)]
                per = x.groupby("params.sid").apply(_pooled_mae)
                cells.append(cell(per.mean() if len(per) else np.nan, "{:.0f}"))
        x = d[(d["params.method"] == m) & (d["budget"] == 10_000)]
        cells.append(cell(num(x, "metrics.online_s").median()))
        lines.append(f"{METHOD_LABELS[m]} & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    store = Store()
    OUT.mkdir(parents=True, exist_ok=True)
    tables = (
        ("study_grid.tex", grid_table()),
        ("study_bodies.tex", bodies_table(store)),
        ("study_arms.tex", arms_table(store)),
        ("study_budget.tex", budget_table(store)),
    )
    for name, text in tables:
        (OUT / name).write_text(text)
        print("wrote", OUT / name)
