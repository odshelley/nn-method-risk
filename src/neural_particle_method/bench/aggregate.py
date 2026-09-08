"""Query the bench experiment into a tidy summary table and a short digest."""
from pathlib import Path

import pandas as pd

from .runner import BENCH_EXPERIMENT

METRIC_COLS = ("pooled_rmse_bp", "pooled_max_bp", "wings_rmse_bp", "wings_max_bp", "n_failed")
COLUMNS = ("sid", "algo", "n_particles", "seed", "status") + METRIC_COLS + \
          ("total_s", "fit_s", "intraday_s", "git_hash") + ("budget", "knob_name", "knob_value")


def aggregate(store, out_csv="results/summary.csv", out_md="results/digest.md", experiment=BENCH_EXPERIMENT):
    raw = store.search(experiment)
    rows = []
    for _, r in raw.iterrows():
        row = {"sid": r.get("params.sid"), "algo": r.get("params.algo"),
               "n_particles": int(r["params.n_particles"]) if pd.notna(r.get("params.n_particles")) else None,
               "seed": int(r["params.seed"]) if pd.notna(r.get("params.seed")) else None,
               "status": {"FINISHED": "ok", "FAILED": "failed"}.get(r["status"], r["status"].lower())}
        for c in METRIC_COLS + ("total_s", "fit_s", "intraday_s"):
            row[c] = r.get(f"metrics.{c}")
        row["git_hash"] = r.get("params.git_hash")
        row["budget"] = int(r["params.budget"]) if pd.notna(r.get("params.budget")) else None
        row["knob_name"] = r.get("params.knob_name")
        row["knob_value"] = float(r["params.knob_value"]) if pd.notna(r.get("params.knob_value")) else None
        rows.append(row)
    df = pd.DataFrame(rows, columns=COLUMNS)
    if len(df):
        df = df.sort_values(["sid", "algo", "n_particles", "seed"]).reset_index(drop=True)
        df["n_failed"] = df["n_failed"].astype("Int64")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    n_fail = int((df.status == "failed").sum())
    lines = ["# Benchmark digest", f"{len(df)} runs, {n_fail} failed", ""]
    ok = df[df.status == "ok"]
    if len(ok):
        best = ok.groupby(["sid", "algo"]).pooled_rmse_bp.mean().reset_index() \
                 .sort_values("pooled_rmse_bp").groupby("sid").first()
        for sid, r in best.iterrows():
            lines.append(f"- {sid}: best = {r.algo} ({r.pooled_rmse_bp:.1f} bp)")
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines) + "\n")
    return df
