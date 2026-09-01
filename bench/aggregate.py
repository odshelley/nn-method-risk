"""Flatten run JSONs into a tidy summary table and a short digest."""
import json
from pathlib import Path

import pandas as pd

METRIC_COLS = ("pooled_rmse_bp", "pooled_max_bp", "wings_rmse_bp", "wings_max_bp", "n_failed")
COLUMNS = ("sid", "algo", "n_particles", "seed", "status") + METRIC_COLS + \
          ("total_s", "fit_s", "git_hash")


def aggregate(runs_dir="results/runs", out_csv="results/summary.csv",
              out_md="results/digest.md"):
    rows = []
    for p in sorted(Path(runs_dir).rglob("*.json")):
        d = json.loads(p.read_text())
        row = {k: d.get(k) for k in ("sid", "algo", "n_particles", "seed", "status")}
        m, t = d.get("metrics") or {}, d.get("timings") or {}
        for c in METRIC_COLS:
            row[c] = m.get(c)
        row["total_s"], row["fit_s"] = t.get("total_s"), t.get("fit_s")
        row["git_hash"] = d.get("git_hash")
        rows.append(row)
    df = pd.DataFrame(rows, columns=COLUMNS)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    n_fail = int((df.status == "failed").sum())
    lines = [f"# Benchmark digest", f"{len(df)} runs, {n_fail} failed", ""]
    ok = df[df.status == "ok"]
    if len(ok):
        best = ok.groupby(["sid", "algo"]).pooled_rmse_bp.mean().reset_index() \
                 .sort_values("pooled_rmse_bp").groupby("sid").first()
        for sid, r in best.iterrows():
            lines.append(f"- {sid}: best = {r.algo} ({r.pooled_rmse_bp:.1f} bp)")
    Path(out_md).write_text("\n".join(lines) + "\n")
    return df
