"""Compare the three arms on the six study scenarios: main suite (30k fits), full-cloud, converged."""
import pandas as pd

from neural_particle_method.tracking.store import Store

pd.set_option("display.width", 240)
s = Store()
SIDS = ["s01", "s02", "s05", "s09", "s11", "s16"]
ORDER = ["stale_L", "nw_resolve", "explicit_stale", "implicit_stale", "explicit_rkhs", "implicit_rkhs",
         "explicit_ridge", "implicit_ridge"]


def num(df, c):
    return pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(float("nan"), index=df.index)


print("== Offline bodies at 500k (anchor scores on the overnight surface, pooled / wings MAE bp, lev_rmse, train s)")
rows = []
for exp, label in (("suite_offline", "paper, 30k fits"), ("suite_offline_full", "paper, 100k fits"),
                   ("suite_offline_conv", "converged, 100k fits")):
    d = s.search(exp)
    d = d[(d.status == "FINISHED") & d["params.sid"].isin(SIDS) & (d["params.n_particles"] == "500000")]
    for body, g in d.groupby("params.body"):
        rows.append({"arm": label, "body": body, "n": len(g), "pooled": num(g, "metrics.pooled_mae_bp").mean(),
                     "wings": num(g, "metrics.wings_mae_bp").mean(), "T=2": num(g, "metrics.mae_bp/T2").mean(),
                     "lev_rmse": num(g, "metrics.lev_rmse").median(), "train_s": num(g, "metrics.total_s").median()})
print(pd.DataFrame(rows).round(1).to_string(index=False))

for lagk in ("surface", "surface_spot"):
    out = {}
    for exp, label in (("suite_lagged", "30k"), ("suite_lagged_full", "100k"), ("suite_lagged_conv", "conv")):
        d = s.search(exp)
        d = d[(d.status == "FINISHED") & d["params.sid"].isin(SIDS) & (d["params.lag"] == lagk)
              & d["params.offline_n"].isin(["500000", "0"])]
        if len(d) == 0:
            continue
        g = d.groupby("params.method").apply(lambda x: pd.Series({
            "MAE": num(x, "metrics.pooled_mae_bp").mean(), "wings": num(x, "metrics.wings_mae_bp").mean(),
            "T=2": num(x, "metrics.mae_bp/T2").mean(), "s": num(x, "metrics.online_s").median()}))
        out[label] = g
    t = pd.concat(out, axis=1).reindex(ORDER).round(1)
    print(f"\n== Lagged ({lagk}), 500k bodies, six scenarios x 2 seeds")
    print(t.to_string())
