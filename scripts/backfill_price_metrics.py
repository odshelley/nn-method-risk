"""Back-fill price-space metrics onto every finished suite run from its stored implied-vol error
grid.

The model implied vol is target + err, and the model price is Black-Scholes at that vol (the
inversion was exact), so the absolute price error is recovered without any simulation. Metrics
logged, in bp of spot: price_bp/pooled, price_bp/wings, price_bp/near, price_bp/max, price_bp/T<T>.
Runs that already carry price_bp/pooled are skipped. Usage: python scripts/backfill_price_metrics.py
"""
import json
import tempfile
import time

import numpy as np
from mlflow.entities import Metric

from neural_particle_method.bench.scenarios import full_registry, quote_k_grid
from neural_particle_method.market.bs import bs_call
from neural_particle_method.pricing.reprice import snap_times
from neural_particle_method.suite.lag import LAGS, lagged_scenario
from neural_particle_method.tracking.store import Store

EXPERIMENTS = ("suite_cold", "suite_pde_floor", "suite_offline", "suite_lagged", "suite_cold_full",
               "suite_offline_full", "suite_lagged_full", "suite_offline_conv", "suite_lagged_conv",
               "suite_budget", "suite_cold_sub30k", "suite_offline_sub30k", "suite_lagged_sub30k")
LAG_BY_KIND = {lag.kind: lag for lag in LAGS}


def price_metrics(sc, err_bp, n_steps=200):
    k = quote_k_grid()
    mats = list(sc.maturities)
    ts = snap_times(mats, n_steps)
    tgt = sc.target_ivs(k, ts)
    mod = tgt + np.asarray(err_bp, dtype=float) / 1e4
    K = sc.s0 * np.exp(k)
    p_t = np.stack([bs_call(sc.s0, K, t, tgt[i]) for i, t in enumerate(ts)])
    p_m = np.stack([bs_call(sc.s0, K, t, mod[i]) for i, t in enumerate(ts)])
    dp = np.abs(p_m - p_t) / sc.s0 * 1e4
    wing = np.abs(k) > 0.25
    out = {"price_bp/pooled": np.nanmean(dp), "price_bp/wings": np.nanmean(dp[:, wing]),
           "price_bp/near": np.nanmean(dp[:, ~wing]), "price_bp/max": np.nanmax(dp)}
    out.update({f"price_bp/T{m:g}": np.nanmean(dp[i]) for i, m in enumerate(mats)})
    return {key: float(v) for key, v in out.items() if np.isfinite(v)}


def main():
    store = Store()
    reg = full_registry()
    for exp in EXPERIMENTS:
        if store.client.get_experiment_by_name(exp) is None:
            continue
        df = store.search(exp)
        df = df[df.status == "FINISHED"]
        if "metrics.price_bp/pooled" in df.columns:
            df = df[df["metrics.price_bp/pooled"].isna()]
        n = 0
        with tempfile.TemporaryDirectory() as d:
            for _, r in df.iterrows():
                sc = reg[r["params.sid"]]
                lag_kind = r.get("params.lag")
                if isinstance(lag_kind, str) and lag_kind in LAG_BY_KIND:
                    sc = lagged_scenario(sc, LAG_BY_KIND[lag_kind])
                try:
                    err = json.loads(store.download(r.run_id, "iv_err_bp.json", d).read_text())
                except Exception as e:  # noqa: BLE001 - a run without the artifact is skipped
                    print(f"skip {exp} {r.run_id}: {e!r}")
                    continue
                m = price_metrics(sc, err)
                ts_ms = int(time.time() * 1000)
                metrics = [Metric(key, v, ts_ms, 0) for key, v in m.items()]
                store.client.log_batch(r.run_id, metrics=metrics)
                n += 1
        print(f"{exp}: {n} runs back-filled", flush=True)


if __name__ == "__main__":
    main()
