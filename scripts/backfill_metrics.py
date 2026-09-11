"""Back-fill price-space and liquid-quote metrics onto every finished suite run from its stored
implied-vol error grid.

The model implied vol is target + err, and the model price is Black-Scholes at that vol (the
inversion was exact), so the absolute price error is recovered without any simulation. Metrics
logged, in bp of spot: price_bp/pooled, price_bp/wings, price_bp/near, price_bp/liquid,
price_bp/max, price_bp/T<T>; plus liquid_mae_bp, the mean absolute vol error over the liquid
quotes (|k| <= 0.25 and T >= 0.5), which runs recorded before that metric existed do not carry.
A run is refilled when it lacks price_bp/pooled or liquid_mae_bp, and only the missing keys are
logged. Usage: python scripts/backfill_metrics.py
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
               "suite_budget", "suite_budget_tuned", "suite_cold_sub30k", "suite_offline_sub30k",
               "suite_lagged_sub30k")
LAG_BY_KIND = {lag.kind: lag for lag in LAGS}
WING_CUT = 0.25
LIQUID_T = 0.5


def liquid_mask(sc, k, wing_cut=WING_CUT, liquid_T=LIQUID_T):
    """Quotes a desk would actually trade: near the money and not the front maturity."""
    mats = np.asarray([float(T) for T in sc.maturities])
    return (mats[:, None] >= liquid_T) & (np.abs(np.asarray(k))[None, :] <= wing_cut)


def liquid_metrics(sc, err_bp):
    """Mean absolute implied-vol error, in vol bp, over the liquid quotes."""
    err = np.abs(np.asarray(err_bp, dtype=float))
    m = liquid_mask(sc, quote_k_grid())
    v = np.nanmean(err[m]) if m.any() else float("nan")
    return {"liquid_mae_bp": float(v)} if np.isfinite(v) else {}


def _absent(row, key):
    v = row.get(key)
    return v is None or (isinstance(v, float) and np.isnan(v))


def _wanted(row):
    """Which metric families this run is still missing."""
    return tuple(name for name, key in (("price", "metrics.price_bp/pooled"),
                                        ("liquid", "metrics.liquid_mae_bp"))
                 if _absent(row, key))


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
    wing = np.abs(k) > WING_CUT
    liquid = liquid_mask(sc, k)
    out = {"price_bp/pooled": np.nanmean(dp), "price_bp/wings": np.nanmean(dp[:, wing]),
           "price_bp/near": np.nanmean(dp[:, ~wing]), "price_bp/max": np.nanmax(dp),
           "price_bp/liquid": np.nanmean(dp[liquid]) if liquid.any() else float("nan")}
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
        df = df[[_wanted(r) != () for _, r in df.iterrows()]]
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
                want = _wanted(r)
                m = {}
                if "price" in want:
                    m.update(price_metrics(sc, err))
                if "liquid" in want:
                    m.update(liquid_metrics(sc, err))
                if not m:
                    continue
                ts_ms = int(time.time() * 1000)
                metrics = [Metric(key, v, ts_ms, 0) for key, v in m.items()]
                store.client.log_batch(r.run_id, metrics=metrics)
                n += 1
        print(f"{exp}: {n} runs back-filled", flush=True)


if __name__ == "__main__":
    main()
