"""PDE floor: the Fokker-Planck reference leverage repriced under the suite's Monte Carlo."""
import json
import tempfile

from ..bench.reference_runs import run_reference
from ..bench.scenarios import full_registry, quote_k_grid
from ..pricing.metrics import iv_metrics
from ..pricing.reprice import reprice_iv, snap_times
from ..simulate.leverage import LeverageField
from ..tracking.store import git_hash
from .config import FULL


def score_field(field, sc, seed, reprice):
    """Reprice `field` on scenario `sc` and score it; returns (metrics dict in run_one's names, err
    grid)."""
    k = quote_k_grid()
    mats = list(sc.maturities)
    iv_model = reprice_iv(field, sc.dynamics, sc.s0, mats, k, reprice, seed=seed + 10_000)
    iv_target = sc.target_ivs(k, snap_times(mats, reprice.n_steps))
    m = iv_metrics(iv_model, iv_target, k, mats)
    out = {c: m[c] for c in ("pooled_rmse_bp", "pooled_max_bp", "wings_rmse_bp", "wings_max_bp",
                             "n_failed", "pooled_mae_bp", "wings_mae_bp", "liquid_mae_bp")}
    out.update({f"rmse_bp/T{r['T']:g}": r["rmse_bp"] for r in m["per_maturity"]})
    out.update({f"mae_bp/T{r['T']:g}": r["mae_bp"] for r in m["mae_per_maturity"]})
    return out, ((iv_model - iv_target) * 1e4).tolist()


def run_pde_floor(store, sid, seed, settings=FULL):
    n_steps = settings.explicit.n_steps
    key = {"sid": sid, "seed": int(seed), "n_steps": int(n_steps)}
    exp = settings.experiment("suite_pde_floor")
    existing = store.find_finished(exp, key)
    if existing is not None:
        return existing
    sc = full_registry()[sid]
    ref = run_reference(store, sid, n_steps=n_steps, n_x=settings.n_x, n_v=settings.n_v)
    with tempfile.TemporaryDirectory() as d:
        raw = store.download(ref, "leverage.json", d).read_text()
        field = LeverageField.from_json(json.loads(raw))
    params = {**key, "reference_run": ref, "git_hash": git_hash(), **sc.as_params(),
              **{f"reprice.{k}": v for k, v in settings.reprice.as_params().items()}}
    with store.run(exp, params) as h:
        metrics, err = score_field(field, sc, seed, settings.reprice)
        solve_s = store.get_metrics(ref)["runtime_s"]
        metrics.update({"fit_s": 0.0, "total_s": 0.0, "lev_rmse": 0.0, "solve_s": solve_s})
        h.log_metrics(metrics)
        h.log_json("leverage.json", field.to_json())
        h.log_json("iv_err_bp.json", err)
        return h.run_id
