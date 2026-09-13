# Far-Wing Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score the stored leverage fields on far-wing vanilla quotes (|k| up to 0.8) with an importance-sampled out-of-the-money repricer, so the tilt study's leverage gain is measured where it acts, on quotes a desk holds.

**Architecture (the spec, agreed with Osian 13 Sept evening):** the existing `reprice_iv` (calls, untilted) stays bit-for-bit and keeps scoring section 4. A new `reprice_iv_otm` prices the out-of-the-money option at each strike (put for k < 0, call for k > 0), optionally on a tilted cloud (the study's three-component mixture, weights accumulated exactly as `calibrate_explicit` does, payoffs weighted), and inverts through put-call parity with the existing `implied_vol`. A `far_wing_metrics` keeps only quotes whose target OTM price is at least `1e-5` of spot (the three-month far wings drop out) and reports the MAE over |k| > 0.25 of that grid. `suite/far_wings.py` re-scores stored runs (no recalibration) and logs `far_*` metrics and `far_iv_err_bp.json` onto them; `nparticle tilt farwings` drives it; `tilt_far.tex` and a notes paragraph report it.

**Tech Stack:** numpy, scipy (`brentq` through `market/bs.py`), mlflow via `tracking.store.Store` (`client.log_batch`, `client.log_artifact`), pytest, ruff, `uv run`.

## Global Constraints

- `reprice_iv` and `iv_metrics` are untouched; `uv run pytest tests/test_golden_tiny.py -q` (unmarked) must pass after every task.
- `reprice_iv_otm(..., mixture=None)` draws `zb`, `zp` per step in the same order as `reprice_iv`, so its call prices (k > 0) are bit-identical to `reprice_iv`'s on the same field, seed and config.
- Far grid: `far_k_grid()` = `np.log(np.geomspace(0.45, 2.2, 21))` (|k| up to about 0.80). Price floor `FAR_PRICE_FLOOR = 1e-5` (of spot). Wing cut 0.25. Reprice tilt for scoring: `TiltDesign("constant", 3.0)` (weights bounded by 2).
- Metric names: `far_wings_mae_bp`, `far_wings_n`, `far_all_mae_bp`, `far_mae_bp/T{T:g}`, `far_n/T{T:g}`; artifact `far_iv_err_bp.json` (list of lists, NaN where failed or below the floor).
- Re-score seed: `params.seed + 20_000`; step count and path count from the run's own `params["reprice.n_steps"]` and `params["reprice.n_particles"]`.
- Line length 100 (`awk 'length > 100' <file>`), ruff clean, `uv run pytest -q` green; never run a full-size command; commit trailers:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01EhbyGqFA5kYDyMwYouSgvS`.

---

### Task 1: OTM tilted repricer and far-wing metrics

**Files:**
- Modify: `src/neural_particle_method/pricing/reprice.py`
- Modify: `src/neural_particle_method/pricing/metrics.py`
- Test: `tests/pricing/test_far_wings.py`

**Interfaces:**
- Consumes: `calibrate/importance.MixtureDesign` (`.alphas`, `.thetas`, `.etas`, `.scheduled`), `heston_step(..., theta_p=)`, `market/bs.implied_vol`, `market/bs.bs_call`.
- Produces: `far_k_grid() -> ndarray(21)`; `reprice_iv_otm(field, params, s0, maturities, k_grid, cfg=RepriceConfig(), seed=10_000, mixture=None) -> (ivs, prices, ess_final)`; `far_wing_metrics(iv_model, iv_target, k_grid, times, s0, wing_cut=0.25, price_floor=FAR_PRICE_FLOOR) -> (metrics dict, err_bp ndarray with NaN where dropped)`; constant `FAR_PRICE_FLOOR = 1e-5`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/pricing/test_far_wings.py
import numpy as np

from neural_particle_method.bench.scenarios import make_registry
from neural_particle_method.calibrate.importance import TiltDesign
from neural_particle_method.market.bs import bs_call
from neural_particle_method.pricing.metrics import far_wing_metrics
from neural_particle_method.pricing.reprice import (
    FAR_PRICE_FLOOR,
    RepriceConfig,
    far_k_grid,
    reprice_iv,
    reprice_iv_otm,
    snap_times,
)
from neural_particle_method.simulate.leverage import DEFAULT_GRID, LeverageField, Slice


def _flat_field(n_steps, T):
    return LeverageField([Slice(k * T / n_steps, DEFAULT_GRID.copy(), np.ones_like(DEFAULT_GRID),
                                np.full_like(DEFAULT_GRID, 0.04)) for k in range(n_steps)])


def test_far_grid():
    k = far_k_grid()
    assert k.shape == (21,) and abs(k[0] - np.log(0.45)) < 1e-12 and abs(k[-1] - np.log(2.2)) < 1e-12


def test_untilted_calls_are_bit_identical_to_reprice_iv():
    sc = make_registry()["s01"]
    cfg = RepriceConfig(4_000, 8)
    k = np.array([0.05, 0.15, 0.3])
    a = reprice_iv(_flat_field(8, sc.T), sc.dynamics, sc.s0, [0.5, 1.0], k, cfg, seed=5)
    b, prices, ess = reprice_iv_otm(_flat_field(8, sc.T), sc.dynamics, sc.s0, [0.5, 1.0], k, cfg,
                                    seed=5)
    np.testing.assert_array_equal(a, b)
    assert ess == 1.0 and prices.shape == (2, 3)


def test_put_side_inverts_through_parity():
    sc = make_registry()["s01"]
    cfg = RepriceConfig(50_000, 8)
    k = np.array([-0.3, -0.15, 0.15, 0.3])
    ivs, prices, _ = reprice_iv_otm(_flat_field(8, sc.T), sc.dynamics, sc.s0, [1.0], k, cfg, seed=1)
    assert np.all(np.isfinite(ivs))
    # a flat unit leverage on v0 = theta = 0.04 Heston: vols near 20 percent, put and call sides alike
    assert np.all(np.abs(ivs - 0.2) < 0.03)
    assert np.all(prices > 0) and prices[0, 0] < prices[0, 1]      # deeper put is cheaper


def test_tilted_reprice_agrees_with_untilted_and_reaches_the_wings():
    sc = make_registry()["s01"]
    cfg = RepriceConfig(200_000, 8)
    mix = TiltDesign("constant", 3.0).mixture(8, sc.T, sc.dynamics.rho)
    k = np.array([-0.6, -0.3, 0.0, 0.3, 0.6])
    u, pu, _ = reprice_iv_otm(_flat_field(8, sc.T), sc.dynamics, sc.s0, [2.0], k, cfg, seed=2)
    t, pt, ess = reprice_iv_otm(_flat_field(8, sc.T), sc.dynamics, sc.s0, [2.0], k, cfg, seed=2,
                                mixture=mix)
    assert 0.5 < ess < 1.0
    ok = np.isfinite(u) & np.isfinite(t)
    assert ok.sum() >= 3
    assert np.nanmax(np.abs(u[ok] - t[ok])) < 0.01          # within MC noise at 200k paths
    assert np.all(np.isfinite(t[0, [0, -1]]))                # the tilted cloud prices the far wings


def test_far_wing_metrics_drop_quotes_below_the_price_floor():
    k = far_k_grid()
    times = [0.25, 2.0]
    tgt = np.full((2, 21), 0.2)
    mod = tgt + 0.001                                          # 10 bp everywhere
    m, err = far_wing_metrics(mod, tgt, k, times, 1.0)
    K = np.exp(k)
    p = np.stack([bs_call(1.0, K, t, 0.2) for t in times])
    p = np.where(k[None, :] < 0, p - 1.0 + K[None, :], p)      # OTM price via parity
    kept = (p >= FAR_PRICE_FLOOR) & (np.abs(k)[None, :] > 0.25)
    assert m["far_wings_n"] == int(kept.sum()) and m["far_wings_n"] < 2 * 21
    assert abs(m["far_wings_mae_bp"] - 10.0) < 1e-6
    assert np.isnan(err[0, 0]) and abs(err[1, 10]) < 1e-9 + 10.0   # 3-month far put dropped
    assert m["far_n/T0.25"] < m["far_n/T2"]
    assert abs(m["far_all_mae_bp"] - 10.0) < 1e-6
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/pricing/test_far_wings.py -q`
Expected: FAIL with `ImportError` (`far_k_grid`).

- [ ] **Step 3: Implement in `pricing/reprice.py`**

Add after `snap_times`:

```python
FAR_PRICE_FLOOR = 1e-5   # of spot: quotes whose target OTM price is below this are not scored


def far_k_grid():
    """21 log-strikes from 0.45 to 2.2 spot: the quoted 13 plus the far wings out to |k| ~ 0.8."""
    return np.log(np.geomspace(0.45, 2.2, 21))


def reprice_iv_otm(field, params, s0, maturities, k_grid, cfg=RepriceConfig(), seed=10_000,
                   mixture=None):
    """Out-of-the-money repricing: puts for k < 0, calls for k >= 0, inverted to IV through put-call
    parity; optionally on a tilted cloud (the study's mixture, exact discrete-time weights) so the
    far wings are populated. Untilted, the call side is bit-identical to `reprice_iv`.

    Returns (ivs, otm_prices, ess_final) with shapes (n_maturities, n_strikes)."""
    hp = HestonParams.from_dict(params)
    field = LeverageField.from_records(field)
    n_particles, n_steps = cfg.n_particles, cfg.n_steps
    rng = np.random.default_rng(seed)
    T = max(maturities)
    dt, sdt = T / n_steps, np.sqrt(T / n_steps)
    t_snap = dict(zip(maturities, snap_times(maturities, n_steps, T)))
    snap = {}
    for m in maturities:
        snap.setdefault(int(round(m / dt)), []).append(m)  # noqa: RUF046
    lnx = np.full(n_particles, np.log(s0))
    v = np.full(n_particles, hp.v0)
    w = np.ones(n_particles)
    theta_p = None
    if mixture is not None:
        comp = rng.choice(3, size=n_particles, p=list(mixture.alphas))
        thetas_all, etas_all = np.asarray(mixture.thetas), np.asarray(mixture.etas)
        alphas = np.array(mixture.alphas)
        ell = np.zeros((3, n_particles))
    k_grid = np.asarray(k_grid, dtype=float)
    ivs = np.full((len(maturities), len(k_grid)), np.nan)
    prices = np.full_like(ivs, np.nan)
    mat_idx = {m: i for i, m in enumerate(maturities)}
    for step in range(n_steps):
        t = step * dt
        if mixture is not None:
            th = thetas_all[:, step] if mixture.scheduled else thetas_all
            etas = etas_all[:, step] if mixture.scheduled else etas_all
            theta_p, eta_p = th[comp], etas[comp]
        L_p = field.at(t, lnx)
        zb = rng.standard_normal(n_particles)
        zp = rng.standard_normal(n_particles)
        lnx, v = heston_step(lnx, v, L_p, zb, zp, hp, dt, sdt, theta_p=theta_p)
        if mixture is not None:
            dbperp = zp * sdt + eta_p * dt
            ell += etas[:, None] * dbperp[None, :] - 0.5 * (etas ** 2)[:, None] * dt
            w = 1.0 / (alphas @ np.exp(np.clip(ell, -60, 60)))
        if step + 1 in snap:
            x = np.exp(lnx)
            for m in snap[step + 1]:
                for j, k in enumerate(k_grid):
                    K = s0 * np.exp(k)
                    if k < 0:
                        payoff = np.maximum(K - x, 0.0)
                    else:
                        payoff = np.maximum(x - K, 0.0)
                    price = float(np.mean(w * payoff)) if mixture is not None else float(np.mean(payoff))
                    prices[mat_idx[m], j] = price
                    call = price if k >= 0 else price + s0 - K
                    ivs[mat_idx[m], j] = implied_vol(call, s0, K, t_snap[m])
    ess = float(w.sum() ** 2 / (n_particles * (w ** 2).sum())) if mixture is not None else 1.0
    return ivs, prices, ess
```

The untilted call branch computes `float(np.mean(payoff))` with the same `payoff` array `reprice_iv` uses, so it is bit-identical; the `w * payoff` product is taken only under a mixture.

- [ ] **Step 4: Implement `far_wing_metrics` in `pricing/metrics.py`**

```python
from ..market.bs import bs_call            # add to the imports
from .reprice import FAR_PRICE_FLOOR       # add to the imports


def far_wing_metrics(iv_model, iv_target, k_grid, times, s0, wing_cut=0.25,
                     price_floor=FAR_PRICE_FLOOR):
    """MAE in vol bp over the far-wing quotes that are priceable: |k| > wing_cut and target OTM
    price >= price_floor * s0. `times` are the snapped maturities the IVs were inverted at.

    Returns (metrics, err_bp) with err_bp NaN wherever a quote is dropped or failed."""
    k = np.asarray(k_grid, dtype=float)
    K = s0 * np.exp(k)
    tgt = np.asarray(iv_target, dtype=float)
    call = np.stack([bs_call(s0, K, float(t), tgt[i]) for i, t in enumerate(times)])
    otm = np.where(k[None, :] < 0, call - s0 + K[None, :], call)
    err = (np.asarray(iv_model, dtype=float) - tgt) * 1e4
    keep = (otm >= price_floor * s0) & np.isfinite(err)
    wings = np.tile(np.abs(k) > wing_cut, (len(times), 1))
    err = np.where(keep, err, np.nan)

    def mae(mask):
        e = err[mask & keep]
        return float("nan") if e.size == 0 else float(np.mean(np.abs(e)))

    out = {"far_wings_mae_bp": mae(wings), "far_wings_n": int((wings & keep).sum()),
           "far_all_mae_bp": mae(np.ones_like(keep))}
    for i, t in enumerate(times):
        row = np.zeros_like(keep)
        row[i] = True
        out[f"far_mae_bp/T{float(t):g}"] = mae(row & wings)
        out[f"far_n/T{float(t):g}"] = int((row & wings & keep).sum())
    return {kk: v for kk, v in out.items() if not (isinstance(v, float) and np.isnan(v))}, err
```

Note the test keys `far_n/T0.25`, `far_n/T2` come from `float(t):g` formatting of 0.25 and 2.0.

- [ ] **Step 5: Run tests, goldens, ruff**

Run: `uv run pytest tests/pricing -q && uv run pytest tests/test_golden_tiny.py -q && uv run ruff check src tests && awk 'length > 100' src/neural_particle_method/pricing/reprice.py src/neural_particle_method/pricing/metrics.py tests/pricing/test_far_wings.py`
Expected: PASS, clean. If `test_tilted_reprice_agrees_with_untilted_and_reaches_the_wings` is flaky at the 0.01 tolerance, raise the path count to 400 000 rather than the tolerance, and report it.

- [ ] **Step 6: Commit**

```bash
git add src/neural_particle_method/pricing/reprice.py src/neural_particle_method/pricing/metrics.py tests/pricing/test_far_wings.py
git commit -m "pricing: OTM tilted repricer and far-wing metrics with a price floor" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01EhbyGqFA5kYDyMwYouSgvS"
```

---

### Task 2: Re-scorer over stored runs and the CLI

**Files:**
- Create: `src/neural_particle_method/suite/far_wings.py`
- Modify: `src/neural_particle_method/cli.py` (`tilt` subparser gains `farwings`)
- Test: `tests/suite/test_far_wings.py`, `tests/test_cli_tilt.py`

**Interfaces:**
- Consumes: Task 1; `Store` (`search`, `download`, `client.log_batch`, `client.log_artifact`), `full_registry`, `lagged_scenario`, `LAGS`, `LeverageField.from_json`, `TiltDesign`, `snap_times`, `recipe_hash(load_recipe("explicit_opt"))`, `SSVI_SIDS`.
- Produces: `REPRICE_TILT = TiltDesign("constant", 3.0)`; `far_score(field, sc, seed, reprice_cfg) -> (metrics, err)`; `rescore_run(store, run_id, force=False) -> dict | None` (None when already scored and not forced); `select_runs(store) -> list[(experiment, run_id)]`; `run_farwings(store, n_jobs=1, force=False) -> (done, failed)`; CLI `nparticle tilt farwings [--jobs N] [--force]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/suite/test_far_wings.py
import pytest

from neural_particle_method.estimators import recipes as R
from neural_particle_method.suite.config import SuiteSettings
from neural_particle_method.suite.far_wings import REPRICE_TILT, rescore_run, run_farwings, select_runs
from neural_particle_method.suite.tilt import run_tilt_cold
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()
FAST = {"hidden": 16, "depth": 2, "lr": 1e-2, "batch_size": 0, "first_steps": 5,
        "later_steps": 2, "weight_decay": 0.0, "fit_subsample": 400, "warm_start": True,
        "mean_match": True, "monotone": False, "monotone_penalty": 0.0, "tail": "linear",
        "hetero": False}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RECIPE_DIR", tmp_path / "recipes")
    R.save_recipe("explicit_opt", FAST)
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def test_rescore_logs_far_metrics_once(store):
    rid = run_tilt_cold(store, "s01", "nw", TINY.n_online, 0, "none", TINY)
    m = rescore_run(store, rid)
    assert m is not None and "far_wings_mae_bp" in m and m["far_wings_n"] > 0
    got = store.get_metrics(rid)
    assert got["far_wings_n"] == m["far_wings_n"] and "far_all_mae_bp" in got
    names = {a.path for a in store.client.list_artifacts(rid)}
    assert "far_iv_err_bp.json" in names
    assert rescore_run(store, rid) is None            # already scored
    assert rescore_run(store, rid, force=True) is not None
    assert REPRICE_TILT.name == "constant-3"


def test_select_runs_targets_the_80k_rows_and_floors(store):
    a = run_tilt_cold(store, "s01", "nw", 80_000 if False else TINY.n_online, 0, "none", TINY)
    sel = select_runs(store, settings=TINY, budget=TINY.n_online)
    assert (TINY.experiment("tilt_cold"), a) in sel
    done, failed = run_farwings(store, settings=TINY, budget=TINY.n_online)
    assert (done, failed) == (1, 0)
    assert run_farwings(store, settings=TINY, budget=TINY.n_online) == (0, 0)
```

And in `tests/test_cli_tilt.py` add:

```python
def test_farwings_parses():
    a = _parser().parse_args(["tilt", "farwings", "--jobs", "3", "--force"])
    assert (a.tilt_cmd, a.jobs, a.force) == ("farwings", 3, True)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/suite/test_far_wings.py tests/test_cli_tilt.py -q`
Expected: FAIL with `ModuleNotFoundError` / argparse invalid choice.

- [ ] **Step 3: Implement `suite/far_wings.py`**

```python
"""Far-wing re-scoring of stored leverage fields with the OTM tilted repricer (no recalibration).

Selected runs (all at the 80k budget unless told otherwise): the tilt study's cold rows in
`tilt_cold` and `tilt_cold_s400`, the PDE floors at 200 and 400 steps, the tilted online cells in
`tilt_online` and their untilted partners and the NW re-solve in `suite_budget_tuned`. Each run
gets `far_*` metrics and a `far_iv_err_bp.json` artifact; a run already carrying
`far_wings_mae_bp` is skipped unless forced.
"""
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from mlflow.entities import Metric

from ..bench.scenarios import full_registry
from ..calibrate.importance import TiltDesign
from ..estimators.recipes import load_recipe, recipe_exists, recipe_hash
from ..pricing.metrics import far_wing_metrics
from ..pricing.reprice import RepriceConfig, far_k_grid, reprice_iv_otm, snap_times
from ..simulate.leverage import LeverageField
from ..suite.lag import LAGS, lagged_scenario
from ..tracking.store import Store
from .config import FULL, SSVI_SIDS
from .grid import _drain

REPRICE_TILT = TiltDesign("constant", 3.0)
SEED_OFFSET = 20_000
LAG_BY_KIND = {lag.kind: lag for lag in LAGS}


def far_score(field, sc, seed, reprice_cfg):
    """Far-wing IV metrics of `field` on scenario `sc` with the tilted OTM repricer."""
    k = far_k_grid()
    mats = list(sc.maturities)
    times = snap_times(mats, reprice_cfg.n_steps, T=sc.T)
    mix = REPRICE_TILT.mixture(reprice_cfg.n_steps, sc.T, sc.dynamics.rho)
    ivs, _, ess = reprice_iv_otm(field, sc.dynamics, sc.s0, mats, k, reprice_cfg,
                                 seed=seed + SEED_OFFSET, mixture=mix)
    tgt = sc.target_ivs(k, times)
    m, err = far_wing_metrics(ivs, tgt, k, times, sc.s0)
    m["far_reprice_ess"] = ess
    return m, err


def _scenario(params):
    sc = full_registry()[params["sid"]]
    lag_kind = params.get("lag")
    if isinstance(lag_kind, str) and lag_kind in LAG_BY_KIND:
        sc = lagged_scenario(sc, LAG_BY_KIND[lag_kind])
    return sc


def rescore_run(store, run_id, force=False):
    run = store.client.get_run(run_id)
    if not force and "far_wings_mae_bp" in run.data.metrics:
        return None
    p = dict(run.data.params)
    sc = _scenario(p)
    cfg = RepriceConfig(int(p.get("reprice.n_particles", 500_000)),
                        int(p.get("reprice.n_steps", 200)))
    with tempfile.TemporaryDirectory() as d:
        field = LeverageField.from_json(json.loads(store.download(run_id, "leverage.json", d)
                                                   .read_text()))
        m, err = far_score(field, sc, int(p.get("seed", 0)), cfg)
        ts = int(time.time() * 1000)
        store.client.log_batch(run_id, metrics=[Metric(k, float(v), ts, 0) for k, v in m.items()])
        pth = Path(d) / "far_iv_err_bp.json"
        pth.write_text(json.dumps(np.where(np.isfinite(err), err, None).tolist()))
        store.client.log_artifact(run_id, str(pth))
    return m


def _finished(store, exp):
    if store.client.get_experiment_by_name(exp) is None:
        return None
    df = store.search(exp)
    return df[df.status == "FINISHED"] if len(df) else df


def select_runs(store, settings=FULL, budget=80_000):
    """(experiment, run_id) pairs to score, deduplicated, in a stable order."""
    out = []
    h = recipe_hash(load_recipe("explicit_opt")) if recipe_exists("explicit_opt") else None
    sids = set(SSVI_SIDS)

    def take(exp, df):
        for rid in df.run_id:
            out.append((exp, rid))

    for exp in (settings.experiment("tilt_cold"), settings.experiment("tilt_cold_s400")):
        df = _finished(store, exp)
        if df is not None and len(df):
            take(exp, df[(df["params.n_particles"] == str(budget)) & df["params.sid"].isin(sids)])
    for exp in (settings.experiment("suite_pde_floor"), settings.experiment("suite_pde_floor_s400")):
        df = _finished(store, exp)
        if df is not None and len(df):
            take(exp, df[df["params.sid"].isin(sids)])
    df = _finished(store, settings.experiment("tilt_online"))
    if df is not None and len(df):
        take(settings.experiment("tilt_online"), df[df["params.budget"] == str(budget)])
    df = _finished(store, settings.experiment("suite_budget_tuned"))
    if df is not None and len(df):
        sel = df[(df["params.budget"] == str(budget)) & df["params.sid"].isin(sids)]
        if "params.recipe_hash" in sel.columns:
            head = sel[(sel["params.method"] == "explicit_opt_spline")
                       & (sel["params.recipe_hash"] == h)]
        else:
            head = sel.iloc[:0]
        resolve = sel[sel["params.method"] == "nw_resolve"]
        take(settings.experiment("suite_budget_tuned"), head)
        take(settings.experiment("suite_budget_tuned"), resolve)
    seen, uniq = set(), []
    for item in out:
        if item[1] not in seen:
            seen.add(item[1]); uniq.append(item)
    return uniq


def _worker(args):
    uri, root, exp, rid, force, n_jobs = args
    if n_jobs > 1:
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or n_jobs) // n_jobs)))
    try:
        m = rescore_run(Store(uri, root), rid, force=force)
        return (exp, rid, "skip" if m is None else "scored"), None
    except Exception as e:  # noqa: BLE001
        return (exp, rid), repr(e)


def run_farwings(store, settings=FULL, budget=80_000, n_jobs=1, force=False):
    todo = select_runs(store, settings, budget)
    if not force:
        todo = [(e, r) for e, r in todo
                if "far_wings_mae_bp" not in store.client.get_run(r).data.metrics]
    print(f"farwings: {len(todo)} runs to score", flush=True)
    args = [(store.tracking_uri, store.artifact_root, e, r, force, n_jobs) for e, r in todo]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            return _drain(pool.map(_worker, args))
    return _drain(map(_worker, args))
```

Fix the test's silly `80_000 if False else TINY.n_online` to `TINY.n_online` when transcribing (it is there to remind you that `select_runs` filters on `budget`, which the test passes as `TINY.n_online`). `_drain` counts `done` for `err is None`, so a skipped run counts as done; the second `run_farwings` call filters it out beforehand, giving (0, 0).

- [ ] **Step 4: CLI**

In `_parser()`, inside the `tilt` subparsers: `q = ts.add_parser("farwings"); q.add_argument("--jobs", type=int, default=1); q.add_argument("--force", action="store_true"); q.add_argument("--smoke", action="store_true")`. In `main()`'s tilt branch: `if args.tilt_cmd == "farwings": from .suite.far_wings import run_farwings; done, failed = run_farwings(store, settings, budget=settings.n_online if args.smoke else 80_000, n_jobs=args.jobs, force=args.force); print(f"farwings: {done} done, {failed} failed"); return 1 if failed else 0` (place it before the `chosen()` users so it does not need a design).

- [ ] **Step 5: Run tests, ruff**

Run: `uv run pytest tests/suite/test_far_wings.py tests/test_cli_tilt.py -q && uv run ruff check src tests && awk 'length > 100' src/neural_particle_method/suite/far_wings.py src/neural_particle_method/cli.py`
Expected: PASS, clean.

- [ ] **Step 6: Commit**

```bash
git add src/neural_particle_method/suite/far_wings.py src/neural_particle_method/cli.py tests/suite/test_far_wings.py tests/test_cli_tilt.py
git commit -m "suite.far_wings: re-score stored fields on far-wing quotes; nparticle tilt farwings" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01EhbyGqFA5kYDyMwYouSgvS"
```

---

### Task 3: Far-wing table and notes

**Files:**
- Modify: `src/neural_particle_method/suite/tilt_tables.py` (`far_frame`, `_far_tex`, `tilt_tables` writes `tilt_far.tex`)
- Modify: `paper/notes_experiments.tex` (table + caption + results placeholder in section "Importance sampling")
- Test: `tests/suite/test_tilt_tables.py`

**Interfaces:**
- Consumes: metrics `far_wings_mae_bp`, `far_mae_bp/T0.5`, `far_mae_bp/T1`, `far_mae_bp/T2`, `far_wings_n`, `pooled_mae_bp` on the runs Task 2 scores.
- Produces: `far_frame(store, design, settings=FULL, budget=80_000) -> DataFrame` indexed by row label with columns `far`, `t05`, `t1`, `t2`, `n`, `pooled`; `tilt_tables` returns four paths, the fourth `tilt_far.tex`.

Rows, in order (label: source):
1. `"NW, 200 steps, untilted"` / `"NW, 200 steps, tilted"`: `tilt_cold`, algo nw, n_particles = budget, design none / `design`.
2. `"NW, 400 steps, untilted"` / `"NW, 400 steps, tilted"`: `tilt_cold_s400`, same filters.
3. `"Explicit NN, searched, untilted"` / `"... tilted"`: `tilt_cold`, algo explicit_nn_opt, promoted hash.
4. `"NW re-solve (online)"`: `suite_budget_tuned`, method nw_resolve, budget, lag surface.
5. `"Searched body + spline head, untilted"` / `"... tilted"`: `suite_budget_tuned` (promoted hash) / `tilt_online` (design), method explicit_opt_spline, lag surface.
6. `"PDE floor, 200 steps"` / `"PDE floor, 400 steps"`: `suite_pde_floor` (n_steps 200) / `suite_pde_floor_s400`, sid in SSVI_SIDS.
Each cell: per-scenario mean over seeds, then mean over scenarios, of the metric; `n` is the mean `far_wings_n`. Missing runs give NaN (`--`).

- [ ] **Step 1: Write the failing test** (append to `tests/suite/test_tilt_tables.py`; extend the fixture so the `tilt_cold` NW runs at 80k and the floor run carry `far_wings_mae_bp` 30/25, `far_mae_bp/T2` 20/15, `far_wings_n` 18, and add one `suite_pde_floor_s400` run with `far_wings_mae_bp` 12):

```python
def test_far_frame_and_table(store, tmp_path, monkeypatch):
    import neural_particle_method.suite.tilt_tables as T
    monkeypatch.setattr(T, "_promoted_hash", lambda: "abc")
    f = far_frame(store, "constant-3")
    assert f.loc["NW, 200 steps, untilted", "far"] == 30.0
    assert f.loc["NW, 200 steps, tilted", "far"] == 25.0
    assert f.loc["PDE floor, 400 steps", "far"] == 12.0
    assert np.isnan(f.loc["NW, 400 steps, tilted", "far"])
    paths = tilt_tables(store, "constant-3", out_dir=tmp_path)
    assert paths[-1].name == "tilt_far.tex"
    tex = (tmp_path / "tilt_far.tex").read_text()
    assert "NW, 200 steps, tilted & \\textbf{25} &" in tex
    assert "PDE floor, 400 steps & 12 &" in tex        # floors never bold
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/suite/test_tilt_tables.py -q -k far`
Expected: FAIL (`far_frame` missing).

- [ ] **Step 3: Implement** `far_frame` and `_far_tex` in `suite/tilt_tables.py`, following `cold_frame`/`_cold_tex` (per-scenario mean over seeds then mean over scenarios; `_render_rows` with the floor rows' prefixes starting with `"PDE floor"`: `_render_rows` excludes prefixes starting with `FLOOR_LABEL` only, so pass the floor rows with prefix `FLOOR_LABEL` replaced by a local rule: build `rows` for non-floor labels, render them, then append the two floor lines rendered with `_cell` and no bolding). Columns: `far wings & $T{=}0.5$ & $T{=}1$ & $T{=}2$ & quotes & pooled (13-strike)`; formats `{:.0f}` except `n` `{:.0f}`. `tilt_tables` adds `"tilt_far.tex": _far_tex(far_frame(store, design, settings))` as the fourth file.

- [ ] **Step 4: Notes.** In `paper/notes_experiments.tex`, section "Importance sampling", after the 400-step paragraph, add:

```latex
\subsection{Far-wing quotes}\label{sec:tilt-far}
The quoted grid stops at $|k|\approx0.5$ and its three-month wings fail inversion, so the score never sees the region where the tilt changes the leverage. Here every stored field is re-scored, without recalibration, on $21$ log-strikes from $0.45$ to $2.2$ spot ($|k|$ up to $0.8$) with an importance-sampled out-of-the-money repricer: puts below the money, calls above, on a cloud tilted by the constant design of cost $3$ with exact discrete-time weights, inverted through put-call parity; a quote whose target price is below $10^{-5}$ of spot is not scored, which removes the three-month far wings. The far-wing error is the mean absolute implied-vol error over the scored quotes with $|k|>0.25$.

\begin{table}[h]\centering\footnotesize
\caption{Far-wing implied-vol MAE in bp at the $80$k budget: pooled over maturities and at $0.5$, $1$ and $2$ years, the mean number of scored far-wing quotes, and the $13$-strike pooled MAE for reference. NW and the online rows over the $20$ SSVI scenarios; the searched network over the six study scenarios; the floors over the $20$. Bold: smallest in each column over the non-floor rows.}\label{tab:tilt-far}
\input{tables/tilt_far.tex}
\end{table}

\emph{Results.} To be written from the re-scoring.
```

Rebuild the PDF from `paper/` (twice, no `undefined`, no `^!`); the table file exists once `tilt_tables` has run against the store (run `uv run nparticle tilt tables --design constant-9`, read-only, allowed; it renders `--` until the re-score has run).

- [ ] **Step 5: Run tests, ruff, PDF; commit**

Run: `uv run pytest tests/suite/test_tilt_tables.py -q && uv run pytest -q && uv run ruff check src tests && awk 'length > 100' src/neural_particle_method/suite/tilt_tables.py`

```bash
git add src/neural_particle_method/suite/tilt_tables.py tests/suite/test_tilt_tables.py paper/notes_experiments.tex paper/notes_experiments.pdf paper/tables/tilt_far.tex
git commit -m "tilt: far-wing table and notes subsection" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01EhbyGqFA5kYDyMwYouSgvS"
```

---

## Run order (controller)

1. `uv run nparticle tilt farwings --jobs 4 2>&1 | tee results/tilt_farwings.log` (about 430 runs, 30 to 60 s each: one to two hours).
2. `uv run nparticle tilt tables --design constant-9`; results paragraph; PDF; commit; push.
