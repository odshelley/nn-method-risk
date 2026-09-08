# Baseline Comparison Set Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the published competitor estimators (GHL quartic kernel, Muguruza conditional Monte Carlo, RKHS ridge, bins, PURBF) to the benchmark harness on identical terms, with a Heston-market scenario family for acceptance tests, a per-step context hook, budget/sensitivity/acceptance drivers, and three new figures.

**Architecture:** Every baseline is an `Estimator` behind the existing `fit_predict(t, lnx, v, grid, weights=None)` interface, registered in `make_estimator` and in `bench/algos.py::ALGOS`. `calibrate_explicit` gains an optional `StepContext` passed only to estimators that declare `needs_step_context = True`. Scenarios gain a `HestonMarketSpec` family; `ScenarioSpec` and `HestonMarketSpec` both expose `local_vol()` and `target_ivs(k, mats)` so the runner never constructs a local-vol object itself. Sweeps, sensitivity curves, and acceptance runs are drivers over the existing store-backed `run_one`, in new MLflow experiments `baselines`, `sensitivity`, `acceptance`.

**Tech Stack:** Python 3.12, numpy, scipy, torch (CPU), pandas, matplotlib, mlflow (SQLite), pytest, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-09-08-baselines-design.md`. Read it first. Cards: `BASELINES.md` (arrives with Task 1's merge).

## Global Constraints

- **Bit-for-bit fidelity for existing algorithms.** After every task, `uv run pytest -q` (tiny goldens) is green and, for any task touching `calibrate/`, `simulate/`, `estimators/nn.py`, `estimators/ridge.py`, `bench/algos.py`, `bench/runner.py`, or `bench/scenarios.py`, `uv run pytest -m golden -q` (6 archived and warm-path replays) is green. The numpy draw order inside `calibrate_explicit` must not change: `rng.choice` for the mixture component before the loop (mixture only); per step k > 0 one `rng.choice(n_particles, size=min(fit_subsample, n_particles), replace=False)`, then `zb`, then `zp`.
- **Estimator contract:** `supports_weights: bool`, optional `needs_step_context: bool` (absent means False), `fit_predict(self, t, lnx, v, grid, weights=None, ctx=None)`; estimators that do not declare `needs_step_context` are called without `ctx`, exactly as today.
- **Defaults are the published tuning rule**: `GHLKernel(c=1.0)`, `ConditionalMC()`, `RKHSRidge(n_centres=100, lam=1e-9, variance=0.1)`, `Bins(n_bins=20)`, `PURBF(n_centres=40, n_neighbours=5, lam=0.2, prune=1.0)`. The paper comparison calls `make_estimator(name)` with no knobs.
- **Budgets:** `BUDGETS = (1_000, 10_000, 100_000)`; reprice unchanged at `RepriceConfig(500_000, 200)`.
- **MLflow experiments:** `bench` (untouched), `baselines`, `sensitivity`, `acceptance`. Only `tracking/store.py` imports mlflow.
- **Heston market members** (spec values, verbatim; `HestonParams` field order kappa, theta, xi, rho, v0): `li_simple` market (1.5768, 0.0484, 0.5751, -0.7, 0.1024), dynamics same with rho -0.5; `li_complex` same market, dynamics (1.0, 0.0144, 0.5751, 0.0, 0.0144); `bayer` market (2.19, 0.17023, 1.04, -0.83, 0.0045), dynamics (1.0, 0.0144, 0.5751, -0.9, 0.0144).
- **Default test suite** stays under about 30 s; slow things go behind markers `slow` and `golden`.
- **Commits** end with:

```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EhbyGqFA5kYDyMwYouSgvS
```

- Work on branch `refactor-mlflow`. Never `git add -A`; stage explicit paths. Untracked files exist in the tree (`paper/*.pdf`, `results/summary_partial.csv`, `results/sweep_log.txt`, `.claude/`, `.superpowers/`, `.coverage`, `mlruns.db`, `mlartifacts/`) and must not be committed.

## File Structure (final state, new or modified)

```
src/neural_particle_method/
  calibrate/explicit.py            StepContext dataclass; optional ctx passing
  market/heston.py                 + heston_iv
  estimators/__init__.py           make_estimator(name, seed, first_steps, later_steps, local_vol=None, s0=1.0, **knobs)
  estimators/nn.py                 NNRegressor(hidden=64) knob
  estimators/nadaraya_watson.py    + GHLKernel
  estimators/muguruza.py           ConditionalMC
  estimators/rkhs.py, bins.py      (from the papers branch, Task 1)
  estimators/purbf.py              PURBF
  bench/scenarios.py               HestonMarketSpec, heston_registry, local_vol()/target_ivs()/as_params() on both specs
  bench/algos.py                   knobs pass-through; new algos nw_ghl, muguruza, rkhs, bins, purbf
  bench/runner.py                  experiment/knobs/extra params; uses sc.local_vol()/sc.target_ivs()
  bench/sweep.py                   BUDGETS, baselines_grid(), sweep(..., experiment=)
  bench/sensitivity.py             KNOBS, sensitivity_grid(), run_sensitivity()
  bench/acceptance.py              CARDS, run_acceptance()
  bench/aggregate.py               aggregate(..., experiment=)
  figures/fig1_baselines.py, fig3_baselines.py, fig5_sensitivity.py
  cli.py                           sweep --preset baselines; sensitivity; acceptance; list shows heston sids
tests/
  bench/test_heston_market.py, test_baselines_sweep.py, test_sensitivity.py, test_acceptance_driver.py
  estimators/test_ghl.py, test_muguruza.py, test_purbf.py
  calibrate/test_step_context.py
  acceptance/__init__.py, test_cards.py           (marker slow)
  figures/test_baseline_figures.py
  golden/tiny_{nw_ghl,muguruza,rkhs,bins,purbf}.json
BASELINES.md, README.md
```

---

### Task 1: Merge the papers branch and record the archive note

**Files:**
- Merge: branch `baselines-rkhs-bins` (head 79cc526) into `refactor-mlflow`
- Modify: `README.md`

**Interfaces:**
- Produces: `estimators.rkhs.RKHSRidge(n_centres=100, lam=1e-9, variance=0.1)`, `estimators.bins.Bins(n_bins=20)`, both `supports_weights = True`, registered as `"rkhs"` and `"bins"` in `make_estimator` and present in `tests/estimators/test_contract.py::_all()`; `BASELINES.md` at repo root.

- [ ] **Step 1: Merge**

```bash
git merge --no-ff baselines-rkhs-bins -m "Merge baselines-rkhs-bins: BASELINES.md, RKHS ridge, bins estimators"
```

If `src/neural_particle_method/estimators/__init__.py` conflicts, keep both sides' names in `NAMES` and both `make_estimator` branches, `git add` it, and `git commit`. Then `git commit --amend` to append the attribution block to the merge message.

- [ ] **Step 2: Verify**

Run: `uv run pytest -q` (expected 135 passed) and `uv run ruff check .` (clean). If ruff flags anything in the merged files, fix it in a follow-up commit `chore: ruff after merge`.

- [ ] **Step 3: README archive note**

Under the `## Tests` section of `README.md` add:

```markdown
- The golden replays read archived run JSONs under `results/runs/` (gitignored). A fresh clone or
  worktree needs a copy of `results/runs/` from a checkout that has them before `uv run pytest -m golden` can pass.
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: golden replays need the untracked results/runs archive"
```

---

### Task 2: Heston market scenario family

**Files:**
- Modify: `src/neural_particle_method/market/heston.py` (add `heston_iv`)
- Modify: `src/neural_particle_method/bench/scenarios.py`
- Modify: `src/neural_particle_method/bench/algos.py` (use `sc.local_vol()`)
- Modify: `src/neural_particle_method/bench/runner.py` (use `sc.as_params()`, `sc.target_ivs()`)
- Modify: `src/neural_particle_method/cli.py` (`list` prints heston sids)
- Create: `tests/bench/test_heston_market.py`
- Modify: `tests/bench/test_scenarios.py`

**Interfaces:**
- Produces:
  - `ScenarioSpec.family = "ssvi"`, `ScenarioSpec.local_vol() -> SSVILocalVol`, `ScenarioSpec.target_ivs(k_grid, maturities) -> ndarray (len(maturities), len(k_grid))`, `ScenarioSpec.as_params() -> dict` (the `scenario.*` params the runner logs today plus `scenario.family`).
  - `HestonMarketSpec(sid, market: HestonParams, dynamics: HestonParams, s0=1.0, T=1.0, maturities=(1.0,), k_lo=-0.9, k_hi=0.9, n_k=121, t_lo=0.02, n_t=50)` frozen, `family = "heston"`, `local_vol() -> DupireSurface`, `target_ivs(k_grid, maturities)`, `as_params()`.
  - `heston_registry() -> dict[str, HestonMarketSpec]` with keys `li_simple`, `li_complex`, `bayer`; `full_registry()` includes them (29 scenarios).
  - `market.heston.heston_iv(k_grid, T, params: HestonParams, s0=1.0) -> ndarray`.

- [ ] **Step 1: Write the failing tests**

`tests/bench/test_heston_market.py`:

```python
import numpy as np
import pytest

from neural_particle_method.bench.scenarios import HestonMarketSpec, full_registry, heston_registry, make_registry
from neural_particle_method.calibrate.config import ExplicitConfig
from neural_particle_method.calibrate.explicit import calibrate_explicit
from neural_particle_method.estimators import make_estimator
from neural_particle_method.pricing.metrics import iv_metrics
from neural_particle_method.pricing.reprice import RepriceConfig, reprice_iv
from neural_particle_method.simulate.dynamics import HestonParams


def test_registry_members_and_params():
    reg = heston_registry()
    assert list(reg) == ["li_simple", "li_complex", "bayer"]
    li = reg["li_simple"]
    assert li.market == HestonParams(1.5768, 0.0484, 0.5751, -0.7, 0.1024)
    assert li.dynamics == HestonParams(1.5768, 0.0484, 0.5751, -0.5, 0.1024)
    assert reg["li_complex"].dynamics == HestonParams(1.0, 0.0144, 0.5751, 0.0, 0.0144)
    assert reg["bayer"].market == HestonParams(2.19, 0.17023, 1.04, -0.83, 0.0045)
    assert reg["bayer"].dynamics == HestonParams(1.0, 0.0144, 0.5751, -0.9, 0.0144)
    assert li.family == "heston" and make_registry()["s01"].family == "ssvi"
    assert set(heston_registry()) <= set(full_registry()) and len(full_registry()) == 29


def test_both_specs_expose_local_vol_and_targets():
    for sc in (make_registry()["s01"], heston_registry()["li_simple"]):
        lv = sc.local_vol()
        assert hasattr(lv, "sigma") and hasattr(lv, "T_grid")
        k = np.log(np.array([0.9, 1.0, 1.1]))
        ivs = sc.target_ivs(k, list(sc.maturities))
        assert ivs.shape == (len(sc.maturities), 3) and np.all(np.isfinite(ivs)) and np.all(ivs > 0)
        p = sc.as_params()
        assert p["scenario.family"] == sc.family and "scenario.dynamics.kappa" in p


def test_heston_market_targets_have_negative_skew():
    sc = heston_registry()["li_simple"]
    k = np.linspace(-0.3, 0.3, 7)
    iv = sc.target_ivs(k, [1.0])[0]
    assert iv[0] > iv[3]
    assert abs(iv[3] - np.sqrt(0.0484)) < 0.08


def _flat():
    p = HestonParams(1.5768, 0.0484, 1e-4, -0.7, 0.1024)
    return HestonMarketSpec("flat", market=p, dynamics=p)


def test_heston_market_flat_limit_gives_unit_leverage():
    flat = _flat()
    r = calibrate_explicit(flat.local_vol(), flat.dynamics, make_estimator("nw"),
                           ExplicitConfig(n_steps=10, n_particles=20_000, fit_subsample=10_000), s0=1.0, T=1.0, seed=0)
    for s in r.field[2:]:
        mid = np.abs(s.grid) < 0.25
        assert np.abs(s.L[mid] - 1.0).max() < 0.06, s.t


@pytest.mark.slow
def test_heston_market_reprices_market_within_15bp():
    flat = _flat()
    r = calibrate_explicit(flat.local_vol(), flat.dynamics, make_estimator("nw"),
                           ExplicitConfig(n_steps=50, n_particles=100_000), s0=1.0, T=1.0, seed=0)
    k = np.log(np.geomspace(0.7, 1.4, 9))
    ivs = reprice_iv(r.field, flat.dynamics, 1.0, [1.0], k, RepriceConfig(500_000, 200), seed=1)
    m = iv_metrics(ivs, flat.target_ivs(k, [1.0]), k, [1.0])
    assert m["pooled_rmse_bp"] < 15.0, m
```

In `tests/bench/test_scenarios.py` change `assert len(full_registry()) == 26` to `== 29`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/bench/test_heston_market.py -q`. Expected: ImportError on `HestonMarketSpec`.

- [ ] **Step 3: Implement**

Append to `market/heston.py`:

```python
def heston_iv(k_grid, T, params, s0=1.0):
    """Implied vol of the semi-analytic Heston call at log-moneyness k_grid and maturity T."""
    from .bs import implied_vol
    K = s0 * np.exp(np.asarray(k_grid, dtype=float))
    prices = np.atleast_1d(heston_call(K, T, params.v0, params.kappa, params.theta, params.xi, params.rho, s0))
    return np.array([implied_vol(float(p), s0, float(kk), T) for p, kk in zip(prices, K)])
```

In `bench/scenarios.py`, keep `XIS`, `RHOS`, `KAPPAS`, `quote_k_grid`, `_draw_ssvi`, `make_registry`, `fig3_registry` unchanged (the RNG draw order in `make_registry` is golden-guarded). Replace the imports and `ScenarioSpec`, and add the new class, registry, and `full_registry`:

```python
import dataclasses
from dataclasses import dataclass

import numpy as np

from ..market.dupire import DupireSurface
from ..market.heston import heston_call, heston_iv
from ..market.local_vol import SSVILocalVol
from ..market.ssvi import SSVIParams, implied_vol_ssvi, no_arb_ok
from ..simulate.dynamics import HestonParams


@dataclass(frozen=True)
class ScenarioSpec:
    sid: str
    ssvi: SSVIParams
    dynamics: HestonParams
    s0: float = 1.0
    T: float = 2.0
    maturities: tuple = (0.25, 0.5, 1.0, 2.0)
    family: str = "ssvi"

    def local_vol(self):
        return SSVILocalVol(self.ssvi, self.s0, T_max=self.T)

    def target_ivs(self, k_grid, maturities):
        return np.stack([implied_vol_ssvi(self.ssvi, np.asarray(k_grid), t) for t in maturities])

    def as_params(self):
        p = {f"scenario.ssvi.{k}": v for k, v in dataclasses.asdict(self.ssvi).items()}
        p.update({f"scenario.dynamics.{k}": v for k, v in self.dynamics.to_dict().items()})
        p.update({"scenario.family": self.family, "scenario.s0": self.s0, "scenario.T": self.T,
                  "scenario.maturities": str(list(self.maturities))})
        return p


@dataclass(frozen=True)
class HestonMarketSpec:
    """Target surface generated by a Heston market; the calibrated model has its own dynamics."""
    sid: str
    market: HestonParams
    dynamics: HestonParams
    s0: float = 1.0
    T: float = 1.0
    maturities: tuple = (1.0,)
    k_lo: float = -0.9
    k_hi: float = 0.9
    n_k: int = 121
    t_lo: float = 0.02
    n_t: int = 50
    family: str = "heston"

    def local_vol(self):
        m = self.market
        T_grid = np.linspace(self.t_lo, self.T, self.n_t)
        k_grid = np.linspace(self.k_lo, self.k_hi, self.n_k)

        def price_fn(K, T):
            return heston_call(K, T, m.v0, m.kappa, m.theta, m.xi, m.rho, self.s0)

        return DupireSurface.from_price_fn(price_fn, self.s0, T_grid, k_grid)

    def target_ivs(self, k_grid, maturities):
        return np.stack([heston_iv(k_grid, t, self.market, self.s0) for t in maturities])

    def as_params(self):
        p = {f"scenario.market.{k}": v for k, v in self.market.to_dict().items()}
        p.update({f"scenario.dynamics.{k}": v for k, v in self.dynamics.to_dict().items()})
        p.update({"scenario.family": self.family, "scenario.s0": self.s0, "scenario.T": self.T,
                  "scenario.maturities": str(list(self.maturities))})
        return p


def heston_registry():
    li_market = HestonParams(1.5768, 0.0484, 0.5751, -0.7, 0.1024)
    return {
        "li_simple": HestonMarketSpec("li_simple", li_market, HestonParams(1.5768, 0.0484, 0.5751, -0.5, 0.1024)),
        "li_complex": HestonMarketSpec("li_complex", li_market, HestonParams(1.0, 0.0144, 0.5751, 0.0, 0.0144)),
        "bayer": HestonMarketSpec("bayer", HestonParams(2.19, 0.17023, 1.04, -0.83, 0.0045),
                                  HestonParams(1.0, 0.0144, 0.5751, -0.9, 0.0144)),
    }


def full_registry():
    return {**make_registry(), **fig3_registry(), **heston_registry()}
```

`DupireSurface.from_price_fn` calls `price_fn(K, T)` with an array `K`; `heston_call` accepts arrays. Failed inversions in the deep wings at t = 0.02 are filled by strike interpolation inside `from_price_fn`.

`bench/algos.py`: replace both `lv = SSVILocalVol(sc.ssvi, sc.s0, T_max=sc.T)` with `lv = sc.local_vol()`; remove the `SSVILocalVol` import. `bench/runner.py`: replace the `**_prefixed("scenario.ssvi", ...)`, `**_prefixed("scenario.dynamics", ...)`, `"scenario.s0"`, `"scenario.T"`, `"scenario.maturities"` entries with `**sc.as_params()`, and replace `iv_target = target_ivs(sc.ssvi, k, mats, reprice.n_steps)` with `iv_target = sc.target_ivs(k, snap_times(mats, reprice.n_steps))` (import `snap_times` from `..pricing.reprice`; drop the now-unused `target_ivs` import). The target is evaluated at the snapped times exactly as `target_ivs` did. `cli.py` `list`: for specs with `family == "heston"` print `f"  {sid}: heston market xi={sc.market.xi} rho={sc.market.rho}; dynamics rho={sc.dynamics.rho}"`, else the existing line.

- [ ] **Step 4: Run, goldens, commit**

Run: `uv run pytest -q`; `uv run pytest tests/bench/test_heston_market.py -m slow -q` once (about a minute); `uv run pytest -m golden -q` (scenarios, algos, runner changed; the SSVI path must be unchanged).

```bash
git add src/neural_particle_method/market/heston.py src/neural_particle_method/bench src/neural_particle_method/cli.py tests/bench
git commit -m "feat: Heston-market scenario family; specs expose local_vol(), target_ivs(), as_params()"
```

---

### Task 3: Step-context hook in `calibrate_explicit`

**Files:**
- Modify: `src/neural_particle_method/calibrate/explicit.py`
- Modify: `src/neural_particle_method/estimators/base.py`
- Create: `tests/calibrate/test_step_context.py`

**Interfaces:**
- Produces: `calibrate.explicit.StepContext(lnx_prev, v_prev, L_p, zb, zp, theta_p, dt, params)` frozen dataclass with `__getitem__(idx)` returning a subsampled context (arrays indexed, scalars kept); `calibrate_explicit` passes `ctx=` only when `getattr(estimator, "needs_step_context", False)` is true.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from neural_particle_method.calibrate.config import ExplicitConfig
from neural_particle_method.calibrate.explicit import StepContext, calibrate_explicit
from neural_particle_method.simulate.dynamics import HestonParams
from neural_particle_method.simulate.stepper import heston_step

DYN = HestonParams(2.0, 0.04, 0.3, -0.5, 0.04)


class FlatDupire:
    T_grid = np.array([0.004, 1.0])

    def sigma(self, t, x, s0=1.0):
        return np.full(np.shape(np.log(np.asarray(x))), 0.2)


class Recorder:
    supports_weights = True
    needs_step_context = True

    def __init__(self):
        self.seen = []

    def fit_predict(self, t, lnx, v, grid, weights=None, ctx=None):
        self.seen.append((t, lnx.copy(), ctx))
        return np.full(len(grid), 0.04)


class Plain:
    supports_weights = True

    def __init__(self):
        self.calls = 0

    def fit_predict(self, t, lnx, v, grid, weights=None):   # no ctx parameter at all
        self.calls += 1
        return np.full(len(grid), 0.04)


def test_context_delivered_and_consistent():
    rec = Recorder()
    calibrate_explicit(FlatDupire(), DYN, rec, ExplicitConfig(n_steps=5, n_particles=3_000, fit_subsample=1_000), T=0.5, seed=1)
    assert len(rec.seen) == 4
    for t, lnx, ctx in rec.seen:
        assert isinstance(ctx, StepContext) and len(lnx) == 1_000
        for a in (ctx.lnx_prev, ctx.v_prev, ctx.L_p, ctx.zb, ctx.zp):
            assert a.shape == (1_000,)
        assert ctx.theta_p is None and np.isclose(ctx.dt, 0.1) and ctx.params == DYN
        assert np.all(ctx.v_prev >= 0)
        # the recorded previous state and normals reproduce the subsampled current cloud exactly
        lnx1, _ = heston_step(ctx.lnx_prev, ctx.v_prev, ctx.L_p, ctx.zb, ctx.zp, DYN, ctx.dt, np.sqrt(ctx.dt))
        np.testing.assert_array_equal(lnx1, lnx)


def test_plain_estimator_never_receives_ctx():
    est = Plain()
    calibrate_explicit(FlatDupire(), DYN, est, ExplicitConfig(n_steps=5, n_particles=3_000, fit_subsample=1_000), T=0.5, seed=1)
    assert est.calls == 4


def test_getitem_subsamples_arrays_only():
    c = StepContext(np.arange(5.0), np.arange(5.0), np.ones(5), np.zeros(5), np.zeros(5), None, 0.1, DYN)
    s = c[np.array([0, 2])]
    assert s.lnx_prev.tolist() == [0.0, 2.0] and s.dt == 0.1 and s.params is DYN and s.theta_p is None
```

Note on the exact-reproduction assertion: `heston_step` is deterministic given its inputs, and `v_prev` is `np.maximum(v, 0.0)`, which the stepper recomputes identically, so `lnx1` equals the harness's post-step `lnx` bit for bit.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/calibrate/test_step_context.py -q`. Expected: ImportError on `StepContext`.

- [ ] **Step 3: Implement**

In `calibrate/explicit.py`, after the imports:

```python
@dataclass(frozen=True)
class StepContext:
    """What an estimator may know about the step that produced the current cloud (see spec)."""
    lnx_prev: np.ndarray
    v_prev: np.ndarray
    L_p: np.ndarray
    zb: np.ndarray
    zp: np.ndarray
    theta_p: np.ndarray | None
    dt: float
    params: HestonParams

    def __getitem__(self, idx):
        return StepContext(self.lnx_prev[idx], self.v_prev[idx], self.L_p[idx], self.zb[idx], self.zp[idx],
                           None if self.theta_p is None else self.theta_p[idx], self.dt, self.params)
```

In `calibrate_explicit`, before the loop: `wants_ctx = bool(getattr(estimator, "needs_step_context", False))` and `ctx = None`. Replace the fit line:

```python
            if wants_ctx:
                f_grid = estimator.fit_predict(t, lnx[idx], v[idx], grid, weights=wi, ctx=ctx[idx])
            else:
                f_grid = estimator.fit_predict(t, lnx[idx], v[idx], grid, weights=wi)
```

Between `zp = rng.standard_normal(n_particles)` and `lnx, v = heston_step(...)`:

```python
        if wants_ctx:
            ctx = StepContext(lnx, np.maximum(v, 0.0), L_p, zb, zp, theta_p, dt, hp)
```

`heston_step` rebinds `lnx` and `v`, so the context keeps the pre-step arrays without copying. No draw moves.

`estimators/base.py`: signature `fit_predict(self, t, lnx, v, grid, weights=None, ctx=None)` with the docstring line "`ctx` is passed only when the estimator sets `needs_step_context = True`."

- [ ] **Step 4: Run, goldens, commit**

Run: `uv run pytest -q`; `uv run pytest -m golden -q` (6 passed).

```bash
git add src/neural_particle_method/calibrate/explicit.py src/neural_particle_method/estimators/base.py tests/calibrate/test_step_context.py
git commit -m "feat: optional StepContext delivered to estimators that declare needs_step_context"
```

---

### Task 4: `make_estimator` knobs and the GHL quartic kernel

**Files:**
- Modify: `src/neural_particle_method/estimators/__init__.py`
- Modify: `src/neural_particle_method/estimators/nn.py` (`hidden` knob)
- Modify: `src/neural_particle_method/estimators/nadaraya_watson.py`
- Create: `tests/estimators/test_ghl.py`
- Modify: `tests/estimators/test_contract.py`

**Interfaces:**
- Produces:
  - `make_estimator(name, seed=0, first_steps=400, later_steps=120, local_vol=None, s0=1.0, **knobs)`: `knobs` forwarded to the constructor; `local_vol` and `s0` consumed by estimators that need the target surface. `NAMES` gains `"nw_ghl"`.
  - `NNRegressor(seed=0, first_steps=400, later_steps=120, hidden=64)`.
  - `GHLKernel(c=1.0, kernel="quartic", local_vol=None, s0=1.0)`, `supports_weights = True`, method `bandwidth(t, n, v) -> float`.

- [ ] **Step 1: Write the failing tests**

`tests/estimators/test_ghl.py`:

```python
import numpy as np
import pytest

from neural_particle_method.estimators import NAMES, make_estimator
from neural_particle_method.estimators.nadaraya_watson import GHLKernel


class FlatLV:
    T_grid = np.array([0.004, 1.0])

    def sigma(self, t, x, s0=1.0):
        return np.full(np.shape(np.asarray(x, dtype=float)), 0.2)


def _cloud(n=5_000, seed=0):
    rng = np.random.default_rng(seed)
    lnx = rng.normal(0.0, 0.2, n)
    return lnx, 0.04 + 0.02 * lnx


def test_registered_with_knobs_and_context():
    assert "nw_ghl" in NAMES
    est = make_estimator("nw_ghl", local_vol=FlatLV(), s0=1.0, c=2.0)
    assert isinstance(est, GHLKernel) and est.c == 2.0 and est.supports_weights


def test_bandwidth_rule():
    est = GHLKernel(local_vol=FlatLV(), s0=1.0)
    n = 10_000
    assert np.isclose(est.bandwidth(0.5, n, v=None), 1.5 * 0.2 * np.sqrt(0.5) * n ** (-0.2))
    assert np.isclose(est.bandwidth(0.1, n, v=None), 1.5 * 0.2 * np.sqrt(0.25) * n ** (-0.2))
    proxy = GHLKernel()
    assert np.isclose(proxy.bandwidth(0.5, n, v=np.full(n, 0.04)), 1.5 * 0.2 * np.sqrt(0.5) * n ** (-0.2))


def test_constant_and_linear_recovery():
    lnx, v = _cloud()
    grid = np.linspace(-0.3, 0.3, 13)
    est = GHLKernel(local_vol=FlatLV())
    assert np.allclose(est.fit_predict(0.5, lnx, np.full_like(v, 0.05), grid), 0.05)
    f = est.fit_predict(0.5, lnx, v, grid)
    assert np.abs(f - (0.04 + 0.02 * grid)).max() < 2e-3


def test_empty_support_is_interpolated_not_nan():
    lnx, v = _cloud(n=200)
    grid = np.linspace(-2.0, 2.0, 21)
    f = GHLKernel(c=0.2, local_vol=FlatLV()).fit_predict(0.5, lnx, v, grid)
    assert np.all(np.isfinite(f))


def test_weights_are_applied():
    lnx, v = _cloud()
    est = GHLKernel(local_vol=FlatLV())
    w = (lnx > 0).astype(float)
    hi = est.fit_predict(0.5, lnx, v, np.array([0.0]), weights=w)[0]
    lo = est.fit_predict(0.5, lnx, v, np.array([0.0]), weights=1.0 - w)[0]
    assert hi > lo


def test_gaussian_variant_and_bad_kernel():
    lnx, v = _cloud()
    g = GHLKernel(kernel="gaussian", local_vol=FlatLV()).fit_predict(0.5, lnx, v, np.array([0.0, 0.1]))
    q = GHLKernel(kernel="quartic", local_vol=FlatLV()).fit_predict(0.5, lnx, v, np.array([0.0, 0.1]))
    assert np.all(np.isfinite(g)) and not np.allclose(g, q)
    with pytest.raises(ValueError):
        GHLKernel(kernel="triangle")


def test_nn_hidden_knob():
    est = make_estimator("nn", seed=0, hidden=16)
    assert est.net.head.in_features == 16
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/estimators/test_ghl.py -q`. Expected: ImportError on `GHLKernel`.

- [ ] **Step 3: Implement**

Append to `estimators/nadaraya_watson.py`:

```python
def quartic_kernel(u):
    return np.where(np.abs(u) <= 1.0, 15.0 / 16.0 * (1.0 - u ** 2) ** 2, 0.0)


class GHLKernel:
    """Nadaraya-Watson in spot with the GHL rule-of-thumb bandwidth as stated by Cozma et al. (2019):
    h(t) = c * 1.5 * S0 * sigma_LV(S0, t) * sqrt(max(t, 0.25)) * N^(-1/5). Quartic kernel by default.
    Without a local-vol object the ATM vol is proxied by sqrt(mean(v)) of the cloud."""
    supports_weights = True

    def __init__(self, c=1.0, kernel="quartic", local_vol=None, s0=1.0):
        if kernel not in ("quartic", "gaussian"):
            raise ValueError(f"unknown kernel {kernel!r}")
        self.c, self.kernel, self.local_vol, self.s0 = c, kernel, local_vol, s0

    def sigma_atm(self, t, v):
        if self.local_vol is not None:
            s = self.local_vol.sigma(max(t, self.local_vol.T_grid[0]), self.s0, self.s0)
            return float(np.asarray(s, dtype=float).reshape(-1)[0])
        return float(np.sqrt(np.mean(v)))

    def bandwidth(self, t, n, v):
        return self.c * 1.5 * self.s0 * self.sigma_atm(t, v) * np.sqrt(max(t, 0.25)) * n ** (-0.2)

    def fit_predict(self, t, lnx, v, grid, weights=None, ctx=None):
        S, K = np.exp(lnx), np.exp(grid)
        h = self.bandwidth(t, len(lnx), v)
        u = (K[:, None] - S[None, :]) / h
        ker = quartic_kernel(u) if self.kernel == "quartic" else np.exp(-0.5 * u ** 2)
        if weights is not None:
            ker = ker * weights[None, :]
        den, num = ker.sum(axis=1), ker @ v
        ok = den > 0.0
        f = np.empty(len(grid))
        f[ok] = num[ok] / den[ok]
        if not ok.all():
            if not ok.any():
                f[:] = np.average(v, weights=weights)
            else:
                f[~ok] = np.interp(grid[~ok], grid[ok], f[ok])
        return f
```

`estimators/nn.py`: `NNRegressor.__init__(self, seed=0, first_steps=400, later_steps=120, hidden=64)` and `self.net = SliceNet(hidden)` (the `torch.manual_seed(seed)` call stays before it). `estimators/__init__.py`: import `GHLKernel`; `NAMES = ("nn", "nw", "nw_ghl", "ridge", "spline", "bins", "rkhs")`; and

```python
def make_estimator(name, seed=0, first_steps=400, later_steps=120, local_vol=None, s0=1.0, **knobs):
    """Build an estimator by registry name. `knobs` go to the constructor (sensitivity sweeps and
    acceptance overrides); `local_vol` and `s0` are given to estimators that need the target surface."""
    if name == "nn":
        return NNRegressor(seed=seed, first_steps=first_steps, later_steps=later_steps, **knobs)
    if name == "nw":
        return NadarayaWatson(**knobs)
    if name == "nw_ghl":
        return GHLKernel(local_vol=local_vol, s0=s0, **knobs)
    if name == "ridge":
        return SliceRidge(seed=seed, body_steps=first_steps, **knobs)
    if name == "spline":
        return PSpline(**knobs)
    if name == "bins":
        return Bins(**knobs)
    if name == "rkhs":
        return RKHSRidge(**knobs)
    raise KeyError(f"unknown estimator {name!r}; choose from {NAMES}")
```

Add `make_estimator("nw_ghl")` to `tests/estimators/test_contract.py::_all()` (proxy-bandwidth path) and `"GHLKernel": {}` handling is automatic in `_ctor_kwargs` (it returns `{}` for unknown names).

- [ ] **Step 4: Run, goldens (nn.py touched), commit**

Run: `uv run pytest -q`; `uv run pytest -m golden -q`.

```bash
git add src/neural_particle_method/estimators tests/estimators
git commit -m "feat: GHL quartic-kernel estimator; make_estimator forwards knobs and the target surface"
```

---

### Task 5: Muguruza conditional Monte Carlo

**Files:**
- Create: `src/neural_particle_method/estimators/muguruza.py`
- Modify: `src/neural_particle_method/estimators/__init__.py`
- Create: `tests/estimators/test_muguruza.py`

**Interfaces:**
- Consumes: `StepContext` (Task 3).
- Produces: `ConditionalMC()`, `supports_weights = True`, `needs_step_context = True`, registered as `"muguruza"`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from neural_particle_method.calibrate.config import ExplicitConfig
from neural_particle_method.calibrate.explicit import StepContext, calibrate_explicit
from neural_particle_method.estimators import make_estimator
from neural_particle_method.estimators.muguruza import ConditionalMC
from neural_particle_method.simulate.dynamics import HestonParams
from neural_particle_method.simulate.stepper import heston_step

DYN = HestonParams(2.0, 0.04, 0.3, -0.5, 0.04)


def _ctx(lnx_prev, v_prev, L, zb, zp, dt=0.1, params=DYN, theta_p=None):
    return StepContext(np.asarray(lnx_prev, float), np.asarray(v_prev, float), np.asarray(L, float),
                       np.asarray(zb, float), np.asarray(zp, float), theta_p, dt, params)


def test_hand_computed_three_particles():
    ctx = _ctx([0.0, 0.1, -0.1], [0.04, 0.05, 0.03], [1.0, 1.2, 0.8], [0.5, -1.0, 0.2], [0.0, 0.0, 0.0])
    lnx_now, _ = heston_step(ctx.lnx_prev, ctx.v_prev, ctx.L_p, ctx.zb, ctx.zp, DYN, 0.1, np.sqrt(0.1))
    v_now = np.array([0.041, 0.052, 0.029])
    x = np.array([0.02])
    rho = DYN.rho
    mu = ctx.lnx_prev - 0.5 * ctx.L_p ** 2 * ctx.v_prev * 0.1 + ctx.L_p * np.sqrt(ctx.v_prev) * rho * ctx.zb * np.sqrt(0.1)
    s2 = (1 - rho ** 2) * ctx.L_p ** 2 * ctx.v_prev * 0.1
    phi = np.exp(-0.5 * (x[0] - mu) ** 2 / s2) / np.sqrt(s2)
    expected = (phi * v_now).sum() / phi.sum()
    got = ConditionalMC().fit_predict(0.1, lnx_now, v_now, x, ctx=ctx)
    assert abs(got[0] - expected) < 1e-12


def test_zero_variance_particles_are_dropped():
    ctx = _ctx([0.0, 0.0], [0.04, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0])
    f = ConditionalMC().fit_predict(0.1, np.zeros(2), np.array([0.05, 0.9]), np.array([0.0]), ctx=ctx)
    assert abs(f[0] - 0.05) < 1e-12


def test_weights_and_tilt_enter():
    ctx = _ctx([0.0, 0.0], [0.04, 0.04], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0])
    est = ConditionalMC()
    f = est.fit_predict(0.1, np.zeros(2), np.array([0.02, 0.06]), np.array([0.0]), weights=np.array([3.0, 1.0]), ctx=ctx)
    assert abs(f[0] - 0.03) < 1e-12
    tilted = _ctx([0.0, 0.0], [0.04, 0.04], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], theta_p=np.array([0.0, 5.0]))
    g = est.fit_predict(0.1, np.zeros(2), np.array([0.02, 0.06]), np.array([0.0]), ctx=tilted)
    assert g[0] < 0.04


def test_requires_ctx():
    with pytest.raises(ValueError, match="StepContext"):
        ConditionalMC().fit_predict(0.1, np.zeros(3), np.zeros(3), np.zeros(2))


class FlatDupire:
    T_grid = np.array([0.004, 1.0])

    def sigma(self, t, x, s0=1.0):
        return np.full(np.shape(np.log(np.asarray(x))), 0.2)


def test_end_to_end_flat_recovery_low_vov():
    dyn = HestonParams(2.0, 0.04, 0.05, -0.5, 0.04)
    r = calibrate_explicit(FlatDupire(), dyn, make_estimator("muguruza"),
                           ExplicitConfig(n_steps=8, n_particles=20_000, fit_subsample=20_000), T=0.5, seed=3)
    s = r.field[-1]
    mid = np.abs(s.grid) < 0.3
    assert np.abs(s.L[mid] - 1.0).max() < 0.15
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/estimators/test_muguruza.py -q`. Expected: ImportError.

- [ ] **Step 3: Implement**

`estimators/muguruza.py`:

```python
"""Conditional Monte Carlo (Muguruza 2019, Corollary 4.1): kernel-free and bandwidth-free.

Each particle contributes with its own one-step conditional density of log-spot given the vol
path, which under the harness's Euler step with frozen leverage is Gaussian with
    mean     = lnx_prev + (-1/2 L^2 v + L sqrt(v) theta_p) dt + L sqrt(v) rho zb sqrt(dt)
    variance = (1 - rho^2) L^2 v dt.
"""
import numpy as np


class ConditionalMC:
    supports_weights = True
    needs_step_context = True

    def fit_predict(self, t, lnx, v, grid, weights=None, ctx=None):
        if ctx is None:
            raise ValueError("ConditionalMC needs a StepContext (delivered by calibrate_explicit)")
        rho, dt = ctx.params.rho, ctx.dt
        vol = ctx.L_p * np.sqrt(ctx.v_prev)
        drift = -0.5 * ctx.L_p ** 2 * ctx.v_prev
        if ctx.theta_p is not None:
            drift = drift + vol * ctx.theta_p
        mu = ctx.lnx_prev + drift * dt + vol * rho * ctx.zb * np.sqrt(dt)
        s2 = (1.0 - rho ** 2) * vol ** 2 * dt
        keep = s2 > 1e-18
        mu, s2, vv = mu[keep], s2[keep], v[keep]
        w = np.ones(len(vv)) if weights is None else weights[keep]
        phi = np.exp(-0.5 * (grid[:, None] - mu[None, :]) ** 2 / s2[None, :]) / np.sqrt(s2)[None, :] * w[None, :]
        den, num = phi.sum(axis=1), phi @ vv
        fallback = float(np.average(vv, weights=w))
        return np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), fallback)
```

Register in `estimators/__init__.py`: `from .muguruza import ConditionalMC`; add `"muguruza"` to `NAMES`; `if name == "muguruza": return ConditionalMC(**knobs)`; add to `__all__`. Do not add it to `test_contract._all()` (it needs a context); its own tests cover the contract.

- [ ] **Step 4: Run and commit**

Run: `uv run pytest -q` (no golden run needed: no existing algorithm touched).

```bash
git add src/neural_particle_method/estimators tests/estimators/test_muguruza.py
git commit -m "feat: Muguruza conditional Monte Carlo estimator"
```

---

### Task 6: PURBF estimator (Hakala 2019)

**Files:**
- Create: `src/neural_particle_method/estimators/purbf.py`
- Modify: `src/neural_particle_method/estimators/__init__.py`
- Create: `tests/estimators/test_purbf.py`
- Modify: `tests/estimators/test_contract.py`

**Interfaces:**
- Produces: `PURBF(n_centres=40, n_neighbours=5, lam=0.2, prune=1.0, seed=0)`, `supports_weights = True`, registered as `"purbf"`. Source facts (transcribed by the papers session from the typeset PDF): C = 40 centres in all his experiments at N = 2 048 particles; local width from the 5 nearest centres (his preferred variant); ridge regulariser 0.2 (Figure 6 legend, section 4.1); pruning criterion `min_i |c_i - c_j| / h_j <= Theta` with Theta never stated (treated as the free knob `prune`, default 1.0, swept over {0.5, 1, 2}); his printed solution has `-lambda`, inconsistent with his own loss; implement `+lambda`. Global rule-of-thumb width `h = (4 sigma^5 / (3 n))^(1/5)` used only when fewer than `n_neighbours + 1` centres survive pruning.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from neural_particle_method.estimators import NAMES, make_estimator
from neural_particle_method.estimators.purbf import PURBF


def _cloud(n=4_000, seed=0):
    rng = np.random.default_rng(seed)
    lnx = rng.uniform(-0.6, 0.6, n)
    return lnx, 0.04 + 0.1 * lnx ** 2


def test_registered_defaults():
    assert "purbf" in NAMES
    est = make_estimator("purbf")
    assert isinstance(est, PURBF)
    assert (est.n_centres, est.n_neighbours, est.lam, est.prune) == (40, 5, 0.2, 1.0)


def test_centres_include_extremes_and_pruning_is_monotone():
    lnx, _ = _cloud()
    c, h = PURBF(n_centres=40, prune=1.0, seed=1).centres_and_widths(lnx)
    assert c.min() == lnx.min() and c.max() == lnx.max()
    assert 2 <= len(c) <= 40 and len(h) == len(c) and np.all(h > 0) and np.all(np.diff(c) > 0)
    n_loose = len(PURBF(n_centres=40, prune=0.5, seed=1).centres_and_widths(lnx)[0])
    n_tight = len(PURBF(n_centres=40, prune=2.0, seed=1).centres_and_widths(lnx)[0])
    assert n_tight <= len(c) <= n_loose


def test_recovers_quadratic_without_seams():
    lnx, v = _cloud()
    grid = np.linspace(-0.5, 0.5, 101)
    f = PURBF(lam=1e-6, seed=0).fit_predict(0.3, lnx, v, grid)
    assert np.abs(f - (0.04 + 0.1 * grid ** 2)).max() < 5e-3     # blended local constants: O(h^2) bias
    assert np.abs(np.diff(f, 2)).max() < 2e-3                    # no kink at centre boundaries


def test_constant_recovery_with_default_lambda():
    lnx, _ = _cloud()
    f = PURBF().fit_predict(0.3, lnx, np.full_like(lnx, 0.05), np.linspace(-0.5, 0.5, 11))
    assert np.abs(f - 0.05).max() < 5e-4


def test_weights_and_determinism():
    lnx, v = _cloud()
    grid = np.array([0.0])
    a = PURBF(seed=3).fit_predict(0.3, lnx, v, grid)
    b = PURBF(seed=3).fit_predict(0.3, lnx, v, grid)
    assert a == b
    w = (lnx > 0).astype(float)
    hi = PURBF(seed=3).fit_predict(0.3, lnx, v + 0.5 * (lnx > 0), grid, weights=w)[0]
    lo = PURBF(seed=3).fit_predict(0.3, lnx, v + 0.5 * (lnx > 0), grid, weights=1.0 - w)[0]
    assert hi > lo
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/estimators/test_purbf.py -q`. Expected: ImportError.

- [ ] **Step 3: Implement**

`estimators/purbf.py`:

```python
"""Partition-of-unity radial basis functions (Hakala 2019, Frontiers in AI 2:4).

f(x) = sum_j w_j psi_j(x),  psi_j(x) = K_{h_j}(x - c_j) / sum_k K_{h_k}(x - c_k),  K_h(r) = exp(-r^2 / (2 h^2)).
Centres: cloud min and max plus a random subset, pruned so no centre lies within prune * h_j of an
already kept one; h_j = mean distance to the n_neighbours nearest centres. Weights from ridge
normal equations (A^T W A + lam I) w = A^T W y with A_ij = psi_j(x_i). Defaults are his preferred
configuration: 40 centres, 5-NN widths, regulariser 0.2; the pruning constant is never stated in
the paper and is a free knob here.
"""
import numpy as np


class PURBF:
    supports_weights = True

    def __init__(self, n_centres=40, n_neighbours=5, lam=0.2, prune=1.0, seed=0):
        self.n_centres, self.n_neighbours, self.lam, self.prune, self.seed = n_centres, n_neighbours, lam, prune, seed

    def centres_and_widths(self, lnx):
        rng = np.random.default_rng(self.seed)
        lo, hi = float(lnx.min()), float(lnx.max())
        n_rand = max(self.n_centres - 2, 0)
        cand = np.concatenate([[lo, hi], rng.choice(lnx, size=min(n_rand, len(lnx)), replace=False)])
        cand = np.unique(cand)                       # sorted; lo first, hi last
        # widths from the candidate set, then greedy pruning in sorted order; the extremes always stay
        h_all = self._widths(cand)
        kept, kept_h = [], []
        for c, h in zip(cand, h_all):
            if c in (lo, hi) or not kept or np.abs(np.array(kept) - c).min() / h > self.prune:
                kept.append(c); kept_h.append(h)
        c, h = np.array(kept), np.array(kept_h)
        if len(c) <= self.n_neighbours:
            h = np.full(len(c), self._global_width(lnx))
        return c, h

    def _widths(self, c):
        if len(c) <= 1:
            return np.ones(len(c))
        d = np.abs(c[:, None] - c[None, :])
        d[np.eye(len(c), dtype=bool)] = np.inf
        k = min(self.n_neighbours, len(c) - 1)
        return np.sort(d, axis=1)[:, :k].mean(axis=1)

    @staticmethod
    def _global_width(lnx):
        return (4.0 * np.std(lnx) ** 5 / (3.0 * len(lnx))) ** 0.2

    def _basis(self, x, c, h):
        K = np.exp(-0.5 * ((x[:, None] - c[None, :]) / h[None, :]) ** 2)
        return K / np.clip(K.sum(axis=1, keepdims=True), 1e-300, None)

    def fit_predict(self, t, lnx, v, grid, weights=None, ctx=None):
        c, h = self.centres_and_widths(lnx)
        A = self._basis(lnx, c, h)
        wv = np.ones(len(lnx)) if weights is None else weights
        lhs = A.T @ (wv[:, None] * A) + self.lam * np.eye(len(c))
        rhs = A.T @ (wv * v)
        w = np.linalg.solve(lhs, rhs)
        return self._basis(grid, c, h) @ w
```

Register: import `PURBF`; add `"purbf"` to `NAMES`; `if name == "purbf": return PURBF(**knobs)`; `__all__`. Add `make_estimator("purbf")` to `test_contract._all()`.

- [ ] **Step 4: Run and commit**

Run: `uv run pytest -q`.

```bash
git add src/neural_particle_method/estimators tests/estimators
git commit -m "feat: partition-of-unity RBF estimator (Hakala 2019)"
```

---

### Task 7: Register the baselines as algorithms; budgets; `baselines` sweep preset; knobs and extra params through the runner; tiny goldens

**Files:**
- Modify: `src/neural_particle_method/bench/algos.py`
- Modify: `src/neural_particle_method/bench/runner.py`
- Modify: `src/neural_particle_method/bench/sweep.py`
- Modify: `src/neural_particle_method/cli.py`
- Modify: `tests/golden/make_tiny_goldens.py`; create `tests/golden/tiny_{nw_ghl,muguruza,rkhs,bins,purbf}.json`
- Create: `tests/bench/test_baselines_sweep.py`
- Modify: `tests/bench/test_sweep.py` (`paper_grid` count uses the seven paper algos, not `ALGOS`)

**Interfaces:**
- Produces:
  - `bench.algos.PAPER_ALGOS = ("nw", "explicit_nn", "ridge", "explicit_nn_is", "implicit_nn", "spline", "implicit_ridge")` (the seven the archived paper sweep used) and `BASELINE_ALGOS = ("nw_ghl", "muguruza", "rkhs", "bins", "purbf")`; `ALGOS` maps all twelve. `run_algo(name, scenario, n_particles, seed, explicit=..., implicit=..., knobs=None)`; the explicit contenders forward `knobs` to `make_estimator(..., local_vol=lv, s0=sc.s0, **knobs)`.
  - `bench.runner.run_one(store, sid, algo, n_particles, seed, explicit=..., implicit=..., reprice=..., *, experiment="bench", knobs=None, extra_key=None)`: the resumability key is `run_key(...) | extra_key`; params add `budget` (= n_particles), `estimator.<knob>` for each knob, and `extra_key` entries; metrics add `budget` and, if `pde_reference` holds a FINISHED run with param `sid` equal to this scenario and an artifact `leverage.json`, `lev_rmse` (RMSE over all slices and grid points of `L_method - L_ref`, both resampled to `DEFAULT_GRID`) and `lev_rmse/T<t>` per slice.
  - `bench.sweep.BUDGETS = (1_000, 10_000, 100_000)`, `baselines_grid() -> list[(sid, algo, n, seed)]` (SSVI s01..s20 x all twelve algos x BUDGETS x seeds 0,1,2, then the six fig3 scenarios x all twelve x (10_000, 100_000) x seeds 0,1,2), `sweep(store, jobs, n_jobs=1, explicit=..., implicit=..., reprice=..., experiment="bench")`.
  - CLI: `nparticle sweep --preset {paper,baselines}`.

- [ ] **Step 1: Write the failing tests**

`tests/bench/test_baselines_sweep.py`:

```python
import json

import pytest

from neural_particle_method.bench.algos import ALGOS, BASELINE_ALGOS, PAPER_ALGOS, run_algo
from neural_particle_method.bench.runner import run_one
from neural_particle_method.bench.scenarios import heston_registry, make_registry
from neural_particle_method.bench.sweep import BUDGETS, baselines_grid, sweep
from neural_particle_method.pricing.reprice import RepriceConfig
from neural_particle_method.simulate.leverage import LeverageField
from neural_particle_method.tracking.store import Store
from tests.conftest import TINY_EXPLICIT, TINY_IMPLICIT, TINY_N, TINY_REPRICE_N, TINY_REPRICE_STEPS

TINY_REPRICE = RepriceConfig(TINY_REPRICE_N, TINY_REPRICE_STEPS)


def test_registries():
    assert set(PAPER_ALGOS) | set(BASELINE_ALGOS) == set(ALGOS) and len(ALGOS) == 12
    assert BUDGETS == (1_000, 10_000, 100_000)
    jobs = baselines_grid()
    assert len(jobs) == 20 * 12 * 3 * 3 + 6 * 12 * 2 * 3
    assert jobs[0] == ("s01", "nw", 1_000, 0)


@pytest.mark.parametrize("name", BASELINE_ALGOS)
def test_each_baseline_runs_on_both_families(name):
    for sc in (make_registry()["s01"], heston_registry()["li_simple"]):
        res = run_algo(name, sc, 3_000, 0, TINY_EXPLICIT, TINY_IMPLICIT)
        assert len(res.field) == 6 and res.timings["total_s"] > 0


def test_knobs_reach_the_estimator_and_are_logged(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    a = run_one(store, "s01", "bins", TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT, TINY_REPRICE,
                experiment="sensitivity", knobs={"n_bins": 5}, extra_key={"knob_name": "n_bins", "knob_value": 5})
    b = run_one(store, "s01", "bins", TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT, TINY_REPRICE,
                experiment="sensitivity", knobs={"n_bins": 50}, extra_key={"knob_name": "n_bins", "knob_value": 50})
    assert a != b
    p = store.get_params(a)
    assert p["estimator.n_bins"] == "5" and p["knob_value"] == "5" and p["budget"] == str(TINY_N)
    assert store.get_metrics(a)["budget"] == TINY_N
    assert len(store.search("sensitivity")) == 2 and len(store.search("bench")) == 0


def test_lev_rmse_logged_when_pde_reference_exists(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    rid = run_one(store, "s01", "nw", TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT, TINY_REPRICE)
    lev = json.loads(store.download(rid, "leverage.json", tmp_path / "d").read_text())
    with store.run("pde_reference", {"sid": "s01"}) as h:      # a fake reference equal to the nw field
        h.log_json("leverage.json", lev)
    rid2 = run_one(store, "s01", "nw", TINY_N, 1, TINY_EXPLICIT, TINY_IMPLICIT, TINY_REPRICE)
    m = store.get_metrics(rid2)
    assert "lev_rmse" in m and m["lev_rmse"] >= 0.0
    ref = LeverageField.from_json(lev)
    assert any(k.startswith("lev_rmse/T") for k in m) and len([k for k in m if k.startswith("lev_rmse/T")]) == len(ref)


def test_sweep_into_baselines_experiment(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    jobs = [("s01", "bins", TINY_N, 0), ("s01", "nw_ghl", TINY_N, 0)]
    assert sweep(store, jobs, explicit=TINY_EXPLICIT, implicit=TINY_IMPLICIT, reprice=TINY_REPRICE, experiment="baselines") == 2
    assert len(store.search("baselines")) == 2 and len(store.search("bench")) == 0
```

In `tests/bench/test_sweep.py` change `len(jobs) == 20 * len(ALGOS) * 2 * 3 + 6 * 2 * 3` to use `len(PAPER_ALGOS)`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/bench/test_baselines_sweep.py -q`. Expected: ImportError on `BASELINE_ALGOS`.

- [ ] **Step 3: Implement**

`bench/algos.py`: `_explicit` gains `knobs=None` and builds `make_estimator(method, seed=seed, first_steps=cfg.first_steps, later_steps=cfg.later_steps, local_vol=lv, s0=sc.s0, **(knobs or {}))`. Every `_xxx(sc, n, seed, e, i)` wrapper gains a trailing `knobs=None` and forwards it (`_implicit`, `_implicit_ridge`, `_nn_is` forward it to their explicit calls; `_implicit_core`'s warm start ignores knobs). Add:

```python
def _nw_ghl(sc, n, seed, e, i, knobs=None): return _explicit(sc, n, seed, e, "nw_ghl", knobs=knobs)
def _muguruza(sc, n, seed, e, i, knobs=None): return _explicit(sc, n, seed, e, "muguruza", knobs=knobs)
def _rkhs(sc, n, seed, e, i, knobs=None): return _explicit(sc, n, seed, e, "rkhs", knobs=knobs)
def _bins(sc, n, seed, e, i, knobs=None): return _explicit(sc, n, seed, e, "bins", knobs=knobs)
def _purbf(sc, n, seed, e, i, knobs=None): return _explicit(sc, n, seed, e, "purbf", knobs=knobs)

PAPER_ALGOS = ("nw", "explicit_nn", "ridge", "explicit_nn_is", "implicit_nn", "spline", "implicit_ridge")
BASELINE_ALGOS = ("nw_ghl", "muguruza", "rkhs", "bins", "purbf")
ALGOS = {"nw": _nw, "explicit_nn": _nn, "ridge": _ridge, "explicit_nn_is": _nn_is,
         "implicit_nn": _implicit, "spline": _spline, "implicit_ridge": _implicit_ridge,
         "nw_ghl": _nw_ghl, "muguruza": _muguruza, "rkhs": _rkhs, "bins": _bins, "purbf": _purbf}


def run_algo(name, scenario, n_particles, seed, explicit=ExplicitConfig(), implicit=ImplicitConfig(), knobs=None):
    return ALGOS[name](scenario, n_particles, seed, explicit, implicit, knobs=knobs)
```

The order of `ALGOS` keys keeps the seven paper algos first so `sorted(ALGOS)`-based tests are unaffected. `_implicit_ridge` intraday call: the `GlobalRidge` head has no knobs; pass none.

`bench/runner.py`:

```python
def run_one(store, sid, algo, n_particles, seed, explicit=ExplicitConfig(), implicit=ImplicitConfig(),
            reprice=RepriceConfig(), *, experiment=BENCH_EXPERIMENT, knobs=None, extra_key=None):
    key = {**run_key(sid, algo, n_particles, seed), **(extra_key or {})}
    existing = store.find_finished(experiment, key)
    if existing is not None:
        return existing
    sc = full_registry()[sid]
    params = {**key, "git_hash": git_hash(), "schema": SCHEMA, "budget": int(n_particles),
              **_prefixed("explicit", explicit.as_params()), **_prefixed("implicit", implicit.as_params()),
              **_prefixed("reprice", reprice.as_params()), **sc.as_params(),
              **_prefixed("estimator", knobs or {})}
    with store.run(experiment, params) as h:
        res = run_algo(algo, sc, n_particles, seed, explicit, implicit, knobs=knobs)
        ... (unchanged repricing and metrics) ...
        metrics["budget"] = int(n_particles)
        metrics.update(_leverage_error(store, sid, res.field))
        h.log_metrics(metrics)
        ... (unchanged artifacts) ...
```

with

```python
PDE_EXPERIMENT = "pde_reference"


def _leverage_error(store, sid, field):
    """RMSE against the PDE reference leverage for this scenario, when one is in the store."""
    rid = store.find_finished(PDE_EXPERIMENT, {"sid": sid})
    if rid is None:
        return {}
    with tempfile.TemporaryDirectory() as d:
        ref = LeverageField.from_json(json.loads(store.download(rid, "leverage.json", d).read_text()))
    a, b = field.resample(DEFAULT_GRID), ref.resample(DEFAULT_GRID)
    n = min(len(a), len(b))
    per = {f"lev_rmse/T{a[k].t:g}": float(np.sqrt(np.mean((a[k].L - b[k].L) ** 2))) for k in range(n)}
    pooled = float(np.sqrt(np.mean([(a[k].L - b[k].L) ** 2 for k in range(n)])))
    return {"lev_rmse": pooled, **per}
```

(imports: `json`, `tempfile`, `numpy as np`, `LeverageField`, `DEFAULT_GRID`). `SCHEMA` stays 2; `budget` duplicates `n_particles` on purpose so figures can group on one name.

`bench/sweep.py`: add `BUDGETS`, `baselines_grid()` as specified, an `experiment=BENCH_EXPERIMENT` keyword on `sweep` threaded through `_worker` (`args` tuples gain the experiment name), and `paper_grid` iterates `PAPER_ALGOS`:

```python
def baselines_grid():
    jobs = []
    for sid in make_registry():
        for algo in ALGOS:
            for n in BUDGETS:
                for seed in (0, 1, 2):
                    jobs.append((sid, algo, n, seed))
    for sid in fig3_registry():
        for algo in ALGOS:
            for n in BUDGETS[1:]:
                for seed in (0, 1, 2):
                    jobs.append((sid, algo, n, seed))
    return jobs
```

`cli.py`: `sweep --preset` choices `["paper", "baselines"]`; `paper` runs `sweep(store, paper_grid(), n_jobs=...)`, `baselines` runs `sweep(store, baselines_grid(), n_jobs=..., experiment="baselines")`; `run` gains `--algo` choices from the full `ALGOS` (already) and prints the same.

Tiny goldens: change `tests/golden/make_tiny_goldens.py::main` to take `sys.argv[1:]` as the algo names to (re)generate (default: all) and run `uv run python tests/golden/make_tiny_goldens.py nw_ghl muguruza rkhs bins purbf`. Do NOT regenerate the seven existing fixtures. `tests/test_golden_tiny.py` parametrises over `sorted(ALGOS)` and picks the new files up automatically.

- [ ] **Step 4: Run everything, goldens, commit**

Run: `uv run pytest -q` (12 tiny goldens now); `uv run pytest -m golden -q` (algos and runner changed).

```bash
git add src/neural_particle_method/bench src/neural_particle_method/cli.py tests/bench tests/golden tests/test_golden_tiny.py
git commit -m "feat: baseline algos registered; budgets and baselines sweep; knobs, extra keys, and PDE leverage error through the runner"
```

---

### Task 8: Sensitivity driver

**Files:**
- Create: `src/neural_particle_method/bench/sensitivity.py`
- Modify: `src/neural_particle_method/cli.py`
- Create: `tests/bench/test_sensitivity.py`

**Interfaces:**
- Produces: `bench.sensitivity.KNOBS: dict[str, tuple[str, tuple]]` mapping algo -> (knob name, values); `sensitivity_grid(sids=("s01", "li_simple"), budgets=(10_000, 100_000), seeds=(0, 1, 2), algos=None) -> list[(sid, algo, n, seed, knob_name, knob_value)]`; `run_sensitivity(store, jobs, n_jobs=1, explicit=..., implicit=..., reprice=...) -> int` (resumable, experiment `sensitivity`); CLI `nparticle sensitivity [--sids ...] [--budgets ...] [--algos ...] [--jobs K]`.

- [ ] **Step 1: Write the failing test**

```python
from neural_particle_method.bench.sensitivity import KNOBS, run_sensitivity, sensitivity_grid
from neural_particle_method.pricing.reprice import RepriceConfig
from neural_particle_method.tracking.store import Store
from tests.conftest import TINY_EXPLICIT, TINY_IMPLICIT, TINY_N, TINY_REPRICE_N, TINY_REPRICE_STEPS

TINY_REPRICE = RepriceConfig(TINY_REPRICE_N, TINY_REPRICE_STEPS)


def test_knob_table_matches_spec():
    assert KNOBS["nw_ghl"] == ("c", (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0))
    assert KNOBS["rkhs"][0] == "lam" and KNOBS["rkhs"][1] == tuple(10.0 ** -k for k in range(9, 1, -1))
    assert KNOBS["bins"] == ("n_bins", (5, 10, 20, 50, 100, 200, 500))
    assert KNOBS["purbf"] == ("prune", (0.5, 1.0, 2.0))
    assert KNOBS["explicit_nn"] == ("hidden", (8, 16, 32, 64, 128))
    assert KNOBS["ridge"] == ("lam", (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1))
    assert KNOBS["spline"] == ("lam", (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0))
    assert KNOBS["nw"] == ("bandwidth_scale", (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0))
    assert "muguruza" not in KNOBS


def test_grid_enumeration():
    jobs = sensitivity_grid(sids=("s01",), budgets=(1_000,), seeds=(0,), algos=("bins", "purbf"))
    assert len(jobs) == 7 + 3 and jobs[0] == ("s01", "bins", 1_000, 0, "n_bins", 5)


def test_run_is_resumable_and_logs_knob(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    jobs = sensitivity_grid(sids=("s01",), budgets=(TINY_N,), seeds=(0,), algos=("bins",))[:2]
    assert run_sensitivity(store, jobs, explicit=TINY_EXPLICIT, implicit=TINY_IMPLICIT, reprice=TINY_REPRICE) == 2
    assert run_sensitivity(store, jobs, explicit=TINY_EXPLICIT, implicit=TINY_IMPLICIT, reprice=TINY_REPRICE) == 0
    df = store.search("sensitivity")
    assert sorted(df["params.knob_value"].astype(int)) == [5, 10] and set(df["metrics.knob_value"]) == {5.0, 10.0}
```

`NadarayaWatson` needs a `bandwidth_scale=1.0` knob multiplying its Silverman bandwidth (add it in this task; default 1.0 keeps `nw` bit-identical, verified by the tiny golden).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/bench/test_sensitivity.py -q`. Expected: ImportError.

- [ ] **Step 3: Implement**

```python
"""Knob-sensitivity sweeps: one knob per method over a range centred on its published default."""
from concurrent.futures import ProcessPoolExecutor

from ..calibrate.config import ExplicitConfig, ImplicitConfig
from ..pricing.reprice import RepriceConfig
from ..tracking.store import Store
from .runner import run_key, run_one

SENSITIVITY_EXPERIMENT = "sensitivity"
_SCALES = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
KNOBS = {
    "nw_ghl": ("c", _SCALES),
    "nw": ("bandwidth_scale", _SCALES),
    "rkhs": ("lam", tuple(10.0 ** -k for k in range(9, 1, -1))),
    "bins": ("n_bins", (5, 10, 20, 50, 100, 200, 500)),
    "purbf": ("prune", (0.5, 1.0, 2.0)),
    "explicit_nn": ("hidden", (8, 16, 32, 64, 128)),
    "ridge": ("lam", (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)),
    "spline": ("lam", (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)),
}


def sensitivity_grid(sids=("s01", "li_simple"), budgets=(10_000, 100_000), seeds=(0, 1, 2), algos=None):
    jobs = []
    for sid in sids:
        for algo in (algos or KNOBS):
            name, values = KNOBS[algo]
            for n in budgets:
                for seed in seeds:
                    for val in values:
                        jobs.append((sid, algo, n, seed, name, val))
    return jobs


def _key(job):
    sid, algo, n, seed, name, val = job
    return {**run_key(sid, algo, n, seed), "knob_name": name, "knob_value": val}


def _worker(args):
    uri, root, job, explicit, implicit, reprice = args
    sid, algo, n, seed, name, val = job
    store = Store(uri, root)
    run_one(store, sid, algo, n, seed, explicit=explicit, implicit=implicit, reprice=reprice,
            experiment=SENSITIVITY_EXPERIMENT, knobs={name: val}, extra_key={"knob_name": name, "knob_value": val})
    return job


def run_sensitivity(store, jobs, n_jobs=1, explicit=ExplicitConfig(), implicit=ImplicitConfig(), reprice=RepriceConfig()):
    todo = [j for j in jobs if store.find_finished(SENSITIVITY_EXPERIMENT, _key(j)) is None]
    print(f"{len(todo)} sensitivity runs to do", flush=True)
    args = [(store.tracking_uri, store.artifact_root, j, explicit, implicit, reprice) for j in todo]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for done in ex.map(_worker, args):
                print("done:", *done, flush=True)
    else:
        for a in args:
            print("done:", *_worker(a), flush=True)
    return len(todo)
```

`runner.run_one` must log `knob_value` as a metric when `extra_key` contains it: after `metrics["budget"] = ...` add `if extra_key and "knob_value" in extra_key: metrics["knob_value"] = float(extra_key["knob_value"])`.

`NadarayaWatson.__init__(self, bandwidth=None, bandwidth_scale=1.0)`; in `fit_predict`, when `bandwidth is None` compute the Silverman value and multiply by `bandwidth_scale` (multiplying by 1.0 is bit-identical).

CLI: subparser `sensitivity` with `--sids` (nargs `*`, default `["s01", "li_simple"]`), `--budgets` (type int, nargs `*`, default `[10_000, 100_000]`), `--algos` (nargs `*`, default all in `KNOBS`), `--jobs`; calls `run_sensitivity(store, sensitivity_grid(...), n_jobs=args.jobs)`.

- [ ] **Step 4: Run, tiny goldens, commit**

Run: `uv run pytest -q` (the `nw` tiny golden guards the `bandwidth_scale` default).

```bash
git add src/neural_particle_method/bench src/neural_particle_method/estimators/nadaraya_watson.py src/neural_particle_method/cli.py tests/bench/test_sensitivity.py
git commit -m "feat: knob-sensitivity driver and nparticle sensitivity"
```

---

### Task 9: Acceptance driver and card tests

**Files:**
- Create: `src/neural_particle_method/bench/acceptance.py`
- Create: `tests/acceptance/__init__.py`, `tests/acceptance/test_cards.py`
- Create: `tests/bench/test_acceptance_driver.py`
- Modify: `src/neural_particle_method/cli.py`
- Modify: `BASELINES.md` (achieved numbers)

**Interfaces:**
- Produces: `bench.acceptance.Card(name, sid, algo, n_particles, seed, knobs, source_value, tolerance, metric)` frozen dataclass; `CARDS: tuple[Card, ...]`; `check(store, card, explicit=ExplicitConfig(), reprice=RepriceConfig()) -> (ok: bool, achieved: float, run_id: str)` which runs `run_one` in experiment `acceptance` with `extra_key={"card": card.name}` and evaluates `metric`; `run_acceptance(store, names=None) -> list[(name, ok, achieved, run_id)]`; CLI `nparticle acceptance [--cards ...]`.
- Metrics: `"avg_abs_pct"` = mean over the quote strikes at T = 1 of |IV error| in percentage points, computed from the run's `iv_err_bp.json` artifact as `mean(|err_bp|) / 100`; `"pooled_rmse_bp"` = the logged metric; `"delta_vs_nw_ghl_bp"` = this run's pooled RMSE minus the `nw_ghl` acceptance run's pooled RMSE on the same sid, seed, and budget (the driver runs `nw_ghl` first).

- [ ] **Step 1: Write the failing driver test**

`tests/bench/test_acceptance_driver.py`:

```python
from neural_particle_method.bench.acceptance import CARDS, Card, check, run_acceptance
from neural_particle_method.pricing.reprice import RepriceConfig
from neural_particle_method.tracking.store import Store
from tests.conftest import TINY_EXPLICIT, TINY_N, TINY_REPRICE_N, TINY_REPRICE_STEPS

TINY_REPRICE = RepriceConfig(TINY_REPRICE_N, TINY_REPRICE_STEPS)


def test_cards_table():
    names = [c.name for c in CARDS]
    assert names == ["nw_ghl_li_simple", "nw_ghl_li_complex", "muguruza_li_simple", "rkhs_li_simple",
                     "rkhs_li_complex", "bins_li_simple", "bins_li_complex", "purbf_li_simple"]
    by = {c.name: c for c in CARDS}
    assert by["bins_li_simple"].source_value == 0.91 and by["bins_li_complex"].source_value == 1.01
    assert by["nw_ghl_li_simple"].source_value == 1.44 and by["nw_ghl_li_complex"].source_value == 1.18
    assert by["nw_ghl_li_simple"].knobs == {"fixed_scale": 1.0}
    assert by["rkhs_li_simple"].knobs == {"n_centres": 40, "variance": 5.0, "lam": 1e-7}
    assert by["muguruza_li_simple"].metric == "delta_vs_nw_ghl_bp" and by["muguruza_li_simple"].tolerance == 10.0
    assert all(c.n_particles == 100_000 for c in CARDS if c.algo != "muguruza" and c.algo != "purbf")
    assert by["muguruza_li_simple"].n_particles == 50_000 and by["purbf_li_simple"].n_particles == 2_048


def test_check_runs_and_evaluates(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    tiny = Card("tiny_bins", "li_simple", "bins", TINY_N, 0, {}, source_value=100.0, tolerance=0.3, metric="avg_abs_pct")
    ok, achieved, rid = check(store, tiny, explicit=TINY_EXPLICIT, reprice=TINY_REPRICE)
    assert ok and achieved >= 0.0 and store.get_params(rid)["card"] == "tiny_bins"
    assert len(store.search("acceptance")) == 1


def test_run_acceptance_filters_by_name(tmp_path, monkeypatch):
    import neural_particle_method.bench.acceptance as acc
    tiny = Card("tiny_bins", "li_simple", "bins", TINY_N, 0, {}, 100.0, 0.3, "avg_abs_pct")
    monkeypatch.setattr(acc, "CARDS", (tiny,))
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    out = run_acceptance(store, names=["tiny_bins"], explicit=TINY_EXPLICIT, reprice=TINY_REPRICE)
    assert out == [("tiny_bins", True, out[0][2], out[0][3])]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/bench/test_acceptance_driver.py -q`. Expected: ImportError.

- [ ] **Step 3: Implement**

`GHLKernel` gains a `fixed_scale=None` knob: when set, `bandwidth()` returns `fixed_scale * s0 * n ** (-0.2)` (the thesis's `h = S0 N^(-1/5)`), ignoring `c` and the vol term. Add a line to `tests/estimators/test_ghl.py::test_bandwidth_rule` asserting `GHLKernel(fixed_scale=1.0).bandwidth(0.5, 10_000, None) == 10_000 ** -0.2`.

`bench/acceptance.py`:

```python
"""Card acceptance runs: reproduce a source number on the Heston market (see BASELINES.md)."""
import json
import tempfile
from dataclasses import dataclass, field

import numpy as np

from ..calibrate.config import ExplicitConfig, ImplicitConfig
from ..pricing.reprice import RepriceConfig
from .runner import run_key, run_one

ACCEPTANCE_EXPERIMENT = "acceptance"


@dataclass(frozen=True)
class Card:
    name: str
    sid: str
    algo: str
    n_particles: int
    seed: int
    knobs: dict = field(default_factory=dict)
    source_value: float = 0.0
    tolerance: float = 0.3
    metric: str = "avg_abs_pct"     # avg_abs_pct | pooled_rmse_bp | delta_vs_nw_ghl_bp


# Source numbers: Li (2023) Table 2.4 h0 rows (kernel, Heston market) 1.44 / 1.18 %; Table 3.2 (RKHS,
# overrides L=40, variance 5, lam 1e-7) 0.88 / 0.70 %; Table 4.2 (bins, l=20) 0.91 / 1.01 %.
# Muguruza: within 10 bp pooled RMSE of nw_ghl at N=5e4 (his "performs at the level of the particle method").
# PURBF: no numeric benchmark exists in Hakala (2019); the first accepted run is frozen as the target (spec).
# Bayer Figure 5 (ATM error flat for lam in [1e-9, 1e-5] on the `bayer` market) is a curve, not a single run:
# it is produced by `nparticle sensitivity --sids bayer --algos rkhs` and read off fig5, not asserted here.
CARDS = (
    Card("nw_ghl_li_simple", "li_simple", "nw_ghl", 100_000, 0, {"fixed_scale": 1.0}, 1.44),
    Card("nw_ghl_li_complex", "li_complex", "nw_ghl", 100_000, 0, {"fixed_scale": 1.0}, 1.18),
    Card("muguruza_li_simple", "li_simple", "muguruza", 50_000, 0, {}, 0.0, 10.0, "delta_vs_nw_ghl_bp"),
    Card("rkhs_li_simple", "li_simple", "rkhs", 100_000, 0, {"n_centres": 40, "variance": 5.0, "lam": 1e-7}, 0.88),
    Card("rkhs_li_complex", "li_complex", "rkhs", 100_000, 0, {"n_centres": 40, "variance": 5.0, "lam": 1e-7}, 0.70),
    Card("bins_li_simple", "li_simple", "bins", 100_000, 0, {}, 0.91),
    Card("bins_li_complex", "li_complex", "bins", 100_000, 0, {}, 1.01),
    Card("purbf_li_simple", "li_simple", "purbf", 2_048, 0, {}, 1.44),     # provisional target: beat the kernel row
)


def _avg_abs_pct(store, rid):
    with tempfile.TemporaryDirectory() as d:
        err = np.array(json.loads(store.download(rid, "iv_err_bp.json", d).read_text()), dtype=float)
    return float(np.nanmean(np.abs(err[-1])) / 100.0)


def check(store, card, explicit=ExplicitConfig(), implicit=ImplicitConfig(), reprice=RepriceConfig()):
    rid = run_one(store, card.sid, card.algo, card.n_particles, card.seed, explicit, implicit, reprice,
                  experiment=ACCEPTANCE_EXPERIMENT, knobs=card.knobs, extra_key={"card": card.name})
    if card.metric == "avg_abs_pct":
        achieved = _avg_abs_pct(store, rid)
        return achieved <= card.source_value + card.tolerance, achieved, rid
    if card.metric == "pooled_rmse_bp":
        achieved = store.get_metrics(rid)["pooled_rmse_bp"]
        return achieved <= card.source_value + card.tolerance, achieved, rid
    if card.metric == "delta_vs_nw_ghl_bp":
        ref = run_one(store, card.sid, "nw_ghl", card.n_particles, card.seed, explicit, implicit, reprice,
                      experiment=ACCEPTANCE_EXPERIMENT, extra_key={"card": f"ref_nw_ghl_{card.sid}_{card.n_particles}"})
        achieved = store.get_metrics(rid)["pooled_rmse_bp"] - store.get_metrics(ref)["pooled_rmse_bp"]
        return achieved <= card.tolerance, achieved, rid
    raise KeyError(card.metric)


def run_acceptance(store, names=None, explicit=ExplicitConfig(), implicit=ImplicitConfig(), reprice=RepriceConfig()):
    out = []
    for card in CARDS:
        if names and card.name not in names:
            continue
        ok, achieved, rid = check(store, card, explicit, implicit, reprice)
        print(f"{card.name}: {'PASS' if ok else 'FAIL'} achieved={achieved:.4g} source={card.source_value} run={rid}", flush=True)
        out.append((card.name, ok, achieved, rid))
    return out
```

`tests/acceptance/__init__.py` empty; `tests/acceptance/test_cards.py`:

```python
import pytest

from neural_particle_method.bench.acceptance import CARDS, check
from neural_particle_method.tracking.store import Store


@pytest.mark.slow
@pytest.mark.parametrize("card", CARDS, ids=lambda c: c.name)
def test_card(card):
    ok, achieved, rid = check(Store(), card)     # the repo store, so the run is kept
    assert ok, f"{card.name}: achieved {achieved:.4g} vs source {card.source_value} ± {card.tolerance} (run {rid})"
```

CLI: subparser `acceptance` with `--cards` (nargs `*`), calling `run_acceptance(store, names=args.cards or None)` and returning 1 if any card failed.

- [ ] **Step 4: Run the driver tests, then the real cards once, record numbers**

Run: `uv run pytest tests/bench/test_acceptance_driver.py -q`. Then from the repo root `uv run nparticle acceptance` (each card is one 1e5-particle run plus a 500k-path reprice; expect 10 to 20 minutes in total). Copy each card's `achieved` value and run id into the "Achieved" line of the matching card in `BASELINES.md` (add the line if the card lacks one). If a card FAILS, do not loosen it: record the achieved value with "FAIL" and report it as a concern; the paper text depends on knowing which sources reproduce.

```bash
git add src/neural_particle_method/bench/acceptance.py src/neural_particle_method/cli.py src/neural_particle_method/estimators/nadaraya_watson.py tests/acceptance tests/bench/test_acceptance_driver.py tests/estimators/test_ghl.py BASELINES.md
git commit -m "feat: acceptance cards driver, slow card tests, achieved numbers in BASELINES.md"
```

---

### Task 10: Aggregate per experiment and the three baseline figures

**Files:**
- Modify: `src/neural_particle_method/bench/aggregate.py`
- Create: `src/neural_particle_method/figures/fig1_baselines.py`, `fig3_baselines.py`, `fig5_sensitivity.py`
- Modify: `src/neural_particle_method/figures/__init__.py`, `src/neural_particle_method/cli.py`
- Create: `tests/figures/test_baseline_figures.py`
- Modify: `tests/bench/test_aggregate.py`

**Interfaces:**
- Produces: `aggregate(store, out_csv, out_md, experiment="bench")`; `COLUMNS` gains `budget`, `knob_name`, `knob_value` (empty for `bench` runs). `figures.ALL` gains the three modules; each has `make(summary_csv, store, outdir) -> str` and reads the store directly (`baselines` / `sensitivity` experiments), ignoring `summary_csv`. Output files `fig1_baselines.pdf`, `fig3_baselines.pdf`, `fig5_sensitivity.pdf`. The existing `fig1_accuracy`, `fig2_wings`, `fig3_plane`, `fig4_latency` are untouched, so the committed PDFs stay reproducible.

- [ ] **Step 1: Write the failing tests**

`tests/figures/test_baseline_figures.py`:

```python
from pathlib import Path

from neural_particle_method.figures import ALL, fig1_baselines, fig3_baselines, fig5_sensitivity
from neural_particle_method.tracking.store import Store


def _store(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    for sid in ("s01", "s02", "f_xi0.3_rho-0.7", "f_xi0.6_rho-0.3"):
        for algo in ("nw_ghl", "bins", "explicit_nn"):
            for n in (10_000, 100_000):
                with store.run("baselines", {"sid": sid, "algo": algo, "n_particles": n, "seed": 0, "budget": n}) as h:
                    h.log_metrics({"pooled_rmse_bp": 10.0 + n / 1e5, "budget": n})
    for algo, name, vals in (("bins", "n_bins", (5, 20, 100)), ("explicit_nn", "hidden", (16, 64))):
        for v in vals:
            with store.run("sensitivity", {"sid": "s01", "algo": algo, "n_particles": 10_000, "seed": 0,
                                           "knob_name": name, "knob_value": v}) as h:
                h.log_metrics({"pooled_rmse_bp": 5.0 + v / 100, "knob_value": v})
    return store


def test_modules_registered():
    assert fig1_baselines in ALL and fig3_baselines in ALL and fig5_sensitivity in ALL


def test_each_baseline_figure_writes_pdf(tmp_path):
    store = _store(tmp_path)
    for mod in (fig1_baselines, fig3_baselines, fig5_sensitivity):
        out = mod.make("unused.csv", store, str(tmp_path / "out"))
        assert Path(out).exists() and out.endswith(".pdf") and b"/CreationDate" not in Path(out).read_bytes()


def test_figures_survive_empty_store(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    for mod in (fig1_baselines, fig3_baselines, fig5_sensitivity):
        assert Path(mod.make("unused.csv", store, str(tmp_path / "out"))).exists()
```

In `tests/bench/test_aggregate.py` add:

```python
def test_aggregate_other_experiment_and_new_columns(store, tmp_path):
    with store.run("baselines", {"sid": "s01", "algo": "bins", "n_particles": 1000, "seed": 0, "budget": 1000}) as h:
        h.log_metrics({"pooled_rmse_bp": 1.0, "budget": 1000})
    df = aggregate(store, tmp_path / "b.csv", tmp_path / "b.md", experiment="baselines")
    assert len(df) == 1 and df.loc[0, "budget"] == 1000 and "knob_name" in df.columns
    assert len(aggregate(store, tmp_path / "s.csv", tmp_path / "d.md")) == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/figures/test_baseline_figures.py tests/bench/test_aggregate.py -q`. Expected: ImportError / TypeError on `experiment=`.

- [ ] **Step 3: Implement**

`aggregate.py`: signature `aggregate(store, out_csv="results/summary.csv", out_md="results/digest.md", experiment=BENCH_EXPERIMENT)`; `raw = store.search(experiment)`; `COLUMNS = (... existing ...) + ("budget", "knob_name", "knob_value")`; fill `budget` from `params.budget` (int or None), `knob_name` from `params.knob_name`, `knob_value` from `params.knob_value` (float or None). Existing CSV consumers only read the old columns, and `results/summary.csv` gains three empty trailing columns; regenerate it with `uv run nparticle aggregate` and commit the refreshed snapshot (header change only; row values unchanged; verify with `git diff --stat results/summary.csv` showing every line changed by the three trailing commas and nothing else, e.g. `diff <(cut -d, -f1-14 results/summary.csv) <(git show HEAD:results/summary.csv)` empty).

`figures/fig1_baselines.py`:

```python
"""Fig 1 (baselines): pooled IV RMSE per estimator at 1e4 and 1e5 particles, SSVI scenarios."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _frame(store):
    df = store.search("baselines", "attributes.status = 'FINISHED'")
    if len(df) == 0 or "params.sid" not in df:
        return None
    df = df[~df["params.sid"].str.startswith("f_")].copy()
    df["budget"] = df["params.budget"].astype(int)
    df["rmse"] = df["metrics.pooled_rmse_bp"].astype(float)
    return df


def make(summary_csv, store, outdir):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    df = _frame(store)
    if df is not None and len(df):
        g = df.groupby(["params.algo", "budget"]).rmse.agg(["mean", "std"]).reset_index()
        algos = sorted(g["params.algo"].unique(), key=lambda a: g[(g["params.algo"] == a)]["mean"].min())
        width, x = 0.38, np.arange(len(algos))
        for j, n in enumerate((10_000, 100_000)):
            sub = g[g.budget == n].set_index("params.algo").reindex(algos)
            ax.bar(x + (j - 0.5) * width, sub["mean"], width, yerr=sub["std"].fillna(0.0), capsize=2, label=f"N = {n:,}")
        ax.set_xticks(x, algos); ax.tick_params(axis="x", rotation=30); ax.legend(fontsize=8)
    ax.set_ylabel("pooled IV RMSE (bp)")
    out = Path(outdir) / "fig1_baselines.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, metadata={"CreationDate": None}); plt.close(fig)
    return str(out)
```

`figures/fig3_baselines.py`: same data access; filter `params.sid` starting with `f_` and `budget == 100_000`; one heatmap per estimator (`XIS = (0.3, 0.6, 1.0)`, `RHOS = (-0.7, -0.3)`) in a grid of subplots (`ncols=4`), shared colour scale (`vmin`/`vmax` from the pooled data), titles = estimator names, output `fig3_baselines.pdf`; with no data, an empty figure with the axis labels.

`figures/fig5_sensitivity.py`: read `sensitivity` FINISHED runs; for each `params.algo` group by `params.knob_value` (float) and plot mean pooled RMSE against the knob on a log-x axis, one subplot per estimator (`ncols=4`), each with a horizontal dashed line at the `explicit_nn` default's mean RMSE from the `baselines` experiment at the same budget when available (else no line); output `fig5_sensitivity.pdf`; empty store gives an empty figure.

`figures/__init__.py`: `ALL = (fig1_accuracy, fig2_wings, fig3_plane, fig4_latency, fig1_baselines, fig3_baselines, fig5_sensitivity)`. The CLI `figures` command needs no change. `cli.py` `aggregate` gains `--experiment` (default `bench`).

- [ ] **Step 4: Run, regenerate, commit**

Run: `uv run pytest -q`; `uv run nparticle aggregate` and check the snapshot as described; `uv run nparticle figures` and confirm `git status --short figures/out` shows only the three new PDFs (the four old ones byte-identical).

```bash
git add src/neural_particle_method/bench/aggregate.py src/neural_particle_method/figures src/neural_particle_method/cli.py tests/figures/test_baseline_figures.py tests/bench/test_aggregate.py results/summary.csv figures/out/fig1_baselines.pdf figures/out/fig3_baselines.pdf figures/out/fig5_sensitivity.pdf
git commit -m "feat: per-experiment aggregate; baseline accuracy, stress-plane, and sensitivity figures"
```

---

### Task 11: README, smoke of the presets, ruff

**Files:**
- Modify: `README.md`
- Modify: whatever ruff reports

- [ ] **Step 1: README**

Add to the Commands section:

```markdown
- `uv run nparticle sweep --preset baselines --jobs 4` — every estimator at N in {1e3, 1e4, 1e5} on the SSVI grid (resumable; about 2 000 runs)
- `uv run nparticle sensitivity --jobs 4` — one knob per method around its published default on `s01` and `li_simple`
- `uv run nparticle acceptance` — the BASELINES.md card criteria on the Heston market family
- `uv run nparticle figures` — now also writes `fig1_baselines.pdf`, `fig3_baselines.pdf`, `fig5_sensitivity.pdf`
```

and to the Layout list: `bench/sensitivity.py`, `bench/acceptance.py`, and the `HestonMarketSpec` family (`li_simple`, `li_complex`, `bayer`). Under Tests add `uv run pytest -m slow tests/acceptance` for the cards.

- [ ] **Step 2: Smoke the presets at tiny scale**

Run from the repo root against a throwaway store so the real one is untouched:

```bash
uv run nparticle --tracking-uri sqlite:////tmp/smoke.db --artifact-root /tmp/smoke_art run --scenario li_simple --algo muguruza --n 3000 --n-steps 6 --reprice-n 8000 --reprice-steps 8
uv run nparticle --tracking-uri sqlite:////tmp/smoke.db --artifact-root /tmp/smoke_art sensitivity --sids s01 --budgets 3000 --algos bins
```

(`sensitivity` needs a tiny-config path: add `--n-steps` to that subparser mirroring `run`, applied to both configs.) Both must complete; delete `/tmp/smoke.db` and `/tmp/smoke_art` afterwards.

- [ ] **Step 3: Ruff, full suite, goldens, commit**

Run: `uv run ruff check .` (fix findings by hand or `--fix`; `# noqa` rather than touching any numerical expression); `uv run pytest -q`; `uv run pytest -m golden -q`.

```bash
git add README.md src tests
git commit -m "docs: baseline commands; ruff clean"
```

---

## Verification checklist (end of plan)

- [ ] `uv run pytest -q` green, under about 30 s; 12 tiny goldens.
- [ ] `uv run pytest -m golden -q` green (6).
- [ ] `uv run pytest -m slow tests/acceptance -q` run once; achieved numbers and run ids recorded in `BASELINES.md`; any FAIL reported, not loosened.
- [ ] `uv run ruff check .` clean.
- [ ] `nparticle figures` leaves the four original PDFs byte-identical.
- [ ] `grep -rn "import mlflow\|from mlflow" src` shows only `tracking/store.py`.
- [ ] The papers session is told the estimator contract now has `ctx=None` and `make_estimator(**knobs)`, and that `baselines-rkhs-bins` is merged.
