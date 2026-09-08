# Benchmark Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A clean, MLflow-tracked benchmark of every particle-method leverage estimator, cold and lagged (stale bodies with heads), with every trained object saved and reloadable, plus the LaTeX tables for section 4 of `paper/notes_experiments.tex`.

**Architecture:** A new subpackage `neural_particle_method.suite` with one module per stage (lag, artifacts, cold, reference, offline, heads, online, grid, tables). Each table cell is one MLflow run in `suite_cold`, `suite_pde_floor`, `suite_offline` or `suite_lagged`. Existing calibration and bench code is reused through small additive hooks; the only behavioural change to existing code is the variance-floor default.

**Tech Stack:** Python 3.12, numpy, torch (CPU), pandas, mlflow (via `tracking.store.Store`), pytest, ruff. Run everything with `uv run ...` from the repo root. Spec: `docs/superpowers/specs/2026-09-08-benchmark-suite-design.md`.

## Global Constraints

- Branch `refactor-mlflow`. Commit after every task with the trailer lines below.
- Every new keyword or field defaults to the old behaviour. The one intentional change is `fit_v_floor` defaulting to `True` (Task 1). After Task 1, `uv run pytest` (default markers) must stay green through every task; `uv run pytest -m golden` must stay green with the pinned `fit_v_floor=False` replays.
- Suite configuration values, verbatim from the spec: calibration `n_steps=200`; repricing `RepriceConfig(n_particles=500_000, n_steps=200)`; cold and online `n_particles=100_000`; offline bodies at `200_000` and `500_000`; implicit `n_iters=30, alpha=0.5`; reprice seed `seed + 10_000`; cold and online seeds `(0, 1)`; offline seed `0`.
- Scenarios: the 20 of `make_registry()` plus `li_simple`, `li_complex`, `bayer` from `heston_registry()`; never the `fig3_registry()` ones.
- Lags: `Lag("surface", 0.0)` and `Lag("surface_spot", math.log(1.02))`; sticky-strike.
- SSVI surface bump is `calibrate.warm.scaled_bump(p, 1.0)`. Heston market bump: `sqrt(v0)` and `sqrt(theta)` up 0.01, `rho = min(rho + 0.03, -0.05)`, `xi = 0.95 * xi`, `kappa` unchanged.
- Head parameters: `RKHSHead(n_centres=100, lam=1e-6, variance=0.1)`; ridge head `lam=1e-3`, residual shrinkage toward the offline readout.
- MLflow experiment names: `suite_cold`, `suite_pde_floor`, `suite_offline`, `suite_lagged`, and the existing `pde_reference`. Smoke runs use the same names with suffix `_smoke`.
- Artifact names: `leverage.json`, `iv_err_bp.json`, `diagnostics.json`, `model.pt`, `model_meta.json`.
- Line length 100 (`uv run ruff check src tests` clean).
- Commit message trailer (both lines, verbatim):
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01EhbyGqFA5kYDyMwYouSgvS
  ```

---

## File map

Create:
- `src/neural_particle_method/suite/__init__.py` — empty docstring module.
- `src/neural_particle_method/suite/config.py` — `SUITE_EXPLICIT`, `SUITE_IMPLICIT`, `SUITE_REPRICE`, `SMOKE_*`, `SuiteSettings`, `SIDS`, `COLD_ALGOS`, `OFFLINE_SIZES`, `LAGS` re-export.
- `src/neural_particle_method/suite/lag.py` — `Lag`, `LAGS`, `bump_heston`, `ShiftedLocalVol`, `LaggedScenario`, `lagged_scenario`.
- `src/neural_particle_method/suite/artifacts.py` — `SliceBank`, `GlobalNetModel`, `save_model`, `load_run`, `LoadedRun`.
- `src/neural_particle_method/suite/cold.py` — `run_cold`.
- `src/neural_particle_method/suite/reference.py` — `run_pde_floor`.
- `src/neural_particle_method/suite/offline.py` — `run_offline`.
- `src/neural_particle_method/suite/heads.py` — `RKHSHead`, `FeatureRidgeHead`, `online_sweep`, `stale_field`.
- `src/neural_particle_method/suite/online.py` — `ONLINE_METHODS`, `run_online`.
- `src/neural_particle_method/suite/grid.py` — job lists and `run_stage`.
- `src/neural_particle_method/suite/tables.py` — `section4_tables`.
- `tests/suite/__init__.py`, `tests/suite/test_lag.py`, `test_artifacts.py`, `test_cold.py`, `test_reference.py`, `test_offline.py`, `test_heads.py`, `test_online.py`, `test_grid.py`, `test_tables.py`, `test_smoke.py`.

Modify:
- `src/neural_particle_method/calibrate/config.py` (floor default).
- `src/neural_particle_method/experiments/config.py` (`BumpConfig.fit_v_floor` default; `WarmConfig.fit_v_floor` new field).
- `src/neural_particle_method/experiments/warm_suite.py:65`, `calibrate/warm.py:128-135` (thread `fit_v_floor`).
- `tests/test_golden_archived.py`, `tests/test_golden_warm.py` (pin `fit_v_floor=False`), `tests/golden/tiny_*.json` (regenerated), `tests/experiments/test_config.py`.
- `src/neural_particle_method/estimators/nn.py` (`keep_slice_weights`).
- `src/neural_particle_method/bench/algos.py` (`CalibResult.model`).
- `src/neural_particle_method/bench/runner.py` (`save_model`, `n_steps` in the reference lookup).
- `src/neural_particle_method/bench/reference_runs.py` (`scenario` argument, lag key).
- `src/neural_particle_method/cli.py` (`suite` subcommand).
- `paper/notes_experiments.tex` (section 4 tables become `\input`).

---

### Task 1: Variance floor becomes the default; goldens regenerated; archived replays pinned

**Files:**
- Modify: `src/neural_particle_method/calibrate/config.py:19`
- Modify: `src/neural_particle_method/experiments/config.py` (BumpConfig line `fit_v_floor: bool = False`; WarmConfig gains a field)
- Modify: `src/neural_particle_method/experiments/warm_suite.py:65`
- Modify: `src/neural_particle_method/calibrate/warm.py:128-135`
- Modify: `tests/test_golden_archived.py`, `tests/test_golden_warm.py`, `tests/experiments/test_config.py`
- Regenerate: `tests/golden/tiny_*.json` via `tests/golden/make_tiny_goldens.py`
- Test: `tests/calibrate/test_config_defaults.py` (new)

**Interfaces:**
- Produces: `ExplicitConfig().fit_v_floor is True`; `BumpConfig().fit_v_floor is True`; `WarmConfig(fit_v_floor: bool = True)`; `calibrate.warm.full_resolve(lv, params, s0, T, cfg, seed, L0=None, n_iters=None, fit_v_floor=True)`.

- [ ] **Step 1: Write the failing default test**

Create `tests/calibrate/test_config_defaults.py` (create `tests/calibrate/__init__.py` if absent):

```python
from neural_particle_method.calibrate.config import ExplicitConfig
from neural_particle_method.experiments.config import BumpConfig, WarmConfig


def test_variance_floor_is_the_default_everywhere():
    assert ExplicitConfig().fit_v_floor is True
    assert BumpConfig().fit_v_floor is True
    assert WarmConfig().fit_v_floor is True
    assert WarmConfig(fit_v_floor=False).fit_v_floor is False
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/calibrate/test_config_defaults.py -v`
Expected: FAIL (`assert False is True` on the first line).

- [ ] **Step 3: Flip the defaults and thread the flag**

`src/neural_particle_method/calibrate/config.py`: change the `ExplicitConfig` line to

```python
    fit_v_floor: bool = True  # fit the estimator on max(v, 0), the variance the dynamics use; False = pre-fix paper behaviour
```

and change the module docstring's second sentence to `Defaults are the pre-refactor defaults except fit_v_floor (True since the variance-target fix of 8 Sept 2026).`

`src/neural_particle_method/experiments/config.py`: in `BumpConfig` set `fit_v_floor: bool = True` with comment `# False reproduces the paper runs`; in `WarmConfig` add, after `norm_iters: int = 4`:

```python
    fit_v_floor: bool = True   # explicit warm starts fit max(v, 0); False reproduces the paper runs
```

`src/neural_particle_method/experiments/warm_suite.py` line 65: replace `ExplicitConfig(n_steps=cfg.n_steps, n_particles=cfg.N)` with `ExplicitConfig(n_steps=cfg.n_steps, n_particles=cfg.N, fit_v_floor=cfg.fit_v_floor)`.

`src/neural_particle_method/calibrate/warm.py`: change `full_resolve` to

```python
def full_resolve(lv, params, s0, T, cfg, seed, L0=None, n_iters=None, fit_v_floor=True):
    """Explicit warm start (unless L0 given) followed by the damped implicit solve. Returns (field, seconds)."""
    t0 = time.perf_counter()
    if L0 is None:
        ecfg = ExplicitConfig(n_steps=cfg.n_steps, n_particles=cfg.n_particles, fit_v_floor=fit_v_floor)
```

(rest unchanged), and in `warm_suite.py` pass `fit_v_floor=cfg.fit_v_floor` in every `full_resolve(` call (lines 118, 168, 176, 207).

- [ ] **Step 4: Pin the archived replays to the old behaviour**

`tests/test_golden_archived.py`: add `from neural_particle_method.calibrate.config import ExplicitConfig` and change the `run_algo` call in `replay` to `run_algo(algo, sc, n_particles, seed, explicit=ExplicitConfig(fit_v_floor=False))`. Add to the module docstring: `Pinned to fit_v_floor=False: these archives predate the variance-target fix.`

`tests/test_golden_warm.py`: `BumpConfig(N=2000, n_steps=6, sub=1000, n_iters=2, fit_steps=40, fit_v_floor=False)` and `WarmConfig(N=2000, n_steps=6, sub=1000, n_iters=2, fit_steps=40, reprice_N=20000, reprice_steps=20, fit_v_floor=False)`.

`tests/experiments/test_config.py`: `BUMP_SMOKE == BumpConfig(20_000, 12, 8_000, 2, 120)` still holds (positional fields unchanged); add `assert BumpConfig().fit_v_floor is True` to `test_derived_configs`.

- [ ] **Step 5: Regenerate the tiny goldens**

Run: `uv run python tests/golden/make_tiny_goldens.py`
Expected: prints `wrote <algo>` for all 12 algorithms in `ALGOS`. `git diff --stat tests/golden/` shows every `tiny_*.json` changed.

- [ ] **Step 6: Run the suites**

Run: `uv run pytest -q` then `uv run pytest -m golden -q tests/test_golden_warm.py`
Expected: all pass. (`tests/test_golden_archived.py` needs `results/runs` archives and 50k-particle runs; run it if the archives are present: `uv run pytest -m golden tests/test_golden_archived.py -q`, expected pass.)

- [ ] **Step 7: Commit**

```bash
git add src/neural_particle_method/calibrate/config.py src/neural_particle_method/experiments/config.py \
  src/neural_particle_method/experiments/warm_suite.py src/neural_particle_method/calibrate/warm.py \
  tests/test_golden_archived.py tests/test_golden_warm.py tests/experiments/test_config.py \
  tests/calibrate tests/golden/tiny_*.json
git commit -m "Variance floor (fit_v_floor=True) is now the default; tiny goldens regenerated

Intentional drift: every one-pass scheme now fits E[max(v,0)|X], the variance the
dynamics use. Archived paper replays are pinned to fit_v_floor=False."
```
(append the trailer lines from Global Constraints).

---

### Task 2: Additive hooks — per-slice weight snapshots and a `model` slot on `CalibResult`

**Files:**
- Modify: `src/neural_particle_method/estimators/nn.py:22-60`
- Modify: `src/neural_particle_method/bench/algos.py:14-32,44-58`
- Test: `tests/estimators/test_nn_slice_weights.py` (new), `tests/bench/test_algos_model.py` (new)

**Interfaces:**
- Produces: `NNRegressor(seed=0, first_steps=400, later_steps=120, hidden=64, keep_slice_weights=False)` with attribute `slice_weights: list[tuple[float, dict]]` (time, state dict copy) appended after every `fit_predict` when the flag is on. `CalibResult(field, timings, diagnostics, model=None)`; `_explicit` sets `model=est`; `_implicit` sets `model=r.net` (a `GlobalNet`).

- [ ] **Step 1: Failing tests**

`tests/estimators/test_nn_slice_weights.py`:

```python
import numpy as np
import torch

from neural_particle_method.estimators.nn import NNRegressor


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return x, 0.04 + 0.01 * x**2


def test_default_keeps_nothing_and_is_numerically_unchanged():
    x, v = _data()
    a, b = NNRegressor(seed=0, first_steps=5, later_steps=2), NNRegressor(seed=0, first_steps=5, later_steps=2, keep_slice_weights=True)
    grid = np.linspace(-1, 1, 5)
    pa = a.fit_predict(0.1, x, v, grid)
    pb = b.fit_predict(0.1, x, v, grid)
    np.testing.assert_array_equal(pa, pb)
    assert a.slice_weights == [] and len(b.slice_weights) == 1


def test_snapshots_are_copies_taken_per_slice():
    x, v = _data()
    est = NNRegressor(seed=0, first_steps=5, later_steps=2, keep_slice_weights=True)
    grid = np.linspace(-1, 1, 5)
    est.fit_predict(0.1, x, v, grid)
    w0 = {k: t.clone() for k, t in est.slice_weights[0][1].items()}
    est.fit_predict(0.2, x, v + 0.01, grid)
    assert [t for t, _ in est.slice_weights] == [0.1, 0.2]
    for k in w0:
        assert torch.equal(w0[k], est.slice_weights[0][1][k])       # first snapshot untouched by later training
    assert any(not torch.equal(est.slice_weights[0][1][k], est.slice_weights[1][1][k]) for k in w0)
```

`tests/bench/test_algos_model.py`:

```python
from neural_particle_method.bench.algos import CalibResult, run_algo
from neural_particle_method.bench.scenarios import make_registry
from neural_particle_method.calibrate.implicit import GlobalNet
from neural_particle_method.estimators.nn import NNRegressor
from tests.conftest import TINY_EXPLICIT, TINY_IMPLICIT, TINY_N


def test_calibresult_model_defaults_to_none():
    assert CalibResult(None, {}, {}).model is None


def test_explicit_nn_and_implicit_expose_their_models():
    sc = make_registry()["s01"]
    ex = run_algo("explicit_nn", sc, TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT, knobs={"keep_slice_weights": True})
    assert isinstance(ex.model, NNRegressor) and len(ex.model.slice_weights) == TINY_EXPLICIT.n_steps - 1
    im = run_algo("implicit_nn", sc, TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT)
    assert isinstance(im.model, GlobalNet)
    assert run_algo("nw", sc, TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT).model is not None  # the NW estimator instance
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/estimators/test_nn_slice_weights.py tests/bench/test_algos_model.py -v`
Expected: FAIL with `TypeError: unexpected keyword argument 'keep_slice_weights'` and `TypeError` on `CalibResult(...).model`.

- [ ] **Step 3: Implement**

`estimators/nn.py`, in `NNRegressor`:

```python
    def __init__(self, seed=0, first_steps=400, later_steps=120, hidden=64, keep_slice_weights=False):
        torch.manual_seed(seed)
        self.net = SliceNet(hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-2)
        self.first_steps, self.later_steps = first_steps, later_steps
        self._n_fits = 0
        self.keep_slice_weights = keep_slice_weights
        self.slice_weights = []   # [(t, state_dict copy)] per fitted slice when keep_slice_weights
```

and at the end of `fit_predict`, before `return self.predict(grid)`:

```python
        if self.keep_slice_weights:
            self.slice_weights.append((float(t), {k: v.detach().clone() for k, v in self.net.state_dict().items()}))
```

`bench/algos.py`:

```python
@dataclass
class CalibResult:
    field: LeverageField
    timings: dict
    diagnostics: dict
    model: object = None   # trained object when there is one (estimator instance or GlobalNet); ignored by bench
```

In `_explicit` return `CalibResult(r.field, {"total_s": total, "fit_s": r.fit_s}, diag, model=est)`. In `_implicit` return `CalibResult(r.field, {...}, {...}, model=r.net)`. Leave `_implicit_ridge`, `_implicit_one` and the mechanism variants unchanged (their `model` stays `None`).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/estimators/test_nn_slice_weights.py tests/bench/test_algos_model.py tests/test_golden_tiny.py -q`
Expected: PASS (goldens unchanged: the snapshot is a copy, no numerics change).

- [ ] **Step 5: Commit**

```bash
git add src/neural_particle_method/estimators/nn.py src/neural_particle_method/bench/algos.py tests/estimators/test_nn_slice_weights.py tests/bench/test_algos_model.py
git commit -m "NNRegressor.keep_slice_weights and CalibResult.model (additive hooks for the suite)"
```

---

### Task 3: Runner and reference hooks — `save_model`, `n_steps`-keyed leverage error, references for arbitrary scenarios

**Files:**
- Modify: `src/neural_particle_method/bench/runner.py:26-76`
- Modify: `src/neural_particle_method/bench/reference_runs.py`
- Test: `tests/bench/test_runner_hooks.py` (new), `tests/bench/test_reference_runs.py` (extend)

**Interfaces:**
- Produces: `run_one(..., save_model=None)` where `save_model(h: RunHandle, res: CalibResult) -> None` is called after metrics are logged; `_leverage_error(store, sid, field, n_steps, lag="none")` looks up `pde_reference` by `{"sid": sid, "n_steps": n_steps, "lag": lag}`; `run_reference(store, sid, n_steps=50, n_x=801, n_v=200, scenario=None, lag="none")` keyed by `{"sid", "n_steps", "lag"}`, solving for `scenario` when given (a `ScenarioSpec`, `HestonMarketSpec` or `LaggedScenario`) with `x0 = log(scenario.s0)`.

- [ ] **Step 1: Failing tests**

`tests/bench/test_runner_hooks.py`:

```python
import pytest

from neural_particle_method.bench.runner import run_one
from neural_particle_method.pricing.reprice import RepriceConfig
from neural_particle_method.tracking.store import Store
from tests.conftest import TINY_EXPLICIT, TINY_IMPLICIT, TINY_N, TINY_REPRICE_N, TINY_REPRICE_STEPS

TINY_REPRICE = RepriceConfig(TINY_REPRICE_N, TINY_REPRICE_STEPS)


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def test_save_model_hook_is_called_with_handle_and_result(store):
    seen = {}

    def hook(h, res):
        seen["run_id"], seen["model"] = h.run_id, res.model
        h.log_json("model_meta.json", {"ok": True})

    rid = run_one(store, "s01", "nw", TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT, TINY_REPRICE,
                  experiment="hooked", save_model=hook)
    assert seen["run_id"] == rid and seen["model"] is not None
    assert "model_meta.json" in [a.path for a in store.client.list_artifacts(rid)]
```

Append to `tests/bench/test_reference_runs.py`:

```python
def test_reference_is_keyed_by_n_steps_and_lag(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    a = run_reference(store, "li_simple", n_steps=6, n_x=41, n_v=20)
    b = run_reference(store, "li_simple", n_steps=4, n_x=41, n_v=20)
    assert a != b
    assert store.get_params(a)["lag"] == "none"
    explicit = replace(TINY_EXPLICIT, n_steps=4)
    rid = run_one(store, "li_simple", "nw", TINY_N, 0, explicit, TINY_IMPLICIT, TINY_REPRICE)
    m = store.get_metrics(rid)
    assert "lev_rmse" in m            # matched the 4-step reference, not the 6-step one


def test_reference_accepts_an_explicit_scenario(tmp_path):
    from dataclasses import replace as dc_replace

    from neural_particle_method.bench.scenarios import full_registry
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    sc = dc_replace(full_registry()["li_simple"], s0=1.02)
    rid = run_reference(store, "li_simple", n_steps=6, n_x=41, n_v=20, scenario=sc, lag="surface_spot")
    p = store.get_params(rid)
    assert p["lag"] == "surface_spot" and p["s0"] == "1.02"
    assert run_reference(store, "li_simple", n_steps=6, n_x=41, n_v=20) != rid
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/bench/test_runner_hooks.py tests/bench/test_reference_runs.py -v`
Expected: FAIL (`unexpected keyword argument 'save_model'`; `unexpected keyword argument 'scenario'`; the `n_steps` test fails on `a != b` being fine but `lev_rmse` absent or wrong key — whichever fires first).

- [ ] **Step 3: Implement**

`bench/reference_runs.py`:

```python
def run_reference(store, sid, n_steps=50, n_x=801, n_v=200, scenario=None, lag="none"):
    """Returns the run id of a finished pde_reference run for (sid, n_steps, lag), computing one if needed.

    `scenario` overrides the registry entry (a lagged scenario, say); `lag` is part of the key so
    references for the same sid under different lags never collide. Legacy runs without a `lag`
    param are not matched: they were all computed on the registry scenario at 50 steps and are
    superseded by re-running."""
    key = {"sid": sid, "n_steps": int(n_steps), "lag": lag}
    existing = store.find_finished(REFERENCE_EXPERIMENT, key)
    if existing is not None:
        return existing
    sc = full_registry()[sid] if scenario is None else scenario
    x_grid = default_x_grid(n=n_x, x0=float(np.log(sc.s0)))
    v_grid = default_v_grid(sc.dynamics, sc.T, n=n_v)
    result = solve_leverage_pde(sc.local_vol(), sc.dynamics, s0=sc.s0, T=sc.T, n_steps=n_steps,
                                x_grid=x_grid, v_grid=v_grid)
    params = {**key, "n_x": int(n_x), "n_v": int(n_v), "s0": sc.s0, "git_hash": git_hash()}
    with store.run(REFERENCE_EXPERIMENT, params) as h:
        h.log_json("leverage.json", result.field.to_json())
        h.log_metrics({"runtime_s": result.runtime_s, "mass": result.mass, "forward": result.forward})
        return h.run_id
```

(add `import numpy as np`). `bench/runner.py`:

```python
def _leverage_error(store, sid, field, n_steps, lag="none"):
    """RMSE against the PDE reference leverage for this scenario at the same step count and lag."""
    rid = store.find_finished(PDE_EXPERIMENT, {"sid": sid, "n_steps": int(n_steps), "lag": lag})
    if rid is None:
        return {}
    ...  # body unchanged
```

`run_one` signature gains `save_model=None` after `extra_key=None`; the `_leverage_error` call becomes `_leverage_error(store, sid, res.field, explicit.n_steps)`; after `h.log_json("diagnostics.json", ...)` add

```python
        if save_model is not None:
            save_model(h, res)
```

Update the CLI `reference` subcommand: no change needed (it passes `n_steps`); `nparticle reference` now keys by lag `"none"`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/bench -q && uv run pytest tests/test_golden_tiny.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/neural_particle_method/bench/runner.py src/neural_particle_method/bench/reference_runs.py tests/bench/test_runner_hooks.py tests/bench/test_reference_runs.py
git commit -m "run_one save_model hook; PDE references keyed by n_steps and lag, solvable for any scenario"
```

---

### Task 4: Lag definition (`suite/lag.py`)

**Files:**
- Create: `src/neural_particle_method/suite/__init__.py`, `src/neural_particle_method/suite/lag.py`
- Test: `tests/suite/__init__.py`, `tests/suite/test_lag.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class Lag: kind: str; spot_move: float
  LAGS = (Lag("surface", 0.0), Lag("surface_spot", math.log(1.02)))
  def bump_heston(m: HestonParams) -> HestonParams
  class ShiftedLocalVol: __init__(inner, s0_ref); T_grid; t_min; sigma(t, x, s0=1.0)
  @dataclass(frozen=True)
  class LaggedScenario: sid, base (the registry spec), lag: Lag, s0_ref: float, s0: float, T, maturities, dynamics, family, ssvi (SSVIParams or None), market (HestonParams or None); local_vol(); target_ivs(k_grid, maturities); as_params()
  def lagged_scenario(sc, lag: Lag) -> LaggedScenario
  ```

- [ ] **Step 1: Failing tests**

`tests/suite/test_lag.py`:

```python
import math

import numpy as np

from neural_particle_method.bench.scenarios import full_registry, make_registry, quote_k_grid
from neural_particle_method.calibrate.warm import bump
from neural_particle_method.market.heston import heston_iv
from neural_particle_method.market.ssvi import implied_vol_ssvi
from neural_particle_method.suite.lag import LAGS, Lag, ShiftedLocalVol, bump_heston, lagged_scenario


def test_lags_are_the_two_agreed_ones():
    assert [l.kind for l in LAGS] == ["surface", "surface_spot"]
    assert LAGS[0].spot_move == 0.0 and math.isclose(LAGS[1].spot_move, math.log(1.02))


def test_surface_lag_reproduces_bump_on_an_admissible_scenario():
    sc = make_registry()["s01"]
    lsc = lagged_scenario(sc, Lag("surface", 0.0))
    assert lsc.ssvi == bump(sc.ssvi) and lsc.s0 == sc.s0 and lsc.s0_ref == sc.s0
    k = quote_k_grid()
    np.testing.assert_allclose(lsc.target_ivs(k, [0.5]), [implied_vol_ssvi(bump(sc.ssvi), k, 0.5)])


def test_spot_lag_shifts_target_and_start_but_not_absolute_local_vol():
    sc = make_registry()["s01"]
    d = math.log(1.02)
    lsc = lagged_scenario(sc, Lag("surface_spot", d))
    assert math.isclose(lsc.s0, sc.s0 * math.exp(d))
    k = quote_k_grid()
    np.testing.assert_allclose(lsc.target_ivs(k, [0.25, 1.0]),
                               np.stack([implied_vol_ssvi(lsc.ssvi, k + d, t) for t in (0.25, 1.0)]))
    lv, base = lsc.local_vol(), lagged_scenario(sc, Lag("surface", 0.0)).local_vol()
    x = np.exp(np.linspace(-0.5, 0.5, 7))
    for s0 in (1.0, 1.02, 0.7):     # the passed s0 is ignored: moneyness is always relative to s0_ref
        np.testing.assert_allclose(lv.sigma(0.5, x, s0), base.sigma(0.5, x, sc.s0))
    assert lv.t_min == base.t_min and np.array_equal(lv.T_grid, base.T_grid)


def test_shifted_local_vol_is_identity_when_ref_equals_passed_s0():
    sc = make_registry()["s02"]
    inner = sc.local_vol()
    sh = ShiftedLocalVol(inner, s0_ref=sc.s0)
    x = np.exp(np.linspace(-0.3, 0.3, 5))
    np.testing.assert_array_equal(sh.sigma(0.3, x, sc.s0), inner.sigma(0.3, x, sc.s0))


def test_bump_heston_moves_vol_level_by_one_point_and_stays_valid():
    for sid in ("li_simple", "li_complex", "bayer"):
        m = full_registry()[sid].market
        b = bump_heston(m)
        assert math.isclose(math.sqrt(b.v0), math.sqrt(m.v0) + 0.01)
        assert math.isclose(math.sqrt(b.theta), math.sqrt(m.theta) + 0.01)
        assert b.rho == min(m.rho + 0.03, -0.05) and math.isclose(b.xi, 0.95 * m.xi) and b.kappa == m.kappa
        assert b.xi > 0 and abs(b.rho) < 1 and b.v0 > 0 and b.theta > 0


def test_heston_lagged_scenario_targets_and_params():
    sc = full_registry()["li_simple"]
    d = math.log(1.02)
    lsc = lagged_scenario(sc, Lag("surface_spot", d))
    k = np.array([-0.2, 0.0, 0.2])
    np.testing.assert_allclose(lsc.target_ivs(k, [1.0]), [heston_iv(k + d, 1.0, bump_heston(sc.market), sc.s0)])
    p = lsc.as_params()
    assert p["lag.kind"] == "surface_spot" and math.isclose(float(p["lag.spot_move"]), d)
    assert p["s0_ref"] == sc.s0 and "bumped.v0" in p and p["scenario.family"] == "heston"
    assert lsc.family == "heston" and lsc.T == sc.T and lsc.maturities == sc.maturities
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/suite/test_lag.py -v`
Expected: FAIL with `ModuleNotFoundError: neural_particle_method.suite`.

- [ ] **Step 3: Implement**

`src/neural_particle_method/suite/__init__.py`:

```python
"""Clean benchmark suite: cold, PDE floor, offline bodies, lagged online methods, tables."""
```

`src/neural_particle_method/suite/lag.py`:

```python
"""What a lag is: a no-arbitrage surface bump, optionally with a sticky-strike spot move.

The particle cloud lives in absolute log-spot (`lnx` starts at log(s0)) and `local_vol.sigma(t, x, s0)`
takes moneyness relative to the `s0` it is passed. Under sticky-strike the implied vol at each
absolute strike is unchanged when spot moves, so the Dupire surface in absolute spot is unchanged
and only the particles' starting point moves: `ShiftedLocalVol` pins moneyness to the overnight
spot `s0_ref` whatever `s0` the harness passes, and `LaggedScenario.s0` is the new spot.
"""
import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from ..bench.scenarios import HestonMarketSpec, ScenarioSpec
from ..calibrate.warm import scaled_bump
from ..market.dupire import DupireSurface
from ..market.heston import heston_call, heston_iv
from ..market.local_vol import SSVILocalVol
from ..market.ssvi import implied_vol_ssvi
from ..simulate.dynamics import HestonParams


@dataclass(frozen=True)
class Lag:
    kind: str          # "surface" | "surface_spot"
    spot_move: float   # log spot return; 0 for a pure surface lag


LAGS = (Lag("surface", 0.0), Lag("surface_spot", math.log(1.02)))


def bump_heston(m):
    """Heston-market analogue of the SSVI bump: one vol point on level, +0.03 on rho, -5% on xi."""
    b = HestonParams(kappa=m.kappa, theta=(math.sqrt(m.theta) + 0.01) ** 2, xi=0.95 * m.xi,
                     rho=min(m.rho + 0.03, -0.05), v0=(math.sqrt(m.v0) + 0.01) ** 2)
    assert b.xi > 0 and abs(b.rho) < 1 and b.v0 > 0 and b.theta > 0
    return b


class ShiftedLocalVol:
    """Local vol whose moneyness is always taken relative to `s0_ref`, ignoring the passed s0."""

    def __init__(self, inner, s0_ref):
        self.inner, self.s0_ref = inner, float(s0_ref)
        self.T_grid = inner.T_grid

    @property
    def t_min(self):
        return self.inner.t_min

    def sigma(self, t, x, s0=1.0):
        return self.inner.sigma(t, x, self.s0_ref)


@dataclass(frozen=True)
class LaggedScenario:
    sid: str
    base: object
    lag: Lag
    s0_ref: float
    s0: float
    T: float
    maturities: tuple
    dynamics: HestonParams
    family: str
    ssvi: object = None      # bumped SSVIParams for family "ssvi"
    market: object = None    # bumped HestonParams for family "heston"

    def _inner_local_vol(self):
        if self.family == "ssvi":
            return SSVILocalVol(self.ssvi, self.s0_ref, T_max=self.T)
        b = self.base
        m = self.market
        T_grid = np.linspace(b.t_lo, b.T, b.n_t)
        k_grid = np.linspace(b.k_lo, b.k_hi, b.n_k)

        def price_fn(K, T):
            return heston_call(K, T, m.v0, m.kappa, m.theta, m.xi, m.rho, self.s0_ref)

        return DupireSurface.from_price_fn(price_fn, self.s0_ref, T_grid, k_grid)

    def local_vol(self):
        return ShiftedLocalVol(self._inner_local_vol(), self.s0_ref)

    def target_ivs(self, k_grid, maturities):
        k = np.asarray(k_grid, dtype=float) + self.lag.spot_move
        if self.family == "ssvi":
            return np.stack([implied_vol_ssvi(self.ssvi, k, t) for t in maturities])
        return np.stack([heston_iv(k, t, self.market, self.s0_ref) for t in maturities])

    def as_params(self):
        p = dict(self.base.as_params())
        p["scenario.s0"] = self.s0
        bumped = dataclasses.asdict(self.ssvi) if self.family == "ssvi" else self.market.to_dict()
        p.update({f"bumped.{k}": v for k, v in bumped.items()})
        p.update({"lag.kind": self.lag.kind, "lag.spot_move": self.lag.spot_move, "s0_ref": self.s0_ref})
        return p


def lagged_scenario(sc, lag):
    """The intraday market S1 for registry scenario `sc` under `lag`."""
    s0_new = sc.s0 * math.exp(lag.spot_move)
    common = dict(sid=sc.sid, base=sc, lag=lag, s0_ref=sc.s0, s0=s0_new, T=sc.T,
                  maturities=tuple(sc.maturities), dynamics=sc.dynamics, family=sc.family)
    if isinstance(sc, ScenarioSpec):
        q, _ = scaled_bump(sc.ssvi, 1.0)
        return LaggedScenario(ssvi=q, **common)
    if isinstance(sc, HestonMarketSpec):
        return LaggedScenario(market=bump_heston(sc.market), **common)
    raise TypeError(f"unsupported scenario type {type(sc).__name__}")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/suite/test_lag.py -v`
Expected: PASS. (If `test_surface_lag_reproduces_bump...` fails because `scaled_bump` shrinks on s01, check `no_arb_ok(bump(sc.ssvi))`; s01 has `eta=0.52, rho=-0.21`, so the unshrunk bump is admissible and the two must coincide.)

- [ ] **Step 5: Commit**

```bash
git add src/neural_particle_method/suite tests/suite
git commit -m "suite.lag: surface and sticky-strike spot lags for SSVI and Heston markets"
```

---

### Task 5: Artifacts — save and reload trained objects (`suite/artifacts.py`)

**Files:**
- Create: `src/neural_particle_method/suite/artifacts.py`
- Test: `tests/suite/test_artifacts.py`

**Interfaces:**
- Consumes: `NNRegressor.slice_weights` (Task 2), `calibrate.implicit.GlobalNet`, `estimators.nn.SliceNet`, `Z_SCALE`, `LeverageField`.
- Produces:
  ```python
  class SliceBank:   # per-slice explicit networks
      times: np.ndarray; nets: list[SliceNet]
      def f(self, t, lnx) -> np.ndarray           # E[V|X] from the slice in force at t
      def features(self, t, lnx) -> np.ndarray    # (n, 65): body features + constant
      def readout(self, t) -> np.ndarray          # (65,): head weight (64) and bias, the offline ridge solution
      def state(self) -> dict                     # serialisable
      @classmethod from_state(cls, d)
      @classmethod from_regressor(cls, est: NNRegressor)
  class GlobalNetModel:
      net: GlobalNet; T: float
      f(t, lnx); features(t, lnx); readout(t); state(); from_state(d)
  def model_kind(model) -> str                   # "explicit_slices" | "implicit_net" | "field_only"
  def save_model(h, model, meta: dict) -> str    # logs model.pt (unless field_only) and model_meta.json; returns kind
  @dataclass
  class LoadedRun: run_id: str; params: dict; metrics: dict; field: LeverageField; model: SliceBank | GlobalNetModel | None; meta: dict
  def load_run(store, run_id) -> LoadedRun
  ```
  Note: the softplus output means `f` is not linear in the readout; `readout(t)` is the pre-activation linear layer, used only as the shrinkage centre of the ridge head (Task 8), which fits on the same features.

- [ ] **Step 1: Failing tests**

`tests/suite/test_artifacts.py`:

```python
import numpy as np
import pytest
import torch

from neural_particle_method.bench.scenarios import make_registry
from neural_particle_method.calibrate.config import ExplicitConfig, ImplicitConfig
from neural_particle_method.calibrate.explicit import calibrate_explicit
from neural_particle_method.calibrate.implicit import calibrate_implicit
from neural_particle_method.estimators.nn import NNRegressor
from neural_particle_method.suite.artifacts import (
    GlobalNetModel,
    SliceBank,
    load_run,
    model_kind,
    save_model,
)
from neural_particle_method.tracking.store import Store

E = ExplicitConfig(n_steps=4, n_particles=500, fit_subsample=300, first_steps=5, later_steps=2)
I = ImplicitConfig(n_steps=4, n_particles=500, n_iters=1, fit_steps=5, pool_subsample=300)


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def _explicit():
    sc = make_registry()["s01"]
    est = NNRegressor(seed=0, first_steps=5, later_steps=2, keep_slice_weights=True)
    r = calibrate_explicit(sc.local_vol(), sc.dynamics, est, E, s0=sc.s0, T=sc.T, seed=0)
    return sc, est, r


def test_slice_bank_reproduces_the_regressor_slice_by_slice():
    sc, est, r = _explicit()
    bank = SliceBank.from_regressor(est)
    assert len(bank.nets) == E.n_steps - 1 and bank.times[0] == r.field[1].t
    x = np.linspace(-0.2, 0.2, 9)
    est.net.load_state_dict(est.slice_weights[0][1])
    np.testing.assert_allclose(bank.f(r.field[1].t, x), est.predict(x), rtol=1e-6)
    assert bank.features(0.3, x).shape == (9, 65) and np.all(bank.features(0.3, x)[:, -1] == 1.0)
    assert bank.readout(0.3).shape == (65,)
    assert model_kind(est) == "explicit_slices"


def test_slice_bank_time_rule_matches_leverage_field():
    sc, est, r = _explicit()
    bank = SliceBank.from_regressor(est)
    x = np.zeros(3)
    dt = sc.T / E.n_steps
    np.testing.assert_allclose(bank.f(dt * 1.5, x), bank.f(dt, x))          # last time <= t
    np.testing.assert_allclose(bank.f(0.0, x), bank.f(dt, x))               # before the first slice: first slice


def test_global_net_model_matches_the_network():
    sc = make_registry()["s01"]
    r = calibrate_implicit(sc.local_vol(), sc.dynamics, I, s0=sc.s0, T=sc.T, seed=0)
    gm = GlobalNetModel(r.net, sc.T)
    x = np.linspace(-0.3, 0.3, 5)
    with torch.no_grad():
        tz = torch.tensor(np.stack([np.full(5, 0.5 / sc.T), x / 0.3], axis=1), dtype=torch.float32)
        ref = r.net(tz).numpy()[:, 0]
    np.testing.assert_allclose(gm.f(0.5, x), ref, rtol=1e-6)
    assert gm.features(0.5, x).shape == (5, 65) and gm.readout(0.5).shape == (65,)
    assert model_kind(r.net) == "implicit_net"


def test_save_and_load_round_trip(store, tmp_path):
    sc, est, r = _explicit()
    with store.run("suite_offline", {"sid": "s01", "body": "explicit"}) as h:
        h.log_json("leverage.json", r.field.to_json())
        kind = save_model(h, est, {"body": "explicit", "n_steps": 4})
        rid = h.run_id
    assert kind == "explicit_slices"
    lr = load_run(store, rid)
    assert lr.meta["body"] == "explicit" and lr.meta["kind"] == "explicit_slices"
    assert isinstance(lr.model, SliceBank) and lr.params["sid"] == "s01"
    x = np.linspace(-0.2, 0.2, 9)
    np.testing.assert_allclose(lr.model.f(0.3, x), SliceBank.from_regressor(est).f(0.3, x), rtol=1e-6)
    for a, b in zip(lr.field, r.field):
        np.testing.assert_array_equal(a.L, b.L)
        np.testing.assert_array_equal(a.f, b.f)


def test_field_only_models_load_with_model_none(store):
    sc, est, r = _explicit()
    from neural_particle_method.estimators import make_estimator
    with store.run("suite_cold", {"sid": "s01", "algo": "nw"}) as h:
        h.log_json("leverage.json", r.field.to_json())
        assert save_model(h, make_estimator("nw"), {"body": "nw"}) == "field_only"
        rid = h.run_id
    lr = load_run(store, rid)
    assert lr.model is None and len(lr.field) == 4
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/suite/test_artifacts.py -v`
Expected: FAIL with `ModuleNotFoundError` for `suite.artifacts`.

- [ ] **Step 3: Implement**

`src/neural_particle_method/suite/artifacts.py`:

```python
"""Trained objects as MLflow artifacts, and their reload.

model.pt (torch): {"kind": ..., "state": ...}; model_meta.json: caller metadata plus "kind".
"""
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..calibrate.implicit import GlobalNet
from ..estimators.nn import NNRegressor, SliceNet, Z_SCALE
from ..simulate.leverage import LeverageField


def _slice_index(times, t):
    return max(int(np.searchsorted(times, t + 1e-12)) - 1, 0)


class SliceBank:
    """Per-slice explicit networks; slice in force at t is the last time <= t (as LeverageField.at)."""

    def __init__(self, times, nets):
        self.times, self.nets = np.asarray(times, dtype=float), list(nets)

    @classmethod
    def from_regressor(cls, est):
        nets = []
        for _, sd in est.slice_weights:
            n = SliceNet(est.net.body[0].out_features)
            n.load_state_dict(sd)
            nets.append(n)
        return cls([t for t, _ in est.slice_weights], nets)

    def _net(self, t):
        return self.nets[_slice_index(self.times, t)]

    def _z(self, lnx):
        return torch.tensor(np.asarray(lnx, dtype=float)[:, None] / Z_SCALE, dtype=torch.float32)

    def f(self, t, lnx):
        with torch.no_grad():
            return self._net(t)(self._z(lnx)).numpy()[:, 0]

    def features(self, t, lnx):
        with torch.no_grad():
            phi = self._net(t).body(self._z(lnx)).numpy()
        return np.concatenate([phi, np.ones((len(phi), 1))], axis=1)

    def readout(self, t):
        h = self._net(t).head
        return np.concatenate([h.weight.detach().numpy()[0], h.bias.detach().numpy()])

    def state(self):
        return {"times": self.times.tolist(), "hidden": self.nets[0].body[0].out_features,
                "state_dicts": [n.state_dict() for n in self.nets]}

    @classmethod
    def from_state(cls, d):
        nets = []
        for sd in d["state_dicts"]:
            n = SliceNet(d["hidden"])
            n.load_state_dict(sd)
            nets.append(n)
        return cls(d["times"], nets)


class GlobalNetModel:
    """The implicit scheme's global network as a (t, x) model with a 65-feature body."""

    def __init__(self, net, T):
        self.net, self.T = net, float(T)

    def _tz(self, t, lnx):
        x = np.asarray(lnx, dtype=float)
        return torch.tensor(np.stack([np.full(len(x), t / max(self.T, 1e-9)), x / Z_SCALE], axis=1),
                            dtype=torch.float32)

    def f(self, t, lnx):
        with torch.no_grad():
            return self.net(self._tz(t, lnx)).numpy()[:, 0]

    def features(self, t, lnx):
        with torch.no_grad():
            phi = self.net.body(self._tz(t, lnx)).numpy()
        return np.concatenate([phi, np.ones((len(phi), 1))], axis=1)

    def readout(self, t):
        h = self.net.head
        return np.concatenate([h.weight.detach().numpy()[0], h.bias.detach().numpy()])

    def state(self):
        return {"T": self.T, "hidden": self.net.body[0].out_features, "state_dict": self.net.state_dict()}

    @classmethod
    def from_state(cls, d):
        net = GlobalNet(d["hidden"])
        net.load_state_dict(d["state_dict"])
        return cls(net, d["T"])


def model_kind(model):
    if isinstance(model, NNRegressor) and model.slice_weights:
        return "explicit_slices"
    if isinstance(model, GlobalNet):
        return "implicit_net"
    if isinstance(model, (SliceBank, GlobalNetModel)):
        return "explicit_slices" if isinstance(model, SliceBank) else "implicit_net"
    return "field_only"


def _to_model(model, T):
    if isinstance(model, NNRegressor):
        return SliceBank.from_regressor(model)
    if isinstance(model, GlobalNet):
        return GlobalNetModel(model, T)
    return model


def save_model(h, model, meta):
    """Log model.pt and model_meta.json to run handle `h`. `meta` must carry "T" for implicit nets."""
    kind = model_kind(model)
    doc = {**meta, "kind": kind}
    if kind != "field_only":
        obj = _to_model(model, meta.get("T", 1.0))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.pt"
            torch.save({"kind": kind, "state": obj.state()}, p)
            h.log_file(p)
    h.log_json("model_meta.json", doc)
    return kind


@dataclass
class LoadedRun:
    run_id: str
    params: dict
    metrics: dict
    field: LeverageField
    model: object
    meta: dict


def load_run(store, run_id):
    """Rebuild a suite run's field and trained object from its artifacts alone."""
    with tempfile.TemporaryDirectory() as d:
        field = LeverageField.from_json(json.loads(store.download(run_id, "leverage.json", d).read_text()))
        names = {a.path for a in store.client.list_artifacts(run_id)}
        meta = json.loads(store.download(run_id, "model_meta.json", d).read_text()) if "model_meta.json" in names else {}
        model = None
        if "model.pt" in names:
            blob = torch.load(store.download(run_id, "model.pt", d), weights_only=True)
            model = (SliceBank if blob["kind"] == "explicit_slices" else GlobalNetModel).from_state(blob["state"])
    return LoadedRun(run_id, store.get_params(run_id), store.get_metrics(run_id), field, model, meta)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/suite/test_artifacts.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/neural_particle_method/suite/artifacts.py tests/suite/test_artifacts.py
git commit -m "suite.artifacts: SliceBank / GlobalNetModel, save_model and load_run over MLflow artifacts"
```

---

### Task 6: Suite configuration, cold stage, and PDE floor stage

**Files:**
- Create: `src/neural_particle_method/suite/config.py`, `src/neural_particle_method/suite/cold.py`, `src/neural_particle_method/suite/reference.py`
- Test: `tests/suite/test_cold.py`, `tests/suite/test_reference.py`

**Interfaces:**
- Consumes: `run_one(..., save_model=)` (Task 3), `save_model` (Task 5), `run_reference(..., scenario=, lag=)` (Task 3).
- Produces:
  ```python
  @dataclass(frozen=True)
  class SuiteSettings:
      explicit: ExplicitConfig; implicit: ImplicitConfig; reprice: RepriceConfig
      n_online: int; offline_sizes: tuple; seeds: tuple; sids: tuple; suffix: str
      def experiment(self, name) -> str      # name + suffix
  FULL: SuiteSettings; SMOKE: SuiteSettings
  COLD_ALGOS = ("nw", "explicit_nn", "implicit_nn", "rkhs", "spline", "nw_ghl", "bins", "muguruza", "purbf")
  def run_cold(store, sid, algo, seed, settings=FULL) -> run_id
  def run_pde_floor(store, sid, seed, settings=FULL) -> run_id
  ```

- [ ] **Step 1: Failing tests**

`tests/suite/test_cold.py`:

```python
import pytest

from neural_particle_method.suite.artifacts import load_run
from neural_particle_method.suite.cold import COLD_ALGOS, run_cold
from neural_particle_method.suite.config import FULL, SMOKE, SuiteSettings
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def test_full_settings_match_the_spec():
    assert (FULL.explicit.n_steps, FULL.implicit.n_steps, FULL.reprice.n_steps) == (200, 200, 200)
    assert FULL.explicit.fit_v_floor is True and FULL.reprice.n_particles == 500_000
    assert (FULL.implicit.n_iters, FULL.implicit.alpha) == (30, 0.5)
    assert FULL.n_online == 100_000 and FULL.offline_sizes == (200_000, 500_000) and FULL.seeds == (0, 1)
    assert len(FULL.sids) == 23 and FULL.sids[:2] == ("s01", "s02") and FULL.sids[-1] == "bayer"
    assert FULL.experiment("suite_cold") == "suite_cold" and SMOKE.experiment("suite_cold") == "suite_cold_smoke"
    assert COLD_ALGOS == ("nw", "explicit_nn", "implicit_nn", "rkhs", "spline", "nw_ghl", "bins", "muguruza", "purbf")


@pytest.mark.parametrize("algo", ["nw", "explicit_nn", "implicit_nn"])
def test_run_cold_logs_and_saves_model(store, algo):
    rid = run_cold(store, "s01", algo, 0, TINY)
    p, m = store.get_params(rid), store.get_metrics(rid)
    assert p["algo"] == algo and p["n_particles"] == str(TINY.n_online) and p["explicit.fit_v_floor"] == "True"
    assert m["pooled_mae_bp"] >= 0 and m["fit_s"] >= 0
    lr = load_run(store, rid)
    if algo == "nw":
        assert lr.model is None and lr.meta["kind"] == "field_only"
    else:
        assert lr.model is not None and lr.meta["kind"] == {"explicit_nn": "explicit_slices", "implicit_nn": "implicit_net"}[algo]
    assert run_cold(store, "s01", algo, 0, TINY) == rid
    assert len(store.search(TINY.experiment("suite_cold"))) == 1
```

`tests/suite/test_reference.py`:

```python
from neural_particle_method.suite.config import SuiteSettings
from neural_particle_method.suite.reference import run_pde_floor
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()


def test_pde_floor_reprices_the_reference_and_is_idempotent(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    rid = run_pde_floor(store, "li_simple", 0, TINY)
    p, m = store.get_params(rid), store.get_metrics(rid)
    assert p["sid"] == "li_simple" and p["n_steps"] == str(TINY.explicit.n_steps) and p["seed"] == "0"
    assert m["pooled_mae_bp"] >= 0 and m["fit_s"] == 0.0 and m["lev_rmse"] == 0.0
    assert run_pde_floor(store, "li_simple", 0, TINY) == rid
    assert len(store.search("pde_reference")) == 1
    assert run_pde_floor(store, "li_simple", 1, TINY) != rid
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/suite/test_cold.py tests/suite/test_reference.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/neural_particle_method/suite/config.py`:

```python
"""Suite settings: the agreed configuration (FULL), a two-minute SMOKE, and a test-sized tiny()."""
from dataclasses import dataclass

from ..bench.scenarios import heston_registry, make_registry
from ..calibrate.config import ExplicitConfig, ImplicitConfig
from ..pricing.reprice import RepriceConfig

SSVI_SIDS = tuple(make_registry())
HESTON_SIDS = tuple(heston_registry())
COLD_ALGOS = ("nw", "explicit_nn", "implicit_nn", "rkhs", "spline", "nw_ghl", "bins", "muguruza", "purbf")


@dataclass(frozen=True)
class SuiteSettings:
    explicit: ExplicitConfig
    implicit: ImplicitConfig
    reprice: RepriceConfig
    n_online: int
    offline_sizes: tuple
    seeds: tuple
    sids: tuple
    suffix: str = ""
    n_x: int = 801       # PDE reference grid
    n_v: int = 200

    def experiment(self, name):
        return name + self.suffix

    @classmethod
    def tiny(cls):
        """Seconds per stage; for unit tests."""
        return cls(ExplicitConfig(n_steps=4, n_particles=600, fit_subsample=300, first_steps=5, later_steps=2),
                   ImplicitConfig(n_steps=4, n_particles=600, n_iters=1, fit_steps=5, pool_subsample=300),
                   RepriceConfig(2_000, 4), n_online=600, offline_sizes=(800,), seeds=(0,),
                   sids=("s01", "li_simple"), suffix="_tiny", n_x=41, n_v=20)


FULL = SuiteSettings(
    ExplicitConfig(n_steps=200, n_particles=100_000, fit_v_floor=True),
    ImplicitConfig(n_steps=200, n_particles=100_000, n_iters=30, alpha=0.5),
    RepriceConfig(500_000, 200),
    n_online=100_000, offline_sizes=(200_000, 500_000), seeds=(0, 1), sids=SSVI_SIDS + HESTON_SIDS)

SMOKE = SuiteSettings(
    ExplicitConfig(n_steps=10, n_particles=2_000, fit_subsample=1_000, first_steps=20, later_steps=5, fit_v_floor=True),
    ImplicitConfig(n_steps=10, n_particles=2_000, n_iters=2, fit_steps=20, pool_subsample=2_000, alpha=0.5),
    RepriceConfig(20_000, 10),
    n_online=2_000, offline_sizes=(4_000,), seeds=(0,), sids=("s01", "li_simple"), suffix="_smoke", n_x=201, n_v=60)
```

`src/neural_particle_method/suite/cold.py`:

```python
"""Cold suite: one calibration from scratch per (scenario, algorithm, seed) at the online budget."""
from ..bench.runner import run_one
from ..bench.scenarios import full_registry
from .artifacts import save_model
from .config import COLD_ALGOS, FULL

__all__ = ["COLD_ALGOS", "run_cold"]


def run_cold(store, sid, algo, seed, settings=FULL):
    sc = full_registry()[sid]
    knobs = {"keep_slice_weights": True} if algo == "explicit_nn" else None

    def save(h, res):
        save_model(h, res.model, {"sid": sid, "algo": algo, "seed": seed, "T": sc.T,
                                  "n_steps": settings.explicit.n_steps, "n_particles": settings.n_online})

    return run_one(store, sid, algo, settings.n_online, seed, settings.explicit, settings.implicit,
                   settings.reprice, experiment=settings.experiment("suite_cold"), knobs=knobs,
                   save_model=save)
```

`src/neural_particle_method/suite/reference.py`:

```python
"""PDE floor: the Fokker-Planck reference leverage repriced under the suite's Monte Carlo."""
import json
import tempfile

from ..bench.reference_runs import run_reference
from ..bench.runner import PDE_EXPERIMENT
from ..bench.scenarios import full_registry, quote_k_grid
from ..pricing.metrics import iv_metrics
from ..pricing.reprice import reprice_iv, snap_times
from ..simulate.leverage import LeverageField
from ..tracking.store import git_hash
from .config import FULL


def score_field(field, sc, seed, reprice):
    """Reprice `field` on scenario `sc` and score it; returns (metrics dict in run_one's names, err grid)."""
    k = quote_k_grid()
    mats = list(sc.maturities)
    iv_model = reprice_iv(field, sc.dynamics, sc.s0, mats, k, reprice, seed=seed + 10_000)
    iv_target = sc.target_ivs(k, snap_times(mats, reprice.n_steps))
    m = iv_metrics(iv_model, iv_target, k, mats)
    out = {c: m[c] for c in ("pooled_rmse_bp", "pooled_max_bp", "wings_rmse_bp", "wings_max_bp", "n_failed",
                             "pooled_mae_bp", "wings_mae_bp")}
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
        field = LeverageField.from_json(json.loads(store.download(ref, "leverage.json", d).read_text()))
    params = {**key, "reference_run": ref, "git_hash": git_hash(), **sc.as_params(),
              **{f"reprice.{k}": v for k, v in settings.reprice.as_params().items()}}
    with store.run(exp, params) as h:
        metrics, err = score_field(field, sc, seed, settings.reprice)
        metrics.update({"fit_s": 0.0, "total_s": 0.0, "lev_rmse": 0.0})
        h.log_metrics(metrics)
        h.log_json("leverage.json", field.to_json())
        h.log_json("iv_err_bp.json", err)
        return h.run_id
```

(`PDE_EXPERIMENT` import can be dropped if unused; keep ruff clean.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/suite -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/neural_particle_method/suite/config.py src/neural_particle_method/suite/cold.py src/neural_particle_method/suite/reference.py tests/suite/test_cold.py tests/suite/test_reference.py
git commit -m "suite: settings (FULL/SMOKE/tiny), cold stage with saved models, PDE floor stage"
```

---

### Task 7: Offline bodies (`suite/offline.py`)

**Files:**
- Create: `src/neural_particle_method/suite/offline.py`
- Test: `tests/suite/test_offline.py`

**Interfaces:**
- Consumes: `score_field` (Task 6), `save_model`/`load_run` (Task 5), `NNRegressor(keep_slice_weights=True)`.
- Produces: `run_offline(store, sid, body, n_particles, settings=FULL, seed=0) -> run_id` in `settings.experiment("suite_offline")`, key `{sid, body, n_particles, seed, n_steps}`; artifacts `leverage.json`, `model.pt`, `model_meta.json`, `iv_err_bp.json`; metrics `fit_s`, `total_s`, and the anchor scores from `score_field` on the overnight surface.

- [ ] **Step 1: Failing test**

`tests/suite/test_offline.py`:

```python
import pytest

from neural_particle_method.suite.artifacts import GlobalNetModel, SliceBank, load_run
from neural_particle_method.suite.config import SuiteSettings
from neural_particle_method.suite.offline import run_offline
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


@pytest.mark.parametrize("body,cls", [("explicit", SliceBank), ("implicit", GlobalNetModel)])
def test_offline_body_is_cached_and_reloadable(store, body, cls):
    n = TINY.offline_sizes[0]
    rid = run_offline(store, "s01", body, n, TINY)
    p, m = store.get_params(rid), store.get_metrics(rid)
    assert p["body"] == body and p["n_particles"] == str(n) and p["seed"] == "0"
    assert m["fit_s"] >= 0 and m["pooled_mae_bp"] >= 0
    lr = load_run(store, rid)
    assert isinstance(lr.model, cls) and len(lr.field) == TINY.explicit.n_steps
    assert lr.meta["body"] == body and lr.meta["n_particles"] == n
    assert run_offline(store, "s01", body, n, TINY) == rid
    assert len(store.search(TINY.experiment("suite_offline"))) == 1


def test_unknown_body_rejected(store):
    with pytest.raises(ValueError):
        run_offline(store, "s01", "spline", 100, TINY)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/suite/test_offline.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/neural_particle_method/suite/offline.py`:

```python
"""Offline bodies: a full calibration on the overnight surface, saved once and reused by every online cell."""
import time
from dataclasses import replace

from ..bench.scenarios import full_registry
from ..calibrate.explicit import calibrate_explicit
from ..calibrate.implicit import calibrate_implicit
from ..estimators.nn import NNRegressor
from ..tracking.store import git_hash
from .artifacts import save_model
from .config import FULL
from .reference import score_field

BODIES = ("explicit", "implicit")


def run_offline(store, sid, body, n_particles, settings=FULL, seed=0):
    if body not in BODIES:
        raise ValueError(f"body must be one of {BODIES}, got {body!r}")
    n_steps = settings.explicit.n_steps
    key = {"sid": sid, "body": body, "n_particles": int(n_particles), "seed": int(seed), "n_steps": int(n_steps)}
    exp = settings.experiment("suite_offline")
    existing = store.find_finished(exp, key)
    if existing is not None:
        return existing
    sc = full_registry()[sid]
    lv = sc.local_vol()
    ecfg = replace(settings.explicit, n_particles=int(n_particles))
    icfg = replace(settings.implicit, n_particles=int(n_particles))
    params = {**key, "git_hash": git_hash(), **sc.as_params(),
              **{f"explicit.{k}": v for k, v in ecfg.as_params().items()},
              **{f"implicit.{k}": v for k, v in icfg.as_params().items()},
              **{f"reprice.{k}": v for k, v in settings.reprice.as_params().items()}}
    with store.run(exp, params) as h:
        t0 = time.perf_counter()
        est = NNRegressor(seed=seed, first_steps=ecfg.first_steps, later_steps=ecfg.later_steps,
                          keep_slice_weights=(body == "explicit"))
        w = calibrate_explicit(lv, sc.dynamics, est, ecfg, s0=sc.s0, T=sc.T, seed=seed)
        if body == "explicit":
            field, model, fit_s = w.field, est, w.fit_s
        else:
            r = calibrate_implicit(lv, sc.dynamics, icfg, s0=sc.s0, T=sc.T, seed=seed, L0=w.field)
            field, model, fit_s = r.field, r.net, w.fit_s + r.fit_s
        total = time.perf_counter() - t0
        metrics, err = score_field(field, sc, seed, settings.reprice)
        metrics.update({"fit_s": fit_s, "total_s": total})
        h.log_metrics(metrics)
        h.log_json("leverage.json", field.to_json())
        h.log_json("iv_err_bp.json", err)
        save_model(h, model, {**key, "T": sc.T})
        return h.run_id
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/suite/test_offline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/neural_particle_method/suite/offline.py tests/suite/test_offline.py
git commit -m "suite.offline: cached explicit and implicit bodies with saved models"
```

---

### Task 8: Heads and the causal online sweep (`suite/heads.py`)

**Files:**
- Create: `src/neural_particle_method/suite/heads.py`
- Test: `tests/suite/test_heads.py`

**Interfaces:**
- Consumes: `SliceBank`/`GlobalNetModel` (`f`, `features`, `readout`), `RKHSRidge`, `heston_step`, `Slice`, `LeverageField`, `DEFAULT_GRID`.
- Produces:
  ```python
  class RKHSHead:      # __init__(n_centres=100, lam=1e-6, variance=0.1); correction(t, lnx, v_plus, f_stale_at_lnx, model, grid) -> ndarray (len(grid))
  class FeatureRidgeHead:  # __init__(lam=1e-3); same correction signature; ridge on model.features, shrunk to model.readout(t)
  def stale_field(model_f, local_vol, s0, T, n_steps, L_max=4.0, grid=DEFAULT_GRID) -> LeverageField
  def online_sweep(model, local_vol, params, s0, T, cfg: ExplicitConfig, head, seed) -> (LeverageField, fit_s)
  ```
  `model_f(t, grid)` is any callable returning E[V|X] on the grid (a `SliceBank.f`, `GlobalNetModel.f`, or a field's own interpolated `f`).

- [ ] **Step 1: Failing tests**

`tests/suite/test_heads.py`:

```python
import numpy as np

from neural_particle_method.bench.scenarios import make_registry
from neural_particle_method.calibrate.config import ExplicitConfig
from neural_particle_method.calibrate.explicit import calibrate_explicit
from neural_particle_method.estimators.nn import NNRegressor
from neural_particle_method.simulate.leverage import DEFAULT_GRID
from neural_particle_method.suite.artifacts import SliceBank
from neural_particle_method.suite.heads import FeatureRidgeHead, RKHSHead, online_sweep, stale_field

E = ExplicitConfig(n_steps=4, n_particles=600, fit_subsample=300, first_steps=5, later_steps=2)


def _body():
    sc = make_registry()["s01"]
    est = NNRegressor(seed=0, first_steps=5, later_steps=2, keep_slice_weights=True)
    r = calibrate_explicit(sc.local_vol(), sc.dynamics, est, E, s0=sc.s0, T=sc.T, seed=0)
    return sc, SliceBank.from_regressor(est), r.field


def _f0(bank, v0):
    """Slice 0 is v0 (never fitted); later slices come from the bank. Same rule as online_sweep."""
    return lambda t, g: np.full(len(g), v0) if t == 0.0 else bank.f(t, g)


def test_rkhs_head_on_zero_residual_is_zero():
    sc, bank, _ = _body()
    rng = np.random.default_rng(0)
    x = rng.normal(scale=0.1, size=300)
    f = bank.f(0.5, x)
    corr = RKHSHead().correction(0.5, x, f, f, bank, np.linspace(-0.2, 0.2, 9))
    assert np.abs(corr).max() < 1e-8


def test_rkhs_head_recovers_a_smooth_residual():
    sc, bank, _ = _body()
    rng = np.random.default_rng(1)
    x = rng.normal(scale=0.1, size=2000)
    f = bank.f(0.5, x)
    grid = np.linspace(-0.15, 0.15, 7)
    corr = RKHSHead().correction(0.5, x, f + 0.01 * x, f, bank, grid)
    np.testing.assert_allclose(corr, 0.01 * grid, atol=2e-3)


def test_feature_ridge_head_reproduces_offline_when_target_is_offline():
    sc, bank, _ = _body()
    rng = np.random.default_rng(2)
    x = rng.normal(scale=0.1, size=2000)
    f = bank.f(0.5, x)
    grid = np.linspace(-0.15, 0.15, 7)
    corr = FeatureRidgeHead().correction(0.5, x, f, f, bank, grid)
    assert np.abs(corr).max() < 5e-4     # softplus output vs linear readout: small but not exactly zero


def test_stale_field_matches_calibrate_explicit_when_f_is_the_bodys_own():
    sc, bank, field = _body()
    lv = sc.local_vol()
    sf = stale_field(_f0(bank, sc.dynamics.v0), lv, sc.s0, sc.T, E.n_steps, L_max=E.L_max)
    np.testing.assert_array_equal(sf[0].f, field[0].f)
    assert len(sf) == E.n_steps and all(np.array_equal(s.grid, DEFAULT_GRID) for s in sf)
    for k in range(1, E.n_steps):     # same denominators, same Dupire => same leverage on the fixed grid
        s = field[k]
        inside = (DEFAULT_GRID >= s.grid.min()) & (DEFAULT_GRID <= s.grid.max())
        np.testing.assert_allclose(sf[k].L[inside], np.interp(DEFAULT_GRID[inside], s.grid, s.L), rtol=0.05)


def test_online_sweep_with_no_head_is_the_stale_field():
    sc, bank, _ = _body()
    lv = sc.local_vol()
    field, fit_s = online_sweep(bank, lv, sc.dynamics, sc.s0, sc.T, E, head=None, seed=5)
    ref = stale_field(_f0(bank, sc.dynamics.v0), lv, sc.s0, sc.T, E.n_steps, L_max=E.L_max)
    assert fit_s == 0.0 and len(field) == E.n_steps
    for a, b in zip(field, ref):
        np.testing.assert_array_equal(a.L, b.L)


def test_online_sweep_with_head_runs_and_changes_f():
    sc, bank, _ = _body()
    lv = sc.local_vol()
    field, fit_s = online_sweep(bank, lv, sc.dynamics, sc.s0, sc.T, E, head=RKHSHead(n_centres=20), seed=5)
    ref = stale_field(_f0(bank, sc.dynamics.v0), lv, sc.s0, sc.T, E.n_steps, L_max=E.L_max)
    assert fit_s > 0 and len(field) == E.n_steps
    assert np.array_equal(field[0].f, ref[0].f)                 # slice 0 is v0 in both
    assert any(not np.array_equal(field[k].f, ref[k].f) for k in range(1, E.n_steps))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/suite/test_heads.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/neural_particle_method/suite/heads.py`:

```python
"""Online heads on a frozen offline body, and the causal one-pass sweep that applies them.

Every head implements `correction(t, lnx, v_plus, f_stale, model, grid)` returning the additive
correction to the stale denominator on `grid`, fitted from the online cloud (`lnx`, `v_plus`) and
the stale values `f_stale = model.f(t, lnx)` at the same particles.
"""
import time

import numpy as np

from ..calibrate.config import ExplicitConfig
from ..estimators.rkhs import RKHSRidge
from ..simulate.dynamics import HestonParams
from ..simulate.leverage import DEFAULT_GRID, LeverageField, Slice
from ..simulate.stepper import heston_step


class RKHSHead:
    """Gaussian kernel ridge on the residual v+ - f_stale (Bayer et al.'s estimator, smaller lambda)."""

    def __init__(self, n_centres=100, lam=1e-6, variance=0.1):
        self.est = RKHSRidge(n_centres=n_centres, lam=lam, variance=variance)
        self.params = {"head": "rkhs", "n_centres": n_centres, "lam": lam, "variance": variance}

    def correction(self, t, lnx, v_plus, f_stale, model, grid):
        return self.est.fit_predict(t, lnx, v_plus - f_stale, grid)


class FeatureRidgeHead:
    """Per-slice ridge readout on the body's frozen features, shrunk toward the offline readout."""

    def __init__(self, lam=1e-3):
        self.lam = lam
        self.params = {"head": "ridge", "lam": lam}

    def correction(self, t, lnx, v_plus, f_stale, model, grid):
        A = model.features(t, lnx)
        w0 = model.readout(t)
        lam = self.lam * len(lnx)
        lhs = A.T @ A + lam * np.eye(A.shape[1])
        rhs = A.T @ v_plus + lam * w0
        w = np.linalg.solve(lhs, rhs)
        return model.features(t, grid) @ w - model.f(t, grid)


def stale_field(model_f, local_vol, s0, T, n_steps, L_max=4.0, grid=DEFAULT_GRID):
    """Stale denominators, fresh Dupire: no particles needed.

    Slice 0 is the one-point slice at log(s0); the caller's `model_f` must return v0 there (the
    explicit pass never fits slice 0). `online_sweep` wraps the body accordingly."""
    dt = T / n_steps
    slices = []
    for k in range(n_steps):
        t = k * dt
        if k == 0:
            g = np.array([np.log(s0)])
            f = np.clip(model_f(t, g), 1e-4, None)
        else:
            g = grid.copy()
            f = np.clip(model_f(t, g), 1e-4, None)
        sig = local_vol.sigma(max(t, local_vol.T_grid[0]), np.exp(g), s0)
        slices.append(Slice(t, g, np.clip(sig / np.sqrt(f), 0.0, L_max), f))
    return LeverageField(slices)


def online_sweep(model, local_vol, params, s0, T, cfg=ExplicitConfig(), head=None, seed=0):
    """calibrate_explicit with the estimator replaced by "stale + head correction".

    Returns (field, fit_s) where fit_s is the head time only. `head=None` is the stale-f, fresh-Dupire
    method and costs nothing: it returns `stale_field` on the fixed grid without simulating.
    """
    hp = HestonParams.from_dict(params)
    if head is None:
        f0 = lambda t, g: np.full(len(g), hp.v0) if t == 0.0 else model.f(t, g)   # noqa: E731
        return stale_field(f0, local_vol, s0, T, cfg.n_steps, L_max=cfg.L_max), 0.0
    n_steps, n_particles, fit_subsample, L_max = cfg.n_steps, cfg.n_particles, cfg.fit_subsample, cfg.L_max
    rng = np.random.default_rng(seed)
    dt, sdt = T / n_steps, np.sqrt(T / n_steps)
    lnx = np.full(n_particles, np.log(s0))
    v = np.full(n_particles, hp.v0)
    slices, fit_s = [], 0.0
    for k in range(n_steps):
        t = k * dt
        if k == 0:
            grid, f_grid = np.array([np.log(s0)]), np.array([hp.v0])
        else:
            grid = np.unique(np.quantile(lnx, np.linspace(0.001, 0.999, 101)))
            idx = rng.choice(n_particles, size=min(fit_subsample, n_particles), replace=False)
            x_sub, v_sub = lnx[idx], np.maximum(v[idx], 0.0)
            t0 = time.perf_counter()
            f_stale = model.f(t, x_sub)
            f_grid = model.f(t, grid) + head.correction(t, x_sub, v_sub, f_stale, model, grid)
            fit_s += time.perf_counter() - t0
        f_grid = np.clip(f_grid, 1e-4, None)
        sig = local_vol.sigma(max(t, local_vol.T_grid[0]), np.exp(grid), s0)
        L_grid = np.clip(sig / np.sqrt(f_grid), 0.0, L_max)
        slices.append(Slice(t, grid.copy(), L_grid.copy(), f_grid.copy()))
        L_p = np.interp(lnx, grid, L_grid) if len(grid) > 1 else np.full(n_particles, L_grid[0])
        zb, zp = rng.standard_normal(n_particles), rng.standard_normal(n_particles)
        lnx, v = heston_step(lnx, v, L_p, zb, zp, hp, dt, sdt)
    return LeverageField(slices), fit_s
```

Note for the implementer: slice 0 must use `v0`, not the model at `t=0` (the explicit pass never fitted slice 0). `online_sweep` handles that with `f0`, and the tests wrap the bank the same way with `_f0`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/suite/test_heads.py -v`
Expected: PASS. If `test_rkhs_head_recovers_a_smooth_residual` misses `atol=2e-3`, the kernel variance `0.1` is wide against a cloud of scale `0.1`; the test tolerance is the spec's, so investigate before loosening it (the fit of a linear residual with a Gaussian kernel of that width on 2000 points should be within 1e-3 on `[-0.15, 0.15]`).

- [ ] **Step 5: Commit**

```bash
git add src/neural_particle_method/suite/heads.py tests/suite/test_heads.py
git commit -m "suite.heads: RKHS and feature-ridge heads, stale field, causal online sweep"
```

---

### Task 9: Online runs (`suite/online.py`)

**Files:**
- Create: `src/neural_particle_method/suite/online.py`
- Test: `tests/suite/test_online.py`

**Interfaces:**
- Consumes: `lagged_scenario`, `LAGS` (Task 4); `load_run` (Task 5); `run_offline` (Task 7); `online_sweep`, `stale_field`, `RKHSHead`, `FeatureRidgeHead` (Task 8); `score_field` (Task 6); `run_reference(..., scenario=, lag=)` and `_leverage_error` (Task 3); `calibrate_explicit`, `make_estimator("nw")`.
- Produces:
  ```python
  ONLINE_METHODS = {"explicit_stale": ("explicit", None), "implicit_stale": ("implicit", None),
                    "explicit_rkhs": ("explicit", "rkhs"), "implicit_rkhs": ("implicit", "rkhs"),
                    "explicit_ridge": ("explicit", "ridge"), "implicit_ridge": ("implicit", "ridge"),
                    "stale_L": ("explicit", "stale_L"), "nw_resolve": (None, "nw_resolve")}
  def run_online(store, sid, method, offline_n, lag: Lag, seed, settings=FULL) -> run_id
  def ensure_lagged_reference(store, sid, lag, settings) -> run_id | None
  ```
  Run key: `{sid, method, offline_n, lag: lag.kind, seed, n_steps}` in `settings.experiment("suite_lagged")`. Metrics: `score_field` names plus `online_s` (head/sweep wall time; `0.0` for the stale rows), `fit_s` (= `online_s`), and `lev_rmse` when a lagged reference exists.

- [ ] **Step 1: Failing tests**

`tests/suite/test_online.py`:

```python
import pytest

from neural_particle_method.suite.config import SuiteSettings
from neural_particle_method.suite.lag import LAGS
from neural_particle_method.suite.online import ONLINE_METHODS, run_online
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def test_method_table_is_the_agreed_one():
    assert list(ONLINE_METHODS) == ["explicit_stale", "implicit_stale", "explicit_rkhs", "implicit_rkhs",
                                    "explicit_ridge", "implicit_ridge", "stale_L", "nw_resolve"]


@pytest.mark.parametrize("method", list(ONLINE_METHODS))
def test_every_method_runs_on_the_spot_lag(store, method):
    n = TINY.offline_sizes[0]
    rid = run_online(store, "s01", method, n, LAGS[1], 0, TINY)
    p, m = store.get_params(rid), store.get_metrics(rid)
    assert p["method"] == method and p["lag"] == "surface_spot" and p["lag.kind"] == "surface_spot"
    assert float(p["scenario.s0"]) > 1.0 and "bumped.sigma0" in p
    assert m["pooled_mae_bp"] >= 0 and m["online_s"] >= 0
    if method in ("explicit_stale", "implicit_stale", "stale_L"):
        assert m["online_s"] == 0.0
    if method != "nw_resolve":
        assert p["offline_run"]
    assert run_online(store, "s01", method, n, LAGS[1], 0, TINY) == rid


def test_offline_bodies_are_shared_across_lags_and_seeds(store):
    n = TINY.offline_sizes[0]
    a = run_online(store, "s01", "explicit_rkhs", n, LAGS[0], 0, TINY)
    b = run_online(store, "s01", "explicit_rkhs", n, LAGS[1], 0, TINY)
    assert a != b
    assert store.get_params(a)["offline_run"] == store.get_params(b)["offline_run"]
    assert len(store.search(TINY.experiment("suite_offline"))) == 1


def test_lagged_reference_gives_lev_rmse(store):
    n = TINY.offline_sizes[0]
    rid = run_online(store, "li_simple", "implicit_stale", n, LAGS[1], 0, TINY)
    assert "lev_rmse" in store.get_metrics(rid)
    refs = store.search("pde_reference")
    assert set(refs["params.lag"]) == {"surface_spot"}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/suite/test_online.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/neural_particle_method/suite/online.py`:

```python
"""Lagged suite, online stage: one run per (scenario, method, offline size, lag, seed)."""
import time
from dataclasses import replace

from ..bench.reference_runs import run_reference
from ..bench.runner import _leverage_error
from ..bench.scenarios import full_registry
from ..calibrate.explicit import calibrate_explicit
from ..estimators import make_estimator
from ..tracking.store import git_hash, to_jsonable
from .artifacts import load_run
from .config import FULL
from .heads import FeatureRidgeHead, RKHSHead, online_sweep
from .lag import lagged_scenario
from .offline import run_offline
from .reference import score_field

ONLINE_METHODS = {
    "explicit_stale": ("explicit", None), "implicit_stale": ("implicit", None),
    "explicit_rkhs": ("explicit", "rkhs"), "implicit_rkhs": ("implicit", "rkhs"),
    "explicit_ridge": ("explicit", "ridge"), "implicit_ridge": ("implicit", "ridge"),
    "stale_L": ("explicit", "stale_L"), "nw_resolve": (None, "nw_resolve"),
}


def _head(kind):
    return {"rkhs": RKHSHead(), "ridge": FeatureRidgeHead()}[kind]


def ensure_lagged_reference(store, sid, lag, settings):
    """PDE reference for the lagged scenario (keyed by lag kind); None if the solve fails."""
    lsc = lagged_scenario(full_registry()[sid], lag)
    try:
        return run_reference(store, sid, n_steps=settings.explicit.n_steps, n_x=settings.n_x, n_v=settings.n_v,
                             scenario=lsc, lag=lag.kind)
    except Exception as e:  # noqa: BLE001 - a failed reference must not block the online run
        print(f"warning: lagged PDE reference failed for {sid}/{lag.kind}: {e}")
        return None


def run_online(store, sid, method, offline_n, lag, seed, settings=FULL):
    if method not in ONLINE_METHODS:
        raise KeyError(f"unknown online method {method!r}; choose from {list(ONLINE_METHODS)}")
    body, head_kind = ONLINE_METHODS[method]
    n_steps = settings.explicit.n_steps
    key = {"sid": sid, "method": method, "offline_n": int(offline_n if body else 0), "lag": lag.kind,
           "seed": int(seed), "n_steps": int(n_steps)}
    exp = settings.experiment("suite_lagged")
    existing = store.find_finished(exp, key)
    if existing is not None:
        return existing
    sc = full_registry()[sid]
    lsc = lagged_scenario(sc, lag)
    lv = lsc.local_vol()
    ecfg = replace(settings.explicit, n_particles=settings.n_online)
    offline = load_run(store, run_offline(store, sid, body, offline_n, settings)) if body else None
    ensure_lagged_reference(store, sid, lag, settings)
    params = {**key, "git_hash": git_hash(), **lsc.as_params(),
              "offline_run": offline.run_id if offline else "",
              **{f"explicit.{k}": v for k, v in ecfg.as_params().items()},
              **{f"reprice.{k}": v for k, v in settings.reprice.as_params().items()}}
    head = _head(head_kind) if head_kind in ("rkhs", "ridge") else None
    if head is not None:
        params.update({f"head.{k}": v for k, v in head.params.items()})
    with store.run(exp, params) as h:
        t0 = time.perf_counter()
        if head_kind == "nw_resolve":
            est = make_estimator("nw", seed=seed, local_vol=lv, s0=lsc.s0)
            r = calibrate_explicit(lv, lsc.dynamics, est, ecfg, s0=lsc.s0, T=lsc.T, seed=seed)
            field, online_s = r.field, time.perf_counter() - t0
        elif head_kind == "stale_L":
            field, online_s = offline.field, 0.0
        else:
            field, online_s = online_sweep(offline.model, lv, lsc.dynamics, lsc.s0, lsc.T, ecfg, head=head,
                                           seed=seed + 1)
            if head is not None:
                online_s = time.perf_counter() - t0
        metrics, err = score_field(field, lsc, seed, settings.reprice)
        metrics.update({"online_s": online_s, "fit_s": online_s, "total_s": online_s})
        metrics.update(_leverage_error(store, sid, field, n_steps, lag=lag.kind))
        h.log_metrics(metrics)
        h.log_json("leverage.json", field.to_json())
        h.log_json("iv_err_bp.json", err)
        h.log_json("diagnostics.json", to_jsonable({"offline_run": params["offline_run"], "lag": lag.kind,
                                                    "spot_move": lag.spot_move,
                                                    "head": head.params if head else None}))
        return h.run_id
```

Two details the implementer must respect: (a) `_leverage_error` is imported from `bench.runner` by its private name; add `__all__`-free re-export or just import it, it is the same function; (b) `online_s` for head methods is the whole sweep (simulation plus head), which is the latency the user asked for; the stale rows are `0.0` because they need no online particles.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/suite/test_online.py -v` (these take up to a minute: the tiny PDE reference is solved for `li_simple` under the spot lag).
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/neural_particle_method/suite/online.py tests/suite/test_online.py
git commit -m "suite.online: eight lagged methods on cached bodies, lagged PDE references"
```

---

### Task 10: Driver, CLI, and smoke test (`suite/grid.py`, `cli.py`)

**Files:**
- Create: `src/neural_particle_method/suite/grid.py`
- Modify: `src/neural_particle_method/cli.py`
- Test: `tests/suite/test_grid.py`, `tests/suite/test_smoke.py` (marked `slow`), `tests/test_cli.py` (extend)

**Interfaces:**
- Produces:
  ```python
  def cold_jobs(settings) -> list[tuple]      # ("cold", sid, algo, seed)
  def pde_jobs(settings) -> list[tuple]       # ("pde", sid, seed)
  def offline_jobs(settings) -> list[tuple]   # ("offline", sid, body, n)
  def online_jobs(settings) -> list[tuple]    # ("online", sid, method, offline_n, lag_kind, seed)
  def run_job(store, job, settings) -> run_id
  def run_stage(store, stage, settings, n_jobs=1, sids=None) -> (n_done, n_failed)
  STAGES = ("pde", "cold", "offline", "online")
  ```
  CLI: `nparticle suite run --stage {pde,cold,offline,online,all} [--jobs N] [--sids ...] [--smoke]`; `nparticle suite load <run_id> [--out DIR]`; `nparticle suite tables [--out paper/tables]` (tables implemented in Task 11; the CLI wires it there).

- [ ] **Step 1: Failing tests**

`tests/suite/test_grid.py`:

```python
import pytest

from neural_particle_method.suite.config import FULL, SuiteSettings
from neural_particle_method.suite.grid import (
    STAGES,
    cold_jobs,
    offline_jobs,
    online_jobs,
    pde_jobs,
    run_stage,
)
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()


def test_full_job_counts():
    assert len(pde_jobs(FULL)) == 23 * 2
    assert len(cold_jobs(FULL)) == 23 * 9 * 2
    assert len(offline_jobs(FULL)) == 23 * 2 * 2
    assert len(online_jobs(FULL)) == 23 * (6 * 2 + 1 + 1) * 2 * 2     # 6 head/stale methods x 2 sizes, stale_L, nw_resolve; 2 lags; 2 seeds
    assert STAGES == ("pde", "cold", "offline", "online")


def test_sid_filter_and_online_refuses_without_bodies(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    assert {j[1] for j in cold_jobs(TINY, sids=["s01"])} == {"s01"}
    with pytest.raises(RuntimeError, match="offline"):
        run_stage(store, "online", TINY, sids=["s01"])


def test_run_stage_runs_and_skips_finished(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    small = SuiteSettings(**{**TINY.__dict__, "sids": ("s01",)})
    done, failed = run_stage(store, "pde", small)
    assert (done, failed) == (1, 0)
    assert run_stage(store, "pde", small) == (0, 0)
```

`tests/suite/test_smoke.py`:

```python
import pytest

from neural_particle_method.cli import main
from neural_particle_method.suite.config import SMOKE
from neural_particle_method.tracking.store import Store


@pytest.mark.slow
def test_smoke_all_stages_end_to_end(tmp_path):
    uri, root = f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art")
    assert main(["--tracking-uri", uri, "--artifact-root", root, "suite", "run", "--stage", "all", "--smoke"]) == 0
    store = Store(uri, root)
    assert len(store.search(SMOKE.experiment("suite_cold"))) == 2 * 9
    assert len(store.search(SMOKE.experiment("suite_pde_floor"))) == 2
    assert len(store.search(SMOKE.experiment("suite_offline"))) == 2 * 2
    assert len(store.search(SMOKE.experiment("suite_lagged"))) == 2 * 8 * 2
    assert (store.search(SMOKE.experiment("suite_lagged"))["status"] == "FINISHED").all()
```

Append to `tests/test_cli.py`:

```python
def test_suite_load_writes_artifacts(tmp_path):
    from neural_particle_method.suite.cold import run_cold
    from neural_particle_method.suite.config import SuiteSettings
    from neural_particle_method.tracking.store import Store
    uri, root = f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art")
    rid = run_cold(Store(uri, root), "s01", "explicit_nn", 0, SuiteSettings.tiny())
    out = tmp_path / "out"
    assert main(["--tracking-uri", uri, "--artifact-root", root, "suite", "load", rid, "--out", str(out)]) == 0
    assert (out / "leverage.json").exists() and (out / "model.pt").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/suite/test_grid.py tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError` / argparse error on `suite`.

- [ ] **Step 3: Implement**

`src/neural_particle_method/suite/grid.py`:

```python
"""Job lists and the multiprocess driver for the four suite stages."""
from concurrent.futures import ProcessPoolExecutor

from ..tracking.store import Store
from .cold import COLD_ALGOS, run_cold
from .config import FULL
from .lag import LAGS
from .offline import BODIES, run_offline
from .online import ONLINE_METHODS, run_online
from .reference import run_pde_floor

STAGES = ("pde", "cold", "offline", "online")
EXPERIMENTS = {"pde": ("pde_reference", "suite_pde_floor"), "cold": ("suite_cold", "pde_reference"),
               "offline": ("suite_offline",), "online": ("suite_lagged", "suite_offline", "pde_reference")}


def _sids(settings, sids):
    return tuple(settings.sids) if sids is None else tuple(s for s in settings.sids if s in set(sids))


def pde_jobs(settings=FULL, sids=None):
    return [("pde", sid, seed) for sid in _sids(settings, sids) for seed in settings.seeds]


def cold_jobs(settings=FULL, sids=None):
    return [("cold", sid, algo, seed) for sid in _sids(settings, sids) for algo in COLD_ALGOS
            for seed in settings.seeds]


def offline_jobs(settings=FULL, sids=None):
    return [("offline", sid, body, n) for sid in _sids(settings, sids) for body in BODIES
            for n in settings.offline_sizes]


def online_jobs(settings=FULL, sids=None):
    jobs = []
    for sid in _sids(settings, sids):
        for lag in LAGS:
            for seed in settings.seeds:
                for method, (body, _) in ONLINE_METHODS.items():
                    sizes = settings.offline_sizes if body and method != "stale_L" else (settings.offline_sizes[0],)
                    if body is None:
                        sizes = (0,)
                    for n in sizes:
                        jobs.append(("online", sid, method, n, lag.kind, seed))
    return jobs


def run_job(store, job, settings=FULL):
    kind = job[0]
    if kind == "pde":
        return run_pde_floor(store, job[1], job[2], settings)
    if kind == "cold":
        return run_cold(store, job[1], job[2], job[3], settings)
    if kind == "offline":
        return run_offline(store, job[1], job[2], job[3], settings)
    if kind == "online":
        lag = {l.kind: l for l in LAGS}[job[4]]
        return run_online(store, job[1], job[2], job[3], lag, job[5], settings)
    raise KeyError(kind)


def _worker(args):
    uri, root, job, settings = args
    try:
        run_job(Store(uri, root), job, settings)
        return job, None
    except Exception as e:  # noqa: BLE001 - the run is already recorded FAILED by Store.run
        return job, repr(e)


JOBS = {"pde": pde_jobs, "cold": cold_jobs, "offline": offline_jobs, "online": online_jobs}


def run_stage(store, stage, settings=FULL, n_jobs=1, sids=None):
    """Run every unfinished job of `stage`; returns (n_done, n_failed). Pre-creates experiments (MLflow race)."""
    if stage == "online":
        missing = [j for j in offline_jobs(settings, sids)
                   if store.find_finished(settings.experiment("suite_offline"),
                                          {"sid": j[1], "body": j[2], "n_particles": j[3], "seed": 0,
                                           "n_steps": settings.explicit.n_steps}) is None]
        if missing:
            raise RuntimeError(f"{len(missing)} offline bodies unfinished; run --stage offline first")
    for name in EXPERIMENTS[stage]:
        store.experiment_id(settings.experiment(name) if name != "pde_reference" else name)
    jobs = JOBS[stage](settings, sids)
    print(f"{stage}: {len(jobs)} jobs listed", flush=True)
    args = [(store.tracking_uri, store.artifact_root, j, settings) for j in jobs]
    done = failed = 0
    results = ProcessPoolExecutor(max_workers=n_jobs).map(_worker, args) if n_jobs > 1 else map(_worker, args)
    for job, err in results:
        if err is None:
            done += 1
            print("done:", *job, flush=True)
        else:
            failed += 1
            print("FAILED:", *job, err, flush=True)
    return done, failed
```

Note: idempotent stages count already-finished jobs as `done` only if they ran now; to make `test_run_stage_runs_and_skips_finished` pass, filter `jobs` to unfinished ones before running. Add a `_is_finished(store, job, settings)` helper that reproduces each stage's key (`pde`: `{sid, seed, n_steps}` in `suite_pde_floor`; `cold`: `run_key(sid, algo, n_online, seed)` in `suite_cold`; `offline`: as above; `online`: `{sid, method, offline_n, lag, seed, n_steps}` in `suite_lagged`), and use `jobs = [j for j in jobs if not _is_finished(store, j, settings)]`.

`cli.py`: add imports `from .suite.artifacts import load_run`, `from .suite.config import FULL as SUITE_FULL, SMOKE as SUITE_SMOKE`, `from .suite.grid import STAGES, run_stage`, `from .suite.tables import section4_tables` (Task 11 creates it; until then import lazily inside the branch). Parser:

```python
    p = sub.add_parser("suite")
    ss = p.add_subparsers(dest="suite_cmd", required=True)
    q = ss.add_parser("run"); q.add_argument("--stage", required=True, choices=list(STAGES) + ["all"])
    q.add_argument("--jobs", type=int, default=1); q.add_argument("--sids", nargs="*", default=None)
    q.add_argument("--smoke", action="store_true")
    q = ss.add_parser("load"); q.add_argument("run_id"); q.add_argument("--out", default=None)
    q = ss.add_parser("tables"); q.add_argument("--out", default="paper/tables")
```

Handler:

```python
    if args.cmd == "suite":
        if args.suite_cmd == "run":
            settings = SUITE_SMOKE if args.smoke else SUITE_FULL
            stages = list(STAGES) if args.stage == "all" else [args.stage]
            total_failed = 0
            for stage in stages:
                done, failed = run_stage(store, stage, settings, n_jobs=args.jobs, sids=args.sids)
                print(f"{stage}: {done} done, {failed} failed", flush=True)
                total_failed += failed
            return 1 if total_failed else 0
        if args.suite_cmd == "load":
            lr = load_run(store, args.run_id)
            out = Path(args.out or f"results/suite/{args.run_id}")
            out.mkdir(parents=True, exist_ok=True)
            (out / "leverage.json").write_text(json.dumps(lr.field.to_json()))
            (out / "params.json").write_text(json.dumps(lr.params, indent=1))
            (out / "metrics.json").write_text(json.dumps(lr.metrics, indent=1))
            if lr.model is not None:
                torch.save({"kind": lr.meta.get("kind"), "state": lr.model.state()}, out / "model.pt")
            print(f"{args.run_id}: {lr.meta.get('kind', 'field_only')} -> {out}")
            for k in ("pooled_mae_bp", "pooled_rmse_bp", "wings_mae_bp", "fit_s", "online_s"):
                if k in lr.metrics:
                    print(f"  {k} = {lr.metrics[k]:.3f}")
            return 0
        if args.suite_cmd == "tables":
            from .suite.tables import section4_tables
            for pth in section4_tables(store, args.out):
                print("wrote", pth)
            return 0
```

(add `import json`, `import torch` at the top of `cli.py`).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/suite/test_grid.py tests/test_cli.py -v` then `uv run pytest -m slow tests/suite/test_smoke.py -v` (a few minutes: the smoke solves two PDE references at 201x60 for two scenarios and their lags, plus 18 cold, 4 offline and 32 online runs).
Expected: PASS. The `tables` subcommand is exercised in Task 11.

- [ ] **Step 5: Commit**

```bash
git add src/neural_particle_method/suite/grid.py src/neural_particle_method/cli.py tests/suite/test_grid.py tests/suite/test_smoke.py tests/test_cli.py
git commit -m "suite driver: staged job lists, process pool, nparticle suite run/load"
```

---

### Task 11: Section-4 tables (`suite/tables.py`) and the notes

**Files:**
- Create: `src/neural_particle_method/suite/tables.py`, `paper/tables/.gitkeep`
- Modify: `paper/notes_experiments.tex` (section 4 tables become `\input{tables/...}`; Heston block text)
- Test: `tests/suite/test_tables.py`

**Interfaces:**
- Consumes: the four experiments' metric and param names as logged by Tasks 6, 7, 9.
- Produces: `section4_tables(store, out_dir="paper/tables", settings=FULL) -> list[Path]` writing `suite_cold_ssvi.tex`, `suite_cold_heston.tex`, `suite_lagged_ssvi.tex`, `suite_lagged_heston.tex`, `suite_appendix.tex`, each a bare `tabular` (no `table` environment) so the notes control placement and captions.

- [ ] **Step 1: Failing test**

`tests/suite/test_tables.py`:

```python
import pytest

from neural_particle_method.suite.config import FULL
from neural_particle_method.suite.tables import (
    COLD_ROWS,
    LAGGED_ROWS,
    cold_frame,
    lagged_frame,
    section4_tables,
)
from neural_particle_method.tracking.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    common = {"n_particles": 100_000}
    for sid, mae, rmse in (("s01", 10.0, 12.0), ("s02", 30.0, 40.0)):
        for seed in (0, 1):
            with s.run("suite_cold", {"sid": sid, "algo": "nw", "seed": seed, **common}) as h:
                h.log_metrics({"pooled_mae_bp": mae + seed, "pooled_rmse_bp": rmse, "wings_mae_bp": mae * 2,
                               "lev_rmse": 0.05, "fit_s": 1.0, "mae_bp/T0.25": mae})
    with s.run("suite_pde_floor", {"sid": "s01", "seed": 0, "n_steps": 200}) as h:
        h.log_metrics({"pooled_mae_bp": 3.0, "pooled_rmse_bp": 4.0, "wings_mae_bp": 5.0, "fit_s": 0.0, "lev_rmse": 0.0})
    with s.run("suite_lagged", {"sid": "s01", "method": "explicit_rkhs", "offline_n": 200_000, "lag": "surface",
                                "seed": 0, "n_steps": 200}) as h:
        h.log_metrics({"pooled_mae_bp": 20.0, "pooled_rmse_bp": 25.0, "wings_mae_bp": 30.0, "mae_bp/T0.25": 35.0,
                       "online_s": 2.5})
    return s


def test_cold_frame_aggregates_mean_and_median_over_scenarios_and_seeds(store):
    df = cold_frame(store, FULL, family="ssvi")
    nw = df.loc["NW"]
    assert nw["mae_mean"] == pytest.approx((10 + 11 + 30 + 31) / 4) and nw["mae_median"] == pytest.approx(20.5)
    assert nw["lat_median"] == 1.0 and nw["lev_median"] == 0.05
    assert df.loc["PDE (attainable floor)"]["mae_mean"] == 3.0
    assert df.loc["Explicit NN"].isna().all()
    assert list(df.index) == [r[1] for r in COLD_ROWS]


def test_lagged_frame_and_files(store, tmp_path):
    df = lagged_frame(store, FULL, family="ssvi")
    row = df.loc[("surface", "Explicit NN + RKHS head", "200k")]
    assert row["mae_mean"] == 20.0 and row["t025_mean"] == 35.0 and row["lat_median"] == 2.5
    assert ("surface_spot", "NW re-solve on $S_1$", "--") in df.index
    assert len(LAGGED_ROWS) == 8
    paths = section4_tables(store, tmp_path / "tables", FULL)
    names = sorted(p.name for p in paths)
    assert names == ["suite_appendix.tex", "suite_cold_heston.tex", "suite_cold_ssvi.tex",
                     "suite_lagged_heston.tex", "suite_lagged_ssvi.tex"]
    cold = (tmp_path / "tables" / "suite_cold_ssvi.tex").read_text()
    assert "\\begin{tabular}" in cold and "NW & 20 & 20 &" in cold and "Explicit NN & -- & --" in cold
    assert "PDE (attainable floor) & 3 & 3 &" in cold
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/suite/test_tables.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/neural_particle_method/suite/tables.py`:

```python
"""Section-4 tables of paper/notes_experiments.tex from the suite experiments. Bare tabulars."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FULL, HESTON_SIDS, SSVI_SIDS

COLD_ROWS = (("nw", "NW"), ("explicit_nn", "Explicit NN"), ("implicit_nn", "Implicit NN"), ("rkhs", "RKHS"),
             ("spline", "Spline"), ("nw_ghl", "GHL kernel"), ("bins", "Bins"), ("muguruza", "Muguruza"),
             ("purbf", "PU-RBF"), ("pde", "PDE (attainable floor)"))
LAGGED_ROWS = (("stale_L", "Stale leverage, no action"), ("nw_resolve", "NW re-solve on $S_1$"),
               ("explicit_stale", "Explicit NN, stale $f$, fresh Dupire"),
               ("implicit_stale", "Implicit NN, stale $f$, fresh Dupire"),
               ("explicit_rkhs", "Explicit NN + RKHS head"), ("implicit_rkhs", "Implicit NN + RKHS head"),
               ("explicit_ridge", "Explicit NN + ridge head"), ("implicit_ridge", "Implicit NN + ridge head"))
LAG_TITLES = {"surface": "Surface lag", "surface_spot": "Surface plus $2\\%$ sticky-strike spot lag"}


def _sids(family):
    return set(SSVI_SIDS if family == "ssvi" else HESTON_SIDS)


def _finished(store, experiment):
    df = store.search(experiment)
    return df[df["status"] == "FINISHED"] if len(df) else df


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def _agg(g):
    """Per-scenario means over seeds, then mean/median over scenarios."""
    per_sid = g.groupby("params.sid").agg(mae=("mae", "mean"), rmse=("rmse", "mean"), wings=("wings", "mean"),
                                          t025=("t025", "mean"), lev=("lev", "median"), lat=("lat", "median"))
    return pd.Series({"mae_mean": per_sid.mae.mean(), "mae_median": per_sid.mae.median(),
                      "rmse_mean": per_sid.rmse.mean(), "rmse_median": per_sid.rmse.median(),
                      "wings_mean": per_sid.wings.mean(), "wings_median": per_sid.wings.median(),
                      "t025_mean": per_sid.t025.mean(), "lev_median": per_sid.lev.median(),
                      "lat_median": per_sid.lat.median(), "n_sids": len(per_sid)})


def _prep(df):
    out = pd.DataFrame(index=df.index)
    out["params.sid"] = df["params.sid"]
    out["mae"], out["rmse"], out["wings"] = _num(df, "metrics.pooled_mae_bp"), _num(df, "metrics.pooled_rmse_bp"), _num(df, "metrics.wings_mae_bp")
    out["t025"], out["lev"] = _num(df, "metrics.mae_bp/T0.25"), _num(df, "metrics.lev_rmse")
    lat = _num(df, "metrics.online_s") if "metrics.online_s" in df.columns else _num(df, "metrics.fit_s")
    out["lat"] = lat
    return out


EMPTY = pd.Series({k: np.nan for k in ("mae_mean", "mae_median", "rmse_mean", "rmse_median", "wings_mean",
                                        "wings_median", "t025_mean", "lev_median", "lat_median", "n_sids")})


def cold_frame(store, settings=FULL, family="ssvi"):
    cold = _finished(store, settings.experiment("suite_cold"))
    pde = _finished(store, settings.experiment("suite_pde_floor"))
    rows = {}
    for algo, label in COLD_ROWS:
        src = pde if algo == "pde" else cold
        if len(src) == 0:
            rows[label] = EMPTY.copy()
            continue
        sel = src[src["params.sid"].isin(_sids(family))]
        if algo != "pde":
            sel = sel[sel["params.algo"] == algo]
        rows[label] = _agg(_prep(sel)) if len(sel) else EMPTY.copy()
    return pd.DataFrame(rows).T


def lagged_frame(store, settings=FULL, family="ssvi"):
    lagged = _finished(store, settings.experiment("suite_lagged"))
    rows = {}
    for lag in ("surface", "surface_spot"):
        for method, label in LAGGED_ROWS:
            sizes = ("--",) if method in ("stale_L", "nw_resolve") else tuple(f"{n // 1000}k" for n in settings.offline_sizes)
            for size in sizes:
                key = (lag, label, size)
                if len(lagged) == 0:
                    rows[key] = EMPTY.copy()
                    continue
                sel = lagged[(lagged["params.sid"].isin(_sids(family))) & (lagged["params.method"] == method)
                             & (lagged["params.lag"] == lag)]
                if size != "--":
                    sel = sel[pd.to_numeric(sel["params.offline_n"]) == int(size[:-1]) * 1000]
                rows[key] = _agg(_prep(sel)) if len(sel) else EMPTY.copy()
    df = pd.DataFrame(rows).T
    df.index = pd.MultiIndex.from_tuples(df.index, names=["lag", "method", "size"])
    return df


def _cell(x, fmt="{:.0f}"):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)


def _cold_tex(df):
    lines = ["\\begin{tabular}{l rr rr rr r r}", "\\toprule",
             " & \\multicolumn{2}{c}{pooled MAE (bp)} & \\multicolumn{2}{c}{pooled RMSE (bp)} & "
             "\\multicolumn{2}{c}{wings MAE (bp)} & lev.\\ RMSE & latency (s)\\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             "method & mean & median & mean & median & mean & median & median & median\\\\", "\\midrule"]
    for label, r in df.iterrows():
        cells = [_cell(r.mae_mean), _cell(r.mae_median), _cell(r.rmse_mean), _cell(r.rmse_median),
                 _cell(r.wings_mean), _cell(r.wings_median), _cell(r.lev_median, "{:.3f}"), _cell(r.lat_median, "{:.1f}")]
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _lagged_tex(df):
    lines = ["\\begin{tabular}{l l rr rr r}", "\\toprule",
             "method & offline $N$ & pooled MAE & pooled RMSE & wings MAE & $T=0.25$ MAE & online latency (s)\\\\",
             "\\midrule"]
    for lag in ("surface", "surface_spot"):
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{LAG_TITLES[lag]}}}}}\\\\")
        for (lg, label, size), r in df.iterrows():
            if lg != lag:
                continue
            cells = [_cell(r.mae_mean), _cell(r.rmse_mean), _cell(r.wings_mean), _cell(r.t025_mean),
                     _cell(r.lat_median, "{:.1f}")]
            lines.append(f"{label} & {size} & " + " & ".join(cells) + "\\\\")
        if lag == "surface":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _appendix_tex(store, settings):
    """Per-scenario pooled MAE (seeds averaged): one block per experiment, scenarios as rows."""
    blocks = []
    cold = _finished(store, settings.experiment("suite_cold"))
    if len(cold):
        p = _prep(cold)
        p["algo"] = cold["params.algo"]
        piv = p.groupby(["params.sid", "algo"]).mae.mean().unstack("algo")
        cols = [a for a, _ in COLD_ROWS if a in piv.columns]
        piv = piv.reindex(columns=cols)
        head = " & ".join(dict(COLD_ROWS)[a] for a in cols)
        lines = ["\\begin{tabular}{l " + "r" * len(cols) + "}", "\\toprule", f"sid & {head}\\\\", "\\midrule"]
        for sid, r in piv.iterrows():
            lines.append(f"{sid} & " + " & ".join(_cell(v) for v in r.values) + "\\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        blocks.append("% cold suite, pooled MAE bp\n" + "\n".join(lines))
    lagged = _finished(store, settings.experiment("suite_lagged"))
    if len(lagged):
        p = _prep(lagged)
        p["col"] = lagged["params.method"] + "/" + lagged["params.offline_n"].astype(str) + "/" + lagged["params.lag"]
        piv = p.groupby(["params.sid", "col"]).mae.mean().unstack("col")
        lines = ["\\begin{tabular}{l " + "r" * len(piv.columns) + "}", "\\toprule",
                 "sid & " + " & ".join(c.replace("_", "\\_") for c in piv.columns) + "\\\\", "\\midrule"]
        for sid, r in piv.iterrows():
            lines.append(f"{sid} & " + " & ".join(_cell(v) for v in r.values) + "\\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        blocks.append("% lagged suite, pooled MAE bp\n" + "\n".join(lines))
    return "\n\n".join(blocks) + "\n" if blocks else "% no finished suite runs\n"


def section4_tables(store, out_dir="paper/tables", settings=FULL):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {"suite_cold_ssvi.tex": _cold_tex(cold_frame(store, settings, "ssvi")),
             "suite_cold_heston.tex": _cold_tex(cold_frame(store, settings, "heston")),
             "suite_lagged_ssvi.tex": _lagged_tex(lagged_frame(store, settings, "ssvi")),
             "suite_lagged_heston.tex": _lagged_tex(lagged_frame(store, settings, "heston")),
             "suite_appendix.tex": _appendix_tex(store, settings)}
    paths = []
    for name, text in files.items():
        (out / name).write_text(text)
        paths.append(out / name)
    return paths
```

Then in `paper/notes_experiments.tex`, section 4: replace the body of the `table` environment labelled `tab:suite-cold` (the `\begin{tabular}...\end{tabular}`) with `\input{tables/suite_cold_ssvi.tex}`; add directly after it

```latex
\begin{table}[h]\centering\footnotesize
\caption{Cold suite, Heston-market scenarios (one-year maturity, same $13$ quotes): mean and median over the three scenarios and two seeds. TBD.}\label{tab:suite-cold-heston}
\input{tables/suite_cold_heston.tex}
\end{table}
```

replace the `\resizebox{\textwidth}{!}{\begin{tabular}...\end{tabular}}` of `tab:suite-lagged` with `\resizebox{\textwidth}{!}{\input{tables/suite_lagged_ssvi.tex}}`, add a Heston lagged table after it in the same style (`\label{tab:suite-lagged-heston}`, caption "Lagged suite, Heston-market scenarios. TBD."), and add before `\subsection{Reproducibility}` an appendix-style paragraph:

```latex
\paragraph{Per-scenario numbers.} \input{tables/suite_appendix.tex}
```

Change the sentence "The Heston block is reported separately because it scores a single maturity on a wider strike range." to "The Heston block is reported separately because it scores a single one-year maturity (the same $13$ quoted strikes)." and delete "The same table is repeated for the three Heston-market scenarios (one maturity, no wings split), and the per-scenario numbers behind both go in an appendix table." Run `uv run nparticle suite tables` once (empty store gives `--` cells) so the `\input` files exist, commit `paper/tables/*.tex`, and rebuild: `cd paper && pdflatex -interaction=nonstopmode notes_experiments.tex && pdflatex -interaction=nonstopmode notes_experiments.tex`; check the log has no `undefined` and no `Overfull` above 10pt.

- [ ] **Step 4: Run tests and build**

Run: `uv run pytest tests/suite/test_tables.py -v && uv run ruff check src tests && uv run pytest -q`
Expected: PASS, ruff clean, full default suite green. PDF builds with the placeholder tables.

- [ ] **Step 5: Commit**

```bash
git add src/neural_particle_method/suite/tables.py tests/suite/test_tables.py paper/tables paper/notes_experiments.tex paper/notes_experiments.pdf
git commit -m "suite.tables: section-4 LaTeX tables from the store; notes input them"
```

---

### Task 12: Launch the full suite (operator task, not code)

**Files:** none. Logs to `$CLAUDE_JOB_DIR/tmp/suite_<stage>.log` or `results/suite_<stage>.log`.

- [ ] **Step 1: References and cold, 4 workers**

Run (background, from repo root):
```bash
uv run nparticle suite run --stage pde --jobs 4 > results/suite_pde.log 2>&1
uv run nparticle suite run --stage cold --jobs 4 > results/suite_cold.log 2>&1
```
Expected: `pde: 46 done, 0 failed`; `cold: 414 done, 0 failed` (about 2.5 h). Any `FAILED:` line is investigated from the run's `traceback.txt` artifact before moving on.

- [ ] **Step 2: Offline bodies, 4 workers, overnight**

```bash
uv run nparticle suite run --stage offline --jobs 4 > results/suite_offline.log 2>&1
```
Expected: `offline: 92 done, 0 failed` (about 12 h).

- [ ] **Step 3: Online**

```bash
uv run nparticle suite run --stage online --jobs 4 > results/suite_online.log 2>&1
```
Expected: `online: 1288 done, 0 failed` (under 1 h; the count is 23 scenarios x 14 rows x 2 lags x 2 seeds).

- [ ] **Step 4: Tables and notes**

```bash
uv run nparticle suite tables && cd paper && pdflatex -interaction=nonstopmode notes_experiments.tex && pdflatex -interaction=nonstopmode notes_experiments.tex
```
Then remove "(planned; tables to be filled)" from the section-4 title and the "TBD" from the four captions, put the PDE seed-to-seed noise floor (from `suite_pde_floor`, seed 0 vs 1 pooled MAE, median over scenarios) in the cold caption, rebuild, and commit `paper/tables/*.tex`, the tex and the pdf with message `Suite results: section-4 tables filled`.

---

## Self-review

**Spec coverage.** Floor default and goldens: Task 1. Hooks (`keep_slice_weights`, `CalibResult.model`, `save_model`, `n_steps`-keyed reference, `scenario` argument): Tasks 2 and 3. Lag with SSVI and Heston bumps, sticky-strike shift, `as_params`: Task 4. Artifacts and `load_run`, CLI `suite load`: Tasks 5 and 10. Cold suite with nine algorithms and saved models, PDE floor with two seeds: Task 6. Offline bodies with anchor scores: Task 7. RKHS and ridge heads, causal sweep, stale-f: Task 8. Eight online methods, lagged references, online-only timing: Task 9. Driver with pre-created experiments, failure logging, online refusing without bodies, smoke settings and slow smoke test: Task 10. Tables with mean/median, appendix, `\input` in the notes, Heston block: Task 11. The run itself: Task 12.

**Deviations from the spec, deliberate.** (1) The Heston cold table keeps the wings columns because the Heston scenarios are scored on the same 13 quotes, so wings are defined; the notes text is corrected in Task 11. (2) `SuiteSettings.tiny()` is added for unit tests (the spec only named `--smoke`). (3) `run_reference` also logs `mass` and `forward` so the x-grid mass check the spec asks for is visible in the store; a scenario whose reference fails simply has no `lev_rmse`, and the PDE floor run for it is FAILED, which the appendix shows as `--`.

**Type consistency.** `score_field(field, sc, seed, reprice) -> (metrics, err)` is defined in Task 6 and used in Tasks 7 and 9. `save_model(h, model, meta) -> kind` and `load_run(store, run_id) -> LoadedRun` (Task 5) are used in Tasks 6, 7, 9, 10. `online_sweep(model, local_vol, params, s0, T, cfg, head, seed) -> (field, fit_s)` (Task 8) is used in Task 9. `run_reference(store, sid, n_steps, n_x, n_v, scenario, lag)` (Task 3) is used in Tasks 6 and 9. `_leverage_error(store, sid, field, n_steps, lag)` (Task 3) is used by `run_one` and Task 9. `Lag.kind` strings `"surface"`/`"surface_spot"` are used identically in Tasks 4, 9, 10, 11.
