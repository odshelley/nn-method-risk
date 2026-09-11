# Optuna Offline-Body Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An Optuna search over the per-slice network recipe (knobs plus structural priors) on held-out tuning clouds, tracked in MLflow, with validation of the top recipes as frozen bodies plus spline head at 80k online particles, and promotion of the winner into the suite as a cold row and a body.

**Architecture:** New estimator options on `NNRegressor` and a `tail` rule in the explicit pass, all default-off and bit-for-bit. A `recipes` module is the single place a recipe becomes a regressor and an `ExplicitConfig`. Tuning clouds are MLflow runs (`optuna_clouds`); the study is one parent run with nested trial runs (`optuna_offline`); the Optuna SQLite storage is an artifact. Validation reuses `run_offline` (body kind `explicit_opt`) and a package-level budget cell (`suite/budget.py`, extracted from `scripts/budget_sweep.py`).

**Tech Stack:** Python 3.12, numpy, torch, optuna (new dependency), mlflow, pytest, `uv run`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-11-optuna-offline-design.md`.
- Every new option defaults to today's behaviour bit for bit: `uv run pytest -q -m golden` green and `git status tests/golden` clean after every task.
- Line length ≤ 100 in every touched `.py` file, checked with `awk 'length > 100 {print FILENAME": "FNR}' <files>` (ruff does not enforce it). `uv run ruff check src tests scripts` clean.
- Tuning registry: 24 scenarios, ids `t01`…`t24`, seed 5000, at least 8 with Feller ratio < 0.3 and 8 with > 0.7, never in `full_registry()`.
- Cloud slice times `(0.10, 0.50, 1.00, 1.75)`; fit pool = first 80 % of the recorded (already permuted) particles, held-out = last 20 %; NW held-out loss metrics `nw_loss/t<t:g>`.
- Trial score = mean over slices and scenarios of `log(held-out MSE of the recipe / held-out MSE of NW)`; 8 scenarios per trial drawn with `np.random.default_rng(trial.number)`; `MedianPruner(n_startup_trials=10, n_warmup_steps=1)`; `TPESampler(seed=0)`; direction minimize.
- Experiments: `optuna_clouds`, `optuna_offline` (tags `optuna.kind` ∈ {`study`, `trial`}, `optuna.state` ∈ {COMPLETE, PRUNED, FAIL}); validation bodies in `suite_offline` (body `explicit_opt`, key gains `recipe_hash`), head cells in `suite_budget_tuned` (method `explicit_opt_spline`, budget 80000, key gains `recipe_hash`).
- Recipe keys and distributions exactly as in spec §2; `batch_size` 0 means full batch; `monotone_sign` is `-sign(rho)` (penalise slopes of that sign), computed from the scenario, never stored in the recipe.
- Promoted recipe file: `src/neural_particle_method/estimators/recipes/explicit_opt.json`; cold algorithm `explicit_nn_opt`; body kind `explicit_opt`; online methods `explicit_opt_stale`, `explicit_opt_spline`; table labels "Explicit NN, searched" and "Searched body + spline head".
- Commit messages end with the two trailer lines:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01EhbyGqFA5kYDyMwYouSgvS`.
- No full-size run from any task; the controller runs the study.

---

## File structure

- Modify `src/neural_particle_method/bench/scenarios.py`: `feller_ratio`, `make_tuning_registry`.
- Modify `src/neural_particle_method/estimators/nadaraya_watson.py`: `nw_local_variance`.
- Modify `src/neural_particle_method/estimators/nn.py`: `SliceNet.log_scale` buffer; `NNRegressor` knobs `weight_decay`, `warm_start`, `mean_match`, `monotone_penalty`, `monotone_sign`, `hetero`.
- Create `src/neural_particle_method/estimators/recipes.py`: `RECIPE_TYPES`, `recipe_hash`, `regressor_from_recipe`, `explicit_config_from_recipe`, `load_recipe`, `save_recipe`, `recipe_exists`, `RECIPE_DIR`.
- Create `src/neural_particle_method/estimators/recipes/` (package data dir, `.gitkeep`).
- Modify `src/neural_particle_method/calibrate/config.py`: `ExplicitConfig.tail`.
- Modify `src/neural_particle_method/calibrate/explicit.py`: `extend_tail`, applied per slice.
- Modify `src/neural_particle_method/suite/artifacts.py`: `SliceBank.from_state` loads with `strict=False`.
- Create `src/neural_particle_method/suite/optuna_clouds.py`: `RecordingNW`, `Cloud`, `ensure_cloud`, `load_cloud`, `heldout_loss`, `quantile_grid`.
- Create `src/neural_particle_method/suite/optuna_search.py`: `suggest_recipe`, `fit_chain`, `score_recipe`, `run_study`, `top_recipes`.
- Create `src/neural_particle_method/suite/budget.py`: `run_budget_cell`, `budget_methods`, `head_for` (moved from the script).
- Modify `scripts/budget_sweep.py`: thin wrapper over `suite/budget.py`, gains `--body explicit_opt`.
- Create `src/neural_particle_method/suite/optuna_validate.py`: `validate`, `promote`, `comparison_table`.
- Modify `src/neural_particle_method/suite/offline.py`: body `explicit_opt`, `recipe` argument.
- Modify `src/neural_particle_method/suite/online.py`, `grid.py`: `explicit_opt_*` methods gated on the recipe file.
- Modify `src/neural_particle_method/bench/algos.py`: `explicit_nn_opt`.
- Modify `src/neural_particle_method/suite/config.py`, `tables.py`: rows and `COLD_ALGOS` gated on the recipe file.
- Modify `src/neural_particle_method/cli.py`: `nparticle optuna {clouds,run,validate,promote}`.
- Tests: `tests/bench/test_tuning_registry.py`, `tests/estimators/test_nn_priors.py`, `tests/estimators/test_recipes.py`, `tests/calibrate/test_tail.py`, `tests/suite/test_optuna_clouds.py`, `tests/suite/test_optuna_search.py`, `tests/suite/test_budget.py`, `tests/suite/test_optuna_validate.py`, plus edits to `tests/suite/test_offline.py`, `test_online.py`, `test_grid.py`, `test_tables.py`, `tests/scripts/test_budget_sweep.py`.

---

### Task 1: Tuning registry

**Files:**
- Modify: `src/neural_particle_method/bench/scenarios.py` (after `make_registry`)
- Test: `tests/bench/test_tuning_registry.py`

**Interfaces:**
- Produces: `feller_ratio(dyn: HestonParams) -> float`; `make_tuning_registry(n=24, seed=5000) -> dict[str, ScenarioSpec]` with ids `t01`…`t{n:02d}`; `TUNING_SIDS = tuple(make_tuning_registry())` exported from `suite/config.py` in Task 5.

- [ ] **Step 1: Write the failing test**

```python
# tests/bench/test_tuning_registry.py
from neural_particle_method.bench.scenarios import feller_ratio, full_registry, make_tuning_registry


def test_tuning_registry_is_stratified_deterministic_and_disjoint():
    reg = make_tuning_registry()
    assert list(reg) == [f"t{i:02d}" for i in range(1, 25)]
    fr = [feller_ratio(s.dynamics) for s in reg.values()]
    assert sum(f < 0.3 for f in fr) == 8 and sum(f > 0.7 for f in fr) == 8
    assert all(0.3 <= f <= 0.7 for f in fr if not (f < 0.3 or f > 0.7))
    again = make_tuning_registry()
    assert all(reg[k].ssvi == again[k].ssvi and reg[k].dynamics == again[k].dynamics for k in reg)
    assert not set(reg) & set(full_registry())


def test_feller_ratio():
    from neural_particle_method.simulate.dynamics import HestonParams
    assert feller_ratio(HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.5, v0=0.04)) == 1.0
```

- [ ] **Step 2: Run it**

Run: `uv run pytest -q tests/bench/test_tuning_registry.py`
Expected: FAIL, `ImportError: cannot import name 'feller_ratio'`.

- [ ] **Step 3: Implement**

Add after `make_registry` in `bench/scenarios.py`:

```python
def feller_ratio(dyn):
    return 2.0 * dyn.kappa * dyn.theta / dyn.xi ** 2


TUNING_STRATA = ((0.0, 0.3), (0.3, 0.7), (0.7, float("inf")))   # Feller ratio bands


def make_tuning_registry(n=24, seed=5000):
    """A held-out set for estimator tuning: the registry's parameter ranges, stratified so that
    each Feller band gets n/3 scenarios. Deterministic in `seed`; ids t01..t{n}."""
    quota = [n // 3] * 3
    quota[1] += n - sum(quota)
    picked, i = [], 0
    while len(picked) < n:
        rng = np.random.default_rng(seed + i)
        i += 1
        if i > 100_000:
            raise RuntimeError("tuning registry: strata not fillable")
        p = _draw_ssvi(rng)
        dyn = HestonParams(kappa=float(rng.choice(KAPPAS)), theta=p.sigma0 ** 2,
                           xi=float(rng.choice(XIS)), rho=float(rng.choice(RHOS)),
                           v0=p.sigma0 ** 2)
        fr = feller_ratio(dyn)
        band = next(j for j, (lo, hi) in enumerate(TUNING_STRATA) if lo <= fr < hi
                    or (j == 2 and fr >= lo))
        if quota[band] == 0:
            continue
        quota[band] -= 1
        picked.append((p, dyn))
    return {f"t{k:02d}": ScenarioSpec(f"t{k:02d}", p, dyn) for k, (p, dyn) in
            enumerate(picked, start=1)}
```

Check that `ScenarioSpec`, `HestonParams`, `KAPPAS`, `XIS`, `RHOS`, `_draw_ssvi` are the names already in the file (they are, lines 13 and 81–98).

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q tests/bench/test_tuning_registry.py`
Expected: PASS. Note: `fr == 0.3` falls in band 1 and `fr == 0.7` in band 1 by the `lo <= fr < hi` rule; the test's third assertion uses closed bounds, consistent with that.

- [ ] **Step 5: Commit**

```bash
git add src/neural_particle_method/bench/scenarios.py tests/bench/test_tuning_registry.py
git commit -m "scenarios: held-out tuning registry, stratified by Feller ratio"
```

---

### Task 2: Estimator priors on `NNRegressor`

**Files:**
- Modify: `src/neural_particle_method/estimators/nn.py`
- Modify: `src/neural_particle_method/estimators/nadaraya_watson.py` (append)
- Modify: `src/neural_particle_method/suite/artifacts.py` (`SliceBank.from_state`)
- Test: `tests/estimators/test_nn_priors.py`

**Interfaces:**
- Produces: `NNRegressor(seed=0, first_steps=400, later_steps=120, hidden=64, keep_slice_weights=False, depth=2, lr=1e-2, batch_size=None, weight_decay=0.0, warm_start=True, mean_match=False, monotone_penalty=0.0, monotone_sign=1.0, hetero=False)`; `SliceNet` has a buffer `log_scale` (shape `(1,)`, zeros) multiplying the output by `exp(log_scale)`; `nw_local_variance(lnx, v, weights=None, bandwidth=None, n_grid=101) -> np.ndarray` (per-particle conditional variance of `v` given `lnx`, floored at `1e-8`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/estimators/test_nn_priors.py
import numpy as np
import pytest
import torch

from neural_particle_method.estimators.nadaraya_watson import nw_local_variance
from neural_particle_method.estimators.nn import NNRegressor, SliceNet

rng = np.random.default_rng(0)
LNX = rng.normal(0.0, 0.3, 300)
V = 0.04 * np.exp(-2.0 * LNX) * (1 + 0.3 * rng.normal(size=300))   # decreasing target
GRID = np.linspace(-0.8, 0.8, 41)


def test_defaults_are_bit_for_bit():
    a = NNRegressor(seed=0, first_steps=5, later_steps=2).fit_predict(0.5, LNX, V, GRID)
    b = NNRegressor(seed=0, first_steps=5, later_steps=2, weight_decay=0.0, warm_start=True,
                    mean_match=False, monotone_penalty=0.0, hetero=False).fit_predict(
        0.5, LNX, V, GRID)
    np.testing.assert_array_equal(a, b)


def test_log_scale_buffer_defaults_to_identity_and_loads_old_state():
    net = SliceNet()
    assert "log_scale" in net.state_dict() and float(net.log_scale) == 0.0
    old = {k: v for k, v in net.state_dict().items() if k != "log_scale"}
    fresh = SliceNet()
    fresh.load_state_dict(old, strict=False)
    z = torch.zeros(3, 1)
    torch.testing.assert_close(fresh(z), net(z))


def test_mean_match_reproduces_the_sample_mean():
    est = NNRegressor(seed=0, first_steps=20, later_steps=5, mean_match=True)
    est.fit_predict(0.5, LNX, V, GRID)
    assert abs(est.predict(LNX).mean() - V.mean()) < 1e-6 * V.mean()


def test_mean_match_is_weighted():
    w = rng.uniform(0.5, 1.5, 300)
    est = NNRegressor(seed=0, first_steps=20, later_steps=5, mean_match=True)
    est.fit_predict(0.5, LNX, V, GRID, weights=w)
    assert abs(np.average(est.predict(LNX), weights=w) - np.average(V, weights=w)) < 1e-6 * V.mean()


def test_monotone_penalty_is_zero_on_a_decreasing_fit_and_positive_on_an_increasing_one():
    dec = NNRegressor(seed=0, first_steps=200, later_steps=5, monotone_penalty=1.0,
                      monotone_sign=1.0)
    dec.fit_predict(0.5, LNX, V, GRID)
    assert dec.last_penalty < 1e-6
    inc = NNRegressor(seed=0, first_steps=200, later_steps=5, monotone_penalty=1.0,
                      monotone_sign=-1.0)
    inc.fit_predict(0.5, LNX, V, GRID)
    assert inc.last_penalty > 1e-4


def test_warm_start_off_refits_from_scratch():
    cold = NNRegressor(seed=0, first_steps=5, later_steps=2, warm_start=False)
    a = cold.fit_predict(0.5, LNX, V, GRID)
    b = cold.fit_predict(0.6, LNX, V, GRID)
    fresh = NNRegressor(seed=0, first_steps=5, later_steps=2).fit_predict(0.5, LNX, V, GRID)
    np.testing.assert_array_equal(a, fresh)
    assert not np.array_equal(a, b)        # re-initialised with a different seed offset
    assert cold._n_fits == 2


def test_hetero_weights_are_positive_with_mean_one():
    var = nw_local_variance(LNX, V)
    assert var.shape == (300,) and (var > 0).all()
    est = NNRegressor(seed=0, first_steps=3, later_steps=1, hetero=True)
    est.fit_predict(0.5, LNX, V, GRID)
    w = est.last_weights
    assert w.shape == (300,) and (w > 0).all() and abs(w.mean() - 1.0) < 1e-9


def test_weight_decay_reaches_the_optimiser():
    est = NNRegressor(weight_decay=1e-4)
    assert est.opt.param_groups[0]["weight_decay"] == pytest.approx(1e-4)
```

- [ ] **Step 2: Run them**

Run: `uv run pytest -q tests/estimators/test_nn_priors.py`
Expected: FAIL (`ImportError: nw_local_variance`, then `TypeError` on the new keyword arguments).

- [ ] **Step 3: Implement `nw_local_variance`**

Append to `estimators/nadaraya_watson.py`:

```python
def nw_local_variance(lnx, v, weights=None, bandwidth=None, n_grid=101):
    """Conditional variance of v given lnx at every particle: NW estimates of E[v|x] and E[v^2|x]
    on a quantile grid, interpolated back to the particles, floored at 1e-8."""
    grid = np.unique(np.quantile(lnx, np.linspace(0.001, 0.999, n_grid)))
    m1 = nw_estimate(lnx, v, grid, weights=weights, bandwidth=bandwidth)
    m2 = nw_estimate(lnx, v ** 2, grid, weights=weights, bandwidth=bandwidth)
    var = np.clip(m2 - m1 ** 2, 1e-8, None)
    return np.interp(lnx, grid, var)
```

- [ ] **Step 4: Implement the `NNRegressor` knobs**

Replace `SliceNet` and `NNRegressor.__init__`, `fit`, `predict`, `fit_predict` in `estimators/nn.py` with:

```python
class SliceNet(nn.Module):
    def __init__(self, hidden=64, depth=2):
        super().__init__()
        layers = []
        in_dim = 1
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)
        # a post-hoc multiplier (mean matching); a buffer, not a parameter, so training is unchanged
        self.register_buffer("log_scale", torch.zeros(1))

    def forward(self, z):
        return nn.functional.softplus(self.head(self.body(z))) * V_SCALE * torch.exp(self.log_scale)
```

```python
class NNRegressor:
    """...(keep the existing docstring)...

    Priors (all off by default, every default bit-for-bit with the plain regressor):
    `weight_decay` (Adam L2), `warm_start=False` re-initialises the network before every slice
    and trains it `first_steps` steps, `mean_match` rescales the fitted slice so its weighted
    mean over the fit sample equals the weighted mean of the targets, `monotone_penalty` adds
    lambda * mean(relu(monotone_sign * df/dz))^2 / V_SCALE^2 to the loss (penalising slopes of
    sign `monotone_sign`), `hetero` weights the loss by 1 / local variance of the target.
    """
    supports_weights = True

    def __init__(self, seed=0, first_steps=400, later_steps=120, hidden=64,
                 keep_slice_weights=False, depth=2, lr=1e-2, batch_size=None, weight_decay=0.0,
                 warm_start=True, mean_match=False, monotone_penalty=0.0, monotone_sign=1.0,
                 hetero=False):
        self.seed, self.hidden, self.depth, self.lr = seed, hidden, depth, lr
        self.weight_decay = weight_decay
        self._build(seed)
        self.first_steps, self.later_steps = first_steps, later_steps
        self._n_fits = 0
        self.keep_slice_weights = keep_slice_weights
        self.slice_weights = []   # [(t, state_dict copy)] per fitted slice when keep_slice_weights
        self.batch_size = batch_size
        self._gen = torch.Generator().manual_seed(seed)
        self.warm_start, self.mean_match, self.hetero = warm_start, mean_match, hetero
        self.monotone_penalty, self.monotone_sign = monotone_penalty, monotone_sign
        self.last_penalty, self.last_weights = 0.0, None

    def _build(self, seed):
        torch.manual_seed(seed)
        self.net = SliceNet(self.hidden, self.depth)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)

    def reset(self):
        """Zero the fit counter so the next `fit_predict` call again uses `first_steps`."""
        self._n_fits = 0

    def fit(self, lnx, v, steps=120, weights=None):
        z = torch.tensor(lnx[:, None] / Z_SCALE, dtype=torch.float32)
        tv = torch.tensor(v[:, None], dtype=torch.float32)
        tw = None if weights is None else torch.tensor(weights[:, None], dtype=torch.float32)
        n = z.shape[0]
        pen = torch.zeros(())
        for _ in range(steps):
            if self.batch_size is None:
                zb, tvb, twb = z, tv, tw
            else:
                idx = torch.randint(0, n, (self.batch_size,), generator=self._gen)
                zb, tvb = z[idx], tv[idx]
                twb = None if tw is None else tw[idx]
            self.opt.zero_grad()
            if self.monotone_penalty > 0:
                zb = zb.clone().requires_grad_(True)
                out = self.net(zb)
                slope = torch.autograd.grad(out.sum(), zb, create_graph=True)[0]
                pen = (torch.relu(self.monotone_sign * slope) ** 2).mean() / V_SCALE ** 2
            else:
                out = self.net(zb)
            resid = (out - tvb) ** 2
            loss = (resid if twb is None else twb * resid).mean() + self.monotone_penalty * pen
            loss.backward()
            self.opt.step()
        self.last_penalty = float(pen.detach())
        return float(loss.detach())

    def predict(self, lnx_grid):
        with torch.no_grad():
            z = torch.tensor(lnx_grid[:, None] / Z_SCALE, dtype=torch.float32)
            return self.net(z).numpy()[:, 0]

    def fit_predict(self, t, lnx, v, grid, weights=None):
        if not self.warm_start and self._n_fits > 0:
            self._build(self.seed + self._n_fits)
        steps = self.first_steps if (self._n_fits == 0 or not self.warm_start) else self.later_steps
        self._n_fits += 1
        w = weights
        if self.hetero:
            wh = 1.0 / nw_local_variance(lnx, v, weights=weights)
            wh = wh / wh.mean()
            w = wh if weights is None else weights * wh
        self.last_weights = w
        self.fit(lnx, v, steps=steps, weights=w)
        if self.mean_match:
            pred = self.predict(lnx)
            wm = np.ones_like(v) if weights is None else weights
            ratio = np.average(v, weights=wm) / max(np.average(pred, weights=wm), 1e-12)
            with torch.no_grad():
                self.net.log_scale += float(np.log(max(ratio, 1e-12)))
        if self.keep_slice_weights:
            snap = {k: v_.detach().clone() for k, v_ in self.net.state_dict().items()}
            self.slice_weights.append((float(t), snap))
        return self.predict(grid)
```

Add `import numpy as np` and `from .nadaraya_watson import nw_local_variance` at the top of `nn.py`. Note `mean_match` uses the *loss* weights `weights` (not the hetero weights) for the averages, so the matched mean is the physical-measure mean. `torch.manual_seed(seed)` moved into `_build` is still called exactly once at the same point for the default path, so initialisation is unchanged.

- [ ] **Step 5: `SliceBank.from_state` tolerates the new buffer**

In `suite/artifacts.py::SliceBank.from_state`, where each stored state dict is loaded into a `SliceNet`, change `net.load_state_dict(sd)` to `net.load_state_dict(sd, strict=False)` (so stored banks without `log_scale` load, with the buffer at zero). Check `GlobalNetModel` is untouched.

- [ ] **Step 6: Run the tests, the goldens and the bank round trip**

Run: `uv run pytest -q tests/estimators/test_nn_priors.py tests/estimators tests/suite/test_artifacts.py && uv run pytest -q -m golden && git status --short tests/golden`
Expected: all PASS; no golden file modified. If `test_monotone_penalty...` is flaky on the increasing side, raise `first_steps` to 400 in that test rather than loosening the threshold.

- [ ] **Step 7: Commit**

```bash
git add src/neural_particle_method/estimators/nn.py src/neural_particle_method/estimators/nadaraya_watson.py src/neural_particle_method/suite/artifacts.py tests/estimators/test_nn_priors.py
git commit -m "NNRegressor: weight decay, cold restarts, mean matching, monotone penalty, hetero loss"
```

---

### Task 3: Tail rule in the explicit pass

**Files:**
- Modify: `src/neural_particle_method/calibrate/config.py` (`ExplicitConfig`)
- Modify: `src/neural_particle_method/calibrate/explicit.py`
- Test: `tests/calibrate/test_tail.py`

**Interfaces:**
- Produces: `ExplicitConfig.tail: str = "flat"`; `extend_tail(grid, f_grid, tail, estimator=None, dx=TAIL_DX) -> (grid, f_grid)` with `TAIL_DX = 0.5`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/calibrate/test_tail.py
import numpy as np
import pytest

from neural_particle_method.calibrate.config import ExplicitConfig
from neural_particle_method.calibrate.explicit import TAIL_DX, extend_tail

GRID = np.array([-0.2, 0.0, 0.2])
F = np.array([0.06, 0.04, 0.03])


def test_flat_is_a_no_op():
    g, f = extend_tail(GRID, F, "flat")
    np.testing.assert_array_equal(g, GRID)
    np.testing.assert_array_equal(f, F)


def test_linear_continues_with_the_end_slopes():
    g, f = extend_tail(GRID, F, "linear")
    assert g[0] == pytest.approx(-0.2 - TAIL_DX) and g[-1] == pytest.approx(0.2 + TAIL_DX)
    assert f[0] == pytest.approx(0.06 + (0.06 - 0.04) / 0.2 * TAIL_DX)
    assert f[-1] == pytest.approx(0.03 + (0.03 - 0.04) / 0.2 * TAIL_DX)
    np.testing.assert_array_equal(f[1:-1], F)


def test_free_asks_the_estimator():
    class Est:
        def predict(self, x):
            return 0.01 + 0.0 * x
    g, f = extend_tail(GRID, F, "free", Est())
    assert f[0] == pytest.approx(0.01) and f[-1] == pytest.approx(0.01)


def test_unknown_tail_and_default():
    assert ExplicitConfig().tail == "flat"
    with pytest.raises(ValueError):
        extend_tail(GRID, F, "quadratic")
```

- [ ] **Step 2: Run them**

Run: `uv run pytest -q tests/calibrate/test_tail.py`
Expected: FAIL, `ImportError: TAIL_DX`.

- [ ] **Step 3: Implement**

In `calibrate/config.py::ExplicitConfig` add after `fit_v_floor`:

```python
    # continuation of a fitted slice beyond its quantile grid: "flat" (np.interp's end values),
    # "linear" (end slopes), "free" (the estimator evaluated TAIL_DX beyond each end)
    tail: str = "flat"
```

In `calibrate/explicit.py` add after the imports:

```python
TAIL_DX = 0.5   # log-spot distance of the tail anchor points beyond the grid ends
TAILS = ("flat", "linear", "free")


def extend_tail(grid, f_grid, tail, estimator=None, dx=TAIL_DX):
    """Append one anchor point beyond each end of `grid` so np.interp continues the slice by the
    chosen rule. "flat" returns the inputs unchanged (np.interp already holds the end values)."""
    if tail not in TAILS:
        raise ValueError(f"tail must be one of {TAILS}, got {tail!r}")
    if tail == "flat" or len(grid) < 2:
        return grid, f_grid
    lo, hi = grid[0] - dx, grid[-1] + dx
    if tail == "linear":
        f_lo = f_grid[0] + (f_grid[0] - f_grid[1]) / (grid[1] - grid[0]) * dx
        f_hi = f_grid[-1] + (f_grid[-1] - f_grid[-2]) / (grid[-1] - grid[-2]) * dx
    else:
        f_lo, f_hi = (float(x) for x in estimator.predict(np.array([lo, hi])))
    return (np.concatenate([[lo], grid, [hi]]),
            np.concatenate([[f_lo], f_grid, [f_hi]]))
```

In `calibrate_explicit`, directly after the `else:` branch that computes `f_grid` (the `fit_predict` call, before `f_grid = np.clip(f_grid, 1e-4, None)`), add inside that `else:` block:

```python
            grid, f_grid = extend_tail(grid, f_grid, cfg.tail, estimator)
```

so the clip, `sig`, `L_grid` and the stored `Slice` all use the extended grid.

- [ ] **Step 4: Run tests and goldens**

Run: `uv run pytest -q tests/calibrate && uv run pytest -q -m golden && git status --short tests/golden`
Expected: PASS, goldens untouched (default "flat" is a no-op).

- [ ] **Step 5: Commit**

```bash
git add src/neural_particle_method/calibrate/config.py src/neural_particle_method/calibrate/explicit.py tests/calibrate/test_tail.py
git commit -m "calibrate_explicit: tail rule for the fitted slice beyond its quantile grid"
```

---

### Task 4: Recipes module

**Files:**
- Create: `src/neural_particle_method/estimators/recipes.py`
- Create: `src/neural_particle_method/estimators/recipes/.gitkeep`
- Modify: `pyproject.toml` (package data: ensure `recipes/*.json` ships; check `[tool.hatch]`/`[tool.setuptools]` config and add an include if the build backend does not pick up non-Python files by default)
- Test: `tests/estimators/test_recipes.py`

**Interfaces:**
- Produces:
  - `RECIPE_TYPES: dict[str, type]` = `{"hidden": int, "depth": int, "lr": float, "batch_size": int, "first_steps": int, "later_steps": int, "weight_decay": float, "fit_subsample": int, "warm_start": bool, "mean_match": bool, "monotone": bool, "monotone_penalty": float, "tail": str, "hetero": bool}`.
  - `coerce_recipe(d: dict) -> dict` (casts strings from MLflow params; `"True"/"False"` → bool; fills `monotone_penalty=0.0` when absent).
  - `recipe_hash(recipe) -> str` (first 10 hex chars of sha1 of `json.dumps(recipe, sort_keys=True)`).
  - `regressor_from_recipe(recipe, seed=0, keep_slice_weights=False, monotone_sign=1.0) -> NNRegressor`.
  - `explicit_config_from_recipe(cfg: ExplicitConfig, recipe) -> ExplicitConfig` (replaces `first_steps`, `later_steps`, `tail`, and `fit_subsample=min(recipe["fit_subsample"], cfg.n_particles)`).
  - `RECIPE_DIR = Path(__file__).parent / "recipes"`; `load_recipe(name) -> dict`; `save_recipe(name, recipe) -> Path`; `recipe_exists(name) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/estimators/test_recipes.py
import json

import pytest

from neural_particle_method.calibrate.config import ExplicitConfig
from neural_particle_method.estimators import recipes as R

RECIPE = {"hidden": 32, "depth": 3, "lr": 1e-3, "batch_size": 0, "first_steps": 500,
          "later_steps": 100, "weight_decay": 1e-5, "fit_subsample": 250_000,
          "warm_start": False, "mean_match": True, "monotone": True, "monotone_penalty": 0.1,
          "tail": "linear", "hetero": True}


def test_coerce_from_mlflow_strings():
    raw = {k: str(v) for k, v in RECIPE.items()}
    assert R.coerce_recipe(raw) == RECIPE
    assert R.coerce_recipe({**raw, "monotone": "False"})["monotone_penalty"] == pytest.approx(0.1)
    without = {k: v for k, v in raw.items() if k != "monotone_penalty"}
    assert R.coerce_recipe(without)["monotone_penalty"] == 0.0


def test_hash_is_order_independent_and_short():
    h = R.recipe_hash(RECIPE)
    assert h == R.recipe_hash(dict(reversed(list(RECIPE.items())))) and len(h) == 10


def test_regressor_and_config_from_recipe():
    est = R.regressor_from_recipe(RECIPE, seed=3, monotone_sign=-1.0)
    assert (est.hidden, est.depth, est.lr, est.batch_size) == (32, 3, 1e-3, None)
    assert (est.first_steps, est.later_steps) == (500, 100)
    assert est.opt.param_groups[0]["weight_decay"] == pytest.approx(1e-5)
    assert est.warm_start is False and est.mean_match and est.hetero
    assert est.monotone_penalty == pytest.approx(0.1) and est.monotone_sign == -1.0
    off = R.regressor_from_recipe({**RECIPE, "monotone": False}, seed=3)
    assert off.monotone_penalty == 0.0
    cfg = R.explicit_config_from_recipe(ExplicitConfig(n_particles=100_000), RECIPE)
    assert (cfg.first_steps, cfg.later_steps, cfg.tail, cfg.fit_subsample) == (500, 100, "linear",
                                                                              100_000)


def test_save_load_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RECIPE_DIR", tmp_path)
    assert not R.recipe_exists("explicit_opt")
    p = R.save_recipe("explicit_opt", RECIPE)
    assert p == tmp_path / "explicit_opt.json" and json.loads(p.read_text()) == RECIPE
    assert R.recipe_exists("explicit_opt") and R.load_recipe("explicit_opt") == RECIPE
```

- [ ] **Step 2: Run them**

Run: `uv run pytest -q tests/estimators/test_recipes.py`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/neural_particle_method/estimators/recipes.py
"""A recipe is the dict of per-slice estimator knobs an Optuna trial samples. This module is the
only place a recipe becomes a regressor or an ExplicitConfig, so the search, the validation
bodies and the promoted algorithm cannot drift apart."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from .nn import NNRegressor

RECIPE_TYPES = {"hidden": int, "depth": int, "lr": float, "batch_size": int, "first_steps": int,
                "later_steps": int, "weight_decay": float, "fit_subsample": int,
                "warm_start": bool, "mean_match": bool, "monotone": bool,
                "monotone_penalty": float, "tail": str, "hetero": bool}
RECIPE_DIR = Path(__file__).parent / "recipes"


def _cast(typ, value):
    if typ is bool and isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return typ(value)


def coerce_recipe(d):
    """Cast a recipe read back from MLflow params (all strings) to its typed form."""
    out = {k: _cast(typ, d[k]) for k, typ in RECIPE_TYPES.items() if k in d}
    out.setdefault("monotone_penalty", 0.0)
    missing = set(RECIPE_TYPES) - set(out)
    if missing:
        raise KeyError(f"recipe is missing {sorted(missing)}")
    return out


def recipe_hash(recipe):
    return hashlib.sha1(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:10]


def regressor_from_recipe(recipe, seed=0, keep_slice_weights=False, monotone_sign=1.0):
    r = coerce_recipe(recipe)
    return NNRegressor(seed=seed, first_steps=r["first_steps"], later_steps=r["later_steps"],
                       hidden=r["hidden"], depth=r["depth"], lr=r["lr"],
                       batch_size=r["batch_size"] or None, weight_decay=r["weight_decay"],
                       warm_start=r["warm_start"], mean_match=r["mean_match"],
                       monotone_penalty=r["monotone_penalty"] if r["monotone"] else 0.0,
                       monotone_sign=monotone_sign, hetero=r["hetero"],
                       keep_slice_weights=keep_slice_weights)


def explicit_config_from_recipe(cfg, recipe):
    r = coerce_recipe(recipe)
    return replace(cfg, first_steps=r["first_steps"], later_steps=r["later_steps"],
                   tail=r["tail"], fit_subsample=min(r["fit_subsample"], cfg.n_particles))


def _path(name):
    return RECIPE_DIR / f"{name}.json"


def recipe_exists(name):
    return _path(name).exists()


def load_recipe(name):
    return coerce_recipe(json.loads(_path(name).read_text()))


def save_recipe(name, recipe):
    RECIPE_DIR.mkdir(parents=True, exist_ok=True)
    p = _path(name)
    p.write_text(json.dumps(coerce_recipe(recipe), indent=1, sort_keys=True) + "\n")
    return p
```

`recipe_exists`, `load_recipe` and `save_recipe` must read `RECIPE_DIR` at call time (module attribute lookup through `_path`), so the test's monkeypatch works. Create the empty `recipes/.gitkeep`. In `pyproject.toml`, if the build backend is hatchling, add `[tool.hatch.build.targets.wheel] packages = ["src/neural_particle_method"]` only if not already present (JSON files inside the package directory ship by default with hatchling and setuptools' `include_package_data`); verify with `uv build` is NOT required — a `uv run python -c "from neural_particle_method.estimators import recipes; print(recipes.RECIPE_DIR)"` printing the source path is enough for an editable install.

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q tests/estimators/test_recipes.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/neural_particle_method/estimators/recipes.py src/neural_particle_method/estimators/recipes/.gitkeep tests/estimators/test_recipes.py pyproject.toml
git commit -m "estimators.recipes: one place a recipe becomes a regressor and a config"
```

---

### Task 5: Tuning clouds in MLflow

**Files:**
- Create: `src/neural_particle_method/suite/optuna_clouds.py`
- Modify: `src/neural_particle_method/suite/config.py` (export `TUNING_SIDS`)
- Test: `tests/suite/test_optuna_clouds.py`

**Interfaces:**
- Consumes: `make_tuning_registry` (Task 1), `NadarayaWatson`, `nw_estimate`, `calibrate_explicit`, `Store`.
- Produces:
  - `SLICE_TIMES = (0.10, 0.50, 1.00, 1.75)`, `EXPERIMENT = "optuna_clouds"`, `HELD_FRAC = 0.2`.
  - `quantile_grid(lnx) -> np.ndarray` (`np.unique(np.quantile(lnx, np.linspace(0.001, 0.999, 101)))`).
  - `heldout_loss(grid, f_grid, lnx_held, v_held) -> float` (mean squared error of `np.interp(lnx_held, grid, f_grid)` against `v_held`).
  - `class RecordingNW(NadarayaWatson)`: `__init__(self, times, dt)`, records `(t, lnx.copy(), v.copy())` in `self.records` when `round(t / dt)` equals one of the target steps.
  - `@dataclass Cloud`: `sid`, `rho`, `times: list[float]`, `fit: list[tuple[lnx, v]]`, `held: list[tuple[lnx, v]]`, `nw_loss: list[float]`.
  - `cloud_key(sid, settings) -> dict` = `{"sid", "n_particles": settings.offline_sizes[-1], "n_steps": settings.explicit.n_steps, "seed": 0}`.
  - `ensure_cloud(store, sid, settings=FULL, registry=None, times=SLICE_TIMES) -> run_id` (idempotent; artifact `cloud.npz`, metrics `nw_loss/t<t:g>`, params key + scenario params + `times`).
  - `load_cloud(store, run_id, cache_dir=None) -> Cloud` (downloads `cloud.npz` into `cache_dir or results/optuna/clouds/<run_id>` once).

- [ ] **Step 1: Write the failing tests**

```python
# tests/suite/test_optuna_clouds.py
import numpy as np
import pytest

from neural_particle_method.bench.scenarios import make_tuning_registry
from neural_particle_method.suite.config import TUNING_SIDS, SuiteSettings
from neural_particle_method.suite.optuna_clouds import (
    EXPERIMENT, Cloud, ensure_cloud, heldout_loss, load_cloud, quantile_grid)
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()
TIMES = (0.25, 0.5)            # inside the tiny horizon, on its 4-step grid


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def test_tuning_sids_are_the_registry():
    assert TUNING_SIDS == tuple(make_tuning_registry())


def test_heldout_loss_is_interpolated_mse():
    grid, f = np.array([0.0, 1.0]), np.array([0.0, 2.0])
    assert heldout_loss(grid, f, np.array([0.5, 2.0]), np.array([1.0, 3.0])) == pytest.approx(0.5)


def test_quantile_grid_is_sorted_unique_101_or_fewer():
    g = quantile_grid(np.random.default_rng(0).normal(size=5000))
    assert len(g) <= 101 and np.all(np.diff(g) > 0)


def test_ensure_cloud_is_idempotent_and_records_the_slices(store, tmp_path):
    reg = make_tuning_registry()
    rid = ensure_cloud(store, "t01", TINY, reg, times=TIMES)
    assert ensure_cloud(store, "t01", TINY, reg, times=TIMES) == rid
    assert len(store.search(EXPERIMENT)) == 1
    m = store.get_metrics(rid)
    assert set(k for k in m if k.startswith("nw_loss/")) == {"nw_loss/t0.25", "nw_loss/t0.5"}
    c = load_cloud(store, rid, cache_dir=tmp_path / "cache")
    assert isinstance(c, Cloud) and c.sid == "t01" and c.times == [0.25, 0.5]
    n = TINY.offline_sizes[-1]
    for (lf, vf), (lh, vh) in zip(c.fit, c.held):
        assert len(lf) == len(vf) == n - int(0.2 * n) and len(lh) == len(vh) == int(0.2 * n)
        assert (vf >= 0).all() and (vh >= 0).all()
    assert c.rho == reg["t01"].dynamics.rho
    assert all(loss > 0 for loss in c.nw_loss)
```

- [ ] **Step 2: Run them**

Run: `uv run pytest -q tests/suite/test_optuna_clouds.py`
Expected: FAIL (`ImportError: TUNING_SIDS`).

- [ ] **Step 3: Implement**

In `suite/config.py` add `from ..bench.scenarios import heston_registry, make_registry, make_tuning_registry` and `TUNING_SIDS = tuple(make_tuning_registry())` next to `SSVI_SIDS`.

```python
# src/neural_particle_method/suite/optuna_clouds.py
"""Tuning clouds for the estimator search: one MLflow run per held-out scenario holding the
particle slices an offline pass sees, split into a fit pool and a held-out set, with NW's
held-out loss per slice as the yardstick."""
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..bench.scenarios import make_tuning_registry
from ..calibrate.explicit import calibrate_explicit
from ..estimators.nadaraya_watson import NadarayaWatson, nw_estimate
from ..tracking.store import git_hash
from .config import FULL

EXPERIMENT = "optuna_clouds"
SLICE_TIMES = (0.10, 0.50, 1.00, 1.75)
HELD_FRAC = 0.2


def quantile_grid(lnx):
    return np.unique(np.quantile(lnx, np.linspace(0.001, 0.999, 101)))


def heldout_loss(grid, f_grid, lnx_held, v_held):
    return float(np.mean((np.interp(lnx_held, grid, f_grid) - v_held) ** 2))


class RecordingNW(NadarayaWatson):
    """NW estimator that also keeps the (t, lnx, v) it was handed at the requested slice times."""

    def __init__(self, times, dt):
        super().__init__()
        self.steps = {int(round(t / dt)): float(t) for t in times}
        self.records = []

    def fit_predict(self, t, lnx, v, grid, weights=None):
        k = int(round(t / (t / max(round(t / self._dt), 1)))) if False else None  # noqa: F841
        return super().fit_predict(t, lnx, v, grid, weights=weights)
```

Replace the last method with this (the line above is a placeholder to delete; the real method is):

```python
    def fit_predict(self, t, lnx, v, grid, weights=None):
        k = int(round(t / self.dt))
        if k in self.steps:
            self.records.append((self.steps[k], lnx.copy(), v.copy()))
        return super().fit_predict(t, lnx, v, grid, weights=weights)
```

and set `self.dt = dt` in `__init__`. Then:

```python
@dataclass
class Cloud:
    sid: str
    rho: float
    times: list
    fit: list       # [(lnx, v_plus)] per slice, the fit pool
    held: list      # [(lnx, v_plus)] per slice, held out
    nw_loss: list   # NW held-out MSE per slice, fitted on the whole fit pool


def cloud_key(sid, settings=FULL):
    return {"sid": sid, "n_particles": int(settings.offline_sizes[-1]),
            "n_steps": int(settings.explicit.n_steps), "seed": 0}


def _split(lnx, v):
    n_held = int(HELD_FRAC * len(lnx))
    return (lnx[:-n_held], v[:-n_held]), (lnx[-n_held:], v[-n_held:])


def ensure_cloud(store, sid, settings=FULL, registry=None, times=SLICE_TIMES):
    key = cloud_key(sid, settings)
    existing = store.find_finished(EXPERIMENT, key)
    if existing is not None:
        return existing
    sc = (registry or make_tuning_registry())[sid]
    n = key["n_particles"]
    # every fit on the whole cloud; rng.choice without replacement hands the estimator a random
    # permutation, which is the split's randomness
    ecfg = replace(settings.explicit, n_particles=n, fit_subsample=n, fit_v_floor=True)
    est = RecordingNW(times, sc.T / ecfg.n_steps)
    params = {**key, "git_hash": git_hash(), "times": ",".join(f"{t:g}" for t in times),
              **sc.as_params()}
    with store.run(EXPERIMENT, params) as h:
        calibrate_explicit(sc.local_vol(), sc.dynamics, est, ecfg, s0=sc.s0, T=sc.T, seed=0)
        arrays, metrics = {"times": np.array([r[0] for r in est.records]), "rho": sc.dynamics.rho}, {}
        for i, (t, lnx, v) in enumerate(est.records):
            (lf, vf), (lh, vh) = _split(lnx, v)
            grid = quantile_grid(lf)
            loss = heldout_loss(grid, nw_estimate(lf, vf, grid), lh, vh)
            arrays.update({f"lnx_fit_{i}": lf, f"v_fit_{i}": vf, f"lnx_held_{i}": lh,
                           f"v_held_{i}": vh})
            metrics[f"nw_loss/t{t:g}"] = loss
        h.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cloud.npz"
            np.savez_compressed(p, **arrays)
            h.log_file(p)
        return h.run_id


def load_cloud(store, run_id, cache_dir=None):
    cache = Path(cache_dir or f"results/optuna/clouds/{run_id}")
    p = cache / "cloud.npz"
    if not p.exists():
        store.download(run_id, "cloud.npz", cache)
    z = np.load(p)
    times = [float(t) for t in z["times"]]
    m = store.get_metrics(run_id)
    return Cloud(sid=store.get_params(run_id)["sid"], rho=float(z["rho"]), times=times,
                 fit=[(z[f"lnx_fit_{i}"], z[f"v_fit_{i}"]) for i in range(len(times))],
                 held=[(z[f"lnx_held_{i}"], z[f"v_held_{i}"]) for i in range(len(times))],
                 nw_loss=[m[f"nw_loss/t{t:g}"] for t in times])
```

Delete the placeholder `fit_predict` and keep only the real one. `calibrate_explicit` already passes `np.maximum(v, 0)` when `fit_v_floor` is True (line 84 of explicit.py), so the recorded `v` is `v_plus`. Check `ScenarioSpec` has `T`, `s0`, `local_vol()`, `dynamics`, `as_params()` (it does; `run_offline` uses them).

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q tests/suite/test_optuna_clouds.py tests/suite/test_config.py 2>/dev/null || uv run pytest -q tests/suite/test_optuna_clouds.py`
Expected: PASS. If the tiny pass never reaches `t=0.5` because `T` of `t01` is 2 and the tiny grid has 4 steps (`dt=0.5`), `round(0.25/0.5)=0` collides with `k=0` (no fit at step 0): change `TIMES` in the test to `(0.5, 1.0)`.

- [ ] **Step 5: Commit**

```bash
git add src/neural_particle_method/suite/optuna_clouds.py src/neural_particle_method/suite/config.py tests/suite/test_optuna_clouds.py
git commit -m "suite.optuna_clouds: held-out tuning clouds with NW's held-out loss, in MLflow"
```

---

### Task 6: Objective and study runner

**Files:**
- Create: `src/neural_particle_method/suite/optuna_search.py`
- Modify: `pyproject.toml` (`uv add optuna`)
- Test: `tests/suite/test_optuna_search.py`

**Interfaces:**
- Consumes: `regressor_from_recipe`, `explicit_config_from_recipe` (Task 4); `extend_tail` (Task 3); `Cloud`, `load_cloud`, `ensure_cloud`, `quantile_grid`, `heldout_loss` (Task 5).
- Produces:
  - `EXPERIMENT = "optuna_offline"`, `STORAGE_DIR = Path("results/optuna")`.
  - `suggest_recipe(trial) -> dict` (spec §2 distributions).
  - `fit_chain(recipe, cloud, seed) -> list[float]` (log loss ratios per slice).
  - `score_recipe(recipe, clouds, seed=0, report=None) -> (score: float, detail: dict, fit_s: float)` where `report(i, running_mean)` may raise `optuna.TrialPruned`; `detail = {sid: {t: logratio}}`.
  - `study_parent(store, study) -> run_id` (find-or-create the parent run; params `{study, tuning_seed: 5000, git_hash}`, tag `optuna.kind=study`; sets status RUNNING).
  - `run_study(store, study, n_trials, n_jobs=1, settings=FULL, sids=TUNING_SIDS, per_trial=8, times=SLICE_TIMES, registry=None, storage_dir=STORAGE_DIR, cache_dir=None) -> run_id` (ensures clouds, runs trials in `n_jobs` processes, logs `optuna.db`, `best.json`, metrics `best_score`, `n_trials` on the parent).
  - `top_recipes(store, study, k) -> list[dict]` with keys `trial_number`, `score`, `recipe`, `run_id` (COMPLETE trials sorted by score ascending).

- [ ] **Step 1: Add the dependency**

Run: `uv add optuna` and check `pyproject.toml` lists `optuna>=4` under `dependencies` and `uv.lock` changed.

- [ ] **Step 2: Write the failing tests**

```python
# tests/suite/test_optuna_search.py
import numpy as np
import optuna
import pytest

from neural_particle_method.bench.scenarios import make_tuning_registry
from neural_particle_method.estimators.nadaraya_watson import nw_estimate
from neural_particle_method.suite.config import SuiteSettings
from neural_particle_method.suite.optuna_clouds import (
    Cloud, ensure_cloud, heldout_loss, load_cloud, quantile_grid)
from neural_particle_method.suite.optuna_search import (
    EXPERIMENT, run_study, score_recipe, suggest_recipe, top_recipes)
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()
TIMES = (0.5, 1.0)
FAST = {"hidden": 16, "depth": 2, "lr": 1e-2, "batch_size": 0, "first_steps": 5,
        "later_steps": 2, "weight_decay": 0.0, "fit_subsample": 400, "warm_start": True,
        "mean_match": False, "monotone": False, "monotone_penalty": 0.0, "tail": "flat",
        "hetero": False}


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def _synthetic_cloud():
    rng = np.random.default_rng(1)
    fit, held, nw_loss = [], [], []
    for _ in range(2):
        lnx = rng.normal(0, 0.3, 2000)
        v = np.maximum(0.04 * np.exp(-lnx) + 0.01 * rng.normal(size=2000), 0)
        (lf, vf), (lh, vh) = (lnx[:1600], v[:1600]), (lnx[1600:], v[1600:])
        g = quantile_grid(lf)
        fit.append((lf, vf)); held.append((lh, vh))
        nw_loss.append(heldout_loss(g, nw_estimate(lf, vf, g), lh, vh))
    return Cloud("x", -0.5, [0.5, 1.0], fit, held, nw_loss)


def test_nw_itself_scores_zero(monkeypatch):
    import neural_particle_method.suite.optuna_search as S
    cloud = _synthetic_cloud()

    class NWStub:
        def __init__(self):
            self.i = 0
        def fit_predict(self, t, lnx, v, grid, weights=None):
            return nw_estimate(lnx, v, grid)
        def predict(self, x):
            return np.zeros_like(x)

    monkeypatch.setattr(S, "regressor_from_recipe", lambda *a, **k: NWStub())
    score, detail, fit_s = score_recipe({**FAST, "fit_subsample": 1600}, [cloud], seed=0)
    assert score == pytest.approx(0.0, abs=1e-12) and detail == {"x": {0.5: 0.0, 1.0: 0.0}}
    assert fit_s >= 0


def test_score_is_finite_for_a_real_recipe_and_reports_for_pruning():
    cloud = _synthetic_cloud()
    seen = []
    score, detail, _ = score_recipe(FAST, [cloud, cloud], seed=0, report=lambda i, m: seen.append(i))
    assert np.isfinite(score) and seen == [0, 1] and set(detail) == {"x"}


def test_suggest_recipe_covers_the_search_space():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    keys = set()
    for _ in range(20):
        t = study.ask()
        r = suggest_recipe(t)
        study.tell(t, 0.0)
        keys |= set(r)
        assert r["hidden"] in (32, 64, 128) and r["depth"] in (2, 3, 4)
        assert 1e-4 <= r["lr"] <= 1e-2 and r["batch_size"] in (2048, 8192, 0)
        assert 500 <= r["first_steps"] <= 4000 and r["first_steps"] % 250 == 0
        assert 100 <= r["later_steps"] <= 1500 and r["later_steps"] % 50 == 0
        assert r["fit_subsample"] in (100_000, 250_000, 400_000) and r["tail"] in ("free", "flat",
                                                                                 "linear")
        assert (r["monotone_penalty"] == 0.0) == (not r["monotone"])
    assert keys == set(FAST)


def test_run_study_logs_parent_and_children_and_resumes(store, tmp_path, monkeypatch):
    import neural_particle_method.suite.optuna_search as S
    monkeypatch.setattr(S, "suggest_recipe", lambda trial: {**FAST, "lr": trial.suggest_float(
        "lr", 1e-3, 1e-2, log=True)})
    reg = make_tuning_registry()
    sids = ("t01", "t02")
    kw = dict(settings=TINY, sids=sids, per_trial=2, times=TIMES, registry=reg,
              storage_dir=tmp_path / "optuna", cache_dir=tmp_path / "cache")
    parent = run_study(store, "unit", n_trials=2, n_jobs=1, **kw)
    runs = store.search(EXPERIMENT)
    kids = runs[runs["tags.mlflow.parentRunId"] == parent]
    assert len(kids) == 2 and set(kids["tags.optuna.state"]) <= {"COMPLETE", "PRUNED"}
    assert {"metrics.score", "params.lr", "params.trial_number"} <= set(kids.columns)
    assert run_study(store, "unit", n_trials=1, n_jobs=1, **kw) == parent
    assert len(optuna.load_study(study_name="unit",
                                 storage=f"sqlite:///{tmp_path / 'optuna' / 'unit.db'}").trials) == 3
    top = top_recipes(store, "unit", 2)
    assert len(top) >= 1 and top[0]["score"] <= top[-1]["score"]
    assert set(top[0]) == {"trial_number", "score", "recipe", "run_id"}
    assert top[0]["recipe"]["hidden"] == 16
```

- [ ] **Step 3: Run them**

Run: `uv run pytest -q tests/suite/test_optuna_search.py`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 4: Implement**

```python
# src/neural_particle_method/suite/optuna_search.py
"""Optuna search over the per-slice estimator recipe on the tuning clouds, tracked in MLflow: one
parent run per study, one nested run per trial, the Optuna SQLite storage as an artifact."""
import json
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import optuna

from ..calibrate.explicit import extend_tail
from ..estimators.recipes import coerce_recipe, regressor_from_recipe
from ..tracking.store import Store, git_hash
from .config import FULL, TUNING_SIDS
from .optuna_clouds import SLICE_TIMES, ensure_cloud, heldout_loss, load_cloud, quantile_grid

EXPERIMENT = "optuna_offline"
STORAGE_DIR = Path("results/optuna")
TUNING_SEED = 5000


def suggest_recipe(trial):
    monotone = trial.suggest_categorical("monotone", [False, True])
    return {
        "hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
        "depth": trial.suggest_categorical("depth", [2, 3, 4]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [2048, 8192, 0]),
        "first_steps": trial.suggest_int("first_steps", 500, 4000, step=250),
        "later_steps": trial.suggest_int("later_steps", 100, 1500, step=50),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "fit_subsample": trial.suggest_categorical("fit_subsample", [100_000, 250_000, 400_000]),
        "warm_start": trial.suggest_categorical("warm_start", [True, False]),
        "mean_match": trial.suggest_categorical("mean_match", [False, True]),
        "monotone": monotone,
        "monotone_penalty": (trial.suggest_float("monotone_penalty", 1e-4, 1.0, log=True)
                             if monotone else 0.0),
        "tail": trial.suggest_categorical("tail", ["free", "flat", "linear"]),
        "hetero": trial.suggest_categorical("hetero", [False, True]),
    }


def fit_chain(recipe, cloud, seed):
    """Fit the cloud's slices in order with one regressor; log held-out loss ratios to NW."""
    r = coerce_recipe(recipe)
    est = regressor_from_recipe(r, seed=seed, monotone_sign=-float(np.sign(cloud.rho) or 1.0))
    rng = np.random.default_rng(seed)
    out = []
    for i, ((lf, vf), (lh, vh)) in enumerate(zip(cloud.fit, cloud.held)):
        idx = rng.choice(len(lf), size=min(r["fit_subsample"], len(lf)), replace=False)
        grid = quantile_grid(lf[idx])
        f = est.fit_predict(cloud.times[i], lf[idx], vf[idx], grid)
        grid, f = extend_tail(grid, f, r["tail"], est)
        f = np.clip(f, 1e-4, None)
        out.append(float(np.log(heldout_loss(grid, f, lh, vh) / cloud.nw_loss[i])))
    return out


def score_recipe(recipe, clouds, seed=0, report=None):
    detail, logs, t0 = {}, [], time.perf_counter()
    for i, cloud in enumerate(clouds):
        ratios = fit_chain(recipe, cloud, seed + i)
        detail[cloud.sid] = dict(zip(cloud.times, ratios))
        logs += ratios
        if report is not None:
            report(i, float(np.mean(logs)))
    return float(np.mean(logs)), detail, time.perf_counter() - t0


def study_parent(store, study):
    exp = store.experiment_id(EXPERIMENT)
    runs = store.client.search_runs([exp], f"params.study = '{study}' and tags.`optuna.kind` = 'study'",
                                    max_results=1)
    if runs:
        rid = runs[0].info.run_id
        store.client.update_run(rid, status="RUNNING")
        return rid
    r = store.client.create_run(exp, tags={"optuna.kind": "study"})
    store.client.log_batch(r.info.run_id, params=[])
    for k, v in {"study": study, "tuning_seed": TUNING_SEED, "git_hash": git_hash()}.items():
        store.client.log_param(r.info.run_id, k, v)
    return r.info.run_id


def _storage(storage_dir, study):
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{Path(storage_dir) / f'{study}.db'}"


def _trial_worker(args):
    """One process: runs `n` trials of the shared study, logging each as a nested MLflow run."""
    uri, root, study, storage, parent, cloud_runs, per_trial, n, cache_dir, seed = args
    store = Store(uri, root)
    clouds = {rid: load_cloud(store, rid, cache_dir=None if cache_dir is None
                              else Path(cache_dir) / rid) for rid in cloud_runs}
    st = optuna.load_study(study_name=study, storage=storage)
    exp = store.experiment_id(EXPERIMENT)

    def objective(trial):
        rng = np.random.default_rng(trial.number)
        picked = [cloud_runs[j] for j in rng.choice(len(cloud_runs), size=min(per_trial,
                                                                              len(cloud_runs)),
                                                    replace=False)]
        recipe = suggest_recipe(trial)
        run = store.client.create_run(exp, tags={"mlflow.parentRunId": parent,
                                                 "optuna.kind": "trial"})
        rid = run.info.run_id
        for k, v in {**recipe, "trial_number": trial.number}.items():
            store.client.log_param(rid, k, v)

        def report(i, running):
            trial.report(running, step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            score, detail, fit_s = score_recipe(recipe, [clouds[r] for r in picked], seed=seed,
                                                report=report)
        except optuna.TrialPruned:
            store.client.set_tag(rid, "optuna.state", "PRUNED")
            store.client.set_terminated(rid, status="FINISHED")
            raise
        except Exception:
            store.client.set_tag(rid, "optuna.state", "FAIL")
            store.client.set_terminated(rid, status="FAILED")
            raise
        store.client.log_metric(rid, "score", score)
        store.client.log_metric(rid, "fit_s", fit_s)
        for sid, per in detail.items():
            store.client.log_metric(rid, f"score/{sid}", float(np.mean(list(per.values()))))
            for t, lr_ in per.items():
                store.client.log_metric(rid, f"loss_ratio/{sid}/t{t:g}", lr_)
        store.client.set_tag(rid, "optuna.state", "COMPLETE")
        store.client.set_terminated(rid, status="FINISHED")
        return score

    st.optimize(objective, n_trials=n, catch=(Exception,))
    return n


def run_study(store, study, n_trials, n_jobs=1, settings=FULL, sids=TUNING_SIDS, per_trial=8,
              times=SLICE_TIMES, registry=None, storage_dir=STORAGE_DIR, cache_dir=None, seed=0):
    cloud_runs = [ensure_cloud(store, sid, settings, registry, times) for sid in sids]
    storage = _storage(storage_dir, study)
    optuna.create_study(study_name=study, storage=storage, load_if_exists=True,
                        direction="minimize", sampler=optuna.samplers.TPESampler(seed=0),
                        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
    parent = study_parent(store, study)
    per_proc = [n_trials // n_jobs + (1 if i < n_trials % n_jobs else 0) for i in range(n_jobs)]
    args = [(store.tracking_uri, store.artifact_root, study, storage, parent, cloud_runs,
             per_trial, n, cache_dir, seed) for n in per_proc if n > 0]
    if n_jobs == 1:
        _trial_worker(args[0])
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            list(ex.map(_trial_worker, args))
    st = optuna.load_study(study_name=study, storage=storage)
    done = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
    store.client.log_metric(parent, "n_trials", len(st.trials))
    if done:
        best = min(done, key=lambda t: t.value)
        store.client.log_metric(parent, "best_score", best.value)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "best.json"
            p.write_text(json.dumps({"trial_number": best.number, "score": best.value,
                                     "params": best.params}, indent=1))
            store.client.log_artifact(parent, str(p))
    store.client.log_artifact(parent, str(Path(storage_dir) / f"{study}.db"))
    store.client.set_terminated(parent, status="FINISHED")
    return parent


def top_recipes(store, study, k):
    parent = study_parent(store, study)
    df = store.search(EXPERIMENT)
    kids = df[(df["tags.mlflow.parentRunId"] == parent) & (df["tags.optuna.state"] == "COMPLETE")]
    kids = kids.sort_values("metrics.score").head(k)
    out = []
    for _, r in kids.iterrows():
        recipe = coerce_recipe({key: r[f"params.{key}"] for key in coerce_recipe.__globals__[
            "RECIPE_TYPES"] if f"params.{key}" in r and r[f"params.{key}"] is not None})
        out.append({"trial_number": int(r["params.trial_number"]), "score": float(r["metrics.score"]),
                    "recipe": recipe, "run_id": r["run_id"]})
    store.client.set_terminated(parent, status="FINISHED")
    return out
```

Replace the `coerce_recipe.__globals__[...]` hack with a plain import: `from ..estimators.recipes import RECIPE_TYPES, coerce_recipe, regressor_from_recipe` and `{key: r[f"params.{key}"] for key in RECIPE_TYPES if ...}`. Check `Store` exposes `tracking_uri` and `artifact_root` (the budget script uses them). `study_parent` must not leave a `params=[]` no-op call; delete that `log_batch` line. Keep every line ≤ 100 characters (wrap the long `search_runs` filter string).

- [ ] **Step 5: Run tests**

Run: `uv run pytest -q tests/suite/test_optuna_search.py -x`
Expected: PASS. The resume test's third trial: `run_study` with `n_trials=1` adds one trial to the storage (3 total) and one child run (the assertion on `len(kids) == 2` is taken before the resume call, so it holds).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/neural_particle_method/suite/optuna_search.py tests/suite/test_optuna_search.py
git commit -m "suite.optuna_search: recipe objective on tuning clouds; study as a parent MLflow run with nested trials"
```

---

### Task 7: Budget cell into the package; `explicit_opt` body

**Files:**
- Create: `src/neural_particle_method/suite/budget.py`
- Modify: `scripts/budget_sweep.py` (thin wrapper)
- Modify: `src/neural_particle_method/suite/offline.py`
- Modify: `tests/scripts/test_budget_sweep.py`, `tests/suite/test_offline.py`
- Test: `tests/suite/test_budget.py`

**Interfaces:**
- Produces:
  - `suite/budget.py`: `EXPERIMENTS = {"explicit_tuned": "suite_budget_tuned", "explicit": "suite_budget", "explicit_opt": "suite_budget_tuned"}`, `BUDGETS`, `SEEDS`, `budget_methods(body) -> tuple[str, ...]` (as the script's `methods`), `head_for(method)` (as the script's `_head`), `body_run(store, sid, body, settings=FULL, recipe=None) -> run_id`, `run_budget_cell(store, sid, method, budget, lag, seed, body="explicit_tuned", settings=FULL, recipe=None) -> run_id` (the script's `run_cell` with `settings` instead of the `FULL` global, the key gaining `"recipe_hash": recipe_hash(recipe)` when `body == "explicit_opt"`, and the params gaining `recipe.<key>`), `budget_jobs(sids, body, budgets=BUDGETS, seeds=SEEDS) -> list`.
  - `run_offline(store, sid, body, n_particles, settings=FULL, seed=0, recipe=None)`: `body == "explicit_opt"` uses `recipe` (or `load_recipe("explicit_opt")` when None), key gains `recipe_hash`, estimator `regressor_from_recipe(recipe, seed, keep_slice_weights=True, monotone_sign=-sign(rho))`, config `explicit_config_from_recipe(ecfg, recipe)`, params gain `recipe.<key>`. `BODIES = ("explicit", "explicit_tuned", "implicit", "explicit_opt")`, `STAGE_BODIES = ("explicit", "explicit_tuned", "implicit")` (used by `grid.offline_jobs`), `EXPLICIT_BODIES` gains `"explicit_opt"`.
  - `scripts/budget_sweep.py` keeps its CLI, adds `--body explicit_opt` and `--recipe <name>` (default `explicit_opt`), and calls the package.

- [ ] **Step 1: Write the failing tests**

```python
# tests/suite/test_budget.py
import pytest

from neural_particle_method.suite.budget import (
    EXPERIMENTS, budget_jobs, budget_methods, head_for, run_budget_cell)
from neural_particle_method.suite.config import SuiteSettings
from neural_particle_method.suite.heads import SplineHead
from neural_particle_method.suite.lag import LAGS
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()
FAST = {"hidden": 16, "depth": 2, "lr": 1e-2, "batch_size": 0, "first_steps": 5,
        "later_steps": 2, "weight_decay": 0.0, "fit_subsample": 400, "warm_start": True,
        "mean_match": True, "monotone": False, "monotone_penalty": 0.0, "tail": "linear",
        "hetero": False}


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def test_methods_and_heads():
    assert budget_methods("explicit_opt") == ("nw_resolve", "explicit_opt_stale",
                                              "explicit_opt_rkhs", "explicit_opt_spline",
                                              "explicit_opt_ridge")
    assert isinstance(head_for("explicit_opt_spline"), SplineHead)
    assert head_for("nw_resolve") is None and head_for("explicit_opt_stale") is None
    assert len(budget_jobs(("s01",), "explicit_tuned")) == 60
    assert EXPERIMENTS["explicit_opt"] == "suite_budget_tuned"


def test_opt_cell_runs_on_tiny_and_is_keyed_by_recipe(store):
    rid = run_budget_cell(store, "s01", "explicit_opt_spline", TINY.n_online, LAGS[0], 0,
                          body="explicit_opt", settings=TINY, recipe=FAST)
    p = store.get_params(rid)
    assert p["body"] == "explicit_opt" and len(p["recipe_hash"]) == 10
    assert p["recipe.tail"] == "linear"
    assert run_budget_cell(store, "s01", "explicit_opt_spline", TINY.n_online, LAGS[0], 0,
                           body="explicit_opt", settings=TINY, recipe=FAST) == rid
    other = run_budget_cell(store, "s01", "explicit_opt_spline", TINY.n_online, LAGS[0], 0,
                            body="explicit_opt", settings=TINY, recipe={**FAST, "tail": "flat"})
    assert other != rid
    assert store.get_metrics(rid)["liquid_mae_bp"] >= 0
```

Add to `tests/suite/test_offline.py`:

```python
def test_opt_body_uses_the_recipe_and_is_keyed_by_its_hash(store):
    from neural_particle_method.estimators.recipes import recipe_hash
    from neural_particle_method.suite.offline import STAGE_BODIES
    recipe = {"hidden": 16, "depth": 2, "lr": 1e-2, "batch_size": 0, "first_steps": 5,
              "later_steps": 2, "weight_decay": 0.0, "fit_subsample": 400, "warm_start": True,
              "mean_match": False, "monotone": True, "monotone_penalty": 0.01, "tail": "free",
              "hetero": False}
    n = TINY.offline_sizes[0]
    rid = run_offline(store, "s01", "explicit_opt", n, TINY, recipe=recipe)
    p = store.get_params(rid)
    assert p["recipe_hash"] == recipe_hash(recipe) and p["explicit.tail"] == "free"
    assert p["explicit.first_steps"] == "5" and p["recipe.monotone"] == "True"
    lr = load_run(store, rid)
    assert isinstance(lr.model, SliceBank) and len(lr.field) == TINY.explicit.n_steps
    assert run_offline(store, "s01", "explicit_opt", n, TINY, recipe=recipe) == rid
    assert BODIES == ("explicit", "explicit_tuned", "implicit", "explicit_opt")
    assert STAGE_BODIES == ("explicit", "explicit_tuned", "implicit")
```

and update `test_bodies_are_the_three_agreed_ones` to assert `STAGE_BODIES == ("explicit", "explicit_tuned", "implicit")` (rename it `test_stage_bodies_are_the_three_agreed_ones`). In `tests/scripts/test_budget_sweep.py`, replace imports of `jobs`, `_head` from the script by `budget_jobs`, `head_for` from `neural_particle_method.suite.budget` and keep the assertions (`len(budget_jobs(("s01",), "explicit_tuned")) == 60`, etc.).

- [ ] **Step 2: Run them**

Run: `uv run pytest -q tests/suite/test_budget.py tests/suite/test_offline.py tests/scripts/test_budget_sweep.py`
Expected: FAIL (`ModuleNotFoundError: suite.budget`, unknown body).

- [ ] **Step 3: Implement `suite/offline.py`**

```python
BODIES = ("explicit", "explicit_tuned", "implicit", "explicit_opt")
STAGE_BODIES = ("explicit", "explicit_tuned", "implicit")   # what `suite run --stage offline` trains
EXPLICIT_BODIES = ("explicit", "explicit_tuned", "explicit_opt")


def run_offline(store, sid, body, n_particles, settings=FULL, seed=0, recipe=None):
    if body not in BODIES:
        raise ValueError(f"body must be one of {BODIES}, got {body!r}")
    n_steps = settings.explicit.n_steps
    key = {"sid": sid, "body": body, "n_particles": int(n_particles), "seed": int(seed),
           "n_steps": int(n_steps)}
    if body == "explicit_opt":
        recipe = coerce_recipe(recipe if recipe is not None else load_recipe("explicit_opt"))
        key["recipe_hash"] = recipe_hash(recipe)
    exp = settings.experiment("suite_offline")
    existing = store.find_finished(exp, key)
    if existing is not None:
        return existing
    sc = full_registry()[sid]
    lv = sc.local_vol()
    tuned = body == "explicit_tuned"
    ecfg = replace(settings.explicit, n_particles=int(n_particles),
                   **(TUNED_STEPS if tuned else {}))
    if body == "explicit_opt":
        ecfg = explicit_config_from_recipe(ecfg, recipe)
    icfg = replace(settings.implicit, n_particles=int(n_particles))
    params = {**key, "git_hash": git_hash(), **sc.as_params(),
              **{f"explicit.{k}": v for k, v in ecfg.as_params().items()},
              **{f"implicit.{k}": v for k, v in icfg.as_params().items()},
              **{f"reprice.{k}": v for k, v in settings.reprice.as_params().items()}}
    if body == "explicit_opt":
        params.update({f"recipe.{k}": v for k, v in recipe.items()})
    with store.run(exp, params) as h:
        t0 = time.perf_counter()
        if body == "explicit_opt":
            est = regressor_from_recipe(recipe, seed=seed, keep_slice_weights=True,
                                        monotone_sign=-float(np.sign(sc.dynamics.rho) or 1.0))
        elif tuned:
            est = NNRegressor(seed=seed, keep_slice_weights=True, **TUNED_KNOBS, **TUNED_STEPS)
        else:
            est = NNRegressor(seed=seed, first_steps=ecfg.first_steps,
                              later_steps=ecfg.later_steps,
                              keep_slice_weights=(body == "explicit"))
        ... (rest unchanged)
```

Imports: `import numpy as np`, `from ..estimators.recipes import (coerce_recipe, explicit_config_from_recipe, load_recipe, recipe_hash, regressor_from_recipe)`. In `suite/grid.py::offline_jobs` and the online-stage precheck in `run_stage`, iterate `STAGE_BODIES` instead of `BODIES` (import it). `save_model` for a `SliceBank` must store `hidden`/`depth` from the regressor (F13 does) — nothing else needed.

- [ ] **Step 4: Implement `suite/budget.py`**

Move the body of `scripts/budget_sweep.py` (everything except argparse and `__main__`) into `src/neural_particle_method/suite/budget.py`, with these changes: `EXPERIMENTS` gains `"explicit_opt": "suite_budget_tuned"`; `methods` → `budget_methods`; `_head` → `head_for`; `_body_run` → `body_run(store, sid, body, settings=FULL, recipe=None)` which for `explicit_opt` calls `run_offline(store, sid, "explicit_opt", settings.offline_sizes[-1], settings, recipe=recipe)` and for `explicit_tuned` uses `settings.offline_sizes[-1]` and `settings`; `run_cell` → `run_budget_cell(store, sid, method, budget, lag, seed, body="explicit_tuned", settings=FULL, recipe=None)` using `settings.explicit`, `settings.reprice`, `settings.explicit.n_steps` in the key, and:

```python
    if body == "explicit_opt":
        recipe = coerce_recipe(recipe if recipe is not None else load_recipe("explicit_opt"))
        key["recipe_hash"] = recipe_hash(recipe)
    ...
    params = {**key, "git_hash": git_hash(), "body_run": body_rid, "body": body, ...}
    if body == "explicit_opt":
        params.update({f"recipe.{k}": v for k, v in recipe.items()})
```

`jobs` → `budget_jobs(sids, body, budgets=BUDGETS, seeds=SEEDS)` returning `(sid, method, budget, lag.kind, seed)` tuples. `_worker` stays in the script. The script becomes: imports from `neural_particle_method.suite.budget`, argparse with `--jobs`, `--body {explicit_tuned,explicit,explicit_opt}`, `--sids`, `--recipe` (name, default `explicit_opt`, loaded with `load_recipe` when body is `explicit_opt`), and the pool loop calling `run_budget_cell(..., body=..., recipe=...)`.

- [ ] **Step 5: Run tests, goldens, lint, length**

Run: `uv run pytest -q tests/suite/test_budget.py tests/suite/test_offline.py tests/scripts tests/suite/test_grid.py && uv run pytest -q -m golden && uv run ruff check src tests scripts && awk 'length > 100 {print FILENAME": "FNR}' src/neural_particle_method/suite/budget.py src/neural_particle_method/suite/offline.py scripts/budget_sweep.py`
Expected: all PASS, nothing printed by awk.

- [ ] **Step 6: Commit**

```bash
git add src/neural_particle_method/suite/budget.py src/neural_particle_method/suite/offline.py src/neural_particle_method/suite/grid.py scripts/budget_sweep.py tests/suite/test_budget.py tests/suite/test_offline.py tests/scripts/test_budget_sweep.py
git commit -m "suite.budget: budget cell in the package; explicit_opt body trained from a recipe"
```

---

### Task 8: Validation, promotion, suite hooks, CLI

**Files:**
- Create: `src/neural_particle_method/suite/optuna_validate.py`
- Modify: `src/neural_particle_method/bench/algos.py`, `src/neural_particle_method/suite/config.py`, `suite/online.py`, `suite/grid.py`, `suite/tables.py`, `src/neural_particle_method/cli.py`
- Modify tests: `tests/suite/test_online.py`, `test_grid.py`, `test_tables.py`, `tests/suite/test_cold.py`
- Test: `tests/suite/test_optuna_validate.py`

**Interfaces:**
- Consumes: `top_recipes` (Task 6), `run_offline`, `run_budget_cell` (Task 7), `save_recipe`, `recipe_exists`, `load_recipe` (Task 4).
- Produces:
  - `validate(store, study, top=3, jobs=1, sids=SSVI_SIDS, settings=FULL, budget=80_000, seeds=(0, 1)) -> pandas.DataFrame` indexed by `trial_number` (plus a row `nw_resolve`) with columns `mae_surface`, `mae_surface_spot`, `liquid_surface`, `liquid_surface_spot`, `online_s`; also writes `results/optuna/<study>_top.json` (`[{"trial_number", "score", "recipe"}]`).
  - `comparison_table(store, recipes, sids, settings, budget) -> DataFrame` (the numbers above, from `suite_budget_tuned`, two-stage mean over seeds then scenarios).
  - `promote(store, study, trial_number) -> Path` (writes `recipes/explicit_opt.json` from that trial's params).
  - `bench/algos.py`: `_nn_opt` registered as `"explicit_nn_opt"` in `SUITE_ALGOS`.
  - `suite/config.py`: `COLD_ALGOS` unchanged; `cold_algos() -> tuple` = `COLD_ALGOS + ("explicit_nn_opt",)` if `recipe_exists("explicit_opt")` else `COLD_ALGOS`; `grid.cold_jobs` uses `cold_algos()`.
  - `suite/online.py`: `ONLINE_METHODS` gains `"explicit_opt_stale": ("explicit_opt", None)` and `"explicit_opt_spline": ("explicit_opt", "spline")` (after `explicit_tuned_spline`); `grid.online_jobs` skips methods whose body is `explicit_opt` unless `recipe_exists("explicit_opt")`, and gives them `sizes = (max(settings.offline_sizes),)`.
  - `suite/tables.py`: `COLD_ROWS` gains `("explicit_nn_opt", "Explicit NN, searched")` after the tuned row; `BODY_ROWS` gains `("explicit_opt", "Explicit NN, searched")` after the tuned body; `HEAD_ROWS` gains `("explicit_opt_spline", "Searched body + spline head")` after the tuned spline row.
  - CLI: `nparticle optuna clouds [--jobs J] [--sids ...]`, `nparticle optuna run --study S --trials N [--jobs J] [--per-trial 8]`, `nparticle optuna validate --study S [--top 3] [--jobs J] [--budget 80000]`, `nparticle optuna promote --study S --trial N`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/suite/test_optuna_validate.py
import json

import pytest

from neural_particle_method.bench.scenarios import make_tuning_registry
from neural_particle_method.estimators import recipes as R
from neural_particle_method.suite.config import SuiteSettings
from neural_particle_method.suite.optuna_search import run_study
from neural_particle_method.suite.optuna_validate import promote, validate
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()
FAST = {"hidden": 16, "depth": 2, "lr": 1e-2, "batch_size": 0, "first_steps": 5,
        "later_steps": 2, "weight_decay": 0.0, "fit_subsample": 400, "warm_start": True,
        "mean_match": False, "monotone": False, "monotone_penalty": 0.0, "tail": "flat",
        "hetero": False}


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def test_validate_and_promote_on_tiny(store, tmp_path, monkeypatch):
    import neural_particle_method.suite.optuna_search as S
    monkeypatch.setattr(S, "suggest_recipe", lambda trial: {**FAST, "lr": trial.suggest_float(
        "lr", 1e-3, 1e-2, log=True)})
    monkeypatch.setattr(R, "RECIPE_DIR", tmp_path / "recipes")
    monkeypatch.chdir(tmp_path)
    run_study(store, "unit", n_trials=2, n_jobs=1, settings=TINY, sids=("t01",), per_trial=1,
              times=(0.5, 1.0), registry=make_tuning_registry(), storage_dir=tmp_path / "o",
              cache_dir=tmp_path / "c")
    df = validate(store, "unit", top=1, jobs=1, sids=("s01",), settings=TINY,
                  budget=TINY.n_online, seeds=(0,))
    assert set(df.columns) == {"mae_surface", "mae_surface_spot", "liquid_surface",
                               "liquid_surface_spot", "online_s"}
    assert "nw_resolve" in df.index and len(df) == 2
    top = json.loads((tmp_path / "results" / "optuna" / "unit_top.json").read_text())
    assert len(top) == 1 and set(top[0]) == {"trial_number", "score", "recipe"}
    bodies = store.search(TINY.experiment("suite_offline"))
    assert (bodies["params.body"] == "explicit_opt").sum() == 1
    cells = store.search("suite_budget_tuned")
    assert set(cells["params.method"]) == {"explicit_opt_spline", "nw_resolve"}
    p = promote(store, "unit", top[0]["trial_number"])
    assert p == tmp_path / "recipes" / "explicit_opt.json"
    assert R.load_recipe("explicit_opt")["hidden"] == 16
```

Edits to existing tests:
- `tests/suite/test_online.py::test_method_table_is_the_agreed_one`: the list gains `"explicit_opt_stale", "explicit_opt_spline"` after `"explicit_tuned_spline"`. The parametrized "every method runs" test must skip the two `explicit_opt_*` methods unless a recipe exists: add at the top of that test `if method.startswith("explicit_opt"): pytest.importorskip("optuna"); monkeypatch RECIPE_DIR to tmp_path and save FAST there first` — concretely, give the test a `monkeypatch` fixture and, for those two methods, `monkeypatch.setattr(R, "RECIPE_DIR", tmp_path)` then `R.save_recipe("explicit_opt", FAST)` before calling `run_online`.
- `tests/suite/test_grid.py::test_full_job_counts`: unchanged numbers when no recipe file exists; add `test_job_lists_grow_when_a_recipe_is_promoted(tmp_path, monkeypatch)`: monkeypatch `R.RECIPE_DIR`, `save_recipe("explicit_opt", FAST)`, then `len(cold_jobs(FULL)) == 23 * 11 * 2` and `len(online_jobs(FULL)) == 23 * (8 * 2 + 4 + 2 + 1 + 1) * 2 * 2`.
- `tests/suite/test_cold.py::test_full_settings_match_the_spec`: `COLD_ALGOS` unchanged (10); add an assertion `cold_algos() == COLD_ALGOS` when no recipe exists.
- `tests/suite/test_tables.py`: the row constants gain the three new entries (assert `("explicit_nn_opt", "Explicit NN, searched") in COLD_ROWS`, etc.); the fixture has no such runs, so the new rows render `--` (assert the `suite_cold.tex` line `Explicit NN, searched & -- & -- & -- & -- & -- & --\\`).

- [ ] **Step 2: Run them**

Run: `uv run pytest -q tests/suite/test_optuna_validate.py tests/suite/test_online.py tests/suite/test_grid.py tests/suite/test_tables.py tests/suite/test_cold.py`
Expected: FAIL on the new imports and lists.

- [ ] **Step 3: Implement the suite hooks**

`bench/algos.py`:

```python
def _nn_opt(sc, n, seed, e, i, knobs=None):
    """Per-slice network with the promoted Optuna recipe (estimators/recipes/explicit_opt.json)."""
    recipe = load_recipe("explicit_opt")
    est = regressor_from_recipe(recipe, seed=seed,
                                monotone_sign=-float(np.sign(sc.dynamics.rho) or 1.0),
                                **(knobs or {}))
    return _explicit(sc, n, seed, explicit_config_from_recipe(e, recipe), "nn", estimator=est)


SUITE_ALGOS = {"explicit_nn_tuned": _nn_tuned, "explicit_nn_opt": _nn_opt}
```

(`regressor_from_recipe` accepts `keep_slice_weights` through `knobs`; `run_cold` passes `{"keep_slice_weights": True}` for `explicit_nn_opt` as it does for the other explicit NN rows — update `suite/cold.py` accordingly.) Imports: `numpy as np`, `from ..estimators.recipes import explicit_config_from_recipe, load_recipe, regressor_from_recipe`.

`suite/config.py`:

```python
def cold_algos():
    """The cold rows to run: the agreed ten, plus the searched network once a recipe is promoted."""
    return COLD_ALGOS + ("explicit_nn_opt",) if recipe_exists("explicit_opt") else COLD_ALGOS
```

`suite/grid.py`: `cold_jobs` iterates `cold_algos()`; `online_jobs` computes `body, _ = ONLINE_METHODS[method]`, skips when `body == "explicit_opt" and not recipe_exists("explicit_opt")`, and uses `sizes = (max(settings.offline_sizes),)` for bodies `explicit_tuned` and `explicit_opt`. `run_stage`'s online precheck iterates `STAGE_BODIES` (Task 7) so the opt body is trained lazily by `run_online` through `run_offline` (which loads the promoted recipe when `recipe=None`).

`suite/online.py`: the two `ONLINE_METHODS` entries. `run_online` needs no change.

`suite/tables.py`: the three row entries.

- [ ] **Step 4: Implement `suite/optuna_validate.py`**

```python
"""Validate the top recipes of a study as frozen bodies plus spline head at the practical online
budget, against NW re-solved at the same budget; promote the winner into the estimator recipes."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..estimators.recipes import coerce_recipe, save_recipe
from ..tracking.store import Store
from .budget import EXPERIMENTS, run_budget_cell
from .config import FULL, SSVI_SIDS
from .lag import LAGS
from .offline import run_offline
from .optuna_search import STORAGE_DIR, top_recipes

LAG_BY_KIND = {lag.kind: lag for lag in LAGS}


def _cell(args):
    uri, root, sid, method, budget, lag_kind, seed, settings, recipe = args
    store = Store(uri, root)
    if method == "body":
        return run_offline(store, sid, "explicit_opt", settings.offline_sizes[-1], settings,
                           recipe=recipe)
    return run_budget_cell(store, sid, method, budget, LAG_BY_KIND[lag_kind], seed,
                           body="explicit_opt" if method != "nw_resolve" else "explicit_tuned",
                           settings=settings, recipe=recipe)


def _run_all(store, jobs, n_jobs):
    if n_jobs == 1:
        return [_cell(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(_cell, jobs))


def comparison_table(store, recipes, sids, settings=FULL, budget=80_000):
    """Rows: one per recipe (index = trial_number) plus 'nw_resolve'; the two-stage mean of
    pooled and liquid MAE per lag and the median online seconds, from the budget experiment."""
    df = store.search(EXPERIMENTS["explicit_opt"])
    df = df[(df["status"] == "FINISHED") & df["params.sid"].isin(sids)
            & (pd.to_numeric(df["params.budget"]) == budget)]
    rows = {}
    targets = [("nw_resolve", df[df["params.method"] == "nw_resolve"])]
    targets += [(r["trial_number"], df[(df["params.method"] == "explicit_opt_spline")
                                       & (df.get("params.recipe_hash") == r["hash"])])
                for r in recipes]
    for name, sel in targets:
        row = {}
        for lag in LAG_BY_KIND:
            x = sel[sel["params.lag"] == lag]
            for col, key in (("mae", "metrics.pooled_mae_bp"), ("liquid", "metrics.liquid_mae_bp")):
                per = x.groupby("params.sid")[key].apply(lambda v: pd.to_numeric(v).mean())
                row[f"{col}_{lag}"] = float(per.mean()) if len(per) else np.nan
        row["online_s"] = float(pd.to_numeric(sel["metrics.online_s"]).median()) if len(sel) else np.nan
        rows[name] = row
    return pd.DataFrame(rows).T


def validate(store, study, top=3, jobs=1, sids=SSVI_SIDS, settings=FULL, budget=80_000,
             seeds=(0, 1)):
    from ..estimators.recipes import recipe_hash
    best = top_recipes(store, study, top)
    out = Path(STORAGE_DIR) / f"{study}_top.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([{k: b[k] for k in ("trial_number", "score", "recipe")}
                               for b in best], indent=1))
    uri, root = store.tracking_uri, store.artifact_root
    for b in best:
        b["hash"] = recipe_hash(b["recipe"])
        _run_all(store, [(uri, root, sid, "body", 0, "surface", 0, settings, b["recipe"])
                         for sid in sids], jobs)
        _run_all(store, [(uri, root, sid, "explicit_opt_spline", budget, lag.kind, seed, settings,
                          b["recipe"]) for sid in sids for lag in LAGS for seed in seeds], jobs)
    _run_all(store, [(uri, root, sid, "nw_resolve", budget, lag.kind, seed, settings, None)
                     for sid in sids for lag in LAGS for seed in seeds], jobs)
    table = comparison_table(store, best, sids, settings, budget)
    print(table.round(1).to_string())
    return table


def promote(store, study, trial_number):
    for b in top_recipes(store, study, 10_000):
        if b["trial_number"] == trial_number:
            return save_recipe("explicit_opt", coerce_recipe(b["recipe"]))
    raise KeyError(f"trial {trial_number} is not a COMPLETE trial of study {study!r}")
```

Notes for the implementer: `run_budget_cell` for `nw_resolve` ignores the body except for the experiment name, so passing `body="explicit_tuned"` keeps it in `suite_budget_tuned` without a recipe. `_cell` is a module-level function (spawn-safe). `df.get("params.recipe_hash")` returns None when the column is absent; guard with `"params.recipe_hash" in df.columns`.

- [ ] **Step 5: CLI**

In `cli.py`, after the `suite` subparsers:

```python
    p = sub.add_parser("optuna")
    os_ = p.add_subparsers(dest="optuna_cmd", required=True)
    q = os_.add_parser("clouds"); q.add_argument("--jobs", type=int, default=1)
    q.add_argument("--sids", nargs="*", default=None)
    q = os_.add_parser("run"); q.add_argument("--study", required=True)
    q.add_argument("--trials", type=int, required=True); q.add_argument("--jobs", type=int, default=1)
    q.add_argument("--per-trial", type=int, default=8)
    q = os_.add_parser("validate"); q.add_argument("--study", required=True)
    q.add_argument("--top", type=int, default=3); q.add_argument("--jobs", type=int, default=1)
    q.add_argument("--budget", type=int, default=80_000)
    q = os_.add_parser("promote"); q.add_argument("--study", required=True)
    q.add_argument("--trial", type=int, required=True)
```

and in `main`:

```python
    if args.cmd == "optuna":
        from .suite.config import TUNING_SIDS
        from .suite.optuna_clouds import ensure_cloud
        from .suite.optuna_search import run_study
        from .suite.optuna_validate import promote, validate
        if args.optuna_cmd == "clouds":
            sids = tuple(args.sids or TUNING_SIDS)
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                for sid, rid in zip(sids, ex.map(_cloud_worker, [(store.tracking_uri,
                                                                  store.artifact_root, s)
                                                                 for s in sids])):
                    print("cloud", sid, rid, flush=True)
            return 0
        if args.optuna_cmd == "run":
            print("study", run_study(store, args.study, args.trials, n_jobs=args.jobs,
                                     per_trial=args.per_trial))
            return 0
        if args.optuna_cmd == "validate":
            validate(store, args.study, top=args.top, jobs=args.jobs, budget=args.budget)
            return 0
        if args.optuna_cmd == "promote":
            print("wrote", promote(store, args.study, args.trial))
            return 0
```

with a module-level `_cloud_worker(args)` that builds a `Store(uri, root)` and calls `ensure_cloud(store, sid)`. Add a test to `tests/test_cli.py` (or the existing CLI test file) that `build_parser().parse_args(["optuna", "run", "--study", "x", "--trials", "3"])` parses with `per_trial == 8` — locate the parser factory's real name first (`ap` is built in a function in `cli.py`; use that function).

- [ ] **Step 6: Run everything**

Run: `uv run pytest -q -m "not slow" && uv run pytest -q -m golden && git status --short tests/golden && uv run ruff check src tests scripts && awk 'length > 100 {print FILENAME": "FNR}' $(git diff --name-only HEAD -- '*.py') && uv run pytest -m slow tests/suite/test_smoke.py -q`
Expected: all green, goldens untouched, nothing from awk, smoke test wall time reported.

- [ ] **Step 7: Spec and notes stub, commit**

Add to `docs/superpowers/specs/2026-09-11-optuna-offline-design.md` a short "Implementation notes" section: `STAGE_BODIES` vs `BODIES`, the recipe-file gating of `cold_algos()`/`online_jobs`, `monotone_sign = -sign(rho)`. Do not touch `paper/notes_experiments.tex` (the controller writes the section after the study).

```bash
git add -A src tests scripts docs
git commit -m "optuna: validation at the practical budget, promotion into the suite, CLI"
```

---

## Self-review

- **Spec coverage.** §1 tuning set → Task 1; clouds, split, NW loss → Task 5. §2 knobs → Task 2 (regressor), Task 3 (tail), Task 4 (recipe mapping; `fit_subsample` and `tail` via `explicit_config_from_recipe`). §3 objective, per-trial scenario draw, pruner → Task 6. §4 MLflow parent/child, storage artifact, resume, `top_recipes` from MLflow → Task 6. §5 validation at 80k, `explicit_opt` body with `recipe_hash`, `explicit_opt_spline` cells, promotion file, `explicit_nn_opt` cold row, table rows → Tasks 7–8. §6 tests → each task. §7 run plan → controller.
- **Placeholders.** The `RecordingNW` block in Task 5 Step 3 contains a placeholder `fit_predict` that the text tells the implementer to delete; the real method is given immediately after. Task 6's `top_recipes` `__globals__` hack is likewise replaced by the plain import stated below it. No TBDs remain.
- **Type consistency.** `regressor_from_recipe(recipe, seed=0, keep_slice_weights=False, monotone_sign=1.0)` is used with those names in Tasks 6, 7, 8. `run_budget_cell(store, sid, method, budget, lag, seed, body=..., settings=..., recipe=...)` matches Tasks 7 and 8. `Cloud` fields (`sid, rho, times, fit, held, nw_loss`) match Tasks 5 and 6. `top_recipes` returns dicts with `trial_number, score, recipe, run_id`; `validate` adds `hash` locally. `STAGE_BODIES` is defined in Task 7 and used in Task 8.
