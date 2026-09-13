# Importance Sampling Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `nparticle tilt` suite stage that measures whether the Girsanov tilt buys budget equivalence, in an isolated slice layer and end to end, with tables and figures for a new notes section.

**Architecture:** A `TiltDesign` (schedule plus cost cap) builds the existing `MixtureDesign`, now allowed to carry a per-step schedule; `calibrate_explicit` and a new tilted `online_sweep` accumulate the same weights. `suite/tilt.py` holds the three cells (slice, cold, online job lists), `suite/tilt_tables.py` the scoring and tables, `suite/tilt_figures.py` the figures, all logged to MLflow experiments `tilt_slices`, `tilt_cold`, `tilt_online`.

**Tech Stack:** Python 3.12, numpy, torch, scipy, pandas, mlflow (`tracking.store.Store`), matplotlib (Agg), pytest, ruff, `uv run`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-13-importance-sampling-design.md`. Branch `refactor-mlflow`.
- Bit-for-bit: `uv run pytest -m golden tests/test_golden_tiny.py -q` must pass unchanged after every task (it replays `explicit_nn_is` with `design_mixture`). Untilted code paths must be arithmetically untouched.
- `TiltDesign("constant", 3).mixture(n_steps, T, rho)` must reproduce `design_mixture(dynamics, T)`'s thetas whenever that cap binds (it does on every suite scenario).
- Schedules: `constant`: `theta = sqrt(cost * (1 - rho**2) / T)`; `inverse_sqrt`: `theta_k = c / sqrt(max(k*dt, dt))` with `c = sqrt(cost * (1 - rho**2) / (1 + H_{n-1}))`, `H_m = sum_{j=1}^{m} 1/j`. Defensive weight `alpha0 = 0.5`. Design names `f"{schedule}-{cost:g}"`, untilted `"none"`.
- `DESIGNS = [TiltDesign(s, c) for s in ("constant", "inverse_sqrt") for c in (1, 3, 9)]`.
- Experiments: `tilt_slices`, `tilt_cold`, `tilt_online` (through `settings.experiment(...)`, so `_smoke`/`_tiny` suffixes apply).
- `SLICE_SIDS = ("s01", "s02", "s05", "s09", "s11", "s16")`; layer-1 particles `(10_000, 30_000, 80_000)`, seeds `(0, 1, 2, 3, 4)`; cold particles `(10_000, 30_000, 80_000)`, seeds `(0, 1)`; online budgets `(10_000, 80_000)`, seeds `(0, 1)`.
- Job counts (FULL): slices 630, cold 312 (NW on 20 sids + `explicit_nn_opt` on 6, x3 budgets x2 seeds x2 designs), online 160.
- Winner: smallest NW wing std at N = 30k averaged over `SLICE_SIDS` and the four maturities, disqualified if `|bias| > 1.5 * |untilted bias| + 1.0` (percent of f; the one-point floor keeps a near-zero untilted bias from disqualifying every tilt).
- Slice cells skip any quoted maturity that snaps to t = 0 (a one-point cloud has no bandwidth); the doc lists the snapped maturities actually scored, deduplicated in order.
- The untilted online partners are the `suite_budget_tuned` cells of the promoted recipe (`params.recipe_hash`), which is where `scripts/budget_sweep.py --body explicit_opt` writes them.
- Heads: only `SplineHead` accepts weights; `RKHSHead`/`FeatureRidgeHead` raise `ValueError` on non-None weights.
- Line length 100 (ruff does not enforce E501; check with `awk 'length > 100' <file>`). Tests under `tests/`, `uv run pytest -q` green, `uv run ruff check src tests` clean after every task.
- Commit trailer on every commit:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01EhbyGqFA5kYDyMwYouSgvS`.
- Never run a full-size (FULL) suite or tilt command; tests use `SuiteSettings.tiny()` and the smoke test uses `SMOKE`.

---

## File map

- Modify `src/neural_particle_method/calibrate/importance.py`: `MixtureDesign.thetas` may be `(3, n_steps)`; add `TiltDesign`, `DESIGNS`, `design_by_name`.
- Modify `src/neural_particle_method/calibrate/explicit.py`: scheduled mixture, `ess_min_slice` in `is_diag`.
- Modify `src/neural_particle_method/estimators/spline.py`: weighted P-spline.
- Modify `src/neural_particle_method/suite/heads.py`: `weights` on every head, tilted `online_sweep` returning `(field, fit_s, is_diag)`.
- Modify `src/neural_particle_method/bench/algos.py`, `bench/runner.py`: `mixture=` plumbing.
- Modify `src/neural_particle_method/suite/budget.py`: `design=` on `run_budget_cell`.
- Modify `src/neural_particle_method/suite/online.py`: unpack the third return of `online_sweep`.
- Create `src/neural_particle_method/suite/tilt.py`: constants, `run_slice_cell`, `run_tilt_cold`, job lists, `run_tilt_job`, `run_tilt_stage`.
- Create `src/neural_particle_method/suite/tilt_tables.py`: `slice_frame`, `winner`, `cold_frame`, `online_frame`, `tilt_tables`.
- Create `src/neural_particle_method/suite/tilt_figures.py`: `make_figures`.
- Modify `src/neural_particle_method/cli.py`: `nparticle tilt ...`.
- Modify `paper/notes_experiments.tex`: section "Importance sampling".
- Tests: `tests/calibrate/test_tilt_design.py`, `tests/estimators/test_spline.py`, `tests/suite/test_heads.py`, `tests/suite/test_tilt.py`, `tests/suite/test_tilt_tables.py`, `tests/suite/test_tilt_figures.py`, `tests/test_cli_tilt.py`, `tests/suite/test_smoke.py`.

---

### Task 1: `TiltDesign` and the scheduled mixture in the explicit calibrator

**Files:**
- Modify: `src/neural_particle_method/calibrate/importance.py`
- Modify: `src/neural_particle_method/calibrate/explicit.py` (the `mixture` block of `calibrate_explicit`, lines ~78-136)
- Test: `tests/calibrate/test_tilt_design.py`

**Interfaces:**
- Produces: `TiltDesign(schedule: str, cost: float, alpha0: float = 0.5)` with `.name`, `.mixture(n_steps, T, rho) -> MixtureDesign`; `DESIGNS: list[TiltDesign]`; `design_by_name(name) -> TiltDesign | None` (`"none"` -> `None`, unknown -> `KeyError`); `MixtureDesign.thetas` may be a `(3, n_steps)` ndarray; `ExplicitResult.is_diag["ess_min_slice"]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/calibrate/test_tilt_design.py
import numpy as np
import pytest

from neural_particle_method.bench.scenarios import make_registry
from neural_particle_method.calibrate.config import ExplicitConfig
from neural_particle_method.calibrate.explicit import calibrate_explicit
from neural_particle_method.calibrate.importance import (
    DESIGNS,
    TiltDesign,
    design_by_name,
    design_mixture,
)
from neural_particle_method.estimators import make_estimator


def _cost(thetas, dt, rho):
    th = np.asarray(thetas, dtype=float)
    if th.ndim == 1:
        th = th[:, None] * np.ones((3, int(round(1.0 / dt))))
    return float((th[2] ** 2).sum() * dt / (1 - rho ** 2))


@pytest.mark.parametrize("sid", ["s01", "s11"])
def test_constant_cost_3_reproduces_design_mixture(sid):
    sc = make_registry()[sid]
    ref = design_mixture(sc.dynamics.to_dict(), sc.T)
    got = TiltDesign("constant", 3.0).mixture(200, sc.T, sc.dynamics.rho)
    assert got.alphas == ref.alphas and got.rho == ref.rho
    np.testing.assert_allclose(np.asarray(got.thetas), np.asarray(ref.thetas), rtol=1e-12)


def test_inverse_sqrt_schedule_has_the_same_total_cost():
    n, T, rho = 200, 2.0, -0.6
    d = TiltDesign("inverse_sqrt", 3.0).mixture(n, T, rho)
    th = np.asarray(d.thetas)
    assert th.shape == (3, n) and np.all(th[1] == 0.0) and np.all(th[2] > 0) and np.all(th[0] < 0)
    assert th[2, 0] == th[2, 1]                     # max(t, dt) at k = 0 equals dt
    assert th[2, 1] > th[2, -1]                     # decays in t
    assert abs((th[2] ** 2).sum() * (T / n) / (1 - rho ** 2) - 3.0) < 1e-12
    c = TiltDesign("constant", 3.0).mixture(n, T, rho)
    assert abs(c.thetas[2] ** 2 * T / (1 - rho ** 2) - 3.0) < 1e-12


def test_names_and_lookup():
    assert [d.name for d in DESIGNS] == ["constant-1", "constant-3", "constant-9",
                                         "inverse_sqrt-1", "inverse_sqrt-3", "inverse_sqrt-9"]
    assert design_by_name("none") is None
    assert design_by_name("inverse_sqrt-9") == TiltDesign("inverse_sqrt", 9.0)
    with pytest.raises(KeyError):
        design_by_name("constant-2")
    with pytest.raises(ValueError):
        TiltDesign("linear", 1.0).mixture(4, 1.0, -0.5)


def test_scheduled_and_constant_mixtures_agree_when_the_schedule_is_flat():
    sc = make_registry()["s01"]
    cfg = ExplicitConfig(n_steps=6, n_particles=3_000, fit_subsample=1_000)
    const = TiltDesign("constant", 3.0).mixture(6, sc.T, sc.dynamics.rho)
    flat = type(const)(const.alphas, np.asarray(const.thetas)[:, None] * np.ones((3, 6)), const.rho)
    a = calibrate_explicit(sc.local_vol(), sc.dynamics, make_estimator("nw"), cfg, s0=sc.s0,
                           T=sc.T, seed=1, mixture=const)
    b = calibrate_explicit(sc.local_vol(), sc.dynamics, make_estimator("nw"), cfg, s0=sc.s0,
                           T=sc.T, seed=1, mixture=flat)
    np.testing.assert_array_equal(a.weights, b.weights)
    for sa, sb in zip(a.field, b.field):
        np.testing.assert_array_equal(sa.L, sb.L)
    assert 0 < a.is_diag["ess_min_slice"] <= 1.0
    assert a.is_diag["ess_min_slice"] <= a.is_diag["ess_frac"] + 1e-12


def test_bit_for_bit_against_design_mixture_on_a_tiny_run():
    sc = make_registry()["s01"]
    cfg = ExplicitConfig(n_steps=4, n_particles=2_000, fit_subsample=600)
    ref = design_mixture(sc.dynamics.to_dict(), sc.T)
    new = TiltDesign("constant", 3.0).mixture(4, sc.T, sc.dynamics.rho)
    a = calibrate_explicit(sc.local_vol(), sc.dynamics, make_estimator("nw"), cfg, s0=sc.s0,
                           T=sc.T, seed=0, mixture=ref)
    b = calibrate_explicit(sc.local_vol(), sc.dynamics, make_estimator("nw"), cfg, s0=sc.s0,
                           T=sc.T, seed=0, mixture=new)
    np.testing.assert_array_equal(a.weights, b.weights)
    for sa, sb in zip(a.field, b.field):
        np.testing.assert_array_equal(sa.f, sb.f)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/calibrate/test_tilt_design.py -q`
Expected: FAIL with `ImportError: cannot import name 'TiltDesign'`.

- [ ] **Step 3: Implement `TiltDesign` in `calibrate/importance.py`**

Replace the file's body after the imports with:

```python
SCHEDULES = ("constant", "inverse_sqrt")


@dataclass(frozen=True)
class MixtureDesign:
    """Three-component defensive mixture: tilts (-theta, 0, +theta) with weights `alphas`.

    `thetas` is either three floats (a constant tilt, the original design) or a `(3, n_steps)`
    array (a schedule in time); `etas` are the tilts on the orthogonal Brownian direction."""
    alphas: tuple
    thetas: object
    rho: float

    @property
    def etas(self):
        r = np.sqrt(1 - self.rho ** 2)
        th = np.asarray(self.thetas, dtype=float)
        if th.ndim == 2:
            return th / r
        return tuple(t / r for t in self.thetas)

    @property
    def scheduled(self):
        return np.asarray(self.thetas).ndim == 2


def design_mixture(dynamics, T, k_target=0.47, alpha0=0.5, ess_cost_cap=3.0):
    """Moment-matched wing tilts with a defensive untilted component and a kernel-cost cap."""
    p = HestonParams.from_dict(dynamics)
    rho, v0, theta_bar = p.rho, p.v0, p.theta
    sig_bar = np.sqrt(0.5 * (v0 + theta_bar))
    th = k_target / (sig_bar * T)
    th_cap = np.sqrt(ess_cost_cap * (1 - rho ** 2) / T)
    th = min(th, th_cap)
    a_wing = 0.5 * (1 - alpha0)
    return MixtureDesign(alphas=(a_wing, alpha0, a_wing), thetas=(-th, 0.0, th), rho=rho)


@dataclass(frozen=True)
class TiltDesign:
    """A tilt schedule and its cost cap: `cost` bounds sum_k theta_k^2 dt / (1 - rho^2) over [0, T]
    (the log-weight variance of a pure tilt; twice its relative entropy)."""
    schedule: str
    cost: float
    alpha0: float = 0.5

    @property
    def name(self):
        return f"{self.schedule}-{self.cost:g}"

    def mixture(self, n_steps, T, rho):
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}, got {self.schedule!r}")
        a_wing = 0.5 * (1 - self.alpha0)
        alphas = (a_wing, self.alpha0, a_wing)
        r2 = 1 - rho ** 2
        if self.schedule == "constant":
            th = float(np.sqrt(self.cost * r2 / T))
            return MixtureDesign(alphas=alphas, thetas=(-th, 0.0, th), rho=rho)
        dt = T / n_steps
        t = np.arange(n_steps) * dt
        harmonic = 1.0 + np.sum(1.0 / np.arange(1, n_steps))     # 1 + H_{n-1}
        c = np.sqrt(self.cost * r2 / harmonic)
        th = c / np.sqrt(np.maximum(t, dt))
        thetas = np.stack([-th, np.zeros(n_steps), th])
        return MixtureDesign(alphas=alphas, thetas=thetas, rho=rho)


DESIGNS = [TiltDesign(s, float(c)) for s in SCHEDULES for c in (1, 3, 9)]
UNTILTED = "none"


def design_by_name(name):
    """`"none"` is the untilted arm (None); anything else must be one of DESIGNS."""
    if name == UNTILTED:
        return None
    for d in DESIGNS:
        if d.name == name:
            return d
    raise KeyError(f"unknown tilt design {name!r}; choose from {[d.name for d in DESIGNS]}")
```

Check the total cost of the inverse-sqrt schedule by hand: `sum_k th_k^2 dt = c^2 sum_k dt / max(k dt, dt) = c^2 (1 + H_{n-1})`, so `cost = c^2 (1 + H_{n-1}) / r2`, which is the `c` above.

- [ ] **Step 4: Scheduled mixture in `calibrate_explicit`**

In `src/neural_particle_method/calibrate/explicit.py`, replace the mixture set-up block

```python
    theta_p = None
    if mixture is not None:
        comp = rng.choice(3, size=n_particles, p=list(mixture.alphas))
        theta_p = np.array(mixture.thetas)[comp]
        etas = np.array(mixture.etas)
        eta_p = etas[comp]
        ell = np.zeros((3, n_particles))
    w = None
```

with

```python
    theta_p = None
    ess_min = 1.0
    if mixture is not None:
        comp = rng.choice(3, size=n_particles, p=list(mixture.alphas))
        scheduled = mixture.scheduled
        thetas_all, etas_all = np.asarray(mixture.thetas), np.asarray(mixture.etas)
        if not scheduled:
            theta_p = np.array(mixture.thetas)[comp]
            etas = np.array(mixture.etas)
            eta_p = etas[comp]
        ell = np.zeros((3, n_particles))
    w = None
```

and at the top of the step loop body (first statement inside `for k in range(n_steps):`) add

```python
        if mixture is not None and scheduled:
            theta_p = thetas_all[:, k][comp]
            etas = etas_all[:, k]
            eta_p = etas[comp]
```

and in the weight update block, after `w = 1.0 / (...)`, add

```python
            ess_min = min(ess_min, float(w.sum() ** 2 / (n_particles * (w ** 2).sum())))
```

and extend `is_diag`:

```python
        is_diag = {"max_w": float(w.max()),
                   "ess_frac": float(w.sum() ** 2 / (len(w) * (w ** 2).sum())),
                   "ess_min_slice": ess_min}
```

The constant path keeps `np.array(mixture.thetas)[comp]` and `np.array(mixture.etas)` exactly as before, so it is bit-for-bit.

- [ ] **Step 5: Run the tests, the goldens, ruff**

Run: `uv run pytest tests/calibrate/test_tilt_design.py tests/calibrate/test_importance.py -q && uv run pytest -m golden tests/test_golden_tiny.py -q && uv run ruff check src tests && awk 'length > 100' src/neural_particle_method/calibrate/importance.py src/neural_particle_method/calibrate/explicit.py`
Expected: all PASS, no ruff findings, no long lines.

- [ ] **Step 6: Commit**

```bash
git add src/neural_particle_method/calibrate/importance.py src/neural_particle_method/calibrate/explicit.py tests/calibrate/test_tilt_design.py
git commit -m "calibrate: TiltDesign (constant / inverse-sqrt schedules under a cost cap), scheduled mixtures, per-slice ESS" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01EhbyGqFA5kYDyMwYouSgvS"
```

---

### Task 2: Weighted P-spline, weighted heads, tilted online sweep

**Files:**
- Modify: `src/neural_particle_method/estimators/spline.py`
- Modify: `src/neural_particle_method/suite/heads.py`
- Modify: `src/neural_particle_method/suite/online.py` (the `online_sweep(...)` call), `src/neural_particle_method/suite/budget.py` (same), `src/neural_particle_method/suite/optuna_validate.py` if it calls `online_sweep` (grep `online_sweep(` under `src/` and update every unpacking to three values)
- Test: `tests/estimators/test_spline.py`, `tests/suite/test_heads.py`

**Interfaces:**
- Consumes: `MixtureDesign` (Task 1).
- Produces: `spline_estimate(lnx, v, lnx_grid, n_knots=25, lam=1.0, degree=3, weights=None)`; `PSpline.supports_weights = True`; every head's `correction(self, t, lnx, v_plus, f_stale, model, grid, weights=None)`; `online_sweep(model, local_vol, params, s0, T, cfg=ExplicitConfig(), head=None, seed=0, mixture=None) -> (field, fit_s, is_diag)` with `is_diag` `None` when untilted, else `{"max_w", "ess_frac", "ess_min_slice"}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/estimators/test_spline.py`:

```python
import pytest

from neural_particle_method.estimators.spline import spline_estimate


def test_unit_weights_equal_the_unweighted_fit():
    rng = np.random.default_rng(4)
    lnx = rng.uniform(-0.5, 0.5, 2_000)
    v = 0.04 + 0.1 * lnx ** 2 + rng.normal(scale=0.01, size=lnx.size)
    g = np.linspace(-0.3, 0.3, 13)
    a = spline_estimate(lnx, v, g)
    b = spline_estimate(lnx, v, g, weights=np.ones_like(v))
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


def test_weight_two_equals_a_duplicated_point():
    rng = np.random.default_rng(5)
    lnx = rng.uniform(-0.5, 0.5, 500)
    v = 0.04 + 0.1 * lnx ** 2 + rng.normal(scale=0.01, size=lnx.size)
    g = np.linspace(-0.3, 0.3, 13)
    w = np.ones_like(v); w[:50] = 2.0
    # knots are quantiles of the sample; pin them so both fits share the same basis
    q = np.quantile(lnx, np.linspace(0.005, 0.995, 25))
    a = spline_estimate(lnx, v, g, weights=w, knots=q)
    b = spline_estimate(np.concatenate([lnx, lnx[:50]]), np.concatenate([v, v[:50]]), g, knots=q)
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_pspline_accepts_weights():
    rng = np.random.default_rng(6)
    lnx = rng.uniform(-0.5, 0.5, 1_000)
    v = 0.04 + 0.1 * lnx ** 2
    g = np.linspace(-0.3, 0.3, 7)
    est = PSpline(lam=1e-3)
    assert est.supports_weights
    f = est.fit_predict(0.0, lnx, v, g, weights=np.ones_like(v))
    assert np.abs(f - (0.04 + 0.1 * g ** 2)).max() < 1e-3
    with pytest.raises(ValueError):
        est.fit_predict(0.0, lnx, v, g, weights=-np.ones_like(v))
```

Append to `tests/suite/test_heads.py`:

```python
from neural_particle_method.calibrate.importance import TiltDesign


def test_heads_take_weights_and_only_the_spline_uses_them():
    _, bank, _ = _body()
    rng = np.random.default_rng(7)
    x = rng.normal(scale=0.1, size=1500)
    f = bank.f(0.5, x)
    resid = 0.01 * np.cos(4 * x)
    grid = np.linspace(-0.15, 0.15, 7)
    w = np.ones_like(x)
    a = SplineHead().correction(0.5, x, f + resid, f, bank, grid)
    b = SplineHead().correction(0.5, x, f + resid, f, bank, grid, weights=w)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)
    for head in (RKHSHead(), FeatureRidgeHead()):
        head.correction(0.5, x, f + resid, f, bank, grid, weights=None)
        with pytest.raises(ValueError, match="weights"):
            head.correction(0.5, x, f + resid, f, bank, grid, weights=w)


def test_online_sweep_untilted_is_unchanged_and_returns_no_diag():
    sc, bank, _ = _body()
    field, fit_s, diag = online_sweep(bank, sc.local_vol(), sc.dynamics, sc.s0, sc.T, E,
                                      head=SplineHead(), seed=3)
    assert diag is None and fit_s >= 0 and len(field) == E.n_steps


def test_online_sweep_with_a_mixture_weights_the_head():
    sc, bank, _ = _body()
    mix = TiltDesign("inverse_sqrt", 3.0).mixture(E.n_steps, sc.T, sc.dynamics.rho)
    field, fit_s, diag = online_sweep(bank, sc.local_vol(), sc.dynamics, sc.s0, sc.T, E,
                                      head=SplineHead(), seed=3, mixture=mix)
    assert set(diag) == {"max_w", "ess_frac", "ess_min_slice"}
    assert diag["max_w"] <= 1 / mix.alphas[1] + 1e-9 and 0 < diag["ess_min_slice"] <= 1
    assert len(field) == E.n_steps and np.all(np.isfinite(field[-1].L))
    with pytest.raises(ValueError, match="weights"):
        online_sweep(bank, sc.local_vol(), sc.dynamics, sc.s0, sc.T, E, head=RKHSHead(), seed=3,
                     mixture=mix)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/estimators/test_spline.py tests/suite/test_heads.py -q`
Expected: FAIL (`TypeError: unexpected keyword argument 'weights'` / `'knots'` / `'mixture'`; the untilted sweep test fails on unpacking three values).

- [ ] **Step 3: Weighted P-spline**

Replace `spline_estimate` and `PSpline` in `src/neural_particle_method/estimators/spline.py`:

```python
def spline_estimate(lnx, v, lnx_grid, n_knots=25, lam=1.0, degree=3, weights=None, knots=None):
    """Penalised cubic B-spline (P-spline) estimate of E[V | ln X = .] on a grid.

    `weights` (optional, positive) are importance weights: weighted least squares on the data
    term, penalty unchanged. `knots` overrides the quantile knots (tests)."""
    from scipy.interpolate import BSpline
    q = np.quantile(lnx, np.linspace(0.005, 0.995, n_knots)) if knots is None else np.asarray(knots)
    q = np.unique(q)
    t = np.concatenate([[q[0]] * degree, q, [q[-1]] * degree])
    lo, hi = t[degree], t[-degree - 1]
    x = np.clip(lnx, lo, hi - 1e-12)
    B = BSpline.design_matrix(x, t, degree).toarray()
    p = B.shape[1]
    D2 = np.diff(np.eye(p), n=2, axis=0)
    if weights is None:
        gram, rhs = B.T @ B, B.T @ v
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != np.shape(v) or np.any(w < 0):
            raise ValueError("spline weights must be non-negative and one per particle")
        gram, rhs = B.T @ (w[:, None] * B), B.T @ (w * v)
    c = np.linalg.solve(gram + lam * (D2.T @ D2), rhs)
    Bg = BSpline.design_matrix(np.clip(lnx_grid, lo, hi - 1e-12), t, degree).toarray()
    return Bg @ c


class PSpline:
    supports_weights = True

    def __init__(self, n_knots=25, lam=1.0, degree=3):
        self.n_knots, self.lam, self.degree = n_knots, lam, degree

    def fit_predict(self, t, lnx, v, grid, weights=None):
        return spline_estimate(lnx, v, grid, n_knots=self.n_knots, lam=self.lam,
                               degree=self.degree, weights=weights)
```

`weights=None` keeps `B.T @ B` and `B.T @ v`, the previous arithmetic. Then delete `test_spline_rejects_mixture` from `tests/calibrate/test_importance.py` (the spline now accepts a mixture) and replace it with:

```python
def test_spline_accepts_mixture():
    d = design_mixture(DYN, T=0.5)
    cfg = ExplicitConfig(n_steps=4, n_particles=2_000, fit_subsample=500)
    r = calibrate_explicit(FlatDupire(), DYN, make_estimator("spline"), cfg, T=0.5, seed=3, mixture=d)
    assert np.all(np.isfinite(r.field[-1].L))
```

- [ ] **Step 4: Heads take `weights`; tilted `online_sweep`**

In `src/neural_particle_method/suite/heads.py`:

- Module docstring first line becomes: ``Every head implements `correction(t, lnx, v_plus, f_stale, model, grid, weights=None)` ...``.
- `RKHSHead.correction(self, t, lnx, v_plus, f_stale, model, grid, weights=None)`: first line `if weights is not None: raise ValueError("RKHS head does not support importance weights")`.
- `SplineHead.correction(..., weights=None)`: `corr = est.fit_predict(t, lnx, v_plus - f_stale, grid, weights=weights)`.
- `FeatureRidgeHead.correction(..., weights=None)`: first line `if weights is not None: raise ValueError("ridge head does not support importance weights")`.
- Add `from ..calibrate.importance import MixtureDesign  # noqa: F401` only if used for typing; otherwise no import is needed.
- `online_sweep(model, local_vol, params, s0, T, cfg=ExplicitConfig(), head=None, seed=0, mixture=None)`; docstring gains: "`mixture` (a `MixtureDesign`, constant or scheduled) tilts the online cloud exactly as `calibrate_explicit` does and hands the weights to the head; returns `(field, fit_s, is_diag)`, `is_diag` None when untilted." The stale branch returns `stale_field(...), 0.0, None`. The particle loop becomes:

```python
    n_steps, n_particles = cfg.n_steps, cfg.n_particles
    fit_subsample, L_max = cfg.fit_subsample, cfg.L_max
    rng = np.random.default_rng(seed)
    dt, sdt = T / n_steps, np.sqrt(T / n_steps)
    lnx = np.full(n_particles, np.log(s0))
    v = np.full(n_particles, hp.v0)
    theta_p, w, ess_min = None, None, 1.0
    if mixture is not None:
        comp = rng.choice(3, size=n_particles, p=list(mixture.alphas))
        thetas_all, etas_all = np.asarray(mixture.thetas), np.asarray(mixture.etas)
        alphas = np.array(mixture.alphas)
        ell = np.zeros((3, n_particles))
    slices, fit_s = [], 0.0
    for k in range(n_steps):
        t = k * dt
        if mixture is not None:
            th = thetas_all[:, k] if mixture.scheduled else thetas_all
            etas = etas_all[:, k] if mixture.scheduled else etas_all
            theta_p, eta_p = th[comp], etas[comp]
        if k == 0:
            grid, f_grid = np.array([np.log(s0)]), np.array([hp.v0])
        else:
            grid = np.unique(np.quantile(lnx, np.linspace(0.001, 0.999, 101)))
            idx = rng.choice(n_particles, size=min(fit_subsample, n_particles), replace=False)
            x_sub, v_sub = lnx[idx], np.maximum(v[idx], 0.0)
            w_sub = None if w is None else w[idx]
            t0 = time.perf_counter()
            f_stale = model.f(t, x_sub)
            corr = head.correction(t, x_sub, v_sub, f_stale, model, grid, weights=w_sub)
            f_grid = model.f(t, grid) + corr
            fit_s += time.perf_counter() - t0
            grid, f_grid = extend_tail(grid, f_grid, cfg.tail, _ModelTail(model, t))
        f_grid = np.clip(f_grid, 1e-4, None)
        sig = local_vol.sigma(max(t, local_vol.T_grid[0]), np.exp(grid), s0)
        L_grid = np.clip(sig / np.sqrt(f_grid), 0.0, L_max)
        slices.append(Slice(t, grid.copy(), L_grid.copy(), f_grid.copy()))
        L_p = np.interp(lnx, grid, L_grid) if len(grid) > 1 else np.full(n_particles, L_grid[0])
        zb, zp = rng.standard_normal(n_particles), rng.standard_normal(n_particles)
        lnx, v = heston_step(lnx, v, L_p, zb, zp, hp, dt, sdt, theta_p=theta_p)
        if mixture is not None:
            dbperp = zp * sdt + eta_p * dt
            ell += etas[:, None] * dbperp[None, :] - 0.5 * (etas ** 2)[:, None] * dt
            w = 1.0 / (alphas @ np.exp(np.clip(ell, -60, 60)))
            ess_min = min(ess_min, float(w.sum() ** 2 / (n_particles * (w ** 2).sum())))
    is_diag = None
    if mixture is not None:
        is_diag = {"max_w": float(w.max()),
                   "ess_frac": float(w.sum() ** 2 / (len(w) * (w ** 2).sum())),
                   "ess_min_slice": ess_min}
    return LeverageField(slices), fit_s, is_diag
```

The untilted path draws the same random numbers in the same order as before (`rng.choice` for the subsample, then `zb`, `zp`), so it is bit-for-bit; the `rng.choice(3, ...)` for components happens only under a mixture.

Then update every caller of `online_sweep` (grep `online_sweep(` in `src/`): `suite/budget.py` `field, _ = online_sweep(...)` -> `field, _, _ = online_sweep(...)`; `suite/online.py` likewise; any other (e.g. `suite/optuna_validate.py`) likewise.

- [ ] **Step 5: Run tests, goldens, ruff**

Run: `uv run pytest tests/estimators/test_spline.py tests/suite/test_heads.py tests/calibrate/test_importance.py tests/suite/test_budget.py tests/suite/test_online.py -q && uv run pytest -m golden tests/test_golden_tiny.py -q && uv run ruff check src tests && awk 'length > 100' src/neural_particle_method/estimators/spline.py src/neural_particle_method/suite/heads.py`
Expected: PASS, clean.

- [ ] **Step 6: Commit**

```bash
git add src/neural_particle_method/estimators/spline.py src/neural_particle_method/suite/heads.py src/neural_particle_method/suite/online.py src/neural_particle_method/suite/budget.py tests/estimators/test_spline.py tests/suite/test_heads.py tests/calibrate/test_importance.py
git add -u src/neural_particle_method/suite
git commit -m "heads: weighted P-spline, weights on every head, tilted online_sweep with weight diagnostics" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01EhbyGqFA5kYDyMwYouSgvS"
```

---

### Task 3: Cold cells and the tilt module skeleton

**Files:**
- Modify: `src/neural_particle_method/bench/algos.py` (`_nw`, `_nn_opt`, `run_algo`), `src/neural_particle_method/bench/runner.py` (`run_one`)
- Create: `src/neural_particle_method/suite/tilt.py`
- Test: `tests/suite/test_tilt.py`

**Interfaces:**
- Consumes: `TiltDesign`, `design_by_name`, `DESIGNS`, `UNTILTED` (Task 1); `run_one`, `cold_extra_key`, `SSVI_SIDS`.
- Produces: `run_algo(..., mixture=None)`, `run_one(..., mixture=None)`; in `suite/tilt.py`: `SLICE_SIDS`, `SLICE_PARTICLES = (10_000, 30_000, 80_000)`, `SLICE_SEEDS = (0, 1, 2, 3, 4)`, `COLD_PARTICLES = (10_000, 30_000, 80_000)`, `COLD_SEEDS = (0, 1)`, `ONLINE_BUDGETS = (10_000, 80_000)`, `ONLINE_SEEDS = (0, 1)`, `COLD_ALGO_SIDS = {"nw": SSVI_SIDS, "explicit_nn_opt": SLICE_SIDS}`, `mixture_for(design, n_steps, T, rho) -> MixtureDesign | None`, `run_tilt_cold(store, sid, algo, n_particles, seed, design_name, settings=FULL) -> run_id`, `cold_jobs(design_name, settings=FULL, sids=None) -> list[("cold", sid, algo, n, seed, design_name)]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/suite/test_tilt.py
import pytest

from neural_particle_method.estimators import recipes as R
from neural_particle_method.suite.config import SuiteSettings
from neural_particle_method.suite.tilt import (
    COLD_PARTICLES,
    SLICE_SIDS,
    cold_jobs,
    mixture_for,
    run_tilt_cold,
)
from neural_particle_method.tracking.store import Store

TINY = SuiteSettings.tiny()
FAST = {"hidden": 16, "depth": 2, "lr": 1e-2, "batch_size": 0, "first_steps": 5,
        "later_steps": 2, "weight_decay": 0.0, "fit_subsample": 400, "warm_start": True,
        "mean_match": True, "monotone": False, "monotone_penalty": 0.0, "tail": "linear",
        "hetero": False}


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


@pytest.fixture
def promoted(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RECIPE_DIR", tmp_path / "recipes")
    R.save_recipe("explicit_opt", FAST)


def test_cold_job_counts(promoted):
    jobs = cold_jobs("constant-3")
    assert len(jobs) == (20 + 6) * 3 * 2 * 2
    assert {j[5] for j in jobs} == {"none", "constant-3"}
    assert {j[3] for j in jobs} == set(COLD_PARTICLES)
    only_s02 = cold_jobs("constant-3", sids=("s02",))
    assert len(only_s02) == 2 * 3 * 2 * 2 and all(j[1] == "s02" for j in only_s02)
    assert SLICE_SIDS == ("s01", "s02", "s05", "s09", "s11", "s16")
    with pytest.raises(KeyError):
        cold_jobs("constant-2")


def test_mixture_for():
    assert mixture_for("none", 4, 1.0, -0.5) is None
    m = mixture_for("inverse_sqrt-1", 4, 1.0, -0.5)
    assert m.scheduled and m.thetas.shape == (3, 4)


def test_cold_cell_is_keyed_by_design_and_logs_diagnostics(store, promoted):
    a = run_tilt_cold(store, "s01", "nw", TINY.n_online, 0, "none", TINY)
    b = run_tilt_cold(store, "s01", "nw", TINY.n_online, 0, "constant-3", TINY)
    assert a != b
    assert run_tilt_cold(store, "s01", "nw", TINY.n_online, 0, "constant-3", TINY) == b
    pa, pb = store.get_params(a), store.get_params(b)
    assert pa["design"] == "none" and pb["design"] == "constant-3"
    ma, mb = store.get_metrics(a), store.get_metrics(b)
    assert "ess_min_slice" not in ma
    assert 0 < mb["ess_min_slice"] <= 1 and mb["max_w"] <= 2 + 1e-9
    assert mb["pooled_mae_bp"] >= 0 and "liquid_mae_bp" in mb
    assert len(store.search(TINY.experiment("tilt_cold"))) == 2


def test_cold_cell_on_the_searched_network_carries_the_recipe_hash(store, promoted):
    rid = run_tilt_cold(store, "s01", "explicit_nn_opt", TINY.n_online, 0, "constant-1", TINY)
    p = store.get_params(rid)
    assert p["recipe_hash"] == R.recipe_hash(FAST) and p["design"] == "constant-1"
    with pytest.raises(ValueError):
        run_tilt_cold(store, "s01", "spline", TINY.n_online, 0, "constant-1", TINY)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/suite/test_tilt.py -q`
Expected: FAIL with `ModuleNotFoundError: neural_particle_method.suite.tilt`.

- [ ] **Step 3: `mixture=` through `run_algo` and `run_one`**

In `src/neural_particle_method/bench/algos.py`:

```python
def _nw(sc, n, seed, e, i, knobs=None, mixture=None):
    return _explicit(sc, n, seed, e, "nw", mixture=mixture, knobs=knobs)
```

(keep the other one-liners as they are), and

```python
def _nn_opt(sc, n, seed, e, i, knobs=None, mixture=None):
    """Per-slice network with the promoted Optuna recipe (estimators/recipes/explicit_opt.json)."""
    recipe = load_recipe("explicit_opt")
    est = regressor_from_recipe(recipe, seed=seed,
                                monotone_sign=-float(np.sign(sc.dynamics.rho) or 1.0),
                                **(knobs or {}))
    return _explicit(sc, n, seed, explicit_config_from_recipe(e, recipe), "nn", estimator=est,
                     mixture=mixture)


TILT_ALGOS = ("nw", "explicit_nn_opt")


def run_algo(name, scenario, n_particles, seed, explicit=ExplicitConfig(),
             implicit=ImplicitConfig(), knobs=None, mixture=None):
    fn = {**ALGOS, **MECHANISM_ALGOS, **SUITE_ALGOS}[name]
    if mixture is None:
        return fn(scenario, n_particles, seed, explicit, implicit, knobs=knobs)
    if name not in TILT_ALGOS:
        raise ValueError(f"algorithm {name!r} does not take a tilt mixture; use one of {TILT_ALGOS}")
    return fn(scenario, n_particles, seed, explicit, implicit, knobs=knobs, mixture=mixture)
```

In `src/neural_particle_method/bench/runner.py`, `run_one(..., save_model=None, mixture=None)` and the call becomes `res = run_algo(algo, sc, n_particles, seed, explicit, implicit, knobs=knobs, mixture=mixture)`. After `metrics.update(res.timings)` add:

```python
        if "is_diag" in res.diagnostics:
            metrics.update({k: float(v) for k, v in res.diagnostics["is_diag"].items()})
```

(the diagnostics dict was already logged as JSON; the metrics copy is what tables read).

- [ ] **Step 4: Create `suite/tilt.py` with the cold cell and job list**

```python
"""Importance-sampling study: does the Girsanov tilt buy budget equivalence?

Layer 1 (`tilt_slices`): particles under the frozen PDE leverage, no feedback, per-slice
estimates at the quoted strikes against the PDE conditional expectation. Layer 2: cold
calibration (`tilt_cold`) and the frozen body + spline head (`tilt_online`) with and without the
tilt, scored like section 4. Spec: docs/superpowers/specs/2026-09-13-importance-sampling-design.md.
"""
from ..bench.runner import run_one
from ..bench.scenarios import full_registry
from ..calibrate.importance import UNTILTED, design_by_name
from .cold import NN_ALGOS, cold_extra_key
from .config import FULL, SSVI_SIDS

SLICE_SIDS = ("s01", "s02", "s05", "s09", "s11", "s16")
SLICE_PARTICLES = (10_000, 30_000, 80_000)
SLICE_SEEDS = (0, 1, 2, 3, 4)
COLD_PARTICLES = (10_000, 30_000, 80_000)
COLD_SEEDS = (0, 1)
COLD_ALGO_SIDS = {"nw": SSVI_SIDS, "explicit_nn_opt": SLICE_SIDS}
ONLINE_BUDGETS = (10_000, 80_000)
ONLINE_SEEDS = (0, 1)
ONLINE_METHOD = "explicit_opt_spline"


def mixture_for(design_name, n_steps, T, rho):
    """The MixtureDesign for a design name at this scenario's step count; None for "none"."""
    d = design_by_name(design_name)
    return None if d is None else d.mixture(n_steps, T, rho)


def run_tilt_cold(store, sid, algo, n_particles, seed, design_name, settings=FULL):
    """One cold calibration, tilted or not, scored as the cold suite is; keyed by `design`."""
    if algo not in COLD_ALGO_SIDS:
        raise ValueError(f"tilt cold rows are {tuple(COLD_ALGO_SIDS)}, got {algo!r}")
    sc = full_registry()[sid]
    mixture = mixture_for(design_name, settings.explicit.n_steps, sc.T, sc.dynamics.rho)
    knobs = {"keep_slice_weights": True} if algo in NN_ALGOS else None
    extra = {"design": design_name, **(cold_extra_key(algo) or {})}
    return run_one(store, sid, algo, int(n_particles), seed, settings.explicit, settings.implicit,
                   settings.reprice, experiment=settings.experiment("tilt_cold"), knobs=knobs,
                   extra_key=extra, mixture=mixture)


def _restrict(sids, wanted):
    return tuple(sids) if wanted is None else tuple(s for s in sids if s in set(wanted))


def cold_jobs(design_name, settings=FULL, sids=None, particles=COLD_PARTICLES, seeds=COLD_SEEDS):
    """Untilted and tilted cold cells, paired by seed, for both algorithms."""
    design_by_name(design_name)      # KeyError for an unknown design
    jobs = []
    for algo, algo_sids in COLD_ALGO_SIDS.items():
        for sid in _restrict(algo_sids, sids):
            for n in particles:
                for seed in seeds:
                    for d in (UNTILTED, design_name):
                        jobs.append(("cold", sid, algo, int(n), int(seed), d))
    return jobs
```

- [ ] **Step 5: Run tests, goldens, ruff**

Run: `uv run pytest tests/suite/test_tilt.py tests/bench -q && uv run pytest -m golden tests/test_golden_tiny.py -q && uv run ruff check src tests && awk 'length > 100' src/neural_particle_method/suite/tilt.py src/neural_particle_method/bench/algos.py src/neural_particle_method/bench/runner.py`
Expected: PASS, clean.

- [ ] **Step 6: Commit**

```bash
git add src/neural_particle_method/bench/algos.py src/neural_particle_method/bench/runner.py src/neural_particle_method/suite/tilt.py tests/suite/test_tilt.py
git commit -m "suite.tilt: cold cells with a tilt design in the key; mixture plumbing through run_one" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01EhbyGqFA5kYDyMwYouSgvS"
```

---

### Task 4: Online cells with a design

**Files:**
- Modify: `src/neural_particle_method/suite/budget.py` (`run_budget_cell`)
- Modify: `src/neural_particle_method/suite/tilt.py` (append `run_tilt_online`, `online_jobs`)
- Test: `tests/suite/test_tilt.py`

**Interfaces:**
- Consumes: `online_sweep(..., mixture=)` (Task 2), `mixture_for` (Task 3).
- Produces: `run_budget_cell(..., design=None)`: when `design` is a non-`None` name the experiment is `settings.experiment("tilt_online")`, the key gains `"design"`, and the sweep is tilted; `run_tilt_online(store, sid, budget, lag, seed, design_name, settings=FULL, recipe=None)`; `online_jobs(design_name, settings=FULL, sids=None) -> list[("online", sid, budget, lag_kind, seed, design_name)]` (tilted cells only: 160 at FULL).

- [ ] **Step 1: Write the failing tests** (append to `tests/suite/test_tilt.py`)

```python
from neural_particle_method.suite.lag import LAGS
from neural_particle_method.suite.tilt import online_jobs, run_tilt_online


def test_online_job_count_and_shape(promoted):
    jobs = online_jobs("constant-3")
    assert len(jobs) == 20 * 2 * 2 * 2 and all(j[5] == "constant-3" for j in jobs)
    assert {j[2] for j in jobs} == {10_000, 80_000}
    assert len(online_jobs("constant-3", sids=("s01", "s02"))) == 16


def test_online_cell_is_tilted_keyed_and_weighted(store, promoted):
    rid = run_tilt_online(store, "s01", TINY.n_online, LAGS[1], 0, "inverse_sqrt-3", TINY,
                          recipe=FAST)
    p, m = store.get_params(rid), store.get_metrics(rid)
    assert p["design"] == "inverse_sqrt-3" and p["method"] == "explicit_opt_spline"
    assert p["recipe_hash"] == R.recipe_hash(FAST)
    assert 0 < m["ess_min_slice"] <= 1 and m["online_s"] > 0
    assert run_tilt_online(store, "s01", TINY.n_online, LAGS[1], 0, "inverse_sqrt-3", TINY,
                           recipe=FAST) == rid
    assert len(store.search(TINY.experiment("tilt_online"))) == 1
    # the untilted partner lives in the budget experiment, not here
    assert store.search(TINY.experiment("suite_budget_tuned")).empty
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/suite/test_tilt.py -q -k online`
Expected: FAIL with `ImportError: cannot import name 'online_jobs'`.

- [ ] **Step 3: `design=` on `run_budget_cell`**

In `src/neural_particle_method/suite/budget.py`, add the import `from ..calibrate.importance import design_by_name` and change the signature to

```python
def run_budget_cell(store, sid, method, budget, lag, seed, body=DEFAULT_BODY, settings=FULL,
                    recipe=None, design=None):
```

After `exp = settings.experiment(EXPERIMENTS[body])` and the key construction, add:

```python
    tilt = None
    if design is not None:
        if method == "nw_resolve" or head_for(method) is None:
            raise ValueError("a tilt design applies to head cells only")
        exp = settings.experiment("tilt_online")
        key["design"] = design
        tilt = design_by_name(design)
```

Inside the run, replace the `online_sweep` call with:

```python
            sc_mix = None if tilt is None else tilt.mixture(ecfg.n_steps, lsc.T, lsc.dynamics.rho)
            field, _, diag = online_sweep(loaded.model, lv, lsc.dynamics, lsc.s0, lsc.T, ecfg,
                                          head=head, seed=seed + 1, mixture=sc_mix)
```

and after `metrics.update({"online_s": ...})` add `if diag is not None: metrics.update(diag)`. Set `diag = None` before the `if method == "nw_resolve":` branch so the name exists on both paths. (Untilted cells: `design=None`, identical behaviour to today.)

- [ ] **Step 4: Append to `suite/tilt.py`**

```python
from .budget import run_budget_cell          # add to the imports at the top
from .lag import LAGS                         # add to the imports at the top


def run_tilt_online(store, sid, budget, lag, seed, design_name, settings=FULL, recipe=None):
    """Frozen searched body + weighted spline head on a tilted online cloud."""
    design_by_name(design_name)
    if design_name == UNTILTED:
        raise ValueError("the untilted online cells are the suite_budget_tuned cells")
    return run_budget_cell(store, sid, ONLINE_METHOD, int(budget), lag, int(seed),
                           body="explicit_opt", settings=settings, recipe=recipe,
                           design=design_name)


def online_jobs(design_name, settings=FULL, sids=None, budgets=ONLINE_BUDGETS, seeds=ONLINE_SEEDS):
    design_by_name(design_name)
    return [("online", sid, int(b), lag.kind, int(seed), design_name)
            for sid in _restrict(SSVI_SIDS, sids) for lag in LAGS for seed in seeds for b in budgets]
```

- [ ] **Step 5: Run tests, ruff**

Run: `uv run pytest tests/suite/test_tilt.py tests/suite/test_budget.py -q && uv run ruff check src tests && awk 'length > 100' src/neural_particle_method/suite/tilt.py src/neural_particle_method/suite/budget.py`
Expected: PASS, clean.

- [ ] **Step 6: Commit**

```bash
git add src/neural_particle_method/suite/budget.py src/neural_particle_method/suite/tilt.py tests/suite/test_tilt.py
git commit -m "suite.tilt: tilted online cells (budget cell with a design key, tilt_online experiment)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01EhbyGqFA5kYDyMwYouSgvS"
```

---

### Task 5: Layer 1, the slice cell

**Files:**
- Modify: `src/neural_particle_method/suite/tilt.py` (append)
- Test: `tests/suite/test_tilt.py`

**Interfaces:**
- Consumes: `run_reference` (bench/reference_runs), `LeverageField`, `heston_step`, `NadarayaWatson`, `nw_estimate`, `regressor_from_recipe`, `load_recipe`, `quote_k_grid`, `snap_times`, `mixture_for`.
- Produces: `simulate_frozen(field, params, s0, T, n_steps, n_particles, seed, mixture=None, keep_times=()) -> dict{t: (lnx, v, w)}` plus `ess_path` (list of per-step ESS fractions, empty untilted); `slice_estimates(lnx, v, w, k_grid, recipe, seed) -> dict` with `"nw"`, `"net"`, `"ess_local"`, `"bandwidth"`; `run_slice_cell(store, sid, n_particles, design_name, seed, settings=FULL) -> run_id` logging artifact `slice_scores.json` (`{"maturities": [...], "k": [...], "f_ref": [[...]], "L_ref": [[...]], "f_hat": {"nw": [[...]], "net": [[...]]}, "L_hat": {...}, "f_rel_err": {...}, "lev_rel_err": {...}, "ess_local": [[...]], "ess_slice": [...]}`), artifact `cloud_T{t:g}.npz` (5 000-particle subsample: `lnx`, `v`, `w`) per maturity, metrics `wings_abs_f_rel/nw`, `wings_abs_f_rel/net`, `ess_final`, `ess_min_slice`, `max_w`, `sim_s`, `fit_s/nw`, `fit_s/net`; `slice_jobs(settings=FULL, sids=None) -> list[("slice", sid, n, design_name, seed)]` (630 at FULL: 6 sids x 3 x 7 x 5).

- [ ] **Step 1: Write the failing tests** (append to `tests/suite/test_tilt.py`)

```python
import json
import tempfile

import numpy as np

from neural_particle_method.bench.reference_runs import run_reference
from neural_particle_method.suite.tilt import run_slice_cell, simulate_frozen, slice_jobs


def test_slice_job_count():
    jobs = slice_jobs()
    assert len(jobs) == 6 * 3 * 7 * 5
    assert len({j[3] for j in jobs}) == 7 and "none" in {j[3] for j in jobs}
    assert len(slice_jobs(sids=("s11",))) == 3 * 7 * 5


def test_simulate_frozen_untilted_has_unit_weights_and_tilted_has_bounded_weights():
    from neural_particle_method.bench.scenarios import make_registry
    from neural_particle_method.simulate.leverage import DEFAULT_GRID, LeverageField, Slice
    sc = make_registry()["s01"]
    n_steps = 8
    field = LeverageField([Slice(k * sc.T / n_steps, DEFAULT_GRID.copy(),
                                 np.ones_like(DEFAULT_GRID), np.full_like(DEFAULT_GRID, 0.04))
                           for k in range(n_steps)])
    clouds, ess = simulate_frozen(field, sc.dynamics, sc.s0, sc.T, n_steps, 2_000, 0,
                                  keep_times=(0.5, 1.0))
    assert set(clouds) == {0.5, 1.0} and ess == []
    lnx, v, w = clouds[1.0]
    assert lnx.shape == v.shape == w.shape == (2_000,) and np.all(w == 1.0)
    mix = mixture_for("constant-3", n_steps, sc.T, sc.dynamics.rho)
    clouds_t, ess_t = simulate_frozen(field, sc.dynamics, sc.s0, sc.T, n_steps, 2_000, 0,
                                      mixture=mix, keep_times=(0.5, 1.0))
    _, _, wt = clouds_t[1.0]
    assert wt.max() <= 2 + 1e-9 and abs(wt.mean() - 1) < 0.1 and len(ess_t) == n_steps
    assert np.std(clouds_t[1.0][0]) > np.std(lnx)     # the tilt spreads the cloud


def test_slice_cell_logs_scores_and_clouds(store, promoted):
    run_reference(store, "s01", n_steps=TINY.explicit.n_steps, n_x=TINY.n_x, n_v=TINY.n_v)
    rid = run_slice_cell(store, "s01", 800, "constant-3", 0, TINY)
    assert run_slice_cell(store, "s01", 800, "constant-3", 0, TINY) == rid
    m = store.get_metrics(rid)
    for k in ("wings_abs_f_rel/nw", "wings_abs_f_rel/net", "ess_final", "ess_min_slice", "max_w",
              "sim_s", "fit_s/nw", "fit_s/net"):
        assert k in m
    with tempfile.TemporaryDirectory() as d:
        doc = json.loads(store.download(rid, "slice_scores.json", d).read_text())
        names = {a.path for a in store.client.list_artifacts(rid)}
    n_mat, n_k = len(doc["maturities"]), 13
    assert np.array(doc["f_ref"]).shape == (n_mat, n_k)
    assert np.array(doc["f_hat"]["nw"]).shape == (n_mat, n_k)
    assert np.array(doc["ess_local"]).shape == (n_mat, n_k) and len(doc["ess_slice"]) == n_mat
    assert all(f"cloud_T{t:g}.npz" in names for t in doc["maturities"])
    p = store.get_params(rid)
    assert p["design"] == "constant-3" and p["n_particles"] == "800"
    untilted = run_slice_cell(store, "s01", 800, "none", 0, TINY)
    assert store.get_metrics(untilted)["ess_final"] == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/suite/test_tilt.py -q -k "slice or frozen"`
Expected: FAIL with `ImportError: cannot import name 'run_slice_cell'`.

- [ ] **Step 3: Implement the frozen simulation, the slice estimates and the cell** (append to `suite/tilt.py`; add the imports at the top of the module)

```python
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from ..bench.reference_runs import run_reference
from ..bench.scenarios import quote_k_grid
from ..estimators.nadaraya_watson import NadarayaWatson, nw_estimate
from ..estimators.recipes import load_recipe, recipe_hash, regressor_from_recipe
from ..pricing.reprice import snap_times
from ..simulate.dynamics import HestonParams
from ..simulate.leverage import LeverageField
from ..simulate.stepper import heston_step
from ..tracking.store import git_hash
from ..calibrate.importance import DESIGNS

CLOUD_KEEP = 5_000
WING_CUT = 0.25


def simulate_frozen(field, params, s0, T, n_steps, n_particles, seed, mixture=None,
                    keep_times=()):
    """Particles under a frozen leverage field, tilted or not, with the calibrator's weights.

    Returns ({t: (lnx, v_plus, w)} at the requested (grid-snapped) times, [per-step ESS]).
    Clouds are taken at the start of the step whose time equals `t` (the same convention as the
    field's slices)."""
    hp = HestonParams.from_dict(params) if isinstance(params, dict) else params
    rng = np.random.default_rng(seed)
    dt, sdt = T / n_steps, np.sqrt(T / n_steps)
    lnx = np.full(n_particles, np.log(s0))
    v = np.full(n_particles, hp.v0)
    w = np.ones(n_particles)
    theta_p, ess_path = None, []
    if mixture is not None:
        comp = rng.choice(3, size=n_particles, p=list(mixture.alphas))
        thetas_all, etas_all = np.asarray(mixture.thetas), np.asarray(mixture.etas)
        alphas = np.array(mixture.alphas)
        ell = np.zeros((3, n_particles))
    wanted = {int(round(t / dt)): float(t) for t in keep_times}
    out = {}
    for k in range(n_steps + 1):
        if k in wanted:
            out[wanted[k]] = (lnx.copy(), np.maximum(v, 0.0), w.copy())
        if k == n_steps:
            break
        t = k * dt
        if mixture is not None:
            th = thetas_all[:, k] if mixture.scheduled else thetas_all
            etas = etas_all[:, k] if mixture.scheduled else etas_all
            theta_p, eta_p = th[comp], etas[comp]
        L_p = field.at(t, lnx)
        zb, zp = rng.standard_normal(n_particles), rng.standard_normal(n_particles)
        lnx, v = heston_step(lnx, v, L_p, zb, zp, hp, dt, sdt, theta_p=theta_p)
        if mixture is not None:
            dbperp = zp * sdt + eta_p * dt
            ell += etas[:, None] * dbperp[None, :] - 0.5 * (etas ** 2)[:, None] * dt
            w = 1.0 / (alphas @ np.exp(np.clip(ell, -60, 60)))
            ess_path.append(float(w.sum() ** 2 / (n_particles * (w ** 2).sum())))
    return out, ess_path


def _local_ess(lnx, w, k, bandwidth):
    ker = np.exp(-0.5 * ((k[:, None] - lnx[None, :]) / bandwidth) ** 2) * w[None, :]
    return (ker.sum(axis=1) ** 2) / np.clip((ker ** 2).sum(axis=1), 1e-300, None)


def slice_estimates(lnx, v, w, k, recipe, seed, monotone_sign):
    """NW and a fresh network fit on one weighted cloud, evaluated at the strikes `k`."""
    bw = 1.06 * np.std(lnx) * len(lnx) ** (-1 / 5)
    weights = None if np.all(w == 1.0) else w
    t0 = time.perf_counter()
    f_nw = nw_estimate(lnx, v, k, weights=weights, bandwidth=bw)
    s_nw = time.perf_counter() - t0
    net = regressor_from_recipe(recipe, seed=seed, monotone_sign=monotone_sign)
    t0 = time.perf_counter()
    f_net = net.fit_predict(0.0, lnx, v, k, weights=weights)
    s_net = time.perf_counter() - t0
    return {"nw": f_nw, "net": f_net, "ess_local": _local_ess(lnx, w, k, bw),
            "bandwidth": bw, "fit_s": {"nw": s_nw, "net": s_net}}


def _reference_field(store, sid, settings):
    rid = run_reference(store, sid, n_steps=settings.explicit.n_steps, n_x=settings.n_x,
                        n_v=settings.n_v)
    with tempfile.TemporaryDirectory() as d:
        raw = store.download(rid, "leverage.json", d).read_text()
    return rid, LeverageField.from_json(json.loads(raw))


def _slice_at(field, t):
    i = max(int(np.searchsorted(field.times, t + 1e-12)) - 1, 0)
    return field[i]


def run_slice_cell(store, sid, n_particles, design_name, seed, settings=FULL):
    n_steps = settings.explicit.n_steps
    key = {"sid": sid, "n_particles": int(n_particles), "design": design_name, "seed": int(seed),
           "n_steps": int(n_steps)}
    exp = settings.experiment("tilt_slices")
    existing = store.find_finished(exp, key)
    if existing is not None:
        return existing
    sc = full_registry()[sid]
    recipe = load_recipe("explicit_opt")
    ref_rid, ref = _reference_field(store, sid, settings)
    mixture = mixture_for(design_name, n_steps, sc.T, sc.dynamics.rho)
    # a maturity that snaps to t = 0 is a one-point cloud with no bandwidth: skip it, and score
    # each snapped time once (coarse grids can map two maturities onto one slice)
    mats = list(dict.fromkeys(t for t in snap_times(list(sc.maturities), n_steps, T=sc.T)
                              if t > 0))
    k = quote_k_grid() + np.log(sc.s0)
    wings = np.abs(quote_k_grid()) > WING_CUT
    params = {**key, "git_hash": git_hash(), "reference_run": ref_rid,
              "recipe_hash": recipe_hash(recipe), **sc.as_params()}
    with store.run(exp, params) as h:
        t0 = time.perf_counter()
        clouds, ess_path = simulate_frozen(ref, sc.dynamics, sc.s0, sc.T, n_steps,
                                           int(n_particles), seed, mixture=mixture,
                                           keep_times=mats)
        sim_s = time.perf_counter() - t0
        doc = {"maturities": mats, "k": k.tolist(), "f_ref": [], "L_ref": [],
               "f_hat": {"nw": [], "net": []}, "L_hat": {"nw": [], "net": []},
               "f_rel_err": {"nw": [], "net": []}, "lev_rel_err": {"nw": [], "net": []},
               "ess_local": [], "ess_slice": []}
        fit_s = {"nw": 0.0, "net": 0.0}
        rng = np.random.default_rng(seed + 77)
        for t in mats:
            lnx, v, w = clouds[t]
            s = _slice_at(ref, t)
            f_ref = np.interp(k, s.grid, s.f)
            L_ref = np.interp(k, s.grid, s.L)
            sig = sc.local_vol().sigma(max(t, sc.local_vol().T_grid[0]), np.exp(k), sc.s0)
            est = slice_estimates(lnx, v, w, k, recipe, seed,
                                  -float(np.sign(sc.dynamics.rho) or 1.0))
            doc["f_ref"].append(f_ref.tolist()); doc["L_ref"].append(L_ref.tolist())
            for name in ("nw", "net"):
                f_hat = np.clip(est[name], 1e-4, None)
                L_hat = np.clip(sig / np.sqrt(f_hat), 0.0, settings.explicit.L_max)
                doc["f_hat"][name].append(f_hat.tolist())
                doc["L_hat"][name].append(L_hat.tolist())
                doc["f_rel_err"][name].append(((f_hat - f_ref) / f_ref).tolist())
                doc["lev_rel_err"][name].append(((L_hat - L_ref) / L_ref).tolist())
                fit_s[name] += est["fit_s"][name]
            doc["ess_local"].append(est["ess_local"].tolist())
            doc["ess_slice"].append(float(w.sum() ** 2 / (len(w) * (w ** 2).sum())))
            keep = rng.choice(len(lnx), size=min(CLOUD_KEEP, len(lnx)), replace=False)
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / f"cloud_T{t:g}.npz"
                np.savez_compressed(p, lnx=lnx[keep].astype(np.float32),
                                    v=v[keep].astype(np.float32), w=w[keep].astype(np.float32))
                h.log_file(p)
        h.log_json("slice_scores.json", doc)
        w_last = clouds[mats[-1]][2]
        metrics = {"sim_s": sim_s, "fit_s/nw": fit_s["nw"], "fit_s/net": fit_s["net"],
                   "ess_final": float(w_last.sum() ** 2 / (len(w_last) * (w_last ** 2).sum())),
                   "ess_min_slice": float(min(ess_path)) if ess_path else 1.0,
                   "max_w": float(w_last.max())}
        for name in ("nw", "net"):
            e = np.abs(np.array(doc["f_rel_err"][name]))[:, wings]
            metrics[f"wings_abs_f_rel/{name}"] = float(e.mean())
        h.log_metrics(metrics)
        return h.run_id


def slice_jobs(settings=FULL, sids=None, particles=SLICE_PARTICLES, seeds=SLICE_SEEDS):
    names = [UNTILTED] + [d.name for d in DESIGNS]
    return [("slice", sid, int(n), d, int(seed)) for sid in _restrict(SLICE_SIDS, sids)
            for n in particles for d in names for seed in seeds]
```

Notes for the implementer: `field.at(t, lnx)` interpolates the frozen leverage at the particles (`LeverageField.at`); `quote_k_grid()` is log-moneyness so the strike abscissa is `k + log(s0)`; the local-vol call mirrors `calibrate_explicit`'s (`max(t, T_grid[0])`). The `tiny()` settings have 4 steps over T = 2, so `snap_times` maps the four maturities onto the grid (duplicates are fine: `keep_times` is a set-like dict).

- [ ] **Step 4: Run the tests, ruff**

Run: `uv run pytest tests/suite/test_tilt.py -q && uv run ruff check src tests && awk 'length > 100' src/neural_particle_method/suite/tilt.py`
Expected: PASS, clean. In `tiny()` (4 steps over T = 2) the maturities snap to (0, 0.5, 1, 2), so the doc lists three maturities (0.5, 1, 2).

- [ ] **Step 5: Commit**

```bash
git add src/neural_particle_method/suite/tilt.py tests/suite/test_tilt.py
git commit -m "suite.tilt: layer-1 slice cells under the frozen PDE leverage (tilt_slices)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01EhbyGqFA5kYDyMwYouSgvS"
```

---

### Task 6: Scoring, winner and the three tables

**Files:**
- Create: `src/neural_particle_method/suite/tilt_tables.py`
- Test: `tests/suite/test_tilt_tables.py`

**Interfaces:**
- Consumes: `store.search`, `_render_rows`, `_cell`, `_num`, `_finished`, `FLOOR_LABEL` from `suite/tables.py`; `DESIGNS`, `UNTILTED`; `SLICE_SIDS`, `SSVI_SIDS`.
- Produces: `slice_frame(store, settings=FULL) -> DataFrame` indexed by `(estimator, n_particles, design)` with columns `std_T0.25`, `std_pooled`, `bias_pooled` (wing values, percent of f); `winner(frame) -> str` (design name, or `"none"` when nothing beats untilted); `cold_frame(store, design, settings=FULL)`, `online_frame(store, design, settings=FULL)`; `tilt_tables(store, design, out_dir="paper/tables", settings=FULL) -> list[Path]` writing `tilt_slices.tex`, `tilt_cold.tex`, `tilt_online.tex`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/suite/test_tilt_tables.py
import json

import numpy as np
import pandas as pd
import pytest

from neural_particle_method.suite.tilt_tables import (
    cold_frame,
    online_frame,
    slice_frame,
    tilt_tables,
    winner,
)
from neural_particle_method.tracking.store import Store

K = np.log(np.geomspace(0.6, 1.6, 13)).tolist()
MATS = [0.25, 0.5, 1.0, 2.0]


def _doc(scale, bias, seed):
    # seeds 0, 1, 2 give bias - scale, bias, bias + scale at every strike: mean = bias and
    # std (ddof=1) = scale exactly, so the expected frame values are deterministic
    err = np.full((4, 13), bias + scale * (seed - 1)).tolist()
    return {"maturities": MATS, "k": K, "f_ref": np.ones((4, 13)).tolist(),
            "L_ref": np.ones((4, 13)).tolist(),
            "f_hat": {"nw": (1 + np.array(err)).tolist(), "net": (1 + np.array(err)).tolist()},
            "L_hat": {"nw": np.ones((4, 13)).tolist(), "net": np.ones((4, 13)).tolist()},
            "f_rel_err": {"nw": err, "net": err}, "lev_rel_err": {"nw": err, "net": err},
            "ess_local": np.ones((4, 13)).tolist(), "ess_slice": [1.0] * 4}


@pytest.fixture
def store(tmp_path):
    s = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    # layer 1: untilted noisy, constant-3 half the noise, inverse_sqrt-9 quarter noise but biased
    spec = {"none": (0.10, 0.0), "constant-3": (0.05, 0.0), "inverse_sqrt-9": (0.025, 0.4)}
    for sid in ("s01", "s02"):
        for n in (10_000, 30_000):
            for design, (scale, bias) in spec.items():
                for seed in range(3):
                    params = {"sid": sid, "n_particles": n, "design": design, "seed": seed,
                              "n_steps": 200}
                    with s.run("tilt_slices", params) as h:
                        h.log_json("slice_scores.json", _doc(scale, bias, seed))
                        h.log_metrics({"ess_final": 1.0})
    # layer 2 cold: nw untilted vs tilted at two budgets, one seed
    for n, (mae_u, mae_t) in {10_000: (57.0, 50.0), 80_000: (44.0, 43.0)}.items():
        for design, mae in (("none", mae_u), ("constant-3", mae_t)):
            params = {"sid": "s01", "algo": "nw", "n_particles": n, "seed": 0, "design": design}
            with s.run("tilt_cold", params) as h:
                m = {"pooled_mae_bp": mae, "wings_mae_bp": mae + 20, "mae_bp/T0.25": mae + 40,
                     "liquid_mae_bp": mae - 30, "lev_rmse": 0.1, "fit_s": 3.0}
                if design != "none":
                    m["ess_min_slice"] = 0.6
                h.log_metrics(m)
    with s.run("suite_pde_floor", {"sid": "s01", "seed": 0, "n_steps": 200}) as h:
        h.log_metrics({"pooled_mae_bp": 42.0, "wings_mae_bp": 62.0, "mae_bp/T0.25": 100.0,
                       "liquid_mae_bp": 13.0, "lev_rmse": 0.0, "fit_s": 0.0})
    # layer 2 online: untilted in suite_budget_tuned, tilted in tilt_online
    for budget, (mae_u, mae_t) in {10_000: (46.0, 44.0), 80_000: (42.4, 42.0)}.items():
        base = {"sid": "s01", "method": "explicit_opt_spline", "budget": budget,
                "lag": "surface", "seed": 0, "n_steps": 200, "recipe_hash": "abc"}
        with s.run("suite_budget_tuned", base) as h:
            h.log_metrics({"pooled_mae_bp": mae_u, "wings_mae_bp": mae_u + 20,
                           "liquid_mae_bp": 16.0, "online_s": 7.0})
        with s.run("tilt_online", {**base, "design": "constant-3"}) as h:
            h.log_metrics({"pooled_mae_bp": mae_t, "wings_mae_bp": mae_t + 20,
                           "liquid_mae_bp": 15.0, "online_s": 7.2, "ess_min_slice": 0.5})
    return s


def test_slice_frame_splits_std_and_bias(store):
    df = slice_frame(store)
    u = df.loc[("nw", 30_000, "none")]
    c = df.loc[("nw", 30_000, "constant-3")]
    b = df.loc[("nw", 30_000, "inverse_sqrt-9")]
    np.testing.assert_allclose([u["std_pooled"], c["std_pooled"], b["std_pooled"]],
                               [10.0, 5.0, 2.5], rtol=1e-9)              # percent of f
    np.testing.assert_allclose([u["bias_pooled"], b["bias_pooled"]], [0.0, 40.0], atol=1e-9)
    assert u["std_T0.25"] == u["std_pooled"]
    assert set(df.index.get_level_values(0)) == {"nw", "net"}


def test_winner_prefers_lowest_std_but_disqualifies_bias(store):
    df = slice_frame(store)
    assert winner(df) == "constant-3"
    # with the biased design's bias removed it would win on variance alone
    fixed = df.copy()
    fixed.loc[("nw", 30_000, "inverse_sqrt-9"), "bias_pooled"] = 0.0
    assert winner(fixed) == "inverse_sqrt-9"
    # nothing beats untilted -> "none"
    worse = df.copy()
    worse.loc[("nw", 30_000, "constant-3"), "std_pooled"] = 1e3
    worse.loc[("nw", 30_000, "inverse_sqrt-9"), "std_pooled"] = 1e3
    assert winner(worse) == "none"


def test_cold_and_online_frames_pair_tilted_with_untilted(store, monkeypatch):
    import neural_particle_method.suite.tilt_tables as T
    monkeypatch.setattr(T, "_promoted_hash", lambda: "abc")
    c = cold_frame(store, "constant-3")
    assert c.loc[("NW", 10_000, "tilted"), "mae"] == 50.0
    assert c.loc[("NW", 10_000, "untilted"), "mae"] == 57.0
    assert np.isnan(c.loc[("NW", 10_000, "untilted"), "ess_min"])
    assert c.loc[("NW", 10_000, "tilted"), "ess_min"] == 0.6
    assert c.loc[("PDE (attainable floor)", 0, ""), "mae"] == 42.0
    o = online_frame(store, "constant-3")
    assert o.loc[(10_000, "surface", "tilted"), "mae"] == 44.0
    assert o.loc[(10_000, "surface", "untilted"), "mae"] == 46.0
    assert np.isnan(o.loc[(80_000, "surface_spot", "tilted"), "mae"])


def test_tables_are_written_with_bold_best(store, tmp_path, monkeypatch):
    import neural_particle_method.suite.tilt_tables as T
    monkeypatch.setattr(T, "_promoted_hash", lambda: "abc")
    paths = tilt_tables(store, "constant-3", out_dir=tmp_path)
    assert [p.name for p in paths] == ["tilt_slices.tex", "tilt_cold.tex", "tilt_online.tex"]
    cold = (tmp_path / "tilt_cold.tex").read_text()
    # bold is the column minimum over every non-floor row: 43 at 80k tilted, not 50 at 10k
    assert "NW, 10k, tilted & 50 &" in cold
    assert "NW, 80k, tilted & \\textbf{43} &" in cold
    assert "PDE (attainable floor) & 42 &" in cold
    assert "& 0.60 &" in cold and "& -- &" in cold          # ess_min: value / untilted
    sl = (tmp_path / "tilt_slices.tex").read_text()
    assert "constant-3" in sl and "NW, 30k &" in sl and "\\textbf{2.5}" in sl
    on = (tmp_path / "tilt_online.tex").read_text()
    assert "10k, surface, tilted & 44 &" in on
    assert "80k, surface, tilted & \\textbf{42} &" in on
    assert "80k, surface, untilted & \\textbf{42} &" in on   # ties at displayed precision
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/suite/test_tilt_tables.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `suite/tilt_tables.py`**

```python
"""Tables of the importance-sampling study (paper/tables/tilt_*.tex) and the design winner."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..calibrate.importance import DESIGNS, UNTILTED
from ..estimators.recipes import load_recipe, recipe_exists, recipe_hash
from .config import FULL, SSVI_SIDS
from .tables import FLOOR_LABEL, _cell, _finished, _num, _render_rows
from .tilt import COLD_PARTICLES, ONLINE_BUDGETS, ONLINE_METHOD, SLICE_SIDS, WING_CUT

LAGS = ("surface", "surface_spot")
BIAS_TOLERANCE = 1.5     # a design may carry at most this multiple of the untilted wing bias...
BIAS_FLOOR = 1.0         # ...plus one percentage point, so a near-zero untilted bias is not a wall
WINNER_N = 30_000
ALGO_LABELS = {"nw": "NW", "explicit_nn_opt": "Explicit NN, searched"}
EST_LABELS = {"nw": "NW", "net": "Net"}


def _promoted_hash():
    return recipe_hash(load_recipe("explicit_opt")) if recipe_exists("explicit_opt") else None


def _label(n):
    return f"{int(n) // 1000}k"


def _slice_docs(store, settings):
    """One row per (sid, n, design, seed, estimator, maturity, strike) of wing relative error."""
    df = _finished(store, settings.experiment("tilt_slices"))
    rows = []
    if len(df) == 0:
        return pd.DataFrame(rows, columns=["sid", "n", "design", "seed", "est", "T", "k", "err"])
    df = df[df["params.sid"].isin(SLICE_SIDS)]
    with tempfile.TemporaryDirectory() as d:
        for _, r in df.iterrows():
            doc = json.loads(store.download(r["run_id"], "slice_scores.json",
                                            Path(d) / r["run_id"]).read_text())
            k = np.array(doc["k"])                 # log-strike; s0 = 1 on the SSVI scenarios
            wings = np.abs(k) > WING_CUT
            for est in ("nw", "net"):
                err = np.array(doc["f_rel_err"][est])
                for i, t in enumerate(doc["maturities"]):
                    for j in np.flatnonzero(wings):
                        rows.append((r["params.sid"], int(r["params.n_particles"]),
                                     r["params.design"], int(r["params.seed"]), est, float(t),
                                     float(k[j]), float(err[i, j])))
    return pd.DataFrame(rows, columns=["sid", "n", "design", "seed", "est", "T", "k", "err"])


def slice_frame(store, settings=FULL):
    """Wing std and bias (percent of f) per (estimator, n, design): std and mean over seeds at
    each (sid, T, k), then averaged over strikes, scenarios and (for pooled) maturities."""
    long = _slice_docs(store, settings)
    if len(long) == 0:
        return pd.DataFrame(columns=["std_T0.25", "std_pooled", "bias_pooled"])
    g = long.groupby(["est", "n", "design", "sid", "T", "k"])["err"]
    per = pd.DataFrame({"std": g.std(ddof=1), "bias": g.mean()}).reset_index()
    out = {}
    for (est, n, design), d in per.groupby(["est", "n", "design"]):
        first = d[np.isclose(d["T"], d["T"].min())]
        out[(est, int(n), design)] = pd.Series({
            "std_T0.25": 100 * first["std"].mean(),
            "std_pooled": 100 * d["std"].mean(),
            "bias_pooled": 100 * d["bias"].mean(),
        })
    df = pd.DataFrame(out).T
    df.index = pd.MultiIndex.from_tuples(df.index, names=["est", "n", "design"])
    return df.sort_index()


def winner(frame, n=WINNER_N, est="nw"):
    """The design with the smallest NW wing std at `n`, unless its wing bias exceeds
    BIAS_TOLERANCE x the untilted bias plus BIAS_FLOOR points; "none" when no admissible design
    beats untilted (a NaN std, e.g. a single seed, never beats anything)."""
    if (est, n, UNTILTED) not in frame.index:
        return UNTILTED
    base = frame.loc[(est, n, UNTILTED)]
    best, best_std = UNTILTED, base["std_pooled"]
    for d in DESIGNS:
        if (est, n, d.name) not in frame.index:
            continue
        row = frame.loc[(est, n, d.name)]
        if abs(row["bias_pooled"]) > BIAS_TOLERANCE * abs(base["bias_pooled"]) + BIAS_FLOOR:
            continue
        if row["std_pooled"] < best_std:
            best, best_std = d.name, row["std_pooled"]
    return best


def _cold_metrics(sel):
    if len(sel) == 0:
        return pd.Series({c: np.nan for c in ("mae", "wings", "t025", "liquid", "lev",
                                              "ess_min", "secs")})
    per = pd.DataFrame({
        "sid": sel["params.sid"], "mae": _num(sel, "metrics.pooled_mae_bp"),
        "wings": _num(sel, "metrics.wings_mae_bp"), "t025": _num(sel, "metrics.mae_bp/T0.25"),
        "liquid": _num(sel, "metrics.liquid_mae_bp"), "lev": _num(sel, "metrics.lev_rmse"),
        "ess_min": _num(sel, "metrics.ess_min_slice"), "secs": _num(sel, "metrics.fit_s"),
    }).groupby("sid").mean(numeric_only=True)
    return pd.Series({"mae": per.mae.mean(), "wings": per.wings.mean(), "t025": per.t025.mean(),
                      "liquid": per.liquid.mean(), "lev": per.lev.median(),
                      "ess_min": per.ess_min.mean(), "secs": per.secs.median()})


def cold_frame(store, design, settings=FULL):
    df = _finished(store, settings.experiment("tilt_cold"))
    rows = {}
    for algo, label in ALGO_LABELS.items():
        for n in COLD_PARTICLES:
            for arm, dname in (("untilted", UNTILTED), ("tilted", design)):
                sel = df.iloc[:0] if len(df) == 0 else df[
                    (df["params.algo"] == algo) & (_num(df, "params.n_particles") == n)
                    & (df["params.design"] == dname) & df["params.sid"].isin(SSVI_SIDS)]
                if algo == "explicit_nn_opt" and len(sel):
                    sel = (sel[sel["params.recipe_hash"] == _promoted_hash()]
                           if "params.recipe_hash" in sel.columns else sel.iloc[:0])
                rows[(label, int(n), arm)] = _cold_metrics(sel)
    floor = _finished(store, settings.experiment("suite_pde_floor"))
    floor = floor[floor["params.sid"].isin(SSVI_SIDS)] if len(floor) else floor
    rows[(FLOOR_LABEL, 0, "")] = _cold_metrics(floor)
    out = pd.DataFrame(rows).T
    out.index = pd.MultiIndex.from_tuples(out.index, names=["method", "n", "arm"])
    return out


def online_frame(store, design, settings=FULL):
    tilted = _finished(store, settings.experiment("tilt_online"))
    base = _finished(store, "suite_budget_tuned")
    h = _promoted_hash()
    rows = {}
    for b in ONLINE_BUDGETS:
        for lag in LAGS:
            for arm, df, dname in (("untilted", base, None), ("tilted", tilted, design)):
                if len(df) == 0:
                    sel = df
                else:
                    sel = df[(df["params.method"] == ONLINE_METHOD)
                             & (_num(df, "params.budget") == b) & (df["params.lag"] == lag)
                             & df["params.sid"].isin(SSVI_SIDS)]
                    if "params.recipe_hash" in sel.columns:
                        sel = sel[sel["params.recipe_hash"] == h]
                    if dname is not None:
                        sel = sel[sel["params.design"] == dname]
                if len(sel) == 0:
                    rows[(b, lag, arm)] = pd.Series({"mae": np.nan, "wings": np.nan,
                                                     "liquid": np.nan, "secs": np.nan})
                    continue
                per = pd.DataFrame({"sid": sel["params.sid"],
                                    "mae": _num(sel, "metrics.pooled_mae_bp"),
                                    "wings": _num(sel, "metrics.wings_mae_bp"),
                                    "liquid": _num(sel, "metrics.liquid_mae_bp"),
                                    "secs": _num(sel, "metrics.online_s")}).groupby("sid").mean()
                rows[(b, lag, arm)] = pd.Series({"mae": per.mae.mean(), "wings": per.wings.mean(),
                                                 "liquid": per.liquid.mean(),
                                                 "secs": per.secs.median()})
    out = pd.DataFrame(rows).T
    out.index = pd.MultiIndex.from_tuples(out.index, names=["budget", "lag", "arm"])
    return out


def _slices_tex(df, design):
    names = [UNTILTED] + [d.name for d in DESIGNS]
    cols = " ".join("rrr" for _ in names)
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{n}}}" for n in names)
    sub = " & ".join("std $T{=}0.25$ & std & bias" for _ in names)
    lines = [f"\\begin{{tabular}}{{l {cols}}}", "\\toprule", f"estimator, $N$ & {head}\\\\",
             "& " + sub + "\\\\", "\\midrule"]
    rows = []
    for est, elabel in EST_LABELS.items():
        for n in sorted({i[1] for i in df.index}) if len(df) else []:
            cells = []
            for name in names:
                r = df.loc[(est, n, name)] if (est, n, name) in df.index else None
                for c in ("std_T0.25", "std_pooled", "bias_pooled"):
                    cells.append((np.nan if r is None else float(r[c]), "{:.1f}"))
            rows.append((f"{elabel}, {_label(n)}", cells))
    lines += _render_rows(rows)
    lines += ["\\bottomrule", "\\end{tabular}", f"% winner: {design}"]
    return "\n".join(lines) + "\n"


def _cold_tex(df):
    """Errors and seconds go through `_render_rows` (bold = column minimum); the ESS column is
    "bigger is better" and is appended unbolded, `--` for the untilted rows and the floor."""
    lines = ["\\begin{tabular}{l rrrr r r r}", "\\toprule",
             "method, $N$, arm & pooled & wings & $T{=}0.25$ & liquid & lev.\\ RMSE & min ESS & s\\\\",
             "\\midrule"]
    rows, ess = [], []
    for (label, n, arm), r in df.iterrows():
        prefix = label if label == FLOOR_LABEL else f"{label}, {_label(n)}, {arm}"
        cells = [(r["mae"], "{:.0f}"), (r["wings"], "{:.0f}"), (r["t025"], "{:.0f}"),
                 (r["liquid"], "{:.0f}"), (r["lev"], "{:.3f}"), (r["secs"], "{:.1f}")]
        rows.append((prefix, cells))
        ess.append(_cell(r["ess_min"], "{:.2f}"))
    rendered = _render_rows(rows, secs_cols={5})
    for line, e in zip(rendered, ess):
        head, secs = line[: -len("\\\\")].rsplit(" & ", 1)
        lines.append(f"{head} & {e} & {secs}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _online_tex(df):
    lines = ["\\begin{tabular}{l rrr r}", "\\toprule",
             "budget, lag, arm & pooled & wings & liquid & online s\\\\", "\\midrule"]
    rows = [(f"{_label(b)}, {lag}, {arm}",
             [(r["mae"], "{:.0f}"), (r["wings"], "{:.0f}"), (r["liquid"], "{:.0f}"),
              (r["secs"], "{:.1f}")]) for (b, lag, arm), r in df.iterrows()]
    lines += _render_rows(rows, secs_cols={3})
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def tilt_tables(store, design, out_dir="paper/tables", settings=FULL):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "tilt_slices.tex": _slices_tex(slice_frame(store, settings), design),
        "tilt_cold.tex": _cold_tex(cold_frame(store, design, settings)),
        "tilt_online.tex": _online_tex(online_frame(store, design, settings)),
    }
    paths = []
    for name, text in files.items():
        (out / name).write_text(text)
        paths.append(out / name)
    return paths
```

Note on `_render_rows`: it bolds the minimum per column across all rows except the floor. In the cold table that means the best (method, N, arm) per column, which is the "budget equivalence" read: a tilted 30k row bold against untilted 80k says the tilt reached the larger budget. The ESS column is spliced in after rendering so it is never bolded.

- [ ] **Step 4: Run the tests, ruff**

Run: `uv run pytest tests/suite/test_tilt_tables.py tests/suite/test_tables.py -q && uv run ruff check src tests && awk 'length > 100' src/neural_particle_method/suite/tilt_tables.py`
Expected: PASS, clean.

- [ ] **Step 5: Commit**

```bash
git add src/neural_particle_method/suite/tilt_tables.py tests/suite/test_tilt_tables.py
git commit -m "suite.tilt_tables: layer-1 std/bias split, design winner, tilt_cold and tilt_online tables" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01EhbyGqFA5kYDyMwYouSgvS"
```

---

### Task 7: Figures

**Files:**
- Create: `src/neural_particle_method/suite/tilt_figures.py`
- Test: `tests/suite/test_tilt_figures.py`

**Interfaces:**
- Consumes: `tilt_slices` runs (`slice_scores.json`, `cloud_T{t}.npz`), `LeverageField` from the reference run, `quote_k_grid`, `WING_CUT`.
- Produces: `make_figures(store, design, out_dir="figures/out", settings=FULL, sids=("s02", "s11"), n_particles=30_000, seed=0, times=(0.25, 1.0)) -> list[Path]` writing `tilt_cloud_{sid}.png`, `tilt_leverage_{sid}.png`, `tilt_ess_{sid}.png` per sid.

- [ ] **Step 1: Write the failing test**

```python
# tests/suite/test_tilt_figures.py
import pytest

from neural_particle_method.bench.reference_runs import run_reference
from neural_particle_method.estimators import recipes as R
from neural_particle_method.suite.config import SuiteSettings
from neural_particle_method.suite.tilt import run_slice_cell
from neural_particle_method.suite.tilt_figures import make_figures
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
    s = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    run_reference(s, "s01", n_steps=TINY.explicit.n_steps, n_x=TINY.n_x, n_v=TINY.n_v)
    for design in ("none", "constant-3", "inverse_sqrt-1"):
        run_slice_cell(s, "s01", 800, design, 0, TINY)
    return s


def test_figures_are_written(store, tmp_path):
    paths = make_figures(store, "constant-3", out_dir=tmp_path, settings=TINY, sids=("s01",),
                         n_particles=800, seed=0, times=(0.5, 1.0))
    names = sorted(p.name for p in paths)
    assert names == ["tilt_cloud_s01.png", "tilt_ess_s01.png", "tilt_leverage_s01.png"]
    assert all(p.stat().st_size > 1_000 for p in paths)


def test_missing_cells_raise_a_clear_error(store, tmp_path):
    with pytest.raises(RuntimeError, match="tilt_slices"):
        make_figures(store, "constant-9", out_dir=tmp_path, settings=TINY, sids=("s01",),
                     n_particles=800, seed=0, times=(0.5, 1.0))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/suite/test_tilt_figures.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `suite/tilt_figures.py`**

```python
"""Figures of the importance-sampling study: clouds, leverage and weights, tilted beside untilted."""
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..bench.scenarios import quote_k_grid  # noqa: E402
from ..calibrate.importance import DESIGNS, UNTILTED  # noqa: E402
from ..pricing.reprice import snap_times  # noqa: E402
from .config import FULL  # noqa: E402
from .tilt import WING_CUT  # noqa: E402


def _cell(store, settings, sid, n, design, seed):
    key = {"sid": sid, "n_particles": int(n), "design": design, "seed": int(seed),
           "n_steps": int(settings.explicit.n_steps)}
    rid = store.find_finished(settings.experiment("tilt_slices"), key)
    if rid is None:
        raise RuntimeError(f"no finished tilt_slices cell for {key}")
    return rid


def _load(store, rid, times):
    with tempfile.TemporaryDirectory() as d:
        doc = json.loads(store.download(rid, "slice_scores.json", d).read_text())
        clouds = {}
        for t in times:
            z = np.load(store.download(rid, f"cloud_T{t:g}.npz", d))
            clouds[t] = (z["lnx"], z["v"], z["w"])
    return doc, clouds


def _row_index(doc, t):
    return int(np.argmin(np.abs(np.array(doc["maturities"]) - t)))


def _panels(times, arms):
    fig, axes = plt.subplots(len(times), len(arms), figsize=(5.2 * len(arms), 3.4 * len(times)),
                             squeeze=False, sharex="row")
    return fig, axes


def _cloud_figure(sid, docs, clouds, times, arms, out):
    k = quote_k_grid()
    fig, axes = _panels(times, arms)
    for i, t in enumerate(times):
        for j, arm in enumerate(arms):
            ax = axes[i, j]
            doc, (lnx, v, w) = docs[arm], clouds[arm][t]
            r = _row_index(doc, t)
            ax.scatter(lnx, v, s=6 * w, alpha=0.15, color="0.5", linewidths=0)
            kk = np.array(doc["k"])
            ax.plot(kk, doc["f_ref"][r], "k-", lw=1.5, label="PDE $E[V|X]$")
            ax.plot(kk, doc["f_hat"]["nw"][r], "C0--", lw=1.2, label="NW")
            ax.plot(kk, doc["f_hat"]["net"][r], "C3:", lw=1.4, label="net")
            for x in kk[np.abs(k) > WING_CUT]:
                ax.axvline(x, color="C1", lw=0.6, alpha=0.6)
            ax.set_ylim(0, np.percentile(v, 99.5) * 1.1)
            ax.set_title(f"{sid}, t = {t:g}, {arm}")
            ax.set_xlabel("log-spot"); ax.set_ylabel("V")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def _leverage_figure(sid, docs, clouds, times, arms, out):
    k = quote_k_grid()
    fig, axes = _panels(times, arms)
    for i, t in enumerate(times):
        for j, arm in enumerate(arms):
            ax = axes[i, j]
            doc, (lnx, _, w) = docs[arm], clouds[arm][t]
            r = _row_index(doc, t)
            kk = np.array(doc["k"])
            ax.plot(kk, doc["L_ref"][r], "k-", lw=1.5, label="PDE $L$")
            ax.plot(kk, doc["L_hat"]["nw"][r], "C0--", lw=1.2, label="NW")
            ax.plot(kk, doc["L_hat"]["net"][r], "C3:", lw=1.4, label="net")
            for x in kk[np.abs(k) > WING_CUT]:
                ax.axvline(x, color="C1", lw=0.6, alpha=0.6)
            ax2 = ax.twinx()
            ax2.hist(lnx, bins=60, weights=w, density=True, color="0.7", alpha=0.35)
            ax2.set_yticks([])
            ax.set_title(f"{sid}, t = {t:g}, {arm}")
            ax.set_xlabel("log-spot"); ax.set_ylabel("leverage")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def _ess_figure(sid, store, settings, n, seed, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    k = quote_k_grid()
    outer = [0, len(k) - 1]
    for name in [UNTILTED] + [d.name for d in DESIGNS]:
        key = {"sid": sid, "n_particles": int(n), "design": name, "seed": int(seed),
               "n_steps": int(settings.explicit.n_steps)}
        rid = store.find_finished(settings.experiment("tilt_slices"), key)
        if rid is None:
            continue
        with tempfile.TemporaryDirectory() as d:
            doc = json.loads(store.download(rid, "slice_scores.json", d).read_text())
        a.plot(doc["maturities"], doc["ess_slice"], marker="o", label=name)
        loc = np.array(doc["ess_local"])[:, outer].mean(axis=1)
        b.plot(doc["maturities"], loc, marker="o", label=name)
    a.set_xlabel("t"); a.set_ylabel("ESS fraction (slice)"); a.set_ylim(0, 1.05)
    b.set_xlabel("t"); b.set_ylabel("local ESS at the outermost strikes")
    a.legend(fontsize=7); a.set_title(sid)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def make_figures(store, design, out_dir="figures/out", settings=FULL, sids=("s02", "s11"),
                 n_particles=30_000, seed=0, times=(0.25, 1.0)):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arms = {"untilted": UNTILTED, "tilted": design}
    paths = []
    for sid in sids:
        snapped = snap_times(list(times), settings.explicit.n_steps, T=2.0)
        docs, clouds = {}, {}
        for arm, name in arms.items():
            rid = _cell(store, settings, sid, n_particles, name, seed)
            docs[arm], clouds[arm] = _load(store, rid, snapped)
        p = out / f"tilt_cloud_{sid}.png"
        _cloud_figure(sid, docs, clouds, snapped, list(arms), p); paths.append(p)
        p = out / f"tilt_leverage_{sid}.png"
        _leverage_figure(sid, docs, clouds, snapped, list(arms), p); paths.append(p)
        p = out / f"tilt_ess_{sid}.png"
        _ess_figure(sid, store, settings, n_particles, seed, p); paths.append(p)
    return paths
```

`T=2.0` in `snap_times` is the SSVI scenarios' horizon (`ScenarioSpec.T`); read it from `full_registry()[sid].T` instead of the literal (import `full_registry` from `..bench.scenarios`).

- [ ] **Step 4: Run the tests, ruff**

Run: `uv run pytest tests/suite/test_tilt_figures.py -q && uv run ruff check src tests && awk 'length > 100' src/neural_particle_method/suite/tilt_figures.py`
Expected: PASS, clean.

- [ ] **Step 5: Commit**

```bash
git add src/neural_particle_method/suite/tilt_figures.py tests/suite/test_tilt_figures.py
git commit -m "suite.tilt_figures: cloud, leverage and ESS figures, tilted beside untilted" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01EhbyGqFA5kYDyMwYouSgvS"
```

---

### Task 8: CLI, stage runner, smoke test, notes section

**Files:**
- Modify: `src/neural_particle_method/suite/tilt.py` (append `run_tilt_job`, `run_tilt_stage`)
- Modify: `src/neural_particle_method/cli.py`
- Modify: `paper/notes_experiments.tex`
- Test: `tests/test_cli_tilt.py`, `tests/suite/test_smoke.py`, `tests/suite/test_tilt.py`

**Interfaces:**
- Produces: `run_tilt_job(store, job, settings, recipe=None)`; `run_tilt_stage(store, stage, settings=FULL, n_jobs=1, sids=None, design=None, budgets=None, seeds=None) -> (done, failed)` for `stage` in `("slices", "cold", "online")`, skipping finished cells; CLI `nparticle tilt {slices,cold,online,winner,tables,figures}` with `--jobs`, `--sids`, `--design`, `--budgets`, `--seeds`, `--smoke`, `--out`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli_tilt.py
from neural_particle_method.cli import _parser


def test_tilt_subcommands_parse():
    a = _parser().parse_args(["tilt", "slices", "--jobs", "4", "--sids", "s01", "s02"])
    assert (a.cmd, a.tilt_cmd, a.jobs, a.sids, a.smoke) == ("tilt", "slices", 4, ["s01", "s02"], False)
    b = _parser().parse_args(["tilt", "cold", "--design", "constant-3", "--smoke"])
    assert (b.tilt_cmd, b.design, b.smoke, b.jobs) == ("cold", "constant-3", True, 1)
    c = _parser().parse_args(["tilt", "online", "--budgets", "10000", "--seeds", "0"])
    assert (c.tilt_cmd, c.budgets, c.seeds, c.design) == ("online", [10_000], [0], None)
    d = _parser().parse_args(["tilt", "tables", "--out", "x"])
    assert (d.tilt_cmd, d.out) == ("tables", "x")
    assert _parser().parse_args(["tilt", "winner"]).tilt_cmd == "winner"
    assert _parser().parse_args(["tilt", "figures", "--design", "none"]).design == "none"
```

Append to `tests/suite/test_tilt.py`:

```python
from neural_particle_method.suite.tilt import run_tilt_stage


def test_stage_runner_skips_finished_cells(store, promoted):
    run_reference(store, "s01", n_steps=TINY.explicit.n_steps, n_x=TINY.n_x, n_v=TINY.n_v)
    done, failed = run_tilt_stage(store, "slices", TINY, sids=("s01",), particles=(800,),
                                  seeds=(0,), designs=("none", "constant-3"))
    assert (done, failed) == (2, 0)
    done, failed = run_tilt_stage(store, "slices", TINY, sids=("s01",), particles=(800,),
                                  seeds=(0,), designs=("none", "constant-3"))
    assert (done, failed) == (0, 0)
    done, failed = run_tilt_stage(store, "cold", TINY, sids=("s01",), design="constant-3",
                                  particles=(TINY.n_online,), seeds=(0,))
    assert (done, failed) == (4, 0)          # nw + explicit_nn_opt, untilted + tilted
```

Extend `tests/suite/test_smoke.py` with a second slow test:

```python
@pytest.mark.slow
def test_smoke_tilt_end_to_end(tmp_path, monkeypatch):
    from neural_particle_method.estimators.recipes import save_recipe
    monkeypatch.setattr(R, "RECIPE_DIR", tmp_path / "recipes")
    save_recipe("explicit_opt", {"hidden": 16, "depth": 2, "lr": 1e-2, "batch_size": 0,
                                 "first_steps": 20, "later_steps": 5, "weight_decay": 0.0,
                                 "fit_subsample": 1_000, "warm_start": True, "mean_match": True,
                                 "monotone": False, "monotone_penalty": 0.0, "tail": "linear",
                                 "hetero": False})
    uri, root = f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art")
    base = ["--tracking-uri", uri, "--artifact-root", root, "tilt"]
    assert main(base + ["slices", "--smoke"]) == 0
    assert main(base + ["winner", "--smoke"]) == 0
    assert main(base + ["cold", "--smoke", "--design", "constant-3"]) == 0
    assert main(base + ["online", "--smoke", "--design", "constant-3"]) == 0
    assert main(base + ["tables", "--smoke", "--design", "constant-3",
                        "--out", str(tmp_path / "tables")]) == 0
    assert main(base + ["figures", "--smoke", "--design", "constant-3",
                        "--out", str(tmp_path / "figs")]) == 0
    store = Store(uri, root)
    assert len(store.search(SMOKE.experiment("tilt_slices"))) == 1 * 1 * 2 * 1
    assert len(store.search(SMOKE.experiment("tilt_cold"))) == 2 * 1 * 1 * 2
    assert len(store.search(SMOKE.experiment("tilt_online"))) == 1 * 2 * 1 * 1
    assert (tmp_path / "tables" / "tilt_cold.tex").exists()
    assert (tmp_path / "figs" / "tilt_cloud_s01.png").exists()
```

Smoke settings for the tilt stage (defined in `suite/tilt.py` as `SMOKE_GRID`): sids `("s01",)`, slice particles `(SMOKE.n_online,)`, slice seeds `(0,)`, designs `("none", "constant-3")`, cold particles `(SMOKE.n_online,)`, cold seeds `(0,)`, online budgets `(SMOKE.n_online,)`, online seeds `(0,)`, figures at `n_particles=SMOKE.n_online`, `times=(0.5, 1.0)`. The online smoke needs the searched body: `run_budget_cell` trains it through `body_run` (the SMOKE offline size 4 000), so no separate offline stage is required.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cli_tilt.py tests/suite/test_tilt.py -q -k "parse or stage"`
Expected: FAIL (`argparse` error: invalid choice 'tilt'; `ImportError: run_tilt_stage`).

- [ ] **Step 3: Stage runner in `suite/tilt.py`** (append; add `import os`, `from concurrent.futures import ProcessPoolExecutor`, `from .grid import _drain` at the top)

```python
SMOKE_GRID = {"sids": ("s01",), "seeds": (0,), "designs": (UNTILTED, "constant-3")}


def run_tilt_job(store, job, settings=FULL, recipe=None):
    kind = job[0]
    if kind == "slice":
        return run_slice_cell(store, job[1], job[2], job[3], job[4], settings)
    if kind == "cold":
        return run_tilt_cold(store, job[1], job[2], job[3], job[4], job[5], settings)
    if kind == "online":
        lag = {lg.kind: lg for lg in LAGS}[job[3]]
        return run_tilt_online(store, job[1], job[2], lag, job[4], job[5], settings, recipe=recipe)
    raise KeyError(kind)


def _is_finished(store, job, settings):
    kind, n_steps = job[0], int(settings.explicit.n_steps)
    if kind == "slice":
        key = {"sid": job[1], "n_particles": job[2], "design": job[3], "seed": job[4],
               "n_steps": n_steps}
        return store.find_finished(settings.experiment("tilt_slices"), key) is not None
    if kind == "cold":
        key = {"sid": job[1], "algo": job[2], "n_particles": job[3], "seed": job[4],
               "design": job[5], **(cold_extra_key(job[2]) or {})}
        return store.find_finished(settings.experiment("tilt_cold"), key) is not None
    if kind == "online":
        key = {"sid": job[1], "method": ONLINE_METHOD, "budget": job[2], "lag": job[3],
               "seed": job[4], "n_steps": n_steps, "design": job[5],
               "recipe_hash": recipe_hash(load_recipe("explicit_opt"))}
        return store.find_finished(settings.experiment("tilt_online"), key) is not None
    raise KeyError(kind)


def _worker(args):
    uri, root, job, settings, n_jobs = args
    if n_jobs > 1:
        import torch
        k = max(1, (os.cpu_count() or n_jobs) // n_jobs)
        os.environ.setdefault("OMP_NUM_THREADS", str(k))
        torch.set_num_threads(k)
    try:
        run_tilt_job(Store(uri, root), job, settings)
        return job, None
    except Exception as e:  # noqa: BLE001 - the run is already recorded FAILED by Store.run
        return job, repr(e)


def tilt_jobs(stage, settings=FULL, sids=None, design=None, particles=None, seeds=None,
              designs=None, budgets=None):
    if stage == "slices":
        kw = {k: v for k, v in (("particles", particles), ("seeds", seeds)) if v is not None}
        jobs = slice_jobs(settings, sids, **kw)
        if designs is not None:
            jobs = [j for j in jobs if j[3] in set(designs)]
        return jobs
    if design is None:
        raise ValueError(f"stage {stage!r} needs a design name (nparticle tilt winner)")
    if stage == "cold":
        kw = {k: v for k, v in (("particles", particles), ("seeds", seeds)) if v is not None}
        return cold_jobs(design, settings, sids, **kw)
    if stage == "online":
        kw = {k: v for k, v in (("budgets", budgets), ("seeds", seeds)) if v is not None}
        return online_jobs(design, settings, sids, **kw)
    raise KeyError(stage)


def run_tilt_stage(store, stage, settings=FULL, n_jobs=1, sids=None, design=None, particles=None,
                   seeds=None, designs=None, budgets=None):
    """Run every unfinished cell of `stage`; returns (n_done, n_failed). Pre-creates the
    experiments (MLflow race) and, for online, the body experiment."""
    for name in ("tilt_slices", "tilt_cold", "tilt_online", "suite_offline", "suite_budget_tuned"):
        store.experiment_id(settings.experiment(name))
    store.experiment_id("pde_reference")
    jobs = tilt_jobs(stage, settings, sids, design, particles, seeds, designs, budgets)
    if stage == "slices":
        # the PDE reference of each scenario is computed once here, serially, so pooled workers
        # never race on the same (sid, n_steps) reference run
        for sid in sorted({j[1] for j in jobs}):
            run_reference(store, sid, n_steps=settings.explicit.n_steps, n_x=settings.n_x,
                          n_v=settings.n_v)
    jobs = [j for j in jobs if not _is_finished(store, j, settings)]
    print(f"tilt {stage}: {len(jobs)} jobs listed", flush=True)
    args = [(store.tracking_uri, store.artifact_root, j, settings, n_jobs) for j in jobs]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            return _drain(pool.map(_worker, args))
    return _drain(map(_worker, args))
```

Move the shared `Store` import (`from ..tracking.store import Store, git_hash`) to the top of the module. `run_slice_cell` also calls `run_reference` itself, so a cell run outside the stage runner still finds or computes its reference.

- [ ] **Step 4: CLI**

In `src/neural_particle_method/cli.py::_parser`, after the `optuna` parser add:

```python
    p = sub.add_parser("tilt")
    ts = p.add_subparsers(dest="tilt_cmd", required=True)
    for name in ("slices", "cold", "online"):
        q = ts.add_parser(name)
        q.add_argument("--jobs", type=int, default=1); q.add_argument("--sids", nargs="*", default=None)
        q.add_argument("--design", default=None); q.add_argument("--smoke", action="store_true")
        q.add_argument("--budgets", nargs="*", type=int, default=None)
        q.add_argument("--seeds", nargs="*", type=int, default=None)
    q = ts.add_parser("winner"); q.add_argument("--smoke", action="store_true")
    q = ts.add_parser("tables"); q.add_argument("--design", default=None)
    q.add_argument("--smoke", action="store_true"); q.add_argument("--out", default="paper/tables")
    q = ts.add_parser("figures"); q.add_argument("--design", default=None)
    q.add_argument("--smoke", action="store_true"); q.add_argument("--out", default="figures/out")
```

and in `main`, before the final `return 1`:

```python
    if args.cmd == "tilt":
        from .suite.tilt import SMOKE_GRID, run_tilt_stage
        from .suite.tilt_figures import make_figures
        from .suite.tilt_tables import slice_frame, tilt_tables, winner
        settings = SUITE_SMOKE if args.smoke else SUITE_FULL
        grid = SMOKE_GRID if args.smoke else {}

        def chosen():
            d = getattr(args, "design", None)
            return d if d is not None else winner(slice_frame(store, settings))

        if args.tilt_cmd == "winner":
            print(winner(slice_frame(store, settings)))
            return 0
        if args.tilt_cmd in ("slices", "cold", "online"):
            sids = args.sids if args.sids is not None else grid.get("sids")
            particles = (settings.n_online,) if args.smoke else None
            seeds = args.seeds if args.seeds is not None else grid.get("seeds")
            budgets = args.budgets if args.budgets is not None else (
                (settings.n_online,) if args.smoke else None)
            design = None if args.tilt_cmd == "slices" else chosen()
            done, failed = run_tilt_stage(store, args.tilt_cmd, settings, n_jobs=args.jobs,
                                          sids=sids, design=design, particles=particles,
                                          seeds=seeds, designs=grid.get("designs"),
                                          budgets=budgets)
            print(f"tilt {args.tilt_cmd}: {done} done, {failed} failed", flush=True)
            return 1 if failed else 0
        if args.tilt_cmd == "tables":
            for pth in tilt_tables(store, chosen(), args.out, settings):
                print("wrote", pth)
            return 0
        if args.tilt_cmd == "figures":
            kw = ({"sids": ("s01",), "n_particles": settings.n_online, "times": (0.5, 1.0)}
                  if args.smoke else {})
            for pth in make_figures(store, chosen(), args.out, settings, **kw):
                print("wrote", pth)
            return 0
```

- [ ] **Step 5: Notes section**

In `paper/notes_experiments.tex`, after the `\subsection{Reproducibility}` paragraph and before `\end{document}`, add:

```latex
\section{Importance sampling}\label{sec:tilt}

\emph{Question.} The volatility-preserving Girsanov tilt (the defensive three-component mixture of \code{calibrate/importance.py}: drifts $-\theta,0,+\theta$ on log-spot with weights $\tfrac14,\tfrac12,\tfrac14$, exact discrete-time weights bounded by $2$) was a null result in the pre-floor grid, where the error was propagated bias. The tilt removes estimation variance only, so its regime is small particle budgets and short-dated far wings. The claim tested here is \emph{budget equivalence}: a tilted calibration at a small budget matching an untilted one at a larger budget.

\emph{Designs.} A design is a schedule and a cost cap $c$ on $\sum_k \theta_k^2\,\Delta t/(1-\rho^2)$, the log-weight variance of a pure tilt. \emph{Constant}: $\theta=\sqrt{c(1-\rho^2)/T}$, which at $c=3$ is the mixture already in the code. \emph{Inverse square root}: $\theta_k = a/\sqrt{\max(t_k,\Delta t)}$ with $a$ fixed by the same total cost, concentrating the tilt at short times. Six designs, $c\in\{1,3,9\}$ per schedule, plus untilted.

\subsection{Slice layer}\label{sec:tilt-slices}
Particles are simulated under the frozen PDE-reference leverage of each of the six study scenarios (s01, s02, s05, s09, s11, s16), so no estimate feeds back; at the four quoted maturities NW and the searched network are fitted on the weighted cloud and evaluated at the 13 quoted strikes against the PDE conditional expectation. Budgets $10$k, $30$k, $80$k; five seeds. The error at each strike is split into its standard deviation over seeds (the variance the tilt can remove) and its mean (bias, common to both arms). The winning design is the one with the smallest NW wing standard deviation at $30$k, disqualified if it moves the bias by more than half.

\begin{table}[h]\centering\footnotesize
\caption{Slice layer: wing standard deviation over seeds (at $T=0.25$ and pooled over maturities) and wing bias, in percent of $\E[V\mid X]$, by estimator, budget and design (experiment \code{tilt\_slices}). Bold: smallest in each column.}\label{tab:tilt-slices}
\resizebox{\textwidth}{!}{\input{tables/tilt_slices.tex}}
\end{table}

\begin{figure}[h]\centering
\includegraphics[width=\textwidth]{../figures/out/tilt_cloud_s02.png}
\includegraphics[width=\textwidth]{../figures/out/tilt_cloud_s11.png}
\caption{Particle clouds at $30$k under the frozen PDE leverage, untilted (left) and under the winning design (right), at $t=0.25$ (top) and $t=1$ (bottom), with the PDE conditional expectation (solid), the NW (dashed) and network (dotted) estimates, and the quoted wing strikes; marker area is the importance weight. s02 (Feller satisfied) and s11 (Feller violated).}\label{fig:tilt-cloud}
\end{figure}

\begin{figure}[h]\centering
\includegraphics[width=\textwidth]{../figures/out/tilt_leverage_s02.png}
\includegraphics[width=\textwidth]{../figures/out/tilt_leverage_s11.png}
\caption{The same slices in leverage: PDE (solid) against the NW and network estimates, with the weighted marginal density of the cloud behind.}\label{fig:tilt-leverage}
\end{figure}

\subsection{End to end}\label{sec:tilt-e2e}
Cold: NW on the $20$ SSVI scenarios and the searched network on the six study scenarios, at $10$k, $30$k and $80$k particles, seeds $0$ and $1$, untilted and under the winning design, scored as \cref{sec:suite-cold} (experiment \code{tilt\_cold}). Online: the searched $500\,000$-particle body with the spline head on a tilted online cloud at $10$k and $80$k, both lags (experiment \code{tilt\_online}), against the untilted cells of \cref{tab:suite-heads}; the P-spline is fitted by weighted least squares. Repricing is untilted throughout.

\begin{table}[h]\centering\footnotesize
\caption{Cold calibration with and without the tilt by budget: pooled, wings, three-month and liquid MAE in vol bp, leverage RMSE against the PDE reference, the minimum per-slice effective sample fraction, and median seconds. Bold: smallest in each column over all rows but the floor, so a bold tilted row at a small budget reads as budget equivalence.}\label{tab:tilt-cold}
\input{tables/tilt_cold.tex}
\end{table}

\begin{table}[h]\centering\footnotesize
\caption{Searched body with the spline head on an untilted and a tilted online cloud, by online budget and lag: pooled, wings and liquid MAE in vol bp and online seconds.}\label{tab:tilt-online}
\input{tables/tilt_online.tex}
\end{table}

\emph{Results.} To be written from the runs.

\emph{Diagnosis.} Three checks decide a null result: whether the slice layer shows any seed variance to remove at the quoted strikes; whether the local effective sample size at those strikes actually rose under the tilt or the defensive component dominated; and whether a drop in leverage RMSE fails to appear in the implied-vol error, which means the $200$-step scoring floor is binding in the wings.
```

The preamble must load `graphicx` (`grep -n graphicx paper/notes_experiments.tex`; add `\usepackage{graphicx}` after the other `\usepackage` lines if absent). Then run, from `paper/`: `pdflatex -interaction=nonstopmode notes_experiments.tex` twice and check the log for `undefined` and `^!`. The `\input` files may not exist yet: run `uv run nparticle tilt tables --design constant-3` against the real store first (it writes `--` cells when experiments are empty; this is a read-only query, allowed), and create empty placeholder PNGs if the figures are missing (`\includegraphics` of a missing file is a hard error): copy `figures/out/pde_leverage.png` to the four `tilt_*.png` names as placeholders and note it in the report; they are overwritten by `nparticle tilt figures`.

- [ ] **Step 6: Run the tests, the smoke test, ruff; report the smoke wall time**

Run: `uv run pytest -q && uv run pytest -m slow tests/suite/test_smoke.py::test_smoke_tilt_end_to_end -q && uv run pytest -m golden tests/test_golden_tiny.py -q && uv run ruff check src tests && awk 'length > 100' src/neural_particle_method/suite/tilt.py src/neural_particle_method/cli.py`
Expected: all PASS; report the smoke test's wall time (under three minutes expected).

- [ ] **Step 7: Commit**

```bash
git add src/neural_particle_method/suite/tilt.py src/neural_particle_method/cli.py paper/notes_experiments.tex paper/tables/tilt_slices.tex paper/tables/tilt_cold.tex paper/tables/tilt_online.tex tests/test_cli_tilt.py tests/suite/test_smoke.py tests/suite/test_tilt.py
git commit -m "nparticle tilt: slices/cold/online/winner/tables/figures; notes section 5 skeleton" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01EhbyGqFA5kYDyMwYouSgvS"
```

---

## Run order (controller, after the code lands; never overlapping)

1. `uv run nparticle tilt slices --jobs 4 2>&1 | tee results/tilt_slices.log` (about 2 h).
2. `uv run nparticle tilt winner`; `uv run nparticle tilt figures`.
3. `uv run nparticle tilt cold --jobs 4 --sids <20 SSVI sids> 2>&1 | tee results/tilt_cold.log` (NW rows about 2 h, network rows about 16 h; the stage lists both; run overnight).
4. `uv run nparticle tilt online --jobs 4 2>&1 | tee results/tilt_online.log` (about 2 h).
5. `uv run nparticle tilt tables`; write the results paragraph; rebuild the PDF; commit; push.
