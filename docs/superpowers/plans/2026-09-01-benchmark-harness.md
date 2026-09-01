# Benchmark Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CLI-driven benchmark testing five LSV calibration algorithms on synthetic SSVI surfaces, with fresh-seed IV-repricing error as the metric, feeding the Risk paper's Figs 1-4.

**Architecture:** Library modules in `src/neural_particle_method/` (SSVI surfaces + analytic Dupire, implicit scheme, mixture importance sampling, repricing); a `bench/` package with scenario/algo registries, a resumable runner writing one JSON per (scenario, algo, N, seed), an aggregator to `summary.csv`, and figure scripts reading only aggregated outputs.

**Tech Stack:** Python via uv, numpy, scipy, torch, QuantLib, pandas, matplotlib, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-01-benchmark-harness-design.md`. Read it before starting.
- All commands run through uv: `uv run pytest`, `uv run bench ...`. Dependencies added with `uv add`.
- Tests must stay fast: particle counts <= 20_000 and n_steps <= 12 in tests; the whole suite under ~60 s.
- `results/runs/` is gitignored; `results/summary.csv` and `figures/out/` are committed when produced.
- Match existing code style: plain numpy, short one-line docstrings, no type-annotation ceremony (the existing modules have none).
- IV inversion uses the existing `bs.implied_vol` (Brent); QuantLib validates it in tests and provides the LocalVolSurface cross-check. This is the one amendment to the spec's "QuantLib inverts to IV" line, made because `bs.implied_vol` already exists and is tested; note it if the author asks.
- Existing interfaces you will consume (do not change signatures except where a task says so):
  - `calibrate_explicit(dupire, params, s0=1.0, T=1.0, n_steps=50, n_particles=200_000, method="nn", fit_subsample=30_000, seed=0, L_max=4.0, first_steps=400, later_steps=120, snapshot_times=()) -> (lnx_T, info)` where `info["L_records"]` is a list of `(t, grid_lnx, L_grid, f_grid)` per slice. `dupire` needs `.sigma(t, x, s0)` and `.T_grid` (array; only `T_grid[0]` is used).
  - `nw_estimate(lnx, v, lnx_grid, weights=None, bandwidth=None)`; `NNRegressor(seed).fit(lnx, v, steps, weights=None)` / `.predict(grid)`; `RidgeHead(seed).train_body(lnx, v, steps)` / `.fit_predict(lnx, v, grid)` / `.trained`.
  - `bs_call(s0, K, T, sigma)`, `implied_vol(price, s0, K, T)` (returns NaN on failure).
  - `DupireSurface` in dupire.py (not used by the harness directly; SSVI provides its own local vol).

---

### Task 1: Dependencies, packaging, CLI skeleton

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `bench/__init__.py`, `bench/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: console script `bench` -> `bench.cli:main`; `main(argv=None) -> int` with subcommands `list|run|sweep|aggregate|figures` (all but `list` raise `NotImplementedError` for now; `list` prints a placeholder line).

- [ ] **Step 1: Add dependencies**

Run: `uv add QuantLib pandas matplotlib`
Expected: pyproject gains the three deps; `uv run python -c "import QuantLib, pandas, matplotlib"` exits 0.

- [ ] **Step 2: Write the failing CLI test**

```python
# tests/test_cli.py
import subprocess, sys

def test_bench_list_runs():
    r = subprocess.run(["uv", "run", "bench", "list"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "scenarios" in r.stdout.lower()
```

- [ ] **Step 3: Run it, verify failure**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL (no `bench` script).

- [ ] **Step 4: Packaging + skeleton**

In `pyproject.toml` add (merge with existing sections; the project uses hatchling with src layout):

```toml
[project.scripts]
bench = "bench.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/neural_particle_method", "bench"]
```

```python
# bench/__init__.py
```

```python
# bench/cli.py
"""CLI for the calibration benchmark."""
import argparse


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("run", "sweep", "aggregate", "figures"):
        sub.add_parser(name)
    args = ap.parse_args(argv)
    if args.cmd == "list":
        print("scenarios: (none registered yet)")
        return 0
    raise NotImplementedError(args.cmd)
```

Append to `.gitignore`: `results/runs/`

- [ ] **Step 5: Re-sync and run test**

Run: `uv sync && uv run pytest tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .gitignore bench tests/test_cli.py
git commit -m "feat: bench CLI skeleton, QuantLib/pandas/matplotlib deps"
```

---

### Task 2: SSVI surfaces with analytic Dupire local vol

**Files:**
- Create: `src/neural_particle_method/ssvi.py`
- Test: `tests/test_ssvi.py`

**Interfaces:**
- Produces:
  - `SSVIParams(sigma0, eta, gamma, rho)` frozen dataclass; `theta_T = sigma0**2 * T`.
  - `no_arb_ok(p) -> bool` (power-law sufficient conditions: `sigma0 > 0`, `0 < gamma <= 0.5`, `abs(rho) < 1`, `eta * (1 + abs(rho)) <= 2`).
  - `total_variance(p, k, T) -> ndarray`, `implied_vol_ssvi(p, k, T) -> ndarray`.
  - `SSVILocalVol(p, s0=1.0, t_min=0.004, T_max=2.0)` with attribute `T_grid = np.array([t_min, T_max])` and method `sigma(t, x, s0=1.0) -> ndarray` (drop-in for the `dupire` argument of `calibrate_explicit`).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ssvi.py
import numpy as np
import QuantLib as ql
from neural_particle_method.ssvi import (SSVIParams, no_arb_ok, total_variance,
                                         implied_vol_ssvi, SSVILocalVol)

GOOD = SSVIParams(sigma0=0.2, eta=1.0, gamma=0.4, rho=-0.6)

def test_no_arb_rejects_bad_params():
    assert no_arb_ok(GOOD)
    assert not no_arb_ok(SSVIParams(0.2, eta=1.5, gamma=0.4, rho=-0.6))  # eta(1+|rho|)=2.4>2
    assert not no_arb_ok(SSVIParams(0.2, eta=1.0, gamma=0.7, rho=-0.6))  # gamma>1/2

def test_atm_total_variance():
    T = 1.3
    w = total_variance(GOOD, np.array([0.0]), T)
    assert np.isclose(w[0], GOOD.sigma0**2 * T)

def test_skew_sign():
    iv = implied_vol_ssvi(GOOD, np.array([-0.2, 0.0, 0.2]), 1.0)
    assert iv[0] > iv[1]  # negative rho: put wing above ATM

def test_local_vol_matches_quantlib():
    lv = SSVILocalVol(GOOD)
    today = ql.Date(1, 9, 2026)
    ql.Settings.instance().evaluationDate = today
    dc, cal = ql.Actual365Fixed(), ql.NullCalendar()
    ts = np.arange(0.1, 1.81, 0.02)
    ks = np.arange(-0.5, 0.501, 0.02)
    dates = [today + ql.Period(int(round(t * 365)), ql.Days) for t in ts]
    strikes = list(np.exp(ks))
    vols = ql.Matrix(len(strikes), len(dates))
    for j, t in enumerate(ts):
        col = implied_vol_ssvi(GOOD, ks, t)
        for i in range(len(strikes)):
            vols[i][j] = float(col[i])
    bvs = ql.BlackVarianceSurface(today, cal, dates, strikes, vols, dc)
    bvs.setInterpolation("bicubic")
    spot = ql.QuoteHandle(ql.SimpleQuote(1.0))
    flat = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, dc))
    qlv = ql.LocalVolSurface(ql.BlackVolTermStructureHandle(bvs), flat, flat, spot)
    for t in (0.5, 1.0, 1.5):
        for k in (-0.3, -0.1, 0.0, 0.1, 0.3):
            ours = float(lv.sigma(t, np.exp(k)))
            theirs = qlv.localVol(t, float(np.exp(k)))
            assert abs(ours - theirs) / theirs < 0.05, (t, k, ours, theirs)
```

- [ ] **Step 2: Run, verify failure**

Run: `uv run pytest tests/test_ssvi.py -v` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# src/neural_particle_method/ssvi.py
"""SSVI (power-law) implied variance surface with analytic Dupire local volatility."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SSVIParams:
    sigma0: float
    eta: float
    gamma: float
    rho: float


def no_arb_ok(p):
    """Gatheral-Jacquier sufficient conditions for the power-law parameterisation."""
    return (p.sigma0 > 0 and 0 < p.gamma <= 0.5 and abs(p.rho) < 1
            and p.eta * (1 + abs(p.rho)) <= 2.0)


def _w_and_derivs(p, k, T):
    """w, dw/dk, d2w/dk2, dw/dT at (k, T); all closed form."""
    k = np.asarray(k, dtype=float)
    th = p.sigma0 ** 2 * T
    phi = p.eta * th ** (-p.gamma)
    dphi = -p.gamma * p.eta * th ** (-p.gamma - 1)
    u = phi * k + p.rho
    R = np.sqrt(u ** 2 + 1 - p.rho ** 2)
    w = 0.5 * th * (1 + p.rho * phi * k + R)
    dwdk = 0.5 * th * phi * (p.rho + u / R)
    d2wdk2 = 0.5 * th * phi ** 2 * (1 - p.rho ** 2) / R ** 3
    dwdth = w / th + 0.5 * th * dphi * k * (p.rho + u / R)
    dwdT = dwdth * p.sigma0 ** 2
    return w, dwdk, d2wdk2, dwdT


def total_variance(p, k, T):
    return _w_and_derivs(p, k, T)[0]


def implied_vol_ssvi(p, k, T):
    return np.sqrt(total_variance(p, k, T) / T)


class SSVILocalVol:
    """Analytic Dupire local vol from an SSVI surface; duck-types DupireSurface for calibrate_explicit."""

    def __init__(self, p, s0=1.0, t_min=0.004, T_max=2.0):
        self.p, self.s0 = p, s0
        self.T_grid = np.array([t_min, T_max])

    def sigma(self, t, x, s0=1.0):
        t = float(np.clip(t, self.T_grid[0], self.T_grid[-1]))
        k = np.log(np.asarray(x, dtype=float) / s0)
        w, dwdk, d2wdk2, dwdT = _w_and_derivs(self.p, k, T=t)
        denom = (1.0 - k / w * dwdk
                 + 0.25 * (-0.25 - 1.0 / w + (k / w) ** 2) * dwdk ** 2
                 + 0.5 * d2wdk2)
        v_loc = np.clip(dwdT, 1e-8, None) / np.clip(denom, 0.05, None)
        return np.sqrt(np.clip(v_loc, 1e-8, 9.0))
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ssvi.py -v` — Expected: PASS. If the QuantLib comparison fails only at the outermost points, check the denominator clamp first; do not loosen the 5% tolerance without noting it in the commit message.

- [ ] **Step 5: Commit**

```bash
git add src/neural_particle_method/ssvi.py tests/test_ssvi.py
git commit -m "feat: SSVI surface with analytic Dupire local vol, QuantLib cross-check"
```

---

### Task 3: Repricing and metrics

**Files:**
- Create: `src/neural_particle_method/reprice.py`
- Test: `tests/test_reprice.py`

**Interfaces:**
- Consumes: `bs_call`, `implied_vol` from `.bs`; `info["L_records"]` format `(t, grid_lnx, L_grid, f_grid)`.
- Produces:
  - `L_lookup(L_records) -> L_fn` where `L_fn(t, lnx) -> ndarray` (piecewise-constant in t, linear interp in lnx).
  - `reprice_iv(L_records, dynamics, s0, maturities, k_grid, n_particles=500_000, n_steps=100, seed=10_000) -> ndarray` shape `(len(maturities), len(k_grid))`, NaN where inversion fails. `dynamics` is a dict with keys kappa, theta, xi, rho, v0.
  - `iv_metrics(iv_model, iv_target, k_grid, maturities, wing_cut=0.25) -> dict` with keys `pooled_rmse_bp, pooled_max_bp, wings_rmse_bp, wings_max_bp, n_failed, per_maturity` (list of `{"T", "rmse_bp", "max_bp"}`). Errors in vol bp: `(iv_model - iv_target) * 1e4`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_reprice.py
import numpy as np
from neural_particle_method.reprice import L_lookup, reprice_iv, iv_metrics

FLAT_DYN = {"kappa": 0.0, "theta": 0.04, "xi": 0.0, "rho": 0.0, "v0": 0.04}

def const_records(L0, T=1.0, n=4):
    g = np.linspace(-1.0, 1.0, 5)
    return [(i * T / n, g, np.full(5, L0), np.full(5, 0.04)) for i in range(n)]

def test_L_lookup_piecewise():
    L = L_lookup(const_records(1.5))
    assert np.allclose(L(0.3, np.array([0.0, 0.2])), 1.5)

def test_gbm_limit_recovers_flat_iv():
    # xi=0, kappa=0: V frozen at v0; L=1 gives GBM with vol sqrt(v0)=0.2
    ivs = reprice_iv(const_records(1.0), FLAT_DYN, s0=1.0, maturities=[0.5, 1.0],
                     k_grid=np.log(np.array([0.8, 1.0, 1.25])),
                     n_particles=20_000, n_steps=12, seed=7)
    assert np.nanmax(np.abs(ivs - 0.2)) < 0.01

def test_iv_metrics_shapes():
    k = np.log(np.array([0.7, 1.0, 1.5]))
    target = np.full((2, 3), 0.2)
    model = target + np.array([[0.001, 0.0, -0.002], [0.0, 0.0, np.nan]])
    m = iv_metrics(model, target, k, [0.5, 1.0])
    assert m["n_failed"] == 1
    assert m["pooled_max_bp"] == 20.0
    assert m["wings_rmse_bp"] > 0  # |k|>0.25 covers 0.7 and 1.5 columns
    assert len(m["per_maturity"]) == 2
```

- [ ] **Step 2: Run, verify failure** — `uv run pytest tests/test_reprice.py -v` fails (module missing).

- [ ] **Step 3: Implement**

```python
# src/neural_particle_method/reprice.py
"""Fresh-seed repricing of vanillas under a calibrated leverage, and IV error metrics."""
import numpy as np

from .bs import implied_vol


def L_lookup(L_records):
    """Piecewise-constant-in-t, interp-in-lnx leverage function from calibration records."""
    ts = np.array([r[0] for r in L_records])

    def L(t, lnx):
        i = max(int(np.searchsorted(ts, t + 1e-12)) - 1, 0)
        _, grid, Lg, _ = L_records[i]
        if len(grid) == 1:
            return np.full_like(lnx, Lg[0])
        return np.interp(lnx, grid, Lg)

    return L


def reprice_iv(L_records, dynamics, s0, maturities, k_grid,
               n_particles=500_000, n_steps=100, seed=10_000):
    """Simulate fresh paths under L, price calls at each maturity, invert to IV."""
    kappa, theta, xi, rho, v0 = (dynamics[k] for k in ("kappa", "theta", "xi", "rho", "v0"))
    L = L_lookup(L_records)
    rng = np.random.default_rng(seed)
    T = max(maturities)
    dt, sdt = T / n_steps, np.sqrt(T / n_steps)
    snap = {int(round(m / dt)): m for m in maturities}
    lnx = np.full(n_particles, np.log(s0))
    v = np.full(n_particles, v0)
    ivs = np.full((len(maturities), len(k_grid)), np.nan)
    mat_idx = {m: i for i, m in enumerate(maturities)}
    for step in range(n_steps):
        t = step * dt
        L_p = L(t, lnx)
        zb = rng.standard_normal(n_particles)
        zp = rng.standard_normal(n_particles)
        z1 = rho * zb + np.sqrt(1 - rho ** 2) * zp   # W increment
        vp = np.maximum(v, 0.0)
        lnx = lnx + (-0.5 * L_p ** 2 * vp) * dt + L_p * np.sqrt(vp) * sdt * z1
        v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp) * sdt * zb
        if step + 1 in snap:
            m = snap[step + 1]
            x = np.exp(lnx)
            for j, k in enumerate(k_grid):
                K = s0 * np.exp(k)
                price = float(np.mean(np.maximum(x - K, 0.0)))
                ivs[mat_idx[m], j] = implied_vol(price, s0, K, m)
    return ivs


def iv_metrics(iv_model, iv_target, k_grid, maturities, wing_cut=0.25):
    """RMSE/max IV errors in vol bp: pooled, wings-only, per maturity; NaNs counted and excluded."""
    err = (np.asarray(iv_model) - np.asarray(iv_target)) * 1e4
    ok = np.isfinite(err)
    wings = np.abs(np.asarray(k_grid)) > wing_cut

    def stats(mask):
        e = err[mask & ok]
        if e.size == 0:
            return float("nan"), float("nan")
        return float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e)))

    pooled = stats(np.ones_like(ok, dtype=bool))
    wing_stats = stats(np.tile(wings, (err.shape[0], 1)))
    per = []
    for i, T in enumerate(maturities):
        row = np.zeros_like(ok, dtype=bool)
        row[i] = True
        r, mx = stats(row)
        per.append({"T": float(T), "rmse_bp": r, "max_bp": mx})
    return {"pooled_rmse_bp": pooled[0], "pooled_max_bp": pooled[1],
            "wings_rmse_bp": wing_stats[0], "wings_max_bp": wing_stats[1],
            "n_failed": int((~ok).sum()), "per_maturity": per}
```

Note the Euler scheme draws `zb` (the B increment, driving V) and `zp` (the orthogonal B-perp) and builds the W increment as `rho*zb + sqrt(1-rho^2)*zp`. Task 5 relies on this decomposition, so keep it exactly.

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_reprice.py -v` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/neural_particle_method/reprice.py tests/test_reprice.py
git commit -m "feat: fresh-seed repricer and IV error metrics"
```

---

### Task 4: Scenario registry

**Files:**
- Create: `bench/scenarios.py`
- Test: `tests/test_scenarios.py`

**Interfaces:**
- Consumes: `SSVIParams`, `no_arb_ok` from `neural_particle_method.ssvi`.
- Produces:
  - `ScenarioSpec(sid, ssvi, dynamics, s0=1.0, T=2.0, maturities=(0.25, 0.5, 1.0, 2.0))` frozen dataclass; `dynamics` is the dict consumed by `reprice_iv` and `calibrate_explicit` (keys kappa, theta, xi, rho, v0).
  - `quote_k_grid() -> ndarray`: `np.log(np.geomspace(0.6, 1.6, 13))`.
  - `make_registry() -> dict[str, ScenarioSpec]` with keys `s01..s20`, deterministic.
  - `fig3_registry() -> dict[str, ScenarioSpec]` with keys like `f_xi0.3_rho-0.7`: scenario s01's surface crossed with `xi in (0.3, 0.6, 1.0)` x `rho in (-0.7, -0.3)`.
  - `full_registry() -> dict` merging both.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_scenarios.py
from neural_particle_method.ssvi import no_arb_ok
from bench.scenarios import make_registry, fig3_registry, full_registry, quote_k_grid

def test_registry_deterministic_and_arb_free():
    a, b = make_registry(), make_registry()
    assert list(a) == [f"s{i:02d}" for i in range(1, 21)]
    assert all(a[k].ssvi == b[k].ssvi for k in a)
    assert all(no_arb_ok(s.ssvi) for s in a.values())

def test_fig3_cross():
    f = fig3_registry()
    assert len(f) == 6
    assert len({(s.dynamics["xi"], s.dynamics["rho"]) for s in f.values()}) == 6

def test_grids():
    k = quote_k_grid()
    assert len(k) == 13 and k[0] < 0 < k[-1]
    assert len(full_registry()) == 26
```

- [ ] **Step 2: Run, verify failure** — module missing.

- [ ] **Step 3: Implement**

```python
# bench/scenarios.py
"""Scenario registry: SSVI surface draws paired with Heston dynamics."""
from dataclasses import dataclass, field

import numpy as np

from neural_particle_method.ssvi import SSVIParams, no_arb_ok

XIS, RHOS, KAPPAS = (0.3, 0.6, 1.0), (-0.7, -0.3), (1.0, 2.0, 3.0)


@dataclass(frozen=True)
class ScenarioSpec:
    sid: str
    ssvi: SSVIParams
    dynamics: dict
    s0: float = 1.0
    T: float = 2.0
    maturities: tuple = (0.25, 0.5, 1.0, 2.0)


def quote_k_grid():
    return np.log(np.geomspace(0.6, 1.6, 13))


def _draw_ssvi(rng):
    for _ in range(100):
        p = SSVIParams(sigma0=rng.uniform(0.12, 0.35), eta=rng.uniform(0.5, 2.0),
                       gamma=rng.uniform(0.3, 0.5), rho=rng.uniform(-0.85, -0.1))
        if no_arb_ok(p):
            return p
    raise RuntimeError("no-arb resampling exhausted")


def make_registry():
    reg = {}
    for i in range(1, 21):
        rng = np.random.default_rng(1000 + i)
        p = _draw_ssvi(rng)
        dyn = {"kappa": float(rng.choice(KAPPAS)), "theta": p.sigma0 ** 2,
               "xi": float(rng.choice(XIS)), "rho": float(rng.choice(RHOS)),
               "v0": p.sigma0 ** 2}
        reg[f"s{i:02d}"] = ScenarioSpec(f"s{i:02d}", p, dyn)
    return reg


def fig3_registry():
    base = make_registry()["s01"]
    reg = {}
    for xi in XIS:
        for rho in RHOS:
            sid = f"f_xi{xi}_rho{rho}"
            dyn = dict(base.dynamics, xi=xi, rho=rho)
            reg[sid] = ScenarioSpec(sid, base.ssvi, dyn)
    return reg


def full_registry():
    return {**make_registry(), **fig3_registry()}
```

- [ ] **Step 4: Run tests** — PASS expected.
- [ ] **Step 5: Commit** — `git add bench/scenarios.py tests/test_scenarios.py && git commit -m "feat: scenario registry with no-arb SSVI draws and fig3 cross"`

---

### Task 5: Defensive-mixture importance sampling

**Files:**
- Create: `src/neural_particle_method/importance.py`
- Modify: `src/neural_particle_method/explicit.py` (add `mixture=None` parameter; switch the Euler noise decomposition to `(zb, zp)`; per-slice fit timing)
- Modify: `src/neural_particle_method/condexp.py` (weights in `RidgeHead.fit_predict`)
- Test: `tests/test_importance.py`

**Interfaces:**
- Produces:
  - `MixtureDesign(alphas, thetas, rho)` frozen dataclass, `alphas=(a_minus, a0, a_plus)`, `thetas=(th_minus, 0.0, th_plus)`; property `etas = tuple(th / sqrt(1 - rho**2))`.
  - `design_mixture(dynamics, T, k_target=0.47, alpha0=0.5, ess_cost_cap=3.0) -> MixtureDesign`. Moment matching: `sig_bar = sqrt(0.5*(v0 + theta))`, `th = k_target / (sig_bar * T)`, capped so `th**2 * T / (1 - rho**2) <= ess_cost_cap`.
  - `calibrate_explicit(..., mixture=None)`: unchanged behaviour when `mixture is None`; with a `MixtureDesign`, particles are assigned to components (proportions `alphas`, deterministic given seed), the X-drift gains `L*sqrt(v)*theta_j*dt`, per-particle log-ratios `ell_k` accumulate `eta_k*dBperp - 0.5*eta_k**2*dt` with `dBperp = zp*sdt + eta_j*dt`, the balance-heuristic weight `w = 1/sum_k alpha_k*exp(ell_k)` is passed to every estimator fit, and `info` gains `"weights"` (final per-particle w) and `"is_diag"` (`{"max_w", "ess_frac"}`). `info` also gains `"fit_s"` (accumulated regression wall-clock) in ALL cases.
  - `RidgeHead.fit_predict(lnx, v, lnx_grid, weights=None)`: weighted normal equations `A.T @ (w[:,None]*A) + lam*I` and `A.T @ (w*v)`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_importance.py
import numpy as np
from neural_particle_method.importance import MixtureDesign, design_mixture
from neural_particle_method.explicit import calibrate_explicit

DYN = {"kappa": 2.0, "theta": 0.04, "xi": 0.5, "rho": -0.6, "v0": 0.04}

class FlatDupire:
    T_grid = np.array([0.004, 1.0])
    def sigma(self, t, x, s0=1.0):
        return np.full(np.shape(np.log(np.asarray(x))), 0.2)

def test_design_caps_cost():
    d = design_mixture(DYN, T=1.0)
    assert d.alphas[1] == 0.5 and d.thetas[1] == 0.0
    assert d.thetas[2] > 0 > d.thetas[0]
    cost = d.thetas[2] ** 2 * 1.0 / (1 - DYN["rho"] ** 2)
    assert cost <= 3.0 + 1e-9

def test_weights_bounded_and_normalised():
    d = design_mixture(DYN, T=0.5)
    _, info = calibrate_explicit(FlatDupire(), DYN, T=0.5, n_steps=8,
                                 n_particles=20_000, method="nw",
                                 fit_subsample=5_000, seed=3, mixture=d)
    w = info["weights"]
    assert w.max() <= 1 / d.alphas[1] + 1e-9
    assert abs(w.mean() - 1.0) < 0.05

def test_flat_recovery_with_mixture():
    d = design_mixture(DYN, T=0.5)
    _, info = calibrate_explicit(FlatDupire(), DYN, T=0.5, n_steps=8,
                                 n_particles=20_000, method="nw",
                                 fit_subsample=5_000, seed=3, mixture=d)
    t, grid, Lg, _ = info["L_records"][-1]
    mid = np.abs(grid) < 0.3
    assert np.abs(Lg[mid] - 1.0).max() < 0.15
    assert "fit_s" in info and info["is_diag"]["max_w"] <= 2.0 + 1e-9
```

- [ ] **Step 2: Run, verify failure** — import errors.

- [ ] **Step 3: Implement importance.py**

```python
# src/neural_particle_method/importance.py
"""Defensive-mixture importance sampling: offline design targeting the optimal marginal."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MixtureDesign:
    alphas: tuple
    thetas: tuple
    rho: float

    @property
    def etas(self):
        r = np.sqrt(1 - self.rho ** 2)
        return tuple(th / r for th in self.thetas)


def design_mixture(dynamics, T, k_target=0.47, alpha0=0.5, ess_cost_cap=3.0):
    """Moment-matched wing tilts with a defensive untilted component and a kernel-cost cap."""
    rho, v0, theta_bar = dynamics["rho"], dynamics["v0"], dynamics["theta"]
    sig_bar = np.sqrt(0.5 * (v0 + theta_bar))
    th = k_target / (sig_bar * T)
    th_cap = np.sqrt(ess_cost_cap * (1 - rho ** 2) / T)
    th = min(th, th_cap)
    a_wing = 0.5 * (1 - alpha0)
    return MixtureDesign(alphas=(a_wing, alpha0, a_wing), thetas=(-th, 0.0, th), rho=rho)
```

- [ ] **Step 4: Modify condexp.py RidgeHead**

Change the signature to `def fit_predict(self, lnx, v, lnx_grid, weights=None):` and inside replace the two normal-equation lines with:

```python
        wv = np.ones(len(lnx)) if weights is None else weights
        lhs = A.T @ (wv[:, None] * A) + lam * np.eye(A.shape[1])
        rhs = A.T @ (wv * v)
```

(keep the residual-centering `rhs + lam*self.w_prev` logic unchanged after this).

- [ ] **Step 5: Modify explicit.py**

New signature: append `mixture=None` after `snapshot_times=()`. Inside:

```python
    import time
    # after reg is constructed:
    fit_s = 0.0
    if mixture is not None:
        comp = rng.choice(3, size=n_particles, p=list(mixture.alphas))
        theta_p = np.array(mixture.thetas)[comp]
        etas = np.array(mixture.etas)
        eta_p = etas[comp]
        ell = np.zeros((3, n_particles))
    w = None
```

Replace the per-slice fit block (the `if k >= 1:` body) so every estimator receives `weights=w` and is timed:

```python
        if k == 0:
            f_grid = np.full(len(grid), v0)
        else:
            idx = rng.choice(n_particles, size=min(fit_subsample, n_particles), replace=False)
            wi = None if w is None else w[idx]
            t0 = time.perf_counter()
            if method == "nn":
                reg.fit(lnx[idx], v[idx], steps=first_steps if k == 1 else later_steps, weights=wi)
                f_grid = reg.predict(grid)
            elif method == "ridge":
                if not reg.trained:
                    reg.train_body(lnx[idx], v[idx], steps=first_steps)
                f_grid = reg.fit_predict(lnx[idx], v[idx], grid, weights=wi)
            elif method == "spline":
                f_grid = spline_estimate(lnx[idx], v[idx], grid)
            else:
                f_grid = nw_estimate(lnx[idx], v[idx], grid, weights=wi)
            fit_s += time.perf_counter() - t0
```

Replace the Euler step with the `(zb, zp)` decomposition plus tilt and weight recursion:

```python
        L_p = np.interp(lnx, grid, L_grid) if len(grid) > 1 else np.full(n_particles, L_grid[0])
        zb = rng.standard_normal(n_particles)
        zp = rng.standard_normal(n_particles)
        z1 = rho * zb + np.sqrt(1 - rho ** 2) * zp
        vp = np.maximum(v, 0.0)
        drift_x = -0.5 * L_p ** 2 * vp
        if mixture is not None:
            drift_x = drift_x + L_p * np.sqrt(vp) * theta_p
        lnx = lnx + drift_x * dt + L_p * np.sqrt(vp) * sdt * z1
        v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp) * sdt * zb
        if mixture is not None:
            dbperp = zp * sdt + eta_p * dt
            ell += etas[:, None] * dbperp[None, :] - 0.5 * (etas ** 2)[:, None] * dt
            w = 1.0 / (np.array(mixture.alphas) @ np.exp(np.clip(ell, -60, 60)))
```

At the return, extend info:

```python
    info = {"L_records": L_records, "snapshots": snapshots, "fit_s": fit_s}
    if mixture is not None:
        info["weights"] = w
        info["is_diag"] = {"max_w": float(w.max()),
                          "ess_frac": float(w.sum() ** 2 / (len(w) * (w ** 2).sum()))}
    return lnx, info
```

Note: the noise decomposition change means untilted runs consume the RNG differently than before, so historical experiment JSONs are not bit-reproducible. That is acceptable; they are archived.

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_importance.py tests/test_sanity.py -v`
Expected: all PASS (test_sanity's flat-Dupire test must still pass with the new decomposition).

- [ ] **Step 7: Commit**

```bash
git add src/neural_particle_method/importance.py src/neural_particle_method/explicit.py src/neural_particle_method/condexp.py tests/test_importance.py
git commit -m "feat: defensive-mixture IS design and tilted explicit scheme with balance weights"
```

---

### Task 6: Implicit damped scheme

**Files:**
- Create: `src/neural_particle_method/implicit.py`
- Test: `tests/test_implicit.py`

**Interfaces:**
- Consumes: `calibrate_explicit` (for default initialisation), `V_SCALE, Z_SCALE` from `.condexp`, `L_lookup` from `.reprice`.
- Produces: `calibrate_implicit(dupire, params, s0=1.0, T=1.0, n_steps=50, n_particles=50_000, alpha=0.5, n_iters=6, seed=0, L_max=4.0, fit_steps=300, pool_subsample=60_000, L0_records=None) -> (L_records, info)` where `L_records` has the same `(t, grid, L_grid, f_grid)` format on a FIXED grid `np.linspace(log(0.4), log(2.2), 81)` per slice, and `info = {"deltas": [sup|L_next - L| per iter], "fit_s": float}`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_implicit.py
import numpy as np
from neural_particle_method.implicit import calibrate_implicit

DYN = {"kappa": 2.0, "theta": 0.04, "xi": 0.4, "rho": -0.5, "v0": 0.04}

class FlatDupire:
    T_grid = np.array([0.004, 1.0])
    def sigma(self, t, x, s0=1.0):
        return np.full(np.shape(np.log(np.asarray(x))), 0.2)

def test_damped_update_is_contraction_on_toy():
    # scalar analogue: Phi(L) = 0.2/sqrt(f(L)) with f pinned at 0.04 -> fixed point 1
    L, alpha = 3.0, 0.5
    for _ in range(20):
        L = (1 - alpha) * L + alpha * 1.0
    assert abs(L - 1.0) < 1e-5

def test_implicit_flat_recovery_smoke():
    recs, info = calibrate_implicit(FlatDupire(), DYN, T=0.5, n_steps=6,
                                    n_particles=8_000, alpha=0.5, n_iters=3,
                                    seed=1, pool_subsample=10_000, fit_steps=150)
    assert len(recs) == 6
    t, grid, Lg, fg = recs[3]
    mid = np.abs(grid) < 0.3
    assert np.abs(Lg[mid] - 1.0).max() < 0.2
    assert len(info["deltas"]) == 3 and info["deltas"][-1] <= info["deltas"][0] + 0.05
```

- [ ] **Step 2: Run, verify failure** — module missing.

- [ ] **Step 3: Implement**

```python
# src/neural_particle_method/implicit.py
"""Implicit scheme: global network over (t, x) with damped leverage iteration."""
import time

import numpy as np
import torch
import torch.nn as nn

from .condexp import V_SCALE, Z_SCALE
from .reprice import L_lookup


class GlobalNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, tz):
        return nn.functional.softplus(self.head(self.body(tz))) * V_SCALE


def _simulate(L_records, params, s0, T, n_steps, n_particles, rng):
    """Forward Euler under a fixed leverage; returns pooled (t, lnx, v) slices."""
    kappa, theta, xi, rho, v0 = (params[k] for k in ("kappa", "theta", "xi", "rho", "v0"))
    L = L_lookup(L_records)
    dt, sdt = T / n_steps, np.sqrt(T / n_steps)
    lnx = np.full(n_particles, np.log(s0))
    v = np.full(n_particles, v0)
    out = []
    for k in range(n_steps):
        t = k * dt
        L_p = L(t, lnx)
        zb = rng.standard_normal(n_particles)
        zp = rng.standard_normal(n_particles)
        z1 = rho * zb + np.sqrt(1 - rho ** 2) * zp
        vp = np.maximum(v, 0.0)
        lnx = lnx + (-0.5 * L_p ** 2 * vp) * dt + L_p * np.sqrt(vp) * sdt * z1
        v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp) * sdt * zb
        out.append((t + dt, lnx.copy(), np.maximum(v, 0.0).copy()))
    return out


def calibrate_implicit(dupire, params, s0=1.0, T=1.0, n_steps=50, n_particles=50_000,
                       alpha=0.5, n_iters=6, seed=0, L_max=4.0, fit_steps=300,
                       pool_subsample=60_000, L0_records=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    grid = np.linspace(np.log(0.4), np.log(2.2), 81)
    dt = T / n_steps
    v0 = params["v0"]
    if L0_records is None:
        sig0 = dupire.sigma(dupire.T_grid[0], np.exp(grid), s0)
        L0 = np.clip(sig0 / np.sqrt(v0), 0.0, L_max)
        L_records = [(k * dt, grid.copy(), L0.copy(), np.full(len(grid), v0)) for k in range(n_steps)]
    else:
        L_records = [(t, grid.copy(),
                      np.interp(grid, g, Lg) if len(g) > 1 else np.full(len(grid), Lg[0]),
                      np.interp(grid, g, fg) if len(g) > 1 else np.full(len(grid), fg[0]))
                     for (t, g, Lg, fg) in L0_records]
    net = GlobalNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    deltas, fit_s = [], 0.0
    for _ in range(n_iters):
        slices = _simulate(L_records, params, s0, T, n_steps, n_particles, rng)
        ts = np.concatenate([np.full(len(x), t) for t, x, _ in slices])
        xs = np.concatenate([x for _, x, _ in slices])
        vs = np.concatenate([v for _, _, v in slices])
        idx = rng.choice(len(ts), size=min(pool_subsample, len(ts)), replace=False)
        tz = torch.tensor(np.stack([ts[idx] / max(T, 1e-9), xs[idx] / Z_SCALE], axis=1),
                          dtype=torch.float32)
        tv = torch.tensor(vs[idx][:, None], dtype=torch.float32)
        t0 = time.perf_counter()
        for _ in range(fit_steps):
            opt.zero_grad()
            loss = ((net(tz) - tv) ** 2).mean()
            loss.backward()
            opt.step()
        fit_s += time.perf_counter() - t0
        new_records, sup = [], 0.0
        with torch.no_grad():
            for (t, g, Lg, _) in L_records:
                tzg = torch.tensor(np.stack([np.full(len(g), (t + dt) / max(T, 1e-9)),
                                             g / Z_SCALE], axis=1), dtype=torch.float32)
                f = np.clip(net(tzg).numpy()[:, 0], 1e-4, None)
                sig = dupire.sigma(max(t, dupire.T_grid[0]), np.exp(g), s0)
                Phi = np.clip(sig / np.sqrt(f), 0.0, L_max)
                L_new = (1 - alpha) * Lg + alpha * Phi
                sup = max(sup, float(np.abs(L_new - Lg).max()))
                new_records.append((t, g, L_new, f))
        L_records = new_records
        deltas.append(sup)
    return L_records, {"deltas": deltas, "fit_s": fit_s}
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_implicit.py -v` — PASS expected (the smoke test tolerances are loose by design; if the 0.2 recovery bound fails, raise fit_steps in the TEST, not the tolerance).

- [ ] **Step 5: Commit** — `git add src/neural_particle_method/implicit.py tests/test_implicit.py && git commit -m "feat: implicit damped-iteration scheme with global network"`

---

### Task 7: Algorithm registry

**Files:**
- Create: `bench/algos.py`
- Test: `tests/test_algos.py`

**Interfaces:**
- Consumes: `calibrate_explicit` (with `mixture=`), `calibrate_implicit`, `design_mixture`, `SSVILocalVol`, `ScenarioSpec`.
- Produces:
  - `CalibResult` dataclass: `L_records: list`, `timings: dict` (`{"total_s": float, "fit_s": float}`), `diagnostics: dict`.
  - `ALGOS: dict[str, callable]` with keys `nw, explicit_nn, ridge, explicit_nn_is, implicit_nn`.
  - `run_algo(name, scenario, n_particles, seed, cfg=None) -> CalibResult`. `cfg` overrides: `n_steps` (default 50), `fit_subsample` (default 30_000), implicit `alpha` (0.5) and `n_iters` (6). The surface is `SSVILocalVol(scenario.ssvi, scenario.s0, T_max=scenario.T)`; calibration horizon `T=scenario.T`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_algos.py
import numpy as np
import pytest
from bench.algos import ALGOS, run_algo
from bench.scenarios import make_registry

TINY = {"n_steps": 6, "fit_subsample": 4_000, "n_iters": 2, "first_steps": 80, "later_steps": 30, "fit_steps": 80}

@pytest.mark.parametrize("name", list(ALGOS))
def test_each_algo_runs_and_returns_records(name):
    sc = make_registry()["s01"]
    res = run_algo(name, sc, n_particles=6_000, seed=0, cfg=TINY)
    assert len(res.L_records) == 6
    t, grid, Lg, fg = res.L_records[-1]
    assert np.all(np.isfinite(Lg)) and Lg.min() >= 0.0 and Lg.max() <= 4.0
    assert res.timings["total_s"] > 0
    if name == "explicit_nn_is":
        assert res.diagnostics["is_diag"]["max_w"] <= 2.0 + 1e-9
```

- [ ] **Step 2: Run, verify failure** — module missing.

- [ ] **Step 3: Implement**

```python
# bench/algos.py
"""Algorithm registry: one calibrate interface over the five contenders."""
import time
from dataclasses import dataclass

from neural_particle_method.explicit import calibrate_explicit
from neural_particle_method.implicit import calibrate_implicit
from neural_particle_method.importance import design_mixture
from neural_particle_method.ssvi import SSVILocalVol


@dataclass
class CalibResult:
    L_records: list
    timings: dict
    diagnostics: dict


def _cfg(cfg, key, default):
    return default if cfg is None or key not in cfg else cfg[key]


def _explicit(scenario, n_particles, seed, cfg, method, mixture=None):
    lv = SSVILocalVol(scenario.ssvi, scenario.s0, T_max=scenario.T)
    t0 = time.perf_counter()
    _, info = calibrate_explicit(
        lv, scenario.dynamics, s0=scenario.s0, T=scenario.T,
        n_steps=_cfg(cfg, "n_steps", 50), n_particles=n_particles, method=method,
        fit_subsample=_cfg(cfg, "fit_subsample", 30_000), seed=seed,
        first_steps=_cfg(cfg, "first_steps", 400), later_steps=_cfg(cfg, "later_steps", 120),
        mixture=mixture)
    total = time.perf_counter() - t0
    diag = {"is_diag": info["is_diag"]} if "is_diag" in info else {}
    return CalibResult(info["L_records"], {"total_s": total, "fit_s": info["fit_s"]}, diag)


def _nw(sc, n, seed, cfg): return _explicit(sc, n, seed, cfg, "nw")
def _nn(sc, n, seed, cfg): return _explicit(sc, n, seed, cfg, "nn")
def _ridge(sc, n, seed, cfg): return _explicit(sc, n, seed, cfg, "ridge")


def _nn_is(sc, n, seed, cfg):
    mix = design_mixture(sc.dynamics, sc.T)
    return _explicit(sc, n, seed, cfg, "nn", mixture=mix)


def _implicit(sc, n, seed, cfg):
    lv = SSVILocalVol(sc.ssvi, sc.s0, T_max=sc.T)
    warm = _explicit(sc, n, seed, cfg, "nn")
    t0 = time.perf_counter()
    recs, info = calibrate_implicit(
        lv, sc.dynamics, s0=sc.s0, T=sc.T, n_steps=_cfg(cfg, "n_steps", 50),
        n_particles=n, alpha=_cfg(cfg, "alpha", 0.5), n_iters=_cfg(cfg, "n_iters", 6),
        seed=seed, fit_steps=_cfg(cfg, "fit_steps", 300), L0_records=warm.L_records)
    total = time.perf_counter() - t0 + warm.timings["total_s"]
    return CalibResult(recs, {"total_s": total, "fit_s": info["fit_s"] + warm.timings["fit_s"]},
                       {"deltas": info["deltas"]})


ALGOS = {"nw": _nw, "explicit_nn": _nn, "ridge": _ridge,
         "explicit_nn_is": _nn_is, "implicit_nn": _implicit}


def run_algo(name, scenario, n_particles, seed, cfg=None):
    return ALGOS[name](scenario, n_particles, seed, cfg)
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_algos.py -v` — PASS (implicit is the slow one; TINY keeps it under ~15 s).
- [ ] **Step 5: Commit** — `git add bench/algos.py tests/test_algos.py && git commit -m "feat: algorithm registry over the five contenders"`

---

### Task 8: Runner with self-describing JSON

**Files:**
- Create: `bench/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `run_algo`, `full_registry`, `quote_k_grid`, `reprice_iv`, `iv_metrics`, `implied_vol_ssvi`.
- Produces: `run_one(sid, algo, n_particles, seed, results_dir="results/runs", cfg=None, reprice_particles=500_000, reprice_steps=100) -> pathlib.Path` writing `<results_dir>/<sid>/<algo>/n<n_particles>_s<seed>.json` with keys `schema=1, sid, algo, n_particles, seed, git_hash, status ("ok"|"failed"), scenario (ssvi params + dynamics), timings, diagnostics, metrics, iv_err_bp (2D list, NaN->None), error (failed only)`. Also `run_path(sid, algo, n_particles, seed, results_dir) -> Path`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_runner.py
import json
import numpy as np
from bench import runner
from bench.runner import run_one, run_path

TINY = {"n_steps": 6, "fit_subsample": 4_000, "first_steps": 80, "later_steps": 30}

def test_run_one_writes_valid_json(tmp_path):
    p = run_one("s01", "nw", 4_000, 0, results_dir=tmp_path, cfg=TINY,
                reprice_particles=8_000, reprice_steps=8)
    assert p == run_path("s01", "nw", 4_000, 0, tmp_path)
    d = json.loads(p.read_text())
    assert d["status"] == "ok" and d["schema"] == 1
    assert d["metrics"]["pooled_rmse_bp"] >= 0
    assert len(d["iv_err_bp"]) == 4 and len(d["iv_err_bp"][0]) == 13
    assert d["git_hash"]

def test_failure_writes_failed_json(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(runner, "run_algo", boom)
    p = run_one("s01", "nw", 4_000, 1, results_dir=tmp_path, cfg=TINY)
    d = json.loads(p.read_text())
    assert d["status"] == "failed" and "synthetic failure" in d["error"]
```

- [ ] **Step 2: Run, verify failure** — module missing.

- [ ] **Step 3: Implement**

```python
# bench/runner.py
"""Run one (scenario, algo, N, seed) and write a self-describing result JSON."""
import dataclasses
import json
import subprocess
import traceback
from pathlib import Path

import numpy as np

from neural_particle_method.reprice import reprice_iv, iv_metrics
from neural_particle_method.ssvi import implied_vol_ssvi

from .algos import run_algo
from .scenarios import full_registry, quote_k_grid


def run_path(sid, algo, n_particles, seed, results_dir="results/runs"):
    return Path(results_dir) / sid / algo / f"n{n_particles}_s{seed}.json"


def _git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if not np.isfinite(f) else f
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def run_one(sid, algo, n_particles, seed, results_dir="results/runs", cfg=None,
            reprice_particles=500_000, reprice_steps=100):
    sc = full_registry()[sid]
    out = run_path(sid, algo, n_particles, seed, results_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": 1, "sid": sid, "algo": algo, "n_particles": n_particles,
           "seed": seed, "git_hash": _git_hash(),
           "scenario": {"ssvi": dataclasses.asdict(sc.ssvi), "dynamics": sc.dynamics,
                        "s0": sc.s0, "T": sc.T, "maturities": list(sc.maturities)}}
    try:
        res = run_algo(algo, sc, n_particles, seed, cfg)
        k = quote_k_grid()
        mats = list(sc.maturities)
        iv_model = reprice_iv(res.L_records, sc.dynamics, sc.s0, mats, k,
                              n_particles=reprice_particles, n_steps=reprice_steps,
                              seed=seed + 10_000)
        iv_target = np.stack([implied_vol_ssvi(sc.ssvi, k, m) for m in mats])
        doc.update(status="ok", timings=_jsonable(res.timings),
                   diagnostics=_jsonable(res.diagnostics),
                   metrics=_jsonable(iv_metrics(iv_model, iv_target, k, mats)),
                   iv_err_bp=_jsonable(((iv_model - iv_target) * 1e4).tolist()))
    except Exception:
        doc.update(status="failed", error=traceback.format_exc())
    out.write_text(json.dumps(doc, indent=1))
    return out
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_runner.py -v` — PASS.
- [ ] **Step 5: Commit** — `git add bench/runner.py tests/test_runner.py && git commit -m "feat: resumable run writer with metrics and failure capture"`

---

### Task 9: Aggregation

**Files:**
- Create: `bench/aggregate.py`
- Test: `tests/test_aggregate.py`

**Interfaces:**
- Consumes: run JSON schema from Task 8.
- Produces: `aggregate(runs_dir="results/runs", out_csv="results/summary.csv", out_md="results/digest.md") -> pandas.DataFrame`; one row per run with columns `sid, algo, n_particles, seed, status, pooled_rmse_bp, pooled_max_bp, wings_rmse_bp, wings_max_bp, n_failed, total_s, fit_s, git_hash` (NaN for failed runs' metrics). Digest lists failure count and, per scenario, the algo with lowest mean pooled_rmse_bp.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_aggregate.py
import json
from bench.aggregate import aggregate

OK = {"schema": 1, "sid": "s01", "algo": "nw", "n_particles": 1000, "seed": 0,
      "git_hash": "abc", "status": "ok", "timings": {"total_s": 1.0, "fit_s": 0.2},
      "diagnostics": {}, "metrics": {"pooled_rmse_bp": 12.0, "pooled_max_bp": 30.0,
      "wings_rmse_bp": 20.0, "wings_max_bp": 30.0, "n_failed": 0, "per_maturity": []}}
BAD = {"schema": 1, "sid": "s01", "algo": "ridge", "n_particles": 1000, "seed": 0,
       "git_hash": "abc", "status": "failed", "error": "boom"}

def test_aggregate(tmp_path):
    d = tmp_path / "s01" / "nw"; d.mkdir(parents=True)
    (d / "n1000_s0.json").write_text(json.dumps(OK))
    d2 = tmp_path / "s01" / "ridge"; d2.mkdir(parents=True)
    (d2 / "n1000_s0.json").write_text(json.dumps(BAD))
    df = aggregate(tmp_path, tmp_path / "summary.csv", tmp_path / "digest.md")
    assert len(df) == 2 and set(df.status) == {"ok", "failed"}
    assert (tmp_path / "summary.csv").exists()
    md = (tmp_path / "digest.md").read_text()
    assert "1 failed" in md and "s01" in md
```

- [ ] **Step 2: Run, verify failure.**

- [ ] **Step 3: Implement**

```python
# bench/aggregate.py
"""Flatten run JSONs into a tidy summary table and a short digest."""
import json
from pathlib import Path

import pandas as pd

METRIC_COLS = ("pooled_rmse_bp", "pooled_max_bp", "wings_rmse_bp", "wings_max_bp", "n_failed")


def aggregate(runs_dir="results/runs", out_csv="results/summary.csv",
              out_md="results/digest.md"):
    rows = []
    for p in sorted(Path(runs_dir).rglob("*.json")):
        d = json.loads(p.read_text())
        row = {k: d.get(k) for k in ("sid", "algo", "n_particles", "seed", "status", "git_hash")}
        m, t = d.get("metrics") or {}, d.get("timings") or {}
        for c in METRIC_COLS:
            row[c] = m.get(c)
        row["total_s"], row["fit_s"] = t.get("total_s"), t.get("fit_s")
        rows.append(row)
    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    n_fail = int((df.status == "failed").sum()) if len(df) else 0
    lines = [f"# Benchmark digest", f"{len(df)} runs, {n_fail} failed", ""]
    ok = df[df.status == "ok"]
    if len(ok):
        best = ok.groupby(["sid", "algo"]).pooled_rmse_bp.mean().reset_index() \
                 .sort_values("pooled_rmse_bp").groupby("sid").first()
        for sid, r in best.iterrows():
            lines.append(f"- {sid}: best = {r.algo} ({r.pooled_rmse_bp:.1f} bp)")
    Path(out_md).write_text("\n".join(lines) + "\n")
    return df
```

- [ ] **Step 4: Run tests** — PASS.
- [ ] **Step 5: Commit** — `git add bench/aggregate.py tests/test_aggregate.py && git commit -m "feat: run aggregation to summary.csv and digest"`

---

### Task 10: Full CLI (run, sweep, aggregate, figures)

**Files:**
- Modify: `bench/cli.py`
- Test: `tests/test_cli.py` (extend)

**Interfaces:**
- Consumes: `full_registry`, `ALGOS`, `run_one`, `run_path`, `aggregate`.
- Produces the final CLI:
  - `bench list` — prints scenario ids with sigma0/xi/rho and algo names.
  - `bench run --scenario SID --algo NAME --n N --seed S [--n-steps K] [--reprice-n N] [--reprice-steps K] [--results-dir D]`
  - `bench sweep --preset paper [--jobs J] [--results-dir D]` — grid = make_registry() x 5 algos x N in (50_000, 200_000) x seeds (0, 1, 2), PLUS fig3_registry() x (nw, explicit_nn) x N=200_000 x seeds (0, 1, 2); skips existing files; `--jobs` uses `concurrent.futures.ProcessPoolExecutor`.
  - `bench aggregate [--runs-dir D] [--out CSV]`
  - `bench figures [--summary CSV] [--runs-dir D] [--outdir D]` — calls `make(...)` in each of the four figure modules (Task 11); before Task 11 lands it prints "no figure modules" and returns 0 if the import fails.

- [ ] **Step 1: Extend the CLI test**

Append to `tests/test_cli.py`:

```python
import json

def test_bench_run_and_sweep_skip(tmp_path):
    args = ["uv", "run", "bench", "run", "--scenario", "s01", "--algo", "nw",
            "--n", "3000", "--seed", "0", "--n-steps", "6",
            "--reprice-n", "5000", "--reprice-steps", "8",
            "--results-dir", str(tmp_path)]
    r = subprocess.run(args, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = tmp_path / "s01" / "nw" / "n3000_s0.json"
    assert json.loads(out.read_text())["status"] == "ok"
    r2 = subprocess.run(args, capture_output=True, text=True)
    assert "skip" in (r2.stdout + r2.stderr).lower()
```

- [ ] **Step 2: Run, verify failure** — current stub raises NotImplementedError.

- [ ] **Step 3: Implement**

```python
# bench/cli.py
"""CLI for the calibration benchmark."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .aggregate import aggregate
from .algos import ALGOS
from .runner import run_one, run_path
from .scenarios import fig3_registry, full_registry, make_registry


def _paper_grid():
    jobs = []
    for sid in make_registry():
        for algo in ALGOS:
            for n in (50_000, 200_000):
                for seed in (0, 1, 2):
                    jobs.append((sid, algo, n, seed))
    for sid in fig3_registry():
        for algo in ("nw", "explicit_nn"):
            for seed in (0, 1, 2):
                jobs.append((sid, algo, 200_000, seed))
    return jobs


def _do_run(job):
    sid, algo, n, seed, results_dir = job
    run_one(sid, algo, n, seed, results_dir=results_dir)
    return sid, algo, n, seed


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("run")
    p.add_argument("--scenario", required=True)
    p.add_argument("--algo", required=True, choices=list(ALGOS))
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--reprice-n", type=int, default=500_000)
    p.add_argument("--reprice-steps", type=int, default=100)
    p.add_argument("--results-dir", default="results/runs")
    p = sub.add_parser("sweep")
    p.add_argument("--preset", default="paper", choices=["paper"])
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--results-dir", default="results/runs")
    p = sub.add_parser("aggregate")
    p.add_argument("--runs-dir", default="results/runs")
    p.add_argument("--out", default="results/summary.csv")
    p = sub.add_parser("figures")
    p.add_argument("--summary", default="results/summary.csv")
    p.add_argument("--runs-dir", default="results/runs")
    p.add_argument("--outdir", default="figures/out")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        print(f"scenarios ({len(full_registry())}):")
        for sid, sc in full_registry().items():
            d = sc.dynamics
            print(f"  {sid}: sigma0={sc.ssvi.sigma0:.3f} xi={d['xi']} rho={d['rho']}")
        print("algos:", ", ".join(ALGOS))
        return 0
    if args.cmd == "run":
        cfg = {"n_steps": args.n_steps} if args.n_steps else None
        out = run_path(args.scenario, args.algo, args.n, args.seed, args.results_dir)
        if out.exists():
            print(f"skip (exists): {out}")
            return 0
        p = run_one(args.scenario, args.algo, args.n, args.seed,
                    results_dir=args.results_dir, cfg=cfg,
                    reprice_particles=args.reprice_n, reprice_steps=args.reprice_steps)
        print(f"wrote {p}")
        return 0
    if args.cmd == "sweep":
        todo = [(s, a, n, sd, args.results_dir) for s, a, n, sd in _paper_grid()
                if not run_path(s, a, n, sd, args.results_dir).exists()]
        print(f"{len(todo)} runs to do")
        if args.jobs > 1:
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                for done in ex.map(_do_run, todo):
                    print("done:", *done)
        else:
            for job in todo:
                print("done:", *_do_run(job))
        return 0
    if args.cmd == "aggregate":
        df = aggregate(args.runs_dir, args.out)
        print(f"{len(df)} runs -> {args.out}")
        return 0
    if args.cmd == "figures":
        try:
            import figures.fig1_accuracy, figures.fig2_wings, figures.fig3_plane, figures.fig4_latency
        except ImportError:
            print("no figure modules")
            return 0
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        for mod in (figures.fig1_accuracy, figures.fig2_wings,
                    figures.fig3_plane, figures.fig4_latency):
            print("wrote", mod.make(args.summary, args.runs_dir, args.outdir))
        return 0
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_cli.py -v` — PASS.
- [ ] **Step 5: Commit** — `git add bench/cli.py tests/test_cli.py && git commit -m "feat: full bench CLI with resumable sweep"`

---

### Task 11: Figure scripts

**Files:**
- Create: `figures/__init__.py`, `figures/fig1_accuracy.py`, `figures/fig2_wings.py`, `figures/fig3_plane.py`, `figures/fig4_latency.py`
- Modify: `pyproject.toml` (add `"figures"` to the hatch wheel packages list so `import figures` works)
- Test: `tests/test_figures.py`

**Interfaces:**
- Each module exposes `make(summary_csv, runs_dir, outdir) -> str` (path of the PDF written). All read `summary.csv` via pandas; fig2 and fig3 additionally read run JSONs under `runs_dir` for `iv_err_bp` / `is_diag`.

- [ ] **Step 1: Write failing smoke test**

```python
# tests/test_figures.py
import json
from pathlib import Path
import pandas as pd
import pytest
import figures.fig1_accuracy as f1
import figures.fig2_wings as f2
import figures.fig3_plane as f3
import figures.fig4_latency as f4

def _summary(tmp_path):
    rows = []
    for sid in ("s01", "s02", "f_xi0.3_rho-0.7", "f_xi0.6_rho-0.7"):
        for algo in ("nw", "explicit_nn", "ridge", "explicit_nn_is", "implicit_nn"):
            rows.append(dict(sid=sid, algo=algo, n_particles=50_000, seed=0, status="ok",
                             git_hash="x", pooled_rmse_bp=10.0, pooled_max_bp=20.0,
                             wings_rmse_bp=15.0, wings_max_bp=25.0, n_failed=0,
                             total_s=5.0, fit_s=1.0))
    p = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p

def _runs(tmp_path):
    for sid in ("s01", "f_xi0.3_rho-0.7"):
        for algo in ("nw", "explicit_nn", "explicit_nn_is"):
            d = tmp_path / "runs" / sid / algo
            d.mkdir(parents=True, exist_ok=True)
            doc = {"schema": 1, "sid": sid, "algo": algo, "n_particles": 50_000, "seed": 0,
                   "status": "ok", "iv_err_bp": [[float(i - 6) for i in range(13)]] * 4,
                   "diagnostics": {"is_diag": {"max_w": 1.8, "ess_frac": 0.6}} if "is" in algo else {},
                   "scenario": {"maturities": [0.25, 0.5, 1.0, 2.0]}}
            (d / "n50000_s0.json").write_text(json.dumps(doc))
    return tmp_path / "runs"

@pytest.mark.parametrize("mod", [f1, f2, f3, f4])
def test_each_figure_writes_pdf(tmp_path, mod):
    out = mod.make(str(_summary(tmp_path)), str(_runs(tmp_path)), str(tmp_path / "out"))
    assert Path(out).exists() and out.endswith(".pdf")
```

- [ ] **Step 2: Run, verify failure** — modules missing.

- [ ] **Step 3: Implement the four modules**

```python
# figures/__init__.py
```

```python
# figures/fig1_accuracy.py
"""Fig 1: pooled IV RMSE per algorithm at fixed budget (bars, mean over scenarios/seeds)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def make(summary_csv, runs_dir, outdir):
    df = pd.read_csv(summary_csv)
    ok = df[(df.status == "ok") & (~df.sid.str.startswith("f_"))]
    g = ok.groupby("algo").pooled_rmse_bp.agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(g.index, g["mean"], yerr=g["std"].fillna(0.0), capsize=3)
    ax.set_ylabel("pooled IV RMSE (bp)")
    ax.tick_params(axis="x", rotation=20)
    out = Path(outdir) / "fig1_accuracy.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return str(out)
```

```python
# figures/fig2_wings.py
"""Fig 2: wing IV error with and without the tilt, from per-run iv_err_bp."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _profile(runs_dir, algo):
    errs = []
    for p in Path(runs_dir).rglob("*.json"):
        d = json.loads(p.read_text())
        if d.get("algo") == algo and d.get("status") == "ok" and d.get("iv_err_bp"):
            e = np.array(d["iv_err_bp"], dtype=float)
            errs.append(np.abs(e[-1]))  # longest maturity row
    return np.nanmean(np.stack(errs), axis=0) if errs else None


def make(summary_csv, runs_dir, outdir):
    k = np.log(np.geomspace(0.6, 1.6, 13))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for algo, label in (("explicit_nn", "no tilt"), ("explicit_nn_is", "defensive mixture")):
        prof = _profile(runs_dir, algo)
        if prof is not None:
            ax.plot(k, prof, marker="o", label=label)
    ax.set_xlabel("log-moneyness"); ax.set_ylabel("|IV error| (bp), longest maturity")
    ax.legend()
    out = Path(outdir) / "fig2_wings.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return str(out)
```

```python
# figures/fig3_plane.py
"""Fig 3: (xi, rho)-plane pooled RMSE heatmaps, kernel vs network."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

XIS, RHOS = (0.3, 0.6, 1.0), (-0.7, -0.3)


def make(summary_csv, runs_dir, outdir):
    df = pd.read_csv(summary_csv)
    ok = df[(df.status == "ok") & (df.sid.str.startswith("f_"))]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    for ax, algo in zip(axes, ("nw", "explicit_nn")):
        grid = [[ok[(ok.algo == algo) & (ok.sid == f"f_xi{x}_rho{r}")].pooled_rmse_bp.mean()
                 for x in XIS] for r in RHOS]
        im = ax.imshow(grid, aspect="auto", origin="lower")
        ax.set_xticks(range(len(XIS)), XIS); ax.set_yticks(range(len(RHOS)), RHOS)
        ax.set_xlabel("xi"); ax.set_title(algo)
    axes[0].set_ylabel("rho")
    fig.colorbar(im, ax=axes, label="pooled IV RMSE (bp)")
    out = Path(outdir) / "fig3_plane.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return str(out)
```

```python
# figures/fig4_latency.py
"""Fig 4: latency-accuracy frontier, one point per (algo, N)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def make(summary_csv, runs_dir, outdir):
    df = pd.read_csv(summary_csv)
    ok = df[(df.status == "ok") & (~df.sid.str.startswith("f_"))]
    g = ok.groupby(["algo", "n_particles"]).agg(
        rmse=("pooled_rmse_bp", "mean"), t=("total_s", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for algo, sub in g.groupby("algo"):
        ax.plot(sub.t, sub.rmse, marker="o", label=algo)
    ax.set_xlabel("calibration wall-clock (s)"); ax.set_ylabel("pooled IV RMSE (bp)")
    ax.set_xscale("log"); ax.legend(fontsize=8)
    out = Path(outdir) / "fig4_latency.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return str(out)
```

In `pyproject.toml` change the wheel packages line to:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/neural_particle_method", "bench", "figures"]
```

- [ ] **Step 4: Run tests** — `uv sync && uv run pytest tests/test_figures.py -v` — PASS.
- [ ] **Step 5: Commit** — `git add figures pyproject.toml uv.lock tests/test_figures.py && git commit -m "feat: four figure scripts reading aggregated outputs"`

---

### Task 12: Retire legacy scripts, README, full-suite gate

**Files:**
- Delete: `experiments/exp_a_estimator.py`, `experiments/exp_bc_calibration.py`, `experiments/exp_d_heads.py` (keep `experiments/results/` as archive)
- Create: `README.md`
- Test: full suite

**Interfaces:** none new.

- [ ] **Step 1: Delete legacy scripts**

```bash
git rm experiments/exp_a_estimator.py experiments/exp_bc_calibration.py experiments/exp_d_heads.py
```

- [ ] **Step 2: Write README**

```markdown
# neural_particle_method

Neural L2 calibration of LSV models (Risk paper workspace).

- Paper notes: `paper/notes.tex`; spec and plans under `docs/superpowers/`.
- Benchmark: `uv run bench list | run | sweep --preset paper --jobs 4 | aggregate | figures`.
- One JSON per (scenario, algo, N, seed) under `results/runs/` (gitignored); `results/summary.csv` and `figures/out/` are the committed artifacts.
- Tests: `uv run pytest` (fast, no large simulations).
- Legacy exploratory results are archived in `experiments/results/`.
```

- [ ] **Step 3: Run the whole suite and a real tiny run**

```bash
uv run pytest -q
uv run bench run --scenario s01 --algo ridge --n 20000 --seed 0 --n-steps 25 --reprice-n 50000 --reprice-steps 25
uv run bench aggregate
```

Expected: suite passes; the run writes an `ok` JSON with pooled RMSE printed by aggregate.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: retire legacy experiment scripts, add README"
```

---

## Self-Review Notes

- Spec coverage: layout (T1, T11), CLI (T1, T10), SSVI + no-arb + analytic Dupire + QL cross-check (T2), metric protocol incl. wings and NaN policy (T3, T8), scenarios incl. fig3 cross (T4), five contenders (T5-T7), mixture design from the notes' Algorithm 4 (T5), run artifacts with git hash and failure capture (T8), aggregation + digest (T9), resumable sweep with --jobs (T10), four figures (T11), legacy retirement (T12). Real-market surfaces, PURBF/bins, PDE are out of scope per spec.
- Type consistency: `L_records` tuples `(t, grid, L_grid, f_grid)` everywhere; `dynamics` dict keys `kappa, theta, xi, rho, v0` everywhere; `CalibResult.timings` keys `total_s`/`fit_s` consumed by T8/T9/T11.
- Known judgment calls recorded in Global Constraints: bs.implied_vol instead of QuantLib for the hot-loop inversion; RNG decomposition change in explicit.py.
