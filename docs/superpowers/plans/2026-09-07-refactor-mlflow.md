# Package Refactor, Unit Tests, and MLflow Tracking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the repository into one layered package under `src/neural_particle_method/`, replace the JSON result stores with MLflow, and raise test coverage, while reproducing every archived result bit for bit.

**Architecture:** Subpackages by responsibility (`market`, `simulate`, `estimators`, `calibrate`, `pricing`, `tracking`, `bench`, `experiments`, `figures`) plus one `nparticle` CLI. The Heston Euler step and the ridge readout exist once. Leverage records become a typed `LeverageField`. Only `tracking/store.py` imports mlflow; drivers call it, library code never logs. Migration is bottom-up with import shims at old paths until the final task.

**Tech Stack:** Python 3.12, numpy, scipy, torch (CPU), pandas, matplotlib, QuantLib (tests only), mlflow (SQLite backend), pytest, pytest-cov, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-09-07-refactor-mlflow-design.md`. Read it first.

## Global Constraints

- **Bit-for-bit fidelity.** Every numerical expression is moved verbatim: same operation order, same clip constants, same RNG call order. If a golden test fails and the cause is a bug in the old code, STOP and report to the user with evidence; never change the expected value.
- **RNG draw order** (from the spec, do not change): `calibrate_explicit` draws `rng.choice(3, n, p=alphas)` once before the loop only under a mixture; per step k > 0 one `rng.choice(n_particles, size=min(fit_subsample, n_particles), replace=False)`, then `zb = rng.standard_normal(n)`, then `zp = rng.standard_normal(n)`. `simulate_slices` and `reprice_iv` draw `zb` then `zp` per step. `calibrate_implicit` seeds numpy then torch, and per iteration simulates then draws one `rng.choice(len(pool), size=..., replace=False)`. Torch is seeded inside estimator constructors and in `calibrate_implicit` before `GlobalNet()`.
- **Python 3.12**, `requires-python = ">=3.12"`. Run everything with `uv run`.
- **Test suite**: default `uv run pytest` must stay under about 30 s and green after every task. Markers `golden` and `slow` are deselected by default.
- **Console script** is `nparticle`, never `npm`.
- **Tracking URI**: `MLFLOW_TRACKING_URI` if set, else `sqlite:///<cwd>/mlruns.db` with artifact root `<cwd>/mlartifacts`. Both gitignored.
- **Metric keys** use `/` as separator (`rmse_bp/<strategy>`). MLflow accepts `/` in metric and param keys.
- **Commit after every task** with a message in the form shown; end commit messages with the attribution block:

```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EhbyGqFA5kYDyMwYouSgvS
```

- Root `experiments/*.py` scripts are legacy from Task 5 onward (their imports break when the calibrate API changes). Do not run them; they are rewritten inside the package in Task 9 and deleted in Task 12.

## File Structure (final state)

```
src/neural_particle_method/
  __init__.py
  market/__init__.py, bs.py, heston.py, ssvi.py, dupire.py, local_vol.py
  simulate/__init__.py, dynamics.py, stepper.py, leverage.py
  estimators/__init__.py, base.py, nn.py, nadaraya_watson.py, spline.py, ridge.py, kalman.py
  calibrate/__init__.py, config.py, explicit.py, implicit.py, importance.py, warm.py
  pricing/__init__.py, reprice.py, metrics.py
  tracking/__init__.py, store.py, importer.py
  bench/__init__.py, scenarios.py, algos.py, runner.py, sweep.py, aggregate.py
  experiments/__init__.py, config.py, bump_correct.py, warm_suite.py
  figures/__init__.py, fig1_accuracy.py, fig2_wings.py, fig3_plane.py, fig4_latency.py
  cli.py
tests/
  golden/make_tiny_goldens.py, golden/tiny_<algo>.json
  test_golden_tiny.py, test_golden_archived.py
  market/, simulate/, estimators/, calibrate/, pricing/, tracking/, bench/, experiments/, figures/
```

Existing tests are moved into the matching subdirectory when their module moves. `tests/__init__.py` and each `tests/<sub>/__init__.py` exist (empty) so test modules import as `tests.<sub>.<name>` and never shadow real packages such as `bench`.

---

### Task 1: Tooling, markers, and golden tests

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `tests/golden/make_tiny_goldens.py`
- Create: `tests/golden/tiny_<algo>.json` (generated, seven files)
- Create: `tests/test_golden_tiny.py`
- Create: `tests/test_golden_archived.py`
- Create: `tests/__init__.py` (empty), `tests/conftest.py`

**Interfaces:**
- Produces: `tests/conftest.py::TINY`, `TINY_N`, `TINY_REPRICE_N`, `TINY_REPRICE_STEPS` constants (Task 5 adds dataclass equivalents); `tests/test_golden_tiny.py::run_tiny(algo)` helper that later tasks update when the bench API changes.

- [ ] **Step 1: Add dependencies, pytest markers, and gitignore entries**

Edit `pyproject.toml` so the relevant sections read:

```toml
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
    "torch>=2.3",
    "matplotlib>=3.8",
    "tqdm>=4.66",
    "quantlib>=1.43",
    "pandas>=3.0.5",
    "mlflow>=2.14",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.6",
    "ipykernel>=6.29",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "golden: bit-for-bit replay of archived runs; minutes; run with -m golden",
    "slow: multi-minute smoke runs; run with -m slow",
]
addopts = "-m 'not golden and not slow'"
```

Leave `[project.scripts] bench = "bench.cli:main"` in place for now (Task 12 replaces it). Append to `.gitignore`:

```
# MLflow local store
mlruns.db
mlruns/
mlartifacts/
```

Run: `uv sync` and then `uv run python -c "import mlflow; print(mlflow.__version__)"`.
Expected: a version at or above 2.14 prints.

- [ ] **Step 2: Make tests a package and write conftest with the tiny configs**

Create an empty `tests/__init__.py`. Create `tests/conftest.py`:

```python
"""Shared tiny configurations. Small enough that every algorithm runs in about a second."""
TINY = {"n_steps": 6, "fit_subsample": 4_000, "n_iters": 2,
        "first_steps": 80, "later_steps": 30, "fit_steps": 80}
TINY_N = 4_000
TINY_REPRICE_N = 8_000
TINY_REPRICE_STEPS = 8
```

- [ ] **Step 3: Write the tiny golden generator**

Create `tests/golden/make_tiny_goldens.py`:

```python
"""Generate tests/golden/tiny_<algo>.json from the CURRENT code. Run once before the refactor.

Usage: uv run python tests/golden/make_tiny_goldens.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.conftest import TINY, TINY_N, TINY_REPRICE_N, TINY_REPRICE_STEPS  # noqa: E402

from bench.algos import ALGOS, run_algo  # noqa: E402
from bench.scenarios import make_registry, quote_k_grid  # noqa: E402
from neural_particle_method.reprice import iv_metrics, reprice_iv, snap_times  # noqa: E402
from neural_particle_method.ssvi import implied_vol_ssvi  # noqa: E402

OUT = Path(__file__).parent


def main():
    sc = make_registry()["s01"]
    k = quote_k_grid()
    mats = list(sc.maturities)
    for algo in ALGOS:
        res = run_algo(algo, sc, TINY_N, 0, TINY)
        ivs = reprice_iv(res.L_records, sc.dynamics, sc.s0, mats, k,
                         n_particles=TINY_REPRICE_N, n_steps=TINY_REPRICE_STEPS, seed=10_000)
        ts = snap_times(mats, TINY_REPRICE_STEPS)
        tgt = np.stack([implied_vol_ssvi(sc.ssvi, k, t) for t in ts])
        doc = {
            "algo": algo, "n_particles": TINY_N, "seed": 0, "cfg": TINY,
            "L_records": [{"t": float(t), "grid": g.tolist(), "L": L.tolist(), "f": f.tolist()}
                          for (t, g, L, f) in res.L_records],
            "metrics": iv_metrics(ivs, tgt, k, mats),
            "iv_err_bp": ((ivs - tgt) * 1e4).tolist(),
        }
        (OUT / f"tiny_{algo}.json").write_text(json.dumps(doc))
        print("wrote", algo)


if __name__ == "__main__":
    main()
```

Run: `uv run python tests/golden/make_tiny_goldens.py`
Expected: seven lines `wrote <algo>` and seven JSON files in `tests/golden/`.

- [ ] **Step 4: Write the fast tiny golden test**

Create `tests/test_golden_tiny.py`:

```python
"""Bit-for-bit replay of tiny goldens generated from the pre-refactor code."""
import json
from pathlib import Path

import numpy as np
import pytest

from bench.algos import ALGOS, run_algo
from tests.conftest import TINY, TINY_N, TINY_REPRICE_N, TINY_REPRICE_STEPS
from bench.scenarios import make_registry, quote_k_grid
from neural_particle_method.reprice import iv_metrics, reprice_iv, snap_times
from neural_particle_method.ssvi import implied_vol_ssvi

GOLDEN = Path(__file__).parent / "golden"


def run_tiny(algo):
    """Run one algorithm at the tiny config. Later tasks update this helper as the API moves;
    the assertions below never change."""
    sc = make_registry()["s01"]
    k = quote_k_grid()
    mats = list(sc.maturities)
    res = run_algo(algo, sc, TINY_N, 0, TINY)
    ivs = reprice_iv(res.L_records, sc.dynamics, sc.s0, mats, k,
                     n_particles=TINY_REPRICE_N, n_steps=TINY_REPRICE_STEPS, seed=10_000)
    ts = snap_times(mats, TINY_REPRICE_STEPS)
    tgt = np.stack([implied_vol_ssvi(sc.ssvi, k, t) for t in ts])
    records = [(float(t), np.asarray(g), np.asarray(L), np.asarray(f)) for (t, g, L, f) in res.L_records]
    return records, iv_metrics(ivs, tgt, k, mats), (ivs - tgt) * 1e4


@pytest.mark.parametrize("algo", sorted(ALGOS))
def test_tiny_golden_bit_for_bit(algo):
    doc = json.loads((GOLDEN / f"tiny_{algo}.json").read_text())
    records, metrics, err = run_tiny(algo)
    assert len(records) == len(doc["L_records"])
    for (t, g, L, f), ref in zip(records, doc["L_records"]):
        assert t == ref["t"]
        np.testing.assert_array_equal(g, np.array(ref["grid"]))
        np.testing.assert_array_equal(L, np.array(ref["L"]))
        np.testing.assert_array_equal(f, np.array(ref["f"]))
    for key in ("pooled_rmse_bp", "pooled_max_bp", "wings_rmse_bp", "wings_max_bp", "n_failed"):
        assert metrics[key] == doc["metrics"][key], key
    for got, ref in zip(metrics["per_maturity"], doc["metrics"]["per_maturity"]):
        assert got == ref
    np.testing.assert_array_equal(err, np.array(doc["iv_err_bp"], dtype=float))
```

- [ ] **Step 5: Run the tiny golden test twice to prove determinism**

Run: `uv run pytest tests/test_golden_tiny.py -v` twice.
Expected: 7 passed both times. If any algorithm differs between runs, torch is nondeterministic on this machine at this size; report to the user before continuing.

- [ ] **Step 6: Write the archived golden test**

Create `tests/test_golden_archived.py`:

```python
"""Bit-for-bit replay of four archived paper runs. Slow: run with `uv run pytest -m golden`."""
import json
from pathlib import Path

import numpy as np
import pytest

from bench.algos import run_algo
from bench.scenarios import make_registry, quote_k_grid
from neural_particle_method.reprice import iv_metrics, reprice_iv, snap_times
from neural_particle_method.ssvi import implied_vol_ssvi

RUNS = Path(__file__).resolve().parents[1] / "results" / "runs"
CASES = ["nw", "explicit_nn", "explicit_nn_is", "implicit_ridge"]


def replay(algo, n_particles=50_000, seed=0, sid="s01"):
    sc = make_registry()[sid]
    k = quote_k_grid()
    mats = list(sc.maturities)
    res = run_algo(algo, sc, n_particles, seed, None)
    ivs = reprice_iv(res.L_records, sc.dynamics, sc.s0, mats, k,
                     n_particles=500_000, n_steps=200, seed=seed + 10_000)
    ts = snap_times(mats, 200)
    tgt = np.stack([implied_vol_ssvi(sc.ssvi, k, t) for t in ts])
    return iv_metrics(ivs, tgt, k, mats), (ivs - tgt) * 1e4


@pytest.mark.golden
@pytest.mark.parametrize("algo", CASES)
def test_archived_run_replays_bit_for_bit(algo):
    doc = json.loads((RUNS / "s01" / algo / "n50000_s0.json").read_text())
    assert doc["status"] == "ok"
    metrics, err = replay(algo)
    for key in ("pooled_rmse_bp", "pooled_max_bp", "wings_rmse_bp", "wings_max_bp", "n_failed"):
        assert metrics[key] == doc["metrics"][key], key
    ref = np.array([[np.nan if v is None else v for v in row] for row in doc["iv_err_bp"]])
    np.testing.assert_array_equal(err, ref)
```

- [ ] **Step 7: Run the archived goldens once with the current code**

Run: `uv run pytest -m golden -v` (several minutes).
Expected: 4 passed. If any fails, the current code does not reproduce the archive on this machine. STOP and report the exact failure to the user; do not proceed to Task 2.

- [ ] **Step 8: Run the full default suite and commit**

Run: `uv run pytest -q`
Expected: 58 passed (51 existing plus 7 tiny goldens); golden tests deselected.

```bash
git add pyproject.toml uv.lock .gitignore tests/__init__.py tests/conftest.py tests/golden tests/test_golden_tiny.py tests/test_golden_archived.py
git commit -m "test: tiny and archived golden replays; add mlflow, pytest-cov, markers"
```

---

### Task 2: `market/` subpackage with a single Dupire formula

**Files:**
- Create: `src/neural_particle_method/market/__init__.py`
- Create: `src/neural_particle_method/market/bs.py` (moved from `bs.py`)
- Create: `src/neural_particle_method/market/heston.py` (moved from `heston.py`)
- Create: `src/neural_particle_method/market/ssvi.py` (moved from `ssvi.py`, minus `SSVILocalVol`)
- Create: `src/neural_particle_method/market/local_vol.py`
- Create: `src/neural_particle_method/market/dupire.py` (moved from `dupire.py`)
- Modify: `src/neural_particle_method/bs.py`, `heston.py`, `ssvi.py`, `dupire.py` become shims
- Create: `tests/market/__init__.py`, `tests/market/test_local_vol.py`, `tests/market/test_ssvi_derivs.py`
- Move: `tests/test_sanity.py` -> `tests/market/test_sanity.py`, `tests/test_ssvi.py` -> `tests/market/test_ssvi.py`

**Interfaces:**
- Produces: `market.local_vol.LocalVol` (Protocol with `sigma(t, x, s0=1.0)` and `t_min`), `market.local_vol.dupire_local_vol(w, dwdk, d2wdk2, dwdT, k)`, `market.local_vol.SSVILocalVol`, `market.dupire.DupireSurface`, `market.bs.bs_call`, `market.bs.implied_vol`, `market.heston.heston_call`, `market.heston.heston_cf`, `market.ssvi.SSVIParams`, `no_arb_ok`, `w_and_derivs`, `total_variance`, `implied_vol_ssvi`.
- Both local-vol classes keep the `T_grid` attribute (existing callers use `dupire.T_grid[0]`) and add `t_min` as a property returning `T_grid[0]`.

- [ ] **Step 1: Write the failing test for the shared Dupire formula**

Create `tests/market/__init__.py` (empty) and `tests/market/test_local_vol.py`:

```python
import numpy as np

from neural_particle_method.market.dupire import DupireSurface
from neural_particle_method.market.local_vol import SSVILocalVol, dupire_local_vol
from neural_particle_method.market.ssvi import SSVIParams, total_variance, w_and_derivs

P = SSVIParams(sigma0=0.2, eta=1.0, gamma=0.4, rho=-0.6)


def test_dupire_local_vol_flat_surface_is_flat():
    k = np.linspace(-0.5, 0.5, 11)
    w = np.full_like(k, 0.04)          # w = sigma^2 T with sigma = 0.2, T = 1
    sig = dupire_local_vol(w, np.zeros_like(k), np.zeros_like(k), np.full_like(k, 0.04), k)
    assert np.allclose(sig, 0.2)


def test_ssvi_local_vol_uses_shared_formula():
    lv = SSVILocalVol(P)
    k = np.array([-0.2, 0.0, 0.3])
    w, dwdk, d2wdk2, dwdT = w_and_derivs(P, k, 1.0)
    expected = dupire_local_vol(w, dwdk, d2wdk2, dwdT, k)
    np.testing.assert_array_equal(lv.sigma(1.0, np.exp(k)), expected)


def test_grid_surface_agrees_with_analytic_on_same_ssvi():
    T_grid = np.linspace(0.1, 1.5, 29)
    k_grid = np.linspace(-0.6, 0.6, 61)
    w = np.stack([total_variance(P, k_grid, T) for T in T_grid])
    grid = DupireSurface(T_grid, k_grid, w)
    analytic = SSVILocalVol(P)
    for t in (0.5, 1.0):
        for k in (-0.2, 0.0, 0.2):
            ours = float(grid.sigma(t, np.exp(k)))
            theirs = float(analytic.sigma(t, np.exp(k)))
            assert abs(ours - theirs) / theirs < 0.05, (t, k, ours, theirs)


def test_t_min_property():
    assert SSVILocalVol(P).t_min == SSVILocalVol(P).T_grid[0]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/market/test_local_vol.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'neural_particle_method.market'`.

- [ ] **Step 3: Create the subpackage**

`git mv src/neural_particle_method/bs.py src/neural_particle_method/market/bs.py` and likewise for `heston.py`, `ssvi.py`, `dupire.py` (create the directory first). Create `src/neural_particle_method/market/__init__.py`:

```python
"""Market models: Black-Scholes, Heston, SSVI surfaces, Dupire local volatility."""
```

Create `src/neural_particle_method/market/local_vol.py`:

```python
"""Local volatility: the Dupire formula (written once) and the LocalVol protocol."""
from typing import Protocol

import numpy as np

from .ssvi import w_and_derivs


class LocalVol(Protocol):
    T_grid: np.ndarray

    @property
    def t_min(self) -> float: ...

    def sigma(self, t, x, s0=1.0): ...


def dupire_local_vol(w, dwdk, d2wdk2, dwdT, k):
    """Dupire local vol from total implied variance w(k, T) and its derivatives.

    Clipping constants are part of the calibrated behaviour: denominator floored at 0.05,
    local variance in [1e-8, 9]. Do not change them.
    """
    denom = (1.0 - k / w * dwdk
             + 0.25 * (-0.25 - 1.0 / w + (k / w) ** 2) * dwdk ** 2
             + 0.5 * d2wdk2)
    v_loc = np.clip(dwdT, 1e-8, None) / np.clip(denom, 0.05, None)
    return np.sqrt(np.clip(v_loc, 1e-8, 9.0))


class SSVILocalVol:
    """Analytic Dupire local vol from an SSVI surface."""

    def __init__(self, p, s0=1.0, t_min=0.004, T_max=2.0):
        self.p, self.s0 = p, s0
        self.T_grid = np.array([t_min, T_max])

    @property
    def t_min(self):
        return float(self.T_grid[0])

    def sigma(self, t, x, s0=1.0):
        t = float(np.clip(t, self.T_grid[0], self.T_grid[-1]))
        k = np.log(np.asarray(x, dtype=float) / s0)
        w, dwdk, d2wdk2, dwdT = w_and_derivs(self.p, k, T=t)
        return dupire_local_vol(w, dwdk, d2wdk2, dwdT, k)
```

Edit `market/ssvi.py`: rename `_w_and_derivs` to `w_and_derivs` (update the two internal callers), delete the `SSVILocalVol` class from it.

Edit `market/dupire.py` so the constructor uses the shared formula and the class exposes `t_min`:

```python
"""Dupire local volatility surface from a call-price function, via total implied variance."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .bs import implied_vol
from .local_vol import dupire_local_vol


class DupireSurface:
    """sigma_Dup(t, x) built on a (T, k) grid, k = log(K/s0). Inputs clamped to the grid."""

    def __init__(self, T_grid, k_grid, w):
        self.T_grid, self.k_grid = T_grid, k_grid
        K = k_grid[None, :]
        W = w
        dwdT = np.gradient(W, T_grid, axis=0)
        dwdk = np.gradient(W, k_grid, axis=1)
        d2wdk2 = np.gradient(dwdk, k_grid, axis=1)
        self.sigma_loc = dupire_local_vol(W, dwdk, d2wdk2, dwdT, K)
        self._interp = RegularGridInterpolator(
            (T_grid, k_grid), self.sigma_loc, bounds_error=False, fill_value=None)

    @property
    def t_min(self):
        return float(self.T_grid[0])

    @classmethod
    def from_price_fn(cls, price_fn, s0, T_grid, k_grid):
        w = np.empty((len(T_grid), len(k_grid)))
        for i, T in enumerate(T_grid):
            K = s0 * np.exp(k_grid)
            prices = price_fn(K, T)
            for j, (p, k) in enumerate(zip(prices, K)):
                iv = implied_vol(p, s0, k, T)
                w[i, j] = (iv ** 2) * T if np.isfinite(iv) else np.nan
        for i in range(w.shape[0]):
            row = w[i]
            if np.isnan(row).any():
                idx = np.arange(len(row))
                good = ~np.isnan(row)
                w[i] = np.interp(idx, idx[good], row[good])
        return cls(T_grid, k_grid, w)

    def sigma(self, t, x, s0=1.0):
        t = np.clip(t, self.T_grid[0], self.T_grid[-1])
        k = np.clip(np.log(np.asarray(x) / s0), self.k_grid[0], self.k_grid[-1])
        pts = np.stack([np.full_like(k, t, dtype=float), k], axis=-1)
        return self._interp(pts)
```

Note the original built `denom` from `K = k_grid[None, :]` broadcast against 2-D `W`; passing `K` as `k` reproduces the same broadcast expression exactly.

Create the four shims at the old paths, each one line, e.g. `src/neural_particle_method/ssvi.py`:

```python
from .market.ssvi import *  # noqa: F401,F403  (shim; removed in Task 12)
from .market.ssvi import SSVIParams, no_arb_ok, total_variance, implied_vol_ssvi  # noqa: F401
from .market.local_vol import SSVILocalVol  # noqa: F401
```

`bs.py` shim: `from .market.bs import bs_call, implied_vol  # noqa: F401`.
`heston.py` shim: `from .market.heston import heston_cf, heston_call  # noqa: F401`.
`dupire.py` shim: `from .market.dupire import DupireSurface  # noqa: F401`.

- [ ] **Step 4: Write the SSVI derivative test**

Create `tests/market/test_ssvi_derivs.py`:

```python
import numpy as np

from neural_particle_method.market.ssvi import SSVIParams, w_and_derivs

P = SSVIParams(sigma0=0.25, eta=0.8, gamma=0.45, rho=-0.5)


def _fd(fn, x, h=1e-5):
    return (fn(x + h) - fn(x - h)) / (2 * h)


def test_k_derivatives_match_finite_differences():
    k = np.array([-0.4, -0.1, 0.0, 0.2, 0.5])
    T = 0.8
    w, dwdk, d2wdk2, _ = w_and_derivs(P, k, T)
    fd1 = _fd(lambda kk: w_and_derivs(P, kk, T)[0], k)
    fd2 = _fd(lambda kk: w_and_derivs(P, kk, T)[1], k)
    assert np.allclose(dwdk, fd1, rtol=1e-6, atol=1e-9)
    assert np.allclose(d2wdk2, fd2, rtol=1e-5, atol=1e-8)


def test_T_derivative_matches_finite_difference():
    k = np.array([-0.3, 0.0, 0.3])
    T = 1.2
    _, _, _, dwdT = w_and_derivs(P, k, T)
    fd = _fd(lambda TT: w_and_derivs(P, k, TT)[0], T)
    assert np.allclose(dwdT, fd, rtol=1e-6, atol=1e-9)
```

- [ ] **Step 5: Move the existing market tests and update imports**

`git mv tests/test_sanity.py tests/market/test_sanity.py`, `git mv tests/test_ssvi.py tests/market/test_ssvi.py`. In both, change imports to `neural_particle_method.market.heston`, `.market.bs`, `.market.dupire`, `.market.ssvi`, and `from neural_particle_method.market.local_vol import SSVILocalVol`.

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass, including `tests/test_golden_tiny.py` (the shims keep old imports working). Count: 58 + 4 + 2 = 64.

- [ ] **Step 7: Commit**

```bash
git add -A src/neural_particle_method tests
git commit -m "refactor: market subpackage; Dupire formula written once in local_vol"
```

---

### Task 3: `simulate/` subpackage: `HestonParams`, `heston_step`, `LeverageField`

**Files:**
- Create: `src/neural_particle_method/simulate/__init__.py`, `dynamics.py`, `stepper.py`, `leverage.py`
- Modify: `src/neural_particle_method/explicit.py` (use stepper), `implicit.py` (use stepper and `LeverageField.resample`), `reprice.py` (`L_lookup` delegates to `LeverageField`)
- Create: `tests/simulate/__init__.py`, `tests/simulate/test_stepper.py`, `tests/simulate/test_leverage.py`, `tests/simulate/test_dynamics.py`

**Interfaces:**
- Produces:
  - `HestonParams(kappa, theta, xi, rho, v0)` frozen dataclass with `from_dict(d)` classmethod and `to_dict()`.
  - `heston_step(lnx, v, L_p, zb, zp, params, dt, sdt, theta_p=None) -> (lnx_new, v_new)`.
  - `Slice(t, grid, L, f)` frozen dataclass and `LeverageField(slices)` with `times`, `__len__`, `__getitem__`, `__iter__`, `at(t, lnx)`, `resample(grid)`, `to_records()`, `from_records(records)` classmethod, `to_json()`, `from_json(d)` classmethod, `L_matrix()`.
  - `DEFAULT_GRID = np.linspace(np.log(0.4), np.log(2.2), 81)`.

- [ ] **Step 1: Write the failing tests**

`tests/simulate/__init__.py` empty. `tests/simulate/test_dynamics.py`:

```python
from neural_particle_method.simulate.dynamics import HestonParams


def test_round_trip_dict():
    d = {"kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7, "v0": 0.05}
    p = HestonParams.from_dict(d)
    assert p.kappa == 2.0 and p.v0 == 0.05
    assert p.to_dict() == d


def test_from_params_is_identity():
    p = HestonParams(1.0, 0.04, 0.3, -0.5, 0.04)
    assert HestonParams.from_dict(p) is p
```

`tests/simulate/test_stepper.py`:

```python
import numpy as np

from neural_particle_method.simulate.dynamics import HestonParams
from neural_particle_method.simulate.stepper import heston_step

P = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)


def test_step_matches_hand_computation():
    lnx = np.array([0.0]); v = np.array([0.04]); L = np.array([1.5])
    zb = np.array([0.5]); zp = np.array([-1.0])
    dt, sdt = 0.01, 0.1
    x1, v1 = heston_step(lnx, v, L, zb, zp, P, dt, sdt)
    z1 = P.rho * zb + np.sqrt(1 - P.rho ** 2) * zp
    exp_x = lnx + (-0.5 * L ** 2 * 0.04) * dt + L * 0.2 * sdt * z1
    exp_v = v + P.kappa * (P.theta - 0.04) * dt + P.xi * 0.2 * sdt * zb
    np.testing.assert_array_equal(x1, exp_x)
    np.testing.assert_array_equal(v1, exp_v)


def test_negative_variance_is_floored_in_coefficients():
    lnx = np.zeros(1); v = np.array([-0.01]); L = np.ones(1)
    x1, v1 = heston_step(lnx, v, L, np.ones(1), np.ones(1), P, 0.01, 0.1)
    assert x1[0] == 0.0                      # sqrt(max(v,0)) = 0 kills the diffusion
    assert v1[0] == -0.01 + P.kappa * P.theta * 0.01


def test_tilt_adds_drift_only():
    lnx = np.zeros(2); v = np.full(2, 0.04); L = np.ones(2)
    zb = np.zeros(2); zp = np.zeros(2)
    theta_p = np.array([0.0, 2.0])
    x1, v1 = heston_step(lnx, v, L, zb, zp, P, 0.01, 0.1, theta_p=theta_p)
    assert x1[1] - x1[0] == (1.0 * 0.2 * 2.0) * 0.01
    np.testing.assert_array_equal(v1, np.full(2, 0.04))
```

`tests/simulate/test_leverage.py`:

```python
import json

import numpy as np

from neural_particle_method.simulate.leverage import DEFAULT_GRID, LeverageField, Slice


def _field():
    g = np.linspace(-1.0, 1.0, 5)
    return LeverageField([Slice(i * 0.25, g, np.full(5, 1.0 + i), np.full(5, 0.04)) for i in range(4)])


def test_at_is_piecewise_constant_in_t_and_interpolates_in_x():
    f = _field()
    assert np.allclose(f.at(0.3, np.array([0.0, 0.2])), 2.0)      # slice index 1
    assert np.allclose(f.at(0.0, np.array([0.0])), 1.0)
    assert np.allclose(f.at(5.0, np.array([0.0])), 4.0)             # clamps to last slice
    g = np.array([-1.0, 1.0])
    f2 = LeverageField([Slice(0.0, g, np.array([1.0, 3.0]), np.array([0.04, 0.04]))])
    assert f2.at(0.0, np.array([0.0]))[0] == 2.0


def test_single_point_grid_is_constant():
    f = LeverageField([Slice(0.0, np.array([0.0]), np.array([1.7]), np.array([0.04]))])
    np.testing.assert_array_equal(f.at(0.0, np.array([-2.0, 0.5])), np.array([1.7, 1.7]))


def test_resample_matches_np_interp_and_single_point_rule():
    g = np.array([-1.0, 1.0])
    single = Slice(0.0, np.array([0.0]), np.array([1.7]), np.array([0.05]))
    two = Slice(0.5, g, np.array([1.0, 3.0]), np.array([0.04, 0.06]))
    r = LeverageField([single, two]).resample(DEFAULT_GRID)
    np.testing.assert_array_equal(r[0].L, np.full(len(DEFAULT_GRID), 1.7))
    np.testing.assert_array_equal(r[0].f, np.full(len(DEFAULT_GRID), 0.05))
    np.testing.assert_array_equal(r[1].L, np.interp(DEFAULT_GRID, g, two.L))
    np.testing.assert_array_equal(r[1].f, np.interp(DEFAULT_GRID, g, two.f))
    assert r.times.tolist() == [0.0, 0.5]


def test_records_and_json_round_trip_exact():
    f = _field()
    recs = f.to_records()
    assert isinstance(recs[0], tuple) and len(recs[0]) == 4
    g = LeverageField.from_records(recs)
    h = LeverageField.from_json(json.loads(json.dumps(f.to_json())))
    for a, b, c in zip(f, g, h):
        assert a.t == b.t == c.t
        np.testing.assert_array_equal(a.L, b.L); np.testing.assert_array_equal(a.L, c.L)
        np.testing.assert_array_equal(a.f, c.f); np.testing.assert_array_equal(a.grid, c.grid)


def test_L_matrix_stacks_slices():
    m = _field().L_matrix()
    assert m.shape == (4, 5) and m[2, 0] == 3.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/simulate -q`
Expected: ImportError on `neural_particle_method.simulate`.

- [ ] **Step 3: Implement the subpackage**

`src/neural_particle_method/simulate/__init__.py`:

```python
"""Particle simulation: Heston-plus-leverage dynamics, the Euler step, and the leverage field."""
```

`simulate/dynamics.py`:

```python
"""Heston dynamics parameters."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HestonParams:
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float

    @classmethod
    def from_dict(cls, d):
        if isinstance(d, cls):
            return d
        return cls(kappa=float(d["kappa"]), theta=float(d["theta"]), xi=float(d["xi"]),
                   rho=float(d["rho"]), v0=float(d["v0"]))

    def to_dict(self):
        return asdict(self)
```

`simulate/stepper.py`:

```python
"""One Euler step of (ln X, V) under a Heston variance and a leverage value per particle.

Never draws random numbers: callers pass the two standard normals so RNG order is theirs.
The expressions are verbatim from the pre-refactor loops; keep the operation order.
"""
import numpy as np


def heston_step(lnx, v, L_p, zb, zp, params, dt, sdt, theta_p=None):
    """Return (lnx_new, v_new). `theta_p` is the optional per-particle mixture tilt."""
    rho, kappa, theta, xi = params.rho, params.kappa, params.theta, params.xi
    z1 = rho * zb + np.sqrt(1 - rho ** 2) * zp
    vp = np.maximum(v, 0.0)
    drift_x = -0.5 * L_p ** 2 * vp
    if theta_p is not None:
        drift_x = drift_x + L_p * np.sqrt(vp) * theta_p
    lnx_new = lnx + drift_x * dt + L_p * np.sqrt(vp) * sdt * z1
    v_new = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp) * sdt * zb
    return lnx_new, v_new
```

Note: the original explicit loop wrote `lnx + drift_x * dt + ...` and the implicit and reprice loops wrote `lnx + (-0.5 * L_p ** 2 * vp) * dt + ...`; with `theta_p=None` these are the same floating-point expression.

`simulate/leverage.py`:

```python
"""Leverage field L(t, x) on per-slice grids, with the value E[V | ln X] used to build it."""
from dataclasses import dataclass

import numpy as np

DEFAULT_GRID = np.linspace(np.log(0.4), np.log(2.2), 81)


@dataclass(frozen=True)
class Slice:
    t: float
    grid: np.ndarray
    L: np.ndarray
    f: np.ndarray


class LeverageField:
    """Piecewise-constant in t, linear in ln x; a one-point grid means a constant slice."""

    def __init__(self, slices):
        self.slices = list(slices)
        self._times = np.array([s.t for s in self.slices])

    @property
    def times(self):
        return self._times

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, k):
        return self.slices[k]

    def __iter__(self):
        return iter(self.slices)

    def at(self, t, lnx):
        i = max(int(np.searchsorted(self._times, t + 1e-12)) - 1, 0)
        s = self.slices[i]
        if len(s.grid) == 1:
            return np.full_like(lnx, s.L[0])
        return np.interp(lnx, s.grid, s.L)

    def resample(self, grid):
        out = []
        for s in self.slices:
            if len(s.grid) > 1:
                L, f = np.interp(grid, s.grid, s.L), np.interp(grid, s.grid, s.f)
            else:
                L, f = np.full(len(grid), s.L[0]), np.full(len(grid), s.f[0])
            out.append(Slice(s.t, grid.copy(), L, f))
        return LeverageField(out)

    def L_matrix(self):
        return np.stack([s.L for s in self.slices])

    def to_records(self):
        return [(s.t, s.grid, s.L, s.f) for s in self.slices]

    @classmethod
    def from_records(cls, records):
        if isinstance(records, cls):
            return records
        return cls(Slice(float(t), np.asarray(g, dtype=float), np.asarray(L, dtype=float),
                         np.asarray(f, dtype=float)) for (t, g, L, f) in records)

    def to_json(self):
        return {"slices": [{"t": float(s.t), "grid": s.grid.tolist(), "L": s.L.tolist(),
                            "f": s.f.tolist()} for s in self.slices]}

    @classmethod
    def from_json(cls, d):
        return cls(Slice(float(s["t"]), np.array(s["grid"], dtype=float),
                         np.array(s["L"], dtype=float), np.array(s["f"], dtype=float))
                   for s in d["slices"])
```

- [ ] **Step 4: Switch the three loops to the stepper and `LeverageField`**

In `src/neural_particle_method/explicit.py` add `from .simulate.dynamics import HestonParams` and `from .simulate.stepper import heston_step`. Replace the parameter unpacking line with:

```python
    hp = HestonParams.from_dict(params)
    kappa, theta, xi, rho, v0 = hp.kappa, hp.theta, hp.xi, hp.rho, hp.v0
```

(`kappa`, `xi` remain used by nothing after the switch; remove them if ruff flags them.) Replace the block from `z1 = rho * zb + ...` through `v = v + kappa * ...` with:

```python
        lnx, v = heston_step(lnx, v, L_p, zb, zp, hp, dt, sdt,
                             theta_p=theta_p if mixture is not None else None)
```

Keep the `zb = ...` and `zp = ...` draws immediately before it, in that order, and keep the mixture `ell`/`w` update after it unchanged. Note the original computed `vp` before the step and the mixture block after it does not use `vp`, so nothing else changes.

In `src/neural_particle_method/implicit.py` add the same imports plus `from .simulate.leverage import DEFAULT_GRID, LeverageField`. In `_simulate`, replace the parameter line with `hp = HestonParams.from_dict(params)` and the four dynamics lines with `lnx, v = heston_step(lnx, v, L_p, zb, zp, hp, dt, sdt)`; the `out.append((t + dt, lnx.copy(), np.maximum(v, 0.0).copy()))` line stays. Replace `L = L_lookup(L_records)` with `L = LeverageField.from_records(L_records).at`. In `calibrate_implicit`, replace `grid = np.linspace(np.log(0.4), np.log(2.2), 81)` with `grid = DEFAULT_GRID.copy()`, and replace the `else:` branch that interpolates `L0_records` with:

```python
    else:
        L_records = LeverageField.from_records(L0_records).resample(grid).to_records()
```

In `src/neural_particle_method/reprice.py`, replace the body of `L_lookup` with:

```python
def L_lookup(L_records):
    """Piecewise-constant-in-t, interp-in-lnx leverage function from calibration records."""
    from .simulate.leverage import LeverageField
    return LeverageField.from_records(L_records).at
```

and in `reprice_iv` replace the parameter unpacking with `hp = HestonParams.from_dict(dynamics)` (keep `v0 = hp.v0`) and the four dynamics lines with `lnx, v = heston_step(lnx, v, L_p, zb, zp, hp, dt, sdt)`.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass, including the seven tiny goldens. If a golden fails here, the stepper is not verbatim; diff it against the original expressions before anything else.

- [ ] **Step 6: Run the archived goldens and commit**

Run: `uv run pytest -m golden -q` (minutes). Expected: 4 passed.

```bash
git add -A src/neural_particle_method tests/simulate
git commit -m "refactor: simulate subpackage; one Euler step and a typed LeverageField"
```

---

### Task 4: `estimators/` subpackage with one interface and one ridge readout

**Files:**
- Create: `src/neural_particle_method/estimators/__init__.py`, `base.py`, `nn.py`, `nadaraya_watson.py`, `spline.py`, `ridge.py`, `kalman.py`
- Modify: `src/neural_particle_method/condexp.py` (shim), `readouts.py` (shim), `implicit.py` (`GlobalRidgeHead` becomes a shim alias of `estimators.ridge.GlobalRidge`; `GlobalNet` stays here until Task 5)
- Modify: `src/neural_particle_method/explicit.py` (dispatch through estimator objects; RNG order untouched)
- Create: `tests/estimators/__init__.py`, `test_contract.py`, `test_ridge.py`, `test_kalman.py`, `test_spline.py`, `test_nw.py`
- Move: `tests/test_condexp.py` -> `tests/estimators/test_slice_ridge.py`

**Interfaces:**
- Produces:
  - `estimators.base.Estimator` Protocol: `supports_weights: bool`; `fit_predict(t, lnx, v, grid, weights=None) -> np.ndarray`.
  - `estimators.nn.SliceNet`, `V_SCALE`, `Z_SCALE`, `NNRegressor(seed=0, first_steps=400, later_steps=120)` (also keeps `fit(lnx, v, steps, weights)` and `predict(grid)` for direct use).
  - `estimators.nadaraya_watson.NadarayaWatson(bandwidth=None)` and function `nw_estimate` (moved).
  - `estimators.spline.PSpline(n_knots=25, lam=1.0, degree=3)` and function `spline_estimate` (moved); `supports_weights = False`.
  - `estimators.ridge.RidgeHead(features_fn, lam=1e-3, residual=True)` with `fit_predict(t, lnx, v, grid, weights=None)`, attribute `w_prev`, method `features(t, lnx)`.
  - `estimators.ridge.SliceRidge(seed=0, lam=1e-3, residual=True, body_steps=400)`: lazily trains its own `SliceNet` body on the first `fit_predict`; exposes `train_body(lnx, v, steps)`, `trained`, `mu`, `sd`, `w_prev`, `features(t, lnx)`.
  - `estimators.ridge.GlobalRidge(net, T, lam=1e-3, residual=True)`: features from a `GlobalNet` body at `(t / T, lnx / Z_SCALE)`; exposes `w_prev`, `features(t, lnx)`.
  - `estimators.kalman.KalmanHead` (moved verbatim; `_features` renamed `features`, old name kept as alias) and `bspline_basis`.
  - `estimators.make_estimator(name, seed=0, first_steps=400, later_steps=120)` for names `nn`, `nw`, `ridge`, `spline`.

- [ ] **Step 1: Write the contract test and the specific tests**

`tests/estimators/__init__.py` empty. `tests/estimators/test_contract.py`:

```python
import numpy as np
import pytest
import torch

from neural_particle_method.estimators import make_estimator
from neural_particle_method.estimators.ridge import GlobalRidge
from neural_particle_method.implicit import GlobalNet

RNG = np.random.default_rng(0)
LNX = RNG.normal(0.0, 0.2, 3_000)
V = 0.04 + 0.01 * LNX + RNG.normal(0.0, 0.002, 3_000)
GRID = np.linspace(-0.4, 0.4, 17)


def _global_net():
    torch.manual_seed(0)
    return GlobalNet()


def _all():
    return [make_estimator("nn", seed=0, first_steps=40, later_steps=20),
            make_estimator("nw"),
            make_estimator("ridge", seed=0, first_steps=40),
            make_estimator("spline"),
            GlobalRidge(_global_net(), T=1.0)]


@pytest.mark.parametrize("est", _all(), ids=lambda e: type(e).__name__)
def test_shape_finite_positive(est):
    f = est.fit_predict(0.5, LNX, V, GRID)
    assert f.shape == GRID.shape and np.all(np.isfinite(f))
    if type(est).__name__ in ("NNRegressor", "NadarayaWatson"):   # softplus output / positive data
        assert np.all(f > 0)


@pytest.mark.parametrize("est", [e for e in _all() if e.supports_weights], ids=lambda e: type(e).__name__)
def test_unit_weights_equal_no_weights(est):
    a = est.fit_predict(0.5, LNX, V, GRID)
    # a fresh copy of the same estimator, same seed, weighted with ones
    est2 = type(est)(**_ctor_kwargs(est))
    b = est2.fit_predict(0.5, LNX, V, GRID, weights=np.ones(len(LNX)))
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9)


def _ctor_kwargs(est):
    name = type(est).__name__
    if name == "NNRegressor":
        return dict(seed=0, first_steps=40, later_steps=20)
    if name == "SliceRidge":
        return dict(seed=0, body_steps=40)
    if name == "GlobalRidge":
        return dict(net=_global_net(), T=1.0)
    return {}


def test_spline_rejects_weights():
    with pytest.raises(ValueError, match="importance weights"):
        make_estimator("spline").fit_predict(0.5, LNX, V, GRID, weights=np.ones(len(LNX)))
```

`make_estimator` maps `first_steps` to `body_steps` for ridge exactly as the old `calibrate_explicit` did (`train_body(..., steps=first_steps)`).

`tests/estimators/test_nw.py`:

```python
import numpy as np

from neural_particle_method.estimators.nadaraya_watson import NadarayaWatson, nw_estimate


def test_constant_data_gives_constant_estimate():
    lnx = np.random.default_rng(1).normal(0, 0.3, 500)
    v = np.full(500, 0.05)
    f = NadarayaWatson().fit_predict(0.1, lnx, v, np.linspace(-0.5, 0.5, 7))
    assert np.allclose(f, 0.05)


def test_class_matches_function():
    rng = np.random.default_rng(2)
    lnx, v, g = rng.normal(0, 0.3, 400), rng.uniform(0.01, 0.09, 400), np.linspace(-0.3, 0.3, 5)
    np.testing.assert_array_equal(NadarayaWatson().fit_predict(0.0, lnx, v, g), nw_estimate(lnx, v, g))
```

`tests/estimators/test_spline.py`:

```python
import numpy as np

from neural_particle_method.estimators.spline import PSpline


def test_recovers_quadratic():
    rng = np.random.default_rng(3)
    lnx = rng.uniform(-0.5, 0.5, 5_000)
    v = 0.04 + 0.1 * lnx ** 2
    g = np.linspace(-0.3, 0.3, 13)
    f = PSpline(lam=1e-3).fit_predict(0.0, lnx, v, g)
    assert np.abs(f - (0.04 + 0.1 * g ** 2)).max() < 1e-3
```

`tests/estimators/test_ridge.py`:

```python
import numpy as np

from neural_particle_method.estimators.ridge import RidgeHead


def _feat(t, lnx):
    return np.stack([lnx, lnx ** 2, np.ones_like(lnx)], axis=1)


def test_ridge_solution_matches_closed_form_without_residual():
    rng = np.random.default_rng(0)
    lnx = rng.normal(0, 0.3, 2_000)
    v = 0.04 + 0.02 * lnx + rng.normal(0, 1e-3, 2_000)
    head = RidgeHead(_feat, lam=1e-3, residual=False)
    g = np.linspace(-0.2, 0.2, 5)
    f = head.fit_predict(0.0, lnx, v, g)
    A = _feat(0.0, lnx)
    lam = 1e-3 * len(lnx)
    w = np.linalg.solve(A.T @ A + lam * np.eye(3), A.T @ v)
    np.testing.assert_allclose(f, _feat(0.0, g) @ w, rtol=1e-10)
    np.testing.assert_allclose(head.w_prev, w, rtol=1e-10)


def test_residual_centres_on_previous_coefficients():
    rng = np.random.default_rng(1)
    lnx = rng.normal(0, 0.3, 1_000)
    v = np.full(1_000, 0.05)
    head = RidgeHead(_feat, lam=10.0, residual=True)
    head.fit_predict(0.0, lnx, v, lnx[:3])
    w1 = head.w_prev.copy()
    head.fit_predict(0.1, lnx, v + 0.01, lnx[:3])
    # with a huge lam the second solve barely moves from w1 (shrinks toward it, not toward 0)
    assert np.abs(head.w_prev - w1).max() < 0.02
```

`tests/estimators/test_kalman.py`:

```python
import numpy as np
import torch

from neural_particle_method.estimators.kalman import KalmanHead, bspline_basis
from neural_particle_method.implicit import GlobalNet


def _head(**kw):
    torch.manual_seed(0)
    return KalmanHead(GlobalNet(), T=1.0, **kw)


def test_zero_prior_reproduces_least_squares():
    rng = np.random.default_rng(0)
    lnx = rng.normal(0, 0.2, 2_000)
    v = 0.04 + 0.01 * lnx + rng.normal(0, 1e-3, 2_000)
    head = _head(n_prior=0, n_eff=1_000)
    g = np.linspace(-0.2, 0.2, 5)
    f = head.update(1, 0.5, lnx, v, g)
    A = head.features(0.5, lnx)
    w = np.linalg.lstsq(A, v, rcond=None)[0]
    np.testing.assert_allclose(f, head.features(0.5, g) @ w, rtol=1e-3, atol=1e-5)


def test_infinite_prior_leaves_beta_unchanged():
    rng = np.random.default_rng(1)
    lnx = rng.normal(0, 0.2, 1_000)
    v = np.full(1_000, 0.05)
    head = _head(n_prior=1e12, n_eff=1)
    beta0 = np.linspace(0.1, 0.2, head.features(0.5, lnx).shape[1])
    head.init_from({1: beta0})
    head.update(1, 0.5, lnx, v, lnx[:2])
    np.testing.assert_allclose(head.betas()[1], beta0, rtol=1e-8)


def test_bspline_basis_partition_of_unity():
    B = bspline_basis(np.linspace(-0.5, 0.5, 50), -0.6, 0.6, 10)
    assert B.shape == (50, 10) and np.allclose(B.sum(axis=1), 1.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/estimators -q`
Expected: ImportError on `neural_particle_method.estimators`.

- [ ] **Step 3: Implement the subpackage**

`estimators/base.py`:

```python
"""Conditional-expectation estimator interface: E[V | ln X = .] on a grid, one slice at a time."""
from typing import Protocol

import numpy as np


class Estimator(Protocol):
    supports_weights: bool

    def fit_predict(self, t: float, lnx: np.ndarray, v: np.ndarray, grid: np.ndarray,
                    weights: np.ndarray | None = None) -> np.ndarray: ...
```

`estimators/nn.py` (moved from `condexp.py`: `V_SCALE`, `Z_SCALE`, `SliceNet`, `NNRegressor` verbatim, then add):

```python
class NNRegressor:
    """Warm-startable per-slice regressor for E[V | ln X = .]."""
    supports_weights = True

    def __init__(self, seed=0, first_steps=400, later_steps=120):
        torch.manual_seed(seed)
        self.net = SliceNet()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-2)
        self.first_steps, self.later_steps = first_steps, later_steps
        self._n_fits = 0

    def fit(self, lnx, v, steps=120, weights=None):
        z = torch.tensor(lnx[:, None] / Z_SCALE, dtype=torch.float32)
        tv = torch.tensor(v[:, None], dtype=torch.float32)
        tw = None if weights is None else torch.tensor(weights[:, None], dtype=torch.float32)
        for _ in range(steps):
            self.opt.zero_grad()
            resid = (self.net(z) - tv) ** 2
            loss = (resid if tw is None else tw * resid).mean()
            loss.backward()
            self.opt.step()
        return float(loss.detach())

    def predict(self, lnx_grid):
        with torch.no_grad():
            z = torch.tensor(lnx_grid[:, None] / Z_SCALE, dtype=torch.float32)
            return self.net(z).numpy()[:, 0]

    def fit_predict(self, t, lnx, v, grid, weights=None):
        steps = self.first_steps if self._n_fits == 0 else self.later_steps
        self._n_fits += 1
        self.fit(lnx, v, steps=steps, weights=weights)
        return self.predict(grid)
```

The old loop used `first_steps if k == 1 else later_steps`; the first fit always happens at k = 1, so a fit counter is equivalent.

`estimators/nadaraya_watson.py`: move `nw_estimate` verbatim, add:

```python
class NadarayaWatson:
    supports_weights = True

    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth

    def fit_predict(self, t, lnx, v, grid, weights=None):
        return nw_estimate(lnx, v, grid, weights=weights, bandwidth=self.bandwidth)
```

`estimators/spline.py`: move `spline_estimate` verbatim, add:

```python
class PSpline:
    supports_weights = False

    def __init__(self, n_knots=25, lam=1.0, degree=3):
        self.n_knots, self.lam, self.degree = n_knots, lam, degree

    def fit_predict(self, t, lnx, v, grid, weights=None):
        if weights is not None:
            raise ValueError("spline estimator does not support importance weights")
        return spline_estimate(lnx, v, grid, n_knots=self.n_knots, lam=self.lam, degree=self.degree)
```

`estimators/ridge.py`:

```python
"""Ridge readouts on frozen learned features, optionally centred on the previous slice's coefficients."""
import numpy as np
import torch

from .nn import Z_SCALE, SliceNet


class RidgeHead:
    """Per-slice ridge on features A = features_fn(t, lnx). Residual mode shrinks toward w_prev."""
    supports_weights = True

    def __init__(self, features_fn, lam=1e-3, residual=True):
        self._features_fn = features_fn
        self.lam, self.residual = lam, residual
        self.w_prev = None

    def features(self, t, lnx):
        return self._features_fn(t, lnx)

    def fit_predict(self, t, lnx, v, grid, weights=None):
        A = self.features(t, lnx)
        lam = self.lam * len(lnx)
        wv = np.ones(len(lnx)) if weights is None else weights
        lhs = A.T @ (wv[:, None] * A) + lam * np.eye(A.shape[1])
        rhs = A.T @ (wv * v)
        if self.residual and self.w_prev is not None:
            rhs = rhs + lam * self.w_prev
        w = np.linalg.solve(lhs, rhs)
        self.w_prev = w
        return self.features(t, grid) @ w


class SliceRidge(RidgeHead):
    """Frozen per-slice body (trained once on the first slice) + ridge readout. Ignores t."""

    def __init__(self, seed=0, lam=1e-3, residual=True, body_steps=400):
        torch.manual_seed(seed)
        self.net = SliceNet()
        self.body_steps = body_steps
        self.trained = False
        self.mu, self.sd = None, None
        super().__init__(self._slice_features, lam=lam, residual=residual)

    def train_body(self, lnx, v, steps=400):
        self.mu, self.sd = lnx.mean(), max(lnx.std(), 1e-6)
        z = torch.tensor(((lnx - self.mu) / self.sd)[:, None], dtype=torch.float32)
        tv = torch.tensor(v[:, None], dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-2)
        for _ in range(steps):
            opt.zero_grad()
            loss = ((self.net(z) - tv) ** 2).mean()
            loss.backward()
            opt.step()
        self.trained = True

    def _slice_features(self, t, lnx):
        with torch.no_grad():
            z = torch.tensor(((lnx - self.mu) / self.sd)[:, None], dtype=torch.float32)
            phi = self.net.body(z).numpy()
        return np.concatenate([phi, np.ones((len(lnx), 1))], axis=1)

    def fit_predict(self, t, lnx, v, grid, weights=None):
        if not self.trained:
            self.train_body(lnx, v, steps=self.body_steps)
        return super().fit_predict(t, lnx, v, grid, weights)


class GlobalRidge(RidgeHead):
    """Frozen implicit-scheme body as a (t, x) feature extractor with a per-slice ridge readout."""

    def __init__(self, net, T, lam=1e-3, residual=True):
        self.net, self.T = net, T
        super().__init__(self._global_features, lam=lam, residual=residual)

    def _global_features(self, t, lnx):
        with torch.no_grad():
            tz = torch.tensor(np.stack([np.full(len(lnx), t / max(self.T, 1e-9)),
                                        lnx / Z_SCALE], axis=1), dtype=torch.float32)
            phi = self.net.body(tz).numpy()
        return np.concatenate([phi, np.ones((len(lnx), 1))], axis=1)
```

Check against the originals: `condexp.RidgeHead.fit_predict` and `implicit.GlobalRidgeHead.fit_predict` both compute `lam = self.lam * len(lnx)`, the same `lhs`, `rhs`, residual shift, and solve. The old per-slice `train_body` is verbatim. Old code called `train_body` from `calibrate_explicit` with `steps=first_steps` before the first `fit_predict`, which is what the lazy call reproduces.

`estimators/kalman.py`: move `readouts.py` content verbatim, rename `_features` to `features`, add `_features = features` alias line inside the class for the experiments code that still calls it, and import `Z_SCALE` from `.nn`.

`estimators/__init__.py`:

```python
"""Conditional-expectation estimators sharing the Estimator interface."""
from .base import Estimator
from .kalman import KalmanHead
from .nadaraya_watson import NadarayaWatson
from .nn import NNRegressor
from .ridge import GlobalRidge, RidgeHead, SliceRidge
from .spline import PSpline

NAMES = ("nn", "nw", "ridge", "spline")


def make_estimator(name, seed=0, first_steps=400, later_steps=120):
    if name == "nn":
        return NNRegressor(seed=seed, first_steps=first_steps, later_steps=later_steps)
    if name == "nw":
        return NadarayaWatson()
    if name == "ridge":
        return SliceRidge(seed=seed, body_steps=first_steps)
    if name == "spline":
        return PSpline()
    raise KeyError(f"unknown estimator {name!r}; choose from {NAMES}")


__all__ = ["Estimator", "GlobalRidge", "KalmanHead", "NNRegressor", "NadarayaWatson", "PSpline",
           "RidgeHead", "SliceRidge", "make_estimator", "NAMES"]
```

Shims: `condexp.py` becomes

```python
from .estimators.nn import V_SCALE, Z_SCALE, NNRegressor, SliceNet  # noqa: F401
from .estimators.nadaraya_watson import nw_estimate  # noqa: F401
from .estimators.spline import spline_estimate  # noqa: F401
from .estimators.ridge import SliceRidge as RidgeHead  # noqa: F401
```

`readouts.py` becomes `from .estimators.kalman import KalmanHead, bspline_basis  # noqa: F401`. In `implicit.py`, delete the `GlobalRidgeHead` class and add `from .estimators.ridge import GlobalRidge as GlobalRidgeHead  # noqa: F401` plus `from .estimators.nn import V_SCALE, Z_SCALE`.

- [ ] **Step 4: Route `calibrate_explicit` through estimator objects**

In `explicit.py`, replace the `reg = None if regressor ...` line with:

```python
    if regressor is not None:
        est = regressor
    else:
        est = make_estimator(method, seed=seed, first_steps=first_steps, later_steps=later_steps)
    if mixture is not None and not est.supports_weights:
        raise ValueError(f"{type(est).__name__} estimator does not support importance weights")
```

and delete the old `if method == "spline" and mixture is not None: raise ...` check at the top (the new one replaces it and keeps the message substring `importance weights`). Replace the whole `if regressor is not None: ... else: f_grid = nw_estimate(...)` chain with one line:

```python
            f_grid = est.fit_predict(t, lnx[idx], v[idx], grid, weights=wi)
```

Keep `idx = rng.choice(...)`, `wi`, the `t0` timer, and `fit_s` accumulation exactly where they are. Import `from .estimators import make_estimator`.

Construction order check: the old code built `NNRegressor(seed)` or `RidgeHead(seed)` (each calling `torch.manual_seed`) before the loop and after `rng = default_rng(seed)`; `make_estimator` is called at the same point, so torch seeding order is unchanged.

- [ ] **Step 5: Move the old condexp test**

`git mv tests/test_condexp.py tests/estimators/test_slice_ridge.py`; change the import to `from neural_particle_method.estimators.ridge import SliceRidge as RidgeHead` and replace the two `reg._features(x, mu0, sd0)` calls with `reg.features(0.0, x)` (the body's `mu`/`sd` are now read from the object, which is exactly what the test asserts is fixed).

- [ ] **Step 6: Run the whole suite, then the goldens**

Run: `uv run pytest -q`. Expected: all pass including the seven tiny goldens.
Run: `uv run pytest -m golden -q`. Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add -A src/neural_particle_method tests
git commit -m "refactor: estimators subpackage with one interface and one ridge readout"
```

---

### Task 5: `calibrate/` subpackage: configs, results, estimator injection, warm helpers

**Files:**
- Create: `src/neural_particle_method/calibrate/__init__.py`, `config.py`, `explicit.py`, `implicit.py`, `importance.py`, `warm.py`
- Modify: old `explicit.py`, `implicit.py`, `importance.py` become shims that adapt the OLD signatures to the new ones (so `bench/` and tests still work until Task 8)
- Modify: `bench/algos.py` to call the new API directly
- Create: `tests/calibrate/__init__.py`, `test_config.py`, `test_warm.py`
- Move: `tests/test_implicit.py`, `tests/test_implicit_ridge.py`, `tests/test_importance.py` into `tests/calibrate/` and port to the new API
- Modify: `tests/conftest.py` (add dataclass tiny configs), `tests/test_golden_tiny.py` (helper uses new `run_algo` signature)

**Interfaces:**
- Produces:
  - `calibrate.config.ExplicitConfig(n_steps=50, n_particles=200_000, fit_subsample=30_000, L_max=4.0, first_steps=400, later_steps=120, snapshot_times=())`, `ImplicitConfig(n_steps=50, n_particles=50_000, alpha=0.5, n_iters=6, L_max=4.0, fit_steps=300, pool_subsample=60_000)`. Both frozen, both have `as_params()` returning a flat `{str: str|int|float}` for logging.
  - `calibrate.explicit.ExplicitResult(lnx, field, fit_s, snapshots, snapshot_weights, weights, is_diag)` and `calibrate_explicit(local_vol, params, estimator, cfg=ExplicitConfig(), *, s0=1.0, T=1.0, seed=0, mixture=None) -> ExplicitResult`.
  - `calibrate.implicit.GlobalNet`, `simulate_slices(field, params, s0, T, n_steps, n_particles, rng) -> list[(t, lnx, v)]`, `ImplicitResult(field, deltas, fit_s, net)`, `calibrate_implicit(local_vol, params, cfg=ImplicitConfig(), *, s0=1.0, T=1.0, seed=0, L0=None) -> ImplicitResult`.
  - `calibrate.importance.MixtureDesign`, `design_mixture(params, T, ...)` (accepts `HestonParams` or dict).
  - `calibrate.warm`: `ALPHA = 0.5`, `bump(p)`, `scaled_bump(p, s)`, `seq_path(p, n, rng)`, `dyn_variants(params) -> dict[str, HestonParams]`, `distil(net, slices, lv, T, n_steps, s0, v0, sub, rng) -> (betas, field)`, `records_from_betas(net, betas, lv, T, n_steps, s0, v0) -> LeverageField`, `records_from_head(head, betas, lv, T, n_steps, s0, v0) -> LeverageField`, `beta_correction(net, betas, lv, params, T, n_steps, N, s0, v0, sub, seed) -> dict`, `kalman_pass(head, betas, lv, params, T, n_steps, N, s0, v0, sub, seed) -> dict`, `full_resolve(lv, params, s0, T, cfg: ImplicitConfig, seed, L0=None, n_iters=None) -> (LeverageField, seconds)` where the explicit warm start inside uses `ExplicitConfig(n_steps=cfg.n_steps, n_particles=cfg.n_particles)`.
  - `bench.algos.run_algo(name, scenario, n_particles, seed, explicit=ExplicitConfig(), implicit=ImplicitConfig()) -> CalibResult(field, timings, diagnostics)`.
  - `tests/conftest.py::TINY_EXPLICIT = ExplicitConfig(n_steps=6, fit_subsample=4_000, first_steps=80, later_steps=30)`, `TINY_IMPLICIT = ImplicitConfig(n_steps=6, n_iters=2, fit_steps=80)`.

- [ ] **Step 1: Write the config and warm tests**

`tests/calibrate/__init__.py` empty. `tests/calibrate/test_config.py`:

```python
import dataclasses

from neural_particle_method.calibrate.config import ExplicitConfig, ImplicitConfig


def test_defaults_match_pre_refactor_values():
    e, i = ExplicitConfig(), ImplicitConfig()
    assert (e.n_steps, e.n_particles, e.fit_subsample, e.L_max, e.first_steps, e.later_steps) == \
        (50, 200_000, 30_000, 4.0, 400, 120)
    assert (i.n_steps, i.n_particles, i.alpha, i.n_iters, i.L_max, i.fit_steps, i.pool_subsample) == \
        (50, 50_000, 0.5, 6, 4.0, 300, 60_000)


def test_frozen_and_replaceable():
    e = dataclasses.replace(ExplicitConfig(), n_steps=6)
    assert e.n_steps == 6
    try:
        e.n_steps = 7
        assert False
    except dataclasses.FrozenInstanceError:
        pass


def test_as_params_is_flat_and_stringifiable():
    p = ExplicitConfig(snapshot_times=(0.25, 0.5)).as_params()
    assert p["snapshot_times"] == "(0.25, 0.5)" and p["n_steps"] == 50
    assert all(isinstance(v, (str, int, float)) for v in p.values())
```

`tests/calibrate/test_warm.py`:

```python
import numpy as np
import torch

from neural_particle_method.calibrate.implicit import GlobalNet, simulate_slices
from neural_particle_method.calibrate.warm import (bump, distil, dyn_variants, records_from_betas,
                                                    scaled_bump, seq_path)
from neural_particle_method.market.local_vol import SSVILocalVol
from neural_particle_method.market.ssvi import SSVIParams, no_arb_ok
from neural_particle_method.simulate.dynamics import HestonParams
from neural_particle_method.simulate.leverage import DEFAULT_GRID, LeverageField, Slice

P = SSVIParams(sigma0=0.2, eta=1.0, gamma=0.4, rho=-0.6)
DYN = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.6, v0=0.04)


def test_bumps_stay_arbitrage_free():
    assert no_arb_ok(bump(P))
    q, eff = scaled_bump(P, 2.0)
    assert no_arb_ok(q) and 0 < eff <= 2.0
    path = seq_path(P, 4, np.random.default_rng(0))
    assert len(path) == 4 and all(no_arb_ok(q) for q in path)


def test_dyn_variants_returns_heston_params():
    d = dyn_variants(DYN)
    assert set(d) == {"xi_up", "rho_dn", "kappa_dn"}
    assert d["xi_up"].xi == DYN.xi * 1.3 and d["kappa_dn"].kappa == DYN.kappa * 0.6


def test_distil_then_records_round_trip():
    torch.manual_seed(0)
    net = GlobalNet()
    lv = SSVILocalVol(P, T_max=0.5)
    n_steps, T = 4, 0.5
    flat = LeverageField([Slice(k * T / n_steps, DEFAULT_GRID, np.ones(len(DEFAULT_GRID)),
                                np.full(len(DEFAULT_GRID), 0.04)) for k in range(n_steps)])
    rng = np.random.default_rng(0)
    slices = simulate_slices(flat, DYN, 1.0, T, n_steps, 2_000, rng)
    betas, field = distil(net, slices, lv, T, n_steps, 1.0, DYN.v0, 1_000, rng)
    assert set(betas) == {1, 2, 3} and len(field) == n_steps
    rebuilt = records_from_betas(net, betas, lv, T, n_steps, 1.0, DYN.v0)
    for a, b in zip(field, rebuilt):
        np.testing.assert_array_equal(a.L, b.L)
        np.testing.assert_array_equal(a.f, b.f)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/calibrate -q`. Expected: ImportError on `neural_particle_method.calibrate`.

- [ ] **Step 3: Implement configs and importance**

`calibrate/__init__.py`:

```python
"""Calibration schemes: explicit per-slice pass, implicit damped iteration, mixture importance sampling."""
```

`calibrate/config.py`:

```python
"""Frozen configurations. Defaults are the pre-refactor defaults; do not change them."""
from dataclasses import asdict, dataclass


def _flat(d):
    return {k: (str(v) if isinstance(v, (tuple, list)) else v) for k, v in d.items()}


@dataclass(frozen=True)
class ExplicitConfig:
    n_steps: int = 50
    n_particles: int = 200_000
    fit_subsample: int = 30_000
    L_max: float = 4.0
    first_steps: int = 400
    later_steps: int = 120
    snapshot_times: tuple = ()

    def as_params(self):
        return _flat(asdict(self))


@dataclass(frozen=True)
class ImplicitConfig:
    n_steps: int = 50
    n_particles: int = 50_000
    alpha: float = 0.5
    n_iters: int = 6
    L_max: float = 4.0
    fit_steps: int = 300
    pool_subsample: int = 60_000

    def as_params(self):
        return _flat(asdict(self))
```

`calibrate/importance.py`: `git mv` from the old path. Change `design_mixture` so its first line is `p = HestonParams.from_dict(dynamics)` and it reads `p.rho`, `p.v0`, `p.theta`. Everything else verbatim. Old `importance.py` shim: `from .calibrate.importance import MixtureDesign, design_mixture  # noqa: F401`.

- [ ] **Step 4: Implement `calibrate/explicit.py`**

```python
"""Explicit (per-slice) neural particle calibration of a Heston-type LSV model."""
import time
from dataclasses import dataclass

import numpy as np

from ..simulate.dynamics import HestonParams
from ..simulate.leverage import LeverageField, Slice
from ..simulate.stepper import heston_step
from .config import ExplicitConfig


@dataclass
class ExplicitResult:
    lnx: np.ndarray
    field: LeverageField
    fit_s: float
    snapshots: dict
    snapshot_weights: dict
    weights: np.ndarray | None
    is_diag: dict | None


def calibrate_explicit(local_vol, params, estimator, cfg=ExplicitConfig(), *,
                       s0=1.0, T=1.0, seed=0, mixture=None):
    """Single forward pass. Under a mixture the returned lnx and snapshots are proposal-distributed;
    use `weights` / `snapshot_weights` to recover physical-measure statistics."""
    if mixture is not None and not estimator.supports_weights:
        raise ValueError(f"{type(estimator).__name__} estimator does not support importance weights")
    hp = HestonParams.from_dict(params)
    v0 = hp.v0
    n_steps, n_particles, fit_subsample, L_max = cfg.n_steps, cfg.n_particles, cfg.fit_subsample, cfg.L_max
    rng = np.random.default_rng(seed)
    dt, sdt = T / n_steps, np.sqrt(T / n_steps)

    lnx = np.full(n_particles, np.log(s0))
    v = np.full(n_particles, v0)

    slices, snapshots, snapshot_weights = [], {}, {}
    snap_steps = {int(round(t / dt)): t for t in cfg.snapshot_times}

    fit_s = 0.0
    theta_p = None
    if mixture is not None:
        comp = rng.choice(3, size=n_particles, p=list(mixture.alphas))
        theta_p = np.array(mixture.thetas)[comp]
        etas = np.array(mixture.etas)
        eta_p = etas[comp]
        ell = np.zeros((3, n_particles))
    w = None

    for k in range(n_steps):
        t = k * dt
        qs = np.linspace(0.001, 0.999, 101)
        grid = np.quantile(lnx, qs) if k > 0 else np.array([np.log(s0)])
        grid = np.unique(grid)
        if k == 0:
            f_grid = np.full(len(grid), v0)
        else:
            idx = rng.choice(n_particles, size=min(fit_subsample, n_particles), replace=False)
            wi = None if w is None else w[idx]
            t0 = time.perf_counter()
            f_grid = estimator.fit_predict(t, lnx[idx], v[idx], grid, weights=wi)
            fit_s += time.perf_counter() - t0
        f_grid = np.clip(f_grid, 1e-4, None)
        sig = local_vol.sigma(max(t, local_vol.T_grid[0]), np.exp(grid), s0)
        L_grid = np.clip(sig / np.sqrt(f_grid), 0.0, L_max)
        slices.append(Slice(t, grid.copy(), L_grid.copy(), f_grid.copy()))

        L_p = np.interp(lnx, grid, L_grid) if len(grid) > 1 else np.full(n_particles, L_grid[0])
        zb = rng.standard_normal(n_particles)
        zp = rng.standard_normal(n_particles)
        lnx, v = heston_step(lnx, v, L_p, zb, zp, hp, dt, sdt, theta_p=theta_p)
        if mixture is not None:
            dbperp = zp * sdt + eta_p * dt
            ell += etas[:, None] * dbperp[None, :] - 0.5 * (etas ** 2)[:, None] * dt
            w = 1.0 / (np.array(mixture.alphas) @ np.exp(np.clip(ell, -60, 60)))
        if k + 1 in snap_steps:
            snapshots[snap_steps[k + 1]] = lnx.copy()
            if mixture is not None:
                snapshot_weights[snap_steps[k + 1]] = w.copy()

    is_diag = None
    if mixture is not None:
        is_diag = {"max_w": float(w.max()),
                   "ess_frac": float(w.sum() ** 2 / (len(w) * (w ** 2).sum()))}
    return ExplicitResult(lnx, LeverageField(slices), fit_s, snapshots, snapshot_weights, w, is_diag)
```

Compare line by line with the old body after Task 4; only the packaging of inputs and outputs differs. `mc_smile` moves to `pricing/reprice.py` in Task 6; until then it stays in the old `explicit.py` shim.

Old `explicit.py` becomes a shim that keeps the OLD signature for Task 5's transitional callers:

```python
"""Shim (removed in Task 12): old calibrate_explicit signature over the new implementation."""
import numpy as np

from .calibrate.config import ExplicitConfig
from .calibrate.explicit import calibrate_explicit as _new
from .estimators import make_estimator


def calibrate_explicit(dupire, params, s0=1.0, T=1.0, n_steps=50, n_particles=200_000,
                       method="nn", fit_subsample=30_000, seed=0, L_max=4.0,
                       first_steps=400, later_steps=120, snapshot_times=(), mixture=None, regressor=None):
    est = regressor if regressor is not None else make_estimator(method, seed=seed, first_steps=first_steps,
                                                                    later_steps=later_steps)
    cfg = ExplicitConfig(n_steps=n_steps, n_particles=n_particles, fit_subsample=fit_subsample,
                         L_max=L_max, first_steps=first_steps, later_steps=later_steps,
                         snapshot_times=tuple(snapshot_times))
    r = _new(dupire, params, est, cfg, s0=s0, T=T, seed=seed, mixture=mixture)
    info = {"L_records": r.field.to_records(), "snapshots": r.snapshots, "fit_s": r.fit_s}
    if mixture is not None:
        info.update(weights=r.weights, snapshot_weights=r.snapshot_weights, is_diag=r.is_diag)
    return r.lnx, info


def mc_smile(lnx_T, K_grid, s0=1.0, weights=None):
    x = np.exp(lnx_T)
    if weights is None:
        return np.array([np.mean(np.maximum(x - K, 0.0)) for K in K_grid])
    wn = weights / weights.sum()
    return np.array([np.sum(wn * np.maximum(x - K, 0.0)) for K in K_grid])
```

- [ ] **Step 5: Implement `calibrate/implicit.py`**

```python
"""Implicit scheme: global network over (t, x) with damped leverage iteration."""
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..estimators.nn import V_SCALE, Z_SCALE
from ..simulate.dynamics import HestonParams
from ..simulate.leverage import DEFAULT_GRID, LeverageField, Slice
from ..simulate.stepper import heston_step
from .config import ImplicitConfig


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


def simulate_slices(field, params, s0, T, n_steps, n_particles, rng):
    """Forward Euler under a fixed leverage; returns pooled (t, lnx, v) slices."""
    hp = HestonParams.from_dict(params)
    field = LeverageField.from_records(field)
    dt, sdt = T / n_steps, np.sqrt(T / n_steps)
    lnx = np.full(n_particles, np.log(s0))
    v = np.full(n_particles, hp.v0)
    out = []
    for k in range(n_steps):
        t = k * dt
        L_p = field.at(t, lnx)
        zb = rng.standard_normal(n_particles)
        zp = rng.standard_normal(n_particles)
        lnx, v = heston_step(lnx, v, L_p, zb, zp, hp, dt, sdt)
        out.append((t + dt, lnx.copy(), np.maximum(v, 0.0).copy()))
    return out


@dataclass
class ImplicitResult:
    field: LeverageField
    deltas: list
    fit_s: float
    net: GlobalNet


def calibrate_implicit(local_vol, params, cfg=ImplicitConfig(), *, s0=1.0, T=1.0, seed=0, L0=None):
    hp = HestonParams.from_dict(params)
    n_steps, n_particles, alpha, L_max = cfg.n_steps, cfg.n_particles, cfg.alpha, cfg.L_max
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    grid = DEFAULT_GRID.copy()
    dt = T / n_steps
    v0 = hp.v0
    if L0 is None:
        sig0 = local_vol.sigma(local_vol.T_grid[0], np.exp(grid), s0)
        L0g = np.clip(sig0 / np.sqrt(v0), 0.0, L_max)
        field = LeverageField([Slice(k * dt, grid.copy(), L0g.copy(), np.full(len(grid), v0))
                               for k in range(n_steps)])
    else:
        field = LeverageField.from_records(L0).resample(grid)
    net = GlobalNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    deltas, fit_s = [], 0.0
    for _ in range(cfg.n_iters):
        slices = simulate_slices(field, hp, s0, T, n_steps, n_particles, rng)
        ts = np.concatenate([np.full(len(x), t) for t, x, _ in slices])
        xs = np.concatenate([x for _, x, _ in slices])
        vs = np.concatenate([v for _, _, v in slices])
        idx = rng.choice(len(ts), size=min(cfg.pool_subsample, len(ts)), replace=False)
        tz = torch.tensor(np.stack([ts[idx] / max(T, 1e-9), xs[idx] / Z_SCALE], axis=1),
                          dtype=torch.float32)
        tv = torch.tensor(vs[idx][:, None], dtype=torch.float32)
        t0 = time.perf_counter()
        for _ in range(cfg.fit_steps):
            opt.zero_grad()
            loss = ((net(tz) - tv) ** 2).mean()
            loss.backward()
            opt.step()
        fit_s += time.perf_counter() - t0
        new_slices, sup = [], 0.0
        with torch.no_grad():
            for s in field:
                t, g, Lg = s.t, s.grid, s.L
                tzg = torch.tensor(np.stack([np.full(len(g), t / max(T, 1e-9)),
                                             g / Z_SCALE], axis=1), dtype=torch.float32)
                f = np.clip(net(tzg).numpy()[:, 0], 1e-4, None)
                sig = local_vol.sigma(max(t, local_vol.T_grid[0]), np.exp(g), s0)
                Phi = np.clip(sig / np.sqrt(f), 0.0, L_max)
                L_new = (1 - alpha) * Lg + alpha * Phi
                sup = max(sup, float(np.abs(L_new - Lg).max()))
                new_slices.append(Slice(t, g, L_new, f))
        field = LeverageField(new_slices)
        deltas.append(sup)
    return ImplicitResult(field, deltas, fit_s, net)
```

Old `implicit.py` shim keeps the old signature:

```python
from .calibrate.config import ImplicitConfig
from .calibrate.implicit import GlobalNet, calibrate_implicit as _new, simulate_slices  # noqa: F401
from .estimators.ridge import GlobalRidge as GlobalRidgeHead  # noqa: F401


def _simulate(L_records, params, s0, T, n_steps, n_particles, rng):
    return simulate_slices(L_records, params, s0, T, n_steps, n_particles, rng)


def calibrate_implicit(dupire, params, s0=1.0, T=1.0, n_steps=50, n_particles=50_000,
                       alpha=0.5, n_iters=6, seed=0, L_max=4.0, fit_steps=300,
                       pool_subsample=60_000, L0_records=None):
    cfg = ImplicitConfig(n_steps=n_steps, n_particles=n_particles, alpha=alpha, n_iters=n_iters,
                         L_max=L_max, fit_steps=fit_steps, pool_subsample=pool_subsample)
    r = _new(dupire, params, cfg, s0=s0, T=T, seed=seed, L0=L0_records)
    return r.field.to_records(), {"deltas": r.deltas, "fit_s": r.fit_s, "net": r.net}
```

- [ ] **Step 6: Implement `calibrate/warm.py`**

Move the helpers from `experiments/bump_correct.py` and `experiments/warm_suite.py`. Each body is verbatim except: `GlobalRidgeHead` becomes `GlobalRidge`, `head._features` becomes `head.features`, `_simulate` becomes `simulate_slices`, record lists become `LeverageField`, `dyn` dicts become `HestonParams`, and `full_resolve` takes an `ImplicitConfig`.

```python
"""Warm-start helpers: surface bumps, body distillation, beta-space corrections, Kalman passes."""
import dataclasses
import time

import numpy as np

from ..calibrate.config import ExplicitConfig, ImplicitConfig
from ..calibrate.explicit import calibrate_explicit
from ..calibrate.implicit import calibrate_implicit, simulate_slices
from ..estimators import make_estimator
from ..estimators.ridge import GlobalRidge
from ..market.ssvi import SSVIParams, no_arb_ok
from ..simulate.dynamics import HestonParams
from ..simulate.leverage import DEFAULT_GRID as GRID, LeverageField, Slice

ALPHA = 0.5


def bump(p):
    q = SSVIParams(sigma0=p.sigma0 + 0.01, eta=0.95 * p.eta, gamma=p.gamma,
                   rho=min(p.rho + 0.03, -0.05))
    assert no_arb_ok(q)
    return q


def scaled_bump(p, s):
    for shrink in (1.0, 0.75, 0.5, 0.25):
        eff = s * shrink
        q = SSVIParams(sigma0=p.sigma0 + 0.01 * eff, eta=p.eta * max(1 - 0.05 * eff, 0.3),
                       gamma=p.gamma, rho=min(p.rho + 0.03 * eff, -0.05))
        if no_arb_ok(q):
            return q, eff
    raise RuntimeError("no-arb bump exhausted")


def seq_path(p, n, rng):
    """Cumulative random walk on the SSVI parameters, no-arb preserved by halving."""
    out, cur = [], p
    for _ in range(n):
        step = 1.0
        for _ in range(8):
            q = SSVIParams(sigma0=cur.sigma0 + step * rng.normal(0, 0.004),
                           eta=cur.eta * float(np.exp(step * rng.normal(0, 0.02))),
                           gamma=cur.gamma,
                           rho=float(np.clip(cur.rho + step * rng.normal(0, 0.015), -0.85, -0.05)))
            if no_arb_ok(q):
                break
            step *= 0.5
        else:
            q = cur
        out.append(q)
        cur = q
    return out


def dyn_variants(params):
    p = HestonParams.from_dict(params)
    return {
        "xi_up": HestonParams(p.kappa, p.theta, p.xi * 1.3, p.rho, p.v0),
        "rho_dn": HestonParams(p.kappa, p.theta, p.xi, max(p.rho - 0.15, -0.9), p.v0),
        "kappa_dn": HestonParams(p.kappa * 0.6, p.theta, p.xi, p.rho, p.v0),
    }


def _slice0(lv, s0, v0):
    sig0 = lv.sigma(max(0.0, lv.T_grid[0]), np.exp(GRID), s0)
    return Slice(0.0, GRID.copy(), np.clip(sig0 / np.sqrt(v0), 0.0, 4.0), np.full(len(GRID), v0))


def _slice_from_f(lv, t, f, s0):
    sig = lv.sigma(max(t, lv.T_grid[0]), np.exp(GRID), s0)
    return Slice(t, GRID.copy(), np.clip(sig / np.sqrt(f), 0.0, 4.0), f)


def distil(net, slices, lv, T, n_steps, s0, v0, sub, rng):
    """Per-slice ridge heads on a simulated cloud; same-time pairing throughout."""
    dt = T / n_steps
    head = GlobalRidge(net, T)
    betas, out = {}, [_slice0(lv, s0, v0)]
    for k in range(1, n_steps):
        t = k * dt
        _, lnx, v = slices[k - 1]
        idx = rng.choice(len(lnx), size=min(sub, len(lnx)), replace=False)
        f = np.clip(head.fit_predict(t, lnx[idx], v[idx], GRID), 1e-4, None)
        betas[k] = head.w_prev.copy()
        out.append(_slice_from_f(lv, t, f, s0))
    return betas, LeverageField(out)


def records_from_betas(net, betas, lv, T, n_steps, s0, v0):
    return records_from_head(GlobalRidge(net, T), betas, lv, T, n_steps, s0, v0)


def records_from_head(head, betas, lv, T, n_steps, s0, v0):
    """Rebuild the leverage field from a head's feature map and per-slice betas."""
    dt = T / n_steps
    out = [_slice0(lv, s0, v0)]
    for k in range(1, n_steps):
        t = k * dt
        f = np.clip(head.features(t, GRID) @ betas[k], 1e-4, None)
        out.append(_slice_from_f(lv, t, f, s0))
    return LeverageField(out)


def beta_correction(net, betas, lv, params, T, n_steps, N, s0, v0, sub, seed):
    """One damped correction in beta space: simulate under L(beta), refit heads, damp."""
    field = records_from_betas(net, betas, lv, T, n_steps, s0, v0)
    rng = np.random.default_rng(seed)
    slices = simulate_slices(field, params, s0, T, n_steps, N, rng)
    hat, _ = distil(net, slices, lv, T, n_steps, s0, v0, sub, rng)
    return {k: (1 - ALPHA) * betas[k] + ALPHA * hat[k] for k in betas}


def kalman_pass(head, betas, lv, params, T, n_steps, N, s0, v0, sub, seed):
    """Simulate under the current leverage, one Kalman measurement update per slice."""
    field = records_from_head(head, betas, lv, T, n_steps, s0, v0)
    rng = np.random.default_rng(seed)
    slices = simulate_slices(field, params, s0, T, n_steps, N, rng)
    for k in range(1, n_steps):
        t = k * (T / n_steps)
        _, lnx, v = slices[k - 1]
        idx = rng.choice(len(lnx), size=min(sub, len(lnx)), replace=False)
        head.update(k, t, lnx[idx], v[idx], GRID)
    return head.betas()


def full_resolve(lv, params, s0, T, cfg, seed, L0=None, n_iters=None):
    """Explicit warm start (unless L0 given) followed by the damped implicit solve. Returns (field, seconds)."""
    t0 = time.perf_counter()
    if L0 is None:
        ecfg = ExplicitConfig(n_steps=cfg.n_steps, n_particles=cfg.n_particles)
        L0 = calibrate_explicit(lv, params, make_estimator("nn", seed=seed, first_steps=ecfg.first_steps,
                                                            later_steps=ecfg.later_steps),
                                ecfg, s0=s0, T=T, seed=seed).field
    icfg = cfg if n_iters is None else dataclasses.replace(cfg, n_iters=n_iters)
    r = calibrate_implicit(lv, params, icfg, s0=s0, T=T, seed=seed, L0=L0)
    return r.field, time.perf_counter() - t0
```

`records_from_betas` in the old code built its own `GlobalRidgeHead` and called `_features`; delegating to `records_from_head` with a fresh `GlobalRidge` computes the same features. The old `full_resolve` explicit call used the defaults `first_steps=400, later_steps=120`, which `ExplicitConfig()` reproduces; note the warm suite's `Cfg` passes `N` as `n_particles` and `alpha=ALPHA=0.5` equals `ImplicitConfig.alpha`.

- [ ] **Step 7: Port bench/algos.py, the conftest, and the calibrate tests**

Replace `bench/algos.py` with:

```python
"""Algorithm registry: one calibrate interface over the seven contenders."""
import time
from dataclasses import dataclass, replace

from neural_particle_method.calibrate.config import ExplicitConfig, ImplicitConfig
from neural_particle_method.calibrate.explicit import calibrate_explicit
from neural_particle_method.calibrate.implicit import calibrate_implicit
from neural_particle_method.calibrate.importance import design_mixture
from neural_particle_method.estimators import make_estimator
from neural_particle_method.estimators.ridge import GlobalRidge
from neural_particle_method.market.local_vol import SSVILocalVol
from neural_particle_method.simulate.leverage import LeverageField


@dataclass
class CalibResult:
    field: LeverageField
    timings: dict
    diagnostics: dict


def _explicit(sc, n, seed, ecfg, method, mixture=None, estimator=None):
    lv = SSVILocalVol(sc.ssvi, sc.s0, T_max=sc.T)
    cfg = replace(ecfg, n_particles=n)
    est = estimator if estimator is not None else make_estimator(
        method, seed=seed, first_steps=cfg.first_steps, later_steps=cfg.later_steps)
    t0 = time.perf_counter()
    r = calibrate_explicit(lv, sc.dynamics, est, cfg, s0=sc.s0, T=sc.T, seed=seed, mixture=mixture)
    total = time.perf_counter() - t0
    diag = {"is_diag": r.is_diag} if r.is_diag is not None else {}
    return CalibResult(r.field, {"total_s": total, "fit_s": r.fit_s}, diag)


def _nw(sc, n, seed, e, i): return _explicit(sc, n, seed, e, "nw")
def _nn(sc, n, seed, e, i): return _explicit(sc, n, seed, e, "nn")
def _ridge(sc, n, seed, e, i): return _explicit(sc, n, seed, e, "ridge")
def _spline(sc, n, seed, e, i): return _explicit(sc, n, seed, e, "spline")


def _nn_is(sc, n, seed, e, i):
    return _explicit(sc, n, seed, e, "nn", mixture=design_mixture(sc.dynamics, sc.T))


def _implicit_core(sc, n, seed, e, i):
    lv = SSVILocalVol(sc.ssvi, sc.s0, T_max=sc.T)
    warm = _explicit(sc, n, seed, e, "nn")
    t0 = time.perf_counter()
    r = calibrate_implicit(lv, sc.dynamics, replace(i, n_particles=n), s0=sc.s0, T=sc.T,
                           seed=seed, L0=warm.field)
    total = time.perf_counter() - t0 + warm.timings["total_s"]
    return lv, warm, r, total


def _implicit(sc, n, seed, e, i):
    _, warm, r, total = _implicit_core(sc, n, seed, e, i)
    return CalibResult(r.field, {"total_s": total, "fit_s": r.fit_s + warm.timings["fit_s"]},
                       {"deltas": r.deltas})


def _implicit_ridge(sc, n, seed, e, i):
    lv, warm, r, overnight = _implicit_core(sc, n, seed, e, i)
    head = GlobalRidge(r.net, sc.T)
    t1 = time.perf_counter()
    er = calibrate_explicit(lv, sc.dynamics, head, replace(e, n_particles=n), s0=sc.s0, T=sc.T, seed=seed + 1)
    intraday = time.perf_counter() - t1
    return CalibResult(er.field, {"total_s": overnight + intraday, "fit_s": er.fit_s},
                       {"deltas": r.deltas, "overnight_s": overnight, "intraday_s": intraday})


ALGOS = {"nw": _nw, "explicit_nn": _nn, "ridge": _ridge, "explicit_nn_is": _nn_is,
         "implicit_nn": _implicit, "spline": _spline, "implicit_ridge": _implicit_ridge}


def run_algo(name, scenario, n_particles, seed, explicit=ExplicitConfig(), implicit=ImplicitConfig()):
    return ALGOS[name](scenario, n_particles, seed, explicit, implicit)
```

Fidelity notes against the old `bench/algos.py`: the old `_implicit` passed `n_steps` from cfg and `alpha`, `n_iters`, `fit_steps` with the same defaults; `pool_subsample` was never passed (default 60 000), and `replace(i, n_particles=n)` keeps it. The old `_implicit_ridge` intraday call used `method="nn"` defaults but a `regressor`, and `fit_subsample=cfg`, `first_steps`/`later_steps` unused by the ridge head; `replace(e, n_particles=n)` passes the same `fit_subsample`. The implicit-ridge warm explicit call inside `_implicit_core` uses seed `seed`; the intraday call uses `seed + 1`, as before.

Add to `tests/conftest.py`:

```python
from neural_particle_method.calibrate.config import ExplicitConfig, ImplicitConfig

TINY_EXPLICIT = ExplicitConfig(n_steps=6, fit_subsample=4_000, first_steps=80, later_steps=30)
TINY_IMPLICIT = ImplicitConfig(n_steps=6, n_iters=2, fit_steps=80)
```

In `tests/test_golden_tiny.py` and `tests/test_golden_archived.py`, change `run_algo(algo, sc, TINY_N, 0, TINY)` to `run_algo(algo, sc, TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT)` and `run_algo(algo, sc, n_particles, seed, None)` to `run_algo(algo, sc, n_particles, seed)`, and `res.L_records` to `res.field.to_records()`. Update `tests/test_algos.py` and `tests/test_runner.py` (still at top level until Task 8): `test_algos` passes the two dataclasses and reads `res.field[-1]` as a `Slice`; `bench/runner.py` passes `res.field.to_records()` to `reprice_iv` and takes `explicit`/`implicit` kwargs instead of `cfg` (`test_runner` and `bench/cli.py` updated accordingly: the CLI builds `replace(ExplicitConfig(), n_steps=args.n_steps)` when `--n-steps` is given).

Change `tests/estimators/test_contract.py` and `tests/estimators/test_kalman.py` to import `GlobalNet` from `neural_particle_method.calibrate.implicit`. Move `tests/test_implicit.py`, `tests/test_implicit_ridge.py`, `tests/test_importance.py` into `tests/calibrate/` and port each call to the new API. Porting rules:

- `calibrate_explicit(FlatDupire(), DYN, T=0.5, n_steps=8, n_particles=20_000, method="nw", fit_subsample=5_000, seed=3, mixture=d)` becomes `calibrate_explicit(FlatDupire(), DYN, make_estimator("nw"), ExplicitConfig(n_steps=8, n_particles=20_000, fit_subsample=5_000), T=0.5, seed=3, mixture=d)`; read `r.weights`, `r.field[-1].L`, `r.is_diag`, `r.snapshots`, `r.snapshot_weights`, `r.lnx`.
- `calibrate_implicit(FlatDupire(), DYN, T=0.5, n_steps=6, n_particles=8_000, alpha=0.5, n_iters=3, seed=1, pool_subsample=10_000, fit_steps=150)` becomes `calibrate_implicit(FlatDupire(), DYN, ImplicitConfig(n_steps=6, n_particles=8_000, n_iters=3, pool_subsample=10_000, fit_steps=150), T=0.5, seed=1)`; read `r.field`, `r.deltas`, `r.net`.
- `GlobalRidgeHead(info["net"], T=0.5)` becomes `GlobalRidge(r.net, T=0.5)`; `regressor=head` becomes the estimator positional argument.
- `test_spline_rejects_mixture` becomes `pytest.raises(ValueError, match="importance weights")` around a call with `make_estimator("spline")`.
- `mc_smile` is imported from `neural_particle_method.explicit` until Task 6.

- [ ] **Step 8: Run everything, then the goldens; commit**

Run: `uv run pytest -q`. Expected: all pass (tiny goldens included).
Run: `uv run pytest -m golden -q`. Expected: 4 passed.

```bash
git add -A src/neural_particle_method bench tests
git commit -m "refactor: calibrate subpackage with frozen configs, typed results, estimator injection, warm helpers"
```

---

### Task 6: `pricing/` subpackage

**Files:**
- Create: `src/neural_particle_method/pricing/__init__.py`, `reprice.py`, `metrics.py`
- Modify: old `reprice.py` becomes a shim; old `explicit.py` shim drops `mc_smile` in favour of a re-export
- Create: `tests/pricing/__init__.py`; move `tests/test_reprice.py` -> `tests/pricing/test_reprice.py` and split its metrics test into `tests/pricing/test_metrics.py`

**Interfaces:**
- Produces: `pricing.reprice.RepriceConfig(n_particles=500_000, n_steps=200)` frozen with `as_params()`; `reprice_iv(field, params, s0, maturities, k_grid, cfg=RepriceConfig(), seed=10_000) -> ndarray`; `snap_times(maturities, n_steps, T=None)`; `mc_smile(lnx_T, K_grid, s0=1.0, weights=None)`; `pricing.metrics.iv_metrics(iv_model, iv_target, k_grid, maturities, wing_cut=0.25)`; `pricing.metrics.target_ivs(ssvi_params, k_grid, maturities, n_steps) -> ndarray` (the `snap_times` + `implied_vol_ssvi` stack used in three places today).

- [ ] **Step 1: Write the failing test for `target_ivs` and the config**

`tests/pricing/__init__.py` empty. `tests/pricing/test_metrics.py`:

```python
import numpy as np

from neural_particle_method.market.ssvi import SSVIParams, implied_vol_ssvi
from neural_particle_method.pricing.metrics import iv_metrics, target_ivs
from neural_particle_method.pricing.reprice import RepriceConfig, snap_times

P = SSVIParams(sigma0=0.2, eta=1.0, gamma=0.4, rho=-0.6)


def test_target_ivs_uses_snapped_times():
    k = np.log(np.array([0.9, 1.0, 1.1]))
    mats = [0.25, 0.5]
    got = target_ivs(P, k, mats, n_steps=25)
    ts = snap_times(mats, 25)
    exp = np.stack([implied_vol_ssvi(P, k, t) for t in ts])
    np.testing.assert_array_equal(got, exp)


def test_iv_metrics_shapes():
    k = np.log(np.array([0.7, 1.0, 1.5]))
    target = np.full((2, 3), 0.2)
    model = target + np.array([[0.001, 0.0, -0.002], [0.0, 0.0, np.nan]])
    m = iv_metrics(model, target, k, [0.5, 1.0])
    assert m["n_failed"] == 1
    assert m["pooled_max_bp"] == 20.0
    assert m["wings_rmse_bp"] > 0
    assert len(m["per_maturity"]) == 2


def test_reprice_config_defaults():
    c = RepriceConfig()
    assert (c.n_particles, c.n_steps) == (500_000, 200) and c.as_params() == {"n_particles": 500_000, "n_steps": 200}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/pricing -q`. Expected: ImportError.

- [ ] **Step 3: Implement**

`pricing/__init__.py`: docstring only. `pricing/reprice.py`:

```python
"""Fresh-seed repricing of vanillas under a calibrated leverage field, and MC smiles from particles."""
from dataclasses import asdict, dataclass

import numpy as np

from ..market.bs import implied_vol
from ..simulate.dynamics import HestonParams
from ..simulate.leverage import LeverageField
from ..simulate.stepper import heston_step


@dataclass(frozen=True)
class RepriceConfig:
    n_particles: int = 500_000
    n_steps: int = 200

    def as_params(self):
        return asdict(self)


def snap_times(maturities, n_steps, T=None):
    """Grid-snapped time for each requested maturity, mirroring reprice_iv's dt exactly."""
    T = max(maturities) if T is None else T
    dt = T / n_steps
    return [int(round(m / dt)) * dt for m in maturities]


def reprice_iv(field, params, s0, maturities, k_grid, cfg=RepriceConfig(), seed=10_000):
    """Simulate fresh paths under the field, price calls at each maturity, invert to IV at the
    SNAPPED time (the time the simulated cloud actually reached)."""
    hp = HestonParams.from_dict(params)
    field = LeverageField.from_records(field)
    n_particles, n_steps = cfg.n_particles, cfg.n_steps
    rng = np.random.default_rng(seed)
    T = max(maturities)
    dt, sdt = T / n_steps, np.sqrt(T / n_steps)
    t_snap = dict(zip(maturities, snap_times(maturities, n_steps, T)))
    snap = {}
    for m in maturities:
        snap.setdefault(int(round(m / dt)), []).append(m)
    lnx = np.full(n_particles, np.log(s0))
    v = np.full(n_particles, hp.v0)
    ivs = np.full((len(maturities), len(k_grid)), np.nan)
    mat_idx = {m: i for i, m in enumerate(maturities)}
    for step in range(n_steps):
        t = step * dt
        L_p = field.at(t, lnx)
        zb = rng.standard_normal(n_particles)
        zp = rng.standard_normal(n_particles)
        lnx, v = heston_step(lnx, v, L_p, zb, zp, hp, dt, sdt)
        if step + 1 in snap:
            x = np.exp(lnx)
            for m in snap[step + 1]:
                for j, k in enumerate(k_grid):
                    K = s0 * np.exp(k)
                    price = float(np.mean(np.maximum(x - K, 0.0)))
                    ivs[mat_idx[m], j] = implied_vol(price, s0, K, t_snap[m])
    return ivs


def mc_smile(lnx_T, K_grid, s0=1.0, weights=None):
    """MC call prices from terminal particles; pass importance weights for a self-normalised mean."""
    x = np.exp(lnx_T)
    if weights is None:
        return np.array([np.mean(np.maximum(x - K, 0.0)) for K in K_grid])
    wn = weights / weights.sum()
    return np.array([np.sum(wn * np.maximum(x - K, 0.0)) for K in K_grid])
```

`pricing/metrics.py`: move `iv_metrics` verbatim and add:

```python
from ..market.ssvi import implied_vol_ssvi
from .reprice import snap_times


def target_ivs(ssvi_params, k_grid, maturities, n_steps):
    """Exact SSVI implied vols at the times reprice_iv actually reaches."""
    ts = snap_times(maturities, n_steps)
    return np.stack([implied_vol_ssvi(ssvi_params, k_grid, t) for t in ts])
```

Old `reprice.py` shim:

```python
from .pricing.metrics import iv_metrics  # noqa: F401
from .pricing.reprice import RepriceConfig, mc_smile, snap_times  # noqa: F401
from .pricing.reprice import reprice_iv as _new
from .simulate.leverage import LeverageField


def L_lookup(L_records):
    return LeverageField.from_records(L_records).at


def reprice_iv(L_records, dynamics, s0, maturities, k_grid, n_particles=500_000, n_steps=200, seed=10_000):
    return _new(L_records, dynamics, s0, maturities, k_grid,
                RepriceConfig(n_particles=n_particles, n_steps=n_steps), seed=seed)
```

Replace `mc_smile` in the old `explicit.py` shim with `from .pricing.reprice import mc_smile  # noqa: F401`.

- [ ] **Step 4: Port the tests**

`git mv tests/test_reprice.py tests/pricing/test_reprice.py`; import from `neural_particle_method.pricing.reprice`; `const_records` returns a `LeverageField.from_records(...)`; `test_L_lookup_piecewise` becomes `field.at(0.3, ...)`; `reprice_iv(..., n_particles=20_000, n_steps=12, seed=7)` becomes `reprice_iv(..., RepriceConfig(20_000, 12), seed=7)`; delete `test_iv_metrics_shapes` from this file (it now lives in `test_metrics.py`). Update `tests/test_golden_tiny.py`, `tests/test_golden_archived.py`, `tests/golden/make_tiny_goldens.py`, `bench/runner.py`, and `tests/calibrate/test_importance.py` to import `reprice_iv`, `RepriceConfig`, `iv_metrics`, `target_ivs`, `mc_smile` from `pricing`; replace their `snap_times`/`implied_vol_ssvi` stacks with `target_ivs(sc.ssvi, k, mats, n_steps)`.

- [ ] **Step 5: Run everything, then goldens; commit**

Run: `uv run pytest -q` then `uv run pytest -m golden -q`. Expected: all pass, 4 golden passed.

```bash
git add -A src/neural_particle_method bench tests
git commit -m "refactor: pricing subpackage; RepriceConfig and target_ivs"
```

---

### Task 7: `tracking/store.py`: the only module that imports mlflow

**Files:**
- Create: `src/neural_particle_method/tracking/__init__.py`, `store.py`
- Create: `tests/tracking/__init__.py`, `tests/tracking/test_store.py`

**Interfaces:**
- Produces:
  - `tracking.store.default_tracking_uri() -> str` and `default_artifact_root() -> str`.
  - `Store(tracking_uri=None, artifact_root=None)` with:
    - `experiment_id(name) -> str` (creates on first use, with `artifact_location=<artifact_root>/<name>`)
    - `find_finished(experiment, params: dict) -> str | None` (run id of a FINISHED run whose params match every key/value, values compared as strings)
    - `run(experiment, params: dict, tags: dict | None = None)` context manager yielding a `RunHandle`; on exception it logs `traceback.txt`, sets status FAILED, and re-raises
    - `download(run_id, artifact_path, dst_dir) -> Path`
    - `search(experiment, filter_string="", max_results=100_000) -> pandas.DataFrame` (columns `run_id`, `status`, `start_time`, `params.*`, `metrics.*`, `tags.*`)
    - `get_params(run_id) -> dict`, `get_metrics(run_id) -> dict`
  - `RunHandle` with `run_id`, `log_params(dict)`, `log_metrics(dict, step=None)`, `log_json(name, obj)`, `log_file(path, artifact_path=None)`, `set_tags(dict)`.
  - `tracking.store.to_jsonable(x)` (moved from `bench/runner._jsonable`) and `git_hash()` (moved from `bench/runner._git_hash`).
  - `tracking.store.flatten_metrics(prefix, d) -> dict` turning `{"a": {"b": 1.0}}` into `{"prefix/a/b": 1.0}`, skipping non-numeric leaves.

- [ ] **Step 1: Write the failing tests**

`tests/tracking/__init__.py` empty. `tests/tracking/test_store.py`:

```python
import json

import numpy as np
import pytest

from neural_particle_method.tracking.store import Store, flatten_metrics, to_jsonable


@pytest.fixture
def store(tmp_path):
    return Store(tracking_uri=f"sqlite:///{tmp_path / 'mlruns.db'}", artifact_root=str(tmp_path / "art"))


def test_log_and_query_round_trip(store, tmp_path):
    with store.run("bench", {"sid": "s01", "algo": "nw", "n_particles": 1000, "seed": 0}) as h:
        h.log_metrics({"pooled_rmse_bp": 12.5, "rmse_bp/T0.25": 3.0})
        h.log_json("leverage.json", {"slices": [{"t": 0.0}]})
        rid = h.run_id
    df = store.search("bench")
    assert len(df) == 1 and df.loc[0, "status"] == "FINISHED"
    assert df.loc[0, "params.algo"] == "nw" and df.loc[0, "metrics.pooled_rmse_bp"] == 12.5
    assert store.get_metrics(rid)["rmse_bp/T0.25"] == 3.0
    p = store.download(rid, "leverage.json", tmp_path / "dl")
    assert json.loads(p.read_text())["slices"][0]["t"] == 0.0


def test_find_finished_hits_on_identical_params_only(store):
    key = {"sid": "s01", "algo": "nw", "n_particles": 1000, "seed": 0}
    assert store.find_finished("bench", key) is None
    with store.run("bench", key) as h:
        rid = h.run_id
    assert store.find_finished("bench", key) == rid
    assert store.find_finished("bench", {**key, "seed": 1}) is None


def test_failed_run_is_recorded_and_does_not_block(store):
    key = {"sid": "s01", "algo": "nw", "n_particles": 1000, "seed": 0}
    with pytest.raises(RuntimeError, match="boom"):
        with store.run("bench", key):
            raise RuntimeError("boom")
    df = store.search("bench")
    assert df.loc[0, "status"] == "FAILED"
    assert store.find_finished("bench", key) is None


def test_metrics_with_step(store):
    with store.run("warm", {"arm": "seq", "sid": "s01", "seed": 0}) as h:
        for j in range(3):
            h.log_metrics({"rmse_bp/refresh": float(j)}, step=j)
        rid = h.run_id
    assert store.get_metrics(rid)["rmse_bp/refresh"] == 2.0     # latest value


def test_to_jsonable_and_flatten():
    assert to_jsonable({"a": np.float64(1.5), "b": [np.int64(2)], "c": float("nan")}) == {"a": 1.5, "b": [2], "c": None}
    assert flatten_metrics("rmse_bp", {"x": {"y": 1.0}, "z": 2, "s": "skip"}) == {"rmse_bp/x/y": 1.0, "rmse_bp/z": 2.0}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/tracking -q`. Expected: ImportError.

- [ ] **Step 3: Implement**

`tracking/__init__.py`: docstring only. `tracking/store.py`:

```python
"""MLflow-backed run store. This is the only module in the package that imports mlflow."""
import contextlib
import json
import os
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient


def default_tracking_uri():
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{Path.cwd() / 'mlruns.db'}"


def default_artifact_root():
    return os.environ.get("NPARTICLE_ARTIFACT_ROOT") or str(Path.cwd() / "mlartifacts")


def git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=False).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if not np.isfinite(f) else f
    if isinstance(x, np.integer):
        return int(x)
    return x


def flatten_metrics(prefix, d):
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_metrics(key, v))
        elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


class RunHandle:
    def __init__(self, client, run_id):
        self._c, self.run_id = client, run_id

    def log_params(self, params):
        self._c.log_batch(self.run_id, params=[Param(str(k), str(v)) for k, v in params.items()])

    def log_metrics(self, metrics, step=None):
        ts = int(time.time() * 1000)
        self._c.log_batch(self.run_id, metrics=[Metric(str(k), float(v), ts, step or 0)
                                                for k, v in metrics.items() if v is not None])

    def set_tags(self, tags):
        self._c.log_batch(self.run_id, tags=[RunTag(str(k), str(v)) for k, v in tags.items()])

    def log_file(self, path, artifact_path=None):
        self._c.log_artifact(self.run_id, str(path), artifact_path)

    def log_json(self, name, obj):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / name
            p.write_text(json.dumps(to_jsonable(obj)))
            self.log_file(p)


class Store:
    def __init__(self, tracking_uri=None, artifact_root=None):
        self.tracking_uri = tracking_uri or default_tracking_uri()
        self.artifact_root = artifact_root or default_artifact_root()
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def experiment_id(self, name):
        e = self.client.get_experiment_by_name(name)
        if e is not None:
            return e.experiment_id
        return self.client.create_experiment(name, artifact_location=str(Path(self.artifact_root) / name))

    @staticmethod
    def _filter(params, status=None):
        parts = [f"params.`{k}` = '{v}'" for k, v in params.items()]
        if status:
            parts.append(f"attributes.status = '{status}'")
        return " and ".join(parts)

    def find_finished(self, experiment, params):
        runs = self.client.search_runs([self.experiment_id(experiment)],
                                       self._filter(params, "FINISHED"), max_results=1)
        return runs[0].info.run_id if runs else None

    @contextlib.contextmanager
    def run(self, experiment, params, tags=None):
        r = self.client.create_run(self.experiment_id(experiment), tags=tags or {})
        h = RunHandle(self.client, r.info.run_id)
        h.log_params(params)
        try:
            yield h
        except BaseException:
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "traceback.txt"
                p.write_text(traceback.format_exc())
                h.log_file(p)
            self.client.set_terminated(h.run_id, status="FAILED")
            raise
        self.client.set_terminated(h.run_id, status="FINISHED")

    def download(self, run_id, artifact_path, dst_dir):
        Path(dst_dir).mkdir(parents=True, exist_ok=True)
        return Path(self.client.download_artifacts(run_id, artifact_path, str(dst_dir)))

    def get_params(self, run_id):
        return dict(self.client.get_run(run_id).data.params)

    def get_metrics(self, run_id):
        return dict(self.client.get_run(run_id).data.metrics)

    def search(self, experiment, filter_string="", max_results=100_000):
        runs = self.client.search_runs([self.experiment_id(experiment)], filter_string, max_results=max_results)
        rows = []
        for r in runs:
            row = {"run_id": r.info.run_id, "status": r.info.status, "start_time": r.info.start_time}
            row.update({f"params.{k}": v for k, v in r.data.params.items()})
            row.update({f"metrics.{k}": v for k, v in r.data.metrics.items()})
            row.update({f"tags.{k}": v for k, v in r.data.tags.items()})
            rows.append(row)
        return pd.DataFrame(rows)
```

If `search_runs` rejects the backtick-quoted key form on the installed mlflow version, use `params.sid = 's01'` without backticks for plain keys and keep backticks only for keys containing `/` or `.`.

MLflow's SQLite backend runs schema migrations the first time a database is opened, about a second each. Per-test databases are fine for the five store tests; if the default suite drifts past 30 s in later tasks, switch the bench, figures, and CLI fixtures to module scope and give each test its own experiment name instead of its own database.

- [ ] **Step 4: Run, then commit**

Run: `uv run pytest tests/tracking -q`. Expected: 5 passed. Then `uv run pytest -q` for the full suite.

```bash
git add -A src/neural_particle_method/tracking tests/tracking
git commit -m "feat: MLflow-backed Store wrapper with resumable lookup and failure capture"
```

---

### Task 8: `bench/` inside the package, logging to the store

**Files:**
- Move: `bench/scenarios.py`, `bench/algos.py` -> `src/neural_particle_method/bench/` (imports adjusted)
- Create: `src/neural_particle_method/bench/__init__.py`, `runner.py`, `sweep.py`, `aggregate.py`
- Modify: top-level `bench/__init__.py`, `bench/scenarios.py`, `bench/algos.py`, `bench/runner.py`, `bench/aggregate.py` become shims re-exporting the package versions (deleted in Task 12); `bench/cli.py` keeps working by calling the new functions with a `Store`
- Create: `tests/bench/__init__.py`, `test_runner.py`, `test_sweep.py`, `test_aggregate.py`; move `tests/test_scenarios.py`, `tests/test_algos.py` into `tests/bench/`; delete `tests/test_runner.py`, `tests/test_aggregate.py`, `tests/test_cli.py` (replaced)
- Modify: `tests/test_golden_tiny.py`, `tests/test_golden_archived.py`, `tests/golden/make_tiny_goldens.py` import paths

**Interfaces:**
- Produces:
  - `bench.scenarios.ScenarioSpec(sid, ssvi, dynamics: HestonParams, s0, T, maturities)`; `dynamics` is now a `HestonParams` (registries construct it with `HestonParams(...)`; the dict form `{"kappa": ..., ...}` is preserved by `dynamics.to_dict()` for logging).
  - `bench.runner.BENCH_EXPERIMENT = "bench"`, `run_key(sid, algo, n_particles, seed) -> dict`, `run_one(store, sid, algo, n_particles, seed, explicit=ExplicitConfig(), implicit=ImplicitConfig(), reprice=RepriceConfig()) -> str` (run id; if a FINISHED run exists for the key it returns that id without running). Logs params `run_key + explicit.as_params() prefixed "explicit." + implicit prefixed "implicit." + reprice prefixed "reprice." + git_hash + schema=2`, metrics `pooled_rmse_bp, pooled_max_bp, wings_rmse_bp, wings_max_bp, n_failed, total_s, fit_s, intraday_s (if present), overnight_s (if present), rmse_bp/T<m>` and artifacts `leverage.json`, `iv_err_bp.json`, `diagnostics.json`. Failures raise after being recorded as FAILED.
  - `bench.sweep.paper_grid() -> list[(sid, algo, n, seed)]`, `sweep(store, jobs, n_jobs=1, **cfgs) -> int` (count executed; skips existing FINISHED runs; parallel via `ProcessPoolExecutor`, each worker builds its own `Store` from the parent's URIs).
  - `bench.aggregate.COLUMNS`, `aggregate(store, out_csv="results/summary.csv", out_md="results/digest.md") -> DataFrame` with today's columns (`sid, algo, n_particles, seed, status, pooled_rmse_bp, pooled_max_bp, wings_rmse_bp, wings_max_bp, n_failed, total_s, fit_s, intraday_s, git_hash`); status column is `ok` for FINISHED and `failed` otherwise, so the CSV keeps the old vocabulary.

- [ ] **Step 1: Write the failing tests**

`tests/bench/__init__.py` empty. `tests/bench/test_runner.py`:

```python
import json

import pytest

from neural_particle_method.bench import runner
from neural_particle_method.bench.runner import run_key, run_one
from neural_particle_method.pricing.reprice import RepriceConfig
from neural_particle_method.tracking.store import Store
from tests.conftest import TINY_EXPLICIT, TINY_IMPLICIT, TINY_N, TINY_REPRICE_N, TINY_REPRICE_STEPS

TINY_REPRICE = RepriceConfig(TINY_REPRICE_N, TINY_REPRICE_STEPS)


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def test_run_one_logs_params_metrics_artifacts(store, tmp_path):
    rid = run_one(store, "s01", "nw", TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT, TINY_REPRICE)
    p, m = store.get_params(rid), store.get_metrics(rid)
    assert p["sid"] == "s01" and p["explicit.n_steps"] == "6" and p["schema"] == "2" and p["git_hash"]
    assert m["pooled_rmse_bp"] >= 0 and m["rmse_bp/T0.25"] >= 0 and m["total_s"] > 0
    err = json.loads(store.download(rid, "iv_err_bp.json", tmp_path / "d").read_text())
    assert len(err) == 4 and len(err[0]) == 13
    lev = json.loads(store.download(rid, "leverage.json", tmp_path / "d").read_text())
    assert len(lev["slices"]) == 6


def test_run_one_is_resumable(store):
    a = run_one(store, "s01", "nw", TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT, TINY_REPRICE)
    b = run_one(store, "s01", "nw", TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT, TINY_REPRICE)
    assert a == b and len(store.search("bench")) == 1


def test_failure_recorded_then_raised(store, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(runner, "run_algo", boom)
    with pytest.raises(RuntimeError):
        run_one(store, "s01", "nw", TINY_N, 1, TINY_EXPLICIT, TINY_IMPLICIT, TINY_REPRICE)
    df = store.search("bench")
    assert df.loc[0, "status"] == "FAILED"
    assert run_key("s01", "nw", TINY_N, 1) == {"sid": "s01", "algo": "nw", "n_particles": TINY_N, "seed": 1}
```

`tests/bench/test_sweep.py`:

```python
import pytest

from neural_particle_method.bench.algos import ALGOS
from neural_particle_method.bench.runner import run_one
from neural_particle_method.bench.sweep import paper_grid, sweep
from neural_particle_method.pricing.reprice import RepriceConfig
from neural_particle_method.tracking.store import Store
from tests.conftest import TINY_EXPLICIT, TINY_IMPLICIT, TINY_N, TINY_REPRICE_N, TINY_REPRICE_STEPS

TINY_REPRICE = RepriceConfig(TINY_REPRICE_N, TINY_REPRICE_STEPS)


def test_paper_grid_size():
    jobs = paper_grid()
    assert len(jobs) == 20 * len(ALGOS) * 2 * 3 + 6 * 2 * 3
    assert jobs[0] == ("s01", "nw", 50_000, 0)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_sweep_skips_finished_runs(tmp_path, n_jobs):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    run_one(store, "s01", "nw", TINY_N, 0, TINY_EXPLICIT, TINY_IMPLICIT, TINY_REPRICE)
    jobs = [("s01", "nw", TINY_N, 0), ("s01", "nw", TINY_N, 1)]
    done = sweep(store, jobs, n_jobs=n_jobs, explicit=TINY_EXPLICIT, implicit=TINY_IMPLICIT, reprice=TINY_REPRICE)
    assert done == 1 and len(store.search("bench")) == 2
```

`tests/bench/test_aggregate.py`:

```python
import pytest

from neural_particle_method.bench.aggregate import COLUMNS, aggregate
from neural_particle_method.tracking.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def _fake_run(store, algo, rmse, fail=False):
    key = {"sid": "s01", "algo": algo, "n_particles": 1000, "seed": 0}
    if fail:
        with pytest.raises(RuntimeError):
            with store.run("bench", {**key, "git_hash": "abc"}):
                raise RuntimeError("x")
        return
    with store.run("bench", {**key, "git_hash": "abc"}) as h:
        h.log_metrics({"pooled_rmse_bp": rmse, "pooled_max_bp": 30.0, "wings_rmse_bp": 20.0,
                       "wings_max_bp": 30.0, "n_failed": 0, "total_s": 1.0, "fit_s": 0.2})


def test_aggregate_writes_csv_and_digest(store, tmp_path):
    _fake_run(store, "nw", 12.0)
    _fake_run(store, "ridge", 8.0)
    _fake_run(store, "spline", 0.0, fail=True)
    df = aggregate(store, tmp_path / "s.csv", tmp_path / "d.md")
    assert list(df.columns) == list(COLUMNS) and len(df) == 3
    assert set(df.status) == {"ok", "failed"}
    md = (tmp_path / "d.md").read_text()
    assert "1 failed" in md and "s01: best = ridge" in md


def test_aggregate_empty(store, tmp_path):
    df = aggregate(store, tmp_path / "s.csv", tmp_path / "d.md")
    assert len(df) == 0 and list(df.columns) == list(COLUMNS)
    assert "0 runs, 0 failed" in (tmp_path / "d.md").read_text()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/bench -q`. Expected: ImportError on `neural_particle_method.bench`.

- [ ] **Step 3: Move scenarios and algos into the package**

`git mv bench/scenarios.py src/neural_particle_method/bench/scenarios.py`, `git mv bench/algos.py src/neural_particle_method/bench/algos.py`. In `scenarios.py`: import `from ..market.ssvi import SSVIParams, no_arb_ok` and `from ..simulate.dynamics import HestonParams`; type `dynamics: HestonParams`; in `make_registry` build `dyn = HestonParams(kappa=float(rng.choice(KAPPAS)), theta=p.sigma0 ** 2, xi=float(rng.choice(XIS)), rho=float(rng.choice(RHOS)), v0=p.sigma0 ** 2)` (same RNG call order: `kappa` drawn first, then `xi`, then `rho`, exactly as in the dict literal); in `fig3_registry` build `dyn = dataclasses.replace(base.dynamics, xi=xi, rho=rho)`. In `algos.py` make imports relative (`from ..calibrate...`). `bench/__init__.py` in the package: docstring only.

- [ ] **Step 4: Implement `runner.py`**

```python
"""Run one (scenario, algo, N, seed) and record it in the store."""
import dataclasses

from ..calibrate.config import ExplicitConfig, ImplicitConfig
from ..pricing.metrics import iv_metrics, target_ivs
from ..pricing.reprice import RepriceConfig, reprice_iv
from ..tracking.store import git_hash, to_jsonable
from .algos import run_algo
from .scenarios import full_registry, quote_k_grid

BENCH_EXPERIMENT = "bench"
SCHEMA = 2


def run_key(sid, algo, n_particles, seed):
    return {"sid": sid, "algo": algo, "n_particles": int(n_particles), "seed": int(seed)}


def _prefixed(prefix, d):
    return {f"{prefix}.{k}": v for k, v in d.items()}


def run_one(store, sid, algo, n_particles, seed, explicit=ExplicitConfig(), implicit=ImplicitConfig(),
            reprice=RepriceConfig()):
    key = run_key(sid, algo, n_particles, seed)
    existing = store.find_finished(BENCH_EXPERIMENT, key)
    if existing is not None:
        return existing
    sc = full_registry()[sid]
    params = {**key, "git_hash": git_hash(), "schema": SCHEMA,
              **_prefixed("explicit", explicit.as_params()), **_prefixed("implicit", implicit.as_params()),
              **_prefixed("reprice", reprice.as_params()),
              **_prefixed("scenario.ssvi", dataclasses.asdict(sc.ssvi)),
              **_prefixed("scenario.dynamics", sc.dynamics.to_dict()),
              "scenario.s0": sc.s0, "scenario.T": sc.T, "scenario.maturities": str(list(sc.maturities))}
    with store.run(BENCH_EXPERIMENT, params) as h:
        res = run_algo(algo, sc, n_particles, seed, explicit, implicit)
        k = quote_k_grid()
        mats = list(sc.maturities)
        iv_model = reprice_iv(res.field, sc.dynamics, sc.s0, mats, k, reprice, seed=seed + 10_000)
        iv_target = target_ivs(sc.ssvi, k, mats, reprice.n_steps)
        m = iv_metrics(iv_model, iv_target, k, mats)
        metrics = {c: m[c] for c in ("pooled_rmse_bp", "pooled_max_bp", "wings_rmse_bp", "wings_max_bp", "n_failed")}
        metrics.update({f"rmse_bp/T{r['T']:g}": r["rmse_bp"] for r in m["per_maturity"]})
        metrics.update(res.timings)
        for c in ("intraday_s", "overnight_s"):
            if c in res.diagnostics:
                metrics[c] = res.diagnostics[c]
        h.log_metrics(metrics)
        h.log_json("leverage.json", res.field.to_json())
        h.log_json("iv_err_bp.json", ((iv_model - iv_target) * 1e4).tolist())
        h.log_json("diagnostics.json", to_jsonable(res.diagnostics))
        return h.run_id
```

Fidelity: the old runner called `reprice_iv(..., seed=seed + 10_000)` and computed the target at `snap_times(mats, reprice_steps)`; both are preserved.

- [ ] **Step 5: Implement `sweep.py`**

```python
"""Paper grid enumeration and a resumable, optionally parallel sweep."""
from concurrent.futures import ProcessPoolExecutor

from ..calibrate.config import ExplicitConfig, ImplicitConfig
from ..pricing.reprice import RepriceConfig
from ..tracking.store import Store
from .algos import ALGOS
from .runner import BENCH_EXPERIMENT, run_key, run_one
from .scenarios import fig3_registry, make_registry


def paper_grid():
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


def _worker(args):
    uri, root, job, explicit, implicit, reprice = args
    store = Store(uri, root)
    run_one(store, *job, explicit=explicit, implicit=implicit, reprice=reprice)
    return job


def sweep(store, jobs, n_jobs=1, explicit=ExplicitConfig(), implicit=ImplicitConfig(), reprice=RepriceConfig()):
    todo = [j for j in jobs if store.find_finished(BENCH_EXPERIMENT, run_key(*j)) is None]
    print(f"{len(todo)} runs to do", flush=True)
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

SQLite tolerates a few concurrent writers with short waits; if a `database is locked` error appears at high `n_jobs`, rerun the sweep, which resumes.

- [ ] **Step 6: Implement `aggregate.py`**

```python
"""Query the bench experiment into a tidy summary table and a short digest."""
from pathlib import Path

import pandas as pd

from .runner import BENCH_EXPERIMENT

METRIC_COLS = ("pooled_rmse_bp", "pooled_max_bp", "wings_rmse_bp", "wings_max_bp", "n_failed")
COLUMNS = ("sid", "algo", "n_particles", "seed", "status") + METRIC_COLS + \
          ("total_s", "fit_s", "intraday_s", "git_hash")


def aggregate(store, out_csv="results/summary.csv", out_md="results/digest.md"):
    raw = store.search(BENCH_EXPERIMENT)
    rows = []
    for _, r in raw.iterrows():
        row = {"sid": r.get("params.sid"), "algo": r.get("params.algo"),
               "n_particles": int(r["params.n_particles"]) if pd.notna(r.get("params.n_particles")) else None,
               "seed": int(r["params.seed"]) if pd.notna(r.get("params.seed")) else None,
               "status": "ok" if r["status"] == "FINISHED" else "failed"}
        for c in METRIC_COLS + ("total_s", "fit_s", "intraday_s"):
            row[c] = r.get(f"metrics.{c}")
        row["git_hash"] = r.get("params.git_hash")
        rows.append(row)
    df = pd.DataFrame(rows, columns=COLUMNS)
    if len(df):
        df = df.sort_values(["sid", "algo", "n_particles", "seed"]).reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    n_fail = int((df.status == "failed").sum())
    lines = ["# Benchmark digest", f"{len(df)} runs, {n_fail} failed", ""]
    ok = df[df.status == "ok"]
    if len(ok):
        best = ok.groupby(["sid", "algo"]).pooled_rmse_bp.mean().reset_index() \
                 .sort_values("pooled_rmse_bp").groupby("sid").first()
        for sid, r in best.iterrows():
            lines.append(f"- {sid}: best = {r.algo} ({r.pooled_rmse_bp:.1f} bp)")
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines) + "\n")
    return df
```

- [ ] **Step 7: Shims and the transitional CLI**

Top-level `bench/scenarios.py`: `from neural_particle_method.bench.scenarios import *  # noqa` plus explicit names `ScenarioSpec, make_registry, fig3_registry, full_registry, quote_k_grid, XIS, RHOS, KAPPAS`. `bench/algos.py`: re-export `ALGOS, CalibResult, run_algo`. `bench/runner.py`: re-export `run_one, run_key`. `bench/aggregate.py`: re-export `aggregate, COLUMNS`. Rewrite `bench/cli.py` so `run`, `sweep`, `aggregate` build `Store()` and call the package functions (`--results-dir` and `--runs-dir` flags are removed; `--tracking-uri` optional flag added), and `figures` keeps calling the top-level `figures` modules with `results/summary.csv` (fig2 breaks until Task 11; that is acceptable because `bench figures` has no test at this point and Task 11 fixes it).

Delete `tests/test_runner.py`, `tests/test_aggregate.py`, `tests/test_cli.py`. Move `tests/test_scenarios.py` and `tests/test_algos.py` into `tests/bench/` with package imports; `test_scenarios` gains `assert isinstance(next(iter(make_registry().values())).dynamics, HestonParams)`; `test_fig3_cross` reads `s.dynamics.xi`, `s.dynamics.rho`. Update the golden tests and generator to import `run_algo`, `make_registry`, `quote_k_grid` from `neural_particle_method.bench`.

- [ ] **Step 8: Run everything, goldens, commit**

Run: `uv run pytest -q` (expect all green; the sweep test with `n_jobs=2` may take ~10 s), then `uv run pytest -m golden -q`.

```bash
git add -A src/neural_particle_method bench tests
git commit -m "refactor: bench inside the package; runs, sweeps, and aggregation go through the MLflow store"
```

---

### Task 9: `experiments/` inside the package, overnight cache as a run

**Files:**
- Create: `src/neural_particle_method/experiments/__init__.py`, `config.py`, `bump_correct.py`, `warm_suite.py`
- Create: `tests/experiments/__init__.py`, `test_config.py`, `test_warm_suite.py`
- Root `experiments/bump_correct.py`, `experiments/warm_suite.py` are left untouched (deleted in Task 12)

**Interfaces:**
- Produces:
  - `experiments.config.WarmConfig(N=200_000, n_steps=50, sub=30_000, n_iters=6, fit_steps=300, reprice_N=300_000, reprice_steps=200, seq_len=6, xover_scales=(0.5, 1.0, 2.0, 4.0), norm_iters=4)`, `FULL = WarmConfig()`, `SMOKE = WarmConfig(N=20_000, n_steps=12, sub=8_000, n_iters=2, fit_steps=120, reprice_N=60_000, reprice_steps=100, seq_len=3, xover_scales=(1.0, 4.0), norm_iters=2)`, method `as_params()`, properties `implicit -> ImplicitConfig(n_steps, n_particles=N, alpha=0.5, n_iters, fit_steps)`, `reprice -> RepriceConfig(reprice_N, reprice_steps)`.
  - `experiments.config.BumpConfig(N=200_000, n_steps=50, sub=30_000, n_iters=6, fit_steps=300)`, `BUMP_FULL`, `BUMP_SMOKE = BumpConfig(20_000, 12, 8_000, 2, 120)`, `reprice -> RepriceConfig(300_000, 200)`.
  - `experiments.warm_suite.score(field, params, s0, mats, kq, ssvi_p, seed, reprice_cfg) -> float`.
  - `experiments.warm_suite.overnight(store, sc, seed, cfg) -> (net, betas, fieldS, overnight_s, lvS)`; looks up experiment `overnight` by `{"sid", "seed", "N", "n_steps"}`; on miss computes, saves `overnight.pt` (`{"net": state_dict, "betas": betas, "recsS": field.to_records(), "overnight_s": s}`) as an artifact and logs metric `overnight_s`.
  - `arm_dyn(store, sc, seed, cfg) -> dict`, `arm_seq`, `arm_xover`, `arm_norm` with the same return documents as today; `ARMS`, `DEFAULT_SIDS`, `job_list(arms, sids, dyn_seeds=(0, 1), other_seeds=(0,))`, `run_warm(store, arms, sids, cfg) -> int`, which for each job skips a FINISHED run keyed `{"arm", "sid", "seed"}` in experiment `warm`, else runs the arm inside `store.run` and logs: params `cfg.as_params()`, `git_hash`; metrics from the doc (`flatten_metrics("rmse_bp", doc["rmse_bp"])` and `flatten_metrics("build_s", doc["build_s"])` for dyn/xover; for seq each `steps[j]` row as `rmse_bp/<s>` at `step=j` and `build_s/<s>` from `build_s[f"step_{j}"]` at `step=j`, plus `rmse_bp/resolve_cold_final`, `build_s/resolve_cold_final`, `build_s/overnight`; for norm `A_norm`, `error_fraction_pred` (if not None), `eps`, and `A_norm_history` at `step=i`); metric `wall_s`; artifact `result.json` with the full doc.
  - `experiments.warm_suite.summarise(store) -> str` (the markdown from today's `aggregate()`, rebuilt from `result.json` artifacts of FINISHED `warm` runs).
  - `experiments.bump_correct.run_pair(store, sc, seed, cfg: BumpConfig) -> str` (run id; skips if FINISHED for `{"sid", "seed"}` in experiment `bump`); logs params `cfg.as_params()`, `bumped.*` SSVI fields, `git_hash`; metrics `rmse_bp/<k>`, `build_s/<k>`; artifact `result.json`.

- [ ] **Step 1: Write the failing tests**

`tests/experiments/__init__.py` empty. `tests/experiments/test_config.py`:

```python
from neural_particle_method.experiments.config import BUMP_SMOKE, FULL, SMOKE, BumpConfig, WarmConfig


def test_full_and_smoke_match_legacy_values():
    assert (FULL.N, FULL.n_steps, FULL.sub, FULL.n_iters, FULL.fit_steps, FULL.reprice_N,
            FULL.reprice_steps, FULL.seq_len, FULL.xover_scales, FULL.norm_iters) == \
        (200_000, 50, 30_000, 6, 300, 300_000, 200, 6, (0.5, 1.0, 2.0, 4.0), 4)
    assert SMOKE.N == 20_000 and SMOKE.xover_scales == (1.0, 4.0)
    assert BUMP_SMOKE == BumpConfig(20_000, 12, 8_000, 2, 120)


def test_derived_configs():
    i = WarmConfig().implicit
    assert (i.n_steps, i.n_particles, i.alpha, i.n_iters, i.fit_steps) == (50, 200_000, 0.5, 6, 300)
    r = BumpConfig().reprice
    assert (r.n_particles, r.n_steps) == (300_000, 200)
    assert WarmConfig().as_params()["xover_scales"] == "(0.5, 1.0, 2.0, 4.0)"
```

`tests/experiments/test_warm_suite.py`:

```python
import pytest

from neural_particle_method.experiments.bump_correct import run_pair
from neural_particle_method.experiments.config import BUMP_SMOKE, SMOKE
from neural_particle_method.experiments.warm_suite import ARMS, job_list, run_warm, summarise
from neural_particle_method.bench.scenarios import make_registry
from neural_particle_method.tracking.store import Store


def test_job_list_enumerates_expected_jobs():
    jobs = job_list(["dyn", "norm"], ["s01", "s02"])
    assert jobs == [("dyn", "s01", 0), ("dyn", "s01", 1), ("dyn", "s02", 0), ("dyn", "s02", 1),
                    ("norm", "s01", 0), ("norm", "s02", 0)]
    assert set(ARMS) == {"dyn", "seq", "xover", "norm"}


@pytest.mark.slow
@pytest.mark.parametrize("arm", sorted(ARMS))
def test_each_arm_smokes_through_store(tmp_path, arm):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    assert run_warm(store, [arm], ["s01"], SMOKE) == 1
    assert run_warm(store, [arm], ["s01"], SMOKE) == 0          # resumable
    assert len(store.search("overnight")) == 1                   # cache shared, logged once
    md = summarise(store)
    assert arm in md


@pytest.mark.slow
def test_bump_smokes_through_store(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    rid = run_pair(store, make_registry()["s01"], 0, BUMP_SMOKE)
    m = store.get_metrics(rid)
    assert "rmse_bp/causal_sweep" in m and "build_s/full_resolve" in m
    assert run_pair(store, make_registry()["s01"], 0, BUMP_SMOKE) == rid
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/experiments -q`. Expected: ImportError.

- [ ] **Step 3: Implement `config.py`**

```python
"""Experiment configurations (values verbatim from the legacy scripts)."""
from dataclasses import asdict, dataclass

from ..calibrate.config import ImplicitConfig
from ..pricing.reprice import RepriceConfig


def _flat(d):
    return {k: (str(v) if isinstance(v, (tuple, list)) else v) for k, v in d.items()}


@dataclass(frozen=True)
class WarmConfig:
    N: int = 200_000
    n_steps: int = 50
    sub: int = 30_000
    n_iters: int = 6
    fit_steps: int = 300
    reprice_N: int = 300_000
    reprice_steps: int = 200
    seq_len: int = 6
    xover_scales: tuple = (0.5, 1.0, 2.0, 4.0)
    norm_iters: int = 4

    def as_params(self):
        return _flat(asdict(self))

    @property
    def implicit(self):
        return ImplicitConfig(n_steps=self.n_steps, n_particles=self.N, alpha=0.5,
                              n_iters=self.n_iters, fit_steps=self.fit_steps)

    @property
    def reprice(self):
        return RepriceConfig(self.reprice_N, self.reprice_steps)


FULL = WarmConfig()
SMOKE = WarmConfig(N=20_000, n_steps=12, sub=8_000, n_iters=2, fit_steps=120,
                   reprice_N=60_000, reprice_steps=100, seq_len=3,
                   xover_scales=(1.0, 4.0), norm_iters=2)


@dataclass(frozen=True)
class BumpConfig:
    N: int = 200_000
    n_steps: int = 50
    sub: int = 30_000
    n_iters: int = 6
    fit_steps: int = 300

    def as_params(self):
        return _flat(asdict(self))

    @property
    def implicit(self):
        return ImplicitConfig(n_steps=self.n_steps, n_particles=self.N, alpha=0.5,
                              n_iters=self.n_iters, fit_steps=self.fit_steps)

    @property
    def reprice(self):
        return RepriceConfig(300_000, 200)


BUMP_FULL = BumpConfig()
BUMP_SMOKE = BumpConfig(20_000, 12, 8_000, 2, 120)
```

- [ ] **Step 4: Implement `warm_suite.py`**

Port each arm from `experiments/warm_suite.py` (root) with these mechanical substitutions and nothing else: `dyn` dict -> `sc.dynamics` (`HestonParams`); `dict(dyn, xi=...)` -> `dyn_variants` from `calibrate.warm`; `_simulate` -> `simulate_slices`; `GlobalRidgeHead` -> `GlobalRidge`; `records_from_*` now return `LeverageField`; `calibrate_explicit`/`calibrate_implicit` old signatures -> new ones with `ExplicitConfig(n_steps=cfg.n_steps, n_particles=cfg.N)` and `cfg.implicit`; `score(..., cfg)` -> `score(..., cfg.reprice)`; `[r[2] for r in base_recs]` -> `base.L_matrix()`; the perturbed field in `arm_norm` built as `LeverageField([Slice(s.t, s.grid, np.clip(s.L + eps * u[k], 0.0, 4.0), s.f) for k, s in enumerate(base)])`. Every `seed + <offset>` stays identical.

The parts that change beyond substitution:

```python
"""Warm-start suite: dynamics bumps, sequential tracking, crossover sweep, feedback-norm measurement."""
import dataclasses
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from ..bench.scenarios import make_registry, quote_k_grid
from ..calibrate.config import ExplicitConfig
from ..calibrate.explicit import calibrate_explicit
from ..calibrate.implicit import GlobalNet, calibrate_implicit, simulate_slices
from ..calibrate.warm import (beta_correction, distil, dyn_variants, full_resolve, kalman_pass,
                              records_from_betas, records_from_head, scaled_bump, seq_path)
from ..estimators import make_estimator
from ..estimators.kalman import KalmanHead
from ..estimators.ridge import GlobalRidge
from ..market.local_vol import SSVILocalVol
from ..pricing.metrics import iv_metrics, target_ivs
from ..pricing.reprice import reprice_iv
from ..simulate.leverage import DEFAULT_GRID as GRID, LeverageField, Slice
from ..tracking.store import flatten_metrics, git_hash, to_jsonable
from .config import WarmConfig

OVERNIGHT_EXPERIMENT = "overnight"
WARM_EXPERIMENT = "warm"


def score(field, params, s0, mats, kq, ssvi_p, seed, reprice_cfg):
    ivs = reprice_iv(field, params, s0, mats, kq, reprice_cfg, seed=seed)
    tgt = target_ivs(ssvi_p, kq, mats, reprice_cfg.n_steps)
    return iv_metrics(ivs, tgt, kq, mats)["pooled_rmse_bp"]


def overnight_key(sc, seed, cfg):
    return {"sid": sc.sid, "seed": seed, "N": cfg.N, "n_steps": cfg.n_steps}


def overnight(store, sc, seed, cfg):
    """Overnight solve, cached as an `overnight` run: implicit field, trained net, distilled betas."""
    lvS = SSVILocalVol(sc.ssvi, sc.s0, T_max=sc.T)
    key = overnight_key(sc, seed, cfg)
    rid = store.find_finished(OVERNIGHT_EXPERIMENT, key)
    if rid is not None:
        with tempfile.TemporaryDirectory() as d:
            blob = torch.load(store.download(rid, "overnight.pt", d), weights_only=False)
        net = GlobalNet()
        net.load_state_dict(blob["net"])
        return net, blob["betas"], LeverageField.from_records(blob["recsS"]), blob["overnight_s"], lvS
    dyn, s0, T = sc.dynamics, sc.s0, sc.T
    with store.run(OVERNIGHT_EXPERIMENT, {**key, "git_hash": git_hash()}) as h:
        t0 = time.perf_counter()
        w = calibrate_explicit(lvS, dyn, make_estimator("nn", seed=seed),
                               ExplicitConfig(n_steps=cfg.n_steps, n_particles=cfg.N), s0=s0, T=T, seed=seed)
        r = calibrate_implicit(lvS, dyn, cfg.implicit, s0=s0, T=T, seed=seed, L0=w.field)
        net = r.net
        rng = np.random.default_rng(seed + 50)
        slices = simulate_slices(r.field, dyn, s0, T, cfg.n_steps, cfg.N, rng)
        betas, _ = distil(net, slices, lvS, T, cfg.n_steps, s0, dyn.v0, cfg.sub, rng)
        elapsed = time.perf_counter() - t0
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "overnight.pt"
            torch.save({"net": net.state_dict(), "betas": betas, "recsS": r.field.to_records(),
                        "overnight_s": elapsed}, p)
            h.log_file(p)
        h.log_metrics({"overnight_s": elapsed})
    return net, betas, r.field, elapsed, lvS
```

(the old `overnight` called `calibrate_explicit(lvS, dyn, s0=s0, T=T, n_steps=cfg.n_steps, n_particles=cfg.N, seed=seed)` with default `method="nn"`, `first_steps=400`, `later_steps=120`, which `make_estimator("nn", seed=seed)` reproduces.)

Arm functions take `store` as first argument only to call `overnight(store, ...)`. Driver:

```python
ARMS = {"dyn": arm_dyn, "seq": arm_seq, "xover": arm_xover, "norm": arm_norm}
DEFAULT_SIDS = ("s01", "s02", "s03", "s04", "s05")


def job_list(arms, sids, dyn_seeds=(0, 1), other_seeds=(0,)):
    jobs = []
    for arm in arms:
        for sid in sids:
            for seed in (dyn_seeds if arm == "dyn" else other_seeds):
                jobs.append((arm, sid, seed))
    return jobs


def _log_doc(h, arm, doc):
    if arm == "norm":
        m = {"A_norm": doc["A_norm"], "eps": doc["eps"]}
        if doc.get("error_fraction_pred") is not None:
            m["error_fraction_pred"] = doc["error_fraction_pred"]
        h.log_metrics(m)
        for i, a in enumerate(doc["A_norm_history"]):
            h.log_metrics({"A_norm_history": a}, step=i)
    elif arm == "seq":
        for j, row in enumerate(doc["rmse_bp"]["steps"]):
            h.log_metrics(flatten_metrics("rmse_bp", row), step=j)
            h.log_metrics(flatten_metrics("build_s", doc["build_s"][f"step_{j}"]), step=j)
        h.log_metrics({"rmse_bp/resolve_cold_final": doc["rmse_bp"]["resolve_cold_final"],
                       "build_s/resolve_cold_final": doc["build_s"]["resolve_cold_final"],
                       "build_s/overnight": doc["build_s"]["overnight"]})
    else:
        h.log_metrics(flatten_metrics("rmse_bp", doc["rmse_bp"]))
        h.log_metrics(flatten_metrics("build_s", doc["build_s"]))
    h.log_metrics({"wall_s": doc["wall_s"]})
    h.log_json("result.json", doc)


def run_warm(store, arms, sids, cfg):
    reg = make_registry()
    n = 0
    for arm, sid, seed in job_list(arms, sids):
        key = {"arm": arm, "sid": sid, "seed": seed}
        if store.find_finished(WARM_EXPERIMENT, key) is not None:
            continue
        with store.run(WARM_EXPERIMENT, {**key, **cfg.as_params(), "git_hash": git_hash()}) as h:
            t0 = time.perf_counter()
            doc = ARMS[arm](store, reg[sid], seed, cfg)
            doc.update({"arm": arm, "sid": sid, "seed": seed, "wall_s": round(time.perf_counter() - t0, 1)})
            _log_doc(h, arm, to_jsonable(doc))
        n += 1
        print(f"{arm} {sid} s{seed}: done in {doc['wall_s']}s", flush=True)
    return n


def summarise(store):
    """Markdown summary rebuilt from result.json artifacts (same layout as the legacy summary.md)."""
    df = store.search(WARM_EXPERIMENT, "attributes.status = 'FINISHED'")
    rows = []
    with tempfile.TemporaryDirectory() as d:
        for rid in df.get("run_id", []):
            rows.append(json.loads(store.download(rid, "result.json", Path(d) / rid).read_text()))
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    lines = ["# Warm-start suite summary", ""]
    for arm in ("dyn", "xover", "norm", "seq"):
        docs = by_arm.get(arm, [])
        if not docs:
            continue
        lines.append(f"## {arm} ({len(docs)} runs)")
        if arm == "seq":
            n_steps = min(len(d["rmse_bp"]["steps"]) for d in docs)
            strategies = list(docs[0]["rmse_bp"]["steps"][0].keys())
            lines.append("| step | " + " | ".join(strategies) + " |")
            lines.append("|" + "---|" * (len(strategies) + 1))
            for j in range(n_steps):
                vals = [np.mean([d["rmse_bp"]["steps"][j][s] for d in docs]) for s in strategies]
                lines.append(f"| {j} | " + " | ".join(f"{v:.0f}" for v in vals) + " |")
            lines.append(f"cold re-solve at final step: "
                         f"{np.mean([d['rmse_bp']['resolve_cold_final'] for d in docs]):.0f} bp")
        elif arm == "norm":
            for d in docs:
                lines.append(f"- {d['sid']} s{d['seed']}: |A| = {d['A_norm']}, "
                             f"predicted error fraction {d['error_fraction_pred']}, "
                             f"history {d['A_norm_history']}")
        else:
            keys = sorted({k for d in docs for k in d["rmse_bp"] if not k.endswith("scale_eff")})
            lines.append("| strategy | mean rmse bp | mean build s |")
            lines.append("|---|---|---|")
            for k in keys:
                vals = [d["rmse_bp"][k] for d in docs if k in d["rmse_bp"]]
                ts = [d["build_s"].get(k, "") for d in docs if k in d["rmse_bp"]]
                ts = [t for t in ts if t != ""]
                lines.append(f"| {k} | {np.mean(vals):.0f} | {np.mean(ts):.1f} |" if ts
                             else f"| {k} | {np.mean(vals):.0f} | |")
        lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 5: Implement `bump_correct.py`**

Port `run_pair` from the root script with the same substitutions; `N, n_steps, sub, n_iters, fit_steps` come from `cfg`; wrap in `store.run("bump", {...})` keyed `{"sid": sc.sid, "seed": seed}` with the resumable lookup first; log `flatten_metrics("rmse_bp", res)`, `flatten_metrics("build_s", times)`, params `bumped.<field>` from `dataclasses.asdict(pB)`, artifact `result.json` with `{"sid", "seed", "rmse_bp", "build_s", "bumped"}`. Return the run id. The old `score` here used `n_particles=300_000, n_steps=200`, which `cfg.reprice` reproduces. The old `calibrate_explicit(lvB, dyn, ..., fit_subsample=sub, seed=seed + 1, regressor=head)` becomes `calibrate_explicit(lvB, dyn, head, ExplicitConfig(n_steps=n_steps, n_particles=N, fit_subsample=sub), s0=s0, T=T, seed=seed + 1)`.

- [ ] **Step 6: Run fast tests, then the slow smokes once, commit**

Run: `uv run pytest tests/experiments -q` (fast ones). Then `uv run pytest tests/experiments -m slow -q` (a few minutes). Expected: all pass. Then `uv run pytest -q`.

```bash
git add -A src/neural_particle_method/experiments tests/experiments
git commit -m "refactor: experiments inside the package; overnight cache and results live in the store"
```

---

### Task 10: Legacy importer

**Files:**
- Create: `src/neural_particle_method/tracking/importer.py`
- Create: `tests/tracking/test_importer.py`

**Interfaces:**
- Produces: `tracking.importer.import_bench(store, runs_dir="results/runs") -> int`, `import_bump(store, dir="results/bump") -> int`, `import_warm(store, dir="results/warm") -> int`, `import_overnight_cache(store, dir="results/warm/cache") -> int`, `import_all(store, root="results") -> dict[str, int]`. Each returns the number of runs created; a file whose path is already stored as param `legacy_path` (relative to the repo root, forward slashes) is skipped. Every imported run carries tag `source=legacy_json` and param `legacy_path`; bench runs also carry the original `git_hash` and `schema` params.

- [ ] **Step 1: Write the failing test**

`tests/tracking/test_importer.py`:

```python
import json

import pytest

from neural_particle_method.tracking.importer import import_all, import_bench, import_warm
from neural_particle_method.tracking.store import Store

OK = {"schema": 1, "sid": "s01", "algo": "nw", "n_particles": 1000, "seed": 0, "git_hash": "abc",
      "scenario": {"ssvi": {"sigma0": 0.2, "eta": 1.0, "gamma": 0.4, "rho": -0.6},
                   "dynamics": {"kappa": 1.0, "theta": 0.04, "xi": 0.3, "rho": -0.7, "v0": 0.04},
                   "s0": 1.0, "T": 2.0, "maturities": [0.25, 0.5, 1.0, 2.0]},
      "status": "ok", "timings": {"total_s": 1.0, "fit_s": 0.2}, "diagnostics": {"intraday_s": 0.5},
      "metrics": {"pooled_rmse_bp": 12.0, "pooled_max_bp": 30.0, "wings_rmse_bp": 20.0, "wings_max_bp": 30.0,
                  "n_failed": 0, "per_maturity": [{"T": 0.25, "rmse_bp": 3.0, "max_bp": 5.0}]},
      "iv_err_bp": [[1.0, None]]}
BAD = {"schema": 1, "sid": "s01", "algo": "ridge", "n_particles": 1000, "seed": 0, "git_hash": "abc",
       "status": "failed", "error": "Traceback: boom"}
SEQ = {"arm": "seq", "sid": "s01", "seed": 0, "wall_s": 9.5,
       "rmse_bp": {"steps": [{"refresh": 10.0, "ridge": 8.0}, {"refresh": 11.0, "ridge": 7.0}],
                   "resolve_cold_final": 5.0, "path": []},
       "build_s": {"overnight": 100.0, "step_0": {"refresh": 0.1, "ridge": 2.0},
                   "step_1": {"refresh": 0.1, "ridge": 2.1}, "resolve_cold_final": 50.0}}


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))


def _bench_dir(tmp_path):
    d = tmp_path / "runs" / "s01" / "nw"; d.mkdir(parents=True)
    (d / "n1000_s0.json").write_text(json.dumps(OK))
    d2 = tmp_path / "runs" / "s01" / "ridge"; d2.mkdir(parents=True)
    (d2 / "n1000_s0.json").write_text(json.dumps(BAD))
    return tmp_path / "runs"


def test_import_bench_creates_runs_and_is_idempotent(store, tmp_path):
    runs = _bench_dir(tmp_path)
    assert import_bench(store, runs) == 2
    assert import_bench(store, runs) == 0
    df = store.search("bench")
    assert set(df.status) == {"FINISHED", "FAILED"} and set(df["tags.source"]) == {"legacy_json"}
    ok = df[df.status == "FINISHED"].iloc[0]
    assert ok["metrics.pooled_rmse_bp"] == 12.0 and ok["metrics.rmse_bp/T0.25"] == 3.0
    assert ok["metrics.intraday_s"] == 0.5 and ok["params.git_hash"] == "abc"
    rid = ok["run_id"]
    err = json.loads(store.download(rid, "iv_err_bp.json", tmp_path / "d").read_text())
    assert err == [[1.0, None]]


def test_import_warm_seq_logs_steps(store, tmp_path):
    d = tmp_path / "warm"; d.mkdir()
    (d / "seq_s01_s0.json").write_text(json.dumps(SEQ))
    assert import_warm(store, d) == 1
    df = store.search("warm")
    assert df.loc[0, "params.arm"] == "seq"
    assert df.loc[0, "metrics.rmse_bp/refresh"] == 11.0          # last step wins as "latest"
    assert df.loc[0, "metrics.rmse_bp/resolve_cold_final"] == 5.0 and df.loc[0, "metrics.wall_s"] == 9.5


def test_import_all_reports_counts(store, tmp_path):
    _bench_dir(tmp_path)
    (tmp_path / "bump").mkdir(); (tmp_path / "warm").mkdir()
    counts = import_all(store, tmp_path)
    assert counts == {"bench": 2, "bump": 0, "warm": 0, "overnight": 0}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/tracking/test_importer.py -q`. Expected: ImportError.

- [ ] **Step 3: Implement**

```python
"""One-shot import of legacy JSON results into the store. Idempotent on `legacy_path`."""
import json
import re
import shutil
import tempfile
from pathlib import Path

import torch

from ..experiments.warm_suite import _log_doc
from .store import flatten_metrics, to_jsonable

TAGS = {"source": "legacy_json"}


def _rel(path):
    p = Path(path).resolve()
    try:
        return p.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return p.as_posix()


def _already(store, experiment, path):
    exp = store.experiment_id(experiment)
    runs = store.client.search_runs([exp], f"params.legacy_path = '{_rel(path)}'", max_results=1)
    return bool(runs)


def _read(p):
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError) as e:
        print(f"warning: skipping unreadable {p}: {e}")
        return None


def import_bench(store, runs_dir="results/runs"):
    n = 0
    for p in sorted(Path(runs_dir).rglob("*.json")):
        d = _read(p)
        if d is None or _already(store, "bench", p):
            continue
        params = {"sid": d["sid"], "algo": d["algo"], "n_particles": d["n_particles"], "seed": d["seed"],
                  "git_hash": d.get("git_hash", "unknown"), "schema": d.get("schema", 1), "legacy_path": _rel(p)}
        sc = d.get("scenario") or {}
        params.update({f"scenario.ssvi.{k}": v for k, v in (sc.get("ssvi") or {}).items()})
        params.update({f"scenario.dynamics.{k}": v for k, v in (sc.get("dynamics") or {}).items()})
        for k in ("s0", "T", "maturities"):
            if k in sc:
                params[f"scenario.{k}"] = str(sc[k]) if k == "maturities" else sc[k]
        if d.get("status") != "ok":
            try:
                with store.run("bench", params, TAGS):
                    raise RuntimeError(d.get("error", "legacy failure"))
            except RuntimeError:
                pass
            n += 1
            continue
        with store.run("bench", params, TAGS) as h:
            m = d.get("metrics") or {}
            metrics = {c: m.get(c) for c in ("pooled_rmse_bp", "pooled_max_bp", "wings_rmse_bp", "wings_max_bp", "n_failed")}
            metrics.update({f"rmse_bp/T{r['T']:g}": r["rmse_bp"] for r in m.get("per_maturity", [])})
            metrics.update(d.get("timings") or {})
            diag = d.get("diagnostics") or {}
            for c in ("intraday_s", "overnight_s"):
                if c in diag:
                    metrics[c] = diag[c]
            h.log_metrics(metrics)
            h.log_json("iv_err_bp.json", d.get("iv_err_bp"))
            h.log_json("diagnostics.json", diag)
        n += 1
    return n


def import_bump(store, bump_dir="results/bump"):
    n = 0
    for p in sorted(Path(bump_dir).glob("*.json")):
        d = _read(p)
        if d is None or _already(store, "bump", p):
            continue
        params = {"sid": d["sid"], "seed": d["seed"], "legacy_path": _rel(p)}
        params.update({f"bumped.{k}": v for k, v in (d.get("bumped") or {}).items()})
        with store.run("bump", params, TAGS) as h:
            h.log_metrics(flatten_metrics("rmse_bp", d["rmse_bp"]))
            h.log_metrics(flatten_metrics("build_s", d["build_s"]))
            h.log_json("result.json", d)
        n += 1
    return n


def import_warm(store, warm_dir="results/warm"):
    n = 0
    for p in sorted(Path(warm_dir).glob("*.json")):
        d = _read(p)
        if d is None or "arm" not in d or _already(store, "warm", p):
            continue
        params = {"arm": d["arm"], "sid": d["sid"], "seed": d["seed"], "legacy_path": _rel(p)}
        with store.run("warm", params, TAGS) as h:
            _log_doc(h, d["arm"], to_jsonable(d))
        n += 1
    return n


_CACHE = re.compile(r"^(?P<sid>s\d+)_s(?P<seed>\d+)_N(?P<N>\d+)_k(?P<k>\d+)\.pt$")


def import_overnight_cache(store, cache_dir="results/warm/cache"):
    n = 0
    for p in sorted(Path(cache_dir).glob("*.pt")):
        m = _CACHE.match(p.name)
        if not m or _already(store, "overnight", p):
            continue
        blob = torch.load(p, weights_only=False)
        params = {"sid": m["sid"], "seed": int(m["seed"]), "N": int(m["N"]), "n_steps": int(m["k"]),
                  "git_hash": "unknown", "legacy_path": _rel(p)}
        with store.run("overnight", params, TAGS) as h:
            with tempfile.TemporaryDirectory() as d:
                dst = Path(d) / "overnight.pt"          # warm_suite.overnight looks up this exact name
                shutil.copy(p, dst)
                h.log_file(dst)
            h.log_metrics({"overnight_s": float(blob.get("overnight_s", 0.0))})
        n += 1
    return n


def import_all(store, root="results"):
    root = Path(root)
    return {"bench": import_bench(store, root / "runs"), "bump": import_bump(store, root / "bump"),
            "warm": import_warm(store, root / "warm"),
            "overnight": import_overnight_cache(store, root / "warm" / "cache")}
```

- [ ] **Step 4: Run tests, then import the real archive, commit**

Run: `uv run pytest tests/tracking -q`. Expected: all pass.
Run the real import from the repo root: `uv run python -c "from neural_particle_method.tracking.store import Store; from neural_particle_method.tracking.importer import import_all; print(import_all(Store()))"`.
Expected: `{'bench': 876, 'bump': 10, 'warm': 25, 'overnight': 12}` or close (the digest says 876 runs; report the exact numbers). Then `uv run python -c "from neural_particle_method.tracking.store import Store; from neural_particle_method.bench.aggregate import aggregate; aggregate(Store(), 'results/summary_from_store.csv', 'results/digest_from_store.md')"` and diff: `python - <<'EOF'` comparing `results/summary.csv` and `results/summary_from_store.csv` sorted by `(sid, algo, n_particles, seed)` on the metric columns with `pandas.testing.assert_frame_equal(check_like=True)` after dropping `total_s`, `fit_s`, `intraday_s`, `git_hash` if they differ only by float formatting. Expected: identical metric columns. Delete the two `_from_store` files afterwards.

```bash
git add -A src/neural_particle_method/tracking tests/tracking
git commit -m "feat: idempotent importer for legacy bench, bump, warm, and overnight-cache results"
```

---

### Task 11: `figures/` inside the package; fig2 reads artifacts

**Files:**
- Move: `figures/fig1_accuracy.py`, `fig3_plane.py`, `fig4_latency.py` -> `src/neural_particle_method/figures/` unchanged except the signature
- Create: `src/neural_particle_method/figures/__init__.py`, rewrite `fig2_wings.py`
- Move: `tests/test_figures.py` -> `tests/figures/test_figures.py`
- Top-level `figures/` package: delete now (only `bench/cli.py figures` used it; update that command to call the package)

**Interfaces:**
- Produces: every figure module exposes `make(summary_csv, store, outdir) -> str`. `figures.ALL = (fig1_accuracy, fig2_wings, fig3_plane, fig4_latency)`. `fig2_wings.profile(store, algo, n_particles=200_000) -> ndarray | None` averages `abs(iv_err_bp[-1])` over FINISHED bench runs with `params.algo == algo`, `params.n_particles == n_particles`, and `sid` not starting with `f_`.

- [ ] **Step 1: Port the figure test**

`tests/figures/__init__.py` empty; `tests/figures/test_figures.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

from neural_particle_method.figures import ALL, fig1_accuracy as f1, fig2_wings as f2
from neural_particle_method.tracking.store import Store


def _summary(tmp_path):
    rows = []
    for sid in ("s01", "s02", "f_xi0.3_rho-0.7", "f_xi0.6_rho-0.7"):
        for algo in ("nw", "explicit_nn", "ridge", "explicit_nn_is", "implicit_nn"):
            rows.append(dict(sid=sid, algo=algo, n_particles=50_000, seed=0, status="ok", git_hash="x",
                             pooled_rmse_bp=10.0, pooled_max_bp=20.0, wings_rmse_bp=15.0, wings_max_bp=25.0,
                             n_failed=0, total_s=5.0, fit_s=1.0))
    p = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _store(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'db'}", str(tmp_path / "art"))
    for sid in ("s01", "f_xi0.3_rho-0.7"):
        for algo in ("nw", "explicit_nn", "explicit_nn_is"):
            with store.run("bench", {"sid": sid, "algo": algo, "n_particles": 200_000, "seed": 0}) as h:
                h.log_json("iv_err_bp.json", [[float(i - 6) for i in range(13)]] * 4)
    return store


@pytest.mark.parametrize("mod", ALL)
def test_each_figure_writes_pdf(tmp_path, mod):
    out = mod.make(str(_summary(tmp_path)), _store(tmp_path), str(tmp_path / "out"))
    assert Path(out).exists() and out.endswith(".pdf")


def test_fig2_profile_excludes_stress_scenarios(tmp_path):
    prof = f2.profile(_store(tmp_path), "explicit_nn")
    assert prof.shape == (13,) and prof[0] == 6.0


def test_fig1_is_byte_identical_across_runs(tmp_path):
    summary, store = _summary(tmp_path), _store(tmp_path)
    a = Path(f1.make(str(summary), store, str(tmp_path / "a"))).read_bytes()
    b = Path(f1.make(str(summary), store, str(tmp_path / "b"))).read_bytes()
    assert a == b and b"/CreationDate" not in a
```

- [ ] **Step 2: Run to verify failure, then implement**

`git mv figures/fig1_accuracy.py src/neural_particle_method/figures/` and the same for fig3, fig4; rename their second parameter to `store` (unused). New `fig2_wings.py`:

```python
"""Fig 2: wing IV error with and without the tilt, from per-run iv_err_bp artifacts."""
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def profile(store, algo, n_particles=200_000):
    """Mean |IV error| on the longest maturity over non-stress scenarios at one particle budget."""
    df = store.search("bench", f"params.algo = '{algo}' and params.n_particles = '{n_particles}' "
                               f"and attributes.status = 'FINISHED'")
    errs = []
    with tempfile.TemporaryDirectory() as d:
        for _, r in df.iterrows():
            if str(r.get("params.sid", "")).startswith("f_"):
                continue
            p = store.download(r["run_id"], "iv_err_bp.json", Path(d) / r["run_id"])
            e = np.array(json.loads(p.read_text()), dtype=float)
            errs.append(np.abs(e[-1]))
    return np.nanmean(np.stack(errs), axis=0) if errs else None


def make(summary_csv, store, outdir, n_particles=200_000):
    k = np.log(np.geomspace(0.6, 1.6, 13))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for algo, label in (("explicit_nn", "no tilt"), ("explicit_nn_is", "defensive mixture")):
        prof = profile(store, algo, n_particles)
        if prof is not None:
            ax.plot(k, prof, marker="o", label=label)
    ax.set_xlabel("log-moneyness"); ax.set_ylabel("|IV error| (bp), longest maturity")
    ax.legend()
    out = Path(outdir) / "fig2_wings.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, metadata={"CreationDate": None}); plt.close(fig)
    return str(out)
```

`json.loads` turns the stored `null` into `None`; `np.array(..., dtype=float)` makes it NaN, matching the old behaviour. `figures/__init__.py`:

```python
from . import fig1_accuracy, fig2_wings, fig3_plane, fig4_latency

ALL = (fig1_accuracy, fig2_wings, fig3_plane, fig4_latency)
```

Delete the top-level `figures/__init__.py` (the four `fig*.py` were moved with `git mv`); keep `figures/out/`, which holds the committed PDFs. Update `bench/cli.py figures` to iterate `neural_particle_method.figures.ALL` with a `Store()`. Remove `"figures"` from `[tool.hatch.build.targets.wheel] packages` and add nothing yet (Task 12 rewrites it). Delete `tests/test_figures.py`.

- [ ] **Step 3: Run tests and regenerate the paper figures**

Run: `uv run pytest -q`. Expected: all pass.
Run: `uv run bench figures` from the repo root. Expected: four PDFs rewritten under `figures/out/`. `git status` should show `figures/out/fig1_accuracy.pdf`, `fig3`, `fig4` unchanged (byte-identical, since summary.csv is unchanged) and `fig2_wings.pdf` unchanged too (same data via artifacts). If fig2 differs, inspect the profile against the old JSON-based `_profile` on the same runs before continuing.

```bash
git add -A src/neural_particle_method/figures figures tests bench pyproject.toml
git commit -m "refactor: figures inside the package; fig2 reads iv_err_bp artifacts from the store"
```

---

### Task 12: `nparticle` CLI, delete shims and legacy packages, docs

**Files:**
- Create: `src/neural_particle_method/cli.py`, `tests/test_cli.py`
- Delete: top-level `bench/`, `experiments/bump_correct.py`, `experiments/warm_suite.py`, all shim modules in `src/neural_particle_method/` (`bs.py`, `heston.py`, `ssvi.py`, `dupire.py`, `condexp.py`, `readouts.py`, `explicit.py`, `implicit.py`, `importance.py`, `reprice.py`)
- Modify: `pyproject.toml`, `README.md`, `src/neural_particle_method/__init__.py`

**Interfaces:**
- Produces: `cli.main(argv=None) -> int` with subcommands: `list`; `run --scenario --algo --n --seed [--n-steps] [--reprice-n] [--reprice-steps]`; `sweep --preset paper [--jobs K]`; `aggregate [--out results/summary.csv] [--digest results/digest.md]`; `figures [--summary results/summary.csv] [--outdir figures/out]`; `experiment {bump,warm} [--smoke] [--arms ...] [--sids ...]`; `warm-summary`; `import-legacy [--root results]`. Global option `--tracking-uri` (default from env or SQLite in cwd).

- [ ] **Step 1: Write the CLI test**

`tests/test_cli.py`:

```python
from neural_particle_method.cli import main
from tests.conftest import TINY_N, TINY_REPRICE_N, TINY_REPRICE_STEPS


def test_list(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out.lower()
    assert "scenarios" in out and "implicit_ridge" in out


def test_run_then_skip_then_aggregate(tmp_path, capsys):
    uri = ["--tracking-uri", f"sqlite:///{tmp_path / 'db'}", "--artifact-root", str(tmp_path / "art")]
    args = uri + ["run", "--scenario", "s01", "--algo", "nw", "--n", str(TINY_N), "--seed", "0",
                  "--n-steps", "6", "--reprice-n", str(TINY_REPRICE_N), "--reprice-steps", str(TINY_REPRICE_STEPS)]
    assert main(args) == 0
    assert "run_id" in capsys.readouterr().out
    assert main(args) == 0
    assert "skip" in capsys.readouterr().out.lower()
    assert main(uri + ["aggregate", "--out", str(tmp_path / "s.csv"), "--digest", str(tmp_path / "d.md")]) == 0
    assert (tmp_path / "s.csv").exists()


def test_import_legacy_on_empty_root(tmp_path):
    uri = ["--tracking-uri", f"sqlite:///{tmp_path / 'db'}", "--artifact-root", str(tmp_path / "art")]
    (tmp_path / "results").mkdir()
    assert main(uri + ["import-legacy", "--root", str(tmp_path / "results")]) == 0
```

- [ ] **Step 2: Implement `cli.py`**

```python
"""nparticle: benchmark, experiments, aggregation, figures, and legacy import over the MLflow store."""
import argparse
from dataclasses import replace
from pathlib import Path

from .bench.aggregate import aggregate
from .bench.algos import ALGOS
from .bench.runner import BENCH_EXPERIMENT, run_key, run_one
from .bench.scenarios import full_registry
from .bench.sweep import paper_grid, sweep
from .calibrate.config import ExplicitConfig, ImplicitConfig
from .experiments.bump_correct import run_pair
from .experiments.config import BUMP_FULL, BUMP_SMOKE, FULL, SMOKE
from .experiments.warm_suite import ARMS, DEFAULT_SIDS, run_warm, summarise
from .figures import ALL as FIGURES
from .pricing.reprice import RepriceConfig
from .tracking.importer import import_all
from .tracking.store import Store


def _parser():
    ap = argparse.ArgumentParser(prog="nparticle")
    ap.add_argument("--tracking-uri", default=None)
    ap.add_argument("--artifact-root", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("run")
    p.add_argument("--scenario", required=True); p.add_argument("--algo", required=True, choices=list(ALGOS))
    p.add_argument("--n", type=int, required=True); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--reprice-n", type=int, default=500_000); p.add_argument("--reprice-steps", type=int, default=200)
    p = sub.add_parser("sweep"); p.add_argument("--preset", default="paper", choices=["paper"]); p.add_argument("--jobs", type=int, default=1)
    p = sub.add_parser("aggregate"); p.add_argument("--out", default="results/summary.csv"); p.add_argument("--digest", default="results/digest.md")
    p = sub.add_parser("figures"); p.add_argument("--summary", default="results/summary.csv"); p.add_argument("--outdir", default="figures/out")
    p = sub.add_parser("experiment"); p.add_argument("which", choices=["bump", "warm"]); p.add_argument("--smoke", action="store_true")
    p.add_argument("--arms", nargs="*", default=list(ARMS)); p.add_argument("--sids", nargs="*", default=list(DEFAULT_SIDS))
    sub.add_parser("warm-summary")
    p = sub.add_parser("import-legacy"); p.add_argument("--root", default="results")
    return ap


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.cmd == "list":
        reg = full_registry()
        print(f"scenarios ({len(reg)}):")
        for sid, sc in reg.items():
            d = sc.dynamics
            print(f"  {sid}: sigma0={sc.ssvi.sigma0:.3f} xi={d.xi} rho={d.rho}")
        print("algos:", ", ".join(ALGOS))
        return 0
    store = Store(args.tracking_uri, args.artifact_root)
    if args.cmd == "run":
        e = ExplicitConfig() if args.n_steps is None else replace(ExplicitConfig(), n_steps=args.n_steps)
        i = ImplicitConfig() if args.n_steps is None else replace(ImplicitConfig(), n_steps=args.n_steps)
        if store.find_finished(BENCH_EXPERIMENT, run_key(args.scenario, args.algo, args.n, args.seed)):
            print("skip (finished run exists)")
            return 0
        rid = run_one(store, args.scenario, args.algo, args.n, args.seed, e, i,
                      RepriceConfig(args.reprice_n, args.reprice_steps))
        print(f"run_id {rid}")
        return 0
    if args.cmd == "sweep":
        sweep(store, paper_grid(), n_jobs=args.jobs)
        return 0
    if args.cmd == "aggregate":
        df = aggregate(store, args.out, args.digest)
        print(f"{len(df)} runs -> {args.out}")
        return 0
    if args.cmd == "figures":
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        for mod in FIGURES:
            print("wrote", mod.make(args.summary, store, args.outdir))
        return 0
    if args.cmd == "experiment":
        from .bench.scenarios import make_registry
        if args.which == "bump":
            cfg = BUMP_SMOKE if args.smoke else BUMP_FULL
            sids = ["s01"] if args.smoke else args.sids
            for sid in sids:
                for seed in ((0,) if args.smoke else (0, 1)):
                    print("run_id", run_pair(store, make_registry()[sid], seed, cfg))
        else:
            cfg = SMOKE if args.smoke else FULL
            run_warm(store, args.arms, ["s01"] if args.smoke else args.sids, cfg)
        return 0
    if args.cmd == "warm-summary":
        md = summarise(store)
        Path("results/warm_summary.md").write_text(md)
        print(md)
        return 0
    if args.cmd == "import-legacy":
        print(import_all(store, args.root))
        return 0
    return 1
```

- [ ] **Step 3: Delete legacy code and shims; update packaging and docs**

`git rm -r bench figures/__init__.py experiments/bump_correct.py experiments/warm_suite.py` (keep `experiments/results/` archive and `figures/out/`). `git rm` the ten shim modules under `src/neural_particle_method/`. Grep the tree for any remaining `from neural_particle_method.explicit`, `.implicit`, `.condexp`, `.readouts`, `.reprice`, `.ssvi`, `.bs`, `.heston`, `.dupire`, `.importance`, `from bench`, `import figures`: expected none outside `paper/` and docs.

`pyproject.toml`:

```toml
[project.scripts]
nparticle = "neural_particle_method.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/neural_particle_method"]
```

`src/neural_particle_method/__init__.py`:

```python
"""Neural particle calibration of LSV models: market models, simulation, estimators, calibration,
pricing, and an MLflow-backed benchmark and experiment harness."""
```

`README.md` (replace the whole file):

```markdown
# neural_particle_method

Neural L2 calibration of LSV models (Risk paper workspace).

## Layout

- `src/neural_particle_method/market` — Black-Scholes, Heston, SSVI, Dupire local vol
- `simulate` — Heston-plus-leverage Euler step and the `LeverageField`
- `estimators` — conditional-expectation estimators behind one `fit_predict` interface
- `calibrate` — explicit and implicit schemes, mixture importance sampling, warm-start helpers
- `pricing` — fresh-seed repricing and IV metrics
- `tracking` — MLflow store wrapper and the legacy importer
- `bench`, `experiments`, `figures` — the paper's benchmark, the bump and warm-start suites, figures 1-4
- Paper notes: `paper/`; specs and plans under `docs/superpowers/`.

## Tracking

All runs go to MLflow. Default store: `sqlite:///mlruns.db` and `mlartifacts/` in the working directory
(both gitignored). Set `MLFLOW_TRACKING_URI` to use a server. Browse with `uv run mlflow ui --backend-store-uri sqlite:///mlruns.db`.

## Commands

- `uv run nparticle list`
- `uv run nparticle run --scenario s01 --algo ridge --n 200000 --seed 1`
- `uv run nparticle sweep --preset paper --jobs 4` (resumable: finished runs are skipped)
- `uv run nparticle aggregate` -> `results/summary.csv`, `results/digest.md` (committed snapshots)
- `uv run nparticle figures` -> `figures/out/`
- `uv run nparticle experiment bump|warm [--smoke]`; `uv run nparticle warm-summary`
- `uv run nparticle import-legacy` (one-shot import of the archived JSON results under `results/`)

## Tests

- `uv run pytest` — fast suite (about 30 s)
- `uv run pytest -m golden` — bit-for-bit replay of four archived runs (minutes)
- `uv run pytest -m slow` — smoke runs of every experiment arm
- `uv run pytest --cov=neural_particle_method` — coverage report
```

- [ ] **Step 4: Run tests, goldens, and the CLI end to end; commit**

Run: `uv sync` (the script entry point changed), `uv run pytest -q`, `uv run pytest -m golden -q`, then `uv run nparticle list`, `uv run nparticle aggregate`, `uv run nparticle figures`, and check `git status` shows `results/summary.csv`, `results/digest.md`, and the four PDFs unchanged.

```bash
git add -A
git commit -m "feat: nparticle CLI; remove shims and legacy top-level packages; README"
```

---

### Task 13: Ruff clean and coverage

**Files:**
- Modify: whatever ruff reports; `pyproject.toml` if a rule needs a targeted ignore

- [ ] **Step 1: Run ruff and fix**

Run: `uv run ruff check . --fix` then `uv run ruff check .`. Fix remaining findings by hand: unused unpacked variables (`RUF059`) by replacing with `_`; `subprocess.run` without `check` by adding `check=False`; `B008` on `cfg=ExplicitConfig()` defaults is acceptable for frozen dataclasses, so add to `pyproject.toml`:

```toml
[tool.ruff.lint]
extend-ignore = ["B008"]
```

only if ruff flags it; otherwise leave the config untouched. Do not touch any numerical expression to satisfy a lint rule; add a `# noqa` with the rule code instead.

- [ ] **Step 2: Verify nothing changed numerically**

Run: `uv run pytest -q` and `uv run pytest -m golden -q`. Expected: green.

- [ ] **Step 3: Coverage report and commit**

Run: `uv run pytest --cov=neural_particle_method --cov-report=term-missing -q`. Paste the total line into the commit body.

```bash
git add -A
git commit -m "chore: ruff clean; coverage reporting"
```

---

## Verification checklist (end of plan)

- [ ] `uv run pytest -q` green, under about 30 s.
- [ ] `uv run pytest -m golden -q` green (4 archived replays).
- [ ] `uv run pytest -m slow -q` green (experiment smokes).
- [ ] `uv run ruff check .` clean.
- [ ] `uv run nparticle aggregate` regenerates `results/summary.csv` with identical metric columns to the committed file.
- [ ] `uv run nparticle figures` regenerates byte-identical PDFs.
- [ ] No module outside `tracking/store.py` imports `mlflow` (`grep -rn "import mlflow\|from mlflow" src` shows only `tracking/store.py`).
