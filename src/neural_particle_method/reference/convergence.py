"""Grid-refinement study for docs/pde_reference.md. Run: uv run python -m neural_particle_method.reference.convergence

Three tables, IV errors in vol bp on the quote grid (13 strikes in [0.6, 1.6]):
  1. pure Heston (L = 1) vs `heston_call` at the Li parameters: PDE error alone, over (n_x, n_v);
  2. LSV calibrated to the Li Heston market (Dupire surface from `heston_call`) with different
     dynamics, over n_steps: the frozen-leverage scheme's O(dt) bias on top of the PDE error;
  3. SSVI scenario s01 with the analytic local vol, per maturity, over n_steps.
"""
import sys
import time

import numpy as np

from ..bench.scenarios import make_registry, quote_k_grid
from ..market.bs import implied_vol
from ..market.dupire import DupireSurface
from ..market.heston import heston_call
from ..market.local_vol import SSVILocalVol
from ..market.ssvi import implied_vol_ssvi
from ..pricing.reprice import snap_times
from ..simulate.dynamics import HestonParams
from .fokker_planck import FokkerPlanck
from .pde import call_prices, default_v_grid, default_x_grid, implied_vols, solve_leverage_pde

LI = HestonParams(kappa=1.5768, theta=0.0484, xi=0.5751, rho=-0.7, v0=0.1024)
CAL = HestonParams(kappa=1.0, theta=0.06, xi=0.35, rho=-0.5, v0=0.08)
K_GRID = quote_k_grid()


def _stats(err_bp):
    return f"{np.sqrt(np.mean(err_bp ** 2)):5.2f} | {np.abs(err_bp).max():5.2f}"


def pure_heston(params, T, nx, nv, n_steps=50, n_sub=2):
    fp = FokkerPlanck(default_x_grid(nx), default_v_grid(params, T, nv), params)
    m, A, dt = fp.initial(0.0, params.v0), fp.operator(1.0), T / n_steps
    t0 = time.perf_counter()
    for k in range(n_steps):
        m = fp.advance(m, A, dt, n_sub, first=(k == 0))
    el = time.perf_counter() - t0
    K = np.exp(K_GRID)
    iv = np.array([implied_vol(p, 1.0, kk, T) for p, kk in
                   zip(call_prices(fp.x, fp.marginal(m), K, dx=fp.dx), K)])
    ex = heston_call(K, T, params.v0, params.kappa, params.theta, params.xi, params.rho)
    iv_ex = np.array([implied_vol(p, 1.0, kk, T) for p, kk in zip(ex, K)])
    return (iv - iv_ex) * 1e4, el


def table_pure_heston(out):
    out("## 1. Pure Heston, Li parameters, T = 1: PDE error alone (rms | max, vol bp)\n")
    out("| n_x (h) | n_v | n_steps x n_sub | rms | max | runtime s |")
    out("|---|---|---|---|---|---|")
    for nx, nv, nst, nsub in ((201, 50, 50, 2), (401, 100, 50, 2), (801, 200, 50, 2),
                              (801, 200, 100, 2), (801, 200, 50, 8), (1601, 400, 50, 2)):
        e, el = pure_heston(LI, 1.0, nx, nv, nst, nsub)
        h = 6.0 / (nx - 1)
        out(f"| {nx} ({h:.4f}) | {nv} | {nst} x {nsub} | {_stats(e)} | {el:.0f} |")
        sys.stdout.flush()


def table_lsv(out):
    T = 1.0

    def price(K, t):
        return heston_call(K, t, LI.v0, LI.kappa, LI.theta, LI.xi, LI.rho)

    surf = DupireSurface.from_price_fn(price, 1.0, np.linspace(0.01, T, 100),
                                       np.linspace(np.log(0.25), np.log(3.0), 161))
    K = np.exp(K_GRID)
    iv_mkt = np.array([implied_vol(p, 1.0, kk, T) for p, kk in zip(price(K, T), K)])
    out("\n## 2. LSV calibrated to the Li Heston market, T = 1 (rms | max, vol bp)\n")
    out("Dynamics `cal`: kappa=1, theta=0.06, xi=0.35, rho=-0.5, v0=0.08. `Li`: calibrated model = market, "
        "so the exact leverage is 1.\n")
    out("| dynamics | n_x | n_v | n_steps | rms | max | forward - 1 | runtime s |")
    out("|---|---|---|---|---|---|---|---|")
    for label, params in (("cal", CAL), ("Li", LI)):
        for nx, nv, nst in ((401, 100, 50), (801, 200, 50), (801, 200, 100), (801, 200, 200), (801, 200, 400)):
            res = solve_leverage_pde(surf, params, T=T, n_steps=nst, x_grid=default_x_grid(nx),
                                     v_grid=default_v_grid(params, T, nv))
            e = (implied_vols(res.x_grid, res.density, K_GRID, T, dx=res.dx) - iv_mkt) * 1e4
            out(f"| {label} | {nx} | {nv} | {nst} | {_stats(e)} | {res.forward - 1:+.1e} | {res.runtime_s:.0f} |")
            sys.stdout.flush()


def table_ssvi(out):
    sc = make_registry()["s01"]
    lv, mats = SSVILocalVol(sc.ssvi), list(sc.maturities)
    out("\n## 3. SSVI scenario s01 (analytic Dupire local vol), T = 2, per snapped maturity (rms | max, vol bp)\n")
    out("| n_x | n_v | n_steps | T=0.25 | T=0.5 | T=1 | T=2 | runtime s |")
    out("|---|---|---|---|---|---|---|---|")
    for nx, nv, nst in ((401, 100, 50), (801, 200, 50), (801, 200, 200), (801, 200, 800)):
        ts = snap_times(mats, nst, sc.T)
        res = solve_leverage_pde(lv, sc.dynamics, T=sc.T, n_steps=nst, x_grid=default_x_grid(nx),
                                 v_grid=default_v_grid(sc.dynamics, sc.T, nv), snapshot_times=ts)
        cells = []
        for t in ts:
            e = (implied_vols(res.x_grid, res.snapshots[t], K_GRID, t, dx=res.dx)
                 - implied_vol_ssvi(sc.ssvi, K_GRID, t)) * 1e4
            cells.append(_stats(e))
        out(f"| {nx} | {nv} | {nst} | " + " | ".join(cells) + f" | {res.runtime_s:.0f} |")
        sys.stdout.flush()


def main():
    def out(line=""):
        print(line, flush=True)
    which = sys.argv[1:] or ["heston", "lsv", "ssvi"]
    if "heston" in which:
        table_pure_heston(out)
    if "lsv" in which:
        table_lsv(out)
    if "ssvi" in which:
        table_ssvi(out)


if __name__ == "__main__":
    main()
