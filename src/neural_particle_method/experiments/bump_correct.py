"""Bump-and-correct: intraday strategies on a bumped surface from an overnight implicit body."""
import dataclasses
import time

import numpy as np

from ..bench.scenarios import quote_k_grid
from ..calibrate.config import ExplicitConfig
from ..calibrate.explicit import calibrate_explicit
from ..calibrate.implicit import calibrate_implicit, simulate_slices
from ..calibrate.warm import beta_correction, bump, distil, records_from_betas
from ..estimators import make_estimator
from ..estimators.ridge import GlobalRidge
from ..market.local_vol import SSVILocalVol
from ..tracking.store import flatten_metrics, git_hash
from .warm_suite import score

BUMP_EXPERIMENT = "bump"


def run_pair(store, sc, seed, cfg):
    """Bump-and-correct strategies on the overnight body for one scenario/seed. Returns the run id,
    skipping recomputation if a FINISHED run for {"sid", "seed"} already exists."""
    key = {"sid": sc.sid, "seed": seed}
    rid = store.find_finished(BUMP_EXPERIMENT, key)
    if rid is not None:
        return rid

    dyn, s0, T, mats, kq = sc.dynamics, sc.s0, sc.T, list(sc.maturities), quote_k_grid()
    v0 = dyn.v0
    N, n_steps, sub, _, _ = cfg.N, cfg.n_steps, cfg.sub, cfg.n_iters, cfg.fit_steps
    lvS = SSVILocalVol(sc.ssvi, s0, T_max=T)
    pB = bump(sc.ssvi)
    lvB = SSVILocalVol(pB, s0, T_max=T)
    res, times = {}, {}

    params = {**key, **cfg.as_params(),
              **{f"bumped.{k}": v for k, v in dataclasses.asdict(pB).items()},
              "git_hash": git_hash()}
    with store.run(BUMP_EXPERIMENT, params) as h:
        t0 = time.perf_counter()
        w = calibrate_explicit(lvS, dyn, make_estimator("nn", seed=seed),
                               ExplicitConfig(n_steps=n_steps, n_particles=N), s0=s0, T=T, seed=seed)
        r = calibrate_implicit(lvS, dyn, cfg.implicit, s0=s0, T=T, seed=seed, L0=w.field)
        net = r.net
        rng = np.random.default_rng(seed + 50)
        slices = simulate_slices(r.field, dyn, s0, T, n_steps, N, rng)
        betas0, recs_b0 = distil(net, slices, lvS, T, n_steps, s0, v0, sub, rng)
        times["overnight"] = time.perf_counter() - t0
        res["anchor_unbumped_implicit"] = score(r.field, dyn, s0, mats, kq, sc.ssvi, seed + 900, cfg.reprice)
        res["anchor_unbumped_distilled"] = score(recs_b0, dyn, s0, mats, kq, sc.ssvi, seed + 900, cfg.reprice)

        # kept verbatim from the legacy script; the first t0 is unused
        t0 = time.perf_counter()
        res["stale_L"] = score(r.field, dyn, s0, mats, kq, pB, seed + 901, cfg.reprice)
        times["stale_L"] = 0.0

        t0 = time.perf_counter()
        recs_js = records_from_betas(net, betas0, lvB, T, n_steps, s0, v0)
        times["stale_f_fresh_sigma"] = time.perf_counter() - t0
        res["stale_f_fresh_sigma"] = score(recs_js, dyn, s0, mats, kq, pB, seed + 902, cfg.reprice)

        t0 = time.perf_counter()
        head = GlobalRidge(net, T)
        e = calibrate_explicit(lvB, dyn, head,
                               ExplicitConfig(n_steps=n_steps, n_particles=N, fit_subsample=sub),
                               s0=s0, T=T, seed=seed + 1)
        times["causal_sweep"] = time.perf_counter() - t0
        res["causal_sweep"] = score(e.field, dyn, s0, mats, kq, pB, seed + 903, cfg.reprice)

        b = dict(betas0)
        for i in (1, 2):
            t0 = time.perf_counter()
            b = beta_correction(net, b, lvB, dyn, T, n_steps, N, s0, v0, sub, seed + 10 + i)
            times[f"beta_corr_{i}"] = (time.perf_counter() - t0
                                       + (times.get(f"beta_corr_{i-1}", 0.0) if i > 1 else 0.0))
            res[f"beta_corr_{i}"] = score(records_from_betas(net, b, lvB, T, n_steps, s0, v0),
                                          dyn, s0, mats, kq, pB, seed + 904 + i, cfg.reprice)

        t0 = time.perf_counter()
        w2 = calibrate_explicit(lvB, dyn, make_estimator("nn", seed=seed + 2),
                                ExplicitConfig(n_steps=n_steps, n_particles=N), s0=s0, T=T, seed=seed + 2)
        r2 = calibrate_implicit(lvB, dyn, cfg.implicit, s0=s0, T=T, seed=seed + 2, L0=w2.field)
        times["full_resolve"] = time.perf_counter() - t0
        res["full_resolve"] = score(r2.field, dyn, s0, mats, kq, pB, seed + 907, cfg.reprice)

        doc = {"sid": sc.sid, "seed": seed, "rmse_bp": res,
               "build_s": {k: round(v, 2) for k, v in times.items()},
               "bumped": dataclasses.asdict(pB)}
        h.log_metrics(flatten_metrics("rmse_bp", res))
        h.log_metrics(flatten_metrics("build_s", doc["build_s"]))
        h.log_json("result.json", doc)
        rid = h.run_id
    return rid
