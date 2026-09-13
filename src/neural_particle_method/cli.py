"""nparticle: benchmark, experiments, aggregation, figures, and legacy import over the MLflow store."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import torch

from .bench.acceptance import run_acceptance
from .bench.aggregate import aggregate
from .bench.algos import ALGOS
from .bench.reference_runs import run_reference
from .bench.runner import BENCH_EXPERIMENT, run_key, run_one
from .bench.scenarios import full_registry
from .bench.sensitivity import KNOBS, run_sensitivity, sensitivity_grid
from .bench.sweep import baselines_grid, paper_grid, sweep
from .calibrate.config import ExplicitConfig, ImplicitConfig
from .experiments.bump_correct import run_pair
from .experiments.config import BUMP_FULL, BUMP_SMOKE, FULL, SMOKE
from .experiments.warm_suite import ARMS, DEFAULT_SIDS, run_warm, summarise
from .figures import ALL as FIGURES
from .pricing.reprice import RepriceConfig
from .suite.artifacts import load_run
from .suite.config import FULL as SUITE_FULL
from .suite.config import SMOKE as SUITE_SMOKE
from .suite.grid import STAGES, run_stage
from .suite.optuna_search import TRIAL_TIMEOUT_S
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
    p = sub.add_parser("sweep"); p.add_argument("--preset", default="paper", choices=["paper", "baselines"]); p.add_argument("--jobs", type=int, default=1)
    p = sub.add_parser("sensitivity")
    p.add_argument("--sids", nargs="*", default=["s01", "li_simple"])
    p.add_argument("--budgets", type=int, nargs="*", default=[10_000, 100_000])
    p.add_argument("--algos", nargs="*", default=list(KNOBS))
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--n-steps", type=int, default=None)
    p = sub.add_parser("acceptance"); p.add_argument("--cards", nargs="*", default=None)
    p = sub.add_parser("reference")
    p.add_argument("--sids", nargs="*", default=["li_simple"])
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--n-x", type=int, default=801)
    p.add_argument("--n-v", type=int, default=200)
    p = sub.add_parser("aggregate"); p.add_argument("--out", default="results/summary.csv"); p.add_argument("--digest", default="results/digest.md")
    p.add_argument("--experiment", default="bench")
    p = sub.add_parser("figures"); p.add_argument("--summary", default="results/summary.csv"); p.add_argument("--outdir", default="figures/out")
    p = sub.add_parser("experiment"); p.add_argument("which", choices=["bump", "warm"]); p.add_argument("--smoke", action="store_true")
    p.add_argument("--arms", nargs="*", default=list(ARMS)); p.add_argument("--sids", nargs="*", default=list(DEFAULT_SIDS))
    sub.add_parser("warm-summary")
    p = sub.add_parser("import-legacy"); p.add_argument("--root", default="results")
    p = sub.add_parser("suite")
    ss = p.add_subparsers(dest="suite_cmd", required=True)
    q = ss.add_parser("run")
    q.add_argument("--stage", required=True, choices=list(STAGES) + ["all"])
    q.add_argument("--jobs", type=int, default=1); q.add_argument("--sids", nargs="*", default=None)
    q.add_argument("--smoke", action="store_true")
    q = ss.add_parser("load"); q.add_argument("run_id"); q.add_argument("--out", default=None)
    q = ss.add_parser("tables"); q.add_argument("--out", default="paper/tables")
    p = sub.add_parser("optuna")
    os_ = p.add_subparsers(dest="optuna_cmd", required=True)
    q = os_.add_parser("clouds"); q.add_argument("--jobs", type=int, default=1)
    q.add_argument("--sids", nargs="*", default=None)
    q = os_.add_parser("run"); q.add_argument("--study", required=True)
    q.add_argument("--trials", type=int, required=True)
    q.add_argument("--jobs", type=int, default=1)
    q.add_argument("--per-trial", type=int, default=8)
    q.add_argument("--trial-timeout", type=float, default=TRIAL_TIMEOUT_S,
                   help="wall-clock cap on one trial, in seconds (0 for none)")
    q.add_argument("--timeout-hours", type=float, default=None,
                   help="wall-clock cap on the whole study, in hours")
    q = os_.add_parser("validate"); q.add_argument("--study", required=True)
    q.add_argument("--top", type=int, default=3); q.add_argument("--jobs", type=int, default=1)
    q.add_argument("--budget", type=int, default=80_000)
    q = os_.add_parser("promote"); q.add_argument("--study", required=True)
    q.add_argument("--trial", type=int, required=True)
    p = sub.add_parser("tilt")
    ts = p.add_subparsers(dest="tilt_cmd", required=True)
    for name in ("slices", "cold", "online"):
        q = ts.add_parser(name)
        q.add_argument("--jobs", type=int, default=1)
        q.add_argument("--sids", nargs="*", default=None)
        q.add_argument("--design", default=None); q.add_argument("--smoke", action="store_true")
        q.add_argument("--budgets", nargs="*", type=int, default=None)
        q.add_argument("--seeds", nargs="*", type=int, default=None)
    q = ts.add_parser("winner"); q.add_argument("--smoke", action="store_true")
    q = ts.add_parser("tables"); q.add_argument("--design", default=None)
    q.add_argument("--smoke", action="store_true"); q.add_argument("--out", default="paper/tables")
    q = ts.add_parser("figures"); q.add_argument("--design", default=None)
    q.add_argument("--smoke", action="store_true"); q.add_argument("--out", default="figures/out")
    return ap


def _cloud_worker(args):
    """One tuning cloud in its own process (the clouds are independent full offline passes)."""
    from .suite.optuna_clouds import ensure_cloud
    torch.set_num_threads(1)   # one BLAS thread per worker; the pool is the parallelism
    uri, root, sid = args
    return ensure_cloud(Store(uri, root), sid)


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.cmd == "list":
        reg = full_registry()
        print(f"scenarios ({len(reg)}):")
        for sid, sc in reg.items():
            d = sc.dynamics
            if sc.family == "heston":
                print(f"  {sid}: heston market xi={sc.market.xi} rho={sc.market.rho}; dynamics rho={sc.dynamics.rho}")
            else:
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
        if args.preset == "baselines":
            sweep(store, baselines_grid(), n_jobs=args.jobs, experiment="baselines")
        else:
            sweep(store, paper_grid(), n_jobs=args.jobs)
        return 0
    if args.cmd == "sensitivity":
        jobs = sensitivity_grid(sids=args.sids, budgets=args.budgets, algos=args.algos)
        e = ExplicitConfig() if args.n_steps is None else replace(ExplicitConfig(), n_steps=args.n_steps)
        i = ImplicitConfig() if args.n_steps is None else replace(ImplicitConfig(), n_steps=args.n_steps)
        run_sensitivity(store, jobs, n_jobs=args.jobs, explicit=e, implicit=i)
        return 0
    if args.cmd == "acceptance":
        out = run_acceptance(store, names=args.cards or None)
        return 1 if any(status == "WORSE" for _, status, _, _ in out) else 0
    if args.cmd == "reference":
        for sid in args.sids:
            rid = run_reference(store, sid, n_steps=args.n_steps, n_x=args.n_x, n_v=args.n_v)
            print(f"{sid}: run_id {rid}")
        return 0
    if args.cmd == "aggregate":
        df = aggregate(store, args.out, args.digest, experiment=args.experiment)
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
                blob = {"kind": lr.meta.get("kind"), "state": lr.model.state()}
                torch.save(blob, out / "model.pt")
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
    if args.cmd == "optuna":
        from .suite.config import TUNING_SIDS
        from .suite.optuna_search import run_study
        from .suite.optuna_validate import promote, validate
        if args.optuna_cmd == "clouds":
            sids = tuple(args.sids or TUNING_SIDS)
            from .suite.optuna_clouds import EXPERIMENT as CLOUD_EXPERIMENT
            store.experiment_id(CLOUD_EXPERIMENT)   # pre-create: workers must not race on it
            work = [(store.tracking_uri, store.artifact_root, s) for s in sids]
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                for sid, rid in zip(sids, ex.map(_cloud_worker, work)):
                    print("cloud", sid, rid, flush=True)
            return 0
        if args.optuna_cmd == "run":
            hours = args.timeout_hours
            print("study", run_study(store, args.study, args.trials, n_jobs=args.jobs,
                                     per_trial=args.per_trial,
                                     trial_timeout_s=args.trial_timeout,
                                     study_timeout_s=None if hours is None else hours * 3600))
            return 0
        if args.optuna_cmd == "validate":
            validate(store, args.study, top=args.top, jobs=args.jobs, budget=args.budget)
            return 0
        if args.optuna_cmd == "promote":
            print("wrote", promote(store, args.study, args.trial))
            return 0
    if args.cmd == "tilt":
        from .calibrate.importance import UNTILTED
        from .suite.tilt import SMOKE_GRID, run_tilt_stage
        from .suite.tilt_figures import make_figures
        from .suite.tilt_tables import slice_frame, tilt_tables, winner
        settings = SUITE_SMOKE if args.smoke else SUITE_FULL
        grid = SMOKE_GRID if args.smoke else {}

        def chosen():
            """The design to act on; None when the slice layer picked no tilt (a null result)."""
            d = getattr(args, "design", None)
            if d is not None:
                return d
            d = winner(slice_frame(store, settings))
            if d == UNTILTED:
                print("no tilt design beat untilted in the slice layer; "
                      "pass --design explicitly", flush=True)
                return None
            return d

        if args.tilt_cmd == "winner":
            print(winner(slice_frame(store, settings)))
            return 0
        if args.tilt_cmd in ("slices", "cold", "online"):
            design = None
            if args.tilt_cmd != "slices":
                design = chosen()
                if design is None:
                    return 1
            sids = args.sids if args.sids is not None else grid.get("sids")
            particles = (settings.n_online,) if args.smoke else None
            seeds = args.seeds if args.seeds is not None else grid.get("seeds")
            budgets = args.budgets if args.budgets is not None else (
                (settings.n_online,) if args.smoke else None)
            done, failed = run_tilt_stage(store, args.tilt_cmd, settings, n_jobs=args.jobs,
                                          sids=sids, design=design, particles=particles,
                                          seeds=seeds, designs=grid.get("designs"),
                                          budgets=budgets)
            print(f"tilt {args.tilt_cmd}: {done} done, {failed} failed", flush=True)
            return 1 if failed else 0
        if args.tilt_cmd == "tables":
            design = chosen()
            if design is None:
                return 1
            for pth in tilt_tables(store, design, args.out, settings):
                print("wrote", pth)
            return 0
        if args.tilt_cmd == "figures":
            design = chosen()
            if design is None:
                return 1
            kw = ({"sids": ("s01",), "n_particles": settings.n_online, "times": (0.5, 1.0)}
                  if args.smoke else {})
            for pth in make_figures(store, design, args.out, settings, **kw):
                print("wrote", pth)
            return 0
    return 1
