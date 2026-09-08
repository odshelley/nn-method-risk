"""nparticle: benchmark, experiments, aggregation, figures, and legacy import over the MLflow store."""
import argparse
from dataclasses import replace
from pathlib import Path

from .bench.acceptance import run_acceptance
from .bench.aggregate import aggregate
from .bench.algos import ALGOS
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
    p = sub.add_parser("acceptance"); p.add_argument("--cards", nargs="*", default=None)
    p = sub.add_parser("aggregate"); p.add_argument("--out", default="results/summary.csv"); p.add_argument("--digest", default="results/digest.md")
    p.add_argument("--experiment", default="bench")
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
        run_sensitivity(store, jobs, n_jobs=args.jobs)
        return 0
    if args.cmd == "acceptance":
        out = run_acceptance(store, names=args.cards or None)
        return 0 if all(ok for _, ok, _, _ in out) else 1
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
    return 1
