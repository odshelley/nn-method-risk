"""CLI for the calibration benchmark."""
import argparse
from dataclasses import replace
from pathlib import Path

from neural_particle_method.bench.aggregate import aggregate as _aggregate
from neural_particle_method.bench.algos import ALGOS
from neural_particle_method.bench.runner import BENCH_EXPERIMENT, run_key, run_one
from neural_particle_method.bench.scenarios import full_registry
from neural_particle_method.bench.sweep import paper_grid, sweep as _sweep
from neural_particle_method.calibrate.config import ExplicitConfig, ImplicitConfig
import neural_particle_method.figures as figures
from neural_particle_method.pricing.reprice import RepriceConfig
from neural_particle_method.tracking.store import Store


def _add_store_args(p):
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--artifact-root", default=None)


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
    p.add_argument("--reprice-steps", type=int, default=200)
    _add_store_args(p)
    p = sub.add_parser("sweep")
    p.add_argument("--preset", default="paper", choices=["paper"])
    p.add_argument("--jobs", type=int, default=1)
    _add_store_args(p)
    p = sub.add_parser("aggregate")
    p.add_argument("--out", default="results/summary.csv")
    _add_store_args(p)
    p = sub.add_parser("figures")
    p.add_argument("--summary", default="results/summary.csv")
    p.add_argument("--outdir", default="figures/out")
    _add_store_args(p)
    args = ap.parse_args(argv)

    if args.cmd == "list":
        print(f"scenarios ({len(full_registry())}):")
        for sid, sc in full_registry().items():
            d = sc.dynamics
            print(f"  {sid}: sigma0={sc.ssvi.sigma0:.3f} xi={d.xi} rho={d.rho}")
        print("algos:", ", ".join(ALGOS))
        return 0
    if args.cmd == "run":
        explicit = replace(ExplicitConfig(), n_steps=args.n_steps) if args.n_steps else ExplicitConfig()
        implicit = replace(ImplicitConfig(), n_steps=args.n_steps) if args.n_steps else ImplicitConfig()
        reprice = RepriceConfig(args.reprice_n, args.reprice_steps)
        store = Store(args.tracking_uri, args.artifact_root)
        key = run_key(args.scenario, args.algo, args.n, args.seed)
        if store.find_finished(BENCH_EXPERIMENT, key) is not None:
            print("skip (finished run exists)")
            return 0
        rid = run_one(store, args.scenario, args.algo, args.n, args.seed,
                      explicit=explicit, implicit=implicit, reprice=reprice)
        print(f"wrote run {rid}")
        return 0
    if args.cmd == "sweep":
        store = Store(args.tracking_uri, args.artifact_root)
        n = _sweep(store, paper_grid(), n_jobs=args.jobs)
        print(f"{n} runs executed")
        return 0
    if args.cmd == "aggregate":
        store = Store(args.tracking_uri, args.artifact_root)
        df = _aggregate(store, args.out)
        print(f"{len(df)} runs -> {args.out}")
        return 0
    if args.cmd == "figures":
        store = Store(args.tracking_uri, args.artifact_root)
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        for mod in figures.ALL:
            print("wrote", mod.make(args.summary, store, args.outdir))
        return 0
