"""CLI for the calibration benchmark."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from neural_particle_method.calibrate.config import ExplicitConfig, ImplicitConfig

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
    p.add_argument("--reprice-steps", type=int, default=200)
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
        explicit = replace(ExplicitConfig(), n_steps=args.n_steps) if args.n_steps else ExplicitConfig()
        implicit = replace(ImplicitConfig(), n_steps=args.n_steps) if args.n_steps else ImplicitConfig()
        out = run_path(args.scenario, args.algo, args.n, args.seed, args.results_dir)
        if out.exists():
            print(f"skip (exists): {out}")
            return 0
        p = run_one(args.scenario, args.algo, args.n, args.seed,
                    results_dir=args.results_dir, explicit=explicit, implicit=implicit,
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
        import figures.fig1_accuracy, figures.fig2_wings, figures.fig3_plane, figures.fig4_latency
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        for mod in (figures.fig1_accuracy, figures.fig2_wings,
                    figures.fig3_plane, figures.fig4_latency):
            print("wrote", mod.make(args.summary, args.runs_dir, args.outdir))
        return 0
