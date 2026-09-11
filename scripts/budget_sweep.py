"""Small-online-budget sweep: a frozen offline body (500k) with online heads against NW re-solved
at the same budget, for online budgets of 10k, 30k and 80k particles, both lags, two seeds. Every
fit uses the full online cloud. The cell itself lives in `neural_particle_method.suite.budget`.

The default body is the tuned per-slice network (`explicit_tuned`); `--body explicit` is the
legacy short-trained body; `--body explicit_opt` trains the body from the recipe named by
`--recipe` (default `explicit_opt`) and keys its cells by the recipe hash.

Usage: python scripts/budget_sweep.py [--jobs N] [--body BODY] [--sids s01 ...] [--recipe NAME]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

from neural_particle_method.estimators.recipes import load_recipe
from neural_particle_method.suite.budget import (
    DEFAULT_BODY,
    EXPERIMENTS,
    SIDS,
    budget_jobs,
    run_budget_cell,
)
from neural_particle_method.suite.lag import LAGS
from neural_particle_method.tracking.store import Store


def _worker(args):
    uri, root, job, body, recipe = args
    lag = {lg.kind: lg for lg in LAGS}[job[3]]
    try:
        run_budget_cell(Store(uri, root), job[0], job[1], job[2], lag, job[4], body,
                        recipe=recipe)
        return job, None
    except Exception as e:  # noqa: BLE001
        return job, repr(e)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="online-budget sweep on a frozen offline body")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--body", choices=sorted(EXPERIMENTS), default=DEFAULT_BODY)
    p.add_argument("--sids", nargs="+", default=list(SIDS))
    p.add_argument("--recipe", default="explicit_opt", help="recipe name for --body explicit_opt")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    store = Store()
    store.experiment_id(EXPERIMENTS[a.body])
    recipe = load_recipe(a.recipe) if a.body == "explicit_opt" else None
    todo = budget_jobs(tuple(a.sids), a.body)
    print(f"{len(todo)} cells on the {a.body} body", flush=True)
    args = [(store.tracking_uri, store.artifact_root, j, a.body, recipe) for j in todo]
    done = failed = 0
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for job, err in ex.map(_worker, args):
            if err is None:
                done += 1
                print("done:", *job, flush=True)
            else:
                failed += 1
                print("FAILED:", *job, err, flush=True)
    print(f"budget: {done} done, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
