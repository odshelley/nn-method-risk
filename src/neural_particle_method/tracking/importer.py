"""One-shot import of legacy JSON results into the store. Idempotent on `legacy_path`."""
import json
import re
import shutil
import tempfile
from pathlib import Path

import torch

from ..experiments.warm_suite import log_warm_doc
from .store import flatten_metrics, repo_root, to_jsonable

TAGS = {"source": "legacy_json"}


def _rel(path):
    p = Path(path).resolve()
    try:
        return p.relative_to(repo_root()).as_posix()
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
        if d is None or "arm" not in d:
            continue
        # legacy smoke-test runs (filename prefixed "smoke_", arm field itself is bare)
        # go to a separate experiment so they never collide with genuine runs sharing
        # the same (arm, sid, seed) params, and never leak into the paper summary.
        is_smoke = p.name.startswith("smoke_")
        experiment = "warm_smoke" if is_smoke else "warm"
        if _already(store, experiment, p):
            continue
        params = {"arm": d["arm"], "sid": d["sid"], "seed": d["seed"], "legacy_path": _rel(p)}
        tags = {**TAGS, "smoke": "true"} if is_smoke else TAGS
        with store.run(experiment, params, tags) as h:
            log_warm_doc(h, d["arm"], to_jsonable(d))
        n += 1
    return n


_CACHE = re.compile(r"^(?P<sid>s\d+)_s(?P<seed>\d+)_N(?P<N>\d+)_k(?P<k>\d+)\.pt$")


def import_overnight_cache(store, cache_dir="results/warm/cache"):
    n = 0
    for p in sorted(Path(cache_dir).glob("*.pt")):
        m = _CACHE.match(p.name)
        if not m:
            print(f"warning: skipping unrecognised cache file {p}")
            continue
        if _already(store, "overnight", p):
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
