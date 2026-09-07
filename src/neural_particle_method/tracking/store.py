"""MLflow-backed run store. This is the only module in the package that imports mlflow."""
import contextlib
import json
import logging
import os
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient

# MLflow's SQLite backend runs alembic schema migrations on first open of a
# database; alembic and sqlalchemy log INFO/WARNING noise to the root logger
# by default. Quiet them at the source rather than via pytest config.
logging.getLogger("alembic").setLevel(logging.WARNING)
logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)


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
        try:
            h.log_params(params)
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

    def search(self, experiment, filter_string="", max_results=50_000):
        runs = self.client.search_runs([self.experiment_id(experiment)], filter_string, max_results=max_results)
        rows = []
        for r in runs:
            row = {"run_id": r.info.run_id, "status": r.info.status, "start_time": r.info.start_time}
            row.update({f"params.{k}": v for k, v in r.data.params.items()})
            row.update({f"metrics.{k}": v for k, v in r.data.metrics.items()})
            row.update({f"tags.{k}": v for k, v in r.data.tags.items()})
            rows.append(row)
        if not rows:
            return pd.DataFrame(rows, columns=["run_id", "status", "start_time"])
        return pd.DataFrame(rows)
