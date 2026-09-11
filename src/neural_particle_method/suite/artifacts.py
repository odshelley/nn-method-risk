"""Trained objects as MLflow artifacts, and their reload.

model.pt (torch): {"kind": ..., "state": ...}; model_meta.json: caller metadata plus "kind".
"""
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..calibrate.implicit import GlobalNet
from ..estimators.nn import Z_SCALE, NNRegressor, SliceNet
from ..simulate.leverage import LeverageField


def _slice_index(times, t):
    return max(int(np.searchsorted(times, t + 1e-12)) - 1, 0)


def _body_depth(body):
    return sum(1 for m in body if isinstance(m, torch.nn.Linear))


class SliceBank:
    """Per-slice explicit networks; slice in force at t is the last time <= t (as
    LeverageField.at)."""

    def __init__(self, times, nets):
        self.times, self.nets = np.asarray(times, dtype=float), list(nets)

    @classmethod
    def from_regressor(cls, est):
        nets = []
        depth = _body_depth(est.net.body)
        for _, sd in est.slice_weights:
            n = SliceNet(est.net.body[0].out_features, depth)
            n.load_state_dict(sd)
            nets.append(n)
        return cls([t for t, _ in est.slice_weights], nets)

    def _net(self, t):
        return self.nets[_slice_index(self.times, t)]

    def _z(self, lnx):
        return torch.tensor(np.asarray(lnx, dtype=float)[:, None] / Z_SCALE, dtype=torch.float32)

    def f(self, t, lnx):
        with torch.no_grad():
            return self._net(t)(self._z(lnx)).numpy()[:, 0]

    def features(self, t, lnx):
        with torch.no_grad():
            phi = self._net(t).body(self._z(lnx)).numpy()
        return np.concatenate([phi, np.ones((len(phi), 1))], axis=1)

    def readout(self, t):
        h = self._net(t).head
        return np.concatenate([h.weight.detach().numpy()[0], h.bias.detach().numpy()])

    def state(self):
        return {"times": self.times.tolist(), "hidden": self.nets[0].body[0].out_features,
                "depth": _body_depth(self.nets[0].body),
                "state_dicts": [n.state_dict() for n in self.nets]}

    @classmethod
    def from_state(cls, d):
        nets = []
        depth = d.get("depth", 2)
        for sd in d["state_dicts"]:
            n = SliceNet(d["hidden"], depth)
            n.load_state_dict(sd)
            nets.append(n)
        return cls(d["times"], nets)


class GlobalNetModel:
    """The implicit scheme's global network as a (t, x) model with a 65-feature body."""

    def __init__(self, net, T):
        self.net, self.T = net, float(T)

    def _tz(self, t, lnx):
        x = np.asarray(lnx, dtype=float)
        return torch.tensor(
            np.stack([np.full(len(x), t / max(self.T, 1e-9)), x / Z_SCALE], axis=1),
            dtype=torch.float32,
        )

    def f(self, t, lnx):
        with torch.no_grad():
            return self.net(self._tz(t, lnx)).numpy()[:, 0]

    def features(self, t, lnx):
        with torch.no_grad():
            phi = self.net.body(self._tz(t, lnx)).numpy()
        return np.concatenate([phi, np.ones((len(phi), 1))], axis=1)

    def readout(self, t):
        h = self.net.head
        return np.concatenate([h.weight.detach().numpy()[0], h.bias.detach().numpy()])

    def state(self):
        return {"T": self.T, "hidden": self.net.body[0].out_features,
                "state_dict": self.net.state_dict()}

    @classmethod
    def from_state(cls, d):
        net = GlobalNet(d["hidden"])
        net.load_state_dict(d["state_dict"])
        return cls(net, d["T"])


def model_kind(model):
    if isinstance(model, NNRegressor) and model.slice_weights:
        return "explicit_slices"
    if isinstance(model, GlobalNet):
        return "implicit_net"
    if isinstance(model, SliceBank):
        return "explicit_slices"
    if isinstance(model, GlobalNetModel):
        return "implicit_net"
    return "field_only"


def _to_model(model, T):
    if isinstance(model, NNRegressor):
        return SliceBank.from_regressor(model)
    if isinstance(model, GlobalNet):
        return GlobalNetModel(model, T)
    return model


def save_model(h, model, meta):
    """Log model.pt and model_meta.json to run handle `h`.

    `meta` must carry "T" for implicit nets.
    """
    kind = model_kind(model)
    doc = {**meta, "kind": kind}
    if kind != "field_only":
        obj = _to_model(model, meta.get("T", 1.0))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.pt"
            torch.save({"kind": kind, "state": obj.state()}, p)
            h.log_file(p)
    h.log_json("model_meta.json", doc)
    return kind


@dataclass
class LoadedRun:
    run_id: str
    params: dict
    metrics: dict
    field: LeverageField
    model: object
    meta: dict


def load_run(store, run_id):
    """Rebuild a suite run's field and trained object from its artifacts alone."""
    with tempfile.TemporaryDirectory() as d:
        leverage_text = store.download(run_id, "leverage.json", d).read_text()
        field = LeverageField.from_json(json.loads(leverage_text))
        names = {a.path for a in store.client.list_artifacts(run_id)}
        meta = (json.loads(store.download(run_id, "model_meta.json", d).read_text())
                if "model_meta.json" in names else {})
        model = None
        if "model.pt" in names:
            blob = torch.load(store.download(run_id, "model.pt", d), weights_only=True)
            cls = SliceBank if blob["kind"] == "explicit_slices" else GlobalNetModel
            model = cls.from_state(blob["state"])
    return LoadedRun(
        run_id, store.get_params(run_id), store.get_metrics(run_id), field, model, meta
    )
