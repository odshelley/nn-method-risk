"""Card acceptance runs: reproduce a source number on the Heston market (see BASELINES.md)."""
import json
import tempfile
from dataclasses import dataclass, field

import numpy as np

from ..calibrate.config import ExplicitConfig, ImplicitConfig
from ..pricing.reprice import RepriceConfig
from .runner import run_one

ACCEPTANCE_EXPERIMENT = "acceptance"


@dataclass(frozen=True)
class Card:
    name: str
    sid: str
    algo: str
    n_particles: int
    seed: int
    knobs: dict = field(default_factory=dict)
    source_value: float = 0.0
    tolerance: float = 0.3
    metric: str = "avg_abs_pct"     # avg_abs_pct | pooled_rmse_bp | delta_vs_nw_ghl_bp


# Source numbers: Li (2023) Table 2.4 h0 rows (kernel, Heston market) 1.44 / 1.18 %; Table 3.2 (RKHS,
# overrides L=40, variance 5, lam 1e-7) 0.88 / 0.70 %; Table 4.2 (bins, l=20) 0.91 / 1.01 %.
# Muguruza: within 10 bp pooled RMSE of nw_ghl at N=5e4 (his "performs at the level of the particle method").
# PURBF: no numeric benchmark exists in Hakala (2019); the first accepted run is frozen as the target (spec).
# Bayer Figure 5 (ATM error flat for lam in [1e-9, 1e-5] on the `bayer` market) is a curve, not a single run:
# it is produced by `nparticle sensitivity --sids bayer --algos rkhs` and read off fig5, not asserted here.
CARDS = (
    Card("nw_ghl_li_simple", "li_simple", "nw_ghl", 100_000, 0, {"fixed_scale": 1.0}, 1.44),
    Card("nw_ghl_li_complex", "li_complex", "nw_ghl", 100_000, 0, {"fixed_scale": 1.0}, 1.18),
    Card("muguruza_li_simple", "li_simple", "muguruza", 50_000, 0, {}, 0.0, 10.0, "delta_vs_nw_ghl_bp"),
    Card("rkhs_li_simple", "li_simple", "rkhs", 100_000, 0, {"n_centres": 40, "variance": 5.0, "lam": 1e-7}, 0.88),
    Card("rkhs_li_complex", "li_complex", "rkhs", 100_000, 0, {"n_centres": 40, "variance": 5.0, "lam": 1e-7}, 0.70),
    Card("bins_li_simple", "li_simple", "bins", 100_000, 0, {}, 0.91),
    Card("bins_li_complex", "li_complex", "bins", 100_000, 0, {}, 1.01),
    Card("purbf_li_simple", "li_simple", "purbf", 2_048, 0, {}, 1.44),     # provisional target: beat the kernel row
)


def _avg_abs_pct(store, rid):
    with tempfile.TemporaryDirectory() as d:
        err = np.array(json.loads(store.download(rid, "iv_err_bp.json", d).read_text()), dtype=float)
    return float(np.nanmean(np.abs(err[-1])) / 100.0)


def check(store, card, explicit=ExplicitConfig(), implicit=ImplicitConfig(), reprice=RepriceConfig()):
    rid = run_one(store, card.sid, card.algo, card.n_particles, card.seed, explicit, implicit, reprice,
                  experiment=ACCEPTANCE_EXPERIMENT, knobs=card.knobs, extra_key={"card": card.name})
    if card.metric == "avg_abs_pct":
        achieved = _avg_abs_pct(store, rid)
        return achieved <= card.source_value + card.tolerance, achieved, rid
    if card.metric == "pooled_rmse_bp":
        achieved = store.get_metrics(rid)["pooled_rmse_bp"]
        return achieved <= card.source_value + card.tolerance, achieved, rid
    if card.metric == "delta_vs_nw_ghl_bp":
        ref = run_one(store, card.sid, "nw_ghl", card.n_particles, card.seed, explicit, implicit, reprice,
                      experiment=ACCEPTANCE_EXPERIMENT, extra_key={"card": f"ref_nw_ghl_{card.sid}_{card.n_particles}"})
        achieved = store.get_metrics(rid)["pooled_rmse_bp"] - store.get_metrics(ref)["pooled_rmse_bp"]
        return achieved <= card.tolerance, achieved, rid
    raise KeyError(card.metric)


def run_acceptance(store, names=None, explicit=ExplicitConfig(), implicit=ImplicitConfig(), reprice=RepriceConfig()):
    out = []
    for card in CARDS:
        if names and card.name not in names:
            continue
        ok, achieved, rid = check(store, card, explicit, implicit, reprice)
        print(f"{card.name}: {'PASS' if ok else 'FAIL'} achieved={achieved:.4g} source={card.source_value} run={rid}", flush=True)
        out.append((card.name, ok, achieved, rid))
    return out
