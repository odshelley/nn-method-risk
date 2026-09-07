"""Heston dynamics parameters."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HestonParams:
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float

    @classmethod
    def from_dict(cls, d):
        if isinstance(d, cls):
            return d
        return cls(kappa=float(d["kappa"]), theta=float(d["theta"]), xi=float(d["xi"]),
                   rho=float(d["rho"]), v0=float(d["v0"]))

    def to_dict(self):
        return asdict(self)
