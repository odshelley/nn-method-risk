"""Neural-net conditional-expectation estimator: warm-startable per-slice regressor."""
import torch
from torch import nn

V_SCALE = 0.04  # rough variance scale for output conditioning
Z_SCALE = 0.3   # rough log-spot scale for input conditioning


class SliceNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, z):
        return nn.functional.softplus(self.head(self.body(z))) * V_SCALE


class NNRegressor:
    """Warm-startable per-slice regressor for E[V | ln X = .].

    An instance is meant for exactly one `calibrate_explicit` pass: the step budget is chosen
    per instance (`first_steps` on the first `fit_predict` call, `later_steps` on every call
    after), not per calibration. Every caller constructs a fresh regressor, so this is
    equivalent to the pre-refactor per-calibration choice; reusing one instance across two
    passes gives the SECOND pass's first slice `later_steps` instead of `first_steps`. Call
    `reset()` before a new pass if you must reuse an instance.
    """
    supports_weights = True

    def __init__(self, seed=0, first_steps=400, later_steps=120):
        torch.manual_seed(seed)
        self.net = SliceNet()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-2)
        self.first_steps, self.later_steps = first_steps, later_steps
        self._n_fits = 0

    def reset(self):
        """Zero the fit counter so the next `fit_predict` call again uses `first_steps`."""
        self._n_fits = 0

    def fit(self, lnx, v, steps=120, weights=None):
        z = torch.tensor(lnx[:, None] / Z_SCALE, dtype=torch.float32)
        tv = torch.tensor(v[:, None], dtype=torch.float32)
        tw = None if weights is None else torch.tensor(weights[:, None], dtype=torch.float32)
        for _ in range(steps):
            self.opt.zero_grad()
            resid = (self.net(z) - tv) ** 2
            loss = (resid if tw is None else tw * resid).mean()
            loss.backward()
            self.opt.step()
        return float(loss.detach())

    def predict(self, lnx_grid):
        with torch.no_grad():
            z = torch.tensor(lnx_grid[:, None] / Z_SCALE, dtype=torch.float32)
            return self.net(z).numpy()[:, 0]

    def fit_predict(self, t, lnx, v, grid, weights=None):
        steps = self.first_steps if self._n_fits == 0 else self.later_steps
        self._n_fits += 1
        self.fit(lnx, v, steps=steps, weights=weights)
        return self.predict(grid)
