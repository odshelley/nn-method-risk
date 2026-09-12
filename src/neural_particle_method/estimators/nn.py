"""Neural-net conditional-expectation estimator: warm-startable per-slice regressor."""
import numpy as np
import torch
from torch import nn

from .nadaraya_watson import nw_local_variance

V_SCALE = 0.04  # rough variance scale for output conditioning
Z_SCALE = 0.3   # rough log-spot scale for input conditioning
HETERO_W_MAX = 1e3  # cap on a normalised heteroscedastic weight


class SliceNet(nn.Module):
    def __init__(self, hidden=64, depth=2):
        super().__init__()
        layers = []
        in_dim = 1
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)
        # a post-hoc multiplier (mean matching); a buffer, not a parameter, so training is unchanged
        self.register_buffer("log_scale", torch.zeros(1))

    def forward(self, z):
        return nn.functional.softplus(self.head(self.body(z))) * V_SCALE * torch.exp(self.log_scale)


class NNRegressor:
    """Warm-startable per-slice regressor for E[V | ln X = .].

    An instance is meant for exactly one `calibrate_explicit` pass: the step budget is chosen
    per instance (`first_steps` on the first `fit_predict` call, `later_steps` on every call
    after), not per calibration. Every caller constructs a fresh regressor, so this is
    equivalent to the pre-refactor per-calibration choice; reusing one instance across two
    passes gives the SECOND pass's first slice `later_steps` instead of `first_steps`. Call
    `reset()` before a new pass if you must reuse an instance.

    Priors (all off by default, every default bit-for-bit with the plain regressor):
    `weight_decay` (Adam L2), `warm_start=False` re-initialises the network before every slice
    and trains it `first_steps` steps, `mean_match` rescales the fitted slice so its weighted
    mean over the fit sample equals the weighted mean of the targets, `monotone_penalty` adds
    lambda * mean(relu(monotone_sign * df/dz))^2 / V_SCALE^2 to the loss (penalising slopes of
    sign `monotone_sign`), `hetero` weights the loss by 1 / local variance of the target,
    normalised to mean one and capped at `HETERO_W_MAX`. `mean_match` is skipped when the
    weighted target mean is not positive: there is no scale to match there.
    """
    supports_weights = True

    def __init__(self, seed=0, first_steps=400, later_steps=120, hidden=64,
                 keep_slice_weights=False, depth=2, lr=1e-2, batch_size=None, weight_decay=0.0,
                 warm_start=True, mean_match=False, monotone_penalty=0.0, monotone_sign=1.0,
                 hetero=False):
        self.seed, self.hidden, self.depth, self.lr = seed, hidden, depth, lr
        self.weight_decay = weight_decay
        self._build(seed)
        self.first_steps, self.later_steps = first_steps, later_steps
        self._n_fits = 0
        self.keep_slice_weights = keep_slice_weights
        self.slice_weights = []   # [(t, state_dict copy)] per fitted slice when keep_slice_weights
        self.batch_size = batch_size
        self._gen = torch.Generator().manual_seed(seed)
        self.warm_start, self.mean_match, self.hetero = warm_start, mean_match, hetero
        self.monotone_penalty, self.monotone_sign = monotone_penalty, monotone_sign
        self.last_penalty, self.last_weights = 0.0, None

    def _build(self, seed):
        torch.manual_seed(seed)
        self.net = SliceNet(self.hidden, self.depth)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)

    def reset(self):
        """Zero the fit counter so the next `fit_predict` call again uses `first_steps`."""
        self._n_fits = 0

    def fit(self, lnx, v, steps=120, weights=None):
        z = torch.tensor(lnx[:, None] / Z_SCALE, dtype=torch.float32)
        tv = torch.tensor(v[:, None], dtype=torch.float32)
        tw = None if weights is None else torch.tensor(weights[:, None], dtype=torch.float32)
        n = z.shape[0]
        pen = torch.zeros(())
        for _ in range(steps):
            if self.batch_size is None:
                zb, tvb, twb = z, tv, tw
            else:
                idx = torch.randint(0, n, (self.batch_size,), generator=self._gen)
                zb, tvb = z[idx], tv[idx]
                twb = None if tw is None else tw[idx]
            self.opt.zero_grad()
            if self.monotone_penalty > 0:
                zb = zb.clone().requires_grad_(True)
                out = self.net(zb)
                slope = torch.autograd.grad(out.sum(), zb, create_graph=True)[0]
                pen = (torch.relu(self.monotone_sign * slope) ** 2).mean() / V_SCALE ** 2
            else:
                out = self.net(zb)
            resid = (out - tvb) ** 2
            loss = (resid if twb is None else twb * resid).mean() + self.monotone_penalty * pen
            loss.backward()
            self.opt.step()
        self.last_penalty = float(pen.detach())
        return float(loss.detach())

    def predict(self, lnx_grid):
        with torch.no_grad():
            z = torch.tensor(lnx_grid[:, None] / Z_SCALE, dtype=torch.float32)
            return self.net(z).numpy()[:, 0]

    def fit_predict(self, t, lnx, v, grid, weights=None):
        if not self.warm_start and self._n_fits > 0:
            self._build(self.seed + self._n_fits)
        steps = self.first_steps if (self._n_fits == 0 or not self.warm_start) else self.later_steps
        self._n_fits += 1
        w = weights
        if self.hetero:
            wh = 1.0 / nw_local_variance(lnx, v, weights=weights)
            wh = wh / wh.mean()
            # a region where the target is almost noiseless sends 1 / var to the 1e-8 floor's
            # reciprocal and would drown every other particle; cap the normalised weight there
            wh = np.clip(wh, None, HETERO_W_MAX)
            w = wh if weights is None else weights * wh
        self.last_weights = w
        self.fit(lnx, v, steps=steps, weights=w)
        if self.mean_match:
            pred = self.predict(lnx)
            wm = np.ones_like(v) if weights is None else weights
            target_mean = float(np.average(v, weights=wm))
            # a non-positive weighted target mean has no scale to match: the ratio would be zero
            # or negative and `log_scale` would collapse the slice. Leave the fit alone.
            if target_mean > 0.0:
                ratio = target_mean / max(np.average(pred, weights=wm), 1e-12)
                with torch.no_grad():
                    self.net.log_scale += float(np.log(max(ratio, 1e-12)))
        if self.keep_slice_weights:
            snap = {k: v_.detach().clone() for k, v_ in self.net.state_dict().items()}
            self.slice_weights.append((float(t), snap))
        return self.predict(grid)
