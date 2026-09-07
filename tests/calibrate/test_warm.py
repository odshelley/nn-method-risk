import numpy as np
import torch

from neural_particle_method.calibrate.implicit import GlobalNet, simulate_slices
from neural_particle_method.calibrate.warm import (bump, distil, dyn_variants, records_from_betas,
                                                    scaled_bump, seq_path)
from neural_particle_method.market.local_vol import SSVILocalVol
from neural_particle_method.market.ssvi import SSVIParams, no_arb_ok
from neural_particle_method.simulate.dynamics import HestonParams
from neural_particle_method.simulate.leverage import DEFAULT_GRID, LeverageField, Slice

P = SSVIParams(sigma0=0.2, eta=1.0, gamma=0.4, rho=-0.6)
DYN = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.6, v0=0.04)


def test_bumps_stay_arbitrage_free():
    assert no_arb_ok(bump(P))
    q, eff = scaled_bump(P, 2.0)
    assert no_arb_ok(q) and 0 < eff <= 2.0
    path = seq_path(P, 4, np.random.default_rng(0))
    assert len(path) == 4 and all(no_arb_ok(q) for q in path)


def test_dyn_variants_returns_heston_params():
    d = dyn_variants(DYN)
    assert set(d) == {"xi_up", "rho_dn", "kappa_dn"}
    assert d["xi_up"].xi == DYN.xi * 1.3 and d["kappa_dn"].kappa == DYN.kappa * 0.6


def test_distil_then_records_round_trip():
    torch.manual_seed(0)
    net = GlobalNet()
    lv = SSVILocalVol(P, T_max=0.5)
    n_steps, T = 4, 0.5
    flat = LeverageField([Slice(k * T / n_steps, DEFAULT_GRID, np.ones(len(DEFAULT_GRID)),
                                np.full(len(DEFAULT_GRID), 0.04)) for k in range(n_steps)])
    rng = np.random.default_rng(0)
    slices = simulate_slices(flat, DYN, 1.0, T, n_steps, 2_000, rng)
    betas, field = distil(net, slices, lv, T, n_steps, 1.0, DYN.v0, 1_000, rng)
    assert set(betas) == {1, 2, 3} and len(field) == n_steps
    rebuilt = records_from_betas(net, betas, lv, T, n_steps, 1.0, DYN.v0)
    for a, b in zip(field, rebuilt):
        np.testing.assert_array_equal(a.L, b.L)
        np.testing.assert_array_equal(a.f, b.f)
