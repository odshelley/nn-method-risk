import numpy as np

from neural_particle_method.simulate.dynamics import HestonParams
from neural_particle_method.simulate.stepper import heston_step

P = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)


def test_step_matches_hand_computation():
    lnx = np.array([0.0]); v = np.array([0.04]); L = np.array([1.5])
    zb = np.array([0.5]); zp = np.array([-1.0])
    dt, sdt = 0.01, 0.1
    x1, v1 = heston_step(lnx, v, L, zb, zp, P, dt, sdt)
    z1 = P.rho * zb + np.sqrt(1 - P.rho ** 2) * zp
    exp_x = lnx + (-0.5 * L ** 2 * 0.04) * dt + L * 0.2 * sdt * z1
    exp_v = v + P.kappa * (P.theta - 0.04) * dt + P.xi * 0.2 * sdt * zb
    np.testing.assert_array_equal(x1, exp_x)
    np.testing.assert_array_equal(v1, exp_v)


def test_negative_variance_is_floored_in_coefficients():
    lnx = np.zeros(1); v = np.array([-0.01]); L = np.ones(1)
    x1, v1 = heston_step(lnx, v, L, np.ones(1), np.ones(1), P, 0.01, 0.1)
    assert x1[0] == 0.0                      # sqrt(max(v,0)) = 0 kills the diffusion
    assert v1[0] == -0.01 + P.kappa * P.theta * 0.01


def test_tilt_adds_drift_only():
    lnx = np.zeros(2); v = np.full(2, 0.04); L = np.ones(2)
    zb = np.zeros(2); zp = np.zeros(2)
    theta_p = np.array([0.0, 2.0])
    x1, v1 = heston_step(lnx, v, L, zb, zp, P, 0.01, 0.1, theta_p=theta_p)
    assert x1[1] - x1[0] == (1.0 * 0.2 * 2.0) * 0.01
    np.testing.assert_array_equal(v1, np.full(2, 0.04))
