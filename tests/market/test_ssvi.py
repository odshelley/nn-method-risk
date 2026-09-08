import numpy as np
import QuantLib as ql

from neural_particle_method.market.local_vol import SSVILocalVol
from neural_particle_method.market.ssvi import (
    SSVIParams,
    implied_vol_ssvi,
    no_arb_ok,
    total_variance,
)

GOOD = SSVIParams(sigma0=0.2, eta=1.0, gamma=0.4, rho=-0.6)

def test_no_arb_rejects_bad_params():
    assert no_arb_ok(GOOD)
    assert not no_arb_ok(SSVIParams(0.2, eta=1.5, gamma=0.4, rho=-0.6))  # eta(1+|rho|)=2.4>2
    assert not no_arb_ok(SSVIParams(0.2, eta=1.0, gamma=0.7, rho=-0.6))  # gamma>1/2

def test_atm_total_variance():
    T = 1.3
    w = total_variance(GOOD, np.array([0.0]), T)
    assert np.isclose(w[0], GOOD.sigma0**2 * T)

def test_skew_sign():
    iv = implied_vol_ssvi(GOOD, np.array([-0.2, 0.0, 0.2]), 1.0)
    assert iv[0] > iv[1]  # negative rho: put wing above ATM

def test_local_vol_matches_quantlib():
    lv = SSVILocalVol(GOOD)
    today = ql.Date(1, 9, 2026)
    ql.Settings.instance().evaluationDate = today
    dc, cal = ql.Actual365Fixed(), ql.NullCalendar()
    ts = np.arange(0.1, 1.81, 0.02)
    ks = np.arange(-0.5, 0.501, 0.02)
    dates = [today + ql.Period(int(round(t * 365)), ql.Days) for t in ts]  # noqa: RUF046
    strikes = list(np.exp(ks))
    vols = ql.Matrix(len(strikes), len(dates))
    for j, t in enumerate(ts):
        col = implied_vol_ssvi(GOOD, ks, t)
        for i in range(len(strikes)):
            vols[i][j] = float(col[i])
    bvs = ql.BlackVarianceSurface(today, cal, dates, strikes, vols, dc)
    bvs.setInterpolation("bicubic")
    spot = ql.QuoteHandle(ql.SimpleQuote(1.0))
    flat = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, dc))
    qlv = ql.LocalVolSurface(ql.BlackVolTermStructureHandle(bvs), flat, flat, spot)
    for t in (0.5, 1.0, 1.5):
        for k in (-0.3, -0.1, 0.0, 0.1, 0.3):
            ours = float(lv.sigma(t, np.exp(k)))
            theirs = qlv.localVol(t, float(np.exp(k)))
            assert abs(ours - theirs) / theirs < 0.05, (t, k, ours, theirs)
