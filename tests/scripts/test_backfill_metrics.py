import numpy as np

from neural_particle_method.bench.scenarios import full_registry, quote_k_grid
from tests.scripts.helpers import load_script

bf = load_script("backfill_metrics")


def _zero_err(sc):
    return np.zeros((len(list(sc.maturities)), len(quote_k_grid())))


def test_zero_error_grid_gives_zero_price_and_liquid_metrics():
    sc = full_registry()["s01"]
    err = _zero_err(sc)
    price = bf.price_metrics(sc, err)
    assert set(price) >= {"price_bp/pooled", "price_bp/wings", "price_bp/near", "price_bp/max",
                          "price_bp/liquid"}
    assert all(v == 0.0 for v in price.values())
    assert bf.liquid_metrics(sc, err) == {"liquid_mae_bp": 0.0}


def test_constant_ten_bp_error_gives_ten_bp_liquid_mae():
    sc = full_registry()["s01"]
    err = _zero_err(sc) + 10.0
    assert bf.liquid_metrics(sc, err)["liquid_mae_bp"] == 10.0


def test_liquid_mask_is_near_money_and_long_dated():
    sc = full_registry()["s01"]
    k = quote_k_grid()
    mask = bf.liquid_mask(sc, k)
    mats = np.asarray([float(T) for T in sc.maturities])
    assert mask.shape == (len(mats), len(k))
    assert mask.any()
    np.testing.assert_array_equal(mask, (mats[:, None] >= 0.5) & (np.abs(k)[None, :] <= 0.25))


def test_the_tuned_budget_experiment_is_back_filled():
    assert "suite_budget_tuned" in bf.EXPERIMENTS
