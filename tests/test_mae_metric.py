import numpy as np

from neural_particle_method.pricing.metrics import iv_metrics


def test_mae_alongside_rmse():
    k = np.array([-0.5, 0.0, 0.5])
    model = np.array([[0.21, 0.20, 0.19], [0.20, 0.20, 0.20]])
    target = np.array([[0.20, 0.20, 0.20], [0.20, 0.20, 0.20]])
    m = iv_metrics(model, target, k, [0.5, 1.0])
    assert m["pooled_mae_bp"] == round(100 * 2 / 6, 9)          # two 100 bp errors over six quotes
    assert m["wings_mae_bp"] == round(100 * 2 / 4, 9)           # both errors sit in the four wing quotes
    assert [r["mae_bp"] for r in m["mae_per_maturity"]] == [round(200 / 3, 9), 0.0]
    assert m["pooled_rmse_bp"] == round(np.sqrt(2 * 100 ** 2 / 6), 9)
    assert set(m["per_maturity"][0]) == {"T", "rmse_bp", "max_bp"}   # untouched for the golden replays
