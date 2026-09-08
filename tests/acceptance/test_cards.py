import pytest

from neural_particle_method.bench.acceptance import CARDS, check
from neural_particle_method.tracking.store import Store


@pytest.mark.slow
@pytest.mark.parametrize("card", CARDS, ids=lambda c: c.name)
def test_card(card):
    status, achieved, rid = check(Store(), card)     # the repo store, so the run is kept
    assert status != "WORSE", (
        f"{card.name}: {status}, achieved {achieved:.4g} vs source {card.source_value} ± {card.tolerance} (run {rid})"
    )
