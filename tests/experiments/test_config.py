from neural_particle_method.experiments.config import (
    BUMP_SMOKE,
    FULL,
    SMOKE,
    BumpConfig,
    WarmConfig,
)


def test_full_and_smoke_match_legacy_values():
    assert (FULL.N, FULL.n_steps, FULL.sub, FULL.n_iters, FULL.fit_steps, FULL.reprice_N,
            FULL.reprice_steps, FULL.seq_len, FULL.xover_scales, FULL.norm_iters) == \
        (200_000, 50, 30_000, 6, 300, 300_000, 200, 6, (0.5, 1.0, 2.0, 4.0), 4)
    assert SMOKE.N == 20_000 and SMOKE.xover_scales == (1.0, 4.0)
    assert BUMP_SMOKE == BumpConfig(20_000, 12, 8_000, 2, 120)


def test_derived_configs():
    i = WarmConfig().implicit
    assert (i.n_steps, i.n_particles, i.alpha, i.n_iters, i.fit_steps) == (50, 200_000, 0.5, 6, 300)
    r = BumpConfig().reprice
    assert (r.n_particles, r.n_steps) == (300_000, 200)
    assert WarmConfig().as_params()["xover_scales"] == "(0.5, 1.0, 2.0, 4.0)"
    assert BumpConfig().fit_v_floor is True
