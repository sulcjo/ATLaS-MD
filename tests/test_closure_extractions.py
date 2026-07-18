"""Tests for pure logic extracted out of god-function closures.

gareus/production.py's run_gareus (~1836 lines) and gareus/seeding.py's
generate_us_starting_states_by_pulling (~1075 lines) each trapped a piece of
pure arithmetic inside nested closures, making it untestable without a live
OpenMM Context. Both pieces were pulled out to module level with every
captured local turned into an explicit parameter, behavior unchanged:
  - gareus.production._exchange_probability (was run_gareus's nested
    _exchange_probability, captured ``beta`` from the enclosing scope)
  - gareus.seeding._score_seed_conformer (was generate_us_starting_states_by_
    pulling's nested ``_score``, two closures deep, captured seven locals)

No OpenMM / PeptideBuilder imports (unit-test rule).
"""
import math

import pytest

from gareus.production import _exchange_probability
from gareus.seeding import _score_seed_conformer


# ---------------------------------------------------------------------------
# _exchange_probability
# ---------------------------------------------------------------------------

def test_downhill_swap_always_accepted():
    # delta <= 0 (new state no higher in energy) -> Metropolis probability 1.0
    assert _exchange_probability(delta_kj=0.0, beta=1.0) == 1.0
    assert _exchange_probability(delta_kj=-5.0, beta=1.0) == 1.0


def test_uphill_swap_follows_boltzmann_factor():
    delta_kj = 2.0
    beta = 0.5
    prob = _exchange_probability(delta_kj, beta)
    assert prob == pytest.approx(math.exp(-beta * delta_kj))
    assert 0.0 < prob < 1.0


def test_extreme_uphill_underflows_to_zero_not_exception():
    # x = -beta*delta_kj < -745 would overflow math.exp -> must clamp, not raise
    assert _exchange_probability(delta_kj=1.0e6, beta=1.0) == 0.0


def test_non_finite_delta_returns_zero():
    assert _exchange_probability(delta_kj=float("nan"), beta=1.0) == 0.0
    assert _exchange_probability(delta_kj=float("inf"), beta=1.0) == 0.0


def test_non_numeric_delta_returns_zero_not_exception():
    assert _exchange_probability(delta_kj="not-a-number", beta=1.0) == 0.0


# ---------------------------------------------------------------------------
# _score_seed_conformer
# ---------------------------------------------------------------------------

def _score(conf, **overrides):
    kwargs = dict(
        window_index=0,
        seed_selection_mode="active-cv",
        primary_seed_scale=0.20,
        target_primary=0.10,
        target_secondary=0.0,
        secondary_available=True,
        seed_secondary_weight=1.0,
        secondary_seed_scale=0.20,
        args="distance",
    )
    kwargs.update(overrides)
    return _score_seed_conformer(conf, **kwargs)


def test_scores_primary_and_secondary_terms():
    conf = {"primary_cv_value": 0.30, "secondary_cv_value": 0.20, "pdb_path": "seed_001.pdb"}
    total, components = _score(conf)
    assert components["abs_primary_delta"] == pytest.approx(0.20)
    assert components["primary_score"] == pytest.approx(1.0)
    assert components["abs_secondary_delta"] == pytest.approx(0.20)
    assert components["secondary_score"] == pytest.approx(1.0)
    assert total == pytest.approx(2.0)
    assert components["conformer_pdb"] == "seed_001.pdb"
    assert components["window"] == 0


def test_distance_mode_prefers_cv_a_over_primary_cv_value():
    conf = {"primary_cv_value": 0.30, "cv_A": 5.0}
    _, components = _score(conf, seed_selection_mode="distance")
    assert components["conformer_primary_cv"] == 5.0


def test_nan_secondary_gets_unit_penalty_not_zero():
    # A NaN secondary must not score 0 (which would make it look like a perfect
    # secondary match) - it must be penalized by exactly seed_secondary_weight.
    conf = {"primary_cv_value": 0.10, "secondary_cv_value": float("nan")}
    _, components = _score(conf, seed_secondary_weight=2.0)
    assert components["secondary_score"] == pytest.approx(2.0)
    assert math.isnan(components["abs_secondary_delta"])


def test_secondary_ignored_when_weight_zero():
    conf = {"primary_cv_value": 0.10, "secondary_cv_value": 0.99}
    _, components = _score(conf, seed_secondary_weight=0.0)
    assert components["secondary_score"] == 0.0


def test_secondary_ignored_when_unavailable():
    conf = {"primary_cv_value": 0.10, "secondary_cv_value": 0.99}
    _, components = _score(conf, secondary_available=False)
    assert components["secondary_score"] == 0.0


def test_unavailable_primary_falls_back_to_secondary_only():
    conf = {"primary_cv_value": float("nan"), "secondary_cv_value": 0.0}
    total, components = _score(conf)
    assert components["primary_score"] == 0.0
    assert total == pytest.approx(components["secondary_score"])


def test_lower_total_score_ranks_better_seed():
    near = {"primary_cv_value": 0.11, "secondary_cv_value": 0.01}
    far = {"primary_cv_value": 0.90, "secondary_cv_value": 0.90}
    total_near, _ = _score(near)
    total_far, _ = _score(far)
    assert total_near < total_far
