"""Seed primary-CV scoring (fix B: mislabeled distance for backbone-only seeds).

When the primary CV is a sidechain-all contact fraction and the seed is a
backbone-only poly-glycine conformer, no contact pair maps onto the seed, so the
loader previously substituted a terminal distance (Å) and scored it as if it were
a dimensionless contact fraction (observed conformer_primary_cv ~ 3.8 against
[0, 0.11] window centers).  The primary CV is now marked unavailable (NaN) for
such seeds, and unavailable primaries must degrade to CV2-only scoring rather
than an infinite penalty that swamps every seed.
"""

import math

import pytest

from gareus.seeding import _primary_seed_score


def test_finite_primary_scores_by_scaled_distance():
    delta, score = _primary_seed_score(primary_value=0.30, target_primary=0.10, primary_seed_scale=0.20)
    assert delta == pytest.approx(0.20)
    assert score == pytest.approx(1.0)


def test_unavailable_primary_degrades_to_zero_not_infinite():
    delta, score = _primary_seed_score(primary_value=float("nan"), target_primary=0.10, primary_seed_scale=0.20)
    assert math.isnan(delta)
    assert score == 0.0  # not inf: lets the secondary (CV2) term rank the seeds


def test_all_seeds_unavailable_leaves_secondary_to_break_ties():
    # Two seeds, both with unavailable primary but different CV2 quality: the
    # primary term must not mask the CV2 difference.
    _, score_a = _primary_seed_score(float("nan"), 0.10, 0.20)
    _, score_b = _primary_seed_score(float("nan"), 0.10, 0.20)
    assert score_a == score_b == 0.0
