"""Ladder designer: endpoints fixed, adjacent overlaps >= target, minimal rungs, hysteresis."""
import numpy as np
import pytest

from gareus.adaptive.ladder_adapt import (aggregate_overlap, design_ladder, fit_centre_model,
                                          plan_ladder_change)
from ladder_adapt_fixture import BETA, ENV, _sample_at  # noqa: E402  (tests dir on sys.path)


@pytest.fixture(scope="module")
def models():
    out = []
    for c in range(4):
        out.append(fit_centre_model({l: _sample_at(l, 3000, 100 + 10 * c + i)
                                     for i, l in enumerate((0.0, 0.25, 0.5, 0.75, 1.0))}, ENV, BETA))
    return out


def test_design_keeps_endpoints_and_meets_target(models):
    d = design_ladder(models, target=0.25)
    assert d.feasible and d.lambdas[0] == 0.0 and d.lambdas[-1] == 1.0
    assert all(p is not None and p >= 0.25 - 1e-6 for p in d.predicted)


def test_design_uses_the_minimum_rung_count(models):
    d = design_ladder(models, target=0.25)
    fewer = list(np.linspace(0.0, 1.0, len(d.lambdas) - 1))
    worst = min(aggregate_overlap(models, a, b) for a, b in zip(fewer, fewer[1:]))
    assert worst < 0.25                     # one rung fewer, even evenly spaced, fails


def test_design_equalises_adjacent_overlaps(models):
    d = design_ladder(models, target=0.25)
    assert max(d.predicted) - min(d.predicted) < 0.03


def test_design_places_rungs_densest_where_overlap_falls_fastest(models):
    d = design_ladder(models, target=0.35)
    gaps = np.diff(d.lambdas)
    assert gaps[0] < gaps[-1]               # overlap falls fastest at low lambda in this family


def test_higher_target_needs_more_rungs(models):
    assert len(design_ladder(models, target=0.35).lambdas) > len(design_ladder(models, target=0.2).lambdas)


def test_infeasible_target_is_reported_not_forced(models):
    d = design_ladder(models, target=0.49, max_rungs=4)
    assert not d.feasible and "max_rungs" in d.reason


def test_no_change_when_current_ladder_is_already_good(models):
    d = design_ladder(models, target=0.25)
    ch = plan_ladder_change(list(d.lambdas), models, d, target=0.25)
    assert ch.drop == () and ch.add == ()


def test_redundant_rung_is_dropped(models):
    d = design_ladder(models, target=0.20)
    current = sorted(set(d.lambdas) | {0.9, 0.95})
    ch = plan_ladder_change(current, models, d, target=0.20, max_moves=4)
    assert set(ch.drop) == {0.9, 0.95} and ch.add == ()
    assert 0.0 not in ch.drop and 1.0 not in ch.drop


def test_a_violated_gap_gets_a_rung(models):
    d = design_ladder(models, target=0.3)
    ch = plan_ladder_change([0.0, 1.0], models, d, target=0.3, max_moves=8)
    assert len(ch.add) == len(d.lambdas) - 2 and ch.drop == ()


def test_moves_per_epoch_are_capped(models):
    d = design_ladder(models, target=0.3)
    ch = plan_ladder_change([0.0, 1.0], models, d, target=0.3, max_moves=1)
    assert len(ch.add) == 1 and ch.drop == ()


def test_a_full_respace_within_budget_is_one_atomic_change(models):
    d = design_ladder(models, target=0.25)
    shifted = [0.0] + [min(0.99, x + 0.1) for x in d.lambdas[1:-1]] + [1.0]
    ch = plan_ladder_change(shifted, models, d, target=0.25, max_moves=4)
    after = sorted((set(shifted) - set(ch.drop)) | set(ch.add))
    assert after == list(d.lambdas) and len(ch.add) == len(ch.drop)      # rung count unchanged


def test_incomplete_centres_are_skipped(models):
    two_rung = fit_centre_model({0.0: _sample_at(0.0, 3000, 7), 0.25: _sample_at(0.25, 3000, 8)}, ENV, BETA)
    with_partial = list(models) + [two_rung]
    # (0.5, 1.0) is outside the partial centre's sampled range: it must not drag the quantile down
    assert aggregate_overlap(with_partial, 0.5, 1.0) == pytest.approx(aggregate_overlap(models, 0.5, 1.0))


def test_unsupported_lambda_is_not_placed():
    only_low = [fit_centre_model({0.0: _sample_at(0.0, 3000, 1), 0.3: _sample_at(0.3, 3000, 2)}, ENV, BETA)]
    d = design_ladder(only_low, target=0.25, lam_max=1.0)
    assert not d.feasible and "support" in d.reason
