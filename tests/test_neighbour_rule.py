"""Spec P7a: one per-pair restraint-width neighbour rule (gareus/adaptive/neighbour_rule.py)."""

from __future__ import annotations

import math

import pytest

from gareus import adaptive_production as ap
from gareus.adaptive import neighbour_rule as nr
from gareus.adaptive.neighbour_rule import NeighbourPoint as P
from gareus.math_helpers import restraint_sigma

T = 300.0


def _sig(k, t=T):
    return restraint_sigma(k, t)


# --- restraint pattern: same semantics as the driver (P6) -------------------

@pytest.mark.parametrize("k", [None, 0, 0.0, -1.0, float("nan"), float("inf"), "abc", "2.5", 1.2, 300])
@pytest.mark.parametrize("c2", [None, 0.3])
def test_restraint_pattern_matches_driver(k, c2):
    assert nr.axis_restrained(k) == ap._axis_restrained(k)
    for k1 in (None, 0.0, 250.0):
        assert nr.restraint_pattern(k1, k, c2) == ap._restraint_pattern(k1, k, c2)


# --- distance ----------------------------------------------------------------

def test_equal_k_one_axis_uses_sqrt2_sigma():
    a, b = P(0.0, 300.0), P(0.1, 300.0)
    expect = 0.1 / (math.sqrt(2.0) * _sig(300.0))
    assert nr.pair_distance(a, b, T) == pytest.approx(expect, rel=1e-12)


def test_unequal_k_real_chignolin9_pair():
    # states 8 and 12 of chignolin_9 (lambda = 0, CV1-only row)
    a = P(0.0697, 519.707, -0.2517, 0.0, 0.0)
    b = P(0.1118, 255.43, -0.2517, 0.0, 0.0)
    expect = abs(0.1118 - 0.0697) / math.sqrt(_sig(519.707) ** 2 + _sig(255.43) ** 2)
    assert nr.pair_distance(a, b, T) == pytest.approx(expect, rel=1e-12)
    # not a global median: the stiff/soft pair differs from either equal-k pair
    assert expect != pytest.approx(abs(0.1118 - 0.0697) / (math.sqrt(2) * _sig(519.707)))


def test_two_axes_combine_in_quadrature():
    a, b = P(0.0, 300.0, 0.0, 1.2), P(0.05, 200.0, 1.0, 1.2)
    d1 = 0.05 / math.hypot(_sig(300.0), _sig(200.0))
    d2 = 1.0 / math.hypot(_sig(1.2), _sig(1.2))
    assert nr.pair_distance(a, b, T) == pytest.approx(math.hypot(d1, d2), rel=1e-12)


def test_symmetry():
    pts = [P(0.0, 300.0, 0.0, 1.2), P(0.05, 200.0, 1.0, 0.8),
           P(0.1, 0.0, 0.5, 1.2, sampled_mean=(0.07, None)), P(0.2, 400.0, -0.2, 0.0, sampled_mean=(None, 0.4))]
    for a in pts:
        for b in pts:
            assert nr.pair_distance(a, b, T, (0.03, 0.9)) == nr.pair_distance(b, a, T, (0.03, 0.9))


def test_scale_invariance_of_cv_units():
    s = 7.3
    a = P(0.1, 300.0, 0.2, 1.2, sampled_mean=(None, None))
    b = P(0.17, 180.0, 1.3, 0.9)
    c = P(0.4, 0.0, 1.0, 1.1, sampled_mean=(0.25, None))
    scaled = [P(p.primary_center * s, p.primary_k / s**2 if p.primary_k else p.primary_k,
                p.secondary_center * s, p.secondary_k / s**2 if p.secondary_k else p.secondary_k,
                p.rung, tuple(None if m is None else m * s for m in p.sampled_mean)) for p in (a, b, c)]
    for (x, y), (xs, ys) in [((a, b), (scaled[0], scaled[1])), ((a, c), (scaled[0], scaled[2]))]:
        assert nr.pair_distance(xs, ys, T, (0.05 * s, 0.9 * s)) == pytest.approx(
            nr.pair_distance(x, y, T, (0.05, 0.9)), rel=1e-12)


def test_temperature_and_k_scale_together():
    a, b = P(0.0, 300.0, 0.0, 1.2), P(0.05, 200.0, 1.0, 1.2)
    c = 1.7
    a2, b2 = P(0.0, 300.0 * c, 0.0, 1.2 * c), P(0.05, 200.0 * c, 1.0, 1.2 * c)
    assert nr.pair_distance(a2, b2, T * c) == pytest.approx(nr.pair_distance(a, b, T), rel=1e-12)


def test_axis_unrestrained_on_both_is_ignored_placeholders_never_read():
    a, b = P(0.1, 300.0, -0.25, 0.0), P(0.15, 300.0, 5.0, 0.0)   # different CV2 placeholders
    assert nr.pair_distance(a, b, T) == pytest.approx(0.05 / (math.sqrt(2) * _sig(300.0)))
    c, d = P(0.1, 300.0, None, None), P(0.15, 300.0, None, 1.2)  # no CV2 centre = unrestrained
    assert nr.pair_distance(c, d, T) == pytest.approx(0.05 / (math.sqrt(2) * _sig(300.0)))


def test_mixed_axis_uses_pooled_sd_and_sampled_mean():
    fixed = P(0.2, 300.0, 0.0, 1.2)
    free = P(0.5021, 0.0, 0.3, 1.2, sampled_mean=(0.26, None))   # CV1 placeholder 0.5021 is never used
    d1 = abs(0.2 - 0.26) / math.hypot(_sig(300.0), 0.08)
    d2 = 0.3 / math.hypot(_sig(1.2), _sig(1.2))
    assert nr.pair_distance(fixed, free, T, (0.08, None)) == pytest.approx(math.hypot(d1, d2), rel=1e-12)


def test_mixed_axis_without_pool_or_mean_is_unmeasurable():
    fixed = P(0.2, 300.0)
    assert nr.pair_distance(fixed, P(0.2, 0.0, 0.0, 1.2), T, (0.08, None)) == math.inf   # no sampled mean
    assert nr.pair_distance(fixed, P(0.2, 0.0, 0.0, 1.2, sampled_mean=(0.2, None)), T) == math.inf  # no pool
    assert nr.pair_distance(fixed, P(0.2, 0.0, 0.0, 1.2, sampled_mean=(0.2, None)), T, (0.0, None)) == math.inf


def test_unrecorded_k_on_restrained_axis_is_unmeasurable():
    assert nr.pair_distance(P(0.1, None), P(0.2, 300.0), T) == math.inf


def test_no_shared_axis_is_inf():
    assert nr.pair_distance(P(0.1, 0.0, 0.0, 0.0), P(0.2, 0.0, 0.1, 0.0), T) == math.inf


# --- neighbour criterion -----------------------------------------------------

def test_radius_threshold_inclusive():
    a = P(0.0, 300.0)
    step = math.sqrt(2) * _sig(300.0)
    assert nr.is_neighbour(a, P(2.5 * step * (1 - 1e-9), 300.0), T)
    assert not nr.is_neighbour(a, P(2.5 * step * (1 + 1e-9), 300.0), T)
    assert nr.is_neighbour(a, P(1.0 * step, 300.0), T, radius=1.0 + 1e-9)


def test_rung_separation():
    a, b = P(0.1, 300.0, 0.0, 1.2, 0.0), P(0.1, 300.0, 0.0, 1.2, 0.234892)
    assert not nr.is_neighbour(a, b, T)
    assert nr.is_neighbour(a, P(0.12, 300.0, 0.0, 1.2, 0.0000001), T)  # same rung to 1e-6
    assert not nr.is_neighbour(P(0.1, 300.0), a, T)                     # unknown vs known rung
    assert nr.is_neighbour(P(0.1, 300.0), P(0.11, 300.0), T)


def test_pattern_must_match_by_default():
    both = P(0.2, 300.0, 0.0, 1.2, 0.0)
    cv1_only = P(0.2, 300.0, -0.25, 0.0, 0.0, sampled_mean=(None, 0.3))
    cv2_only = P(0.5021, 0.0, 0.0, 1.2, 0.0, sampled_mean=(0.2, None))
    assert not nr.is_neighbour(both, cv1_only, T)
    assert not nr.is_neighbour(both, cv2_only, T, pooled_sd=(0.08, None))
    # spanning mode (3.1) can link them when the free axis is measurable
    assert nr.is_neighbour(both, cv1_only, T, same_pattern_only=False, pooled_sd=(None, 0.9))
    assert nr.is_neighbour(both, cv2_only, T, same_pattern_only=False, pooled_sd=(0.08, None))


def test_anchor_has_no_neighbours():
    anchor = P(0.5021, 0.0, -0.25, 0.0, 0.0)
    others = [P(0.5021, 0.0, -0.25, 0.0, 0.0), P(0.5, 300.0, -0.25, 0.0, 0.0), P(0.5, 0.0, -0.25, 1.2, 0.0)]
    for o in others:
        assert not nr.is_neighbour(anchor, o, T, same_pattern_only=False, pooled_sd=(0.1, 0.9))
    assert not nr.is_neighbour(anchor, anchor, T)


def test_neighbour_pairs_lists_and_components():
    step = math.sqrt(2) * _sig(300.0)
    pts = [P(0.0, 300.0, rung=0.0), P(step, 300.0, rung=0.0), P(2 * step, 300.0, rung=0.0),
           P(10 * step, 300.0, rung=0.0), P(step, 300.0, rung=1.0), P(0.0, 0.0, 0.0, 0.0, rung=0.0)]
    pairs = nr.neighbour_pairs(pts, T, radius=1.5)
    assert pairs == [(0, 1), (1, 2)]
    lists = nr.neighbour_lists(len(pts), pairs)
    assert lists == {0: [1], 1: [0, 2], 2: [1], 3: [], 4: [], 5: []}
    assert nr.components(len(pts), pairs) == [[0, 1, 2], [3], [4], [5]]
    assert nr.neighbour_pairs(pts, T, radius=2.5) == [(0, 1), (0, 2), (1, 2)]


def test_from_state_reads_window_state():
    st = ap.WindowState(state_id=3, primary_center=0.2, primary_k=300.0, secondary_center=0.1,
                        secondary_k=1.2, gamd_lambda=0.25)
    p = P.from_state(st)
    assert (p.primary_center, p.primary_k, p.secondary_center, p.secondary_k, p.rung) == (0.2, 300.0, 0.1, 1.2, 0.25)
