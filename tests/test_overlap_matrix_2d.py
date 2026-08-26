"""Secondary-CV-aware window overlap (`gareus.mbar_analysis.solvers.overlap_matrix`).

Motivating real failure (chignolin_6, see
docs/chignolin_6_low_ess_root_cause.md, "Secondary findings" #1): 20 of that
run's 27 umbrella states share just TWO primary-CV centres (0.0 and 0.0654)
and differ only in the SECONDARY CV, where the centre-spacing/sigma ratio was
2.6-9.3 (healthy umbrella overlap wants ~1-1.5). Because `overlap_matrix` only
ever histogrammed the primary-CV marginal, it reported 0.6-0.99 overlap for
exactly those pairs -- so the run's health check declared window overlap fine
while the axis that had actually failed was invisible to it. The pipeline was
structurally unable to see the gap.

These tests pin down the fix and, just as importantly, the two ways the fix
could itself go wrong: silently changing the existing 1D numbers, and being so
sparsity-deflated at real per-state sample counts that it flags healthy pairs.
"""
from __future__ import annotations

import numpy as np
import pytest

from gareus.mbar_analysis.solvers import (
    overlap_matrix, make_overlap_bins, OVERLAP_SECONDARY_BINS,
)


# --- Frozen copy of the pre-fix 1D implementation -----------------------------
# Bit-identity of the 1D path is a hard requirement of this change (every
# existing caller passes the old 4-argument signature), so it is asserted
# against a frozen reference rather than argued for in review. This also pins
# the two quirks that must survive: the `bi[cv==bins[-1]]=B-1` right-edge
# inclusion, and `H` being zero-initialised BEFORE the `if np.any(mask)` guard
# so an all-out-of-range input returns zeros instead of raising.
def _overlap_matrix_1d_reference(cv, window, bins, K):
    cv = np.asarray(cv, dtype=np.float64)
    window = np.asarray(window, dtype=np.int64)
    B = len(bins) - 1
    H = np.zeros((K, B), dtype=np.float64)
    if cv.size and K > 0 and B > 0:
        bi = np.searchsorted(bins, cv, side='right') - 1
        bi[cv == bins[-1]] = B - 1
        mask = (window >= 0) & (window < K) & (bi >= 0) & (bi < B)
        if np.any(mask):
            linear = window[mask] * B + bi[mask]
            H = np.bincount(linear, minlength=K * B).reshape(K, B).astype(np.float64)
            row_sums = H.sum(axis=1)
            nz = row_sums > 0
            H[nz] /= row_sums[nz, None]
    return np.minimum(H[:, None, :], H[None, :, :]).sum(axis=2)


def _two_state_samples(rng, n, c1_a, c1_b, c2_a, c2_b, s1=0.05, s2=0.07):
    """Two umbrella states, `n` samples each, Gaussian in both CVs."""
    cv = np.concatenate([rng.normal(c1_a, s1, n), rng.normal(c1_b, s1, n)])
    cv2 = np.concatenate([rng.normal(c2_a, s2, n), rng.normal(c2_b, s2, n)])
    window = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    return cv, cv2, window


# --- Backward compatibility ---------------------------------------------------

@pytest.mark.parametrize("K,nbins,n", [(2, 10, 500), (5, 30, 5000), (27, 60, 20000)])
def test_1d_path_is_bit_identical_to_the_frozen_reference(K, nbins, n):
    rng = np.random.default_rng(7)
    cv = rng.normal(0.2, 0.15, n)
    window = rng.integers(0, K, n)
    bins = np.linspace(-0.2, 0.6, nbins + 1)
    got = overlap_matrix(cv, window, bins, K)
    assert np.array_equal(got, _overlap_matrix_1d_reference(cv, window, bins, K))


def test_1d_path_preserves_right_edge_inclusion_and_out_of_range_zeros():
    bins = np.linspace(0.0, 1.0, 5)
    # exact right edge (must land in the LAST bin, not out of range), exact
    # left edge, plus values below/above the axis and an out-of-range window.
    cv = np.array([1.0, 0.0, -0.5, 1.5, 0.5, 0.5])
    window = np.array([0, 0, 0, 1, 1, 9])
    got = overlap_matrix(cv, window, bins, 2)
    assert np.array_equal(got, _overlap_matrix_1d_reference(cv, window, bins, 2))
    # all-out-of-range input still returns zeros rather than raising
    empty = overlap_matrix(np.array([-9.0, 9.0]), np.array([0, 1]), bins, 2)
    assert np.array_equal(empty, np.zeros((2, 2)))


def test_stats_out_dict_is_filled_on_the_1d_path():
    bins = np.linspace(0.0 , 1.0, 5)
    cv = np.array([0.1, 0.2, 0.9, -5.0])
    window = np.array([0, 0, 1, 1])
    stats = {}
    overlap_matrix(cv, window, bins, 2, stats=stats)
    assert stats['dim'] == 1
    assert stats['n_total'] == 4
    assert stats['n_used'] == 3           # the -5.0 sample is off-axis
    assert stats['n_excluded_nonfinite_cv2'] == 0
    assert stats['retained_fraction'] == pytest.approx(0.75)


# --- The case that motivated the fix -----------------------------------------

def test_identical_cv1_separated_cv2_is_near_zero_in_2d_but_near_one_in_1d():
    """The chignolin_6 geometry: same primary centre, CV2 centres 2.85 apart at
    sigma 0.07 (spacing/sigma ~ 40, i.e. no overlap whatsoever on that axis)."""
    rng = np.random.default_rng(0)
    cv, cv2, window = _two_state_samples(rng, 20000, 0.0, 0.0, -1.4379, 1.4070)
    bins = np.linspace(-0.05, 0.42, 61)
    bins2 = make_overlap_bins(cv2)

    o1d = overlap_matrix(cv, window, bins, 2)[0, 1]
    o2d = overlap_matrix(cv, window, bins, 2, cv2=cv2, bins2=bins2)[0, 1]

    assert o1d > 0.95, f"1D marginal should be blind to the CV2 gap, got {o1d}"
    assert o2d < 0.01, f"2D overlap should expose the CV2 gap, got {o2d}"


def test_marginal_cv2_spacing_lands_between_the_two_extremes():
    """A spacing/sigma of ~2.6 (the run's own 20->21 pair) is genuinely weak but
    not disjoint -- the 2D number must be small yet nonzero, so a threshold can
    discriminate it from the healthy case below."""
    rng = np.random.default_rng(1)
    cv, cv2, window = _two_state_samples(rng, 20000, 0.0, 0.0, 0.0, 2.6 * 0.07)
    bins = np.linspace(-0.05, 0.42, 61)
    o2d = overlap_matrix(cv, window, bins, 2, cv2=cv2, bins2=make_overlap_bins(cv2))[0, 1]
    assert 0.0 < o2d < 0.30, o2d


def test_coincident_states_stay_high_overlap_at_realistic_sample_counts():
    """Positive control against sparsity deflation. Histogram-intersection
    overlap is systematically deflated as the cell count grows relative to
    samples per state, and the sparsest real states in chignolin_6 hold only
    ~9k samples each. Two genuinely coincident distributions must still report
    high overlap at that N with the recommended bin counts, or the new
    diagnostic would flag everything -- as useless as the old one flagging
    nothing."""
    rng = np.random.default_rng(2)
    for sigma2 in (0.07, 0.30):
        cv, cv2, window = _two_state_samples(rng, 9000, 0.0, 0.0, 0.0, 0.0, s2=sigma2)
        bins = np.linspace(-0.05, 0.42, 61)
        bins2 = make_overlap_bins(cv2)
        o2d = overlap_matrix(cv, window, bins, 2, cv2=cv2, bins2=bins2)[0, 1]
        assert o2d > 0.80, f"sigma2={sigma2}: coincident states deflated to {o2d}"


# --- NaN / null secondary CV --------------------------------------------------

def test_all_nonfinite_cv2_falls_back_to_the_exact_1d_matrix():
    """Data.cv2 is non-Optional in this codebase and is NaN-filled for a pure
    CV1-only run, so a caller wiring cv2=d.cv2 unconditionally WILL pass an
    all-NaN column. Excluding every sample would return an all-zero matrix and
    make a perfectly healthy run look catastrophically un-overlapped."""
    rng = np.random.default_rng(3)
    cv = rng.normal(0.2, 0.1, 5000)
    window = rng.integers(0, 4, 5000)
    bins = np.linspace(-0.2, 0.6, 41)
    cv2 = np.full(cv.shape, np.nan)
    stats = {}
    got = overlap_matrix(cv, window, bins, 4, cv2=cv2, bins2=np.linspace(-1, 1, 31), stats=stats)
    assert np.array_equal(got, overlap_matrix(cv, window, bins, 4))
    assert stats['dim'] == 1
    assert stats['fell_back_to_marginal'] is True


def test_nonfinite_cv2_samples_are_excluded_not_treated_as_zero():
    """A null/masked cv2 must never be silently read as the value 0.0 (the
    2026-08-15 three-layer bias-fabrication bug in CLAUDE.md). State 0's cv2 is
    entirely null while state 1 sits at cv2=0: if NaN were binned as 0.0 the two
    would look perfectly overlapped. They must instead report zero overlap, and
    the exclusion must be counted."""
    rng = np.random.default_rng(4)
    n = 4000
    cv = np.concatenate([rng.normal(0.0, 0.05, n), rng.normal(0.0, 0.05, n)])
    cv2 = np.concatenate([np.full(n, np.nan), rng.normal(0.0, 0.07, n)])
    window = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    bins = np.linspace(-0.2, 0.2, 41)
    bins2 = np.linspace(-1.0, 1.0, 31)
    stats = {}
    O = overlap_matrix(cv, window, bins, 2, cv2=cv2, bins2=bins2, stats=stats)
    assert stats['dim'] == 2
    assert stats['fell_back_to_marginal'] is False
    assert O[0, 1] == 0.0
    assert O[0, 0] == 0.0                # state 0 contributed no 2D samples at all
    assert O[1, 1] == pytest.approx(1.0)
    assert stats['n_excluded_nonfinite_cv2'] == n
    assert stats['retained_fraction'] == pytest.approx(0.5, abs=0.02)


def test_infinite_cv2_is_excluded_like_nan():
    bins = np.linspace(0.0, 1.0, 5)
    bins2 = np.linspace(0.0, 1.0, 5)
    cv = np.array([0.5, 0.5, 0.5])
    cv2 = np.array([0.5, np.inf, -np.inf])
    window = np.array([0, 0, 0])
    stats = {}
    overlap_matrix(cv, window, bins, 1, cv2=cv2, bins2=bins2, stats=stats)
    assert stats['n_used'] == 1
    assert stats['n_excluded_nonfinite_cv2'] == 2


# --- Malformed input fails closed --------------------------------------------

def test_mismatched_cv2_length_raises():
    bins = np.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError):
        overlap_matrix(np.zeros(10), np.zeros(10, dtype=np.int64), bins, 1,
                       cv2=np.zeros(9), bins2=bins)


def test_degenerate_bins2_raises():
    bins = np.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError):
        overlap_matrix(np.zeros(10), np.zeros(10, dtype=np.int64), bins, 1,
                       cv2=np.zeros(10), bins2=np.array([0.0]))


def test_cv2_without_bins2_stays_on_the_1d_path():
    """Both halves of the secondary spec are required to switch paths; passing
    only one must not half-apply the change."""
    rng = np.random.default_rng(5)
    cv, cv2, window = _two_state_samples(rng, 2000, 0.0, 0.0, -1.5, 1.5)
    bins = np.linspace(-0.2, 0.2, 21)
    ref = overlap_matrix(cv, window, bins, 2)
    assert np.array_equal(overlap_matrix(cv, window, bins, 2, cv2=cv2), ref)
    assert np.array_equal(overlap_matrix(cv, window, bins, 2, bins2=bins), ref)


# --- Cell-chunked reduction ---------------------------------------------------

def test_cell_chunked_reduction_matches_a_single_chunk():
    """The 2D pairwise-min reduction accumulates over cell chunks so peak memory
    is K^2 * chunk rather than K^2 * (B * B2). The chunk size must not change
    the result beyond float round-off."""
    rng = np.random.default_rng(6)
    cv, cv2, window = _two_state_samples(rng, 3000, 0.0, 0.1, -0.3, 0.3)
    window = rng.integers(0, 6, cv.size)
    bins = np.linspace(-0.3, 0.4, 25)
    bins2 = np.linspace(-0.8, 0.8, 25)
    big = overlap_matrix(cv, window, bins, 6, cv2=cv2, bins2=bins2,
                         max_cells_per_chunk=10 ** 9)
    small = overlap_matrix(cv, window, bins, 6, cv2=cv2, bins2=bins2,
                           max_cells_per_chunk=7)
    assert np.allclose(big, small, atol=1e-12)


def test_2d_matrix_is_symmetric_with_unit_diagonal_for_populated_states():
    rng = np.random.default_rng(8)
    cv, cv2, window = _two_state_samples(rng, 5000, 0.0, 0.08, -0.2, 0.2)
    bins = np.linspace(-0.2, 0.3, 31)
    bins2 = np.linspace(-0.6, 0.6, 31)
    O = overlap_matrix(cv, window, bins, 2, cv2=cv2, bins2=bins2)
    assert np.allclose(O, O.T)
    assert np.allclose(np.diag(O), 1.0)


# --- make_overlap_bins --------------------------------------------------------

def test_make_overlap_bins_ignores_nonfinite_values():
    values = np.array([0.0, 1.0, np.nan, np.inf, -np.inf, 0.5])
    edges = make_overlap_bins(values, nbins=4)
    assert edges is not None
    assert len(edges) == 5
    assert edges[0] <= 0.0 and edges[-1] >= 1.0
    assert np.all(np.isfinite(edges))


def test_make_overlap_bins_returns_none_when_unusable():
    assert make_overlap_bins(np.full(10, np.nan)) is None
    assert make_overlap_bins(np.array([])) is None
    assert make_overlap_bins(np.full(10, 0.25)) is None      # one distinct value


def test_wide_secondary_axis_still_tracks_real_overlap_at_realistic_n():
    """The production geometry, which is the OPPOSITE sparsity regime to the
    test above: `bins2` is built once over the whole campaign's cv2 column
    (chignolin_6's spans ~3.8) while a single state occupies sigma ~0.04-0.12 of
    it, so a state lands in only 1-3 secondary bins and quantization -- not a
    genuine gap -- could collapse min(H_i, H_j) for two states that really do
    overlap. It does not: at the recommended OVERLAP_SECONDARY_BINS over a
    3.8-wide axis the reported overlap stays monotone in centre separation and
    close to the analytic Gaussian value (0.62 at 1 sigma), which is what makes
    the number comparable against args.min_neighbor_overlap."""
    rng = np.random.default_rng(9)
    n, sigma2 = 9000, 0.07
    bins = np.linspace(-0.05, 0.42, 61)
    bins2 = np.linspace(-2.2, 1.6, OVERLAP_SECONDARY_BINS + 1)
    got = {}
    for spacing_sigma in (0.0, 1.0, 5.0):
        cv, cv2, window = _two_state_samples(
            rng, n, 0.0, 0.0, 0.0, spacing_sigma * sigma2, s2=sigma2)
        got[spacing_sigma] = overlap_matrix(cv, window, bins, 2, cv2=cv2, bins2=bins2)[0, 1]
    assert got[0.0] > 0.90, got            # coincident states not quantization-deflated
    assert 0.50 < got[1.0] < 0.85, got     # ~0.62 analytic, resolved not collapsed
    assert got[5.0] < 0.05, got            # a real gap is still exposed
    assert got[0.0] > got[1.0] > got[5.0]
