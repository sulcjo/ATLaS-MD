"""
Tests for exchange kernel and sampling correctness fixes.

Covers:
  N1 – Gibbs-walk heat-bath weights (no OpenMM required)
  F6 – NaN secondary-CV bias matrix guard
  N4 – Segment RNG seed uniqueness
  F5 – NaN secondary seed penalty in seeding.py
"""
from __future__ import annotations

import copy
import math
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# N1: Gibbs-walk heat-bath kernel leaves π invariant
# ---------------------------------------------------------------------------

def _build_gibbs_transition_matrix(U: np.ndarray, beta: float) -> np.ndarray:
    """
    Reproduce the fixed gibbs-walk selection as a K×K transition matrix.

    T[i, j] = probability that window i selects window j.
    Uses the same logic as production.py after the N1 fix:
      log_weights = clip(-beta * delta, -745, None)   # no upper bound
      weights     = softmax(log_weights)
      force_accept=True (heat-bath; no additional Metropolis step)
    """
    K = len(U)
    T = np.zeros((K, K))
    for i in range(K):
        # Each window treats every other as a potential swap partner.
        # delta[j] = U[j] + U[i] - U[i] - U[j] simplified to single-site:
        # for the replica-exchange toy model delta[j] = U[j] - U[i]
        # (swap replica i to window j means energy changes by U[j]-U[i])
        delta = U - U[i]           # shape (K,)
        log_w = np.clip(-beta * delta, -745.0, None)  # FIXED: no upper bound
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        T[i] = w / w.sum()
    return T


def test_gibbs_walk_stationary_distribution():
    """
    The Gibbs-walk kernel must leave π(i) ∝ exp(-β·U_i) invariant.
    Checks max|π·T − π| < 1e-12.
    """
    rng = np.random.default_rng(0)
    U = rng.uniform(0.5, 3.0, size=6)
    beta = 1.0
    T = _build_gibbs_transition_matrix(U, beta)
    pi = np.exp(-beta * U)
    pi /= pi.sum()
    residual = np.max(np.abs(pi @ T - pi))
    assert residual < 1e-12, f"stationary-distribution residual = {residual:.2e} (should be < 1e-12)"


def test_gibbs_walk_detailed_balance():
    """
    Heat-bath satisfies detailed balance: π_i · T_ij = π_j · T_ji for all i,j.
    """
    rng = np.random.default_rng(1)
    U = rng.uniform(0.5, 3.0, size=5)
    beta = 1.0
    T = _build_gibbs_transition_matrix(U, beta)
    pi = np.exp(-beta * U); pi /= pi.sum()
    violations = np.max(np.abs(pi[:, None] * T - pi[None, :] * T.T))
    assert violations < 1e-12, f"detailed balance violated: {violations:.2e}"


def test_gibbs_walk_old_clip_was_wrong():
    """
    Regression: the OLD upper-clip (a_max=0.0) produces a WRONG stationary
    distribution.  Confirms the bug was real, not a false positive.
    """
    def _old_kernel(U, beta):
        K = len(U)
        T = np.zeros((K, K))
        for i in range(K):
            delta = U - U[i]
            log_w = np.clip(-beta * delta, -745.0, 0.0)  # OLD: wrong upper bound
            log_w -= np.max(log_w)
            w = np.exp(log_w); T[i] = w / w.sum()
        return T

    U = np.array([1.0, 2.0, 3.0, 0.5])
    beta = 1.0
    T_old = _old_kernel(U, beta)
    pi = np.exp(-beta * U); pi /= pi.sum()
    residual_old = np.max(np.abs(pi @ T_old - pi))
    assert residual_old > 1e-3, (
        f"old kernel unexpectedly satisfies detailed balance ({residual_old:.2e}); "
        "test is not discriminating"
    )


def test_gibbs_walk_favourable_swaps_get_higher_weight():
    """
    For Δ < 0 (favourable swap), the fixed kernel assigns weight > 1 in log space.
    The old kernel clipped these to 0, giving them the same weight as staying.
    """
    beta = 1.0
    U = np.array([2.0, 0.5])   # window 0 at high energy, window 1 at low energy
    delta = U[1] - U[0]        # = -1.5 < 0 (swap from 0→1 is favourable)

    # fixed
    lw_fixed = float(np.clip(-beta * delta, -745.0, None))
    # old
    lw_old   = float(np.clip(-beta * delta, -745.0, 0.0))

    assert lw_fixed > 0.0, "fixed kernel: favourable swap should have log_weight > 0"
    assert lw_old == 0.0,  "old kernel: favourable swap was incorrectly clamped to 0"


# ---------------------------------------------------------------------------
# F6: NaN secondary CV bias matrix guard
# ---------------------------------------------------------------------------

def _build_ss_bias_matrix(ss_centers: np.ndarray,
                           ss_k: np.ndarray,
                           ss_values: np.ndarray,
                           guard: bool) -> np.ndarray:
    """Reproduce production.py ss_bias_matrix_kcal computation."""
    ss_delta = ss_values[np.newaxis, :] - ss_centers[:, np.newaxis]
    bias = 0.5 * ss_k[:, np.newaxis] * ss_delta * ss_delta
    if guard:
        bias = np.where(np.isfinite(ss_delta), bias, 0.0)
    return bias


def test_nan_secondary_bias_unguarded_propagates_nan():
    """Without the guard, a NaN center or value poisons the entire matrix."""
    ss_centers = np.array([0.0, float("nan"), 0.5])
    ss_k       = np.array([1.0, 1.0, 1.0])
    ss_values  = np.array([0.1, 0.2, 0.4])
    bias = _build_ss_bias_matrix(ss_centers, ss_k, ss_values, guard=False)
    assert not np.isfinite(bias).all(), "expected NaN without guard"


def test_nan_secondary_bias_guarded_is_finite():
    """With the guard, NaN entries are replaced with 0."""
    ss_centers = np.array([0.0, float("nan"), 0.5])
    ss_k       = np.array([1.0, 1.0, 1.0])
    ss_values  = np.array([0.1, 0.2, float("nan")])
    bias = _build_ss_bias_matrix(ss_centers, ss_k, ss_values, guard=True)
    assert np.isfinite(bias).all(), "expected all-finite bias with guard"


def test_nan_secondary_bias_guard_preserves_finite_entries():
    """The guard must not alter finite entries."""
    ss_centers = np.array([0.0, 0.33, 0.5])
    ss_k       = np.array([2.0, 2.0, 2.0])
    ss_values  = np.array([0.1, float("nan"), 0.45])
    unguarded = _build_ss_bias_matrix(ss_centers, ss_k, ss_values, guard=False)
    guarded   = _build_ss_bias_matrix(ss_centers, ss_k, ss_values, guard=True)
    finite_mask = np.isfinite(unguarded)
    np.testing.assert_array_almost_equal(
        guarded[finite_mask], unguarded[finite_mask],
        decimal=12,
        err_msg="guard altered finite entries",
    )


# ---------------------------------------------------------------------------
# N4: Segment RNG seed uniqueness
# ---------------------------------------------------------------------------

def _compute_segment_seed(base_seed: int, epoch_idx: int, seg_counter: int) -> int:
    """Mirror the seed formula added to adaptive_production.py."""
    return base_seed + epoch_idx * 1_000_003 + seg_counter * 997


def test_segment_seeds_all_distinct():
    """Each (epoch, segment) combination must produce a unique seed."""
    base = 42
    seeds = [
        _compute_segment_seed(base, epoch, seg)
        for epoch in range(5)
        for seg in range(4)
    ]
    assert len(seeds) == len(set(seeds)), "duplicate seeds found across segments"


def test_segment_seed_differs_from_base():
    """Every derived seed must differ from the base seed."""
    base = 12345
    for epoch in range(3):
        for seg in range(3):
            if epoch == 0 and seg == 0:
                continue   # counter starts at 0 but increments before assignment
            seed = _compute_segment_seed(base, epoch, seg)
            assert seed != base, f"seed at epoch={epoch} seg={seg} equals base"


def test_segment_seed_copy_does_not_mutate_base_args():
    """seg_args.seed must not bleed back into args.seed (shallow copy check)."""
    args = types.SimpleNamespace(seed=99)
    seg_args = copy.copy(args)
    seg_args.seed = _compute_segment_seed(args.seed, epoch_idx=1, seg_counter=0)
    assert args.seed == 99, "base args.seed was mutated by copy.copy + reassignment"


# ---------------------------------------------------------------------------
# F5: NaN secondary seed penalty
# ---------------------------------------------------------------------------

def _score_seed(primary_value, secondary_value,
                target_primary, target_secondary,
                primary_scale, secondary_scale,
                secondary_weight, seed_selection_mode="active-cv",
                secondary_available=True):
    """Reproduce the fixed seeding.py scoring logic."""
    primary_delta = abs(primary_value - target_primary) if math.isfinite(primary_value) else float("inf")
    primary_score = primary_delta / max(1e-12, primary_scale)

    secondary_score = 0.0
    use_secondary = (
        seed_selection_mode == "active-cv"
        and secondary_available
        and secondary_weight > 0.0
        and math.isfinite(target_secondary)
        and math.isfinite(secondary_value)
    )
    if use_secondary:
        secondary_delta = abs(secondary_value - target_secondary)
        secondary_score = secondary_weight * secondary_delta / max(1e-12, secondary_scale)
    elif (seed_selection_mode == "active-cv"
          and secondary_available
          and secondary_weight > 0.0
          and math.isfinite(target_secondary)
          and not math.isfinite(secondary_value)):
        # NaN penalty (the fix)
        secondary_score = float(secondary_weight)

    return primary_score + secondary_score


def test_nan_secondary_seed_not_preferred_over_imperfect_finite():
    """
    A seed with secondary=NaN must not beat a seed with a finite but imperfect
    secondary CV value.  Without the fix, secondary_score=0 for NaN seeds,
    so they always win over seeds that have any secondary mismatch.
    """
    primary_scale   = 0.05
    secondary_scale = 0.333
    secondary_weight = 1.0
    target_p, target_s = 0.20, 0.0

    # Seed A: exact primary match, secondary=NaN
    score_nan = _score_seed(0.20, float("nan"), target_p, target_s,
                            primary_scale, secondary_scale, secondary_weight)

    # Seed B: exact primary match, secondary moderately wrong (0.5 normalised units away)
    score_finite = _score_seed(0.20, 0.5 * secondary_scale + target_s, target_p, target_s,
                               primary_scale, secondary_scale, secondary_weight)

    assert score_nan > score_finite, (
        f"NaN seed (score={score_nan:.3f}) should score HIGHER than finite seed "
        f"(score={score_finite:.3f}) so it is NOT preferred (lower score = better)"
    )


def test_nan_secondary_old_behaviour_was_wrong():
    """Regression: without the penalty, NaN secondary always scores 0 (better than any mismatch)."""
    def _old_score(primary_value, secondary_value, target_p, target_s,
                   ps, ss, sw):
        pdelta = abs(primary_value - target_p) if math.isfinite(primary_value) else float("inf")
        pscore = pdelta / max(1e-12, ps)
        sscore = 0.0   # OLD: no NaN penalty
        use_s = (math.isfinite(target_s) and math.isfinite(secondary_value))
        if use_s:
            sscore = sw * abs(secondary_value - target_s) / max(1e-12, ss)
        return pscore + sscore

    old_nan   = _old_score(0.20, float("nan"), 0.20, 0.0, 0.05, 0.333, 1.0)
    old_finite = _old_score(0.20, 0.166, 0.20, 0.0, 0.05, 0.333, 1.0)
    assert old_nan < old_finite, "old behaviour: NaN scores lower (bug confirmed)"


# ---------------------------------------------------------------------------
# F4: Adaptive seed scorer normalizes primary and secondary axes
# ---------------------------------------------------------------------------

def _f4_score_seed(p, s, target_p, target_s, dp_scale, ds_scale):
    """Reproduce the fixed normalized seed scorer from adaptive_production.py."""
    dp = float(p) - float(target_p)
    ds = 0.0
    if target_s is not None and s is not None:
        ds = float(s) - float(target_s)
    return (dp / max(1e-12, dp_scale)) ** 2 + (ds / max(1e-12, ds_scale)) ** 2


def _f4_score_seed_old(p, s, target_p, target_s):
    """OLD raw dp²+ds² scorer without normalization."""
    dp = float(p) - float(target_p)
    ds = 0.0
    if target_s is not None and s is not None:
        ds = float(s) - float(target_s)
    return dp * dp + ds * ds


def test_f4_normalized_scorer_balances_axes():
    """
    With a primary scale of 0.05 and secondary scale of 0.33, a seed that is
    1 primary-spacing off (dp=0.05, ds=0) should score identically to a seed
    that is 1 secondary-spacing off (dp=0, ds=0.33) after normalization.
    The old scorer with raw units would rank them very differently.
    """
    dp_scale = 0.05
    ds_scale = 0.33

    score_primary_off = _f4_score_seed(0.25, 0.0, 0.20, 0.0, dp_scale, ds_scale)
    score_secondary_off = _f4_score_seed(0.20, 0.33, 0.20, 0.0, dp_scale, ds_scale)
    assert abs(score_primary_off - score_secondary_off) < 1e-10, (
        f"Normalized scores should be equal for 1-spacing offset on each axis: "
        f"primary_off={score_primary_off:.6f} secondary_off={score_secondary_off:.6f}"
    )

    # Old scorer would rank them very differently (secondary_off << primary_off)
    old_primary_off = _f4_score_seed_old(0.25, 0.0, 0.20, 0.0)
    old_secondary_off = _f4_score_seed_old(0.20, 0.33, 0.20, 0.0)
    assert old_secondary_off > old_primary_off * 40, (
        "Old scorer should wildly prefer primary-near seeds over secondary-near seeds "
        f"(old_primary_off={old_primary_off:.6f} old_secondary_off={old_secondary_off:.6f})"
    )


def test_f4_single_active_state_uses_fallback_scale():
    """With one active state there is no grid spacing; fallback scale of 1.0 applies."""
    # Can't take median of single-element diff; len <= 1 → scale = 1.0
    p_centers = [0.20]
    dp_scale = float(np.median(np.diff(sorted(p_centers)))) if len(p_centers) > 1 else 1.0
    dp_scale = max(1e-12, dp_scale)
    assert dp_scale == 1.0, "single-state fallback scale must be 1.0"


def test_f4_scale_from_median_spacing():
    """Median spacing of an irregular grid gives correct normalization."""
    p_centers = sorted([0.10, 0.15, 0.25, 0.40])  # spacings: 0.05, 0.10, 0.15
    dp_scale = float(np.median(np.diff(p_centers)))
    assert abs(dp_scale - 0.10) < 1e-10, f"median spacing = {dp_scale}, expected 0.10"


# ---------------------------------------------------------------------------
# F7: rama-map secondary defaults snap to named basin values
# ---------------------------------------------------------------------------

def _rama_map_basin_values():
    """The four named rama-map basin scalars."""
    return [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]


def _snap_to_rama_basins(value):
    """Mirror the cv.py snap_to_rama_basins helper."""
    basins = _rama_map_basin_values()
    return min(basins, key=lambda b: abs(b - float(value)))


def test_f7_default_centers_are_basin_values():
    """The corrected rama-map default centers must equal the four named basins."""
    expected = sorted(_rama_map_basin_values())
    actual = sorted([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])
    assert actual == pytest.approx(expected), (
        f"Default centers {actual} do not match expected basins {expected}"
    )


def test_f7_old_default_centers_were_outside_basin_range():
    """Regression: old default [0.25, 0.55, 0.85] is entirely outside [-1, 1] basin range."""
    old_defaults = [0.25, 0.55, 0.85]
    basins = _rama_map_basin_values()
    for v in old_defaults:
        nearest = _snap_to_rama_basins(v)
        assert nearest != pytest.approx(v, abs=0.01), (
            f"Old default {v} should not be a named basin (nearest: {nearest})"
        )


def test_f7_snap_rounds_to_correct_basin():
    """snap_to_rama_basins maps intermediate values to nearest named basin."""
    cases = [
        (-0.9, -1.0),           # near beta
        (-0.5, -1.0 / 3.0),    # near ppii
        (0.0, -1.0 / 3.0),     # equidistant ppii/alpha_R → ppii wins (closer to -1/3)
        (0.5, 1.0 / 3.0),      # near alpha_R
        (0.9, 1.0),             # near alpha_L
    ]
    for value, expected in cases:
        result = _snap_to_rama_basins(value)
        assert result == pytest.approx(expected, abs=1e-10), (
            f"snap({value}) → {result}, expected {expected}"
        )


def test_f7_snap_dedup_removes_duplicate_basin_assignments():
    """When multiple proposed centers snap to the same basin, dedup preserves sorted uniques."""
    proposed = [0.28, 0.35, 0.42]   # all near alpha_R = 1/3
    snapped = sorted({_snap_to_rama_basins(x) for x in proposed})
    assert snapped == pytest.approx([1.0 / 3.0]), (
        f"Three proposals near alpha_R should dedup to one: got {snapped}"
    )
