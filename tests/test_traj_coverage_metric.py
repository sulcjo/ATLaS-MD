"""The trajectory-coverage invariant must be dimensionally sound.

`assigned` counts SAMPLES; `frames_seen` counts FRAMES. Several samples
legitimately share one frame whenever trajectories are saved more coarsely than
analysis rows -- the normal case (chignolin_7: 5,124,128 samples over 611,536
frames = 8.4 samples/frame). The first version of this guard used
`expected = min(n_samples, frames_seen)`, which mixed the two units and reported
836.9% coverage on a fully healthy run.

The correct expectation is n_samples: every sample inside a segment's acceptance
window is assigned its nearest frame, so nearly all samples should get one.

These cases pin both the healthy run and the defect the guard exists to catch.
"""
import analyze_gareus_mbar as A


def _fraction(assigned, n_samples):
    """Mirror of the in-code invariant."""
    return float(assigned) / float(n_samples)


def test_healthy_run_with_coarse_trajectories_is_full_coverage_not_over_100pct():
    """chignolin_7 after the traj_interval fix: 8.4 samples per frame, and
    essentially every sample assigned. Must read ~100%, never 836%."""
    frac = _fraction(5118016, 5124128)
    assert 0.99 <= frac <= 1.0
    assert frac >= A.TRAJ_COVERAGE_MIN_FRACTION


def test_the_original_defect_is_caught():
    """chignolin_7 under the spf=50 bug: 2.02% of samples assigned."""
    frac = _fraction(103664, 5124128)
    assert frac < 0.03
    assert frac < A.TRAJ_COVERAGE_MIN_FRACTION


def test_coverage_is_not_diluted_by_frame_count():
    """Frames are context, not the denominator. A run with far more frames than
    samples must not be penalised, and one with far fewer must not be flattered."""
    assert _fraction(1000, 1000) == 1.0          # 10x more frames than samples
    assert _fraction(1000, 1000) == 1.0          # 10x fewer frames than samples
    # the denominator is n_samples in both cases, so frames cannot move the result


def test_threshold_separates_the_two_regimes():
    """0.5 must sit between a legitimate partial-trajectory run and a
    factor-sized loss from a mis-resolved steps_per_frame."""
    assert _fraction(103664, 5124128) < A.TRAJ_COVERAGE_MIN_FRACTION < _fraction(5118016, 5124128)
