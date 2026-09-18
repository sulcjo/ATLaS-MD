"""The ladder-axis report must grade MBAR state overlap at the state-overlap
calibration, not at the CV-histogram one.

`ladder_overlap_by_axis` grades `symmetric_state_overlap = sqrt(O_ab * O_ba)`
from `mbar_state_overlap`. It used to default to 0.30 and the analysis call site
passed `--min-neighbor-overlap`, which targets CV1-marginal
histogram-intersection overlap -- a different quantity on a different scale
(`gareus.mbar_analysis.pmf` says so of overlap_matrix.csv explicitly). On
chignolin_7's 64 states that graded every pair on both axes as failing
(0/48 lambda, 0/60 CV1) while the CV1-marginal check on the same axis passed at
worst 0.491 / connected -- an internal contradiction, not a measurement.

0.15 is the project's own calibration for THIS metric: the adaptive driver
gates rung edges at `AdaptiveDecisionPolicy.min_rung_overlap`, calibrated on the
S3 pilot's measured adjacent-rung entries 0.240-0.298
(docs/superpowers/specs/2026-09-07-adaptive-ladder-rungs-design.md:18, line 107
recording that symmetrising O_ij leaves that calibration valid).

The load-bearing check here is the drift guard: the analysis side and the driver
side must keep grading the same measurement at the same number.
"""
import inspect

from gareus.adaptive_production import AdaptiveDecisionPolicy
from gareus.mbar_analysis.ladder_overlap import (
    LADDER_STATE_OVERLAP_MIN,
    ladder_overlap_by_axis,
    ladder_overlap_health_checks,
)


def test_analysis_threshold_matches_the_driver_rung_gate():
    """The bug class was the wrong NUMBER ARRIVING, not wrong grading logic.
    If these two ever diverge, the analysis report and the adaptive driver
    disagree about the same measurement on the same run."""
    assert LADDER_STATE_OVERLAP_MIN == AdaptiveDecisionPolicy().min_rung_overlap


def test_function_default_is_the_shared_constant():
    assert inspect.signature(ladder_overlap_by_axis).parameters['thr'].default == LADDER_STATE_OVERLAP_MIN


def test_cli_default_is_the_shared_constant():
    """The flag exists and defaults to the same constant -- catches the case
    where the argparse default is re-typed as a literal and drifts."""
    import analyze_gareus_mbar as A

    args = A.parse_args(['run']) if hasattr(A, 'parse_args') else None
    if args is None:  # pragma: no cover - parser shape changed
        import pytest
        pytest.skip('analyze_gareus_mbar exposes no parse_args')
    assert args.min_ladder_state_overlap == LADDER_STATE_OVERLAP_MIN


def test_cli_flag_overrides_the_default():
    import analyze_gareus_mbar as A

    args = A.parse_args(['run', '--min-ladder-state-overlap', '0.22'])
    assert args.min_ladder_state_overlap == 0.22


def _axis(worst, connected=True, n_components=1, expected=1):
    return {
        'pairs': [(0, 1, worst)], 'worst': worst, 'worst_pair': (0, 1), 'n_pairs': 1,
        'n_components': n_components, 'connected': connected,
        'expected_components': expected,
    }


def _statuses(lo, thr):
    return {c['name']: c['status'] for c in ladder_overlap_health_checks(lo, thr)}


def test_grading_boundaries_around_the_threshold():
    """OVERLAP_FAIL_FRACTION is 0.5, so at thr=0.15 the bands are
    FAIL < 0.075 <= CAUTION < 0.15 <= PASS."""
    thr = LADDER_STATE_OVERLAP_MIN
    for worst, expected in ((0.20, 'pass'), (0.15, 'pass'), (0.10, 'caution'),
                            (0.074, 'fail')):
        lo = {'lambda_direction': _axis(worst), 'cv1_direction': _axis(worst)}
        got = _statuses(lo, thr)['Overlap along λ']
        assert got == expected, f'worst={worst} at thr={thr} graded {got}, expected {expected}'


def test_chignolin7_measured_values_are_not_all_failing_at_the_calibrated_threshold():
    """Regression on the symptom: chignolin_7's measured medians (lambda 0.0854
    on its weakest edge, CV1 0.1388) are CAUTION at 0.15, not FAIL. Under the
    old CV-histogram 0.30 every one of them graded FAIL, which is what made the
    report unreadable."""
    thr = LADDER_STATE_OVERLAP_MIN
    lo = {'lambda_direction': _axis(0.0854), 'cv1_direction': _axis(0.1388)}
    st = _statuses(lo, thr)
    assert st['Overlap along λ'] == 'caution'
    assert st['Overlap across CV1'] == 'caution'
