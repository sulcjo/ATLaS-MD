"""ladder_overlap_by_axis on a 2D (CV1, CV2) x lambda layout.

Regression for chignolin_9 (59 (CV1, CV2) centres x 4 rungs, 16 distinct CV1
values over 5 CV2 rows): grouping rungs by CV1 alone chained states of
different CV2 centres into one "lambda" chain (e.g. states 76-69, CV2 1.35 vs
-0.78), reporting 48 lambda pairs / 16 expected groups instead of 177 / 59, and
paired CV1 neighbours across CV2 rows for the "across CV1" axis.
"""
import numpy as np

from gareus.mbar_analysis.ladder_overlap import ladder_overlap_by_axis

# 3 centres x 2 rungs. Centres A and B share CV1 = 0.5 but sit in different
# CV2 rows; centre C is CV1 = 0.7 in A's CV2 row.
#   state: 0 A l0, 1 B l0, 2 C l0, 3 A l1, 4 B l1, 5 C l1
CEN = np.array([0.5, 0.5, 0.7, 0.5, 0.5, 0.7])
SEC = np.array([-1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
LAM = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

TRUE_RUNG = {(0, 3), (1, 4), (2, 5)}
TRUE_CV1 = {(0, 2), (3, 5)}  # same rung, same CV2 row, adjacent CV1


def _pair_ov(a, b):
    a, b = sorted((a, b))
    if (a, b) in TRUE_RUNG:
        return 0.30
    if (a, b) in TRUE_CV1:
        return 0.25
    return 0.01  # any cross-CV2 pairing is a bogus edge and must not appear


def _pairs(axis):
    return {tuple(sorted((int(a), int(b)))) for a, b, _ in axis["pairs"]}


def test_lambda_pairs_share_both_centres():
    out, warnings = ladder_overlap_by_axis(None, LAM, CEN, pair_overlap=_pair_ov, secondary_centers=SEC)
    assert warnings == []
    lam = out["lambda_direction"]
    assert _pairs(lam) == TRUE_RUNG
    assert lam["worst"] == 0.30
    assert lam["expected_components"] == 3
    assert lam["connected"] is True


def test_cv1_pairs_stay_within_their_cv2_row():
    out, _ = ladder_overlap_by_axis(None, LAM, CEN, pair_overlap=_pair_ov, secondary_centers=SEC)
    cv1 = out["cv1_direction"]
    assert _pairs(cv1) == TRUE_CV1
    assert cv1["worst"] == 0.25
    # (rung, CV2 row) groups: l0/-1, l0/+1, l1/-1, l1/+1; B's rows hold one state each
    assert cv1["expected_components"] == 4
    assert cv1["connected"] is True


def test_without_secondary_centres_the_1d_behaviour_is_unchanged():
    cen = np.array([0.1, 0.1, 0.5, 0.5])
    lam = np.array([0.0, 1.0, 0.0, 1.0])
    out, _ = ladder_overlap_by_axis(None, lam, cen, pair_overlap=lambda a, b: 0.3)
    assert _pairs(out["lambda_direction"]) == {(0, 1), (2, 3)}
    assert _pairs(out["cv1_direction"]) == {(0, 2), (1, 3)}


def test_nan_secondary_centres_behave_like_1d():
    cen = np.array([0.1, 0.1, 0.5, 0.5])
    lam = np.array([0.0, 1.0, 0.0, 1.0])
    out, _ = ladder_overlap_by_axis(None, lam, cen, pair_overlap=lambda a, b: 0.3,
                                    secondary_centers=np.full(4, np.nan))
    assert _pairs(out["lambda_direction"]) == {(0, 1), (2, 3)}
    assert _pairs(out["cv1_direction"]) == {(0, 2), (1, 3)}


def test_short_secondary_centres_degrade_to_named_warning():
    out, warnings = ladder_overlap_by_axis(None, LAM, CEN, pair_overlap=_pair_ov,
                                           secondary_centers=SEC[:3])
    assert out["lambda_direction"]["n_pairs"] == 0
    assert "length mismatch" in warnings[0]


def test_cv1_unrestrained_windows_stay_on_lambda_axis_but_leave_cv1_axis():
    # One CV2 row, one rung pair: CV1 centres 0.4, 0.5 (k1 = 0: CV2-only window,
    # its CV1 "centre" is a placeholder), 0.6.  chignolin_8/9 sparse layout.
    cen = np.array([0.4, 0.5, 0.6, 0.4, 0.5, 0.6])
    sec = np.zeros(6)
    lam = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    k1 = np.array([300.0, 0.0, 300.0, 300.0, 0.0, 300.0])
    out, _ = ladder_overlap_by_axis(None, lam, cen, pair_overlap=lambda a, b: 0.3,
                                    secondary_centers=sec, primary_k=k1)
    assert _pairs(out["lambda_direction"]) == {(0, 3), (1, 4), (2, 5)}
    assert _pairs(out["cv1_direction"]) == {(0, 2), (3, 5)}
    assert out["cv1_direction"]["expected_components"] == 2  # one per rung
    assert out["cv1_direction"]["connected"] is True
