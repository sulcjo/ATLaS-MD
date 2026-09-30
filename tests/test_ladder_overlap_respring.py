"""ladder_overlap_by_axis after a CV2 respring (--ap-cv2-respring).

A respring retires a centre's states and adds new ones at the SAME (CV1, CV2) centre with a
new k2 on every rung; the retired states keep usable_for_mbar, so both sit in the union.
Rungs of one centre share their springs, so the lambda axis must group by centre + restrained
springs (as ladder_adapt._centre_spring_key does), never pair a retired rung with a new one.
On the CV1/CV2 axes the old and new states tie on the chain coordinate; both must still be
linked to their spatial neighbours.
"""
import numpy as np

from gareus.mbar_analysis.ladder_overlap import ladder_overlap_by_axis

# state: 0 A_old l0, 1 A_old l1, 2 A_new l0, 3 A_new l1, 4 C l0, 5 C l1
CEN = np.array([0.5, 0.5, 0.5, 0.5, 0.7, 0.7])
SEC = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
LAM = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
K1 = np.full(6, 100.0)
K2 = np.array([1.18, 1.18, 2.0, 2.0, 1.18, 1.18])


def _pairs(axis):
    return {tuple(sorted((int(a), int(b)))) for a, b, _ in axis["pairs"]}


def _run(k2=K2):
    return ladder_overlap_by_axis(None, LAM, CEN, pair_overlap=lambda a, b: 0.3,
                                  secondary_centers=SEC, primary_k=K1, secondary_k=k2)


def test_lambda_pairs_never_cross_springs():
    out, warnings = _run()
    assert warnings == []
    lam = out["lambda_direction"]
    assert _pairs(lam) == {(0, 1), (2, 3), (4, 5)}
    assert lam["expected_components"] == 3
    assert lam["connected"] is True


def test_cv1_axis_links_both_old_and_new_state_to_the_neighbour():
    out, _ = _run()
    cv1 = out["cv1_direction"]
    assert _pairs(cv1) == {(0, 4), (2, 4), (1, 5), (3, 5)}
    assert cv1["connected"] is True


def test_cv2_axis_links_both_old_and_new_state_to_the_neighbour():
    sec = np.array([-1.0, -1.0, -1.0, -1.0, 0.0, 0.0])      # C one CV2 row up, same CV1 column
    cen = np.full(6, 0.5)
    out, _ = ladder_overlap_by_axis(None, LAM, cen, pair_overlap=lambda a, b: 0.3,
                                    secondary_centers=sec, primary_k=K1, secondary_k=K2)
    assert _pairs(out["lambda_direction"]) == {(0, 1), (2, 3), (4, 5)}
    assert _pairs(out["cv2_direction"]) == {(0, 4), (2, 4), (1, 5), (3, 5)}


def test_without_a_respring_grouping_matches_the_springless_call():
    k2 = np.full(6, 1.18)
    cen = np.array([0.3, 0.3, 0.5, 0.5, 0.7, 0.7])
    with_k, _ = ladder_overlap_by_axis(None, LAM, cen, pair_overlap=lambda a, b: 0.3,
                                       secondary_centers=SEC, primary_k=K1, secondary_k=k2)
    without_k, _ = ladder_overlap_by_axis(None, LAM, cen, pair_overlap=lambda a, b: 0.3,
                                          secondary_centers=SEC)
    for axis in ("lambda_direction", "cv1_direction", "cv2_direction"):
        assert _pairs(with_k[axis]) == _pairs(without_k[axis])
        assert with_k[axis]["expected_components"] == without_k[axis]["expected_components"]


def test_shape_layout_k2_varying_along_a_row_keeps_the_cv1_chain():
    k2 = np.array([1.1, 1.1, 2.7, 2.7, 13.4, 13.4])         # per-centre k2 (3.2 shape layout)
    cen = np.array([0.3, 0.3, 0.5, 0.5, 0.7, 0.7])
    out, _ = ladder_overlap_by_axis(None, LAM, cen, pair_overlap=lambda a, b: 0.3,
                                    secondary_centers=SEC, primary_k=K1, secondary_k=k2)
    assert _pairs(out["cv1_direction"]) == {(0, 2), (2, 4), (1, 3), (3, 5)}
    assert out["cv1_direction"]["connected"] is True
