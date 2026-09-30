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


# ---- connectivity counts slots, not states (verification F3) ----------------------------------

def test_tied_states_on_a_row_without_a_neighbour_are_not_a_split():
    # rows: CV2 -1 holds A_old/A_new (tied, no other CV1 slot); CV2 0 holds C and D (a chain)
    cen = np.array([0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.7, 0.7])
    sec = np.array([-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    lam = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    k2 = np.array([1.18, 1.18, 2.0, 2.0, 1.18, 1.18, 1.18, 1.18])
    out, _ = ladder_overlap_by_axis(None, lam, cen, pair_overlap=lambda a, b: 0.40, thr=0.15,
                                    secondary_centers=sec, primary_k=np.full(8, 100.0), secondary_k=k2)
    cv1 = out["cv1_direction"]
    assert cv1["expected_components"] == 4 and cv1["n_components"] == 4 and cv1["connected"] is True
    assert cv1["n_tied_states"] == 2                      # A_old ~ A_new at both rungs
    assert out["lambda_direction"]["connected"] is True


def test_a_real_break_between_distinct_slots_stays_disconnected():
    def ov(a, b):                                          # A (0/2 tied) -- C (4) cut at rung 0
        return 0.02 if {a, b} & {0, 2} and 4 in (a, b) else 0.40
    out, _ = ladder_overlap_by_axis(None, LAM, CEN, pair_overlap=ov, thr=0.15, secondary_centers=SEC,
                                    primary_k=K1, secondary_k=K2)
    cv1 = out["cv1_direction"]
    assert cv1["connected"] is False and cv1["n_components"] == cv1["expected_components"] + 1
    from gareus.mbar_analysis.ladder_overlap import ladder_overlap_health_checks
    row = next(c for c in ladder_overlap_health_checks(out, 0.15) if c["name"] == "Connectivity across CV1")
    assert row["status"] == "fail"                        # 0.02 < OVERLAP_FAIL_FRACTION x 0.15


def test_one_tied_member_bridging_is_enough():
    def ov(a, b):                                          # the retired A_old is poor, A_new bridges
        return 0.05 if 0 in (a, b) or 1 in (a, b) else 0.40
    out, _ = _run_with(ov)
    assert out["cv1_direction"]["connected"] is True
    assert out["cv1_direction"]["worst"] == 0.05          # the summary still reports the poor pair


def _run_with(ov):
    return ladder_overlap_by_axis(None, LAM, CEN, pair_overlap=ov, thr=0.15, secondary_centers=SEC,
                                  primary_k=K1, secondary_k=K2)
