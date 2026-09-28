"""Index-adjacent window overlap is graded only when index order is CV order.

chignolin_9 (59 (CV1, CV2) centres x 4 rungs) read "Window overlap FAIL" from
"worst 0.000 CV1-marginal index-adjacent (windows 7-8)": consecutive state ids
there are not neighbours in CV space at all, while the CV-space pairings
(joint 0.367, marginal 0.668) and both connectivity checks passed.
"""
import numpy as np

import gareus_report as gr
from gareus.mbar_analysis.pmf import index_order_is_cv_order


def test_monotone_1d_ladder_with_rung_repeats_is_cv_ordered():
    cen = np.repeat([0.1, 0.2, 0.3], 4)          # centre-major, 4 rungs each
    assert index_order_is_cv_order(cen, np.full(12, np.nan)) is True


def test_decreasing_1d_ladder_is_cv_ordered():
    assert index_order_is_cv_order(np.array([0.9, 0.7, 0.5]), None) is True


def test_shuffled_1d_ladder_is_not_cv_ordered():
    assert index_order_is_cv_order(np.array([0.1, 0.5, 0.3]), None) is False


def test_varying_cv2_is_not_cv_ordered_even_if_cv1_is_monotone():
    cen = np.array([0.1, 0.1, 0.2, 0.2])
    sec = np.array([-1.0, 1.0, -1.0, 1.0])
    assert index_order_is_cv_order(cen, sec) is False


def test_constant_cv2_behaves_like_1d():
    cen = np.array([0.1, 0.2, 0.3])
    assert index_order_is_cv_order(cen, np.array([0.5, 0.5, 0.5])) is True


def _summary(flag, index_worst=0.0):
    return {
        "n_samples": 1_000_000, "n_windows": 10,
        "neighbor_overlap": [0.6] * 4 + [index_worst] + [0.6] * 4,
        "cv_space_neighbor_overlap": [{"window": 0, "neighbor": 1, "overlap": 0.66},
                                      {"window": 1, "neighbor": 0, "overlap": 0.66}],
        **({} if flag is None else {"index_order_is_cv_order": flag}),
    }


def _window_overlap(s):
    v = gr.build_health_verdict(s, 0.30)
    return next(c for c in v["checks"] if c["name"] == "Window overlap")


def test_index_adjacent_is_reported_but_not_graded_when_index_is_not_cv_order():
    ov = _window_overlap(_summary(False))
    assert ov["status"] == "pass", ov
    assert "index-adjacent" in ov["detail"] and "not graded" in ov["detail"]
    assert "0.000" in ov["detail"]


def test_index_adjacent_is_still_graded_on_a_cv_ordered_ladder():
    ov = _window_overlap(_summary(True))
    assert ov["status"] == "fail", ov


def test_summary_without_the_flag_keeps_grading_index_adjacency():
    ov = _window_overlap(_summary(None))
    assert ov["status"] == "fail", ov


def test_index_adjacent_is_graded_when_it_is_the_only_number():
    s = _summary(False)
    del s["cv_space_neighbor_overlap"]
    ov = _window_overlap(s)
    assert ov["status"] == "fail", ov

