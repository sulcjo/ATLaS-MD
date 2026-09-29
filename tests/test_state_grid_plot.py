"""Spec 3.7: data prep for the explicit-state-coordinate figure (gareus.adaptive.state_grid_plot)."""
from __future__ import annotations

import pandas as pd

from gareus.adaptive import state_grid_plot as sg

REG = [
    {"state_id": 0, "active": True, "primary_center": 0.5, "primary_k": 0.0, "secondary_center": -0.25,
     "secondary_k": 0.0, "gamd_lambda": 0.0},                                      # anchor, both placeholders
    {"state_id": 1, "active": True, "primary_center": 0.3, "primary_k": 300.0, "secondary_center": 9.9,
     "secondary_k": 0.0, "gamd_lambda": 0.0},                                      # CV1-only
    {"state_id": 2, "active": "True", "primary_center": 0.7, "primary_k": 300.0, "secondary_center": 1.3,
     "secondary_k": 1.2, "gamd_lambda": 0.2348921},                               # both, rung 2
    {"state_id": 3, "active": "False", "primary_center": 0.7, "primary_k": 300.0, "secondary_center": 1.0,
     "secondary_k": 1.2, "gamd_lambda": 0.0},                                      # retired
    {"state_id": 4, "active": True, "primary_center": 0.6, "primary_k": 0.0, "secondary_center": 0.3,
     "secondary_k": 1.2, "gamd_lambda": 0.0},                                      # CV2-only, no summary row
]
SUMMARY = {"states": [{"state_id": 0, "cv1_mean": 0.57, "cv2_mean": -0.45},
                      {"state_id": 1, "cv1_mean": 0.31, "cv2_mean": -0.6},
                      {"state_id": 2, "cv1_mean": 0.69, "cv2_mean": 1.1, "trapped_or_orthogonal": True}],
           "edges": [{"state_i": 1, "state_j": 2, "graded": True, "weak": True, "measured": True},
                     {"state_i": 0, "state_j": 1, "graded": True, "weak": False, "measured": False},
                     {"state_i": 1, "state_j": 4, "graded": True, "weak": False, "measured": True},
                     {"state_i": 0, "state_j": 2, "graded": False, "weak": None, "measured": None}]}


def _by_sid(pts):
    return {p["state_id"]: p for p in pts}


def test_inactive_states_are_dropped_and_rungs_rounded():
    pts = _by_sid(sg.state_grid_points(REG, {0: 10, 1: 20, 2: 30}, SUMMARY))
    assert sorted(pts) == [0, 1, 2, 4]
    assert pts[2]["lambda"] == 0.234892 and pts[0]["samples"] == 10 and pts[4]["samples"] == 0


def test_unrestrained_axes_use_the_sampled_mean_restrained_axes_the_centre():
    pts = _by_sid(sg.state_grid_points(REG, {}, SUMMARY))
    assert (pts[0]["x"], pts[0]["y"]) == (0.57, -0.45) and pts[0]["sampled_position"] and not pts[0]["placeholder"]
    assert (pts[1]["x"], pts[1]["y"]) == (0.3, -0.6) and pts[1]["sampled_position"]
    assert (pts[2]["x"], pts[2]["y"]) == (0.7, 1.3) and not pts[2]["sampled_position"]


def test_no_sampled_mean_leaves_the_placeholder_flagged():
    pts = _by_sid(sg.state_grid_points(REG, {}, None))
    assert (pts[4]["x"], pts[4]["y"]) == (0.6, 0.3) and pts[4]["placeholder"] and not pts[4]["sampled_position"]
    assert pts[1]["y"] == 9.9 and pts[1]["placeholder"]


def test_markers_follow_the_restraint_pattern():
    pts = _by_sid(sg.state_grid_points(REG, {}, SUMMARY))
    assert [pts[s]["marker"] for s in (0, 1, 2, 4)] == ["D", "s", "o", "^"]
    assert pts[2]["trapped"] and not pts[1]["trapped"]


def test_registry_dataframe_records_work():
    df = pd.DataFrame(REG)
    assert len(sg.state_grid_points(df.to_dict("records"), {}, None)) == 4


def test_flagged_edges_weak_and_unmeasured_only():
    assert sg.flagged_edges(SUMMARY) == [{"i": 1, "j": 2, "kind": "weak"}, {"i": 0, "j": 1, "kind": "unmeasured"}]
    assert sg.flagged_edges(None) == []


def test_count_range_handles_lockstep_counts():
    assert sg._count_range([{"samples": 100}, {"samples": 100}]) == (50.0, 200.0)
    assert sg._count_range([{"samples": 0}, {"samples": 10}, {"samples": 40}]) == (10, 40)
    assert sg._count_range([{"samples": 0}]) == (0.5, 2.0)
