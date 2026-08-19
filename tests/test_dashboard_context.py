import argparse
import collections
import math

from gareus.dashboard.context import (
    DashboardContext,
    acceptance_by_pair,
    acceptance_by_window,
    build_context,
    delta_by_window,
    overlap_by_pair,
)
from gareus.dashboard.sidecar import SidecarSnapshot
from gareus.logger import DistanceLogger


def _rows(n=4):
    return [{"replica": i, "window": i, "center_A": 4.0 + 0.55 * i,
             "k_kcal_mol_A2": 2.5, "cv_A": 4.0 + 0.55 * i + 0.1,
             "umbrella_bias_kcal_mol": 0.01, "umbrella_pull_kcal_mol_A": 0.2}
            for i in range(n)]


def test_acceptance_by_pair_converts_attempt_counts_to_rates():
    pairs = acceptance_by_pair({"0-1": {"attempts": 50, "accepted": 10},
                                "1-2": {"attempts": 0, "accepted": 0}})
    assert pairs[(0, 1)] == 0.2
    assert math.isnan(pairs[(1, 2)])            # no attempts yet is unknown, not zero


def test_acceptance_by_window_takes_the_worst_neighbour_of_each_window():
    pairs = {(0, 1): 0.40, (1, 2): 0.05}
    per_window = acceptance_by_window(pairs, n_windows=3)
    assert per_window[0] == 0.40
    assert per_window[1] == 0.05                # worst of its two sides
    assert per_window[2] == 0.05


def test_overlap_by_pair_is_high_for_identical_and_low_for_disjoint_histories():
    history = {0: (1.0, 1.05, 1.1, 1.15, 1.2, 1.25),
               1: (1.0, 1.05, 1.1, 1.15, 1.2, 1.25),
               2: (8.5, 8.55, 8.6, 8.65, 8.7, 8.75)}
    ov = overlap_by_pair(history, centers_a=(1.0, 1.1, 9.0))
    assert ov[(0, 1)] > 0.9
    assert ov[(1, 2)] < 0.1


def test_delta_by_window_is_the_signed_distance_from_the_restraint_centre():
    d = delta_by_window(_rows(2))
    assert math.isclose(d[0], 0.1, abs_tol=1e-9)


def test_build_context_copies_histories_into_immutable_tuples(tmp_path):
    args = argparse.Namespace(timestep_fs=2.0, temperature_k=310.0)
    logger = DistanceLogger(tmp_path, args, no_file_persistence=True)
    logger.history_by_window[0] = collections.deque([1.0, 2.0])
    ctx = build_context(
        logger=logger, rows=_rows(), phase="gareus_production", step=100,
        total_steps=1000, summary={}, dashboard_info={"centers_a": [4.0, 4.55, 5.1, 5.65],
                                                     "n_windows": 4},
        sidecar=SidecarSnapshot(), term_w=140, term_h=45, now=1000.0,
        view="progress", glyphs="unicode",
    )
    assert isinstance(ctx, DashboardContext)
    assert ctx.cv_history_by_window[0] == (1.0, 2.0)
    logger.history_by_window[0].append(3.0)
    assert ctx.cv_history_by_window[0] == (1.0, 2.0)     # snapshot, not a live view
    assert ctx.temperature_k == 310.0
    assert ctx.n_windows == 4
    assert ctx.is_2d is False
    # render_distance_ascii needs per-replica history and the ascii knobs too.
    assert ctx.cv_history_by_replica == {}
    assert (ctx.ascii_mode, ctx.ascii_max_replicas) == ("hist3d", 32)


def test_build_context_marks_a_2d_run_and_records_its_secondary_cv_type(tmp_path):
    args = argparse.Namespace(timestep_fs=2.0)
    logger = DistanceLogger(tmp_path, args, no_file_persistence=True)
    ctx = build_context(
        logger=logger, rows=_rows(), phase="gareus_production", step=1, total_steps=10,
        summary={}, dashboard_info={
            "centers_a": [4.0, 4.55, 5.1, 5.65], "n_windows": 4,
            "secondary_cv_centers": [-1.0, 1.0, -1.0, 1.0],
            "secondary_cv": {"explicit_2d_windows": True, "grid": False, "type": "tica-linear"},
        },
        sidecar=SidecarSnapshot(), term_w=140, term_h=45, now=1.0,
        view="physics", glyphs="unicode",
    )
    assert ctx.is_2d is True
    assert ctx.secondary_cv_type == "tica-linear"
    assert ctx.topology_label == "sparse explicit 2D"


def test_build_context_prefers_the_sidecar_temperature_when_args_lack_one(tmp_path):
    logger = DistanceLogger(tmp_path, argparse.Namespace(), no_file_persistence=True)
    ctx = build_context(
        logger=logger, rows=_rows(), phase="p", step=1, total_steps=2, summary={},
        dashboard_info={"centers_a": [4.0], "n_windows": 1},
        sidecar=SidecarSnapshot(gamd={"temperature_K": 277.0}),
        term_w=100, term_h=30, now=1.0, view="progress", glyphs="ascii",
    )
    assert ctx.temperature_k == 277.0
