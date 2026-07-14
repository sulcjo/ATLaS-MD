"""
Regression tests for the chignolin_2d_run7 "frontier blindly commits" bug.

Root cause (see CLAUDE.md / gareus/adaptive_feedback.py history): the 2D
adaptive-feedback dispatchers read secondary-CV samples from a legacy flat
``samples.csv`` (column ``secondary_cv``) that current production no longer
writes (samples now live in ``samples/seg_*/*.parquet`` with columns
``cv1``/``cv2``). With no ``samples.csv`` on disk, ``n_secondary_samples`` was
silently 0 for every window every round, so the per-window off-center
diagnostic (``cell_status`` == "off_center"/"low_2d_hit") could never fire --
it was permanently stuck at "insufficient_samples". Separately, even when
cell diagnostics work, the edge-based overlap/exchange-acceptance check that
actually drives convergence/patch proposals cannot detect a frontier window
whose umbrella force constant is too weak to hold it near its own target: its
samples just drift back into a neighbor's basin, which makes pairwise overlap
between the two windows look perfectly healthy.

These tests cover both fixes:
  1. A secondary-axis fallback (mirroring the existing primary-axis fallback)
     so cell/edge secondary-sample counts reflect the live in-memory history
     instead of always reading as empty.
  2. ``_adaptive_feedback_2d_flag_off_center_edges``, which cross-references
     per-window off-center status onto edges so a deceptively-"ok" edge
     touching a broken window is escalated into the patch-candidate pipeline.
"""
import numpy as np
import pytest

from gareus.synth.argspec import make_feedback_args
from gareus.adaptive_feedback import (
    _adaptive_feedback_2d_cell_diagnostics,
    _adaptive_feedback_2d_edge_diagnostics_from_explicit_edges,
    _adaptive_feedback_2d_flag_off_center_edges,
    _adaptive_feedback_2d_sparse_patch_candidates,
    run_adaptive_feedback_dispatcher_2d_explicit_sparse,
)

# 2 primary anchors (0.0 "near", 0.8 "frontier") x 3 secondary anchors,
# mirroring chignolin_2d_run7's actual shape (4 primary x 6 secondary,
# collapsed here to the minimal case that reproduces the bug).
PRIMARY_CENTERS = np.array([0.0, 0.0, 0.0, 0.8, 0.8, 0.8])
SECONDARY_CENTERS = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
K_LIST = [75.0, 75.0, 75.0, 11.0, 11.0, 11.0]
SEC_K_LIST = [5.0] * 6


def _synthetic_samples(rng):
    """Near group (0,1,2) sits at its own center; frontier group (3,4,5)'s
    umbrella (weak k=11) is too weak to hold it near 0.8 -- its samples drift
    back and substantially overlap the near group's range, exactly like the
    real chignolin_2d_run7 windows 18-23."""
    samples_d, samples_s = {}, {}
    for w in range(6):
        if w < 3:
            samples_d[w] = rng.normal(0.05, 0.05, 80).tolist()
        else:
            samples_d[w] = rng.normal(0.15, 0.15, 80).tolist()
        samples_s[w] = rng.normal(SECONDARY_CENTERS[w], 0.15, 80).tolist()
    return samples_d, samples_s


def test_off_center_cell_only_detected_with_secondary_samples():
    """Without secondary samples, cell_status is stuck at insufficient_samples
    (the bug); with them, the frontier group is correctly diagnosed."""
    rng = np.random.default_rng(0)
    samples_d, samples_s = _synthetic_samples(rng)
    args = make_feedback_args(secondary_cv_centers=list(SECONDARY_CENTERS))

    broken = _adaptive_feedback_2d_cell_diagnostics(
        samples_d, {w: [] for w in range(6)}, PRIMARY_CENTERS, SECONDARY_CENTERS, K_LIST, SEC_K_LIST, args,
    )
    assert all(r["cell_status"] == "insufficient_samples" for r in broken), (
        "with no secondary samples every cell should read insufficient_samples (the bug)"
    )

    fixed = _adaptive_feedback_2d_cell_diagnostics(
        samples_d, samples_s, PRIMARY_CENTERS, SECONDARY_CENTERS, K_LIST, SEC_K_LIST, args,
    )
    near = [r for r in fixed if r["window"] < 3]
    frontier = [r for r in fixed if r["window"] >= 3]
    assert all(r["cell_status"] == "ok" for r in near)
    assert all(r["cell_status"] in {"off_center", "low_2d_hit"} for r in frontier), (
        "frontier windows whose samples never approach their own center must be flagged, not silently ok"
    )


def test_edge_diagnostics_alone_miss_off_center_frontier_window():
    """The raw edge-overlap check is fooled: overlap between a well-behaved
    window and a weak-k frontier window that drifted into its range reads as
    'ok', even though the frontier window itself never sits near its target."""
    rng = np.random.default_rng(0)
    samples_d, samples_s = _synthetic_samples(rng)
    args = make_feedback_args(secondary_cv_centers=list(SECONDARY_CENTERS))

    edge_rows = _adaptive_feedback_2d_edge_diagnostics_from_explicit_edges(
        samples_d, samples_s, PRIMARY_CENTERS, SECONDARY_CENTERS,
        {"accepted": 0, "attempts": 0, "pairs": {}}, 0.30, args,
    )
    primary_axis_edges = [r for r in edge_rows if r["window_i"] < 3 <= r["window_j"]]
    assert primary_axis_edges, "expected at least one edge bridging the near/frontier groups"
    assert all(r["edge_status"] == "ok" for r in primary_axis_edges), (
        "pairwise overlap alone must NOT catch the off-center frontier window "
        "(this is the deceptive-overlap failure mode being guarded against)"
    )


def test_flag_off_center_edges_escalates_deceptive_ok_edges():
    """_adaptive_feedback_2d_flag_off_center_edges must downgrade exactly the
    edges touching an off-center window, and leave everything else alone."""
    rng = np.random.default_rng(0)
    samples_d, samples_s = _synthetic_samples(rng)
    args = make_feedback_args(secondary_cv_centers=list(SECONDARY_CENTERS))

    cell_rows = _adaptive_feedback_2d_cell_diagnostics(
        samples_d, samples_s, PRIMARY_CENTERS, SECONDARY_CENTERS, K_LIST, SEC_K_LIST, args,
    )
    edge_rows = _adaptive_feedback_2d_edge_diagnostics_from_explicit_edges(
        samples_d, samples_s, PRIMARY_CENTERS, SECONDARY_CENTERS,
        {"accepted": 0, "attempts": 0, "pairs": {}}, 0.30, args,
    )
    off_center_windows = {r["window"] for r in cell_rows if r["cell_status"] in {"off_center", "low_2d_hit"}}
    assert off_center_windows == {3, 4, 5}

    flagged = _adaptive_feedback_2d_flag_off_center_edges(edge_rows, cell_rows)
    by_edge = {r["edge"]: r for r in flagged}
    for row in edge_rows:
        edge = row["edge"]
        touches_bad = row["window_i"] in off_center_windows or row["window_j"] in off_center_windows
        if row["edge_status"] == "ok" and touches_bad:
            assert by_edge[edge]["edge_status"] == "off_center_window", edge
        elif row["edge_status"] != "ok":
            # never escalate away from an already-flagged worse status
            assert by_edge[edge]["edge_status"] == row["edge_status"], edge
        else:
            assert by_edge[edge]["edge_status"] == "ok", edge


def test_flag_off_center_edges_is_noop_when_nothing_off_center():
    """No off-center cells -> edge rows pass through completely unchanged
    (no spurious patches on an otherwise healthy run)."""
    edge_rows = [
        {"edge": "0-1", "window_i": 0, "window_j": 1, "edge_status": "ok"},
        {"edge": "1-2", "window_i": 1, "window_j": 2, "edge_status": "low_overlap"},
    ]
    cell_rows = [{"window": w, "cell_status": "ok"} for w in range(3)]
    result = _adaptive_feedback_2d_flag_off_center_edges(edge_rows, cell_rows)
    assert result == edge_rows


def test_sparse_patch_candidates_treats_off_center_window_as_bad_status():
    """A defect row with edge_status='off_center_window' and
    defect_scope='local_patch_candidate' must produce a patch candidate, same
    as the existing low_overlap/low_exchange statuses."""
    defect = {
        "edge": "1-4", "window_i": 1, "window_j": 4,
        "edge_type": "distance_axis+knn_geometry",
        "defect_scope": "local_patch_candidate",
        "edge_status": "off_center_window",
        "local_patch_center_A": 0.4, "local_patch_secondary_cv_center": 0.0,
        "decision_overlap": 0.42, "target_overlap": 0.30,
        "exchange_acceptance": 0.9, "low_exchange_cut": 0.25,
    }
    args = make_feedback_args(secondary_cv_centers=list(SECONDARY_CENTERS))
    patch_rows, explicit_rows, skipped = _adaptive_feedback_2d_sparse_patch_candidates(
        [defect], [], list(PRIMARY_CENTERS), K_LIST, list(SECONDARY_CENTERS), SEC_K_LIST, args,
        secondary_min=-6.0, secondary_max=6.0,
    )
    assert len(patch_rows) == 1
    assert patch_rows[0]["primary_cv_center"] == pytest.approx(0.4)
    assert patch_rows[0]["priority"] >= 50.0, "off_center_window defects must not be starved out by the cap"


def test_dispatcher_backfills_secondary_samples_when_samples_csv_missing(tmp_path):
    """End-to-end: with no samples.csv on disk (current production format),
    passing fallback_secondary_history_by_window must make the dispatcher
    correctly diagnose the frontier group instead of reporting
    insufficient_samples for everything, and must not silently converge."""
    rng = np.random.default_rng(1)
    samples_d, samples_s = _synthetic_samples(rng)
    secondary_cv_metadata = {"enabled": True, "explicit_2d_windows": True, "range_min": -6.0, "range_max": 6.0}
    args = make_feedback_args(secondary_cv_centers=list(SECONDARY_CENTERS))

    assert not (tmp_path / "samples.csv").exists()
    summary = run_adaptive_feedback_dispatcher_2d_explicit_sparse(
        args, tmp_path, list(PRIMARY_CENTERS), K_LIST,
        {"accepted": 0, "attempts": 0, "pairs": {}},
        list(SECONDARY_CENTERS), SEC_K_LIST, secondary_cv_metadata,
        fallback_history_by_window=samples_d,
        fallback_secondary_history_by_window=samples_s,
    )
    assert summary is not None
    assert summary["converged"] is False
    assert summary["two_d_sparse_patch_proposal"]["patch_candidate_windows"] > 0

    import csv
    with (tmp_path / "adaptive_feedback_2d_cells.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert all(int(r["n_secondary_samples"]) > 0 for r in rows), (
        "secondary-axis fallback must populate real sample counts, not leave them at 0"
    )
    frontier_statuses = {r["cell_status"] for r in rows if float(r["center_A"]) > 0.5}
    assert frontier_statuses <= {"off_center", "low_2d_hit"}
    assert "insufficient_samples" not in frontier_statuses


def test_dispatcher_without_secondary_fallback_stays_insufficient(tmp_path):
    """Sanity control: omitting the new fallback reproduces the pre-fix bug
    (secondary samples empty, cells stuck at insufficient_samples) so this
    test would fail if the fallback wiring were ever accidentally removed."""
    rng = np.random.default_rng(1)
    samples_d, _ = _synthetic_samples(rng)
    secondary_cv_metadata = {"enabled": True, "explicit_2d_windows": True, "range_min": -6.0, "range_max": 6.0}
    args = make_feedback_args(secondary_cv_centers=list(SECONDARY_CENTERS))

    summary = run_adaptive_feedback_dispatcher_2d_explicit_sparse(
        args, tmp_path, list(PRIMARY_CENTERS), K_LIST,
        {"accepted": 0, "attempts": 0, "pairs": {}},
        list(SECONDARY_CENTERS), SEC_K_LIST, secondary_cv_metadata,
        fallback_history_by_window=samples_d,
        fallback_secondary_history_by_window=None,
    )
    import csv
    with (tmp_path / "adaptive_feedback_2d_cells.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert all(int(r["n_secondary_samples"]) == 0 for r in rows)
    assert all(r["cell_status"] == "insufficient_samples" for r in rows)
