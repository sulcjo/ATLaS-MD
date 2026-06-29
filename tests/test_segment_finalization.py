"""Unit tests for segment finalization logic (gareus.store.finalize_segment).

These tests verify that a segment is marked ``"complete"`` only after both the
production loop finishes cleanly AND all parquet writers flush/close without
error.  Any failure must leave the segment as ``"interrupted"``.

No OpenMM, PeptideBuilder, or gamd-openmm imports are used here.
"""
from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Helper: build a SegmentRegistry with one open segment ready to finalize.
# ---------------------------------------------------------------------------

def _open_registry(tmp_path):
    from gareus.store import SegmentRegistry

    reg = SegmentRegistry(tmp_path)
    seg_id = reg.open_segment(run_id="run_test", parent_id=None, round_id=1)
    return reg, seg_id


def _read_status(tmp_path, seg_id: str) -> dict:
    data = json.loads((tmp_path / "segments.json").read_text())
    for seg in data:
        if seg["segment_id"] == seg_id:
            return seg
    raise KeyError(seg_id)


# ---------------------------------------------------------------------------
# finalize_segment — success path
# ---------------------------------------------------------------------------

def test_finalize_segment_success_marks_complete(tmp_path):
    """Clean run + writers OK → status must be 'complete'."""
    from gareus.store import finalize_segment

    reg, seg_id = _open_registry(tmp_path)
    finalize_segment(reg, seg_id, completed_cleanly=True, writers_ok=True, end_step=500_000)

    seg = _read_status(tmp_path, seg_id)
    assert seg["status"] == "complete"
    assert seg["end_step"] == 500_000


# ---------------------------------------------------------------------------
# finalize_segment — interrupted paths
# ---------------------------------------------------------------------------

def test_finalize_segment_crash_marks_interrupted(tmp_path):
    """Production loop crashed (completed_cleanly=False) → 'interrupted'."""
    from gareus.store import finalize_segment

    reg, seg_id = _open_registry(tmp_path)
    finalize_segment(reg, seg_id, completed_cleanly=False, writers_ok=True, end_step=100_000)

    seg = _read_status(tmp_path, seg_id)
    assert seg["status"] == "interrupted"
    assert seg["end_step"] == 100_000


def test_finalize_segment_writer_failure_marks_interrupted(tmp_path):
    """Writer close raised (writers_ok=False) even with clean loop → 'interrupted'.

    This is the key regression: a truncated parquet file must NOT be marked
    complete even when the simulation loop itself finished without exception.
    """
    from gareus.store import finalize_segment

    reg, seg_id = _open_registry(tmp_path)
    finalize_segment(reg, seg_id, completed_cleanly=True, writers_ok=False, end_step=500_000)

    seg = _read_status(tmp_path, seg_id)
    assert seg["status"] == "interrupted"


def test_finalize_segment_both_failed_marks_interrupted(tmp_path):
    """Both crashed loop and writer failure → 'interrupted'."""
    from gareus.store import finalize_segment

    reg, seg_id = _open_registry(tmp_path)
    finalize_segment(reg, seg_id, completed_cleanly=False, writers_ok=False, end_step=0)

    seg = _read_status(tmp_path, seg_id)
    assert seg["status"] == "interrupted"


# ---------------------------------------------------------------------------
# seal_segment does not overwrite an already-complete segment
# ---------------------------------------------------------------------------

def test_seal_segment_does_not_overwrite_complete(tmp_path):
    """seal_segment called after close_segment must leave status='complete'."""
    from gareus.store import SegmentRegistry

    reg = SegmentRegistry(tmp_path)
    seg_id = reg.open_segment("run_test", None, 1)
    reg.close_segment(seg_id, end_step=200_000)
    # Simulate an erroneous second seal (e.g. from a retry path).
    reg.seal_segment(seg_id, absolute_end_step=100_000, status="interrupted")

    seg = _read_status(tmp_path, seg_id)
    assert seg["status"] == "complete", "seal_segment must not overwrite a complete segment"
    assert seg["end_step"] == 200_000, "end_step must not be overwritten"


# ---------------------------------------------------------------------------
# SegmentRegistry close_segment produces status="complete"
# ---------------------------------------------------------------------------

def test_close_segment_produces_complete_status(tmp_path):
    """close_segment → status 'complete' (direct registry method)."""
    from gareus.store import SegmentRegistry

    reg = SegmentRegistry(tmp_path)
    seg_id = reg.open_segment("run_test", None, 1)
    reg.close_segment(seg_id, end_step=300_000)

    seg = _read_status(tmp_path, seg_id)
    assert seg["status"] == "complete"
    assert seg["end_step"] == 300_000


# ---------------------------------------------------------------------------
# SegmentRegistry seal_segment produces status="interrupted"
# ---------------------------------------------------------------------------

def test_seal_segment_produces_interrupted_status(tmp_path):
    """seal_segment → status 'interrupted' (direct registry method)."""
    from gareus.store import SegmentRegistry

    reg = SegmentRegistry(tmp_path)
    seg_id = reg.open_segment("run_test", None, 1)
    reg.seal_segment(seg_id, absolute_end_step=50_000, status="interrupted")

    seg = _read_status(tmp_path, seg_id)
    assert seg["status"] == "interrupted"
    assert seg["end_step"] == 50_000


# ---------------------------------------------------------------------------
# finalize_segment end_step is recorded correctly in both outcomes
# ---------------------------------------------------------------------------

def test_finalize_segment_end_step_recorded_on_interrupted(tmp_path):
    """end_step must be persisted even for interrupted segments (resume needs it)."""
    from gareus.store import finalize_segment

    reg, seg_id = _open_registry(tmp_path)
    finalize_segment(reg, seg_id, completed_cleanly=False, writers_ok=True, end_step=77_777)

    seg = _read_status(tmp_path, seg_id)
    assert seg["end_step"] == 77_777


def test_finalize_segment_persists_across_instances(tmp_path):
    """Finalized segment status survives a new SegmentRegistry instance (disk persistence)."""
    from gareus.store import SegmentRegistry, finalize_segment

    reg1 = SegmentRegistry(tmp_path)
    seg_id = reg1.open_segment("run_test", None, 1)
    finalize_segment(reg1, seg_id, completed_cleanly=True, writers_ok=True, end_step=400_000)

    reg2 = SegmentRegistry(tmp_path)
    seg = reg2.get_segment(seg_id)
    assert seg is not None
    assert seg["status"] == "complete"
    assert seg["end_step"] == 400_000
