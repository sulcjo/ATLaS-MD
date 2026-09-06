"""Counting the frames mdtraj's QCP kernel refuses to rotate.

``Trajectory.superpose`` solves for the optimal rotation with a Newton
iteration on the largest eigenvalue. When that iteration does not converge the
C kernel prints

    theobald_rmsd.cpp UNCONVERGED ROTATION MATRIX. RETURNING IDENTITY

to stderr and returns the IDENTITY rotation, leaving those frames UNALIGNED
while every other frame is superposed. Nothing raises and nothing is returned
to say it happened.

Measured on a real 29M-sample chignolin run the rate was 2 frames against
100,000 fit frames and 15,930,096 projected frames -- about 1e-7, far too few
to move a PCA basis. So the defect being fixed is not the numerics but the
invisibility: a genuinely malformed trajectory produces the same two lines of
scrollback as the benign case, and no count reaches pmf_summary.json either
way.

These tests drive the counter directly with a fake stderr-writing superpose,
because the real event is far too rare to provoke on demand -- a scan of
20,500 real frames across all eight phases of that run produced zero.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import analyze_gareus_mbar as agm  # noqa: E402

MARKER = 'theobald_rmsd.cpp UNCONVERGED ROTATION MATRIX. RETURNING IDENTITY=300'


class _FakeTraj:
    """Stands in for an mdtraj Trajectory, writing to the REAL stderr fd.

    The kernel writes from C, so it bypasses ``sys.stderr`` entirely and goes
    straight to file descriptor 2. Writing via os.write here is what makes this
    a faithful stand-in: a test that used ``print(..., file=sys.stderr)`` would
    pass against an implementation that only captured Python-level stderr.
    """

    def __init__(self, n_frames, n_unconverged=0, other_stderr=''):
        self.n_frames = n_frames
        self._n_unconverged = n_unconverged
        self._other = other_stderr
        self.superposed = 0

    def superpose(self, ref, frame=0):
        self.superposed += 1
        payload = ''.join(MARKER + '\n' for _ in range(self._n_unconverged)) + self._other
        if payload:
            os.write(2, payload.encode())


def test_counts_unconverged_frames_and_totals():
    stats = {}
    agm._superpose_counting_unconverged(_FakeTraj(1000, n_unconverged=3), object(), stats)
    agm._superpose_counting_unconverged(_FakeTraj(500, n_unconverged=1), object(), stats)
    assert stats['unconverged'] == 4
    assert stats['frames'] == 1500


def test_a_clean_superpose_records_frames_and_no_failures():
    stats = {}
    t = _FakeTraj(250)
    agm._superpose_counting_unconverged(t, object(), stats)
    assert stats.get('unconverged', 0) == 0
    assert stats['frames'] == 250
    assert t.superposed == 1


def test_unrelated_stderr_is_passed_through_not_swallowed(capfd):
    """Counting our message must not cost us everybody else's.

    Suppressing all stderr for the duration of the call would hide any other
    diagnostic mdtraj or numpy emitted there -- trading a small blind spot for
    a larger one.
    """
    stats = {}
    agm._superpose_counting_unconverged(
        _FakeTraj(10, n_unconverged=2, other_stderr='some other warning\n'), object(), stats)
    err = capfd.readouterr().err
    assert 'some other warning' in err
    assert 'UNCONVERGED' not in err, 'the counted message should not also be re-printed'
    assert stats['unconverged'] == 2


def test_stats_is_optional_and_superpose_still_runs():
    """Callers that do not want the accounting must be unaffected."""
    t = _FakeTraj(10, n_unconverged=5)
    agm._superpose_counting_unconverged(t, object(), None)
    assert t.superposed == 1


def test_uncounted_calls_are_recorded_when_stderr_cannot_be_duplicated(monkeypatch):
    """A zero must never be reported when the answer is "not measured".

    The capture works by duplicating fd 2. If that fails -- stderr closed, or
    an fd table exhausted -- the superposition must still happen, and the
    result must say the count is a lower bound rather than silently reading as
    "no failures".
    """
    def _boom(fd):
        raise OSError('cannot duplicate')

    monkeypatch.setattr(os, 'dup', _boom)
    stats = {}
    t = _FakeTraj(42)
    agm._superpose_counting_unconverged(t, object(), stats)
    assert t.superposed == 1, 'the superposition must still run'
    assert stats['uncounted_calls'] == 1
    assert stats['frames'] == 42
    assert 'unconverged' not in stats, 'must not claim zero failures when unmeasured'


def test_capture_targets_fd_2_not_sys_stderr(monkeypatch):
    """The message comes from C, which writes to fd 2 and never consults
    sys.stderr. Replacing sys.stderr must not disable the counting."""
    class _Replaced:
        def fileno(self):
            raise OSError('no fileno')

        def write(self, s):
            return len(s)

    monkeypatch.setattr(sys, 'stderr', _Replaced())
    stats = {}
    agm._superpose_counting_unconverged(_FakeTraj(7, n_unconverged=2), object(), stats)
    assert stats['unconverged'] == 2, 'counting must not depend on sys.stderr'


def test_marker_matches_what_mdtraj_actually_prints():
    """Pin the substring against the real message, including its '=300' tail,
    so a partial match is not mistaken for a full one."""
    assert agm._SUPERPOSE_UNCONVERGED_MARKER in MARKER


def test_aligned_coords_threads_stats_through():
    """The helper the PCA passes actually call must feed the same counter."""
    stats = {}

    class _Chunk(_FakeTraj):
        def __init__(self):
            super().__init__(4, n_unconverged=1)
            self.xyz = np.zeros((4, 3, 3), dtype=np.float32)

    out = agm._aligned_flattened_coords_A(_Chunk(), np.arange(3), object(),
                                          pre_sliced=True, stats=stats)
    assert out.shape == (4, 9)
    assert stats['unconverged'] == 1
    assert stats['frames'] == 4
