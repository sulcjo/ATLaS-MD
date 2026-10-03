"""Frames of a resumed trajectory segment lie on the reporter's step grid.

OpenMM's trajectory reporters write a frame whenever the step counter is a
multiple of ``traj_interval``, and a resumed run continues the counter from
``prod_done``. A segment resumed at a step R that is NOT a multiple of
``traj_interval`` (a SIGTERM checkpoint) therefore holds frames at the grid
points after R, not at R + k * traj_interval. Mapping with
``round((s - R) / spf) - 1`` put every sample one frame early whenever
R mod spf > spf / 2 (chignolin_9: 1,652 of 8,968 merged segments; recomputed
CV1 of the mapped frame missed the sample's recorded CV1 until shifted +1).
"""
import numpy as np

from analyze_gareus_mbar import (_MERGED_TRAJ_STEP_STRIDE, _sample_to_segment_frame,
                                 _segment_grid_origin)

SPF = 250


def test_off_grid_resume_maps_samples_to_their_own_frames():
    # Resumed at 140400 (residue 150 > spf/2): frames at 140500, 140750, 141000, ...
    steps = np.array([140500, 140750, 141000, 141250], dtype=float)
    mask, local = _sample_to_segment_frame(steps, 140400, 4, SPF, grid_origin=0)
    assert mask.all()
    assert list(local) == [0, 1, 2, 3]


def test_off_grid_resume_with_small_residue_unchanged():
    # Resumed at 28800 (residue 50 < spf/2): frames at 29000, 29250, ...
    steps = np.array([29000, 29250, 29500], dtype=float)
    mask, local = _sample_to_segment_frame(steps, 28800, 3, SPF, grid_origin=0)
    assert list(local) == [0, 1, 2]


def test_window_ends_at_last_grid_frame():
    # 3 frames at 140500..141000; a sample at 141250 has no frame in this segment.
    steps = np.array([140500, 141000, 141250], dtype=float)
    mask, local = _sample_to_segment_frame(steps, 140400, 3, SPF, grid_origin=0)
    assert list(mask) == [True, True, False]
    assert list(local) == [0, 2]


def test_on_grid_resume_matches_legacy_mapping():
    steps = np.arange(5250, 7750, SPF, dtype=float)
    new = _sample_to_segment_frame(steps, 5000, 10, SPF, grid_origin=0)
    legacy = _sample_to_segment_frame(steps, 5000, 10, SPF)
    assert list(new[1]) == list(legacy[1]) == list(range(10))


def test_merged_segment_grid_is_anchored_at_its_phase_offset():
    src = 3
    resume = src * _MERGED_TRAJ_STEP_STRIDE + 140400
    origin = _segment_grid_origin(resume, resume)
    assert origin == src * _MERGED_TRAJ_STEP_STRIDE
    steps = src * _MERGED_TRAJ_STEP_STRIDE + np.array([140500, 140750], dtype=float)
    _, local = _sample_to_segment_frame(steps, resume, 2, SPF, grid_origin=origin)
    assert list(local) == [0, 1]


def test_gamd_anchor_origin_is_the_calibration_offset():
    # Non-merged GaMD: effective start = calib + resume_start; grid origin = calib.
    assert _segment_grid_origin(305000 + 50000, 50000) == 305000
    assert _segment_grid_origin(305000, 0) == 305000


def test_every_analyzer_call_site_passes_the_grid_origin():
    """Rg, PCA, extra observables and chignolin_fes all map frames through this
    function; one call left on the legacy default would re-open the off-by-one."""
    import inspect
    import re

    import analyze_gareus_mbar as A

    src = inspect.getsource(A)
    calls = re.findall(r"_sample_to_segment_frame\((.*?)\)\n", src)
    calls = [c for c in calls if "sample_steps" not in c]  # skip the definition
    assert len(calls) >= 4, calls
    assert all("grid_origin=" in c for c in calls), calls
