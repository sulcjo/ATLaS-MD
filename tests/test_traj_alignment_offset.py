import numpy as np

from analyze_gareus_mbar import _base_segment_resume_start, _sample_to_segment_frame


def test_gamd_offset_inferred_from_absolute_sample_steps():
    # GaMD counts equilibration in the absolute sample step (~305000) while the
    # production trajectory starts at frame 0; base segment must absorb the offset.
    steps = np.arange(306000, 326000, 1000, dtype=float)  # 20 production samples
    rs = _base_segment_resume_start(0, False, steps, 1000)
    assert rs == 305000


def test_cmd_run_left_unchanged():
    # cmd samples are production-relative (start near 0); inferred offset is
    # negative and must NOT be applied.
    steps = np.arange(1000, 21000, 1000, dtype=float)
    assert _base_segment_resume_start(0, False, steps, 5000) == 0


def test_resume_segment_anchored_to_absolute_clock():
    # Resume filenames encode production-relative prod_done; sample steps are
    # absolute (calib + prod_done). The anchor must shift the resume offset into
    # absolute coordinates: calib (305000) + resume_start (50000) = 355000.
    steps = np.arange(306000, 326000, 1000, dtype=float)
    assert _base_segment_resume_start(50000, False, steps, 1000) == 355000


def test_cmd_resume_segment_unchanged():
    # cmd run: calib=0 (steps production-relative), so a resume segment keeps its
    # production-relative filename offset.
    steps = np.arange(1000, 21000, 1000, dtype=float)  # min == spf -> calib 0
    assert _base_segment_resume_start(7000, False, steps, 1000) == 7000


def test_merged_adaptive_path_not_overridden():
    steps = np.arange(306000, 326000, 1000, dtype=float)
    assert _base_segment_resume_start(0, True, steps, 1000) == 0


def test_empty_steps_unchanged():
    assert _base_segment_resume_start(0, False, np.array([]), 1000) == 0


def test_alignment_round_trips_with_inferred_offset():
    # End-to-end: with the inferred offset, 1:1 cadence samples map onto frames.
    spf = 1000
    steps = np.arange(306000, 326000, spf, dtype=float)  # 20 samples
    n_frames = 20
    rs = _base_segment_resume_start(0, False, steps, spf)
    mask, local = _sample_to_segment_frame(steps, rs, n_frames, spf)
    assert mask.sum() == 20
    assert list(local) == list(range(20))
