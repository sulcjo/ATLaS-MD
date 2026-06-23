"""Oracle (grid answer) tests. No OpenMM."""
from __future__ import annotations

import numpy as np

from gareus.synth.landscapes import mixture_wells
from gareus.synth.oracle import (grid_overlap, ideal_axis_ladder, low_f_mask,
                                 reference_pmf)
from gareus.synth.sampler import Window


def test_overlap_in_unit_interval_and_monotone():
    lp = mixture_wells()
    a = Window(0.40, 60, 0.0, 60)
    near = Window(0.46, 60, 0.0, 60)
    far = Window(0.85, 60, 0.0, 60)
    o_near = grid_overlap(lp, a, near, beta=1.0, res=120)
    o_far = grid_overlap(lp, a, far, beta=1.0, res=120)
    assert 0.0 <= o_far <= o_near <= 1.0
    assert o_near > o_far


def test_low_f_mask_nonempty_and_subset():
    lp = mixture_wells()
    c1, c2, mask = low_f_mask(lp, res=100, threshold_kbt=5.0)
    assert mask.shape == (100, 100)
    assert mask.any() and not mask.all()


def test_ideal_ladder_neighbors_meet_target_overlap():
    lp = mixture_wells()
    centers = ideal_axis_ladder(lp, axis="cv1", target_overlap=0.30, k=50.0, res=120)
    assert len(centers) >= 2
    assert centers == sorted(centers)
    for a, b in zip(centers[:-1], centers[1:]):
        o = grid_overlap(lp, Window(a, 50.0), Window(b, 50.0), res=120)
        assert o >= 0.30 - 0.08


def test_reference_pmf_min_zero():
    lp = mixture_wells()
    x, pmf = reference_pmf(lp, axis="cv1", res=120)
    assert x.shape == pmf.shape
    assert np.isclose(pmf.min(), 0.0)
