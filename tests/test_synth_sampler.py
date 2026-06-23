"""Exact biased sampler tests for the synthetic harness. No OpenMM."""
from __future__ import annotations

import numpy as np

from gareus.synth.landscapes import mixture_wells
from gareus.synth.sampler import Window, sample_window_exact


def test_exact_sampler_shape_and_domain():
    lp = mixture_wells()
    rng = np.random.default_rng(0)
    w = Window(center1=0.5, k1=40.0, center2=0.0, k2=40.0)
    s = sample_window_exact(lp, w, 2000, beta=1.0, res=120, rng=rng)
    assert s.shape == (2000, 2)
    assert s[:, 0].min() >= 0.0 and s[:, 0].max() <= 1.0
    assert s[:, 1].min() >= -1.0 and s[:, 1].max() <= 1.0


def test_exact_sampler_concentrates_near_strong_restraint_center():
    lp = mixture_wells()
    rng = np.random.default_rng(1)
    w = Window(center1=0.3, k1=500.0, center2=0.4, k2=500.0)
    s = sample_window_exact(lp, w, 4000, beta=1.0, res=160, rng=rng)
    assert abs(s[:, 0].mean() - 0.3) < 0.05
    assert abs(s[:, 1].mean() - 0.4) < 0.05


def test_exact_sampler_unbiased_follows_F_basin():
    lp = mixture_wells()
    rng = np.random.default_rng(2)
    w = Window(center1=0.5, k1=0.0, center2=0.0, k2=0.0)
    s = sample_window_exact(lp, w, 6000, beta=1.0, res=160, rng=rng)
    # global min basin near (0.18, -0.65)
    assert abs(np.median(s[:, 0]) - 0.18) < 0.12
    assert abs(np.median(s[:, 1]) - (-0.65)) < 0.20


def test_exact_sampler_is_deterministic_under_seed():
    lp = mixture_wells()
    w = Window(0.4, 60.0, 0.1, 60.0)
    a = sample_window_exact(lp, w, 500, rng=np.random.default_rng(7))
    b = sample_window_exact(lp, w, 500, rng=np.random.default_rng(7))
    assert np.allclose(a, b)
