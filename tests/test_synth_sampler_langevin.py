"""Adversarial Langevin sampler tests for the synthetic harness."""
from __future__ import annotations

import numpy as np

from gareus.synth.landscapes import mixture_wells, slow_cv2_double_branch
from gareus.synth.sampler import Window
from gareus.synth.sampler_langevin import sample_window_langevin


def test_langevin_shape_and_domain():
    lp = mixture_wells()
    rng = np.random.default_rng(0)
    w = Window(0.5, 60.0, 0.0, 60.0)
    s = sample_window_langevin(lp, w, 1000, beta=1.0, rng=rng)
    assert s.shape == (1000, 2)
    assert s[:, 0].min() >= 0.0 and s[:, 0].max() <= 1.0
    assert s[:, 1].min() >= -1.0 and s[:, 1].max() <= 1.0


def test_langevin_stiff_restraint_concentrates_near_center():
    lp = mixture_wells()
    rng = np.random.default_rng(1)
    w = Window(0.3, 800.0, 0.4, 800.0)
    s = sample_window_langevin(lp, w, 2000, beta=1.0, rng=rng,
                              D1=1.0, D2=1.0, dt=1e-3, steps_per_sample=10)
    assert abs(s[:, 0].mean() - 0.3) < 0.08
    assert abs(s[:, 1].mean() - 0.4) < 0.08


def test_langevin_is_deterministic_under_seed():
    lp = mixture_wells()
    w = Window(0.4, 60.0, 0.1, 60.0)
    a = sample_window_langevin(lp, w, 500, beta=1.0, rng=np.random.default_rng(7))
    b = sample_window_langevin(lp, w, 500, beta=1.0, rng=np.random.default_rng(7))
    assert np.allclose(a, b)


def test_langevin_slow_cv2_is_non_ergodic_short_run():
    # No CV2 bias (k2=0) -> the exact equilibrium is BIMODAL (both ~6 kBT-deep
    # branches equally populated). A short Langevin run started in the lower
    # branch with slow CV2 diffusion cannot cross the saddle, so it stays
    # trapped -- the non-ergodicity the exact sampler never shows.
    lp = slow_cv2_double_branch()
    rng = np.random.default_rng(2)
    w = Window(0.5, 5.0, 0.0, 0.0)   # zero CV2 restraint
    s = sample_window_langevin(lp, w, 1500, beta=1.0, rng=rng,
                              D1=1.0, D2=0.02, dt=2e-3, steps_per_sample=10,
                              burn_in=200, x0=(0.5, -0.6))
    frac_lower = float(np.mean(s[:, 1] < 0.0))
    assert frac_lower > 0.85   # trapped in the starting branch

    # Contrast: the exact sampler populates BOTH branches for the same window.
    from gareus.synth.sampler import sample_window_exact
    se = sample_window_exact(lp, w, 4000, beta=1.0, res=160,
                             rng=np.random.default_rng(2))
    frac_lower_exact = float(np.mean(se[:, 1] < 0.0))
    assert 0.3 < frac_lower_exact < 0.7   # both branches sampled
