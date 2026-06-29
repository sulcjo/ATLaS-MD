"""ESS / burn-in cost-model tests for the doubly-adaptive study."""
from __future__ import annotations

import numpy as np

from gareus.synth.ess import effective_count, tau_int, thin_to_ess
from gareus.synth.landscapes import mixture_wells, gated_barrier
from gareus.synth.sampler import Window


def test_tau_increases_with_tight_k():
    lp = mixture_wells()
    loose = tau_int(lp, Window(0.5, 25.0, 0.0, 25.0))
    tight = tau_int(lp, Window(0.5, 200.0, 0.0, 200.0))
    assert tight > loose >= 1.0


def test_tau_higher_on_a_barrier():
    # gated-barrier has a steep ridge near cv1=0.5; tau there should exceed a flat basin.
    lp = gated_barrier()
    on_ridge = tau_int(lp, Window(0.5, 50.0, 0.20, 50.0))
    in_basin = tau_int(lp, Window(0.18, 50.0, -0.65, 50.0))
    assert on_ridge > in_basin


def test_effective_count_burn_in_and_decorrelation():
    # no burn-in for an existing window; burn-in subtracted for a new one
    assert effective_count(1000, 1.0, burn_in=120, is_new=False) == 1000 / 3.0
    assert effective_count(1000, 1.0, burn_in=120, is_new=True) == (1000 - 120) / 3.0
    # higher tau -> fewer effective
    assert effective_count(1000, 4.0) < effective_count(1000, 1.0)
    # cannot go negative
    assert effective_count(50, 1.0, burn_in=120, is_new=True) == 0.0


def test_thin_to_ess_shapes():
    s = np.random.default_rng(0).normal(size=(1000, 2))
    assert thin_to_ess(s, 100).shape == (100, 2)
    assert thin_to_ess(s, 5000).shape == (1000, 2)   # capped at len
    assert thin_to_ess(s, 0).shape == (0, 2)
