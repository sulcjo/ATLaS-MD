"""Landscape invariants for the synthetic adaptive-window harness. No OpenMM."""
from __future__ import annotations

import numpy as np

from gareus.synth.landscapes import LANDSCAPES, mixture_wells


def test_mixture_wells_registered_and_min_zero():
    lp = LANDSCAPES["mixture-wells"]
    assert lp is not None
    c1, c2, f = lp.grid(res=80)
    assert f.shape == (80, 80)
    assert np.isclose(f.min(), 0.0)
    assert np.all(np.isfinite(f))
    assert f.max() > 3.0


def test_mixture_wells_basins_are_low_f():
    lp = mixture_wells()
    assert len(lp.basins) >= 2
    c1, c2, f = lp.grid(res=80)
    for (b1, b2) in lp.basins:
        fb = float(lp.energy(np.array([b1]), np.array([b2]))[0]) - float(lp.energy(*np.meshgrid(c1, c2, indexing="ij")).min())
        assert fb < float(f.mean())


def test_all_landscapes_min_zero_and_finite():
    for name, lp in LANDSCAPES.items():
        c1, c2, f = lp.grid(res=60)
        assert np.isclose(f.min(), 0.0), name
        assert np.all(np.isfinite(f)), name


def test_gated_barrier_has_high_ridge_with_low_pass():
    lp = LANDSCAPES["gated-barrier"]
    c1, c2, f = lp.grid(res=120)
    # column nearest cv1=0.5 is the ridge; it must be high somewhere but have a low pass
    j = int(np.argmin(np.abs(c1 - 0.5)))
    ridge_col = f[j, :]
    assert ridge_col.max() > 4.0          # a real barrier
    assert ridge_col.min() < ridge_col.max() - 2.0   # a pass exists


def test_chaos_2d_properties():
    lp = LANDSCAPES["chaos-2d"]
    c1, c2, f = lp.grid(res=80)
    assert np.isclose(f.min(), 0.0)
    assert np.all(np.isfinite(f))
    assert f.max() > 20.0, "chaos_2d must have barriers >20 kBT"
    assert len(lp.basins) == 24, f"expected 24 sub-basins, got {len(lp.basins)}"
    # barriers exist along both CV axes
    mid1 = int(np.argmin(np.abs(c1 - 0.50)))
    mid2 = int(np.argmin(np.abs(c2 - 0.00)))
    assert f[mid1, :].max() > 15.0, "main CV1 barrier must be >15 kBT"
    assert f[:, mid2].max() > 8.0,  "CV2 barrier at 0 must be >8 kBT"


def test_double_branch_has_two_cv2_minima():
    lp = LANDSCAPES["slow-cv2-double-branch"]
    c1, c2, f = lp.grid(res=120)
    # marginal over cv1 at the central column: two wells in cv2 separated by a saddle
    i = int(np.argmin(np.abs(c1 - 0.5)))
    col = f[i, :]
    mid = int(np.argmin(np.abs(c2 - 0.0)))
    assert col[mid] > col[:mid].min() + 1.0
    assert col[mid] > col[mid:].min() + 1.0
