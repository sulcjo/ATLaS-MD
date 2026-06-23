"""Feedback campaign integration tests (drive the REAL 2D dispatcher)."""
from __future__ import annotations

import pytest

pytest.importorskip("scipy")

from gareus.synth.drivers import feedback_campaign
from gareus.synth.landscapes import mixture_wells
import numpy as np


def test_campaign_runs_and_returns_records():
    lp = mixture_wells()
    recs = feedback_campaign(
        lp, initial_centers1=[0.2, 0.5, 0.8], initial_k1=[40.0, 40.0, 40.0],
        initial_centers2=[-0.5, 0.5], initial_k2=[40.0, 40.0],
        mode="exact", n_rounds=3, samples_per_window=1200, res=100, seed=0)
    assert len(recs) >= 1
    assert recs[-1].n_windows >= 2
    assert np.isfinite(recs[-1].overlap_mean)


def _distinct(centers):
    return len(set(round(float(c), 3) for c in centers))


def test_overdense_windows_get_pruned():
    # Five tightly-packed, stiff windows over-resolve a single region: the
    # dispatcher prunes the redundant ones (distinct-count drop is the
    # behavior-level action discriminator; 1D/2D proposals expose different
    # bookkeeping keys so we assert on the observable layout, not the keys).
    lp = mixture_wells()
    recs = feedback_campaign(
        lp, initial_centers1=[0.50, 0.52, 0.54, 0.56, 0.58],
        initial_k1=[120.0] * 5, initial_centers2=[0.0], initial_k2=[60.0],
        mode="exact", n_rounds=2, samples_per_window=1500, res=120, seed=1,
        aggressiveness="aggressive")
    assert _distinct(recs[-1].centers1) <= 3       # 5 -> <=3


def test_coverage_gap_gets_a_new_window():
    # Two far-apart windows with a low-F bridge between: the dispatcher inserts
    # midpoint window(s) to close the gap.
    lp = mixture_wells()
    recs = feedback_campaign(
        lp, initial_centers1=[0.18, 0.84], initial_k1=[50.0, 50.0],
        initial_centers2=[0.0], initial_k2=[60.0],
        mode="exact", n_rounds=2, samples_per_window=1500, res=120, seed=2,
        aggressiveness="aggressive")
    assert _distinct(recs[-1].centers1) >= 4       # 2 -> >=4


def test_converged_layout_is_a_fixpoint():
    # Mutation/negative guard: run to the dispatcher's OWN converged layout, then
    # re-seed that layout. A correct policy leaves it ~unchanged; a regression
    # that always adds/removes/shifts (e.g. a sign flip) would churn it.
    lp = mixture_wells()
    warm = feedback_campaign(
        lp, initial_centers1=[0.2, 0.5, 0.8], initial_k1=[40.0] * 3,
        initial_centers2=[0.0], initial_k2=[60.0], mode="exact", n_rounds=5,
        samples_per_window=1500, res=120, seed=4, aggressiveness="balanced")
    layout = warm[-1]
    n_conv = _distinct(layout.centers1)
    rerun = feedback_campaign(
        lp, initial_centers1=list(layout.centers1), initial_k1=list(layout.k1),
        initial_centers2=list(layout.centers2), initial_k2=list(layout.k2),
        mode="exact", n_rounds=2, samples_per_window=1500, res=120, seed=4,
        aggressiveness="balanced")
    assert abs(_distinct(rerun[-1].centers1) - n_conv) <= 1   # stable fixpoint


def test_campaign_is_deterministic_under_seed():
    lp = mixture_wells()
    kw = dict(initial_centers1=[0.2, 0.5, 0.8], initial_k1=[40.0] * 3,
              initial_centers2=[0.0], initial_k2=[60.0], mode="exact",
              n_rounds=2, samples_per_window=1000, res=90, seed=5)
    a = feedback_campaign(lp, **kw)
    b = feedback_campaign(lp, **kw)
    assert [r.centers1 for r in a] == [r.centers1 for r in b]
