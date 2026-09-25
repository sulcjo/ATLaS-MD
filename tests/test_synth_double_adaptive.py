"""Doubly-adaptive workflow integration tests (feedback -> production epochs)."""
from __future__ import annotations

import math

import pytest

pytest.importorskip("scipy")

from gareus.adaptive_production import AdaptiveDecisionPolicy
from gareus.synth.drivers import double_adaptive_campaign
from gareus.synth.landscapes import mixture_wells

# a frozen handoff layout so the production policy is what varies
LAYOUT = {"centers1": [0.10, 0.22, 0.34, 0.46, 0.58, 0.70, 0.82, 0.94],
          "k1": [60.0] * 8, "centers2": [-0.5, 0.5], "k2": [50.0, 50.0]}


def test_double_adaptive_runs_end_to_end_from_layout():
    lp = mixture_wells()
    out = double_adaptive_campaign(lp, layout=LAYOUT, mode="exact", n_epochs=4,
                                  epoch_raw_budget=800, res=80, seed=0)
    assert out["epochs"]                      # produced epoch records
    assert out["total_raw_budget"] > 0
    assert out["final_n_active"] >= 8         # min_active_states floor (16 seeded -> >=8)
    assert math.isfinite(out["final_pmf_rmse_lowf"])


def test_double_adaptive_runs_feedback_then_production():
    # No frozen layout: run the full pipeline (feedback hands off to production).
    lp = mixture_wells()
    out = double_adaptive_campaign(
        lp, feedback_cfg=dict(initial_centers1=[0.2, 0.5, 0.8], initial_k1=[50.0] * 3,
                              initial_centers2=[-0.5, 0.5], initial_k2=[50.0, 50.0],
                              n_rounds=3, samples_per_window=800),
        mode="exact", n_epochs=3, epoch_raw_budget=700, res=80, seed=1)
    assert out["layout"]["centers1"]          # a layout was produced and handed off
    assert out["epochs"]
    assert out["all_connected"] in (True, False)


def test_double_adaptive_hard_gate_connectivity_holds():
    lp = mixture_wells()
    out = double_adaptive_campaign(lp, layout=LAYOUT, mode="exact", n_epochs=4,
                                  epoch_raw_budget=800, res=80, seed=2)
    # the real decision refuses disconnecting retirements, so the graph stays connected
    assert out["all_connected"] is True


def test_topup_heavy_vs_eager_add_differ_under_ess():
    # The ESS cost model must let epoch strategies separate (eager-add pays burn-in;
    # top-up-heavy concentrates budget). Their PMF-at-fixed-budget should differ.
    lp = mixture_wells()
    # budget must exceed the per-state floor (n_active*min_state_steps) so the
    # scheduler's score-based allocation (where the policies differ) engages.
    eager = AdaptiveDecisionPolicy(max_new_windows_per_epoch=6, min_samples_for_add=30,
                                   weak_edge_bonus=6.0, low_sample_bonus=0.5,
                                   target_overlap=0.45, min_state_steps=40, max_state_steps=2000)
    topup = AdaptiveDecisionPolicy(max_new_windows_per_epoch=1, low_sample_bonus=6.0,
                                   weak_edge_bonus=0.5, min_samples_for_retire=300,
                                   redundant_overlap=0.60, min_state_steps=40, max_state_steps=2000)
    a = double_adaptive_campaign(lp, layout=LAYOUT, policy=eager, mode="exact",
                                n_epochs=4, epoch_raw_budget=4000, res=80, seed=0)
    b = double_adaptive_campaign(lp, layout=LAYOUT, policy=topup, mode="exact",
                                n_epochs=4, epoch_raw_budget=4000, res=80, seed=0)
    # different policies -> different evolution (window count and/or PMF)
    assert (a["final_n_active"] != b["final_n_active"]) or (
        abs(a["final_pmf_rmse_lowf"] - b["final_pmf_rmse_lowf"]) > 1e-6)


def test_retire_fires_with_raw_sample_gating():
    # Regression for the ESS-vs-raw gating bug: the production retire gate keys off
    # min_samples_for_retire (200). When the diagnostics reported ESS-thinned counts
    # (which stayed <200), retire could NEVER fire. With RAW accumulated counts it
    # does, on a redundant layout at adequate budget -- while respecting the floor.
    lp = mixture_wells()
    redundant = {"centers1": [0.10, 0.18, 0.46, 0.50, 0.54, 0.58, 0.74, 0.90],
                 "k1": [60.0] * 8, "centers2": [-0.5, 0.5], "k2": [50.0, 50.0]}
    pol = AdaptiveDecisionPolicy(redundant_overlap=0.40, min_samples_for_retire=200,
                                 retire_converged=True, min_state_steps=40, max_state_steps=4000)
    # Budget doubled when the score allocator was removed (effective top-ups, task 8):
    # the old score weights (new states got max_state_steps//2 = 5x the default)
    # concentrated enough MD for 6000/40000 to reach the retire gate.  Under the
    # uniform schedule 6000/40000 retires nothing in 5 epochs; 12000/80000 retires
    # in epoch 4 (measured).
    out = double_adaptive_campaign(lp, layout=redundant, policy=pol, mode="exact",
                                  n_epochs=5, epoch_raw_budget=12000, total_budget=80000,
                                  res=80, seed=0, default_steps=400)
    assert sum(len(e["retired"]) for e in out["epochs"]) >= 1     # retire now fires
    assert all(e["n_active"] >= pol.min_active_states for e in out["epochs"])  # floor held
