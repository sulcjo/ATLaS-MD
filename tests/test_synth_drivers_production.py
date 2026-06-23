"""Production campaign integration tests.

These drive the REAL decision pipeline (propose_actions_from_diagnostics +
_apply_registry_actions): the add/retire/extend decisions, graph
articulation/connectivity checks, and the min_active_states floor are shipped
code. The harness only assembles synthetic samples + diagnostics.
"""
from __future__ import annotations

import numpy as np

from gareus.adaptive_production import AdaptiveDecisionPolicy
from gareus.synth.drivers import production_campaign
from gareus.synth.landscapes import mixture_wells


def test_production_retires_down_to_floor_then_stops():
    # 11-state over-dense ladder (incl. a near-duplicate). The real decision
    # retires redundant non-critical states down to the min_active_states floor
    # (8) and then refuses further retirement. Deterministic under the exact
    # sampler (geometry-driven, seed-independent).
    lp = mixture_wells()
    recs = production_campaign(lp, mode="exact", n_epochs=3,
                              samples_per_window=1500, res=100, seed=1)
    assert recs[-1].n_active == 8                      # floor hit exactly
    total_retired = sum(len(r.retired_this_epoch) for r in recs)
    assert total_retired == 3                          # 11 -> 8
    assert recs[-1].retired_this_epoch == []           # no retire once at floor


def test_production_never_drops_below_floor_for_any_seed():
    lp = mixture_wells()
    for seed in (0, 1, 2, 3):
        recs = production_campaign(lp, mode="exact", n_epochs=4,
                                  samples_per_window=1200, res=90, seed=seed)
        assert all(r.n_active >= 8 for r in recs)
        assert recs[-1].n_active == 8


def test_production_no_retire_when_already_at_floor():
    # Exactly 8 well-separated states: nothing redundant, already at the floor,
    # so the real decision must NOT retire anything (mutation guard: a policy
    # that retired regardless would fail here).
    lp = mixture_wells()
    states = [(float(c), 50.0) for c in np.linspace(0.06, 0.94, 8)]
    recs = production_campaign(lp, initial_states=states, mode="exact",
                              n_epochs=3, samples_per_window=1500, res=100, seed=0)
    assert all(len(r.retired_this_epoch) == 0 for r in recs)
    assert recs[-1].n_active == 8


def test_production_below_floor_start_is_not_retired():
    # Start below the floor (6 states): no retirement is permissible at all.
    lp = mixture_wells()
    states = [(float(c), 50.0) for c in np.linspace(0.1, 0.9, 6)]
    recs = production_campaign(lp, initial_states=states, mode="exact",
                              n_epochs=2, samples_per_window=1500, res=90, seed=0)
    assert sum(len(r.retired_this_epoch) for r in recs) == 0


def test_production_tops_up_undersampled_states():
    # Under-sampled states (cumulative < min_samples_for_retire=200) must trigger
    # the real 'extend' (top-up) action, which then stops once they accrue enough.
    # 8 dense states (high overlap -> no weak edge -> no add; under floor -> no
    # retire) isolate the extend behavior.
    lp = mixture_wells()
    states = [(float(c), 60.0) for c in np.linspace(0.30, 0.58, 8)]
    recs = production_campaign(lp, initial_states=states, mode="exact", n_epochs=3,
                              samples_per_window=1200, res=90, seed=0,
                              sample_count_per_epoch=150, initial_cumulative=0)
    assert recs[0].extended_this_epoch                      # epoch 1: 150 < 200 -> top-up
    assert recs[-1].extended_this_epoch == []               # accrued >=200 -> top-up satisfied
    assert sum(len(r.retired_this_epoch) for r in recs) == 0


def test_production_adds_at_a_real_gap():
    # A ladder with a genuine low-overlap gap must trigger the real 'add' action
    # (weak-edge midpoint insertion) on the production side.
    lp = mixture_wells()
    states = [(float(c), 80.0) for c in [0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95]]
    recs = production_campaign(lp, initial_states=states, mode="exact", n_epochs=2,
                              samples_per_window=1500, res=100, seed=0)
    assert any(r.added_this_epoch for r in recs)            # midpoint(s) inserted at the gap
    assert recs[-1].n_active > 8                            # net growth from the add
