"""Spec P2: the split/insert applier validates a whole action, then applies it atomically.

Every mutating action is validated completely first (whole-centre resolution, anchor/axis/
mandatory refusal, restraint-aware duplicate check against the registry and against the
action's own earlier children, the cv2_k_max clamp, the 3.4 coupling gate, the 3.5 budget for
all of the action's states); only then is it applied. A refused action changes nothing and is
recorded in ``refused_actions`` with a reason code and its index. No apply ever changes an
existing state_id's Hamiltonian.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gareus.adaptive_production as ap  # noqa: E402
from gareus.adaptive_production import (AdaptiveDecisionPolicy, AdaptiveProductionController,  # noqa: E402
                                        WindowStateRegistry)

LADDER = (0.0, 0.2, 0.5, 1.0)


def _grid(lambdas=(0.0,), c1s=(0.1, 0.2, 0.3), c2s=(-1.0, 0.0, 1.0), k2=5.0):
    reg = WindowStateRegistry()
    for c1 in c1s:
        for c2 in c2s:
            for lam in lambdas:
                reg.add_state(c1, 800.0, secondary_center=c2, secondary_k=k2, gamd_lambda=lam,
                              epoch=0, source="seed")
    return reg


def _centre(reg, c1, c2, active=None):
    return [s for s in reg.all_states()
            if abs(s.primary_center - c1) < 1e-9 and s.secondary_center is not None
            and abs(s.secondary_center - c2) < 1e-9 and (active is None or s.active == active)]


def _apply(reg, actions, policy=None, **kw):
    ctl = AdaptiveProductionController(reg, policy=policy, **kw)
    ctl.apply_actions(0, actions)
    return ctl


def _snapshot(reg):
    return json.dumps(reg.to_dict(), sort_keys=True)


def _parent_id(reg, c1, c2, lam=0.0):
    return next(s.state_id for s in _centre(reg, c1, c2) if abs(s.gamd_lambda - lam) < 1e-9)


# ---- split: whole centre, every rung, atomic ---------------------------------------------------

def test_split_adds_children_on_every_rung_and_retires_every_rung_of_the_parent_centre():
    reg = _grid(LADDER)
    parent = _parent_id(reg, 0.2, 0.0, lam=0.5)          # named by a non-zero rung member
    n0 = len(reg.active_states())
    ctl = _apply(reg, [("split", parent, [(0.2, 800.0, -0.4, 8.0), (0.2, 800.0, 0.4, 8.0)], "bimodal")])
    assert ctl.refused_actions == []
    retired = _centre(reg, 0.2, 0.0)
    assert len(retired) == 4 and all(not s.active for s in retired)
    for c2 in (-0.4, 0.4):
        kids = _centre(reg, 0.2, c2, active=True)
        assert sorted(s.gamd_lambda for s in kids) == list(LADDER)
        assert all(s.parent_state_id == parent and s.source == "adaptive_production_split" for s in kids)
    assert len(reg.active_states()) == n0 - 4 + 8


def test_retired_parent_stays_in_the_registry_usable_for_mbar_with_its_own_hamiltonian():
    reg = _grid(LADDER)
    parent = _parent_id(reg, 0.2, 0.0)
    before = {s.state_id: (s.primary_center, s.primary_k, s.secondary_center, s.secondary_k, s.gamd_lambda)
              for s in _centre(reg, 0.2, 0.0)}
    _apply(reg, [("split", parent, [(0.2, 800.0, -0.4, 8.0), (0.2, 800.0, 0.4, 8.0)], "bimodal")])
    for sid, ham in before.items():
        st = reg.get_state(sid)
        assert st is not None and not st.active and st.usable_for_mbar and st.retired_epoch == 1
        assert (st.primary_center, st.primary_k, st.secondary_center, st.secondary_k, st.gamd_lambda) == ham
    assert set(before) <= {s.state_id for s in reg.all_states() if s.usable_for_mbar}


def test_retired_split_parent_is_a_union_mbar_column_biased_with_its_native_epoch_params(tmp_path):
    from test_adaptive_segmented_diagnostics import _write_parquet_epoch_run
    from test_union_state_mbar_native_params import (NATIVE_PRIMARY, NATIVE_SECONDARY, _registry,
                                                     _self_bias_kcal, _write_map)
    registry = _registry(tmp_path, NATIVE_PRIMARY, NATIVE_SECONDARY)
    adaptive = tmp_path / "adaptive"
    epoch = adaptive / "epoch_000"
    _write_parquet_epoch_run(epoch, n_windows=3, rows_per_window=40)
    _write_map(epoch, [{"epoch_window": w, "state_id": w, "primary_center": NATIVE_PRIMARY[w],
                        "secondary_center": NATIVE_SECONDARY[w]} for w in range(3)])
    ctl = _apply(registry, [("split", 1, [(0.5, 10.0, -0.25, 5.0), (0.5, 10.0, 0.25, 5.0)], "modes")])
    assert ctl.refused_actions == [] and not registry.get_state(1).active

    meta = ap.build_union_state_mbar_inputs(adaptive, registry)
    with np.load(meta["arrays_npz"], allow_pickle=False) as d:
        cols = [int(x) for x in d["state_ids"]]
    assert 1 in cols and len(cols) == 5                   # retired parent + both children
    own = _self_bias_kcal(meta, 1)
    assert own.size > 0 and float(np.median(own)) < 0.5  # its own samples, its own (native) restraint


@pytest.mark.parametrize("flag_rung", [0.0, 1.0])
def test_split_is_refused_if_any_member_of_the_centre_is_mandatory(flag_rung):
    reg = _grid(LADDER)
    parent = _parent_id(reg, 0.2, 0.0)
    reg.get_state(_parent_id(reg, 0.2, 0.0, lam=flag_rung)).metadata["mandatory"] = True
    snap = _snapshot(reg)
    ctl = _apply(reg, [("split", parent, [(0.2, 800.0, -0.4, 8.0)], "x")])
    assert _snapshot(reg) == snap
    assert [(r["action"], r["reason"], r["index"]) for r in ctl.refused_actions] == [("split", "mandatory", 0)]


@pytest.mark.parametrize("k1,k2", [(0.0, 5.0), (800.0, 0.0), (0.0, 0.0)])
def test_split_is_refused_on_axis_and_anchor_states_structurally(k1, k2):
    reg = _grid()
    odd = reg.add_state(0.5, k1, secondary_center=0.0, secondary_k=k2, epoch=0, source="seed")
    snap = _snapshot(reg)
    ctl = _apply(reg, [("add", 0, (0.15, 800.0, -1.0, 5.0), "fine"),
                       ("split", odd.state_id, [(0.5, 800.0, -0.4, 8.0), (0.5, 800.0, 0.4, 8.0)], "x")])
    assert [(r["reason"], r["index"]) for r in ctl.refused_actions] == [("anchor_or_axis", 1)]
    assert not any(abs(s.primary_center - 0.5) < 1e-9 and s.secondary_center in (-0.4, 0.4)
                   for s in reg.all_states())
    assert reg.get_state(odd.state_id).active and _snapshot(reg) != snap   # the valid add still ran


def test_a_cv1_only_state_in_a_two_d_campaign_is_an_axis_state():
    reg = _grid()
    cv1_only = reg.add_state(0.5, 800.0, epoch=0, source="seed")           # no CV2 centre at all
    ctl = _apply(reg, [("split", cv1_only.state_id, [(0.45, 800.0), (0.55, 800.0)], "x")])
    assert [r["reason"] for r in ctl.refused_actions] == ["anchor_or_axis"]


def test_a_split_in_a_cv1_only_campaign_is_allowed():
    reg = WindowStateRegistry()
    for c in (0.1, 0.2, 0.3):
        reg.add_state(c, 800.0, epoch=0, source="seed")
    ctl = _apply(reg, [("split", 1, [(0.18, 800.0), (0.22, 800.0)], "x")])
    assert ctl.refused_actions == [] and not reg.get_state(1).active
    assert sorted(round(s.primary_center, 3) for s in reg.active_states()) == [0.1, 0.18, 0.22, 0.3]


def test_split_child_at_the_parent_centre_is_a_duplicate_even_though_the_parent_retires():
    reg = _grid()
    parent = _parent_id(reg, 0.2, 0.0)
    snap = _snapshot(reg)
    ctl = _apply(reg, [("split", parent, [(0.2, 800.0, 0.0, 5.0), (0.2, 800.0, 0.4, 8.0)], "x")])
    assert _snapshot(reg) == snap and [r["reason"] for r in ctl.refused_actions] == ["duplicate"]


def test_split_children_duplicating_each_other_are_refused_as_one_unit():
    reg = _grid()
    parent = _parent_id(reg, 0.2, 0.0)
    snap = _snapshot(reg)
    ctl = _apply(reg, [("split", parent, [(0.2, 800.0, 0.4, 8.0), (0.2, 800.0, 0.4 + 1e-4, 8.0)], "x")])
    assert _snapshot(reg) == snap and [r["reason"] for r in ctl.refused_actions] == ["duplicate"]


def test_duplicate_identity_is_restraint_aware():
    # A CV2-only child (k1 = 0) at an existing 2D centre's coordinates is a different window (P6).
    reg = _grid()
    ctl = _apply(reg, [("add", 0, (0.2, 0.0, 0.0, 5.0), "cv2-only at a 2D centre")])
    assert ctl.refused_actions == []
    assert len(_centre(reg, 0.2, 0.0)) == 2


def test_split_budget_is_the_whole_action_children_times_rungs_minus_the_retired_parent_rungs():
    reg = _grid(LADDER)                                   # 36 states
    parent = _parent_id(reg, 0.2, 0.0)
    children = [(0.2, 800.0, -0.4, 8.0), (0.2, 800.0, 0.4, 8.0)]   # +8 -4 = +4
    snap = _snapshot(reg)
    ctl = _apply(reg, [("split", parent, children, "x")], policy=AdaptiveDecisionPolicy(max_replicas_budget=39))
    assert _snapshot(reg) == snap
    r = ctl.refused_actions[0]
    assert (r["reason"], r["added"], r["n_active"], r["budget"], r["index"]) == ("max_replicas_budget", 4, 36, 39, 0)
    ctl = _apply(reg, [("split", parent, children, "x")], policy=AdaptiveDecisionPolicy(max_replicas_budget=40))
    assert ctl.refused_actions == [] and len(reg.active_states()) == 40


def test_split_validation_precedence_mandatory_before_duplicate_before_budget():
    reg = _grid()
    parent = _parent_id(reg, 0.2, 0.0)
    reg.get_state(parent).metadata["mandatory"] = True
    ctl = _apply(reg, [("split", parent, [(0.2, 800.0, 0.0, 5.0)] * 3, "x")],
                 policy=AdaptiveDecisionPolicy(max_replicas_budget=1))
    assert [r["reason"] for r in ctl.refused_actions] == ["mandatory"]


def test_split_of_an_unknown_or_retired_parent_is_refused():
    reg = _grid()
    reg.retire_state(4, 0, "gone")
    ctl = _apply(reg, [("split", 999, [(0.2, 800.0, 0.4, 8.0)], "x"),
                       ("split", 4, [(0.2, 800.0, 0.4, 8.0)], "x")])
    assert [(r["reason"], r["index"]) for r in ctl.refused_actions] == [("unknown_state", 0), ("inactive", 1)]


# ---- k2 ceiling and coupling gate --------------------------------------------------------------

def test_child_k2_is_clamped_to_the_live_cv2_k_max_and_the_clamp_is_recorded():
    reg = _grid()
    parent = _parent_id(reg, 0.2, 0.0)
    ctl = _apply(reg, [("split", parent, [(0.2, 800.0, -0.4, 400.0), (0.2, 800.0, 0.4, 8.0)], "x"),
                       ("add", 0, (0.15, 800.0, -1.0, 300.0), "bridge")], secondary_k_max=200.0)
    assert ctl.refused_actions == []
    (kid,) = _centre(reg, 0.2, -0.4)
    (bridge,) = _centre(reg, 0.15, -1.0)
    assert kid.secondary_k == 200.0 and "clamped" in kid.reason
    assert bridge.secondary_k == 200.0 and "clamped" in bridge.reason
    assert _centre(reg, 0.2, 0.4)[0].secondary_k == 8.0


def test_without_a_ceiling_k2_passes_through_as_today():
    reg = _grid()
    _apply(reg, [("add", 0, (0.15, 800.0, -1.0, 300.0), "bridge")])
    assert _centre(reg, 0.15, -1.0)[0].secondary_k == 300.0


class _FakeGate:
    """Duck-typed AdaptiveCouplingGate: refuse below k_min, else lower to ``cap``."""

    def __init__(self, cap, k_min=0.0):
        self.cap, self.k_min, self.contexts = cap, k_min, []

    def gate(self, c1, k1, c2, k2, *, context):
        self.contexts.append(context)
        if k2 is None:
            return None, None
        if k2 <= self.cap:
            return k2, None
        if self.cap < self.k_min:
            return None, "coupling gate: below k_min"
        return self.cap, f"secondary_k lowered {k2} -> {self.cap} by the CV2 coupling gate"


def test_a_coupling_refusal_of_any_child_refuses_the_whole_split():
    reg = _grid()
    parent = _parent_id(reg, 0.2, 0.0)
    snap = _snapshot(reg)
    gate = _FakeGate(cap=6.0, k_min=7.0)
    ctl = _apply(reg, [("split", parent, [(0.2, 800.0, -0.4, 5.0), (0.2, 800.0, 0.4, 9.0)], "x")],
                 coupling_gate=gate)
    assert _snapshot(reg) == snap and [r["reason"] for r in ctl.refused_actions] == ["below_k_min"]
    assert all(c.startswith("apply: ") for c in gate.contexts)


def test_a_lowering_gate_sets_the_child_k2_and_notes_it():
    reg = _grid()
    ctl = _apply(reg, [("tica_coverage_add", 0, (0.1, 800.0, 2.0, 9.0), "cov", {})], coupling_gate=_FakeGate(6.0))
    assert ctl.refused_actions == []
    (kid,) = _centre(reg, 0.1, 2.0)
    assert kid.secondary_k == 6.0 and "coupling gate" in kid.reason


def test_na_or_absent_gate_changes_nothing():
    from gareus.adaptive.pair_runtime import AdaptiveCouplingGate, DriverPair
    actions = [("add", 0, (0.15, 800.0, -1.0, 5.0), "a"), ("tica_coverage_add", 0, (0.1, 800.0, 2.0, 9.0), "c", {})]
    r1, r2 = _grid(LADDER), _grid(LADDER)
    _apply(r1, actions)
    _apply(r2, actions, coupling_gate=AdaptiveCouplingGate(DriverPair("NA", "not residual"), temperature_k=300.0))
    assert _snapshot(r1) == _snapshot(r2)


def test_regating_a_k2_the_gate_already_lowered_is_idempotent():
    from test_cv2_coupling_gate import T, _adds, _bridge_case, _fit
    from gareus.adaptive.pair_runtime import AdaptiveCouplingGate, DriverPair
    fit, _, _ = _fit(2)
    reg, diag, pol = _bridge_case(fit)
    gate = AdaptiveCouplingGate(DriverPair("bound", fit=fit, j=1), temperature_k=T)
    (action,) = _adds(reg, diag, pol, gate)
    ctl = _apply(reg, [action], policy=pol, coupling_gate=gate)
    assert ctl.refused_actions == []
    new = [s for s in reg.active_states() if s.source == "adaptive_production"]
    assert len(new) == 1 and new[0].secondary_k == action[2][3]
    assert new[0].reason == action[3]                       # no second "lowered" note
    assert gate.records[-1]["context"].startswith("apply: ")
    assert gate.records[-1]["k2"] == pytest.approx(action[2][3], rel=1e-9)   # at the threshold again


# ---- behaviour differences vs the pre-P2 applier ----------------------------------------------

def test_an_add_duplicating_an_earlier_add_of_the_same_batch_is_refused():
    reg = _grid(LADDER)
    ctl = _apply(reg, [("add", 0, (0.15, 800.0, -1.0, 5.0), "edge a"),
                       ("add", 3, (0.15 + 1e-4, 800.0, -1.0, 5.0), "edge b")])
    assert [(r["action"], r["reason"], r["index"]) for r in ctl.refused_actions] == [("add", "duplicate", 1)]
    assert len(_centre(reg, 0.15, -1.0)) == 4                      # one centre, four rungs


def test_coverage_batch_duplicates_are_refused_and_without_refused_keeps_the_seed_pairing():
    reg = _grid()
    batch = [("tica_coverage_add", 0, (0.1, 800.0, 2.0, 7.0), "c0", {"k": 0}),
             ("tica_coverage_add", 0, (0.1, 800.0, 2.0, 7.0), "c1-dup", {"k": 1}),
             ("tica_coverage_add", 1, (0.2, 800.0, -2.0, 9.0), "c2", {"k": 2})]
    before = {s.state_id for s in reg.active_states()}
    kept = ap._without_refused(batch, ap._apply_registry_actions(reg, batch, 0))
    new_ids = sorted(s.state_id for s in reg.active_states() if s.state_id not in before)
    assert [a[3] for a in kept] == ["c0", "c2"] and len(new_ids) == 2
    for action, sid in zip(kept, new_ids):                        # the pairing the seed extraction uses
        st = reg.get_state(sid)
        assert (st.primary_center, st.secondary_center) == (action[2][0], action[2][2])


def test_a_refused_mandatory_retire_is_recorded_instead_of_silently_listed_as_applied():
    reg = _grid()
    reg.get_state(4).metadata["mandatory"] = True
    ctl = _apply(reg, [("retire", 4, "conv"), ("retire", 5, "conv")])
    assert [(r["action"], r["reason"], r["index"]) for r in ctl.refused_actions] == [("retire", "mandatory", 0)]
    assert reg.get_state(4).active and not reg.get_state(5).active


@pytest.mark.parametrize("reg_lams,drop,add,budget,reason", [
    ((0.0,), [0.3], [0.2], 0, "no_ladder"),
    ((0.0, 0.3, 0.6, 1.0), [1.0], [0.2], 0, "ladder_endpoint"),
    ((0.0, 0.3, 0.6, 1.0), [0.45], [0.3], 0, "no_change"),
    ((0.0, 0.3, 0.6, 1.0), [], [0.2], 36, "max_replicas_budget"),
])
def test_a_refused_respace_is_recorded(reg_lams, drop, add, budget, reason):
    reg = _grid(reg_lams)
    snap = _snapshot(reg)
    ctl = _apply(reg, [("respace_ladder", drop, add, "r")], policy=AdaptiveDecisionPolicy(max_replicas_budget=budget))
    assert _snapshot(reg) == snap
    assert [(r["action"], r["reason"], r["index"]) for r in ctl.refused_actions] == [("respace_ladder", reason, 0)]


def test_refusals_reach_the_applied_actions_ledger(tmp_path):
    reg = _grid()
    reg.get_state(4).metadata["mandatory"] = True
    actions = [("add", 0, (0.15, 800.0, -1.0, 5.0), "a"), ("split", 4, [(0.2, 800.0, 0.4, 8.0)], "x"),
               ("add", 0, (0.15, 800.0, -1.0, 5.0), "a again")]
    refused = ap._apply_registry_actions(reg, actions, 0)
    paths = reg.save(tmp_path)
    led = json.loads(ap._record_applied_actions(tmp_path, 0, actions, Path(paths["registry_json"]),
                                                refused=refused).read_text())
    assert [a[3] for a in led["actions"]] == ["a"]
    assert [(r["reason"], r["proposal"][0]) for r in led["refused"]] == [("mandatory", "split"), ("duplicate", "add")]


# ---- unchanged outcomes for today's valid actions ---------------------------------------------

def test_valid_adds_add_rung_and_respace_reproduce_the_pre_p2_registry():
    # Pinned against the pre-P2 applier's output (state ids, centres, springs, reasons, parents).
    reg = _grid((0.0, 1.0))
    _apply(reg, [("add", 0, (0.15, 800.0, -1.0, 5.0), "bridge a"), ("add_rung", 0.5, "gap"),
                 ("tica_coverage_add", 1, (0.2, 800.0, -2.0, 9.0), "cov", {"population_count": 7})])
    new = [s for s in reg.all_states() if s.state_id >= 18]
    assert [(s.state_id, s.primary_center, s.secondary_center, s.gamd_lambda, s.parent_state_id, s.source)
            for s in new[:2]] == [(18, 0.15, -1.0, 0.0, 0, "adaptive_production"),
                                 (19, 0.15, -1.0, 1.0, 0, "adaptive_production")]
    assert new[0].reason == "bridge a; rung lambda=0.0"
    assert sum(1 for s in new if s.source == "adaptive_production_rung") == 10
    assert [s.gamd_lambda for s in new if s.source == "tica_coverage"] == [0.0, 0.5, 1.0]


def test_a_cv1_only_add_with_two_tuple_params_still_works():
    reg = WindowStateRegistry()
    for c in (0.1, 0.2, 0.3):
        reg.add_state(c, 800.0, epoch=0, source="seed")
    ctl = _apply(reg, [("add", None, (0.15, 800.0), "a")])
    assert ctl.refused_actions == [] and len(reg.active_states()) == 4


# ---- state_id Hamiltonian immutability ---------------------------------------------------------

def test_an_apply_that_mutates_an_existing_state_hamiltonian_raises(monkeypatch):
    reg = _grid()
    real = AdaptiveProductionController._add_centre_on_every_rung

    def evil(self, *a, **k):
        self.registry.get_state(0).secondary_k = 99.0
        return real(self, *a, **k)

    monkeypatch.setattr(AdaptiveProductionController, "_add_centre_on_every_rung", evil)
    with pytest.raises(RuntimeError, match="Hamiltonian"):
        _apply(reg, [("add", 0, (0.15, 800.0, -1.0, 5.0), "a")])


def test_an_apply_that_deletes_a_state_raises(monkeypatch):
    reg = _grid()

    def evil(self, epoch, lam, reason):
        del self.registry._states[0]
        return []

    monkeypatch.setattr(AdaptiveProductionController, "_add_rung_at_every_centre", evil)
    with pytest.raises(RuntimeError, match="state_id"):
        _apply(reg, [("add_rung", 0.5, "r")])


def test_nan_and_none_parameters_compare_equal_in_the_hamiltonian_check():
    reg = WindowStateRegistry()
    reg.add_state(0.1, 800.0, secondary_center=None, epoch=0, source="seed")
    reg.add_state(0.2, 800.0, secondary_center=float("nan"), secondary_k=float("nan"), epoch=0, source="seed")
    _apply(reg, [("extend", 0, "more")])                         # must not raise


# ---- convergence -------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["split", "refine"])
def test_a_proposed_split_or_refine_blocks_convergence(kind):
    pol = AdaptiveDecisionPolicy()
    assert ap._adaptive_production_converged([], {"edges": []}, pol)
    assert not ap._adaptive_production_converged([(kind, 0, [], "x")], {"edges": []}, pol)


# ---- the epoch loop threads the live ceiling and the gate to the applier ----------------------

@pytest.mark.parametrize("gate_on", [False, True])
def test_epoch_loop_hands_the_live_cv2_k_max_and_the_gate_to_the_applier(tmp_path, monkeypatch, gate_on):
    from test_ladder_adapt_resume_e2e import _ReachedFinal, _campaign, _stub
    args, out, adaptive = _campaign(tmp_path)
    args.cv2_k_max = 123.0
    args.adaptive_production_cv2_coupling_gate = gate_on
    _stub(monkeypatch, {"propose": 0, "diag_states": []}, {"armed": False})
    seen = []
    real = ap._apply_registry_actions

    def spy(registry, actions, epoch, policy=None, **kw):
        seen.append(kw)
        return real(registry, actions, epoch, policy=policy, **kw)

    monkeypatch.setattr(ap, "_apply_registry_actions", spy)
    with pytest.raises(_ReachedFinal):
        ap.run_adaptive_production_auto_loop(args, out, None, None, None, None, None, None)
    assert seen and seen[0]["secondary_k_max"] == 123.0
    assert (seen[0]["coupling_gate"] is not None) is gate_on
