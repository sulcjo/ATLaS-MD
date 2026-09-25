import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gareus.adaptive_production import (  # noqa: E402
    AdaptiveDecisionPolicy,
    WindowStateRegistry,
    _adaptive_production_converged,
    _edge_is_measured_weak,
    evaluate_adaptive_quality_gate,
    propose_actions_from_diagnostics,
)


POL = AdaptiveDecisionPolicy()


def _state_row(state_id, sample_count=1000):
    return {"state_id": int(state_id), "sample_count": int(sample_count), "warnings": []}


def _two_state_registry():
    reg = WindowStateRegistry()
    reg.add_state(primary_center=0.0, primary_k=800.0, epoch=0, source="initial", reason="seed")
    reg.add_state(primary_center=1.0, primary_k=800.0, epoch=0, source="initial", reason="seed")
    return reg


def test_an_unmeasured_rung_edge_is_not_weak():
    assert _edge_is_measured_weak({"edge_type": "rung", "overlap": None, "mbar_overlap": None}, POL) is False


def test_a_measured_low_rung_edge_is_weak():
    assert _edge_is_measured_weak({"edge_type": "rung", "mbar_overlap": 0.05}, POL) is True


def test_a_spatial_edge_uses_its_cv_overlap():
    assert _edge_is_measured_weak({"edge_type": "nearest_2d", "overlap": 0.9}, POL) is False
    assert _edge_is_measured_weak({"edge_type": "nearest_2d", "overlap": 0.1}, POL) is True
    assert _edge_is_measured_weak({"edge_type": "nearest_2d", "overlap": None}, POL) is False


def test_low_measured_acceptance_still_counts_for_spatial_edges():
    assert _edge_is_measured_weak({"edge_type": "nearest_2d", "overlap": 0.9,
                                   "exchange_acceptance": 0.01}, POL) is True


def test_rung_edge_uses_min_rung_overlap_not_topup_weak_overlap():
    pol = AdaptiveDecisionPolicy(min_rung_overlap=0.20, topup_weak_overlap=0.10)
    assert _edge_is_measured_weak({"edge_type": "rung", "mbar_overlap": 0.15}, pol) is True


def test_quality_gate_does_not_fail_on_an_unmeasured_spatial_edge_overlap():
    reg = _two_state_registry()
    ids = sorted(int(s.state_id) for s in reg.active_states())
    pol = AdaptiveDecisionPolicy()
    final_diag = {
        "n_sample_rows": 2000,
        "states": [_state_row(sid) for sid in ids],
        "edges": [{"state_i": ids[0], "state_j": ids[1], "edge_type": "nearest_2d", "overlap": None}],
    }
    with tempfile.TemporaryDirectory() as tmp:
        gate = evaluate_adaptive_quality_gate(Path(tmp), reg, final_diag, pol)
    assert gate.get("weak_edges") == [], gate.get("weak_edges")
    assert gate.get("status") == "ok", gate


def test_propose_actions_does_not_bridge_an_unmeasured_spatial_edge():
    reg = _two_state_registry()
    ids = sorted(int(s.state_id) for s in reg.active_states())
    pol = AdaptiveDecisionPolicy()
    diag = {
        "states": [_state_row(sid) for sid in ids],
        "edges": [{"state_i": ids[0], "state_j": ids[1], "edge_type": "nearest_2d", "overlap": None}],
    }
    actions = propose_actions_from_diagnostics(reg, diag, policy=pol)
    assert not any(str(a[0]) == "add" for a in actions), actions


def test_converged_is_not_blocked_by_an_unscored_rung_edge():
    """An unscored rung edge (mbar_overlap is None -- only available after the
    campaign-end union solve) must never make _adaptive_production_converged
    report it as weak, otherwise convergence is impossible on any ladder run
    and write_epoch_action_report's converged_by_current_policy disagrees with
    the (already-fixed) gate."""
    pol = AdaptiveDecisionPolicy()
    diag = {
        "states": [_state_row(0), _state_row(1)],
        "edges": [{"state_i": 0, "state_j": 1, "edge_type": "rung",
                   "overlap": None, "mbar_overlap": None}],
    }
    assert _adaptive_production_converged([], diag, pol) is True
