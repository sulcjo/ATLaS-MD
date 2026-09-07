import numpy as np


def _traces(n=8, sigma=5.0, seed=0):
    rng = np.random.default_rng(seed)
    return {m: {"v_pep_kj": rng.normal(-50, sigma, 500), "v_dih_kj": rng.normal(20, 2.0, 500)} for m in range(n)}


def test_coverage_gate_fails_on_empty_cell_and_low_completion():
    from gareus.swarm.gates import coverage_gate
    rows = [{"member_id": i, "cell_id": "0_0_0" if i < 4 else "1_0_0"} for i in range(8)]
    assert coverage_gate(rows, set(range(8)))["ok"]
    assert not coverage_gate(rows, {0, 1, 2, 3})["ok"]                 # cell 1_0_0 empty
    assert not coverage_gate(rows, {0, 1, 2, 3, 4, 5}, min_done_fraction=0.9)["ok"]


def test_envelope_stability_passes_iid_and_fails_shifted_halves():
    from gareus.swarm.gates import envelope_stability_gate
    assert envelope_stability_gate(_traces(), discard=0)["ok"]
    tr = _traces()
    for m in tr:
        if m % 2: tr[m]["v_pep_kj"] = tr[m]["v_pep_kj"] + 40.0        # odd members shifted by 8σ
    g = envelope_stability_gate(tr, discard=0)
    assert not g["ok"] and "Total" in " ".join(g["reasons"])


def test_ladder_ess_gate_and_extension_plan():
    from gareus.swarm.gates import ladder_ess_gate, extension_plan
    assert ladder_ess_gate({"lambdas": [0, 0.5, 1], "ess_per_rung": [500, 200, 120], "extrapolated_from_rung": None})["ok"]
    bad = ladder_ess_gate({"lambdas": [0, 0.5, 1], "ess_per_rung": [500, 20, 5], "extrapolated_from_rung": 1})
    assert not bad["ok"]
    plan = extension_plan({"replicates_per_cell": 3}, {"status": "fail", "gates": {"ladder_ess": bad, "coverage": {"ok": True}, "envelope_stability": {"ok": True}, "graft": {"ok": True}}})
    assert plan["extra_replicates_per_cell"] == 3


def test_graft_gate_counts_total_failed_fraction():
    from gareus.swarm.gates import graft_gate
    # 10 summaries: 2 md_failed, 0 graft_failed (20% total failed)
    summaries = [{"status": "ok"} for _ in range(8)] + [{"status": "md_failed"}, {"status": "md_failed"}]
    # At max 0.10 (10%): 20% > 10% → should FAIL
    g = graft_gate(summaries, max_fallback_fraction=0.10)
    assert not g["ok"] and g["n_md_failed"] == 2 and g["n_graft_failed"] == 0 and abs(g["failed_fraction"] - 0.2) < 0.01
    # At max 0.25 (25%): 20% <= 25% → should PASS
    g = graft_gate(summaries, max_fallback_fraction=0.25)
    assert g["ok"] and g["failed_fraction"] == 0.2


def test_envelope_stability_gate_fails_closed_on_empty_halves():
    from gareus.swarm.gates import envelope_stability_gate
    # Single member: cannot evaluate (no pair)
    tr_single = {0: {"v_pep_kj": np.array([1.0, 2.0, 3.0]), "v_dih_kj": np.array([10.0, 20.0, 30.0])}}
    g = envelope_stability_gate(tr_single, discard=0)
    assert not g["ok"] and any("cannot evaluate" in r.lower() for r in g["reasons"])


def test_envelope_stability_gate_fails_closed_on_pooling_error():
    from gareus.swarm.gates import envelope_stability_gate
    # Two members with empty arrays (after discard, nothing left)
    tr_empty = {
        0: {"v_pep_kj": np.array([]), "v_dih_kj": np.array([])},
        1: {"v_pep_kj": np.array([]), "v_dih_kj": np.array([])},
    }
    g = envelope_stability_gate(tr_empty, discard=0)
    assert not g["ok"] and any("pooling" in r.lower() or "finite" in r.lower() for r in g["reasons"])


def test_extension_plan_caps_at_base_replicates():
    from gareus.swarm.gates import extension_plan
    # base_replicates_per_cell=3, current=6, stability failure → extra=6, total=12, capped at 4*3=12 ✓
    bad_gate = {"status": "fail", "gates": {"stability": {"ok": False}, "ladder_ess": {"ok": True}, "coverage": {"ok": True}, "graft": {"ok": True}}}
    plan = extension_plan({"replicates_per_cell": 6, "base_replicates_per_cell": 3}, {"status": "fail", "gates": {"envelope_stability": {"ok": False}, "ladder_ess": {"ok": True}, "coverage": {"ok": True}, "graft": {"ok": True}}})
    assert plan["extra_replicates_per_cell"] == 6  # doubles current (6), total=12, capped at 4*3=12
    # current=12, base=3, stability fails → extra would be 12, but total would be 24, capped at 12 → extra=0
    plan = extension_plan({"replicates_per_cell": 12, "base_replicates_per_cell": 3}, {"status": "fail", "gates": {"envelope_stability": {"ok": False}, "ladder_ess": {"ok": True}, "coverage": {"ok": True}, "graft": {"ok": True}}})
    assert plan["extra_replicates_per_cell"] == 0  # already at cap
