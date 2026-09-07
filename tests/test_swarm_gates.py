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
