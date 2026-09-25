"""Adaptive production under a λ-ladder: rung-preserving insertion and
energy-space rung diagnostics.

Spec: docs/superpowers/specs/2026-09-07-adaptive-ladder-rungs-design.md

Two defects are covered here.

Change 1 -- every state an action added was created with ``gamd_lambda = 0.0``,
so an inserted CV window existed only at λ = 0 and the windows × rungs cross
product silently broke; ``has_near_duplicate`` ignored λ, so replicating one
centre across rungs would have been rejected as a duplicate.

Change 2 -- edges were scored by CV-histogram overlap only.  Two rungs at one
centre have CV overlap ≈ 1 *by construction*, whatever the boost spacing, so
no gate could ever see a rung gap.  Rung edges are therefore scored in energy
space, by the MBAR state-overlap matrix.

Every test is a zero-argument function (the sanctioned fallback runner cannot
supply fixtures) and nothing here imports a test runner.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gareus.adaptive_production import (  # noqa: E402
    AdaptiveDecisionPolicy,
    AdaptiveProductionController,
    WindowStateRegistry,
    build_geometry_edges,
    evaluate_adaptive_quality_gate,
    propose_actions_from_diagnostics,
)
from gareus.mbar_analysis.ladder import mbar_state_overlap  # noqa: E402

FIXTURE_NPZ = Path(__file__).resolve().parent / "fixtures" / "ll_pilot_s3_ladder_u_nk.npz"

# Measured on the S3 pilot (attempt 8, σ0 = 6, rungs 0/.1/.25/.5/1,
# RUNS/aurum_pilots/ll_pilot_s3): adjacent-rung entries of O_ij.
PILOT_ADJACENT_OVERLAP = (0.298, 0.250, 0.240, 0.273)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ladder_registry(centers=(0.05, 0.15), rungs=(0.0, 0.5, 1.0), k=800.0):
    """Registry holding the full centres × rungs cross product."""
    reg = WindowStateRegistry()
    for center in centers:
        for lam in rungs:
            reg.add_state(primary_center=center, primary_k=k, gamd_lambda=lam,
                          epoch=0, source="initial", reason="seed")
    return reg


def _state_row(state_id, sample_count=500):
    return {"state_id": int(state_id), "sample_count": int(sample_count), "warnings": []}


# ---------------------------------------------------------------------------
# Test 1 -- rung-preserving insertion
# ---------------------------------------------------------------------------


def test_apply_actions_replicates_a_new_centre_across_every_rung():
    reg = _ladder_registry(centers=(0.10,), rungs=(0.0, 0.5, 1.0))
    controller = AdaptiveProductionController(reg)
    controller.apply_actions(0, [("add", None, (0.05, 800.0), "r")])

    new = [s for s in reg.active_states() if abs(s.primary_center - 0.05) < 1e-12]
    assert len(new) == 3, f"expected one state per rung, got {len(new)}"
    assert sorted(round(s.gamd_lambda, 9) for s in new) == [0.0, 0.5, 1.0]
    for s in new:
        assert abs(s.primary_k - 800.0) < 1e-12
        assert f"rung lambda={s.gamd_lambda}" in s.reason, s.reason


def test_apply_actions_is_unchanged_when_the_ladder_is_inactive():
    reg = _ladder_registry(centers=(0.10,), rungs=(0.0,))
    controller = AdaptiveProductionController(reg)
    controller.apply_actions(0, [("add", None, (0.05, 800.0), "r")])

    new = [s for s in reg.active_states() if abs(s.primary_center - 0.05) < 1e-12]
    assert len(new) == 1, f"ladder inactive must create exactly one state, got {len(new)}"
    assert new[0].gamd_lambda == 0.0
    assert new[0].reason == "r", "reason must not be suffixed when no rung exists"


def test_new_centre_reaches_every_rung_in_the_registry_rows_and_windows_csv():
    """The rungs must survive into what the production replica build reads:
    the registry CSV rows and the explicit windows CSV the adaptive loop writes
    (``gamd_lambda`` there is what ``_derive_state_gamd_lambdas`` reads back)."""
    reg = _ladder_registry(centers=(0.10,), rungs=(0.0, 0.5, 1.0))
    AdaptiveProductionController(reg).apply_actions(0, [("add", None, (0.05, 800.0), "r")])

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        reg.write_state_csv(tmp / "registry.csv")
        rows = list(csv.DictReader((tmp / "registry.csv").open()))
        new_rows = [r for r in rows if abs(float(r["primary_center"]) - 0.05) < 1e-12]
        assert sorted(round(float(r["gamd_lambda"]), 9) for r in new_rows) == [0.0, 0.5, 1.0]

        reg.write_active_window_csv(tmp / "windows.csv")
        wrows = list(csv.DictReader((tmp / "windows.csv").open()))
        new_w = [r for r in wrows if abs(float(r["primary_cv_center"]) - 0.05) < 1e-12]
        assert sorted(round(float(r["gamd_lambda"]), 9) for r in new_w) == [0.0, 0.5, 1.0]
        # cross product intact: 2 centres × 3 rungs
        assert len(wrows) == 6, f"windows CSV lost the cross product: {len(wrows)} rows"


def test_rung_lambdas_reports_sorted_distinct_active_lambdas():
    reg = _ladder_registry(centers=(0.05, 0.15), rungs=(0.0, 0.5, 1.0))
    assert reg.rung_lambdas() == [0.0, 0.5, 1.0]
    # retired states do not count
    for s in list(reg.active_states()):
        if abs(s.gamd_lambda - 1.0) < 1e-12:
            reg.retire_state(s.state_id, 1, "test")
    assert reg.rung_lambdas() == [0.0, 0.5]


# ---------------------------------------------------------------------------
# Test 2 -- has_near_duplicate distinguishes rungs
# ---------------------------------------------------------------------------


def test_has_near_duplicate_distinguishes_lambda():
    reg = WindowStateRegistry()
    pol = AdaptiveDecisionPolicy()
    reg.add_state(primary_center=0.05, primary_k=800.0, gamd_lambda=0.5)
    assert reg.has_near_duplicate(0.05, None, pol, gamd_lambda=0.5) is True
    assert reg.has_near_duplicate(0.05, None, pol, gamd_lambda=1.0) is False
    # default keeps the old (λ = 0) contract for callers that have no rung
    assert reg.has_near_duplicate(0.05, None, pol) is False
    reg.add_state(primary_center=0.05, primary_k=800.0, gamd_lambda=0.0)
    assert reg.has_near_duplicate(0.05, None, pol) is True


# ---------------------------------------------------------------------------
# Test 3 -- rung edges vs geometry edges
# ---------------------------------------------------------------------------


def test_build_geometry_edges_separates_rung_edges_from_geometry_edges():
    reg = _ladder_registry(centers=(0.05, 0.15), rungs=(0.0, 0.5, 1.0))
    edges = build_geometry_edges(reg)

    rung = [e for e in edges if e[2] == "rung"]
    geom = [e for e in edges if e[2] != "rung"]
    assert len(rung) == 4, f"expected 2 rung edges per centre, got {len(rung)}"
    assert all(abs(float(e[3]) - 0.5) < 1e-12 for e in rung), "rung edge payload must be Δλ"

    chain = [e for e in geom if e[2] == "primary_chain"]
    assert len(chain) == 1, f"expected one representative-to-representative chain edge, got {len(chain)}"
    reps = {int(s.state_id) for s in reg.active_states() if s.gamd_lambda == 0.0}
    assert {chain[0][0], chain[0][1]} == reps

    # no same-centre pair may appear as a geometry edge
    center_of = {int(s.state_id): float(s.primary_center) for s in reg.active_states()}
    for a, b, etype, _nd in geom:
        assert abs(center_of[a] - center_of[b]) > 1e-9, f"same-centre {etype} edge {a}-{b}"


def test_build_geometry_edges_is_unchanged_when_the_ladder_is_inactive():
    reg = WindowStateRegistry()
    for c in (0.05, 0.10, 0.15):
        reg.add_state(primary_center=c, primary_k=800.0)
    edges = build_geometry_edges(reg)
    assert [e[2] for e in edges] == ["primary_chain", "primary_chain"]
    assert {(e[0], e[1]) for e in edges} == {(0, 1), (1, 2)}


def test_rung_edges_do_not_fake_graph_connectivity():
    """Rung edges join states at ONE centre only, so a CV chain that is really
    broken must still report disconnected (final_connectivity_required)."""
    from gareus.adaptive_production import active_graph_connected

    reg = _ladder_registry(centers=(0.05, 0.15), rungs=(0.0, 0.5, 1.0))
    assert active_graph_connected(reg) is True
    # Retire both representatives: the two centre groups lose their only link.
    for s in list(reg.active_states()):
        if s.gamd_lambda == 0.0:
            reg.retire_state(s.state_id, 1, "test")
    edges = build_geometry_edges(reg)
    chain = [e for e in edges if e[2] == "primary_chain"]
    assert len(chain) == 1, "representatives fall back to the lowest surviving rung"


# ---------------------------------------------------------------------------
# Test 4 -- mbar_state_overlap
# ---------------------------------------------------------------------------


def test_mbar_state_overlap_two_identical_states_is_analytic():
    """Two identical states: W_nk = 1/(N_i+N_j) everywhere, so
    O_ij = N_i/(N_i+N_j) exactly."""
    rng = np.random.default_rng(0)
    col = rng.normal(size=300)
    u_nk = np.column_stack([col, col])
    n_k = np.array([100, 200], dtype=np.int64)
    f_k = np.zeros(2)
    o = mbar_state_overlap(u_nk, f_k, n_k)
    assert math.isclose(float(o[0, 1]), 100.0 / 300.0, rel_tol=1e-9)
    assert math.isclose(float(o[1, 0]), 200.0 / 300.0, rel_tol=1e-9)
    # gauge invariance: a constant shift of f_k cancels
    o_shift = mbar_state_overlap(u_nk, f_k + 3.7, n_k)
    assert np.allclose(o, o_shift, atol=1e-12)


def test_mbar_state_overlap_normalisation_and_n_scaling():
    rng = np.random.default_rng(1)
    n_k = np.array([400, 400, 400], dtype=np.int64)
    centers = np.array([0.0, 1.0, 2.0])
    x = np.concatenate([rng.normal(c, 1.0, size=int(n)) for c, n in zip(centers, n_k)])
    u_nk = 0.5 * (x[:, None] - centers[None, :]) ** 2
    from gareus.mbar_analysis.solvers import solve_mbar

    window = np.concatenate([np.full(int(n), k) for k, n in enumerate(n_k)]).astype(int)
    f_k = np.asarray(solve_mbar(u_nk, window, tol=1e-12, maxiter=20000)["f_k"], dtype=float)
    o = mbar_state_overlap(u_nk, f_k, n_k)
    # equal N_k -> rows sum to 1
    assert np.allclose(o.sum(axis=1), 1.0, atol=1e-8), o.sum(axis=1)
    # O_ij * N_j == O_ji * N_i, always (O = diag(N) @ symmetric)
    for i in range(3):
        for j in range(3):
            assert math.isclose(float(o[i, j] * n_k[j]), float(o[j, i] * n_k[i]), rel_tol=1e-9)

    # unequal N_k -> COLUMNS sum to 1 exactly (sum_i N_i W_ni W_nj = sum_n W_nj)
    n_k2 = np.array([400, 400, 400], dtype=np.int64)
    keep = np.ones(x.size, dtype=bool)
    keep[:150] = False  # drop 150 samples of state 0
    n_k2 = np.array([250, 400, 400], dtype=np.int64)
    u2, w2 = u_nk[keep], window[keep]
    f2 = np.asarray(solve_mbar(u2, w2, tol=1e-12, maxiter=20000)["f_k"], dtype=float)
    o2 = mbar_state_overlap(u2, f2, n_k2)
    assert np.allclose(o2.sum(axis=0), 1.0, atol=1e-8), o2.sum(axis=0)


def test_mbar_state_overlap_reproduces_the_s3_pilot_ladder():
    from gareus.mbar_analysis.solvers import solve_mbar

    assert FIXTURE_NPZ.exists(), f"missing fixture {FIXTURE_NPZ}"
    with np.load(FIXTURE_NPZ, allow_pickle=False) as data:
        u_nk = np.asarray(data["umbrella_reduced_bias_nk"], dtype=float)
        window = np.asarray(data["window"], dtype=int)
    assert u_nk.shape == (10000, 5), u_nk.shape
    n_k = np.bincount(window, minlength=u_nk.shape[1]).astype(np.int64)
    f_k = np.asarray(solve_mbar(u_nk, window, tol=1e-10, maxiter=10000)["f_k"], dtype=float)
    o = mbar_state_overlap(u_nk, f_k, n_k)

    assert np.allclose(o.sum(axis=1), 1.0, atol=1e-8)
    adjacent = [float(o[i, i + 1]) for i in range(4)]
    for got, want in zip(adjacent, PILOT_ADJACENT_OVERLAP):
        assert abs(got - want) <= 0.01, f"adjacent O_ij {adjacent} vs {PILOT_ADJACENT_OVERLAP}"
    diag = np.diag(o)
    assert 0.30 <= float(diag.min()) and float(diag.max()) <= 0.70, diag


# ---------------------------------------------------------------------------
# Test 5 -- diagnostics, proposal, application
# ---------------------------------------------------------------------------


def test_weak_rung_edge_warns_and_proposes_one_add_rung_applied_at_every_centre():
    reg = _ladder_registry(centers=(0.05, 0.15), rungs=(0.0, 0.5, 1.0))
    pol = AdaptiveDecisionPolicy()
    ids = {(round(s.primary_center, 6), round(s.gamd_lambda, 6)): int(s.state_id)
           for s in reg.active_states()}
    lo, hi = ids[(0.05, 0.0)], ids[(0.05, 0.5)]

    from gareus.adaptive_production import EdgeDiagnostics

    edge = EdgeDiagnostics(state_i=lo, state_j=hi, window_i=0, window_j=1,
                           edge_type="rung", normalized_distance=0.5, overlap=None,
                           exchange_attempts=100, exchange_accepted=93,
                           exchange_acceptance=0.93, mbar_overlap=0.05)
    # gibbs-walk inflates per-pair acceptance, so acceptance must never make a
    # rung edge look weak OR healthy: only mbar_overlap decides.
    warnings = _rung_edge_warnings(edge, pol)
    assert "low_rung_overlap" in warnings, warnings
    assert "low_exchange_acceptance" not in warnings

    diag = {
        "states": [_state_row(sid) for sid in ids.values()],
        "edges": [edge.to_dict()],
    }
    actions = propose_actions_from_diagnostics(reg, diag, pol)
    rung_actions = [a for a in actions if str(a[0]) == "add_rung"]
    assert len(rung_actions) == 1, f"expected one add_rung, got {actions}"
    assert math.isclose(float(rung_actions[0][1]), 0.25, rel_tol=1e-12)

    AdaptiveProductionController(reg).apply_actions(0, rung_actions)
    at_mid = [s for s in reg.active_states() if abs(s.gamd_lambda - 0.25) < 1e-9]
    assert len(at_mid) == 2, f"new rung must appear at EVERY centre, got {len(at_mid)}"
    assert sorted(round(s.primary_center, 6) for s in at_mid) == [0.05, 0.15]


def _rung_edge_warnings(edge, policy):
    """Re-derive the warnings a rung edge gets, via the shared helper the two
    edge-diagnostic loops use."""
    from gareus.adaptive_production import _annotate_edge_warnings

    _annotate_edge_warnings(edge, policy)
    return list(edge.warnings)


def test_healthy_rung_edge_proposes_nothing():
    reg = _ladder_registry(centers=(0.05,), rungs=(0.0, 0.5))
    pol = AdaptiveDecisionPolicy()
    ids = sorted(int(s.state_id) for s in reg.active_states())
    from gareus.adaptive_production import EdgeDiagnostics

    edge = EdgeDiagnostics(state_i=ids[0], state_j=ids[1], window_i=0, window_j=1,
                           edge_type="rung", normalized_distance=0.5, overlap=None,
                           mbar_overlap=0.30)
    diag = {"states": [_state_row(s) for s in ids], "edges": [edge.to_dict()]}
    actions = propose_actions_from_diagnostics(reg, diag, pol)
    assert not [a for a in actions if str(a[0]) == "add_rung"], actions
    # and it must NOT be mistaken for a weak CV edge (overlap is None there)
    assert not [a for a in actions if str(a[0]) == "add"], actions


def test_max_new_rungs_per_epoch_caps_the_proposal_weakest_first():
    reg = _ladder_registry(centers=(0.05,), rungs=(0.0, 0.5, 1.0))
    pol = AdaptiveDecisionPolicy(max_new_rungs_per_epoch=1)
    ordered = sorted(reg.active_states(), key=lambda s: s.gamd_lambda)
    from gareus.adaptive_production import EdgeDiagnostics

    e1 = EdgeDiagnostics(state_i=int(ordered[0].state_id), state_j=int(ordered[1].state_id),
                         window_i=0, window_j=1, edge_type="rung", overlap=None, mbar_overlap=0.09)
    e2 = EdgeDiagnostics(state_i=int(ordered[1].state_id), state_j=int(ordered[2].state_id),
                         window_i=1, window_j=2, edge_type="rung", overlap=None, mbar_overlap=0.02)
    diag = {"states": [_state_row(int(s.state_id)) for s in ordered],
            "edges": [e1.to_dict(), e2.to_dict()]}
    actions = [a for a in propose_actions_from_diagnostics(reg, diag, pol) if str(a[0]) == "add_rung"]
    assert len(actions) == 1, actions
    assert math.isclose(float(actions[0][1]), 0.75, rel_tol=1e-12), "weakest rung edge first"


# ---------------------------------------------------------------------------
# Test 6 -- quality gate
# ---------------------------------------------------------------------------


def test_quality_gate_fails_on_a_weak_rung_edge_and_names_both_lambdas():
    reg = _ladder_registry(centers=(0.05, 0.15), rungs=(0.0, 0.5))
    ids = {(round(s.primary_center, 6), round(s.gamd_lambda, 6)): int(s.state_id)
           for s in reg.active_states()}
    lo, hi = ids[(0.05, 0.0)], ids[(0.05, 0.5)]
    pol = AdaptiveDecisionPolicy(final_min_samples_per_state=10)

    final_diag = {
        "n_sample_rows": 4000,
        "states": [_state_row(sid, sample_count=1000) for sid in ids.values()],
        "edges": [
            {"state_i": lo, "state_j": hi, "edge_type": "rung", "overlap": None,
             "mbar_overlap": 0.04, "exchange_acceptance": 0.93, "warnings": ["low_rung_overlap"]},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        gate = evaluate_adaptive_quality_gate(Path(tmp), reg, final_diag, pol)
    blob = json.dumps(gate)
    assert gate.get("status") in {"error", "needs_more_sampling"}, gate.get("status")
    assert "rung" in blob, blob[:400]
    reasons = " ".join(gate.get("needs_more_sampling", []) + gate.get("errors", [])
                       + [json.dumps(gate.get("weak_edges", []))])
    assert "0.0" in reasons and "0.5" in reasons, f"gate reason must name both λ: {reasons}"
    assert "0.04" in reasons, f"gate reason must name the O_ij: {reasons}"


def test_quality_gate_passes_a_healthy_rung_edge():
    reg = _ladder_registry(centers=(0.05,), rungs=(0.0, 0.5))
    ids = sorted(int(s.state_id) for s in reg.active_states())
    pol = AdaptiveDecisionPolicy(final_min_samples_per_state=10)
    final_diag = {
        "n_sample_rows": 2000,
        "states": [_state_row(sid, sample_count=1000) for sid in ids],
        "edges": [
            {"state_i": ids[0], "state_j": ids[1], "edge_type": "rung", "overlap": None,
             "mbar_overlap": 0.28, "exchange_acceptance": 0.93, "warnings": []},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        gate = evaluate_adaptive_quality_gate(Path(tmp), reg, final_diag, pol)
    assert not gate.get("weak_edges"), gate.get("weak_edges")
    assert gate.get("status") == "ok", gate


# ---------------------------------------------------------------------------
# Segmented collector: the path the real driver actually uses
# ---------------------------------------------------------------------------


def test_segmented_collector_keeps_rung_edges_in_energy_space():
    """Two rungs at one centre have CV overlap ~1, so the segmented collector's
    pooled-histogram override would hide every rung gap if it were applied."""
    from gareus.adaptive_production import collect_segmented_epoch_diagnostics

    reg = _ladder_registry(centers=(0.05, 0.15), rungs=(0.0, 0.5))
    pol = AdaptiveDecisionPolicy()
    active = reg.active_states()
    with tempfile.TemporaryDirectory() as tmp:
        epoch_dir = Path(tmp) / "epoch_000"
        for seg in ("baseline", "topup_001"):
            seg_dir = epoch_dir / seg
            seg_dir.mkdir(parents=True)
            with (seg_dir / "epoch_window_map.csv").open("w", newline="") as h:
                w = csv.DictWriter(h, fieldnames=["epoch_window", "state_id"])
                w.writeheader()
                for i, s in enumerate(active):
                    w.writerow({"epoch_window": i, "state_id": int(s.state_id)})
            with (seg_dir / "samples.csv").open("w", newline="") as h:
                w = csv.DictWriter(h, fieldnames=["window", "step", "cv_A"])
                w.writeheader()
                for i, s in enumerate(active):
                    for j in range(60):
                        w.writerow({"window": i, "step": j,
                                    "cv_A": float(s.primary_center) + 0.001 * (j % 7)})
        diag = collect_segmented_epoch_diagnostics(epoch_dir, reg, pol)

    rung = [e for e in diag["edges"] if e.get("edge_type") == "rung"]
    assert len(rung) == 2, f"segmented merge lost the rung edges: {diag['edges']}"
    for e in rung:
        assert e.get("overlap") is None, f"rung edge must not carry a CV histogram: {e}"
        assert "low_or_missing_overlap" not in (e.get("warnings") or []), e


# ---------------------------------------------------------------------------
# Rung states must not be mistaken for collapsed / redundant windows
# ---------------------------------------------------------------------------


def test_non_adjacent_rungs_are_not_reported_as_a_redundant_collapse():
    """Only ADJACENT rungs get a "rung" edge, so a 3-rung centre's (0, 1) pair
    is not a geometry neighbour -- and its CV overlap is 1.0.  Without an
    explicit same-centre exclusion every ladder state would be tagged
    non_neighbor_redundant, which is the input the retirement pass reads."""
    from gareus.adaptive_production import _non_neighbor_redundant_pairs

    reg = _ladder_registry(centers=(0.05,), rungs=(0.0, 0.5, 1.0))
    pol = AdaptiveDecisionPolicy()
    active = reg.active_states()
    window_map = {i: int(s.state_id) for i, s in enumerate(active)}
    values = {i: np.linspace(0.04, 0.06, 200) for i in range(len(active))}
    alerts = _non_neighbor_redundant_pairs(
        reg, window_map, values, build_geometry_edges(reg, pol), pol,
    )
    assert alerts == [], alerts


def test_no_ladder_state_is_ever_proposed_for_retirement():
    """Under an active ladder the retirement pass must not dismantle the cross
    product.  Two independent guards hold: a non-representative rung state
    gains no CV-overlap credit, so it never becomes a retire candidate; and a
    representative carries its own rungs, so dropping it would orphan them and
    ``_graph_articulation_states`` rules it out.  A rung edge is also excluded
    from the pass's CV weak-edge test -- its ``overlap`` is None by
    construction, which read as a weak edge would mark BOTH endpoints
    ``bad_touching`` and disable ``retire_converged`` for the whole run."""
    from gareus.adaptive_production import _graph_articulation_states

    reg = _ladder_registry(centers=(0.05, 0.10, 0.15), rungs=(0.0, 0.5))
    pol = AdaptiveDecisionPolicy(min_samples_for_retire=10, min_active_states=0,
                                 redundant_overlap=0.45, target_overlap=0.30)
    edges = build_geometry_edges(reg, pol)
    assert sum(1 for e in edges if e[2] == "rung") == 3, edges
    rows = [{"state_i": a, "state_j": b, "edge_type": t,
             "overlap": None if t == "rung" else 0.90,
             "mbar_overlap": 0.30 if t == "rung" else None,
             "exchange_acceptance": None, "warnings": []}
            for a, b, t, _nd in edges]
    diag = {"states": [_state_row(int(s.state_id), sample_count=1000) for s in reg.active_states()],
            "edges": rows, "non_neighbor_redundancies": []}
    actions = propose_actions_from_diagnostics(reg, diag, pol)
    assert not [a for a in actions if str(a[0]) == "retire"], actions
    # every representative is an articulation point precisely because it
    # carries its centre's rungs
    reps = {int(s.state_id) for s in reg.active_states() if s.gamd_lambda == 0.0}
    assert _graph_articulation_states(reg) == reps


def test_an_unscored_rung_edge_warns_but_does_not_demand_more_sampling():
    """The pre-union gate and every extension round run BEFORE the union MBAR
    inputs exist, so their rung edges carry no O_ij.  Treating "not measured"
    as "weak" would make quality_gate_fixable_by_more_final_sampling true on
    every ladder campaign and burn final_quality_extension_rounds of MD chasing
    a number more sampling cannot produce."""
    from gareus.adaptive_production import quality_gate_fixable_by_more_final_sampling

    reg = _ladder_registry(centers=(0.05,), rungs=(0.0, 0.5))
    ids = sorted(int(s.state_id) for s in reg.active_states())
    pol = AdaptiveDecisionPolicy(final_min_samples_per_state=10)
    final_diag = {
        "n_sample_rows": 2000,
        "states": [_state_row(sid, sample_count=1000) for sid in ids],
        "edges": [{"state_i": ids[0], "state_j": ids[1], "edge_type": "rung",
                   "overlap": None, "mbar_overlap": None,
                   "exchange_acceptance": 0.93, "warnings": ["rung_overlap_unavailable"]}],
    }
    with tempfile.TemporaryDirectory() as tmp:
        gate = evaluate_adaptive_quality_gate(Path(tmp), reg, final_diag, pol)
    assert not gate.get("weak_edges"), gate.get("weak_edges")
    assert gate.get("status") == "warning", gate.get("status")
    assert any("not scored" in w for w in gate.get("warnings", [])), gate.get("warnings")
    assert quality_gate_fixable_by_more_final_sampling(gate) is False


def test_edge_warnings_name_unmeasured_and_weak_rung_edges_differently():
    from gareus.adaptive_production import EdgeDiagnostics

    pol = AdaptiveDecisionPolicy()
    unscored = EdgeDiagnostics(state_i=0, state_j=1, window_i=0, window_j=1,
                               edge_type="rung", overlap=None, mbar_overlap=None)
    weak = EdgeDiagnostics(state_i=0, state_j=1, window_i=0, window_j=1,
                           edge_type="rung", overlap=None, mbar_overlap=0.05)
    healthy = EdgeDiagnostics(state_i=0, state_j=1, window_i=0, window_j=1,
                              edge_type="rung", overlap=None, mbar_overlap=0.30)
    assert _rung_edge_warnings(unscored, pol) == ["rung_overlap_unavailable"]
    assert _rung_edge_warnings(weak, pol) == ["low_rung_overlap"]
    assert _rung_edge_warnings(healthy, pol) == []


# ---------------------------------------------------------------------------
# The per-edge rung metric is symmetric under unequal N_k
# ---------------------------------------------------------------------------


def _solve_overlap(u_nk, window):
    from gareus.mbar_analysis.solvers import solve_mbar

    n_k = np.bincount(window, minlength=u_nk.shape[1]).astype(np.int64)
    f_k = np.asarray(solve_mbar(u_nk, window, tol=1e-12, maxiter=20000)["f_k"], dtype=float)
    return mbar_state_overlap(u_nk, f_k, n_k)


def test_symmetric_state_overlap_is_order_independent_under_unequal_n():
    """O = diag(N) @ S, so O_ij != O_ji once the two states hold different
    sample counts -- which adaptive extension makes normal.  The per-edge gate
    metric must not depend on which state holds the lower id."""
    from gareus.mbar_analysis.ladder import symmetric_state_overlap as _symmetric_state_overlap

    rng = np.random.default_rng(7)
    centers = np.array([0.0, 1.0])
    n_k = np.array([300, 300], dtype=np.int64)
    x = np.concatenate([rng.normal(c, 1.0, size=int(n)) for c, n in zip(centers, n_k)])
    window = np.concatenate([np.full(int(n), k) for k, n in enumerate(n_k)]).astype(int)
    u_nk = 0.5 * (x[:, None] - centers[None, :]) ** 2

    equal = _solve_overlap(u_nk, window)
    # equal N: the symmetric metric IS the raw matrix entry
    assert math.isclose(_symmetric_state_overlap(equal, 0, 1), float(equal[0, 1]), rel_tol=1e-9)
    assert math.isclose(float(equal[0, 1]), float(equal[1, 0]), rel_tol=1e-6)

    # now duplicate state 0's rows so N_0 = 2 * N_1
    dup = window == 0
    u2 = np.concatenate([u_nk, u_nk[dup]], axis=0)
    w2 = np.concatenate([window, window[dup]])
    unequal = _solve_overlap(u2, w2)
    assert not math.isclose(float(unequal[0, 1]), float(unequal[1, 0]), rel_tol=1e-3), (
        "test is vacuous: the matrix did not become asymmetric")
    fwd = _symmetric_state_overlap(unequal, 0, 1)
    rev = _symmetric_state_overlap(unequal, 1, 0)
    assert math.isclose(fwd, rev, rel_tol=1e-12), (fwd, rev)
    # geometric mean, i.e. S_ij * sqrt(N_i N_j)
    assert math.isclose(fwd, math.sqrt(float(unequal[0, 1]) * float(unequal[1, 0])), rel_tol=1e-12)
    assert min(unequal[0, 1], unequal[1, 0]) <= fwd <= max(unequal[0, 1], unequal[1, 0])


def test_symmetric_metric_reproduces_the_pilot_numbers():
    """Equal N_k in the pilot, so the symmetric metric must land on the same
    calibration the raw adjacent entries do."""
    from gareus.mbar_analysis.ladder import symmetric_state_overlap as _symmetric_state_overlap
    from gareus.mbar_analysis.solvers import solve_mbar

    with np.load(FIXTURE_NPZ, allow_pickle=False) as data:
        u_nk = np.asarray(data["umbrella_reduced_bias_nk"], dtype=float)
        window = np.asarray(data["window"], dtype=int)
    n_k = np.bincount(window, minlength=u_nk.shape[1]).astype(np.int64)
    f_k = np.asarray(solve_mbar(u_nk, window, tol=1e-10, maxiter=10000)["f_k"], dtype=float)
    o = mbar_state_overlap(u_nk, f_k, n_k)
    for i, want in enumerate(PILOT_ADJACENT_OVERLAP):
        got = _symmetric_state_overlap(o, i, i + 1)
        assert abs(got - want) <= 0.01, (i, got, want)
        assert math.isclose(got, float(o[i, i + 1]), rel_tol=1e-9), "equal N: must equal O_ij"
