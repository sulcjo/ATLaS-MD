"""P7b and the bugs that ride with it (spec 2026-09-29 adaptive-CV2, Sections 3.0 P7b, 3.1, 13).

* A: ``build_geometry_edges`` chains only true neighbours (spec-T3 c9 artefacts, T2 wrap-around).
* B: the 3.1 appended graph edges never block ``retire_converged``.
* C: a bridge's reason quotes the metric that made its edge weak.
* D: ``layout_plan.json`` v2 and the consumers switched to the P7a rule behind a gate.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from gareus.adaptive_production import (AdaptiveDecisionPolicy, WindowStateRegistry, _edge_is_measured_weak,
                                        _restraint_pattern, active_graph_connected, build_geometry_edges,
                                        propose_actions_from_diagnostics)

# chignolin_9's lambda = 0 layout (state_registry.json, read 2026-09-29): state id, c1, k1, c2, k2.
# Placeholders: CV1-only rows carry c2 = -0.2517 (k2 = 0), CV2-only windows c1 = 0.5021 (k1 = 0).
C9_LAMBDA0 = [
    (0, 0.5021, 0.0, -0.2517, 0.0), (4, 0.5771, 349.7951, -0.2517, 0.0), (8, 0.0697, 519.7073, -0.2517, 0.0),
    (12, 0.1118, 255.4299, -0.2517, 0.0), (16, 0.1767, 293.1075, -0.2517, 0.0),
    (20, 0.2416, 270.0115, -0.2517, 0.0), (24, 0.3063, 325.6154, -0.2517, 0.0),
    (28, 0.3709, 321.6274, -0.2517, 0.0), (32, 0.4359, 299.4758, -0.2517, 0.0),
    (36, 0.5005, 256.489, -0.2517, 0.0), (40, 0.6301, 373.334, -0.2517, 0.0),
    (44, 0.6949, 272.8659, -0.2517, 0.0), (48, 0.7598, 341.2721, -0.2517, 0.0),
    (52, 0.8244, 246.672, -0.2517, 0.0), (56, 0.8892, 186.2469, -0.2517, 0.0),
    (60, 0.9345, 331.8321, -0.2517, 0.0), (64, 0.5021, 0.0, -1.8513, 1.1795),
    (68, 0.5021, 0.0, -0.7849, 1.1795), (72, 0.5021, 0.0, 0.2815, 1.1795), (76, 0.5021, 0.0, 1.3479, 1.1795),
    (80, 0.0697, 519.7073, -1.8513, 1.1795), (84, 0.9345, 331.8321, 1.3479, 1.1795),
    (88, 0.6301, 373.334, 0.2815, 1.1795), (92, 0.3709, 321.6274, -0.7849, 1.1795),
    (96, 0.3063, 325.6154, -0.7849, 1.1795), (100, 0.6949, 272.8659, 0.2815, 1.1795),
    (104, 0.8892, 186.2469, 1.3479, 1.1795), (108, 0.1118, 255.4299, -1.8513, 1.1795),
    (112, 0.4359, 299.4758, -0.7849, 1.1795), (116, 0.5771, 349.7951, 0.2815, 1.1795),
    (120, 0.2416, 270.0115, -0.7849, 1.1795), (124, 0.7598, 341.2721, 0.2815, 1.1795),
    (128, 0.1767, 293.1075, -1.8513, 1.1795), (132, 0.8244, 246.672, 1.3479, 1.1795),
    (136, 0.5005, 256.489, 0.2815, 1.1795), (140, 0.5005, 256.489, -0.7849, 1.1795),
    (144, 0.1767, 293.1075, -0.7849, 1.1795), (148, 0.8244, 246.672, 0.2815, 1.1795),
    (152, 0.2416, 270.0115, -1.8513, 1.1795), (156, 0.7598, 341.2721, 1.3479, 1.1795),
    (160, 0.4359, 299.4758, 0.2815, 1.1795), (164, 0.5771, 349.7951, -0.7849, 1.1795),
    (168, 0.1118, 255.4299, -0.7849, 1.1795), (172, 0.8892, 186.2469, 0.2815, 1.1795),
    (176, 0.3063, 325.6154, -1.8513, 1.1795), (180, 0.6949, 272.8659, 1.3479, 1.1795),
    (184, 0.3709, 321.6274, 0.2815, 1.1795), (188, 0.6301, 373.334, -0.7849, 1.1795),
    (192, 0.0697, 519.7073, -0.7849, 1.1795), (196, 0.9345, 331.8321, 0.2815, 1.1795),
    (200, 0.6301, 373.334, 1.3479, 1.1795), (204, 0.3709, 321.6274, -1.8513, 1.1795),
    (208, 0.3063, 325.6154, 0.2815, 1.1795), (212, 0.6949, 272.8659, -0.7849, 1.1795),
    (216, 0.4359, 299.4758, -1.8513, 1.1795), (220, 0.5771, 349.7951, 1.3479, 1.1795),
    (224, 0.2416, 270.0115, 0.2815, 1.1795), (228, 0.7598, 341.2721, -0.7849, 1.1795),
    (232, 0.5005, 256.489, -1.8513, 1.1795),
]
C9_LAMBDAS = (0.0, 0.2335, 0.6387, 1.0)
C9_REGISTRY = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_9/"
                   "adaptive_production/state_registry.json")


def _c9_registry(lambdas=(0.0,)):
    """chignolin_9's layout; state ids = c9's for lambda = 0 (+1, +2, +3 on the upper rungs)."""
    reg = WindowStateRegistry()
    for sid, c1, k1, c2, k2 in C9_LAMBDA0:
        for r, lam in enumerate(lambdas):
            st = reg.add_state(primary_center=c1, primary_k=k1, secondary_center=c2, secondary_k=k2,
                               gamd_lambda=lam, epoch=0, source="c9")
            _renumber(reg, st, sid + r)
    return reg


def _renumber(reg, st, new_id):
    old = st.state_id
    if old == new_id:
        return
    del reg._states[old]
    st.state_id = new_id
    reg._states[new_id] = st
    reg._next_state_id = max(reg._next_state_id, new_id + 1)


def _pattern(st):
    return _restraint_pattern(st.primary_k, st.secondary_k, st.secondary_center)


def _assert_true_neighbours_only(reg, edges):
    """Every chain edge joins adjacent states of one row/column; cross-pattern = pattern_link."""
    states = {s.state_id: s for s in reg.active_states() if abs(float(s.gamd_lambda or 0.0)) < 1e-9}
    r6 = lambda v: round(float(v), 6)  # noqa: E731
    for a, b, t, _d in edges:
        if t == "rung":
            continue
        sa, sb = states[a], states[b]
        if _pattern(sa) != _pattern(sb):
            assert t == "pattern_link", (a, b, t)
            continue
        assert t in ("primary_chain", "secondary_chain", "nearest_2d"), (a, b, t)
        if t == "nearest_2d":
            continue
        axis = 0 if t == "primary_chain" else 1
        c = (lambda s: s.primary_center) if axis == 0 else (lambda s: s.secondary_center)
        other = (lambda s: s.secondary_center) if axis == 0 else (lambda s: s.primary_center)
        if _pattern(sa)[1 - axis]:
            assert r6(other(sa)) == r6(other(sb)), (a, b, t, "different row/column")
        lo, hi = sorted((c(sa), c(sb)))
        between = [s.state_id for s in states.values() if _pattern(s) == _pattern(sa) and lo < c(s) < hi
                   and (not _pattern(sa)[1 - axis] or r6(other(s)) == r6(other(sa)))]
        assert not between, (a, b, t, "skips", between)


def _line_registry(n=8, spacing=0.04, k=200.0):
    reg = WindowStateRegistry()
    for i in range(n):
        reg.add_state(primary_center=0.3 + spacing * i, primary_k=k, epoch=0, source="seed")
    return reg


def _diag(reg, edges, sample_count=1000):
    return {"states": [{"state_id": s, "sample_count": sample_count, "gamd_boost_sd_kcal_mol": 0.0}
                       for s in reg.active_state_ids()],
            "edges": edges}


# ---- B: graded-only graph edges do not block retirement --------------------------------------

def test_appended_graph_edges_with_no_marginal_overlap_do_not_block_retirement():
    reg = _line_registry()
    pol = AdaptiveDecisionPolicy(min_active_states=2, edge_metric="pairwise-mbar")
    geom = [{"state_i": a, "state_j": b, "edge_type": t, "overlap": 0.9, "exchange_acceptance": 0.5}
            for a, b, t, _ in build_geometry_edges(reg)]
    base = Counter(a[0] for a in propose_actions_from_diagnostics(reg, _diag(reg, geom), pol))
    assert base["retire"] > 0
    appended = [{"state_i": i, "state_j": i + 2, "edge_type": kind, "overlap": None,
                 "exchange_acceptance": None} for i, kind in ((0, "neighbour"), (3, "spanning"))]
    got = Counter(a[0] for a in propose_actions_from_diagnostics(reg, _diag(reg, geom + appended), pol))
    assert got["retire"] == base["retire"]


def test_an_unmeasured_geometry_edge_still_blocks_retirement_of_its_ends():
    # Deliberate: only the graded-only 3.1 edges are skipped. A geometry edge with no marginal
    # overlap (a state without samples) keeps protecting its endpoints, as before P7b.
    reg = _line_registry()
    pol = AdaptiveDecisionPolicy(min_active_states=2)
    geom = [{"state_i": a, "state_j": b, "edge_type": t, "overlap": (None if (a, b) == (3, 4) else 0.9),
             "exchange_acceptance": 0.5} for a, b, t, _ in build_geometry_edges(reg)]
    retired = {a[1] for a in propose_actions_from_diagnostics(reg, _diag(reg, geom), pol) if a[0] == "retire"}
    assert retired and not retired & {3, 4}


# ---- C: the bridge reason quotes the metric that made the edge weak ---------------------------

def _two_state_registry():
    reg = WindowStateRegistry()
    reg.add_state(primary_center=0.30, primary_k=200.0, epoch=0, source="seed")
    reg.add_state(primary_center=0.50, primary_k=200.0, epoch=0, source="seed")
    return reg


def _pw_edge(**extra):
    edge = {"state_i": 0, "state_j": 1, "edge_type": "primary_chain", "overlap": 0.777,
            "exchange_acceptance": 0.4,
            "pairwise_mbar": {"status": "ok", "overlap": 0.080, "overlap_point": 0.080,
                              "overlap_lower": 0.078, "overlap_upper": 0.082, "pattern_pair": "same"}}
    edge.update(extra)
    return edge


def test_a_pairwise_weak_bridge_quotes_the_pairwise_value_not_the_marginal():
    reg = _two_state_registry()
    pol = AdaptiveDecisionPolicy(edge_metric="pairwise-mbar")
    plan = []
    adds = [a for a in propose_actions_from_diagnostics(reg, _diag(reg, [_pw_edge()]), pol,
                                                         bridge_plan_out=plan) if a[0] == "add"]
    assert adds
    reason = adds[0][3]
    assert "pairwise_mbar_overlap=0.0800" in reason and "q90 0.0820" in reason
    assert "< min_rung_overlap=0.15" in reason
    assert not reason.split(": ", 1)[1].startswith("overlap=0.777")
    assert "cv1_marginal_overlap=0.777" in reason          # still recorded, labelled
    assert plan[0]["graded_overlap"] == 0.080 and plan[0]["graded_overlap_source"] == "pairwise_mbar"
    assert plan[0]["overlap"] == 0.777                      # the old key keeps its meaning


def test_a_union_scored_edge_quotes_the_union_value():
    reg = _two_state_registry()
    pol = AdaptiveDecisionPolicy(edge_metric="pairwise-mbar")
    adds = [a for a in propose_actions_from_diagnostics(reg, _diag(reg, [_pw_edge(mbar_overlap=0.1)]), pol)
            if a[0] == "add"]
    assert "pairwise_mbar_overlap=0.1000 (union)" in adds[0][3]


def test_the_marginal_reason_text_is_unchanged():
    reg = _two_state_registry()
    edge = {"state_i": 0, "state_j": 1, "edge_type": "primary_chain", "overlap": 0.12,
            "exchange_acceptance": 0.4}
    plan = []
    adds = [a for a in propose_actions_from_diagnostics(reg, _diag(reg, [edge]), AdaptiveDecisionPolicy(),
                                                         bridge_plan_out=plan) if a[0] == "add"]
    assert adds[0][3].startswith("weak edge 0-1: overlap=0.12, exchange_acceptance=0.4; bridge 1/")
    assert plan[0]["graded_overlap"] == 0.12 and plan[0]["graded_overlap_source"] == "cv1_marginal"


# ---- A: geometry chains join true neighbours only ---------------------------------------------

def _pairs(edges, etype=None):
    return {(a, b) for a, b, t, _d in edges if etype is None or t == etype}


def test_t2_wraparound_reproducer_no_top_of_column_to_bottom_of_next():
    reg = WindowStateRegistry()
    for c1, c2 in ((0.3, -0.5), (0.3, 0.5), (0.4, -0.5), (0.4, 0.5)):
        reg.add_state(primary_center=c1, primary_k=300.0, secondary_center=c2, secondary_k=30.0, epoch=0)
    edges = build_geometry_edges(reg)
    assert _pairs(edges, "primary_chain") == {(0, 2), (1, 3)}      # CV1 within each CV2 row
    assert _pairs(edges, "secondary_chain") == {(0, 1), (2, 3)}    # CV2 within each CV1 column
    assert (1, 2) not in _pairs(edges) and (0, 3) not in _pairs(edges)
    assert active_graph_connected(reg)


@pytest.mark.parametrize("lambdas", [(0.0,), C9_LAMBDAS])
def test_c9_layout_has_no_artefact_edges_and_stays_connected(lambdas):
    reg = _c9_registry(lambdas)
    edges = build_geometry_edges(reg)
    _assert_true_neighbours_only(reg, edges)
    pairs = _pairs(edges)
    # spec-T3 artefacts: CV2-only three/two rows apart, skip-a-row in one column, diagonal
    for bad in ((64, 76), (68, 76), (160, 216), (136, 232), (184, 204), (164, 200), (152, 224), (188, 200)):
        assert bad not in pairs, bad
    # their true neighbours are there instead
    for good in ((64, 68), (68, 72), (72, 76), (112, 160), (112, 216), (164, 188), (116, 164)):
        assert good in pairs, good
    assert active_graph_connected(reg)
    counts = Counter(t for _a, _b, t, _d in edges)
    assert counts["primary_chain"] == 49 and counts["secondary_chain"] == 27 and counts["pattern_link"] == 3
    assert counts["rung"] == (0 if len(lambdas) == 1 else 59 * (len(lambdas) - 1))


def test_pattern_links_join_each_pattern_once_on_shared_axes():
    edges = build_geometry_edges(_c9_registry())
    links = {(a, b) for a, b, t, _d in edges if t == "pattern_link"}
    by_id = {sid: (c1, k1, c2, k2) for sid, c1, k1, c2, k2 in C9_LAMBDA0}
    kinds = set()
    for a, b in links:
        pa = (by_id[a][1] > 0, by_id[a][3] > 0)
        pb = (by_id[b][1] > 0, by_id[b][3] > 0)
        kinds.add(frozenset((pa, pb)))
        if pa != (False, False) and pb != (False, False):
            shared = [ax for ax in (0, 1) if pa[ax] and pb[ax]]
            assert shared and all(by_id[a][2 * ax] == by_id[b][2 * ax] for ax in shared)   # same centre on it
    assert kinds == {frozenset({(True, False), (True, True)}), frozenset({(False, True), (True, True)}),
                     frozenset({(False, False), (True, True)})}


def _legacy_1d_edges(registry):
    """The pre-P7b builder for a registry with no CV2 centre anywhere (verbatim logic)."""
    from gareus.adaptive_production import _split_rung_groups
    active = registry.active_states()
    policy = AdaptiveDecisionPolicy()
    rung_edges = []
    if any(float(s.gamd_lambda or 0.0) > 0.0 for s in active):
        active, rung_edges = _split_rung_groups(active, policy)
    primary = np.asarray([s.primary_center for s in active], dtype=float)
    ids = [int(s.state_id) for s in active]
    edges = {}
    order = list(np.argsort(primary))
    for left, right in zip(order[:-1], order[1:]):
        a, b = sorted((ids[int(left)], ids[int(right)]))
        edges[(a, b)] = (a, b, "primary_chain", None)
    return rung_edges + list(edges.values())


@pytest.mark.parametrize("seed", range(6))
def test_a_cv1_only_layout_is_byte_identical_to_the_old_chain(seed):
    rng = np.random.default_rng(seed)
    reg = WindowStateRegistry()
    centres = np.round(rng.uniform(0.0, 1.0, 16), 4)
    lambdas = (0.0,) if seed % 2 == 0 else (0.0, 0.3, 1.0)
    for c in centres:
        for lam in lambdas:
            reg.add_state(primary_center=float(c), primary_k=float(rng.uniform(100, 500)), gamd_lambda=lam,
                          epoch=0)
    assert build_geometry_edges(reg) == _legacy_1d_edges(reg)


C7_REGISTRY = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/"
                   "adaptive_production/state_registry.json")


@pytest.mark.skipif(not C7_REGISTRY.exists(), reason="chignolin_7 registry not available")
def test_real_chignolin7_registry_is_byte_identical():
    reg = WindowStateRegistry.load_json(C7_REGISTRY)
    assert build_geometry_edges(reg) == _legacy_1d_edges(reg)


@pytest.mark.skipif(not C9_REGISTRY.exists(), reason="chignolin_9 registry not available")
def test_real_chignolin9_registry_true_neighbours_and_connected():
    reg = WindowStateRegistry.load_json(C9_REGISTRY)
    edges = build_geometry_edges(reg)
    _assert_true_neighbours_only(reg, edges)
    assert active_graph_connected(reg)
    assert (64, 76) not in _pairs(edges) and (164, 200) not in _pairs(edges)


def test_an_off_grid_state_with_no_row_or_column_partner_is_joined_by_its_nearest_neighbour():
    reg = WindowStateRegistry()
    for c1, c2 in ((0.3, -0.5), (0.3, 0.5), (0.4, -0.5), (0.4, 0.5), (0.47, 0.11)):
        reg.add_state(primary_center=c1, primary_k=300.0, secondary_center=c2, secondary_k=30.0, epoch=0)
    edges = build_geometry_edges(reg)
    touching = [(a, b, t) for a, b, t, _d in edges if 4 in (a, b)]
    assert touching and all(t == "nearest_2d" for _a, _b, t in touching)
    assert active_graph_connected(reg)


def test_axis_only_patterns_without_a_2d_state_or_anchor_stay_connected():
    reg = WindowStateRegistry()
    for c in (0.2, 0.3, 0.4):
        reg.add_state(primary_center=c, primary_k=300.0, secondary_center=-0.25, secondary_k=0.0, epoch=0)
    for z in (-1.0, 0.0, 1.0):
        reg.add_state(primary_center=0.3, primary_k=0.0, secondary_center=z, secondary_k=1.2, epoch=0)
    edges = build_geometry_edges(reg)
    assert active_graph_connected(reg)
    assert _pairs(edges, "primary_chain") == {(0, 1), (1, 2)}
    assert _pairs(edges, "secondary_chain") == {(3, 4), (4, 5)}
    assert Counter(t for *_x, t, _d in edges)["pattern_link"] == 1


def test_unrecorded_k_ranks_by_the_fallback_width():
    reg = WindowStateRegistry()
    for c1, c2 in ((0.3, -0.5), (0.3, 0.5), (0.4, -0.5), (0.4, 0.5), (0.47, 0.11)):
        st = reg.add_state(primary_center=c1, primary_k=300.0, secondary_center=c2, secondary_k=30.0, epoch=0)
        st.primary_k, st.secondary_k = None, None          # an old registry: k not recorded
    edges = build_geometry_edges(reg)
    assert _pairs(edges, "primary_chain") == {(0, 2), (1, 3)}
    assert any(4 in (a, b) for a, b, *_r in edges) and active_graph_connected(reg)


def test_pattern_link_is_never_weak_and_gives_no_redundancy_credit():
    edge = {"state_i": 0, "state_j": 1, "edge_type": "pattern_link", "overlap": 0.01, "exchange_acceptance": 0.0,
            "pairwise_mbar": {"status": "ok", "overlap": 0.01, "overlap_upper": 0.01, "pattern_pair": "cross"}}
    for pol in (AdaptiveDecisionPolicy(), AdaptiveDecisionPolicy(edge_metric="pairwise-mbar")):
        assert not _edge_is_measured_weak(edge, pol)
    # an anchor (k1 = 0, a leaf on its link, so not an articulation point) next to a CV1 chain:
    # a CV1-marginal 0.95 across patterns is no redundancy, so the anchor is not retired
    reg = WindowStateRegistry()
    reg.add_state(primary_center=0.3, primary_k=0.0, epoch=0)
    for c in (0.2, 0.3, 0.4, 0.5):
        reg.add_state(primary_center=c, primary_k=300.0, epoch=0)
    geom = build_geometry_edges(reg)
    assert [(a, b) for a, b, t, _d in geom if t == "pattern_link"] == [(0, 2)]    # to the central state
    rows = [{"state_i": a, "state_j": b, "edge_type": t, "exchange_acceptance": 0.5,
             "overlap": 0.95 if t == "pattern_link" else 0.35} for a, b, t, _d in geom]   # ok, not redundant
    acts = propose_actions_from_diagnostics(reg, _diag(reg, rows), AdaptiveDecisionPolicy(min_active_states=1))
    assert not [a for a in acts if a[0] == "retire"]


# ---- D: layout_plan.json schema v2 ---------------------------------------------------------

def _swarm_plan(n1=4, n2=3, n_rungs=2, cap=40, reps=(3,)):
    from gareus.swarm.ladder_design import design_exploration_layout, layout_plan_record, layout_rows
    plan = design_exploration_layout(n1, n2, n_rungs=n_rungs, max_replicas=cap, region_centre_indices=list(reps))
    rows = layout_rows(plan, np.linspace(0.1, 0.9, n1), np.full(n1, 200.0), np.linspace(-1, 1, n2), np.full(n2, 0.7))
    lambdas = list(np.linspace(0.0, 1.0, n_rungs))
    region_of_centre = ["cv1_region_00"] * (n1 - 1) + ["cv1_region_01"]
    inventory = {"regions": [{"region_id": "cv1_region_00"}, {"region_id": "cv1_region_01"}],
                 "region_of_centre": region_of_centre, "unresolved": []}
    rec = layout_plan_record(plan, rows, lambdas, region_inventory=inventory, region_of_centre=region_of_centre)
    return plan, rows, lambdas, rec


def test_layout_plan_v2_lists_every_state_with_region_and_pattern_and_round_trips(tmp_path):
    import json
    from gareus.layout_plan import LAYOUT_PLAN_V2, read_layout_plan
    plan, rows, lambdas, rec = _swarm_plan()
    assert rec["version"] == LAYOUT_PLAN_V2 and rec["schema_version"] == 2
    assert len(rec["states"]) == rec["n_states"] == len(plan["cells"]) * len(lambdas)
    for s in rec["states"]:
        assert {"state_id", "center1", "k1", "center2", "k2", "region", "role", "restrained"} <= set(s)
        assert s["restrained"] == [s["k1"] > 0, s["k2"] > 0]
        assert (s["region"] is None) == (s["cell"][0] is None)
    anchor = rec["states"][0]
    assert anchor["role"] == "unrestrained_anchor" and anchor["region"] is None and anchor["restrained"] == [False, False]
    (tmp_path / "layout_plan.json").write_text(json.dumps(rec))
    raw, states = read_layout_plan(tmp_path / "layout_plan.json")
    assert [(s.state_id, s.role, s.mandatory, s.region, s.restrained) for s in states] == \
        [(r["state_id"], r["role"], r["mandatory"], r["region"], tuple(r["restrained"])) for r in rec["states"]]


def test_a_v1_layout_plan_still_reads_with_region_from_the_inventory(tmp_path):
    import json
    from gareus.layout_plan import read_layout_plan, schema_version
    _plan, _rows, _lambdas, rec = _swarm_plan()
    v1 = dict(rec, version="layout_plan_v1")
    v1.pop("schema_version")
    v1["states"] = [{k: v for k, v in s.items() if k not in ("region", "restrained")} for s in rec["states"]]
    (tmp_path / "layout_plan.json").write_text(json.dumps(v1))
    raw, states = read_layout_plan(tmp_path / "layout_plan.json")
    assert schema_version(raw) == 1
    assert [(s.region, s.restrained, s.role, s.mandatory) for s in states] == \
        [(r["region"], tuple(r["restrained"]), r["role"], r["mandatory"]) for r in rec["states"]]


C9_LAYOUT_PLAN = C9_REGISTRY.parents[1] / "swarm" / "analysis" / "layout_plan.json"


@pytest.mark.skipif(not C9_LAYOUT_PLAN.exists(), reason="chignolin_9 layout plan not available")
def test_the_real_chignolin9_v1_plan_reads():
    from gareus.layout_plan import read_layout_plan, schema_version
    raw, states = read_layout_plan(C9_LAYOUT_PLAN)
    assert schema_version(raw) == 1 and len(states) == 236
    assert states[0].role == "unrestrained_anchor" and states[0].restrained == (False, False)
    assert {s.region for s in states if s.restrained[0]} == {"cv1_region_00"}
    assert all(s.region is None for s in states if not s.restrained[0])


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_both_loaders_take_roles_from_either_schema(tmp_path, version):
    import json
    from gareus.adaptive_production import registry_from_window_csv
    from gareus.swarm.ladder_design import write_ladder_windows_2d_csv
    from gareus.windows import load_explicit_2d_window_csv
    _plan, rows, lambdas, rec = _swarm_plan()
    if version == "v1":
        rec = dict(rec, version="layout_plan_v1", states=[{k: v for k, v in s.items() if k not in ("region", "restrained")}
                                                          for s in rec["states"]])
        rec.pop("schema_version")
    path = write_ladder_windows_2d_csv(tmp_path / "windows_lambda_ladder.csv", rows, lambdas)
    (tmp_path / "layout_plan.json").write_text(json.dumps(rec))

    class Args:
        primary_cv = "nonlocal-contacts"; secondary_cv = "residual-torsion-pc"; contact_k_kcal = None
        secondary_cv_k_kcal = 1.0
    *_rest, meta = load_explicit_2d_window_csv(Args(), path)
    assert meta["mandatory_window_indices"] == rec["mandatory_state_ids"]
    assert meta["state_roles"] == [s["role"] for s in rec["states"]]
    reg = registry_from_window_csv(path, epoch=0, source="t")
    assert [bool(s.metadata.get("mandatory")) for s in reg.all_states()] == [s["mandatory"] for s in rec["states"]]
    if version == "v2":
        assert [s.metadata.get("region") for s in reg.all_states()] == [s["region"] for s in rec["states"]]


# ---- D: consumers on the P7a rule, behind --layout-neighbour-rule ----------------------------

def _c9_windows(lambdas=C9_LAMBDAS):
    """chignolin_9's window table order (centre-major, rungs inner): c1, k1, c2, k2, lambda."""
    rows = [(c1, k1, c2, k2, lam) for _sid, c1, k1, c2, k2 in C9_LAMBDA0 for lam in lambdas]
    return [np.asarray([r[i] for r in rows], dtype=float) for i in range(5)]


class _Args:
    def __init__(self, rule):
        self.layout_neighbour_rule = rule
        self.temperature_k = 300.0
        self.secondary_cv = "residual-torsion-pc"


def test_the_exchange_graph_is_unchanged_under_the_default_rule():
    from gareus.windows import build_explicit_2d_neighbor_edges
    c1, k1, c2, k2, lam = _c9_windows()
    plain = build_explicit_2d_neighbor_edges(c1, c2, args=None)
    assert build_explicit_2d_neighbor_edges(c1, c2, args=_Args("legacy"), k1=k1, k2=k2, lambdas=lam) == plain
    assert {e["edge_type"] for e in plain} <= {"distance_axis", "secondary_axis", "knn_geometry",
                                               "connectivity_bridge"} | {
        "+".join(sorted(t)) for t in (("distance_axis", "knn_geometry"), ("knn_geometry", "secondary_axis"),
                                      ("distance_axis", "knn_geometry", "secondary_axis"),
                                      ("distance_axis", "secondary_axis"))}


def _pattern_of(k1, k2):
    return (bool(k1 > 0), bool(k2 > 0))


def test_the_restraint_width_exchange_graph_on_the_c9_ladder():
    from gareus.adaptive.neighbour_rule import DEFAULT_RADIUS
    from gareus.windows import build_explicit_2d_neighbor_edges
    c1, k1, c2, k2, lam = _c9_windows()
    edges = build_explicit_2d_neighbor_edges(c1, c2, args=_Args("restraint-width"), k1=k1, k2=k2, lambdas=lam)
    n = len(c1)
    types = Counter()
    for e in edges:
        a, b, ts = e["wi"], e["wj"], set(e["edge_type"].split("+"))
        types.update(ts)
        if lam[a] != lam[b]:                       # across rungs: only adjacent rungs of one centre
            assert ts == {"lambda_neighbor"}, e
            assert (c1[a], c2[a], k1[a], k2[a]) == (c1[b], c2[b], k1[b], k2[b])
            assert abs(C9_LAMBDAS.index(lam[a]) - C9_LAMBDAS.index(lam[b])) == 1
            continue
        if _pattern_of(k1[a], k2[a]) != _pattern_of(k1[b], k2[b]):
            assert ts == {"pattern_link"}, e       # no placeholder-coordinate neighbour
            continue
        if ts == {"p7a_neighbor"}:
            assert e["normalized_distance"] <= DEFAULT_RADIUS + 1e-12
    assert types["lambda_neighbor"] == 59 * 3
    assert types["pattern_link"] == 3 * 4           # per rung: CV1-only/2D, CV2-only/2D, anchor
    # connected, and the anchor stack exchanges with someone on every rung
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for e in edges:
        ra, rb = find(e["wi"]), find(e["wj"])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    assert len({find(w) for w in range(n)}) == 1
    anchors = [w for w in range(n) if k1[w] == 0 and k2[w] == 0]
    assert all(any(w in (e["wi"], e["wj"]) and lam[e["wi"]] == lam[e["wj"]] for e in edges) for w in anchors)


def test_the_restraint_width_exchange_graph_falls_back_when_k_is_missing():
    from gareus.windows import build_explicit_2d_neighbor_edges
    c1 = np.array([0.3, 0.3, 0.4, 0.4])
    c2 = np.array([-0.5, 0.5, -0.5, 0.5])
    edges = build_explicit_2d_neighbor_edges(c1, c2, args=_Args("restraint-width"))   # no k, no lambdas
    pairs = {(e["wi"], e["wj"]) for e in edges}
    assert {(0, 2), (1, 3), (0, 1), (2, 3)} <= pairs and all(np.isfinite(e["normalized_distance"]) for e in edges)


def test_an_unknown_rule_is_refused():
    from gareus.windows import build_explicit_2d_neighbor_edges
    with pytest.raises(ValueError):
        build_explicit_2d_neighbor_edges(np.array([0.1, 0.2]), np.array([0.0, 0.0]), args=_Args("nope"))


def test_topup_partners_follow_the_rule_and_p6_negative_k():
    from gareus.adaptive.neighbour_rule import DEFAULT_RADIUS, NeighbourPoint, pair_distance
    from gareus.layout_neighbours import spatial_neighbour_pairs
    c1, k1, c2, k2, lam = _c9_windows()
    legacy = spatial_neighbour_pairs(c1, c2, lam, k1, k2, 300.0)
    assert spatial_neighbour_pairs(c1, c2, lam, k1, k2, 300.0, rule="legacy") == legacy
    p7a = spatial_neighbour_pairs(c1, c2, lam, k1, k2, 300.0, rule="restraint-width")
    assert set(legacy) <= set(p7a)                  # P7a: a superset on c9 (235 vs 108 edges per rung)
    assert len(p7a) == 4 * 235 and len(legacy) == 4 * 108
    for a, b in p7a:
        assert lam[a] == lam[b] and _pattern_of(k1[a], k2[a]) == _pattern_of(k1[b], k2[b])
        pa = NeighbourPoint(c1[a], k1[a], c2[a], k2[a]); pb = NeighbourPoint(c1[b], k1[b], c2[b], k2[b])
        assert pair_distance(pa, pb, 300.0) <= DEFAULT_RADIUS + 1e-12
    # a negative k is unrestrained (P6), as a recorded 0 already was
    neg = spatial_neighbour_pairs([0.1, 0.2, 0.3], [0.0, 0.0, 0.5], [0, 0, 0], [300.0, 300.0, -5.0],
                                  [0.0, 0.0, 1.2], 300.0)
    assert all(2 not in p for p in neg)


def test_topup_layout_neighbours_read_the_campaign_rule(monkeypatch):
    import gareus.adaptive_production as ap
    import gareus.layout_neighbours as ln
    seen = []
    real = ln.spatial_neighbour_pairs
    monkeypatch.setattr(ln, "spatial_neighbour_pairs", lambda *a, **k: (seen.append(k.get("rule")), real(*a, **k))[1])
    reg = _c9_registry()
    for rule in ("legacy", "restraint-width"):
        ap._topup_layout_neighbours(_Args(rule), reg.active_states())
    assert seen == ["legacy", "restraint-width"]


def test_the_rule_is_a_frozen_decision_setting_threaded_into_phase_args(tmp_path):
    import gareus.adaptive_production as ap
    from gareus.cli import parse_args
    args = parse_args(["--seq", "GYDPETGTWG", "--out", str(tmp_path)])
    assert args.layout_neighbour_rule == "legacy"
    args = parse_args(["--seq", "GYDPETGTWG", "--out", str(tmp_path), "--layout-neighbour-rule", "restraint-width"])
    pol = ap.policy_from_args(args)
    assert pol.layout_neighbour_rule == "restraint-width" and "layout_neighbour_rule" in ap.DECISION_SETTINGS_FIELDS
    ap._resolve_decision_settings(tmp_path, AdaptiveDecisionPolicy())            # first job: legacy frozen
    later, rec = ap._resolve_decision_settings(tmp_path, pol)                    # a later job's flag is ignored
    assert later.layout_neighbour_rule == "legacy" and rec["settings"]["layout_neighbour_rule"] == "legacy"
    src = Path(ap.__file__).read_text()
    assert "args.layout_neighbour_rule = str(policy.layout_neighbour_rule)" in src


def test_ladder_overlap_gains_a_cv2_axis_on_true_cv2_neighbours():
    from gareus.mbar_analysis.ladder_overlap import ladder_overlap_by_axis, ladder_overlap_health_checks
    c1, k1, c2, k2, lam = _c9_windows()
    seen = []

    def _ov(a, b):
        seen.append((a, b))
        return 0.38
    out, warnings = ladder_overlap_by_axis(None, lam, c1, pair_overlap=_ov, secondary_centers=c2,
                                           primary_k=k1, secondary_k=k2)
    assert warnings == []
    cv2 = out["cv2_direction"]
    pairs = {tuple(sorted((int(a), int(b)))) for a, b, _v in cv2["pairs"]}
    for a, b in pairs:
        assert lam[a] == lam[b] and _pattern_of(k1[a], k2[a]) == _pattern_of(k1[b], k2[b])
        if k1[a] > 0:
            assert c1[a] == c1[b]                  # one CV1 column
        lo, hi = sorted((c2[a], c2[b]))
        between = [w for w in range(len(c1)) if lam[w] == lam[a] and _pattern_of(k1[w], k2[w]) ==
                   _pattern_of(k1[a], k2[a]) and (k1[a] == 0 or c1[w] == c1[a]) and lo < c2[w] < hi]
        assert not between                         # adjacent rows only
    assert cv2["n_pairs"] == 27 * 4 and cv2["connected"] is True
    names = [c["name"] for c in ladder_overlap_health_checks(out, 0.15)]
    assert "Overlap across CV2" in names and "Connectivity across CV2" in names and len(names) == 6


def test_a_cv1_only_ladder_overlap_report_keeps_its_two_axes():
    from gareus.mbar_analysis.ladder_overlap import ladder_overlap_by_axis, ladder_overlap_health_checks
    lam = np.array([0.0, 0.0, 1.0, 1.0])
    cen = np.array([0.2, 0.3, 0.2, 0.3])
    out, _w = ladder_overlap_by_axis(None, lam, cen, pair_overlap=lambda a, b: 0.3,
                                     secondary_centers=np.full(4, np.nan))
    assert set(out) == {"lambda_direction", "cv1_direction"}
    assert len(ladder_overlap_health_checks(out, 0.15)) == 4
