"""3.1: the edge-metric record is re-graded after the union overlap is written onto the edges.

The collector grades edges pre-union (two-state BAR, bootstrap q90). A scheduled epoch's union
solve later writes ``mbar_overlap``, which the weak predicate already prefers; the record's
counts, per-edge ``below_threshold``, warnings and ``components`` (read by 3.3 R1 as the
structural-gap set) must follow it, or R1 bridges a split the union already joins.
"""
from __future__ import annotations

import copy
import json

import gareus.adaptive_production as ap
from gareus.adaptive import edge_metric as em
from gareus.adaptive_production import AdaptiveDecisionPolicy, collect_epoch_diagnostics

from test_edge_metric_two_state_mbar import PAIRWISE, _flat_epoch


def _edge(payload, i, j):
    return next(e for e in payload["edges"] if sorted((e["state_i"], e["state_j"])) == [i, j])


def test_union_joining_a_pre_union_gap_merges_the_components(tmp_path):
    reg, epoch = _flat_epoch(tmp_path)
    diag = collect_epoch_diagnostics(epoch, reg, PAIRWISE)
    assert diag["edge_metric"]["components"]["n_components"] == 2          # 0.40 -> 0.70 gap
    gap = _edge(diag, 1, 2)
    assert gap["pairwise_mbar"]["below_threshold"] is True

    ap._apply_union_edge_overlap(diag, {(1, 2): 0.40}, PAIRWISE)

    rec = diag["edge_metric"]
    assert rec["stage"] == "post_union"
    assert rec["components"]["n_components"] == 1
    assert rec["pre_union"]["components"]["n_components"] == 2
    assert gap["pairwise_mbar"]["below_threshold"] is False
    assert "low_pairwise_mbar_overlap" not in gap["warnings"]
    assert not any("components" in w for w in rec["warnings"])


def test_union_below_threshold_marks_a_pre_union_ok_edge_weak(tmp_path):
    reg, epoch = _flat_epoch(tmp_path)
    diag = collect_epoch_diagnostics(epoch, reg, PAIRWISE)
    near = _edge(diag, 0, 1)
    assert near["pairwise_mbar"]["below_threshold"] is False
    n_weak = diag["edge_metric"]["n_weak"]
    n_below = diag["edge_metric"]["n_below_threshold"]

    ap._apply_union_edge_overlap(diag, {(0, 1): 0.05}, PAIRWISE)

    rec = diag["edge_metric"]
    assert near["pairwise_mbar"]["below_threshold"] is True
    assert "low_pairwise_mbar_overlap" in near["warnings"]
    assert rec["n_weak"] == n_weak + 1
    assert rec["n_below_threshold"] == n_below + 1
    # 0 and 1 still meet through the CV1-only state's spanning edges: no new split
    assert rec["components"]["n_components"] == 2


def test_refresh_is_idempotent(tmp_path):
    reg, epoch = _flat_epoch(tmp_path)
    diag = collect_epoch_diagnostics(epoch, reg, PAIRWISE)
    ap._apply_union_edge_overlap(diag, {(1, 2): 0.40}, PAIRWISE)
    once = copy.deepcopy(diag)
    em.refresh_edge_metric_after_union(diag, PAIRWISE)
    assert json.dumps(diag, sort_keys=True) == json.dumps(once, sort_keys=True)


def test_marginal_metric_payload_is_untouched(tmp_path):
    reg, epoch = _flat_epoch(tmp_path)
    policy = AdaptiveDecisionPolicy()
    diag = collect_epoch_diagnostics(epoch, reg, policy)
    before = copy.deepcopy(diag)
    em.refresh_edge_metric_after_union(diag, policy)
    assert diag == before
    assert "edge_metric" not in diag


def test_error_record_is_left_as_is():
    diag = {"edges": [], "edge_metric": {"status": "error", "error": "x", "stage": "pre_union"}}
    before = copy.deepcopy(diag)
    em.refresh_edge_metric_after_union(diag, PAIRWISE)
    assert diag == before
