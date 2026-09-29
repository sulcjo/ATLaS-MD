"""Spec 3.7: the per-phase CV2-resolution table (gareus.adaptive.cv2_resolution_summary)."""
from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from gareus.adaptive import cv2_resolution as cr
from gareus.adaptive import cv2_resolution_rules as rules
from gareus.adaptive import cv2_resolution_summary as crs

T = 300.0


def _state(sid, c1=0.5, k1=300.0, c2=0.2, k2=1.2, lam=0.0, var2=0.25, mean2=0.1, paired=True):
    row = {"state_id": sid, "sample_count": 1000, "epoch_window": sid, "secondary_mean": mean2}
    if paired:
        row["paired_cv"] = {"n_pairs": 1000, "n_rows": 1000,
                            "cv1": {"mean": c1 + 0.01, "var": 0.001},
                            "cv2": {"mean": mean2, "var": var2, "skewness": 0.0, "kurtosis_excess": -1.2},
                            "restraint": {"primary_center": c1, "primary_k": k1, "secondary_center": c2,
                                          "secondary_k": k2, "gamd_lambda": lam,
                                          "cv1_restrained": k1 > 0, "cv2_restrained": k2 > 0}}
    return row


def _pm(status="ok", overlap=0.3, lo=0.28, hi=0.32, pattern="same", kind="neighbour", reason=None):
    return {"status": status, "reason": reason, "overlap": overlap if status == "ok" else None,
            "overlap_lower": lo if status == "ok" else None, "overlap_upper": hi if status == "ok" else None,
            "pattern_pair": pattern, "graph_kind": kind, "n_eff": [300.0, 250.0]}


def _edge(i, j, etype="primary_chain", pm=None, overlap=0.6, joint=0.4, union=None):
    e = {"state_i": i, "state_j": j, "edge_type": etype, "overlap": overlap, "overlap_joint_2d": joint,
         "overlap_joint_2d_reason": None, "mbar_overlap": union}
    if pm is not None:
        e["pairwise_mbar"] = pm
    return e


def _payload(edges=None, edge_metric=True, states=None):
    p = {"schema_version": "adaptive_epoch_diagnostics_v2_segmented",
         "states": states if states is not None else [_state(0), _state(1, c2=1.0), _state(2, k2=0.0)],
         "edges": edges if edges is not None else [_edge(0, 1, pm=_pm())]}
    if edge_metric:
        p["edge_metric"] = {"metric": "pairwise-mbar", "status": "ok", "stage": "pre_union", "threshold": 0.15,
                            "temperature_k": T, "n_weak": 0, "n_unmeasured": 0,
                            "components": {"n_components": 1}}
    return p


def _r3(sid, decision="flagged", transitions=5, lower_bound=None, trapped=True, sd=0.5, sw=0.71):
    m = {"cv2_sd": sd, "sigma_w2": sw, "sd_over_sigma_w": sd / sw, "sarle_bimodality": 0.6,
         "mixture": {"components": [{"mean": -0.4, "sd": 0.2, "weight": 0.1, "accepted": False},
                                    {"mean": 0.2, "sd": 0.3, "weight": 0.6, "accepted": True},
                                    {"mean": 1.1, "sd": 0.1, "weight": 0.3, "accepted": True}]},
         "modes": [{"mean": 0.2}, {"mean": 1.1}], "depth": {"depth_kT": 1.5},
         "trapped_or_orthogonal": trapped, "transitions_state_series": 370}
    if lower_bound is None:
        m["transitions"] = transitions
    else:
        m["transitions"] = None
        m["transitions_lower_bound"] = lower_bound
    return {"rule": "R3", "kind": "state", "state_ids": [sid], "decision": decision, "reason": "why",
            "refusal": None, "metrics": m}


def _report(cands=None, apply=None):
    rep = {"schema_version": "cv2_resolution_report_v1", "status": "ok", "epoch": 2, "stage": "numbered_epoch",
           "settings": {"temperature_k": T}, "rules": {"R1": {"status": "ok"}, "R3": {"status": "ok"}},
           "candidates": cands if cands is not None else [_r3(0)], "budget": {"mode": "reserve"},
           "summary": {"n_proposed": 0, "n_blocking": 0, "n_trapped_or_orthogonal": 1}}
    if apply is not None:
        rep["apply"] = apply
    return rep


# ---- per state -----------------------------------------------------------------------------

def test_sigma_w_and_confinement_match_the_3p3_r3_metrics():
    view = cr.StateView(0, 0.5, 300.0, 0.2, 1.2, 0.0, var2=0.25, mean2=0.1, n_pairs=1000)
    settings = cr.ResolutionSettings.from_policy(object(), temperature_k=T, k1_min=0.0, k2_min=0.0, k2_max=1000.0)
    ref = rules._r3_metrics(view, settings)
    s = crs.build_summary(_payload(), None, label="epoch_002")
    row = next(x for x in s["states"] if x["state_id"] == 0)
    assert row["sigma_w2"] == pytest.approx(ref["sigma_w2"], rel=1e-6)
    assert row["cv2_sd"] == pytest.approx(ref["cv2_sd"], rel=1e-12)
    assert row["confinement_ratio"] == pytest.approx(ref["sd_over_sigma_w"], rel=1e-6)


def test_unrestrained_cv2_has_no_sigma_w_or_ratio_and_json_stays_finite():
    s = crs.build_summary(_payload(), None, label="e")
    row = next(x for x in s["states"] if x["state_id"] == 2)
    assert row["cv2_restrained"] is False and row["sigma_w2"] is None and row["confinement_ratio"] is None
    json.dumps(s, allow_nan=False)


def test_r3_fields_replica_estimator_and_accepted_modes():
    s = crs.build_summary(_payload(), _report(), label="e")
    row = next(x for x in s["states"] if x["state_id"] == 0)
    assert row["r3_evaluated"] and row["r3_decision"] == "flagged" and row["trapped_or_orthogonal"] is True
    assert row["transitions"] == 5 and row["transitions_estimator"] == "replica"
    assert row["n_modes"] == 2 and [m["mean"] for m in row["modes"]] == [0.2, 1.1]
    assert row["r3_mode_pair"] == [0.2, 1.1] and row["depth_kT"] == 1.5 and row["sarle_bimodality"] == 0.6


def test_r3_lower_bound_estimator_and_the_two_not_evaluated_reasons():
    s = crs.build_summary(_payload(), _report([_r3(0, lower_bound=12)]), label="e")
    rows = {x["state_id"]: x for x in s["states"]}
    assert rows[0]["transitions"] == 12 and rows[0]["transitions_estimator"] == "state_series_lower_bound"
    assert rows[1]["r3_evaluated"] is False and "not an R3 candidate" in rows[1]["r3_reason"]
    assert rows[1]["transitions_estimator"] == "none" and rows[1]["trapped_or_orthogonal"] is None
    s2 = crs.build_summary(_payload(), None, label="e")
    assert "no 3.3 report" in s2["states"][0]["r3_reason"]


def test_old_payload_takes_restraints_from_the_registry():
    reg = {0: {"primary_center": "0.4", "primary_k": "250", "secondary_center": "0.3", "secondary_k": "0",
               "gamd_lambda": "0.5"}}
    s = crs.build_summary(_payload(states=[_state(0, paired=False)], edge_metric=False), None, label="e",
                          registry_rows=reg, temperature_k=T)
    row = s["states"][0]
    assert row["restraint_source"] == "registry" and row["c1"] == 0.4 and row["lambda"] == 0.5
    assert row["cv2_restrained"] is False and row["cv2_sd"] is None and row["cv2_mean"] == 0.1


# ---- per edge ------------------------------------------------------------------------------

def test_weak_is_the_edge_metric_predicate_not_below_threshold():
    edges = [_edge(0, 1, "primary_chain", _pm(overlap=0.1, lo=0.09, hi=0.12)),        # weak
             _edge(0, 2, "neighbour", _pm(overlap=0.1, lo=0.09, hi=0.12)),            # below, never weak
             _edge(1, 2, "primary_chain", _pm(overlap=0.1, lo=0.09, hi=0.12, pattern="cross")),
             _edge(0, 3, "secondary_chain", _pm(status="unmeasured", reason="n_eff")),
             _edge(1, 3, "primary_chain", _pm(status="unmeasured"), union=0.4),        # union measures it
             _edge(0, 4, "rung", None)]
    s = crs.build_summary(_payload(edges=edges), None, label="e")
    e = s["edges"]
    assert [x["weak"] for x in e[:3]] == [True, False, False]
    assert [x["below_threshold"] for x in e[:3]] == [True, True, True]
    assert e[3]["measured"] is False and e[3]["pairwise_reason"] == "n_eff"
    assert e[4]["measured"] is True and e[4]["graded_space"] == "union_mbar"
    assert e[5]["graded"] is False and e[5]["weak"] is None
    c = s["counts"]
    assert (c["n_weak"], c["n_unmeasured"], c["n_below_threshold"], c["n_graded_edges"]) == (1, 1, 3, 5)


def test_edge_space_stamps_and_marginal_joint_values():
    e = crs.build_summary(_payload(), None, label="e")["edges"][0]
    assert (e["marginal_space"], e["joint_space"], e["pairwise_space"]) == ("cv1_marginal", "cv1_cv2_joint",
                                                                            "two_state_mbar")
    assert e["marginal_overlap"] == 0.6 and e["joint_2d_overlap"] == 0.4 and e["pairwise_n_eff_min"] == 250.0
    assert e["graded_space"] == "two_state_mbar"


def test_no_edge_metric_grades_nothing():
    s = crs.build_summary(_payload(edges=[_edge(0, 1)], edge_metric=False), None, label="e", temperature_k=T)
    assert s["edge_metric"] is None and s["counts"]["n_graded_edges"] == 0
    assert s["counts"]["n_components"] is None and s["edges"][0]["weak"] is None


def test_budget_refusals_counted_from_candidates_and_apply():
    cands = [_r3(0), {**_r3(1, trapped=False), "decision": "refused", "refusal": "no_reserve"}]
    rep = _report(cands, apply={"refused": [{"reason": "max_replicas_budget"}, {"reason": "duplicate"}]})
    c = crs.build_summary(_payload(), rep, label="e")["counts"]
    assert c["n_refused_budget"] == 2 and c["n_trapped_or_orthogonal"] == 1 and c["report_status"] == "ok"
    assert crs.build_summary(_payload(), None, label="e")["counts"]["n_refused_budget"] is None


def test_temperature_resolution_order():
    p = _payload()
    assert crs.resolve_temperature(p, _report(), 310.0) == (310.0, "override")
    assert crs.resolve_temperature(p, _report()) == (T, "edge_metric")
    q = _payload(edge_metric=False)
    assert crs.resolve_temperature(q, _report()) == (T, "cv2_resolution_report")
    assert crs.resolve_temperature(q, None) == (None, None)


# ---- I/O, CLI, hook ------------------------------------------------------------------------

def _campaign(tmp_path: Path) -> Path:
    ap = tmp_path / "adaptive_production"
    (ap / "epoch_002" / "baseline").mkdir(parents=True)
    (ap / "epoch_002" / crs.EPOCH_DIAGNOSTICS).write_text(json.dumps(_payload()))
    (ap / "epoch_002" / "baseline" / crs.EPOCH_DIAGNOSTICS).write_text(json.dumps(_payload()))
    (ap / "epoch_002" / crs.REPORT_NAME).write_text(json.dumps(_report()))
    (ap / crs.FINAL_DIAGNOSTICS).write_text(json.dumps(_payload()))
    (ap / "state_registry.csv").write_text("state_id,primary_center\n0,0.5\n")
    return ap


def test_discover_uses_phase_level_payloads_and_final_combined(tmp_path):
    ap = _campaign(tmp_path)
    found = crs.discover_phases(ap)
    assert [(lab, Path(d).parent.name) for lab, d, _r in found] == [("epoch_002", "epoch_002"),
                                                                     ("final_combined", "adaptive_production")]
    assert found[0][2] is not None and found[1][2] is None


def test_cli_default_paths_and_csvs(tmp_path):
    ap = _campaign(tmp_path)
    assert crs.main([str(ap)]) == 0
    epoch_json = ap / "epoch_002" / crs.SUMMARY_NAME
    final_json = ap / "cv2_resolution_summary_final_combined.json"
    assert epoch_json.is_file() and final_json.is_file()
    s = json.loads(epoch_json.read_text())
    assert s["schema_version"] == crs.SCHEMA_VERSION and s["sources"]["report"].endswith(crs.REPORT_NAME)
    with open(ap / "epoch_002" / "cv2_resolution_summary_states.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3 and rows[0]["transitions_estimator"] == "replica" and rows[0]["mode_means"] == "0.2;1.1"
    assert (ap / "cv2_resolution_summary_final_combined_edges.csv").is_file()


def test_cli_explicit_inputs_and_out_dir(tmp_path):
    ap = _campaign(tmp_path)
    out = tmp_path / "out"
    rc = crs.main([str(ap), "--diagnostics", str(ap / crs.FINAL_DIAGNOSTICS), "--report",
                   str(ap / "epoch_002" / crs.REPORT_NAME), "--label", "replay", "--out", str(out)])
    assert rc == 0 and (out / "replay" / crs.SUMMARY_NAME).is_file()
    assert not (ap / "cv2_resolution_summary_final_combined.json").exists()


def test_write_epoch_summary_writes_beside_the_report_and_never_raises(tmp_path, capsys):
    ap = _campaign(tmp_path)
    path = crs.write_epoch_summary(ap / "epoch_002", _payload())
    assert path == ap / "epoch_002" / crs.SUMMARY_NAME and path.is_file()
    assert json.loads(path.read_text())["counts"]["n_trapped_or_orthogonal"] == 1
    assert crs.write_epoch_summary(ap / "epoch_002", {"states": [{"state_id": "x"}]}) is None
    assert "WARNING" in capsys.readouterr().out


def test_driver_hook_only_inside_the_cv2_resolution_flag_block():
    src = (Path(__file__).resolve().parents[1] / "gareus" / "adaptive_production.py").read_text()
    tree = ast.parse(src)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "cv2_resolution" in ast.unparse(node.test):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and ast.unparse(sub.func) == "write_epoch_summary":
                    hits.append(ast.unparse(node.test))
    assert hits == ["bool(policy.cv2_resolution)"]
    assert src.count("write_epoch_summary(") == 1


def test_hook_writes_an_in_memory_payload_holding_numpy_scalars(tmp_path):
    ap = _campaign(tmp_path)
    payload = _payload()
    payload["states"][0]["sample_count"] = np.int64(1000)
    payload["states"][0]["paired_cv"]["n_pairs"] = np.int64(1000)
    payload["states"][0]["paired_cv"]["cv2"]["var"] = np.float64(0.25)
    payload["edges"][0]["pairwise_mbar"]["n_eff"] = np.array([300.0, 250.0])
    payload["edge_metric"]["components"]["n_components"] = np.int64(1)
    path = crs.write_epoch_summary(ap / "epoch_002", payload)
    assert path is not None
    s = json.loads(path.read_text())
    assert s["states"][0]["sample_count"] == 1000 and s["counts"]["n_components"] == 1
