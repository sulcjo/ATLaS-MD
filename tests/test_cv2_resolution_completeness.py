"""The CV2-resolution row never PASSes on an incomplete evaluation (verification F1).

The 3.3 report records ``requested`` per rule (R1, R3 always; R2 with top-ups on); the summary
counts which requested rules completed; the grade CAUTIONs anything short of complete.
"""
from __future__ import annotations

from types import SimpleNamespace

from gareus.adaptive import cv2_resolution_grade as g
from gareus.adaptive import cv2_resolution_summary as crs
from gareus.adaptive.cv2_resolution_io import requested_rules

from test_cv2_resolution_summary import _payload, _report


def _rules(r1="ok", r2="unavailable", r3="ok", topups=False, **reasons):
    raw = {"R1": {"status": r1, "reason": reasons.get("why1")}, "R2": {"status": r2, "reason": reasons.get("why2")},
           "R3": {"status": r3, "reason": reasons.get("why3")}}
    return requested_rules(SimpleNamespace(topups_enabled=topups), raw)


def _grade(report, payload=None, sources=None):
    s = crs.build_summary(payload or _payload(), report, label="epoch_002", sources=sources)
    return g.check_cv2_resolution({"cv2_resolution": s}), s["counts"]


def _rep(rules, cands=None):
    rep = _report(cands=cands if cands is not None else [])
    rep["rules"] = rules
    return rep


def test_all_rules_unavailable_is_caution_not_pass():
    rules = _rules("unavailable", "unavailable", "unavailable", topups=True,
                   why1="requires --ap-edge-metric pairwise-mbar")
    row, c = _grade(_rep(rules))
    assert row["status"] == "caution"
    assert c["evaluation_complete"] is False and c["n_rules_requested"] == 3 and c["n_rules_unavailable"] == 3
    assert "R1 unavailable (requires --ap-edge-metric pairwise-mbar)" in row["detail"]


def test_one_rule_error_is_caution_with_the_error():
    rules = _rules(r3="error", why3="boom")
    row, c = _grade(_rep(rules))
    assert row["status"] == "caution" and c["n_rules_error"] == 1 and "R3 error (boom)" in row["detail"]


def test_all_requested_rules_complete_is_pass():
    row, c = _grade(_rep(_rules("ok", "ok", "ok", topups=True)))
    assert row["status"] == "pass" and c["evaluation_complete"] is True and c["n_rules_ok"] == 3


def test_top_ups_off_does_not_request_r2():
    rules = _rules("ok", "unavailable", "ok", topups=False, why2="top-ups off")
    assert rules["R2"]["requested"] is False
    row, c = _grade(_rep(rules))
    assert row["status"] == "pass" and c["n_rules_requested"] == 2
    # R1/R3 still decide: R1 unavailable under the marginal metric is incomplete (user choice b)
    row, _c = _grade(_rep(_rules("unavailable", "unavailable", "ok", why1="requires --ap-edge-metric pairwise-mbar")))
    assert row["status"] == "caution" and "R2" not in row["detail"].split(" -- ")[0]


def test_cv1_only_stays_na():
    payload = _payload(edges=[], edge_metric=False, states=[])
    row, _c = _grade(None, payload=payload)
    assert row["status"] == "na"


def test_cv2_table_without_a_report_is_caution():
    row, c = _grade(None)
    assert row["status"] == "caution" and c["evaluation_metadata"] == "no_report"
    assert "R1-R3 not evaluated" in row["detail"]


def test_legacy_report_and_legacy_summary_are_caution():
    rep = _report(cands=[])                                # rules without ``requested``
    row, c = _grade(rep)
    assert row["status"] == "caution" and c["evaluation_metadata"] == "legacy"
    old = {"label": "final_combined", "counts": {"n_cv2_restrained": 10, "n_graded_edges": 5, "n_weak": 0,
                                                 "n_components": 1, "report_status": "ok"}}
    row = g.check_cv2_resolution({"cv2_resolution": old})
    assert row["status"] == "caution" and "legacy" in row["detail"]


def test_carried_report_never_vouches_for_this_tables_r1():
    rules = _rules("ok", "unavailable", "ok")                # epoch R1 ok
    payload = _payload(edge_metric=False)                  # final-combined table: no pairwise metric
    row, c = _grade(_rep(rules), payload=payload, sources={"report_carried_from": "epoch_002"})
    assert row["status"] == "caution" and "R1 unavailable" in row["detail"]
    row, c = _grade(_rep(rules), payload=_payload(), sources={"report_carried_from": "epoch_002"})
    assert row["status"] == "pass" and c["evaluation_complete"] is True


def test_report_error_is_graded_once():
    rep = {"status": "error", "error": "x", "candidates": [], "summary": {}}
    row, c = _grade(rep)
    assert row["status"] == "caution" and c["evaluation_metadata"] == "report_error"
    assert row["detail"].count("3.3 report error") == 1 and "incomplete evaluation" not in row["detail"]


def test_respring_only_campaign_requests_no_rules():
    payload = _payload()
    payload["policy"] = {"cv2_resolution": False, "cv2_respring": True}
    row, c = _grade(None, payload=payload)
    assert row["status"] == "pass" and c["evaluation_metadata"] == "not_requested"
    assert c["n_rules_requested"] == 0 and c["evaluation_complete"] is True
    payload["policy"]["cv2_resolution"] = True             # on, but no report written: incomplete
    row, c = _grade(None, payload=payload)
    assert row["status"] == "caution" and c["evaluation_metadata"] == "no_report"


def test_carried_r2_r3_are_labelled_as_carried():
    rules = _rules("ok", "ok", "ok", topups=True)
    row, c = _grade(_rep(rules), payload=_payload(), sources={"report_carried_from": "epoch_002"})
    assert row["status"] == "pass" and c["rules_carried"] == ["R2", "R3"]
    assert "R2/R3/budget from epoch_002" in row["detail"]
