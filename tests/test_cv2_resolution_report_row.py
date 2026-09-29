"""Spec 3.7: gareus_report's "CV2 resolution" row (gareus.adaptive.cv2_resolution_grade)."""
from __future__ import annotations

import json

import gareus_report as gr
from gareus.adaptive import cv2_resolution_grade as g

from test_gareus_report import _good_summary


def _counts(**kw):
    c = {"n_states": 236, "n_cv2_restrained": 172, "n_r3_evaluated": 39, "n_trapped_or_orthogonal": 0,
         "n_graded_edges": 276, "n_weak": 0, "n_unmeasured": 0, "n_components": 1,
         "edge_metric_status": "ok", "report_status": "ok", "n_refused_budget": 0}
    c.update(kw)
    return {"label": "epoch_002", "counts": c}


def _verdict(**kw):
    s = _good_summary()
    s["cv2_resolution"] = _counts(**kw)
    return gr.build_health_verdict(s)


def _cv2_row(v):
    return next(c for c in v["checks"] if c["name"] == g.NAME)


def test_no_source_means_no_row_and_an_unchanged_verdict():
    base = gr.build_health_verdict(_good_summary())
    assert all(c["name"] != g.NAME for c in base["checks"])
    assert g.check_cv2_resolution(_good_summary())["status"] == "na"
    s = _good_summary()
    s["production_dir"] = "/nonexistent/adaptive_production"
    assert gr.build_health_verdict(s) == base


def test_row_is_a_strict_superset_of_the_existing_checks():
    base = gr.build_health_verdict(_good_summary())
    v = _verdict()
    assert v["checks"][:-1] == base["checks"] and v["checks"][-1]["name"] == g.NAME
    assert v["overall"] == base["overall"] == "PASS"


def test_pass():
    row = _cv2_row(_verdict())
    assert row["status"] == "pass" and "276 graded edges, 0 weak" in row["detail"] and g.RULE_TEXT in row["detail"]


def test_cv1_only_is_na():
    row = _cv2_row(_verdict(n_cv2_restrained=0, n_graded_edges=0, report_status=None))
    assert row["status"] == "na" and "CV1-only" in row["detail"]


def test_caution_triggers():
    for kw, text in (({"n_trapped_or_orthogonal": 3}, "3 trapped_or_orthogonal"),
                     ({"n_unmeasured": 2}, "2 unmeasured"), ({"n_refused_budget": 1}, "refused for budget"),
                     ({"edge_metric_status": "error"}, "edge metric error"),
                     ({"report_status": "error"}, "3.3 report error")):
        v = _verdict(**kw)
        row = _cv2_row(v)
        assert row["status"] == "caution" and text in row["detail"], kw
        assert v["overall"] == "CAUTION"


def test_fail_triggers():
    for kw in ({"n_weak": 1}, {"n_components": 2}):
        v = _verdict(**kw)
        assert _cv2_row(v)["status"] == "fail" and v["overall"] == "FAIL"


def test_components_not_failed_twice_when_overlap_connectivity_already_fails():
    row = g.check_cv2_resolution({"cv2_resolution": _counts(n_components=3)}, connectivity_failed=True)
    assert row["status"] == "caution" and "already FAILed by Overlap connectivity" in row["detail"]
    row = g.check_cv2_resolution({"cv2_resolution": _counts(n_components=3, n_weak=1)}, connectivity_failed=True)
    assert row["status"] == "fail"


def test_discovery_reads_only_the_final_combined_summary(tmp_path):
    ap = tmp_path / "adaptive_production"
    for name in ("epoch_001", "final"):
        (ap / name).mkdir(parents=True)
        (ap / name / g.SUMMARY_NAME).write_text(json.dumps({"label": name, "counts": _counts(n_weak=4)["counts"]}))
    s = _good_summary()
    s["production_dir"] = str(ap)
    assert g.find_summary(ap) is None                      # per-epoch tables never grade a union PMF
    assert gr.build_health_verdict(s) == gr.build_health_verdict(_good_summary())
    (ap / g.FINAL_NAME).write_text(json.dumps({"label": "final_combined", "counts": _counts()["counts"]}))
    assert g.find_summary(ap) == ap / g.FINAL_NAME
    row = _cv2_row(gr.build_health_verdict(s))
    assert row["status"] == "pass" and row["detail"].startswith("[final_combined]")
    s["cv2_resolution"] = _counts(n_weak=1)                # an attached block wins over the file
    assert _cv2_row(gr.build_health_verdict(s))["status"] == "fail"


def test_unreadable_summary_file_grades_na(tmp_path):
    (tmp_path / g.FINAL_NAME).write_text("{not json")
    s = _good_summary()
    s["production_dir"] = str(tmp_path)
    row = _cv2_row(gr.build_health_verdict(s))
    assert row["status"] == "na" and "unreadable" in row["detail"]


def test_row_renders_in_the_markdown_table():
    lines = gr.render_verdict_md(_verdict(n_trapped_or_orthogonal=3), [])
    assert any(line.startswith(f"| {g.NAME} | ⚠ caution |") for line in lines)
