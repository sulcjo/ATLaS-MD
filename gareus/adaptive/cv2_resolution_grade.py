"""The "CV2 resolution" health row (spec 3.7), graded from a ``cv2_resolution_summary_v1`` table.

Stdlib only and presentation only: it reads the precomputed ``counts`` block that
``gareus.adaptive.cv2_resolution_summary`` wrote and never recomputes a metric, so
``gareus_report`` can call it on an in-process summary dict or a JSON round-trip alike.

Source: ``s["cv2_resolution"]`` (a summary dict a caller attached), else
``<s["production_dir"]>/cv2_resolution_summary_final_combined.json`` -- the table over the
same pooled data the PMF is computed on. Per-epoch summaries are never auto-discovered: an
epoch-era flag would grade a union PMF (chignolin_9: epoch_002 CAUTION, final-combined PASS).
The file is not checked for freshness; the row prints the table's label. No source: no row
(``gareus_report`` appends it only then).

Rule (stated in the row text):
  FAIL    -- a same-pattern geometry edge is weak under the pairwise-MBAR metric
             (edge_metric.edge_is_weak_pairwise), or the pre-union spatial graph splits
             into more than one component. The components FAIL is lowered to CAUTION when
             the "Overlap connectivity" row already FAILs, so one split is not graded twice.
  CAUTION -- graded edges left unmeasured (never weak, but not shown to be connected),
             trapped_or_orthogonal windows (3.3 R3: bimodal without transitions -- more CV2
             windows cannot resolve that), resolution actions refused for budget
             (no_reserve, resolution_budget, max_replicas_budget), resolution windows refused
             at the spring cap (k2_capped_below_compression: the 4 x parent k2 / cv2_k_max cap
             or the coupling gate holds k2 below the mean-compression floor -- the window the
             rule asked for was not placed), or an edge-metric / 3.3-report error. None of the
             CAUTIONs blocks the convergence gate.
  PASS    -- none of those. NA -- a CV1-only table with no graded edges and no 3.3 report.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

# Mirror gareus_report's status vocabulary (kept literal so this module imports nothing).
PASS, CAUTION, FAIL, NA = "pass", "caution", "fail", "na"
NAME = "CV2 resolution"
SUMMARY_NAME = "cv2_resolution_summary.json"            # mirrors cv2_resolution_summary
FINAL_NAME = "cv2_resolution_summary_final_combined.json"
RULE_TEXT = ("rule: FAIL if a pairwise-MBAR weak edge or >1 spatial component; CAUTION for "
             "unmeasured edges, trapped_or_orthogonal windows, budget refusals, spring cap below the "
             "mean-compression floor, metric/report errors")


def find_summary(production_dir: Any) -> Optional[Path]:
    """The final-combined summary ``gareus_report`` grades for this analysis, or None."""
    if not production_dir:
        return None
    path = Path(str(production_dir)) / FINAL_NAME
    return path if path.is_file() else None


def summary_block(s: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The attached summary, else the discovered file's contents; None when there is no source."""
    blk = (s or {}).get("cv2_resolution")
    if isinstance(blk, dict):
        return blk
    path = find_summary((s or {}).get("production_dir"))
    if path is None:
        return None
    try:
        out = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"label": path.parent.name, "counts": {}, "unreadable": str(path)}
    return out if isinstance(out, dict) else None


def _issues(c: Mapping[str, Any], connectivity_failed: bool) -> tuple:
    fails, cautions = [], []
    comps = c.get("n_components")
    if comps is not None and int(comps) > 1:
        (cautions if connectivity_failed else fails).append(
            f"spatial graph splits into {int(comps)} components"
            + (" (already FAILed by Overlap connectivity)" if connectivity_failed else ""))
    for key, text, bucket in (("n_weak", "weak pairwise-MBAR edge(s)", fails),
                              ("n_unmeasured", "unmeasured graded edge(s)", cautions),
                              ("n_trapped_or_orthogonal", "trapped_or_orthogonal window(s)", cautions),
                              ("n_refused_budget", "action(s) refused for budget", cautions),
                              ("n_refused_spring_cap", "window(s) refused at the spring cap "
                               "(k2_capped_below_compression)", cautions)):
        n = int(c.get(key) or 0)
        if n > 0:
            bucket.append(f"{n} {text}")
    for key, what in (("edge_metric_status", "edge metric"), ("report_status", "3.3 report")):
        if c.get(key) == "error":
            cautions.append(f"{what} error")
    return fails, cautions


def check_cv2_resolution(s: Mapping[str, Any], connectivity_failed: bool = False) -> Dict[str, Any]:
    """The row dict ({name, status, detail[, metric]}); see the module docstring for the rule."""
    blk = summary_block(s)
    if blk is None:
        return {"name": NAME, "status": NA, "detail": "no CV2-resolution summary (cv2_resolution_summary.json)"}
    if blk.get("unreadable"):
        return {"name": NAME, "status": NA, "detail": f"unreadable summary {blk['unreadable']}"}
    c = blk.get("counts") or {}
    label = blk.get("label") or "?"
    carried = (blk.get("sources") or {}).get("report_carried_from")
    if carried:
        label = f"{label}; R3/budget from {carried}"
    graded = int(c.get("n_graded_edges") or 0)
    if not int(c.get("n_cv2_restrained") or 0) and not graded and c.get("report_status") is None:
        return {"name": NAME, "status": NA, "detail": f"[{label}] CV1-only: no CV2-restrained state, "
                                                      "no graded edge, no 3.3 report"}
    fails, cautions = _issues(c, connectivity_failed)
    status = FAIL if fails else (CAUTION if cautions else PASS)
    facts = (f"{graded} graded edges, {int(c.get('n_weak') or 0)} weak, components {c.get('n_components')}; "
             f"{int(c.get('n_r3_evaluated') or 0)} R3-evaluated windows")
    found = "; ".join(fails + cautions)
    detail = f"[{label}] " + (found + " -- " if found else "") + facts + f" ({RULE_TEXT})"
    return {"name": NAME, "status": status, "detail": detail, "metric": int(c.get("n_weak") or 0)}


__all__ = ["NAME", "RULE_TEXT", "check_cv2_resolution", "find_summary", "summary_block"]
