"""The "CV2 resolution" health row (spec 3.7), graded from a ``cv2_resolution_summary_v1`` table.

Stdlib only and presentation only: it reads the precomputed ``counts`` block that
``gareus.adaptive.cv2_resolution_summary`` wrote and never recomputes a metric, so
``gareus_report`` can call it on an in-process summary dict or a JSON round-trip alike.

Source, first found: ``s["cv2_resolution"]`` (a summary dict a caller attached), else a file
under ``s["production_dir"]``: ``cv2_resolution_summary_final_combined.json`` at the adaptive
root, else the latest phase's ``<phase>/cv2_resolution_summary.json`` (final_extension_NNN >
final > epoch_NNN, by number). No source: no row (``gareus_report`` appends it only then).

Rule (stated in the row text):
  FAIL    -- a same-pattern geometry edge is weak under the pairwise-MBAR metric
             (edge_metric.edge_is_weak_pairwise), or the pre-union spatial graph splits
             into more than one component. The components FAIL is lowered to CAUTION when
             the "Overlap connectivity" row already FAILs, so one split is not graded twice.
  CAUTION -- graded edges left unmeasured (never weak, but not shown to be connected),
             trapped_or_orthogonal windows (3.3 R3: bimodal without transitions -- more CV2
             windows cannot resolve that), resolution actions refused for budget
             (no_reserve, resolution_budget, max_replicas_budget), or an edge-metric /
             3.3-report error.
  PASS    -- none of those. NA -- a CV1-only table with no graded edges and no 3.3 report.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

# Mirror gareus_report's status vocabulary (kept literal so this module imports nothing).
PASS, CAUTION, FAIL, NA = "pass", "caution", "fail", "na"
NAME = "CV2 resolution"
SUMMARY_NAME = "cv2_resolution_summary.json"            # mirrors cv2_resolution_summary
FINAL_NAME = "cv2_resolution_summary_final_combined.json"
RULE_TEXT = ("rule: FAIL if a pairwise-MBAR weak edge or >1 spatial component; CAUTION for "
             "unmeasured edges, trapped_or_orthogonal windows, budget refusals, metric/report errors")
_PHASE_RX = re.compile(r"^(epoch|final_extension)_(\d+)$|^final$")


def _phase_key(name: str) -> Optional[tuple]:
    m = _PHASE_RX.match(name)
    if m is None:
        return None
    if name == "final":
        return (1, 0)
    return (0 if m.group(1) == "epoch" else 2, int(m.group(2)))


def find_summary(production_dir: Any) -> Optional[Path]:
    """The summary file ``gareus_report`` grades for this analysis, or None."""
    if not production_dir:
        return None
    root = Path(str(production_dir))
    if (root / FINAL_NAME).is_file():
        return root / FINAL_NAME
    try:
        found = [(k, d / SUMMARY_NAME) for d in root.iterdir()
                 if d.is_dir() and (k := _phase_key(d.name)) is not None and (d / SUMMARY_NAME).is_file()]
    except OSError:
        return None
    return max(found)[1] if found else None


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
                              ("n_refused_budget", "action(s) refused for budget", cautions)):
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
