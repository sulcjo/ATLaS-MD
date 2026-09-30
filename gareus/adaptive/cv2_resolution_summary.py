"""Spec 3.7 reporting: one "CV2 resolution" table per phase, from JSONs that already exist.

Pure collector. It reads a phase's diagnostics payload (P4 ``paired_cv`` per state, the 3.1
``pairwise_mbar`` block per edge and payload ``edge_metric``), the 3.3
``cv2_resolution_report.json`` when present, and optionally ``state_registry.csv`` (restraints
for payloads that predate P4). It computes nothing new except sigma_w and one ratio. Any
missing source leaves its fields ``None`` (NA). A flag-off, old or CV1-only campaign still
gets a valid summary; it just carries fewer numbers.

Written only by the CLI (``python -m gareus.adaptive.cv2_resolution_summary <adaptive_dir>``)
and, with ``--ap-cv2-resolution`` on, by the driver after each numbered epoch's apply
(``write_epoch_summary``) and after every final-combined collection
(``write_final_combined_summary``, the file gareus_report grades). With the flag off nothing
is written unless the CLI is run.

Summary ``cv2_resolution_summary.json`` (schema ``cv2_resolution_summary_v1``)::

    schema_version, label, sources {diagnostics, report, registry, diagnostics_schema,
    report_status}, temperature_k, temperature_source, definitions {...},
    edge_metric {metric, status, stage, threshold, n_components, n_weak, n_unmeasured} | null,
    report {status, rules {R1|R2|R3: status}, summary} | null,
    counts   -- the grading inputs gareus_report's "CV2 resolution" row reads (see _counts),
    states   -- per state: state_id, c1, k1, c2, k2, lambda, cv1_restrained, cv2_restrained,
                restraint_source, sample_count, n_pairs, cv1_mean, cv2_mean, cv2_sd, sigma_w2,
                confinement_ratio, sarle_bimodality, r3_evaluated, r3_decision, r3_reason,
                n_modes + modes [{mean, sd, weight}] (the accepted mixture components),
                r3_mode_pair (means of the pair R3 judged, or null), depth_kT, transitions,
                transitions_estimator, transitions_state_series, trapped_or_orthogonal,
                r3_gate + r3_gate_values (the R3 test that decided and its numbers),
    edges    -- per edge: state_i, state_j, edge_type, graph_kind, pattern_pair, graded,
                pairwise_status, pairwise_reason, pairwise_overlap, pairwise_q10, pairwise_q90,
                pairwise_n_eff_min, union_mbar_overlap, below_threshold, weak, measured,
                marginal_overlap, joint_2d_overlap, joint_2d_reason, graded_space,
                and one ``*_space`` stamp per overlap value.

Next to it: ``cv2_resolution_summary_states.csv`` and ``cv2_resolution_summary_edges.csv``.
Default location: the phase directory; the final-combined table goes to the adaptive root
with a ``_final_combined`` suffix. ``--out DIR`` writes ``DIR/<label>/`` instead.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from gareus.adaptive.edge_metric import edge_below_threshold, edge_is_weak_pairwise
from gareus.math_helpers import restraint_sigma

SCHEMA_VERSION = "cv2_resolution_summary_v1"
SUMMARY_NAME = "cv2_resolution_summary.json"
FINAL_LABEL = "final_combined"
FINAL_DIAGNOSTICS = "adaptive_final_combined_diagnostics.json"
EPOCH_DIAGNOSTICS = "adaptive_epoch_diagnostics.json"
REPORT_NAME = "cv2_resolution_report.json"          # mirrors cv2_resolution.REPORT_NAME

# Overlap-space stamps. The first two mirror gareus.mbar_analysis.pmf.OVERLAP_SPACE_MARGINAL /
# _JOINT (and gareus_report's copies) so one vocabulary covers pmf_summary and this table.
SPACE_MARGINAL = "cv1_marginal"
SPACE_JOINT = "cv1_cv2_joint"
SPACE_PAIRWISE = "two_state_mbar"      # 3.1: BAR on the pair's own samples, umbrella terms only
SPACE_UNION = "union_mbar"             # the union solve's pairwise value, when applied
BUDGET_REFUSALS = ("no_reserve", "resolution_budget", "max_replicas_budget")
SPRING_CAP_REFUSAL = "k2_capped_below_compression"

DEFINITIONS = {
    "sigma_w2": "sqrt(k_B T / k2): the CV2 umbrella's own Gaussian width (math_helpers."
                "restraint_sigma); null when k2 <= 0 or not recorded (unrestrained axis).",
    "confinement_ratio": "cv2_sd / sigma_w2, with cv2_sd = sqrt(paired_cv.cv2.var) (ddof 0, all "
                         "paired rows of the phase). < 1: the landscape confines CV2 more than the "
                         "spring; ~1: spring-dominated; > 1: broader than the spring (a mode mixture "
                         "or a soft direction). Landscape-confinement diagnostic only, never a trigger.",
    "transitions": "core-to-core CV2 transitions between the two R3 modes (3.3 report). Estimator "
                   "'replica': within replica residences (R3's default); 'state-series': every "
                   "switch of the state's series, exchange swaps included (>= replica always: the "
                   "permissive choice); 'state_series_lower_bound': no replica-resolved series, a "
                   "lower bound on the state-series count (an upper bound on the replica count); "
                   "'none': not evaluated.",
    "r3_gate": "the 3.3 R3 test that decided the window (cv2_resolution.R3_GATES): single_component, "
               "member_support, no_density_minimum, depth_below_1kT, mode_weight_below_10pct, "
               "too_few_samples, no_replica_series, transitions_below_min or passed; r3_gate_values "
               "holds the measured numbers (members, weights, depth per accepted pair, transitions).",
    "weak": "gareus.adaptive.edge_metric.edge_is_weak_pairwise at the payload threshold: same-"
            "pattern collector geometry edge, measured, bootstrap q90 (or union value) < threshold. "
            "neighbour/spanning/cross-pattern/unmeasured edges are never weak.",
    "measured": "not weak and pairwise status ok, or a union mbar_overlap is present.",
}

_STATE_COLS = ["state_id", "c1", "k1", "c2", "k2", "lambda", "cv1_restrained", "cv2_restrained",
               "restraint_source", "sample_count", "n_pairs", "cv1_mean", "cv2_mean", "cv2_sd", "sigma_w2",
               "confinement_ratio", "sarle_bimodality", "r3_evaluated", "r3_decision", "n_modes",
               "mode_means", "r3_mode_pair", "depth_kT", "transitions", "transitions_estimator", "transitions_state_series",
               "trapped_or_orthogonal", "r3_gate", "r3_reason"]
_EDGE_COLS = ["state_i", "state_j", "edge_type", "graph_kind", "pattern_pair", "graded", "pairwise_status",
              "pairwise_reason", "pairwise_overlap", "pairwise_q10", "pairwise_q90", "pairwise_n_eff_min",
              "pairwise_space", "union_mbar_overlap", "union_space", "below_threshold", "weak", "measured",
              "marginal_overlap", "marginal_space", "joint_2d_overlap", "joint_2d_reason", "joint_space",
              "graded_space"]


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def sigma_w(k: Any, temperature_k: Optional[float]) -> Optional[float]:
    """The umbrella width on one axis, or None (unrestrained, unknown k or unknown T)."""
    kk, t = _finite(k), _finite(temperature_k)
    if kk is None or kk <= 0.0 or t is None or t <= 0.0:
        return None
    return _finite(restraint_sigma(kk, t))


def _restrained(flag: Any, k: Any) -> Optional[bool]:
    if isinstance(flag, bool):
        return flag
    kk = _finite(k)
    return None if kk is None else kk > 0.0


# ---- per state --------------------------------------------------------------------------

def load_registry_rows(adaptive_dir: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    """``state_registry.csv`` rows by state id ({} when absent/unreadable)."""
    if adaptive_dir is None:
        return {}
    path = Path(adaptive_dir) / "state_registry.csv"
    try:
        with open(path, newline="") as fh:
            return {int(r["state_id"]): r for r in csv.DictReader(fh)}
    except (OSError, KeyError, ValueError):
        return {}


def _restraint(row: Mapping[str, Any], reg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    rec = ((row.get("paired_cv") or {}).get("restraint")) or {}
    src = "paired_cv" if rec else ("registry" if reg else None)
    rec = rec or {"primary_center": (reg or {}).get("primary_center"), "primary_k": (reg or {}).get("primary_k"),
                  "secondary_center": (reg or {}).get("secondary_center"),
                  "secondary_k": (reg or {}).get("secondary_k"), "gamd_lambda": (reg or {}).get("gamd_lambda")}
    k1, k2 = _finite(rec.get("primary_k")), _finite(rec.get("secondary_k"))
    return {"c1": _finite(rec.get("primary_center")), "k1": k1, "c2": _finite(rec.get("secondary_center")),
            "k2": k2, "lambda": _finite(rec.get("gamd_lambda")),
            "cv1_restrained": _restrained(rec.get("cv1_restrained"), k1),
            "cv2_restrained": _restrained(rec.get("cv2_restrained"), k2), "restraint_source": src}


def _r3_by_state(report: Optional[Mapping[str, Any]]) -> Dict[int, Mapping[str, Any]]:
    out: Dict[int, Mapping[str, Any]] = {}
    for c in (report or {}).get("candidates", []) or []:
        if c.get("rule") == "R3" and len(c.get("state_ids") or []) == 1:
            out[int(c["state_ids"][0])] = c
    return out


_LOWER_BOUND_KEYS = ("transitions_state_series_lower_bound",   # 3.3 report v2
                     "transitions_lower_bound")                 # v1 (same value, misleading name)


def _transitions(m: Mapping[str, Any]) -> Tuple[Optional[int], str]:
    """(count, estimator) from a 3.3 R3 candidate's metrics, v1 or v2 report.

    ``replica``: crossings within one replica's residence (R3's default). ``state-series``:
    every switch of the state-indexed series = the within-residence crossings PLUS the label
    changes an exchange swap causes, so state-series >= replica always and
    ``--ap-refine-transition-count state-series`` is the permissive choice.
    ``state_series_lower_bound``: no replica-resolved series; the value bounds the state-series
    count from below and says nothing that could raise the replica count (it bounds it above)."""
    if m.get("transitions") is not None:
        return int(m["transitions"]), str(m.get("transitions_estimator") or "replica")
    for key in _LOWER_BOUND_KEYS:
        if m.get(key) is not None:
            return int(m[key]), "state_series_lower_bound"
    return None, "none"


def _r3_fields(cand: Optional[Mapping[str, Any]], has_report: bool) -> Dict[str, Any]:
    if cand is None:
        why = "not an R3 candidate in the 3.3 report" if has_report else "no 3.3 report for this phase"
        return {"r3_evaluated": False, "r3_decision": None, "r3_reason": why, "n_modes": None, "modes": None,
                "r3_mode_pair": None,
                "depth_kT": None, "transitions": None, "transitions_estimator": "none",
                "transitions_state_series": None, "trapped_or_orthogonal": None, "r3_gate": None,
                "r3_gate_values": None}
    m = cand.get("metrics") or {}
    comps = [x for x in ((m.get("mixture") or {}).get("components") or []) if x.get("accepted")]
    modes = [{"mean": _finite(x.get("mean")), "sd": _finite(x.get("sd")), "weight": _finite(x.get("weight"))}
             for x in comps]
    pair = [_finite(x.get("mean")) for x in (m.get("modes") or [])]
    n, est = _transitions(m)
    return {"r3_evaluated": True, "r3_decision": cand.get("decision"), "r3_reason": cand.get("reason"),
            "n_modes": len(modes), "modes": modes, "r3_mode_pair": pair or None,
            "depth_kT": _finite((m.get("depth") or {}).get("depth_kT")),
            "transitions": n, "transitions_estimator": est,
            "transitions_state_series": _int(m.get("transitions_state_series")),
            "trapped_or_orthogonal": bool(m.get("trapped_or_orthogonal", False)),
            "r3_gate": m.get("r3_gate"), "r3_gate_values": m.get("r3_gate_values")}


def state_record(row: Mapping[str, Any], reg: Optional[Mapping[str, Any]], r3: Optional[Mapping[str, Any]],
                 temperature_k: Optional[float], has_report: bool) -> Dict[str, Any]:
    """One table row for one state (see the module docstring for the fields)."""
    pc = row.get("paired_cv") or {}
    m1, m2 = pc.get("cv1") or {}, pc.get("cv2") or {}
    rest = _restraint(row, reg)
    var2 = _finite(m2.get("var"))
    sd2 = math.sqrt(max(var2, 0.0)) if var2 is not None else None
    sw2 = sigma_w(rest["k2"], temperature_k)
    sarle = ((r3 or {}).get("metrics") or {}).get("sarle_bimodality")
    return {"state_id": int(row.get("state_id")), **rest, "sample_count": _int(row.get("sample_count")),
            "n_pairs": _int(pc.get("n_pairs")), "cv1_mean": _finite(m1.get("mean")),
            "cv2_mean": _finite(m2.get("mean")) if m2 else _finite(row.get("secondary_mean")),
            "cv2_sd": sd2, "sigma_w2": sw2,
            "confinement_ratio": sd2 / sw2 if sd2 is not None and sw2 else None,
            "sarle_bimodality": _finite(sarle), **_r3_fields(r3, has_report)}


# ---- per edge ---------------------------------------------------------------------------

def edge_record(edge: Mapping[str, Any], threshold: Optional[float]) -> Dict[str, Any]:
    """One table row for one edge. Rung edges are listed but never graded."""
    pm = edge.get("pairwise_mbar") or {}
    graded = str(edge.get("edge_type")) != "rung" and bool(pm) and threshold is not None
    union = _finite(edge.get("mbar_overlap"))
    weak = edge_is_weak_pairwise(edge, threshold) if graded else None
    measured = (bool(weak) or pm.get("status") == "ok" or union is not None) if graded else None
    raw = pm.get("n_eff")
    neff = [v for v in (_finite(x) for x in (list(raw) if raw is not None else [])) if v is not None]
    return {"state_i": int(edge.get("state_i")), "state_j": int(edge.get("state_j")),
            "edge_type": edge.get("edge_type"), "graph_kind": pm.get("graph_kind"),
            "pattern_pair": pm.get("pattern_pair"), "graded": graded,
            "pairwise_status": pm.get("status"), "pairwise_reason": pm.get("reason"),
            "pairwise_overlap": _finite(pm.get("overlap")), "pairwise_q10": _finite(pm.get("overlap_lower")),
            "pairwise_q90": _finite(pm.get("overlap_upper")), "pairwise_n_eff_min": min(neff) if neff else None,
            "pairwise_space": SPACE_PAIRWISE, "union_mbar_overlap": union, "union_space": SPACE_UNION,
            "below_threshold": edge_below_threshold(edge, threshold) if graded else None,
            "weak": weak, "measured": measured,
            "marginal_overlap": _finite(edge.get("overlap")), "marginal_space": SPACE_MARGINAL,
            "joint_2d_overlap": _finite(edge.get("overlap_joint_2d")),
            "joint_2d_reason": edge.get("overlap_joint_2d_reason"), "joint_space": SPACE_JOINT,
            "graded_space": (SPACE_UNION if union is not None else SPACE_PAIRWISE) if graded else None}


# ---- counts (the grading inputs) --------------------------------------------------------

def _budget_refusals(report: Optional[Mapping[str, Any]]) -> int:
    rep = report or {}
    n = sum(1 for c in rep.get("candidates", []) or [] if c.get("refusal") in BUDGET_REFUSALS)
    n += sum(1 for r in ((rep.get("apply") or {}).get("refused") or [])
             if str(r.get("reason", "")) in BUDGET_REFUSALS)
    return n


def _spring_cap_refusals(report: Mapping[str, Any]) -> int:
    return sum(1 for c in report.get("candidates", []) or [] if c.get("refusal") == SPRING_CAP_REFUSAL)


def _counts(states: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]],
            em: Optional[Mapping[str, Any]], report: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    graded = [e for e in edges if e["graded"]]
    comps = ((em or {}).get("components") or {}).get("n_components")
    summ = (report or {}).get("summary") or {}
    return {"n_states": len(states), "n_cv2_restrained": sum(1 for s in states if s["cv2_restrained"]),
            "n_with_paired_cv": sum(1 for s in states if s["n_pairs"]),
            "n_r3_evaluated": sum(1 for s in states if s["r3_evaluated"]),
            "n_trapped_or_orthogonal": sum(1 for s in states if s["trapped_or_orthogonal"]),
            "n_edges": len(edges), "n_graded_edges": len(graded),
            "n_weak": sum(1 for e in graded if e["weak"]),
            "n_unmeasured": sum(1 for e in graded if not e["measured"]),
            "n_below_threshold": sum(1 for e in graded if e["below_threshold"]),
            "n_components": int(comps) if comps is not None else None,
            "edge_metric_status": (em or {}).get("status"),
            "report_status": (report or {}).get("status"),
            "n_proposed": summ.get("n_proposed"), "n_blocking": summ.get("n_blocking"),
            "n_refused_budget": _budget_refusals(report) if report else None,
            "n_refused_spring_cap": _spring_cap_refusals(report) if report else None}


# ---- the summary ------------------------------------------------------------------------

def resolve_temperature(diagnostics: Mapping[str, Any], report: Optional[Mapping[str, Any]],
                        override: Optional[float] = None) -> Tuple[Optional[float], Optional[str]]:
    """Explicit override, else the edge metric's, else the 3.3 report's temperature."""
    for value, src in ((override, "override"), ((diagnostics.get("edge_metric") or {}).get("temperature_k"),
                                                 "edge_metric"),
                       (((report or {}).get("settings") or {}).get("temperature_k"), "cv2_resolution_report")):
        if _finite(value) is not None and float(value) > 0:
            return float(value), src
    return None, None


def build_summary(diagnostics: Mapping[str, Any], report: Optional[Mapping[str, Any]] = None, *,
                  label: str, registry_rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
                  temperature_k: Optional[float] = None,
                  sources: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The phase's CV2-resolution table. Pure: reads its arguments only."""
    reg = registry_rows or {}
    temp, temp_src = resolve_temperature(diagnostics, report, temperature_k)
    em = diagnostics.get("edge_metric")
    threshold = _finite((em or {}).get("threshold")) if em else None
    r3 = _r3_by_state(report)
    states = [state_record(row, reg.get(int(row.get("state_id"))), r3.get(int(row.get("state_id"))), temp,
                           report is not None) for row in diagnostics.get("states", []) or []]
    edges = [edge_record(e, threshold) for e in diagnostics.get("edges", []) or []]
    em_rec = None if not em else {k: em.get(k) for k in ("metric", "status", "stage", "threshold", "error",
                                                          "n_weak", "n_unmeasured", "temperature_k")}
    if em_rec is not None:
        em_rec["n_components"] = ((em.get("components") or {}).get("n_components"))
    rep_rec = None if report is None else {
        "status": report.get("status"), "epoch": report.get("epoch"), "stage": report.get("stage"),
        "rules": {k: (v or {}).get("status") for k, v in (report.get("rules") or {}).items()},
        "summary": report.get("summary"), "budget_mode": (report.get("budget") or {}).get("mode")}
    return {"schema_version": SCHEMA_VERSION, "label": label,
            "sources": {**dict(sources or {}), "diagnostics_schema": diagnostics.get("schema_version"),
                        "report_status": (report or {}).get("status")},
            "temperature_k": temp, "temperature_source": temp_src, "definitions": DEFINITIONS,
            "edge_metric": em_rec, "report": rep_rec, "counts": _counts(states, edges, em, report),
            "states": states, "edges": edges}


# ---- I/O --------------------------------------------------------------------------------

def _cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return v


def _json_default(o: Any) -> Any:
    """numpy scalars/arrays from an in-memory payload (the driver hook) -> plain JSON."""
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _join(values: Any) -> str:
    return ";".join(f"{v:.4g}" for v in values if v is not None)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], cols: Sequence[str]) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([_cell(r.get(c)) for c in cols])
    os.replace(tmp, path)


def write_summary(summary: Mapping[str, Any], json_path: Path) -> Dict[str, Path]:
    """``json_path`` plus ``<stem>_states.csv`` / ``<stem>_edges.csv`` beside it (atomic)."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = json_path.with_name(json_path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(summary, indent=1, sort_keys=False, allow_nan=False, default=_json_default))
    os.replace(tmp, json_path)
    stem = json_path.with_suffix("")
    states = [{**s, "mode_means": _join(m.get("mean") for m in (s.get("modes") or [])),
               "r3_mode_pair": _join(s.get("r3_mode_pair") or [])} for s in summary["states"]]
    paths = {"json": json_path, "states_csv": Path(f"{stem}_states.csv"), "edges_csv": Path(f"{stem}_edges.csv")}
    _write_csv(paths["states_csv"], states, _STATE_COLS)
    _write_csv(paths["edges_csv"], summary["edges"], _EDGE_COLS)
    return paths


def _read_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        print(f"WARNING: unreadable {path} ({exc}); treated as absent")
        return None


def default_json_path(adaptive_dir: Path, label: str) -> Path:
    """Where a phase's summary lives by default (gareus_report reads the final-combined one)."""
    if label == FINAL_LABEL:
        return Path(adaptive_dir) / f"{Path(SUMMARY_NAME).stem}_{FINAL_LABEL}.json"
    return Path(adaptive_dir) / label / SUMMARY_NAME


def discover_phases(adaptive_dir: Path) -> List[Tuple[str, Path, Optional[Path]]]:
    """(label, diagnostics, report-or-None) for every numbered epoch/final phase with a
    phase-level (pooled) diagnostics file, plus the final-combined payload."""
    root = Path(adaptive_dir)
    out: List[Tuple[str, Path, Optional[Path]]] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and re.match(r"^(epoch_\d+|final(_extension_\d+)?)$",
                                                                          p.name)):
        if (d / EPOCH_DIAGNOSTICS).exists():
            rep = d / REPORT_NAME
            out.append((d.name, d / EPOCH_DIAGNOSTICS, rep if rep.exists() else None))
    if (root / FINAL_DIAGNOSTICS).exists():
        out.append((FINAL_LABEL, root / FINAL_DIAGNOSTICS, None))
    return out


def summarise(label: str, diagnostics_path: Path, report_path: Optional[Path], json_path: Path, *,
              adaptive_dir: Optional[Path] = None, temperature_k: Optional[float] = None) -> Dict[str, Any]:
    """Read one phase's sources, build and write its summary; returns the summary."""
    diag = _read_json(diagnostics_path)
    if diag is None:
        raise FileNotFoundError(f"no readable diagnostics at {diagnostics_path}")
    report = _read_json(report_path)
    reg = load_registry_rows(adaptive_dir)
    summary = build_summary(diag, report, label=label, registry_rows=reg, temperature_k=temperature_k,
                            sources={"diagnostics": str(diagnostics_path),
                                     "report": str(report_path) if report is not None else None,
                                     "registry": str(Path(adaptive_dir) / "state_registry.csv") if reg else None})
    write_summary(summary, json_path)
    return summary


def write_epoch_summary(epoch_dir: Path, diagnostics: Mapping[str, Any]) -> Optional[Path]:
    """Driver hook (``--ap-cv2-resolution`` only, after the apply): the epoch's summary next to
    its 3.3 report. Never raises."""
    try:
        epoch_dir = Path(epoch_dir)
        report_path = epoch_dir / REPORT_NAME
        report = _read_json(report_path)
        reg = load_registry_rows(epoch_dir.parent)
        summary = build_summary(diagnostics, report, label=epoch_dir.name, registry_rows=reg,
                                sources={"diagnostics": str(epoch_dir / EPOCH_DIAGNOSTICS),
                                         "report": str(report_path) if report is not None else None,
                                         "registry": str(epoch_dir.parent / "state_registry.csv") if reg else None})
        path = epoch_dir / SUMMARY_NAME
        write_summary(summary, path)
        return path
    except Exception as exc:  # reporting never kills a completed epoch
        print(f"WARNING: CV2 resolution summary failed for {epoch_dir} ({type(exc).__name__}: {exc})")
        return None


def write_final_combined_summary(adaptive_dir: Path, diagnostics: Mapping[str, Any]) -> Optional[Path]:
    """Driver hook (``--ap-cv2-resolution`` only): the final-combined table that the
    gareus_report "CV2 resolution" row reads, rewritten every time the final-combined
    diagnostics are (so later extensions refresh it). No 3.3 report exists for the final
    phase. Never raises."""
    try:
        adaptive_dir = Path(adaptive_dir)
        reg = load_registry_rows(adaptive_dir)
        summary = build_summary(diagnostics, None, label=FINAL_LABEL, registry_rows=reg,
                                sources={"diagnostics": str(adaptive_dir / FINAL_DIAGNOSTICS), "report": None,
                                         "registry": str(adaptive_dir / "state_registry.csv") if reg else None})
        path = default_json_path(adaptive_dir, FINAL_LABEL)
        write_summary(summary, path)
        return path
    except Exception as exc:  # reporting never kills a completed campaign phase
        print(f"WARNING: final-combined CV2 resolution summary failed in {adaptive_dir} "
              f"({type(exc).__name__}: {exc})")
        return None


def _one_line(summary: Mapping[str, Any]) -> str:
    c = summary["counts"]
    return (f"{summary['label']}: {c['n_states']} states ({c['n_cv2_restrained']} CV2-restrained, "
            f"{c['n_r3_evaluated']} R3-evaluated, {c['n_trapped_or_orthogonal']} trapped_or_orthogonal); "
            f"{c['n_graded_edges']} graded edges, {c['n_weak']} weak, {c['n_unmeasured']} unmeasured, "
            f"components {c['n_components']}; budget refusals {c['n_refused_budget']}, "
            f"spring-cap refusals {c['n_refused_spring_cap']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m gareus.adaptive.cv2_resolution_summary",
                                description="Spec 3.7 per-phase CV2 resolution table (read-only on inputs).")
    p.add_argument("adaptive_dir", type=Path)
    p.add_argument("--phases", nargs="*", default=None, help="labels to keep (default: all discovered)")
    p.add_argument("--out", type=Path, default=None, help="write DIR/<label>/ instead of the phase dirs")
    p.add_argument("--diagnostics", type=Path, default=None, help="explicit payload (one phase; needs --label)")
    p.add_argument("--report", type=Path, default=None, help="explicit 3.3 report for --diagnostics")
    p.add_argument("--label", default=None)
    p.add_argument("--temperature-k", type=float, default=None)
    a = p.parse_args(argv)
    if a.diagnostics is not None:
        jobs = [(a.label or a.diagnostics.stem, a.diagnostics, a.report)]
    else:
        jobs = [j for j in discover_phases(a.adaptive_dir) if a.phases is None or j[0] in a.phases]
    if not jobs:
        print(f"no phase diagnostics found under {a.adaptive_dir}")
        return 1
    for label, diag, rep in jobs:
        dest = (a.out / label / SUMMARY_NAME) if a.out is not None else default_json_path(a.adaptive_dir, label)
        summary = summarise(label, diag, rep, dest, adaptive_dir=a.adaptive_dir, temperature_k=a.temperature_k)
        print(_one_line(summary) + f" -> {dest}")
    return 0


__all__ = ["DEFINITIONS", "FINAL_LABEL", "SCHEMA_VERSION", "SUMMARY_NAME", "build_summary", "default_json_path",
           "discover_phases", "edge_record", "load_registry_rows", "main", "resolve_temperature", "sigma_w",
           "state_record", "summarise", "write_epoch_summary", "write_final_combined_summary", "write_summary"]


if __name__ == "__main__":
    sys.exit(main())
