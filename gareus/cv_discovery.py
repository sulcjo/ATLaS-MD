"""Analysis-driven CV suggestion reports for completed GAREUS runs.

This is intentionally lightweight: it uses the production scalar arrays already
written by GAREUS (analysis_arrays.npz or analysis_chunks/*.npz) and produces a
human-readable heuristic report.  It does not change sampling, reweighting, or
claim to discover exact reaction coordinates; it flags symptoms that suggest the
next CV/window refinement to try.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .analysis import validate_analysis_metadata_readiness
from .diagnostics import _hist_overlap_np, _read_csv_dicts, _safe_float
from .io import read_json_file, write_json, _json_ready


def _load_arrays(run_dir: Path, keys: Iterable[str] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load selected analysis arrays from consolidated NPZ or chunk manifest."""
    run_dir = Path(run_dir)
    wanted = set(keys or [])
    arrays_path = run_dir / "analysis_arrays.npz"
    if arrays_path.exists():
        with np.load(arrays_path, allow_pickle=False) as data:
            out = {k: np.asarray(data[k]) for k in data.files if not wanted or k in wanted}
        return out, {"source": str(arrays_path), "source_kind": "consolidated_npz"}

    manifest_path = run_dir / "analysis_chunks_manifest.json"
    manifest = read_json_file(manifest_path, {}) if manifest_path.exists() else {}
    chunks = manifest.get("chunks", []) if isinstance(manifest, dict) else []
    if not chunks:
        return {}, {"source": "", "source_kind": "missing", "warning": "no analysis_arrays.npz or analysis_chunks_manifest.json found"}

    loaded: dict[str, list[np.ndarray]] = {}
    for chunk in chunks:
        chunk_path = Path(str(chunk.get("path", "")))
        if not chunk_path.is_absolute():
            chunk_path = run_dir / chunk_path
        if not chunk_path.exists():
            # Some manifests record paths relative to the run parent.
            chunk_path = run_dir / str(chunk.get("path", ""))
        if not chunk_path.exists():
            continue
        with np.load(chunk_path, allow_pickle=False) as data:
            for k in data.files:
                if wanted and k not in wanted:
                    continue
                loaded.setdefault(k, []).append(np.asarray(data[k]))
    arrays = {k: np.concatenate(v, axis=0) for k, v in loaded.items() if v}
    return arrays, {"source": str(manifest_path), "source_kind": "chunk_manifest", "n_chunks_loaded": int(len(chunks))}


def _finite(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    return a[np.isfinite(a)]


def _summary_stats(arr) -> dict[str, Any]:
    a = _finite(arr)
    if a.size == 0:
        return {"n": 0, "min": None, "max": None, "mean": None, "std": None, "q05": None, "q50": None, "q95": None}
    return {
        "n": int(a.size),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "q05": float(np.quantile(a, 0.05)),
        "q50": float(np.quantile(a, 0.50)),
        "q95": float(np.quantile(a, 0.95)),
    }


def _window_table(run_dir: Path) -> list[dict[str, str]]:
    p = run_dir / "umbrella_explicit_windows.csv"
    if p.exists():
        return _read_csv_dicts(p)
    return _read_csv_dicts(run_dir / "umbrella_windows.csv")


def _centers_from_table(rows: list[dict[str, str]]) -> tuple[list[float], list[float]]:
    primary = []
    secondary = []
    for r in rows:
        pv = _safe_float(r.get("primary_center", r.get("distance_center_A", r.get("center_A"))), float("nan"))
        if math.isfinite(pv):
            primary.append(float(pv))
        sv = _safe_float(r.get("secondary_cv_center"), float("nan"))
        if math.isfinite(sv):
            secondary.append(float(sv))
    return primary, secondary


def _count_low_overlap_neighbors(cv: np.ndarray, win: np.ndarray, n_windows: int, rows: list[dict[str, str]], target: float) -> list[dict[str, Any]]:
    centers, _secondary = _centers_from_table(rows)
    finite_centers = [x for x in centers if math.isfinite(float(x))]
    lo = min(finite_centers) if finite_centers else float(np.nanmin(cv))
    hi = max(finite_centers) if finite_centers else float(np.nanmax(cv))
    out = []
    for i in range(max(0, int(n_windows) - 1)):
        a = cv[win == i]
        b = cv[win == i + 1]
        ov = _hist_overlap_np(a, b, lo, hi, bins=80)
        if not math.isfinite(ov) or ov < float(target):
            out.append({"window_i": int(i), "window_j": int(i + 1), "overlap": None if not math.isfinite(ov) else float(ov), "left_count": int(a.size), "right_count": int(b.size)})
    return out


def suggest_cvs(run_dir: Path, target_overlap: float = 0.25) -> dict[str, Any]:
    run_dir = Path(run_dir)
    keys = ["cv_A", "secondary_cv", "window", "center_A", "secondary_cv_center", "gamd_boost_total_kj_mol"]
    arrays, source = _load_arrays(run_dir, keys=keys)
    rows = _window_table(run_dir)
    primary_centers, secondary_centers = _centers_from_table(rows)
    n_windows = len(rows)
    suggestions: list[dict[str, Any]] = []
    warnings: list[str] = []

    cv = np.asarray(arrays.get("cv_A", []), dtype=float)
    win = np.asarray(arrays.get("window", []), dtype=int) if "window" in arrays else np.asarray([], dtype=int)
    sec = np.asarray(arrays.get("secondary_cv", []), dtype=float) if "secondary_cv" in arrays else np.asarray([], dtype=float)
    primary_stats = _summary_stats(cv)
    secondary_stats = _summary_stats(sec)

    readiness = {}
    try:
        readiness = validate_analysis_metadata_readiness(run_dir, target_overlap=target_overlap)
    except Exception as exc:
        warnings.append(f"readiness validation failed: {exc}")

    if cv.size == 0:
        suggestions.append({
            "priority": "fatal",
            "title": "No production CV arrays found",
            "details": "Run production with analysis chunks/consolidated arrays enabled before asking for data-driven CV suggestions.",
            "try": "Keep --write-analysis-chunks enabled or allow analysis_arrays.npz to be written.",
        })
    else:
        if n_windows > 1 and win.size == cv.size:
            # Windows are a flat list; on sparse explicit 2D grids flat index
            # i/i+1 is not a CV-space neighbor relation. Detect sparse 2D the
            # same way gareus.analysis.validate_analysis_metadata_readiness
            # does (explicit_2d/rectangular_grid flags on the window table)
            # so the flat-adjacency loop is gated off even if the `readiness`
            # call above failed to produce a sparse_2d verdict of its own.
            explicit_2d = any(str(r.get("explicit_2d", "0")) in {"1", "True", "true"} for r in rows)
            rectangular = all(str(r.get("rectangular_grid", "0")) in {"1", "True", "true"} for r in rows) if explicit_2d and rows else False
            sparse_2d = bool(explicit_2d and not rectangular)
            if sparse_2d:
                # `readiness` already computed graph-adjacent overlaps the
                # sparse-2D-safe way; reuse that instead of re-deriving flat
                # adjacency here. If it is unavailable, soften to "no known
                # weak pairs" rather than fall back to flat i,i+1.
                weak = list(readiness.get("neighbor_pairs_below_target_overlap", []) or [])
                weak_label = "graph-neighbor pair(s)"
            else:
                weak = _count_low_overlap_neighbors(cv, win, n_windows, rows, target_overlap)
                weak_label = "adjacent pair(s)"
            if weak:
                suggestions.append({
                    "priority": "high",
                    "title": "Primary-CV neighbor overlap is weak in one or more adjacent windows",
                    "details": f"{len(weak)} {weak_label} are below target overlap {target_overlap:.2f}.",
                    "try": "Use --window-mode adaptive-feedback, add midpoint windows near weak pairs, or lower local umbrella k.",
                    "evidence": weak[:12],
                })
        finite_primary = _finite(cv)
        if finite_primary.size and primary_centers:
            pmin, pmax = min(primary_centers), max(primary_centers)
            outside = float(np.mean((finite_primary < pmin) | (finite_primary > pmax)))
            edge_band = max(1.0e-12, 0.05 * max(1.0e-12, pmax - pmin))
            edge_frac = float(np.mean((finite_primary <= pmin + edge_band) | (finite_primary >= pmax - edge_band)))
            if outside > 0.05 or edge_frac > 0.30:
                suggestions.append({
                    "priority": "medium",
                    "title": "Primary-CV sampling piles up near or beyond the window range",
                    "details": f"outside_range_fraction={outside:.3f}, edge_band_fraction={edge_frac:.3f}.",
                    "try": "Expand primary centers or run an adaptive prescan before final production.",
                })

    finite_sec = _finite(sec)
    has_secondary_windows = bool(secondary_centers)
    if not has_secondary_windows:
        suggestions.append({
            "priority": "high",
            "title": "Only a primary CV appears to be biased",
            "details": "For short peptides, terminal distance or contacts alone often collapse distinct backbone basins into one coordinate.",
            "try": "Use --cv2 rama-map for a general backbone-basin axis, or --cv2 alpha-coil-beta for a simpler alpha/extended transition coordinate.",
        })
    elif finite_sec.size:
        smin, smax = min(secondary_centers), max(secondary_centers)
        if np.std(finite_sec) < 0.08 and (smax - smin) > 0.5:
            suggestions.append({
                "priority": "high",
                "title": "Secondary CV did not move much despite multiple CV2 targets",
                "details": f"secondary_std={float(np.std(finite_sec)):.3f} across target range {smin:.2f}..{smax:.2f}.",
                "try": "Increase --us-pull-steps-per-window, set --us-2d-start-distance-fraction 0.05, raise --us-2d-start-secondary-k-pull-scale, or generate CV2-aware seeds.",
            })
        finite_centers = np.asarray(secondary_centers, dtype=float)
        coverage = []
        for c in sorted(set(float(x) for x in finite_centers if math.isfinite(float(x)))):
            near = float(np.mean(np.abs(finite_sec - c) < 0.20)) if finite_sec.size else 0.0
            coverage.append({"center": c, "fraction_within_0p20": near})
        poorly = [x for x in coverage if x["fraction_within_0p20"] < 0.03]
        if poorly:
            suggestions.append({
                "priority": "medium",
                "title": "Some secondary-CV target regions are rarely visited",
                "details": "Low occupancy near one or more CV2 centers suggests missing backbone basins or too-weak/too-short setup pulls.",
                "try": "Use active-CV GENPEPT seeding, longer 2D starting pulls, or prune unreachable CV2 centers after a pilot.",
                "evidence": poorly,
            })

    if rows:
        primary_cv_labels = sorted({str(r.get("primary_cv", "distance")) for r in rows if r.get("primary_cv", "")})
        if primary_cv_labels == ["distance"] or not primary_cv_labels:
            suggestions.append({
                "priority": "medium",
                "title": "Distance CV may miss topology-changing compact states",
                "details": "Equal terminal distance can hide different contact maps/backbone folds.",
                "try": "Compare a pilot with --cv1 contacts --cv2 rama-map, or add a contacts-based campaign for cross-checking.",
            })
        elif any("contact" in x for x in primary_cv_labels):
            suggestions.append({
                "priority": "medium",
                "title": "Contact CV is nondirectional unless paired with backbone/geometry information",
                "details": "A high contact fraction can mean many different collapsed topologies.",
                "try": "Keep --cv2 rama-map or use selected motif contacts rather than all atom-pair contacts for interpretability.",
            })

    payload = {
        "schema_version": "cv-suggestions-v1",
        "run_dir": str(run_dir),
        "array_source": source,
        "n_windows": int(n_windows),
        "n_samples": int(cv.size),
        "primary_cv_stats": primary_stats,
        "secondary_cv_stats": secondary_stats,
        "primary_centers": [float(x) for x in primary_centers],
        "secondary_centers": [float(x) for x in secondary_centers],
        "readiness_status": readiness.get("status") if isinstance(readiness, dict) else None,
        "readiness_errors": readiness.get("errors", []) if isinstance(readiness, dict) else [],
        "readiness_warnings": readiness.get("warnings", []) if isinstance(readiness, dict) else [],
        "warnings": warnings,
        "suggestions": suggestions,
    }
    return payload


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = []
    lines.append("# GAREUS CV suggestion report")
    lines.append("")
    lines.append(f"Run directory: `{payload.get('run_dir')}`")
    lines.append(f"Array source: `{(payload.get('array_source') or {}).get('source_kind', 'unknown')}`")
    lines.append(f"Samples: {payload.get('n_samples', 0)}; windows: {payload.get('n_windows', 0)}")
    lines.append("")
    lines.append("## Primary CV statistics")
    ps = payload.get("primary_cv_stats", {}) or {}
    lines.append(f"min={ps.get('min')}, median={ps.get('q50')}, max={ps.get('max')}, std={ps.get('std')}")
    lines.append("")
    lines.append("## Secondary CV statistics")
    ss = payload.get("secondary_cv_stats", {}) or {}
    lines.append(f"min={ss.get('min')}, median={ss.get('q50')}, max={ss.get('max')}, std={ss.get('std')}")
    lines.append("")
    lines.append("## Suggestions")
    suggestions = payload.get("suggestions", []) or []
    if not suggestions:
        lines.append("No strong heuristic CV-change suggestions were triggered. Keep comparing independent campaigns; molecules are sneaky little bureaucrats.")
    for i, item in enumerate(suggestions, start=1):
        lines.append(f"### {i}. [{item.get('priority', 'info')}] {item.get('title', '')}")
        lines.append("")
        if item.get("details"):
            lines.append(str(item.get("details")))
            lines.append("")
        if item.get("try"):
            lines.append("Try: " + str(item.get("try")))
            lines.append("")
        if item.get("evidence"):
            lines.append("Evidence preview:")
            lines.append("```json")
            lines.append(json.dumps(item.get("evidence"), indent=2, sort_keys=True))
            lines.append("```")
            lines.append("")
    if payload.get("readiness_errors") or payload.get("readiness_warnings"):
        lines.append("## Analysis-readiness notes")
        for e in payload.get("readiness_errors", []) or []:
            lines.append(f"- ERROR: {e}")
        for w in payload.get("readiness_warnings", []) or []:
            lines.append(f"- WARNING: {w}")
    lines.append("")
    lines.append("This report is heuristic.  It suggests next exploratory CV/window choices; it is not a substitute for MBAR uncertainty, independent-repeat convergence, or structural inspection.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="gareus-suggest-cvs",
        description="Generate a heuristic, analysis-driven CV/window suggestion report from a completed GAREUS run.",
    )
    p.add_argument("--run-dir", default=".", help="Completed GAREUS run directory.")
    p.add_argument("--out-prefix", default="cv_suggestions", help="Output prefix under the run directory.")
    p.add_argument("--target-overlap", type=float, default=0.25, help="Neighbor histogram overlap threshold used for weak-window suggestions.")
    p.add_argument("--print", dest="print_report", action="store_true", help="Print the Markdown report to stdout as well as writing files.")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    payload = suggest_cvs(run_dir, target_overlap=float(args.target_overlap))
    json_path = run_dir / f"{args.out_prefix}.json"
    md_path = run_dir / f"{args.out_prefix}.md"
    write_json(json_path, _json_ready(payload))
    md = _markdown_report(payload)
    md_path.write_text(md, encoding="utf-8")
    if args.print_report:
        print(md)
    else:
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
