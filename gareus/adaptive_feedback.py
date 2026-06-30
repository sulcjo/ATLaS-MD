"""
Adaptive-feedback orchestration and window-refinement helpers.

This module contains the real adaptive-feedback implementation extracted from
``gareus_peptide.py``.  It covers pilot-round orchestration, adaptive proposal
construction, 1D/2D window diagnostics, sparse 2D patch generation, workflow
cleanup, and final-production dispatch.  It imports all helper functionality
from modular ``gareus.*`` modules and no longer reaches into the historical
monolith.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .io import write_json, read_json_file, _json_ready
from .progress import GuiProgressSink
from .lifecycle import _graceful_shutdown
from .diagnostics import compute_gamd_reweighting_diagnostics, validate_us_mbar_inputs
from .analysis import validate_analysis_metadata_readiness
from .production import (
    run_gareus,
    write_explicit_2d_neighbor_graph_files,
    write_standardized_output_layout,
)
from .windows import (
    build_explicit_2d_neighbor_edges,
    _adaptive_window_aggressiveness_settings,
    _secondary_center_count_for_total_budget,
    _adaptive_total_window_limits,
    _axis_count_bounds_from_total_window_budget,
    _thin_axis_centers_to_count,
    _apply_total_window_budget_to_factorized_grid,
    build_adaptive_window_centers_a,
    adaptive_force_constants_kcal_a2,
    adaptive_contact_centers,
    adaptive_contact_force_constants_kcal,
)
from .cv import (
    adaptive_secondary_force_constants_kcal,
    format_primary_delta_value,
    primary_cv_is_contacts,
    primary_cv_label,
    primary_cv_mode,
    primary_cv_units,
    primary_k_units,
    secondary_cv_enabled,
    secondary_cv_is_transition,
    secondary_cv_mode,
)
from .math_helpers import _adaptive_hist_overlap
from .genpept_window_prior import build_genpept_window_prior, genpept_prior_enabled

__all__ = [
    "adaptive_secondary_default_centers",
    "adaptive_contact_centers",
    "adaptive_contact_force_constants_kcal",
    "build_adaptive_window_centers_a",
    "adaptive_force_constants_kcal_a2",
    "write_adaptive_feedback_summary_report",
    "run_adaptive_feedback_dispatcher",
    "run_adaptive_feedback_dispatcher_2d",
    "run_adaptive_feedback_dispatcher_2d_explicit_sparse",
    "cleanup_adaptive_feedback_pilot_directory",
    "run_adaptive_feedback_auto_loop",
]



def _count_csv_rows(path: Path) -> int:
    try:
        with Path(path).open(newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except Exception:
        return 0


def write_adaptive_feedback_summary_report(out_dir: Path) -> dict:
    """Summarize adaptive-feedback pilots, sparse patches, and final production."""
    out_dir = Path(out_dir)
    driver_path = out_dir / "adaptive_feedback_driver_summary.json"
    driver = read_json_file(driver_path, {}) or {}
    rounds = list(driver.get("rounds", []) or [])
    round_summaries = []
    for r in rounds:
        rdir = Path(r.get("directory", ""))
        proposal = read_json_file(rdir / "adaptive_feedback_proposal.json", {}) or {}
        sparse_csv = r.get("adaptive_feedback_sparse_windows_2d_csv_for_next_pilot_or_final") or proposal.get("adaptive_feedback_explicit_window_candidates_csv", "") or proposal.get("sparse_patch_proposal", {}).get("explicit_window_candidates_csv", "")
        sparse_count = _count_csv_rows(Path(sparse_csv)) if sparse_csv else 0
        local_candidates = 0
        try:
            local_candidates = int((proposal.get("sparse_patch_proposal", {}) or {}).get("n_candidates", 0) or 0)
        except Exception:
            local_candidates = 0
        round_summaries.append({
            "round": int(r.get("round", len(round_summaries) + 1)),
            "directory": str(rdir),
            "production_steps": int(r.get("production_steps", 0) or 0),
            "old_n_windows": int(r.get("old_n_windows", proposal.get("old_n_windows", 0)) or 0),
            "new_n_windows": int(r.get("new_n_windows", proposal.get("new_n_windows", 0)) or 0),
            "converged": bool(proposal.get("converged", False)),
            "proposed_centers_A": r.get("proposed_centers_A", proposal.get("proposed_centers_A", [])),
            "proposed_secondary_cv_centers": r.get("proposed_secondary_cv_centers", proposal.get("proposed_secondary_cv_centers", [])),
            "sparse_candidate_csv": str(sparse_csv) if sparse_csv else "",
            "sparse_candidate_rows": int(sparse_count),
            "local_patch_candidates": int(local_candidates),
        })
    final_dir = Path(driver.get("final_production_dir", out_dir / "final_production"))
    final_report = read_json_file(final_dir / "final_report.json", {}) or {}
    exchange_report = read_json_file(final_dir / "exchange_tuning_report.json", {}) or {}
    validation = read_json_file(final_dir / "analysis_metadata_validation.json", {}) or {}
    payload = {
        "mode": "adaptive_feedback_human_summary",
        "driver_summary_json": str(driver_path),
        "target_neighbor_overlap": driver.get("target_neighbor_overlap"),
        "pilot_rounds_requested": driver.get("pilot_rounds_requested"),
        "pilot_rounds_summarized": len(round_summaries),
        "rounds": round_summaries,
        "stopped_early": bool(driver.get("stopped_early", False)),
        "final_production_dir": str(final_dir),
        "final_window_mode": driver.get("final_window_mode", ""),
        "final_windows_2d_csv": driver.get("final_windows_2d_csv", ""),
        "final_report_status": final_report.get("status", "unknown"),
        "analysis_validation_status": validation.get("status", "unknown"),
        "exchange_tuning_status": exchange_report.get("status", "unknown"),
        "exchange_acceptance_fraction": exchange_report.get("acceptance_fraction"),
        "recommended_final_analysis_dir": str(final_dir),
    }
    write_json(out_dir / "adaptive_feedback_summary.json", _json_ready(payload))
    lines = ["# Adaptive-feedback summary", "", f"Requested pilot rounds: **{driver.get('pilot_rounds_requested', '?')}**", f"Target neighbor overlap: **{driver.get('target_neighbor_overlap', '?')}**", f"Final production: `{final_dir}`", ""]
    if driver.get("stopped_early"):
        lines.append(f"Stopped early after round **{driver.get('stopped_after_round')}**: {driver.get('early_stop_reason', '')}")
        lines.append("")
    if round_summaries:
        lines.append("## Pilot rounds")
        lines.append("| round | steps | old windows | proposed windows | sparse explicit rows | local candidates | converged |")
        lines.append("|---:|---:|---:|---:|---:|---:|---|")
        for r in round_summaries:
            lines.append(f"| {r['round']} | {r['production_steps']} | {r['old_n_windows']} | {r['new_n_windows']} | {r['sparse_candidate_rows']} | {r['local_patch_candidates']} | {str(r['converged']).lower()} |")
        lines.append("")
        last = round_summaries[-1]
        lines.append("## Latest proposal")
        lines.append(f"Distance centers: `{', '.join(f'{float(x):.3g}' for x in (last.get('proposed_centers_A') or []))}`")
        if last.get("proposed_secondary_cv_centers"):
            lines.append(f"Secondary-CV centers: `{', '.join(f'{float(x):.3g}' for x in last.get('proposed_secondary_cv_centers'))}`")
        if last.get("sparse_candidate_csv"):
            lines.append(f"Sparse explicit candidate table: `{last.get('sparse_candidate_csv')}`")
        lines.append("")
    lines.append("## Final production readiness")
    lines.append(f"Final window mode: `{driver.get('final_window_mode', 'unknown')}`")
    if driver.get("final_windows_2d_csv"):
        lines.append(f"Final explicit 2D window table: `{driver.get('final_windows_2d_csv')}`")
    lines.append(f"Final report status: **{payload['final_report_status']}**")
    lines.append(f"Analysis validation status: **{payload['analysis_validation_status']}**")
    lines.append(f"Exchange tuning status: **{payload['exchange_tuning_status']}**")
    if payload.get("exchange_acceptance_fraction") is not None:
        lines.append(f"Exchange acceptance: **{100.0*float(payload['exchange_acceptance_fraction']):.1f}%**")
    lines.append("")
    lines.append("Use `final_production/` for downstream PMF/MBAR unless you intentionally analyze pilot diagnostics.")
    (out_dir / "adaptive_feedback_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _adaptive_feedback_read_grouped_samples(samples_csv: Path, n_windows: int, value_key: str = "cv_A", npz_key: Optional[str] = None) -> dict[int, list[float]]:
    """Read production samples grouped by umbrella window for one scalar column."""
    grouped = {i: [] for i in range(int(n_windows))}
    samples_csv = Path(samples_csv)
    npz_key = str(npz_key or value_key)
    arrays_npz = samples_csv.with_name("analysis_arrays.npz")
    if arrays_npz.exists() and arrays_npz.stat().st_size > 0:
        try:
            with np.load(arrays_npz, allow_pickle=False) as data:
                if npz_key in data:
                    windows = np.asarray(data["window"], dtype=np.int64)
                    values = np.asarray(data[npz_key], dtype=np.float64)
                    good = np.isfinite(values) & (windows >= 0) & (windows < int(n_windows))
                    for w in range(int(n_windows)):
                        grouped[w] = values[good & (windows == w)].astype(float).tolist()
                    return grouped
        except Exception:
            pass
    if not samples_csv.exists() or samples_csv.stat().st_size <= 0:
        return grouped
    try:
        with samples_csv.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    w = int(row.get("window", -1))
                    value = float(row.get(value_key, "nan"))
                except Exception:
                    continue
                if 0 <= w < int(n_windows) and math.isfinite(value):
                    grouped.setdefault(w, []).append(value)
    except Exception:
        return grouped
    return grouped


def _adaptive_feedback_read_samples(samples_csv: Path, n_windows: int) -> dict[int, list[float]]:
    """Read production distance-CV samples grouped by umbrella window."""
    return _adaptive_feedback_read_grouped_samples(samples_csv, n_windows, value_key="cv_A", npz_key="cv_A")


def _adaptive_feedback_read_secondary_samples(samples_csv: Path, n_windows: int) -> dict[int, list[float]]:
    """Read production secondary-CV samples grouped by umbrella window."""
    return _adaptive_feedback_read_grouped_samples(samples_csv, n_windows, value_key="secondary_cv", npz_key="secondary_cv")


def adaptive_secondary_default_centers(args) -> list[float]:
    """Initial secondary-CV ladder used by adaptive-feedback when none is supplied."""
    mode = secondary_cv_mode(args)
    if mode == "alpha-coil-beta":
        return [-0.80, 0.0, 0.80]
    return [0.25, 0.55, 0.85]


def _unique_axis_values(values, ndigits: int = 4) -> list[float]:
    out = sorted({round(float(x), int(ndigits)) for x in values if math.isfinite(float(x))})
    return [float(x) for x in out]


def _axis_index(value: float, centers: list[float]) -> int:
    return int(min(range(len(centers)), key=lambda i: abs(float(centers[i]) - float(value))))


def _group_window_samples_by_axis(samples_by_window: dict[int, list[float]], window_axis_values, axis_centers: list[float]) -> dict[int, list[float]]:
    grouped = {i: [] for i in range(len(axis_centers))}
    axis_vals = [float(x) for x in np.asarray(window_axis_values, dtype=float).tolist()]
    for w, vals in samples_by_window.items():
        if int(w) < 0 or int(w) >= len(axis_vals):
            continue
        ai = _axis_index(axis_vals[int(w)], axis_centers)
        grouped.setdefault(ai, []).extend(float(x) for x in vals if math.isfinite(float(x)))
    return grouped


def _adaptive_feedback_finite_float_list(values) -> list[float]:
    """Return finite float values from an arbitrary iterable."""
    out = []
    for value in values or []:
        try:
            v = float(value)
        except Exception:
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def _adaptive_feedback_mean_sd(values) -> tuple[float, float]:
    vals = _adaptive_feedback_finite_float_list(values)
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def _adaptive_feedback_fraction_within(values, center: float, radius: float) -> float:
    vals = _adaptive_feedback_finite_float_list(values)
    if not vals or not math.isfinite(float(center)) or not math.isfinite(float(radius)) or float(radius) <= 0.0:
        return float("nan")
    c = float(center)
    r = float(radius)
    return float(sum(1 for v in vals if abs(float(v) - c) <= r) / max(1, len(vals)))


def _adaptive_feedback_harmonic_sigma(rt_kcal_mol: float, k_value: float) -> float:
    try:
        k = float(k_value)
    except Exception:
        return float("nan")
    if not math.isfinite(k) or k <= 0.0:
        return float("nan")
    return float(math.sqrt(max(0.0, float(rt_kcal_mol)) / k))


def _adaptive_feedback_radius_from_sigma(sigma: float, fallback: float, lo: float, hi: float, sigma_mult: float = 1.0) -> float:
    try:
        s = float(sigma)
    except Exception:
        s = float("nan")
    if not math.isfinite(s) or s <= 0.0:
        s = float(fallback)
    r = float(sigma_mult) * s
    return float(max(float(lo), min(float(hi), r)))


def _adaptive_feedback_2d_cell_diagnostics(
    samples_distance_by_window: dict[int, list[float]],
    samples_secondary_by_window: dict[int, list[float]],
    centers_arr,
    secondary_centers_arr,
    k_list,
    secondary_k_list,
    args,
) -> list[dict]:
    """Per-window 2D diagnostics without axis pooling.

    These rows intentionally preserve the actual rectangular-grid cell identity.
    They diagnose whether individual windows are sampled near their own 2D target,
    which catches local row/column failures that can be hidden by pooled 1D axis
    samples.
    """
    centers = np.asarray(centers_arr, dtype=float)
    secondary = np.asarray(secondary_centers_arr, dtype=float)
    k_primary = [float(x) for x in list(k_list)]
    k_secondary = [float(x) for x in list(secondary_k_list or [])]
    nwin = int(min(len(centers), len(secondary)))
    if nwin <= 0:
        return []
    rt = 0.00198720425864083 * float(getattr(args, "temperature_k", 300.0) or 300.0)
    min_samples = int(getattr(args, "adaptive_2d_min_cell_samples", 8) or 8)
    sigma_mult = float(getattr(args, "adaptive_2d_hit_sigma", 1.0) or 1.0)
    rows = []
    for w in range(nwin):
        dvals = _adaptive_feedback_finite_float_list(samples_distance_by_window.get(w, []))
        svals = _adaptive_feedback_finite_float_list(samples_secondary_by_window.get(w, []))
        paired_n = min(len(dvals), len(svals))
        dcenter = float(centers[w])
        scenter = float(secondary[w])
        kd = float(k_primary[w]) if w < len(k_primary) else float("nan")
        ks = float(k_secondary[w]) if w < len(k_secondary) else float("nan")
        dmean, dsd = _adaptive_feedback_mean_sd(dvals)
        smean, ssd = _adaptive_feedback_mean_sd(svals)
        dsigma = _adaptive_feedback_harmonic_sigma(rt, kd)
        ssigma = _adaptive_feedback_harmonic_sigma(rt, ks)
        dradius = _adaptive_feedback_radius_from_sigma(dsigma, fallback=(dsd if math.isfinite(dsd) and dsd > 0 else 0.75), lo=0.25, hi=2.0, sigma_mult=sigma_mult)
        sradius = _adaptive_feedback_radius_from_sigma(ssigma, fallback=(ssd if math.isfinite(ssd) and ssd > 0 else 0.15), lo=0.03, hi=0.50, sigma_mult=sigma_mult)
        dhit = _adaptive_feedback_fraction_within(dvals, dcenter, dradius)
        shit = _adaptive_feedback_fraction_within(svals, scenter, sradius)
        hit2d = float("nan")
        norm_offset = float("nan")
        cov_ds = float("nan")
        corr_ds = float("nan")
        if paired_n > 0:
            da = np.asarray(dvals[:paired_n], dtype=float)
            sa = np.asarray(svals[:paired_n], dtype=float)
            good = np.isfinite(da) & np.isfinite(sa)
            da = da[good]
            sa = sa[good]
            if da.size:
                hit2d = float(np.mean((np.abs(da - dcenter) <= dradius) & (np.abs(sa - scenter) <= sradius)))
                dz = (float(np.nanmean(da)) - dcenter) / dradius if dradius > 0 else float("nan")
                sz = (float(np.nanmean(sa)) - scenter) / sradius if sradius > 0 else float("nan")
                if math.isfinite(dz) and math.isfinite(sz):
                    norm_offset = float(math.sqrt(dz * dz + sz * sz))
                if da.size >= 2:
                    cov = np.cov(da, sa)
                    cov_ds = float(cov[0, 1])
                    if float(np.nanstd(da)) > 1.0e-12 and float(np.nanstd(sa)) > 1.0e-12:
                        corr_ds = float(np.corrcoef(da, sa)[0, 1])
        if paired_n < min_samples:
            status = "insufficient_samples"
        elif math.isfinite(hit2d) and hit2d < 0.10:
            status = "low_2d_hit"
        elif math.isfinite(norm_offset) and norm_offset > 1.75:
            status = "off_center"
        else:
            status = "ok"
        rows.append({
            "window": int(w),
            "center_A": dcenter,
            "secondary_cv_center": scenter,
            "k_kcal_mol_A2": kd,
            "secondary_cv_k_kcal_mol": ks,
            "n_distance_samples": int(len(dvals)),
            "n_secondary_samples": int(len(svals)),
            "n_paired_samples": int(paired_n),
            "distance_mean_A": dmean,
            "distance_sd_A": dsd,
            "secondary_mean": smean,
            "secondary_sd": ssd,
            "distance_mean_minus_center_A": float(dmean - dcenter) if math.isfinite(dmean) else float("nan"),
            "secondary_mean_minus_center": float(smean - scenter) if math.isfinite(smean) else float("nan"),
            "distance_sigma_A_from_k": dsigma,
            "secondary_sigma_from_k": ssigma,
            "distance_hit_radius_A": dradius,
            "secondary_hit_radius": sradius,
            "distance_hit_fraction": dhit,
            "secondary_hit_fraction": shit,
            "two_d_hit_fraction": hit2d,
            "normalized_center_offset": norm_offset,
            "distance_secondary_covariance": cov_ds,
            "distance_secondary_correlation": corr_ds,
            "cell_status": status,
        })
    return rows


def _adaptive_feedback_2d_edge_diagnostics(
    samples_distance_by_window: dict[int, list[float]],
    samples_secondary_by_window: dict[int, list[float]],
    centers_arr,
    secondary_centers_arr,
    exchange_stats: dict,
    n_primary: int,
    n_secondary: int,
    target_overlap: float,
    args,
) -> list[dict]:
    """Per-neighbor 2D diagnostics for rectangular distance x secondary grids.

    The current adaptive proposal pools samples by axis.  These edge rows keep the
    orthogonal slice identity, so a single broken row/column can be seen directly
    and later upgraded to a sparse local-patch window instead of a full cross-grid
    expansion.
    """
    centers = np.asarray(centers_arr, dtype=float)
    secondary = np.asarray(secondary_centers_arr, dtype=float)
    target_exchange_acceptance = 0.30
    low_exchange_cut = max(0.01, target_exchange_acceptance - 0.05)
    min_samples = int(getattr(args, "adaptive_2d_min_edge_samples", 8) or 8)
    rows = []

    def _overlap_for(vals_i, vals_j, ci, cj):
        vi = _adaptive_feedback_finite_float_list(vals_i)
        vj = _adaptive_feedback_finite_float_list(vals_j)
        base = vi + vj + [float(ci), float(cj)]
        finite = [x for x in base if math.isfinite(float(x))]
        if len(vi) < min_samples or len(vj) < min_samples or len(finite) < 2:
            return float("nan")
        lo = min(finite)
        hi = max(finite)
        return _adaptive_hist_overlap(vi, vj, lo, hi)

    def _add_edge(wi: int, wj: int, edge_type: str, primary_pair_index: int, secondary_pair_index: int, primary_slice_index: int, secondary_slice_index: int) -> None:
        d_i = _adaptive_feedback_finite_float_list(samples_distance_by_window.get(wi, []))
        d_j = _adaptive_feedback_finite_float_list(samples_distance_by_window.get(wj, []))
        s_i = _adaptive_feedback_finite_float_list(samples_secondary_by_window.get(wi, []))
        s_j = _adaptive_feedback_finite_float_list(samples_secondary_by_window.get(wj, []))
        d_ov = _overlap_for(d_i, d_j, centers[wi], centers[wj])
        s_ov = _overlap_for(s_i, s_j, secondary[wi], secondary[wj])
        attempts, accepted, acc = _adaptive_feedback_exchange_acceptance(exchange_stats, wi, wj)
        if edge_type == "distance_axis":
            decision_overlap = d_ov
            orthogonal_overlap = s_ov
        else:
            decision_overlap = s_ov
            orthogonal_overlap = d_ov
        insufficient = min(len(d_i), len(d_j), len(s_i), len(s_j)) < min_samples
        low_overlap = math.isfinite(decision_overlap) and decision_overlap < float(target_overlap)
        low_exchange = math.isfinite(acc) and acc < low_exchange_cut
        if insufficient:
            status = "insufficient_samples"
        elif low_overlap and low_exchange:
            status = "low_overlap_low_exchange"
        elif low_overlap:
            status = "low_overlap"
        elif low_exchange:
            status = "low_exchange"
        else:
            status = "ok"
        patch_center_a = 0.5 * (float(centers[wi]) + float(centers[wj]))
        patch_secondary = 0.5 * (float(secondary[wi]) + float(secondary[wj]))
        rows.append({
            "edge": f"{min(int(wi), int(wj))}-{max(int(wi), int(wj))}",
            "window_i": int(wi),
            "window_j": int(wj),
            "edge_type": str(edge_type),
            "primary_pair_index": int(primary_pair_index),
            "secondary_pair_index": int(secondary_pair_index),
            "primary_slice_index": int(primary_slice_index),
            "secondary_slice_index": int(secondary_slice_index),
            "center_i_A": float(centers[wi]),
            "center_j_A": float(centers[wj]),
            "secondary_i": float(secondary[wi]),
            "secondary_j": float(secondary[wj]),
            "n_distance_i": int(len(d_i)),
            "n_distance_j": int(len(d_j)),
            "n_secondary_i": int(len(s_i)),
            "n_secondary_j": int(len(s_j)),
            "distance_overlap": float(d_ov) if math.isfinite(d_ov) else float("nan"),
            "secondary_overlap": float(s_ov) if math.isfinite(s_ov) else float("nan"),
            "decision_overlap": float(decision_overlap) if math.isfinite(decision_overlap) else float("nan"),
            "orthogonal_overlap": float(orthogonal_overlap) if math.isfinite(orthogonal_overlap) else float("nan"),
            "target_overlap": float(target_overlap),
            "exchange_attempts": int(attempts),
            "exchange_accepted": int(accepted),
            "exchange_acceptance": float(acc),
            "low_exchange_cut": float(low_exchange_cut),
            "local_patch_center_A": float(patch_center_a),
            "local_patch_secondary_cv_center": float(patch_secondary),
            "edge_status": status,
            "recommendation": "keep" if status == "ok" else "diagnose_local_patch_or_axis_refinement",
        })

    for ip in range(int(n_primary)):
        for js in range(int(n_secondary) - 1):
            wi = ip * int(n_secondary) + js
            wj = wi + 1
            _add_edge(wi, wj, "secondary_axis", primary_pair_index=-1, secondary_pair_index=js, primary_slice_index=ip, secondary_slice_index=js)
    for ip in range(int(n_primary) - 1):
        for js in range(int(n_secondary)):
            wi = ip * int(n_secondary) + js
            wj = (ip + 1) * int(n_secondary) + js
            _add_edge(wi, wj, "distance_axis", primary_pair_index=ip, secondary_pair_index=-1, primary_slice_index=ip, secondary_slice_index=js)
    return rows


_AXIS_EDGE_TYPES = frozenset({"distance_axis", "secondary_axis"})


def _adaptive_feedback_2d_annotate_locality(edge_rows: list[dict], args) -> tuple[list[dict], list[dict]]:
    """Annotate bad 2D edges as local-patch or full-axis refinement candidates.

    Pure axis edges (distance_axis, secondary_axis) are grouped by interval and
    may be promoted to full_axis_refinement_candidate when a large fraction of
    the interval is bad.  Non-axis edges (knn_geometry, delaunay_bridge, compound
    axis+knn types) are annotated per-edge and always map to local_patch_candidate.
    """
    if not edge_rows:
        return [], []
    threshold = float(getattr(args, "adaptive_2d_global_bad_fraction", 0.50) or 0.50)
    threshold = max(0.0, min(1.0, threshold))

    groups: dict[tuple[str, int], list[int]] = {}
    non_axis_indices: list[int] = []

    for idx, row in enumerate(edge_rows):
        pure_types = set(str(row.get("edge_type", "")).split("+")) - {""}
        if pure_types and pure_types <= _AXIS_EDGE_TYPES:
            # Pure axis edge — group by interval for bad-fraction logic
            et = "distance_axis" if "distance_axis" in pure_types else "secondary_axis"
            key_field = "primary_pair_index" if et == "distance_axis" else "secondary_pair_index"
            key = (et, int(row.get(key_field, -1)))
            groups.setdefault(key, []).append(idx)
        else:
            # Non-axis (knn_geometry, delaunay_bridge, compound, unknown): annotate per-edge
            non_axis_indices.append(idx)

    annotated = [dict(r) for r in edge_rows]
    defects = []
    bad_statuses = {"low_overlap", "low_exchange", "low_overlap_low_exchange"}
    defect_statuses = bad_statuses | {"insufficient_samples"}

    for key, indices in groups.items():
        bad = [i for i in indices if str(annotated[i].get("edge_status")) in bad_statuses]
        bad_fraction = float(len(bad) / max(1, len(indices)))
        if bad:
            scope = "full_axis_refinement_candidate" if bad_fraction >= threshold else "local_patch_candidate"
        else:
            scope = "needs_more_samples" if any(str(annotated[i].get("edge_status")) == "insufficient_samples" for i in indices) else "none"
        for i in indices:
            annotated[i]["bad_edges_in_axis_interval"] = int(len(bad))
            annotated[i]["total_edges_in_axis_interval"] = int(len(indices))
            annotated[i]["bad_fraction_in_axis_interval"] = float(bad_fraction)
            annotated[i]["defect_scope"] = scope if str(annotated[i].get("edge_status")) in defect_statuses else "none"
            if str(annotated[i].get("edge_status")) in defect_statuses:
                rec = scope
                if scope == "local_patch_candidate":
                    rec = "add_or_test_local_midpoint_patch_only"
                elif scope == "full_axis_refinement_candidate":
                    rec = "prefer_full_axis_center_addition"
                annotated[i]["recommendation"] = rec
                defects.append(dict(annotated[i]))

    for i in non_axis_indices:
        annotated[i]["bad_edges_in_axis_interval"] = 0
        annotated[i]["total_edges_in_axis_interval"] = 1
        annotated[i]["bad_fraction_in_axis_interval"] = float("nan")
        status = str(annotated[i].get("edge_status", ""))
        if status in defect_statuses:
            annotated[i]["defect_scope"] = "local_patch_candidate"
            annotated[i]["recommendation"] = "add_or_test_local_midpoint_patch_only"
            defects.append(dict(annotated[i]))
        else:
            annotated[i]["defect_scope"] = "none"

    return annotated, defects


def _adaptive_feedback_secondary_k_from_centers(centers: list[float], args) -> list[float]:
    """Return secondary-CV k values for adaptive-feedback proposed centers.

    Historical behavior was fixed-k for every secondary target.  When
    --secondary-cv-k-mode spacing/adaptive is selected, this now mirrors the
    primary CV's spacing-derived k assignment in dimensionless secondary-CV units.
    """
    return adaptive_secondary_force_constants_kcal(centers, args)


def _adaptive_feedback_axis_proposal(axis_name: str, centers: list[float], samples_by_center: dict[int, list[float]], args, target_overlap: float, center_min: Optional[float] = None, center_max: Optional[float] = None, is_secondary: bool = False, exchange_stats_by_pair: Optional[dict] = None, seed_offset: int = 0) -> dict:
    """Small, robust 1D adaptive proposal used for each axis of a 2D grid."""
    centers = [float(x) for x in sorted(float(c) for c in centers)]
    n = len(centers)
    if n == 0:
        return {"axis": axis_name, "old_centers": [], "proposed_centers": [], "diagnostics": [], "pair_rows": [], "changed": False, "converged": True}
    if n == 1:
        return {"axis": axis_name, "old_centers": centers, "proposed_centers": centers, "diagnostics": [], "pair_rows": [], "changed": False, "converged": True}

    aggr = _adaptive_window_aggressiveness_settings(args)
    contact_axis = (not bool(is_secondary)) and primary_cv_is_contacts(args)
    target_overlap = max(0.04, min(0.95, float(target_overlap)))
    severe_overlap = min(0.10, max(0.01, float(aggr.get("severe_overlap_factor", 0.33)) * target_overlap))
    high_overlap = min(0.95, max(target_overlap + float(aggr.get("high_overlap_offset", 0.25)), float(aggr.get("high_overlap_factor", 1.65)) * target_overlap))
    max_add_per_round = max(1, min(int(aggr.get("max_add_cap", 4)), max(1, int(math.ceil(n * float(aggr.get("max_add_fraction", 0.34)))))))
    max_remove_per_round = max(1, min(int(aggr.get("max_remove_cap", 3)), max(1, int(math.ceil(n * float(aggr.get("max_remove_fraction", 0.25)))))))
    min_samples_per_center = 8
    spacings = np.diff(np.asarray(centers, dtype=float))
    med_spacing = float(np.median(spacings)) if spacings.size else 1.0
    if contact_axis:
        hit_radius = max(1.0e-5, float(getattr(args, "contact_adaptive_hit_radius", 0.08) or 0.08))
        shift_fraction = 0.35
        max_shift = max(1.0e-5, float(getattr(args, "contact_adaptive_max_center_shift", 0.08) or 0.08))
        min_spacing = max(1.0e-6, float(getattr(args, "contact_adaptive_min_new_spacing", 0.04) or 0.04))
    else:
        hit_radius = 0.75 if not is_secondary else max(0.08, 0.35 * med_spacing)
        shift_fraction = 0.30 if not is_secondary else 0.45
        max_shift = 0.75 if not is_secondary else max(0.10, 0.45 * med_spacing)
        min_spacing = float(aggr.get("min_new_spacing_A", 0.35)) if not is_secondary else max(0.05, 0.30 * med_spacing)
    target_exchange_acceptance = 0.30
    low_exchange_cut = max(0.01, target_exchange_acceptance - 0.05)
    minimum_valid_overlap = _minimum_valid_overlap(target_overlap)

    pair_rows = []
    add_candidates = []
    overlaps = {}
    for i in range(n - 1):
        va = [float(x) for x in samples_by_center.get(i, []) if math.isfinite(float(x))]
        vb = [float(x) for x in samples_by_center.get(i + 1, []) if math.isfinite(float(x))]
        vals = va + vb + [centers[i], centers[i + 1]]
        lo = min(vals)
        hi = max(vals)
        ov = _adaptive_hist_overlap(va, vb, lo, hi)
        overlaps[(i, i + 1)] = ov
        acc = float(exchange_stats_by_pair.get((i, i + 1), float("nan"))) if exchange_stats_by_pair else float("nan")
        overlap_ci = _adaptive_feedback_bootstrap_overlap_ci(va, vb, lo, hi, seed=seed_offset * 100 + i) if len(va) >= 5 and len(vb) >= 5 else {"overlap_ci_low": float("nan"), "overlap_ci_high": float("nan")}
        action = "keep"
        added = []
        if len(va) < min_samples_per_center or len(vb) < min_samples_per_center or not math.isfinite(ov):
            action = "insufficient samples; keep"
        elif ov < severe_overlap:
            added = [round((2.0 * centers[i] + centers[i + 1]) / 3.0, 4), round((centers[i] + 2.0 * centers[i + 1]) / 3.0, 4)]
            action = "candidate: add two centers; severe low overlap"
        elif ov < target_overlap:
            added = [round(0.5 * (centers[i] + centers[i + 1]), 4)]
            action = f"candidate: add midpoint; below overlap target {target_overlap:.2f}"
        elif math.isfinite(acc) and acc < low_exchange_cut and ov < high_overlap:
            added = [round(0.5 * (centers[i] + centers[i + 1]), 4)]
            action = f"candidate: add midpoint; low exchange acceptance {acc:.2f} on {axis_name}"
        for c in added:
            if center_min is not None and c < float(center_min):
                continue
            if center_max is not None and c > float(center_max):
                continue
            priority = float((target_overlap - ov) * 100.0 if math.isfinite(ov) else 10.0)
            if math.isfinite(acc):
                priority += max(0.0, target_exchange_acceptance - acc) * 25.0
            add_candidates.append({
                "center": float(c),
                "priority": priority,
                "left_index": int(i),
                "right_index": int(i + 1),
                "reason": action,
            })
        pair_rows.append({
            "axis": axis_name,
            "pair_index": int(i),
            "left_index": int(i),
            "right_index": int(i + 1),
            "left_center": float(centers[i]),
            "right_center": float(centers[i + 1]),
            "left_samples": int(len(va)),
            "right_samples": int(len(vb)),
            "overlap": float(ov) if math.isfinite(ov) else float("nan"),
            "overlap_ci_low": float(overlap_ci.get("overlap_ci_low", float("nan"))),
            "overlap_ci_high": float(overlap_ci.get("overlap_ci_high", float("nan"))),
            "exchange_acceptance": float(acc),
            "target_overlap": float(target_overlap),
            "action": action,
            "candidate_added_centers": ";".join(f"{x:.4f}" for x in added),
        })

    diagnostics = []
    for i, c in enumerate(centers):
        vals = [float(x) for x in samples_by_center.get(i, []) if math.isfinite(float(x))]
        arr = np.asarray(vals, dtype=float)
        mean = float(np.nanmean(arr)) if arr.size else float("nan")
        sd = float(np.nanstd(arr)) if arr.size else float("nan")
        hit = float(np.mean(np.abs(arr - c) <= hit_radius)) if arr.size else float("nan")
        diagnostics.append({
            "axis": axis_name,
            "index": int(i),
            "center": float(c),
            "n_samples": int(arr.size),
            "mean": mean,
            "sd": sd,
            "mean_minus_center": float(mean - c) if math.isfinite(mean) else float("nan"),
            "hit_radius": float(hit_radius),
            "hit_fraction": hit,
            "action": "keep",
        })

    # Conservative bypass removals on each axis independently.
    # Gate on bootstrap CI-low so pilot noise cannot cause removal on a lucky single snapshot.
    remove_candidates = []
    for i in range(1, n - 1):
        left_ov = overlaps.get((i - 1, i), float("nan"))
        right_ov = overlaps.get((i, i + 1), float("nan"))
        if not (math.isfinite(left_ov) and math.isfinite(right_ov) and min(left_ov, right_ov) >= high_overlap):
            continue
        left_vals = samples_by_center.get(i - 1, [])
        right_vals = samples_by_center.get(i + 1, [])
        if len(left_vals) < min_samples_per_center or len(right_vals) < min_samples_per_center:
            continue
        vals = left_vals + right_vals + [centers[i - 1], centers[i + 1]]
        lo_b = min(vals)
        hi_b = max(vals)
        bypass_ci = _adaptive_feedback_bootstrap_overlap_ci(left_vals, right_vals, lo_b, hi_b, seed=seed_offset * 100 + 1000 + i)
        bypass_ci_low = float(bypass_ci.get("overlap_ci_low", float("nan")))
        bypass_mean = float(bypass_ci.get("overlap_mean", float("nan")))
        if not math.isfinite(bypass_ci_low):
            bypass_ci_low = bypass_mean
        acc_left = float(exchange_stats_by_pair.get((i - 1, i), float("nan"))) if exchange_stats_by_pair else float("nan")
        acc_right = float(exchange_stats_by_pair.get((i, i + 1), float("nan"))) if exchange_stats_by_pair else float("nan")
        acc_ok = (not math.isfinite(acc_left) or acc_left >= low_exchange_cut) and (not math.isfinite(acc_right) or acc_right >= low_exchange_cut)
        if math.isfinite(bypass_ci_low) and bypass_ci_low >= minimum_valid_overlap and acc_ok:
            remove_candidates.append({
                "index": int(i),
                "center": float(centers[i]),
                "score": float(min(left_ov, right_ov) + bypass_mean if math.isfinite(bypass_mean) else min(left_ov, right_ov)),
                "reason": f"redundant on {axis_name}; bypass overlap CI-low {bypass_ci_low:.3f}",
            })

    removed = []
    blocked = set()
    for cand in sorted(remove_candidates, key=lambda d: (-float(d.get("score", 0.0)), int(d.get("index", 0)))):
        if len(removed) >= max_remove_per_round:
            break
        idx = int(cand["index"])
        if idx in blocked:
            continue
        removed.append(cand)
        blocked.update({idx - 1, idx, idx + 1})

    provisional = [c for i, c in enumerate(centers) if i not in {int(r["index"]) for r in removed}]

    selected_additions = []
    for cand in sorted(add_candidates, key=lambda d: (-float(d.get("priority", 0.0)), float(d.get("center", 0.0)))):
        if len(selected_additions) >= max_add_per_round:
            break
        c = float(cand["center"])
        if any(abs(c - x) < min_spacing for x in provisional):
            continue
        provisional.append(c)
        selected_additions.append(cand)

    # Gentle center shifts based on mean visitation.
    shifted = []
    for row in diagnostics:
        i = int(row["index"])
        c = float(row["center"])
        if i <= 0 or i >= n - 1 or any(abs(c - float(r["center"])) < 1.0e-6 for r in removed):
            continue
        delta = float(row.get("mean_minus_center", float("nan")))
        hit = float(row.get("hit_fraction", float("nan")))
        ns = int(row.get("n_samples", 0) or 0)
        if ns < min_samples_per_center or not (math.isfinite(delta) and math.isfinite(hit)):
            continue
        shift_threshold = max(0.5 * hit_radius, 0.35 * med_spacing) if contact_axis else max(0.12, 0.35 * med_spacing)
        if hit >= 0.20 or abs(delta) < shift_threshold:
            continue
        old = c
        raw = max(-max_shift, min(max_shift, shift_fraction * delta))
        lo = centers[i - 1] + min_spacing
        hi = centers[i + 1] - min_spacing
        if center_min is not None:
            lo = max(lo, float(center_min))
        if center_max is not None:
            hi = min(hi, float(center_max))
        newc = max(lo, min(hi, old + raw))
        min_shift = max(0.01, 0.20 * med_spacing) if contact_axis else 0.08 * max(1.0, med_spacing)
        if abs(newc - old) < min_shift:
            continue
        for j, x in enumerate(provisional):
            if abs(float(x) - old) < 1.0e-6:
                provisional[j] = round(float(newc), 4)
                shifted.append({"index": i, "old_center": old, "new_center": round(float(newc), 4), "reason": f"shift toward sampled mean {float(row['mean']):.4f}"})
                break

    proposed = sorted(_unique_axis_values(provisional))
    if contact_axis and bool(getattr(args, "contact_normalize", True)):
        cmin = max(0.0, float(getattr(args, "contact_adaptive_min", 0.0) or 0.0))
        cmax = min(1.0, float(getattr(args, "contact_adaptive_max", 0.80) or 0.80))
        proposed = sorted({round(max(cmin, min(cmax, float(x))), 6) for x in proposed})
        # Gap-fill: unconditional coverage pass that runs after the per-round add
        # cap.  Any gap wider than 1.5x target spacing gets a midpoint inserted so
        # the full configured range stays covered even when windows are removed.
        if len(proposed) >= 2:
            target_sp = max(1.0e-6, float(getattr(args, "contact_adaptive_target_spacing", 0.15) or 0.15))
            gap_fills = [
                round(0.5 * (proposed[k] + proposed[k + 1]), 4)
                for k in range(len(proposed) - 1)
                if proposed[k + 1] - proposed[k] > 1.5 * target_sp
            ]
            if gap_fills:
                proposed = sorted({round(max(cmin, min(cmax, float(x))), 6) for x in proposed + gap_fills})
    changed = (len(proposed) != len(centers)) or any(abs(a - b) > 1.0e-4 for a, b in zip(proposed, centers))
    unresolved = [r for r in pair_rows if math.isfinite(float(r.get("overlap", float("nan"))) ) and float(r.get("overlap", float("nan"))) < 0.90 * target_overlap]
    for r in removed:
        diagnostics[int(r["index"])] ["action"] = str(r.get("reason", "removed"))
    for a in selected_additions:
        pair_rows[int(a.get("left_index", 0))]["action"] = str(a.get("reason", "added"))
    for sh in shifted:
        diagnostics[int(sh["index"])] ["action"] = str(sh.get("reason", "shifted"))

    return {
        "axis": axis_name,
        "old_centers": [float(x) for x in centers],
        "proposed_centers": [float(x) for x in proposed],
        "diagnostics": diagnostics,
        "pair_rows": pair_rows,
        "selected_additions": selected_additions,
        "removed_centers": removed,
        "shifted_centers": shifted,
        "changed": bool(changed),
        "converged": bool((not changed) and not unresolved),
        "unresolved_pairs": unresolved,
    }


def _grid_neighbor_pairs_2d(n_primary: int, n_secondary: int, parity: int = 0) -> list[tuple[int, int]]:
    """Neighbor pairs on a rectangular primary x secondary grid."""
    pairs = []
    n_primary = int(n_primary)
    n_secondary = int(n_secondary)
    if n_primary <= 0 or n_secondary <= 0:
        return pairs
    for ip in range(n_primary):
        for js in range(n_secondary - 1):
            if ((ip + js) & 1) == (int(parity) & 1):
                a = ip * n_secondary + js
                pairs.append((a, a + 1))
    for ip in range(n_primary - 1):
        for js in range(n_secondary):
            if ((ip + js + 1) & 1) == (int(parity) & 1):
                a = ip * n_secondary + js
                pairs.append((a, a + n_secondary))
    return pairs





def _adaptive_feedback_safe_float(value, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)

def _adaptive_feedback_nearest_k_value(center: float, centers: list[float], k_values: list[float], default: float) -> float:
    """Return the k value attached to the nearest center."""
    try:
        c = float(center)
    except Exception:
        return float(default)
    pairs = []
    for x, k in zip(list(centers or []), list(k_values or [])):
        try:
            xf = float(x)
            kf = float(k)
            if math.isfinite(xf) and math.isfinite(kf):
                pairs.append((abs(xf - c), kf))
        except Exception:
            continue
    if not pairs:
        return float(default)
    pairs.sort(key=lambda x: x[0])
    return float(pairs[0][1])


def _adaptive_feedback_2d_sparse_patch_candidates(
    local_defect_rows: list[dict],
    base_grid_rows: list[dict],
    proposed_primary: list[float],
    proposed_k: list[float],
    proposed_secondary: list[float],
    proposed_secondary_k: list[float],
    args,
    secondary_min: Optional[float] = None,
    secondary_max: Optional[float] = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Propose sparse local midpoint windows from 2D local-defect edges.

    This helper is intentionally diagnostic/proposal-only.  The current MD path
    still uses the rectangular crossed grid.  The returned explicit rows are a
    candidate future window table: base rectangular proposal plus local patches.
    """
    base_rows = []
    for i, row in enumerate(base_grid_rows or []):
        try:
            pc = float(row.get("primary_cv_center", row.get("primary_center_A", row.get("distance_center_A"))))
            pk = float(row.get("primary_cv_k_kcal", row.get("primary_k_kcal_mol_A2", row.get("distance_k_kcal_mol_A2"))))
            base_rows.append({
                "candidate_window": int(i),
                "window_type": row.get("window_type", "base_rectangular_grid"),
                "primary_cv_mode": str(row.get("primary_cv_mode", primary_cv_mode(args))),
                "primary_cv_center": float(pc),
                "primary_cv_k_kcal": float(pk),
                # Legacy aliases retained because much of the downstream MD path still stores the
                # primary coordinate in center_A-style arrays, even for non-distance CVs.
                "distance_center_A": float(pc),
                "distance_k_kcal_mol_A2": float(pk),
                "secondary_cv_center": float(row.get("secondary_cv_center")),
                "secondary_cv_k_kcal_mol": float(row.get("secondary_cv_k_kcal_mol")),
                "secondary_cv_mode": str(row.get("secondary_cv_mode", secondary_cv_mode(args))),
                "parent_edge": row.get("parent_edge", ""),
                "source_window_i": row.get("source_window_i", ""),
                "source_window_j": row.get("source_window_j", ""),
                "source_edge_type": row.get("source_edge_type", ""),
                "decision_overlap": row.get("decision_overlap", ""),
                "target_overlap": row.get("target_overlap", ""),
                "exchange_acceptance": row.get("exchange_acceptance", ""),
                "priority": row.get("priority", ""),
                "reason": row.get("reason", "base grid from adaptive-feedback proposal"),
                "patch_lifecycle": row.get("patch_lifecycle", "base_or_manual_window"),
                "use_in_current_production": False,
                "used_by_adaptive_feedback_next_pilot_when_available": row.get("used_by_adaptive_feedback_next_pilot_when_available", True),
                "used_by_adaptive_feedback_final_production": row.get("used_by_adaptive_feedback_final_production", True),
                "requires_future_explicit_2d_window_support": False,
            })
        except Exception:
            continue

    # Existing base-grid centers are skipped exactly/near-exactly.  Local patches
    # are allowed to sit between base centers, but duplicates of axis-added centers
    # should not be reported as additional sparse windows.
    base_keys = set()
    for row in base_rows:
        base_keys.add((round(float(row["distance_center_A"]), 4), round(float(row["secondary_cv_center"]), 4)))

    max_patches = int(getattr(args, "adaptive_2d_max_local_patches", 12) or 12)
    max_patches = max(0, max_patches)
    bad_statuses = {"low_overlap", "low_exchange", "low_overlap_low_exchange"}
    candidates_by_key: dict[tuple[float, float], dict] = {}
    skipped_rows = []

    for defect in local_defect_rows or []:
        if str(defect.get("defect_scope")) != "local_patch_candidate":
            continue
        if str(defect.get("edge_status")) not in bad_statuses:
            continue
        try:
            pc = float(defect.get("local_patch_center_A"))
            sc = float(defect.get("local_patch_secondary_cv_center"))
        except Exception:
            continue
        if not (math.isfinite(pc) and math.isfinite(sc)):
            continue
        if secondary_min is not None and sc < float(secondary_min) - 1.0e-12:
            continue
        if secondary_max is not None and sc > float(secondary_max) + 1.0e-12:
            continue
        key = (round(pc, 4), round(sc, 4))
        if key in base_keys:
            skipped = dict(defect)
            skipped["skip_reason"] = "candidate already present in proposed rectangular base grid"
            skipped_rows.append(skipped)
            continue

        target = _adaptive_feedback_safe_float(defect.get("target_overlap"), float("nan"))
        ov = _adaptive_feedback_safe_float(defect.get("decision_overlap"), float("nan"))
        acc = _adaptive_feedback_safe_float(defect.get("exchange_acceptance"), float("nan"))
        low_cut = _adaptive_feedback_safe_float(defect.get("low_exchange_cut"), 0.25)
        priority = 0.0
        if math.isfinite(target) and math.isfinite(ov):
            priority += max(0.0, target - ov) * 100.0
        elif not math.isfinite(ov):
            priority += 5.0
        if math.isfinite(acc):
            priority += max(0.0, low_cut - acc) * 40.0
        status = str(defect.get("edge_status"))
        if status == "low_overlap_low_exchange":
            priority += 20.0
        elif status in {"low_overlap", "low_exchange"}:
            priority += 8.0

        row = candidates_by_key.get(key)
        source_edge = str(defect.get("edge", ""))
        if row is None:
            if primary_cv_is_contacts(args):
                dk_default = float(getattr(args, "contact_adaptive_default_k_kcal", getattr(args, "contact_adaptive_min_k_kcal", 25.0)) or 25.0)
            else:
                dk_default = float(getattr(args, "default_window_k_kcal_a2", 1.0) or 1.0)
            sk_default = float(getattr(args, "secondary_cv_k_kcal", 25.0) or 25.0)
            patch_pk = _adaptive_feedback_nearest_k_value(pc, proposed_primary, proposed_k, dk_default)
            patch_sk = _adaptive_feedback_nearest_k_value(sc, proposed_secondary, proposed_secondary_k, sk_default)
            row = {
                "candidate_window": -1,
                "window_type": "local_midpoint_patch_candidate",
                "primary_cv_mode": primary_cv_mode(args),
                "primary_cv_center": float(pc),
                "primary_cv_k_kcal": float(patch_pk),
                # Backward-compatible aliases.  In contact mode these are contact-CV values, not Angstroms.
                "distance_center_A": float(pc),
                "distance_k_kcal_mol_A2": float(patch_pk),
                "secondary_cv_mode": secondary_cv_mode(args),
                "secondary_cv_center": float(sc),
                "secondary_cv_k_kcal_mol": float(patch_sk),
                "parent_edge": source_edge,
                "source_window_i": int(defect.get("window_i", -1)),
                "source_window_j": int(defect.get("window_j", -1)),
                "source_edge_type": str(defect.get("edge_type", "")),
                "decision_overlap": ov,
                "target_overlap": target,
                "exchange_acceptance": acc,
                "priority": float(priority),
                "supporting_edge_count": 1,
                "supporting_edges": source_edge,
                "reason": f"local {status} on {defect.get('edge_type', '2d_edge')}; midpoint patch only, not full row/column",
                "use_in_current_production": False,
                "used_by_adaptive_feedback_next_pilot_when_available": True,
                "used_by_adaptive_feedback_final_production": True,
                "requires_future_explicit_2d_window_support": False,
            }
            candidates_by_key[key] = row
        else:
            # Merge duplicate suggestions and keep the strongest evidence.
            row["priority"] = max(float(row.get("priority", 0.0) or 0.0), float(priority))
            row["supporting_edge_count"] = int(row.get("supporting_edge_count", 1) or 1) + 1
            edges = [x for x in str(row.get("supporting_edges", "")).split(";") if x]
            if source_edge and source_edge not in edges:
                edges.append(source_edge)
            row["supporting_edges"] = ";".join(edges)

    ranked_patch_rows = sorted(candidates_by_key.values(), key=lambda r: float(r.get("priority", 0.0) or 0.0), reverse=True)
    adaptive_cap = int(getattr(args, "adaptive_max_windows", 0) or 0)
    if max_patches <= 0:
        allowed_patches = 0
    else:
        allowed_patches = max_patches
    if adaptive_cap > 0:
        allowed_patches = min(allowed_patches, max(0, adaptive_cap - len(base_rows)))
    # Honour the explicit total-window/replica budget (--max-total-windows) so
    # sparse patches cannot grow the final window count past the cap. Without
    # this, the budget would bound only the factorized base grid and patches
    # could leak past it. No-op when no total budget is set (max_total == 0).
    _max_total_windows = int(_adaptive_total_window_limits(args)[1] or 0)
    if _max_total_windows > 0:
        allowed_patches = min(allowed_patches, max(0, _max_total_windows - len(base_rows)))
    patch_rows = ranked_patch_rows[:allowed_patches]
    for row in ranked_patch_rows[allowed_patches:]:
        skipped = dict(row)
        skipped["skip_reason"] = "candidate dropped by --adaptive-max-windows or --adaptive-2d-max-local-patches cap"
        skipped_rows.append(skipped)
    for j, row in enumerate(patch_rows):
        row["candidate_window"] = int(len(base_rows) + j)
        row["patch_lifecycle"] = "new_patch"

    explicit_rows = [dict(r) for r in base_rows] + [dict(r) for r in patch_rows]
    return patch_rows, explicit_rows, skipped_rows


def _adaptive_feedback_2d_edge_diagnostics_from_explicit_edges(
    samples_distance_by_window: dict[int, list[float]],
    samples_secondary_by_window: dict[int, list[float]],
    centers_arr: np.ndarray,
    sec_arr: np.ndarray,
    exchange_stats: dict,
    target_overlap: float,
    args,
) -> list[dict]:
    """Per-edge diagnostics for arbitrary explicit/sparse 2D neighbor graphs."""
    edges = build_explicit_2d_neighbor_edges(centers_arr, sec_arr, args=args)
    rows = []
    for edge in edges:
        wi = int(edge.get("wi", -1))
        wj = int(edge.get("wj", -1))
        if wi < 0 or wj < 0 or wi >= len(centers_arr) or wj >= len(centers_arr):
            continue
        di = float(centers_arr[wi]); dj = float(centers_arr[wj])
        si = float(sec_arr[wi]); sj = float(sec_arr[wj])
        edge_type = str(edge.get("edge_type", "explicit_2d_graph"))
        primary_like = abs(dj - di) >= abs(sj - si)
        axis_values_i = samples_distance_by_window.get(wi, []) if primary_like else samples_secondary_by_window.get(wi, [])
        axis_values_j = samples_distance_by_window.get(wj, []) if primary_like else samples_secondary_by_window.get(wj, [])
        lo = min(di, dj) if primary_like else min(si, sj)
        hi = max(di, dj) if primary_like else max(si, sj)
        if hi <= lo:
            all_vals = [float(x) for x in axis_values_i + axis_values_j if math.isfinite(float(x))]
            if all_vals:
                lo = min(all_vals); hi = max(all_vals)
        span = max(1.0e-6, hi - lo)
        lo -= 0.25 * span
        hi += 0.25 * span
        overlap = _adaptive_hist_overlap(axis_values_i, axis_values_j, lo, hi, bins=80) if len(axis_values_i) >= 2 and len(axis_values_j) >= 2 else float("nan")
        attempts, accepted, acc = _adaptive_feedback_exchange_acceptance(exchange_stats, wi, wj)
        low_exchange_cut = float(getattr(args, "adaptive_feedback_low_exchange_cutoff", 0.08) or 0.08)
        min_edge_samples = int(getattr(args, "adaptive_2d_min_edge_samples", 5) or 5)
        if len(axis_values_i) < min_edge_samples or len(axis_values_j) < min_edge_samples:
            status = "insufficient_samples"
        elif math.isfinite(overlap) and overlap < target_overlap and math.isfinite(acc) and acc < low_exchange_cut:
            status = "low_overlap_low_exchange"
        elif math.isfinite(overlap) and overlap < target_overlap:
            status = "low_overlap"
        elif math.isfinite(acc) and acc < low_exchange_cut:
            status = "low_exchange"
        else:
            status = "ok"
        rows.append({
            "edge": f"{min(wi, wj)}-{max(wi, wj)}",
            "window_i": int(wi),
            "window_j": int(wj),
            "edge_type": edge_type,
            "center_i_A": float(di),
            "center_j_A": float(dj),
            "secondary_i": float(si),
            "secondary_j": float(sj),
            "local_patch_center_A": float(0.5 * (di + dj)),
            "local_patch_secondary_cv_center": float(0.5 * (si + sj)),
            "normalized_distance": float(edge.get("normalized_distance", float("nan"))),
            "n_i": int(len(axis_values_i)),
            "n_j": int(len(axis_values_j)),
            "decision_overlap": float(overlap),
            "target_overlap": float(target_overlap),
            "exchange_attempts": int(attempts),
            "exchange_accepted": int(accepted),
            "exchange_acceptance": float(acc),
            "low_exchange_cut": float(low_exchange_cut),
            "edge_status": status,
        })
    return rows


def _adaptive_feedback_existing_explicit_rows_for_proposal(centers_a, k_list, secondary_cv_centers, secondary_cv_k_kcal_list, secondary_cv_metadata: dict) -> list[dict]:
    """Return current explicit windows with lifecycle/source metadata preserved."""
    rows = []
    normalized = []
    if isinstance(secondary_cv_metadata, dict):
        normalized = list(secondary_cv_metadata.get("normalized_rows", []) or [])
    for i, (c, k) in enumerate(zip(list(centers_a), list(k_list))):
        meta = normalized[i] if i < len(normalized) and isinstance(normalized[i], dict) else {}
        wtype = str(meta.get("window_type", meta.get("parent", meta.get("source", "explicit_2d_window"))))
        lifecycle = "retained_patch" if "patch" in wtype.lower() else "base_or_manual_window"
        sec_val = float(secondary_cv_centers[i]) if secondary_cv_centers is not None and i < len(secondary_cv_centers) else ""
        sec_k_val = float(secondary_cv_k_kcal_list[i]) if secondary_cv_k_kcal_list is not None and i < len(secondary_cv_k_kcal_list) else ""
        row = {
            "candidate_window": int(i),
            "window_type": wtype,
            "patch_lifecycle": lifecycle,
            "primary_cv_mode": str(meta.get("primary_cv_mode", "")),
            "primary_cv_center": float(c),
            "primary_cv_k_kcal": float(k),
            # Legacy aliases retained for current production arrays.
            "distance_center_A": float(c),
            "distance_k_kcal_mol_A2": float(k),
            "secondary_cv_mode": str(meta.get("secondary_cv_mode", "")),
            "secondary_cv_center": sec_val,
            "secondary_cv_k_kcal_mol": sec_k_val,
            "parent_edge": str(meta.get("parent_edge", "")),
            "source_window_i": str(meta.get("source_window_i", "")),
            "source_window_j": str(meta.get("source_window_j", "")),
            "source_edge_type": str(meta.get("source_edge_type", "")),
            "decision_overlap": "",
            "target_overlap": "",
            "exchange_acceptance": "",
            "priority": "",
            "reason": "retained from active explicit 2D window table",
            "use_in_current_production": False,
            "used_by_adaptive_feedback_next_pilot_when_available": True,
            "used_by_adaptive_feedback_final_production": True,
            "requires_future_explicit_2d_window_support": False,
        }
        rows.append(row)
    return rows


def run_adaptive_feedback_dispatcher_2d_explicit_sparse(args, out_dir: Path, centers_a, k_list, exchange_stats: dict, secondary_cv_centers, secondary_cv_k_kcal_list, secondary_cv_metadata: dict, fallback_history_by_window: Optional[dict[int, list[float]]] = None) -> Optional[dict]:
    """Adaptive-feedback proposal for explicit sparse/non-rectangular 2D windows.

    Unlike the rectangular dispatcher, this keeps the active explicit 2D table and
    proposes additional local midpoint patches on the geometry neighbor graph.
    """
    out_dir = Path(out_dir)
    centers_arr = np.asarray(centers_a, dtype=float)
    sec_arr = np.asarray(secondary_cv_centers, dtype=float) if secondary_cv_centers is not None else np.asarray([], dtype=float)
    nwin = int(centers_arr.size)
    if nwin < 2 or sec_arr.size != nwin:
        return run_adaptive_feedback_dispatcher(args, out_dir, centers_a, k_list, exchange_stats, fallback_history_by_window=fallback_history_by_window)

    samples_distance_by_window = _adaptive_feedback_read_samples(out_dir / "samples.csv", nwin)
    samples_secondary_by_window = _adaptive_feedback_read_secondary_samples(out_dir / "samples.csv", nwin)
    total_samples = sum(len(v) for v in samples_distance_by_window.values())
    if total_samples == 0 and fallback_history_by_window:
        for w in range(nwin):
            samples_distance_by_window[w] = [float(x) for x in fallback_history_by_window.get(w, []) if math.isfinite(float(x))]
        total_samples = sum(len(v) for v in samples_distance_by_window.values())

    aggr = _adaptive_window_aggressiveness_settings(args)
    requested_target_overlap = float(getattr(args, "adaptive_feedback_target_overlap", 0.30))
    if not math.isfinite(requested_target_overlap) or requested_target_overlap <= 0.0:
        requested_target_overlap = 0.30
    target_overlap = max(0.04, min(0.95, requested_target_overlap * float(aggr.get("target_overlap_factor", 1.0))))

    sec_min = float((secondary_cv_metadata or {}).get("range_min", -1.0 if secondary_cv_is_transition(args) else 0.0))
    sec_max = float((secondary_cv_metadata or {}).get("range_max", 1.0))
    cell_rows_2d = _adaptive_feedback_2d_cell_diagnostics(
        samples_distance_by_window, samples_secondary_by_window,
        centers_arr, sec_arr, k_list, secondary_cv_k_kcal_list, args,
    )
    edge_rows_2d = _adaptive_feedback_2d_edge_diagnostics_from_explicit_edges(
        samples_distance_by_window, samples_secondary_by_window,
        centers_arr, sec_arr, exchange_stats, target_overlap, args,
    )
    edge_rows_2d, local_defect_rows_2d = _adaptive_feedback_2d_annotate_locality(edge_rows_2d, args)

    retained_rows = _adaptive_feedback_existing_explicit_rows_for_proposal(centers_arr, k_list, sec_arr, secondary_cv_k_kcal_list, secondary_cv_metadata)
    existing_primary = [float(x) for x in centers_arr]
    existing_secondary = [float(x) for x in sec_arr]
    sparse_patch_rows, explicit_candidate_rows, sparse_patch_skipped_rows = _adaptive_feedback_2d_sparse_patch_candidates(
        local_defect_rows_2d, retained_rows,
        existing_primary, [float(x) for x in k_list], existing_secondary,
        [float(x) for x in (secondary_cv_k_kcal_list or [float(getattr(args, "secondary_cv_k_kcal", 25.0) or 25.0)] * nwin)],
        args, secondary_min=sec_min, secondary_max=sec_max,
    )
    # Mark retained windows distinctly from newly proposed local patches.
    for row in explicit_candidate_rows:
        if str(row.get("patch_lifecycle", "")) == "":
            row["patch_lifecycle"] = "new_patch" if "patch" in str(row.get("window_type", "")).lower() else "base_or_manual_window"
        row["used_by_adaptive_feedback_next_pilot_when_available"] = True
        row["used_by_adaptive_feedback_final_production"] = True

    graph_summary = write_explicit_2d_neighbor_graph_files(out_dir, centers_arr, sec_arr, args=args, prefix="adaptive_feedback_explicit_2d_neighbor_graph")

    outputs = [
        (out_dir / "adaptive_feedback_2d_cells.csv", cell_rows_2d),
        (out_dir / "adaptive_feedback_2d_edges.csv", edge_rows_2d),
        (out_dir / "adaptive_feedback_2d_local_defects.csv", local_defect_rows_2d),
        (out_dir / "adaptive_feedback_sparse_patch_candidates.csv", sparse_patch_rows),
        (out_dir / "adaptive_feedback_sparse_patch_skipped.csv", sparse_patch_skipped_rows),
        (out_dir / "adaptive_feedback_explicit_window_candidates.csv", explicit_candidate_rows),
    ]
    for path, rows in outputs:
        with Path(path).open("w", newline="") as handle:
            fieldnames = list(rows[0].keys()) if rows else ["empty"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            if rows:
                writer.writerows(rows)

    unresolved_edges = [r for r in edge_rows_2d if str(r.get("edge_status")) in {"low_overlap", "low_exchange", "low_overlap_low_exchange", "insufficient_samples"}]
    converged = bool(len(sparse_patch_rows) == 0 and len(unresolved_edges) == 0)
    summary = {
        "mode": "adaptive-feedback-2d-explicit-sparse",
        "description": "Adaptive-feedback on an explicit sparse/non-rectangular 2D window table. Existing windows are retained with lifecycle metadata; local graph-edge defects can add sparse midpoint patches for the next pilot/final production without reverting to a full rectangular cross-product.",
        "total_samples": int(total_samples),
        "adaptive_window_aggressiveness": str(aggr.get("mode", "balanced")),
        "aggressiveness_settings": aggr,
        "requested_target_overlap": float(requested_target_overlap),
        "effective_target_overlap": float(target_overlap),
        "old_n_windows": int(nwin),
        "new_n_windows": int(len(explicit_candidate_rows)),
        "converged": bool(converged),
        "convergence_status": "converged" if converged else "continue_explicit_sparse_refinement",
        "old_centers_A": [float(x) for x in centers_arr],
        "old_secondary_cv_centers": [float(x) for x in sec_arr],
        "proposed_centers_A": [float(x) for x in centers_arr],
        "proposed_k_kcal_mol_A2": [float(x) for x in k_list],
        "proposed_secondary_cv_centers": [float(x) for x in sec_arr],
        "proposed_secondary_cv_k_kcal_mol": [float(x) for x in (secondary_cv_k_kcal_list or [])],
        "expanded_proposed_centers_A": [float(r["distance_center_A"]) for r in explicit_candidate_rows],
        "expanded_proposed_k_kcal_mol_A2": [float(r["distance_k_kcal_mol_A2"]) for r in explicit_candidate_rows],
        "expanded_proposed_secondary_cv_centers": [float(r["secondary_cv_center"]) for r in explicit_candidate_rows],
        "expanded_proposed_secondary_cv_k_kcal_mol": [float(r["secondary_cv_k_kcal_mol"]) for r in explicit_candidate_rows],
        "two_d_diagnostics": {
            "cell_rows": int(len(cell_rows_2d)),
            "edge_rows": int(len(edge_rows_2d)),
            "local_defect_rows": int(len(local_defect_rows_2d)),
            "unresolved_edge_rows": int(len(unresolved_edges)),
            "cells_csv": str(out_dir / "adaptive_feedback_2d_cells.csv"),
            "edges_csv": str(out_dir / "adaptive_feedback_2d_edges.csv"),
            "local_defects_csv": str(out_dir / "adaptive_feedback_2d_local_defects.csv"),
            "neighbor_graph_csv": graph_summary.get("edge_csv", ""),
            "neighbor_graph_json": graph_summary.get("edge_json", ""),
        },
        "two_d_sparse_patch_proposal": {
            "enabled_in_current_run": True,
            "used_by_adaptive_feedback_next_pilot_when_available": True,
            "used_by_adaptive_feedback_final_production_when_available": True,
            "base_rectangular_grid_windows": int(len(retained_rows)),
            "retained_existing_patch_windows": int(sum(1 for r in retained_rows if "patch" in str(r.get("window_type", "")).lower())),
            "patch_candidate_windows": int(len(sparse_patch_rows)),
            "skipped_duplicate_or_capped_patch_candidates": int(len(sparse_patch_skipped_rows)),
            "explicit_candidate_windows_if_enabled": int(len(explicit_candidate_rows)),
            "candidate_new_n_windows_if_enabled": int(len(explicit_candidate_rows)),
            "adaptive_max_windows": int(getattr(args, "adaptive_max_windows", 0) or 0),
            "max_patch_candidates": int(getattr(args, "adaptive_2d_max_local_patches", 12) or 12),
            "patch_candidates_csv": str(out_dir / "adaptive_feedback_sparse_patch_candidates.csv"),
            "skipped_patch_candidates_csv": str(out_dir / "adaptive_feedback_sparse_patch_skipped.csv"),
            "explicit_window_candidates_csv": str(out_dir / "adaptive_feedback_explicit_window_candidates.csv"),
            "proposal_json": str(out_dir / "adaptive_feedback_sparse_patch_proposal.json"),
        },
        "secondary_cv": _json_ready(secondary_cv_metadata or {}),
        "adaptive_feedback_explicit_window_candidates_csv": str(out_dir / "adaptive_feedback_explicit_window_candidates.csv"),
        "suggested_cli_fragment": f"--windows-2d-csv {out_dir / 'adaptive_feedback_explicit_window_candidates.csv'}",
    }
    write_json(out_dir / "adaptive_feedback_sparse_patch_proposal.json", _json_ready({
        "mode": "adaptive-feedback-2d-explicit-sparse-patch-candidates",
        "description": "Iterative sparse adaptive-feedback proposal. The explicit candidate table retains current windows and adds any new local midpoint patches selected from graph-edge diagnostics.",
        "active_explicit_windows": retained_rows,
        "local_patch_candidates": sparse_patch_rows,
        "skipped_patch_candidates": sparse_patch_skipped_rows,
        "explicit_window_candidates": explicit_candidate_rows,
        "neighbor_graph": graph_summary,
    }))
    write_json(out_dir / "adaptive_feedback_proposal.json", _json_ready(summary))
    print(f"Adaptive-feedback explicit sparse 2D proposal written to {out_dir / 'adaptive_feedback_proposal.json'}")
    print(f"    Explicit sparse graph: {graph_summary.get('n_edges', 0)} edges; local patch candidates: {len(sparse_patch_rows)}; candidate windows: {len(explicit_candidate_rows)}")
    return summary

def run_adaptive_feedback_dispatcher_2d(args, out_dir: Path, centers_a, k_list, exchange_stats: dict, secondary_cv_centers, secondary_cv_k_kcal_list, secondary_cv_metadata: dict, fallback_history_by_window: Optional[dict[int, list[float]]] = None) -> Optional[dict]:
    """Adaptive-feedback proposal for a distance x secondary-structure CV grid."""
    out_dir = Path(out_dir)
    centers_arr = np.asarray(centers_a, dtype=float)
    sec_arr = np.asarray(secondary_cv_centers, dtype=float) if secondary_cv_centers is not None else np.asarray([], dtype=float)
    if centers_arr.size < 2 or sec_arr.size < 2:
        return run_adaptive_feedback_dispatcher(args, out_dir, centers_a, k_list, exchange_stats, fallback_history_by_window=fallback_history_by_window)

    primary_centers = _unique_axis_values(centers_arr)
    secondary_centers = _unique_axis_values(sec_arr)
    n_primary = len(primary_centers)
    n_secondary = len(secondary_centers)
    nwin = int(centers_arr.size)
    if n_primary * n_secondary != nwin:
        if isinstance(secondary_cv_metadata, dict) and bool(secondary_cv_metadata.get("explicit_2d_windows", False)):
            return run_adaptive_feedback_dispatcher_2d_explicit_sparse(
                args, out_dir, centers_a, k_list, exchange_stats,
                secondary_cv_centers, secondary_cv_k_kcal_list, secondary_cv_metadata,
                fallback_history_by_window=fallback_history_by_window,
            )
        # Fallback for unusual non-rectangular input.
        return run_adaptive_feedback_dispatcher(args, out_dir, centers_a, k_list, exchange_stats, fallback_history_by_window=fallback_history_by_window)

    samples_distance_by_window = _adaptive_feedback_read_samples(out_dir / "samples.csv", nwin)
    samples_secondary_by_window = _adaptive_feedback_read_secondary_samples(out_dir / "samples.csv", nwin)
    total_samples = sum(len(v) for v in samples_distance_by_window.values())
    if total_samples == 0 and fallback_history_by_window:
        for w in range(nwin):
            samples_distance_by_window[w] = [float(x) for x in fallback_history_by_window.get(w, []) if math.isfinite(float(x))]
        total_samples = sum(len(v) for v in samples_distance_by_window.values())

    primary_samples = _group_window_samples_by_axis(samples_distance_by_window, centers_arr, primary_centers)
    secondary_samples = _group_window_samples_by_axis(samples_secondary_by_window, sec_arr, secondary_centers)

    aggr = _adaptive_window_aggressiveness_settings(args)
    requested_target_overlap = float(getattr(args, "adaptive_feedback_target_overlap", 0.30))
    if not math.isfinite(requested_target_overlap) or requested_target_overlap <= 0.0:
        requested_target_overlap = 0.30
    target_overlap = max(0.04, min(0.95, requested_target_overlap * float(aggr.get("target_overlap_factor", 1.0))))

    cell_rows_2d = _adaptive_feedback_2d_cell_diagnostics(
        samples_distance_by_window, samples_secondary_by_window,
        centers_arr, sec_arr, k_list, secondary_cv_k_kcal_list, args,
    )
    edge_rows_2d = _adaptive_feedback_2d_edge_diagnostics(
        samples_distance_by_window, samples_secondary_by_window,
        centers_arr, sec_arr, exchange_stats, n_primary, n_secondary,
        target_overlap, args,
    )
    edge_rows_2d, local_defect_rows_2d = _adaptive_feedback_2d_annotate_locality(edge_rows_2d, args)

    # Compute per-axis exchange acceptance using min over the orthogonal axis.
    # Min (not mean) ensures a localized failure in any row/column is not masked.
    secondary_axis_exchange: dict[tuple[int, int], float] = {}
    for js in range(n_secondary - 1):
        accs = []
        for ip in range(n_primary):
            wi = ip * n_secondary + js
            wj = ip * n_secondary + js + 1
            _, _, frac = _adaptive_feedback_exchange_acceptance(exchange_stats, wi, wj)
            if math.isfinite(frac):
                accs.append(frac)
        if accs:
            secondary_axis_exchange[(js, js + 1)] = float(min(accs))

    primary_axis_exchange: dict[tuple[int, int], float] = {}
    for ip in range(n_primary - 1):
        accs = []
        for js in range(n_secondary):
            wi = ip * n_secondary + js
            wj = (ip + 1) * n_secondary + js
            _, _, frac = _adaptive_feedback_exchange_acceptance(exchange_stats, wi, wj)
            if math.isfinite(frac):
                accs.append(frac)
        if accs:
            primary_axis_exchange[(ip, ip + 1)] = float(min(accs))

    primary_axis_name = "contact_fraction" if primary_cv_is_contacts(args) else "distance_A"
    if primary_cv_is_contacts(args) and bool(getattr(args, "contact_normalize", True)):
        _pcmin: Optional[float] = max(0.0, float(getattr(args, "contact_adaptive_min", 0.0) or 0.0))
        _pcmax: Optional[float] = min(1.0, float(getattr(args, "contact_adaptive_max", 0.80) or 0.80))
    else:
        _pcmin = None
        _pcmax = None
    primary = _adaptive_feedback_axis_proposal(
        primary_axis_name, primary_centers, primary_samples, args, target_overlap,
        center_min=_pcmin, center_max=_pcmax, is_secondary=False,
        exchange_stats_by_pair=primary_axis_exchange, seed_offset=1,
    )
    sec_min = float((secondary_cv_metadata or {}).get("range_min", -1.0 if secondary_cv_is_transition(args) else 0.0))
    sec_max = float((secondary_cv_metadata or {}).get("range_max", 1.0))
    secondary = _adaptive_feedback_axis_proposal(
        "secondary_cv", secondary_centers, secondary_samples, args, target_overlap,
        center_min=sec_min, center_max=sec_max, is_secondary=True,
        exchange_stats_by_pair=secondary_axis_exchange, seed_offset=2,
    )

    proposed_primary = [float(x) for x in primary["proposed_centers"]]
    proposed_secondary = [float(x) for x in secondary["proposed_centers"]]
    proposed_primary, proposed_secondary, total_budget_meta = _apply_total_window_budget_to_factorized_grid(
        args, proposed_primary, proposed_secondary
    )
    if total_budget_meta.get("notes"):
        primary.setdefault("total_budget_notes", []).extend(total_budget_meta.get("notes", []))
    proposed_k = _adaptive_feedback_k_from_centers(proposed_primary, args)
    proposed_secondary_k = _adaptive_feedback_secondary_k_from_centers(proposed_secondary, args)

    grid_rows = []
    expanded_centers = []
    expanded_k = []
    expanded_secondary = []
    expanded_secondary_k = []
    idx = 0
    for pc, pk in zip(proposed_primary, proposed_k):
        for sc, sk in zip(proposed_secondary, proposed_secondary_k):
            expanded_centers.append(float(pc))
            expanded_k.append(float(pk))
            expanded_secondary.append(float(sc))
            expanded_secondary_k.append(float(sk))
            grid_rows.append({
                "window": int(idx),
                "primary_cv_mode": primary_cv_mode(args),
                "primary_cv_center": float(pc),
                "primary_cv_k_kcal": float(pk),
                "primary_center_A": float(pc),
                "primary_k_kcal_mol_A2": float(pk),
                "secondary_cv_mode": secondary_cv_mode(args),
                "secondary_cv_center": float(sc),
                "secondary_cv_k_kcal_mol": float(sk),
            })
            idx += 1

    sparse_patch_rows, explicit_candidate_rows, sparse_patch_skipped_rows = _adaptive_feedback_2d_sparse_patch_candidates(
        local_defect_rows_2d, grid_rows,
        proposed_primary, proposed_k, proposed_secondary, proposed_secondary_k,
        args, secondary_min=sec_min, secondary_max=sec_max,
    )
    sparse_2d_enabled = bool(getattr(args, "sparse_2d_patches_enabled", True))
    if primary_cv_is_contacts(args):
        sparse_2d_enabled = sparse_2d_enabled and bool(getattr(args, "contact_sparse_2d_patches_enabled", True))
    if not sparse_2d_enabled:
        sparse_patch_rows = []
        explicit_candidate_rows = list(grid_rows)
        sparse_patch_skipped_rows = list(sparse_patch_skipped_rows) + [{
            "reason": "sparse explicit 2D patches disabled by configuration; using factorized primary x secondary proposal",
            "primary_cv": primary_cv_mode(args),
        }]

    # Write compact diagnostics and proposal tables.
    outputs = [
        (out_dir / "adaptive_feedback_primary_axis.csv", primary.get("diagnostics", [])),
        (out_dir / "adaptive_feedback_primary_pairs.csv", primary.get("pair_rows", [])),
        (out_dir / "adaptive_feedback_secondary_axis.csv", secondary.get("diagnostics", [])),
        (out_dir / "adaptive_feedback_secondary_pairs.csv", secondary.get("pair_rows", [])),
        (out_dir / "adaptive_feedback_2d_cells.csv", cell_rows_2d),
        (out_dir / "adaptive_feedback_2d_edges.csv", edge_rows_2d),
        (out_dir / "adaptive_feedback_2d_local_defects.csv", local_defect_rows_2d),
        (out_dir / "adaptive_feedback_sparse_patch_candidates.csv", sparse_patch_rows),
        (out_dir / "adaptive_feedback_sparse_patch_skipped.csv", sparse_patch_skipped_rows),
        (out_dir / "adaptive_feedback_explicit_window_candidates.csv", explicit_candidate_rows),
        (out_dir / "adaptive_feedback_proposal.csv", grid_rows),
    ]
    for path, rows in outputs:
        with Path(path).open("w", newline="") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

    axes_converged = bool(primary.get("converged", False) and secondary.get("converged", False))
    # Axis-level convergence alone is not enough in 2D: a rectangular grid can
    # look converged after pooling by axis while one local edge still needs a
    # sparse midpoint repair.  Keep pilot refinement alive when local patch
    # candidates exist, and let the adaptive-feedback driver use the explicit
    # sparse table for the final clean production run.
    converged = bool(axes_converged and len(sparse_patch_rows) == 0)
    windows_arg = " ".join(f"{x:.4g}" for x in proposed_primary)
    k_arg = " ".join(f"{x:.4g}" for x in proposed_k)
    ss_arg = " ".join(f"{x:.4g}" for x in proposed_secondary)
    summary = {
        "mode": "adaptive-feedback-2d",
        "primary_cv": primary_cv_mode(args),
        "primary_cv_label": primary_cv_label(args),
        "primary_cv_units": primary_cv_units(args),
        "primary_k_units": primary_k_units(args),
        "legacy_primary_cv_column_names": True,
        "description": "Adaptive-feedback on a 2D grid: primary-CV centers and secondary-structure CV centers are diagnosed and proposed independently, then crossed into the next production grid. In contact mode, proposed_centers_A/proposed_k_kcal_mol_A2 are compatibility aliases for contact centers and kcal/mol/CV^2 force constants.",
        "total_samples": int(total_samples),
        "adaptive_window_aggressiveness": str(aggr.get("mode", "balanced")),
        "aggressiveness_settings": aggr,
        "requested_target_overlap": float(requested_target_overlap),
        "effective_target_overlap": float(target_overlap),
        "old_n_primary_windows": int(n_primary),
        "old_n_secondary_centers": int(n_secondary),
        "old_n_windows": int(nwin),
        "new_n_primary_windows": int(len(proposed_primary)),
        "new_n_secondary_centers": int(len(proposed_secondary)),
        "new_n_windows": int(len(proposed_primary) * len(proposed_secondary)),
        "total_window_budget": total_budget_meta,
        "converged": bool(converged),
        "axes_converged": bool(axes_converged),
        "convergence_status": "converged" if converged else ("continue_sparse_local_patch_refinement" if len(sparse_patch_rows) > 0 else "continue"),
        "old_centers_A": [float(x) for x in primary_centers],
        "old_secondary_cv_centers": [float(x) for x in secondary_centers],
        "proposed_centers_A": proposed_primary,
        "proposed_k_kcal_mol_A2": proposed_k,
        "proposed_secondary_cv_centers": proposed_secondary,
        "proposed_secondary_cv_k_kcal_mol": proposed_secondary_k,
        "expanded_proposed_centers_A": expanded_centers,
        "expanded_proposed_k_kcal_mol_A2": expanded_k,
        "expanded_proposed_secondary_cv_centers": expanded_secondary,
        "expanded_proposed_secondary_cv_k_kcal_mol": expanded_secondary_k,
        "two_d_diagnostics": {
            "description": "Per-cell and per-neighbor diagnostics preserve local 2D failures that can be hidden by pooled axis diagnostics. Local defects can become sparse midpoint patch candidates that the adaptive-feedback driver consumes for final production.",
            "cell_rows": int(len(cell_rows_2d)),
            "edge_rows": int(len(edge_rows_2d)),
            "local_defect_rows": int(len(local_defect_rows_2d)),
            "cell_status_counts": {str(k): int(sum(1 for r in cell_rows_2d if str(r.get('cell_status')) == str(k))) for k in sorted({str(r.get('cell_status')) for r in cell_rows_2d})},
            "edge_status_counts": {str(k): int(sum(1 for r in edge_rows_2d if str(r.get('edge_status')) == str(k))) for k in sorted({str(r.get('edge_status')) for r in edge_rows_2d})},
            "defect_scope_counts": {str(k): int(sum(1 for r in local_defect_rows_2d if str(r.get('defect_scope')) == str(k))) for k in sorted({str(r.get('defect_scope')) for r in local_defect_rows_2d})},
            "cells_csv": str(out_dir / "adaptive_feedback_2d_cells.csv"),
            "edges_csv": str(out_dir / "adaptive_feedback_2d_edges.csv"),
            "local_defects_csv": str(out_dir / "adaptive_feedback_2d_local_defects.csv"),
        },
        "two_d_sparse_patch_proposal": {
            "description": "Candidate sparse midpoint windows derived from local 2D edge defects. In --window-mode adaptive-feedback, the automatic driver will use the explicit candidate table for the next pilot round and the final clean production run when patch candidates are present.",
            "enabled_in_current_run": bool(sparse_2d_enabled),
            "used_by_adaptive_feedback_next_pilot_when_available": True,
            "used_by_adaptive_feedback_final_production_when_available": True,
            "base_rectangular_grid_windows": int(len(grid_rows)),
            "patch_candidate_windows": int(len(sparse_patch_rows)),
            "skipped_duplicate_patch_candidates": int(len(sparse_patch_skipped_rows)),
            "explicit_candidate_windows_if_enabled": int(len(explicit_candidate_rows)),
            "candidate_new_n_windows_if_enabled": int(len(explicit_candidate_rows)),
            "max_patch_candidates": int(getattr(args, "adaptive_2d_max_local_patches", 12) or 12),
            "patch_candidates_csv": str(out_dir / "adaptive_feedback_sparse_patch_candidates.csv"),
            "skipped_patch_candidates_csv": str(out_dir / "adaptive_feedback_sparse_patch_skipped.csv"),
            "explicit_window_candidates_csv": str(out_dir / "adaptive_feedback_explicit_window_candidates.csv"),
            "proposal_json": str(out_dir / "adaptive_feedback_sparse_patch_proposal.json"),
            "next_implementation_step": "Already supported in this build: --window-mode adaptive-feedback will consume this explicit table for final production when patch candidates are present.",
        },
        "primary_axis": primary,
        "secondary_axis": secondary,
        "secondary_cv": _json_ready(secondary_cv_metadata or {}),
        "diagnostics_csv": str(out_dir / "adaptive_feedback_primary_axis.csv"),
        "secondary_diagnostics_csv": str(out_dir / "adaptive_feedback_secondary_axis.csv"),
        "proposal_csv": str(out_dir / "adaptive_feedback_proposal.csv"),
        "suggested_cli_fragment": (
            f"--window-mode manual --contact-centers {windows_arg} --contact-k-kcal {k_arg} --secondary-cv-centers {ss_arg}"
            if primary_cv_is_contacts(args) else
            f"--window-mode manual --windows-a {windows_arg} --window-k-kcal-a2 {k_arg} --secondary-cv-centers {ss_arg}"
        ),
    }
    write_json(out_dir / "adaptive_feedback_sparse_patch_proposal.json", _json_ready({
        "mode": "adaptive-feedback-2d-sparse-patch-candidates",
        "description": "Sparse local midpoint patches built from local 2D edge defects. The current pilot still uses its active window set, but --window-mode adaptive-feedback will consume this explicit candidate table for the next pilot and final clean production when patch candidates are present.",
        "enabled_in_current_run": bool(sparse_2d_enabled),
        "used_by_adaptive_feedback_next_pilot_when_available": True,
        "used_by_adaptive_feedback_final_production_when_available": True,
        "base_rectangular_grid": grid_rows,
        "local_patch_candidates": sparse_patch_rows,
        "skipped_duplicate_patch_candidates": sparse_patch_skipped_rows,
        "explicit_window_candidates": explicit_candidate_rows,
        "base_rectangular_grid_windows": int(len(grid_rows)),
        "patch_candidate_windows": int(len(sparse_patch_rows)),
        "explicit_candidate_windows_if_enabled": int(len(explicit_candidate_rows)),
        "patch_candidates_csv": str(out_dir / "adaptive_feedback_sparse_patch_candidates.csv"),
        "explicit_window_candidates_csv": str(out_dir / "adaptive_feedback_explicit_window_candidates.csv"),
        "next_implementation_step": "This build already supports explicit 2D window input and graph-based exchange; adaptive-feedback final production can use these rows automatically.",
    }))
    write_json(out_dir / "adaptive_feedback_proposal.json", _json_ready(summary))
    print(f"Adaptive-feedback 2D proposal written to {out_dir / 'adaptive_feedback_proposal.json'}")
    print(f"    2D local diagnostics: {len(cell_rows_2d)} cells, {len(edge_rows_2d)} edges, {len(local_defect_rows_2d)} local defect rows")
    if sparse_2d_enabled:
        print(f"    Sparse 2D patch candidates: {len(sparse_patch_rows)} local windows; adaptive-feedback will use the explicit candidate table in the next pilot/final when candidates are present ({len(explicit_candidate_rows)} windows)")
    else:
        print(f"    Sparse 2D patches disabled; using factorized primary x secondary grid ({len(expanded_centers)} windows)")
    print(f"    {primary_cv_label(args)} centers {n_primary} -> {len(proposed_primary)}; secondary centers {n_secondary} -> {len(proposed_secondary)}; replicas {nwin} -> {len(expanded_centers)}")
    if total_budget_meta.get("enabled"):
        print(
            "    Total replica budget: "
            f"min={total_budget_meta.get('min_total_windows', 0) or 'none'}, "
            f"max={total_budget_meta.get('max_total_windows', 0) or 'none'}, "
            f"proposed={total_budget_meta.get('new_total_windows')}"
        )
        for note in total_budget_meta.get("notes", [])[:4]:
            print(f"      budget note: {note}")
    if primary.get("selected_additions"):
        _unit = primary_cv_units(args)
        _suffix = "" if _unit in {"", "dimensionless"} else f" {_unit}"
        print("    Added primary-CV centers: " + ", ".join(f"{float(a['center']):.3f}{_suffix}" for a in primary.get("selected_additions", [])))
    if secondary.get("selected_additions"):
        print("    Added secondary-CV centers: " + ", ".join(f"{float(a['center']):.3f}" for a in secondary.get("selected_additions", [])))
    if not converged:
        print(f"    Suggested next fragment: {summary['suggested_cli_fragment']}")
    else:
        print("    Adaptive-feedback 2D converged; proceeding with proposed fixed grid.")
    return summary


def _adaptive_feedback_exchange_acceptance(exchange_stats: dict, wi: int, wj: int) -> tuple[int, int, float]:
    key = f"{min(int(wi), int(wj))}-{max(int(wi), int(wj))}"
    st = (exchange_stats or {}).get("pairs", {}).get(key, {}) if isinstance(exchange_stats, dict) else {}
    attempts = int(st.get("attempts", 0) or 0)
    accepted = int(st.get("accepted", 0) or 0)
    frac = float(accepted) / float(attempts) if attempts > 0 else float("nan")
    return attempts, accepted, frac


def _minimum_valid_overlap(target_overlap: float, aggr: dict | None = None) -> float:
    """Minimum measured neighbor overlap a pruned/bypassed bridge may retain.

    Raised to a 0.20 band at the default target (was an effective 0.15 ceiling).
    Aggressive modes may still lower the floor via ``aggr`` overrides, but never
    below the 0.10 hard minimum implied by the default floor.
    """
    aggr = aggr or {}
    return max(
        float(aggr.get("minimum_valid_overlap_floor", 0.10)),
        min(
            float(aggr.get("minimum_valid_overlap_ceiling", 0.25)),
            float(aggr.get("minimum_valid_overlap_factor", 2.0 / 3.0)) * float(target_overlap),
        ),
    )


def _adaptive_feedback_region_key(left_center_a: float, right_center_a: float, bin_width_a: float = 1.0) -> str:
    """Stable-ish key for adaptive memory over a CV region.

    Window centers change between pilot rounds, so exact pair indices are not a
    good memory key.  This groups observations by approximate midpoint along the
    CV axis, letting later rounds remember that a region was repeatedly over- or
    under-resolved even if a midpoint window was added or removed.
    """
    lo = min(float(left_center_a), float(right_center_a))
    hi = max(float(left_center_a), float(right_center_a))
    mid = 0.5 * (lo + hi)
    bw = max(0.1, float(bin_width_a))
    mid_bin = round(mid / bw) * bw
    return f"mid_{mid_bin:.2f}A"


def _adaptive_feedback_get_memory(args) -> dict:
    """Return the driver-level adaptive-feedback memory attached to args."""
    mem = getattr(args, "adaptive_feedback_memory", None)
    if not isinstance(mem, dict):
        mem = {}
    mem.setdefault("regions", {})
    mem.setdefault("rounds", [])
    return mem


def _adaptive_feedback_ema(current: float, previous: float, alpha: float = 0.65) -> float:
    """Exponential moving average that tolerates NaN/missing values."""
    try:
        cur = float(current)
    except Exception:
        cur = float("nan")
    try:
        prev = float(previous)
    except Exception:
        prev = float("nan")
    if not math.isfinite(cur):
        return prev
    if not math.isfinite(prev):
        return cur
    return float(alpha * cur + (1.0 - alpha) * prev)


def _adaptive_feedback_update_memory(memory: dict, pair_rows: list[dict], target_overlap: float, target_exchange_acceptance: float, round_index: int) -> dict:
    """Update persistent adaptive-feedback memory after one pilot round.

    The goal is to converge toward the smallest useful replica set without
    thrashing: low-overlap regions are remembered, and over-resolved/high-
    exchange regions need repeated evidence before routine removal.
    """
    memory = dict(memory or {})
    regions = dict(memory.get("regions", {}) or {})
    low_cut = max(0.02, float(target_overlap) - 0.05)
    high_cut = min(0.95, float(target_overlap) + 0.10)
    for row in pair_rows or []:
        key = str(row.get("region_key") or "")
        if not key:
            continue
        reg = dict(regions.get(key, {}) or {})
        ov = float(row.get("overlap", float("nan")))
        acc = float(row.get("exchange_acceptance", float("nan")))
        prev_ov = float(reg.get("ema_overlap", float("nan")))
        prev_acc = float(reg.get("ema_exchange_acceptance", float("nan")))
        ema_ov = _adaptive_feedback_ema(ov, prev_ov, alpha=0.65)
        ema_acc = _adaptive_feedback_ema(acc, prev_acc, alpha=0.65)
        reg["observations"] = int(reg.get("observations", 0) or 0) + 1
        reg["last_round"] = int(round_index)
        reg["last_left_center_A"] = float(row.get("left_center_A", float("nan")))
        reg["last_right_center_A"] = float(row.get("right_center_A", float("nan")))
        reg["last_overlap"] = ov
        reg["last_exchange_acceptance"] = acc
        reg["ema_overlap"] = ema_ov
        reg["ema_exchange_acceptance"] = ema_acc
        reg["target_overlap"] = float(target_overlap)
        reg["target_exchange_acceptance"] = float(target_exchange_acceptance)

        if math.isfinite(ema_ov) and ema_ov < low_cut:
            reg["low_overlap_streak"] = int(reg.get("low_overlap_streak", 0) or 0) + 1
        else:
            reg["low_overlap_streak"] = 0

        acc_ok = (math.isfinite(ema_acc) and ema_acc >= float(target_exchange_acceptance)) or not math.isfinite(ema_acc)
        if math.isfinite(ema_ov) and ema_ov > high_cut and acc_ok:
            reg["high_redundancy_streak"] = int(reg.get("high_redundancy_streak", 0) or 0) + 1
        else:
            reg["high_redundancy_streak"] = 0

        if math.isfinite(ema_ov) and abs(ema_ov - float(target_overlap)) <= 0.05:
            reg["near_target_streak"] = int(reg.get("near_target_streak", 0) or 0) + 1
        else:
            reg["near_target_streak"] = 0
        regions[key] = reg

    memory["regions"] = regions
    memory["target_overlap"] = float(target_overlap)
    memory["target_exchange_acceptance"] = float(target_exchange_acceptance)
    memory["last_round"] = int(round_index)
    memory.setdefault("rounds", [])
    return memory


def _adaptive_feedback_k_from_centers(centers_a: list[float], args) -> list[float]:
    """Feedback-mode primary-CV k proposal from local spacing.

    Distance mode returns kcal/mol/A^2. Contact mode returns kcal/mol/CV^2.
    Legacy field names may still say k_kcal_mol_A2 for compatibility, but this
    function deliberately uses contact-specific clamps when the primary CV is
    nonlocal-contacts.
    """
    centers = np.asarray(sorted(float(x) for x in centers_a), dtype=float)
    if centers.size == 0:
        return []
    if primary_cv_is_contacts(args):
        return adaptive_contact_force_constants_kcal(centers, args)
    if centers.size == 1:
        return [float(getattr(args, "default_window_k_kcal_a2", 1.0))]
    rt_kcal_mol = 0.00198720425864083 * float(getattr(args, "temperature_k", 300.0))
    spacings = np.diff(centers)
    local = np.empty_like(centers)
    local[0] = spacings[0]
    local[-1] = spacings[-1]
    if centers.size > 2:
        local[1:-1] = 0.5 * (spacings[:-1] + spacings[1:])
    aggr = _adaptive_window_aggressiveness_settings(args)
    overlap_factor = float(aggr.get("k_overlap_factor", 1.75))
    min_k_floor = float(aggr.get("feedback_min_k_floor", 0.30))
    min_k = max(min_k_floor, float(getattr(args, "adaptive_min_k_kcal_a2", min_k_floor)))
    max_k = min(5.0, float(getattr(args, "adaptive_max_k_kcal_a2", 5.0)))
    if max_k < min_k:
        max_k = min_k
    ks = []
    for spacing in local:
        sigma = max(0.25, float(spacing) / overlap_factor)
        k = rt_kcal_mol / (sigma * sigma)
        ks.append(float(max(min_k, min(max_k, k))))
    return ks



def _adaptive_feedback_interval_density(samples_by_window: dict[int, list[float]], centers: list[float]) -> list[dict]:
    """Return simple per-interval CV coverage diagnostics."""
    rows = []
    if len(centers) < 2:
        return rows
    all_vals = []
    for vals in samples_by_window.values():
        all_vals.extend(float(x) for x in vals if math.isfinite(float(x)))
    if not all_vals:
        return rows
    for i in range(len(centers) - 1):
        lo = float(min(centers[i], centers[i + 1]))
        hi = float(max(centers[i], centers[i + 1]))
        width = max(1.0e-12, hi - lo)
        vals = [x for x in all_vals if lo <= x <= hi]
        density = float(len(vals)) / width
        rows.append({
            "interval": int(i),
            "left_window": int(i),
            "right_window": int(i + 1),
            "left_center_A": float(lo),
            "right_center_A": float(hi),
            "width_A": float(width),
            "n_samples": int(len(vals)),
            "density_samples_per_A": float(density),
        })
    finite = [r["density_samples_per_A"] for r in rows if math.isfinite(float(r.get("density_samples_per_A", float("nan")))) and float(r.get("density_samples_per_A", 0.0)) > 0.0]
    median_density = float(np.median(np.asarray(finite, dtype=float))) if finite else 0.0
    for r in rows:
        d = float(r.get("density_samples_per_A", 0.0))
        ratio = d / median_density if median_density > 0.0 else float("nan")
        r["median_density_samples_per_A"] = median_density
        r["density_ratio_to_median"] = float(ratio) if math.isfinite(ratio) else float("nan")
        r["coverage_action"] = "keep"
    return rows


def _adaptive_feedback_observed_score(pair_rows: list[dict], diagnostics: list[dict], coverage_rows: list[dict], n_windows: int, target_overlap: float, target_exchange_acceptance: float, window_count_weight: float = 0.05) -> dict:
    """Score the current pilot; lower is better.

    The score is deliberately simple and diagnostic, not a thermodynamic
    objective.  It lets the adaptive loop remember whether proposals are
    improving rather than applying local rules with no sense of progress.
    """
    overlap_low = 0.0
    overlap_high = 0.0
    exchange_low = 0.0
    for r in pair_rows or []:
        ov = float(r.get("decision_overlap", r.get("overlap", float("nan"))))
        if math.isfinite(ov):
            overlap_low += max(0.0, float(target_overlap) - ov) ** 2
            overlap_high += max(0.0, ov - (float(target_overlap) + 0.30)) ** 2
        acc = float(r.get("decision_exchange_acceptance", r.get("exchange_acceptance", float("nan"))))
        if math.isfinite(acc):
            exchange_low += max(0.0, float(target_exchange_acceptance) - acc) ** 2
    center_miss = 0.0
    low_sample = 0.0
    for r in diagnostics or []:
        n = int(r.get("n_samples", 0) or 0)
        if n < 8:
            low_sample += 1.0
        hit = float(r.get("hit_fraction_0p75A", float("nan")))
        delta = abs(float(r.get("mean_minus_center_A", float("nan"))))
        if math.isfinite(hit) and math.isfinite(delta):
            center_miss += max(0.0, 0.15 - hit) * max(0.0, delta - 0.5)
    coverage_penalty = 0.0
    for r in coverage_rows or []:
        ratio = float(r.get("density_ratio_to_median", float("nan")))
        width = float(r.get("width_A", 0.0) or 0.0)
        if math.isfinite(ratio) and ratio < 0.25 and width > 1.0:
            coverage_penalty += (0.25 - ratio) * width
    replica_count_penalty = float(max(0.0, window_count_weight)) * float(max(0, n_windows))
    components = {
        "overlap_low": 40.0 * overlap_low,
        "overlap_high": 2.0 * overlap_high,
        "exchange_low": 10.0 * exchange_low,
        "center_miss": 4.0 * center_miss,
        "coverage_gap": 4.0 * coverage_penalty,
        "low_sample": 0.5 * low_sample,
        "replica_count": replica_count_penalty,
    }
    total = float(sum(float(v) for v in components.values()))
    return {"total": total, "components": components}



def _adaptive_feedback_bootstrap_overlap_ci(
    values_a: list[float],
    values_b: list[float],
    lo: float,
    hi: float,
    bins: int = 80,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> dict:
    """Bootstrap uncertainty estimate for pairwise CV histogram overlap.

    Short adaptive pilots can make single overlap estimates noisy.  The lower
    confidence bound is used for replica pruning, while the point estimate can
    still guide additions.  This keeps the adaptive search biased toward the
    smallest ladder that is statistically connected, not just the smallest
    ladder suggested by one lucky pilot snapshot.
    """
    a = np.asarray([float(x) for x in values_a if math.isfinite(float(x))], dtype=float)
    b = np.asarray([float(x) for x in values_b if math.isfinite(float(x))], dtype=float)
    if a.size < 5 or b.size < 5:
        return {
            "overlap_mean": float("nan"),
            "overlap_ci_low": float("nan"),
            "overlap_ci_high": float("nan"),
            "overlap_bootstrap_n": 0,
        }
    lo = float(lo)
    hi = float(hi)
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        allv = np.concatenate([a, b])
        lo = float(np.nanmin(allv))
        hi = float(np.nanmax(allv))
    if hi <= lo:
        hi = lo + 1.0
    n_bootstrap = max(0, int(n_bootstrap))
    if n_bootstrap <= 0:
        ov = _adaptive_hist_overlap(a, b, lo, hi, bins=bins)
        return {
            "overlap_mean": float(ov),
            "overlap_ci_low": float(ov),
            "overlap_ci_high": float(ov),
            "overlap_bootstrap_n": 0,
        }
    rng = np.random.default_rng(int(seed))
    vals = []
    for _ in range(n_bootstrap):
        aa = a[rng.integers(0, a.size, size=a.size)]
        bb = b[rng.integers(0, b.size, size=b.size)]
        vals.append(_adaptive_hist_overlap(aa, bb, lo, hi, bins=bins))
    arr = np.asarray([x for x in vals if math.isfinite(float(x))], dtype=float)
    if arr.size == 0:
        return {
            "overlap_mean": float("nan"),
            "overlap_ci_low": float("nan"),
            "overlap_ci_high": float("nan"),
            "overlap_bootstrap_n": 0,
        }
    return {
        "overlap_mean": float(np.nanmean(arr)),
        "overlap_ci_low": float(np.nanpercentile(arr, 5.0)),
        "overlap_ci_high": float(np.nanpercentile(arr, 95.0)),
        "overlap_bootstrap_n": int(arr.size),
    }


def _adaptive_feedback_pair_overlap_ci(
    args,
    values_a: list[float],
    values_b: list[float],
    lo: float,
    hi: float,
    label_seed: int,
) -> dict:
    """Convenience wrapper with deterministic seeds for adaptive diagnostics."""
    n_bootstrap = int(getattr(args, "adaptive_feedback_bootstrap_samples", 200) or 200)
    seed = int(getattr(args, "seed", 0) or 0) + 1000003 * int(getattr(args, "adaptive_feedback_round_index", 0) or 0) + int(label_seed)
    return _adaptive_feedback_bootstrap_overlap_ci(values_a, values_b, lo, hi, bins=80, n_bootstrap=n_bootstrap, seed=seed)

def run_adaptive_feedback_dispatcher(args, out_dir: Path, centers_a, k_list, exchange_stats: dict, fallback_history_by_window: Optional[dict[int, list[float]]] = None) -> Optional[dict]:
    """Diagnose a pilot run and propose a cleaner fixed window set.

    The dispatcher now uses a memory-guided optimizer instead of one-shot local
    rules.  It can add windows, remove redundant windows, gently shift centers,
    and retune k values with bounded per-round changes.  It still never mutates
    a running production stage; the proposal is used for the next pilot or the
    final clean production stage.
    """
    out_dir = Path(out_dir)
    centers = [float(x) for x in np.asarray(centers_a, dtype=float).tolist()]
    ks_old = [float(x) for x in list(k_list)]
    nwin = len(centers)
    if nwin < 2:
        return None

    samples_by_window = _adaptive_feedback_read_samples(out_dir / "samples.csv", nwin)
    total_samples = sum(len(v) for v in samples_by_window.values())
    if total_samples == 0 and fallback_history_by_window:
        for w in range(nwin):
            samples_by_window[w] = [float(x) for x in fallback_history_by_window.get(w, []) if math.isfinite(float(x))]
        total_samples = sum(len(v) for v in samples_by_window.values())

    feedback_memory = _adaptive_feedback_get_memory(args)
    memory_regions = feedback_memory.get("regions", {}) if isinstance(feedback_memory, dict) else {}
    round_index = int(getattr(args, "adaptive_feedback_round_index", 0) or 0)

    contact_primary = primary_cv_is_contacts(args)
    primary_unit = primary_cv_units(args)
    primary_unit_suffix = "" if primary_unit in {"", "dimensionless"} else f" {primary_unit}"
    primary_axis_name = primary_cv_label(args)
    if contact_primary:
        hit_radius = max(1.0e-5, float(getattr(args, "contact_adaptive_hit_radius", 0.08) or 0.08))
        tight_hit_radius = max(1.0e-5, float(getattr(args, "contact_adaptive_tight_hit_radius", 0.04) or 0.04))
    else:
        hit_radius = 0.75
        tight_hit_radius = 0.50
    aggr = _adaptive_window_aggressiveness_settings(args)
    requested_target_overlap = float(getattr(args, "adaptive_feedback_target_overlap", 0.30))
    if not math.isfinite(requested_target_overlap) or requested_target_overlap <= 0.0:
        requested_target_overlap = 0.30
    target_overlap = requested_target_overlap * float(aggr.get("target_overlap_factor", 1.0))
    target_overlap = max(0.04, min(0.95, target_overlap))
    target_exchange_acceptance = 0.30
    severe_overlap = min(0.10, max(0.01, float(aggr.get("severe_overlap_factor", 0.33)) * target_overlap))
    # Minimum statistically supported overlap for a connected, least-replica
    # ladder.  Aggressive modes lower this bound so the optimizer can accept
    # a thinner but still explicitly tested bridge when pruning windows.
    minimum_valid_overlap = _minimum_valid_overlap(target_overlap, aggr)
    low_exchange_cut = max(0.01, target_exchange_acceptance - 0.05)
    high_overlap = min(0.95, max(target_overlap + float(aggr.get("high_overlap_offset", 0.25)), float(aggr.get("high_overlap_factor", 1.65)) * target_overlap))
    very_high_overlap = min(0.98, max(target_overlap + float(aggr.get("very_high_overlap_offset", 0.40)), high_overlap + 0.12))
    min_samples_per_window = 8
    min_center_hit_for_removal = float(aggr.get("min_center_hit_for_removal", 0.15))
    max_add_per_round = max(1, min(int(aggr.get("max_add_cap", 4)), max(1, int(math.ceil(nwin * float(aggr.get("max_add_fraction", 0.34)))))))
    max_remove_per_round = max(1, min(int(aggr.get("max_remove_cap", 3)), max(1, int(math.ceil(nwin * float(aggr.get("max_remove_fraction", 0.25)))))))
    max_shift_per_round = max(1, min(3, max(1, nwin // 4)))
    max_center_shift_A = max(1.0e-5, float(getattr(args, "contact_adaptive_max_center_shift", 0.08) or 0.08)) if contact_primary else 0.75
    max_k_factor = float(aggr.get("max_k_factor", 1.50))
    min_k_factor = 1.0 / max_k_factor
    remove_block_radius = int(aggr.get("remove_block_radius", 1))
    min_new_spacing_A = max(1.0e-6, float(getattr(args, "contact_adaptive_min_new_spacing", 0.04) or 0.04)) if contact_primary else float(aggr.get("min_new_spacing_A", 0.35))
    coverage_gap_ratio = float(aggr.get("coverage_gap_ratio", 0.25))
    coverage_gap_width_A = max(1.0e-6, float(getattr(args, "contact_adaptive_coverage_gap_width", 0.10) or 0.10)) if contact_primary else float(aggr.get("coverage_gap_width_A", 1.25))
    window_count_weight = float(aggr.get("window_count_weight", 0.05))

    old_center_keys = set(round(float(c), 4) for c in centers)
    diagnostics = []
    pair_rows = []
    coverage_rows = []
    overlaps: dict[tuple[int, int], float] = {}
    add_candidates: list[dict] = []
    proposal_reasons: dict[float, list[str]] = {round(float(c), 4): ["existing window"] for c in centers}

    # Pair agent: overlap/exchange diagnostics and candidate window insertions.
    for i in range(nwin - 1):
        va = samples_by_window.get(i, [])
        vb = samples_by_window.get(i + 1, [])
        lo = min(centers[i], centers[i + 1], min(va) if va else centers[i], min(vb) if vb else centers[i + 1])
        hi = max(centers[i], centers[i + 1], max(va) if va else centers[i], max(vb) if vb else centers[i + 1])
        ov = _adaptive_hist_overlap(va, vb, lo, hi)
        overlap_ci = _adaptive_feedback_pair_overlap_ci(args, va, vb, lo, hi, label_seed=7919 * (i + 1))
        attempts, accepted, acc = _adaptive_feedback_exchange_acceptance(exchange_stats, i, i + 1)
        region_key = _adaptive_feedback_region_key(centers[i], centers[i + 1])
        region_memory = memory_regions.get(region_key, {}) if isinstance(memory_regions, dict) else {}
        prev_ema_overlap = float(region_memory.get("ema_overlap", float("nan"))) if isinstance(region_memory, dict) else float("nan")
        prev_ema_acc = float(region_memory.get("ema_exchange_acceptance", float("nan"))) if isinstance(region_memory, dict) else float("nan")
        decision_overlap = _adaptive_feedback_ema(ov, prev_ema_overlap, alpha=0.65)
        decision_acc = _adaptive_feedback_ema(acc, prev_ema_acc, alpha=0.65)
        low_streak_prev = int(region_memory.get("low_overlap_streak", 0) or 0) if isinstance(region_memory, dict) else 0
        high_streak_prev = int(region_memory.get("high_redundancy_streak", 0) or 0) if isinstance(region_memory, dict) else 0
        action = "keep"
        added = []
        if (not math.isfinite(ov)) or len(va) < min_samples_per_window or len(vb) < min_samples_per_window:
            action = "insufficient samples; keep and inspect"
        elif decision_overlap < severe_overlap:
            added = [
                round((2.0 * centers[i] + centers[i + 1]) / 3.0, 4),
                round((centers[i] + 2.0 * centers[i + 1]) / 3.0, 4),
            ]
            action = "candidate: add two windows; severe low overlap"
        elif decision_overlap < target_overlap:
            added = [round(0.5 * (centers[i] + centers[i + 1]), 4)]
            action = f"candidate: add midpoint; below {target_overlap:.2f} effective overlap target"
            if low_streak_prev > 0:
                action += "; repeated region"
        elif math.isfinite(decision_acc) and decision_acc < low_exchange_cut and decision_overlap < high_overlap:
            added = [round(0.5 * (centers[i] + centers[i + 1]), 4)]
            action = f"candidate: add midpoint; below {target_exchange_acceptance:.2f} exchange target"
        elif decision_overlap >= high_overlap:
            action = "redundancy candidate; above target"
        for c in added:
            priority = 0.0
            if math.isfinite(decision_overlap):
                priority += max(0.0, target_overlap - decision_overlap) * 100.0
            if math.isfinite(decision_acc):
                priority += max(0.0, target_exchange_acceptance - decision_acc) * 25.0
            priority += 10.0 * max(0, low_streak_prev)
            if decision_overlap < severe_overlap:
                priority += 50.0
            add_candidates.append({
                "center_A": float(c),
                "priority": float(priority),
                "source": "pair_overlap_exchange",
                "left_window": int(i),
                "right_window": int(i + 1),
                "reason": action,
            })
        overlaps[(i, i + 1)] = ov
        pair_rows.append({
            "pair_index": i,
            "left_window": i,
            "right_window": i + 1,
            "left_center_A": centers[i],
            "right_center_A": centers[i + 1],
            "overlap": ov,
            "overlap_bootstrap_mean": overlap_ci.get("overlap_mean", float("nan")),
            "overlap_ci_low": overlap_ci.get("overlap_ci_low", float("nan")),
            "overlap_ci_high": overlap_ci.get("overlap_ci_high", float("nan")),
            "overlap_bootstrap_n": overlap_ci.get("overlap_bootstrap_n", 0),
            "decision_overlap": decision_overlap,
            "previous_ema_overlap": prev_ema_overlap,
            "exchange_attempts": attempts,
            "exchange_accepted": accepted,
            "exchange_acceptance": acc,
            "decision_exchange_acceptance": decision_acc,
            "previous_ema_exchange_acceptance": prev_ema_acc,
            "region_key": region_key,
            "previous_low_overlap_streak": low_streak_prev,
            "previous_high_redundancy_streak": high_streak_prev,
            "action": action,
            "candidate_added_centers_A": ";".join(f"{x:.4f}" for x in added),
        })

    # Window agent: center visitation and localization diagnostics.
    for i, center in enumerate(centers):
        vals = [float(x) for x in samples_by_window.get(i, []) if math.isfinite(float(x))]
        arr = np.asarray(vals, dtype=float) if vals else np.asarray([], dtype=float)
        n = int(arr.size)
        mean = float(np.nanmean(arr)) if n else float("nan")
        sd = float(np.nanstd(arr)) if n else float("nan")
        hit_tight = float(np.mean(np.abs(arr - center) <= tight_hit_radius)) if n else float("nan")
        hit = float(np.mean(np.abs(arr - center) <= hit_radius)) if n else float("nan")
        mean_delta = float(mean - center) if math.isfinite(mean) else float("nan")
        left_ov = overlaps.get((i - 1, i), float("nan")) if i > 0 else float("nan")
        right_ov = overlaps.get((i, i + 1), float("nan")) if i + 1 < nwin else float("nan")
        left_att, left_acc, left_frac = _adaptive_feedback_exchange_acceptance(exchange_stats, i - 1, i) if i > 0 else (0, 0, float("nan"))
        right_att, right_acc, right_frac = _adaptive_feedback_exchange_acceptance(exchange_stats, i, i + 1) if i + 1 < nwin else (0, 0, float("nan"))
        action_bits = []
        if n < min_samples_per_window:
            action_bits.append("low sample count")
        elif math.isfinite(hit) and hit < 0.05 and math.isfinite(mean_delta) and abs(mean_delta) > 1.5:
            action_bits.append("poor center visitation; shift/retune candidate")
        elif math.isfinite(hit) and hit < 0.15:
            action_bits.append("weak center visitation")
        diagnostics.append({
            "window": i,
            "center_A": center,
            "k_kcal_mol_A2": ks_old[i] if i < len(ks_old) else float("nan"),
            "n_samples": n,
            "mean_cv_A": mean,
            "sd_cv_A": sd,
            "mean_minus_center_A": mean_delta,
            "hit_fraction_0p50A": hit_tight,
            "hit_fraction_0p75A": hit,
            "overlap_left": left_ov,
            "overlap_right": right_ov,
            "exchange_acceptance_left": left_frac,
            "exchange_acceptance_right": right_frac,
            "action": "; ".join(action_bits) if action_bits else "keep",
        })

    diag_by_window = {int(row["window"]): row for row in diagnostics}

    # Coverage agent: add at persistent low-density gaps that are not already
    # caught by pair-overlap logic.  This catches broad, deceptive histograms
    # with a thin unsampled valley between centers.
    coverage_rows = _adaptive_feedback_interval_density(samples_by_window, centers)
    for cr in coverage_rows:
        i = int(cr.get("left_window", -1))
        if i < 0 or i + 1 >= nwin:
            continue
        width = float(cr.get("width_A", 0.0) or 0.0)
        ratio = float(cr.get("density_ratio_to_median", float("nan")))
        pair_decision = float(pair_rows[i].get("decision_overlap", float("nan"))) if i < len(pair_rows) else float("nan")
        if width >= coverage_gap_width_A and math.isfinite(ratio) and ratio < coverage_gap_ratio and (not math.isfinite(pair_decision) or pair_decision < target_overlap + 0.15):
            c = round(0.5 * (centers[i] + centers[i + 1]), 4)
            add_candidates.append({
                "center_A": float(c),
                "priority": float(20.0 + (coverage_gap_ratio - ratio) * 20.0),
                "source": "coverage_gap",
                "left_window": int(i),
                "right_window": int(i + 1),
                "reason": f"coverage gap: interval density ratio {ratio:.3f}",
            })
            cr["coverage_action"] = f"candidate add {c:.4f} A"

    # Replica reduction agent: remove clearly redundant interior windows.  This
    # is memory-guided and deliberately conservative; the optimization objective
    # is minimum necessary replicas near the 0.30 overlap/exchange target, not a
    # one-round slash-and-burn.
    removal_candidates = []
    for pr in pair_rows:
        i = int(pr.get("left_window", -1))
        j = int(pr.get("right_window", -1))
        if i < 0 or j != i + 1:
            continue
        decision_ov = float(pr.get("decision_overlap", pr.get("overlap", float("nan"))))
        decision_acc = float(pr.get("decision_exchange_acceptance", pr.get("exchange_acceptance", float("nan"))))
        high_streak_prev = int(pr.get("previous_high_redundancy_streak", 0) or 0)
        if not math.isfinite(decision_ov) or decision_ov < high_overlap:
            continue
        exchange_ok = (math.isfinite(decision_acc) and decision_acc >= target_exchange_acceptance) or decision_ov >= very_high_overlap
        memory_ok = high_streak_prev > 0 or decision_ov >= very_high_overlap
        if aggr.get("mode") in {"aggressive", "very-aggressive"} and decision_ov >= high_overlap and exchange_ok:
            memory_ok = True
        if not (exchange_ok and memory_ok):
            continue
        for w in (i, j):
            if w <= 0 or w >= nwin - 1:
                continue
            di = diag_by_window.get(w, {})
            n = int(di.get("n_samples", 0) or 0)
            hit = float(di.get("hit_fraction_0p75A", float("nan")))
            if n < min_samples_per_window or not math.isfinite(hit) or hit < min_center_hit_for_removal:
                continue
            left_ov = overlaps.get((w - 1, w), float("nan"))
            right_ov = overlaps.get((w, w + 1), float("nan"))
            if not (math.isfinite(left_ov) and math.isfinite(right_ov)):
                continue
            if min(left_ov, right_ov) < target_overlap:
                continue
            left_acc = _adaptive_feedback_exchange_acceptance(exchange_stats, w - 1, w)[2]
            right_acc = _adaptive_feedback_exchange_acceptance(exchange_stats, w, w + 1)[2]
            finite_accs = [x for x in (left_acc, right_acc) if math.isfinite(float(x))]
            acc_safety = min(finite_accs) if finite_accs else float("nan")
            score = min(left_ov, right_ov) + 0.25 * hit
            if math.isfinite(acc_safety):
                score += 0.25 * min(1.0, acc_safety / max(1.0e-12, target_exchange_acceptance))
            removal_candidates.append({
                "window": int(w),
                "center_A": float(centers[w]),
                "score": float(score),
                "trigger_pair": f"w{i:02d}-w{j:02d}",
                "trigger_overlap": float(pr.get("overlap", float("nan"))),
                "decision_overlap": float(decision_ov),
                "previous_high_redundancy_streak": int(high_streak_prev),
                "trigger_exchange_acceptance": float(pr.get("exchange_acceptance", float("nan"))),
                "decision_exchange_acceptance": float(decision_acc) if math.isfinite(decision_acc) else float("nan"),
                "left_overlap": float(left_ov),
                "right_overlap": float(right_ov),
                "left_exchange_acceptance": float(left_acc) if math.isfinite(left_acc) else float("nan"),
                "right_exchange_acceptance": float(right_acc) if math.isfinite(right_acc) else float("nan"),
                "bypass_overlap": float("nan"),
                "bypass_overlap_ci_low": float("nan"),
                "bypass_overlap_ci_high": float("nan"),
                "source": "local_redundancy",
                "reason": f"redundant: decision overlap {decision_ov:.3f}; effective target overlap/exchange {target_overlap:.2f}/{target_exchange_acceptance:.2f}",
            })

    # Bypass-prune agent: explicitly test whether neighboring windows remain
    # statistically connected if an interior replica is removed.  This targets
    # the least valid replica count instead of merely trimming windows from
    # already-high-overlap adjacent pairs.
    for w in range(1, nwin - 1):
        left_vals = samples_by_window.get(w - 1, [])
        right_vals = samples_by_window.get(w + 1, [])
        mid_vals = samples_by_window.get(w, [])
        if len(left_vals) < min_samples_per_window or len(right_vals) < min_samples_per_window:
            continue
        lo = min(centers[w - 1], centers[w + 1], min(left_vals), min(right_vals))
        hi = max(centers[w - 1], centers[w + 1], max(left_vals), max(right_vals))
        bypass_ov = _adaptive_hist_overlap(left_vals, right_vals, lo, hi)
        bypass_ci = _adaptive_feedback_pair_overlap_ci(args, left_vals, right_vals, lo, hi, label_seed=104729 * (w + 1))
        bypass_ci_low = float(bypass_ci.get("overlap_ci_low", float("nan")))
        bypass_ci_high = float(bypass_ci.get("overlap_ci_high", float("nan")))
        if not math.isfinite(bypass_ov):
            continue
        left_pair = pair_rows[w - 1] if w - 1 < len(pair_rows) else {}
        right_pair = pair_rows[w] if w < len(pair_rows) else {}
        left_ci_low = float(left_pair.get("overlap_ci_low", left_pair.get("overlap", float("nan"))))
        right_ci_low = float(right_pair.get("overlap_ci_low", right_pair.get("overlap", float("nan"))))
        direct_valid = math.isfinite(bypass_ci_low) and bypass_ci_low >= minimum_valid_overlap
        direct_probably_valid = (not math.isfinite(bypass_ci_low)) and bypass_ov >= target_overlap
        adjacent_valid = (
            math.isfinite(left_ci_low) and left_ci_low >= minimum_valid_overlap
            and math.isfinite(right_ci_low) and right_ci_low >= minimum_valid_overlap
        )
        if not ((direct_valid or direct_probably_valid) and adjacent_valid):
            continue
        hit = float(diag_by_window.get(w, {}).get("hit_fraction_0p75A", float("nan")))
        mid_n = int(diag_by_window.get(w, {}).get("n_samples", len(mid_vals)) or len(mid_vals))
        weak_mid_bonus = 0.25 if (math.isfinite(hit) and hit < 0.15) else 0.0
        score = float(bypass_ov + 0.5 * max(0.0, bypass_ci_low if math.isfinite(bypass_ci_low) else 0.0) + weak_mid_bonus)
        removal_candidates.append({
            "window": int(w),
            "center_A": float(centers[w]),
            "score": float(score),
            "trigger_pair": f"w{w-1:02d}-w{w+1:02d}",
            "trigger_overlap": float(bypass_ov),
            "decision_overlap": float(bypass_ov),
            "previous_high_redundancy_streak": 0,
            "trigger_exchange_acceptance": float("nan"),
            "decision_exchange_acceptance": float("nan"),
            "left_overlap": float(left_pair.get("overlap", float("nan"))),
            "right_overlap": float(right_pair.get("overlap", float("nan"))),
            "left_exchange_acceptance": float(left_pair.get("exchange_acceptance", float("nan"))),
            "right_exchange_acceptance": float(right_pair.get("exchange_acceptance", float("nan"))),
            "bypass_overlap": float(bypass_ov),
            "bypass_overlap_ci_low": float(bypass_ci_low),
            "bypass_overlap_ci_high": float(bypass_ci_high),
            "bypass_overlap_bootstrap_n": int(bypass_ci.get("overlap_bootstrap_n", 0)),
            "source": "bypass_prune",
            "reason": (
                f"least-replica prune: bypass w{w-1:02d}-w{w+1:02d} overlap {bypass_ov:.3f} "
                f"with CI-low {bypass_ci_low:.3f} >= minimum valid {minimum_valid_overlap:.3f}; "
                f"middle window samples={mid_n}"
            ),
        })

    selected_removals = []
    blocked = set()
    for cand in sorted(removal_candidates, key=lambda d: (-float(d.get("score", 0.0)), int(d.get("window", 0)))):
        if len(selected_removals) >= max_remove_per_round:
            break
        w = int(cand["window"])
        if w in blocked:
            continue
        if any(abs(w - int(prev["window"])) <= remove_block_radius for prev in selected_removals):
            continue
        selected_removals.append(cand)
        blocked.update(range(w - remove_block_radius, w + remove_block_radius + 1))

    removed_windows = [int(c["window"]) for c in selected_removals]
    removed_center_keys = set(round(float(centers[w]), 4) for w in removed_windows)

    # Limit additions per round, avoiding duplicates and centers too close to
    # existing/proposed centers.  This prevents oscillatory over-correction.
    provisional_centers = set(old_center_keys) - removed_center_keys
    selected_additions = []
    for cand in sorted(add_candidates, key=lambda d: (-float(d.get("priority", 0.0)), float(d.get("center_A", 0.0)))):
        if len(selected_additions) >= max_add_per_round:
            break
        c = round(float(cand.get("center_A", float("nan"))), 4)
        if not math.isfinite(c):
            continue
        if any(abs(float(c) - float(x)) < min_new_spacing_A for x in provisional_centers):
            continue
        provisional_centers.add(c)
        selected_additions.append(cand)
        proposal_reasons.setdefault(c, []).append(str(cand.get("reason", "adaptive add")))

    # Center-shift agent: gently move persistently missed interior centers
    # toward their sampled means.  It is bounded and skipped in severe gap
    # regions, where adding windows is safer.
    shift_candidates = []
    for row in diagnostics:
        w = int(row.get("window", -1))
        if w <= 0 or w >= nwin - 1 or w in removed_windows:
            continue
        key = round(float(centers[w]), 4)
        if key not in provisional_centers:
            continue
        n = int(row.get("n_samples", 0) or 0)
        hit = float(row.get("hit_fraction_0p75A", float("nan")))
        delta = float(row.get("mean_minus_center_A", float("nan")))
        if n < min_samples_per_window or not (math.isfinite(hit) and math.isfinite(delta)):
            continue
        left_dec = float(pair_rows[w - 1].get("decision_overlap", float("nan"))) if w - 1 < len(pair_rows) else float("nan")
        right_dec = float(pair_rows[w].get("decision_overlap", float("nan"))) if w < len(pair_rows) else float("nan")
        severe_neighbor = (math.isfinite(left_dec) and left_dec < severe_overlap) or (math.isfinite(right_dec) and right_dec < severe_overlap)
        if severe_neighbor:
            continue
        if hit < 0.15 and abs(delta) > hit_radius:
            raw_shift = max(-max_center_shift_A, min(max_center_shift_A, 0.30 * delta))
            lo = centers[w - 1] + min_new_spacing_A
            hi = centers[w + 1] - min_new_spacing_A
            shifted = max(lo, min(hi, centers[w] + raw_shift))
            if abs(shifted - centers[w]) >= 0.15:
                shift_candidates.append({
                    "window": int(w),
                    "old_center_A": float(centers[w]),
                    "new_center_A": float(round(shifted, 4)),
                    "priority": float(abs(delta) * (0.20 - min(hit, 0.20))),
                    "mean_cv_A": float(row.get("mean_cv_A", float("nan"))),
                    "hit_fraction_0p75A": float(hit),
                    "reason": f"shifted 30% toward sampled mean; hit {hit:.3f}, mean-center {delta:.3f} A",
                })

    selected_shifts = []
    shifted_old_to_new: dict[float, float] = {}
    for cand in sorted(shift_candidates, key=lambda d: -float(d.get("priority", 0.0))):
        if len(selected_shifts) >= max_shift_per_round:
            break
        old_key = round(float(cand["old_center_A"]), 4)
        new_key = round(float(cand["new_center_A"]), 4)
        if old_key not in provisional_centers:
            continue
        if any(abs(float(new_key) - float(x)) < min_new_spacing_A for x in provisional_centers if x != old_key):
            continue
        provisional_centers.remove(old_key)
        provisional_centers.add(new_key)
        shifted_old_to_new[old_key] = new_key
        selected_shifts.append(cand)
        proposal_reasons.setdefault(new_key, []).append(str(cand.get("reason", "center shifted")))

    proposed_centers = sorted(float(x) for x in provisional_centers)
    if contact_primary and bool(getattr(args, "contact_normalize", True)):
        lo_c = max(0.0, float(getattr(args, "contact_adaptive_min", 0.0) or 0.0))
        hi_c = min(1.0, float(getattr(args, "contact_adaptive_max", 0.80) or 0.80))
        proposed_centers = sorted({round(max(lo_c, min(hi_c, float(x))), 6) for x in proposed_centers})
        if len(proposed_centers) >= 2:
            target_sp = max(1.0e-6, float(getattr(args, "contact_adaptive_target_spacing", 0.15) or 0.15))
            gap_fills = [
                round(0.5 * (proposed_centers[k] + proposed_centers[k + 1]), 4)
                for k in range(len(proposed_centers) - 1)
                if proposed_centers[k + 1] - proposed_centers[k] > 1.5 * target_sp
            ]
            if gap_fills:
                proposed_centers = sorted({round(max(lo_c, min(hi_c, float(x))), 6) for x in proposed_centers + gap_fills})
    proposed_k = _adaptive_feedback_k_from_centers(proposed_centers, args)

    # Source matching for bounded k changes.  Existing and shifted centers keep
    # memory of their old k; added centers use spacing-based k directly.
    source_old_index: dict[float, Optional[int]] = {}
    for c in proposed_centers:
        key = round(float(c), 4)
        if key in old_center_keys:
            source_old_index[key] = min(range(nwin), key=lambda idx: abs(float(centers[idx]) - float(c)))
        else:
            old_match = None
            for old_key, new_key in shifted_old_to_new.items():
                if abs(float(new_key) - float(key)) < 1.0e-6:
                    old_match = min(range(nwin), key=lambda idx: abs(float(centers[idx]) - float(old_key)))
                    break
            source_old_index[key] = old_match

    if contact_primary:
        min_k = max(0.0, float(getattr(args, "contact_adaptive_min_k_kcal", 5.0) or 5.0))
        max_k = max(min_k, float(getattr(args, "contact_adaptive_max_k_kcal", 120.0) or 120.0))
    else:
        min_k = max(0.30, float(getattr(args, "adaptive_min_k_kcal_a2", 0.30)))
        max_k = min(5.0, float(getattr(args, "adaptive_max_k_kcal_a2", 5.0)))
    for idx, c in enumerate(proposed_centers):
        key = round(float(c), 4)
        old_idx = source_old_index.get(key)
        if old_idx is None or old_idx < 0 or old_idx >= len(ks_old):
            proposed_k[idx] = float(max(min_k, min(max_k, proposed_k[idx])))
            proposal_reasons.setdefault(key, []).append("spacing-based k for added window")
            continue
        old_k = float(ks_old[old_idx])
        k = float(proposed_k[idx])
        k = max(old_k * min_k_factor, min(old_k * max_k_factor, k))
        row = diag_by_window.get(old_idx, {})
        hit = float(row.get("hit_fraction_0p75A", float("nan")))
        sd = float(row.get("sd_cv_A", float("nan")))
        local_left = abs(float(c) - proposed_centers[idx - 1]) if idx > 0 else float("nan")
        local_right = abs(proposed_centers[idx + 1] - float(c)) if idx + 1 < len(proposed_centers) else float("nan")
        local_spacing_vals = [x for x in (local_left, local_right) if math.isfinite(x)]
        local_spacing = min(local_spacing_vals) if local_spacing_vals else float("nan")
        if math.isfinite(hit) and math.isfinite(sd) and math.isfinite(local_spacing):
            if hit < 0.10 and sd > 0.75 * local_spacing:
                k = min(max_k, max(k, old_k * 1.25))
                proposal_reasons.setdefault(key, []).append("k increased: broad distribution and poor center visitation")
            low_neighbor = False
            if old_idx > 0:
                low_neighbor = low_neighbor or (math.isfinite(overlaps.get((old_idx - 1, old_idx), float("nan"))) and overlaps[(old_idx - 1, old_idx)] < target_overlap)
            if old_idx + 1 < nwin:
                low_neighbor = low_neighbor or (math.isfinite(overlaps.get((old_idx, old_idx + 1), float("nan"))) and overlaps[(old_idx, old_idx + 1)] < target_overlap)
            if low_neighbor and sd < 0.35 * local_spacing:
                k = max(min_k, min(k, old_k * 0.80))
                proposal_reasons.setdefault(key, []).append("k decreased: narrow local sampling with weak neighbor overlap")
        proposed_k[idx] = float(max(min_k, min(max_k, k)))

    # Annotate diagnostics/pairs after proposal selection.
    selected_add_centers = [round(float(c["center_A"]), 4) for c in selected_additions]
    for pr in pair_rows:
        i = int(pr.get("left_window", -1))
        adds_here = [a for a in selected_additions if int(a.get("left_window", -999)) == i]
        if adds_here:
            pr["action"] = "; ".join(str(a.get("reason", "add")) for a in adds_here)
            pr["added_centers_A"] = ";".join(f"{float(a['center_A']):.4f}" for a in adds_here)
        else:
            pr["added_centers_A"] = ""
        removed_here = [w for w in removed_windows if w in (int(pr.get("left_window", -1)), int(pr.get("right_window", -1)))]
        if removed_here:
            pr["action"] = f"remove redundant window(s): {','.join('w%02d' % w for w in removed_here)}"
    for row in diagnostics:
        w = int(row.get("window", -1))
        notes = [] if row.get("action") == "keep" else [str(row.get("action"))]
        if w in removed_windows:
            notes.append("removed redundant window")
        for sh in selected_shifts:
            if int(sh.get("window", -999)) == w:
                notes.append(f"center shifted to {float(sh['new_center_A']):.4f} A")
        row["action"] = "; ".join(notes) if notes else "keep"

    proposal_rows = []
    for i, (c, k) in enumerate(zip(proposed_centers, proposed_k)):
        key = round(float(c), 4)
        if key in old_center_keys:
            source = "existing"
        elif key in selected_add_centers:
            source = "added"
        elif key in set(shifted_old_to_new.values()):
            source = "shifted"
        else:
            source = "proposed"
        proposal_rows.append({
            "window": int(i),
            "center_A": float(c),
            "k_kcal_mol_A2": float(k),
            "source": source,
            "reason": "; ".join(proposal_reasons.get(key, ["spacing/memory optimized"])),
        })

    observed_score = _adaptive_feedback_observed_score(
        pair_rows,
        diagnostics,
        coverage_rows,
        nwin,
        target_overlap,
        target_exchange_acceptance,
        window_count_weight=window_count_weight,
    )
    previous_score = float(feedback_memory.get("last_score", float("nan"))) if isinstance(feedback_memory, dict) else float("nan")
    predicted_replica_delta = window_count_weight * (float(len(proposed_centers)) - float(nwin))
    predicted_score = float(observed_score["total"] + predicted_replica_delta)
    improvement_vs_previous = previous_score - observed_score["total"] if math.isfinite(previous_score) else float("nan")

    adaptive_memory_after = _adaptive_feedback_update_memory(
        feedback_memory,
        pair_rows,
        target_overlap=target_overlap,
        target_exchange_acceptance=target_exchange_acceptance,
        round_index=round_index,
    )
    adaptive_memory_after["last_score"] = float(observed_score["total"])
    adaptive_memory_after["last_score_components"] = observed_score["components"]
    adaptive_memory_after["last_predicted_next_score"] = float(predicted_score)
    adaptive_memory_after.setdefault("rounds", [])
    adaptive_memory_after["rounds"].append({
        "round": int(round_index),
        "out_dir": str(out_dir),
        "old_n_windows": int(nwin),
        "new_n_windows": int(len(proposed_centers)),
        "score": float(observed_score["total"]),
        "score_components": observed_score["components"],
        "predicted_next_score": float(predicted_score),
        "proposed_centers_A": [float(x) for x in proposed_centers],
        "requested_target_overlap": float(requested_target_overlap),
        "effective_target_overlap": float(target_overlap),
        "aggressiveness": aggr,
        "removed_windows": [int(w) for w in removed_windows],
        "added_centers_A": [float(c) for c in proposed_centers if round(float(c), 4) not in old_center_keys and round(float(c), 4) not in set(shifted_old_to_new.values())],
        "shifted_centers": selected_shifts,
    })

    k_rel_changes = []
    for c, k in zip(proposed_centers, proposed_k):
        old_idx = source_old_index.get(round(float(c), 4))
        if old_idx is not None and 0 <= old_idx < len(ks_old) and float(ks_old[old_idx]) > 0.0:
            k_rel_changes.append(abs(float(k) / float(ks_old[old_idx]) - 1.0))
    max_k_rel_change = max(k_rel_changes) if k_rel_changes else 0.0
    unresolved_pairs = [r for r in pair_rows if math.isfinite(float(r.get("decision_overlap", float("nan")))) and float(r.get("decision_overlap", float("nan"))) < 0.90 * target_overlap]
    coverage_gaps = [r for r in coverage_rows if str(r.get("coverage_action", "keep")) != "keep"]
    severe_center_misses = [r for r in diagnostics if math.isfinite(float(r.get("hit_fraction_0p75A", float("nan")))) and float(r.get("hit_fraction_0p75A", 1.0)) < 0.05 and math.isfinite(float(r.get("mean_minus_center_A", float("nan")))) and abs(float(r.get("mean_minus_center_A", 0.0))) > 1.5]
    center_changed = len(selected_shifts) > 0
    structure_changed = len(proposed_centers) != nwin or len(selected_additions) > 0 or len(removed_windows) > 0 or center_changed
    k_changed = max_k_rel_change > 0.05
    converged = (not structure_changed) and (not k_changed) and (not unresolved_pairs) and (not coverage_gaps) and (not severe_center_misses)
    if math.isfinite(improvement_vs_previous) and improvement_vs_previous < 0.10 * max(1.0, previous_score) and (not structure_changed) and max_k_rel_change < 0.10:
        converged = True
    convergence_status = "converged" if converged else "continue"

    diag_csv = out_dir / "adaptive_feedback_diagnostics.csv"
    pair_csv = out_dir / "adaptive_feedback_pair_overlaps.csv"
    prop_csv = out_dir / "adaptive_feedback_proposal.csv"
    cov_csv = out_dir / "adaptive_feedback_coverage_gaps.csv"
    for path, rows in ((diag_csv, diagnostics), (pair_csv, pair_rows), (prop_csv, proposal_rows), (cov_csv, coverage_rows)):
        with path.open("w", newline="") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

    windows_arg = " ".join(f"{x:.4g}" for x in proposed_centers)
    k_arg = " ".join(f"{x:.4g}" for x in proposed_k)
    suggested_cli = (
        f"--window-mode manual --contact-centers {windows_arg} --contact-k-kcal {k_arg}"
        if contact_primary else
        f"--window-mode manual --windows-a {windows_arg} --window-k-kcal-a2 {k_arg}"
    )
    summary = {
        "mode": "adaptive-feedback",
        "primary_cv": primary_cv_mode(args),
        "primary_cv_label": primary_axis_name,
        "primary_cv_units": primary_unit,
        "primary_k_units": primary_k_units(args),
        "legacy_primary_cv_column_names": True,
        "description": "Memory-guided adaptive window feedback. It optimizes add/remove/shift/k proposals toward the minimum statistically connected replica ladder. For nonlocal-contact mode, proposed_centers_A/proposed_k_kcal_mol_A2 are compatibility aliases for contact centers and kcal/mol/CV^2 force constants.",
        "agents": [
            "center_hit_agent",
            "neighbor_overlap_agent",
            "exchange_acceptance_agent",
            "coverage_gap_agent",
            "center_shift_agent",
            "bounded_k_tuning_agent",
            "bootstrap_overlap_ci_agent",
            "bypass_prune_agent",
            "minimum_connected_ladder_agent",
            "proposal_scoring_agent",
            "convergence_agent",
            "memory_convergence_agent",
            "replica_reduction_agent",
            "aggressiveness_control_agent",
        ],
        "total_samples": int(total_samples),
        "adaptive_window_aggressiveness": str(aggr.get("mode", "balanced")),
        "aggressiveness_settings": aggr,
        "requested_target_overlap": float(requested_target_overlap),
        "effective_target_overlap": float(target_overlap),
        "minimum_valid_overlap": float(minimum_valid_overlap),
        "window_count_weight": float(window_count_weight),
        "old_n_windows": int(nwin),
        "new_n_windows": int(len(proposed_centers)),
        "converged": bool(converged),
        "convergence_status": convergence_status,
        "observed_score": observed_score,
        "previous_score": float(previous_score) if math.isfinite(previous_score) else float("nan"),
        "improvement_vs_previous": float(improvement_vs_previous) if math.isfinite(improvement_vs_previous) else float("nan"),
        "predicted_next_score": float(predicted_score),
        "max_k_relative_change": float(max_k_rel_change),
        "selected_additions": selected_additions,
        "selected_center_shifts": selected_shifts,
        "removed_windows": [int(w) for w in removed_windows],
        "removed_centers_A": [float(centers[int(w)]) for w in removed_windows],
        "removal_candidates": selected_removals,
        "coverage_gap_candidates": coverage_gaps,
        "adaptive_memory_before": feedback_memory,
        "adaptive_memory_after": adaptive_memory_after,
        "old_centers_A": centers,
        "old_k_kcal_mol_A2": ks_old,
        "proposed_centers_A": proposed_centers,
        "proposed_k_kcal_mol_A2": proposed_k,
        "diagnostics_csv": str(diag_csv),
        "pair_overlaps_csv": str(pair_csv),
        "coverage_gaps_csv": str(cov_csv),
        "proposal_csv": str(prop_csv),
        "suggested_cli_fragment": suggested_cli,
        "thresholds": {
            "target_neighbor_overlap": float(target_overlap),
            "target_exchange_acceptance": float(target_exchange_acceptance),
            "minimum_valid_overlap_for_pruning": float(minimum_valid_overlap),
            "bootstrap_samples": int(getattr(args, "adaptive_feedback_bootstrap_samples", 200) or 200),
            "severe_overlap": float(severe_overlap),
            "high_overlap_redundant": float(high_overlap),
            "very_high_overlap_redundant": float(very_high_overlap),
            "max_add_per_round": int(max_add_per_round),
            "max_remove_per_round": int(max_remove_per_round),
            "max_shift_per_round": int(max_shift_per_round),
            "max_center_shift_A": float(max_center_shift_A),
            "max_k_factor_per_round": float(max_k_factor),
            "min_k_factor_per_round": float(min_k_factor),
            "min_center_hit_for_removal": float(min_center_hit_for_removal),
            "center_hit_radius_A": float(hit_radius),
            "tight_center_hit_radius_A": float(tight_hit_radius),
            "min_samples_per_window": int(min_samples_per_window),
        },
    }
    write_json(out_dir / "adaptive_feedback_proposal.json", _json_ready(summary))
    print(f"Adaptive-feedback proposal written to {out_dir / 'adaptive_feedback_proposal.json'}")
    print(f"    Score: {observed_score['total']:.3f}; status: {convergence_status}; windows {nwin} -> {len(proposed_centers)}")
    if selected_additions:
        print("    Added centers: " + ", ".join(f"{float(a['center_A']):.3f}{primary_unit_suffix}" for a in selected_additions))
    if selected_shifts:
        print("    Shifted centers: " + ", ".join(f"w{int(s['window']):02d} {float(s['old_center_A']):.3f}->{float(s['new_center_A']):.3f}{primary_unit_suffix}" for s in selected_shifts))
    if removed_windows:
        print(f"    Replica-reduction agent removed windows: {', '.join('w%02d' % int(w) for w in removed_windows)}")
    if not converged:
        print(f"    Suggested next fragment: {summary['suggested_cli_fragment']}")
    else:
        print("    Adaptive-feedback converged; proceeding with current/proposed fixed windows.")
    return summary

def _json_ready(obj):
    """Convert numpy/scalar containers into plain JSON-friendly objects.

    Keys starting with '_' are treated as private/precomputed and skipped so
    that cached numpy arrays stored in cv-definition dicts don't end up in JSON.
    """
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_ready(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj



def _clone_args(args):
    """Deep-copy an argparse namespace for an internal adaptive-feedback sub-run."""
    return argparse.Namespace(**copy.deepcopy(vars(args)))


def cleanup_adaptive_feedback_pilot_directory(round_dir: Path) -> dict:
    """Discard bulky pilot data after the adaptive proposal has been produced.

    Adaptive-feedback pilot rounds exist only to test window placement and force
    constants.  Keeping their trajectories/samples next to the final production
    invites accidental MBAR contamination, so remove the heavy per-frame/per-step
    outputs and keep only compact diagnostics/proposals.
    """
    round_dir = Path(round_dir)
    keep_names = {
        "adaptive_feedback_diagnostics.csv",
        "adaptive_feedback_pair_overlaps.csv",
        "adaptive_feedback_proposal.csv",
        "adaptive_feedback_proposal.json",
        "gareus_metadata.json",
        "umbrella_windows.csv",
        "umbrella_pymbar_metadata.json",
        "production_probe_report.json",
        "replica_shared_gamd_copy_report.json",
    }
    keep_prefixes = ("adaptive_feedback_",)
    remove_names = {
        "samples.csv",
        "analysis_arrays.npz",
        "analysis_arrays_metadata.json",
        "distances.csv",
        "distances.jsonl",
        "progress.jsonl",
        "exchange_attempts.csv",
        "shared_gamd_setup_state.xml",
        "shared_gamd_setup_final.pdb",
        "shared_gamd_setup_globals.json",
        "shared_gamd_setup_context.chk",
    }
    remove_dirs = {
        "checkpoints",
        "us_starting_structures",
    }
    remove_suffixes = (".dcd", ".xtc", ".trr", ".chk", ".chkpt")
    removed = []
    kept = []
    errors = []

    if not round_dir.exists():
        return {"round_dir": str(round_dir), "removed": [], "kept": [], "errors": ["round directory does not exist"]}

    for child in sorted(round_dir.iterdir(), key=lambda x: x.name):
        name = child.name
        keep = name in keep_names or name.startswith(keep_prefixes)
        should_remove = (name in remove_names) or (child.is_dir() and name in remove_dirs) or name.endswith(remove_suffixes)
        # A few OpenMM/debug files can be large even with unrecognized names; keep
        # explicit diagnostics, but discard obvious structure/state artifacts from pilots.
        if not keep and name.startswith(("trajectory", "replica_", "state_")):
            should_remove = True
        if keep or not should_remove:
            kept.append(name)
            continue
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            removed.append(name)
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    manifest = {
        "round_dir": str(round_dir),
        "policy": "Adaptive pilot data discarded after diagnostics/proposal; use final_production for MBAR/PMF.",
        "removed": removed,
        "kept": kept,
        "errors": errors,
    }
    try:
        write_json(round_dir / "adaptive_feedback_cleanup_manifest.json", _json_ready(manifest))
    except Exception as exc:
        manifest["errors"].append(f"could not write cleanup manifest: {exc}")
    return manifest


def _load_adaptive_feedback_proposal(proposal_path: Path) -> Optional[dict]:
    proposal_path = Path(proposal_path)
    if not proposal_path.exists():
        return None
    try:
        payload = json.loads(proposal_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: could not read adaptive-feedback proposal {proposal_path}: {exc}")
        return None
    centers = payload.get("proposed_centers_A")
    ks = payload.get("proposed_k_kcal_mol_A2")
    if not isinstance(centers, list) or not isinstance(ks, list) or len(centers) == 0 or len(centers) != len(ks):
        print(f"WARNING: adaptive-feedback proposal {proposal_path} is missing matched centers/k arrays")
        return None
    try:
        payload["proposed_centers_A"] = [float(x) for x in centers]
        payload["proposed_k_kcal_mol_A2"] = [float(x) for x in ks]
        sec_centers = payload.get("proposed_secondary_cv_centers")
        if isinstance(sec_centers, list) and sec_centers:
            payload["proposed_secondary_cv_centers"] = [float(x) for x in sec_centers]
        sec_k = payload.get("proposed_secondary_cv_k_kcal_mol")
        if isinstance(sec_k, list) and sec_k:
            payload["proposed_secondary_cv_k_kcal_mol"] = [float(x) for x in sec_k]
    except Exception as exc:
        print(f"WARNING: adaptive-feedback proposal {proposal_path} has non-numeric centers/k: {exc}")
        return None
    return payload




def _adaptive_feedback_sparse_candidate_csv_from_proposal(args, proposal: Optional[dict], proposal_path: Path) -> Optional[Path]:
    """Return the sparse explicit-2D candidate CSV from a 2D proposal, if useful.

    The proposal still contains the axis-factorized rectangular centers for
    backwards compatibility.  When local 2D defects were found, the dispatcher
    also writes adaptive_feedback_explicit_window_candidates.csv, containing the
    rectangular base grid plus sparse midpoint patch windows.  The automatic
    adaptive-feedback driver uses that table for final clean production without
    adding a new user-facing refinement flag.
    """
    if not isinstance(proposal, dict):
        return None
    sparse_2d_enabled = bool(getattr(args, "sparse_2d_patches_enabled", True))
    proposal_primary = str(proposal.get("primary_cv", primary_cv_mode(args) if args is not None else "distance"))
    if proposal_primary == "nonlocal-contacts":
        sparse_2d_enabled = sparse_2d_enabled and bool(getattr(args, "contact_sparse_2d_patches_enabled", True))
    if not sparse_2d_enabled:
        return None
    sparse = proposal.get("two_d_sparse_patch_proposal")
    if not isinstance(sparse, dict):
        return None
    try:
        patch_count = int(sparse.get("patch_candidate_windows", 0) or 0)
        retained_patch_count = int(sparse.get("retained_existing_patch_windows", 0) or 0)
        explicit_count = int(sparse.get("explicit_candidate_windows_if_enabled", 0) or 0)
        base_count = int(sparse.get("base_rectangular_grid_windows", 0) or 0)
    except Exception:
        return None
    wants_next = bool(sparse.get("used_by_adaptive_feedback_next_pilot_when_available", False))
    wants_final = bool(sparse.get("used_by_adaptive_feedback_final_production_when_available", sparse.get("used_by_adaptive_feedback_final_production", False)))
    if explicit_count <= 0:
        return None
    if patch_count <= 0 and retained_patch_count <= 0 and explicit_count <= base_count and not (wants_next or wants_final):
        return None
    csv_name = sparse.get("explicit_window_candidates_csv") or proposal.get("adaptive_feedback_explicit_window_candidates_csv")
    if not csv_name:
        return None
    try:
        csv_path = Path(str(csv_name))
        if not csv_path.is_absolute():
            csv_path = Path(proposal_path).parent / csv_path
        if csv_path.exists() and csv_path.stat().st_size > 0:
            return csv_path
    except Exception:
        return None
    return None

def run_adaptive_feedback_auto_loop(args, out_dir: Path, openmm, app, unit, forcefield, topology, equil_state, progress: Optional[GuiProgressSink] = None):
    """Run short adaptive-feedback pilot round(s), optionally followed by clean final production.

    The pilot rounds use --adaptive-feedback-pilot-fraction of the final
    production length, default 1/20.  The driver keeps a memory of previous
    pilot diagnostics so it can converge toward the smallest useful replica set
    while keeping neighbor overlap and exchange acceptance close to 0.30.  The
    final run is still a clean fixed-window GaREUS run for MBAR/reweighting.
    In 2D secondary-CV runs, local sparse patch candidates from each pilot can
    be consumed by the next pilot and by final production via the existing
    adaptive-feedback mode; no separate refinement flag is required.  When
    ``args.adaptive_feedback_skip_final_production`` is true, the driver stops
    after the pilots and returns a handoff-ready proposal for double-adaptive
    workflows instead of launching ``final_production/``.
    """
    out_dir = Path(out_dir)
    n_rounds = max(0, int(getattr(args, "adaptive_feedback_rounds", 1) or 0))
    raw_secondary_centers = getattr(args, "secondary_cv_centers", None)
    adaptive_secondary_enabled = bool(secondary_cv_enabled(args) and n_rounds > 0 and str(getattr(args, "adaptive_secondary_cv", "auto") or "auto") != "fixed")
    if adaptive_secondary_enabled and (raw_secondary_centers is None or len(raw_secondary_centers) <= 1):
        args.secondary_cv_centers = adaptive_secondary_default_centers(args)
        print("    adaptive secondary-CV feedback: starting centers " + ", ".join(f"{x:.3g}" for x in args.secondary_cv_centers))
    target_prod_steps = max(1, int(getattr(args, "gamd_production_steps", 1) or 1))
    skip_final_production = bool(
        getattr(args, "adaptive_feedback_skip_final_production", False)
        or getattr(args, "double_adaptive_handoff", False)
    )
    pilot_fraction = float(getattr(args, "adaptive_feedback_pilot_fraction", 0.05) or 0.05)
    if not math.isfinite(pilot_fraction) or pilot_fraction <= 0.0:
        pilot_fraction = 0.05
    try:
        pilot_steps_override = int(getattr(args, "adaptive_feedback_pilot_steps", -1))
    except Exception:
        pilot_steps_override = -1
    if pilot_steps_override > 0:
        pilot_steps = int(pilot_steps_override)
    else:
        pilot_steps = max(1, int(round(pilot_fraction * float(target_prod_steps))))
    target_overlap = float(getattr(args, "adaptive_feedback_target_overlap", 0.30))
    if not math.isfinite(target_overlap) or target_overlap <= 0.0:
        target_overlap = 0.30
    aggr = _adaptive_window_aggressiveness_settings(args)
    effective_target_overlap = max(0.04, min(0.95, target_overlap * float(aggr.get("target_overlap_factor", 1.0))))

    validation_fraction = float(getattr(args, "adaptive_feedback_validation_fraction", 0.10) or 0.0)
    if not math.isfinite(validation_fraction) or validation_fraction < 0.0:
        validation_fraction = 0.10
    try:
        validation_steps_override = int(getattr(args, "adaptive_feedback_validation_steps", -1))
    except Exception:
        validation_steps_override = -1
    if n_rounds > 1:
        if validation_steps_override > 0:
            validation_pilot_steps = int(validation_steps_override)
            validation_step_source = "explicit --adaptive-feedback-validation-steps"
        elif validation_steps_override == 0:
            validation_pilot_steps = int(pilot_steps)
            validation_step_source = "disabled longer validation; using ordinary pilot length"
        else:
            validation_pilot_steps = max(pilot_steps, int(round(validation_fraction * float(target_prod_steps))))
            validation_step_source = f"validation fraction {validation_fraction:.4g}"
    else:
        validation_pilot_steps = int(pilot_steps)
        validation_step_source = "single round; ordinary pilot length"

    print("Adaptive-feedback automatic workflow")
    print(f"    pilot rounds: {n_rounds}")
    _pilot_source = (
        f"explicit --pilot-steps override"
        if pilot_steps_override > 0
        else f"{pilot_fraction:.4g} x final target {target_prod_steps}"
    )
    print(f"    pilot production steps per round: {pilot_steps} ({_pilot_source})")
    if n_rounds > 1:
        print(f"    final validation pilot steps: {validation_pilot_steps} ({validation_step_source})")
    print(f"    requested neighbor overlap: {target_overlap:.3f}")
    print(f"    effective neighbor overlap: {effective_target_overlap:.3f} ({aggr.get('mode', 'balanced')} aggressiveness)")
    print("    target exchange acceptance for reduction: 0.300")
    print("    feedback memory: enabled; repeated over/under-resolved regions guide add/remove decisions")
    if skip_final_production:
        print("    final production: skipped; proposal will be handed to adaptive-production")
    else:
        print(f"    final production will be written under: {out_dir / 'final_production'}")

    driver_summary = {
        "mode": "adaptive-feedback-auto-loop",
        "description": "Memory-guided short pilot GaREUS round(s) converge windows/k toward the smallest useful replica set near 0.30 neighbor overlap/exchange, then execute a final clean fixed-window production run. In 2D adaptive-feedback runs, sparse local patch candidates are iterated through later pilots and consumed automatically for final production.",
        "target_neighbor_overlap": float(target_overlap),
        "pilot_rounds_requested": int(n_rounds),
        "pilot_production_fraction": float(pilot_fraction),
        "pilot_production_steps": int(pilot_steps),
        "validation_pilot_fraction": float(validation_fraction),
        "validation_pilot_steps": int(validation_pilot_steps),
        "validation_pilot_step_source": str(validation_step_source),
        "final_production_steps": int(target_prod_steps),
        "rounds": [],
        "final_production_dir": str(out_dir / "final_production"),
        "final_production_skipped": bool(skip_final_production),
        "handoff_to_adaptive_production": bool(skip_final_production),
        "note": "Pilot outputs are diagnostic. Use final_production for final PMF/MBAR unless intentionally analyzing pilot stages separately. In double-adaptive mode, final_production is intentionally skipped and the latest proposal is handed to adaptive-production.",
    }
    write_json(out_dir / "adaptive_feedback_driver_summary.json", driver_summary)

    adaptive_workflow_start_wall = time.time()
    adaptive_workflow_total_steps = int(max(0, n_rounds - 1) * pilot_steps + (validation_pilot_steps if n_rounds > 0 else 0) + (0 if skip_final_production else target_prod_steps))
    adaptive_workflow_done_so_far = 0
    current_centers = None
    current_k = None
    current_secondary_centers = [float(x) for x in getattr(args, "secondary_cv_centers", [])] if secondary_cv_enabled(args) and getattr(args, "secondary_cv_centers", None) else None
    current_secondary_k = None
    current_windows_2d_csv = None
    current_windows_2d_source_round = None
    genpept_prior_summary = None
    if getattr(args, "windows_2d_csv", None):
        current_windows_2d_csv = Path(str(args.windows_2d_csv))
        current_windows_2d_source_round = -1
        driver_summary["initial_windows_2d_csv"] = str(current_windows_2d_csv)
        driver_summary["initial_window_mode"] = "explicit_sparse_2d_from_user_csv"
        write_json(out_dir / "adaptive_feedback_driver_summary.json", _json_ready(driver_summary))
        print(f"    Adaptive-feedback round 1 will use explicit 2D window table: {current_windows_2d_csv}")
    elif genpept_prior_enabled(args):
        try:
            genpept_prior_summary = build_genpept_window_prior(args, out_dir, topology)
        except Exception as exc:
            if bool(getattr(args, "genpept_prior_required", False)):
                raise
            genpept_prior_summary = {
                "enabled": True,
                "used": False,
                "reason": f"GENPEPT prior build failed: {exc}",
            }
            print(f"WARNING: GENPEPT prior build failed; falling back to ordinary adaptive-feedback: {exc}")
        if isinstance(genpept_prior_summary, dict):
            driver_summary["genpept_prior"] = _json_ready(genpept_prior_summary)
            prior_csv = genpept_prior_summary.get("windows_2d_csv") if genpept_prior_summary.get("used") else None
            if prior_csv:
                current_windows_2d_csv = Path(str(prior_csv))
                current_windows_2d_source_round = 0
                driver_summary["initial_windows_2d_csv"] = str(current_windows_2d_csv)
                driver_summary["initial_window_mode"] = "explicit_sparse_2d_from_genpept_prior"
                prior_dir = genpept_prior_summary.get("genpept_dir")
                if prior_dir and getattr(args, "seed_conformers_dir", None) is None and (Path(str(prior_dir)) / "final_survivor_seeds.csv").exists():
                    args.seed_conformers_dir = Path(str(prior_dir))
                    driver_summary["seed_conformers_dir_from_genpept_prior"] = str(args.seed_conformers_dir)
                print(f"    GENPEPT prior round-zero window table: {current_windows_2d_csv}")
            else:
                reason = str(genpept_prior_summary.get("reason", "no usable GENPEPT prior"))
                print(f"    GENPEPT prior not used: {reason}")
            write_json(out_dir / "adaptive_feedback_driver_summary.json", _json_ready(driver_summary))
    adaptive_memory = {
        "mode": "memory-guided adaptive-feedback",
        "target_overlap": float(target_overlap),
        "effective_target_overlap": float(effective_target_overlap),
        "aggressiveness": aggr,
        "target_exchange_acceptance": 0.30,
        "regions": {},
        "rounds": [],
    }
    write_json(out_dir / "adaptive_feedback_memory.json", _json_ready(adaptive_memory))

    for round_index in range(n_rounds):
        round_no = round_index + 1
        # Capture centers used THIS pilot before the proposal replaces them (used for CV2 coverage check).
        _loop_pilot_secondary_centers: list[float] = list(current_secondary_centers) if current_secondary_centers else []
        _pilot_coverage: dict = {}
        _pilot_cv1_blocked = False
        _pilot_cv2_blocked = False
        current_pilot_steps = int(validation_pilot_steps if (n_rounds > 1 and round_index == n_rounds - 1) else pilot_steps)
        round_dir = out_dir / f"adaptive_feedback_round_{round_no:02d}"

        # On resume: if this round already produced a valid proposal, skip its MD
        # and restore state from the saved proposal so the loop can continue.
        _existing_proposal = _load_adaptive_feedback_proposal(round_dir / "adaptive_feedback_proposal.json")
        if _existing_proposal is not None:
            current_centers = _existing_proposal["proposed_centers_A"]
            current_k = _existing_proposal["proposed_k_kcal_mol_A2"]
            if isinstance(_existing_proposal.get("proposed_secondary_cv_centers"), list):
                current_secondary_centers = [float(x) for x in _existing_proposal.get("proposed_secondary_cv_centers")]
            if isinstance(_existing_proposal.get("proposed_secondary_cv_k_kcal_mol"), list):
                current_secondary_k = [float(x) for x in _existing_proposal.get("proposed_secondary_cv_k_kcal_mol")]
            adaptive_memory = _existing_proposal.get("adaptive_memory_after", adaptive_memory) if isinstance(_existing_proposal, dict) else adaptive_memory
            current_windows_2d_csv = _adaptive_feedback_sparse_candidate_csv_from_proposal(args, _existing_proposal, round_dir / "adaptive_feedback_proposal.json")
            current_windows_2d_source_round = int(round_no) if current_windows_2d_csv is not None else None
            adaptive_workflow_done_so_far += current_pilot_steps
            print(f"    [resume] Pilot round {round_no}/{n_rounds} already complete; skipping MD and restoring proposal.")
            if bool(_existing_proposal.get("converged", False)) and round_no < n_rounds:
                print(f"    [resume] Round {round_no} had previously converged; stopping pilots early.")
                break
            continue

        round_dir.mkdir(parents=True, exist_ok=True)
        round_args = _clone_args(args)
        round_args.out = str(round_dir)
        round_args.gamd_production_steps = int(current_pilot_steps)
        round_args.resume = False
        round_args.adaptive_feedback_enabled = True
        round_args.adaptive_feedback_target_overlap = float(target_overlap)
        round_args.adaptive_feedback_pilot = True
        round_args.adaptive_feedback_final_production = False
        round_args.adaptive_feedback_round_index = int(round_no)
        round_args.adaptive_feedback_rounds_total = int(n_rounds)
        round_args.adaptive_feedback_final_steps = int(target_prod_steps)
        round_args.adaptive_feedback_workflow_start_wall = float(adaptive_workflow_start_wall)
        round_args.adaptive_feedback_workflow_done_before = int(adaptive_workflow_done_so_far)
        round_args.adaptive_feedback_workflow_total_steps = int(adaptive_workflow_total_steps)
        round_args.adaptive_feedback_memory = adaptive_memory
        round_args.adaptive_feedback_prev_rounds = [
            {
                "round": int(r["round"]),
                "converged": bool(r.get("converged", False)),
                "n_windows": int(r.get("new_n_windows", 0)),
                "overlap_mean": float(r.get("overlap_mean", float("nan"))),
                "exchange_frac": float(r.get("exchange_frac", float("nan"))),
            }
            for r in driver_summary.get("rounds", [])
        ]
        if current_windows_2d_csv is not None:
            # Iterate sparse explicit 2D tables through later pilot rounds while
            # keeping the public workflow spelling as --window-mode adaptive-feedback.
            round_args.window_mode = "manual"
            round_args.windows_2d_csv = str(current_windows_2d_csv)
            round_args.windows_a = []
            round_args.window_k_kcal_a2 = []
            print(f"    Adaptive-feedback pilot round {round_no} will use sparse explicit 2D window table from round {current_windows_2d_source_round}: {current_windows_2d_csv}")
        elif current_centers is None:
            round_args.window_mode = "adaptive-feedback"
            round_args.windows_2d_csv = None
        else:
            round_args.window_mode = "manual"
            round_args.windows_2d_csv = None
            if primary_cv_is_contacts(round_args):
                round_args.contact_centers = [float(x) for x in current_centers]
                round_args.contact_k_kcal = [float(x) for x in current_k]
                round_args.windows_a = []
                round_args.window_k_kcal_a2 = []
            else:
                round_args.windows_a = [float(x) for x in current_centers]
                round_args.window_k_kcal_a2 = [float(x) for x in current_k]
        if current_windows_2d_csv is None and secondary_cv_enabled(round_args) and current_secondary_centers is not None:
            round_args.secondary_cv_centers = [float(x) for x in current_secondary_centers]
            if current_secondary_k and str(getattr(round_args, "secondary_cv_k_mode", "fixed") or "fixed").lower() in {"fixed", "constant"}:
                # In fixed-k mode, the production builder accepts one secondary k
                # scale; keep the adaptive list in proposal JSON and use its
                # median for the next crossed grid. In spacing/adaptive mode, the
                # builder recomputes per-center secondary k values directly.
                round_args.secondary_cv_k_kcal = float(np.median(np.asarray(current_secondary_k, dtype=float)))

        print(f"\n=== Adaptive-feedback pilot round {round_no}/{n_rounds}: {current_pilot_steps} production steps ===")
        run_gareus(round_args, round_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress)
        if _graceful_shutdown.is_set():
            print("Graceful shutdown: aborting adaptive-feedback after pilot round checkpoint.", flush=True)
            return driver_summary

        proposal_path = round_dir / "adaptive_feedback_proposal.json"
        proposal = _load_adaptive_feedback_proposal(proposal_path)
        if proposal is None:
            raise RuntimeError(f"Adaptive-feedback round {round_no} did not produce a usable proposal: {proposal_path}")
        current_centers = proposal["proposed_centers_A"]
        current_k = proposal["proposed_k_kcal_mol_A2"]
        if isinstance(proposal.get("proposed_secondary_cv_centers"), list):
            current_secondary_centers = [float(x) for x in proposal.get("proposed_secondary_cv_centers")]
        if isinstance(proposal.get("proposed_secondary_cv_k_kcal_mol"), list):
            current_secondary_k = [float(x) for x in proposal.get("proposed_secondary_cv_k_kcal_mol")]
        current_windows_2d_csv = _adaptive_feedback_sparse_candidate_csv_from_proposal(args, proposal, proposal_path)
        current_windows_2d_source_round = int(round_no) if current_windows_2d_csv is not None else None

        # Delaunay placement: override with data-driven positions from pilot samples.
        # Must run BEFORE cleanup_adaptive_feedback_pilot_directory deletes samples.csv.
        _delaunay_trigger = int(getattr(args, "delaunay_after_round", 0) or 0)
        if str(getattr(args, "window_mode", "")) == "delaunay-feedback" and _delaunay_trigger == 0:
            _delaunay_trigger = 1
        # delaunay-feedback iterates every round >= trigger by default; explicit flag overrides.
        _di_flag = getattr(args, "delaunay_iterate", None)
        _delaunay_iterate = (
            bool(_di_flag) if _di_flag is not None
            else str(getattr(args, "window_mode", "")) == "delaunay-feedback"
        )
        _delaunay_stability_tol = float(getattr(args, "delaunay_stability_tol", 0.05) or 0.05)
        _d_round_cond = (
            int(round_no) >= _delaunay_trigger if _delaunay_iterate
            else int(round_no) == _delaunay_trigger
        )
        if _delaunay_trigger > 0 and _d_round_cond and secondary_cv_enabled(args):
            try:
                from .windows import build_delaunay_windows_from_pilot_samples as _bdelaunay
                import csv as _csv_mod
                _d_rows, _d_meta = _bdelaunay(
                    round_dir / "samples.csv", args, out_dir=None, round_index=int(round_no))
                if _d_rows:
                    # Stability check: skip layout update if anchors barely moved
                    _prev_anchors = driver_summary.get("_prev_delaunay_anchors")
                    _new_anchors = _d_meta.get("anchor_norm_positions", [])
                    _is_stable = False
                    if _delaunay_iterate and _prev_anchors and _new_anchors and len(_prev_anchors) == len(_new_anchors):
                        try:
                            import numpy as _np_d
                            from scipy.spatial import cKDTree as _CKD
                            _t = _CKD(_np_d.array(_new_anchors))
                            _dists, _ = _t.query(_np_d.array(_prev_anchors), k=1)
                            _max_shift = float(_np_d.max(_dists))
                            if _max_shift < _delaunay_stability_tol:
                                _is_stable = True
                                print(
                                    f"    Delaunay stable: max anchor shift {_max_shift:.4f} "
                                    f"< tol {_delaunay_stability_tol} — layout unchanged."
                                )
                        except Exception:
                            pass
                    driver_summary["_prev_delaunay_anchors"] = _new_anchors
                    if not _is_stable:
                        _d_csv_path = round_dir / "delaunay_initial_windows.csv"
                        with _d_csv_path.open("w", newline="") as _fh:
                            _dw = _csv_mod.DictWriter(_fh, fieldnames=list(_d_rows[0].keys()), extrasaction="ignore")
                            _dw.writeheader()
                            _dw.writerows(_d_rows)
                        current_windows_2d_csv = _d_csv_path
                        current_windows_2d_source_round = int(round_no)
                        driver_summary["delaunay_initial_windows_csv"] = str(_d_csv_path)
                        driver_summary["delaunay_trigger_round"] = int(round_no)
                        print(f"    Delaunay placement: {len(_d_rows)} windows from round {round_no} samples -> {_d_csv_path}")
            except Exception as _delaunay_exc:
                print(f"WARNING: Delaunay placement failed (round {round_no}): {_delaunay_exc}; "
                      f"keeping axis-factorized proposal.")

        adaptive_memory = proposal.get("adaptive_memory_after", adaptive_memory) if isinstance(proposal, dict) else adaptive_memory
        write_json(out_dir / "adaptive_feedback_memory.json", _json_ready(adaptive_memory))

        # ── Pilot CV coverage health check ────────────────────────────────────
        # Must run BEFORE cleanup deletes samples.csv.
        # Flags insufficient CV1/CV2 exploration and blocks premature early-stop.
        _samples_path_cov = round_dir / "samples.csv"
        if _samples_path_cov.exists():
            import csv as _csv_cov
            _all_cv1_cov: list[float] = []
            _all_cv2_cov: list[float] = []
            try:
                with _samples_path_cov.open() as _fh_cov:
                    for _row_cov in _csv_cov.DictReader(_fh_cov):
                        try:
                            _all_cv1_cov.append(float(_row_cov["cv_A"]))
                        except (KeyError, ValueError):
                            pass
                        try:
                            _all_cv2_cov.append(float(_row_cov["secondary_cv"]))
                        except (KeyError, ValueError):
                            pass
            except Exception as _exc_cov:
                print(f"    WARNING: pilot coverage check could not read samples.csv: {_exc_cov}")

            # CV1 coverage: compare sampled span to achievable range from boundary pull.
            if primary_cv_is_contacts(args):
                _cv1_lo = float(getattr(args, "contact_adaptive_min", 0.0) or 0.0)
                _cv1_hi = float(getattr(args, "contact_adaptive_max", 0.80) or 0.80)
            else:
                _cv1_lo = float(getattr(args, "adaptive_min_a", getattr(args, "windows_min_a", 0.0)) or 0.0)
                _cv1_hi = float(getattr(args, "adaptive_max_a", getattr(args, "windows_max_a", 30.0)) or 30.0)
            _cv1_expected_span = max(1e-9, _cv1_hi - _cv1_lo)
            if _all_cv1_cov:
                _cv1_span = max(_all_cv1_cov) - min(_all_cv1_cov)
                _cv1_frac = _cv1_span / _cv1_expected_span
                _cv1_thr = float(getattr(args, "pilot_min_cv1_coverage", 0.60) or 0.60)
                _pilot_coverage.update({
                    "cv1_sampled_min": min(_all_cv1_cov),
                    "cv1_sampled_max": max(_all_cv1_cov),
                    "cv1_expected_min": _cv1_lo,
                    "cv1_expected_max": _cv1_hi,
                    "cv1_coverage_fraction": _cv1_frac,
                    "cv1_coverage_threshold": _cv1_thr,
                })
                if _cv1_frac < _cv1_thr:
                    _pilot_cv1_blocked = True
                    print(
                        f"    WARNING [coverage gate]: CV1 pilot coverage "
                        f"{_cv1_frac:.1%} < {_cv1_thr:.0%} of achievable range "
                        f"[{_cv1_lo:.4f}, {_cv1_hi:.4f}] "
                        f"(sampled [{min(_all_cv1_cov):.4f}, {max(_all_cv1_cov):.4f}]). "
                        f"Ensemble may be trapped — early-stop blocked."
                    )

            # CV2 coverage: compare sampled span to the secondary-CV center range
            # that was ACTIVE during this pilot (saved at loop top before proposal update).
            if _all_cv2_cov and _loop_pilot_secondary_centers and len(_loop_pilot_secondary_centers) >= 2:
                _cv2_lo = float(min(_loop_pilot_secondary_centers))
                _cv2_hi = float(max(_loop_pilot_secondary_centers))
                _cv2_expected_span = max(1e-9, _cv2_hi - _cv2_lo)
                _cv2_span = max(_all_cv2_cov) - min(_all_cv2_cov)
                _cv2_frac = _cv2_span / _cv2_expected_span
                _cv2_thr = float(getattr(args, "pilot_min_cv2_coverage", 0.70) or 0.70)
                _pilot_coverage.update({
                    "cv2_sampled_min": min(_all_cv2_cov),
                    "cv2_sampled_max": max(_all_cv2_cov),
                    "cv2_expected_min": _cv2_lo,
                    "cv2_expected_max": _cv2_hi,
                    "cv2_coverage_fraction": _cv2_frac,
                    "cv2_coverage_threshold": _cv2_thr,
                })
                if _cv2_frac < _cv2_thr:
                    _pilot_cv2_blocked = True
                    print(
                        f"    WARNING [coverage gate]: CV2 pilot coverage "
                        f"{_cv2_frac:.1%} < {_cv2_thr:.0%} of secondary CV range "
                        f"[{_cv2_lo:.4f}, {_cv2_hi:.4f}] "
                        f"(sampled [{min(_all_cv2_cov):.4f}, {max(_all_cv2_cov):.4f}]). "
                        f"Ramachandran space under-sampled — early-stop blocked."
                    )
            if _pilot_coverage:
                print(
                    f"    Pilot coverage: CV1={_pilot_coverage.get('cv1_coverage_fraction', float('nan')):.1%}"
                    + (f"  CV2={_pilot_coverage.get('cv2_coverage_fraction', float('nan')):.1%}" if "cv2_coverage_fraction" in _pilot_coverage else "")
                )
        # ──────────────────────────────────────────────────────────────────────

        cleanup_manifest = cleanup_adaptive_feedback_pilot_directory(round_dir)
        print(f"    Discarded adaptive pilot MD data; kept diagnostics/proposal in {round_dir}")
        _obs = proposal.get("observed_score", {}) if isinstance(proposal.get("observed_score"), dict) else {}
        _exch = proposal.get("exchange_acceptance_fraction", float("nan"))
        driver_summary["rounds"].append({
            "round": int(round_no),
            "directory": str(round_dir),
            "production_steps": int(current_pilot_steps),
            "proposal_json": str(proposal_path),
            "old_n_windows": int(proposal.get("old_n_windows", 0) or 0),
            "new_n_windows": int(proposal.get("new_n_windows", len(current_centers)) or len(current_centers)),
            "proposed_centers_A": current_centers,
            "proposed_k_kcal_mol_A2": current_k,
            "proposed_secondary_cv_centers": current_secondary_centers,
            "proposed_secondary_cv_k_kcal_mol": current_secondary_k,
            "suggested_cli_fragment": proposal.get("suggested_cli_fragment", ""),
            "adaptive_feedback_sparse_windows_2d_csv_for_next_pilot_or_final": str(current_windows_2d_csv) if current_windows_2d_csv is not None else "",
            "adaptive_feedback_sparse_source_round": current_windows_2d_source_round,
            "cleanup_manifest": cleanup_manifest,
            "pilot_cv_coverage": _pilot_coverage,
            "converged": bool(proposal.get("converged", False)),
            "overlap_mean": float(_obs.get("overlap_mean", float("nan"))),
            "exchange_frac": float(_exch) if _exch is not None else float("nan"),
        })
        driver_summary["latest_proposed_centers_A"] = current_centers
        driver_summary["latest_proposed_k_kcal_mol_A2"] = current_k
        driver_summary["latest_proposed_secondary_cv_centers"] = current_secondary_centers
        driver_summary["latest_proposed_secondary_cv_k_kcal_mol"] = current_secondary_k
        driver_summary["latest_sparse_windows_2d_csv_for_next_pilot_or_final"] = str(current_windows_2d_csv) if current_windows_2d_csv is not None else ""
        driver_summary["latest_sparse_source_round"] = current_windows_2d_source_round
        driver_summary["adaptive_memory_json"] = str(out_dir / "adaptive_feedback_memory.json")
        driver_summary["adaptive_memory"] = adaptive_memory
        write_json(out_dir / "adaptive_feedback_driver_summary.json", _json_ready(driver_summary))
        adaptive_workflow_done_so_far += int(current_pilot_steps)
        if bool(proposal.get("converged", False)) and round_no < n_rounds:
            # Gate 1: enforce minimum pilot rounds before any early-stop.
            _min_rounds_es = int(getattr(args, "adaptive_feedback_min_rounds", 2) or 2)
            # Gate 2: CV coverage — block if CV1 or CV2 under-sampled this round.
            _es_block_reason: str = ""
            if round_no < _min_rounds_es:
                _es_block_reason = f"min_rounds={_min_rounds_es} not yet reached (round {round_no})"
            elif _pilot_cv1_blocked or _pilot_cv2_blocked:
                _blocked_axes = ", ".join(filter(None, [
                    "CV1" if _pilot_cv1_blocked else "",
                    "CV2" if _pilot_cv2_blocked else "",
                ]))
                _es_block_reason = f"coverage gate on {_blocked_axes}"
            if _es_block_reason:
                print(f"    Early-stop proposed but blocked ({_es_block_reason}); continuing pilots.")
            else:
                driver_summary["stopped_early"] = True
                driver_summary["stopped_after_round"] = int(round_no)
                driver_summary["early_stop_reason"] = "adaptive feedback converged by score/overlap/exchange/change criteria"
                write_json(out_dir / "adaptive_feedback_driver_summary.json", _json_ready(driver_summary))
                print(f"    Adaptive-feedback converged after round {round_no}; skipping remaining pilot rounds.")
                break

    if skip_final_production:
        driver_summary["final_production_completed"] = False
        driver_summary["final_production_skipped"] = True
        driver_summary["handoff_to_adaptive_production"] = True
        driver_summary["handoff_ready"] = bool(current_windows_2d_csv is not None or current_centers is not None)
        if current_windows_2d_csv is not None:
            driver_summary["handoff_window_mode"] = "explicit_sparse_2d_from_adaptive_feedback"
            driver_summary["handoff_windows_2d_csv"] = str(current_windows_2d_csv)
            driver_summary["handoff_sparse_patch_source_round"] = current_windows_2d_source_round
        elif current_centers is not None:
            driver_summary["handoff_window_mode"] = "rectangular_axis_factorized"
            driver_summary["handoff_centers_A"] = current_centers
            driver_summary["handoff_k_kcal_mol_A2"] = current_k
            driver_summary["handoff_secondary_cv_centers"] = current_secondary_centers
            driver_summary["handoff_secondary_cv_k_kcal_mol"] = current_secondary_k
        else:
            driver_summary["handoff_window_mode"] = "initial_adaptive_windows"
            driver_summary["handoff_warning"] = "No adaptive-feedback proposal was generated, usually because adaptive-feedback-rounds=0; adaptive-production should initialize from its ordinary adaptive windows."
        driver_summary["adaptive_memory_json"] = str(out_dir / "adaptive_feedback_memory.json")
        driver_summary["adaptive_memory"] = adaptive_memory
        write_json(out_dir / "adaptive_feedback_driver_summary.json", _json_ready(driver_summary))
        try:
            summary_payload = write_adaptive_feedback_summary_report(out_dir)
            driver_summary["adaptive_feedback_summary_md"] = str(out_dir / "adaptive_feedback_summary.md")
            driver_summary["adaptive_feedback_summary_json"] = str(out_dir / "adaptive_feedback_summary.json")
            driver_summary["adaptive_feedback_summary_status"] = str(summary_payload.get("final_report_status", "unknown"))
            write_json(out_dir / "adaptive_feedback_driver_summary.json", _json_ready(driver_summary))
        except Exception as exc:
            print(f"WARNING: adaptive-feedback summary report failed: {exc}")
        try:
            write_standardized_output_layout(out_dir)
        except Exception as exc:
            print(f"WARNING: adaptive-feedback output layout failed: {exc}")
        print("Adaptive-feedback pilot workflow complete; final production skipped for adaptive-production handoff.")
        return driver_summary

    final_dir = out_dir / "final_production"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_args = _clone_args(args)
    final_args.out = str(final_dir)
    final_args.gamd_production_steps = int(target_prod_steps)
    final_args.adaptive_feedback_enabled = False
    final_args.adaptive_feedback_target_overlap = float(target_overlap)
    final_args.adaptive_feedback_pilot = False
    final_args.adaptive_feedback_final_production = True
    final_args.adaptive_feedback_rounds_total = int(n_rounds)
    final_args.adaptive_feedback_workflow_start_wall = float(adaptive_workflow_start_wall)
    final_args.adaptive_feedback_workflow_done_before = int(adaptive_workflow_done_so_far)
    final_args.adaptive_feedback_workflow_total_steps = int(adaptive_workflow_total_steps)
    if current_windows_2d_csv is not None:
        # Existing adaptive-feedback spelling controls the refinement.  The final
        # clean run consumes sparse local 2D patch candidates through the explicit
        # window-table path and graph-based neighbor exchange, but users still
        # request the workflow with --window-mode adaptive-feedback.
        final_args.window_mode = "manual"
        final_args.windows_2d_csv = str(current_windows_2d_csv)
        total_replicas = "?"
        try:
            with Path(current_windows_2d_csv).open(newline="") as handle:
                total_replicas = sum(1 for _ in csv.DictReader(handle))
        except Exception:
            pass
        print(f"\n=== Final production with adaptive-feedback sparse 2D proposal: {total_replicas} explicit windows from round {current_windows_2d_source_round}, {target_prod_steps} steps ===")
        print(f"    explicit window table: {current_windows_2d_csv}")
    elif current_centers is not None and current_k is not None:
        final_args.window_mode = "manual"
        final_args.windows_2d_csv = None
        if primary_cv_is_contacts(final_args):
            final_args.contact_centers = [float(x) for x in current_centers]
            final_args.contact_k_kcal = [float(x) for x in current_k]
            final_args.windows_a = []
            final_args.window_k_kcal_a2 = []
        else:
            final_args.windows_a = [float(x) for x in current_centers]
            final_args.window_k_kcal_a2 = [float(x) for x in current_k]
        if secondary_cv_enabled(final_args) and current_secondary_centers is not None:
            final_args.secondary_cv_centers = [float(x) for x in current_secondary_centers]
            if current_secondary_k and str(getattr(final_args, "secondary_cv_k_mode", "fixed") or "fixed").lower() in {"fixed", "constant"}:
                final_args.secondary_cv_k_kcal = float(np.median(np.asarray(current_secondary_k, dtype=float)))
        total_replicas = len(current_centers) * (len(current_secondary_centers) if secondary_cv_enabled(final_args) and current_secondary_centers is not None else 1)
        _pcv_label = primary_cv_label(final_args)
        print(f"\n=== Final production with adaptive-feedback proposal: {len(current_centers)} {_pcv_label} centers, {len(current_secondary_centers) if current_secondary_centers is not None else 1} secondary centers, {total_replicas} replicas, {target_prod_steps} steps ===")
    else:
        # Handles --adaptive-feedback-rounds 0: run a full adaptive production with the initial automatic windows.
        final_args.window_mode = "adaptive"
        final_args.windows_2d_csv = None
        print(f"\n=== Final production with initial adaptive windows: {target_prod_steps} steps ===")
    run_gareus(final_args, final_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress)
    if _graceful_shutdown.is_set():
        print("Graceful shutdown: final production checkpointed; use --resume to continue.", flush=True)
        return driver_summary

    driver_summary["final_production_completed"] = True
    driver_summary["final_production_dir"] = str(final_dir)
    if current_windows_2d_csv is not None:
        driver_summary["final_window_mode"] = "explicit_sparse_2d_from_adaptive_feedback"
        driver_summary["final_windows_2d_csv"] = str(current_windows_2d_csv)
        driver_summary["final_sparse_patch_source_round"] = current_windows_2d_source_round
    elif current_centers is not None:
        driver_summary["final_window_mode"] = "rectangular_axis_factorized"
        driver_summary["final_centers_A"] = current_centers
        driver_summary["final_k_kcal_mol_A2"] = current_k
        driver_summary["final_secondary_cv_centers"] = current_secondary_centers
        driver_summary["final_secondary_cv_k_kcal_mol"] = current_secondary_k
    driver_summary["adaptive_memory_json"] = str(out_dir / "adaptive_feedback_memory.json")
    driver_summary["adaptive_memory"] = adaptive_memory
    write_json(out_dir / "adaptive_feedback_driver_summary.json", _json_ready(driver_summary))
    try:
        summary_payload = write_adaptive_feedback_summary_report(out_dir)
        driver_summary["adaptive_feedback_summary_md"] = str(out_dir / "adaptive_feedback_summary.md")
        driver_summary["adaptive_feedback_summary_json"] = str(out_dir / "adaptive_feedback_summary.json")
        driver_summary["adaptive_feedback_summary_status"] = str(summary_payload.get("final_report_status", "unknown"))
        write_json(out_dir / "adaptive_feedback_driver_summary.json", _json_ready(driver_summary))
    except Exception as exc:
        print(f"WARNING: adaptive-feedback summary report failed: {exc}")
    try:
        write_standardized_output_layout(out_dir)
    except Exception as exc:
        print(f"WARNING: adaptive-feedback output layout failed: {exc}")
    print(f"Adaptive-feedback automatic workflow complete. Final production: {final_dir}")



