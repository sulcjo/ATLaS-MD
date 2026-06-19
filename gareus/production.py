"""
Production run helpers for the GAREUS package.

This module contains the production-side OpenMM/GaMD/REUS orchestration code
extracted from ``gareus_peptide.py``.  It includes the real ``run_gareus``
implementation plus the production-specific helpers it directly relies on:
window table serialization, analysis-array buffering, checkpoint handling,
shared GaMD setup, exchange-stat restoration, final reports and output-layout
normalization.

The production loop now imports its dependencies from modular ``gareus.*``
modules directly; it no longer reaches back into the historical monolith.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .io import BufferedCsvDictWriter, write_json, read_json_file, _json_ready
from .logger import DistanceLogger, is_gamd_production_phase
from .store import ParquetSampleWriter, ParquetExchangeWriter, SegmentRegistry, WindowSnapshot, parse_gamd_boost_components
from .progress import GuiProgressSink, release_openmm_contexts
from .lifecycle import _graceful_shutdown, _register_graceful_shutdown
from .units import kcal_to_kj, kcal_a2_to_kj_nm2
from .system_setup import (
    create_system,
    platform_and_properties,
    setup_platform_and_properties,
    platform_summary,
    replica_platform_properties,
    resolve_cpu_threads_for_replicas,
    make_trajectory_reporter,
    write_solute_only_topology_pdb,
    write_state_pdb,
)
from .seeding import generate_us_starting_states_by_pulling, deserialize_system
from .diagnostics import compute_gamd_reweighting_diagnostics, validate_us_mbar_inputs
from .imports import import_gamd_factory, import_openmm
from .state import _scalar_to_float, _energy_to_kj_mol
from .cv import (
    apply_primary_cv_metadata_to_args,
    choose_cv_atoms,
    format_primary_delta_value,
    prepare_primary_cv_definition,
    primary_center_to_openmm_value,
    primary_cv_is_contacts,
    primary_cv_is_distance,
    primary_cv_label,
    primary_cv_mode,
    primary_cv_units,
    primary_cv_value_from_positions_nm,
    solute_atom_indices,
    primary_k_to_openmm_value,
    primary_k_units,
    primary_openmm_k_units,
    rama_region_definitions,
    rama_map_definitions,
    secondary_cv_enabled,
    secondary_cv_mode,
    secondary_cv_range,
    secondary_cv_target_angles,
    secondary_structure_score_from_positions_nm,
    secondary_structure_torsions,
)
from .windows import (
    build_explicit_2d_neighbor_edges,
    expand_windows_for_secondary_cv,
    choose_windows,
    load_explicit_2d_window_csv,
    set_window,
    _adaptive_window_aggressiveness_settings,
    _rounded_unique_sorted,
    _csv_first_present,
    _csv_float_field,
)
from .analysis import validate_analysis_metadata_readiness
from .forces import (
    validate_openmm_force_group,
    add_umbrella_force,
    add_contact_umbrella_force,
)
from .provenance import initialize_run_manifest, update_run_manifest, finalize_run_manifest

__all__ = [
    "add_secondary_structure_cv_force",
    "add_primary_umbrella_force",
    "make_gamd_integrator",
    "make_production_integrator",
    "production_run_mode",
    "gamd_enabled",
    "window_assignment_rows",
    "write_window_assignment_csv",
    "load_explicit_2d_window_csv",
    "write_explicit_2d_neighbor_graph_files",
    "write_explicit_window_analysis_files",
    "print_window_assignment_table",
    "all_integrator_globals",
    "set_integrator_globals_from_dict",
    "AnalysisArrayWriter",
    "write_standardized_output_layout",
    "write_exchange_tuning_report",
    "write_final_run_report",
    "compare_gamd_global_sets",
    "checkpoint_manifest_path",
    "save_production_checkpoint",
    "sync_scratch_to_main",
    "load_production_checkpoint",
    "restore_exchange_stats_from_csv_if_needed",
    "run_production_probe",
    "run_shared_gamd_setup_article_a",
    "_gibbs_window_proposal_distribution",
    "_gibbs_mh_acceptance_probability",
    "run_gareus",
]



def _flatten_numeric_object(obj, prefix: str, unit=None, energy: bool = False) -> dict[str, float]:
    """Flatten dict/list/scalar results from gamd-openmm diagnostic helpers."""
    out: dict[str, float] = {}
    if obj is None:
        return out
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = [(str(i), v) for i, v in enumerate(obj)]
    else:
        items = [("value", obj)]
    for key, value in items:
        name = f"{prefix}_{key}"
        val = _energy_to_kj_mol(value, unit) if energy and unit is not None else _scalar_to_float(value)
        if val is not None and math.isfinite(float(val)):
            out[str(name)] = float(val)
    return out

def native_gamd_boost_components_kj(integrator, unit) -> dict[str, float]:
    """Read boost potentials using gamd-openmm's native helper when present.

    gamd-openmm exposes get_boost_potentials() on its stage integrators.  The
    exact return shape varies by boost type/version, so normalize it into a
    component dictionary in kJ/mol.  This is more reliable than guessing from
    global-variable names alone.
    """
    for method_name in ("get_boost_potentials", "getBoostPotentials"):
        method = getattr(integrator, method_name, None)
        if not callable(method):
            continue
        try:
            result = method()
        except Exception:
            continue
        comps = _flatten_numeric_object(result, "boost", unit=unit, energy=True)
        # Keep only finite components.  Do not require them to be nonzero; zero
        # is a valid boost early in production or for frames above threshold.
        if comps:
            return comps
    return {}

def native_gamd_statistics(integrator, unit=None) -> dict[str, float]:
    """Best-effort read of gamd-openmm statistics for diagnostics JSON."""
    for method_name in ("get_statistics", "getStatistics"):
        method = getattr(integrator, method_name, None)
        if not callable(method):
            continue
        try:
            result = method()
        except Exception:
            continue
        vals = _flatten_numeric_object(result, "stat", unit=unit, energy=False)
        if vals:
            return vals
    return {}

def infer_gamd_boost_kj_from_globals(globals_now: dict[str, float]) -> Optional[float]:
    """Infer total GaMD boost from a globals dictionary, already in kJ/mol."""
    if not globals_now:
        return None
    candidates = []
    for name, value in globals_now.items():
        lower = str(name).lower()
        if "boost" not in lower:
            continue
        if any(skip in lower for skip in ("sigma", "threshold", "count", "step", "window")):
            continue
        try:
            val = float(value)
        except Exception:
            continue
        if np.isfinite(val):
            candidates.append((str(name), val))
    if not candidates:
        return None
    # Native get_boost_potentials() entries are normalized to kJ/mol and named
    # native_boost_..._kj_mol.  Prefer their sum for dual boosts.
    native = [(n, v) for n, v in candidates if n.lower().startswith("native_boost")]
    if native:
        return float(sum(v for _, v in native))
    totals = [v for n, v in candidates if "total" in n.lower() or "du" in n.lower()]
    if totals:
        return float(totals[-1])
    return float(sum(v for _, v in candidates))

def _add_torsion_score_force(openmm, torsions, target_rad: float, sigma_rad: float, norm_name: str, target_name: str, sigma_name: str = "ss_sigma"):
    """Return a CustomTorsionForce that sums normalized periodic Gaussian scores."""
    force = openmm.CustomTorsionForce(f"{norm_name}*exp(-(1-cos(theta-{target_name}))/({sigma_name}*{sigma_name}))")
    force.addGlobalParameter(str(target_name), float(target_rad))
    force.addGlobalParameter(str(sigma_name), float(sigma_rad))
    force.addGlobalParameter(str(norm_name), 1.0 / max(1, len(torsions)))
    for a, b, c, d in torsions:
        force.addTorsion(int(a), int(b), int(c), int(d), [])
    return force

def primary_secondary_and_potential_from_state(
    context,
    primary_cv_def: dict,
    args,
    unit,
    ss_info: Optional[dict] = None,
    read_potential_energy: bool = True,
) -> tuple[float, float, float]:
    """Read positions once, compute CV observables, and optionally read energy.

    Potential energy is useful diagnostics but not required for REUS exchange or
    MBAR umbrella-bias matrices.  Allowing callers to disable it avoids an extra
    GPU force/energy evaluation at every scalar logging interval without changing
    coordinates, velocities, forces, GaMD state, or exchange probabilities.
    """
    state = context.getState(
        getPositions=True,
        getEnergy=bool(read_potential_energy),
        enforcePeriodicBox=True,
    )
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    primary_value = primary_cv_value_from_positions_nm(pos, primary_cv_def, args)
    ss = secondary_structure_score_from_positions_nm(pos, ss_info) if ss_info and ss_info.get("enabled") else float("nan")
    potential_kj = float("nan")
    if read_potential_energy:
        potential_kj = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    return primary_value, ss, potential_kj


def _ss_scalar_from_sub_cv_values(sub_cv_values, metadata: dict) -> float:
    """Reconstruct the secondary-structure scalar CV from CustomCVForce sub-variable values.

    Mirrors the scalar expression baked into the OpenMM energy function so that
    CustomCVForce.getCollectiveVariableValues() can replace a positions-based
    recompute.  The force already evaluated these values on the GPU during the
    preceding step(); this function is pure Python arithmetic over scalars.

    For rama-map/rama-regions: sub-CVs interleave phi/psi per region in
    definition order — [phi_0, psi_0, phi_1, psi_1, ...].
    For alpha-coil-beta: [alpha_phi, alpha_psi, beta_phi, beta_psi].
    For simple modes (alpha/beta/custom): [ss_phi, ss_psi].
    """
    mode = secondary_cv_mode(metadata)
    arr = np.asarray(sub_cv_values, dtype=np.float64)
    if mode == "alpha-coil-beta":
        return float(0.5 * (arr[0] + arr[1]) - 0.5 * (arr[2] + arr[3]))
    if mode in {"rama-regions", "rama-map"}:
        regions = metadata.get("regions", [])
        if not regions:
            return 0.0
        values = np.array([float(r["value"]) for r in regions], dtype=np.float64)
        phi_scores = arr[0::2]
        psi_scores = arr[1::2]
        scores = 0.5 * (phi_scores + psi_scores)
        denom = float(np.sum(scores)) + 1e-8
        return float(np.dot(values, scores) / denom)
    # alpha, beta, custom: [ss_phi, ss_psi]
    return float(0.5 * (arr[0] + arr[1]))


def _first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None

def _as_float_list(value, name: str) -> list[float]:
    try:
        out = [float(x) for x in value]
    except Exception as exc:
        raise RuntimeError(f"Could not parse {name} from resume metadata: {exc}") from exc
    if not out:
        raise RuntimeError(f"Resume metadata field {name} is empty")
    return out

def integrator_globals(integrator, unit=None, include_all: bool = False) -> dict[str, float]:
    """Return GaMD diagnostic globals plus native boost/statistics helpers.

    Earlier versions only searched CustomIntegrator global names containing the
    literal substring "boost".  That fails for some gamd-openmm versions/boost
    types where the public boost potential is exposed by get_boost_potentials()
    rather than by a conveniently named global.  This function now includes both
    paths.
    """
    all_globals = all_integrator_globals(integrator)
    if include_all:
        out = dict(all_globals)
    else:
        interesting = (
            "boost", "sigma", "vmax", "vmin", "vavg", "energy",
            "threshold", "k0", "step", "stage", "window", "potential",
        )
        out = {name: value for name, value in all_globals.items() if any(s in name.lower() for s in interesting)}
    # Add native stage/step where available; these names are stable in
    # gamd-openmm's GamdStageIntegrator API.
    for method_name, key in (("get_stage", "native_stage"), ("get_step_count", "native_stepCount"), ("get_window_count", "native_windowCount")):
        method = getattr(integrator, method_name, None)
        if callable(method):
            try:
                val = _scalar_to_float(method())
                if val is not None and math.isfinite(float(val)):
                    out[key] = float(val)
            except Exception:
                pass
    if unit is not None:
        for name, value in native_gamd_boost_components_kj(integrator, unit).items():
            out[f"native_{name}_kj_mol"] = float(value)
    out.update(native_gamd_statistics(integrator, unit=unit))
    return out

def extract_gamd_boost_kj(integrator, unit, globals_now: Optional[dict[str, float]] = None) -> tuple[Optional[float], dict[str, float], str]:
    """Return total boost, component boosts, and source label for this frame."""
    comps = native_gamd_boost_components_kj(integrator, unit)
    if comps:
        return float(sum(comps.values())), comps, "get_boost_potentials"
    if globals_now is None:
        globals_now = integrator_globals(integrator, unit=unit, include_all=False)
    boost = infer_gamd_boost_kj_from_globals(globals_now)
    if boost is not None:
        return float(boost), {}, "integrator_globals"
    return None, {}, "unavailable"


def _gibbs_window_proposal_distribution(
    beta: float,
    bias_matrix_kj: np.ndarray,
    replica_index: int,
    current_window: int,
    replica_of_window: np.ndarray,
) -> dict[str, np.ndarray]:
    """Heat-bath proposal over target windows for one replica transposition."""
    rep = int(replica_index)
    wi = int(current_window)
    holders = np.asarray(replica_of_window, dtype=np.int64)
    all_windows = np.arange(holders.size, dtype=np.int32)
    valid_mask = holders >= 0
    valid_windows = all_windows[valid_mask]
    target_reps = holders[valid_mask]
    if valid_windows.size <= 0:
        return {
            "windows": np.asarray([], dtype=np.int32),
            "target_replicas": np.asarray([], dtype=np.int64),
            "deltas_kj": np.asarray([], dtype=np.float64),
            "probabilities": np.asarray([], dtype=np.float64),
        }

    bias = np.asarray(bias_matrix_kj, dtype=np.float64)
    old_e = bias[wi, rep] + bias[valid_windows, target_reps]
    new_e = bias[valid_windows, rep] + bias[wi, target_reps]
    delta = np.asarray(new_e - old_e, dtype=np.float64)
    stay_idx = np.where(valid_windows == wi)[0]
    if stay_idx.size:
        delta[int(stay_idx[0])] = 0.0
    log_weights = np.clip(-float(beta) * delta, -745.0, 0.0)
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        return {
            "windows": np.asarray([], dtype=np.int32),
            "target_replicas": np.asarray([], dtype=np.int64),
            "deltas_kj": np.asarray([], dtype=np.float64),
            "probabilities": np.asarray([], dtype=np.float64),
        }
    valid_windows = valid_windows[finite]
    target_reps = target_reps[finite]
    delta = delta[finite]
    log_weights = log_weights[finite]
    m = float(np.max(log_weights))
    weights = np.exp(log_weights - m)
    sw = float(np.sum(weights))
    if not math.isfinite(sw) or sw <= 0.0:
        probabilities = np.full(valid_windows.shape, 1.0 / max(1, valid_windows.size), dtype=np.float64)
    else:
        probabilities = weights / sw
    return {
        "windows": valid_windows.astype(np.int32, copy=False),
        "target_replicas": target_reps.astype(np.int64, copy=False),
        "deltas_kj": delta.astype(np.float64, copy=False),
        "probabilities": probabilities.astype(np.float64, copy=False),
    }


def _gibbs_mh_acceptance_probability(
    delta_kj: float,
    beta: float,
    q_forward: float,
    q_reverse: float,
) -> float:
    """Metropolis-Hastings correction for nonuniform Gibbs-walk proposals."""
    try:
        delta = float(delta_kj)
        beta_val = float(beta)
        qf = float(q_forward)
        qr = float(q_reverse)
    except Exception:
        return 0.0
    if not (math.isfinite(delta) and math.isfinite(beta_val) and math.isfinite(qf) and math.isfinite(qr)):
        return 0.0
    if qf <= 0.0 or qr <= 0.0:
        return 0.0
    log_alpha = -beta_val * delta + math.log(qr) - math.log(qf)
    if log_alpha >= 0.0:
        return 1.0
    if log_alpha < -745.0:
        return 0.0
    return float(math.exp(log_alpha))

def load_resume_run_definition(out_dir: Path, topology, args, manifest: Optional[dict] = None) -> dict:
    """Recover CV/window/GaMD bookkeeping from previous outputs for true --resume."""
    out_dir = Path(out_dir)
    manifest = manifest if isinstance(manifest, dict) else read_json_file(checkpoint_manifest_path(out_dir), {}) or {}
    metadata = read_json_file(out_dir / "gareus_metadata.json", {}) or {}
    pymbar = read_json_file(out_dir / "umbrella_pymbar_metadata.json", {}) or {}
    shared = read_json_file(out_dir / "shared_gamd_setup_globals.json", {}) or {}

    centers = _first_present(
        manifest.get("windows_A"),
        manifest.get("window_centers_A"),
        metadata.get("windows_A"),
        pymbar.get("window_centers_A"),
    )
    k_values = _first_present(
        manifest.get("window_k_kcal_mol_A2"),
        manifest.get("window_k_kcal_per_mol_A2"),
        metadata.get("window_k_kcal_per_mol_A2"),
        pymbar.get("window_k_kcal_mol_A2"),
    )
    if centers is None or k_values is None:
        if str(getattr(args, "window_mode", "adaptive")) == "manual" and getattr(args, "windows_a", None):
            centers = args.windows_a
            if getattr(args, "window_k_kcal_a2", None):
                k_values = args.window_k_kcal_a2
                if len(k_values) == 1 and len(centers) > 1:
                    k_values = list(k_values) * len(centers)
            else:
                k_values = [float(args.default_window_k_kcal_a2)] * len(centers)
        else:
            raise RuntimeError(
                "--resume found production checkpoints but could not recover umbrella windows. "
                "Keep gareus_metadata.json or umbrella_pymbar_metadata.json in the output directory, "
                "or rerun with the exact manual --windows-a/--window-k-kcal-a2 values."
            )
    centers_a = np.asarray(_as_float_list(centers, "window centers"), dtype=float)
    k_list = _as_float_list(k_values, "window force constants")
    if len(k_list) != len(centers_a):
        raise RuntimeError(f"Resume metadata has {len(centers_a)} centers but {len(k_list)} force constants")

    cv_atom1 = _first_present(manifest.get("cv_atom1_index"), metadata.get("cv_atom1_index"), pymbar.get("cv_atom1_index"))
    cv_atom2 = _first_present(manifest.get("cv_atom2_index"), metadata.get("cv_atom2_index"), pymbar.get("cv_atom2_index"))
    cv_label = _first_present(manifest.get("cv_label"), metadata.get("cv_label"), pymbar.get("cv_label"))
    if cv_atom1 is None or cv_atom2 is None:
        cv_atom1, cv_atom2, cv_label = choose_cv_atoms(topology, args)
    else:
        cv_atom1 = int(cv_atom1)
        cv_atom2 = int(cv_atom2)
        cv_label = str(cv_label or f"atom {cv_atom1}--atom {cv_atom2}")

    calib_steps = _first_present(
        manifest.get("shared_gamd_calibration_steps"),
        metadata.get("shared_gamd_calibration_steps"),
        pymbar.get("shared_gamd_calibration_steps"),
        shared.get("calibration_steps"),
    )
    if calib_steps is None:
        calib_steps = int(
            args.gamd_cmd_prep_steps
            + args.gamd_cmd_steps
            + args.gamd_equil_prep_steps
            + args.gamd_equil_steps
        )

    secondary_meta = _first_present(
        manifest.get("secondary_cv"),
        metadata.get("secondary_cv"),
        pymbar.get("secondary_cv"),
        {"enabled": False},
    )
    secondary_centers = _first_present(
        manifest.get("secondary_cv_centers"),
        metadata.get("secondary_cv_centers"),
        pymbar.get("secondary_cv_centers"),
    )
    secondary_k = _first_present(
        manifest.get("secondary_cv_k_kcal_mol"),
        metadata.get("secondary_cv_k_kcal_mol"),
        pymbar.get("secondary_cv_k_kcal_mol"),
    )
    if isinstance(secondary_meta, dict) and secondary_meta.get("enabled"):
        if secondary_centers is None or secondary_k is None:
            raise RuntimeError("Resume metadata says secondary CV was enabled, but secondary centers/k arrays are missing")
        secondary_centers = np.asarray(_as_float_list(secondary_centers, "secondary CV centers"), dtype=float)
        secondary_k = _as_float_list(secondary_k, "secondary CV force constants")
        if len(secondary_centers) != len(centers_a) or len(secondary_k) != len(centers_a):
            raise RuntimeError("Resume secondary-CV metadata length does not match window count")
    else:
        secondary_meta = {"enabled": False}
        secondary_centers = None
        secondary_k = None

    return {
        "centers_a": centers_a,
        "k_list": [float(x) for x in k_list],
        "secondary_cv_metadata": secondary_meta,
        "secondary_cv_centers": secondary_centers,
        "secondary_cv_k_kcal_list": secondary_k,
        "cv_atom1": int(cv_atom1),
        "cv_atom2": int(cv_atom2),
        "cv_label": str(cv_label),
        "window_metadata": {"mode": "resume_from_previous_outputs", "source_files": ["production_checkpoint_manifest.json", "gareus_metadata.json", "umbrella_pymbar_metadata.json"]},
        "shared_gamd_globals_all": dict(shared.get("all_globals", {}) or {}),
        "shared_gamd_globals_interesting": dict(shared.get("interesting_globals", {}) or {}),
        "calib_steps": int(calib_steps),
        "metadata": metadata,
        "pymbar_metadata": pymbar,
    }

def _safe_relative_path(path: Path, base: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(base).resolve()))
    except Exception:
        return str(path)

def _mkdir_and_link_or_note(src: Path, dst: Path) -> dict:
    """Create a non-invasive symlink mirror entry; never move original files."""
    src = Path(src)
    dst = Path(dst)
    row = {"source": str(src), "mirror": str(dst), "kind": "missing", "ok": False}
    if not src.exists():
        return row
    dst.parent.mkdir(parents=True, exist_ok=True)
    row["kind"] = "symlink"
    try:
        if dst.exists() or dst.is_symlink():
            try:
                if dst.resolve() == src.resolve():
                    row["ok"] = True
                    row["kind"] = "existing_symlink"
                    return row
            except Exception:
                pass
            return {**row, "kind": "exists", "ok": True}
        rel = os.path.relpath(src.resolve(), dst.parent.resolve())
        os.symlink(rel, dst)
        row["ok"] = True
        return row
    except Exception as exc:
        row["kind"] = "manifest_only"
        row["error"] = str(exc)
        return row

def _exchange_pair_rows(exchange_stats: dict) -> list[dict]:
    rows = []
    pair_meta = exchange_stats.get("pair_metadata", {}) if isinstance(exchange_stats, dict) else {}
    for key, st in sorted((exchange_stats.get("pairs", {}) or {}).items()):
        try:
            attempts = int(st.get("attempts", 0) or 0)
            accepted = int(st.get("accepted", 0) or 0)
        except Exception:
            continue
        frac = accepted / max(1, attempts)
        meta = pair_meta.get(key, {}) if isinstance(pair_meta, dict) else {}
        try:
            left, right = [int(x) for x in str(key).split("-", 1)]
        except Exception:
            left, right = -1, -1
        rows.append({
            "pair": str(key),
            "window_i": left,
            "window_j": right,
            "attempts": attempts,
            "accepted": accepted,
            "acceptance_fraction": float(frac),
            "edge_type": str(meta.get("edge_type", "")),
            "normalized_distance": meta.get("normalized_distance", ""),
            "neighbor_source": str(meta.get("neighbor_source", "")),
        })
    return rows

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

def explicit_2d_neighbor_pairs_for_exchange(centers_a, secondary_cv_centers, parity: int, args=None) -> tuple[list[tuple[int, int, str, float]], int, list[dict]]:
    """Return a disjoint subset of graph edges for one exchange interval."""
    edges = build_explicit_2d_neighbor_edges(centers_a, secondary_cv_centers, args=args)
    slots = max(1, int(getattr(args, "explicit_2d_exchange_slots", 4) if args is not None else 4))
    slot = int(parity) % slots
    chosen = []
    used = set()
    # Round-robin edge coloring by edge index, then greedy disjoint filtering.
    for idx, edge in enumerate(edges):
        if idx % slots != slot:
            continue
        wi, wj = int(edge["wi"]), int(edge["wj"])
        if wi in used or wj in used:
            continue
        chosen.append((wi, wj, str(edge.get("edge_type", "explicit_2d_graph")), float(edge.get("normalized_distance", float("nan")))))
        used.add(wi)
        used.add(wj)
    # If a slot was empty for a small graph, try all edges greedily so exchanges do not stall.
    if not chosen and edges:
        for edge in edges:
            wi, wj = int(edge["wi"]), int(edge["wj"])
            if wi in used or wj in used:
                continue
            chosen.append((wi, wj, str(edge.get("edge_type", "explicit_2d_graph")), float(edge.get("normalized_distance", float("nan")))))
            used.add(wi)
            used.add(wj)
    return chosen, slots, edges

def explicit_window_analysis_rows(
    centers_a,
    k_list,
    secondary_cv_centers=None,
    secondary_cv_k_kcal_list=None,
    secondary_cv_metadata: Optional[dict] = None,
    window_metadata: Optional[dict] = None,
) -> list[dict]:
    """Return one explicit analysis metadata row per thermodynamic state.

    This table is deliberately window-major and does not assume a rectangular 2D
    grid.  It is the canonical mapping for sparse 2D MBAR/reweighting: sample
    rows reference windows by integer index, and this table defines each index's
    distance and optional secondary-CV umbrella parameters.
    """
    centers = np.asarray(centers_a, dtype=float)
    k_arr = np.asarray(k_list, dtype=float)
    n = int(len(centers))
    sec = np.asarray(secondary_cv_centers, dtype=float) if secondary_cv_centers is not None else None
    sec_k = np.asarray(secondary_cv_k_kcal_list, dtype=float) if secondary_cv_k_kcal_list is not None else None
    meta = secondary_cv_metadata or {}
    wmeta = window_metadata or {}
    normalized = []
    if isinstance(meta, dict):
        normalized = list(meta.get("normalized_rows", []) or [])
    if not normalized and isinstance(wmeta, dict):
        normalized = list(wmeta.get("normalized_rows", []) or [])
    norm_by_window = {}
    for row in normalized:
        try:
            norm_by_window[int(row.get("window", len(norm_by_window)))] = row
        except Exception:
            continue
    explicit_2d = bool(isinstance(meta, dict) and meta.get("explicit_2d_windows", False))
    rectangular = bool(isinstance(meta, dict) and meta.get("grid", False))
    primary_mode = str((wmeta or {}).get("primary_cv", "distance") or "distance")
    contact_mode = primary_mode == "nonlocal-contacts"
    primary_label = str((wmeta or {}).get("primary_cv_label", "nonlocal contact fraction" if contact_mode else "terminal distance"))
    primary_units = str((wmeta or {}).get("primary_cv_units", "dimensionless" if contact_mode else "A"))
    primary_k_units_val = str((wmeta or {}).get("primary_k_units", "kcal/mol/CV^2" if contact_mode else "kcal/mol/A^2"))
    rows: list[dict] = []
    rt_mode = str((meta or {}).get("mode", "none"))
    for i in range(n):
        src = norm_by_window.get(i, {}) or {}
        window_type = str(src.get("window_type") or src.get("source") or ("grid" if rectangular else "explicit_sparse" if explicit_2d else "distance_only"))
        lifecycle = str(src.get("lifecycle") or src.get("patch_lifecycle") or window_type)
        s0 = float(sec[i]) if sec is not None and i < len(sec) and math.isfinite(float(sec[i])) else float("nan")
        sk = float(sec_k[i]) if sec_k is not None and i < len(sec_k) and math.isfinite(float(sec_k[i])) else float("nan")
        rows.append({
            "window": int(i),
            "distance_center_A": float(centers[i]),
            "distance_center_nm": "" if contact_mode else float(centers[i]) * 0.1,
            "distance_k_kcal_mol_A2": float(k_arr[i]),
            "distance_k_kj_mol_nm2": "" if contact_mode else float(kcal_a2_to_kj_nm2(k_arr[i])),
            "primary_cv": primary_mode,
            "primary_cv_label": primary_label,
            "primary_cv_units": primary_units,
            "primary_center": float(centers[i]),
            "primary_k": float(k_arr[i]),
            "primary_k_units": primary_k_units_val,
            "primary_openmm_k": float(kcal_to_kj(k_arr[i]) if contact_mode else kcal_a2_to_kj_nm2(k_arr[i])),
            "primary_openmm_k_units": "kJ/mol/CV^2" if contact_mode else "kJ/mol/nm^2",
            "secondary_cv_center": "" if not math.isfinite(s0) else float(s0),
            "secondary_cv_k_kcal_mol": "" if not math.isfinite(sk) else float(sk),
            "secondary_cv_k_kj_mol": "" if not math.isfinite(sk) else float(kcal_to_kj(sk)),
            "secondary_cv_mode": rt_mode if math.isfinite(s0) else "none",
            "window_type": window_type,
            "lifecycle": lifecycle,
            "source_csv": str(src.get("source_csv") or (meta.get("source_csv") if isinstance(meta, dict) else "") or (wmeta.get("source_csv") if isinstance(wmeta, dict) else "")),
            "source_row": src.get("source_row", ""),
            "explicit_2d": int(explicit_2d),
            "rectangular_grid": int(rectangular),
        })
    return rows

def _latest_exact_gamd_globals_from_samples(out_dir: Path, nrep: int, absolute_step: int) -> list[Optional[dict]]:
    """Recover per-replica CustomIntegrator globals from samples.csv at an exact step.

    This is only a compatibility fallback for checkpoints written before the
    checkpoint manifest stored replica_integrator_globals_all.  It requires
    samples.csv to contain gamd_globals_json rows, which are only present when
    --write-gamd-globals-json was enabled.  Exact-step recovery is deliberate:
    applying globals from an older sample can move the GaMD stage/averages
    backward relative to the binary checkpoint.
    """
    path = Path(out_dir) / "samples.csv"
    out: list[Optional[dict]] = [None] * int(nrep)
    if not path.exists() or path.stat().st_size <= 0:
        return out
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if "gamd_globals_json" not in (reader.fieldnames or []):
                return out
            for row in reader:
                try:
                    step = int(float(row.get("step", "nan")))
                    if step != int(absolute_step):
                        continue
                    rep = int(float(row.get("replica", "nan")))
                    if rep < 0 or rep >= int(nrep):
                        continue
                    raw = str(row.get("gamd_globals_json", "") or "").strip()
                    if not raw:
                        continue
                    vals = json.loads(raw)
                    if isinstance(vals, dict) and vals:
                        out[rep] = vals
                except Exception:
                    continue
    except Exception:
        return [None] * int(nrep)
    return out

def _restore_gamd_integrator_globals_after_checkpoint(out_dir: Path, sims: list, manifest: dict) -> dict:
    """Explicitly restore saved GaMD CustomIntegrator globals after checkpoint load.

    OpenMM binary checkpoints are expected to restore Context state, but for
    gamd-openmm it is safer to make the restart auditable: save all readable
    CustomIntegrator globals in the manifest and set them back after loading.
    For older checkpoints, try an exact-step samples.csv fallback if verbose
    gamd_globals_json was available.
    """
    nrep = len(sims)
    absolute_step = int(manifest.get("absolute_step", 0) or 0)
    manifest_globals = manifest.get("replica_integrator_globals_all")
    source = "manifest"
    if not isinstance(manifest_globals, list) or len(manifest_globals) != nrep:
        recovered = _latest_exact_gamd_globals_from_samples(out_dir, nrep, absolute_step)
        if any(isinstance(x, dict) and x for x in recovered):
            manifest_globals = recovered
            source = "samples.csv exact-step fallback"
        else:
            manifest_globals = [None] * nrep
            source = "none"

    rows = []
    restored = 0
    skipped_replicas = 0
    for r, sim in enumerate(sims):
        before = all_integrator_globals(sim.integrator)
        vals = manifest_globals[r] if r < len(manifest_globals) else None
        copied = {}
        skipped = {}
        if isinstance(vals, dict) and vals:
            copied, skipped = set_integrator_globals_from_dict(sim.integrator, vals)
            if copied:
                restored += 1
        else:
            skipped_replicas += 1
        after = all_integrator_globals(sim.integrator)
        cmp_to_saved = compare_gamd_global_sets(vals if isinstance(vals, dict) else {}, after) if vals else {"missing_count": 0, "extra_count": 0, "mismatch_count": 0, "missing": [], "extra": [], "mismatches": []}
        boost_kj, boost_components, boost_source = extract_gamd_boost_kj(sim.integrator, import_openmm()[2], after)
        rows.append({
            "replica": int(r),
            "source": source if isinstance(vals, dict) and vals else "none",
            "saved_global_count": int(len(vals)) if isinstance(vals, dict) else 0,
            "copied_count": int(len(copied)),
            "skipped_count": int(len(skipped)),
            "post_restore_global_count": int(len(after)),
            "post_restore_boost_kj_mol": float(boost_kj) if boost_kj is not None and math.isfinite(float(boost_kj)) else None,
            "post_restore_boost_source": boost_source,
            "skipped_globals_sample": list(skipped)[:20],
            "comparison_to_saved": cmp_to_saved,
        })
    ok = restored == nrep and all(int(row.get("comparison_to_saved", {}).get("mismatch_count", 0) or 0) == 0 for row in rows if row.get("source") != "none")
    report = {
        "schema": "gareus_resume_gamd_integrator_restore_v1",
        "absolute_step": int(absolute_step),
        "n_replicas": int(nrep),
        "source": source,
        "restored_replicas": int(restored),
        "skipped_replicas": int(skipped_replicas),
        "ok": bool(ok),
        "rows": rows,
        "note": "GaMD/CustomIntegrator globals are explicitly restored from checkpoint manifest when available. If source is none, this resume relies only on OpenMM binary checkpoint internals and should be treated with caution.",
    }
    try:
        write_json(Path(out_dir) / "resume_gamd_integrator_restore_report.json", _json_ready(report))
    except Exception:
        pass
    return report


def run_adaptive_feedback_dispatcher(*args, **kwargs):
    """Lazy bridge to adaptive-feedback dispatcher to avoid import cycles."""
    from .adaptive_feedback import run_adaptive_feedback_dispatcher as _impl
    return _impl(*args, **kwargs)


def run_adaptive_feedback_dispatcher_2d(*args, **kwargs):
    """Lazy bridge to 2D adaptive-feedback dispatcher to avoid import cycles."""
    from .adaptive_feedback import run_adaptive_feedback_dispatcher_2d as _impl
    return _impl(*args, **kwargs)


def add_secondary_structure_cv_force(openmm, system, topology, args, force_group: int = 29) -> dict:
    """Add an optional harmonic bias on a smooth backbone secondary-structure CV.

    Modes ``alpha``, ``beta``, and ``custom`` use a 0..1 content score:

        ss = 0.5 * (mean_phi(score(phi, phi0)) + mean_psi(score(psi, psi0)))

    Mode ``alpha-coil-beta`` uses a signed transition coordinate:

        ss = alpha_content - beta_content

    where alpha-like structures are near +1, beta-like structures are near -1,
    and disordered/coil-like regions tend to sit near 0 because both content
    scores are low.  This is still a lightweight phi/psi proxy, not DSSP.
    """
    if not secondary_cv_enabled(args):
        return {"enabled": False}
    force_group = validate_openmm_force_group(force_group, "--secondary-cv-force-group")
    phi_torsions, psi_torsions = secondary_structure_torsions(topology)
    if not phi_torsions and not psi_torsions:
        raise RuntimeError("--secondary-cv was requested but no backbone phi/psi torsions could be identified")

    mode = secondary_cv_mode(args)
    sigma_deg = float(getattr(args, "secondary_cv_sigma_deg", 35.0) or 35.0)
    sigma = max(math.radians(1.0), math.radians(sigma_deg))

    if mode == "alpha-coil-beta":
        alpha_phi0 = math.radians(-60.0)
        alpha_psi0 = math.radians(-45.0)
        beta_phi0 = math.radians(-135.0)
        beta_psi0 = math.radians(135.0)
        cv_force = openmm.CustomCVForce(
            "0.5*ss_k*((0.5*(alpha_phi+alpha_psi)-0.5*(beta_phi+beta_psi))-ss0)^2"
        )
        cv_force.addGlobalParameter("ss_k", 0.0)
        cv_force.addGlobalParameter("ss0", 0.0)
        cv_force.addCollectiveVariable("alpha_phi", _add_torsion_score_force(openmm, phi_torsions, alpha_phi0, sigma, "norm_alpha_phi", "alpha_phi0"))
        cv_force.addCollectiveVariable("alpha_psi", _add_torsion_score_force(openmm, psi_torsions, alpha_psi0, sigma, "norm_alpha_psi", "alpha_psi0"))
        cv_force.addCollectiveVariable("beta_phi", _add_torsion_score_force(openmm, phi_torsions, beta_phi0, sigma, "norm_beta_phi", "beta_phi0"))
        cv_force.addCollectiveVariable("beta_psi", _add_torsion_score_force(openmm, psi_torsions, beta_psi0, sigma, "norm_beta_psi", "beta_psi0"))
        cv_force.setForceGroup(int(force_group))
        system.addForce(cv_force)
        return {
            "enabled": True,
            "mode": "alpha-coil-beta",
            "label": "alpha-coil-beta signed transition coordinate (+alpha, 0 coil, -beta)",
            "range_min": -1.0,
            "range_max": 1.0,
            "coil_center": 0.0,
            "alpha_phi0_deg": math.degrees(alpha_phi0),
            "alpha_psi0_deg": math.degrees(alpha_psi0),
            "beta_phi0_deg": math.degrees(beta_phi0),
            "beta_psi0_deg": math.degrees(beta_psi0),
            "sigma_deg": sigma_deg,
            "force_group": int(force_group),
            "n_phi_torsions": int(len(phi_torsions)),
            "n_psi_torsions": int(len(psi_torsions)),
            "phi_torsions": [list(map(int, t)) for t in phi_torsions],
            "psi_torsions": [list(map(int, t)) for t in psi_torsions],
        }

    if mode in {"rama-regions", "rama-map"}:
        regions = rama_map_definitions() if mode == "rama-map" else rama_region_definitions()
        score_terms = []
        weighted_terms = []
        cv_force = openmm.CustomCVForce("0")
        cv_force.addGlobalParameter("ss_k", 0.0)
        cv_force.addGlobalParameter("ss0", 0.0)
        for region in regions:
            name = str(region["name"])
            phi_name = f"rama_{name}_phi"
            psi_name = f"rama_{name}_psi"
            score_term = f"0.5*({phi_name}+{psi_name})"
            score_terms.append(score_term)
            weighted_terms.append(f"({float(region['value']):.12g})*({score_term})")
            phi0 = math.radians(float(region["phi_deg"]))
            psi0 = math.radians(float(region["psi_deg"]))
            cv_force.addCollectiveVariable(
                phi_name,
                _add_torsion_score_force(openmm, phi_torsions, phi0, sigma, f"norm_{phi_name}", f"{phi_name}0"),
            )
            cv_force.addCollectiveVariable(
                psi_name,
                _add_torsion_score_force(openmm, psi_torsions, psi0, sigma, f"norm_{psi_name}", f"{psi_name}0"),
            )
        denom = " + ".join(score_terms)
        numer = " + ".join(weighted_terms)
        cv_expr = f"(({numer})/(1.0e-8 + ({denom})))"
        cv_force.setEnergyFunction(f"0.5*ss_k*(({cv_expr})-ss0)^2")
        cv_force.setForceGroup(int(force_group))
        system.addForce(cv_force)
        return {
            "enabled": True,
            "mode": mode,
            "label": "explicit Ramachandran basin map (beta/PPII/right-alpha/left-alpha)" if mode == "rama-map" else "soft Ramachandran-region coordinate (beta/PPII/turn/alpha/left-alpha)",
            "range_min": -1.0,
            "range_max": 1.0,
            "region_values": {str(r["name"]): float(r["value"]) for r in regions},
            "regions": [dict(r) for r in regions],
            "sigma_deg": sigma_deg,
            "force_group": int(force_group),
            "n_phi_torsions": int(len(phi_torsions)),
            "n_psi_torsions": int(len(psi_torsions)),
            "phi_torsions": [list(map(int, t)) for t in phi_torsions],
            "psi_torsions": [list(map(int, t)) for t in psi_torsions],
            "note": "Rama-map secondary CV: one dimensionless map coordinate whose centers correspond to explicit phi/psi basins, including native left-alpha. It is not a dense phi/psi grid." if mode == "rama-map" else "Scalar softmax-like Ramachandran basin coordinate; useful as an adaptive umbrella ladder but not a full 2D phi/psi free-energy surface.",
        }

    phi0, psi0, label = secondary_cv_target_angles(args)
    phi_force = _add_torsion_score_force(openmm, phi_torsions, phi0, sigma, "norm_phi", "phi0")
    psi_force = _add_torsion_score_force(openmm, psi_torsions, psi0, sigma, "norm_psi", "psi0")

    cv_force = openmm.CustomCVForce("0.5*ss_k*(0.5*(ss_phi+ss_psi)-ss0)^2")
    cv_force.addGlobalParameter("ss_k", 0.0)
    cv_force.addGlobalParameter("ss0", 0.0)
    cv_force.addCollectiveVariable("ss_phi", phi_force)
    cv_force.addCollectiveVariable("ss_psi", psi_force)
    cv_force.setForceGroup(int(force_group))
    system.addForce(cv_force)
    return {
        "enabled": True,
        "mode": mode,
        "label": label,
        "range_min": 0.0,
        "range_max": 1.0,
        "phi0_deg": math.degrees(phi0),
        "psi0_deg": math.degrees(psi0),
        "sigma_deg": sigma_deg,
        "force_group": int(force_group),
        "n_phi_torsions": int(len(phi_torsions)),
        "n_psi_torsions": int(len(psi_torsions)),
        "phi_torsions": [list(map(int, t)) for t in phi_torsions],
        "psi_torsions": [list(map(int, t)) for t in psi_torsions],
    }

def add_primary_umbrella_force(openmm, system, primary_cv_def: dict, args, force_group: int = 31):
    if primary_cv_mode(primary_cv_def) == "nonlocal-contacts":
        return add_contact_umbrella_force(openmm, system, list(primary_cv_def.get("contact_pairs", [])), args, force_group)
    return add_umbrella_force(openmm, system, int(primary_cv_def["cv_atom1"]), int(primary_cv_def["cv_atom2"]), force_group)

def production_run_mode(args) -> str:
    """Return the canonical high-level dynamics mode.

    ``gamd`` preserves the historical package behavior. ``cmd`` runs the same
    umbrella/REUS workflow with a plain LangevinMiddleIntegrator and does not
    require gamd-openmm. ``hmr-cmd`` is the same conventional workflow with HMR
    enabled. ``hmr-gamd`` is a convenience alias for GaMD with HMR enabled during
    system construction.
    """
    mode = str(getattr(args, "run_mode", "gamd") or "gamd").strip().lower().replace("_", "-")
    if mode not in {"cmd", "hmr-cmd", "gamd", "hmr-gamd"}:
        mode = "gamd"
    return mode


def gamd_enabled(args) -> bool:
    """Return True when production should use gamd-openmm integrators."""
    return production_run_mode(args) in {"gamd", "hmr-gamd"}


def make_cmd_integrator(openmm, args, unit):
    """Create a plain conventional-MD LangevinMiddleIntegrator."""
    integrator = openmm.LangevinMiddleIntegrator(
        float(args.temperature_k) * unit.kelvin,
        float(args.friction_per_ps) / unit.picosecond,
        float(args.timestep_fs) * unit.femtosecond,
    )
    try:
        integrator.setRandomNumberSeed(int(args.seed))
    except Exception:
        pass
    return integrator, {
        "mode": production_run_mode(args),
        "description": "Conventional HMR Langevin MD production integrator; gamd-openmm disabled." if production_run_mode(args) == "hmr-cmd" else "Conventional Langevin MD production integrator; gamd-openmm disabled.",
        "temperature_K": float(args.temperature_k),
        "friction_per_ps": float(args.friction_per_ps),
        "timestep_fs": float(args.timestep_fs),
    }


def make_production_integrator(openmm, system, args, unit):
    """Create the production integrator selected by --run-mode."""
    if gamd_enabled(args):
        return make_gamd_integrator(system, args, unit)
    return make_cmd_integrator(openmm, args, unit)


def make_gamd_integrator(system, args, unit):
    GamdIntegratorFactory = import_gamd_factory()
    factory = GamdIntegratorFactory()
    total_steps = (
        args.gamd_cmd_prep_steps
        + args.gamd_cmd_steps
        + args.gamd_equil_prep_steps
        + args.gamd_equil_steps
        + args.gamd_production_steps
    )
    sigma0p = args.sigma0p_kcal_mol * unit.kilocalories_per_mole
    sigma0d = args.sigma0d_kcal_mol * unit.kilocalories_per_mole
    result = factory.get_integrator(
        args.gamd_boost_type,
        system,
        args.temperature_k * unit.kelvin,
        args.timestep_fs * unit.femtosecond,
        args.gamd_cmd_prep_steps,
        args.gamd_cmd_steps,
        args.gamd_equil_prep_steps,
        args.gamd_equil_steps,
        total_steps,
        args.gamd_averaging_window,
        sigma0p=sigma0p,
        sigma0d=sigma0d,
    )
    integrator = result[2]
    integrator.setRandomNumberSeed(args.seed)
    try:
        integrator.setFriction(args.friction_per_ps / unit.picosecond)
    except Exception:
        pass
    return integrator, result

def window_assignment_rows(centers_a: np.ndarray, k_list: list[float], temperature_k: float, secondary_centers=None, secondary_k_list=None, args=None) -> list[dict]:
    """Return an explicit table of umbrella centers and force constants.

    Legacy distance-named columns are preserved. In nonlocal-contact mode they are
    compatibility aliases for the generic primary-CV columns.
    """
    centers = np.asarray(centers_a, dtype=float)
    k_arr = np.asarray(k_list, dtype=float)
    ss_centers = np.asarray(secondary_centers, dtype=float) if secondary_centers is not None else None
    ss_k_arr = np.asarray(secondary_k_list, dtype=float) if secondary_k_list is not None else None
    rt_kcal_mol = 0.00198720425864083 * float(temperature_k)
    contact_mode = primary_cv_is_contacts(args or "distance")
    rows = []
    for i, (center, k) in enumerate(zip(centers, k_arr)):
        prev_spacing = float(center - centers[i - 1]) if i > 0 else ""
        next_spacing = float(centers[i + 1] - center) if i + 1 < centers.size else ""
        sigma = math.sqrt(rt_kcal_mol / float(k)) if float(k) > 0 else float("inf")
        row = {
            "window": int(i),
            "center_A": float(center),
            "k_kcal_mol_A2": float(k),
            "k_kj_mol_nm2": "" if contact_mode else float(kcal_a2_to_kj_nm2(k)),
            "harmonic_sigma_A": float(sigma),
            "spacing_to_previous_A": prev_spacing,
            "spacing_to_next_A": next_spacing,
            "primary_cv": primary_cv_mode(args or "distance"),
            "primary_cv_label": primary_cv_label(args or "distance"),
            "primary_cv_units": primary_cv_units(args or "distance"),
            "primary_center": float(center),
            "primary_k": float(k),
            "primary_k_units": primary_k_units(args or "distance"),
            "primary_openmm_k": float(primary_k_to_openmm_value(k, args or "distance")),
            "primary_openmm_k_units": primary_openmm_k_units(args or "distance"),
            "primary_harmonic_sigma": float(sigma),
            "legacy_primary_cv_column_names": True,
            "secondary_cv_center": "",
            "secondary_cv_k_kcal_mol": "",
            "secondary_cv_k_kj_mol": "",
        }
        if ss_centers is not None and ss_k_arr is not None and i < len(ss_centers) and i < len(ss_k_arr):
            row.update({
                "secondary_cv_center": float(ss_centers[i]),
                "secondary_cv_k_kcal_mol": float(ss_k_arr[i]),
                "secondary_cv_k_kj_mol": float(kcal_to_kj(ss_k_arr[i])),
            })
        rows.append(row)
    return rows

def write_window_assignment_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "window", "center_A", "k_kcal_mol_A2", "k_kj_mol_nm2",
        "harmonic_sigma_A", "spacing_to_previous_A", "spacing_to_next_A",
        "primary_cv", "primary_cv_label", "primary_cv_units", "primary_center", "primary_k",
        "primary_k_units", "primary_openmm_k", "primary_openmm_k_units",
        "primary_harmonic_sigma", "legacy_primary_cv_column_names",
        "secondary_cv_center", "secondary_cv_k_kcal_mol", "secondary_cv_k_kj_mol",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def write_explicit_2d_neighbor_graph_files(out_dir: Path, centers_a, secondary_cv_centers, args=None, prefix: str = "explicit_2d_neighbor_graph") -> dict:
    """Write the geometry neighbor graph used for explicit/sparse 2D exchange.

    This is diagnostic-only and does not change the exchange schedule.  It makes
    sparse explicit 2D runs auditable by recording the exact graph edges that
    neighbor exchange can draw from.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = build_explicit_2d_neighbor_edges(centers_a, secondary_cv_centers, args=args)
    csv_path = out_dir / f"{prefix}.csv"
    rows = []
    centers = np.asarray(centers_a, dtype=float)
    secondary = np.asarray(secondary_cv_centers, dtype=float) if secondary_cv_centers is not None else None
    for edge_id, edge in enumerate(edges):
        wi = int(edge.get("wi", -1))
        wj = int(edge.get("wj", -1))
        row = {
            "edge_id": int(edge_id),
            "window_i": wi,
            "window_j": wj,
            "edge": f"{min(wi, wj)}-{max(wi, wj)}",
            "edge_type": str(edge.get("edge_type", "explicit_2d_graph")),
            "normalized_distance": float(edge.get("normalized_distance", float("nan"))),
            "distance_i_A": float(centers[wi]) if 0 <= wi < len(centers) else "",
            "distance_j_A": float(centers[wj]) if 0 <= wj < len(centers) else "",
            "secondary_i": float(secondary[wi]) if secondary is not None and 0 <= wi < len(secondary) else "",
            "secondary_j": float(secondary[wj]) if secondary is not None and 0 <= wj < len(secondary) else "",
            "exchange_slot": int(edge_id % max(1, int(getattr(args, "explicit_2d_exchange_slots", 4) if args is not None else 4))),
        }
        rows.append(row)
    with csv_path.open("w", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["edge_id", "window_i", "window_j", "edge", "edge_type", "normalized_distance", "exchange_slot"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "mode": "explicit-2d-neighbor-graph",
        "description": "Geometry neighbor graph used for explicit/sparse 2D neighbor exchange; exchange intervals use disjoint slot subsets of these edges.",
        "n_windows": int(len(centers)),
        "n_edges": int(len(rows)),
        "edge_csv": str(csv_path),
        "exchange_slots": int(max(1, int(getattr(args, "explicit_2d_exchange_slots", 4) if args is not None else 4))),
        "edge_type_counts": {str(k): int(sum(1 for r in rows if str(r.get("edge_type")) == str(k))) for k in sorted({str(r.get("edge_type")) for r in rows})},
    }
    json_path = out_dir / f"{prefix}.json"
    write_json(json_path, _json_ready(payload))
    payload["edge_json"] = str(json_path)
    return payload

def write_explicit_window_analysis_files(
    out_dir: Path,
    centers_a,
    k_list,
    secondary_cv_centers=None,
    secondary_cv_k_kcal_list=None,
    secondary_cv_metadata: Optional[dict] = None,
    window_metadata: Optional[dict] = None,
    neighbor_graph_summary: Optional[dict] = None,
) -> dict:
    """Write sparse-safe explicit window metadata files for MBAR/reweighting."""
    out_dir = Path(out_dir)
    rows = explicit_window_analysis_rows(
        centers_a, k_list, secondary_cv_centers, secondary_cv_k_kcal_list,
        secondary_cv_metadata=secondary_cv_metadata,
        window_metadata=window_metadata,
    )
    csv_path = out_dir / "umbrella_explicit_windows.csv"
    json_path = out_dir / "umbrella_explicit_windows.json"
    fieldnames = [
        "window", "distance_center_A", "distance_center_nm", "distance_k_kcal_mol_A2", "distance_k_kj_mol_nm2",
        "primary_cv", "primary_cv_label", "primary_cv_units", "primary_center", "primary_k",
        "primary_k_units", "primary_openmm_k", "primary_openmm_k_units",
        "secondary_cv_center", "secondary_cv_k_kcal_mol", "secondary_cv_k_kj_mol", "secondary_cv_mode",
        "window_type", "lifecycle", "source_csv", "source_row", "explicit_2d", "rectangular_grid",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    explicit_2d = any(int(r.get("explicit_2d", 0) or 0) for r in rows)
    rectangular = all(int(r.get("rectangular_grid", 0) or 0) for r in rows) if rows and explicit_2d else False
    payload = {
        "schema_version": "2.0-sparse-2d-safe",
        "description": "Canonical window-major table for MBAR/reweighting. Sparse/non-rectangular 2D grids are represented explicitly; do not infer window identity from a rectangular product unless rectangular_grid is true.",
        "n_windows": int(len(rows)),
        "explicit_2d": bool(explicit_2d),
        "rectangular_grid": bool(rectangular),
        "window_csv": str(csv_path),
        "rows": rows,
        "neighbor_graph": neighbor_graph_summary or {},
        "bias_formula": {
            "primary_user_units": "0.5 * primary_k[window] * (primary_cv_value - primary_center[window])**2; cv_A/center_A are legacy aliases for the selected primary CV",
            "distance_user_units_legacy": "0.5 * distance_k_kcal_mol_A2[window] * (cv_A - distance_center_A[window])**2; valid literally only for primary_cv=distance",
            "secondary_user_units": "0.5 * secondary_cv_k_kcal_mol[window] * (secondary_cv - secondary_cv_center[window])**2 when secondary CV is enabled",
            "total_kcal_mol": "primary_umbrella_bias_kcal_mol + secondary_cv_bias_kcal_mol (distance_umbrella_bias_kcal_mol is a legacy alias)",
            "reduced_bias": "beta_1_over_kJ_mol * total_kcal_mol * 4.184",
        },
    }
    write_json(json_path, _json_ready(payload))
    return {
        "schema_version": payload["schema_version"],
        "n_windows": int(len(rows)),
        "explicit_2d": bool(explicit_2d),
        "rectangular_grid": bool(rectangular),
        "csv": str(csv_path),
        "json": str(json_path),
    }

def print_window_assignment_table(rows: list[dict], max_rows: int = 40) -> None:
    if not rows:
        return
    has_ss = any(str(r.get("secondary_cv_center", "")) not in {"", "nan"} for r in rows)
    first = rows[0]
    pcv = str(first.get("primary_cv", "distance") or "distance")
    plabel = str(first.get("primary_cv_label", "terminal distance") or "terminal distance")
    punits = str(first.get("primary_cv_units", "A") or "A")
    pk_units = str(first.get("primary_k_units", "kcal/mol/A^2") or "kcal/mol/A^2")
    pok_units = str(first.get("primary_openmm_k_units", "kJ/mol/nm^2") or "kJ/mol/nm^2")
    print(f"Umbrella window assignment: OpenMM bias = primary CV ({plabel})" + (" + secondary-CV umbrella" if has_ss else ""))
    print(f"  primary CV mode: {pcv}; center units: {punits}; k user units: {pk_units} | OpenMM units: {pok_units}")
    if has_ss:
        print("  secondary CV is dimensionless smooth phi/psi content; k units: kcal/mol/CV^2")
        print("  win     center      primary_k      openmm_k     sigma    ss0      ss_k")
    else:
        print("  win     center      primary_k      openmm_k     sigma    dprev    dnext")
    shown = rows[:max(1, int(max_rows))]
    for row in shown:
        openmm_k = row.get("primary_openmm_k", row.get("k_kj_mol_nm2", ""))
        openmm_k_s = f"{float(openmm_k):12.3f}" if str(openmm_k) not in {"", "nan"} else "           -"
        if has_ss:
            ss0 = row.get("secondary_cv_center", "")
            ssk = row.get("secondary_cv_k_kcal_mol", "")
            ss0_s = f"{float(ss0):7.3f}" if ss0 != "" else "      -"
            ssk_s = f"{float(ssk):8.3f}" if ssk != "" else "       -"
            print(
                f"  {int(row['window']):3d}"
                f" {float(row.get('primary_center', row['center_A'])):10.3f}"
                f" {float(row.get('primary_k', row['k_kcal_mol_A2'])):14.4f}"
                f" {openmm_k_s}"
                f" {float(row.get('primary_harmonic_sigma', row['harmonic_sigma_A'])):9.3f}"
                f" {ss0_s} {ssk_s}"
            )
        else:
            dprev = row["spacing_to_previous_A"]
            dnext = row["spacing_to_next_A"]
            dprev_s = f"{float(dprev):7.3f}" if dprev != "" else "      -"
            dnext_s = f"{float(dnext):7.3f}" if dnext != "" else "      -"
            print(
                f"  {int(row['window']):3d}"
                f" {float(row.get('primary_center', row['center_A'])):10.3f}"
                f" {float(row.get('primary_k', row['k_kcal_mol_A2'])):14.4f}"
                f" {openmm_k_s}"
                f" {float(row.get('primary_harmonic_sigma', row['harmonic_sigma_A'])):9.3f}"
                f" {dprev_s} {dnext_s}"
            )
    if len(rows) > len(shown):
        print(f"  ... {len(rows) - len(shown)} more windows omitted")

def all_integrator_globals(integrator) -> dict[str, float]:
    """Return all readable CustomIntegrator global variables by name."""
    out: dict[str, float] = {}
    try:
        n = integrator.getNumGlobalVariables()
    except Exception:
        return out
    for i in range(n):
        try:
            out[str(integrator.getGlobalVariableName(i))] = float(integrator.getGlobalVariable(i))
        except Exception:
            continue
    return out

def set_integrator_globals_from_dict(integrator, values: dict[str, float]) -> tuple[dict[str, float], dict[str, str]]:
    """Copy same-name global variables into an integrator.

    This is the practical OpenMM equivalent of the article-style GaREUS setup:
    run one GaMD calibration/equilibration, then give every umbrella replica the
    same GaMD thresholds/statistics before starting production.  Variables that
    do not exist in the destination integrator are reported but ignored.
    """
    copied: dict[str, float] = {}
    skipped: dict[str, str] = {}
    try:
        name_to_index = {str(integrator.getGlobalVariableName(i)): i for i in range(integrator.getNumGlobalVariables())}
    except Exception as exc:
        return copied, {"<all>": f"cannot inspect destination integrator globals: {exc}"}
    for name, raw_value in (values or {}).items():
        if name not in name_to_index:
            skipped[str(name)] = "not present in destination integrator"
            continue
        try:
            value = float(raw_value)
            integrator.setGlobalVariable(int(name_to_index[name]), value)
            copied[str(name)] = value
        except Exception as exc:
            skipped[str(name)] = str(exc)
    return copied, skipped


_SHARED_GAMD_SETUP_FILES = (
    "shared_gamd_setup_globals.json",
    "shared_gamd_setup_context.chk",
    "shared_gamd_setup_state.xml",
    "shared_gamd_setup_final.pdb",
    "shared_gamd_setup_state_write_warning.json",
)


def _path_arg(value: Any) -> Optional[Path]:
    """Return a non-empty Path from an argparse value, or None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def _copy_shared_gamd_setup_files(src_dir: Path, dst_dir: Path) -> dict[str, str]:
    """Copy auditable shared-GaMD setup artifacts when they exist."""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    copied: dict[str, str] = {}
    try:
        if src_dir.resolve() == dst_dir.resolve():
            return copied
    except Exception:
        pass
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in _SHARED_GAMD_SETUP_FILES:
        src = src_dir / name
        if not src.exists():
            continue
        dst = dst_dir / name
        try:
            shutil.copy2(src, dst)
            copied[name] = str(dst)
        except Exception as exc:
            copied[name] = f"copy failed: {exc}"
    return copied


def load_reusable_shared_gamd_setup(args, out_dir: Path) -> Optional[tuple[dict[str, float], dict[str, float], int, Optional[bytes], dict]]:
    """Load a previously calibrated shared GaMD setup for this worker run.

    This is used by adaptive production so all epoch/baseline/topup/final worker
    runs share the same GaMD thresholds/statistics.  The local worker directory
    still receives a copy of the setup JSON/checkpoint for provenance.
    """
    setup_dir = _path_arg(getattr(args, "shared_gamd_setup_dir", None))
    if setup_dir is None:
        setup_dir = _path_arg(getattr(args, "_global_shared_gamd_setup_dir", None))
    if setup_dir is None:
        return None
    globals_path = setup_dir / "shared_gamd_setup_globals.json"
    if not globals_path.exists():
        return None
    payload = read_json_file(globals_path, None)
    if not isinstance(payload, dict):
        return None
    all_globals = dict(payload.get("all_globals", {}) or {})
    interesting = dict(payload.get("interesting_globals", {}) or {})
    chk_path = setup_dir / "shared_gamd_setup_context.chk"
    checkpoint = None
    if chk_path.exists():
        try:
            checkpoint = chk_path.read_bytes()
        except Exception:
            checkpoint = None
    if not all_globals and checkpoint is None:
        return None
    calib_steps = int(payload.get("calibration_steps", 0) or 0)
    copied = _copy_shared_gamd_setup_files(setup_dir, out_dir)
    local_payload = dict(payload)
    reuse_note = {
        "reused_shared_gamd_setup": True,
        "source_shared_gamd_setup_dir": str(setup_dir),
        "source_shared_gamd_globals_json": str(globals_path),
        "source_shared_gamd_context_checkpoint": str(chk_path) if chk_path.exists() else "",
        "local_copied_files": copied,
        "description": "This worker reused a previously calibrated shared GaMD setup instead of recalibrating. Coordinates/window parameters are still worker-local; GaMD thresholds/statistics are campaign-global.",
    }
    local_payload.update(reuse_note)
    write_json(Path(out_dir) / "shared_gamd_setup_globals.json", _json_ready(local_payload))
    if checkpoint is not None and not (Path(out_dir) / "shared_gamd_setup_context.chk").exists():
        try:
            (Path(out_dir) / "shared_gamd_setup_context.chk").write_bytes(checkpoint)
        except Exception:
            pass
    return all_globals, interesting, calib_steps, checkpoint, reuse_note


def export_shared_gamd_setup_if_requested(args, out_dir: Path) -> dict[str, str]:
    """Export this worker's calibrated shared-GaMD setup for later workers."""
    export_dir = _path_arg(getattr(args, "shared_gamd_export_dir", None))
    if export_dir is None:
        export_dir = _path_arg(getattr(args, "_global_shared_gamd_export_dir", None))
    if export_dir is None:
        return {}
    copied = _copy_shared_gamd_setup_files(Path(out_dir), export_dir)
    manifest = {
        "schema_version": "gareus_shared_gamd_export_v1",
        "source_worker_dir": str(out_dir),
        "export_dir": str(export_dir),
        "copied_files": copied,
        "note": "Adaptive production uses this directory as the campaign-wide shared GaMD setup source for all later epoch/topup/final worker runs.",
    }
    write_json(export_dir / "shared_gamd_export_manifest.json", _json_ready(manifest))
    return copied

class AnalysisArrayWriter:
    """Chunk production samples into NumPy files for fast downstream analysis.

    The writer keeps only one chunk in memory.  Heavy N x K matrices are written to
    analysis_chunks/chunk_XXXXXX.npz, with a manifest in analysis_chunks_manifest.json.
    A consolidated legacy analysis_arrays.npz can still be written for compatibility
    when --analysis-write-consolidated-npz is enabled.
    """

    def __init__(self, out_dir: Path, n_windows: int, args=None):
        self.out_dir = Path(out_dir)
        self.n_windows = int(n_windows)
        self.args = args
        self.path = self.out_dir / "analysis_arrays.npz"
        self.meta_path = self.out_dir / "analysis_arrays_metadata.json"
        self.chunk_dir = self.out_dir / str(getattr(args, "analysis_chunk_dir", "analysis_chunks") if args is not None else "analysis_chunks")
        self.manifest_path = self.out_dir / "analysis_chunks_manifest.json"
        self.chunk_size = max(1, int(getattr(args, "analysis_chunk_size", 10000) if args is not None else 10000))
        self.write_chunks = bool(getattr(args, "write_analysis_chunks", True) if args is not None else True)
        self.write_consolidated = bool(getattr(args, "analysis_write_consolidated_npz", True) if args is not None else True)
        self.consolidated_max_elements = int(getattr(args, "analysis_consolidated_max_elements", 100_000_000) if args is not None else 100_000_000)
        self.consolidated_dtype = str(getattr(args, "analysis_array_dtype", "float64") if args is not None else "float64").lower()
        self.compressed = bool(getattr(args, "analysis_npz_compressed", False) if args is not None else False)
        self.chunk_index = 0
        self.n_samples = 0
        self.chunks: list[dict] = []
        self.beta_1_over_kj_mol: Optional[float] = None
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        self.step: list[int] = []
        self.replica: list[int] = []
        self.window: list[int] = []
        self.cv_A: list[float] = []
        self.secondary_cv: list[float] = []
        self.secondary_cv_center: list[float] = []
        self.secondary_cv_k_kcal_mol: list[float] = []
        self.center_A: list[float] = []
        self.k_kcal_mol_A2: list[float] = []
        self.potential_kj_mol: list[float] = []
        self.gamd_boost_total_kj_mol: list[float] = []
        self.distance_umbrella_bias_kcal_mol: list[float] = []
        self.secondary_cv_bias_kcal_mol: list[float] = []
        self.umbrella_bias_kcal_mol: list[float] = []
        self.umbrella_bias_kj_mol: list[float] = []
        self.umbrella_reduced_bias: list[float] = []
        self.distance_umbrella_bias_all_windows_kcal_mol: list[np.ndarray] = []
        self.secondary_cv_bias_all_windows_kcal_mol: list[np.ndarray] = []
        self.umbrella_bias_all_windows_kcal_mol: list[np.ndarray] = []

    def _coerce_window_vector(self, value, fill: float = np.nan) -> np.ndarray:
        if value is None:
            return np.full(self.n_windows, fill, dtype=np.float64)
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (self.n_windows,):
            return np.full(self.n_windows, fill, dtype=np.float64)
        return arr.astype(np.float64, copy=False)

    def append(
        self,
        row: dict,
        reduced_bias_all_windows,
        distance_bias_all_windows_kcal=None,
        secondary_bias_all_windows_kcal=None,
        umbrella_bias_all_windows_kcal=None,
        umbrella_bias_all_windows_kj=None,
    ) -> None:
        vec = np.asarray(reduced_bias_all_windows, dtype=np.float64)
        if vec.shape != (self.n_windows,):
            return
        dist_vec = self._coerce_window_vector(distance_bias_all_windows_kcal)
        sec_vec = self._coerce_window_vector(secondary_bias_all_windows_kcal, fill=0.0)
        total_kcal_vec = self._coerce_window_vector(umbrella_bias_all_windows_kcal)
        self.step.append(int(row.get("step", 0)))
        self.replica.append(int(row.get("replica", 0)))
        self.window.append(int(row.get("window", 0)))
        self.cv_A.append(float(row.get("cv_A", np.nan)))
        self.secondary_cv.append(float(row.get("secondary_cv", np.nan)) if row.get("secondary_cv", "") != "" else float("nan"))
        self.secondary_cv_center.append(float(row.get("secondary_cv_center", np.nan)) if row.get("secondary_cv_center", "") != "" else float("nan"))
        self.secondary_cv_k_kcal_mol.append(float(row.get("secondary_cv_k_kcal_mol", np.nan)) if row.get("secondary_cv_k_kcal_mol", "") != "" else float("nan"))
        self.center_A.append(float(row.get("center_A", np.nan)))
        self.k_kcal_mol_A2.append(float(row.get("k_kcal_mol_A2", np.nan)))
        self.potential_kj_mol.append(float(row.get("potential_kj_mol", np.nan)))
        try:
            boost = float(row.get("gamd_boost_total_kj_mol", np.nan))
        except Exception:
            boost = float("nan")
        self.gamd_boost_total_kj_mol.append(boost)
        self.distance_umbrella_bias_kcal_mol.append(float(row.get("distance_umbrella_bias_kcal_mol", np.nan)))
        self.secondary_cv_bias_kcal_mol.append(float(row.get("secondary_cv_bias_kcal_mol", np.nan)))
        self.umbrella_bias_kcal_mol.append(float(row.get("umbrella_bias_kcal_mol", np.nan)))
        self.umbrella_bias_kj_mol.append(float(row.get("umbrella_bias_kj_mol", np.nan)))
        self.umbrella_reduced_bias.append(float(row.get("umbrella_reduced_bias", np.nan)))
        if self.beta_1_over_kj_mol is None:
            try:
                beta_row = float(row.get("beta_1_over_kJ_mol", np.nan))
                if math.isfinite(beta_row):
                    self.beta_1_over_kj_mol = beta_row
            except Exception:
                pass
        self.distance_umbrella_bias_all_windows_kcal_mol.append(dist_vec)
        self.secondary_cv_bias_all_windows_kcal_mol.append(sec_vec)
        self.umbrella_bias_all_windows_kcal_mol.append(total_kcal_vec)
        self.n_samples += 1
        if self.write_chunks and len(self.step) >= self.chunk_size:
            self.flush_chunk()

    def _arrays_from_buffers(self) -> dict[str, np.ndarray]:
        n = len(self.step)
        if n <= 0:
            return {}
        dist_mat = np.vstack(self.distance_umbrella_bias_all_windows_kcal_mol).astype(np.float64, copy=False)
        sec_mat = np.vstack(self.secondary_cv_bias_all_windows_kcal_mol).astype(np.float64, copy=False)
        total_kcal_mat = np.vstack(self.umbrella_bias_all_windows_kcal_mol).astype(np.float64, copy=False)
        total_kj_mat = (4.184 * total_kcal_mat).astype(np.float64, copy=False)
        beta_val = float(self.beta_1_over_kj_mol) if self.beta_1_over_kj_mol is not None else float("nan")
        red_mat = (beta_val * total_kj_mat).astype(np.float64, copy=False)
        return {
            "step": np.asarray(self.step, dtype=np.int64),
            "replica": np.asarray(self.replica, dtype=np.int32),
            "window": np.asarray(self.window, dtype=np.int32),
            "cv_A": np.asarray(self.cv_A, dtype=np.float64),
            "secondary_cv": np.asarray(self.secondary_cv, dtype=np.float64),
            "secondary_cv_center": np.asarray(self.secondary_cv_center, dtype=np.float64),
            "secondary_cv_k_kcal_mol": np.asarray(self.secondary_cv_k_kcal_mol, dtype=np.float64),
            "center_A": np.asarray(self.center_A, dtype=np.float64),
            "k_kcal_mol_A2": np.asarray(self.k_kcal_mol_A2, dtype=np.float64),
            "potential_kj_mol": np.asarray(self.potential_kj_mol, dtype=np.float64),
            "gamd_boost_total_kj_mol": np.asarray(self.gamd_boost_total_kj_mol, dtype=np.float64),
            "distance_umbrella_bias_kcal_mol": np.asarray(self.distance_umbrella_bias_kcal_mol, dtype=np.float64),
            "secondary_cv_bias_kcal_mol": np.asarray(self.secondary_cv_bias_kcal_mol, dtype=np.float64),
            "umbrella_bias_kcal_mol": np.asarray(self.umbrella_bias_kcal_mol, dtype=np.float64),
            "umbrella_bias_kj_mol": np.asarray(self.umbrella_bias_kj_mol, dtype=np.float64),
            "umbrella_reduced_bias": np.asarray(self.umbrella_reduced_bias, dtype=np.float64),
            "distance_umbrella_bias_kcal_mol_nk": dist_mat,
            "primary_umbrella_bias_kcal_mol_nk": dist_mat,
            "secondary_cv_bias_kcal_mol_nk": sec_mat,
            "umbrella_bias_kcal_mol_nk": total_kcal_mat,
            "umbrella_bias_kj_mol_nk": total_kj_mat,
            "umbrella_reduced_bias_kn": red_mat.T,
            "umbrella_reduced_bias_nk": red_mat,
        }

    def _save_npz(self, path: Path, arrays: dict[str, np.ndarray]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        saver = np.savez_compressed if self.compressed else np.savez
        saver(path, **arrays)

    def flush_chunk(self) -> Optional[Path]:
        arrays = self._arrays_from_buffers()
        if not arrays:
            return None
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = self.chunk_dir / f"chunk_{self.chunk_index:06d}.npz"
        self._save_npz(chunk_path, arrays)
        n = int(arrays["step"].size)
        self.chunks.append({
            "chunk_id": int(self.chunk_index),
            "path": str(chunk_path),
            "n_samples": n,
            "n_windows": int(self.n_windows),
            "first_step": int(arrays["step"][0]) if n else None,
            "last_step": int(arrays["step"][-1]) if n else None,
            "compressed": bool(self.compressed),
        })
        self.chunk_index += 1
        self._reset_buffers()
        self.write_manifest()
        return chunk_path

    def write_manifest(self) -> Path:
        payload = {
            "schema_version": "analysis-chunks-v1",
            "n_samples": int(self.n_samples),
            "n_windows": int(self.n_windows),
            "chunk_size_target": int(self.chunk_size),
            "chunks": self.chunks,
            "consolidated_npz": str(self.path) if self.path.exists() else "",
            "shape_convention": "sample-major arrays; *_nk arrays have shape [n_samples, n_windows]",
        }
        write_json(self.manifest_path, _json_ready(payload))
        return self.manifest_path

    def _estimated_consolidated_matrix_elements(self) -> int:
        # Conservative estimate for dense N x K matrix arrays.  Vector arrays are
        # cheap by comparison; these matrices dominate memory and file size.
        return int(max(0, self.n_samples) * max(0, self.n_windows))

    def _maybe_cast_consolidated_array(self, arr: np.ndarray) -> np.ndarray:
        if self.consolidated_dtype == "float32" and np.issubdtype(arr.dtype, np.floating):
            return arr.astype(np.float32, copy=False)
        return arr

    def _concatenate_chunk_arrays(self, key: str, vals: list[np.ndarray]) -> np.ndarray:
        """Concatenate one logical analysis array across chunk files.

        Most arrays are sample-major, so appending chunks means axis=0.  The
        legacy PyMBAR-compatible ``umbrella_reduced_bias_kn`` array is the one
        intentional exception: it is window/state-major [n_windows, n_samples],
        so chunks must be appended along axis=1.  Without this special case, a
        final short chunk makes consolidation crash with mismatched dimension 1.
        """
        clean_vals = [np.asarray(v) for v in vals if v is not None]
        if not clean_vals:
            return np.asarray([])

        if key == "umbrella_reduced_bias_kn":
            fixed_vals: list[np.ndarray] = []
            for arr in clean_vals:
                if arr.ndim != 2:
                    raise ValueError(
                        f"Cannot consolidate {key}: expected 2D [n_windows, n_samples] chunks, "
                        f"got shape {arr.shape}."
                    )
                if arr.shape[0] == self.n_windows:
                    fixed_vals.append(arr)
                elif arr.shape[1] == self.n_windows:
                    # Be tolerant of accidental sample-major legacy chunks.
                    fixed_vals.append(arr.T)
                else:
                    raise ValueError(
                        f"Cannot consolidate {key}: chunk shape {arr.shape} does not match "
                        f"n_windows={self.n_windows}."
                    )
            return np.concatenate(fixed_vals, axis=1)

        try:
            return np.concatenate(clean_vals, axis=0)
        except ValueError as exc:
            shapes = ", ".join(str(tuple(v.shape)) for v in clean_vals[:12])
            if len(clean_vals) > 12:
                shapes += ", ..."
            raise ValueError(
                f"Cannot consolidate analysis array {key!r} across chunks; "
                f"chunk shapes are: {shapes}. Sample-major arrays should vary only "
                f"along axis 0; legacy *_kn arrays must be handled explicitly."
            ) from exc

    def _write_consolidated_from_chunks(self) -> Optional[Path]:
        if not self.chunks:
            return None
        est_elements = self._estimated_consolidated_matrix_elements()
        if self.consolidated_max_elements > 0 and est_elements > self.consolidated_max_elements:
            # Avoid leaving a stale legacy consolidated file around when a large
            # rerun intentionally uses chunked arrays only.
            try:
                if self.path.exists():
                    self.path.unlink()
            except Exception:
                pass
            return None
        loaded: dict[str, list[np.ndarray]] = {}
        for chunk in self.chunks:
            with np.load(chunk["path"], allow_pickle=False) as data:
                for key in data.files:
                    loaded.setdefault(key, []).append(np.asarray(data[key]))
        arrays = {
            key: self._maybe_cast_consolidated_array(self._concatenate_chunk_arrays(key, vals))
            for key, vals in loaded.items()
            if vals
        }
        if not arrays:
            return None
        self._save_npz(self.path, arrays)
        return self.path

    def write(self) -> Optional[Path]:
        if len(self.step) > 0:
            self.flush_chunk()
        if self.n_samples <= 0:
            return None
        consolidated_path = None
        if self.write_consolidated:
            consolidated_path = self._write_consolidated_from_chunks()
        manifest_path = self.write_manifest()
        write_json(self.meta_path, {
            "analysis_arrays_npz": str(consolidated_path) if consolidated_path is not None else "",
            "analysis_chunks_manifest_json": str(manifest_path),
            "analysis_chunk_dir": str(self.chunk_dir),
            "n_samples": int(self.n_samples),
            "n_windows": int(self.n_windows),
            "chunked": True,
            "consolidated_npz_written": bool(consolidated_path is not None),
            "consolidated_npz_skipped_by_size_guard": bool(self.write_consolidated and consolidated_path is None and self._estimated_consolidated_matrix_elements() > self.consolidated_max_elements > 0),
            "analysis_consolidated_max_elements": int(self.consolidated_max_elements),
            "analysis_array_dtype": str(self.consolidated_dtype),
            "shape_convention": "sample-major arrays; umbrella_reduced_bias_nk has shape [n_samples, n_windows]",
            "pymbar_note": "Prefer analysis_chunks_manifest.json for large runs; use consolidated analysis_arrays.npz only when written.",
            "matrix_arrays": {
                "distance_umbrella_bias_kcal_mol_nk": "legacy alias for primary-only umbrella U_k(x_n), kcal/mol",
            "primary_umbrella_bias_kcal_mol_nk": "primary-only umbrella U_k(x_n), kcal/mol; same values as distance_umbrella_bias_kcal_mol_nk",
                "secondary_cv_bias_kcal_mol_nk": "secondary-CV umbrella U_k(x_n), kcal/mol; zeros for 1D runs",
                "umbrella_bias_kcal_mol_nk": "total umbrella U_k(x_n), kcal/mol",
                "umbrella_bias_kj_mol_nk": "total umbrella U_k(x_n), kJ/mol",
                "umbrella_reduced_bias_nk": "beta * total umbrella U_k(x_n), dimensionless",
                "umbrella_reduced_bias_kn": "legacy PyMBAR transpose of umbrella_reduced_bias_nk with shape [n_windows, n_samples]",
            },
            "csv_companion": str(self.out_dir / "samples.csv"),
        })
        return consolidated_path or manifest_path

def write_standardized_output_layout(out_dir: Path) -> dict:
    """Write a stable, non-destructive organized view of important outputs.

    The workflow historically writes most files at the run-root for backward
    compatibility.  This helper creates category directories and symlink mirrors
    for humans/tools without moving or renaming canonical outputs.
    """
    out_dir = Path(out_dir)
    categories = {
        "config": [
            "effective_config.yaml", "effective_config.json", "command_line.txt", "resume_command.sh",
        ],
        "setup": [
            "00_built_peptide.pdb", "01_solvated_start.pdb", "02_minimized.pdb", "02_nvt_warm.pdb",
            "02_npt_equil.csv", "03_npt_equilibrated.pdb", "03_npt_equilibrated_state.xml",
            "shared_gamd_setup_globals.json", "shared_gamd_setup_state.xml", "shared_gamd_setup_final.pdb",
            "us_starting_structures/us_pulling_starting_structures.csv",
            "us_starting_structures/us_starting_structure_quality.json",
            "us_starting_structures/graft_report.json",
        ],
        "adaptive_feedback": [
            "adaptive_feedback_driver_summary.json", "adaptive_feedback_summary.md", "adaptive_feedback_summary.json",
            "adaptive_feedback_memory.json",
        ],
        "final_production": [
            "samples.csv", "exchanges.csv", "distances.csv", "distances.jsonl", "progress.jsonl",
            "production_probe_report.json", "replica_shared_gamd_copy_report.json",
        ],
        "analysis": [
            "analysis_arrays.npz", "analysis_arrays_metadata.json", "umbrella_pymbar_metadata.json",
            "umbrella_windows.csv", "umbrella_explicit_windows.csv", "umbrella_explicit_windows.json",
            "analysis_metadata_validation.json", "legacy_1d_mbar_validation.json",
            "exchange_tuning_report.md", "exchange_tuning_report.json",
            "final_report.md", "final_report.json",
            "explicit_2d_neighbor_graph.csv", "explicit_2d_neighbor_graph.json",
        ],
        "checkpoints": [
            "checkpoints/production_checkpoint_manifest.json",
        ],
        "logs": [
            "progress.jsonl", "distances.jsonl", "02_npt_equil.csv",
        ],
    }
    created_dirs = []
    mirrors = []
    for cat, names in categories.items():
        cat_dir = out_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        created_dirs.append(str(cat_dir))
        for name in names:
            src = out_dir / name
            mirror_name = Path(name).name
            dst = cat_dir / mirror_name
            # Avoid self-links when canonical output already lives under the same category.
            try:
                if src.resolve() == dst.resolve():
                    continue
            except Exception:
                pass
            row = _mkdir_and_link_or_note(src, dst)
            row["category"] = cat
            mirrors.append(row)
    # Add adaptive round directories to the manifest without duplicating their internals.
    adaptive_rounds = sorted(str(p) for p in out_dir.glob("adaptive_feedback_round_*"))
    final_dir = out_dir / "final_production"
    payload = {
        "mode": "non_destructive_standardized_layout",
        "description": "Category directories are a symlink/index layer only; canonical output paths remain unchanged for backward compatibility.",
        "out_dir": str(out_dir),
        "created_category_dirs": created_dirs,
        "adaptive_round_dirs": adaptive_rounds,
        "final_production_dir": str(final_dir) if final_dir.exists() else "",
        "mirrors": mirrors,
    }
    write_json(out_dir / "output_layout.json", _json_ready(payload))
    lines = ["# Output layout", "", "Canonical files remain at their original paths. Category folders are a non-destructive mirror/index.", ""]
    for cat in categories:
        present = [m for m in mirrors if m.get("category") == cat and m.get("ok")]
        lines.append(f"## {cat}")
        if present:
            for m in present:
                lines.append(f"- `{_safe_relative_path(Path(m['mirror']), out_dir)}` -> `{_safe_relative_path(Path(m['source']), out_dir)}`")
        else:
            lines.append("- no indexed files present yet")
        lines.append("")
    if adaptive_rounds:
        lines.append("## adaptive_feedback rounds")
        for r in adaptive_rounds:
            lines.append(f"- `{_safe_relative_path(Path(r), out_dir)}`")
        lines.append("")
    (out_dir / "output_layout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload

def write_exchange_tuning_report(out_dir: Path, args, exchange_stats: dict, centers_a=None, secondary_cv_centers=None, secondary_cv_metadata: Optional[dict] = None) -> dict:
    """Write human-readable exchange diagnostics and practical tuning suggestions."""
    out_dir = Path(out_dir)
    exchange_stats = exchange_stats or {}
    attempts = int(exchange_stats.get("attempts", 0) or 0)
    accepted = int(exchange_stats.get("accepted", 0) or 0)
    total_acceptance = accepted / max(1, attempts)
    pair_rows = _exchange_pair_rows(exchange_stats)
    attempted_pairs = [r for r in pair_rows if int(r.get("attempts", 0) or 0) > 0]
    weak_pairs = [r for r in attempted_pairs if int(r.get("attempts", 0) or 0) >= 5 and float(r.get("acceptance_fraction", 0.0)) < 0.08]
    high_pairs = [r for r in attempted_pairs if int(r.get("attempts", 0) or 0) >= 5 and float(r.get("acceptance_fraction", 0.0)) > 0.60]
    no_attempt_pairs = [r for r in pair_rows if int(r.get("attempts", 0) or 0) <= 0]
    mode = str(exchange_stats.get("mode", getattr(args, "exchange_mode", "neighbor")))
    gibbs_choices = int(exchange_stats.get("gibbs_choices", 0) or 0)
    gibbs_moves = int(exchange_stats.get("gibbs_moves", 0) or 0)
    gibbs_stays = int(exchange_stats.get("gibbs_stays", 0) or 0)
    gibbs_move_fraction = gibbs_moves / max(1, gibbs_choices) if gibbs_choices else float("nan")
    jump_rows = []
    for key, st in sorted((exchange_stats.get("jump_bins", {}) or {}).items()):
        att = int(st.get("attempts", 0) or 0)
        acc = int(st.get("accepted", 0) or 0)
        jump_rows.append({"jump_bin": str(key), "attempts": att, "accepted": acc, "acceptance_fraction": acc / max(1, att)})
    recommendations = []
    if attempts <= 0 and gibbs_choices <= 0:
        recommendations.append("No exchange attempts recorded yet; check --exchange-interval and whether production reached an exchange boundary.")
    elif total_acceptance < 0.08 and attempts > 0:
        recommendations.append("Total Metropolis exchange acceptance is very low; add/refine windows near weak pairs, reduce overly stiff k, or increase overlap targets.")
    elif total_acceptance > 0.60 and attempts > 0:
        recommendations.append("Total exchange acceptance is high; the ladder may be over-resolved, so adaptive pruning or more aggressive spacing may reduce replica count.")
    else:
        recommendations.append("Overall exchange acceptance is in a workable range; inspect weak local pairs before changing global settings.")
    if weak_pairs:
        labels = ", ".join(r["pair"] for r in weak_pairs[:8])
        recommendations.append(f"Weak local exchange pairs: {labels}. Prefer local sparse patches or local k/spacing adjustments over global changes.")
    if high_pairs and len(high_pairs) >= max(2, len(attempted_pairs) // 3):
        recommendations.append("Many adjacent/graph edges have very high acceptance; consider allowing adaptive pruning or a more aggressive window setting.")
    if math.isfinite(gibbs_move_fraction):
        if gibbs_move_fraction < 0.10:
            recommendations.append("Gibbs-walk mostly stays put; the heat-bath choices are too local/expensive. Improve window overlap or consider neighbor exchange until the graph is healthier.")
        elif gibbs_move_fraction > 0.70:
            recommendations.append("Gibbs-walk move fraction is high; verify it is not jumping between poorly overlapping states by checking PMF/MBAR overlap diagnostics.")
    interval = int(getattr(args, "exchange_interval", 0) or 0)
    if interval > 0 and interval < int(getattr(args, "distance_output_interval", interval) or interval):
        recommendations.append("Exchange interval is shorter than distance-output interval; this is fine, but diagnostics may under-sample exchange-state changes.")
    if secondary_cv_metadata and secondary_cv_metadata.get("explicit_2d_windows") and not (exchange_stats.get("neighbor_graph") or any(r.get("neighbor_source") for r in pair_rows)):
        recommendations.append("Explicit sparse 2D windows are present but graph metadata is missing from exchange stats; verify neighbor-graph exchange path was active.")
    status = "ok"
    if weak_pairs or (attempts > 0 and total_acceptance < 0.08):
        status = "warning"
    if attempts <= 0 and gibbs_choices <= 0:
        status = "warning"
    payload = {
        "status": status,
        "exchange_mode": mode,
        "attempts": attempts,
        "accepted": accepted,
        "acceptance_fraction": float(total_acceptance),
        "gibbs_choices": gibbs_choices,
        "gibbs_moves": gibbs_moves,
        "gibbs_stays": gibbs_stays,
        "gibbs_move_fraction": float(gibbs_move_fraction) if math.isfinite(gibbs_move_fraction) else None,
        "pair_rows": pair_rows,
        "weak_pairs": weak_pairs,
        "high_acceptance_pairs": high_pairs,
        "no_attempt_pairs": no_attempt_pairs,
        "jump_bins": jump_rows,
        "neighbor_graph": exchange_stats.get("neighbor_graph", {}),
        "recommendations": recommendations,
    }
    write_json(out_dir / "exchange_tuning_report.json", _json_ready(payload))
    lines = ["# Exchange tuning report", "", f"Mode: `{mode}`", f"Attempts: **{attempts}**, accepted: **{accepted}**, acceptance: **{100.0*total_acceptance:.1f}%**", ""]
    if math.isfinite(gibbs_move_fraction):
        lines.append(f"Gibbs choices/moves/stays: **{gibbs_choices}/{gibbs_moves}/{gibbs_stays}**; move fraction **{100.0*gibbs_move_fraction:.1f}%**")
        lines.append("")
    lines.append("## Recommendations")
    for rec in recommendations:
        lines.append(f"- {rec}")
    lines.append("")
    if weak_pairs:
        lines.append("## Weak pairs")
        lines.append("| pair | attempts | accepted | acceptance | edge type |")
        lines.append("|---|---:|---:|---:|---|")
        for r in weak_pairs[:25]:
            lines.append(f"| {r['pair']} | {int(r['attempts'])} | {int(r['accepted'])} | {100.0*float(r['acceptance_fraction']):.1f}% | {r.get('edge_type','')} |")
        lines.append("")
    if attempted_pairs:
        lines.append("## All attempted pairs")
        lines.append("| pair | attempts | accepted | acceptance | edge type |")
        lines.append("|---|---:|---:|---:|---|")
        for r in attempted_pairs[:80]:
            lines.append(f"| {r['pair']} | {int(r['attempts'])} | {int(r['accepted'])} | {100.0*float(r['acceptance_fraction']):.1f}% | {r.get('edge_type','')} |")
        if len(attempted_pairs) > 80:
            lines.append(f"| ... | ... | ... | ... | {len(attempted_pairs)-80} more pairs omitted |")
    (out_dir / "exchange_tuning_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload

def write_final_run_report(out_dir: Path, args, centers_a, k_list, exchange_stats: dict) -> dict:
    """Write final_report.md plus machine-readable validation diagnostics."""
    out_dir = Path(out_dir)
    target_overlap = float(getattr(args, "adaptive_feedback_target_overlap", 0.30) or 0.30)
    aggr = _adaptive_window_aggressiveness_settings(args)
    effective_target_overlap = max(0.04, min(0.95, target_overlap * float(aggr.get("target_overlap_factor", 1.0))))
    mbar = validate_analysis_metadata_readiness(out_dir, target_overlap=effective_target_overlap)
    # Keep the historical validator output too for simple 1D sanity checks, but
    # treat the sparse-aware readiness report above as the source of truth.
    legacy_mbar = validate_us_mbar_inputs(out_dir, float(getattr(args, "temperature_k", 300.0) or 300.0), target_overlap=target_overlap)
    gamd = compute_gamd_reweighting_diagnostics(out_dir, float(getattr(args, "temperature_k", 300.0) or 300.0))
    pull_quality_path = out_dir / "us_starting_structures" / "us_starting_structure_quality.json"
    pull_quality = None
    if pull_quality_path.exists():
        try:
            pull_quality = json.loads(pull_quality_path.read_text(encoding="utf-8"))
        except Exception:
            pull_quality = {"status": "error", "path": str(pull_quality_path)}
    graft_report_path = out_dir / "us_starting_structures" / "graft_report.json"
    graft_report = None
    if graft_report_path.exists():
        try:
            graft_report = json.loads(graft_report_path.read_text(encoding="utf-8"))
        except Exception:
            graft_report = None
    windows_rows = window_assignment_rows(np.asarray(centers_a, dtype=float), list(k_list), float(getattr(args, "temperature_k", 300.0) or 300.0), args=args)
    report_payload = {
        "status": "ok" if mbar.get("status") != "error" else "error",
        "sequence": str(getattr(args, "seq", "")),
        "out_dir": str(out_dir),
        "n_windows": int(len(centers_a)),
        "production_steps": int(getattr(args, "gamd_production_steps", 0) or 0),
        "exchange_mode": str(getattr(args, "exchange_mode", "neighbor")),
        "primary_cv": primary_cv_mode(args),
        "primary_cv_label": primary_cv_label(args),
        "primary_cv_units": primary_cv_units(args),
        "primary_k_units": primary_k_units(args),
        "target_neighbor_overlap": float(target_overlap),
        "effective_target_neighbor_overlap": float(effective_target_overlap),
        "adaptive_window_aggressiveness": str(aggr.get("mode", "balanced")),
        "aggressiveness_settings": aggr,
        "umbrella_windows": windows_rows,
        "mbar_input_validation": mbar,
        "legacy_1d_mbar_validation": legacy_mbar,
        "gamd_reweighting_diagnostics": gamd,
        "us_starting_structure_quality": pull_quality,
        "graft_report": graft_report,
        "exchange_stats": exchange_stats,
    }
    write_json(out_dir / "analysis_metadata_validation.json", _json_ready(mbar))
    write_json(out_dir / "legacy_1d_mbar_validation.json", _json_ready(legacy_mbar))
    write_json(out_dir / "final_report.json", _json_ready(report_payload))

    def status_word(d):
        return str((d or {}).get("status", "n/a")).upper()

    lines = []
    lines.append(f"# GaREUS/GaMD final run report")
    lines.append("")
    lines.append(f"Sequence: `{getattr(args, 'seq', '')}`")
    lines.append(f"Output directory: `{out_dir}`")
    lines.append(f"Windows/replicas: **{len(centers_a)}**")
    lines.append(f"Production steps: **{int(getattr(args, 'gamd_production_steps', 0) or 0)}**")
    lines.append(f"Exchange mode: `{getattr(args, 'exchange_mode', 'neighbor')}`")
    lines.append(f"Primary CV: **{primary_cv_label(args)}** ({primary_cv_units(args)})")
    lines.append(f"Primary umbrella k units: `{primary_k_units(args)}`")
    lines.append("")
    lines.append("## What to use for analysis")
    lines.append("")
    lines.append("Use this directory for final MBAR/PMF analysis. Adaptive pilot directories are diagnostic only.")
    lines.append("")
    lines.append("- `analysis_arrays.npz` — binary production-only arrays")
    lines.append("- `samples.csv` — human-readable production-only samples")
    lines.append("- `umbrella_windows.csv` — fixed final centers and k values")
    lines.append("- `umbrella_pymbar_metadata.json` — sparse-safe US/MBAR metadata")
    lines.append("- `umbrella_explicit_windows.csv` — canonical window-major table; safe for sparse 2D grids")
    lines.append("- `analysis_metadata_validation.json` — sparse-aware metadata/readiness validation")
    lines.append("- `exchange_tuning_report.md` — practical exchange acceptance/Gibbs/graph tuning recommendations")
    lines.append("- `output_layout.md` — non-destructive organized index of the run directory")
    lines.append("- `explicit_2d_neighbor_graph.csv` — graph edges for sparse explicit 2D neighbor exchange, when applicable")
    lines.append("")
    lines.append("## MBAR/US input validation")
    lines.append("")
    lines.append(f"Status: **{status_word(mbar)}**")
    lines.append(f"Samples: **{int(mbar.get('n_samples', 0) or 0)}**")
    if mbar.get("sample_counts_by_window"):
        counts = ", ".join(str(x) for x in mbar.get("sample_counts_by_window", []))
        lines.append(f"Samples/window: `{counts}`")
    if mbar.get("sparse_2d"):
        lines.append("Sparse 2D metadata: **enabled**; overlap diagnostics use graph edges rather than flattened window order.")
    if mbar.get("neighbor_overlaps"):
        def _edge_label(r):
            if "window_i" in r:
                return f"{int(r['window_i'])}-{int(r['window_j'])}:{float(r['overlap']):.2f}"
            return f"{int(r['left_window'])}-{int(r['right_window'])}:{float(r['overlap']):.2f}"
        ovtxt = ", ".join(_edge_label(r) for r in mbar.get("neighbor_overlaps", []) if math.isfinite(float(r.get("overlap", float("nan")))))
        lines.append(f"Neighbor CV overlaps: `{ovtxt}`")
    for msg in mbar.get("errors", []):
        lines.append(f"- ERROR: {msg}")
    for msg in mbar.get("warnings", []):
        lines.append(f"- WARNING: {msg}")
    lines.append("")
    lines.append("## GaMD boost/reweighting diagnostics")
    lines.append("")
    lines.append(f"Status: **{status_word(gamd)}**")
    if gamd.get("n_finite", 0):
        lines.append(f"Finite boost samples: **{int(gamd.get('n_finite', 0))} / {int(gamd.get('n_total', 0))}**")
        lines.append(f"Boost mean ± sd: **{float(gamd.get('boost_mean_kcal_mol', float('nan'))):.3f} ± {float(gamd.get('boost_sd_kcal_mol', float('nan'))):.3f} kcal/mol**")
        an = gamd.get("anharmonicity", {}) or {}
        if math.isfinite(float(an.get("score", float("nan")))):
            lines.append(f"Anharmonicity score: **{float(an.get('score')):.3f}**; skew `{float(an.get('skew', float('nan'))):.3f}`, excess kurtosis `{float(an.get('excess_kurtosis', float('nan'))):.3f}`")
        if math.isfinite(float(gamd.get("reweighting_ess_fraction", float("nan")))):
            lines.append(f"Approx. reweighting ESS: **{float(gamd.get('reweighting_ess', float('nan'))):.1f}** ({float(gamd.get('reweighting_ess_fraction', float('nan'))):.3f} fraction)")
    for msg in gamd.get("warnings", []):
        lines.append(f"- WARNING: {msg}")
    lines.append("")
    lines.append("## US starting-structure quality")
    lines.append("")
    if pull_quality:
        lines.append(f"OK/warn/bad: **{int(pull_quality.get('n_ok', 0))}/{int(pull_quality.get('n_warn', 0))}/{int(pull_quality.get('n_bad', 0))}**")
        _pq_delta = pull_quality.get('max_abs_primary_delta', pull_quality.get('max_abs_delta_A', float('nan')))
        lines.append(f"Max |start-center|: **{format_primary_delta_value(_pq_delta, args, precision=3)}**")
        lines.append(f"Max starting production umbrella bias: **{float(pull_quality.get('max_production_umbrella_bias_kcal_mol', float('nan'))):.3f} kcal/mol**")
        if int(pull_quality.get("n_bad", 0) or 0) or int(pull_quality.get("n_warn", 0) or 0):
            lines.append(f"See `{pull_quality_path}` for per-window warnings.")
    else:
        lines.append("No pulled-start quality file found, probably because `--us-starting-structure-mode` was not `pull`.")
    if graft_report:
        n_seeded = sum(1 for g in graft_report if not g.get("fallback", True))
        n_fb = len(graft_report) - n_seeded
        deltas = [abs(float(g["conformer_cv_A"]) - float(g["window_center_A"]))
                  for g in graft_report if not g.get("fallback", True)
                  and g.get("conformer_cv_A") is not None and g.get("window_center_A") is not None]
        lines.append("")
        lines.append("## GENPEPT conformer seeding")
        lines.append("")
        lines.append(f"Windows seeded from GENPEPT: **{n_seeded}/{len(graft_report)}**")
        lines.append(f"Windows fell back to NPT-pull: **{n_fb}**")
        if deltas:
            lines.append(f"Mean |conformer CV − window center|: **{format_primary_delta_value(float(np.mean(deltas)), args, precision=3)}**")
            lines.append(f"Max  |conformer CV − window center|: **{format_primary_delta_value(float(np.max(deltas)), args, precision=3)}**")
        lines.append(f"See `{graft_report_path}` for per-window details.")
    lines.append("")
    lines.append("## Final umbrella windows")
    lines.append("")
    lines.append(f"| window | {primary_cv_label(args)} | k {primary_k_units(args)} | harmonic sigma ({primary_cv_units(args)}) |")
    lines.append("|---:|---:|---:|---:|")
    for r in windows_rows:
        lines.append(
            f"| {int(r['window'])} | "
            f"{float(r.get('primary_center', r['center_A'])):.4f} | "
            f"{float(r.get('primary_k', r['k_kcal_mol_A2'])):.4f} | "
            f"{float(r.get('primary_harmonic_sigma', r['harmonic_sigma_A'])):.4f} |"
        )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    if mbar.get("status") == "error":
        lines.append("Do **not** trust final PMF/MBAR yet; fix the validation errors above.")
    elif gamd.get("status") == "warning" or mbar.get("status") == "warning":
        lines.append("Run analysis, but treat the PMF cautiously and inspect warnings/overlap/ESS first.")
    else:
        lines.append("Inputs look structurally usable for downstream MBAR/PMF analysis.")
    (out_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_payload

def compare_gamd_global_sets(reference: dict[str, float], current: dict[str, float], rtol: float = 1.0e-10, atol: float = 1.0e-10) -> dict:
    """Compare CustomIntegrator globals after copying a shared GaMD setup."""
    ref = reference or {}
    cur = current or {}
    missing = sorted([k for k in ref if k not in cur])
    extra = sorted([k for k in cur if k not in ref])
    mismatches = []
    for name in sorted(set(ref).intersection(cur)):
        try:
            a = float(ref[name])
            b = float(cur[name])
        except Exception:
            continue
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            mismatches.append({"name": name, "reference": a, "current": b, "abs_delta": abs(a - b)})
    return {
        "missing_count": len(missing),
        "extra_count": len(extra),
        "mismatch_count": len(mismatches),
        "missing": missing,
        "extra": extra,
        "mismatches": mismatches[:50],
        "mismatches_truncated": max(0, len(mismatches) - 50),
    }

def checkpoint_manifest_path(out_dir: Path) -> Path:
    return Path(out_dir) / "checkpoints" / "production_checkpoint_manifest.json"

def save_production_checkpoint(out_dir: Path, sims: list, assignments: list[int], prod_done: int, absolute_step: int,
                               parity: int, attempt: int, next_exchange: int, next_log: int, exchange_stats: dict, rng,
                               centers_a=None, k_list=None, cv_atom1=None, cv_atom2=None, cv_label=None, calib_steps=None,
                               secondary_cv_metadata=None, secondary_cv_centers=None, secondary_cv_k_kcal_list=None,
                               primary_cv_metadata=None) -> None:
    """Write restart checkpoints for all production replicas.

    The OpenMM binary checkpoint is platform/version specific but it is the most
    complete way to preserve Context state, velocities, RNG state, integrator
    state, and current global parameters.  The JSON manifest stores the Python
    bookkeeping needed to resume the GaREUS loop.

    In practice, some CustomIntegrator/gamd-openmm state can be hard to audit
    from the binary checkpoint alone.  Therefore the manifest also stores the
    full set of readable CustomIntegrator global variables per replica and the
    resume path explicitly restores those globals after Context.loadCheckpoint().
    This makes GaMD boost continuation inspectable instead of relying on a black
    box checkpoint assumption.
    """
    chk_dir = Path(out_dir) / "checkpoints"
    chk_dir.mkdir(parents=True, exist_ok=True)
    replica_files = []
    replica_integrator_globals_all = []
    replica_integrator_global_counts = []
    for r, sim in enumerate(sims):
        # Snapshot GaMD/CustomIntegrator globals at the same production state as
        # the checkpoint.  This is intentionally independent of samples.csv and
        # --write-gamd-globals-json, because restart correctness must not depend
        # on verbose sample logging being enabled.
        try:
            g = all_integrator_globals(sim.integrator)
        except Exception:
            g = {}
        replica_integrator_globals_all.append(_json_ready(g))
        replica_integrator_global_counts.append(int(len(g)))
        rel = f"replica_{r:03d}.chk"
        (chk_dir / rel).write_bytes(sim.context.createCheckpoint())
        replica_files.append(rel)
    manifest = {
        "schema": "gareus_production_checkpoint_v1",
        "created_unix_time": time.time(),
        "prod_done": int(prod_done),
        "absolute_step": int(absolute_step),
        "assignments": [int(x) for x in assignments],
        "parity": int(parity),
        "attempt": int(attempt),
        "next_exchange": int(next_exchange),
        "next_log": int(next_log),
        "exchange_stats": exchange_stats,
        "rng_bit_generator": rng.bit_generator.__class__.__name__,
        "rng_state": _json_ready(rng.bit_generator.state),
        "replica_checkpoint_files": replica_files,
        "replica_integrator_globals_all": replica_integrator_globals_all,
        "replica_integrator_global_counts": replica_integrator_global_counts,
        "integrator_globals_restore_note": "On resume, these CustomIntegrator globals are explicitly restored after Context.loadCheckpoint() so GaMD boost state does not depend only on opaque binary checkpoint behavior.",
        "note": "Resume requires the same OpenMM version/platform/system/topology and compatible command-line settings.",
    }
    if centers_a is not None:
        manifest["windows_A"] = [float(x) for x in centers_a]
    if k_list is not None:
        manifest["window_k_kcal_mol_A2"] = [float(x) for x in k_list]
    if cv_atom1 is not None:
        manifest["cv_atom1_index"] = int(cv_atom1)
    if cv_atom2 is not None:
        manifest["cv_atom2_index"] = int(cv_atom2)
    if cv_label is not None:
        manifest["cv_label"] = str(cv_label)
    if primary_cv_metadata is not None:
        manifest["primary_cv"] = _json_ready(primary_cv_metadata)
    if calib_steps is not None:
        manifest["shared_gamd_calibration_steps"] = int(calib_steps)
    if secondary_cv_metadata is not None:
        manifest["secondary_cv"] = _json_ready(secondary_cv_metadata)
    if secondary_cv_centers is not None:
        manifest["secondary_cv_centers"] = [float(x) for x in secondary_cv_centers]
    if secondary_cv_k_kcal_list is not None:
        manifest["secondary_cv_k_kcal_mol"] = [float(x) for x in secondary_cv_k_kcal_list]
    tmp = chk_dir / "production_checkpoint_manifest.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(checkpoint_manifest_path(out_dir))

def sync_scratch_to_main(scratch_dir: Path, main_dir: Path) -> None:
    """Copy everything from scratchdir to maindir (called at each checkpoint).

    Uses shutil.copytree with dirs_exist_ok so repeated calls are safe.
    Errors are printed as warnings; a failed sync does not abort the run.
    """
    scratch_dir = Path(scratch_dir)
    main_dir = Path(main_dir)
    if scratch_dir.resolve() == main_dir.resolve():
        return
    if not scratch_dir.exists():
        return
    main_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(str(scratch_dir), str(main_dir), dirs_exist_ok=True, copy_function=shutil.copy2)
    except Exception as exc:
        print(f"WARNING [scratchdir sync]: {scratch_dir} → {main_dir} failed: {exc}")

def load_production_checkpoint(out_dir: Path, sims: list, centers_nm, ks_kj_nm2, rng, secondary_centers=None, secondary_ks_kj=None) -> Optional[dict]:
    """Load a production checkpoint manifest and all replica checkpoints if available."""
    manifest_path = checkpoint_manifest_path(out_dir)
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("replica_checkpoint_files", [])
    if len(files) != len(sims):
        raise RuntimeError(f"Checkpoint replica count mismatch: manifest has {len(files)} files, current run has {len(sims)} replicas")
    chk_dir = manifest_path.parent
    pre_load_globals = []
    for sim in sims:
        try:
            pre_load_globals.append(all_integrator_globals(sim.integrator))
        except Exception:
            pre_load_globals.append({})
    for r, sim in enumerate(sims):
        data = (chk_dir / files[r]).read_bytes()
        sim.context.loadCheckpoint(data)
    # Do not rely only on the opaque OpenMM binary checkpoint for gamd-openmm
    # CustomIntegrator globals.  New manifests store them explicitly; old
    # manifests may recover them from exact-step gamd_globals_json if available.
    gamd_restore_report = _restore_gamd_integrator_globals_after_checkpoint(Path(out_dir), sims, manifest)
    manifest["resume_gamd_integrator_restore_report"] = gamd_restore_report
    if not bool(gamd_restore_report.get("ok", False)):
        try:
            post_load_globals = [all_integrator_globals(sim.integrator) for sim in sims]
            changed = 0
            for before, after in zip(pre_load_globals, post_load_globals):
                cmp = compare_gamd_global_sets(before, after)
                if int(cmp.get("mismatch_count", 0) or 0) > 0 or int(cmp.get("missing_count", 0) or 0) > 0 or int(cmp.get("extra_count", 0) or 0) > 0:
                    changed += 1
            if changed <= 0 and str(gamd_restore_report.get("source", "none")) == "none":
                print("WARNING: GaMD resume could not restore explicit integrator globals, and checkpoint load did not visibly change CustomIntegrator globals. Boost state may have restarted; see resume_gamd_integrator_restore_report.json")
            else:
                print("WARNING: GaMD resume integrator-global restore was incomplete; see resume_gamd_integrator_restore_report.json")
        except Exception:
            print("WARNING: GaMD resume integrator-global restore was incomplete; see resume_gamd_integrator_restore_report.json")
    else:
        print(f"    GaMD integrator globals restored for {gamd_restore_report.get('restored_replicas', 0)}/{len(sims)} replicas from {gamd_restore_report.get('source')}.")
    assignments = [int(x) for x in manifest.get("assignments", list(range(len(sims))))]
    if len(assignments) != len(sims):
        raise RuntimeError("Checkpoint assignment count does not match replica count")
    for r, sim in enumerate(sims):
        set_window(sim.context, centers_nm, ks_kj_nm2, int(assignments[r]), secondary_centers, secondary_ks_kj)
    try:
        if manifest.get("rng_state") is not None:
            rng.bit_generator.state = manifest["rng_state"]
    except Exception as exc:
        print(f"WARNING: could not restore NumPy RNG state from checkpoint: {exc}")
    return manifest

def restore_exchange_stats_from_csv_if_needed(out_dir: Path, args, exchange_stats: dict, max_step: Optional[int] = None) -> dict:
    """Best-effort rebuild of exchange dashboard stats from exchanges.csv.

    Checkpoint manifests are preferred because they exactly match the resumed
    physical state.  This fallback is useful for older manifests or interrupted
    runs that did not yet store the full exchange_stats dictionary.
    """
    exchange_stats = exchange_stats if isinstance(exchange_stats, dict) else {}
    if int(exchange_stats.get("attempts", 0) or 0) > 0:
        return {"restored": False, "reason": "manifest already contains exchange attempts"}
    path = Path(out_dir) / "exchanges.csv"
    if not path.exists() or path.stat().st_size <= 0:
        return {"restored": False, "reason": "exchanges.csv missing"}
    restored = {"attempts": 0, "accepted": 0, "pairs": {}, "jump_bins": {}, "mode": str(getattr(args, "exchange_mode", "neighbor")), "gibbs_choices": 0, "gibbs_moves": 0, "gibbs_stays": 0}
    rows_read = 0
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    step = int(float(row.get("step", "nan")))
                    if max_step is not None and step > int(max_step):
                        continue
                except Exception:
                    if max_step is not None:
                        continue
                try:
                    wi = int(float(row.get("window_i", "nan")))
                    wj = int(float(row.get("window_j", "nan")))
                except Exception:
                    continue
                accepted_raw = str(row.get("accepted", "")).strip().lower()
                accepted = accepted_raw in {"1", "true", "t", "yes", "y"}
                pair_key = f"{min(wi, wj)}-{max(wi, wj)}"
                jump_key = f"dw{abs(wi - wj)}"
                pst = restored.setdefault("pairs", {}).setdefault(pair_key, {"attempts": 0, "accepted": 0})
                jst = restored.setdefault("jump_bins", {}).setdefault(jump_key, {"attempts": 0, "accepted": 0})
                pst["attempts"] = int(pst.get("attempts", 0) or 0) + 1
                jst["attempts"] = int(jst.get("attempts", 0) or 0) + 1
                restored["attempts"] = int(restored.get("attempts", 0) or 0) + 1
                if accepted:
                    pst["accepted"] = int(pst.get("accepted", 0) or 0) + 1
                    jst["accepted"] = int(jst.get("accepted", 0) or 0) + 1
                    restored["accepted"] = int(restored.get("accepted", 0) or 0) + 1
                rows_read += 1
    except Exception as exc:
        return {"restored": False, "reason": f"failed reading exchanges.csv: {exc}"}
    if rows_read <= 0:
        return {"restored": False, "reason": "no usable exchange rows"}
    exchange_stats.clear()
    exchange_stats.update(restored)
    return {"restored": True, "rows": int(rows_read), "source": str(path)}

def run_production_probe(args, out_dir: Path, sims: list, assignments: list[int], centers_nm, ks_kj_nm2,
                         primary_cv_def: dict, cv_atom1: int, cv_atom2: int, unit, reference_gamd_globals: dict[str, float]) -> dict:
    """Run and roll back a tiny production probe to catch NaNs before the long run.

    Context checkpoints are restored afterwards, so the probe does not consume
    production time and does not affect the first production sample.
    """
    nsteps = int(getattr(args, "production_probe_steps", 0) or 0)
    report = {"enabled": nsteps > 0, "steps": nsteps, "replicas": []}
    if nsteps <= 0:
        write_json(Path(out_dir) / "production_probe_report.json", report)
        return report
    saved = [sim.context.createCheckpoint() for sim in sims]
    failures = []
    try:
        for r, sim in enumerate(sims):
            sim.step(nsteps)
            state = sim.context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True)
            pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            cv_value = primary_cv_value_from_positions_nm(pos_nm, primary_cv_def, args)
            pe_kj = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
            globals_now = integrator_globals(sim.integrator, unit=unit)
            boost_kj, boost_components, boost_source = extract_gamd_boost_kj(sim.integrator, unit, globals_now)
            row = {
                "replica": int(r),
                "window": int(assignments[r]),
                "cv_A": float(cv_value),
                "primary_cv": primary_cv_mode(args),
                "primary_cv_units": primary_cv_units(args),
                "potential_kj_mol": float(pe_kj),
                "gamd_boost_total_kj_mol": float(boost_kj) if boost_kj is not None and math.isfinite(float(boost_kj)) else None,
                "gamd_boost_source": str(boost_source),
                "gamd_boost_components_kj_mol": boost_components,
                "finite_cv": bool(math.isfinite(float(cv_value))),
                "finite_potential": bool(math.isfinite(float(pe_kj))),
                "finite_boost_or_unavailable": bool(boost_kj is None or math.isfinite(float(boost_kj))),
            }
            row["ok"] = bool(row["finite_cv"] and row["finite_potential"] and row["finite_boost_or_unavailable"])
            if not row["ok"]:
                failures.append(row)
            report["replicas"].append(row)
    except Exception as exc:
        failures.append({"exception": str(exc)})
        report["exception"] = str(exc)
    finally:
        for sim, checkpoint in zip(sims, saved):
            sim.context.loadCheckpoint(checkpoint)
        for r, sim in enumerate(sims):
            set_window(sim.context, centers_nm, ks_kj_nm2, int(assignments[r]))
    report["ok"] = len(failures) == 0
    report["failures"] = failures
    write_json(Path(out_dir) / "production_probe_report.json", report)
    if failures and not bool(getattr(args, "production_probe_warn_only", False)):
        raise RuntimeError(f"Production probe failed; see {Path(out_dir) / 'production_probe_report.json'}")
    if failures:
        print(f"WARNING: production probe found problems; see {Path(out_dir) / 'production_probe_report.json'}")
    else:
        print(f"    Production probe passed: {nsteps} rollback steps on {len(sims)} replicas")
    return report

def run_shared_gamd_setup_article_a(
    args,
    out_dir: Path,
    openmm,
    app,
    unit,
    topology,
    base_system,
    equil_state,
    platform,
    props,
    progress: Optional[GuiProgressSink] = None,
) -> tuple[dict[str, float], dict[str, float], int, bytes | None]:
    """Run one article-style shared GaMD calibration/equilibration.

    GaREUS in the original implementation obtains one GaMD setup for the peptide
    and reuses those GaMD parameters for every REUS/umbrella replica.  This
    function performs that single setup from the NPT-equilibrated structure with
    the umbrella force present but disabled (k=0), then returns all integrator
    globals so each production replica can start directly in the GaMD production
    stage with identical GaMD parameters.
    """
    calib_steps = int(
        args.gamd_cmd_prep_steps
        + args.gamd_cmd_steps
        + args.gamd_equil_prep_steps
        + args.gamd_equil_steps
    )
    shared_system = deserialize_system(openmm, base_system)
    shared_integrator, _shared_result = make_gamd_integrator(shared_system, args, unit)
    shared_sim = app.Simulation(topology, shared_system, shared_integrator, platform, props)

    box = equil_state.getPeriodicBoxVectors()
    if box is not None:
        shared_sim.context.setPeriodicBoxVectors(*box)
    shared_sim.context.setPositions(equil_state.getPositions())
    try:
        vel = equil_state.getVelocities()
        if vel is not None:
            shared_sim.context.setVelocities(vel)
        else:
            shared_sim.context.setVelocitiesToTemperature(args.temperature_k * unit.kelvin, args.seed + 101)
    except Exception:
        shared_sim.context.setVelocitiesToTemperature(args.temperature_k * unit.kelvin, args.seed + 101)

    # Keep the umbrella force in the System so the force layout matches the
    # production replicas, but disable it for the shared peptide GaMD setup.
    try:
        shared_sim.context.setParameter("k", 0.0)
    except Exception:
        pass
    try:
        shared_sim.context.setParameter("r0", 0.0)
    except Exception:
        pass

    chunk_default = int(getattr(args, "distance_output_interval", 0) or 0)
    if chunk_default <= 0:
        chunk_default = int(getattr(args, "report_interval", 1000) or 1000)
    if chunk_default <= 0:
        chunk_default = 1000
    chunk_default = max(1, chunk_default)

    print(f"    Shared article-style GaMD setup: one peptide calibration/equilibration for {calib_steps} steps")
    if progress is not None:
        progress.progress(
            "shared_gamd_setup",
            0,
            calib_steps,
            message="one shared GaMD setup; replicas will start in production",
            timestep_fs=float(args.timestep_fs),
            n_replicas=1,
            force=True,
        )

    done = 0
    while done < calib_steps:
        chunk = min(chunk_default, calib_steps - done)
        shared_sim.step(int(chunk))
        done += int(chunk)
        if progress is not None:
            progress.progress(
                "shared_gamd_setup",
                done,
                calib_steps,
                message="one shared GaMD setup; replicas will start in production",
                timestep_fs=float(args.timestep_fs),
                n_replicas=1,
                force=(done >= calib_steps),
            )

    shared_globals_all = all_integrator_globals(shared_integrator)
    shared_globals_interesting = integrator_globals(shared_integrator)

    payload = {
        "mode": "article_a_single_equilibrated_shared_gamd",
        "description": "One GaMD calibration/equilibration was run from the NPT-equilibrated peptide with umbrella k=0. The resulting same-name CustomIntegrator globals are copied to every GaREUS replica before production.",
        "calibration_steps": int(calib_steps),
        "temperature_K": float(args.temperature_k),
        "pressure_bar": float(getattr(args, "pressure_bar", 1.0)),
        "production_ensemble": str(getattr(args, "production_ensemble", "npt")),
        "production_barostat": "OpenMM MonteCarloBarostat" if str(getattr(args, "production_ensemble", "npt")) == "npt" else "none",
        "production_barostat_frequency": int(getattr(args, "production_barostat_frequency", 0) or getattr(args, "barostat_frequency", 100)),
        "timestep_fs": float(args.timestep_fs),
        "gamd_boost_type": str(args.gamd_boost_type),
        "sigma0p_kcal_mol": float(args.sigma0p_kcal_mol),
        "sigma0d_kcal_mol": float(args.sigma0d_kcal_mol),
        "interesting_globals": shared_globals_interesting,
        "all_globals": shared_globals_all,
        "note": "If using a gamd-openmm version that stores additional production restart information outside CustomIntegrator globals, compare this JSON with the package's native restart/log files before production science.",
    }
    write_json(out_dir / "shared_gamd_setup_globals.json", payload)

    shared_context_checkpoint = None
    try:
        shared_context_checkpoint = shared_sim.context.createCheckpoint()
        (out_dir / "shared_gamd_setup_context.chk").write_bytes(shared_context_checkpoint)
    except Exception as exc:
        print(f"WARNING: could not write shared GaMD setup Context checkpoint; falling back to global-variable copy only: {exc}")

    try:
        state = shared_sim.context.getState(getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=True)
        (out_dir / "shared_gamd_setup_state.xml").write_text(openmm.XmlSerializer.serialize(state), encoding="utf-8")
        with (out_dir / "shared_gamd_setup_final.pdb").open("w") as handle:
            app.PDBFile.writeFile(topology, state.getPositions(), handle, keepIds=True)
    except Exception as exc:
        write_json(out_dir / "shared_gamd_setup_state_write_warning.json", {"warning": str(exc)})

    print(f"    Shared GaMD globals written to {out_dir / 'shared_gamd_setup_globals.json'}")
    if shared_context_checkpoint is not None:
        print(f"    Shared GaMD Context checkpoint written to {out_dir / 'shared_gamd_setup_context.chk'}")
    # Post-equil sigma0 diagnostic: compare configured sigma0 to the actual
    # potential-energy spread (sigmaV) accumulated during calibration/equil.
    # sigmaV >> sigma0 → high anharmonicity, cumulant2 reweighting less reliable.
    try:
        _ig = shared_globals_interesting
        _sig0_kj = float(_ig.get("sigma0_Total", 0.0) or 0.0)
        _sigV_kj = float(_ig.get("sigmaV_Total", 0.0) or 0.0)
        _kj_to_kcal = 1.0 / 4.184
        if _sig0_kj > 0.0 and _sigV_kj > 0.0:
            _ratio = _sigV_kj / _sig0_kj
            _sig0_kcal = _sig0_kj * _kj_to_kcal
            _sigV_kcal = _sigV_kj * _kj_to_kcal
            print(
                f"    [GaMD sigma0 diagnostic] sigma0={_sig0_kcal:.2f} kcal/mol  "
                f"sigmaV(total)={_sigV_kcal:.2f} kcal/mol  ratio={_ratio:.2f}"
            )
            if _ratio > 3.0:
                _rec_kcal = _sigV_kcal * 0.40
                print(
                    f"    WARNING: sigmaV/sigma0 = {_ratio:.1f} > 3 — boost likely anharmonic. "
                    f"Consider sigma0p/sigma0d ≤ {_rec_kcal:.1f} kcal/mol "
                    f"(= 0.4 × sigmaV) to reduce anharmonicity score."
                )
            elif _ratio > 2.0:
                print(
                    f"    NOTE: sigmaV/sigma0 = {_ratio:.1f}; moderate anharmonicity expected. "
                    f"cumulant2 reweighting is appropriate."
                )
            else:
                print(f"    sigma0 calibration OK (sigmaV/sigma0 = {_ratio:.1f}).")
    except Exception:
        pass
    release_openmm_contexts(shared_sim, shared_integrator, shared_system)
    return shared_globals_all, shared_globals_interesting, int(calib_steps), shared_context_checkpoint

def run_gareus(args, out_dir: Path, openmm, app, unit, forcefield, topology, equil_state, progress: Optional[GuiProgressSink] = None):
    _register_graceful_shutdown()
    platform, props = platform_and_properties(openmm, args.platform, args.precision, args.device_index, args.cpu_threads, args=args)
    setup_platform, setup_props = setup_platform_and_properties(openmm, args)
    if not (Path(out_dir) / "run_manifest.json").exists():
        initialize_run_manifest(args, out_dir, argv=[])
    update_run_manifest(out_dir, {
        "status": "production_started",
        "openmm_runtime": {
            "setup_platform": str(setup_platform.getName()),
            "setup_platform_properties": dict(setup_props),
            "production_platform": str(platform.getName()),
            "production_platform_properties": dict(props),
        },
    })
    run_mode = production_run_mode(args)
    use_gamd = gamd_enabled(args)
    print(f"[setup] OpenMM setup platform: {platform_summary(setup_platform, setup_props)}")
    print(f"[production] OpenMM production platform: {platform_summary(platform, props)}")
    print(f"[production] Run mode: {run_mode} ({'GaMD/REUS' if use_gamd else 'conventional HMR umbrella/REUS MD, no gamd-openmm' if run_mode == 'hmr-cmd' else 'conventional umbrella/REUS MD, no gamd-openmm'})")
    try:
        _device_tokens = [x.strip() for x in str(getattr(args, "device_index", "") or "").split(",") if x.strip()]
        _device_mode = str(getattr(args, "replica_device_mode", "auto") or "auto")
        _device_map = str(getattr(args, "replica_device_map", "") or "")
        if _device_tokens or _device_map:
            print(f"[production] Replica device routing: mode={_device_mode}, device_index={','.join(_device_tokens) or '<empty>'}, map={_device_map or '<auto>'}")
    except Exception:
        pass
    resume_manifest = read_json_file(checkpoint_manifest_path(out_dir), None) if bool(getattr(args, "resume", False)) else None
    fast_resume = isinstance(resume_manifest, dict) and bool(resume_manifest.get("replica_checkpoint_files"))

    if fast_resume:
        resume_def = load_resume_run_definition(out_dir, topology, args, manifest=resume_manifest)
        cv_atom1 = int(resume_def["cv_atom1"])
        cv_atom2 = int(resume_def["cv_atom2"])
        cv_label = str(resume_def["cv_label"])
        # Resume older distance runs by default; contact runs reconstruct pairs from topology.
        try:
            resume_meta = (resume_def.get("metadata", {}) or {})
            resume_primary = resume_meta.get("primary_cv", getattr(args, "primary_cv", "distance"))
            args.primary_cv = primary_cv_mode(resume_primary)
            apply_primary_cv_metadata_to_args(args, resume_meta)
        except Exception:
            args.primary_cv = primary_cv_mode(getattr(args, "primary_cv", "distance"))
        primary_cv_def = prepare_primary_cv_definition(topology, args, cv_atom1=cv_atom1, cv_atom2=cv_atom2, cv_label=cv_label)
        cv_label = str(primary_cv_def.get("label", cv_label))
        centers_a = np.asarray(resume_def["centers_a"], dtype=float)
        k_list = [float(x) for x in resume_def["k_list"]]
        secondary_cv_metadata = dict(resume_def.get("secondary_cv_metadata", {"enabled": False}) or {"enabled": False})
        secondary_cv_centers = resume_def.get("secondary_cv_centers")
        secondary_cv_k_kcal_list = resume_def.get("secondary_cv_k_kcal_list")
        if secondary_cv_metadata.get("enabled") and not secondary_cv_enabled(args):
            # Resume must rebuild the same optional CV force even if the user did
            # not repeat the --secondary-cv flags on the command line.
            args.secondary_cv = str(secondary_cv_metadata.get("mode", "custom") or "custom")
            args.secondary_cv_phi0_deg = float(secondary_cv_metadata.get("phi0_deg", getattr(args, "secondary_cv_phi0_deg", -60.0)))
            args.secondary_cv_psi0_deg = float(secondary_cv_metadata.get("psi0_deg", getattr(args, "secondary_cv_psi0_deg", -45.0)))
            args.secondary_cv_sigma_deg = float(secondary_cv_metadata.get("sigma_deg", getattr(args, "secondary_cv_sigma_deg", 35.0)))
        window_metadata = dict(resume_def.get("window_metadata", {}))
        shared_gamd_globals_all = dict(resume_def.get("shared_gamd_globals_all", {}) or {})
        shared_gamd_globals_interesting = dict(resume_def.get("shared_gamd_globals_interesting", {}) or {})
        calib_steps = int(resume_def.get("calib_steps", 0) or 0)
        shared_gamd_context_checkpoint = None
        print(f"[resume] Production checkpoint manifest found; skipping window generation, US pulling, and shared GaMD setup.")
    else:
        if equil_state is None:
            raise RuntimeError("No equilibrated state is available; cannot start GaREUS without a production checkpoint or saved 03_npt_equilibrated_state.xml.")
        cv_atom1, cv_atom2, distance_cv_label = choose_cv_atoms(topology, args)
        primary_cv_def = prepare_primary_cv_definition(topology, args, cv_atom1=cv_atom1, cv_atom2=cv_atom2, cv_label=distance_cv_label)
        cv_label = str(primary_cv_def.get("label", distance_cv_label))
        if getattr(args, "windows_2d_csv", None):
            centers_a, k_list, secondary_cv_centers, secondary_cv_k_kcal_list, secondary_cv_metadata, window_metadata = load_explicit_2d_window_csv(args, Path(args.windows_2d_csv))
            print(f"    Explicit 2D window table loaded: {len(centers_a)} windows from {args.windows_2d_csv}")
            try:
                with (out_dir / "explicit_2d_window_table_loaded.csv").open("w", newline="") as handle:
                    norm_rows = window_metadata.get("normalized_rows", []) if isinstance(window_metadata, dict) else []
                    if norm_rows:
                        writer = csv.DictWriter(handle, fieldnames=list(norm_rows[0].keys()), extrasaction="ignore")
                        writer.writeheader()
                        writer.writerows(norm_rows)
                write_json(out_dir / "explicit_2d_window_table_loaded.json", _json_ready(window_metadata))
            except Exception as exc:
                print(f"WARNING: failed to write normalized explicit 2D window table: {exc}")
        else:
            centers_a, k_list, window_metadata = choose_windows(
                args, out_dir, openmm, app, unit, forcefield, topology, equil_state, cv_atom1, cv_atom2, progress=progress
            )
            centers_a, k_list, secondary_cv_centers, secondary_cv_k_kcal_list, secondary_cv_metadata = expand_windows_for_secondary_cv(args, centers_a, k_list)
            if secondary_cv_metadata.get("enabled"):
                window_metadata = dict(window_metadata)
                window_metadata["secondary_cv_expansion"] = secondary_cv_metadata

    window_metadata = dict(window_metadata or {})
    window_metadata.update({
        "primary_cv": primary_cv_mode(args),
        "primary_cv_label": primary_cv_label(args),
        "primary_cv_units": primary_cv_units(args),
        "primary_k_units": primary_k_units(args),
        "primary_openmm_k_units": primary_openmm_k_units(args),
        "primary_cv_definition": _json_ready(primary_cv_def),
    })
    window_rows = window_assignment_rows(centers_a, k_list, args.temperature_k, secondary_cv_centers, secondary_cv_k_kcal_list, args=args)
    write_window_assignment_csv(out_dir / "umbrella_windows.csv", window_rows)
    print_window_assignment_table(window_rows)
    graph_summary = None
    if isinstance(secondary_cv_metadata, dict) and bool(secondary_cv_metadata.get("explicit_2d_windows", False)) and secondary_cv_centers is not None:
        try:
            graph_summary = write_explicit_2d_neighbor_graph_files(out_dir, centers_a, secondary_cv_centers, args=args, prefix="explicit_2d_neighbor_graph")
            window_metadata = dict(window_metadata or {})
            window_metadata["explicit_2d_neighbor_graph"] = graph_summary
            secondary_cv_metadata = dict(secondary_cv_metadata or {})
            secondary_cv_metadata["explicit_2d_neighbor_graph"] = graph_summary
            print(f"    Explicit 2D neighbor graph: {graph_summary.get('n_edges', 0)} edges -> {graph_summary.get('edge_csv', '')}")
        except Exception as exc:
            print(f"WARNING: failed to write explicit 2D neighbor graph diagnostics: {exc}")

    try:
        explicit_window_table_summary = write_explicit_window_analysis_files(
            out_dir, centers_a, k_list, secondary_cv_centers, secondary_cv_k_kcal_list,
            secondary_cv_metadata=secondary_cv_metadata,
            window_metadata=window_metadata,
            neighbor_graph_summary=graph_summary,
        )
        window_metadata = dict(window_metadata or {})
        window_metadata["explicit_window_table"] = explicit_window_table_summary
        print(f"    Sparse-safe window metadata: {explicit_window_table_summary.get('csv', '')}")
    except Exception as exc:
        explicit_window_table_summary = {}
        print(f"WARNING: failed to write sparse-safe explicit window metadata: {exc}")

    # Historical variable names retained: in distance mode centers_nm/ks_kj_nm2 are
    # nm and kJ/mol/nm^2; in contact mode they are CV units and kJ/mol/CV^2.
    centers_nm = np.asarray([primary_center_to_openmm_value(c, args) for c in centers_a], dtype=float)
    ks_kj_nm2 = np.asarray([primary_k_to_openmm_value(k, args) for k in k_list], dtype=float)
    secondary_cv_ks_kj = np.asarray([kcal_to_kj(k) for k in secondary_cv_k_kcal_list], dtype=float) if secondary_cv_k_kcal_list is not None else None
    nrep = len(centers_nm)

    _max_replicas = int(getattr(args, "max_replicas", 0) or 0)
    if _max_replicas > 0 and nrep > _max_replicas:
        print(f"[production] --max-replicas {_max_replicas}: truncating {nrep} windows to {_max_replicas}")
        centers_nm = centers_nm[:_max_replicas]
        ks_kj_nm2 = ks_kj_nm2[:_max_replicas]
        centers_a = centers_a[:_max_replicas]
        k_list = k_list[:_max_replicas]
        if secondary_cv_ks_kj is not None:
            secondary_cv_ks_kj = secondary_cv_ks_kj[:_max_replicas]
        if secondary_cv_centers is not None:
            secondary_cv_centers = np.asarray(secondary_cv_centers, dtype=float)[:_max_replicas]
        if secondary_cv_k_kcal_list is not None:
            secondary_cv_k_kcal_list = list(secondary_cv_k_kcal_list)[:_max_replicas]
        nrep = _max_replicas

    # Resolve per-replica CPU threads now that nrep is known.
    # --cpu-budget distributes total cores evenly; --max-cpu-per-replica caps the result.
    if str(getattr(args, "platform", "")).upper() == "CPU":
        _resolved_threads = resolve_cpu_threads_for_replicas(args, nrep)
        if _resolved_threads != int(getattr(args, "cpu_threads", 1) or 1):
            print(f"[production] CPU budget: {_resolved_threads} threads/replica "
                  f"(budget={getattr(args, 'cpu_budget', 0)}, max_per={getattr(args, 'max_cpu_per_replica', 0)}, replicas={nrep})")
        props["Threads"] = str(_resolved_threads)

    production_ensemble = str(getattr(args, "production_ensemble", "npt") or "npt").lower()
    if production_ensemble not in {"npt", "nvt"}:
        raise ValueError(f"Unsupported --production-ensemble {production_ensemble!r}; use npt or nvt")
    production_include_barostat = production_ensemble == "npt"
    production_barostat_frequency = int(getattr(args, "production_barostat_frequency", 0) or getattr(args, "barostat_frequency", 100))

    # Base production system for GaREUS. Default production is now NPT with
    # OpenMM's MonteCarloBarostat; --production-ensemble nvt preserves the old
    # fixed-box behavior explicitly.
    base_system = create_system(
        app, unit, forcefield, topology, args,
        include_barostat=production_include_barostat,
        barostat_frequency=production_barostat_frequency,
    )
    add_primary_umbrella_force(openmm, base_system, primary_cv_def, args, args.umbrella_force_group)
    secondary_cv_force_info = add_secondary_structure_cv_force(
        openmm, base_system, topology, args, force_group=int(getattr(args, "secondary_cv_force_group", 29))
    ) if (secondary_cv_metadata or {}).get("enabled") else {"enabled": False}
    if secondary_cv_force_info.get("enabled"):
        secondary_cv_metadata.update(secondary_cv_force_info)
        print(
            f"    Secondary CV enabled: {secondary_cv_metadata.get('label', 'secondary structure')} "
            f"with {secondary_cv_metadata.get('n_phi_torsions', 0)} phi and {secondary_cv_metadata.get('n_psi_torsions', 0)} psi torsions"
        )

    if fast_resume:
        pos = vel = box = None
        window_start_positions = [None] * nrep
        window_start_velocities = [None] * nrep
    else:
        # Keep pre-production US starting-structure pulling on the previous fixed-box
        # system so changing the default production ensemble does not silently alter
        # the pulling preparation. The shared GaMD setup and production replicas use
        # the production system above.
        starting_structure_system = create_system(app, unit, forcefield, topology, args, include_barostat=False)
        add_primary_umbrella_force(openmm, starting_structure_system, primary_cv_def, args, args.umbrella_force_group)
        if (secondary_cv_metadata or {}).get("enabled"):
            add_secondary_structure_cv_force(openmm, starting_structure_system, topology, args, force_group=int(getattr(args, "secondary_cv_force_group", 29)))
        pos = equil_state.getPositions()
        vel = equil_state.getVelocities()
        box = equil_state.getPeriodicBoxVectors()

        window_start_positions, window_start_velocities = generate_us_starting_states_by_pulling(
            args, out_dir, openmm, app, unit, topology, starting_structure_system, centers_nm, ks_kj_nm2,
            equil_state, primary_cv_def, cv_atom1, cv_atom2, setup_platform, setup_props, progress=progress,
            secondary_cv_centers=secondary_cv_centers, secondary_cv_ks_kj=secondary_cv_ks_kj,
            secondary_cv_metadata=secondary_cv_metadata,
        )

        if use_gamd:
            reusable_gamd = load_reusable_shared_gamd_setup(args, out_dir)
            if reusable_gamd is not None:
                shared_gamd_globals_all, shared_gamd_globals_interesting, calib_steps, shared_gamd_context_checkpoint, reuse_note = reusable_gamd
                print(
                    "    GaMD shared setup: reusing campaign/global calibration from "
                    f"{reuse_note.get('source_shared_gamd_setup_dir', '<unknown>')}"
                )
            else:
                # GaMD calibration is sensitive to the starting PE range.  The NPT-
                # equilibrated structure is typically extended (high PE for chignolin),
                # which causes Vmin to be overestimated and makes compact-window boosts
                # disproportionately large.  Use the most compact pulled window instead:
                # it is closest to the PMF minimum and gives a Vmin that covers the
                # energy range production replicas actually visit.
                if primary_cv_is_contacts(args) and window_start_positions:
                    try:
                        _compact_idx = int(np.nanargmax(np.asarray(centers_a, dtype=float)))
                    except Exception:
                        _compact_idx = 0
                else:
                    _compact_idx = 0
                _compact_pos = window_start_positions[_compact_idx] if window_start_positions and window_start_positions[_compact_idx] is not None else None
                _compact_vel = window_start_velocities[_compact_idx] if window_start_velocities and window_start_velocities[_compact_idx] is not None else None
                if _compact_pos is not None:
                    _equil_box = equil_state.getPeriodicBoxVectors()
                    class _CompactState:
                        def getPositions(self): return _compact_pos
                        def getVelocities(self): return _compact_vel
                        def getPeriodicBoxVectors(self): return _equil_box
                    gamd_start_state = _CompactState()
                    print(f"    GaMD shared setup: starting from primary-CV window {_compact_idx} (better Vmin estimate than extended equil structure)")
                else:
                    gamd_start_state = equil_state
                    print("    GaMD shared setup: compact window unavailable, falling back to NPT-equilibrated structure")
                shared_gamd_globals_all, shared_gamd_globals_interesting, calib_steps, shared_gamd_context_checkpoint = run_shared_gamd_setup_article_a(
                    args, out_dir, openmm, app, unit, topology, base_system, gamd_start_state, setup_platform, setup_props, progress=progress
                )
                exported = export_shared_gamd_setup_if_requested(args, out_dir)
                if exported:
                    print(
                        "    GaMD shared setup: exported campaign/global calibration to "
                        f"{getattr(args, 'shared_gamd_export_dir', '') or getattr(args, '_global_shared_gamd_export_dir', '')}"
                    )
        else:
            calib_steps = 0
            shared_gamd_globals_all = {}
            shared_gamd_globals_interesting = {}
            shared_gamd_context_checkpoint = None
            write_json(out_dir / "shared_gamd_setup_globals.json", {
                "mode": f"disabled_{run_mode}",
                "description": f"GaMD disabled by --run-mode {run_mode}. Production uses plain LangevinMiddleIntegrator with the same umbrella/REUS machinery.",
                "calibration_steps": 0,
                "run_mode": run_mode,
                "temperature_K": float(args.temperature_k),
                "timestep_fs": float(args.timestep_fs),
                "friction_per_ps": float(args.friction_per_ps),
                "all_globals": {},
                "interesting_globals": {},
            })
            print(f"    GaMD disabled by --run-mode {run_mode}; skipping shared GaMD calibration and using conventional Langevin production integrators.")
        # Make sure all short-lived preparation Contexts are released before the
        # burst of production replica Context creation.  OpenCL in particular may
        # otherwise fail at clCreateContext(-6) even before the first replica.
        release_openmm_contexts()

    sims = []
    assignments = list(range(nrep))
    traj_dir = out_dir / "replica_trajectories"
    # Adaptive-feedback pilots are diagnostic and short-lived.  Unless the user
    # explicitly opts in, skip trajectory reporters during pilots to avoid
    # filesystem overhead and large disposable DCD/XTC output.
    adaptive_phase_info = getattr(args, "_adaptive_phase_info", {}) or {}
    pilot_traj_enabled = bool(getattr(args, "adaptive_pilot_trajectories", False))
    if adaptive_phase_info.get("is_pilot") and not pilot_traj_enabled:
        effective_traj_interval = 0
    else:
        effective_traj_interval = int(getattr(args, "traj_interval", 0) or 0)
    if effective_traj_interval > 0:
        traj_dir.mkdir(exist_ok=True)

    # Solute-only trajectories (--traj-solute-only): record just the peptide atoms
    # (ascending indices) and write a companion solute_only.pdb so analysis can read
    # the trimmed frames. Subset is derived from the topology so it stays consistent
    # across resume segments even when positions are loaded from a checkpoint.
    traj_atom_subset = None
    if effective_traj_interval > 0 and bool(getattr(args, "traj_solute_only", False)):
        traj_atom_subset = solute_atom_indices(topology) or None
        if traj_atom_subset is not None and pos is not None:
            _solute_pdb = write_solute_only_topology_pdb(out_dir, app, topology, pos, traj_atom_subset)
            print(f"[setup] traj-solute-only: recording {len(traj_atom_subset)} solute atoms; topology -> {_solute_pdb}")

    if primary_cv_is_contacts(args):
        print(
            f"[4/4] Building {nrep} GaREUS replicas with primary CV {cv_label}: "
            f"{primary_cv_def.get('n_contact_terms', primary_cv_def.get('n_contact_pairs', 0))} weighted contact terms; "
            f"effective contacts {primary_cv_def.get('n_effective_contact_pairs', primary_cv_def.get('n_contact_pairs', 0)):.3g}; "
            f"scheme {primary_cv_def.get('contact_scheme', 'atom-pairs')}"
        )
    else:
        print(f"[4/4] Building {nrep} GaREUS replicas with CV {cv_label}: atoms {cv_atom1}, {cv_atom2}")
    print("    Each replica receives the same shared GaMD setup globals and starts directly in production.")
    if progress is not None:
        progress.progress("replica_construction", 0, nrep, message=f"{nrep} replicas", force=True)
    replica_gamd_copy_report = []
    for i in range(nrep):
        system_i = deserialize_system(openmm, base_system)
        integrator_i, gamd_result = make_production_integrator(openmm, system_i, args, unit)
        copied_globals, skipped_globals = {}, {}
        loaded_shared_gamd_checkpoint = False
        props_i = replica_platform_properties(platform, props, args, i)
        try:
            sim_i = app.Simulation(topology, system_i, integrator_i, platform, props_i)
        except Exception as exc:
            # Context construction can fail transiently if Python still holds old
            # OpenMM contexts from setup/pulling phases.  Force a collection and
            # retry once before surfacing a targeted diagnostic.
            release_openmm_contexts()
            try:
                sim_i = app.Simulation(topology, system_i, integrator_i, platform, props_i)
            except Exception as exc2:
                platform_name = str(platform.getName()) if platform is not None else str(getattr(args, "platform", "auto"))
                raise RuntimeError(
                    f"Failed to initialize OpenMM context for replica {i}/{nrep} on platform {platform_name}: {exc2}. "
                    "If this is OpenCL clCreateContext(-6), the driver could not allocate host/platform resources. "
                    "Use --platform CUDA when available, reduce the number of adaptive windows/replicas, split devices with --device-index, "
                    "or force --platform CPU for a small debugging run. Also start from a fresh output directory after this failure."
                ) from exc2
        if use_gamd and not fast_resume and shared_gamd_context_checkpoint is not None:
            try:
                # Loading the calibrated shared Context checkpoint preserves the
                # gamd-openmm native stage/step state in addition to readable
                # CustomIntegrator globals.  We overwrite coordinates, velocities,
                # and umbrella parameters below, so only the GaMD setup state is reused.
                sim_i.context.loadCheckpoint(shared_gamd_context_checkpoint)
                loaded_shared_gamd_checkpoint = True
            except Exception as exc:
                skipped_globals["<shared_context_checkpoint>"] = f"load failed; falling back to CustomIntegrator globals only: {exc}"
        if use_gamd and not fast_resume:
            # Always copy calibrated globals from the dict regardless of whether the
            # checkpoint loaded.  loadCheckpoint() can silently succeed without
            # actually restoring CustomIntegrator globals (e.g. cross-platform or
            # cross-context mismatch), leaving Vmax/Vmin/k0/ForceScalingFactor at
            # zero/default and causing NaN forces at step 0.  The dict copy is
            # authoritative for calibration outputs; the checkpoint provides the
            # opaque GaMD binary stage state as a bonus.
            if shared_gamd_globals_all:
                copied_globals, copied_skipped = set_integrator_globals_from_dict(integrator_i, shared_gamd_globals_all)
                skipped_globals.update({k: v for k, v in copied_skipped.items() if k not in skipped_globals})
        replica_gamd_copy_report.append({
            "replica": int(i),
            "copied_count": int(len(copied_globals)),
            "skipped_count": int(len(skipped_globals)),
            "copied_globals": copied_globals if i == 0 else {},
            "skipped_globals": skipped_globals if i == 0 else {},
            "loaded_from_context_checkpoint": bool(fast_resume),
            "loaded_from_shared_gamd_setup_checkpoint": bool(loaded_shared_gamd_checkpoint),
        })
        if not fast_resume:
            sim_i.context.setPeriodicBoxVectors(*box)
            start_pos = window_start_positions[i] if i < len(window_start_positions) and window_start_positions[i] is not None else pos
            start_vel = window_start_velocities[i] if i < len(window_start_velocities) and window_start_velocities[i] is not None else vel
            sim_i.context.setPositions(start_pos)
            if args.randomize_replica_velocities:
                sim_i.context.setVelocitiesToTemperature(args.temperature_k * unit.kelvin, args.seed + i + 17)
            else:
                sim_i.context.setVelocities(start_vel)
            set_window(sim_i.context, centers_nm, ks_kj_nm2, i, secondary_cv_centers, secondary_cv_ks_kj)
        if effective_traj_interval > 0 and not bool(getattr(args, "resume", False)):
            reporter = make_trajectory_reporter(app, traj_dir / f"replica_{i:03d}", effective_traj_interval, args, atom_subset=traj_atom_subset)
            if reporter is not None:
                sim_i.reporters.append(reporter)
        sims.append(sim_i)
        if progress is not None:
            progress.progress("replica_construction", i + 1, nrep, message=f"built replica {i + 1}/{nrep}")

    if progress is not None:
        progress.progress("replica_construction", nrep, nrep, message=f"{nrep} replicas ready", force=True)

    # ── Fast CV path: cache per-replica CustomCVForce references ────────────────
    # After step(), OpenMM caches collective-variable values inside each Context.
    # getCollectiveVariableValues(context) reads those cached scalars without any
    # GPU→CPU position transfer, saving the dominant DMA cost at each sample and
    # exchange interval.  All replica systems are deserialized copies of base_system
    # and share the same force-index layout.
    _primary_umbrella_fg = int(getattr(args, "umbrella_force_group", 31))
    _secondary_cv_fg = int(getattr(args, "secondary_cv_force_group", 29))
    _fast_primary_force_idx: int = -1
    _fast_ss_force_idx: int = -1
    if sims:
        _probe_sys = sims[0].system
        for _fi in range(_probe_sys.getNumForces()):
            _f = _probe_sys.getForce(_fi)
            if not hasattr(_f, "getCollectiveVariableValues"):
                continue
            _fg = _f.getForceGroup()
            if _fg == _primary_umbrella_fg:
                _fast_primary_force_idx = _fi
            elif _fg == _secondary_cv_fg:
                _fast_ss_force_idx = _fi
    _fast_primary_forces = (
        [sim.system.getForce(_fast_primary_force_idx) for sim in sims]
        if _fast_primary_force_idx >= 0 else [None] * nrep
    )
    _fast_ss_forces = (
        [sim.system.getForce(_fast_ss_force_idx) for sim in sims]
        if _fast_ss_force_idx >= 0 else [None] * nrep
    )
    _ss_enabled_global = bool((secondary_cv_metadata or {}).get("enabled"))
    _use_fast_cv_path = (
        _fast_primary_force_idx >= 0
        and (not _ss_enabled_global or _fast_ss_force_idx >= 0)
    )
    # ────────────────────────────────────────────────────────────────────────────

    copy_sanity_rows = []
    copy_problem_count = 0
    if not use_gamd:
        copy_report_payload = {
            "description": f"GaMD disabled by --run-mode {run_mode}; no shared GaMD globals are copied.",
            "shared_gamd_globals_file": str(out_dir / "shared_gamd_setup_globals.json"),
            "replicas": replica_gamd_copy_report,
            "sanity_check": {
                "ok": True,
                "problem_count": 0,
                "per_replica": [],
                "note": "Conventional MD mode uses OpenMM LangevinMiddleIntegrator and has no gamd-openmm state.",
            },
        }
        write_json(out_dir / "replica_shared_gamd_copy_report.json", copy_report_payload)
        print(f"    Shared GaMD copy sanity: skipped (--run-mode {run_mode}).")
    elif fast_resume:
        copy_report_payload = {
            "description": "Skipped shared-GaMD global copy sanity because --resume restores each Context directly from OpenMM binary checkpoints, including integrator state.",
            "shared_gamd_globals_file": str(out_dir / "shared_gamd_setup_globals.json"),
            "replicas": replica_gamd_copy_report,
            "sanity_check": {
                "ok": True,
                "problem_count": 0,
                "per_replica": [],
                "note": "Context.loadCheckpoint() supersedes the pre-production global-copy restart path.",
            },
        }
        write_json(out_dir / "replica_shared_gamd_copy_report.json", copy_report_payload)
        print("    Shared GaMD copy sanity: skipped for checkpoint resume; integrator state will be loaded from checkpoint.")
    else:
        for i, sim in enumerate(sims):
            cmp = compare_gamd_global_sets(shared_gamd_globals_all, all_integrator_globals(sim.integrator))
            row = {"replica": int(i), **cmp}
            copy_sanity_rows.append(row)
            copy_problem_count += int(cmp.get("missing_count", 0)) + int(cmp.get("mismatch_count", 0)) + int(replica_gamd_copy_report[i].get("skipped_count", 0))
        copy_report_payload = {
            "description": "Same-name CustomIntegrator globals copied from the single shared GaMD setup into each production replica, then re-read and compared before production.",
            "shared_gamd_globals_file": str(out_dir / "shared_gamd_setup_globals.json"),
            "replicas": replica_gamd_copy_report,
            "sanity_check": {
                "ok": copy_problem_count == 0,
                "problem_count": int(copy_problem_count),
                "per_replica": copy_sanity_rows,
                "note": "Skipped/missing/mismatched globals may mean this gamd-openmm version stores restart state differently than expected.",
            },
        }
        write_json(out_dir / "replica_shared_gamd_copy_report.json", copy_report_payload)
        print(f"    Shared GaMD copy sanity: {'OK' if copy_problem_count == 0 else 'WARNING'} ({copy_problem_count} copied-global problems)")
        if copy_problem_count and bool(getattr(args, "shared_gamd_copy_strict", False)):
            raise RuntimeError(f"Shared GaMD copy sanity check failed; see {out_dir / 'replica_shared_gamd_copy_report.json'}")

    metadata = {
        "sequence": args.seq,
        "primary_cv": primary_cv_mode(args),
        "primary_cv_label": primary_cv_label(args),
        "primary_cv_units": primary_cv_units(args),
        "primary_k_units": primary_k_units(args),
        "primary_sample_column": "cv_A",
        "primary_center_column": "center_A",
        "primary_k_column": "k_kcal_mol_A2",
        "legacy_primary_cv_column_names": True,
        "primary_cv_definition": _json_ready(primary_cv_def),
        "cv_label": cv_label,
        "cv_atom1_index": cv_atom1,
        "cv_atom2_index": cv_atom2,
        "windows_A": centers_a.tolist(),
        "window_k_kcal_per_mol_A2": [float(x) for x in k_list],
        "window_assignment_table_csv": str(out_dir / "umbrella_windows.csv"),
        "window_assignment_rows": _json_ready(window_rows),
        "window_metadata": _json_ready(window_metadata),
        "explicit_window_table": _json_ready(explicit_window_table_summary),
        "secondary_cv": _json_ready(secondary_cv_metadata),
        "secondary_cv_centers": [float(x) for x in secondary_cv_centers] if secondary_cv_centers is not None else [],
        "secondary_cv_k_kcal_mol": [float(x) for x in secondary_cv_k_kcal_list] if secondary_cv_k_kcal_list is not None else [],
        "us_starting_structure_mode": str(getattr(args, "us_starting_structure_mode", "pull")),
        "us_starting_structures_dir": str(out_dir / "us_starting_structures"),
        "us_pulling_log_csv": str(out_dir / "us_starting_structures" / "us_pulling_starting_structures.csv"),
        "run_mode": run_mode,
        "gamd_enabled": bool(use_gamd),
        "hmr": bool(getattr(args, "hmr", False)),
        "hydrogen_mass_amu": float(getattr(args, "hydrogen_mass_amu", 0.0) or 0.0),
        "production_ensemble": production_ensemble,
        "production_barostat": "OpenMM MonteCarloBarostat" if production_include_barostat else "none",
        "production_pressure_bar": float(getattr(args, "pressure_bar", 1.0)),
        "production_barostat_frequency": int(production_barostat_frequency) if production_include_barostat else 0,
        "gamd_setup_mode": "article_a_single_equilibrated_shared_gamd" if use_gamd else f"disabled_{run_mode}",
        "shared_gamd_calibration_steps": int(calib_steps),
        "shared_gamd_globals_json": str(out_dir / "shared_gamd_setup_globals.json"),
        "shared_gamd_context_checkpoint": str(out_dir / "shared_gamd_setup_context.chk"),
        "replica_shared_gamd_copy_report_json": str(out_dir / "replica_shared_gamd_copy_report.json"),
        "replicas_start_directly_in_gamd_production": bool(use_gamd),
        "exchange_mode": str(getattr(args, "exchange_mode", "neighbor")),
        "exchange_note": "neighbor/random-pair/all-pair-sweep use Metropolis swaps between umbrella states; gibbs-walk is a heat-bath-like long-jump update over window swaps. Shared GaMD terms cancel from exchange energies.",
        "gamd_boost_type": args.gamd_boost_type,
        "sigma0p_kcal_mol": args.sigma0p_kcal_mol,
        "sigma0d_kcal_mol": args.sigma0d_kcal_mol,
        "progress_jsonl": str(out_dir / str(getattr(args, "progress_jsonl", "progress.jsonl"))),
        "distance_csv": str(out_dir / str(getattr(args, "distance_csv", "distances.csv"))),
        "distance_jsonl": str(out_dir / str(getattr(args, "distance_jsonl", "distances.jsonl"))),
        "note": "Quick GaREUS-like OpenMM scaffold. Validate force groups and reweighting for production free energies.",
        "resume_fast_path": bool(fast_resume),
    }
    write_json(out_dir / "gareus_metadata.json", metadata)

    beta = 1.0 / (unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin) * args.temperature_k)
    explicit_2d_run = bool((secondary_cv_metadata or {}).get("explicit_2d_windows", False))
    rectangular_2d_run = bool((secondary_cv_metadata or {}).get("grid", False)) if explicit_2d_run else False
    sparse_2d_run = bool(explicit_2d_run and not rectangular_2d_run)
    pymbar_metadata = {
        "schema_version": "2.0-sparse-2d-safe",
        "sparse_2d_safe": True,
        "description": "Umbrella bias definition and constants for later PyMBAR/MBAR analysis. Window definitions are explicit and window-major; sparse/non-rectangular 2D grids must not be reconstructed from a rectangular product unless rectangular_grid is true.",
        "bias_expression_user_units": "U_total_kcal = 0.5 * k_primary_i * (cv_A - center_A_i)**2 + 0.5 * k_secondary_i * (secondary_cv - secondary_center_i)**2 when secondary CV is enabled; cv_A is a legacy alias for the selected primary CV",
        "bias_expression_openmm_units": "distance: 0.5*k_kj_mol_nm2*(r_nm-r0_nm)^2; nonlocal-contacts: 0.5*k_kj_mol_CV2*(C-C0)^2; plus optional secondary CV bias",
        "temperature_K": float(args.temperature_k),
        "pressure_bar": float(getattr(args, "pressure_bar", 1.0)),
        "production_ensemble": production_ensemble,
        "production_barostat": "OpenMM MonteCarloBarostat" if production_include_barostat else "none",
        "production_barostat_frequency": int(production_barostat_frequency) if production_include_barostat else 0,
        "run_mode": run_mode,
        "gamd_enabled": bool(use_gamd),
        "hmr": bool(getattr(args, "hmr", False)),
        "beta_1_over_kJ_mol": float(beta),
        "primary_cv": primary_cv_mode(args),
        "primary_cv_label": primary_cv_label(args),
        "primary_cv_units": primary_cv_units(args),
        "primary_k_units": primary_k_units(args),
        "primary_cv_definition": _json_ready(primary_cv_def),
        "primary_sample_column": "cv_A",
        "primary_center_column": "center_A",
        "primary_k_column": "k_kcal_mol_A2",
        "legacy_primary_cv_column_names": True,
        "cv_label": cv_label,
        "cv_atom1_index": int(cv_atom1),
        "cv_atom2_index": int(cv_atom2),
        "n_windows": int(nrep),
        "explicit_2d": bool(explicit_2d_run),
        "rectangular_grid": bool(rectangular_2d_run),
        "sparse_2d": bool(sparse_2d_run),
        "explicit_window_table_csv": str((explicit_window_table_summary or {}).get("csv", out_dir / "umbrella_explicit_windows.csv")),
        "explicit_window_table_json": str((explicit_window_table_summary or {}).get("json", out_dir / "umbrella_explicit_windows.json")),
        "neighbor_graph_csv": str((graph_summary or {}).get("edge_csv", "")),
        "neighbor_graph_json": str((graph_summary or {}).get("edge_json", "")),
        "window_centers_A": [float(x) for x in centers_a],
        "window_centers_nm": [float(primary_center_to_openmm_value(x, args)) for x in centers_a],
        "window_k_kcal_mol_A2": [float(x) for x in k_list],
        "window_k_kj_mol_nm2": [float(primary_k_to_openmm_value(x, args)) for x in k_list],
        "window_centers_primary_units": [float(x) for x in centers_a],
        "window_k_primary_units": [float(x) for x in k_list],
        "secondary_cv": _json_ready(secondary_cv_metadata),
        "secondary_cv_centers": [float(x) for x in secondary_cv_centers] if secondary_cv_centers is not None else [],
        "secondary_cv_k_kcal_mol": [float(x) for x in secondary_cv_k_kcal_list] if secondary_cv_k_kcal_list is not None else [],
        "secondary_cv_k_kj_mol": [float(kcal_to_kj(x)) for x in secondary_cv_k_kcal_list] if secondary_cv_k_kcal_list is not None else [],
        "umbrella_windows_csv": str(out_dir / "umbrella_windows.csv"),
        "samples_csv": str(out_dir / "samples.csv"),
        "analysis_arrays_npz": str(out_dir / "analysis_arrays.npz"),
        "analysis_chunks_manifest_json": str(out_dir / "analysis_chunks_manifest.json"),
        "analysis_arrays_note": "Binary NumPy companion for PyMBAR/adaptive feedback. Large runs should prefer analysis_chunks_manifest.json; consolidated analysis_arrays.npz is optional compatibility output. Matrix arrays are sample-major [n_samples, n_windows] and safe for sparse/non-rectangular 2D grids because each column is the explicit window index.",
        "analysis_arrays_required_for_sparse_mbar": [
            "window", "cv_A", "secondary_cv",
            "distance_umbrella_bias_kcal_mol_nk",
            "secondary_cv_bias_kcal_mol_nk",
            "umbrella_bias_kcal_mol_nk",
            "umbrella_bias_kj_mol_nk",
            "umbrella_reduced_bias_nk",
        ],
        "gamd_setup_mode": "article_a_single_equilibrated_shared_gamd" if use_gamd else f"disabled_{run_mode}",
        "shared_gamd_calibration_steps": int(calib_steps),
        "shared_gamd_globals_json": str(out_dir / "shared_gamd_setup_globals.json"),
        "shared_gamd_context_checkpoint": str(out_dir / "shared_gamd_setup_context.chk"),
        "replica_shared_gamd_copy_report_json": str(out_dir / "replica_shared_gamd_copy_report.json"),
        "samples_written_only_for_phases": ["gareus_production", "gamd_production", "production"],
        "samples_note": "samples.csv is opened lazily after the single shared GaMD setup and contains only GaREUS production samples; calibration/equilibration rows are not written for MBAR." if use_gamd else f"samples.csv contains conventional umbrella/REUS production samples; GaMD boost fields are empty in --run-mode {run_mode}.",
        "samples_columns_for_mbar": {
            "cv_A": f"sampled primary CV ({primary_cv_label(args)}) in {primary_cv_units(args)}; legacy column name",
            "window": "thermodynamic umbrella state assigned to the replica at this sample",
            "umbrella_bias_all_windows_kcal_mol_json": "optional list of U_i(r_n) for all umbrella windows i, kcal/mol; written only with --write-full-bias-csv-vectors",
            "umbrella_bias_all_windows_kj_mol_json": "optional list of U_i(r_n) for all umbrella windows i, kJ/mol; written only with --write-full-bias-csv-vectors",
            "umbrella_reduced_bias_all_windows_json": "optional list of beta*U_i(r_n), dimensionless; written only with --write-full-bias-csv-vectors",
            "gamd_boost_total_kj_mol": "GaMD boost estimate. Preferred source is gamd-openmm integrator.get_boost_potentials(); fallback is named CustomIntegrator globals.",
            "gamd_boost_source": "get_boost_potentials, integrator_globals, or unavailable",
            "gamd_boost_components_kj_mol_json": "component boost potentials from gamd-openmm native get_boost_potentials(), kJ/mol",
        },
    }
    write_json(out_dir / "umbrella_pymbar_metadata.json", pymbar_metadata)
    print(f"PyMBAR umbrella constants written to {out_dir / 'umbrella_windows.csv'} and {out_dir / 'umbrella_pymbar_metadata.json'}")

    rng = np.random.default_rng(args.seed)

    _seg_registry = SegmentRegistry(out_dir)
    _run_id = str(out_dir.name)
    _parent_seg = _seg_registry.get_latest_segment()
    _parent_seg_id = _parent_seg["segment_id"] if _parent_seg else None
    _parent_was_running = bool(_parent_seg and _parent_seg.get("status") == "running")
    _round_id = int(getattr(args, "adaptive_feedback_round", 1))
    _seg_id = _seg_registry.open_segment(_run_id, _parent_seg_id, _round_id)
    _win_snapshot_windows = [
        {
            "window_id": int(wi),
            "center1": float(centers_a[wi]),
            "k1": float(k_list[wi]),
            **({"center2": float(secondary_cv_centers[wi]), "k2": float(secondary_cv_k_kcal_list[wi])}
               if secondary_cv_centers is not None and secondary_cv_k_kcal_list is not None else {}),
        }
        for wi in range(nrep)
    ]
    _cv2_type = (secondary_cv_metadata or {}).get("mode") if secondary_cv_centers is not None else None
    WindowSnapshot(out_dir).snapshot(_seg_id, _win_snapshot_windows, cv1_type=primary_cv_mode(args), cv2_type=_cv2_type)
    parquet_sample_writer = ParquetSampleWriter(
        out_dir / "samples" / _seg_id,
        flush_rows=int(getattr(args, "parquet_flush_rows", 5000) or 5000),
    )
    parquet_exchange_writer = ParquetExchangeWriter(
        out_dir / "exchanges" / _seg_id,
        flush_rows=int(getattr(args, "parquet_flush_rows", 5000) or 5000),
    )
    exchange_csv = None
    sample_csv = None
    sample_writer = None
    samples_started = False
    analysis_arrays_written = False
    distance_logger = DistanceLogger(out_dir, args, progress=progress, no_file_persistence=True)
    exchange_stats = {"attempts": 0, "accepted": 0, "pairs": {}, "jump_bins": {}, "mode": str(getattr(args, "exchange_mode", "neighbor")), "gibbs_choices": 0, "gibbs_moves": 0, "gibbs_stays": 0}
    dashboard_info = {
        "centers_a": [float(x) for x in centers_a],
        "primary_cv": primary_cv_mode(args),
        "primary_cv_label": primary_cv_label(args),
        "primary_cv_units": primary_cv_units(args),
        "primary_k_units": primary_k_units(args),
        "primary_cv_axis_min": float(getattr(args, "contact_adaptive_effective_min", getattr(args, "contact_adaptive_min", 0.0)) or 0.0) if primary_cv_is_contacts(args) else (float(np.nanmin(centers_a)) if len(centers_a) else 0.0),
        "primary_cv_axis_max": float(getattr(args, "contact_adaptive_effective_max", getattr(args, "contact_adaptive_max", float(np.nanmax(centers_a)) if len(centers_a) else 1.0)) or (float(np.nanmax(centers_a)) if len(centers_a) else 1.0)) if primary_cv_is_contacts(args) else (float(np.nanmax(centers_a)) if len(centers_a) else 1.0),
        "k_list": [float(x) for x in k_list],
        "n_windows": int(nrep),
        "exchange_stats": exchange_stats,
        "secondary_cv": _json_ready(secondary_cv_metadata or {}),
        "secondary_cv_centers": [float(x) for x in secondary_cv_centers] if secondary_cv_centers is not None else [],
        "secondary_cv_k_kcal_mol": [float(x) for x in secondary_cv_k_kcal_list] if secondary_cv_k_kcal_list is not None else [],
        "window_metadata": _json_ready(window_metadata or {}),
        "explicit_2d": bool((secondary_cv_metadata or {}).get("explicit_2d_windows", False)),
        "rectangular_2d": bool((secondary_cv_metadata or {}).get("grid", False)),
        "sparse_2d": bool((secondary_cv_metadata or {}).get("explicit_2d_windows", False) and not (secondary_cv_metadata or {}).get("grid", False)),
        "neighbor_graph": _json_ready((window_metadata or {}).get("explicit_2d_neighbor_graph", {})),
    }

    # Hot-path immutable NumPy arrays.  Keep these outside sample/exchange loops
    # so the production path does not rebuild identical arrays at every logging
    # or exchange interval.
    centers_a_arr = np.asarray(centers_a, dtype=np.float64)
    centers_nm_arr = np.asarray(centers_nm, dtype=np.float64)
    k_arr = np.asarray(k_list, dtype=np.float64)
    ks_arr = np.asarray(ks_kj_nm2, dtype=np.float64)
    if secondary_cv_centers is not None and secondary_cv_k_kcal_list is not None and secondary_cv_ks_kj is not None:
        ss_centers_arr_global = np.asarray(secondary_cv_centers, dtype=np.float64)
        ss_k_kcal_arr_global = np.asarray(secondary_cv_k_kcal_list, dtype=np.float64)
        ss_ks_kj_arr_global = np.asarray(secondary_cv_ks_kj, dtype=np.float64)
    else:
        ss_centers_arr_global = np.full(len(centers_a_arr), np.nan, dtype=np.float64)
        ss_k_kcal_arr_global = np.zeros(len(centers_a_arr), dtype=np.float64)
        ss_ks_kj_arr_global = None

    # Inverse assignment map for O(1) window -> replica lookup.  This is used by
    # neighbor, all-pair, random-pair, and Gibbs-walk exchange modes; the previous
    # assignments.index(window) lookup is small but sits in an inner loop.
    replica_of_window = np.full(nrep, -1, dtype=np.int32)

    def _refresh_replica_of_window() -> None:
        replica_of_window.fill(-1)
        for _replica, _window in enumerate(assignments):
            _w = int(_window)
            if 0 <= _w < nrep:
                replica_of_window[_w] = int(_replica)

    _refresh_replica_of_window()

    # Exchange-neighbor schedules depend only on window geometry, not on current
    # coordinates.  Build them once instead of rebuilding the same O(N^2) graph at
    # every exchange interval.  This matters most for sparse explicit 2D pilot
    # grids, where exchange intervals can be short and the graph is diagnostic but
    # static.
    explicit_2d_exchange_graph_cache = None
    if bool((secondary_cv_metadata or {}).get("explicit_2d_windows", False)) and secondary_cv_centers is not None:
        try:
            _graph_edges = build_explicit_2d_neighbor_edges(centers_a, secondary_cv_centers, args=args)
            _graph_slots = max(1, int(getattr(args, "explicit_2d_exchange_slots", 4) or 4))
            _pairs_by_slot: list[list[tuple[int, int, str, float]]] = []
            for _slot in range(_graph_slots):
                _chosen: list[tuple[int, int, str, float]] = []
                _used: set[int] = set()
                for _idx, _edge in enumerate(_graph_edges):
                    if _idx % _graph_slots != _slot:
                        continue
                    _wi, _wj = int(_edge["wi"]), int(_edge["wj"])
                    if _wi in _used or _wj in _used:
                        continue
                    _chosen.append((_wi, _wj, str(_edge.get("edge_type", "explicit_2d_graph")), float(_edge.get("normalized_distance", float("nan")))))
                    _used.add(_wi)
                    _used.add(_wj)
                if not _chosen and _graph_edges:
                    for _edge in _graph_edges:
                        _wi, _wj = int(_edge["wi"]), int(_edge["wj"])
                        if _wi in _used or _wj in _used:
                            continue
                        _chosen.append((_wi, _wj, str(_edge.get("edge_type", "explicit_2d_graph")), float(_edge.get("normalized_distance", float("nan")))))
                        _used.add(_wi)
                        _used.add(_wj)
                _pairs_by_slot.append(_chosen)
            explicit_2d_exchange_graph_cache = {
                "edges": _graph_edges,
                "slots": int(_graph_slots),
                "pairs_by_slot": _pairs_by_slot,
            }
        except Exception as exc:
            print(f"WARNING: could not precompute explicit 2D exchange graph; falling back to per-interval graph build: {exc}")
            explicit_2d_exchange_graph_cache = None

    n_primary_grid_cached = int((secondary_cv_metadata or {}).get("n_primary_windows", 0) or 0)
    n_secondary_grid_cached = int((secondary_cv_metadata or {}).get("n_secondary_centers", 0) or 0)
    grid_neighbor_pairs_by_parity = None
    if (secondary_cv_metadata or {}).get("grid") and n_primary_grid_cached > 0 and n_secondary_grid_cached > 1 and n_primary_grid_cached * n_secondary_grid_cached == nrep:
        grid_neighbor_pairs_by_parity = (
            _grid_neighbor_pairs_2d(n_primary_grid_cached, n_secondary_grid_cached, parity=0),
            _grid_neighbor_pairs_2d(n_primary_grid_cached, n_secondary_grid_cached, parity=1),
        )

    # Per-step observable cache.  If a distance/dashboard sample and an exchange
    # occur at the same production step, the sample path already paid for one
    # getState() per replica and constructed bias_matrix_kj.  Reuse it for the
    # following exchange instead of synchronizing all contexts again.
    observable_cache: dict[str, object] = {}

    # Stuck-replica rescue: detect replicas pinned at CV1≈0 (fully-extended dead zone where
    # the contact switching function gradient is ~0 and the umbrella provides no actual force).
    # After _stuck_max_intervals consecutive exchange intervals below _stuck_threshold, copy
    # positions from the nearest non-stuck replica and reinitialise velocities.
    _stuck_reseed_flag = getattr(args, "cv1_stuck_reseed", None)
    _stuck_enabled = bool(_stuck_reseed_flag) if _stuck_reseed_flag is not None else primary_cv_is_contacts(args)
    _stuck_threshold = float(getattr(args, "cv1_stuck_threshold", 0.03) or 0.03)
    _stuck_max_intervals = int(getattr(args, "cv1_stuck_detect_intervals", 500) or 500)
    _stuck_counter = np.zeros(nrep, dtype=int)
    _stuck_rescue_total = 0

    # Shared thread pool for parallel step_all() and getState() across replicas.
    # OpenMM releases the GIL during both context.step() and context.getState(),
    # so Python threads genuinely run in parallel.  One thread per replica is the
    # right fan-out; do not create a new pool each call.
    _sim_pool = ThreadPoolExecutor(max_workers=nrep)
    if str(getattr(args, "platform", "")).upper() == "CPU" and int(getattr(args, "cpu_threads", 1)) == 0 and nrep > 1:
        print(
            f"WARNING: --cpu-threads 0 (use all cores) combined with {nrep} parallel replicas "
            "will oversubscribe CPU cores. Set --cpu-threads 1 for parallel step_all()."
        )
    try:
        exchange_writer = parquet_exchange_writer

        def ensure_sample_writer():
            nonlocal samples_started
            if not samples_started:
                samples_started = True
                print(f"    Production samples writing to {out_dir / 'samples' / _seg_id}")
            return parquet_sample_writer

        def sample(step: int, phase: str) -> list[dict]:
            """Collect one reporting sample from every replica.

            The expensive part is the OpenMM State read, so this does exactly one
            getState(getPositions=True, getEnergy=<sample_potential_energy>) per replica, then computes
            all umbrella bias vectors as a single NumPy matrix:

                bias_matrix_kj[window, replica]

            The same matrix feeds samples.csv, analysis_arrays.npz, and dashboard
            rows without recomputing per-window JSON vectors in Python loops.
            """
            rows: list[dict] = []
            is_prod = is_gamd_production_phase(phase)
            primary_mode_name = primary_cv_mode(args)
            primary_label_name = primary_cv_label(args)
            primary_units_name = primary_cv_units(args)
            primary_k_units_name = primary_k_units(args)
            is_distance_primary = primary_mode_name == "distance"
            write_full_bias_vectors = bool(getattr(args, "write_full_bias_csv_vectors", False))
            write_gamd_globals = bool(getattr(args, "write_gamd_globals_json", False))
            read_sample_potential = bool(getattr(args, "sample_potential_energy", True))
            temperature_value = float(args.temperature_k)
            beta_value = float(beta)
            primary_values = np.empty(nrep, dtype=np.float64)
            ss_values = np.full(nrep, np.nan, dtype=np.float64)
            potentials_kj = np.empty(nrep, dtype=np.float64)

            def _fetch_state(r_sim):
                r, sim = r_sim
                if _use_fast_cv_path:
                    ctx = sim.context
                    pf = _fast_primary_forces[r]
                    raw = float(pf.getCollectiveVariableValues(ctx)[0])
                    norm = float(ctx.getParameter("contact_norm")) if bool(getattr(args, "contact_normalize", True)) else 0.0
                    cv = raw / norm if norm > 0.0 else raw
                    sf = _fast_ss_forces[r]
                    ss = (
                        _ss_scalar_from_sub_cv_values(sf.getCollectiveVariableValues(ctx), secondary_cv_metadata)
                        if sf is not None else float("nan")
                    )
                    if read_sample_potential:
                        state = ctx.getState(getEnergy=True, enforcePeriodicBox=True)
                        pe = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
                    else:
                        pe = float("nan")
                    return r, cv, ss, pe
                return r, *primary_secondary_and_potential_from_state(
                    sim.context, primary_cv_def, args, unit, secondary_cv_metadata,
                    read_potential_energy=read_sample_potential,
                )

            for r, cv, ss, pe in _sim_pool.map(_fetch_state, enumerate(sims)):
                primary_values[r], ss_values[r], potentials_kj[r] = cv, ss, pe
            primary_delta_matrix = primary_values[np.newaxis, :] - centers_a_arr[:, np.newaxis]
            distance_bias_matrix_kcal = 0.5 * k_arr[:, np.newaxis] * primary_delta_matrix * primary_delta_matrix
            if secondary_cv_centers is not None and secondary_cv_k_kcal_list is not None:
                ss_centers_arr = ss_centers_arr_global
                ss_k_arr = ss_k_kcal_arr_global
                ss_delta_matrix = ss_values[np.newaxis, :] - ss_centers_arr[:, np.newaxis]
                ss_bias_matrix_kcal = 0.5 * ss_k_arr[:, np.newaxis] * ss_delta_matrix * ss_delta_matrix
            else:
                ss_centers_arr = ss_centers_arr_global
                ss_k_arr = ss_k_kcal_arr_global
                ss_bias_matrix_kcal = np.zeros_like(distance_bias_matrix_kcal)
            bias_matrix_kcal = distance_bias_matrix_kcal + ss_bias_matrix_kcal
            bias_matrix_kj = 4.184 * bias_matrix_kcal
            reduced_bias_matrix = float(beta) * bias_matrix_kj
            observable_cache["step"] = int(step)
            observable_cache["primary_values"] = primary_values
            observable_cache["cvs_nm"] = (primary_values / 10.0) if is_distance_primary else primary_values
            observable_cache["bias_matrix_kj"] = bias_matrix_kj
            if is_prod:
                ensure_sample_writer()
            for r, sim in enumerate(sims):
                w = int(assignments[r])
                cv_a = float(primary_values[r])
                center_a = float(centers_a_arr[w])
                k_kcal_a2 = float(k_arr[w])
                all_bias_kcal = bias_matrix_kcal[:, r]
                all_bias_kj = bias_matrix_kj[:, r]
                all_reduced_bias = reduced_bias_matrix[:, r]
                all_distance_bias_kcal = distance_bias_matrix_kcal[:, r]
                all_ss_bias_kcal = ss_bias_matrix_kcal[:, r]
                sampled_bias_kcal = float(all_bias_kcal[w])
                sampled_bias_kj = float(all_bias_kj[w])
                row = {
                    "step": int(step),
                    "phase": phase,
                    "replica": r,
                    "window": w,
                    "center_A": center_a,
                    "k_kcal_mol_A2": k_kcal_a2,
                    "cv_A": cv_a,
                    "primary_cv": primary_mode_name,
                    "primary_cv_label": primary_label_name,
                    "primary_cv_units": primary_units_name,
                    "primary_cv_value": cv_a,
                    "primary_cv_center": center_a,
                    "primary_cv_k": k_kcal_a2,
                    "primary_cv_k_units": primary_k_units_name,
                    "primary_umbrella_bias_kcal_mol": float(all_distance_bias_kcal[w]),
                    "secondary_cv": float(ss_values[r]) if math.isfinite(float(ss_values[r])) else "",
                    "secondary_cv_center": float(ss_centers_arr[w]) if math.isfinite(float(ss_centers_arr[w])) else "",
                    "secondary_cv_k_kcal_mol": float(ss_k_arr[w]) if math.isfinite(float(ss_k_arr[w])) else "",
                    "distance_umbrella_bias_kcal_mol": float(all_distance_bias_kcal[w]),
                    "secondary_cv_bias_kcal_mol": float(all_ss_bias_kcal[w]),
                    "umbrella_bias_kcal_mol": sampled_bias_kcal,
                    "umbrella_bias_kj_mol": sampled_bias_kj,
                    "umbrella_reduced_bias": float(all_reduced_bias[w]),
                    "umbrella_restoring_force_kcal_mol_per_A": k_kcal_a2 * (center_a - cv_a),
                    "potential_kj_mol": float(potentials_kj[r]),
                    "temperature_K": temperature_value,
                    "beta_1_over_kJ_mol": beta_value,
                }
                if write_full_bias_vectors:
                    row.update({
                        "primary_umbrella_bias_all_windows_kcal_mol_json": json.dumps([float(x) for x in all_distance_bias_kcal]),
                        "distance_umbrella_bias_all_windows_kcal_mol_json": json.dumps([float(x) for x in all_distance_bias_kcal]),
                        "secondary_cv_bias_all_windows_kcal_mol_json": json.dumps([float(x) for x in all_ss_bias_kcal]),
                        "umbrella_bias_all_windows_kcal_mol_json": json.dumps([float(x) for x in all_bias_kcal]),
                        "umbrella_bias_all_windows_kj_mol_json": json.dumps([float(x) for x in all_bias_kj]),
                        "umbrella_reduced_bias_all_windows_json": json.dumps([float(x) for x in all_reduced_bias]),
                    })
                boost_kj, boost_components_kj, boost_source = extract_gamd_boost_kj(sim.integrator, unit, None)
                row["gamd_boost_total_kj_mol"] = boost_kj if boost_kj is not None else ""
                row["gamd_boost_total_kcal_mol"] = (boost_kj / 4.184) if boost_kj is not None else ""
                row["gamd_boost_source"] = boost_source
                row["gamd_boost_components_kj_mol_json"] = json.dumps(boost_components_kj, sort_keys=True)
                row["gamd_globals_json"] = (
                    json.dumps(integrator_globals(sim.integrator, unit=unit), sort_keys=True)
                    if write_gamd_globals else ""
                )
                if is_prod:
                    _boost_raw = row.get("gamd_boost_total_kj_mol")
                    _boost_total = float(_boost_raw) if _boost_raw not in (None, "") else None
                    _boost_comps = json.loads(row.get("gamd_boost_components_kj_mol_json", "{}") or "{}")
                    _, _boost_dihe, _boost_nonb = parse_gamd_boost_components(_boost_total, _boost_comps)
                    _ss_val = float(ss_values[r])
                    parquet_sample_writer.write_sample(
                        step=int(step),
                        replica=r,
                        window_id=w,
                        cv1=cv_a,
                        cv2=_ss_val if math.isfinite(_ss_val) else None,
                        potential=float(potentials_kj[r]),
                        boost_total=_boost_total,
                        boost_dihedral=_boost_dihe,
                        boost_nonbonded=_boost_nonb,
                    )
                rows.append(row)
            if is_prod and bool(getattr(args, "flush_every_log", True)):
                parquet_sample_writer.flush()
            return rows

        def flush_scalar_writers() -> None:
            """Flush open writers before checkpoints/final reports."""
            parquet_sample_writer.flush()
            parquet_exchange_writer.flush()

        def _write_production_nan_diagnostics(exc: Exception, requested_steps: int, completed_substeps: int = 0) -> Path:
            """Write targeted diagnostics when production stepping creates NaNs.

            Reporter exceptions often hide which replica actually exploded.  This
            scan runs after a failed production chunk and records per-replica
            finiteness, potential energy, assignment/window, and a crash PDB for
            any replica whose coordinates can still be serialized.
            """
            diag_step = int(prod_done) + int(completed_substeps)
            diag_path = Path(out_dir) / f"PRODUCTION_NAN_DIAGNOSTICS_prodstep_{diag_step:09d}.json"
            rows_diag = []
            for rr, sim in enumerate(sims):
                row = {
                    "replica": int(rr),
                    "window": int(assignments[rr]) if rr < len(assignments) else None,
                    "production_step": int(diag_step),
                    "requested_chunk_steps": int(requested_steps),
                    "finite_positions": False,
                    "state_read_ok": False,
                }
                try:
                    st = sim.context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True)
                    row["state_read_ok"] = True
                    try:
                        pos_nm = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                        finite = bool(np.isfinite(np.asarray(pos_nm, dtype=float)).all())
                        row["finite_positions"] = finite
                        row["n_nonfinite_values"] = int(np.size(pos_nm) - np.isfinite(np.asarray(pos_nm, dtype=float)).sum())
                    except Exception as pos_exc:
                        row["position_check_error"] = str(pos_exc)
                        finite = False
                    try:
                        row["potential_kj_mol"] = float(st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
                    except Exception as e_pe:
                        row["potential_error"] = str(e_pe)
                    try:
                        crash_pdb = Path(out_dir) / f"CRASH_production_replica_{rr:03d}_prodstep_{diag_step:09d}.pdb"
                        write_state_pdb(crash_pdb, app, topology, st.getPositions())
                        row["crash_pdb"] = str(crash_pdb)
                    except Exception as write_exc:
                        row["crash_pdb_error"] = str(write_exc)
                except Exception as state_exc:
                    row["state_error"] = str(state_exc)
                rows_diag.append(row)
            payload = {
                "event": "production_nan_or_step_failure",
                "primary_cv": primary_cv_mode(args),
                "primary_cv_label": primary_cv_label(args),
                "production_step": int(diag_step),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "requested_chunk_steps": int(requested_steps),
                "completed_substeps_before_failure": int(completed_substeps),
                "timestep_fs": float(getattr(args, "timestep_fs", 0.0) or 0.0),
                "traj_interval": int(getattr(args, "traj_interval", 0) or 0),
                "traj_format": str(getattr(args, "traj_format", "")),
                "replicas": rows_diag,
                "hint": (
                    "For contact-primary pilots, first try timestep_fs 2.0, traj_format none/traj_interval 0, "
                    "lower contact_adaptive_k_scale, lower secondary_cv_adaptive_max_k_kcal, and softer contact_beta_a_inv. "
                    "The pilot trajectories are diagnostic and usually do not need coordinate reporters."
                ),
            }
            try:
                write_json(diag_path, payload)
            except Exception:
                pass
            print(f"WARNING: production stepping failed; wrote diagnostics to {diag_path}")
            return diag_path

        def step_all(nsteps: int):
            nsteps = int(nsteps)
            if nsteps <= 0:
                return
            safe_chunk = int(getattr(args, "production_safe_chunk_steps", 0) or 0)

            def _step_item(item):
                _idx, _sim, _n = item
                _sim.step(int(_n))
                return int(_idx)

            completed = 0
            try:
                if safe_chunk <= 0 or safe_chunk >= nsteps:
                    list(_sim_pool.map(_step_item, [(i, s, nsteps) for i, s in enumerate(sims)]))
                else:
                    remaining = nsteps
                    while remaining > 0:
                        sub = min(int(safe_chunk), int(remaining))
                        list(_sim_pool.map(_step_item, [(i, s, sub) for i, s in enumerate(sims)]))
                        completed += sub
                        remaining -= sub
            except Exception as exc:
                diag_path = None
                if bool(getattr(args, "production_nan_diagnostics", True)):
                    diag_path = _write_production_nan_diagnostics(exc, nsteps, completed_substeps=completed)
                msg = (
                    f"Production stepping failed after {completed}/{nsteps} requested chunk steps: {exc}. "
                    f"Diagnostics: {diag_path if diag_path is not None else 'disabled'}. "
                    "This usually means one replica became unstable; for contact-primary adaptive pilots, use a smaller timestep, lower CV k values, and disable pilot trajectories."
                )
                raise RuntimeError(msg) from exc

        def _exchange_probability(delta_kj: float) -> float:
            """Metropolis probability for a proposed umbrella-window swap."""
            try:
                delta_kj = float(delta_kj)
            except Exception:
                return 0.0
            if not math.isfinite(delta_kj):
                return 0.0
            if delta_kj <= 0.0:
                return 1.0
            x = -float(beta) * delta_kj
            if x < -745.0:
                return 0.0
            return float(math.exp(x))

        def _record_exchange_stats(wi: int, wj: int, accepted: bool) -> None:
            wi = int(wi)
            wj = int(wj)
            pair_key = f"{min(wi, wj)}-{max(wi, wj)}"
            gap = abs(wi - wj)
            jump_key = f"dw{gap}"
            pair_stats = exchange_stats.setdefault("pairs", {}).setdefault(pair_key, {"attempts": 0, "accepted": 0})
            pair_stats["attempts"] = int(pair_stats.get("attempts", 0) or 0) + 1
            jump_stats = exchange_stats.setdefault("jump_bins", {}).setdefault(jump_key, {"attempts": 0, "accepted": 0})
            jump_stats["attempts"] = int(jump_stats.get("attempts", 0) or 0) + 1
            exchange_stats["attempts"] = int(exchange_stats.get("attempts", 0) or 0) + 1
            if bool(accepted):
                pair_stats["accepted"] = int(pair_stats.get("accepted", 0) or 0) + 1
                jump_stats["accepted"] = int(jump_stats.get("accepted", 0) or 0) + 1
                exchange_stats["accepted"] = int(exchange_stats.get("accepted", 0) or 0) + 1

        def _replica_holding_window(window_index: int) -> Optional[int]:
            try:
                w = int(window_index)
            except Exception:
                return None
            if w < 0 or w >= int(replica_of_window.size):
                return None
            rep = int(replica_of_window[w])
            return rep if rep >= 0 else None

        def _attempt_window_swap(wi: int, wj: int, primary_values: np.ndarray, bias_matrix_kj: np.ndarray, absolute_step: int, attempt: int, p_override: Optional[float] = None, force_accept: bool = False) -> int:
            """Attempt/apply a swap between two umbrella windows using vectorized bias energies."""
            wi = int(wi)
            wj = int(wj)
            if wi == wj:
                return attempt
            i = _replica_holding_window(wi)
            j = _replica_holding_window(wj)
            if i is None or j is None or i == j:
                return attempt
            # bias_matrix_kj[window, replica] = U_window(x_replica)
            old_e = float(bias_matrix_kj[wi, i] + bias_matrix_kj[wj, j])
            new_e = float(bias_matrix_kj[wj, i] + bias_matrix_kj[wi, j])
            delta = float(new_e - old_e)
            pacc = float(p_override) if p_override is not None else _exchange_probability(delta)
            pacc = max(0.0, min(1.0, pacc)) if math.isfinite(pacc) else 0.0
            accepted = bool(force_accept) or (rng.random() < pacc)
            _record_exchange_stats(wi, wj, accepted)
            if accepted:
                assignments[i], assignments[j] = assignments[j], assignments[i]
                replica_of_window[int(assignments[i])] = int(i)
                replica_of_window[int(assignments[j])] = int(j)
                set_window(sims[i].context, centers_nm, ks_kj_nm2, assignments[i], secondary_cv_centers, secondary_cv_ks_kj)
                set_window(sims[j].context, centers_nm, ks_kj_nm2, assignments[j], secondary_cv_centers, secondary_cv_ks_kj)
            parquet_exchange_writer.write_exchange(
                step=int(absolute_step),
                replica_i=int(i),
                replica_j=int(j),
                window_i=int(wi),
                window_j=int(wj),
                delta_e=float(delta),
                accepted=bool(accepted),
            )
            return attempt + 1

        def _current_exchange_arrays(absolute_step: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
            # Prefer the sample/log observables from the same production step.
            # This avoids a second getState() pass when next_log and next_exchange
            # coincide, while keeping the exchange energy exactly the same matrix
            # used for samples.csv / analysis_arrays.npz.
            if absolute_step is not None and int(observable_cache.get("step", -1)) == int(absolute_step):
                cached_cvs = observable_cache.get("primary_values")
                cached_bias = observable_cache.get("bias_matrix_kj")
                if isinstance(cached_cvs, np.ndarray) and isinstance(cached_bias, np.ndarray):
                    if cached_cvs.shape == (nrep,) and cached_bias.shape == (nrep, nrep):
                        return cached_cvs, cached_bias

            # Compute CV values for every replica.  When the fast path is active
            # (contacts primary + CustomCVForce secondary) this reads cached scalars
            # via getCollectiveVariableValues() — no positions DMA at all.  The
            # full U[window, replica] umbrella matrix is then built in NumPy for all
            # swap/Gibbs candidates.  Potential energies are not needed here.
            primary_values = np.empty(nrep, dtype=np.float64)
            ss_values = np.full(nrep, np.nan, dtype=np.float64)
            _ss_enabled = bool((secondary_cv_metadata or {}).get("enabled"))

            def _fetch_exchange_state(r_sim):
                r, sim = r_sim
                if _use_fast_cv_path:
                    ctx = sim.context
                    pf = _fast_primary_forces[r]
                    raw = float(pf.getCollectiveVariableValues(ctx)[0])
                    norm = float(ctx.getParameter("contact_norm")) if bool(getattr(args, "contact_normalize", True)) else 0.0
                    cv = raw / norm if norm > 0.0 else raw
                    sf = _fast_ss_forces[r]
                    ss = (
                        _ss_scalar_from_sub_cv_values(sf.getCollectiveVariableValues(ctx), secondary_cv_metadata)
                        if sf is not None and _ss_enabled else float("nan")
                    )
                else:
                    state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
                    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                    cv = primary_cv_value_from_positions_nm(pos, primary_cv_def, args)
                    ss = secondary_structure_score_from_positions_nm(pos, secondary_cv_metadata) if _ss_enabled else float("nan")
                return r, cv, ss

            for r, cv, ss in _sim_pool.map(_fetch_exchange_state, enumerate(sims)):
                primary_values[r] = cv
                ss_values[r] = ss
            dprimary = primary_values[np.newaxis, :] - centers_a_arr[:, np.newaxis]
            bias_matrix_kj = 4.184 * 0.5 * k_arr[:, np.newaxis] * dprimary * dprimary
            if secondary_cv_centers is not None and secondary_cv_ks_kj is not None and ss_ks_kj_arr_global is not None:
                dss = ss_values[np.newaxis, :] - ss_centers_arr_global[:, np.newaxis]
                bias_matrix_kj = bias_matrix_kj + 0.5 * ss_ks_kj_arr_global[:, np.newaxis] * dss * dss
            return primary_values, bias_matrix_kj

        def _candidate_delta_for_window_swap(wi: int, wj: int, bias_matrix_kj: np.ndarray) -> Optional[float]:
            wi = int(wi)
            wj = int(wj)
            if wi == wj:
                return 0.0
            i = _replica_holding_window(wi)
            j = _replica_holding_window(wj)
            if i is None or j is None or i == j:
                return None
            old_e = float(bias_matrix_kj[wi, i] + bias_matrix_kj[wj, j])
            new_e = float(bias_matrix_kj[wj, i] + bias_matrix_kj[wi, j])
            return float(new_e - old_e)

        def attempt_exchanges(absolute_step: int, parity: int, attempt: int) -> tuple[int, int]:
            mode = str(getattr(args, "exchange_mode", "neighbor"))
            exchange_stats["mode"] = mode
            primary_values, bias_matrix_kj = _current_exchange_arrays(absolute_step)
            windows = list(range(nrep))

            if mode == "neighbor":
                # Standard REUS/HREX ladder.  For a 2D distance x secondary-CV
                # grid, use true nearest-neighbor edges on the rectangle.  For an
                # explicit sparse 2D table, use a geometry graph so local patch
                # windows exchange with nearby states instead of a flattened list.
                n_primary_grid = n_primary_grid_cached
                n_secondary_grid = n_secondary_grid_cached
                if (secondary_cv_metadata or {}).get("explicit_2d_windows") and secondary_cv_centers is not None:
                    if explicit_2d_exchange_graph_cache is not None:
                        graph_edges = explicit_2d_exchange_graph_cache["edges"]
                        graph_slots = int(explicit_2d_exchange_graph_cache["slots"])
                        graph_pairs = explicit_2d_exchange_graph_cache["pairs_by_slot"][int(parity) % max(1, graph_slots)]
                    else:
                        graph_pairs, graph_slots, graph_edges = explicit_2d_neighbor_pairs_for_exchange(
                            centers_a, secondary_cv_centers, parity=parity, args=args
                        )
                    exchange_stats["mode"] = "neighbor-graph-explicit-2d"
                    exchange_stats["neighbor_graph"] = {
                        "source": "explicit_2d_window_geometry",
                        "edge_count": int(len(graph_edges)),
                        "attempted_edges_this_interval": int(len(graph_pairs)),
                        "slots": int(graph_slots),
                        "slot": int(parity) % max(1, int(graph_slots)),
                    }
                    pair_meta = exchange_stats.setdefault("pair_metadata", {})
                    for wi, wj, edge_type, ndist in graph_pairs:
                        key = f"{min(int(wi), int(wj))}-{max(int(wi), int(wj))}"
                        pair_meta.setdefault(key, {
                            "edge_type": str(edge_type),
                            "normalized_distance": float(ndist) if math.isfinite(float(ndist)) else None,
                            "neighbor_source": "explicit_2d_geometry_graph",
                        })
                        attempt = _attempt_window_swap(wi, wj, primary_values, bias_matrix_kj, absolute_step, attempt)
                    parquet_exchange_writer.flush() if bool(getattr(args, "flush_every_log", True)) else None
                    return (int(parity) + 1) % max(1, int(graph_slots)), attempt
                if grid_neighbor_pairs_by_parity is not None:
                    pairs = grid_neighbor_pairs_by_parity[int(parity) & 1]
                else:
                    pairs = [(w, w + 1) for w in range(parity, nrep - 1, 2)]
                for wi, wj in pairs:
                    attempt = _attempt_window_swap(wi, wj, primary_values, bias_matrix_kj, absolute_step, attempt)
                parquet_exchange_writer.flush() if bool(getattr(args, "flush_every_log", True)) else None
                return 1 - parity, attempt

            if mode == "random-pair":
                rng.shuffle(windows)
                requested = int(getattr(args, "exchange_random_pairs", 0) or 0)
                max_pairs = max(1, nrep // 2) if requested <= 0 else max(1, requested)
                pairs = [(windows[k], windows[k + 1]) for k in range(0, len(windows) - 1, 2)][:max_pairs]
                for wi, wj in pairs:
                    attempt = _attempt_window_swap(wi, wj, primary_values, bias_matrix_kj, absolute_step, attempt)
                parquet_exchange_writer.flush() if bool(getattr(args, "flush_every_log", True)) else None
                return parity, attempt

            if mode == "all-pair-sweep":
                pairs = [(i, j) for i in range(nrep) for j in range(i + 1, nrep)]
                rng.shuffle(pairs)
                limit = int(getattr(args, "exchange_max_pairs_per_interval", 0) or 0)
                if limit > 0:
                    pairs = pairs[:limit]
                for wi, wj in pairs:
                    attempt = _attempt_window_swap(wi, wj, primary_values, bias_matrix_kj, absolute_step, attempt)
                parquet_exchange_writer.flush() if bool(getattr(args, "flush_every_log", True)) else None
                return parity, attempt

            if mode == "gibbs-walk":
                # Heat-bath-like long-jump proposal with MH correction.  The
                # proposal is nonuniform, so selected moves cannot be accepted
                # unconditionally without biasing the permutation chain.
                order = list(range(nrep))
                rng.shuffle(order)
                limit = int(getattr(args, "exchange_max_pairs_per_interval", 0) or 0)
                if limit > 0:
                    order = order[:limit]
                for rep in order:
                    rep = int(rep)
                    wi = int(assignments[rep])
                    proposal = _gibbs_window_proposal_distribution(
                        beta=float(beta),
                        bias_matrix_kj=bias_matrix_kj,
                        replica_index=rep,
                        current_window=wi,
                        replica_of_window=replica_of_window,
                    )
                    valid_windows = proposal["windows"]
                    probs = proposal["probabilities"]
                    deltas = proposal["deltas_kj"]
                    if valid_windows.size <= 0 or probs.size <= 0:
                        exchange_stats["gibbs_all_nan_skips"] = exchange_stats.get("gibbs_all_nan_skips", 0) + 1
                        continue
                    choice_index = int(rng.choice(valid_windows.size, p=probs))
                    wj = int(valid_windows[choice_index])
                    exchange_stats["gibbs_choices"] = int(exchange_stats.get("gibbs_choices", 0) or 0) + 1
                    if wj == wi:
                        exchange_stats["gibbs_stays"] = int(exchange_stats.get("gibbs_stays", 0) or 0) + 1
                        continue
                    exchange_stats["gibbs_moves"] = int(exchange_stats.get("gibbs_moves", 0) or 0) + 1
                    q_forward = float(probs[choice_index])
                    holders_after = replica_of_window.astype(np.int64, copy=True)
                    target_rep = int(holders_after[wj])
                    holders_after[wi] = target_rep
                    holders_after[wj] = rep
                    reverse = _gibbs_window_proposal_distribution(
                        beta=float(beta),
                        bias_matrix_kj=bias_matrix_kj,
                        replica_index=rep,
                        current_window=wj,
                        replica_of_window=holders_after,
                    )
                    rev_windows = reverse["windows"]
                    rev_probs = reverse["probabilities"]
                    rev_idx = np.where(rev_windows == wi)[0]
                    q_reverse = float(rev_probs[int(rev_idx[0])]) if rev_idx.size else 0.0
                    pacc = _gibbs_mh_acceptance_probability(
                        delta_kj=float(deltas[choice_index]),
                        beta=float(beta),
                        q_forward=q_forward,
                        q_reverse=q_reverse,
                    )
                    exchange_stats["gibbs_mh_corrected"] = True
                    attempt = _attempt_window_swap(wi, wj, primary_values, bias_matrix_kj, absolute_step, attempt, p_override=pacc)
                parquet_exchange_writer.flush() if bool(getattr(args, "flush_every_log", True)) else None
                return parity, attempt

            raise ValueError(f"Unknown --exchange-mode {mode!r}")

        distance_interval = int(getattr(args, "distance_output_interval", 0) or 0)
        if distance_interval <= 0:
            distance_interval = max(1, min(int(args.report_interval), int(args.exchange_interval)))
        distance_interval = max(1, distance_interval)

        # Article-style GaREUS: the GaMD setup was already performed once above and
        # copied into every replica.  Do not run per-replica calibration/equilibration
        # here; all replicas start directly in the GaMD production stage.
        if use_gamd:
            print(f"    Shared GaMD setup already completed once ({calib_steps} steps); replicas start directly in production")
        else:
            print("    Conventional MD mode: no GaMD setup/calibration; replicas start directly in umbrella/REUS production")

        prod_total = int(args.gamd_production_steps)
        print(f"    {'GaREUS' if use_gamd else 'US/REUS CMD'} production with {getattr(args, 'exchange_mode', 'neighbor')} exchanges: {prod_total} steps")
        prod_done = 0
        attempt = 0
        parity = 0
        next_exchange = int(args.exchange_interval) if int(args.exchange_interval) > 0 else prod_total
        next_log = distance_interval

        # Dashboard ETA should describe the current production segment, not the
        # previously completed shared-GaMD setup/calibration bookkeeping.
        production_eta_start_wall = time.time()
        dashboard_info["display_total_steps"] = int(prod_total)
        dashboard_info["eta_start_wall"] = float(production_eta_start_wall)
        adaptive_phase_info = None
        if bool(getattr(args, "adaptive_feedback_pilot", False)):
            adaptive_phase_info = {
                "is_pilot": True,
                "round": int(getattr(args, "adaptive_feedback_round_index", 0) or 0),
                "rounds": int(getattr(args, "adaptive_feedback_rounds_total", 0) or 0),
                "target_overlap": float(getattr(args, "adaptive_feedback_target_overlap", 0.30) or 0.30),
                "final_steps": int(getattr(args, "adaptive_feedback_final_steps", prod_total) or prod_total),
                "workflow_start_wall": float(getattr(args, "adaptive_feedback_workflow_start_wall", production_eta_start_wall) or production_eta_start_wall),
                "workflow_done_before": int(getattr(args, "adaptive_feedback_workflow_done_before", 0) or 0),
                "workflow_total_steps": int(getattr(args, "adaptive_feedback_workflow_total_steps", prod_total) or prod_total),
                "discard_after_diagnostics": True,
                "prev_rounds": list(getattr(args, "adaptive_feedback_prev_rounds", []) or []),
            }
        elif bool(getattr(args, "adaptive_feedback_final_production", False)):
            adaptive_phase_info = {
                "is_final": True,
                "rounds": int(getattr(args, "adaptive_feedback_rounds_total", 0) or 0),
                "target_overlap": float(getattr(args, "adaptive_feedback_target_overlap", 0.30) or 0.30),
                "prev_rounds": list(getattr(args, "adaptive_feedback_prev_rounds", []) or []),
            }
        if adaptive_phase_info is not None:
            dashboard_info["adaptive_phase"] = adaptive_phase_info
        # Merge adaptive-production epoch/topup context from _adaptive_phase_info if present.
        _api = getattr(args, "_adaptive_phase_info", {}) or {}
        if _api:
            if adaptive_phase_info is None:
                adaptive_phase_info = {
                    "is_adaptive_epoch": True,
                    "epoch_index": int(_api.get("epoch_index", 0)),
                    "epoch_total": _api.get("epoch_total", 1) if _api.get("epoch_total", 1) == "?" else int(_api.get("epoch_total", 1)),
                    "segment_name": str(_api.get("segment_name", "")),
                    "is_topup": bool(_api.get("is_topup", False)),
                    "topup_index": int(_api.get("topup_index", 0)),
                    "prev_epochs": list(_api.get("prev_epochs", [])),
                }
                dashboard_info["adaptive_phase"] = adaptive_phase_info
            else:
                adaptive_phase_info["epoch_index"] = int(_api.get("epoch_index", 0))
                _et = _api.get("epoch_total", 1)
                adaptive_phase_info["epoch_total"] = _et if _et == "?" else int(_et)

        if bool(getattr(args, "resume", False)):
            manifest = load_production_checkpoint(out_dir, sims, centers_nm, ks_kj_nm2, rng, secondary_cv_centers, secondary_cv_ks_kj)
            if manifest is not None:
                assignments[:] = [int(x) for x in manifest.get("assignments", assignments)]
                _refresh_replica_of_window()
                prod_done = int(manifest.get("prod_done", 0))
                attempt = int(manifest.get("attempt", 0))
                parity = int(manifest.get("parity", 0))
                next_exchange = int(manifest.get("next_exchange", next_exchange))
                next_log = int(manifest.get("next_log", next_log))
                exchange_stats.clear()
                exchange_stats.update(manifest.get("exchange_stats", {"attempts": 0, "accepted": 0, "pairs": {}}))
                exchange_stats.setdefault("pairs", {})
                exchange_stats.setdefault("jump_bins", {})
                exchange_stats.setdefault("gibbs_choices", 0)
                exchange_stats.setdefault("gibbs_moves", 0)
                exchange_stats.setdefault("gibbs_stays", 0)
                exchange_stats["mode"] = str(getattr(args, "exchange_mode", exchange_stats.get("mode", "neighbor")))
                resume_tui_report = distance_logger.restore_from_previous_outputs(n_windows=nrep, max_step=int(manifest.get("absolute_step", 0) or 0) or None)
                if resume_tui_report.get("restored"):
                    dashboard_info["resume_tui_history"] = resume_tui_report
                    print(
                        "    Restored TUI/dashboard history from "
                        f"{resume_tui_report.get('source')} "
                        f"({resume_tui_report.get('rows')} rows; "
                        f"{resume_tui_report.get('window_histories')} window histories)."
                    )
                else:
                    fallback_exchange_report = restore_exchange_stats_from_csv_if_needed(out_dir, args, exchange_stats, max_step=int(manifest.get("absolute_step", 0) or 0) or None)
                    if fallback_exchange_report.get("restored"):
                        print(f"    Restored exchange dashboard stats from {fallback_exchange_report.get('source')} ({fallback_exchange_report.get('rows')} rows).")
                    print(f"    TUI/dashboard history not restored: {resume_tui_report.get('reason', 'unknown reason')}")
                print(f"    Resumed GaREUS production from checkpoint at production step {prod_done}/{prod_total}; attempt {attempt}")
                if effective_traj_interval > 0:
                    for i, sim in enumerate(sims):
                        reporter = make_trajectory_reporter(app, traj_dir / f"replica_{i:03d}_resume_from_{prod_done:09d}", effective_traj_interval, args, atom_subset=traj_atom_subset)
                        if reporter is not None:
                            sim.reporters.append(reporter)
                # Seal the previous crashed segment: rows beyond the checkpoint
                # step have wrong window_id labels (exchange state was rolled back)
                # and must be excluded from MBAR analysis.
                if _parent_was_running and _parent_seg_id is not None:
                    _seg_registry.seal_segment(
                        _parent_seg_id,
                        absolute_end_step=int(manifest.get("absolute_step", 0)),
                        status="interrupted",
                    )
                # Record the absolute start step of the resumed segment.
                _seg_registry.set_segment_start_step(_seg_id, int(manifest.get("absolute_step", 0)))
            else:
                print("    --resume requested, but no production checkpoint manifest was found; starting production from step 0")
                # No checkpoint: any previously-running segment has no valid
                # end boundary — mark it abandoned so it is skipped in analysis.
                if _parent_was_running and _parent_seg_id is not None:
                    _seg_registry.seal_segment(_parent_seg_id, absolute_end_step=-1, status="abandoned")
                if effective_traj_interval > 0:
                    for i, sim in enumerate(sims):
                        reporter = make_trajectory_reporter(app, traj_dir / f"replica_{i:03d}_resume_fresh_{int(time.time())}", effective_traj_interval, args, atom_subset=traj_atom_subset)
                        if reporter is not None:
                            sim.reporters.append(reporter)
        elif _parent_was_running and _parent_seg_id is not None:
            # Fresh start (no --resume) while a previous segment is still marked
            # running: the prior run was abandoned without a checkpoint.
            _seg_registry.seal_segment(_parent_seg_id, absolute_end_step=-1, status="abandoned")

        if prod_done <= 0:
            run_production_probe(args, out_dir, sims, assignments, centers_nm, ks_kj_nm2, primary_cv_def, cv_atom1, cv_atom2, unit, shared_gamd_globals_all)

        checkpoint_interval = int(getattr(args, "checkpoint_interval", 0) or 0)
        next_checkpoint = prod_done + checkpoint_interval if checkpoint_interval > 0 else prod_total + 1

        if progress is not None:
            progress.progress("gareus_production", prod_done, prod_total, message=f"{nrep} replicas | {getattr(args, 'exchange_mode', 'neighbor')} exchange attempts {attempt}", timestep_fs=float(args.timestep_fs), n_replicas=nrep, force=True)

        while prod_done < prod_total:
            if _graceful_shutdown.is_set():
                print(f"Graceful shutdown: saving checkpoint at step {prod_done}/{prod_total} and exiting.", flush=True)
                flush_scalar_writers()
                save_production_checkpoint(
                    out_dir, sims, assignments, prod_done, absolute_step,
                    parity, attempt, next_exchange, next_log, exchange_stats, rng,
                    centers_a=centers_a, k_list=k_list, cv_atom1=cv_atom1, cv_atom2=cv_atom2,
                    cv_label=cv_label, calib_steps=calib_steps,
                    secondary_cv_metadata=secondary_cv_metadata,
                    secondary_cv_centers=secondary_cv_centers,
                    secondary_cv_k_kcal_list=secondary_cv_k_kcal_list,
                    primary_cv_metadata=_json_ready(primary_cv_def),
                )
                break
            target = prod_total
            if next_exchange > prod_done:
                target = min(target, next_exchange)
            if next_log > prod_done:
                target = min(target, next_log)
            if checkpoint_interval > 0 and next_checkpoint > prod_done:
                target = min(target, next_checkpoint)
            chunk = max(1, target - prod_done)
            step_all(chunk)
            prod_done += chunk
            absolute_step = calib_steps + prod_done
            summary = {}

            if prod_done >= next_log or prod_done >= prod_total:
                rows = sample(absolute_step, "gareus_production")
                dashboard_info["display_step"] = int(prod_done)
                dashboard_info["display_total_steps"] = int(prod_total)
                summary = distance_logger.log(rows, "gareus_production", absolute_step, calib_steps + prod_total, dashboard_info=dashboard_info)
                while next_log <= prod_done:
                    next_log += distance_interval

            if prod_done >= next_exchange and prod_done < prod_total + 1:
                parity, attempt = attempt_exchanges(absolute_step, parity, attempt)
                while next_exchange <= prod_done:
                    next_exchange += int(args.exchange_interval) if int(args.exchange_interval) > 0 else prod_total + 1
                # Stuck-replica rescue: replicas pinned at CV1≈0 have near-zero umbrella gradient
                # (contact switching function decays to 0 when all pairs >> r0) so GaMD alone
                # can't drive escape.  Copy positions from the closest non-stuck replica and
                # reinitialise velocities to break the dead zone.
                if _stuck_enabled:
                    _pv = observable_cache.get("primary_values")
                    if isinstance(_pv, np.ndarray) and len(_pv) == nrep:
                        for _r in range(nrep):
                            if float(_pv[_r]) < _stuck_threshold:
                                _stuck_counter[_r] += 1
                            else:
                                _stuck_counter[_r] = 0
                            if _stuck_counter[_r] >= _stuck_max_intervals:
                                _w = int(assignments[_r])
                                _target_cv = float(centers_a[_w])
                                _dists = np.abs(_pv.astype(float) - _target_cv)
                                _dists[_r] = 1e9
                                _src = int(np.argmin(_dists))
                                if float(_pv[_src]) > _stuck_threshold:
                                    _src_state = sims[_src].context.getState(
                                        getPositions=True, enforcePeriodicBox=True
                                    )
                                    sims[_r].context.setPositions(_src_state.getPositions())
                                    sims[_r].context.setVelocitiesToTemperature(
                                        float(args.temperature_k) * unit.kelvin,
                                        int(args.seed) + _r + int(absolute_step % 99991),
                                    )
                                    _stuck_counter[_r] = 0
                                    _stuck_rescue_total += 1
                                    print(
                                        f"STUCK-RESCUE step {absolute_step}: replica {_r} window {_w} "
                                        f"(target CV1={_target_cv:.3f}) stuck at CV1={float(_pv[_r]):.3f} "
                                        f"for {_stuck_max_intervals} intervals; "
                                        f"positions from replica {_src} (CV1={float(_pv[_src]):.3f}); "
                                        f"rescue #{_stuck_rescue_total}",
                                        flush=True,
                                    )

            if checkpoint_interval > 0 and (prod_done >= next_checkpoint or prod_done >= prod_total):
                flush_scalar_writers()
                save_production_checkpoint(
                    out_dir, sims, assignments, prod_done, absolute_step, parity, attempt,
                    next_exchange, next_log, exchange_stats, rng,
                    centers_a=centers_a, k_list=k_list, cv_atom1=cv_atom1, cv_atom2=cv_atom2,
                    cv_label=cv_label, calib_steps=calib_steps,
                    secondary_cv_metadata=secondary_cv_metadata,
                    secondary_cv_centers=secondary_cv_centers,
                    secondary_cv_k_kcal_list=secondary_cv_k_kcal_list,
                    primary_cv_metadata=_json_ready(primary_cv_def),
                )
                _scratch_main = getattr(args, "_main_dir", None)
                if _scratch_main:
                    sync_scratch_to_main(out_dir, Path(_scratch_main))
                while next_checkpoint <= prod_done:
                    next_checkpoint += checkpoint_interval

            if progress is not None:
                msg = f"{nrep} replicas | {getattr(args, 'exchange_mode', 'neighbor')} exchange attempts {attempt}"
                if summary:
                    _unit = primary_cv_units(args)
                    _unit_suffix = "" if _unit in {"", "dimensionless"} else f" {_unit}"
                    msg += f" | cv {summary.get('cv_min_A', float('nan')):.2f}-{summary.get('cv_max_A', float('nan')):.2f}{_unit_suffix} mean {summary.get('cv_mean_A', float('nan')):.2f}"
                    if "k_min_kcal_mol_A2" in summary:
                        msg += f" | k {summary.get('k_min_kcal_mol_A2', float('nan')):.3f}-{summary.get('k_max_kcal_mol_A2', float('nan')):.3f}"
                    if "umbrella_bias_mean_kcal_mol" in summary:
                        msg += f" | Ubias mean/max {summary.get('umbrella_bias_mean_kcal_mol', float('nan')):.1f}/{summary.get('umbrella_bias_max_kcal_mol', float('nan')):.1f} kcal"
                progress.progress(
                    "gareus_production", prod_done, prod_total,
                    message=msg,
                    timestep_fs=float(args.timestep_fs),
                    n_replicas=nrep,
                    extra={"exchange_attempts": attempt, **summary},
                    force=(prod_done >= prod_total),
                )

        if not analysis_arrays_written:
            flush_scalar_writers()
            metadata["parquet_samples_dir"] = str(out_dir / "samples")
            metadata["parquet_exchanges_dir"] = str(out_dir / "exchanges")
            metadata["parquet_segment_id"] = _seg_id
            write_json(out_dir / "gareus_metadata.json", metadata)
            analysis_arrays_written = True

        if bool(getattr(args, "adaptive_feedback_enabled", False)) or str(getattr(args, "window_mode", "adaptive")) == "adaptive-feedback":
            try:
                if (secondary_cv_metadata or {}).get("enabled") and secondary_cv_centers is not None and len(set(round(float(x), 4) for x in secondary_cv_centers)) > 1:
                    feedback_summary = run_adaptive_feedback_dispatcher_2d(
                        args, out_dir, centers_a, k_list, exchange_stats,
                        secondary_cv_centers, secondary_cv_k_kcal_list, secondary_cv_metadata,
                        fallback_history_by_window=distance_logger.history_by_window,
                    )
                else:
                    feedback_summary = run_adaptive_feedback_dispatcher(
                        args, out_dir, centers_a, k_list, exchange_stats,
                        fallback_history_by_window=distance_logger.history_by_window,
                    )
                if feedback_summary is not None:
                    metadata["adaptive_feedback_proposal_json"] = str(out_dir / "adaptive_feedback_proposal.json")
                    metadata["adaptive_feedback_new_n_windows"] = int(feedback_summary.get("new_n_windows", nrep))
                    write_json(out_dir / "gareus_metadata.json", metadata)
            except Exception as exc:
                print(f"WARNING: adaptive-feedback proposal failed: {exc}")

        # Save one final PDB per configuration replica.
        final_dir = out_dir / "final_pdbs"
        final_dir.mkdir(exist_ok=True)
        for r, sim in enumerate(sims):
            state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
            with (final_dir / f"replica_{r:03d}_window_{assignments[r]:03d}.pdb").open("w") as handle:
                app.PDBFile.writeFile(topology, state.getPositions(), handle, keepIds=True)

        try:
            exchange_report = write_exchange_tuning_report(out_dir, args, exchange_stats, centers_a=centers_a, secondary_cv_centers=secondary_cv_centers, secondary_cv_metadata=secondary_cv_metadata)
            metadata["exchange_tuning_report_md"] = str(out_dir / "exchange_tuning_report.md")
            metadata["exchange_tuning_report_json"] = str(out_dir / "exchange_tuning_report.json")
            metadata["exchange_tuning_status"] = str(exchange_report.get("status", "unknown"))
        except Exception as exc:
            print(f"WARNING: exchange tuning report failed: {exc}")
        try:
            report = write_final_run_report(out_dir, args, centers_a, k_list, exchange_stats)
            metadata["final_report_md"] = str(out_dir / "final_report.md")
            metadata["final_report_json"] = str(out_dir / "final_report.json")
            metadata["final_report_status"] = str(report.get("status", "unknown"))
            try:
                layout = write_standardized_output_layout(out_dir)
                metadata["output_layout_md"] = str(out_dir / "output_layout.md")
                metadata["output_layout_json"] = str(out_dir / "output_layout.json")
                metadata["output_layout_status"] = "ok" if layout else "unknown"
            except Exception as layout_exc:
                print(f"WARNING: standardized output layout failed: {layout_exc}")
            write_json(out_dir / "gareus_metadata.json", metadata)
            try:
                finalize_run_manifest(args, out_dir, status="completed")
                metadata["run_manifest_json"] = str(out_dir / "run_manifest.json")
                metadata["run_manifest_yaml"] = str(out_dir / "run_manifest.yaml")
                write_json(out_dir / "gareus_metadata.json", metadata)
            except Exception as prov_exc:
                print(f"WARNING: run provenance finalization failed: {prov_exc}")
            print(f"Final run report written to {out_dir / 'final_report.md'}")
        except Exception as exc:
            print(f"WARNING: final report/validation failed: {exc}")

    finally:
        try:
            _sim_pool.shutdown(wait=True)
        except Exception:
            pass
        try:
            parquet_sample_writer.close()
        except Exception:
            pass
        try:
            parquet_exchange_writer.close()
        except Exception:
            pass
        try:
            _seg_registry.close_segment(_seg_id, end_step=int(calib_steps + prod_done))
        except Exception:
            pass
        distance_logger.close()

    print(f"Done. Outputs in {out_dir}")
