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
import copy
import csv
import itertools
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, NamedTuple, Optional

import numpy as np

from .io import BufferedCsvDictWriter, write_json, read_json_file, _json_ready, acquire_run_lock
from .logger import DistanceLogger, is_gamd_production_phase
from .store import ParquetSampleWriter, ParquetExchangeWriter, SegmentRegistry, WindowSnapshot, parse_gamd_boost_components, finalize_segment
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
    resolve_barostat_ownership,
    preflight_barostat_ownership,
    _production_barostat_frequency,
)
from .seeding import (
    generate_us_starting_states_by_pulling,
    deserialize_system,
    filter_explicit_2d_windows_by_seed_reachability,
)
from .diagnostics import compute_gamd_reweighting_diagnostics, validate_us_mbar_inputs
from .imports import import_gamd_factory, import_openmm
from .state import _scalar_to_float, _energy_to_kj_mol
from .cv import (
    apply_primary_cv_metadata_to_args,
    choose_cv_atoms,
    contact_normalization_denominator,
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
from .provenance import initialize_run_manifest, update_run_manifest, finalize_run_manifest, pair_model_sha256

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
    "rewrite_epoch_window_map_after_drop",
    "rewrite_epoch_window_map_after_unreachable_filter",
    "repair_epoch_window_map_from_surviving_windows",
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


class _ReplicaAffinityExecutor:
    """One dedicated single-thread worker per replica, for the replica's
    entire lifetime (Context construction through every later step()).

    A plain ThreadPoolExecutor does not guarantee which worker thread
    picks up which task across separate .map()/.submit() calls, so a
    replica's CUDA Context built during replica_construction (main
    thread, sequential) could get its first step() dispatched from a
    different OS thread by the production step pool. Concurrent
    first-touch of many CUDA contexts across threads is a driver-level
    segfault hazard - matches a SIGSEGV firing the instant after
    replica_construction hit N/N, invariant across CUDA_LAUNCH_BLOCKING
    and MPS toggles. Pinning each replica to one thread for good removes
    the hazard; interface mirrors ThreadPoolExecutor.map/.shutdown so
    the call sites in run_gareus are unchanged.
    """

    def __init__(self, n):
        self._pools = [ThreadPoolExecutor(max_workers=1) for _ in range(n)]

    def submit(self, replica_index, fn, *args, **kwargs):
        return self._pools[replica_index].submit(fn, *args, **kwargs)

    def map(self, fn, items):
        items = list(items)
        futures = [self._pools[i].submit(fn, item) for i, item in enumerate(items)]
        return [f.result() for f in futures]

    def shutdown(self, wait=True):
        for p in self._pools:
            p.shutdown(wait=wait)


def _bootstrap_torsion_state_path(args, out_dir: Path) -> Path:
    configured = str(getattr(args, "bootstrap_torsion_state_file", "") or "")
    if configured:
        return Path(configured)
    return Path(out_dir) / "tica" / "bootstrap_torsion_cv.json"


def _seed_projection_centers(values: np.ndarray, n_centers: int = 3) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n_centers = max(3, int(n_centers))
    if arr.size < n_centers:
        raise ValueError(f"torsion-pca center selection needs at least {n_centers} finite projections, got {arr.size}")
    # Span (near-)full seed distribution, not the inner 70%: the outer windows
    # must reach the folded/extended tails of the seed ensemble so the umbrella
    # ladder covers the fold coordinate. Robust 2/98% quantiles avoid a single
    # outlier stretching the range. The dispatcher then refines the count.
    quantiles = np.linspace(0.02, 0.98, n_centers)
    centers = [float(x) for x in np.quantile(arr, quantiles)]
    centers = [max(-6.0, min(6.0, c)) for c in centers]
    if len({round(c, 6) for c in centers}) < n_centers:
        raise ValueError(f"torsion-pca seed projections collapsed; cannot choose {n_centers} distinct CV2 centers")
    return centers


def _ensure_bootstrap_torsion_cv_ready(args, out_dir: Path, topology, primary_cv_def: dict) -> dict:
    from .cv import secondary_structure_torsions
    from .seeding import load_genpept_conformer_library, map_topology_torsions_to_conformer
    from .tica import backbone_dihedral_features, compute_bootstrap_torsion_pca, project_tica1, TICAResult

    if secondary_cv_mode(args) != "torsion-pca":
        return {}

    source = str(getattr(args, "bootstrap_torsion_source", "seeds") or "seeds").strip().lower()
    if source != "seeds":
        raise ValueError(f"Unsupported bootstrap_torsion_source={source!r}; only 'seeds' is supported")

    state_path = _bootstrap_torsion_state_path(args, out_dir)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    args.bootstrap_torsion_state_file = str(state_path)

    phi_torsions, psi_torsions = secondary_structure_torsions(topology)
    if not phi_torsions and not psi_torsions:
        raise ValueError("torsion-pca requires at least one backbone phi/psi torsion")
    expected_features = 2 * len(phi_torsions) + 2 * len(psi_torsions)

    seed_dir = str(getattr(args, "seed_conformers_dir", "") or "")
    if not seed_dir:
        raise ValueError("cv2=torsion-pca requires seed_conformers_dir")

    library = load_genpept_conformer_library(
        Path(seed_dir),
        primary_cv_def=primary_cv_def,
        args=args,
        topology=topology,
        secondary_cv_metadata=None,
    )
    usable = []
    for entry in library:
        pos = np.asarray(entry.get("positions_nm", []), dtype=np.float64)
        if pos.ndim != 2:
            continue
        atom_map = entry.get("topology_to_conformer_atom_index")
        if isinstance(atom_map, dict):
            seed_phi_torsions = map_topology_torsions_to_conformer(phi_torsions, topology, atom_map)
            seed_psi_torsions = map_topology_torsions_to_conformer(psi_torsions, topology, atom_map)
        else:
            seed_phi_torsions = map_topology_torsions_to_conformer(phi_torsions, None, None)
            seed_psi_torsions = map_topology_torsions_to_conformer(psi_torsions, None, None)
        if len(seed_phi_torsions) != len(phi_torsions) or len(seed_psi_torsions) != len(psi_torsions):
            continue
        usable.append((entry, pos, seed_phi_torsions, seed_psi_torsions))
    min_count = int(getattr(args, "bootstrap_torsion_min_seed_count", 20) or 20)
    if len(usable) < min_count:
        raise ValueError(
            f"cv2=torsion-pca requires at least {min_count} usable seed conformers, got {len(usable)}"
        )

    X = np.vstack(
        [
            backbone_dihedral_features(pos, seed_phi_torsions, seed_psi_torsions)
            for _entry, pos, seed_phi_torsions, seed_psi_torsions in usable
        ]
    )
    if X.shape[1] != expected_features:
        raise ValueError(f"torsion-pca feature count {X.shape[1]} != expected {expected_features}")

    cv1 = np.asarray([float(entry.get("primary_cv_value", np.nan)) for entry, _pos, _phi, _psi in usable], dtype=np.float64)
    # Default OFF: residualizing the backbone-torsion features against cv1 removes
    # the cv1-correlated component, but for a folder the fold IS correlated torsion
    # + contact motion -- so residualizing strips the fold signal and collapses the
    # seed PC1 variance, leaving CV2 window centers crammed into a thin band that
    # never spans the coordinate the runtime restraint explores. Opt in only when
    # an orthogonal-to-cv1 secondary motion is genuinely wanted.
    residualize = bool(getattr(args, "bootstrap_torsion_residualize_against_cv1", False))
    if residualize and not np.isfinite(cv1).all():
        print(
            "WARNING: cv2=torsion-pca residualization requested, but seed primary_cv_value "
            "is unavailable for at least one usable seed; using non-residualized bootstrap torsion PCA."
        )
        residualize = False

    if state_path.exists():
        result = TICAResult.load(state_path)
        if str(getattr(result, "method", "") or "") != "pca":
            raise ValueError(
                f"bootstrap torsion state {state_path} has method {result.method!r}; expected method='pca'"
            )
        if len(result.weights) != expected_features:
            raise ValueError(
                f"bootstrap torsion state {state_path} has {len(result.weights)} weights; expected {expected_features}"
            )
        explained = result.explained_variance_ratio
        if explained is None:
            explained = result.eigenvalue
        explained = float(explained)
        if not np.isfinite(explained) or explained <= 0.0:
            raise ValueError(f"bootstrap torsion state {state_path} has no valid PCA variance")
    else:
        result = compute_bootstrap_torsion_pca(
            X,
            cv1=cv1 if residualize else None,
            residualize=residualize,
            component=int(getattr(args, "bootstrap_torsion_component", 5) or 5),
            phi_torsion_indices=phi_torsions,
            psi_torsion_indices=psi_torsions,
        )
        result.save(state_path)

    projections = project_tica1(X, result)
    if getattr(args, "_cv2_auto_centers", False) and not getattr(args, "secondary_cv_centers", None):
        n_centers = int(getattr(args, "cv2_n_centers", 3) or 3)
        centers = _seed_projection_centers(projections, n_centers=n_centers)
        args.cv2_centers = centers
        args.secondary_cv_centers = centers

    explained = result.explained_variance_ratio
    if explained is None:
        explained = result.eigenvalue
    return {
        "state_file": str(state_path),
        "method": str(result.method),
        "n_samples": int(result.n_samples),
        "explained_variance_ratio": float(explained),
        "residualized_against_cv1": bool(residualize),
        "seed_projection_min": float(np.min(projections)),
        "seed_projection_max": float(np.max(projections)),
        "seed_projection_centers": [float(x) for x in getattr(args, "secondary_cv_centers", []) or []],
    }


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
    """Return a CustomTorsionForce that sums normalized periodic Gaussian scores.

    Uses OpenMM's ``theta`` directly, unlike ``_add_weighted_trig_torsion_force``
    below. That's correct here, not an oversight: ``target_rad`` comes from
    hardcoded literature Ramachandran angles (see ``secondary_cv_target_angles``
    and ``rama_map_definitions``) already expressed in OpenMM's own convention,
    never from ``tica._dihedral_rad``. Don't "fix" this by analogy with the
    tica-linear/torsion-pca sign fix -- cos(theta-target) is not symmetric in
    theta the way a bare cos(theta) is, so negating theta here would silently
    break alpha/beta and rama-map scoring instead of correcting it.
    """
    force = openmm.CustomTorsionForce(f"{norm_name}*exp(-(1-cos(theta-{target_name}))/({sigma_name}*{sigma_name}))")
    force.addGlobalParameter(str(target_name), float(target_rad))
    force.addGlobalParameter(str(sigma_name), float(sigma_rad))
    force.addGlobalParameter(str(norm_name), 1.0 / max(1, len(torsions)))
    for a, b, c, d in torsions:
        force.addTorsion(int(a), int(b), int(c), int(d), [])
    return force


def _add_weighted_trig_torsion_force(openmm, torsions, weights, trig: str):
    """Return one force summing per-torsion weighted sin/cos contributions.

    Uses ``-theta`` because OpenMM's CustomTorsionForce ``theta`` is the negative
    of the dihedral convention used by ``tica._dihedral_rad`` (no ``b0`` flip),
    which is the basis every stored TICAResult (weights/offset), window center,
    and seed-bank rescore value is expressed in. cos(-theta) == cos(theta), so
    this only matters for the sin terms, but flipping unconditionally keeps the
    weight-slicing/grouping logic below untouched. Without it, this force pulls
    toward a value that disagrees with project_tica1() on the same structure by
    an amount that grows with the sin-component weight -- reproducible in the
    US-seeding quality gate as a fixed-looking per-window secondary_cv_delta
    that no amount of extra pull time or restraint strength can close, because
    the restraint is converging correctly to the wrong number.
    """
    expr = f"w*{trig}(-theta)"
    force = openmm.CustomTorsionForce(expr)
    force.addPerTorsionParameter("w")
    for (a, b, c, d), weight in zip(torsions, weights):
        force.addTorsion(int(a), int(b), int(c), int(d), [float(weight)])
    return force


def _add_residual_torsion_cv_force(openmm, system, phi_torsions, psi_torsions, contact_pairs,
                                   runtime, args, *, force_group):
    """Harmonic umbrella on a residual torsion component, chain rule included.

    z2 = ( v.phi - K0 - K1 a - K2 a^2 ) / sigma_j with a = (contacts/norm - mu_c)/sigma_c.
    The whole expression is handed to one CustomCVForce so OpenMM differentiates
    the anchor-dependent terms too: dropping them would sample a different CV
    from the one the analysis reconstructs. The contact sum is a private copy
    of the CV1 umbrella's child force (OpenMM allows one parent per child), with
    the identical expression from forces.add_contact_umbrella_force.

    ``ss_k`` arrives in kJ/mol/CV^2 (production converts before set_window), so
    the energy expression carries no unit factor.
    """
    runtime.check_topology(phi_torsions, psi_torsions)
    runtime.check_anchor(args, contact_pairs)
    fit, j = runtime.fit, int(runtime.j)
    v = np.asarray(fit.right_vectors[j - 1], dtype=np.float64)
    n_phi, n_psi = len(phi_torsions), len(psi_torsions)
    if v.size != 2 * n_phi + 2 * n_psi:
        raise RuntimeError(f"pair model width {v.size} != topology features {2 * n_phi + 2 * n_psi}")
    cv_force = openmm.CustomCVForce("0")
    names = []
    grouped = (
        ("sum_sin_phi", phi_torsions, v[0:2 * n_phi:2], "sin"),
        ("sum_cos_phi", phi_torsions, v[1:2 * n_phi:2], "cos"),
        ("sum_sin_psi", psi_torsions, v[2 * n_phi::2], "sin"),
        ("sum_cos_psi", psi_torsions, v[2 * n_phi + 1::2], "cos"),
    )
    for fname, torsions, weights, trig in grouped:
        if len(torsions):
            cv_force.addCollectiveVariable(fname, _add_weighted_trig_torsion_force(openmm, torsions, weights, trig))
            names.append(fname)
    r0_nm = float(args.contact_r0_a) * 0.1
    beta_nm_inv = float(args.contact_beta_a_inv) * 10.0
    contact_sum = openmm.CustomBondForce(
        f"contact_weight*0.5*(1-tanh(0.5*{beta_nm_inv:.17g}*(r-{r0_nm:.17g})))")
    contact_sum.addPerBondParameter("contact_weight")
    for pair in contact_pairs:
        contact_sum.addBond(int(pair[0]), int(pair[1]), [float(pair[2]) if len(pair) > 2 else 1.0])
    cv_force.addCollectiveVariable("res_contacts", contact_sum)
    norm = float(runtime.anchor_definition.get("norm", contact_normalization_denominator(list(contact_pairs), args)))
    K0 = float(v @ (fit.coefficients[0] + fit.residual_mean) + fit.projection_mean[j - 1])
    K1 = float(v @ fit.coefficients[1])
    K2 = float(v @ fit.coefficients[2])
    a_raw = f"((res_contacts/{norm:.17g}) - {fit.anchor_mean:.17g})/{fit.anchor_std:.17g}"
    lo, hi = fit.anchor_clamp
    a_expr = f"min({hi:.17g}, max({lo:.17g}, {a_raw}))" if fit.degree == 2 else a_raw
    z2 = (f"(({' + '.join(names)}) - {K0:.17g} - {K1:.17g}*({a_expr}) - {K2:.17g}*({a_expr})^2)"
          f"/{fit.projection_std[j - 1]:.17g}")
    cv_force.addGlobalParameter("ss_k", 0.0)
    cv_force.addGlobalParameter("ss0", 0.0)
    cv_force.setEnergyFunction(f"0.5*ss_k*({z2}-ss0)^2")
    cv_force.setForceGroup(int(force_group))
    system.addForce(cv_force)
    return {
        "enabled": True,
        "mode": "residual-torsion-pc",
        "label": f"residual torsion component {j} (degree {fit.degree}) against {runtime.anchor_kind}",
        "pair_model_sha256": runtime.pair_sha256,
        "component_index": j,
        "degree": int(fit.degree),
        "anchor_clamp": [float(lo), float(hi)],
        "range_min": -6.0,
        "range_max": 6.0,
        "n_phi_torsions": int(n_phi),
        "n_psi_torsions": int(n_psi),
        "phi_torsions": [list(map(int, t)) for t in phi_torsions],
        "psi_torsions": [list(map(int, t)) for t in psi_torsions],
        "contact_pairs": [[int(p[0]), int(p[1]), float(p[2]) if len(p) > 2 else 1.0] for p in contact_pairs],
        "linear_subcv_names": list(names),
        "force_group": int(force_group),
        # Stripped from the JSON manifest by _json_ready's underscore rule; the
        # scorer reloads from the recorded paths after a resume.
        "_runtime": runtime,
    }


def _linear_torsion_state_for_mode(args, mode: str) -> tuple[Path, "TICAResult"]:
    from .tica import TICAResult

    if mode == "tica-linear":
        state_file = str(getattr(args, "tica_state_file", "") or "")
        if not state_file or not Path(state_file).exists():
            raise FileNotFoundError(state_file)
    elif mode == "torsion-pca":
        state_file = str(getattr(args, "bootstrap_torsion_state_file", "") or "")
        if not state_file or not Path(state_file).exists():
            raise RuntimeError("cv2=torsion-pca selected but bootstrap torsion state file is missing")
    else:
        raise ValueError(f"Unsupported linear torsion CV mode {mode!r}")
    path = Path(state_file)
    result = TICAResult.load(path)
    method = str(getattr(result, "method", "tica") or "tica")
    if mode == "torsion-pca" and method != "pca":
        raise RuntimeError(f"cv2=torsion-pca requires state method='pca', got {method!r}")
    if mode == "tica-linear" and method != "tica":
        raise RuntimeError(f"cv2=tica-linear requires state method='tica', got {method!r}")
    return path, result


def _restore_secondary_cv_args_from_metadata(args, secondary_cv_metadata: dict, out_dir: Optional[Path] = None) -> None:
    meta = dict(secondary_cv_metadata or {})
    if not meta.get("enabled"):
        return
    mode = secondary_cv_mode(str(meta.get("mode", "custom") or "custom"))
    args.secondary_cv = mode
    args.secondary_cv_phi0_deg = float(meta.get("phi0_deg", getattr(args, "secondary_cv_phi0_deg", -60.0)))
    args.secondary_cv_psi0_deg = float(meta.get("psi0_deg", getattr(args, "secondary_cv_psi0_deg", -45.0)))
    args.secondary_cv_sigma_deg = float(meta.get("sigma_deg", getattr(args, "secondary_cv_sigma_deg", 35.0)))
    if mode not in {"tica-linear", "torsion-pca"}:
        return

    raw_state = str(meta.get("tica_state_path", "") or "")
    candidates: list[Path] = []
    if raw_state:
        state_path = Path(raw_state)
        candidates.append(state_path)
        if out_dir is not None and not state_path.is_absolute():
            candidates.append(Path(out_dir) / state_path)
    chosen = next((path for path in candidates if path.exists()), None)
    if chosen is None:
        raise RuntimeError(f"Resume metadata for {mode} has missing tica_state_path: {raw_state or '<empty>'}")
    if mode == "tica-linear":
        args.tica_state_file = str(chosen)
    else:
        args.bootstrap_torsion_state_file = str(chosen)


def _add_linear_torsion_cv_force(
    openmm,
    system,
    phi_torsions,
    psi_torsions,
    result,
    *,
    mode: str,
    state_path: Path,
    force_group: int,
) -> dict:
    weights = np.asarray(result.weights, dtype=np.float64)
    offset = float(result.offset)
    n_phi = len(phi_torsions)
    n_psi = len(psi_torsions)
    expected_feats = 2 * n_phi + 2 * n_psi
    if len(weights) != expected_feats:
        raise RuntimeError(
            f"{mode} weight vector has {len(weights)} components but topology provides "
            f"{expected_feats} features ({n_phi} phi + {n_psi} psi torsions, 2 features each)"
        )
    cv_force = openmm.CustomCVForce("0")
    sub_cv_names = []
    grouped_terms = [
        ("sum_sin_phi", phi_torsions, weights[0 : 2 * n_phi : 2], "sin"),
        ("sum_cos_phi", phi_torsions, weights[1 : 2 * n_phi : 2], "cos"),
        ("sum_sin_psi", psi_torsions, weights[2 * n_phi : 2 * n_phi + 2 * n_psi : 2], "sin"),
        ("sum_cos_psi", psi_torsions, weights[2 * n_phi + 1 : 2 * n_phi + 2 * n_psi : 2], "cos"),
    ]
    for fname, torsions, grouped_weights, trig in grouped_terms:
        if len(torsions) == 0:
            continue
        sub_f = _add_weighted_trig_torsion_force(openmm, torsions, grouped_weights, trig)
        cv_force.addCollectiveVariable(fname, sub_f)
        sub_cv_names.append(fname)
    cv_force.addGlobalParameter("ss_k", 0.0)
    cv_force.addGlobalParameter("ss0", 0.0)
    linear_terms = " + ".join(sub_cv_names)
    linear_expr = f"({linear_terms} + ({offset:.12g}))"
    cv_force.setEnergyFunction(f"0.5*ss_k*({linear_expr}-ss0)^2")
    cv_force.setForceGroup(int(force_group))
    system.addForce(cv_force)

    label = "inter-epoch tICA linear CV (tIC1, slowest mode)"
    if mode == "torsion-pca":
        label = "bootstrap torsion PC1 from GENPEPT seed ensemble"
    return {
        "enabled": True,
        "mode": mode,
        "linear_cv_kind": str(getattr(result, "method", "tica") or "tica"),
        "label": label,
        "range_min": -6.0,
        "range_max": 6.0,
        "n_phi_torsions": int(n_phi),
        "n_psi_torsions": int(n_psi),
        "tica_state_path": str(state_path),
        "weights": weights.tolist(),
        "tica_offset": float(offset),
        "phi_torsions": [list(map(int, t)) for t in phi_torsions],
        "psi_torsions": [list(map(int, t)) for t in psi_torsions],
        "linear_subcv_mode": "grouped-weighted-sums",
        "linear_subcv_names": list(sub_cv_names),
        "force_group": int(force_group),
        "eigenvalue": float(result.eigenvalue),
        "n_samples": int(result.n_samples),
        "method": str(getattr(result, "method", "tica") or "tica"),
        "explained_variance_ratio": (
            float(result.explained_variance_ratio)
            if getattr(result, "explained_variance_ratio", None) is not None
            else None
        ),
    }


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
        groups=physical_energy_groups_for_args(args),
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

    For rama-map: sub-CVs interleave phi/psi per region in
    definition order — [phi_0, psi_0, phi_1, psi_1, ...].
    For alpha-coil-beta: [alpha_phi, alpha_psi, beta_phi, beta_psi].
    For simple modes (alpha/beta/custom): [ss_phi, ss_psi].
    """
    mode = secondary_cv_mode(metadata)
    arr = np.asarray(sub_cv_values, dtype=np.float64)
    if mode in {"tica-linear", "torsion-pca"}:
        offset = float(metadata.get("tica_offset", 0.0))
        if str(metadata.get("linear_subcv_mode", "")) == "grouped-weighted-sums":
            return float(np.sum(arr) + offset)
        weights = np.asarray(metadata.get("weights", []), dtype=np.float64)
        return float(arr @ weights + offset)
    if mode == "alpha-coil-beta":
        return float(0.5 * (arr[0] + arr[1]) - 0.5 * (arr[2] + arr[3]))
    if mode == "rama-map":
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
    # No upper clip: true heat-bath weights exp(-β·δ) can exceed 1 for favourable
    # moves (δ < 0).  The log-sum-exp shift on line ~390 handles overflow.
    # Lower bound -745 prevents underflow to exact zero (loses that candidate).
    log_weights = np.clip(-float(beta) * delta, -745.0, 745.0)
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


def _exchange_probability(delta_kj: float, beta: float) -> float:
    """Metropolis probability for a proposed umbrella-window swap.

    Extracted from run_gareus's exchange-loop closures (was a nested def
    capturing beta from the enclosing scope) so it's unit-testable without a
    live OpenMM Context.
    """
    try:
        delta_kj = float(delta_kj)
        beta = float(beta)
    except Exception:
        return 0.0
    if not (math.isfinite(delta_kj) and math.isfinite(beta)):
        return 0.0
    if delta_kj <= 0.0:
        return 1.0
    x = -beta * delta_kj
    if x < -745.0:
        return 0.0
    return float(math.exp(x))


def swap_candidate_replicas(replica_of_window, wi: int, wj: int):
    """``(i, j)`` for an attemptable swap, else ``None``.

    Split out of :func:`apply_window_swap` so a caller can decide whether the
    move is attemptable *before* drawing a random number. Production's original
    closure returned early on these cases without touching the rng, and any
    extra draw here shifts every downstream random stream -- the same hazard
    ``force_accept`` is guarded against.
    """
    try:
        wi = int(wi)
        wj = int(wj)
    except (TypeError, ValueError):
        return None
    if wi == wj:
        return None
    n_win = int(replica_of_window.size)
    if not (0 <= wi < n_win and 0 <= wj < n_win):
        return None
    i = int(replica_of_window[wi])
    j = int(replica_of_window[wj])
    if i < 0 or j < 0 or i == j:
        return None
    return i, j


class SwapOutcome(NamedTuple):
    """What one attempted window swap did. Returned by :func:`apply_window_swap`."""
    replica_i: int
    replica_j: int
    delta_kj: float
    pacc: float
    accepted: bool


def apply_window_swap(
    bias_matrix_kj,
    beta: float,
    assignments,
    replica_of_window,
    wi: int,
    wj: int,
    uniform: Optional[float],
    *,
    p_override: Optional[float] = None,
    force_accept: bool = False,
) -> "Optional[SwapOutcome]":
    """Decide and apply one umbrella-window swap; the pure core of the REUS move.

    Extracted from ``run_gareus``'s closure so the exchange kernel can be driven
    from a test against a known target distribution. Everything here is pure
    NumPy/arithmetic: ``assignments`` and ``replica_of_window`` are mutated in
    place exactly as production does, and the caller applies the side effects
    (``set_window`` on the two contexts, the Parquet journal entry) from the
    returned outcome.

    Returns ``None`` when there is nothing to attempt -- same window, either
    window unheld, or both held by the same replica -- so the caller can leave
    its attempt counter alone.

    The move swaps *state labels*, never configurations, which is why the
    unbiased potential and the GaMD boost cancel exactly from ``delta`` and only
    the umbrella biases appear: ``x_i`` and ``x_j`` are untouched, so
    ``U0(x_i)+U0(x_j)`` and ``dV(x_i)+dV(x_j)`` are identical before and after.

    ``uniform`` is the already-drawn ``rng.random()``. It is a parameter rather
    than an rng handle so that the kernel is deterministic under test, and so
    the caller can preserve production's short-circuit: ``force_accept`` must
    not consume a random number, or every downstream RNG stream shifts.
    """
    pair = swap_candidate_replicas(replica_of_window, wi, wj)
    if pair is None:
        return None
    i, j = pair
    wi = int(wi)
    wj = int(wj)
    # bias_matrix_kj[window, replica] = U_window(x_replica)
    old_e = float(bias_matrix_kj[wi, i] + bias_matrix_kj[wj, j])
    new_e = float(bias_matrix_kj[wj, i] + bias_matrix_kj[wi, j])
    delta = float(new_e - old_e)
    pacc = float(p_override) if p_override is not None else _exchange_probability(delta, beta)
    pacc = max(0.0, min(1.0, pacc)) if math.isfinite(pacc) else 0.0
    if force_accept:
        accepted = True
    else:
        accepted = bool(float(uniform) < pacc) if uniform is not None else False
    if accepted:
        assignments[i], assignments[j] = assignments[j], assignments[i]
        replica_of_window[int(assignments[i])] = int(i)
        replica_of_window[int(assignments[j])] = int(j)
    return SwapOutcome(i, j, delta, pacc, accepted)


class GibbsProposal(NamedTuple):
    """One replica's heat-bath proposal and its Metropolis-Hastings correction.

    ``pacc`` is what the caller must hand to :func:`apply_window_swap` as
    ``p_override``. It is NOT 1.0: the heat-bath proposal is nonuniform, so
    accepting a selected move unconditionally biases the permutation chain.
    """
    current_window: int
    proposed_window: int
    delta_kj: float
    q_forward: float
    q_reverse: float
    pacc: float
    stayed: bool          # the proposal selected the current window
    no_candidates: bool   # every candidate was NaN-masked out


def gibbs_propose_one_replica(
    bias_matrix_kj,
    beta: float,
    assignments,
    replica_of_window,
    replica_index: int,
    choose,
) -> "GibbsProposal":
    """Decide one `gibbs-walk` move: propose, reverse-propose, MH-correct.

    Split out of ``run_gareus``'s exchange closure so the exact-enumeration
    tests drive *this* code rather than a copy of it. A hand-written mirror of
    these six steps in a test proves only that the mirror is self-consistent --
    it cannot see a defect introduced on the production side.

    ``choose(n_candidates, probabilities) -> index`` is the sampling step,
    injected so the caller supplies the RNG and a test can enumerate every
    branch deterministically. Production passes ``rng.choice``.

    The reverse proposal is evaluated against ``holders_after`` -- the holder
    table as it *would* be post-swap -- because MH requires the reverse
    probability from the destination state, not the current one. Getting this
    wrong is the classic heat-bath-with-MH bug and it is silent: the chain
    still runs, it just no longer targets pi.
    """
    rep = int(replica_index)
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
        return GibbsProposal(wi, wi, 0.0, 0.0, 0.0, 0.0, False, True)

    choice_index = int(choose(int(valid_windows.size), probs))
    wj = int(valid_windows[choice_index])
    q_forward = float(probs[choice_index])
    if wj == wi:
        return GibbsProposal(wi, wi, 0.0, q_forward, 0.0, 0.0, True, False)

    holders_after = np.asarray(replica_of_window).astype(np.int64, copy=True)
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
    return GibbsProposal(
        wi, wj, float(deltas[choice_index]), q_forward, q_reverse, float(pacc), False, False
    )




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
    # Reconcile BEFORE the enabled/disabled branch below: secondary_meta and
    # secondary_cv_centers/secondary_k are read independently from the
    # checkpoint manifest, so a desynced checkpoint can have real centers/k
    # arrays with metadata that is missing or has enabled=False. If we only
    # reconciled after this point (as a downstream post-processing step), the
    # `else` branch immediately below would already have nulled centers/k by
    # the time any caller saw them, making the desync unrecoverable and
    # unobservable. Reconciling here, against the raw values, is the only
    # place that can actually catch and fix it.
    secondary_meta = reconcile_resume_secondary_cv_metadata(
        secondary_meta, secondary_centers,
        current_pair_sha256=pair_model_sha256(getattr(args, "secondary_cv_model", None)),
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
    gamd_lambdas=None,
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
    lam_arr = (
        np.zeros(n, dtype=float)
        if gamd_lambdas is None
        else np.asarray(gamd_lambdas, dtype=float)
    )
    if lam_arr.size != n:
        raise ValueError(
            f"gamd_lambdas has {lam_arr.size} entries but there are {n} windows"
        )
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
            "gamd_lambda": float(lam_arr[i]),
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


def add_secondary_structure_cv_force(openmm, system, topology, args, force_group: int = 29, *,
                                     primary_cv_def=None) -> dict:
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
    if mode == "auto":
        # Left unresolved, "auto" would fall through to secondary_cv_target_angles,
        # which returns 0,0,"disabled" for an unknown mode and builds a phi0=psi0=0
        # content force without a word. Refuse loudly instead.
        raise RuntimeError("cv2=auto must be resolved by the swarm stage before production; "
                           "run with window_mode=adaptive-production or pass the frozen model")
    if mode == "residual-torsion-pc":
        if primary_cv_def is None:
            raise RuntimeError("cv2=residual-torsion-pc needs the primary CV definition (contact pairs)")
        paths = tuple(getattr(args, key, None) for key in
                      ("secondary_cv_model", "secondary_cv_candidate_set", "secondary_cv_feature_schema"))
        if any(p is None or not str(p) for p in paths):
            raise RuntimeError("cv2=residual-torsion-pc needs --secondary-cv-model, "
                               "--secondary-cv-candidate-set and --secondary-cv-feature-schema")
        from .cv_selection.models import PairModelRuntime

        runtime = PairModelRuntime.load(*paths)
        info = _add_residual_torsion_cv_force(
            openmm, system, phi_torsions, psi_torsions,
            list(primary_cv_def.get("contact_pairs", [])), runtime, args, force_group=force_group)
        info.update({"pair_model_path": str(paths[0]), "candidate_set_path": str(paths[1]),
                     "feature_schema_path": str(paths[2])})
        return info
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

    if mode == "rama-map":
        regions = rama_map_definitions()
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

    if mode in {"tica-linear", "torsion-pca"}:
        try:
            state_path, result = _linear_torsion_state_for_mode(args, mode)
        except FileNotFoundError:
            return {
                "enabled": False,
                "mode": "tica-linear",
                "tica_state_path": str(getattr(args, "tica_state_file", "") or ""),
                "note": "no tica_state_file; secondary CV disabled for this epoch",
            }
        except Exception as exc:
            if mode == "torsion-pca":
                raise RuntimeError(f"Failed to load bootstrap torsion state: {exc}") from exc
            raise RuntimeError(f"Failed to load tICA state: {exc}") from exc
        return _add_linear_torsion_cv_force(
            openmm,
            system,
            phi_torsions,
            psi_torsions,
            result,
            mode=mode,
            state_path=state_path,
            force_group=force_group,
        )

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


from .pep_gamd import (
    boost_target_energy_kj,
    build_pep_gamd_integrator,
    find_aux_force as _find_pep_gamd_aux_force,
    is_pep_gamd,
    ladder_supports_boost_type,
    k0max_from_globals,
    pep_gamd_boost_matrix_kj,
    peptide_essential_energy_kj,
    physical_energy_groups_for_args,
    physical_potential_energy_kj,
    prepare_pep_gamd_args,
    set_replica_lambda_for_window,
    total_energy_groups_for_args,
    AUX_NONBONDED_GROUP as _PEP_GAMD_AUX_GROUP,
    DIHEDRAL_GROUP,
    PepGamdEnvelope,
)
from .npt_driver import (
    NPT_CHECKPOINT_PHASE_NOTE,
    NPT_REPORT_PHASE_NOTE,
    NptRunContext,
    ReplicaStepDriver,
    npt_seed_check_run,
    npt_seed_recon_window,
    npt_seed_replica,
    npt_seed_shared_setup,
)


def _resolve_npt_adapter(args, system):
    """Locate the stage-aware effective-potential (U*) adapter for this run.

    The adapter implementations are package 2's half of the NPT correction
    (``gareus/pep_gamd.py``: the stage-aware separation of physical energy,
    auxiliary energy and boost inputs).  This single seam is how package B
    consumes it: when the biased-MC backend is resolved, the adapter factory
    must exist or the run fails here -- loudly, never by silently falling
    back to the wrong acceptance energy.  Tests patch this one function.
    """
    from . import pep_gamd
    factory = getattr(pep_gamd, "make_npt_target_adapter", None)
    if factory is None:
        raise RuntimeError(
            "The biased-MC NPT backend requires the stage-aware effective-potential "
            "adapter from gareus/pep_gamd.py (NPT correction package 1), which is not "
            "present in this build. Refusing to run boosted NPT rather than accept "
            "volume moves against the wrong energy; use --production-ensemble nvt or "
            "a conventional (cmd) run mode until the adapter lands."
        )
    return _LazyNptAdapter(factory, args, system)


class _LazyNptAdapter:
    """Defers adapter construction until a real driving Context exists.

    The seam is resolved once, run-wide, immediately after the base System is
    built -- but ``make_npt_target_adapter`` validates the *integrator* (its
    Total-channel group arithmetic, its group dict and its required globals)
    and the *system's* Pep-GaMD partition, and at resolve time neither exists:
    ``make_gamd_integrator`` adds the partition to each run stage's own
    deserialized copy of the base System (shared setup, recon windows,
    production replicas), never to the base System itself.  So the real
    adapter is built on the first ``snapshot()``/``ensure_built()`` call, from
    the System of the Context handed to that call, and cached.

    Sharing one adapter across the run's contexts is safe: every context that
    drives the volume controller is a deserialized copy of the same base
    System passed through the same ``make_gamd_integrator`` partition, so the
    force-group layout the constructor inspects is identical for all of them,
    and ``snapshot``/``evaluate`` read the context and integrator handed to
    them, never anything captured at construction.
    """

    def __init__(self, factory, args, system):
        self._factory = factory
        self._args = args
        self._system = system
        self._adapter = None

    def _resolve(self, context, integrator):
        if self._adapter is None:
            # The System must be the one this Context was built from: it
            # carries the Pep-GaMD partition, which the pre-partition base
            # System frozen at resolve time never has.
            system = None
            get_system = getattr(context, "getSystem", None)
            if get_system is not None:
                system = get_system()
            if system is None:
                system = self._system
            self._adapter = self._factory(system, integrator, self._args)
        return self._adapter

    def ensure_built(self, context, integrator=None):
        """Build the real adapter now, if it has not been built yet.

        The controller init/restore paths must see the real ``adapter_id``
        (checkpoint compatibility and restore both compare it) before any
        volume move has been attempted; this gives them a build point that
        needs no snapshot semantics.
        """
        if integrator is None:
            integrator = context.getIntegrator()
        return self._resolve(context, integrator)

    @property
    def adapter_id(self) -> str:
        return str(getattr(self._adapter, "adapter_id", "")) if self._adapter is not None else ""

    def snapshot(self, context, integrator):
        return self._resolve(context, integrator).snapshot(context, integrator)

    def evaluate(self, context, snapshot):
        if self._adapter is None:
            raise RuntimeError(
                "NPT adapter evaluate() called before snapshot(); the adapter is built "
                "from the integrator handed to snapshot()"
            )
        return self._adapter.evaluate(context, snapshot)


def _production_barostat_description(args) -> str:
    """Human-readable description of the production pressure algorithm actually in force."""
    ensemble = str(getattr(args, "production_ensemble", "npt"))
    if ensemble != "npt":
        return "none"
    backend = str(getattr(args, "npt_barostat_backend", "auto") or "auto")
    if backend == "biased_mc":
        freq = _production_barostat_frequency(args)
        return f"gareus BiasedMCBarostatController (biased Metropolis, every {freq} steps)"
    return "OpenMM MonteCarloBarostat"


def assemble_bias_matrices(distance_bias_kcal, ss_bias_kcal, boost_bias_kj):
    """Combine the umbrella components and the Pep-GaMD boost into one (kcal, kj) pair.

    Both unit matrices carry the SAME quantity -- umbrella + boost -- so the
    invariant ``bias_kj == 4.184 * bias_kcal`` holds regardless of whether the
    λ-ladder is active. ``boost_bias_kj`` is all-zero when it is not, so this
    is a no-op reduction to the pre-ladder umbrella-only matrix in that case.
    Pulled out to module level so both assembly sites in ``run_gareus`` (the
    log/sample path and ``_current_exchange_arrays``) share one definition and
    so it is directly testable without a live OpenMM Context.
    """
    distance_bias_kcal = np.asarray(distance_bias_kcal, dtype=np.float64)
    ss_bias_kcal = np.asarray(ss_bias_kcal, dtype=np.float64)
    boost_bias_kj = np.asarray(boost_bias_kj, dtype=np.float64)
    boost_bias_kcal = boost_bias_kj / 4.184
    bias_kcal = distance_bias_kcal + ss_bias_kcal + boost_bias_kcal
    bias_kj = 4.184 * bias_kcal
    return bias_kcal, bias_kj


def _fetch_v_pep_v_dih(ctx, pep_env, unit) -> tuple[float, float]:
    """Read (V_pep, V_dih) in kJ/mol for the Pep-GaMD boost from a live Context.

    Returns ``(nan, nan)`` without touching the Context when ``pep_env`` is
    None (the λ-ladder is not active), matching every call site's prior
    inline guard. Shared by both fetch closures' both branches in
    ``run_gareus`` (log/sample and exchange, fast-CV and non-fast).
    """
    if pep_env is None:
        return float("nan"), float("nan")
    # gamd-openmm's factory puts PeriodicTorsion/CMAPTorsion forces in group 2 for every
    # stock boost type (integrator_factory.set_dihedral_group), the same id as
    # DIHEDRAL_GROUP in the Pep-GaMD partition, so this read is valid on both paths.
    v_dih = (ctx.getState(getEnergy=True, groups={DIHEDRAL_GROUP}).getPotentialEnergy()
             .value_in_unit(unit.kilojoule_per_mole))
    if not getattr(pep_env, "has_total", True):
        # single dihedral boost: no Total channel, no auxiliary force to read
        return float("nan"), v_dih
    v_pep = peptide_essential_energy_kj(ctx, unit)
    return v_pep, v_dih


def gamd_enabled(args) -> bool:
    """Return True when production should use gamd-openmm integrators."""
    return production_run_mode(args) in {"gamd", "hmr-gamd"}


def make_cmd_integrator(openmm, args, unit, system=None):
    """Create a plain conventional-MD LangevinMiddleIntegrator.

    A plain integrator applies every force group, so if `system` carries the Pep-GaMD
    auxiliary water-only force it must be excluded here or water-water is counted twice.
    """
    integrator = openmm.LangevinMiddleIntegrator(
        float(args.temperature_k) * unit.kelvin,
        float(args.friction_per_ps) / unit.picosecond,
        float(args.timestep_fs) * unit.femtosecond,
    )
    try:
        integrator.setRandomNumberSeed(int(args.seed))
    except Exception:
        pass
    if system is not None and _find_pep_gamd_aux_force(system)[1] is not None:
        integrator.setIntegrationForceGroups(set(range(32)) - {_PEP_GAMD_AUX_GROUP})
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
    return make_cmd_integrator(openmm, args, unit, system=system)


def make_gamd_integrator(system, args, unit):
    if is_pep_gamd(args):
        result = build_pep_gamd_integrator(system, args, unit)
    else:
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

def snapshot_window_rows(centers_a, k_list, secondary_centers, secondary_k_list,
                         gamd_lambdas=None, n=None) -> list[dict]:
    """Per-segment window snapshot rows (windows/<segment_id>.json).

    Carries ``gamd_lambda`` per window. Without it every MBAR loader reading a
    snapshot had to INFER each state's rung by nanmedian over that window's own
    samples' ``gamd_lambda`` column -- an inference that is only as good as the
    sample set (a never-sampled window silently reads 0.0, i.e. "no ladder")
    and that merges nothing when the true λ is right there in the writer.
    2026-09-07 final review, I2.

    Note the downstream contract this creates: ``gareus.query.reconstruct_bias_matrix``
    refuses a window with ``gamd_lambda > 0`` unless ``v_pep``/``v_dih``/``envelope``
    are supplied, so every caller reconstructing from these rows must pass them.
    """
    count = int(n) if n is not None else len(list(centers_a))
    rows = []
    for wi in range(count):
        row = {
            "window_id": int(wi),
            "center1": float(centers_a[wi]),
            "k1": float(k_list[wi]),
        }
        if secondary_centers is not None and secondary_k_list is not None:
            row["center2"] = float(secondary_centers[wi])
            row["k2"] = float(secondary_k_list[wi])
        lam = 0.0
        if gamd_lambdas is not None and wi < len(gamd_lambdas):
            try:
                lam = float(gamd_lambdas[wi] or 0.0)
            except (TypeError, ValueError):
                lam = 0.0
        row["gamd_lambda"] = lam
        rows.append(row)
    return rows


def window_assignment_rows(centers_a: np.ndarray, k_list: list[float], temperature_k: float, secondary_centers=None, secondary_k_list=None, args=None, gamd_lambdas=None) -> list[dict]:
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
            "gamd_lambda": float(gamd_lambdas[i]) if gamd_lambdas is not None else 0.0,
        }
        if ss_centers is not None and ss_k_arr is not None and i < len(ss_centers) and i < len(ss_k_arr):
            row.update({
                "secondary_cv_center": float(ss_centers[i]),
                "secondary_cv_k_kcal_mol": float(ss_k_arr[i]),
                "secondary_cv_k_kj_mol": float(kcal_to_kj(ss_k_arr[i])),
            })
        rows.append(row)
    return rows

CAMPAIGN_LADDER_REGISTRY_NAMES = ("state_registry.csv", "final_registry_used_for_mbar.csv")


def campaign_ladder_registry_lambda(out_dir) -> Optional[tuple]:
    """First ``(registry filename, λ)`` with λ > 0 at or above ``out_dir``, else None.

    The λ-ladder is a property of the CAMPAIGN, but ``args.state_gamd_lambdas``
    only ever describes the states handed to ONE sub-run. An adaptive top-up may
    legitimately be given nothing but λ=0 states, and deciding "no ladder here"
    from that subset is what produced chignolin_7's defect: the λ=0 replicas kept
    the shared calibration's full ``k0`` instead of ``λ·k0max = 0`` and so ran
    boosted, while their channel energies went unrecorded. The registry
    (``StateRegistry.write_state_csv``) knows every state in the campaign, so ask
    it instead.

    The walk stops at ``adaptive_production/`` deliberately: several campaigns
    commonly share one parent directory, and an unbounded walk would let one
    campaign's ladder switch on another's boost recording.
    """
    base = Path(out_dir)
    for candidate in (base, *base.parents):
        for name in CAMPAIGN_LADDER_REGISTRY_NAMES:
            path = candidate / name
            if not path.exists():
                continue
            try:
                with path.open(newline="") as handle:
                    for row in csv.DictReader(handle):
                        raw = row.get("gamd_lambda")
                        if raw in (None, "", "None"):
                            continue
                        try:
                            lam = float(raw)
                        except (TypeError, ValueError):
                            continue
                        if lam > 0.0:
                            return (path.name, lam)
            except OSError:
                continue
        if candidate.name == "adaptive_production":
            break
    return None


def resolve_ladder_active(state_lambdas, out_dir) -> bool:
    """Is the λ-ladder active for this run?

    True when this sub-run holds a boosted state itself (the original test), OR
    when the campaign registry above it says the campaign runs a ladder. The
    second clause is what keeps an all-λ=0 top-up on the ladder code path, so its
    replicas get ``k0 = 0·k0max`` and its samples carry ``v_pep``/``v_dih``.

    Deliberately NOT true for a plain umbrella/REUS or plain-GaMD campaign: the
    ladder path makes every logged sample read two extra Context energies, which
    would be pure cost for a boost that is identically zero.
    """
    lam_arr = np.asarray(state_lambdas, dtype=float)
    if lam_arr.size and bool(np.any(lam_arr > 0.0)):
        return True
    return campaign_ladder_registry_lambda(out_dir) is not None


def _warn_if_ladder_was_zeroed(previous, new, where: str) -> bool:
    """Loudly warn when a previously non-zero λ-ladder re-derives to all zeros.

    Both post-drop re-derives pass ``existing=None`` on purpose (a pre-drop
    vector is not safe to reuse across an arbitrary, non-suffix index drop),
    and ``_derive_state_gamd_lambdas``' own fallback-of-last-resort is all
    zeros. On a cold resume (``--resume`` with no replica checkpoints) the
    window metadata that carried the rungs may not be reconstructible, so a
    restored ladder can be silently flattened -- and ``_persist_state_gamd_lambdas``
    then writes the zeros back into run_manifest.json, destroying the record
    of what the campaign was.

    The run stays SELF-consistent afterwards (every state is λ=0, i.e. plain
    umbrella), so this is a warning, not an error -- but it silently changes
    what the run is, which must never be invisible. 2026-09-07 final review,
    I3; the full fix (restoring the ladder rather than re-deriving it) has its
    own ticket. Returns True when it warned.
    """
    def _vals(x):
        if x is None:
            return []
        try:
            return [float(v or 0.0) for v in x]
        except (TypeError, ValueError):
            return []
    prev = _vals(previous)
    now = _vals(new)
    if not any(v > 0.0 for v in prev) or any(v > 0.0 for v in now):
        return False
    print(
        f"WARNING: λ-ladder LOST at {where}: {sum(1 for v in prev if v > 0.0)} of "
        f"{len(prev)} states carried gamd_lambda > 0 before this re-derive and NONE "
        f"do after it. The run continues as plain umbrella sampling (self-consistent, "
        f"but no longer a ladder), and run_manifest.json will record the zeros. This "
        f"is the known cold-resume/auto-drop re-derive gap -- verify the ladder before "
        f"trusting any PMF from this run."
    )
    return True


def _derive_state_gamd_lambdas(window_metadata: Optional[dict], n: int, existing=None) -> list[float]:
    """Derive one gamd_lambda per surviving window, filter/drop-safe.

    ``window_metadata["normalized_rows"]`` is the source of truth whenever it is
    usable: present, aligned to exactly ``n`` surviving windows, and every row
    carries its own ``gamd_lambda``. Reindexing operations that drop/reorder
    windows (filter_explicit_2d_windows_by_seed_reachability,
    drop_bad_us_windows_and_rebuild's _resubscript_normalized_rows) already
    subset+renumber normalized_rows correctly, so reading gamd_lambda back out
    of it here is safe across any such reindex.

    ``existing`` is used ONLY when normalized_rows cannot be trusted, and ONLY
    if it is already exactly length ``n`` -- a caller must not pass a
    pre-reindex value across a filter/drop boundary where the alignment cannot
    be verified from here; pass ``existing=None`` at any such call site.

    Zeros (ladder inactive) is the fallback of last resort, never a guess.
    """
    rows = (window_metadata or {}).get("normalized_rows") or []
    if rows and len(rows) == int(n) and all("gamd_lambda" in r for r in rows):
        return [float(r.get("gamd_lambda", 0.0) or 0.0) for r in rows]
    existing_list = list(existing or [])
    if len(existing_list) == int(n):
        return existing_list
    return [0.0] * int(n)

def _reload_state_gamd_lambdas_on_resume(args, out_dir: Path) -> None:
    """Restore the frozen λ-ladder from run_manifest.json on ``--resume``.

    Controller ruling (Task 2's implementer found this and deferred it to
    Task 9): ``--resume`` never persisted ``args.state_gamd_lambdas`` across a
    restart on its own. ``_derive_state_gamd_lambdas`` below is only as good
    as the window_metadata it is handed on the resumed path, and nothing
    upstream guarantees that still carries a per-row ``gamd_lambda`` after a
    restart (an older run directory, or a checkpoint manifest's
    window_metadata predating the rung dimension). Left uncorrected,
    ``args.state_gamd_lambdas`` would come out unset/empty and every replica
    would silently run at λ=0 -- silently, because an all-zero ladder is also
    the valid "ladder disabled" state, so nothing downstream would complain.
    Spec §3.6 requires the ladder to stay frozen for the whole campaign,
    which includes surviving a restart, so it is read back from wherever it
    was first frozen: ``run_manifest.json``'s ``method_settings`` (see
    ``gareus/provenance.py:_method_settings``), which is written once at
    campaign start (``initialize_run_manifest``) and never overwritten by a
    resume's ``update_run_manifest`` patches.

    CALL ORDER IS LOAD-BEARING: this must run BEFORE ``_derive_state_gamd_lambdas``,
    never after. A review caught this call sitting after
    ``_derive_state_gamd_lambdas`` in an earlier revision, which made it dead
    code on every real run: that function's own fallback-of-last-resort is
    ``[0.0] * n``, a non-empty (hence truthy) list, so by the time a
    post-derive reload call inspected ``args.state_gamd_lambdas`` it always
    looked already-populated and the manifest was never read. Called BEFORE
    ``_derive_state_gamd_lambdas`` (its real call site, in ``run_gareus``),
    this only ever observes ``args.state_gamd_lambdas`` in its true
    pre-derivation state: unset on the fast-resume/choose_windows paths, or
    already populated straight from the window table on the explicit-2D path
    (which this function correctly leaves alone) -- either way,
    ``_derive_state_gamd_lambdas``'s own ``existing=`` parameter then
    naturally receives whatever this function restored.

    A no-op unless ``args.resume`` is set and ``args.state_gamd_lambdas`` is
    still unset/empty at the point it is called -- an already-populated
    ladder (from an explicit window table, or a caller that set it
    explicitly) is never overridden. This is a plain truthiness check, not a
    content check: an all-zero ``args.state_gamd_lambdas`` set by a caller is
    a legitimate disabled ladder and must never be treated as "unset" and
    silently replaced.
    """
    if not bool(getattr(args, "resume", False)):
        return
    if getattr(args, "state_gamd_lambdas", None):
        return
    manifest = read_json_file(Path(out_dir) / "run_manifest.json", {}) or {}
    method_settings = (manifest or {}).get("method_settings", {}) or {}
    lambdas = method_settings.get("state_gamd_lambdas")
    if lambdas:
        args.state_gamd_lambdas = [float(x) for x in lambdas]

def _persist_state_gamd_lambdas(args, out_dir: Path) -> None:
    """Patch the CURRENT args.state_gamd_lambdas into run_manifest.json's
    method_settings, overwriting any earlier snapshot written by a previous
    call.

    Review finding: the first cut of this patch was a single call placed
    right after the initial _derive_state_gamd_lambdas() in run_gareus, but
    args.state_gamd_lambdas is mutated TWICE afterwards on some runs --
    the --max-replicas truncation (`args.state_gamd_lambdas =
    list(args.state_gamd_lambdas)[:_max_replicas]`) and the post-pull US
    auto-drop re-derive (`args.state_gamd_lambdas =
    _derive_state_gamd_lambdas(window_metadata, len(centers_a),
    existing=None)`) -- and neither re-patched the manifest. Persisting the
    PRE-truncation/PRE-drop value is exactly what
    _derive_state_gamd_lambdas' own docstring warns a caller never to do
    with `existing=` across a filter/drop boundary, because
    _reload_state_gamd_lambdas_on_resume feeds this same manifest field
    back in as `existing=` on the next --resume: a stale, wrong-length
    snapshot there either falls back to _derive_state_gamd_lambdas' own
    [0.0]*n (silently disabling the ladder) or, worse, if the post-drop
    count happens to match, resumes with a misaligned per-state λ.

    Call this again at EVERY point in run_gareus that reassigns
    args.state_gamd_lambdas, not just once after the initial derive --
    the last call before production starts is what ends up in the
    manifest, so it must be the call closest to (after) the final
    mutation, not the first available opportunity.
    """
    update_run_manifest(out_dir, {"method_settings": {"state_gamd_lambdas": list(getattr(args, "state_gamd_lambdas", None) or [])}})

def write_window_assignment_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "window", "center_A", "k_kcal_mol_A2", "k_kj_mol_nm2",
        "harmonic_sigma_A", "spacing_to_previous_A", "spacing_to_next_A",
        "primary_cv", "primary_cv_label", "primary_cv_units", "primary_center", "primary_k",
        "primary_k_units", "primary_openmm_k", "primary_openmm_k_units",
        "primary_harmonic_sigma", "legacy_primary_cv_column_names",
        "secondary_cv_center", "secondary_cv_k_kcal_mol", "secondary_cv_k_kj_mol",
        "gamd_lambda",
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
    gamd_lambdas=None,
) -> dict:
    """Write sparse-safe explicit window metadata files for MBAR/reweighting."""
    out_dir = Path(out_dir)
    rows = explicit_window_analysis_rows(
        centers_a, k_list, secondary_cv_centers, secondary_cv_k_kcal_list,
        secondary_cv_metadata=secondary_cv_metadata,
        window_metadata=window_metadata,
        gamd_lambdas=gamd_lambdas,
    )
    csv_path = out_dir / "umbrella_explicit_windows.csv"
    json_path = out_dir / "umbrella_explicit_windows.json"
    fieldnames = [
        "window", "gamd_lambda", "distance_center_A", "distance_center_nm", "distance_k_kcal_mol_A2", "distance_k_kj_mol_nm2",
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

def _connected_components_count(n_nodes: int, edges: list[dict]) -> int:
    """Union-find component count over an explicit-2D neighbor-graph edge list."""
    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for edge in edges:
        wi, wj = int(edge.get("wi", -1)), int(edge.get("wj", -1))
        if 0 <= wi < n_nodes and 0 <= wj < n_nodes:
            ri, rj = find(wi), find(wj)
            if ri != rj:
                parent[ri] = rj

    return len({find(i) for i in range(n_nodes)}) if n_nodes else 0



# ---------------------------------------------------------------------------
# epoch_window_map.csv repair after a window drop
#
# An adaptive-production phase's epoch_window_map.csv is written by the registry
# INTO the phase directory BEFORE the sub-run starts: row i says "local window
# index i is global state state_id", with that state's restraint params.  It is
# the only thing that tells analyze_gareus_mbar.py which umbrella Hamiltonian a
# sample logged under window_id = i was actually generated by.
#
# TWO independent drop paths renumber a phase's windows after that map is
# written, and both invalidate it the same way:
#
#   1. the PRE-pull seed-reachability filter (seeding.py's
#      filter_explicit_2d_windows_by_seed_reachability, recorded as
#      "dropped_unreachable_windows"), which runs while the window table is
#      being loaded, and
#   2. the POST-pull auto-drop quality gate (drop_bad_us_windows_and_rebuild
#      below, recorded as "dropped_post_pull_bad_windows").
#
# Both can fire in the SAME phase, in that order - in which case the second one
# legitimately hands us indices in the ALREADY-COMPACTED space the first one
# produced, and applying it is correct.  What must be impossible is applying the
# same drop set twice.  See _epoch_window_map_rewrite_already_applied for the
# invariant that makes that checkable rather than a matter of call ordering.
# ---------------------------------------------------------------------------

_EPOCH_WINDOW_MAP_NAME = "epoch_window_map.csv"
_EPOCH_WINDOW_MAP_LEDGER_NAME = "epoch_window_map_rewrites.json"
_EPOCH_WINDOW_MAP_LEDGER_SCHEMA = "gareus_epoch_window_map_rewrite_ledger_v1"
# The ledger file carries two independent lists.  They are not one list, and
# `_append_epoch_window_map_reorder_ledger` is where the reason lives.
_EPOCH_WINDOW_MAP_LEDGER_APPLIED_KEY = "applied"
_EPOCH_WINDOW_MAP_LEDGER_REORDERS_KEY = "reorders"


def _int_or_none(value) -> Optional[int]:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def _read_epoch_window_map_rows(path: Path) -> tuple[list[str], list[dict]]:
    """``(fieldnames, rows)`` of an epoch_window_map.csv.

    Field names come from the header, not from the surviving rows: dropping every
    window is reachable (``--us-auto-drop-max-fraction 1.0``) and would leave
    nothing to read column names off.
    """
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(r) for r in reader]


def _epoch_window_map_state_ids(rows) -> list:
    """Per-row ``state_id`` in row order (``None`` where unparseable)."""
    return [_int_or_none(r.get("state_id")) for r in rows]


def _write_epoch_window_map_rows(path: Path, fieldnames, rows) -> None:
    """Atomically replace an epoch_window_map.csv with `rows`."""
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except Exception:
            pass
        raise


def _read_epoch_window_map_ledger_list(out_dir: Path, key: str) -> list:
    """One of the ledger file's two entry lists, oldest first.

    A missing/damaged ledger reads as empty: the ledger only ever *adds* a
    refusal (it can never be the reason a needed rewrite happens), so losing it
    degrades to the pre-ledger behavior rather than to a wrong map.
    """
    payload = read_json_file(Path(out_dir) / _EPOCH_WINDOW_MAP_LEDGER_NAME, None)
    if not isinstance(payload, dict):
        return []
    entries = payload.get(key)
    return [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []


def _read_epoch_window_map_rewrite_ledger(out_dir: Path) -> list:
    """Applied-DROP ledger entries for this phase, oldest first.

    Deliberately reads only ``applied`` and never ``reorders``: the idempotence
    rule in `_epoch_window_map_rewrite_already_applied` is a statement about drop
    sets, and a reorder has no drop set to key on.  See
    `_append_epoch_window_map_reorder_ledger` for why the two lists are kept
    apart rather than merged.
    """
    return _read_epoch_window_map_ledger_list(out_dir, _EPOCH_WINDOW_MAP_LEDGER_APPLIED_KEY)


def _read_epoch_window_map_reorder_ledger(out_dir: Path) -> list:
    """Applied-REORDER records for this phase, oldest first.

    A pure log.  Nothing branches on it - see
    `_append_epoch_window_map_reorder_ledger`.
    """
    return _read_epoch_window_map_ledger_list(out_dir, _EPOCH_WINDOW_MAP_LEDGER_REORDERS_KEY)


def _append_epoch_window_map_ledger_entry(out_dir: Path, key: str, entry: dict) -> bool:
    """Append one record to one of the ledger's two lists.  True when it landed.

    Rewrites BOTH lists every time, not only the one being appended to: the file
    is replaced wholesale by `write_json`, so an appender that rebuilt the payload
    from its own list alone would silently delete the other one - which is exactly
    how the reorder record (added after the drop record already existed) would go
    missing on any phase that did both.

    Never raises: a phase that has already corrected its map must not die because
    the bookkeeping file could not be written (the map itself is the artifact that
    matters; the drop list only guards against a *second* application).  Returns
    whether the record actually reached disk, so a caller that reports "recorded"
    in its own summary reports what happened rather than what it intended.
    """
    path = Path(out_dir) / _EPOCH_WINDOW_MAP_LEDGER_NAME
    try:
        lists = {
            _EPOCH_WINDOW_MAP_LEDGER_APPLIED_KEY: _read_epoch_window_map_rewrite_ledger(out_dir),
            _EPOCH_WINDOW_MAP_LEDGER_REORDERS_KEY: _read_epoch_window_map_reorder_ledger(out_dir),
        }
        lists[key].append(dict(entry))
        write_json(path, _json_ready({
            "schema": _EPOCH_WINDOW_MAP_LEDGER_SCHEMA,
            "description": (
                "What has already been applied to this phase's epoch_window_map.csv, in order. "
                "`applied` holds DROP SETS: each entry is keyed by (n_windows_before, "
                "dropped_window_indices) - the index space the drop set was expressed in - plus "
                "the state_id sequence the rewrite produced. A rewrite whose key is already "
                "present AND whose recorded surviving state_ids still match the map on disk has "
                "already been applied and is refused. `reorders` holds REORDERINGS onto the "
                "surviving-window table, which remove no rows and so have no such key; they are a "
                "record only and are never consulted to refuse anything."
            ),
            "map": str(Path(out_dir) / _EPOCH_WINDOW_MAP_NAME),
            _EPOCH_WINDOW_MAP_LEDGER_APPLIED_KEY: lists[_EPOCH_WINDOW_MAP_LEDGER_APPLIED_KEY],
            _EPOCH_WINDOW_MAP_LEDGER_REORDERS_KEY: lists[_EPOCH_WINDOW_MAP_LEDGER_REORDERS_KEY],
        }))
        return True
    except Exception as exc:
        print(
            f"WARNING: could not record the epoch_window_map.csv rewrite in {path}: {exc}. "
            "The map itself was corrected; only the idempotence ledger is missing."
        )
        return False


def _append_epoch_window_map_rewrite_ledger(out_dir: Path, entry: dict) -> None:
    """Append one applied-DROP record next to the map it describes."""
    _append_epoch_window_map_ledger_entry(out_dir, _EPOCH_WINDOW_MAP_LEDGER_APPLIED_KEY, entry)


def _append_epoch_window_map_reorder_ledger(out_dir: Path, entry: dict) -> bool:
    """Append one applied-REORDER record.  True when it is on disk.

    A separate list from ``applied``, and the separation is load-bearing rather
    than tidiness.  `_epoch_window_map_rewrite_already_applied` refuses a request
    whose ``source`` matches an entry's while the map still holds that entry's
    ``surviving_state_ids``; a reorder entry written under source
    ``surviving_window_table`` would therefore be able to refuse a later GENUINE
    drop rewrite from that same source - a guard turning into the bug it guards
    against.  The drop list's idempotence rule also rests on "a successful rewrite
    always removes at least one row, so its key can never legitimately recur for
    one map file", which a row-preserving reorder falsifies outright: two reorders
    of one phase would write byte-identical entries.

    Why record it at all, given a reorder is idempotent by projection: because
    once it has run, the map and the window table AGREE and nothing in either
    artifact says they ever did not.  The rows really were rewritten in place, and
    every sample already logged for those windows was logged against the old
    order, so an analysis produced before the reorder is invalid - which is the
    one fact this record preserves and the artifacts alone cannot.  Until this
    existed the only record was the summary `run_gareus` stores in the phase's
    gareus_metadata.json, which a process killed between the map write and that
    store never reaches: the map would be reordered on disk, the ledger empty, and
    a resumed repair would return "consistent" with nothing anywhere saying a
    reorder had happened.  The exposure that remains is one filesystem operation
    wide (a kill between the map write and this append), not the whole remainder
    of phase setup.
    """
    return _append_epoch_window_map_ledger_entry(
        out_dir, _EPOCH_WINDOW_MAP_LEDGER_REORDERS_KEY, entry)


def _epoch_window_map_rewrite_already_applied(
    out_dir: Path,
    dropped: list,
    n_windows_before: int,
    current_state_ids: list,
    source: str,
) -> Optional[dict]:
    """The ledger entry proving this drop is already reflected in the map, if any.

    This is the idempotence rule, spelled out because "the same drop set can never
    be applied twice" has to be a checkable property of what is on disk, not an
    ordering convention between call sites that a future caller can break.

    Every entry records the state_id sequence its rewrite PRODUCED, so
    ``current_state_ids == entry["surviving_state_ids"]`` answers "does the map
    still hold that rewrite, or has it been replaced since?".  That conjunct is
    load-bearing rather than defensive padding: a phase interrupted and re-run
    WITHOUT a usable production checkpoint gets a fresh identity map written by the
    adaptive driver and then legitimately re-runs the same filter with the same
    drop set.  Refusing on the drop set alone would call that a replay and leave
    the fresh map stale - the exact bug this whole mechanism exists to prevent.

    Given the map still holds a previous rewrite, two things mark a new request as
    a replay of it rather than a genuine further drop:

    * **Same drop path** (``source``).  Each drop path fires at most once over a
      given map: the pre-pull reachability filter runs once while the window table
      is loaded, the post-pull auto-drop runs once after pulling.  A second
      request from the same path, against a map that still holds that path's own
      rewrite, is that path being owned by two call sites.  This is the clause
      that closes the hazard the other two guards cannot see: a second owner that
      derives ``n_windows_before`` from the *current* (already compacted) map
      instead of the pre-drop window count presents a key nothing else can
      distinguish from a legitimate later drop, because both counts shifted
      together.
    * **Same (n_windows_before, dropped) key**, whatever the label.  A successful
      rewrite always removes at least one row, so this pair - "there were this
      many rows, remove these positions" - can never legitimately recur for one
      map file; two drops in one phase necessarily carry different
      ``n_windows_before``.  This catches a differently-labelled duplicate.

    Conversely both drop paths firing on the same phase (filter first, then
    auto-drop on its survivors) is legitimate and must go through: different
    ``source``, and a ``n_windows_before`` that is the compacted count rather than
    the original one.

    Not covered here, on purpose: an *unlogged* prior application (something
    rewrote the map without recording it) leaves fewer rows than
    ``n_windows_before``, which the row-count guard in
    rewrite_epoch_window_map_after_drop rejects.  The two are complementary.
    """
    for entry in _read_epoch_window_map_rewrite_ledger(out_dir):
        recorded_after = entry.get("surviving_state_ids")
        if not isinstance(recorded_after, list):
            continue
        if [_int_or_none(x) for x in recorded_after] != list(current_state_ids):
            continue  # the map is no longer what this entry left behind
        if str(entry.get("source", "")) == str(source):
            return entry
        try:
            entry_dropped = [int(i) for i in (entry.get("dropped_window_indices") or [])]
            entry_before = int(entry.get("n_windows_before"))
        except (TypeError, ValueError):
            continue
        if entry_dropped == list(dropped) and entry_before == int(n_windows_before):
            return entry
    return None


def rewrite_epoch_window_map_after_drop(
    out_dir: Path,
    dropped_window_indices,
    n_windows_before: int,
    source: str = "post_pull_auto_drop",
) -> Optional[dict]:
    """Rewrite this phase's ``epoch_window_map.csv`` after windows were dropped.

    `dropped_window_indices` are positions in the window set as it stood when
    there were `n_windows_before` windows - i.e. in whatever index space the
    calling drop path works in.  `source` names that path, for the ledger and the
    log line; see this section's header comment for the two real ones.

    A drop renumbers every window-indexed array 0..N-1 around the survivors, which
    silently invalidates the map: local window 20 keeps claiming to be state 20
    while it is physically running what used to be window 21.  On the real run
    this was found on (chignolin_6, see docs/chignolin_6_low_ess_root_cause.md)
    that mis-attributed ~1.82M of 12.1M samples (15% of the campaign), scored
    state 21's data against a sign-flipped CV2 center (2,012 kT of fabricated
    bias), turned state 20 into a phantom duplicate of 21 that was never actually
    sampled, and pooled three different Hamiltonians into state 22's slot - base
    MBAR ESS 0.74%, health verdict FAIL.  Nothing downstream can detect or repair
    this reliably after the fact, so the map has to be corrected here, while the
    drop set is still known.

    Drops the dropped rows, renumbers the survivors' ``epoch_window`` to 0..N-1 in
    their original order, and keeps every surviving row's own ``state_id`` and
    center/k values verbatim - the index moves, the physics does not.  Works for a
    topup phase's non-identity subset map (dense ``epoch_window``, sparse
    ``state_id``) exactly as it does for a baseline's identity map.

    Returns a summary dict, or ``None`` when there is no map to rewrite (a plain
    non-adaptive-production run has none; that is not an error).  Refuses -
    loudly, without touching the file - rather than guessing whenever the
    positional identity this rewrite depends on is not established, or whenever
    the ledger shows this exact drop set is already reflected in the map on disk;
    a stale map is recoverable by re-running the analysis, a wrongly-rewritten one
    is not, and raising here would kill a production phase that is otherwise fine.
    """
    out_dir = Path(out_dir)
    path = out_dir / _EPOCH_WINDOW_MAP_NAME
    if not path.exists():
        return None

    dropped = sorted({int(i) for i in (dropped_window_indices or [])})
    if not dropped:
        # Reachable: drop_bad_us_windows_and_rebuild's connectivity restore can
        # hand back every flagged window, leaving nothing to drop.  Rewriting
        # would be a no-op on the file but would log a zero-row-change ledger
        # entry, breaking the strictly-decreasing row count the idempotence
        # invariant above is built on.
        return {"status": "no_drop", "path": str(path), "source": str(source)}

    try:
        fieldnames, rows = _read_epoch_window_map_rows(path)
    except Exception as exc:
        print(f"WARNING: could not read {path} to correct it after the window drop: {exc}")
        return {"status": "skipped_unreadable", "path": str(path), "error": str(exc), "source": str(source)}

    if not fieldnames or "epoch_window" not in fieldnames:
        print(
            f"WARNING: {path} has no 'epoch_window' column; leaving it untouched after the window drop. "
            "MBAR sample-to-state attribution for this phase may be wrong."
        )
        return {"status": "skipped_missing_epoch_window_column", "path": str(path), "source": str(source)}

    already = _epoch_window_map_rewrite_already_applied(
        out_dir, dropped, int(n_windows_before), _epoch_window_map_state_ids(rows), str(source),
    )
    if already is not None:
        # Not a warning: two owners both correctly identifying the same drop is
        # benign as long as only one of them writes, which is exactly what this
        # branch enforces.
        print(
            f"    epoch_window_map.csv: drop set {dropped} (of {int(n_windows_before)} windows) is already "
            f"applied to {path} by {already.get('source', 'an earlier rewrite')}; not applying it again."
        )
        return {
            "status": "skipped_already_applied",
            "path": str(path),
            "source": str(source),
            "applied_by": already.get("source"),
            "dropped_window_indices": dropped,
            "n_windows_before": int(n_windows_before),
            "n_windows_after": int(len(rows)),
        }

    if len(rows) != int(n_windows_before):
        # The map must cover exactly the pre-drop window set for row position to
        # mean "local window index".  If it does not, the drop indices index into
        # something else and any rewrite would corrupt a map that is merely stale.
        # This is also the guard that catches an UNLOGGED prior application of
        # this same drop set (see _epoch_window_map_rewrite_already_applied).
        print(
            f"WARNING: {path} has {len(rows)} rows but {int(n_windows_before)} windows existed before the "
            f"{source} drop; cannot establish which row is which local window, leaving it untouched. "
            "MBAR sample-to-state attribution for this phase will be wrong - correct the map by hand "
            "before analyzing this run."
        )
        return {
            "status": "skipped_row_count_mismatch",
            "path": str(path),
            "source": str(source),
            "n_map_rows": int(len(rows)),
            "n_windows_before": int(n_windows_before),
        }

    try:
        map_indices = [int(float(r.get("epoch_window", ""))) for r in rows]
    except Exception:
        map_indices = []
    if map_indices != list(range(len(rows))):
        print(
            f"WARNING: {path}'s epoch_window column is not a dense 0..{len(rows) - 1} range "
            f"({map_indices[:8]}...); leaving it untouched after the window drop. "
            "MBAR sample-to-state attribution for this phase may be wrong."
        )
        return {"status": "skipped_non_dense_epoch_window", "path": str(path), "source": str(source)}

    if any(i < 0 or i >= len(rows) for i in dropped):
        print(
            f"WARNING: {source} dropped window indices {dropped} fall outside this phase's "
            f"{len(rows)}-window map; leaving {path} untouched. MBAR sample-to-state attribution "
            "for this phase may be wrong."
        )
        return {"status": "skipped_dropped_index_out_of_range", "path": str(path), "dropped": dropped,
                "source": str(source)}

    dropped_set = set(dropped)
    dropped_state_ids = [_int_or_none(rows[i].get("state_id")) for i in dropped]
    survivors = [dict(r) for i, r in enumerate(rows) if i not in dropped_set]
    for new_idx, row in enumerate(survivors):
        row["epoch_window"] = int(new_idx)

    try:
        _write_epoch_window_map_rows(path, fieldnames, survivors)
    except Exception as exc:
        print(f"WARNING: failed to rewrite {path} after the window drop: {exc}")
        return {"status": "error", "path": str(path), "error": str(exc), "source": str(source)}

    summary = {
        "status": "rewritten",
        "path": str(path),
        "source": str(source),
        "n_windows_before": int(len(rows)),
        "n_windows_after": int(len(survivors)),
        "dropped_window_indices": dropped,
        "dropped_state_ids": [s for s in dropped_state_ids if s is not None],
        "surviving_state_ids": _epoch_window_map_state_ids(survivors),
    }
    _append_epoch_window_map_rewrite_ledger(out_dir, summary)

    print(
        f"WARNING [{source}]: rewrote {path} to match the surviving window set: "
        f"{len(rows)} -> {len(survivors)} rows; dropped local windows {dropped} "
        f"(state_id {dropped_state_ids}); surviving windows renumbered 0..{len(survivors) - 1} "
        "keeping their own state_id and restraint centers. Without this the MBAR loader would "
        "attribute this phase's samples to the wrong umbrella state."
    )
    return summary


def _dropped_unreachable_window_indices(*metadata_dicts) -> list:
    """Pre-pull reachability-filter drop indices, from the metadata it emits.

    filter_explicit_2d_windows_by_seed_reachability records its drop as
    ``{"dropped_unreachable_windows": [{"window": i, ...}, ...]}`` on BOTH the
    window metadata and the secondary-CV metadata it returns; take the first dict
    that carries it so this keeps working if either copy is dropped.
    """
    for meta in metadata_dicts:
        if not isinstance(meta, dict):
            continue
        entries = meta.get("dropped_unreachable_windows")
        if not entries:
            continue
        out = []
        for entry in entries:
            idx = _int_or_none(entry.get("window")) if isinstance(entry, dict) else _int_or_none(entry)
            if idx is not None:
                out.append(idx)
        if out:
            return sorted(set(out))
    return []


def rewrite_epoch_window_map_after_unreachable_filter(
    out_dir: Path,
    secondary_cv_metadata,
    window_metadata,
    n_windows_before: int,
) -> Optional[dict]:
    """Correct ``epoch_window_map.csv`` for the PRE-pull seed-reachability filter.

    The second, independent drop path (see this section's header comment).  Takes
    the metadata dicts exactly as filter_explicit_2d_windows_by_seed_reachability
    returns them and `n_windows_before` = how many windows the explicit table held
    BEFORE that filter ran, and returns None when the filter dropped nothing.

    This is the drop path that shifted chignolin_6's ``epoch_001/topup_004`` (map
    3 rows -> 2 real windows, local 0 -> state 21 and 1 -> state 22, not 20/21)
    and ``final/topup_003`` (2 -> 1, local 0 -> state 21, not 20).  Those phases
    have no ``dropped_post_pull_bad_windows`` record at all, so the post-pull
    rewrite could never have covered them.

    Exactly one site owns this call - run_gareus's explicit-2D window-loading
    block - deliberately: the filter itself lives in seeding.py and could rewrite
    the map there instead, but then both sites would apply the same drop set to
    the same file.  The ledger makes that impossible rather than merely unlikely,
    so a future second caller degrades to a no-op instead of double-compacting.
    """
    dropped = _dropped_unreachable_window_indices(window_metadata, secondary_cv_metadata)
    if not dropped:
        return None
    return rewrite_epoch_window_map_after_drop(
        out_dir, dropped, int(n_windows_before), source="pre_pull_seed_reachability_filter",
    )


def _first_map_float(row: dict, *keys) -> float:
    """First parseable float among `keys` on a CSV row dict, else NaN."""
    for key in keys:
        raw = row.get(key, "")
        if raw in ("", None, "None", "nan"):
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return float("nan")


def _phase_surviving_window_centers(out_dir: Path) -> Optional[list]:
    """``[(primary_center, secondary_center)]`` per real window from this phase's
    own ``umbrella_explicit_windows.csv``, or None when it has none.

    That table is written from the window arrays production actually runs with -
    post-filter, post-drop, and (on a checkpoint resume) rebuilt from the
    checkpoint manifest's own post-drop ``windows_A`` - so it is a direct record
    of the surviving window set rather than a record of what was removed.
    """
    path = Path(out_dir) / "umbrella_explicit_windows.csv"
    if not path.exists():
        return None
    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        print(f"WARNING: could not read {path}: {exc}")
        return None
    if not rows:
        return None
    return [(_first_map_float(r, "primary_center", "distance_center_A"),
             _first_map_float(r, "secondary_cv_center", "secondary_center")) for r in rows]


def _centers_close(a: float, b: float, tol: float = 1.0e-6) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= tol + tol * max(abs(a), abs(b))


def _match_map_rows_to_surviving_windows(map_rows: list, surviving_centers: list) -> Optional[list]:
    """The subsequence of `map_rows` whose centers are the surviving windows', in
    order - i.e. the map with its phantom rows removed.  None when any surviving
    window has no match left, which means these two artifacts do not describe the
    same window set and nothing may be inferred from them.

    Order-preserving on purpose: every drop path renumbers the survivors without
    reordering them, so the correct answer is always a subsequence and a greedy
    left-to-right scan can never reach back past a row it already consumed.  Both
    files are written from the same in-memory floats, so the tolerance only has to
    absorb text round-tripping - it is orders of magnitude below any real window
    spacing.
    """
    picked = []
    j = 0
    for c1, c2 in surviving_centers:
        while j < len(map_rows):
            row = map_rows[j]
            j += 1
            row_c1 = float(_first_map_float(row, "primary_center", "primary_cv_center", "distance_center_A"))
            row_c2 = float(_first_map_float(row, "secondary_center", "secondary_cv_center"))
            if _centers_close(row_c1, c1) and _centers_close(row_c2, c2):
                picked.append(row)
                break
        else:
            return None
    return picked


def _map_row_center_pairs(rows: list) -> list:
    """``[(primary_center, secondary_center)]`` per epoch_window_map.csv row."""
    return [(_first_map_float(r, "primary_center", "primary_cv_center", "distance_center_A"),
             _first_map_float(r, "secondary_center", "secondary_cv_center")) for r in rows]


def _has_duplicate_center_pairs(pairs) -> bool:
    """True when two windows share a (primary, secondary) restraint centre.

    Duplicate centres are legal - the explicit-2D loader deliberately keeps a
    user's duplicated rows as separate thermodynamic states - but they make
    centre matching ambiguous about WHICH of two identical rows a survivor is,
    and picking the wrong one attaches the wrong state_id (the bias would be
    identical; the MBAR state pooling would not).  Every caller must refuse to
    infer rather than guess.
    """
    keyed = [(round(a, 9) if not math.isnan(a) else "nan", round(b, 9) if not math.isnan(b) else "nan")
             for a, b in pairs]
    return len(set(keyed)) != len(keyed)


def _reorder_map_rows_onto_surviving_windows(rows: list, surviving_pairs: list) -> Optional[list]:
    """`rows` REORDERED so that row *i* carries surviving window *i*'s restraint
    centers, or None when the two artifacts do not list the same window set.

    The non-order-preserving sibling of `_match_map_rows_to_surviving_windows`,
    for the other defect.  That one is order-preserving because a drop renumbers
    survivors without reordering them, so its answer is a subsequence; a
    PERMUTATION (every window that really ran has a row, only the order is wrong)
    fails it outright, since a greedy left-to-right scan cannot reach back past a
    row it already consumed.

    The matching rule itself is NOT reimplemented here: it is
    `gareus/mbar_analysis/loaders_adaptive.py`'s
    `_permute_map_rows_onto_window_table`, the same function the load-side guard
    uses to decide the same question about the same two artifacts.  Two subtly
    different answers to "is this a permutation" is exactly the drift that made
    this a write-side/load-side divergence in the first place, so the write side
    borrows the rule and contributes only what is genuinely its own: this file's
    column-name aliasing (`primary_cv_center`/`distance_center_A`/
    `secondary_cv_center`, which the load side's reader does not carry).  The
    shadow rows below are that adapter - centers read here with the write side's
    aliases, handed over under the two names the shared matcher reads, and mapped
    back to the caller's real row objects by position.  A float's ``str()``
    round-trips exactly in Python 3 and ``str(nan)`` reads back as NaN, so a
    CV1-only phase's absent secondary axis survives the hand-off unchanged.

    The import is function-local to keep production's module-import graph free of
    the analysis subpackage (`loaders_adaptive` names this file in its own
    docstrings; a module-level edge would invite a cycle).  It cannot fail in
    practice - same distribution, and its own imports are a subset of the ones
    this module already performs at import time - but the caller treats an
    exception here as a refusal rather than letting it escape, because
    `repair_epoch_window_map_from_surviving_windows` documents "Never raises" and
    run_gareus calls it outside any try/except.
    """
    from .mbar_analysis.loaders_adaptive import _permute_map_rows_onto_window_table

    shadow = [{"primary_center": str(c1), "secondary_center": str(c2), "_row_index": i}
              for i, (c1, c2) in enumerate(_map_row_center_pairs(rows))]
    picked = _permute_map_rows_onto_window_table(shadow, list(surviving_pairs))
    if picked is None:
        return None
    return [rows[int(s["_row_index"])] for s in picked]


def _select_map_rows_onto_surviving_windows(rows: list, surviving_pairs: list) -> tuple:
    """``(picked, problem)`` - one map row per surviving window, in the TABLE's
    order, when that assignment is forced; ``(None, phrase)`` when it is not.

    For the compound stale map: MORE rows than the phase has windows AND those
    windows listed in the wrong order.  `_match_map_rows_to_surviving_windows`
    cannot see it (order-preserving, so a permutation defeats it) and
    `_reorder_map_rows_onto_surviving_windows` cannot either (it requires every
    map row to be consumed, so the extra rows defeat it), which is why the shape
    used to come back from this file as ``skipped_centers_do_not_match``.

    Same borrowing rule as `_reorder_map_rows_onto_surviving_windows`, for the
    same reason: the matching itself is
    `gareus/mbar_analysis/loaders_adaptive.py`'s
    `_select_map_rows_onto_window_table`, so the two sides cannot answer "is this
    map derivable, and to what" differently - the divergence this whole
    write-side/load-side pairing exists to prevent.  What is genuinely local is
    this file's column-name aliasing, handed over through the same shadow rows.

    That matcher is FORCED rather than greedy - it refuses a window with two
    candidate rows, and two windows sharing one - and the strictness matters more
    here than in the equal-count reorder: with more rows than windows a candidate
    may be a PHANTOM, a state this phase dropped and never sampled, so an
    arbitrary pick can hand a real window's samples to a state that never ran.
    That is the chignolin_6 failure itself, not a hypothetical.  The
    duplicate-centre abstain upstream already refuses most of that ground; this
    keeps the guarantee at the matcher rather than resting it on call ordering.
    """
    from .mbar_analysis.loaders_adaptive import _select_map_rows_onto_window_table

    shadow = [{"primary_center": str(c1), "secondary_center": str(c2), "_row_index": i}
              for i, (c1, c2) in enumerate(_map_row_center_pairs(rows))]
    picked, problem = _select_map_rows_onto_window_table(shadow, list(surviving_pairs))
    if picked is None:
        return None, problem
    return [rows[int(s["_row_index"])] for s in picked], None


def _reorder_positional_problem(out_dir: Path, rows: list) -> Optional[str]:
    """None when both records may be trusted POSITIONALLY, else a phrase saying
    which one may not and why.

    The gate that separates re-deriving a mapping from fabricating one, for
    BOTH repairs that take their output order from the window table: the
    equal-count reorder and the compound (longer AND out-of-order) selection.
    (The name said "equal_count" while it was only used by the first; the check
    itself never was equal-count-specific.)  Removing rows alone (the plain drop
    repair) keeps the map's own order, so a mis-ordered record makes that matcher
    refuse; taking the output order *from* the surviving-window table instead
    means that if the table's rows are not this phase's local windows 0..N-1 by
    position the rewritten map is wrong while looking right - the window sets
    still match, so nothing downstream would notice.  Symmetrically, a sample's local window index is resolved through the
    map's ``epoch_window`` VALUE, so position and value must agree there before
    moving a row to a new position means anything.

    Neither condition is producible by any current writer (`write_epoch_window_map`
    enumerates, every rewrite path renumbers 0..N-1, and
    `explicit_window_analysis_rows` writes ``window: int(i)`` over the surviving
    windows in order) and neither occurs on any real phase on disk.  They are
    checked because when they do not hold, which of the two orders is the
    local-window order becomes an inference rather than a reading, and this repair
    abstains rather than guesses.

    The table side delegates to the load-side reader that already owns this rule
    (`_read_phase_window_table`, which returns exactly such a phrase), for the same
    no-second-implementation reason as `_reorder_map_rows_onto_surviving_windows`;
    it reads the centers through the same column aliases in the same order as
    `_phase_surviving_window_centers`, so the two never disagree about the table's
    contents, only about whether its ordering may be trusted.
    """
    from .mbar_analysis.loaders_adaptive import _read_phase_window_table

    numbering = [_int_or_none(r.get("epoch_window")) for r in rows]
    if numbering != list(range(len(rows))):
        return (f"the map's epoch_window column is not numbered 0..{len(rows) - 1} in file order "
                f"({numbering[:8]}...), so its rows are not this phase's local windows by position")
    return _read_phase_window_table(out_dir)[1]


def repair_epoch_window_map_from_surviving_windows(
    out_dir: Path,
    source: str = "surviving_window_table",
) -> Optional[dict]:
    """Re-derive ``epoch_window_map.csv`` from this phase's surviving window table.

    The durable counterpart to the two drop-time rewrites above.  Those run inside
    the process that performs the drop; this one needs nothing but what is already
    on disk in the phase directory, so it also repairs a map that was overwritten
    with a fresh (un-pruned) identity map AFTER the drop had been applied.

    That is the normal case on a multi-day run: the driver rewrites
    ``epoch_window_map.csv`` from the registry every time it (re)starts a phase,
    and the registry still holds every dropped state as active - while a resumed
    run takes the production-checkpoint fast path and therefore never re-runs the
    pull or the drop that would have corrected the map again.  Without this the
    drop-time rewrite contributes nothing at all to an interrupted phase.

    Two shapes of stale map are repaired, and they are different repairs:

    * MORE rows than the phase has real windows - the drop case.  The phantom rows
      are removed and the survivors renumbered 0..N-1 in their original order.
      Destructive and non-idempotent, hence the ledger
      (``epoch_window_map_rewrites.json``); see
      `_epoch_window_map_rewrite_already_applied`.
    * EQUALLY many rows listing exactly this phase's windows in the WRONG ORDER -
      a permutation.  The rows are reordered onto the window table and renumbered;
      nothing is added or removed.  Recorded in the ledger file's separate
      ``reorders`` list, NOT in the ``applied`` drop list whose key is a drop set
      in an index space that a reorder does not have (see
      `_append_epoch_window_map_reorder_ledger`).  It needs no replay guard -
      ordering rows into the table's order is a projection, so applying it twice
      is applying it once, which a second call demonstrates by taking the
      order-preserving branch and returning ``consistent`` without writing - but
      it does need a record, because afterwards the two artifacts agree and
      nothing else on disk says they once did not.  The returned summary spells
      this out under ``ledger`` and is additionally stored in the phase's
      ``gareus_metadata.json`` by the caller.
  * MORE rows than real windows AND in the wrong order - the compound case.  The
      rows that carry the real windows' centres are selected in the table's order
      and renumbered, the rest are dropped, and because rows ARE removed this one
      does take an ``applied`` ledger entry like any other drop.  Repaired only
      when each real window has exactly one map row carrying its centres; see
      `_select_map_rows_onto_surviving_windows`.

    Every other equal-count outcome is a refusal, because the map cannot be
    re-derived from what is on disk: a genuinely different window set of the same
    size (``inconsistent_equal_count_centers_do_not_match`` - the one residual the
    count check alone cannot see), duplicate restraint centers making the match
    ambiguous, or a record whose own ordering cannot be trusted positionally
    (``inconsistent_equal_count_reorder_not_positionally_verifiable``).  A phase
    that never dropped or reordered anything is never touched at all.

    Returns None when there is no map or no surviving-window table to work from,
    otherwise a status dict.  Never raises.
    """
    out_dir = Path(out_dir)
    path = out_dir / _EPOCH_WINDOW_MAP_NAME
    if not path.exists():
        return None
    surviving = _phase_surviving_window_centers(out_dir)
    if surviving is None:
        return None

    try:
        fieldnames, rows = _read_epoch_window_map_rows(path)
    except Exception as exc:
        print(f"WARNING: could not read {path} to check it against the surviving window table: {exc}")
        return {"status": "skipped_unreadable", "path": str(path), "error": str(exc), "source": str(source)}

    if len(rows) == len(surviving):
        # An agreeing row COUNT is not an agreeing window SET.  Two attempts at the
        # same phase can drop equally many but different windows - attempt 1 drops
        # window a, attempt 2 (after an interruption re-pulls) drops window b - and
        # the map left behind by attempt 1 then has exactly the right number of rows
        # while describing the wrong states from index min(a,b) onwards.  That is
        # documented residual #1 of the drop-rewrite fix, and the matcher that
        # settles it is already in this file and already called on the mismatch
        # path, so checking here costs one extra scan over a handful of rows.
        #
        # Whether an equal-count map can be REPAIRED turns on which of two things
        # went wrong, and telling them apart is the whole job of this branch: a
        # different window set has no row at all for a window that really ran, so
        # there is nothing to re-derive its state_id FROM and the honest outcome is
        # a loud refusal; a permutation has every row and only the order wrong, so
        # the mapping is read off the two artifacts and the map is reordered in
        # place.  Either way the outcome is durable - the caller stores every
        # outcome except a verified pass into the phase's gareus_metadata.json, see
        # _epoch_window_map_repair_must_be_recorded - and the load-time guard in
        # gareus/mbar_analysis/loaders_adaptive.py makes the same call, from the
        # same evidence, for runs already on disk that this side can never reach.
        surviving_pairs = list(surviving)
        map_pairs_equal = _map_row_center_pairs(rows)
        if _has_duplicate_center_pairs(surviving_pairs) or _has_duplicate_center_pairs(map_pairs_equal):
            # Ambiguous by construction - see _has_duplicate_center_pairs.  Report
            # the count agreement, but do not claim the centres were verified.
            #
            # Also what shields the reorder attempt below: the shared permutation
            # matcher is a greedy first-unused-match, so it pairs two rows with
            # identical centers arbitrarily and would attach one of them the
            # other's state_id (identical bias, different MBAR state pooling).
            # Delete this abstain and a duplicate-centre phase gets reordered on a
            # coin flip - measured, not assumed.  Its position relative to the
            # reorder attempt is not what matters, only that it wins.
            return {"status": "consistent", "path": str(path), "n_windows": int(len(rows)),
                    "source": str(source), "centers_verified": False,
                    "centers_check": "skipped_duplicate_centers"}
        if _match_map_rows_to_surviving_windows(rows, surviving_pairs) is not None:
            return {"status": "consistent", "path": str(path), "n_windows": int(len(rows)),
                    "source": str(source), "centers_verified": True}

        # The map does not list this phase's windows in order.  Two very different
        # things produce that, and until this revision both were reported as the
        # first one - including the false explanation "two attempts at this phase
        # dropped equally many but different windows":
        #
        #   * a genuinely DIFFERENT window set of the same size.  Some window that
        #     really ran has no row at all, its state_id is recorded nowhere in
        #     these two artifacts, and nothing can re-derive one.  Refused, below.
        #   * a PERMUTATION - every window that really ran does have a row, only
        #     at the wrong position.  Then the mapping is read off rather than
        #     inferred: table row *i*'s centers are the restraint local window *i*
        #     really ran under, so the state that belongs to its samples is the one
        #     on the map row carrying those centers, wherever that row sits.
        #
        # The load-side guard (gareus/mbar_analysis/loaders_adaptive.py's
        # `_consistent_map_membership_notes`) already draws that distinction and
        # repairs the second case in memory from these same two files.  This side
        # declining a repair the other side proves derivable was a straight
        # divergence, so both now ask the same matcher the same question.
        try:
            reordered = _reorder_map_rows_onto_surviving_windows(rows, surviving_pairs)
            positional_problem = (
                None if reordered is None else _reorder_positional_problem(out_dir, rows)
            )
        except Exception as exc:
            # "Never raises" is this function's contract and run_gareus calls it
            # outside any try/except, so a failure of the one check that reaches
            # into another module degrades to a recorded refusal rather than
            # killing a production phase whose MD is fine.
            print(
                f"WARNING [{source}]: could not check {path} for a reordering of this phase's "
                f"windows: {exc}. Leaving it untouched; if the map IS misordered this phase's "
                "samples may be attributed to the wrong umbrella state - see "
                "docs/chignolin_6_low_ess_root_cause.md."
            )
            return {"status": "skipped_reorder_check_failed", "path": str(path),
                    "n_windows": int(len(rows)), "source": str(source),
                    "centers_verified": False, "error": str(exc)}

        if reordered is None:
            print(
                f"WARNING [{source}]: {path} has the right number of rows ({len(rows)}) for this phase's "
                f"{len(surviving)} windows, but its rows do not contain those windows' restraint centers "
                "in any order - so the map describes a DIFFERENT window set of the same size (two attempts "
                "at this phase dropped equally many but different windows). At least one window that really "
                "ran has no row in the map, so the map cannot be re-derived from itself and is left "
                "untouched; this phase's samples may be attributed to the wrong umbrella state - see "
                "docs/chignolin_6_low_ess_root_cause.md."
            )
            return {"status": "inconsistent_equal_count_centers_do_not_match", "path": str(path),
                    "n_windows": int(len(rows)), "source": str(source), "centers_verified": False}

        if positional_problem is not None:
            print(
                f"WARNING [{source}]: {path} lists the same window set as this phase's own post-drop "
                f"window table but in a different order, and that reordering cannot be applied because "
                f"{positional_problem}. Leaving the map untouched rather than guessing which of the two "
                "orders is the local-window order; this phase's samples may be attributed to the wrong "
                "umbrella state - see docs/chignolin_6_low_ess_root_cause.md."
            )
            return {"status": "inconsistent_equal_count_reorder_not_positionally_verifiable",
                    "path": str(path), "n_windows": int(len(rows)), "source": str(source),
                    "centers_verified": False, "positional_problem": str(positional_problem)}

        state_ids_before = _epoch_window_map_state_ids(rows)
        # How many rows physically changed position, counted from row identity
        # rather than from state_id: a map holding the same state_id on two rows
        # with different centers really would be reordered while `moved` below
        # stays empty, and reporting "0 windows moved" for a rewrite that did
        # happen is the same kind of untrue claim this branch exists to stop.
        n_rows_moved = sum(1 for i, row in enumerate(reordered) if row is not rows[i])
        renumbered = [dict(r) for r in reordered]
        for new_idx, row in enumerate(renumbered):
            row["epoch_window"] = int(new_idx)
        state_ids_after = _epoch_window_map_state_ids(renumbered)
        try:
            _write_epoch_window_map_rows(path, fieldnames, renumbered)
        except Exception as exc:
            print(f"WARNING: failed to reorder {path} onto the surviving window table: {exc}")
            return {"status": "error", "path": str(path), "error": str(exc), "source": str(source)}

        moved = [{"epoch_window": int(i), "state_id_before": before, "state_id_after": after}
                 for i, (before, after) in enumerate(zip(state_ids_before, state_ids_after))
                 if before != after]
        summary = {
            "status": "rewritten_reordered",
            "path": str(path),
            "source": str(source),
            "n_windows": int(len(rows)),
            "centers_verified": True,
            "n_rows_moved": int(n_rows_moved),
            "moved_windows": moved,
            "state_ids_before": state_ids_before,
            "state_ids_after": state_ids_after,
            # Deliberately no dropped_state_ids/dropped_window_indices: a reorder
            # removes nothing, and an empty pair of those keys next to an
            # unchanged row count is an invitation to answer "was this map
            # rewritten?" by comparing counts - the proxy that let this defect
            # sit unnoticed on the load side too.  The status and `moved_windows`
            # answer it directly.
        }
        # Recorded in the ledger file's OWN list, next to the map it rewrote, and
        # written here rather than left to the caller: run_gareus stores this
        # summary in the phase's gareus_metadata.json only after returning, so a
        # process killed in between used to leave the map reordered on disk with
        # nothing anywhere saying so - and a resumed repair then answers
        # "consistent", because by then the two artifacts agree.  See
        # `_append_epoch_window_map_reorder_ledger` for why this must not go in
        # the drop list, and why "idempotent, therefore no record needed" does
        # not follow.
        reorder_recorded = _append_epoch_window_map_reorder_ledger(out_dir, summary)
        summary["ledger"] = {
            "appended": bool(reorder_recorded),
            "list": _EPOCH_WINDOW_MAP_LEDGER_REORDERS_KEY,
            "reason": (
                "epoch_window_map_rewrites.json's `applied` list records DROP SETS, keyed by "
                "(n_windows_before, dropped_window_indices), and its idempotence rule rests on a "
                "successful rewrite always removing at least one row so that key can never "
                "legitimately recur for one map file. A reorder removes no rows, so it has no such "
                "key, two reorders of one phase would write byte-identical entries, and an entry "
                "under this source could refuse a later genuine drop rewrite from the same source. "
                "It is therefore recorded in the separate `reorders` list, which nothing consults "
                "to refuse anything. It needs no replay guard: ordering rows into the window "
                "table's order is a projection, so applying it twice is applying it once - a "
                "re-invocation finds the order-preserving matcher succeeding and returns "
                "'consistent' without writing. run_gareus additionally stores this summary in the "
                "phase's gareus_metadata.json, which is the record that is lost if the process dies "
                "before it gets there."
            ),
        }
        print(
            f"WARNING [{source}]: {path} lists exactly the windows this phase runs but "
            f"{n_rows_moved} of {len(rows)} sat at the wrong local window - the map is a PERMUTATION of "
            f"the right rows, not a different window set, so the correct mapping was re-derived from "
            f"umbrella_explicit_windows.csv by matching restraint centers and the map was REORDERED in "
            f"place. Corrected local->state mapping: "
            f"{', '.join(str(m['epoch_window']) + ': ' + str(m['state_id_before']) + '->' + str(m['state_id_after']) for m in moved)}. "
            "Every sample already written for those windows was logged against the old order, so any "
            "analysis of this phase from before this correction is invalid - see "
            "docs/chignolin_6_low_ess_root_cause.md."
        )
        return summary

    if len(rows) < len(surviving):
        print(
            f"WARNING: {path} has {len(rows)} rows but this phase runs {len(surviving)} windows; the map "
            "covers fewer windows than exist, which no window drop can produce. Leaving it untouched - "
            "samples from the uncovered windows cannot be attributed to a state at all."
        )
        return {"status": "skipped_map_shorter_than_window_set", "path": str(path),
                "n_map_rows": int(len(rows)), "n_windows": int(len(surviving)), "source": str(source)}

    # Duplicate (primary, secondary) window centers make center matching ambiguous
    # about WHICH of two identical rows survived - refuse rather than guess.  See
    # _has_duplicate_center_pairs for why they are legal in the first place.
    map_pairs = _map_row_center_pairs(rows)
    if _has_duplicate_center_pairs(surviving) or _has_duplicate_center_pairs(map_pairs):
        print(
            f"WARNING: {path} has {len(rows)} rows for {len(surviving)} real windows, but the window "
            "centers contain duplicates, so matching them cannot tell which of two identical rows "
            "survived. Leaving the map untouched; MBAR sample-to-state attribution for this phase may "
            "be wrong - see docs/chignolin_6_low_ess_root_cause.md."
        )
        return {"status": "skipped_duplicate_centers", "path": str(path),
                "n_map_rows": int(len(rows)), "n_windows": int(len(surviving)), "source": str(source)}

    matched = _match_map_rows_to_surviving_windows(rows, surviving)
    reordered_onto_table = False
    if matched is None:
        # Not "the map is unusable" - only "the extra rows cannot simply be
        # removed, because the rows that remain are also in the wrong order".
        # The compound stale map: longer AND permuted.  It is derivable exactly
        # when each surviving window has one and only one map row carrying its
        # restraint centers, and refused otherwise, which is what
        # `_select_map_rows_onto_surviving_windows` decides (and the load-side
        # guard decides identically, from the same two files - checked by running
        # both on one fixture, not argued).
        try:
            selected, select_problem = _select_map_rows_onto_surviving_windows(rows, surviving)
            positional_problem = (
                None if selected is None else _reorder_positional_problem(out_dir, rows)
            )
        except Exception as exc:
            # Same contract as the equal-count reorder's own check: "Never raises"
            # and run_gareus calls this outside any try/except, so a failure of
            # the one check that reaches into another module degrades to a
            # recorded refusal rather than killing a production phase.
            print(
                f"WARNING [{source}]: could not check {path} for a reordering of this phase's "
                f"windows: {exc}. Leaving it untouched; if the map IS misordered this phase's "
                "samples may be attributed to the wrong umbrella state - see "
                "docs/chignolin_6_low_ess_root_cause.md."
            )
            return {"status": "skipped_reorder_check_failed", "path": str(path),
                    "n_map_rows": int(len(rows)), "n_windows": int(len(surviving)),
                    "source": str(source), "centers_verified": False, "error": str(exc)}
        if selected is None:
            print(
                f"WARNING: {path} has {len(rows)} rows for {len(surviving)} real windows, but its rows do not "
                "contain the surviving windows' restraint centers in order, and they cannot be matched onto "
                f"those windows out of order either: {select_problem}. The correct local-window -> state "
                "mapping therefore cannot be re-derived. Leaving the map untouched; MBAR sample-to-state "
                "attribution for this phase may be wrong - see docs/chignolin_6_low_ess_root_cause.md."
            )
            return {"status": "skipped_centers_do_not_match", "path": str(path),
                    "n_map_rows": int(len(rows)), "n_windows": int(len(surviving)),
                    "source": str(source), "centers_verified": False,
                    "select_problem": str(select_problem)}
        if positional_problem is not None:
            print(
                f"WARNING [{source}]: {path} has more rows than this phase's {len(surviving)} windows and "
                f"lists those windows out of order, and the correction cannot be applied because "
                f"{positional_problem}. Leaving the map untouched rather than guessing which of the two "
                "orders is the local-window order; this phase's samples may be attributed to the wrong "
                "umbrella state - see docs/chignolin_6_low_ess_root_cause.md."
            )
            return {"status": "skipped_reorder_not_positionally_verifiable", "path": str(path),
                    "n_map_rows": int(len(rows)), "n_windows": int(len(surviving)),
                    "source": str(source), "centers_verified": False,
                    "positional_problem": str(positional_problem)}
        matched = selected
        reordered_onto_table = True

    # By row identity rather than by an order-assuming walk: `matched` is in the
    # window TABLE's order, which for the compound case above is deliberately not
    # the map's.  Every row object here came from one `csv.DictReader` pass, so
    # `id()` identifies it uniquely; for the order-preserving case this returns
    # exactly the positions the previous left-to-right pointer walk did.
    position_of_row = {id(row): i for i, row in enumerate(rows)}
    kept_positions = [position_of_row[id(row)] for row in matched]
    dropped_positions = [i for i in range(len(rows)) if i not in set(kept_positions)]
    dropped_state_ids = [_int_or_none(rows[i].get("state_id")) for i in dropped_positions]

    survivors = [dict(r) for r in matched]
    for new_idx, row in enumerate(survivors):
        row["epoch_window"] = int(new_idx)
    try:
        _write_epoch_window_map_rows(path, fieldnames, survivors)
    except Exception as exc:
        print(f"WARNING: failed to rewrite {path} from the surviving window table: {exc}")
        return {"status": "error", "path": str(path), "error": str(exc), "source": str(source)}

    summary = {
        # A status a human reading gareus_metadata.json can act on: the compound
        # case removed rows AND moved the ones it kept, and "rewritten" alone
        # would say only the first half.
        "status": "rewritten_reordered_and_pruned" if reordered_onto_table else "rewritten",
        "path": str(path),
        "source": str(source),
        "reordered_onto_window_table": bool(reordered_onto_table),
        "n_windows_before": int(len(rows)),
        "n_windows_after": int(len(survivors)),
        "dropped_window_indices": dropped_positions,
        "dropped_state_ids": [s for s in dropped_state_ids if s is not None],
        "surviving_state_ids": _epoch_window_map_state_ids(survivors),
    }
    if reordered_onto_table:
        # Only for the compound case.  When the map's own row order is preserved
        # (the plain drop repair), `dropped_window_indices` already says exactly
        # which local windows shifted and by how much; a reordering has no such
        # compact description, so the before/after mapping is spelled out.
        summary["moved_windows"] = [
            {"epoch_window": int(i), "state_id_before": _int_or_none(rows[i].get("state_id")),
             "state_id_after": _int_or_none(row.get("state_id"))}
            for i, row in enumerate(survivors)
            if _int_or_none(rows[i].get("state_id")) != _int_or_none(row.get("state_id"))
        ]
    # An `applied` ledger entry either way: this branch REMOVES rows, so it has a
    # real drop key (n_windows_before, dropped_window_indices) and the
    # strictly-decreasing row count the idempotence rule rests on - unlike the
    # equal-count reorder, which has neither and is recorded separately.
    _append_epoch_window_map_rewrite_ledger(out_dir, summary)
    if reordered_onto_table:
        print(
            f"WARNING [{source}]: {path} listed {len(rows)} windows for the {len(survivors)} this phase "
            f"runs AND listed them in the wrong order - a compound stale map. Every real window had "
            f"exactly one map row carrying its restraint centers, so the correct mapping was read off "
            f"umbrella_explicit_windows.csv: dropped map rows {dropped_positions} (state_id "
            f"{dropped_state_ids}), the rest REORDERED onto the window table and renumbered "
            f"0..{len(survivors) - 1} keeping their own state_id and centers. Corrected local->state "
            f"mapping: "
            f"{', '.join(str(m['epoch_window']) + ': ' + str(m['state_id_before']) + '->' + str(m['state_id_after']) for m in summary['moved_windows'])}. "
            "Every sample already written for those windows was logged against the old order, so any "
            "analysis of this phase from before this correction is invalid - see "
            "docs/chignolin_6_low_ess_root_cause.md."
        )
    else:
        print(
            f"WARNING [{source}]: {path} listed {len(rows)} windows but this phase runs {len(survivors)}; "
            f"re-derived it from umbrella_explicit_windows.csv by matching restraint centers - dropped map "
            f"rows {dropped_positions} (state_id {dropped_state_ids}), survivors renumbered "
            f"0..{len(survivors) - 1} keeping their own state_id and centers. A stale map here would have "
            "attributed this phase's samples to the wrong umbrella state."
        )
    return summary


def _epoch_window_map_repair_must_be_recorded(summary: Optional[dict]) -> bool:
    """True when a `repair_epoch_window_map_from_surviving_windows` result belongs
    in the phase's gareus_metadata.json.

    Durable unless the map was actually VERIFIED against this phase's surviving
    window table - not merely "not rewritten".  The two are different claims and
    the difference is the whole point: the equal-count branch abstains (status
    "consistent", `centers_verified` False, `centers_check`
    "skipped_duplicate_centers") when duplicate restraint centers make the
    membership check ambiguous, so it reports the row counts agreeing WITHOUT
    having established that the map describes this phase's window set.  Keying the
    record off `status != "consistent"` alone would drop exactly that case on the
    floor, leaving an operator reading the metadata later unable to tell "centres
    verified" from "centres unverifiable" - while every one of its siblings out of
    that same branch (`rewritten_reordered`,
    `inconsistent_equal_count_centers_do_not_match`,
    `inconsistent_equal_count_reorder_not_positionally_verifiable`, and every
    `skipped_*`) is on record.  A skip case must never be indistinguishable from a
    pass.

    Written as "anything but a verified pass" rather than a list of the statuses to
    keep, so a future branch that returns "consistent" without verifying the
    centres is durable by default instead of silently inheriting the old fate.

    Not a warning and not a health signal: duplicate (primary, secondary) centers
    are legal (the explicit-2D loader keeps a user's duplicated rows as separate
    thermodynamic states), so a run that uses them records this on every phase.
    Nothing consumes the key programmatically - it is forensic evidence for a human
    reading the phase directory afterwards, which is the only place the question
    "was this map ever actually checked?" can still be answered.
    """
    if not summary:
        return False
    if summary.get("status") != "consistent":
        return True
    return summary.get("centers_verified") is not True


def drop_bad_us_windows_and_rebuild(
    out_dir: Path,
    dropped_window_indices: list[int],
    centers_a, k_list, centers_nm, ks_kj_nm2,
    secondary_cv_centers, secondary_cv_k_kcal_list, secondary_cv_ks_kj,
    window_start_positions, window_start_velocities,
    secondary_cv_metadata: dict, window_metadata: dict,
    args,
) -> dict:
    """Drop windows the post-pull US quality gate flagged 'bad' (--us-auto-drop-bad-windows)
    and rebuild every window-indexed artifact - arrays, neighbor graph, window CSVs - so
    production proceeds with a smaller, internally consistent window set.

    Mirrors the reindexing pattern used by filter_explicit_2d_windows_by_seed_reachability,
    but runs after the pull (using the gate's actual achieved-position verdict) instead of
    before it (using seed availability alone) - some windows only fail to converge once
    pulling actually starts, regardless of how good their seed looked on paper.
    """
    nwin_before = len(centers_a)
    dropped_requested = sorted({int(i) for i in dropped_window_indices})

    def _components_for_drop(drop_set):
        keep_idx = [i for i in range(nwin_before) if i not in drop_set]
        ca = np.asarray([centers_a[i] for i in keep_idx], dtype=float)
        sc = np.asarray([secondary_cv_centers[i] for i in keep_idx], dtype=float)
        edges = build_explicit_2d_neighbor_edges(ca, sc, args=args)
        return _connected_components_count(len(keep_idx), edges)

    needs_2d_check = (
        isinstance(secondary_cv_metadata, dict)
        and bool(secondary_cv_metadata.get("explicit_2d_windows", False))
        and secondary_cv_centers is not None
    )

    def _minimal_restore_for_connectivity(drop_set):
        """Smallest subset of drop_set whose restoration reconnects the graph, or None."""
        drop_list = sorted(drop_set)
        if _components_for_drop(set(drop_list)) <= 1:
            return []
        if len(drop_list) <= 12:
            for r in range(1, len(drop_list) + 1):
                for combo in itertools.combinations(drop_list, r):
                    if _components_for_drop(set(drop_list) - set(combo)) <= 1:
                        return list(combo)
            return None
        # Drop sets this large are not expected in practice (bounded by
        # us_auto_drop_max_fraction); fall back to a greedy hill-climb rather than
        # the exponential brute force above.
        remaining = list(drop_list)
        restored: list[int] = []
        n_components = _components_for_drop(set(remaining))
        while n_components > 1 and remaining:
            best_candidate, best_components = None, n_components
            for candidate in remaining:
                trial = _components_for_drop(set(remaining) - {candidate})
                if trial < best_components:
                    best_components, best_candidate = trial, candidate
            if best_candidate is None:
                return None
            remaining.remove(best_candidate)
            restored.append(best_candidate)
            n_components = best_components
        return restored

    restored: list[int] = []
    dropped = list(dropped_requested)
    if needs_2d_check and dropped:
        # Dropping every flagged-bad window can fragment the REUS exchange graph even
        # though dropping none of them (or a subset) would not. Rather than hard-failing
        # a whole adaptive epoch over this, restore the fewest bad windows back into
        # production needed to keep the graph connected - same principle as the
        # auto-drop itself (favor continuing over crashing), just extended to the case
        # where the drop set and the graph topology interact badly.
        restore_needed = _minimal_restore_for_connectivity(set(dropped))
        if restore_needed is None:
            raise RuntimeError(
                f"US auto-drop: even restoring every flagged-bad window ({dropped_requested}) back into "
                "production, the window set is still disconnected. This is not a marginal auto-drop edge "
                "case - the base window/CV grid itself has a connectivity gap. Fix the window grid, or "
                "pass --us-allow-bad-windows."
            )
        if restore_needed:
            restored = sorted(restore_needed)
            dropped = [i for i in dropped if i not in restore_needed]
            print(
                f"WARNING [US auto-drop]: keeping {restored} in production despite failing the "
                f"post-pull quality gate - dropping the full bad-window set {dropped_requested} would "
                "have disconnected the explicit-2D neighbor exchange graph. These replicas start further "
                "from their umbrella center than the gate normally allows; expect elevated initial bias "
                "and possibly slower equilibration for them."
            )

    keep = [i for i in range(nwin_before) if i not in dropped]

    def _sub(seq):
        if seq is None:
            return None
        return [seq[i] for i in keep]

    def _resubscript_normalized_rows(meta: Optional[dict], label: str) -> Optional[dict]:
        """Subset + renumber a metadata dict's per-window ``normalized_rows``.

        Same bookkeeping the pre-pull reachability filter already does for its own
        drop (seeding.py's filter_explicit_2d_windows_by_seed_reachability), which
        this function's docstring claims to mirror but did not.  These rows are
        consumed *positionally* - explicit_window_analysis_rows looks up
        ``normalized_rows[i]`` for post-drop window index i to fill that window's
        window_type/lifecycle/source_csv/source_row provenance, logger.py keys TUI
        window labels off each row's own "window" field, and adaptive_feedback.py
        zips them straight against the live center arrays.  Left at their pre-drop
        indices, every window at or past the first dropped one inherits a different
        window's provenance (confirmed on the real chignolin_6 final/baseline:
        post-drop window 20 reported source_row 22, which belongs to the window
        that was dropped, not to the state actually running there).
        """
        if not isinstance(meta, dict):
            return meta
        rows_in = meta.get("normalized_rows") or []
        if not rows_in:
            return dict(meta)
        out = dict(meta)
        if len(rows_in) != nwin_before:
            # Row position is the original window index only if the row list covers
            # the whole pre-drop set; anything else and subsetting would be a guess.
            print(
                f"WARNING: {label}['normalized_rows'] has {len(rows_in)} rows but {nwin_before} windows "
                "existed before the post-pull drop; leaving them unrenumbered (per-window provenance and "
                "diagnostic window labels may be off by the drop)."
            )
            return out
        kept_rows = [dict(rows_in[i]) for i in keep]
        for new_idx, row in enumerate(kept_rows):
            row["window"] = int(new_idx)
        out["normalized_rows"] = kept_rows
        if "n_windows" in out:
            out["n_windows"] = int(len(kept_rows))
        return out

    # Do this before the window tables below are rewritten - they read these rows.
    secondary_cv_metadata = _resubscript_normalized_rows(secondary_cv_metadata, "secondary_cv_metadata")
    window_metadata = _resubscript_normalized_rows(window_metadata, "window_metadata")

    new_centers_a = np.asarray(_sub(list(centers_a)), dtype=float)
    new_k_list = _sub(list(k_list))
    new_centers_nm = np.asarray(_sub(list(centers_nm)), dtype=float)
    new_ks_kj_nm2 = np.asarray(_sub(list(ks_kj_nm2)), dtype=float)
    new_secondary_cv_centers = np.asarray(_sub(list(secondary_cv_centers)), dtype=float) if secondary_cv_centers is not None else None
    new_secondary_cv_k_kcal_list = _sub(list(secondary_cv_k_kcal_list)) if secondary_cv_k_kcal_list is not None else None
    new_secondary_cv_ks_kj = np.asarray(_sub(list(secondary_cv_ks_kj)), dtype=float) if secondary_cv_ks_kj is not None else None
    new_window_start_positions = _sub(list(window_start_positions))
    new_window_start_velocities = _sub(list(window_start_velocities))

    graph_summary = None
    if needs_2d_check and new_secondary_cv_centers is not None:
        graph_summary = write_explicit_2d_neighbor_graph_files(
            out_dir, new_centers_a, new_secondary_cv_centers, args=args, prefix="explicit_2d_neighbor_graph"
        )

    # window_metadata was just resubscripted above (_resubscript_normalized_rows),
    # so its normalized_rows already carry each surviving window's own gamd_lambda,
    # correctly reindexed by this function's own (possibly connectivity-restored)
    # keep set. Without this, the rewritten umbrella_windows.csv would silently
    # zero every state's lambda -- and that file is load-bearing (the legacy MBAR
    # loader reconstructs biases from it), not merely diagnostic.
    new_gamd_lambdas = _derive_state_gamd_lambdas(window_metadata, len(new_centers_a), existing=None)
    _warn_if_ladder_was_zeroed(getattr(args, "state_gamd_lambdas", None), new_gamd_lambdas,
                               "post-pull US auto-drop window-table rewrite")
    window_rows = window_assignment_rows(
        new_centers_a, new_k_list, args.temperature_k,
        new_secondary_cv_centers, new_secondary_cv_k_kcal_list, args=args,
        gamd_lambdas=new_gamd_lambdas,
    )
    write_window_assignment_csv(out_dir / "umbrella_windows.csv", window_rows)
    try:
        explicit_window_table_summary = write_explicit_window_analysis_files(
            out_dir, new_centers_a, new_k_list, new_secondary_cv_centers, new_secondary_cv_k_kcal_list,
            secondary_cv_metadata=secondary_cv_metadata,
            window_metadata=window_metadata,
            neighbor_graph_summary=graph_summary,
            gamd_lambdas=new_gamd_lambdas,
        )
    except Exception as exc:
        explicit_window_table_summary = {}
        print(f"WARNING: failed to rewrite sparse-safe explicit window metadata after auto-drop: {exc}")

    secondary_cv_metadata = dict(secondary_cv_metadata or {})
    secondary_cv_metadata["dropped_post_pull_bad_windows"] = dropped
    window_metadata = dict(window_metadata or {})
    window_metadata["dropped_post_pull_bad_windows"] = dropped
    if graph_summary is not None:
        secondary_cv_metadata["explicit_2d_neighbor_graph"] = graph_summary
        window_metadata["explicit_2d_neighbor_graph"] = graph_summary
    if explicit_window_table_summary:
        window_metadata["explicit_window_table"] = explicit_window_table_summary
    # n_windows must follow the arrays even for a metadata dict that carries no
    # normalized_rows for _resubscript_normalized_rows to have covered above -
    # otherwise this phase's gareus_metadata.json reports a PRE-drop window count
    # next to POST-drop windows, i.e. one file contradicting itself (the real
    # chignolin_6 final/baseline reported n_windows 27 beside 24 real windows).
    if "n_windows" in window_metadata:
        window_metadata["n_windows"] = int(len(keep))
    if "n_windows" in secondary_cv_metadata:
        secondary_cv_metadata["n_windows"] = int(len(keep))

    # The map that tells the MBAR loader which global state each local window index
    # really is was written before the pull, over the pre-drop window set - correct
    # it now or every sample this phase logs gets attributed to the wrong umbrella
    # state downstream (see rewrite_epoch_window_map_after_drop's own docstring).
    map_rewrite = rewrite_epoch_window_map_after_drop(
        out_dir, dropped, nwin_before, source="post_pull_auto_drop",
    )
    if map_rewrite is not None:
        window_metadata["epoch_window_map_rewrite"] = map_rewrite

    print(
        f"    US auto-drop: {len(dropped)}/{nwin_before} windows dropped post-pull; "
        f"{len(keep)} windows remain for production (window indices reassigned 0..{len(keep) - 1})."
    )

    return {
        "centers_a": new_centers_a,
        "k_list": new_k_list,
        "centers_nm": new_centers_nm,
        "ks_kj_nm2": new_ks_kj_nm2,
        "secondary_cv_centers": new_secondary_cv_centers,
        "secondary_cv_k_kcal_list": new_secondary_cv_k_kcal_list,
        "secondary_cv_ks_kj": new_secondary_cv_ks_kj,
        "window_start_positions": new_window_start_positions,
        "window_start_velocities": new_window_start_velocities,
        "secondary_cv_metadata": secondary_cv_metadata,
        "window_metadata": window_metadata,
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
        self.v_pep_kj_mol: list[float] = []
        self.v_dih_kj_mol: list[float] = []
        self.gamd_lambda: list[float] = []
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
        self.v_pep_kj_mol.append(float(row.get("v_pep_kj_mol", np.nan)))
        self.v_dih_kj_mol.append(float(row.get("v_dih_kj_mol", np.nan)))
        self.gamd_lambda.append(float(row.get("gamd_lambda", 0.0)))
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
            "v_pep_kj_mol": np.asarray(self.v_pep_kj_mol, dtype=np.float64),
            "v_dih_kj_mol": np.asarray(self.v_dih_kj_mol, dtype=np.float64),
            "gamd_lambda": np.asarray(self.gamd_lambda, dtype=np.float64),
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
    is_gibbs = mode == "gibbs-walk"
    if attempts <= 0 and gibbs_choices <= 0:
        recommendations.append("No exchange attempts recorded yet; check --exchange-interval and whether production reached an exchange boundary.")
    elif total_acceptance < 0.08 and attempts > 0:
        recommendations.append("Total Metropolis exchange acceptance is very low; add/refine windows near weak pairs, reduce overly stiff k, or increase overlap targets.")
    elif total_acceptance > 0.60 and attempts > 0 and not is_gibbs:
        recommendations.append("Total exchange acceptance is high; the ladder may be over-resolved, so adaptive pruning or more aggressive spacing may reduce replica count.")
    elif total_acceptance > 0.60 and attempts > 0 and is_gibbs:
        # High MH acceptance among proposed Gibbs-walk moves is expected: the
        # heat-bath proposal deliberately favours energetically favourable targets,
        # so most accepted proposals are downhill.  Over-resolution should be judged
        # from gibbs_move_fraction (fraction of choices that were actual moves) and
        # PMF/MBAR diagnostics, not from MH acceptance alone.
        recommendations.append(
            "Gibbs-walk MH acceptance among proposed moves is high (>60%). "
            "This is normal when the heat-bath proposal preferentially selects energetically favourable targets. "
            "Use gibbs_move_fraction below as the primary health signal; if it is also high (>0.70), "
            "check MBAR overlap to ensure replicas are not jumping between poorly-overlapping states."
        )
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

def compare_gamd_global_sets(reference: dict[str, float], current: dict[str, float], rtol: float = 1.0e-10, atol: float = 1.0e-10,
                              ignore_names: frozenset = frozenset()) -> dict:
    """Compare CustomIntegrator globals after copying a shared GaMD setup.

    ``ignore_names`` drops globals from ref AND cur before any of missing/
    extra/mismatch is computed, so an intentionally-rescaled global (the
    lambda-ladder's k0_Total/k0_Dihedral -- see the call site in run_gareus,
    which passes {"k0_Total", "k0_Dihedral"} whenever the ladder is active)
    never shows up as a false "copied-global problem": every replica whose
    window lambda != 1 is SUPPOSED to differ from the raw shared-setup
    (lambda=1) reference here, by design (set_replica_lambda_for_window runs
    right after this same shared-globals copy), not because the copy failed.
    """
    ref = {k: v for k, v in (reference or {}).items() if k not in ignore_names}
    cur = {k: v for k, v in (current or {}).items() if k not in ignore_names}
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

def merge_adaptive_phase_info(base: dict, api: dict) -> dict:
    """Merge a segment's `_adaptive_phase_info` into the dashboard phase dict.

    Only keys the producer actually supplied are carried across.  The previous
    merge read `int(_api.get("epoch_index", 0))` unconditionally, so a stage
    that legitimately has no epoch index -- the terminal `final` stage -- had a
    plausible-looking `0` stamped onto it, and the header announced epoch 0 of a
    run with three epochs behind it.  An absent key must stay absent rather than
    acquire a default that reads as a measurement.

    Returns a new dict; `base` is never mutated.
    """
    merged = dict(base or {})
    for key in ("epoch_index", "epoch_total", "is_final_stage"):
        if key in (api or {}):
            merged[key] = api[key]
    if merged.get("is_final_stage"):
        # A final stage has no position in the epoch sequence; drop anything a
        # caller may have pre-seeded so the renderer cannot read a stale index.
        merged.pop("epoch_index", None)
        merged.pop("epoch_total", None)
    return merged


def checkpoint_manifest_path(out_dir: Path) -> Path:
    return Path(out_dir) / "checkpoints" / "production_checkpoint_manifest.json"

def save_production_checkpoint(out_dir: Path, sims: list, assignments: list[int], prod_done: int, absolute_step: int,
                               parity: int, attempt: int, next_exchange: int, next_log: int, exchange_stats: dict, rng,
                               centers_a=None, k_list=None, cv_atom1=None, cv_atom2=None, cv_label=None, calib_steps=None,
                               secondary_cv_metadata=None, secondary_cv_centers=None, secondary_cv_k_kcal_list=None,
                               primary_cv_metadata=None, openmm_version: Optional[str] = None,
                               platform_name: Optional[str] = None,
                               drivers: Optional[list] = None, pool=None, npt_runtime: Optional[NptRunContext] = None) -> None:
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

    NPT correction: when ``drivers``/``pool``/``npt_runtime`` are supplied, each
    replica's binary checkpoint AND its volume-controller ``state_dict()`` are
    captured together on the replica's own context-owning worker thread (closing
    the checkpoint-save affinity hole), and the manifest carries the per-replica
    NPT block (backend, adapter ids, T/P, frequency, fixed width, counters,
    last/next due step, RNG algorithm/state, molecule-partition fingerprint and
    controller schema version -- everything ``state_dict()`` contains).  The
    manifest write is the atomic commit point for the whole set.
    """
    chk_dir = Path(out_dir) / "checkpoints"
    chk_dir.mkdir(parents=True, exist_ok=True)
    replica_files = []
    replica_payloads = []
    replica_integrator_globals_all = []
    replica_integrator_global_counts = []
    npt_controller_states: list = []

    def _checkpoint_replica(r: int):
        """Capture one replica's binary checkpoint + controller state on its
        own worker thread; returns (relpath, globals, n_state)."""
        sim = sims[r]
        driver = drivers[r] if drivers is not None and r < len(drivers) else None
        from .correctness.repo_adapters import checkpoint_globals
        g = checkpoint_globals(sim.integrator, all_integrator_globals)
        rel = f"replica_{r:03d}.chk"
        replica_payloads.append(sim.context.createCheckpoint())
        n_state = None
        if driver is not None and getattr(driver, "controller", None) is not None:
            n_state = _json_ready(driver.controller.state_dict())
        return rel, g, n_state

    for r, sim in enumerate(sims):
        if pool is not None:
            rel, g, n_state = pool.submit(r, _checkpoint_replica, r).result()
        else:
            rel, g, n_state = _checkpoint_replica(r)
        replica_files.append(rel)
        replica_integrator_globals_all.append(_json_ready(g))
        replica_integrator_global_counts.append(int(len(g)))
        npt_controller_states.append(n_state)
    manifest = {
        "schema": "gareus_production_checkpoint_v1",
        "openmm_version": str(openmm_version) if openmm_version is not None else None,
        "platform_name": str(platform_name) if platform_name is not None else None,
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
    if npt_runtime is not None and drivers is not None:
        has_controllers = any(state is not None for state in npt_controller_states)
        if npt_runtime.backend == "biased_mc" and not has_controllers:
            raise RuntimeError(
                "biased_mc NPT checkpoint requested but no replica carries a volume "
                "controller; refusing to write a checkpoint that cannot be resumed"
            )
        manifest["npt"] = npt_runtime.manifest_block(
            controllers=npt_controller_states,
            n_atoms=[int(sim.system.getNumParticles()) for sim in sims],
        )
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
    from .correctness.checkpoint_store import publish_generation
    publish_generation(out_dir, replica_payloads, manifest)

def sync_scratch_to_main(scratch_dir: Path, main_dir: Path) -> None:
    """Checkpoint-safe quiescent copy; mutable sample files are individually atomic.

    This is not yet a generation-index transaction across all Parquet output.
    Any failure remains visible instead of claiming a successful backup.
    """
    from .correctness.repo_adapters import sync_run_tree_quiescent
    sync_run_tree_quiescent(scratch_dir, main_dir)

def _validate_npt_checkpoint_compatibility(out_dir: Path, manifest: dict, npt_runtime: NptRunContext, sims: list) -> Optional[str]:
    """Validate backend/adapter/T/P/topology before accepting a checkpoint.

    Returns ``"restored"``, ``"fresh_init"`` or ``None`` (nothing to do).
    Raises ``RuntimeError`` for every incompatible combination, including the
    legacy boosted-NPT checkpoint case: removing the native barostat from the
    application-controlled System changes the System itself, so binary
    compatibility with a pre-correction boosted-NPT checkpoint is explicitly
    NOT promised (spec section 8).
    """
    npt_block = manifest.get("npt")
    current = npt_runtime.backend
    if current == "none":
        return None
    if npt_block is None:
        # Legacy manifest (pre-correction).  Whether resuming is meaningful
        # depends on what the legacy run actually was; gareus_metadata.json
        # recorded it.
        meta_path = Path(out_dir) / "gareus_metadata.json"
        meta = read_json_file(meta_path, None) if meta_path.exists() else None
        if not isinstance(meta, dict):
            raise RuntimeError(
                "Cannot verify NPT compatibility of the production checkpoint: "
                f"{meta_path} is missing or unreadable, and the checkpoint manifest "
                "predates the NPT correction (no 'npt' block). Refusing to resume "
                "rather than guess who owned volume moves."
            )
        legacy_ensemble = str(meta.get("production_ensemble", "") or "").lower()
        legacy_gamd = bool(meta.get("gamd_enabled", False))
        if current == "biased_mc" and legacy_ensemble == "npt" and legacy_gamd:
            raise RuntimeError(
                "This production checkpoint is a LEGACY boosted-NPT checkpoint from "
                "before the NPT correction: its System carried a native "
                "MonteCarloBarostat whose acceptance energy omitted the GaMD boost "
                "and included the Pep-GaMD auxiliary force. Removing that barostat "
                "changes the System, so the binary Context checkpoint is NOT "
                "compatible with the corrected biased-MC backend and no exact "
                "continuation exists. Start a new run (a fresh segment with an "
                "explicitly identified equilibration), or use the separately "
                "requested migration to extract positions/velocities/box under the "
                "legacy System. Historical frames must not be relabeled as "
                "corrected samples."
            )
        if current == "biased_mc" and legacy_ensemble == "nvt":
            print(
                "    [npt] Resuming a legacy NVT checkpoint under the biased-MC NPT "
                "backend: the physical System is unchanged, so the binary "
                "checkpoint is loadable, but there is no volume-move schedule or "
                "RNG stream to restore -- initializing fresh controllers from the "
                "resumed step. This is a new-segment ensemble change (NVT -> NPT), "
                "not an exact continuation."
            )
            return "fresh_init"
        if current == "biased_mc":
            raise RuntimeError(
                f"Legacy checkpoint has no NPT block and gareus_metadata.json reports "
                f"ensemble={legacy_ensemble!r}, gamd_enabled={legacy_gamd}; cannot "
                "resume it under the biased-MC NPT backend."
            )
        return None
    # Current-generation manifest: the stored block must match this run.
    stored_backend = str(npt_block.get("backend", ""))
    if stored_backend != current:
        raise RuntimeError(
            f"Checkpoint was written with npt backend {stored_backend!r} but this run "
            f"resolved {current!r}: removing (or adding) the native barostat changes "
            "the System, so the binary checkpoints are not compatible. Start a new "
            "run or re-select the same backend."
        )
    if current != "biased_mc":
        return None
    controllers = npt_block.get("controllers") or []
    if len(controllers) != len(sims) or any(c is None for c in controllers):
        raise RuntimeError(
            "biased_mc checkpoint manifest does not carry a volume-controller state "
            "for every replica; cannot restore the exact schedule and random stream"
        )
    adapter_ids = list(npt_block.get("adapter_ids") or [])
    # Build the lazily-resolved adapter from a live replica Context before
    # comparing ids: the stored ids are the real adapter's, and the lazy
    # wrapper still holds its empty pre-build placeholder until a Context
    # exists to build from.
    _ensure = getattr(npt_runtime.adapter, "ensure_built", None)
    if _ensure is not None and sims:
        _ensure(sims[0].context)
    this_adapter_id = str(getattr(npt_runtime.adapter, "adapter_id", ""))
    for i, stored_id in enumerate(adapter_ids):
        if str(stored_id) != this_adapter_id:
            raise RuntimeError(
                f"Checkpoint adapter id {stored_id!r} (replica {i}) does not match "
                f"this run's effective-potential adapter {this_adapter_id!r}: the "
                "volume-move acceptance energy would silently change meaning."
            )
    if abs(float(npt_block.get("temperature_k", 0.0)) - float(npt_runtime.ownership.temperature_k)) > 1e-6:
        raise RuntimeError("Checkpoint temperature does not match this run's temperature")
    if abs(float(npt_block.get("pressure_bar", 0.0)) - float(npt_runtime.ownership.pressure_bar)) > 1e-6:
        raise RuntimeError("Checkpoint pressure does not match this run's pressure")
    stored_atoms = npt_block.get("n_atoms") or []
    for r, sim in enumerate(sims):
        if r < len(stored_atoms) and int(stored_atoms[r]) != int(sim.system.getNumParticles()):
            raise RuntimeError(
                f"Checkpoint topology has {stored_atoms[r]} particles for replica "
                f"{r} but this run's System has {sim.system.getNumParticles()}"
            )
    return "restored"


def load_production_checkpoint(out_dir: Path, sims: list, centers_nm, ks_kj_nm2, rng, secondary_centers=None, secondary_ks_kj=None,
                               openmm_version: Optional[str] = None, platform_name: Optional[str] = None,
                               strict_gamd_restore: bool = False,
                               state_lambdas=None, k0max_by_channel: Optional[dict] = None,
                               drivers: Optional[list] = None, pool=None,
                               npt_runtime: Optional[NptRunContext] = None,
                               args=None) -> Optional[dict]:
    """Load a production checkpoint manifest and all replica checkpoints if available.

    state_lambdas/k0max_by_channel need not both be given: k0max_by_channel is None
    on every run that does not have the λ-ladder active (plain GaMD, conventional
    MD, or GaMD without a ladder -- the common case), while state_lambdas is an
    unconditionally-populated array regardless of run mode. Only the combination
    of an active ladder (k0max_by_channel is not None) with no per-window λ
    (state_lambdas is None) is a real misconfiguration; see
    set_replica_lambda_for_window, which enforces exactly that and nothing more.

    NPT correction: with ``drivers``/``pool``/``npt_runtime`` the binary
    checkpoints load on each replica's own worker thread (closing the
    checkpoint-load affinity hole), the volume-controller compatibility is
    validated BEFORE anything is loaded, and each controller's exact schedule
    and random stream is restored from the manifest's per-replica
    ``state_dict()`` -- restoring never attempts an extra move.
    """
    from .correctness.checkpoint_store import read_validated_generation
    from .correctness.repo_adapters import validate_rng_restore
    manifest_path = checkpoint_manifest_path(out_dir)
    _generation = read_validated_generation(out_dir)
    if _generation is None:
        return None
    manifest = dict(_generation.manifest)
    validate_rng_restore(manifest, rng)
    manifest_openmm_version = manifest.get("openmm_version")
    if openmm_version is not None and manifest_openmm_version is not None and str(manifest_openmm_version) != str(openmm_version):
        print(f"WARNING: checkpoint manifest was written with OpenMM {manifest_openmm_version}, but this run is using OpenMM {openmm_version}; a version bump is not automatically incompatible, but resume state should be checked carefully.")
    manifest_platform_name = manifest.get("platform_name")
    if platform_name is not None and manifest_platform_name is not None and str(manifest_platform_name) != str(platform_name):
        print(f"WARNING: checkpoint manifest was written on platform '{manifest_platform_name}', but this run is using platform '{platform_name}'; resume state should be checked carefully.")
    files = manifest.get("replica_checkpoint_files", [])
    if len(files) != len(sims):
        raise RuntimeError(f"Checkpoint replica count mismatch: manifest has {len(files)} files, current run has {len(sims)} replicas")
    npt_resume_mode = None
    if npt_runtime is not None:
        npt_resume_mode = _validate_npt_checkpoint_compatibility(out_dir, manifest, npt_runtime, sims)
    chk_dir = manifest_path.parent
    pre_load_globals = []
    for sim in sims:
        try:
            pre_load_globals.append(all_integrator_globals(sim.integrator))
        except Exception:
            pre_load_globals.append({})
    npt_block = manifest.get("npt") or {}

    def _load_replica(r: int):
        sim = sims[r]
        # Load the exact bytes already validated before any Context was changed.
        data = _generation.replica_payloads[r]
        sim.context.loadCheckpoint(data)
        controller = None
        if npt_runtime is not None and npt_runtime.needs_controller:
            if npt_resume_mode == "restored":
                controller = npt_runtime.restore_controller(
                    sim.context, npt_block.get("controllers")[r]
                )
            elif npt_resume_mode == "fresh_init":
                controller = npt_runtime.initialize_controller(
                    sim.context, seed=npt_seed_replica(args, r)
                )
        if controller is not None and drivers is not None and r < len(drivers):
            drivers[r].controller = controller
        return r

    for r, sim in enumerate(sims):
        if pool is not None:
            pool.submit(r, _load_replica, r).result()
        else:
            _load_replica(r)
    # Do not rely only on the opaque OpenMM binary checkpoint for gamd-openmm
    # CustomIntegrator globals.  New manifests store them explicitly; old
    # manifests may recover them from exact-step gamd_globals_json if available.
    gamd_restore_report = _restore_gamd_integrator_globals_after_checkpoint(Path(out_dir), sims, manifest)
    manifest["resume_gamd_integrator_restore_report"] = gamd_restore_report
    if not bool(gamd_restore_report.get("ok", False)):
        if strict_gamd_restore:
            raise RuntimeError(
                f"GaMD integrator-global restore was incomplete after checkpoint load (source={gamd_restore_report.get('source')}); "
                "aborting because --strict-gamd-restore is set. See resume_gamd_integrator_restore_report.json for details."
            )
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

    def _apply_assignment(r: int):
        sim = sims[r]
        set_window(sim.context, centers_nm, ks_kj_nm2, int(assignments[r]), secondary_centers, secondary_ks_kj)
        # The GaMD-integrator-globals restore above is best-effort (see the
        # comments above): when it is incomplete, k0_Total/k0_Dihedral are still
        # whatever _build_context_i set from this replica's BUILD-time index, not
        # necessarily its RESUMED window assignment. Re-derive k0 from the
        # restored assignment unconditionally so a partial restore can never
        # leave a replica's boost strength mismatched with its window. No-ops
        # when the ladder is inactive (k0max_by_channel is None).
        set_replica_lambda_for_window(sim.integrator, assignments[r], state_lambdas, k0max_by_channel)
        return r

    for r, sim in enumerate(sims):
        if pool is not None:
            pool.submit(r, _apply_assignment, r).result()
        else:
            _apply_assignment(r)
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
                         primary_cv_def: dict, cv_atom1: int, cv_atom2: int, unit, reference_gamd_globals: dict[str, float],
                         *, drivers: Optional[list] = None, pool=None,
                         npt_runtime: Optional[NptRunContext] = None) -> dict:
    """Run and roll back a tiny production probe to catch NaNs before the long run.

    Context checkpoints are restored afterwards, so the probe does not consume
    production time and does not affect the first production sample.

    NPT correction: the probe advances the SAME driver machinery as production
    (volume moves included, on each replica's own worker thread when ``pool``
    is given -- closing the probe affinity hole), with reports suppressed
    because the probe states are transient and get rolled back.  Any volume
    controllers also roll back to their pre-probe state via
    ``state_dict()``/``restore()`` so the probe consumes neither schedule nor
    random stream.

    The probe's potential energy is read over the PHYSICAL force groups
    (``physical_energy_groups_for_args``): a raw Context potential under a
    Pep-GaMD partition includes the auxiliary water-only force and must never
    be presented as physical energy.
    """
    nsteps = int(getattr(args, "production_probe_steps", 0) or 0)
    report = {"enabled": nsteps > 0, "steps": nsteps, "replicas": []}
    if nsteps <= 0:
        write_json(Path(out_dir) / "production_probe_report.json", report)
        return report

    def _submit(r: int, fn, *a, **kw):
        return pool.submit(r, fn, *a, **kw) if pool is not None else None

    def _run(r: int, fn, *a, **kw):
        fut = _submit(r, fn, *a, **kw)
        return fut.result() if fut is not None else fn(*a, **kw)

    def _capture_controller_state(driver):
        return driver.controller.state_dict() if driver.controller is not None else None

    saved = [_run(r, sim.context.createCheckpoint) for r, sim in enumerate(sims)]
    saved_controllers = [
        _run(r, _capture_controller_state, driver)
        for r, driver in enumerate(drivers or [])
    ]
    failures = []
    try:
        def _probe_one(item):
            r, sim = item
            driver = drivers[r] if drivers is not None and r < len(drivers) else None
            if driver is not None:
                # advance() with reports=False: transient probe states must not
                # emit trajectory frames (the context is rolled back below).
                driver.advance(nsteps, reports=False)
            else:
                sim.step(nsteps)
            state = sim.context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True,
                                         groups=physical_energy_groups_for_args(args))
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
                "potential_source": "physical_force_groups",
                "gamd_boost_total_kj_mol": float(boost_kj) if boost_kj is not None and math.isfinite(float(boost_kj)) else None,
                "gamd_boost_source": str(boost_source),
                "gamd_boost_components_kj_mol": boost_components,
                "finite_cv": bool(math.isfinite(float(cv_value))),
                "finite_potential": bool(math.isfinite(float(pe_kj))),
                "finite_boost_or_unavailable": bool(boost_kj is None or math.isfinite(float(boost_kj))),
            }
            row["ok"] = bool(row["finite_cv"] and row["finite_potential"] and row["finite_boost_or_unavailable"])
            return row

        items = list(enumerate(sims))
        if pool is not None:
            rows = [pool.submit(r, _probe_one, item).result() for r, item in enumerate(items)]
        else:
            rows = [_probe_one(item) for item in items]
        for row in rows:
            if not row["ok"]:
                failures.append(row)
            report["replicas"].append(row)
    except Exception as exc:
        failures.append({"exception": str(exc)})
        report["exception"] = str(exc)
    finally:
        for r, sim in enumerate(sims):
            _run(r, sim.context.loadCheckpoint, saved[r])
        for r, sim in enumerate(sims):
            _run(r, set_window, sim.context, centers_nm, ks_kj_nm2, int(assignments[r]))
        # Roll the volume controllers back to their pre-probe state so the
        # probe consumed neither schedule nor random stream.
        if drivers is not None and npt_runtime is not None and npt_runtime.needs_controller:
            for r, driver in enumerate(drivers):
                if r < len(saved_controllers) and saved_controllers[r] is not None:
                    driver.controller = _run(
                        r, npt_runtime.restore_controller, sims[r].context, saved_controllers[r]
                    )
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

def _gamd_boost_group_targets(system, args, unit) -> tuple[list[tuple[str, Optional[int]]], object]:
    """Discover this run's GaMD boost-group targets and force-group ids.

    Building a throwaway gamd-openmm integrator for `system` mutates its Force
    objects' force groups as a side effect (gamd-openmm's own
    GamdIntegratorFactory.get_integrator convention: set_all_forces_to_group(0),
    then e.g. set_non_bonded_group/set_dihedral_group override specific Force
    classes) and exposes the resulting group names generically via
    integrator.get_group_dict()/get_statistics_names() -- valid for any
    --gamd-boost-type, not just the flagship lower-dual-nonbonded-dihedral.
    """
    integrator, _result = make_gamd_integrator(system, args, unit)
    group_dict = integrator.get_group_dict()  # {force_group_id: group_name}
    targets: list[tuple[str, Optional[int]]] = [(name, int(gid)) for gid, name in group_dict.items()]
    stat_names = integrator.get_statistics_names()
    if any(name.endswith("_Total") for name in stat_names) and not any(t[0] == "Total" for t in targets):
        targets.append(("Total", None))
    if not targets:
        raise RuntimeError(
            f"Could not determine GaMD boost-group targets for --gamd-boost-type {args.gamd_boost_type!r}; "
            f"get_group_dict()={group_dict!r}, get_statistics_names()={stat_names!r}"
        )
    return targets, integrator

def run_multiwindow_gamd_recon(
    args, openmm, app, unit, topology, base_system, platform, props,
    centers_nm, ks_kj_nm2, secondary_cv_centers, secondary_cv_ks_kj,
    window_start_positions, window_start_velocities, equil_state,
    progress: Optional[GuiProgressSink] = None,
    *,
    integrator_kind: str = "cmd",
    seed_globals: Optional[dict] = None,
    recon_steps: Optional[int] = None,
    prep_steps: Optional[int] = None,
    phase_label: str = "gamd_multiwindow_recon",
    npt_runtime: Optional[NptRunContext] = None,
) -> dict[str, "PooledEnvelope"]:
    """Recon every initial/pilot window's boost-group potential energy under its
    own umbrella bias, then pool per boost-group across all windows.

    `base_system` already has the primary umbrella force (and secondary CV
    force, if enabled) added -- deserialize_system(openmm, base_system) gives
    each window a ready-to-bias system copy with no further force additions
    needed (mirrors how production replicas are built at the call site below).

    `integrator_kind` selects the stepping integrator:
      - "cmd": plain conventional MD (boost OFF) -- used to seed the boost with
        an initial Vmax/Vmin/Vavg/sigmaV (the GaMD integrator needs starting
        extrema before it can boost).
      - "gamd": the GaMD boost integrator seeded with `seed_globals` (a full
        integrator-globals dict, expected to carry the fixed-boost production
        `stage`). This measures sigmaV under the boost the run will actually use.
    The measured potential energy is the boost-group force-group energy in both
    cases (the boost is applied by the integrator, not as a separate force, so
    getState(groups=...) returns the true group energy regardless of boost).
    """
    from gareus.gamd_calibration import WelfordAccumulator, pool_window_stats

    nwin = int(len(centers_nm))
    if prep_steps is None:
        prep_steps = int(getattr(args, "gamd_multiwindow_recon_prep_steps", 2000) or 0)
    else:
        prep_steps = int(prep_steps)
    if recon_steps is None:
        recon_steps = int(getattr(args, "gamd_multiwindow_recon_steps", 20000) or 0)
    else:
        recon_steps = int(recon_steps)
    report_interval = int(getattr(args, "gamd_multiwindow_recon_report_interval", 0) or 0)
    if report_interval <= 0:
        report_interval = max(1, recon_steps // 200)
    if str(getattr(args, "platform", "")).upper() == "CPU" and int(getattr(args, "cpu_threads", 1)) == 0 and nwin > 1:
        print(
            f"WARNING: --cpu-threads 0 (use all cores) combined with {nwin} parallel recon windows "
            "will oversubscribe CPU cores. Set --cpu-threads 1 for parallel recon."
        )

    equil_box = equil_state.getPeriodicBoxVectors()

    # Build every window's Context serially, exactly like the production replica
    # burst (see the "[4/4] Building N GaREUS replicas" loop): concurrent context
    # creation on OpenCL/CUDA can trip clCreateContext(-6)-style failures, so
    # construction stays single-threaded even though stepping will fan out.
    sims: list = []
    systems: list = []
    integrators: list = []
    targets_per_window: list = []
    for i in range(nwin):
        system_i = deserialize_system(openmm, base_system)
        targets, _peek_integrator = _gamd_boost_group_targets(system_i, args, unit)
        if integrator_kind == "gamd":
            step_integrator, _ = make_gamd_integrator(system_i, args, unit)
        else:
            step_integrator, _ = make_cmd_integrator(openmm, args, unit, system=system_i)
        props_i = replica_platform_properties(platform, props, args, i)
        sim_i = app.Simulation(topology, system_i, step_integrator, platform, props_i)
        if integrator_kind == "gamd" and seed_globals:
            set_integrator_globals_from_dict(step_integrator, seed_globals)
        pos = window_start_positions[i] if window_start_positions and window_start_positions[i] is not None else equil_state.getPositions()
        vel = window_start_velocities[i] if window_start_velocities and window_start_velocities[i] is not None else None
        if equil_box is not None:
            sim_i.context.setPeriodicBoxVectors(*equil_box)
        sim_i.context.setPositions(pos)
        if vel is not None:
            sim_i.context.setVelocities(vel)
        else:
            sim_i.context.setVelocitiesToTemperature(args.temperature_k * unit.kelvin, args.seed + 4021 + i)
        set_window(sim_i.context, centers_nm, ks_kj_nm2, i, secondary_cv_centers, secondary_cv_ks_kj)
        sims.append(sim_i)
        systems.append(system_i)
        integrators.append(step_integrator)
        targets_per_window.append(targets)

    # NPT correction: each calibration window gets its OWN volume controller
    # with an INDEPENDENT random stream (spec section 6: never clone a
    # calibration RNG stream into every replica -- and the production
    # replicas' streams are separate from these again).  Construction stays
    # on this thread exactly like the Contexts above (existing recon
    # discipline); stepping fans out per window below.
    # When these windows are conventional MD (integrator_kind != "gamd" above ->
    # make_cmd_integrator, a plain Langevin that excludes the auxiliary group),
    # the run-wide adapter does not describe them: it is built for this run's
    # boost type and reads boost state a plain integrator does not carry. Give
    # them their OWN adapter, resolved through the documented cmd route, and
    # leave the shared instance untouched -- ensure_built() caches, so resolving
    # the shared one here would hand every later boosted context a zero-boost
    # acceptance energy.
    recon_adapter = None
    if npt_runtime is not None and npt_runtime.needs_controller and integrator_kind != "gamd":
        cmd_args = copy.copy(args)
        cmd_args.run_mode = "hmr-cmd" if "hmr" in str(
            getattr(args, "run_mode", "") or "").lower() else "cmd"
        recon_adapter = _resolve_npt_adapter(cmd_args, base_system)

    drivers: list = []
    for i in range(nwin):
        controller_i = None
        if npt_runtime is not None and npt_runtime.needs_controller:
            controller_i = npt_runtime.initialize_controller(
                sims[i].context, seed=npt_seed_recon_window(args, i),
                adapter=recon_adapter,
            )
        drivers.append(ReplicaStepDriver(sims[i], controller=controller_i, label=f"recon_window_{i}"))

    # Parallel prep + recon across windows, mirroring the production step_all()
    # fan-out: OpenMM releases the GIL during context.step()/getState(), so one
    # thread per window genuinely runs concurrently across GPUs/devices.
    recon_pool = ThreadPoolExecutor(max_workers=nwin)
    try:
        if prep_steps > 0:
            list(recon_pool.map(lambda driver_i: driver_i.advance(int(prep_steps)), drivers))

        accumulators_by_window = [
            {name: WelfordAccumulator() for name, _gid in targets_per_window[i]}
            for i in range(nwin)
        ]

        _recon_total_groups = total_energy_groups_for_args(args)

        def _recon_chunk(item):
            i, driver_i = item
            driver_i.advance(int(chunk))
            sim_i = driver_i.sim
            for name, gid in targets_per_window[i]:
                pe_kj = boost_target_energy_kj(sim_i.context, gid, unit, total_groups=_recon_total_groups)
                accumulators_by_window[i][name].update(pe_kj)
            return i

        done = 0
        while done < recon_steps:
            chunk = min(report_interval, recon_steps - done)
            list(recon_pool.map(_recon_chunk, enumerate(drivers)))
            done += int(chunk)
            if progress is not None:
                progress.progress(
                    phase_label, done, recon_steps,
                    message=f"{nwin} windows in parallel | recon {done}/{recon_steps} steps",
                    n_replicas=nwin, force=(done >= recon_steps),
                )
    finally:
        recon_pool.shutdown(wait=True)

    per_group_window_stats: dict[str, list] = {}
    for i in range(nwin):
        for name, acc in accumulators_by_window[i].items():
            per_group_window_stats.setdefault(name, []).append(acc.to_stats(name, i))
        release_openmm_contexts(sims[i], integrators[i], systems[i])

    if not per_group_window_stats:
        raise RuntimeError("Multi-window GaMD recon collected no boost-group statistics")
    return {name: pool_window_stats(stats) for name, stats in per_group_window_stats.items()}

def apply_joint_envelope_gamd_calibration(
    args, openmm, app, unit, topology, base_system, setup_platform, setup_props,
    out_dir, nrep, centers_nm, ks_kj_nm2, secondary_cv_centers, secondary_cv_ks_kj,
    window_start_positions, window_start_velocities, equil_state,
    shared_gamd_globals_all, shared_gamd_globals_interesting, calib_steps,
    progress: Optional[GuiProgressSink] = None,
    npt_runtime: Optional[NptRunContext] = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Recon every initial window, pool per boost-group, and overwrite the
    physics globals in the shared-setup globals dicts before export.

    Returns (shared_gamd_globals_all, shared_gamd_globals_interesting) with
    Vmax/Vmin/Vavg/sigmaV/k0/threshold_energy/k overwritten per boost group;
    every other global (stepCount, stage, windowCount, ForceScalingFactor,
    sigma0_<group>, ...) is unchanged. Also writes the updated
    shared_gamd_setup_globals.json (superseding the one
    run_shared_gamd_setup_article_a already wrote) and runs a 50-step
    finite-energy verification before returning, raising RuntimeError if it
    is not finite.
    """
    from gareus.gamd_calibration import (
        compute_group_calibration, overwrite_physics_globals, run_calibration_convergence,
    )

    _peek_system = deserialize_system(openmm, base_system)
    joint_targets, _peek_integrator = _gamd_boost_group_targets(_peek_system, args, unit)
    release_openmm_contexts(_peek_integrator, _peek_system)

    cmd_seed_steps = int(getattr(args, "gamd_multiwindow_recon_cmd_steps", 20000) or 0)
    boosted_steps = int(getattr(args, "gamd_multiwindow_recon_steps", 20000) or 0)
    boosted_iters = int(getattr(args, "gamd_recon_boosted_iters", 0) or 0)
    boosted_tol = float(getattr(args, "gamd_recon_boosted_tol", 0.05) or 0.05)
    group_names = [name for name, _gid in joint_targets]

    # Stage 1: short conventional-MD seed recon (boost OFF) -> initial envelope.
    print(f"    GaMD calibration: cMD seed recon of {nrep} windows "
          f"({cmd_seed_steps} steps each) for boost group(s) {', '.join(group_names)}")
    seed_envelopes = run_multiwindow_gamd_recon(
        args, openmm, app, unit, topology, base_system, setup_platform, setup_props,
        centers_nm, ks_kj_nm2, secondary_cv_centers, secondary_cv_ks_kj,
        window_start_positions, window_start_velocities, equil_state,
        integrator_kind="cmd", recon_steps=cmd_seed_steps,
        phase_label="gamd_recon_cmd_seed", progress=progress,
        npt_runtime=npt_runtime,
    )
    for name in group_names:
        if name not in seed_envelopes:
            raise RuntimeError(f"Multi-window recon produced no pooled statistics for boost group {name!r}")
    sigma0_by_group = {
        name: float(shared_gamd_globals_all.get(f"sigma0_{name}", 0.0) or 0.0)
        for name in group_names
    }

    # Stage 2: measure sigmaV under the boost, iterating to self-consistency.
    def _measure_boosted(calibrations):
        seed_g = dict(overwrite_physics_globals(shared_gamd_globals_all, calibrations))
        seed_g["stage"] = 5.0  # fixed-boost production stage
        return run_multiwindow_gamd_recon(
            args, openmm, app, unit, topology, base_system, setup_platform, setup_props,
            centers_nm, ks_kj_nm2, secondary_cv_centers, secondary_cv_ks_kj,
            window_start_positions, window_start_velocities, equil_state,
            integrator_kind="gamd", seed_globals=seed_g, recon_steps=boosted_steps,
            phase_label="gamd_recon_boosted", progress=progress,
            npt_runtime=npt_runtime,
        )

    if boosted_iters > 0:
        print(f"    GaMD calibration: iterating boosted recon to self-consistency "
              f"(max {boosted_iters} iters, {boosted_steps} steps/window/iter, tol {boosted_tol:.3f})")
    conv = run_calibration_convergence(
        _measure_boosted, str(args.gamd_boost_type), sigma0_by_group,
        seed_envelopes, max_iters=boosted_iters, tol=boosted_tol,
    )
    calibrations = {name: conv[name].calibration for name in conv}

    shared_gamd_globals_all = overwrite_physics_globals(shared_gamd_globals_all, calibrations)
    shared_gamd_globals_interesting = overwrite_physics_globals(shared_gamd_globals_interesting, calibrations)

    boosted_calibration_report = {
        name: {
            "cmd_seed_sigmaV_kj_mol": seed_envelopes[name].sigmav,
            "final_sigmaV_kj_mol": conv[name].calibration.sigmav,
            "sigmaV_trace_kj_mol": list(conv[name].sigmav_trace),
            "converged": bool(conv[name].converged),
            "boosted_iters": int(conv[name].iters),
        }
        for name in conv
    }
    joint_envelope_report = {
        name: {
            "vmax_kj_mol": c.vmax, "vmin_kj_mol": c.vmin, "vavg_kj_mol": c.vavg,
            "sigmaV_kj_mol": c.sigmav,
            "sigma0_kj_mol": float(shared_gamd_globals_all.get(f"sigma0_{name}", 0.0) or 0.0),
            "k0": c.k0, "k": c.k, "threshold_energy_kj_mol": c.threshold_energy,
            "boosted": c.boosted, "n_windows_pooled": c.n_windows, "n_samples_pooled": c.n_total,
        }
        for name, c in calibrations.items()
    }
    write_json(out_dir / "shared_gamd_setup_globals.json", {
        "mode": "joint_envelope_gamd_calibration",
        "description": (
            "Vmax/Vmin/Vavg/sigmaV/k0/threshold_energy were calibrated in two stages: "
            "(1) a short conventional-MD recon under each initial window's own umbrella "
            "bias seeds the boost, then (2) the boost is turned on and the per-window "
            "recon is repeated and pooled, iterating to self-consistency on sigmaV so the "
            "frozen boost matches the boosted production ensemble (not the unboosted one). "
            "The resulting CustomIntegrator globals are copied to every GaREUS replica "
            "before production."
        ),
        "calibration_steps": int(calib_steps),
        "gamd_multiwindow_recon_cmd_steps": cmd_seed_steps,
        "gamd_multiwindow_recon_prep_steps": int(getattr(args, "gamd_multiwindow_recon_prep_steps", 0) or 0),
        "gamd_multiwindow_recon_steps": boosted_steps,
        "gamd_recon_boosted_iters": boosted_iters,
        "gamd_recon_boosted_tol": boosted_tol,
        "boosted_calibration": boosted_calibration_report,
        "joint_envelope": joint_envelope_report,
        "temperature_K": float(args.temperature_k),
        "gamd_boost_type": str(args.gamd_boost_type),
        "sigma0p_kcal_mol": float(args.sigma0p_kcal_mol),
        "sigma0d_kcal_mol": float(args.sigma0d_kcal_mol),
        "interesting_globals": shared_gamd_globals_interesting,
        "all_globals": shared_gamd_globals_all,
    })
    print("    GaMD calibration: " + ", ".join(
        f"{name} k0={c.k0:.3f} sigmaV={c.sigmav:.1f} threshold={c.threshold_energy:.1f} kJ/mol "
        f"(pooled {c.n_windows} windows, {c.n_total} samples, {conv[name].iters} boosted iters"
        f"{'' if conv[name].converged else ', NOT converged'})"
        for name, c in calibrations.items()
    ))
    # k0-saturation warning: the sigmaV<=sigma0 guardrail is inert once k0 hits 1.0.
    for name, c in calibrations.items():
        if c.boosted and c.k0 >= 0.999:
            sig0 = float(shared_gamd_globals_all.get(f"sigma0_{name}", 0.0) or 0.0)
            print(
                f"    WARNING: GaMD boost group {name!r} k0={c.k0:.3f} is saturated at 1.0 -- "
                f"the sigmaV<=sigma0 guardrail is inert, so sigma_DV is uncontrolled and "
                f"cumulant2 reweighting may remain unreliable even after self-consistent "
                f"calibration. Consider a smaller sigma0 (sigma0_{name}={sig0:.2f} kJ/mol)."
            )

    _check_sim = _check_integrator = _check_system = None
    try:
        _check_system = deserialize_system(openmm, base_system)
        _check_integrator, _ = make_gamd_integrator(_check_system, args, unit)
        set_integrator_globals_from_dict(_check_integrator, shared_gamd_globals_all)
        _check_sim = app.Simulation(topology, _check_system, _check_integrator, setup_platform, setup_props)
        _check_box = equil_state.getPeriodicBoxVectors()
        if _check_box is not None:
            _check_sim.context.setPeriodicBoxVectors(*_check_box)
        _check_sim.context.setPositions(equil_state.getPositions())
        _check_sim.context.setVelocitiesToTemperature(args.temperature_k * unit.kelvin, args.seed + 909)
        set_window(_check_sim.context, centers_nm, ks_kj_nm2, 0, secondary_cv_centers, secondary_cv_ks_kj)
        _check_controller = None
        if npt_runtime is not None and npt_runtime.needs_controller:
            _check_controller = npt_runtime.initialize_controller(
                _check_sim.context, seed=npt_seed_check_run(args)
            )
        _check_driver = ReplicaStepDriver(
            _check_sim, controller=_check_controller, label="gamd_calibration_check"
        )
        _check_driver.advance(50)
        _check_pe = physical_potential_energy_kj(_check_sim.context, _check_system, unit)
        if not math.isfinite(_check_pe):
            raise RuntimeError(
                f"Joint-envelope GaMD calibration produced a non-finite potential energy "
                f"({_check_pe}) on a 50-step verification run; refusing to proceed to production."
            )
        print(f"    GaMD joint-envelope calibration verified: 50-step check run finite (PE={_check_pe:.1f} kJ/mol)")
    finally:
        release_openmm_contexts(_check_sim, _check_integrator, _check_system)

    return shared_gamd_globals_all, shared_gamd_globals_interesting

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
    npt_runtime: Optional[NptRunContext] = None,
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

    shared_controller = None
    if npt_runtime is not None and npt_runtime.needs_controller:
        shared_controller = npt_runtime.initialize_controller(
            shared_sim.context, seed=npt_seed_shared_setup(args)
        )
    shared_driver = ReplicaStepDriver(
        shared_sim, controller=shared_controller, label="shared_gamd_setup"
    )

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
        shared_driver.advance(int(chunk))
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

    # F2: Extract Vmin/Vmax from calibration globals for downstream diagnostics.
    # gamd-openmm stores these as "Vmin_Total"/"Vmax_Total" (kJ/mol) in the
    # CustomIntegrator.  The exact key name is version-dependent, so we scan
    # case-insensitively and prefer keys containing "total".
    def _extract_vminmax(globals_dict: dict) -> tuple:
        """Return (vmin_kj, vmax_kj) from integrator globals, defaulting to nan."""
        vmin_kj = float("nan")
        vmax_kj = float("nan")
        candidates_min = [(k, v) for k, v in globals_dict.items() if "vmin" in k.lower()]
        candidates_max = [(k, v) for k, v in globals_dict.items() if "vmax" in k.lower()]
        # Prefer "total" variant if available, else take the first candidate
        def _pick(candidates):
            total_hits = [(k, v) for k, v in candidates if "total" in k.lower()]
            chosen = total_hits[0] if total_hits else (candidates[0] if candidates else None)
            if chosen is None:
                return float("nan")
            try:
                val = float(chosen[1])
                return val if math.isfinite(val) else float("nan")
            except Exception:
                return float("nan")
        return _pick(candidates_min), _pick(candidates_max)

    _vmin_kj, _vmax_kj = _extract_vminmax(shared_globals_all)

    payload = {
        "mode": "article_a_single_equilibrated_shared_gamd",
        "description": "One GaMD calibration/equilibration was run from the NPT-equilibrated peptide with umbrella k=0. The resulting same-name CustomIntegrator globals are copied to every GaREUS replica before production.",
        "calibration_steps": int(calib_steps),
        "temperature_K": float(args.temperature_k),
        "pressure_bar": float(getattr(args, "pressure_bar", 1.0)),
        "production_ensemble": str(getattr(args, "production_ensemble", "npt")),
        "production_barostat": _production_barostat_description(args),
        "production_barostat_frequency": int(getattr(args, "production_barostat_frequency", 0) or getattr(args, "barostat_frequency", 100)),
        "timestep_fs": float(args.timestep_fs),
        "gamd_boost_type": str(args.gamd_boost_type),
        "sigma0p_kcal_mol": float(args.sigma0p_kcal_mol),
        "sigma0d_kcal_mol": float(args.sigma0d_kcal_mol),
        "interesting_globals": shared_globals_interesting,
        "all_globals": shared_globals_all,
        "note": "If using a gamd-openmm version that stores additional production restart information outside CustomIntegrator globals, compare this JSON with the package's native restart/log files before production science.",
        "gamd_calibration_note": (
            "Single shared GaMD calibration performed with umbrella k=0 (free peptide). "
            "Vmin/Vmax are frozen from this run and copied to all production replicas. "
            "Windows sampling PEs far outside [Vmin, Vmax] will have saturated or zero boost; "
            "see calibration_range_warnings in the diagnostics output."
        ),
        "gamd_vmin_kj": _vmin_kj,
        "gamd_vmax_kj": _vmax_kj,
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


def reconcile_resume_secondary_cv_metadata(
    secondary_cv_metadata: Optional[dict], secondary_cv_centers, *,
    current_pair_sha256: Optional[str] = None,
) -> dict:
    """Reconcile a resumed run's secondary-CV metadata against its centers.

    The fast-resume path reads ``secondary_cv_metadata`` and
    ``secondary_cv_centers`` independently from the checkpoint manifest
    (they are stored as separate keys). If a 2D run resumes with
    ``secondary_cv_centers`` present and non-empty but ``secondary_cv_metadata``
    missing or ``enabled`` false/absent, CV2 seed scoring/biasing would
    silently drop out of the resumed run even though production windows are
    still 2D — the exact "silent collapse to a lower-dimensional proxy"
    regression this project forbids.

    This is a pure function (no I/O, no OpenMM): given the two
    independently-read values, it returns a metadata dict with ``enabled``
    reconciled to ``True`` whenever centers are present, preserving whatever
    other fields (mode, phi0_deg, psi0_deg, sigma_deg, torsion lists, ...)
    were already present. It always warns loudly when it has to reconcile;
    it never silently continues with CV2 dropped.

    A residual-torsion-pc run also pins its frozen pair model by the
    ``sha256`` digest recorded inside the checkpoint's
    ``pair_model_sha256`` field. When the current run's
    ``--secondary-cv-model`` resolves to a different digest, the resume is
    refused outright (a frozen pair is never redefined mid-campaign); when
    the current run carries no model, a WARNING is printed and the recorded
    digest is left untouched; when the checkpoint predates digest recording,
    the current digest is recorded as the first one.
    """
    meta = dict(secondary_cv_metadata or {})
    has_centers = secondary_cv_centers is not None and len(secondary_cv_centers) > 0
    if has_centers and not meta.get("enabled"):
        print(
            "!" * 78 + "\n"
            "WARNING [RESUME]: secondary_cv_centers present "
            f"({len(secondary_cv_centers)} windows) but secondary_cv_metadata is "
            f"missing/disabled (enabled={meta.get('enabled')!r}). This resumed run "
            "would silently drop CV2 from seed scoring/biasing. Reconciling by "
            "enabling secondary-CV metadata from the fields available in the "
            "checkpoint manifest so CV2 is not silently dropped.\n" + "!" * 78
        )
        meta["enabled"] = True
    recorded = meta.get("pair_model_sha256")
    if isinstance(recorded, str) and recorded:
        if current_pair_sha256 is None:
            print(
                "!" * 78 + "\n"
                "WARNING [RESUME]: the checkpoint recorded a frozen CV2 pair model "
                f"digest ({recorded}), but the current run carries no pair model "
                "(no --secondary-cv-model). Proceeding with the recorded digest.\n"
                + "!" * 78
            )
        elif current_pair_sha256 != recorded:
            raise RuntimeError(
                "resume refused: the frozen CV2 pair model changed "
                f"(checkpoint {recorded}, current {current_pair_sha256}); "
                "a frozen pair is never redefined mid-campaign -- point "
                "--secondary-cv-model back at the original artifact"
            )
    elif current_pair_sha256 is not None:
        meta["pair_model_sha256"] = current_pair_sha256
    return meta


def _find_adaptive_dir_root(out_dir: Path) -> Optional[Path]:
    """Walk up from a segment's own out_dir to find the campaign's adaptive_production/ root.

    Always named "adaptive_production" throughout this codebase (see
    adaptive_production.py's own adaptive_dir construction). Walking up
    rather than assuming a fixed nesting depth, since baseline/topup
    segments nest one level deeper than a plain epoch, and "final" one
    level deeper still.
    """
    for parent in Path(out_dir).resolve().parents:
        if parent.name == "adaptive_production":
            return parent
    return None


def _seed_bank_row_exists(seed_bank_dir: Path, seed_name: str) -> bool:
    csv_path = Path(seed_bank_dir) / "final_survivor_seeds.csv"
    if not csv_path.exists() or csv_path.stat().st_size <= 0:
        return False
    try:
        with csv_path.open(newline="") as handle:
            return any(row.get("seed_name") == seed_name for row in csv.DictReader(handle))
    except Exception:
        return False


def _augment_seed_bank_with_campaign_search(
    args, out_dir: Path, topology, centers_a, secondary_cv_centers,
) -> None:
    """For topup segments, search the campaign's own accumulated trajectory
    history for a real frame close to each window's target, adding one to
    the seed bank when found - before generate_us_starting_states_by_pulling
    runs its own per-window nearest-conformer selection over whatever the
    (otherwise static) GENPEPT library currently offers.

    Scoped to topup-tagged segments only (``_adaptive_phase_info["is_topup"]``):
    a baseline/initial-epoch window getting seeded wrong sets a worse
    foundation for everything the adaptive process subsequently builds on
    top of it, so those keep the stricter, unmodified library-only path. A
    topup is always re-seeding an ALREADY-established window, and the
    accumulated trajectory is exactly the kind of real data a static
    pre-generated library can never anticipate (confirmed on real data:
    chignolin_6 state 27, an original edge-of-range window, kept failing its
    seed-preflight check on every topup retry with only the static library
    to draw from).

    Never raises, and only ever ADDS a new seed-bank row - never removes or
    replaces anything - so generate_us_starting_states_by_pulling's own
    existing per-window nearest-conformer selection just naturally discovers
    and prefers it if (and only if) it scores closer than every existing
    candidate. Every fail-closed/ground-truth-verification guarantee in
    resolve_seed_frame_pdb/search_campaign_for_near_frame still applies -
    this can only ever help or be a no-op, never make a window's seeding
    worse than it already was.
    """
    phase_info = getattr(args, "_adaptive_phase_info", {}) or {}
    if not phase_info.get("is_topup"):
        return
    if secondary_cv_centers is None or centers_a is None or len(centers_a) == 0:
        return
    seed_bank_dir_value = getattr(args, "seed_conformers_dir", None)
    if not seed_bank_dir_value:
        return
    seed_bank_dir = Path(seed_bank_dir_value)
    if not seed_bank_dir.is_dir():
        return
    adaptive_dir = _find_adaptive_dir_root(out_dir)
    if adaptive_dir is None:
        return

    try:
        from .tica import search_campaign_for_near_frame
        from .seeding import _finite_spacing_scale
    except Exception:
        return

    primary_scale = _finite_spacing_scale(list(centers_a), fallback=1.0)
    finite_secondary = [
        float(c) for c in secondary_cv_centers if c is not None and math.isfinite(float(c))
    ]
    secondary_scale = _finite_spacing_scale(finite_secondary, fallback=0.25)
    max_score = float(getattr(args, "us_seed_preflight_max_score", 1.2) or 1.2)

    n_added = 0
    n_searched = 0
    for w in range(len(centers_a)):
        secondary_c = secondary_cv_centers[w] if secondary_cv_centers is not None else None
        if secondary_c is None or not math.isfinite(float(secondary_c)):
            continue
        primary_c = float(centers_a[w])
        secondary_c = float(secondary_c)
        seed_name = f"campaign_search_p{primary_c:.4f}_s{secondary_c:.4f}"
        if _seed_bank_row_exists(seed_bank_dir, seed_name):
            continue
        n_searched += 1
        pdb_path = seed_bank_dir / "pdbs" / f"{seed_name}.pdb"
        try:
            info = search_campaign_for_near_frame(
                adaptive_dir, topology, primary_c, secondary_c,
                primary_scale, secondary_scale, max_score, pdb_path,
            )
        except Exception as exc:
            print(f"WARNING: campaign seed search failed for window {w} ({exc})")
            continue
        if info is None:
            continue
        try:
            from .adaptive_production import _append_seed_bank_row
            _append_seed_bank_row(seed_bank_dir, {
                "seed_name": seed_name,
                "survivor_pdb_path": str(Path("pdbs") / pdb_path.name),
                "source_run_dir": info["seed_frame"]["epoch_dir"],
                "source_pdb_path": info["source_xtc"],
                "source_label": "campaign_search_topup",
                "source_state_id": "",
                "source_epoch_window": info["frame_index"],
                "primary_cv_value": primary_c,
                "secondary_cv_value": secondary_c,
            })
        except Exception as exc:
            print(f"WARNING: failed to register campaign-search seed for window {w} ({exc})")
            continue
        n_added += 1
    if n_searched:
        print(
            f"    Campaign seed search (topup): searched {n_searched} window(s) with no prior "
            f"campaign-search seed, added {n_added} real extracted frame(s) to {seed_bank_dir}"
        )


def run_gareus(args, out_dir: Path, openmm, app, unit, forcefield, topology, equil_state, progress: Optional[GuiProgressSink] = None):
    from .correctness.sampling_policy import require_rescue_disabled_in_current_driver
    require_rescue_disabled_in_current_driver(args)
    _register_graceful_shutdown()
    acquire_run_lock(out_dir)
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
        # secondary_cv_metadata is already reconciled against secondary_cv_centers
        # inside load_resume_run_definition (against the raw, pre-null values —
        # see the comment there for why it cannot be done here).
        secondary_cv_metadata = dict(resume_def.get("secondary_cv_metadata", {"enabled": False}) or {"enabled": False})
        secondary_cv_centers = resume_def.get("secondary_cv_centers")
        secondary_cv_k_kcal_list = resume_def.get("secondary_cv_k_kcal_list")
        if secondary_cv_metadata.get("enabled"):
            # Resume must rebuild the same optional CV force, including linear
            # torsion state files, even if the user omits CV flags.
            _restore_secondary_cv_args_from_metadata(args, secondary_cv_metadata, out_dir=out_dir)
        window_metadata = dict(resume_def.get("window_metadata", {}))
        shared_gamd_globals_all = dict(resume_def.get("shared_gamd_globals_all", {}) or {})
        shared_gamd_globals_interesting = dict(resume_def.get("shared_gamd_globals_interesting", {}) or {})
        calib_steps = int(resume_def.get("calib_steps", 0) or 0)
        bootstrap_torsion_summary = {}
        shared_gamd_context_checkpoint = None
        print(f"[resume] Production checkpoint manifest found; skipping window generation, US pulling, and shared GaMD setup.")
    else:
        if equil_state is None:
            raise RuntimeError("No equilibrated state is available; cannot start GaREUS without a production checkpoint or saved 03_npt_equilibrated_state.xml.")
        cv_atom1, cv_atom2, distance_cv_label = choose_cv_atoms(topology, args)
        primary_cv_def = prepare_primary_cv_definition(topology, args, cv_atom1=cv_atom1, cv_atom2=cv_atom2, cv_label=distance_cv_label)
        cv_label = str(primary_cv_def.get("label", distance_cv_label))
        bootstrap_torsion_summary = _ensure_bootstrap_torsion_cv_ready(args, out_dir, topology, primary_cv_def)
        if getattr(args, "windows_2d_csv", None):
            centers_a, k_list, secondary_cv_centers, secondary_cv_k_kcal_list, secondary_cv_metadata, window_metadata = load_explicit_2d_window_csv(args, Path(args.windows_2d_csv))
            args.state_gamd_lambdas = list(secondary_cv_metadata.get("gamd_lambdas") or [0.0] * len(centers_a))
            # Captured BEFORE the filter: it returns only the survivors, and the
            # map correction below needs the index space the drop was expressed in.
            _n_windows_before_reachability_filter = int(len(centers_a))
            centers_a, k_list, secondary_cv_centers, secondary_cv_k_kcal_list, secondary_cv_metadata, window_metadata = filter_explicit_2d_windows_by_seed_reachability(
                args, out_dir, topology, primary_cv_def,
                centers_a, k_list, secondary_cv_centers, secondary_cv_k_kcal_list,
                secondary_cv_metadata, window_metadata,
            )
            # The filter just renumbered every window-indexed array around its
            # survivors, which invalidates the epoch_window_map.csv the adaptive
            # driver wrote into this phase directory before the sub-run started.
            # This is the SECOND drop path that does that (the post-pull auto-drop
            # is the other) and it is the one that shifted chignolin_6's
            # epoch_001/topup_004 and final/topup_003 - phases with no post-pull
            # drop record at all, so the post-pull rewrite could never fix them.
            _unreachable_map_rewrite = rewrite_epoch_window_map_after_unreachable_filter(
                out_dir, secondary_cv_metadata, window_metadata,
                _n_windows_before_reachability_filter,
            )
            if _unreachable_map_rewrite is not None:
                window_metadata = dict(window_metadata or {})
                window_metadata["epoch_window_map_rewrite_unreachable_filter"] = _unreachable_map_rewrite
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
    if bootstrap_torsion_summary:
        window_metadata["bootstrap_torsion_cv"] = bootstrap_torsion_summary
    window_metadata.update({
        "primary_cv": primary_cv_mode(args),
        "primary_cv_label": primary_cv_label(args),
        "primary_cv_units": primary_cv_units(args),
        "primary_k_units": primary_k_units(args),
        "primary_openmm_k_units": primary_openmm_k_units(args),
        "primary_cv_definition": _json_ready(primary_cv_def),
    })
    # Controller ruling (Task 9, Part B; corrected after review): reload a
    # frozen ladder from run_manifest.json BEFORE _derive_state_gamd_lambdas
    # runs, not after. _derive_state_gamd_lambdas's own fallback-of-last-resort
    # is [0.0] * n -- a non-empty (hence truthy) list -- so calling the reload
    # afterward meant its unset-guard always saw an already-"set" value and
    # never actually read the manifest on any real run (the bug a review
    # caught: the reload was dead code in practice). Placed here, the reload
    # only ever sees args.state_gamd_lambdas in its true pre-derivation state
    # (unset on the fast-resume/choose_windows paths; already populated from
    # the window table on the explicit-2D path, which the reload correctly
    # leaves alone), so _derive_state_gamd_lambdas's `existing=` below
    # naturally receives whatever the reload restored.
    _reload_state_gamd_lambdas_on_resume(args, out_dir)

    # Re-derive args.state_gamd_lambdas from the (possibly reachability-filtered
    # and reindexed) normalized_rows rather than trusting the value set right
    # after load_explicit_2d_window_csv: filter_explicit_2d_windows_by_seed_reachability
    # can drop and renumber rows, and each surviving normalized_row already
    # carries its own correct "gamd_lambda", so this stays aligned with
    # centers_a/k_list no matter which branch (explicit-2D, choose_windows,
    # or fast-resume) produced them.
    args.state_gamd_lambdas = _derive_state_gamd_lambdas(
        window_metadata, len(centers_a), existing=getattr(args, "state_gamd_lambdas", None)
    )
    # initialize_run_manifest() (cli.py, before minimize_and_npt_equilibrate/
    # run_gareus are ever called) runs _method_settings(args) long before
    # args.state_gamd_lambdas exists as an attribute at all -- that dict
    # comprehension is `{k: getattr(args, k, None) for k in keys if
    # hasattr(args, k)}`, so a not-yet-set attribute is silently OMITTED, not
    # written as null. finalize_run_manifest() never recomputes
    # method_settings either (it only patches status/end_time/artifact
    # hashes), so without this patch "state_gamd_lambdas" never appears in
    # run_manifest.json on ANY run -- silently defeating
    # _reload_state_gamd_lambdas_on_resume's manifest read on every --resume.
    # Patch it in now that this stage's value is known -- NOT the final word,
    # though: --max-replicas truncation and the post-pull US auto-drop
    # re-derive (further below) can still reassign args.state_gamd_lambdas,
    # each of which re-calls _persist_state_gamd_lambdas so the manifest
    # always reflects the LAST mutation, not this first one.
    _persist_state_gamd_lambdas(args, out_dir)
    window_rows = window_assignment_rows(centers_a, k_list, args.temperature_k, secondary_cv_centers, secondary_cv_k_kcal_list, args=args, gamd_lambdas=args.state_gamd_lambdas)
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
            gamd_lambdas=args.state_gamd_lambdas,
        )
        window_metadata = dict(window_metadata or {})
        window_metadata["explicit_window_table"] = explicit_window_table_summary
        print(f"    Sparse-safe window metadata: {explicit_window_table_summary.get('csv', '')}")
    except Exception as exc:
        explicit_window_table_summary = {}
        print(f"WARNING: failed to write sparse-safe explicit window metadata: {exc}")

    # umbrella_explicit_windows.csv has just been written from the window arrays
    # this process will actually run - which, on the checkpoint fast-resume branch
    # above, came from the checkpoint manifest's own POST-drop windows_A.  So this
    # is the first point at which an interrupted phase can tell that the
    # epoch_window_map.csv the driver rewrote from the (still un-pruned) registry
    # on restart covers more windows than the phase really runs.  A resumed
    # segment never re-runs the pull, hence never re-runs the drop that corrected
    # the map the first time, so without this the drop-time rewrites above
    # contribute nothing at all to any interrupted phase - the normal case on a
    # multi-day run.  A no-op whenever the map already agrees with the window set.
    _map_repair = repair_epoch_window_map_from_surviving_windows(
        out_dir, source="fast_resume_surviving_window_table" if fast_resume else "surviving_window_table",
    )
    if _epoch_window_map_repair_must_be_recorded(_map_repair):
        window_metadata = dict(window_metadata or {})
        window_metadata["epoch_window_map_repair_from_window_table"] = _map_repair

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
        if getattr(args, "state_gamd_lambdas", None) is not None:
            args.state_gamd_lambdas = list(args.state_gamd_lambdas)[:_max_replicas]
            # Re-patch: the manifest snapshot written earlier (right after the
            # initial derive) is now stale-length -- see _persist_state_gamd_lambdas'
            # docstring for why persisting a pre-truncation value would corrupt
            # a later --resume.
            _persist_state_gamd_lambdas(args, out_dir)
        nrep = _max_replicas

    # The rung dimension: one gamd_lambda per surviving state, aligned with
    # centers_a/k_list/nrep above (including any --max-replicas truncation).
    state_lambdas = np.asarray(getattr(args, "state_gamd_lambdas", None) or [0.0] * nrep, dtype=float)
    if state_lambdas.size != nrep:
        raise ValueError(f"state_gamd_lambdas has {state_lambdas.size} entries for {nrep} states")
    # Campaign-scoped, not sub-run-scoped: see resolve_ladder_active.
    ladder_active = resolve_ladder_active(state_lambdas, out_dir)
    if ladder_active and not np.any(state_lambdas > 0.0):
        _reg = campaign_ladder_registry_lambda(out_dir)
        print(f"    λ-ladder: this sub-run holds only λ=0 states, but {_reg[0]} shows the "
              f"campaign runs a ladder (λ up to {_reg[1]:g}); keeping the ladder path so these "
              "replicas get k0 = 0 and their v_pep/v_dih are recorded")
    if ladder_active and not ladder_supports_boost_type(args):
        raise ValueError("a gamd_lambda ladder requires --gamd-boost-type pep-gamd-lower-dual or lower-dihedral")

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

    # NPT correction (spec section 5): barostat ownership is decided BEFORE the
    # base System is built, so a biased_mc run never creates a Context over a
    # System that still carries the native MonteCarloBarostat.  The ownership
    # object is the single source of truth for every downstream construction
    # site (recon windows, shared GaMD setup, production replicas, probe).
    barostat_ownership = resolve_barostat_ownership(
        args, ensemble=production_ensemble,
        run_mode=str(getattr(args, "run_mode", "cmd") or "cmd"),
        boost_type=str(getattr(args, "gamd_boost_type", "") or ""),
    )
    production_include_barostat = barostat_ownership.include_native_barostat
    production_barostat_frequency = barostat_ownership.barostat_frequency

    # Base production system for GaREUS. Default production is now NPT with
    # OpenMM's MonteCarloBarostat; --production-ensemble nvt preserves the old
    # fixed-box behavior explicitly.
    base_system = create_system(
        app, unit, forcefield, topology, args,
        include_barostat=production_include_barostat,
        barostat_frequency=production_barostat_frequency,
    )
    prepare_pep_gamd_args(args, topology)
    add_primary_umbrella_force(openmm, base_system, primary_cv_def, args, args.umbrella_force_group)
    secondary_cv_force_info = add_secondary_structure_cv_force(
        openmm, base_system, topology, args, force_group=int(getattr(args, "secondary_cv_force_group", 29)),
        primary_cv_def=primary_cv_def,
    ) if (secondary_cv_metadata or {}).get("enabled") else {"enabled": False}
    if secondary_cv_force_info.get("enabled"):
        secondary_cv_metadata.update(secondary_cv_force_info)
        print(
            f"    Secondary CV enabled: {secondary_cv_metadata.get('label', 'secondary structure')} "
            f"with {secondary_cv_metadata.get('n_phi_torsions', 0)} phi and {secondary_cv_metadata.get('n_psi_torsions', 0)} psi torsions"
        )

    # NPT correction: the physical preflight runs on the fully-assembled
    # application-controlled System (umbrella + secondary CV forces included),
    # before the first Context is created from it.  Also yields the ownership
    # report recorded into the run manifest below.
    barostat_preflight = preflight_barostat_ownership(base_system, barostat_ownership)

    # NPT correction (spec section 5): the effective-potential adapter is the
    # one cross-package seam to package 2; it only needs to exist when the
    # biased-MC controller will evaluate volume-move acceptance energies.
    npt_adapter = _resolve_npt_adapter(args, base_system) if barostat_ownership.backend == "biased_mc" else None
    npt_runtime = NptRunContext(ownership=barostat_ownership, adapter=npt_adapter)

    # GaMD production-envelope recalibration: adaptive-production's epoch 0 already
    # runs real GaMD-boosted sampling to bootstrap tICA, so it is also the cheapest
    # place to measure the boost groups' *actual* potential-energy envelope and
    # recalibrate the shared k0/threshold for epoch 1+ instead of trusting the
    # throwaway calibration recon forever. Opt-in per-worker via _adaptive_phase_info
    # (set by adaptive_production.py); every other run pays zero extra cost.
    _phase_info = getattr(args, "_adaptive_phase_info", {}) or {}
    try:
        _phase_epoch_index = int(_phase_info.get("epoch_index", -1))
    except (TypeError, ValueError):
        _phase_epoch_index = -1
    _gamd_recal_active = bool(
        use_gamd
        and _phase_info.get("is_adaptive_epoch")
        and _phase_epoch_index == 0
        and getattr(args, "adaptive_production_gamd_recalibrate_after_epoch0", True)
    )
    _gamd_recal_targets: list[tuple[str, Optional[int]]] = []
    _gamd_recal_accumulators: dict[str, dict[int, Any]] = {}
    if _gamd_recal_active:
        from gareus.gamd_calibration import WelfordAccumulator as _WelfordAccumulator
        _peek_system = deserialize_system(openmm, base_system)
        _gamd_recal_targets, _peek_integrator = _gamd_boost_group_targets(_peek_system, args, unit)
        release_openmm_contexts(_peek_integrator, _peek_system)
        _gamd_recal_accumulators = {name: {} for name, _gid in _gamd_recal_targets}
        print(
            "    GaMD production-envelope recalibration: sampling boost-group energies "
            f"this epoch for groups {[name for name, _gid in _gamd_recal_targets]} "
            "(will recalibrate the shared envelope for epoch 1+)"
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
            add_secondary_structure_cv_force(openmm, starting_structure_system, topology, args,
                                             force_group=int(getattr(args, "secondary_cv_force_group", 29)),
                                             primary_cv_def=primary_cv_def)
        pos = equil_state.getPositions()
        vel = equil_state.getVelocities()
        box = equil_state.getPeriodicBoxVectors()

        _augment_seed_bank_with_campaign_search(args, out_dir, topology, centers_a, secondary_cv_centers)

        window_start_positions, window_start_velocities, dropped_window_indices = generate_us_starting_states_by_pulling(
            args, out_dir, openmm, app, unit, topology, starting_structure_system, centers_nm, ks_kj_nm2,
            equil_state, primary_cv_def, cv_atom1, cv_atom2, setup_platform, setup_props, progress=progress,
            secondary_cv_centers=secondary_cv_centers, secondary_cv_ks_kj=secondary_cv_ks_kj,
            secondary_cv_metadata=secondary_cv_metadata,
        )

        if dropped_window_indices:
            _drop_result = drop_bad_us_windows_and_rebuild(
                out_dir, dropped_window_indices,
                centers_a, k_list, centers_nm, ks_kj_nm2,
                secondary_cv_centers, secondary_cv_k_kcal_list, secondary_cv_ks_kj,
                window_start_positions, window_start_velocities,
                secondary_cv_metadata, window_metadata, args,
            )
            centers_a = _drop_result["centers_a"]
            k_list = _drop_result["k_list"]
            centers_nm = _drop_result["centers_nm"]
            ks_kj_nm2 = _drop_result["ks_kj_nm2"]
            secondary_cv_centers = _drop_result["secondary_cv_centers"]
            secondary_cv_k_kcal_list = _drop_result["secondary_cv_k_kcal_list"]
            secondary_cv_ks_kj = _drop_result["secondary_cv_ks_kj"]
            window_start_positions = _drop_result["window_start_positions"]
            window_start_velocities = _drop_result["window_start_velocities"]
            secondary_cv_metadata = _drop_result["secondary_cv_metadata"]
            window_metadata = _drop_result["window_metadata"]
            # drop_bad_us_windows_and_rebuild recomputed the explicit window table
            # over the survivors; pick that up here too, or the local summary
            # written into gareus_metadata.json below stays at the PRE-drop
            # n_windows while every array beside it is POST-drop (the real run this
            # was found on reported n_windows 27 next to 24 actual windows).
            _drop_table_summary = (window_metadata or {}).get("explicit_window_table")
            if _drop_table_summary:
                explicit_window_table_summary = _drop_table_summary
            nrep = len(centers_nm)

            # The post-pull auto-drop can remove/restore arbitrary (non-suffix)
            # indices, so state_lambdas computed before this point is stale and
            # potentially misaligned. drop_bad_us_windows_and_rebuild already
            # subsets+renumbers window_metadata["normalized_rows"] using its own
            # (possibly connectivity-restored) keep set via _resubscript_normalized_rows
            # -- re-derive from that rather than guessing the keep set here.
            # existing=None deliberately: the pre-drop args.state_gamd_lambdas is not
            # safe to reuse across an arbitrary (non-suffix) index drop.
            _prev_state_gamd_lambdas = getattr(args, "state_gamd_lambdas", None)
            args.state_gamd_lambdas = _derive_state_gamd_lambdas(window_metadata, len(centers_a), existing=None)
            _warn_if_ladder_was_zeroed(_prev_state_gamd_lambdas, args.state_gamd_lambdas,
                                       "post-pull US auto-drop state_gamd_lambdas re-derive")
            # Re-patch: see _persist_state_gamd_lambdas' docstring -- the manifest
            # must reflect this post-drop, re-derived value, not whatever an
            # earlier call (initial derive, or --max-replicas truncation) wrote.
            _persist_state_gamd_lambdas(args, out_dir)
            state_lambdas = np.asarray(args.state_gamd_lambdas, dtype=float)
            if state_lambdas.size != nrep:
                raise ValueError(f"state_gamd_lambdas has {state_lambdas.size} entries for {nrep} states after US auto-drop")
            # Campaign-scoped here too: an auto-drop can leave a run holding only
            # λ=0 states, which must not silently demote it off the ladder path.
            ladder_active = resolve_ladder_active(state_lambdas, out_dir)
            if ladder_active and not ladder_supports_boost_type(args):
                raise ValueError("a gamd_lambda ladder requires --gamd-boost-type pep-gamd-lower-dual or lower-dihedral")

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
                    args, out_dir, openmm, app, unit, topology, base_system, gamd_start_state, setup_platform, setup_props, progress=progress,
                    npt_runtime=npt_runtime,
                )
                shared_gamd_globals_all, shared_gamd_globals_interesting = apply_joint_envelope_gamd_calibration(
                    args, openmm, app, unit, topology, base_system, setup_platform, setup_props,
                    out_dir, nrep, centers_nm, ks_kj_nm2, secondary_cv_centers, secondary_cv_ks_kj,
                    window_start_positions, window_start_velocities, equil_state,
                    shared_gamd_globals_all, shared_gamd_globals_interesting, calib_steps,
                    progress=progress,
                    npt_runtime=npt_runtime,
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

    # The λ-ladder rung dimension: k0max_by_channel is the top-rung (λ=1) k0 for
    # each boost channel, read once from the shared calibrated globals. Only
    # non-None when the ladder is actually active on a Pep-GaMD run -- every other
    # run mode (plain GaMD, no ladder, non-pep-gamd) leaves every replica's k0 at
    # the shared globals, exactly as before this feature existed.
    k0max_by_channel = k0max_from_globals(shared_gamd_globals_all) if (use_gamd and ladder_active) else None
    # The per-state boost envelope, built once from the same calibrated globals
    # every replica's k0max is drawn from above. None whenever the ladder is not
    # active, in which case the exchange bias matrix carries no boost term at all
    # (an all-zero contribution) and every other run mode is unaffected.
    pep_env = PepGamdEnvelope.from_integrator_globals(shared_gamd_globals_all) if (use_gamd and ladder_active) else None

    # _ReplicaAffinityExecutor lives at module scope (see its docstring) so the
    # per-replica thread-affinity invariant it exists to enforce can be unit
    # tested directly; it was function-local here, which left that invariant -
    # a driver-level SIGSEGV hazard - with no coverage at all.
    _sim_pool = _ReplicaAffinityExecutor(nrep)

    sims = []
    drivers: list = []
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

        def _build_context_i():
            # Runs entirely on replica i's dedicated _sim_pool thread: Context
            # construction and every context-touching call below must happen on
            # the same OS thread that will later step() this replica, so the
            # CUDA driver never has to migrate a Context to a different thread
            # on first use (see _ReplicaAffinityExecutor docstring above).
            try:
                sim_i = app.Simulation(topology, system_i, integrator_i, platform, props_i)
            except Exception:
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
            loaded_checkpoint = False
            copied, skipped = {}, {}
            if (
                use_gamd and not fast_resume and shared_gamd_context_checkpoint is not None
                and bool(getattr(args, "gamd_reuse_context_checkpoint", False))
            ):
                try:
                    # Loading the calibrated shared Context checkpoint preserves the
                    # gamd-openmm native stage/step state in addition to readable
                    # CustomIntegrator globals.  We overwrite coordinates, velocities,
                    # and umbrella parameters below, so only the GaMD setup state is reused.
                    #
                    # Off by default: loadCheckpoint() restores a binary snapshot into a
                    # freshly deserialize_system()'d Context for this epoch's (possibly
                    # smaller, reachability-/auto-drop-filtered) window set. Confirmed on
                    # a live run to segfault at the first step() of every epoch that reused
                    # a checkpoint exported from a different epoch's Context - reproduced
                    # with clean-quality windows, MPS on/off, PME-stream on/off, and down
                    # to 1 replica/GPU, so it is not a data-quality or concurrency issue.
                    # The globals-dict copy below is authoritative for calibration anyway
                    # (see comment there); the checkpoint only adds opaque native stage
                    # state on top, which is not worth this crash risk by default.
                    sim_i.context.loadCheckpoint(shared_gamd_context_checkpoint)
                    loaded_checkpoint = True
                except Exception as exc:
                    skipped["<shared_context_checkpoint>"] = f"load failed; falling back to CustomIntegrator globals only: {exc}"
            if use_gamd and not fast_resume:
                # Always copy calibrated globals from the dict regardless of whether the
                # checkpoint loaded.  loadCheckpoint() can silently succeed without
                # actually restoring CustomIntegrator globals (e.g. cross-platform or
                # cross-context mismatch), leaving Vmax/Vmin/k0/ForceScalingFactor at
                # zero/default and causing NaN forces at step 0.  The dict copy is
                # authoritative for calibration outputs; the checkpoint provides the
                # opaque GaMD binary stage state as a bonus.
                if shared_gamd_globals_all:
                    copied, copied_skipped = set_integrator_globals_from_dict(integrator_i, shared_gamd_globals_all)
                    skipped.update({k: v for k, v in copied_skipped.items() if k not in skipped})
                set_replica_lambda_for_window(integrator_i, i, state_lambdas, k0max_by_channel)
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
            controller_i = None
            if npt_runtime is not None and npt_runtime.needs_controller and not fast_resume:
                # NPT correction: the biased-MC volume controller is initialized
                # on this replica's own worker, with an INDEPENDENT random
                # stream (spec section 6: never clone one controller seed
                # into every replica).  fast_resume restores it from the
                # checkpoint instead (load_production_checkpoint).
                controller_i = npt_runtime.initialize_controller(
                    sim_i.context, seed=npt_seed_replica(args, i)
                )
            return sim_i, loaded_checkpoint, copied, skipped, controller_i

        sim_i, loaded_shared_gamd_checkpoint, copied_globals, skipped_globals, controller_i = _sim_pool.submit(i, _build_context_i).result()
        replica_gamd_copy_report.append({
            "replica": int(i),
            "copied_count": int(len(copied_globals)),
            "skipped_count": int(len(skipped_globals)),
            "copied_globals": copied_globals if i == 0 else {},
            "skipped_globals": skipped_globals if i == 0 else {},
            "loaded_from_context_checkpoint": bool(fast_resume),
            "loaded_from_shared_gamd_setup_checkpoint": bool(loaded_shared_gamd_checkpoint),
        })
        # NPT correction: every replica steps through a ReplicaStepDriver.  The
        # driver owns the ordering integrate -> finish due volume move ->
        # report; the reporters therefore live on the driver, never on
        # sim.reporters (Simulation.step would otherwise fire them on the
        # pre-move state and again inside the driver, duplicating frames).
        # on_volume_move clears the CV cache after EVERY attempted move
        # (accepted or rejected: even a rejection rescales coordinates and
        # invalidates cached CustomCVForce values) -- late-binding closure;
        # observable_cache is assigned before the first advance() runs.
        def _on_volume_move(_result, _i=i):
            observable_cache.clear()

        driver_i = npt_runtime.make_driver(
            sim_i, controller=controller_i, on_volume_move=_on_volume_move, label=f"replica_{i:03d}"
        )
        if effective_traj_interval > 0 and not bool(getattr(args, "resume", False)):
            reporter = make_trajectory_reporter(app, traj_dir / f"replica_{i:03d}", effective_traj_interval, args, atom_subset=traj_atom_subset)
            if reporter is not None:
                driver_i.register_reporter(reporter)
        sims.append(sim_i)
        drivers.append(driver_i)
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

    # ── tICA dihedral observation buffers (optional, gated by tica_obs_interval) ─
    _tica_obs_interval = int(getattr(args, "tica_obs_interval", 0) or 0)
    _dihedral_buffers = None
    if _tica_obs_interval > 0:
        from .tica import DihedralObsBuffer
        _phi_obs_tors, _psi_obs_tors = secondary_structure_torsions(topology)
        _dihedral_buffers = [
            DihedralObsBuffer(_phi_obs_tors, _psi_obs_tors, r, out_dir)
            for r in range(nrep)
        ]
        _tica_obs_counter = [0]  # mutable int in a list so the closure can update it
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
        # See compare_gamd_global_sets' ignore_names docstring: k0_Total/k0_Dihedral
        # are rescaled per-rung by set_replica_lambda_for_window right after this
        # same shared-globals copy, so they must be excluded from the comparison
        # whenever the ladder is active -- otherwise every replica assigned a
        # window with lambda != 1 reports a false copied-global "mismatch".
        _copy_sanity_ignore = frozenset({"k0_Total", "k0_Dihedral"}) if ladder_active else frozenset()
        for i, sim in enumerate(sims):
            cmp = compare_gamd_global_sets(shared_gamd_globals_all, all_integrator_globals(sim.integrator), ignore_names=_copy_sanity_ignore)
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
        "production_barostat": _production_barostat_description(args),
        "production_pressure_bar": float(getattr(args, "pressure_bar", 1.0)),
        "production_barostat_frequency": int(production_barostat_frequency) if production_include_barostat else 0,
        "npt_backend": barostat_ownership.backend,
        "npt_barostat_requested": barostat_ownership.requested,
        "barostat_volume_step_fraction": float(barostat_ownership.volume_step_fraction),
        "barostat_preflight": barostat_preflight,
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
        "production_barostat": _production_barostat_description(args),
        "production_barostat_frequency": int(production_barostat_frequency) if production_include_barostat else 0,
        "npt_backend": barostat_ownership.backend,
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
        "energy_column_semantics": {
            "potential_kj_mol": "physical force groups only (excludes the Pep-GaMD auxiliary water-only force and the boost); read via getState(groups=physical_energy_groups_for_args)",
            "v_pep_kj_mol": "Pep-GaMD peptide boost-target energy (Total channel), physical groups minus the water-only auxiliary group",
            "v_dih_kj_mol": "Pep-GaMD dihedral-channel boost-target energy",
        },
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
            "umbrella_bias_all_windows_kcal_mol_json": "optional list of the TOTAL bias of sample r_n under every state i, kcal/mol: the umbrella term U_i(r_n) PLUS, on a λ-ladder run, state i's own Pep-GaMD boost of this configuration (assemble_bias_matrices folds the boost in). Umbrella-only on every non-ladder run. Written only with --write-full-bias-csv-vectors",
            "umbrella_bias_all_windows_kj_mol_json": "same total (umbrella + λ-ladder boost) per state i, kJ/mol; written only with --write-full-bias-csv-vectors",
            "umbrella_reduced_bias_all_windows_json": "same total reduced by beta, dimensionless -- beta*(U_i(r_n) + boost_i(r_n)); this is the u_nk row MBAR consumes. Written only with --write-full-bias-csv-vectors",
            "gamd_boost_total_kj_mol": "GaMD boost estimate. Preferred source is gamd-openmm integrator.get_boost_potentials(); fallback is named CustomIntegrator globals.",
            "gamd_boost_source": "get_boost_potentials, integrator_globals, or unavailable",
            "gamd_boost_components_kj_mol_json": "component boost potentials from gamd-openmm native get_boost_potentials(), kJ/mol",
            "umbrella_bias_kcal_mol": "TOTAL bias of this sample under its OWN assigned state, kcal/mol: the umbrella term plus, on a λ-ladder run, that state's Pep-GaMD boost of this configuration. Umbrella-only on every non-ladder run; see sampled_umbrella_bias_kj for the umbrella term alone",
            "umbrella_bias_kj_mol": "the same own-state total in kJ/mol",
            "sampled_umbrella_bias_kj": "umbrella-only component (primary + secondary CV) of the sampled window's bias, kJ/mol; excludes the λ-ladder Pep-GaMD boost even when umbrella_bias_kj_mol/umbrella_bias_kcal_mol carry it",
            "sampled_boost_bias_kj": "Pep-GaMD boost of this replica's configuration under its own assigned window's λ, kJ/mol; zero on every run where the λ-ladder is not active",
            "v_pep_kj_mol": "raw peptide-dihedral+nonbonded channel energy (gamd-openmm 'total potential energy of the boosted group') at this replica's configuration, kJ/mol; NaN when the λ-ladder is not active",
            "v_dih_kj_mol": "raw peptide-dihedral-only channel energy at this replica's configuration, kJ/mol; NaN when the λ-ladder is not active",
            "gamd_lambda": "the λ-ladder boost strength of the state/window this sample was assigned to; 0.0 on every run where the λ-ladder is not active, and identically 0.0 on λ=0 rungs even when the ladder is active",
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
    _win_snapshot_windows = snapshot_window_rows(
        centers_a[:nrep], k_list[:nrep], secondary_cv_centers, secondary_cv_k_kcal_list,
        getattr(args, "state_gamd_lambdas", None), n=nrep,
    )
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
    # Conservative guard: exploratory rescue needs separate policy/ledger wiring.
    # No contact-CV default or final-production override may mutate configurations.
    _stuck_enabled = False
    _stuck_threshold = float(getattr(args, "cv1_stuck_threshold", 0.03) or 0.03)
    _stuck_max_intervals = int(getattr(args, "cv1_stuck_detect_intervals", 500) or 500)
    _stuck_counter = np.zeros(nrep, dtype=int)
    _stuck_rescue_total = 0

    # _sim_pool (one thread per replica, pinned since before replica_construction)
    # was created above, before Context construction, so every replica's Context
    # is built and stepped on the same OS thread for its whole lifetime.
    if str(getattr(args, "platform", "")).upper() == "CPU" and int(getattr(args, "cpu_threads", 1)) == 0 and nrep > 1:
        print(
            f"WARNING: --cpu-threads 0 (use all cores) combined with {nrep} parallel replicas "
            "will oversubscribe CPU cores. Set --cpu-threads 1 for parallel step_all()."
        )
    # Sentinel variables initialised here so the finally block can always read
    # them safely, even if the try body raises before the loop sets prod_done or
    # prod_total.
    _prod_completed_cleanly = False
    prod_done = 0
    prod_total = 0
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
            nonlocal _gamd_recal_active
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
            v_pep_kj = np.empty(nrep, dtype=np.float64)
            v_dih_kj = np.empty(nrep, dtype=np.float64)

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
                        state = ctx.getState(getEnergy=True, enforcePeriodicBox=True,
                                             groups=physical_energy_groups_for_args(args))
                        pe = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
                    else:
                        pe = float("nan")
                    v_pep, v_dih = _fetch_v_pep_v_dih(ctx, pep_env, unit)
                    return r, cv, ss, pe, v_pep, v_dih
                cv, ss, pe = primary_secondary_and_potential_from_state(
                    sim.context, primary_cv_def, args, unit, secondary_cv_metadata,
                    read_potential_energy=read_sample_potential,
                )
                v_pep, v_dih = _fetch_v_pep_v_dih(sim.context, pep_env, unit)
                return r, cv, ss, pe, v_pep, v_dih

            for r, cv, ss, pe, v_pep, v_dih in _sim_pool.map(_fetch_state, enumerate(sims)):
                primary_values[r], ss_values[r], potentials_kj[r] = cv, ss, pe
                v_pep_kj[r], v_dih_kj[r] = v_pep, v_dih
            primary_delta_matrix = primary_values[np.newaxis, :] - centers_a_arr[:, np.newaxis]
            distance_bias_matrix_kcal = 0.5 * k_arr[:, np.newaxis] * primary_delta_matrix * primary_delta_matrix
            if secondary_cv_centers is not None and secondary_cv_k_kcal_list is not None:
                ss_centers_arr = ss_centers_arr_global
                ss_k_arr = ss_k_kcal_arr_global
                ss_delta_matrix = ss_values[np.newaxis, :] - ss_centers_arr[:, np.newaxis]
                ss_bias_matrix_kcal = 0.5 * ss_k_arr[:, np.newaxis] * ss_delta_matrix * ss_delta_matrix
                # Guard NaN: windows without a secondary center (ss_centers_arr init to NaN)
                # or replicas with a missing secondary value produce NaN in the Metropolis term,
                # silently freezing that replica's exchanges and corrupting the cached bias matrix.
                # Zero those entries — their exchange criterion falls back to primary CV only.
                ss_bias_matrix_kcal = np.where(np.isfinite(ss_delta_matrix), ss_bias_matrix_kcal, 0.0)
            else:
                ss_centers_arr = ss_centers_arr_global
                ss_k_arr = ss_k_kcal_arr_global
                ss_bias_matrix_kcal = np.zeros_like(distance_bias_matrix_kcal)
            boost_bias_matrix_kj = (pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, state_lambdas, pep_env)
                                    if pep_env is not None else np.zeros((nrep, nrep)))
            bias_matrix_kcal, bias_matrix_kj = assemble_bias_matrices(
                distance_bias_matrix_kcal, ss_bias_matrix_kcal, boost_bias_matrix_kj
            )
            reduced_bias_matrix = float(beta) * bias_matrix_kj
            observable_cache["step"] = int(step)
            observable_cache["primary_values"] = primary_values
            observable_cache["cvs_nm"] = (primary_values / 10.0) if is_distance_primary else primary_values
            observable_cache["bias_matrix_kj"] = bias_matrix_kj
            observable_cache["v_pep_kj"] = v_pep_kj
            observable_cache["v_dih_kj"] = v_dih_kj
            observable_cache["boost_bias_matrix_kj"] = boost_bias_matrix_kj
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
                sampled_umbrella_bias_kj = float(4.184 * (all_distance_bias_kcal[w] + all_ss_bias_kcal[w]))
                sampled_boost_bias_kj = float(boost_bias_matrix_kj[w, r])
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
                    "v_pep_kj_mol": float(v_pep_kj[r]) if pep_env is not None else float("nan"),
                    "v_dih_kj_mol": float(v_dih_kj[r]) if pep_env is not None else float("nan"),
                    "gamd_lambda": float(state_lambdas[w]),
                    "umbrella_bias_kcal_mol": sampled_bias_kcal,
                    "umbrella_bias_kj_mol": sampled_bias_kj,
                    "sampled_umbrella_bias_kj": sampled_umbrella_bias_kj,
                    "sampled_boost_bias_kj": sampled_boost_bias_kj,
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
                if is_prod and _gamd_recal_active:
                    try:
                        for _gname, _gid in _gamd_recal_targets:
                            _gpe = boost_target_energy_kj(sim.context, _gid, unit, total_groups=total_energy_groups_for_args(args))
                            _gamd_recal_accumulators[_gname].setdefault(w, _WelfordAccumulator()).update(_gpe)
                    except Exception as _gamd_recal_sample_exc:
                        # Best-effort measurement only: a failure here must never take
                        # down a real production epoch. Disable for the rest of this
                        # run; whatever was accumulated before the failure is still
                        # written out and pooled (partial data degrades gracefully at
                        # the recalibration step, same as "no data").
                        print(
                            "WARNING: GaMD production-envelope sampling failed "
                            f"({_gamd_recal_sample_exc}); disabling for the rest of this run"
                        )
                        _gamd_recal_active = False
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
                        v_pep=float(v_pep_kj[r]) if pep_env is not None else float("nan"),
                        v_dih=float(v_dih_kj[r]) if pep_env is not None else float("nan"),
                        gamd_lambda=float(state_lambdas[w]),
                    )
                rows.append(row)
            if is_prod and bool(getattr(args, "flush_every_log", False)):
                parquet_sample_writer.flush()

            # ── tICA dihedral observation recording (opt-in via tica_obs_interval) ──
            if is_prod and _dihedral_buffers is not None:
                _tica_obs_counter[0] += 1
                if _tica_obs_counter[0] % _tica_obs_interval == 0:
                    for r, sim in enumerate(sims):
                        w = int(assignments[r])
                        _pos_state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
                        _pos_nm = _pos_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                        _dihedral_buffers[r].record(
                            _pos_nm, int(step), w,
                            cv_primary=float(primary_values[r]),
                            cv_secondary=float(ss_values[r]),
                        )
            # ──────────────────────────────────────────────────────────────────────

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
                    st = sim.context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True,
                                              groups=physical_energy_groups_for_args(args))
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
                _idx, _driver, _n = item
                _driver.advance(int(_n))
                return int(_idx)

            completed = 0
            try:
                if safe_chunk <= 0 or safe_chunk >= nsteps:
                    list(_sim_pool.map(_step_item, [(i, d, nsteps) for i, d in enumerate(drivers)]))
                else:
                    remaining = nsteps
                    while remaining > 0:
                        sub = min(int(safe_chunk), int(remaining))
                        list(_sim_pool.map(_step_item, [(i, d, sub) for i, d in enumerate(drivers)]))
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
            # Decision + bookkeeping live in the module-level kernel so a test can
            # drive them; the side effects below stay here. `force_accept` must not
            # consume a random number -- production short-circuited `rng.random()`
            # away, and drawing one anyway would shift every later RNG stream.
            # Check attemptability BEFORE drawing: the original closure returned
            # early on these cases without consuming a random number, and an
            # extra draw shifts every downstream stream.
            if swap_candidate_replicas(replica_of_window, wi, wj) is None:
                return attempt
            outcome = apply_window_swap(
                bias_matrix_kj, beta, assignments, replica_of_window, wi, wj,
                None if force_accept else rng.random(),
                p_override=p_override, force_accept=force_accept,
            )
            if outcome is None:
                return attempt
            i, j = outcome.replica_i, outcome.replica_j
            _record_exchange_stats(wi, wj, outcome.accepted)
            if outcome.accepted:
                # NPT correction: the umbrella-parameter writes run on each
                # replica's own worker (same affinity rule as stepping), and an
                # accepted swap re-labels replicas so the cached CV/bias
                # observables are stale until the next sample().
                def _apply_swap_to_replica(replica_index: int):
                    set_window(sims[replica_index].context, centers_nm, ks_kj_nm2, assignments[replica_index], secondary_cv_centers, secondary_cv_ks_kj)
                    set_replica_lambda_for_window(sims[replica_index].integrator, assignments[replica_index], state_lambdas, k0max_by_channel)

                _sim_pool.submit(i, _apply_swap_to_replica, i).result()
                _sim_pool.submit(j, _apply_swap_to_replica, j).result()
                observable_cache.clear()
            parquet_exchange_writer.write_exchange(
                step=int(absolute_step),
                replica_i=int(i),
                replica_j=int(j),
                window_i=int(wi),
                window_j=int(wj),
                delta_e=float(outcome.delta_kj),
                accepted=bool(outcome.accepted),
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
            # swap/Gibbs candidates.  Potential energies are not needed here -- except
            # when the λ-ladder is active (pep_env is not None), which adds three
            # getState(getEnergy=True, ...) reads per replica per exchange attempt
            # (peptide_essential_energy_kj's two group reads plus the dihedral group)
            # to price the Pep-GaMD boost under every candidate state's λ.
            primary_values = np.empty(nrep, dtype=np.float64)
            ss_values = np.full(nrep, np.nan, dtype=np.float64)
            v_pep_kj = np.empty(nrep, dtype=np.float64)
            v_dih_kj = np.empty(nrep, dtype=np.float64)
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
                    v_pep, v_dih = _fetch_v_pep_v_dih(ctx, pep_env, unit)
                else:
                    state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
                    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                    cv = primary_cv_value_from_positions_nm(pos, primary_cv_def, args)
                    ss = secondary_structure_score_from_positions_nm(pos, secondary_cv_metadata) if _ss_enabled else float("nan")
                    v_pep, v_dih = _fetch_v_pep_v_dih(sim.context, pep_env, unit)
                return r, cv, ss, v_pep, v_dih

            for r, cv, ss, v_pep, v_dih in _sim_pool.map(_fetch_exchange_state, enumerate(sims)):
                primary_values[r] = cv
                ss_values[r] = ss
                v_pep_kj[r] = v_pep
                v_dih_kj[r] = v_dih
            dprimary = primary_values[np.newaxis, :] - centers_a_arr[:, np.newaxis]
            distance_bias_kcal = 0.5 * k_arr[:, np.newaxis] * dprimary * dprimary
            if secondary_cv_centers is not None and secondary_cv_ks_kj is not None and ss_ks_kj_arr_global is not None:
                dss = ss_values[np.newaxis, :] - ss_centers_arr_global[:, np.newaxis]
                ss_bias_kcal = (0.5 * ss_ks_kj_arr_global[:, np.newaxis] * dss * dss) / 4.184
            else:
                ss_bias_kcal = np.zeros_like(distance_bias_kcal)
            boost_bias_matrix_kj = (pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, state_lambdas, pep_env)
                                    if pep_env is not None else np.zeros((nrep, nrep)))
            _, bias_matrix_kj = assemble_bias_matrices(distance_bias_kcal, ss_bias_kcal, boost_bias_matrix_kj)
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
                    prop = gibbs_propose_one_replica(
                        bias_matrix_kj, float(beta), assignments, replica_of_window,
                        rep, lambda k, p: int(rng.choice(k, p=p)),
                    )
                    if prop.no_candidates:
                        exchange_stats["gibbs_all_nan_skips"] = exchange_stats.get("gibbs_all_nan_skips", 0) + 1
                        continue
                    exchange_stats["gibbs_choices"] = int(exchange_stats.get("gibbs_choices", 0) or 0) + 1
                    if prop.stayed:
                        exchange_stats["gibbs_stays"] = int(exchange_stats.get("gibbs_stays", 0) or 0) + 1
                        continue
                    # Counted here, before the MH test, so this is a count of
                    # PROPOSED moves. `gibbs_move_fraction` derived from it is a
                    # proposal rate, not an acceptance rate.
                    exchange_stats["gibbs_moves"] = int(exchange_stats.get("gibbs_moves", 0) or 0) + 1
                    exchange_stats["gibbs_mh_corrected"] = True
                    attempt = _attempt_window_swap(
                        prop.current_window, prop.proposed_window, primary_values,
                        bias_matrix_kj, absolute_step, attempt, p_override=prop.pacc,
                    )
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
                adaptive_phase_info = merge_adaptive_phase_info({
                    "is_adaptive_epoch": True,
                    "segment_name": str(_api.get("segment_name", "")),
                    "is_topup": bool(_api.get("is_topup", False)),
                    "topup_index": int(_api.get("topup_index", 0)),
                    "prev_epochs": list(_api.get("prev_epochs", [])),
                }, _api)
                dashboard_info["adaptive_phase"] = adaptive_phase_info
            else:
                adaptive_phase_info.update(merge_adaptive_phase_info(adaptive_phase_info, _api))

        if bool(getattr(args, "resume", False)):
            manifest = load_production_checkpoint(
                out_dir, sims, centers_nm, ks_kj_nm2, rng, secondary_cv_centers, secondary_cv_ks_kj,
                openmm_version=str(getattr(getattr(openmm, "version", None), "version", None)),
                platform_name=str(platform.getName()),
                strict_gamd_restore=bool(getattr(args, "strict_gamd_restore", False)),
                state_lambdas=state_lambdas, k0max_by_channel=k0max_by_channel,
                drivers=drivers, pool=_sim_pool, npt_runtime=npt_runtime, args=args,
            )
            if manifest is not None:
                assignments[:] = [int(x) for x in manifest.get("assignments", assignments)]
                dashboard_info["gamd_integrator_restore_ok"] = bool(manifest.get("resume_gamd_integrator_restore_report", {}).get("ok", False))
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
                    for i, driver in enumerate(drivers):
                        reporter = make_trajectory_reporter(app, traj_dir / f"replica_{i:03d}_resume_from_{prod_done:09d}", effective_traj_interval, args, atom_subset=traj_atom_subset)
                        if reporter is not None:
                            driver.register_reporter(reporter)
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
                    for i, driver in enumerate(drivers):
                        reporter = make_trajectory_reporter(app, traj_dir / f"replica_{i:03d}_resume_fresh_{int(time.time())}", effective_traj_interval, args, atom_subset=traj_atom_subset)
                        if reporter is not None:
                            driver.register_reporter(reporter)
        elif _parent_was_running and _parent_seg_id is not None:
            # Fresh start (no --resume) while a previous segment is still marked
            # running: the prior run was abandoned without a checkpoint.
            _seg_registry.seal_segment(_parent_seg_id, absolute_end_step=-1, status="abandoned")

        if prod_done <= 0:
            run_production_probe(args, out_dir, sims, assignments, centers_nm, ks_kj_nm2, primary_cv_def, cv_atom1, cv_atom2, unit, shared_gamd_globals_all,
                                 drivers=drivers, pool=_sim_pool, npt_runtime=npt_runtime)

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
                    openmm_version=str(getattr(getattr(openmm, "version", None), "version", None)),
                    platform_name=str(platform.getName()),
                    drivers=drivers, pool=_sim_pool, npt_runtime=npt_runtime,
                )
                _scratch_main = getattr(args, "_main_dir", None)
                if _scratch_main:
                    sync_scratch_to_main(out_dir, Path(_scratch_main))
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
                                    # NPT correction: the read and the two writes
                                    # each run on that replica's own worker (the
                                    # same affinity rule as stepping), and the
                                    # rescue invalidates the CV cache like any
                                    # other out-of-band coordinate change.
                                    _src_state = _sim_pool.submit(_src, lambda: sims[_src].context.getState(
                                        getPositions=True, enforcePeriodicBox=True
                                    )).result()
                                    _src_pos = _src_state.getPositions()

                                    def _rescue_positions(_pos=_src_pos, _r=_r):
                                        sims[_r].context.setPositions(_pos)
                                        sims[_r].context.setVelocitiesToTemperature(
                                            float(args.temperature_k) * unit.kelvin,
                                            int(args.seed) + _r + int(absolute_step % 99991),
                                        )

                                    _sim_pool.submit(_r, _rescue_positions).result()
                                    observable_cache.clear()
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
                    openmm_version=str(getattr(getattr(openmm, "version", None), "version", None)),
                    platform_name=str(platform.getName()),
                    drivers=drivers, pool=_sim_pool, npt_runtime=npt_runtime,
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
                        fallback_secondary_history_by_window=distance_logger.secondary_history_by_window,
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

        # Mark the production run as cleanly completed only when the loop ran
        # to completion (prod_done >= prod_total).  A graceful-shutdown break
        # leaves prod_done < prod_total and correctly keeps the flag False.
        _prod_completed_cleanly = (prod_done >= prod_total > 0)

        # Write out whatever was accumulated even if sampling was disabled partway
        # through the run (see the try/except around the per-frame update above) --
        # partial data still degrades gracefully at the recalibration step.
        if any(_gamd_recal_accumulators.get(name) for name, _gid in _gamd_recal_targets):
            _stats_payload = {}
            for _gname, _accs in _gamd_recal_accumulators.items():
                _window_stats = []
                for _window, _acc in _accs.items():
                    if _acc.n < 2:
                        continue
                    _s = _acc.to_stats(_gname, _window)
                    _window_stats.append({
                        "group": _s.group, "window": _s.window, "vmax": _s.vmax,
                        "vmin": _s.vmin, "mean": _s.mean, "var": _s.var, "n": _s.n,
                    })
                if _window_stats:
                    _stats_payload[_gname] = _window_stats
            if _stats_payload:
                _stats_path = out_dir / "gamd_production_envelope_stats.json"
                write_json(_stats_path, {
                    "schema_version": "gamd_production_envelope_stats_v1",
                    "gamd_boost_type": str(args.gamd_boost_type),
                    "epoch_index": int(_phase_epoch_index),
                    "per_group_window_stats": _stats_payload,
                })
                print(f"    GaMD production-envelope recalibration: wrote boost-group energy stats -> {_stats_path}")

    finally:
        try:
            _sim_pool.shutdown(wait=True)
        except Exception:
            pass

        # Flush and close parquet writers.  Track whether ALL of these
        # succeed so the segment registry can distinguish a fully written
        # segment from a truncated or crashed one.
        _writers_ok = True
        try:
            if _prod_completed_cleanly:
                parquet_sample_writer.flush()
                parquet_exchange_writer.flush()
        except Exception as _flush_exc:
            _writers_ok = False
            print(f"WARNING: parquet writer flush failed: {_flush_exc}", flush=True)
        try:
            parquet_sample_writer.close()
        except Exception as _close_exc:
            _writers_ok = False
            print(f"WARNING: parquet_sample_writer.close() failed: {_close_exc}", flush=True)
        try:
            parquet_exchange_writer.close()
        except Exception as _close_exc:
            _writers_ok = False
            print(f"WARNING: parquet_exchange_writer.close() failed: {_close_exc}", flush=True)

        # Gate segment-complete marking on BOTH a clean loop exit AND
        # successful writer flush/close.  Any failure leaves the segment
        # as "interrupted" so the next resume knows the last valid checkpoint.
        try:
            finalize_segment(
                _seg_registry,
                _seg_id,
                completed_cleanly=_prod_completed_cleanly,
                writers_ok=_writers_ok,
                end_step=int(calib_steps + prod_done),
            )
        except Exception as _finalize_exc:
            print(f"WARNING: segment registry finalization failed: {_finalize_exc}", flush=True)

        distance_logger.close()

        # Flush tICA dihedral observation buffers if active.
        if _dihedral_buffers is not None:
            for _buf in _dihedral_buffers:
                try:
                    _buf.save()
                except Exception as _buf_exc:
                    print(f"WARNING: dihedral obs buffer flush failed for replica {_buf._replica}: {_buf_exc}", flush=True)

    print(f"Done. Outputs in {out_dir}")
