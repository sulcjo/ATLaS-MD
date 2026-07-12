"""
Seeding and US starting-structure helpers.

This module contains the GENPEPT-seed loader, Kabsch grafting utilities, and
umbrella starting-structure generation code extracted from ``gareus_peptide.py``.
It is intentionally OpenMM-light at import time: OpenMM modules are passed in by
callers rather than imported at module scope.
"""

from __future__ import annotations

import concurrent.futures
import copy
import csv
import math
import queue
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from .cv import (
    contact_normalization_denominator,
    contact_us_pull_k_user,
    find_atom_in_residue,
    peptide_residues,
    primary_cv_format_value,
    primary_cv_is_contacts,
    primary_cv_is_distance,
    primary_cv_label,
    primary_cv_mode,
    primary_cv_units,
    primary_cv_value_from_positions_nm,
    primary_k_to_openmm_value,
    secondary_structure_score_from_positions_nm,
)
from .io import _json_ready, write_json
from .progress import GuiProgressSink
from .state import _scalar_to_float, cv_distance_from_positions_nm, cv_distance_nm, cv_distance_and_potential_from_state
from .system_setup import (
    _first_device_index,
    make_langevin_integrator,
    platform_and_properties,
    run_steps_safely,
    write_state_pdb,
)
from .units import kj_nm2_to_kcal_a2


def _safe_max_finite(rows, key) -> float:
    """Max of ``rows[*][key]`` over finite numeric values; NaN if none.

    Tolerates blanks (``""``), ``None``, NaN, and OpenMM quantities. Quality
    rows store ``""`` for inactive coordinates (e.g. the secondary-CV bias when
    ``cv2`` is ``none``), which a raw ``float()`` would choke on.
    """
    vals = []
    for r in rows:
        v = _scalar_to_float(r.get(key, None))
        if v is not None and math.isfinite(v):
            vals.append(v)
    return float(max(vals)) if vals else float("nan")


def deserialize_system(openmm, system):
    """Deep-copy an OpenMM System via XML serialization."""
    return openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))


def kabsch_align_positions(
    mobile_ca: np.ndarray,
    target_ca: np.ndarray,
    mobile_all: np.ndarray,
) -> np.ndarray:
    """Align mobile_all onto target by Kabsch-rotating mobile_ca onto target_ca.

    Returns the transformed mobile_all positions (same shape as input).
    mobile_ca and target_ca must have the same number of rows (>= 2).
    """
    mobile_center = np.mean(mobile_ca, axis=0)
    target_center = np.mean(target_ca, axis=0)
    mob_c = mobile_ca - mobile_center
    tgt_c = target_ca - target_center
    H = mob_c.T @ tgt_c
    U, _S, Vt = np.linalg.svd(H)
    # Correct for reflection: ensure proper rotation (det = +1, not -1)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return (mobile_all - mobile_center) @ R.T + target_center


# Minimum number of real-topology atoms that must map onto seed atoms for a
# backbone-subset graft.  Two residues' worth of backbone (N, CA, C, O) is enough
# for a stable Kabsch superposition and a meaningful backbone conformation.
_MIN_GRAFT_BACKBONE_ATOMS = 6


def _assemble_grafted_positions_nm(
    full_pos_nm: np.ndarray,
    seed_pos_nm: np.ndarray,
    topo_indices: np.ndarray,
    seed_indices: np.ndarray,
) -> np.ndarray:
    """Backbone-subset graft: Kabsch-align the seed onto the mapped topology atoms,
    then overwrite only those mapped atoms with the aligned seed coordinates.

    ``topo_indices[k]`` and ``seed_indices[k]`` are a corresponding pair (same real
    atom identified in the production topology and in the seed conformer).  Unmapped
    atoms (e.g. sidechains absent from a poly-glycine backbone seed) keep their
    existing positions and are relaxed by the caller's energy minimization.

    Pure NumPy so the mapping/scatter logic is unit-testable without an OpenMM
    context.  Returns a new array; ``full_pos_nm`` is not mutated.
    """
    topo_indices = np.asarray(topo_indices, dtype=int)
    seed_indices = np.asarray(seed_indices, dtype=int)
    target_sub = full_pos_nm[topo_indices, :]
    mobile_sub = seed_pos_nm[seed_indices, :]
    aligned_seed_all = kabsch_align_positions(mobile_sub, target_sub, seed_pos_nm)
    new_full_pos = np.array(full_pos_nm, dtype=float, copy=True)
    new_full_pos[topo_indices, :] = aligned_seed_all[seed_indices, :]
    return new_full_pos


def _primary_seed_score(
    primary_value: float,
    target_primary: float,
    primary_seed_scale: float,
) -> tuple[float, float]:
    """Return ``(abs_delta, score)`` for a seed's primary CV vs a window target.

    A non-finite ``primary_value`` means the primary CV is unavailable for this
    seed (e.g. a sidechain-contact CV cannot be evaluated on a backbone-only
    poly-glycine seed).  Return score ``0`` so seed selection falls to the
    secondary CV rather than an infinite penalty that swamps every seed equally
    and defeats CV2 ranking.
    """
    if math.isfinite(primary_value):
        delta = abs(float(primary_value) - float(target_primary))
        return delta, delta / max(1.0e-12, float(primary_seed_scale))
    return float("nan"), 0.0


def harmonic_bias_energy_kj(distance_nm: float, center_nm: float, k_kj_nm2: float) -> float:
    dr = float(distance_nm) - float(center_nm)
    return 0.5 * float(k_kj_nm2) * dr * dr


def _peptide_atom_offset_and_count(topology) -> tuple[int, int]:
    """Return the first peptide atom index and the number of peptide atoms.

    GENPEPT survivor PDBs are peptide-only.  GAREUS solvated topologies normally
    keep peptide atoms first, but this helper makes the absolute→relative mapping
    explicit for seed scoring instead of relying on lucky atom-index folklore.
    """
    pep = peptide_residues(topology)
    if not pep:
        return 0, 0
    atom_indices = [int(atom.index) for res in pep for atom in res.atoms()]
    if not atom_indices:
        return 0, 0
    first = min(atom_indices)
    return first, len(atom_indices)


def _pdb_atom_name(line: str) -> str:
    return str(line[12:16]).strip()


def _canonical_atom_name(name: str) -> str:
    return str(name or "").strip().upper()


def _read_pdb_conformer_atoms(pdb_path: Path) -> tuple[np.ndarray, list[dict]]:
    positions_a = []
    atoms = []
    residue_ord = -1
    last_residue_key = None
    with Path(pdb_path).open() as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except Exception:
                continue
            residue_key = (line[21:22], line[22:26], line[26:27])
            if residue_key != last_residue_key:
                residue_ord += 1
                last_residue_key = residue_key
            atom_index = len(positions_a)
            positions_a.append([x, y, z])
            atoms.append({
                "index": int(atom_index),
                "name": _pdb_atom_name(line),
                "residue_ordinal": int(residue_ord),
            })
    return np.asarray(positions_a, dtype=float) / 10.0, atoms


def _topology_to_conformer_atom_index(topology, conformer_atoms: list[dict]) -> dict[int, int]:
    """Map production topology atom indices to peptide-only conformer PDB indices.

    GENPEPT survivor PDBs can differ from the production sequence or atom order
    (for example all-glycine backbone seeds for a sidechain-rich peptide), so
    offsetting by the first peptide atom is not enough.  Backbone atoms needed
    for torsions still map by residue ordinal and atom name.
    """
    if topology is None:
        return {}
    seed_by_res_atom: dict[tuple[int, str], int] = {}
    for atom in conformer_atoms or []:
        key = (int(atom.get("residue_ordinal", -1)), _canonical_atom_name(str(atom.get("name", ""))))
        seed_by_res_atom.setdefault(key, int(atom.get("index", -1)))
    mapped: dict[int, int] = {}
    topology_atom_indices: list[int] = []
    for res_ord, residue in enumerate(peptide_residues(topology)):
        for atom in residue.atoms():
            topology_atom_indices.append(int(atom.index))
            key = (int(res_ord), _canonical_atom_name(str(getattr(atom, "name", ""))))
            seed_idx = seed_by_res_atom.get(key)
            if seed_idx is not None and seed_idx >= 0:
                mapped[int(atom.index)] = int(seed_idx)
    if len(conformer_atoms or []) == len(topology_atom_indices):
        for seed_idx, topology_idx in enumerate(topology_atom_indices):
            mapped.setdefault(int(topology_idx), int(seed_idx))
    return mapped


def _map_topology_atom_index(
    atom_index: int,
    topology,
    topology_to_seed: Optional[dict[int, int]] = None,
) -> Optional[int]:
    if topology_to_seed is not None:
        mapped = topology_to_seed.get(int(atom_index))
        return int(mapped) if mapped is not None else None
    if topology is None:
        return int(atom_index)
    first, n_pep = _peptide_atom_offset_and_count(topology)
    rel = int(atom_index) - int(first)
    return int(rel) if 0 <= rel < n_pep else None


def map_topology_torsions_to_conformer(
    torsions,
    topology,
    topology_to_seed: Optional[dict[int, int]] = None,
) -> list[tuple[int, int, int, int]]:
    mapped = []
    for tor in torsions or []:
        rel = [_map_topology_atom_index(int(x), topology, topology_to_seed) for x in tor]
        if len(rel) == 4 and all(x is not None for x in rel):
            mapped.append(tuple(int(x) for x in rel))
    return mapped


def _relative_distance_cv_def_for_conformer(
    cv_atom1,
    cv_atom2,
    topology,
    topology_to_seed: Optional[dict[int, int]] = None,
) -> Optional[dict]:
    try:
        a1 = _map_topology_atom_index(int(cv_atom1), topology, topology_to_seed)
        a2 = _map_topology_atom_index(int(cv_atom2), topology, topology_to_seed)
    except Exception:
        return None
    if a1 is None or a2 is None:
        return None
    return {
        "mode": "distance",
        "label": "terminal distance",
        "units": "A",
        "cv_atom1": int(a1),
        "cv_atom2": int(a2),
        "contact_pairs": [],
    }


def _relative_primary_cv_def_for_conformer(
    primary_cv_def: Optional[dict],
    topology,
    topology_to_seed: Optional[dict[int, int]] = None,
) -> Optional[dict]:
    """Map an absolute-topology primary-CV definition to peptide-only seed indices."""
    if not isinstance(primary_cv_def, dict):
        return None
    first, n_pep = _peptide_atom_offset_and_count(topology)
    if n_pep <= 0:
        return None
    mode = primary_cv_mode(primary_cv_def)
    rel = dict(primary_cv_def)
    # Private NumPy arrays must be rebuilt for the peptide-only atom numbering.
    for key in list(rel.keys()):
        if str(key).startswith("_np_"):
            rel.pop(key, None)
    if mode == "distance":
        try:
            a1 = _map_topology_atom_index(int(primary_cv_def["cv_atom1"]), topology, topology_to_seed)
            a2 = _map_topology_atom_index(int(primary_cv_def["cv_atom2"]), topology, topology_to_seed)
        except Exception:
            return None
        if a1 is None or a2 is None:
            return None
        rel["cv_atom1"] = int(a1)
        rel["cv_atom2"] = int(a2)
        rel["contact_pairs"] = []
        return rel
    if mode == "nonlocal-contacts":
        pairs = []
        for pair in (primary_cv_def.get("contact_pairs", []) or []):
            try:
                i = _map_topology_atom_index(int(pair[0]), topology, topology_to_seed)
                j = _map_topology_atom_index(int(pair[1]), topology, topology_to_seed)
                w = float(pair[2]) if len(pair) > 2 else 1.0
            except Exception:
                continue
            if i is not None and j is not None and i != j:
                pairs.append((int(i), int(j), float(w)))
        if not pairs:
            return None
        idx = np.array([[int(p[0]), int(p[1])] for p in pairs], dtype=np.int32)
        weights = np.array([float(p[2]) for p in pairs], dtype=np.float64)
        rel["contact_pairs"] = pairs
        rel["n_contact_pairs"] = int(len(pairs))
        rel["n_contact_terms"] = int(len(pairs))
        rel["contact_weight_sum"] = float(np.sum(weights))
        rel["_np_idx_i"] = idx[:, 0]
        rel["_np_idx_j"] = idx[:, 1]
        rel["_np_weights"] = weights
        rel["_np_r0_nm"] = float(primary_cv_def.get("_np_r0_nm", primary_cv_def.get("contact_switch_r0_nm", 0.45)))
        rel["_np_beta_nm_inv"] = float(primary_cv_def.get("_np_beta_nm_inv", 60.0))
        rel["_np_norm_denom"] = contact_normalization_denominator(pairs, type("_SeedArgsProxy", (), {"contact_normalize": primary_cv_def.get("contact_normalize", True)})())
        return rel
    return None


def _relative_secondary_cv_metadata_for_conformer(
    secondary_cv_metadata: Optional[dict],
    topology,
    topology_to_seed: Optional[dict[int, int]] = None,
) -> Optional[dict]:
    """Map secondary-CV torsion indices to peptide-only GENPEPT survivor indices."""
    if not isinstance(secondary_cv_metadata, dict) or not secondary_cv_metadata.get("enabled"):
        return None
    first, n_pep = _peptide_atom_offset_and_count(topology)
    if n_pep <= 0:
        return None
    meta = dict(secondary_cv_metadata)
    for key in list(meta.keys()):
        if str(key).startswith("_np_"):
            meta.pop(key, None)
    def _map_torsions(torsions):
        return map_topology_torsions_to_conformer(torsions, topology, topology_to_seed)
    meta["phi_torsions"] = _map_torsions(meta.get("phi_torsions", []))
    meta["psi_torsions"] = _map_torsions(meta.get("psi_torsions", []))
    if not meta["phi_torsions"] and not meta["psi_torsions"]:
        return None
    return meta


def detect_seed_scoring_degradations(
    primary_is_contacts: bool,
    rel_primary: Optional[dict],
    secondary_available: bool,
    seed_secondary_weight: float,
    seed_selection_mode: str,
    rel_secondary: Optional[dict],
) -> list[dict]:
    """Detect silent collapse of GENPEPT seed scoring to a lower-dimensional proxy.

    Pure helper (no I/O, no OpenMM): takes already-resolved active-CV flags and
    definitions and flags two scientific-audit failure modes that must never
    pass silently when 2D sampling is active:

      (a) ``primary_cv_collapsed_to_distance`` — the run's primary CV is
          ``contacts``, but the loader could not map any production contact
          pair into the peptide-only seed atom range, so it silently
          substituted terminal Ca-Ca distance for every seed.
      (b) ``secondary_cv_dropped`` — the secondary CV is intended to be part
          of active-CV seed selection (available, weighted > 0, and
          ``seed_selection_mode == "active-cv"``), but its relative CV
          metadata resolved to ``None`` (missing/disabled/unmapped), so seed
          scoring silently became CV1-only.

    A run that legitimately chose distance as its primary CV, or that
    deliberately excludes CV2 from seed scoring (``seed_selection_mode ==
    "distance"`` or ``seed_secondary_weight == 0``), is not a degradation.
    """
    degradations: list[dict] = []
    if primary_is_contacts and isinstance(rel_primary, dict) and str(rel_primary.get("mode")) == "distance":
        degradations.append({
            "kind": "primary_cv_collapsed_to_distance",
            "message": (
                "Seed scoring collapsed: primary CV is 'contacts' but no production "
                "contact pair mapped into the peptide-only seed atom range, so every "
                "seed was scored by terminal Ca-Ca distance instead "
                "(CV1 contacts -> terminal-distance proxy)."
            ),
        })
    secondary_intended_active = (
        bool(secondary_available)
        and float(seed_secondary_weight or 0.0) > 0.0
        and str(seed_selection_mode) == "active-cv"
    )
    if secondary_intended_active and rel_secondary is None:
        degradations.append({
            "kind": "secondary_cv_dropped",
            "message": (
                "Seed scoring collapsed: secondary CV was intended active for seed "
                "selection but its relative CV metadata resolved to None (metadata "
                "missing/disabled or unmapped to the seed's peptide-only atoms), so "
                "seed scoring silently became CV1-only "
                "(CV2 rama-map dropped: metadata disabled while centers present, or unmapped)."
            ),
        })
    return degradations


def load_genpept_conformer_library(
    seed_conformers_dir: Path,
    cv_atom1: Optional[int] = None,
    cv_atom2: Optional[int] = None,
    *,
    primary_cv_def: Optional[dict] = None,
    args=None,
    topology=None,
    secondary_cv_metadata: Optional[dict] = None,
    resolved_cv_defs_out: Optional[dict] = None,
) -> list[dict]:
    """Load GENPEPT final survivor PDBs and score them in active GAREUS CV space.

    Historical behavior computed only terminal distance and therefore helped only
    distance-CV workflows.  The current loader computes the selected primary CV
    (distance or nonlocal contacts) and, when a secondary CV is active, the same
    smooth backbone score used by GAREUS.  Returned dicts keep legacy ``cv_A``
    fields for old reports but also include generic ``primary_cv_value`` and
    ``secondary_cv_value`` fields.
    """
    csv_path = Path(seed_conformers_dir) / "final_survivor_seeds.csv"
    if not csv_path.exists():
        print(f"WARNING: --seed-conformers-dir: {csv_path} not found; falling back to NPT-pull for all windows.")
        if resolved_cv_defs_out is not None:
            resolved_cv_defs_out["rel_primary"] = None
            resolved_cv_defs_out["rel_secondary"] = None
        return []
    try:
        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as exc:
        print(f"WARNING: --seed-conformers-dir: could not read {csv_path}: {exc}")
        if resolved_cv_defs_out is not None:
            resolved_cv_defs_out["rel_primary"] = None
            resolved_cv_defs_out["rel_secondary"] = None
        return []

    rel_primary = _relative_primary_cv_def_for_conformer(primary_cv_def, topology) if topology is not None else None
    if rel_primary is None and cv_atom1 is not None and cv_atom2 is not None:
        rel_primary = _relative_distance_cv_def_for_conformer(cv_atom1, cv_atom2, topology)
    rel_secondary = _relative_secondary_cv_metadata_for_conformer(secondary_cv_metadata, topology) if topology is not None else None
    if resolved_cv_defs_out is not None:
        resolved_cv_defs_out["rel_primary"] = rel_primary
        resolved_cv_defs_out["rel_secondary"] = rel_secondary
    mode = primary_cv_mode(rel_primary or primary_cv_def or "distance")
    units = primary_cv_units(rel_primary or primary_cv_def or "distance")

    library = []
    skipped = 0
    for row in rows:
        pdb_path = Path(row.get("survivor_pdb_path", ""))
        if not pdb_path.exists():
            if pdb_path.is_absolute():
                # Paths are baked in at GENPEPT-generation time. If the run
                # tree was since moved/renamed (e.g. RUNS/runs3 -> RUNS/runs_rdy),
                # the baked path is stale but its tail still starts at the
                # genpept output directory itself, so re-root the tail under
                # the current seed_conformers_dir.
                seed_dir_name = Path(seed_conformers_dir).name
                parts = pdb_path.parts
                if seed_dir_name in parts:
                    idx = len(parts) - 1 - parts[::-1].index(seed_dir_name)
                    candidate = Path(seed_conformers_dir).joinpath(*parts[idx + 1 :])
                    if candidate.exists():
                        pdb_path = candidate
            else:
                candidate = Path(seed_conformers_dir) / pdb_path
                if candidate.exists():
                    pdb_path = candidate
        if not pdb_path.exists():
            skipped += 1
            continue
        try:
            pos_nm, conformer_atoms = _read_pdb_conformer_atoms(pdb_path)
            if pos_nm.size == 0:
                skipped += 1
                continue
            atom_map = _topology_to_conformer_atom_index(topology, conformer_atoms) if topology is not None else None
            row_rel_primary = (
                _relative_primary_cv_def_for_conformer(primary_cv_def, topology, atom_map)
                if topology is not None
                else rel_primary
            )
            if row_rel_primary is None and cv_atom1 is not None and cv_atom2 is not None:
                row_rel_primary = _relative_distance_cv_def_for_conformer(cv_atom1, cv_atom2, topology, atom_map)
            row_rel_secondary = (
                _relative_secondary_cv_metadata_for_conformer(secondary_cv_metadata, topology, atom_map)
                if topology is not None
                else rel_secondary
            )
            if resolved_cv_defs_out is not None:
                if resolved_cv_defs_out.get("rel_primary") is None and row_rel_primary is not None:
                    resolved_cv_defs_out["rel_primary"] = row_rel_primary
                if resolved_cv_defs_out.get("rel_secondary") is None and row_rel_secondary is not None:
                    resolved_cv_defs_out["rel_secondary"] = row_rel_secondary

            if row_rel_primary is not None:
                primary_value = primary_cv_value_from_positions_nm(pos_nm, row_rel_primary, args)
                # A contacts primary that resolved to a terminal-distance def on this
                # seed (e.g. a backbone-only poly-glycine seed with no mappable
                # sidechain contacts) is a units mismatch: the distance (Å) is NOT a
                # dimensionless contact fraction.  Scoring on it corrupts active-CV
                # seed selection (observed as conformer_primary_cv ~ 3.8 vs [0,1]
                # centers).  Mark the primary CV unavailable so selection falls to the
                # backbone-torsion secondary CV; the collapse is still flagged by
                # detect_seed_scoring_degradations above.
                if primary_cv_is_contacts(args) and primary_cv_mode(row_rel_primary) == "distance":
                    primary_value = float("nan")
            else:
                primary_value = float("nan")
            secondary_value = float("nan")
            if row_rel_secondary is not None:
                secondary_value = secondary_structure_score_from_positions_nm(pos_nm, row_rel_secondary)
            entry = {
                "primary_cv": mode,
                "primary_cv_value": float(primary_value),
                "primary_cv_units": units,
                "secondary_cv_value": float(secondary_value),
                "secondary_cv_mode": str((secondary_cv_metadata or {}).get("mode", "none")) if row_rel_secondary is not None else "none",
                "pdb_path": pdb_path,
                "positions_nm": pos_nm,
                "topology_to_conformer_atom_index": dict(atom_map or {}),
                "source_row": dict(row),
            }
            # Legacy reports/tools look for cv_A.  In contact mode this is a
            # compatibility alias for the primary CV, not an Angstrom distance.
            entry["cv_A"] = float(primary_value)
            library.append(entry)
        except Exception as exc:
            print(f"WARNING: --seed-conformers-dir: could not load {pdb_path}: {exc}")
            skipped += 1
    if skipped:
        print(f"WARNING: --seed-conformers-dir: {skipped}/{len(rows)} survivors skipped.")
    def _sort_key(entry):
        value = float(entry.get("primary_cv_value", float("inf")))
        if not math.isfinite(value):
            value = float("inf")
        return value, str(entry.get("pdb_path", ""))
    library.sort(key=_sort_key)
    sec_msg = ""
    finite_sec = sum(1 for x in library if math.isfinite(float(x.get("secondary_cv_value", float("nan")))))
    if finite_sec:
        sec_msg = f", {finite_sec} with secondary-CV scores"
    print(
        f"    GENPEPT conformer library: {len(library)} survivors loaded from {seed_conformers_dir} "
        f"using active primary CV '{mode}' ({units}){sec_msg}"
    )
    return library


def _finite_spacing_scale(values, fallback: float = 1.0) -> float:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size <= 1:
        return max(1.0e-12, float(fallback))
    uniq = np.unique(np.round(arr, decimals=10))
    if uniq.size <= 1:
        return max(1.0e-12, float(fallback))
    diffs = np.diff(np.sort(uniq))
    diffs = diffs[np.isfinite(diffs) & (diffs > 1.0e-12)]
    if diffs.size:
        return max(1.0e-12, float(np.median(diffs)))
    span = float(np.nanmax(uniq) - np.nanmin(uniq))
    return max(1.0e-12, span / max(1, uniq.size - 1), float(fallback))


def _format_seed_component_dict(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (float, np.floating)):
            out[k] = float(v) if math.isfinite(float(v)) else None
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        else:
            out[k] = v
    return out

def graft_conformer_into_context(
    sim,
    topology,
    conformer: dict,
    cv_atom1: int,
    cv_atom2: int,
    temperature_k: float,
    unit,
    minimize_iters: int = 100,
    seed: int = 0,
) -> dict:
    """Graft a gas-phase GENPEPT conformer into the solvated context, minimize clashes, re-thermalize.

    GENPEPT survivor seeds are frequently poly-glycine backbone-only conformers with
    fewer atoms than the real sidechain-bearing peptide.  We map seed atoms onto the
    real topology by (residue, atom name) and overwrite only the mapped (backbone)
    atoms, retaining the real sidechains for minimization to relax.  A seed whose atom
    count matches the peptide and carries no name map keeps the legacy wholesale path.
    """
    # Early guards run unguarded — these are programming errors if they fail
    pep_residues = peptide_residues(topology)
    n_pep = sum(sum(1 for _ in res.atoms()) for res in pep_residues)
    first_pep_atom_index = next(iter(pep_residues[0].atoms())).index if pep_residues else 0
    seed_pos_nm = np.asarray(conformer["positions_nm"], dtype=float)
    n_seed = int(seed_pos_nm.shape[0])

    # Real-topology peptide atom index -> seed conformer atom index, restricted to
    # the peptide atom range and to seed atoms that actually exist.
    topo_to_seed = {
        int(t): int(s)
        for t, s in (conformer.get("topology_to_conformer_atom_index") or {}).items()
        if first_pep_atom_index <= int(t) < first_pep_atom_index + n_pep and 0 <= int(s) < n_seed
    }

    ca_abs = []
    ca_rel = []
    for res in pep_residues:
        try:
            idx = find_atom_in_residue(res, "CA")
            ca_abs.append(idx)
            ca_rel.append(idx - first_pep_atom_index)
        except ValueError:
            pass
    if len(ca_abs) < 2:
        return {"fallback": True, "fallback_reason": "too_few_ca"}
    ca_abs_arr = np.array(ca_abs, dtype=int)
    ca_rel_arr = np.array(ca_rel, dtype=int)

    # Legacy path: same atom count and no name map -> wholesale overwrite by
    # peptide-block order.  Otherwise do a name-mapped backbone-subset graft.
    wholesale = (n_seed == n_pep) and not topo_to_seed
    if not wholesale and len(topo_to_seed) < _MIN_GRAFT_BACKBONE_ATOMS:
        return {"fallback": True, "fallback_reason": "too_few_mapped_atoms"}

    # OpenMM calls — wrapped so a platform failure falls back gracefully
    try:
        state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
        full_pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        graft_mode = "wholesale" if wholesale else "backbone-subset"
        if wholesale:
            target_ca = full_pos_nm[ca_abs_arr, :]
            mobile_ca = seed_pos_nm[ca_rel_arr, :]
            aligned_pep_pos_nm = kabsch_align_positions(mobile_ca, target_ca, seed_pos_nm)
            new_full_pos = full_pos_nm.copy()
            new_full_pos[first_pep_atom_index : first_pep_atom_index + n_pep, :] = aligned_pep_pos_nm
            ca_aligned_nm = aligned_pep_pos_nm[ca_rel_arr, :]
            n_grafted = int(n_pep)
        else:
            topo_idx = np.fromiter(topo_to_seed.keys(), dtype=int, count=len(topo_to_seed))
            seed_idx = np.fromiter((topo_to_seed[int(t)] for t in topo_idx), dtype=int, count=len(topo_to_seed))
            new_full_pos = _assemble_grafted_positions_nm(full_pos_nm, seed_pos_nm, topo_idx, seed_idx)
            n_grafted = int(topo_idx.size)
            # Cα RMSD only over Cα atoms that mapped to a seed atom.
            ca_map = {int(a): int(topo_to_seed[int(a)]) for a in ca_abs if int(a) in topo_to_seed}
            if len(ca_map) >= 1:
                aligned_seed_all = kabsch_align_positions(
                    seed_pos_nm[seed_idx, :], full_pos_nm[topo_idx, :], seed_pos_nm
                )
                ca_abs_arr = np.array(list(ca_map.keys()), dtype=int)
                ca_aligned_nm = aligned_seed_all[np.array(list(ca_map.values()), dtype=int), :]
            else:
                ca_abs_arr = np.array([], dtype=int)
                ca_aligned_nm = None
        sim.context.setPositions(new_full_pos * unit.nanometer)
        sim.minimizeEnergy(maxIterations=minimize_iters)
        min_state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
        min_pos_nm = min_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        if np.isnan(min_pos_nm).any():
            return {"fallback": True, "fallback_reason": "minimization_nan"}
        sim.context.setVelocitiesToTemperature(temperature_k * unit.kelvin, seed)
        cv_after_nm = cv_distance_from_positions_nm(min_pos_nm, cv_atom1, cv_atom2)
        # Cα RMSD: Kabsch-aligned conformer vs post-minimization peptide — measures
        # how much clash minimization distorted the graft; large values (>2 Å) indicate
        # the conformer was strained after insertion into the solvated box.
        if ca_aligned_nm is not None and ca_abs_arr.size:
            ca_post_nm = min_pos_nm[ca_abs_arr, :]
            ca_rmsd_A = float(np.sqrt(np.mean(np.sum((ca_aligned_nm - ca_post_nm) ** 2, axis=1)))) * 10.0
        else:
            ca_rmsd_A = float("nan")
        return {
            "fallback": False,
            "used_conformer": str(conformer["pdb_path"]),
            "graft_mode": graft_mode,
            "n_grafted_atoms": n_grafted,
            "cv_before_A": float(conformer.get("cv_A", float("nan"))),
            "cv_after_A": float(cv_after_nm * 10.0),
            "primary_cv_before": float(conformer.get("primary_cv_value", conformer.get("cv_A", float("nan")))),
            "primary_cv_units": str(conformer.get("primary_cv_units", "")),
            "secondary_cv_before": float(conformer.get("secondary_cv_value", float("nan"))),
            "ca_rmsd_A": ca_rmsd_A,
        }
    except Exception as e:
        return {"fallback": True, "fallback_reason": f"exception:{e}"}


def generate_us_starting_states_by_pulling(
    args,
    out_dir: Path,
    openmm,
    app,
    unit,
    topology,
    base_system,
    centers_nm,
    ks_kj_nm2,
    equil_state,
    primary_cv_def: dict,
    cv_atom1: int,
    cv_atom2: int,
    platform,
    props,
    progress: Optional[GuiProgressSink] = None,
    secondary_cv_centers=None,
    secondary_cv_ks_kj=None,
    secondary_cv_metadata: Optional[dict] = None,
):
    """Generate one starting structure per umbrella window by restrained CV pulling.

    This is intentionally separate from the GaMD/GaREUS production replicas.  It
    uses a plain Langevin integrator and the same umbrella functional form, then
    walks the terminal-distance CV from the NPT-equilibrated conformation toward
    the requested umbrella centers.  The returned positions are later used as the
    initial coordinates for the corresponding US/GaREUS window, instead of
    starting every window from the same midpoint structure.
    """
    nwin = int(len(centers_nm))
    if nwin <= 0:
        return [], []

    mode = str(getattr(args, "us_starting_structure_mode", "pull") or "pull").lower()
    if mode in {"npt", "equilibrated", "same", "none"}:
        pos = equil_state.getPositions()
        vel = None
        try:
            vel = equil_state.getVelocities()
        except Exception:
            vel = None
        return [pos for _ in range(nwin)], [vel for _ in range(nwin)]

    pull_steps = int(getattr(args, "us_pull_steps_per_window", 5000) or 0)
    pull_k_kcal_a2_requested = float(getattr(args, "us_pull_k_kcal_a2", 5.0) or 5.0)
    if primary_cv_is_contacts(args):
        pull_k_kcal_a2 = contact_us_pull_k_user(args, pull_k_kcal_a2_requested)
        if pull_k_kcal_a2 < pull_k_kcal_a2_requested - 1.0e-12 and getattr(args, "contact_us_pull_k_kcal", None) is None:
            print(
                f"WARNING [contact US start]: legacy us_pull_k_kcal_a2={pull_k_kcal_a2_requested:g} "
                f"was capped to {pull_k_kcal_a2:g} kcal/mol/CV^2 for contact-mode starting pulls. "
                "Set contact_us_pull_k_kcal to override."
            )
    else:
        pull_k_kcal_a2 = pull_k_kcal_a2_requested
    pull_k_kj_nm2 = primary_k_to_openmm_value(pull_k_kcal_a2, args)
    minimize_iters = int(getattr(args, "us_pull_minimize_iterations", 100) or 0)
    safe_chunk = int(getattr(args, "equil_safe_chunk_steps", 100) or 100)
    ts = float(getattr(args, "us_pull_timestep_fs", 0.0) or 0.0)
    _explicit_ts = ts > 0.0
    if not _explicit_ts:
        ts = min(float(getattr(args, "timestep_fs", 2.0) or 2.0), 2.0)
        if primary_cv_is_contacts(args):
            contact_ts = float(getattr(args, "contact_us_pull_timestep_fs", 1.0) or 1.0)
            if math.isfinite(contact_ts) and contact_ts > 0.0 and ts > contact_ts:
                print(
                    f"WARNING [contact US start]: auto pull timestep capped to {contact_ts:g} fs "
                    f"for contact-mode starting pulls. Set --us-pull-timestep-fs explicitly to override."
                )
                ts = contact_ts
        # Contact forces can change sharply when many switch functions respond at once;
        # smaller chunks make NaN localization and crash PDBs much more useful.
        contact_chunk = int(getattr(args, "contact_us_pull_safe_chunk_steps", 100) or 100)
        if contact_chunk > 0:
            safe_chunk = min(safe_chunk, contact_chunk)
    friction = float(getattr(args, "us_pull_friction_per_ps", 10.0) or 10.0)
    if primary_cv_is_contacts(args):
        friction = max(friction, float(getattr(args, "contact_us_pull_min_friction_per_ps", 20.0) or 20.0))

    seed_dir = getattr(args, "seed_conformers_dir", None)
    conformer_library: list[dict] = []
    resolved_cv_defs: dict = {}
    if seed_dir is not None:
        conformer_library = load_genpept_conformer_library(
            Path(seed_dir), cv_atom1, cv_atom2,
            primary_cv_def=primary_cv_def,
            args=args,
            topology=topology,
            secondary_cv_metadata=secondary_cv_metadata,
            resolved_cv_defs_out=resolved_cv_defs,
        )

    pull_dir = out_dir / "us_starting_structures"
    pull_dir.mkdir(exist_ok=True)
    log_path = pull_dir / "us_pulling_starting_structures.csv"

    # --- worker-count detection ---
    _setup_pname = platform.getName() if hasattr(platform, "getName") else str(platform)
    _dev_tok_str = str(props.get("DeviceIndex", "") or "")
    if not _dev_tok_str:
        _dev_tok_str = (
            str(getattr(args, "setup_device_index", "") or "")
            or _first_device_index(str(getattr(args, "device_index", "0") or "0"))
        )
    device_tokens = [x.strip() for x in _dev_tok_str.split(",") if x.strip()] or ["0"]
    _us_pull_workers_arg = str(getattr(args, "us_pull_workers", "auto") or "auto").lower().strip()
    if _us_pull_workers_arg in {"auto", "0", ""}:
        n_pull_workers = len(device_tokens) if _setup_pname in {"CUDA", "HIP", "OpenCL"} else 1
    else:
        try:
            n_pull_workers = max(1, int(_us_pull_workers_arg))
        except ValueError:
            n_pull_workers = 1
    n_pull_workers = max(1, min(n_pull_workers, nwin))

    _pull_progress = progress if n_pull_workers == 1 else None

    def _make_pull_sim(device_idx: str, seed_offset: int = 4242):
        ps = deserialize_system(openmm, base_system)
        ig = make_langevin_integrator(openmm, unit, args, timestep_fs=ts, friction_per_ps=friction, seed_offset=seed_offset)
        _plt_prec = str(props.get("Precision", "") or getattr(args, "setup_precision", "") or getattr(args, "precision", "mixed") or "mixed")
        _plt_thr = int(props.get("Threads", 0) or getattr(args, "cpu_threads", 1) or 1)
        _plt, _prp = platform_and_properties(openmm, _setup_pname, _plt_prec, device_idx, _plt_thr, args=args)
        s = app.Simulation(topology, ps, ig, _plt, _prp)
        return s

    if n_pull_workers > 1:
        print(f"    US pulling: {n_pull_workers} concurrent workers (devices: {', '.join(device_tokens[:n_pull_workers])}). k={pull_k_kcal_a2:.1f} kcal/mol, ts={ts:.1f} fs, {pull_steps:,} steps/win.")
        if pull_steps > 0 and nwin > 1:
            _centers_arr_tmp = np.asarray(centers_nm, dtype=float)
            _cv_span = float(np.max(_centers_arr_tmp) - np.min(_centers_arr_tmp)) * (1.0 if primary_cv_is_contacts(args) else 10.0)
            print(
                f"    NOTE: concurrent mode — each window pulls from equilibrated. "
                f"Extreme windows traverse the full CV span ({_cv_span:.2f} {primary_cv_units(args)}). "
                f"Increase --us-pull-steps-per-window (current: {pull_steps}) if quality check reports 'bad'."
            )
        _sim_pool: queue.Queue = queue.Queue()
        for _wi in range(n_pull_workers):
            _sim_pool.put(_make_pull_sim(device_tokens[_wi % len(device_tokens)], seed_offset=4242 + _wi * 137))
        sim = None
    else:
        print(f"    US pulling: 1 worker (sequential walk). k={pull_k_kcal_a2:.1f} kcal/mol, ts={ts:.1f} fs, {pull_steps:,} steps/win.")
        sim = _make_pull_sim(device_tokens[0], seed_offset=4242)
        sim.context.setPeriodicBoxVectors(*equil_state.getPeriodicBoxVectors())
        sim.context.setPositions(equil_state.getPositions())
        try:
            sim.context.setVelocities(equil_state.getVelocities())
        except Exception:
            sim.context.setVelocitiesToTemperature(float(args.temperature_k) * unit.kelvin, int(args.seed) + 4242)
        _sim_pool = None

    _init_pos_nm = equil_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    initial_cv_user = primary_cv_value_from_positions_nm(_init_pos_nm, primary_cv_def, args)
    centers_nm_arr = np.asarray(centers_nm, dtype=float)
    centers_user_arr = centers_nm_arr.copy() if primary_cv_is_contacts(args) else centers_nm_arr * 10.0
    nearest = int(np.argmin(np.abs(centers_user_arr - float(initial_cv_user))))
    positions_by_window = [None] * nwin
    velocities_by_window = [None] * nwin

    secondary_available = secondary_cv_centers is not None and secondary_cv_ks_kj is not None
    two_d_relax_mode = str(getattr(args, "us_2d_start_relax_mode", "auto") or "auto").lower()
    staged_2d_relax = bool(secondary_available and two_d_relax_mode in {"auto", "staged", "ramp", "on", "yes"})
    secondary_ramp_stages = max(1, int(getattr(args, "us_2d_start_secondary_ramp_stages", 3) or 3))
    distance_fraction = float(getattr(args, "us_2d_start_distance_fraction", 0.50) or 0.50)
    if not math.isfinite(distance_fraction):
        distance_fraction = 0.50
    distance_fraction = max(0.0, min(0.95, distance_fraction))
    pull_k_scale = max(1.0, float(getattr(args, "us_2d_start_secondary_k_pull_scale", 1.0) or 1.0))
    if staged_2d_relax and pull_steps > 0:
        _cv2_steps = max(1, int(pull_steps * (1.0 - distance_fraction)))
        _per_stage_ps = _cv2_steps * ts / (max(1, secondary_ramp_stages) * 1000.0)
        _effective_threshold = max(0.5, 10.0 / max(1.0, pull_k_scale))
        if _per_stage_ps < _effective_threshold:
            _needed = int(math.ceil(
                _effective_threshold * secondary_ramp_stages * 1000.0
                / max(ts, 1e-6)
                / max(1.0 - distance_fraction, 1e-3)
            ))
            print(
                f"WARNING [2D start pull]: secondary-CV ramp has only {_per_stage_ps:.1f} ps/stage "
                f"({secondary_ramp_stages} stage(s) × {_cv2_steps // secondary_ramp_stages} steps @ {ts:.1f} fs, "
                f"pull_k_scale={pull_k_scale:.1f}×). "
                f"Backbone φ/ψ transitions may not respond in time — replicas may miss their CV2 target. "
                f"Raise us_pull_steps_per_window to ≥{_needed} "
                f"and/or raise us_2d_start_secondary_k_pull_scale (try 5.0) to push harder."
            )

    def _set_secondary_restraint_for_window(sim, w: int, scale: float = 1.0) -> None:
        if not secondary_available:
            return
        try:
            sim.context.setParameter("ss0", float(secondary_cv_centers[w]))
            sim.context.setParameter("ss_k", float(secondary_cv_ks_kj[w]) * max(0.0, float(scale)))
        except Exception:
            pass

    def _current_primary_cv_from_context(sim) -> float:
        try:
            state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
            pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            value = primary_cv_value_from_positions_nm(pos_nm, primary_cv_def, args)
            return float(value) if math.isfinite(float(value)) else float("nan")
        except Exception:
            return float("nan")

    def _run_primary_pull_segment(
        sim,
        target_center: float,
        final_k_openmm: float,
        nsteps: int,
        label: str,
        progress_offset: int,
        message: str,
        start_center: Optional[float] = None,
    ) -> None:
        """Run a primary-CV pull segment, ramping contact pulls to avoid NaNs.

        Distance pulls keep the historical immediate target/k behavior.  Contact
        pulls are bounded but many-term CVs; applying full k to a far target in a
        single step can create a large collective impulse and NaN before the row
        even starts.  Ramp both r0 and k over short stages.
        """
        nsteps = int(nsteps)
        if nsteps <= 0:
            return
        if not primary_cv_is_contacts(args):
            sim.context.setParameter("r0", float(target_center))
            sim.context.setParameter("k", float(final_k_openmm))
            run_steps_safely(
                sim, nsteps, label,
                out_dir, app, topology, unit,
                chunk_size=safe_chunk, progress=_pull_progress,
                progress_total=max(1, nwin * max(1, pull_steps)),
                progress_offset=progress_offset,
                timestep_fs=ts, n_replicas=1,
                message=message,
            )
            return

        stages = max(1, int(getattr(args, "contact_us_pull_ramp_stages", 8) or 8))
        stages = min(stages, max(1, nsteps))
        if start_center is None or not math.isfinite(float(start_center)):
            start_center = _current_primary_cv_from_context(sim)
        if not math.isfinite(float(start_center)):
            start_center = float(target_center)
        start_center = float(start_center)
        target_center = float(target_center)
        final_k_openmm = float(final_k_openmm)
        base = nsteps // stages
        rem = nsteps % stages
        done = 0
        for idx in range(stages):
            nstage = base + (1 if idx < rem else 0)
            if nstage <= 0:
                continue
            frac = float(idx + 1) / float(stages)
            # Smooth the k ramp more than the center ramp.  Early stages mostly move
            # the restraint center near the current CV without violently pulling.
            center = start_center + frac * (target_center - start_center)
            k_stage = final_k_openmm * frac * frac
            sim.context.setParameter("r0", float(center))
            sim.context.setParameter("k", float(k_stage))
            if idx == 0 and minimize_iters > 0 and bool(getattr(args, "contact_us_pull_minimize_first_ramp", True)):
                try:
                    sim.minimizeEnergy(maxIterations=max(1, min(minimize_iters, 25)))
                except TypeError:
                    sim.minimizeEnergy()
            run_steps_safely(
                sim, int(nstage), label,
                out_dir, app, topology, unit,
                chunk_size=safe_chunk, progress=_pull_progress,
                progress_total=max(1, nwin * max(1, pull_steps)),
                progress_offset=progress_offset + done,
                timestep_fs=ts, n_replicas=1,
                message=message + f"; c-ramp {idx+1}/{stages} ({frac*100:.0f}%k)",
            )
            done += int(nstage)
        sim.context.setParameter("r0", float(target_center))
        sim.context.setParameter("k", float(final_k_openmm))

    def relax_to_window(sim, w: int, direction_label: str):
        """Pull sim to window w and return (row_dict, positions, velocities)."""
        if primary_cv_is_contacts(args):
            # Start contact pulls from the current CV with k=0, then ramp in the
            # MD segment.  This avoids an instantaneous many-contact impulse before
            # the first integration step.
            cur_primary = _current_primary_cv_from_context(sim)
            if math.isfinite(float(cur_primary)):
                sim.context.setParameter("r0", float(cur_primary))
            else:
                sim.context.setParameter("r0", float(centers_nm_arr[w]))
            sim.context.setParameter("k", 0.0)
        else:
            sim.context.setParameter("r0", float(centers_nm_arr[w]))
            sim.context.setParameter("k", float(pull_k_kj_nm2))
        if secondary_available:
            _set_secondary_restraint_for_window(sim, w, 0.0 if staged_2d_relax else 1.0)
        if minimize_iters > 0:
            try:
                sim.minimizeEnergy(maxIterations=minimize_iters)
            except TypeError:
                sim.minimizeEnergy()
        progress_base = int(sum(1 for x in positions_by_window if x is not None) * max(1, pull_steps))
        if pull_steps > 0:
            if staged_2d_relax:
                distance_steps = int(round(float(pull_steps) * distance_fraction))
                distance_steps = max(0, min(int(pull_steps), distance_steps))
                remaining_steps = int(pull_steps) - int(distance_steps)
                done_local = 0
                if distance_steps > 0:
                    _set_secondary_restraint_for_window(sim, w, 0.0)
                    _run_primary_pull_segment(
                        sim,
                        float(centers_nm_arr[w]), float(pull_k_kj_nm2), distance_steps,
                        "us_starting_pull_primary",
                        progress_base + done_local,
                        f"Pull w{w+1}/{nwin}: cv1→{centers_user_arr[w]:.3f} cv2→{float(secondary_cv_centers[w]):+.2f} k={pull_k_kcal_a2:.1f} [cv1 phase]",
                    )
                    done_local += int(distance_steps)
                stage_steps = []
                if remaining_steps > 0:
                    base = remaining_steps // secondary_ramp_stages
                    rem = remaining_steps % secondary_ramp_stages
                    stage_steps = [base + (1 if i < rem else 0) for i in range(secondary_ramp_stages)]
                for idx, nstage in enumerate(stage_steps):
                    if nstage <= 0:
                        continue
                    scale = float(idx + 1) / float(max(1, len(stage_steps)))
                    _set_secondary_restraint_for_window(sim, w, scale * pull_k_scale)
                    if minimize_iters > 0 and bool(getattr(args, "us_2d_start_minimize_each_ramp", False)):
                        try:
                            sim.minimizeEnergy(maxIterations=max(1, minimize_iters // max(1, secondary_ramp_stages)))
                        except TypeError:
                            sim.minimizeEnergy()
                    run_steps_safely(
                        sim, int(nstage), "us_starting_pull_secondary_ramp",
                        out_dir, app, topology, unit,
                        chunk_size=safe_chunk, progress=_pull_progress,
                        progress_total=max(1, nwin * max(1, pull_steps)),
                        progress_offset=progress_base + done_local,
                        timestep_fs=ts, n_replicas=1,
                        message=f"Pull w{w+1}/{nwin}: cv2-ramp {idx+1}/{len(stage_steps)} ({scale*pull_k_scale:.1f}×) cv1→{centers_user_arr[w]:.3f} cv2→{float(secondary_cv_centers[w]):+.2f}",
                    )
                    done_local += int(nstage)
                _set_secondary_restraint_for_window(sim, w, 1.0)  # revert to production k before saving
            else:
                _run_primary_pull_segment(
                    sim,
                    float(centers_nm_arr[w]), float(pull_k_kj_nm2), pull_steps,
                    "us_starting_pull",
                    progress_base,
                    f"Pull w{w+1}/{nwin}: cv1→{centers_user_arr[w]:.3f}{primary_cv_units(args)} k={pull_k_kcal_a2:.1f}",
                )
        state = sim.context.getState(getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=True)
        pos = state.getPositions()
        vel = state.getVelocities()
        positions_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        primary_value = primary_cv_value_from_positions_nm(positions_nm, primary_cv_def, args)
        pe_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        secondary_value = float("nan")
        secondary_center = float("nan")
        secondary_k_kcal = float("nan")
        secondary_bias_kcal = float("nan")
        secondary_delta = float("nan")
        if secondary_available and secondary_cv_metadata and secondary_cv_metadata.get("enabled"):
            secondary_value = secondary_structure_score_from_positions_nm(positions_nm, secondary_cv_metadata)
            try:
                secondary_center = float(secondary_cv_centers[w])
                secondary_k_kcal = float(secondary_cv_ks_kj[w]) / 4.184
                secondary_delta = float(secondary_value) - float(secondary_center)
                secondary_bias_kcal = 0.5 * float(secondary_k_kcal) * secondary_delta * secondary_delta
            except Exception:
                pass
        _unit_tag = "C" if primary_cv_is_contacts(args) else "A"
        _center_user = float(centers_user_arr[w])
        pdb_path = pull_dir / f"window_{w:03d}_center_{_center_user:.3f}{_unit_tag}_start.pdb"
        write_state_pdb(pdb_path, app, topology, pos)
        return {
            "window": int(w),
            "center_A": float(centers_user_arr[w]),
            "achieved_cv_A": float(primary_value),
            "delta_A": float(primary_value - centers_user_arr[w]),
            "primary_cv_value": float(primary_value),
            "primary_cv_center": float(centers_user_arr[w]),
            "primary_cv_delta": float(primary_value - centers_user_arr[w]),
            "primary_cv_units": primary_cv_units(args),
            "pull_k_kcal_mol_A2": float(pull_k_kcal_a2),
            "pull_steps": int(pull_steps),
            "pull_timestep_fs": float(ts),
            "two_d_relax_mode": "staged" if staged_2d_relax else "single_stage",
            "secondary_cv_center": secondary_center if math.isfinite(float(secondary_center)) else "",
            "achieved_secondary_cv": secondary_value if math.isfinite(float(secondary_value)) else "",
            "secondary_cv_delta": secondary_delta if math.isfinite(float(secondary_delta)) else "",
            "secondary_cv_k_kcal_mol": secondary_k_kcal if math.isfinite(float(secondary_k_kcal)) else "",
            "production_secondary_cv_bias_kcal_mol": secondary_bias_kcal if math.isfinite(float(secondary_bias_kcal)) else "",
            "potential_kj_mol": float(pe_kj),
            "direction": str(direction_label),
            "pdb": str(pdb_path),
        }, pos, vel

    rows = []
    graft_results: list[dict] = []
    seed_selection_rows: list[dict] = []
    seed_usage_counts: dict[str, int] = {}
    seed_selection_lock = threading.Lock()
    seed_selection_mode = str(getattr(args, "seed_selection_mode", "auto") or "auto").strip().lower().replace("_", "-")
    if seed_selection_mode == "auto":
        seed_selection_mode = "active-cv"
    seed_secondary_weight = max(0.0, float(getattr(args, "seed_secondary_weight", 1.0) or 0.0))
    seed_max_reuse = int(getattr(args, "seed_max_reuse_per_conformer", 0) or 0)
    primary_seed_scale = _finite_spacing_scale(centers_user_arr, fallback=(0.2 if primary_cv_is_contacts(args) else 1.0))
    secondary_seed_scale = _finite_spacing_scale(secondary_cv_centers if secondary_cv_centers is not None else [], fallback=0.25)

    seed_scoring_degradations: list[dict] = []
    if conformer_library:
        # Scientific-audit requirement: active-CV seed scoring must never
        # silently collapse to a lower-dimensional proxy (e.g. CV1 contacts
        # -> terminal distance, or CV2 dropped while intended active).  Detect
        # it here from the already-resolved flags/defs, warn loudly, and
        # record it in the seed_selection_report — do not hard-raise.
        seed_scoring_degradations = detect_seed_scoring_degradations(
            primary_is_contacts=primary_cv_is_contacts(args),
            rel_primary=resolved_cv_defs.get("rel_primary"),
            secondary_available=secondary_available,
            seed_secondary_weight=seed_secondary_weight,
            seed_selection_mode=seed_selection_mode,
            rel_secondary=resolved_cv_defs.get("rel_secondary"),
        )
        for _deg in seed_scoring_degradations:
            print(
                "!" * 78 + "\n"
                f"WARNING [SEED SCORING COLLAPSE] ({_deg['kind']}): {_deg['message']}\n"
                + "!" * 78
            )

    def _select_conformer_for_window(w: int) -> tuple[dict, dict]:
        """Choose the best GENPEPT seed for a window using active CV-space distance."""
        if not conformer_library:
            raise ValueError("empty conformer library")
        target_primary = float(centers_user_arr[w])
        target_secondary = float(secondary_cv_centers[w]) if secondary_available else float("nan")

        def _score(conf: dict) -> tuple[float, dict]:
            primary_value = float(conf.get("primary_cv_value", conf.get("cv_A", float("nan"))))
            # Legacy explicit distance mode is kept for reproducibility/debugging.
            if seed_selection_mode == "distance":
                primary_value = float(conf.get("cv_A", primary_value))
            primary_delta, primary_score = _primary_seed_score(primary_value, target_primary, primary_seed_scale)
            secondary_value = float(conf.get("secondary_cv_value", float("nan")))
            secondary_delta = float("nan")
            secondary_score = 0.0
            use_secondary = bool(
                seed_selection_mode == "active-cv"
                and secondary_available
                and seed_secondary_weight > 0.0
                and math.isfinite(target_secondary)
                and math.isfinite(secondary_value)
            )
            if use_secondary:
                secondary_delta = abs(secondary_value - target_secondary)
                secondary_score = seed_secondary_weight * secondary_delta / max(1.0e-12, secondary_seed_scale)
            elif (seed_selection_mode == "active-cv"
                  and secondary_available
                  and seed_secondary_weight > 0.0
                  and math.isfinite(target_secondary)
                  and not math.isfinite(secondary_value)):
                # NaN secondary CV: assign a unit penalty so this seed never beats a seed
                # with any finite secondary value.  Without this, NaN seeds score 0
                # on the secondary term and are systematically preferred.
                secondary_score = float(seed_secondary_weight)
            total = float(primary_score + secondary_score)
            components = {
                "window": int(w),
                "seed_selection_mode": seed_selection_mode,
                "target_primary_cv": target_primary,
                "primary_cv_units": primary_cv_units(args),
                "conformer_primary_cv": primary_value,
                "abs_primary_delta": primary_delta,
                "primary_scale": primary_seed_scale,
                "primary_score": primary_score,
                "target_secondary_cv": target_secondary,
                "conformer_secondary_cv": secondary_value,
                "abs_secondary_delta": secondary_delta,
                "secondary_scale": secondary_seed_scale,
                "secondary_weight": seed_secondary_weight,
                "secondary_score": secondary_score,
                "total_score": total,
                "conformer_pdb": str(conf.get("pdb_path", "")),
            }
            return total, components

        with seed_selection_lock:
            ranked = []
            for conf in conformer_library:
                key = str(conf.get("pdb_path", ""))
                if seed_max_reuse > 0 and seed_usage_counts.get(key, 0) >= seed_max_reuse:
                    continue
                total, components = _score(conf)
                ranked.append((total, key, conf, components))
            if not ranked:
                # Reuse cap was too strict; fall back to nearest unrestricted seed.
                ranked = []
                for conf in conformer_library:
                    total, components = _score(conf)
                    components["reuse_cap_relaxed"] = True
                    ranked.append((total, str(conf.get("pdb_path", "")), conf, components))
            ranked.sort(key=lambda item: (float(item[0]), item[1]))
            _total, key, conf, components = ranked[0]
            seed_usage_counts[key] = seed_usage_counts.get(key, 0) + 1
            components["reuse_count_after_selection"] = int(seed_usage_counts[key])
            return conf, _format_seed_component_dict(components)

    def _reset_sim_to_equil(s, seed_offset: int = 4242) -> None:
        s.context.setPeriodicBoxVectors(*equil_state.getPeriodicBoxVectors())
        s.context.setPositions(equil_state.getPositions())
        try:
            s.context.setVelocities(equil_state.getVelocities())
        except Exception:
            s.context.setVelocitiesToTemperature(float(args.temperature_k) * unit.kelvin, int(args.seed) + seed_offset)

    if conformer_library:
        print(
            f"    Generating US starting structures with GENPEPT seeding: "
            f"{len(conformer_library)} conformers available, {nwin} windows."
        )
        if n_pull_workers > 1:
            def _seed_task(w):
                s = _sim_pool.get()
                try:
                    _reset_sim_to_equil(s, seed_offset=5000 + w)
                    nearest_conf, seed_score_info = _select_conformer_for_window(w)
                    graft_status = graft_conformer_into_context(
                        s, topology, nearest_conf, cv_atom1, cv_atom2,
                        float(args.temperature_k), unit,
                        minimize_iters=minimize_iters,
                        seed=int(args.seed) + 5000 + w,
                    )
                    if graft_status["fallback"]:
                        _reset_sim_to_equil(s, seed_offset=5000 + w)
                        direction = "pull_fallback"
                    else:
                        direction = "seeded"
                    row, pos, vel = relax_to_window(s, w, direction)
                    return w, row, pos, vel, graft_status, seed_score_info
                finally:
                    _sim_pool.put(s)

            with concurrent.futures.ThreadPoolExecutor(max_workers=n_pull_workers) as ex:
                futures = {ex.submit(_seed_task, w): w for w in range(nwin)}
                for fut in concurrent.futures.as_completed(futures):
                    w, row, pos, vel, graft_status, seed_score_info = fut.result()
                    positions_by_window[w] = pos
                    velocities_by_window[w] = vel
                    rows.append(row)
                    seed_selection_rows.append(dict(seed_score_info))
                    graft_results.append({
                        "window": w,
                        "window_center_A": float(centers_user_arr[w]),
                        "window_primary_center": float(centers_user_arr[w]),
                        "primary_cv_units": primary_cv_units(args),
                        "secondary_cv_center": float(secondary_cv_centers[w]) if secondary_available else "",
                        "conformer_pdb": graft_status.get("used_conformer"),
                        "conformer_cv_A": graft_status.get("cv_before_A"),
                        "conformer_primary_cv": graft_status.get("primary_cv_before"),
                        "conformer_secondary_cv": graft_status.get("secondary_cv_before"),
                        "cv_after_graft_A": graft_status.get("cv_after_A"),
                        "ca_rmsd_A": graft_status.get("ca_rmsd_A"),
                        "seed_score": seed_score_info.get("total_score"),
                        "seed_score_components": seed_score_info,
                        "fallback": bool(graft_status["fallback"]),
                        "fallback_reason": graft_status.get("fallback_reason"),
                    })
                    print(f"    [US pull] window {w+1}/{nwin} done")
        else:
            for w in range(nwin):
                _reset_sim_to_equil(sim, seed_offset=5000 + w)
                nearest_conf, seed_score_info = _select_conformer_for_window(w)
                graft_status = graft_conformer_into_context(
                    sim, topology, nearest_conf, cv_atom1, cv_atom2,
                    float(args.temperature_k), unit,
                    minimize_iters=minimize_iters,
                    seed=int(args.seed) + 5000 + w,
                )
                if graft_status["fallback"]:
                    _reset_sim_to_equil(sim, seed_offset=5000 + w)
                    direction = "pull_fallback"
                else:
                    direction = "seeded"
                row, pos, vel = relax_to_window(sim, w, direction)
                positions_by_window[w] = pos
                velocities_by_window[w] = vel
                rows.append(row)
                seed_selection_rows.append(dict(seed_score_info))
                graft_results.append({
                    "window": w,
                    "window_center_A": float(centers_user_arr[w]),
                    "window_primary_center": float(centers_user_arr[w]),
                    "primary_cv_units": primary_cv_units(args),
                    "secondary_cv_center": float(secondary_cv_centers[w]) if secondary_available else "",
                    "conformer_pdb": graft_status.get("used_conformer"),
                    "conformer_cv_A": graft_status.get("cv_before_A"),
                    "conformer_primary_cv": graft_status.get("primary_cv_before"),
                    "conformer_secondary_cv": graft_status.get("secondary_cv_before"),
                    "cv_after_graft_A": graft_status.get("cv_after_A"),
                    "ca_rmsd_A": graft_status.get("ca_rmsd_A"),
                    "seed_score": seed_score_info.get("total_score"),
                    "seed_score_components": seed_score_info,
                    "fallback": bool(graft_status["fallback"]),
                    "fallback_reason": graft_status.get("fallback_reason"),
                })

        n_seeded = sum(1 for g in graft_results if not g["fallback"])
        n_fallback = len(graft_results) - n_seeded
        print(f"    Graft summary: {n_seeded}/{nwin} windows seeded, {n_fallback} fell back to NPT-pull.")
        _cv_units = primary_cv_units(args)
        _hdr = f"    {'win':>4}  {'center':>8}  {'CV_before':>9}  {'CV_after':>9}  {'CaRMSD(A)':>9}  {'conformer / status'}"
        print(_hdr)
        print("    " + "-" * (len(_hdr) - 4))
        for _g in sorted(graft_results, key=lambda x: int(x.get("window", 0))):
            _w = int(_g.get("window", 0))
            _ctr = _g.get("window_center_A", float("nan"))
            _ctr_s = f"{float(_ctr):.3f}" if _ctr is not None else "     n/a"
            if _g["fallback"]:
                _cv_b = _cv_a = _rmsd = "      n/a"
                _status = f"FALLBACK  reason={_g.get('fallback_reason', '?')}"
            else:
                _cv_b_v = _g.get("conformer_cv_A")
                _cv_a_v = _g.get("cv_after_graft_A")
                _rmsd_v = _g.get("ca_rmsd_A")
                _cv_b = f"{float(_cv_b_v):.3f}" if _cv_b_v is not None else "      n/a"
                _cv_a = f"{float(_cv_a_v):.3f}" if _cv_a_v is not None else "      n/a"
                _rmsd = f"{float(_rmsd_v):.3f}" if _rmsd_v is not None else "      n/a"
                _pdb = str(_g.get("conformer_pdb") or "")
                _status = _pdb if len(_pdb) <= 40 else "..." + _pdb[-37:]
            print(f"    {_w:>4}  {_ctr_s:>8}  {_cv_b:>9}  {_cv_a:>9}  {_rmsd:>9}  {_status}")
        print(f"    ({_cv_units}; CaRMSD: Kabsch-aligned conformer vs post-minimize peptide)")
        write_json(pull_dir / "graft_report.json", _json_ready(graft_results))
        if seed_selection_rows:
            seed_selection_rows_sorted = sorted(seed_selection_rows, key=lambda r: int(r.get("window", 0)))
            write_json(pull_dir / "seed_selection_report.json", _json_ready({
                "mode": "active_cv_seed_selection",
                "description": "GENPEPT survivor choice per umbrella window, scored in the active GAREUS CV space before graft/minimize/pull.",
                "seed_selection_mode": seed_selection_mode,
                "seed_secondary_weight": float(seed_secondary_weight),
                "seed_max_reuse_per_conformer": int(seed_max_reuse),
                "primary_scale": float(primary_seed_scale),
                "secondary_scale": float(secondary_seed_scale),
                "primary_cv": primary_cv_mode(args),
                "primary_cv_units": primary_cv_units(args),
                "secondary_cv_enabled": bool(secondary_available),
                "degradations": seed_scoring_degradations,
                "rows": seed_selection_rows_sorted,
            }))
            report_csv = pull_dir / "seed_selection_report.csv"
            with report_csv.open("w", newline="") as handle:
                fieldnames = [
                    "window", "seed_selection_mode", "conformer_pdb", "total_score",
                    "target_primary_cv", "conformer_primary_cv", "abs_primary_delta", "primary_score", "primary_cv_units",
                    "target_secondary_cv", "conformer_secondary_cv", "abs_secondary_delta", "secondary_score", "secondary_weight",
                    "reuse_count_after_selection", "reuse_cap_relaxed",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(seed_selection_rows_sorted)

    else:
        if primary_cv_is_contacts(args):
            print("    WARNING: nonlocal-contact CV is nondirectional; high-contact windows encourage collapse but do not specify which contacts form.")
        print(f"    US pulling: {nwin} windows, nearest={nearest+1}, initial cv1={initial_cv_user:.3f} {primary_cv_units(args)}, k={pull_k_kcal_a2:.1f} kcal/mol, ts={ts:.1f} fs, {pull_steps:,} steps/win.")

        if n_pull_workers > 1:
            # Concurrent: each window pulls independently from equilibrated state.
            # The sequential spine/row-base-state optimization is not applied;
            # every window bears the full CV-distance cost from equilibrated.
            def _pull_task(w):
                s = _sim_pool.get()
                try:
                    _reset_sim_to_equil(s, seed_offset=4242 + w)
                    row, pos, vel = relax_to_window(s, w, "independent")
                    return w, row, pos, vel
                finally:
                    _sim_pool.put(s)

            with concurrent.futures.ThreadPoolExecutor(max_workers=n_pull_workers) as ex:
                futures = {ex.submit(_pull_task, w): w for w in range(nwin)}
                for fut in concurrent.futures.as_completed(futures):
                    w, row, pos, vel = fut.result()
                    positions_by_window[w] = pos
                    velocities_by_window[w] = vel
                    rows.append(row)
                    print(f"    [US pull] window {w+1}/{nwin} done")

        else:
            # Sequential walk: nearest → compact branch → extended branch.
            # Each window starts from its neighbour's end state so even extreme
            # windows reach their target in the pull_steps budget.
            # In a 2D window grid (staged_2d_relax + multiple unique distances) the
            # sequential walk can carry a secondary-CV extreme from one row into the
            # next.  Fix: for each unique distance pre-pull a "row base state" with
            # ss_k=0, then reset the context at each row boundary before calling
            # relax_to_window.  1D or non-staged paths are unaffected.
            unique_dist_nm = sorted({round(float(d), 8) for d in centers_nm_arr})
            in_2d_grid_pull = staged_2d_relax and len(unique_dist_nm) > 1
            row_base_states: dict = {}
            if in_2d_grid_pull:
                _reset_sim_to_equil(sim, seed_offset=9998)
                if secondary_available:
                    try:
                        sim.context.setParameter("ss_k", 0.0)
                    except Exception:
                        pass
                row_pull_steps = max(1, int(pull_steps * distance_fraction))
                row_base_done = 0
                previous_row_center = _current_primary_cv_from_context(sim) if primary_cv_is_contacts(args) else None
                for row_dist_nm in unique_dist_nm:
                    _run_primary_pull_segment(
                        sim,
                        float(row_dist_nm), float(pull_k_kj_nm2), row_pull_steps,
                        "us_starting_pull_row_base",
                        row_base_done,
                        f"US start 2D row base to {primary_cv_format_value(float(row_dist_nm if primary_cv_is_contacts(args) else row_dist_nm * 10.0), args)}",
                        start_center=previous_row_center,
                    )
                    previous_row_center = float(row_dist_nm)
                    row_base_done += int(row_pull_steps)
                    row_base_states[round(float(row_dist_nm), 8)] = sim.context.getState(
                        getPositions=True, getVelocities=True, enforcePeriodicBox=True
                    )
                _reset_sim_to_equil(sim, seed_offset=4242)

            row, pos, vel = relax_to_window(sim, nearest, "nearest")
            positions_by_window[nearest] = pos
            velocities_by_window[nearest] = vel
            rows.append(row)
            nearest_state = sim.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
            nearest_pos = nearest_state.getPositions()
            nearest_vel = nearest_state.getVelocities()

            for w in range(nearest - 1, -1, -1):
                cur_dist = round(float(centers_nm_arr[w]), 8)
                if in_2d_grid_pull and cur_dist in row_base_states:
                    base = row_base_states[cur_dist]
                    sim.context.setPeriodicBoxVectors(*base.getPeriodicBoxVectors())
                    sim.context.setPositions(base.getPositions())
                    try:
                        sim.context.setVelocities(base.getVelocities())
                    except Exception:
                        sim.context.setVelocitiesToTemperature(float(args.temperature_k) * unit.kelvin, int(args.seed) + 4343 + w)
                row, pos, vel = relax_to_window(sim, w, "compact_branch")
                positions_by_window[w] = pos
                velocities_by_window[w] = vel
                rows.append(row)

            sim.context.setPeriodicBoxVectors(*equil_state.getPeriodicBoxVectors())
            sim.context.setPositions(nearest_pos)
            try:
                sim.context.setVelocities(nearest_vel)
            except Exception:
                sim.context.setVelocitiesToTemperature(float(args.temperature_k) * unit.kelvin, int(args.seed) + 4343)

            for w in range(nearest + 1, nwin):
                cur_dist = round(float(centers_nm_arr[w]), 8)
                if in_2d_grid_pull and cur_dist in row_base_states:
                    base = row_base_states[cur_dist]
                    sim.context.setPeriodicBoxVectors(*base.getPeriodicBoxVectors())
                    sim.context.setPositions(base.getPositions())
                    try:
                        sim.context.setVelocities(base.getVelocities())
                    except Exception:
                        sim.context.setVelocitiesToTemperature(float(args.temperature_k) * unit.kelvin, int(args.seed) + 4344 + w)
                row, pos, vel = relax_to_window(sim, w, "extended_branch")
                positions_by_window[w] = pos
                velocities_by_window[w] = vel
                rows.append(row)

            # Fallback for any skipped windows, though the loops above should fill all.
            for w in range(nwin):
                if positions_by_window[w] is None:
                    _reset_sim_to_equil(sim, seed_offset=4444 + w)
                    row, pos, vel = relax_to_window(sim, w, "fallback")
                    positions_by_window[w] = pos
                    velocities_by_window[w] = vel
                    rows.append(row)

    sorted_rows = sorted(rows, key=lambda r: int(r["window"]))
    # Quality-control the actual starting conformers under the production umbrella
    # definitions.  The pulling force constant is only a setup aid; for later US/MBAR
    # sanity the important question is whether each start is close to its production
    # center and whether its production umbrella bias is already absurd.
    pe_vals = np.asarray([float(r.get("potential_kj_mol", np.nan)) for r in sorted_rows], dtype=float)
    pe_med = float(np.nanmedian(pe_vals)) if pe_vals.size else float("nan")
    pe_mad = float(np.nanmedian(np.abs(pe_vals - pe_med))) if pe_vals.size and math.isfinite(pe_med) else float("nan")
    quality_rows = []
    n_warn = 0
    n_bad = 0
    for r in sorted_rows:
        w = int(r["window"])
        achieved_coord = float(r["achieved_cv_A"]) / 10.0 if primary_cv_is_distance(args) else float(r["achieved_cv_A"])
        center_coord = float(centers_nm_arr[w])
        prod_k_openmm = float(ks_kj_nm2[w])
        prod_k_kcal_a2 = kj_nm2_to_kcal_a2(prod_k_openmm) if primary_cv_is_distance(args) else prod_k_openmm / 4.184
        prod_bias_kj = harmonic_bias_energy_kj(achieved_coord, center_coord, prod_k_openmm)
        prod_bias_kcal = prod_bias_kj / 4.184
        abs_delta = abs(float(r.get("delta_A", float("nan"))))
        warnings = []
        status = "ok"
        if primary_cv_is_contacts(args):
            contact_scale = contact_normalization_denominator(primary_cv_def.get("contact_pairs", []), args)
            warn_delta = 0.15 if bool(getattr(args, "contact_normalize", True)) else max(1.0, 0.15 * max(1.0, contact_scale))
            bad_delta = 0.30 if bool(getattr(args, "contact_normalize", True)) else max(2.0, 0.30 * max(1.0, contact_scale))
            delta_units = primary_cv_units(args)
        else:
            warn_delta = 0.75
            bad_delta = 1.5
            delta_units = "A"
        if math.isfinite(abs_delta) and abs_delta > bad_delta:
            warnings.append(f"start is {abs_delta:.3g} {delta_units} from target center")
            status = "bad"
        elif math.isfinite(abs_delta) and abs_delta > warn_delta:
            warnings.append(f"start is {abs_delta:.3g} {delta_units} from target center")
            status = "warn"
        if math.isfinite(prod_bias_kcal) and prod_bias_kcal > 5.0:
            warnings.append(f"production umbrella bias at start is {prod_bias_kcal:.2f} kcal/mol")
            status = "bad"
        elif math.isfinite(prod_bias_kcal) and prod_bias_kcal > 1.0 and status == "ok":
            warnings.append(f"production umbrella bias at start is {prod_bias_kcal:.2f} kcal/mol")
            status = "warn"
        pe = float(r.get("potential_kj_mol", float("nan")))
        if math.isfinite(pe) and math.isfinite(pe_med) and math.isfinite(pe_mad) and pe_mad > 1.0e-6:
            robust_z = abs(pe - pe_med) / (1.4826 * pe_mad)
            r["potential_robust_z"] = float(robust_z)
            if robust_z > 8.0:
                warnings.append(f"potential energy robust-z {robust_z:.1f} relative to other starts")
                if status != "bad":
                    status = "warn"
        else:
            r["potential_robust_z"] = ""
        sec_bias = _scalar_to_float(r.get("production_secondary_cv_bias_kcal_mol", None))
        sec_delta = _scalar_to_float(r.get("secondary_cv_delta", None))
        total_bias_kcal = float(prod_bias_kcal) + (float(sec_bias) if sec_bias is not None and math.isfinite(float(sec_bias)) else 0.0)
        if sec_delta is not None and math.isfinite(float(sec_delta)) and abs(float(sec_delta)) > float(getattr(args, "us_2d_start_secondary_warn_delta", 0.35) or 0.35):
            warnings.append(f"secondary CV start is {float(sec_delta):+.3f} from target")
            if status == "ok":
                status = "warn"
        if sec_bias is not None and math.isfinite(float(sec_bias)) and float(sec_bias) > float(getattr(args, "us_2d_start_secondary_bad_bias_kcal", 5.0) or 5.0):
            warnings.append(f"secondary CV umbrella bias at start is {float(sec_bias):.2f} kcal/mol")
            status = "bad"
        elif sec_bias is not None and math.isfinite(float(sec_bias)) and float(sec_bias) > float(getattr(args, "us_2d_start_secondary_warn_bias_kcal", 1.0) or 1.0) and status == "ok":
            warnings.append(f"secondary CV umbrella bias at start is {float(sec_bias):.2f} kcal/mol")
            status = "warn"
        r["production_k_kcal_mol_A2"] = float(prod_k_kcal_a2)
        r["production_umbrella_bias_kcal_mol"] = float(prod_bias_kcal)
        r["production_umbrella_bias_kj_mol"] = float(prod_bias_kj)
        r["production_total_umbrella_bias_kcal_mol"] = float(total_bias_kcal)
        r["quality_status"] = status
        r["quality_warnings"] = "; ".join(warnings)
        quality_rows.append(dict(r))
        if status == "bad":
            n_bad += 1
        elif status == "warn":
            n_warn += 1

    with log_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "window", "center_A", "achieved_cv_A", "delta_A",
            "primary_cv_value", "primary_cv_center", "primary_cv_delta", "primary_cv_units",
            "pull_k_kcal_mol_A2",
            "production_k_kcal_mol_A2", "production_umbrella_bias_kcal_mol", "production_umbrella_bias_kj_mol",
            "secondary_cv_center", "achieved_secondary_cv", "secondary_cv_delta", "secondary_cv_k_kcal_mol",
            "production_secondary_cv_bias_kcal_mol", "production_total_umbrella_bias_kcal_mol",
            "pull_steps", "pull_timestep_fs", "two_d_relax_mode", "potential_kj_mol", "potential_robust_z",
            "quality_status", "quality_warnings", "direction", "pdb",
        ], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(quality_rows)

    quality_payload = {
        "mode": "us_starting_structure_quality",
        "description": "Checks pulled starting conformers against the final production umbrella centers/k values; these are setup diagnostics, not reweighting inputs.",
        "n_windows": int(nwin),
        "n_ok": int(sum(1 for r in quality_rows if r.get("quality_status") == "ok")),
        "n_warn": int(n_warn),
        "n_bad": int(n_bad),
        "primary_cv": primary_cv_mode(args),
        "primary_cv_label": primary_cv_label(args),
        "primary_cv_units": primary_cv_units(args),
        "max_abs_delta_A": float(max((abs(float(r.get("delta_A", 0.0))) for r in quality_rows), default=float("nan"))),
        "max_abs_primary_delta": float(max((abs(float(r.get("primary_cv_delta", r.get("delta_A", 0.0)))) for r in quality_rows), default=float("nan"))),
        "primary_delta_units": primary_cv_units(args),
        "max_production_umbrella_bias_kcal_mol": _safe_max_finite(quality_rows, "production_umbrella_bias_kcal_mol"),
        "max_production_secondary_cv_bias_kcal_mol": _safe_max_finite(quality_rows, "production_secondary_cv_bias_kcal_mol"),
        "max_production_total_umbrella_bias_kcal_mol": _safe_max_finite(quality_rows, "production_total_umbrella_bias_kcal_mol"),
        "two_d_relax_mode": "staged" if staged_2d_relax else "single_stage",
        "csv": str(log_path),
        "rows": quality_rows,
    }
    write_json(pull_dir / "us_starting_structure_quality.json", _json_ready(quality_payload))
    if n_bad or n_warn:
        _qmsg = f"WARNING: US starting-structure quality: {n_bad} bad, {n_warn} warn"
        if n_bad > 0 and secondary_available and staged_2d_relax:
            _cv2_steps_q = max(1, int(pull_steps * (1.0 - distance_fraction)))
            _ps_stage_q = _cv2_steps_q * ts / (max(1, secondary_ramp_stages) * 1000.0)
            _needed_q = int(math.ceil(
                10.0 * secondary_ramp_stages * 1000.0
                / max(ts, 1e-6) / max(1.0 - distance_fraction, 1e-3)
            ))
            _qmsg += (
                f"\n  {n_bad}/{nwin} windows missed their CV2 target "
                f"(ramp = {_ps_stage_q:.1f} ps/stage × {secondary_ramp_stages} stages = "
                f"{_ps_stage_q * secondary_ramp_stages:.1f} ps total). "
                f"Production will show all replicas oscillating near CV2≈0. "
                f"Fix: set us_pull_steps_per_window ≥ {_needed_q} "
                f"(current: {pull_steps}) and/or raise us_2d_start_secondary_k_pull_scale "
                f"(current: {pull_k_scale:.1f}, try 5.0) and raise secondary_cv_adaptive_max_k_kcal "
                f"(current max: {float(getattr(args, 'secondary_cv_adaptive_max_k_kcal', 10.0)):.1f}, try 50.0)."
            )
        print(_qmsg + f"\n  See: {pull_dir / 'us_starting_structure_quality.json'}")
    # Inter-window pairwise Cα RMSD — diversity check for all final US starting structures
    try:
        _pep_res = peptide_residues(topology)
        _ca_abs = []
        for _res in _pep_res:
            try:
                _ca_abs.append(find_atom_in_residue(_res, "CA"))
            except ValueError:
                pass
        if len(_ca_abs) >= 2:
            _ca_arr = np.array(_ca_abs, dtype=int)
            _valid_wins = sorted(w for w, p in enumerate(positions_by_window) if p is not None)
            if len(_valid_wins) >= 2:
                _ca_pos = {}
                for _w in _valid_wins:
                    _p = positions_by_window[_w]
                    _pos_nm = np.array(_p.value_in_unit(unit.nanometer)) if hasattr(_p, "value_in_unit") else np.asarray(_p)
                    _ca_pos[_w] = _pos_nm[_ca_arr, :]

                def _kabsch_rmsd_a(p, q):
                    pc = p - np.mean(p, axis=0)
                    qc = q - np.mean(q, axis=0)
                    H = pc.T @ qc
                    _U, _Sv, _Vt = np.linalg.svd(H)
                    _d = np.linalg.det(_Vt.T @ _U.T)
                    _D = np.diag([1.0, 1.0, _d])
                    _R = _Vt.T @ _D @ _U.T
                    return float(np.sqrt(np.mean(np.sum(((pc @ _R.T) - qc) ** 2, axis=1)))) * 10.0

                _nv = len(_valid_wins)
                _rmsd_mat = np.zeros((_nv, _nv))
                for _ii, _wi in enumerate(_valid_wins):
                    for _jj, _wj in enumerate(_valid_wins):
                        if _ii < _jj:
                            _r = _kabsch_rmsd_a(_ca_pos[_wi], _ca_pos[_wj])
                            _rmsd_mat[_ii, _jj] = _r
                            _rmsd_mat[_jj, _ii] = _r
                _off = _rmsd_mat[np.triu_indices(_nv, k=1)]
                print(
                    f"    Inter-window Cα RMSD (Å, Kabsch-aligned, {_nv} windows): "
                    f"min={float(np.min(_off)):.2f}  mean={float(np.mean(_off)):.2f}  max={float(np.max(_off)):.2f}"
                )
                _rmsd_csv = pull_dir / "starting_structure_rmsd.csv"
                with _rmsd_csv.open("w", newline="") as _fh:
                    _cw = csv.writer(_fh)
                    _cw.writerow(["window"] + [str(_w) for _w in _valid_wins])
                    for _ii, _wi in enumerate(_valid_wins):
                        _cw.writerow([_wi] + [f"{_rmsd_mat[_ii, _jj]:.4f}" for _jj in range(_nv)])
                print(f"    Inter-window RMSD matrix → {_rmsd_csv}")
    except Exception as _rmsd_exc:
        print(f"    WARNING: inter-window RMSD computation failed: {_rmsd_exc}")

    print(f"    US starting structures written to {pull_dir}")
    return positions_by_window, velocities_by_window


__all__ = [
    "deserialize_system",
    "harmonic_bias_energy_kj",
    "kabsch_align_positions",
    "cv_distance_nm",
    "cv_distance_and_potential_from_state",
    "load_genpept_conformer_library",
    "graft_conformer_into_context",
    "generate_us_starting_states_by_pulling",
]
