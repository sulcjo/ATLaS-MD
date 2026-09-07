"""Seed-library loading, per-row validation and per-member conformer resolution.

Split out of ``driver.py`` to keep that module under the ~400-line cap. Covers:
round 0's GENPEPT library (delegated to ``gareus.seeding.load_genpept_conformer_library``),
round >= 1's production-frames CSV (validated here, since a malformed row would otherwise
silently corrupt the frozen stratification coordinate), and resolving one plan row's
grafting conformer by identity rather than a bare positional index (the library the driver
reloads at run time is not guaranteed to match the one in effect when ``plan.csv`` was
written -- see ``_resolve_conformer``).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional

from gareus.cv import find_atom_in_residue, peptide_residues
from gareus.seeding import _read_pdb_conformer_atoms, load_genpept_conformer_library


def _ca_indices_in_seed(library: List[dict], topology) -> Optional[List[int]]:
    """Topology CA indices mapped through a library entry's own atom map.

    GENPEPT survivor conformers share one atom-naming convention, so the first
    entry whose map covers every peptide CA is representative for the whole
    library. Falls back to ``None`` (every atom) when no entry's map is
    complete -- e.g. a CA-trace-only seed where "every atom" already is the CA
    trace (``describe_seeds`` docstring).
    """
    ca_topology = [find_atom_in_residue(res, "CA") for res in peptide_residues(topology)]
    for entry in library:
        atom_map = entry.get("topology_to_conformer_atom_index") or {}
        if atom_map and all(idx in atom_map for idx in ca_topology):
            return [int(atom_map[idx]) for idx in ca_topology]
    return None


_PRODUCTION_FRAME_NUMERIC_COLUMNS = ("cv1", "rg_nm", "e2e_nm")


def _validate_production_frame_row(csv_path, row_index: int, row: dict) -> dict:
    """Validate one production-frames CSV row (round >= 1 seed source).

    ``pdb_path`` must be a non-empty string; ``cv1``/``rg_nm``/``e2e_nm`` must
    each parse to a finite float. Any violation raises ``ValueError`` naming
    the file, the (0-based, header-excluded) row and the offending column.
    Before this check, an empty numeric field silently became ``NaN``
    (``float(x or "nan")``), which ``np.searchsorted`` places in the LAST
    stratification bin -- corrupting the frozen-edge property without a
    trace -- and a non-empty garbage field raised a bare ``ValueError`` with
    no row/file context to debug from.
    """
    pdb_path = row.get("pdb_path")
    if not str(pdb_path or "").strip():
        raise ValueError(f"{csv_path} row {row_index}: column 'pdb_path' is {pdb_path!r}, expected a non-empty path")
    values: Dict[str, float] = {}
    for column in _PRODUCTION_FRAME_NUMERIC_COLUMNS:
        raw = row.get(column)
        text = str(raw).strip() if raw is not None else ""
        value = float("nan")
        if text:
            try:
                value = float(text)
            except ValueError:
                value = float("nan")
        if not text or not math.isfinite(value):
            raise ValueError(f"{csv_path} row {row_index}: column {column!r} is {raw!r}, expected a finite number")
        values[column] = value
    return {"pdb_path": str(pdb_path).strip(), **values}


def _load_production_frame_library(csv_path) -> List[dict]:
    """Round >= 1 seed source: ``pdb_path,cv1,rg_nm,e2e_nm`` rows (S5 re-seeding).

    Each row's PDB is a solute-only frame written by a previous swarm member
    (``write_solute_only_pdb``), so its atom count/order already match the
    peptide topology exactly -- an empty ``topology_to_conformer_atom_index``
    triggers ``graft_conformer_into_context``'s wholesale (Kabsch-aligned)
    path, no name-mapping needed. Every row is validated (see
    ``_validate_production_frame_row``) before use; the validated ``rg_nm``/
    ``e2e_nm`` are stored on the entry so ``build_or_load_plan`` never has to
    re-parse the raw CSV strings.
    """
    library: List[dict] = []
    with Path(csv_path).open(newline="") as f:
        for row_index, row in enumerate(csv.DictReader(f)):
            validated = _validate_production_frame_row(csv_path, row_index, row)
            pos_nm, _atoms = _read_pdb_conformer_atoms(validated["pdb_path"])
            library.append({
                "pdb_path": validated["pdb_path"],
                "positions_nm": pos_nm,
                "topology_to_conformer_atom_index": {},
                "source_row": dict(row),
                "cv_A": validated["cv1"],
                "primary_cv_value": validated["cv1"],
                "primary_cv_units": "",
                "secondary_cv_value": float("nan"),
                "validated_rg_nm": validated["rg_nm"],
                "validated_e2e_nm": validated["e2e_nm"],
            })
    return library


def _load_seed_library_for_round(args, round_index: int, *, topology, primary_cv_def) -> List[dict]:
    if round_index <= 0:
        return load_genpept_conformer_library(
            Path(args.seed_conformers_dir), primary_cv_def=primary_cv_def, args=args, topology=topology,
        )
    seed_source = str(getattr(args, "swarm_seed_source", "genpept") or "genpept")
    if seed_source != "production-frames":
        raise SystemExit(f"--swarm-round {round_index} (>= 1) needs --swarm-seed-source production-frames")
    csv_path = getattr(args, "swarm_production_seed_csv", None)
    if not csv_path or not Path(csv_path).exists():
        raise SystemExit(
            f"--swarm-round {round_index} needs --swarm-production-seed-csv PATH "
            "(existing CSV with columns pdb_path,cv1,rg_nm,e2e_nm)"
        )
    return _load_production_frame_library(csv_path)


def _resolve_conformer(row: dict, library: List[dict], library_by_path: Dict[str, dict]) -> dict:
    """Resolve one plan row's grafting conformer by its recorded ``seed_pdb`` path.

    The GENPEPT library re-sorts by ``primary_cv_value`` and later rounds only
    add seeds (frozen envelope, spec Sec.3.6), so the library reloaded at run
    time can differ in length/order from the one in effect when ``plan.csv``
    was written -- a bare positional ``seed_id`` index is not safe on its own.
    Resolves by ``row["seed_pdb"]`` first; the positional fallback (via
    ``seed_id``) is accepted only when it ALSO points at that same recorded
    path. Raises ``ValueError`` naming both paths when neither resolves.
    """
    seed_pdb = str(row["seed_pdb"])
    conformer = library_by_path.get(seed_pdb)
    if conformer is not None:
        return conformer
    seed_index = int(str(row["seed_id"]).split("_")[1])
    if 0 <= seed_index < len(library):
        candidate = library[seed_index]
        candidate_path = str(candidate.get("pdb_path", ""))
        if candidate_path == seed_pdb:
            return candidate
        raise ValueError(
            f"swarm member {row.get('member_id')}: plan seed_pdb {seed_pdb!r} (seed_id {row['seed_id']!r}) "
            f"not found by path, and the positional fallback library[{seed_index}] has pdb_path "
            f"{candidate_path!r} instead -- the seed library changed since plan.csv was written"
        )
    raise ValueError(
        f"swarm member {row.get('member_id')}: plan seed_pdb {seed_pdb!r} (seed_id {row['seed_id']!r}) "
        f"is not resolvable in a library of {len(library)} entries -- the seed library changed "
        "since plan.csv was written, or the plan/library pairing is otherwise broken"
    )
