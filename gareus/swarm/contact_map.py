"""Per-frame residue contact map of the swarm (contact-map CV1 spec, step 1).

Every swarm member records, next to ``trace.csv`` and ``torsion_features.npy``, one row per
trace row of the **soft-min heavy-atom distance** (A) between every pair of peptide residues
at least ``min_sequence_separation`` apart:

    d_ij = -lambda ln sum_{a in i, b in j} exp(-r_ab / lambda)        (r_ab in A)

computed in the numerically stable form ``m - lambda ln sum exp(-(r - m)/lambda)`` with
``m = min r``, so it is a smooth upper-bounded approximation of the minimum distance
(m - lambda ln N <= d_ij <= m). Distances, not switched contacts, are stored so the contact
switch (r0, shape) can be re-tuned later without new MD; the 2026-10-01 calibration chose a
rational switch at r0 4.5 A on top of this (spec ``2026-10-01-contact-map-cv1.md``).

Files per member: ``contact_map_features.npy`` (n_rows x n_pairs, float64, row i = trace row i;
n_pairs may be 0 for a peptide shorter than the separation, never an error)
and ``contact_map_index.json`` (schema ``swarm_contact_map_v1``: residue pairs, their heavy
atom indices, lambda, separation, and a sha256 of that definition). Heavy atoms = every atom
of the residue whose element is not hydrogen; residues = ``cv.peptide_residues`` (water and
ions excluded), in topology order. Caveats (not changed here): ``peptide_residues`` also keeps
caps (ACE/NME count as residues for the separation) and any other non-water/non-ion solute,
and does not distinguish chains; its ion test compares ``res.name.upper()`` with a set holding
"Na+"/"Cl-", so Amber-style "Na+"/"Cl-" ions would be included (Modeller's NA/CL are not). Positions are the member's own frame positions (one
molecule, kept whole by ``enforcePeriodicBox``), so no minimum image is applied -- the same
convention as the trace's ``e2e_nm`` and the torsion features.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from ..correctness._io import digest, json_bytes

SCHEMA = "swarm_contact_map_v1"
QUANTITY = "soft_min_heavy_atom_distance_angstrom"
DEFAULT_MIN_SEQUENCE_SEPARATION = 3
DEFAULT_LAMBDA_ANGSTROM = 0.2
FEATURES_NAME = "contact_map_features.npy"
INDEX_NAME = "contact_map_index.json"
#: Atom selections. "heavy": soft-min over the residues' heavy atoms (the 2026-10-01 calibration
#: definition); "ca": the C-alpha distance (soft-min of one atom pair = the distance itself). The CA
#: map is what production runs as CV1: additive over residue pairs, so it has a cheap force that is
#: exact on GPU (the heavy soft-min has none in plain OpenMM, see gareus/contact_map_force.py).
ATOMS_HEAVY = "heavy"
ATOMS_CA = "ca"
CA_QUANTITY = "ca_distance_angstrom"
CA_FEATURES_NAME = "ca_map_features.npy"
CA_INDEX_NAME = "ca_map_index.json"


def file_names(atoms: str = ATOMS_HEAVY) -> tuple:
    """(features, index) file names of a member's map for an atom selection."""
    if atoms == ATOMS_CA:
        return CA_FEATURES_NAME, CA_INDEX_NAME
    if atoms == ATOMS_HEAVY:
        return FEATURES_NAME, INDEX_NAME
    raise ValueError(f"unknown contact-map atom selection {atoms!r}")


def definition_atoms(definition: Mapping[str, Any]) -> str:
    return str(definition.get("atom_selection", ATOMS_HEAVY))


_H_NAME = re.compile(r"^\d*H")       # H, HA, 1HB, 2HD1 ... (PDB-style names) when no element


def _is_hydrogen(atom) -> bool:
    element = getattr(atom, "element", None)
    symbol = getattr(element, "symbol", None) if element is not None else None
    if symbol:
        return str(symbol).upper() == "H"
    return bool(_H_NAME.match(str(atom.name).strip().upper()))


def contact_map_definition(topology, *, min_sequence_separation: int = DEFAULT_MIN_SEQUENCE_SEPARATION,
                           lambda_angstrom: float = DEFAULT_LAMBDA_ANGSTROM, atoms: str = ATOMS_HEAVY) -> Dict[str, Any]:
    """The residue pairs and their atoms (heavy atoms, or the C-alpha) for ``topology`` (JSON-safe).

    The heavy-atom definition keeps its historical bytes (no ``atom_selection`` field)."""
    from ..cv import peptide_residues

    sep = int(min_sequence_separation)
    lam = float(lambda_angstrom)
    if sep < 1:
        raise ValueError("min_sequence_separation must be >= 1")
    if not lam > 0.0:
        raise ValueError("lambda_angstrom must be > 0")
    residues = peptide_residues(topology)
    file_names(atoms)
    if atoms == ATOMS_CA:
        # residues without exactly one CA (caps such as ACE/NME) are not part of the CA map; the
        # sequence separation counts the remaining residues
        residues = [r for r in residues if sum(1 for a in r.atoms() if str(a.name).strip() == "CA") == 1]
        heavy = [[int(a.index) for a in res.atoms() if str(a.name).strip() == "CA"] for res in residues]
    else:
        heavy = [[int(a.index) for a in res.atoms() if not _is_hydrogen(a)] for res in residues]
    if any(not h for h in heavy):
        raise ValueError("a peptide residue has no heavy atom")
    pairs = [{"i": i, "j": j, "atoms_i": heavy[i], "atoms_j": heavy[j]}
             for i in range(len(residues)) for j in range(i + sep, len(residues))]
    body = {
        "schema": SCHEMA, "quantity": CA_QUANTITY if atoms == ATOMS_CA else QUANTITY,
        **({"atom_selection": ATOMS_CA} if atoms == ATOMS_CA else {}),
        "min_sequence_separation": sep, "lambda_angstrom": lam,
        "residues": [{"index": k, "name": str(r.name), "id": str(r.id)} for k, r in enumerate(residues)],
        "pairs": pairs,
    }
    return {**body, "sha256": digest(json_bytes(body))}


class ContactMapEvaluator:
    """Vectorised soft-min distances for one definition (precomputed flat atom-pair arrays)."""

    def __init__(self, definition: Mapping[str, Any]):
        if definition.get("schema") != SCHEMA:
            raise ValueError(f"not a {SCHEMA} definition")
        body = {k: v for k, v in definition.items() if k != "sha256"}
        if "sha256" in definition and digest(json_bytes(body)) != definition["sha256"]:
            raise ValueError("contact-map definition digest mismatch (edited after it was built?)")
        self.lam = float(definition["lambda_angstrom"])
        a_idx, b_idx, starts = [], [], []
        for p in definition["pairs"]:
            starts.append(len(a_idx))
            for a in p["atoms_i"]:
                for b in p["atoms_j"]:
                    a_idx.append(int(a)); b_idx.append(int(b))
        self.n_pairs = len(starts)
        if not starts:                     # a peptide too short for any pair: empty rows
            return
        self.a = np.asarray(a_idx, dtype=np.int64)
        self.b = np.asarray(b_idx, dtype=np.int64)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.group = np.repeat(np.arange(len(starts)), np.diff(np.append(self.starts, len(a_idx))))

    def __call__(self, positions_nm) -> np.ndarray:
        """(n_pairs,) soft-min heavy-atom distances in A at one frame's positions (nm)."""
        if self.n_pairs == 0:
            return np.zeros(0, dtype=np.float64)
        x = np.asarray(positions_nm, dtype=np.float64)
        r = np.linalg.norm(x[self.b] - x[self.a], axis=1) * 10.0               # A
        m = np.minimum.reduceat(r, self.starts)
        s = np.add.reduceat(np.exp(-(r - m[self.group]) / self.lam), self.starts)
        return m - self.lam * np.log(s)


def soft_min_distance(r_angstrom: Sequence[float], lambda_angstrom: float) -> float:
    """Reference (unvectorised) soft-min of one residue pair's atom distances."""
    r = np.asarray(r_angstrom, dtype=np.float64)
    m = float(r.min())
    return float(m - lambda_angstrom * np.log(np.sum(np.exp(-(r - m) / lambda_angstrom))))


def write_index(path, definition: Mapping[str, Any]) -> None:
    from ..io import write_json
    write_json(path, dict(definition))


def read_index(path) -> Dict[str, Any]:
    with open(path) as fh:
        data = json.load(fh)
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} index")
    body = {k: v for k, v in data.items() if k != "sha256"}
    if digest(json_bytes(body)) != data.get("sha256"):
        raise ValueError(f"{path}: contact-map definition digest mismatch")
    return data


def load_member_contact_map(member_dir, atoms: str = ATOMS_HEAVY):
    """(features, definition) of one member's map for ``atoms``, held to each other: the index's
    digest and the feature width = its pair count. None when the member predates that map (a round
    resumed across the 2026-10-01 deploys can mix members with and without it)."""
    from pathlib import Path
    member_dir = Path(member_dir)
    f_name, i_name = file_names(atoms)
    feats, index = member_dir / f_name, member_dir / i_name
    if not (feats.exists() and index.exists()):
        return None
    definition = read_index(index)
    if definition_atoms(definition) != atoms:
        raise ValueError(f"{index}: atom selection {definition_atoms(definition)!r}, expected {atoms!r}")
    X = np.load(feats)
    if X.ndim != 2 or X.shape[1] != len(definition["pairs"]):
        raise ValueError(f"{feats}: shape {X.shape} does not match {len(definition['pairs'])} pairs in {index}")
    return X, definition


def pair_labels(definition: Mapping[str, Any]) -> List[str]:
    res = definition["residues"]
    return [f"{res[p['i']]['name']}{res[p['i']]['id']}-{res[p['j']]['name']}{res[p['j']]['id']}"
            for p in definition["pairs"]]


__all__ = ["ATOMS_CA", "ATOMS_HEAVY", "CA_FEATURES_NAME", "CA_INDEX_NAME", "CA_QUANTITY", "definition_atoms",
           "file_names", "DEFAULT_LAMBDA_ANGSTROM", "DEFAULT_MIN_SEQUENCE_SEPARATION", "FEATURES_NAME", "INDEX_NAME",
           "QUANTITY", "SCHEMA", "ContactMapEvaluator", "contact_map_definition", "load_member_contact_map",
           "pair_labels", "read_index",
           "soft_min_distance", "write_index"]
