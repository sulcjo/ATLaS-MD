"""
GENPEPT prescan utilities.

This module provides simple helpers for rescoring GENPEPT output structures
using a crude contact‐fraction estimator.  It is intentionally lightweight and
does not depend on OpenMM or the full GAREUS CV machinery.  The purpose is to
provide a quick look at how far the nonlocal contact CV might reach in a set
of seed structures before expensive MD is launched.

Note that this is not a replacement for proper CV evaluation.  It simply
computes a fraction of Cα–Cα contacts below a cutoff distance, normalized by
the number of possible residue pairs, respecting a minimum sequence separation.
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import numpy as np

def _parse_ca_coordinates(pdb_path: str) -> Tuple[List[int], np.ndarray]:
    """
    Parse Cα atoms from a PDB file.

    Returns a tuple of (residue_ids, coords_nm) where residue_ids is a list of
    integer residue sequence numbers, and coords_nm is an array of positions in
    nanometres.  If no Cα atoms are found the coords array will be empty.
    """
    res_ids: List[int] = []
    coords: List[Tuple[float, float, float]] = []
    try:
        with open(pdb_path, "r") as handle:
            for line in handle:
                if not line.startswith("ATOM"):
                    continue
                # PDB columns: https://www.wwpdb.org/documentation/file-format-content/format23/sect9.html#ATOM
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue
                # residue sequence number
                try:
                    resnum = int(line[22:26])
                except Exception:
                    # fallback to sequential index if residue ID missing
                    resnum = len(res_ids)
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except Exception:
                    continue
                res_ids.append(resnum)
                # convert Å to nm
                coords.append((x * 0.1, y * 0.1, z * 0.1))
    except Exception:
        # If the file cannot be read, return empty results.
        return [], np.zeros((0, 3), dtype=float)
    return res_ids, np.asarray(coords, dtype=float)


def compute_contact_fraction(
    res_ids: Iterable[int],
    coords_nm: np.ndarray,
    r0_nm: float = 0.45,
    min_seq_sep: int = 3,
) -> float:
    """
    Compute a simple contact fraction for a peptide.

    Pairs of Cα atoms separated by at least min_seq_sep residues are counted.
    A pair contributes 1.0 to the total if its distance is below r0_nm,
    otherwise 0.0.  The return value is the total divided by the number of
    eligible pairs, or 0.0 if there are no eligible pairs.
    """
    n = len(res_ids)
    if n < 2 or coords_nm.shape[0] < 2:
        return 0.0
    total_pairs = 0
    contacts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(int(res_ids[j]) - int(res_ids[i])) < int(min_seq_sep):
                continue
            total_pairs += 1
            rij = float(np.linalg.norm(coords_nm[i] - coords_nm[j]))
            if rij < float(r0_nm):
                contacts += 1
    if total_pairs <= 0:
        return 0.0
    return float(contacts) / float(total_pairs)


def scan_genpept_directory(
    search_dir: str,
    min_seq_sep: int = 3,
    r0_nm: float = 0.45,
    recursive: bool = True,
) -> List[float]:
    """
    Walk a directory and compute contact fractions for all .pdb files.

    This helper will recursively search the given directory for PDB files and
    return a list of computed contact fractions.  Files that cannot be read
    or parsed yield no contribution.
    """
    fractions: List[float] = []
    base = Path(search_dir)
    if not base.exists():
        return fractions
    for root, _dirs, files in os.walk(base):
        for name in files:
            if not name.lower().endswith(".pdb"):
                continue
            pdb_path = os.path.join(root, name)
            res_ids, coords_nm = _parse_ca_coordinates(pdb_path)
            if coords_nm.size == 0:
                continue
            frac = compute_contact_fraction(res_ids, coords_nm, r0_nm=r0_nm, min_seq_sep=min_seq_sep)
            fractions.append(frac)
        if not recursive:
            # Do not descend into subdirectories unless explicitly requested.
            break
    return fractions


def estimate_contact_frontier(
    prescan_dirs: Iterable[str],
    percentile: float = 99.0,
    margin: float = 0.02,
    min_span: float = 0.10,
    min_valid: int = 10,
    min_seq_sep: int = 3,
    r0_nm: float = 0.45,
) -> Dict[str, float]:
    """
    Given a list of prescan directories, compute an estimated contact frontier.

    The frontier_max is computed as:

      frontier_max = min(1.0, np.percentile(fractions, percentile) + margin)

    If fewer than min_valid fractions are available, returns default low/high
    contact limits of 0.0 and min_span.
    """
    all_fractions: List[float] = []
    for d in prescan_dirs:
        fracs = scan_genpept_directory(d, min_seq_sep=min_seq_sep, r0_nm=r0_nm, recursive=True)
        all_fractions.extend(fracs)
    if len(all_fractions) >= int(min_valid):
        arr = np.asarray(all_fractions, dtype=float)
        perc = np.percentile(arr, float(percentile))
        # clamp to [0, 1]
        frontier = max(0.0, min(1.0, float(perc) + float(margin)))
        return {
            "has_data": True,
            "n_structures": len(all_fractions),
            "percentile": float(percentile),
            "observed_max": float(np.nanmax(arr)),
            "frontier_max": frontier,
        }
    # If not enough data, return a conservative span.
    return {
        "has_data": False,
        "n_structures": len(all_fractions),
        "frontier_max": float(min_span),
    }