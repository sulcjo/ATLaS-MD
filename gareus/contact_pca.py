"""Contact-pattern PCA: two stratification axes from a conformer's residue-pair contact matrix.

One definition, shared by GENPEPT (seed selection) and the swarm (member stratification), so a
seed's axes mean the same thing wherever they are computed:

* residues: the peptide's C-alpha-bearing residues in chain order (no ACE/NME caps -- they have
  no CA), ``n_residues`` of them;
* contact vector: binary, one entry per residue pair (i, j) with j - i >= ``min_sep`` (the upper
  triangle of the residue contact matrix, ``np.triu_indices(n, k=min_sep)`` order), 1 when the
  CA-CA distance is below ``cutoff_A`` -- exactly GENPEPT's ``contact_vector_from_coords``;
* basis: PCA (mean-centred SVD) of a pool of contact vectors, the top two components, each signed
  so its largest-|loading| entry is positive (deterministic across refits of the same pool).

The basis is persisted (``contact_pca_basis.json``, ``CONTACT_PCA_SCHEMA``, with a content
sha256) and reused unchanged downstream: GENPEPT fits it once on its generation pool; the swarm
loads it from the seed library (or fits it on the library when an older library has none) and
freezes it in ``swarm/round_000/`` for later rounds. numpy only: GENPEPT imports it without the
rest of ``gareus``.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

CONTACT_PCA_SCHEMA = "contact_pca_basis_v1"
BASIS_FILENAME = "contact_pca_basis.json"
DEFAULT_CUTOFF_A = 8.0
DEFAULT_MIN_SEP = 3
N_COMPONENTS = 2
AXIS_NAMES = ("cpc1", "cpc2")


def pair_indices(n_residues: int, min_sep: int) -> Tuple[np.ndarray, np.ndarray]:
    n, k = int(n_residues), max(0, int(min_sep))
    if n <= 1 or k >= n:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    i, j = np.triu_indices(n, k=k)
    return i.astype(np.int32), j.astype(np.int32)


def contact_vector(ca_A: np.ndarray, *, cutoff_A: float = DEFAULT_CUTOFF_A, min_sep: int = DEFAULT_MIN_SEP) -> np.ndarray:
    """Binary contact vector of one conformer from its CA coordinates in Angstrom (n x 3)."""
    ca = np.asarray(ca_A, dtype=float)
    i, j = pair_indices(ca.shape[0], min_sep)
    if i.size == 0:
        return np.zeros(0, dtype=np.int8)
    d2 = np.einsum("ij,ij->i", ca[i] - ca[j], ca[i] - ca[j])
    return (d2 < float(cutoff_A) ** 2).astype(np.int8)


@dataclass(frozen=True)
class ContactPCABasis:
    cutoff_A: float
    min_sep: int
    n_residues: int
    mean: Tuple[float, ...]
    components: Tuple[Tuple[float, ...], ...]       # N_COMPONENTS rows of length n_pairs
    explained_variance_ratio: Tuple[float, ...]
    n_fit: int
    source: str = ""

    @property
    def n_pairs(self) -> int:
        return len(self.mean)

    def project(self, contacts: np.ndarray) -> np.ndarray:
        """(n, 2) axis values of contact vectors (n x n_pairs); a single vector gives (1, 2)."""
        c = np.atleast_2d(np.asarray(contacts, dtype=float))
        if c.shape[1] != self.n_pairs:
            raise ValueError(f"contact vectors have {c.shape[1]} pairs, the basis {self.n_pairs} "
                             f"({self.n_residues} residues, min_sep {self.min_sep})")
        return (c - np.asarray(self.mean)) @ np.asarray(self.components).T

    def project_ca(self, ca_A: np.ndarray) -> Tuple[float, float]:
        ca = np.asarray(ca_A, dtype=float)
        if ca.shape[0] != self.n_residues:
            raise ValueError(f"{ca.shape[0]} CA atoms, the basis was fitted on {self.n_residues} residues")
        v = self.project(contact_vector(ca, cutoff_A=self.cutoff_A, min_sep=self.min_sep))[0]
        return float(v[0]), float(v[1])

    def as_record(self) -> dict:
        body = {"schema_version": CONTACT_PCA_SCHEMA, "cutoff_A": float(self.cutoff_A), "min_sep": int(self.min_sep),
                "n_residues": int(self.n_residues), "pair_order": "triu_indices(n_residues, k=min_sep)",
                "mean": [float(x) for x in self.mean],
                "components": [[float(x) for x in row] for row in self.components],
                "explained_variance_ratio": [float(x) for x in self.explained_variance_ratio],
                "n_fit": int(self.n_fit), "source": str(self.source)}
        body["sha256"] = _digest(body)
        return body

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.as_record(), indent=2))
        tmp.replace(path)
        return path

    @property
    def sha256(self) -> str:
        return self.as_record()["sha256"]


def _digest(body: dict) -> str:
    payload = {k: v for k, v in body.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def fit_basis(contacts: np.ndarray, *, cutoff_A: float, min_sep: int, n_residues: int,
              source: str = "") -> ContactPCABasis:
    """PCA basis of a pool of binary contact vectors (rows). Constant pairs simply load zero."""
    c = np.atleast_2d(np.asarray(contacts, dtype=float))
    n, p = c.shape
    expected = pair_indices(n_residues, min_sep)[0].size
    if p != expected:
        raise ValueError(f"contact vectors have {p} pairs; {n_residues} residues at min_sep {min_sep} give {expected}")
    mean = c.mean(axis=0) if n else np.zeros(p)
    x = c - mean
    comps = np.zeros((N_COMPONENTS, p))
    evr = np.zeros(N_COMPONENTS)
    if n >= 2 and p >= 1:
        _u, s, vt = np.linalg.svd(x, full_matrices=False)
        var = s ** 2
        total = float(var.sum())
        for k in range(min(N_COMPONENTS, vt.shape[0])):
            row = vt[k]
            j = int(np.argmax(np.abs(row)))
            comps[k] = row if row[j] >= 0 else -row
            evr[k] = float(var[k] / total) if total > 0 else 0.0
    return ContactPCABasis(float(cutoff_A), int(min_sep), int(n_residues), tuple(float(v) for v in mean),
                           tuple(tuple(float(v) for v in r) for r in comps), tuple(float(v) for v in evr), int(n),
                           str(source))


def load_basis(path) -> ContactPCABasis:
    rec = json.loads(Path(path).read_text())
    if rec.get("schema_version") != CONTACT_PCA_SCHEMA:
        raise ValueError(f"{path}: schema {rec.get('schema_version')!r}, expected {CONTACT_PCA_SCHEMA}")
    if rec.get("sha256") and rec["sha256"] != _digest(rec):
        raise ValueError(f"{path}: sha256 mismatch (file edited after it was written)")
    return ContactPCABasis(float(rec["cutoff_A"]), int(rec["min_sep"]), int(rec["n_residues"]),
                           tuple(float(x) for x in rec["mean"]),
                           tuple(tuple(float(x) for x in r) for r in rec["components"]),
                           tuple(float(x) for x in rec.get("explained_variance_ratio", (0.0, 0.0))),
                           int(rec.get("n_fit", 0)), str(rec.get("source", "")))


def read_pdb_ca_A(path) -> np.ndarray:
    """C-alpha coordinates (Angstrom, file order) of a PDB: ATOM/HETATM records named CA, first
    altloc only. Caps have no CA, so this is the peptide's CA trace."""
    out = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA" and line[16] in (" ", "A"):
            out.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(out, dtype=float).reshape(-1, 3)


def quantile_edges(values: Sequence[float], n_bins: int) -> np.ndarray:
    """Quantile bin edges (de-duplicated; outer edges cover every finite value)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n_bins = max(1, int(n_bins))
    if v.size == 0:
        return np.array([0.0, 1.0])
    edges = np.unique(np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        edges = np.array([edges[0], edges[0] + 1e-9])
    edges[0] = min(edges[0], v.min())
    edges[-1] = max(edges[-1], v.max()) + 1e-12
    return edges


def bin_index(x: float, edges: np.ndarray) -> int:
    e = np.asarray(edges, dtype=float)
    if not math.isfinite(float(x)):
        return 0
    return int(min(max(np.searchsorted(e, x, side="right") - 1, 0), e.size - 2))


__all__ = ["AXIS_NAMES", "BASIS_FILENAME", "CONTACT_PCA_SCHEMA", "ContactPCABasis", "DEFAULT_CUTOFF_A",
           "DEFAULT_MIN_SEP", "bin_index", "contact_vector", "fit_basis", "load_basis", "pair_indices",
           "quantile_edges", "read_pdb_ca_A"]
