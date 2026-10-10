"""S0: describe GENPEPT seeds in label-free coordinates and stratify them.

No native reference anywhere: heavy-CV1 (nonlocal contacts), Rg and end-to-end
distance are the descriptors, optionally plus two contact-pattern PCA axes
(``cpc1``/``cpc2``, gareus/contact_pca.py: PCA of the CA residue-pair contact
vector) when ``--swarm-bins`` has five entries. Equal quota per occupied cell
corrects the draw (r7 survivors are compactness-skewed), not the ensemble;
``--swarm-max-members`` caps the total (round-robin over cells).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class SeedDescriptor:
    seed_id: str
    pdb_path: str
    cv1: float
    rg_nm: float
    e2e_nm: float
    cpc1: float = float("nan")      # contact-PCA axes (only with 5-axis --swarm-bins)
    cpc2: float = float("nan")


BASE_AXES = ("cv1", "rg", "e2e")
CONTACT_AXES = ("cpc1", "cpc2")
_AXIS_ATTR = {"cv1": "cv1", "rg": "rg_nm", "e2e": "e2e_nm", "cpc1": "cpc1", "cpc2": "cpc2"}


def axes_for_bins(bins) -> Tuple[str, ...]:
    """Stratification axes named by the length of ``--swarm-bins`` (3: cv1/rg/e2e, 5: + cpc1/cpc2)."""
    n = len(tuple(bins))
    if n == 3:
        return BASE_AXES
    if n == 5:
        return BASE_AXES + CONTACT_AXES
    raise ValueError(f"--swarm-bins needs 3 (cv1,rg,e2e) or 5 (+ two contact-PCA axes) ints, got {n}")


def axis_value(seed: "SeedDescriptor", axis: str) -> float:
    return float(getattr(seed, _AXIS_ATTR[axis]))


def rg_and_e2e_nm(ca_positions_nm: np.ndarray) -> tuple[float, float]:
    ca = np.asarray(ca_positions_nm, dtype=float)
    if ca.ndim != 2 or ca.shape[0] < 2:
        return float("nan"), float("nan")
    centre = ca.mean(axis=0)
    rg = float(np.sqrt(((ca - centre) ** 2).sum(axis=1).mean()))
    e2e = float(np.linalg.norm(ca[-1] - ca[0]))
    return rg, e2e


CSV_RG_KEY, CSV_E2E_KEY = "rg_nm", "end_to_end_nm"     # GENPEPT ConformerRecord fields inherited by final_survivor_seeds.csv
CROSSCHECK_TOL_NM = 0.05


def describe_seeds(library: List[dict], ca_indices_in_seed: Optional[List[int]], contact_pairs, args,
                   mismatches: Optional[List[dict]] = None,
                   dropped_out: Optional[List[tuple]] = None, contact_basis=None) -> List[SeedDescriptor]:
    """One descriptor per library entry; entries whose geometry cannot be evaluated are dropped.

    heavy-CV1 is ALWAYS computed here (the library has a raw CA ``contact_count`` but no
    heavy-atom contact-fraction column). Rg/E2E are recomputed from the seed PDB and compared
    with the library's own ``rg_nm``/``end_to_end_nm`` when those columns exist
    (``entry["source_row"]``); a |Δ| > CROSSCHECK_TOL_NM is appended to ``mismatches`` and the
    recomputed value wins. ``ca_indices_in_seed`` are indices into ``entry["positions_nm"]``
    (seed-atom numbering); None -> every atom (backbone-only seeds).

    A dropped entry (empty/malformed positions, a CV1 exception, or a non-finite
    cv1/rg/e2e) is never silent: when ``dropped_out`` is given it receives one
    ``(seed_id, reason)`` tuple per drop, and if anything was dropped a single
    WARNING line is printed naming the count and the first reason (so a malformed
    ``contact_pairs``/``args`` that discards every seed is diagnosable at the call
    site instead of surfacing only as a distant "no occupied stratification cells").

    ``contact_basis`` (a ``gareus.contact_pca.ContactPCABasis``): also the two contact-PCA axes,
    from the CA atoms of the seed's own PDB (``pdb_path``); a seed whose CA count does not match
    the basis is dropped with that reason.
    """
    from gareus.cv import nonlocal_contact_cv_from_positions_nm
    out: List[SeedDescriptor] = []
    dropped: List[tuple] = []
    for i, entry in enumerate(library):
        seed_id = f"seed_{i:05d}"
        pos = np.asarray(entry.get("positions_nm"), dtype=float)
        if pos.ndim != 2 or pos.shape[0] == 0:
            dropped.append((seed_id, "empty or malformed positions_nm"))
            continue
        try:
            cv1 = float(nonlocal_contact_cv_from_positions_nm(pos, contact_pairs, args))
        except Exception as exc:
            cv1 = float("nan")
            cv1_exc = exc
        else:
            cv1_exc = None
        ca = pos if ca_indices_in_seed is None else pos[[j for j in ca_indices_in_seed if 0 <= j < pos.shape[0]]]
        rg, e2e = rg_and_e2e_nm(ca)
        if not (math.isfinite(cv1) and math.isfinite(rg) and math.isfinite(e2e)):
            reason = repr(cv1_exc) if cv1_exc is not None else "non-finite rg/e2e"
            dropped.append((seed_id, reason))
            continue
        src = entry.get("source_row") or {}
        for key, mine in ((CSV_RG_KEY, rg), (CSV_E2E_KEY, e2e)):
            try:
                theirs = float(src.get(key, "nan"))
            except (TypeError, ValueError):
                theirs = float("nan")
            if math.isfinite(theirs) and abs(theirs - mine) > CROSSCHECK_TOL_NM and mismatches is not None:
                mismatches.append({"seed_index": i, "pdb_path": str(entry.get("pdb_path", "")), "column": key, "csv": theirs, "recomputed": mine})
        cpc = (float("nan"), float("nan"))
        if contact_basis is not None:
            try:
                cpc = seed_contact_axes(entry, contact_basis)
            except Exception as exc:
                dropped.append((seed_id, f"contact-PCA axes: {exc}"))
                continue
        out.append(SeedDescriptor(seed_id=seed_id, pdb_path=str(entry.get("pdb_path", "")), cv1=cv1, rg_nm=rg, e2e_nm=e2e,
                                  cpc1=cpc[0], cpc2=cpc[1]))
    if dropped:
        print(f"WARNING: swarm.describe_seeds: dropped {len(dropped)} of {len(library)} seeds "
              f"(first reason: {dropped[0][1]})")
    if dropped_out is not None:
        dropped_out.extend(dropped)
    return out


def seed_contact_axes(entry: dict, basis) -> Tuple[float, float]:
    """(cpc1, cpc2) of one library entry from the CA atoms of its PDB file."""
    from gareus.contact_pca import read_pdb_ca_A
    return basis.project_ca(read_pdb_ca_A(entry.get("pdb_path", "")))


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n_bins = max(1, int(n_bins))
    if v.size == 0:
        return np.array([0.0, 1.0])
    edges = np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:                       # all values identical
        edges = np.array([edges[0], edges[0] + 1e-9])
    edges[0] = min(edges[0], v.min()); edges[-1] = max(edges[-1], v.max()) + 1e-12
    return edges


def _bin_index(x: float, edges: np.ndarray) -> int:
    return int(min(max(np.searchsorted(edges, x, side="right") - 1, 0), edges.size - 2))


def stratification_edges(seeds: List[SeedDescriptor], bins) -> Dict[str, np.ndarray]:
    """Quantile edges per axis (axes from ``axes_for_bins``)."""
    axes = axes_for_bins(bins)
    return {a: quantile_edges(np.array([axis_value(s, a) for s in seeds]), int(n)) for a, n in zip(axes, bins)}


def cells_from_edges(seeds: List[SeedDescriptor], edges: Dict[str, "np.ndarray"], axes) -> Dict[tuple, List[SeedDescriptor]]:
    cells: Dict[tuple, List[SeedDescriptor]] = {}
    arrays = {a: np.asarray(edges[a], dtype=float) for a in axes}
    for s in seeds:
        key = tuple(_bin_index(axis_value(s, a), arrays[a]) for a in axes)
        cells.setdefault(key, []).append(s)
    return dict(sorted(cells.items()))


def stratify_cells(seeds: List[SeedDescriptor], bins) -> Dict[tuple, List[SeedDescriptor]]:
    if not seeds:
        return {}
    return cells_from_edges(seeds, stratification_edges(seeds, bins), axes_for_bins(bins))


def cell_id(key: Tuple[int, int, int]) -> str:
    return "_".join(str(int(k)) for k in key)


VELOCITY_SEED_STRIDE = 100003     # large prime: keeps per-member velocity seeds apart across different base_seed values


def plan_members(cells: Dict[tuple, List[SeedDescriptor]], replicates_per_cell: int, *,
                 seed_ns: float, budget_ns: Optional[float], base_seed: int, axes=BASE_AXES,
                 max_members: int = 0) -> tuple[List[dict], dict]:
    """Equal quota R per occupied cell. budget_ns, when given, derives R = floor(budget / (cells·seed_ns)).

    ``max_members`` > 0 caps the total: members are dealt round-robin over the cells (largest
    cells first, then by key) -- every cell its first (distinct) seed before any cell its second,
    and so on up to R per cell -- until the cap. Rows keep today's cell-grouped order. One
    ``cell_<axis>`` column per stratification axis."""
    n_cells = len(cells)
    if n_cells == 0:
        raise ValueError("no occupied stratification cells")
    seed_ns = float(seed_ns)
    if budget_ns is not None:
        r = int(math.floor(float(budget_ns) / (n_cells * seed_ns)))
        if r < 1:
            raise ValueError(f"budget {budget_ns} ns cannot fund one {seed_ns} ns member per cell ({n_cells} cells)")
    else:
        r = int(replicates_per_cell)
        if r < 1:
            raise ValueError("replicates_per_cell must be >= 1")
    quota = {key: r for key in cells}
    cap = int(max_members or 0)
    if cap > 0 and cap < r * n_cells:
        quota = {key: 0 for key in cells}
        order = sorted(cells, key=lambda k: (-len(cells[k]), k))
        given = 0
        for _j in range(r):
            for key in order:
                if given >= cap:
                    break
                quota[key] += 1
                given += 1
    rows: List[dict] = []
    n_distinct = n_vel = 0
    member = 0
    for key, members in cells.items():
        ordered = sorted(members, key=lambda s: s.seed_id)
        for j in range(quota[key]):
            s = ordered[j % len(ordered)]
            replicate = j // len(ordered)
            if replicate == 0: n_distinct += 1
            else: n_vel += 1
            row = {"member_id": member, "cell_id": cell_id(key)}
            row.update({f"cell_{a}": int(k) for a, k in zip(axes, key)})
            row.update({"seed_id": s.seed_id, "seed_pdb": s.pdb_path, "replicate": replicate,
                        "velocity_seed": int(base_seed) * VELOCITY_SEED_STRIDE + member})
            rows.append(row)
            member += 1
    meta = {"n_cells": n_cells, "replicates_per_cell": r, "n_members": len(rows),
            "budget_ns": float(len(rows) * seed_ns), "seed_ns": seed_ns,
            "n_distinct_seeds": n_distinct, "n_velocity_replicates": n_vel}
    if cap > 0:
        meta.update({"max_members": cap, "n_cells_with_members": sum(1 for q in quota.values() if q > 0),
                     "members_per_cell_min": min(quota.values()), "members_per_cell_max": max(quota.values())})
    return rows, meta
