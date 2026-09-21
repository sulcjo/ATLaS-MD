"""Collision-aware solvent repacking for grafted seeds (repair spec F03; findings N01, N02).

A graft overwrites the peptide with a compact seed aligned onto the extended-chain frame and
leaves the water box untouched, so waters that occupied the space the compact peptide now
fills overlap it -- at r ~ 0.01 nm the 1/r^12 wall is 1e18 kJ/mol and L-BFGS cannot move.
The first repair pushed each overlapping molecule radially and independently; that made two
waters exactly coincident (N01) and oscillated between two obstacles without saying so (N02).

This module moves whole solvent molecules rigidly, never the peptide, and for every candidate
move checks BOTH the peptide-solvent clearance and an emergency solvent-solvent clearance
against every other solvent molecule, under the periodic box. Candidate directions are the
radial escape plus a fixed deterministic spherical set; translation lengths are bounded; a
molecule that cannot be freed alone may be moved together with a small local cluster; and
the work is bounded by the policy. RESOLVED is only ever declared by a final, independent
validation pass. Everything is initialization geometry: these thresholds are singularity
avoidance, not thermodynamic exclusion radii.

NumPy only. Python 3.9 compatible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPAIR_POLICY_VERSION = "solvent_repair_policy_v1"

STATUS_RESOLVED = "RESOLVED"
STATUS_UNRESOLVED = "UNRESOLVED"
STATUS_INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True)
class RepairPolicy:
    """Initialization clearance policy. Distances in nm."""

    peptide_heavy_heavy_nm: float = 0.22
    peptide_hydrogen_nm: float = 0.16          # any peptide-solvent pair involving a hydrogen
    solvent_heavy_heavy_nm: float = 0.16       # emergency thresholds between DISTINCT solvent molecules
    solvent_hydrogen_nm: float = 0.10
    margin_nm: float = 0.03
    max_rounds: int = 25
    max_translation_nm: float = 0.60
    n_translation_steps: int = 6
    max_cluster_size: int = 4
    version: str = REPAIR_POLICY_VERSION

    def peptide_threshold(self, heavy_solvent: np.ndarray, heavy_peptide: np.ndarray) -> np.ndarray:
        return np.where(heavy_solvent[:, None] & heavy_peptide[None, :], self.peptide_heavy_heavy_nm,
                        self.peptide_hydrogen_nm)

    def solvent_threshold(self, heavy_a: np.ndarray, heavy_b: np.ndarray) -> np.ndarray:
        return np.where(heavy_a[:, None] & heavy_b[None, :], self.solvent_heavy_heavy_nm, self.solvent_hydrogen_nm)


@dataclass
class RepairResult:
    status: str
    reason_code: str
    positions_nm: np.ndarray
    n_groups_moved: int = 0
    n_attempts: int = 0
    n_rounds: int = 0
    n_peptide_solvent_clashes: int = 0
    n_solvent_solvent_clashes: int = 0
    minimum_clearance_by_pair_class: Dict[str, float] = field(default_factory=dict)
    moved_groups: List[int] = field(default_factory=list)
    seed_id: Optional[str] = None
    repair_seed: int = 0
    repair_policy_version: str = REPAIR_POLICY_VERSION

    def as_record(self) -> dict:
        return {
            "status": self.status, "reason_code": self.reason_code,
            "n_groups_moved": int(self.n_groups_moved), "n_attempts": int(self.n_attempts),
            "n_rounds": int(self.n_rounds),
            "n_peptide_solvent_clashes": int(self.n_peptide_solvent_clashes),
            "n_solvent_solvent_clashes": int(self.n_solvent_solvent_clashes),
            "minimum_clearance_by_pair_class": {k: float(v) for k, v in self.minimum_clearance_by_pair_class.items()},
            "seed_id": self.seed_id, "repair_seed": int(self.repair_seed),
            "repair_policy_version": self.repair_policy_version,
        }


# ---------------------------------------------------------------------------------------------
# Periodic geometry
# ---------------------------------------------------------------------------------------------

def validate_reduced_box(box_nm) -> np.ndarray:
    """Accept an orthorhombic or OpenMM-reduced triclinic box (rows a, b, c) or raise ValueError."""
    box = np.asarray(box_nm, dtype=float)
    if box.shape != (3, 3) or not np.all(np.isfinite(box)):
        raise ValueError("box must be a finite 3x3 array of row vectors")
    a, b, c = box
    if not (a[0] > 0 and b[1] > 0 and c[2] > 0):
        raise ValueError("box diagonal must be positive")
    if abs(a[1]) > 1e-12 or abs(a[2]) > 1e-12 or abs(b[2]) > 1e-12:
        raise ValueError("box is not in OpenMM reduced form (a along x, b in the xy plane)")
    if abs(b[0]) > 0.5 * a[0] + 1e-9 or abs(c[0]) > 0.5 * a[0] + 1e-9 or abs(c[1]) > 0.5 * b[1] + 1e-9:
        raise ValueError("box is not in OpenMM reduced form (off-diagonal components exceed half the diagonal)")
    return box


def safe_radius_nm(box: np.ndarray) -> float:
    """Half the smallest inter-plane spacing of the cell.

    Two atoms closer than this have a unique minimum image and the sequential rounding in
    :func:`min_image` (OpenMM's convention) recovers it exactly. Clearance checks run far
    below it; the repair refuses a cell whose safe radius does not cover its search cutoff."""
    a, b, c = box
    volume = abs(float(np.dot(a, np.cross(b, c))))
    widths = [volume / np.linalg.norm(np.cross(b, c)), volume / np.linalg.norm(np.cross(a, c)),
              volume / np.linalg.norm(np.cross(a, b))]
    return 0.5 * float(min(widths))


def min_image(d: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Minimum-image displacement for a reduced triclinic box (OpenMM's convention)."""
    d = np.array(d, dtype=float, copy=True)
    a, b, c = box[0], box[1], box[2]
    d -= c * np.round(d[..., 2:3] / c[2])
    d -= b * np.round(d[..., 1:2] / b[1])
    d -= a * np.round(d[..., 0:1] / a[0])
    return d


def _pair_distances(pa: np.ndarray, pb: np.ndarray, box: np.ndarray) -> np.ndarray:
    return np.linalg.norm(min_image(pa[:, None, :] - pb[None, :, :], box), axis=-1)


# ---------------------------------------------------------------------------------------------
# Clash detection: brute-force oracle and cell-list search
# ---------------------------------------------------------------------------------------------

def peptide_solvent_penetration(group_pos, group_heavy, pep_pos, pep_heavy, box, policy: RepairPolicy):
    """(total penetration, n violating pairs, min clearance heavy-heavy, min clearance H) for one molecule."""
    r = _pair_distances(group_pos, pep_pos, box)
    thresh = policy.peptide_threshold(group_heavy, pep_heavy)
    pen = np.clip(thresh - r, 0.0, None)
    hh = group_heavy[:, None] & pep_heavy[None, :]
    min_hh = float(r[hh].min()) if hh.any() else float("inf")
    min_h = float(r[~hh].min()) if (~hh).any() else float("inf")
    return float(pen.sum()), int((pen > 0).sum()), min_hh, min_h


def solvent_solvent_penetration(group_pos, group_heavy, other_pos, other_heavy, box, policy: RepairPolicy):
    """Same for one molecule against a set of other solvent atoms (distinct molecules only)."""
    if other_pos.shape[0] == 0:
        return 0.0, 0, float("inf"), float("inf")
    r = _pair_distances(group_pos, other_pos, box)
    thresh = policy.solvent_threshold(group_heavy, other_heavy)
    pen = np.clip(thresh - r, 0.0, None)
    hh = group_heavy[:, None] & other_heavy[None, :]
    min_hh = float(r[hh].min()) if hh.any() else float("inf")
    min_h = float(r[~hh].min()) if (~hh).any() else float("inf")
    return float(pen.sum()), int((pen > 0).sum()), min_hh, min_h


class _CellList:
    """Cell list over solvent atoms for neighbour queries under the periodic box.

    Cells are >= ``cutoff`` wide along each fractional axis, so every atom within ``cutoff``
    of a query point lies in the query cell or one of its 26 neighbours. Validated against
    the brute-force oracle in the tests."""

    def __init__(self, pos: np.ndarray, box: np.ndarray, cutoff: float):
        self.box = box
        self.inv = np.linalg.inv(box)
        lengths = np.array([box[0, 0], box[1, 1], box[2, 2]])
        self.n = np.maximum(1, np.floor(lengths / cutoff).astype(int))
        self.n = np.minimum(self.n, 64)
        self.frac = (pos @ self.inv) % 1.0
        cells = np.floor(self.frac * self.n).astype(int) % self.n
        keys = (cells[:, 0] * self.n[1] + cells[:, 1]) * self.n[2] + cells[:, 2]
        order = np.argsort(keys, kind="stable")
        self.sorted_idx = order
        self.sorted_keys = keys[order]

    def neighbours(self, points: np.ndarray) -> np.ndarray:
        """Indices of all atoms in the 27 cells around each query point (union, unique)."""
        frac = (points @ self.inv) % 1.0
        cells = np.floor(frac * self.n).astype(int) % self.n
        out = []
        for c in np.unique(cells, axis=0):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        cx = (c[0] + dx) % self.n[0]; cy = (c[1] + dy) % self.n[1]; cz = (c[2] + dz) % self.n[2]
                        key = (cx * self.n[1] + cy) * self.n[2] + cz
                        lo = np.searchsorted(self.sorted_keys, key, side="left")
                        hi = np.searchsorted(self.sorted_keys, key, side="right")
                        if hi > lo:
                            out.append(self.sorted_idx[lo:hi])
        if not out:
            return np.zeros(0, dtype=int)
        return np.unique(np.concatenate(out))


# ---------------------------------------------------------------------------------------------
# Candidate directions
# ---------------------------------------------------------------------------------------------

def _fixed_directions() -> np.ndarray:
    """26 deterministic directions: the axes, face diagonals and body diagonals."""
    dirs = []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                if x == y == z == 0:
                    continue
                v = np.array([x, y, z], dtype=float)
                dirs.append(v / np.linalg.norm(v))
    return np.array(dirs)


_FIXED_DIRECTIONS = _fixed_directions()


# ---------------------------------------------------------------------------------------------
# The repair
# ---------------------------------------------------------------------------------------------

def _validate_inputs(pos, box, pep, groups, heavy):
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("positions must be (n, 3)")
    if not np.all(np.isfinite(pos)):
        raise ValueError("positions contain non-finite values")
    n = pos.shape[0]
    if heavy.shape != (n,):
        raise ValueError("heavy mask must have one entry per atom")
    if pep.size and (pep.min() < 0 or pep.max() >= n):
        raise ValueError("peptide indices out of range")
    seen = np.zeros(n, dtype=bool)
    seen[pep] = True
    for g in groups:
        if g.size == 0 or g.min() < 0 or g.max() >= n:
            raise ValueError("solvent group indices out of range or empty")
        if seen[g].any():
            raise ValueError("solvent groups overlap each other or the peptide")
        seen[g] = True


def repair_solvent_clashes(positions_nm, box_nm, peptide_indices, solvent_groups, heavy_mask, *,
                           policy: Optional[RepairPolicy] = None, seed: int = 0,
                           seed_id: Optional[str] = None, massless_mask=None) -> RepairResult:
    """Move overlapping solvent molecules rigidly until the policy's clearances hold, or say why not.

    ``massless_mask`` (virtual sites) atoms are carried rigidly with their molecule but never
    take part in clearance checks; the caller recomputes them through OpenMM afterwards.
    """
    policy = policy or RepairPolicy()
    rng = np.random.default_rng(int(seed))
    pos = np.array(positions_nm, dtype=float, copy=True)
    pep = np.asarray(peptide_indices, dtype=int)
    groups = [np.asarray(g, dtype=int) for g in solvent_groups]
    heavy = np.asarray(heavy_mask, dtype=bool)
    try:
        box = validate_reduced_box(box_nm)
        _validate_inputs(pos, box, pep, groups, heavy)
        search_cutoff = (max(policy.solvent_heavy_heavy_nm, policy.peptide_heavy_heavy_nm)
                         + policy.max_translation_nm + policy.margin_nm)
        if safe_radius_nm(box) < search_cutoff:
            raise ValueError(f"cell too small or too skewed: safe minimum-image radius {safe_radius_nm(box):.3f} nm "
                             f"< search cutoff {search_cutoff:.3f} nm")
    except ValueError as exc:
        return RepairResult(STATUS_INVALID_INPUT, f"invalid_input:{exc}", pos, seed_id=seed_id, repair_seed=int(seed))
    active = np.ones(pos.shape[0], dtype=bool)          # atoms that take part in clearance checks
    if massless_mask is not None:
        active &= ~np.asarray(massless_mask, dtype=bool)
    pep_act = pep[active[pep]]
    pep_pos, pep_heavy = pos[pep_act], heavy[pep_act]
    group_act = [g[active[g]] for g in groups]
    # which molecule owns each solvent atom, for excluding intramolecular pairs
    owner = np.full(pos.shape[0], -1, dtype=int)
    for gi, g in enumerate(groups):
        owner[g] = gi
    solvent_atoms = np.concatenate(group_act) if group_act else np.zeros(0, dtype=int)
    cutoff = max(policy.solvent_heavy_heavy_nm, policy.peptide_heavy_heavy_nm) + policy.max_translation_nm + policy.margin_nm

    def others_near(gi: int, cells: _CellList, probe: np.ndarray) -> np.ndarray:
        cand = solvent_atoms[cells.neighbours(probe)] if solvent_atoms.size else np.zeros(0, dtype=int)
        return cand[owner[cand] != gi]

    def score_group(gi: int, gpos: np.ndarray, cells: _CellList):
        g = group_act[gi]
        p_pen, p_n, p_hh, p_h = peptide_solvent_penetration(gpos, heavy[g], pep_pos, pep_heavy, box, policy)
        near = others_near(gi, cells, gpos)
        s_pen, s_n, s_hh, s_h = solvent_solvent_penetration(gpos, heavy[g], pos[near], heavy[near], box, policy)
        return p_pen, p_n, s_pen, s_n

    n_attempts = 0
    moved: set = set()
    n_rounds = 0
    reason = "resolved"
    for _round in range(int(policy.max_rounds)):
        n_rounds += 1
        cells = _CellList(pos[solvent_atoms], box, cutoff) if solvent_atoms.size else None
        # cells index into solvent_atoms; wrap so neighbours() returns positions in solvent_atoms order
        progress = False
        any_violation = False
        for gi, g in enumerate(group_act):
            if g.size == 0:
                continue
            p_pen, p_n, s_pen, s_n = score_group(gi, pos[g], cells)
            if p_n == 0 and s_n == 0:
                continue
            any_violation = True
            current = p_pen + s_pen
            # radial escape from the nearest peptide atom, plus the fixed set
            r = _pair_distances(pos[g], pep_pos, box)
            k, j = np.unravel_index(np.argmin(r), r.shape)
            radial = min_image(pos[g][k] - pep_pos[j], box)
            if np.linalg.norm(radial) < 1e-6:
                radial = rng.normal(size=3)
            radial = radial / np.linalg.norm(radial)
            directions = np.vstack([radial[None, :], _FIXED_DIRECTIONS])
            needed = max(policy.peptide_heavy_heavy_nm - float(r.min()), 0.0) + policy.margin_nm
            steps = np.linspace(needed, policy.max_translation_nm, int(policy.n_translation_steps))
            best = None
            for di, direction in enumerate(directions):
                for si, dist in enumerate(steps):
                    n_attempts += 1
                    trial = pos[g] + direction * dist
                    tp, tpn, ts, tsn = score_group(gi, trial, cells)
                    if tsn > 0 and s_n == 0:
                        continue                                  # never create a new solvent-solvent collision
                    total = tp + ts
                    key = (total, di, si)
                    if total < current - 1e-12 and (best is None or key < best[0]):
                        best = (key, trial, tpn, tsn)
            if best is None and policy.max_cluster_size > 1:
                # bounded local cluster move: carry the nearest solvent molecules along
                near = others_near(gi, cells, pos[g])
                near_groups = sorted({int(owner[a]) for a in near})[: policy.max_cluster_size - 1]
                if near_groups:
                    cluster = [gi] + near_groups
                    for di, direction in enumerate(directions):
                        for si, dist in enumerate(steps):
                            n_attempts += 1
                            trial_pos = pos.copy()
                            for cg in cluster:
                                trial_pos[groups[cg]] += direction * dist
                            tot = 0.0; ok = True
                            for cg in cluster:
                                gp = group_act[cg]
                                pp, pn, _, _ = peptide_solvent_penetration(trial_pos[gp], heavy[gp], pep_pos, pep_heavy, box, policy)
                                near_c = others_near(cg, cells, trial_pos[gp])
                                near_c = near_c[np.isin(owner[near_c], cluster, invert=True)]
                                sp, sn, _, _ = solvent_solvent_penetration(trial_pos[gp], heavy[gp], trial_pos[near_c], heavy[near_c], box, policy)
                                if sn > 0:
                                    ok = False
                                    break
                                tot += pp + sp
                            if ok and tot < current - 1e-12:
                                key = (tot, di, si)
                                if best is None or key < best[0]:
                                    best = (key, (cluster, direction * dist), None, None)
                    if best is not None and isinstance(best[1], tuple):
                        cluster, shift = best[1]
                        for cg in cluster:
                            pos[groups[cg]] += shift
                            moved.add(int(cg))
                        progress = True
                        continue
            if best is None:
                continue
            _, trial, _, _ = best
            shift = trial[0] - pos[g][0]
            pos[groups[gi]] += shift
            moved.add(int(gi))
            progress = True
        if not any_violation:
            break
        if not progress:
            reason = "exhausted:no_improving_move"
            break
    else:
        reason = "exhausted:max_rounds"

    # ---- final independent validation ------------------------------------------------------
    result = _final_validation(pos, box, pep_act, group_act, heavy, owner, solvent_atoms, policy)
    status = STATUS_RESOLVED if result["n_pep"] == 0 and result["n_sol"] == 0 else STATUS_UNRESOLVED
    if status == STATUS_RESOLVED:
        reason = "resolved"
    elif reason == "resolved":
        reason = "unresolved:validation_failed"
    return RepairResult(status, reason, pos, n_groups_moved=len(moved), n_attempts=n_attempts, n_rounds=n_rounds,
                        n_peptide_solvent_clashes=result["n_pep"], n_solvent_solvent_clashes=result["n_sol"],
                        minimum_clearance_by_pair_class=result["min"], moved_groups=sorted(moved),
                        seed_id=seed_id, repair_seed=int(seed), repair_policy_version=policy.version)


def _final_validation(pos, box, pep_act, group_act, heavy, owner, solvent_atoms, policy: RepairPolicy) -> dict:
    """Recount every peptide-solvent clash and every solvent-solvent emergency collision.

    Peptide-solvent is checked for every molecule. Solvent-solvent is checked for every
    molecule against its cell-list neighbourhood (the cutoff exceeds every threshold), so
    the count is exact, not inferred from the loop having stopped."""
    n_pep = 0; n_sol = 0
    mins = {"peptide_heavy_heavy": float("inf"), "peptide_hydrogen": float("inf"),
            "solvent_heavy_heavy": float("inf"), "solvent_hydrogen": float("inf")}
    pep_pos, pep_heavy = pos[pep_act], heavy[pep_act]
    cutoff = max(policy.solvent_heavy_heavy_nm, policy.peptide_heavy_heavy_nm) + 0.05
    cells = _CellList(pos[solvent_atoms], box, cutoff) if solvent_atoms.size else None
    for gi, g in enumerate(group_act):
        if g.size == 0:
            continue
        _, pn, phh, ph = peptide_solvent_penetration(pos[g], heavy[g], pep_pos, pep_heavy, box, policy)
        n_pep += pn
        mins["peptide_heavy_heavy"] = min(mins["peptide_heavy_heavy"], phh)
        mins["peptide_hydrogen"] = min(mins["peptide_hydrogen"], ph)
        if cells is not None:
            near = solvent_atoms[cells.neighbours(pos[g])]
            near = near[owner[near] > gi]        # each distinct pair once
            _, sn, shh, sh = solvent_solvent_penetration(pos[g], heavy[g], pos[near], heavy[near], box, policy)
            n_sol += sn
            mins["solvent_heavy_heavy"] = min(mins["solvent_heavy_heavy"], shh)
            mins["solvent_hydrogen"] = min(mins["solvent_hydrogen"], sh)
    return {"n_pep": int(n_pep), "n_sol": int(n_sol), "min": mins}


def brute_force_clashes(positions_nm, box_nm, peptide_indices, solvent_groups, heavy_mask,
                        policy: Optional[RepairPolicy] = None) -> Tuple[int, int]:
    """Test oracle: (peptide-solvent violations, distinct-molecule solvent-solvent violations), all pairs."""
    policy = policy or RepairPolicy()
    pos = np.asarray(positions_nm, dtype=float); box = validate_reduced_box(box_nm)
    pep = np.asarray(peptide_indices, dtype=int); heavy = np.asarray(heavy_mask, dtype=bool)
    groups = [np.asarray(g, dtype=int) for g in solvent_groups]
    n_pep = 0
    for g in groups:
        _, pn, _, _ = peptide_solvent_penetration(pos[g], heavy[g], pos[pep], heavy[pep], box, policy)
        n_pep += pn
    n_sol = 0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            gi, gj = groups[i], groups[j]
            _, sn, _, _ = solvent_solvent_penetration(pos[gi], heavy[gi], pos[gj], heavy[gj], box, policy)
            n_sol += sn
    return int(n_pep), int(n_sol)
