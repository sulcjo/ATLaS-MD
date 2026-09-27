"""Continue frozen-final extension rounds from the previous segments' end states.

A frozen-final extension (``final_extension_NNN``) runs the SAME window set as the
final phase, so its windows already have equilibrated chains: the newest earlier
segment's end state for each state. Re-grafting seeds and US-pulling every window
again cost 51 % of the chignolin_9 extension job's wall (job 2680578, ~2 h of 3 h 51 min)
and threw those chains away.

Two seed sources, both resolved per state through the parent's OWN
``epoch_window_map.csv`` (never window-index identity):

* ``final_window_states/`` -- the exact State export top-ups use (positions,
  velocities, box, recorded CVs and restraint). Exact continuation.
* ``final_pdbs/`` -- every segment's per-window full-system PDB (positions rounded
  to 1e-3 A, box from CRYST1, no velocities). The fallback for segments that never
  exported States (every top-ups-off segment); the caller re-applies constraints
  and draws fresh Maxwell-Boltzmann velocities.

For each state the NEWEST parent that holds it wins, whichever kind it holds -- an
older export must never outrank a newer PDB-only parent, or the chain would restart
from an earlier point.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .topup_seeding import (
    DIR_NAME,
    SeedState,
    load_seed_index,
    load_seed_states,
    state_id_of_window_from_epoch_map,
)

logger = logging.getLogger(__name__)

PDB_DIR = "final_pdbs"
_PDB_RE = re.compile(r"^replica_(\d+)_window_(\d+)\.pdb$")
EXTENSION_RE = re.compile(r"^final_extension_(\d+)$")


@dataclass(frozen=True)
class PdbSeed:
    """One window's end positions from a parent's ``final_pdbs/``.

    Restraint fields are in the parent checkpoint manifest's units (A / kcal), so the
    caller compares them against its own ``centers_a``/``k_list`` -- not nm/kJ.
    Attribute names match :class:`SeedState` so ``assert_seed_restraint_matches``
    works on either.
    """
    path: Path
    source: str
    state_id: int
    primary_center: Optional[float]
    primary_k: Optional[float]
    secondary_center: Optional[float]
    secondary_k: Optional[float]


def _final_phase_parent_dirs(final_dir: Path) -> List[Path]:
    """The final phase's own segments, oldest first.

    A flat final phase is one parent; a scheduled one is ``baseline`` then its
    ``topup_*`` segments ordered by when their end states were written (the
    ``topup_NNN_<steps>`` suffix is not chronological -- see
    ``topup_parent_dirs_by_creation_order``).
    """
    if (final_dir / PDB_DIR).is_dir() or (final_dir / DIR_NAME).is_dir():
        return [final_dir]
    segments = [d for d in [final_dir / "baseline", *final_dir.glob("topup_*")] if d.is_dir()]

    def _written_ns(d: Path) -> float:
        # export_seq (a time_ns stamp inside the export) survives a copy; mtime does
        # not, so it is only the fallback for a segment that never exported States.
        try:
            record = json.loads((d / DIR_NAME / "index.json").read_text())
            seqs = [v.get("export_seq") for v in record.values()
                    if isinstance(v, dict) and isinstance(v.get("export_seq"), (int, float))]
            if seqs:
                return float(max(seqs))
        except Exception:
            pass
        try:
            return float((d / PDB_DIR).stat().st_mtime_ns)
        except OSError:
            return -1.0

    baseline = [d for d in segments if d.name == "baseline"]
    topups = sorted((d for d in segments if d.name != "baseline"), key=_written_ns)
    return baseline + topups


def extension_parent_dirs(adaptive_dir, ext_index: int) -> List[Path]:
    """Parents of extension round ``ext_index + 1``, oldest first.

    The final phase's segments, then every earlier ``final_extension_NNN`` in round
    order (round number, not name or mtime, is the chronology here).
    """
    adaptive_dir = Path(adaptive_dir)
    parents = _final_phase_parent_dirs(adaptive_dir / "final")
    for k in range(1, int(ext_index) + 1):
        d = adaptive_dir / f"final_extension_{k:03d}"
        if d.is_dir():
            parents.append(d)
    return parents


def _parent_restraints(parent: Path) -> Optional[Dict[str, List[Optional[float]]]]:
    """Per-window restraints the parent actually ran, in its own post-drop window order."""
    path = parent / "checkpoints" / "production_checkpoint_manifest.json"
    try:
        m = json.loads(path.read_text())
        centers = [float(x) for x in m["windows_A"]]
        ks = [float(x) for x in m["window_k_kcal_mol_A2"]]
    except Exception as exc:
        logger.warning("Extension seeding: %s has no readable restraint table (%s); skipping its PDBs", path, exc)
        return None
    n = len(centers)

    def _opt(key: str) -> List[Optional[float]]:
        vals = m.get(key)
        if not isinstance(vals, list) or len(vals) != n:
            return [None] * n
        return [None if v is None else float(v) for v in vals]

    return {
        "primary_center": centers,
        "primary_k": ks,
        "secondary_center": _opt("secondary_cv_centers"),
        "secondary_k": _opt("secondary_cv_k_kcal_mol"),
    }


def pdb_seeds_of_parent(parent) -> Dict[int, PdbSeed]:
    """state_id -> PdbSeed for every ``final_pdbs`` file the parent's window map can place."""
    parent = Path(parent)
    pdb_dir = parent / PDB_DIR
    if not pdb_dir.is_dir() or not (parent / "epoch_window_map.csv").exists():
        return {}
    restraints = _parent_restraints(parent)
    if restraints is None:
        return {}
    state_of_window = state_id_of_window_from_epoch_map(parent)
    n = len(restraints["primary_center"])
    out: Dict[int, PdbSeed] = {}
    for path in sorted(pdb_dir.iterdir()):
        match = _PDB_RE.match(path.name)
        if match is None:
            continue
        w = int(match.group(2))
        sid = state_of_window.get(w)
        if sid is None or w >= n:
            logger.warning("Extension seeding: %s has no window-map row / restraint for window %d; skipped", path, w)
            continue
        out[int(sid)] = PdbSeed(
            path=path, source=str(parent), state_id=int(sid),
            primary_center=restraints["primary_center"][w],
            primary_k=restraints["primary_k"][w],
            secondary_center=restraints["secondary_center"][w],
            secondary_k=restraints["secondary_k"][w],
        )
    return out


def resolve_seed_sources(parent_dirs: Iterable, state_ids: Iterable[int]) -> Dict[int, Tuple[str, Path]]:
    """state_id -> (``"export"``|``"pdb"``, parent) for the NEWEST parent holding it.

    Within one parent an export beats that parent's own PDB (same moment, exact
    velocities). Across parents, recency wins regardless of kind.
    """
    wanted = {int(s) for s in state_ids}
    out: Dict[int, Tuple[str, Path]] = {}
    for parent in parent_dirs:                      # oldest first; later overrides
        parent = Path(parent)
        exported = set(load_seed_index([parent]))
        for sid in wanted & set(pdb_seeds_of_parent(parent)):
            out[sid] = ("pdb", parent)
        for sid in wanted & exported:
            out[sid] = ("export", parent)
    return out


def load_extension_seeds(parent_dirs: Sequence, state_ids: Iterable[int]) -> Dict[int, Any]:
    """state_id -> SeedState (export) or PdbSeed, per :func:`resolve_seed_sources`.

    A state absent from the result has no seed; the caller decides what that means.
    """
    state_ids = [int(s) for s in state_ids]
    sources = resolve_seed_sources(parent_dirs, state_ids)
    out: Dict[int, Any] = {}
    by_parent: Dict[Tuple[str, Path], List[int]] = {}
    for sid, key in sources.items():
        by_parent.setdefault(key, []).append(sid)
    for (kind, parent), sids in by_parent.items():
        if kind == "export":
            out.update(load_seed_states([parent], sids))
        else:
            pdbs = pdb_seeds_of_parent(parent)
            out.update({sid: pdbs[sid] for sid in sids})
    return out


def read_pdb_seed(seed: PdbSeed):
    """(positions, box) from the seed's PDB; box from CRYST1."""
    from openmm import app  # noqa: PLC0415
    pdb = app.PDBFile(str(seed.path))
    return pdb.getPositions(), pdb.getTopology().getPeriodicBoxVectors()


def is_state_seed(seed: Any) -> bool:
    return isinstance(seed, SeedState)
