"""
adaptive_production
===================

Epoch-based adaptive production support for GAREUS.

The first implementation intentionally uses the existing ``production.run_gareus``
engine as the MD worker.  Adaptive production is therefore expressed as a series
of fixed-bias production epochs:

    epoch N: run a fixed explicit window table
             collect samples/exchanges/diagnostics
             update a persistent state registry
             write the explicit table for epoch N+1

This keeps every individual epoch simple and auditable.  Windows are added or
retired only between epochs, never while an OpenMM Context is mid-step.  Old
states are never overwritten: if the center or force constant changes, a new
state ID is created.  The final phase freezes the active window set and runs a
clean production segment suitable for downstream MBAR/FES analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import copy
import csv
import json
import logging
import math
import re
import shutil
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .checkpoints import production_checkpoint_available
from .io import write_json, read_json_file, _json_ready
from .lifecycle import _graceful_shutdown

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class WindowState:
    """One persistent thermodynamic umbrella/REUS state."""

    state_id: int
    primary_center: float
    primary_k: float
    secondary_center: Optional[float] = None
    secondary_k: Optional[float] = None
    gamd_sigma0p: Optional[float] = None
    gamd_sigma0d: Optional[float] = None
    active: bool = True
    created_epoch: int = 0
    retired_epoch: Optional[int] = None
    parent_state_id: Optional[int] = None
    source: str = "initial"
    reason: str = ""
    usable_for_mbar: bool = True
    burnin_steps: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mark_retired(self, epoch: int) -> None:
        if self.retired_epoch is not None:
            logger.warning("State %s already retired at epoch %s", self.state_id, self.retired_epoch)
            return
        self.active = False
        self.retired_epoch = int(epoch)

    def to_dict(self) -> Dict[str, Any]:
        return _json_ready(asdict(self))

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "WindowState":
        data = dict(row)
        data["state_id"] = int(data.get("state_id", 0))
        data["primary_center"] = float(data.get("primary_center", data.get("distance_center_A", 0.0)))
        data["primary_k"] = float(data.get("primary_k", data.get("distance_k_kcal_mol_A2", 0.0)))
        for key in ("secondary_center", "secondary_k", "gamd_sigma0p", "gamd_sigma0d"):
            val = data.get(key)
            if val in ("", "None", None):
                data[key] = None
            else:
                data[key] = float(val)
        for key in ("created_epoch", "burnin_steps"):
            data[key] = int(data.get(key, 0) or 0)
        if data.get("retired_epoch") in ("", "None", None):
            data["retired_epoch"] = None
        else:
            data["retired_epoch"] = int(data.get("retired_epoch"))
        if data.get("parent_state_id") in ("", "None", None):
            data["parent_state_id"] = None
        else:
            data["parent_state_id"] = int(data.get("parent_state_id"))
        data["active"] = bool(data.get("active", True))
        data["usable_for_mbar"] = bool(data.get("usable_for_mbar", True))
        if not isinstance(data.get("metadata"), dict):
            data["metadata"] = {}
        keep = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in keep})


@dataclass
class LifecycleEvent:
    """Auditable registry event."""

    epoch: int
    event: str
    state_id: int
    parent_state_id: Optional[int] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_ready(asdict(self))

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "LifecycleEvent":
        data = dict(row)
        data["epoch"] = int(data.get("epoch", 0) or 0)
        data["state_id"] = int(data.get("state_id", -1))
        if data.get("parent_state_id") in ("", "None", None):
            data["parent_state_id"] = None
        else:
            data["parent_state_id"] = int(data.get("parent_state_id"))
        if not isinstance(data.get("metadata"), dict):
            data["metadata"] = {}
        keep = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in keep})


@dataclass
class StateDiagnostics:
    state_id: int
    epoch_window: int
    sample_count: int = 0
    cv_min: Optional[float] = None
    cv_max: Optional[float] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    secondary_min: Optional[float] = None
    secondary_max: Optional[float] = None
    secondary_mean: Optional[float] = None
    secondary_std: Optional[float] = None
    gamd_boost_sd_kcal_mol: Optional[float] = None
    gamd_ess_fraction_proxy: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass
class EdgeDiagnostics:
    state_i: int
    state_j: int
    window_i: int
    window_j: int
    edge_type: str = "geometry"
    normalized_distance: Optional[float] = None
    overlap: Optional[float] = None
    exchange_attempts: int = 0
    exchange_accepted: int = 0
    exchange_acceptance: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass
class AdaptiveDecisionPolicy:
    """Conservative first-pass decision policy.

    The first useful adaptive-production version should add windows at weak
    edges.  Retirement is available but off by default because it is much easier
    to add useful states than to prove a bridge state is disposable.  Naturally,
    the machine would prefer to save GPU time by burning correctness first.
    """

    target_overlap: float = 0.25
    min_exchange_acceptance: float = 0.08
    min_samples_for_add: int = 50
    min_samples_for_retire: int = 200
    max_new_windows_per_epoch: int = 4
    retire_converged: bool = False
    max_gamd_boost_sd_kcal_mol: float = 6.0
    duplicate_primary_tol: float = 1.0e-4
    duplicate_secondary_tol: float = 1.0e-4
    final_connectivity_required: bool = True
    final_min_samples_per_state: int = 100
    quality_min_primary_coverage_fraction: float = 0.25
    quality_hard_fail: bool = False
    final_quality_extension_rounds: int = 0
    final_quality_extension_steps: int = 0
    propagate_seed_bank: bool = True
    seed_bank_max_per_state: int = 1
    allocation_scheduler: bool = True
    epoch_step_budget: int = 0
    min_state_steps: int = 0
    max_state_steps: int = 0
    new_state_steps: int = 0
    frontier_bonus: float = 1.5
    weak_edge_bonus: float = 3.0
    low_sample_bonus: float = 2.0
    high_boost_bonus: float = 1.0
    final_allocation_scheduler: bool = True
    final_step_budget: int = 0
    state_aware_seed_filtering: bool = True
    scheduled_final_segments: bool = True
    convergence_min_samples_per_state: int = 50
    convergence_max_weak_edges: int = 0
    convergence_allow_extend_actions: bool = True
    require_convergence_before_final: bool = False
    total_md_pool_ns: float = 0.0
    final_pool_fraction: float = 0.50
    min_final_pool_ns: float = 0.0
    pool_hard_stop: bool = True
    context_reuse: bool = False
    context_reuse_require: bool = False
    context_reuse_mode: str = "off"


class WindowStateRegistry:
    """Persistent registry of all active and dormant adaptive-production states."""

    def __init__(self) -> None:
        self._states: Dict[int, WindowState] = {}
        self._next_state_id: int = 0
        self.lifecycle: List[LifecycleEvent] = []

    # ---- state management -------------------------------------------------

    def add_state(
        self,
        primary_center: float,
        primary_k: float,
        secondary_center: Optional[float] = None,
        secondary_k: Optional[float] = None,
        gamd_sigma0p: Optional[float] = None,
        gamd_sigma0d: Optional[float] = None,
        parent_state_id: Optional[int] = None,
        epoch: int = 0,
        source: str = "adaptive",
        reason: str = "added",
        usable_for_mbar: bool = True,
        burnin_steps: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WindowState:
        state_id = self._next_state_id
        self._next_state_id += 1
        state = WindowState(
            state_id=state_id,
            primary_center=float(primary_center),
            primary_k=float(primary_k),
            secondary_center=None if secondary_center is None else float(secondary_center),
            secondary_k=None if secondary_k is None else float(secondary_k),
            gamd_sigma0p=None if gamd_sigma0p is None else float(gamd_sigma0p),
            gamd_sigma0d=None if gamd_sigma0d is None else float(gamd_sigma0d),
            active=True,
            created_epoch=int(epoch),
            parent_state_id=parent_state_id,
            source=str(source),
            reason=str(reason),
            usable_for_mbar=bool(usable_for_mbar),
            burnin_steps=int(burnin_steps or 0),
            metadata=dict(metadata or {}),
        )
        self._states[state_id] = state
        self.lifecycle.append(
            LifecycleEvent(
                epoch=int(epoch),
                event="add",
                state_id=state_id,
                parent_state_id=parent_state_id,
                reason=str(reason),
                metadata={"source": str(source)},
            )
        )
        return state

    def retire_state(self, state_id: int, epoch: int, reason: str = "retired") -> None:
        state = self._states.get(int(state_id))
        if state is None:
            raise KeyError(f"unknown state_id {state_id}")
        state.mark_retired(int(epoch))
        self.lifecycle.append(
            LifecycleEvent(epoch=int(epoch), event="retire", state_id=int(state_id), reason=str(reason))
        )

    def record_extend(self, state_id: int, epoch: int, reason: str = "extended") -> None:
        if int(state_id) not in self._states:
            raise KeyError(f"unknown state_id {state_id}")
        self.lifecycle.append(
            LifecycleEvent(epoch=int(epoch), event="extend", state_id=int(state_id), reason=str(reason))
        )

    def get_state(self, state_id: int) -> Optional[WindowState]:
        return self._states.get(int(state_id))

    def all_states(self) -> List[WindowState]:
        return [self._states[k] for k in sorted(self._states)]

    def active_states(self) -> List[WindowState]:
        return [s for s in self.all_states() if s.active]

    def dormant_states(self) -> List[WindowState]:
        return [s for s in self.all_states() if not s.active]

    def active_state_ids(self) -> List[int]:
        return [int(s.state_id) for s in self.active_states()]

    def next_state_id(self) -> int:
        return int(self._next_state_id)

    def has_near_duplicate(self, primary: float, secondary: Optional[float], policy: AdaptiveDecisionPolicy) -> bool:
        for state in self.all_states():
            if abs(float(state.primary_center) - float(primary)) > float(policy.duplicate_primary_tol):
                continue
            if state.secondary_center is None and secondary is None:
                return True
            if state.secondary_center is not None and secondary is not None:
                if abs(float(state.secondary_center) - float(secondary)) <= float(policy.duplicate_secondary_tol):
                    return True
        return False

    # ---- persistence ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "adaptive_production_registry_v1",
            "next_state_id": int(self._next_state_id),
            "states": [s.to_dict() for s in self.all_states()],
            "lifecycle": [e.to_dict() for e in self.lifecycle],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "WindowStateRegistry":
        reg = cls()
        states = payload.get("states", []) if isinstance(payload, dict) else []
        for row in states:
            state = WindowState.from_dict(row)
            reg._states[int(state.state_id)] = state
            reg._next_state_id = max(reg._next_state_id, int(state.state_id) + 1)
        next_id = payload.get("next_state_id") if isinstance(payload, dict) else None
        if next_id is not None:
            reg._next_state_id = max(reg._next_state_id, int(next_id))
        for row in payload.get("lifecycle", []) if isinstance(payload, dict) else []:
            reg.lifecycle.append(LifecycleEvent.from_dict(row))
        return reg

    def save_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, self.to_dict())
        return path

    @classmethod
    def load_json(cls, path: Path) -> "WindowStateRegistry":
        return cls.from_dict(read_json_file(Path(path), {}))

    def write_state_csv(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [s.to_dict() for s in self.all_states()]
        fieldnames = [
            "state_id", "active", "created_epoch", "retired_epoch", "parent_state_id",
            "primary_center", "primary_k", "secondary_center", "secondary_k",
            "gamd_sigma0p", "gamd_sigma0d", "source", "reason", "usable_for_mbar", "burnin_steps",
        ]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def write_lifecycle_jsonl(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for event in self.lifecycle:
                handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        return path

    def save(self, directory: Path) -> Dict[str, str]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return {
            "registry_json": str(self.save_json(directory / "state_registry.json")),
            "registry_csv": str(self.write_state_csv(directory / "state_registry.csv")),
            "lifecycle_jsonl": str(self.write_lifecycle_jsonl(directory / "lifecycle.jsonl")),
        }

    @classmethod
    def load(cls, directory: Path) -> "WindowStateRegistry":
        return cls.load_json(Path(directory) / "state_registry.json")

    # ---- window table writers --------------------------------------------

    def write_active_window_csv(self, path: Path, map_path: Optional[Path] = None) -> Path:
        """Write active states as a GAREUS-compatible explicit window table.

        The generated CSV can be passed to ``--windows-2d-csv``.  It includes
        compatibility columns consumed by ``windows.load_explicit_2d_window_csv``
        plus ``state_id`` and lifecycle provenance columns for adaptive analysis.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        active = self.active_states()
        if not active:
            raise RuntimeError("cannot write active window CSV: registry has no active states")
        has_secondary = any(s.secondary_center is not None for s in active)
        fieldnames = [
            "epoch_window", "state_id", "primary_cv_center", "primary_cv_k_kcal",
            "distance_center_A", "distance_k_kcal_mol_A2", "secondary_cv_center",
            "secondary_cv_k_kcal_mol", "window_type", "patch_lifecycle", "parent_state_id",
            "created_epoch", "source", "reason", "usable_for_mbar", "burnin_steps",
        ]
        rows = []
        for i, state in enumerate(active):
            row = {
                "epoch_window": int(i),
                "state_id": int(state.state_id),
                "primary_cv_center": float(state.primary_center),
                "primary_cv_k_kcal": float(state.primary_k),
                "distance_center_A": float(state.primary_center),
                "distance_k_kcal_mol_A2": float(state.primary_k),
                "secondary_cv_center": "" if state.secondary_center is None else float(state.secondary_center),
                "secondary_cv_k_kcal_mol": "" if state.secondary_k is None else float(state.secondary_k),
                "window_type": "adaptive_production_state",
                "patch_lifecycle": "adaptive_production_active",
                "parent_state_id": "" if state.parent_state_id is None else int(state.parent_state_id),
                "created_epoch": int(state.created_epoch),
                "source": str(state.source),
                "reason": str(state.reason),
                "usable_for_mbar": int(bool(state.usable_for_mbar)),
                "burnin_steps": int(state.burnin_steps),
            }
            if has_secondary and state.secondary_center is None:
                raise RuntimeError("active registry mixes 1D and 2D states; cannot write one explicit 2D table")
            rows.append(row)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        if map_path is not None:
            self.write_epoch_window_map(map_path, active=active)
        return path

    def write_epoch_window_map(self, path: Path, active: Optional[Sequence[WindowState]] = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, state in enumerate(active if active is not None else self.active_states()):
            rows.append({
                "epoch_window": int(i),
                "state_id": int(state.state_id),
                "primary_center": float(state.primary_center),
                "primary_k": float(state.primary_k),
                "secondary_center": "" if state.secondary_center is None else float(state.secondary_center),
                "secondary_k": "" if state.secondary_k is None else float(state.secondary_k),
            })
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["epoch_window", "state_id"])
            writer.writeheader()
            writer.writerows(rows)
        return path


# ---------------------------------------------------------------------------
# Registry seeding from existing GAREUS tables
# ---------------------------------------------------------------------------


def _csv_first(row: Dict[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in row and str(row.get(name, "")).strip() != "":
            return row.get(name)
    return default


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, "", "None", "nan"):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def registry_from_window_csv(path: Path, epoch: int = 0, source: str = "window_csv") -> WindowStateRegistry:
    """Seed a registry from ``umbrella_windows.csv`` or ``umbrella_explicit_windows.csv``."""
    path = Path(path)
    rows: List[Dict[str, str]] = []
    with path.open(newline="") as handle:
        rows = [dict(r) for r in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"cannot seed adaptive-production registry from empty CSV: {path}")
    reg = WindowStateRegistry()
    for idx, row in enumerate(rows):
        primary = _float_or_none(_csv_first(row, ["primary_cv_center", "primary_center", "distance_center_A", "center_A", "r0_A"]))
        k = _float_or_none(_csv_first(row, ["primary_cv_k_kcal", "primary_k", "distance_k_kcal_mol_A2", "k_kcal_mol_A2", "k_kcal"]))
        if primary is None:
            raise ValueError(f"row {idx + 2} in {path} has no primary center")
        if k is None:
            k = 0.0
        secondary = _float_or_none(_csv_first(row, ["secondary_cv_center", "secondary_center", "ss0"]))
        secondary_k = _float_or_none(_csv_first(row, ["secondary_cv_k_kcal_mol", "secondary_k", "ss_k"]))
        state_id_raw = _csv_first(row, ["state_id"], None)
        state = reg.add_state(
            primary_center=primary,
            primary_k=k,
            secondary_center=secondary,
            secondary_k=secondary_k,
            epoch=epoch,
            source=source,
            reason=f"seeded from {path.name}",
            metadata={"source_csv": str(path), "source_row": idx + 2},
        )
        if state_id_raw not in (None, ""):
            # Preserve source state IDs when reading our own active-window CSVs.
            old_id = state.state_id
            new_id = int(float(state_id_raw))
            del reg._states[old_id]
            state.state_id = new_id
            reg._states[new_id] = state
            reg._next_state_id = max(reg._next_state_id, new_id + 1)
            reg.lifecycle[-1].state_id = new_id
    return reg


def find_best_window_table(run_dir: Path) -> Optional[Path]:
    run_dir = Path(run_dir)
    for name in ("umbrella_explicit_windows.csv", "umbrella_windows.csv"):
        path = run_dir / name
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


# ---------------------------------------------------------------------------
# Epoch-to-epoch seed-bank propagation
# ---------------------------------------------------------------------------


_FINAL_PDB_WINDOW_RE = re.compile(r".*_window_(\d+)\.pdb$")


def _window_index_from_final_pdb(path: Path) -> Optional[int]:
    match = _FINAL_PDB_WINDOW_RE.match(Path(path).name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def write_epoch_seed_bank(
    source_run_dir: Path,
    seed_bank_dir: Path,
    registry: WindowStateRegistry,
    *,
    source_label: str,
    max_per_state: int = 1,
) -> Dict[str, Any]:
    """Create a GENPEPT-compatible seed library from one completed epoch/final run.

    ``production.run_gareus`` writes one final PDB per replica under
    ``final_pdbs/replica_XXX_window_YYY.pdb``.  This helper maps the local
    epoch/final window index back to the persistent adaptive ``state_id`` via
    ``epoch_window_map.csv``, copies a limited number of final PDBs per state
    into ``seed_bank_dir/pdbs``, and writes ``final_survivor_seeds.csv``.

    The existing seeding machinery already understands a directory containing
    ``final_survivor_seeds.csv`` with a ``survivor_pdb_path`` column, so the
    next epoch can consume this seed bank simply by setting
    ``args.seed_conformers_dir`` to the returned directory.  This avoids the
    painfully wasteful pattern where every adaptive epoch forgets what the last
    one just learned.  Computers are good at amnesia; we do not need to help.
    """
    source_run_dir = Path(source_run_dir)
    seed_bank_dir = Path(seed_bank_dir)
    pdb_src_dir = source_run_dir / "final_pdbs"
    pdb_dst_dir = seed_bank_dir / "pdbs"
    seed_bank_dir.mkdir(parents=True, exist_ok=True)
    pdb_dst_dir.mkdir(parents=True, exist_ok=True)
    window_map = _load_epoch_window_map(source_run_dir, registry)
    rows: List[Dict[str, Any]] = []
    per_state_count: Dict[int, int] = {}
    copied = 0
    skipped = 0
    if not pdb_src_dir.exists():
        payload = {
            "schema_version": "adaptive_seed_bank_v1",
            "status": "missing_final_pdbs",
            "source_run_dir": str(source_run_dir),
            "seed_bank_dir": str(seed_bank_dir),
            "rows": [],
            "warnings": [f"{pdb_src_dir} not found"],
        }
        write_json(seed_bank_dir / "seed_bank_report.json", payload)
        return payload

    for src in sorted(pdb_src_dir.glob("*.pdb")):
        local_window = _window_index_from_final_pdb(src)
        if local_window is None:
            skipped += 1
            continue
        state_id = int(window_map.get(int(local_window), int(local_window)))
        if state_id not in registry.active_state_ids() and registry.get_state(state_id) is None:
            skipped += 1
            continue
        if max_per_state > 0 and per_state_count.get(state_id, 0) >= int(max_per_state):
            continue
        idx = per_state_count.get(state_id, 0)
        dst_name = f"{source_label}_state_{state_id:04d}_seed_{idx:02d}.pdb"
        dst = pdb_dst_dir / dst_name
        try:
            shutil.copy2(src, dst)
        except Exception:
            skipped += 1
            continue
        per_state_count[state_id] = idx + 1
        copied += 1
        state = registry.get_state(state_id)
        rows.append({
            "seed_name": f"{source_label}_state_{state_id:04d}_seed_{idx:02d}",
            "survivor_pdb_path": str(Path("pdbs") / dst_name),
            "source_run_dir": str(source_run_dir),
            "source_pdb_path": str(src),
            "source_label": str(source_label),
            "source_state_id": int(state_id),
            "source_epoch_window": int(local_window),
            "primary_cv_value": "" if state is None else float(state.primary_center),
            "secondary_cv_value": "" if state is None or state.secondary_center is None else float(state.secondary_center),
        })

    csv_path = seed_bank_dir / "final_survivor_seeds.csv"
    fieldnames = [
        "seed_name", "survivor_pdb_path", "source_run_dir", "source_pdb_path",
        "source_label", "source_state_id", "source_epoch_window",
        "primary_cv_value", "secondary_cv_value",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": "adaptive_seed_bank_v1",
        "status": "ok" if rows else "empty",
        "source_run_dir": str(source_run_dir),
        "seed_bank_dir": str(seed_bank_dir),
        "final_survivor_seeds_csv": str(csv_path),
        "copied_pdbs": int(copied),
        "skipped_pdbs": int(skipped),
        "max_per_state": int(max_per_state),
        "states_with_seeds": sorted(int(k) for k in per_state_count),
        "rows": rows,
    }
    write_json(seed_bank_dir / "seed_bank_report.json", payload)
    if rows:
        payload["state_aware_seed_assignments"] = select_state_aware_seeds_for_targets(seed_bank_dir, registry)
        write_json(seed_bank_dir / "seed_bank_report.json", payload)
    lines = [
        "# Adaptive seed bank",
        "",
        f"Status: **{payload['status']}**",
        f"Source run: `{source_run_dir}`",
        f"Seed CSV: `{csv_path}`",
        f"Copied PDBs: **{copied}**; skipped: **{skipped}**",
        "",
    ]
    if rows:
        lines.append("## Seeds")
        lines.append("| state | seed | pdb |")
        lines.append("|---:|---|---|")
        for row in rows[:80]:
            lines.append(f"| {row['source_state_id']} | {row['seed_name']} | `{row['survivor_pdb_path']}` |")
        lines.append("")
    (seed_bank_dir / "seed_bank_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def write_seed_bank_from_run_dirs(
    source_run_dirs: Sequence[Path],
    seed_bank_dir: Path,
    registry: WindowStateRegistry,
    *,
    source_label: str,
    max_per_state: int = 1,
) -> Dict[str, Any]:
    """Build one seed bank from several scheduled epoch segment directories."""
    seed_bank_dir = Path(seed_bank_dir)
    pdb_dst_dir = seed_bank_dir / "pdbs"
    seed_bank_dir.mkdir(parents=True, exist_ok=True)
    pdb_dst_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    per_state_count: Dict[int, int] = {}
    copied = 0
    skipped = 0
    for source_run_dir in [Path(p) for p in source_run_dirs]:
        window_map = _load_epoch_window_map(source_run_dir, registry)
        for src in sorted((source_run_dir / "final_pdbs").glob("*.pdb")):
            local_window = _window_index_from_final_pdb(src)
            if local_window is None:
                skipped += 1
                continue
            state_id = int(window_map.get(int(local_window), int(local_window)))
            if registry.get_state(state_id) is None:
                skipped += 1
                continue
            if max_per_state > 0 and per_state_count.get(state_id, 0) >= int(max_per_state):
                continue
            idx = per_state_count.get(state_id, 0)
            dst_name = f"{source_label}_state_{state_id:04d}_seed_{idx:02d}.pdb"
            dst = pdb_dst_dir / dst_name
            try:
                shutil.copy2(src, dst)
            except Exception:
                skipped += 1
                continue
            per_state_count[state_id] = idx + 1
            copied += 1
            state = registry.get_state(state_id)
            rows.append({
                "seed_name": f"{source_label}_state_{state_id:04d}_seed_{idx:02d}",
                "survivor_pdb_path": str(Path("pdbs") / dst_name),
                "source_run_dir": str(source_run_dir),
                "source_pdb_path": str(src),
                "source_label": str(source_label),
                "source_state_id": int(state_id),
                "source_epoch_window": int(local_window),
                "primary_cv_value": "" if state is None else float(state.primary_center),
                "secondary_cv_value": "" if state is None or state.secondary_center is None else float(state.secondary_center),
            })
    csv_path = seed_bank_dir / "final_survivor_seeds.csv"
    fieldnames = [
        "seed_name", "survivor_pdb_path", "source_run_dir", "source_pdb_path",
        "source_label", "source_state_id", "source_epoch_window",
        "primary_cv_value", "secondary_cv_value",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": "adaptive_seed_bank_v2_multi_run",
        "status": "ok" if rows else "empty",
        "source_run_dirs": [str(p) for p in source_run_dirs],
        "seed_bank_dir": str(seed_bank_dir),
        "final_survivor_seeds_csv": str(csv_path),
        "copied_pdbs": int(copied),
        "skipped_pdbs": int(skipped),
        "states_with_seeds": sorted(int(k) for k in per_state_count),
        "rows": rows,
    }
    write_json(seed_bank_dir / "seed_bank_report.json", payload)
    select_state_aware_seeds_for_targets(seed_bank_dir, registry)
    lines = ["# Adaptive multi-run seed bank", "", f"Status: **{payload['status']}**", f"Copied PDBs: **{copied}**", ""]
    if rows:
        lines.append("| state | seed | source segment |")
        lines.append("|---:|---|---|")
        for row in rows[:100]:
            lines.append(f"| {row['source_state_id']} | {row['seed_name']} | `{row['source_run_dir']}` |")
    (seed_bank_dir / "seed_bank_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def select_state_aware_seeds_for_targets(seed_bank_dir: Path, registry: WindowStateRegistry) -> Dict[str, Any]:
    """Score seed-bank structures against active target states by CV distance.

    The assignments are also consumed by scheduled adaptive-production segments
    when state-aware seed filtering is enabled.  In that mode each subset run
    receives a small filtered seed bank containing the closest available seed
    for the states in that segment, instead of handing the worker the entire
    mixed conformer pool and hoping it guesses politely.
    """
    seed_bank_dir = Path(seed_bank_dir)
    rows = _read_csv_dicts(seed_bank_dir / "final_survivor_seeds.csv")
    assignments: List[Dict[str, Any]] = []
    for target in registry.active_states():
        best = None
        best_score = float("inf")
        for row in rows:
            p = _float_or_none(row.get("primary_cv_value"))
            s = _float_or_none(row.get("secondary_cv_value"))
            if p is None:
                continue
            dp = float(p) - float(target.primary_center)
            ds = 0.0
            if target.secondary_center is not None and s is not None:
                ds = float(s) - float(target.secondary_center)
            score = dp * dp + ds * ds
            if score < best_score:
                best_score = score
                best = row
        if best is not None:
            assignments.append({
                "target_state_id": int(target.state_id),
                "target_primary_center": float(target.primary_center),
                "target_secondary_center": "" if target.secondary_center is None else float(target.secondary_center),
                "seed_name": best.get("seed_name", ""),
                "seed_pdb_path": best.get("survivor_pdb_path", ""),
                "seed_source_state_id": best.get("source_state_id", ""),
                "score": float(best_score),
            })
    csv_path = seed_bank_dir / "state_aware_seed_assignments.csv"
    with csv_path.open("w", newline="") as handle:
        fieldnames = ["target_state_id", "target_primary_center", "target_secondary_center", "seed_name", "seed_pdb_path", "seed_source_state_id", "score"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(assignments)
    payload = {
        "schema_version": "adaptive_state_aware_seed_assignments_v1",
        "seed_bank_dir": str(seed_bank_dir),
        "assignments_csv": str(csv_path),
        "n_assignments": int(len(assignments)),
        "assignments": assignments,
        "note": "State-aware seed assignments are used by adaptive-production scheduled segments to build filtered per-segment seed banks when --adaptive-production-state-aware-seed-filtering is enabled.",
    }
    write_json(seed_bank_dir / "state_aware_seed_assignments.json", payload)
    return payload


def filter_seed_bank_for_state_ids(
    seed_bank_dir: Path,
    target_state_ids: Sequence[int],
    output_dir: Path,
    *,
    max_per_state: int = 1,
) -> Dict[str, Any]:
    """Write a filtered GENPEPT-compatible seed bank for selected target states.

    ``production.run_gareus`` currently accepts a seed-conformer directory as a
    pool rather than a per-window map.  Scheduled adaptive production often runs
    only a subset of states, so giving each subset segment a filtered pool is the
    simplest practical way to make seed propagation state aware without changing
    the lower-level seeding machinery.  The helper prefers
    ``state_aware_seed_assignments.csv`` and falls back to source-state rows in
    ``final_survivor_seeds.csv``.
    """
    seed_bank_dir = Path(seed_bank_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_ids = [int(x) for x in target_state_ids]
    target_set = set(target_ids)
    max_per_state = max(1, int(max_per_state or 1))
    base_rows = _read_csv_dicts(seed_bank_dir / "final_survivor_seeds.csv")
    assignment_rows = _read_csv_dicts(seed_bank_dir / "state_aware_seed_assignments.csv")
    chosen: List[Dict[str, Any]] = []
    seen_paths: set[str] = set()

    def _copy_row(row: Dict[str, Any], target_state_id: Optional[int], source: str) -> None:
        src_text = str(row.get("seed_pdb_path", row.get("survivor_pdb_path", ""))).strip()
        if not src_text:
            return
        src = Path(src_text)
        if not src.is_absolute():
            src = seed_bank_dir / src
        if not src.exists():
            # Some rows store paths relative to the original seed bank parent.
            alt = seed_bank_dir / "pdbs" / Path(src_text).name
            if alt.exists():
                src = alt
        if not src.exists():
            return
        key = str(src.resolve())
        if key in seen_paths:
            return
        seen_paths.add(key)
        dst_dir = output_dir / "pdbs"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / Path(src).name
        if not dst.exists():
            shutil.copy2(src, dst)
        chosen.append({
            "seed_name": str(row.get("seed_name", Path(src).stem)),
            "survivor_pdb_path": str(dst),
            "source_seed_bank": str(seed_bank_dir),
            "source_pdb_path": str(src),
            "target_state_id": "" if target_state_id is None else int(target_state_id),
            "source_state_id": row.get("seed_source_state_id", row.get("source_state_id", "")),
            "primary_cv_value": row.get("representative_primary_cv_value", row.get("primary_cv_value", "")),
            "secondary_cv_value": row.get("representative_secondary_cv_value", row.get("secondary_cv_value", "")),
            "filter_source": source,
        })

    # Prefer nearest-seed assignments for target states.
    per_target_count: Dict[int, int] = {sid: 0 for sid in target_ids}
    for row in assignment_rows:
        sid_val = _float_or_none(row.get("target_state_id"))
        if sid_val is None:
            continue
        sid = int(sid_val)
        if sid not in target_set or per_target_count.get(sid, 0) >= max_per_state:
            continue
        _copy_row(row, sid, "state_aware_assignment")
        per_target_count[sid] = per_target_count.get(sid, 0) + 1

    # Fallback: preserve seeds whose source state is part of the target subset.
    for row in base_rows:
        sid_val = _float_or_none(row.get("source_state_id"))
        if sid_val is None:
            continue
        sid = int(sid_val)
        if sid not in target_set or per_target_count.get(sid, 0) >= max_per_state:
            continue
        _copy_row(row, sid, "source_state_match")
        per_target_count[sid] = per_target_count.get(sid, 0) + 1

    # Last fallback: if nothing matched, copy a very small generic pool so the
    # worker can still initialize instead of failing over an empty directory.
    if not chosen:
        for row in base_rows[: max(1, min(len(base_rows), len(target_ids) or 1))]:
            _copy_row(row, None, "generic_fallback")

    csv_path = output_dir / "final_survivor_seeds.csv"
    fieldnames = [
        "seed_name", "survivor_pdb_path", "source_seed_bank", "source_pdb_path",
        "target_state_id", "source_state_id", "primary_cv_value", "secondary_cv_value", "filter_source",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(chosen)
    payload = {
        "schema_version": "adaptive_filtered_seed_bank_v1",
        "status": "ok" if chosen else "empty",
        "source_seed_bank": str(seed_bank_dir),
        "filtered_seed_bank_dir": str(output_dir),
        "final_survivor_seeds_csv": str(csv_path),
        "target_state_ids": [int(x) for x in target_ids],
        "n_rows": int(len(chosen)),
        "rows": _json_ready(chosen[:200]),
    }
    write_json(output_dir / "filtered_seed_bank_report.json", payload)
    lines = [
        "# Filtered adaptive seed bank",
        "",
        f"Status: **{payload['status']}**",
        f"Source seed bank: `{seed_bank_dir}`",
        f"Targets: `{', '.join(map(str, target_ids))}`",
        f"Rows: **{len(chosen)}**",
        "",
    ]
    if chosen:
        lines.append("| target | source | seed |")
        lines.append("|---:|---:|---|")
        for row in chosen[:100]:
            lines.append(f"| {row.get('target_state_id','')} | {row.get('source_state_id','')} | `{row.get('survivor_pdb_path','')}` |")
    (output_dir / "filtered_seed_bank_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# Diagnostics collection
# ---------------------------------------------------------------------------


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="") as handle:
        return [dict(r) for r in csv.DictReader(handle)]


def _has_parquet_rows(run_dir: Path, dirname: str) -> bool:
    data_dir = Path(run_dir) / str(dirname)
    return data_dir.exists() and any(data_dir.glob("**/*.parquet"))


def _run_dir_has_samples(run_dir: Path) -> bool:
    run_dir = Path(run_dir)
    csv_path = run_dir / "samples.csv"
    return _has_parquet_rows(run_dir, "samples") or (csv_path.exists() and csv_path.stat().st_size > 0)


def _value_at(columns: Dict[str, Any], key: str, index: int, default: Any = None) -> Any:
    arr = columns.get(key)
    if arr is None:
        return default
    try:
        value = arr[index]
    except Exception:
        return default
    try:
        if np.ma.is_masked(value):
            return default
    except Exception:
        pass
    try:
        return value.item()
    except Exception:
        return value


def _blank_if_nonfinite(value: Any) -> Any:
    try:
        f = float(value)
    except Exception:
        return ""
    return float(f) if math.isfinite(f) else ""


def _metadata_beta_1_over_kj_mol(run_dir: Path) -> Optional[float]:
    meta = read_json_file(Path(run_dir) / "gareus_metadata.json", {}) or {}
    temp = _float_or_none(meta.get("temperature_K", meta.get("temperature_k")))
    if temp is None or temp <= 0.0:
        return None
    return 1.0 / (8.314462618e-3 * float(temp))


def _read_sample_dicts(run_dir: Path) -> List[Dict[str, Any]]:
    """Read canonical Parquet production samples, falling back to legacy CSV."""
    run_dir = Path(run_dir)
    if _has_parquet_rows(run_dir, "samples"):
        try:
            from .query import load_samples
            data = load_samples(run_dir)
        except Exception:
            data = {}
        n = int(len(data.get("step", []))) if data and "step" in data else 0
        if n > 0:
            beta = _metadata_beta_1_over_kj_mol(run_dir)
            rows: List[Dict[str, Any]] = []
            for i in range(n):
                boost_kj = _blank_if_nonfinite(_value_at(data, "gamd_boost_total", i, ""))
                row = {
                    "step": _value_at(data, "step", i, ""),
                    "replica": _value_at(data, "replica", i, ""),
                    "window": _value_at(data, "window_id", i, ""),
                    "cv_A": _blank_if_nonfinite(_value_at(data, "cv1", i, "")),
                    "primary_cv_value": _blank_if_nonfinite(_value_at(data, "cv1", i, "")),
                    "secondary_cv": _blank_if_nonfinite(_value_at(data, "cv2", i, "")),
                    "potential_kj_mol": _blank_if_nonfinite(_value_at(data, "potential", i, "")),
                    "gamd_boost_total_kj_mol": boost_kj,
                    "gamd_boost_total_kcal_mol": (float(boost_kj) / 4.184) if boost_kj != "" else "",
                    "beta_1_over_kJ_mol": "" if beta is None else float(beta),
                    "segment_id": _value_at(data, "segment_id", i, ""),
                }
                rows.append(row)
            return rows
    return _read_csv_dicts(run_dir / "samples.csv")


def _read_exchange_dicts(run_dir: Path) -> List[Dict[str, Any]]:
    """Read canonical Parquet exchange events, falling back to legacy CSV."""
    run_dir = Path(run_dir)
    if _has_parquet_rows(run_dir, "exchanges"):
        try:
            from .query import load_exchanges
            data = load_exchanges(run_dir)
        except Exception:
            data = {}
        n = int(len(data.get("step", []))) if data and "step" in data else 0
        if n > 0:
            rows: List[Dict[str, Any]] = []
            for i in range(n):
                rows.append({
                    "step": _value_at(data, "step", i, ""),
                    "replica_i": _value_at(data, "replica_i", i, ""),
                    "replica_j": _value_at(data, "replica_j", i, ""),
                    "window_i": _value_at(data, "window_i", i, ""),
                    "window_j": _value_at(data, "window_j", i, ""),
                    "delta_e": _blank_if_nonfinite(_value_at(data, "delta_e", i, "")),
                    "accepted": _value_at(data, "accepted", i, False),
                    "segment_id": _value_at(data, "segment_id", i, ""),
                })
            return rows
    return _read_csv_dicts(run_dir / "exchanges.csv")


def _finite_float_list(values: Iterable[Any]) -> np.ndarray:
    out = []
    for val in values:
        f = _float_or_none(val)
        if f is not None:
            out.append(float(f))
    return np.asarray(out, dtype=float)


def _summary(values: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return None, None, None, None
    return float(np.min(arr)), float(np.max(arr)), float(np.mean(arr)), float(np.std(arr))


def _hist_overlap(a: np.ndarray, b: np.ndarray, bins: int = 80) -> Optional[float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return None
    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return None
    pad = max(1.0e-12, 0.02 * (hi - lo))
    ha, _ = np.histogram(a, bins=max(8, int(bins)), range=(lo - pad, hi + pad))
    hb, _ = np.histogram(b, bins=max(8, int(bins)), range=(lo - pad, hi + pad))
    if ha.sum() <= 0 or hb.sum() <= 0:
        return None
    pa = ha.astype(float) / float(ha.sum())
    pb = hb.astype(float) / float(hb.sum())
    return float(np.minimum(pa, pb).sum())


def _load_epoch_window_map(epoch_dir: Path, fallback_registry: WindowStateRegistry) -> Dict[int, int]:
    path = Path(epoch_dir) / "epoch_window_map.csv"
    rows = _read_csv_dicts(path)
    if rows:
        out = {}
        for row in rows:
            w = _float_or_none(row.get("epoch_window"))
            sid = _float_or_none(row.get("state_id"))
            if w is not None and sid is not None:
                out[int(w)] = int(sid)
        if out:
            return out
    # Fallback: assume current active state order matched local window order.
    return {i: int(s.state_id) for i, s in enumerate(fallback_registry.active_states())}


def build_geometry_edges(registry: WindowStateRegistry) -> List[Tuple[int, int, str, Optional[float]]]:
    """Build a light geometry graph between active states.

    For 1D states, this is a simple sorted nearest-neighbor chain.  For 2D
    states, use immediate row/column-like nearest neighbors plus one nearest
    Euclidean neighbor per state to keep sparse patches connected.
    """
    active = registry.active_states()
    if len(active) <= 1:
        return []
    has_secondary = any(s.secondary_center is not None for s in active)
    primary = np.asarray([s.primary_center for s in active], dtype=float)
    secondary = np.asarray([0.0 if s.secondary_center is None else s.secondary_center for s in active], dtype=float)
    ids = [int(s.state_id) for s in active]
    edges: Dict[Tuple[int, int], Tuple[int, int, str, Optional[float]]] = {}

    def add(a_idx: int, b_idx: int, etype: str, nd: Optional[float]) -> None:
        if a_idx == b_idx:
            return
        a, b = sorted((ids[a_idx], ids[b_idx]))
        edges[(a, b)] = (a, b, etype, nd)

    order = list(np.argsort(primary))
    for left, right in zip(order[:-1], order[1:]):
        add(int(left), int(right), "primary_chain", None)
    if has_secondary:
        p_scale = _positive_scale(primary)
        s_scale = _positive_scale(secondary)
        coords = np.column_stack([primary / p_scale, secondary / s_scale])
        for i in range(len(active)):
            dist = np.sqrt(np.sum((coords - coords[i]) ** 2, axis=1))
            dist[i] = np.inf
            j = int(np.argmin(dist))
            if math.isfinite(float(dist[j])):
                add(i, j, "nearest_2d", float(dist[j]))
    return list(edges.values())


def _positive_scale(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return 1.0
    unique = np.unique(np.round(vals, 8))
    if unique.size <= 1:
        return max(1.0, abs(float(unique[0])))
    diffs = np.diff(np.sort(unique))
    diffs = diffs[diffs > 1.0e-12]
    return float(np.median(diffs)) if diffs.size else 1.0


def collect_epoch_diagnostics(epoch_dir: Path, registry: WindowStateRegistry, policy: Optional[AdaptiveDecisionPolicy] = None) -> Dict[str, Any]:
    """Collect simple per-state and per-edge diagnostics from one epoch output."""
    epoch_dir = Path(epoch_dir)
    policy = policy or AdaptiveDecisionPolicy()
    samples = _read_sample_dicts(epoch_dir)
    exchanges = _read_exchange_dicts(epoch_dir)
    window_map = _load_epoch_window_map(epoch_dir, registry)

    by_state: Dict[int, List[Dict[str, str]]] = {sid: [] for sid in registry.active_state_ids()}
    by_window_values: Dict[int, np.ndarray] = {}
    by_window_secondary: Dict[int, np.ndarray] = {}
    for row in samples:
        w = _float_or_none(row.get("window"))
        if w is None:
            continue
        sid = window_map.get(int(w), int(w))
        by_state.setdefault(sid, []).append(row)

    state_rows: List[StateDiagnostics] = []
    state_by_id: Dict[int, StateDiagnostics] = {}
    for epoch_window, state_id in sorted(window_map.items()):
        rows = by_state.get(int(state_id), [])
        cv = _finite_float_list(r.get("cv_A", r.get("primary_cv_value")) for r in rows)
        sec = _finite_float_list(r.get("secondary_cv") for r in rows)
        boost_kcal = _finite_float_list(r.get("gamd_boost_total_kcal_mol") for r in rows)
        cv_min, cv_max, cv_mean, cv_std = _summary(cv)
        smin, smax, smean, sstd = _summary(sec)
        _bmin, _bmax, _bmean, bstd = _summary(boost_kcal)
        diag = StateDiagnostics(
            state_id=int(state_id),
            epoch_window=int(epoch_window),
            sample_count=int(len(rows)),
            cv_min=cv_min,
            cv_max=cv_max,
            cv_mean=cv_mean,
            cv_std=cv_std,
            secondary_min=smin,
            secondary_max=smax,
            secondary_mean=smean,
            secondary_std=sstd,
            gamd_boost_sd_kcal_mol=bstd,
        )
        if len(rows) < int(policy.min_samples_for_add):
            diag.warnings.append("low_sample_count")
        if bstd is not None and bstd > float(policy.max_gamd_boost_sd_kcal_mol):
            diag.warnings.append("high_gamd_boost_sd")
        state_rows.append(diag)
        state_by_id[int(state_id)] = diag
        by_window_values[int(epoch_window)] = cv
        by_window_secondary[int(epoch_window)] = sec

    pair_stats: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for row in exchanges:
        wi = _float_or_none(row.get("window_i"))
        wj = _float_or_none(row.get("window_j"))
        if wi is None or wj is None:
            continue
        si = window_map.get(int(wi), int(wi))
        sj = window_map.get(int(wj), int(wj))
        key = tuple(sorted((int(si), int(sj))))
        attempts, accepted = pair_stats.get(key, (0, 0))
        attempts += 1
        acc_raw = str(row.get("accepted", "0")).strip().lower()
        if acc_raw in {"1", "true", "yes", "y", "t"}:
            accepted += 1
        pair_stats[key] = (attempts, accepted)

    state_to_window = {sid: w for w, sid in window_map.items()}
    edge_rows: List[EdgeDiagnostics] = []
    for si, sj, etype, nd in build_geometry_edges(registry):
        wi = state_to_window.get(int(si), -1)
        wj = state_to_window.get(int(sj), -1)
        overlap = None
        if wi >= 0 and wj >= 0:
            overlap = _hist_overlap(by_window_values.get(wi, np.asarray([])), by_window_values.get(wj, np.asarray([])))
        attempts, accepted = pair_stats.get(tuple(sorted((int(si), int(sj)))), (0, 0))
        acceptance = (float(accepted) / float(attempts)) if attempts > 0 else None
        edge = EdgeDiagnostics(
            state_i=int(si),
            state_j=int(sj),
            window_i=int(wi),
            window_j=int(wj),
            edge_type=str(etype),
            normalized_distance=nd,
            overlap=overlap,
            exchange_attempts=int(attempts),
            exchange_accepted=int(accepted),
            exchange_acceptance=acceptance,
        )
        if overlap is None or overlap < float(policy.target_overlap):
            edge.warnings.append("low_or_missing_overlap")
        if attempts > 0 and acceptance is not None and acceptance < float(policy.min_exchange_acceptance):
            edge.warnings.append("low_exchange_acceptance")
        edge_rows.append(edge)

    payload = {
        "schema_version": "adaptive_production_epoch_diagnostics_v1",
        "epoch_dir": str(epoch_dir),
        "n_samples_rows": int(len(samples)),
        "n_exchange_rows": int(len(exchanges)),
        "window_map": {str(k): int(v) for k, v in sorted(window_map.items())},
        "states": [s.to_dict() for s in state_rows],
        "edges": [e.to_dict() for e in edge_rows],
        "policy": _json_ready(asdict(policy)),
    }
    write_json(epoch_dir / "adaptive_epoch_diagnostics.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Post-hoc union-state MBAR helpers
# ---------------------------------------------------------------------------


def _first_finite_float(rows: Sequence[Dict[str, Any]], names: Sequence[str], default: Optional[float] = None) -> Optional[float]:
    for row in rows:
        for name in names:
            value = _float_or_none(row.get(name))
            if value is not None:
                return float(value)
    return default


def _sample_sources_from_run_root(label: str, run_dir: Path) -> List[Tuple[str, Path]]:
    """Return sample-bearing directories for a plain or scheduled run root."""
    run_dir = Path(run_dir)
    out: List[Tuple[str, Path]] = []
    if _run_dir_has_samples(run_dir):
        out.append((label, run_dir))
    for child in sorted(run_dir.iterdir() if run_dir.exists() else []):
        if not child.is_dir():
            continue
        if child.name == "baseline" or child.name.startswith("topup_"):
            if _run_dir_has_samples(child):
                out.append((f"{label}/{child.name}", child))
    return out


def _epoch_sample_sources(adaptive_dir: Path, include_epochs: bool) -> List[Tuple[str, Path]]:
    adaptive_dir = Path(adaptive_dir)
    sources: List[Tuple[str, Path]] = []
    if include_epochs:
        for epoch_dir in sorted(adaptive_dir.glob("epoch_[0-9][0-9][0-9]")):
            sources.extend(_sample_sources_from_run_root(epoch_dir.name, epoch_dir))
    final_dir = adaptive_dir / "final"
    sources.extend(_sample_sources_from_run_root("final", final_dir))
    for ext_dir in sorted(adaptive_dir.glob("final_extension_[0-9][0-9][0-9]")):
        sources.extend(_sample_sources_from_run_root(ext_dir.name, ext_dir))
    return sources


def build_union_state_mbar_inputs(
    adaptive_dir: Path,
    registry: WindowStateRegistry,
    *,
    include_epochs: bool = False,
    output_prefix: str = "adaptive_union_mbar",
) -> Dict[str, Any]:
    """Build post-hoc bias matrices over the union of registry states.

    Adaptive production can add and retire states, so per-epoch analysis arrays
    may have different widths.  This helper reconstructs one sample-major
    matrix against the *union* of registry states from scalar sample traces:

        U_nk = 0.5*k_i*(cv_n - center_i)^2 + optional secondary term

    It writes both a human-readable combined sample CSV and an NPZ containing
    the matrices required for MBAR-like downstream analysis.  This is still a
    conservative helper: it only includes final-phase samples by default unless
    ``include_epochs`` is true.
    """
    adaptive_dir = Path(adaptive_dir)
    out_prefix = adaptive_dir / output_prefix
    states = [s for s in registry.all_states() if bool(s.usable_for_mbar)]
    if not states:
        raise RuntimeError("cannot build union MBAR inputs: registry has no usable states")
    state_ids = np.asarray([int(s.state_id) for s in states], dtype=np.int64)
    primary_centers = np.asarray([float(s.primary_center) for s in states], dtype=np.float64)
    primary_k = np.asarray([float(s.primary_k) for s in states], dtype=np.float64)
    secondary_centers = np.asarray([
        np.nan if s.secondary_center is None else float(s.secondary_center) for s in states
    ], dtype=np.float64)
    secondary_k = np.asarray([
        0.0 if s.secondary_k is None else float(s.secondary_k) for s in states
    ], dtype=np.float64)

    sample_rows: List[Dict[str, Any]] = []
    for source_label, sample_dir in _epoch_sample_sources(adaptive_dir, include_epochs=include_epochs):
        rows = _read_sample_dicts(sample_dir)
        if not rows:
            continue
        window_map = _load_epoch_window_map(sample_dir, registry)
        for row in rows:
            w = _float_or_none(row.get("window"))
            cv = _float_or_none(row.get("cv_A", row.get("primary_cv_value")))
            if w is None or cv is None:
                continue
            sampled_state = window_map.get(int(w), int(w))
            sec = _float_or_none(row.get("secondary_cv"))
            beta = _float_or_none(row.get("beta_1_over_kJ_mol"))
            boost_kj = _float_or_none(row.get("gamd_boost_total_kj_mol"))
            step = _float_or_none(row.get("step"))
            sample_rows.append({
                "source": source_label,
                "source_dir": str(sample_dir),
                "step": "" if step is None else int(step),
                "sampled_epoch_window": int(w),
                "sampled_state_id": int(sampled_state),
                "cv_A": float(cv),
                "secondary_cv": "" if sec is None else float(sec),
                "beta_1_over_kJ_mol": "" if beta is None else float(beta),
                "gamd_boost_total_kj_mol": "" if boost_kj is None else float(boost_kj),
                "usable_for_mbar": int(source_label.startswith("final") or include_epochs),
            })

    if not sample_rows:
        raise RuntimeError(f"no usable sample rows found under {adaptive_dir}")

    cv_values = np.asarray([float(r["cv_A"]) for r in sample_rows], dtype=np.float64)
    secondary_values = np.asarray([
        np.nan if r["secondary_cv"] == "" else float(r["secondary_cv"]) for r in sample_rows
    ], dtype=np.float64)
    beta_values = np.asarray([
        np.nan if r["beta_1_over_kJ_mol"] == "" else float(r["beta_1_over_kJ_mol"]) for r in sample_rows
    ], dtype=np.float64)
    beta_finite = beta_values[np.isfinite(beta_values)]
    beta = float(beta_finite[0]) if beta_finite.size else float("nan")

    primary_delta = cv_values[:, np.newaxis] - primary_centers[np.newaxis, :]
    primary_bias_kcal = 0.5 * primary_k[np.newaxis, :] * primary_delta * primary_delta
    secondary_bias_kcal = np.zeros_like(primary_bias_kcal)
    has_secondary_samples = np.isfinite(secondary_values).any()
    has_secondary_states = np.isfinite(secondary_centers).any()
    if has_secondary_samples and has_secondary_states:
        dsec = secondary_values[:, np.newaxis] - secondary_centers[np.newaxis, :]
        secondary_mask = np.isfinite(dsec) & np.isfinite(secondary_centers[np.newaxis, :])
        secondary_k_mat = np.broadcast_to(secondary_k[np.newaxis, :], dsec.shape)
        secondary_bias_kcal[secondary_mask] = 0.5 * secondary_k_mat[secondary_mask] * dsec[secondary_mask] * dsec[secondary_mask]
    umbrella_bias_kcal = primary_bias_kcal + secondary_bias_kcal
    umbrella_bias_kj = 4.184 * umbrella_bias_kcal
    if math.isfinite(beta):
        umbrella_reduced_bias_nk = float(beta) * umbrella_bias_kj
    else:
        umbrella_reduced_bias_nk = np.full_like(umbrella_bias_kj, np.nan)

    id_to_index = {int(sid): i for i, sid in enumerate(state_ids.tolist())}
    sampled_state_ids = np.asarray([int(r["sampled_state_id"]) for r in sample_rows], dtype=np.int64)
    n_k = np.zeros(len(states), dtype=np.int64)
    for sid in sampled_state_ids:
        idx = id_to_index.get(int(sid))
        if idx is not None:
            n_k[idx] += 1

    csv_path = out_prefix.with_suffix(".samples.csv")
    with csv_path.open("w", newline="") as handle:
        fieldnames = [
            "source", "source_dir", "step", "sampled_epoch_window", "sampled_state_id",
            "cv_A", "secondary_cv", "beta_1_over_kJ_mol", "gamd_boost_total_kj_mol", "usable_for_mbar",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sample_rows)

    npz_path = out_prefix.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        state_ids=state_ids,
        sampled_state_ids=sampled_state_ids,
        cv_A=cv_values,
        secondary_cv=secondary_values,
        primary_centers=primary_centers,
        primary_k=primary_k,
        secondary_centers=secondary_centers,
        secondary_k=secondary_k,
        umbrella_bias_kcal_mol_nk=umbrella_bias_kcal,
        umbrella_bias_kj_mol_nk=umbrella_bias_kj,
        umbrella_reduced_bias_nk=umbrella_reduced_bias_nk,
        N_k=n_k,
    )
    meta = {
        "schema_version": "adaptive_union_mbar_inputs_v1",
        "adaptive_dir": str(adaptive_dir),
        "include_epochs": bool(include_epochs),
        "n_samples": int(len(sample_rows)),
        "n_states": int(len(states)),
        "state_ids": [int(x) for x in state_ids.tolist()],
        "N_k": [int(x) for x in n_k.tolist()],
        "beta_1_over_kJ_mol": None if not math.isfinite(beta) else float(beta),
        "samples_csv": str(csv_path),
        "arrays_npz": str(npz_path),
        "matrix_shape_convention": "sample-major [n_samples, n_states]; transpose to u_kn if PyMBAR expects [K,N]",
        "note": "Biases are reconstructed post-hoc from scalar CV traces against the union of registry states. Final-only samples are the conservative default.",
    }
    json_path = out_prefix.with_suffix(".json")
    write_json(json_path, _json_ready(meta))
    return meta


def _parse_union_fes_bins(value: Any) -> tuple[int, int]:
    text = str(value or "80,40").replace("x", ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return 80, 40
    try:
        nx = max(4, int(parts[0]))
    except Exception:
        nx = 80
    try:
        ny = max(4, int(parts[1])) if len(parts) > 1 else 40
    except Exception:
        ny = 40
    return nx, ny


def _npz_path_from_union_meta(adaptive_dir: Path, union_meta: Dict[str, Any], output_prefix: str) -> Path:
    raw = str((union_meta or {}).get("arrays_npz", "") or "")
    if raw:
        return Path(raw)
    return Path(adaptive_dir) / f"{output_prefix}.npz"


def run_union_mbar_analysis(
    adaptive_dir: Path,
    union_meta: Dict[str, Any],
    *,
    output_prefix: str = "adaptive_union_mbar_analysis",
    fes_bins: Any = "80,40",
) -> Dict[str, Any]:
    """Run a post-hoc PyMBAR state/free-energy diagnostic on union-state inputs.

    This is deliberately a *diagnostic* layer.  It checks whether the union-state
    reduced-bias matrix can be solved by PyMBAR, writes state free energies and
    overlap matrices when available, and writes simple coverage histograms.  The
    coverage histograms are not labelled as final PMFs; the final publication FES
    still needs the user's chosen binning/reweighting protocol.  Tiny caveat,
    because apparently free energy estimators still refuse to read our minds.
    """
    adaptive_dir = Path(adaptive_dir)
    output_base = adaptive_dir / output_prefix
    npz_path = _npz_path_from_union_meta(adaptive_dir, union_meta, "adaptive_union_mbar")
    payload: Dict[str, Any] = {
        "schema_version": "adaptive_union_mbar_analysis_v1",
        "adaptive_dir": str(adaptive_dir),
        "input_npz": str(npz_path),
        "status": "missing",
        "warnings": [],
        "errors": [],
    }
    if not npz_path.exists():
        payload["errors"].append("union-state NPZ is missing")
        write_json(output_base.with_suffix(".json"), payload)
        _write_union_mbar_analysis_markdown(output_base.with_suffix(".md"), payload)
        return payload

    try:
        with np.load(npz_path, allow_pickle=False) as data:
            reduced_nk = np.asarray(data["umbrella_reduced_bias_nk"], dtype=float)
            n_k = np.asarray(data["N_k"], dtype=np.int64)
            state_ids = np.asarray(data["state_ids"], dtype=np.int64)
            cv = np.asarray(data["cv_A"], dtype=float)
            secondary = np.asarray(data.get("secondary_cv", np.full_like(cv, np.nan)), dtype=float)
            sampled_state_ids = np.asarray(data.get("sampled_state_ids", np.full(cv.shape, -1)), dtype=np.int64)
    except Exception as exc:
        payload["status"] = "error"
        payload["errors"].append(f"could not read union-state NPZ: {exc}")
        write_json(output_base.with_suffix(".json"), payload)
        _write_union_mbar_analysis_markdown(output_base.with_suffix(".md"), payload)
        return payload

    if reduced_nk.ndim != 2 or reduced_nk.shape[0] <= 0 or reduced_nk.shape[1] <= 0:
        payload["status"] = "error"
        payload["errors"].append(f"umbrella_reduced_bias_nk has invalid shape {reduced_nk.shape}")
        write_json(output_base.with_suffix(".json"), payload)
        _write_union_mbar_analysis_markdown(output_base.with_suffix(".md"), payload)
        return payload
    if not np.isfinite(reduced_nk).all():
        frac = float(np.mean(np.isfinite(reduced_nk)))
        payload["warnings"].append(f"reduced bias matrix contains non-finite values; finite_fraction={frac:.6f}")
        finite_rows = np.isfinite(reduced_nk).all(axis=1)
        reduced_nk = reduced_nk[finite_rows]
        cv = cv[finite_rows]
        secondary = secondary[finite_rows]
        sampled_state_ids = sampled_state_ids[finite_rows]
        # Recompute N_k after dropping non-finite rows.
        sid_to_idx = {int(sid): i for i, sid in enumerate(state_ids.tolist())}
        n_k = np.zeros(len(state_ids), dtype=np.int64)
        for sid in sampled_state_ids:
            idx = sid_to_idx.get(int(sid))
            if idx is not None:
                n_k[idx] += 1

    payload.update({
        "n_samples": int(reduced_nk.shape[0]),
        "n_states": int(reduced_nk.shape[1]),
        "state_ids": [int(x) for x in state_ids.tolist()],
        "N_k": [int(x) for x in n_k.tolist()],
        "states_with_zero_samples": [int(state_ids[i]) for i, n in enumerate(n_k.tolist()) if int(n) <= 0],
    })
    if reduced_nk.shape[0] <= 0:
        payload["status"] = "error"
        payload["errors"].append("no finite samples remain after filtering")
        write_json(output_base.with_suffix(".json"), payload)
        _write_union_mbar_analysis_markdown(output_base.with_suffix(".md"), payload)
        return payload

    # Always write coverage diagnostics, even if PyMBAR is unavailable.
    coverage = _write_union_coverage_outputs(output_base, cv, secondary, sampled_state_ids, state_ids, fes_bins=fes_bins)
    payload["coverage"] = coverage

    try:
        from pymbar import MBAR  # type: ignore
    except Exception as exc:
        payload["status"] = "warning"
        payload["warnings"].append(f"PyMBAR unavailable; wrote coverage diagnostics only: {exc}")
        write_json(output_base.with_suffix(".json"), _json_ready(payload))
        _write_union_mbar_analysis_markdown(output_base.with_suffix(".md"), payload)
        return payload

    try:
        # PyMBAR expects u_kn with shape [K, N].  Union input stores [N, K].
        mbar = MBAR(reduced_nk.T, n_k, initialize="zeros", solver_protocol="robust")
        fe = mbar.compute_free_energy_differences()
        delta_f = np.asarray(fe.get("Delta_f", []), dtype=float)
        d_delta_f = np.asarray(fe.get("dDelta_f", []), dtype=float)
        state_free_csv = output_base.with_name(output_base.name + ".state_free_energies.csv")
        with state_free_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["state_id", "index", "N_k", "f_relative_to_state0", "df_relative_to_state0"])
            writer.writeheader()
            for i, sid in enumerate(state_ids.tolist()):
                writer.writerow({
                    "state_id": int(sid),
                    "index": int(i),
                    "N_k": int(n_k[i]) if i < len(n_k) else 0,
                    "f_relative_to_state0": float(delta_f[0, i]) if delta_f.shape[0] > 0 and i < delta_f.shape[1] else "",
                    "df_relative_to_state0": float(d_delta_f[0, i]) if d_delta_f.shape[0] > 0 and i < d_delta_f.shape[1] else "",
                })
        payload["state_free_energies_csv"] = str(state_free_csv)
        payload["delta_f_shape"] = list(delta_f.shape)

        try:
            overlap_result = mbar.compute_overlap()
            overlap_matrix = np.asarray(overlap_result.get("matrix", overlap_result), dtype=float)
            overlap_csv = output_base.with_name(output_base.name + ".overlap_matrix.csv")
            _write_matrix_csv(overlap_csv, overlap_matrix, state_ids)
            payload["overlap_matrix_csv"] = str(overlap_csv)
            payload["overlap_scalar"] = _json_ready(overlap_result.get("scalar")) if isinstance(overlap_result, dict) else None
            if overlap_matrix.size:
                off = overlap_matrix.copy()
                np.fill_diagonal(off, np.nan)
                payload["min_offdiag_overlap"] = None if not np.isfinite(off).any() else float(np.nanmin(off))
                payload["max_offdiag_overlap"] = None if not np.isfinite(off).any() else float(np.nanmax(off))
        except Exception as exc:
            payload["warnings"].append(f"PyMBAR overlap calculation failed: {exc}")

        try:
            eff = mbar.compute_effective_sample_number()
            payload["effective_sample_number"] = _json_ready(eff)
        except Exception as exc:
            payload["warnings"].append(f"PyMBAR effective sample number failed: {exc}")

        payload["status"] = "warning" if payload["warnings"] else "ok"
    except Exception as exc:
        payload["status"] = "error"
        payload["errors"].append(f"PyMBAR solve failed: {exc}")

    write_json(output_base.with_suffix(".json"), _json_ready(payload))
    _write_union_mbar_analysis_markdown(output_base.with_suffix(".md"), payload)
    return payload


def _write_matrix_csv(path: Path, matrix: np.ndarray, state_ids: np.ndarray) -> None:
    matrix = np.asarray(matrix, dtype=float)
    with Path(path).open("w", newline="") as handle:
        fields = ["state_id"] + [f"state_{int(sid)}" for sid in state_ids.tolist()]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, sid in enumerate(state_ids.tolist()):
            row = {"state_id": int(sid)}
            for j, sj in enumerate(state_ids.tolist()):
                row[f"state_{int(sj)}"] = float(matrix[i, j]) if i < matrix.shape[0] and j < matrix.shape[1] else ""
            writer.writerow(row)


def _write_union_coverage_outputs(output_base: Path, cv: np.ndarray, secondary: np.ndarray, sampled_state_ids: np.ndarray, state_ids: np.ndarray, *, fes_bins: Any) -> Dict[str, Any]:
    nx, ny = _parse_union_fes_bins(fes_bins)
    cv = np.asarray(cv, dtype=float)
    secondary = np.asarray(secondary, dtype=float)
    finite_cv = cv[np.isfinite(cv)]
    coverage: Dict[str, Any] = {"primary_bins": int(nx), "secondary_bins": int(ny)}
    if finite_cv.size:
        hist, edges = np.histogram(finite_cv, bins=nx)
        primary_csv = output_base.with_name(output_base.name + ".primary_coverage.csv")
        with primary_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["bin", "lo", "hi", "count"])
            writer.writeheader()
            for i, count in enumerate(hist.tolist()):
                writer.writerow({"bin": int(i), "lo": float(edges[i]), "hi": float(edges[i + 1]), "count": int(count)})
        coverage["primary_coverage_csv"] = str(primary_csv)
        coverage["primary_nonempty_bins"] = int(np.sum(hist > 0))
    if finite_cv.size and np.isfinite(secondary).any():
        mask = np.isfinite(cv) & np.isfinite(secondary)
        hist2d, xedges, yedges = np.histogram2d(cv[mask], secondary[mask], bins=(nx, ny))
        coverage_npz = output_base.with_name(output_base.name + ".coverage_2d.npz")
        np.savez_compressed(coverage_npz, hist2d=hist2d, primary_edges=xedges, secondary_edges=yedges)
        coverage["coverage_2d_npz"] = str(coverage_npz)
        coverage["coverage_2d_nonempty_bins"] = int(np.sum(hist2d > 0))

    state_count_csv = output_base.with_name(output_base.name + ".sampled_state_counts.csv")
    counts = {int(sid): 0 for sid in state_ids.tolist()}
    for sid in sampled_state_ids.tolist():
        if int(sid) in counts:
            counts[int(sid)] += 1
    with state_count_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["state_id", "sample_count"])
        writer.writeheader()
        for sid in state_ids.tolist():
            writer.writerow({"state_id": int(sid), "sample_count": int(counts.get(int(sid), 0))})
    coverage["sampled_state_counts_csv"] = str(state_count_csv)
    return coverage


def _write_union_mbar_analysis_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# Adaptive union MBAR analysis",
        "",
        f"Status: **{payload.get('status', 'unknown')}**",
        f"Input NPZ: `{payload.get('input_npz', '')}`",
        f"Samples/states: **{payload.get('n_samples', 0)} / {payload.get('n_states', 0)}**",
        "",
    ]
    if payload.get("errors"):
        lines.append("## Errors")
        for err in payload.get("errors", []):
            lines.append(f"- {err}")
        lines.append("")
    if payload.get("warnings"):
        lines.append("## Warnings")
        for warn in payload.get("warnings", []):
            lines.append(f"- {warn}")
        lines.append("")
    lines.append("## Outputs")
    for key in ("state_free_energies_csv", "overlap_matrix_csv"):
        if payload.get(key):
            lines.append(f"- `{key}`: `{payload.get(key)}`")
    coverage = payload.get("coverage", {}) or {}
    for key in ("primary_coverage_csv", "coverage_2d_npz", "sampled_state_counts_csv"):
        if coverage.get(key):
            lines.append(f"- `{key}`: `{coverage.get(key)}`")
    lines.append("")
    if payload.get("min_offdiag_overlap") is not None:
        lines.append(f"Minimum off-diagonal MBAR overlap: **{payload.get('min_offdiag_overlap')}**")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")




def _final_sample_dirs(adaptive_dir: Path) -> List[Tuple[str, Path]]:
    """Return final + final-extension directories with sample files."""
    adaptive_dir = Path(adaptive_dir)
    out: List[Tuple[str, Path]] = []
    final_dir = adaptive_dir / "final"
    out.extend(_sample_sources_from_run_root("final", final_dir))
    for ext_dir in sorted(adaptive_dir.glob("final_extension_[0-9][0-9][0-9]")):
        out.extend(_sample_sources_from_run_root(ext_dir.name, ext_dir))
    return out


def collect_final_combined_diagnostics(adaptive_dir: Path, registry: WindowStateRegistry, policy: Optional[AdaptiveDecisionPolicy] = None) -> Dict[str, Any]:
    """Collect diagnostics over the frozen final phase plus final extensions.

    Each final segment is run with the same frozen active window table, but it
    still has its own local output directory.  This helper combines samples and
    exchange counts across those directories so the quality gate evaluates the
    full final sampling effort rather than only the last segment.  The combined
    JSON is written to ``adaptive_final_combined_diagnostics.json``.
    """
    adaptive_dir = Path(adaptive_dir)
    policy = policy or AdaptiveDecisionPolicy()
    sources = _final_sample_dirs(adaptive_dir)
    by_state: Dict[int, List[Dict[str, Any]]] = {sid: [] for sid in registry.active_state_ids()}
    by_window_values: Dict[int, List[float]] = {}
    pair_stats: Dict[Tuple[int, int], Tuple[int, int]] = {}
    source_summaries = []
    total_samples = 0
    total_exchanges = 0

    # Use the final active state ordering as the canonical window order for
    # geometry diagnostics.  Each segment writes an epoch_window_map.csv, so
    # sample/exchange rows are mapped back to persistent state_id.
    canonical_active = registry.active_states()
    canonical_state_to_window = {int(s.state_id): int(i) for i, s in enumerate(canonical_active)}
    canonical_window_map = {int(i): int(s.state_id) for i, s in enumerate(canonical_active)}

    for source_label, sample_dir in sources:
        window_map = _load_epoch_window_map(sample_dir, registry)
        samples = _read_sample_dicts(sample_dir)
        exchanges = _read_exchange_dicts(sample_dir)
        total_samples += len(samples)
        total_exchanges += len(exchanges)
        source_summaries.append({
            "source": source_label,
            "dir": str(sample_dir),
            "sample_rows": int(len(samples)),
            "exchange_rows": int(len(exchanges)),
            "window_map": {str(k): int(v) for k, v in sorted(window_map.items())},
        })
        for row in samples:
            w = _float_or_none(row.get("window"))
            if w is None:
                continue
            sid = window_map.get(int(w), int(w))
            by_state.setdefault(int(sid), []).append(row)
            cv = _float_or_none(row.get("cv_A", row.get("primary_cv_value")))
            if cv is not None:
                canonical_w = canonical_state_to_window.get(int(sid))
                if canonical_w is not None:
                    by_window_values.setdefault(int(canonical_w), []).append(float(cv))
        for row in exchanges:
            wi = _float_or_none(row.get("window_i"))
            wj = _float_or_none(row.get("window_j"))
            if wi is None or wj is None:
                continue
            si = window_map.get(int(wi), int(wi))
            sj = window_map.get(int(wj), int(wj))
            key = tuple(sorted((int(si), int(sj))))
            attempts, accepted = pair_stats.get(key, (0, 0))
            attempts += 1
            acc_raw = str(row.get("accepted", "0")).strip().lower()
            if acc_raw in {"1", "true", "yes", "y", "t"}:
                accepted += 1
            pair_stats[key] = (attempts, accepted)

    state_rows: List[StateDiagnostics] = []
    for epoch_window, state_id in sorted(canonical_window_map.items()):
        rows = by_state.get(int(state_id), [])
        cv = _finite_float_list(r.get("cv_A", r.get("primary_cv_value")) for r in rows)
        sec = _finite_float_list(r.get("secondary_cv") for r in rows)
        boost_kcal = _finite_float_list(r.get("gamd_boost_total_kcal_mol") for r in rows)
        cv_min, cv_max, cv_mean, cv_std = _summary(cv)
        smin, smax, smean, sstd = _summary(sec)
        _bmin, _bmax, _bmean, bstd = _summary(boost_kcal)
        diag = StateDiagnostics(
            state_id=int(state_id),
            epoch_window=int(epoch_window),
            sample_count=int(len(rows)),
            cv_min=cv_min, cv_max=cv_max, cv_mean=cv_mean, cv_std=cv_std,
            secondary_min=smin, secondary_max=smax, secondary_mean=smean, secondary_std=sstd,
            gamd_boost_sd_kcal_mol=bstd,
        )
        if len(rows) < int(policy.final_min_samples_per_state):
            diag.warnings.append("low_final_sample_count")
        if bstd is not None and bstd > float(policy.max_gamd_boost_sd_kcal_mol):
            diag.warnings.append("high_gamd_boost_sd")
        state_rows.append(diag)

    edge_rows: List[EdgeDiagnostics] = []
    for si, sj, etype, nd in build_geometry_edges(registry):
        wi = canonical_state_to_window.get(int(si), -1)
        wj = canonical_state_to_window.get(int(sj), -1)
        overlap = None
        if wi >= 0 and wj >= 0:
            overlap = _hist_overlap(
                np.asarray(by_window_values.get(wi, []), dtype=float),
                np.asarray(by_window_values.get(wj, []), dtype=float),
            )
        attempts, accepted = pair_stats.get(tuple(sorted((int(si), int(sj)))), (0, 0))
        acceptance = (float(accepted) / float(attempts)) if attempts > 0 else None
        edge = EdgeDiagnostics(
            state_i=int(si), state_j=int(sj), window_i=int(wi), window_j=int(wj),
            edge_type=str(etype), normalized_distance=nd, overlap=overlap,
            exchange_attempts=int(attempts), exchange_accepted=int(accepted), exchange_acceptance=acceptance,
        )
        if overlap is None or overlap < float(policy.target_overlap):
            edge.warnings.append("low_or_missing_overlap")
        if attempts > 0 and acceptance is not None and acceptance < float(policy.min_exchange_acceptance):
            edge.warnings.append("low_exchange_acceptance")
        edge_rows.append(edge)

    payload = {
        "schema_version": "adaptive_production_final_combined_diagnostics_v1",
        "adaptive_dir": str(adaptive_dir),
        "sources": source_summaries,
        "n_sample_rows": int(total_samples),
        "n_exchange_rows": int(total_exchanges),
        "window_map": {str(k): int(v) for k, v in sorted(canonical_window_map.items())},
        "states": [s.to_dict() for s in state_rows],
        "edges": [e.to_dict() for e in edge_rows],
        "policy": _json_ready(asdict(policy)),
    }
    write_json(adaptive_dir / "adaptive_final_combined_diagnostics.json", payload)
    return payload


def evaluate_adaptive_quality_gate(
    adaptive_dir: Path,
    registry: WindowStateRegistry,
    final_diagnostics: Dict[str, Any],
    policy: Optional[AdaptiveDecisionPolicy] = None,
    *,
    union_inputs: Optional[Dict[str, Any]] = None,
    union_analysis: Optional[Dict[str, Any]] = None,
    output_prefix: str = "adaptive_quality_gate",
) -> Dict[str, Any]:
    """Evaluate whether the frozen final phase is analysis-ready.

    The gate is intentionally conservative and transparent.  It reports three
    buckets: hard errors, sampling issues that normally require more final
    sampling, and softer warnings from post-hoc union/MBAR diagnostics.
    """
    adaptive_dir = Path(adaptive_dir)
    policy = policy or AdaptiveDecisionPolicy()
    errors: List[str] = []
    needs_more_sampling: List[str] = []
    warnings: List[str] = []
    recommendations: List[str] = []

    if bool(policy.final_connectivity_required) and not active_graph_connected(registry):
        errors.append("active final window geometry graph is disconnected")
        recommendations.append("Add bridge windows or disable final connectivity requirement only for debugging.")

    n_samples = int(final_diagnostics.get("n_sample_rows", final_diagnostics.get("n_samples_rows", 0)) or 0)
    if n_samples <= 0:
        errors.append("no final-phase sample rows were found")

    low_states = []
    high_boost = []
    for state in final_diagnostics.get("states", []) or []:
        sid = int(state.get("state_id", -1))
        count = int(state.get("sample_count", 0) or 0)
        if count < int(policy.final_min_samples_per_state):
            low_states.append({"state_id": sid, "sample_count": count, "minimum": int(policy.final_min_samples_per_state)})
        bsd = state.get("gamd_boost_sd_kcal_mol")
        if bsd is not None and float(bsd) > float(policy.max_gamd_boost_sd_kcal_mol):
            high_boost.append({"state_id": sid, "boost_sd_kcal_mol": float(bsd), "maximum": float(policy.max_gamd_boost_sd_kcal_mol)})
    if low_states:
        needs_more_sampling.append(f"{len(low_states)} state(s) have fewer than {policy.final_min_samples_per_state} final samples")
        recommendations.append("Run additional frozen final sampling with the same active window table.")
    if high_boost:
        needs_more_sampling.append(f"{len(high_boost)} state(s) exceed the GaMD boost SD threshold")
        recommendations.append("Inspect GaMD boost distributions; consider lower sigma0 or final unboosted/weakly boosted production.")

    weak_edges = []
    for edge in final_diagnostics.get("edges", []) or []:
        overlap = edge.get("overlap")
        acc = edge.get("exchange_acceptance")
        weak = overlap is None or float(overlap) < float(policy.target_overlap)
        if acc is not None and float(acc) < float(policy.min_exchange_acceptance):
            weak = True
        if weak:
            weak_edges.append(edge)
    if weak_edges:
        needs_more_sampling.append(f"{len(weak_edges)} final edge(s) are below overlap/exchange thresholds")
        recommendations.append("Add bridge windows in another adaptive epoch or extend final sampling if the weak edges are sample-limited.")

    coverage_fraction = None
    if union_analysis:
        if union_analysis.get("status") == "error":
            warnings.append("union MBAR analysis returned error status")
        coverage = union_analysis.get("coverage", {}) or {}
        nonempty = coverage.get("primary_nonempty_bins")
        bins = coverage.get("primary_bins")
        if nonempty is not None and bins:
            coverage_fraction = float(nonempty) / max(1.0, float(bins))
            if coverage_fraction < float(policy.quality_min_primary_coverage_fraction):
                warnings.append(
                    f"primary diagnostic coverage fraction {coverage_fraction:.3f} below target {policy.quality_min_primary_coverage_fraction:.3f}"
                )
        min_overlap = union_analysis.get("min_offdiag_overlap")
        if min_overlap is not None and math.isfinite(float(min_overlap)) and float(min_overlap) <= 0.0:
            warnings.append("union MBAR overlap matrix has zero/negative minimum off-diagonal overlap")
    if union_inputs and union_inputs.get("error"):
        warnings.append(f"union MBAR input builder failed: {union_inputs.get('error')}")

    if errors:
        status = "error"
    elif needs_more_sampling:
        status = "needs_more_sampling"
    elif warnings:
        status = "warning"
    else:
        status = "ok"

    payload = {
        "schema_version": "adaptive_production_quality_gate_v1",
        "status": status,
        "adaptive_dir": str(adaptive_dir),
        "errors": errors,
        "needs_more_sampling": needs_more_sampling,
        "warnings": warnings,
        "recommendations": sorted(set(recommendations)),
        "low_sample_states": low_states,
        "high_boost_states": high_boost,
        "weak_edges": _json_ready(weak_edges),
        "primary_coverage_fraction": coverage_fraction,
        "policy": _json_ready(asdict(policy)),
        "final_diagnostics_json": str(adaptive_dir / "adaptive_final_combined_diagnostics.json"),
        "union_inputs": union_inputs or {},
        "union_analysis_status": None if not union_analysis else union_analysis.get("status"),
        "hard_fail_requested": bool(policy.quality_hard_fail),
    }
    json_path = adaptive_dir / f"{output_prefix}.json"
    md_path = adaptive_dir / f"{output_prefix}.md"
    write_json(json_path, _json_ready(payload))
    _write_quality_gate_markdown(md_path, payload)
    payload["json"] = str(json_path)
    payload["md"] = str(md_path)
    return payload


def _write_quality_gate_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# Adaptive production quality gate",
        "",
        f"Status: **{payload.get('status', 'unknown')}**",
        f"Diagnostics: `{payload.get('final_diagnostics_json', '')}`",
        "",
    ]
    for title, key in (("Errors", "errors"), ("Needs more sampling", "needs_more_sampling"), ("Warnings", "warnings"), ("Recommendations", "recommendations")):
        items = payload.get(key, []) or []
        if not items:
            continue
        lines.append(f"## {title}")
        for item in items:
            lines.append(f"- {item}")
        lines.append("")
    weak_edges = payload.get("weak_edges", []) or []
    if weak_edges:
        lines.append("## Weak final edges")
        lines.append("| state_i | state_j | overlap | exchange_acceptance | warnings |")
        lines.append("|---:|---:|---:|---:|---|")
        for edge in weak_edges[:50]:
            lines.append(
                f"| {edge.get('state_i')} | {edge.get('state_j')} | {edge.get('overlap')} | "
                f"{edge.get('exchange_acceptance')} | {', '.join(edge.get('warnings', []) or [])} |"
            )
        lines.append("")
    low = payload.get("low_sample_states", []) or []
    if low:
        lines.append("## Low-sample states")
        lines.append("| state_id | sample_count | minimum |")
        lines.append("|---:|---:|---:|")
        for row in low[:100]:
            lines.append(f"| {row.get('state_id')} | {row.get('sample_count')} | {row.get('minimum')} |")
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def propose_actions_from_diagnostics(registry: WindowStateRegistry, diagnostics: Dict[str, Any], policy: Optional[AdaptiveDecisionPolicy] = None) -> List[Tuple]:
    policy = policy or AdaptiveDecisionPolicy()
    actions: List[Tuple] = []
    state_rows = {int(r["state_id"]): r for r in diagnostics.get("states", [])}
    edge_rows = diagnostics.get("edges", [])

    # 1. Add midpoint states at weak edges.  This is the safest first adaptive
    # production behavior because it only increases resolution where data say
    # the current graph is bad.
    added = 0
    weak_edges = []
    for edge in edge_rows:
        overlap = edge.get("overlap")
        acc = edge.get("exchange_acceptance")
        weak = (overlap is None or float(overlap) < float(policy.target_overlap))
        if acc is not None:
            weak = weak or float(acc) < float(policy.min_exchange_acceptance)
        if not weak:
            continue
        si, sj = int(edge["state_i"]), int(edge["state_j"])
        di, dj = state_rows.get(si, {}), state_rows.get(sj, {})
        if int(di.get("sample_count", 0) or 0) < int(policy.min_samples_for_add):
            continue
        if int(dj.get("sample_count", 0) or 0) < int(policy.min_samples_for_add):
            continue
        weak_edges.append(edge)

    weak_edges.sort(key=lambda e: (
        1.0 if e.get("overlap") is None else float(e.get("overlap")),
        1.0 if e.get("exchange_acceptance") is None else float(e.get("exchange_acceptance")),
    ))

    for edge in weak_edges:
        if added >= int(policy.max_new_windows_per_epoch):
            break
        s1 = registry.get_state(int(edge["state_i"]))
        s2 = registry.get_state(int(edge["state_j"]))
        if s1 is None or s2 is None:
            continue
        primary = 0.5 * (float(s1.primary_center) + float(s2.primary_center))
        primary_k = 0.5 * (float(s1.primary_k) + float(s2.primary_k))
        if s1.secondary_center is not None and s2.secondary_center is not None:
            secondary = 0.5 * (float(s1.secondary_center) + float(s2.secondary_center))
            secondary_k = 0.5 * (float(s1.secondary_k or 0.0) + float(s2.secondary_k or 0.0))
        else:
            secondary = None
            secondary_k = None
        if registry.has_near_duplicate(primary, secondary, policy):
            continue
        params = (primary, primary_k, secondary, secondary_k)
        reason = (
            f"weak edge {s1.state_id}-{s2.state_id}: "
            f"overlap={edge.get('overlap')}, exchange_acceptance={edge.get('exchange_acceptance')}"
        )
        actions.append(("add", int(s1.state_id), params, reason))
        added += 1

    # 2. Optionally retire clearly converged non-critical states.  Disabled by
    # default in the policy because graph-safe retirement needs conservative use.
    if bool(policy.retire_converged):
        graph_critical = _graph_articulation_states(registry)
        bad_touching = set()
        for edge in edge_rows:
            if edge.get("overlap") is None or float(edge.get("overlap") or 0.0) < float(policy.target_overlap):
                bad_touching.add(int(edge["state_i"])); bad_touching.add(int(edge["state_j"]))
            acc = edge.get("exchange_acceptance")
            if acc is not None and float(acc) < float(policy.min_exchange_acceptance):
                bad_touching.add(int(edge["state_i"])); bad_touching.add(int(edge["state_j"]))
        for state in registry.active_states():
            sid = int(state.state_id)
            diag = state_rows.get(sid, {})
            if sid in graph_critical or sid in bad_touching:
                continue
            if int(diag.get("sample_count", 0) or 0) < int(policy.min_samples_for_retire):
                continue
            bsd = diag.get("gamd_boost_sd_kcal_mol")
            if bsd is not None and float(bsd) > float(policy.max_gamd_boost_sd_kcal_mol):
                continue
            actions.append(("retire", sid, "converged and non-critical under conservative policy"))

    # 3. Explicitly record extensions for states that are obviously undersampled.
    action_keys = {(a[0], a[1]) for a in actions if len(a) > 1}
    for state in registry.active_states():
        sid = int(state.state_id)
        if ("retire", sid) in action_keys:
            continue
        diag = state_rows.get(sid, {})
        if int(diag.get("sample_count", 0) or 0) < int(policy.min_samples_for_retire):
            actions.append(("extend", sid, "below minimum retirement sample count"))
    return actions


def _graph_articulation_states(registry: WindowStateRegistry) -> set[int]:
    active_ids = registry.active_state_ids()
    if len(active_ids) <= 2:
        return set(active_ids)
    edges = [(a, b) for a, b, _t, _d in build_geometry_edges(registry)]
    out = set()
    for removed in active_ids:
        remaining = [sid for sid in active_ids if sid != removed]
        if not remaining:
            out.add(removed)
            continue
        adj = {sid: set() for sid in remaining}
        for a, b in edges:
            if a == removed or b == removed:
                continue
            if a in adj and b in adj:
                adj[a].add(b); adj[b].add(a)
        seen = set()
        stack = [remaining[0]]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(sorted(adj.get(cur, set()) - seen))
        if len(seen) != len(remaining):
            out.add(removed)
    return out


def active_graph_connected(registry: WindowStateRegistry) -> bool:
    """Return True if the active-state geometry graph is connected."""
    active_ids = registry.active_state_ids()
    if len(active_ids) <= 1:
        return True
    adj = {sid: set() for sid in active_ids}
    for a, b, _etype, _nd in build_geometry_edges(registry):
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    seen = set()
    stack = [active_ids[0]]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(sorted(adj.get(cur, set()) - seen))
    return len(seen) == len(active_ids)


# ---------------------------------------------------------------------------
# Adaptive allocation scheduler
# ---------------------------------------------------------------------------


def write_state_subset_window_csv(
    registry: WindowStateRegistry,
    path: Path,
    state_ids: Sequence[int],
    map_path: Optional[Path] = None,
) -> Path:
    """Write a GAREUS explicit-window CSV for a selected state subset."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    states = [registry.get_state(int(sid)) for sid in state_ids]
    states = [s for s in states if s is not None and s.active]
    if not states:
        raise RuntimeError("cannot write state-subset window CSV: no active states selected")
    has_secondary = any(s.secondary_center is not None for s in states)
    rows: List[Dict[str, Any]] = []
    for i, state in enumerate(states):
        if has_secondary and state.secondary_center is None:
            raise RuntimeError("state subset mixes 1D and 2D states; cannot write one explicit table")
        rows.append({
            "epoch_window": int(i),
            "state_id": int(state.state_id),
            "primary_cv_center": float(state.primary_center),
            "primary_cv_k_kcal": float(state.primary_k),
            "distance_center_A": float(state.primary_center),
            "distance_k_kcal_mol_A2": float(state.primary_k),
            "secondary_cv_center": "" if state.secondary_center is None else float(state.secondary_center),
            "secondary_cv_k_kcal_mol": "" if state.secondary_k is None else float(state.secondary_k),
            "window_type": "adaptive_production_state_subset",
            "patch_lifecycle": "adaptive_production_scheduled_segment",
            "parent_state_id": "" if state.parent_state_id is None else int(state.parent_state_id),
            "created_epoch": int(state.created_epoch),
            "source": str(state.source),
            "reason": str(state.reason),
            "usable_for_mbar": int(bool(state.usable_for_mbar)),
            "burnin_steps": int(state.burnin_steps),
        })
    fieldnames = [
        "epoch_window", "state_id", "primary_cv_center", "primary_cv_k_kcal",
        "distance_center_A", "distance_k_kcal_mol_A2", "secondary_cv_center",
        "secondary_cv_k_kcal_mol", "window_type", "patch_lifecycle", "parent_state_id",
        "created_epoch", "source", "reason", "usable_for_mbar", "burnin_steps",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if map_path is not None:
        map_path = Path(map_path)
        map_path.parent.mkdir(parents=True, exist_ok=True)
        with map_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch_window", "state_id", "primary_center", "secondary_center"])
            writer.writeheader()
            for i, state in enumerate(states):
                writer.writerow({
                    "epoch_window": int(i),
                    "state_id": int(state.state_id),
                    "primary_center": float(state.primary_center),
                    "secondary_center": "" if state.secondary_center is None else float(state.secondary_center),
                })
    return path


def _state_rows_by_id(diagnostics: Optional[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not isinstance(diagnostics, dict):
        return out
    for row in diagnostics.get("states", []) or []:
        try:
            out[int(row.get("state_id"))] = dict(row)
        except Exception:
            pass
    return out


def _weak_edge_touch_counts(diagnostics: Optional[Dict[str, Any]], policy: AdaptiveDecisionPolicy) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    if not isinstance(diagnostics, dict):
        return counts
    for edge in diagnostics.get("edges", []) or []:
        overlap = edge.get("overlap")
        acc = edge.get("exchange_acceptance")
        weak = overlap is None or float(overlap) < float(policy.target_overlap)
        if acc is not None:
            weak = weak or float(acc) < float(policy.min_exchange_acceptance)
        if weak:
            for key in ("state_i", "state_j"):
                try:
                    sid = int(edge.get(key))
                    counts[sid] = counts.get(sid, 0) + 1
                except Exception:
                    pass
    return counts


def build_adaptive_epoch_schedule(
    registry: WindowStateRegistry,
    diagnostics: Optional[Dict[str, Any]],
    policy: AdaptiveDecisionPolicy,
    *,
    epoch: int,
    default_steps: int,
    final: bool = False,
) -> List[Dict[str, Any]]:
    """Allocate per-state MD steps for the next epoch/final phase.

    The schedule is intentionally conservative: every active state receives a
    baseline all-state segment so graph/exchange connectivity is sampled at
    least briefly.  Extra top-up segments are assigned to uncertain states.
    """
    active = registry.active_states()
    if not active:
        return []
    default_steps = max(1, int(default_steps))
    min_steps = int(policy.min_state_steps or max(1000, min(default_steps, default_steps // 4 if default_steps >= 4 else 1)))
    max_steps = int(policy.max_state_steps or max(default_steps, 4 * default_steps))
    if max_steps < min_steps:
        max_steps = min_steps
    total_budget = int((policy.final_step_budget if final else policy.epoch_step_budget) or (len(active) * default_steps))
    total_budget = max(len(active) * min_steps, total_budget)
    remaining = max(0, total_budget - len(active) * min_steps)
    state_rows = _state_rows_by_id(diagnostics)
    weak_counts = _weak_edge_touch_counts(diagnostics, policy)
    articulation = _graph_articulation_states(registry)
    scored: List[Dict[str, Any]] = []
    for state in active:
        sid = int(state.state_id)
        diag = state_rows.get(sid, {})
        sample_count = int(float(diag.get("sample_count", 0) or 0))
        boost_sd = diag.get("gamd_boost_sd_kcal_mol")
        score = 1.0
        reasons = ["baseline"]
        if sample_count <= 0:
            score += 2.0 * float(policy.low_sample_bonus)
            reasons.append("no_samples_yet")
        elif sample_count < int(policy.min_samples_for_retire):
            deficit = 1.0 - min(1.0, sample_count / max(1.0, float(policy.min_samples_for_retire)))
            score += float(policy.low_sample_bonus) * deficit
            reasons.append("low_effective_sample_proxy")
        if sid in weak_counts:
            score += float(policy.weak_edge_bonus) * float(weak_counts[sid])
            reasons.append(f"touches_{weak_counts[sid]}_weak_edge(s)")
        if sid in articulation:
            score += float(policy.frontier_bonus)
            reasons.append("graph_bridge_state")
        if int(state.created_epoch) >= int(epoch):
            score += float(policy.frontier_bonus) + 1.0
            reasons.append("new_state")
        if boost_sd is not None:
            try:
                if float(boost_sd) > float(policy.max_gamd_boost_sd_kcal_mol):
                    score += float(policy.high_boost_bonus)
                    reasons.append("high_gamd_boost_sd")
            except Exception:
                pass
        scored.append({
            "state_id": sid,
            "score": float(max(0.0, score)),
            "sample_count": sample_count,
            "requested_steps": int(min_steps),
            "baseline_steps": int(min_steps),
            "extra_steps": 0,
            "allocation_reason": "; ".join(reasons),
        })
    score_sum = sum(float(r["score"]) for r in scored) or float(len(scored))
    for row in scored:
        extra = int(round(float(remaining) * float(row["score"]) / score_sum)) if remaining > 0 else 0
        requested = min(max_steps, int(row["baseline_steps"]) + max(0, extra))
        state = registry.get_state(int(row["state_id"]))
        if state is not None and int(state.created_epoch) >= int(epoch):
            new_steps = int(policy.new_state_steps or max(default_steps, max_steps // 2))
            requested = min(max_steps, max(requested, new_steps))
        row["requested_steps"] = int(requested)
        row["extra_steps"] = max(0, int(requested) - int(row["baseline_steps"]))
    return scored


def write_epoch_schedule_files(epoch_dir: Path, schedule: Sequence[Dict[str, Any]], *, prefix: str = "epoch_schedule") -> Dict[str, str]:
    epoch_dir = Path(epoch_dir)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    csv_path = epoch_dir / f"{prefix}.csv"
    json_path = epoch_dir / f"{prefix}.json"
    md_path = epoch_dir / f"{prefix}.md"
    fieldnames = ["state_id", "requested_steps", "baseline_steps", "extra_steps", "score", "sample_count", "allocation_reason"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(schedule)
    write_json(json_path, {"schema_version": "adaptive_epoch_schedule_v1", "rows": list(schedule)})
    lines = ["# Adaptive epoch allocation schedule", ""]
    lines.append("| state | requested steps | baseline | extra | score | reason |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    for row in schedule:
        lines.append(
            f"| {row.get('state_id')} | {row.get('requested_steps')} | {row.get('baseline_steps')} | "
            f"{row.get('extra_steps')} | {float(row.get('score', 0.0)):.3f} | {row.get('allocation_reason', '')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path), "md": str(md_path)}


def _quantized_extra_steps(value: int, quantum: int = 1000) -> int:
    value = max(0, int(value))
    quantum = max(1, int(quantum))
    return int(round(value / float(quantum)) * quantum)



@dataclass
class AdaptiveRuntimePool:
    """Track a global aggregate-MD runtime budget for adaptive production.

    The pool is expressed in aggregate nanoseconds across all active states or
    replicas.  A segment with ``n_states`` windows and ``steps`` MD steps per
    window consumes

        n_states * steps * timestep_fs / 1e6

    nanoseconds from the pool.  This makes the resource pool independent of the
    current number of active windows.
    """

    total_ns: float = 0.0
    timestep_fs: float = 2.0
    used_ns: float = 0.0
    events: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return bool(math.isfinite(float(self.total_ns)) and float(self.total_ns) > 0.0)

    def segment_ns(self, n_states: int, steps: int) -> float:
        n_states = max(0, int(n_states))
        steps = max(0, int(steps))
        return float(n_states) * float(steps) * float(self.timestep_fs) / 1.0e6

    def remaining_ns(self) -> float:
        if not self.enabled:
            return float("inf")
        return max(0.0, float(self.total_ns) - float(self.used_ns))

    def remaining_state_steps(self, reserve_ns: float = 0.0) -> int:
        if not self.enabled:
            return 10**18
        usable_ns = max(0.0, self.remaining_ns() - max(0.0, float(reserve_ns)))
        if float(self.timestep_fs) <= 0.0:
            return 0
        return int(math.floor(usable_ns * 1.0e6 / float(self.timestep_fs)))

    def max_steps_for_segment(self, n_states: int, reserve_ns: float = 0.0) -> int:
        n_states = max(1, int(n_states))
        return int(self.remaining_state_steps(reserve_ns=reserve_ns) // n_states)

    def clip_steps(self, n_states: int, requested_steps: int, reserve_ns: float = 0.0, hard_stop: bool = True) -> int:
        requested_steps = max(0, int(requested_steps))
        if requested_steps <= 0 or not self.enabled or not hard_stop:
            return requested_steps
        return max(0, min(requested_steps, self.max_steps_for_segment(n_states, reserve_ns=reserve_ns)))

    def consume(self, *, label: str, n_states: int, steps: int, path: Optional[Path] = None, kind: str = "segment") -> Dict[str, Any]:
        ns = self.segment_ns(n_states, steps)
        event = {
            "label": str(label),
            "kind": str(kind),
            "path": "" if path is None else str(path),
            "n_states": int(n_states),
            "steps_per_state": int(steps),
            "timestep_fs": float(self.timestep_fs),
            "consumed_ns": float(ns),
            "used_ns_before": float(self.used_ns),
            "used_ns_after": float(self.used_ns + ns),
            "remaining_ns_after": float(max(0.0, float(self.total_ns) - (float(self.used_ns) + ns))) if self.enabled else None,
        }
        self.used_ns += ns
        self.events.append(event)
        if self.enabled and self.used_ns > self.total_ns + 1.0e-9:
            msg = f"adaptive-production MD pool exceeded: used {self.used_ns:.6g} ns > total {self.total_ns:.6g} ns"
            if msg not in self.warnings:
                self.warnings.append(msg)
        return event

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "adaptive_runtime_pool_v1",
            "enabled": bool(self.enabled),
            "total_ns": float(self.total_ns),
            "timestep_fs": float(self.timestep_fs),
            "used_ns": float(self.used_ns),
            "remaining_ns": None if not self.enabled else float(self.remaining_ns()),
            "events": _json_ready(self.events),
            "warnings": list(self.warnings),
            "definition": "aggregate_ns = n_states * steps_per_state * timestep_fs / 1e6",
        }


def _make_runtime_pool(args: Any, policy: AdaptiveDecisionPolicy) -> AdaptiveRuntimePool:
    return AdaptiveRuntimePool(
        total_ns=float(getattr(args, "adaptive_production_total_md_pool_ns", policy.total_md_pool_ns) or 0.0),
        timestep_fs=float(getattr(args, "timestep_fs", 2.0) or 2.0),
    )


def _runtime_pool_final_reserve_ns(pool: AdaptiveRuntimePool, policy: AdaptiveDecisionPolicy, *, final: bool = False) -> float:
    if final or not pool.enabled:
        return 0.0
    frac = max(0.0, min(1.0, float(policy.final_pool_fraction)))
    reserve = max(float(policy.min_final_pool_ns or 0.0), float(pool.total_ns) * frac)
    return min(pool.remaining_ns(), reserve)


def _epoch_steps_from_pool(
    pool: AdaptiveRuntimePool,
    policy: AdaptiveDecisionPolicy,
    n_states: int,
    epochs_remaining: int,
    timestep_fs: float,
) -> int:
    """Per-state step target for one epoch: spread remaining epoch budget evenly.

    Divides (remaining_ns - final_reserve) across epochs_remaining and converts
    to per-state steps using n_states and timestep_fs.  clip_steps() then acts
    as a safety net, but since target = fair share it should not clip.
    """
    if not pool.enabled or n_states <= 0 or epochs_remaining <= 0 or timestep_fs <= 0:
        return 0
    final_reserve = _runtime_pool_final_reserve_ns(pool, policy, final=False)
    available_ns = max(0.0, pool.remaining_ns() - final_reserve)
    epoch_ns = available_ns / epochs_remaining
    return max(1, int(epoch_ns * 1e6 / timestep_fs / n_states))


def _final_steps_from_pool(
    pool: AdaptiveRuntimePool,
    n_states: int,
    timestep_fs: float,
) -> int:
    """Per-state step target for the frozen final phase: spend entire remaining budget."""
    if not pool.enabled or n_states <= 0 or timestep_fs <= 0:
        return 0
    available_ns = max(0.0, pool.remaining_ns())
    return max(1, int(available_ns * 1e6 / timestep_fs / n_states))


def _write_runtime_pool_reports(adaptive_dir: Path, pool: AdaptiveRuntimePool) -> Dict[str, str]:
    adaptive_dir = Path(adaptive_dir)
    json_path = adaptive_dir / "adaptive_runtime_pool.json"
    md_path = adaptive_dir / "adaptive_runtime_pool.md"
    payload = pool.to_dict()
    write_json(json_path, _json_ready(payload))
    lines = [
        "# Adaptive production MD runtime pool",
        "",
        f"Enabled: **{payload.get('enabled')}**",
        f"Total pool: **{payload.get('total_ns')} ns**",
        f"Used: **{payload.get('used_ns'):.6g} ns**" if isinstance(payload.get('used_ns'), (int, float)) else "Used: n/a",
        f"Remaining: **{payload.get('remaining_ns'):.6g} ns**" if isinstance(payload.get('remaining_ns'), (int, float)) else "Remaining: n/a",
        "",
        "Definition: `aggregate_ns = n_states * steps_per_state * timestep_fs / 1e6`.",
        "",
        "## Events",
        "",
        "| label | kind | states | steps/state | consumed ns | used after |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for ev in payload.get("events", []) or []:
        lines.append(
            f"| {ev.get('label')} | {ev.get('kind')} | {ev.get('n_states')} | {ev.get('steps_per_state')} | "
            f"{float(ev.get('consumed_ns', 0.0)):.6g} | {float(ev.get('used_ns_after', 0.0)):.6g} |"
        )
    if payload.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for w in payload.get("warnings") or []:
            lines.append(f"- {w}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}



def _adaptive_runtime_pool_from_dict(cls, payload: Dict[str, Any]) -> "AdaptiveRuntimePool":
    """Build a runtime-pool object from a persisted report payload."""
    pool = cls(
        total_ns=float(payload.get("total_ns", 0.0) or 0.0),
        timestep_fs=float(payload.get("timestep_fs", 2.0) or 2.0),
        used_ns=float(payload.get("used_ns", 0.0) or 0.0),
        events=list(payload.get("events", []) or []),
        warnings=list(payload.get("warnings", []) or []),
    )
    return pool


def _adaptive_runtime_pool_validate(adaptive_dir: Path, pool: "AdaptiveRuntimePool", *, label: str = "runtime_pool") -> Dict[str, Any]:
    """Validate runtime-pool accounting and segment path consistency.

    This is intentionally independent of OpenMM.  It verifies that the ledger is
    internally consistent on resume: used ns equals the sum of event ns values,
    events are monotonic, required fields are present, and recorded segment paths
    still exist when paths were written.
    """
    adaptive_dir = Path(adaptive_dir)
    errors: List[str] = []
    warnings: List[str] = []
    events = list(pool.events or [])
    summed = 0.0
    previous_used_after = 0.0
    event_rows = []
    for idx, ev in enumerate(events):
        try:
            consumed = float(ev.get("consumed_ns", 0.0) or 0.0)
        except Exception:
            consumed = float("nan")
        if not math.isfinite(consumed) or consumed < -1.0e-12:
            errors.append(f"event {idx} has invalid consumed_ns={ev.get('consumed_ns')!r}")
            consumed = 0.0
        summed += consumed
        try:
            used_after = float(ev.get("used_ns_after", summed) or summed)
        except Exception:
            used_after = summed
        if used_after + 1.0e-8 < previous_used_after:
            errors.append(f"event {idx} used_ns_after is non-monotonic")
        previous_used_after = max(previous_used_after, used_after)
        raw_path = str(ev.get("path", "") or "")
        path_exists = None
        if raw_path:
            ep = Path(raw_path)
            path_exists = ep.exists()
            if not path_exists:
                warnings.append(f"event {idx} path does not exist anymore: {raw_path}")
        event_rows.append({
            "index": idx,
            "label": str(ev.get("label", "")),
            "kind": str(ev.get("kind", "")),
            "n_states": int(ev.get("n_states", 0) or 0),
            "steps_per_state": int(ev.get("steps_per_state", 0) or 0),
            "consumed_ns": consumed,
            "used_ns_after": used_after,
            "path": raw_path,
            "path_exists": path_exists,
        })
    if abs(float(pool.used_ns) - summed) > max(1.0e-8, 1.0e-6 * max(1.0, summed)):
        errors.append(f"pool.used_ns={pool.used_ns:.9g} does not match sum(events)={summed:.9g}")
    if pool.enabled and float(pool.used_ns) > float(pool.total_ns) + 1.0e-8:
        warnings.append(f"pool used_ns={pool.used_ns:.9g} exceeds total_ns={pool.total_ns:.9g}")
    status = "ok"
    if warnings:
        status = "warning"
    if errors:
        status = "error"
    payload = {
        "schema_version": "adaptive_runtime_pool_validation_v1",
        "label": str(label),
        "status": status,
        "adaptive_dir": str(adaptive_dir),
        "enabled": bool(pool.enabled),
        "total_ns": float(pool.total_ns),
        "timestep_fs": float(pool.timestep_fs),
        "used_ns": float(pool.used_ns),
        "summed_event_ns": float(summed),
        "remaining_ns": None if not pool.enabled else float(pool.remaining_ns()),
        "n_events": int(len(events)),
        "errors": errors,
        "warnings": warnings,
        "events": event_rows,
    }
    write_json(adaptive_dir / f"{label}_validation.json", _json_ready(payload))
    lines = [
        f"# {label.replace('_', ' ').title()} Validation",
        "",
        f"Status: **{status}**",
        f"Events: **{len(events)}**",
        f"Used ns: **{float(pool.used_ns):.6g}**",
        f"Summed event ns: **{summed:.6g}**",
        f"Remaining ns: **{payload.get('remaining_ns')}**",
        "",
    ]
    if errors:
        lines += ["## Errors", ""] + [f"- {x}" for x in errors] + [""]
    if warnings:
        lines += ["## Warnings", ""] + [f"- {x}" for x in warnings] + [""]
    lines += ["## Event ledger", "", "| i | label | kind | states | steps | ns | path exists |", "|---:|---|---|---:|---:|---:|---|"]
    for row in event_rows:
        lines.append(f"| {row['index']} | {row['label']} | {row['kind']} | {row['n_states']} | {row['steps_per_state']} | {row['consumed_ns']:.6g} | {row['path_exists']} |")
    (adaptive_dir / f"{label}_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _load_or_initialize_runtime_pool(adaptive_dir: Path, args: Any, policy: AdaptiveDecisionPolicy, *, resume_requested: bool) -> Tuple[AdaptiveRuntimePool, Dict[str, Any]]:
    """Load runtime-pool ledger on resume, otherwise create a fresh ledger."""
    adaptive_dir = Path(adaptive_dir)
    fresh = _make_runtime_pool(args, policy)
    report_path = adaptive_dir / "adaptive_runtime_pool.json"
    validation: Dict[str, Any] = {}
    if resume_requested and report_path.exists():
        old = read_json_file(report_path, {})
        pool = AdaptiveRuntimePool(
            total_ns=float(old.get("total_ns", fresh.total_ns) or fresh.total_ns),
            timestep_fs=float(old.get("timestep_fs", fresh.timestep_fs) or fresh.timestep_fs),
            used_ns=float(old.get("used_ns", 0.0) or 0.0),
            events=list(old.get("events", []) or []),
            warnings=list(old.get("warnings", []) or []),
        )
        if fresh.enabled and abs(float(pool.total_ns) - float(fresh.total_ns)) > 1.0e-9:
            pool.warnings.append(
                f"resume requested with total pool {fresh.total_ns:.6g} ns, but persisted ledger has {pool.total_ns:.6g} ns; using persisted value"
            )
        if abs(float(pool.timestep_fs) - float(fresh.timestep_fs)) > 1.0e-9:
            pool.warnings.append(
                f"resume timestep {fresh.timestep_fs:.6g} fs differs from persisted pool timestep {pool.timestep_fs:.6g} fs; using persisted value for accounting"
            )
        validation = _adaptive_runtime_pool_validate(adaptive_dir, pool, label="adaptive_runtime_pool_resume")
    else:
        pool = fresh
        validation = _adaptive_runtime_pool_validate(adaptive_dir, pool, label="adaptive_runtime_pool_initial")
    _write_runtime_pool_reports(adaptive_dir, pool)
    return pool, validation


def evaluate_context_reuse_readiness(args: Any, adaptive_dir: Path, registry: Optional[WindowStateRegistry] = None) -> Dict[str, Any]:
    """Write a guarded v12 context-reuse readiness report.

    True in-process context reuse requires keeping OpenMM Simulation objects,
    GaMD CustomIntegrator globals, assignments, trajectory reporters and
    checkpoint state alive across adaptive epochs.  The current adaptive driver
    deliberately uses ``run_gareus`` as an epoch worker, so this function exposes
    a safe gate: requested context reuse is acknowledged, audited and refused
    unless a future worker advertises support.
    """
    adaptive_dir = Path(adaptive_dir)
    requested = bool(getattr(args, "adaptive_production_context_reuse", False))
    require = bool(getattr(args, "adaptive_production_context_reuse_require", False))
    mode = str(getattr(args, "adaptive_production_context_reuse_mode", "off") or "off")
    active_count = len(registry.active_states()) if registry is not None else None
    blockers = []
    if requested:
        blockers.extend([
            "adaptive driver currently uses production.run_gareus() as an isolated epoch worker",
            "OpenMM Simulation/Context objects are owned inside run_gareus() and released at worker exit",
            "GaMD CustomIntegrator globals and reporter/checkpoint state are not yet represented as reusable epoch objects",
        ])
    status = "disabled" if not requested else "fallback_worker_mode"
    if requested and require:
        status = "error"
    payload = {
        "schema_version": "adaptive_context_reuse_readiness_v1",
        "requested": requested,
        "required": require,
        "mode": mode,
        "status": status,
        "active_state_count": active_count,
        "supported_now": False,
        "worker": "run_gareus_epoch_worker",
        "blockers": blockers,
        "safe_fallback": "epoch-worker mode with registry persistence, seed-bank propagation, runtime-pool accounting and resume validation",
        "next_required_refactor": [
            "extract ContextBundle creation from run_gareus()",
            "store per-replica Simulation, assignment, integrator globals and reporter state in an AdaptiveContextPool",
            "add reassign_state(context, old_state_id, new_state_id) with burn-in tagging",
            "checkpoint reusable context pool independently from epoch outputs",
        ],
    }
    write_json(adaptive_dir / "adaptive_context_reuse_readiness.json", _json_ready(payload))
    lines = [
        "# Adaptive Production Context Reuse Readiness",
        "",
        f"Requested: **{requested}**",
        f"Required: **{require}**",
        f"Mode: `{mode}`",
        f"Status: **{status}**",
        "",
    ]
    if blockers:
        lines += ["## Blockers", ""] + [f"- {b}" for b in blockers] + [""]
    lines += ["## Safe fallback", "", str(payload["safe_fallback"]), "", "## Next refactor", ""]
    lines += [f"- {x}" for x in payload["next_required_refactor"]]
    (adaptive_dir / "adaptive_context_reuse_readiness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if requested and require:
        raise RuntimeError("adaptive-production context reuse was required, but the current package only supports safe epoch-worker fallback; see adaptive_context_reuse_readiness.json")
    return payload


def _count_window_csv_rows(path: Optional[Path]) -> Optional[int]:
    if path is None:
        return None
    try:
        p = Path(path)
        if not p.exists():
            return None
        with p.open(newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except Exception:
        return None


def _estimate_active_state_count(args: Any, registry: Optional[WindowStateRegistry], windows_csv: Optional[Path]) -> int:
    if registry is not None:
        return max(1, len(registry.active_states()))
    n = _count_window_csv_rows(windows_csv)
    if n is not None and n > 0:
        return int(n)
    for name in ("adaptive_max_total_windows", "contact_adaptive_max_total_windows", "adaptive_max_windows", "contact_adaptive_max_windows"):
        try:
            val = int(getattr(args, name, 0) or 0)
            if val > 0:
                return val
        except Exception:
            pass
    return 1


def _policy_with_pool_step_budget(
    registry: WindowStateRegistry,
    policy: AdaptiveDecisionPolicy,
    pool: AdaptiveRuntimePool,
    *,
    default_steps: int,
    final: bool,
) -> AdaptiveDecisionPolicy:
    if not pool.enabled:
        return policy
    active_count = max(1, len(registry.active_states()))
    reserve = _runtime_pool_final_reserve_ns(pool, policy, final=final)
    available_state_steps = max(0, pool.remaining_state_steps(reserve_ns=reserve))
    requested_total = int((policy.final_step_budget if final else policy.epoch_step_budget) or (active_count * max(1, int(default_steps))))
    clipped_total = min(max(0, requested_total), available_state_steps)
    p = copy.copy(policy)
    if final:
        p.final_step_budget = int(clipped_total)
    else:
        p.epoch_step_budget = int(clipped_total)
    if clipped_total > 0 and p.min_state_steps > 0 and clipped_total < active_count * int(p.min_state_steps):
        p.min_state_steps = max(1, int(clipped_total // active_count))
    return p


# Threshold for the nonstationary_overlap warning: when the pooled overlap
# exceeds the per-segment minimum by more than this amount, the edge's CV
# distribution is likely non-stationary across segments (e.g. a topup that
# only touched one endpoint inflated the minimum to 0).
NONSTATIONARY_OVERLAP_DELTA: float = 0.2


def collect_segmented_epoch_diagnostics(epoch_dir: Path, registry: WindowStateRegistry, policy: AdaptiveDecisionPolicy) -> Dict[str, Any]:
    """Collect diagnostics for an epoch made of baseline/top-up sub-runs.

    Edge overlap is computed from **pooled** per-state cv_A arrays across all
    segments, not from the nanmin of per-segment overlaps.  A topup that only
    sampled one endpoint of an edge contributes valid data for that endpoint
    while the other endpoint gets an empty array → the per-segment overlap for
    that edge is degenerate (0.0), but the pooled overlap reflects the full
    data from all segments and is unaffected.

    Two extra fields are added to each edge dict:
    - ``segment_min_overlap``: the minimum per-segment overlap (0.0 when a
      segment had no samples for one endpoint), for diagnostic purposes only.
      This field does NOT drive the weak-edge decision.
    - ``nonstationary_overlap`` warning: appended when
      ``pooled_overlap - segment_min_overlap > NONSTATIONARY_OVERLAP_DELTA``,
      indicating that CV coverage varied significantly across segments.
    """
    epoch_dir = Path(epoch_dir)
    segment_dirs = [p for p in sorted(epoch_dir.iterdir()) if p.is_dir() and _run_dir_has_samples(p)]
    if not segment_dirs:
        return collect_epoch_diagnostics(epoch_dir, registry, policy=policy)

    state_acc: Dict[int, Dict[str, Any]] = {}
    edge_acc: Dict[Tuple[int, int], Dict[str, Any]] = {}
    segment_payloads = []

    # Pooled raw cv_A values per state_id — accumulated across all segments.
    pooled_cv_by_state: Dict[int, List[float]] = {}
    # Per-segment overlap values for each edge pair, using 0.0 when one
    # endpoint of the edge had no samples in that segment (degenerate topup).
    seg_overlap_by_edge: Dict[Tuple[int, int], List[float]] = {}

    # Hoist the static edge list once — build_geometry_edges depends only on
    # the registry, which is constant for the lifetime of this call.
    geometry_edges = build_geometry_edges(registry)

    for seg in segment_dirs:
        diag = collect_epoch_diagnostics(seg, registry, policy=policy)
        segment_payloads.append({"segment": seg.name, "diagnostics_json": str(seg / "adaptive_epoch_diagnostics.json")})

        # --- Accumulate raw cv_A values per state from this segment's samples ---
        seg_window_map = _load_epoch_window_map(seg, registry)
        seg_cv_by_state: Dict[int, List[float]] = {}
        for sample_row in _read_sample_dicts(seg):
            w = _float_or_none(sample_row.get("window"))
            if w is None:
                continue
            sid = seg_window_map.get(int(w), int(w))
            cv = _float_or_none(sample_row.get("cv_A", sample_row.get("primary_cv_value")))
            if cv is not None:
                seg_cv_by_state.setdefault(sid, []).append(float(cv))
        for sid, vals in seg_cv_by_state.items():
            pooled_cv_by_state.setdefault(sid, []).extend(vals)

        # --- Compute per-segment per-edge overlap (0.0 for degenerate segments) ---
        for si, sj, _etype, _nd in geometry_edges:
            key = (int(min(si, sj)), int(max(si, sj)))
            a = np.asarray(seg_cv_by_state.get(int(si), []), dtype=float)
            b = np.asarray(seg_cv_by_state.get(int(sj), []), dtype=float)
            # When at least one endpoint contributed samples to this segment,
            # record an overlap value.  Use 0.0 when one side is missing
            # (degenerate topup) rather than silently dropping the segment.
            if a.size > 0 or b.size > 0:
                ov = _hist_overlap(a, b)
                seg_overlap_by_edge.setdefault(key, []).append(0.0 if ov is None else ov)

        # --- State diagnostics aggregation (unchanged logic from Task 1) ---
        for row in diag.get("states", []) or []:
            sid = int(row.get("state_id"))
            acc = state_acc.setdefault(sid, {"state_id": sid, "epoch_window": row.get("epoch_window", -1), "sample_count": 0, "warnings": []})
            n0 = int(acc.get("sample_count", 0) or 0)
            n1 = int(row.get("sample_count", 0) or 0)
            acc["sample_count"] = n0 + n1
            for key in ("cv_min", "secondary_min"):
                val = row.get(key)
                if val is not None:
                    acc[key] = val if acc.get(key) is None else min(float(acc[key]), float(val))
            for key in ("cv_max", "secondary_max"):
                val = row.get(key)
                if val is not None:
                    acc[key] = val if acc.get(key) is None else max(float(acc[key]), float(val))
            # Weighted means are only approximate because segment-level std cannot
            # be combined exactly without second moments; this is a scheduler
            # diagnostic, not publication analysis.
            for key in ("cv_mean", "secondary_mean", "gamd_boost_sd_kcal_mol"):
                val = row.get(key)
                if val is not None and n1 > 0:
                    old = acc.get(key)
                    acc[key] = float(val) if old is None or n0 <= 0 else (float(old) * n0 + float(val) * n1) / float(n0 + n1)
            acc.setdefault("warnings", [])
            for w in row.get("warnings", []) or []:
                if w not in acc["warnings"]:
                    acc["warnings"].append(w)

        # --- Edge exchange stats aggregation (overlap handled below via pooled data) ---
        for edge in diag.get("edges", []) or []:
            ekey = tuple(sorted((int(edge.get("state_i")), int(edge.get("state_j")))))
            acc = edge_acc.setdefault(ekey, {
                "state_i": ekey[0], "state_j": ekey[1], "window_i": edge.get("window_i", -1), "window_j": edge.get("window_j", -1),
                "edge_type": edge.get("edge_type", "segmented"), "exchange_attempts": 0, "exchange_accepted": 0,
                "warnings": [],
            })
            acc["exchange_attempts"] += int(edge.get("exchange_attempts", 0) or 0)
            acc["exchange_accepted"] += int(edge.get("exchange_accepted", 0) or 0)
            # Do NOT union per-segment warnings here: low_or_missing_overlap is
            # re-derived from the pooled overlap below.  Union only non-overlap warnings.
            for w in edge.get("warnings", []) or []:
                if w != "low_or_missing_overlap" and w not in acc["warnings"]:
                    acc["warnings"].append(w)

    # Compute pooled numpy arrays once for all states.
    pooled_np: Dict[int, np.ndarray] = {
        sid: np.asarray(vals, dtype=float)
        for sid, vals in pooled_cv_by_state.items()
    }

    states = [dict(v) for _k, v in sorted(state_acc.items())]
    edges = []
    for ekey, acc in sorted(edge_acc.items()):
        si, sj = ekey
        attempts = int(acc.pop("exchange_attempts", 0) or 0)
        accepted = int(acc.pop("exchange_accepted", 0) or 0)
        acc["exchange_attempts"] = attempts
        acc["exchange_accepted"] = accepted
        acc["exchange_acceptance"] = (accepted / float(attempts)) if attempts > 0 else None

        # Pooled overlap — computed once from all segments' raw cv_A data.
        pooled_overlap = _hist_overlap(
            pooled_np.get(si, np.asarray([])),
            pooled_np.get(sj, np.asarray([])),
        )
        acc["overlap"] = pooled_overlap

        # segment_min_overlap: minimum across per-segment overlap values
        # (0.0 when a segment only touched one endpoint of the edge).
        per_seg_overlaps = seg_overlap_by_edge.get(ekey, [])
        acc["segment_min_overlap"] = float(min(per_seg_overlaps)) if per_seg_overlaps else None

        # Re-derive low_or_missing_overlap from pooled overlap only.
        if pooled_overlap is None or float(pooled_overlap) < float(policy.target_overlap):
            if "low_or_missing_overlap" not in acc["warnings"]:
                acc["warnings"].append("low_or_missing_overlap")

        # Nonstationary warning: pooled is fine but segment min was much lower.
        if (
            pooled_overlap is not None
            and acc["segment_min_overlap"] is not None
            and float(pooled_overlap) - float(acc["segment_min_overlap"]) > NONSTATIONARY_OVERLAP_DELTA
        ):
            if "nonstationary_overlap" not in acc["warnings"]:
                acc["warnings"].append("nonstationary_overlap")

        edges.append(dict(acc))

    payload = {
        "schema_version": "adaptive_epoch_diagnostics_v2_segmented",
        "epoch_dir": str(epoch_dir),
        "segmented_epoch": True,
        "segments": segment_payloads,
        "states": _json_ready(states),
        "edges": _json_ready(edges),
        "active_graph_connected": active_graph_connected(registry),
        "policy": _json_ready(asdict(policy)),
    }
    write_json(epoch_dir / "adaptive_epoch_diagnostics.json", payload)
    return payload


def _assert_epoch_has_samples(
    diagnostics: Dict[str, Any],
    registry: "WindowStateRegistry",
    epoch_dir: Path,
    steps: int,
) -> None:
    """Raise RuntimeError if a non-empty epoch produced zero samples in diagnostics.

    This guard catches the controller-freeze bug where the flat collector reads
    ``epoch_dir/samples.csv`` which does not exist in segmented epochs
    (data lives in ``baseline/`` and ``topup_*/`` subdirectories).  If the epoch
    actually ran (``steps > 0``) but all states report zero samples, the
    diagnostics are corrupt and must not be fed to the controller.

    Also raises if any active state is missing from the diagnostics entirely,
    since a missing state is indistinguishable from a zero-sample state for the
    controller.
    """
    if steps <= 0:
        return

    def _safe_int_sample_count(s: Any) -> int:
        """Return sample_count as int; coerce non-numeric/None to 0."""
        try:
            return int(s.get("sample_count", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _safe_state_id(s: Any) -> Optional[int]:
        """Return state_id as int, or None if absent/non-numeric."""
        raw = s.get("state_id")
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    try:
        total_samples = sum(_safe_int_sample_count(s) for s in diagnostics.get("states", []))
        reported_ids: set = set()
        for s in diagnostics.get("states", []):
            sid = _safe_state_id(s)
            if sid is not None:
                reported_ids.add(sid)
    except Exception as exc:
        raise RuntimeError(
            f"Epoch diagnostics in {epoch_dir} are malformed and cannot be validated: {exc}"
        ) from exc

    active_ids = set(registry.active_state_ids())
    missing_ids = active_ids - reported_ids
    if total_samples == 0 or missing_ids:
        parts = [
            f"Epoch diagnostics in {epoch_dir} are corrupt (controller-freezing data, not a soft quality issue)."
        ]
        if total_samples == 0:
            parts.append(
                f"Zero samples collected across all states despite epoch running {steps} steps. "
                "This typically means the flat collector was used on a segmented epoch dir "
                "(samples live in baseline/ and topup_*/ subdirs, not in the root)."
            )
        if missing_ids:
            parts.append(
                f"Active state(s) missing from diagnostics entirely: {sorted(missing_ids)}."
            )
        raise RuntimeError(" ".join(parts))


def run_scheduled_adaptive_epoch(
    args: Any,
    epoch_dir: Path,
    registry: WindowStateRegistry,
    schedule: Sequence[Dict[str, Any]],
    run_gareus_callable,
    openmm,
    app,
    unit,
    forcefield,
    topology,
    equil_state,
    *,
    progress=None,
    current_seed_bank: Optional[Path] = None,
    policy: Optional[AdaptiveDecisionPolicy] = None,
    runtime_pool: Optional[AdaptiveRuntimePool] = None,
    pool_reserve_ns: float = 0.0,
) -> Dict[str, Any]:
    """Run one allocated epoch as all-state baseline plus top-up subset segments."""
    epoch_dir = Path(epoch_dir)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    policy = policy or AdaptiveDecisionPolicy()
    write_epoch_schedule_files(epoch_dir, schedule)
    active_ids = [int(r["state_id"]) for r in schedule if int(r.get("requested_steps", 0) or 0) > 0]
    if not active_ids:
        raise RuntimeError("adaptive allocation schedule selected no states")
    baseline_steps = min(int(r.get("baseline_steps", 0) or 0) for r in schedule if int(r.get("requested_steps", 0) or 0) > 0)
    baseline_steps = max(1, baseline_steps)
    segment_summaries: List[Dict[str, Any]] = []
    resume_requested = _arg_bool(args, "adaptive_production_resume", False)

    def run_segment(name: str, state_ids: Sequence[int], steps: int) -> Path:
        seg_dir = epoch_dir / name
        seg_dir.mkdir(parents=True, exist_ok=True)
        windows_csv = epoch_dir / f"{name}_windows.csv"
        write_state_subset_window_csv(registry, windows_csv, state_ids, map_path=seg_dir / "epoch_window_map.csv")
        seg_args = copy.copy(args)
        seg_args.out = str(seg_dir)
        seg_args.gamd_production_steps = int(steps)
        seg_args.window_mode = "adaptive"
        seg_args.windows_2d_csv = str(windows_csv)
        seg_args.adaptive_feedback_enabled = False
        seg_args.adaptive_feedback_pilot = False
        seg_args.adaptive_feedback_final_production = False
        seg_args.resume = bool(resume_requested and production_checkpoint_available(seg_dir))
        if seg_args.resume:
            print(f"      scheduled segment {name}: checkpoint manifest found; resuming from {seg_dir}")
        # Derive epoch index from directory name for TUI epoch/topup display.
        _epoch_dir_name = epoch_dir.name  # e.g. "epoch_001"
        try:
            _epoch_idx = int(_epoch_dir_name.rsplit("_", 1)[-1])
        except Exception:
            _epoch_idx = 0
        setattr(seg_args, "_adaptive_phase_info", {
            "is_adaptive_epoch": True,
            "epoch_index": _epoch_idx,
            "epoch_total": "?",
            "segment_name": name,
            "is_topup": name.startswith("topup"),
            "topup_index": int(name.split("_")[1]) if name.startswith("topup") and "_" in name[6:] else 0,
        })
        seed_report = None
        if current_seed_bank is not None and Path(current_seed_bank).exists():
            if bool(getattr(args, "adaptive_production_state_aware_seed_filtering", True)):
                filtered_seed_dir = seg_dir / "filtered_seed_bank"
                seed_report = filter_seed_bank_for_state_ids(
                    Path(current_seed_bank), state_ids, filtered_seed_dir,
                    max_per_state=max(1, int(getattr(args, "adaptive_production_seed_bank_max_per_state", 1) or 1)),
                )
                if seed_report.get("status") == "ok":
                    seg_args.seed_conformers_dir = filtered_seed_dir
                else:
                    seg_args.seed_conformers_dir = Path(current_seed_bank)
            else:
                seg_args.seed_conformers_dir = Path(current_seed_bank)
        # Adaptive-production segments are real sampling, not disposable
        # adaptive-feedback pilots, so preserve coordinate trajectories by
        # default using the normal --traj-interval/--traj-format settings.
        # Users can still disable this explicitly for low-I/O diagnostics.
        if _arg_bool(args, "adaptive_production_trajectories", True) is False:
            seg_args.traj_interval = 0
            seg_args.traj_format = "none"
        requested_steps = int(steps)
        actual_steps = requested_steps
        pool_event = None
        if runtime_pool is not None and runtime_pool.enabled:
            actual_steps = runtime_pool.clip_steps(
                len(state_ids), requested_steps,
                reserve_ns=float(pool_reserve_ns),
                hard_stop=bool(getattr(args, "adaptive_production_pool_hard_stop", True)),
            )
            if actual_steps < requested_steps:
                print(
                    f"      scheduled segment {name}: clipped by MD pool "
                    f"{requested_steps}->{actual_steps} steps for {len(state_ids)} state(s)"
                )
        if actual_steps <= 0:
            print(f"      scheduled segment {name}: skipped because adaptive MD pool is exhausted")
            segment_summaries.append({
                "segment": name,
                "dir": str(seg_dir),
                "windows_csv": str(windows_csv),
                "state_ids": [int(x) for x in state_ids],
                "steps": 0,
                "requested_steps": int(requested_steps),
                "skipped_by_runtime_pool": True,
                "seed_bank": seed_report or {},
            })
            return seg_dir
        seg_args.gamd_production_steps = int(actual_steps)
        print(f"      scheduled segment {name}: {len(state_ids)} state(s), {actual_steps} steps")
        run_gareus_callable(seg_args, seg_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress)
        if _graceful_shutdown.is_set():
            segment_summaries.append({
                "segment": name,
                "dir": str(seg_dir),
                "windows_csv": str(windows_csv),
                "state_ids": [int(x) for x in state_ids],
                "steps": int(actual_steps),
                "requested_steps": int(requested_steps),
                "interrupted_after_checkpoint": True,
                "seed_bank": seed_report or {},
            })
            return seg_dir
        if runtime_pool is not None:
            pool_event = runtime_pool.consume(
                label=f"{epoch_dir.name}/{name}",
                kind="scheduled_final" if "final" in str(epoch_dir) else "scheduled_epoch",
                n_states=len(state_ids),
                steps=int(actual_steps),
                path=seg_dir,
            )
        segment_summaries.append({
            "segment": name,
            "dir": str(seg_dir),
            "windows_csv": str(windows_csv),
            "state_ids": [int(x) for x in state_ids],
            "steps": int(actual_steps),
            "requested_steps": int(requested_steps),
            "runtime_pool_event": pool_event or {},
            "seed_bank": seed_report or {},
        })
        return seg_dir

    run_segment("baseline", active_ids, baseline_steps)
    if _graceful_shutdown.is_set():
        payload = {
            "schema_version": "adaptive_scheduled_epoch_v1",
            "status": "interrupted_after_checkpoint",
            "epoch_dir": str(epoch_dir),
            "baseline_steps": int(baseline_steps),
            "segments": segment_summaries,
            "diagnostics_json": "",
            "schedule_csv": str(epoch_dir / "epoch_schedule.csv"),
        }
        write_json(epoch_dir / "scheduled_epoch_summary.json", payload)
        return {"summary": payload, "diagnostics": {}}
    groups: Dict[int, List[int]] = {}
    for row in schedule:
        extra = _quantized_extra_steps(int(row.get("requested_steps", 0) or 0) - baseline_steps)
        if extra <= 0:
            continue
        groups.setdefault(extra, []).append(int(row["state_id"]))
    for idx, (extra_steps, state_ids) in enumerate(sorted(groups.items()), start=1):
        run_segment(f"topup_{idx:03d}_{extra_steps}", state_ids, extra_steps)
        if _graceful_shutdown.is_set():
            payload = {
                "schema_version": "adaptive_scheduled_epoch_v1",
                "status": "interrupted_after_checkpoint",
                "epoch_dir": str(epoch_dir),
                "baseline_steps": int(baseline_steps),
                "segments": segment_summaries,
                "diagnostics_json": "",
                "schedule_csv": str(epoch_dir / "epoch_schedule.csv"),
            }
            write_json(epoch_dir / "scheduled_epoch_summary.json", payload)
            return {"summary": payload, "diagnostics": {}}
    diagnostics = collect_segmented_epoch_diagnostics(epoch_dir, registry, policy)
    payload = {
        "schema_version": "adaptive_scheduled_epoch_v1",
        "epoch_dir": str(epoch_dir),
        "baseline_steps": int(baseline_steps),
        "segments": segment_summaries,
        "diagnostics_json": str(epoch_dir / "adaptive_epoch_diagnostics.json"),
        "schedule_csv": str(epoch_dir / "epoch_schedule.csv"),
    }
    write_json(epoch_dir / "scheduled_epoch_summary.json", payload)
    return {"summary": payload, "diagnostics": diagnostics}




class AdaptiveProductionController:
    """Base controller for custom in-process adaptive production.

    The auto-loop below uses ``run_gareus`` externally per epoch, but this class
    remains useful for future in-process Context recycling.
    """

    def __init__(self, registry: WindowStateRegistry, epoch_steps: int = 50000, registry_dir: Optional[Path] = None):
        self.registry = registry
        self.epoch_steps = int(epoch_steps)
        self.registry_dir = None if registry_dir is None else Path(registry_dir)
        self.current_epoch = 0

    def run(self, max_epochs: Optional[int] = None) -> None:
        epoch = 0
        while max_epochs is None or epoch < int(max_epochs):
            self.current_epoch = epoch
            self.run_epoch(epoch)
            if self.converged():
                break
            epoch += 1

    def run_epoch(self, epoch: int) -> None:
        self.perform_md_sampling(epoch)
        diagnostics = self.analyze_epoch(epoch)
        actions = self.propose_actions(epoch, diagnostics)
        self.apply_actions(epoch, actions)
        if self.registry_dir is not None:
            self.registry.save(self.registry_dir)

    def apply_actions(self, epoch: int, actions: Sequence[Tuple]) -> None:
        for action in actions:
            kind = str(action[0])
            if kind == "retire":
                _, state_id, reason = action
                self.registry.retire_state(int(state_id), int(epoch) + 1, str(reason))
            elif kind == "extend":
                _, state_id, reason = action
                self.registry.record_extend(int(state_id), int(epoch) + 1, str(reason))
            elif kind == "add":
                _, parent, params, reason = action
                self.registry.add_state(
                    primary_center=params[0], primary_k=params[1],
                    secondary_center=params[2] if len(params) > 2 else None,
                    secondary_k=params[3] if len(params) > 3 else None,
                    parent_state_id=None if parent is None else int(parent),
                    epoch=int(epoch) + 1, source="adaptive_production", reason=str(reason),
                )
            elif kind == "split":
                _, parent, children_params, reason = action
                self.registry.retire_state(int(parent), int(epoch) + 1, f"split: {reason}")
                for child in children_params:
                    self.registry.add_state(
                        primary_center=child[0], primary_k=child[1],
                        secondary_center=child[2] if len(child) > 2 else None,
                        secondary_k=child[3] if len(child) > 3 else None,
                        parent_state_id=int(parent), epoch=int(epoch) + 1,
                        source="adaptive_production_split", reason=f"split child: {reason}",
                    )
            else:
                raise ValueError(f"unknown adaptive-production action {kind!r}")

    def perform_md_sampling(self, epoch: int) -> None:
        raise NotImplementedError

    def analyze_epoch(self, epoch: int) -> Dict[str, Any]:
        raise NotImplementedError

    def propose_actions(self, epoch: int, diagnostics: Dict[str, Any]) -> List[Tuple]:
        return []

    def converged(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Existing-engine epoch driver
# ---------------------------------------------------------------------------


def _arg_bool(args: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(args, name, default))


def _arg_int(args: Any, name: str, default: int) -> int:
    try:
        return int(getattr(args, name, default))
    except Exception:
        return int(default)


def _arg_float(args: Any, name: str, default: float) -> float:
    try:
        return float(getattr(args, name, default))
    except Exception:
        return float(default)


def policy_from_args(args: Any) -> AdaptiveDecisionPolicy:
    return AdaptiveDecisionPolicy(
        target_overlap=_arg_float(args, "adaptive_production_target_overlap", _arg_float(args, "adaptive_feedback_target_overlap", 0.25)),
        min_exchange_acceptance=_arg_float(args, "adaptive_production_min_exchange", 0.08),
        min_samples_for_add=_arg_int(args, "adaptive_production_min_samples", 50),
        min_samples_for_retire=_arg_int(args, "adaptive_production_retire_min_samples", 200),
        max_new_windows_per_epoch=_arg_int(args, "adaptive_production_max_new_windows_per_epoch", 4),
        retire_converged=_arg_bool(args, "adaptive_production_retire_converged", False),
        max_gamd_boost_sd_kcal_mol=_arg_float(args, "adaptive_production_max_gamd_boost_sd_kcal_mol", 6.0),
        final_connectivity_required=_arg_bool(args, "adaptive_production_final_connectivity_required", True),
        final_min_samples_per_state=_arg_int(args, "adaptive_production_final_min_samples_per_state", 100),
        quality_min_primary_coverage_fraction=_arg_float(args, "adaptive_production_quality_min_primary_coverage_fraction", 0.25),
        quality_hard_fail=_arg_bool(args, "adaptive_production_quality_hard_fail", False),
        final_quality_extension_rounds=_arg_int(args, "adaptive_production_final_quality_extension_rounds", 0),
        final_quality_extension_steps=_arg_int(args, "adaptive_production_final_quality_extension_steps", 0),
        propagate_seed_bank=_arg_bool(args, "adaptive_production_propagate_seed_bank", True),
        seed_bank_max_per_state=_arg_int(args, "adaptive_production_seed_bank_max_per_state", 1),
        allocation_scheduler=_arg_bool(args, "adaptive_production_allocation_scheduler", True),
        epoch_step_budget=_arg_int(args, "adaptive_production_epoch_step_budget", 0),
        min_state_steps=_arg_int(args, "adaptive_production_min_state_steps", 0),
        max_state_steps=_arg_int(args, "adaptive_production_max_state_steps", 0),
        new_state_steps=_arg_int(args, "adaptive_production_new_state_steps", 0),
        final_allocation_scheduler=_arg_bool(args, "adaptive_production_final_allocation_scheduler", True),
        final_step_budget=_arg_int(args, "adaptive_production_final_step_budget", 0),
        state_aware_seed_filtering=_arg_bool(args, "adaptive_production_state_aware_seed_filtering", True),
        scheduled_final_segments=_arg_bool(args, "adaptive_production_scheduled_final_segments", True),
        convergence_min_samples_per_state=_arg_int(args, "adaptive_production_convergence_min_samples_per_state", 50),
        convergence_max_weak_edges=_arg_int(args, "adaptive_production_convergence_max_weak_edges", 0),
        convergence_allow_extend_actions=_arg_bool(args, "adaptive_production_convergence_allow_extend_actions", True),
        require_convergence_before_final=_arg_bool(args, "adaptive_production_require_convergence_before_final", False),
        total_md_pool_ns=_arg_float(args, "adaptive_production_total_md_pool_ns", 0.0),
        final_pool_fraction=_arg_float(args, "adaptive_production_final_pool_fraction", 0.50),
        min_final_pool_ns=_arg_float(args, "adaptive_production_min_final_pool_ns", 0.0),
        pool_hard_stop=_arg_bool(args, "adaptive_production_pool_hard_stop", True),
        context_reuse=_arg_bool(args, "adaptive_production_context_reuse", False),
        context_reuse_require=_arg_bool(args, "adaptive_production_context_reuse_require", False),
        context_reuse_mode=str(getattr(args, "adaptive_production_context_reuse_mode", "off") or "off"),
    )


def run_adaptive_production_auto_loop(args, out_dir: Path, openmm, app, unit, forcefield, topology, equil_state, progress=None) -> Dict[str, Any]:
    """Run adaptive production by calling the existing GAREUS worker per epoch.

    This is the first fully wired version: it reuses ``run_gareus`` for each
    fixed-window epoch, persists a registry after each epoch, proposes midpoint
    windows at weak edges, and finally runs a frozen final-production phase.
    """
    from .production import run_gareus

    out_dir = Path(out_dir)
    adaptive_dir = out_dir / "adaptive_production"
    adaptive_dir.mkdir(parents=True, exist_ok=True)
    policy = policy_from_args(args)
    max_epochs = max(1, _arg_int(args, "adaptive_production_epochs", 3))
    # Explicit overrides take precedence. When unset (0), epoch/final steps are
    # computed dynamically per-epoch from the pool balance and actual state count.
    # The fallback (gamd_production_steps//20) is used only when the pool is disabled.
    _explicit_epoch_steps = _arg_int(args, "adaptive_production_epoch_steps", 0)
    _explicit_final_steps = _arg_int(args, "adaptive_production_final_steps", 0)
    _timestep_fs = float(getattr(args, "timestep_fs", 2.0) or 2.0)
    _gamd_fallback = max(1, int(getattr(args, "gamd_production_steps", 100000) or 100000) // 20)
    # epoch_steps / final_steps used only as fallback when pool is disabled or for
    # the scheduled-epoch path which needs a default_steps before the loop.
    epoch_steps = int(_explicit_epoch_steps) if _explicit_epoch_steps > 0 else _gamd_fallback
    final_steps = int(_explicit_final_steps) if _explicit_final_steps > 0 else _gamd_fallback
    use_epoch_samples_for_mbar = _arg_bool(args, "adaptive_production_use_epoch_samples_for_mbar", False)
    global_shared_gamd_dir: Optional[Path] = None
    if _arg_bool(args, "adaptive_production_global_shared_gamd", True):
        requested_shared_dir = str(getattr(args, "shared_gamd_setup_dir", "") or "").strip()
        global_shared_gamd_dir = Path(requested_shared_dir) if requested_shared_dir else adaptive_dir / "global_shared_gamd_setup"
        global_shared_gamd_dir.mkdir(parents=True, exist_ok=True)
        # These private attrs are propagated by copy.copy(args) into every
        # epoch/baseline/topup/final worker.  The first worker exports the setup;
        # later workers reuse it.  This keeps GaMD thresholds/statistics global
        # over the whole adaptive-production campaign.
        setattr(args, "_global_shared_gamd_setup_dir", str(global_shared_gamd_dir))
        setattr(args, "_global_shared_gamd_export_dir", str(global_shared_gamd_dir))
        if not str(getattr(args, "shared_gamd_setup_dir", "") or "").strip():
            setattr(args, "shared_gamd_setup_dir", str(global_shared_gamd_dir))
        if not str(getattr(args, "shared_gamd_export_dir", "") or "").strip():
            setattr(args, "shared_gamd_export_dir", str(global_shared_gamd_dir))
        write_json(adaptive_dir / "global_shared_gamd_setup_policy.json", {
            "schema_version": "adaptive_production_global_shared_gamd_policy_v1",
            "enabled": True,
            "global_shared_gamd_dir": str(global_shared_gamd_dir),
            "behavior": "First adaptive-production worker calibrates shared GaMD and exports it here; all later epoch, scheduled topup, final, and final-extension workers reuse the same setup.",
        })
    current_seed_bank: Optional[Path] = None
    if bool(policy.propagate_seed_bank) and getattr(args, "seed_conformers_dir", None) is not None:
        current_seed_bank = Path(getattr(args, "seed_conformers_dir"))

    resume_requested = _arg_bool(args, "adaptive_production_resume", False)
    runtime_pool, runtime_pool_resume_validation = _load_or_initialize_runtime_pool(adaptive_dir, args, policy, resume_requested=resume_requested)
    context_reuse_readiness = evaluate_context_reuse_readiness(args, adaptive_dir, registry=None)
    registry: Optional[WindowStateRegistry] = None
    current_windows_csv: Optional[Path] = Path(str(args.windows_2d_csv)) if getattr(args, "windows_2d_csv", None) else None
    epoch_summaries: List[Dict[str, Any]] = []
    start_epoch = 0
    previous_diagnostics: Optional[Dict[str, Any]] = None

    summary_path = adaptive_dir / "adaptive_production_driver_summary.json"
    registry_path = adaptive_dir / "state_registry.json"
    if resume_requested and registry_path.exists():
        registry = WindowStateRegistry.load(adaptive_dir)
        old_summary = read_json_file(summary_path, {}) if summary_path.exists() else {}
        epoch_summaries = list(old_summary.get("epoch_summaries", []) or []) if isinstance(old_summary, dict) else []
        start_epoch = int(old_summary.get("epochs_completed", len(epoch_summaries)) or len(epoch_summaries)) if isinstance(old_summary, dict) else len(epoch_summaries)
        current_windows_csv = adaptive_dir / f"windows_epoch_{start_epoch:03d}.csv"
        if not current_windows_csv.exists():
            current_windows_csv = adaptive_dir / "resume_active_windows.csv"
            registry.write_active_window_csv(current_windows_csv, map_path=adaptive_dir / f"window_map_epoch_{start_epoch:03d}.csv")
        print(f"    Adaptive-production resume: loaded {len(registry.all_states())} states; continuing at epoch {start_epoch}.")
        context_reuse_readiness = evaluate_context_reuse_readiness(args, adaptive_dir, registry=registry)

    for epoch in range(start_epoch, max_epochs):
        epoch_dir = adaptive_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        scheduled_summary = None
        # For both scheduled and unscheduled paths: recompute default steps from pool
        # when pool is enabled and no explicit override, so each epoch gets its fair share.
        if _explicit_epoch_steps <= 0 and runtime_pool.enabled:
            _sched_n = _estimate_active_state_count(args, registry, current_windows_csv)
            _epochs_left = max(1, max_epochs - epoch)
            _pool_epoch_steps = _epoch_steps_from_pool(runtime_pool, policy, _sched_n, _epochs_left, _timestep_fs)
            if _pool_epoch_steps > 0:
                epoch_steps = _pool_epoch_steps
        if registry is not None and bool(policy.allocation_scheduler):
            schedule_policy = _policy_with_pool_step_budget(
                registry, policy, runtime_pool, default_steps=epoch_steps, final=False
            )
            schedule = build_adaptive_epoch_schedule(
                registry,
                previous_diagnostics,
                schedule_policy,
                epoch=epoch,
                default_steps=epoch_steps,
                final=False,
            )
            print(
                f"    Adaptive-production scheduled epoch {epoch + 1}/{max_epochs}: "
                f"{len(schedule)} active states -> {epoch_dir}"
            )
            result = run_scheduled_adaptive_epoch(
                args,
                epoch_dir,
                registry,
                schedule,
                run_gareus,
                openmm,
                app,
                unit,
                forcefield,
                topology,
                equil_state,
                progress=progress,
                current_seed_bank=current_seed_bank,
                policy=schedule_policy,
                runtime_pool=runtime_pool,
                pool_reserve_ns=_runtime_pool_final_reserve_ns(runtime_pool, policy, final=False),
            )
            diagnostics = result["diagnostics"]
            scheduled_summary = result["summary"]
            if _graceful_shutdown.is_set():
                _write_runtime_pool_reports(adaptive_dir, runtime_pool)
                payload = {
                    "schema_version": "adaptive_production_driver_summary_v1",
                    "status": "interrupted_after_checkpoint",
                    "interrupted_segment": str(epoch_dir),
                    "epochs_completed": int(start_epoch + len(epoch_summaries)),
                    "epoch_summaries": _json_ready(epoch_summaries),
                    "scheduled_epoch": _json_ready(scheduled_summary),
                }
                write_json(summary_path, payload)
                return payload
        else:
            epoch_args = copy.copy(args)
            epoch_args.out = str(epoch_dir)
            setattr(epoch_args, "_adaptive_phase_info", {
                "is_adaptive_epoch": True,
                "epoch_index": int(epoch),
                "epoch_total": int(max_epochs),
                "segment_name": "epoch",
                "is_topup": False,
                "topup_index": 0,
                "prev_epochs": [
                    {
                        "epoch": int(s["epoch"]),
                        "stop_adaptive": bool(s.get("convergence_gate", {}).get("stop_adaptive", False)),
                        "n_issues": len(s.get("convergence_gate", {}).get("issues", [])),
                        "n_active": int(s.get("n_active_after_epoch", 0)),
                    }
                    for s in epoch_summaries
                ],
            })
            epoch_state_estimate = _estimate_active_state_count(args, registry, current_windows_csv)
            # Compute fair per-epoch target steps from pool when no explicit override.
            # Distributes remaining epoch budget evenly across remaining epochs so each
            # epoch gets its share regardless of how many states were active at config time.
            if _explicit_epoch_steps <= 0 and runtime_pool.enabled:
                epochs_remaining = max(1, max_epochs - epoch)
                _target_epoch_steps = _epoch_steps_from_pool(
                    runtime_pool, policy, epoch_state_estimate, epochs_remaining, _timestep_fs
                )
                if _target_epoch_steps > 0:
                    epoch_steps = _target_epoch_steps
            actual_epoch_steps = runtime_pool.clip_steps(
                epoch_state_estimate,
                int(epoch_steps),
                reserve_ns=_runtime_pool_final_reserve_ns(runtime_pool, policy, final=False),
                hard_stop=bool(policy.pool_hard_stop),
            )
            if actual_epoch_steps <= 0:
                if registry is None:
                    raise RuntimeError("adaptive-production MD pool is exhausted before the first epoch could initialize a state registry")
                print("    Adaptive-production MD pool exhausted before next unscheduled epoch; entering finalization.")
                break
            epoch_args.gamd_production_steps = int(actual_epoch_steps)
            epoch_args.window_mode = "adaptive"
            epoch_args.adaptive_feedback_enabled = False
            epoch_args.adaptive_feedback_pilot = False
            epoch_args.adaptive_feedback_final_production = False
            epoch_args.resume = bool(resume_requested and production_checkpoint_available(epoch_dir))
            if epoch_args.resume:
                print(f"    Adaptive-production epoch {epoch + 1}/{max_epochs}: checkpoint manifest found; resuming from {epoch_dir}")
            if current_windows_csv is not None:
                epoch_args.windows_2d_csv = str(current_windows_csv)
            if bool(policy.propagate_seed_bank) and current_seed_bank is not None and Path(current_seed_bank).exists():
                epoch_args.seed_conformers_dir = Path(current_seed_bank)
            # Adaptive-production epochs are real sampling, so inherit
            # --traj-interval/--traj-format by default.  The separate
            # adaptive_pilot_trajectories switch remains limited to feedback
            # pilots.
            if _arg_bool(args, "adaptive_production_trajectories", True) is False:
                epoch_args.traj_interval = 0
                epoch_args.traj_format = "none"

            if actual_epoch_steps != int(epoch_steps):
                print(
                    f"    Adaptive-production epoch {epoch + 1}/{max_epochs}: "
                    f"clipped by MD pool {epoch_steps}->{actual_epoch_steps} steps -> {epoch_dir}"
                )
            else:
                print(f"    Adaptive-production epoch {epoch + 1}/{max_epochs}: {actual_epoch_steps} steps -> {epoch_dir}")
            if current_windows_csv is not None:
                print(f"      active window table: {current_windows_csv}")
            run_gareus(epoch_args, epoch_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress)
            if _graceful_shutdown.is_set():
                _write_runtime_pool_reports(adaptive_dir, runtime_pool)
                payload = {
                    "schema_version": "adaptive_production_driver_summary_v1",
                    "status": "interrupted_after_checkpoint",
                    "interrupted_segment": str(epoch_dir),
                    "epochs_completed": int(start_epoch + len(epoch_summaries)),
                    "epoch_summaries": _json_ready(epoch_summaries),
                }
                write_json(summary_path, payload)
                return payload

            if registry is None:
                table = find_best_window_table(epoch_dir)
                if table is None:
                    raise RuntimeError(f"epoch {epoch} completed but no umbrella window table was found in {epoch_dir}")
                registry = registry_from_window_csv(table, epoch=0, source="epoch0_windows")
                registry.write_epoch_window_map(epoch_dir / "epoch_window_map.csv")
            runtime_pool.consume(
                label=f"epoch_{epoch:03d}",
                kind="adaptive_epoch",
                n_states=max(1, len(registry.active_states())),
                steps=int(actual_epoch_steps),
                path=epoch_dir,
            )
            diagnostics = collect_segmented_epoch_diagnostics(epoch_dir, registry, policy)
            _assert_epoch_has_samples(diagnostics, registry, epoch_dir, int(actual_epoch_steps))
        actions = propose_actions_from_diagnostics(registry, diagnostics, policy=policy)
        action_report = None
        if _arg_bool(args, "adaptive_production_write_action_reports", True):
            try:
                action_report = write_epoch_action_report(epoch_dir, epoch, registry, diagnostics, actions, policy)
            except Exception as exc:
                # Diagnostics reports are useful, but they must never invalidate
                # a completed MD epoch. Keep going and record the failure in the
                # driver summary.
                action_report = {"status": "write_failed", "error": str(exc)}
                print(f"WARNING: failed to write adaptive epoch action report for epoch {epoch}: {exc}")
        convergence_gate = evaluate_adaptive_convergence_gate(
            epoch_dir,
            epoch,
            registry,
            diagnostics,
            actions,
            policy,
        )

        seed_bank_report = None
        if bool(policy.propagate_seed_bank):
            seed_bank_dir = adaptive_dir / f"seed_bank_epoch_{epoch:03d}"
            if scheduled_summary is not None:
                seg_dirs = [Path(s.get("dir")) for s in scheduled_summary.get("segments", []) if s.get("dir")]
                seed_bank_report = write_seed_bank_from_run_dirs(
                    seg_dirs,
                    seed_bank_dir,
                    registry,
                    source_label=f"epoch_{epoch:03d}",
                    max_per_state=max(1, int(policy.seed_bank_max_per_state or 1)),
                )
            else:
                seed_bank_report = write_epoch_seed_bank(
                    epoch_dir,
                    seed_bank_dir,
                    registry,
                    source_label=f"epoch_{epoch:03d}",
                    max_per_state=max(1, int(policy.seed_bank_max_per_state or 1)),
                )
            if seed_bank_report.get("status") == "ok":
                current_seed_bank = seed_bank_dir

        _apply_registry_actions(registry, actions, epoch)
        registry_paths = registry.save(adaptive_dir)
        runtime_pool_paths = _write_runtime_pool_reports(adaptive_dir, runtime_pool)

        next_csv = adaptive_dir / f"windows_epoch_{epoch + 1:03d}.csv"
        next_map = adaptive_dir / f"window_map_epoch_{epoch + 1:03d}.csv"
        registry.write_active_window_csv(next_csv, map_path=next_map)
        current_windows_csv = next_csv
        # Also write the next map into the next epoch dir just before it runs.
        next_epoch_dir = adaptive_dir / f"epoch_{epoch + 1:03d}"
        next_epoch_dir.mkdir(parents=True, exist_ok=True)
        registry.write_epoch_window_map(next_epoch_dir / "epoch_window_map.csv")

        summary = {
            "epoch": int(epoch),
            "epoch_dir": str(epoch_dir),
            "n_active_after_epoch": int(len(registry.active_states())),
            "n_total_states": int(len(registry.all_states())),
            "diagnostics_json": str(epoch_dir / "adaptive_epoch_diagnostics.json"),
            "actions": [_action_to_dict(a) for a in actions],
            "action_report": action_report or {},
            "convergence_gate": convergence_gate or {},
            "seed_bank": seed_bank_report or {},
            "scheduled_epoch": scheduled_summary or {},
            "next_windows_csv": str(next_csv),
            "registry": registry_paths,
            "runtime_pool": runtime_pool.to_dict(),
            "runtime_pool_reports": runtime_pool_paths,
        }
        previous_diagnostics = diagnostics
        epoch_summaries.append(summary)
        write_json(adaptive_dir / "adaptive_production_driver_summary.json", {
            "schema_version": "adaptive_production_driver_summary_v1",
            "status": "running",
            "epochs_completed": int(epoch + 1),
            "epoch_summaries": epoch_summaries,
            "global_shared_gamd_setup_dir": str(global_shared_gamd_dir) if global_shared_gamd_dir is not None else "",
            "global_shared_gamd_enabled": bool(global_shared_gamd_dir is not None),
        })

        if bool(convergence_gate.get("stop_adaptive", False)):
            print(
                f"    Adaptive-production convergence gate passed after epoch {epoch + 1}: "
                f"{convergence_gate.get('status')}"
            )
            break

        if epoch + 1 >= max_epochs and bool(policy.require_convergence_before_final):
            raise RuntimeError(
                "adaptive-production reached the maximum number of adaptive epochs without passing the convergence gate; "
                f"see {convergence_gate.get('json')}"
            )

    if registry is None:
        raise RuntimeError("adaptive production ended without a registry")

    if bool(policy.final_connectivity_required) and not active_graph_connected(registry):
        raise RuntimeError(
            "adaptive-production final phase refused to start because the active window graph is disconnected; "
            "disable --no-adaptive-production-final-connectivity-required only for debugging."
        )

    final_dir = adaptive_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_windows_csv = adaptive_dir / "final_active_windows.csv"
    final_registry_csv = adaptive_dir / "final_registry_used_for_mbar.csv"
    registry.write_active_window_csv(final_windows_csv, map_path=final_dir / "epoch_window_map.csv")
    registry.write_state_csv(final_registry_csv)
    final_schedule_files = {}
    final_schedule = None
    if bool(policy.final_allocation_scheduler):
        final_schedule_policy = _policy_with_pool_step_budget(
            registry, policy, runtime_pool, default_steps=final_steps, final=True
        )
        final_schedule = build_adaptive_epoch_schedule(
            registry,
            previous_diagnostics,
            final_schedule_policy,
            epoch=max_epochs,
            default_steps=final_steps,
            final=True,
        )
        final_schedule_files = write_epoch_schedule_files(final_dir, final_schedule, prefix="final_state_schedule")

    final_seed_bank = None
    final_scheduled_summary = None
    if bool(policy.scheduled_final_segments) and bool(policy.final_allocation_scheduler) and final_schedule is not None:
        print(f"    Adaptive-production final frozen scheduled phase -> {final_dir}")
        print(f"      final active window table: {final_windows_csv}")
        result = run_scheduled_adaptive_epoch(
            args,
            final_dir,
            registry,
            final_schedule,
            run_gareus,
            openmm,
            app,
            unit,
            forcefield,
            topology,
            equil_state,
            progress=progress,
            current_seed_bank=current_seed_bank,
            policy=final_schedule_policy if 'final_schedule_policy' in locals() else policy,
            runtime_pool=runtime_pool,
            pool_reserve_ns=0.0,
        )
        final_scheduled_summary = result.get("summary", {})
        if bool(policy.propagate_seed_bank):
            seg_dirs = [Path(s.get("dir")) for s in final_scheduled_summary.get("segments", []) if s.get("dir")]
            final_seed_bank = write_seed_bank_from_run_dirs(
                seg_dirs,
                adaptive_dir / "seed_bank_final",
                registry,
                source_label="final",
                max_per_state=max(1, int(policy.seed_bank_max_per_state or 1)),
            )
            if final_seed_bank.get("status") == "ok":
                current_seed_bank = adaptive_dir / "seed_bank_final"
    else:
        final_args = copy.copy(args)
        final_args.out = str(final_dir)
        final_n_states = max(1, len(registry.active_states()))
        # Compute final steps from pool when no explicit override: spend all remaining budget.
        if _explicit_final_steps <= 0 and runtime_pool.enabled:
            _target_final = _final_steps_from_pool(runtime_pool, final_n_states, _timestep_fs)
            if _target_final > 0:
                final_steps = _target_final
        actual_final_steps = runtime_pool.clip_steps(
            final_n_states,
            int(final_steps),
            reserve_ns=0.0,
            hard_stop=bool(policy.pool_hard_stop),
        )
        if actual_final_steps <= 0:
            if bool(policy.pool_hard_stop):
                raise RuntimeError("adaptive-production MD pool exhausted before frozen final production could run")
            actual_final_steps = int(final_steps)
        final_args.gamd_production_steps = int(actual_final_steps)
        final_args.window_mode = "adaptive"
        final_args.windows_2d_csv = str(final_windows_csv)
        final_args.adaptive_feedback_enabled = False
        final_args.adaptive_feedback_pilot = False
        final_args.adaptive_feedback_final_production = False
        final_args.resume = bool(resume_requested and production_checkpoint_available(final_dir))
        if final_args.resume:
            print(f"    Adaptive-production final frozen phase: checkpoint manifest found; resuming from {final_dir}")
        if _arg_bool(args, "adaptive_production_trajectories", True) is False:
            final_args.traj_interval = 0
            final_args.traj_format = "none"
        if bool(policy.propagate_seed_bank) and current_seed_bank is not None and Path(current_seed_bank).exists():
            if bool(policy.state_aware_seed_filtering):
                filtered_final_seed_dir = final_dir / "filtered_seed_bank"
                seed_filter_report = filter_seed_bank_for_state_ids(
                    Path(current_seed_bank), registry.active_state_ids(), filtered_final_seed_dir,
                    max_per_state=max(1, int(policy.seed_bank_max_per_state or 1)),
                )
                final_args.seed_conformers_dir = filtered_final_seed_dir if seed_filter_report.get("status") == "ok" else Path(current_seed_bank)
            else:
                final_args.seed_conformers_dir = Path(current_seed_bank)
        if actual_final_steps != int(final_steps):
            print(f"    Adaptive-production final frozen phase: clipped by MD pool {final_steps}->{actual_final_steps} steps -> {final_dir}")
        else:
            print(f"    Adaptive-production final frozen phase: {actual_final_steps} steps -> {final_dir}")
        print(f"      final active window table: {final_windows_csv}")
        run_gareus(final_args, final_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress)
        if _graceful_shutdown.is_set():
            _write_runtime_pool_reports(adaptive_dir, runtime_pool)
            payload = {
                "schema_version": "adaptive_production_driver_summary_v1",
                "status": "interrupted_after_checkpoint",
                "interrupted_segment": str(final_dir),
                "epochs_completed": int(start_epoch + len(epoch_summaries)),
                "epoch_summaries": _json_ready(epoch_summaries),
            }
            write_json(summary_path, payload)
            return payload
        runtime_pool.consume(
            label="final",
            kind="final",
            n_states=max(1, len(registry.active_states())),
            steps=int(actual_final_steps),
            path=final_dir,
        )
        _write_runtime_pool_reports(adaptive_dir, runtime_pool)
        collect_segmented_epoch_diagnostics(final_dir, registry, policy)
        if bool(policy.propagate_seed_bank):
            final_seed_bank = write_epoch_seed_bank(
                final_dir,
                adaptive_dir / "seed_bank_final",
                registry,
                source_label="final",
                max_per_state=max(1, int(policy.seed_bank_max_per_state or 1)),
            )
            if final_seed_bank.get("status") == "ok":
                current_seed_bank = adaptive_dir / "seed_bank_final"

    _write_runtime_pool_reports(adaptive_dir, runtime_pool)

    # Optional frozen-final extension rounds.  These do not change the active
    # window set; they simply add more final-phase samples when the quality gate
    # says the frozen phase is under-sampled.
    extension_summaries: List[Dict[str, Any]] = []
    final_diag = collect_final_combined_diagnostics(adaptive_dir, registry, policy=policy)
    quality_gate = evaluate_adaptive_quality_gate(
        adaptive_dir, registry, final_diag, policy=policy, output_prefix="adaptive_quality_gate_pre_union"
    )
    ext_rounds = max(0, int(policy.final_quality_extension_rounds or 0))
    ext_steps = int(policy.final_quality_extension_steps or final_steps)
    for ext_index in range(ext_rounds):
        if quality_gate.get("status") not in {"needs_more_sampling"}:
            break
        ext_dir = adaptive_dir / f"final_extension_{ext_index + 1:03d}"
        ext_dir.mkdir(parents=True, exist_ok=True)
        ext_args = copy.copy(args)
        ext_args.out = str(ext_dir)
        actual_ext_steps = runtime_pool.clip_steps(
            max(1, len(registry.active_states())),
            int(ext_steps),
            reserve_ns=0.0,
            hard_stop=bool(policy.pool_hard_stop),
        )
        if actual_ext_steps <= 0:
            print("    Adaptive-production MD pool exhausted before frozen-final extension; stopping extensions.")
            break
        ext_args.gamd_production_steps = int(actual_ext_steps)
        ext_args.window_mode = "adaptive"
        ext_args.windows_2d_csv = str(final_windows_csv)
        ext_args.adaptive_feedback_enabled = False
        ext_args.adaptive_feedback_pilot = False
        ext_args.adaptive_feedback_final_production = False
        ext_args.resume = bool(resume_requested and production_checkpoint_available(ext_dir))
        if ext_args.resume:
            print(f"    Adaptive-production frozen final extension {ext_index + 1}/{ext_rounds}: checkpoint manifest found; resuming from {ext_dir}")
        if _arg_bool(args, "adaptive_production_trajectories", True) is False:
            ext_args.traj_interval = 0
            ext_args.traj_format = "none"
        if bool(policy.propagate_seed_bank) and current_seed_bank is not None and Path(current_seed_bank).exists():
            if bool(policy.state_aware_seed_filtering):
                filtered_ext_seed_dir = ext_dir / "filtered_seed_bank"
                seed_filter_report = filter_seed_bank_for_state_ids(
                    Path(current_seed_bank), registry.active_state_ids(), filtered_ext_seed_dir,
                    max_per_state=max(1, int(policy.seed_bank_max_per_state or 1)),
                )
                ext_args.seed_conformers_dir = filtered_ext_seed_dir if seed_filter_report.get("status") == "ok" else Path(current_seed_bank)
            else:
                ext_args.seed_conformers_dir = Path(current_seed_bank)
        registry.write_epoch_window_map(ext_dir / "epoch_window_map.csv")
        print(
            f"    Adaptive-production frozen final extension {ext_index + 1}/{ext_rounds}: "
            f"{actual_ext_steps} steps -> {ext_dir}"
        )
        run_gareus(ext_args, ext_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress)
        if _graceful_shutdown.is_set():
            _write_runtime_pool_reports(adaptive_dir, runtime_pool)
            payload = {
                "schema_version": "adaptive_production_driver_summary_v1",
                "status": "interrupted_after_checkpoint",
                "interrupted_segment": str(ext_dir),
                "epochs_completed": int(start_epoch + len(epoch_summaries)),
                "epoch_summaries": _json_ready(epoch_summaries),
            }
            write_json(summary_path, payload)
            return payload
        runtime_pool.consume(
            label=f"final_extension_{ext_index + 1:03d}",
            kind="final_extension",
            n_states=max(1, len(registry.active_states())),
            steps=int(actual_ext_steps),
            path=ext_dir,
        )
        _write_runtime_pool_reports(adaptive_dir, runtime_pool)
        ext_diag = collect_segmented_epoch_diagnostics(ext_dir, registry, policy)
        final_diag = collect_final_combined_diagnostics(adaptive_dir, registry, policy=policy)
        quality_gate = evaluate_adaptive_quality_gate(
            adaptive_dir,
            registry,
            final_diag,
            policy=policy,
            output_prefix=f"adaptive_quality_gate_after_extension_{ext_index + 1:03d}",
        )
        extension_summaries.append({
            "extension": int(ext_index + 1),
            "dir": str(ext_dir),
            "steps": int(actual_ext_steps),
            "requested_steps": int(ext_steps),
            "diagnostics_json": str(ext_dir / "adaptive_epoch_diagnostics.json"),
            "combined_diagnostics_json": str(adaptive_dir / "adaptive_final_combined_diagnostics.json"),
            "quality_gate": {"status": quality_gate.get("status"), "json": quality_gate.get("json"), "md": quality_gate.get("md")},
        })

    final_registry_paths = registry.save(adaptive_dir)
    union_inputs = None
    union_analysis = None
    if _arg_bool(args, "adaptive_production_write_union_mbar_inputs", True):
        try:
            union_inputs = build_union_state_mbar_inputs(
                adaptive_dir,
                registry,
                include_epochs=use_epoch_samples_for_mbar,
                output_prefix="adaptive_union_mbar",
            )
        except Exception as exc:
            print(f"WARNING: adaptive-production union MBAR input builder failed: {exc}")
            union_inputs = {"error": str(exc)}
        if union_inputs and not union_inputs.get("error") and _arg_bool(args, "adaptive_production_run_union_mbar_analysis", True):
            try:
                union_analysis = run_union_mbar_analysis(
                    adaptive_dir,
                    union_inputs,
                    output_prefix="adaptive_union_mbar_analysis",
                    fes_bins=getattr(args, "adaptive_production_union_fes_bins", "80,40"),
                )
            except Exception as exc:
                print(f"WARNING: adaptive-production union MBAR analysis failed: {exc}")
                union_analysis = {"error": str(exc)}

    # Re-run the quality gate with union-analysis context now available.
    quality_gate = evaluate_adaptive_quality_gate(
        adaptive_dir,
        registry,
        final_diag,
        policy=policy,
        union_inputs=union_inputs if isinstance(union_inputs, dict) else None,
        union_analysis=union_analysis if isinstance(union_analysis, dict) else None,
        output_prefix="adaptive_quality_gate",
    )
    if bool(policy.quality_hard_fail) and quality_gate.get("status") in {"error", "needs_more_sampling"}:
        raise RuntimeError(
            "adaptive-production quality gate failed after final phase; "
            f"see {quality_gate.get('json')}"
        )

    payload = {
        "schema_version": "adaptive_production_driver_summary_v1",
        "status": "completed",
        "adaptive_dir": str(adaptive_dir),
        "policy": _json_ready(asdict(policy)),
        "epochs_completed": int(len(epoch_summaries)),
        "epoch_summaries": epoch_summaries,
        "final_dir": str(final_dir),
        "final_windows_csv": str(final_windows_csv),
        "final_registry_used_for_mbar_csv": str(final_registry_csv),
        "final_state_schedule": final_schedule_files,
        "scheduled_final": final_scheduled_summary or {},
        "final_diagnostics_json": str(adaptive_dir / "adaptive_final_combined_diagnostics.json"),
        "initial_final_diagnostics_json": str(final_dir / "adaptive_epoch_diagnostics.json"),
        "final_extension_summaries": extension_summaries,
        "final_seed_bank": final_seed_bank or {},
        "quality_gate": quality_gate or {},
        "runtime_pool": runtime_pool.to_dict(),
        "runtime_pool_reports": _write_runtime_pool_reports(adaptive_dir, runtime_pool),
        "runtime_pool_resume_validation": runtime_pool_resume_validation,
        "context_reuse_readiness": context_reuse_readiness,
        "registry": final_registry_paths,
        "union_mbar_inputs": union_inputs or {},
        "union_mbar_analysis": union_analysis or {},
        "use_epoch_samples_for_mbar": bool(use_epoch_samples_for_mbar),
        "global_shared_gamd_setup_dir": str(global_shared_gamd_dir) if global_shared_gamd_dir is not None else "",
        "global_shared_gamd_enabled": bool(global_shared_gamd_dir is not None),
        "note": "Final phase freezes the active window set. Epoch samples are retained and tagged, but final-only MBAR remains the conservative default. When global_shared_gamd_enabled is true, all adaptive-production workers reuse one campaign-wide shared GaMD setup.",
    }
    write_json(adaptive_dir / "adaptive_production_driver_summary.json", payload)
    _write_adaptive_production_markdown(adaptive_dir / "adaptive_production_summary.md", payload)
    return payload


def _apply_registry_actions(registry: WindowStateRegistry, actions: Sequence[Tuple], epoch: int) -> None:
    controller = AdaptiveProductionController(registry)
    controller.apply_actions(epoch, actions)


def write_epoch_action_report(
    epoch_dir: Path,
    epoch: int,
    registry: WindowStateRegistry,
    diagnostics: Dict[str, Any],
    actions: Sequence[Tuple],
    policy: AdaptiveDecisionPolicy,
) -> Dict[str, str]:
    """Write a human/audit-friendly report explaining one adaptive decision step."""
    epoch_dir = Path(epoch_dir)
    action_dicts = [_action_to_dict(a) for a in actions]
    state_rows = diagnostics.get("states", []) or []
    edge_rows = diagnostics.get("edges", []) or []
    weak_edges = []
    for edge in edge_rows:
        overlap = edge.get("overlap")
        acc = edge.get("exchange_acceptance")
        weak = overlap is None or float(overlap) < float(policy.target_overlap)
        if acc is not None and float(acc) < float(policy.min_exchange_acceptance):
            weak = True
        if weak:
            weak_edges.append(edge)
    undersampled = [
        s for s in state_rows
        if int(s.get("sample_count", 0) or 0) < int(policy.min_samples_for_retire)
    ]
    report = {
        "schema_version": "adaptive_production_epoch_action_report_v1",
        "epoch": int(epoch),
        "epoch_dir": str(epoch_dir),
        "n_active_states_before_actions": int(len(registry.active_states())),
        "n_total_states_before_actions": int(len(registry.all_states())),
        "policy": _json_ready(asdict(policy)),
        "diagnostics_json": str(epoch_dir / "adaptive_epoch_diagnostics.json"),
        "n_state_diagnostics": int(len(state_rows)),
        "n_edge_diagnostics": int(len(edge_rows)),
        "weak_edges": _json_ready(weak_edges),
        "undersampled_states": _json_ready(undersampled),
        "actions": action_dicts,
        "action_counts": {
            "add": int(sum(1 for a in actions if str(a[0]) == "add")),
            "retire": int(sum(1 for a in actions if str(a[0]) == "retire")),
            "extend": int(sum(1 for a in actions if str(a[0]) == "extend")),
            "split": int(sum(1 for a in actions if str(a[0]) == "split")),
        },
        "converged_by_current_policy": bool(_adaptive_production_converged(actions, diagnostics, policy)),
    }
    json_path = epoch_dir / "adaptive_epoch_actions.json"
    md_path = epoch_dir / "adaptive_epoch_actions.md"
    write_json(json_path, _json_ready(report))
    _write_epoch_action_markdown(md_path, report)
    return {"json": str(json_path), "md": str(md_path)}


def _write_epoch_action_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        f"# Adaptive production epoch {report.get('epoch')} actions",
        "",
        f"Active states before actions: **{report.get('n_active_states_before_actions')}**",
        f"Weak edges: **{len(report.get('weak_edges', []) or [])}**",
        f"Undersampled states: **{len(report.get('undersampled_states', []) or [])}**",
        f"Converged by current policy: **{report.get('converged_by_current_policy')}**",
        "",
        "## Actions",
        "",
    ]
    actions = report.get("actions", []) or []
    if not actions:
        lines.append("No registry-changing actions were proposed.")
    for action in actions:
        kind = action.get("action")
        reason = action.get("reason", "")
        if kind == "add":
            lines.append(f"- **add** parent `{action.get('parent_state_id')}` params `{action.get('params')}` — {reason}")
        elif kind in {"retire", "extend"}:
            lines.append(f"- **{kind}** state `{action.get('state_id')}` — {reason}")
        else:
            lines.append(f"- **{kind}** — {reason}")
    lines.extend(["", "## Weak edges", ""])
    weak = report.get("weak_edges", []) or []
    if not weak:
        lines.append("No weak edges under the current policy.")
    else:
        lines.append("| state_i | state_j | overlap | exchange_acceptance | warnings |")
        lines.append("|---:|---:|---:|---:|---|")
        for edge in weak[:50]:
            lines.append(
                f"| {edge.get('state_i')} | {edge.get('state_j')} | {edge.get('overlap')} | "
                f"{edge.get('exchange_acceptance')} | {', '.join(edge.get('warnings', []) or [])} |"
            )
    lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _action_to_dict(action: Tuple) -> Dict[str, Any]:
    kind = str(action[0])
    if kind == "add":
        _, parent, params, reason = action
        return {"action": kind, "parent_state_id": parent, "params": list(params), "reason": reason}
    if kind == "retire":
        _, sid, reason = action
        return {"action": kind, "state_id": sid, "reason": reason}
    if kind == "extend":
        _, sid, reason = action
        return {"action": kind, "state_id": sid, "reason": reason}
    if kind == "split":
        _, parent, children, reason = action
        return {"action": kind, "parent_state_id": parent, "children": children, "reason": reason}
    return {"action": kind, "raw": list(action)}


def _adaptive_production_converged(actions: Sequence[Tuple], diagnostics: Dict[str, Any], policy: AdaptiveDecisionPolicy) -> bool:
    if any(str(a[0]) == "add" for a in actions):
        return False
    weak_edges = []
    for edge in diagnostics.get("edges", []):
        overlap = edge.get("overlap")
        acc = edge.get("exchange_acceptance")
        if overlap is None or float(overlap) < float(policy.target_overlap):
            weak_edges.append(edge)
            continue
        if acc is not None and float(acc) < float(policy.min_exchange_acceptance):
            weak_edges.append(edge)
    return len(weak_edges) == 0


def _write_adaptive_production_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# Adaptive production summary",
        "",
        f"Status: **{payload.get('status', 'unknown')}**",
        f"Epochs completed: **{payload.get('epochs_completed', 0)}**",
        f"Final directory: `{payload.get('final_dir', '')}`",
        f"Final active window table: `{payload.get('final_windows_csv', '')}`",
        f"Final registry CSV: `{payload.get('final_registry_used_for_mbar_csv', '')}`",
        "",
        "## Quality gate",
        "",
    ]
    quality = payload.get("quality_gate", {}) or {}
    if quality:
        lines.append(f"Status: **{quality.get('status', 'unknown')}**")
        if quality.get("json"):
            lines.append(f"JSON: `{quality.get('json')}`")
        if quality.get("md"):
            lines.append(f"Report: `{quality.get('md')}`")
        for key, title in (("errors", "Errors"), ("needs_more_sampling", "Needs more sampling"), ("warnings", "Warnings")):
            items = quality.get(key, []) or []
            if items:
                lines.append(f"{title}:")
                for item in items[:8]:
                    lines.append(f"- {item}")
    else:
        lines.append("Quality gate was not run.")
    extensions = payload.get("final_extension_summaries", []) or []
    if extensions:
        lines.extend(["", "## Frozen final extensions", ""])
        for item in extensions:
            q = item.get("quality_gate", {}) or {}
            lines.append(f"- extension {item.get('extension')}: `{item.get('dir')}`; steps **{item.get('steps')}**; gate **{q.get('status', 'unknown')}**")
    lines.extend([
        "",
        "## Union-state MBAR inputs",
        "",
    ])
    union = payload.get("union_mbar_inputs", {}) or {}
    if union.get("arrays_npz"):
        lines.append(f"Arrays: `{union.get('arrays_npz')}`")
        lines.append(f"Samples CSV: `{union.get('samples_csv')}`")
        lines.append(f"Samples/states: **{union.get('n_samples')} / {union.get('n_states')}**")
    elif union.get("error"):
        lines.append(f"Union input builder failed: `{union.get('error')}`")
    else:
        lines.append("Union input builder was not run.")
    lines.extend([
        "",
        "## Union MBAR analysis",
        "",
    ])
    analysis = payload.get("union_mbar_analysis", {}) or {}
    if analysis.get("status"):
        lines.append(f"Status: **{analysis.get('status')}**")
        if analysis.get("state_free_energies_csv"):
            lines.append(f"State free energies: `{analysis.get('state_free_energies_csv')}`")
        if analysis.get("overlap_matrix_csv"):
            lines.append(f"Overlap matrix: `{analysis.get('overlap_matrix_csv')}`")
        if analysis.get("warnings"):
            lines.append("Warnings:")
            for warn in analysis.get("warnings", [])[:10]:
                lines.append(f"- {warn}")
        if analysis.get("errors"):
            lines.append("Errors:")
            for err in analysis.get("errors", [])[:10]:
                lines.append(f"- {err}")
    elif analysis.get("error"):
        lines.append(f"Union MBAR analysis failed: `{analysis.get('error')}`")
    else:
        lines.append("Union MBAR analysis was not run.")
    lines.extend([
        "",
        "## Epochs",
        "",
    ])
    for ep in payload.get("epoch_summaries", []):
        lines.append(f"### Epoch {ep.get('epoch')}")
        lines.append(f"Directory: `{ep.get('epoch_dir')}`")
        lines.append(f"Active states after epoch: **{ep.get('n_active_after_epoch')}**")
        actions = ep.get("actions", []) or []
        if actions:
            lines.append("Actions:")
            for action in actions:
                lines.append(f"- `{action.get('action')}`: {action.get('reason', '')}")
        else:
            lines.append("Actions: none")
        gate = ep.get("convergence_gate", {}) or {}
        if gate:
            lines.append(f"Convergence gate: **{gate.get('status', 'unknown')}**; stop adaptive: **{gate.get('stop_adaptive', False)}**")
            if gate.get("md"):
                lines.append(f"Gate report: `{gate.get('md')}`")
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_adaptive_convergence_gate(
    epoch_dir: Path,
    epoch: int,
    registry: WindowStateRegistry,
    diagnostics: Dict[str, Any],
    actions: Sequence[Tuple],
    policy: AdaptiveDecisionPolicy,
) -> Dict[str, Any]:
    """Decide whether adaptive epochs are ready to stop before final production.

    This gate is intentionally separate from the final quality gate.  The final
    quality gate asks whether the frozen production data are analysis-ready;
    this gate asks whether the adaptive phase has done enough topology/window
    refinement to justify entering that frozen phase.  In other words, it keeps
    the sampler from triumphantly declaring victory while a flaming weak edge
    is still sitting in the graph wearing a tiny party hat.
    """
    epoch_dir = Path(epoch_dir)
    action_rows = [_action_to_dict(a) for a in actions]
    add_like = [a for a in action_rows if a.get("action") in {"add", "split"}]
    extend_like = [a for a in action_rows if a.get("action") == "extend"]

    errors: List[str] = []
    continue_reasons: List[str] = []
    warnings: List[str] = []
    recommendations: List[str] = []

    if not active_graph_connected(registry):
        errors.append("active adaptive-state graph is disconnected")
        recommendations.append("Add bridge windows before frozen final production.")

    weak_edges: List[Dict[str, Any]] = []
    for edge in diagnostics.get("edges", []) or []:
        overlap = edge.get("overlap")
        acc = edge.get("exchange_acceptance")
        weak = overlap is None or float(overlap) < float(policy.target_overlap)
        if acc is not None and float(acc) < float(policy.min_exchange_acceptance):
            weak = True
        if weak:
            weak_edges.append(edge)
    if len(weak_edges) > int(policy.convergence_max_weak_edges):
        continue_reasons.append(
            f"{len(weak_edges)} weak edge(s) exceed allowed maximum {policy.convergence_max_weak_edges}"
        )
        recommendations.append("Continue adaptive epochs or add midpoint/bridge windows near the weak edges.")

    low_sample_states: List[Dict[str, Any]] = []
    for row in diagnostics.get("states", []) or []:
        try:
            sid = int(row.get("state_id"))
            count = int(row.get("sample_count", 0) or 0)
        except Exception:
            continue
        if sid not in registry.active_state_ids():
            continue
        if count < int(policy.convergence_min_samples_per_state):
            low_sample_states.append({
                "state_id": sid,
                "sample_count": count,
                "minimum": int(policy.convergence_min_samples_per_state),
            })
    if low_sample_states:
        continue_reasons.append(
            f"{len(low_sample_states)} active state(s) below convergence sample threshold"
        )
        recommendations.append("Run at least one more adaptive/scheduled epoch or lower the convergence sample threshold.")

    if add_like:
        continue_reasons.append(f"{len(add_like)} add/split action(s) proposed")
        recommendations.append("Apply proposed new windows and run another epoch before freezing final production.")

    if extend_like and not bool(policy.convergence_allow_extend_actions):
        continue_reasons.append(f"{len(extend_like)} extend action(s) proposed")

    high_boost_states = []
    for row in diagnostics.get("states", []) or []:
        bsd = row.get("gamd_boost_sd_kcal_mol")
        if bsd is not None and math.isfinite(float(bsd)) and float(bsd) > float(policy.max_gamd_boost_sd_kcal_mol):
            high_boost_states.append({
                "state_id": int(row.get("state_id", -1)),
                "boost_sd_kcal_mol": float(bsd),
                "maximum": float(policy.max_gamd_boost_sd_kcal_mol),
            })
    if high_boost_states:
        warnings.append(f"{len(high_boost_states)} state(s) exceed GaMD boost SD threshold")
        recommendations.append("Inspect boost distributions before trusting reweighted adaptive samples.")

    if errors:
        status = "error"
        stop_adaptive = False
    elif continue_reasons:
        status = "continue"
        stop_adaptive = False
    elif warnings:
        status = "warning_stop_allowed"
        stop_adaptive = True
    else:
        status = "converged"
        stop_adaptive = True

    payload = {
        "schema_version": "adaptive_production_convergence_gate_v1",
        "epoch": int(epoch),
        "epoch_dir": str(epoch_dir),
        "status": status,
        "stop_adaptive": bool(stop_adaptive),
        "errors": errors,
        "continue_reasons": continue_reasons,
        "warnings": warnings,
        "recommendations": sorted(set(recommendations)),
        "weak_edges": _json_ready(weak_edges),
        "low_sample_states": low_sample_states,
        "high_boost_states": high_boost_states,
        "actions": action_rows,
        "policy": _json_ready(asdict(policy)),
    }
    json_path = epoch_dir / "adaptive_convergence_gate.json"
    md_path = epoch_dir / "adaptive_convergence_gate.md"
    write_json(json_path, _json_ready(payload))
    _write_adaptive_convergence_gate_markdown(md_path, payload)
    payload["json"] = str(json_path)
    payload["md"] = str(md_path)
    return payload


def _write_adaptive_convergence_gate_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# Adaptive production convergence gate",
        "",
        f"Epoch: **{payload.get('epoch')}**",
        f"Status: **{payload.get('status')}**",
        f"Stop adaptive phase: **{payload.get('stop_adaptive')}**",
        "",
    ]
    for title, key in (
        ("Errors", "errors"),
        ("Continue reasons", "continue_reasons"),
        ("Warnings", "warnings"),
        ("Recommendations", "recommendations"),
    ):
        items = payload.get(key, []) or []
        if not items:
            continue
        lines.append(f"## {title}")
        for item in items:
            lines.append(f"- {item}")
        lines.append("")

    weak = payload.get("weak_edges", []) or []
    if weak:
        lines.append("## Weak adaptive edges")
        lines.append("| state_i | state_j | overlap | exchange acceptance | warnings |")
        lines.append("|---:|---:|---:|---:|---|")
        for edge in weak[:50]:
            lines.append(
                f"| {edge.get('state_i')} | {edge.get('state_j')} | {edge.get('overlap')} | "
                f"{edge.get('exchange_acceptance')} | {', '.join(edge.get('warnings', []) or [])} |"
            )
        lines.append("")

    low = payload.get("low_sample_states", []) or []
    if low:
        lines.append("## Low-sample active states")
        lines.append("| state_id | sample_count | minimum |")
        lines.append("|---:|---:|---:|")
        for row in low[:100]:
            lines.append(f"| {row.get('state_id')} | {row.get('sample_count')} | {row.get('minimum')} |")
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
