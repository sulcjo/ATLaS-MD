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
from .io import write_json, read_json_file, resolve_run_temperature_k, _json_ready, acquire_run_lock
from .lifecycle import _graceful_shutdown
from .store import SegmentRegistry
# Real-frame seed extraction lives in tica.py (a dependency-free leaf module)
# so seeding.py can also use it for campaign-wide seed search without a
# circular import - seeding.py -> production.py -> adaptive_production.py is
# an existing chain, so this module cannot be imported from seeding.py.
from .tica import _append_seed_bank_row, resolve_seed_frame_pdb

logger = logging.getLogger(__name__)


def _is_adaptive_production_completed(adaptive_dir: Path) -> bool:
    """Return True iff adaptive_production_driver_summary.json has status == 'completed'."""
    summary_path = Path(adaptive_dir) / "adaptive_production_driver_summary.json"
    if not summary_path.exists():
        return False
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        return str(data.get("status", "")) == "completed"
    except Exception:
        return False


def _detect_extend_mode(out_dir: Path) -> str:
    """Detect the appropriate extension mode based on what already exists in out_dir.

    Returns one of: "regular", "adaptive", "frozen".
    - "regular"  – no adaptive_production/ directory found; fall back to regular --resume.
    - "adaptive" – adaptive directory exists but summary is missing or status != 'completed'.
    - "frozen"   – adaptive production completed; add frozen extension rounds.
    """
    adaptive_dir = Path(out_dir) / "adaptive_production"
    if not adaptive_dir.exists():
        return "regular"
    summary_path = adaptive_dir / "adaptive_production_driver_summary.json"
    if not summary_path.exists():
        return "adaptive"
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        status = str(data.get("status", ""))
    except Exception:
        return "adaptive"
    if status == "completed":
        return "frozen"
    return "adaptive"


def _resolve_and_apply_extend_mode(args: Any, out_dir: Path) -> str:
    """Resolve extend_mode (handling 'auto') and apply per-mode arg mutations to args.

    Must be called before the CLI dispatch chain so that the 'regular' mode can
    set args.resume = True and fall through to the existing --resume branch.

    Returns the resolved extend_mode string.
    """
    if not bool(getattr(args, "extend", False)):
        return str(getattr(args, "extend_mode", "auto"))

    mode = str(getattr(args, "extend_mode", "auto"))

    if mode == "auto":
        mode = _detect_extend_mode(out_dir)
        args.extend_mode = mode
        print(f"[extend] auto-detected extend_mode: {mode}")

    if mode == "frozen":
        rounds = max(1, int(getattr(args, "ap_extend_rounds", 1) or 1))
        args.adaptive_production_final_quality_extension_rounds = rounds
        steps = int(getattr(args, "ap_extend_steps", 0) or 0)
        if steps > 0:
            args.adaptive_production_final_quality_extension_steps = steps

    elif mode == "adaptive":
        current_epochs = int(getattr(args, "adaptive_production_epochs", 3) or 3)
        extra = max(1, int(getattr(args, "ap_extend_rounds", 1) or 1))
        # Read epochs_completed from the summary JSON so that max_epochs always
        # exceeds the *completed* count by exactly `extra`, regardless of the
        # configured cap (e.g. if 3 of 5 were done, use 3 not 5 as the base).
        summary_path = Path(out_dir) / "adaptive_production" / "adaptive_production_driver_summary.json"
        epochs_completed = 0
        try:
            if summary_path.exists():
                data = json.loads(summary_path.read_text(encoding="utf-8"))
                epochs_completed = int(data.get("epochs_completed", 0) or 0)
        except Exception:
            epochs_completed = 0
        effective_base = max(epochs_completed, current_epochs)
        args.adaptive_production_epochs = effective_base + extra

    elif mode == "topup":
        args.adaptive_production_topup_only = True

    elif mode == "regular":
        args.resume = True

    return mode


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
    gamd_lambda: float = 0.0
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
        lam = data.get("gamd_lambda")
        data["gamd_lambda"] = 0.0 if lam in ("", "None", None) else float(lam)
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
    primary_target_deviation_sigma: Optional[float] = None
    secondary_target_deviation_sigma: Optional[float] = None
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

    target_overlap: float = 0.30
    min_exchange_acceptance: float = 0.08
    min_samples_for_add: int = 50
    min_samples_for_retire: int = 200
    max_new_windows_per_epoch: int = 4
    retire_converged: bool = True
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
    redundant_overlap: float = 0.45
    min_active_states: int = 8
    # A state's achieved cv_mean deviating this many of its own sample std-devs
    # from its nominal target center means the window failed to reach target
    # (e.g. blocked by a secondary-CV-coupled physical barrier), not just
    # normal thermal wobble around the restraint minimum.
    max_target_deviation_sigma: float = 3.0
    # tica_coverage_add extrapolates a new window's secondary-CV restraint onto
    # a previously-uncovered region, unlike weak-edge bridges which interpolate
    # between two already-sampled neighbors. Inheriting the nearest parent's
    # secondary_k unchanged assumes the free-energy curvature there matches the
    # parent's own (already-covered) location; when it doesn't, the restraint is
    # too soft to hold the new window and it relaxes back into the parent's
    # basin instead of sampling the new region (observed: a window landed >10
    # sigma from its own target after being added this way).
    #
    # Rather than guess a flat multiplier, the new secondary_k is estimated by
    # equipartition (k = kT/Var) from the observed spread of the frames that
    # populate the newly-discovered region -- an approximation grounded in
    # data actually available at the new target, not the parent's unrelated
    # location. Those frames were still sampled under some other window's
    # bias though, so this can still be noisy (population near the minimum
    # threshold) or degenerate (near-zero spread, e.g. a coarse/quantized CV):
    # this cap bounds how far above the parent's own k that estimate is
    # allowed to push, and the estimate is always floored at the parent's k
    # (only ever tighten, never loosen).
    coverage_k_stiffen_cap: float = 10.0
    # A weak-edge "bridge" window has always been placed at the plain midpoint
    # of the two endpoints it is meant to reconnect, with no check that a
    # midpoint can actually reach either of them.  It often cannot: umbrella
    # neighbours only exchange usefully while their centre spacing is within
    # roughly 1-1.5 of each endpoint's own harmonic width
    # sigma = sqrt(kT/k), and a stiff endpoint has a very small sigma.
    # Observed on chignolin_6: state 24 was created to repair edge 14-20
    # (overlap 0.127) and its midpoint at CV2 = -0.7509 is the arithmetically
    # correct midpoint -- but the two endpoints are 1.755 CV2 units apart with
    # sigma 0.073 / 0.081, so the new window landed ~12 and ~11 sigma away and
    # its measured overlap to state 20 came out 0.055, *worse* than the 0.127
    # it was built to fix.  One window cannot bridge that gap; ceil(gap /
    # (1.5 * sigma_min)) - 1 = 16 evenly spaced ones can.
    #
    # ``bridge_healthy_spacing_sigma`` is that healthy spacing/sigma band, and
    # ``bridge_multi_window`` places the number of bridges the prediction
    # actually calls for (bounded by ``max_new_windows_per_epoch``) instead of
    # exactly one.  Set ``bridge_multi_window=False`` to restore the historical
    # single-midpoint behaviour; the prediction and any shortfall are still
    # recorded in the new state's ``reason`` either way, so the decision stays
    # auditable from the registry CSV alone.
    bridge_healthy_spacing_sigma: float = 1.5
    bridge_multi_window: bool = True
    # Recording the prediction was not enough: it never influenced WHICH edge
    # got the budget.  Allocation was flat round-robin over weak edges, worst
    # overlap first, so chignolin_6's edge 14-20 -- which needs 14 windows
    # before the chain reaches even its softer endpoint -- always took a
    # round-1 slot ahead of an edge whose SECOND bridge would have completed a
    # real repair.  ``bridge_repairable_first`` serves the edges that can
    # actually be reconnected within this epoch's budget first, each seeded
    # with its own ``min_useful_bridges``, and only then hands what is left to
    # the rest.  Set it False to restore the flat round-robin.
    #
    # ``bridge_skip_unreachable`` is the stronger policy: refuse an edge that
    # cannot be reconnected this epoch instead of spending anything on it. It
    # is OFF by default, and deliberately so -- an evenly spaced placement
    # HALVES the gap, so a lone bridge on an unreachable edge is a bisection
    # step (measured: 8 -> 4 -> 2 -> 1 bridges needed over successive epochs),
    # and refusing it makes a run whose per-epoch budget is smaller than every
    # weak edge's requirement add zero windows forever.  Turn it on to spend
    # the whole budget on edges that finish this epoch, at the price of that
    # stall.  Either way the edge is reported, never silently dropped.
    bridge_repairable_first: bool = True
    bridge_skip_unreachable: bool = False


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
        gamd_lambda: float = 0.0,
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
            gamd_lambda=0.0 if gamd_lambda is None else float(gamd_lambda),
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
            "gamd_sigma0p", "gamd_sigma0d", "gamd_lambda", "source", "reason", "usable_for_mbar", "burnin_steps",
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
        path = Path(directory) / "state_registry.json"
        reg = cls.load_json(path)
        if path.exists() and not reg.all_states():
            print(
                f"WARNING: state registry file {path} exists but parsed to zero states "
                "(empty or corrupt file) -- this is not the same as a fresh campaign with "
                "no registry file yet; check the file if this is unexpected."
            )
        return reg

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
            "gamd_lambda",
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
                "gamd_lambda": float(state.gamd_lambda),
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
        gamd_lambda = _float_or_none(_csv_first(row, ["gamd_lambda"])) or 0.0
        state_id_raw = _csv_first(row, ["state_id"], None)
        state = reg.add_state(
            primary_center=primary,
            primary_k=k,
            secondary_center=secondary,
            secondary_k=secondary_k,
            gamd_lambda=gamd_lambda,
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
    # Pre-compute axis spacings for normalized 2D scoring.  Raw dp²+ds² mixes
    # incompatible units (e.g. contact fraction 0-1 vs rama-map -1 to 1), so a
    # 1-window-spacing displacement on each axis should contribute equally.
    _p_sorted = sorted({float(t.primary_center) for t in registry.active_states()})
    _dp_scale = float(np.median(np.diff(_p_sorted))) if len(_p_sorted) > 1 else 1.0
    _dp_scale = max(1e-12, _dp_scale)
    _s_sorted = sorted({float(t.secondary_center) for t in registry.active_states() if t.secondary_center is not None})
    _ds_scale = float(np.median(np.diff(_s_sorted))) if len(_s_sorted) > 1 else 1.0
    _ds_scale = max(1e-12, _ds_scale)
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
            score = (dp / _dp_scale) ** 2 + (ds / _ds_scale) ** 2
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


def rescore_seed_bank_secondary_cv(
    seed_bank_dir: Path,
    tica_result: "TICAResult",
    app,
    unit,
) -> Dict[str, Any]:
    """Replace each seed-bank row's placeholder secondary_cv_value with a real
    measurement of that row's own PDB, projected onto the given tICA model.

    write_epoch_seed_bank()/write_seed_bank_from_run_dirs() write
    secondary_cv_value as a copy of the *source* state's own secondary_center
    at seed-bank-write time -- never a measurement of the seed structure
    itself -- and that write happens before _apply_tica_centers_to_registry
    moves any center. select_state_aware_seeds_for_targets() then "matches"
    every state to its own seed trivially (both numbers are the same
    not-yet-updated value), and that assignment is never revisited once
    centers move to their post-tICA-switch tIC1 medians.

    Confirmed empirically on a real crashed run (chignolin_6 epoch_001): all
    12 gate-failing "seeded" (non-preflight-unreachable) windows matched their
    own source state via this tautological comparison, each still 0.6-1.6 tIC1
    units off its actual (post-switch) target. Re-measuring every available
    seed under the current model and re-running select_state_aware_seeds_for_
    targets lets a window borrow a *different* state's seed when that is the
    closer match under the new coordinate -- collapsed the same 12 deltas to
    <0.1 in an offline replay using only frames already on disk (no new MD).

    Call only after the registry's active-state centers reflect the update
    this ``tica_result`` produced (i.e. after _apply_tica_centers_to_registry
    and any _propose_tica_coverage_actions/_apply_registry_actions for newly
    added states), so the following select_state_aware_seeds_for_targets call
    scores against final, not intermediate, targets.
    """
    from .tica import backbone_dihedral_features, project_tica1

    seed_bank_dir = Path(seed_bank_dir)
    csv_path = seed_bank_dir / "final_survivor_seeds.csv"
    rows = _read_csv_dicts(csv_path)
    if not rows:
        return {"status": "empty", "seed_bank_dir": str(seed_bank_dir)}
    if not tica_result.phi_torsion_indices and not tica_result.psi_torsion_indices:
        return {"status": "no_torsion_indices", "seed_bank_dir": str(seed_bank_dir)}

    n_ok = 0
    n_failed = 0
    for row in rows:
        pdb_text = str(row.get("survivor_pdb_path", "")).strip()
        pdb_path = Path(pdb_text) if pdb_text else None
        if pdb_path is not None and not pdb_path.is_absolute():
            pdb_path = seed_bank_dir / pdb_path
        if pdb_path is not None and not pdb_path.exists():
            alt = seed_bank_dir / "pdbs" / Path(pdb_text).name
            pdb_path = alt if alt.exists() else pdb_path
        if pdb_path is None or not pdb_path.exists():
            n_failed += 1
            continue
        try:
            pdb = app.PDBFile(str(pdb_path))
            positions_nm = np.asarray(
                pdb.positions.value_in_unit(unit.nanometer), dtype=np.float64
            )
            feats = backbone_dihedral_features(
                positions_nm, tica_result.phi_torsion_indices, tica_result.psi_torsion_indices
            )
            tic1 = float(project_tica1(feats[np.newaxis, :], tica_result)[0])
        except Exception:
            n_failed += 1
            continue
        if not math.isfinite(tic1):
            n_failed += 1
            continue
        row["secondary_cv_value"] = tic1
        n_ok += 1

    fieldnames = [
        "seed_name", "survivor_pdb_path", "source_run_dir", "source_pdb_path",
        "source_label", "source_state_id", "source_epoch_window",
        "primary_cv_value", "secondary_cv_value",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return {
        "status": "ok" if n_ok else "all_failed",
        "seed_bank_dir": str(seed_bank_dir),
        "n_rescored": int(n_ok),
        "n_failed": int(n_failed),
    }


def _seed_bank_dir_is_usable(seed_bank_dir: Path) -> bool:
    """True if seed_bank_dir has a real, non-empty final_survivor_seeds.csv."""
    csv_path = seed_bank_dir / "final_survivor_seeds.csv"
    if not csv_path.exists():
        return False
    report_path = seed_bank_dir / "seed_bank_report.json"
    if report_path.exists():
        try:
            report = read_json_file(report_path)
            if report.get("status") not in (None, "ok"):
                return False
        except Exception:
            pass
    try:
        return len(_read_csv_dicts(csv_path)) > 0
    except Exception:
        return False


def _discover_latest_seed_bank(adaptive_dir: Path) -> Optional[Path]:
    """Find the most advanced on-disk adaptive seed bank for a resumed run.

    current_seed_bank is a plain local variable in run_adaptive_production_
    auto_loop -- it only ever gets advanced in-process, right after
    write_epoch_seed_bank()/write_seed_bank_from_run_dirs() succeeds for the
    epoch that just finished. Nothing persists it, so a fresh process
    invocation (every --resume is a fresh process) re-initializes it from
    args.seed_conformers_dir -- the raw GENPEPT library set once in the YAML
    -- regardless of how many epochs of real, state-tagged adaptive sampling
    already exist on disk. If the resumed run picks straight back up inside
    an epoch's baseline/topup segment (i.e. past the point in the loop where
    current_seed_bank would normally be advanced this process), the adaptive
    seed bank is silently never used again: filter_seed_bank_for_state_ids
    finds no source_state_id match in the untagged GENPEPT library, seeding
    falls back to generic/no-seed starts, and the US starting-structure
    quality gate fails unpredictably window-by-window -- indistinguishable,
    from the crash message alone, from a genuinely bad seed/CV problem.
    Confirmed on a real run (chignolin_6): filtered_seed_bank_report.json
    recorded source_seed_bank as the raw GENPEPT dir with 0 matched rows on
    a resume that should have used seed_bank_epoch_000.

    Prefers seed_bank_final over the highest-numbered seed_bank_epoch_NNN,
    matching this module's own end-of-campaign precedence. Only returns a
    directory whose own write recorded real matched rows, so an empty/failed
    seed-bank write is not mistaken for a usable one.
    """
    final_dir = adaptive_dir / "seed_bank_final"
    if _seed_bank_dir_is_usable(final_dir):
        return final_dir
    candidates: List[Tuple[int, Path]] = []
    for child in adaptive_dir.glob("seed_bank_epoch_*"):
        if not child.is_dir():
            continue
        suffix = child.name[len("seed_bank_epoch_"):]
        if suffix.isdigit():
            candidates.append((int(suffix), child))
    for _, child in sorted(candidates, reverse=True):
        if _seed_bank_dir_is_usable(child):
            return child
    return None


def extract_and_register_coverage_seeds(
    coverage_actions: List[Tuple],
    new_state_ids: List[int],
    seed_bank_dir: Path,
    topology,
    source_label: str,
) -> Dict[str, Any]:
    """Extract a real seed structure for each coverage action that has one, register it.

    ``new_state_ids`` must be the state ids created by applying
    ``coverage_actions``, in the same order. The call site derives this from
    a before/after set-difference on ``registry.active_states()``: safe there
    because ``add_state`` assigns ids strictly increasing and
    ``AdaptiveProductionController.apply_actions`` applies a list of actions
    in order with nothing else adding states in between - not a generically
    reusable pairing outside that specific call path.
    """
    coverage_actions = list(coverage_actions)
    new_state_ids = list(new_state_ids)
    if len(new_state_ids) != len(coverage_actions):
        logger.warning(
            "extract_and_register_coverage_seeds: %d actions but %d new state ids - "
            "id/action pairing assumption broke, skipping extraction this round",
            len(coverage_actions), len(new_state_ids),
        )
        return {"n_extracted": 0, "n_skipped": len(coverage_actions), "rows": [], "error": "id_count_mismatch"}

    seed_bank_dir = Path(seed_bank_dir)
    pdb_dir = seed_bank_dir / "pdbs"
    n_extracted = 0
    n_skipped = 0
    rows: List[Dict[str, Any]] = []
    for action, state_id in zip(coverage_actions, new_state_ids):
        metadata = action[4] if len(action) > 4 else None
        seed_frame = metadata.get("seed_frame") if isinstance(metadata, dict) else None
        if not seed_frame:
            continue
        pdb_path = pdb_dir / f"tica_coverage_state_{int(state_id):04d}.pdb"
        info = resolve_seed_frame_pdb(seed_frame, topology, pdb_path)
        if info is None:
            n_skipped += 1
            continue
        row = {
            "seed_name": f"tica_coverage_state_{int(state_id):04d}",
            "survivor_pdb_path": str(Path("pdbs") / pdb_path.name),
            "source_run_dir": str(seed_frame.get("epoch_dir", "")),
            "source_pdb_path": info["source_xtc"],
            "source_label": str(source_label),
            "source_state_id": int(state_id),
            "source_epoch_window": info["frame_index"],
            "primary_cv_value": seed_frame.get("primary_cv", ""),
            "secondary_cv_value": seed_frame.get("secondary_cv", ""),
        }
        _append_seed_bank_row(seed_bank_dir, row)
        rows.append(row)
        n_extracted += 1
    return {"n_extracted": n_extracted, "n_skipped": n_skipped, "rows": rows}


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
    """Resolve 1/(kB*T) in mol/kJ for a per-window run directory.

    See gareus.io.resolve_run_temperature_k: gareus_metadata.json never
    carries a temperature field in practice, so this used to silently
    resolve to None for every run, leaving every union-MBAR sample's beta
    blank and the reduced-bias matrix NaN.
    """
    temp = resolve_run_temperature_k(run_dir)
    if temp is None:
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
                    "v_pep_kj_mol": _blank_if_nonfinite(_value_at(data, "v_pep_kj_mol", i, "")),
                    "v_dih_kj_mol": _blank_if_nonfinite(_value_at(data, "v_dih_kj_mol", i, "")),
                    "gamd_lambda": _blank_if_nonfinite(_value_at(data, "gamd_lambda", i, "")),
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


def _phase_window_map_is_owned_by_a_resuming_phase(
    phase_dir: Path,
    map_path: Optional[Path] = None,
    *,
    resume_requested: bool,
    expected_rows: Optional[int] = None,
) -> bool:
    """True when ``phase_dir``'s existing ``epoch_window_map.csv`` must not be rewritten.

    ``epoch_window_map.csv`` is the only record of which umbrella Hamiltonian a
    sample logged under ``window_id = i`` was actually generated by, and the MBAR
    loader also reads each row's own ``primary_center``/``primary_k``/
    ``secondary_center``/``secondary_k`` to rebuild that epoch's bias matrix (the
    2026-08-04 per-epoch-bias fix).  The driver writes it from the live registry
    *before* a phase runs, which is right for a phase that has not run yet and
    wrong for one that has:

    * ``--us-auto-drop-bad-windows`` prunes windows post-pull and renumbers every
      window-indexed array around the survivors; the drop then rewrites this
      phase's map to match (``production.rewrite_epoch_window_map_after_drop``).
      The drop only runs on a *fresh* start -- a resumed phase takes the
      ``fast_resume`` path, which rebuilds its window set from the phase's own
      recorded post-drop tables and never re-pulls -- so nothing recreates that
      correction.  Overwriting it with a fresh identity map off the still-unpruned
      registry puts the run straight back into the chignolin_6 failure
      (~1.82M of 12.1M samples attributed to the wrong state, up to 2,012 kT of
      fabricated bias, base MBAR ESS 0.74%; see
      docs/chignolin_6_low_ess_root_cause.md).
    * even with no drop involved, the registry's own centers move mid-campaign
      (``_apply_tica_centers_to_registry`` on the tICA CV2 switch).  A phase's
      already-written samples were generated under the centers that were in
      effect when it started, which is exactly what its existing map records --
      rewriting them to today's values is the 2026-08-04 bug again.

    So the rule is: leave it alone if, and only if, this phase is actually about
    to resume real production that was already logged against the file on disk.
    That is three conditions, all required:

    1. ``resume_requested`` -- the campaign-level ``--ap-resume``.  Without it
       ``run_gareus`` starts fresh, re-pulls, and may drop a *different* window
       set, so the map on disk is not this run's map at all and preserving it
       would be a new bug rather than a fix.
    2. an ``epoch_window_map.csv`` already exists.  Epoch 0's bootstrap site
       writes the phase's first map *after* the phase ran (the registry is seeded
       from the post-drop table there, so identity is correct); keying off "this
       phase resumed" instead of "a map is already there" would leave that phase
       with no map at all.
    3. ``production_checkpoint_available(phase_dir)`` -- the same predicate the
       driver itself uses to set ``args.resume``, so the guard can never disagree
       with the resume decision it is guarding.  No checkpoint means no sampling
       happened behind that map; it is a stale plan, not a record.

    Note that ``production.run_gareus``'s own ``fast_resume`` is a *third*,
    weaker predicate (bare truthiness of the manifest's
    ``replica_checkpoint_files``, with no per-file size or median-outlier
    check).  On a driver-invoked phase it nevertheless cannot disagree with
    condition 3, and the real relationship is worth spelling out because the
    natural reading of "weaker predicate" is that the two can diverge here.
    ``run_gareus`` only looks at the manifest at all when ``args.resume`` is set
    (``resume_manifest = read_json_file(...) if bool(getattr(args, "resume",
    False)) else None``), so ``fast_resume`` implies ``args.resume``; and every
    driver site sets ``args.resume`` to exactly this guard's own condition 3 --
    ``bool(resume_requested and production_checkpoint_available(<phase dir>))``
    at ``run_segment``, at the flat epoch, at the frozen final phase and at each
    final extension round.  ``production_checkpoint_available`` is strictly the
    stronger reading of the same manifest, so the chain runs one way only: this
    guard preserves => ``args.resume`` => ``fast_resume``.  A phase whose map is
    preserved therefore always takes the fast path that rebuilds its window set
    from its own recorded post-drop tables, by construction rather than by two
    predicates happening to diverge in a safe direction.

    The caveat is timing, not predicates.  The guard is consulted slightly
    earlier in a phase's setup than the ``args.resume`` assignment -- most
    visibly at the frozen-final site, which hands the guard the bare
    ``resume_requested`` and lets it evaluate
    ``production_checkpoint_available(final_dir)`` itself -- so the two agree
    because nothing creates or removes a checkpoint in that phase directory in
    between, not because a single expression is evaluated once and shared.  Keep
    that true when reordering phase setup.

    The weaker ``fast_resume`` predicate is reachable on its own only outside the
    driver's phase loop: a standalone ``gareus --resume`` invocation takes
    ``args.resume`` straight from the flag (as does ``--extend --extend-mode
    regular``, via ``_resolve_and_apply_extend_mode``), so a torn/undersized
    checkpoint there does give ``fast_resume`` with
    ``production_checkpoint_available() == False`` -- but those paths run a plain
    non-adaptive-production run, never write a phase's map from a registry, and
    never call this guard, so no map decision rides on the difference.

    This is a **no-clobber** guarantee, not a repair path: it cannot fix a phase
    whose map was never corrected in the first place (interrupted before the drop
    rewrite), and it does not stop the registry from handing dropped states out
    to *future* phases -- the registry still marks them active/usable_for_mbar.
    Propagating the drop back into the registry is the actual root fix and lives
    elsewhere.  Being a veto, it also presupposes that *something* writes the map
    when the phase is not resuming; the flat-epoch loop-entry refresh (see the
    long comment at that call site) exists because a flat numbered epoch >= 1 was
    the one phase kind where nothing did.

    Keying on a *checkpoint* is what this guard can see, not the whole rule.  A
    checkpoint too torn to resume from does not un-write the Parquet rows a
    previous attempt already flushed, so every call site pairs this guard with
    :func:`_phase_holds_samples_logged_against_its_window_map`, which vetoes on
    the samples themselves.  Read the two together: this one is "someone is about
    to resume behind this map", that one is "someone already sampled behind it".
    """
    if not bool(resume_requested):
        return False
    phase_dir = Path(phase_dir)
    map_path = Path(map_path) if map_path is not None else phase_dir / "epoch_window_map.csv"
    if not map_path.exists():
        return False
    if not production_checkpoint_available(phase_dir):
        return False
    existing_rows: Optional[int] = None
    try:
        with map_path.open(newline="") as handle:
            existing_rows = sum(1 for _ in csv.DictReader(handle))
    except Exception:
        existing_rows = None
    detail = ""
    if existing_rows is not None and expected_rows is not None and int(existing_rows) != int(expected_rows):
        # Report the observation, not a diagnosis.  This clause used to assert
        # "looks like a post-pull window drop was already applied here", which is
        # only one of the ways the two counts can differ -- and it is the wrong
        # one at the epoch-0 post-run bootstrap call site, where the registry has
        # just been rebuilt from *this* attempt's own post-drop window table, so a
        # row-count difference there would mean the file describes some other
        # window set (a previous attempt's) rather than this drop's correction.
        # The decision does not turn on which of the two it is: either way the
        # rows on disk are what a previous attempt's samples were logged against,
        # and what the caller was about to write is not.
        #
        # `expected_rows` is deliberately described as "the caller expected",
        # not as the registry's active-state count: run_segment hands this guard
        # ``len(state_ids)`` -- the subset of states scheduled for that one
        # segment -- while every other site hands it
        # ``len(registry.active_states())``.  Naming either one specifically
        # makes the message wrong at the other sites, which is the same defect
        # (a message asserting a conclusion that does not hold everywhere the
        # guard is consulted) that the causal clause above was removed for.
        detail = (
            f" (map covers {existing_rows} window(s) against the {int(expected_rows)} "
            "the caller was about to write, so it does not describe that window set)"
        )
    print(
        f"    Adaptive-production: keeping existing {map_path} for the resuming phase "
        f"rather than rewriting it from the registry{detail}"
    )
    return True


# Segment statuses whose Parquet rows ``gareus.query._load_segmented_parquet``
# will actually hand to the MBAR loaders.  Mirrored here, next to the only
# consumer, rather than imported: query.py expresses the same filter inline as
# part of a DuckDB read, and this module must be able to answer the question
# without opening a database or reading a single row.
_ANALYSIS_VISIBLE_SEGMENT_STATUSES = ("complete", "interrupted")


def _phase_holds_samples_logged_against_its_window_map(
    phase_dir: Path,
    map_path: Optional[Path] = None,
) -> bool:
    """True when ``phase_dir`` already holds production samples logged against its map.

    The companion veto to
    :func:`_phase_window_map_is_owned_by_a_resuming_phase`, covering the case
    that guard structurally cannot: it asks "will a checkpoint be resumed from
    behind this map", but the question that actually decides whether the map may
    be replaced is "are there already samples on disk that were labelled against
    it".  A torn checkpoint does not un-write Parquet rows, so those are not the
    same question:

        attempt 1 pulls -> ``--us-auto-drop-bad-windows`` drops set D1 ->
        ``production.rewrite_epoch_window_map_after_drop`` compacts this phase's
        ``epoch_window_map.csv`` -> production writes samples into
        ``samples/seg_001`` -> the process dies without leaving a checkpoint
        ``production_checkpoint_available`` will accept (an exception before the
        first ``--checkpoint-interval``, a SIGKILL or a disk-full mid-checkpoint
        write -- the exact case that function's median-outlier check exists for).

    Afterwards ``production_checkpoint_available`` is False, so the resume guard
    stays silent, ``args.resume`` is False, and the phase re-pulls from scratch.
    Attempt 1's rows are nevertheless still on disk, still labelled with local
    ``window_id`` values that only the compacted map resolves.  Replacing that
    map with a fresh registry identity map re-attributes every one of them --
    the chignolin_6 failure this branch exists to remove
    (docs/chignolin_6_low_ess_root_cause.md).

    "Already holds samples" is deliberately narrower than "a Parquet file
    exists", and mirrors ``gareus.query._load_segmented_parquet``'s own
    per-segment filter, because a bare file-presence test would withdraw the
    flat-epoch loop-entry refresh in precisely the case that refresh was added
    for.  Per segment recorded in ``segments.json``:

    * ``complete`` -- visible, every row read.
    * ``interrupted`` with ``end_step >= 0`` -- visible up to ``end_step``.  This
      is the *common* shape of the failure above, not an exotic one:
      ``finalize_segment`` runs from ``run_gareus``'s ``finally`` block, so any
      exception mid-production seals the segment this way with a real
      ``end_step``, whether or not a checkpoint was ever written.
    * ``interrupted`` with no/negative ``end_step`` -- skipped by the loader (no
      valid data boundary), so not visible.
    * ``abandoned`` -- skipped by the loader.
    * ``running`` -- treated as NOT visible.  Not because the rows are absent,
      but because they are about to become unreachable: the loader reads a
      ``running`` segment only while it is the *last* one, and the very next
      ``run_gareus`` call opens its own segment before writing anything
      (``SegmentRegistry.open_segment`` and the first ``ParquetSampleWriter`` are
      both top-level statements of ``run_gareus``, and both come strictly after
      its pull/drop block -- see
      ``test_the_pull_and_drop_block_precedes_the_first_sample_segment``, which
      pins that ordering because this bullet depends on it), which demotes this
      one to non-last; the fresh-start branch then seals it ``abandoned``
      outright.  A SIGKILLed attempt leaves exactly this status -- and so does
      the attempt the flat-epoch refresh was written for, so counting it as
      durable would stop that refresh from ever firing again once a single
      5,000-row chunk had been flushed (the default ``--checkpoint-interval``
      50,000 puts the first chunk long before the first checkpoint).
    * no ``segments.json`` at all, or one that cannot be parsed -- visible if any
      Parquet file exists.  The loader's own no-registry fallback reads *every*
      file under ``samples/``, and an unparseable registry makes the phase
      unloadable rather than empty; preserving is the safe direction for both.

    Residual, stated precisely rather than hand-waved: a ``running``-and-last
    segment's rows *are* visible to the loader in the window between this
    decision and the next ``open_segment``.  So attempt 2 refreshes the map and
    then dies inside its own pull, before opening a segment, and a campaign
    analysed in that window sees attempt 1's rows under a map that no longer
    describes them.  It closes itself as soon as any later attempt reaches
    ``open_segment``, and once the two attempts' drop sets differ there is no
    single map file that would be correct for both anyway -- but it is real for a
    mid-campaign analysis, and it is the reason the skip is logged below instead
    of being silent.

    Like its sibling this is a **no-clobber** veto, not a repair path: it keeps
    an existing record from being overwritten and requires an existing map to
    protect (``map_path`` must exist -- with no map there is nothing those
    samples were logged against, and writing one is the only way the phase gets a
    map at all).
    """
    phase_dir = Path(phase_dir)
    map_path = Path(map_path) if map_path is not None else phase_dir / "epoch_window_map.csv"
    if not map_path.exists():
        return False
    samples_dir = phase_dir / "samples"
    if not samples_dir.is_dir():
        return False

    def _chunked_segment_dirs() -> List[str]:
        try:
            children = sorted(samples_dir.iterdir())
        except OSError:
            return []
        return [d.name for d in children if d.is_dir() and any(d.glob("*.parquet"))]

    def _has_parquet(seg_id: str) -> bool:
        seg_dir = samples_dir / str(seg_id)
        return seg_dir.is_dir() and any(seg_dir.glob("*.parquet"))

    segments_path = phase_dir / "segments.json"
    visible: List[str] = []
    skipped: List[str] = []
    if not segments_path.exists():
        visible = _chunked_segment_dirs()
        source = "no segments.json (the loader reads every Parquet file under samples/)"
    else:
        try:
            segments = SegmentRegistry(phase_dir).all_segments()
        except Exception as exc:
            visible = _chunked_segment_dirs()
            source = f"unreadable segments.json ({exc}); treating every Parquet file as data"
        else:
            source = "segments.json"
            for seg in segments:
                if not isinstance(seg, dict):
                    continue
                seg_id = str(seg.get("segment_id", ""))
                if not seg_id or not _has_parquet(seg_id):
                    continue
                status = str(seg.get("status", "running") or "running")
                end_step = seg.get("end_step")
                if status not in _ANALYSIS_VISIBLE_SEGMENT_STATUSES:
                    # "abandoned", "running" (see the docstring) or a status this
                    # module has never heard of.  An unknown status lands here
                    # rather than in the visible half deliberately: this veto only
                    # ever protects rows it can positively show the loader reads,
                    # and a status neither side recognises is not that.
                    is_visible = False
                elif status == "interrupted":
                    # Visible only up to `end_step`; the loader treats a missing or
                    # negative boundary as "no valid data boundary known" and
                    # skips the segment outright.
                    try:
                        is_visible = end_step is not None and int(end_step) >= 0
                    except (TypeError, ValueError):
                        is_visible = False
                else:
                    is_visible = True  # "complete": every row read
                (visible if is_visible else skipped).append(f"{seg_id}[{status}]")

    if not visible:
        if skipped:
            # The self-closing residual documented above: these rows are on disk
            # and this write is about to make the map stop describing them, but
            # every one of them is in a segment the analysis layer discards (or is
            # about to discard, for a `running` one, as soon as the next
            # run_gareus opens its own segment).  Say so rather than saying
            # nothing, so a mid-campaign analysis of this phase has a breadcrumb.
            print(
                f"    Adaptive-production: refreshing {map_path} even though "
                f"{samples_dir} holds sample segment(s) {', '.join(skipped)} from a "
                "previous attempt -- the analysis layer does not read those "
                "(gareus.query._load_segmented_parquet skips an abandoned segment, an "
                "interrupted one with no valid end_step boundary, and any running one "
                "that is no longer the last), so no record is lost"
            )
        return False
    print(
        f"    Adaptive-production: keeping existing {map_path} rather than rewriting it "
        f"from the registry -- {samples_dir} already holds production samples from a "
        f"previous attempt that were logged against it (visible segment(s) "
        f"{', '.join(visible)}, per {source}). A checkpoint that cannot be resumed "
        "from does not un-write those rows; replacing their window map would "
        "re-attribute every one of them to the wrong umbrella state."
    )
    return True


def _write_phase_window_map(
    registry: "WindowStateRegistry",
    map_path: Path,
    *,
    resume_requested: bool,
    expected_rows: Optional[int] = None,
) -> Optional[Path]:
    """Write a phase's ``epoch_window_map.csv``, unless an existing record owns it.

    Single funnel for every driver-side map write, so the two no-clobber vetoes
    exist in one place instead of being re-derived (and eventually forgotten) at
    each site.  Returns the path written, or ``None`` when the existing file was
    preserved.

    The two vetoes are a plain OR, and their triggers are disjoint by
    construction: :func:`_phase_window_map_is_owned_by_a_resuming_phase` requires
    ``production_checkpoint_available(phase_dir)``, while
    :func:`_phase_holds_samples_logged_against_its_window_map` exists exactly for
    the case where that is False and rows were written anyway.  Preserving wins
    the disjunction, which is the safe direction: a preserved map can only ever
    be the one this phase's existing rows were labelled against, whereas a
    replaced one is guaranteed wrong for them the moment a drop set is involved.
    """
    map_path = Path(map_path)
    if _phase_window_map_is_owned_by_a_resuming_phase(
        map_path.parent, map_path,
        resume_requested=bool(resume_requested),
        expected_rows=expected_rows,
    ):
        return None
    if _phase_holds_samples_logged_against_its_window_map(map_path.parent, map_path):
        return None
    return registry.write_epoch_window_map(map_path)


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


def _target_deviation_sigma(
    mean: Optional[float], std: Optional[float], target: Optional[float], min_abs_deviation: float = 1.0e-3
) -> Optional[float]:
    """How many of a state's own achieved sample std-devs its mean sits from
    its nominal restraint target. Using the state's own std as the scale (not
    an external kT/k estimate) keeps this CV-agnostic (contacts, distance,
    rama, ...).  A window that failed to reach its target due to a physical
    barrier typically also has an artificially small std (pinned near the
    barrier), which makes the deviation large in sigma units even when the
    raw offset looks modest.  Returns None when mean/target are unavailable.
    """
    if mean is None or target is None:
        return None
    deviation = abs(float(mean) - float(target))
    if deviation < float(min_abs_deviation):
        return 0.0
    scale = float(std) if std is not None and math.isfinite(float(std)) else 0.0
    return deviation / max(scale, 1.0e-9)


def _non_neighbor_redundant_pairs(
    registry: WindowStateRegistry,
    window_map: Dict[int, int],
    by_window_values: Dict[int, np.ndarray],
    geometry_edges: List[Tuple[int, int, str, Optional[float]]],
    policy: AdaptiveDecisionPolicy,
) -> List[Dict[str, Any]]:
    """Flag active state pairs with high ACHIEVED-sample overlap that are not
    already nominal geometry neighbors.

    ``build_geometry_edges`` only wires up nominal near-neighbors (sorted
    primary chain plus one nearest-2D neighbor per state), so a window that
    failed to reach its target and drifted onto a distant window's basin
    (e.g. blocked by a secondary-CV-coupled physical barrier) is invisible to
    the normal weak/redundant-edge machinery -- its geometry neighbors are
    still distinct real states, so its edges look fine, while its true
    duplicate sits several rungs away and is never compared.
    """
    active_ids = sorted(registry.active_state_ids())
    if len(active_ids) < 2:
        return []
    existing_pairs = {tuple(sorted((int(a), int(b)))) for a, b, _et, _nd in geometry_edges}
    state_to_window = {sid: w for w, sid in window_map.items() if sid in active_ids}
    alerts: List[Dict[str, Any]] = []
    for idx_i in range(len(active_ids)):
        for idx_j in range(idx_i + 1, len(active_ids)):
            si, sj = active_ids[idx_i], active_ids[idx_j]
            key = (si, sj)
            if key in existing_pairs:
                continue
            wi = state_to_window.get(si)
            wj = state_to_window.get(sj)
            if wi is None or wj is None:
                continue
            overlap = _hist_overlap(by_window_values.get(wi, np.asarray([])), by_window_values.get(wj, np.asarray([])))
            if overlap is None or overlap < float(policy.redundant_overlap):
                continue
            st_i = registry.get_state(si)
            st_j = registry.get_state(sj)
            alerts.append(
                {
                    "state_i": int(si),
                    "state_j": int(sj),
                    "window_i": int(wi),
                    "window_j": int(wj),
                    "overlap": float(overlap),
                    "primary_center_i": None if st_i is None else float(st_i.primary_center),
                    "primary_center_j": None if st_j is None else float(st_j.primary_center),
                }
            )
    alerts.sort(key=lambda a: -float(a["overlap"]))
    return alerts


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
        if len(rows) >= int(policy.min_samples_for_add):
            state_obj = registry.get_state(int(state_id))
            if state_obj is not None:
                dev_sigma = _target_deviation_sigma(cv_mean, cv_std, state_obj.primary_center)
                diag.primary_target_deviation_sigma = dev_sigma
                if dev_sigma is not None and dev_sigma >= float(policy.max_target_deviation_sigma):
                    diag.warnings.append("off_target_primary")
                if state_obj.secondary_center is not None:
                    sec_dev_sigma = _target_deviation_sigma(smean, sstd, state_obj.secondary_center)
                    diag.secondary_target_deviation_sigma = sec_dev_sigma
                    if sec_dev_sigma is not None and sec_dev_sigma >= float(policy.max_target_deviation_sigma):
                        diag.warnings.append("off_target_secondary")
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
    geometry_edges = build_geometry_edges(registry)
    edge_rows: List[EdgeDiagnostics] = []
    for si, sj, etype, nd in geometry_edges:
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

    non_neighbor_redundancies = _non_neighbor_redundant_pairs(
        registry, window_map, by_window_values, geometry_edges, policy
    )
    for alert in non_neighbor_redundancies:
        for sid in (int(alert["state_i"]), int(alert["state_j"])):
            diag = state_by_id.get(sid)
            if diag is not None and "non_neighbor_redundant" not in diag.warnings:
                diag.warnings.append("non_neighbor_redundant")

    payload = {
        "schema_version": "adaptive_production_epoch_diagnostics_v1",
        "epoch_dir": str(epoch_dir),
        "n_samples_rows": int(len(samples)),
        "n_exchange_rows": int(len(exchanges)),
        "window_map": {str(k): int(v) for k, v in sorted(window_map.items())},
        "states": [s.to_dict() for s in state_rows],
        "edges": [e.to_dict() for e in edge_rows],
        "non_neighbor_redundancies": non_neighbor_redundancies,
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


def _write_tica_version_marker(run_dir: Path, args) -> None:
    """Write tica_cv_version.txt to run_dir before MD starts (MBAR cross-epoch guard)."""
    version = str(getattr(args, "tica_cv_version", None) or "disabled")
    try:
        (Path(run_dir) / "tica_cv_version.txt").write_text(version)
    except Exception:
        pass


def _apply_tica_cv2_switch(args, tica_update_report: dict, *, next_epoch: int) -> bool:
    """Apply one-shot tica-linear CV2 switch only after a valid tICA update."""
    if not getattr(args, "tica_switch_cv2", False):
        return False
    prev_cv2 = str(getattr(args, "secondary_cv", "none") or "none")
    if prev_cv2 == "tica-linear":
        return False
    if not isinstance(tica_update_report, dict) or tica_update_report.get("status") != "updated":
        return False
    state_file = str(tica_update_report.get("state_file", "") or getattr(args, "tica_state_file", "") or "")
    if not state_file or not Path(state_file).exists():
        tica_update_report["cv2_switch_skipped"] = {
            "reason": "missing_tica_state_file",
            "state_file": state_file,
        }
        return False

    k_min = float(getattr(args, "tica_linear_k_min", 5.0) or 5.0)
    k_max = float(getattr(args, "tica_linear_k_max", 50.0) or 50.0)
    args.secondary_cv = "tica-linear"
    args.tica_state_file = state_file
    args.cv2_k_min = k_min
    args.cv2_k_max = k_max
    tica_update_report["cv2_switched"] = {
        "from": prev_cv2,
        "to": "tica-linear",
        "state_file": state_file,
        "effective_epoch": int(next_epoch),
    }
    return True


def _apply_tica_centers_to_registry(
    registry: "WindowStateRegistry",
    per_window_tic1_centers: Dict[int, float],
    epoch_dir: Path,
) -> int:
    """Update registry secondary_center for active states using per-window tIC1 medians.

    Returns number of states updated.
    """
    window_map = _load_epoch_window_map(epoch_dir, registry)
    updated = 0
    for win_idx, state_id in window_map.items():
        if int(win_idx) in per_window_tic1_centers:
            state = registry.get_state(int(state_id))
            if state is not None and state.active:
                state.secondary_center = float(per_window_tic1_centers[int(win_idx)])
                updated += 1
    return updated


# Boltzmann constant in kcal/mol/K, used to turn a spring constant into the
# harmonic width sigma = sqrt(kT/k) of the window it restrains.
_KB_KCAL_PER_MOL_K: float = 1.987204e-3


def _args_temperature_k(args) -> float:
    """Run temperature in kelvin, read from the flag the CLI actually defines.

    ``gareus/cli.py`` defines exactly one temperature flag, ``--temperature-k``
    (``args.temperature_k``, default 300.0); there is no ``--temperature`` and no
    compat shim that sets ``args.temperature``.  Three call sites in this module
    nevertheless read ``getattr(args, "temperature", 298.0)``, so every one of
    them silently took the 298.0 default on every run ever made -- inert at the
    300 K default in the sense that nothing crashed, but not inert numerically:
    these values feed ``sigma = sqrt(kB*T/k)``, which decides how many bridge
    windows a weak edge gets (``_bridge_placement_prediction``) and how MBAR
    weights are formed for the tICA refit (``_compute_mbar_weights_for_tica``).
    sigma scales as sqrt(T), so a 350 K run was assuming a sigma 8.4% too small
    and over-counting the bridges its geometry called for.

    ``temperature`` is still accepted as a second choice so a hand-built
    namespace that only carries the old (never-CLI-reachable) name keeps
    working.  The final fallback is 300.0 -- the CLI's own default -- not the
    298.0 the buggy call sites used, because "no temperature attribute at all"
    should behave like a default run, not like a number nothing ever set.
    """
    for attr in ("temperature_k", "temperature"):
        try:
            value = float(getattr(args, attr, None))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value
    return 300.0


def _resolve_secondary_k_max(args) -> Optional[float]:
    """Configured ceiling on any window's secondary-CV spring constant.

    ``--cv2-k-max`` is honoured when the *initial* umbrella window table is
    generated (``gareus/cli.py`` mirrors it onto
    ``secondary_cv_adaptive_max_k_kcal``, which the window generator reads), but
    it never reached the states adaptive production creates for itself.
    Observed on chignolin_6, whose config set ``cv2_k_max: 200.0``: state 22
    came out of the tICA-coverage path at k2 = 402.47 kcal/mol and state 25 then
    inherited half of that through a weak-edge midpoint average,
    0.5 * (148.46 + 402.47) = 275.47.  Both are badly over-restrained --
    sigma = sqrt(kT/k) collapses to 0.038 at k = 402 -- which is a large part of
    why states 22/23 ended the campaign with no usable overlap to anything.

    Read this **at state-creation time**, never as a loop-start snapshot:
    :func:`_apply_tica_cv2_switch` rewrites ``args.cv2_k_max`` to
    ``tica_linear_k_max`` when the one-shot tica-linear CV2 switch fires, and
    every state created after that point belongs to the new CV2 regime and must
    be clamped against the new ceiling.  (chignolin_6's state 22 was created
    after its own switch had already fired, so this is the live case, not a
    hypothetical.)  ``secondary_cv_adaptive_max_k_kcal`` is only a fallback: it
    is a copy made once in the CLI shim and is *not* updated by the switch.
    """
    for attr in ("cv2_k_max", "secondary_cv_adaptive_max_k_kcal"):
        try:
            value = float(getattr(args, attr, None))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value
    return None


def _clamp_secondary_k(
    secondary_k: Optional[float],
    secondary_k_max: Optional[float],
    *,
    context: str,
) -> Tuple[Optional[float], Optional[str]]:
    """Clamp a newly proposed secondary spring to the configured ceiling.

    Returns ``(secondary_k, warning_or_None)``.  A missing/non-positive/
    non-finite ceiling is treated as "no ceiling configured" and passes the
    value through untouched, so a run that never set ``--cv2-k-max`` keeps its
    previous behaviour exactly.  CV1-only states (``secondary_k is None``) are
    likewise untouched.  The warning is returned rather than printed so the
    caller can both log it and fold it into the new state's ``reason`` string,
    where it survives into ``state_registry.csv`` for later audit.
    """
    if secondary_k is None:
        return None, None
    try:
        k = float(secondary_k)
    except (TypeError, ValueError):
        return secondary_k, None
    if secondary_k_max is None:
        return k, None
    try:
        cap = float(secondary_k_max)
    except (TypeError, ValueError):
        return k, None
    if not math.isfinite(cap) or cap <= 0.0:
        return k, None
    if not math.isfinite(k) or k <= cap:
        return k, None
    sigma_before = math.sqrt(_KB_KCAL_PER_MOL_K * 300.0 / k)
    sigma_after = math.sqrt(_KB_KCAL_PER_MOL_K * 300.0 / cap)
    return cap, (
        f"secondary_k clamped {k:.2f} -> {cap:.2f} kcal/mol by cv2_k_max "
        f"({context}); sigma at 300 K {sigma_before:.4f} -> {sigma_after:.4f}"
    )


def _propose_tica_coverage_actions(
    registry: "WindowStateRegistry",
    primary_cv: np.ndarray,
    tic1: np.ndarray,
    policy: Optional[AdaptiveDecisionPolicy] = None,
    *,
    temperature_K: float = 298.0,
    population_fraction: float = 0.01,
    umbrella_state_ids: Optional[np.ndarray] = None,
    replica_ids: Optional[np.ndarray] = None,
    steps: Optional[np.ndarray] = None,
    epoch_dir: Optional[Path] = None,
    secondary_k_max: Optional[float] = None,
) -> List[Tuple]:
    """Add home states for substantially populated uncovered tIC1 regions.

    This runs only after a tICA coordinate change.  It uses the newly projected
    tIC1 values, rather than stale scalar CV2 values from the preceding epoch.
    Normal weak-edge additions are intentionally not used as a cap here: the
    existing runtime replica limit remains responsible for launch capacity.

    ``replica_ids``/``steps`` (both from :func:`gareus.tica.load_epoch_dihedral_obs`,
    same length/order as ``primary_cv``/``tic1``) are optional. When given, each
    new state's action metadata records a ``seed_frame`` pointing at one frame
    that actually populated the newly covered region - real, already-visited
    coordinates, unlike the parent's own GENPEPT-library seed which may sit
    nowhere near the new target (confirmed on chignolin_6: parent state 18's
    seed was 1.5 tIC1 units from a child state added for a region 82 real
    frames had already visited). ``umbrella_state_ids`` is recorded alongside
    purely for diagnostics (which umbrella was biasing the replica at that
    moment) - it is NOT the file/replica identifier needed to locate a
    trajectory; that is ``replica_ids``.

    ``seed_frame`` itself only records the reference; resolving it to an
    actual structure (locating the right segment's
    ``replica_trajectories/replica_<replica_id>.xtc``, if trajectory
    recording was even on, confirming/relaxing to the nearest available frame
    for that segment's own ``--traj-interval``, and reading it with the
    matching topology) is a separate, deliberately isolated step - see
    :func:`resolve_seed_frame_to_conformer`.
    """
    policy = policy or AdaptiveDecisionPolicy()
    primary = np.asarray(primary_cv, dtype=float)
    secondary = np.asarray(tic1, dtype=float)
    umbrella_state_ids_arr = np.asarray(umbrella_state_ids) if umbrella_state_ids is not None else None
    replica_ids_arr = np.asarray(replica_ids) if replica_ids is not None else None
    steps_arr = np.asarray(steps) if steps is not None else None
    active = [s for s in registry.active_states()
              if s.secondary_center is not None and (s.secondary_k or 0.0) > 0.0]
    finite = np.isfinite(primary) & np.isfinite(secondary)
    if not active or not np.any(finite):
        return []

    kbt_kcal = 1.987204e-3 * float(temperature_K)
    centers = np.asarray([float(s.secondary_center) for s in active], dtype=float)
    springs = np.asarray([float(s.secondary_k) for s in active], dtype=float)
    radii = 2.0 * np.sqrt(kbt_kcal / springs)
    values = secondary[finite]
    nearest = np.min(np.abs(values[:, None] - centers[None, :]) - radii[None, :], axis=1)
    uncovered = nearest > 0.0
    if not np.any(uncovered):
        return []

    uncovered_values = values[uncovered]
    uncovered_primary = primary[finite][uncovered]
    uncovered_umbrella_state_ids = (
        umbrella_state_ids_arr[finite][uncovered]
        if umbrella_state_ids_arr is not None and len(umbrella_state_ids_arr) == len(finite)
        else None
    )
    uncovered_replica_ids = (
        replica_ids_arr[finite][uncovered]
        if replica_ids_arr is not None and len(replica_ids_arr) == len(finite)
        else None
    )
    uncovered_steps = (
        steps_arr[finite][uncovered]
        if steps_arr is not None and len(steps_arr) == len(finite)
        else None
    )
    width = max(float(np.median(radii)), np.finfo(float).eps)
    origin = float(np.min(uncovered_values))
    bin_ids = np.floor((uncovered_values - origin) / width).astype(int)
    threshold = max(3, int(math.ceil(float(population_fraction) * int(finite.sum()))))
    actions: List[Tuple] = []
    unique_ids, per_bin_counts = np.unique(bin_ids, return_counts=True)
    unique_ids = unique_ids[per_bin_counts >= threshold]
    start = 0
    while start < len(unique_ids):
        end = start + 1
        while end < len(unique_ids) and unique_ids[end] == unique_ids[end - 1] + 1:
            end += 1
        contiguous_ids = unique_ids[start:end]
        start = end
        segment_start = 0
        while segment_start < len(contiguous_ids):
            segment_end = segment_start + 1
            while (segment_end < len(contiguous_ids)
                   and (contiguous_ids[segment_end] - contiguous_ids[segment_start] + 1) * width <= 2.0 * width):
                segment_end += 1
            group_ids = contiguous_ids[segment_start:segment_end]
            segment_start = segment_end
            mask = np.isin(bin_ids, group_ids)
            population = int(mask.sum())
            target_primary = float(np.median(uncovered_primary[mask]))
            target_secondary = float(np.median(uncovered_values[mask]))
            if registry.has_near_duplicate(target_primary, target_secondary, policy):
                continue
            # Select a parent using harmonic distance in both umbrella dimensions.
            distances = []
            for state in active:
                p_width = math.sqrt(kbt_kcal / float(state.primary_k))
                s_width = math.sqrt(kbt_kcal / float(state.secondary_k))
                distances.append(
                    ((target_primary - float(state.primary_center)) / p_width) ** 2
                    + ((target_secondary - float(state.secondary_center)) / s_width) ** 2
                )
            parent = active[int(np.argmin(distances))]
            # Extrapolating past the parent's own coverage, not interpolating
            # between two neighbors: derive the new secondary_k from the
            # observed spread of the frames that actually populate this
            # region (equipartition, k = kT/Var) instead of guessing a flat
            # multiplier or blindly inheriting the parent's own -- see
            # coverage_k_stiffen_cap docstring for the rationale and bounds.
            group_secondary_std = float(np.std(uncovered_values[mask], ddof=1))
            if group_secondary_std > 1.0e-6:
                k_from_spread = kbt_kcal / (group_secondary_std ** 2)
            else:
                k_from_spread = float(parent.secondary_k)
            stiffen_cap = max(float(policy.coverage_k_stiffen_cap), 1.0)
            stiffened_secondary_k = float(np.clip(
                k_from_spread,
                float(parent.secondary_k),
                float(parent.secondary_k) * stiffen_cap,
            ))
            # The kT/Var estimate above is clipped only against the parent's own
            # k and coverage_k_stiffen_cap -- neither of which knows about
            # --cv2-k-max, so a tight populated group could hand out a restraint
            # far stiffer than the run ever asked for (chignolin_6 state 22:
            # k2 = 402.47 under cv2_k_max = 200.0). Bind it to the configured
            # ceiling here, at creation time, and record the clamp in the
            # state's own reason so it survives into state_registry.csv.
            stiffened_secondary_k, _k_warning = _clamp_secondary_k(
                stiffened_secondary_k,
                secondary_k_max,
                context=f"tica_coverage_add near tic1={target_secondary:.4g}",
            )
            params = (target_primary, float(parent.primary_k), target_secondary, stiffened_secondary_k)
            lo = float(np.min(uncovered_values[mask]))
            hi = float(np.max(uncovered_values[mask]))
            reason = (
                "post-tICA populated uncovered region: "
                f"tic1=[{lo:.6g},{hi:.6g}], n={population}"
            )
            if _k_warning:
                logging.warning("adaptive-production: %s", _k_warning)
                reason = f"{reason}; {_k_warning}"
            metadata = {
                "population_count": population,
                "population_fraction": population / int(finite.sum()),
                "tic1_min": lo,
                "tic1_max": hi,
                "coverage_radius": float(radii[int(np.argmin(np.abs(centers - target_secondary)))]),
            }
            if uncovered_replica_ids is not None and uncovered_steps is not None:
                # Frame in this group closest to the group's own median target -
                # a representative already-visited structure, not the closest
                # thing in a static seed library that may be nowhere near here.
                group_indices = np.flatnonzero(mask)
                best_local = group_indices[
                    int(np.argmin(np.abs(uncovered_values[group_indices] - target_secondary)))
                ]
                metadata["seed_frame"] = {
                    "replica_id": int(uncovered_replica_ids[best_local]),
                    "step": int(uncovered_steps[best_local]),
                    "epoch_dir": str(epoch_dir) if epoch_dir is not None else None,
                    "primary_cv": float(uncovered_primary[best_local]),
                    "secondary_cv": float(uncovered_values[best_local]),
                    "umbrella_state_at_frame": (
                        int(uncovered_umbrella_state_ids[best_local])
                        if uncovered_umbrella_state_ids is not None else None
                    ),
                }
            actions.append(("tica_coverage_add", int(parent.state_id), params, reason, metadata))
    return actions


def _compute_mbar_weights_for_tica(
    primary_cv: np.ndarray,
    secondary_cv: np.ndarray,
    window_ids: np.ndarray,
    epoch_dir: Path,
    registry: "WindowStateRegistry",
    temperature_K: float = 298.0,
) -> np.ndarray:
    """Compute MBAR importance weights for tICA reweighting.

    Runs a standalone MBAR solve on the tICA dihedral-obs frames using each
    frame's primary (and optional secondary) CV value against all epoch-active
    registry states.  The resulting weights reweight the biased REUS ensemble
    toward the unbiased equilibrium distribution so that the tICA eigenvectors
    reflect true physical slow modes rather than the sampling distribution.

    Falls back silently to uniform weights (1/N) on any failure (pymbar
    unavailable, NaN CVs, numerical singularity, etc.).

    Parameters
    ----------
    primary_cv, secondary_cv : np.ndarray, shape (N,)
        Per-frame CV values from dihedral obs.  secondary_cv may be all-NaN
        when CV2 is inactive; those frames receive zero secondary-bias contribution.
    window_ids : np.ndarray, shape (N,)
        Local epoch window index for each frame.
    epoch_dir : Path
        Epoch directory used to load the epoch window map.
    registry : WindowStateRegistry
    temperature_K : float
        Simulation temperature in Kelvin.

    Returns
    -------
    np.ndarray, shape (N,), sums to 1.0
        MBAR importance weights.
    """
    N = len(primary_cv)
    uniform = np.full(N, 1.0 / N, dtype=np.float64)

    # Cannot reweight without primary CV (old obs files)
    if not np.isfinite(primary_cv).any():
        return uniform

    try:
        from pymbar import MBAR as _MBAR  # noqa: PLC0415
        from scipy.special import logsumexp as _logsumexp  # noqa: PLC0415
    except ImportError:
        return uniform

    try:
        # Map local window indices to registry state IDs
        epoch_wmap = _load_epoch_window_map(epoch_dir, registry)
        sample_state_ids = np.array([epoch_wmap.get(int(w), int(w)) for w in window_ids], dtype=np.int64)
        unique_sids = sorted(set(epoch_wmap.values()))
        states = [registry.get_state(int(sid)) for sid in unique_sids]
        states = [s for s in states if s is not None]
        if not states:
            return uniform

        K = len(states)
        primary_centers = np.array([float(s.primary_center) for s in states])
        primary_k = np.array([float(s.primary_k) for s in states])
        secondary_centers = np.array([
            np.nan if s.secondary_center is None else float(s.secondary_center)
            for s in states
        ])
        secondary_k = np.array([
            0.0 if s.secondary_k is None else float(s.secondary_k)
            for s in states
        ])

        # Reduced potential matrix u_nk[n, k] = beta * U_k(x_n), dimensionless
        kB_kcal = 1.987204e-3  # kcal/mol/K
        beta_kcal = 1.0 / (kB_kcal * temperature_K)

        primary_delta = primary_cv[:, None] - primary_centers[None, :]
        u_nk = beta_kcal * 0.5 * primary_k[None, :] * primary_delta ** 2

        # Secondary bias (only where both CV and state have finite values)
        has_sec_sample = np.isfinite(secondary_cv)
        has_sec_state = np.isfinite(secondary_centers) & (secondary_k > 0)
        if has_sec_sample.any() and has_sec_state.any():
            sec_delta = np.where(
                has_sec_sample[:, None] & has_sec_state[None, :],
                secondary_cv[:, None] - secondary_centers[None, :],
                0.0,
            )
            u_nk += beta_kcal * 0.5 * secondary_k[None, :] * sec_delta ** 2

        # Samples per state (from obs window map)
        state_id_to_ki = {int(sid): ki for ki, sid in enumerate(unique_sids)}
        N_k = np.zeros(K, dtype=np.int64)
        for sid_sample in sample_state_ids:
            ki = state_id_to_ki.get(int(sid_sample))
            if ki is not None:
                N_k[ki] += 1

        # pymbar convention: u_kn shape (K, N)
        mbar = _MBAR(u_nk.T, N_k, verbose=False)
        f_k = mbar.f_k  # dimensionless free energies, shape (K,)

        # MBAR weights for unbiased (zero-bias) ensemble:
        #   w_n ∝ 1 / sum_k N_k * exp(f_k - u_nk[n,k])
        log_N_k = np.where(N_k > 0, np.log(N_k.astype(np.float64)), -np.inf)
        log_N_f = log_N_k + f_k  # log(N_k) + f_k, shape (K,)
        # logsumexp over k for each n: log denominator
        log_denom = _logsumexp(log_N_f[None, :] - u_nk, axis=1)  # shape (N,)
        log_w = -log_denom
        log_w -= _logsumexp(log_w)  # log-normalise
        weights = np.exp(log_w)

        # Sanity: all finite, non-negative, sum to ~1
        if not np.isfinite(weights).all() or (weights < 0).any():
            return uniform
        weights = np.clip(weights, 0.0, None)
        weights /= weights.sum()
        return weights

    except Exception:
        return uniform


def _maybe_update_tica_cvaux(epoch: int, epoch_dir: Path, adaptive_dir: Path, args, *, registry=None) -> dict:
    """Refit tICA from epoch observations and update args for the next epoch.

    Completely opt-in: returns an empty dict immediately if tica_obs_interval == 0
    or the epoch is not in the tica_update_after_epochs list.

    When active:
    1. Loads dihedral obs from epoch_dir/tica_obs/
    2. Fits tICA (enforcing sign continuity against the previous model)
    3. Saves TICAResult to adaptive_dir/tica_state.json
    4. Sets args.tica_state_file so the next epoch's force builder loads the new weights
    5. Sets args.tica_cv_version for downstream MBAR cross-epoch filtering

    Returns a summary dict stored in the epoch summary JSON.
    """
    tica_obs_interval = int(getattr(args, "tica_obs_interval", 0) or 0)
    if tica_obs_interval <= 0:
        return {}

    update_after = getattr(args, "tica_update_after_epochs", None)
    epochs_per_cycle = int(getattr(args, "tica_epochs_per_cycle", 0) or 0)

    should_update = False
    if update_after is not None and int(epoch) in [int(e) for e in update_after]:
        should_update = True
    if epochs_per_cycle > 0 and (int(epoch) + 1) % epochs_per_cycle == 0:
        should_update = True

    if not should_update:
        return {}

    tica_dir = epoch_dir / "tica_obs"
    if not tica_dir.exists() or not any(tica_dir.glob("dihedral_obs_*.npz")):
        print(f"    tICA: no dihedral observations found in {tica_dir}; skipping update for epoch {epoch}")
        return {"status": "skipped_no_obs", "epoch": int(epoch)}

    lag = int(getattr(args, "tica_lag_frames", 50) or 50)
    try:
        from .tica import compute_combined_tica_from_epoch_obs, load_epoch_dihedral_obs, window_tica_centers, TICAResult  # noqa: F401 – all used below
    except ImportError as exc:
        print(f"    tICA: import failed ({exc}); skipping update")
        return {"status": "skipped_import_error", "error": str(exc)}

    # Load existing model for sign continuity
    prev_result = None
    prev_state_path = str(getattr(args, "tica_state_file", "") or "")
    if prev_state_path and Path(prev_state_path).exists():
        try:
            prev_result = TICAResult.load(prev_state_path)
        except Exception:
            prev_result = None

    # Determine torsion indices from a previous model or from topology
    phi_indices = list(prev_result.phi_torsion_indices) if prev_result else []
    psi_indices = list(prev_result.psi_torsion_indices) if prev_result else []
    if not phi_indices and not psi_indices:
        # First-ever fit (no prev_result), or a prior fit saved before this
        # fallback existed: nothing to carry forward. The "or from topology"
        # half of the comment above was never implemented, so TICAResult got
        # saved with permanently empty index lists -- harmless for the
        # window-center update (that only ever projects the *already*
        # feature-extracted dihedral_obs arrays, never round-tripping through
        # the saved indices) but it silently makes the saved TICAResult
        # unusable for projecting any *new* structure (e.g. re-scoring a
        # candidate seed PDB against tIC1). The two-stage torsion-pca ->
        # tica-linear design fits both CV2 stages on the identical backbone
        # phi/psi basis (see helptext: "Epoch 0 (torsion-pca) ... Epoch 1+
        # (tica-linear)"), so the one-time bootstrap CV2 fit's indices are the
        # correct -- and only available -- source.
        bootstrap_state = adaptive_dir / "epoch_000" / "tica" / "bootstrap_torsion_cv.json"
        if bootstrap_state.exists():
            try:
                _bs = read_json_file(bootstrap_state)
                phi_indices = [tuple(t) for t in _bs.get("phi_torsion_indices", [])]
                psi_indices = [tuple(t) for t in _bs.get("psi_torsion_indices", [])]
            except Exception:
                phi_indices, psi_indices = [], []

    # Compute MBAR importance weights to reweight biased REUS frames to unbiased distribution.
    # Falls back to uniform (unweighted tICA) silently on any failure.
    mbar_weights = None
    mbar_reweighted = False
    if registry is not None:
        try:
            _, _wids_tmp, _pcv_tmp, _scv_tmp, _, _, _ = load_epoch_dihedral_obs(epoch_dir)
            if np.isfinite(_pcv_tmp).any():
                temp_K = _args_temperature_k(args)
                mbar_weights = _compute_mbar_weights_for_tica(
                    _pcv_tmp, _scv_tmp, _wids_tmp, epoch_dir, registry, temp_K
                )
                mbar_reweighted = True
                print(
                    f"    tICA: MBAR reweighting from {int(np.isfinite(_pcv_tmp).sum())} frames "
                    f"across {len(np.unique(_wids_tmp))} windows"
                )
            else:
                print("    tICA: no primary_cv in obs (old files); using unweighted tICA")
        except Exception as _mw_exc:
            print(f"    tICA: MBAR weight computation failed ({_mw_exc}); using unweighted tICA")

    tica_n_components = int(getattr(args, "tica_component_count", 1) or 1)
    try:
        result = compute_combined_tica_from_epoch_obs(
            epoch_dir, lag, tica_n_components, phi_indices, psi_indices,
            previous_result=prev_result, weights=mbar_weights,
        )
    except Exception as exc:
        print(f"    tICA: fitting failed ({exc}); skipping update for epoch {epoch}")
        return {"status": "skipped_fit_error", "error": str(exc), "epoch": int(epoch)}

    eigenvalue_threshold = float(getattr(args, "tica_min_eigenvalue", 0.0) or 0.0)
    if result.eigenvalue < eigenvalue_threshold:
        print(
            f"    tICA: eigenvalue {result.eigenvalue:.4f} below threshold {eigenvalue_threshold:.4f}; "
            f"skipping update for epoch {epoch}"
        )
        return {"status": "skipped_low_eigenvalue", "eigenvalue": float(result.eigenvalue), "epoch": int(epoch)}

    state_path = adaptive_dir / "tica_state.json"
    result.save(state_path)
    _n_comp_note = f", combining top {tica_n_components} components" if tica_n_components > 1 else ""
    print(f"    tICA: fitted from epoch {epoch} ({result.n_samples} samples, eigenvalue {result.eigenvalue:.4f}{_n_comp_note}) -> {state_path}")

    # Compute per-window tIC1 medians for registry secondary_center update.
    per_window_centers: Dict[int, float] = {}
    try:
        X_all, window_ids, _, _, _, _, _ = load_epoch_dihedral_obs(epoch_dir)
        per_window_centers = window_tica_centers(X_all, window_ids, result)
    except Exception as _wc_exc:
        print(f"    tICA: per-window center computation failed ({_wc_exc}); registry centers will not be updated")

    args.tica_state_file = str(state_path)
    version_tag = f"v{epoch + 1}"
    args.tica_cv_version = version_tag

    return {
        "status": "updated",
        "epoch": int(epoch),
        "eigenvalue": float(result.eigenvalue),
        "n_samples": int(result.n_samples),
        "lag_frames": int(lag),
        "n_components": int(tica_n_components),
        "state_file": str(state_path),
        "version": version_tag,
        "mbar_reweighted": mbar_reweighted,
        "per_window_tic1_centers": {int(k): float(v) for k, v in per_window_centers.items()},
    }


def _maybe_recalibrate_gamd_boost(
    epoch: int,
    epoch_dir: Path,
    args,
    global_shared_gamd_dir: Optional[Path],
) -> dict:
    """Recalibrate the campaign-shared GaMD envelope from epoch 0's real sampling.

    Epoch 0 already runs full GaMD-boosted production to bootstrap tICA, so its
    replicas' actual per-boost-group potential-energy statistics (written by
    gareus/production.py to ``gamd_production_envelope_stats.json`` when
    ``_adaptive_phase_info`` marks epoch_index == 0) are a real, in-campaign
    measurement of the envelope -- unlike the throwaway calibration recon, which
    only ever saw a handful of pre-production windows.  This recomputes
    k0/threshold/Vmax/Vmin/Vavg/sigmaV per boost group from that real data
    (same pure-Python formulas the recon uses) and overwrites the exported
    shared envelope so every epoch 1+ worker loads the recalibrated version
    instead of the original recon's guess.

    Fires at most once (only ``epoch == 0``), is a no-op if GaMD/shared-envelope
    is disabled, and degrades to a no-op (not an error) if no stats files were
    written (e.g. epoch 0 was itself resumed from a checkpoint that predates
    this feature). Reweighting validity does not depend on which envelope was
    active for a given frame (per-frame delta-V is recorded), so recalibrating
    is efficiency/variance-only -- exactly like the existing shared-envelope
    reuse policy, just recalibrated from real data instead of frozen forever.
    """
    if int(epoch) != 0:
        return {}
    if global_shared_gamd_dir is None:
        return {}
    if not bool(getattr(args, "adaptive_production_gamd_recalibrate_after_epoch0", True)):
        return {}
    if "gamd" not in str(getattr(args, "run_mode", "") or "").lower():
        return {}

    stat_files = sorted(Path(epoch_dir).rglob("gamd_production_envelope_stats.json"))
    if not stat_files:
        return {"status": "skipped_no_stats", "epoch": int(epoch)}

    from .gamd_calibration import WindowEnergyStats, pool_window_stats, compute_group_calibration, overwrite_physics_globals

    globals_path = Path(global_shared_gamd_dir) / "shared_gamd_setup_globals.json"
    payload = read_json_file(globals_path, None)
    if not isinstance(payload, dict) or not payload.get("all_globals"):
        return {"status": "skipped_no_shared_envelope", "epoch": int(epoch), "globals_path": str(globals_path)}
    all_globals = dict(payload.get("all_globals", {}) or {})
    interesting_globals = dict(payload.get("interesting_globals", {}) or {})

    per_group_raw: Dict[str, List[WindowEnergyStats]] = {}
    for sf in stat_files:
        file_payload = read_json_file(sf, None)
        if not isinstance(file_payload, dict):
            continue
        for group, window_stats in (file_payload.get("per_group_window_stats", {}) or {}).items():
            for ws in window_stats or []:
                try:
                    per_group_raw.setdefault(str(group), []).append(WindowEnergyStats(
                        group=str(group), window=int(ws["window"]), vmax=float(ws["vmax"]),
                        vmin=float(ws["vmin"]), mean=float(ws["mean"]), var=float(ws["var"]), n=int(ws["n"]),
                    ))
                except (KeyError, TypeError, ValueError):
                    continue

    if not per_group_raw:
        return {"status": "skipped_no_usable_stats", "epoch": int(epoch), "n_stats_files": len(stat_files)}

    boost_type = str(getattr(args, "gamd_boost_type", "lower-total"))
    calibrations = {}
    group_reports = {}
    for group, stats in per_group_raw.items():
        sigma0_key = f"sigma0_{group}"
        if sigma0_key not in all_globals:
            continue
        try:
            envelope = pool_window_stats(stats)
            calib = compute_group_calibration(boost_type, envelope, float(all_globals[sigma0_key]))
        except Exception as exc:
            group_reports[group] = {"status": "failed", "error": str(exc)}
            continue
        calibrations[group] = calib
        group_reports[group] = {
            "status": "recalibrated",
            "n_windows": envelope.n_windows,
            "n_total_samples": envelope.n_total,
            "sigmaV_before_kj_mol": float(all_globals.get(f"sigmaV_{group}", float("nan"))),
            "sigmaV_after_kj_mol": float(calib.sigmav),
            "k0_before": float(all_globals.get(f"k0_{group}", float("nan"))),
            "k0_after": float(calib.k0),
            "threshold_energy_before_kj_mol": float(all_globals.get(f"threshold_energy_{group}", float("nan"))),
            "threshold_energy_after_kj_mol": float(calib.threshold_energy),
            "boosted": bool(calib.boosted),
        }

    if not calibrations:
        return {"status": "skipped_no_matching_groups", "epoch": int(epoch), "n_stats_files": len(stat_files)}

    all_globals = overwrite_physics_globals(all_globals, calibrations)
    interesting_globals = overwrite_physics_globals(interesting_globals, calibrations) if interesting_globals else interesting_globals
    envelope_version = int(payload.get("gamd_envelope_version", 1) or 1) + 1
    history = list(payload.get("recalibration_history", []) or [])
    history.append({"recalibrated_from_epoch": int(epoch), "groups": group_reports, "envelope_version": envelope_version})

    new_payload = dict(payload)
    new_payload.update({
        "all_globals": all_globals,
        "interesting_globals": interesting_globals,
        "gamd_envelope_version": envelope_version,
        "recalibration_history": history,
        "description": str(payload.get("description", "")) + (
            " | Recalibrated after epoch 0 from real production sampling "
            "(see recalibration_history)."
        ),
    })
    write_json(globals_path, _json_ready(new_payload))
    print(
        f"    GaMD recalibration: updated shared envelope (version {envelope_version}) from "
        f"epoch 0 sampling ({len(stat_files)} worker file(s)) -> {globals_path}"
    )
    for group, rep in group_reports.items():
        if rep.get("status") == "recalibrated":
            print(
                f"      {group}: sigmaV {rep['sigmaV_before_kj_mol']:.1f} -> {rep['sigmaV_after_kj_mol']:.1f} kJ/mol, "
                f"k0 {rep['k0_before']:.3f} -> {rep['k0_after']:.3f}"
            )

    return {
        "status": "recalibrated",
        "epoch": int(epoch),
        "envelope_version": envelope_version,
        "n_stats_files": len(stat_files),
        "shared_gamd_dir": str(global_shared_gamd_dir),
        "groups": group_reports,
    }


def _epoch_sample_sources(
    adaptive_dir: Path,
    include_epochs: bool,
    *,
    pilot_dirs: "List[Path] | None" = None,
    tica_cv_version: "Optional[str]" = None,
) -> List[Tuple[str, Path]]:
    adaptive_dir = Path(adaptive_dir)
    sources: List[Tuple[str, Path]] = []
    for pilot in (pilot_dirs or []):
        sources.extend(_sample_sources_from_run_root(f"pilot::{Path(pilot).name}", Path(pilot)))
    if include_epochs:
        for epoch_dir in sorted(adaptive_dir.glob("epoch_[0-9][0-9][0-9]")):
            if tica_cv_version and tica_cv_version != "disabled":
                marker = epoch_dir / "tica_cv_version.txt"
                epoch_version = marker.read_text().strip() if marker.exists() else "disabled"
                if epoch_version != tica_cv_version:
                    print(
                        f"    tICA MBAR guard: skipping {epoch_dir.name} "
                        f"(CV version {epoch_version!r} != {tica_cv_version!r})"
                    )
                    continue
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
    include_epochs: bool = True,
    pilot_dirs: "List[Path] | None" = None,
    output_prefix: str = "adaptive_union_mbar",
    tica_cv_version: "Optional[str]" = None,
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
    for source_label, sample_dir in _epoch_sample_sources(
        adaptive_dir, include_epochs=include_epochs, pilot_dirs=pilot_dirs, tica_cv_version=tica_cv_version
    ):
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
            v_pep = _float_or_none(row.get("v_pep_kj_mol"))
            v_dih = _float_or_none(row.get("v_dih_kj_mol"))
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
                "v_pep_kj_mol": "" if v_pep is None else float(v_pep),
                "v_dih_kj_mol": "" if v_dih is None else float(v_dih),
                "usable_for_mbar": int(source_label.startswith("final") or include_epochs),
            })

    if not sample_rows:
        raise RuntimeError(f"no usable sample rows found under {adaptive_dir}")

    # Per-state equilibration-discard + autocorrelation subsampling.
    # Group row indices by sampled_state_id, thin each group's cv_A trace with
    # equilibrated_subsample_indices, then rebuild sample_rows from kept indices.
    # All downstream numpy arrays are derived from sample_rows so alignment is
    # preserved automatically.
    from .mbar_subsample import equilibrated_subsample_indices as _esi  # noqa: PLC0415
    _state_to_indices: Dict[int, List[int]] = {}
    for _gi, _row in enumerate(sample_rows):
        _sid = int(_row["sampled_state_id"])
        _state_to_indices.setdefault(_sid, []).append(_gi)
    _kept_global: List[int] = []
    _subsample_counts: Dict[str, Any] = {}
    for _sid, _idx_list in _state_to_indices.items():
        _trace = np.asarray([float(sample_rows[i]["cv_A"]) for i in _idx_list], dtype=np.float64)
        _keep = _esi(_trace)
        _kept_global.extend(_idx_list[k] for k in _keep.tolist())
        _subsample_counts[str(_sid)] = {"raw": len(_idx_list), "kept": int(len(_keep))}
    sample_rows = [sample_rows[i] for i in sorted(_kept_global)]

    cv_values = np.asarray([float(r["cv_A"]) for r in sample_rows], dtype=np.float64)
    secondary_values = np.asarray([
        np.nan if r["secondary_cv"] == "" else float(r["secondary_cv"]) for r in sample_rows
    ], dtype=np.float64)
    beta_values = np.asarray([
        np.nan if r["beta_1_over_kJ_mol"] == "" else float(r["beta_1_over_kJ_mol"]) for r in sample_rows
    ], dtype=np.float64)
    beta_finite = beta_values[np.isfinite(beta_values)]
    beta = float(beta_finite[0]) if beta_finite.size else float("nan")
    v_pep_values = np.asarray([
        np.nan if r.get("v_pep_kj_mol", "") == "" else float(r["v_pep_kj_mol"]) for r in sample_rows
    ], dtype=np.float64)
    v_dih_values = np.asarray([
        np.nan if r.get("v_dih_kj_mol", "") == "" else float(r["v_dih_kj_mol"]) for r in sample_rows
    ], dtype=np.float64)

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

    # λ-ladder boost term: a closed form of each sample's own stored raw
    # channel energies (v_pep, v_dih) evaluated under every state's own
    # gamd_lambda. Zero (not merely small) whenever no registry state carries
    # a nonzero rung -- the common case (plain umbrella/REUS, plain GaMD) is
    # untouched. When a rung IS active, missing raw energies must fail loudly
    # rather than silently reweight without the term (see reconstruct_bias_matrix's
    # docstring for the same invariant on the loader path).
    state_lambdas = np.asarray([float(getattr(s, "gamd_lambda", 0.0) or 0.0) for s in states], dtype=np.float64)
    gamd_boost_kj_nk = np.zeros_like(umbrella_bias_kj)
    if np.any(state_lambdas > 0.0):
        if not np.any(np.isfinite(v_pep_values)):
            raise ValueError(
                "registry states carry gamd_lambda > 0 but no sample row has a finite v_pep_kj_mol; "
                "the λ-ladder cannot be reweighted without the raw channel energies"
            )
        if not np.any(np.isfinite(v_dih_values)):
            raise ValueError(
                "registry states carry gamd_lambda > 0 but no sample row has a finite v_dih_kj_mol "
                "(v_pep_kj_mol is present); the λ-ladder cannot be reweighted without the raw channel energies"
            )
        from .pep_gamd import PepGamdEnvelope, pep_gamd_boost_matrix_kj  # noqa: PLC0415
        envelope_path = adaptive_dir / "global_shared_gamd_setup" / "shared_gamd_setup_globals.json"
        if not envelope_path.exists():
            raise ValueError(
                f"registry states carry gamd_lambda > 0 but the frozen GaMD envelope "
                f"{envelope_path} does not exist; v_pep/v_dih cannot be reweighted under the "
                f"ladder without it"
            )
        envelope = PepGamdEnvelope.from_json(envelope_path)
        gamd_boost_kj_nk = pep_gamd_boost_matrix_kj(v_pep_values, v_dih_values, state_lambdas, envelope).T
    total_bias_kj = umbrella_bias_kj + gamd_boost_kj_nk

    if math.isfinite(beta):
        umbrella_reduced_bias_nk = float(beta) * total_bias_kj
    else:
        umbrella_reduced_bias_nk = np.full_like(total_bias_kj, np.nan)

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
            "cv_A", "secondary_cv", "beta_1_over_kJ_mol", "gamd_boost_total_kj_mol",
            "v_pep_kj_mol", "v_dih_kj_mol", "usable_for_mbar",
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
        gamd_boost_kj_nk=gamd_boost_kj_nk,
        state_lambdas=state_lambdas,
        v_pep_kj_mol=v_pep_values,
        v_dih_kj_mol=v_dih_values,
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
        "subsample_counts_per_state": _subsample_counts,
        "gamd_ladder": bool(np.any(state_lambdas > 0.0)),
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
        else:
            # Same off-target check collect_epoch_diagnostics runs during the
            # adaptive-discovery loop (where it reliably flags a genuine
            # restraint-too-soft window), but that check was never wired into
            # the frozen final phase: the pooled final samples are exactly
            # what feeds the PMF, so an off-target window here matters more,
            # not less, than during discovery.
            state_obj = registry.get_state(int(state_id))
            if state_obj is not None:
                dev_sigma = _target_deviation_sigma(cv_mean, cv_std, state_obj.primary_center)
                diag.primary_target_deviation_sigma = dev_sigma
                if dev_sigma is not None and dev_sigma >= float(policy.max_target_deviation_sigma):
                    diag.warnings.append("off_target_primary")
                if state_obj.secondary_center is not None:
                    sec_dev_sigma = _target_deviation_sigma(smean, sstd, state_obj.secondary_center)
                    diag.secondary_target_deviation_sigma = sec_dev_sigma
                    if sec_dev_sigma is not None and sec_dev_sigma >= float(policy.max_target_deviation_sigma):
                        diag.warnings.append("off_target_secondary")
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
    off_target_states = []
    for state in final_diagnostics.get("states", []) or []:
        sid = int(state.get("state_id", -1))
        count = int(state.get("sample_count", 0) or 0)
        if count < int(policy.final_min_samples_per_state):
            low_states.append({"state_id": sid, "sample_count": count, "minimum": int(policy.final_min_samples_per_state)})
        bsd = state.get("gamd_boost_sd_kcal_mol")
        if bsd is not None and float(bsd) > float(policy.max_gamd_boost_sd_kcal_mol):
            high_boost.append({"state_id": sid, "boost_sd_kcal_mol": float(bsd), "maximum": float(policy.max_gamd_boost_sd_kcal_mol)})
        state_warnings = state.get("warnings", []) or []
        if "off_target_primary" in state_warnings or "off_target_secondary" in state_warnings:
            off_target_states.append({
                "state_id": sid,
                "primary_target_deviation_sigma": state.get("primary_target_deviation_sigma"),
                "secondary_target_deviation_sigma": state.get("secondary_target_deviation_sigma"),
                "maximum_sigma": float(policy.max_target_deviation_sigma),
                "warnings": [w for w in state_warnings if w.startswith("off_target_")],
            })
    if low_states:
        needs_more_sampling.append(f"{len(low_states)} state(s) have fewer than {policy.final_min_samples_per_state} final samples")
        recommendations.append("Run additional frozen final sampling with the same active window table.")
    if high_boost:
        needs_more_sampling.append(f"{len(high_boost)} state(s) exceed the GaMD boost SD threshold")
        recommendations.append("Inspect GaMD boost distributions; consider lower sigma0 or final unboosted/weakly boosted production.")
    if off_target_states:
        needs_more_sampling.append(
            f"{len(off_target_states)} final state(s) sample more than {policy.max_target_deviation_sigma:g} "
            "of their own sigma away from their restraint target"
        )
        recommendations.append(
            "Restraint is too soft relative to the free-energy gradient at these windows -- "
            "raise the umbrella force constant (or use a per-window table) for a future run; "
            "this run's PMF should treat these windows' contribution cautiously."
        )

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
        "off_target_states": off_target_states,
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


def quality_gate_fixable_by_more_final_sampling(quality_gate: Dict[str, Any]) -> bool:
    """True only for findings more frozen-final sampling can actually fix.

    ``status == "needs_more_sampling"`` also covers off_target_states and
    high_boost_states, neither of which improves by adding more samples at
    the same restraint center/GaMD calibration -- an off-target window is a
    restraint-vs-landscape-stiffness problem, and pooling in more samples
    from the same collapsed basin can even inflate the achieved std enough
    to shrink dev_sigma back under threshold on a later round, silently
    clearing the flag with no physical change. The frozen-final extension
    loop must not treat those as a reason to keep spinning.
    """
    return bool(quality_gate.get("low_sample_states")) or bool(quality_gate.get("weak_edges"))


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
    off_target = payload.get("off_target_states", []) or []
    if off_target:
        lines.append("## Off-target final states")
        lines.append("Mean sampled CV sits more than the configured sigma threshold away from the window's own restraint target -- the restraint is too soft relative to the free-energy gradient there.")
        lines.append("")
        lines.append("| state_id | primary dev (sigma) | secondary dev (sigma) | max allowed |")
        lines.append("|---:|---:|---:|---:|")
        for row in off_target[:100]:
            lines.append(
                f"| {row.get('state_id')} | {row.get('primary_target_deviation_sigma')} | "
                f"{row.get('secondary_target_deviation_sigma')} | {row.get('maximum_sigma')} |"
            )
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def _bridge_placement_prediction(
    s1: WindowState,
    s2: WindowState,
    *,
    temperature_K: float = 298.0,
    healthy_spacing_sigma: float = 1.5,
) -> Dict[str, Any]:
    """Predict, before placing anything, whether a bridge can reach both endpoints.

    Two umbrella windows only exchange usefully while the spacing between their
    centres is within roughly ``healthy_spacing_sigma`` of each window's own
    harmonic width ``sigma = sqrt(kT/k)``.  A single midpoint bridge sits
    ``gap / 2`` from each endpoint, so its reachability is
    ``(gap / 2) / sigma_endpoint`` -- evaluated against *each endpoint's own*
    sigma, because a stiff endpoint has a small sigma and is therefore the hard
    one to reach even though the midpoint is equidistant from both.

    Evaluated per CV axis and reduced on the worst axis, never averaged: on
    chignolin_6's edge 14-20 the primary axis is a comfortable 0.45 sigma (both
    endpoints k1 = 200, centres 0.0654 vs 0.1141) while the secondary axis is
    12.1 / 10.8 sigma.  Averaging the two would have declared that edge healthy.

    ``bridges_needed`` is how many evenly spaced windows it takes for every
    resulting sub-interval to fall inside the healthy band:
    ``ceil(gap / (healthy * sigma_min)) - 1``, floored at 1 so an edge the
    caller has already judged weak always gets at least the historical single
    midpoint.  ``sigma_min`` is the tighter of the two endpoints' sigmas -- the
    intermediate windows' own sigmas are not known yet (their springs are
    interpolated from the endpoints), so the tighter endpoint is the honest
    conservative choice.

    ``min_useful_bridges`` is the weaker, allocation-facing question: how many
    evenly spaced windows it takes for the chain to reach *at least one* of the
    two endpoints, i.e. ``min`` over the endpoints of
    ``ceil(gap / (healthy * sigma_endpoint)) - 1``.  A chain hanging off one
    endpoint is a partial repair the next epoch can extend from; a chain
    hanging off neither is an island that reconnects nothing while still
    costing a window slot and real MD.  It is always ``<= bridges_needed``
    (reaching one endpoint cannot be harder than reaching both) and never
    below 1.

    Reachability is reduced PER ENDPOINT ACROSS ALL AXES, and only then
    compared between the two endpoints -- never per axis first.  Taking
    ``min(ratio_i, ratio_j)`` inside each axis and then the worst axis is the
    obvious-looking reduction and it is wrong: an endpoint that is soft on the
    primary axis and stiff on the secondary, paired with an endpoint that is
    the mirror image, gives every axis one nearby endpoint while NEITHER
    endpoint is actually within reach on both axes at once.  See
    ``tests/test_bridge_placement_veto.py::
    test_reachability_reduces_per_endpoint_over_all_axes_not_per_axis``.
    """
    kbt_kcal = _KB_KCAL_PER_MOL_K * float(temperature_K)
    healthy = max(1.0e-9, float(healthy_spacing_sigma))
    axes: Dict[str, Dict[str, float]] = {}
    for label, c_i, k_i, c_j, k_j in (
        ("primary", s1.primary_center, s1.primary_k, s2.primary_center, s2.primary_k),
        ("secondary", s1.secondary_center, s1.secondary_k, s2.secondary_center, s2.secondary_k),
    ):
        if c_i is None or c_j is None or not k_i or not k_j:
            continue
        try:
            gap = abs(float(c_j) - float(c_i))
            sigma_i = math.sqrt(kbt_kcal / float(k_i))
            sigma_j = math.sqrt(kbt_kcal / float(k_j))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if not (math.isfinite(gap) and math.isfinite(sigma_i) and math.isfinite(sigma_j)):
            continue
        if sigma_i <= 0.0 or sigma_j <= 0.0:
            continue
        sigma_min = min(sigma_i, sigma_j)
        ratio_i = (0.5 * gap) / sigma_i
        ratio_j = (0.5 * gap) / sigma_j
        axes[label] = {
            "gap": gap,
            "sigma_i": sigma_i,
            "sigma_j": sigma_j,
            "midpoint_spacing_sigma_i": ratio_i,
            "midpoint_spacing_sigma_j": ratio_j,
            "worst_midpoint_spacing_sigma": max(ratio_i, ratio_j),
            "intervals_needed": max(1, int(math.ceil(gap / (healthy * sigma_min)))),
            # How many sub-intervals this axis alone needs before the chain's
            # outermost window is inside the healthy band of endpoint i (resp.
            # j) specifically -- the per-endpoint input to min_useful_bridges.
            "intervals_to_reach_i": max(1, int(math.ceil(gap / (healthy * sigma_i)))),
            "intervals_to_reach_j": max(1, int(math.ceil(gap / (healthy * sigma_j)))),
        }
    if not axes:
        # Nothing measurable (e.g. a zero/absent spring on both axes): fall back
        # to the historical single midpoint rather than refuse to bridge.
        # min_useful_bridges = 1 / reaches_either = True on purpose: the veto is
        # evidence-based and fires only where the geometry PROVES a chain
        # reaches neither endpoint.  An unmeasurable pair is not that proof, so
        # it keeps the historical placement rather than being refused.
        return {
            "axes": {},
            "worst_axis": None,
            "worst_midpoint_spacing_sigma": float("nan"),
            "worst_midpoint_spacing_sigma_i": float("nan"),
            "worst_midpoint_spacing_sigma_j": float("nan"),
            "healthy_spacing_sigma": healthy,
            "single_bridge_sufficient": True,
            "single_bridge_reaches_i": True,
            "single_bridge_reaches_j": True,
            "single_bridge_reaches_either": True,
            "bridges_needed": 1,
            "min_useful_bridges": 1,
        }
    worst_axis = max(axes, key=lambda a: axes[a]["worst_midpoint_spacing_sigma"])
    worst_ratio = axes[worst_axis]["worst_midpoint_spacing_sigma"]
    intervals = max(int(a["intervals_needed"]) for a in axes.values())
    # Per endpoint, worst axis first (see the docstring for why the transposed
    # reduction is wrong), and only then the easier of the two endpoints.
    worst_ratio_i = max(float(a["midpoint_spacing_sigma_i"]) for a in axes.values())
    worst_ratio_j = max(float(a["midpoint_spacing_sigma_j"]) for a in axes.values())
    reach_i = max(int(a["intervals_to_reach_i"]) for a in axes.values()) - 1
    reach_j = max(int(a["intervals_to_reach_j"]) for a in axes.values()) - 1
    return {
        "axes": axes,
        "worst_axis": worst_axis,
        "worst_midpoint_spacing_sigma": worst_ratio,
        "worst_midpoint_spacing_sigma_i": worst_ratio_i,
        "worst_midpoint_spacing_sigma_j": worst_ratio_j,
        "healthy_spacing_sigma": healthy,
        "single_bridge_sufficient": bool(worst_ratio <= healthy),
        "single_bridge_reaches_i": bool(worst_ratio_i <= healthy),
        "single_bridge_reaches_j": bool(worst_ratio_j <= healthy),
        "single_bridge_reaches_either": bool(min(worst_ratio_i, worst_ratio_j) <= healthy),
        "bridges_needed": max(1, intervals - 1),
        "min_useful_bridges": max(1, min(reach_i, reach_j)),
    }


def _finite_or_none(value: Any) -> Optional[float]:
    """``float(value)`` when it is finite, else ``None``.

    For numbers that go into a JSON artefact: ``write_json`` -> ``json.dumps``
    keeps ``allow_nan`` on, so a NaN would be written as the bare token ``NaN``
    -- readable by Python's own json, rejected by every strict parser (jq, any
    browser, most other languages).  ``null`` says the same thing portably.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _interpolate_bridge_axis(c1: float, c2: float, frac: float) -> float:
    """Interpolate one CV axis (or spring) along ``c1 -> c2`` at ``frac``.

    ``frac == 0.5`` is special-cased to the plain average ``0.5*(c1+c2)`` on
    purpose.  The general form ``c1 + frac*(c2-c1)`` is *not* bit-identical to
    it: measured over 200,000 random double pairs the two differ by one ULP for
    6.7% of them.  One ULP in a restraint centre has no physical consequence
    whatsoever, but the multi-bridge placement's own comment (and its test's
    name) claim the historical single-midpoint case is unchanged bit-for-bit,
    and a claim that cheap to honour should be honoured rather than softened to
    "close enough" -- otherwise the test that guards it can only ever assert
    something weaker than what it says.
    """
    if frac == 0.5:
        return 0.5 * (float(c1) + float(c2))
    return float(c1) + float(frac) * (float(c2) - float(c1))


def propose_actions_from_diagnostics(
    registry: WindowStateRegistry,
    diagnostics: Dict[str, Any],
    policy: Optional[AdaptiveDecisionPolicy] = None,
    *,
    temperature_K: float = 298.0,
    secondary_k_max: Optional[float] = None,
    bridge_plan_out: Optional[List[Dict[str, Any]]] = None,
) -> List[Tuple]:
    """Propose registry-changing actions from one epoch's diagnostics.

    ``bridge_plan_out``, when given, is filled with one record per weak edge
    considered for bridging -- what the geometry asked for, what the budget
    allowed, what was actually placed, and why an edge got nothing.  The driver
    threads it into ``write_epoch_action_report`` so an edge nothing could be
    done about survives the epoch as a written artefact rather than only as a
    log line that scrolls away.  It is appended to, never read, so a caller
    that does not want it can leave it out.
    """
    policy = policy or AdaptiveDecisionPolicy()
    actions: List[Tuple] = []
    state_rows = {int(r["state_id"]): r for r in diagnostics.get("states", [])}
    edge_rows = diagnostics.get("edges", [])

    # 1. Add bridge states at weak edges.  This is the safest first adaptive
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

    # Two passes, not one.  The first cut of the multi-bridge placement walked
    # the sorted weak edges greedily -- each edge took
    # min(needed, budget_left) -- so the single worst edge could consume the
    # whole max_new_windows_per_epoch budget and leave every other weak edge
    # with nothing.  Measured against chignolin_6's own diagnostics that is a
    # behaviour regression, not an improvement: its worst weak edge needs 16
    # bridges, so with the run's own budget of 4 it took all four and the four
    # other weak edges got zero -- where the historical single-midpoint
    # behaviour repaired four different edges.  Multi-bridging is still right in
    # principle (one midpoint genuinely cannot span 12 sigma), it just must not
    # be funded by starving every other weak edge.
    #
    # So: predict every candidate edge's need first, then hand the budget out
    # round-robin, worst overlap first inside each round.  Round 1 reproduces
    # the historical "one midpoint per weak edge" allocation exactly; only
    # genuinely spare budget deepens an edge that needs more than one.
    max_new = max(0, int(policy.max_new_windows_per_epoch))
    candidates: List[Dict[str, Any]] = []
    for edge in weak_edges:
        s1 = registry.get_state(int(edge["state_i"]))
        s2 = registry.get_state(int(edge["state_j"]))
        if s1 is None or s2 is None:
            continue
        # Verify the placement before committing to it.  A midpoint bridge is
        # only useful while it is within ~1-1.5 harmonic widths of *each*
        # endpoint it is supposed to reconnect, and a stiff endpoint's width
        # sigma = sqrt(kT/k) can be very small.  chignolin_6's state 24 was
        # created to repair edge 14-20 at overlap 0.127 and its measured
        # overlap to state 20 came out 0.055 -- worse than what it was built to
        # fix -- because the midpoint sat ~12 and ~11 sigma from the two
        # endpoints. Nothing checked, and nothing warned.
        prediction = _bridge_placement_prediction(
            s1, s2,
            temperature_K=float(temperature_K),
            healthy_spacing_sigma=float(policy.bridge_healthy_spacing_sigma),
        )
        # ``cap`` is the most windows this edge may ever receive;
        # ``min_useful`` the fewest that make the placement reconnect anything
        # at all (reach at least one endpoint -- see _bridge_placement_prediction).
        # An edge is repairable *this epoch* only if that floor fits both.
        cap = int(prediction["bridges_needed"]) if bool(policy.bridge_multi_window) else 1
        min_useful = int(prediction["min_useful_bridges"])
        candidates.append({
            "edge": edge,
            "s1": s1,
            "s2": s2,
            "prediction": prediction,
            "needed": int(prediction["bridges_needed"]),
            "cap": cap,
            "min_useful": min_useful,
            # Deliberately measured against the WHOLE budget, not what happens
            # to be left when this edge is reached: the classification is then
            # a property of the edge and the policy, identical in the log, in
            # the plan record and in the allocation, rather than an artefact of
            # iteration order.
            "repairable_this_epoch": bool(min_useful <= cap and min_useful <= max_new),
            "funded_repairable": False,
            "allocated": 0,
        })

    # bridge_multi_window=False caps the *allocation* at one bridge per edge and
    # deliberately leaves ``needed`` alone: the true geometric requirement still
    # goes into the new state's reason string and into the shortfall warning, so
    # turning the placement off never hides the diagnosis.
    #
    # Two phases, and the first one is the fix for "a hopeless edge always takes
    # a round-1 slot ahead of a fixable edge's second bridge".  Flat round-robin
    # is fair per EDGE but not per outcome: it funds a first window everywhere
    # before a second window anywhere, so an edge that needs 14 windows before
    # it reconnects anything outranks an edge that two windows would finish.
    #
    #   Phase 1 -- edges that can be reconnected within this epoch's budget,
    #   CHEAPEST FIRST (fewest windows to reach an endpoint), each seeded with
    #   its own min_useful_bridges.  An edge whose seed no longer fits because
    #   earlier repairable edges already spent the budget falls through to
    #   phase 2 rather than being half-funded.
    #
    #   Cheapest-first, not worst-overlap-first, and that ordering is
    #   load-bearing: with a budget of 4, one overlap-0.02 edge needing 4
    #   windows and three overlap-0.10..0.20 edges that ONE midpoint fully
    #   repairs, worst-overlap-first spends the entire budget on the single
    #   worst edge and starves three complete repairs -- the same
    #   one-greedy-edge-eats-the-budget regression the flat round-robin was
    #   introduced to fix, just moved into phase 1.  Every window costs the
    #   same MD, so completing the most edges per epoch is what the budget
    #   buys; the expensive edge still gets its bisection step from phase 2.
    #   The sort is stable, so equally cheap edges keep the worst-overlap-first
    #   order the candidate list already carries.
    #   Phase 2 -- the historical flat round-robin, one window at a time, over
    #   everything still under its cap.  This is what keeps an under-budgeted
    #   run moving: an evenly spaced bridge halves the gap, so a lone window on
    #   an unreachable edge is a bisection step (8 -> 4 -> 2 -> 1 over epochs),
    #   not a wasted slot.  bridge_skip_unreachable=True excludes unfunded
    #   edges from this phase, which is exactly the stall that keeps it opt-in.
    #
    # With bridge_repairable_first=False (and the veto off) phase 1 is skipped
    # entirely and phase 2 alone reproduces the previous allocation exactly.
    _skip_unreachable = bool(policy.bridge_skip_unreachable)
    _seed_repairable = bool(policy.bridge_repairable_first) or _skip_unreachable
    _remaining_budget = max_new
    if _seed_repairable:
        for cand in sorted(candidates, key=lambda c: int(c["min_useful"])):
            if not cand["repairable_this_epoch"]:
                continue
            _seed = int(cand["min_useful"])
            if _seed <= _remaining_budget:
                cand["allocated"] = _seed
                cand["funded_repairable"] = True
                _remaining_budget -= _seed
            # An edge whose seed no longer fits gets nothing here and falls to
            # phase 2 with the rest.  Its ``repairable_this_epoch`` is left
            # TRUE: it really was repairable, the budget simply went to worse
            # edges first, and overwriting the classification would make the
            # plan record blame the geometry for a scheduling outcome.
            # ``funded_repairable`` is the one that says what happened.
    while _remaining_budget > 0:
        _progressed = False
        for cand in candidates:
            if _remaining_budget <= 0:
                break
            if _skip_unreachable and not cand["funded_repairable"]:
                continue
            if int(cand["allocated"]) >= int(cand["cap"]):
                continue
            cand["allocated"] = int(cand["allocated"]) + 1
            _remaining_budget -= 1
            _progressed = True
        if not _progressed:
            break

    for cand in candidates:
        edge = cand["edge"]
        s1, s2 = cand["s1"], cand["s2"]
        prediction = cand["prediction"]
        needed = int(cand["needed"])
        n_place = int(cand["allocated"])
        min_useful = int(cand["min_useful"])
        prediction_note = (
            f"predicted worst midpoint spacing/sigma "
            f"{float(prediction['worst_midpoint_spacing_sigma']):.2f} on "
            f"{prediction['worst_axis']} vs healthy <= "
            f"{float(prediction['healthy_spacing_sigma']):.2f}"
        )
        # The actionable half of the diagnosis: how many windows it would take
        # before this edge reconnects anything at all, and the flag that buys
        # them.  Without it an operator reads "under-bridged" every epoch with
        # no way to tell a one-window-short edge from a hopeless one.
        reach_note = (
            f"needs >= {min_useful} window(s) before the chain reaches either "
            f"endpoint; raise --ap-max-new-windows to >= {min_useful} (or lower "
            f"--ap-bridge-healthy-spacing-sigma) to repair it in one epoch"
        )
        _bisection_only = 0 < n_place < min_useful
        # Only an edge the geometry itself rules out gets the REFUSED wording.
        # A repairable edge that simply lost the budget race is NOT refused --
        # telling its operator to "raise --ap-max-new-windows to >= 3" on a run
        # already configured for 4 is a wrong instruction, and the
        # budget-contention branch below already says the right thing.
        _refused = (n_place <= 0 and _skip_unreachable
                    and not cand["funded_repairable"]
                    and not cand["repairable_this_epoch"])

        def _record(outcome: str, placed: int, note: str) -> None:
            """One plan row per weak edge considered, placed or not."""
            if bridge_plan_out is None:
                return
            bridge_plan_out.append({
                "state_i": int(s1.state_id),
                "state_j": int(s2.state_id),
                "overlap": edge.get("overlap"),
                "exchange_acceptance": edge.get("exchange_acceptance"),
                "bridges_needed": needed,
                "min_useful_bridges": min_useful,
                "bridges_allocated": n_place,
                "bridges_placed": int(placed),
                "repairable_this_epoch": bool(cand["repairable_this_epoch"]),
                "funded_repairable": bool(cand["funded_repairable"]),
                "worst_axis": prediction.get("worst_axis"),
                # None, not NaN: an unmeasurable geometry (no spring on either
                # axis) leaves this non-finite, and write_json ultimately calls
                # json.dumps with allow_nan left on, which emits a bare NaN
                # token that no strict JSON parser will read back.
                "worst_midpoint_spacing_sigma": _finite_or_none(
                    prediction.get("worst_midpoint_spacing_sigma")),
                "healthy_spacing_sigma": float(prediction["healthy_spacing_sigma"]),
                "single_bridge_reaches_either": bool(
                    prediction["single_bridge_reaches_either"]),
                "max_new_windows_per_epoch": int(max_new),
                "outcome": outcome,
                "note": note,
            })

        # shortfall is now derived from what was actually allocated, never from a
        # clamped-to-at-least-one count.  The previous version computed
        # n_place = max(1, min(needed, budget_left)), so an edge reached after
        # the budget was spent reported "under-bridged by needed - 1" while the
        # inner loop placed zero bridges -- a warning with the wrong number, for
        # an edge nothing was even attempted on.
        shortfall = max(0, needed - n_place)
        if n_place <= 0:
            if _refused:
                logging.warning(
                    "adaptive-production: weak edge %s-%s REFUSED, no bridge "
                    "windows placed: %s (%s), and bridge_skip_unreachable is on "
                    "so the budget (max_new_windows_per_epoch=%d) went to edges "
                    "that can be reconnected this epoch instead; this edge stays "
                    "weak and will resurface in the next epoch's diagnostics",
                    s1.state_id, s2.state_id, reach_note, prediction_note, max_new,
                )
            else:
                logging.warning(
                    "adaptive-production: weak edge %s-%s needs %d bridge window(s) "
                    "(%s) but no bridge windows could be placed this epoch -- the "
                    "per-epoch budget (max_new_windows_per_epoch=%d) was spent on "
                    "worse edges; this edge is left unbridged and will resurface in "
                    "the next epoch's diagnostics. It %s",
                    s1.state_id, s2.state_id, needed, prediction_note, max_new,
                    reach_note,
                )
            _record("refused_unreachable" if _refused else "skipped_no_budget",
                    0, reach_note)
            continue
        if _bisection_only:
            # Placed, but honest about what it is: the chain reaches neither
            # endpoint yet.  It still halves the gap, so the next epoch's
            # diagnostics see a strictly easier edge -- that bisection is the
            # whole reason this is not refused outright by default.
            logging.warning(
                "adaptive-production: weak edge %s-%s cannot be reconnected "
                "within this epoch's budget -- placing %d bisection step(s) "
                "only (%s). It %s",
                s1.state_id, s2.state_id, n_place, prediction_note, reach_note,
            )
        if shortfall:
            # Placing fewer bridges than the geometry calls for still leaves a
            # disconnected edge; say so loudly and by how much rather than
            # letting the next epoch's diagnostics rediscover it from scratch.
            logging.warning(
                "adaptive-production: weak edge %s-%s needs %d bridge window(s) "
                "(%s) but only %d could be placed this epoch "
                "(max_new_windows_per_epoch=%d, bridge_multi_window=%s); the edge "
                "will stay under-bridged by %d window(s)",
                s1.state_id, s2.state_id, needed, prediction_note, n_place,
                max_new, bool(policy.bridge_multi_window), shortfall,
            )
        # Evenly spaced interior points of the s1 -> s2 segment. n_place == 1
        # reproduces the historical plain midpoint exactly (see
        # _interpolate_bridge_axis for why that needs a special case).
        n_intervals = n_place + 1
        placed_this_edge = 0
        for j in range(1, n_intervals):
            if added >= max_new:
                break
            frac = float(j) / float(n_intervals)
            # Centres *and* springs are interpolated along s1 -> s2, so a bridge
            # sitting next to a stiff endpoint gets a comparably stiff restraint
            # instead of the whole chain sharing one endpoint-averaged spring.
            primary = _interpolate_bridge_axis(s1.primary_center, s2.primary_center, frac)
            primary_k = _interpolate_bridge_axis(s1.primary_k, s2.primary_k, frac)
            if s1.secondary_center is not None and s2.secondary_center is not None:
                secondary = _interpolate_bridge_axis(
                    s1.secondary_center, s2.secondary_center, frac)
                secondary_k = _interpolate_bridge_axis(
                    s1.secondary_k or 0.0, s2.secondary_k or 0.0, frac)
            else:
                secondary = None
                secondary_k = None
            # Parent drives seed selection (state_aware_seed_filtering), so a
            # bridge should inherit from whichever endpoint it actually sits
            # next to. frac == 0.5 resolves to s1, keeping the historical
            # single-midpoint parent unchanged.
            parent = s1 if frac <= 0.5 else s2
            if registry.has_near_duplicate(primary, secondary, policy):
                # A slot freed by a near-duplicate skip is deliberately *not*
                # handed back to another edge: the allocation above is what the
                # warnings above were emitted against, and silently topping up a
                # different edge afterwards would make those numbers wrong.
                continue
            # A midpoint average cannot exceed the cap once both endpoints are
            # under it, but a legacy registry (or a hand-edited window table)
            # can still hold an over-cap endpoint -- chignolin_6's state 25 was
            # 0.5 * (148.46 + 402.47) = 275.47 against cv2_k_max = 200.0 -- so
            # this path clamps its own output rather than trusting its inputs.
            secondary_k, _k_warning = _clamp_secondary_k(
                secondary_k,
                secondary_k_max,
                context=f"weak-edge bridge {s1.state_id}-{s2.state_id}",
            )
            params = (primary, primary_k, secondary, secondary_k)
            placed_this_edge += 1
            reason = (
                f"weak edge {s1.state_id}-{s2.state_id}: "
                f"overlap={edge.get('overlap')}, exchange_acceptance={edge.get('exchange_acceptance')}"
                f"; bridge {placed_this_edge}/{n_place} of {needed} needed ({prediction_note})"
            )
            if shortfall:
                reason = f"{reason}; UNDER-BRIDGED by {shortfall}"
            if _bisection_only:
                reason = f"{reason}; BISECTION-ONLY ({reach_note})"
            if _k_warning:
                logging.warning("adaptive-production: %s", _k_warning)
                reason = f"{reason}; {_k_warning}"
            actions.append(("add", int(parent.state_id), params, reason))
            added += 1
        # Outcome names the DECISION (what the allocation funded);
        # bridges_placed reports what survived near-duplicate filtering and the
        # global budget guard, so the two can be compared instead of one
        # silently standing in for the other.
        _record("bisection_step" if _bisection_only else "bridged",
                placed_this_edge,
                reach_note if _bisection_only else "reconnects at least one endpoint")

    # 2. Optionally retire clearly converged non-critical states.  Only reclaim
    # windows that are genuinely redundant (measured overlap >= redundant_overlap).
    if bool(policy.retire_converged):
        graph_critical = _graph_articulation_states(registry)
        bad_touching = set()
        state_max_overlap: Dict[int, float] = {}
        for edge in edge_rows:
            ov = edge.get("overlap")
            if ov is None or float(ov) < float(policy.target_overlap):
                bad_touching.add(int(edge["state_i"])); bad_touching.add(int(edge["state_j"]))
            acc = edge.get("exchange_acceptance")
            if acc is not None and float(acc) < float(policy.min_exchange_acceptance):
                bad_touching.add(int(edge["state_i"])); bad_touching.add(int(edge["state_j"]))
            if ov is not None:
                for key in ("state_i", "state_j"):
                    s = int(edge[key])
                    state_max_overlap[s] = max(state_max_overlap.get(s, 0.0), float(ov))
        # Non-geometry-neighbor collapses (see collect_epoch_diagnostics /
        # _non_neighbor_redundant_pairs) must also count toward redundancy —
        # otherwise a window that drifted onto a distant window's basin never
        # becomes retirement-eligible just because its actual geometry edges
        # (to its still-distinct real neighbors) look fine.
        #
        # _hist_overlap (and therefore the alert) is SYMMETRIC: both the
        # healthy anchor a collapsed window drifted onto and the collapsed
        # window itself get the same overlap value. Crediting it to both
        # sides indiscriminately can make the retirement path drop the
        # correct on-target anchor instead of (or alongside) the actual
        # interloper. Use the off-target signal computed above to credit the
        # overlap ONLY to whichever side actually failed to reach its
        # target; if neither or both sides are flagged off-target the pair
        # is ambiguous and is skipped rather than guessed at.
        def _is_off_target(sid: int) -> bool:
            warnings = state_rows.get(int(sid), {}).get("warnings") or []
            return "off_target_primary" in warnings or "off_target_secondary" in warnings

        for alert in diagnostics.get("non_neighbor_redundancies", []) or []:
            ov = alert.get("overlap")
            if ov is None:
                continue
            si, sj = int(alert["state_i"]), int(alert["state_j"])
            off_i, off_j = _is_off_target(si), _is_off_target(sj)
            if off_i == off_j:
                continue  # ambiguous (both or neither off-target) -- do not guess
            interloper = si if off_i else sj
            state_max_overlap[interloper] = max(state_max_overlap.get(interloper, 0.0), float(ov))
        retire_candidates: List[int] = []
        for state in registry.active_states():
            sid = int(state.state_id)
            diag = state_rows.get(sid, {})
            if sid in graph_critical or sid in bad_touching:
                continue
            # only reclaim genuinely redundant (over-overlapped) windows
            if float(state_max_overlap.get(sid, 0.0)) < float(policy.redundant_overlap):
                continue
            if int(diag.get("sample_count", 0) or 0) < int(policy.min_samples_for_retire):
                continue
            bsd = diag.get("gamd_boost_sd_kcal_mol")
            if bsd is not None and float(bsd) > float(policy.max_gamd_boost_sd_kcal_mol):
                continue
            retire_candidates.append(sid)
        # Single-node articulation exclusion is necessary but NOT sufficient: in a
        # cyclic (2D) geometry graph, co-retiring two individually-safe non-articulation
        # nodes can disconnect the graph. Admit retirements greedily (most-redundant
        # first), re-checking connectivity on the trial-reduced graph and REFUSING +
        # logging any drop that would disconnect. Restore active flags before returning
        # (retirement is applied later by _apply_registry_actions), so this function
        # leaves the registry unmutated.
        retire_candidates.sort(key=lambda s: float(state_max_overlap.get(s, 0.0)), reverse=True)
        # Hard minimum-replica floor: never retire the active set below
        # policy.min_active_states (default 8), independent of how many windows are
        # redundant. Snapshot the count before any tentative removal.
        min_active = max(0, int(getattr(policy, "min_active_states", 0)))
        n_active_start = len(registry.active_state_ids())
        admitted: List[int] = []
        for sid in retire_candidates:
            if n_active_start - len(admitted) - 1 < min_active:
                logger.info(
                    "adaptive-production: min_active_states floor (%d) reached; keeping the "
                    "remaining redundant window(s)", min_active
                )
                break
            st = registry.get_state(sid)
            if st is None:
                continue
            st.active = False
            if active_graph_connected(registry):
                admitted.append(sid)  # keep tentatively removed so the next check is cumulative
            else:
                st.active = True  # refuse this drop; it would disconnect the active graph
                logger.warning(
                    "adaptive-production: refused retirement of state %s — would disconnect "
                    "the active window graph", sid
                )
        for sid in admitted:
            registry.get_state(sid).active = True  # restore; apply step performs the real retire
            actions.append(("retire", sid, "redundant (over-overlapped) and non-critical"))

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
            "gamd_lambda": float(state.gamd_lambda),
        })
    fieldnames = [
        "epoch_window", "state_id", "primary_cv_center", "primary_cv_k_kcal",
        "distance_center_A", "distance_k_kcal_mol_A2", "secondary_cv_center",
        "secondary_cv_k_kcal_mol", "window_type", "patch_lifecycle", "parent_state_id",
        "created_epoch", "source", "reason", "usable_for_mbar", "burnin_steps",
        "gamd_lambda",
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


def _load_existing_epoch_schedule(round_dir: Path, *, prefix: str = "epoch_schedule") -> Optional[List[Dict[str, Any]]]:
    """Reload a schedule already written by ``write_epoch_schedule_files`` for this round, if any.

    On resume, a previously-interrupted epoch/final round must reuse its own
    already-written schedule verbatim rather than calling
    ``build_adaptive_epoch_schedule`` again: a fresh build scores against the
    *current* (dwindling) pool budget and diagnostics snapshot, which can change
    the set/ordering of top-up step sizes and therefore the ``topup_{idx:03d}_*``
    directory names run_scheduled_adaptive_epoch derives from them, even though
    those directories may already contain real, checkpointed sampling under the
    old names. Reusing the persisted rows keeps segment naming stable across
    restarts. Returns ``None`` when no schedule file exists yet for this round
    (a genuinely new round, which should still be built fresh).
    """
    path = Path(round_dir) / f"{prefix}.json"
    if not path.exists():
        return None
    payload = read_json_file(path, None)
    if not isinstance(payload, dict):
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return None
    return rows


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


def _final_extension_steps(policy: AdaptiveDecisionPolicy, fallback_steps: int) -> int:
    """Per-state steps for one optional frozen-final quality-extension round.

    ``fallback_steps`` must be the pool-*independent* per-state fallback
    (``--ap-final-steps`` or ``gamd_production_steps // 20``), never the
    pool-derived target :func:`_scheduled_final_default_steps` computes for the
    final schedule.  Extension rounds run *after* the final phase has already
    drawn its share of the pool, so seeding them from "the whole remaining
    budget" would make every round request essentially the entire pool for a
    top-up; with ``pool_hard_stop`` that is a hard failure rather than a
    truncation.  Kept as its own function purely so that contract is greppable
    and testable instead of implied by which local variable happens to be in
    scope 200 lines into the driver.
    """
    return max(1, int(policy.final_quality_extension_steps or max(1, int(fallback_steps))))


def _scheduled_final_default_steps(
    pool: Optional[AdaptiveRuntimePool],
    policy: AdaptiveDecisionPolicy,
    *,
    n_states: int,
    timestep_fs: float,
    fallback_steps: int,
    explicit_final_steps: int,
) -> int:
    """Per-state ``default_steps`` for the *scheduled* frozen final phase.

    The scheduled-epoch path recomputes its per-state step target from the live
    runtime pool every epoch (:func:`_epoch_steps_from_pool`).  The scheduled
    *final* path did not: it handed ``build_adaptive_epoch_schedule`` the
    pool-independent fallback ``gamd_production_steps // 20``, because
    :func:`_final_steps_from_pool` was only ever reached inside the
    non-scheduled final branch further down.  That asymmetry, not the
    allocator's own scoring, is why a final phase can come out microscopic.

    Measured on chignolin_6 (``adaptive_runtime_pool.json``): 27 active states
    entering the final phase with 7,530.5 ns still in a 10,000 ns pool, and a
    fallback of 500,000 steps at 4 fs -- so the whole final phase requested
    27 * 500,000 * 4 fs = 54 ns, 0.7% of what was available, and actually
    consumed ~50 ns.  Every state born in the last epoch is only ever sampled
    by the final phase, so states 24/25/26 came out with ~10k samples each
    against 457k-967k elsewhere, despite the schedule having correctly given
    them the three *largest* per-state allocations in the whole run.

    ``explicit_final_steps`` (``--ap-final-steps``) still wins outright, and a
    disabled/absent pool falls back to the previous value, so nothing changes
    for runs that either pinned the number or never enabled the pool.  The
    returned value is only a *target*: ``_policy_with_pool_step_budget`` clips
    the resulting total against the pool's real remaining balance (rescaling
    ``min_state_steps`` when it has to), so a larger target cannot overrun the
    budget -- it can only stop leaving most of it unspent.
    """
    fallback_steps = max(1, int(fallback_steps))
    if int(explicit_final_steps) > 0:
        return int(explicit_final_steps)
    if pool is None or not pool.enabled:
        return fallback_steps
    target = _final_steps_from_pool(pool, int(n_states), float(timestep_fs))
    return int(target) if target > 0 else fallback_steps


def _epoch0_scaled_steps(steps: int, epoch: int, fraction: float) -> int:
    """Scale one epoch's step budget by ``fraction`` when ``epoch == 0``, else pass through.

    Epoch 0 also bootstraps tICA and (optionally) recalibrates the shared GaMD
    envelope from real sampling, so it needs only a short look, not a full-length
    epoch. Steps left unused by a shorter epoch 0 stay in the runtime pool and
    are redistributed across the remaining epochs by the normal per-epoch
    pool-share recompute -- this is a plain multiply, no bookkeeping needed here.
    """
    steps = max(0, int(steps))
    if epoch != 0 or steps <= 0:
        return steps
    fraction = max(0.0, min(1.0, float(fraction)))
    if fraction == 1.0:
        return steps
    return max(1, int(round(steps * fraction)))


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
            if float(fresh.total_ns) > float(pool.total_ns):
                print(
                    f"    Adaptive-production runtime pool budget raised on resume/extend: "
                    f"{pool.total_ns:.6g} ns -> {fresh.total_ns:.6g} ns"
                )
                pool.warnings.append(
                    f"resume/extend raised total pool budget from {pool.total_ns:.6g} ns to {fresh.total_ns:.6g} ns"
                )
                pool.total_ns = float(fresh.total_ns)
            else:
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


def _segment_checkpoint_prod_done(seg_dir: Path) -> Optional[int]:
    """Return a segment's already-checkpointed production step count, or None if unavailable.

    Lazy-imports checkpoint_manifest_path from .production (not at module level)
    because production.py -> adaptive_production.py is an existing import chain;
    importing it back here at module scope would make it circular.
    """
    from .production import checkpoint_manifest_path
    manifest = read_json_file(checkpoint_manifest_path(seg_dir), None)
    if not isinstance(manifest, dict):
        return None
    try:
        return int(manifest.get("prod_done", 0))
    except (TypeError, ValueError):
        return None


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
    _seg_call_counter: List[int] = [0]
    # States whose baseline segment this round was skipped outright (pool
    # exhaustion zeroed a large baseline's clip_steps allowance while a much
    # smaller topup subset could still afford nonzero steps in the SAME
    # round) - a state in here has had NO real production this round despite
    # being active, so a topup segment covering it is that state's first-ever
    # real sampling this round, not a re-seed of an already-established
    # window. Consulted by seeding.py's topup-scoped auto-allow-bad-windows
    # fallback via _adaptive_phase_info, since that fallback's whole premise
    # depends on the window already being established.
    _baseline_skipped_state_ids: set = set()

    def run_segment(name: str, state_ids: Sequence[int], steps: int) -> Path:
        seg_dir = epoch_dir / name
        seg_dir.mkdir(parents=True, exist_ok=True)
        windows_csv = epoch_dir / f"{name}_windows.csv"
        # Decide resume BEFORE writing anything: this segment's own
        # epoch_window_map.csv may already have been corrected for a post-pull
        # window drop, and that correction is not recreated on resume (the drop
        # only runs on a fresh start). Writing a fresh identity map over it is
        # the chignolin_6 mis-attribution.  See
        # _phase_window_map_is_owned_by_a_resuming_phase.
        _seg_will_resume = bool(resume_requested and production_checkpoint_available(seg_dir))
        _seg_map_path = seg_dir / "epoch_window_map.csv"
        # Both no-clobber vetoes, not just the resume one: a segment can also
        # already hold samples from an attempt whose checkpoint is unusable, and
        # those rows were logged against the map on disk just the same.  Folded
        # into the single `_seg_map_preserved` local rather than OR-ed inline at
        # the `map_path=` keyword below, so the writer call keeps one guarded
        # conditional to read (and one for the structural test to recognise).
        _seg_map_preserved = bool(
            _phase_window_map_is_owned_by_a_resuming_phase(
                seg_dir, _seg_map_path,
                resume_requested=_seg_will_resume,
                expected_rows=len(state_ids),
            )
            or _phase_holds_samples_logged_against_its_window_map(seg_dir, _seg_map_path)
        )
        # The windows table itself is still written: it is inert on a fast resume
        # (load_resume_run_definition rebuilds the window set from this phase's
        # own recorded post-drop tables) and stays a faithful record of what the
        # driver asked for.
        write_state_subset_window_csv(
            registry, windows_csv, state_ids,
            map_path=None if _seg_map_preserved else _seg_map_path,
        )
        seg_args = copy.copy(args)
        seg_args.out = str(seg_dir)
        seg_args.gamd_production_steps = int(steps)
        seg_args.window_mode = "adaptive"
        seg_args.windows_2d_csv = str(windows_csv)
        seg_args.adaptive_feedback_enabled = False
        seg_args.adaptive_feedback_pilot = False
        seg_args.adaptive_feedback_final_production = False
        seg_args.resume = _seg_will_resume
        if seg_args.resume:
            print(f"      scheduled segment {name}: checkpoint manifest found; resuming from {seg_dir}")
        # Derive epoch index from directory name for TUI epoch/topup display.
        _epoch_dir_name = epoch_dir.name  # e.g. "epoch_001"
        try:
            _epoch_idx = int(_epoch_dir_name.rsplit("_", 1)[-1])
        except Exception:
            _epoch_idx = 0
        # Offset seed per segment so each segment gets fresh exchange-RNG and
        # thermostat streams.  Without this every segment replays identical
        # random sequences (N4: re-correlated RNG across segments).
        seg_args.seed = int(args.seed) + _epoch_idx * 1_000_003 + _seg_call_counter[0] * 997
        _seg_call_counter[0] += 1
        setattr(seg_args, "_adaptive_phase_info", {
            "is_adaptive_epoch": True,
            "epoch_index": _epoch_idx,
            "epoch_total": "?",
            "segment_name": name,
            "is_topup": name.startswith("topup"),
            "topup_index": int(name.split("_")[1]) if name.startswith("topup") and "_" in name[6:] else 0,
            "states_without_baseline_this_round": sorted(_baseline_skipped_state_ids.intersection(int(x) for x in state_ids)),
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
            if name == "baseline":
                _baseline_skipped_state_ids.update(int(x) for x in state_ids)
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
        # runtime_pool.consume() below must be charged the NEW steps this call
        # actually computes (delta since the last checkpoint), not the call's
        # full target - the checkpoint's own prod_done already reflects any
        # real progress from earlier invocations, including ones whose target
        # was smaller (an earlier restart, more pool remaining) or larger (a
        # later restart, less pool remaining, clip_steps returned less) than
        # this one.  Charging the full target every time double/triple/N-charges
        # the same real work on every restart that resumes an unfinished or
        # already-finished segment; for a segment with no next phase to advance
        # into (e.g. "final") that repeats indefinitely and silently drains the
        # campaign's MD-time budget on pure overhead - confirmed on chignolin_6,
        # where this drained ~8960 of 10000ns before the wrapper's pool-exhausted
        # check finally stopped the self-chain.
        _prior_prod_done = int(_segment_checkpoint_prod_done(seg_dir) or 0) if seg_args.resume else 0
        _delta_steps = actual_steps - _prior_prod_done
        if _delta_steps <= 0:
            # Already at or past this call's target - resuming would do zero
            # new MD steps.  Skip the run entirely (saves a no-op replica
            # reconstruction) rather than just charging zero, since there is
            # nothing left for run_gareus_callable to usefully do here.
            print(
                f"      scheduled segment {name}: already complete "
                f"({_prior_prod_done}/{actual_steps} steps checkpointed); skipping resume, no pool charge"
            )
            # run_gareus_callable is never invoked on this fast path, so the normal
            # end-of-run reseal (finalize_segment(), called from run_gareus's own
            # `finally` block) never runs either. The checkpoint read above proves
            # a prior invocation's production loop reached this same target, but it
            # cannot prove that process reached a *terminal* checkpoint write versus
            # being killed at some earlier interior checkpoint boundary -- actual_steps
            # can legitimately shrink on a later restart (pool budget dwindling), so
            # a checkpoint that looked "at target" here may sit at a non-terminal
            # checkpoint relative to what an earlier restart originally requested.
            # Sealing as "complete" would let phantom rows past that checkpoint slip
            # into MBAR unfiltered; seal as "interrupted" instead, matching every
            # other resume-time reseal path in this codebase, so downstream analysis
            # correctly truncates to the checkpoint boundary rather than trusting data
            # that was never actually written.
            _seg_registry_check = SegmentRegistry(seg_dir)
            _latest_seg_entry = _seg_registry_check.get_latest_segment()
            if _latest_seg_entry is not None and _latest_seg_entry.get("status") == "running":
                from .production import checkpoint_manifest_path
                _seg_manifest = read_json_file(checkpoint_manifest_path(seg_dir), {}) or {}
                _seg_end_step = int(_seg_manifest.get("absolute_step", _prior_prod_done) or _prior_prod_done)
                _seg_registry_check.seal_segment(_latest_seg_entry["segment_id"], absolute_end_step=_seg_end_step, status="interrupted")
                print(f"      scheduled segment {name}: resealed stale 'running' segment registry entry -> 'interrupted'")
            segment_summaries.append({
                "segment": name,
                "dir": str(seg_dir),
                "windows_csv": str(windows_csv),
                "state_ids": [int(x) for x in state_ids],
                "steps": int(actual_steps),
                "requested_steps": int(requested_steps),
                "already_complete": True,
                "seed_bank": seed_report or {},
            })
            return seg_dir
        print(f"      scheduled segment {name}: {len(state_ids)} state(s), {actual_steps} steps")
        run_gareus_callable(seg_args, seg_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress)
        if _graceful_shutdown.is_set():
            # Charge the pool for what actually ran before returning. Returning
            # straight out left an interrupted segment's MD unbooked, so the
            # ledger reported budget that had already been spent and a resumed
            # campaign would allocate against a figure that is too large.
            # Observed on RUNS/chignolin_6: an interrupted final burned ~376 ns
            # while the pool still claimed 8999 of 10000 ns remaining and carried
            # no `final` event at all.
            #
            # Prefer the checkpoint's own progress over the planned delta; fall
            # back to the planned delta when no checkpoint was written, which
            # over-charges rather than under-charges -- the safe direction for a
            # budget, since under-charging is the defect being fixed.
            _int_done = int(_segment_checkpoint_prod_done(seg_dir) or 0)
            _int_ran = max(0, _int_done - _prior_prod_done) if _int_done else int(_delta_steps)
            _int_event = None
            if runtime_pool is not None and _int_ran > 0:
                _int_event = runtime_pool.consume(
                    label=f"{epoch_dir.name}/{name}",
                    kind="scheduled_final" if "final" in str(epoch_dir) else "scheduled_epoch",
                    n_states=len(state_ids),
                    steps=int(_int_ran),
                    path=seg_dir,
                )
            segment_summaries.append({
                "segment": name,
                "dir": str(seg_dir),
                "windows_csv": str(windows_csv),
                "state_ids": [int(x) for x in state_ids],
                "steps": int(actual_steps),
                "requested_steps": int(requested_steps),
                "interrupted_after_checkpoint": True,
                "runtime_pool_event": _int_event or {},
                "seed_bank": seed_report or {},
            })
            return seg_dir
        if runtime_pool is not None:
            pool_event = runtime_pool.consume(
                label=f"{epoch_dir.name}/{name}",
                kind="scheduled_final" if "final" in str(epoch_dir) else "scheduled_epoch",
                n_states=len(state_ids),
                steps=int(_delta_steps),
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
            elif kind == "tica_coverage_add":
                _, parent, params, reason, metadata = action
                self.registry.add_state(
                    primary_center=params[0], primary_k=params[1],
                    secondary_center=params[2] if len(params) > 2 else None,
                    secondary_k=params[3] if len(params) > 3 else None,
                    parent_state_id=None if parent is None else int(parent),
                    epoch=int(epoch) + 1, source="tica_coverage", reason=str(reason),
                    metadata=dict(metadata),
                )
            elif kind == "split":
                # No producer emits "split" today (grep: only "add",
                # "tica_coverage_add", "retire" and "extend" are ever appended).
                # Whichever future producer does must run its children's
                # secondary_k through _clamp_secondary_k the way the other two
                # new-state paths do -- this applier has no access to the live
                # cv2_k_max, so the clamp cannot be enforced from here.
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
        target_overlap=_arg_float(args, "adaptive_production_target_overlap", _arg_float(args, "adaptive_feedback_target_overlap", 0.30)),
        min_exchange_acceptance=_arg_float(args, "adaptive_production_min_exchange", 0.08),
        min_samples_for_add=_arg_int(args, "adaptive_production_min_samples", 50),
        min_samples_for_retire=_arg_int(args, "adaptive_production_retire_min_samples", 200),
        max_new_windows_per_epoch=_arg_int(args, "adaptive_production_max_new_windows_per_epoch", 4),
        retire_converged=_arg_bool(args, "adaptive_production_retire_converged", True),
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
        bridge_healthy_spacing_sigma=_arg_float(
            args, "adaptive_production_bridge_healthy_spacing_sigma", 1.5),
        bridge_multi_window=_arg_bool(
            args, "adaptive_production_bridge_multi_window", True),
        bridge_repairable_first=_arg_bool(
            args, "adaptive_production_bridge_repairable_first", True),
        bridge_skip_unreachable=_arg_bool(
            args, "adaptive_production_bridge_skip_unreachable", False),
        context_reuse=_arg_bool(args, "adaptive_production_context_reuse", False),
        context_reuse_require=_arg_bool(args, "adaptive_production_context_reuse_require", False),
        context_reuse_mode=str(getattr(args, "adaptive_production_context_reuse_mode", "off") or "off"),
        redundant_overlap=_arg_float(args, "adaptive_production_redundant_overlap", 0.45),
        min_active_states=_arg_int(args, "adaptive_production_min_active_states", _arg_int(args, "min_total_windows", 0) or 8),
        max_target_deviation_sigma=_arg_float(args, "adaptive_production_max_target_deviation_sigma", 3.0),
        coverage_k_stiffen_cap=_arg_float(args, "adaptive_production_coverage_k_stiffen_cap", 10.0),
    )


def _epoch_loop_missing_convergence(epoch: int, max_epochs: int, gate_converged: bool,
                                     require_convergence_before_final: bool) -> bool:
    """True when the adaptive epoch loop must raise instead of proceeding to final.

    Convergence at any one epoch no longer breaks the loop early (see
    run_adaptive_production_auto_loop: it just continues to the next
    scheduled epoch), so the loop always runs its configured budget. This
    only fires on the very last epoch, and only when that epoch never
    converged and the policy demands convergence before final.
    """
    return bool(epoch + 1 >= max_epochs and not gate_converged and require_convergence_before_final)


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
    # run_gareus (production.py) locks each individual segment's own out_dir,
    # but two whole adaptive-production invocations against the SAME campaign
    # (a stale self-chain job that never actually exited, or a manual duplicate
    # resubmission) could otherwise race for a while before either one reaches
    # a locked segment directory - lock the campaign root itself too.
    acquire_run_lock(adaptive_dir)
    policy = policy_from_args(args)
    max_epochs = max(1, _arg_int(args, "adaptive_production_epochs", 3))
    # Epoch 0 also bootstraps the tICA model and (when enabled) the shared GaMD
    # envelope recalibration -- both need only a short look at real sampling, not
    # a full-length epoch. Scale its step budget down; the unused steps stay in
    # the pool and are redistributed across the remaining epochs automatically.
    epoch0_step_fraction = max(0.0, min(1.0, _arg_float(args, "adaptive_production_epoch0_step_fraction", 0.5)))
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
    use_epoch_samples_for_mbar = _arg_bool(args, "adaptive_production_use_epoch_samples_for_mbar", True)
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
    if resume_requested and bool(policy.propagate_seed_bank):
        _resumed_seed_bank = _discover_latest_seed_bank(adaptive_dir)
        if _resumed_seed_bank is not None:
            current_seed_bank = _resumed_seed_bank
            print(
                f"    resume: found existing adaptive seed bank {_resumed_seed_bank}; "
                "using it instead of the initial seed_conformers_dir"
            )
    runtime_pool, runtime_pool_resume_validation = _load_or_initialize_runtime_pool(adaptive_dir, args, policy, resume_requested=resume_requested)
    context_reuse_readiness = evaluate_context_reuse_readiness(args, adaptive_dir, registry=None)
    registry: Optional[WindowStateRegistry] = None
    current_windows_csv: Optional[Path] = Path(str(args.windows_2d_csv)) if getattr(args, "windows_2d_csv", None) else None
    epoch_summaries: List[Dict[str, Any]] = []
    start_epoch = 0
    # Extension rounds already recorded as done by a *previous* invocation (e.g. an
    # earlier --extend call), read once here -- before this run's own writes to
    # summary_path start overwriting it -- exactly like epoch_summaries/start_epoch
    # above. Used to make frozen-final extension-round numbering resume-safe instead
    # of always restarting at final_extension_001.
    prior_extension_summaries: List[Dict[str, Any]] = []
    previous_diagnostics: Optional[Dict[str, Any]] = None

    summary_path = adaptive_dir / "adaptive_production_driver_summary.json"
    registry_path = adaptive_dir / "state_registry.json"
    if resume_requested and registry_path.exists():
        registry = WindowStateRegistry.load(adaptive_dir)
        old_summary = read_json_file(summary_path, {}) if summary_path.exists() else {}
        epoch_summaries = list(old_summary.get("epoch_summaries", []) or []) if isinstance(old_summary, dict) else []
        start_epoch = int(old_summary.get("epochs_completed", len(epoch_summaries)) or len(epoch_summaries)) if isinstance(old_summary, dict) else len(epoch_summaries)
        prior_extension_summaries = list(old_summary.get("final_extension_summaries", []) or []) if isinstance(old_summary, dict) else []
        current_windows_csv = adaptive_dir / f"windows_epoch_{start_epoch:03d}.csv"
        if not current_windows_csv.exists():
            current_windows_csv = adaptive_dir / "resume_active_windows.csv"
            registry.write_active_window_csv(current_windows_csv, map_path=adaptive_dir / f"window_map_epoch_{start_epoch:03d}.csv")
        print(f"    Adaptive-production resume: loaded {len(registry.all_states())} states; continuing at epoch {start_epoch}.")
        # Restore tICA state from the most recent successful refit so the resumed
        # run inherits the correct model weights and MBAR cross-epoch guard version.
        _latest_tica = None
        for _es in reversed(epoch_summaries):
            _tu = _es.get("tica_update") or {}
            if _tu.get("status") == "updated" and _tu.get("state_file") and _tu.get("version"):
                _latest_tica = _tu
                break
        if _latest_tica is not None:
            _sf = str(_latest_tica["state_file"])
            _ver = str(_latest_tica["version"])
            if Path(_sf).exists():
                args.tica_state_file = _sf
                args.tica_cv_version = _ver
                print(f"    tICA resume: restored model {_sf} (version {_ver})")
            else:
                print(f"    tICA resume: WARNING: prior tICA state file {_sf} not found; starting fresh")

        # Restore CV2 switch if it fired during a completed epoch.
        for _es in reversed(epoch_summaries):
            _sw = (_es.get("tica_update") or {}).get("cv2_switched")
            if _sw and _sw.get("to") == "tica-linear":
                if str(getattr(args, "secondary_cv", "none") or "none") != "tica-linear":
                    args.secondary_cv = "tica-linear"
                    args.cv2_k_min = float(getattr(args, "tica_linear_k_min", 5.0) or 5.0)
                    args.cv2_k_max = float(getattr(args, "tica_linear_k_max", 50.0) or 50.0)
                    print(
                        f"    tICA resume: restored CV2 switch '{_sw.get('from')}' → 'tica-linear' "
                        f"(k=[{args.cv2_k_min:.1f}, {args.cv2_k_max:.1f}] kcal/mol) "
                        f"from epoch {_es['epoch']} summary"
                    )
                break

        # Restore previous_diagnostics from the most recently completed epoch's
        # diagnostics_json. Without this, the first genuinely new epoch/final
        # round built after a restart (i.e. one with no existing schedule file
        # for _load_existing_epoch_schedule to reuse) would score every state
        # against previous_diagnostics=None, which build_adaptive_epoch_schedule
        # treats as "no_samples_yet" for every state -- discarding real sampling
        # history the campaign already has.
        if epoch_summaries:
            _last_diag_path = epoch_summaries[-1].get("diagnostics_json")
            _restored_diag = read_json_file(Path(_last_diag_path), None) if _last_diag_path else None
            if isinstance(_restored_diag, dict):
                previous_diagnostics = _restored_diag
                print(f"    Adaptive-production resume: restored diagnostics from {_last_diag_path}")
            else:
                print(
                    f"    Adaptive-production resume: WARNING: prior diagnostics file {_last_diag_path} "
                    "not found or unreadable; next new schedule will score states as no_samples_yet"
                )
        context_reuse_readiness = evaluate_context_reuse_readiness(args, adaptive_dir, registry=registry)

    # Warn when tica_epochs_per_cycle is set but observation collection is disabled:
    # the refit will always be skipped (no obs → no data), making the cycle a no-op.
    _tica_epc = int(getattr(args, "tica_epochs_per_cycle", 0) or 0)
    _tica_obs = int(getattr(args, "tica_obs_interval", 0) or 0)
    if _tica_epc > 0 and _tica_obs <= 0:
        print(
            f"WARNING: tica_epochs_per_cycle={_tica_epc} but tica_obs_interval=0 — "
            "tICA refitting requires dihedral observations (set tica_obs_interval > 0). "
            "All cycle refits will be silently skipped."
        )
    if getattr(args, "tica_switch_cv2", False) and _tica_obs <= 0:
        print(
            "WARNING: tica_switch_cv2=True but tica_obs_interval=0 — "
            "CV2 auto-switch requires dihedral observations. Switch will never trigger."
        )

    # topup-only mode: skip the epoch loop, let the final-phase skip guard fire, run extension rounds.
    _topup_only = _arg_bool(args, "adaptive_production_topup_only", False)
    if _topup_only:
        max_epochs = start_epoch  # Skip epoch loop entirely.

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
            epoch_default_steps = _epoch0_scaled_steps(epoch_steps, epoch, epoch0_step_fraction)
            schedule_policy = _policy_with_pool_step_budget(
                registry, policy, runtime_pool, default_steps=epoch_default_steps, final=False
            )
            schedule = _load_existing_epoch_schedule(epoch_dir) if resume_requested else None
            if schedule is not None:
                print(
                    f"    Adaptive-production scheduled epoch {epoch + 1}/{max_epochs}: "
                    f"reusing existing schedule from {epoch_dir / 'epoch_schedule.json'} "
                    "(resume-stable segment naming)"
                )
            else:
                schedule = build_adaptive_epoch_schedule(
                    registry,
                    previous_diagnostics,
                    schedule_policy,
                    epoch=epoch,
                    default_steps=epoch_default_steps,
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
                    "epochs_completed": int(len(epoch_summaries)),
                    "epoch_summaries": _json_ready(epoch_summaries),
                    "scheduled_epoch": _json_ready(scheduled_summary),
                }
                write_json(summary_path, payload)
                return payload
            # Guard: only fires on a real completed scheduled epoch (shutdown path
            # already returned above).  Uses epoch_default_steps (the default budget
            # used to build the schedule) as the steps sentinel — sufficient because
            # the guard only needs steps > 0 to confirm the epoch was non-trivial.
            _assert_epoch_has_samples(diagnostics, registry, epoch_dir, int(epoch_default_steps))
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
            actual_epoch_steps = _epoch0_scaled_steps(actual_epoch_steps, epoch, epoch0_step_fraction)
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
            # Refresh THIS epoch's own epoch_window_map.csv before it runs.
            # A flat (non-scheduled) epoch >= 1 has its map written by the
            # *previous* iteration's next_epoch_dir site, i.e. a whole epoch
            # earlier, and nothing touched it since.  That leaves one combination
            # unrepairable by either write-side correction: a first attempt at
            # this epoch that got far enough to pull, drop windows and have
            # production compact this map (rewrite_epoch_window_map_after_drop),
            # and then died before a usable production checkpoint existed.  The
            # campaign re-enters the epoch with a *compacted* map on disk (say 24
            # rows) against a fresh un-pruned window table
            # (windows_epoch_NNN.csv, 27 rows) that run_gareus will re-pull from
            # scratch, possibly dropping a different set.  The drop-time rewrite
            # then refuses with skipped_row_count_mismatch (fewer map rows than
            # pre-drop windows) and repair_epoch_window_map_from_surviving_windows
            # refuses with skipped_map_shorter_than_window_set -- both correctly,
            # since neither can tell "the previous attempt's compaction, now
            # stale" from "the right map for a different drop set".  Writing the
            # pre-drop map here removes that ambiguity instead of asking a later
            # reader to resolve it.
            #
            # This is also the precondition
            # production._epoch_window_map_rewrite_already_applied's own
            # docstring already assumes ("a phase interrupted and re-run WITHOUT
            # a usable production checkpoint gets a fresh identity map written by
            # the adaptive driver and then legitimately re-runs the same filter
            # with the same drop set"): with the fresh map on disk the ledger
            # sees a map that no longer matches its recorded surviving state_ids
            # and lets that drop set be applied again, instead of calling it a
            # replay and leaving a stale map.  The ledger itself is append-only
            # and untouched by this write.
            #
            # Both no-clobber vetoes inside _write_phase_window_map still
            # apply, and between them they narrow this refresh to the cases where
            # nothing is lost by it: a phase that really is resuming (usable
            # production checkpoint behind that map) keeps its corrected map, and
            # so does one that already holds analysis-visible samples logged
            # against it -- an attempt that crashed mid-production seals its
            # segment "interrupted" with a real end_step from run_gareus's
            # `finally`, and those rows stay readable whether or not a usable
            # checkpoint survived.  What is left for this refresh is the case it
            # was actually written for: attempt 1 got as far as the pull/drop and
            # left either no samples at all or only a segment the analysis layer
            # discards.  That ordering is not incidental -- run_gareus's
            # pull/drop block is a top-level statement strictly before its
            # SegmentRegistry.open_segment and its first ParquetSampleWriter, so
            # a compacted map can exist with no samples behind it, but never the
            # reverse within one attempt.
            #
            # Writing from the live registry rather than from
            # current_windows_csv is correct only because nothing mutates the
            # registry between the previous iteration's paired
            # write_active_window_csv(next_csv) + next-epoch-map write (including
            # the tICA block's re-write of that same pair) and this point; keep
            # them paired if the epoch tail is ever reordered.
            if registry is not None:
                _write_phase_window_map(
                    registry, epoch_dir / "epoch_window_map.csv",
                    resume_requested=resume_requested,
                    expected_rows=len(registry.active_states()),
                )
            _write_tica_version_marker(epoch_dir, epoch_args)
            run_gareus(epoch_args, epoch_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress)
            if _graceful_shutdown.is_set():
                _write_runtime_pool_reports(adaptive_dir, runtime_pool)
                payload = {
                    "schema_version": "adaptive_production_driver_summary_v1",
                    "status": "interrupted_after_checkpoint",
                    "interrupted_segment": str(epoch_dir),
                    "epochs_completed": int(len(epoch_summaries)),
                    "epoch_summaries": _json_ready(epoch_summaries),
                }
                write_json(summary_path, payload)
                return payload

            if registry is None:
                table = find_best_window_table(epoch_dir)
                if table is None:
                    raise RuntimeError(f"epoch {epoch} completed but no umbrella window table was found in {epoch_dir}")
                registry = registry_from_window_csv(table, epoch=0, source="epoch0_windows")
                # Normally there is no map here yet (this bootstrap writes the
                # phase's first one, from the post-drop table, so identity is
                # correct); the guard only bites if a previous invocation already
                # left a map behind a real checkpoint.
                _write_phase_window_map(
                    registry, epoch_dir / "epoch_window_map.csv",
                    resume_requested=resume_requested,
                    expected_rows=len(registry.active_states()),
                )
            runtime_pool.consume(
                label=f"epoch_{epoch:03d}",
                kind="adaptive_epoch",
                n_states=max(1, len(registry.active_states())),
                steps=int(actual_epoch_steps),
                path=epoch_dir,
            )
            diagnostics = collect_segmented_epoch_diagnostics(epoch_dir, registry, policy)
            _assert_epoch_has_samples(diagnostics, registry, epoch_dir, int(actual_epoch_steps))
        bridge_plan: List[Dict[str, Any]] = []
        actions = propose_actions_from_diagnostics(
            registry, diagnostics, policy=policy,
            temperature_K=_args_temperature_k(args),
            # Read live, not from a loop-start snapshot: _apply_tica_cv2_switch
            # rewrites args.cv2_k_max mid-campaign. See _resolve_secondary_k_max.
            secondary_k_max=_resolve_secondary_k_max(args),
            bridge_plan_out=bridge_plan,
        )
        action_report = None
        if _arg_bool(args, "adaptive_production_write_action_reports", True):
            try:
                action_report = write_epoch_action_report(
                    epoch_dir, epoch, registry, diagnostics, actions, policy,
                    bridge_plan=bridge_plan)
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
        _write_phase_window_map(
            registry, next_epoch_dir / "epoch_window_map.csv",
            resume_requested=resume_requested,
            expected_rows=len(registry.active_states()),
        )

        # tICA CVaux inter-epoch update (opt-in; no-op by default)
        tica_update_report = {}
        try:
            tica_update_report = _maybe_update_tica_cvaux(epoch, epoch_dir, adaptive_dir, args, registry=registry)
            if tica_update_report.get("status") == "updated" and registry is not None:
                per_window_centers = tica_update_report.get("per_window_tic1_centers", {})
                if per_window_centers:
                    n_updated = _apply_tica_centers_to_registry(registry, per_window_centers, epoch_dir)
                    tica_update_report["registry_states_updated"] = n_updated
                    print(f"    tICA: updated secondary_center for {n_updated} active registry states")
                    if n_updated > 0:
                        try:
                            from .tica import TICAResult, load_epoch_dihedral_obs, project_tica1
                            # NOTE: the 5th return value is per-*file* segment_lengths, not
                            # per-frame steps (a pre-existing naming mismatch here, now fixed
                            # by using the real per-frame steps/replica_ids arrays below).
                            _X, _wids, _primary, _secondary, _, _steps, _replica_ids = load_epoch_dihedral_obs(epoch_dir)
                            _result = TICAResult.load(tica_update_report["state_file"])
                            _coverage_actions = _propose_tica_coverage_actions(
                                registry, _primary, project_tica1(_X, _result), policy,
                                temperature_K=_args_temperature_k(args),
                                umbrella_state_ids=_wids, replica_ids=_replica_ids, steps=_steps,
                                epoch_dir=epoch_dir,
                                # NOTE: _apply_tica_cv2_switch runs *after* this
                                # block within the same epoch, so the ceiling
                                # read here is still the outgoing CV2 regime's
                                # cv2_k_max even though these new centers are
                                # already tIC1 projections. Identical whenever
                                # tica_linear_k_max == cv2_k_max (true for
                                # chignolin_6: both 200.0); a run that sets them
                                # differently clamps coverage states against the
                                # outgoing ceiling for one epoch. Reordering the
                                # switch ahead of coverage proposal is a
                                # separate change, not made here.
                                secondary_k_max=_resolve_secondary_k_max(args),
                            )
                            _state_ids_before = {s.state_id for s in registry.active_states()}
                            if _coverage_actions:
                                _apply_registry_actions(registry, _coverage_actions, epoch)
                            tica_update_report["tica_coverage_actions"] = [
                                {"action": action[0], "parent_state_id": action[1],
                                 "params": list(action[2]), "reason": action[3],
                                 "metadata": action[4]}
                                for action in _coverage_actions
                            ]
                            print(f"    tICA: added {len(_coverage_actions)} populated-region coverage state(s)")
                            if _coverage_actions and current_seed_bank is not None:
                                _new_state_ids = sorted(
                                    s.state_id for s in registry.active_states()
                                    if s.state_id not in _state_ids_before
                                )
                                _extraction_report = extract_and_register_coverage_seeds(
                                    _coverage_actions, _new_state_ids, Path(current_seed_bank),
                                    topology, source_label=f"epoch_{epoch:03d}_tica_coverage",
                                )
                                tica_update_report["coverage_seed_extraction"] = _extraction_report
                                if _extraction_report["n_extracted"] or _extraction_report["n_skipped"]:
                                    print(
                                        f"    tICA coverage: extracted {_extraction_report['n_extracted']} real "
                                        f"starting structure(s) from trajectory, {_extraction_report['n_skipped']} "
                                        "not extractable (fell back to normal seed-library matching for those)"
                                    )
                        except Exception as _coverage_exc:
                            tica_update_report["tica_coverage_error"] = str(_coverage_exc)
                            print(f"WARNING: tICA coverage extension skipped ({_coverage_exc})")

                        # Re-measure every seed-bank candidate's secondary CV under
                        # the model that just moved every target center, then
                        # re-run the nearest-seed assignment against those real
                        # values instead of the write-time placeholder (which was
                        # always each state's own pre-update center, making every
                        # state trivially "match" its own seed regardless of how
                        # far that seed actually now sits from its new target).
                        # Must run after the coverage-action block above so any
                        # newly added state is already in the registry and gets
                        # a real assignment too, not left with none at all.
                        if current_seed_bank is not None:
                            try:
                                _rescore_report = rescore_seed_bank_secondary_cv(
                                    Path(current_seed_bank), _result, app, unit,
                                )
                                tica_update_report["seed_bank_rescore"] = _rescore_report
                                if _rescore_report.get("status") == "ok":
                                    select_state_aware_seeds_for_targets(Path(current_seed_bank), registry)
                                    print(
                                        f"    tICA: re-scored {_rescore_report.get('n_rescored', 0)} seed-bank "
                                        f"candidate(s) and re-ran state-aware seed assignment against updated centers"
                                    )
                                else:
                                    print(f"WARNING: seed-bank rescore skipped ({_rescore_report.get('status')})")
                            except Exception as _rescore_exc:
                                tica_update_report["seed_bank_rescore_error"] = str(_rescore_exc)
                                print(f"WARNING: seed-bank rescore failed ({_rescore_exc})")

                        registry.save(adaptive_dir)
                        # Re-write window CSV so the next epoch sees updated centers.
                        # (Initial write at lines above precedes this block; without
                        # this re-write the center update would be silently lost.)
                        registry.write_active_window_csv(next_csv, map_path=next_map)
                        _write_phase_window_map(
                            registry, next_epoch_dir / "epoch_window_map.csv",
                            resume_requested=resume_requested,
                            expected_rows=len(registry.active_states()),
                        )
                        print(f"    tICA: re-wrote {next_csv.name} with updated secondary centers")

            # CV2 auto-switch: permanently replace secondary_cv with tica-linear
            # only after a successful tICA fit wrote a state file.
            if _apply_tica_cv2_switch(args, tica_update_report, next_epoch=epoch + 1):
                _sw = tica_update_report.get("cv2_switched", {})
                print(
                    f"    tICA: switched CV2 '{_sw.get('from')}' → 'tica-linear' "
                    f"(k=[{float(getattr(args, 'cv2_k_min', 0.0)):.1f}, {float(getattr(args, 'cv2_k_max', 0.0)):.1f}] kcal/mol). "
                    f"Effective from epoch {epoch + 1}."
                )
            elif isinstance(tica_update_report, dict) and tica_update_report.get("cv2_switch_skipped"):
                _skip = tica_update_report.get("cv2_switch_skipped", {})
                print(f"WARNING: tICA CV2 switch skipped: {_skip.get('reason', 'unknown')}")
        except Exception as _tica_exc:
            print(f"WARNING: _maybe_update_tica_cvaux failed for epoch {epoch}: {_tica_exc}")
            tica_update_report = {"status": "error", "error": str(_tica_exc)}

        # GaMD shared-envelope recalibration from epoch 0's real sampling (opt-in;
        # fires at most once, no-op for epoch >= 1 and when GaMD/shared-envelope
        # reuse is disabled).
        gamd_recal_report = {}
        try:
            gamd_recal_report = _maybe_recalibrate_gamd_boost(epoch, epoch_dir, args, global_shared_gamd_dir)
        except Exception as _gamd_recal_exc:
            print(f"WARNING: _maybe_recalibrate_gamd_boost failed for epoch {epoch}: {_gamd_recal_exc}")
            gamd_recal_report = {"status": "error", "error": str(_gamd_recal_exc)}

        # Surface segments that the runtime pool forced to zero steps this epoch
        # (recorded per-segment as skipped_by_runtime_pool in run_scheduled_adaptive_epoch)
        # so silent baseline-only sampling is visible at the epoch level instead of
        # only buried inside the segmented scheduled_epoch payload.
        _segments_skipped_by_pool = [
            {"segment": s.get("segment"), "state_ids": s.get("state_ids", [])}
            for s in (scheduled_summary or {}).get("segments", [])
            if s.get("skipped_by_runtime_pool")
        ]
        if _segments_skipped_by_pool:
            print(
                f"WARNING: adaptive-production MD pool exhaustion skipped "
                f"{len(_segments_skipped_by_pool)} scheduled segment(s) in epoch {epoch}, reducing "
                f"this epoch to baseline-only sampling for the affected states: "
                + ", ".join(f"{s['segment']} (states {s['state_ids']})" for s in _segments_skipped_by_pool)
            )

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
            "segments_skipped_by_pool": _segments_skipped_by_pool,
            "next_windows_csv": str(next_csv),
            "registry": registry_paths,
            "runtime_pool": runtime_pool.to_dict(),
            "runtime_pool_reports": runtime_pool_paths,
            "tica_update": tica_update_report,
            "gamd_recalibration": gamd_recal_report,
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

        gate_converged = bool(convergence_gate.get("stop_adaptive", False))
        if gate_converged:
            # Convergence at one epoch does not mean the discovery loop is
            # done: it only means this epoch proposed no further actions.
            # Keep running the configured adaptive_production_epochs budget
            # and only fall through to the final phase once that budget is
            # exhausted -- jumping straight to final here just meant the
            # final-phase quality gate would immediately demand another
            # --extend round anyway (see epoch_000 of chignolin_sigma3_2d).
            print(
                f"    Adaptive-production convergence gate passed after epoch {epoch + 1}: "
                f"{convergence_gate.get('status')} (continuing to the next scheduled epoch)"
            )

        if _epoch_loop_missing_convergence(epoch, max_epochs, gate_converged, bool(policy.require_convergence_before_final)):
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

    # If extending a completed run, skip final phase if it already has sample data.
    # _run_dir_has_samples(adaptive_dir / "final") alone would never be True for
    # the default scheduled_final_segments=True case: a segmented final phase's
    # samples live in final/baseline/, final/topup_*/, not directly in final/
    # itself, so use the same flat-or-segmented lookup _sample_sources_from_run_root
    # already uses elsewhere for this exact directory (see collect_final_combined_
    # diagnostics and its union-MBAR-inputs equivalent) - otherwise this check can
    # never fire, and every self-chain restart re-enters and re-runs the final
    # phase's already-complete segments indefinitely.
    _final_already_done = (
        resume_requested
        and _is_adaptive_production_completed(adaptive_dir)
        and bool(_sample_sources_from_run_root("final", adaptive_dir / "final"))
    )
    if _final_already_done:
        print("[extend] Final phase already complete; skipping to extension rounds.")
        # Restore the registry + paths the caller below expects to exist.
        final_dir = adaptive_dir / "final"
        final_windows_csv = adaptive_dir / "final_active_windows.csv"
        final_registry_csv = adaptive_dir / "final_registry_used_for_mbar.csv"
        final_schedule_files = {}
        final_seed_bank = None
        final_scheduled_summary = None
        # Jump to extension rounds. (Extension rounds block follows below.)
    if not _final_already_done:
        final_dir = adaptive_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_windows_csv = adaptive_dir / "final_active_windows.csv"
        final_registry_csv = adaptive_dir / "final_registry_used_for_mbar.csv"
        _final_map_path = final_dir / "epoch_window_map.csv"
        _final_map_preserved = bool(
            _phase_window_map_is_owned_by_a_resuming_phase(
                final_dir, _final_map_path,
                resume_requested=resume_requested,
                expected_rows=len(registry.active_states()),
            )
            # Same second veto as run_segment's: the final phase can equally have
            # died leaving real samples behind an unusable checkpoint.
            or _phase_holds_samples_logged_against_its_window_map(final_dir, _final_map_path)
        )
        registry.write_active_window_csv(
            final_windows_csv,
            map_path=None if _final_map_preserved else _final_map_path,
        )
        registry.write_state_csv(final_registry_csv)
        final_schedule_files = {}
        final_schedule = None
        if bool(policy.final_allocation_scheduler):
            # Recompute the per-state target from the live pool before building
            # the schedule -- the scheduled-epoch path above already does this
            # via _epoch_steps_from_pool, but the scheduled *final* path used to
            # go straight in with the pool-independent gamd_production_steps//20
            # fallback and so spent a fraction of a percent of the remaining
            # budget. See _scheduled_final_default_steps for the chignolin_6
            # numbers.
            #
            # Deliberately a *local*: `final_steps` itself must NOT be
            # reassigned. It is read again after the final phase by the
            # quality-extension loop (`ext_steps = policy.
            # final_quality_extension_steps or final_steps`), and that loop runs
            # once the pool is already nearly spent. Widening `final_steps` there
            # would make every extension round request the whole remaining
            # budget for a top-up. The non-scheduled final branch below does its
            # own `_final_steps_from_pool` recompute, so it is unaffected either.
            final_schedule_steps = _scheduled_final_default_steps(
                runtime_pool,
                policy,
                n_states=max(1, len(registry.active_states())),
                timestep_fs=_timestep_fs,
                fallback_steps=final_steps,
                explicit_final_steps=_explicit_final_steps,
            )
            if final_schedule_steps != int(final_steps):
                print(
                    f"    Adaptive-production final scheduled phase: per-state step target "
                    f"from MD pool {final_steps}->{final_schedule_steps} "
                    f"({runtime_pool.remaining_ns():.1f} ns remaining, "
                    f"{max(1, len(registry.active_states()))} active states)"
                )
            final_schedule_policy = _policy_with_pool_step_budget(
                registry, policy, runtime_pool, default_steps=final_schedule_steps, final=True
            )
            final_schedule = _load_existing_epoch_schedule(final_dir, prefix="final_state_schedule") if resume_requested else None
            if final_schedule is not None:
                print(
                    f"    Adaptive-production final phase: reusing existing schedule from "
                    f"{final_dir / 'final_state_schedule.json'} (resume-stable segment naming)"
                )
                final_schedule_files = {
                    "csv": str(final_dir / "final_state_schedule.csv"),
                    "json": str(final_dir / "final_state_schedule.json"),
                    "md": str(final_dir / "final_state_schedule.md"),
                }
            else:
                final_schedule = build_adaptive_epoch_schedule(
                    registry,
                    previous_diagnostics,
                    final_schedule_policy,
                    epoch=max_epochs,
                    default_steps=final_schedule_steps,
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
            # The scheduled final was the only one of the three phase-running
            # sites that ignored this. The epoch loop and the non-scheduled final
            # both return here; without it a campaign cut short mid-final fell
            # through the quality gate and the union-MBAR build to the
            # unconditional "completed" at the end of this function. On
            # RUNS/chignolin_6 that reported a completed campaign for a final
            # phase that had run 17% of its schedule and left one state with zero
            # samples, while the quality gate underneath said needs_more_sampling.
            if _graceful_shutdown.is_set():
                _write_runtime_pool_reports(adaptive_dir, runtime_pool)
                payload = {
                    "schema_version": "adaptive_production_driver_summary_v1",
                    "status": "interrupted_after_checkpoint",
                    "interrupted_segment": str(final_dir),
                    "epochs_completed": int(len(epoch_summaries)),
                    "epoch_summaries": _json_ready(epoch_summaries),
                    "scheduled_final": _json_ready(final_scheduled_summary),
                }
                write_json(summary_path, payload)
                return payload
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
            _write_tica_version_marker(final_dir, final_args)
            run_gareus(final_args, final_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress)
            if _graceful_shutdown.is_set():
                _write_runtime_pool_reports(adaptive_dir, runtime_pool)
                payload = {
                    "schema_version": "adaptive_production_driver_summary_v1",
                    "status": "interrupted_after_checkpoint",
                    "interrupted_segment": str(final_dir),
                    "epochs_completed": int(len(epoch_summaries)),
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
    # Seed from any extension rounds a *previous* invocation already recorded
    # (see prior_extension_summaries above) so the round index/directory naming
    # below continues where that invocation left off instead of restarting at
    # final_extension_001 and colliding with already-completed round directories.
    extension_summaries: List[Dict[str, Any]] = list(prior_extension_summaries)
    start_ext_round = len(prior_extension_summaries)
    final_diag = collect_final_combined_diagnostics(adaptive_dir, registry, policy=policy)
    quality_gate = evaluate_adaptive_quality_gate(
        adaptive_dir, registry, final_diag, policy=policy, output_prefix="adaptive_quality_gate_pre_union"
    )
    ext_rounds = max(0, int(policy.final_quality_extension_rounds or 0))
    ext_steps = _final_extension_steps(policy, final_steps)
    ext_round_total = start_ext_round + ext_rounds
    for ext_index in range(start_ext_round, ext_round_total):
        if not quality_gate_fixable_by_more_final_sampling(quality_gate):
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
            print(f"    Adaptive-production frozen final extension {ext_index + 1}/{ext_round_total}: checkpoint manifest found; resuming from {ext_dir}")
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
        _write_phase_window_map(
            registry, ext_dir / "epoch_window_map.csv",
            resume_requested=resume_requested,
            expected_rows=len(registry.active_states()),
        )
        # Delta-charge the runtime pool for extension rounds the same way
        # run_segment() does for scheduled topup segments (see
        # _segment_checkpoint_prod_done docstring / its use above): a resumed
        # round's checkpoint may already carry real progress from an earlier
        # invocation (an earlier restart, or an earlier --extend call reusing
        # this exact round directory), so charging actual_ext_steps in full on
        # every resume would double/triple-charge the same MD steps against the
        # campaign pool. Skip run_gareus entirely (and charge nothing) once the
        # checkpoint has already reached this round's target.
        _prior_ext_prod_done = int(_segment_checkpoint_prod_done(ext_dir) or 0) if ext_args.resume else 0
        if _prior_ext_prod_done >= actual_ext_steps:
            print(
                f"    Adaptive-production frozen final extension {ext_index + 1}/{ext_round_total}: "
                f"already complete ({_prior_ext_prod_done}/{actual_ext_steps} steps checkpointed); "
                "skipping run, no pool charge"
            )
        else:
            print(
                f"    Adaptive-production frozen final extension {ext_index + 1}/{ext_round_total}: "
                f"{actual_ext_steps} steps -> {ext_dir}"
            )
            _write_tica_version_marker(ext_dir, ext_args)
            run_gareus(ext_args, ext_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress)
            if _graceful_shutdown.is_set():
                _write_runtime_pool_reports(adaptive_dir, runtime_pool)
                payload = {
                    "schema_version": "adaptive_production_driver_summary_v1",
                    "status": "interrupted_after_checkpoint",
                    "interrupted_segment": str(ext_dir),
                    "epochs_completed": int(len(epoch_summaries)),
                    "epoch_summaries": _json_ready(epoch_summaries),
                }
                write_json(summary_path, payload)
                return payload
            runtime_pool.consume(
                label=f"final_extension_{ext_index + 1:03d}",
                kind="final_extension",
                n_states=max(1, len(registry.active_states())),
                steps=int(actual_ext_steps - _prior_ext_prod_done),
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
            # Pilot pooling is OPT-IN only. Adaptive-feedback pilot windows are NOT
            # registry states and pilot round dirs carry no epoch_window_map.csv, so
            # auto-pooling them would misattribute samples to the wrong MBAR states
            # and corrupt N_k. Only pool explicitly-provided pilot dirs (which must
            # carry a valid window map); see spec B4 (pilot pooling is a stretch).
            _pilot_dirs = [Path(p) for p in (getattr(args, "adaptive_production_pilot_sample_dirs", None) or [])]
            if _pilot_dirs:
                print(
                    "WARNING: pooling explicit pilot dirs into union MBAR — each MUST carry "
                    "an epoch_window_map.csv matching the registry states, or samples will be "
                    "misattributed."
                )
            union_inputs = build_union_state_mbar_inputs(
                adaptive_dir,
                registry,
                include_epochs=use_epoch_samples_for_mbar,
                pilot_dirs=_pilot_dirs,
                output_prefix="adaptive_union_mbar",
                tica_cv_version=getattr(args, "tica_cv_version", None),
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
    bridge_plan: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, str]:
    """Write a human/audit-friendly report explaining one adaptive decision step.

    ``bridge_plan`` is ``propose_actions_from_diagnostics``'s own per-weak-edge
    record (see its ``bridge_plan_out``).  It is threaded in rather than
    recomputed here on purpose: recomputing the placement prediction in the
    report writer would let the audit trail and the decision drift apart, which
    is precisely how the original placement bug stayed invisible.
    """
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
        # One row per weak edge the bridge planner considered, including the
        # ones it could place nothing for -- an unrepairable weak edge is a
        # finding, not an absence.
        "bridge_plan": _json_ready(list(bridge_plan or [])),
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
    plan = report.get("bridge_plan", []) or []
    if plan:
        lines.extend(["", "## Weak-edge bridge plan", ""])
        lines.append("Outcome `bridged` reconnects at least one endpoint; "
                     "`bisection_step` halves the gap for the next epoch; "
                     "`skipped_no_budget` / `refused_unreachable` placed nothing.")
        lines.append("")
        lines.append("| edge | overlap | needed | min useful | allocated | placed | outcome | note |")
        lines.append("|---|---:|---:|---:|---:|---:|---|---|")
        for row in plan[:50]:
            lines.append(
                f"| {row.get('state_i')}-{row.get('state_j')} | {row.get('overlap')} | "
                f"{row.get('bridges_needed')} | {row.get('min_useful_bridges')} | "
                f"{row.get('bridges_allocated')} | {row.get('bridges_placed')} | "
                f"{row.get('outcome')} | {row.get('note')} |"
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
