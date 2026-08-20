"""Slow-changing campaign state, read from JSON already written to disk.

The MD-facing loop must not carry pool budgets or GaMD calibration values
around: they change at phase boundaries (minutes to hours) while CV samples
change every report interval. An mtime-gated read every few seconds is enough,
and keeps the stepping path untouched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SIDECAR_MIN_INTERVAL_S = 5.0
_MAX_PARENTS_SEARCHED = 5

_POOL_REL = Path("adaptive_production") / "adaptive_runtime_pool.json"
_GAMD_REL = Path("global_shared_gamd_setup") / "shared_gamd_setup_globals.json"
_GATE_REL = Path("adaptive_production") / "adaptive_quality_gate.json"
_SEEDING_NAME = "us_starting_structure_quality.json"


@dataclass(frozen=True)
class SidecarSnapshot:
    pool: Optional[dict] = None
    pool_mtime: Optional[float] = None
    gamd: Optional[dict] = None
    gamd_mtime: Optional[float] = None
    quality_gate: Optional[dict] = None
    seeding_quality: Optional[dict] = None
    errors: tuple[str, ...] = field(default=())
    # The resolved run root (`SidecarCache.run_root`, from `find_run_root`) --
    # carried onto the snapshot so `gareus/dashboard/context.py`'s `build_context`
    # can name the spine's identity line after the real run, not the segment
    # directory `DistanceLogger` was actually constructed with. `None` for a
    # hand-built snapshot (most tests, and any caller that never went through
    # `SidecarCache.snapshot`) -- callers must fall back to `out_dir` themselves.
    run_root: Optional[Path] = None


def find_run_root(out_dir: Path) -> Path:
    """Walk up from a segment directory to the run root.

    Keyed on ``adaptive_production/``, which only exists at the run root --
    ``global_shared_gamd_setup/`` also appears one level down and would give a
    false positive.
    """
    here = Path(out_dir)
    for candidate in [here, *list(here.parents)[:_MAX_PARENTS_SEARCHED]]:
        if (candidate / "adaptive_production").is_dir():
            return candidate
    for candidate in [here, *list(here.parents)[:_MAX_PARENTS_SEARCHED]]:
        if (candidate / "global_shared_gamd_setup").is_dir():
            return candidate
    return here


def _read_json(path: Path, errors: list[str]) -> tuple[Optional[dict], Optional[float]]:
    try:
        if not path.is_file():
            return None, None
        mtime = path.stat().st_mtime
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            errors.append(f"{path.name}: not a JSON object")
            return None, None
        return payload, mtime
    except Exception as exc:                      # unreadable is a display state
        errors.append(f"{path.name}: {type(exc).__name__}")
        return None, None


class SidecarCache:
    """Re-reads the sidecar files at most every ``min_interval_s`` seconds."""

    def __init__(self, out_dir: Path, min_interval_s: float = SIDECAR_MIN_INTERVAL_S) -> None:
        self.out_dir = Path(out_dir)
        self.run_root = find_run_root(self.out_dir)
        self.min_interval_s = float(min_interval_s)
        self._last_read_wall = float("-inf")
        self._snapshot = SidecarSnapshot()

    def _seeding_candidates(self) -> list[Path]:
        bases = [self.out_dir, self.out_dir.parent]
        return [base / sub / _SEEDING_NAME
                for base in bases for sub in ("setup", "us_starting_structures")]

    def snapshot(self, now: float) -> SidecarSnapshot:
        if float(now) - self._last_read_wall < self.min_interval_s:
            return self._snapshot
        errors: list[str] = []
        pool, pool_mtime = _read_json(self.run_root / _POOL_REL, errors)
        gamd, gamd_mtime = _read_json(self.run_root / _GAMD_REL, errors)
        gate, _gate_mtime = _read_json(self.run_root / _GATE_REL, errors)
        seeding = None
        for candidate in self._seeding_candidates():
            seeding, _mtime = _read_json(candidate, errors)
            if seeding is not None:
                break
        self._snapshot = SidecarSnapshot(
            pool=pool, pool_mtime=pool_mtime, gamd=gamd, gamd_mtime=gamd_mtime,
            quality_gate=gate, seeding_quality=seeding, errors=tuple(errors),
            run_root=self.run_root,
        )
        self._last_read_wall = float(now)
        return self._snapshot


__all__ = ["SIDECAR_MIN_INTERVAL_S", "SidecarCache", "SidecarSnapshot", "find_run_root"]
