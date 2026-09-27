"""Optional wall-time accounting of the production loop's phases.

Enabled with ``--production-phase-timers`` (YAML ``production_phase_timers``,
off by default). Diagnostics only: nothing here changes what production
computes, and a disabled instance is a no-op context manager.

Why it exists: the replica-admission cap gave +57 % in the MD-only harness but
+2.4 % in production, leaving ~19 s per 2000 steps unattributed
(docs/superpowers/specs/performance-upgrades/review-2026-09-27-p5-p7.md). The
phases below split one production iteration into stepping, sampling, logging,
exchange and checkpointing so that cost can be located before any further
upgrade is ranked.

Phase names containing a dot (``"sample.fetch"``) are sub-phases: they are
reported, but excluded from ``top_level_sum_s`` so the parent is not counted
twice. ``unattributed_s`` is loop wall time outside every top-level phase.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Optional

STEPS_REFERENCE = 2000  # LCM of the 250-step sample and 400-step exchange cadences


class PhaseTimers:
    """Accumulate wall seconds and call counts per named phase."""

    def __init__(self, enabled: bool, clock: Callable[[], float] = time.perf_counter):
        self.enabled = bool(enabled)
        self._clock = clock
        self._seconds: dict[str, float] = {}
        self._calls: dict[str, int] = {}
        self._t0: Optional[float] = None
        self._steps = 0

    def begin(self) -> None:
        """Start the loop wall clock (call once, right before the production loop)."""
        if self.enabled:
            self._t0 = self._clock()

    def note_steps(self, n: int) -> None:
        if self.enabled:
            self._steps += int(n)

    def phase(self, name: str):
        """Context manager timing one occurrence of ``name`` (no-op when disabled)."""
        if not self.enabled:
            return nullcontext()
        return self._timed(name)

    @contextmanager
    def _timed(self, name: str) -> Iterator[None]:
        t = self._clock()
        try:
            yield
        finally:
            self._seconds[name] = self._seconds.get(name, 0.0) + (self._clock() - t)
            self._calls[name] = self._calls.get(name, 0) + 1

    def snapshot(self) -> dict:
        if not self.enabled:
            return {"enabled": False}
        wall = (self._clock() - self._t0) if self._t0 is not None else 0.0
        scale = (STEPS_REFERENCE / self._steps) if self._steps > 0 else None
        phases = {}
        for name in sorted(self._seconds):
            sec = self._seconds[name]
            calls = self._calls.get(name, 0)
            phases[name] = {
                "seconds": sec,
                "calls": calls,
                "mean_s": sec / calls if calls else None,
                "fraction_of_wall": sec / wall if wall > 0 else None,
                f"seconds_per_{STEPS_REFERENCE}_steps": sec * scale if scale is not None else None,
            }
        top = sum(s for n, s in self._seconds.items() if "." not in n)
        return {
            "enabled": True,
            "wall_s": wall,
            "steps": self._steps,
            "top_level_sum_s": top,
            "unattributed_s": wall - top,
            "phases": phases,
        }

    def summary_line(self) -> str:
        snap = self.snapshot()
        if not snap.get("enabled"):
            return "[phase-timers] disabled"
        wall = snap["wall_s"] or 0.0
        tops = sorted(((n, p["seconds"]) for n, p in snap["phases"].items() if "." not in n),
                      key=lambda kv: -kv[1])
        parts = [f"{n} {100.0 * s / wall:.1f}%" for n, s in tops] if wall > 0 else []
        if wall > 0:
            parts.append(f"unattributed {100.0 * snap['unattributed_s'] / wall:.1f}%")
        return f"[phase-timers] {snap['steps']} steps in {wall:.1f} s: " + ", ".join(parts)

    def write(self, path: Path, extra: Optional[Mapping] = None) -> None:
        """Atomically write the snapshot (plus ``extra``) as JSON; no-op when disabled."""
        if not self.enabled:
            return
        payload = {**self.snapshot(), **(dict(extra) if extra else {})}
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp, path)


def aggregate_npt_timings(drivers: Iterable) -> dict:
    """Sum every replica controller's NPT ``_timings`` (read/scale/restore/evaluate/verify, attempts)."""
    total: dict[str, float] = {}
    n = 0
    for driver in drivers:
        timings = getattr(getattr(driver, "controller", None), "_timings", None)
        if not isinstance(timings, Mapping):
            continue
        n += 1
        for key, value in timings.items():
            total[key] = total.get(key, 0) + value
    return {**total, "controllers": n}
