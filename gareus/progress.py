"""
Progress helpers and OpenMM resource cleanup.

This module encapsulates the :class:`GuiProgressSink` class and
``release_openmm_contexts`` function, extracted from the monolithic
``gareus_peptide.py`` script.  The progress sink provides console
progress reporting and JSONL logging for use by the gareus
framework.  Keeping these routines in a dedicated module improves
testability and allows them to be reused across the package.
"""

from __future__ import annotations

import gc
import shutil
import time
from pathlib import Path
from typing import Optional

from .io import BufferedJsonlWriter
from .colors import color_text
from .tui import (
    make_progress_bar,
    format_duration,
    strip_ansi_len,
    strip_ansi,
    write_tui_frame,
)

__all__ = [
    "release_openmm_contexts",
    "GuiProgressSink",
]


def release_openmm_contexts(*objects) -> None:
    """Best-effort release of OpenMM/OpenCL/CUDA resources before creating many contexts.

    OpenMM contexts own platform resources that Python may not release until a
    later garbage‑collection pass.  That delay is normally harmless, but a
    workflow that creates many short‑lived preparation simulations and then
    constructs dozens of production replicas can trip OpenCL context creation
    errors such as ``clCreateContext(-6)``.  Explicitly deleting references and
    forcing a collection at phase boundaries is cheap and avoids keeping old
    contexts around like very expensive ghosts.
    """
    for obj in objects:
        try:
            if obj is not None and hasattr(obj, "context"):
                # Drop the heavy Context reference first when possible.
                obj.context = None  # type: ignore[attr-defined]
        except Exception:
            pass
    try:
        gc.collect()
    except Exception:
        pass


class GuiProgressSink:
    """Console progress plus JSONL event stream for an external GUI.

    The JSONL stream is deliberately boring: one JSON object per line. A GUI can
    tail it without importing OpenMM, because even progress bars deserve fewer
    dependencies.  This class writes progress updates both to the console and
    to a JSONL file, throttled to an update interval.
    """

    def __init__(self, out_dir: Path, args) -> None:
        self.out_dir = Path(out_dir)
        # Normalise progress and TUI modes from the args namespace.  Default
        # values are taken when attributes are missing or falsy.
        self.mode = str(getattr(args, "progress_mode", "both") or "both").lower()
        self.tui_mode = str(getattr(args, "tui_mode", "dashboard") or "dashboard").lower()
        self.args = args
        self.progress_jsonl = self.out_dir / str(getattr(args, "progress_jsonl", "progress.jsonl"))
        self.update_interval_s = float(getattr(args, "progress_update_interval_sec", 0.25) or 0.25)
        self.bar_width = int(getattr(args, "progress_bar_width", 36) or 36)
        self.start_wall = time.time()
        # Keep per‑phase start times so each phase displays its own ETA.
        self.phase_start: dict[str, float] = {}
        # First step seen in this process for each phase. ``step`` is cumulative
        # across a resumed chain while ``phase_start`` restarts every job, so any
        # rate must be computed from the difference and not from ``step`` itself.
        # Before this existed, chignolin_7 reported 5,293 ns/day against an
        # actual ~283 -- high by the ratio of campaign age to job age.
        self.baseline_step: dict[str, int] = {}
        self.last_console = 0.0
        self.last_json = 0.0
        self.handle: Optional[BufferedJsonlWriter] = None
        if self.mode in {"jsonl", "both"}:
            self.handle = BufferedJsonlWriter(
                self.progress_jsonl,
                append=True,
                flush_rows=int(getattr(args, "jsonl_flush_rows", 500) or 500),
            )

    def close(self) -> None:
        """Close any open JSONL handle.

        Errors during close are intentionally suppressed, matching the
        behaviour of the monolithic implementation.  Closing multiple
        times is safe.
        """
        if self.handle is not None:
            try:
                self.handle.close()
            except Exception:
                pass
            self.handle = None

    def emit(self, event: dict) -> None:
        """Write a JSON event to the progress log.

        A ``wall_time_s`` and ``elapsed_s`` field are added if missing.  If
        there is no JSONL handle (e.g. progress mode is console‑only), this
        function silently does nothing.
        """
        payload = dict(event)
        payload.setdefault("wall_time_s", time.time())
        payload.setdefault("elapsed_s", time.time() - self.start_wall)
        if self.handle is not None:
            try:
                self.handle.write_json(payload)
            except Exception:
                pass

    def progress(
        self,
        phase: str,
        step: int,
        total_steps: Optional[int],
        message: str = "",
        timestep_fs: Optional[float] = None,
        n_replicas: int = 1,
        force: bool = False,
        extra: Optional[dict] = None,
    ) -> None:
        """Report progress for a simulation phase.

        Args:
            phase: Name of the current phase (e.g. "npt_equilibration").
            step: Current iteration within the phase.
            total_steps: Total number of steps, or ``None`` if indeterminate.
            message: Optional free‑text annotation.
            timestep_fs: Simulation timestep in femtoseconds for performance metrics.
            n_replicas: Number of replicas contributing to aggregate metrics.
            force: Emit an update regardless of the configured interval.
            extra: Additional key/value pairs to include in the JSON payload.
        """
        now = time.time()
        phase = str(phase)
        self.phase_start.setdefault(phase, now)
        total = int(total_steps) if total_steps is not None else 0
        step_int = int(step)
        self.baseline_step.setdefault(phase, step_int)
        segment_steps = step_int - self.baseline_step[phase]
        frac = (float(step_int) / float(total)) if total > 0 else 0.0
        frac = max(0.0, min(1.0, frac))
        elapsed = now - self.phase_start[phase]
        # ETA from this segment's observed rate. Using ``frac`` of the whole run
        # against a segment-local ``elapsed`` would assume this process produced
        # every completed step, which after a resume it did not.
        eta: Optional[float] = None
        if segment_steps > 0 and elapsed > 0.0 and total > step_int:
            eta = (total - step_int) * elapsed / segment_steps

        payload: dict[str, float | int | str | None] = {
            "event": "progress",
            "phase": phase,
            "step": step_int,
            "segment_steps": segment_steps,
            "total_steps": total,
            "fraction": frac,
            "percent": 100.0 * frac,
            "message": message,
            "elapsed_s": elapsed,
            "eta_s": eta,
            "wall_elapsed_s": now - self.start_wall,
        }
        if timestep_fs is not None and float(timestep_fs) > 0.0:
            sim_time_ns = step_int * float(timestep_fs) / 1.0e6
            aggregate_ns = sim_time_ns * max(1, int(n_replicas))
            payload["sim_time_ps"] = sim_time_ns * 1000.0
            payload["sim_time_ns"] = sim_time_ns
            payload["aggregate_sim_time_ns"] = aggregate_ns
            # Rates describe THIS process's segment; the cumulative totals above
            # stay cumulative, because as totals they are correct.
            if elapsed > 0 and segment_steps > 0:
                segment_ns = segment_steps * float(timestep_fs) / 1.0e6
                payload["ns_per_day"] = segment_ns / elapsed * 86400.0
                payload["aggregate_ns_per_day"] = (
                    segment_ns * max(1, int(n_replicas)) / elapsed * 86400.0
                )
                payload["wall_s_per_ns"] = elapsed / segment_ns
                payload["wall_h_per_us"] = elapsed / segment_ns * 1000.0 / 3600.0
                payload["wall_ms_per_step"] = elapsed / segment_steps * 1000.0
                payload["steps_per_s"] = segment_steps / elapsed
        if extra:
            payload.update(extra)

        # JSONL emission — throttle based on update interval.
        if self.mode in {"jsonl", "both"} and (
            force or (now - self.last_json) >= self.update_interval_s
        ):
            self.emit(payload)
            self.last_json = now

        # Skip console output entirely if in JSONL‑only mode or if the TUI is disabled.
        if self.mode not in {"console", "both"} or self.tui_mode == "none":
            return

        # In dashboard/interactive modes, let the distance logger own the frame for
        # production and GaMD calibration phases.  This avoids flickering the
        # progress bar underneath the dashboard.
        if self.tui_mode in {"dashboard", "interactive"} and phase in {"gamd_calibration", "gareus_production"}:
            return

        # Console throttling.
        if not force and (now - self.last_console) < self.update_interval_s:
            return
        self.last_console = now

        # Build the console line.
        bar = make_progress_bar(frac, self.bar_width)
        parts: list[str] = [
            color_text(f"[{phase}]", "cyan", bold=True),
            f"[{bar}]",
            f"{step_int}/{total}" if total > 0 else f"{step_int}",
            color_text(
                f"{100.0 * frac:5.1f}%", "green" if frac >= 1.0 else "white", bold=True
            ),
            f"wall {format_duration(elapsed)}",
            f"eta {format_duration(eta)}",
        ]
        if "sim_time_ns" in payload:
            parts.append(f"sim {payload['sim_time_ns']:.3g} ns/rep")
        if "ns_per_day" in payload:
            parts.append(f"perf {payload['ns_per_day']:.2g} ns/day/rep")
            parts.append(f"{payload['aggregate_ns_per_day']:.2g} aggregate")
        if message:
            parts.append(str(message))
        line = " | ".join(parts)
        term_w = shutil.get_terminal_size((160, 24)).columns
        # Truncate the line if it would wrap at the terminal edge.  Strip ANSI
        # codes before computing visible length.
        if strip_ansi_len(line) > term_w - 1:
            raw = strip_ansi(line)
            line = raw[: max(20, term_w - 4)] + "..."
        if self.tui_mode in {"dashboard", "interactive"}:
            write_tui_frame(line, self.args)
        else:
            end = "\n" if force else "\r"
            print(line, end=end, flush=True)