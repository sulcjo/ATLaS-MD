"""One production-style progress stream for a whole swarm round, plus its live dashboard state.

The swarm round is a many-replica unbiased production (one replica per member), so it
reports exactly like an umbrella epoch does: phase ``gareus_production``, per-replica
``step``/``total_steps`` (mean member production steps out of one member's target) and
``n_replicas`` = members in this call's range. The monitor then shows percent, ETA and
aggregate ns/day with no swarm-specific code. ``extra`` carries ``swarm: True``,
``epoch: 0`` and member counts so readers can tell it apart from an umbrella epoch.

Members run as threads and the progress sink is not thread-safe, so every update goes
through one lock, and emissions are throttled here (members step every frame).

Dashboard (spec 2026-10-04-swarm-dashboard-design): the same object holds per-member phase,
device, rate ring, live CV coverage counters and a ring of finished-member events; under
``--tui-mode dashboard|interactive`` it renders ``gareus.dashboard.swarm_view`` frames, at most
one per ``--dashboard-render-interval-sec`` (2 s when unset), from a snapshot taken under the
lock but rendered and written outside it by whichever member thread is due -- never waiting
(a render in progress is skipped). ``live_status.json`` is written every 15 s. Display only:
nothing here changes what a member samples or writes for analysis.
"""
from __future__ import annotations

import bisect
import json
import math
import os
import statistics
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PHASE = "gareus_production"
DEFAULT_FRAME_INTERVAL_S = 2.0
STATUS_INTERVAL_S = 15.0
RATE_WINDOW_S = 60.0
STALL_S = 600.0
COMPLETION_BIN_S = 600.0
N_COMPLETION_BINS = 12
EVENTS_KEPT = 50
HIST_BINS = 48
DASHBOARD_MODES = ("dashboard", "interactive")
ACTIVE = ("graft_wait", "grafting", "equilibrating", "production")


class _Member:
    __slots__ = ("phase", "device", "t_start", "t_phase", "steps", "ring", "status", "ns_per_day", "cell_id")

    def __init__(self, phase: str = "queued", cell_id: Optional[str] = None) -> None:
        self.phase = phase
        self.device: Optional[str] = None
        self.t_start: Optional[float] = None
        self.t_phase: Optional[float] = None
        self.steps = 0
        self.ring: deque = deque()
        self.status: Optional[str] = None
        self.ns_per_day: Optional[float] = None
        self.cell_id = cell_id


class SwarmRoundProgress:
    def __init__(
        self,
        sink,
        *,
        n_members: int,
        n_prod_steps: int,
        n_already_done: int,
        timestep_fs: float,
        round_index: int,
        min_interval_s: float = 2.0,
        member_ids: Optional[Sequence[int]] = None,
        done_ids: Optional[Sequence[int]] = None,
        cells: Optional[Dict[int, str]] = None,
        edges: Optional[Dict[str, Sequence[float]]] = None,
        run_label: str = "",
        live_status_path: Optional[Path] = None,
    ) -> None:
        self.sink = sink
        self.n_members = max(1, int(n_members))
        self.n_prod_steps = max(1, int(n_prod_steps))
        self.timestep_fs = float(timestep_fs)
        self.round_index = int(round_index)
        self.min_interval_s = float(min_interval_s)
        self._lock = threading.Lock()
        self._member_steps = int(n_already_done) * self.n_prod_steps
        self._n_done = int(n_already_done)
        self._n_failed = 0
        self._running: set = set()
        self._steps_by_member: dict = {}
        self._last_emit = 0.0
        # --- dashboard state ---
        ids = list(member_ids) if member_ids is not None else list(range(self.n_members))
        done = {int(i) for i in (done_ids or ())}
        cells = cells or {}
        self._members: Dict[int, _Member] = {}
        for i in ids:
            m = _Member("done" if int(i) in done else "queued", cells.get(int(i)))
            if int(i) in done:
                m.steps, m.status = self.n_prod_steps, "ok"
            self._members[int(i)] = m
        self._t0 = time.time()
        self._events: deque = deque(maxlen=EVENTS_KEPT)
        self._completions: List[float] = []
        self.run_label = str(run_label)
        self.live_status_path = Path(live_status_path) if live_status_path else None
        self._edges = {k: [float(x) for x in v] for k, v in (edges or {}).items() if v is not None and len(v) >= 2}
        self._hist: Dict[str, Dict[str, Any]] = {}
        for name in ("cv1", "rg", "e2e"):
            e = self._edges.get(name)
            if e:
                lo, hi = min(e), max(e)
                pad = 0.1 * (hi - lo) if hi > lo else 0.1
                self._hist[name] = {"lo": lo - pad, "hi": hi + pad, "counts": [0] * HIST_BINS, "below": 0, "above": 0}
        self._cells_visited: set = set()
        self._cells_total = 0
        if self._edges:
            self._cells_total = 1
            for e in self._edges.values():
                self._cells_total *= len(e) - 1
        # frames
        args = getattr(sink, "args", None)
        mode = str(getattr(sink, "tui_mode", "") or "").lower()
        console = str(getattr(sink, "mode", "both") or "both").lower() in ("console", "both")
        self.frames_enabled = sink is not None and mode in DASHBOARD_MODES and console
        interval = float(getattr(args, "dashboard_render_interval_sec", 0.0) or 0.0) if args is not None else 0.0
        self.frame_interval_s = interval if interval > 0 else DEFAULT_FRAME_INTERVAL_S
        self._glyphs = str(getattr(args, "tui_glyphs", "unicode") or "unicode") if args is not None else "unicode"
        self._last_frame = float("-inf")      # the first frame / status file is due at once
        self._last_status = float("-inf")
        self._render_lock = threading.Lock()
        self._render_failed = False

    @property
    def step(self) -> int:
        """Mean production steps per member (the per-replica step of a production epoch)."""
        return min(self.n_prod_steps, self._member_steps // self.n_members)

    # --- hooks ---------------------------------------------------------------------------

    def _member(self, member_id) -> _Member:
        m = self._members.get(int(member_id))
        if m is None:
            m = self._members[int(member_id)] = _Member()
        return m

    def member_started(self, member_id, device: Optional[str] = None) -> None:
        with self._lock:
            self._running.add(member_id)
            m = self._member(member_id)
            now = time.time()
            m.phase, m.t_start, m.t_phase = "graft_wait", now, now
            m.device = None if device is None else str(device)
            self._emit_locked(force=True)
        self._after_update()

    def member_phase(self, member_id, phase: str) -> None:
        with self._lock:
            m = self._member(member_id)
            if m.phase not in ("done", "failed"):
                m.phase, m.t_phase = str(phase), time.time()
                if phase == "production":
                    m.ring.append((m.t_phase, m.steps))
        self._after_update()

    def add_prod_steps(self, member_id, n: int) -> None:
        with self._lock:
            self._member_steps += int(n)
            self._steps_by_member[member_id] = self._steps_by_member.get(member_id, 0) + int(n)
            m = self._member(member_id)
            now = time.time()
            m.steps += int(n)
            if m.phase not in ("done", "failed"):
                m.phase = "production"
            m.ring.append((now, m.steps))
            while len(m.ring) > 2 and now - m.ring[0][0] > RATE_WINDOW_S:
                m.ring.popleft()
            self._emit_locked(force=False)
        self._after_update()

    def add_frame(self, member_id, cv1: float, rg_nm: float, e2e_nm: float) -> None:
        """Live coverage counters from one swarm trace row (cv1, Rg nm, end-to-end nm)."""
        vals = {"cv1": cv1, "rg": rg_nm, "e2e": e2e_nm}
        with self._lock:
            key, complete = [], True
            for name, v in vals.items():
                try:
                    x = float(v)
                except (TypeError, ValueError):
                    x = float("nan")
                h = self._hist.get(name)
                if h is not None and math.isfinite(x):
                    if x < h["lo"]:
                        h["below"] += 1
                    elif x >= h["hi"]:
                        h["above"] += 1
                    else:
                        h["counts"][min(HIST_BINS - 1, int((x - h["lo"]) / (h["hi"] - h["lo"]) * HIST_BINS))] += 1
                e = self._edges.get(name)
                if e:
                    if math.isfinite(x):
                        key.append(min(len(e) - 2, max(0, bisect.bisect_right(e, x) - 1)))
                    else:
                        complete = False
            if self._cells_total and complete:
                self._cells_visited.add(tuple(key))

    def member_finished(self, member_id, ok: bool, status: Optional[str] = None,
                        ns_per_day: Optional[float] = None) -> None:
        """Close a member; a failed one is credited its missing steps so the round reaches 100 %."""
        with self._lock:
            self._running.discard(member_id)
            run = self._steps_by_member.pop(member_id, 0)
            self._member_steps += max(0, self.n_prod_steps - run)
            self._n_done += 1
            if not ok:
                self._n_failed += 1
            m = self._member(member_id)
            now = time.time()
            m.phase, m.t_phase = ("done" if ok else "failed"), now
            m.status = str(status or ("ok" if ok else "failed"))
            m.ns_per_day = None if ns_per_day is None else float(ns_per_day)
            self._completions.append(now)
            self._events.appendleft({"member_id": int(member_id), "status": m.status, "cell_id": m.cell_id,
                                     "ns_per_day": m.ns_per_day, "wall": now})
            self._emit_locked(force=True)
        self._after_update()

    def emit(self, force: bool = True) -> None:
        with self._lock:
            self._emit_locked(force=force)
        self._after_update()

    def emit_event(self, event: dict) -> None:
        """A raw sink event from a member thread, serialised with the progress lines."""
        with self._lock:
            if self.sink is not None:
                self.sink.emit(event)

    def close(self) -> None:
        """Final line, then forget this phase's rate baseline.

        The sink is one per process and keys its ETA/ns-per-day baseline by phase name; a
        later umbrella epoch in the same job reports as ``gareus_production`` too and must
        not measure its rate from the swarm's start time and mean-member step.
        """
        with self._lock:
            self._emit_locked(force=True)
            for attr in ("baseline_step", "phase_start"):
                table = getattr(self.sink, attr, None)
                if isinstance(table, dict):
                    table.pop(PHASE, None)
        self._after_update(final=True)

    def _emit_locked(self, force: bool) -> None:
        if self.sink is None:
            return
        now = time.time()
        if not force and (now - self._last_emit) < self.min_interval_s:
            return
        self._last_emit = now
        failed = f", {self._n_failed} failed" if self._n_failed else ""
        message = (
            f"epoch 0 swarm round {self.round_index}: {self.n_members} replicas, "
            f"{len(self._running)} running, {self._n_done}/{self.n_members} members done{failed}"
        )
        self.sink.progress(
            PHASE, self.step, self.n_prod_steps, message=message, timestep_fs=self.timestep_fs,
            n_replicas=self.n_members, force=force,
            extra={
                "swarm": True, "epoch": 0, "swarm_round": self.round_index,
                "members_total": self.n_members, "members_done": self._n_done,
                "members_running": len(self._running), "members_failed": self._n_failed,
                "swarm_dashboard": bool(self.frames_enabled and not self._render_failed),
            },
        )

    # --- snapshot ------------------------------------------------------------------------

    @staticmethod
    def _rate(m: _Member) -> Optional[float]:
        """Recent production steps per second of one member (its ring's span)."""
        if len(m.ring) < 2:
            return None
        (t0, s0), (t1, s1) = m.ring[0], m.ring[-1]
        return (s1 - s0) / (t1 - t0) if t1 > t0 and s1 > s0 else None

    def snapshot(self, now: Optional[float] = None):
        """A detached, plain-data ``SwarmSnapshot`` taken under the lock."""
        from gareus.dashboard.swarm_view import DeviceStat, Histogram, SwarmSnapshot  # noqa: PLC0415
        now = time.time() if now is None else float(now)
        per_day = self.timestep_fs * 1e-6 * 86400.0          # steps/s -> ns/day
        with self._lock:
            ids = sorted(self._members)
            ms = [self._members[i] for i in ids]
            fracs = tuple(min(1.0, m.steps / self.n_prod_steps) for m in ms)
            rates = {i: self._rate(m) for i, m in zip(ids, ms)}
            active = [(i, m) for i, m in zip(ids, ms) if m.phase in ACTIVE]
            prod_rates = [rates[i] for i, m in active if m.phase == "production" and rates[i]]
            agg = sum(prod_rates)
            med = statistics.median(prod_rates) if prod_rates else None
            remaining = sum(self.n_prod_steps - min(self.n_prod_steps, m.steps) for m in ms
                            if m.phase not in ("done", "failed"))
            eta_mean = remaining / agg if agg > 0 else None
            eta_last = None
            if med:
                per = [(self.n_prod_steps - min(self.n_prod_steps, m.steps)) / (rates[i] or med) for i, m in active]
                if any(m.phase == "queued" for m in ms):
                    per.append(self.n_prod_steps / med)
                eta_last = max(per) if per else 0.0
            stalled = tuple(i for i, m in active if m.phase == "production" and m.ring
                            and now - m.ring[-1][0] > STALL_S)
            devs: Dict[str, Dict[str, float]] = {}
            for i, m in active:
                d = devs.setdefault(m.device or "?", {"n": 0, "rate": 0.0, "frac": 0.0})
                d["n"] += 1
                d["rate"] += rates[i] or 0.0
                d["frac"] += min(1.0, m.steps / self.n_prod_steps)
            dev_rates = [d["rate"] for d in devs.values() if d["rate"] > 0]
            dmed = statistics.median(dev_rates) if dev_rates else 0.0
            devices = tuple(DeviceStat(k, int(v["n"]), v["rate"] * per_day, v["frac"] / max(1, v["n"]),
                                       slow=bool(dmed > 0 and 0 < v["rate"] < 0.5 * dmed))
                            for k, v in sorted(devs.items()))
            hists = tuple(Histogram(n, h["lo"], h["hi"], tuple(h["counts"]), h["below"], h["above"])
                          for n, h in self._hist.items())
            nbins = [0] * N_COMPLETION_BINS
            for t in self._completions:
                age = now - t
                if 0 <= age < COMPLETION_BIN_S * N_COMPLETION_BINS:
                    nbins[N_COMPLETION_BINS - 1 - int(age // COMPLETION_BIN_S)] += 1
            starts = [m.t_start for m in ms if m.t_start]
            t0 = min(starts) if starts else self._t0
            return SwarmSnapshot(
                now=now, run_label=self.run_label, round_index=self.round_index, n_members=len(ms),
                timestep_fs=self.timestep_fs, n_prod_steps=self.n_prod_steps,
                bins=tuple(len(self._edges[n]) - 1 for n in ("cv1", "rg", "e2e") if self._edges.get(n)),
                member_ids=tuple(ids), phases=tuple(m.phase for m in ms), production_fraction=fracs,
                elapsed_s=max(0.0, now - t0), mean_fraction=(sum(fracs) / len(fracs)) if fracs else 0.0,
                aggregate_ns_per_day=agg * per_day,
                median_member_ns_per_day=med * per_day if med else float("nan"),
                slowest_member_ns_per_day=min(prod_rates) * per_day if prod_rates else float("nan"),
                eta_mean_s=eta_mean, eta_last_s=eta_last, stalled=stalled, devices=devices, histograms=hists,
                cells_visited=len(self._cells_visited), cells_total=self._cells_total,
                completions_per_bin=tuple(nbins), completion_bin_s=COMPLETION_BIN_S,
                events=tuple(dict(e) for e in self._events))

    # --- frames and status sidecar ---------------------------------------------------------

    def _after_update(self, final: bool = False) -> None:
        """Outside the lock: render a frame / write the status file when due (never waits,
        except for the round's final frame)."""
        now = time.time()
        frame = status = False
        with self._lock:
            if self.frames_enabled and not self._render_failed and (final or now - self._last_frame >= self.frame_interval_s):
                self._last_frame, frame = now, True
            if self.live_status_path is not None and (final or now - self._last_status >= STATUS_INTERVAL_S):
                self._last_status, status = now, True
        if not (frame or status):
            return
        if not self._render_lock.acquire(blocking=final):
            return
        try:
            snap = self.snapshot(now)
            if status:
                self._write_status(snap)
            if frame:
                self._render(snap)
        finally:
            self._render_lock.release()

    def _render(self, snap) -> None:
        try:
            import shutil  # noqa: PLC0415
            from gareus.dashboard.swarm_view import render_swarm_screen  # noqa: PLC0415
            from gareus.tui import write_tui_frame  # noqa: PLC0415
            size = shutil.get_terminal_size((160, 40))
            text = render_swarm_screen(snap, term_w=size.columns, term_h=size.lines, glyphs=self._glyphs)
            write_tui_frame(text, getattr(self.sink, "args", None))
        except Exception as exc:     # a display failure never stops a member
            self._render_failed = True
            print(f"WARNING: swarm dashboard disabled after a render error ({type(exc).__name__}: {exc})",
                  file=sys.stderr, flush=True)

    def _write_status(self, snap) -> None:
        try:
            rec = {"schema_version": "swarm_live_status_v1", "wall_time_s": snap.now, "round": snap.round_index,
                   "n_members": snap.n_members,
                   "counts": {p: snap.count(p) for p in ("queued", *ACTIVE, "done", "failed")},
                   "member_ids": list(snap.member_ids), "phases": list(snap.phases),
                   "production_fraction": [round(f, 4) for f in snap.production_fraction],
                   "mean_fraction": snap.mean_fraction, "aggregate_ns_per_day": snap.aggregate_ns_per_day,
                   "eta_mean_s": snap.eta_mean_s, "eta_last_s": snap.eta_last_s, "stalled": list(snap.stalled),
                   "devices": [{"device": d.device, "n_active": d.n_active, "ns_per_day": d.ns_per_day,
                                "production_fraction": d.production_fraction, "slow": d.slow} for d in snap.devices],
                   "cells_visited": snap.cells_visited, "cells_total": snap.cells_total}
            path = self.live_status_path
            tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
            tmp.write_text(json.dumps(_finite(rec)))
            os.replace(tmp, path)
        except Exception as exc:     # the sidecar is a convenience; never stop a member for it
            print(f"WARNING: swarm live_status.json not written ({type(exc).__name__}: {exc})", file=sys.stderr)


def _finite(x):
    """JSON-safe copy: non-finite floats become null."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


def make_round_progress(sink, *, n_members: int, n_prod_steps: int, n_already_done: int,
                        timestep_fs: float, round_index: int, **dashboard: Any) -> Optional[SwarmRoundProgress]:
    if sink is None or n_members <= 0:
        return None
    return SwarmRoundProgress(sink, n_members=n_members, n_prod_steps=n_prod_steps,
                              n_already_done=n_already_done, timestep_fs=timestep_fs,
                              round_index=round_index, **dashboard)
