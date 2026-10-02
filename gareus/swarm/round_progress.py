"""One production-style progress stream for a whole swarm round.

The swarm round is a many-replica unbiased production (one replica per member), so it
reports exactly like an umbrella epoch does: phase ``gareus_production``, per-replica
``step``/``total_steps`` (mean member production steps out of one member's target) and
``n_replicas`` = members in this call's range. The monitor then shows percent, ETA and
aggregate ns/day with no swarm-specific code. ``extra`` carries ``swarm: True``,
``epoch: 0`` and member counts so readers can tell it apart from an umbrella epoch.

Members run as threads and the progress sink is not thread-safe, so every update goes
through one lock, and emissions are throttled here (members step every frame).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

PHASE = "gareus_production"


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

    @property
    def step(self) -> int:
        """Mean production steps per member (the per-replica step of a production epoch)."""
        return min(self.n_prod_steps, self._member_steps // self.n_members)

    def member_started(self, member_id) -> None:
        with self._lock:
            self._running.add(member_id)
            self._emit_locked(force=True)

    def add_prod_steps(self, member_id, n: int) -> None:
        with self._lock:
            self._member_steps += int(n)
            self._steps_by_member[member_id] = self._steps_by_member.get(member_id, 0) + int(n)
            self._emit_locked(force=False)

    def member_finished(self, member_id, ok: bool) -> None:
        """Close a member; a failed one is credited its missing steps so the round reaches 100 %."""
        with self._lock:
            self._running.discard(member_id)
            run = self._steps_by_member.pop(member_id, 0)
            self._member_steps += max(0, self.n_prod_steps - run)
            self._n_done += 1
            if not ok:
                self._n_failed += 1
            self._emit_locked(force=True)

    def emit(self, force: bool = True) -> None:
        with self._lock:
            self._emit_locked(force=force)

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
            },
        )


def make_round_progress(sink, *, n_members: int, n_prod_steps: int, n_already_done: int,
                        timestep_fs: float, round_index: int) -> Optional[SwarmRoundProgress]:
    if sink is None or n_members <= 0:
        return None
    return SwarmRoundProgress(sink, n_members=n_members, n_prod_steps=n_prod_steps,
                              n_already_done=n_already_done, timestep_fs=timestep_fs,
                              round_index=round_index)
