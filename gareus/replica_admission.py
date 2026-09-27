"""Bounded per-GPU replica admission for production stepping.

Spec: docs/superpowers/specs/2026-09-26-replica-admission-design.md (§4, §6).

At 59 resident contexts per GPU under MPS, submitting every replica's step at
once keeps ~236 host threads busy and makes MPS interleave 59 clients per GPU.
``AdmissionDispatcher`` advances every replica by exactly ``nsteps`` in short
turns, with at most ``limit`` turns per GPU queue in flight, on the existing
per-replica affinity pool (each replica's turns still run on its own thread).

Protocol (spec §4): all shared state is touched only under one plain Lock,
which is never held across ``pool.submit``, ``add_done_callback``, the turn
function or ``Event.wait``. A done-callback runs in the worker thread, or
synchronously in the attaching thread when the future had already finished;
both paths go through ``_on_done``. Synchronous callbacks are trampolined
through a per-thread pending list so many instant turns never recurse.
"""

from __future__ import annotations

import logging
import operator
import threading
from collections import deque
from concurrent.futures import CancelledError
from typing import Callable, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

SHARED_QUEUE_KEY = "shared"

# Indirection so tests can wrap Event.wait (lock-discipline test).
_Event = threading.Event

# Marks a popped turn that was never submitted because the run had already failed.
_NOT_STARTED = object()


def _positive_int(value, name: str) -> int:
    """An integer >= 1; rejects bools and floats instead of truncating (numpy ints are fine)."""
    if isinstance(value, bool):
        raise ValueError(f"AdmissionDispatcher {name} must be an integer >= 1, got {value!r}")
    try:
        number = operator.index(value)
    except TypeError:
        raise ValueError(f"AdmissionDispatcher {name} must be an integer >= 1, got {value!r}") from None
    if number < 1:
        raise ValueError(f"AdmissionDispatcher {name} must be an integer >= 1, got {value!r}")
    return int(number)


class AdmissionDispatcher:
    """Advance replicas in capped per-queue FIFO turns (see module docstring)."""

    def __init__(self, pool, gpu_of_replica: Sequence[str], limit: int, turn_steps: int):
        limit = _positive_int(limit, "limit")
        turn_steps = _positive_int(turn_steps, "turn_steps")
        self._pool = pool
        self._queue_of = [str(k) for k in gpu_of_replica]
        self._keys = list(dict.fromkeys(self._queue_of))
        resident = {k: self._queue_of.count(k) for k in self._keys}
        self.limit_per_queue = {k: min(int(limit), resident[k]) for k in self._keys}
        self.turn_steps = int(turn_steps)
        self._lock = self._new_lock()
        self._tls = threading.local()
        self._running = False
        self._reset_state_locked()

    @staticmethod
    def _new_lock():
        return threading.Lock()

    # ------------------------------------------------------------ state

    def _reset_state_locked(self) -> None:
        self._queues = {k: deque() for k in self._keys}
        self._remaining = [0] * len(self._queue_of)
        self._in_flight = {k: 0 for k in self._keys}
        self._total_in_flight = 0
        self._failed = False
        self._first_exc: Optional[BaseException] = None
        self._later_excs: list[BaseException] = []
        self._done = _Event()
        self._fn: Optional[Callable[[int, int], None]] = None

    def _record_failure_locked(self, exc: BaseException) -> None:
        if not self._failed:
            self._failed = True
            self._first_exc = exc
        else:
            self._later_excs.append(exc)
        for q in self._queues.values():
            q.clear()

    def _record_turn_locked(self, i: int, n: int, exc) -> None:
        if exc is _NOT_STARTED:
            return
        if exc is not None:
            self._record_failure_locked(exc)
            return
        if self._failed:
            return
        self._remaining[i] -= n
        if self._remaining[i] > 0:
            self._queues[self._queue_of[i]].append(i)

    def _release_slot_locked(self, i: int) -> None:
        self._in_flight[self._queue_of[i]] -= 1
        self._total_in_flight -= 1

    def _pop_launches_locked(self) -> list[tuple[int, int]]:
        if self._failed:
            return []
        launches = []
        for k in self._keys:
            q = self._queues[k]
            while q and self._in_flight[k] < self.limit_per_queue[k]:
                i = q.popleft()
                self._in_flight[k] += 1
                self._total_in_flight += 1
                launches.append((i, min(self.turn_steps, self._remaining[i])))
        return launches

    # ------------------------------------------------------- completion

    def _on_done(self, i: int, n: int, exc) -> None:
        """Account one finished (or never-started) turn and launch what is now free.

        Total by construction: an internal error is recorded as the run's
        failure, the slot is always released, and the Event is always set once
        nothing is in flight -- concurrent.futures only logs an exception raised
        inside a done-callback, so an unguarded one would hang ``run``.
        """
        launches: list[tuple[int, int]] = []
        with self._lock:
            try:
                self._record_turn_locked(i, n, exc)
            except BaseException as internal:  # noqa: BLE001 - must not escape a callback
                self._record_failure_locked(internal)
            finally:
                self._release_slot_locked(i)
            try:
                launches = self._pop_launches_locked()
            except BaseException as internal:  # noqa: BLE001
                self._record_failure_locked(internal)
                launches = []
            if self._total_in_flight == 0 and not launches:
                self._done.set()
        self._submit(launches)

    def _callback(self, i: int, n: int):
        def _cb(fut) -> None:
            try:
                exc = CancelledError() if fut.cancelled() else fut.exception()
            except BaseException as internal:  # noqa: BLE001
                exc = internal
            self._on_done(i, n, exc)

        return _cb

    # -------------------------------------------------------- submission

    def _submit(self, launches: list[tuple[int, int]]) -> None:
        tls = self._tls
        if getattr(tls, "active", False):
            # A synchronous callback inside this thread's own submit loop:
            # hand the launches to the outer loop instead of recursing.
            tls.pending.extend(launches)
            return
        tls.active = True
        tls.pending = deque(launches)
        try:
            while tls.pending:
                i, n = tls.pending.popleft()
                self._submit_one(i, n)
        finally:
            tls.active = False

    def _submit_one(self, i: int, n: int) -> None:
        with self._lock:
            skip = self._failed
        if skip:
            self._on_done(i, n, _NOT_STARTED)
            return
        try:
            fut = self._pool.submit(i, self._fn, i, n)
        except BaseException as exc:  # noqa: BLE001 - returned to the run as its failure
            self._on_done(i, n, exc)
            return
        fut.add_done_callback(self._callback(i, n))

    # --------------------------------------------------------------- run

    def run(self, fn: Callable[[int, int], None], nsteps: int) -> None:
        """Advance every replica by exactly ``nsteps``, in turns of at most ``turn_steps``.

        ``fn(replica_index, n)`` runs on the replica's own pool thread. Returns
        when every replica has completed ``nsteps``; on failure, re-raises the
        first exception only after no turn is in flight.
        """
        nsteps = int(nsteps)
        if nsteps <= 0:
            return
        with self._lock:
            if self._running:
                raise RuntimeError("AdmissionDispatcher.run is not re-entrant (called while already running)")
            self._running = True
            self._reset_state_locked()
            self._fn = fn
            for i, k in enumerate(self._queue_of):
                self._remaining[i] = nsteps
                self._queues[k].append(i)
            launches = self._pop_launches_locked()
            if not launches:
                self._done.set()
        try:
            self._submit(launches)
            self._done.wait()
        finally:
            with self._lock:
                first = self._first_exc
                later = list(self._later_excs)
                complete = all(r == 0 for r in self._remaining)
                self._running = False
                self._fn = None
        if first is not None:
            for extra in later:
                logger.error("replica admission: additional turn failure after the first: %r", extra)
            raise first
        if not complete:
            raise RuntimeError("AdmissionDispatcher.run finished with replicas short of nsteps (dispatcher bug)")


# ------------------------------------------------------------------ helpers


def _advance_item(item) -> int:
    """Today's step_all._step_item, unchanged: one replica's whole advance."""
    idx, driver, n = item
    driver.advance(int(n))
    return int(idx)


def advance_replicas(pool, drivers: Sequence, nsteps: int, dispatcher: Optional[AdmissionDispatcher]) -> None:
    """Advance every driver by exactly ``nsteps``.

    ``dispatcher is None`` is today's path: one ``pool.map`` over every
    driver's ``advance(nsteps)``. Otherwise the dispatcher runs capped turns.
    """
    nsteps = int(nsteps)
    if dispatcher is None:
        list(pool.map(_advance_item, [(i, d, nsteps) for i, d in enumerate(drivers)]))
        return
    dispatcher.run(lambda i, n: drivers[i].advance(n), nsteps)


def queue_keys_from_platform_props(props_list: Sequence[Mapping[str, str]]) -> list[str]:
    """One admission queue key per replica: its DeviceIndex, or ``"shared"``.

    A ``single-context-split`` DeviceIndex such as ``"0,1,2,3"`` is kept as
    one key, so every such context shares one queue (each spans every GPU).
    """
    keys = []
    for props in props_list:
        dev = str((props or {}).get("DeviceIndex", "") or "").strip()
        keys.append(dev if dev else SHARED_QUEUE_KEY)
    return keys


def effective_limits(queue_keys: Sequence[str], active_replicas_per_gpu) -> dict[str, int]:
    """Per-queue concurrency actually in force: ``min(cap, resident)``, or resident for ``"all"``."""
    resident: dict[str, int] = {}
    for k in queue_keys:
        resident[str(k)] = resident.get(str(k), 0) + 1
    if active_replicas_per_gpu == "all":
        return dict(resident)
    cap = int(active_replicas_per_gpu)
    return {k: min(cap, n) for k, n in resident.items()}


def make_dispatcher(pool, queue_keys: Sequence[str], active_replicas_per_gpu, turn_steps: int) -> Optional[AdmissionDispatcher]:
    """``None`` for ``"all"`` (today's code path), else an ``AdmissionDispatcher``."""
    if active_replicas_per_gpu == "all":
        return None
    return AdmissionDispatcher(pool, list(queue_keys), limit=int(active_replicas_per_gpu), turn_steps=int(turn_steps))
