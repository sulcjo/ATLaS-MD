"""AdmissionDispatcher: exact step totals, per-queue caps, failure drain, lock rule.

Spec: docs/superpowers/specs/2026-09-26-replica-admission-design.md §4, §7.
All CPU-only, no OpenMM.
"""

from __future__ import annotations

import random
import threading
import time
from collections import defaultdict

import pytest

import gareus.replica_admission as ra
from gareus.production import _ReplicaAffinityExecutor
from gareus.replica_admission import AdmissionDispatcher, SHARED_QUEUE_KEY

TIMEOUT_S = 30.0


class Recorder:
    """Turn function that records steps and an independent per-queue in-flight count."""

    def __init__(self, queue_of, sleep_s=0.0, sleep_fn=None):
        self.queue_of = list(queue_of)
        self.sleep_s = sleep_s
        self.sleep_fn = sleep_fn
        self.lock = threading.Lock()
        self.steps = defaultdict(int)
        self.in_flight = defaultdict(int)
        self.max_in_flight = defaultdict(int)
        self.per_replica_active = defaultdict(int)
        self.overlap_seen = False
        self.threads = defaultdict(set)

    def __call__(self, i, n):
        q = self.queue_of[i]
        with self.lock:
            self.in_flight[q] += 1
            self.max_in_flight[q] = max(self.max_in_flight[q], self.in_flight[q])
            self.per_replica_active[i] += 1
            if self.per_replica_active[i] > 1:
                self.overlap_seen = True
            self.threads[i].add(threading.get_ident())
        try:
            delay = self.sleep_fn() if self.sleep_fn else self.sleep_s
            if delay:
                time.sleep(delay)
        finally:
            with self.lock:
                self.in_flight[q] -= 1
                self.per_replica_active[i] -= 1
                self.steps[i] += n


def run_with_timeout(dispatcher, fn, nsteps, timeout=TIMEOUT_S):
    """Run dispatcher.run in a thread; fail the test on a hang, re-raise its exception."""
    box = {}

    def target():
        try:
            dispatcher.run(fn, nsteps)
        except BaseException as exc:  # noqa: BLE001 - the test inspects it
            box["exc"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), "AdmissionDispatcher.run hung"
    if "exc" in box:
        raise box["exc"]


@pytest.fixture(scope="module")
def pool():
    p = _ReplicaAffinityExecutor(320)
    yield p
    p.shutdown(wait=True)


def test_every_replica_advances_exactly_nsteps_over_consecutive_runs(pool):
    queues = ["0"] * 5 + ["1"] * 5
    rec = Recorder(queues, sleep_s=0.001)
    d = AdmissionDispatcher(pool, queues, limit=2, turn_steps=50)
    for n in (400, 250, 150, 400):
        run_with_timeout(d, rec, n)
    assert dict(rec.steps) == {i: 1200 for i in range(10)}
    assert not rec.overlap_seen


def test_in_flight_never_exceeds_limit_and_reaches_it(pool):
    queues = ["0"] * 8
    rec = Recorder(queues, sleep_s=0.02)
    d = AdmissionDispatcher(pool, queues, limit=3, turn_steps=50)
    run_with_timeout(d, rec, 200)
    assert rec.max_in_flight["0"] == 3
    assert d.limit_per_queue == {"0": 3}


def test_uneven_gpu_populations_59_59_59_58(pool):
    queues = ["0"] * 59 + ["1"] * 59 + ["2"] * 59 + ["3"] * 58
    rec = Recorder(queues, sleep_s=0.0005)
    d = AdmissionDispatcher(pool, queues, limit=8, turn_steps=50)
    run_with_timeout(d, rec, 150)
    assert all(rec.steps[i] == 150 for i in range(len(queues)))
    assert all(rec.max_in_flight[q] <= 8 for q in "0123")


def test_single_shared_queue(pool):
    queues = [SHARED_QUEUE_KEY] * 6
    rec = Recorder(queues, sleep_s=0.002)
    d = AdmissionDispatcher(pool, queues, limit=2, turn_steps=30)
    run_with_timeout(d, rec, 100)
    assert all(rec.steps[i] == 100 for i in range(6))
    assert rec.max_in_flight[SHARED_QUEUE_KEY] <= 2


def test_limit_one(pool):
    queues = ["0", "0", "0", "1", "1"]
    rec = Recorder(queues, sleep_s=0.002)
    d = AdmissionDispatcher(pool, queues, limit=1, turn_steps=25)
    run_with_timeout(d, rec, 100)
    assert all(rec.steps[i] == 100 for i in range(5))
    assert rec.max_in_flight["0"] == 1 and rec.max_in_flight["1"] == 1


def test_limit_above_resident_count_is_clipped(pool):
    queues = ["0", "0", "1"]
    d = AdmissionDispatcher(pool, queues, limit=8, turn_steps=50)
    assert d.limit_per_queue == {"0": 2, "1": 1}


def test_turn_length_not_dividing_nsteps_last_turn_shorter(pool):
    queues = ["0"] * 3
    calls = defaultdict(list)
    lock = threading.Lock()

    def fn(i, n):
        with lock:
            calls[i].append(n)

    d = AdmissionDispatcher(pool, queues, limit=1, turn_steps=30)
    run_with_timeout(d, fn, 100)
    assert all(calls[i] == [30, 30, 30, 10] for i in range(3))


def test_nsteps_below_turn_steps_is_one_turn(pool):
    queues = ["0"] * 4
    calls = defaultdict(list)
    lock = threading.Lock()

    def fn(i, n):
        with lock:
            calls[i].append(n)

    d = AdmissionDispatcher(pool, queues, limit=2, turn_steps=1000)
    for n in (400, 250):
        run_with_timeout(d, fn, n)
    assert all(calls[i] == [400, 250] for i in range(4))


def test_instant_turns_many_replicas_no_recursion(pool):
    # Instant fn: futures are often already finished when add_done_callback is
    # attached, so the callback runs synchronously in the submitting thread.
    # 300 replicas x 100 one-step turns would overflow the stack if each
    # synchronous callback recursed into the next submit.
    queues = [str(i % 4) for i in range(300)]
    rec = Recorder(queues)
    d = AdmissionDispatcher(pool, queues, limit=8, turn_steps=1)
    run_with_timeout(d, rec, 100, timeout=120)
    assert all(rec.steps[i] == 100 for i in range(300))


def test_failure_mid_run_raises_no_hang_and_dispatcher_is_reusable(pool):
    queues = ["0"] * 4 + ["1"] * 4
    state = {"armed": True}
    lock = threading.Lock()

    def fn(i, n):
        time.sleep(0.002)
        with lock:
            if i == 5 and state["armed"]:
                state["armed"] = False
                raise ValueError("boom replica 5")

    d = AdmissionDispatcher(pool, queues, limit=2, turn_steps=20)
    with pytest.raises(ValueError, match="boom replica 5"):
        run_with_timeout(d, fn, 100)
    rec = Recorder(queues)
    run_with_timeout(d, rec, 60)
    assert all(rec.steps[i] == 60 for i in range(8))


def test_drain_across_gpus_raise_waits_for_running_turns(pool):
    queues = ["0"] * 3 + ["1"] * 3 + ["2"] * 3 + ["3"] * 3
    lock = threading.Lock()
    starts, exits = [], []
    fail = {}
    # Replica 0 fails only after the first turn on every GPU (replicas 0, 3, 6, 9)
    # has started; otherwise its failure can land before run's initial submit
    # loop reaches 3/6/9 and they never start (measured flake: 1/100 unguarded).
    first_wave = threading.Barrier(4)

    def fn(i, n):
        t0 = time.monotonic()
        with lock:
            starts.append((i, t0))
            gate = i in (0, 3, 6, 9) and f"gated{i}" not in fail
            if gate:
                fail[f"gated{i}"] = True
        if gate:
            first_wave.wait(timeout=5)
        if i == 0:
            fail["t"] = time.monotonic()
            raise RuntimeError("replica 0 failed")
        time.sleep(0.2)
        with lock:
            exits.append(time.monotonic())

    d = AdmissionDispatcher(pool, queues, limit=1, turn_steps=10)
    with pytest.raises(RuntimeError, match="replica 0 failed"):
        run_with_timeout(d, fn, 100)
    raised_at = time.monotonic()
    assert len(exits) == 3, "the turns already running on GPUs 1-3 must finish"
    assert max(exits) <= raised_at
    late = [i for i, t in starts if t > fail["t"] + 0.05]
    assert late == [], f"turns started after the failure: {late}"


def test_two_near_simultaneous_failures_raise_exactly_one(pool):
    queues = ["0", "1", "2", "3"]
    barrier = threading.Barrier(2)

    def fn(i, n):
        if i in (1, 2):
            barrier.wait(timeout=5)
            raise ValueError(f"fail {i}")
        time.sleep(0.01)

    d = AdmissionDispatcher(pool, queues, limit=1, turn_steps=100)
    with pytest.raises(ValueError, match=r"fail [12]"):
        run_with_timeout(d, fn, 100)


def test_first_exception_wins_over_a_later_one(pool):
    queues = ["0", "1"]
    second_started = threading.Event()

    def fn(i, n):
        if i == 0:
            # Fail only once replica 1 is running, so its later failure is real.
            second_started.wait(timeout=5)
            raise ValueError("first failure")
        second_started.set()
        time.sleep(0.2)
        raise ValueError("later failure")

    d = AdmissionDispatcher(pool, queues, limit=1, turn_steps=100)
    with pytest.raises(ValueError, match="first failure"):
        run_with_timeout(d, fn, 100)


class SyncPool:
    """Runs every turn inside submit and returns an already-finished future,
    so every done-callback runs synchronously in the submitting thread."""

    def __init__(self):
        self.calls = []

    def submit(self, i, fn, *args):
        from concurrent.futures import Future

        self.calls.append(i)
        fut = Future()
        try:
            fut.set_result(fn(*args))
        except BaseException as exc:  # noqa: BLE001
            fut.set_exception(exc)
        return fut


def test_synchronous_callbacks_do_not_recurse():
    # 3000 one-step turns, every callback synchronous: without the trampoline
    # each turn nests several frames deeper, the RecursionError is swallowed
    # inside the callback by concurrent.futures, and run hangs.
    pool = SyncPool()
    d = AdmissionDispatcher(pool, ["0"], limit=1, turn_steps=1)
    steps = []
    run_with_timeout(d, lambda i, n: steps.append(n), 3000, timeout=20)
    assert sum(steps) == 3000


def test_popped_turns_are_not_started_after_a_failure():
    # run pops replicas 0, 1, 2 at once; replica 0 fails synchronously inside
    # submit, so 1 and 2 are still pending in the same trampoline and must be
    # skipped, never submitted.
    pool = SyncPool()
    d = AdmissionDispatcher(pool, ["0", "0", "0"], limit=3, turn_steps=10)

    def fn(i, n):
        if i == 0:
            raise ValueError("fail 0")

    with pytest.raises(ValueError, match="fail 0"):
        run_with_timeout(d, fn, 10)
    assert pool.calls == [0]


def test_internal_callback_error_raises_instead_of_hanging(pool, monkeypatch):
    queues = ["0"] * 4
    d = AdmissionDispatcher(pool, queues, limit=2, turn_steps=10)
    real = d._record_turn_locked
    state = {"armed": True}

    def flaky(i, n, exc):
        if state["armed"]:
            state["armed"] = False
            raise KeyError("injected bookkeeping error")
        return real(i, n, exc)

    monkeypatch.setattr(d, "_record_turn_locked", flaky)
    with pytest.raises(KeyError, match="injected bookkeeping error"):
        run_with_timeout(d, lambda i, n: time.sleep(0.001), 50)


class OwnerLock:
    """Plain-lock proxy that records which thread holds it."""

    def __init__(self):
        self._lock = threading.Lock()
        self.owner = None

    def acquire(self, *a, **k):
        got = self._lock.acquire(*a, **k)
        if got:
            self.owner = threading.get_ident()
        return got

    def release(self):
        self.owner = None
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()

    def held_by_me(self):
        return self.owner == threading.get_ident()


def test_lock_never_held_across_fn_submit_or_wait(pool, monkeypatch):
    queues = ["0"] * 3 + ["1"] * 3
    d = AdmissionDispatcher(pool, queues, limit=2, turn_steps=10)
    lk = OwnerLock()
    d._lock = lk
    violations = []

    class CheckingPool:
        def submit(self, i, fn, *args):
            if lk.held_by_me():
                violations.append("submit")
            return pool.submit(i, fn, *args)

    d._pool = CheckingPool()

    class CheckingEvent(threading.Event):
        def wait(self, timeout=None):
            if lk.held_by_me():
                violations.append("wait")
            return super().wait(timeout)

    monkeypatch.setattr(ra, "_Event", CheckingEvent)

    def fn(i, n):
        if lk.held_by_me():
            violations.append("fn")
        time.sleep(0.001)

    run_with_timeout(d, fn, 50)
    assert violations == []
    assert type(threading.Lock()) is type(d._new_lock())  # plain Lock, not RLock


def test_run_is_not_reentrant(pool):
    queues = ["0", "0"]
    d = AdmissionDispatcher(pool, queues, limit=1, turn_steps=10)

    def fn(i, n):
        d.run(lambda j, m: None, 10)

    with pytest.raises(RuntimeError, match="not re-entrant"):
        run_with_timeout(d, fn, 10)


def test_randomized_stress(pool):
    rng = random.Random(20260927)
    for _ in range(30):
        n_rep = rng.randint(20, 300)
        n_q = rng.randint(1, 4)
        queues = [str(rng.randrange(n_q)) for _ in range(n_rep)]
        limit = rng.randint(1, 12)
        turn = rng.randint(1, 80)
        nsteps = rng.randint(1, 300)
        instant = rng.random() < 0.3
        rec = Recorder(queues, sleep_fn=None if instant else (lambda: rng.random() * 0.002))
        d = AdmissionDispatcher(pool, queues, limit=limit, turn_steps=turn)
        run_with_timeout(d, rec, nsteps, timeout=60)
        assert all(rec.steps[i] == nsteps for i in range(n_rep))
        resident = defaultdict(int)
        for q in queues:
            resident[q] += 1
        assert all(rec.max_in_flight[q] <= min(limit, resident[q]) for q in resident)
        assert not rec.overlap_seen
        assert all(len(rec.threads[i]) == 1 for i in range(n_rep))


def test_invalid_limit_and_turn_rejected(pool):
    with pytest.raises(ValueError, match="limit"):
        AdmissionDispatcher(pool, ["0"], limit=0, turn_steps=10)
    with pytest.raises(ValueError, match="turn_steps"):
        AdmissionDispatcher(pool, ["0"], limit=1, turn_steps=0)
    with pytest.raises(ValueError, match="limit"):
        AdmissionDispatcher(pool, ["0"], limit=2.5, turn_steps=10)
    with pytest.raises(ValueError, match="limit"):
        AdmissionDispatcher(pool, ["0"], limit=True, turn_steps=10)
