# Replica Admission Cap and MPS Thread Share Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Opt-in cap on how many replicas per GPU step at once in production (short FIFO turns), plus an opt-in process-wide MPS active-thread percentage, with defaults that leave production on today's exact code path.

**Architecture:** A new pure-Python `AdmissionDispatcher` (`gareus/replica_admission.py`) advances every replica by exactly `nsteps` in turns of at most `turn_steps`, admitting at most `limit` turns per GPU queue at once, on the existing per-replica affinity pool. `step_all` in `run_gareus` calls one module-level `advance_replicas(pool, drivers, n, dispatcher)`, which with `dispatcher=None` is today's `pool.map`. A separate small module (`gareus/mps_share.py`) sets `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` right after `parse_args`, before any OpenMM import. Settings are recorded per job in `run_manifest.json` under one key, `method_settings["replica_admission"]`, plus an append-only history list.

**Tech Stack:** Python 3, `threading`, `concurrent.futures`, argparse + YAML config (`gareus/config.py`), OpenMM (tests only: Reference platform), pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-replica-admission-design.md` (read §3–§7 and §11 before starting any task).

## Global Constraints

- Defaults: `active_replicas_per_gpu: all`, `active_replica_turn_steps: 50`, `cuda_mps_active_thread_percentage: inherit`. With `all`, the dispatcher factory returns `None` and `advance_replicas` calls `pool.map` exactly as `step_all` does today — same code path, not a limit of "infinity".
- Validation rejects 0, negatives, non-integers, booleans, and percentages outside 1–100 with a `ValueError` whose message names the flag (dashes form, e.g. `--active-replicas-per-gpu`).
- The dispatcher lock is a plain `threading.Lock`, never an `RLock`, and is never held across `pool.submit`, `add_done_callback`, the turn function, or `Event.wait()`.
- `run` returns or raises only when nothing is in flight. First exception wins; later ones are logged.
- One manifest representation only: `method_settings["replica_admission"]`. No flat `active_replicas_per_gpu` / `active_replica_turn_steps` / `cuda_mps_active_thread_percentage` keys under `method_settings`.
- MPS: never silently override an inherited `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` with a different value (exit with an error naming both); refuse to set it if `openmm` is already in `sys.modules`.
- Recon, setup, US-pull and GaMD setup phases are not capped (only production `step_all`). Context reads (`_fetch_state`, `_fetch_exchange_state`) are not capped.
- Tests are targeted: run only the test files named in each task (user rule: never the full suite). If a hook blocks `pytest` in this shell, run the identical command through `opencode run "pytest -q <files>"`.
- Work on branch `feat/replica-admission` created from `docs/replica-admission-spec` (which holds the spec). Commit messages: conventional commits, ending with the session's attribution lines.
- Do not edit `smoke/config/chignolin_9.yaml` or any launcher in this change (spec §8, §9).

## Checked Facts and Deviations From the Spec

- **`method_settings` readers are all by explicit key** (checked 2026-09-27): the F04 kernel-identity guard (`production.py` ~629-645) reads `exchange_energy_version`/`cv_evaluator_version` only; `gareus/io.py` reads `temperature_k`; `gareus/mbar_analysis/thermo.py` reads `gamd_boost_type`; `plot_adaptive_diagnostics.py` reads `secondary_cv`. Nothing compares or hashes the whole dict, so a `replica_admission` record that differs per job (queue keys, inherited env) cannot trip a resume guard. Any future whole-dict comparison must exclude `replica_admission`.
- **History granularity is per phase directory, not per job.** `record_replica_admission` runs once per `run_gareus` call, i.e. once per phase output directory (baseline, each top-up, each epoch). A job spanning two phases writes one entry into each phase's own manifest. Spec §3 is updated to say so.
- **`resolved_args` still holds the flat values** of the first job, like every other argument (a first-job snapshot). Readers must use `method_settings["replica_admission"]`, never `resolved_args`, for these settings.
- **No `gareus/helptext.py` entry** (spec §8 listed one): helptext has no performance/MPS topic, and `-hh` already prints the full argparse flag list, so the three flags' help strings (Task 3) carry the documentation.
- **Drivers test uses tiny 4-particle Reference systems** (the `tests/test_npt_driver_scheduling.py` pattern) instead of `tests/pep_gamd_fixture.build_small_simulation` (spec §7): same driver code path, seconds instead of minutes, and no solvated-system build per replica.

## Verification Record (2026-09-27, before execution)

The plan's code was applied verbatim to a scratch worktree and run; it was not taken on trust.

- **Targeted tests:** 84 passed, 0 failed, 6.7 s (the three new files plus `tests/test_replica_affinity_executor.py`, `tests/test_npt_driver_scheduling.py`, `tests/test_npt_cli_args.py`).
- **Mutation battery** (24 injected bugs, each must fail its task's tests): first pass 18 killed, 6 survived. Three survivors were real test gaps and are now closed by the tests above: no trampoline (M02, `test_synchronous_callbacks_do_not_recurse`), last exception wins (M11, `test_first_exception_wins_over_a_later_one`), popped turns started after a failure (M13, `test_popped_turns_are_not_started_after_a_failure`). Three are equivalent mutants with no observable effect: `_pop_launches_locked`'s `failed` early return (M04) and `_record_turn_locked`'s `failed` no-requeue (M10) are redundant layers, since `_record_failure_locked` already clears every queue; and changing the driver's `ctrl_due <= end` to `<` (M16) still services the move at `end`, because the service check (`next_due_step <= now`) is independent of the subdivision target. Two mutants (submit while holding the lock, and removing the re-entry guard) are killed only by a hang, not an assertion: the module-scoped pool's `shutdown(wait=True)` blocks on the deadlocked worker. That is acceptable for a mutant, but a real regression of that kind shows up as a hung test run.
- **Drain-test flake:** 1 in 100 without the barrier gate, reproduced with the exact predicted message `the turns already running on GPUs 1-3 must finish`.
- **Re-check after the fixes:** M02, M11 and M13 are now killed (so 21 of 24 mutants are killed, and the remaining 3 are the equivalent mutants above). The gated drain test failed 0 of 100 runs, and the Task 1 file passes 20 of 20.
- **Small board** on this plan: ACCEPT-WITH-CHANGES (kimi 80, thinker 85; mini REJECT 92). The conditions are resolved as follows:
  - Drain race: fixed.
  - `update_run_manifest` creating a missing manifest and returning its payload: checked in `gareus/provenance.py` and pinned by `test_record_without_existing_manifest_creates_one`.
  - The "reached" stress assertion: moved to a deterministic test (see the notes above).
  - Float truncation in the constructor: now rejected (`_positive_int`).
  - Popped-before-failure window: spec §4 wording amended.
  - Driver, reporter and YAML protocols: confirmed by the passing tests.
  - mini's dissent (a trampoline race that loses pending launches) is not adopted. `_tls` is per thread, so a callback on another thread starts its own loop. A same-thread callback runs only inside `add_done_callback`, inside the loop body, and the loop re-checks `pending` afterwards. The 30-iteration stress test and the synchronous-pool test both pass.

## Review Focus

1. YAML `active_replicas_per_gpu: true` (YAML boolean, which is an `int` subclass in Python) must be rejected, not read as 1 — pinned in Task 3.
2. A resume whose previous job recorded a different queue set (e.g. `{"shared": 4}` then `{"0": 59, "1": 59}`) must not leave the old queue key inside `effective_per_queue` — `update_run_manifest`'s `_deep_update` merges nested dicts, so the record must be replaced wholesale — pinned in Task 3.
3. Many instant turns (a turn function that returns immediately, so futures finish before `add_done_callback`) must not recurse once per turn and hit `RecursionError` — pinned in Task 1 (`test_instant_turns_many_replicas_no_recursion`).
4. `active_replica_turn_steps` larger than every `nsteps` in the run (e.g. 1000 with 400-step boundaries) must behave as one turn per replica per call, not stall — pinned in Task 1.
5. An inherited `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` equal to the requested value (e.g. `"25"` inherited, `25` requested, or `" 25 "`) must be accepted, not reported as a conflict — pinned in Task 4.

---

## File Structure

- Create `gareus/replica_admission.py` — `AdmissionDispatcher`, `advance_replicas`, `make_dispatcher`, `queue_keys_from_platform_props`, `effective_limits`, and the two value parsers `parse_active_replicas_per_gpu`, `parse_mps_thread_percentage`. Pure Python; imports no OpenMM.
- Create `gareus/mps_share.py` — `apply_mps_thread_percentage(args, environ=None, modules=None)`. Pure Python.
- Modify `gareus/cli.py` — three flags in `_add_platform_args`, `_validate_replica_admission_args` called from `parse_args`, MPS apply in `main`.
- Modify `gareus/provenance.py` — `admission_manifest_record`, `record_replica_admission`, `_method_settings` emits the record.
- Modify `gareus/production.py` — collect per-replica platform props, build the dispatcher after replica construction, record it, and route `step_all` through `advance_replicas`.
- Create tests: `tests/test_replica_admission.py`, `tests/test_replica_admission_drivers.py`, `tests/test_replica_admission_config.py`.
- Modify `CLAUDE.md` — one short handoff section (Task 5).

---

### Task 1: `AdmissionDispatcher`

**Files:**
- Create: `gareus/replica_admission.py`
- Test: `tests/test_replica_admission.py`

**Interfaces:**
- Consumes: a pool object with `submit(replica_index, fn, *args) -> concurrent.futures.Future` (production's `_ReplicaAffinityExecutor` in `gareus/production.py:155-188`; tests use the same class).
- Produces:
  - `SHARED_QUEUE_KEY: str = "shared"`
  - `class AdmissionDispatcher(pool, gpu_of_replica: Sequence[str], limit: int, turn_steps: int)`
    - attribute `limit_per_queue: dict[str, int]` (limit clipped to each queue's resident count)
    - attribute `turn_steps: int`
    - method `run(fn: Callable[[int, int], None], nsteps: int) -> None`
  - module attribute `_Event = threading.Event` (tests monkeypatch it)

- [ ] **Step 1: Create the branch**

```bash
git checkout docs/replica-admission-spec
git checkout -b feat/replica-admission
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_replica_admission.py`:

```python
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
```

Notes on these tests:
- `test_randomized_stress`: the `rng` used by `sleep_fn` is called from worker threads; `random.Random.random()` is GIL-atomic, and determinism of the sleeps is not asserted. It asserts `max_in_flight <= min(limit, resident)` only; the spec's "and is reached when there is enough work" is pinned deterministically by `test_in_flight_never_exceeds_limit_and_reaches_it` instead (a "reached" check under random 0–2 ms sleeps would flake).
- `SyncPool` runs each turn inside `submit` and returns an already-finished future, so every done-callback runs synchronously in the submitting thread. That is the only way to pin the trampoline (`test_synchronous_callbacks_do_not_recurse`) and the skip-after-failure check (`test_popped_turns_are_not_started_after_a_failure`) deterministically; with a real thread pool both paths are timing-dependent and the mutation battery showed the real-pool tests do not catch their removal.
- `test_drain_across_gpus_raise_waits_for_running_turns` gates replica 0's failure on a `Barrier` with the first turns of the other three GPUs. Without it the failure can land before `run`'s initial submit loop reaches them (measured flake: 1 in 100).

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest -q tests/test_replica_admission.py`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'gareus.replica_admission'`.

- [ ] **Step 4: Implement the dispatcher**

Create `gareus/replica_admission.py`:

```python
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
from typing import Callable, Optional, Sequence

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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest -q tests/test_replica_admission.py`
Expected: all pass (20 tests, under ~30 s).

If `test_lock_never_held_across_fn_submit_or_wait` fails on the last line, the check is that `_new_lock()` returns a plain `threading.Lock` — fix the implementation, not the test.

- [ ] **Step 6: Commit**

```bash
git add gareus/replica_admission.py tests/test_replica_admission.py
git commit -m "feat: AdmissionDispatcher for capped per-GPU replica turns"
```

---

### Task 2: `advance_replicas`, factory helpers, driver equivalence

**Files:**
- Modify: `gareus/replica_admission.py` (append)
- Test: `tests/test_replica_admission_drivers.py`

**Interfaces:**
- Consumes: `AdmissionDispatcher`, `SHARED_QUEUE_KEY` (Task 1); `gareus.npt_driver.ReplicaStepDriver(sim, controller=None, on_volume_move=None, label="")` with `advance(nsteps)` and `register_reporter(reporter)` (`gareus/npt_driver.py:127-270`).
- Produces:
  - `advance_replicas(pool, drivers: Sequence, nsteps: int, dispatcher: Optional[AdmissionDispatcher]) -> None`
  - `make_dispatcher(pool, queue_keys: Sequence[str], active_replicas_per_gpu, turn_steps: int) -> Optional[AdmissionDispatcher]` — `None` when `active_replicas_per_gpu == "all"`.
  - `queue_keys_from_platform_props(props_list: Sequence[Mapping[str, str]]) -> list[str]` — `DeviceIndex` string, else `SHARED_QUEUE_KEY`.
  - `effective_limits(queue_keys: Sequence[str], active_replicas_per_gpu) -> dict[str, int]` — per queue `min(limit, resident)`, or resident count for `"all"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_replica_admission_drivers.py`:

```python
"""advance_replicas over real ReplicaStepDrivers: turns visit the same events.

Spec §6/§7: splitting a replica's advance(n) into capped turns must give the
same step counts and the same volume-move and report steps, in the same order,
as today's one-call-per-replica pool.map. Tiny 4-particle Reference systems
(the test_npt_driver_scheduling pattern) keep this CPU-only and fast.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import LangevinMiddleIntegrator, Platform, System, Vec3, unit  # noqa: E402
from openmm.app import Simulation, Topology  # noqa: E402
from openmm.app.element import Element  # noqa: E402

from gareus.npt_driver import ReplicaStepDriver  # noqa: E402
from gareus.production import _ReplicaAffinityExecutor  # noqa: E402
from gareus.replica_admission import (  # noqa: E402
    SHARED_QUEUE_KEY,
    AdmissionDispatcher,
    advance_replicas,
    effective_limits,
    make_dispatcher,
    queue_keys_from_platform_props,
)

BOUNDARIES = (400, 250, 150, 400)


def _tiny_sim(seed):
    system = System()
    for _ in range(4):
        system.addParticle(12.0)
    bond = openmm.HarmonicBondForce()
    for i in range(3):
        bond.addBond(i, i + 1, 0.15, 500.0)
    system.addForce(bond)
    system.setDefaultPeriodicBoxVectors(Vec3(2.5, 0, 0), Vec3(0, 2.5, 0), Vec3(0, 0, 2.5))
    integ = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 1.0 * unit.femtoseconds)
    integ.setRandomNumberSeed(seed)
    top = Topology()
    res = top.addResidue("LIG", top.addChain())
    for _ in range(4):
        top.addAtom("C", Element.getByAtomicNumber(6), res)
    sim = Simulation(top, system, integ, Platform.getPlatformByName("Reference"))
    sim.context.setPositions([[0, 0, 0], [0.15, 0, 0], [0.3, 0, 0], [0.45, 0, 0]])
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, seed)
    return sim


class RecordingController:
    """Volume-move stand-in: records the local step of every attempt."""

    def __init__(self, frequency, journal):
        self.frequency = int(frequency)
        self.next_due_step = self.frequency
        self.journal = journal

    def attempt_due(self, now):
        self.journal.append(("volume", int(now)))
        self.next_due_step = int(now) + self.frequency
        return None


class RecordingReporter:
    """Legacy 5-tuple reporter firing every `interval` steps; records the step."""

    def __init__(self, interval, journal):
        self.interval = int(interval)
        self.journal = journal

    def describeNextReport(self, sim):
        steps = self.interval - int(sim.currentStep) % self.interval
        return (steps, False, False, False, False)

    def report(self, sim, state):
        self.journal.append(("report", int(sim.currentStep)))


def _build(n_rep, controller_freq=100, report_interval=250):
    drivers, journals = [], []
    for i in range(n_rep):
        journal = []
        sim = _tiny_sim(seed=100 + i)
        driver = ReplicaStepDriver(sim, controller=RecordingController(controller_freq, journal), label=f"r{i}")
        driver.register_reporter(RecordingReporter(report_interval, journal))
        drivers.append(driver)
        journals.append(journal)
    return drivers, journals


def _run(n_rep, queue_keys, dispatcher_args, **build_kw):
    pool = _ReplicaAffinityExecutor(n_rep)
    try:
        drivers, journals = _build(n_rep, **build_kw)
        dispatcher = None
        if dispatcher_args is not None:
            limit, turn = dispatcher_args
            dispatcher = AdmissionDispatcher(pool, queue_keys, limit=limit, turn_steps=turn)
        for n in BOUNDARIES:
            advance_replicas(pool, drivers, n, dispatcher)
        steps = [d.current_step for d in drivers]
        energies = [
            d.sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            for d in drivers
        ]
        return steps, journals, energies
    finally:
        pool.shutdown(wait=True)


@pytest.mark.parametrize(
    "n_rep,queue_keys,dispatcher_args",
    [
        (4, ["0", "0", "1", "1"], (1, 50)),
        (4, ["0", "0", "1", "1"], (1, 30)),
        (8, ["0"] * 4 + ["1"] * 4, (2, 50)),  # cap below resident: 2 of 4 wait per queue
        (4, ["0", "0", "1", "1"], (8, 50)),   # cap >= resident
    ],
)
def test_turns_visit_same_events_as_one_call(n_rep, queue_keys, dispatcher_args):
    ref_steps, ref_journals, _ = _run(n_rep, queue_keys, None)
    steps, journals, energies = _run(n_rep, queue_keys, dispatcher_args)
    assert steps == ref_steps == [sum(BOUNDARIES)] * n_rep
    assert journals == ref_journals
    assert all(np.isfinite(e) for e in energies)


def test_deadline_exactly_on_turn_end_is_serviced_once():
    # controller every 50 steps, turns of 50: every deadline lands on a turn end.
    _, ref_journals, _ = _run(2, ["0", "0"], None, controller_freq=50, report_interval=50)
    _, journals, _ = _run(2, ["0", "0"], (1, 50), controller_freq=50, report_interval=50)
    assert journals == ref_journals
    volume_steps = [s for kind, s in journals[0] if kind == "volume"]
    assert volume_steps == list(range(50, sum(BOUNDARIES) + 1, 50))
    assert len(volume_steps) == len(set(volume_steps))


class RecordingPool:
    def __init__(self):
        self.map_calls = []
        self.submit_calls = 0

    def map(self, fn, items):
        items = list(items)
        self.map_calls.append(items)
        return [fn(item) for item in items]

    def submit(self, *a, **k):
        self.submit_calls += 1
        raise AssertionError("submit must not be used on the default path")


class CountingDriver:
    def __init__(self):
        self.calls = []

    def advance(self, n):
        self.calls.append(int(n))


def test_defaults_use_single_pool_map_and_no_dispatcher():
    pool = RecordingPool()
    assert make_dispatcher(pool, ["0", "0"], "all", 50) is None
    drivers = [CountingDriver(), CountingDriver()]
    advance_replicas(pool, drivers, 400, None)
    assert len(pool.map_calls) == 1
    assert [item[2] for item in pool.map_calls[0]] == [400, 400]
    assert pool.submit_calls == 0
    assert [d.calls for d in drivers] == [[400], [400]]


def test_make_dispatcher_with_integer_cap():
    pool = _ReplicaAffinityExecutor(3)
    try:
        d = make_dispatcher(pool, ["0", "0", "1"], 8, 50)
        assert isinstance(d, AdmissionDispatcher)
        assert d.limit_per_queue == {"0": 2, "1": 1}
        assert d.turn_steps == 50
    finally:
        pool.shutdown(wait=True)


def test_queue_keys_from_platform_props():
    props = [{"DeviceIndex": "0", "Precision": "mixed"}, {"DeviceIndex": "1"}, {}, {"DeviceIndex": "0,1,2,3"}]
    assert queue_keys_from_platform_props(props) == ["0", "1", SHARED_QUEUE_KEY, "0,1,2,3"]


def test_effective_limits():
    keys = ["0"] * 59 + ["1"] * 58
    assert effective_limits(keys, 8) == {"0": 8, "1": 8}
    assert effective_limits(keys, "all") == {"0": 59, "1": 58}
    assert effective_limits(["0", "0"], 8) == {"0": 2}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest -q tests/test_replica_admission_drivers.py`
Expected: FAIL at import with `ImportError: cannot import name 'advance_replicas'`.

- [ ] **Step 3: Implement the helpers**

Append to `gareus/replica_admission.py` (and add `Mapping` to the `typing` import line: `from typing import Callable, Mapping, Optional, Sequence`):

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest -q tests/test_replica_admission_drivers.py tests/test_replica_admission.py`
Expected: all pass.

If `test_turns_visit_same_events_as_one_call` fails with differing journals, do not loosen the comparison: the spec's §6 invariant is exactly this equality. Print both journals for the first differing replica and check `ReplicaStepDriver.advance`'s `due <= end` handling.

- [ ] **Step 5: Commit**

```bash
git add gareus/replica_admission.py tests/test_replica_admission_drivers.py
git commit -m "feat: advance_replicas and admission helpers with driver-equivalence tests"
```

---

### Task 3: CLI flags, validation, manifest record

**Files:**
- Modify: `gareus/replica_admission.py` (append the two parsers)
- Modify: `gareus/cli.py` (`_add_platform_args` near line 879; new `_validate_replica_admission_args` next to `_validate_npt_args` near line 1088; call it in `parse_args` after `_validate_npt_args(args)` near line 1693)
- Modify: `gareus/provenance.py` (`_method_settings` near line 289; new functions after `update_run_manifest` near line 638)
- Test: `tests/test_replica_admission_config.py`

**Interfaces:**
- Consumes: `gareus.provenance._read_manifest`, `_write_manifest`, `_utc_now`, `update_run_manifest`, `initialize_run_manifest` (existing).
- Produces:
  - `parse_active_replicas_per_gpu(value) -> "all" | int`
  - `parse_mps_thread_percentage(value) -> "inherit" | int`
  - after `parse_args`: `args.active_replicas_per_gpu` is `"all"` or `int`; `args.active_replica_turn_steps` is `int`; `args.cuda_mps_active_thread_percentage` is `"inherit"` or `int`
  - `admission_manifest_record(args, effective_per_queue=None) -> dict` with keys `active_replicas_per_gpu`, `active_replica_turn_steps`, `cuda_mps_active_thread_percentage` (`{"requested": ..., "inherited_env": ...}`), `effective_per_queue`
  - `record_replica_admission(out_dir, args, effective_per_queue) -> dict` (returns the written payload)
  - reads `args._cuda_mps_inherited_env` if Task 4 set it (absent → the current environment value), `"unset"` when the variable is not set

- [ ] **Step 1: Write the failing tests**

Create `tests/test_replica_admission_config.py`:

```python
"""Replica-admission settings: parsing, validation, one manifest representation.

Spec §3, §5, §7. The MPS tests are added in Task 4.
"""

from __future__ import annotations

import json

import pytest

from gareus.cli import parse_args
from gareus.provenance import (
    _method_settings,
    admission_manifest_record,
    initialize_run_manifest,
    record_replica_admission,
)

MINIMAL = ["--seq", "AA", "--out", "unused"]
FLAT_KEYS = ("active_replicas_per_gpu", "active_replica_turn_steps", "cuda_mps_active_thread_percentage")


def test_defaults():
    args = parse_args(MINIMAL)
    assert args.active_replicas_per_gpu == "all"
    assert args.active_replica_turn_steps == 50
    assert args.cuda_mps_active_thread_percentage == "inherit"


def test_cli_values():
    args = parse_args(MINIMAL + ["--active-replicas-per-gpu", "8", "--active-replica-turn-steps", "30",
                                 "--cuda-mps-active-thread-percentage", "25"])
    assert args.active_replicas_per_gpu == 8
    assert args.active_replica_turn_steps == 30
    assert args.cuda_mps_active_thread_percentage == 25


def test_yaml_values(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("active_replicas_per_gpu: 8\nactive_replica_turn_steps: 50\n"
                   "cuda_mps_active_thread_percentage: 25\n")
    args = parse_args(MINIMAL + ["--config", str(cfg)])
    assert args.active_replicas_per_gpu == 8
    assert args.active_replica_turn_steps == 50
    assert args.cuda_mps_active_thread_percentage == 25


def test_yaml_all_and_inherit_words(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("active_replicas_per_gpu: ALL\ncuda_mps_active_thread_percentage: Inherit\n")
    args = parse_args(MINIMAL + ["--config", str(cfg)])
    assert args.active_replicas_per_gpu == "all"
    assert args.cuda_mps_active_thread_percentage == "inherit"


@pytest.mark.parametrize("bad", ["0", "-1", "2.5", "eight", ""])
def test_bad_cap_rejected(bad):
    with pytest.raises(ValueError, match="--active-replicas-per-gpu"):
        parse_args(MINIMAL + ["--active-replicas-per-gpu", bad])


def test_yaml_boolean_cap_rejected(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("active_replicas_per_gpu: true\n")
    with pytest.raises(ValueError, match="--active-replicas-per-gpu"):
        parse_args(MINIMAL + ["--config", str(cfg)])


@pytest.mark.parametrize("bad", ["0", "-5"])
def test_bad_turn_steps_rejected(bad):
    with pytest.raises(ValueError, match="--active-replica-turn-steps"):
        parse_args(MINIMAL + ["--active-replica-turn-steps", bad])


@pytest.mark.parametrize("bad", ["0", "101", "-1", "25.5", "half"])
def test_bad_percentage_rejected(bad):
    with pytest.raises(ValueError, match="--cuda-mps-active-thread-percentage"):
        parse_args(MINIMAL + ["--cuda-mps-active-thread-percentage", bad])


def test_method_settings_has_nested_record_and_no_flat_keys():
    args = parse_args(MINIMAL + ["--active-replicas-per-gpu", "8"])
    settings = _method_settings(args)
    for key in FLAT_KEYS:
        assert key not in settings
    rec = settings["replica_admission"]
    assert rec["active_replicas_per_gpu"] == 8
    assert rec["active_replica_turn_steps"] == 50
    assert rec["effective_per_queue"] is None
    assert set(rec["cuda_mps_active_thread_percentage"]) == {"requested", "inherited_env"}
    json.dumps(settings)


def test_record_on_fresh_and_resumed_complete_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "111")
    job1 = parse_args(MINIMAL)
    initialize_run_manifest(job1, tmp_path, argv=[])
    record_replica_admission(tmp_path, job1, {"shared": 4})

    # Second job: a resume against the now-complete manifest, cap switched on,
    # queues moved to real GPUs. The old "shared" key must not survive.
    monkeypatch.setenv("SLURM_JOB_ID", "222")
    job2 = parse_args(MINIMAL + ["--active-replicas-per-gpu", "8"])
    record_replica_admission(tmp_path, job2, {"0": 8, "1": 8})

    payload = json.loads((tmp_path / "run_manifest.json").read_text())
    rec = payload["method_settings"]["replica_admission"]
    assert rec["active_replicas_per_gpu"] == 8
    assert rec["effective_per_queue"] == {"0": 8, "1": 8}
    for key in FLAT_KEYS:
        assert key not in payload["method_settings"]
    hist = payload["replica_admission_history"]
    assert [h["slurm_job_id"] for h in hist] == ["111", "222"]
    assert hist[0]["active_replicas_per_gpu"] == "all"
    assert hist[0]["effective_per_queue"] == {"shared": 4}
    assert all("recorded_utc" in h for h in hist)


def test_record_without_existing_manifest_creates_one(tmp_path):
    args = parse_args(MINIMAL)
    record_replica_admission(tmp_path, args, {"shared": 2})
    payload = json.loads((tmp_path / "run_manifest.json").read_text())
    assert payload["method_settings"]["replica_admission"]["effective_per_queue"] == {"shared": 2}
    assert len(payload["replica_admission_history"]) == 1


def test_record_reads_inherited_env_captured_by_mps_apply():
    args = parse_args(MINIMAL)
    args._cuda_mps_inherited_env = None
    rec = admission_manifest_record(args)
    assert rec["cuda_mps_active_thread_percentage"] == {"requested": "inherit", "inherited_env": "unset"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest -q tests/test_replica_admission_config.py`
Expected: FAIL at import with `ImportError: cannot import name 'admission_manifest_record'`.

- [ ] **Step 3: Add the value parsers**

Append to `gareus/replica_admission.py`:

```python
# ------------------------------------------------------------ value parsing


def _int_or_word(value, *, word: str, flag: str, lo: int, hi: Optional[int]):
    """``word`` (case-insensitive) or an integer in [lo, hi]; booleans are rejected."""
    if isinstance(value, bool):
        raise ValueError(f"{flag} must be '{word}' or an integer, got the boolean {value!r}")
    if isinstance(value, str):
        text = value.strip()
        if text.lower() == word:
            return word
        try:
            number = int(text)
        except ValueError:
            raise ValueError(f"{flag} must be '{word}' or an integer, got {value!r}") from None
    elif isinstance(value, int):
        number = int(value)
    else:
        raise ValueError(f"{flag} must be '{word}' or an integer, got {value!r}")
    if number < lo or (hi is not None and number > hi):
        bound = f">= {lo}" if hi is None else f"in {lo}-{hi}"
        raise ValueError(f"{flag} must be '{word}' or an integer {bound}, got {value!r}")
    return number


def parse_active_replicas_per_gpu(value):
    """``"all"`` or an integer >= 1."""
    return _int_or_word(value, word="all", flag="--active-replicas-per-gpu", lo=1, hi=None)


def parse_mps_thread_percentage(value):
    """``"inherit"`` or an integer 1-100."""
    return _int_or_word(value, word="inherit", flag="--cuda-mps-active-thread-percentage", lo=1, hi=100)
```

- [ ] **Step 4: Add the flags and the validator in `gareus/cli.py`**

In `_add_platform_args`, directly after the `--cuda-mps` argument, add:

```python
    p.add_argument("--active-replicas-per-gpu", default="all",
                   help="Production stepping: at most this many replicas per GPU advance at once, in "
                        "FIFO turns ('all' = every replica at once, today's behaviour). Measured best "
                        "with MPS at 59 contexts/GPU: 6-8. Spec 2026-09-26-replica-admission-design.")
    p.add_argument("--active-replica-turn-steps", type=int, default=50,
                   help="Steps a replica runs per admitted turn before rejoining its GPU's queue. "
                        "Ignored when --active-replicas-per-gpu is 'all'.")
    p.add_argument("--cuda-mps-active-thread-percentage", default="inherit",
                   help="Set CUDA_MPS_ACTIVE_THREAD_PERCENTAGE for this process before any CUDA "
                        "context exists ('inherit' = leave the environment alone). Errors if the "
                        "environment already holds a different value. Applies to every phase.")
```

Next to `_validate_npt_args`, add:

```python
def _validate_replica_admission_args(args: argparse.Namespace) -> None:
    """Normalise the replica-admission settings (spec 2026-09-26 §3); raise ValueError if invalid."""
    from .replica_admission import parse_active_replicas_per_gpu, parse_mps_thread_percentage

    args.active_replicas_per_gpu = parse_active_replicas_per_gpu(getattr(args, "active_replicas_per_gpu", "all"))
    turn = getattr(args, "active_replica_turn_steps", 50)
    if isinstance(turn, bool) or not isinstance(turn, int) or int(turn) < 1:
        raise ValueError(f"--active-replica-turn-steps must be an integer >= 1, got {turn!r}")
    args.active_replica_turn_steps = int(turn)
    args.cuda_mps_active_thread_percentage = parse_mps_thread_percentage(
        getattr(args, "cuda_mps_active_thread_percentage", "inherit"))
```

In `parse_args`, after `_validate_npt_args(args)`, add `_validate_replica_admission_args(args)`.

Note: `--active-replica-turn-steps` has `type=int`, so a CLI value like `2.5` is rejected by argparse itself (`SystemExit`), and a YAML value reaches the validator as whatever YAML typed it; the validator's `isinstance(turn, int)` check covers YAML floats and booleans.

- [ ] **Step 5: Add the manifest record in `gareus/provenance.py`**

At the end of `_method_settings`, before its `return settings`, add:

```python
    settings["replica_admission"] = admission_manifest_record(args)
```

After `update_run_manifest`, add:

```python
def admission_manifest_record(args: Any, effective_per_queue: Optional[Mapping[str, int]] = None) -> dict[str, Any]:
    """The one manifest representation of the replica-admission settings (spec 2026-09-26 §3).

    Stored as ``method_settings["replica_admission"]``; never also as flat keys,
    which would go stale on a resume (a resume leaves a complete manifest's
    method_settings untouched -- see ensure_run_manifest_initialized).
    """
    if hasattr(args, "_cuda_mps_inherited_env"):
        inherited = getattr(args, "_cuda_mps_inherited_env")
    else:
        inherited = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
    return {
        "active_replicas_per_gpu": getattr(args, "active_replicas_per_gpu", "all"),
        "active_replica_turn_steps": int(getattr(args, "active_replica_turn_steps", 50)),
        "cuda_mps_active_thread_percentage": {
            "requested": getattr(args, "cuda_mps_active_thread_percentage", "inherit"),
            "inherited_env": inherited if inherited is not None else "unset",
        },
        "effective_per_queue": dict(effective_per_queue) if effective_per_queue is not None else None,
    }


def record_replica_admission(out_dir: Path, args: Any, effective_per_queue: Mapping[str, int]) -> dict[str, Any]:
    """Record this job's admission settings; runs on every job, fresh or resumed.

    Replaces ``method_settings["replica_admission"]`` wholesale (``_deep_update``
    would merge nested dicts and keep a previous job's queue keys) and appends
    one ``replica_admission_history`` entry. One job process is the manifest's
    only writer while it runs, so read-modify-write needs no file lock.
    """
    out_dir = Path(out_dir)
    record = admission_manifest_record(args, effective_per_queue)
    payload = _read_manifest(out_dir) or update_run_manifest(out_dir, {})
    method_settings = payload.get("method_settings")
    if not isinstance(method_settings, dict):
        method_settings = {}
    history = payload.get("replica_admission_history")
    history = list(history) if isinstance(history, list) else []
    entry = {"recorded_utc": _utc_now(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"), **record}
    payload = {
        **payload,
        "method_settings": {**method_settings, "replica_admission": record},
        "replica_admission_history": history + [entry],
        "last_updated_utc": _utc_now(),
    }
    _write_manifest(out_dir, payload)
    return payload
```

Check the top of `gareus/provenance.py` already imports `os`, `Path`, `Any`, `Mapping`, `Optional`; add any that are missing to the existing import lines.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest -q tests/test_replica_admission_config.py tests/test_npt_cli_args.py`
Expected: all pass (`test_npt_cli_args.py` guards that the new validator does not disturb existing parsing).

- [ ] **Step 7: Commit**

```bash
git add gareus/replica_admission.py gareus/cli.py gareus/provenance.py tests/test_replica_admission_config.py
git commit -m "feat: replica-admission and MPS-share settings with per-job manifest record"
```

---

### Task 4: MPS thread-share apply

**Files:**
- Create: `gareus/mps_share.py`
- Modify: `gareus/cli.py` (`main`, line ~1983, right after `args = parse_args(argv_list)`)
- Test: `tests/test_replica_admission_config.py` (append)

**Interfaces:**
- Consumes: `args.cuda_mps_active_thread_percentage` (`"inherit"` or `int`, Task 3).
- Produces: `apply_mps_thread_percentage(args, environ=None, modules=None) -> None`; sets `args._cuda_mps_inherited_env` (the inherited value or `None`), read by `admission_manifest_record` (Task 3).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_replica_admission_config.py`:

```python
# ------------------------------------------------------------------ MPS share

import subprocess  # noqa: E402
import sys  # noqa: E402
import textwrap  # noqa: E402

from gareus.mps_share import ENV_VAR, apply_mps_thread_percentage  # noqa: E402


def _args(pct):
    return parse_args(MINIMAL + ["--cuda-mps-active-thread-percentage", str(pct)])


def test_inherit_records_and_leaves_env_alone():
    env = {ENV_VAR: "40"}
    args = parse_args(MINIMAL)
    apply_mps_thread_percentage(args, environ=env, modules={})
    assert env == {ENV_VAR: "40"}
    assert args._cuda_mps_inherited_env == "40"


def test_sets_env_when_unset():
    env = {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/pipe"}
    args = _args(25)
    apply_mps_thread_percentage(args, environ=env, modules={})
    assert env[ENV_VAR] == "25"
    assert args._cuda_mps_inherited_env is None


@pytest.mark.parametrize("inherited", ["25", " 25 "])
def test_equal_inherited_value_is_accepted(inherited):
    env = {ENV_VAR: inherited, "CUDA_MPS_PIPE_DIRECTORY": "/tmp/pipe"}
    apply_mps_thread_percentage(_args(25), environ=env, modules={})
    assert env[ENV_VAR].strip() == "25"


def test_conflicting_inherited_value_exits_naming_both():
    env = {ENV_VAR: "40"}
    with pytest.raises(SystemExit, match=r"40.*25|25.*40"):
        apply_mps_thread_percentage(_args(25), environ=env, modules={})
    assert env[ENV_VAR] == "40"


def test_openmm_already_imported_exits():
    with pytest.raises(SystemExit, match="openmm"):
        apply_mps_thread_percentage(_args(25), environ={}, modules={"openmm": object()})


def test_warns_when_mps_not_running(capsys):
    env = {}
    apply_mps_thread_percentage(_args(25), environ=env, modules={})
    assert "CUDA_MPS_PIPE_DIRECTORY" in capsys.readouterr().err
    assert env[ENV_VAR] == "25"


def test_real_entry_chain_loads_no_openmm_and_sets_env():
    script = textwrap.dedent(
        """
        import os, sys
        os.environ.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/unused-pipe"
        import gareus.core
        import gareus.cli
        from gareus.mps_share import apply_mps_thread_percentage
        args = gareus.cli.parse_args(["--seq", "AA", "--out", "unused",
                                      "--cuda-mps-active-thread-percentage", "25"])
        apply_mps_thread_percentage(args)
        print("openmm" in sys.modules, "gamd" in sys.modules,
              os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"))
        """
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=True)
    assert out.stdout.split()[-3:] == ["False", "False", "25"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest -q tests/test_replica_admission_config.py`
Expected: FAIL at import with `ModuleNotFoundError: No module named 'gareus.mps_share'`.

- [ ] **Step 3: Implement `gareus/mps_share.py`**

```python
"""Process-wide CUDA MPS active-thread percentage (spec 2026-09-26 §5).

Applied once, right after parse_args, before any OpenMM platform exists: the
MPS client reads CUDA_MPS_ACTIVE_THREAD_PERCENTAGE when CUDA first
initialises, and setup, US-pull and production all run in this one process.
Importing gareus.core/gareus.cli and running parse_args loads no openmm
(pinned by tests/test_replica_admission_config.py), so the variable is set in
time. The openmm guard cannot see CUDA initialised by a non-OpenMM import
(none exists today). MPS reports no effective share back, so the manifest
records the value as "requested".
"""

from __future__ import annotations

import os
import sys
from typing import Any, MutableMapping, Optional

ENV_VAR = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"


def apply_mps_thread_percentage(
    args: Any,
    environ: Optional[MutableMapping[str, str]] = None,
    modules: Optional[dict] = None,
) -> None:
    """Apply ``args.cuda_mps_active_thread_percentage``; record the inherited value on ``args``."""
    env = os.environ if environ is None else environ
    mods = sys.modules if modules is None else modules
    requested = getattr(args, "cuda_mps_active_thread_percentage", "inherit")
    inherited = env.get(ENV_VAR)
    args._cuda_mps_inherited_env = inherited
    if requested == "inherit":
        return
    if "openmm" in mods:
        raise SystemExit(
            f"[mps] --cuda-mps-active-thread-percentage {requested} requested, but openmm is already "
            "imported in this process; CUDA may already have read the MPS environment. Refusing to set "
            f"{ENV_VAR} too late."
        )
    if inherited is not None and inherited.strip() != str(requested):
        raise SystemExit(
            f"[mps] {ENV_VAR}={inherited!r} is already set in the environment, but "
            f"--cuda-mps-active-thread-percentage {requested} was requested. Remove one of them; "
            "the setting is never overridden silently."
        )
    env[ENV_VAR] = str(requested)
    if not env.get("CUDA_MPS_PIPE_DIRECTORY"):
        print(
            f"[mps] WARNING: --cuda-mps-active-thread-percentage {requested} set, but CUDA_MPS_PIPE_DIRECTORY "
            "is not set; MPS does not appear to be running and the setting has no effect.",
            file=sys.stderr,
            flush=True,
        )
```

- [ ] **Step 4: Call it from `gareus/cli.py` `main`**

Directly after `args = parse_args(argv_list)` in `main` (line ~1983):

```python
    # Before any OpenMM platform/context exists (spec 2026-09-26 §5).
    from .mps_share import apply_mps_thread_percentage
    apply_mps_thread_percentage(args)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest -q tests/test_replica_admission_config.py`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add gareus/mps_share.py gareus/cli.py tests/test_replica_admission_config.py
git commit -m "feat: apply CUDA MPS active-thread percentage before CUDA init"
```

---

### Task 5: Production wiring and handoff note

**Files:**
- Modify: `gareus/production.py` — replica construction loop (`props_i = replica_platform_properties(platform, props, args, i)`, line ~6992), right after the loop (after `if progress is not None: progress.progress("replica_construction", ...)`, line ~7155), and `step_all` (line ~7872-7892).
- Modify: `CLAUDE.md` (new short section at the top, after the "Adaptive top-ups" section).
- Test: existing `tests/test_replica_affinity_executor.py`, `tests/test_npt_driver_scheduling.py`, plus the three new files.

**Interfaces:**
- Consumes: `advance_replicas`, `make_dispatcher`, `queue_keys_from_platform_props`, `effective_limits` (Task 2); `record_replica_admission` (Task 3).
- Produces: nothing new for other tasks.

Do not touch the recon loop's `replica_platform_properties` call at line ~5748 (recon is out of scope).

- [ ] **Step 1: Collect each replica's platform properties**

Before the production replica loop (`for i in range(nrep):` just above line ~6992, where `replica_gamd_copy_report = []` is initialised), add:

```python
    _replica_platform_props: list[dict] = []
```

Directly after `props_i = replica_platform_properties(platform, props, args, i)` inside that loop, add:

```python
        _replica_platform_props.append(dict(props_i))
```

- [ ] **Step 2: Build and record the dispatcher after replica construction**

After the replica loop ends (after the `progress.progress("replica_construction", ...)` block, before `if _velocity_randomize_skip_count:`), add:

```python
    # Replica admission (spec 2026-09-26-replica-admission-design §4/§6): None
    # for the default 'all', which keeps step_all on today's pool.map path.
    _admission_queue_keys = queue_keys_from_platform_props(_replica_platform_props)
    _admission_cap = getattr(args, "active_replicas_per_gpu", "all")
    _admission_turn = int(getattr(args, "active_replica_turn_steps", 50))
    _admission_dispatcher = make_dispatcher(_sim_pool, _admission_queue_keys, _admission_cap, _admission_turn)
    _admission_effective = effective_limits(_admission_queue_keys, _admission_cap)
    print(
        f"[production] Replica admission: active_replicas_per_gpu={_admission_cap}, "
        f"turn_steps={_admission_turn if _admission_dispatcher is not None else 'n/a'}, "
        f"effective per queue={_admission_effective}",
        flush=True,
    )
    record_replica_admission(out_dir, args, _admission_effective)
```

Add the imports at the top of `gareus/production.py` next to the other `from .` imports:

```python
from .replica_admission import advance_replicas, effective_limits, make_dispatcher, queue_keys_from_platform_props
```

and add `record_replica_admission` to the existing `from .provenance import ...` line (find it with `grep -n "from .provenance import" gareus/production.py`).

- [ ] **Step 3: Route `step_all` through `advance_replicas`**

Replace the body of `step_all`'s `try:` block and the `_step_item` closure. Current code (line ~7872-7892):

```python
            def _step_item(item):
                _idx, _driver, _n = item
                _driver.advance(int(_n))
                return int(_idx)

            completed = 0
            try:
                if safe_chunk <= 0 or safe_chunk >= nsteps:
                    list(_sim_pool.map(_step_item, [(i, d, nsteps) for i, d in enumerate(drivers)]))
                else:
                    remaining = nsteps
                    while remaining > 0:
                        sub = min(int(safe_chunk), int(remaining))
                        list(_sim_pool.map(_step_item, [(i, d, sub) for i, d in enumerate(drivers)]))
                        completed += sub
                        remaining -= sub
```

New code:

```python
            completed = 0
            try:
                if safe_chunk <= 0 or safe_chunk >= nsteps:
                    advance_replicas(_sim_pool, drivers, nsteps, _admission_dispatcher)
                else:
                    remaining = nsteps
                    while remaining > 0:
                        sub = min(int(safe_chunk), int(remaining))
                        advance_replicas(_sim_pool, drivers, sub, _admission_dispatcher)
                        completed += sub
                        remaining -= sub
```

The `except Exception as exc:` block below stays exactly as it is.

- [ ] **Step 4: Confirm nothing else referenced `_step_item`**

Run: `grep -n "_step_item" gareus/production.py`
Expected: no output.

- [ ] **Step 5: Compile and run the targeted tests**

Run:
```bash
python -m py_compile gareus/production.py gareus/cli.py gareus/provenance.py gareus/replica_admission.py gareus/mps_share.py
pytest -q tests/test_replica_admission.py tests/test_replica_admission_drivers.py tests/test_replica_admission_config.py tests/test_replica_affinity_executor.py tests/test_npt_driver_scheduling.py tests/test_npt_cli_args.py
```
Expected: compile silent; all pass.

- [ ] **Step 6: Add the CLAUDE.md handoff section**

Insert after the "Adaptive top-ups" section of `CLAUDE.md`:

```markdown
## Replica admission cap + MPS thread share (`--active-replicas-per-gpu`, off by default)

- Files: `gareus/replica_admission.py` (`AdmissionDispatcher`, `advance_replicas`, `make_dispatcher`), `gareus/mps_share.py`, wiring in `gareus/production.py` (`step_all`, after replica construction), settings in `gareus/cli.py`, record in `gareus/provenance.py`. Spec `docs/superpowers/specs/2026-09-26-replica-admission-design.md`.
- Defaults `all` / `50` / `inherit` keep `step_all` on today's `pool.map` path (`dispatcher is None`). Benchmark best (harness, not yet production): 8 per GPU, 50-step turns, MPS 25 % = +78 % node ns/day at 236 contexts (jobs 2664328, 2665264).
- Turns are neutral: `ReplicaStepDriver.advance` admits deadlines `due <= end`, so a volume move or report on a turn end is serviced inside that turn, once. Pinned by `tests/test_replica_admission_drivers.py`.
- Dispatcher lock is a plain `Lock`, never held across submit/callback/fn/wait; synchronous callbacks are trampolined (no recursion on instant turns); first failure drains all GPUs before re-raising, so the pool is idle when NaN diagnostics read Contexts (today's `pool.map` path does not guarantee that).
- Manifest: ONE key, `method_settings["replica_admission"]`, replaced wholesale every job by `record_replica_admission` (a resume leaves a complete manifest's `method_settings` untouched otherwise), plus `replica_admission_history`. Never add flat keys.
- Readers use `method_settings["replica_admission"]`, never `resolved_args` (a stale first-job snapshot). History is one entry per `run_gareus` call, i.e. per phase directory.
- MPS percentage is set right after `parse_args`; errors if `openmm` is already imported or the environment holds a different value. At 25 %, single-context setup/recon phases can be up to ~4x slower on a fresh campaign (unmeasured).
- Tests: `tests/test_replica_admission.py`, `tests/test_replica_admission_drivers.py`, `tests/test_replica_admission_config.py`.
```

- [ ] **Step 7: Commit**

```bash
git add gareus/production.py CLAUDE.md
git commit -m "feat: route production step_all through replica admission"
```

- [ ] **Step 8: Hand off rollout (not part of this plan's code)**

Report to the user: the change is inert by default. Enabling on chignolin_9 (spec §9) needs a deploy to both aurum2 trees between jobs and three YAML keys; it is the user's decision and is not done here.
