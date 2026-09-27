"""advance_replicas over real ReplicaStepDrivers: turns visit the same events.

Spec §6/§7: splitting a replica's advance(n) into capped turns must give the
same step counts and the same volume-move and report steps, in the same order,
as today's one-call-per-replica pool.map. Tiny 4-particle Reference systems
(the test_npt_driver_scheduling pattern) keep this CPU-only and fast.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import LangevinMiddleIntegrator, Platform, System, Vec3, unit  # noqa: E402
from openmm.app import Simulation, Topology  # noqa: E402
from openmm.app.element import Element  # noqa: E402

import gareus.replica_admission as replica_admission  # noqa: E402
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


def test_advance_replicas_rejects_driver_count_mismatch():
    pool = _ReplicaAffinityExecutor(3)
    try:
        dispatcher = AdmissionDispatcher(pool, ["0", "0", "1"], limit=8, turn_steps=50)
        drivers = [CountingDriver(), CountingDriver()]  # 2 drivers, dispatcher built for 3
        with pytest.raises(ValueError):
            advance_replicas(pool, drivers, 10, dispatcher)
    finally:
        pool.shutdown(wait=True)


def test_run_interrupted_mid_flight_stops_new_turns_and_recovers(monkeypatch):
    """KeyboardInterrupt from Event.wait re-raises, and no turn starts after it settles.

    A later run() on the same dispatcher still works once the earlier (still
    in-flight when interrupted) turns have had time to finish.
    """
    starts: list[float] = []
    starts_lock = threading.Lock()

    def fn(i, n):
        with starts_lock:
            starts.append(time.monotonic())
        time.sleep(0.005)

    raised = {"done": False}

    class _RaiseOnceEvent(threading.Event):
        def wait(self, timeout=None):
            if not raised["done"]:
                raised["done"] = True
                time.sleep(0.05)
                raise KeyboardInterrupt()
            return super().wait(timeout)

    monkeypatch.setattr(replica_admission, "_Event", _RaiseOnceEvent)

    pool = _ReplicaAffinityExecutor(2)
    try:
        dispatcher = AdmissionDispatcher(pool, ["0", "1"], limit=1, turn_steps=1)
        interrupt_at = time.monotonic() + 0.05
        with pytest.raises(KeyboardInterrupt):
            dispatcher.run(fn, 10_000)

        # Give any turn that was already in flight at the moment of the interrupt
        # a chance to finish, then confirm no new turn started after that point.
        time.sleep(0.1)
        assert all(s <= interrupt_at + 0.05 for s in starts)

        # Reusing the dispatcher works once the earlier turns have fully drained.
        time.sleep(0.1)
        starts.clear()
        dispatcher.run(fn, 5)
        assert starts
    finally:
        pool.shutdown(wait=True)
