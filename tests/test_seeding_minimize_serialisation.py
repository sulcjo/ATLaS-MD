"""Energy minimisation must never run in two seeding workers at once.

OpenMM's LocalEnergyMinimizer is not thread-safe. Seeding runs
--us-pull-workers contexts concurrently (28 in the chignolin_7 campaign), and
minimising from several of them corrupted the heap: `free(): invalid pointer`
with every worker thread inside openmm.minimize, under
graft_conformer_into_context (job 2379581).

The MD, pulling and state I/O stay concurrent; only the minimiser is serialised.
"""
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from gareus.seeding import _minimize_energy


class _RecordingSim:
    """Stands in for a Simulation, recording how many minimisations overlap."""

    def __init__(self, tracker):
        self.tracker = tracker

    def minimizeEnergy(self, **kwargs):
        self.tracker.enter(kwargs)
        # Yield the GIL inside the critical section: without a lock, other
        # workers would be inside minimizeEnergy at this point too.
        threading.Event().wait(0.002)
        self.tracker.leave()


class _Tracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.live = 0
        self.max_live = 0
        self.calls = []

    def enter(self, kwargs):
        with self.lock:
            self.live += 1
            self.max_live = max(self.max_live, self.live)
            self.calls.append(kwargs)

    def leave(self):
        with self.lock:
            self.live -= 1


class MinimizeSerialisationTests(unittest.TestCase):
    def test_only_one_minimisation_runs_at_a_time(self):
        tracker = _Tracker()
        sims = [_RecordingSim(tracker) for _ in range(28)]
        with ThreadPoolExecutor(max_workers=28) as ex:
            list(ex.map(lambda s: _minimize_energy(s, maxIterations=500), sims))
        self.assertEqual(tracker.max_live, 1,
                         f"{tracker.max_live} concurrent minimisations; must be 1")
        self.assertEqual(len(tracker.calls), 28)

    def test_arguments_are_passed_through(self):
        tracker = _Tracker()
        _minimize_energy(_RecordingSim(tracker), maxIterations=123)
        self.assertEqual(tracker.calls, [{"maxIterations": 123}])

    def test_supports_the_no_argument_form(self):
        tracker = _Tracker()
        _minimize_energy(_RecordingSim(tracker))
        self.assertEqual(tracker.calls, [{}])

    def test_the_lock_is_released_when_minimisation_raises(self):
        """A failed graft must not deadlock every other worker."""
        class _Boom:
            def minimizeEnergy(self, **kw):
                raise RuntimeError("particle position is NaN")

        with self.assertRaises(RuntimeError):
            _minimize_energy(_Boom())
        tracker = _Tracker()
        _minimize_energy(_RecordingSim(tracker))   # would hang if still held
        self.assertEqual(len(tracker.calls), 1)


if __name__ == "__main__":
    unittest.main()
