"""Running swarm members concurrently, which is what makes epoch 0 affordable.

The four sharded jobs used to BE the parallelism: gareus/swarm ran members in a
strict serial loop, and four SLURM jobs ran four copies of it. Folding the swarm
into one self-contained run removes those shards, so the concurrency has to come
from inside -- otherwise 174 members x 1 ns is ~1.6 days on one GPU instead of
~10 h across four.

Two properties matter beyond raw speed. Every planned member must run exactly
once, resumed members included; and each member must keep its own velocity seed,
which is not free because run_member sets args.seed and restores it, so workers
sharing one args object would overwrite each other's seed.
"""
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gareus.io import write_json
from gareus.swarm.driver import execute_members, resolve_member_workers


class _Args:
    def __init__(self, **kw):
        self.setup_platform = "CUDA"
        self.device_index = "0,1,2,3"
        self.swarm_member_workers = "auto"
        self.seed = 1234
        for k, v in kw.items():
            setattr(self, k, v)


def _rows(n):
    return [{"member_id": i, "velocity_seed": 9000 + i, "replicate": i % 3} for i in range(n)]


class ResolveWorkersTests(unittest.TestCase):
    def test_auto_gives_one_worker_per_device_on_gpu(self):
        n, devices = resolve_member_workers(_Args())
        self.assertEqual(devices, ["0", "1", "2", "3"])
        self.assertEqual(n, 4)

    def test_auto_is_serial_on_cpu(self):
        n, devices = resolve_member_workers(_Args(setup_platform="CPU"))
        self.assertEqual(n, 1)

    def test_an_explicit_count_wins(self):
        n, _ = resolve_member_workers(_Args(swarm_member_workers="8"))
        self.assertEqual(n, 8)

    def test_a_nonsense_value_falls_back_to_serial_rather_than_crashing(self):
        n, _ = resolve_member_workers(_Args(swarm_member_workers="banana"))
        self.assertEqual(n, 1)


class ExecuteMembersTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.rd = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_every_planned_member_runs_exactly_once(self):
        rows = _rows(17)
        seen = []
        lock = threading.Lock()

        def run_one(worker_args, row, member_dir, device):
            with lock:
                seen.append(int(row["member_id"]))
            return {"status": "ok"}

        result = execute_members(_Args(), rows, range(len(rows)), self.rd, run_one,
                                 n_workers=4, device_tokens=["0", "1", "2", "3"])
        self.assertEqual(sorted(seen), list(range(17)))
        self.assertEqual(result["n_run"], 17)

    def test_each_member_keeps_its_own_velocity_seed(self):
        """Workers must not share one args object; run_member writes args.seed."""
        rows = _rows(40)
        observed = {}
        lock = threading.Lock()

        def run_one(worker_args, row, member_dir, device):
            worker_args.seed = int(row["velocity_seed"])   # what run_member does
            # Yield so a shared object would be visibly clobbered here.
            threading.Event().wait(0.001)
            with lock:
                observed[int(row["member_id"])] = worker_args.seed
            return {"status": "ok"}

        execute_members(_Args(), rows, range(len(rows)), self.rd, run_one,
                        n_workers=8, device_tokens=["0", "1"])
        self.assertEqual(observed, {i: 9000 + i for i in range(40)})

    def test_finished_members_are_skipped_not_rerun(self):
        rows = _rows(6)
        for i in (0, 2, 4):
            write_json(self.rd / f"member_{i:04d}" / "done.json", {"status": "ok"})
        ran = []
        lock = threading.Lock()

        def run_one(worker_args, row, member_dir, device):
            with lock:
                ran.append(int(row["member_id"]))
            return {"status": "ok"}

        result = execute_members(_Args(), rows, range(len(rows)), self.rd, run_one,
                                 n_workers=3, device_tokens=["0"])
        self.assertEqual(sorted(ran), [1, 3, 5])
        self.assertEqual(result["n_skipped_resume"], 3)

    def test_devices_are_handed_out_round_robin(self):
        rows = _rows(8)
        got = {}
        lock = threading.Lock()

        def run_one(worker_args, row, member_dir, device):
            with lock:
                got[int(row["member_id"])] = device
            return {"status": "ok"}

        execute_members(_Args(), rows, range(len(rows)), self.rd, run_one,
                        n_workers=4, device_tokens=["0", "1", "2", "3"])
        self.assertEqual({got[i] for i in range(8)}, {"0", "1", "2", "3"})

    def test_one_failing_member_does_not_abort_the_round(self):
        rows = _rows(6)

        def run_one(worker_args, row, member_dir, device):
            if int(row["member_id"]) == 3:
                raise RuntimeError("Particle coordinate is NaN")
            return {"status": "ok"}

        result = execute_members(_Args(), rows, range(len(rows)), self.rd, run_one,
                                 n_workers=3, device_tokens=["0"])
        self.assertEqual(result["n_run"], 5)
        self.assertIn(3, result["failed_in_range"])


if __name__ == "__main__":
    unittest.main()
