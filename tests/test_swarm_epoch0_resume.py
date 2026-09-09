"""What a chained job must conclude about epoch 0 from the filesystem alone.

Epoch 0 is long enough to be cut by a walltime stop, so the job that picks it up
has no memory of what the previous one did. It has to decide -- from disk only --
whether to keep running members, to analyse, to go on to epoch 1, or to stop.
Getting that wrong is expensive in both directions: redoing a finished epoch
burns the budget, and proceeding on a half-finished one silently designs the
ladder from a subset of the swarm.
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gareus.io import write_csv_atomic, write_json
from gareus.swarm.epoch0 import (
    EPOCH0_ANALYZE, EPOCH0_PROCEED, EPOCH0_RUN_MEMBERS, EPOCH0_STOP, epoch0_status,
    mark_epoch0_complete,
)

PLAN_HEADER = ["member_id", "replicate"]


class Epoch0StatusTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.out = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.rd = self.out / "swarm" / "round_000"
        self.an = self.out / "swarm" / "analysis"

    # -- helpers ---------------------------------------------------------
    def _plan(self, n):
        write_csv_atomic(self.rd / "plan.csv", PLAN_HEADER, ([i, i % 3] for i in range(n)))
        write_json(self.rd / "plan_meta.json", {"n_members": n})

    def _member(self, i, status="ok"):
        write_json(self.rd / f"member_{i:04d}" / "done.json", {"status": status})

    def _ladder(self):
        write_csv_atomic(self.an / "windows_lambda_ladder.csv",
                         ["window", "primary_cv_center"], [[0, 0.5]])
        write_json(self.an / "shared_gamd_setup" / "shared_gamd_setup_globals.json", {"k0_Total": 0.38})
        (self.an / "seed_bank").mkdir(parents=True, exist_ok=True)

    def _gate(self, status):
        write_json(self.an / "swarm_gate.json", {"status": status})

    # -- cases -----------------------------------------------------------
    def test_nothing_on_disk_means_run_members(self):
        st = epoch0_status(self.out)
        self.assertEqual(st["state"], "not_started")
        self.assertEqual(st["next_action"], EPOCH0_RUN_MEMBERS)

    def test_a_partly_finished_round_resumes_only_the_missing_members(self):
        self._plan(6)
        for i in (0, 1, 3):
            self._member(i)
        st = epoch0_status(self.out)
        self.assertEqual(st["state"], "members_pending")
        self.assertEqual(st["next_action"], EPOCH0_RUN_MEMBERS)
        self.assertEqual(st["n_planned"], 6)
        self.assertEqual(st["n_done"], 3)
        self.assertEqual(st["pending_member_ids"], [2, 4, 5])

    def test_a_torn_done_marker_counts_as_not_done(self):
        self._plan(2)
        self._member(0)
        md = self.rd / "member_0001"
        md.mkdir(parents=True, exist_ok=True)
        (md / "done.json").write_text('{"status": "o')      # killed mid-write
        st = epoch0_status(self.out)
        self.assertEqual(st["pending_member_ids"], [1])

    def test_all_members_done_but_not_analysed_means_analyse(self):
        self._plan(3)
        for i in range(3):
            self._member(i)
        st = epoch0_status(self.out)
        self.assertEqual(st["state"], "members_complete")
        self.assertEqual(st["next_action"], EPOCH0_ANALYZE)

    def test_a_failed_member_does_not_block_analysis(self):
        """Some members legitimately NaN; the gate decides whether enough survived."""
        self._plan(3)
        self._member(0); self._member(1, status="failed"); self._member(2)
        st = epoch0_status(self.out)
        self.assertEqual(st["next_action"], EPOCH0_ANALYZE)
        self.assertEqual(st["n_failed"], 1)

    def test_a_recorded_gate_failure_stops_rather_than_proceeding(self):
        self._plan(1); self._member(0)
        self._gate("fail")
        st = epoch0_status(self.out)
        self.assertEqual(st["state"], "blocked")
        self.assertEqual(st["next_action"], EPOCH0_STOP)

    def test_a_complete_epoch0_is_skipped(self):
        self._plan(1); self._member(0)
        self._ladder(); self._gate("pass")
        mark_epoch0_complete(self.out, n_members=1, ns_charged=174.0)
        st = epoch0_status(self.out)
        self.assertEqual(st["state"], "complete")
        self.assertEqual(st["next_action"], EPOCH0_PROCEED)
        self.assertEqual(st["ns_charged"], 174.0)

    def test_a_marker_without_its_artefacts_is_not_trusted(self):
        """A marker is only ever written after the artefacts, so this means damage."""
        self._plan(1); self._member(0)
        self._ladder(); self._gate("pass")
        mark_epoch0_complete(self.out, n_members=1, ns_charged=1.0)
        (self.an / "windows_lambda_ladder.csv").unlink()
        st = epoch0_status(self.out)
        self.assertNotEqual(st["state"], "complete")
        self.assertEqual(st["next_action"], EPOCH0_ANALYZE)

    def test_marker_refuses_to_be_written_before_the_artefacts_exist(self):
        self._plan(1); self._member(0)
        with self.assertRaises(FileNotFoundError):
            mark_epoch0_complete(self.out, n_members=1, ns_charged=1.0)

    def test_status_is_stable_when_called_twice(self):
        self._plan(2); self._member(0)
        first = epoch0_status(self.out)
        second = epoch0_status(self.out)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
