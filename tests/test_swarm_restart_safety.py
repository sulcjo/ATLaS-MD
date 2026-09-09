"""The swarm epoch must survive a walltime kill at any instant.

Epoch 0 runs long enough to be cut in half by a 4 h job limit and continued by
the next job in the chain. The files below are the ones a continuing job reads
back as fact, so each must be all-or-nothing. A truncated plan runs fewer
members than were designed and a truncated ladder yields fewer production
states -- neither raises, so neither would be noticed.
"""
import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from gareus.swarm.driver import _load_plan, _write_plan, PLAN_COLUMNS
from gareus.swarm.ladder_design import write_ladder_windows_csv


def _plan_rows(n):
    rows = []
    for i in range(n):
        row = {c: 0 for c in PLAN_COLUMNS}
        row["member_id"] = i
        row["replicate"] = i % 3
        rows.append(row)
    return rows


class PlanWriteAtomicityTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.rd = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_round_trips_every_planned_member(self):
        rows = _plan_rows(174)
        _write_plan(self.rd, rows, {"n_members": 174})
        back, meta = _load_plan(self.rd)
        self.assertEqual(len(back), 174)
        self.assertEqual(meta["n_members"], 174)
        self.assertEqual([r["member_id"] for r in back], list(range(174)))

    def test_a_kill_mid_write_cannot_leave_a_short_plan(self):
        _write_plan(self.rd, _plan_rows(174), {"n_members": 174})
        intact = (self.rd / "plan.csv").read_text()

        # A row source that dies part-way stands in for the process being
        # killed while the plan is still being written.
        def rows_that_die_part_way():
            for n, row in enumerate(_plan_rows(174)):
                if n > 5:
                    raise RuntimeError("walltime kill")
                yield row

        with self.assertRaises(RuntimeError):
            _write_plan(self.rd, rows_that_die_part_way(), {"n_members": 174})

        self.assertEqual((self.rd / "plan.csv").read_text(), intact,
                         "a partially written plan replaced the good one")
        back, _ = _load_plan(self.rd)
        self.assertEqual(len(back), 174)

    def test_no_staging_files_survive_a_failed_write(self):
        _write_plan(self.rd, _plan_rows(10), {"n_members": 10})
        expected = {"plan.csv", "plan_meta.json"}
        self.assertEqual({p.name for p in self.rd.iterdir()}, expected)


class LadderCsvAtomicityTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.an = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_writes_the_full_cross_product(self):
        path = write_ladder_windows_csv(
            self.an / "windows_lambda_ladder.csv",
            centers=[0.105, 0.5, 0.906], ks_kcal=[110.0, 300.0, 505.0],
            lambdas=[0.0, 0.5, 1.0])
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 9)

    def test_a_kill_mid_write_cannot_leave_a_short_ladder(self):
        target = self.an / "windows_lambda_ladder.csv"
        write_ladder_windows_csv(target, [0.1, 0.9], [110.0, 505.0], [0.0, 1.0])
        intact = target.read_text()

        def centres_that_die_part_way():
            for i in range(20):
                if i > 3:
                    raise RuntimeError("walltime kill")
                yield 0.1 * i

        with self.assertRaises(RuntimeError):
            write_ladder_windows_csv(target, centres_that_die_part_way(),
                                     [110.0] * 20, [0.0, 1.0])

        self.assertEqual(target.read_text(), intact,
                         "a partially written ladder replaced the good one")

    def test_target_is_never_absent_while_being_rewritten(self):
        """analyze() must not unlink the ladder before rebuilding it."""
        target = self.an / "windows_lambda_ladder.csv"
        write_ladder_windows_csv(target, [0.1], [110.0], [0.0])
        seen = []
        real_replace = Path.replace

        def watch(self_path, dest):
            seen.append(Path(dest).exists())
            return real_replace(self_path, dest)

        with mock.patch.object(Path, "replace", watch):
            write_ladder_windows_csv(target, [0.2], [120.0], [0.0])
        self.assertEqual(seen, [True], "ladder vanished before the rename")


if __name__ == "__main__":
    unittest.main()
