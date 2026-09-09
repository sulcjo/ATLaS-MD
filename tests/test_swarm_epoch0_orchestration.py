"""Driving epoch 0 to completion across however many jobs it takes.

The orchestrator is the part that must not do the wrong thing twice: it decides
whether to run members, analyse, or step straight past a swarm that a previous
job already finished. MD is injected here, so these cover the decisions only.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gareus.io import write_csv_atomic, write_json
from gareus.swarm.epoch0 import (
    Epoch0GateFailure, mark_epoch0_complete, run_or_resume_epoch0,
)


class _Args:
    swarm_round = 0
    swarm_seed_ns = 1.0


class Epoch0OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.out = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.rd = self.out / "swarm" / "round_000"
        self.an = self.out / "swarm" / "analysis"
        self.calls = []

    # -- fakes -----------------------------------------------------------
    def _plan_and_members(self, n):
        write_csv_atomic(self.rd / "plan.csv", ["member_id", "replicate"],
                         ([i, 0] for i in range(n)))
        write_json(self.rd / "plan_meta.json", {"n_members": n})
        for i in range(n):
            write_json(self.rd / f"member_{i:04d}" / "done.json", {"status": "ok"})

    def _artefacts(self):
        write_csv_atomic(self.an / "windows_lambda_ladder.csv",
                         ["window", "primary_cv_center"], [[0, 0.5]])
        write_json(self.an / "shared_gamd_setup" / "shared_gamd_setup_globals.json", {"k0": 0.38})
        (self.an / "seed_bank").mkdir(parents=True, exist_ok=True)

    def _run_members(self, n=4, gate="pass"):
        def fake(args, out_dir, progress=None):
            self.calls.append("run_members")
            self._plan_and_members(n)
            return {"status": "ok", "n_members": n}
        return fake

    def _analyze(self, gate="pass"):
        def fake(out_dir, args):
            self.calls.append("analyze")
            if gate == "pass":
                self._artefacts()
            write_json(self.an / "swarm_gate.json", {"status": gate})
            return {"status": gate}
        return fake

    # -- cases -----------------------------------------------------------
    def test_a_cold_start_runs_members_then_analyses(self):
        ladder = run_or_resume_epoch0(
            _Args(), self.out,
            run_members=self._run_members(), analyze=self._analyze())
        self.assertEqual(self.calls, ["run_members", "analyze"])
        self.assertEqual(ladder.name, "windows_lambda_ladder.csv")
        self.assertTrue(ladder.exists())

    def test_a_finished_epoch0_is_not_run_again(self):
        self._plan_and_members(4)
        self._artefacts()
        write_json(self.an / "swarm_gate.json", {"status": "pass"})
        mark_epoch0_complete(self.out, n_members=4, ns_charged=4.0)

        ladder = run_or_resume_epoch0(
            _Args(), self.out,
            run_members=self._run_members(), analyze=self._analyze())
        self.assertEqual(self.calls, [], "a completed swarm was run again")
        self.assertTrue(ladder.exists())

    def test_members_already_done_skips_straight_to_analysis(self):
        self._plan_and_members(4)
        run_or_resume_epoch0(
            _Args(), self.out,
            run_members=self._run_members(), analyze=self._analyze())
        self.assertEqual(self.calls, ["analyze"])

    def test_a_gate_failure_raises_rather_than_returning_a_ladder(self):
        with self.assertRaises(Epoch0GateFailure):
            run_or_resume_epoch0(
                _Args(), self.out,
                run_members=self._run_members(), analyze=self._analyze(gate="fail"))

    def test_a_previously_recorded_gate_failure_is_not_retried(self):
        self._plan_and_members(4)
        write_json(self.an / "swarm_gate.json", {"status": "fail"})
        with self.assertRaises(Epoch0GateFailure):
            run_or_resume_epoch0(
                _Args(), self.out,
                run_members=self._run_members(), analyze=self._analyze())
        self.assertEqual(self.calls, [], "a failed round was re-run instead of stopping")

    def test_completion_is_marked_so_the_next_job_can_skip_it(self):
        run_or_resume_epoch0(
            _Args(), self.out,
            run_members=self._run_members(n=6), analyze=self._analyze())
        marker = self.an / "epoch0_complete.json"
        self.assertTrue(marker.exists())
        import json
        self.assertEqual(json.loads(marker.read_text())["n_members"], 6)

    def test_the_swarm_is_charged_to_the_budget_exactly_once(self):
        args = _Args()
        args.swarm_seed_ns = 2.0
        charged = []
        run_or_resume_epoch0(args, self.out, run_members=self._run_members(n=5),
                             analyze=self._analyze(), charge_ns=charged.append)
        self.assertEqual(charged, [10.0])          # 5 members x 2 ns

        # A later job in the chain must not charge the same swarm again.
        run_or_resume_epoch0(args, self.out, run_members=self._run_members(n=5),
                             analyze=self._analyze(), charge_ns=charged.append)
        self.assertEqual(charged, [10.0])


if __name__ == "__main__":
    unittest.main()
