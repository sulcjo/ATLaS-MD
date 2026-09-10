"""Crash- and race-safety of the on-disk writes the swarm epoch depends on.

Epoch 0 (the swarm) can be interrupted by a walltime stop at any instant and is
resumed by a later job in the chain, so every file it leaves behind must be
either complete or absent -- never half-written. Two separate hazards are
covered here:

* a process killed mid-write must not truncate a file a later job will read;
* two processes writing the same path concurrently must not collide on a
  shared temporary name (the failure that killed swarm shard 3 on job 2374840,
  where all four shards raced on ``run_args.json.tmp``).
"""
import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from gareus.io import write_csv_atomic, write_json, write_text_atomic


class WriteJsonAtomicityTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_leaves_no_temporary_file_behind(self):
        target = self.root / "done.json"
        write_json(target, {"status": "ok"})
        strays = [p.name for p in self.root.iterdir() if p.name != "done.json"]
        self.assertEqual(strays, [], f"temporary files left behind: {strays}")

    def test_failure_mid_write_leaves_previous_content_intact(self):
        target = self.root / "done.json"
        write_json(target, {"status": "ok", "generation": 1})

        # A payload the encoder cannot serialise fails after the temp file is
        # opened; the original must survive untouched.
        with self.assertRaises(TypeError):
            write_json(target, {"status": object()})

        self.assertEqual(json.loads(target.read_text()),
                         {"status": "ok", "generation": 1})

    def test_temporary_name_is_process_unique(self):
        """A shared '<name>.tmp' lets concurrent writers rename over each other."""
        target = self.root / "run_args.json"
        seen = []
        real_replace = Path.replace

        def capture(self_path, dest):
            seen.append(Path(self_path).name)
            return real_replace(self_path, dest)

        with mock.patch.object(Path, "replace", capture):
            write_json(target, {"a": 1})

        self.assertEqual(len(seen), 1)
        self.assertIn(str(os.getpid()), seen[0],
                      f"temp name {seen[0]!r} is not process-unique")


class WriteCsvAtomicTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_writes_header_and_rows(self):
        target = self.root / "plan.csv"
        write_csv_atomic(target, ["window", "center"], [[0, "0.105"], [1, "0.240"]])
        self.assertEqual(target.read_text().splitlines(),
                         ["window,center", "0,0.105", "1,0.240"])

    def test_creates_missing_parent_directories(self):
        target = self.root / "swarm" / "analysis" / "windows_lambda_ladder.csv"
        write_csv_atomic(target, ["window"], [[0]])
        self.assertTrue(target.exists())

    def test_failure_mid_write_leaves_previous_content_intact(self):
        target = self.root / "plan.csv"
        write_csv_atomic(target, ["window"], [[0], [1]])
        before = target.read_text()

        def exploding_rows():
            yield [2]
            raise RuntimeError("killed mid-write")

        with self.assertRaises(RuntimeError):
            write_csv_atomic(target, ["window"], exploding_rows())

        self.assertEqual(target.read_text(), before)
        strays = [p.name for p in self.root.iterdir() if p.name != "plan.csv"]
        self.assertEqual(strays, [], f"temporary files left behind: {strays}")

    def test_concurrent_writers_never_yield_a_partial_file(self):
        """Every reader must see one complete payload, never an interleaving."""
        target = self.root / "plan.csv"
        payloads = {n: [[i] for i in range(n)] for n in (50, 400, 1200)}

        def write(n):
            write_csv_atomic(target, ["window"], payloads[n])

        for _ in range(12):
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(write, payloads))
            body = target.read_text().splitlines()
            self.assertEqual(body[0], "window")
            self.assertIn(len(body) - 1, payloads,
                          f"partial file: {len(body) - 1} data rows")


class WriteTextAtomicTests(unittest.TestCase):
    """The runtime-pool ledger's Markdown twin is now rewritten on every charge.

    It is a human-readable report rather than something the code parses back, but
    it is written beside the JSON on every single consume(), so a kill mid-write
    must leave the previous report intact rather than a truncated one.
    """

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_writes_content_and_creates_parents(self):
        target = self.root / "nested" / "report.md"
        write_text_atomic(target, "# Report\n")
        self.assertEqual(target.read_text(), "# Report\n")

    def test_leaves_no_temporary_file_behind(self):
        target = self.root / "report.md"
        write_text_atomic(target, "body")
        self.assertEqual([p.name for p in self.root.iterdir()], ["report.md"])

    def test_failure_mid_write_leaves_previous_content_intact(self):
        target = self.root / "report.md"
        write_text_atomic(target, "original")

        with mock.patch("gareus.io._persist", side_effect=RuntimeError("killed mid-write")):
            with self.assertRaises(RuntimeError):
                write_text_atomic(target, "replacement")

        self.assertEqual(target.read_text(), "original")
        strays = [p.name for p in self.root.iterdir() if p.name != "report.md"]
        self.assertEqual(strays, [], f"temporary files left behind: {strays}")

    def test_concurrent_writers_never_yield_a_partial_file(self):
        target = self.root / "report.md"
        payloads = {n: ("x" * n) for n in (50, 400, 1200)}

        def write(n):
            write_text_atomic(target, payloads[n])

        for _ in range(12):
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(write, payloads))
            self.assertIn(len(target.read_text()), payloads,
                          "partial file observed")


if __name__ == "__main__":
    unittest.main()
