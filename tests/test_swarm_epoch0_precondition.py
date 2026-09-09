"""Epoch 0 must only engage for campaigns that actually have a swarm to run.

The hook fires when no window table was handed in, which is also true of plenty
of adaptive-production runs that never wanted a swarm at all -- they get their
states from a registry, an epoch directory, or a resume. Running the swarm for
those turns a working run into `SystemExit: --swarm-stage run needs
--seed-conformers-dir <GENPEPT library>`.

The precondition is exactly run_swarm_stage's own: a seed library with
final_survivor_seeds.csv in it. Anything less and the old behaviour stands.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gareus.swarm.epoch0 import epoch0_is_available


class _Args:
    def __init__(self, seed_conformers_dir=None):
        self.seed_conformers_dir = seed_conformers_dir


class Epoch0PreconditionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_no_seed_directory_means_no_swarm(self):
        self.assertFalse(epoch0_is_available(_Args()))
        self.assertFalse(epoch0_is_available(_Args("")))

    def test_a_missing_directory_means_no_swarm(self):
        self.assertFalse(epoch0_is_available(_Args(str(self.root / "nope"))))

    def test_a_directory_without_the_library_index_means_no_swarm(self):
        """A seed bank from a previous campaign is not a GENPEPT library."""
        bank = self.root / "seed_bank"
        bank.mkdir()
        (bank / "window_000.pdb").write_text("ATOM\n")
        self.assertFalse(epoch0_is_available(_Args(str(bank))))

    def test_a_genpept_library_means_the_swarm_can_run(self):
        lib = self.root / "genpept"
        lib.mkdir()
        (lib / "final_survivor_seeds.csv").write_text("pdb_path\nx.pdb\n")
        self.assertTrue(epoch0_is_available(_Args(str(lib))))

    def test_an_empty_library_index_does_not_count(self):
        lib = self.root / "genpept_empty"
        lib.mkdir()
        (lib / "final_survivor_seeds.csv").write_text("")
        self.assertFalse(epoch0_is_available(_Args(str(lib))))


if __name__ == "__main__":
    unittest.main()
