import math
import unittest
import numpy as np
from gareus.mbar_analysis.ladder_overlap import ladder_overlap_by_axis


class LadderOverlapAxesTests(unittest.TestCase):
    def setUp(self):
        # 2 centres x 3 rungs, centre-major: state i -> centre i//3, rung i%3
        self.centers = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
        self.lambdas = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
        self.overlap = np.full((6, 6), 0.01)
        for i in range(6):
            self.overlap[i, i] = 1.0
        for a, b in ((0, 1), (1, 2), (3, 4), (4, 5)):      # along lambda
            self.overlap[a, b] = self.overlap[b, a] = 0.95
        for a, b in ((0, 3), (1, 4), (2, 5)):              # across CV1
            self.overlap[a, b] = self.overlap[b, a] = 0.40

    def test_separates_the_two_axes(self):
        out = ladder_overlap_by_axis(self.overlap, self.lambdas, self.centers)
        self.assertEqual(out["lambda_direction"]["n_pairs"], 4)
        self.assertEqual(out["cv1_direction"]["n_pairs"], 3)
        self.assertAlmostEqual(out["lambda_direction"]["worst"], 0.95)
        self.assertAlmostEqual(out["cv1_direction"]["worst"], 0.40)

    def test_worst_pair_is_reported(self):
        out = ladder_overlap_by_axis(self.overlap, self.lambdas, self.centers)
        self.assertEqual(tuple(sorted(out["cv1_direction"]["worst_pair"])), (0, 3))

    def test_non_ladder_run_has_no_lambda_pairs(self):
        out = ladder_overlap_by_axis(self.overlap, np.zeros(6), self.centers)
        self.assertEqual(out["lambda_direction"]["n_pairs"], 0)


class LadderOverlapAsymmetryTests(unittest.TestCase):
    """mbar_state_overlap is asymmetric whenever the two states' N_k differ --
    unequal per-state sample counts are the normal case under adaptive
    extension, not the exception (see gareus.mbar_analysis.ladder docstring
    and gareus/adaptive_production.py:3697-3725). A single pair must be
    reported as sqrt(O_ab * O_ba), never the raw O[a, b] or O[b, a]: either
    raw entry would make the reported value depend on which state happens to
    have the lower index, which is not physical.
    """

    def test_uses_symmetrised_value_not_raw_entry(self):
        centers = np.array([0.1, 0.1])
        lambdas = np.array([0.0, 1.0])
        # Deliberately asymmetric: e.g. state 0 has few samples, state 1 many.
        overlap = np.array([[1.0, 0.2],
                             [0.8, 1.0]])
        out = ladder_overlap_by_axis(overlap, lambdas, centers)
        expected = math.sqrt(0.2 * 0.8)

        self.assertEqual(out["lambda_direction"]["n_pairs"], 1)
        self.assertAlmostEqual(out["lambda_direction"]["worst"], expected)
        # A raw-value implementation would report one of these instead --
        # this is exactly what this test must fail against.
        self.assertNotAlmostEqual(out["lambda_direction"]["worst"], 0.2)
        self.assertNotAlmostEqual(out["lambda_direction"]["worst"], 0.8)


if __name__ == "__main__":
    unittest.main()
