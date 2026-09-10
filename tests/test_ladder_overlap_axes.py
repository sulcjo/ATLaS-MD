import math
import unittest
import numpy as np
from gareus.mbar_analysis.ladder_overlap import ladder_overlap_by_axis, ladder_overlap_health_checks


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
        out, warnings = ladder_overlap_by_axis(self.overlap, self.lambdas, self.centers)
        self.assertEqual(warnings, [])
        self.assertEqual(out["lambda_direction"]["n_pairs"], 4)
        self.assertEqual(out["cv1_direction"]["n_pairs"], 3)
        self.assertAlmostEqual(out["lambda_direction"]["worst"], 0.95)
        self.assertAlmostEqual(out["cv1_direction"]["worst"], 0.40)

    def test_worst_pair_is_reported(self):
        out, _ = ladder_overlap_by_axis(self.overlap, self.lambdas, self.centers)
        self.assertEqual(tuple(sorted(out["cv1_direction"]["worst_pair"])), (0, 3))

    def test_non_ladder_run_has_no_lambda_pairs(self):
        out, _ = ladder_overlap_by_axis(self.overlap, np.zeros(6), self.centers)
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
        out, _ = ladder_overlap_by_axis(overlap, lambdas, centers)
        expected = math.sqrt(0.2 * 0.8)

        self.assertEqual(out["lambda_direction"]["n_pairs"], 1)
        self.assertAlmostEqual(out["lambda_direction"]["worst"], expected)
        # A raw-value implementation would report one of these instead --
        # this is exactly what this test must fail against.
        self.assertNotAlmostEqual(out["lambda_direction"]["worst"], 0.2)
        self.assertNotAlmostEqual(out["lambda_direction"]["worst"], 0.8)


class LadderOverlapConnectivityTests(unittest.TestCase):
    """Whole-branch review, IMPORTANT 2: the spec requires each axis to carry
    its own connectivity verdict (n_components/connected), computed via
    gareus.mbar_analysis.pmf.overlap_components on that axis' own subgraph --
    not on the full state matrix, and not graded against n_components == 1
    (impossible whenever there is more than one CV1 centre).
    """

    def setUp(self):
        # 2 centres x 3 rungs, same layout as above: fully bridged both ways.
        self.centers = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
        self.lambdas = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
        self.overlap = np.full((6, 6), 0.01)
        for i in range(6):
            self.overlap[i, i] = 1.0
        for a, b in ((0, 1), (1, 2), (3, 4), (4, 5)):
            self.overlap[a, b] = self.overlap[b, a] = 0.95
        for a, b in ((0, 3), (1, 4), (2, 5)):
            self.overlap[a, b] = self.overlap[b, a] = 0.40

    def test_healthy_ladder_is_connected_per_axis_despite_multiple_centres(self):
        # Lambda axis is a forest of 2 per-centre chains -- n_components == 2
        # (one per centre) is the BEST possible outcome, not a defect, so
        # connected must be True even though n_components != 1.
        out, _ = ladder_overlap_by_axis(self.overlap, self.lambdas, self.centers, thr=0.30)
        lam = out["lambda_direction"]
        self.assertEqual(lam["n_components"], 2)
        self.assertTrue(lam["connected"])
        cv1 = out["cv1_direction"]
        self.assertEqual(cv1["n_components"], 3)  # one per rung
        self.assertTrue(cv1["connected"])

    def test_broken_link_within_a_centre_is_not_connected(self):
        # Snap the 1-2 lambda link at centre 0.1 below threshold: that
        # centre's chain now splits into {0} and {1,2} -- 3 lambda
        # components total instead of the achievable 2.
        overlap = self.overlap.copy()
        overlap[1, 2] = overlap[2, 1] = 0.01
        out, _ = ladder_overlap_by_axis(overlap, self.lambdas, self.centers, thr=0.30)
        lam = out["lambda_direction"]
        self.assertEqual(lam["n_components"], 3)
        self.assertEqual(lam["expected_components"], 2)
        self.assertFalse(lam["connected"])

    def test_health_checks_include_a_fail_row_for_the_severely_broken_axis(self):
        # Weakest link 0.01 against a 0.30 target is well below
        # OVERLAP_FAIL_FRACTION (0.5) * thr == 0.15 -- a genuine hard failure.
        overlap = self.overlap.copy()
        overlap[1, 2] = overlap[2, 1] = 0.01
        out, _ = ladder_overlap_by_axis(overlap, self.lambdas, self.centers, thr=0.30)
        checks = ladder_overlap_health_checks(out, 0.30)
        by_name = {c["name"]: c for c in checks}
        self.assertEqual(by_name["Connectivity along λ"]["status"], "fail")
        self.assertEqual(by_name["Connectivity across CV1"]["status"], "pass")

    def test_mildly_under_target_split_cautions_not_fails(self):
        # Weakest link 0.29 against a 0.30 target: already a CAUTION on the
        # pairwise row (0.15 <= 0.29 < 0.30). The connectivity row must not
        # escalate the SAME mildly-under-target link to a hard FAIL --
        # gareus_report._check_overlap_connectivity's own documented
        # convention is "never better than CAUTION since a split is a split",
        # not "any split is a FAIL".
        overlap = self.overlap.copy()
        overlap[1, 2] = overlap[2, 1] = 0.29
        out, _ = ladder_overlap_by_axis(overlap, self.lambdas, self.centers, thr=0.30)
        lam = out["lambda_direction"]
        self.assertFalse(lam["connected"])
        self.assertAlmostEqual(lam["worst"], 0.29)
        checks = ladder_overlap_health_checks(out, 0.30)
        by_name = {c["name"]: c for c in checks}
        self.assertEqual(by_name["Overlap along λ"]["status"], "caution")
        self.assertEqual(by_name["Connectivity along λ"]["status"], "caution")

    def test_no_ladder_pairs_grades_na_not_fail(self):
        # lam=zeros: no state differs from any other in lambda, so
        # lambda_direction has zero PAIRS -- there is nothing on this axis to
        # bridge at all (e.g. a single-rung ladder). This must match the
        # pairwise row's own NA ("no adjacent pairs on this axis"), not
        # report a vacuous FAIL from counting singleton nodes as a split.
        out, _ = ladder_overlap_by_axis(self.overlap, np.zeros(6), self.centers, thr=0.30)
        lam = out["lambda_direction"]
        self.assertEqual(lam["n_pairs"], 0)
        self.assertEqual(lam["n_components"], 0)
        self.assertIsNone(lam["connected"])
        checks = ladder_overlap_health_checks(out, 0.30)
        by_name = {c["name"]: c for c in checks}
        self.assertEqual(by_name["Connectivity along λ"]["status"], "na")
        self.assertEqual(by_name["Overlap along λ"]["status"], "na")

    def test_no_eligible_states_reports_na_connectivity(self):
        # n_k all zero: overlap_components has nothing to grade at all
        # (n_components == 0), which IS the "nothing graded" case -- distinct
        # from "graded and split".
        n_k = np.zeros(6)
        out, _ = ladder_overlap_by_axis(self.overlap, self.lambdas, self.centers, thr=0.30, n_k=n_k)
        lam = out["lambda_direction"]
        self.assertEqual(lam["n_components"], 0)
        self.assertIsNone(lam["connected"])
        checks = ladder_overlap_health_checks(out, 0.30)
        by_name = {c["name"]: c for c in checks}
        self.assertEqual(by_name["Connectivity along λ"]["status"], "na")


class LadderOverlapBadInputTests(unittest.TestCase):
    """Whole-branch review, MINOR 4: state_lambdas=None (reachable -- the
    CONTRADICTION case at gareus.mbar_analysis.crosscheck:131) and a
    length-mismatched centers/overlap array must degrade to a named warning,
    never an opaque IndexError bubbling into the caller's generic except.
    """

    def test_state_lambdas_none_returns_empty_summary_and_named_warning(self):
        overlap = np.eye(4)
        centers = np.array([0.1, 0.1, 0.5, 0.5])
        out, warnings = ladder_overlap_by_axis(overlap, None, centers)
        self.assertEqual(out["lambda_direction"]["n_pairs"], 0)
        self.assertIsNone(out["lambda_direction"]["worst"])
        self.assertEqual(out["cv1_direction"]["n_pairs"], 0)
        self.assertEqual(len(warnings), 1)
        self.assertIn("state_lambdas is None", warnings[0])

    def test_short_centers_returns_empty_summary_and_named_warning_not_indexerror(self):
        overlap = np.eye(4)
        lambdas = np.array([0.0, 0.5, 0.0, 0.5])
        centers = np.array([0.1, 0.1])  # too short: K=4, len(centers)=2
        out, warnings = ladder_overlap_by_axis(overlap, lambdas, centers)
        self.assertEqual(out["lambda_direction"]["n_pairs"], 0)
        self.assertEqual(len(warnings), 1)
        self.assertIn("length mismatch", warnings[0])

    def test_short_state_lambdas_returns_empty_summary_and_named_warning(self):
        overlap = np.eye(4)
        lambdas = np.array([0.0, 0.5])  # too short
        centers = np.array([0.1, 0.1, 0.5, 0.5])
        out, warnings = ladder_overlap_by_axis(overlap, lambdas, centers)
        self.assertEqual(out["cv1_direction"]["n_pairs"], 0)
        self.assertEqual(len(warnings), 1)
        self.assertIn("length mismatch", warnings[0])


if __name__ == "__main__":
    unittest.main()
