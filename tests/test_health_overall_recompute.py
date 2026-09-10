"""Whole-branch review, IMPORTANT 3: analyze_gareus_mbar.py used to append the
two ladder-axis health checks AFTER gareus_report.build_health_verdict had
already computed s['health']['overall'], so a fresh FAIL row could render
underneath 'RESULT HEALTH: PASS'. The fix recomputes overall from the full
checks list via the new gareus_report.overall_from_checks, reusing
build_health_verdict's own worst-status rule rather than reimplementing it.

These tests cover the reusable primitives directly (overall_from_checks, and
gareus.mbar_analysis.ladder_overlap.ladder_overlap_health_checks, isolated
from the report-writer wiring the same way boost_report's
gamd_boost_by_rung_report is) rather than the whole analyze_gareus_mbar.py
pipeline, which needs a full Data/MBAR fixture to reach that call site.
"""
import unittest

import gareus_report as gr
from gareus.mbar_analysis.ladder_overlap import ladder_overlap_by_axis, ladder_overlap_health_checks


def _good_summary():
    return {
        "mbar": {"converged": True, "max_delta": 1e-8, "n_k": [100, 100, 100]},
        "n_samples": 300,
        "selected_unbiased_method": "umbrella_only",
    }


class OverallFromChecksMatchesBuildHealthVerdictTests(unittest.TestCase):
    """Regression safety net for the build_health_verdict refactor: it must
    still compute overall exactly as before, now via the shared helper."""

    def test_matches_on_a_clean_summary(self):
        v = gr.build_health_verdict(_good_summary(), 0.30)
        self.assertEqual(v["overall"], gr.overall_from_checks(v["checks"]))

    def test_matches_on_an_empty_summary(self):
        v = gr.build_health_verdict({}, 0.30)
        self.assertEqual(v["overall"], gr.overall_from_checks(v["checks"]))

    def test_worst_status_wins(self):
        checks = [{"status": gr.PASS}, {"status": gr.FAIL}, {"status": gr.CAUTION}]
        self.assertEqual(gr.overall_from_checks(checks), "FAIL")

    def test_all_na_is_unknown(self):
        checks = [{"status": gr.NA}, {"status": gr.NA}]
        self.assertEqual(gr.overall_from_checks(checks), "UNKNOWN")

    def test_empty_list_is_unknown(self):
        self.assertEqual(gr.overall_from_checks([]), "UNKNOWN")


class LadderChecksCannotContradictOverallTests(unittest.TestCase):
    """Reproduces the exact defect: a health verdict that was PASS before the
    ladder-axis checks existed must become FAIL once a broken axis' checks
    are appended AND overall is recomputed -- it must never stay PASS with a
    FAIL row rendered underneath it.
    """

    def setUp(self):
        # 2 centres x 3 rungs, lambda link 1-2 at centre 0.1 snapped below
        # threshold -- same broken-axis fixture as
        # test_ladder_overlap_axes.LadderOverlapConnectivityTests.
        import numpy as np
        self.centers = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
        self.lambdas = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
        overlap = np.full((6, 6), 0.01)
        for i in range(6):
            overlap[i, i] = 1.0
        for a, b in ((0, 1), (3, 4), (4, 5)):
            overlap[a, b] = overlap[b, a] = 0.95
        overlap[1, 2] = overlap[2, 1] = 0.01  # broken link
        for a, b in ((0, 3), (1, 4), (2, 5)):
            overlap[a, b] = overlap[b, a] = 0.40
        self.overlap = overlap

    def test_appending_a_broken_axis_flips_overall_to_fail(self):
        health = gr.build_health_verdict(_good_summary(), 0.30)
        self.assertEqual(health["overall"], "PASS")

        lo, warnings = ladder_overlap_by_axis(self.overlap, self.lambdas, self.centers, thr=0.30)
        self.assertEqual(warnings, [])
        health["checks"].extend(ladder_overlap_health_checks(lo, 0.30))
        health["overall"] = gr.overall_from_checks(health["checks"])

        self.assertEqual(health["overall"], "FAIL")
        statuses = {c["name"]: c["status"] for c in health["checks"]}
        self.assertEqual(statuses["Connectivity along λ"], "fail")

    def test_forgetting_to_recompute_reproduces_the_bug_this_guards_against(self):
        # Sanity check that the test above is discriminating: WITHOUT the
        # recompute step, overall stays the stale PASS even though a FAIL row
        # now sits in checks -- this is the exact contradiction the fix
        # removes, kept here as a control.
        health = gr.build_health_verdict(_good_summary(), 0.30)
        lo, _ = ladder_overlap_by_axis(self.overlap, self.lambdas, self.centers, thr=0.30)
        health["checks"].extend(ladder_overlap_health_checks(lo, 0.30))
        # overall deliberately NOT recomputed here.
        self.assertEqual(health["overall"], "PASS")
        self.assertTrue(any(c["status"] == "fail" for c in health["checks"]))


if __name__ == "__main__":
    unittest.main()
