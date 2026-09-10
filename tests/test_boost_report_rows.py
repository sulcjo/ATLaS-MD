import unittest
from types import SimpleNamespace

import numpy as np
from gareus.mbar_analysis.boost_report import boost_report_rows, gamd_boost_by_rung_report
from gareus.units import KJ_PER_KCAL


class BoostReportTests(unittest.TestCase):
    def test_one_row_per_rung_in_lambda_order(self):
        lam = np.repeat([0.0, 0.5, 1.0], 500)
        dv = np.concatenate([np.zeros(500), np.full(500, 5.0), np.full(500, 9.0)])
        rows = boost_report_rows(dv, lam, kt_kcal=0.5962)
        self.assertEqual([r["lambda"] for r in rows], [0.0, 0.5, 1.0])
        self.assertEqual(rows[0]["n"], 500)
        self.assertAlmostEqual(rows[2]["mean_dv_kcal"], 9.0)

    def test_unboosted_rung_reports_zero_not_nan(self):
        lam = np.zeros(100)
        rows = boost_report_rows(np.zeros(100), lam, kt_kcal=0.5962)
        self.assertEqual(rows[0]["mean_dv_kcal"], 0.0)
        self.assertEqual(rows[0]["anharm_nats"], 0.0)

    def test_gaussian_boost_is_near_zero_anharmonicity(self):
        rng = np.random.default_rng(0)
        dv = rng.normal(10.0, 2.0, 20000)
        rows = boost_report_rows(dv, np.ones(20000), kt_kcal=0.5962)
        self.assertLess(abs(rows[0]["anharm_nats"]), 0.02)

    def test_bisection_style_lambdas_that_do_not_round_trip_all_get_rows(self):
        # Adaptive bisection (adaptive_production.py:4419, lam_mid =
        # 0.5*(lam_i+lam_j)) reaches values like 1/128 = 0.0078125, whose
        # true value differs from its own 6-dp rounding by more than the old
        # 1e-9 selection tolerance -- the exact case that silently dropped
        # rows (reviewer reproduced: linspace(0,1,7) dropped 4 of 7).
        lam_values = np.array([0.0, 1.0 / 128.0, 1.0 / 64.0, 1.0 / 32.0,
                                1.0 / 16.0, 1.0 / 8.0, 1.0])
        n_per = 50
        lam = np.repeat(lam_values, n_per)
        rng = np.random.default_rng(2)
        dv = rng.normal(5.0, 1.0, lam.size)
        rows = boost_report_rows(dv, lam, kt_kcal=0.5962)
        self.assertEqual(len(rows), len(lam_values))
        for r in rows:
            self.assertGreater(r["n"], 0)
        self.assertEqual(sum(r["n"] for r in rows), lam.size)

    def test_all_nan_boost_degrades_to_zero_rows_not_raise(self):
        lam = np.repeat([0.0, 0.5, 1.0], 10)
        dv = np.full(lam.size, np.nan)
        rows = boost_report_rows(dv, lam, kt_kcal=0.5962)
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertEqual(r["n"], 0)
            self.assertEqual(r["mean_dv_kcal"], 0.0)
            self.assertEqual(r["anharm_nats"], 0.0)

    def test_empty_lambdas_degrades_to_empty_list_not_raise(self):
        rows = boost_report_rows(np.array([]), np.array([]), kt_kcal=0.5962)
        self.assertEqual(rows, [])


def _fake_data(gamd_ladder, state_lambdas, window, boost_kj):
    """Minimal stand-in for gareus.mbar_analysis.data.Data: only the four
    attributes gamd_boost_by_rung_report actually reads."""
    return SimpleNamespace(
        meta={'gamd_ladder': gamd_ladder},
        state_lambdas=state_lambdas,
        window=window,
        boost_kj=boost_kj,
    )


class GamdBoostByRungReportWiringTests(unittest.TestCase):
    """Covers the wiring seam (analyze_gareus_mbar.py's call site delegates
    everything here), not just boost_report_rows' own happy paths -- these
    are the degradation paths the task's binding constraints name."""

    def test_non_ladder_run_is_a_no_op(self):
        d = _fake_data(
            gamd_ladder=False,
            state_lambdas=np.array([0.0, 0.0]),
            window=np.array([0, 1, 0, 1]),
            boost_kj=np.zeros(4),
        )
        rows, warnings = gamd_boost_by_rung_report(d, kbt_kcal=0.5962)
        # None (not []) is the caller's signal to skip
        # s['gamd_boost_by_rung'] = ... entirely -- a non-ladder run's
        # pmf_summary.json must never gain the key.
        self.assertIsNone(rows)
        self.assertEqual(warnings, [])

    def test_all_nan_boost_kj_degrades_to_zero_count_rows(self):
        state_lambdas = np.array([0.0, 0.56, 1.0])
        window = np.repeat([0, 1, 2], 10)
        boost_kj = np.full(window.size, np.nan)
        d = _fake_data(True, state_lambdas, window, boost_kj)
        rows, warnings = gamd_boost_by_rung_report(d, kbt_kcal=0.5962)
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertEqual(r["n"], 0)
            self.assertEqual(r["mean_dv_kcal"], 0.0)
        # 3 distinct rungs were found (from state_lambdas), so this is not
        # the <2-distinct-rungs case -- no spurious warning.
        self.assertEqual(warnings, [])

    def test_state_lambdas_none_degrades_to_empty_rows_with_warning(self):
        window = np.repeat([0, 1, 2], 10)
        boost_kj = np.full(window.size, 5.0 * KJ_PER_KCAL)
        d = _fake_data(True, None, window, boost_kj)
        rows, warnings = gamd_boost_by_rung_report(d, kbt_kcal=0.5962)
        self.assertEqual(rows, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("fewer than 2 distinct lambda rungs", warnings[0])

    def test_collapsed_single_rung_warns_instead_of_silently_pooling(self):
        # meta['gamd_ladder'] is True but every state carries the same
        # lambda (e.g. an all-zero state_registry.csv) -- exactly the
        # pooling-across-rungs failure mode this report exists to expose.
        state_lambdas = np.array([0.0, 0.0, 0.0])
        window = np.repeat([0, 1, 2], 10)
        boost_kj = np.zeros(window.size)
        d = _fake_data(True, state_lambdas, window, boost_kj)
        rows, warnings = gamd_boost_by_rung_report(d, kbt_kcal=0.5962)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(warnings), 1)
        self.assertIn("fewer than 2 distinct lambda rungs", warnings[0])

    def test_exception_inside_degrades_to_empty_rows_with_warning_not_raise(self):
        # window values out of range for state_lambdas -> IndexError inside
        # the function; must degrade, never propagate.
        d = _fake_data(
            gamd_ladder=True,
            state_lambdas=np.array([0.0, 1.0]),
            window=np.array([0, 1, 5]),  # 5 is out of range
            boost_kj=np.zeros(3),
        )
        rows, warnings = gamd_boost_by_rung_report(d, kbt_kcal=0.5962)
        self.assertEqual(rows, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("gamd boost/rung report failed", warnings[0])


if __name__ == "__main__":
    unittest.main()
