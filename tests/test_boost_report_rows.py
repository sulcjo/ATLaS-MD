import unittest
import numpy as np
from gareus.mbar_analysis.boost_report import boost_report_rows


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


if __name__ == "__main__":
    unittest.main()
