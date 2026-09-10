import unittest
from gareus.production import explicit_window_analysis_rows


class ExplicitWindowTableLambdaTests(unittest.TestCase):
    def test_rows_carry_the_supplied_rung(self):
        rows = explicit_window_analysis_rows(
            centers_a=[0.11, 0.11, 0.30],
            k_list=[246.2, 246.2, 300.0],
            gamd_lambdas=[0.0, 0.5, 1.0],
        )
        self.assertEqual([r["gamd_lambda"] for r in rows], [0.0, 0.5, 1.0])

    def test_absent_lambda_defaults_to_zero_not_missing(self):
        """A non-ladder run must still emit the column, so readers never KeyError."""
        rows = explicit_window_analysis_rows(centers_a=[0.11], k_list=[246.2])
        self.assertEqual(rows[0]["gamd_lambda"], 0.0)


if __name__ == "__main__":
    unittest.main()
