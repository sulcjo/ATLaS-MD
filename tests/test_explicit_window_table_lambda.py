import csv
import tempfile
import unittest
from pathlib import Path

from gareus.production import (
    explicit_window_analysis_rows,
    write_explicit_window_analysis_files,
)


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

    def test_written_csv_carries_the_rungs(self):
        """The extrasaction='ignore' trap: a row key absent from the writer's
        fieldnames list is silently dropped, so this must exercise the real
        CSV round-trip, not just the in-memory row dicts."""
        with tempfile.TemporaryDirectory() as tmp:
            write_explicit_window_analysis_files(
                Path(tmp),
                centers_a=[0.11, 0.11, 0.30],
                k_list=[246.2, 246.2, 300.0],
                gamd_lambdas=[0.0, 0.5, 1.0],
            )
            with (Path(tmp) / "umbrella_explicit_windows.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([float(r["gamd_lambda"]) for r in rows], [0.0, 0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
