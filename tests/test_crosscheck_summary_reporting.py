import unittest
from analyze_gareus_mbar import _crosscheck_summary_fields


class CrosscheckSummaryTests(unittest.TestCase):
    def test_skipped_result_still_names_its_reason_and_bin_count(self):
        out = _crosscheck_summary_fields({"status": "skipped", "reason": "gamd_ladder not active"})
        self.assertEqual(out["status"], "skipped")
        self.assertEqual(out["n_bins_compared"], 0)
        self.assertIn("reason", out)

    def test_pass_carries_bins_and_tolerance_source(self):
        out = _crosscheck_summary_fields({
            "status": "pass", "max_abs_diff_kcal": 0.21,
            "n_bins_compared": 47, "tolerance_kcal": 0.5,
            "tolerance_source": "fixed_default",
        })
        self.assertEqual(out["n_bins_compared"], 47)
        self.assertEqual(out["tolerance_source"], "fixed_default")


if __name__ == "__main__":
    unittest.main()
