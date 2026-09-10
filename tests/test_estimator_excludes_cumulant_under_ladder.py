import unittest

from gareus.mbar_analysis.estimators import ladder_excluded_methods, choose_site_method


class EstimatorExclusionTests(unittest.TestCase):
    def test_cumulant_methods_excluded_when_ladder_active(self):
        excluded = ladder_excluded_methods(gamd_ladder=True)
        self.assertIn("gamd_cumulant2", excluded)
        self.assertIn("gamd_cumulant3", excluded)
        self.assertIn("gamd_exp", excluded)

    def test_nothing_excluded_without_a_ladder(self):
        self.assertEqual(ladder_excluded_methods(gamd_ladder=False), frozenset())

    # Beyond the brief's literal snippet: the real codebase's method name is
    # "gamd_exponential", not "gamd_exp" (grep confirms "gamd_exp" matches
    # nothing anywhere in the repo). gamd_exponential double-counts the
    # ladder boost exactly like the cumulant methods do -- it reweights by
    # exp(beta*boost) on top of u_nk that already carries that boost exactly
    # -- so it must be excluded too, under its real name.
    def test_gamd_exponential_excluded_under_its_real_name(self):
        self.assertIn("gamd_exponential", ladder_excluded_methods(gamd_ladder=True))


class ChooseSiteMethodTests(unittest.TestCase):
    """choose_site_method mirrors the analyze_*() call sites' inline pattern
    `selected if selected in {3 gamd methods} else 'gamd_cumulant2'`, except the
    candidate set has ladder-excluded methods genuinely removed (set difference,
    not a re-rank) and the fallback itself is swapped to 'umbrella_only' when the
    fallback is excluded -- so a ladder run can never fall through to a cumulant
    correction from the back door either.
    """

    def test_no_ladder_matches_original_inline_ternary(self):
        excluded = ladder_excluded_methods(gamd_ladder=False)
        self.assertEqual(choose_site_method("gamd_cumulant2", excluded), "gamd_cumulant2")
        self.assertEqual(choose_site_method("gamd_exponential", excluded), "gamd_exponential")
        self.assertEqual(choose_site_method("umbrella_only", excluded), "gamd_cumulant2")
        self.assertEqual(choose_site_method("anything_else", excluded), "gamd_cumulant2")

    def test_ladder_active_never_returns_an_excluded_method(self):
        excluded = ladder_excluded_methods(gamd_ladder=True)
        for selected in ("umbrella_only", "gamd_exponential", "gamd_cumulant2", "gamd_cumulant3"):
            self.assertEqual(choose_site_method(selected, excluded), "umbrella_only")


if __name__ == "__main__":
    unittest.main()
