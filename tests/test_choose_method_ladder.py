"""Whole-branch review, IMPORTANT 1: _choose_method (analyze_gareus_mbar.py,
used for contact FESs/torsions/Ramachandran/scalar observables) was the one
selection site left as a raw ternary, never routed through
choose_site_method/ladder_excluded_methods like the other six sites. Under an
active ladder with an explicit --selected-method gamd_cumulant2 it kept
returning a cumulant method while the six converted sites had already fallen
back to umbrella_only -- a three-way split in what estimator a single run
reports.
"""
import unittest

from analyze_gareus_mbar import _choose_method
from gareus.mbar_analysis.estimators import ladder_excluded_methods


class ChooseMethodNonLadderUnaffectedTests(unittest.TestCase):
    """excluded=frozenset() (the default, and what a non-ladder run passes)
    must reproduce today's behaviour exactly."""

    def test_boost_not_ok_is_always_umbrella_only(self):
        for selected in ("gamd_cumulant2", "gamd_exponential", "umbrella_only", "bogus"):
            self.assertEqual(_choose_method(selected, False), "umbrella_only")

    def test_boost_ok_selected_gamd_method_is_kept(self):
        for selected in ("gamd_exponential", "gamd_cumulant2", "gamd_cumulant3"):
            self.assertEqual(_choose_method(selected, True), selected)

    def test_boost_ok_selected_umbrella_only_is_kept(self):
        self.assertEqual(_choose_method("umbrella_only", True), "umbrella_only")

    def test_boost_ok_invalid_selected_falls_back_to_cumulant2(self):
        self.assertEqual(_choose_method("not_a_real_method", True), "gamd_cumulant2")

    def test_default_excluded_matches_explicit_empty_frozenset(self):
        self.assertEqual(
            _choose_method("gamd_cumulant2", True),
            _choose_method("gamd_cumulant2", True, frozenset()),
        )


class ChooseMethodLadderExcludedTests(unittest.TestCase):
    """Under an active ladder, this site must agree with the six already-
    converted sites: an explicit --selected-method gamd_cumulant2 must NOT
    reach a cumulant/exponential estimator here either."""

    def setUp(self):
        self.excluded = ladder_excluded_methods(True)

    def test_forced_cumulant2_falls_back_to_umbrella_only(self):
        self.assertEqual(_choose_method("gamd_cumulant2", True, self.excluded), "umbrella_only")

    def test_forced_cumulant3_falls_back_to_umbrella_only(self):
        self.assertEqual(_choose_method("gamd_cumulant3", True, self.excluded), "umbrella_only")

    def test_forced_exponential_falls_back_to_umbrella_only(self):
        self.assertEqual(_choose_method("gamd_exponential", True, self.excluded), "umbrella_only")

    def test_umbrella_only_selected_stays_umbrella_only(self):
        self.assertEqual(_choose_method("umbrella_only", True, self.excluded), "umbrella_only")

    def test_agrees_with_the_six_converted_sites_choose_site_method(self):
        from gareus.mbar_analysis.estimators import choose_site_method
        for selected in ("gamd_cumulant2", "gamd_exponential", "gamd_cumulant3", "umbrella_only", "bogus"):
            other_sites = choose_site_method(selected, self.excluded)
            this_site = _choose_method(selected, True, self.excluded)
            self.assertEqual(this_site, other_sites,
                              f"selected={selected!r}: this site={this_site!r} vs other sites={other_sites!r}")


if __name__ == "__main__":
    unittest.main()
