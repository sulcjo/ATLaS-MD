"""Which unbiasing estimators may be selected, given how the run was biased."""

# The brief for this module names "gamd_exp" as one of the three excluded
# method literals. Grep confirms that string matches nothing anywhere in this
# codebase -- the real method is "gamd_exponential". Both are kept: "gamd_exp"
# for fidelity to the prescribed contract/test, "gamd_exponential" because it
# is the actual name that reaches choose_site_method() below and it double-
# counts the ladder boost exactly like the cumulant methods do (it reweights
# by exp(beta*boost) on top of a u_nk that already carries that boost
# exactly). Omitting it would leave gamd_exponential selectable under a
# ladder whenever `selected` is forced to it (e.g. --selected-method).
_CUMULANT_METHODS = frozenset({"gamd_cumulant2", "gamd_cumulant3", "gamd_exp", "gamd_exponential"})

# Mirrors the {'gamd_exponential','gamd_cumulant2','gamd_cumulant3'} literal
# repeated at every analyze_*() call site in analyze_gareus_mbar.py and
# gareus/mbar_analysis/pmf.py.
_GAMD_METHODS = frozenset({"gamd_exponential", "gamd_cumulant2", "gamd_cumulant3"})


def ladder_excluded_methods(gamd_ladder: bool) -> frozenset:
    """Estimators that must not be selectable for a λ-ladder run.

    Under an active ladder the GaMD boost is already carried exactly in u_nk by
    apply_ladder_boost_to_u, so every GaMD-specific unbiasing method would remove
    it a second time. Excluded, not merely out-ranked: out-ranking leaves them
    selectable whenever they happen to score better.
    """
    return _CUMULANT_METHODS if gamd_ladder else frozenset()


def choose_site_method(selected: str, excluded: frozenset,
                        candidates: frozenset = _GAMD_METHODS,
                        fallback: str = "gamd_cumulant2") -> str:
    """Pick the per-site GaMD PMF method for one analyze_*() call site.

    Mirrors the inline pattern duplicated across analyze_rg,
    analyze_distance_rg_2d_fes, analyze_pca_2d_fes, analyze_cv1_cv2_2d_fes,
    analyze_chignolin_fes, and analyze_secondary_cv_pmf::

        chosen = selected if selected in {'gamd_exponential','gamd_cumulant2','gamd_cumulant3'} else 'gamd_cumulant2'

    with one change: ``excluded`` is genuinely removed from ``candidates``
    (a set difference) before the membership test, rather than merely being
    out-ranked. That matters because ``selected`` upstream is already
    ladder-aware (``select_unbiased_method`` returns ``'umbrella_only'`` when
    a ladder is active) -- but this inline pattern's hard-coded
    ``else 'gamd_cumulant2'`` previously clobbered that back to a cumulant
    method, since it treats "selected is not one of the 3 GaMD methods" as
    "selected is invalid" without asking why.

    When ``excluded`` makes ``fallback`` itself unselectable (i.e. a ladder is
    active, since 'gamd_cumulant2' is always in ``_CUMULANT_METHODS``), the
    fallback becomes ``'umbrella_only'`` instead -- so a ladder run can never
    fall through to a cumulant/exponential correction via the fallback path
    either, even if ``selected`` were corrupted upstream (e.g. by a forced
    ``--selected-method``).

    With ``excluded == frozenset()`` (no ladder) this reduces exactly to the
    original inline expression: no behaviour change for a non-ladder run.
    """
    live = candidates - excluded
    if selected in live:
        return selected
    return fallback if fallback not in excluded else "umbrella_only"
