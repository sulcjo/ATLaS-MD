import numpy as np
import pytest
from gareus.cv_selection.anchor import AnchorCandidate, rank_anchors, score_anchor


def _cand(kind, values, **definition):
    definition = definition or {"placeholder": True}
    return AnchorCandidate(kind, definition, np.asarray(values, dtype=float))


def test_r7_like_contact_range_is_not_deployable_and_says_why():
    rng = np.random.default_rng(0)
    s = score_anchor(_cand("nonlocal-contact-fraction", rng.uniform(0.0, 0.069, 5000), r0_angstrom=12.0),
                     temperature_k=300.0, k_max_kcal=1200.0, min_windows=4)
    assert s.n_resolvable == 2 and s.deployable is False and s.forceable is True
    assert any("resolvable" in r for r in s.reasons)


def test_dynamic_range_is_invariant_to_units_while_n_resolvable_is_not():
    rng = np.random.default_rng(1)
    rg_nm = rng.uniform(0.55, 1.10, 5000)
    nm = score_anchor(_cand("radius-of-gyration", rg_nm), temperature_k=300.0, k_max_kcal=1200.0, min_windows=4)
    ang = score_anchor(_cand("radius-of-gyration", rg_nm * 10.0), temperature_k=300.0, k_max_kcal=1200.0, min_windows=4)
    assert np.isclose(nm.dynamic_range, ang.dynamic_range)
    assert ang.n_resolvable > 5 * nm.n_resolvable


def test_an_unforceable_kind_is_scored_but_never_deployable():
    rng = np.random.default_rng(2)
    s = score_anchor(_cand("radius-of-gyration", rng.uniform(0.55, 1.10, 5000)),
                     temperature_k=300.0, k_max_kcal=1200.0, min_windows=4)
    assert s.forceable is False and s.deployable is False
    assert any("no runtime force" in r for r in s.reasons)


def test_native_derived_kind_is_refused_before_scoring():
    with pytest.raises(ValueError, match="native-blind"):
        score_anchor(_cand("rmsd-to-native-pdb", np.linspace(0, 1, 10)),
                     temperature_k=300.0, k_max_kcal=1200.0, min_windows=1)
