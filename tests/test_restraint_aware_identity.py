"""Spec P6: centre identity includes the restraint pattern; placeholder coordinates of
unrestrained axes are never identity."""
import numpy as np

import gareus.adaptive_production as ap
from gareus.adaptive_production import AdaptiveDecisionPolicy, WindowStateRegistry
from gareus.mbar_analysis.ladder_overlap import ladder_overlap_by_axis


def _pol():
    return AdaptiveDecisionPolicy()


def _state(reg, p, k1, s=None, k2=None, lam=0.0):
    obj = reg.add_state(p, k1, secondary_center=s, secondary_k=k2, gamd_lambda=lam, epoch=0, source="seed")
    sid = int(getattr(obj, "state_id", obj))
    return next(st for st in reg.active_states() if st.state_id == sid)


def test_a_cv2_only_window_is_not_the_cv1_window_at_its_placeholder():
    reg = WindowStateRegistry()
    cv2_only = _state(reg, 0.502, 0.0, 1.0, 50.0)
    restrained = _state(reg, 0.502, 800.0, 1.0, 50.0)
    pol = _pol()
    assert ap._centre_group_key(cv2_only, pol) != ap._centre_group_key(restrained, pol)
    assert ap._centre_group_key(cv2_only, pol)[0] is None


def test_cv2_only_windows_differ_only_in_cv2_whatever_their_placeholder():
    reg = WindowStateRegistry()
    a = _state(reg, 0.502, 0.0, 1.0, 50.0)
    b = _state(reg, 0.37, 0.0, 1.0, 50.0)       # different placeholder, same CV2 window
    c = _state(reg, 0.502, 0.0, -1.0, 50.0)
    pol = _pol()
    assert ap._centre_group_key(a, pol) == ap._centre_group_key(b, pol)
    assert ap._centre_group_key(a, pol) != ap._centre_group_key(c, pol)


def test_a_cv2_placeholder_with_k2_zero_is_not_identity():
    reg = WindowStateRegistry()
    a = _state(reg, 0.3, 800.0, 0.9, 0.0)
    b = _state(reg, 0.3, 800.0, -0.4, 0.0)
    assert ap._centre_group_key(a, _pol()) == ap._centre_group_key(b, _pol()) == (ap._centre_group_key(a, _pol())[0], None)


def test_legacy_states_without_recorded_k_keep_their_old_key():
    reg = WindowStateRegistry()
    st = _state(reg, 0.3, 800.0, 0.9, None)
    pol = _pol()
    p = int(round(0.3 / pol.duplicate_primary_tol))
    s = int(round(0.9 / pol.duplicate_secondary_tol))
    assert ap._centre_group_key(st, pol) == (p, s)


def test_has_near_duplicate_respects_the_restraint_pattern():
    reg = WindowStateRegistry()
    _state(reg, 0.502, 0.0, 1.0, 50.0)          # CV2-only window
    pol = _pol()
    assert reg.has_near_duplicate(0.502, 1.0, pol) is False                      # a restrained proposal
    assert reg.has_near_duplicate(0.9, 1.0, pol, primary_k=0.0, secondary_k=50.0) is True
    assert reg.has_near_duplicate(0.9, -1.0, pol, primary_k=0.0, secondary_k=50.0) is False


def test_has_near_duplicate_unchanged_for_ordinary_states():
    reg = WindowStateRegistry()
    _state(reg, 0.3, 800.0, 0.9, 50.0)
    _state(reg, 0.1, 800.0)
    pol = _pol()
    assert reg.has_near_duplicate(0.3, 0.9, pol) is True
    assert reg.has_near_duplicate(0.3, None, pol) is False
    assert reg.has_near_duplicate(0.1, None, pol) is True
    assert reg.has_near_duplicate(0.1, 0.2, pol) is False


def _pair(overlap_value):
    return lambda a, b: overlap_value


def test_ladder_overlap_separates_a_cv2_only_window_from_a_restrained_one():
    # Two rungs each of a CV1-restrained window and a CV2-only window, all at CV1 0.502.
    lam = np.array([0.0, 1.0, 0.0, 1.0])
    cen = np.array([0.502] * 4)
    sec = np.array([1.0, 1.0, 1.0, 1.0])
    k1 = np.array([800.0, 800.0, 0.0, 0.0])
    k2 = np.array([50.0] * 4)
    lo, _ = ladder_overlap_by_axis(None, lam, cen, n_k=np.ones(4), pair_overlap=_pair(0.3),
                                   secondary_centers=sec, primary_k=k1, secondary_k=k2)
    assert lo["lambda_direction"]["n_pairs"] == 2                # 0-1 and 2-3; one mixed pair before P6
    assert lo["lambda_direction"]["expected_components"] == 2


def test_ladder_overlap_ignores_a_k2_zero_placeholder():
    lam = np.array([0.0, 1.0])
    cen = np.array([0.3, 0.3])
    sec = np.array([0.9, -0.4])                  # placeholders differ
    lo, _ = ladder_overlap_by_axis(None, lam, cen, n_k=np.ones(2), pair_overlap=_pair(0.3),
                                   secondary_centers=sec, primary_k=np.array([800.0, 800.0]),
                                   secondary_k=np.array([0.0, 0.0]))
    assert lo["lambda_direction"]["n_pairs"] == 1                # still one window's two rungs


def test_ladder_overlap_without_secondary_k_is_unchanged():
    lam = np.array([0.0, 1.0])
    cen = np.array([0.3, 0.3])
    sec = np.array([0.9, -0.4])
    lo, _ = ladder_overlap_by_axis(None, lam, cen, n_k=np.ones(2), pair_overlap=_pair(0.3),
                                   secondary_centers=sec, primary_k=np.array([800.0, 800.0]))
    assert lo["lambda_direction"]["n_pairs"] == 0                # different CV2 centres
