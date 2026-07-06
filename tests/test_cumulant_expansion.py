import numpy as np
import pytest

pytest.importorskip("analyze_gareus_mbar")

from analyze_gareus_mbar import cumulant2, cumulant2_2d, cumulant3, cumulant3_2d


BETA = 0.1  # 1/(kJ/mol), arbitrary but fixed for hand-checkable numbers
KBT_KCAL = 0.6
BINS = np.array([-1.0, 1.0])  # single CV bin; everything below falls in it


def test_cumulant2_matches_hand_computed_mean_var_logfac():
    cv = np.zeros(3)
    boost = np.array([10.0, 20.0, 30.0])
    base_w = np.ones(3)
    _pmf, diag = cumulant2(cv, base_w, boost, BINS, BETA, KBT_KCAL)
    mean = 20.0
    var = (100.0 + 0.0 + 100.0) / 3.0  # 66.6667
    expected_logfac = BETA * mean + 0.5 * BETA * BETA * var
    assert diag["boost_mean_kj"][0] == pytest.approx(mean)
    assert diag["boost_var_kj2"][0] == pytest.approx(var)
    assert diag["log_reweight_factor"][0] == pytest.approx(expected_logfac)
    assert np.all(np.isfinite(_pmf["pmf"]))


def test_cumulant3_matches_cumulant2_when_boost_is_symmetric():
    boost = np.concatenate([np.full(500, 35.0), np.full(500, 45.0)])  # symmetric about 40
    cv = np.zeros_like(boost)
    base_w = np.ones_like(boost)
    _pmf2, diag2 = cumulant2(cv, base_w, boost, BINS, BETA, KBT_KCAL)
    _pmf3, diag3 = cumulant3(cv, base_w, boost, BINS, BETA, KBT_KCAL)
    assert diag3["boost_kappa3_kj3"][0] == pytest.approx(0.0, abs=1e-6)
    assert diag3["log_reweight_factor"][0] == pytest.approx(diag2["log_reweight_factor"][0], abs=1e-9)


def test_cumulant3_adds_positive_correction_for_right_skewed_boost():
    core = np.full(990, 40.0)
    tail = np.full(10, 150.0)  # rare large-boost outliers -> positive skew, matches GaMD's right-tail-heavy ΔV
    boost = np.concatenate([core, tail])
    cv = np.zeros_like(boost)
    base_w = np.ones_like(boost)
    _pmf2, diag2 = cumulant2(cv, base_w, boost, BINS, BETA, KBT_KCAL)
    _pmf3, diag3 = cumulant3(cv, base_w, boost, BINS, BETA, KBT_KCAL)
    assert diag3["boost_kappa3_kj3"][0] > 0.0
    # CE3 = CE2 + (beta^3/6)*kappa3, so a positive kappa3 must raise logfac.
    expected = diag2["log_reweight_factor"][0] + (BETA ** 3 / 6.0) * diag3["boost_kappa3_kj3"][0]
    assert diag3["log_reweight_factor"][0] == pytest.approx(expected)
    assert diag3["log_reweight_factor"][0] > diag2["log_reweight_factor"][0]
    assert np.all(np.isfinite(_pmf3["pmf"]))


def test_cumulant_expansion_rejects_bad_order():
    from analyze_gareus_mbar import _cumulant_expansion
    with pytest.raises(ValueError):
        _cumulant_expansion(np.zeros(3), np.ones(3), np.ones(3), BINS, BETA, KBT_KCAL, order=4)


def test_cumulant3_2d_matches_cumulant2_2d_when_boost_is_symmetric():
    xbins = np.array([-1.0, 1.0])
    ybins = np.array([-1.0, 1.0])
    boost = np.concatenate([np.full(500, 35.0), np.full(500, 45.0)])
    x = np.zeros_like(boost)
    y = np.zeros_like(boost)
    base_w = np.ones_like(boost)
    _fes2, diag2 = cumulant2_2d(x, y, base_w, boost, xbins, ybins, BETA, KBT_KCAL)
    _fes3, diag3 = cumulant3_2d(x, y, base_w, boost, xbins, ybins, BETA, KBT_KCAL)
    assert diag3["boost_kappa3_kj3"][0, 0] == pytest.approx(0.0, abs=1e-6)
    assert diag3["log_reweight_factor"][0, 0] == pytest.approx(diag2["log_reweight_factor"][0, 0], abs=1e-9)


def test_cumulant3_2d_adds_positive_correction_for_right_skewed_boost():
    xbins = np.array([-1.0, 1.0])
    ybins = np.array([-1.0, 1.0])
    core = np.full(990, 40.0)
    tail = np.full(10, 150.0)
    boost = np.concatenate([core, tail])
    x = np.zeros_like(boost)
    y = np.zeros_like(boost)
    base_w = np.ones_like(boost)
    _fes2, diag2 = cumulant2_2d(x, y, base_w, boost, xbins, ybins, BETA, KBT_KCAL)
    _fes3, diag3 = cumulant3_2d(x, y, base_w, boost, xbins, ybins, BETA, KBT_KCAL)
    assert diag3["boost_kappa3_kj3"][0, 0] > 0.0
    assert diag3["log_reweight_factor"][0, 0] > diag2["log_reweight_factor"][0, 0]
    assert np.all(np.isfinite(_fes3["pmf"]))
