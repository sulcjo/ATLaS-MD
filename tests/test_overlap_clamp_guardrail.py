"""Guard-rail tests: a clamped per-window umbrella force constant must not
silently collapse the neighbor overlap below the MBAR-disconnection floor.

Exercises the CV2/secondary (rama) path in ``gareus.cv`` and the CV1/primary
(contacts) path in ``gareus.windows``. Both are pure numpy/stdlib -- no
OpenMM import, per project rule.
"""
from __future__ import annotations

import types
import warnings
from pathlib import Path

import pytest

from gareus.cv import (
    MBAR_DISCONNECT_OVERLAP_FLOOR,
    _implied_neighbor_overlap_from_k,
    adaptive_secondary_force_constants_kcal,
)
from gareus.windows import adaptive_contact_force_constants_kcal

RT_300K_KCAL_MOL = 0.00198720425864083 * 300.0
RAMA_CENTERS = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]
RAMA_SPACING = 2.0 / 3.0


def _cv2_args(**overrides):
    defaults = dict(
        secondary_cv_k_kcal=50.0,
        secondary_cv_k_mode="adaptive",
        secondary_cv_adaptive_overlap_sigma=2.3,
        secondary_cv_adaptive_min_sigma=0.02,
        secondary_cv_adaptive_min_k_kcal=5.0,
        secondary_cv_adaptive_max_k_kcal=50.0,
        secondary_cv_adaptive_k_scale=1.0,
        temperature_k=300.0,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _cv1_args(**overrides):
    defaults = dict(
        contact_adaptive_k_mode="spacing",
        contact_adaptive_overlap_sigma=2.3,
        contact_adaptive_min_sigma=0.02,
        contact_adaptive_min_k_kcal=5.0,
        contact_adaptive_max_k_kcal=50.0,
        contact_adaptive_k_scale=1.0,
        contact_adaptive_default_k_kcal=25.0,
        temperature_k=300.0,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _no_subfloor_warning(records) -> bool:
    return not any("MBAR-DISCONNECTED" in str(rec.message) for rec in records)


# ---------------------------------------------------------------------------
# CV2 (secondary / rama) guard-rail
# ---------------------------------------------------------------------------

def test_cv2_warns_when_clamp_pushes_overlap_below_floor():
    """overlap_sigma=20 targets a far-too-tight sigma for the rama axis;
    even after clamping to k_max=50 the axis is MBAR-disconnected."""
    args = _cv2_args(
        secondary_cv_adaptive_overlap_sigma=20.0,
        secondary_cv_adaptive_k_scale=2.0,
        secondary_cv_adaptive_min_k_kcal=20.0,
        secondary_cv_adaptive_max_k_kcal=50.0,
    )
    with pytest.warns(UserWarning, match="CV2"):
        ks = adaptive_secondary_force_constants_kcal(RAMA_CENTERS, args)
    assert all(abs(k - 50.0) < 1e-9 for k in ks), "expected every k clamped at k_max=50"
    overlap = _implied_neighbor_overlap_from_k(ks[0], RAMA_SPACING, RT_300K_KCAL_MOL)
    assert overlap < MBAR_DISCONNECT_OVERLAP_FLOOR


def test_cv2_no_warning_with_sane_unclamped_settings():
    args = _cv2_args()  # overlap_sigma=2.3, scale=1.0, min_k=5, max_k=50
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ks = adaptive_secondary_force_constants_kcal(RAMA_CENTERS, args)
    assert _no_subfloor_warning(rec)
    overlap = _implied_neighbor_overlap_from_k(ks[0], RAMA_SPACING, RT_300K_KCAL_MOL)
    assert overlap >= 0.2


# ---------------------------------------------------------------------------
# CV1 (primary / contacts) guard-rail
# ---------------------------------------------------------------------------

def test_cv1_warns_when_clamp_pushes_overlap_below_floor():
    args = _cv1_args(
        contact_adaptive_overlap_sigma=20.0,
        contact_adaptive_k_scale=2.0,
        contact_adaptive_min_k_kcal=20.0,
        contact_adaptive_max_k_kcal=50.0,
    )
    centers = [0.0, 0.6]
    with pytest.warns(UserWarning, match="CV1"):
        ks = adaptive_contact_force_constants_kcal(centers, args)
    assert all(abs(k - 50.0) < 1e-9 for k in ks), "expected every k clamped at k_max=50"
    overlap = _implied_neighbor_overlap_from_k(ks[0], 0.6, RT_300K_KCAL_MOL)
    assert overlap < MBAR_DISCONNECT_OVERLAP_FLOOR


def test_cv1_no_warning_with_sane_unclamped_settings():
    args = _cv1_args()
    centers = [0.0, 0.6]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ks = adaptive_contact_force_constants_kcal(centers, args)
    assert _no_subfloor_warning(rec)
    overlap = _implied_neighbor_overlap_from_k(ks[0], 0.6, RT_300K_KCAL_MOL)
    assert overlap >= 0.2


# ---------------------------------------------------------------------------
# Config-level: flagship YAML rama (CV2) axis reaches a sane overlap
# ---------------------------------------------------------------------------

def test_flagship_config_cv2_overlap_at_least_point_two():
    yaml = pytest.importorskip("yaml")
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "examples" / "chignolin_fulltreatment2.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cv2_cfg = cfg["cv2"]
    args = _cv2_args(
        secondary_cv_adaptive_overlap_sigma=float(cv2_cfg["cv2_adaptive_overlap_sigma"]),
        secondary_cv_adaptive_min_k_kcal=float(cv2_cfg["cv2_k_min"]),
        secondary_cv_adaptive_max_k_kcal=float(cv2_cfg["cv2_k_max"]),
        secondary_cv_adaptive_k_scale=float(cv2_cfg["cv2_k_scale"]),
        secondary_cv_k_mode=str(cv2_cfg["cv2_k_mode"]),
    )
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ks = adaptive_secondary_force_constants_kcal(RAMA_CENTERS, args)
    assert _no_subfloor_warning(rec)
    overlap = _implied_neighbor_overlap_from_k(ks[0], RAMA_SPACING, RT_300K_KCAL_MOL)
    assert overlap >= 0.2
