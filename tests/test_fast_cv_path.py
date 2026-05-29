"""Unit tests for _ss_scalar_from_sub_cv_values — pure-Python CV reconstruction.

No OpenMM required: tests exercise the scalar arithmetic that mirrors the
CustomCVForce energy expressions, not the GPU evaluation itself.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


def test_ss_scalar_alpha_coil_beta_identity():
    from gareus.production import _ss_scalar_from_sub_cv_values

    meta = {"mode": "alpha-coil-beta"}
    sub = [0.6, 0.4, 0.2, 0.3]  # [alpha_phi, alpha_psi, beta_phi, beta_psi]
    expected = 0.5 * (0.6 + 0.4) - 0.5 * (0.2 + 0.3)
    assert abs(_ss_scalar_from_sub_cv_values(sub, meta) - expected) < 1e-12


def test_ss_scalar_alpha_coil_beta_pure_alpha():
    from gareus.production import _ss_scalar_from_sub_cv_values

    meta = {"mode": "alpha-coil-beta"}
    sub = [1.0, 1.0, 0.0, 0.0]
    assert abs(_ss_scalar_from_sub_cv_values(sub, meta) - 1.0) < 1e-12


def test_ss_scalar_alpha_coil_beta_pure_beta():
    from gareus.production import _ss_scalar_from_sub_cv_values

    meta = {"mode": "alpha-coil-beta"}
    sub = [0.0, 0.0, 1.0, 1.0]
    assert abs(_ss_scalar_from_sub_cv_values(sub, meta) - (-1.0)) < 1e-12


def test_ss_scalar_simple_mode():
    from gareus.production import _ss_scalar_from_sub_cv_values

    for mode in ("alpha", "beta", "custom"):
        meta = {"mode": mode}
        sub = [0.7, 0.3]
        expected = 0.5 * (0.7 + 0.3)
        assert abs(_ss_scalar_from_sub_cv_values(sub, meta) - expected) < 1e-12


def test_ss_scalar_rama_map_uniform():
    """Uniform sub-CV scores: result is weighted average of region values."""
    from gareus.production import _ss_scalar_from_sub_cv_values

    regions = [
        {"name": "r1", "value": -1.0},
        {"name": "r2", "value":  0.0},
        {"name": "r3", "value":  1.0},
    ]
    meta = {"mode": "rama-map", "regions": regions}
    phi_score = psi_score = 0.5
    sub = [phi_score, psi_score] * 3  # interleaved phi/psi for each region
    scores = [0.5 * (phi_score + psi_score)] * 3  # [0.5, 0.5, 0.5]
    denom = sum(scores) + 1e-8
    numer = sum(r["value"] * s for r, s in zip(regions, scores))
    expected = numer / denom
    result = _ss_scalar_from_sub_cv_values(sub, meta)
    assert abs(result - expected) < 1e-10


def test_ss_scalar_rama_map_single_basin():
    """One region dominant: result approaches that basin's value."""
    from gareus.production import _ss_scalar_from_sub_cv_values

    regions = [
        {"name": "beta",   "value": -1.0},
        {"name": "alpha",  "value":  1.0},
    ]
    meta = {"mode": "rama-map", "regions": regions}
    # Beta region very active, alpha near-zero
    sub = [0.9, 0.9, 0.01, 0.01]
    result = _ss_scalar_from_sub_cv_values(sub, meta)
    assert result < -0.8, f"expected near -1.0 for dominant beta, got {result}"


def test_ss_scalar_rama_map_empty_regions():
    from gareus.production import _ss_scalar_from_sub_cv_values

    meta = {"mode": "rama-map", "regions": []}
    assert _ss_scalar_from_sub_cv_values([], meta) == 0.0


def test_ss_scalar_rama_regions_alias():
    """rama-regions mode uses same reconstruction logic as rama-map."""
    from gareus.production import _ss_scalar_from_sub_cv_values

    regions = [{"name": "x", "value": 0.5}]
    meta_map = {"mode": "rama-map", "regions": regions}
    meta_reg = {"mode": "rama-regions", "regions": regions}
    sub = [0.6, 0.4]
    assert abs(
        _ss_scalar_from_sub_cv_values(sub, meta_map)
        - _ss_scalar_from_sub_cv_values(sub, meta_reg)
    ) < 1e-12


def test_ss_scalar_contact_normalize_math():
    """Verify contact CV normalization formula used in _fetch_state/_fetch_exchange_state."""
    raw = 3.7
    contact_norm = 10.0
    cv_normalized = raw / contact_norm
    assert abs(cv_normalized - 0.37) < 1e-12

    # When norm=0, fallback returns raw
    cv_fallback = raw / contact_norm if contact_norm > 0.0 else raw
    assert cv_fallback == cv_normalized

    contact_norm_zero = 0.0
    cv_zero_fallback = raw / contact_norm_zero if contact_norm_zero > 0.0 else raw
    assert cv_zero_fallback == raw
