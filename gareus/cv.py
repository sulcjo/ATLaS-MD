"""
Collective variable (CV) helpers.

This module contains a collection of functions for working with
primary and secondary collective variables used in the Gareus umbrella
sampling workflow.  These helpers were originally defined in
``gareus_peptide.py`` but have been separated here to improve
organisation and reusability.  The functions cover primary CV
selection (distance versus nonlocal contacts), contact‑pair
construction, unit conversions, and secondary‑CV support.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Iterable, Tuple, List, Dict, Any

import numpy as np

from .constants import WATER_RESNAMES, ION_RESNAMES
from .units import (
    kcal_a2_to_kj_nm2,
    kcal_to_kj,
)
from .state import cv_distance_from_positions_nm
from .math_helpers import _batch_torsion_angles_rad, _mean_torsion_score_from_angles

__all__ = [
    "primary_cv_mode",
    "primary_cv_is_contacts",
    "primary_cv_is_distance",
    "primary_cv_label",
    "primary_cv_units",
    "primary_k_units",
    "primary_openmm_k_units",
    "primary_center_to_openmm_value",
    "primary_k_to_openmm_value",
    "contact_us_pull_k_user",
    "primary_delta_label",
    "primary_cv_axis_label",
    "primary_cv_unit_suffix",
    "format_primary_cv_value",
    "primary_cv_format_value",
    "format_primary_delta_value",
    "secondary_cv_mode",
    "secondary_cv_enabled",
    "secondary_cv_is_transition",
    "secondary_cv_range",
    "build_tica_linear_metadata",
    "secondary_cv_target_angles",
    "secondary_structure_torsions",
    "rama_map_definitions",
    "secondary_structure_score_from_positions_nm",
    "adaptive_secondary_force_constants_kcal",
    "contact_scheme",
    "contact_atom_allowed",
    "select_contact_atoms",
    "peptide_residues",
    "find_atom_in_residue",
    "parse_atom_selector",
    "choose_cv_atoms",
    "_contact_pair_indices",
    "_contact_pair_weight",
    "contact_normalization_denominator",
    "contact_pair_summary",
    "build_nonlocal_contact_pairs",
    "nonlocal_contact_cv_from_positions_nm",
    "prepare_primary_cv_definition",
    "apply_primary_cv_metadata_to_args",
    "primary_cv_value_from_positions_nm",
]

# ---------------------------------------------------------------------------
# Primary CV helpers
# ---------------------------------------------------------------------------

def primary_cv_mode(args_or_mode) -> str:
    """Return the canonical primary umbrella CV mode.

    The primary CV may be specified explicitly via a ``primary_cv`` key or
    attribute, or implicitly via the generic ``mode`` key when it matches
    one of the supported names.  Supported values are ``distance`` and
    ``nonlocal-contacts``; a number of historical aliases are accepted.
    """
    if isinstance(args_or_mode, str):
        mode = args_or_mode
    elif isinstance(args_or_mode, dict):
        if args_or_mode.get("primary_cv") not in (None, ""):
            mode = args_or_mode.get("primary_cv")
        else:
            maybe_mode = str(args_or_mode.get("mode", "distance") or "distance").strip().lower().replace("_", "-")
            mode = maybe_mode if maybe_mode in {"distance", "nonlocal-contacts"} else "distance"
    else:
        mode = getattr(args_or_mode, "primary_cv", "distance")
    mode = str(mode or "distance").strip().lower().replace("_", "-")
    aliases = {
        "terminal-distance": "distance",
        "terminal": "distance",
        "contacts": "nonlocal-contacts",
        "nonlocal-contact": "nonlocal-contacts",
        "nonlocal-contact-fraction": "nonlocal-contacts",
        "contact-fraction": "nonlocal-contacts",
    }
    return aliases.get(mode, mode)


def primary_cv_is_contacts(args_or_mode) -> bool:
    """Return True if the selected primary CV mode is ``nonlocal-contacts``."""
    return primary_cv_mode(args_or_mode) == "nonlocal-contacts"


def primary_cv_is_distance(args_or_mode) -> bool:
    """Return True if the selected primary CV mode is ``distance``."""
    return primary_cv_mode(args_or_mode) == "distance"


def primary_cv_label(args_or_mode) -> str:
    """Return a human‑readable label for the selected primary CV."""
    if isinstance(args_or_mode, dict) and args_or_mode.get("label"):
        return str(args_or_mode.get("label"))
    if primary_cv_is_contacts(args_or_mode):
        normalized = True
        if not isinstance(args_or_mode, str):
            normalized = bool(getattr(args_or_mode, "contact_normalize", True))
            if isinstance(args_or_mode, dict):
                normalized = bool(args_or_mode.get("contact_normalize", normalized))
        return "nonlocal contact fraction" if normalized else "nonlocal contact count"
    return "terminal distance"


def primary_cv_units(args_or_mode) -> str:
    """Return the user‑facing units for the primary CV.

    Contact modes are dimensionless fractions by default.  Distance mode
    uses Angstrom units (``A``).
    """
    if isinstance(args_or_mode, dict) and args_or_mode.get("units"):
        return str(args_or_mode.get("units"))
    if primary_cv_is_contacts(args_or_mode):
        normalized = True
        if not isinstance(args_or_mode, str):
            normalized = bool(getattr(args_or_mode, "contact_normalize", True))
            if isinstance(args_or_mode, dict):
                normalized = bool(args_or_mode.get("contact_normalize", normalized))
        return "dimensionless" if normalized else "count"
    return "A"


def primary_k_units(args_or_mode) -> str:
    """Return the user‑facing units for the primary umbrella force constant."""
    if isinstance(args_or_mode, dict) and args_or_mode.get("k_units"):
        return str(args_or_mode.get("k_units"))
    if primary_cv_is_contacts(args_or_mode):
        return "kcal/mol/CV^2"
    return "kcal/mol/A^2"


def primary_openmm_k_units(args_or_mode) -> str:
    """Return the OpenMM units for the primary umbrella force constant."""
    if isinstance(args_or_mode, dict) and args_or_mode.get("openmm_k_units"):
        return str(args_or_mode.get("openmm_k_units"))
    if primary_cv_is_contacts(args_or_mode):
        return "kJ/mol/CV^2"
    return "kJ/mol/nm^2"


def primary_center_to_openmm_value(center: float, args_or_mode) -> float:
    """Convert a user‑facing umbrella center into an OpenMM global parameter.

    For distance CVs the user units are Å, so multiply by 0.1 to obtain nm.
    For nonlocal contacts the center is dimensionless and returned unchanged.
    """
    if primary_cv_is_contacts(args_or_mode):
        return float(center)
    return float(center) * 0.1


def primary_k_to_openmm_value(k_user: float, args_or_mode) -> float:
    """Convert a user‑facing umbrella force constant to an OpenMM global parameter.

    For distance CVs the user units are kcal/mol/Å²; convert to kJ/mol/nm².
    For nonlocal contacts the user units are kcal/mol/CV²; convert to kJ/mol/CV².
    """
    if primary_cv_is_contacts(args_or_mode):
        return kcal_to_kj(k_user)
    return kcal_a2_to_kj_nm2(k_user)


def contact_us_pull_k_user(args, requested_legacy_k: float) -> float:
    """Return a contact-safe starting-pull k in kcal/mol/CV^2.

    ``--us-pull-k-kcal-a2`` is a historical distance-CV name.  In contact mode a
    value copied from a distance setup (for example 80 kcal/mol/A^2) can be far
    too impulsive when applied to a bounded, many-term contact fraction.  Prefer
    an explicit contact-specific value; otherwise softly cap the legacy value to
    a conservative maximum for pre-production pulling.  Production umbrella k
    values are not affected by this setup-only helper.
    """
    explicit = getattr(args, "contact_us_pull_k_kcal", None)
    if explicit is not None:
        try:
            return max(0.0, float(explicit))
        except Exception:
            pass
    requested = max(0.0, float(requested_legacy_k))
    cap = float(getattr(args, "contact_us_pull_max_k_kcal", 20.0) or 20.0)
    if math.isfinite(cap) and cap > 0.0:
        return min(requested, cap)
    return requested


def primary_delta_label(args_or_mode) -> str:
    """Return a short label for a primary‑CV deviation (``delta_CV`` or ``delta_A``)."""
    return "delta_CV" if primary_cv_is_contacts(args_or_mode) else "delta_A"


def primary_cv_axis_label(args_or_mode, include_legacy: bool = False) -> str:
    """Return a compact axis label for plots and reports.

    In contact mode the dimensionless fraction label is shown explicitly even
    when legacy column names refer to Angstrom distances.  Set
    ``include_legacy`` to True to append the historical column name.
    """
    label = primary_cv_label(args_or_mode)
    units = primary_cv_units(args_or_mode)
    if units and units != "dimensionless":
        label = f"{label} ({units})"
    elif units == "dimensionless":
        label = f"{label} (dimensionless)"
    if include_legacy:
        label += " [legacy cv_A]"
    return label


def primary_cv_unit_suffix(args_or_mode) -> str:
    """Return a space‑prefixed unit suffix for value formatting."""
    units = primary_cv_units(args_or_mode)
    if units in {"", "dimensionless"}:
        return ""
    return f" {units}"


def format_primary_cv_value(value: float, args_or_mode, precision: int = 3) -> str:
    """Format a primary CV value for display."""
    try:
        v = float(value)
    except Exception:
        return "n/a"
    if not math.isfinite(v):
        return "n/a"
    suffix = primary_cv_unit_suffix(args_or_mode)
    return f"{v:.{int(precision)}f}{suffix}"


def primary_cv_format_value(value: float, args_or_mode, precision: int = 3) -> str:
    """Backward‑compatible alias for :func:`format_primary_cv_value`."""
    return format_primary_cv_value(value, args_or_mode, precision=precision)


def format_primary_delta_value(value: float, args_or_mode, precision: int = 3) -> str:
    """Format a primary CV deviation for display."""
    return format_primary_cv_value(value, args_or_mode, precision=precision)

# ---------------------------------------------------------------------------
# Secondary CV helpers
# ---------------------------------------------------------------------------

def secondary_cv_mode(args_or_mode) -> str:
    """Return the canonical secondary‑CV mode name.

    Accepted aliases are kept intentionally short for command‑line use, but
    all metadata is written with the canonical name so resume and analysis
    do not have to remember every spelling humans invent under caffeine.
    """
    if isinstance(args_or_mode, str):
        mode = args_or_mode
    elif isinstance(args_or_mode, dict):
        if args_or_mode.get("secondary_cv") not in (None, ""):
            mode = args_or_mode.get("secondary_cv")
        elif args_or_mode.get("secondary_cv_mode") not in (None, ""):
            mode = args_or_mode.get("secondary_cv_mode")
        else:
            mode = args_or_mode.get("mode", "none")
    else:
        mode = getattr(args_or_mode, "secondary_cv", "none")
    mode = str(mode or "none").strip().lower().replace("_", "-")
    aliases = {
        "acb": "alpha-coil-beta",
        "alpha-beta": "alpha-coil-beta",
        "alpha-to-beta": "alpha-coil-beta",
        "alpha/beta": "alpha-coil-beta",
        "transition": "alpha-coil-beta",
        "ss-transition": "alpha-coil-beta",
        "rama": "rama-map",
        "rama-map": "rama-map",
        "rama-basin-map": "rama-map",
        "tica": "tica-linear",
        "bootstrap-torsion": "torsion-pca",
        "bootstrap-linear": "torsion-pca",
        "torsion-linear": "torsion-pca",
        "torsion-pca": "torsion-pca",
    }
    return aliases.get(mode, mode)


def secondary_cv_enabled(args) -> bool:
    """Return True if any secondary CV other than ``none`` is selected."""
    return secondary_cv_mode(args) != "none"


def secondary_cv_is_transition(args_or_mode) -> bool:
    """Return True if the secondary CV is a transition coordinate."""
    return secondary_cv_mode(args_or_mode) in {"alpha-coil-beta", "rama-map", "tica-linear", "torsion-pca"}


def secondary_cv_range(args_or_mode) -> Tuple[float, float]:
    """Return the valid scalar range for the selected secondary CV."""
    mode = secondary_cv_mode(args_or_mode)
    if mode in {"tica-linear", "torsion-pca"}:
        return (-6.0, 6.0)
    return (-1.0, 1.0) if secondary_cv_is_transition(args_or_mode) else (0.0, 1.0)


def build_tica_linear_metadata(
    *,
    enabled: bool,
    n_phi: int = 0,
    n_psi: int = 0,
    tica_state_path: str = "",
) -> dict:
    """Return a secondary-CV metadata dict for the tica-linear mode.

    Used by ``add_secondary_structure_cv_force`` and stored in
    ``umbrella_pymbar_metadata.json`` so downstream analysis knows the CV
    definition used for each epoch.
    """
    return {
        "enabled": bool(enabled),
        "mode": "tica-linear",
        "label": "inter-epoch tICA linear CV (tIC1, slowest mode)",
        "range_min": -6.0,
        "range_max": 6.0,
        "n_phi_torsions": int(n_phi),
        "n_psi_torsions": int(n_psi),
        "tica_state_path": str(tica_state_path),
        "note": (
            "tIC1 projection: (X - mean) @ weights. Weights are updated each epoch via "
            "inter-epoch tICA refitting of backbone sin/cos dihedral features. "
            "Samples from different tica_cv_version epochs must not be mixed in MBAR."
        ),
    }



# ---------------------------------------------------------------------------
# Secondary CV topology/target helpers
# ---------------------------------------------------------------------------

def _find_optional_atom_in_residue(residue, atom_name: str) -> Optional[int]:
    try:
        return find_atom_in_residue(residue, atom_name)
    except Exception:
        return None

def secondary_cv_target_angles(args) -> tuple[float, float, str]:
    """Return target phi/psi angles in radians for simple content CVs."""
    mode = secondary_cv_mode(args)
    if mode == "alpha":
        return math.radians(-60.0), math.radians(-45.0), "alpha-like phi/psi content"
    if mode == "beta":
        return math.radians(-135.0), math.radians(135.0), "beta-like phi/psi content"
    if mode == "custom":
        return math.radians(float(getattr(args, "secondary_cv_phi0_deg", -60.0))), math.radians(float(getattr(args, "secondary_cv_psi0_deg", -45.0))), "custom phi/psi content"
    if mode == "alpha-coil-beta":
        return 0.0, 0.0, "alpha-coil-beta transition coordinate"
    return 0.0, 0.0, "disabled"

def secondary_structure_torsions(topology) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Return phi and psi backbone torsions for canonical peptide residues.

    The score intentionally ignores terminal missing torsions and any residue with
    missing backbone atoms.  Proline/glycine are still included if the atoms exist;
    the CV is a geometric secondary-structure proxy, not a DSSP classifier.
    """
    residues = peptide_residues(topology)
    phi: list[tuple[int, int, int, int]] = []
    psi: list[tuple[int, int, int, int]] = []
    for i, res in enumerate(residues):
        n = _find_optional_atom_in_residue(res, "N")
        ca = _find_optional_atom_in_residue(res, "CA")
        c = _find_optional_atom_in_residue(res, "C")
        if n is None or ca is None or c is None:
            continue
        if i > 0:
            c_prev = _find_optional_atom_in_residue(residues[i - 1], "C")
            if c_prev is not None:
                phi.append((int(c_prev), int(n), int(ca), int(c)))
        if i + 1 < len(residues):
            n_next = _find_optional_atom_in_residue(residues[i + 1], "N")
            if n_next is not None:
                psi.append((int(n), int(ca), int(c), int(n_next)))
    return phi, psi

def rama_map_definitions() -> List[Dict[str, Any]]:
    """Built-in explicit Ramachandran basin map for the ``rama-map`` CV.

    ``rama-map`` is still represented as one dimensionless secondary CV so it can
    be crossed with the primary umbrella coordinate without creating a full 3D
    distance x phi x psi grid.  Each center corresponds to a named Ramachandran
    basin that should be reported
    and interpreted as a map target: beta/extended, PPII/coil, right-alpha, and
    left-alpha.  The left-alpha basin is intentionally included as a native
    target rather than hidden inside a generic coil value.
    """
    return [
        {"name": "beta", "label": "beta/extended", "phi_deg": -135.0, "psi_deg": 135.0, "value": -1.0},
        {"name": "ppii", "label": "PPII/coil", "phi_deg": -75.0, "psi_deg": 145.0, "value": -1.0 / 3.0},
        {"name": "alpha_r", "label": "right-alpha", "phi_deg": -60.0, "psi_deg": -45.0, "value": 1.0 / 3.0},
        {"name": "alpha_l", "label": "left-alpha", "phi_deg": 60.0, "psi_deg": 40.0, "value": 1.0},
    ]


def snap_to_rama_basins(value: float) -> float:
    """Snap *value* to the nearest named rama-map basin scalar.

    Basin scalars are the softmax output values defined in
    ``rama_map_definitions()``: β=-1, PPII=-1/3, α_R=1/3, α_L=+1.
    Useful for constraining adaptive-feedback proposals to physically
    meaningful window centers instead of arbitrary inter-basin positions.
    """
    basins = [d["value"] for d in rama_map_definitions()]
    return min(basins, key=lambda b: abs(b - float(value)))


def _mean_torsion_scores_from_angles(angles: np.ndarray, targets_rad: np.ndarray, sigma_rad: float) -> np.ndarray:
    """Vectorized version of _mean_torsion_score_from_angles for many targets."""
    targets = np.asarray(targets_rad, dtype=np.float64)
    if targets.size == 0:
        return np.empty(0, dtype=np.float64)
    if len(angles) == 0:
        return np.zeros(targets.size, dtype=np.float64)
    ang = np.asarray(angles, dtype=np.float64)[:, np.newaxis]
    denom = max(1.0e-12, float(sigma_rad) * float(sigma_rad))
    scores = np.exp(-(1.0 - np.cos(ang - targets[np.newaxis, :])) / denom)
    valid = np.isfinite(scores)
    counts = valid.sum(axis=0)
    sums = np.where(valid, scores, 0.0).sum(axis=0)
    out = np.zeros(targets.size, dtype=np.float64)
    good = counts > 0
    out[good] = sums[good] / counts[good]
    return out


def _ensure_secondary_cv_numeric_cache(ss_info: Dict[str, Any], mode: str) -> float:
    """Populate immutable NumPy/radian caches used by secondary-CV scoring."""
    if "_np_phi_idx" not in ss_info:
        phi_t = ss_info.get("phi_torsions", []) or []
        psi_t = ss_info.get("psi_torsions", []) or []
        ss_info["_np_phi_idx"] = np.array(phi_t, dtype=np.int32).reshape(-1, 4) if phi_t else np.empty((0, 4), dtype=np.int32)
        ss_info["_np_psi_idx"] = np.array(psi_t, dtype=np.int32).reshape(-1, 4) if psi_t else np.empty((0, 4), dtype=np.int32)
    if "_np_sigma_rad" not in ss_info:
        ss_info["_np_sigma_rad"] = math.radians(float(ss_info.get("sigma_deg", 35.0)))
    sigma = float(ss_info["_np_sigma_rad"])
    if mode == "alpha-coil-beta" and "_np_alpha_beta_phi_targets" not in ss_info:
        ss_info["_np_alpha_beta_phi_targets"] = np.asarray([
            math.radians(float(ss_info.get("alpha_phi0_deg", -60.0))),
            math.radians(float(ss_info.get("beta_phi0_deg", -135.0))),
        ], dtype=np.float64)
        ss_info["_np_alpha_beta_psi_targets"] = np.asarray([
            math.radians(float(ss_info.get("alpha_psi0_deg", -45.0))),
            math.radians(float(ss_info.get("beta_psi0_deg", 135.0))),
        ], dtype=np.float64)
    elif mode == "rama-map" and "_np_rama_region_values" not in ss_info:
        values = []
        phi_targets = []
        psi_targets = []
        for region in (ss_info.get("regions") or rama_map_definitions()):
            try:
                values.append(float(region["value"]))
                phi_targets.append(math.radians(float(region["phi_deg"])))
                psi_targets.append(math.radians(float(region["psi_deg"])))
            except Exception:
                continue
        ss_info["_np_rama_region_values"] = np.asarray(values, dtype=np.float64)
        ss_info["_np_rama_phi_targets"] = np.asarray(phi_targets, dtype=np.float64)
        ss_info["_np_rama_psi_targets"] = np.asarray(psi_targets, dtype=np.float64)
    elif mode not in {"alpha-coil-beta", "rama-map", "tica-linear", "torsion-pca"} and "_np_simple_phi_psi_targets" not in ss_info:
        ss_info["_np_simple_phi_psi_targets"] = np.asarray([
            math.radians(float(ss_info.get("phi0_deg", -60.0))),
            math.radians(float(ss_info.get("psi0_deg", -45.0))),
        ], dtype=np.float64)
    return sigma


def secondary_structure_score_from_positions_nm(positions_nm, ss_info: Optional[Dict[str, Any]]) -> float:
    """Compute the same smooth secondary-structure score used by the OpenMM CV.

    The function mutates ``ss_info`` lazily by caching NumPy torsion index arrays
    and radian target arrays under private keys.  The cached targets avoid repeated
    degree/radian conversion and the multi-basin modes score all targets in one
    NumPy pass.
    """
    if not ss_info or not ss_info.get("enabled"):
        return float("nan")
    mode = secondary_cv_mode(ss_info)
    sigma = _ensure_secondary_cv_numeric_cache(ss_info, mode)

    if mode in {"tica-linear", "torsion-pca"}:
        from .tica import backbone_dihedral_features

        weights = np.asarray(ss_info.get("weights", []), dtype=np.float64)
        feats = backbone_dihedral_features(
            np.asarray(positions_nm, dtype=np.float64),
            [tuple(map(int, quart)) for quart in ss_info.get("phi_torsions", [])],
            [tuple(map(int, quart)) for quart in ss_info.get("psi_torsions", [])],
        )
        if weights.shape != feats.shape:
            raise ValueError(f"{mode} weights shape {weights.shape} does not match feature shape {feats.shape}")
        offset = float(ss_info.get("tica_offset", 0.0))
        return float(feats @ weights + offset)

    phi_angles = _batch_torsion_angles_rad(positions_nm, ss_info["_np_phi_idx"])
    psi_angles = _batch_torsion_angles_rad(positions_nm, ss_info["_np_psi_idx"])
    has_phi = len(phi_angles) > 0
    has_psi = len(psi_angles) > 0

    if mode == "alpha-coil-beta":
        phi_scores = _mean_torsion_scores_from_angles(phi_angles, ss_info["_np_alpha_beta_phi_targets"], sigma)
        psi_scores = _mean_torsion_scores_from_angles(psi_angles, ss_info["_np_alpha_beta_psi_targets"], sigma)
        # score arrays are [alpha, beta]
        return float(0.5 * (phi_scores[0] + psi_scores[0]) - 0.5 * (phi_scores[1] + psi_scores[1]))

    if mode == "rama-map":
        values = ss_info["_np_rama_region_values"]
        if values.size == 0:
            return 0.0
        phi_scores = _mean_torsion_scores_from_angles(phi_angles, ss_info["_np_rama_phi_targets"], sigma)
        psi_scores = _mean_torsion_scores_from_angles(psi_angles, ss_info["_np_rama_psi_targets"], sigma)
        if has_phi and has_psi:
            scores = 0.5 * (phi_scores + psi_scores)
        elif has_phi:
            scores = phi_scores
        elif has_psi:
            scores = psi_scores
        else:
            scores = np.zeros_like(values)
        valid = np.isfinite(scores)
        denom = float(np.sum(scores[valid])) if np.any(valid) else 0.0
        if denom <= 1.0e-12:
            return 0.0
        numer = float(np.sum(values[valid] * scores[valid]))
        return float(max(-1.0, min(1.0, numer / denom)))

    phi0, psi0 = ss_info["_np_simple_phi_psi_targets"]
    phi_mean = _mean_torsion_score_from_angles(phi_angles, float(phi0), sigma) if has_phi else None
    psi_mean = _mean_torsion_score_from_angles(psi_angles, float(psi0), sigma) if has_psi else None
    if phi_mean is None and psi_mean is None:
        return float("nan")
    if phi_mean is not None and psi_mean is not None:
        return 0.5 * (phi_mean + psi_mean)
    return float(phi_mean if phi_mean is not None else psi_mean)


# MBAR-disconnection guard-rail: below this analytic flat-PMF neighbor
# overlap, adjacent umbrella windows can no longer be stitched by MBAR.
MBAR_DISCONNECT_OVERLAP_FLOOR = 0.03


def _implied_neighbor_overlap_from_k(k_kcal: float, spacing: float, rt_kcal_mol: float) -> float:
    """Return the analytic flat-PMF neighbor overlap implied by ``k``.

    Two adjacent equal-variance Gaussian umbrella windows with
    ``sigma_eff = sqrt(RT / k)`` and centers separated by ``spacing`` overlap
    (flat-PMF approximation) by ``erfc(d / (2*sqrt(2)))`` with
    ``d = spacing / sigma_eff``.  Returns ``nan`` for non-finite/non-positive
    ``k`` or ``spacing`` -- an unrestrained (``k=0``) window is not evaluated.
    """
    if not (math.isfinite(k_kcal) and k_kcal > 0.0):
        return float("nan")
    if not (math.isfinite(spacing) and spacing > 0.0):
        return float("nan")
    sigma_eff = math.sqrt(rt_kcal_mol / k_kcal)
    if not (math.isfinite(sigma_eff) and sigma_eff > 0.0):
        return float("nan")
    d = spacing / sigma_eff
    return math.erfc(d / (2.0 * math.sqrt(2.0)))


def _warn_on_subfloor_clamped_overlap(
    offenders: List[Tuple[float, float, float, float]],
    axis_label: str,
    floor: float = MBAR_DISCONNECT_OVERLAP_FLOOR,
) -> None:
    """Emit one aggregated warning for clamp-induced sub-floor overlap.

    ``offenders`` is a list of ``(center, spacing, k_clamped, overlap)``
    tuples for windows whose post-clamp force constant implies a neighbor
    overlap below ``floor`` -- the point at which MBAR can no longer stitch
    neighboring windows together.  Emits a single warning per call (not
    per-frame) listing every offending window.
    """
    if not offenders:
        return
    detail = "; ".join(
        f"center={c:.4f} (spacing={sp:.4f}): k_clamped={k:.3f} kcal/mol/CV^2 "
        f"-> overlap={ov:.4f}"
        for c, sp, k, ov in offenders
    )
    warnings.warn(
        f"{axis_label} umbrella window(s) MBAR-DISCONNECTED from a neighbor "
        f"(overlap < floor {floor:g}) because the [min_k, max_k] clamp left "
        f"the force constant too high for the local spacing: {detail}. "
        f"Loosen this axis's overlap_sigma / lower k_max / lower k_min so "
        f"the clamp does not over-restrain it.",
        UserWarning,
        stacklevel=2,
    )


def adaptive_secondary_force_constants_kcal(centers: Iterable[float], args) -> List[float]:
    """Return secondary‑CV umbrella force constants in kcal/mol/CV².

    In ``fixed`` mode this preserves the historical behaviour: one
    user‑provided ``--secondary-cv-k-kcal`` value is repeated for every
    secondary‑CV target.  In ``spacing``/``adaptive`` mode it mirrors the
    primary distance‑CV adaptive‑k idea, but in dimensionless CV units:

        sigma_CV = local_secondary_spacing / overlap_sigma
        k_CV     = kBT / sigma_CV²

    The extra secondary‑specific overlap/min/max/scale knobs exist because the
    phi/psi content CV is dimensionless and its useful stiffness scale is not
    directly comparable to kcal/mol/Å² for the terminal‑distance umbrella.
    """
    vals = [float(x) for x in list(centers or [])]
    if not vals:
        return []
    base = float(getattr(args, "secondary_cv_k_kcal", 50.0) or 50.0)
    if base < 0.0:
        raise ValueError("--secondary-cv-k-kcal must be non-negative")
    mode = str(getattr(args, "secondary_cv_k_mode", "fixed") or "fixed").strip().lower()
    if mode in {"fixed", "constant"} or len(vals) < 2:
        return [float(base)] * len(vals)
    if mode not in {"spacing", "adaptive"}:
        raise ValueError(f"Unsupported --secondary-cv-k-mode {mode!r}; use fixed or spacing")
    centers_arr = np.asarray(vals, dtype=float)
    # Use sorted unique coordinates for spacing, then map the local spacing back
    # to each input centre.  This preserves rectangular-grid generation order and
    # also handles repeated secondary centres in explicit/advisory tables.
    unique = sorted({round(float(x), 8) for x in centers_arr if math.isfinite(float(x))})
    if len(unique) < 2:
        return [float(base)] * len(vals)
    unique_arr = np.asarray(unique, dtype=float)
    spacings = np.diff(unique_arr)
    local_unique = np.empty_like(unique_arr)
    local_unique[0] = spacings[0]
    local_unique[-1] = spacings[-1]
    if unique_arr.size > 2:
        local_unique[1:-1] = 0.5 * (spacings[:-1] + spacings[1:])
    overlap_sigma = float(getattr(args, "secondary_cv_adaptive_overlap_sigma", 0.0) or 0.0)
    if overlap_sigma <= 0.0:
        overlap_sigma = float(getattr(args, "adaptive_overlap_sigma", 1.25) or 1.25)
    overlap_sigma = max(0.05, overlap_sigma)
    min_sigma = max(1.0e-6, float(getattr(args, "secondary_cv_adaptive_min_sigma", 0.02) or 0.02))
    min_k = max(0.0, float(getattr(args, "secondary_cv_adaptive_min_k_kcal", 0.0) or 0.0))
    max_k = max(min_k, float(getattr(args, "secondary_cv_adaptive_max_k_kcal", 500.0) or 500.0))
    scale = max(0.0, float(getattr(args, "secondary_cv_adaptive_k_scale", 1.0) or 1.0))
    rt_kcal_mol = 0.00198720425864083 * float(getattr(args, "temperature_k", 300.0) or 300.0)
    k_by_center: Dict[float, float] = {}
    offenders: List[Tuple[float, float, float, float]] = []
    for c, local in zip(unique_arr, local_unique):
        sigma_cv = max(min_sigma, float(local) / overlap_sigma)
        k_val = scale * rt_kcal_mol / (sigma_cv * sigma_cv)
        k_clamped = float(max(min_k, min(max_k, k_val)))
        k_by_center[round(float(c), 8)] = k_clamped
        overlap = _implied_neighbor_overlap_from_k(k_clamped, float(local), rt_kcal_mol)
        if math.isfinite(overlap) and overlap < MBAR_DISCONNECT_OVERLAP_FLOOR:
            offenders.append((float(c), float(local), k_clamped, overlap))
    _warn_on_subfloor_clamped_overlap(offenders, "CV2 (secondary)")
    return [float(k_by_center.get(round(float(c), 8), base)) for c in centers_arr]

# ---------------------------------------------------------------------------
# Contact selection and nonlocal contact CV
# ---------------------------------------------------------------------------

def _atom_is_hydrogen(atom) -> bool:
    """Return True if an atom is hydrogen based on its element or name."""
    try:
        if atom.element is not None and atom.element.symbol == "H":
            return True
    except Exception:
        pass
    return str(getattr(atom, "name", "")).upper().startswith("H")


def contact_scheme(args_or_scheme) -> str:
    """Return the canonical nonlocal‑contact construction scheme.

    Supported schemes are ``atom-pairs``, ``residue-balanced``, and ``ca-pairs``.
    A number of historical aliases are accepted.
    """
    if isinstance(args_or_scheme, str):
        scheme = args_or_scheme
    elif isinstance(args_or_scheme, dict):
        scheme = args_or_scheme.get("contact_scheme", "atom-pairs")
    else:
        scheme = getattr(args_or_scheme, "contact_scheme", "atom-pairs")
    scheme = str(scheme or "atom-pairs").strip().lower().replace("_", "-")
    aliases = {
        "atom": "atom-pairs",
        "atoms": "atom-pairs",
        "atom-pair": "atom-pairs",
        "pair": "atom-pairs",
        "pairs": "atom-pairs",
        "residue": "residue-balanced",
        "residue-pairs": "residue-balanced",
        "residue-balanced-pairs": "residue-balanced",
        "residue-heavy": "residue-balanced",
        "balanced": "residue-balanced",
        "ca": "ca-pairs",
        "ca-pair": "ca-pairs",
    }
    return aliases.get(scheme, scheme)


_BACKBONE_HEAVY_ATOM_NAMES = {"N", "CA", "C", "O", "OXT"}
_BACKBONE_HYDROGEN_ATOM_NAMES = {"H", "H1", "H2", "H3", "HA", "HA2", "HA3"}


def contact_atom_allowed(atom, mode: str) -> bool:
    """Return True if an atom is eligible for nonlocal contacts under the selection mode.

    ``mode`` may be ``all``, ``heavy``, ``ca``, ``backbone-heavy``,
    ``sidechain-heavy``, or ``sidechain-all``.  In all cases the atom must
    belong to the peptide.
    """
    mode = str(mode or "heavy").strip().lower().replace("_", "-")
    name = str(getattr(atom, "name", ""))
    if mode == "all":
        return True
    if mode == "heavy":
        return not _atom_is_hydrogen(atom)
    if mode == "ca":
        return name == "CA"
    if mode == "backbone-heavy":
        return name in {"N", "CA", "C", "O"}
    if mode == "sidechain-heavy":
        return (not _atom_is_hydrogen(atom)) and name not in _BACKBONE_HEAVY_ATOM_NAMES
    if mode == "sidechain-all":
        return name not in _BACKBONE_HEAVY_ATOM_NAMES and name not in _BACKBONE_HYDROGEN_ATOM_NAMES
    raise ValueError(f"Unsupported --contact-atom-selection {mode!r}")


def peptide_residues(topology) -> List[Any]:
    """Return a list of peptide residues excluding water and ions."""
    residues = []
    for res in topology.residues():
        if res.name not in WATER_RESNAMES and res.name.upper() not in ION_RESNAMES:
            residues.append(res)
    return residues


def solute_atom_indices(topology) -> List[int]:
    """Return atom indices of solute (non-water, non-ion) residues, ascending.

    Used to record solute-only trajectories (``--traj-solute-only``): for a tiny
    peptide in a large water box this shrinks frames ~100x. The ascending order
    matches OpenMM's ``atomSubset`` output and a Modeller-deleted topology PDB,
    so the saved trajectory and its companion ``solute_only.pdb`` stay aligned.
    """
    idx = [int(atom.index) for res in peptide_residues(topology) for atom in res.atoms()]
    return sorted(idx)


def find_atom_in_residue(residue, atom_name: str) -> int:
    """Return the index of the atom with ``atom_name`` in the given residue."""
    for atom in residue.atoms():
        if atom.name == atom_name:
            return atom.index
    raise ValueError(f"Atom {atom_name!r} not found in residue {residue.index + 1} {residue.name}")


def parse_atom_selector(selector: str, residues: Iterable[Any]) -> int:
    """Parse a human‑readable atom selector of the form ``res:atom`` or an absolute index.

    Selector format examples: ``'1:CA'``, ``'-1:CA'``, ``'10:C'``, or an absolute integer
    atom index as a string.  Negative residue numbers count from the end.
    """
    selector = str(selector).strip()
    if selector.isdigit():
        return int(selector)
    if ":" not in selector:
        raise ValueError(f"Bad atom selector {selector!r}; use e.g. 1:CA or -1:CA")
    res_text, atom_name = selector.split(":", 1)
    res_i = int(res_text)
    res_list = list(residues)
    if res_i < 0:
        residue = res_list[res_i]
    else:
        residue = res_list[res_i - 1]
    return find_atom_in_residue(residue, atom_name)


def choose_cv_atoms(topology, args) -> Tuple[int, int, str]:
    """Select the two atoms defining a distance CV based on command‑line options."""
    residues = peptide_residues(topology)
    if len(residues) < 2:
        raise ValueError("Could not identify at least two peptide residues in the topology.")
    if getattr(args, "cv_atom1", None) and getattr(args, "cv_atom2", None):
        a1 = parse_atom_selector(args.cv_atom1, residues)
        a2 = parse_atom_selector(args.cv_atom2, residues)
        label = f"{args.cv_atom1}--{args.cv_atom2}"
        return a1, a2, label
    if getattr(args, "cv_mode", None) == "terminal-ca":
        return find_atom_in_residue(residues[0], "CA"), find_atom_in_residue(residues[-1], "CA"), "terminal CA--CA"
    if getattr(args, "cv_mode", None) == "terminal-n-c":
        return find_atom_in_residue(residues[0], "N"), find_atom_in_residue(residues[-1], "C"), "terminal N--C"
    raise ValueError(f"Unknown --cv-mode {getattr(args, 'cv_mode', None)!r}")


def select_contact_atoms(topology, args) -> List[Tuple[int, int]]:
    """Return selected solute peptide atoms as (atom_index, peptide_residue_index)."""
    residues = peptide_residues(topology)
    scheme = contact_scheme(args)
    mode = "ca" if scheme == "ca-pairs" else str(getattr(args, "contact_atom_selection", "heavy") or "heavy")
    selected: List[Tuple[int, int]] = []
    for ri, residue in enumerate(residues):
        for atom in residue.atoms():
            if contact_atom_allowed(atom, mode):
                selected.append((int(atom.index), int(ri)))
    if not selected:
        raise ValueError(f"No atoms selected for nonlocal contacts with --contact-atom-selection {mode!r}")
    return selected


def _contact_pair_indices(pair: Tuple[Any, ...]) -> Tuple[int, int]:
    """Extract the atom indices from a contact pair record."""
    return int(pair[0]), int(pair[1])


def _contact_pair_weight(pair: Tuple[Any, ...]) -> float:
    """Extract the weight from a contact pair record (defaults to 1.0)."""
    try:
        return float(pair[2])
    except Exception:
        return 1.0


def contact_normalization_denominator(contact_pairs: Iterable[Tuple[Any, ...]], args) -> float:
    """Return the normalization factor for a weighted contact sum."""
    pairs = list(contact_pairs)
    if not pairs:
        return float("nan")
    if not bool(getattr(args, "contact_normalize", True)):
        return 1.0
    # For atom‑pairs this is the number of atom pairs.  For residue‑balanced this
    # is the number of eligible residue pairs because all atom pairs belonging to
    # one residue pair have weights summing to one.
    total_weight = float(sum(_contact_pair_weight(p) for p in pairs))
    return max(1.0e-12, total_weight)


def contact_pair_summary(contact_pairs: Iterable[Tuple[Any, ...]]) -> Dict[str, Any]:
    """Return summary statistics for a collection of contact pairs."""
    pairs = list(contact_pairs)
    weights = [float(_contact_pair_weight(p)) for p in pairs]
    total_weight = float(sum(weights)) if weights else 0.0
    unique_weight_count = len({round(w, 12) for w in weights}) if weights else 0
    return {
        "n_contact_terms": int(len(pairs)),
        "contact_weight_sum": float(total_weight),
        "n_effective_contact_pairs": float(total_weight),
        "min_contact_weight": float(min(weights)) if weights else float("nan"),
        "max_contact_weight": float(max(weights)) if weights else float("nan"),
        "n_unique_contact_weights": int(unique_weight_count),
    }


def build_nonlocal_contact_pairs(topology, args) -> List[Tuple[int, int, float]]:
    """Select weighted reference‑free nonlocal contact terms.

    Returned terms are (atom_i, atom_j, weight).  In atom‑pairs/ca‑pairs mode
    the weight is 1.0 for every eligible atom pair.  In residue‑balanced mode,
    all selected atom pairs connecting the same residue pair share total
    weight 1.0, making the normalized CV a residue‑pair contact fraction
    rather than an atom‑pair contact fraction.
    """
    atoms = select_contact_atoms(topology, args)
    min_sep = max(1, int(getattr(args, "contact_min_sequence_separation", 4) or 4))
    scheme = contact_scheme(args)
    if scheme not in {"atom-pairs", "residue-balanced", "ca-pairs"}:
        raise ValueError(f"Unsupported --contact-scheme {scheme!r}; use atom-pairs, residue-balanced, or ca-pairs")
    pairs: List[Tuple[int, int, float]]
    if scheme == "residue-balanced":
        grouped: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for idx, (ai, ri) in enumerate(atoms):
            for aj, rj in atoms[idx + 1:]:
                if abs(int(ri) - int(rj)) >= min_sep:
                    key = (min(int(ri), int(rj)), max(int(ri), int(rj)))
                    grouped.setdefault(key, []).append((int(ai), int(aj)))
        pairs = []
        for _res_pair, atom_pairs in sorted(grouped.items()):
            if not atom_pairs:
                continue
            w = 1.0 / float(len(atom_pairs))
            pairs.extend((int(i), int(j), float(w)) for i, j in atom_pairs)
    else:
        pairs = []
        for idx, (ai, ri) in enumerate(atoms):
            for aj, rj in atoms[idx + 1:]:
                if abs(int(ri) - int(rj)) >= min_sep:
                    pairs.append((int(ai), int(aj), 1.0))
    if not pairs:
        raise ValueError(
            "No nonlocal contact pairs selected.  Lower --contact-min-sequence-separation "
            "or use a broader --contact-atom-selection."
        )
    warn_threshold = int(getattr(args, "contact_pair_warning_threshold", 5000) or 5000)
    if len(pairs) > warn_threshold:
        print(
            f"WARNING: selected {len(pairs)} weighted nonlocal contact terms; "
            "consider --contact-scheme residue-balanced, --contact-atom-selection ca, or backbone-heavy for speed."
        )
    return pairs


def nonlocal_contact_cv_from_positions_nm(positions_nm: np.ndarray, contact_pairs: Iterable[Tuple[int, int, float]], args) -> float:
    """Compute the smooth nonlocal contact fraction/count from positions in nm."""
    pairs = list(contact_pairs)
    if not pairs:
        return float("nan")
    r0_nm = float(getattr(args, "contact_r0_a", 4.5) or 4.5) * 0.1
    beta_nm_inv = float(getattr(args, "contact_beta_a_inv", 6.0) or 6.0) * 10.0
    total = 0.0
    for pair in pairs:
        i, j = _contact_pair_indices(pair)
        weight = _contact_pair_weight(pair)
        rij_nm = float(np.linalg.norm(positions_nm[int(i)] - positions_nm[int(j)]))
        x = beta_nm_inv * (rij_nm - r0_nm)
        # Numerically stable logistic contact switch.  The algebraic identity
        #   1/(1+exp(x)) == 0.5*(1-tanh(0.5*x))
        # avoids exp() overflow and matches the OpenMM force expression used
        # for the biased contact CV.
        val = 0.5 * (1.0 - math.tanh(0.5 * x))
        total += weight * val
    if bool(getattr(args, "contact_normalize", True)):
        return float(total / contact_normalization_denominator(pairs, args))
    return float(total)


def prepare_primary_cv_definition(
    topology,
    args,
    cv_atom1: Optional[int] = None,
    cv_atom2: Optional[int] = None,
    cv_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Build metadata and cached atom/pair information for the selected primary CV."""
    mode = primary_cv_mode(args)
    if mode == "distance":
        if cv_atom1 is None or cv_atom2 is None:
            cv_atom1, cv_atom2, cv_label = choose_cv_atoms(topology, args)
        return {
            "mode": "distance",
            "label": str(cv_label or "terminal distance"),
            "units": "A",
            "k_units": "kcal/mol/A^2",
            "openmm_k_units": "kJ/mol/nm^2",
            "cv_atom1": int(cv_atom1),
            "cv_atom2": int(cv_atom2),
            "contact_pairs": [],
            "n_contact_pairs": 0,
        }
    if mode == "nonlocal-contacts":
        pairs = build_nonlocal_contact_pairs(topology, args)
        pair_summary = contact_pair_summary(pairs)
        scheme = contact_scheme(args)
        label_scheme = "residue-balanced " if scheme == "residue-balanced" else "CA " if scheme == "ca-pairs" else ""
        _r0_nm = float(getattr(args, "contact_r0_a", 4.5) or 4.5) * 0.1
        _beta_nm_inv = float(getattr(args, "contact_beta_a_inv", 6.0) or 6.0) * 10.0
        _idx_arr = np.array([[int(p[0]), int(p[1])] for p in pairs], dtype=np.int32)
        _weights_arr = np.array([_contact_pair_weight(p) for p in pairs], dtype=np.float64)
        _norm_denom = contact_normalization_denominator(pairs, args)
        normalized = bool(getattr(args, "contact_normalize", True))
        return {
            "mode": "nonlocal-contacts",
            "label": (label_scheme + "nonlocal contact fraction") if normalized else (label_scheme + "nonlocal contact count"),
            "units": "dimensionless" if normalized else "count",
            "k_units": "kcal/mol/CV^2",
            "openmm_k_units": "kJ/mol/CV^2",
            "cv_atom1": int(cv_atom1) if cv_atom1 is not None else None,
            "cv_atom2": int(cv_atom2) if cv_atom2 is not None else None,
            "distance_cv_label_for_compatibility": str(cv_label or "terminal distance"),
            "contact_pairs": pairs,
            "n_contact_pairs": int(len(pairs)),
            "n_contact_terms": int(pair_summary.get("n_contact_terms", len(pairs))),
            "n_effective_contact_pairs": float(pair_summary.get("n_effective_contact_pairs", len(pairs))),
            "contact_weight_sum": float(pair_summary.get("contact_weight_sum", len(pairs))),
            "contact_weight_min": float(pair_summary.get("min_contact_weight", 1.0)),
            "contact_weight_max": float(pair_summary.get("max_contact_weight", 1.0)),
            "contact_scheme": scheme,
            "contact_atom_selection": str(getattr(args, "contact_atom_selection", "heavy")),
            "contact_min_sequence_separation": int(getattr(args, "contact_min_sequence_separation", 4) or 4),
            "contact_r0_A": float(getattr(args, "contact_r0_a", 4.5) or 4.5),
            "contact_beta_A_inv": float(getattr(args, "contact_beta_a_inv", 6.0) or 6.0),
            "contact_switch_r0_nm": _r0_nm,
            "contact_beta_nm_inv": _beta_nm_inv,
            "contact_normalize": normalized,
            "_np_idx_i": _idx_arr[:, 0],
            "_np_idx_j": _idx_arr[:, 1],
            "_np_weights": _weights_arr,
            "_np_r0_nm": _r0_nm,
            "_np_beta_nm_inv": _beta_nm_inv,
            "_np_norm_denom": _norm_denom,
        }
    raise ValueError(f"Unsupported --primary-cv {mode!r}")


def apply_primary_cv_metadata_to_args(args, metadata: Optional[Dict[str, Any]]) -> None:
    """Restore contact‑CV settings from previous run metadata for --resume.

    True production resume skips window generation, but the System still has to be
    rebuilt with the exact same contact‑pair selection and switching parameters
    as the original run before Context.loadCheckpoint() is attempted.  This
    helper copies those durable settings from gareus_metadata.json /
    checkpoint metadata into args when available.
    """
    if not isinstance(metadata, dict):
        return
    primary_meta = metadata.get("primary_cv_definition") or metadata.get("primary_cv") or metadata
    if not isinstance(primary_meta, dict):
        return
    mode = primary_cv_mode(primary_meta)
    if mode != "nonlocal-contacts":
        return
    args.primary_cv = "nonlocal-contacts"
    mapping = {
        "contact_scheme": "contact_scheme",
        "contact_atom_selection": "contact_atom_selection",
        "contact_min_sequence_separation": "contact_min_sequence_separation",
        "contact_r0_A": "contact_r0_a",
        "contact_beta_A_inv": "contact_beta_a_inv",
        "contact_normalize": "contact_normalize",
    }
    for src_key, arg_key in mapping.items():
        if src_key in primary_meta and primary_meta[src_key] is not None:
            setattr(args, arg_key, primary_meta[src_key])


def primary_cv_value_from_positions_nm(
    positions_nm: np.ndarray,
    primary_cv_def: Dict[str, Any],
    args,
) -> float:
    """Return the primary CV in user‑facing units.

    For distance CVs the return value is in Å; for nonlocal contacts the
    return value is either a dimensionless fraction or a raw count.
    """
    mode = primary_cv_mode(primary_cv_def)
    if mode == "distance":
        return 10.0 * cv_distance_from_positions_nm(positions_nm, int(primary_cv_def["cv_atom1"]), int(primary_cv_def["cv_atom2"]))
    if mode == "nonlocal-contacts":
        if "_np_idx_i" in primary_cv_def:
            pos = np.asarray(positions_nm, dtype=np.float64)
            diff = pos[primary_cv_def["_np_idx_i"]] - pos[primary_cv_def["_np_idx_j"]]
            rij = np.sqrt((diff * diff).sum(axis=1))
            x = primary_cv_def["_np_beta_nm_inv"] * (rij - primary_cv_def["_np_r0_nm"])
            # Stable vectorized logistic contact switch; avoids np.exp()
            # overflow and mirrors the OpenMM tanh form.
            contact_values = 0.5 * (1.0 - np.tanh(0.5 * x))
            total = float(np.dot(primary_cv_def["_np_weights"], contact_values))
            return total / primary_cv_def["_np_norm_denom"] if primary_cv_def.get("contact_normalize", True) else total
        return nonlocal_contact_cv_from_positions_nm(positions_nm, list(primary_cv_def.get("contact_pairs", [])), args)
    raise ValueError(f"Unsupported primary CV mode {mode!r}")
