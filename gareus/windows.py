"""
Window planning helpers.

This module contains a handful of functions related to umbrella
window expansion and thermodynamic state assignment.  The functions
defined here were originally part of ``gareus_peptide.py`` but have
been extracted to improve modularity.  In particular, functions for
combining primary and secondary collective variables into a 2D grid
and for setting OpenMM global parameters on a per‑window basis are
provided.
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Iterable, Dict, Any
from pathlib import Path

import csv
import math
import numpy as np

from .cv import (
    peptide_residues,
    primary_cv_is_contacts,
    primary_cv_label,
    primary_cv_mode,
    primary_cv_units,
    primary_cv_value_from_positions_nm,
    prepare_primary_cv_definition,
    contact_scheme,
    secondary_cv_enabled,
    secondary_cv_mode,
    secondary_cv_is_transition,
    secondary_cv_range,
    adaptive_secondary_force_constants_kcal,
)
from .state import cv_distance_nm, _scalar_to_float
from .io import write_json, _json_ready
from .progress import GuiProgressSink, release_openmm_contexts
from .system_setup import (
    create_system,
    make_langevin_integrator,
    platform_summary,
    run_steps_safely,
    setup_platform_and_properties,
    write_state_pdb,
)

__all__ = [
    "expand_windows_for_secondary_cv",
    "set_window",
    "build_explicit_2d_neighbor_edges",
    "estimate_terminal_cv_envelope_a",
    "run_contact_cv_autocalibration",
    "broadcast_contact_k_for_centers",
    "run_adaptive_prescan",
    "choose_windows",
]



def _csv_first_present(row: dict, names: list[str]):
    """Return the first non-empty CSV field among several accepted aliases."""
    for name in names:
        if name in row:
            value = row.get(name)
            if value is not None and str(value).strip() != "":
                return value
    return None


def _csv_float_field(row: dict, names: list[str], row_number: int, label: str, default=None) -> float:
    value = _csv_first_present(row, names)
    if value is None:
        if default is not None:
            return float(default)
        raise ValueError(f"{label} missing in explicit 2D window CSV row {row_number}; accepted columns: {', '.join(names)}")
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"{label} in explicit 2D window CSV row {row_number} must be numeric; got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{label} in explicit 2D window CSV row {row_number} must be finite; got {value!r}")
    return out


def _unique_axis_values(values, ndigits: int = 4) -> list[float]:
    out = sorted({round(float(x), int(ndigits)) for x in values if math.isfinite(float(x))})
    return [float(x) for x in out]



def expand_windows_for_secondary_cv(
    args,
    centers_a: Iterable[float],
    k_list: Iterable[float],
) -> Tuple[np.ndarray, List[float], Optional[np.ndarray], Optional[List[float]], Dict[str, Any]]:
    """Optionally cross primary‑CV windows with secondary‑CV centres.

    With ``--secondary-cv-centers`` containing more than one value, this
    creates a simple 2D umbrella grid: every primary‑CV centre is paired
    with every secondary‑structure target.  With one centre, all primary
    windows receive the same secondary restraint.  If no secondary CV is
    enabled, the input centres and force constants are returned unchanged.

    Returns a tuple of (primary_centers, primary_k_values, secondary_centers,
    secondary_k_values, metadata).
    """
    centers = np.asarray(list(centers_a), dtype=float)
    k_primary = [float(x) for x in list(k_list)]
    if not secondary_cv_enabled(args):
        return centers, k_primary, None, None, {"enabled": False}
    mode = secondary_cv_mode(args)
    transition_mode = secondary_cv_is_transition(mode)
    raw_centers = getattr(args, "secondary_cv_centers", None)
    if raw_centers is None or len(raw_centers) == 0:
        single_center = getattr(args, "secondary_cv_center", None)
        if single_center is None:
            default_center = 0.0 if transition_mode else 0.65
        else:
            default_center = float(single_center)
        ss_centers_in = [float(default_center)]
    else:
        ss_centers_in = [float(x) for x in raw_centers]
    range_min, range_max = secondary_cv_range(args)
    ss_centers_in = [max(range_min, min(range_max, float(x))) for x in ss_centers_in]
    ss_k_values_in = adaptive_secondary_force_constants_kcal(ss_centers_in, args)
    ss_k_mode = str(getattr(args, "secondary_cv_k_mode", "fixed") or "fixed").strip().lower()
    if len(ss_centers_in) == 1:
        ss_centers = np.asarray([ss_centers_in[0]] * len(centers), dtype=float)
        ss_k_list = [float(ss_k_values_in[0])] * len(centers)
        meta = {
            "enabled": True,
            "mode": mode,
            "grid": False,
            "input_centers": ss_centers_in,
            "input_k_kcal_mol": [float(x) for x in ss_k_values_in],
            "secondary_cv_k_mode": ss_k_mode,
            "range_min": float(range_min),
            "range_max": float(range_max),
            "n_primary_windows": int(len(centers)),
            "n_secondary_centers": 1,
            "n_total_windows": int(len(centers)),
        }
        return centers, k_primary, ss_centers, ss_k_list, meta
    # Build a full 2D grid by crossing each primary window with each secondary target.
    expanded_centers: List[float] = []
    expanded_k: List[float] = []
    expanded_ss_centers: List[float] = []
    expanded_ss_k: List[float] = []
    for c, k in zip(centers, k_primary):
        for ss0, ss_k in zip(ss_centers_in, ss_k_values_in):
            expanded_centers.append(float(c))
            expanded_k.append(float(k))
            expanded_ss_centers.append(float(ss0))
            expanded_ss_k.append(float(ss_k))
    meta = {
        "enabled": True,
        "mode": mode,
        "grid": True,
        "input_centers": ss_centers_in,
        "input_k_kcal_mol": [float(x) for x in ss_k_values_in],
        "secondary_cv_k_mode": ss_k_mode,
        "range_min": float(range_min),
        "range_max": float(range_max),
        "n_primary_windows": int(len(centers)),
        "n_secondary_centers": int(len(ss_centers_in)),
        "n_total_windows": int(len(expanded_centers)),
        "note": "2D grid formed by crossing each primary-CV window with each secondary-structure target.",
    }
    return (
        np.asarray(expanded_centers, dtype=float),
        expanded_k,
        np.asarray(expanded_ss_centers, dtype=float),
        expanded_ss_k,
        meta,
    )


def set_window(
    context: Any,
    centers_nm: Iterable[float],
    ks_kj_nm2: Iterable[float],
    window_index: int,
    secondary_centers: Optional[Iterable[float]] = None,
    secondary_ks_kj: Optional[Iterable[float]] = None,
) -> None:
    """Assign OpenMM global parameters for the given replica ``window_index``.

    ``centers_nm`` and ``ks_kj_nm2`` are the umbrella centres and force
    constants in OpenMM units (nm and kJ/mol/nm² for distance CVs, or
    dimensionless contact units and kJ/mol/CV² for contact CVs).  If
    ``secondary_centers`` and ``secondary_ks_kj`` are provided, those values
    are also assigned on the current context for the optional secondary CV.
    """
    # centres_nm/ks_kj_nm2 are historical names.  In nonlocal‑contact mode they
    # hold dimensionless contact centres and kJ/mol/CV², respectively.
    # Convert iterables to lists so indices can be taken multiple times.
    centers_list = list(centers_nm)
    k_list = list(ks_kj_nm2)
    context.setParameter("r0", float(centers_list[int(window_index)]))
    context.setParameter("k", float(k_list[int(window_index)]))
    if secondary_centers is not None and secondary_ks_kj is not None:
        try:
            sc_list = list(secondary_centers)
            sk_list = list(secondary_ks_kj)
            context.setParameter("ss0", float(sc_list[int(window_index)]))
            context.setParameter("ss_k", float(sk_list[int(window_index)]))
        except Exception:
            # The system may not contain the optional secondary CV force.
            pass



# Adaptive window-count/force-constant helpers.
def _adaptive_window_aggressiveness_settings(args) -> dict:
    """Return knobs that bias automatic/adaptive-feedback window selection.

    balanced preserves the old behavior. More aggressive modes deliberately
    prefer fewer umbrella windows: wider initial spacing, lower effective
    overlap targets, fewer additions, more pruning, and broader k proposals.
    This trades statistical comfort for a smaller replica ladder; the output
    JSON records the effective values so the choice is auditable.
    """
    mode = str(getattr(args, "adaptive_window_aggressiveness", "balanced") or "balanced").lower()
    table = {
        "conservative": {
            "initial_spacing_factor": 0.85,
            "target_overlap_factor": 1.15,
            "minimum_valid_overlap_factor": 0.60,
            "minimum_valid_overlap_floor": 0.10,
            "minimum_valid_overlap_ceiling": 0.18,
            "severe_overlap_factor": 0.38,
            "high_overlap_offset": 0.30,
            "high_overlap_factor": 1.85,
            "very_high_overlap_offset": 0.45,
            "max_add_fraction": 0.40,
            "max_add_cap": 5,
            "max_remove_fraction": 0.20,
            "max_remove_cap": 2,
            "remove_block_radius": 1,
            "min_center_hit_for_removal": 0.18,
            "min_new_spacing_A": 0.35,
            "coverage_gap_ratio": 0.28,
            "coverage_gap_width_A": 1.20,
            "k_overlap_factor": 1.90,
            "feedback_min_k_floor": 0.30,
            "max_k_factor": 1.35,
            "window_count_weight": 0.035,
        },
        "balanced": {
            "initial_spacing_factor": 1.00,
            "target_overlap_factor": 1.00,
            "minimum_valid_overlap_factor": 0.50,
            "minimum_valid_overlap_floor": 0.08,
            "minimum_valid_overlap_ceiling": 0.15,
            "severe_overlap_factor": 0.33,
            "high_overlap_offset": 0.25,
            "high_overlap_factor": 1.65,
            "very_high_overlap_offset": 0.40,
            "max_add_fraction": 0.34,
            "max_add_cap": 4,
            "max_remove_fraction": 0.25,
            "max_remove_cap": 3,
            "remove_block_radius": 1,
            "min_center_hit_for_removal": 0.15,
            "min_new_spacing_A": 0.35,
            "coverage_gap_ratio": 0.25,
            "coverage_gap_width_A": 1.25,
            "k_overlap_factor": 1.75,
            "feedback_min_k_floor": 0.30,
            "max_k_factor": 1.50,
            "window_count_weight": 0.05,
        },
        "aggressive": {
            "initial_spacing_factor": 1.35,
            "target_overlap_factor": 0.80,
            "minimum_valid_overlap_factor": 0.42,
            "minimum_valid_overlap_floor": 0.055,
            "minimum_valid_overlap_ceiling": 0.12,
            "severe_overlap_factor": 0.25,
            "high_overlap_offset": 0.16,
            "high_overlap_factor": 1.35,
            "very_high_overlap_offset": 0.28,
            "max_add_fraction": 0.20,
            "max_add_cap": 3,
            "max_remove_fraction": 0.40,
            "max_remove_cap": 5,
            "remove_block_radius": 1,
            "min_center_hit_for_removal": 0.08,
            "min_new_spacing_A": 0.50,
            "coverage_gap_ratio": 0.18,
            "coverage_gap_width_A": 1.60,
            "k_overlap_factor": 1.35,
            "feedback_min_k_floor": 0.05,
            "max_k_factor": 2.00,
            "window_count_weight": 0.12,
        },
        "very-aggressive": {
            "initial_spacing_factor": 1.75,
            "target_overlap_factor": 0.65,
            "minimum_valid_overlap_factor": 0.35,
            "minimum_valid_overlap_floor": 0.04,
            "minimum_valid_overlap_ceiling": 0.10,
            "severe_overlap_factor": 0.20,
            "high_overlap_offset": 0.10,
            "high_overlap_factor": 1.20,
            "very_high_overlap_offset": 0.20,
            "max_add_fraction": 0.12,
            "max_add_cap": 2,
            "max_remove_fraction": 0.55,
            "max_remove_cap": 8,
            "remove_block_radius": 0,
            "min_center_hit_for_removal": 0.03,
            "min_new_spacing_A": 0.75,
            "coverage_gap_ratio": 0.12,
            "coverage_gap_width_A": 2.00,
            "k_overlap_factor": 1.10,
            "feedback_min_k_floor": 0.02,
            "max_k_factor": 3.00,
            "window_count_weight": 0.20,
        },
    }
    if mode not in table:
        mode = "balanced"
    out = dict(table[mode])
    out["mode"] = mode
    user_weight = _scalar_to_float(getattr(args, "adaptive_window_count_weight", None))
    if user_weight is not None and math.isfinite(float(user_weight)) and float(user_weight) >= 0.0:
        out["window_count_weight"] = float(user_weight)
    return out

def _secondary_center_count_for_total_budget(args) -> int:
    """Return current configured secondary-axis multiplicity for total-window budgeting.

    A rectangular 2D run expands primary centers x secondary centers.  Historical
    max-window options applied only to the primary axis, which could silently turn
    ``max_windows=7`` into ``7*len(secondary_cv_centers)`` replicas.  This helper
    lets adaptive generation honor an overall replica/window budget instead.
    """
    try:
        if not secondary_cv_enabled(args):
            return 1
    except Exception:
        return 1
    centers = getattr(args, "secondary_cv_centers", None)
    if centers is None or len(centers) == 0:
        return 1
    vals = []
    for x in centers:
        try:
            v = float(x)
            if math.isfinite(v):
                vals.append(round(v, 6))
        except Exception:
            pass
    return max(1, len(sorted(set(vals))))

def _adaptive_total_window_limits(args) -> tuple[int, int]:
    """Return (min_total, max_total) for expanded 1D/2D window count.

    ``adaptive_*_windows`` and ``contact_adaptive_*_windows`` are legacy axis
    counts.  These total-window options are the preferred user-facing budget for
    2D workflows because they refer to the actual number of replicas that will be
    constructed after crossing primary and secondary CV centers.
    """
    def _ival(name: str) -> int:
        try:
            return max(0, int(getattr(args, name, 0) or 0))
        except Exception:
            return 0
    min_total = _ival("adaptive_min_total_windows")
    max_total = _ival("adaptive_max_total_windows")
    if primary_cv_is_contacts(args):
        min_total = _ival("contact_adaptive_min_total_windows") or min_total
        max_total = _ival("contact_adaptive_max_total_windows") or max_total
    return int(min_total), int(max_total)

def _axis_count_bounds_from_total_window_budget(
    args,
    n_secondary: int,
    legacy_min_axis: int,
    legacy_max_axis: int,
    label: str = "primary",
) -> tuple[int, int, dict]:
    """Convert total 2D replica limits to primary-axis center-count bounds."""
    n_secondary = max(1, int(n_secondary or 1))
    legacy_min_axis = max(2, int(legacy_min_axis or 2))
    legacy_max_axis = max(legacy_min_axis, int(legacy_max_axis or legacy_min_axis))
    min_total, max_total = _adaptive_total_window_limits(args)
    using_total = bool(min_total > 0 or max_total > 0)
    if not using_total:
        return legacy_min_axis, legacy_max_axis, {
            "enabled": False,
            "n_secondary": int(n_secondary),
            "legacy_min_axis_windows": int(legacy_min_axis),
            "legacy_max_axis_windows": int(legacy_max_axis),
        }
    if min_total > 0 and max_total > 0 and min_total > max_total:
        raise ValueError("--adaptive-min-total-windows/replicas cannot exceed --adaptive-max-total-windows/replicas")
    if max_total > 0 and max_total < 2 * n_secondary:
        raise ValueError(
            f"Total adaptive max {max_total} is too small for a rectangular 2D grid with {n_secondary} secondary centers; "
            f"need at least {2*n_secondary} total replicas for two {label} centers. Reduce secondary centers or raise the total max."
        )
    axis_min = int(math.ceil(float(min_total) / float(n_secondary))) if min_total > 0 else legacy_min_axis
    axis_max = int(math.floor(float(max_total) / float(n_secondary))) if max_total > 0 else legacy_max_axis
    axis_min = max(2, axis_min)
    axis_max = max(2, axis_max)
    if axis_min > axis_max:
        raise ValueError(
            f"No rectangular 2D grid can satisfy total window budget min={min_total}, max={max_total} "
            f"with {n_secondary} secondary centers: {label}-axis bounds would be {axis_min}>{axis_max}. "
            "Adjust total min/max or the number of secondary CV centers."
        )
    return axis_min, axis_max, {
        "enabled": True,
        "n_secondary": int(n_secondary),
        "min_total_windows": int(min_total),
        "max_total_windows": int(max_total),
        "axis_min_windows": int(axis_min),
        "axis_max_windows": int(axis_max),
        "legacy_min_axis_windows": int(legacy_min_axis),
        "legacy_max_axis_windows": int(legacy_max_axis),
        "meaning": "Total limits apply to expanded primary x secondary windows/replicas; axis limits shown here are derived by dividing by the secondary-axis count.",
    }

def _thin_axis_centers_to_count(centers: list[float], max_count: int) -> list[float]:
    """Deterministically thin centers to a count while preserving endpoints."""
    vals = sorted(_unique_axis_values([float(x) for x in centers]))
    max_count = int(max_count)
    if max_count <= 0 or len(vals) <= max_count:
        return vals
    if max_count == 1:
        return [vals[len(vals)//2]]
    idxs = np.linspace(0, len(vals) - 1, max_count)
    chosen = sorted({int(round(float(i))) for i in idxs})
    # Rounding can create duplicates for small arrays. Fill missing slots from
    # largest gaps so the thinned ladder still covers the axis evenly.
    while len(chosen) < max_count:
        gaps = []
        for a, b in zip(chosen[:-1], chosen[1:]):
            if b - a > 1:
                gaps.append((b - a, (a + b) // 2))
        if not gaps:
            break
        chosen.append(max(gaps)[1])
        chosen = sorted(set(chosen))
    return [float(vals[i]) for i in chosen[:max_count]]

def _apply_total_window_budget_to_factorized_grid(args, primary_centers: list[float], secondary_centers: list[float]) -> tuple[list[float], list[float], dict]:
    """Enforce total replica/window cap on a factorized primary x secondary grid."""
    primary = sorted(_unique_axis_values([float(x) for x in primary_centers]))
    secondary = sorted(_unique_axis_values([float(x) for x in secondary_centers])) or [0.0]
    min_total, max_total = _adaptive_total_window_limits(args)
    notes = []
    old_total = len(primary) * len(secondary)
    if max_total > 0:
        # Prefer thinning the primary axis first, because in 2D peptide workflows
        # secondary/Rama centers are often semantically discrete labels.  If the
        # secondary axis alone makes the budget impossible, then thin secondary too.
        if len(secondary) > max_total // 2 and len(secondary) > 1:
            max_secondary = max(1, int(max_total // max(2, min(len(primary), max(2, len(primary))))))
            if len(secondary) > max_secondary:
                old = len(secondary)
                secondary = _thin_axis_centers_to_count(secondary, max_secondary)
                notes.append(f"thinned secondary axis {old}->{len(secondary)} to respect total replica cap {max_total}")
        if len(primary) * len(secondary) > max_total:
            max_primary = int(max_total // max(1, len(secondary)))
            if max_primary < 2:
                raise ValueError(
                    f"Total adaptive max {max_total} is too small for {len(secondary)} secondary centers; "
                    f"need at least {2*len(secondary)} replicas for two primary centers."
                )
            if len(primary) > max_primary:
                old = len(primary)
                primary = _thin_axis_centers_to_count(primary, max_primary)
                notes.append(f"thinned primary axis {old}->{len(primary)} to respect total replica cap {max_total}")
        if len(primary) * len(secondary) > max_total and len(secondary) > 1:
            max_secondary = max(1, int(max_total // max(1, len(primary))))
            if len(secondary) > max_secondary:
                old = len(secondary)
                secondary = _thin_axis_centers_to_count(secondary, max_secondary)
                notes.append(f"thinned secondary axis {old}->{len(secondary)} to respect total replica cap {max_total}")
    new_total = len(primary) * len(secondary)
    if min_total > 0 and new_total < min_total:
        notes.append(f"total replicas {new_total} below requested minimum {min_total}; rectangular grid/axis caps or sampled proposals limited expansion")
    return primary, secondary, {
        "enabled": bool(min_total > 0 or max_total > 0),
        "min_total_windows": int(min_total),
        "max_total_windows": int(max_total),
        "old_total_windows": int(old_total),
        "new_total_windows": int(new_total),
        "primary_windows": int(len(primary)),
        "secondary_windows": int(len(secondary)),
        "notes": notes,
    }

def build_adaptive_window_centers_a(low_a: float, high_a: float, args) -> np.ndarray:
    low_a = float(low_a)
    high_a = float(high_a)
    if not np.isfinite(low_a) or not np.isfinite(high_a) or high_a <= low_a:
        raise ValueError(f"Bad adaptive window range: low={low_a}, high={high_a}")
    if int(args.n_windows) > 0:
        n_windows = int(args.n_windows)
    else:
        aggr = _adaptive_window_aggressiveness_settings(args)
        spacing = max(0.1, float(args.adaptive_target_spacing_a)) * float(aggr.get("initial_spacing_factor", 1.0))
        n_windows = int(math.ceil((high_a - low_a) / spacing)) + 1
        # More aggressive modes are allowed to start with fewer windows, unless
        # the user explicitly raises --adaptive-min-windows.  In 2D runs, prefer
        # total-window budgets so max/min refer to actual replicas after crossing
        # with secondary-CV centers.
        nsec_budget = _secondary_center_count_for_total_budget(args)
        axis_min, axis_max, _budget_meta = _axis_count_bounds_from_total_window_budget(
            args, nsec_budget, int(args.adaptive_min_windows), int(args.adaptive_max_windows), label="distance"
        )
        n_windows = max(axis_min, min(axis_max, n_windows))
    if n_windows < 2:
        raise ValueError("Need at least two umbrella windows.")
    return np.round(np.linspace(low_a, high_a, n_windows), 4)

def adaptive_force_constants_kcal_a2(centers_a: np.ndarray, args) -> list[float]:
    centers_a = np.asarray(centers_a, dtype=float)
    if centers_a.size < 2:
        return [float(args.default_window_k_kcal_a2)]
    if str(args.adaptive_k_mode) == "constant":
        return [float(args.default_window_k_kcal_a2)] * int(centers_a.size)
    rt_kcal_mol = 0.00198720425864083 * float(args.temperature_k)
    spacings = np.diff(centers_a)
    local = np.empty_like(centers_a)
    local[0] = spacings[0]
    local[-1] = spacings[-1]
    if centers_a.size > 2:
        local[1:-1] = 0.5 * (spacings[:-1] + spacings[1:])
    overlap_sigma = max(0.25, float(args.adaptive_overlap_sigma))
    # sigma ~= spacing/overlap_sigma, k = kBT/sigma^2.
    ks = rt_kcal_mol / np.maximum(0.05, local / overlap_sigma) ** 2
    return [float(max(args.adaptive_min_k_kcal_a2, min(args.adaptive_max_k_kcal_a2, k))) for k in ks]

def adaptive_contact_centers(args) -> np.ndarray:
    """Initial adaptive centers for the dimensionless nonlocal-contact primary CV.

    If --contact-centers is provided with adaptive/adaptive-feedback, treat it as
    the initial ladder.  Otherwise generate a bounded contact-fraction ladder.
    This keeps manual contact windows and adaptive contact discovery compatible
    without reusing Angstrom-based distance bounds.
    """
    if getattr(args, "contact_centers", None):
        centers = np.asarray([float(x) for x in args.contact_centers], dtype=float)
    else:
        lo = float(getattr(args, "contact_adaptive_effective_min", getattr(args, "contact_adaptive_min", 0.0)) or 0.0)
        hi = float(getattr(args, "contact_adaptive_effective_max", getattr(args, "contact_adaptive_max", 0.80)) or 0.80)
        if bool(getattr(args, "contact_normalize", True)):
            lo = max(0.0, min(1.0, lo))
            hi = max(0.0, min(1.0, hi))
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            raise ValueError(f"Bad contact adaptive range: min={lo}, max={hi}")
        if int(getattr(args, "n_windows", 0) or 0) > 0:
            n = int(getattr(args, "n_windows"))
        else:
            spacing = max(1.0e-4, float(getattr(args, "contact_adaptive_target_spacing", 0.15) or 0.15))
            aggr = _adaptive_window_aggressiveness_settings(args)
            spacing *= float(aggr.get("initial_spacing_factor", 1.0))
            n = int(math.ceil((hi - lo) / spacing)) + 1
            nsec_budget = _secondary_center_count_for_total_budget(args)
            legacy_min = int(getattr(args, "contact_adaptive_min_windows", 4) or 4)
            legacy_max = int(getattr(args, "contact_adaptive_max_windows", 12) or 12)
            axis_min, axis_max, _budget_meta = _axis_count_bounds_from_total_window_budget(
                args, nsec_budget, legacy_min, legacy_max, label="contact"
            )
            n = max(axis_min, min(axis_max, n))
        centers = np.asarray(np.linspace(lo, hi, max(2, int(n))), dtype=float)
    if centers.size < 2:
        raise ValueError("Need at least two contact windows for adaptive contact mode.")
    if bool(getattr(args, "contact_normalize", True)):
        bad = [float(x) for x in centers if float(x) < -1.0e-8 or float(x) > 1.0 + 1.0e-8]
        if bad:
            raise ValueError("Normalized contact adaptive centers should be within [0, 1]")
    # Ensure sorted unique values.  Duplicate centers break exchange/proposal logic.
    unique = sorted({round(float(x), 6) for x in centers if math.isfinite(float(x))})
    if len(unique) < 2:
        raise ValueError("Contact adaptive centers collapsed to fewer than two unique values.")
    return np.asarray(unique, dtype=float)

def adaptive_contact_force_constants_kcal(centers_c: np.ndarray, args) -> list[float]:
    """Spacing-derived contact-CV k in kcal/mol/CV^2."""
    centers = np.asarray(centers_c, dtype=float)
    if centers.size < 2:
        return [float(getattr(args, "contact_adaptive_default_k_kcal", 25.0) or 25.0)]
    if str(getattr(args, "contact_adaptive_k_mode", "spacing") or "spacing").lower() in {"fixed", "constant"}:
        return [float(getattr(args, "contact_adaptive_default_k_kcal", 25.0) or 25.0)] * int(centers.size)
    rt_kcal_mol = 0.00198720425864083 * float(getattr(args, "temperature_k", 300.0) or 300.0)
    spacings = np.diff(centers)
    local = np.empty_like(centers)
    local[0] = spacings[0]
    local[-1] = spacings[-1]
    if centers.size > 2:
        local[1:-1] = 0.5 * (spacings[:-1] + spacings[1:])
    overlap_sigma = max(0.05, float(getattr(args, "contact_adaptive_overlap_sigma", getattr(args, "adaptive_overlap_sigma", 1.25)) or 1.25))
    min_sigma = max(1.0e-5, float(getattr(args, "contact_adaptive_min_sigma", 0.02) or 0.02))
    min_k = max(0.0, float(getattr(args, "contact_adaptive_min_k_kcal", 5.0) or 5.0))
    max_k = max(min_k, float(getattr(args, "contact_adaptive_max_k_kcal", 120.0) or 120.0))
    scale = max(0.0, float(getattr(args, "contact_adaptive_k_scale", 1.0) or 1.0))
    ks = []
    for spacing in local:
        sigma = max(min_sigma, float(spacing) / overlap_sigma)
        k = scale * rt_kcal_mol / (sigma * sigma)
        ks.append(float(max(min_k, min(max_k, k))))
    return ks

# Adaptive/manual window selection helpers.
def estimate_terminal_cv_envelope_a(topology, args) -> tuple[float, float, dict]:
    residues = peptide_residues(topology)
    nres = len(residues)
    if nres < 2:
        raise ValueError("Adaptive windows need at least two peptide residues.")
    if args.cv_atom1 and args.cv_atom2:
        contour_a = 3.8 * (nres - 1) + 2.0
        compact_a = max(2.5, float(args.adaptive_compact_floor_a))
        cv_kind = "explicit"
    elif args.cv_mode == "terminal-n-c":
        contour_a = 3.8 * (nres - 1) + 1.35
        compact_a = max(2.5, float(args.adaptive_compact_floor_a) - 0.5)
        cv_kind = "terminal-n-c"
    else:
        contour_a = 3.8 * (nres - 1)
        compact_a = float(args.adaptive_compact_floor_a)
        cv_kind = "terminal-ca"
    low_a = compact_a if float(args.adaptive_window_min_a) <= 0 else float(args.adaptive_window_min_a)
    high_a = max(low_a + 0.5, float(args.adaptive_extension_fraction) * contour_a)
    if float(args.adaptive_window_max_a) > 0:
        high_a = float(args.adaptive_window_max_a)
    info = {
        "n_residues": nres,
        "cv_kind": cv_kind,
        "estimated_contour_A": contour_a,
        "sequence_lower_A": low_a,
        "sequence_upper_A": high_a,
        "extension_fraction": float(args.adaptive_extension_fraction),
    }
    return low_a, high_a, info

def run_contact_cv_autocalibration(
    args,
    out_dir: Path,
    openmm,
    app,
    unit,
    forcefield,
    topology,
    equil_state,
    cv_atom1: Optional[int] = None,
    cv_atom2: Optional[int] = None,
    cv_label: Optional[str] = None,
    progress: Optional[GuiProgressSink] = None,
) -> dict:
    """Short unbiased/weakly-free probe used to calibrate contact-CV windows.

    The nonlocal-contact CV is dimensionless and system-dependent: a target of
    0.65 can mean "reasonable collapse" for one definition and "nearly all
    nonlocal residue pairs touch at once" for another.  This prescan samples the
    selected contact CV from the equilibrated structure with no contact umbrella,
    then chooses a conservative first-round adaptive range that extends beyond
    the observed values without blindly using an unreachable user maximum.
    """
    if not primary_cv_is_contacts(args):
        return {"enabled": False, "reason": "primary CV is not nonlocal-contacts"}
    if equil_state is None:
        return {"enabled": False, "reason": "no equilibrated state available"}

    _autocalib_steps = getattr(args, "contact_autocalibration_steps", None)
    if _autocalib_steps is None:
        _prescan_steps = int(getattr(args, "adaptive_prescan_steps", 0) or 0)
        steps = _prescan_steps if _prescan_steps > 0 else 5000
    else:
        steps = int(_autocalib_steps)
    if steps <= 0:
        return {"enabled": False, "reason": "contact_autocalibration_steps <= 0"}
    sample_interval = max(1, int(getattr(args, "contact_autocalibration_sample_interval", 100) or 100))
    safe_chunk = max(1, int(getattr(args, "contact_autocalibration_safe_chunk_steps", min(sample_interval, 100)) or min(sample_interval, 100)))
    _autocalib_ts = getattr(args, "contact_autocalibration_timestep_fs", None)
    if _autocalib_ts is None:
        _prescan_ts = float(getattr(args, "adaptive_prescan_timestep_fs", 1.0) or 1.0)
        timestep_fs = _prescan_ts if _prescan_ts > 0.0 else 1.0
    else:
        timestep_fs = float(_autocalib_ts)
    friction = float(getattr(args, "contact_autocalibration_friction_per_ps", 20.0) or 20.0)
    if timestep_fs <= 0.0:
        timestep_fs = min(float(getattr(args, "timestep_fs", 2.0) or 2.0), 1.0)
    if friction <= 0.0:
        friction = 20.0

    primary_def = prepare_primary_cv_definition(topology, args, cv_atom1=cv_atom1, cv_atom2=cv_atom2, cv_label=cv_label)
    platform, props = setup_platform_and_properties(openmm, args)
    system = create_system(app, unit, forcefield, topology, args, include_barostat=False)
    integrator = make_langevin_integrator(
        openmm, unit, args,
        timestep_fs=timestep_fs,
        temperature_k=float(getattr(args, "temperature_k", 300.0) or 300.0),
        friction_per_ps=friction,
        seed_offset=707,
    )
    sim = app.Simulation(topology, system, integrator, platform, props)
    try:
        box = equil_state.getPeriodicBoxVectors()
        if box is not None:
            sim.context.setPeriodicBoxVectors(*box)
    except Exception:
        pass
    sim.context.setPositions(equil_state.getPositions())
    try:
        vel = equil_state.getVelocities()
        if vel is not None:
            sim.context.setVelocities(vel)
        else:
            sim.context.setVelocitiesToTemperature(float(getattr(args, "temperature_k", 300.0)) * unit.kelvin, int(getattr(args, "seed", 1234)) + 707)
    except Exception:
        sim.context.setVelocitiesToTemperature(float(getattr(args, "temperature_k", 300.0)) * unit.kelvin, int(getattr(args, "seed", 1234)) + 707)

    out_dir = Path(out_dir)
    values: list[float] = []
    rows: list[dict] = []

    def _sample(step: int) -> None:
        try:
            state = sim.context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True)
            pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            cv = primary_cv_value_from_positions_nm(pos, primary_def, args)
            pe = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
            if math.isfinite(float(cv)):
                values.append(float(cv))
                rows.append({"step": int(step), "primary_cv_value": float(cv), "potential_kj_mol": float(pe)})
        except Exception as exc:
            rows.append({"step": int(step), "error": str(exc)})

    print(
        f"    Contact-CV autocalibration: {steps} steps at {timestep_fs:g} fs, "
        f"sampling every {sample_interval} steps on {platform_summary(platform, props)}"
    )
    if progress is not None:
        progress.progress(
            "contact_cv_autocalibration", 0, steps,
            message="unbiased contact-CV prescan for adaptive window bounds",
            timestep_fs=timestep_fs,
            force=True,
        )
    _sample(0)
    done = 0
    last_sample = 0
    try:
        while done < steps:
            chunk = min(safe_chunk, steps - done)
            sim.step(int(chunk))
            done += int(chunk)
            if done - last_sample >= sample_interval or done >= steps:
                _sample(done)
                last_sample = done
            if progress is not None:
                progress.progress(
                    "contact_cv_autocalibration", done, steps,
                    message="unbiased contact-CV prescan for adaptive window bounds",
                    timestep_fs=timestep_fs,
                    force=(done >= steps),
                )
    except Exception as exc:
        crash_path = out_dir / "CRASH_contact_cv_autocalibration.pdb"
        try:
            state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
            write_state_pdb(crash_path, app, topology, state.getPositions())
        except Exception:
            pass
        payload = {
            "enabled": True,
            "ok": False,
            "error": str(exc),
            "crash_pdb": str(crash_path) if crash_path.exists() else "",
            "values": values[-20:],
        }
        write_json(out_dir / "contact_cv_autocalibration.json", _json_ready(payload))
        release_openmm_contexts(sim, integrator, system)
        if bool(getattr(args, "contact_autocalibration_required", False)):
            raise
        print(f"WARNING: contact-CV autocalibration failed ({exc}); falling back to configured contact adaptive bounds.")
        return payload

    arr = np.asarray([float(x) for x in values if math.isfinite(float(x))], dtype=float)
    if arr.size == 0:
        payload = {"enabled": True, "ok": False, "reason": "no finite contact-CV samples"}
        write_json(out_dir / "contact_cv_autocalibration.json", _json_ready(payload))
        release_openmm_contexts(sim, integrator, system)
        return payload

    p05, p25, p50, p75, p95, p99 = [float(np.nanpercentile(arr, q)) for q in (5, 25, 50, 75, 95, 99)]
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    mean = float(np.nanmean(arr))
    sd = float(np.nanstd(arr))

    user_min = float(getattr(args, "contact_adaptive_min", 0.0) or 0.0)
    user_max = float(getattr(args, "contact_adaptive_max", 0.80) or 0.80)
    if bool(getattr(args, "contact_normalize", True)):
        user_min = max(0.0, min(1.0, user_min))
        user_max = max(0.0, min(1.0, user_max))
    margin = max(0.0, float(getattr(args, "contact_autocalibration_margin", 0.02) or 0.02))
    expand_factor = max(0.0, float(getattr(args, "contact_autocalibration_expand_factor", 4.0) or 4.0))
    max_multiple = max(1.0, float(getattr(args, "contact_autocalibration_max_multiple", 4.0) or 4.0))
    min_span = max(1.0e-6, float(getattr(args, "contact_autocalibration_min_span", 0.10) or 0.10))
    percentile = float(getattr(args, "contact_autocalibration_percentile", 99.0) or 99.0)
    percentile = max(50.0, min(100.0, percentile))
    p_hi = float(np.nanpercentile(arr, percentile))

    # Conservative first-round high bound: include the observed high tail plus a
    # margin, plus a controlled extrapolation.  This keeps the first adaptive
    # grid near the accessible range instead of placing windows at e.g. 0.65 when
    # the unbiased peptide only samples 0.01-0.03.  Later adaptive feedback can
    # still expand/repair as biased sampling discovers new accessible contacts.
    candidate_highs = [
        p_hi + margin,
        p50 + expand_factor * max(1.0e-6, p95 - p05),
        vmax * max_multiple + margin,
        user_min + min_span,
    ]
    effective_min = user_min
    effective_max = min(user_max, max(candidate_highs))
    if bool(getattr(args, "contact_normalize", True)):
        effective_min = max(0.0, min(1.0, effective_min))
        effective_max = max(0.0, min(1.0, effective_max))
    if effective_max <= effective_min:
        effective_max = min(user_max, effective_min + min_span)
    if effective_max <= effective_min:
        effective_max = effective_min + min_span

    setattr(args, "contact_adaptive_effective_min", float(effective_min))
    setattr(args, "contact_adaptive_effective_max", float(effective_max))
    summary = {
        "enabled": True,
        "ok": True,
        "steps": int(steps),
        "timestep_fs": float(timestep_fs),
        "sample_interval": int(sample_interval),
        "n_samples": int(arr.size),
        "primary_cv": primary_cv_mode(args),
        "primary_cv_label": primary_cv_label(args),
        "primary_cv_units": primary_cv_units(args),
        "contact_scheme": contact_scheme(args),
        "contact_atom_selection": str(getattr(args, "contact_atom_selection", "heavy")),
        "contact_r0_A": float(getattr(args, "contact_r0_a", 4.5) or 4.5),
        "contact_beta_A_inv": float(getattr(args, "contact_beta_a_inv", 6.0) or 6.0),
        "observed_min": vmin,
        "observed_p05": p05,
        "observed_p25": p25,
        "observed_median": p50,
        "observed_mean": mean,
        "observed_sd": sd,
        "observed_p75": p75,
        "observed_p95": p95,
        "observed_p99": p99,
        "observed_max": vmax,
        "user_requested_min": float(user_min),
        "user_requested_max": float(user_max),
        "effective_min": float(effective_min),
        "effective_max": float(effective_max),
        "margin": float(margin),
        "expand_factor": float(expand_factor),
        "max_multiple": float(max_multiple),
        "min_span": float(min_span),
        "note": "Initial contact adaptive centers use effective_min/effective_max; this avoids unreachable first-round contact windows. Adaptive-feedback may still expand/shift later.",
    }
    write_json(out_dir / "contact_cv_autocalibration.json", _json_ready({**summary, "samples_tail": rows[-50:]}))
    try:
        with (out_dir / "contact_cv_autocalibration.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["step", "primary_cv_value", "potential_kj_mol", "error"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    except Exception as exc:
        summary["csv_write_warning"] = str(exc)
    print(
        "    Contact-CV autocalibration observed "
        f"{vmin:.4f}..{vmax:.4f} (p95 {p95:.4f}, p99 {p99:.4f}); "
        f"using adaptive range {effective_min:.4f}..{effective_max:.4f} "
        f"instead of requested {user_min:.4f}..{user_max:.4f}"
    )
    release_openmm_contexts(sim, integrator, system)
    return summary

def broadcast_contact_k_for_centers(args, centers_c) -> tuple[list[float], str]:
    centers = list(centers_c)
    if getattr(args, "contact_k_kcal", None) is None or len(args.contact_k_kcal) == 0:
        return adaptive_contact_force_constants_kcal(np.asarray(centers, dtype=float), args), "adaptive-spacing"
    k_list = [float(x) for x in args.contact_k_kcal]
    if len(k_list) == 1 and len(centers) > 1:
        return k_list * len(centers), "manual-single-value-expanded"
    if len(k_list) != len(centers):
        raise ValueError("--contact-k-kcal must have length 1 or the same length as --contact-centers")
    return k_list, "manual-per-window"

def run_adaptive_prescan(args, out_dir: Path, openmm, app, unit, forcefield, topology, equil_state, cv_atom1: int, cv_atom2: int, sequence_low_a: float, sequence_high_a: float, progress: Optional[GuiProgressSink] = None) -> tuple[float, float, dict]:
    info = {"used": False, "steps": int(args.adaptive_prescan_steps)}
    if int(args.adaptive_prescan_steps) <= 0:
        return sequence_low_a, sequence_high_a, info
    platform, props = setup_platform_and_properties(openmm, args)
    print(f"    Adaptive prescan setup platform: {platform_summary(platform, props)}")
    system = create_system(app, unit, forcefield, topology, args, include_barostat=False)
    ts = min(float(args.timestep_fs), float(args.adaptive_prescan_timestep_fs))
    integrator = make_langevin_integrator(
        openmm, unit, args,
        timestep_fs=ts,
        temperature_k=float(args.temperature_k),
        friction_per_ps=float(args.equil_friction_per_ps),
        seed_offset=44,
    )
    sim = app.Simulation(topology, system, integrator, platform, props)
    sim.context.setPeriodicBoxVectors(*equil_state.getPeriodicBoxVectors())
    sim.context.setPositions(equil_state.getPositions())
    try:
        sim.context.setVelocities(equil_state.getVelocities())
    except Exception:
        sim.context.setVelocitiesToTemperature(args.temperature_k * unit.kelvin, int(args.seed) + 44)
    values = []
    prescan_csv = out_dir / "adaptive_window_prescan.csv"
    with prescan_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "cv_A"])
        writer.writeheader()
        done = 0
        try:
            while done < int(args.adaptive_prescan_steps):
                chunk = min(max(1, int(args.report_interval)), int(args.adaptive_prescan_steps) - done)
                run_steps_safely(sim, chunk, "adaptive_prescan", out_dir, app, topology, unit, chunk_size=int(args.equil_safe_chunk_steps), progress=progress, progress_total=int(args.adaptive_prescan_steps), progress_offset=done, timestep_fs=float(ts), message="adaptive CV pre-scan")
                done += chunk
                cv_a = 10.0 * cv_distance_nm(sim.context, cv_atom1, cv_atom2, unit)
                values.append(cv_a)
                writer.writerow({"step": done, "cv_A": cv_a})
        except Exception as exc:
            info["failed"] = str(exc)
            return sequence_low_a, sequence_high_a, info
    if values:
        arr = np.asarray(values, dtype=float)
        span = max(1.0, float(np.nanmax(arr) - np.nanmin(arr)))
        low = min(sequence_low_a, float(np.nanmin(arr)) - float(args.adaptive_prescan_margin_a))
        high = max(sequence_high_a, float(np.nanmax(arr)) + max(float(args.adaptive_prescan_margin_a), 0.25 * span))
        info.update({
            "used": True,
            "observed_min_A": float(np.nanmin(arr)),
            "observed_max_A": float(np.nanmax(arr)),
            "observed_mean_A": float(np.nanmean(arr)),
            "chosen_low_A": low,
            "chosen_high_A": high,
            "csv": str(prescan_csv),
        })
        return low, high, info
    return sequence_low_a, sequence_high_a, info

def choose_windows(args, out_dir: Path, openmm, app, unit, forcefield, topology, equil_state, cv_atom1: int, cv_atom2: int, progress: Optional[GuiProgressSink] = None) -> tuple[np.ndarray, list[float], dict]:
    if primary_cv_is_contacts(args):
        mode = str(getattr(args, "window_mode", "manual") or "manual")
        if mode == "manual":
            if not bool(getattr(args, "resume", False)) and (getattr(args, "contact_centers", None) is None or len(args.contact_centers) == 0):
                raise ValueError("--primary-cv nonlocal-contacts with --window-mode manual requires --contact-centers")
            centers_c = np.asarray([float(x) for x in args.contact_centers], dtype=float)
            k_list, k_assignment = broadcast_contact_k_for_centers(args, centers_c)
        elif mode in {"adaptive", "adaptive-feedback"}:
            autocalibration_info = {"enabled": False, "reason": "disabled or explicit contact centers supplied"}
            if bool(getattr(args, "contact_autocalibrate_windows", True)) and not getattr(args, "contact_centers", None):
                autocalibration_info = run_contact_cv_autocalibration(
                    args, out_dir, openmm, app, unit, forcefield, topology, equil_state,
                    cv_atom1=cv_atom1, cv_atom2=cv_atom2, progress=progress,
                )
            centers_c = adaptive_contact_centers(args)
            k_list, k_assignment = broadcast_contact_k_for_centers(args, centers_c)
        else:
            raise ValueError(f"Unsupported contact window mode {mode!r}")
        if bool(getattr(args, "contact_normalize", True)):
            bad = [float(x) for x in centers_c if float(x) < -1.0e-8 or float(x) > 1.0 + 1.0e-8]
            if bad:
                raise ValueError("Normalized contact centers should be within [0, 1]")
        meta = {
            "mode": mode,
            "primary_cv": "nonlocal-contacts",
            "description": "Primary adaptive/manual windows are dimensionless nonlocal-contact centers, stored in legacy center_A/cv_A columns as compatibility aliases.",
            "k_assignment": k_assignment,
            "contact_centers": [float(x) for x in centers_c],
            "contact_k_kcal": [float(x) for x in k_list],
            "contact_adaptive": {
                "enabled": bool(mode in {"adaptive", "adaptive-feedback"}),
                "min": float(getattr(args, "contact_adaptive_min", 0.0) or 0.0),
                "max": float(getattr(args, "contact_adaptive_max", 0.80) or 0.80),
                "effective_min": float(getattr(args, "contact_adaptive_effective_min", getattr(args, "contact_adaptive_min", 0.0)) or 0.0),
                "effective_max": float(getattr(args, "contact_adaptive_effective_max", getattr(args, "contact_adaptive_max", 0.80)) or 0.80),
                "target_spacing": float(getattr(args, "contact_adaptive_target_spacing", 0.15) or 0.15),
                "min_windows_axis_legacy": int(getattr(args, "contact_adaptive_min_windows", 4) or 4),
                "max_windows_axis_legacy": int(getattr(args, "contact_adaptive_max_windows", 12) or 12),
                "total_window_budget": _axis_count_bounds_from_total_window_budget(
                    args,
                    _secondary_center_count_for_total_budget(args),
                    int(getattr(args, "contact_adaptive_min_windows", 4) or 4),
                    int(getattr(args, "contact_adaptive_max_windows", 12) or 12),
                    label="contact",
                )[2],
            },
            "contact_autocalibration": autocalibration_info if mode in {"adaptive", "adaptive-feedback"} else {"enabled": False, "reason": "manual windows"},
        }
        write_json(out_dir / "adaptive_contact_windows.json", _json_ready(meta))
        return centers_c, k_list, meta

    if str(args.window_mode) == "manual":
        centers_a = np.asarray(args.windows_a, dtype=float)
        if args.window_k_kcal_a2 is None or len(args.window_k_kcal_a2) == 0:
            k_list = [float(args.default_window_k_kcal_a2)] * len(centers_a)
            k_assignment = "default-constant"
        else:
            k_list = [float(x) for x in args.window_k_kcal_a2]
            if len(k_list) == 1 and len(centers_a) > 1:
                k_list = k_list * len(centers_a)
                k_assignment = "manual-single-value-expanded"
            else:
                k_assignment = "manual-per-window"
        if len(k_list) != len(centers_a):
            raise ValueError("--window-k-kcal-a2 must have length 1 or the same length as --windows-a")
        return centers_a, k_list, {"mode": "manual", "k_assignment": k_assignment}

    low_a, high_a, info = estimate_terminal_cv_envelope_a(topology, args)
    low_a, high_a, prescan_info = run_adaptive_prescan(
        args, out_dir, openmm, app, unit, forcefield, topology, equil_state,
        cv_atom1, cv_atom2, low_a, high_a, progress=progress,
    )
    centers_a = build_adaptive_window_centers_a(low_a, high_a, args)
    k_list = adaptive_force_constants_kcal_a2(centers_a, args)
    warnings = []
    try:
        contour_a = float(info.get("estimated_contour_A", float("nan")))
        if math.isfinite(contour_a) and high_a > 1.05 * contour_a:
            warnings.append(
                f"adaptive upper bound {high_a:.2f} A exceeds estimated contour length {contour_a:.2f} A; high-center windows may be physically unreachable."
            )
    except Exception:
        pass
    if int(args.n_windows) > 0 and centers_a.size > 1:
        spacing = float(np.mean(np.diff(centers_a)))
        if spacing > 3.0:
            warnings.append(
                f"only {centers_a.size} windows over {low_a:.2f}-{high_a:.2f} A gives coarse spacing ~{spacing:.2f} A; adaptive k may be clamped weak and overlap may be poor."
            )
    for warning in warnings:
        print("WARNING:", warning)
    meta = {
        "mode": str(args.window_mode),
        "sequence_estimate": info,
        "prescan": prescan_info,
        "warnings": warnings,
        "aggressiveness": _adaptive_window_aggressiveness_settings(args),
    }
    write_json(out_dir / "adaptive_windows.json", {
        "centers_A": centers_a.tolist(),
        "k_kcal_mol_A2": k_list,
        **meta,
    })
    return centers_a, k_list, meta

# Explicit 2D window-topology helpers extracted from gareus_peptide.py.
def _rounded_unique_sorted(values, ndigits: int = 4) -> list[float]:
    seen = set()
    out = []
    if values is None:
        iterable = []
    else:
        iterable = list(values)
    for value in iterable:
        try:
            v = float(value)
        except Exception:
            continue
        if not math.isfinite(v):
            continue
        key = round(v, int(ndigits))
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return sorted(out)


def _positive_spacing_scale(values, fallback: float = 1.0) -> float:
    vals = _rounded_unique_sorted(values, ndigits=6)
    diffs = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1) if abs(vals[i + 1] - vals[i]) > 1.0e-8]
    if diffs:
        return max(1.0e-8, float(np.median(np.asarray(diffs, dtype=float))))
    if len(vals) >= 2:
        return max(1.0e-8, abs(vals[-1] - vals[0]))
    return max(1.0e-8, float(fallback))


def load_explicit_2d_window_csv(args, path: Path) -> tuple[np.ndarray, list[float], Optional[np.ndarray], Optional[list[float]], dict, dict]:
    """Load an explicit per-window table for sparse/non-rectangular 2D umbrella grids.

    Accepted column aliases are intentionally compatible with the advisory table
    written by the sparse-patch proposal helper:

      distance_center_A, distance_k_kcal_mol_A2,
      secondary_cv_center, secondary_cv_k_kcal_mol

    The return shape mirrors choose_windows()+expand_windows_for_secondary_cv(),
    but without forcing a rectangular cross-product.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"--windows-2d-csv file not found: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"--windows-2d-csv is empty: {path}")

    centers_a = []
    k_list = []
    secondary_centers = []
    secondary_k_list = []
    normalized_rows = []
    seen = set()
    duplicate_count = 0
    any_secondary = False
    all_secondary = True
    type_counts: dict[str, int] = {}

    for offset, row in enumerate(rows, start=2):
        primary_center_columns = [
            "primary_cv_center", "primary_center", "primary_center_cv", "primary_cv_value",
            "contact_cv_center", "contact_center", "primary_contact_center",
            "distance_center_A", "primary_center_A", "center_A", "window_center_A", "r0_A",
        ]
        primary_k_columns = [
            "primary_cv_k_kcal", "primary_k_kcal", "primary_cv_k_kcal_mol", "primary_k_kcal_mol",
            "contact_cv_k_kcal", "contact_k_kcal", "primary_contact_k_kcal",
            "distance_k_kcal_mol_A2", "primary_k_kcal_mol_A2", "k_kcal_mol_A2",
            "window_k_kcal_mol_A2", "k_kcal/A2", "k_kcal", "k_kcal_mol_cv2",
        ]
        center_a = _csv_float_field(
            row,
            primary_center_columns,
            offset,
            "primary-CV center (distance in A, contact fraction, or other primary-CV unit)",
        )
        if primary_cv_is_contacts(args):
            _manual_contact_k = getattr(args, "contact_k_kcal", None)
            if _manual_contact_k is not None and len(_manual_contact_k) > 0:
                default_primary_k = float(_manual_contact_k[0])
            else:
                default_primary_k = float(getattr(args, "contact_adaptive_default_k_kcal", getattr(args, "contact_adaptive_min_k_kcal", 25.0)) or 25.0)
        else:
            default_primary_k = float(getattr(args, "default_window_k_kcal_a2", 1.0) or 1.0)
        k = _csv_float_field(
            row,
            primary_k_columns,
            offset,
            "primary-CV force constant (kcal/mol/A^2, kcal/mol/CV^2, or primary-CV units)",
            default=default_primary_k,
        )
        if k < 0.0:
            raise ValueError(f"primary-CV force constant in explicit 2D window CSV row {offset} must be non-negative; got {k}")

        sec_raw = _csv_first_present(row, ["secondary_cv_center", "secondary_center", "secondary", "ss0", "secondary_cv_target"])
        has_secondary = sec_raw is not None
        any_secondary = any_secondary or has_secondary
        all_secondary = all_secondary and has_secondary
        if has_secondary:
            sec = _csv_float_field(row, ["secondary_cv_center", "secondary_center", "secondary", "ss0", "secondary_cv_target"], offset, "secondary-CV center")
            sec_k = _csv_float_field(
                row,
                ["secondary_cv_k_kcal_mol", "secondary_k_kcal_mol", "secondary_cv_k_kcal", "ss_k_kcal_mol", "secondary_k"],
                offset,
                "secondary-CV force constant (kcal/mol/CV^2)",
                default=float(getattr(args, "secondary_cv_k_kcal", 50.0) or 50.0),
            )
            if sec_k < 0.0:
                raise ValueError(f"secondary-CV force constant in explicit 2D window CSV row {offset} must be non-negative; got {sec_k}")
            sec_min, sec_max = secondary_cv_range(args)
            if sec < sec_min - 0.000001 or sec > sec_max + 0.000001:
                raise ValueError(
                    f"secondary-CV center in row {offset} is outside [{sec_min:g}, {sec_max:g}] "
                    f"for {secondary_cv_mode(args)!r} mode: {sec}"
                )
        else:
            sec = None
            sec_k = None

        key = (round(center_a, 5), round(sec if sec is not None else float("nan"), 5))
        if key in seen:
            duplicate_count += 1
            # Keep duplicates as separate thermodynamic states only if the user
            # explicitly duplicated them. Warn through metadata, but do not drop
            # a window behind the user's back.
        seen.add(key)

        wtype = str(_csv_first_present(row, ["window_type", "type", "parent", "source"]) or "explicit_2d_window")
        type_counts[wtype] = int(type_counts.get(wtype, 0)) + 1
        idx = len(centers_a)
        centers_a.append(float(center_a))
        k_list.append(float(k))
        if has_secondary:
            secondary_centers.append(float(sec))
            secondary_k_list.append(float(sec_k))
        primary_mode = primary_cv_mode(args) if hasattr(args, "primary_cv") else "distance"
        normalized_rows.append({
            "window": int(idx),
            "primary_cv_mode": str(primary_mode),
            "primary_cv_center": float(center_a),
            "primary_cv_k_kcal": float(k),
            # Backward-compatible aliases retained for existing downstream code.
            "distance_center_A": float(center_a),
            "distance_k_kcal_mol_A2": float(k),
            "secondary_cv_center": "" if sec is None else float(sec),
            "secondary_cv_k_kcal_mol": "" if sec_k is None else float(sec_k),
            "window_type": wtype,
            "source_row": int(offset),
        })

    if any_secondary and not all_secondary:
        raise ValueError("--windows-2d-csv mixes rows with and without secondary_cv_center; provide secondary columns for every row or none.")
    if any_secondary and not secondary_cv_enabled(args):
        raise ValueError("--windows-2d-csv contains secondary-CV centers, but --secondary-cv is 'none'. Re-run with e.g. --secondary-cv alpha-coil-beta/acb/rama-regions/alpha/beta/custom.")

    centers_arr = np.asarray(centers_a, dtype=float)
    primary_unique = _rounded_unique_sorted(centers_a, ndigits=4)
    secondary_unique = _rounded_unique_sorted(secondary_centers, ndigits=4) if any_secondary else []
    rectangular = False
    if any_secondary and primary_unique and secondary_unique:
        pair_set = {(round(float(c), 4), round(float(s), 4)) for c, s in zip(centers_a, secondary_centers)}
        rectangular = (len(pair_set) == len(primary_unique) * len(secondary_unique) == len(centers_a))

    secondary_metadata = {
        "enabled": bool(any_secondary),
        "mode": secondary_cv_mode(args) if any_secondary else "none",
        "explicit_2d_windows": bool(any_secondary),
        "grid": bool(rectangular),
        "source_csv": str(path),
        "n_total_windows": int(len(centers_a)),
        "n_primary_windows": int(len(primary_unique)),
        "n_secondary_centers": int(len(secondary_unique)),
        "primary_cv_mode": primary_cv_mode(args) if hasattr(args, "primary_cv") else "distance",
        "primary_centers": [float(x) for x in primary_unique],
        "primary_centers_A": [float(x) for x in primary_unique],
        "secondary_centers": [float(x) for x in secondary_unique],
        "window_type_counts": type_counts,
        "duplicate_center_pairs": int(duplicate_count),
        "normalized_rows": normalized_rows,
        "note": "Explicit 2D window table loaded without rectangular cross-product expansion; neighbor exchange uses a geometry graph when --exchange-mode neighbor.",
    }
    if any_secondary:
        sec_min, sec_max = secondary_cv_range(args)
        secondary_metadata.update({
            "range_min": float(sec_min),
            "range_max": float(sec_max),
        })
    window_metadata = {
        "mode": "explicit-2d-csv",
        "source_csv": str(path),
        "n_windows": int(len(centers_a)),
        "n_primary_centers": int(len(primary_unique)),
        "n_secondary_centers": int(len(secondary_unique)),
        "rectangular_grid": bool(rectangular),
        "normalized_rows": normalized_rows,
        "duplicate_center_pairs": int(duplicate_count),
    }
    sec_arr = np.asarray(secondary_centers, dtype=float) if any_secondary else None
    sec_k = [float(x) for x in secondary_k_list] if any_secondary else None
    return centers_arr, [float(x) for x in k_list], sec_arr, sec_k, secondary_metadata, window_metadata


def build_explicit_2d_neighbor_edges(centers_a, secondary_cv_centers, args=None) -> list[dict]:
    """Build a conservative geometry graph for explicit sparse 2D windows."""
    centers = np.asarray(centers_a, dtype=float)
    secondary = np.asarray(secondary_cv_centers, dtype=float) if secondary_cv_centers is not None else None
    n = int(len(centers))
    if n <= 1:
        return []
    if secondary is None or len(secondary) != n:
        return [{"wi": i, "wj": i + 1, "edge_type": "linear_neighbor", "normalized_distance": 1.0} for i in range(n - 1)]

    d_scale = _positive_spacing_scale(centers, fallback=max(1.0, float(np.nanmax(centers) - np.nanmin(centers)) if n > 1 else 1.0))
    s_scale = _positive_spacing_scale(secondary, fallback=0.25 if secondary_cv_is_transition(getattr(args, "secondary_cv", "none")) else 0.15)
    row_tol = max(1.0e-6, 0.25 * s_scale)
    col_tol = max(1.0e-6, 0.25 * d_scale)
    k_nearest = max(0, int(getattr(args, "explicit_2d_exchange_neighbor_k", 2) if args is not None else 2))
    radius = float(getattr(args, "explicit_2d_exchange_radius", 1.65) if args is not None else 1.65)
    if not math.isfinite(radius) or radius <= 0.0:
        radius = 1.65

    edges: dict[tuple[int, int], dict] = {}

    def add_edge(i: int, j: int, edge_type: str, ndist: float) -> None:
        if i == j:
            return
        a, b = sorted((int(i), int(j)))
        if a < 0 or b >= n:
            return
        ndist = float(ndist) if math.isfinite(float(ndist)) else float("nan")
        row = edges.get((a, b))
        if row is None:
            edges[(a, b)] = {"wi": a, "wj": b, "edge_type": str(edge_type), "normalized_distance": ndist}
        else:
            types = set(str(row.get("edge_type", "")).split("+"))
            types.add(str(edge_type))
            row["edge_type"] = "+".join(sorted(t for t in types if t))
            if math.isfinite(ndist) and (not math.isfinite(float(row.get("normalized_distance", float("nan")))) or ndist < float(row.get("normalized_distance"))):
                row["normalized_distance"] = ndist

    for i in range(n):
        # Nearest neighbor to the right along the distance coordinate within the same secondary row.
        candidates = []
        for j in range(n):
            dd = centers[j] - centers[i]
            ds = abs(secondary[j] - secondary[i])
            if dd > 1.0e-8 and ds <= row_tol:
                nd = math.sqrt((dd / d_scale) ** 2 + (ds / s_scale) ** 2)
                candidates.append((dd, nd, j))
        if candidates:
            _dd, nd, j = min(candidates, key=lambda x: (x[0], x[1]))
            add_edge(i, j, "distance_axis", nd)

        # Nearest neighbor upward along the secondary coordinate within the same distance column.
        candidates = []
        for j in range(n):
            ds_signed = secondary[j] - secondary[i]
            dd = abs(centers[j] - centers[i])
            if ds_signed > 1.0e-8 and dd <= col_tol:
                nd = math.sqrt((dd / d_scale) ** 2 + (ds_signed / s_scale) ** 2)
                candidates.append((ds_signed, nd, j))
        if candidates:
            _ds, nd, j = min(candidates, key=lambda x: (x[0], x[1]))
            add_edge(i, j, "secondary_axis", nd)

    # Sparse patch windows can be diagonal or off-row after user edits. Add a
    # small k-nearest fallback in normalized 2D space so every window has a chance
    # to exchange without turning the graph into all-pair confetti.
    if k_nearest > 0:
        for i in range(n):
            candidates = []
            for j in range(n):
                if i == j:
                    continue
                nd = math.sqrt(((centers[j] - centers[i]) / d_scale) ** 2 + ((secondary[j] - secondary[i]) / s_scale) ** 2)
                if nd <= radius + 1.0e-12:
                    candidates.append((nd, j))
            candidates.sort(key=lambda x: (x[0], x[1]))
            for nd, j in candidates[:k_nearest]:
                add_edge(i, j, "knn_geometry", nd)

    out = list(edges.values())
    out.sort(key=lambda r: (float(r.get("normalized_distance", float("inf"))), int(r["wi"]), int(r["wj"]), str(r.get("edge_type", ""))))
    return out
