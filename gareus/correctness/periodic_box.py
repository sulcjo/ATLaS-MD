"""Explicit reduced cells for fresh setup; no barostat or dynamics modifications.

All vectors are rows in nanometers. L is the shortest lattice translation of
these three fixed templates, not an equal-volume cubic edge. These helpers are
not a general shortest-vector algorithm and do not certify future conformations.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np
from ._io import IntegrityError
from .bias import finite_number

CONSTRUCTION_VERSION = "explicit-lattice-v1"


def solvent_construction_model(water_model: str) -> str:
    mapping = {"tip3p": "tip3p", "tip3pfb": "tip3p", "spce": "spce",
               "tip4pew": "tip4pew", "opc": "tip4pew"}
    if water_model not in mapping:
        raise IntegrityError(f"Unsupported ATLaS-MD water model: {water_model!r}")
    return mapping[water_model]


def reduced_box_vectors_nm(shape: str, shortest_translation_nm: float) -> np.ndarray:
    templates = {
        "cube": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        "dodecahedron": ((1, 0, 0), (0, 1, 0), (0.5, 0.5, math.sqrt(0.5))),
        "octahedron": ((1, 0, 0), (1/3, 2*math.sqrt(2)/3, 0),
                       (-1/3, math.sqrt(2)/3, math.sqrt(6)/3)),
    }
    if shape not in templates:
        raise IntegrityError(f"Unsupported box shape: {shape!r}")
    scale = finite_number(shortest_translation_nm, "shortest_translation_nm", positive=True)
    return np.asarray(templates[shape], dtype=float) * scale


def validate_reduced_cell(vectors_nm: np.ndarray) -> np.ndarray:
    box = np.asarray(vectors_nm, dtype=np.float64)
    if box.shape != (3, 3) or not np.isfinite(box).all():
        raise IntegrityError("Box vectors must be a finite (3,3) row-vector array")
    eps = 32 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(box))))
    if np.any(np.abs(np.triu(box, 1)) > eps) or np.any(np.diag(box) <= 0):
        raise IntegrityError("Box is not in OpenMM reduced orientation")
    ax, by, _ = np.diag(box)
    if (ax + eps < 2 * abs(box[1, 0]) or ax + eps < 2 * abs(box[2, 0])
            or by + eps < 2 * abs(box[2, 1])):
        raise IntegrityError("Box does not satisfy OpenMM reduced-cell inequalities")
    return box.copy()


def solute_enclosing_diameter_nm(positions_nm: np.ndarray) -> float:
    positions = np.asarray(positions_nm, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1:] != (3,) or not np.isfinite(positions).all():
        raise IntegrityError("Solute positions must be finite (N,3)")
    if len(positions) == 0:
        return 0.0
    centered = positions - positions.mean(axis=0)
    return 2.0 * float(np.linalg.norm(centered, axis=1).max())


def choose_box_scale_nm(
    shape: str, solute_bound_nm: float, contour_estimate_nm: float,
    padding_nm: float, cutoff_nm: float,
) -> float:
    bound = finite_number(solute_bound_nm, "solute_bound_nm", minimum=0)
    contour = finite_number(contour_estimate_nm, "contour_estimate_nm", minimum=0)
    padding = finite_number(padding_nm, "padding_nm", minimum=0)
    cutoff = finite_number(cutoff_nm, "cutoff_nm", positive=True)
    unit_box = reduced_box_vectors_nm(shape, 1.0)
    # Conservative Context compatibility: all reduced diagonal dimensions must
    # be greater than twice the cutoff, not just the shortest lattice vector.
    cutoff_floor = 2.0 * cutoff / float(np.diag(unit_box).min())
    cutoff_floor = np.nextafter(cutoff_floor * (1.0 + 1e-10), np.inf)
    return float(max(max(bound, contour) + 2.0 * padding, cutoff_floor))


def audit_periodic_box(
    positions_nm: np.ndarray, vectors_nm: np.ndarray, *, shape: str,
    contour_estimate_nm: float, padding_nm: float, cutoff_nm: float,
) -> dict[str, Any]:
    box = validate_reduced_cell(vectors_nm)
    scale = float(box[0, 0])
    template = reduced_box_vectors_nm(shape, scale)
    if not np.allclose(box, template, rtol=1e-12, atol=1e-12):
        raise IntegrityError("Actual box does not match the requested template")
    bound = solute_enclosing_diameter_nm(positions_nm)
    cutoff = finite_number(cutoff_nm, "cutoff_nm", positive=True)
    contour = finite_number(contour_estimate_nm, "contour_estimate_nm", minimum=0)
    padding = finite_number(padding_nm, "padding_nm", minimum=0)
    # If all points lie in one sphere of diameter D, every nonzero translated
    # copy is separated by at least L-D. Negative values are noninformative,
    # not evidence of actual overlap.
    gap_bound = scale - bound
    admissible = bool(np.all(np.diag(box) > 2.0 * cutoff))
    return {
        "schema_version": "2.0-lattice",
        "construction_version": CONSTRUCTION_VERSION,
        "box_shape": shape,
        "box_vectors_nm": box.tolist(),
        "volume_nm3": float(np.linalg.det(box)),
        "shortest_translation_nm": scale,
        "reduced_diagonal_nm": np.diag(box).tolist(),
        "solute_enclosing_diameter_nm": bound,
        "contour_estimate_nm": contour,
        "contour_estimate_is_heuristic": True,
        "padding_nm": padding,
        "padding_convention": "added on both sides of enclosing/contour size",
        "nonbonded_cutoff_nm": cutoff,
        "cutoff_admissible": admissible,
        "current_solute_image_gap_lower_bound_nm": gap_bound,
        "current_solute_clearance_proven": bool(gap_bound >= cutoff),
        "future_conformation_clearance_proven": False,
    }
