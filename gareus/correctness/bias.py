"""Strict shared umbrella-bias reconstruction for the existing ATLaS-MD API.

Energy input convention is unchanged: k1 and k2 are kcal/mol per squared CV
unit; CV values and centers must already use the SAME units. beta is mol/kJ.
The existing gareus.mbar_analysis.ladder helper remains the only boost formula.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence
import numpy as np

from ._io import IntegrityError

KJ_PER_KCAL = 4.184


class MissingCoordinateError(IntegrityError):
    """A requested Hamiltonian needs a coordinate that was not supplied."""


def numeric_vector(value: Any, name: str, length: int | None = None) -> np.ndarray:
    """Preserve masked data as NaN; never flatten or permit broadcasting."""
    try:
        masked = np.ma.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise IntegrityError(f"{name} must be a numeric one-dimensional vector") from exc
    if masked.ndim != 1:
        raise IntegrityError(f"{name} must have shape (N,), got {masked.shape}")
    array = np.asarray(masked.filled(np.nan), dtype=np.float64)
    if length is not None and array.shape != (length,):
        raise IntegrityError(f"{name} must have shape ({length},), got {array.shape}")
    # Infinity is missing/invalid observation, not a physically infinite
    # restraint parameter. It will propagate NaN in columns requiring this CV.
    array = array.copy()
    array[~np.isfinite(array)] = np.nan
    return array


def finite_number(value: Any, label: str, *, minimum: float | None = None,
                  positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise IntegrityError(f"{label} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IntegrityError(f"{label} must be a finite number; got {value!r}") from exc
    if not math.isfinite(result):
        raise IntegrityError(f"{label} must be finite; got {value!r}")
    if positive and result <= 0:
        raise IntegrityError(f"{label} must be positive")
    if minimum is not None and result < minimum:
        raise IntegrityError(f"{label} must be >= {minimum}")
    return result


def _term(window: Mapping[str, Any], axis: int, label: str) -> tuple[float, float]:
    center_key, force_key = f"center{axis}", f"k{axis}"
    if force_key not in window:
        if axis == 2 and center_key not in window:
            return 0.0, 0.0
        raise IntegrityError(f"{label}: missing {force_key}; inactive means explicit zero")
    force = finite_number(window[force_key], f"{label}.{force_key}", minimum=0)
    if force == 0:
        # With zero force the center is irrelevant. Do not require a meaningful
        # placeholder, and do not multiply an unmeasured CV by zero later.
        return 0.0, 0.0
    if center_key not in window:
        raise IntegrityError(f"{label}: active {force_key} requires {center_key}")
    center = finite_number(window[center_key], f"{label}.{center_key}")
    return center, force


def normalize_windows(windows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for index, raw in enumerate(windows):
        if not isinstance(raw, Mapping):
            raise IntegrityError(f"window[{index}] must be a mapping")
        window = dict(raw)
        label = f"window {raw.get('window_id', index)}"
        window["center1"], window["k1"] = _term(raw, 1, label)
        window["center2"], window["k2"] = _term(raw, 2, label)
        window["gamd_lambda"] = finite_number(raw.get("gamd_lambda", 0.0),
                                              f"{label}.gamd_lambda", minimum=0)
        normalized.append(window)
    return normalized


def normalize_verified_legacy_disabled_secondary(
    windows: Sequence[Mapping[str, Any]], *, secondary_disabled: bool,
) -> list[dict[str, Any]]:
    """Explicit adapter for a verified *disabled* old secondary-CV definition.

    Never infer disabled status from bad numbers. Positive or negative finite
    forces conflict with disabled provenance and are rejected. This adapter is
    NOT invoked automatically by reconstruction or by the strict exporter.
    """
    if secondary_disabled is not True:
        raise IntegrityError("Legacy adapter requires explicit disabled-CV evidence")
    out = []
    for index, raw in enumerate(windows):
        window = dict(raw)
        value = raw.get("k2")
        if value is not None:
            try:
                k = float(value)
            except (TypeError, ValueError) as exc:
                raise IntegrityError(f"Unrecognized legacy k2 at window {index}") from exc
            if math.isfinite(k) and k != 0:
                raise IntegrityError(f"Legacy k2={k} contradicts disabled secondary CV")
        window.update(center2=0.0, k2=0.0)
        out.append(window)
    return normalize_windows(out)


def reconstruct_bias_matrix(
    cv_A: np.ndarray,
    cv2: np.ndarray | None,
    windows: list,
    beta: float,
    *,
    v_pep: np.ndarray | None = None,
    v_dih: np.ndarray | None = None,
    envelope=None,
    meta: dict | None = None,
    _ladder_apply: Callable | None = None,
) -> np.ndarray:
    """Return the (N,K) reduced umbrella-plus-ladder bias, not full potential.

    Missing entire required coordinates raise. Missing individual observations
    yield NaN only in states requiring that coordinate. Row filtering belongs
    downstream and must keep origins/counts/aligned arrays consistent.
    _ladder_apply is a test seam, not an alternative production boost definition.
    """
    beta = finite_number(beta, "beta", positive=True)
    primary = numeric_vector(cv_A, "cv_A")
    n = len(primary)
    secondary = None if cv2 is None else numeric_vector(cv2, "cv2", n)
    pep = None if v_pep is None else numeric_vector(v_pep, "v_pep", n)
    dih = None if v_dih is None else numeric_vector(v_dih, "v_dih", n)
    rows = normalize_windows(windows)
    active_secondary = [w.get("window_id", i) for i, w in enumerate(rows) if w["k2"] > 0]
    if active_secondary and secondary is None:
        raise MissingCoordinateError(
            f"cv2 was not supplied, but states {active_secondary} have active secondary restraints"
        )
    matrix = np.zeros((n, len(rows)), dtype=np.float64)
    with np.errstate(over="raise", invalid="ignore"):
        try:
            for column, row in enumerate(rows):
                if row["k1"] > 0:
                    delta = primary - row["center1"]
                    matrix[:, column] += 0.5 * row["k1"] * delta * delta
                if row["k2"] > 0:
                    delta = secondary - row["center2"]
                    matrix[:, column] += 0.5 * row["k2"] * delta * delta
            matrix *= beta * KJ_PER_KCAL
        except FloatingPointError as exc:
            raise IntegrityError("Overflow reconstructing umbrella bias; check CV units/parameters") from exc
    lambdas = np.asarray([w["gamd_lambda"] for w in rows], dtype=np.float64)
    bookkeeping = meta if meta is not None else {}
    if not np.any(lambdas > 0):
        bookkeeping.update(gamd_ladder=False, gamd_ladder_samples_without_raw_energies=0)
        return matrix
    if envelope is None:
        raise IntegrityError("Active gamd_lambda requires the frozen envelope and raw energies")
    if dih is None or (bool(getattr(envelope, "has_total", True)) and pep is None):
        raise IntegrityError("Active gamd_lambda requires v_dih and, for a Total channel, v_pep")
    if pep is None:
        # Truly dihedral-only: NaN is an explicit absent inactive channel,
        # never a fabricated zero energy. The repository helper ignores it.
        pep = np.full(n, np.nan, dtype=np.float64)
    if _ladder_apply is None:
        from gareus.mbar_analysis.ladder import apply_ladder_boost_to_u
        _ladder_apply = apply_ladder_boost_to_u
    return _ladder_apply(matrix, pep, dih, lambdas, envelope, beta, bookkeeping)
