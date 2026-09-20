"""Residual torsion components: phi(x) minus its polynomial dependence on the anchor.

For canonical torsion features ``X`` (rows = frames) and a scalar anchor ``a``
(the contact CV), fit by weighted least squares

    m(a) = B0 + B1 a_std [+ B2 a_std^2],      a_std = (a - mu_c) / sigma_c

and diagonalise the weighted covariance of the residual ``R = X - m(a)``. Each
right singular vector ``v_j`` defines one candidate coordinate

    z2^(j) = ( v_j . (R - mean_R) - mu_j ) / sigma_j .

``sum_i w_i z2_i a_std_i == 0`` holds *exactly* because the residual of a
weighted OLS fit with an intercept is w-orthogonal to every regressor column,
and ``a_std`` is a regressor column. The identity is in the WEIGHTED inner
product; the plain ``np.cov`` is not zero under non-uniform weights and must
never be used as the certificate.

Runtime consequence (plan section 5.2): the CV2 umbrella must differentiate
the whole expression, including ``-(v.B1 + 2 a v.B2) grad a / sigma_c``. That
term also makes the CV2 restraint push on the anchor; its induced curvature is
:func:`coupling_curvature_kcal` and is a deployability criterion, because it
varies by orders of magnitude between otherwise interchangeable components.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from . import contracts as C

#: Two adjacent singular values closer than this fraction of the leading one
#: mark an unstable basis: the frozen vector, not the label "PC j", is identity.
_TIE_FRACTION = 1e-6

#: Training quantiles that bound the anchor at runtime for degree-2 models.
_CLAMP_QUANTILES = (0.005, 0.995)


@dataclass(frozen=True)
class ResidualFit:
    """Everything needed to evaluate every component, in float64."""

    coefficients: np.ndarray        # (3, d): rows B0, B1, B2 (B2 all-zero for degree 1)
    residual_mean: np.ndarray       # (d,)
    singular_values: np.ndarray     # (k,)
    right_vectors: np.ndarray       # (k, d), row j-1 is v_j, sign-fixed
    anchor_mean: float
    anchor_std: float
    anchor_clamp: tuple             # (lo, hi) in a_std units, training quantiles
    projection_mean: np.ndarray     # (k,)
    projection_std: np.ndarray      # (k,)
    degree: int

    @property
    def n_components(self) -> int:
        return int(self.right_vectors.shape[0])

    @property
    def width(self) -> int:
        return int(self.right_vectors.shape[1])


def _design(a_std: np.ndarray, degree: int) -> np.ndarray:
    columns = [np.ones_like(a_std), a_std]
    if degree == 2:
        columns.append(a_std ** 2)
    return np.column_stack(columns)


def _normalised_weights(weights, n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be a finite nonnegative vector of length n with positive sum")
    return w / w.sum()


def fit_residual_components(X, anchor, *, degree: int = 1,
                            n_components: int = C.MAX_COMPONENT_INDEX,
                            weights=None) -> ResidualFit:
    X = np.asarray(X, dtype=np.float64)
    a = np.asarray(anchor, dtype=np.float64)
    if degree not in (1, 2):
        raise ValueError("degree must be 1 or 2")
    if X.ndim != 2 or a.shape != (X.shape[0],):
        raise ValueError("X must be (n, d) and anchor (n,)")
    if not (np.isfinite(X).all() and np.isfinite(a).all()):
        raise ValueError("non-finite input")
    n, d = X.shape
    w = _normalised_weights(weights, n)
    mu_c = float(np.sum(w * a))
    sd_c = float(np.sqrt(np.sum(w * (a - mu_c) ** 2)))
    if sd_c <= 1e-12:
        raise ValueError("anchor has zero variance; a constant anchor defines no coordinate")
    a_std = (a - mu_c) / sd_c
    design = _design(a_std, degree)
    sqrt_w = np.sqrt(w)[:, None]
    B, _, rank, _ = np.linalg.lstsq(design * sqrt_w, X * sqrt_w, rcond=1e-12)
    if rank < design.shape[1]:
        raise ValueError(f"design matrix rank {rank} < {design.shape[1]}; residualisation is undefined")
    if degree == 1:
        B = np.vstack([B, np.zeros((1, d))])
    residual = X - _design(a_std, 2) @ B
    mean_R = np.sum(w[:, None] * residual, axis=0)
    _, singular, Vt = np.linalg.svd((residual - mean_R) * sqrt_w, full_matrices=False)
    k = min(int(n_components), Vt.shape[0])
    V = Vt[:k].copy()
    for row in V:
        pivot = row[np.argmax(np.abs(row))]
        row *= np.sign(pivot) if pivot != 0.0 else 1.0
    scores = (residual - mean_R) @ V.T
    mu = np.sum(w[:, None] * scores, axis=0)
    sd = np.sqrt(np.sum(w[:, None] * (scores - mu) ** 2, axis=0))
    if np.any(sd <= 1e-12):
        raise ValueError("a component has zero variance on the training data")
    clamp = (float(np.quantile(a_std, _CLAMP_QUANTILES[0])),
             float(np.quantile(a_std, _CLAMP_QUANTILES[1])))
    return ResidualFit(B, mean_R, singular[:k], V, mu_c, sd_c, clamp, mu, sd, int(degree))


def standardised_anchor(fit: ResidualFit, anchor, *, clamp: bool = False) -> np.ndarray:
    a_std = (np.asarray(anchor, dtype=np.float64) - fit.anchor_mean) / fit.anchor_std
    return np.clip(a_std, *fit.anchor_clamp) if clamp else a_std


def evaluate_component(fit: ResidualFit, j: int, X, anchor, *, clamp: bool = False) -> np.ndarray:
    """Standardised ``z2^(j)`` for every row of ``X``."""
    if not 1 <= int(j) <= fit.n_components:
        raise ValueError(f"component {j} out of range 1..{fit.n_components}")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != fit.width:
        raise ValueError(f"X must be (n, {fit.width})")
    a_std = standardised_anchor(fit, anchor, clamp=clamp)
    residual = X - _design(a_std, 2) @ fit.coefficients
    score = (residual - fit.residual_mean) @ fit.right_vectors[j - 1]
    return (score - fit.projection_mean[j - 1]) / fit.projection_std[j - 1]


def coupling_curvature_kcal(fit: ResidualFit, j: int, k2_kcal: float) -> float:
    """Curvature the CV2 umbrella induces along the anchor, d²U₂/dc² at a_std = 0.

    With z2 = (v.phi - K0 - K1 a - K2 a²)/sigma_j and a = (c - mu_c)/sigma_c,
    dz2/dc = -(K1 + 2 K2 a)/(sigma_j sigma_c), so the harmonic CV2 restraint
    contributes k2 (dz2/dc)² of curvature along c. Reported at a = 0.
    """
    v = fit.right_vectors[int(j) - 1]
    slope = float(v @ fit.coefficients[1])
    return float(k2_kcal) * slope ** 2 / (float(fit.projection_std[int(j) - 1]) ** 2
                                          * float(fit.anchor_std) ** 2)


def _tie_flags(singular: np.ndarray) -> list[bool]:
    if singular.size == 0:
        return []
    scale = max(float(singular[0]), 1e-300)
    flags = []
    for i, s in enumerate(singular):
        near = []
        if i > 0:
            near.append(abs(float(s) - float(singular[i - 1])))
        if i + 1 < singular.size:
            near.append(abs(float(s) - float(singular[i + 1])))
        flags.append(bool(near) and min(near) < _TIE_FRACTION * scale)
    return flags


def to_candidate_set(fit: ResidualFit, feature_schema: C.FeatureSchema,
                     primary_definition: Mapping[str, Any], physical_system_sha256: str,
                     training_rows_sha256: str, library_versions: Mapping[str, str]) -> C.CandidateSet:
    """Freeze the fit as the contract artifact every later stage reads."""
    if fit.width != feature_schema.width:
        raise ValueError(f"fit width {fit.width} != feature schema width {feature_schema.width}")
    ties = _tie_flags(fit.singular_values)
    components = []
    for j in range(1, fit.n_components + 1):
        components.append({
            "component_index": j,
            "singular_value": float(fit.singular_values[j - 1]),
            "eigenvalue_tie_flagged": ties[j - 1],
            "right_singular_vector": fit.right_vectors[j - 1].tolist(),
            "residual_mean": fit.residual_mean.tolist(),
            "regression_coefficients": [row.tolist() for row in fit.coefficients],
            "primary_mean": float(fit.anchor_mean),
            "primary_std": float(fit.anchor_std),
            "anchor_clamp": [float(fit.anchor_clamp[0]), float(fit.anchor_clamp[1])],
            "projection_mean": float(fit.projection_mean[j - 1]),
            "projection_std": float(fit.projection_std[j - 1]),
        })
    return C.CandidateSet.from_mapping({
        "schema": C.CANDIDATE_SET_VERSION,
        "kind": C.CANDIDATE_KIND_QUADRATIC_RESIDUAL,
        "feature_schema_sha256": feature_schema.sha256,
        "physical_system_sha256": physical_system_sha256,
        "training_rows_sha256": training_rows_sha256,
        "library_versions": dict(library_versions),
        "primary_definition": dict(primary_definition),
        "components": components,
    })


def from_candidate_set(candidates: C.CandidateSet) -> ResidualFit:
    """Invert :func:`to_candidate_set` exactly (the JSON holds float64 verbatim)."""
    comps = sorted(candidates.components, key=lambda c: c.component_index)
    if [c.component_index for c in comps] != list(range(1, len(comps) + 1)):
        raise ValueError("candidate set components must be 1..k without gaps")
    first = comps[0]
    B = np.asarray(first.regression_coefficients, dtype=np.float64)
    degree = 2 if np.any(B[2] != 0.0) else 1
    return ResidualFit(
        coefficients=B,
        residual_mean=np.asarray(first.residual_mean, dtype=np.float64),
        singular_values=np.asarray([c.singular_value for c in comps], dtype=np.float64),
        right_vectors=np.asarray([c.right_singular_vector for c in comps], dtype=np.float64),
        anchor_mean=float(first.primary_mean),
        anchor_std=float(first.primary_std),
        anchor_clamp=(float(first.anchor_clamp[0]), float(first.anchor_clamp[1])),
        projection_mean=np.asarray([c.projection_mean for c in comps], dtype=np.float64),
        projection_std=np.asarray([c.projection_std for c in comps], dtype=np.float64),
        degree=degree,
    )
