"""Slowness at fixed CV1: conditional tICA candidates and the scores that rank them.

A CV2 umbrella only pays off along a coordinate that is slow and barrier-separated at fixed
CV1; a fast, single-peaked direction is merely tilted (chignolin_9's residual PC1: k2 matched
its own curvature, rows sat halfway to their centres, no exploration gain over CV1 alone).
This module adds the slow directions of the CV1-residualised features -- tICA on the same
residual the PCA components use, so every candidate is still one linear direction that the
compiled force evaluates unchanged -- and scores any candidate by its lag autocorrelation and
its bimodality within CV1 cells.

Everything here needs trajectory order (member id and frame index per row). Rows from
different swarm members, or frames that are not exactly ``lag`` apart, are never paired.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import contracts as C
from .models import ResidualFit, _design, standardised_anchor

#: Sarle's bimodality coefficient of a uniform distribution; above it a sample is
#: conventionally read as bimodal.
BIMODALITY_THRESHOLD = 5.0 / 9.0


def lag_pairs(member_ids, frame_index, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Row indices (i, j) with the same member and frame_index[j] == frame_index[i] + lag."""
    lag = int(lag)
    if lag < 1:
        raise ValueError("lag must be at least one frame")
    _, member = np.unique(np.asarray(member_ids), return_inverse=True)
    frame = np.asarray(frame_index, dtype=np.int64)
    if frame.shape != member.shape:
        raise ValueError("member_ids and frame_index must have equal length")
    span = int(frame.max()) + lag + 1
    key = member.astype(np.int64) * span + frame
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    pos = np.searchsorted(sorted_key, key + lag)
    pos = np.minimum(pos, sorted_key.size - 1)
    hit = sorted_key[pos] == key + lag
    return np.flatnonzero(hit), order[pos[hit]]


def _normalised(weights, n: int) -> np.ndarray:
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (n,) or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative and not all zero")
    return w / w.sum()


def _residual(fit: ResidualFit, X, anchor) -> np.ndarray:
    t = standardised_anchor(fit, anchor)
    return np.asarray(X, dtype=np.float64) - _design(t, 2) @ fit.coefficients - fit.residual_mean


def fit_conditional_tica(fit: ResidualFit, X, anchor, member_ids, frame_index, *, lag: int,
                         n_modes: int = C.MAX_TICA_COMPONENTS, weights=None) -> ResidualFit:
    """Return ``fit`` with its slowest ``n_modes`` conditional tICA directions appended.

    The residual is the fit's own (same regression on T(a), same weighted mean), so a tICA
    component differs from a PCA one only in its direction. Lagged pairs are weighted by the
    design weight of their first frame; the time-lagged covariance is symmetrised (reversible
    estimator). Directions are unit vectors, sign-fixed like the PCA ones, standardised on the
    design measure.
    """
    if fit.n_components + int(n_modes) > C.MAX_COMPONENT_INDEX:
        raise ValueError("too many components for the candidate-set contract")
    R = _residual(fit, X, anchor)
    n, d = R.shape
    w = _normalised(weights, n)
    i, j = lag_pairs(member_ids, frame_index, lag)
    if i.size < 10 * d:
        raise ValueError(f"only {i.size} lag-{lag} frame pairs; too few for conditional tICA")
    pw = w[i] / w[i].sum()
    # centre on the lagged-pair set itself (both ends), not on all rows, so the covariances
    # carry no rank-one mean offset when pairs under-sample the ends of trajectories
    pair_mean = 0.5 * (pw @ R[i] + pw @ R[j])
    x, y = R[i] - pair_mean, R[j] - pair_mean
    C0 = 0.5 * ((x * pw[:, None]).T @ x + (y * pw[:, None]).T @ y)
    Ct = 0.5 * ((x * pw[:, None]).T @ y + (y * pw[:, None]).T @ x)
    lam, V = np.linalg.eigh(C0)
    keep = lam > max(float(lam.max()), 1e-300) * 1e-10
    W = V[:, keep] / np.sqrt(lam[keep])
    ev, U = np.linalg.eigh(W.T @ Ct @ W)
    order = np.argsort(ev)[::-1][: int(n_modes)]
    vectors, eigen = [], []
    for k in order:
        v = W @ U[:, k]
        v = v / np.linalg.norm(v)
        pivot = v[np.argmax(np.abs(v))]
        vectors.append(v * (np.sign(pivot) if pivot != 0.0 else 1.0))
        eigen.append(float(ev[k]))
    V2 = np.asarray(vectors)
    scores = R @ V2.T
    mu = np.sum(w[:, None] * scores, axis=0)
    sd = np.sqrt(np.sum(w[:, None] * (scores - mu) ** 2, axis=0))
    if np.any(sd <= 1e-12):
        raise ValueError("a conditional tICA mode has zero variance on the training data")
    m = len(eigen)
    return replace(
        fit,
        singular_values=np.concatenate([fit.singular_values, sd]),
        right_vectors=np.vstack([fit.right_vectors, V2]),
        projection_mean=np.concatenate([fit.projection_mean, mu]),
        projection_std=np.concatenate([fit.projection_std, sd]),
        families=tuple(fit.families) + (C.COMPONENT_FAMILY_TICA,) * m,
        tica_lag_frames=tuple(fit.tica_lag_frames) + (int(lag),) * m,
        tica_eigenvalues=tuple(fit.tica_eigenvalues) + tuple(eigen),
    )


def anchor_cells(anchor, n_cells: int) -> np.ndarray:
    """Equal-count cells of the anchor (CV1)."""
    a = np.asarray(anchor, dtype=np.float64)
    edges = np.quantile(a, np.linspace(0.0, 1.0, int(n_cells) + 1))
    return np.clip(np.searchsorted(edges, a, side="right") - 1, 0, int(n_cells) - 1)


def conditional_autocorrelation(z, cells, member_ids, frame_index, lag: int, weights=None) -> float:
    """Lag autocorrelation of z after removing its mean within each CV1 cell (pooled over
    members; pairs weighted by the first frame's design weight). NaN without pairs."""
    z = np.asarray(z, dtype=np.float64)
    w = _normalised(weights, z.size)
    cells = np.asarray(cells)
    mean = np.zeros(int(cells.max()) + 1)
    for c in np.unique(cells):
        m = cells == c
        mean[c] = np.sum(w[m] * z[m]) / np.sum(w[m])
    r = z - mean[cells]
    i, j = lag_pairs(member_ids, frame_index, lag)
    if i.size == 0:
        return float("nan")
    pw = w[i]
    num = float(np.sum(pw * r[i] * r[j]))
    den = float(0.5 * np.sum(pw * (r[i] ** 2 + r[j] ** 2)))
    return num / den if den > 0 else float("nan")


def bimodality_coefficient(v) -> float:
    """Sarle's sample bimodality coefficient (small-sample corrected); NaN below 4 samples."""
    v = np.asarray(v, dtype=np.float64)
    n = v.size
    if n < 4:
        return float("nan")
    d = v - v.mean()
    m2 = np.mean(d ** 2)
    if m2 <= 0:
        return float("nan")
    g = np.mean(d ** 3) / m2 ** 1.5 * np.sqrt(n * (n - 1)) / (n - 2)
    k = (np.mean(d ** 4) / m2 ** 2 - 3.0)
    k = ((n + 1) * k + 6.0) * (n - 1) / ((n - 2) * (n - 3))
    return float((g ** 2 + 1.0) / (k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def max_bimodality_at_fixed_anchor(z, cells, min_rows: int = 200) -> float:
    """Largest bimodality coefficient of z over CV1 cells holding at least ``min_rows`` rows."""
    z = np.asarray(z, dtype=np.float64)
    cells = np.asarray(cells)
    vals = [bimodality_coefficient(z[cells == c]) for c in np.unique(cells) if (cells == c).sum() >= min_rows]
    vals = [v for v in vals if np.isfinite(v)]
    return float(max(vals)) if vals else float("nan")
