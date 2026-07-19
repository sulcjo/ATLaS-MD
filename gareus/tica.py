"""Inter-epoch tICA CVaux module — pure numpy, no OpenMM.

Computes time-lagged Independent Component Analysis (tICA) over backbone
sin/cos dihedral features collected during adaptive umbrella sampling epochs,
then exposes the slowest tIC as an optional secondary CV for the next epoch.

All entry points are opt-in: none of this module's logic runs unless
``tica_obs_interval > 0`` is set in the YAML config.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "TICAResult",
    "backbone_dihedral_features",
    "compute_bootstrap_torsion_pca",
    "compute_tica",
    "compute_tica_components",
    "compute_tica_combined",
    "project_tica1",
    "window_tica_centers",
    "tica_k_from_spread",
    "DihedralObsBuffer",
    "load_epoch_dihedral_obs",
    "compute_tica_from_epoch_obs",
    "compute_combined_tica_from_epoch_obs",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TICAResult:
    """Fitted tICA model for the slowest mode (tIC1).

    Attributes
    ----------
    weights : np.ndarray, shape (n_features,)
        Eigenvector coefficients in feature space.  Already mean-centred so
        ``project_tica1(X, result) == X @ weights + offset``.
    eigenvalue : float
        Lagged autocorrelation of tIC1 (between 0 and 1 for positive lags).
    mean : np.ndarray, shape (n_features,)
        Feature mean used to centre X before projection.
    offset : float
        Scalar offset so that ``X @ weights + offset == (X - mean) @ weights``.
    lag : int
        Lag in frames used during fitting.
    phi_torsion_indices : List[Tuple[int,int,int,int]]
        Atom-index 4-tuples defining backbone phi torsions.
    psi_torsion_indices : List[Tuple[int,int,int,int]]
        Atom-index 4-tuples defining backbone psi torsions.
    n_samples : int
        Number of rows used to fit this model.
    method : str
        Fit family used to produce this state, e.g. ``tica`` or ``pca``.
    explained_variance_ratio : float, optional
        Fraction of total variance explained by the returned component.
    """
    weights: np.ndarray
    eigenvalue: float
    mean: np.ndarray
    offset: float
    lag: int
    phi_torsion_indices: List[Tuple[int, int, int, int]]
    psi_torsion_indices: List[Tuple[int, int, int, int]]
    n_samples: int = 0
    method: str = "tica"
    explained_variance_ratio: Optional[float] = None
    sign_ref: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d: dict = {
            "eigenvalue": float(self.eigenvalue),
            "method": str(self.method or "tica"),
            "lag": int(self.lag),
            "n_samples": int(self.n_samples),
            "mean": self.mean.tolist(),
            "weights": self.weights.tolist(),
            "offset": float(self.offset),
            "phi_torsion_indices": [list(t) for t in self.phi_torsion_indices],
            "psi_torsion_indices": [list(t) for t in self.psi_torsion_indices],
        }
        if self.explained_variance_ratio is not None:
            d["explained_variance_ratio"] = float(self.explained_variance_ratio)
        if self.sign_ref is not None:
            d["sign_ref"] = self.sign_ref.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TICAResult":
        weights = np.asarray(d["weights"], dtype=np.float64)
        mean = np.asarray(d["mean"], dtype=np.float64)
        sign_ref = np.asarray(d["sign_ref"], dtype=np.float64) if "sign_ref" in d else None
        return cls(
            weights=weights,
            eigenvalue=float(d["eigenvalue"]),
            mean=mean,
            offset=float(d["offset"]),
            lag=int(d.get("lag", 0 if d.get("method") == "pca" else 1)),
            phi_torsion_indices=[tuple(t) for t in d.get("phi_torsion_indices", [])],
            psi_torsion_indices=[tuple(t) for t in d.get("psi_torsion_indices", [])],
            n_samples=int(d.get("n_samples", 0)),
            method=str(d.get("method", "tica") or "tica"),
            explained_variance_ratio=(
                float(d["explained_variance_ratio"])
                if d.get("explained_variance_ratio") is not None
                else None
            ),
            sign_ref=sign_ref,
        )

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path) -> "TICAResult":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


# ---------------------------------------------------------------------------
# Dihedral geometry
# ---------------------------------------------------------------------------

def _dihedral_rad(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute torsion angle (Praxelis convention) in radians, range (-π, π]."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2_norm = b2 / (np.linalg.norm(b2) + 1e-30)
    m1 = np.cross(n1, b2_norm)
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return float(np.arctan2(y, x))


def backbone_dihedral_features(
    positions_nm: np.ndarray,
    phi_torsions: List[Tuple[int, int, int, int]],
    psi_torsions: List[Tuple[int, int, int, int]],
) -> np.ndarray:
    """Convert backbone torsion atom-indices to sin/cos feature vector.

    Feature ordering: [sin_phi_0, cos_phi_0, ..., sin_phi_N, cos_phi_N,
                       sin_psi_0, cos_psi_0, ..., sin_psi_M, cos_psi_M]

    Parameters
    ----------
    positions_nm : np.ndarray, shape (n_atoms, 3)
        Atom positions in nanometres.
    phi_torsions, psi_torsions : list of (i,j,k,l) tuples
        Backbone atom-index quadruplets.

    Returns
    -------
    np.ndarray, shape (2*len(phi) + 2*len(psi),)
    """
    feats: list = []
    for quart in phi_torsions:
        ang = _dihedral_rad(
            positions_nm[quart[0]],
            positions_nm[quart[1]],
            positions_nm[quart[2]],
            positions_nm[quart[3]],
        )
        feats.append(float(np.sin(ang)))
        feats.append(float(np.cos(ang)))
    for quart in psi_torsions:
        ang = _dihedral_rad(
            positions_nm[quart[0]],
            positions_nm[quart[1]],
            positions_nm[quart[2]],
            positions_nm[quart[3]],
        )
        feats.append(float(np.sin(ang)))
        feats.append(float(np.cos(ang)))
    return np.asarray(feats, dtype=np.float64)


# ---------------------------------------------------------------------------
# tICA core
# ---------------------------------------------------------------------------

def _normalize_segments(segments: Optional[np.ndarray], n: int) -> np.ndarray:
    """Normalise a ``segments`` argument to an array of segment lengths.

    Accepts either:
      * ``None`` — the whole array is treated as a single segment.
      * A 1-D array/list of segment lengths (one entry per trajectory
        segment, in concatenation order), summing to ``n``.
      * A per-frame segment-id array of length ``n`` (values need not be
        contiguous integers, but each distinct id must occupy a single
        contiguous run — i.e. the array must already be grouped by
        trajectory segment, which is guaranteed by concatenation order).

    The lengths interpretation is tried FIRST whenever it is
    self-consistent (all positive and summing to ``n``); the per-frame-id
    interpretation is only used as a fallback when the array does not
    parse as a valid lengths array. This avoids ambiguity in the
    degenerate case where the number of segments happens to equal ``n``
    (e.g. every trajectory contributed exactly one frame): a lengths
    array of ``n`` ones must NOT be misread as a single per-frame id run
    covering all ``n`` frames, which would silently reintroduce
    cross-segment lagged pairs.

    Returns
    -------
    np.ndarray, dtype int64
        Segment lengths, in order, summing to ``n``.
    """
    if segments is None:
        return np.asarray([n], dtype=np.int64)
    arr = np.asarray(segments)
    if arr.ndim != 1:
        raise ValueError(f"segments must be 1-D, got shape {arr.shape}")

    as_lengths = arr.astype(np.int64)
    if as_lengths.shape[0] > 0 and as_lengths.sum() == n and np.all(as_lengths > 0):
        return as_lengths

    if arr.shape[0] == n:
        # Per-frame segment-id array: run-length encode contiguous blocks.
        lengths: List[int] = []
        seen_ids = set()
        current_id = arr[0]
        current_len = 1
        seen_ids.add(current_id)
        for val in arr[1:]:
            if val == current_id:
                current_len += 1
            else:
                if val in seen_ids:
                    raise ValueError(
                        f"segment id {val!r} appears in non-contiguous blocks; "
                        "per-frame segment ids must be grouped by trajectory segment"
                    )
                lengths.append(current_len)
                seen_ids.add(val)
                current_id = val
                current_len = 1
        lengths.append(current_len)
        return np.asarray(lengths, dtype=np.int64)

    if as_lengths.sum() != n:
        raise ValueError(
            f"segments sum to {int(as_lengths.sum())}, expected n_samples={n}"
        )
    raise ValueError("segment lengths must all be positive")


def _segment_lagged_pair_indices(lengths: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build within-segment lagged-pair frame indices.

    For each contiguous segment of length ``L`` (offset ``s`` in the
    concatenated array), valid pairs are ``(s+t, s+t+lag)`` for
    ``t in [0, L-lag)``. Segments with ``L <= lag`` contribute zero pairs.
    Pairs are never formed across segment boundaries.

    Returns
    -------
    (left, right) : np.ndarray, np.ndarray
        Integer index arrays into the concatenated array, same length,
        such that ``right = left + lag`` and every ``(left[i], right[i])``
        pair lies within a single segment.
    """
    left_parts: List[np.ndarray] = []
    offset = 0
    for L in lengths:
        L = int(L)
        if L > lag:
            left_parts.append(np.arange(offset, offset + L - lag, dtype=np.int64))
        offset += L
    if left_parts:
        left = np.concatenate(left_parts)
    else:
        left = np.asarray([], dtype=np.int64)
    right = left + lag
    return left, right


def _tica_covariance_matrices(
    X: np.ndarray,
    lag: int,
    *,
    epsilon: float = 1e-10,
    weights: Optional[np.ndarray] = None,
    segments: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build the C(0)/C(tau) lagged-covariance pair shared by every tICA fit.

    Extracted so the single-component (production) and multi-component
    (analysis-only) fitters can never diverge in how C(0)/C(tau) are
    estimated — only in how many eigenpairs each one keeps afterward.

    Returns
    -------
    C0, Ctau : np.ndarray, shape (d, d)
    mean : np.ndarray, shape (d,)
    d : int
        Feature dimensionality.

    Raises
    ------
    ValueError
        If ``lag`` is not positive, X has fewer rows than 2*lag, or no
        segment has more than ``lag`` frames (zero valid within-segment
        lagged pairs).
    """
    if int(lag) <= 0:
        raise ValueError(f"lag must be a positive integer, got {lag!r}")
    lag = int(lag)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    n, d = X.shape
    if n <= 2 * lag:
        raise ValueError(
            f"X has only {n} rows; need > 2*lag={2*lag} for a valid lagged covariance estimate"
        )

    lengths = _normalize_segments(segments, n)
    left, right = _segment_lagged_pair_indices(lengths, lag)
    n_pairs = int(left.shape[0])
    if n_pairs < 2:
        raise ValueError(
            f"too few valid within-segment lagged pairs ({n_pairs}) for a "
            f"stable tICA C(tau) estimate: need >= 2, got segment lengths "
            f"{lengths.tolist()} with lag={lag} (total n={n}); pairs are "
            "never built across trajectory-segment boundaries, so no "
            "segment longer than lag means zero valid pairs"
        )

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n,):
            raise ValueError(f"weights shape {w.shape} != (n_samples,) = ({n},)")
        w_sum = w.sum()
        if w_sum <= 0.0:
            raise ValueError("weights must sum to a positive value")
        w = w / w_sum
    else:
        w = None

    if w is None:
        # Standard unweighted tICA
        mean = X.mean(axis=0)
        Xc = X - mean
        C0 = (Xc.T @ Xc) / (n - 1)
        C0 += epsilon * np.eye(d)
        Xl = Xc[left]
        Xr = Xc[right]
        Ctau = (Xl.T @ Xr) / (n_pairs - 1)
        Ctau = 0.5 * (Ctau + Ctau.T)
    else:
        # Reweighted tICA: symmetric estimator with pair weights
        # (Nüske et al. 2017 / standard MBAR-reweighted TICA form)
        mean = w @ X  # weighted mean, shape (d,)
        Xc = X - mean
        Xl = Xc[left]
        Xr = Xc[right]
        # Arithmetic-mean pair weights: each pair (t, t+lag) gets
        # the average of source and target frame weights, then renormalised.
        w_pairs = 0.5 * (w[left] + w[right])
        w_pairs = w_pairs / w_pairs.sum()
        # Symmetric weighted C(0): average contribution from source and target
        C0 = 0.5 * ((Xl.T * w_pairs) @ Xl + (Xr.T * w_pairs) @ Xr)
        C0 += epsilon * np.eye(d)
        # Symmetric weighted C(tau): symmetrise to enforce time-reversibility
        Ctau = 0.5 * ((Xl.T * w_pairs) @ Xr + (Xr.T * w_pairs) @ Xl)

    return C0, Ctau, mean, d


def _tica_generalized_eigh(C0: np.ndarray, Ctau: np.ndarray, d: int, n_keep: int):
    """Solve C(tau) v = lambda C(0) v, returning the top ``n_keep`` pairs.

    Returned eigenvalues/eigenvectors are NOT guaranteed sorted descending
    (mirrors scipy's ``subset_by_index`` contract, which returns ascending
    order) — callers must sort/select by eigenvalue themselves.
    """
    try:
        from scipy.linalg import eigh as scipy_eigh
        return scipy_eigh(Ctau, C0, subset_by_index=[d - n_keep, d - 1])
    except Exception:
        # Numpy fallback: transform generalized to standard EVP via Cholesky.
        # C0 = L@L.T → A = L^{-1}@Ctau@L^{-T} has same spectrum; eigvecs transform back.
        L = np.linalg.cholesky(C0)
        L_inv = np.linalg.inv(L)
        A = L_inv @ Ctau @ L_inv.T
        A = 0.5 * (A + A.T)
        eigenvalues, eigvec_u = np.linalg.eigh(A)
        eigenvectors = L_inv.T @ eigvec_u
        return eigenvalues, eigenvectors


def compute_tica(
    X: np.ndarray,
    lag: int,
    *,
    phi_torsion_indices: Optional[List] = None,
    psi_torsion_indices: Optional[List] = None,
    previous_result: Optional[TICAResult] = None,
    epsilon: float = 1e-10,
    weights: Optional[np.ndarray] = None,
    segments: Optional[np.ndarray] = None,
) -> TICAResult:
    """Fit a tICA model to feature matrix X and return the slowest mode.

    Solves the generalised eigenvalue problem:
        C(tau) @ v = lambda * C(0) @ v

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Row-major feature matrix.
    lag : int > 0
        Time lag in frames.
    phi_torsion_indices, psi_torsion_indices : lists, optional
        Stored on the returned TICAResult for downstream force-building.
    previous_result : TICAResult, optional
        If given, enforce sign continuity (flip eigenvector if the new one
        anti-correlates with the previous weight vector).
    epsilon : float
        Small regularisation added to C(0) diagonal for numerical stability.
    weights : np.ndarray, shape (n_samples,), optional
        Importance weights for each frame (e.g. MBAR weights to reweight
        the biased REUS ensemble to the unbiased equilibrium distribution).
        Must be non-negative; are normalised internally to sum to 1.
        When None, standard unweighted tICA is used.
    segments : np.ndarray or list, optional
        Trajectory-segment boundaries within the concatenated ``X`` (e.g.
        one segment per replica trajectory file). Either a 1-D array of
        segment lengths (summing to ``n_samples``) or a per-frame
        segment-id array of length ``n_samples`` (each id must occupy one
        contiguous run). When given, lagged pairs ``(x_t, x_{t+lag})`` are
        built ONLY within each segment — pairs never cross a segment
        boundary, so a join between two different replica trajectories
        cannot masquerade as a real transition in C(tau).  A segment with
        length <= lag contributes zero pairs. When ``None`` (default), the
        whole array is treated as a single segment, reproducing the
        previous (pre-fix) behaviour exactly.

    Returns
    -------
    TICAResult
        The fitted model for tIC1 (slowest mode).

    Raises
    ------
    ValueError
        If ``lag`` is not positive, X has fewer rows than 2*lag, or no
        segment has more than ``lag`` frames (zero valid within-segment
        lagged pairs).
    """
    lag = int(lag)
    C0, Ctau, mean, d = _tica_covariance_matrices(
        X, lag, epsilon=epsilon, weights=weights, segments=segments
    )

    eigenvalues, eigenvectors = _tica_generalized_eigh(C0, Ctau, d, n_keep=1)

    idx = int(np.argmax(eigenvalues))
    v = eigenvectors[:, idx].copy()
    ev = float(eigenvalues[idx])

    # Normalise to unit C(0)-norm
    norm_sq = float(v @ C0 @ v)
    if norm_sq > 0.0:
        v /= np.sqrt(norm_sq)

    # Sign continuity
    if previous_result is not None and previous_result.weights is not None:
        ref = previous_result.sign_ref if previous_result.sign_ref is not None else previous_result.weights
        if float(np.dot(v, ref)) < 0.0:
            v = -v

    offset = float(-mean @ v)
    return TICAResult(
        weights=v,
        eigenvalue=ev,
        mean=mean,
        offset=offset,
        lag=lag,
        phi_torsion_indices=list(phi_torsion_indices or []),
        psi_torsion_indices=list(psi_torsion_indices or []),
        n_samples=X.shape[0],
        method="tica",
        sign_ref=v.copy(),
    )


def compute_tica_components(
    X: np.ndarray,
    lag: int,
    n_components: int = 5,
    *,
    phi_torsion_indices: Optional[List] = None,
    psi_torsion_indices: Optional[List] = None,
    epsilon: float = 1e-10,
    weights: Optional[np.ndarray] = None,
    segments: Optional[np.ndarray] = None,
) -> List[TICAResult]:
    """Fit tICA and return the top ``n_components`` modes, not just tIC1.

    Analysis-only sibling of :func:`compute_tica` — same C(0)/C(tau)
    estimate (via :func:`_tica_covariance_matrices`), but keeps the top
    ``n_components`` eigenpairs instead of discarding all but the slowest.
    The live GaMD/REUS runtime CV force always uses :func:`compute_tica`'s
    single tIC1 result; this function exists for offline multi-component
    analysis (e.g. per-component pseudo-trajectory sweeps) and does not
    change production behaviour.

    Returns
    -------
    list of TICAResult, ordered by descending eigenvalue (tIC1 first).
    """
    lag = int(lag)
    C0, Ctau, mean, d = _tica_covariance_matrices(
        X, lag, epsilon=epsilon, weights=weights, segments=segments
    )
    n_components = max(1, min(int(n_components), d))
    eigenvalues, eigenvectors = _tica_generalized_eigh(C0, Ctau, d, n_keep=n_components)

    order = np.argsort(eigenvalues)[::-1]
    results: List[TICAResult] = []
    for idx in order:
        v = eigenvectors[:, idx].copy()
        ev = float(eigenvalues[idx])

        norm_sq = float(v @ C0 @ v)
        if norm_sq > 0.0:
            v /= np.sqrt(norm_sq)

        # Canonical sign per component: largest-magnitude coefficient positive
        # (matches compute_bootstrap_torsion_pca's convention; there is no
        # cross-epoch "previous_result" to chain sign continuity against here).
        pivot = int(np.argmax(np.abs(v)))
        if v[pivot] < 0.0:
            v = -v

        offset = float(-mean @ v)
        results.append(TICAResult(
            weights=v,
            eigenvalue=ev,
            mean=mean,
            offset=offset,
            lag=lag,
            phi_torsion_indices=list(phi_torsion_indices or []),
            psi_torsion_indices=list(psi_torsion_indices or []),
            n_samples=X.shape[0],
            method="tica",
            sign_ref=v.copy(),
        ))
    return results


def compute_tica_combined(
    X: np.ndarray,
    lag: int,
    n_components: int = 1,
    *,
    phi_torsion_indices: Optional[List] = None,
    psi_torsion_indices: Optional[List] = None,
    previous_result: Optional[TICAResult] = None,
    epsilon: float = 1e-10,
    weights: Optional[np.ndarray] = None,
    segments: Optional[np.ndarray] = None,
) -> TICAResult:
    """Fit tICA and combine the top ``n_components`` modes into one CV2 direction.

    Mirrors :func:`compute_bootstrap_torsion_pca`'s ``component``-as-count
    semantics: the top ``n_components`` tICA eigenvectors (ranked by
    eigenvalue, i.e. by slowness rather than variance) are combined into a
    single direction via an eigenvalue-weighted linear combination
    (coefficient_i = sqrt(eigenvalue_i)), then renormalized under the C(0)
    inner product -- the metric tICA eigenvectors are themselves orthonormal
    under (PCA's combination renormalizes under plain L2 instead, since SVD
    right-singular-vectors are already L2-orthonormal; this is the same idea
    adapted to tICA's natural metric). ``n_components=1`` delegates directly
    to :func:`compute_tica` (bit-for-bit identical output), since a single
    positive scalar coefficient never changes a normalized direction's
    identity -- fully backward compatible with the pre-existing single-mode
    behavior.

    CV2 stays a single scalar restraint either way -- this only lets that one
    restraint draw on more than tIC1 alone (e.g. a slow mode that tIC1's own
    ranking doesn't fully capture).

    Returns
    -------
    TICAResult
        ``eigenvalue`` is the *combined* direction's own autocorrelation
        (v^T C(tau) v, re-derived after combining -- not just tIC1's raw
        eigenvalue), mirroring how compute_bootstrap_torsion_pca re-derives
        the combined direction's actual explained variance.

    Raises
    ------
    ValueError
        If ``n_components`` is not in ``1..d`` (feature count), mirroring
        compute_bootstrap_torsion_pca's explicit bounds check.
    """
    lag = int(lag)
    if int(n_components) <= 1:
        return compute_tica(
            X, lag,
            phi_torsion_indices=phi_torsion_indices,
            psi_torsion_indices=psi_torsion_indices,
            previous_result=previous_result,
            epsilon=epsilon,
            weights=weights,
            segments=segments,
        )

    C0, Ctau, mean, d = _tica_covariance_matrices(
        X, lag, epsilon=epsilon, weights=weights, segments=segments
    )
    if int(n_components) > d:
        raise ValueError(f"n_components must be in 1..{d}, got {n_components}")

    eigenvalues, eigenvectors = _tica_generalized_eigh(C0, Ctau, d, n_keep=int(n_components))
    order = np.argsort(eigenvalues)[::-1]

    combined = np.zeros(d, dtype=np.float64)
    for idx in order:
        v = eigenvectors[:, idx].copy()
        norm_sq = float(v @ C0 @ v)
        if norm_sq > 0.0:
            v /= np.sqrt(norm_sq)
        coeff = np.sqrt(max(float(eigenvalues[idx]), 0.0))
        combined += coeff * v

    comb_norm_sq = float(combined @ C0 @ combined)
    if comb_norm_sq <= float(epsilon):
        raise ValueError("zero combined tICA component norm")
    combined /= np.sqrt(comb_norm_sq)

    # Honest eigenvalue for the *combined* direction, mirroring PCA's
    # re-derived explained-variance step -- not just the top component's own
    # eigenvalue (combining can only match or reduce tIC1-alone autocorrelation).
    combined_eigenvalue = float(combined @ Ctau @ combined)

    if previous_result is not None and previous_result.weights is not None:
        ref = previous_result.sign_ref if previous_result.sign_ref is not None else previous_result.weights
        if float(np.dot(combined, ref)) < 0.0:
            combined = -combined
    else:
        pivot = int(np.argmax(np.abs(combined)))
        if combined[pivot] < 0.0:
            combined = -combined

    offset = float(-mean @ combined)
    return TICAResult(
        weights=combined,
        eigenvalue=combined_eigenvalue,
        mean=mean,
        offset=offset,
        lag=lag,
        phi_torsion_indices=list(phi_torsion_indices or []),
        psi_torsion_indices=list(psi_torsion_indices or []),
        n_samples=X.shape[0],
        method="tica",
        sign_ref=combined.copy(),
    )


def compute_bootstrap_torsion_pca(
    X: np.ndarray,
    cv1: Optional[np.ndarray] = None,
    *,
    residualize: bool = True,
    component: int = 1,
    phi_torsion_indices: Optional[List] = None,
    psi_torsion_indices: Optional[List] = None,
    epsilon: float = 1e-12,
) -> TICAResult:
    """Fit a PCA model to seed conformer backbone features.

    ``component`` is a count, not a single index: the top ``component``
    principal components (by variance, 1..component) are combined into one
    unit-norm CV direction via a variance-weighted linear combination
    (coefficient_i = sqrt(variance_i), then renormalized). This keeps CV2 a
    single scalar restraint while letting it draw on more than the single
    top mode. ``component=1`` reduces exactly to "use PC1 alone" (a positive
    scalar coefficient does not change a direction's normalized identity),
    so this is fully backward compatible with the old single-component
    behavior.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    n, d = X.shape
    if n < 2:
        raise ValueError(f"bootstrap torsion PCA needs at least 2 samples, got {n}")
    if d < 1:
        raise ValueError("bootstrap torsion PCA needs at least 1 feature")
    if not np.isfinite(X).all():
        raise ValueError("bootstrap torsion PCA feature matrix contains non-finite values")

    X_raw = X.copy()
    X_work = X_raw.copy()
    raw_mean = X_raw.mean(axis=0)
    cv1_slope = None
    if residualize:
        if cv1 is None:
            raise ValueError("cv1 values are required when residualize=True")
        cv1_arr = np.asarray(cv1, dtype=np.float64)
        if cv1_arr.shape != (n,):
            raise ValueError(f"cv1 shape {cv1_arr.shape} != ({n},)")
        if not np.isfinite(cv1_arr).all():
            raise ValueError("cv1 values contain non-finite values")
        design = np.column_stack([np.ones(n, dtype=np.float64), cv1_arr])
        beta, *_ = np.linalg.lstsq(design, X_work, rcond=None)
        cv1_slope = np.asarray(beta[1], dtype=np.float64)
        X_work = X_work - design @ beta

    mean = X_work.mean(axis=0)
    Xc = X_work - mean
    _, singular_values, vt = np.linalg.svd(Xc, full_matrices=False)
    variances = (singular_values * singular_values) / max(1, n - 1)
    total_variance = float(np.sum(variances))
    if total_variance <= float(epsilon):
        raise ValueError("zero bootstrap torsion PCA variance")
    n_top = int(component)
    if n_top < 1 or n_top > vt.shape[0]:
        raise ValueError(f"component must be in 1..{vt.shape[0]}, got {component}")
    selected_variances = variances[:n_top]
    coeffs = np.sqrt(np.clip(selected_variances, 0.0, None))
    v_raw = (coeffs[:, None] * vt[:n_top]).sum(axis=0)
    norm = float(np.linalg.norm(v_raw))
    if norm <= float(epsilon):
        raise ValueError("zero bootstrap torsion PCA component norm")
    v = v_raw / norm
    explained = float(np.var(Xc @ v, ddof=1) / total_variance) if n > 1 else 0.0
    if explained <= float(epsilon):
        raise ValueError("zero bootstrap torsion PCA variance")
    if cv1_slope is not None:
        slope_norm_sq = float(cv1_slope @ cv1_slope)
        if slope_norm_sq > float(epsilon):
            v = v - cv1_slope * (float(v @ cv1_slope) / slope_norm_sq)
            norm = float(np.linalg.norm(v))
            if norm <= float(epsilon):
                raise ValueError("bootstrap torsion PCA component collapsed after CV1 decorrelation")
            v = v / norm
            component_variance = float(np.var(Xc @ v, ddof=1)) if n > 1 else 0.0
            if component_variance <= float(epsilon):
                raise ValueError("zero bootstrap torsion PCA variance")
            explained = float(component_variance / total_variance)
    pivot = int(np.argmax(np.abs(v)))
    if v[pivot] < 0.0:
        v = -v
    offset = float(-raw_mean @ v)
    return TICAResult(
        weights=v,
        eigenvalue=explained,
        mean=raw_mean,
        offset=offset,
        lag=0,
        phi_torsion_indices=list(phi_torsion_indices or []),
        psi_torsion_indices=list(psi_torsion_indices or []),
        n_samples=n,
        method="pca",
        explained_variance_ratio=explained,
        sign_ref=v.copy(),
    )


def project_tica1(X: np.ndarray, result: TICAResult) -> np.ndarray:
    """Project feature matrix onto tIC1 using the fitted model.

    The projection is: (X - mean) @ weights = X @ weights + offset.
    The two forms are numerically equivalent; the latter avoids a redundant
    broadcast and is what the OpenMM force expression must match.

    Parameters
    ----------
    X : np.ndarray, shape (..., n_features)
        Feature matrix (rows are frames).
    result : TICAResult

    Returns
    -------
    np.ndarray, shape (...,)
        Projected tIC1 values.
    """
    X = np.asarray(X, dtype=np.float64)
    return X @ result.weights + result.offset


# ---------------------------------------------------------------------------
# Window centre helpers
# ---------------------------------------------------------------------------

def window_tica_centers(
    X: np.ndarray,
    window_ids: np.ndarray,
    result: TICAResult,
    percentile: float = 50.0,
) -> Dict[int, float]:
    """Compute per-window tIC1 median (or any percentile).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    window_ids : np.ndarray, shape (n_samples,)
        Integer window assignment for each sample row.
    result : TICAResult
    percentile : float
        Which percentile to use as the window centre (default 50 = median).

    Returns
    -------
    dict mapping window_id (int) → tIC1 centre (float)
    """
    proj = project_tica1(X, result)
    centers: Dict[int, float] = {}
    for wid in np.unique(window_ids):
        mask = window_ids == wid
        centers[int(wid)] = float(np.percentile(proj[mask], percentile))
    return centers


def tica_k_from_spread(
    X: np.ndarray,
    window_ids: np.ndarray,
    result: TICAResult,
    *,
    target_overlap_sigma: float = 2.0,
    k_min_kcal: float = 0.5,
    k_max_kcal: float = 20.0,
    kB_T_kcal: float = 0.5922,
) -> Dict[int, float]:
    """Estimate per-window spring constants from per-window tIC1 spread.

    Uses the harmonic approximation: k = kB*T / (sigma^2 * target_overlap_sigma^2)

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    window_ids : np.ndarray, shape (n_samples,)
    result : TICAResult
    target_overlap_sigma : float
        Desired overlap in units of window sigma (default 2.0).
    k_min_kcal, k_max_kcal : float
        Clamp bounds in kcal/mol per tIC1 squared unit.
    kB_T_kcal : float
        Thermal energy in kcal/mol (default 0.5922 ≈ 298 K).

    Returns
    -------
    dict mapping window_id (int) → spring constant (float, kcal/mol)
    """
    proj = project_tica1(X, result)
    ks: Dict[int, float] = {}
    for wid in np.unique(window_ids):
        mask = window_ids == wid
        sigma = float(np.std(proj[mask]))
        if sigma < 1e-12:
            ks[int(wid)] = float(k_max_kcal)
        else:
            k = kB_T_kcal / (sigma * target_overlap_sigma) ** 2
            ks[int(wid)] = float(np.clip(k, k_min_kcal, k_max_kcal))
    return ks


# ---------------------------------------------------------------------------
# Dihedral observation buffer
# ---------------------------------------------------------------------------

class DihedralObsBuffer:
    """In-memory buffer that records backbone dihedral features during MD.

    Usage
    -----
    Create one buffer per replica.  Call :meth:`record` in the per-step
    sampling callback when ``step % tica_obs_interval == 0``.  At the end
    of the epoch, call :meth:`save` to flush observations to disk.

    Parameters
    ----------
    phi_torsions, psi_torsions : lists of (i,j,k,l) tuples
        Backbone torsion atom-index quadruplets.
    replica_idx : int
        Used to name the output file ``dihedral_obs_{replica_idx:03d}.npz``.
    out_dir : path-like
        Directory where the ``.npz`` is written.  A ``tica_obs/`` subdir is
        created automatically.
    """

    def __init__(
        self,
        phi_torsions: List[Tuple[int, int, int, int]],
        psi_torsions: List[Tuple[int, int, int, int]],
        replica_idx: int,
        out_dir,
    ) -> None:
        self._phi = phi_torsions
        self._psi = psi_torsions
        self._replica = int(replica_idx)
        self._out_dir = Path(out_dir)
        self._features: List[np.ndarray] = []
        self._steps: List[int] = []
        self._windows: List[int] = []
        self._primary_cv: List[float] = []
        self._secondary_cv: List[float] = []

    def record(
        self,
        positions_nm: np.ndarray,
        step: int,
        window: int,
        *,
        cv_primary: float = float("nan"),
        cv_secondary: float = float("nan"),
    ) -> None:
        """Record backbone dihedral features for one frame.

        Parameters
        ----------
        positions_nm : np.ndarray, shape (n_atoms, 3)
            Current atom positions in nanometres.
        step : int
            Absolute simulation step index.
        window : int
            Active umbrella window index biasing this replica.
        cv_primary : float, optional
            Primary CV value for this frame; used for MBAR reweighting.
        cv_secondary : float, optional
            Secondary CV value (NaN when CV2 is inactive).
        """
        feat = backbone_dihedral_features(positions_nm, self._phi, self._psi)
        self._features.append(feat)
        self._steps.append(int(step))
        self._windows.append(int(window))
        self._primary_cv.append(float(cv_primary))
        self._secondary_cv.append(float(cv_secondary))

    def save(self) -> Optional[Path]:
        """Flush buffered observations to ``tica_obs/dihedral_obs_{replica:03d}.npz``.

        Returns
        -------
        Path to the written file, or None if no observations were recorded.
        """
        if not self._features:
            return None
        tica_dir = self._out_dir / "tica_obs"
        tica_dir.mkdir(parents=True, exist_ok=True)
        out_path = tica_dir / f"dihedral_obs_{self._replica:03d}.npz"
        np.savez_compressed(
            out_path,
            features=np.stack(self._features, axis=0),
            steps=np.asarray(self._steps, dtype=np.int64),
            window=np.asarray(self._windows, dtype=np.int64),
            primary_cv=np.asarray(self._primary_cv, dtype=np.float64),
            secondary_cv=np.asarray(self._secondary_cv, dtype=np.float64),
        )
        return out_path

    def clear(self) -> None:
        """Discard buffered observations without saving."""
        self._features.clear()
        self._steps.clear()
        self._windows.clear()
        self._primary_cv.clear()
        self._secondary_cv.clear()

    def __len__(self) -> int:
        return len(self._features)


# ---------------------------------------------------------------------------
# Epoch-level I/O helpers
# ---------------------------------------------------------------------------

def load_epoch_dihedral_obs(
    epoch_dir,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and concatenate all replica dihedral observations from an epoch.

    Each ``dihedral_obs_*.npz`` file is one contiguous replica MD
    trajectory (window swaps do not break the configurational trajectory:
    a replica's MD is continuous within its file regardless of which
    umbrella currently biases it). The per-file frame counts are returned
    as ``segment_lengths`` so downstream lagged-covariance code (see
    :func:`compute_tica`) can avoid forming spurious lagged pairs across
    the boundary between two different replica trajectories.

    Parameters
    ----------
    epoch_dir : path-like
        Epoch output directory containing ``tica_obs/dihedral_obs_*.npz``.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
        Concatenated feature rows from all replicas.
    window_ids : np.ndarray, shape (n_samples,)
        Corresponding window assignments.
    primary_cv : np.ndarray, shape (n_samples,)
        Primary CV values per frame (NaN if not recorded — old obs files).
    secondary_cv : np.ndarray, shape (n_samples,)
        Secondary CV values per frame (NaN if inactive or not recorded).
    segment_lengths : np.ndarray, shape (n_files,)
        Frame count of each source npz file, in concatenation order. Pass
        this straight through as ``segments=`` to :func:`compute_tica` so
        lagged pairs are never built across a replica-trajectory boundary.

    Raises
    ------
    FileNotFoundError
        If no observation files exist in the expected location.
    """
    tica_dir = Path(epoch_dir) / "tica_obs"
    npz_files = sorted(tica_dir.glob("dihedral_obs_*.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"No dihedral observation files found under {tica_dir}"
        )
    feature_chunks: List[np.ndarray] = []
    window_chunks: List[np.ndarray] = []
    primary_cv_chunks: List[np.ndarray] = []
    secondary_cv_chunks: List[np.ndarray] = []
    segment_lengths: List[int] = []
    for npz_path in npz_files:
        data = np.load(npz_path)
        feats = data["features"]
        n = len(feats)
        feature_chunks.append(feats)
        window_chunks.append(data["window"])
        primary_cv_chunks.append(
            data["primary_cv"] if "primary_cv" in data else np.full(n, np.nan)
        )
        secondary_cv_chunks.append(
            data["secondary_cv"] if "secondary_cv" in data else np.full(n, np.nan)
        )
        segment_lengths.append(int(n))
    X = np.concatenate(feature_chunks, axis=0)
    window_ids = np.concatenate(window_chunks, axis=0)
    primary_cv = np.concatenate(primary_cv_chunks, axis=0)
    secondary_cv = np.concatenate(secondary_cv_chunks, axis=0)
    return X, window_ids, primary_cv, secondary_cv, np.asarray(segment_lengths, dtype=np.int64)


def compute_tica_from_epoch_obs(
    epoch_dir,
    lag_frames: int,
    phi_torsion_indices: List[Tuple[int, int, int, int]],
    psi_torsion_indices: List[Tuple[int, int, int, int]],
    *,
    previous_result: Optional[TICAResult] = None,
    weights: Optional[np.ndarray] = None,
) -> TICAResult:
    """Fit tICA from all dihedral observations in an epoch directory.

    Convenience wrapper around :func:`load_epoch_dihedral_obs` +
    :func:`compute_tica` that also propagates sign continuity.

    Parameters
    ----------
    epoch_dir : path-like
    lag_frames : int
    phi_torsion_indices, psi_torsion_indices : torsion atom-index lists
    previous_result : TICAResult, optional
        Used for sign-continuity check across epochs.
    weights : np.ndarray, shape (n_samples,), optional
        Importance weights per frame (e.g. MBAR weights).  When provided,
        weighted covariance matrices are used; see :func:`compute_tica`.

    Returns
    -------
    TICAResult
    """
    X, _, _, _, segment_lengths = load_epoch_dihedral_obs(epoch_dir)
    return compute_tica(
        X,
        lag_frames,
        phi_torsion_indices=phi_torsion_indices,
        psi_torsion_indices=psi_torsion_indices,
        previous_result=previous_result,
        weights=weights,
        segments=segment_lengths,
    )


def compute_combined_tica_from_epoch_obs(
    epoch_dir,
    lag_frames: int,
    n_components: int,
    phi_torsion_indices: List[Tuple[int, int, int, int]],
    psi_torsion_indices: List[Tuple[int, int, int, int]],
    *,
    previous_result: Optional[TICAResult] = None,
    weights: Optional[np.ndarray] = None,
) -> TICAResult:
    """Fit combined multi-component tICA from all dihedral observations in an epoch.

    Convenience wrapper around :func:`load_epoch_dihedral_obs` +
    :func:`compute_tica_combined`, mirroring
    :func:`compute_tica_from_epoch_obs`. ``n_components=1`` reproduces
    :func:`compute_tica_from_epoch_obs` exactly.

    Parameters
    ----------
    epoch_dir : path-like
    lag_frames : int
    n_components : int
        Count of top tICA modes (by eigenvalue) to combine into CV2's
        direction. 1 = tIC1 alone (backward compatible default).
    phi_torsion_indices, psi_torsion_indices : torsion atom-index lists
    previous_result : TICAResult, optional
        Used for sign-continuity check across epochs.
    weights : np.ndarray, shape (n_samples,), optional
        Importance weights per frame (e.g. MBAR weights).

    Returns
    -------
    TICAResult
    """
    X, _, _, _, segment_lengths = load_epoch_dihedral_obs(epoch_dir)
    return compute_tica_combined(
        X,
        lag_frames,
        n_components,
        phi_torsion_indices=phi_torsion_indices,
        psi_torsion_indices=psi_torsion_indices,
        previous_result=previous_result,
        weights=weights,
        segments=segment_lengths,
    )
