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
    "compute_tica",
    "project_tica1",
    "window_tica_centers",
    "tica_k_from_spread",
    "DihedralObsBuffer",
    "load_epoch_dihedral_obs",
    "compute_tica_from_epoch_obs",
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
    """
    weights: np.ndarray
    eigenvalue: float
    mean: np.ndarray
    offset: float
    lag: int
    phi_torsion_indices: List[Tuple[int, int, int, int]]
    psi_torsion_indices: List[Tuple[int, int, int, int]]
    n_samples: int = 0
    sign_ref: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d: dict = {
            "eigenvalue": float(self.eigenvalue),
            "lag": int(self.lag),
            "n_samples": int(self.n_samples),
            "mean": self.mean.tolist(),
            "weights": self.weights.tolist(),
            "offset": float(self.offset),
            "phi_torsion_indices": [list(t) for t in self.phi_torsion_indices],
            "psi_torsion_indices": [list(t) for t in self.psi_torsion_indices],
        }
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
            lag=int(d["lag"]),
            phi_torsion_indices=[tuple(t) for t in d["phi_torsion_indices"]],
            psi_torsion_indices=[tuple(t) for t in d["psi_torsion_indices"]],
            n_samples=int(d.get("n_samples", 0)),
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

def compute_tica(
    X: np.ndarray,
    lag: int,
    *,
    phi_torsion_indices: Optional[List] = None,
    psi_torsion_indices: Optional[List] = None,
    previous_result: Optional[TICAResult] = None,
    epsilon: float = 1e-10,
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

    Returns
    -------
    TICAResult
        The fitted model for tIC1 (slowest mode).

    Raises
    ------
    ValueError
        If ``lag`` is not positive or X has fewer rows than 2*lag.
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

    mean = X.mean(axis=0)
    Xc = X - mean

    # Unlagged covariance C(0)
    C0 = (Xc.T @ Xc) / (n - 1)
    C0 += epsilon * np.eye(d)

    # Lagged covariance C(lag) — symmetrised
    Xl = Xc[:-lag]
    Xr = Xc[lag:]
    Ctau = (Xl.T @ Xr) / (n - lag - 1)
    Ctau = 0.5 * (Ctau + Ctau.T)

    # Generalised eigenvalue problem via scipy, numpy fallback
    try:
        from scipy.linalg import eigh as scipy_eigh
        eigenvalues, eigenvectors = scipy_eigh(Ctau, C0, subset_by_index=[d - 1, d - 1])
    except Exception:
        # Numpy fallback: transform generalized to standard EVP via Cholesky.
        # C0 = L@L.T → A = L^{-1}@Ctau@L^{-T} has same spectrum; eigvecs transform back.
        L = np.linalg.cholesky(C0)
        L_inv = np.linalg.inv(L)
        A = L_inv @ Ctau @ L_inv.T
        A = 0.5 * (A + A.T)
        eigenvalues, eigvec_u = np.linalg.eigh(A)
        eigenvectors = L_inv.T @ eigvec_u

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
        n_samples=n,
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

    def record(self, positions_nm: np.ndarray, step: int, window: int) -> None:
        """Record backbone dihedral features for one frame.

        Parameters
        ----------
        positions_nm : np.ndarray, shape (n_atoms, 3)
            Current atom positions in nanometres.
        step : int
            Absolute simulation step index.
        window : int
            Active umbrella window index biasing this replica.
        """
        feat = backbone_dihedral_features(positions_nm, self._phi, self._psi)
        self._features.append(feat)
        self._steps.append(int(step))
        self._windows.append(int(window))

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
        )
        return out_path

    def clear(self) -> None:
        """Discard buffered observations without saving."""
        self._features.clear()
        self._steps.clear()
        self._windows.clear()

    def __len__(self) -> int:
        return len(self._features)


# ---------------------------------------------------------------------------
# Epoch-level I/O helpers
# ---------------------------------------------------------------------------

def load_epoch_dihedral_obs(epoch_dir) -> Tuple[np.ndarray, np.ndarray]:
    """Load and concatenate all replica dihedral observations from an epoch.

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
    for npz_path in npz_files:
        data = np.load(npz_path)
        feature_chunks.append(data["features"])
        window_chunks.append(data["window"])
    X = np.concatenate(feature_chunks, axis=0)
    window_ids = np.concatenate(window_chunks, axis=0)
    return X, window_ids


def compute_tica_from_epoch_obs(
    epoch_dir,
    lag_frames: int,
    phi_torsion_indices: List[Tuple[int, int, int, int]],
    psi_torsion_indices: List[Tuple[int, int, int, int]],
    *,
    previous_result: Optional[TICAResult] = None,
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

    Returns
    -------
    TICAResult
    """
    X, _ = load_epoch_dihedral_obs(epoch_dir)
    return compute_tica(
        X,
        lag_frames,
        phi_torsion_indices=phi_torsion_indices,
        psi_torsion_indices=psi_torsion_indices,
        previous_result=previous_result,
    )
