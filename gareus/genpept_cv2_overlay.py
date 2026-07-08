"""Overlay the zeroth-epoch secondary CV (torsion-PCA) onto the GENPEPT pseudo-FES.

The GENPEPT ``pca_pseudo_fes`` landscape lives in a contacts/distance-PCA space
("imaginary CV1/CV2").  The runtime CV2 is a backbone-torsion PCA projection, a
*different* space, so CV2 is not a function of the pseudo-FES axes.  This module
draws CV2 over that landscape as an honest approximation: seed conformers are
scattered at their pseudo-FES coordinates and colored by their true CV2 value,
with fitted iso-CV2 contours shown only when the regression of CV2 onto the
pseudo-FES axes is strong enough to be meaningful.

Heavy dependencies (GENPEPT feature/PCA helpers, OpenMM, matplotlib) are imported
lazily inside the functions that need them so the pure regression layer stays
importable and unit-testable on its own.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Minimum finite samples over which a fitted CV2 field is trustworthy enough to
# draw as iso-contours; below this we fall back to scatter-only.
_MIN_FIT_MARGIN = 1


def _design_matrix(pc1: np.ndarray, pc2: np.ndarray, order: int) -> np.ndarray:
    """Polynomial design matrix for CV2 ~ f(pc1, pc2)."""
    cols = [np.ones_like(pc1), pc1, pc2]
    if order >= 2:
        cols += [pc1 * pc1, pc1 * pc2, pc2 * pc2]
    return np.column_stack(cols)


@dataclass(frozen=True)
class Cv2Field:
    """A fitted CV2 ~ f(pc1, pc2) surface with its goodness-of-fit.

    ``usable`` is True only when the fit is full-rank and backed by enough finite
    samples, i.e. safe to render as iso-contours; otherwise the caller should show
    the CV2-colored scatter alone.
    """

    coef: np.ndarray
    order: int
    r2: float
    usable: bool

    def evaluate(self, pc1, pc2) -> np.ndarray:
        pc1_arr = np.asarray(pc1, dtype=np.float64)
        pc2_arr = np.asarray(pc2, dtype=np.float64)
        flat = _design_matrix(pc1_arr.ravel(), pc2_arr.ravel(), self.order) @ self.coef
        return flat.reshape(pc1_arr.shape)


def fit_cv2_field(pc1, pc2, cv2, order: int = 1) -> Cv2Field:
    """Least-squares fit of CV2 onto pseudo-FES coordinates (pc1, pc2).

    Non-finite samples are dropped.  Returns a :class:`Cv2Field` whose ``usable``
    flag reports whether the fit is full-rank and well-supported enough to trust
    for iso-contours.
    """
    pc1 = np.asarray(pc1, dtype=np.float64).ravel()
    pc2 = np.asarray(pc2, dtype=np.float64).ravel()
    cv2 = np.asarray(cv2, dtype=np.float64).ravel()
    mask = np.isfinite(pc1) & np.isfinite(pc2) & np.isfinite(cv2)
    pc1, pc2, cv2 = pc1[mask], pc2[mask], cv2[mask]

    A = _design_matrix(pc1, pc2, order)
    n_cols = A.shape[1]
    n_pts = A.shape[0]
    min_pts = n_cols + _MIN_FIT_MARGIN
    if n_pts < min_pts:
        return Cv2Field(coef=np.zeros(n_cols), order=order, r2=0.0, usable=False)

    coef, _res, rank, _sv = np.linalg.lstsq(A, cv2, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((cv2 - pred) ** 2))
    ss_tot = float(np.sum((cv2 - cv2.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    usable = bool(rank == n_cols and np.isfinite(r2))
    return Cv2Field(coef=coef, order=order, r2=float(r2), usable=usable)


# --- locating the GENPEPT run directory and its seed PDBs --------------------

def locate_genpept_seed_dir(run_root, run_args: Optional[dict]) -> Optional[Path]:
    """Resolve the GENPEPT seed-conformer directory for a run.

    ``seed_conformers_dir`` in ``run_args.json`` is often a relative name (the
    genpept dir is usually a sibling of the run dir) or a stale absolute path
    from a moved run tree.  Try the recorded path first, then re-root under the
    run dir and its parent.  Returns ``None`` when nothing resolves.
    """
    raw = str((run_args or {}).get("seed_conformers_dir", "") or "")
    if not raw:
        return None
    run_root = Path(run_root)
    p = Path(raw)
    if p.is_absolute():
        candidates = [p, run_root / p.name, run_root.parent / p.name]
    else:
        candidates = [run_root / p, run_root.parent / p, p]
    for cand in candidates:
        if cand.is_dir():
            return cand
    return None


def collect_seed_pdbs(seed_dir) -> list[Path]:
    """Enumerate the PDBs GENPEPT's ``pca_pseudo_fes`` stage would use.

    Delegates to GENPEPT's own ``expand_pdb_dirs`` + ``collect_pdbs`` so we cover
    the same broad population (``basin_hop_minima``, survivor seeds, candidate
    fallback, adaptive rounds) that the pseudo-FES map was built from.
    """
    genpept = importlib.import_module("GENPEPT")
    dirs, _notes = genpept.expand_pdb_dirs([Path(seed_dir)])
    return list(genpept.collect_pdbs(dirs))


# --- per-conformer geometry --------------------------------------------------

def _ca_coords_angstrom(atoms: list[dict], positions_nm: np.ndarray) -> np.ndarray:
    """CA coordinates in Angstrom, ordered by residue, from a parsed conformer.

    ``positions_nm`` is what ``seeding._read_pdb_conformer_atoms`` returns (nm);
    GENPEPT's CA-based features expect Angstrom, hence the factor of 10.
    """
    from .seeding import _canonical_atom_name

    positions_nm = np.asarray(positions_nm, dtype=np.float64)
    ca = [
        (int(a.get("residue_ordinal", -1)), int(a.get("index", -1)))
        for a in (atoms or [])
        if _canonical_atom_name(str(a.get("name", ""))) == "CA"
    ]
    ca.sort()
    rows = [positions_nm[idx] for _res, idx in ca if 0 <= idx < len(positions_nm)]
    if not rows:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64) * 10.0


def project_conformer_cv2(positions_nm, atoms: list[dict], topology, result) -> float:
    """Torsion-PCA CV2 for a single conformer, using the saved projection.

    Mirrors the runtime/bootstrap path: map the state's topology-space backbone
    torsions into the conformer's atom order, build the sin/cos feature vector,
    and project with the linear model.  Returns NaN when the conformer's backbone
    cannot be fully mapped.
    """
    from .seeding import _topology_to_conformer_atom_index, map_topology_torsions_to_conformer
    from .tica import backbone_dihedral_features, project_tica1

    atom_map = _topology_to_conformer_atom_index(topology, atoms)
    phi = map_topology_torsions_to_conformer(result.phi_torsion_indices, topology, atom_map)
    psi = map_topology_torsions_to_conformer(result.psi_torsion_indices, topology, atom_map)
    if len(phi) != len(result.phi_torsion_indices) or len(psi) != len(result.psi_torsion_indices):
        return float("nan")
    feat = backbone_dihedral_features(np.asarray(positions_nm, dtype=np.float64), phi, psi)
    if feat.shape[0] != len(result.weights):
        return float("nan")
    return float(project_tica1(feat, result))


# --- pseudo-FES + CV2 assembly ----------------------------------------------

# Gas constant in kJ/mol/K, matching GENPEPT's pseudo-FES energy scale.
_R_KJ_PER_MOL_K = 0.00831446261815324


@dataclass(frozen=True)
class OverlayData:
    """Per-conformer pseudo-FES coordinates + CV2, plus the ΔE background grid."""

    pc1: np.ndarray
    pc2: np.ndarray
    cv2: np.ndarray
    grid_dE: np.ndarray
    pc1_centers: np.ndarray
    pc2_centers: np.ndarray
    pdb_paths: list
    n_used: int


def overlay_from_pdbs(
    pdb_paths,
    topology,
    result,
    *,
    bins: int = 50,
    feature_mode: str = "mixed",
    contact_cutoff: float = 8.0,
    contact_min_sep: int = 3,
    temperature: float = 300.0,
    min_count: int = 1,
) -> OverlayData:
    """Reproduce the GENPEPT pseudo-FES and attach per-conformer CV2.

    Each PDB is parsed once: its CA geometry feeds GENPEPT's contacts/distance
    PCA (the pseudo-FES axes) and its backbone feeds the torsion-PCA CV2, so both
    layers come from identical coordinates.  Conformers with a differing CA count
    or a non-finite feature vector are skipped.
    """
    from .seeding import _read_pdb_conformer_atoms

    genpept = importlib.import_module("GENPEPT")

    feats: list = []
    cv2_vals: list = []
    used_paths: list = []
    n_ca_expected: Optional[int] = None

    for pdb in pdb_paths:
        try:
            positions_nm, atoms = _read_pdb_conformer_atoms(Path(pdb))
        except Exception:
            continue
        if positions_nm.size == 0:
            continue
        ca = _ca_coords_angstrom(atoms, positions_nm)
        if ca.shape[0] < 2:
            continue
        if n_ca_expected is None:
            n_ca_expected = ca.shape[0]
        if ca.shape[0] != n_ca_expected:
            continue
        feat = genpept.make_features(ca, feature_mode, contact_cutoff, contact_min_sep)
        if not np.isfinite(feat).all():
            continue
        feats.append(feat)
        cv2_vals.append(project_conformer_cv2(positions_nm, atoms, topology, result))
        used_paths.append(Path(pdb))

    if len(feats) < 3:
        raise ValueError(
            f"pseudo-FES needs at least 3 usable conformers, got {len(feats)}"
        )

    X = np.vstack(feats)
    scores, _components, _explained, _mu, _sd = genpept.pca_svd(X, 2)
    pc1 = np.asarray(scores[:, 0], dtype=np.float64)
    pc2 = np.asarray(scores[:, 1], dtype=np.float64)
    rt = _R_KJ_PER_MOL_K * float(temperature)
    _hc, _hw, _p, grid_dE, xcenters, ycenters = genpept.deltaE_2d(
        pc1, pc2, bins=bins, RT=rt, min_count=min_count
    )
    return OverlayData(
        pc1=pc1,
        pc2=pc2,
        cv2=np.asarray(cv2_vals, dtype=np.float64),
        grid_dE=np.asarray(grid_dE, dtype=np.float64),
        pc1_centers=np.asarray(xcenters, dtype=np.float64),
        pc2_centers=np.asarray(ycenters, dtype=np.float64),
        pdb_paths=used_paths,
        n_used=len(feats),
    )


def plot_overlay(data, field: Cv2Field, out_png, labels: Optional[dict] = None, r2_min: float = 0.3) -> dict:
    """Render the pseudo-FES with CV2 drawn over it.

    Scatter colored by the true per-conformer CV2 is the primary layer.  Fitted
    iso-CV2 contours and a CV2-gradient arrow are drawn only when the regression
    is full-rank and ``field.r2 >= r2_min``; the R² is always annotated so the
    approximation quality is visible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = labels or {}
    out_png = Path(out_png)
    xc = np.asarray(data.pc1_centers, dtype=np.float64)
    yc = np.asarray(data.pc2_centers, dtype=np.float64)
    dE = np.asarray(data.grid_dE, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    if xc.size and yc.size and np.isfinite(dE).any():
        extent = [xc[0], xc[-1], yc[0], yc[-1]]
        ax.imshow(
            np.ma.masked_invalid(dE.T),
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="bicubic",
            cmap="Greys",
            alpha=0.85,
        )

    sc = ax.scatter(
        data.pc1, data.pc2, c=data.cv2, cmap="viridis", s=18,
        edgecolors="k", linewidths=0.2, zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(labels.get("cv2", "epoch-0 CV2 (torsion-PC1)"))

    contours_drawn = False
    if field is not None and field.usable and np.isfinite(field.r2) and field.r2 >= r2_min and xc.size and yc.size:
        XX, YY = np.meshgrid(xc, yc, indexing="ij")
        Z = field.evaluate(XX, YY)
        try:
            cs = ax.contour(XX, YY, Z, levels=8, colors="k", linewidths=0.8, alpha=0.7, zorder=4)
            ax.clabel(cs, cs.levels[::2], inline=True, fontsize=7, fmt="%.2f")
            contours_drawn = True
        except Exception:
            contours_drawn = False
        # CV2 gradient direction (linear part), drawn from the point cloud centre
        if field.order >= 1 and field.coef.size >= 3:
            gx, gy = float(field.coef[1]), float(field.coef[2])
            norm = float(np.hypot(gx, gy))
            if norm > 0:
                span = 0.25 * float(np.nanmax(xc) - np.nanmin(xc) + 1e-9)
                cx, cy = float(np.mean(data.pc1)), float(np.mean(data.pc2))
                ax.annotate(
                    "", xy=(cx + span * gx / norm, cy + span * gy / norm), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=1.6), zorder=5,
                )

    fit_note = (
        f"CV2≈f(PC1,PC2) fit: R²={field.r2:.2f}"
        + ("" if contours_drawn else "  (too weak for iso-lines — scatter only)")
        if field is not None else ""
    )
    ax.set_xlabel(labels.get("pc1", "pseudo-FES PCA1 (imaginary CV1)"))
    ax.set_ylabel(labels.get("pc2", "pseudo-FES PCA2 (imaginary CV2)"))
    ax.set_title(labels.get("title", "GENPEPT pseudo-FES with epoch-0 CV2 overlay"))
    ax.text(
        0.01, 0.99, fit_note, transform=ax.transAxes, va="top", ha="left",
        fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85),
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return {"contours_drawn": contours_drawn, "r2": float(field.r2) if field is not None else float("nan"), "png": str(out_png)}

