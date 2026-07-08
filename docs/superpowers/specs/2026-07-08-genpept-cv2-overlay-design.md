# GENPEPT pseudo-FES CV2 overlay

Date: 2026-07-08
Status: approved (design + approach)

## Goal

Add an `analyze_gareus_mbar.py` output that reproduces the GENPEPT **pseudo-FES**
(contacts/distance PCA of the seed conformers, `ΔE = -RT·ln P`) and overlays the
**zeroth-epoch secondary CV (CV2 = torsion-PCA)** on it, so the user can see what
CV2 "was" over the initial landscape before any MD.

Because CV2 (a backbone-torsion PCA projection) is **not** a function of the
pseudo-FES axes (a CA contacts/distance PCA), CV2 is drawn as a *fitted
approximation*: regress per-seed CV2 onto (PCA1, PCA2), then draw iso-CV2 contours
and a CV2-gradient arrow, with per-seed scatter colored by the true CV2 as ground
truth. The figure reports the regression R² so the approximation quality is visible.

## Approved approach

**Recompute the pseudo-FES in-process** by importing GENPEPT's primitives
(`make_features`, `pca_svd`, `deltaE_2d`, `read_ca_coords_pdb`, `collect_pdbs`,
`expand_pdb_dirs`, `RUN_ROOT_PDB_DIR_NAMES`). GENPEPT.py is import-safe (`__main__`
guard at line 8742). If persisted genpept artifacts (`pca_points.csv`,
`deltaE_grid.csv`) exist, reuse them for an exact match; otherwise rebuild from the
seed PDBs. Unweighted `deltaE_2d` by default (no energy CSV assumed).

## Data flow

1. **Locate the genpept run dir.** Read `seed_conformers_dir` from the run's
   `run_args.json` (`analyze` already reads `run_args.json` at 259/591/882/4188).
   Re-root relative/stale paths using the existing logic in `seeding.py:425`.
2. **Build (or load) the pseudo-FES.** Gather PDBs from the standard genpept
   subdirs (`RUN_ROOT_PDB_DIR_NAMES`), `make_features(mode="mixed")` per structure,
   `pca_svd(X, 2)` → per-structure `(pc1, pc2)`, `deltaE_2d(...)` → ΔE grid +
   centers. (Reuse `pca_points.csv`/`deltaE_grid.csv` when present.)
3. **Compute per-structure CV2.** Load the torsion-PCA state
   `<epoch>/tica/bootstrap_torsion_cv.json` (`TICAResult.load`); for each structure
   `tica.backbone_dihedral_features(...)` → `tica.project_tica1(X, result)` → CV2.
   Same machinery as `production._ensure_bootstrap_torsion_cv_ready`.
4. **Regress + render.** Fit CV2 ≈ f(pc1, pc2) (linear + optional quadratic),
   evaluate over the ΔE grid → iso-CV2 contours; scatter seeds colored by CV2;
   draw the CV2 gradient arrow; annotate R². Save PNG + a CSV of
   `(pc1, pc2, cv2)` per structure and the fitted iso-CV2 grid.
5. **Gate + degrade gracefully.** Behind a CLI flag
   (`--genpept-cv2-overlay`, default off). Skip with a recorded warning when the
   seed dir, seed PDBs, or torsion state file are missing. Handle the
   adaptive-union load path dropping `meta['secondary_cv']` by pulling the
   projection from per-epoch `umbrella_pymbar_metadata.json` or the state file.

## Module boundaries

New module `gareus/genpept_cv2_overlay.py` (keeps the 9.5k-line analyze script
lean; coding-style favors small focused files):

- `locate_genpept_seed_dir(run_root, run_args) -> Path | None` — pure path logic.
- `build_pseudo_fes(seed_dir) -> PseudoFES` — PDBs → PCA scores + ΔE grid
  (reuse-or-recompute). `PseudoFES` is a small immutable dataclass
  (`pc1`, `pc2`, `pdb_paths`, `grid_dE`, `pc1_centers`, `pc2_centers`, `components`).
- `seed_cv2_values(pdb_paths, tica_result) -> np.ndarray` — per-structure CV2.
- `fit_cv2_field(pc1, pc2, cv2, order=1) -> Cv2Field` — regression + R²; returns a
  callable evaluated on the grid. Pure, unit-testable.
- `plot_pseudo_fes_cv2_overlay(pseudo, cv2, field, out_png, labels) -> dict` — the
  matplotlib render (imshow ΔE + contour iso-CV2 + scatter + arrow).

`analyze_gareus_mbar.py` gets one thin driver `analyze_genpept_cv2_overlay(d, args,
out, warnings)` that wires the module in and is a no-op (warning) when gated off or
data is missing.

## Error handling

- Missing seed dir / PDBs / torsion state → skip, append a human-readable warning to
  the analysis `warnings` list, return `{'available': False, 'reason': ...}`.
- `< 3` valid structures or degenerate PCA → skip with reason.
- Regression rank-deficient → fall back to linear order 1; if still degenerate,
  scatter-only (no contours) and note it on the figure.

## Testing (TDD)

Data layer is pure and unit-tested; plotting is smoke-tested.

- `fit_cv2_field`: recovers a known linear CV2 = a·pc1 + b·pc2 + c with R²≈1;
  reports low R² for noise; handles rank-deficient input.
- `seed_cv2_values`: with a synthetic `TICAResult` (known weights/mean/offset,
  identity torsion indices) and mock dihedral features, matches `project_tica1`.
- `locate_genpept_seed_dir`: absolute, relative, and re-rooted paths; missing → None.
- `build_pseudo_fes`: reuse path (temp `pca_points.csv`/`deltaE_grid.csv`) returns
  those values; recompute path over ≥3 tiny mock PDBs yields finite PCA scores.
- `plot_pseudo_fes_cv2_overlay`: writes a non-empty PNG (Agg backend).
- Driver: returns `available=False` + warning when gated off / data missing.

Real-data smoke check: run against `RUNS/chignolin/chignolin_2d_run3` (seed dir
`chignolin_genpept_r3`, torsion state under `adaptive_feedback_round_03/tica/`) and
confirm a PNG is produced.

## Out of scope

- Reproducing genpept's Boltzmann/energy weighting (unweighted ΔE by default).
- Persisting the overlay back into the genpept run dir.
- The exact-but-slow subprocess-regeneration path (rejected in favor of recompute).
