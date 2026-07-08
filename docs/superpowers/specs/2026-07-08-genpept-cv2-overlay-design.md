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
2. **Enumerate the same population genpept plots.** `expand_pdb_dirs([seed_dir])`
   + `collect_pdbs` (genpept's own helpers) → the `RUN_ROOT_PDB_DIR_NAMES` set
   (`basin_hop_minima`, `final_implicit_survivor_seeds`, `candidate_seeds`
   fallback, adaptive rounds). This is the population behind the genpept
   `pca_pseudo_fes` map — NOT `final_survivor_seeds.csv` (the smaller GAREUS
   seeding list). When persisted `pca_points.csv`/`deltaE_grid.csv` exist, reuse
   them (they carry `pc1/pc2` + `pdb_path` for this same broad population).
3. **One parse per PDB feeds both layers.** `seeding._read_pdb_conformer_atoms`
   → `positions_nm` + atom records (name, residue_ordinal). From the same parse:
   CA coords (Å = nm·10) → `GENPEPT.make_features(mode="mixed")`; backbone atoms →
   CV2. Guarantees the two layers come from identical geometry.
4. **Pseudo-FES background.** Stack CA features, `pca_svd(X, 2)` → per-PDB
   `(pc1, pc2)`, `deltaE_2d(...)` → ΔE grid + centers.
5. **Per-PDB CV2.** OpenMM topology via `PDBFile(top_path).topology`
   (`_find_data_topology_path` finds it). Load torsion-PCA state
   `tica/bootstrap_torsion_cv.json` (`TICAResult.load`; torsions stored in it).
   Per PDB: `_topology_to_conformer_atom_index(top, atoms)` →
   `map_topology_torsions_to_conformer(result.phi/psi_torsion_indices, top, map)`
   → `backbone_dihedral_features(pos_nm, ...)` → `project_tica1(X, result)`. Same
   projection the runtime force uses.
6. **Regress + render (scatter-primary).** Primary layer: scatter PDBs at
   `(pc1, pc2)` colored by CV2 over the ΔE background — the honest ground truth.
   Fit CV2 ≈ f(pc1, pc2) (linear; quadratic optional); draw iso-CV2 contours +
   gradient arrow ONLY when R² ≥ threshold (default 0.3), else scatter-only.
   Always annotate R² on the figure. Save PNG + CSV of `(pc1, pc2, cv2)` per PDB.
7. **Gate + degrade gracefully.** On by default (`--genpept-cv2-overlay` /
   `--no-genpept-cv2-overlay`). "Not applicable to this run" skips (no genpept
   seed dir, or CV2 is not torsion-pca) are silent — only surprising failures
   (missing topology PDB, load/build errors) append to the warnings list. The
   expensive PDB parse runs only once the seed dir + state + topology are all
   present, so non-applicable runs skip cheaply. Handle the adaptive-union load
   path dropping `meta['secondary_cv']` by pulling the projection from per-epoch
   `umbrella_pymbar_metadata.json` or the state file.

## Module boundaries

New module `gareus/genpept_cv2_overlay.py` (keeps the 9.5k-line analyze script
lean; coding-style favors small focused files):

- `locate_genpept_seed_dir(run_root, run_args) -> Path | None` — pure path logic.
- `collect_seed_pdbs(seed_dir) -> list[Path]` — thin wrapper over genpept's
  `expand_pdb_dirs` + `collect_pdbs` (broad `pca_pseudo_fes` population).
- `overlay_from_pdbs(pdb_paths, topology, tica_result) -> OverlayData` — the
  single-parse loop: per PDB, CA→features + backbone→CV2. Builds ΔE grid via
  `pca_svd` + `deltaE_2d`. `OverlayData` is a small immutable dataclass
  (`pc1`, `pc2`, `cv2`, `grid_dE`, `pc1_centers`, `pc2_centers`).
- `fit_cv2_field(pc1, pc2, cv2, order=1) -> Cv2Field` — regression + R²; callable
  evaluated on the grid. Pure, unit-testable (the genuinely new logic).
- `plot_overlay(data, field, out_png, labels, r2_min=0.3) -> dict` — matplotlib
  render: imshow ΔE + scatter colored by CV2 (primary) + iso-CV2 contours/arrow
  only when `field.r2 >= r2_min`; R² annotated always.

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
  reports low R² for noise; handles rank-deficient / collinear input (no crash,
  low R², `usable=False`).
- `overlay_from_pdbs`: over ≥3 tiny mock PDBs + a synthetic OpenMM-like topology,
  yields finite `pc1/pc2`, a finite ΔE grid, and CV2 matching a direct
  `project_tica1` on hand-built features.
- `locate_genpept_seed_dir`: absolute, relative, and re-rooted paths; missing → None.
- `collect_seed_pdbs`: over a temp tree with `basin_hop_minima/` etc., returns the
  expected PDBs (deduped).
- `plot_overlay`: writes a non-empty PNG (Agg backend); draws contours when
  `r2 >= r2_min`, scatter-only below it.
- Driver: returns `available=False` + warning when gated off / data missing.

Real-data smoke check (verify the FIGURE, not just file-exists): run against the
original-checkout `RUNS/chignolin/chignolin_2d_run3` (seed dir `chignolin_genpept_r3`,
torsion state under `adaptive_feedback_round_03/tica/`); confirm the ΔE background
resembles the genpept `pca_pseudo_fes` map and report the CV2 regression R².

## Out of scope

- Reproducing genpept's Boltzmann/energy weighting (unweighted ΔE by default).
- Persisting the overlay back into the genpept run dir.
- The exact-but-slow subprocess-regeneration path (rejected in favor of recompute).
