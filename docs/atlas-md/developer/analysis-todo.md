# TODO — analysis (`gareus-analyze`) follow-ups (opened 2026-09-23)

Context: thermodynamic energy decomposition (branch `feat/thermo-energy-decomposition`,
spec `docs/superpowers/specs/2026-09-23-thermo-energy-decomposition/spec.md`, §11 for the frame-based
phase 2). Phase 1 (ΔG/ΔH/−TΔS, V_pep/U_ee split) and phase 2 (exact V_pp/V_pe from solute frames,
backbone MIE-2 entropy, solvent entropy by difference) are implemented and validated on chignolin_8_US.

## E1 — higher-order configurational entropy (not started)

Why: on chignolin_8_US (low→high contacts) TΔS_conf flips sign from MIE-1 (+3.1 kJ/mol) to MIE-2
(−10.1 ± 3.2 kJ/mol), so expansion truncation matters, and −TΔS_solv (−11.3 ± 7.1) inherits it.

- [ ] **MIST** (King & Tidor 2012): S = Σ H_i − Σ_{(i,j)∈T} I_ij with T the maximum spanning tree of the
      pairwise-MI matrix. `thermo_entropy.mie_entropy` already builds every H_ij: keep the I matrix, take
      `scipy.sparse.csgraph.minimum_spanning_tree(-I)`, return `S_MIST` next to S1/S2. Report
      `TdS_conf_MIST_kj`; base `minus_TdS_solv` on MIST instead of S2. Bootstrap unchanged (replicates
      already reweight by block multinomial counts).
- [ ] **Bias-corrected MI**: subtract a null I_ij from block-shuffled torsion j (keeps marginals and
      autocorrelation, removes the i–j correlation); report raw and corrected. Current raw histogram bias
      ≈ (b_i − 1)(b_j − 1)/(2·ESS) per pair, ≈ 3 nats summed on chignolin_8_US; it only cancels between
      basins while their ESS is similar.
- [ ] **Side-chain χ1/χ2 torsions** (coverage, independent of order): currently only backbone φ/ψ, so
      side-chain entropy is lumped into the "solvent" remainder.
- [ ] **Convergence gate**: report S1 → S2 → S_MIST per basin difference, plus the difference at ½ and ¼
      of the ESS (subsampled blocks); if it still moves by more than ~1 SE, mark TΔS_conf inconclusive.
- [ ] **kNN cross-check** (Kozachenko–Leonenko / Hnizdo) of the full-dimensional torsion entropy on the
      torus (`cKDTree(boxsize=2π)`), weights via resampling ∝ w to the ESS; report on the basin
      difference only, as a separate field.
- [ ] (diagnostic only) third-order MIE at coarse bins (8 bins → 512 cells/triplet; C(18,3) = 816 triplets
      for chignolin); 24 bins is out of reach at ESS ≈ 13k.
- Tests needed: analytic Gaussian/von Mises cases with known joint entropy; a chain-correlated synthetic
  case where MIST is exact and MIE-2 overcounts.

## E2 — pre-existing issues found while building the decomposition (not fixed)

- [ ] `_epoch_source` indexes the phases that **contributed** samples, but `adaptive_epoch_run_dirs`
      lists every discovered phase. Consumers that index the latter by the former are wrong whenever a
      phase contributes nothing: `gareus/mbar_analysis/data.py` (~L435, ~L457–490) and
      `analyze_gareus_mbar.py`'s merged-trajectory helpers (~L195, ~L249). The union loader now also
      writes `_epoch_source_run_dirs` (contributing phases, in index order); switch those consumers to it.
- [ ] `analyze_gareus_mbar._sample_to_segment_frame` assumes the first frame of a resume segment is at
      R + interval; the real first frame is the next multiple of the interval after R. Off by one frame
      for 3 of chignolin_8's 5 segments ("assigned 480872/481616" = 248 replicas × 3 segments dropped, the
      rest shifted by 0.875 ps). Affects Rg / chignolin-FES / PCA frame matching. `thermo_frames` keys
      frames by round(time/dt) instead — reuse that.
- [ ] `tests/test_package_smoke.py::test_tiny_lambda_ladder_run_completes_end_to_end_slow` fails:
      `pyarrow.dataset` over `samples/seg_001/` picks up the production writer's `parquet_manifest.json`
      ("Parquet magic bytes not found"). Test should read only `*.parquet`.

## E3 — electrostatic solvation / solvent-resolved terms (needs output change)

- [ ] Linear-response ΔG_solv,el ≈ ½Δ⟨U_pe,el⟩, GIST/2PT, per-residue pe all need solvent coordinates;
      chignolin_9 writes `traj_solute_only: true`. Decide whether future campaigns save sparse
      full-system frames (≈ 1 per ns), then add a full-frame evaluator (charge-scaling split already
      validated in `tests/test_thermo_frames_openmm.py`).
- [ ] ΔCp needs several temperatures (separate campaign decision).
