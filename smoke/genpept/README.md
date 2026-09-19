# GENPEPT r7 Seed Bank — CV Census Report

**Date:** 2026-09-08
**Dataset:** `RUNS/chignolin_genpept_r7/final_implicit_survivor_seeds/` — **1,970 conformers**, chignolin (`GYDPETGTWG`, 138 atoms, GBSA implicit-minimized survivors; generation config: 200k requested, `diversity_bank_preset=chignolin`, `contact_bias_strength=0.15`, seed 12345).
**Scope:** 298 collective variables per seed (extended pass: full residue-balanced grid, rational-switch contacts, strand-packing descriptors, plus a dedicated residual-torsion-PCA CV2 deep dive on bank and chignolin_6 epoch-0 dynamics); per-CV spread / differentiating-power / redundancy statistics. Bank-side geometry only — dynamics-side checks live in `CV2_DEEPDIVE.md`.
**Differentiating-power metrics:** normalized Shannon entropy `H/ln(20)` (20 equal bins over [min,max]), effective occupied states `exp(H)`, bimodality coefficient BC (flag > 5/9 ≈ 0.555), resolvable 1.5σ window count over the p01–p99 span at k = 1200 kcal/mol/CV², 300 K (`σ_w = √(kT/k)`).

---

## 1. Headline findings

1. **Best CV1 candidates:** backbone-heavy (N,CA,C,O — GAREUS's exact `backbone-heavy` selection, no OXT) and CA-only logistic contact fractions at **r₀ = 10 Å, |i−j| ≥ 4**, any β ≥ 1.5 Å⁻¹ — entropy **0.973–0.980**, IQR 0.36–0.40, **25–27 resolvable windows** at k = 1200. The parameterization picked in the earlier swarm-side test (all-heavy, r₀ = 12 Å, β 1.5–3) ranks immediately below (entropy 0.954–0.958, 23–25 windows) and is statistically co-champion.
2. **The legacy definition (r₀ = 4.5 Å, β = 6) is degenerate on this library** for every atom selection: median ≤ 0.017, ≤ 2 resolvable windows — it only resolves tightly packed near-native conformers. This independently reproduces, from the seed side, the r7 coverage collapse measured swarm-side (coverage 0→0.069).
3. **r₀ placement dominates; β saturates.** Once r₀ is in the 8–12 Å transition band, raising β above ~3 Å⁻¹ changes entropy by < 0.02. r₀ = 14 Å drives the median toward saturation (≥ 0.78) and entropy falls again.
4. **Residue-balanced ≈ plain atom-pairs** (|r| > 0.99 for every matched pair in the bb/ca/heavy/sca selections; the `sch` selection drops to 0.83–0.89 at r₀ ≤ 10 Å, but only on near-degenerate low-spread CVs) — pick one; they are the same CV. For backbone-heavy they are **identical to float roundoff** (max |Δ| = 1.4×10⁻⁷; uniform 4×4 = 16 atom-pairs per residue cell → uniform weights).
5. **Rg / mean-Cα-distance are near-duplicates of mid-r₀ contact fractions** (|r| ≈ 0.94–0.98): pairing Rg as CV1 with a contact CV1 adds nothing.
6. **Backbone H-bond switch counts (N–O ≤ 3.5 Å) are degenerate here**: median 0, p95 ~ 1 over the whole library (implicit-minimized structures rarely satisfy the geometry). Not usable as CVs.
7. **Torsion-PCA CV2:** flat spectrum (PC1 = 12.2 % variance, PC2 = 9.2 %), broad unimodal projections (entropy 0.91–0.94, BC ≈ 0.44). Residualization against the contact CV1 affects only the top 1–2 PCs (r(PC3, resPC3) = 1.000).
8. **Orthogonality for a 2D grid:** residual torsion PC1 is the most decorrelated CV2 candidate vs every contact CV1 (|r| ≤ 0.13); plain torsion PC1/PC2 carry |r| ≈ 0.5/0.03–0.07; `hairpin_closure_ratio` is the most orthogonal geometric CV (r ≈ 0.03–0.24 vs contact CV1s). Angstrom-unit `nw` values are only comparable under an equal-k-per-unit assumption.
9. **Rational switches do not beat logistic.** PLUMED-style `(1−(r/r₀)ⁿ)/(1−(r/r₀)ᵐ)` (n6m10/n6m12/n8m14, r₀ 8/10 Å, heavy/bb/ca) top out at entropy 0.965 vs 0.977 for the logistic at the same selections — keep the logistic.
10. **Strand-interface contact fraction is a disguised duplicate.** Cross-half heavy contact fractions (r₀ 12, β 3) reach entropy 0.958 — but correlate |r| = 0.998 with the all-heavy global fraction (a hairpin's nonlocal contacts ARE the strand pairing). `pack_y2` (heavy neighbors ≤ 5 Å of Tyr2 sidechain) is mildly independent (|r| ≤ 0.35) but bimodally distributed (BC 0.62) and low-spread (ent 0.61). `strand_mindist` is steric-degenerate (std 0.002 Å).
11. **Residualization rotates within one subspace.** Plain vs residualized PCA (vs `c_bb_s4_r010_b3`): PC1 direction changes (|cos| = 0.166) but the top-5 subspace is the same (principal-angle overlap 0.986). The residual-PC1 axis is physically the PETGT turn / strand ψ axis (see `CV2_DEEPDIVE.md` for the loading list).

## 1b. Dynamics-side findings (chignolin_6 epoch-0, 32 traces × 4,768 frames)

12. **Bank-fit CV2 axes are not diffusive.** Hess cosine content of the projected bank plain/residual PC1: median C₁ ≈ 0.07, **0/32 traces ≥ 0.5** (half-trace 10/64 ≥ 0.5). Trace-own PC1 reaches only 0.388 median itself. A seed-bank torsion PCA is a *structure-diversity* coordinate, not a slow-motion coordinate — a slow CV2 must be fit on dynamics (within-segment tICA, per `RUNS/chignolin_rebuild_design/REBUILD.md` decision 2).
13. **Bank axes do not transfer across banks.** Per-trace corr(bank resPC1 projection, chignolin_6's stored bootstrap `secondary_cv`): median −0.540, range [−0.817, +0.075] (mixed sign = not a global axis flip). Each seed bank's eigenvector must be persisted with its own run (`bootstrap_torsion_cv.json`), never re-derived from a moved bank.
14. **2D grid feasibility is fine.** 16×8 quantile-binned joint maps: bank occupancy 94–97% (CV1 = bb r₀10 or heavy r₀12 × bank resPC1); chignolin_6 epoch-0 ensemble 96.9% vs bank resPC1 projection, 100% vs its own stored CV2; per-CV1-bin CV2 IQR stays large (bank 1.35, ensemble map vs stored CV2 1.44) — CV2 windows retain live density across the CV1 range.

## 1c. Advanced checks (DIFFMAP.md, FOURIER.md, TURN_AXIS.md)

15. **Diffusion maps confirm CV1, demote subjective CV2.** On both a z-scored torsion metric and a CA-dRMSD metric, the first nontrivial diffusion coordinate LASSO-identifies as the compaction axis (census CV1 family, R² = 0.985 / torsion PC1, R² = 0.974); the residual-torsion CV2 appears only at the ψ₃ level. The mathematical objective and the hand-picked winner agree on CV1.
16. **TwoNN intrinsic dimension**: d ≈ 3.6 (torsion space), 4.6 (CA-dRMSD) — the bank manifold is ~4-D. A single 2-D grid is a cover, not a parameterization; expect curvature to pressure window count on at least one axis.
17. **Fourier harmonics reveal the hidden signal**: interior ψ torsions are strongly bimodal — R₁ 0.13–0.34 with R₂ 0.58–0.73 — a two-lobe β(+140°)/curl(−10°) structure invisible in linear means. Winding analysis along ordered coordinates finds **no** additional significant periodic pumps (permutation null, N=300). Circular correlations: intra-residue φ–ψ coupling dominates (ρ ≈ −0.35); between residues ≤ 0.19.
18. **The residual CV2 is a strand-flank curl transition.** At matched CV1 (0.433 vs 0.474 across deciles), ψ of Y2, P4, E5, T6, T8, W9 rotates from β-extended (~+91..+149°) to curled (−7..−16°), −63..−163° per residue, while D3/G7 anchor the turn (D3 ψ stable −40°). It tightens the turn (ρ = −0.54 vs `turn3537_mean_ca`) without changing compaction (ρ ≈ 0 vs strand contact fraction) — i.e., exactly the motion a residualized CV2 should expose.

## 2. Verification anchors

| Check | Result |
|---|---|
| vs `HEAVY_CONTACTS_PARAM_TEST_CHIGNOLIN_GENPEPT.txt` (heavy, r≥4, r₀ 12 Å, β 1.5) | mean .5435 vs .5435 · std .1969 vs .1969 · p05 .1789 vs .1789 · p95 .8246 vs .8246 — **exact** |
| vs GAREUS's own CV code (`gareus.cv.nonlocal_contact_cv_from_positions_nm`, heavy s4 r₀12 β3, seed 0) | census 0.6368295550 vs gareus 0.6368295404 — **Δ = 1.5×10⁻⁸** (PDB coordinate precision) |
| per-seed parity vs GAREUS's `build_nonlocal_contact_pairs` on all 1,970 seeds × 6 spot configs (heavy/bb/ca/sch/sca) | max \|Δ\| ≤ **1.8×10⁻⁷** everywhere (checked after dropping a wrongly-included OXT from the `bb` selection; fix verified) |

## 3. Top 15 CVs by normalized entropy

| # | CV | family | median | std | IQR | entropy | BC | nw @k1200 |
|---|---|---|---|---|---|---|---|---|
| 1 | cresb_bb_s4_r010_b6 ≡ c_bb_s4_r010_b6 | contact (resbal) | 0.482 | 0.245 | 0.393 | 0.980 | 0.503 | 27 |
| 3 | c_bb_s4_r010_b3 ≡ cresb_ | contact (resbal) | 0.481 | 0.241 | 0.385 | 0.979 | 0.502 | 26 |
| 5 | cresb_ca_s4_r010_b6 | contact resbal | 0.454 | 0.247 | 0.396 | 0.977 | 0.486 | 27 |
| 6 | c_ca_s4_r010_b6 | contact | 0.454 | 0.247 | 0.396 | 0.977 | 0.486 | 27 |
| 7 | c_ca_s4_r010_b3 | contact | 0.452 | 0.242 | 0.378 | 0.976 | 0.486 | 27 |
| 8 | cresb_ca_s4_r010_b3 | contact resbal | 0.452 | 0.242 | 0.378 | 0.976 | 0.486 | 27 |
| 9 | c_bb_s4_r010_b1.5 ≡ cresb_ | contact (resbal) | 0.480 | 0.229 | 0.364 | 0.975 | 0.500 | 25 |
| 11 | cresb_ca_s4_r010_b1.5 | contact resbal | 0.452 | 0.228 | 0.359 | 0.973 | 0.485 | 26 |
| 12 | c_ca_s4_r010_b1.5 | contact | 0.452 | 0.228 | 0.359 | 0.973 | 0.485 | 26 |
| 13 | crat_ca_s4_n8m14_r010 | contact rational | 0.528 | 0.194 | 0.293 | 0.965 | 0.487 | 22 |
| 14 | crat_ca_s4_n6m12_r010 | contact rational | 0.481 | 0.181 | 0.273 | 0.963 | 0.477 | 21 |
| 15 | c_bb_s3_r010_b3 | contact | 0.587 | 0.198 | 0.311 | 0.962 | 0.486 | 22 |

(`cresb_bb_*` ≡ `c_bb_*` pairs are identical to float roundoff, max |Δ| 1.4×10⁻⁷ — finding 4.)

Best per atom selection (s4): heavy → `r012_b6` ent 0.958 · bb → `r010_b6` ent 0.978 · sch → `r012_b6` ent 0.958 · sca → `r012_b6` ent 0.943 · ca → `r010_b6` ent 0.977.
Legacy comparison (s4, r₀ 4.5, β 6): heavy med 0.006 · sca med 0.002 · bb med 0.003 · sch med 0.00003 · ca med 0.000005 — all ≤ 2 resolvable windows.

## 4. CV1 × CV2 orthogonality (Pearson r / Spearman r)

| CV1 candidate ↓ | torpca_pc1 | restorpca_pc1 | torpca_pc2 | shape_anisotropy | hairpin_closure | contact_order | d_y2_w9_ca | burial_w9 |
|---|---|---|---|---|---|---|---|---|
| c_bb_s4_r010_b3 | +0.52/+0.50 | +0.11/+0.11 | −0.03/−0.04 | −0.75/−0.75 | +0.33/+0.27 | +0.92/+0.93 | −0.85/−0.85 | +0.23/+0.22 |
| c_ca_s4_r010_b3 | +0.52/+0.50 | +0.10/+0.10 | −0.03/−0.05 | −0.76/−0.76 | +0.29/+0.24 | +0.92/+0.93 | −0.85/−0.85 | +0.27/+0.26 |
| c_heavy_s4_r012_b3 | +0.48/+0.43 | −0.00/−0.02 | −0.07/−0.07 | −0.88/−0.87 | +0.10/+0.03 | +0.82/+0.83 | −0.94/−0.94 | +0.46/+0.46 |
| c_sch_s4_r012_b3 | +0.28/+0.26 | −0.13/−0.12 | −0.05/−0.07 | −0.71/−0.71 | −0.22/−0.28 | +0.53/+0.53 | −0.75/−0.75 | +0.67/+0.64 |
| rg_heavy | −0.54/−0.48 | −0.05/−0.03 | +0.08/+0.07 | +0.88/+0.84 | −0.09/−0.00 | −0.76/−0.82 | +0.94/+0.91 | −0.43/−0.47 |
| sasa_total | −0.62/−0.60 | −0.20/−0.20 | +0.04/+0.04 | +0.61/+0.56 | −0.07/+0.01 | −0.76/−0.78 | +0.76/+0.73 | −0.53/−0.52 |

Redundancy: `contact_order_ca8` is highly correlated with every contact CV1 (r ≈ 0.82–0.93) — treat it as a contact-CV readout, not an independent axis. `rg_heavy` ↔ `c_heavy_s4_r012_b*` at r ≈ −0.94..−0.98.

## 5. CV catalog

| Family | Columns | Examples |
|---|---|---|
| Logistic contact fractions (atom-pairs) | 175 (5 selections × seps 2/3/4 × 14 (r₀,β) sets; sep 3 gets the 7 core sets, seps 2/4 get all 14) | `c_bb_s4_r010_b3`, `c_heavy_s4_r012_b3` |
| Residue-balanced contact fractions | 50 (5 selections × sep 4 × 10 (r₀,β) sets) | `cresb_bb_s4_r010_b3` |
| Rational-switch contacts | 18 (3 selections × sep 4 × 6 (r₀; n,m) sets, PLUMED rational) | `crat_ca_s4_n8m14_r010` |
| Strand packing | 10 (2 cross-half pair sets × 3 (r₀,β) + mindist + 3 sidechain packings) | `strand_cross_far_r012_b3`, `pack_y2` |
| Global geometry/shape | 18 | `rg_heavy`, `asphericity`, `shape_anisotropy`, `e2e_nc_ca`, `hairpin_closure_ratio`, `contact_order_ca8` |
| H-bond switches | 4 | `hbond_bb_ge3`, `hbond_i_i3`, `hbond_i_i4`, `hbond_crossstrand` |
| Torsion bins / χ1 | 9 | `rama_alpha/beta/ppii/left`, `cis_omega_any`, `chi1_y2/w9_sin/cos` |
| Torsion PCA | 5 | `torpca_pc1..pc5` |
| Residual torsion PCA | 3 | `restorpca_pc1..pc3` (residualized on `c_heavy_s4_r012_b3`) |
| SASA / burial | 5 | `sasa_total`, `sasa_hydrophobic_frac`, `burial_y2/w9/d3` |
| Reference | 1 | `rmsd_ca_to_survivor000` |

Contact CV semantics identical to `gareus.cv.nonlocal_contact_cv_from_positions_nm`: `0.5·(1−tanh(β·(r−r₀)/2))`, β in Å⁻¹, normalized by the number of terms (atom-pairs) or residue-pair-balanced sum. Rational form: `(1−(r/r₀)ⁿ)/(1−(r/r₀)ᵐ)` (PLUMED convention). tICA-ordered eigenvectors and the residual axis for CV2 reuse live in `bank_torsion_pca_cv2.json` (interleaved sin/cos ordering of `gareus.tica.backbone_dihedral_features`).

## 6. Caveats

- Statistics describe the **seed bank's own diversity** — ensemble coverage after solvation/swarm sampling is a separate measurement (the swarm-side numbers in `HEAVY_CONTACTS_PARAM_TEST_CHIGNOLIN_GENPEPT.txt`).
- `nw` for Å-unit CVs (distances, SASA, RMSD) assumes the same k ceiling per unit; comparable to contact CVs only under equal-k-per-unit conventions.
- Torsion PCA is fit **on this bank** (like the `cv2: torsion-pca` bootstrap); projections are bank-specific coordinates, not transferable constants.
- Entropy depends on the 20-bin convention; it is a ranking metric, not an absolute constant.

## 7. Files

| Path | Content |
|---|---|
| `report_tables.txt` | Exhaustive tables: full contact grid (all selections × seps × r₀ × β), all families, top-30, complete redundancy list, full orthogonality matrix, conclusions |
| `CV2_DEEPDIVE.md` | Residual torsion PCA deep dive: axis loadings, subspace overlap, Hess diffusion content on 32 chignolin_6 epoch-0 traces, axis-transfer check, 16×8 joint-grid occupancies |
| `DIFFMAP.md` / `genpept_r7_diffmap.{json,psi.csv}` | Diffusion-map spectra/robustness/IDs, TwoNN intrinsic dimension, LASSO eigenvector identification |
| `FOURIER.md` / `genpept_r7_fourier.json` | Circular Fourier harmonics, winding/permutation-null signal scan, Fisher-Lee circular correlations |
| `TURN_AXIS.md` / `figures/cv2_turn_axis.png` | Physical elaboration of the residual CV2 turn axis (decile Ramachandran, per-residue rotations, descriptor correlations) |
| `ORTHOGONAL_PAIRS.md` / `genpept_r7_orthogonal_pairs.{csv,json}` | 60 alternative CV1×CV2 pairs scored on independence, occupancy, conditional spread (incl. negative controls) |
| `RESIDUAL_THERM.md` / `genpept_r7_residual_therm.json` | GBn2 decomposition, DSSP, and literature context of the residual motion |
| `DYNAMICS_VALIDATION.md` / `genpept_r7_dynamics_cv1.json` | CV1 candidates on solvated MD (coverage transfer, decorrelation) |
| `STEERABILITY_STABILITY.md` / `genpept_r7_steer_stability.json` | gradient steerability + axis stability across resamples/generators |
| `MI_STATES_ENERGY.md` / `genpept_r7_mi_states_energy.json` / `genpept_r7_energy_components.csv` | mutual information, 6-state curl ladder, full-bank GBn2 decomposition |
| `RIGOR_HANDOFF.md` / `genpept_r7_rigor.json` / `seed_bank_tiling_k16.csv` | bootstrap intervals, morph realism, provenance, contact-bias audit, omega-cis, tiling |
| `PLAIN_PC2_CV2.md` / `genpept_r7_plain_pc2.json` | designated CV2 scalar verification battery (loadings, deciles, C1, KDE modes) |
| `GENPEPT_R7_CV_CAMPAIGN_REPORT.html` | single-file compiled visual report (all figures embedded) |
| `REP3_ENERGY_TERMS.md` / `genpept_rep3_energy_terms.json` | per-term FES maps, trend slopes, compensation, near-native signature |
| `SOLVATION_AXIS.md` / `genpept_rep3_solvation_axis.json` | solvation as an axis: orthogonality + folded-state relation incl. native reference position |
| `REP3_STATS.md`, `REP3_ADVANCED.md`, `genpept_rep3_cv_census_values.csv`, `genpept_rep3_cv_census_stats.csv`, `genpept_rep3_orthogonal_pairs.csv`, `genpept_rep3_energy_components.csv`, `genpept_rep3_advanced.json` | FULL campaign battery rerun on the 10k bank (rep3_massive) |
| `bank_torsion_pca_cv2.json` | Bank-fit torsion PCA axes (plain pc1/pc2, residual-vs-`c_bb_s4_r010_b3` pc1 with slope + unit axis), tICA-canonical interleaved feature order with residue labels |
| `genpept_r7_cv_census_values.csv` | Per-seed CV values, 1,970 × 298, no NaN |
| `genpept_r7_cv_census_stats.csv` | Per-CV statistics (all metrics above), 298 rows |
| `genpept_r7_cv_census_corr_pearson.csv` | Full 298 × 298 Pearson matrix |
| `figures/genpept_r7_cv_histograms.png` | 28 representative CV distributions |
| `figures/genpept_r7_cv_corr.png` | Clustered \|r\| heatmap of representative CVs |
| `figures/genpept_r7_cv_2dmaps.png` | CV1 × CV2 scatters (contact CV1 vs res-torsion PC1 / torsion PC1; Rg vs closure ratio) |
| `figures/cv2_loadings.png` | Residual-PC1 loadings (top 18) + explained variance spectrum |
| `figures/cv2_diffusion_content.png` | Hess C₁ box plots: bank axes vs trace-own PC1 over 32 epoch-0 traces |
| `figures/cv2_joint_maps.png` | CV1 × CV2 16×8 joint maps on bank and chignolin_6 epoch-0, occupancy labeled |
| `figures/r7v2_contact_beta_pseudofes.png` | β-scan pseudo-FES: heavy r₀12 vs backbone-heavy r₀10 vs CA r₀10 (previous PC-test layout) |
| `figures/r7v2_combo_map_{bb,ca}_beta3.png` | sep × midpoint combo maps, grey pseudo-FES base, for the new selections |
| `figures/r7v2_midpoint_scan_{bb,ca}_r4.png` | r₀ 6–12 Å × β 1.5–10 midpoint scans (r≥4) |
| `figures/r7v2_pseudofes_cv_{bb,ca}_beta3_r0_10_r4.png` | pseudo-FES colored by the winning CV1 candidates |
| `figures/r7v2_pseudofes_residual_torsion_pca_cv2.png` | pseudo-FES colored by residual torsion-PCA CV2 (component=5, residualized on bb r₀10) |
| `figures/r7v2_residual_torsion_cv2_geometric_morph.{mp4,pdb,json}` + `_preview.png` | aligned geometric morph along residual-PC1 CV2 at constant CV1 (seeds 73 → 1259; not MD) |
| `figures/index.html` | one-page gallery of all figures |
| `scripts/genpept_r7_cv_census.py` | CV computation (writes values CSV) |
| `scripts/genpept_r7_cv_analyze.py` | Statistics, tables, figures |
| `scripts/genpept_r7_cv2_deepdive.py` | CV2 deep dive (needs `REPO_ROOT` env or default repo path; writes CV2_DEEPDIVE.md, bank_torsion_pca_cv2.json, 3 figures) |
| `scripts/genpept_r7v2_figures.py` | PC-test figure family regenerated on the new CV winners (8 PNGs) |
| `scripts/genpept_r7v2_morph.py` | geometric morph renderer along the residual CV2 axis (mp4 + pdb + json; needs ffmpeg) |
| `scripts/genpept_r7_diffmap.py` | diffusion maps + TwoNN + LASSO identification (writes DIFFMAP.md, psi CSV/JSON, 2 figs) |
| `scripts/genpept_r7_fourier.py` | circular harmonics, winding/null scan, circular correlations (writes FOURIER.md, 3 figs) |
| `scripts/genpept_r7_turn_axis.py` | turn-axis physical elaboration (writes TURN_AXIS.md, 1 fig) |
| `scripts/genpept_r7_pairs.py` | orthogonal-pair scoring (writes ORTHOGONAL_PAIRS.md, CSV/JSON, 1 fig) |
| `scripts/genpept_r7_residual_thermo.py` | GBn2/DSSP/context analysis (writes RESIDUAL_THERM.md, 1 fig, JSON) |
| `scripts/build_cv_report.py` | rebuilds the compiled HTML report from all artifacts |
| `scripts/genpept_r7_dynamics_cv.py` | dynamics-side CV1 validation from replica XTCs |
| `scripts/genpept_r7_steer_stability.py` | steerability gradients + axis stability |
| `scripts/genpept_r7_mi_states_energy.py` | MI + KDE states + full-bank decomposition |
| `scripts/genpept_r7_rigor.py` | bootstrap, geodesic, provenance, bias audit, tiling |
| `scripts/genpept_r7_plain_pc2.py` | plain-PC2 verification battery |

## 8. Reproduction

```bash
# from repo root (mdtraj, scipy, matplotlib, openmm + gareus on sys.path)
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_r7_cv_census.py   # ~17 s, reads RUNS/chignolin_genpept_r7/
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_r7_cv_analyze.py  # ~6 s; both write back into GENPEPT_R7_CV_CENSUS_REPORT/
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_r7_cv2_deepdive.py  # ~5 s; also reads RUNS/chignolin_6/adaptive_production/epoch_000/tica_obs/
```

All scripts are deterministic (no RNG).
