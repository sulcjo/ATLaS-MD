# GaREUS PMF analysis summary

Input: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production`
Samples/windows: **5124128 / 64**
Temperature: **300.00 K**
CV range: **0.021 - 0.986 A**
Selected unbiased PMF: **umbrella_only**
PMF minimum: **0.4872603653309246 A**
PMF span: **4.202 kcal/mol**

## Result health

**Overall: 🔴 FAIL**

selected PMF: **umbrella_only** · samples/windows: **5.12M / 64** · PMF span: **4.20 kcal/mol** · base ESS: **16.3% (833.1k)**

| Check | Status | Detail |
|---|---|---|
| MBAR convergence | ✓ pass | converged (max_delta 9.7e-11) |
| Window sampling | ✓ pass | all 64 windows populated |
| Sampling (base ESS) | ✓ pass | 16.3% of 5.12M samples (833.1k eff.) |
| Window overlap | ✓ pass | worst 0.818 CV1-marginal (windows 3-0, CV-space nearest neighbours); worst 0.491 CV1-marginal index-adjacent (windows 19-20 -- may not be neighbours in CV space); target ≥0.30 |
| Overlap connectivity | ✓ pass | joint (CV1,CV2) not measured (this run has no biased second axis: no window carries a finite positive secondary restraint constant, and/or no sample has a finite secondary CV); CV1-marginal connected (1 component at ≥0.300) |
| Sample-to-state mapping | ✓ pass | worst own-restraint self-bias 4.8 kT median (state 2), p90 8.4 kT; ~1 kT expected for a 2-DOF harmonic (a pass does not prove the mapping is right -- a mis-mapping between two nearby windows stays inside this band) |
| GaMD reweighting | · n/a | umbrella-only PMF (no GaMD boost applied) |
| PMF convergence | ✓ pass | converged by JS/RMSE tail test (tail JS 0.000196, tail RMSE 0.04 kcal/mol) |
| λ-ladder cross-check | ✓ pass | agrees within 0.289 kcal/mol (tolerance 0.500, 30 bins compared) |
| Overlap along λ | ✗ FAIL | worst 0.074 (states 5-6) |
| Overlap across CV1 | ⚠ caution | worst 0.103 (states 50-54) |
| Connectivity along λ | ✗ FAIL | split into 54 components (expected 16); weakest link 0.074 |
| Connectivity across CV1 | ⚠ caution | split into 50 components (expected 4); weakest link 0.103 |
| Trajectory frame coverage | ✓ pass | 836.9% of reachable pairs assigned (5118016/611536); steps_per_frame=2500 from run config (prod_dir or parents) |

### Warnings by severity (3 MEDIUM, 2 INFO)

- **MEDIUM** — [epoch_000 report] Unbiased estimator: λ ladder: boost is inside u_nk, MBAR is exact; no cumulant.
- **MEDIUM** — [epoch_000 report] GaMD cumulant reweighting factor has a bin-to-bin scatter of 0.54 kT, so adjacent bins' probabilities carry up to e^0.54=1.7x of estimator noise; a single excursion can place the PMF minimum on a fluctuation and re-reference the curve. Check whether the excursions are statistically significant (compare each bin's variance against its neighbours' using its own sample count) before trusting or smoothing them: --gamd-smooth-sigma exists but will erase real structure if the variation is real.
- **MEDIUM** — Unbiased estimator: λ ladder: boost is inside u_nk, MBAR is exact; no cumulant.
- **INFO** — [epoch_000 report] GaMD exponential reweighting ESS is very low: 18.4/540192
- **INFO** — GaMD exponential reweighting ESS is very low: 14.7/4583936

### Key diagnostics

- **Basins (CV1):** 1: [0.04–0.97 A]
- **Poincaré fold routes:** 2 routes: 'left-α' (CV2≈+0.83) and 'left-α' (CV2≈+0.84) (median recurrence 0.0 ns)
- **Poincaré unfold routes:** 1 route: 'right-α' (CV2≈+0.12) (median recurrence 0.1 ns)

## MBAR / umbrella diagnostics

Converged: **True** after 69 iterations
Backend: **numba-anderson** / threads: **24**
Base ESS: **833075.2** / 5124128

## GaMD boost diagnostics

Boost mean/std: **2.085 / 1.303 kcal/mol**
Boost range: **0.000 - 13.012 kcal/mol**
Anharmonicity score: **0.7991513940121527**
Boost exponential ESS fraction: **0.000**

### Per-rung boost/reweighting diagnostics

| λ | n | ⟨ΔV⟩ kcal/mol | σ_ΔV kcal/mol | ⟨ΔV⟩ kT | anharmonicity | skew |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 519898 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.232 | 1650509 | 2.097 | 0.943 | 3.518 | 0.042 | 0.626 |
| 0.643 | 1650508 | 2.432 | 1.276 | 4.079 | 0.072 | 0.878 |
| 1.000 | 1303213 | 2.338 | 1.336 | 3.921 | 0.085 | 0.924 |

## Radius of gyration diagnostics

Selected Rg PMF method: **umbrella_only**
Mean ± std Rg: **6.984 ± 1.028 Å**
Rg PMF minimum: **6.594365058495252 Å**
Rg PMF span: **4.545 kcal/mol**

## Distance vs Rg 2D FES

Selected 2D FES method: **umbrella_only**
2D FES minimum: **distance 0.7123382644417384 Å, Rg 6.016019825574859 Å**
2D FES span: **11.008 kcal/mol**
Grid: **30 × 30** bins
Normalization: **Minimum finite free energy shifted to 0 kcal/mol**

## PCA1 vs PCA2 2D FES

Selected PCA 2D FES method: **umbrella_only**
PCA 2D FES minimum: **PCA1 1.439445575078329 Å, PCA2 -5.289497375488281 Å**
PCA 2D FES span: **9.005 kcal/mol**
Grid: **30 × 30** bins
Explained variance: **PC1 0.352, PC2 0.123**
Selection: **protein and name CA** (10 atoms)
Normalization: **Minimum finite free energy shifted to 0 kcal/mol**

## Extra trajectory observable PMFs

Extra observable PMFs unavailable: disabled by --extra-pmf-from-trajectories never

## Outputs

- `pmf_unbiased_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pmf_unbiased.csv`
- `pmf_all_methods_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pmf_all_methods.csv`
- `pmf_umbrella_only_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pmf_umbrella_only.csv`
- `pmf_gamd_exponential_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pmf_gamd_exponential.csv`
- `pmf_gamd_cumulant2_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pmf_gamd_cumulant2.csv`
- `pmf_gamd_cumulant3_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pmf_gamd_cumulant3.csv`
- `overlap_matrix_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/overlap_matrix.csv`
- `window_diagnostics_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/window_diagnostics.csv`
- `summary_md`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pmf_summary.md`
- `summary_json`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pmf_summary.json`
- `epoch_000_pmf_unbiased_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/epoch_000_separate/pmf_unbiased.csv`
- `epoch_000_pmf_all_methods_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/epoch_000_separate/pmf_all_methods.csv`
- `epoch_000_pmf_umbrella_only_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/epoch_000_separate/pmf_umbrella_only.csv`
- `epoch_000_pmf_gamd_exponential_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/epoch_000_separate/pmf_gamd_exponential.csv`
- `epoch_000_pmf_gamd_cumulant2_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/epoch_000_separate/pmf_gamd_cumulant2.csv`
- `epoch_000_pmf_gamd_cumulant3_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/epoch_000_separate/pmf_gamd_cumulant3.csv`
- `epoch_000_overlap_matrix_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/epoch_000_separate/overlap_matrix.csv`
- `epoch_000_window_diagnostics_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/epoch_000_separate/window_diagnostics.csv`
- `rg_pmf_unbiased_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/rg_pmf_unbiased.csv`
- `rg_pmf_all_methods_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/rg_pmf_all_methods.csv`
- `rg_samples_with_weights_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/rg_samples_with_weights.csv`
- `rg_pmf_plot_png`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/rg_pmf_all_methods.png`
- `rg_vs_cv_sampled_png`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/rg_vs_cv_sampled.png`
- `convergence_dir`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/convergence`
- `rg_pmf_convergence_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/rg_convergence/rg_pmf_convergence.csv`
- `rg_pmf_summary_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/rg_convergence/rg_pmf_summary.csv`
- `rg_pmf_convergence_report_md`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/rg_convergence/rg_pmf_convergence_report.md`
- `rg_convergence_plot_index`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/rg_convergence/plot_index.txt`
- `rg_basin_definitions_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/rg_convergence/rg_basin_definitions.csv`
- `rg_basin_populations_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/rg_convergence/rg_basin_populations.csv`
- `rg_basin_populations_png`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/rg_convergence/rg_basin_populations.png`
- `rg_basin_map_png`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/rg_convergence/rg_basin_map.png`
- `distance_rg_2d_fes_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/distance_rg_2d_fes_selected.csv`
- `distance_rg_2d_fes_npz`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/distance_rg_2d_fes_selected.npz`
- `distance_rg_2d_fes_png`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/distance_rg_2d_fes_selected.png`
- `distance_rg_2d_fes_png_0_2`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/distance_rg_2d_fes_selected_0_2.png`
- `distance_rg_2d_fes_png_0_5`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/distance_rg_2d_fes_selected_0_5.png`
- `distance_rg_2d_fes_png_0_10`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/distance_rg_2d_fes_selected_0_10.png`
- `distance_rg_2d_fes_png_0_20`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/distance_rg_2d_fes_selected_0_20.png`
- `distance_rg_2d_fes_png_0_all`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/distance_rg_2d_fes_selected_0_all.png`
- `distance_rg_2d_fes_png_main`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/distance_rg_2d_fes_selected.png`
- `distance_rg_2d_fes_cumulant2_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/distance_rg_2d_fes_cumulant2.csv`
- `distance_rg_2d_fes_cumulant2_npz`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/distance_rg_2d_fes_cumulant2.npz`
- `distance_rg_2d_fes_cumulant2_png`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/distance_rg_2d_fes_cumulant2.png`
- `distance_rg_2d_fes_cumulant2_png_0_2`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/distance_rg_2d_fes_cumulant2_0_2.png`
- `distance_rg_2d_fes_cumulant2_png_0_5`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/distance_rg_2d_fes_cumulant2_0_5.png`
- `distance_rg_2d_fes_cumulant2_png_0_10`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/distance_rg_2d_fes_cumulant2_0_10.png`
- `distance_rg_2d_fes_cumulant2_png_0_20`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/distance_rg_2d_fes_cumulant2_0_20.png`
- `distance_rg_2d_fes_cumulant2_png_0_all`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/distance_rg_2d_fes_cumulant2_0_all.png`
- `distance_rg_2d_fes_cumulant2_png_main`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/distance_rg_2d_fes_cumulant2.png`
- `pca_2d_fes_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pca_2d_fes_selected.csv`
- `pca_2d_fes_npz`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pca_2d_fes_selected.npz`
- `pca_2d_fes_png`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pca_2d_fes_selected.png`
- `pca_2d_fes_png_0_2`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pca_2d_fes_selected_0_2.png`
- `pca_2d_fes_png_0_5`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pca_2d_fes_selected_0_5.png`
- `pca_2d_fes_png_0_10`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pca_2d_fes_selected_0_10.png`
- `pca_2d_fes_png_0_20`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pca_2d_fes_selected_0_20.png`
- `pca_2d_fes_png_0_all`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pca_2d_fes_selected_0_all.png`
- `pca_2d_fes_png_main`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pca_2d_fes_selected.png`
- `pca_2d_fes_cumulant2_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pca_2d_fes_cumulant2.csv`
- `pca_2d_fes_cumulant2_npz`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pca_2d_fes_cumulant2.npz`
- `pca_2d_fes_cumulant2_png`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pca_2d_fes_cumulant2.png`
- `pca_2d_fes_cumulant2_png_0_2`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pca_2d_fes_cumulant2_0_2.png`
- `pca_2d_fes_cumulant2_png_0_5`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pca_2d_fes_cumulant2_0_5.png`
- `pca_2d_fes_cumulant2_png_0_10`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pca_2d_fes_cumulant2_0_10.png`
- `pca_2d_fes_cumulant2_png_0_20`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pca_2d_fes_cumulant2_0_20.png`
- `pca_2d_fes_cumulant2_png_0_all`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pca_2d_fes_cumulant2_0_all.png`
- `pca_2d_fes_cumulant2_png_main`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pca_2d_fes_cumulant2.png`
- `pca_2d_fes_summary_json`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pca_2d_fes_summary.json`
- `pca_scores_npz`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pca_scores.npz`
- `chignolin_fes_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_selected.csv`
- `chignolin_fes_npz`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_selected.npz`
- `chignolin_fes_png`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes.png`
- `chignolin_fes_png_0_5`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_0_5.png`
- `chignolin_fes_png_0_10`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_0_10.png`
- `chignolin_fes_png_0_20`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_0_20.png`
- `chignolin_fes_png_0_40`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_0_40.png`
- `chignolin_fes_png_0_all`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_0_all.png`
- `chignolin_fes_png_main`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes.png`
- `chignolin_fes_cumulant2_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_cumulant2.csv`
- `chignolin_fes_cumulant2_npz`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_cumulant2.npz`
- `chignolin_fes_cumulant2_png_0_5`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_cumulant2_0_5.png`
- `chignolin_fes_cumulant2_png_0_10`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_cumulant2_0_10.png`
- `chignolin_fes_cumulant2_png_0_20`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_cumulant2_0_20.png`
- `chignolin_fes_cumulant2_png_0_40`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_cumulant2_0_40.png`
- `chignolin_fes_cumulant2_png_0_all`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_cumulant2_0_all.png`
- `chignolin_fes_cumulant2_png_main`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_cumulant2.png`
- `chignolin_fes_cumulant3_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_cumulant3.csv`
- `chignolin_fes_cumulant3_npz`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_cumulant3.npz`
- `chignolin_fes_cumulant3_png_0_5`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_cumulant3_0_5.png`
- `chignolin_fes_cumulant3_png_0_10`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_cumulant3_0_10.png`
- `chignolin_fes_cumulant3_png_0_20`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_cumulant3_0_20.png`
- `chignolin_fes_cumulant3_png_0_40`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_cumulant3_0_40.png`
- `chignolin_fes_cumulant3_png_0_all`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_cumulant3_0_all.png`
- `chignolin_fes_cumulant3_png_main`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/chignolin_fes_cumulant3.png`
- `poincare_fold_crossings_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/poincare_fold_crossings.csv`
- `poincare_unfold_crossings_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/poincare_unfold_crossings.csv`
- `poincare_map_png`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/poincare_map.png`
- `poincare_map_summary_json`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/poincare_map_summary.json`
- `poincare_residue_torsions_png`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/poincare_residue_torsions.png`
- `poincare_residue_torsions_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/poincare_residue_torsions.csv`
- `epoch_cv_grid_png`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/epoch_cv_exploration/epoch_cv_grid.png`
- `epoch_cv_cumulative_png`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/epoch_cv_exploration/epoch_cv_cumulative.png`
- `total_pmf_convergence_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/convergence/total_pmf_convergence.csv`
- `total_pmf_summary_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/convergence/total_pmf_summary.csv`
- `total_pmf_convergence_report_md`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/convergence/total_pmf_convergence_report.md`
- `convergence_plot_index`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/convergence/plot_index.txt`
- `total_basin_definitions_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/convergence/total_basin_definitions.csv`
- `total_basin_populations_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/convergence/total_basin_populations.csv`
- `total_basin_populations_png`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/convergence/total_basin_populations.png`
- `total_basin_map_png`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/convergence/total_basin_map.png`
- `pmf_ladder_crosscheck_csv`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pmf_ladder_crosscheck.csv`
- `pmf_ladder_crosscheck_png`: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/pmf_analysis/pmf_ladder_crosscheck.png`

## Warnings

- [epoch_000 report] Unbiased estimator: λ ladder: boost is inside u_nk, MBAR is exact; no cumulant.
- [epoch_000 report] GaMD cumulant reweighting factor has a bin-to-bin scatter of 0.54 kT, so adjacent bins' probabilities carry up to e^0.54=1.7x of estimator noise; a single excursion can place the PMF minimum on a fluctuation and re-reference the curve. Check whether the excursions are statistically significant (compare each bin's variance against its neighbours' using its own sample count) before trusting or smoothing them: --gamd-smooth-sigma exists but will erase real structure if the variation is real.
- [epoch_000 report] GaMD exponential reweighting ESS is very low: 18.4/540192
- Unbiased estimator: λ ladder: boost is inside u_nk, MBAR is exact; no cumulant.
- GaMD exponential reweighting ESS is very low: 14.7/4583936
