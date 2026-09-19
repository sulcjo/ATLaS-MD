# `pmf_analysis/` — the production analysis products of chignolin_7

Diagnostic subset of
`RUNS/chignolin_7/adaptive_production/pmf_analysis/` (571 MB → 5.6 MB). These are the files the
pipeline itself produced; the reports in `../reports/` are the interpretation of them.

**Excluded, and why:** `rg_samples_with_weights.csv` (470 MB — a 2,000-row head sample is kept, see
below), `native_fold_audit_results/` (74 MB — the decision-relevant parts are in `../data/`), and 40
of the 55 PNGs (21 MB; 15 diagnostic ones kept in `figures/`).

## Start here

| file | what it settles |
|---|---|
| `pmf_summary.json` | master summary. `mbar` (converged, max_delta 9.7e-11, base_ess 833,075, n_k per state), `boost` (**mean 2.08 kcal/mol, boost_reweight_ess 51.2 of 4.58M, anharmonicity 0.799**), `ladder_crosscheck` (λ=0 vs full MBAR, **0.289 kcal/mol, pass**), and the `warnings` list |
| `pmf_summary.md` | the same, readable |
| `pmf_all_methods.csv` | **the estimator spread**: CV1 minimum 0.487 / 0.745 / 0.166 / 0.101 across umbrella_only / gamd_exponential / cumulant2 / cumulant3 |
| `rg_pmf_all_methods.csv` | same for Rg: 6.594 / 5.823 / 7.365 / 7.751 Å |
| `window_diagnostics.csv` | per-window samples (24k–95k), `cv_std` 0.032–0.040, self-bias, `cv_space_overlap_marginal` 0.82–0.996 |
| `overlap_matrix.csv` | **CV1-marginal** histogram overlap, 64×64 (note the header comment: this is *not* MBAR state overlap) |
| `pmf_ladder_crosscheck.csv` | the λ=0-only PMF against the full MBAR one |

## The defects, and the file that shows each

| defect | evidence here |
|---|---|
| **D3** — PCA FES built on 2.02 % of samples | `pca_scores.npz` holds 5,124,128 slots of which only 103,664 are finite; `pca_scores_metadata.json` records `n_fit_frames: 52048`, `source: cache` |
| **D4** — trajectory observables forward-filled | `rg_samples_with_weights.HEAD2000.csv`: `cv_A` changes at every 250-step row while `rg_A` repeats in blocks of ~10. Nominal support 5,110,208; real support 611,536 |
| **D5** — weights file unjoinable to coordinates | same file: columns are `step, replica, window, cv_A, rg_A, umbrella_mbar_weight, gamd_exponential_weight` — **no segment/source column**, and steps restart at 1,010,400 every segment, so `(step, replica)` is not a key |
| **D6** — dimensionless CV1 labelled as Å | `pmf_summary.json`: `cv_min_A`, `cv_max_A`, `pmf_minimum_cv_A`, axis label "CV distance (A)" |
| **estimator validity** | `pmf_summary.json` → `boost.boost_reweight_ess` = 51.2, `anharmonicity_score` = 0.799, and the warning *"λ ladder: boost is inside u_nk, MBAR is exact; no cumulant"* |

## The free-energy surfaces

`chignolin_fes_*` — the d1/d2 surface, i.e. **Satoh 2006 Fig 5B**: axes d(Asp3N–Thr8O) ×
d(Asp3N–Gly7O), 30×30 grid. `chignolin_fes_selected.csv` is the `umbrella_only` one. Its minimum is
at 7.206 / 8.377 Å with **both H-bonds broken**; the paper's native ("N") and misfolded ("M") wells
are **both absent**. Native cell sits at +4.09 kcal/mol on 112 frames.

`distance_rg_2d_fes_*` — CV1 × Rg. `pca_2d_fes_*` — the PCA surface, built on the 2 % cache (D3), so
its minima are **not counted** in the audit. `poincare_*` — fold/unfold crossing analysis.
Each family has `_selected` (umbrella_only) and `_cumulant2/3` variants plus `.npz` grids.

## Convergence subdirectories

`convergence/` (CV1 PMF vs time, basin tracking — one basin, prominence 1.69 kcal/mol),
`rg_convergence/` (converged: final JS 1.2e-7, tail RMSE 0.031 kcal/mol over 10 checkpoints),
`epoch_convergence/`, `epoch_000_separate/`, `epoch_cv_exploration/`.

⚠️ Time-convergence of a PMF does **not** mean the sampling is sufficient. The Rg PMF converges
cleanly while the campaign contains ~1 folding event; the estimator is converging on an
under-sampled ensemble. See `../README.md` §3.

## figures/

`pmf_all_methods` and `rg_pmf_all_methods` (estimator disagreement), `gamd_reweight_quality` and
`gamd_cumulant_quality` (why only `umbrella_only` is valid), `gamd_boost_diagnostics` and
`gamd_dv_distribution_per_window` (the weak boost), `overlap_matrix`, `window_sample_counts`,
`pmf_ladder_crosscheck`, `chignolin_fes` (compare against Satoh Fig 5B), `pca_2d_fes_selected` (the
2 % surface), `distance_rg_2d_fes_selected`, `rg_vs_cv_sampled`, `poincare_map`.
