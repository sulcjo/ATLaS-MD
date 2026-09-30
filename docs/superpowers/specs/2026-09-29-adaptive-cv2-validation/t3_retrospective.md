# T3 retrospective: adaptive-CV2 machinery replayed on chignolin_9 and chignolin_7

Spec: `docs/superpowers/specs/2026-09-29-adaptive-cv2-resolution-design.md`, Section 5 (T3).
Run on 2026-09-29 at `963742e` (branch `val/t3-retrospective`). This was a read-only replay of
recorded samples. No MD was run and no state was created.

## 0. Summary

1. **Every edge the new metric calls weak on chignolin_9 is an artefact of the geometry-edge
   builder, not a gap.** `build_geometry_edges` builds `primary_chain` by `np.argsort` over
   `primary_center` alone. Ties are broken arbitrarily, so the chain links states that are not
   neighbours: CV2-only windows at the CV1 placeholder, and 2D windows in the same CV1 column
   that sit two CV2 rows apart. Under the CV1 marginal these edges read about 0.8-0.95 and were
   never weak. Under pairwise-mbar they read 0.07-0.14, so they become weak and get bridged.
   - Across 7 c9 payloads, pairwise-mbar flagged 1-5 weak edges per phase. All were such pairs:
     64-76 (CV2-only, three rows apart); 160-216, 136-232 and 184-204 (same CV1, two rows apart);
     and 164-200 (a diagonal, two rows apart).
   - In each case the true adjacent pairs exist and overlap at >= 0.275 (median 0.33-0.40).
   - The marginal metric flags other artefacts instead: cross-pattern placeholder edges.
   - chignolin_7 (CV1 only) flags 0 weak edges under either metric.
   - Section 5 has the evidence. This is a bug to fix before `pairwise-mbar` becomes the default,
     and before R1 (3.3) uses these edges.
2. **Threshold calibration.** The data show no overlap level at which an edge stops working.
   - At every measured overlap, down to 0.065:
     - the direct two-state Δf agrees with the union's multi-path Δf (median |ΔΔf| 0.04-0.08 kT);
     - realised replica transitions fall smoothly as about O^2.3, from about 115 /ns at 0.45 to
       about 1-1.6 /ns below 0.08, and no pair has zero transitions.
   - The largest error that is actually present is time non-stationarity. First-half vs
     second-half Δf differs by 0.03-0.17 kT (median), 2-3x the iid σ, at every overlap.
   - None of the 105 edges below 0.15 is a cut edge: removing every edge below 0.10, or below
     0.15, still leaves 1 component. So the data show that a direct estimate across an edge at
     O = 0.065-0.15 is accurate. They say nothing about an edge that is a graph's only
     connection, and they do not locate a failure point.
   - Recommendation: keep `min_rung_overlap` = 0.15 as the weak threshold for spatial edges,
     with moderate confidence that it is not too strict. It corresponds to 2.5 sampled-sd spacing
     and gives σ(Δf) ≤ 0.1 kT at n_eff ≥ 470. It should only be applied to true adjacent pairs.
   - Keep 0.25 as the target. Locating the actual failure point needs T2 (synthetic gaps).
3. **Coupling gate.** The data cannot validate 0.25.
   - For CV1-restrained windows the gate never binds. The maximum fraction is 0.0037, and for
     the degree-1 fit the fraction is linear in k2, so it would take k2 of about 80 (70x the
     deployed value) to reach 0.25.
   - The 4 CV1-unrestrained (k1 = 0) windows score 0.42 on the mean-shift bound. But:
     - the realised CV1 shift, as a mean over 6 disjoint phases ± SE, is -0.14 ± 0.05,
       -0.04 ± 0.03, +0.05 ± 0.02 and -0.01 ± 0.06 anchor-sd;
     - the gate's linear-response prediction is +0.21, +0.07, -0.09 and -0.26, which is 3.7-6.8
       SE from the realised value, with the opposite sign in 3 of 4 windows;
     - the bound 0.42 sits 5.5-17 SE above the realised |shift|.
   - Lowering k2 to 0.42, as the gate would, raises the adjacent CV2-only overlap from 0.39 to
     0.47, which makes those windows close to redundant, with no benefit seen.
   - Recommendation: do not enforce the k1 = 0 shift branch with the current mechanical model.
     Either drop it, or gate on a realised shift with its own threshold (≥ 0.5 sd would pass
     every c9 window). The data give no basis to change 0.25 for the curvature branch, because
     it is never approached.
4. **Proposer dry-run.**
   - Validation: with the campaign's own recorded policy and caps, the dry-run reproduces the
     recorded actions exactly, 236 `extend` on c9 and 64 on c7 in each of epochs 0-2. c9 used
     `min_samples_for_add` 20000 and `min_samples_for_retire` 30000.
   - That sample floor blocks every bridge in the numbered epochs, on both metrics. So on c9 as
     configured the artefacts would never have reached the registry.
   - With default rules and caps lifted:
     - pairwise-mbar would propose 3-11 bridge centres per epoch on c9, i.e. 12-44 states on
       4 rungs (5-19 % of 236);
     - marginal would propose 3-8 centres.
   - Every proposal on both metrics targets an artefact edge (item 1). With the edge set fixed,
     the expected need is 0.
   - chignolin_7: 0 proposals on both metrics. No `add_rung` is proposed pre-union (by design).
5. **R1 (3.3)** would act on the wrong edges. The weak 2D edges differ only in CV2 (ΔCV1 = 0),
   so R1 would bridge them, and they are the skip-a-row artefacts. The CV2-only 64-76 edge is
   ineligible (k1 = 0). The graph is 1 component, so the structural branch never fires.

## 1. Data and provenance

| Campaign | Phases replayed | Collector | Registry |
|---|---|---|---|
| chignolin_9 (236 states = 59 centres x 4 rungs; λ = 0 layout: 1 anchor, 15 CV1-only, 4 CV2-only (k1 = 0), 39 2D) | `epoch_000` (flat), `epoch_001`, `epoch_002` (segmented, baseline only), `final` (segmented), `final_extension_001`, `final_extension_002`, final-combined | as the driver: `collect_segmented_epoch_diagnostics` / `collect_final_combined_diagnostics`, `edge_metric="pairwise-mbar"` | `state_registry.json` (constant: all 236 states from epoch 0, only `extend` actions were ever applied) |
| chignolin_7 (64 = 16 CV1-only centres x 4 rungs) | `epoch_000`, `epoch_001`, `epoch_002` (baseline + 3 top-ups each), `final` (baseline + 3 top-ups), final-combined | same | same (constant) |

- **Skipped:** c9 `epoch_003` holds only an `epoch_window_map.csv`.
- **`final_extension_002` is partial.** Its last sample write was at 17:10:55 on 2026-09-29 and its
  checkpoint at 17:10:59. The replay started at 19:16, so the phase had been idle for about 2 h.
  It holds 2 segments and 2.6 ns per replica, against 13.8 ns for `final_extension_001`.
- **Union snapshot** (`adaptive_union_mbar.npz`, written 04:08 on 2026-09-29):
  - It contains `final/baseline` (426,446 rows) plus the part of `final_extension_001` that
    existed then (338,940 rows). Rows are g-subsampled per state, 320-4,700 kept per state.
    It does not contain `final_extension_002`.
  - `f_k` comes from `adaptive_union_mbar_analysis.state_free_energies.csv`, the pymbar
    `Delta_f[0, i]` on the same `umbrella_reduced_bias_nk`, in kT.
  - chignolin_7 has no free-energy CSV. Its union (2.95M rows) was re-solved with pymbar at row
    stride 4. 19 % of its rows are non-finite and were dropped, as the builder does.
- **Read-only.** The collectors' `write_json`, `write_text_atomic` and paired-CV NPZ path were
  redirected to `/tmp/t3/out/`. `find RUNS/chignolin_{7,9} -newer <stamp>` returned nothing after
  all 12 replays, and `final_extension_002` was not excluded from that check.
- **Cost.** c9 final-combined took 558 s at 8.15 GB max RSS; the per-phase replays took 66-202 s
  at 0.9-4.5 GB.
- **Stated limits.**
  - c9's CV2 is residual PC1 (the old gain ranking), not the slowness-ranked component 7.
  - Top-ups were off, so there is no per-epoch union solve and no pre-union rung overlap.
  - P4 subsamples hold ≤ 2,000 pairs per state (strides 1-18). n_eff is the subsample's n_eff.

## 2. Task 1: edge metric replay

Edge classes are on the representative rung (λ = 0):

- **CV1 / CV2 / 2D chain:** the collector's geometry edges (`primary_chain`, `nearest_2d`)
  between two states of that restraint pattern.
- **cross-pattern geometry:** geometry edges between two patterns. These are never weak under
  3.1.
- **spanning / radius-neighbour:** the 3.1 appended edges.
- **rung:** has no pre-union value (unmeasured by design).

Each cell reads median [q10, q90], min. "pw" is the two-state point overlap. "marg" is the CV1
marginal (collector value, or recomputed on the subsample for appended edges). "joint" is the P4
30x30 joint histogram. τ is in subsample frames.

### 2.1 chignolin_9, final-combined (the densest data)

| class | n | pw point | CV1 marginal | joint 2D hist | τ | n_eff | weak old / new | q90 < 0.15 |
|---|---:|---|---|---|---|---|---:|---:|
| 2D chain | 34 | 0.332 [0.198, 0.406], 0.139 | 0.923 [0.487, 0.978] | 0.491 [0.271, 0.634] | 0.04 [0, 0.71] | 1843 [833, 1969] | 0 / 1 | 1 |
| CV1 chain | 2 | 0.345, 0.344 | 0.513 | – | 0.02 | 1903 | 0 / 0 | 0 |
| CV2 chain | 2 | 0.144 (0.080, 0.195) | 0.853 | 0.201 | 0.01 | 1919 | 0 / 1 | 1 |
| cross-pattern geometry | 42 | 0.441 [0.210, 0.465], 0.092 | 0.972 [0.283, 0.979] | 0.276 | 0.03 | 1871 | 6 / 0 | 1 |
| radius-neighbour | 198 | 0.202 [0.095, 0.372], 0.070 | 0.434 [0.142, 0.675] | 0.261 [0.120, 0.511] | 0.00 | 1969 | 0 / 0 | 85 |
| spanning | 19 | 0.269 [0.213, 0.385], 0.182 | 0.405 | 0.311 | 0.07 | 1727 | 0 / 0 | 0 |
| rung | 177 | – (pre-union) | – | – | – | – | 0 / 0 | – |

- The q10-q90 bootstrap width of an edge is about 0.003-0.01. The graph is 1 component.
- Measured components: 2, sizes 58 + 1. The singleton is state 60, the CV1 = 0.9345 end
  window. All 5 of its edges are unmeasured because its n_eff is 146-196, below 200. This is not
  a gap.
- Union pairwise rung overlaps (λ 0-0.23 / 0.23-0.64 / 0.64-1): median 0.233 / 0.269 / 0.401.
  This is the gate check against the ladder_adapt replay's 0.243 / 0.272 / 0.402. The first
  value is 0.01 lower. The ladder_adapt replay pooled more final data.

### 2.2 Weak-edge counts per phase

| phase | weak marginal → pairwise | unmeasured | 2D chain pw median (min) | CV2-only chain pw | CV1 chain pw |
|---|---|---:|---|---|---|
| c9 epoch_000 | 7 → 2 | 23 | 0.331 (0.137) | 0.093, 0.212 | 0.357 |
| c9 epoch_001 | 2 → 5 | 7 | 0.337 (0.128) | 0.065, 0.183 | 0.345 |
| c9 epoch_002 | 2 → 1 | 32 | 0.327 (0.155) | 0.080, 0.192 | 0.353 |
| c9 final | 6 → 2 | 0 | 0.334 (0.137) | 0.082, 0.201 | 0.344 |
| c9 final_extension_001 | 5 → 2 | 4 | 0.333 (0.140) | 0.080, 0.192 | 0.350 |
| c9 final_extension_002 (partial) | 3 → 2 | 15 | 0.328 (0.136) | 0.080, 0.190 | 0.348 |
| c9 final-combined | 6 → 2 | 6 | 0.332 (0.139) | 0.080, 0.195 | 0.345 |
| c7 epoch_000 / 001 / 002 / final / combined | 0 → 0 (all) | 4 / 8 / 6 / 4 / 4 | – | – | 0.320 / 0.322 / 0.316 / 0.316 / 0.316 (min 0.286-0.296) |

- Every marginal-weak c9 edge is a cross-pattern placeholder edge. Every marginal trigger was
  CV1 marginal < 0.30; none was exchange acceptance < 0.08.
- Measured components (edges q90 ≥ 0.15) split off singleton end windows with low n_eff: state
  80 in `final_extension_002`, and states 0 and 60 in every c7 phase.

### 2.3 Every edge that is weak under either metric (c9 final-combined; other phases in `/tmp/t3/tables.txt`)

| edge | class / type | pw point (q90) | CV1 marg | joint | weak under | why |
|---|---|---|---|---|---|---|
| 64-76 | CV2-only, `primary_chain` | 0.080 (0.082) | 0.846 | 0.110 | pairwise | z0 -1.85 vs +1.35 (3 rows apart); 64-68 / 68-72 / 72-76 = 0.38 / 0.39 / 0.39 |
| 164-200 | 2D, `primary_chain` | 0.139 (0.142) | 0.498 | 0.183 | pairwise | (0.577, -0.785) vs (0.630, +1.348): diagonal across a CV1 column boundary, 2 CV2 rows apart |
| 36-72 | cross-pattern | 0.196 | 0.283 | – | marginal | CV1-only vs CV2-only at the CV1 placeholder |
| 68-220 | cross-pattern | 0.092 | 0.264 | 0.115 | marginal | CV2-only vs 2D |
| 0-36, 64-232, 68-140, 72-136 | cross-pattern `nearest_2d` | 0.204-0.212 | 0.273-0.285 | 0.27-0.28 | marginal | anchor / CV2-only vs placeholder coordinates |

c9 `epoch_001` also has 160-216, 136-232 and 184-204 at 0.135-0.143: same CV1, z0 +0.281 vs
-1.851, skipping the -0.785 row. 184-204 is at 0.205 on final-combined. These three edges cross
0.15 back and forth between phases.

## 3. Task 2: calibrating the 0.15 threshold on the pairwise scale

### 3.1 What the number means (equal-width Gaussians, iid)

| pairwise overlap O | separation in sampled sd | σ(Δf)·√n (kT) | n for σ(Δf) = 0.1 kT |
|---:|---:|---:|---:|
| 0.40 | 1.0 | 0.72 | 52 |
| 0.31 | 1.5 (the spec's design spacing) | 1.11 | 123 |
| 0.25 | 1.85 | 1.41 | 200 |
| 0.15 | 2.53 | 2.13 | 467 |
| 0.10 | 2.99 | 2.83 | 800 |
| 0.058 | 3.54 (P7a radius 2.5) | 3.84 | 1,470 |

O(d) = ∫ p_a p_b / (p_a + p_b) and var(Δf) = (1/n)(1/O - 2) (BAR, N_a = N_b = n). Source:
`scripts/t3_gaussian_overlap.py`.

### 3.2 References, per λ = 0 spatial edge of the 3.1 graph (c9: 295 edges, c7: 29 edges)

The estimates below were computed on the union snapshot's own samples (`scripts/t3_union_reference.py`):

- (a) two-state BAR re-solved (`evaluate_edge`) vs `pairwise_state_overlap` with the union f_k,
  and vs union Δf;
- (b) accepted replica exchanges between the two states per ns of per-replica time, over
  `final` + `final_extension_001` (c9, 28.4 ns) or `final` (c7). Counts come from
  `gareus.query.load_exchanges`, which reads committed files only; a raw glob double-counts
  18-35 % of rows in some segments;
- (c) iid bootstrap σ(Δf) vs theory, and first-half vs second-half Δf.

Checks on the method:

- Restraint-record u reproduces the union `umbrella_reduced_bias_nk` on λ = 0 columns to
  1.7e-10 (c9) and 1.3e-10 (c7).
- The P4-subsample point overlap tracks the union two-state value: r = 0.979, median
  |diff| 0.005 (c9); r = 0.988 (c7).
- Exceptions are the free-axis spanning edges: 0-4 is 0.087 on the union vs 0.209 on P4, and
  64-216 is 0.089 vs 0.162. The free CV1 distribution of the anchor and CV2-only states drifts
  between phases.

chignolin_9, binned by the two-state union overlap (`scripts/t3_calibration_summary.py`):

| O bin | n | σ(Δf) kT (med) | σ / iid theory (sanity check) | \|2-state − union Δf\| kT (med, q90) | \|Δf half1 − half2\| kT (med, q90) | \|z_split\| > 2 | union-f overlap − 2-state | acceptance (med) | transitions /ns (med, min) |
|---|---:|---:|---:|---|---|---:|---:|---:|---|
| < 0.08 | 8 | 0.069 | 0.99 | 0.078, 0.240 | 0.174, 0.278 | 5 / 8 | 0.000 | 0.38 | 1.6, 0.9 |
| 0.08-0.10 | 23 | 0.056 | 0.99 | 0.036, 0.158 | 0.173, 0.283 | 12 / 23 | 0.000 | 0.41 | 2.7, 1.1 |
| 0.10-0.12 | 38 | 0.053 | 0.95 | 0.048, 0.108 | 0.110, 0.297 | 14 / 38 | 0.000 | 0.43 | 4.8, 2.5 |
| 0.12-0.15 | 36 | 0.047 | 0.95 | 0.048, 0.154 | 0.102, 0.223 | 13 / 36 | 0.000 | 0.46 | 7.1, 4.5 |
| 0.15-0.20 | 18 | 0.039 | 0.93 | 0.031, 0.101 | 0.094, 0.374 | 8 / 18 | 0.000 | 0.50 | 14.0, 6.6 |
| 0.20-0.25 | 18 | 0.031 | 0.97 | 0.021, 0.151 | 0.063, 0.166 | 5 / 18 | 0.000 | 0.55 | 21.0, 8.7 |
| 0.25-0.30 | 41 | 0.024 | 0.98 | 0.023, 0.058 | 0.089, 0.163 | 24 / 41 | 0.000 | 0.61 | 36.2, 25.3 |
| 0.30-0.40 | 75 | 0.019 | 0.95 | 0.015, 0.062 | 0.057, 0.148 | 40 / 75 | 0.000 | 0.70 | 70.4, 38.5 |
| 0.40-0.51 | 38 | 0.010 | 0.97 | 0.011, 0.077 | 0.033, 0.140 | 22 / 38 | 0.000 | 0.83 | 115.5, 52.7 |

- Every one of the 295 edges had exchange attempts (gibbs-walk, radius 1.65).
- Spearman(O, acceptance) = 0.89. Transitions per ns ∝ O^2.27 (log-log fit), with no knee.
- The minimum measured overlap is 0.065 (124-188, a second-neighbour radius edge): 1.1
  transitions/ns, and its Δf agrees with the union to 0.19 kT.
- chignolin_7 shows the same picture (29 edges, O 0.081-0.431):
  - σ / theory is 0.92-1.08;
  - transitions are 5.1 /ns at O = 0.08 and about 100 /ns at 0.3;
  - the half-to-half drift is 0.03-0.71 kT with no dependence on O. Its union is not
    g-subsampled, so the iid σ understates the error and |z_split| > 2 on 24/29 edges.

The σ / theory column compares an iid bootstrap on g-subsampled rows with the iid asymptotic
formula. The two agree by construction, so the column only confirms that the estimator is
implemented correctly. It is not evidence about gaps.

### 3.3 What the data support

- **No failure seen in range, and none could be seen.** Down to O = 0.065 (c9) and 0.081 (c7):
  - the direct two-state Δf agrees with the union's multi-path Δf;
  - the post-union (`ouf`) and pre-union values are identical to 3 decimals;
  - replicas still cross at ≥ 0.9 /ns;
  - no pair has zero transitions.
- **The graph is redundant.** None of the 105 c9 edges below 0.15 is a cut edge (bridge), and
  the graph stays 1 component after removing every edge below 0.10 or below 0.15. So the
  evidence is that a direct estimate across an edge at O ≥ 0.065 is accurate. It is not
  evidence that an edge at that overlap would still hold the free energies together if it
  were the only connection.
- **The error that exists does not come from overlap.** The half-to-half Δf drift is
  0.03-0.17 kT (median) and exceeds 2 σ_iid on about 50 % of c9 edges in every bin. It grows
  about 5x from high to low overlap in absolute kT. That is slow orthogonal dynamics, which
  thresholding overlap does not fix.
- **What cannot be concluded:**
  - where a real gap starts (no gaps in these data);
  - the false-weak and false-strong rates at 0.15 (this needs T2's synthetic gaps);
  - anything about rung edges beyond the union values above (no pre-union rung metric exists).
- **Proposed thresholds.**
  - Keep weak = 0.15 and target = 0.25 on the pairwise scale. Confidence: moderate that 0.15 is
    not too strict, since non-cut edges at 0.065 are measured accurately. Low confidence on where
    it should be, because no sole-connection edge or real gap exists in these data.
  - Justification: 0.15 is a 2.5-sd spacing, 1.7x the spec's 1.5-sd design spacing. It keeps
    σ(Δf) ≤ 0.1 kT at the n_eff ≥ 470 these edges have. No c9/c7 true adjacent pair falls
    below it:
    - true adjacent CV2-only pairs: 0.38-0.39;
    - CV1 chain: 0.286-0.38;
    - adjacent 2D cells: same column, next row, min 0.37 (24 pairs, median 0.40); next column,
      same row, min 0.275 (35 pairs, median 0.33) on final-combined;
    - the only same-pattern edges below 0.15 are the non-adjacent pairs of Section 5 and radius
      edges.
  - Lowering it to 0.10 would also be consistent with the data. It is not recommended before
    T2, because nothing here tests a real gap.
  - Consider an n_eff-aware floor O_min = 1 / (2 + n_eff σ_tol²), which is 0.14 at n_eff 500
    and σ_tol 0.1 kT. It ties the threshold to the precision the edge can deliver.

## 4. Task 3: coupling-gate data (chignolin_9, component PC1, degree 1)

The fit's compiled component has K1 = 0.173, K2 = 0 (degree 1), σ_j = 1.205 and
σ_c = 0.188, which gives dz/dc = -0.763 everywhere. The swarm pooled CV1 sd is 0.195; the
production anchor (state 0, no restraint) CV1 sd is 0.19-0.22 per phase.

### 4.1 CV1-restrained 2D windows (39)

- Coupling fraction at k2 = 1.18: max 0.0037. It is linear in k2 (B = 0 for degree 1), so it
  reaches 0.25 only at k2 ≈ 80 (about 70x deployed). `largest_passing_k2` leaves every window
  at 1.18.
- Realised CV1 shift against the CV1-only window at the same centre: median |shift| / σ_w1 is
  0.028-0.071, max 0.13-0.35, per phase.
- The mechanical linear-response prediction k2 <z − z0> dz/dc / k1 reaches at most 0.085 σ_w1.
  Its correlation with the realised shift is 0.04-0.40 across phases.
- The gate cannot bind here, so the data say nothing about 0.25 for this branch.

### 4.2 CV1-unrestrained CV2-only windows (64, 68, 72, 76; z0 = -1.85, -0.78, +0.28, +1.35; k2 = 1.18)

- Gate at deployed k2 (swarm sd):
  - fraction 0.419 (production anchor sd: 0.418), from the mean-shift bound (curvature ratio
    0.044);
  - largest passing k2 is 0.42;
  - fraction vs k2: 0.12 at 0.1, 0.25 at 0.42, 0.42 at 1.18, 0.67 at 3.0, 0.73 at 3.54.

Realised CV1 mean offset from the anchor, in anchor sd, per phase. Each cell reads realised /
gate LR prediction / anchor-regression prediction:

| phase | 64 | 68 | 72 | 76 |
|---|---|---|---|---|
| epoch_000 | -0.24 / +0.22 / -0.04 | -0.19 / +0.08 / -0.01 | -0.01 / -0.10 / +0.01 | -0.20 / -0.31 / +0.03 |
| epoch_001 | -0.25 / +0.18 / -0.10 | -0.01 / +0.07 / -0.03 | +0.04 / -0.07 / +0.04 | -0.11 / -0.22 / +0.10 |
| epoch_002 | -0.19 / +0.21 / -0.13 | -0.04 / +0.05 / -0.03 | +0.06 / -0.11 / +0.06 | +0.24 / -0.25 / +0.17 |
| final | +0.06 / +0.21 / +0.02 | +0.01 / +0.07 / +0.01 | -0.01 / -0.09 / -0.01 | -0.01 / -0.26 / -0.02 |
| final_extension_001 | -0.19 / +0.23 / -0.13 | -0.03 / +0.07 / -0.04 | +0.10 / -0.09 / +0.05 | +0.10 / -0.25 / +0.14 |
| final_extension_002 | -0.03 / +0.22 / 0.00 | -0.01 / +0.07 / 0.00 | +0.12 / -0.08 / 0.00 | -0.04 / -0.26 / 0.00 |
| final-combined | -0.06 / +0.22 / -0.05 | -0.01 / +0.07 / -0.02 | +0.05 / -0.09 / +0.02 | +0.04 / -0.25 / +0.05 |

Mean over the 6 disjoint phases (epoch_000 through final_extension_002) ± SE, where
SE = phase sd / √6. Sequential seeding makes the phases not strictly independent.

| window | realised (anchor sd) | phase sd | gate LR prediction | (prediction − realised) / SE | (0.419 − \|realised\|) / SE |
|---|---|---:|---:|---:|---:|
| 64 | -0.139 ± 0.051 | 0.125 | +0.211 | 6.8 | 5.5 |
| 68 | -0.043 ± 0.030 | 0.073 | +0.068 | 3.7 | 12.6 |
| 72 | +0.051 ± 0.022 | 0.053 | -0.089 | -6.4 | 16.9 |
| 76 | -0.005 ± 0.064 | 0.157 | -0.258 | -3.9 | 6.5 |

- The gate's prediction is 3.7-6.8 SE from the realised shift, with the opposite sign in 3 of
  4 windows. The bound 0.42 is 5.5-17 SE above the realised |shift|.
- Individual phases scatter by ±0.2 sd. Single-phase values up to 0.25 are therefore within
  noise of the 0.25 threshold and cannot by themselves show the true shift is below it; the
  pooled means above can (≤ 0.14 + 2 SE = 0.24 for the worst window).
- The realised shift follows the anchor's own equilibrium c-z correlation (corr ≈ 0.13 in
  epoch_002). That correlation is not stable between phases. This is the thermodynamic effect,
  which the explicit-derivative model does not contain.

### 4.3 CV2 neighbour overlap vs k2

F''_2 = RT/var(z) − k2 was estimated at the deployed k2 on final-combined: median 1.36 over the
27 adjacent CV2 pairs, range 0.89-2.48. The CV2-only chain uses Δz = 1.066. Harmonic prediction
for the adjacent overlap (measured pairwise at 1.18: 0.38 / 0.39 / 0.39):

| k2 | 0.1 | 0.2 | 0.42 | 0.6 | 1.0 | 1.18 | 1.5 | 2.0 | 3.0 | 3.54 |
|---|---|---|---|---|---|---|---|---|---|---|
| predicted O (64-68) | 0.498 | 0.493 | 0.475 | 0.456 | 0.408 | 0.387 | 0.351 | 0.301 | 0.220 | 0.187 |
| coupling fraction (k1 = 0) | 0.12 | 0.17 | 0.25 | 0.30 | 0.39 | 0.42 | 0.47 | 0.55 | 0.67 | 0.73 |

- The realised mean separation is 1.03-1.16 sd, which gives a Gaussian O of 0.37-0.39. This
  matches the measured 0.38-0.39, so the model is calibrated at the deployed point.
- At the gate's 0.42 the adjacent CV2-only windows would overlap at 0.475, close to the 0.5 of
  identical states. They would add almost no CV2 resolution.
- **Coupling threshold call.**
  - k1 > 0: the data cannot decide. The gate is never within a factor of 60 of binding.
  - k1 = 0: the evidence goes against enforcing the shift branch at 0.25 with this model. The
    pooled realised shift is ≤ 0.14 ± 0.05 sd, the predictor misses by 3.7-6.8 SE with the
    wrong sign in 3 of 4 windows, and obeying it would roughly halve the useful CV2 resolution
    of the axis windows.
  - If a shift gate is kept, base it on the realised shift, with its own threshold (for
    example ≥ 0.5 sd) and its between-phase spread (±0.2 sd here) as the noise floor.
  - Only c9 (one k2, one CV2 definition) was available, so this is a single-point check, not a
    calibration.

## 5. Bug: geometry chain edges are not neighbours (feeds the weak set and R1)

`gareus/adaptive_production.py` `build_geometry_edges` (lines 2330-2332):

```python
order = list(np.argsort(primary))
for left, right in zip(order[:-1], order[1:]):
    add(int(left), int(right), "primary_chain", None)
```

- The chain is sorted on `primary_center` only. On a 2D layout, several states share a CV1
  centre. On a CV2-only axis, all states share the CV1 placeholder.
- The chain therefore joins pairs in whatever order `argsort` leaves the ties. The edge between
  two CV1 columns lands on arbitrary CV2 rows.

On chignolin_9 λ = 0:

| edge | coordinates | pairwise | the true neighbours |
|---|---|---:|---|
| 64-76 | CV2-only z0 -1.85 vs +1.35 | 0.080 | 64-68 0.38, 68-72 0.39, 72-76 0.39 |
| 68-76 | CV2-only -0.78 vs +1.35 | 0.20 | (68-72, 72-76) |
| 160-216 | CV1 0.436, z0 +0.28 vs -1.85 | 0.14-0.19 | via 112 (-0.78) |
| 136-232 | CV1 0.501, +0.28 vs -1.85 | 0.14-0.20 | via 140 |
| 184-204 | CV1 0.371, +0.28 vs -1.85 | 0.14-0.21 | via 92 |
| 164-200 | (0.577, -0.78) vs (0.630, +1.35) | 0.137-0.140 | 164-188 same row |

- Under the CV1 marginal these edges read about 0.8-0.95 and were never weak. `pairwise-mbar`
  is the first metric that sees their real separation.
- `edge_is_weak_pairwise` makes only collector geometry edges weak-eligible, so these edges are
  the entire weak set on c9. Midpoint bridges on them duplicate existing rows. For example,
  64-76 gets 3 new centres at z = -1.05, -0.25, +0.55, placed between the existing 68 and 72.
- The bridge-reason text also quotes the marginal (`overlap=0.777`) for an edge that was weak
  on pairwise 0.080. This is cosmetic: `reason` formats `edge["overlap"]`.
- **Suggested fix (not applied; production code untouched):** make the weak-eligible set the
  P7a nearest same-pattern neighbours per axis, or chain within each CV1 column by CV2 and
  across columns by the nearest row, instead of `argsort(primary)`. The spec's R1 needs the
  same fix, because as written it would bridge 160-216, 136-232 and 184-204 (ΔCV1 = 0, "differ
  mainly in CV2").

## 6. Task 4: proposer dry-run (cap-ignoring)

**Validation against the real campaign.** First, `--policy-from epoch_002/adaptive_epoch_actions.json
--real-caps` was run: the campaign's own recorded policy, which differs from the defaults in
`min_samples_for_add` 20000, `min_samples_for_retire` 30000, `retire_converged` False, pool
sizes and seed-bank size. It reproduces the recorded actions exactly on both metrics:

- c9 epochs 0-2: 236 `extend` each (reason "below minimum retirement sample count"), 0 adds;
- c7 epochs 0 and 2: 64 `extend` each.

A per-epoch state never reaches 20,000 samples, so under c9's rules neither metric bridges
anything in a numbered epoch. Only the final-combined payload clears the floor. There, with
the recorded caps (4 new windows), marginal would place 4 centres and pairwise-mbar 2 (164-200).
The final phase is frozen and never runs the proposer, so this last result is hypothetical.

The table below uses the DEFAULT rules (`min_samples_for_add` 50) with caps lifted. It shows
what the metric itself asks for.

`propose_actions_from_diagnostics` was run on a deep copy of the registry with
`max_new_windows_per_epoch` = `max_new_rungs_per_epoch` = 10000, `max_replicas_budget` = 0,
T = 300 K and `cv2_k_max` from the run manifest. Nothing was applied
(`scripts/t3_proposer_dryrun.py`). Each bridge centre becomes 4 states (one per rung) at apply
time.

| phase | marginal: weak / centres (states) | pairwise-mbar: weak / centres (states) | pairwise targets |
|---|---|---|---|
| c9 epoch_000 | 7 / 8 (32) + 3 retire | 2 / 5 (20) + 3 retire | 64-76 (3), 160-216 (2) |
| c9 epoch_001 | 2 / 3 (12) | 5 / 11 (44) | 64-76, 164-200, 160-216, 136-232, 184-204 |
| c9 epoch_002 | 2 / 3 (12) | 1 / 3 (12) | 64-76 |
| c9 final | 6 / 7 (28) | 2 / 5 (20) | 64-76, 164-200 |
| c9 final_extension_001 | 5 / 6 (24) | 2 / 5 (20) | 64-76, 164-200 |
| c9 final_extension_002 | 3 / 4 (16) | 2 / 5 (20) | 64-76, 164-200 |
| c9 final-combined | 6 / 7 (28) | 2 / 5 (20) | 64-76, 164-200 |
| c7 all phases | 0 / 0 | 0 / 0 | – |

- Marginal targets are all cross-pattern placeholder edges. Their bridges get averaged restraints
  such as k1 = 116.6 (the mean of 0 and 233). Every marginal trigger is CV1 marginal < 0.30;
  none is acceptance.
- No `add_rung`: rung edges carry no pre-union overlap.
- Neither metric proposes `extend`.
- Reserve sizing (Section 10 item 7): the raw 95th percentile over c9 epochs would be 44 states
  (19 % of 236) under pairwise-mbar. Every one of those proposals is an artefact, so the
  physically motivated need on c9 is 0.
- R1's structural "immediate" case never occurs: 1 component in every phase once unmeasured edges
  are kept.
- Do not set the reserve from these counts until Section 5 is fixed and the dry-run is re-run.
- Under c9's own rules the reserve need was 0 regardless: the sample floor blocked every add.

## 7. Task 5: X8 discovery census (cited, not re-run)

This is the census from CLAUDE.md "X8 discovery census", on the same data:

- c9 is still `discovering` by the shipped rule: cluster last/mean 0.24, or 0.42 without the
  partial `final_extension_002`.
- The cluster curve is nearly flat against the first epoch (last/first 0.036).
- c7: last/mean 0.48, last/first 0.107.

## 8. Recommendations

1. Fix the weak-eligible edge set (Section 5) before flipping `--ap-edge-metric` to
   `pairwise-mbar`, and before implementing R1. With the fix, c9 and c7 have 0 weak edges and
   the proposer's need is 0.
2. Thresholds: keep 0.15 (weak) and 0.25 (target) on the pairwise scale for spatial and rung
   edges. Confidence is moderate that 0.15 is safe and low on its exact location. T2 must supply
   the failure point.
3. Treat time non-stationarity (0.03-0.17 kT half-to-half Δf drift at every overlap) as the
   dominant real error. Overlap thresholds do not address it; X3/X5-type measures are the
   relevant ones.
4. Coupling gate:
   - keep the curvature branch at 0.25, though it is untested in the binding regime;
   - do not use the k1 = 0 mechanical shift bound at 0.25. It would cut the axis windows' k2
     1.18 → 0.42 against realised shifts ≤ 0.25 sd and predictions of the wrong sign.
5. R1 as specified would act on the wrong edges on c9 (the skip-a-row 2D artefacts) and never on
   a real gap, because none exists.

## 9. Scripts and outputs

Scripts are in `docs/superpowers/specs/2026-09-29-adaptive-cv2-validation/scripts/`:

- `t3_replay_phase.py`: one phase, redirected writes;
- `t3_edge_tables.py`: Task 1;
- `t3_gaussian_overlap.py`: the reference curve;
- `t3_union_reference.py`: Task 2 (a)/(c) and the rung gate check;
- `t3_exchange_transitions.py`: Task 2 (b);
- `t3_calibration_summary.py`;
- `t3_coupling.py`: Task 3;
- `t3_proposer_dryrun.py`: Task 4 (`--policy-from`, `--real-caps` for the validation run);
- `t3_cut_edges_and_coupling_se.py`: cut-edge check and the k1 = 0 shift SE.

The full per-phase class tables and every weak edge of every phase (Task 1) are committed as
`t3_edge_tables.txt` next to this report.

Other outputs are in `/tmp/t3/`, outside the repo and not committed:

- payloads and NPZ sidecars: `out/<campaign>/<phase>/`;
- `tables.{json,txt}`;
- `union_c9_vs_final.json`, `union_c7.json`;
- `calib_c9.json`, `calib_c7.json`;
- `exch_c9.json`, `exch_c7.json`;
- `coupling_c9_*.json`;
- `prop_c9.json`, `prop_c7.json`.

Run with `PYTHONPATH=<worktree>`.


## 10. Follow-up replays for P1, 3.2, 3.3 and 3.7 (2026-09-30, branch `feat/cv2-resolution`)

All replays are read-only on chignolin_9, and nothing was written under `RUNS/`. The two kinds of
replay treat the replica cap differently:

- The 3.3 dry runs (10.2) ignore the cap. They are run with `ignore_budget`, so no action is ever
  refused for budget.
- The 3.2 shape replay (10.1) honours it. c9's cap is 236 replicas on 4 rungs, i.e. 59 spatial
  states.

The scripts are in `scripts/t3b_*.py` and `scripts/t3c_*.py`. Every behaviour below is off by
default, and none of it has run in MD.

The numbers below are post-fix: they include review fix (f), which computes F'' from the
de-regularised variance (`variance_curvature`), and carrying the epoch report into the
final-combined table (`aa1a9d2`). Where a number changed, the pre-fix value is given in
parentheses. The sources are the replay JSONs, not prose:

- the shape replay: `c9_shape_r0.0_m8.json`, before and after;
- the dry-run reports: `dry_epoch_002.json`, `dry_epoch_002_ss.json` and `dry_final.json`;
- the 3.7 summaries.

### 10.1 3.2 shape layout on the chignolin_9 swarm

The input is 48,024 frames from 174 members, analysed in 15 CV1 columns. The fit accepts 26 CV2
modes across them. Script: `t3b_shape_layout_replay.py [reserve] [min_mode_members] [out_dir]`.

| Shape rule | Centres at the k2 floor | k2 range (uniform layout: 1.18) | Mean compression k2/(k2+F'') | CV2 centres per column |
|---|---|---|---|---|
| As specified: k2 = RT/sigma_w^2 - F'', variance-space shrinkage | 49 / 63 | 0.001-0.106 (median 0.001) | <= 0.09 | 3-6 |
| + mean-compression floor k2 >= F'' (`980fcaf`) | 0 / 63 | 1.07-2.73 | 0.50 at every centre | 5-7 |
| + precision-space shrinkage (`1d94d8d`) | 0 / 99 | 1.07-13.05 | 0.50 at every centre | 5-9 |
| + F'' from the de-regularised variance (review fix (f)) | 0 / 101 | 1.10-13.45 | 0.50 at every centre | 5-10 |

- **Why the specified rule failed.** The target width is the uniform layout's sigma_w = 0.711.
  That equals the swarm's own per-column CV2 sd of 0.6-0.75, so RT/sigma_w^2 - F'' is <= 0. The
  windows are then restrained in name only, and their sampled means follow the mode rather than
  their centres.
- **What the floor does.** It binds at every chignolin_9 centre (101 of 101), so on this system
  the width target never takes effect. Section 10.5 checks the F'' behind it against production
  data.
- **What precision-space shrinkage changes.** Narrow modes keep their curvature. The recurrent
  narrow mode at CV2 about -1.4 (sd 0.2-0.35) now gets k2 up to 13.45 (13.05 before fix (f)).
- **What fix (f) changes.** Over the 26 accepted modes, the median ratio variance_curvature /
  variance is 0.96 (minimum 0.43), so most F'' values rise by a few per cent. There are now 7
  X7 windows (6 before), with k2 1.12-11.69.
- **The layout is cap-bound. It fills exactly the 59-state spatial cap.**
  - The layout places 2 mandatory stacks: the unrestrained anchor, and one CV1-axis region
    representative (centre index 8).
  - It grants 57 ranked requests:
    - 25 mode cells;
    - 7 X7 per-mode CV1-free windows;
    - 14 remaining CV1-axis states;
    - 4 uniform CV2 rows;
    - 7 of 76 fill cells.
  - 2 + 25 + 7 + 14 + 4 + 7 = 59. The 69 dropped requests are all fills. Before fix (f) it
    was 2 + 25 + 6 + 14 + 4 + 8 = 59, with 66 of 74 fills dropped.
  - With a 0.15 reserve (re-run, `t3b_shape_layout_replay.py 0.15 8`) the fill cap drops to 50
    spatial states: 2 + 25 + 7 + 14 + 2 of the 4 CV2 rows = 50, so all 76 fills and 2 of the
    uniform CV2 rows are dropped.
  - `--ap-ladder-adapt respace` keeps the rung count, so it frees no replicas. Real headroom
    needs fewer rungs (X1, drop lambda = 1) or more replicas.

### 10.2 3.3 R1-R3 dry run

Script: `t3b_cv2_resolution_dryrun.py <adaptive_dir> <payload> <phase> <out.json> [--union NPZ]
[--transition-count ...]`. The payloads are the section-2 collector re-runs with
`--ap-edge-metric pairwise-mbar`.

| | epoch_002 | final-combined |
|---|---|---|
| R1 | 24 edges that differ mainly in CV2, all `ok`. 17 edges were not eligible and 35 were not CV2-mainly. 1 component. | same, 24 `ok` |
| R2 | Unavailable: top-ups were off, so there is no per-epoch union. | Ran on 143,513 lambda = 0 rows of `adaptive_union_mbar.npz` (0.41 GB), 0 candidates. Re-measured with R2's own interval code (`t3c_union_checks.py`) over 72 intervals in 15 columns holding >= 2 % of their slab's weight: 1-6 contributing centres (median 3-4); 58 of the 72 draw a contributor from another CV1 column. Bootstrap sigma 0.015-0.14 kT. The only interval with < 2 contributors holds a window centre, so it can never be a hole. |
| R3 (39 windows, `r3_gate`) | 36 no_action: depth_below_1kT 15, no_density_minimum 11, member_support 9, single_component 1. 3 flagged `trapped_or_orthogonal` (transitions_below_min). | 39 no_action: member_support 16, single_component 12, no_density_minimum 9, depth_below_1kT 2 |
| Actions | 0 | 0 |
| Wall time | 62 s | 146 s |

The three flagged windows sit on the top CV2 row (1.348). Their modes are at about +0.2 and +1.2
in CV2 units. The child values come from the state-series re-run, where the children are
planned:

| State | Depth (kT) | Crossings within replica residence | State-series switches | Child F''_est (lower / upper mode) | Child k2 at the cap (4 x 1.18 = 4.72) | Mean compression at the upper mode |
|---|---|---|---|---|---|---|
| 84 | 1.55 | 5 | 370 | 4.72 / 30.41 (4.7 / 30.3) | 4.72 / 4.72 | 0.13 |
| 200 | 1.10 | 0 | 469 | 1.87 / 7.32 (1.9 / 7.3) | 1.87 / 4.72 | 0.39 |
| 220 | 1.65 | 5 | 458 | 3.73 / 21.90 (3.7 / 21.3) | 3.73 / 4.72 | 0.18 |

- **The crossing count.** The replica-resolved count (`--ap-refine-transition-count replica`,
  the default) excludes exchange swaps that bring in a walker already in the other mode.
  chignolin_9 replicas stay about 2 samples per window, so that count almost never reaches the
  threshold of 10.
- **State-series counting.** Re-run with `--transition-count state-series` (the flag,
  `337ff9b`), all 3 windows pass the crossing test. They are refused
  `k2_capped_below_compression` instead: every upper-mode child, and 84's lower child, whose
  F'' of 4.72 equals the cap. Either way chignolin_9 gets 0 actions.
- **The binding limit.** It is the 4x-parent growth cap against upper-mode curvatures of 7-30,
  not the transition rule. The T2 calibration (`t2_synthetic.md` Section 9) finds this cap
  refusal on almost every synthetic R3 candidate too.
- **Why the 3 flags disappear at final-combined (condition 6).** More data shrink the upper
  CV2 mode, so the mixture gate stops before the transition test. Final-combined has 38,812 vs
  2,750 pairs per state, and 1,941 vs 1,375 subsample frames.
  - 84: epoch_002 had modes 0.25 (w 0.61) / 1.10 (w 0.27, in 16 of 32 blocks). At final the
    upper mode is 1.17 (w 0.075), in 5 of 33 blocks, which fails `member_support`.
  - 220: epoch_002 had 0.19 (w 0.68) / 1.20 (w 0.23, 19 blocks). At final there are two small
    components, 0.83 (w 0.13, 1 block) and 1.54 (w 0.03, 0 blocks), which fail
    `member_support`.
  - 200: epoch_002 had 0.17 (w 0.41) / 1.27 (w 0.36, 28 blocks). At final the accepted pair is
    0.49 (w 0.75) / 0.79 (w 0.20), with no density minimum between them. Its 1.47 component
    (w 0.05) has 1 block. The gate is `no_density_minimum`.
  - This is the mixture test responding to more data, not a code change.
  - These two phases cannot show whether a persistence rule (bimodal in >= 2 consecutive
    epochs) would change the epoch_002 verdict. epoch_001 was not replayed.

### 10.3 3.7 summary and report row

Command: `python -m gareus.adaptive.cv2_resolution_summary`, run on the 10.2 payloads and reports.

| | epoch_002 | final-combined |
|---|---|---|
| States / CV2-restrained / R3-evaluated | 236 / 172 / 39 | 236 / 172 / 39 |
| Graded edges / weak / unmeasured | 276 / 0 / 0 | 276 / 0 / 0 |
| Graded edges with pairwise point < 0.15 (never weak: neighbour, spanning or cross-pattern kinds, or q90 >= 0.15) | 91 | 84 |
| Components | 1 | 1 |
| trapped_or_orthogonal (the table's own counts) | 3 | 0 |
| `gareus_report` "CV2 resolution" row, as the table was built before `aa1a9d2` | CAUTION | PASS |
| Same row since `aa1a9d2`: the final table carries the newest epoch's 3.3 report | CAUTION | CAUTION ("final_combined; R3/budget from epoch_002": 3 trapped_or_orthogonal) |
| Same, state-series crossing count | CAUTION (3 spring-cap refusals) | CAUTION (3 spring-cap refusals, carried from epoch_002) |
| Confinement ratio cv2_sd / sigma_w2, lambda = 0 (43 states) | 0.56-0.84, median 0.72 | 0.55-0.77, median 0.66 |
| Graded-edge median overlap: pairwise / CV1 marginal / joint 2D | 0.25 / 0.52 / 0.50 | 0.26 / 0.51 / 0.52 |

- **The carried report.** The final-combined table now carries epoch_002's 3.3 report.
  - Rebuilt with the `aa1a9d2` CLI on a /tmp adaptive dir, the final table reads
    `sources.report_carried_from = epoch_002`.
  - Its trapped / spring-cap counts come from that report, while its edges and restraints stay
    final-combined.
  - So a campaign run like c9's would grade the final table CAUTION on epoch_002's 3 trapped
    windows, even though the final-combined data no longer show them as bimodal (10.2).
  - The CAUTION is therefore about the last numbered epoch, not the final data.
- The new row leaves chignolin_9's overall verdict at CAUTION, because other rows already
  grade it CAUTION.
- `adaptive_fig5_state_coordinates.png` shows the 4 rungs.
  - The CV1-only windows sit at their sampled CV2 means. These trace a curved valley rather
    than a placeholder row.
  - The 3 trapped windows are ringed in the epoch_002 panel.
- With `--ap-cv2-resolution` on, a campaign writes
  `adaptive_production/cv2_resolution_summary_final_combined.json`, the file the row reads,
  after every final-combined collection (`df8f387`).

### 10.4 What this means for chignolin_10

- **The shape layout.** With both follow-ups, the CV2 windows it designs are genuinely
  restrained on paper, at 5-10 centres per CV1 column. At chignolin_9 scale this does not fit:
  7 of 76 fills fit, and none with a 0.15 reserve.
- **The springs.** On c9 the whole spring field is k2 = F''_est, because the compression floor
  binds everywhere. The production cross-check (10.5) agrees with that F'' only to within a
  factor of about 1.6, so the realised compression will be 0.35-0.73, not 0.50.
- **R1-R3 find nothing to do on chignolin_9's data.** That is expected: its CV2 is residual PC1
  with uniform k2 = 1.18, and the layout is already connected on the pairwise metric.
- **Two decisions still open before launch, now answered by the T2 calibration**
  (`t2_synthetic.md` Section 9):
  - *The crossing count.*
    - `replica` never fires under exchange, and `state-series` passes every bimodal window,
      trapped or not.
    - A replica-path count, which joins a replica's own visits to the state, is the only
      estimator that separated equilibrated from mis-populated windows in the harness. It is
      not implemented in `gareus/`.
    - On c9 epoch_002 it gives 22 / 12 / 21 for states 84 / 200 / 220, against 5 / 0 / 5 for
      `replica` and 370 / 469 / 458 for `state-series`. That comes from the Parquet `replica`
      column (`t3c_replica_path_c9.py`, `t2_data/t3c_replica_path_c9.json`), which reproduces
      the dry run's own two counts exactly.
  - *The 4x cap.*
    - It refused all 202 R3 candidates on the three synthetic landscapes whose CV2 modes are
      real 2-D structure. c9's three need 6-26x.
    - It let through all 34 candidates on hidden-cv3, whose children need only about 2.2x.
      There the CV2 bimodality comes from the hidden slow coordinate.
    - At matched budget, the inserted children gave no PMF benefit in the harness with or
      without the cap.
    - Run R3 flag-only (no inserts) until T4 shows a benefit, whatever the crossing estimator.
      With the replica-path count, the 4x cap would insert only on the hidden-mode landscape
      (15 windows at re/8,000). The value 4 is irrelevant while inserts are off.
- **Set a reserve > 0.** Otherwise every action is refused `no_reserve`.
- **The other knobs.** The calibration recommends:
  - `refine_pmf_sigma_kt` 0.25, because today's bootstrap sigma is about 2.5x too small;
  - `coverage_min_windows` stays 2. It is structurally near-inert on a 2-D grid: on c9, 58 of
    72 heavy intervals count a centre from another CV1 column;
  - `refine_min_sigma` never binds;
  - `refine_min_transitions` 10, on the replica-path count. It is an absolute count and
    does not transfer between series lengths.

### 10.5 Condition 5: the swarm's F''_est against production-sampled curvature

Script: `scripts/t3c_f2_crosscheck.py`, which is read-only. It is reduced by `t3c_f2_summary.py`
to `t2_data/t3c_f2_crosscheck.json`, with the weight attribution in `t2_data/t3c_union_checks.json`.

**Swarm side.** This is the post-fix shape replay (10.1), taking per column:

- the pooled variance;
- the accepted mixture components, with `variance_curvature`;
- the layout's own rule for F''(z): `estimate_f2` of the dominant accepted component at z,
  falling back to the pooled value where a column has no accepted mode.

**The projection is the same.** The replay evaluates the swarm report's
`selected_component_index` = 1, residual PC1. Its uniform CV2 centres (-1.851, -0.785, 0.281,
1.348) are the production registry's. The 15 swarm CV1 columns are exactly c9's 15 CV1 centres.

**Production side.** Everything is lambda = 0 and final-combined, at RT = 0.596 kcal/mol, with
three estimators:

- **P1 window-local.** All 39 CV2-restrained 2-D windows (k2 = 1.18). F''_loc = RT/var(cv2 | cv1)
  - k2, using the conditional variance var2 - cov^2/var1. |corr(cv1, cv2)| <= 0.24, so
  conditional and marginal F'' agree within 11 %.
- **P2 CV1-only windows.** The 15 windows with k2 = 0, at the same CV1 centres. They sample CV2
  in a thin CV1 slab with no CV2 spring, which is the closest analogue of a swarm column.
  - Pooled RT/var2.
  - A mixture fit with 32 time blocks as members, as R3 does.
- **P3 union MBAR.** The 143,513 lambda = 0 rows of `adaptive_union_mbar.npz`, with their own
  MBAR over the 59 lambda = 0 states.
  - Per column, the swarm's own CV1 kernel times the unbiased weights. Kish n_eff is 5k-33k per
    column.
  - Pooled RT/var and a weighted mixture fit.

**Uncertainty.**

- P1/P2: block bootstrap over the P4 subsample's 32 time blocks per state, 200 reps. The 5-95 %
  range is about ±10-15 % of F''.
- P3 pooled: block bootstrap with f fixed, about ±2-5 %.
- P3 modes: 20 refits. Their CIs are wide, and sometimes exclude the point estimate when a refit
  matches a different component, so they are indicative only.
- No replica-level bootstrap is possible, because neither NPZ carries a replica column.
- The swarm F'' has no uncertainty estimate of its own.

| Level (production / swarm) | n | median | 10-90 % | range |
|---|---|---|---|---|
| Pooled per column, P2 CV1-only windows vs swarm RT/pooled var | 15 | 1.18 | 1.06-1.57 | 0.97-1.85 |
| Pooled per column, P3 union vs swarm RT/pooled var | 15 | 1.24 | 1.06-1.86 | 0.95-3.60 |
| Window-local, P1 F''_loc vs swarm model F''(sampled mean) | 39 | 0.86 | 0.51-1.56 | 0.30-1.90 (log-ratio sd 0.45) |
| Per mode, matched pairs only (P3 union vs swarm F''_est; means within one narrower sd) | 11 | see below | | |

**Agreement.**

- **Window-local curvature, the quantity that sets a window's spring.** It agrees to within a
  factor of about 1.6 (one log-sd), with no bias to speak of: the median is 0.86.
  - F''_loc is 0.83-2.75, and it is positive in every one of the 39 windows. Every c9 CV2
    window samples narrower than its sigma_w, which is what makes the "RT/var - k2" subtraction
    of the R-rules usable on real data.
- **Pooled per column.** Production is 18-24 % stiffer than the swarm in the median, and 1.5-3.6x
  stiffer in the lowest column (0.070) and the two highest (0.889, 0.934). The swarm is broader
  than equilibrium there.

**Where they disagree.**

1. **The swarm's recurrent narrow mode at CV2 about -1.4.** This is sd 0.18-0.35, F''_est 4-13,
   in 8 columns.
   - Where production has a matching component, at CV1 0.242 and 0.306, the production mode sits
     at -1.14 / -1.18 with sd 0.41-0.42. F'' is 3.35 / 3.60 (CI 2.5-6.1 / 2.7-4.1) against the
     swarm's 5.06 / 6.01, a ratio of 0.66 / 0.60.
   - In the other 6 columns production has no mode within one sd: the nearest component sits at
     -0.52 to -0.84, broad (sd 0.50-0.63) except at CV1 0.934.
   - So the swarm makes this mode narrower, and pushes it further out, than production samples
     it.
   - The shape layout gives its 2 centres per column k2 4-13. Under the union F'' they would
     compress to 0.60-0.86, not 0.5, and sample narrower than designed.
2. **The broad main mode.** In the 9 matched broad-mode pairs, production is narrower. The
   ratios are 1.08-3.46, median 1.5, and the means are shifted by up to 0.4.
3. **High CV1 (0.82-0.93).** The union fit finds narrow components that the swarm does not have:
   sd 0.09-0.12, weight 0.20-0.36, F'' 38-80, at CV2 about -0.5 and 0.0.
   - Each is carried by 5-13 effective states. The largest single-state share is 14-30 %, and it
     comes from 2-D windows at 0.281 / 1.348 and from CV1-only windows. So it is not one state's
     reweighted samples.
   - But the unbiased CV1-only window of the same column (state 60, CV1 0.934) shows no narrow
     mode there. The band is therefore unconfirmed: it may be real sampling or a
     reweighting/ergodicity artefact.
   - The union-model numbers that depend on it are conditional.
4. **The lowest CV1 column (0.070).** The swarm has no accepted mode there, so it falls back to
   pooled F'' 1.45. Production gives 2.3-2.75.

**What it implies for the springs.** k2 = F''_est at every shape centre. The primary check uses
the 41 of the layout's 101 centres that have a production 2-D window whose sampled mean lies
within the centre's predicted sampled sigma:

- the realised compression k2/(k2 + F''_loc) would be 0.35-0.73, median 0.53;
- 15 of 41 are below the 0.5 target, and 2 are above 0.7.

Conditional on the union mixture model, including the unconfirmed high-CV1 bands:

- the median over all 101 centres is 0.45, 10-90 % 0.23-0.80;
- 60 centres are below 0.5, and 12 below 0.25, almost all of those at high CV1.

The predicted sampled widths would change by 0.55-2.0x (10-90 %). The centre count barely moves
in the median (x1.1), but halves or doubles locally. 2 centres (CV1 0.934, CV2 -1.85 / -1.73)
have < 1 % union weight within one predicted sigma: they are extrapolated.

**Reading.**

- The 0.5 compression is a nominal target. With a design-measure F'' it is realised as about
  0.35-0.75 where production can check it.
- That is roughly what a factor-2 error in F'' gives: 0.33-0.67.
- The swarm does not reproduce the production mode structure, neither the -1.4 mode's width nor
  the high-CV1 bands.
- Options, none implemented:
  - after the first epoch, re-derive each window's k2 from its own samples, F''_loc =
    RT/var - k2. This is the same subtraction R1-R3 already use, and it is validated here in the
    sense that it is positive and stable on c9. Doing so means new state ids, because a
    Hamiltonian never changes under a state_id;
  - or cap how far the floor may rely on F''_est, for example min(F''_est, 2 x pooled F'').
- The T2 calibration (`t2_synthetic.md` Section 9) finds R3 cap-bound and without a measured
  benefit. So R3 cannot currently be relied on as the correction path for a mis-set spring.

**Confounds.**

- Production windows sample only near their own centres, so P1 is local. It checks the spring at
  41 of 101 centres, not the whole support.
- P3 covers the support, but only as well as the union's coverage and reweighting do.
- Bimodal windows make F''_loc an average curvature.
- All comparisons are at lambda = 0.
- The swarm side is one replay with no error bar.

### 10.6 Report v3 on chignolin_9 (follow-ups (h), 2026-09-30)

Read-only dry runs through the shipped code (`scripts/t3b_cv2_resolution_dryrun.py`, v3
defaults: `--ap-refine-transition-count replica-path`, `--ap-refine-r3-mode flag`,
`--ap-coverage-count same-column`, `refine_pmf_sigma_kt` 0.25; outputs outside RUNS, compact
record in `t2_data/t2c_r2v3.json` under `c9`).

- **epoch_002 R3, replica-path.** States 84 / 200 / 220: replica-path **22 / 12 / 21**, the
  same numbers as `t3c_replica_path_c9.py` (10.4), with identical core bounds (e.g. 84:
  0.75698 / 0.87051) and the other two counts recomputed alongside (replica 5 / 0 / 5,
  state-series 370 / 469 / 458). At >= 10 all three pass the transition gate and reach the
  spring code, where the 4 x 1.18 cap refuses them `k2_capped_below_compression` (would_be
  refused; flag mode changes nothing for a refusal). So the epoch_002 CAUTION changes cause:
  3 trapped_or_orthogonal (v2 default) -> 3 spring-cap refusals (v3 default). 36 other windows
  no_action; 0 actions; 61 s.
- **final-combined R2 on the 143,513 lambda = 0 union rows.** Row sources from
  `adaptive_union_mbar.samples.csv` (2 sources); bootstrap g median 1.8 rows (q10 1.2, q90 4.0:
  the builder already thinned them), no state at the 5-block guard, 1 g fallback. `any`: 0
  proposals, as in 10.2. `same-column`: 3 proposals -- CV1 0.177 at CV2 -0.01..0.64 (18 % of
  that slab's weight), 0.824 and 0.889 at -1.79..-1.14 (10 % / 6 %) -- each interval with 3
  contributing centres in total but only 1 of its own column, and bootstrap sigma 0.03-0.14 kT.
  These are the geometric proposals T2 9.9 describes (the neighbouring columns cover the
  interval); the sigma rule at 0.25 fires nowhere. 165 s total, 56 s union.
- For chignolin_10: flag-only R3 cannot insert; same-column R2 at 2 would spend reserve on
  intervals like these three. If that is unwanted, `--ap-coverage-count any` (or
  `--ap-coverage-min-windows 1`) restores the near-inert count rule; the choice is frozen per
  campaign.

### 10.7 Respring dry run on chignolin_9 (`--ap-cv2-respring`, 2026-09-30)

Script: `scripts/t3d_respring_dryrun.py`. It is read-only and runs the shipped
`cv2_respring_io.propose_respring` with the default knobs:

- min n_eff 200;
- tolerance 0.05, so a window triggers when the upper end of its c_real interval is below 0.45;
- per-epoch cap 0.25 x 59 centres = 14.

g is the X5 CV2 inefficiency over the phase's Parquet sources. The payloads are the replayed P4
ones from 10.2. The compact record is `t2_data/t3d_respring_c9.json`. After both runs,
`find RUNS/chignolin_9 -newer <stamp>` returned nothing.

**What this measures.** c9's CV2 springs are the uniform k2 = 1.18, not a shape design. So this
measures how far the uniform spring sits from the shape rule's 0.5 compression. It does not
measure how a shape layout's springs are realised; that is 10.5.

c9's `layout_plan.json` has no `cv2_shape` record, so the sigma target is the sampled sd. Every
triggered window therefore gets k2' = F''_prod, and its predicted c is 0.5 by construction.

| | epoch_002 | final-combined |
|---|---|---|
| candidates (CV2-restrained, lambda = 0) | 43 (39 2-D + 4 CV2-only) | 43 |
| X5 g (CV2), median [min, max] | 4.5 [3.0, 10.5] | 6.9 [3.3, 24.9] |
| n_eff, median [min] | 613 [261] | 5,590 [1,561] |
| c_real, median (10-90 %) [range] | 0.52 (0.40-0.61) [0.32-0.70] | 0.44 (0.37-0.56) [0.30-0.59] |
| F''_prod, median [range] | 1.10 [0.50, 2.57] | 1.53 [0.83, 2.72] |
| 90 % interval width of c_real, median | 0.082 | 0.049 |
| point c_real < 0.5 / < 0.45 | 20 / 11 | 30 / 24 |
| triggered (whole interval < 0.45) | 6 | 20 |
| proposed at the default cap (14) | 6 | 14 (6 deferred) |
| k2', median [range] | 1.86 [1.70, 2.57] | 1.83 [1.60, 2.72] |
| k2' / k2, median [range] | 1.57 [1.44, 2.17] | 1.55 [1.36, 2.30] |
| over-compressed flags / F''_prod <= 0 | 0 / 0 | 0 / 0 |
| wall time (X5 included) | 1.5 s | 15 s |

- **Where.** 15 of the 20 final-combined triggers are on the two outer CV2 rows: 8 at -1.851 and
  7 at +1.348. There the landscape is stiff and the window means sit far inside their centres.
  - 3 triggers are at 0.281 and 2 at -0.785.
  - The CV1 = 0.070 column triggers on both of its lower rows (80, 192; c_real 0.30 / 0.33).
  - None of the 4 CV1-free CV2-only windows (64-76) triggers.
- **epoch_002 vs final-combined.** epoch_002 triggers 6 windows: 80, 108, 128 and 152 at -1.851,
  192 and 196. The final-combined payload has about 9x more samples and triggers 20, a superset.
  A confident-only rule acts on more windows as its intervals shrink.
- **The final-combined intervals are conservative.** Its P4 subsample stride is 20 (n = 38,812,
  about 1,940 kept rows), while g is 3.3-24.9. So the bootstrap blocks are 1-7 rows, and the
  bootstrap sees about n / stride nearly independent rows instead of n_eff = n / g. The ratio is
  0.8-6.2, median 2.9, so the c_real intervals are up to about 1.7x too wide in the median.
  - Point estimates are unaffected. The bias is in the safe direction: fewer triggers.
  - With correctly sized intervals, more of the 24 windows whose point c_real is below 0.45 would
    trigger.
  - epoch_002 is consistent: its stride is 2, and n_eff / (n / stride) is 0.19-0.66.
  - A live epoch with a large P4 stride would show the same bias.
- **Cross-check with 10.5.** It uses the same final-combined payload.
  - The conditional F''_prod equals t3c's F''_loc for all 39 2-D windows (ratio 1.000 at every
    quantile). This uses the same payload and the same formula, so it checks arithmetic and
    wiring only; it is not independent evidence.
  - The marginal F''_prod, which is the value the rule decides on, is 0.91-1.00 of it (median
    0.996).
  - By t3c's point values, 27 of the 39 would sit below 0.5 under k2 = 1.18. The interval rule
    acts on 20.
- **Independent check of the compression premise (payload only, `t2_data/t3d_respring_c_mean_check.json`).**
  c_real is read from the variance under a harmonic assumption. It can also be read from the means:
  c_mean = (window CV2 mean - z0)/(c2 - z0), where z0 is the CV2 mean of the same column's
  CV1-only (k2 = 0) window. The check uses 34 2-D windows; 5 have |c2 - z0| < 0.3 and are
  excluded.
  - The two correlate at 0.72.
  - Over all 34: c_mean median 0.50 (10-90 % 0.41-0.59), c_real 0.42 (0.37-0.57).
  - Over the 18 triggered windows among them: c_mean 0.45 (range 0.32-0.53), and 14 of the 18 are
    below 0.5.
  - So the variance proxy reads about 0.05 lower than the mean-based measure. For 4 triggered
    windows the independent measure says the target is already reached.
  - The check itself assumes that the CV1-only window's mean is the column's unbiased CV2 mean,
    and that the displacement is harmonic.
- **Cost to the layout (predicted, not measured).** At k2' = F''_prod, the sampled CV2 sd of a
  triggered window shrinks by sqrt((k2 + F'')/(2 F'')). That is 0.85-0.93x on c9. Its CV2
  neighbour overlaps drop accordingly, and the next epoch's 3.1 metric re-grades those edges.
- **Not measured.**
  - Whether k2' actually realises 0.5. That needs MD. The prediction assumes F'' is unchanged as
    the window narrows, and a bimodal window's F''_prod is an average curvature.
  - The lambda > 0 rungs. The rule reads lambda = 0 only and replicates k2' onto every rung.
  - The epoch_002 run uses the final registry, as the 3.3 dry runs do.
  - No bootstrap over replicas, because the P4 NPZ carries no replica column. The blocks are
    time blocks of 5 g within a source.

### 10.8 R2 on gareus-analyze's MBAR solver and the re-solved-f bootstrap (2026-09-30)

Read-only, chignolin_9 final-combined, 143,513 lambda = 0 union rows x 59 states, 2 row sources.

**Solver.** R2's own lambda = 0 MBAR now runs on gareus-analyze's `numba-anderson` backend
(`cv2_coverage.solve_rows`, tol 1e-12, deterministic).
- 0.8-1.1 s and 45 iterations, against 42.9 s and 320 sweeps for the old NumPy loop.
- The old loop stopped 1.9e-7 from the answer. The new one is 7e-12 from a tol-1e-13 solve.
- The shipped dry run (`t3b_cv2_resolution_dryrun.py final_combined --union`, same flags as
  10.6) now takes 105 s, of which the union takes 6.7 s. In 10.6 it was 165 s, 56 s of it union.

**R2 re-run, fixed-f (the default).** Same-column still proposes **3**, at the same intervals:
- CV1 0.177 at CV2 -0.01..0.64
- CV1 0.824 at -1.79..-1.14
- CV1 0.889 at -1.79..-1.14

Their sigmas are 0.0333 / 0.0635 / 0.1403 kT, equal to the 10.6 record to 3e-11.

**R2 re-run, `--ap-coverage-bootstrap resolve-f`** (`t3e_r2_bootstrap_modes_c9.py`, plus the
same dry run with `--coverage-bootstrap resolve-f`):
- **Same 3 proposals.** Their sigmas are 0.037 / 0.060 / 0.114 kT, and the count rule still
  decides them: 1 own-column centre, 3 in total.
- **Over all 72 heavy intervals:** the resolve-f / fixed-f sigma ratio is 0.94 / 1.05 / 1.23
  (q10 / q50 / q90). The largest sigma is 0.140 kT fixed-f and 0.136 kT resolve-f. No interval
  exceeds 0.25 under either.
- **Why so little changes:** with 143k rows over well-overlapping windows, the relative f are
  determined far more tightly than the interval weights. The harness holes, where resolve-f
  mattered (T2 9.10), were near-disconnected.
- **Cost:** 100 replicates in 95 s (0.95 s each, median 46 iterations, max 68, 0 not
  converged). The dry run grows from 105 s to 205 s. Peak RSS 1.12 GB against 1.06 GB.

**Verdict for chignolin_10.** Resolve-f is affordable at this scale: about 1.5 min per epoch,
0.26 % of c9 epoch_002's ~10.1 h wall clock. It would change nothing here, and in the harness
it is not calibrated (T2 9.10). The default stays fixed-f with `refine_pmf_sigma_kt` 0.25.
