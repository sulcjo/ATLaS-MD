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
