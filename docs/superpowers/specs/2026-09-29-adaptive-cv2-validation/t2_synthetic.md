# T2 synthetic validation of the adaptive-CV2 spec (Sections 3.1, 3.4, 12 X3)

Date 2026-09-29. Branch `val/t2-synthetic`, base `963742e`. Spec:
`docs/superpowers/specs/2026-09-29-adaptive-cv2-resolution-design.md` (Section 5 T2).
No production decision code was changed. New code is in `gareus/synth/` and
`tests/test_synth_adapter_collectors.py`. Result summaries are in `t2_data/` next to this file.

## 0. Summary

| Item | Result |
|---|---|
| Adapter (review finding A5) | Done. Harness samples are written in the phase layout the shipped collectors read: Parquet or CSV samples, exchanges, `epoch_window_map.csv`, the window table and `run_manifest.json`. The real `collect_epoch_diagnostics` / `collect_segmented_epoch_diagnostics` then run, carrying P4 `paired_cv` and the 3.1 `pairwise_mbar` metric, followed by `propose_actions_from_diagnostics` and `_apply_registry_actions`. Parquet and CSV give the same grades to 1e-4. The calibration evaluator gives the collector's record to 1e-9. |
| Edge threshold (4) | **Keep 0.15. Lower `min_edge_neff` from 200 to 100.** At 0.15 with floor 100, false-weak is 0.7 % overall and 0.0 % on equilibrated pairs. False-strong is 4.8 % / 4.6 %, and all of it comes from edges whose true overlap is within 0.02 below the threshold (30 % miss rate in that band, ≤ 0.15 % beyond it). Floors 100 and 200 give the same rates, but 100 leaves 35 % of edge-measurements unmeasured instead of 47 %. Below 100 the rates double (N_eff 50–100: false-strong 10 %). Below 50 they collapse (false-strong 21 %, false-weak 7.5 %). In a campaign with the correlated double-branch gap, floor 100 roughly halves the pre-union silencing. Final pre-union weak edges are 4.0 against 8.7 post-union (floor 200: 0.3 against 8.0), and 2/3 seeds are resolved against 1/3. It adds no action on harmonic-bowl, and 5 bridges on gated-barrier where the windows are trapped. |
| Bootstrap guard | Correct on iid samples, too narrow on correlated ones. With exact iid draws the q10–q90 interval covers the truth in 80–81 % of cases (nominal 80 %). On correlated MALA chains it covers 50–67 % at N_eff 50–800 and 77 % at N_eff ≥ 800. Starting the chains from exact equilibrium draws changes nothing (50–77 %), so this is not the harness's start transient. It is the shipped moving-block bootstrap (block = ceil(g)) and the blocking τ on correlated series. The error it adds is small: it moves only the ≤ 0.02 band next to the threshold. |
| Threshold grounding | The per-edge Δf error has p90 > 0.5 kT only when the true overlap is < 0.06 (N_eff 100–400) or < 0.03 (N_eff ≥ 400). At the threshold (0.10–0.15) p90 is 0.27 kT. |
| Matched-budget campaigns (3) | Controls (harmonic, stiff, narrow band): 0 actions under pairwise-mbar. Marginal makes 0.7–6.7 adds and 4.7–19 retirements per campaign; part of that difference is the retirement bug below. Asymmetric double-branch CV2 gap: at 100 MALA steps/sample, both metrics find it and today's midpoint bridges resolve it (pre-set tolerance, 3/3 seeds each). At 20 steps/sample (more correlated) nothing resolves in 4 epochs (≤ 1/3 seeds). As states are added and the per-state budget shrinks, the pre-union pairwise verdict goes silent: 10–20 of 21–30 geometry edges end unmeasured. Without usable exchange acceptance, the marginal rule never sees the gap (0–0.3 weak edges; 1/3 resolved). In the branch free-energy error, the bridges barely beat a no-action arm on the same budget (0.60 vs 0.67–0.87 at 20 steps/sample, 0.28 vs 0.26 pairwise at 100). They connect the graph but add little PMF accuracy in 4 epochs. |
| Coupling gate (5) | 0.25 is safe for the channel the gate measures (CV1 width). At cf 0.26 the worst adjacent CV1 overlap goes 0.329 → 0.29, and the PMF error ratio is ≤ 1.75 on ridge rows. It is **not** safe for the tilt channel. When a CV2 row does not follow the CV1 dependence of CV2, the same coupling shifts windows by up to 0.7 σ_w1, and the matched-budget CV1-PMF error grows 3.1–3.8× at cf ≈ 0.26. It already doubles at cf_eff ≈ 0.05–0.07 (spec cf 0.06–0.18). Recommendation: keep 0.25 on curvature and add a tilt bound (mean shift ≤ 0.25 σ_w1, as the k1 = 0 branch already does). If only one number on today's quantity is used, take 0.10 (error ≤ 2.4×) or 0.05 (≤ 1.7×). |
| X3 reseeding (6) | Modest gain where one-sidedness means trapping. On `hidden-slow-cv3-weak` the final 2D PMF error goes 0.38 → 0.31 kT and the CV2 PMF error 0.31 → 0.22 (5/6 seeds better). The side deviation drops 18 %. It is harmful where the hidden mode projects strongly onto CV2 (`hidden-slow-cv3`): about half of the reseeds push equilibrium-one-sided windows to their equilibrium-minority side, and the side deviation does not improve. |
| Union memory (7) | Measured peak ≈ 0.8 GB + 55–58 B per (row × state) cell. `UNION_PEAK_BYTES_PER_CELL` assumes 61.6 B with no intercept. At 250k rows it underestimates by 7–11 %, and at 100k rows by 28–106 %. At the 8 GB guard this is about 0.3 GB over. The kept-row limit is 550k at 236 states, 477k at 272 (+15 % reserve), 406k at 320 and 325k at 400. |
| Production bugs | (1) **`--ap-edge-metric pairwise-mbar` disables `retire_converged` entirely.** Reproducer is below; it is a strict-xfail test. (2) On a 2D grid the `primary_chain` in `build_geometry_edges` includes wrap-around edges from the top of one CV1 column to the bottom of the next. These are weak-eligible under both metrics and account for most of the bridges on `plateau-walls`. |
| Not done | The end-to-end "double-branch resolved" assertion is written and skipped until 3.3 lands (R1–R3 do not exist; see Section 8). Nothing emulates the tICA estimate of the hidden mode or the `_non_neighbor_redundant_pairs` path. |

## 1. Methods

**Adapter** (`gareus/synth/collector_adapter.py`). `write_phase_dir` writes one phase:

- `samples/seg_001/*.parquet` via `ParquetSampleWriter` + `SegmentRegistry` (or `samples.csv`), rows in time order with `step` increasing;
- exchanges: Metropolis swap attempts on the two states' bias energies, 200 per geometry edge;
- an explicit `epoch_window_map.csv`, `umbrella_explicit_windows.csv`, and `run_manifest.json` with `resolved_args.temperature_k = 300`.

Units: harness F is in kBT and k in kBT/CV². The registry receives `k_kcal = k / beta(300 K)`, so reduced energies are identical on both sides. `exact_pair_overlap` gives the truth. It evaluates sqrt(N_a N_b) ∫ p_a p_b / (N_a p_a + N_b p_b), plus df, by quadrature on a local grid (≥ 10 points per σ_w, ±8 σ_w). For 3D surfaces it uses the exact (cv1, cv2) marginal. It matches the closed form on a flat surface to 2e-4.

**Sampler** (`gareus/synth/vec_langevin.py`). Vectorised overdamped Langevin with a Metropolis correction (MALA), so the chains sample exactly exp(−F − U) on the box while keeping real autocorrelation. Plain Euler-Maruyama was measured at +11 % in a restrained variance at D·k·dt = 0.2, which would bias every comparison against quadrature truth. Diffusion is D = (1, 0.05[, 0.05]): CV2 (and CV3) are slow.

**Landscapes** (`gareus/synth/landscapes.py`):

- new: `stiff-bowl` (F''₂ = 400 ≫ k2), `narrow-cv2-band-at-high-cv1` (the spec's shoulder mixture: 80 % N(0, 0.71) + 20 % N(1.2→1.6, 0.15) above CV1 0.75), `plateau-walls` (flat |CV2| < 0.6 between walls of 2000 kBT/CV²), `slow-cv2-double-branch-asym`, `coupled_tilt(a)` and the 3D `hidden-slow-cv3[-weak]`;
- reused: `gated-barrier`, `harmonic-bowl` (F''₂ = 1 < k2) and `slow-cv2-double-branch`.

The asymmetric double branch (+1.73 kT on the + branch) was needed because the symmetric one hides a disconnected MBAR: a solve started at f = 0 is right by symmetry. Its epoch-0 branch Δf error was 0.04 kT with truly disconnected windows.

**Swarm-like input.** `vec_langevin.swarm_design_measure` provides short unbiased runs from Latin-hypercube seeds. It is used by the X3 study, where 80 % of members start on the cv3 = −1 side. Section 3.2's shape layout, which would consume it, is not implemented.

Commands (seeded; reproducible):

```
python -m gareus.synth.t2_calibration --out OUT/calibration --pairs-per-landscape 80
python -m gareus.synth.t2_campaign --out OUT/campaign_sps20 --steps-per-sample 20
python -m gareus.synth.t2_campaign --out OUT/campaign_sps100 --steps-per-sample 100
python -m gareus.synth.t2_campaign --out OUT/campaign_asym_sps{20,100} --steps-per-sample {20,100} \
    --landscapes slow-cv2-double-branch-asym:k2x2 slow-cv2-double-branch-asym:k2x4
python -m gareus.synth.t2_campaign --out OUT/campaign_extra_sps20 --steps-per-sample 20 \
    --arms pairwise-mbar-neff100 no-action \
    --landscapes slow-cv2-double-branch-asym:k2x4 gated-barrier harmonic-bowl
python -m gareus.synth.t2_campaign --out OUT/campaign_extra_sps100 --steps-per-sample 100 \
    --arms pairwise-mbar-neff100 no-action --landscapes slow-cv2-double-branch-asym:k2x4
python -m gareus.synth.t2_calibration --out OUT/calibration_exact --sampler exact --lengths 250 500 1000 2000
python -m gareus.synth.t2_calibration --out OUT/calibration_malaeq --sampler mala-eq
python -m gareus.synth.t2_coupling --out OUT/coupling
python -m gareus.synth.t2_reseed --out OUT/reseed
python -m gareus.synth.t2_memory --out OUT/memory
```

Runtime: calibration 6 min; each campaign set 10–25 min at 14 workers; coupling 10 min; reseed 5 min; memory 5 min.

## 2. Threshold calibration (task 4)

400 window pairs were drawn over 5 landscapes (80 each):

- axis: CV1, CV2 or diagonal;
- k1 ∈ [80, 600] and k2 ∈ [8, 120] kBT/CV², log-uniform;
- the truth overlap was accepted to spread over 0.01–0.45, densest at 0.08–0.25.

Each pair was sampled by MALA from its centres and graded at 7 prefix lengths (250–16 000 raw samples, i.e. median N_eff 73–809), giving 2800 grades. All grading used the shipped `edge_metric.evaluate_edge` with 200 bootstraps. "Equilibrated" excludes grades whose sampled mean sits more than 0.5 exact sd from the exact window mean (15 % of grades; trapping on gated-barrier, narrow-band and harmonic CV2). Decision rule as shipped: an edge is weak iff q90 < threshold. Unmeasured grades are counted separately and are never weak.

Rates (FW = false-weak, P(flagged | truth ≥ t); FS = false-strong, P(not flagged | truth < t)). Each rate cell reads *all / equilibrated*; "unmeas." is out of 2800 grades.

| t | floor | unmeas. | FW | FS | FS in the band t−0.02 ≤ truth < t | FS for truth < t − 0.02 |
|---|---|---|---|---|---|---|
| 0.10 | 100 | 989 | 1.5 % / 0.9 % | 8.5 % / 8.3 % | 23 % | 0.0 % |
| 0.10 | 200 | 1305 | 1.0 % / 0.5 % | 7.4 % / 7.4 % | 20 % | 0.0 % |
| 0.125 | 100 | 989 | 0.9 % / 0.3 % | 5.5 % / 5.6 % | 27 % | 0.5 % |
| **0.15** | **100** | **989** | **0.7 % / 0.0 %** | **4.8 % / 4.6 %** | **30 %** | **0.12 %** |
| 0.15 | 200 | 1305 | 0.7 % / 0.0 % | 5.0 % / 4.6 % | 30 % | 0.15 % |
| 0.15 | 400 | 1689 | 0.2 % / 0.0 % | 4.6 % / 4.5 % | 29 % | 0.0 % |
| 0.20 | 100 | 989 | 1.2 % / 0.2 % | 4.1 % / 3.7 % | 43 % | 0.4 % |
| 0.25 | 100 | 989 | 1.2 % / 0.0 % | 2.3 % / 1.7 % | 29 % | 0.3 % |
| 0.30 | 100 | 989 | 1.6 % / 0.4 % | 1.4 % / 1.0 % | 32 % | 0.5 % |

At the 0.15 threshold, by N_eff bin (the lower N_eff of the pair):

| N_eff | n | FW (all / eq) | FS (all / eq) | q10–q90 covers truth (eq) | rms error / bootstrap sd (eq) | τ not plateaued |
|---|---|---|---|---|---|---|
| < 50 | 497 | 7.5 % / 1.8 % | 21 % / 18 % | 52 % | 1.8 | 100 % |
| 50–100 | 492 | 1.7 % / 1.0 % | 10 % / 9.8 % | 52 % | 2.1 | 98 % |
| 100–200 | 316 | 0.6 % / 0.0 % | 4.0 % / 4.4 % | 60 % | 2.3 | 53 % |
| 200–400 | 384 | 2.1 % / 0.0 % | 6.2 % / 5.1 % | 64 % | 1.9 | 7 % |
| 400–800 | 310 | 0.7 % / 0.0 % | 5.7 % / 5.1 % | 66 % | 1.8 | 1 % |
| ≥ 800 | 801 | 0.0 % / 0.0 % | 4.2 % / 4.3 % | 76 % | 1.6 | 0 % |

Per-edge |Δf error| (kT; median / p90, all grades) by truth overlap:

| truth overlap | N_eff 100–400 | N_eff 400–1000 | N_eff ≥ 1000 |
|---|---|---|---|
| < 0.03 | 0.27 / 0.88 | 0.27 / 0.57 | 0.10 / 0.31 |
| 0.03–0.06 | 0.18 / 0.51 | 0.11 / 0.24 | 0.08 / 0.16 |
| 0.06–0.10 | 0.14 / 0.48 | 0.07 / 0.15 | 0.06 / 0.12 |
| 0.10–0.15 | 0.10 / 0.27 | 0.07 / 0.14 | 0.04 / 0.10 |
| 0.15–0.20 | 0.09 / 0.22 | 0.06 / 0.17 | 0.03 / 0.08 |

Proposal:

- **Threshold 0.15.** The per-edge free-energy error only becomes large (p90 > 0.5 kT) below 0.06 at N_eff ≥ 100, so 0.15 keeps a margin of ~2.5× for chains of edges (error adds in quadrature) and for the bootstrap under-coverage. Below 0.15 the misses fall to the 0.02 band next to the threshold. 0.10 would also keep single-edge p90 < 0.5 kT at N_eff ≥ 100, but leaves no margin at N_eff 100–400 (p90 0.48 in the 0.06–0.10 bin).
- **`min_edge_neff` 100.** Its FW/FS equal floor 200's to 0.3 points. Under matched budgets, N_eff per state falls as states are added, and the 200 floor then silences real CV2 gaps. Floor 100 recovers about half of them in the campaign test (Section 3: final pre-union weak 4.0 vs 0.3 of ~8 real gaps, 2/3 vs 1/3 seeds resolved). 100 is where the blocking τ first plateaus in half the grades. Below 100 the false-strong rate doubles. Do not go below 50. Floor 100 does not close the gap: even there, 14 of 31 final geometry edges are unmeasured. The post-union overlap (`mbar_overlap`, which wins when present) is the more reliable signal late in a campaign, which argues for per-epoch union diagnostics (top-ups on) alongside the lower floor.
- The bootstrap needs separating into estimator and harness. Coverage of the q10–q90 interval at 0.15, equilibrated grades, by N_eff bin (the same 400 pairs under three samplers):

  | sampler | 200–400 | 400–800 | ≥ 800 |
  |---|---|---|---|
  | exact iid draws | 81 % | 80 % | 80 % |
  | MALA from centre (the calibration) | 64 % | 66 % | 76 % |
  | MALA from exact equilibrium draws | 64 % | 67 % | 77 % |

  The estimator is right for iid data, and the start transient is not the cause. Correlation is: at N_eff < 200 the blocking τ has not plateaued in 46–100 % of grades, so g is a lower bound there. A block of ceil(g) ≈ 2τ + 1 frames is also short for a moving-block bootstrap, which usually needs several τ.
  - Fix options, not implemented: blocks of ~3–5 τ, or widening the interval by ~2× below N_eff 800.
  - This matters only in the 0.02 band.
  - The near-threshold false-strong rate is intrinsic to the q90 rule: with iid draws it is still 4–10 % at 0.15.
- Exchange acceptance and overlap nearly coincide for equal harmonic windows: acceptance 0.08 ↔ pairwise overlap 0.153 (separation 2.5 σ). Today's `min_exchange_acceptance` therefore already approximates the 0.15 rule wherever an unbiased two-window swap is attempted. The two diverge on bimodal or trapped windows, on edges the exchange graph never attempts, and on inflated acceptance (gibbs-walk).

## 3. Matched-budget campaigns (task 3)

Each campaign runs 4 epochs of the real collector, then the real proposer, then the real applier, over 3 seeds per landscape and arm. The budget is 2000 samples × (initial states) per epoch, split uniformly over the arm's active states. Arms:

- `marginal`: today's rule, with synthetic acceptance (best case for today);
- `marginal-noexchange`: no exchange rows, so the acceptance test never fires (inflated or unattempted edges);
- `pairwise-mbar`: spec 3.1.

Two correlation regimes: 20 or 100 MALA steps per sample. Unmeasured geometry edges at epoch 0 (N_eff < 200), 20 → 100 steps/sample:

- gated-barrier 20/20 → 15/20;
- harmonic 11/21 → 9/21;
- narrow band 17/23 → 17/23;
- plateau 23/35 → 13/35;
- double branch 0–2/15 in both;
- stiff bowl 0/21 in both.

Slow soft CV2 windows are the unmeasured ones. The harness is therefore more correlated than chignolin_9, where 32/295 edges were unmeasured and the median N_eff was 584.

Starting layouts are today-like uniform grids with per-gap springs. The double branch has rows at ±0.4 with k2 = 2× (truth CV2-edge overlap 0.087) or 4× (0.008, disconnected) the per-gap spring.

Final epoch, seed means. "Weak" counts are under both predicates on the same pairwise payload; "comp v/t" = components by the collector's verdict / by analytic truth.

**Zero-action controls (N_eff regime 100 steps/sample; 20 gives the same pattern):**

| landscape | arm | actions / campaign | states 0→4 | weak marg / pw | CV2 PMF rmse e0→final (kT) |
|---|---|---|---|---|---|
| harmonic-bowl | marginal | add 2.7, retire 6.7 | 12→8 | 0 / 0 | 0.110→0.047 |
| harmonic-bowl | pairwise | none | 12→12 | 0 / 0 | 0.110→0.041 |
| stiff-bowl | marginal | add 3.0, retire 7.0 | 12→8.3 | 0 / 0 | 0.032→0.022 |
| stiff-bowl | pairwise | none | 12→12 | 0 / 0 | 0.032→0.020 |
| narrow-band | marginal | add 6.7, retire 16 | 18→9.3 | 0.3 / 0 | 0.119→0.052 |
| narrow-band | pairwise | none | 18→18 | 1.7 / 0 | 0.119→0.043 |
| plateau-walls | marginal | add 14.7, retire 27 | 24→13 | 5 / 5 | 0.059→0.041 |
| plateau-walls | pairwise | add 13.3 | 24→35 | 5 / 1.3 | 0.059→0.034 |

- On harmonic/stiff/narrow, pairwise-mbar takes zero actions. The marginal arm's actions are mostly retirements of redundant windows, which is legitimate under today's rule. That the pairwise arm never retires is the bug in Section 7.
- `plateau-walls` is not zero under either metric. The weak edges (truth 0.025, the estimate matches to 0.002) are `primary_chain` wrap-around edges from (c1ᵢ, +0.6) to (c1ᵢ₊₁, −0.6) (Section 7 item 2), not confinement triggers. The spec's intent (sd/σ_w and walls never trigger) holds: no such trigger exists in either rule.
- The narrow CV2 band is not found by either metric (no edge crosses it). That is 3.2/R2 work.

**CV2 gap: `slow-cv2-double-branch-asym`.** "Resolved" means all three of:

- branch Δf error ≤ 0.5 kT;
- low-F-weighted CV2-PMF rmse ≤ 0.5 kT;
- one analytic component.

Only the two 0.5 kT values were fixed before any campaign ran. The `components_truth == 1` condition was added after the first results, together with a key-name fix: the first summary compared a missing key and so reported everything unresolved. The spec asks for "PMF error in both branches". This criterion substitutes the between-branch free-energy error and a CV2 rmse whose low-F weighting down-weights the + branch (15 % of the population). No per-branch rmse was computed.

| steps/sample | k2 | arm | adds / retires | e0 weak (marg / pw / post-union) | comp v/t e0 | branch Δf err e0→final | resolved seeds |
|---|---|---|---|---|---|---|---|
| 100 | ×4 | marginal | 14.7 / 7.3 | 9 / 5.7 / 9 | 1.3/2 | 0.72→0.13 | 3/3 |
| 100 | ×4 | noexchange | 2.0 / 3.3 | 0 / 5.7 / 9 | 1.3/2 | 0.72→0.20 | 1/3 |
| 100 | ×4 | pairwise | 14.0 / 0 | 9 / 5.7 / 9 | 1.3/2 | 0.72→0.26 | 3/3 |
| 20 | ×4 | marginal | 15.7 / 1.3 | 9 / 7.3 / 9 | 1.0/2 | 1.35→0.84 | 1/3 |
| 20 | ×4 | noexchange | 1.3 / 3.3 | 0 / 7.3 / 9 | 1.0/2 | 1.35→0.74 | 1/3 |
| 20 | ×4 | pairwise | 9.3 / 0 | 9 / 7.3 / 9 | 1.0/2 | 1.35→0.87 | 1/3 |
| 20 | ×2 | marginal | 14.3 / 6.7 | 8 / 7.0 / 8.3 | 1/1 | 1.18→1.16 | 0/3 |
| 20 | ×2 | pairwise | 4.7 / 0 | 8 / 7.0 / 8.3 | 1/1 | 1.18→0.52 | 1/3 |
| 20 | ×4 | pairwise, `min_edge_neff` 100 | 14.3 / 0 | 9 / 9.0 / 9 | 1.0/2 | 1.35→0.67 | 2/3 |
| 20 | ×4 | no-action (same budget, no adds/retires) | 0 / 0 | 9 / 7.3 / 9 | 1.0/2 | 1.35→0.60 | 0/3 (2 components) |
| 100 | ×4 | pairwise, `min_edge_neff` 100 | 14.0 / 0 | 9 / 5.7 / 9 | 1.3/2 | 0.72→0.26 | 3/3 |
| 100 | ×4 | no-action | 0 / 0 | 9 / 5.7 / 9 | 1.3/2 | 0.72→0.28 | 0/3 (2 components) |

Floor-100 arm, final epoch at 20 steps/sample: 13.7 of 31 geometry edges unmeasured (floor 200: 16.3 of 24), pre-union weak 4.0 vs post-union 8.7 (floor 200: 0.3 vs 8.0), 4.7 verdict flips (floor 200: 7.7). At 100 steps/sample the two floors made identical decisions.

(The symmetric `slow-cv2-double-branch` cannot test resolution: its branch error is small by symmetry even when disconnected (Section 1). Its per-arm numbers are in `t2_data/campaigns_summary_sps*.json`.)

Observations:

- **The gap is seen.** At epoch 0 (×4) the marginal rule flags 9 geometry edges; the pairwise metric flags 5.7–7.3 pre-union and 9 post-union. The CV1 marginal overlap on those edges is 0.86–0.92. The marginal rule sees them only through acceptance (0.000–0.01). With acceptance unavailable it sees nothing (weak 0–0.7): that is the spec's "invisible unless acceptance < 0.08", reproduced.
- **The collector's component verdict hides the disconnection.** Truth is 2 components at epoch 0, the verdict 1.0–1.3, because unmeasured edges are kept as connecting (`components_rule`). The spec's R1 "structural" trigger reads this verdict, so it would not fire here.
- **Pre-union verdicts silence the gap as N_eff falls.** Adding states under a matched budget pushes CV2 edges below N_eff 200, and they go unmeasured: 10–20 of 21–30 geometry edges in the final double-branch epoch. At 20 steps/sample the final pre-union weak count is 0–0.3, while post-union is 3–8. That gives 2.7–7.7 pre/post verdict flips per final epoch at 20 steps/sample and 1.3–2.3 at 100.
- **Bridges connect the graph but barely beat "just sample more".** Today's bridges (midpoint, interpolated k2) turn the ×4 gap into one analytic component in every seen-gap arm. The branch free-energy error, though, is about what the no-action arm reaches on the same budget:
  - 20 steps/sample: 0.60 no-action vs 0.67–0.87 bridging;
  - 100 steps/sample: 0.28 no-action vs 0.26 pairwise; only marginal is better (0.13).

  The saddle bridges at interpolated k2 are bimodal (k2 + F'' < 0) and trapped. In the symmetric ×4 case at 20 steps/sample they *raised* the branch error (0.04 → 0.56 pairwise, → 0.18 marginal). Any 3.3 arm must therefore be judged against the no-action arm at matched budget, not only against the absolute tolerance.

## 4. Coupling gate (task 5)

Setup: `coupled_tilt(a)`, whose CV1 PMF is identical for every a. It carries a CV1 ladder of 8 windows (spacing 0.1, σ_w1 = 0.0667, k1 by the curvature rule) and one CV2 row, with k2 ∈ {10, 30, 60, 120} and a ∈ [0, 1.4]. Here dz/dc = a exactly. The spec's cf = k2 a²/(k1 + F''₁) is the rigid bound. The relaxed curvature is cf_eff = cf · F''_y/(k2 + F''_y), with F''_y = 60. The sampled CV1 sd matches 1/sqrt(1 + cf_eff) to 3 decimals. Two row placements:

- flat, z0 = 0: the row ignores the CV1 trend of CV2;
- ridge, z0 = a(c − 0.5): the row follows the conditional mean.

The PMF error is 6 seeds × 3000 samples/window, as a ratio to the same k2 at a = 0. At a = 0 the error itself grows with k2 (0.064 → 0.142 kT), because unbiasing a CV2 restraint stiffer than F''_y costs ESS.

| cf (spec) | cf_eff | flat: PMF ratio / extra mean shift (σ_w1) | ridge: PMF ratio / min adjacent overlap |
|---|---|---|---|
| 0.02–0.05 | 0.01–0.04 | 1.1–1.7 / 0.02–0.17 | 0.9–1.8 / 0.32–0.33 |
| 0.065–0.087 | 0.03–0.075 | 1.6–2.4 / 0.2–0.33 | 1.1–1.8 / 0.32 |
| 0.13 | 0.044–0.089 | 1.55–2.65 / 0.2–0.39 | 1.1–1.3 / 0.31 |
| 0.26–0.27 | 0.087–0.17 | 3.1–3.8 / 0.38–0.71 | 0.9–1.75 / 0.29 |
| 0.52–0.53 | 0.18–0.26 | 3.8–6.4 / 0.72–1.0 | 1.0–1.4 / 0.26 |
| 1.05 | 0.35 | 5.6 / 1.25 | 1.4 / 0.21 |

What the data support:

- For the quantity the gate computes (induced curvature, i.e. width), 0.25 is conservative. The width channel keeps the worst adjacent overlap ≥ 0.26 up to cf ≈ 0.5. It reaches the 0.25 target near cf 0.5, as the Gaussian prediction d' = 1.5·sqrt(1 + cf_eff) gives.
- The damage that matters comes from the tilt. A row that does not follow E[z|c] pushes CV1 windows by k2·a·(z0 − E[z|c]). It doubles the matched-budget PMF error at cf_eff ≈ 0.05–0.07 (spec cf ≈ 0.06–0.18, depending on k2 / F''_y) and roughly quadruples it at cf ≈ 0.26.
- For degree 1 the gate's formula has no tilt term (its (z − z0) d²z/dc² term vanishes). It bounds the tilt only for k1 = 0 windows (mean shift ≤ 0.25 σ).

**Proposed cut:** keep 0.25 on curvature and add the same mean-shift bound (≤ 0.25 σ_w1) for k1 > 0 windows. In the flat rows here, the 0.25 σ_w1 extra-shift level coincides with PMF ratio ~2. If a single number on today's cf is wanted, 0.10 (flat-row ratio ≤ 2.4) is the data-supported compromise; 0.05 keeps the ratio ≤ 1.7.

Caveat: the harness makes the thermodynamic slope m'(c) = d E[z|c]/dc equal to the geometric dz/dc. For a residual CV2, E[z|c] ≈ 0 on the design measure, so the real coupling is the residual slope. That slope can be estimated per window from P4's `cov_cv1_cv2 / var(cv1)`; it was not measured here. The closed form was not re-derived through `cv2_coupling_fraction` with a hand-built fit. For degree 1 it reduces to k2 (dz/dc)² σ_w1² / RT per its docstring.

## 5. X3 slow-mode reseeding (task 6)

Setup: a 5 × 3 window grid restraining (cv1, cv2), 4 epochs × 1500 samples/window, D3 = 0.05. No window crosses cv3's 7 kBT barrier in the campaign. The swarm seeds are 80 % on cv3 = −1. The `none` arm continues every chain. The `x3` arm runs the shipped `plan_reseed` (fraction 0.25, one-sided ≤ 20 %, within 2 σ_w) over all epochs' end states, with the shipped `split_point`. The hidden coordinate is the oracle cv3, not the tICA estimate. Both arms use the same burn-in. 6 seeds.

| landscape (equilibrium one-sided windows) | arm | final 2D PMF rmse (kT) | final CV2 PMF rmse | mean \|side frac − exact\| | one-sided (exact) | reseeds into eq. minority / reseeds |
|---|---|---|---|---|---|---|
| hidden-slow-cv3 (12/15) | none | 0.61 | 0.53 | 0.185 | 14.2 (12) | – |
| hidden-slow-cv3 (12/15) | x3 | 0.51 | 0.42 | 0.198 | 6.8 (12) | 4.7 / 8.2 |
| hidden-slow-cv3-weak (6/15) | none | 0.38 | 0.31 | 0.321 | 14.3 (6) | – |
| hidden-slow-cv3-weak (6/15) | x3 | 0.31 | 0.22 | 0.264 | 5.7 (6) | 3.8 / 9.0 |

- X3 helps where one-sided means trapped (weak projection). The error falls 19 % (2D) and 30 % (CV2), 5/6 seeds improve, and the side balance moves toward equilibrium.
- Its one-sided rule cannot tell a trapped window from one that is one-sided at equilibrium. That is the common case when the hidden mode projects onto the restrained CV2, as in the strong variant: 43–57 % of reseeds went to the equilibrium-minority side of an equilibrium-one-sided window. There the side deviation did not improve, and the PMF gain (0.61 → 0.51) is within seed spread (0.25–1.21 vs 0.24–0.69).
- Suggestion (not implemented): make eligibility compare against the pooled neighbourhood (e.g. windows at the same CV2 row) rather than the absolute 20 % cut.
- Limits: 15 windows (25 % = 3 reseeds per epoch), the oracle hidden mode, and no preflight.

## 6. Union-MBAR memory (task 7)

Measured with `union_solve_bench` in a fresh process per point (4 rungs × centres):

| states | kept rows | measured peak (GB) | estimate (GB) | measured/estimate | wall (s) |
|---|---|---|---|---|---|
| 60 | 100k | 0.76 | 0.37 | 2.06 | 3.5 |
| 120 | 100k | 1.12 | 0.74 | 1.52 | 6.8 |
| 236 | 100k | 2.09 | 1.45 | 1.44 | 18.8 |
| 272 | 100k | 2.40 | 1.67 | 1.43 | 17.6 |
| 320 | 100k | 2.78 | 1.97 | 1.41 | 25.8 |
| 400 | 100k | 3.15 | 2.46 | 1.28 | 33.2 |
| 236 | 250k | 4.03 | 3.63 | 1.11 | 46.2 |
| 400 | 250k | 6.61 | 6.16 | 1.07 | 80.6 |

Fit: peak ≈ 0.8 GB + 55–58 B/cell (the constant assumes 61.6 B/cell and no intercept). At the 8 GB guard the estimate is ~0.3 GB (4 %) low. A `+1 GB` intercept, or 7.2 × 8 B + 0.8 GB, would make the guard exact.

Kept-row capacity under the 8 GB default, with the budget implications for the P1 reserve:

| states | kept-row capacity | P1 reference |
|---|---|---|
| 236 | 550k | – |
| 272 | 477k | 236 + 15 % |
| 320 | 406k | – |
| 400 | 325k | – |

## 7. Production findings (reported, not fixed)

1. **`--ap-edge-metric pairwise-mbar` disables `retire_converged`.** `attach_edge_metric` appends `neighbour`/`spanning` edges whose `overlap` is None. The retirement loop in `propose_actions_from_diagnostics` marks both endpoints of any edge with `overlap is None or < target_overlap` as `bad_touching`, so no state is ever retire-eligible. Reproducer: 8 CV1 windows 0.04 apart on the harmonic bowl, exact samples. Marginal proposes 2 retirements. Pairwise proposes 0. Pairwise with the appended edges removed proposes 2 again. The test is `test_pairwise_metric_does_not_disable_retirement` (strict xfail; it will fail loudly once fixed). CLAUDE.md says retirement "still [runs] on the marginal"; in fact it is off.
2. **`build_geometry_edges` wrap-around chain edges.** `primary_chain` joins consecutive states by `argsort(primary)` over all 2D states. On a grid it joins the last state of one CV1 column to the first of the next. For states (0.3,−0.5), (0.3,0.5), (0.4,−0.5), (0.4,0.5) it creates edge (1,2) from (0.3,+0.5) to (0.4,−0.5). These edges are weak-eligible under both metrics and get midpoint bridges. On `plateau-walls` they are all 5 weak edges and most of the 13–15 adds. This contradicts the docstring ("immediate row/column-like nearest neighbors").
3. **Component verdict.** Design, not a bug. Unmeasured edges count as connecting, so a true two-component layout (double branch ×4) reads as 1.0–1.3 components. R1's "structural" trigger should not rely on it at low N_eff.
4. **Tilt missing from the coupling gate.** For k1 > 0 windows the gate has no tilt term (Section 4).

## 8. Limits, and what remains after 3.3

- Harness walkers do not exchange between windows (no replica exchange, no λ-ladder). Acceptance is an independent-sample two-window estimate. The CV2 slowness is set by D2/D3, not by a peptide.
- The truth is equilibrium. For trapped windows (15 % of calibration grades) the analytic overlap is not what the samples can support. Those grades are reported separately.
- The calibration uses the P4 stride subsample and blocking exactly as the collector does. The rates depend on the landscape mix: per-landscape FS at 0.15/200 is 2.7–8.2 %, FW 0–1.7 %.
- Not emulated: the tICA hidden-mode estimate (X3), top-up union diagnostics mid-campaign, the λ-ladder, the segmented collector's multi-segment pooling (the adapter supports it; the campaigns use flat phases), and 3.2's shape layout.
- **After 3.3:**
  - Switch on `test_slow_cv2_double_branch_is_resolved_end_to_end`. It runs `slow-cv2-double-branch-asym:k2x4` in the low-N_eff regime (20 steps/sample, 3 seeds) against the tolerance above. Today it fails: 1/3 seeds resolved. That regime is budget-limited (see the no-action baseline in Section 3), so a 3.3 arm must be compared with the no-action arm at the same budget, not only with the absolute tolerance.
  - Re-run `t2_campaign` with a 3.3 arm. R1's shape-rule k2 at the saddle should cure the trapped midpoint bridges.
  - Re-run the controls to confirm R1–R3 add no actions on harmonic/stiff/plateau (after fixing item 2) and that R2 finds the narrow band.
  - Re-calibrate `min_edge_neff` once 3.1 retirement is on the same metric.
