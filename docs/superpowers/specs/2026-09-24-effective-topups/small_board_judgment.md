# The Board's Judgment

## Majority position

**ACCEPT_WITH_CHANGES (unanimous, 3/3).** The board's final positions converged on the same verdict after an adversarial debate in which the design's core statistics were defended successfully, but its cost model and allocator internals were shown to need re-derivation.

The board affirms the following as correct in the submission:

- **Per-epoch union MBAR (Section A)** is the right diagnostic: pooled samples, warm-started f_k, decorrelated subsamples, and symmetric energy-space overlap for *every* spatial and rung edge, so no edge is ever unmeasured. The "unmeasured-is-never-weak" rule is a fallback, not a license to declare zero-overlap edges healthy. MBAR requires a *connected* state graph, not finite overlap on every pair; isolated zero-overlap edges in a connected graph do not produce singular weights or infinite variance.
- **Noise-vs-structural edge routing (Section B)** is statistically sound: more MD on two fixed Hamiltonians cannot increase their true overlap, so structural gaps must go to bridge-state proposals, and low-ESS endpoints should be topped up before declaring a gap structural (a conservative ordering).
- **Same-window checkpoint seeding (Section D)** is correct in principle and a genuine efficiency win (~25 min US-pull per 59 windows plus equilibration transient eliminated). Under the swap scheme's detailed balance, the replica occupying window k at baseline end is an equilibrium draw of window k, so continuation without burn-in is valid *provided the Hamiltonian is identical* — which it is, since windows are seeded from their own final replicas.
- **Exchange vs. MBAR validity** is correctly separated: exchange matters for mixing, not for MBAR validity, and ESS (a decorrelation metric) is the right place for mixing to enter the allocator.

The board holds that the following defects, exposed most sharply in cross-examination, are *load-bearing* and must be fixed before deployment:

1. **The cost currency is wrong.** The design's own benchmark shows a throughput inversion: 16 ctx/GPU = 3,154 ns/day/node aggregate vs. 59 ctx/GPU = ~2,300 ns/day/node — per-context throughput ~5× higher at low occupancy. Yet C3's water-filling optimizes benefit per state-ns (|patch| × L), which is not the binding resource. Cost must be denominated in wall-hours via a measured throughput curve T(n), with the curve filled in at unmeasured occupancies (24/32/48 ctx/GPU, and the low end). C4's packing-to-floor gestures at this but the objective must be re-derived, not patched.
2. **The ESS-rate predictor is mis-specified twice over.** Kish ESS of MBAR weights is not additive in steps (it depends on the pooled weight distribution across all states), and — independently — rates measured under full-graph exchange degrade inside a truncated patch. The allocator needs a per-state autocorrelation-based marginal rate (~1/(1+2τ)) with **online recalibration** from realized top-up performance, not ESS_k/steps_k.
3. **Seeding needs a consistency assertion.** The silent failure mode is a manifest saved before the final swap seeding window k with window j's frame, with no burn-in and no diagnostic downstream that would catch it. The spike test must assert that the seeded frame's reduced potential matches the last logged u_k for that window.
4. **var_k ~ c/ESS_k is a tractable surrogate, not a variance model.** MBAR covariance depends on the whole overlap matrix; the full covariance need not be computed at every allocation decision, but the design should use the σ_k it already computes (A) for the objective, and should not claim optimality for the discrete-L sweep beyond its grid.

On the two posed questions:

- **Walker duplication (E) should not replace separate top-up segments.** The two are complementary allocator actions. Duplication's marginal wall-hour cost is ~0 when spare contexts exist and a baseline segment runs anyway, and it preserves the full exchange graph — but its granularity is coarse (integer walkers × full segment), it has latency (waits for the next baseline), it requires velocity re-randomization on cloning (plus short discard before samples count toward N_k), and it is unavailable at chignolin_9's MPS limit. Separate top-ups win for deep/localized deficits and final epochs. The allocator should support both, chosen by predicted variance reduction per wall-hour under the corrected cost model.
- **Top-ups can beat uniform extension at equal ns here** — whenever deficits are heterogeneous, since concentrating ns on the lowest-ESS states reduces max per-state variance more than uniform spreading (over-service of below-median packing partners is not waste; extra samples always reduce variance). At equal *wall-hours*, the throughput inversion adds up to 1.37× aggregate (and ~5× per targeted state at 64 vs. 236 contexts). When all states are equally deficient, the allocator correctly degenerates to uniform extension (patch = all states). The converse bound: healthy epochs spending ~0 is right, and the 0.3 cap should be treated as applying to the selective part.

The board further notes, from the debate, the largest **unexploited** efficiency lever: the throughput inversion applies to the *whole campaign*, not just top-ups — sequential quarter-panels at ~15–16 ctx/GPU could deliver the same per-state ns in ~0.73× the wall-time of the full 236-state run, at the cost of cross-panel exchange-graph fragmentation (MBAR validity unaffected; equilibration of slow global modes may slow). This should be benchmarked on chignolin_9/10 hardware before adoption, and the A/B design must control for occupancy effects or it will confound targeting benefit with throughput benefit.

## Dissenting positions

None. The board is unanimous: all three members' final verdicts are ACCEPT_WITH_CHANGES. (Substantive disagreements persisted in emphasis — mini held that most objections were already implicit in the design, while kimi and thinker held the cost model must be re-derived — but these are disagreements about the *scope of required changes*, not the verdict, and the majority position above reflects the post-debate synthesis.)

## Confidence

**84 / 100.** Derived from the members' final confidences: kimi 85, mini 84, thinker 82 (mean ≈ 83.7, rounded). The unanimity of verdict despite adversarial cross-examination raises the board's confidence slightly above the simple mean; the residual uncertainty reflects the unmeasured regions of the throughput curve, the untested checkpoint-seeding route (spike pending), and the A/B's statistical power.

## Conditions

Required before deployment, numbered:

1. **Re-derive the allocator's objective in wall-hours.** Replace the |patch| × L state-ns cost with L / T(|patch|), where T(n) is a measured throughput curve; benchmark the curve at 4, 8, 16, 24, 32, 48 ctx/GPU on chignolin_9/10 hardware (the 16 ctx/GPU figure is from chignolin_8 and must be revalidated on the target node). Apply the same currency to the `ap_topup_max_fraction` cap and to the E-vs-D choice.
2. **Fix the ESS-rate predictor.** Use a per-state autocorrelation-based marginal rate computed on the pooled trace (with per-segment or per-graph g to handle heterogeneous exchange environments), not ESS_k/steps_k; implement online recalibration so realized ESS/step in patch segments feeds back into prediction (including a diminishing-returns detector: a top-up that under-delivers halves that state's priority next epoch).
3. **Add the seeding consistency assertion.** Before any top-up sample is counted, assert the seeded frame's reduced potential under window k's parameters matches the last logged u_k for that window (the manifest off-by-one failure is silent and contaminates MBAR undetected). Include this check in the first implementation spike on chignolin_9's real files. For the fallback path, the burn-in length must be determined from τ_int, not fixed arbitrarily, and equilibration detection must treat checkpoint continuation as continuation (no re-discard at segment boundaries).
4. **Harden the deficit diagnostic.** Kish ESS and asymptotic σ_k are blind to unsampled basins; pair them with block/split-halves free-energy comparison per state, use conservative autocorrelation estimators (initial positive/convex sequence), and compute the autocorrelation on the reduced-potential or weight-relevant observable.
5. **Use σ_k (already computed in A) as the allocation objective** rather than the c/ESS_k proxy; acknowledge the discrete-L sweep is optimal only on its grid, and credit partial (not all-or-nothing) variance reduction when sweeping L.
6. **If walker duplication (E) is implemented:** re-randomize velocities and differentiate RNG seeds on cloning, discard a short burn-in before duplicates count toward N_k, support variable replicas-per-state in the driver/manifest, and select between E and D per epoch by predicted variance reduction per wall-hour under the corrected cost model.
7. **Power and de-confound the chignolin_10 A/B.** Use paired seeds/common random numbers for the shared baseline phase, ≥3 replicates per arm (min per-state ESS is a noisy order statistic), ensure the two arms' RNG streams decorrelate after branching, and control for occupancy: either run both arms at the same contexts/GPU or report the targeting and throughput components separately, otherwise equal-ns comparisons confound allocator benefit with the 1.37× throughput inversion. Success criteria should include block-bootstrap free-energy uncertainty (already planned), not ESS alone.
8. **Benchmark the sequential-panel option** (full campaign at ~15–16 ctx/GPU in sequential quarter-panels vs. one 236-context run) on real mixing: measure realized per-state g in panel vs. full-graph mode before deciding whether to restructure the baseline itself; this is the largest available efficiency lever beyond the top-up design.
9. **A/B arms must not share RNG-correlated continuations:** if control and treatment branch from identical checkpoints, seeds must differ so early samples are not duplicated.
10. **Preserve the design's existing correct behaviors as tested invariants:** unmeasured-is-not-weak only as fallback after the per-epoch union solve; healthy states get zero allocation; structural gaps route to bridge proposal, never to repeated MD; noise-weak edges top up both endpoints; the allocator degenerates to uniform extension when all states are equally deficient.

---

Small board: kimi, mini, thinker | Chair: glm | Debate rounds: 1 | Final: majority ACCEPT-WITH-CHANGES at 100% | Unanimous: True
