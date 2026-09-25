# The Board's Judgment

## Majority position

**ACCEPT_WITH_CHANGES.** The board's majority (kimi, confidence 90; thinker's calls failed and cast no vote; the tally breaks the 1–1 split in favor of ACCEPT_WITH_CHANGES) finds the plan's architecture sound — pure diagnostics/allocator modules, one lockstep top-up segment per epoch, plan persistence for resume — but identifies a set of statistical and budget-accounting defects that would systematically waste MD or abort healthy epochs. These are all correctable without architectural change.

**On V1–V5 (all confirmed correct as measured):**

- **V1/V2 — split-halves false-flagging.** The per-pair z statistic is calibrated (4.3% vs 4.6% expected), but the code applies a fixed 2σ threshold to one test per state, K tests per epoch, and one fluctuation on a shared pair flags both endpoints. At K=236 this yields ~10 spurious "unconverged" states per epoch, each triggering a doubling top-up. **Correct fix:** the multiplicity family is the K states, not the neighbours per state — a Bonferroni/Šidák threshold z* = Φ⁻¹(1 − α/(2K)) ≈ 3.73 at K=236 (≈2.50 at K=4) with α = 0.05; retain the contiguous temporal split (V2 confirmed it calibrated — do **not** random-shuffle, which would destroy drift sensitivity); add a practical-significance floor (flag only if |Δf| > max(z*·comb, γ·target_σ)) and a minimum half-size guard; test each unordered pair once rather than inheriting the arbitrary `js[0]` choice.
- **V3 — decorrelation double-counting.** The union NPZ is pre-thinned, so `per_state_inefficiency` on kept samples is ~1 by construction, and `steps_so_far = kept × report_interval` makes the *forward* term overestimate new decorrelated samples by the thinning factor, undersizing every top-up. **Correct fix, dimensionally consistent with the existing formulas:** pass `steps_so_far` = true raw MD steps (kept × thinning × interval) and `g` = raw/kept, giving n_eff = kept and forward gain L/(interval·g); source g from the subsampling meta (`subsample_counts_per_state`), not from a Geyer pass over thinned samples. Note that raw/kept including the equilibration discard makes forward prediction conservative; prefer the post-equilibration raw count for the forward rate.
- **V4 — budget double-shrink.** Baseline is shortened to (1−cap)·default and the cap is then computed against the already-shortened baseline (0.7 × 0.3 = 21%). **Fix:** compute the top-up budget from the un-shortened default (cap × wall_hours(default_steps), or hours(default) − hours(baseline)).
- **V5 — local σ definition** behaves correctly (chain ends not penalised); both flagged deviations are endorsed.

**Further defects the verification missed (must be fixed):**

1. **L-candidate restriction:** the allocator only considers L ∈ need values; when every need exceeds budget it returns "cap_too_small" even when the budget funds a useful partial top-up. Add the budget-feasible maximum (interval-rounded) as a candidate.
2. **Healthy-epoch forfeiture:** the baseline is unconditionally shortened, so a healthy epoch forfeits the reserved cap fraction (~30%) of MD. Spend the reservation on a baseline continuation segment when no deficit exists (this does not violate "no distribution of left-over budget," which prohibits topping up healthy states).
3. **NaN-score argmax poisoning:** an unclamped correction factor c (negative or NaN) can make n_eff + c·extra/(interval·g) ≤ 0 → NaN pred → NaN score; since comparisons against NaN are False, a first NaN candidate sticks. Clamp c to a positive interval (e.g., [0.1, 2]) with EMA smoothing and validate finiteness of `pred`.
4. **Unsampled-state partners:** `partners_for` filters only on membership in σ; a never-sampled state (σ = NaN, ranked −1.0) can be chosen when it is the only pool member, but it has no seed State and no pull source. Exclude states with n_k = 0 from partner pools.
5. **Finite-filter hazard:** `np.isfinite(u).all(axis=1)` drops entire samples when any column is +inf; at scale (K=236) this can discard large, biased fractions of data. Reject NaN, allow +inf (zero Boltzmann weight is handled), and log drop counts. (The minority member's own-column-only filter would inject NaN into pymbar's active columns and break the solve — it is rejected.)
6. **Pull-seeded windows bypass equilibration discard:** a pull fallback is a chain restart, not a continuation; if the top-up segment's samples are assumed equilibrated, pulled windows contaminate the decorrelated set. Mark pulled-seed windows and apply equilibration discard to their top-up samples.
7. **No structural escalation memory:** a weak edge with a deficit endpoint that MD cannot fix attracts top-ups every epoch, forever. Add a per-edge attempt counter that reclassifies a persistently weak edge as structural after N failed top-ups.
8. **Resume/persistence hazards:** JSON round-trip coerces integer state-id keys to strings (predicted_sigma/sigma_before lookups fail silently on resume); NaN serialization must be handled; write topup_plan.json atomically (tmp + rename); persist online-calibration factors; validate a persisted plan against the current layout on resume and fall back to no-top-up rather than crash.
9. **BAR initialization cost:** `initialize="BAR"` performs O(K²) pairwise BAR solves (~28k at K=236) three times per epoch (full + two halves). Use `initialize="adaptive"`/`"zeros"` with the robust protocol, and warm-start the half-solves from the full-solve f.
10. **CV assertion tolerance:** the 1e-4 tolerance on CV reproduction can false-abort against float32 trajectory round-trip; compute CVs self-consistently from the exported positions, and downgrade assertion failure to warning + pull fallback for the affected windows so a healthy epoch is not killed.
11. **Throughput-table extrapolation:** flat extrapolation below 16 contexts/GPU may overestimate small-patch node throughput, underestimating cost; log realised vs predicted segment wall-time to calibrate the cost model, and cap patch size to hardware feasibility.

**Efficiency improvements:** reuse/warm-start the split-halves solves from the full solve; hoist `_worst` out of the L loop; export per-window final States periodically (every report interval) rather than only at segment end, so a crashed segment loses little seeding material; prune stale `final_window_states/` to the latest per window; expose g_k = raw/kept once in UnionDiagnostics instead of recomputing.

## Dissenting positions

"**Verdict:** ACCEPT – the plan is sound after fixing the verified split-halves false-positive rate and decorrelation bookkeeping.
**Confidence:** 92" — mini (final round).

The dissenting member argued that the additional defects raised in debate are mis-diagnosed or non-issues: that the split-halves family is one test per state (so the 2σ tail is the correct rate and Bonferroni would over-tighten), that Task 8's infrastructure already persists and subtracts `steps_completed` on resume so no MD is double-counted, that the cap is applied exactly once and the 0.7 × 0.3 = 21% arithmetic "conflates two independent scalings that are never multiplied together", that excluding structural edges from partner selection would break the intended separation of concerns, and that the gain model is already consistent with `steps_so_far` being the decorrelated count times the report interval. The majority rejects these rebuttals in the debate record: V4 is a *measured* finding of the plan's own T8 wiring, not an inference; the `steps_completed` bookkeeping the dissent invokes is not in the submitted plan text; and the multiplicity family is the K per-epoch state tests (the ~10 false flags/epoch at K=236 are measured, V2), so the dissent's FWER claim contradicts the verification data. The dissent is recorded verbatim above and was not paraphrased away.

## Confidence

**80.** Derivation: the mean of the two voting members' final confidences is (90 + 92)/2 = 91, but the board split 1–1 on the verdict itself and the third member (thinker) contributed no position (two infrastructure failures), so the majority position's effective confidence is discounted from the member-confidence mean to **80**, reflecting that the underlying technical findings (V1–V5, the mechanism-level analysis of the allocator and diagnostics) are agreed at high confidence by both members, while the *severity* judgment — whether the additional defects warrant required changes or mere recommendations — is where the board divided.

## Conditions

Required changes before implementation proceeds:

1. **Split-halves (V1/V2):** replace the fixed 2σ threshold with a per-epoch multiple-comparison threshold over the K-state family — z* = Φ⁻¹(1 − α/(2K)), α = 0.05 — plus a practical floor (|Δf| > max(z*·comb, γ·target_σ)), a minimum half-size guard, testing each unordered pair once, and an assertion that the NPZ stores samples in temporal order (contiguous halves retained; no shuffling).
2. **Gain-model inputs (V3):** pass `steps_so_far` = true raw MD steps and `g` = raw/kept from the subsampling meta, so n_eff = kept and forward decorrelated gain = L/(interval·g); remove the Geyer pass over pre-thinned samples; prefer post-equilibration raw counts for the forward rate.
3. **Budget (V4):** compute the top-up budget from the un-shortened default steps; add a regression test.
4. **Partial-L candidates:** add the budget-feasible maximum as an L candidate so a moderate residual budget funds a useful partial top-up instead of returning "cap_too_small".
5. **Healthy-epoch budget:** spend the reserved fraction on a baseline continuation segment when no top-up is planned, so healthy epochs do not forfeit ~30% of MD.
6. **Correction-factor safety:** clamp c to a positive interval, EMA-smooth it, persist it across resume, and guard `pred`/`score` against NaN (fail the candidate, not the argmax).
7. **Partner validity:** exclude states with n_k = 0 from partner pools; add the corresponding unit test.
8. **Finite filter:** reject NaN but allow +inf in u_kn, log drop counts, and add a test with an inf-containing column.
9. **Pull-fallback equilibration:** mark pull-seeded windows and apply equilibration discard to their top-up samples.
10. **Structural escalation:** persistent per-edge top-up attempt counter that reclassifies a persistently weak edge as structural.
11. **Persistence hygiene:** atomic topup_plan.json writes; int-key coercion on resume; NaN-safe serialization; layout-change validation of persisted plans; persisted calibration state.
12. **Seeding tolerance:** compute CVs self-consistently from exported positions, relax the 1e-4 assertion for float32 round-trip, and downgrade to warning + pull fallback rather than aborting the epoch.
13. **Cost model:** replace `initialize="BAR"` (O(K²)) with adaptive/zeros initialization plus robust solver, warm-start half-solves, and log realised vs predicted segment wall-time to calibrate the throughput table; cap patch size to hardware.
14. **Test additions:** budget-from-default regression (cond. 3); thinning-factor regression for required_steps (cond. 2); simulated FWER at K=236 below α; correction-factor clamping (c ≤ 0, NaN); TopupPlan JSON round-trip; resume-with-layout-change; states with 1 sample / odd counts; all-deficit patch; determinism of the allocator.

Validation remains synthetic-harness-only per the plan's constraint; no conditions require a live MD campaign.

---

Small board: kimi, mini, thinker | Chair: glm | Debate rounds: 1 | Final: majority ACCEPT-WITH-CHANGES at 33% | Unanimous: False
