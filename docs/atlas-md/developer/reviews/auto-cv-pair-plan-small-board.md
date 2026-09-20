# The Board's Judgment

## Majority position

**REJECT.** Two of three members (kimi, thinker) concluded, after independent review and adversarial cross-examination, that executing the plan as written would not deliver a scientifically sound, native-blind, honestly-gated automatic (CV1, CV2) pair for equilibrium MBAR reweighting. The majority position rests on five defects, each of which either stalls execution at the plan's own acceptance tests or — worse — produces a silently wrong or self-deceiving result:

1. **Task 6 — unit double-conversion in the CV2 umbrella (REJECT-grade, silent-wrong-result path).** The energy test sets `ss_k = 40.0` directly via `ctx.setParameter`, bypassing `set_window()`, and expects `0.5·40·4.184·(z2−c2)²`; the shown energy function is `0.5*ss_k*(z2-ss0)^2` with no 4.184 factor, and the plan instructs that the existing `set_window()` path (which multiplies by 4.184) be kept. The test therefore fails by 4.184× against the stated implementation. The dangerous resolution is fix-forward: baking 4.184 into the expression to make the test green while `set_window` also converts yields a simulated stiffness 4.184× the design value; window widths shrink by √4.184 ≈ 2.05, and MBAR `u_nk` reconstructed from the CSV stiffness then mismatches the bias actually applied — silently wrong reweighting to λ = 0, which no tolerance in the test suite would catch. This is exactly the unit trap class the review was charged to find.

2. **Tasks 3/5 vs Task 6/7 — the auto path selects an anchor the runtime cannot force, and the ladder mislabels it.** Task 5's own test asserts the auto anchor picks `radius-of-gyration` (Rg range ~0.55 nm at `k_max = 1200` resolves ~16 windows vs 2 for the r7-like contact fraction). Task 6 explicitly raises `NotImplementedError` for Rg/end-to-end anchors, and no Rg/e2e umbrella force is added anywhere in the plan. Compounding this, Task 7's `write_ladder_windows_2d_csv` hardcodes `primary_cv_mode = "contacts"` regardless of the selected anchor, so an Rg-anchored ladder would either crash production or — if the loader accepts it — place contact-fraction umbrellas at physically unreachable centers (~0.55–1.10 for a CV spanning [0, 0.069]), destroying overlap and ESS only after the swarm budget is spent. The headline feature (`cv1: auto`) is non-functional end-to-end, and the test suite stays green while the real run fails — self-deceiving tests.

3. **Task 4 — the information criterion contradicts its own test and is scientifically inert in the regime that matters.** The implementation (and spec §4.2) uses `min(L(z1), L(z2)) − L(z1, z2)`, but the test `test_incremental_information_is_positive_when_z2_separates_cells_z1_cannot` (cells = (z2 > 0), z1 pure noise) expects `gain > 0.3`; the min formula yields ≈ 0 there — the test fails against the plan's own implementation. Beyond the contradiction, the min formula is an XOR-detector: it credits only strict synergy, giving zero credit to a CV2 that single-handedly determines the discovery cells. Precisely when the anchor is weak (the chignolin fixed-primary case, where contacts resolve 2 windows and L(z1) ≈ H), every component scores ≈ 0, the tie-break installs PC1, and the report presents a content-free selection as principled.

4. **Task 8 — gate/fallback contradiction on the default chignolin branch, and a dishonest fallback.** `pair_gate` returns `ok = False` for `status = "no_deployable_anchor"` even with `fallback = "cv1_only"`, but Task 8's own test expects a 1-D ladder to be written in exactly that case; since the fail path withholds ladder artifacts by filename, the test cannot pass against the implementation. Substantively, the fallback path can run a full-cost campaign on a CV1 that the selector itself just measured as non-deployable — reproducing the phantom-centre failure the spec's §3 commentary explicitly warns against. The spec's requirement that fallback run the *configured* CV1 (not an auto pick) is not implemented, and `cv1: auto` with no deployable anchor leaves the fallback undefined.

5. **Task 5 — CV1 leakage into the discovery-cell criterion.** The discovery cells are constructed from standardized torsion+shape features that include the anchor candidates, so the "incremental structural information" labels are partly defined by CV1 itself. Combined with defect 3, the ranking among CV2 components is contaminated and effectively arbitrary in the weak-anchor regime. The spec acknowledges the criterion is discovery-measure-only and not a thermodynamic claim, but the plan does not gate or label this circularity in the certificate's use for ranking.

The majority acknowledges what is sound: the OLS/WLS residualisation delivers exact `Cov_q(z2, z1) = 0` under the certified weighted measure; the chain-rule term is correctly obtained by OpenMM's automatic differentiation through the `CustomCVForce` expression (the second `CustomBondForce` is a legitimate inner CV, not a duplicate restraint); the native-blind constraint is respected in the dictionary; and the spec's honesty framing (independence as efficiency, not correctness) is scientifically correct. These virtues do not rescue a plan whose acceptance tests contradict its own implementations at Tasks 4, 6, and 8 and whose auto path is undeployable.

## Dissenting positions

One member dissented from the REJECT verdict:

> **Defense / Concession**
>
> I maintain the verdict **ACCEPT‑WITH‑CHANGES** because the plan still contains critical, executable defects: (i) the anchor‑selection logic can pick Rg while the runtime force only supports contacts, leading to a guaranteed runtime failure; (ii) the unit conversion in the CV2 umbrella energy test is missing, causing a systematic energy offset; (iii) the JSON round‑trip test is impossible as written, so the contract for lossless persistence is unsatisfied. These must be fixed before the plan can deliver a scientifically sound, honest pair. The other points raised in the prior review are either minor or already addressed by the specification.
>
> VERDICT: ACCEPT‑WITH‑CHANGES – the plan has serious implementation mismatches (anchor‑force support, unit conversion, and persistence precision) that must be corrected.
> CONFIDENCE: 88

The dissent agrees on the substance of the anchor-force mismatch (defect 2), the unit conversion error (defect 1), and the JSON round-trip impossibility, but judges these correctable within an accept-with-changes frame rather than grounds for rejection. The majority rejects that framing because the defects are not peripheral: they sit in the acceptance criteria and selection semantics themselves (Tasks 4, 6, 8), and the most likely fix-forward for the unit trap produces silently wrong MBAR reweighting rather than a visible failure.

## Confidence

**Aggregate confidence: 83/100.** Derived from the members' final confidences: kimi (REJECT) 80, thinker (REJECT) 90 — majority mean 85 — with mini's dissenting ACCEPT-WITH-CHANGES at 88 pulling the aggregate modestly downward to reflect the split board. The majority's confidence is high on the mechanical contradictions (test-vs-implementation failures at Tasks 4, 6, 8 are verifiable from the plan text alone) and somewhat lower on the claim that fix-forward would necessarily produce the silent 4.184× stiffness error rather than a visible test failure.

## Conditions

The board's REJECT is conditional on resubmission; the following changes are required before the plan can be reconsidered:

1. **Task 6 — establish a single, test-enforced kcal→kJ convention.** Either the energy expression contains the 4.184 conversion (and `set_window` passes kcal unchanged) or `set_window` converts and the test sets `ss_k = 40·4.184`. Add a regression test that would fail under a double conversion, and a test verifying that `u_nk` reconstruction from the CSV stiffness matches the actually applied bias.
2. **Tasks 3/5/6/7 — make the anchor dictionary deployable.** Restrict `cv1: auto` to anchors with implemented runtime forces (contacts only) until Rg/end-to-end umbrella forces and their chain-rule support land, or implement those forces in this plan. Remove the hardcoded `"contacts"` in `write_ladder_windows_2d_csv` and write the true `primary_cv_mode` (with unit-appropriate centers).
3. **Task 4 — fix the incremental-information criterion.** Use `L(z1) − L(z1, z2)` (incremental over the fixed anchor), reconcile the test with the implementation, and add a guard/flag for the degenerate regime where all gains are ≈ 0 rather than silently falling to the lowest-index tie-break.
4. **Task 8 — repair the fallback/gate contract.** `pair_gate` must pass `(no_deployable_anchor, cv1_only)` only when an explicitly configured CV1 exists and is recorded as the fallback; `cv1: auto` with no deployable anchor must refuse. Reconcile `pair_gate` with the 1-D-ladder fallback test, and never emit a ladder from a non-deployable auto pick.
5. **Task 5 — address CV1 leakage in the discovery-cell criterion.** Either construct the cell partition from features excluding the anchor candidates, or explicitly label the criterion as anchor-contaminated and demote it below the nonlinear-R² and resolvability gates in the report.
6. **Task 2 — replace the exact-equality JSON round-trip assertion** (`np.array_equal`) with a tolerance appropriate to decimal serialization, or store binary float64 artifacts, so the persistence contract is testable as written.
7. **Task 6 — validate at load time that the runtime contact definition (pairs, r0, beta, normalization) matches the `PairModel`'s `anchor_definition`**, so a production run cannot silently apply a different anchor than the one the residual model was fitted against.

---

Small board: kimi, mini, thinker | Chair: glm | Debate rounds: 1 | Final: majority REJECT at 67% | Unanimous: False


---
Full transcript: small_board_out/transcript.md
Final report:   small_board_out/final_report.md
