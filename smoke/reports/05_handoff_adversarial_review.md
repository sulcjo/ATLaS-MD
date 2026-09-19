# Adversarial review of `HANDOFF_CV_SELECTION_2026-09-19.md`

Self-review of the handoff, deliberately hostile. Findings graded **CRITICAL** (a stated claim is
wrong), **HIGH** (a claim needs restating), **MEDIUM** (loose or self-serving), **LOW** (presentation).
Every finding is checked against source, not asserted.

---

## R1. CRITICAL — the project is *not* "ab initio" in the sense the report claims

The handoff states, of the GENPEPT seed banks: *"**Ab initio — no native input.**"* That is wrong as
written, and it is the load-bearing claim of the whole project.

`GENPEPT.py:8118` defines `--diversity-bank-preset` with choices `off | broad | chignolin | turbo`,
help text: *"'chignolin' biases toward **hairpin/turn-rich** banks."* The r7 bank was generated with
`diversity_bank_preset=chignolin` (census README, §Dataset). Comparing the two presets
(`GENPEPT.py:1146–1163`):

| bank | `broad` fraction | `chignolin` fraction | dominant weight |
|---|---|---|---|
| β-extended / hairpin_beta | 0.30 (B 0.55) | **0.35 (B 0.60)** | β |
| mixed_turns / turn_rich | 0.25 (T 0.35) | **0.30 (T 0.40)** | turn |
| alpha_compact / compact_alpha_turn | 0.25 (A 0.65) | **0.15 (A 0.50)** | α |
| gly_left / gly_left_turn | 0.10 (L 0.45) | 0.10 (L 0.55, T 0.15) | left-α Gly |

β + turn mass rises 0.55 → 0.65 while α falls 0.25 → 0.15. Chignolin is a β-hairpin. **A preset named
after the target upweights the target's fold class.** No native coordinates enter, but fold-class
prior knowledge does, and "no native input" is not a defensible description of it.

**What this does and does not invalidate.** It does *not* bias the free energies: the preset acts on
seed generation, not on the Hamiltonian, and MBAR reweights from whatever was sampled. It *does*
weaken the discovery claim — finding a hairpin is less notable when the conformer library was
seeded hairpin-rich — and it means the ab initio and validation modes must be separated at the
GENPEPT layer too, exactly as §10 separates them at the CV-scorer layer:

- ab initio production → `diversity_bank_preset: broad`
- methods-validation → `chignolin` permitted, and recorded as such

Two related knobs check out clean and should *not* be lumped in with this:
`contact_bias_strength = 0.15` resolves (`GENPEPT.py:1841–1850`) to `exp(0.15·(ccount − ccount_max))`,
a generic compaction prior with no fold information; and `adaptive_register_threshold.json` is fit
from the library's own N–O distance histogram (rep5_t7: `valley_detected: true`, τ = 5.31 Å over
86,400 distances), not from 1UAO. Note it differs per bank — the verification document records rep4
as `valley_detected: false`, fallback 3.8 Å.

## R2. HIGH — "~34 µs" is the most favourable of three defensible numbers

§5.4 leads its table with **1** folding event under Satoh's exact Cα thresholds, then states a rate
of "0.7–4.7 events per µs → **~34 µs**" for 159 events. The 0.7 corresponds to the *backbone* count
of 4, not to the 1 the table leads with. Over the report's own three definitions:

| definition | events | rate | µs needed for 159 |
|---|---|---|---|
| Satoh-exact Cα | 1 | 0.19 /µs | **851** |
| backbone res 2–9 | 4 | 0.75 /µs | 213 |
| loose (≤ 1.5 Å) | 25 | 4.67 /µs | **34** |

The honest statement is **34–851 µs**, i.e. the shortfall is one to **two and a half** orders of
magnitude, not the single order the "~34 µs" figure implies. The direction of the error favours the
author's own campaign, which is the worst direction for it to run in.

## R3. MEDIUM — the boost mechanism is stated incorrectly

§5.5: *"An upper bound on the rate enhancement is exp(⟨ΔV⟩/k_BT) ≈ 33×, and the realised speedup is
smaller because only boost **at the barrier** counts."*

Two problems. First, exp(⟨ΔV⟩/k_BT) is a heuristic, not an upper bound — there is no derivation in
the report making it one. Second, "only boost at the barrier counts" is the wrong mechanism: for
lower-bound GaMD, ΔV = ½k(E−V)² is *largest in deep wells* and zero once V ≥ E, so the boost works
by filling wells, not by lowering barrier tops. What governs the rate is the **difference** in boost
between the reactant well and the transition region, not the boost at the barrier. The conclusion
(2.08 kcal/mol mean is too small against a microsecond barrier) survives, but the stated reason does
not, and a reader who inherits the wrong mechanism will tune the wrong knob.

## R4. MEDIUM — "the ladder is not what is broken" conflates mixing with overlap

§6 concludes this from 42,724 λ round trips and 0.82 acceptance. Round trips establish **ergodic
mixing across states**; MBAR's free-energy *variance* is governed by pairwise phase-space **overlap**,
which the quality gate measures at 0.065–0.15 against a 0.15 target. Both can be true at once: the
walkers traverse the ladder freely while neighbouring states still share little weight mass. The
sentence as written reads as a clearance of the ladder for free-energy precision, which the data do
not support. It should be scoped to sampling.

## R5. MEDIUM — §10 is softer than the author's own measurement warrants

§10 frames the open decision as *"the scoring function is the whole decision"*. The measurement is
stronger than that. Variance picks PC1, conditional multimodality picks PC4, implied timescale picks
PC1, and **correlation(timescale, AUROC) = −0.033** across six components whose timescales span only
306–473 ps. No unsupervised rule tested recovers the supervised pick.

The defensible statement is: **for this system, ab initio selection of the second CV is currently
unsolved** — not merely "pending a criterion choice". A reader should not be left expecting that
plugging in the η²/reweight scorer will land on PC2; it may well pick PC1 or PC4, and if it does,
that is the ab initio answer and PC2 is not available to an ab initio campaign. The report should
say this before proposing the pluggable scorer, not after.

## R6. LOW — surviving asymmetries

- §5.6 correctly calls the estimator spread "a disclosure, not corroboration", but §5.2's table still
  presents four minima as though jointly informative; only `umbrella_only` is valid.
- §12 step 1 (420 K validation) is the highest-value action and is listed first, but steps 2–3
  (adding Asp3O–Gly7N, the restraint-satisfaction metric) are near-free and gate interpretation of
  everything downstream. They belong before, not after, any new production.
- The report never states its own provenance limitation: every number in it derives from one
  campaign on one peptide with one force field. It reads as more general than it is.

## What survives unchanged

The central diagnosis holds and is, if anything, understated by R2: the campaign is short of
converged 300 K populations by one to two and a half orders of magnitude in folding events, the
cause is a calibration defect that left the boost at 3.5 k_BT, and the ff14SB/TIP3P indictment was
correctly withdrawn because a ~4 kcal/mol native free energy measured from ~1 folding event is not
evidence about a potential. The demultiplexing, the frame↔step map, the CV degeneracy measurements
(3.65 % sufficiency, 56.6 % necessity failure) and the CV ranking are all reproducible from the
shipped scripts and were checked against source here.

## R7. CRITICAL (found by the board, confirmed here) — the fold evidence is concentrated in the wrong-ensemble segments

The board asked why the 33 Q ≥ 0.80 frames carry zero MBAR weight. Checked: **all 33 lie in
`epoch_001/topup_002_3378000`**, one of the three λ=0-only top-ups that ran fully boosted (D7).
Their rows are dropped as wrong-ensemble, correctly. Quantified:

| population | in the 3 wrong-ensemble segments | share |
|---|---|---|
| all frames | 99,264 / 611,536 | 16.2 % |
| RMSD ≤ 1.5 Å | 1,645 / 4,929 | 33.4 % |
| RMSD ≤ 1.0 Å | 415 / 481 | **86.3 %** |
| Q ≥ 0.80 | 33 / 33 | **100 %** |

The correctly-configured 83.8 % of the campaign produced 66 frames ≤ 1.0 Å and **zero** at Q ≥ 0.80.
The handoff's structural claim survives (real ff14SB configurations under a boost) but must carry
this caveat. It also cuts the other way: those segments had a real boost where genuine λ=0 had
exactly 0.000, and that is where the folding happened — an accidental controlled experiment
supporting the chignolin_8 bet.

## R8. HIGH (board) — the boost argument was formally wrong, not merely loose

Sharper than R3: exp(β⟨ΔV⟩) is by **Jensen** a *lower* bound on ⟨exp(βΔV)⟩, so calling it an upper
bound inverts the inequality. And the decisive structural point is that E = V_max was frozen from
35 ps of basin fluctuation, so the transition state very likely sits *above* E, giving
**ΔV(TS) ≈ 0 and barrier enhancement ≈ 1×**.

## R9. MEDIUM (board) — the report was not self-standing

Missing and now added: commit hash (`24acc31`, v0.8.1, 4 files dirty, unpushed), conda environment,
launch and resume commands, a D8 workaround (scipy ≥ 1.13 removed `scipy.linalg.tril`), the target
force field named explicitly, and the §10 first action written as a runnable procedure against
named files. Also: the memories cited in §13 are **not readable** by the receiving agents — all
load-bearing content is now inlined.

## R10. LOW (board, thinker dissent) — the d1/d2 CI "red flag" is a rounding artifact

d1 point 7.206 vs CI lower 7.21, d2 8.377 vs 8.38 — differences of 0.004 and 0.003, i.e. two-decimal
rounding. Not a degenerate resample. The genuine concern there is D4 forward-fill inflating effective
sample size, which the handoff already states.

## Required changes

1. Replace "Ab initio — no native input" for GENPEPT with an accurate statement of the fold-class
   prior, and add the `broad` vs `chignolin` preset split to §10's mode separation. **(R1)**
2. Quote the folding-event shortfall as **34–851 µs**, and make the rate range consistent with the
   table it follows. **(R2)**
3. Correct the GaMD mechanism sentence; drop "upper bound". **(R3)**
4. Scope the ladder conclusion to sampling, not free-energy precision. **(R4)**
5. State plainly in §10 that ab initio CV2 selection is unsolved for this system, before proposing
   the scorer split. **(R5)**
6. Add a provenance caveat: one campaign, one peptide, one force field. **(R6)**
