# Finding the fold ab initio: what would have to change, minimally

Question: *how would you change the current method minimally, to consistently find the folded
peptide structure by MD with no prior knowledge?*

This is my own answer. It is unusual in that **the two obvious proposals were tested on the
existing chignolin_7 data and both failed**, which changes the answer.

## 1. The failure is not sampling

The fold was sampled: 33 frames of 611,536 reach Q ≥ 0.80 against 1UAO's own contact map, best
backbone RMSD 0.435 Å, 529 contiguous episodes across 33 of 64 replicas and 11 run segments. Any
change that buys more sampling is solving a problem this campaign does not have.

## 2. Test A — identification by reference-free geometry and kinetics: FAILS

The natural minimal fix is post-hoc: cluster the existing trajectories and rank basins by a
foldedness score that uses no reference. I built exactly that. Frames were binned in a
reference-free space (PC1, PC2 from the ensemble's own PCA, plus CV1), 429 basins with ≥ 500
frames, scored by *internal tightness + kinetic persistence + independent re-entries +
compactness* — every term sequence-agnostic, no native input.

**Result: the most-native basin (44 % native) ranks #215 of 429 — the middle of the list.**

The reason is already in this campaign's matched null: 25 % of compact frames lie within 1.0 Å of
*some* arbitrary compact reference, and the native basin is ~61× sparser than a typical compact
basin. Tightness-and-persistence scores therefore select ordinary compact basins, of which this
ensemble has hundreds. **A reference-free structural/kinetic foldedness score does not find the
fold.**

## 3. Test B — identification by free energy in structure space: FAILS WORSE

The better-posed version asks not "where is the minimum of a 1-D PMF over a degenerate CV" but
"which basin in *structure* space carries the most MBAR weight". Same 429 basins, ranked by
MBAR-weighted free energy (weights joined per frame, §2.1 of the audit):

**The most-native basin ranks #424 of 429. Every basin above 20 % native ranks 388–427.**

The fold is not merely unresolved by the coordinate — in this ensemble it is among the *least*
thermodynamically favoured basins. Quantitatively:

| native definition | MBAR weight fraction | ΔG above the ensemble | Kish ESS | largest single weight |
|---|---|---|---|---|
| backbone RMSD ≤ 1.5 Å | 1.10 × 10⁻³ | **+4.06 kcal/mol** | 215.3 | 1.6 % |
| backbone RMSD ≤ 1.0 Å | 2.36 × 10⁻⁵ | +6.35 kcal/mol | **6.9** | **33 %** |
| Q ≥ 0.80 | 0 (none of the 33 carries a weight) | — | — | — |

This independently reproduces the audit's F = 3.70 kcal/mol at the native d1/d2 cell, by a
different route. It also settles the outstanding ESS gate: at ≤ 1.5 Å the native subset is
usable (ESS 215); at ≤ 1.0 Å it is not (ESS 6.9, one frame carrying a third of the weight).

## 4. Therefore the binding constraint is the force field, not the coordinate

No analysis change can promote a basin that the potential places ~4 kcal/mol uphill. Experimentally
chignolin is at least marginally folded at 300 K; this ensemble puts the folded basin at ~0.1 %
population. That gap is a **force-field/water-model** statement — ff14SB + TIP3P, with a 0.8 nm
cutoff — not a statement about CVs or estimators.

So the honest ordering of minimal changes is the reverse of the intuitive one:

1. **Force field and water model — ff19SB + OPC.** Already this project's decision for future runs;
   it is not a refinement, it is the precondition. Nothing downstream can compensate. *This fixes
   identification, by making the fold actually low-lying.*
2. **Promote a data-derived second CV.** Fit PCA/TICA on epoch-0 data and drive CV2 with it — the
   code already anticipates this (`tica_switch_cv2`, `tica_obs_interval`, the secondary-CV umbrella
   machinery, all currently off). PC1 separates native from non-native at AUROC 0.978 (held out
   0.964/0.980) against CV1's 0.859, and does so reference-free. Sign ambiguity is not a problem
   because umbrella sampling covers the full range in both directions. *This fixes resolution: it
   stops the PMF marginalising the fold together with ~25× more equally compact topologies.*
3. **Keep `traj_interval = 250`** (already changed in v0.8). At 2500 only 11.9 % of samples had
   coordinates, which makes any structure-space analysis impossible on most of the data.
4. **Report a ranked shortlist of basins, not an argmin.** With a correct potential the fold should
   be at or near the top; a shortlist is falsifiable and degrades gracefully. An argmin over a
   degenerate coordinate cannot be right even in principle.

## 5. What I would not change

- **Not CV1.** It is a good collapse coordinate: its windows overlap well (nearest-neighbour
  marginal median 0.965) and it drives compaction efficiently. Its only failure is resolving
  topology at fixed compaction, which is what CV2 is for.
- **Not the λ ladder or the GaMD scheme, first.** The under-coupling is real (38 edges below 0.15)
  but the λ=0 cross-check reproduces the full PMF to 0.289 kcal/mol. It costs free-energy
  precision, not fold identification. Fix the scheduler bug that sent λ=0 top-ups out boosted —
  that is a bug, not a design change.
- **Not the aggregate simulation time.** The fold appeared 529 times. More of the same buys nothing.
- **Nothing that uses a reference**, including choosing the sign of a data-derived CV by comparing
  against a native structure.

## 6. The claim that cannot be made

"Consistently finds the folded structure with no prior knowledge" is not achievable as stated, and
the reason is now measured rather than argued: identification is only as good as the potential's
ranking of basins, and no sequence-only criterion recovered the fold from a correct ensemble here
(Test A) or from correct weights (Test B). What a corrected pipeline can honestly promise is **a
short, ranked list of reproducible, internally coherent basins, one of which is the fold** — with
external validation still required to say which. Any stronger claim is a claim about the force
field, and should be tested as one.


## 7. What the panels said, and where it collides with the tests above

**The big board did not return a verdict.** Five judges ran, two debate rounds ran, but the chair's
synthesis call failed with `HTTP Error 500` and `deepseek-thinking` failed the same way in tier 1.
The board also never converged: majority 0 %, mean position-card cosine 0.512 after two rounds.
There is therefore **no board judgment to quote**, and what follows is me reading the surviving
judges' own text (`native_fold_audit_results/board_method_review/`), not a synthesised verdict.

**Three of the four surviving judges independently prescribed Test B.**

- *glm*: "cluster the existing coordinate frames on contact maps, reweight every cluster to λ = 0
  with the existing umbrella-only MBAR, and rank basins by reweighted population with propagated
  CIs… no new simulation, no new CV."
- *thinker* (the math/physics seat): "cluster the existing frames (invariant pairwise-distance
  features → TICA → MSM macrostates) and rank macrostates by umbrella_only-MBAR-reweighted free
  energy F_a = −kT ln Σ w at λ = 0 with per-basin ESS and file-block bootstrap CIs; no new
  production MD, no CV2."
- *kimi*: the same, with burial/H-bond tie-breaks, "`tica_switch_cv2` stays False", and
  ff14SB/TIP3P explicitly "second-order".

**That is exactly the procedure run as Test B in §3, and on this data it puts the native basin at
rank 424 of 429.** The prescription is right in principle and fails in practice here, for the
reason §4 gives: the potential places the fold ~4 kcal/mol uphill, and a ranking cannot promote
what the energy function demotes. The judges were briefed before Tests A and B existed, so this is
new information against their position rather than a disagreement they declined to address.

*mini* dissented toward a CV upgrade, arguing a reference-free CV is "required to make the native
basin thermodynamically dominant". **That is a physics error**: the free energy of a basin in
structure space is a property of the potential and the ensemble, not of the coordinate used to
project them. A better CV improves *resolution*, never the ranking — which is precisely why Test B,
done in structure space with no projection at all, still fails.

*kimi* also made the point I now think is the most durable thing to come out of this exercise:
a reference-free ranking identifies **the force field's fold** under an unavoidable
Anfinsen/class-level prior, never a prior-free "native". §4's measurement is what that looks like
when the force field's fold and the experimental fold differ.

### Corrections the judges landed on this report

- **CV1 fails necessity as well as sufficiency** (kimi, verified): of the 4,929 frames within 1.5 Å,
  only **2,140 (43.4 %)** lie in the native CV1 window — **56.6 % fall outside it**, spread over
  CV1 0.53–0.94. Earlier sections discussed only purity (3.65 %). Recall is just as bad, and the
  contrast with PC1 is stark: PC1's native window holds **90.9 %** of near-native frames at 6.07 %
  purity. The degeneracy case is stronger than stated, not weaker.
- **The PC1 hold-out was mildly leaky** (kimi): components were refit on the training half, but the
  *superposition reference* came from the all-frames run. The fit is clean, the alignment is not.
- **"27 % native in PC1's top bin" ignores the MBAR weights** (kimi) — correct, and Test B is the
  weighted version of that question.
- **Episode statistics span ~58 state hops** at 1.4 ps exchange spacing (kimi); already not quoted
  as lifetimes, and this is a further reason not to.
- **The native subset's Kish ESS** was missing and is now measured: 215 at ≤ 1.5 Å, **6.9** at
  ≤ 1.0 Å with one frame holding 33 % of the weight (§3).

**Codex** reached the same structural answer as *mini* — bootstrap a torsion-PCA CV2, run 2-D
adaptive umbrella sampling, identify at basin level — and stated §4's conclusion as a hypothetical:
*"If force field ranks misfolded basin below native basin, no analysis can recover physical
truth."* Tests A and B turn that conditional into a measurement for this system. Codex and I differ
on one point: it says do not touch the force field because doing so "does not address CV
degeneracy". True, and not binding — degeneracy is not what stops identification here.


## 8. Demultiplexing the exchange records — and a correction to §1

The full exchange record survives (33 parquet files, 2,067,726 attempts, columns
`step, replica_i, replica_j, window_i, window_j, delta_e, accepted`). Each segment restarts its
step counter at 1,010,400 and writes a frame every 2500 steps, so frame *i* of a segment sits at
step 1,010,400 + (i+1)·2500 — an exact map, verified against `final/baseline` where
(4,406,800 − 1,010,400)/2500 = 1358.56 against 1358 frames actually written. The records report
each replica's current window at every attempt, so the assignment is read directly rather than
replayed from swap moves. **611,530 of 611,536 frames (99.999 %) now carry their thermodynamic
state.**

### 8.1 The ladder mixes well

| quantity | value |
|---|---|
| exchange acceptance | 0.8199 over 2,067,726 attempts |
| rung occupancy (λ = 0 / 0.232 / 0.643 / 1) | 24.2 % / 25.5 % / 26.1 % / 24.3 % |
| λ = 0 ↔ λ = 1 round trips | **42,724**, median 39 per walker series |
| series with zero round trips | 1 of 448 |

This is healthy mixing, and it settles the earlier alarm over "38 weak rung edges": low pairwise
MBAR overlap coexists with excellent traversal, exactly as the bulk-versus-tail explanation
predicted. The ladder is not what is broken.

### 8.2 Folding events — the campaign is short by one to two orders of magnitude

Satoh's sufficiency statistic is the number of transitions from unfolded (Cα RMSD ≥ 4.0 Å) into
native (< 1.0 Å); they report **159** from 180 ns of multicanonical MD and call it "statistically
sufficient to obtain an accurate free-energy landscape". Counting the same way along each
continuous walker series, with hysteresis so a walker must genuinely unfold before it can re-fold:

| definition | events | median native dwell |
|---|---|---|
| Satoh thresholds on Cα RMSD (< 1.0 / ≥ 4.0 Å) | **1** | — |
| backbone res 2–9, min over 18 models (< 1.0 / ≥ 4.0 Å) | **4** | — |
| loose native (< 1.5 / ≥ 4.0 Å) | **25** | 2,209 ps |
| loose, restricted to the λ = 0 rung | 20 | 875 ps |

Aggregate walker time is 611,536 × 8.75 ps = **5.351 µs**, so the event rate is 0.7–4.7 per µs.
Reaching Satoh's 159 at the loose rate would take **~34 µs** of the same sampling.

**This corrects §1 of this document.** I wrote that "sampling is not the blocker" because the fold
appears in 529 contiguous episodes across 33 replicas. That was wrong in the sense that matters:
those episodes are overwhelmingly *re-entries* into the native region from partly-folded
structures, not folding events. A population estimate needs independent folding events, and there
are of order **one**.

Everything else now falls into place. It is why the native subset's Kish ESS is 6.9 at ≤ 1.0 Å
with a single frame carrying 33 % of the weight (§3); why the free-energy ranking puts the native
basin at 424 of 429; and why the d1/d2 surface shows neither the native nor the misfolded well of
Satoh Fig 5B — with ~1 folding event there is simply nothing to build a basin out of.

### 8.3 Why: the boost is too weak to decouple barrier crossing from 300 K

`boost` statistics: mean **2.08 kcal/mol** (3.5 k_BT), sd 1.30, max 13.01. An upper bound on the
rate enhancement is exp(⟨ΔV⟩/k_BT) ≈ 33×, and the realised speedup is smaller because only boost
*at the barrier* counts. Against a folding time of order microseconds at 300 K, that is not enough
to manufacture folding events. Satoh got 159 events from 180 ns because multicanonical MD performs
a random walk up to 700 K, where the barrier is crossed freely, and then reweights — the crossing
rate is decoupled from 300 K entirely.

### 8.4 Consequence for the force-field question

The ff14SB/TIP3P indictment in §4 must be withdrawn as a conclusion. A ~4 kcal/mol native free
energy measured from a run containing about one folding event is not evidence about the potential;
it is a statement about a badly under-converged estimate. The force field remains a candidate
explanation, untested. **The measured defect is barrier crossing, and it is fixable by method
choice** — temperature replica exchange spanning 300–420 K, or multicanonical sampling as Satoh
used, both of which decouple the crossing rate from 300 K. That is where the next effort belongs.

## Files

`native_fold_audit_results/blind_test.py`, `blind_test.txt`, `blind_identification.json`,
`blind_identification_freeenergy.json`, `demux.py`, `events.py`, `demux.npz`, `demux_summary.txt`; panel material in
`native_fold_audit_results/board_method_review/` (tier 1, both debate rounds, agreement matrices —
no chair judgment: the call failed) and `native_fold_audit_results/codex_method_review.log`.
