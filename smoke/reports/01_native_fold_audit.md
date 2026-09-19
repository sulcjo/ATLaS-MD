# Was the chignolin native fold found? — audit of the ATLAS-MD v0.7 `chignolin_7` production run

**Run** `RUNS/chignolin_7/adaptive_production` · **Analysis** `pmf_analysis/` (regenerated 2026-09-18 16:41–16:56)
**Audit date** 2026-09-18 · **Reference** 1UAO (chignolin, GYDPETGTWG, 18 NMR models)
**Review** adversarial 5-model board (ACCEPT-WITH-CHANGES, unanimous, confidence 84) and an
independent codex review; all conditions from both are applied below and listed in §7.

## Summary

The two questions have **opposite answers**, and that is the result.

1. **Have the correct thermodynamic minima been found?** **No — with the scope stated precisely:
   no minimum of the run's 1-D PMF products (CV1, Rg, d1, d2, from the one valid estimator) lies
   within the native range.** This is *not* the claim that the native fold is not a free-energy
   minimum of the system. §4 shows why a PMF along a degenerate coordinate can hide a native basin
   wherever the true minimum lies, and the native-region free energy remains unresolved and
   ESS-gated (§7). Of the six published PMF minima,
   the four computed on valid products — CV1, Rg, and both hairpin H-bond distances — all lie
   outside the range spanned by the 18 native NMR models, by 4.4 to 11.5 native standard
   deviations. The two PCA minima are *not counted*: that surface is built on 2.02 % of the
   samples (§5.1). The PCA has since been **recomputed on all 611,536 frames** (§6) — the basis
   barely moves, but a free-energy minimum still cannot be quoted for it without per-frame weights
   (§6.3). A caveat now attaches to the other four as well: Rg and d1/d2 are forward-filled across
   sample rows, so their support is 611,536 frames rather than the 5,110,208 reported (§5.4). That
   lowers their effective sample size; it does not move the minima, which remain 4.4–11.5 native
   sd from the fold.
2. **Are correctly folded structures present in the trajectories at all?** **Yes.** The decisive
   evidence is not the RMSD headcount — a matched null shows that for a 10-mer, landing within
   1.0 Å of one of any 18 compact references happens 25 % of the time (§3.1). It is the
   **conjunction of reference-specific criteria**: 33 frames of 611,536 reach a native contact
   fraction Q ≥ 0.80 against 1UAO's own 65 native contacts (5.4 × 10⁻⁵), and *all 33*
   independently satisfy backbone RMSD ≤ 1.0 Å and a formed Tyr2/Trp9 cluster ≤ 6.0 Å. These
   three are related, not independent — Q and backbone RMSD both measure similarity to the same
   reference, and the Tyr2/Trp9 cluster is a subset of what Q counts. The load-bearing point is
   that **Q is reference-specific**: it is defined by 1UAO's own contact map, which a merely
   compact structure has no reason to reproduce. Best structures, reported separately because they
   are **different frames**:

   | | backbone RMSD | Q | Tyr2/Trp9 |
   |---|---|---|---|
   | tightest backbone | **0.435 Å** | 0.752 | 4.81 Å |
   | highest native contact fraction | 0.485 Å | **0.932** | 4.93 Å |
   | native (18 models) | — (internal spread 0.77 mean / 1.37 max) | 0.810–0.963 | 5.10–5.98 Å |

   Neither frame is inside *every* native range — both have a Tyr2/Trp9 cluster slightly tighter
   than any NMR model — so the earlier phrase "native on every metric simultaneously" is withdrawn.
   What holds is that each is inside 1UAO's own backbone spread while independently reproducing
   most of its native contact map.

The fold was **sampled but not scored**. This supersedes the earlier verdict recorded on
2026-09-18 ("did not identify the folded state and could not have"), which was right about the
thermodynamics and wrong about the structural presence — for the reason given in §4.

A third result came out of recomputing the PCA properly (§6): **PC1 separates the native fold far
better than the biasing CV does** — AUROC 0.978 vs 0.859 for CV1, holding under contact-based
definitions too, with its top bin 27 % native against CV1's best of 3.65 %. PC1 is reference-free,
so unlike Q it is usable as a biasing coordinate. That is the most actionable finding here.

Two qualifications that do not change either answer. Among compact structures, an arbitrary set of
18 decoys drawn from the run is matched **61× more often** than the true native set (§3.1) — the
fold is present, not prevalent. And its occupancy is
strongly segment-dependent, 0.00–32.67 % of the native CV1 window (§4.1), so it is under-converged
as well as under-resolved. **No population or free energy is quoted for it here, and none should
be.**

## 1. Coverage

| quantity | value |
|---|---|
| trajectory files read | 1472 / 1472 |
| files that failed to load | **0** |
| coordinate frames analysed | **611,536** |
| frame spacing | 2500 steps × 3.5 fs = **8.75 ps** |
| exchange interval | 400 steps = **1.4 ps** (≈6 exchange attempts between saved frames) |
| MBAR samples in the production PMF | 5,124,128 (2,947,296 usable in the union matrix) |

The structural search below rests on the whole coordinate archive. The *PCA* claim does not —
that product is a 2 % cache (§5.1), which is why §0 counts four minima and not six.


### 1.1 Simulation parameters, and the confound they expose

| | |
|---|---|
| protein force field | **ff14SB** |
| water model | **TIP3P**, 0.15 M ionic strength |
| ensemble | NPT, 300 K, 1 bar; Langevin friction 1.0 ps⁻¹ |
| integration | 3.5 fs with HMR (H = 3.024 amu) |
| box / cutoff | dodecahedron, 1.0 nm padding, 0.8 nm nonbonded cutoff |
| construction | built from sequence GYDPETGTWG; `input_pdb = None` |

These belong in the results, not a footnote, because they open a third explanation for the central
finding. "The fold is sampled but not scored" is attributed in §4 to a degenerate coordinate and in
§5/§7 to a marginal ladder. It may also be, in part, a **force-field** statement: ff14SB with TIP3P
is an older protein/water combination for folding equilibria, and this project has already decided
to move to ff19SB + OPC for subsequent runs. (A 0.8 nm cutoff is also short for an Amber-family
force field, usually run at 0.9–1.0 nm.) Nothing here shows the fold *ought* to be the minimum —
the point is that three candidate causes exist, the audit tests two, and the third is untested.
Separating them requires a native-referenced free energy under this force field, and ideally the
same under ff19SB/OPC.

### 1.2 Reconciliation of the sample counts

Five different totals appear across the run's products:

| count | meaning |
|---|---|
| 5,124,128 | total MBAR samples in `pmf_analysis` (= Σ n_k over the 64 states) |
| 5,110,208 | the finite subset used for the FES products (13,920 dropped) |
| 4,583,936 | samples carrying a finite GaMD boost — exactly 5,124,128 − 540,192, the epoch_000 count |
| 2,947,296 | rows in the separate `adaptive_union_mbar.npz` product |
| 2,387,808 | rows the union *analysis* used — exactly 2,947,296 − 559,488, that file's own `gamd_ladder_samples_without_raw_energies` |
| **611,536** | **coordinate frames — 11.9 % of the samples, and the support of every structural result here** |

The 4,583,936 identity is arithmetically exact but undocumented and should be confirmed rather than
inferred.

### 1.3 Two things this audit still owes

- **No confidence interval exists on any PMF minimum.** Every deviation in §2 is a distance in
  native-ensemble widths, not a significance. Block bootstrapping over blocks longer than both the
  indicator autocorrelation time and several exchange intervals is required before any of them is
  quoted as resolved.
- **No effective sample size is reported for the native-like subset.** The Kish ESS and weight
  concentration for the Q ≥ 0.80 / RMSD ≤ 1.0 Å frames are the gate in §7, and they cannot be
  computed until frames are joined to their MBAR weights (§6.3). Until then no population or free
  energy is claimed for the fold — which is why none appears anywhere in this report.

## 2. Where the native structure sits, on the run's own axes

Every coordinate was recomputed with the run's own definitions. The contact CV was validated
against the run's stored `cv_A` samples: the sampled range matches exactly (0.021 / 0.986) and the
percentiles agree to **three decimals up to the 75th and to two decimals in the tail**
(1/25/50/75/99 %: mine 0.0966 / 0.2904 / 0.4954 / 0.7048 / 0.9043, run 0.0965 / 0.2896 / 0.4957 /
0.7053 / **0.8907** — a 0.0136 difference at the 99th). The tail difference is expected and not a
CV mismatch: the two populations are not the same set. My values come from the 611,536 coordinate
frames (every 10th sample), the run's from all 5.12M samples, so the extreme-compaction tail is
weighted differently.

| coordinate | native (18 models) | native range | **PMF minimum** | deviation |
|---|---|---|---|---|
| CV1 contact fraction (biasing CV) | 0.863 ± 0.033 | 0.82–0.93 | **0.487** | −11.5 sd |
| Rg, heavy atoms (Å) | 5.487 ± 0.165 | 5.14–5.66 | **6.594** | +6.7 sd |
| d1 Asp3N–Thr8O (Å) | 3.569 ± 0.411 | 2.84–4.22 | **7.206** | +8.8 sd |
| d2 Asp3N–Gly7O (Å) | 6.731 ± 0.375 | 6.17–7.56 | **8.377** | +4.4 sd |

Rg, d1 and d2 were flagged provisional because §5.4 shows them forward-filled across ~10 sample
rows each. **They have now been recomputed from matched rows and all three minima are confirmed
unchanged** (§2.1). The defensible count is therefore **four minima outside the native range**,
with two PCA minima still not counted at all.

### 2.1 Recompute from matched rows — the forward-fill does not move any minimum

Block-start rows (the first row of each constant-observable block, one per coordinate frame) were
extracted from `rg_samples_with_weights.csv`: **511,888 rows = 10.02 % of 5,110,208**, confirming
the 10× fill exactly. The 611,536 − 511,888 = 99,648 frames without a row match the 992,896
wrong-ensemble samples ÷ 10 that MBAR discards (§5.1, §A4) to within 0.4 %. Restricting to these
rows is a valid importance-sampling subset because the selection is by time index, independent of
configuration.

| coordinate | published (forward-filled) | **matched rows only** | block-bootstrap 95 % CI | native |
|---|---|---|---|---|
| Rg (Å) | 6.594 | **6.572** | [5.98, 6.97] | 5.14–5.66 — **outside** |
| d1 (Å) | 7.206 | **7.206** | [7.21, 18.12] | 2.84–4.22 — **outside** |
| d2 (Å) | 8.377 | **8.377** | [8.38, 14.83] | 6.17–7.56 — **outside** |

Rg lands in the same bin as the forward-filled surface (max |ΔF| 0.300 kcal/mol, in a bin holding
36 frames; over well-populated bins the differences are ≤0.02). d1/d2, recomputed on the
*published* 30 × 30 grid, reproduce the published minimum bin exactly, with median |ΔF| = 0.021 and
max 1.251 kcal/mol over the 406 well-populated shared bins. At the native cell (d1 ≈ 3.57,
d2 ≈ 6.73) the matched surface gives F = 3.70 kcal/mol against the published 4.09, on 12 frames.

These are also the **first confidence intervals reported for any minimum in this campaign**
(bootstrap over the 64 replicas). All three exclude the native range. They are wide and one-sided
upward — resampling moves the minimum to larger d1/d2 but never below the reported value —
which is what a broad, shallow basin looks like.

**Two honest caveats.** First, the weight attached to each frame comes from the block-start row,
and the CV checksum shows that row is *not* the frame's own: |cv_frame − cv_row| has median 0.0235
against a within-window CV1 spread of ~0.035, so the paired row is a neighbouring configuration
within ~8.75 ps. Rg identifies the frame unambiguously (the two sources agree to 5 decimals with
identical min and max), so this is a weight-pairing uncertainty, not a mis-identification; repeating
the d1/d2 surface with uniform weights gives the same minimum bin, which bounds its influence.
Second, an earlier version of this recompute used my own coarser bins and reported the d1/d2
minimum at (10.75, 9.25) Å; that was a binning artifact and is withdrawn — on the published grid
the minimum is unmoved. **CV1 still has no confidence interval**; it needs none of this correction,
being recorded per row, but the interval is still owed.
| *PCA1 (Å)* | *10.285 ± 0.704* | *8.75–11.44* | *1.439* | *−12.6 sd — **not counted**, 2 % surface* |
| *PCA2 (Å)* | *−1.162 ± 0.409* | *−1.91 to −0.23* | *−5.289* | *−10.1 sd — **not counted**, 2 % surface* |

Four valid minima, all outside. The answer to question 1 is *no* on every product that can
currently be trusted.

**Two disclosures this table needs.**

*First, the deviations are in units of the native ensemble's conformational width, not of any
statistical uncertainty.* No confidence interval is available for any of these minima. "−11.5 sd"
means "11.5 times the spread of the 18 NMR models away", not a significance level.

*Second, these are all `umbrella_only` values, and the estimators disagree strongly:*

| estimator | CV1 minimum | Rg minimum (Å) |
|---|---|---|
| **umbrella_only** (selected, and quoted above) | **0.4873** | **6.594** |
| gamd_exponential | 0.7445 | 5.823 |
| gamd_cumulant2 | 0.1657 | 7.365 |
| gamd_cumulant3 | 0.1014 | 7.751 |
| *native (18 models)* | *0.82–0.93* | *5.14–5.66* |

The CV1 minimum ranges over most of the coordinate and the Rg minimum over ~2 Å; under
`gamd_exponential` the Rg minimum lands 0.16 Å above the native maximum rather than +6.7 widths
from it. This table is a **disclosure, not a corroboration**: three of the four estimators are
invalid (below), so the fact that none of them lands inside the native window is a consensus of
noise and carries no weight. The conclusion rests on `umbrella_only` alone, supported by the λ=0
cross-check.

`umbrella_only` is nevertheless the right choice here, for reasons the run's own diagnostics
settle: in this λ-ladder construction the GaMD boost enters the MBAR reduced potential directly, so
MBAR is exact and no reweighting correction applies (`pmf_summary.json`: *"λ ladder: boost is
inside u_nk, MBAR is exact; no cumulant"*). The alternatives are independently disqualified —
`gamd_exponential` has an effective sample size of **51** out of 4,583,936 (1.1 × 10⁻⁵), and
`anharmonicity_score = 0.799` with excess kurtosis 0.905 puts both cumulant expansions well outside
their validity range.

**The run did not pass its own quality gate.** `adaptive_quality_gate.md` reads
`Status: needs_more_sampling` with 38 rung edges below `min_rung_overlap = 0.15`, and the ladder
diagnostic grades overall RESULT HEALTH as FAIL. The λ=0 rung — the unboosted ensemble these PMFs
are reported *for* — retains only **34.4 %** of its intended samples, because three λ=0-only
top-ups ran fully boosted; correspondingly 14.2 % of the reduced bias matrix is non-finite and
those rows are (correctly) dropped. Any free energy quoted from this campaign inherits those
caveats.

What keeps the minima usable despite that is a direct test of the affected mapping: rebuilding the
PMF from the 519,898 λ=0 samples alone reproduces the full-MBAR result to **0.289 kcal/mol**
against a 0.5 tolerance (`pmf_ladder_crosscheck`, status pass). The structural conclusions in §3
and §6 do not depend on any of this, since they use no free energies at all.

Auxiliary native values used below: Tyr2/Trp9 ring-centroid distance 5.47 ± 0.29 Å, and native
contact fraction Q = 0.897 ± 0.042, where the 65 native contacts are heavy-atom pairs |i−j| ≥ 3
within 4.5 Å in at least half of the 18 models.

## 3. The fold is present

Thresholds were fixed before counting, and counts are given at several so the answer cannot be
tuned by the choice.

| criterion | frames | fraction |
|---|---|---|
| backbone RMSD ≤ 1.0 Å (inside the NMR ensemble's own spread) | 481 | 7.9 × 10⁻⁴ |
| backbone RMSD ≤ 1.5 Å | 4,929 | 8.1 × 10⁻³ |
| backbone RMSD ≤ 2.0 Å | 32,376 | 5.3 × 10⁻² |
| **Q ≥ 0.80** | **33** | **5.4 × 10⁻⁵** |
| Q ≥ 0.70 | 173 | 2.8 × 10⁻⁴ |
| RMSD ≤ 1.0 Å **and** Q ≥ 0.80 **and** Tyr2/Trp9 ≤ 6.0 Å | 33 | 5.4 × 10⁻⁵ |

RMSD is over the 32 backbone N/CA/C/O atoms of residues 2–9 (the Gly termini fray in every
chignolin ensemble), taken as the minimum over all 18 NMR models.

### 3.1 Is it chance? A matched null, and which criterion carries the evidence

**Min-over-18 inflates modestly, and the fixed-reference counts are reported alongside.**

| threshold | min over 18 models | vs model 1 only | inflation |
|---|---|---|---|
| ≤ 1.0 Å | 481 | 293 | 1.64× |
| ≤ 1.5 Å | 4,929 | 1,096 | 4.50× |
| ≤ 2.0 Å | 32,376 | 9,810 | 3.30× |

293 frames are within 1.0 Å of model 1 specifically, and the tightest frame is 0.54 Å from model 1.

**A *matched* null shows the RMSD count alone is weak evidence.** Because the native criterion
takes a minimum over 18 references, the null must too — and both rates must be measured on the
*same* population. One test set of 496 compact frames (CV1 in the native window, one frame per
distinct trajectory file) is scored twice with identical min-over-18 structure; only the reference
set differs. Decoy references are drawn from the run itself, 18 at a time, with their own files
held out of the test set, over 40 trials:

| reference set (same 496-frame test set) | P(≤ 1.0 Å) | P(≤ 1.5 Å) |
|---|---|---|
| 18 compact decoys from this run | **0.247 ± 0.047** | 0.520 |
| 18 native NMR models (1UAO) | **0.0040** | 0.0222 |
| **matched ratio decoy / native** | **61×** | **23.5×** |

For a 10-mer, landing within 1.0 Å of *one of any* 18 compact structures happens a quarter of the
time, so **"481 frames within 1.0 Å" is not by itself proof that the native fold was found**. The
matched comparison says the native basin is ~61× sparser than a typical compact basin. **That
figure is far less precise than two digits suggest:** it rests on ~2 native events in the 496-frame
test set, giving a Poisson 95 % interval on the count of [0.24, 7.20] and hence a ratio interval of
roughly **[17×, 516×]**. The direction is solid; the magnitude is an order-of-magnitude statement.

Two earlier versions of this null were wrong and are superseded: a one-reference null (suggesting
~26×) that was not matched to the min-over-18 criterion, and a 324× figure that divided a
compact-conditioned null probability by an observed rate computed over the whole archive — two
different denominators, since ~90 % of the archive is not compact. For reference, the
compact-conditioned observed rate is 6.1 × 10⁻³ (359 of 58,670) against 7.9 × 10⁻⁴ over the whole
archive; these must not be mixed.

An independent empirical decoy check agrees that the tight end is not generic: the lowest RMSD
reachable in the low-CV1 bins is 2.72 Å ([0.00,0.20)) and 1.98 Å ([0.20,0.40)), far above 0.43 Å.

**Q is the statistic that carries the evidence.** Unlike RMSD, Q is defined by 1UAO's own contacts,
so an accidentally compact structure cannot reproduce it. Q tracks RMSD tightness as a genuine
native basin must:

| population | frames | median Q | p99 | max |
|---|---|---|---|---|
| all frames | 611,536 | 0.042 | 0.326 | 0.932 |
| compact, CV1 ∈ [0.82, 0.93) | 58,670 | 0.174 | 0.500 | 0.932 |
| backbone RMSD ≤ 1.5 Å | 4,929 | 0.294 | 0.779 | 0.932 |
| backbone RMSD ≤ 1.0 Å | 481 | **0.660** | 0.857 | 0.932 |

Only 33 frames reach Q ≥ 0.80 and all 33 independently satisfy the RMSD and cluster criteria.

**Q needs its own matched null, and it changes the emphasis.** The decoy test above was run for
RMSD only. Repeating it for Q — decoy reference sets of 18 structures whose contact maps are
defined by exactly the same rule, on the same compact test population of 7,708 frames — requires
one correction: a first attempt drew 18 *unrelated* compact frames per decoy set, whose consensus
map held only ~15 pairs against 1UAO's 65, making Q trivially satisfiable. 1UAO is a *tight*
ensemble (model-model backbone RMSD 1.37 Å max), so each decoy set was rebuilt as a cluster with
matched tightness (median internal max-RMSD 1.26 Å, median 48 contacts):

| reference set (same 7,708-frame test population) | P(Q ≥ 0.80) | P(Q ≥ 0.70) |
|---|---|---|
| 18 tight compact decoys drawn from the run | **0.0346 ± 0.0276** | 0.0601 |
| 18 native NMR models (1UAO) | **0.0000** (0 / 7,708) | 0.0013 |

So Q ≥ 0.80 is *not* intrinsically hard to reach — against an arbitrary tight reference a compact
frame reaches it 3.5 % of the time. The mechanism is that a decoy reference is drawn from a densely
populated region of this ensemble, whereas 1UAO sits in a sparse one. The null therefore calibrates
**rarity**, consistent with the RMSD result, and it does not undercut conclusion (b): the 33 frames
are genuinely inside 1UAO's basin, which is what "the fold was sampled" means. What it does retire
is any suggestion that Q alone makes 1UAO special. The evidence remains the **conjunction against
the experimental reference specifically** — Q, backbone RMSD and the Tyr2/Trp9 cluster agreeing at
once on the same 33 frames.

### 3.2 What the recurrence statistics may and may not be called

Replica exchange attempts every **1.4 ps** while frames are saved every **8.75 ps**, so roughly six
exchange attempts occur between consecutive saved frames and **a replica index is not a physical
walker**. The exchange permutation record was not reconstructed here. Consequently:

- "529 independent visits" is withdrawn. The defensible statement is **529 file-local contiguous
  native-like episodes below 1.5 Å, spanning 33 of 64 replicas, 67 of 1472 files and 11 run
  segments; their statistical independence is unestablished.**
- The longest run of consecutive frames below 1.5 Å is 551 frames. At 8.75 ps spacing that is
  4.82 ns **of residence in a replica index, not a molecular dwell time**, and it must not be read
  as a folded-state lifetime. (An earlier version of this report gave 2.755 ns, from an incorrect
  5 ps frame spacing; both the number and its interpretation were wrong.)

What survives is qualitative and sufficient for §3: the fold recurs across 33 replicas and 11
segments, so a single isolated excursion is excluded. Any *rate* or *population* from these counts
would require block bootstrapping over blocks longer than both the indicator autocorrelation time
and several exchange intervals.

## 4. Why the PMF misses what the trajectories contain

The biasing coordinate is **degenerate**, not blind. CV1 reaches native compaction — 58,670 frames
lie inside the native CV1 window and the CV spans 0.021–0.986 — but at fixed CV1 the structure is
not determined:

| CV1 bin | frames | RMSD min | median | max | % < 1.5 Å |
|---|---|---|---|---|---|
| [0.00, 0.20) | 81,979 | 2.72 | 5.02 | 6.45 | 0.00 |
| [0.20, 0.40) | 151,705 | 1.98 | 4.22 | 6.10 | 0.00 |
| [0.40, 0.60) | 148,703 | 0.87 | 3.69 | 5.56 | 0.06 |
| [0.60, 0.75) | 113,198 | 0.65 | 2.68 | 4.93 | 1.22 |
| [0.75, 0.82) | 56,208 | 0.47 | 2.61 | 4.55 | 2.34 |
| **[0.82, 0.93)** | **58,670** | **0.43** | **2.56** | **4.27** | **3.65** |
| [0.93, 1.00) | 1,073 | 1.25 | 2.44 | 3.85 | 1.03 |
| **total** | **611,536** | | | | |

In the native CV1 window the backbone RMSD still spans 0.43–4.27 Å and only 3.65 % of frames are
native. Rg behaves the same way (native window 5.14–5.66 Å: 61,521 frames, RMSD 0.43–4.53 Å,
3.16 % native). A PMF along either coordinate therefore averages the fold together with a ~25-fold
excess of equally compact non-native topologies, and the minimum lands where that mixture is
densest. This is the reference-free contact CV behaving as `gareus/cv_discovery.py:235` warns
("a high contact fraction can mean many different collapsed topologies").

### 4.1 Degeneracy and under-convergence are both present

Degeneracy and under-sampling can produce the same aggregate table, so the native CV1 window was
stratified by run segment.

| run segment | frames in native CV1 window | % < 1.5 Å | RMSD p5 | median | p95 |
|---|---|---|---|---|---|
| epoch_000 | 6,073 | 0.48 | 1.66 | 2.23 | 3.20 |
| epoch_001/baseline | 3,321 | 0.00 | 2.23 | 2.63 | 3.70 |
| epoch_001/topup_001 | 1,965 | 0.00 | 2.37 | 2.71 | 3.75 |
| epoch_001/topup_002 | 2,507 | **32.67** | 0.59 | 1.70 | 2.72 |
| epoch_001/topup_003 | 6,396 | 4.86 | 1.51 | 2.65 | 3.71 |
| epoch_002/baseline | 2,951 | 1.36 | 1.68 | 2.36 | 3.14 |
| epoch_002/topup_001 | 1,779 | 0.00 | 2.18 | 2.54 | 3.64 |
| epoch_002/topup_002 | 1,938 | 0.00 | 1.93 | 2.38 | 3.15 |
| epoch_002/topup_003 | 5,604 | 0.05 | 2.11 | 2.65 | 3.10 |
| final/baseline | 7,239 | 0.04 | 1.96 | 2.65 | 3.62 |
| final/topup_001 | 4,416 | **19.95** | 1.32 | 2.40 | 3.81 |
| final/topup_002 | 5,717 | 0.02 | 1.84 | 2.52 | 3.14 |
| final/topup_003 | 8,764 | 0.60 | 1.95 | 2.71 | 3.81 |

**Degeneracy is established.** All 13 segments that sample the native CV1 window contain
predominantly non-native topologies there, with medians confined to 2.23–2.71 Å and p95 of
3.1–3.8 Å. Structures far from native at native CV1 are what this coordinate contains everywhere,
not a fluctuation of one segment.

**The native sub-population is additionally under-converged.** Its share of that window ranges from
0.00 % in five segments to 32.67 % in one. An earlier reading of this table ("coexistence is stable
across segments") was too strong and is withdrawn: the *coexistence* is universal, the *native
fraction* is not.

**Scope of the causal claim.** Degeneracy explains why the CV1 and Rg minima are non-native, and it
implies that better convergence alone cannot recover the native minimum, since the PMF would still
marginalize the fold together with its CV1-degenerate neighbours. It does **not** by itself exclude
other contributors to the numbers — force-field weighting of the folded state, or GaMD/MBAR
reweighting variance — and the PCA discrepancy is a cache defect rather than a property of any
coordinate. The earlier flat statement "the blocker is the coordinate, not the sampling" is
therefore narrowed to: *the coordinate is a sufficient explanation for the CV1/Rg result and is not
removable by more sampling; other contributions are untested.*

## 5. Defects found in the analysis products

**5.1 The PCA free-energy surface is built on 2 % of the data.** `pca_scores.npz` holds 5,124,128
score slots of which only **103,664 are finite (2.02 %)** — the pre-fix throttled set. Every other
surface uses 5,110,208 samples. The cache is reused because the guard in `analyze_gareus_mbar.py`
(~line 1993) accepts it when `p1.shape == d.cv.shape`, and the shape matches although 98 % of the
entries are NaN. **The guard should test the finite count, not the shape.** Note what a regeneration can and
cannot buy: only 611,536 of the 5,124,128 sample slots have coordinates at all, so a correct
recompute reaches **11.9 %**, not parity with the 5.11M-sample surfaces — and by §5.4 those other
surfaces do not have 5.11M independent support either. §6 recomputes the projection on all frames;
neither PCA *minimum* is quoted, which is why §0 counts four minima.

**5.2 The earlier "native hairpin" filter excludes the native structure.** The criterion
`d1 < 4.0 Å and d2 < 6.0 Å` is failed by 1UAO itself: native d2 is **6.17–7.56 Å** across all 18
models, never below 6.0. Any population derived from it is a lower bound on a mis-specified set,
and the earlier "~0.02 % native" figure should not be re-quoted. (It was not misleading in the
other direction — the 350 frames it selects have median backbone RMSD 0.68 Å.)

**5.4 Trajectory-derived observables are forward-filled across sample rows.** In
`rg_samples_with_weights.csv`, `cv_A` changes at every 250-step row while `rg_A` repeats in blocks
of ~10 consecutive rows — the coordinate-frame interval (2500 steps). Rg is therefore computed once
per frame and replicated across the ~10 sample rows that follow, which is why `rg_summary.json`
reports `n_samples = 5,110,208` when only **611,536 distinct Rg values can exist**. The same
applies to the d1/d2 surface, which reports the identical 5,110,208 from the same
`trajectory_reconstruction` path. **Recomputing all three from matched rows (§2.1) leaves every
minimum unchanged**, so the defect is a sample-count and effective-support problem rather than a
bias in these particular results. Two consequences: the apparent sample size of those PMFs is
inflated ~8.4×, and each frame's Rg is paired with the bias and boost of ~10 *different*
configurations (the rows differ in `cv_A`, so they are not repeats of the same structure). The Rg
and d1/d2 minima in §2 should be treated as resting on 611,536-frame support, not 5.11M, and
recomputed from matched rows only before any of them is quoted as converged.

**5.5 The per-sample weight file cannot be joined back to coordinates.**
`rg_samples_with_weights.csv` carries `(step, replica, window, cv_A, rg_A, weights)` but **no
segment or source column**, and production steps repeat across segments. `(step, replica)` is
therefore not a unique key in that file, which blocks exactly the reweighting the next step needs.
`adaptive_union_mbar.samples.csv` does carry `source`/`source_dir` and should be the basis for any
observable-reweighting work.

**5.3 Dimensionless CV1 is labelled as ångström.** `pmf_summary.json` reports the contact fraction
under `cv_min_A`, `cv_max_A`, `pmf_minimum_cv_A` with the axis label `CV distance (A)`. Read cold,
`pmf_minimum_cv_A = 0.487` looks like 0.49 Å. Unchanged from the previous audit.

## 6. The PCA, recomputed on all 611,536 frames

The production surface was built from a cache holding 103,664 finite scores (2.02 %), fitted on
52,048 frames. The basis was refitted here on **all 611,536 frames**, every frame projected, and
the superposition taken to the iteratively converged ensemble mean rather than an arbitrary first
frame (`native_fold_audit_results/pca_all.py`).

**The basis was essentially right; the coverage was not.** The recomputed components are nearly
collinear with the cached ones (|cos| PC1·PC1′ = 0.996, PC2·PC2′ = 0.948) and the explained
variance barely moves (PC1 0.3607 vs 0.3518, PC2 0.1190 vs 0.1227). So the 2 % cache was not
producing a *wrong direction* — it was producing a sparsely populated surface along an almost
correct one.

1UAO projected into the new basis sits at **PC1 = +10.365 ± 0.710, PC2 = −0.163 ± 0.363**
(18 models). These are new-basis numbers and are *not* comparable to the old-basis values in §2.

### 6.1 PC1 separates the fold far better than the biasing CV does

This is the result that matters. Repeating the §4 degeneracy test on PC1:

| PC1 bin (Å) | frames | RMSD min | median | max | % < 1.5 Å | median Q |
|---|---|---|---|---|---|---|
| [−15.0, −10.0) | 9,475 | 4.74 | 5.53 | 6.45 | 0.00 | 0.000 |
| [−10.0, −5.0) | 116,235 | 3.42 | 4.89 | 6.22 | 0.00 | 0.000 |
| [−5.0, 0.0) | 203,835 | 2.28 | 4.00 | 5.68 | 0.00 | 0.005 |
| [0.0, +5.0) | 152,420 | 1.29 | 3.01 | 4.64 | 0.00 | 0.103 |
| [+5.0, +8.0) | 53,270 | 1.10 | 2.59 | 3.95 | 0.21 | 0.089 |
| [+8.0, +9.5) | 34,069 | 0.89 | 2.35 | 3.35 | 1.06 | 0.079 |
| [+9.5, +11.0) | 32,394 | 0.84 | 2.20 | 3.04 | 5.55 | 0.091 |
| **[+11.0, +14.0)** | **9,837** | **0.43** | **1.76** | **2.83** | **26.98** | **0.279** |

The monotonicity is the point. Where CV1's *best* bin was 3.65 % native, PC1's top bin is
**27 % native**, and the RMSD ceiling falls steadily with PC1 instead of staying near 4–6 Å.

Purity of the native window, compared like for like:

| window | frames | % native (RMSD ≤ 1.5 Å) | RMSD span | median Q |
|---|---|---|---|---|
| CV1 in native window [0.82, 0.93) | 58,670 | 3.65 | 0.43–4.27 | 0.174 |
| PC1 in native window (1D) | 73,894 | 6.07 | 0.43–3.35 | 0.098 |
| PC1 **and** PC2 in the 1UAO box (2D) | 17,018 | **10.44** | 0.43–3.35 | 0.164 |
| native PCA box **and** native CV1 window | 5,741 | 10.45 | 0.43–3.35 | 0.258 |

And the genuinely native frames concentrate where 1UAO is: the 33 frames with Q ≥ 0.80 lie at
PC1 = +12.02 ± 0.19, PC2 = −0.42 ± 0.30, with **84.8 %** inside the 1UAO box (60.5 % of the 481
frames under 1.0 Å).

### 6.2 Ranking power, tested four ways

AUROC for "is this frame native" (0.5 = no discrimination). Because PC1 is a linear function of
superposed coordinates and RMSD is a distance in that same space, scoring only against RMSD would
structurally flatter PC1 over the contact-based CV1. The test is therefore repeated with
contact-based definitions, which are what CV1 is made of:

| coordinate | RMSD ≤ 1.5 Å | RMSD ≤ 1.0 Å | Q ≥ 0.70 | Q ≥ 0.80 |
|---|---|---|---|---|
| CV1 contact fraction (the biasing CV) | 0.8591 | 0.9021 | 0.9257 | 0.9241 |
| **PC1 (recomputed, all frames)** | **0.9777** | **0.9951** | **0.9980** | **0.9977** |
| −Rg | 0.8384 | 0.9048 | 0.9388 | 0.9381 |
| −distance to 1UAO in the PC plane | 0.9679 | 0.9693 | 0.9686 | 0.9702 |
| positives | 4,929 | 481 | 173 | 33 |

PC1 wins on every definition, including the contact-based ones, so the advantage is not an artifact
of the scoring metric.

**It is also not in-sample optimism.** The basis above was fitted on the same frames it was scored
on. Refitting on half the *trajectory files* and scoring only the held-out half (splitting by file,
so correlated frames stay on one side):

| split | held-out PC1 | held-out CV1 | positives |
|---|---|---|---|
| fit A → test B (293,018 frames) | 0.9639 | 0.8373 | 984 |
| fit B → test A (318,518 frames) | 0.9799 | 0.8636 | 3,945 |
| in-sample reference | 0.9777 | 0.8591 | 4,929 |

The split also exposes how concentrated the native population is: **all 173 frames with Q ≥ 0.70
fall in one half**, which is §4.1's under-convergence showing up again and a reason not to
over-read a single split.

**On "reference-free", precisely.** PC1 is fitted with no native input, so it is available in a
campaign that has no reference structure. But knowing that *high* PC1 is the native direction came
from labelling frames by RMSD to 1UAO. Prospectively the PCA alone does not say which end to bias
toward — though for full-range umbrella sampling, as run here across CV1 0.021–0.986, the sign
ambiguity costs little since both ends get sampled. PC1 is a good *coordinate*; it is not a native
*detector* without a reference.

Two further caveats before PC1 is adopted. **Biasing on PC1 on the fly requires a fixed
superposition reference structure**, which reintroduces a reference through the back door — a
practical obstacle, since the basis here was defined against an iteratively converged ensemble mean
that does not exist until after the run. And **the AUROC comparison is run on a CV1-biased
ensemble**: the umbrella sampling deliberately flattens the CV1 distribution, which depresses CV1's
apparent discriminating power while leaving unbiased coordinates like PC1 and Rg their natural
separation. The file-split above fixes overfitting, not this distortion; an MBAR-reweighted AUROC
is owed before the 0.978 vs 0.859 gap is quoted as a property of the coordinates rather than of
this ensemble.

One unexplained detail: the native-like frames sit at PC1 = +12.02 ± 0.19 while the 1UAO models
project to +10.365 ± 0.710 — a ~2.3 native-sd offset. The MD native-like structures are
systematically displaced along PC1 from the NMR ensemble, and nothing here explains why.

### 6.3 What the recompute does *not* fix

The recomputed scores give geometry, not thermodynamics. A free-energy surface needs each frame's
MBAR/GaMD weight, and attaching those requires joining frames to sample rows by
`(phase, replica, segment, step)`. That join could not be completed from
`rg_samples_with_weights.csv`, which omits any segment or source column; because production steps
repeat across segments, `(step, replica)` is not unique there, and the reconstructed per-replica
series visibly cycles through the same values. The joinable product is
`adaptive_union_mbar.samples.csv`, which does carry `source` and `source_dir` — that is the route,
and it remains the open item from §7.

So: **no PCA free-energy minimum is claimed here, and the §2 PCA rows stay uncounted.** What the
recompute establishes is geometric and independent of weights — the native basin's location in the
PCA plane, and PC1's discriminating power.

![Recomputed PCA on all frames](native_fold_audit_results/pca_all_frames.png)

![PC1 versus CV1 discrimination](native_fold_audit_results/pca_vs_cv1_discrimination.png)

Panel (a) is sampling density over a biased (umbrella + GaMD) ensemble and is **not** a free-energy
surface; its densest cell (PC1 +9.20, PC2 +5.34) is a statement about where the bias put walkers,
not about stability.

## 7. What to do next — reweight before rerunning

**Lead with the data already in hand.** The existing ~6 µs can be projected onto a native-referenced
axis without new simulation, and that should be done before any rerun is considered:

1. Map each coordinate frame to its MBAR row by the stable identity `(phase, replica, segment,
   step)`. This is exact here — frames are every 2500 steps at 3.5 fs, MBAR samples every 250
   steps, so frames are every 10th sample — but it must be **audited, not assumed**, because
   resumes open new files and the step origin is segment-local.
2. Attach MBAR/GaMD weights **only** to matched rows; never interpolate the observable onto rows
   that carry no coordinates. Coordinate frames are ~12 % of the sample set, which costs precision
   and risks selection bias if writing was not configuration-independent — check that inclusion is
   uniform across phase, replica and state.
3. Build the PMF along Q and along ensemble RMSD, plus the 2D (CV1, Q) surface that lifts the
   degeneracy directly.
4. Report, per bin and especially in the native region: the overlap matrix, Kish effective sample
   size ESS = (Σw)²/Σw², the fraction of total weight carried by the largest single frame, the
   number of independent blocks contributing, block-bootstrap intervals, first/second-half
   reproducibility, and second- vs third-order cumulant sensitivity.

**The gate.** 481 frames can still yield ESS ≈ 3 if a handful of high-boost frames carry nearly all
the weight — structural presence would remain real while the population estimate stayed
meaningless. A rerun is justified only if that gate fails: native-region ESS too small, weight
dominated by one or two frames, the coordinate subset non-random or unmappable, or the (CV1, Q)
surface unstable under blocks.

**If a rerun is needed**, the obvious choice has changed. §6.2 shows **PC1 from the all-frames
PCA is the strongest reference-free discriminator available** (AUROC 0.978–0.998 against every
definition of native tried, versus 0.859–0.926 for CV1), so bias on PC1, or on 2D CV1 + PC1 to keep
the existing ladder commensurable. A native-referenced observable (Q against 1UAO, or ensemble RMSD
over the 18 models) remains the most direct option where using the experimental structure is
acceptable; PC1 is the one to use where it is not. Either way, do not bias to a single NMR model,
and seed from the native-like frames already extracted here.

One caution on PC1: it is fitted on *this* ensemble, so it is not a priori transferable to another
system or to a re-run whose sampling differs. Refit it on the new data and check that the basis is
stable — as it was here, where a 2 % and a 100 % fit agreed to |cos| = 0.996.

## 8. Changes made in response to review

| source | condition | action |
|---|---|---|
| board 1 | best-RMSD and highest-Q frames conflated | separated in §0; "native on every metric" withdrawn |
| board 2, codex 2 | "529 independent visits" unsound under exchange | §3.2: episodes, independence unestablished; dwell-time reading withdrawn |
| board 3, codex 1 | min-over-18 unquantified, no null | §3.1: inflation table, matched 18-reference null, decoy check |
| board 4 | PCA built on 2 % data yet counted | §0/§2 count four minima; PCA marked pending |
| board 5 | "three decimals at every percentile" overstated | §2: three decimals to p75, two in the tail, with the cause |
| board 6, codex 6 | prescription led with rerunning | §6 rewritten to lead with MBAR reweighting + ESS gate |
| board 7 | CV1 unit label | §5.3 |
| board 8 | causal claim too broad; CV1 bin missing | §4.1 scope paragraph; §4 table now sums to 611,536 |
| codex 3 | degeneracy vs under-convergence not separated | §4.1 segment stratification; both found |
| — | frame spacing was wrong (5 ps) | corrected to 8.75 ps throughout (§1) |
| advisor | 324× mixed compact-conditioned and whole-archive denominators | §3.1 recomputed on one test set: **61×** |
| advisor | "three criteria not constructed to agree" overstated | §0: they are related; reference-specificity of Q is the argument |
| user | recompute the PCA on all frames | §6: refitted on 611,536 frames, projected all, 1UAO reprojected |
| advisor | Rg/d1-d2 provenance unchecked | §5.4: forward-fill confirmed; support is 611,536, not 5.11M |
| advisor | recompute implied parity with 5.11M | §5.1: ceiling is 11.9 %, stated explicitly |
| advisor | AUROC could flatter PC1 over CV1 | §6.2: repeated under contact-based definitions; PC1 still wins |
| self-review A1 | minima quoted from one estimator, spread undisclosed | §2: all four estimators tabulated, selection justified |
| self-review A3/A4 | failing quality gate and λ=0 deficit not disclosed | §2: gate, 38 weak edges, 34.4 % retention, cross-check |
| self-review A5 | PC1 AUROC was in-sample | §6.2: held-out file-split validation added |
| small board 1 | "no estimator" framing is consensus-of-noise | §2: reframed as disclosure, not corroboration |
| small board 2 | conclusion (a) scope too broad | §0: narrowed to the run's 1-D PMF products |
| small board 3 | Rg/d1/d2 affected by forward-fill | §2.1: **recomputed from matched rows; all three minima confirmed unchanged, with bootstrap CIs** |
| small board 5 | no Q-based decoy null; 61× over-precise | §3.1: tightness-matched Q null added; 61× given as [17×, 516×] |
| small board 8 | PC1 recommendation over-sold | §6.2: superposition-reference and bias-distortion caveats |
| small board 9 | counts unreconciled, parameters absent | §1.1, §1.2 |
| small board 7 | run health not in main results | §2 |

## Files

| file | content |
|---|---|
| `native_fold_audit_results/frame_metrics.npz` | all 611,536 frames × 11 metrics, with source file and frame index |
| `native_fold_audit_results/native_reference.json` | 1UAO values for every metric, per model |
| `native_fold_audit_results/matched_null_corrected.json` | **the matched null (superseding the 324× figure)** |
| `native_fold_audit_results/matched_null_and_stratification.json` | per-segment stratification (its `sparser_factor` is superseded) |
| `native_fold_audit_results/q_discrimination.json` | Q-based discrimination counts |
| `native_fold_audit_results/pca_all_frames.npz` | **recomputed PCA: scores for all 611,536 frames, basis, 1UAO projection** |
| `native_fold_audit_results/pca_all_frames.json` | recomputed PCA metadata and basis comparison |
| `native_fold_audit_results/pca_analysis.json`, `auroc_fairness.json` | PCA degeneracy, purity and AUROC results |
| `native_fold_audit_results/pca_all_frames.png` | density, native fraction and native-like frames in the new PCA plane |
| `native_fold_audit_results/pca_vs_cv1_discrimination.png` | native fraction vs PC1 and vs CV1 |
| `native_fold_audit_results/pca_report.txt` | full numeric output of the PCA analysis |
| `native_fold_audit_results/report.txt` | full numeric output |
| `native_fold_audit_results/structures/native_like_top25.pdb` | 25 best native-like frames, superposed on 1UAO |
| `native_fold_audit_results/*.py` | `scan`, `report`, `where`, `extract`, `null`, `null2`, `null3`, `qtest` — reproduce everything |

Board review (5 judges, debate, math/physics audit): `native_fold_audit_results/board_review/`
(`final_report.md`, `transcript.md`, `board_summary.json`). Codex review:
`native_fold_audit_results/codex_review.log`. This document is also available as
`NATIVE_FOLD_AUDIT_2026-09-18.docx`.
