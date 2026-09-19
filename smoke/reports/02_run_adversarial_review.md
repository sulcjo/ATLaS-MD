# Adversarial review — the `chignolin_7` run and the native-fold audit

Self-review, deliberately hostile, of two things: (1) `NATIVE_FOLD_AUDIT_2026-09-18.md`, and
(2) the `chignolin_7` production run it draws on. Every number below is measured from the run's
own diagnostics. Findings are graded **CRITICAL** (invalidates a stated conclusion), **HIGH**
(a conclusion needs restating), **MEDIUM** (incomplete or misleading), **LOW** (presentation).

## Part A — attacks on the audit report

### A1. HIGH — the headline deviations are estimator-specific, and this is never disclosed

The report states four PMF minima and their distance from native in units of native sd. Every one
of those numbers comes from a single estimator (`selected_unbiased_method: umbrella_only`) whose
selection the report never justifies or even mentions. The estimators disagree enormously:

| estimator | CV1 minimum | Rg minimum (Å) |
|---|---|---|
| umbrella_only *(the one quoted)* | **0.4873** | **6.594** |
| gamd_exponential | 0.7445 | 5.823 |
| gamd_cumulant2 | 0.1657 | 7.365 |
| gamd_cumulant3 | 0.1014 | 7.751 |
| native (18 models) | 0.82–0.93 | 5.14–5.66 |

The CV1 minimum ranges over 0.10–0.74 — most of the coordinate — and the Rg minimum over
5.82–7.75 Å. Under `gamd_exponential` the Rg minimum is 5.823 Å, which is **0.16 Å above the
native maximum**, not the "+6.7 native sd" the report asserts. A reader given only the
umbrella_only column cannot see this.

**What survives.** The claim survives, but not by the route first written here. "No estimator
places a minimum inside the native window" is a *consensus of noise*: three of the four estimators
are invalid (below), so their agreement corroborates nothing. The spread is a **disclosure**, not
evidence. The actual support is `umbrella_only` plus the λ=0 cross-check — and, separately, the
fact that the reported minima sit 0.1–0.4 in CV1 below a native window the run samples heavily.
**What must change.** The report must publish the estimator spread, justify the choice, and stop
quoting single-estimator sd-distances as if they were the measurement.

**The justification exists and is strong — the report simply never makes it.** For this λ-ladder
construction the GaMD boost enters the MBAR reduced potential directly, so MBAR is exact and no
reweighting correction is needed at all (`pmf_summary.json` warnings: *"λ ladder: boost is inside
u_nk, MBAR is exact; no cumulant"*). The alternatives are independently disqualified:
`boost_reweight_ess = 51.2` out of 4,583,936 samples (fraction 1.1 × 10⁻⁵) makes
`gamd_exponential` statistically empty, and `anharmonicity_score = 0.799` with excess kurtosis
0.905 puts the cumulant expansions far outside their validity range. So umbrella_only is the right
choice — which makes its silent selection a reporting failure, not a physics error.

### A2. HIGH — "native sd" is a structural spread, not an uncertainty

The report expresses every deviation in units of the 18-model NMR spread ("−11.5 sd", "+6.7 sd").
That denominator is the conformational width of the reference ensemble. It is **not** the
uncertainty of the PMF minimum, and the phrasing invites reading these as significance levels.
No confidence interval on any minimum is reported anywhere. Given A1, the honest statement is a
distance in native-ensemble widths, explicitly labelled as such, with the estimator spread beside
it.

### A3. HIGH — the report never discloses the run's own failing health verdict

`adaptive_quality_gate.md` reads **`Status: needs_more_sampling`** with **38 edges below
threshold**, and the project's own ladder diagnostic grades RESULT HEALTH as **FAIL**. The audit
quotes that run's PMF minima for two pages without once telling the reader the run did not pass
its own gate. That is a material omission regardless of whether the conclusions survive it (§B
argues they largely do).

### A4. MEDIUM — the λ=0 sampling deficit is never mentioned

The PMF is reported for the unboosted ensemble, and the λ=0 rung retains only **34.4 %**
(520,016 / 1,512,912) of its intended samples, because three λ=0-only top-ups ran fully boosted.
A further consequence is visible in the union analysis: `finite_fraction = 0.857627`, i.e. **14.2 %
of the reduced bias matrix is non-finite** — those excluded wrong-ensemble rows. MBAR handles this
correctly by dropping them, but a report resting on the λ=0 ensemble must say that the rung it
depends on is a third sampled.

### A5. MEDIUM — PC1's advantage was measured in-sample; now tested, and it holds

§6.2 reports PC1 AUROC 0.978 vs CV1 0.859 with the basis fitted on the same 611,536 frames used to
score it. That is in-sample and was presented without validation. Tested here by splitting the
**trajectory files** (not frames, so correlated frames stay on one side) and fitting on one half:

| split | held-out PC1 | held-out CV1 | positives |
|---|---|---|---|
| fit A → test B (293,018 frames) | 0.9639 | 0.8373 | 984 |
| fit B → test A (318,518 frames) | 0.9799 | 0.8636 | 3,945 |
| in-sample reference | 0.9777 | 0.8591 | 4,929 |

The advantage is real, not optimism. One caveat the split exposes: **all 173 frames with Q ≥ 0.70
fall in a single half**, so the native-like population is concentrated in a small set of files —
consistent with §4.1's under-convergence finding and a reason not to over-read any single split.

### A6. MEDIUM — "PC1 is reference-free" is true of the fit and misleading about the use

PC1 is fitted without any native input, which the report states correctly. But *knowing that high
PC1 is the native direction* came from labelling frames with RMSD to 1UAO. Prospectively — the
situation a new campaign is in — nothing in the PCA itself says which end of PC1 to bias toward.
PC1 is a good *coordinate*; it is not a native *detector* without a reference. The report should
say that plainly rather than leaving "reference-free" to do work it cannot do.

### A7. LOW — the 61× matched-null figure rests on one arbitrary choice

The decoy references are drawn from CV1 ∈ [0.82, 0.93). A different compactness window would give
a different ratio. The direction is not in doubt, but the single number is softer than its two
significant figures suggest.

## Part B — attacks on the `chignolin_7` run

### B1. CRITICAL — the λ ladder is under-coupled almost everywhere

38 rung edges fall below the project's own `min_rung_overlap = 0.15`, with MBAR state overlaps of
**0.065–0.15** (median ≈ 0.085 on the two lower edges). This is not one marginal edge; it is
essentially the whole lower ladder. The two lower edges are ~1.8× weaker than 0.643 ↔ 1.000, so
the spacing is mis-designed in the low-λ region rather than merely under-sampled.

**Mitigation that must be stated with it.** The 0.15 constant was calibrated on a 1-D, 5-state,
single-centre rung ladder. This is a 2-D 16 × 4 grid in which each state shares its unit column sum
with up to four neighbours instead of two, which depresses pairwise overlap mechanically. The
threshold is therefore conservative here by construction, and the FAIL is partly a
threshold-transfer artifact — but "partly" is not "entirely", and nobody has quantified the
correction.

### B2. HIGH — a diagnostic mismatch, now resolved (my first explanation was wrong)

The same edges that fail on MBAR overlap (0.07–0.15) show **exchange acceptance of 0.74–0.86**.
High acceptance normally means neighbouring states are nearly indistinguishable, which should imply
*high* overlap. Two diagnostics of the same coupling disagree by a wide margin and the run reports
both without comment. My first explanation — "acceptance is computed on the boost term only" — is **wrong, and not a
bug**: for states sharing a Hamiltonian and differing only in λ, the physical energy cancels
exactly in the swap criterion, leaving Δ = Δλ·ΔV by construction. That is the correct acceptance
test, not a truncated one.

The real resolution is that the two statistics are **incommensurable, and acceptance is the
misleading one**. Exchange acceptance is a bulk pairwise statistic dominated by typical
configurations; MBAR overlap is a tail-sensitive, globally normalised statistic dominated by where
the weight mass actually sits. The boost distribution here is heavy-tailed — the same physics that
drives `boost_reweight_ess` to 51 — so bulk and tail diagnostics must diverge. Two further effects
push the same way: exchange is attempted every 1.4 ps, so swap pairs are correlated and not
re-equilibrated, which inflates acceptance; and in the 2-D 16 × 4 grid each state shares its unit
column sum with up to four neighbours, mechanically depressing overlap against a threshold
calibrated on a 1-D 5-state ladder.

Consequences: acceptance of 0.74–0.86 must **not** be read as "rungs too close"; the
dilution-corrected overlap is what governs MBAR variance; and the arbiter is the λ=0 cross-check
(B6). Residual genuine under-coupling in the low-λ region stands — 0.065 is low even after a ~2×
dilution allowance — as a caveat, not a refutation.

### B3. HIGH — GaMD reweighting is unusable, which narrows the run to one estimator

`anharmonicity_score = 0.799` (cumulant expansion invalid), `boost_reweight_ess = 51.2` of 4.58M
(exponential reweighting empty). The run therefore has exactly one viable estimator and no
independent reweighting cross-check of it. The saving grace is B6.

### B4. HIGH — one sixth of the campaign was thrown away

Three λ=0-only top-ups ran fully boosted, producing **992,896 wrong-ensemble samples (16.2 %)**
that MBAR correctly excludes. That is not just lost compute: the top-ups were scheduled precisely
to strengthen the λ=0 rung, so the adaptive scheduler's corrective action silently did the opposite
of its intent for three consecutive epochs.

### B5. MEDIUM — analysis-path fragility

`adaptive_union_mbar_analysis.md` reports `PyMBAR unavailable ... module 'scipy.linalg' has no
attribute 'tril'` and consequently **wrote coverage diagnostics only** — that analysis never ran
MBAR at all. It also reports 2,387,808 samples where `adaptive_union_mbar.npz` carries arrays of
2,947,296 rows, a discrepancy no output explains. Separately, `traj_interval = 2500` left only
11.9 % of samples with coordinates (fixed to 250 in v0.8, too late for this campaign), and
trajectory-derived observables are forward-filled across sample rows (audit §5.4).

### B6. What actually holds up — the run is marginal, not broken

Against the above, four independent checks pass and they are the reason the audit's conclusions
survive:

| check | result |
|---|---|
| MBAR solver | converged, 69 iterations, `max_delta = 9.7 × 10⁻¹¹`, all 64 states active |
| base effective sample size | 833,075 |
| **λ=0 ladder cross-check** | **PASS** — PMF from the 519,898 λ=0 samples alone agrees with full MBAR to **0.289 kcal/mol** (tolerance 0.5) |
| Rg PMF time-convergence | converged; final JS 1.2 × 10⁻⁷, tail median RMSE 0.031 kcal/mol over 10 checkpoints |
| CV1-marginal window overlap | nearest-neighbour median 0.965 (min 0.491); adjacent-centre median 0.484 |
| per-window sampling | 24,050–94,720 samples per window; `cv_std` 0.032–0.040, consistent with the applied force constants |

The cross-check is the important one, but it is **lower-power than first claimed here**. It is the
right leave-one-axis-out test — it removes exactly the data path (boosted→unboosted mapping through
weak λ edges) that B1 threatens — and it rests on healthy umbrella-direction overlap. But: both
PMFs share the λ=0 samples, so it is a consistency check rather than independent validation; weak
λ-coupling partly *insulates* the λ=0 estimate, so agreement is somewhat expected by construction
(a broken ladder fails to propagate influence in either direction); the 0.289 kcal/mol figure is
not specified as max or mean, over which CV range, and does not cover d1/d2 or Rg; and both
estimates rest on the same 34.4 %-retained λ=0 rung, so common-mode sampling gaps are invisible to
it. Correct reading: a **reassuring, low-power sensitivity analysis** that licenses the CV1 PMF as
an estimate of the sampled ensemble — not a validation of native-region free energies, which remain
unresolved and ESS-gated.


### B7. HIGH — the force field is a confound nobody costed

Neither document stated the simulation parameters. They are: **ff14SB protein, TIP3P water, 300 K,
NPT at 1 bar, 3.5 fs with HMR (H = 3.024 amu), 0.15 M ionic strength, dodecahedral box, 1.0 nm
padding, 0.8 nm nonbonded cutoff, Langevin friction 1.0 ps⁻¹.**

This matters more than a methods-section omission. The audit's conclusion (a) — the native fold is
not at a free-energy minimum — is reported as if it were a statement about sampling and
coordinates. With ff14SB/TIP3P it may be substantially a statement **about the force field**.
ff14SB was parameterised before the water-model corrections that later work applies to
folded/unfolded balance, and TIP3P is known to under-solvate and to shift small-peptide
folding equilibria; the project's own decision to move to **ff19SB + OPC for subsequent runs** is
an implicit acknowledgement. A separate minor point: a 0.8 nm nonbonded cutoff is short for an
Amber-family force field, which is usually run at 0.9–1.0 nm.

Nothing here is evidence the fold *should* be the minimum. The point is that "the fold is sampled
but not scored" has at least three candidate causes — the degenerate coordinate (established), the
marginal ladder and λ=0 deficit (established), and the force-field/water model (untested) — and the
audit currently attributes the result to the first two only. Distinguishing them needs a
native-referenced free energy under this force field, and ideally the same under ff19SB/OPC.

### B8. MEDIUM — sample counts now reconciled, including the one called unexplained

Five counts circulate. They reconcile as follows:

| count | meaning |
|---|---|
| 5,124,128 | total MBAR samples in `pmf_analysis` (= Σ n_k over 64 states) |
| 5,110,208 | the finite subset used for the FES products (13,920 dropped) |
| 4,583,936 | samples carrying a finite GaMD boost — exactly 5,124,128 − 540,192, the epoch_000 sample count |
| 2,947,296 | rows in the separate `adaptive_union_mbar.npz` product |
| 2,387,808 | rows the union *analysis* used — **exactly 2,947,296 − 559,488**, the npz's own `gamd_ladder_samples_without_raw_energies` |
| 611,536 | coordinate frames (11.9 % of 5,124,128) |

So the 2.39M-vs-2.95M gap the board flagged as unexplained is fully accounted for by the npz's own
field. The 4,583,936 identity is arithmetically exact but undocumented, and should be confirmed
rather than inferred.

## Verdict

**On the run:** genuinely marginal, not broken. The λ ladder is under-coupled and a sixth of the
campaign was wasted on a scheduler bug, but the MBAR solution is converged, well-supported, and
independently cross-checked at λ=0. Free energies from it should carry the failing gate and the
34.4 % λ=0 retention as stated caveats; the *structural* conclusions (audit §3, §6) do not depend
on the free energies at all and are unaffected.

**Reviewed by the small board** (kimi, mini, thinker; chair glm) — unanimous ACCEPT-WITH-CHANGES at
confidence 85, after a split tier-1 and one debate round. It corrected the B2 physics above,
downgraded the λ=0 cross-check as recorded above, demanded the Q-based decoy null (now run, see the
audit §3.1), and required the scope of conclusion (a) to be narrowed to the run's 1-D PMF products
rather than the system's true free-energy landscape. Its full judgment and the transcript are in
`native_fold_audit_results/small_board_review/`.

**On the report:** the two headline answers survive — no minimum sits at native on any estimator,
and the fold is structurally present — but A1 requires a substantive revision (publish the
estimator spread, justify umbrella_only), and A2–A4 require disclosures the report currently
omits. A5 was a real hole and is now closed by held-out validation. None of the findings overturn
a conclusion; three of them change what the report is entitled to claim.
