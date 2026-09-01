# Does ATLaS-MD sample the correct per-state Boltzmann distribution?

Validation report, 2026-09-01. Branch `validate/boltzmann-correctness`.

## Why this exists

The claim under test, made while discussing window pinning:

> Exchange obeys detailed balance, so the joint ensemble stays correct and the samples labelled
> state *k* still converge to state *k*'s Boltzmann distribution.

That is an assertion about a 7,000-line MD driver with a Gibbs-walk exchange kernel, a GaMD boost,
and a Hamiltonian that changes between epochs. It needed evidence, not confidence.

Target agreed up front: **internal confidence**, not a publication artefact. Budget: toy systems
plus one dedicated real run.

## The claim is two claims

Conflating these is the single biggest source of confusion in this area, including in my own first
answer.

| | Statement | Status |
|---|---|---|
| **Layer A — the sampler** | Samples labelled state *k* are Boltzmann for state *k*'s **biased** Hamiltonian | **The subject of this report. Passes, with one stated exception.** |
| **Layer B — the estimator** | The **unbiased** PMF can be recovered from those biased+boosted samples | **Split verdict: exponential reweighting fails structurally; CE2's error is *unresolved* on this run. See Tier 0.** |

A pipeline can be perfectly correct at Layer A and useless at Layer B, and that is roughly where
ATLaS-MD sits at the current boost width: the exponential estimator has infinite weight variance, and
CE2's truncation error cannot be measured because the diagnostic needs the same divergent average.
Both are consequences of one tunable setting. An earlier draft claimed both were fatally broken for a
reason that was wrong; a later one over-corrected and called CE2 adequate. Neither holds — see
Tier 0.

Layer A decomposes into five links:

| Link | Statement | Evidence now |
|---|---|---|
| L1 | The exchange kernel is π-invariant | **Proven exactly** on finite bias matrices (Tier 1); broken for `gibbs-walk` alone when CV2 is NaN |
| L2 | ΔV does not read the umbrella, so it cancels from the acceptance | Proven from the installed integrator; **unguarded** for 4 boost types |
| L3 | Within-state dynamics are Boltzmann for the biased+boosted potential | Untested with an analytic FES (Tier 3, planned) |
| L4 | Recorded parameters equal applied parameters, per phase | Sound; **measured** on both CV axes, consistent with exact agreement, bounded by the test's ~2.4% resolution (Tier 2) |
| L5 | No non-equilibrium interventions | One is armed but has not fired; one NaN path can bias `gibbs-walk` |

---

## What the code audit established

Three parallel audits of the exchange implementation, the existing validation infrastructure, and
every mid-run Hamiltonian change.

**The mechanism is stronger than originally claimed.** The move swaps *state labels only*
(`production.py:6403-6404` calls `set_window`, which touches OpenMM global context parameters and
nothing else — no `setPositions`, no velocity rescaling). Because configurations never move between
contexts, `U₀(x_i)+U₀(x_j)` and `ΔV(x_i)+ΔV(x_j)` appear identically on both sides of the acceptance
ratio and cancel **exactly**, leaving

```
Δ = β[ u_wj(x_i) + u_wi(x_j) − u_wi(x_i) − u_wj(x_j) ]
```

Kinetic terms cancel (momenta unchanged, same T and masses) and so do the NPT `pV` and
volume-measure terms (the volume is unchanged and p, T are the same for every replica).

**Two conditions this rests on, stated precisely** — an earlier phrasing here said the cancellation
holds "even if the boost were per-replica", which is true but ambiguous in the way that matters:

1. The boost must be keyed to the **coordinate slot**, not to the state label. In this driver each
   replica owns its own `Simulation`, context and GaMD integrator, and a swap moves neither
   coordinates nor integrators — so `ΔV(x_i)` is computed by the same integrator on the same
   coordinates before and after, and cancels. A hypothetical *per-window-label* boost would **not**
   cancel, and the acceptance ratio would need the ΔV terms written out explicitly.
2. The boost must not read the umbrella force group. If it did, changing the label would change the
   boosted energy and the cancellation would fail. This holds for the default and for this run's
   `lower-dihedral` (group 2 = `PeriodicTorsionForce`/`CMAPTorsionForce`, while both restraints are
   `CustomCVForce` in unboosted group 0), and **fails for 4 of the 11 selectable boost types** — see
   defect 3.

The provenance note at `:5772` ("shared GaMD terms cancel") reaches the right conclusion by the wrong
route: the shared envelope is not what makes exchange valid. It matters for MBAR state
well-posedness, a different claim.

**The cancellation premise is proven from the installed integrator**, not inferred.
`gamd/stage_integrator.py:748-769` dispatches on `BoostMethod`: `GROUPS` builds ΔV from
`energy1`+`energy2` (force-group energies), while `TOTAL` and `DUAL_DEPENDENT_GROUP_TOTAL` use bare
`energy`. The umbrella lands in group 0 after `set_all_forces_to_group(system, 0)`, so under the
default `lower-dual-nonbonded-dihedral` it is unboosted. Exactly **4 of 11** boost choices are
unsafe: `lower-total`, `upper-total`, `lower-dual`, `upper-dual`. Note `lower-dihedral` and
`lower-nonbonded` are single-*group* boosts and are safe.

**Every Hamiltonian change is at a phase boundary**, and each changed potential is recoverable from
that phase's own directory: restraint centres in `epoch_window_map.csv`; the complete CV2 definition
(`mode`, the 36-element `weights` vector, `tica_offset`, torsion indices) in
`<phase>/gareus_metadata.json` under `/secondary_cv/`; the GaMD envelope in the phase-local
`shared_gamd_setup_globals.json`. GaMD production runs in stage 5, which never calls
`_calculate_primary_boost_statistics`, so boost parameters are frozen mid-phase.

**Sample labelling is correct by construction**: `sample()` is called at `:6769`, before
`attempt_exchanges` at `:6777`, and reads `assignments[r]` directly (`:6130`).

---

## Tier 0 — can GaMD reweighting work at all? (`audit_reweighting_feasibility.py`)

Analytic check on existing data, zero compute.

> **Data snapshot.** `RUNS/chignolin_6` is rsynced from the cluster and grew during this analysis:
> `final/baseline` went from 72 rows to 1,714,788, and the abandoned `final` arrived renamed
> `final_CRASHED_20260901`. All Tier 0 numbers below are from the post-rsync snapshot
> (15,375,316 raw rows). Earlier figures in the git history of this file were computed before that
> and are superseded, not contradicted. Abandoned `*_CRASHED*` phases are now excluded by the audit.

On the current snapshot: **13,187,613 samples across 37 states**, after mapping each row's
phase-local `window_id` through its own phase's `epoch_window_map.csv` and de-duplicating (14.2%
dropped as exact duplicates from the known un-consolidated-`chunk_*.parquet` leftover).

> An earlier version of this section pooled on raw `window_id` and reported 36 windows. That is the
> same defect the 2026-08-25 window-map work exists to fix — `window_id` is phase-local, and 3 of
> the 6 phases have a non-identity map. Every number below is from the corrected version.

### The two GaMD regimes must not be pooled

The shared-envelope recalibration fires **at most once**, so `epoch_000` ran under the
pre-recalibration boost and every later phase under the recalibrated one. This repo already treats
them as incomparable — `analyze_gareus_mbar.py` splits its own PMF and boost report on exactly this
(`_epoch_zero_split_masks`, `epoch_000_separate/`). An earlier version of this audit pooled them into
one `a`:

| regime | n | ⟨βΔV⟩ | βσ | anharmonicity | a |
|---|---|---|---|---|---|
| epoch_000 (pre-recal) | 1,562,528 | 7.50 | 3.37 | 0.699 | **0.388** |
| epoch_001+final (post-recal) | 11,625,085 | 8.82 | 3.57 | 0.601 | **0.369** |
| pooled *(not valid)* | 13,187,613 | 8.66 | 3.57 | 0.606 | 0.377 |

The regimes really do differ, and pooling inflates `a` mechanically — `a ≈ var/(4·mean)` and a
between-regime mean offset adds to the variance. The headline below uses the **post-recalibration**
samples, which are the ones the main PMF report covers. The exponential verdict is unchanged either
way (both regimes sit far above 0.25), so this corrects the number without changing the conclusion.

The two estimators fail for **different reasons**The two estimators fail for **different reasons** and the earlier draft conflated them.

### Exponential reweighting — structurally unusable

For a lower-bound GaMD boost, ΔV = ½k(E−V)², so βΔV is a scaled noncentral chi-square with one
degree of freedom, βΔV ~ a·χ′²₁(λ). Moment-matching gives `a = m − √(m² − v/2)`. The MGF
`E[e^{tX}]` exists only for `t < 1/(2a)`, so

| bound | meaning |
|---|---|
| a < 0.50 | `E[w]` finite — the estimator is defined |
| a < 0.25 | `E[w²]` finite — it has finite variance |

**Measured: a = 0.369** on the post-recalibration samples (0.388 for epoch_000), and 0.367–0.383 in every state with ≥1000 samples. So the
importance weights have **infinite variance**: Kish ESS has no finite limit, does not grow with N,
and any single ESS figure quoted for it is an artefact of which extreme frame happened to be drawn.
This is a structural verdict, not a sampling-quality complaint — and it is the reason the earlier
draft's "ESS = 1.7 of 13.7 million" headline was meaningless rather than alarming. That number was
never converging to anything.

### CE2 — not condemned, but **not resolved either**

The first draft asserted a validity domain of βσ_ΔV ≲ 1 and declared the run 5× outside it. That
criterion was invented, and it is wrong in principle: **CE2 is exact for a Gaussian ΔV at any
width**, because all cumulants above the second vanish. Width alone condemns nothing. What matters
is non-Gaussianity — which this repo already measures, in `gareus.math_helpers.boost_anharmonicity`,
labelled OK/WARN/BAD at 0.5/1.0.

**Measured: anharmonicity = 0.601 → WARN**, not BAD (epoch_000, under the other envelope, gives 0.699 — also WARN).

Because a PMF is defined only up to a constant, the reportable quantity is the **spread across CV
bins** of the neglected terms — a uniform error cancels, a bin-dependent one changes the PMF's shape.
Three estimators, over 11 CV1 bins:

| estimator | spread (kcal/mol) | what it is |
|---|---|---|
| 3rd-order term only | **0.23** | directly measured. A floor, not the tail. |
| parametric (noncentral χ²) | **1.37** | closed form `λa/(1−2a) − ½ln(1−2a)` minus CE2, exact **for the fitted family only** |
| empirical | **3.03** | model-free `ln⟨e^X⟩ − CE2`, but see below |

**Verdict: NOT RESOLVED.** The empirical column needs the very `exp(+βΔV)` average whose variance is
infinite at `a = 0.369`, so it is not a measurement — it is one draw from a distribution with no
finite spread. Worst-bin exponential-average ESS is **0.0001%** of that bin's samples. The parametric
column is a sensitivity analysis, not a bound: two moments fix CE2 but do not constrain the higher
cumulants, and distributions sharing two moments can have arbitrarily different `ln⟨e^X⟩`.

The fitted family is a good description of the real data — on `epoch_000`
(a = 0.388, λ = 18.3) it predicts skew 0.685 against a measured 0.647 and excess kurtosis 0.629
against 0.533. **That does not make the parametric column conservative**, though it is tempting to
read it that way: the MGF weights the tail exponentially, so a distribution with a lighter-looking
bulk and smaller third and fourth cumulants can still have a *larger* `ln⟨e^X⟩` deficit. Matching two
moments, or even four, bounds nothing here.

> This is demonstrated rather than argued, on synthetic data where the answer is known
> (`tests/test_gamd_reweighting_feasibility_math.py`). Drawing 2M samples from an *exact* noncentral
> χ² with these parameters, the closed form gives **34.42** and the Monte-Carlo estimate gives
> **22.24**, differing between seeds. At `a = 0.05` the same code reproduces the closed form to
> better than 0.05. The empirical estimator's failure here is a property of `a`, not of the data.

#### Correction to the previous correction

The version of this report committed earlier today reported **≈0.95 kcal/mol**, obtained by summing
a geometric tail at ratio `2a = 0.754` from the measured third-order term. That was wrong, and the
report's own data contradicted it: the measured 4th-order spread (0.38) exceeds the 3rd (0.23), a
ratio of 1.65, not 0.754.

`2a` governs successive terms **within a bin at fixed (a, λ)**. It does not govern the spread
*across* bins, where `a` and `λ` themselves vary — the `n·a^(n−1)` sensitivity makes the spread ratio
behave like `2a(n+1)/n`, about **1.0** at n=3→4. So the extrapolation was not merely unjustified, it
was biased low. The honest range is 0.23 (measured floor) to ~3 (unreliable), and the two non-floor
estimators both sit **above** the 1 kcal/mol target rather than below it.

So: the first draft's *reason* for condemning CE2 was wrong, and this report's first correction
over-rehabilitated it. CE2 is not shown to be broken; it is **not shown to be adequate either**, and
on this run it cannot be, because the diagnostic needs the same divergent average the exponential
estimator does.

The **1 kcal/mol bar is a chosen convention** — the accuracy target for this kind of PMF work, stated
as such. After criticising the first draft for inventing a βσ ≲ 1 criterion, a second unsourced
threshold should not read as derived.

### The boost width is a setting — but the obvious way to turn it does nothing

```
lower-dihedral,  sigma0p = 2.5 kcal/mol,  k0 = 1.0,  k0' = 1.694
(from adaptive_production/global_shared_gamd_setup/ -- the production calibration)
```

`k0 = min(1, k0')` and `k0'` is proportional to σ0, so **`k0` is clipped at its ceiling**. Two
consequences, the second of which I got wrong in the earlier version:

1. The applied boost is **smaller** than the setting requests, not larger — the clip is protective.
   Measured σ_ΔV = 8.90 kJ/mol against a configured σ0 = 10.46 kJ/mol. GaMD's own σ0 criterion is
   satisfied; the problem is that the target itself is too wide for reweighting.
2. **Every value of σ0p between 1.48 and 2.5 kcal/mol produces the identical boost.** Lowering it
   within that range changes nothing at all, because `k0'` stays above 1 and `k0` stays pinned.

> **Two corrections here, made in sequence.**
>
> *First*, the recommendation σ0p ≲ **1.66** kcal/mol came from scaling 2.5 × (0.25/0.377) because
> `a` is linear in boost strength. `a` *is* linear in `k0` — ΔV is pointwise proportional to `k0`, so
> mean ∝ k0 and variance ∝ k0², giving `a ≈ var/(4·mean) ∝ k0`. But at σ0p = 1.66 the clip is still
> active and **nothing changes**. Scaling only begins at the clip boundary.
>
> *Second*, the `k0'` used to locate that boundary was read from the wrong file. The audit took
> `sorted(glob(...))[0]`, which is positional, and `adaptive_feedback_round_01/` sorts first — the
> short **diagnostic pilot**, which calibrates against its own sampling and reports `k0' = 1.766`.
> Every production phase reports **1.694**. Same failure shape as reading a phase-local `window_id`
> as a state id: a plausible artifact that is not the right one. Selection is now explicit, and the
> pilot is used only if nothing else exists.
>
> Corrected: clip boundary = 2.5/1.694 = **1.48**; `k0` must fall to 0.25/0.369 = 0.678; so
> **σ0p ≈ 1.00 kcal/mol**.

Caveat: this assumes `Vmax`/`Vmin`/`σV` are unchanged by the new setting. They are re-measured at
calibration, so treat 1.00 as a starting point and re-run this audit on the result.

**Consequence.** The first draft concluded "Layer B is not fixable by running longer; any target
requiring ≲1 kcal/mol needs GaMD off." The first clause stands and is now on firmer ground —
`a = 0.369 > 0.25` means the exponential estimator's weight variance is infinite, so no amount of
sampling helps. The second clause is still not established, but not for the reason the first
correction gave: CE2's error is **unresolved**, not shown to be small. What *is* actionable is that
the boost width is a setting, and σ0p ≈ 1.00 kcal/mol would move the exponential estimator inside its
finite-variance bound and make CE2's own error measurable for the first time.

This section no longer claims to reproduce the ala-dipeptide `gamd_cumulant2` +18 kcal/mol result.
That was a different system at different settings, and the mechanism there was not established by
this analysis.

## Tier 1 — the exchange kernel, proven exactly (`tests/test_exchange_kernel_exact.py`)

Because the move permutes labels against a frozen bias matrix, the exchange chain is a **finite
Markov chain on the N! permutations**. It does not need to be sampled: the transition matrix can be
built by driving the shipped kernel and checked to machine precision. No sampling error, no
autocorrelation, no power calculation.

π-invariance survives composition and state-independent mixtures, and every mode's selection is
state-independent, so verifying the elementary kernels suffices — the pair-swap kernel per window
pair covers `neighbor` / `random-pair` / `all-pair-sweep`, and the single-replica gibbs kernel
covers `gibbs-walk`.

For every window pair and every replica, at N = 3, 4, 5, to **1e-12**: rows sum to 1; `πP = π`; and
**detailed balance** `π_σ P[σ,σ'] = π_σ' P[σ',σ]`. Detailed balance is asserted rather than mere
invariance because it localises the offending permutation pair. Composition is checked for
invariance only — a product of reversible kernels is π-invariant but not itself reversible, so
asserting DB there would be wrong rather than stricter. A separate irreducibility check guards what
balance structurally cannot see: a kernel that never moves is perfectly π-invariant.

**Enabling refactor.** Two functions were lifted out of the `run_gareus` closure, where the decision
logic and the `assignments`/`replica_of_window` bookkeeping were unreachable from any test — the
structural reason three of four modes had no coverage:

- `apply_window_swap()` + `SwapOutcome` — the pair-swap decision and its bookkeeping. `uniform` is a
  parameter rather than an rng handle so the caller preserves production's short-circuit:
  `force_accept` must not consume a random number, or every later RNG stream shifts. (A first version
  of this refactor drew a uniform even on the un-attemptable path, which the original closure did
  not. Fixed with `swap_candidate_replicas()`, checked before drawing; verified 0 draws consumed.)
- `gibbs_propose_one_replica()` + `GibbsProposal` — the heat-bath proposal, the reverse proposal
  against the post-swap holder table, and the MH correction. **This one was added only after
  adversarial review**, and the reason is the point of the whole exercise.

**Why the gibbs extraction was necessary.** The first version of this test file re-implemented the
gibbs proposal sequence inline, "mirroring the composition in run_gareus's `gibbs-walk` branch
exactly". A verifier showed that made the test self-consistent and **production-blind**: three
mutations injected into the production branch passed all 15 assertions, because the test never
executed the mutated code. A hand-mirror proves the mirror is right. Now every number in the gibbs
kernel comes out of the two shipped functions, and the only thing the test supplies is the RNG
callback.

This closes a gap `tests/test_gibbs_walk.py` could not: it builds both arrays by hand, so it never
executes the update that `gibbs-walk` reads back to construct its next proposal, and a desync
between them silently corrupts the proposal distribution.

**Mutation battery** — each applied to the shipped file and reverted; counts over the 16 tests in
this file plus the Tier 1b file:

| Mutation | Result |
|---|---|
| `force_accept` in place of the pair-swap MH test | CAUGHT (5 failed) |
| β doubled in the Metropolis probability | CAUGHT (5 failed) |
| `old_e` / `new_e` swapped | CAUGHT (6 failed) |
| `replica_of_window` update omitted | CAUGHT (8 failed) |
| gibbs reverse proposal against **pre**-swap holders | CAUGHT (4 failed) — *previously passed* |
| gibbs `q_forward` / `q_reverse` swapped | CAUGHT (4 failed) — *previously passed* |
| gibbs call site `force_accept` instead of `p_override` | CAUGHT (1 failed) — *previously passed* |
| only one window pair ever offered | passes — control |

The control is the informative one. Invariance is a per-move property, so it is blind to *which*
moves are offered; restricting the pair list breaks ergodicity, not balance, and the irreducibility
test is what guards that.

The last caught mutation needed a different technique. The numeric kernel drives
`gibbs_propose_one_replica` and `apply_window_swap` directly, so it structurally cannot see a
substitution made at the call site *between* them — replacing `p_override=prop.pacc` with
`force_accept=True` still passed all 11 numeric assertions. It is caught by an AST pin in the Tier 1b
file, which asserts the call passes `p_override` from a `GibbsProposal` and never `force_accept`.

Also deleted: 101 lines of `tests/test_exchange_fixes.py` that reimplemented the kernel inside the
test file and asserted a design (`force_accept=True; no additional Metropolis step`) production had
already replaced. It passed against code that no longer existed — coverage in appearance only. The
gibbs hand-mirror above was the same mistake, caught only because the review was adversarial.

## Tier 1b — sample labelling (`tests/test_sample_before_exchange_ordering.py`)

`window_id` is what MBAR uses to pick a sample's bias. A post-swap label would reweight a frame
against a restraint it never felt, and the numbers would stay finite and plausible.

Pinned structurally, since the loop cannot be driven without booting the whole MD driver. Four pins:

1. `sample()` precedes `attempt_exchanges()` — compared by **call line number within the loop body**.
2. `write_sample()`'s `window_id` is bound from the live `assignments` array — at the **binding in
   effect at the call line**.
3. Every site that writes `assignments` is accounted for: exactly one element-wise permutation,
   inside the extracted kernel, and any **bulk** rewrite must be followed by
   `_refresh_replica_of_window()`.
4. The `gibbs-walk` call site passes `p_override` from a `GibbsProposal` and never `force_accept`.

**Six mutations, all caught. Three of them passed the first draft**, and each miss was a different
kind of too-coarse comparison — worth recording, because the pattern is what to watch for in any
AST pin:

| what was compared | how it was defeated |
|---|---|
| `max()` of line numbers **across the whole file** | an extra exchange call ahead of `sample()` left a later one still preceded by a sample |
| **statement index** within the loop body | a `sample()` added inside the *same* `if` block as the exchange shares one top-level statement, so the indices are equal and "later than" is false |
| `any(binding is from assignments)` over the function | a later `w = 0` leaves the good binding in place to satisfy it |

A fourth hole was found by the fixed test itself rather than by review: matching only tuple-target
assignments made the resume path's `assignments[:] = ...` invisible. That write is legitimate — but
only because `_refresh_replica_of_window()` follows it, which is now the stated and enforced
invariant rather than an accident. Scoping matters too: `load_production_checkpoint` builds its own
local `assignments`, so the check is confined to the function that actually owns the driver's array.

## Tier 2 — Boltzmann consistency on production data (`audit_crooks_slope.py`)

The Bennett/Crooks identity in regression form. For two states with overlap,

```
p_k(x) / p_l(x) = exp[(f_k - f_l) - du(x)],    du = u_k(x) - u_l(x)
```

so pooling both states' samples and labelling by origin gives
`P(k|x) = logistic( ln(n_k/n_l) + (f_k - f_l) - du(x) )` — a logistic regression with **slope exactly
−1**, whatever the underlying free-energy surface. The intercept absorbs the unknown Δf *and* the
unequal sample sizes, which is why no reference simulation is needed. It holds with GaMD on, because
the group boost is *k*-independent and cancels from the ratio.

> **This section was rewritten after adversarial verification.** The first version reported
> "28/28 pairs pass, pooled slope −0.98247 ± 0.00463, within 1.8% of the exact null." Every element
> of that sentence was wrong: the 28 pairs were all parallel to one axis, the ± was invalid, 28/28
> was 26/28, and the 1.8% was a modelling artefact. What follows is the corrected analysis, whose
> numbers independently reproduce the verifier's to 3 decimal places.

### The pairing defect — the fits never touched the flagship CV

The grid is 4 CV1 centres × 8 CV2 centres. The old script paired windows by `sorted()` on c1;
`sorted` is stable, so it walked **within** each c1 tie-group. Consequence: all 28 fitted pairs had
**bitwise-identical `(c1, k1)`**, the CV1 harmonic cancelled from `du` to machine precision
(max residual 5.3e-15), and `du` was a pure function of cv2.

**The 28 fits contained zero information about the contact restraint** — the primary CV — while the
docstring advertised detecting exactly that. Three pairs that *did* cross CV1 groups were dropped
silently: their `du` supports are perfectly separated, Newton diverges, the fit returns NaN, and
`if not math.isfinite(slope): continue` printed nothing. "28 pairs fitted" was the output of an
unreported failure.

The identity itself is fine for any pair with overlapping support — pairing affects power and
conditioning, not correctness. But an axis never tested is an axis never tested.

Both pairings are now constructed explicitly, and **reported separately, never pooled**:

| axis | pairs share | varies | tests |
|---|---|---|---|
| CV1 | `(c2, k2)` exactly | c1 | the primary/contact restraint |
| CV2 | `(c1, k1)` exactly | c2 | the secondary restraint |

### Pooling — heterogeneity, not autocorrelation, is what invalidated the interval

Fixed-effect inverse-variance pooling assumes every pair estimates the same slope. Cochran's Q
rejects that decisively on both axes, so the ± is meaningless *however* the per-pair SEs were
computed. DerSimonian-Laird random effects is the headline.

The old "g-corrected" interval was wrong reasoning that landed near a defensible answer by luck: the
true full-trace g ratio is ~11×, not the ~2.6× measured off truncated heads, but a joint block
bootstrap shows the slope is far less sensitive to slow drift than the CV mean, so ×11 would
over-correct. The g probe is retained as a **warning that every interval is a lower bound**, and is
no longer applied as a correction.

Two further refinements, both of which turned out not to change the conclusions — which is worth
recording, since an unchecked approximation that happens to be adequate is indistinguishable from one
that is not:

- **Hartung–Knapp** is reported alongside DerSimonian–Laird. Plain DL treats τ² as known and is
  anti-conservative at these counts (24–28 pairs). HK moves the CV2 interval from ±0.01188 to
  ±0.01212 and leaves CV1 unchanged, so DL was adequate here.
- **Exact χ² survival** replaced the Wilson–Hilferty normal approximation for the heterogeneity
  p-values, because those are reported and compared against a threshold. It moves CV2's p from
  0.0023 to 0.0022 — both far below 0.05, so the "use random effects" call is unaffected.

### Result on `chignolin_6/epoch_000`

Shown at **two equilibration-detection head lengths**, because that choice moves the answer and
hiding it behind one default would misrepresent the precision. Head 8000 is the shipped default.

```
window map   32 rows, identity map, 0 centre mismatch(es)     <- the join is verified
g-convergence  doubling the head changes g by a median factor of 4.0

                              head 4000            head 8000 (default)
median g                      37.1                 53.2

CV1 AXIS (primary/contact) — 24 pairs
  Cochran Q                   70.8 / 23 df         46.7 / 23 df
  p, I^2                      1.1e-06, 68%         0.0025, 51%
  fixed effect (INVALID)      -1.01103 +/- 0.00689 -1.02084 +/- 0.00823
  random effects              -1.01660 +/- 0.01244 -1.01548 +/- 0.01245
  z vs -1                     -1.33                -1.24
  pairs consistent            22/24                23/24

CV2 AXIS (secondary) — 28 pairs
  Cochran Q                   77.9 / 27 df         52.7 / 27 df
  p, I^2                      9.7e-07, 65%         0.0023, 49%
  fixed effect (INVALID)      -0.97618 +/- 0.00643 -0.99645 +/- 0.00829
  random effects              -0.98002 +/- 0.01126 -0.99227 +/- 0.01188
  z vs -1                     +1.77                +0.65
  pairs consistent            26/28                27/28

per-pair gate  |z| > 3.08 (CV1) / 3.12 (CV2), Bonferroni at 5% family-wise
```

Both axes are consistent with the exact null at both head lengths, and the pooled slopes never move
more than ~2% from −1. **Which specific pairs reject changes with the head** (w06-w07 and w29-w30 at
4000; w16-w24 and w30-w31 at 8000), which is itself evidence that the per-pair rejections are
autocorrelation artefacts rather than localised defects.

### The under-resolved `g` cuts *in favour* of the null, not against it

Worth stating plainly, because "every interval is a lower bound" reads as a hedge and here it is the
opposite.

`g` is under-resolved by roughly ×4 per head doubling. An under-resolved `g` means n_eff is
overstated, so the standard errors are too **narrow**, so the test rejects the null too **often**.
Passing under too-narrow errors is therefore a *stronger* result than passing under correct ones —
the analysis is handicapped against itself and the null survives anyway.

The two-head comparison demonstrates it directly rather than by argument: better `g` → wider SEs →
z fell from +1.77 to +0.65 on CV2, and rejections went 26/28 → 27/28. Every step toward a more honest
autocorrelation estimate moved the result *toward* consistency with −1.

**One caveat that keeps this from being a clean a fortiori.** The point estimate moved too
(−0.980 → −0.992), not just the interval, so this is not purely an SE effect — better thinning also
changed which frames enter the fit. The direction of travel is still favourable, but it is evidence,
not a proof by monotonicity.

**The `|z| > 6` gate was invented and far too lax.** At 28 simultaneous tests the 5% family-wise
Bonferroni critical value is |z| = 3.12, not 6. Under the original head-4000 configuration the honest
count is **26/28**, exactly as the verifier found, and it names the same two pairs.

### The two axes give different temperatures — but how significantly is not settled

```
detection head 4000        detection head 8000
cv1  -1.01660 -> 295.1 K   cv1  -1.01548 -> 295.4 K
cv2  -0.98002 -> 306.1 K   cv2  -0.99227 -> 302.3 K
diff -0.0366 +/- 0.0168    diff -0.0232 +/- 0.0172
z = -2.18                  z = -1.35
```

A wrong temperature is **one scalar** and must move both axes the same way, so a genuine split would
exclude a global β error. The split is there at both head lengths — but its significance is **not
robust**: doubling the equilibration-detection head moves it from z = −2.18 to z = −1.35, because
every SE scales with a `g` that the probe shows is still changing by ×4 per doubling.

I am reporting this as **suggestive, not established**. It was tempting to keep the z = −2.18
version, which reads as a clean kill; it is an artefact of the shorter head as much as of the
physics.

What *is* settled regardless of the head: **a single pooled "effective temperature" is not a quantity
this run has.** The old report's "effective temperature of 305.4 K" was a CV2-only measurement read
as though it were global, and the CV1 axis gives 295 K. Only agreement between the axes would license
quoting one number, and they do not agree at either head length.

### The 1.8% was never a measured quantity

`epoch_000` was SIGTERM-interrupted twice. Splitting into 4 time-ordered blocks and refitting moves
the pooled slope by 0.034 (CV1) and 0.043 (CV2) at the default head — **2.7× and 3.6× the
random-effects standard error** — and per-segment refits do not bracket the whole-trace value. Random (non-time-ordered)
strata move it by 0.0008, so this is time structure, not an artefact of the extra intercepts alone.
That systematic appears in none of the intervals above and is larger than all of them.

### What Tier 2 is entitled to claim

> In `epoch_000`, the recorded **(c2, k2)** secondary-restraint parameters agree with the applied
> ones: the fitted deviations are ≤2% and are not distinguishable from zero, so the real statement
> is agreement bounded by the test's own ~2.4% resolution rather than a measured 2% discrepancy. The
> **(c1, k1)** contact restraint — tested with a pairing the
> original script never constructed — is likewise consistent with its recorded values. The window→
> parameter join is verified for this phase. The C1 boost-cancellation premise holds by construction
> (`lower-dihedral` boosts only `PeriodicTorsionForce`/`CMAPTorsionForce`; both restraints are
> `CustomCVForce` in unboosted group 0) and the recorded ΔV does not contradict it. The test resolves
> a mis-scaling of recorded k or β of about **2.4%** at best — 2σ against the random-effects SE of
> ~0.012 this analysis actually defends — and every interval is a lower bound: at g ≈ 255 frames the
> 19.5 ns trace holds ~190 independent samples per window against a folding time orders of magnitude
> longer.

Not "the sampler passes within 1.8% of the exact null."

### Two defects in my own gate code, both real

**The head-truncation bug — a wrong claim in shipped code.** `gareus/mbar_subsample.py`'s docstring
asserted that estimating `t0` from a contiguous head is "the conservative direction: an equilibration
transient lives at the start." That is **backwards**. `detect_equilibration` picks t0 by scanning
*within the series it is given*, so t0 is bounded above by the head length; a transient longer than
the head is truncated, g is fitted on the few points after t0, and comes back far too **small** —
over-keeping frames and overstating n_eff. Measured on this run with a 4000-frame head: 9 of 32
windows pressed against the ceiling, three at t0 = 3760/3860/3808, reporting g = 1.87/2.21/3.17
against full-trace 126/334/396 — over-keeping by 62×/139×/115×.

Fixed: the head now **grows** (doubling, up to the full series) when fewer than `min_remnant` points
survive t0, and `head_exhausted` reports the case where even the full series is too short. Verified
on the affected real windows: window 0 went from g = 39 on a stuck 4000-frame head to a grown
8000-frame head. The residual gap to the uncapped value (384) is real and is why the g probe now
prints a lower-bound warning — uncapped detection is O(n²) and exceeds two minutes for a *single*
window at this trace length, so the cap cannot simply be removed.

**A test that green-lit the property it was supposed to guard.**
`test_the_detection_cap_recovers_the_same_g_far_faster` used AR(1) ρ=0.97 — a process with **one**
short timescale (τ ≈ 66), comfortably inside any head, so the test always passed while the property
failed by 11× on real data. Added
`test_a_transient_longer_than_the_head_does_not_silently_shrink_g`, which puts a 6000-frame
transient through a 4000-frame head; reverting the head-growth fix makes it fail with t0 = 3988 of
4000 — the exact real-data signature.

**And the original gate fix still stands:** `mbar_subsample.py` failed open silently — a bare
`except` returned every index when pymbar was missing or the detector raised, indistinguishable at
the call site from "already uncorrelated". `equilibrated_subsample()` now returns provenance with a
`strict` mode; the legacy `equilibrated_subsample_indices()` keeps its exact contract so the
adaptive-production caller is untouched.

**Also fixed:** `_windows()` read `primary_cv_center`/`primary_cv_k_kcal`, which do not exist in this
schema — it fell through to the legacy `distance_center_A`/`distance_k_kcal_mol_A2` aliases.
Numerically identical here, but those name an Ångström distance while the CV is a dimensionless
contact fraction. Now reads `primary_center`/`primary_openmm_k` (already kJ, no conversion) and
fails loudly on an unknown schema.

---

## Defects found and not yet fixed

Ordered by how much they could hurt.

1. **Non-equilibrium rescue, armed.** `production.py:6780` onward copies positions from another replica
   and reinitialises velocities with **no acceptance test**. Disabled in final production as
   "non-equilibrium", but default **ON whenever CV1 is contacts** — the flagship mode — in epochs
   whose samples reach MBAR. Verified **zero rescue events** across all recent `chignolin_6` jobs, so
   current data is clean. Nothing is written to Parquet, so a consumer could not exclude affected
   frames if it needed to. A prior audit asked for a per-frame `rescued_at_step` marker; it was never
   built.
2. **`gibbs-walk` biases when CV2 is NaN.** Forward and reverse proposals filter candidates over
   different bias rows — `_current_exchange_arrays` (`:6462-6464`) lacks the `np.isfinite` mask `sample()`
   applies at `:6115` — so `q_forward` and `q_reverse` can be normalised over different candidate sets,
   breaking the MH correction. The other three modes fail closed and merely freeze; the one in
   production is the one that biases. **Correction to an earlier draft of this list:** I cited the
   `gibbs_all_nan_skips` counter as evidence that NaNs occur. That inference is wrong — the proposal
   forces the stay candidate's delta to exactly 0.0 before the finite-mask is applied, so as long as
   the replica holds a window there is always one finite candidate and an all-NaN bias row can never
   empty the list. That counter fires only on an entirely unheld window table, which is a different
   condition. The defect itself is unaffected: a NaN removes *other* candidates from the forward and
   reverse sets asymmetrically, which is what breaks the MH correction. Both facts are now pinned by
   tests.
3. **Four C1-unsafe boost types are selectable with no guard**, and
   `examples/chignolin_2d_distance_with_genpept.yaml:136` still sets `gamd_boost_type: lower-dual`.
4. **`gibbs_move_fraction` counts proposals, not accepted moves** (`gibbs_moves` incremented at
   `:6576`, before the MH test; now carries a comment saying so, but the metric is unchanged).
5. **`bias.py:_parse_epoch_window_map_native_params` never consults `secondary_cv.mode`**, so it will
   reconstruct bias energies straight across a CV2 coordinate change without noticing.
6. **Exchange RNG re-seeded identically per segment**, correlating the decision stream across
   segments. Bears on effective sample size, not on per-move balance.
7. **The GaMD boost is pinned at its ceiling.** `k0 = 1.0` against a `k0' = 1.694`, so the requested
   σ0p = 2.5 kcal/mol is not what is being applied and the run sits at maximum boost. This is the
   direct cause of the Layer B verdict in Tier 0, and unlike the rest of this list it is a *config*
   fix, not a code fix: **σ0p ≈ 1.00 kcal/mol**. Note that anything between 1.48 and 2.5 is inert —
   the clip keeps `k0` at 1.0 — so a half-measure here would look like a change and do nothing. Not
   changed here, because it alters the physics of future runs and breaks comparability with past
   ones — that is a decision, not a bug fix.
8. **`tica_cv_version.txt` is absent from every scheduled segment**, so the driver's union-MBAR pool
   is in practice `final` + extensions only — safe by accident for epoch_000 (genuinely different
   CV2), over-conservative for epoch_001.

---

## Where this leaves the original claim

The claim under test was: *"Exchange obeys detailed balance, so the joint ensemble stays correct and
the samples labelled state k still converge to state k's Boltzmann distribution."*

**Proven, with one stated exception.** The exchange kernel preserves the Boltzmann target exactly —
rows sum to 1, πP = π, and full detailed balance to 1e-12 — for the pair-swap family (`neighbor`,
`random-pair`, `all-pair-sweep`) and for `gibbs-walk`, including the `assignments` /
`replica_of_window` bookkeeping that the proposal is built from. Twelve mutations each turn it red;
one control stays green.

**The exception, and it matters because it is the default mode.** That proof is over a **finite**
bias matrix. When a secondary CV value is NaN, `gibbs-walk`'s forward and reverse proposals are
normalised over *different* candidate sets (defect 2 below), so the MH correction no longer uses
matching probabilities and the chain does not target π. The pair-swap modes fail closed instead —
a non-finite Δ clamps `pacc` to 0. So the honest statement is: **balance is proven for all four modes
on finite bias matrices, and is broken for `gibbs-walk` alone when CV2 goes NaN.** The earlier draft
of this section said "for all four modes" without the qualifier, contradicting its own defect list
two paragraphs below.

**Sample labelling** means what MBAR assumes: `sample()` precedes `attempt_exchanges()` in the loop
and reads the live `assignments` array, pinned structurally against six mutations.

**On real production data**, both restraint axes are consistent with their recorded parameters —
23/24 pairs on the contact CV and 27/28 on the secondary at 5% family-wise error, random-effects
pooled slopes of −1.015 ± 0.012 and −0.992 ± 0.012 against an exact null of −1. The two axes
disagree by ~0.02–0.04 in slope — suggestive of a non-global effect, since a temperature error is one
scalar — but its significance is head-dependent (z between −1.35 and −2.18) and is not established. Every interval is a lower bound, and a time-stratification systematic 2.7–3.6× larger
than the quoted errors sits underneath all of them.

**Not yet proven.** That the within-state dynamics under a *live GaMD boost* sample the biased
Boltzmann distribution (Tier 3), and that the whole stack reproduces a known free-energy surface end
to end (Tier 4).

**Layer B.** Exponential reweighting is structurally unusable at this boost width — a = 0.369 >
0.25, infinite weight variance, no amount of sampling changes that. CE2 is neither vindicated nor
condemned: its truncation error is **unresolved**, with a measured floor of 0.23 kcal/mol and two
higher estimators (1.37 parametric, 3.03 empirical) that are respectively model-dependent and
unreliable — the empirical one because it needs the same divergent average. The boost width is a
**setting**, pinned at `k0 = 1.0` against `k0' = 1.694`; **σ0p ≈ 1.00 kcal/mol** (not the 1.4–1.7 an
earlier version of this report gave — that range is still inside the clip and changes nothing) would
put the exponential estimator inside its bound and make CE2's error measurable at all.

---

## Provenance of this document

Every tier was re-checked by an independent adversarial subagent after the first draft was written.
All three found real defects, and this document is the corrected version:

| tier | verdict | what it broke |
|---|---|---|
| 0 | **refuted** | pooled on phase-local `window_id` without the state map; invented the βσ ≲ 1 criterion; treated skew as proof of non-convergence |
| 1 / 1b | **refuted** | the gibbs kernel test was a hand-mirror, so 3 production mutations passed; 3 further structural holes in the ordering pins |
| 2 | **refuted** | all 28 pairs parallel to one axis; invalid fixed-effect pooling; `|z| > 6` gate; wrong g-inflation reasoning; unsupportable "effective temperature" |

The corrected Tier 2 numbers independently reproduce the verifier's to three decimal places,
including the two specific rejecting pairs and the cross-axis temperature split. Two defects in
shipped code (`gareus/mbar_subsample.py`'s head truncation, and the gibbs branch's untestability)
were found only because the verification was adversarial rather than confirmatory.

**A second review pass** then cross-checked the derivations with an independent model and by direct
numerical simulation. It found two more errors, both in the *corrected* version:

| finding | status |
|---|---|
| geometric tail at ratio 2a extrapolates a within-bin ratio onto an across-bin spread | **refuted**; the report's own 4th-order datum (ratio 1.65) already contradicted it |
| σ0p ≲ 1.66 kcal/mol recommendation ignores the `k0` clip and would change nothing | **refuted**; corrected to ≈1.00 |
| `k0'` read positionally from the first glob hit — the diagnostic pilot, not production | **refuted**; 1.766 → 1.694, selection made explicit |
| Tier 0 pooled the pre- and post-recalibration GaMD regimes into one `a` | **refuted**; split, `a` = 0.388 vs 0.369, verdict unchanged |

Verified unchanged on that pass: the moment-matching for `a`, the MGF radius `1/(2a)` and the
0.5/0.25 bounds, the cumulant formula and its `2a` ratio, the logistic slope of exactly −1 and its
insensitivity to `n_k ≠ n_l`, the claim that pairs sharing `(c1,k1)` carry no CV1 information, the
DerSimonian–Laird algebra, the Bonferroni critical value, and every unit conversion. The chi-square
results are now pinned by `tests/test_gamd_reweighting_feasibility_math.py` against draws from a
known distribution rather than taken on trust.

---

# Planned work

## Tier 3 — toy system with exchange AND GaMD (2–3 days, cheap compute)

**Purpose:** the only planned validation of L3 — that the *within-state* dynamics are Boltzmann for
the biased **and boosted** potential. Every existing oracle is REUS-only by construction.

**The trap that must be closed first.** A naive extension of the existing oracles would be vacuous.
Their systems contain only `CustomBondForce`, `CustomExternalForce` and `CustomCompoundBondForce`, so
under the default group boost force groups 1 and 2 are **empty**, E₁ = E₂ ≡ 0, and the boost
formula's guard (`gamd/langevin/base_integrator.py:365-369`) yields **ΔV = 0 exactly** — silently, no
NaN, no warning. The GaMD arm would pass while testing nothing.

**Design.**

- Implement the true well as a **`CustomNonbondedForce`** (NoCutoff, 2 particles), which
  `set_non_bonded_group` matches into group 1 and the boost therefore reads. The pin
  (`CustomExternalForce`) and the umbrella (`CustomBondForce`) stay in group 0, unboosted — which is
  also a direct test of the L2 cancellation premise.
- Freeze `(k0, E, Vmax, Vmin)` as known context globals to bypass calibration.
- The target is then closed-form in one variable,
  `p_k(x) ∝ exp(−β[U_true(x) + ΔV(U_true(x)) + B_k(x)])`, normalisable by 1-D quadrature to ~1e-12.
- Two windows, exchange **on**, GaMD **on**; compare the joint (window, x) distribution to the product
  target.
- Estimator: the **same logistic-slope test as Tier 2**, against exact synthetic draws from the
  analytic target. Not KS — at n_eff ≈ 2000, KS detects only a ≥10% parameter error, whereas a
  one-parameter slope with SE ≈ 1/(σ_Δ√n_eff) is far sharper. Reusing one estimator across tiers also
  means one thing to trust.

**Gates and controls.**

- **Assert ΔV ≢ 0 before trusting any result.** Without this the tier is worthless.
- Mutation control: rerun under `lower-total` and confirm the test **fails**, proving it can see a
  broken boost/bias cancellation.

**Files:** `tests/test_thermodynamic_validity_real_md.py` is the scaffold; the system build at
`:128-144` needs the `CustomNonbondedForce` change.

## Tier 4 — the dedicated run: GaMD-**off** REUS ala-dipeptide

**Why GaMD-off.** A GaMD-reweighting endpoint would re-derive a settled result — Tier 0 already
confirms it analytically, for free. What has *no* real-MD coverage is the thing actually in question:
per-state Boltzmann sampling end to end. The existing oracles are REUS-only toys and the previous
ala-dipeptide campaign was GaMD-on throughout. (Its `umbrella_only = 7.2 kcal/mol` is not an anomaly:
on a GaMD run that estimator ignores the boost by construction. There is **no** GaMD-off datapoint in
that campaign.)

With GaMD off, `umbrella_only` is the **exact** estimator — `u_nk = β·w_k`, no reweighting
approximation anywhere — so any deviation from the reference is unambiguously a sampler, exchange or
coverage defect.

**Setup.** Recover the deleted assets from git `03f4550^`: `validate_ala_dipeptide.py` (160 lines),
`ala_dipeptide_{reference,validation}.yaml`, `run_ala_dipeptide.sh`, `ala_dipeptide_fixture.py`. Then
`run_mode: hmr-cmd`, restore the dropped node `cv2_centers: [-1.0, -1/3, 1/3, 1.0]` (its absence
confounded the original result by under-sampling the dominant basin), keep
`exchange_mode: gibbs-walk`.

**Endpoints.**

- **Primary: the 1-D PMF along cv1**, within **0.2 kcal/mol** of the unbiased reference. It is the
  coordinate the umbrella controls; ~10⁶ frames at τ_int 10–100 ps gives per-bin SE ≈ 0.03–0.1 kT, so
  the threshold is meaningful. Not the φ/ψ map: that is orthogonal to both biased CVs, so a deviation
  there conflates estimator error with orthogonal-coordinate coverage.
- **Secondary: φ/ψ**, restricted to a low-F mask (F ≤ min + 5 kT — `gareus/synth/oracle.py:75` already
  implements `low_f_mask`), reported as RMSD plus per-basin ΔG with block-bootstrap CIs. Never
  max-deviation over the whole map; that statistic is dominated by unsampled high-F corners and is why
  the original numbers were hard to interpret.

**Scope discipline.** One arm. No σ0 scan — it splits statistics and reintroduces the confound Tier 0
already resolved. The 100 ns unbiased reference was lost with the rest of the outputs and must be
re-run; 50 ns per window still clears the precision target.

**Residual gap, to state plainly:** a GaMD-off Tier 4 does not validate the production configuration,
which runs GaMD. That gap is covered only at toy scale, by Tier 3.

## Stopping rule

Tiers 0, 1, 1b and 2 are complete and green. They give exact coverage of the exchange kernel, a closed
hole on sample labelling, and a Boltzmann-consistency check runnable on every existing production run.
**If any of them ever fail, stop and fix before spending the dedicated run — the run is uninterpretable
until they pass.**
