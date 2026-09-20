# In-run automatic selection of an independent (CV1, CV2) pair

Version 0.1 — specification and commentary, 2026-09-20. Nothing here is
implemented; the contracts in `gareus/cv_selection/` are the only landed piece.
Baseline: `cv-select/t00-contracts` at `0dd9623` on `main` `cdee6cc`.

## 1. Goal, and what "independent" is allowed to mean

The user's requirement: one `gareus` submission should **choose both
collective variables and apply them**, with the two CVs as independent as the
data allow — residualised or selected for independence — and no native or
folded reference anywhere.

Three separable notions of independence are in play. The spec must say which
one it delivers, because they are not equivalent and only one is cheap:

| Layer | Definition | Delivered? |
|---|---|---|
| Linear, under the discovery measure `q` | `Cov_q(z2, z1) = 0` | **Yes, exactly, by construction** (§4.1) |
| Nonlinear, under `q` | `z2` not predictable from `z1` by any smooth function: held-out `R²(z2 | z1)` small | **Yes, as a selection criterion** with a declared threshold (§4.2) |
| Statistical, under the equilibrium ensemble | `p_0(z1, z2) = p_0(z1) p_0(z2)` | **No.** Not knowable before sampling; not required for correctness |

Thermodynamic footing for the last row: a bias `B(z1, z2)` leaves the
conditional `p_B(x | z1, z2) = p_0(x | z1, z2)` unchanged whatever the
correlation between `z1` and `z2`. A correlated pair is therefore **not
wrong**, it is **wasteful**: a 2-D window grid over strongly correlated
coordinates spends most of its states on an empty off-diagonal band. So
independence here is an *efficiency* objective — precision at fixed cost — and
must never be confused with a correctness gate. That aligns it with the
selector's stated objective and keeps it out of the pass/fail path.

## 2. Where in the run it happens

The swarm already produces exactly the native-blind discovery dataset the
selection needs: stratified seeds, then ~1 ns of **unbiased** MD per member
(no umbrella, no boost), with torsion features, contact values, Rg and
end-to-end recorded. Selection is therefore a new step **inside swarm
analysis**, between the pooled-trace analysis and the ladder design:

```
epoch 0 (swarm)
  members -> pooled unbiased traces
  -> frozen Pep-GaMD envelope           (unchanged)
  -> [NEW] CV pair selection            -> analysis/cv_pair_model.json
  -> ladder design, now 2-D             -> windows_lambda_ladder.csv (center1,k1,center2,k2,gamd_lambda)
  -> gates (coverage, envelope, ESS, graft, [NEW] pair)
production (adaptive, rung-aware)        reads the frozen pair; never refits it
```

Configuration:

```yaml
cvs:
  cv1: auto          # or an explicit kind, which fixes the anchor and selects CV2 only
  cv2: auto
cv_selection:
  cv1_dictionary: [nonlocal-contact-fraction, radius-of-gyration, end-to-end-distance]
  contact_grid: {r0_a: [10.0, 12.0, 14.0], beta_a_inv: [3.0]}
  residual_degree: 1                 # 2 is a declared alternative family
  torsion_components: [1, 2, 3, 4, 5, 6]
  design_measure: balanced_strata    # equal mass per swarm cell, then per member
  max_nonlinear_r2: 0.20             # held-out R^2(z2 | z1) above this -> next component
  min_resolvable_windows: {cv1: 4, cv2: 3}
  max_gradient_cosine_fraction: 0.50 # flag if |cos| > 0.9 on more than this fraction of frames
  layout: auto                       # joint grid or sparse axis+patch, see 6.1
  fallback: cv1_only                 # or refuse
  tica_switch_cv2: false             # forced; a frozen pair is never redefined mid-campaign
```

`cv1: contacts` with `cv2: auto` is the fixed-primary mode: the anchor is
taken from the config and only CV2 is chosen. That is the milestone the plan
calls `fixed_primary`, and it is the mode to run first.

## 3. CV1 — the anchor

Candidates: the declared dictionary, evaluated on swarm frames with the
declared PBC convention. All are differentiable and already have runtime
forces (contacts) or trivially get one (Rg, end-to-end).

Criteria, computed on the pooled swarm frames under `q`:

1. **Resolvable windows** `n_1 = floor(range_q(a) / (1.5 · σ_w(k_max)))`,
   with `σ_w(k) = sqrt(k_B T / k)` at the run temperature and the configured
   `k_max`. This is the existing `n_resolvable_windows`. Below
   `min_resolvable_windows.cv1` the candidate is not deployable.
2. **Structural coverage**: held-out reconstruction of a coarse geometric
   partition (8 cells, farthest-point on standardised torsion+shape features)
   from `a` alone, by a fixed model class. Reported, used only to break ties.
3. **Numerical**: finite, non-degenerate variance, no PBC discontinuity on any
   swarm frame.

Rule: highest `n_1`; tie → coverage; tie → dictionary order. Deterministic,
recorded with all scores in `cv_selection_report.json`.

Commentary. This criterion exists because of a measured fact: the r7 GENPEPT
library's heavy-atom contact fraction spans ~0 to 0.069, which at
`k_max = 1200` resolves **two** windows, not sixteen. A 16 × 4 ladder over
two resolvable windows is fourteen phantom centres. Choosing the anchor by its
resolvable dynamic range is the cheapest fix for that, and it must be
automatic or the same mistake recurs on the next peptide.

## 4. CV2 — residual torsion component

### 4.1 Construction (exactly the plan's §5.2 contract)

Canonical torsion features `φ(x)` (interleaved sin/cos per torsion, the
runtime order). With `a = (c − μ_c)/σ_c` for the selected anchor `c`, fit by
weighted least squares under `q`:

```
m(a) = b0 + b1 a + b2 a²        (b2 ≡ 0 for degree 1)
r(x) = φ(x) − m(a(x))
```

Diagonalise `Cov_q(r)`; keep individual right singular vectors `v_j`,
`j = 1..6`; store singular values and a tie flag. Candidate coordinates:

```
z2^(j)(x) = ( v_j · (φ(x) − m(a(x)) − mean_q r) − μ_j ) / σ_j
```

Because `m` is an OLS fit with intercept on `a`, `Cov_q(r, a) = 0` exactly,
hence `Cov_q(z2^(j), z1) = 0` exactly for every `j`. That is the linear
independence layer, delivered by construction, not by a later
Gram–Schmidt step.

Runtime force (plan §5.2, non-negotiable): the CV2 umbrella must differentiate
the *whole* expression,

```
∇z2 = (1/σ_j) [ Σ_l v_jl ∇φ_l − (v_j·b1 + 2a v_j·b2) ∇c / σ_c ]
```

so a CV2 restraint also pushes on the contact coordinate through the second
term. **Value independence does not imply force independence.** Dropping the
second term (the current torsion-only `_add_weighted_trig_torsion_force`
pattern) changes the CV being sampled and silently breaks reweighting.

### 4.2 Selection among the six

Per component `j`, on held-out swarm folds split by seed family:

1. **Nonlinear predictability** `R²(z2^(j) | z1)` from a fixed binned
   conditional-mean estimator (10 equal-mass bins of `z1`, no tuning).
   Components above `max_nonlinear_r2` are set aside: linear residualisation
   removed the linear part, but a torsion mode that is a *quadratic* function
   of compaction is still not independent. Degree 2 residualisation is the
   declared alternative when several components fail this.
2. **Incremental structural information**: held-out cross-entropy gain
   `min[L(z1), L(z2)] − L(z1, z2)` on the 8-cell partition and a 24-cell
   partition, same model class for all candidates. This is information about
   the *discovery* partition, not about equilibrium; it is labelled so.
3. **Resolvable windows** `n_2` along `z2^(j)` under `k2_max`; below
   `min_resolvable_windows.cv2` the component is not deployable.
4. **Local geometry diagnostic**: the gradient Gram matrix
   `G = [∇z_i · M⁻¹ ∇z_j]` on a fixed subsample of swarm frames; report the
   distribution of `|cos(∇z1, ∇z2)|` and the small-eigenvalue fraction.
   **Diagnostic, never a veto** (review finding R5: the sin/cos circle
   embedding is rank-one everywhere and perfectly valid). It becomes a flag
   when `|cos| > 0.9` on more than the configured fraction of frames, because
   then the two umbrellas act along one local direction and the 2-D grid is
   effectively 1-D.

Rule: among deployable components passing criterion 1, highest criterion 2;
tie → lowest `j`. **Tie in singular values** (flag set) does not change the
rule — the frozen vector, not the label "PC j", defines the coordinate.

The runner-up components are recorded as `untested_alternatives` in the
report. That is the honest reading of review finding R4: an offline structural
score is a *proposal* and cannot certify sampling efficiency; the spec below
gives the budget-permitting path to actually test them.

## 5. The independence certificate

`cv_pair_model.json` (versioned, hashed — a `PairModel` extension of the
existing `CandidateSet`) records, and `cv_selection_report.json` publishes:

* `Cov_q(z1, z2)` (must be ~0 to floating point) and the design measure used.
* Held-out `R²(z2 | z1)` and `R²(z1 | z2)`.
* The gradient-cosine distribution and the flag state.
* Ranges, quantiles and resolvable-window counts along both axes.
* An explicit sentence: *"Independence is certified under the balanced swarm
  measure; it is not a statement about the 300 K equilibrium ensemble."*

## 6. Application

### 6.1 Two-dimensional ladder design

Spatial states under the 128-replica cap with 4 rungs → ≤ 32 spatial states.
`layout: auto` picks by the resolvable counts:

* `n_1 · n_2 ≤ 32` → **joint grid** at discovery quantiles (e.g. 6 × 4 = 24,
  ×4 rungs = 96 replicas).
* otherwise → **sparse**: 1 bridge (`k1 = k2 = 0`), `n_1` axis-1 windows
  (`k2 = 0`), `n_2` axis-2 windows (`k1 = 0`), and joint patches in the
  best-supported cells up to the cap. Axis-only windows sample the marginals
  and give the 2-D overlap graph a spine; patches close the diagonal.

Stiffness per axis: `σ_window = spacing / 1.5`, `k = k_B T / σ_window²`,
clamped to `[k_min, k_max]` with the clamp recorded. Units: CV1 is
dimensionless, CV2 is standardised (dimensionless); both umbrella constants
are kcal/mol per squared CV unit, and the existing kcal→kJ conversion in
`correctness/bias.py` is reused unchanged.

The λ ladder is unchanged: 4 rungs, λ scales `k0_Total` and `k0_Dihedral`,
threshold pinned at `Vmax`, λ = 0 rung exactly unboosted. **The Pep-GaMD
envelope depends on peptide energies only, not on the umbrella coordinates**,
so choosing a different CV pair does not invalidate the frozen envelope or the
recon calibration. That is the property that makes in-run selection possible
at all.

### 6.2 Runtime force and state identity

One `CustomCVForce` per replica carrying both umbrella terms: inner CVs are
the anchor force (contacts: existing implementation; Rg/end-to-end: bond- or
centroid-based) and a weighted trig torsion force; the outer expression is
`½k1 (z1−c1)² + ½k2 (z2−c2)²` with `z2` written out in full so OpenMM's
automatic differentiation supplies the chain-rule term. Placed in the umbrella
force group, excluded from the Pep-GaMD boost channels (the existing invariant
that `v_pep`/`v_dih` never see umbrella energy).

The state definition embeds the full pair model contents (feature schema,
anchor definition, `b0 b1 b2`, `v_j`, `μ`, `σ`), so `state_definition_sha256`
changes if any coefficient changes; resume against a different model is fatal.
Both CVs are computed and recorded in **every** state, including `k2 = 0`
windows and the bridge, so cross-state bias reconstruction never needs a
forward fill.

### 6.3 Production and analysis

Adaptive production runs as now: rung-aware, new centres replicated on every
rung, rung edges scored by MBAR state overlap, joint (CV1, CV2) overlap
diagnostics (already shipped). `tica_switch_cv2` is forced off under
`cv2: auto`: the pair is frozen at the end of epoch 0 and never redefined,
because a redefinition mid-campaign changes state identity for every later
sample.

Analysis reconstructs `u_nk` from both umbrella terms plus the ladder boost
(`apply_ladder_boost_to_u`, once), reweights to λ = 0, and runs the existing
λ=0-only vs full-ladder PMF crosscheck.

## 7. Gates, abstention, and what a single run can conclude

Pair gate at the end of swarm analysis (added to the four existing gates):

| Condition | Outcome |
|---|---|
| No deployable anchor | `NO_DEPLOYABLE_PAIR`; with `fallback: refuse` the run stops, with `fallback: cv1_only` the *configured* CV1 (not an auto pick) runs alone and the fallback is written into `run_manifest.json` and the decision record |
| Anchor deployable, no component passes | as above, with CV1 = the selected anchor |
| Pair deployable | 2-D ladder written; `Decision` opened with `decision: PROVISIONAL_CHOICE` |

What the recorded `Decision` can honestly say after **one** campaign:
`precision` derived from the primary-panel half-widths; `dependence` and
`reproducibility` from within-campaign diagnostics; `cross_protocol:
NOT_COMPARED`; outcome at best `PROVISIONAL_CHOICE`.

To reach `CONFIRMED_FOR_DECLARED_PANEL` inside one submission:

```yaml
cv_selection:
  confirmation_campaigns: 2      # same frozen pair, fresh seeds and RNG streams
  baseline_arm: false            # true adds a CV1-only campaign at matched cost
```

Two independent campaigns of the same frozen pair give the cross-protocol
agreement check (the R1 guard against a reproducibly trapped arm) and the
between-campaign variance; `baseline_arm: true` additionally answers "did the
second CV buy anything" at the price of a third campaign. Cost scales
linearly; the default is one campaign and an explicitly provisional record.

## 8. Mapping onto existing code and plan tasks

| Piece | Exists | Needed |
|---|---|---|
| Discovery data | swarm members, `DihedralObsBuffer`, contact/Rg/end-to-end | selector-grade rows with exact identity (plan T02) |
| Residual model | `smoke/scripts/cv_hunt.py` (quadratic, scores only), `compute_bootstrap_torsion_pca` (linear, `component` is a **count**) | `models.py` per plan T03 with individual components and stored coefficients; **never** reuse the count semantics |
| Anchor dictionary | contacts force; `n_resolvable_windows` | Rg / end-to-end forces; CV1 ranking (new, small) |
| Runtime CV2 | torsion-only projection | `residual-torsion-pc` mode with chain rule, plan T04 |
| 2-D windows | `load_explicit_2d_window_csv`, joint overlap | 2-D `ladder_design` with `layout: auto` (plan T09 layout compiler) |
| Contracts | T00: `FeatureSchema`, `CandidateSet`, `Decision` with derived statuses | `PairModel` (anchor selection + degree + certificate), `cv1_dictionary` in `NATIVE_BLIND_CV_KINDS` (already contains the three) |
| Analysis | ladder MBAR, λ=0 crosscheck | unchanged |

Order of work: T03 (models) → T04 (force) → 2-D ladder design → wire into
swarm analyze → gates → decision record. Fixed-primary first, then `cv1: auto`.

## 9. Commentary — where this can mislead, stated up front

1. **An offline pick is a proposal.** The structural criteria in §4.2 rank
   candidates by how well they describe the *discovery* distribution. Review
   finding R4 stands: a low-variance torsion mode that controls the actual
   sampling bottleneck can rank below a high-variance irrelevant one. The
   in-run selector therefore records its runner-ups and only a matched-cost
   pilot (`baseline_arm`, or the plan's seven-arm study) can rank by sampling
   performance. The single-run product is `PROVISIONAL_CHOICE`, and the
   record says so.
2. **Zero covariance is not what makes the reweighting right.** §1: the bias
   leaves `p(x | z1, z2)` untouched regardless. Independence is pursued for
   grid efficiency and interpretability. Nobody should read the certificate
   as a thermodynamic claim.
3. **Residualisation is already the transform.** Do not add a Gram–Schmidt
   whitening on top: review finding R8 shows a diagonal umbrella in whitened
   coordinates is a full-matrix umbrella in the original ones, and the
   restraint family changes. Roles are fixed — anchor first — and that
   ordering is part of the protocol identity.
4. **The CV2 force pushes on CV1.** Through the `(v·b1 + 2a v·b2) ∇c` term.
   Expected and correct, but a reader who assumes "independent CVs, independent
   forces" will misread the window placement diagnostics.
5. **The design measure is the swarm, not equilibrium.** Balanced strata over
   a GENPEPT bank plus 1 ns of unbiased dynamics. A pair independent under
   that measure can correlate at 300 K once the folded basin carries real
   weight. Report `Cov` under the production λ=0 samples post hoc; do not
   re-select on it mid-campaign.
6. **Coverage, not the selector, is the binding constraint on chignolin.**
   Two resolvable contact windows at `k_max = 1200` is a property of the
   library and the anchor, and no CV2 repairs it. Automatic CV1 selection
   (§3) is the part of this spec most likely to change the outcome of the next
   run.
7. **Cost.** Selection itself is offline and cheap (minutes on swarm data).
   Everything that turns "provisional" into "confirmed" is a second campaign
   at full cost. There is no cheaper honest path.
