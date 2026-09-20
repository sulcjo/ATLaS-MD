# Adversarial review: automatic equilibrium CV selection

Reviewed document: `atlas_md_equilibrium_cv_selection_spec.md`, version 0.1, dated 2026-09-19.  
Reviewed SHA-256: `dc8cf9ccc0692d55d49e5a9e6089d50ccf5f0bfff4d01f2c920b2e1e325d72b4`.  
Repository scope: the specification’s pinned `main` at `cdee6cc87ac540b0bac9b230f389c77d70383faf`; this is a design review, not a fresh exhaustive repository audit.  
Review evidence: direct reading, mathematical counterexamples, small numerical checks, and primary methodological sources. No peptide simulations or end-to-end selector implementation were run.

## Verdict

**Major revision before implementing the complete selector or launching its proposed competition.** The equilibrium-reweighting foundation is sound. The statistical decision policy, candidate filtering, and several pass/fail gates need correction.

The central design can be retained: native-blind candidates, faithful bias implementation, actual sampling trials, exact ladder-MBAR reweighting, independent confirmation, and permission to return insufficient evidence. However, the current specification puts too much confidence in estimated precision, filters candidates using an imperfect structural proxy, and combines an ambitious statistical claim with very few independent campaigns.

No fatal algebraic error was found in the stated common-target reduced-potential or target-weight formulas. The issues below should not be misreported as evidence that MBAR or a particular ATLaS-MD run is wrong.

## Findings at a glance

| ID | Severity | Finding | Main sections affected |
|---|---|---|---|
| R1 | High | Precision and within-arm reproducibility do not establish thermodynamic accuracy; cross-arm disagreement is not an explicit binding gate | 9–11, 17 |
| R2 | High | The default replication and interval machinery are poorly matched to the claimed 20% efficiency distinction | 8.3, 10, 11.1 |
| R3 | High | The primary uncertainty plan is expensive and under-specified despite an existing joint-sampling MBAR formulation | 10, 12, 16.1 |
| R4 | High | The structural shortlist can systematically discard the useful CV before it receives a sampling trial | 6.1, 7 |
| R5 | High | A hard local gradient-rank gate rejects valid, potentially useful periodic embeddings | 6.3, 11.2, 15.2 |
| R6 | High | Mandatory repeated visits to every coarse cell confuses negligible population with unresolved sampling | 9, 11.2 |
| R7 | Medium | Per-frame weight concentration is dependent on saving cadence | 10.4, 11.2 |
| R8 | Medium | Orthogonalization and asymmetric window allocation impose consequential, arbitrary bias geometry | 6.3, 8.2 |
| R9 | High | Several choices required for an automatic algorithm are still prose placeholders | 6–11, 13 |
| R10 | Medium | The implementation and compute plan are broader than the evidence needed to test the central hypothesis | 8.3, 12–16 |

“High” means correct before treating the selector’s output as confirmatory evidence. “Medium” means a consequential implementation or interpretation risk. These severity labels concern the proposed feature, not existing production correctness.

## R1 — Precision is not accuracy, and different arms can disagree

**What the specification says.** It minimizes the largest normalized confidence-interval half-width, after support and within-arm consistency checks. It also correctly acknowledges that no finite dataset certifies an unseen basin.

**What remains wrong or incomplete.** The operational verdict can still be stronger than the evidence. Independent runs with a shared discovery prior and an unrelaxed hidden degree of freedom can reproduce the same biased population. Small statistical errors quantify the uncertainty within what was sampled, not the missing thermodynamic mass.

An illustrative population estimate:

| Arm | Target truth, known only for this counterexample | Mean estimate | Standard deviation of the arm estimate |
|---|---:|---:|---:|
| A: reproducible trapping | 0.50 | 0.58 | 0.0001 |
| B: adequate exploration | 0.50 | 0.50 | 0.0010 |

The precision-only score favors A by approximately 100-fold in variance. Its mean-squared error is about 6,400 times larger. This is an analytic example, not an observed result from ATLaS-MD. It demonstrates why a variance ranking needs independent evidence that the estimates refer to the same equilibrium distribution.

The present gates explicitly compare independent campaigns and time blocks. They do not unambiguously require agreement between separately implemented, adequately supported arms or an independent thermodynamic reference. Thus two arms can each appear internally stable while reporting incompatible populations.

**Required correction.**

1. Make cross-arm thermodynamic disagreement an explicit diagnostic and a blocking condition for an unqualified equilibrium confirmation when both arms otherwise pass.
2. Do not assume the baseline is correct. If the baseline is visibly unconverged, its disagreement is not proof that the pair is wrong; the result is unresolved until an independent check distinguishes them.
3. Separate output fields: `precision_passed`, `within_arm_reproducibility_passed`, `cross_protocol_consistency`, `coverage_scope`, and `efficiency_advantage`.
4. Use a verdict such as `VALIDATED_ON_DECLARED_OBSERVABLES_AND_SUPPORT`, rather than suggesting that a full molecular distribution has been certified.
5. On synthetic systems, score actual error against known truth. On chignolin, obtain an independent exploration/reference check when feasible; agreement remains evidence, not a proof of global coverage.

The MBAR estimator presupposes adequate equilibrium input; the original estimator does not make unsampled regions identifiable. [Shirts and Chodera](https://arxiv.org/abs/0801.1426)

### Additional counterexample: agreeing marginals

For three binary structural descriptors, put equal probability on even-parity states in distribution P and on odd-parity states in Q. Every one- and two-descriptor marginal agrees, yet their supports are disjoint and `TV(P,Q)=1`.

This is not proof that the proposed full-feature clusters necessarily miss that particular example. It is a control demonstrating that marginal agreement cannot substitute for joint-distribution checks. A finite clustering can also hide differences within its cells.

Add at least one declared multivariate comparison on held-out native-blind structural features, alongside interpretable observables. Do not present that additional test as solving the unseen-support problem.

## R2 — Three or four campaigns do not support the intended efficiency claim reliably

**Location.** The example uses three screening and four confirmation campaigns per arm, simultaneous intervals over a large observable panel, and a desired loss ratio of 0.8.

There are two distinct problems.

### The fallback interval is very conservative

The specified half-width contains

\[
t_{R-1,\,1-0.05/(2M)}\sqrt{v_{\mathrm{check}}}.
\]

Computed critical values:

| Independent campaigns R | M = 8 observables | M = 50 | M = 100 |
|---:|---:|---:|---:|
| 3 | 12.590 | 31.599 | 44.705 |
| 4 | 6.895 | 12.924 | 16.326 |
| 8 | 3.855 | 5.408 | 6.082 |
| 12 | 3.370 | 4.437 | 4.863 |

With four campaigns and 100 primary observables, the multiplier is 16.326. Taking the hull with a narrower bootstrap interval cannot reduce that width. The plan may therefore consume its budget mainly to return insufficient precision.

This is **not** a claim that the Student-t interval is algebraically invalid. For independent, identically distributed normal campaign estimates centered on the target, the sample-variance-based t interval is justified; using the maximum variance diagnostic makes that component wider. The concerns are practicality, non-normality, and shared sampling bias.

### A 20% variance difference is difficult to confirm from four campaign estimates

In an idealized benchmark with unbiased, independent normal campaign estimates, a one-sided 95% upper bound for an observed variance ratio `r` with four estimates per arm is

\[
r\,F_{0.95}(3,3)\approx9.277r.
\]

An observed ratio of 0.8 has an upper bound of about 7.42. The observed ratio must be below roughly 0.108 to exclude equal variance with this particular test. The normal-sample F construction and its distributional assumptions are documented by [NIST](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/ratio_sd.htm).

A numerical benchmark of 200,000 such experiments, with true variance ratio 0.8, found only about **6.8% power** for that one-sided test.

This is a benchmark of the **between-campaign variance route**, not a computed power result for the full proposed MBAR/bootstrap pipeline. Long, well-mixed trajectories can provide additional information through within-campaign error estimation. The correct conclusion is that the actual intended decision procedure needs a power calculation; four outer bootstrap units alone do not establish its sensitivity.

**Required correction.**

- Keep absolute precision and thermodynamic consistency separate from efficiency superiority.
- Predeclare a compact primary estimand panel; use a wider panel for diagnostics rather than automatically assigning every histogram bin the same confirmatory role.
- Validate the actual estimator’s coverage and superiority power on controls before choosing replicate counts and budgets.
- Do not simply split the same budget into more, shorter trajectories. That can worsen equilibration and barrier exploration.
- Permit “adequate pair, efficiency advantage unresolved” without treating it as failure to recover thermodynamics.

## R3 — Use the existing joint-sampling error theory before building thousands of MBAR refits

The cited Li et al. paper does not treat replica exchange only as a speculative extension. Its main text explicitly covers jointly sampled states, including replica exchange umbrella sampling; supplement S2.2 treats joint sampling. This makes it a directly relevant starting point for the estimator architecture. [Li et al., joint-sampling treatment](https://arxiv.org/html/2203.01227v1)

The v0.1 defaults request approximately:

- Screening: `4 arms × 3 campaigns × 200 = 2,400` bootstrap MBAR fits.
- Confirmation: `2 arms × 4 campaigns × 500 = 4,000` fits.
- Total: **6,400** refits, before additional comparison, troubleshooting or calibration work.

This is not necessarily infeasible, but it is not costed. CPU and memory caps do not by themselves bound the wall time of that analysis.

### Recommended estimator architecture

Use a gauge-fixed MBAR estimating-equation formulation. Treat a synchronized sample of the **whole replica ensemble** as one time-series observation. If the parameter estimate solves

\[
\frac1T\sum_{t=1}^{T}g_t(\widehat\theta)=0,
\]

define the sensitivity matrix `J = E[∂g_t/∂θ]` and the long-run covariance

\[
\Omega=\Gamma_0+\sum_{\ell\ge1}
(\Gamma_\ell+\Gamma_\ell^T),
\quad \Gamma_\ell=\operatorname{Cov}(g_t,g_{t+\ell}).
\]

Under the relevant stationarity, mixing and differentiability assumptions,

\[
\operatorname{Cov}(\widehat\theta)
\approx\frac1T J^{-1}\Omega J^{-T}.
\]

Observable uncertainty follows by the delta method or equivalent influence functions. This generic estimating-equation expression is a proposed implementation route; all normalization factors must be reconciled with the repository’s sample/state convention and independently tested.

Practical consequence: fit MBAR, calculate the observable-specific influence time series, estimate their long-run covariance, and retain synchronized block refits as a validation cross-check on selected controls and finalists. Do not estimate correlation only from raw CV1, total energy or unweighted state occupancy: the relevant influence series also contains weight/normalization fluctuations.

Restrict the initial analytical implementation to complete synchronized cycles with the required state-occupancy convention. Asynchronous output, missing states, changing state counts and unequal allocation need an explicit extension; the joint formula must not be applied blindly.

This approach still cannot repair unobserved slow sampling. The same paper reports a molecular example where estimated uncertainty underestimated across-run variability, attributed to difficulty estimating long correlations. Its asymptotic theory is not a convergence certificate. [Li et al., numerical validation](https://arxiv.org/html/2203.01227v1)

**Required correction.** Make the estimator choice a measured engineering decision. Compare the joint analytical route against synchronized block refits on the same synthetic and small molecular datasets. Benchmark runtime, empirical coverage and sensitivity to hidden trapping before mandating 500 full refits per campaign.

## R4 — The offline shortlist is a serious selection bottleneck

The structural score rewards reconstruction of a geometric partition under a chosen discovery distribution. That is not the same objective as reducing error in target equilibrium estimates.

Counterexample family: many broad, rapidly explored conformational variations dominate the discovery variance and cluster labels, while a narrow torsional distinction controls exchange between two thermodynamically important structures. A coordinate that explains the broad variations can rank above the coordinate whose bias would relieve the actual sampling bottleneck.

The original specification acknowledges this distinction, but then retains only two structurally ranked pairs and one random exploration slot. If 494 admissible pairs remain after selecting the top two, one uniformly drawn slot reaches a particular overlooked useful pair with probability only `1/494`, about 0.2%. The actual registry size will differ; the example shows the weakness of relying on one wildcard.

There is a separate evaluation risk: candidates and structural clusters are built from related features and the same discovery design. Held-out folds prevent fitting leakage, but they do not make that structural surrogate an unbiased measure of thermodynamic utility.

**Required correction.**

1. Use geometry to remove implementation-invalid candidates and obvious numerical pathologies; treat most structural rankings as prioritization.
2. On the immediate chignolin companion problem, give all six residual components a bounded actual sampling opportunity, rather than allowing the structural classifier to eliminate five before testing them.
3. For a larger joint search, use a preregistered staged allocation with explicit exploration across candidate families. Keep an independent final confirmation dataset.
4. Calibrate **shortlist recall**: on synthetic systems, how often does the shortlist contain a pair whose actual thermodynamic performance is near the best tested pair? Classifier cross-validation alone does not answer this.
5. Freeze at least some evaluation features or multivariate witnesses independently of the features emphasized in shortlisting.

No correct physical kinetics is needed. The additional information is the mixing and estimator-error behavior of the **accelerated sampler actually used**.

## R5 — The local rank gate rejects useful periodic embeddings

Take a periodic torsion θ and define

\[
z_1=\sin\theta,\qquad z_2=\cos\theta.
\]

Under uniform θ, the covariance matrix is diagonal with entries 1/2. But the gradient Gram matrix is

\[
G=
\begin{pmatrix}
\cos^2\theta&-\sin\theta\cos\theta\\
-\sin\theta\cos\theta&\sin^2\theta
\end{pmatrix},
\qquad \det G=0.
\]

The pair has one local degree of freedom. Nevertheless, it identifies the angle continuously around a circle, while either sine or cosine alone folds different angles onto the same value.

Furthermore, its joint harmonic restraint is an ordinary smooth periodic potential:

\[
\frac{k}{2}\{[\sin\theta-\sin\theta_0]^2+
[\cos\theta-\cos\theta_0]^2\}
=k[1-\cos(\theta-\theta_0)].
\]

Thus rank deficiency does not imply an invalid or useless bias representation. A numerical check on 10,000 angles reproduced zero covariance off-diagonal and determinant zero to roundoff.

This is not a claim that sine/cosine represent two independent intrinsic molecular directions. It is a counterexample to using that requirement as a universal thermodynamic admissibility gate.

**Required correction.** Separate three questions:

- Does the pair encode additional global structural distinctions?
- Does it provide two independent local directions, or a redundant embedding of fewer intrinsic directions?
- Does its actual bias improve target-estimator performance without numerical problems?

Use local rank as a diagnostic/classification. Permit declared manifold embeddings and place centers on their supported manifold. Reject rank-deficient pairs as “not two intrinsic directions” only when that is an explicit requested representation constraint; do not silently equate it with thermodynamic uselessness.

## R6 — Requiring visits to every coarse cell rejects negligible populations

Section 9 makes every coarse cell required. The gate table then requires repeated support in at least two campaigns. Because the cells are geometric and partly bank-derived, one can describe an extremely unfavorable region with negligible target population.

For a simple IID target-sampling example with true cell probability `p = 10^-6` and `n = 1,000`, the probability of zero visits is about 99.90%. Observing zero visits is entirely expected. The one-sided exact 95% upper bound after zero successes is

\[
1-0.05^{1/1000}\approx0.002991,
\]

which is already below the proposed absolute probability tolerance of 0.02. The no-visits condition therefore does not imply inadequate absolute population precision.

Conversely, an unvisited **important** basin has unknown weight and must not receive a zero estimate with zero error. The present rule conflates these cases.

**Required correction.** Give every evaluation region one of three statuses:

| Status | Evidence | Interpretation |
|---|---|---|
| `RESOLVED_POPULATION` | Supported estimate and uncertainty | Report population and supported free energy |
| `MASS_BOUNDED_SMALL` | A valid upper confidence bound below the declared tolerance | Adequate for absolute population precision; a precise free energy may still be unavailable |
| `UNRESOLVED_SUPPORT` | Neither a supported population nor a justified small-mass bound | No thermodynamic clearance |

The binomial bound above applies to the stated IID example. **Do not plug Kish ESS into that formula for general weighted replica-exchange data.** If a suitable bound is unavailable, use `UNRESOLVED_SUPPORT`.

Also harmonize probability and free-energy precision. A ±0.02 population interval around p = 0.01 includes zero and cannot support a finite narrow free-energy interval. Passing one tolerance does not imply passing the other.

## R7 — Per-frame weight caps depend on output cadence

The example cap of 10% maximum weight inside a cell can be passed by recording the same correlated episode more densely.

An exact algebraic illustration: one record carries weight 0.5, with the remaining 0.5 spread over 99 records. Split each record into ten identical copies with one-tenth its weight. Maximum weight drops from 0.5 to 0.05 and Kish ESS rises from 3.96 to 39.6. No information was added.

Literal duplicated observations should already be caught by the identity audit. More frequent recording of almost unchanged configurations creates the same statistical problem without duplicated IDs. The synchronized uncertainty checks help, but the per-frame cap itself remains a poor hard gate.

**Correction.** Keep raw weight concentration as a diagnostic. Add block/episode-level contribution or leverage and leave-one-block/campaign-out sensitivity, with MBAR normalization accounted for. Require precision/support conclusions to be stable under reasonable thinning. Weight ESS is not a correlation-adjusted information count. [PyMBAR weight ESS](https://pymbar.readthedocs.io/en/stable/mbar.html#pymbar.MBAR.compute_effective_sample_number)

## R8 — Orthogonalization changes the available restraint family

The Gram–Schmidt algebra is correct. But applying diagonal umbrellas after the transformation changes the physical bias geometry.

For `z = A(y-μ)`, a diagonal stiffness matrix `Kz` produces, in the original variables,

\[
K_y=A^TK_zA.
\]

For example,

\[
A=\begin{pmatrix}1&0\\-2&1\end{pmatrix},\quad K_z=I
\quad\Rightarrow\quad
K_y=\begin{pmatrix}5&-2\\-2&1\end{pmatrix}.
\]

This is a valid bias, but it is not merely a harmless relabeling of the original diagonal bias. Reproducing the original restraint after a general transformation would usually require off-diagonal stiffness terms.

Likewise, assigning 12 windows to the coordinate named first and 7 to the coordinate named second makes arbitrary ordering consequential. The specification acknowledges that ordering belongs to the protocol, but searches unordered pairs using one ordering. That restricts the optimization beyond “find the two best coordinates.”

**Correction.** Use simple scaling and explicit redundancy diagnostics in the minimum implementation. Treat whitening, role ordering, sparse layout and residualization as parts of the tested sampling design. For general joint selection, test both assignments of coordinate roles when allocations differ, or use a symmetric allocation policy. Do not imply that zero covariance is required for correct thermodynamics.

## R9 — The document is not yet a fully executable algorithm

Several consequential steps are described as policies without a specified algorithm or complete parameters:

| Missing choice | Why two implementations would disagree |
|---|---|
| Exact primitive dictionary and smoothing parameters | Different candidate spaces can produce different winners |
| Cluster algorithm, feature scaling and novelty threshold | Different partitions alter both scores and required support |
| Predictive model and hyperparameter selection | Different nonlinear capacities alter incremental information |
| Ranking order/Pareto policy and tie-breaker | Listing several scores does not specify which wins |
| Equilibration-discard rule | Different retained data change populations and estimated precision |
| Block-length or spectral truncation rule | Different dependence estimates change uncertainty and ranking |
| Definition of plausible thermodynamic relevance | Novelty can trigger rejection unpredictably or never |
| Overlap diagnostic calibration and stopping rule | Deferred calibration can become outcome-dependent tuning |
| Primary observable count and multiplicity family | Confidence thresholds cannot be computed consistently |
| Behavior when two arms fail different gates | A runner cannot infer a justified winner from vague failure labels |

Saying these choices will be calibrated is appropriate for a research proposal. It does not yet make them implementable scientific contracts.

**Correction.** Add a machine-checked `protocol_ready` state. A trial plan cannot be executable until every selection-affecting choice resolves to a versioned algorithm plus parameters and the control suite passes. The readiness check must refuse omitted policies rather than supply silent defaults. Keep a calibration protocol separate from the frozen prospective evaluation protocol.

## R10 — The first implementation is too broad

The proposal introduces 16 modules, a general CV compiler, joint candidate search, several learning models, a new uncertainty framework and a campaign runner before establishing that the selector’s decision rule works on the immediate problem.

There is no scientific necessity to complete all of that before asking whether alternative residual components improve thermodynamic sampling. The generic architecture is a plausible destination, but the first experiment should be smaller.

The 288 GPU-hour arithmetic is correct for the stated four-arm example. It is not evidence that the allocations are scientifically adequate. The document says this correctly; the missing piece is a measured decision-power and analysis-cost budget.

**Correction.** Implement the data/force contracts and a small fixed-primary component comparison first, while explicitly retaining joint selection as the eventual requested goal. Reuse current launch and window machinery through thin adapters. Generalize the primary axis and compiler once a smaller end-to-end study demonstrates valid population estimates and a useful selection decision.

Include setup, burn-in, uncertainty-analysis cost, and expected inability to resolve small performance differences in the budget. Use a short engineering throughput measurement to translate GPU-hours into actual retained samples under the new reporters and CV forces.

## What survives the review

| Design element | Assessment |
|---|---|
| Equilibrium accuracy as the objective | Correct. No requirement to preserve physical kinetics should return. |
| Common-target reduced-potential subtraction | Correct under the explicitly stated shared-Hamiltonian/temperature/pressure/measure assumptions. |
| One GaMD correction through ladder-MBAR | Correct. No additional exponential or cumulant factor belongs on those weights. |
| Exact residual-coordinate and chain-rule force definition | Correct and necessary. |
| Explicit individual-PC identity and feature ordering | Necessary; the known repository conventions must not be silently interchanged. |
| Exact observation/state identity | Necessary, especially across resume and adaptive history. |
| Recording both CVs even in unrestrained states | Necessary for future cross-state evaluation. |
| Excluding adaptation/setup from fixed-state measurement | Sensible conservative initial design. |
| Whole-ensemble synchronized treatment of replica exchange | Correct structural approach to dependence. |
| Native-blind fitting and held-out confirmation | Appropriate protections, provided the resulting claim remains scoped. |
| Ability to return insufficient evidence | Essential; it must also be possible to resolve negligible mass without mandatory visits. |

The review rejects neither two-CV sampling nor residual torsion PCs. It rejects the assumption that the current collection of diagnostic thresholds is already a validated automatic decision procedure.

## Corrected minimum implementation

This is a smaller first milestone, not a replacement claim that joint pair selection has already been solved.

### Step 1 — Validate the thermodynamic and data path

Implement exact observation keys, complete structural observations, frozen state hashes, and runtime-to-MBAR energy agreement. Preserve the canonical boost formula. Add individual-PC import with an explicit feature permutation and residual expression. These are prerequisites regardless of the eventual ranking algorithm.

### Step 2 — Exercise the full decision on known distributions

Before chignolin, run cheap control landscapes with known basin masses. Include hidden barriers, low-variance structural distinctions, circular embeddings, negligible cells, heavy weights, false stationarity, and dependent replica exchange.

Evaluate actual population/free-energy error, interval coverage, shortlist recall, false confirmations and analysis runtime. Do not tune solely for recovering a particular CV index. Freeze the calibrated algorithms before the molecular comparison.

### Step 3 — Test the immediate candidate family directly

For the initial restricted experiment, retain the existing primary contact CV and evaluate its six declared residual components individually, plus the primary-only baseline. All receive a bounded real sampling pilot. A structural score may prioritize execution, but it does not veto a force-valid member of this small family.

The coefficients must be fit on the declared new native-blind discovery set. Replaying old PC2 remains a separate exploratory experiment. No automatic guarantee of selecting PC2 is introduced.

### Step 4 — Estimate error for the actual accelerated sampler

Use a validated joint-sampling influence/covariance estimator when its assumptions and data contract apply. Cross-check selected cases against synchronized block refits and independent campaigns. Where they disagree or correlations are unresolvable, report that limitation rather than averaging away the discrepancy.

For supported observables in an established stationary regime, a useful efficiency diagnostic is

\[
E(P)=\max_a\frac{C\,\widehat{\operatorname{Var}}[
\widehat{\langle A_a\rangle}_0]}{\epsilon_a^2},
\]

where C is measured compute cost. Use equal-cost comparison directly; only extrapolate required cost when approximately inverse-cost variance scaling is demonstrated. This is an efficiency estimate conditional on adequate sampling, not an estimate of unknown systematic bias.

### Step 5 — Compare actual equilibrium results

Require consistency across independent runs, sufficiently separated measurement blocks, and adequately supported competing protocols. Examine a compact primary thermodynamic panel plus broad marginal and multivariate structural diagnostics.

For rare regions, report resolved mass, justified small-mass bounds, or unresolved support separately. Do not calculate a precise free energy from an unsupported zero count.

### Step 6 — Freeze and confirm

Choose a provisional sampler from candidates that pass the integrity and support requirements. Confirm it on fresh independent campaigns. Declare a practical efficiency advantage only when the comparison has sufficient calibrated power; otherwise report thermodynamic adequacy and unresolved efficiency ranking separately.

If reasonable protocols disagree on equilibrium populations, no confidence-interval width competition resolves which is correct. Additional independent exploration is required.

### Step 7 — Expand to joint pair selection

After the end-to-end fixed-primary experiment works, generalize both coordinate slots, candidate families and sparse window placement. Explicitly include coordinate role assignment and orthogonalization policy in the candidate protocol. Re-evaluate shortlist recall and decision power for the expanded search.

This completes the user’s requested joint-selection direction without making a large general compiler and classifier infrastructure the first scientific experiment.

## Required edits to version 0.1

| Priority | Edit | Ready-to-implement condition |
|---|---|---|
| P0 | Scope confirmation to declared observables/support; add cross-protocol disagreement status | Conflicting otherwise-passing arms cannot produce unqualified equilibrium confirmation |
| P0 | Replace rank-deficiency hard veto with geometry classification | The circular-embedding control is evaluated correctly |
| P0 | Replace mandatory cell visits with population/support status | A negligible-cell IID control can pass absolute precision; an unknown important cell cannot |
| P0 | Complete the algorithmic policies and protocol readiness validator | Independent implementations can reproduce the same trial plan and decision |
| P1 | Select an uncertainty architecture using coverage/runtime evidence | Joint covariance and synchronized resampling agree on appropriate controls |
| P1 | Calibrate replication/budget for the actual comparison | Reported superiority power and false-confirmation rate support the chosen defaults |
| P1 | Test shortlist recall, not only structural prediction loss | Useful low-variance candidates are not systematically discarded |
| P1 | Use dependence-aware weight support | Reasonable changes of recording cadence do not manufacture stronger evidence |
| P2 | Expand to joint coordinate/layout optimization | Ordering and whitening effects are explicit tested choices |

The existing specification should remain version 0.1 until these changes are integrated and the revised protocol is clearly distinguished. This review does not silently modify its requirements or claim that the proposed fixes have been validated on molecular data.

## Reproducible numerical checks

The following compact script reproduces the review’s algebraic/statistical checks. It is not an MD simulation or a test of the complete proposed selector. It requires NumPy and SciPy.

```python
import numpy as np
from scipy.stats import f, t

# Small-replicate simultaneous t multipliers.
for R in [3, 4, 8, 12]:
    print('t multipliers', R, {
        M: float(t.ppf(1 - 0.05 / (2 * M), R - 1))
        for M in [8, 50, 100]
    })

# Ideal normal, unpaired, campaign-variance benchmark.
q95 = f.ppf(0.95, 3, 3)
print('F95:', q95)
print('upper bound for observed ratio 0.8:', 0.8 * q95)
print('required observed ratio:', 1 / q95)
print('exact power at true ratio 0.8:', f.cdf(1 / (0.8 * q95), 3, 3))

rng = np.random.default_rng(20260920)
n_experiments, R = 200_000, 4
a = rng.normal(0, np.sqrt(0.8), (n_experiments, R))
b = rng.normal(0, 1, (n_experiments, R))
ratios = a.var(axis=1, ddof=1) / b.var(axis=1, ddof=1)
print('simulated benchmark power:', np.mean(ratios * q95 < 1))

# Negligible population; the confidence bound applies only to this IID example.
print('zero visits probability:', (1 - 1e-6) ** 1000)
print('zero-success upper95:', 1 - 0.05 ** (1 / 1000))

# Matching all one- and two-variable marginals can hide a different joint law.
bits = np.array([[a, b, c] for a in [0, 1]
                            for b in [0, 1] for c in [0, 1]])
p = np.array([0.25 if x.sum() % 2 == 0 else 0 for x in bits])
q = np.array([0.25 if x.sum() % 2 == 1 else 0 for x in bits])
assert np.allclose(p @ bits, q @ bits)
for i in range(3):
    for j in range(i + 1, 3):
        assert np.isclose(p @ (bits[:, i] * bits[:, j]),
                          q @ (bits[:, i] * bits[:, j]))
print('parity total variation:', 0.5 * np.abs(p - q).sum())

# Orthogonal CV values, rank-one local geometry, valid periodic embedding.
theta = np.linspace(-np.pi, np.pi, 10_000, endpoint=False)
z = np.column_stack([np.sin(theta), np.cos(theta)])
gradient = np.column_stack([np.cos(theta), -np.sin(theta)])
G = gradient[:, :, None] * gradient[:, None, :]
print('circular CV covariance:', np.cov(z.T, bias=True))
print('largest determinant residual:', np.abs(np.linalg.det(G)).max())

# Raw concentration can change without added independent information.
w = np.array([0.5] + [0.5 / 99] * 99)
w_dense = np.repeat(w / 10, 10)
print('max weights:', w.max(), w_dense.max())
print('Kish ESS:', 1 / (w @ w), 1 / (w_dense @ w_dense))

# A diagonal restraint in whitened coordinates is not generally diagonal before.
A = np.array([[1.0, 0.0], [-2.0, 1.0]])
print('transformed stiffness:', A.T @ A)
```

These checks establish counterexamples and budget/power concerns. They do not establish which CV pair is best for chignolin. That remains an empirical, native-blind sampling comparison after the corrected data, force and decision contracts are implemented.
