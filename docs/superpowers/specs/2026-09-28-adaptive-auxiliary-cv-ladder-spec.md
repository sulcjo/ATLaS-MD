# ATLaS-MD: adaptive auxiliary CV discovery and Hamiltonian ladder promotion

Status: proposed specification, v0.2 — adversarially revised. Date: 2026-09-28.

Review verdict: viable research architecture; ready for shadow-mode prototyping after API tracing. Not yet an implementation-complete production design or empirically validated folding method. Section 19 records counterexamples, fixes and remaining gates.

## 1. Objective and decision

Improve reference-free peptide conformational exploration beyond the two existing umbrella coordinates. After an epoch, detect structural distinctions unresolved by those coordinates, propose differentiable auxiliary CVs, test their usefulness through short controlled interventions, and promote successful biases into selected Hamiltonian-ladder states.

The primary scientific objective is reliable physical-equilibrium distributions. Chignolin folding is the first benchmark, not a source of labels or structural priors. Correct physical kinetics are not an objective. Discovery is not convergence, and discovering a native-like structure is not proof of its equilibrium population.

The two umbrella CVs remain unchanged by this feature. Auxiliary coordinates add bias terms to ladder states; they do not add umbrella-grid dimensions. Initial scope is the existing GaMD ladder plus one auxiliary scalar bias family. Torsional gREST and umbrella weakening are explicit subsequent experimental options, not prerequisites or silently combined changes.

## 2. Repository grounding and implementation boundary

Repository: https://github.com/sulcjo/ATLaS-MD . Code inspection reference: `f377dc16f70dd5733acf801c306f68492cdbaa71` (search-returned main snapshot). Architecture and adaptive-workflow documentation were separately read from main on 2026-09-28. Recheck HEAD before implementation.

Observed integration points:

| Existing component | Observed responsibility | Required extension |
|---|---|---|
| `gareus/query.py` | Segment-aware sample loading and umbrella/ladder reconstruction API | Model-indexed auxiliary observations and strict eligibility checks |
| `gareus/correctness/bias.py` | Canonical reduced umbrella-plus-GaMD reconstruction; missing-coordinate handling | Add auxiliary terms through a shared immutable Hamiltonian description |
| `gareus/mbar_analysis/cv2_reprojection.py` | Historical CV-model loading and exact-step reprojection | Reuse provenance lessons, not its regime-splicing fallback for this feature |
| `gareus.cv_selection.models.PairModelRuntime` | Referenced by the inspected reprojection code for existing residual CVs | Inspect and reuse compatible feature/model infrastructure before introducing another schema |
| `adaptive_production`, `production`, `forces`, `store`, `checkpoints`, `provenance` | Documented orchestration, runtime and persistence modules | Epoch hooks, force deployment, atomic promotion and resume support |

This is a scientific and software design spec, not a completed line-by-line implementation plan. Runtime internals outside the inspected files still require tracing. No claim is made that arbitrary auxiliary Hamiltonians are already supported.

The inspected reprojection module documents the danger of cross-evaluating a new CV definition using old scalar values and of losing most samples through sparse reprojection. This feature MUST fail closed for unsupported pooled analysis: a reported coordinate splice is not a substitute for evaluating the correct Hamiltonians.

## 3. Non-negotiable contracts

1. No native structure, native contact list, folded/unfolded labels, known turn positions, or supervised structure predictor enters discovery, ranking, trial selection or promotion.
2. Full sequences and force-field chemistry are allowed. Generic stereochemistry, hydrogen-bond eligibility and nonlocal-contact definitions are allowed.
3. Production Hamiltonians and CV models are immutable within a segment. Updates occur only at an epoch boundary with a new identity and re-equilibration.
4. Forces, exchange, NPT moves and analysis evaluate the same potential. Finite-timestep accuracy remains a separate numerical validation requirement.
5. An inability to find a candidate returns `insufficient_evidence`; it never fabricates a new direction.
6. Low correlation, large variance, high exchange acceptance, or repeated movement in the forced CV alone cannot justify promotion.
7. No discovery branch, biased restart selection, or adaptive-training interval is automatically equilibrium production.
8. Final production budget is reserved in advance. Discovery cannot consume it through repeated retries.

## 4. Epoch controller

State sequence:

`seal_epoch -> audit_inputs -> diagnose -> propose -> validate_models -> trial -> decide -> stage_regime -> equilibrate -> sample`

Allowed decision results:

| Result | Meaning | Action |
|---|---|---|
| `promote` | Validated candidate improves held-out exploration | Install bridge states and frozen bias for next epoch |
| `keep` | No convincing improvement | Retain current regime |
| `broaden` | Insufficient diversity to infer an unresolved rearrangement | Allocate bounded generic exploration |
| `topup` | Evidence exists but precision is inadequate | Continue eligible existing lineages under unchanged Hamiltonians |
| `blocked` | Missing identity, force validation, cross-energy or budget requirements | Retain regime; emit actionable reason |

Run diagnosis after every configured epoch; do not interpret an elapsed epoch or absence of a recognizable fold as a failure criterion. Trigger candidate trials from reproducible hidden structural groups, strong initial-condition dependence, or stalled structural mixing under adequate observation coverage. A 2D PMF appearing stable does not override hidden-structure diagnostics.

The final-pool boundary freezes discovery. Later discovery requires an explicit new phase, not mutation of the completed production pool.

## 5. Inputs and observation identity

Each observation needs an immutable key:

`(run_id, segment_id, context_id, absolute_step, observation_phase)`.

Also retain walker/lineage ID, state ID, Hamiltonian hash, topology/atom-map hash, box, physical time, current two CVs, source eligibility and exchange history. Define whether samples are before or after exchange; do not infer this from timestamps.

Join features, frames and energy rows by the full key, never by step alone, floating-point time or nearest-neighbor interpolation. Enforce one-to-one joins. Restart tails beyond committed checkpoints remain excluded.

Two observation tiers:

* Sparse discovery observations: sufficient for fitting and diagnostics, using bounded stratified reservoirs. Sampling probabilities and strata are retained; balanced exploration data are not labeled equilibrium samples.
* Thermodynamic observations: sufficient to evaluate every state admitted to the intended MBAR solve. Features required by active models must be recorded at every thermodynamic sample, or recoverable from exact matching snapshots.

Existing histories without this information can support limited discovery but do not automatically qualify for cross-regime MBAR.

## 6. Feature dictionary

MVP dictionary contains backbone sin/cos torsions and smooth residue-pair contacts. Add smooth backbone hydrogen-bond descriptors once their energy/force implementation passes validation. Hydration and side-chain torsions are extension families.

| Family | Definition requirements | Guard |
|---|---|---|
| Backbone | Ordered sin/cos of valid phi/psi angles | No raw-angle discontinuity; deterministic terminal handling |
| Contacts | Smooth distance switching for topology-defined nonlocal residue pairs | Fixed pair IDs, units, exclusions and PBC convention |
| H bonds | Smooth distance and optional angular switching for eligible donor/acceptor pairs | Chemistry-based eligibility, no native pair selection |
| Side chains | Ordered sin/cos chi angles | Handle symmetry-equivalent atoms explicitly |
| Hydration | Smooth water coordination of defined backbone groups | Full solvent data needed for retrospective evaluation |

Store every switch parameter, atom selection, normalization and convention in a hashed feature schema. Define whole-peptide imaging and minimum-image rules consistently in offline and runtime implementations. Remove effectively constant features using training data only. Use robust scales with a noise floor and family weighting so a large contact dictionary does not dominate merely by column count.

Do not silently change the dictionary with conformation. Feature schema changes create a new version and explicit cross-evaluation requirements.

## 7. Discover unresolved structural distinctions

### 7.1 Partition before learning

Separate independent runs or independent seed lineages for training and validation whenever possible. Otherwise use contiguous blocks with exclusion gaps; label their weaker independence. Exchange-coupled replicas are not independent experiments. Never randomly split neighboring frames and call that held-out validation.

Fit all preprocessing, local neighborhoods, cluster definitions and projections using the training partition. Retain a frozen validation partition and a separate trial seed set.

### 7.2 Conditional neighborhoods

Let `s=(s1,s2)` be the existing umbrella pair and `d(x)` the descriptor vector. In supported local neighborhoods of s, inspect structural diversity and alternative contact/torsion patterns. Neighborhood bandwidths are selected on training data and checked for stability across a modest bandwidth range.

Optionally fit a regularized smooth conditional mean `m(s)` and use residuals:

$$r(x)=d(x)-\widehat m(s(x)).$$

Conditional-mean subtraction removes predictable means, not all dependence. Residual covariance can still depend on s; zero correlation is not independence. Residualization is a discovery operation in MVP. The deployed CV may be a sparse projection of original smooth descriptors, avoiding derivatives and extrapolation of m, but MUST be retested in that exact form. A residual-space candidate does not transfer its properties automatically to a raw-space projection. If deployment restores primary-CV dependence and loses hidden-state discrimination, reject it or deploy a fully specified smooth residual model with the complete chain-rule derivative. Do not silently substitute one form for the other.

Use two labeled views: a balanced exploratory view and, when overlap/ESS supports it, a physical-target-weighted view. Fit neither on uncontrolled mixtures of state populations. Check whether apparent groups are explained by state/rung, source lineage, box handling or preprocessing. State or seed predictability is a confounding diagnostic, not an automatic rejection: genuine trapped basins can be perfectly lineage-associated. Use matched-Hamiltonian seeded probes to distinguish an artifact from real persistent structural differences. Compare groups within overlapping s regions and comparable Hamiltonians. Weight clipping is allowed only as an explicitly labeled exploratory sensitivity analysis, never as an unreported thermodynamic correction.

### 7.3 Clusters and persistence

Find reproducible groups in local residual or descriptor space using a fixed, versioned clustering method. Report cluster stability under block/lineage bootstrap and nearby hyperparameters. A single clustering output is not evidence of separate metastable basins.

Measure dwell and transitions only on continuous, identity-correct trajectory intervals. Censor at Hamiltonian changes, exchange discontinuities, restart discontinuities and missing observations. Follow configurations rather than concatenating state occupants. If unchanged-Hamiltonian intervals are too short to measure persistence, report it unavailable and run fixed-Hamiltonian probes; do not infer persistence across the gaps.

Persistence describes the biased sampler, not physical kinetics. Frame reweighting cannot by itself recover unbiased path probabilities or kinetic timescales.

## 8. Candidate construction and selection

Produce at most three candidates per cycle by default:

1. Sparse regularized discriminant between reproducible local structural groups.
2. A stable residual-covariance mode, retained only if it resolves more than rapid local noise.
3. A simple physically interpretable descriptor or small combination suggested by the same evidence.

Candidate form:

$$z(x)=b+\sum_j a_j\widetilde d_j(x),$$

where all scales and coefficients are frozen. Model IDs include schema, coefficients, offsets, preprocessing, training manifest and code-version hashes. Sign conventions are deterministic. For nearly degenerate eigenvectors compare subspaces and predictions rather than rejecting harmless sign/rotation differences.

Candidate gates:

* Distinguishes held-out structures at matched s more effectively than a predictor using s alone.
* Stable under block/lineage perturbations and not dominated by negligible-variance normalization.
* Has physically interpretable feature loadings and supported observation range.
* Can be evaluated and differentiated consistently online/offline.
* Has acceptable estimated force and runtime cost.

Rank eligible candidates by held-out conditional discrimination, persistence evidence and complementary structural coverage. Report separate metrics rather than claiming an arbitrary weighted score is a physical optimum. Candidate redundancy is measured on held-out data; prefer distinct mechanisms.

Statistical novelty does not require strict Cartesian-gradient orthogonality. A motion coupled to the original CVs may still be valuable. Optional gradient-coupling diagnostics can report this, but MUST NOT project forces onto a local orthogonal subspace unless a consistent conservative potential is separately derived.

## 9. Trial bias construction

MVP uses an immutable, smooth, bounded one-dimensional proposal bias. Density-derived construction is allowed only with an explicit source-distribution contract. Let p_ref(z) be an adequately sampled marginal under the SAME no-auxiliary Hamiltonian used as the reference for a trial, or a supported reweighted estimate of that marginal. Then a partial-flattening proposal is:

$$B(z)=\alpha RT\log\frac{p_{\rm ref}(z)+\epsilon}{p_{\rm ref,max}+\epsilon},\quad 0<\alpha<1.$$

In the ideal uncapped, epsilon-free case under that reference, the marginal becomes proportional to p_ref(z)^(1-alpha). Alpha is a flattening fraction, NOT the OPES well-tempered bias-factor convention. Epsilon has the same density units as p_ref and must be stored with normalization and bandwidth.

A balanced mixture across windows/rungs is NOT p_ref. A target-unbiased marginal is not the marginal of an umbrella-plus-GaMD rung either. Such distributions can suggest exploration features but must not be labeled a flattening estimate for that rung. Unknown mixture-to-reference conversion or unreliable ESS disables density-derived deployment; it does not license a tiny pseudocount as a substitute for data. Estimates with separately trapped trajectories cannot infer relative basin populations from arbitrary trajectory counts.

Permit local exploratory training from matched-Hamiltonian starts, including adaptive bias training as an optional backend, followed by export and freezing. Training remains ineligible for stationary MBAR. If no supported bias can be constructed in the allocated budget, keep the regime or take the separately configured broader-exploration action.

The fitted domain, pseudocount, bandwidth, energy cap, slope bound and tail continuation are mandatory model parameters. Smoothly return the exploratory bias to a common neutral value outside the supported range through a declared transition region; ensure continuity of energy and force and validate the taper. Do not leave a maximal negative-bias plateau over all unknown space. Test the transition region for a new artificial barrier; a neutral tail prevents an unbounded novelty reward but does not guarantee good exploration. No hard reflecting walls or configuration-dependent rejection may truncate physical support without a separately specified target.

Store the final spline/analytic potential, not just its fitting recipe. Training-density support is neither a physical forbidden region nor evidence that missing density is an energy barrier. The bias can be reused on other rungs as a precisely defined arbitrary potential only after local and full-ladder trials; no universal flattening claim transfers with it.

Do not clip force components independently of the potential. Differentiate the actual capped/continued energy. New extrema, spline overshoot, out-of-domain force growth and boundary artifacts are deployment gates.

An OPES training backend is optional later: adaptive OPES intervals remain exploration; export and validate an immutable bias for frozen production. No assumption that instantaneous OPES bias logging makes ordinary stationary MBAR valid.

## 10. Controlled intervention trials

Compare baseline and each candidate with equal assigned node-hours, matched starting structures and comparable stochastic replication. Include training and analysis overhead in cost reporting. Use the same fixed GaMD envelope and umbrella design across arms; auxiliary bias is the only initial intervention. Match initial conditions without pretending identical RNG streams make trajectories independent.

Prototype maximum: three candidates, at least four matched starting structures spanning available groups, two velocity/noise realizations per structure and arm. These are engineering starting values, not convergence thresholds. If the budget cannot support a meaningful comparison, reduce candidates or return `insufficient_evidence`.

Before these isolated trials can justify promotion, run a reduced exchange campaign with the proposed bridge states, matched total active replicas and node-hours against baseline. Include parameter-update and Context scheduling costs. Demonstrate that new structural groups visit low-bias/zero-auxiliary states through actual exchange and propagation. Endpoint transplantation tests relaxation only; it is not evidence of ladder transport.

Trial outputs:

* Bidirectional passage between previously separated structural groups.
* Diversity and coverage in a frozen descriptor set beyond the forced coordinate, with feature-family leave-out sensitivity checks.
* Reproducibility across trial lineages.
* Common-anchor continuation results from candidate discoveries and matched baseline endpoints.
* Runtime overhead, auxiliary-force distribution, geometry failures and support excursions.

Anchor continuations test structural survival and accessibility under a common Hamiltonian; selected endpoints are not equilibrium population estimates. Use standardized endpoint selection for every arm. Mere relaxation survival is insufficient for thermodynamic relevance, and transient valid intermediates are not automatically failures.

Promotion requires predeclared practical improvement in at least one primary mixing/coverage metric with uncertainty assessed over independent trial units, no clear deterioration in the other primary metrics, and all correctness gates. Use a fresh confirmatory comparison for the selected winner to reduce selection bias. Report confirmation separately from winner-selected screening. At minimum, predeclare one primary metric, a positive practical effect margin, a harm margin for each secondary metric, trial duration, independent unit, and uncertainty method before screening. These are mandatory user/configuration inputs or values fixed by an independently documented calibration campaign; no production defaults are established here. Continuous monitoring cannot silently change the stopping rule. When intervals are inconclusive, keep or top up. Never promote solely on the number of forced-CV crossings.

## 11. Hamiltonian integration

For window w and rung r at common thermostat temperature:

$$U_{w,r}(x,V)=U_{\rm phys}(x,V)+\eta_r W_w(s(x))+
\Delta V_r(x,V)+\sum_a c_{r,a}B_a(z_a(x)).$$

MVP fixes `eta_r=1` and retains the existing GaMD definition. Auxiliary terms stay outside the energy channels used to calculate GaMD unless an explicit later design defines and validates another composition. Accidental boosting of the auxiliary force changes the intended Hamiltonian. Before runtime deployment, enumerate every CustomIntegrator force-group expression and native force contribution. Prove that the auxiliary gradient appears exactly once with coefficient c, outside the GaMD chain-rule multiplier. An energy reconstruction test alone cannot validate propagation. If the existing integrator cannot express this separation, runtime promotion is blocked until its force wiring is extended and validated.

Require an anchor with no auxiliary bias and preferably zero GaMD boost for diagnostic comparison. The physical target for analysis also removes the umbrella; an umbrella anchor is not itself the unbiased ensemble.

Introduce coefficients through intermediates, initially trialing `c = 0, 0.25, 0.5, 1` as a proposal, not a universal ladder. Tune using measured state overlap and structural transport. Existing and new bias families must have bridges; arbitrary adjacency of state IDs is insufficient. Limit to one promoted auxiliary family initially. Subsequent families can occupy different branches/rungs without a Cartesian product of every auxiliary CV.

For symmetric pair proposals, evaluate all four full state-dependent energies:

$$\log A=-\beta[U_j(x_a,V_a)+U_i(x_b,V_b)-U_i(x_a,V_a)-U_j(x_b,V_b)].$$

Include the proposal ratio for nonuniform exchanges. CV models and bias histories belong to thermodynamic states, not traveling configurations. NPT trial volume moves must re-evaluate auxiliary CVs/biases on trial coordinates/box using the same Hamiltonian and existing validated proposal Jacobian.

Before promotion, construct the complete next state table within a fixed replica/Context and node-hour cap. Adding four auxiliary strengths to every current state is prohibited as an implicit default. Select or reallocate a bounded set of exploratory rungs; report displaced sampling effort, full-graph connectivity and worst-edge overlap. New-state count and residence-time allocation are part of the reviewed trial design.

Optional next-stage experiments: weaken umbrellas continuously on exploratory rungs, or add torsional gREST through a separately validated energy partition. Neither is automatically activated by a failed CV trial. Record duplicate Hamiltonians if zero umbrella strength makes formerly distinct windows equivalent.

## 12. Thermodynamic analysis and model history

MVP default inference uses fresh equilibrated final-pool samples under one frozen regime. A new file/segment or a fixed discarded percentage does not establish equilibration. Use independent diverse starts, predeclared burn-in sensitivity checks, between-lineage structural-distribution comparisons and continued physical-target diagnostics. Start with a zero-GaMD, zero-auxiliary anchor per retained umbrella region where feasible, or explicitly justify omitted anchors by observed overlap. If relaxation remains unresolved, mark thermodynamic validity inconclusive rather than declaring the retained tail equilibrium. Discovery/training/trial rows remain excluded regardless of whether their instantaneous energies are known. Eligible earlier fixed segments may be included only through an explicitly validated pooling mode.

For every sample n and state k admitted to a pooled MBAR solve, evaluate the correct complete state-dependent energy. Save either:

* a fixed descriptor basis sufficient for every admitted model; or
* matching coordinates and topology sufficient for later exact feature recomputation; or
* complete cross-state reduced energies for a predefined state set.

Snapshot precision is also a contract: compressed coordinates with an exact step ID are not exact coordinates. Validate reconstructed CVs and cross-energy differences against runtime observations, with tolerances on reduced energies appropriate to the intended estimator. Use directly recorded features/energies or lossless snapshots where lossy reconstruction fails, especially for steep biases.

For a common-temperature, common-pressure, common-physical-potential model, a state-independent per-configuration term may be subtracted from all reduced potentials INCLUDING the physical-target column. In that representation the physical target has zero state-dependent bias; do not later attach an inconsistent full-energy target. Multi-temperature or altered-base-potential extensions must explicitly rederive this cancellation.

The active scalar z alone is insufficient for future models. Peptide-only coordinates cannot reconstruct hydration descriptors. Adding a new feature family must first audit historical evaluability.

Missing entries are never zero-filled or replaced by native-regime CV values. Reject the requested pool, or use an explicitly selected, justified complete subset with exclusions and state counts reported. Configuration-dependent missingness requires separate bias assessment; completeness alone is not enough. Disconnected ensembles cannot be aligned by independently normalized weights.

Report overlap, weight concentration, per-region ESS and between-run uncertainty. ESS from importance weights alone does not account for time correlation. Exact Hamiltonian bookkeeping neither fixes poor overlap nor discovers unsampled minima.

## 13. Artifacts and proposed interfaces

New package proposal: `gareus/auxiliary_cv/`, separating observation, discovery, model validation, trial analysis and immutable bias evaluation. Names are proposed, not existing API claims.

Each epoch produces:

| Artifact | Required content |
|---|---|
| `observation_manifest.json` | Source hashes, full join keys, selection probabilities, coverage and eligibility |
| `feature_schema.json` | Atom maps, feature ordering, switches, scales and units |
| `discovery_report.json` | Conditional groups, uncertainty, provenance-confounding checks and decision |
| `candidates/<id>/model.json` | Immutable differentiable model and training references |
| `candidates/<id>/bias.json` | Final energy function, derivatives, domain and tail behavior |
| `trials/<id>/manifest.json` | Arms, starts, seeds, budgets, Hamiltonians and exclusion status |
| `promotion_decision.json` | All gate results, confirmatory evidence and rejection reasons |
| `next_regime.json` | Full state table, exchange graph, hashes and cross-evaluation coverage |

These artifacts supplement existing segment/window manifests; they do not create a second conflicting state registry. Promotion is atomic: validate staged artifacts, checkpoint old regime, commit new regime, then propagate. A crash before commit resumes the old regime; a crash after commit resumes the exact new one. Missing hashes/models block resume.

Proposed configuration (not yet executable):

```yaml
auxiliary_cv:
  mode: shadow                 # off | shadow | trials | promote
  keep_umbrella_pair: true
  feature_families: [backbone_sincos, residue_contacts]
  max_candidates: 3
  max_active_auxiliary_families: 1
  discovery_budget_fraction: 0.10
  final_production_reserve_fraction: 0.50
  analysis_scope: frozen_final
  bias_backend: frozen_smooth_1d
  require_confirmatory_trial: true
  on_insufficient_evidence: keep
```

Budget fractions refer to the total assigned campaign budget, not each remaining balance. Reconcile the final reserve with the existing adaptive final-pool setting; contradictory configurations fail validation. The 10% discovery allocation covers observation, fitting, fixed-Hamiltonian probes, screening, confirmation, transport pilots and setup overhead together. Four arms times four starts times two realizations already means 32 trajectories before confirmation or ladder pilots. The controller must calculate feasibility from measured cost before scheduling; reduce candidates or abstain if adequate durations do not fit. Broader exploration is available through an explicitly budgeted policy. Enabled discovery is opt-in; old configurations preserve behavior.

## 14. Runtime and resource limits

Compute rich descriptors offline from immutable snapshots where possible. Stream statistics and bounded reservoirs; avoid unbounded frame-by-feature matrices. Only deployed CVs run every integration step. Batch cross-state auxiliary evaluations and reuse shared descriptors, while preserving Context ownership and scheduler barriers.

Benchmark force overhead under the actual multi-Context GPU configuration, not a single-context microbenchmark alone. Report retained production budget and effective structural exploration per node-hour. No throughput gain is promised: extra CV work can reduce ns/day while improving useful exploration.

## 15. Validation gates and adversarial cases

| Case | Required outcome |
|---|---|
| Hidden double well with identical existing CVs | Detect distinction; intervention improves passage |
| Large fast spectator fluctuation | Variance alone does not promote it |
| One observed basin | Return insufficient evidence; no invented folding CV |
| Candidate predicts rung/seed label only | Confounding flag; demand independent support/probes |
| Rare structural basin | Report uncertainty; do not erase merely for low count |
| Exchange or restart jump | No false molecular transition |
| Sparse history or new hydration model | Block unsupported pooled cross-evaluation |
| Changed atom order, units, periodic imaging | Reject model or reproduce validated identical result |
| Huge normalized coefficient or spline tail force | Block deployment |
| Bias moves z but not other structure | Fail usefulness gate |
| Missing model on resume | Fail closed |
| High swap acceptance but no structural mixing | Do not label converged |

Numerical tests: analytic gradients versus finite differences; online/offline energies and CVs; PBC-equivalent images; zero-coefficient legacy equivalence; all four exchange energies versus direct evaluations; NPT auxiliary-energy inclusion; restart identity; units at kcal/kJ boundaries; strict missing-data propagation.

Small-system thermodynamic tests use a known multidimensional distribution with a hidden barrier. Recover equilibrium populations within independently estimated uncertainty at multiple bias strengths; separately assess timestep sensitivity. Test adaptive discovery followed by fresh frozen sampling, and explicitly reject naive pooling of adaptive trials.

## 16. Chignolin benchmark and acceptance

Lock sequence, terminal chemistry, force field, solvent, temperature and integrator across arms. Distinguish wild-type chignolin from CLN025 in all reports. Use independently generated non-native starts and record any accidentally native-like seeds after held-out evaluation.

Matched-node-hour arms: existing GaMD+2D-US baseline; adaptive auxiliary procedure including its discovery/trial overhead; optional separately budgeted gREST comparator. Use at least three independent campaigns per arm as an initial feasibility benchmark, expanding only when uncertainty prevents a decision.

Native structure is unavailable to adaptive code. Held-out analysis may quantify native-like discoveries, repeated folding/unfolding, and physical-target folded population after the method is frozen. Also report reference-free basin populations and mixing, so success is not reduced to one RMSD threshold.

MVP acceptance requires all correctness tests plus demonstrable improvement in hidden-structure exploration on the controlled model. Add three distinct test landscapes: energy-separated wells, an entropic narrow passage between otherwise distinguishable basins, and a degenerate scalar projection with separate basins at the same z. Good classification alone must not pass the latter two. A one-dimensional potential cannot preferentially change relative energies between configurations with identical z; require intervention evidence or a different CV/tempering strategy. Peptide efficacy remains experimental until independent campaign results show improvement at matched cost without degraded thermodynamic consistency. Failure to fold does not by itself invalidate equilibrium bookkeeping; successful folding does not establish convergence. Force-field error remains distinct from sampling error.

## 17. Delivery stages

1. Shadow observation and diagnosis; no force changes.
2. Offline candidate generation, immutable model export and force/energy validation.
3. Isolated controlled trials with audited budgets and confirmatory comparison.
4. Auxiliary-state integration through the shared Hamiltonian evaluator, exchange, NPT and checkpoint paths.
5. Frozen-final analysis and strict optional cross-regime evaluation.
6. Independent peptide benchmarking; only then enable automated promotion by default for validated configurations.

Defer deep neural CVs, physical kinetic reconstruction, unrestricted active feature expansion, continuous production bias adaptation and automatic combinations of several new acceleration methods.

## 18. Methodological references and scope

The workflow above is a proposed integration, not an established algorithm with demonstrated ATLaS-MD performance.

* Tiwary and Berne, SGOOP: https://doi.org/10.1073/pnas.1600917113 . Precedent for optimizing coordinates using preliminary biased sampling and additional dynamical information; not a guarantee that unobserved states can be inferred.
* Ribeiro et al., RAVE: https://arxiv.org/abs/1802.03420 . Precedent for iterative coordinate learning and enhanced sampling.
* Rizzi et al., OneOPES: https://doi.org/10.1021/acs.jctc.3c00254 ; author tutorial: https://github.com/valeriorizzi/OneOPES_tutorial . Precedent for complementary exploratory replicas and additional CV biases.
* Kamiya and Sugita, gREST: https://doi.org/10.1063/1.5016222 . Selective energy-term tempering as an alternative when coordinate evidence is inadequate.
* Repository contract: https://github.com/sulcjo/ATLaS-MD/blob/main/docs/atlas-md/guide/thermodynamic-validity.md . State identity, cross-energy evaluation and the distinction between transition-kernel correctness and finite-timestep accuracy.

The design deliberately treats candidate discovery as uncertain inference and promotion as an intervention to be tested. The productive outcome of an epoch may be a new bias, a continuation under the same regime, or a documented lack of evidence.


## 19. Adversarial review record — v0.2

Review performed 2026-09-28 by inspecting v0.1, the code excerpts already retrieved at the pinned revision, primary method documentation, and a minimal numerical counterexample. This was not a full runtime code audit or peptide simulation. Findings below are design defects or unresolved risks, not claims of confirmed bugs in existing ATLaS-MD.

| ID | Severity | Failure exposed | Required correction / status |
|---|---|---|---|
| A01 | High | A pooled exploratory density can give the wrong bias for a target rung | Source-distribution contract; no pooled-density flattening; corrected Section 9 |
| A02 | High | Low density outside training becomes a negative-bias reward for unknown space | Smooth neutral tails, support audit, boundary-barrier test; corrected Section 9 |
| A03 | High | Residual discovery followed by raw deployment restores dependence on existing CVs | Validate exact deployed function or full conservative residual model; corrected Section 7 |
| A04 | High | Scalar classifier distinguishes basins but misses their transition route | Entropic/degenerate tests and intervention gate; corrected Section 16; efficacy remains empirical |
| A05 | High | Isolated candidate success does not mean structures return through the ladder | Mandatory reduced exchange pilot under fixed resources; corrected Section 10 |
| A06 | Critical if violated | Correct recorded energy but wrong GaMD force-group scaling samples another potential | Explicit integrator force-expression audit and force/energy parity; runtime release blocker |
| A07 | High | Fresh frozen segments inherit selected-start nonequilibrium bias | Independent-start and burn-in sensitivity requirements; convergence remains empirical |
| A08 | High | Implicit product expansion multiplies replica count and dilutes sampling | Fixed state/Context cap and explicit rung reallocation; corrected Section 11 |
| A09 | Medium–high | Native-state absence is unverifiable, while mixture/seed confounding checks can suppress real trapped basins | Preserve abstention and matched-Hamiltonian probes; corrected Section 7 |
| A10 | Medium–high | All short trials fit a nominal budget but are too short to reveal any rearrangement | Cost-feasibility gate; include confirmation and transport costs; corrected Section 13 |
| A11 | High | Exact frame IDs conceal coordinate quantization errors in cross-energies | Runtime/recomputed reduced-energy agreement gate; corrected Section 12 |
| A12 | High | Qualitative promotion language allows post-hoc cherry-picking | Mandatory predeclared metric/margins/duration; unresolved calibration blocks automatic promotion |
| A13 | High | Biased structural novelty preferentially rewards unfolded/disordered configurations | Coverage alone remains insufficient; use common-anchor transport and target-population consistency, not a compactness reward |
| A14 | Medium–high | Self-derived cluster labels make cross-validated classification look more meaningful than it is | Distinguish representational reproducibility from dynamical utility; confirmation and independent evaluation descriptor families mandatory |

### 19.1 Explicit counterexample: pooled-density bias can worsen imbalance

Take a target rung with two-state probabilities p=(0.01,0.99), and an unrelated pilot mixture q=(0.99,0.01). In units RT=1, let B=0.5 log(q/q_max), the uncapped v0.1 proposal. The actual new probabilities are proportional to p exp(-B), yielding approximately (0.00101416,0.99898584). Thus the rare state becomes roughly ten times rarer. This calculation was executed locally. It disproves general flattening of a rung from arbitrary pooled occupancy; it does not imply that every such exploratory bias fails.

### 19.2 Explicit counterexample: discovery/deployment mismatch

Let a raw feature be d=10s+h, where s is an existing umbrella coordinate and h is a persistent hidden variable independent of s. Conditional residualization recovers h. Deploying z=d instead of z=h restores the dominant s dependence. Candidate usefulness must therefore be assessed on the precise runtime projection, not just on residual-space fit metrics.

### 19.3 Explicit counterexample: scalar projection degeneracy

Two torsions theta=+pi/3 and theta=-pi/3 have identical cos(theta)=0.5; a bias depending only on that cosine cannot distinguish their energies. A periodic sin/cos dictionary avoids raw-angle discontinuities but its scalar projection can reintroduce degeneracy. Even a nondegenerate basin classifier need not resolve an entropic bottleneck between basins. Classification, density flattening and accelerated mixing are different claims.

### 19.4 Validation that remains mandatory

The revised specification does not yet set empirically justified promotion thresholds, equilibration durations, force caps, density bandwidths, maximum replica reallocations or bridge overlap thresholds. These must be versioned in a benchmark-calibrated policy before `mode: promote` is enabled. Missing policy values block automatic promotion while shadow reports and isolated experiments remain available.

Likewise, a complete implementation plan must inspect the actual production integrator, force groups, state-update operations, adaptive budget controller and NPT adapter at current HEAD. Architectural module names are not sufficient evidence of compatibility. No peptide efficacy or runtime speedup has been verified by this review.

Additional primary references checked during review:

* MBAR's equilibrium-data assumptions: Shirts and Chodera, https://arxiv.org/abs/0801.1426 . Cross-energy completeness does not turn nonequilibrium paths into equilibrium samples.
* PLUMED OPES_METAD documentation, https://www.plumed.org/doc-v2.9/user-doc/html/_o_p_e_s__m_e_t_a_d.html . Degenerate CVs can stall useful exploration despite apparent bias convergence.
* PLUMED OPES_METAD_EXPLORE documentation, https://www.plumed.org/doc-v2.9/user-doc/html/_o_p_e_s__m_e_t_a_d__e_x_p_l_o_r_e.html . Exploration-oriented adaptive training is a possible backend, not proof of stationary-sample validity.
