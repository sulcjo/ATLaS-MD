# ATLaS-MD: Bayesian thermodynamic inference and sampling design

Date: 2026-09-25  
Revision: 1.1, incorporating adversarial design review  
Status: specification; algebra checks performed, implementation and MD validation pending  
Repository baseline: `sulcjo/atlas-md`, `27bb694d7db7067b23be792e0bd29fbe92d11bae`  
Suggested repository destination: `docs/superpowers/specs/bayesian-sampling/spec.md`

## 1. Goal, scope and deployment decision

Reduce the computational cost of estimating equilibrium populations and free-energy differences to a specified, empirically calibrated accuracy. The target deployment is the existing umbrella/Pep-GaMD replica-exchange workflow, typically 236 replicas on four L40S GPUs with 192 allocated logical CPUs. Correct kinetics is not an objective.

Implement four components:

1. A BayesMBAR analysis sidecar, alongside ordinary MBAR, for thermodynamic-state normalization constants and conditional posterior uncertainty.
2. Dependency-aware resampling and held-out checks for observable uncertainty and unresolved structural sampling.
3. Diagnostic-only importance-weight tail analysis, including exploratory Pareto diagnostics.
4. An epoch-boundary sampling-design controller that first records recommendations in shadow mode and only later acts after calibration.

This is an original integration design informed by the methods in §13. Literature support for individual methods does not establish correctness of their combination with adaptive Pep-GaMD, exchange, and repository loaders.

**Initial release is offline/shadow only.** It must not change forces, integration, exchanges, accepted scientific records, production stopping, or the current estimator. Closed-loop decisions are a later deliverable with explicit promotion gates. A failed Bayesian analysis cannot convert an existing scientific failure into a pass.

The design preserves the target Hamiltonian, timestep, NPT measure, random-number ownership, fixed within-epoch bias definitions, and exact per-state/sample provenance. No native structure, folded/unfolded labels, native-contact score, or post hoc RMSD criterion may guide CV selection or scheduling. Structural regions may be defined without supervision on development data and then frozen for evaluation.

### 1.1 Explicit non-goals

This work does not prove complete exploration, cure lack of phase-space overlap, infer equilibrium probabilities from GENPEPT seed counts, restore kinetics, fix integrator/force-field bias, replace exact boost accounting with CE2, or make smoothness evidence of a missing basin's absence. It does not make PSIS smoothing the default estimator or directly port umbrella-integration gradients into Pep-GaMD.

## 2. Repository findings and integration map

Source paths below are verified at the pinned baseline. Proposed modules and flags are new interfaces, not existing capabilities.

| Existing path or symbol | Verified role | Integration requirement |
| --- | --- | --- |
| `gareus/mbar_analysis/data.py:Data` | Per-sample CVs, origin labels, steps, reduced matrix, raw boost channels and state lambdas | Build a strict immutable snapshot adapter; preserve aligned masks |
| `data.py:infer_temp_beta` | Has a warning-based 300 K fallback when provenance is unavailable | Bayesian eligibility must reject inferred defaults without explicit verified temperature |
| `gareus/mbar_analysis/loaders_union_parquet.py:load_parquet_adaptive_union` | Combines epoch data, maps state IDs and reconstructs union biases | Audit every column as one fixed Hamiltonian across all rows |
| `gareus/mbar_analysis/bias.py` | Reconstructs umbrella biases, including regime-dependent inputs | Reuse validated force definitions; forbid guessing missing historical parameters |
| `gareus/mbar_analysis/ladder.py:apply_ladder_boost_to_u` | Adds each state's Pep-GaMD boost to the reduced matrix | Include it exactly once; preserve channel/envelope checks |
| `ladder.py:mbar_state_overlap` | Reduced-potential overlap; repository convention is column-stochastic | Do not substitute a CV histogram or silently transpose conventions |
| `gareus/mbar_analysis/solvers.py:solve_mbar`, `logw_from_fk` | MBAR estimates and target weights after common-potential cancellation | Compare identical rows/counts and generalize explicit target potential carefully |
| `gareus/mbar_analysis/estimators.py:ladder_excluded_methods` | Excludes cumulant/exponential correction under an already represented boost ladder | Bayesian output must retain these exclusions |
| `gareus/mbar_subsample.py:equilibrated_subsample` | Equilibration/subsampling with provenance and optional strict failure | Strict path required for eligibility; not sufficient alone for exchange dependence |
| `gareus/mbar_analysis/pmf.py` | PMFs, boost and uncertainty diagnostics | Add separate reports, not silent replacement of selected estimates |
| `gareus/adaptive_production.py:propose_actions_from_diagnostics` | Existing bridge/rung and registry proposals | Constrain and rank valid proposals rather than bypassing their geometry checks |
| `build_adaptive_epoch_schedule` | Baseline all-state allocation and optional targeted extensions | Respect current scheduling mode, including disabled top-ups |
| `AdaptiveRuntimePool`, schedule reload logic | Budget ledger and persisted resume schedule | A Bayesian decision must be budgeted and replayed exactly once |
| `evaluate_adaptive_convergence_gate`, final quality gate | Separate adaptive readiness and final analysis readiness | New diagnostics cannot override existing failures or equate readiness with global equilibrium |

Important audit boundary: historical state IDs can coexist with changed centers, CV models, or envelopes. This is a risk requiring a strict adapter and tests, not a claim that every current union is invalid. The sidecar must refuse any input for which fixed-state semantics cannot be established.

## 3. Contracts before inference

### 3.1 Immutable snapshot

Define `BayesSnapshot` with a content hash and schema version. It contains:

- Campaign, branch, committed generation, epoch/segment identifiers, source checksums and eligibility/exclusion reasons.
- A row table: immutable sample UID, replica UID, integer step, exact simulated-time span identity, origin thermodynamic-state key, assignment revision, seed lineage, exchange-connected ensemble ID, and independent-campaign ID where actually known.
- A state table: canonical Hamiltonian keys, physical system/temperature/pressure/measure identity, exact umbrella and CV definitions, boost kernel and frozen envelope identity, lambda values and units.
- `u_nk` in float64 with explicit shape `(N_samples, K_sampled_states)`, exact origin-column mapping, counts `N_k`, target reduced potential `u_target_n`, and the canceled common-energy convention.
- Observable/region definitions, frame joins, data split, burn-in masks, dependence blocks, priors, software versions and RNG seeds.

For live decisions, consume only an atomically committed snapshot under the helper-cpus durability contract. That specification exists but its implementation must not be presumed. Before it is available, permit offline analysis of a sealed, audited, completed-run export with its own immutable manifest; label it `legacy_audited`. It cannot silently certify live checkpoint/output consistency.

Never read files that are still being appended, choose the newest row by wall-clock timestamp, or merge replayed branches by numeric step alone. All filtering and frame joins use identities. Missing required energies, inconsistent units, duplicate identities, unresolved temperature, or corrupted joins block eligible inference. A diagnostic export may enumerate bad rows; dropping energy-dependent missingness is not automatically unbiased and cannot earn a pass.

### 3.2 Thermodynamic-state identity

The analysis key hashes the complete potential definition and ensemble measure. A changed bias center/stiffness, CV transformation/version, envelope, lambda, physical Hamiltonian, temperature or pressure makes a new state key even if the registry reuses a numeric ID.

Each column must mean the same `u_k(x)` for every row. If an older configuration cannot be evaluated under a newer CV definition from stored sufficient statistics or coordinates, do not fill the gap from the current registry. Analyze separately with explicit non-comparability, or regenerate the required cross-energies from verified frames. Ineligible unions cannot be repaired by a prior.

States with zero retained samples are prediction targets, not unconstrained sampled-state parameters. Remove them from the denominator and infer their normalization only through target reweighting where support exists. Canonicalize analytically identical potentials; if `u_j(x)=u_i(x)+c` by construction, enforce `f_j=f_i+c`. Numerical coincidence on a few frames is not proof of identity.

### 3.3 Physical target and exact boost accounting

For a common physical system, temperature, pressure and measure, write the sampled potential as

```text
u_k(x,V) = u_phys(x,V) + beta*[B_k(x) + DeltaV_k(x)]
u_target(x,V) = u_phys(x,V)
```

`u_phys` includes the common NPT terms under the implementation's measure. Only after verifying that they are common may the adapter subtract `u_phys` from every sampled-state column and the target, leaving `u'_k=beta*(B_k+DeltaV_k)` and `u'_target=0`. Different thermodynamic measures or temperatures are outside the initial support envelope.

Use the existing validated boost kernel to calculate each cross-state `DeltaV_k` from sufficient raw channels and that state's envelope. Its sign is positive in the sampled potential. Apply neither `exp(beta*DeltaV)` nor a cumulant correction afterward when the boost is already represented in `u_nk`. A common non-ladder boost also cannot disappear merely because it cancels between sampled states: the physical target still differs from those states.

Initial support: plain fixed umbrellas and a verified frozen exact Pep-GaMD lambda ladder. Other GaMD paths, changed real-space kernels and cross-envelope unions require explicit adapters and matching-energy tests. Unsupported input is reported as unsupported, never interpreted as zero boost.

## 4. BayesMBAR sidecar

### 4.1 Model and numerical checks

For retained sample `n` with origin `s_n`, sampled counts `N_k>0`, and dimensionless state free energies `f_k`, use the reverse-logistic model:

```text
q_nk(f) = N_k*exp(f_k-u_nk) / sum_j N_j*exp(f_j-u_nj)
log L(f) = sum_n log q_n,s_n(f)
p(f | D) proportional to L(f)*p(f), with f_reference = 0
```

This is the BayesMBAR construction [S1]. Its uniform-prior MAP equals MBAR on identical inputs. Posterior means need not equal that MAP. The likelihood conditions on observed configurations; its posterior is not a generative model of unsampled molecular structures.

Use a pinned optional BayesMBAR dependency behind a small adapter. Float64 and log-sum-exp are mandatory. Check matrix orientation, origin labels, gauge and all counts before invoking it. Do not substitute fractional effective counts into a likelihood built from all raw rows while claiming the original model remains unchanged.

Reference tests: uniform-prior MAP agrees with the repository solver on identical selected data within `1e-6 kT` for well-conditioned test cases; gradients and gauge transformations agree independently. Additive per-configuration shifts must be applied to target and sampled potentials together. Report residuals, curvature and posterior sampling diagnostics separately from scientific validity.

Use multiple posterior chains with independently seeded initializations. Initial computational gate: rank-normalized split R-hat at most 1.01, bulk/tail ESS at least 400 for every reported free-energy contrast, and no unresolved divergent transitions where applicable. These are inference-engine checks, not MD convergence evidence. Failure yields `inference_failed`, never a narrower fallback interval from a local Hessian without relabeling.

### 4.2 Identifiability and priors

Start with the gauge-fixed flat-prior model on an empirically connected, numerically identifiable sampled-state set. Audit likelihood curvature, overlap bottlenecks and independent bridges. Exact disconnected components have independent unidentified offsets; near-disconnection can make inferences practically unusable. Report component-specific results and block cross-component claims.

A proper weak prior may be run as a sensitivity arm, but may not convert disconnection into measured relative populations. Record bounds/scales and whether posterior mass approaches prior boundaries. Repeat with scales multiplied and divided by two. For controller-relevant quantities, a shift above 25% of the declared scientific error tolerance or a large interval-width change is a `prior_sensitive` result.

Smoothness/GP priors are experimental. Do not equate a biased state's normalization with the physical PMF evaluated at its umbrella center. If using a prior over state normalizers, its input must encode the entire relevant bias family, not just center coordinates while ignoring stiffness/lambda. If instead fitting a physical PMF, define the implied biased partition integrals. Preserve periodic geometry for angular coordinates and test kernel/length-scale sensitivity. GENPEPT occurrence counts never supply equilibrium prior mass.

### 4.3 Observable weights and the missing uncertainty term

For target `t`, use

```text
log r_n(f) = -u_target_n - logsumexp_k(log N_k + f_k - u_nk)
W_n(f) = exp(log r_n - logsumexp_m(log r_m))
P_A(f) = sum_n W_n(f)*1[x_n in A]
DeltaF_AB(f) = -kBT*log(P_A(f)/P_B(f))
```

Recompute counts from exactly the same retained rows used in the fit. Do not pair a filtered subset with full-data counts/free energies and call it an exact refit. Preserve unsampled/empty region status rather than adding a pseudocount and reporting measured mass.

**Posterior variation of `f` alone is only normalization uncertainty conditional on the observed configurations.** In a one-state unbiased simulation there are no unknown free-energy differences after gauge fixing, yet a basin population still has sampling error. Therefore the sidecar must never label the distribution of `P_A(f)` over fixed rows as its complete credible interval.

Produce separate fields:

| Field | Interpretation |
| --- | --- |
| `state_f_posterior` | Conditional BayesMBAR posterior under the specified likelihood/prior |
| `observable_f_sensitivity` | Propagation of that posterior through fixed observed support |
| `observable_block_interval` | Sampling uncertainty from dependency-aware repeated refits (§5) |
| `between_campaign_discrepancy` | Disagreement across genuinely separate campaigns |
| `support_status` | Missing/weak support and unresolved regions, not a numeric error bar |

Do not add posterior and bootstrap variances: they can describe overlapping uncertainty. Initial controller and stopping diagnostics use calibrated block/campaign intervals, while BayesMBAR supplies an independent normalization/prior-sensitivity view. A unified credible interval requires a separately specified generative model and coverage validation; it is not silently synthesized here.

## 5. Dependence, burn-in and uncertainty calibration

### 5.1 Units of evidence

Distinguish frames, decorrelated observations, trajectory blocks and independent campaigns. Two replicas exchanging configurations are not independent campaigns. Restarting from the same checkpoint is not a new independent campaign; cloned seed ancestry is retained even when new random seeds are assigned.

Missing lineage or campaign provenance is explicitly unknown. Conservatively group potentially related runs for dependence analysis; never manufacture independent-campaign IDs from directory names, epoch numbers or random seeds alone. If this leaves too few independent units, confirmation remains unresolved.

Use predeclared production phases with frozen Hamiltonians. Estimate burn-in and correlation from multiple diagnostics: physical/boost energies where available, both selected CVs, target-weight influence, and structural region indicators. A constant trace can indicate trapping; it does not prove zero correlation time. `equilibrated_subsample(..., strict=True)` is a utility, not an independence certificate.

For the reference BayesMBAR fit, choose a documented time-based subsampling policy after burn-in, with actual retained counts. Use contiguous traces respecting state versus walker labels; never concatenate unrelated short visits and estimate their correlation as a continuous state trajectory. Report sensitivity to the selected spacing. The iid likelihood remains an approximation until calibrated on exchange-coupled data.

### 5.2 Block/campaign resampling

Represent each exchange-connected ensemble as synchronized slices of simulation time containing all available replicas and their actual assignment labels. Resample contiguous time blocks of those slices together, preserving within-block exchange coupling. Never bootstrap individual frames or replicas independently. Segment boundaries, changed Hamiltonians and excluded burn-in regions cannot be crossed by a block.

Choose block length on development data from the slowest monitored relaxation, then test lengths `L`, `2L`, and `4L`. Require at least 30 eligible blocks at the adopted length for quantitative controller input; this engineering floor does not certify adequate mixing. If estimates drift or intervals widen materially with longer blocks, report `dependence_unresolved`. For shared lineages across epochs, preserve the dependency grouping and prefer whole independent-campaign resampling when sufficient campaigns exist.

For each resample, refit normalization constants and recompute target weights and all requested observables using that resample's actual counts. Initial implementation uses ordinary MBAR refits for the sampling interval. Do not nest posterior draws inside every bootstrap and describe the resulting mixture as a calibrated Bayesian posterior. Track absent states and disconnected resamples; do not discard these failures and report the successful subset as if complete.

Use at least 1,000 resamples for the default offline interval calculation, with Monte Carlo stability checks on interval endpoints and a configurable compute budget. An insufficient budget yields `uncertainty_unresolved`. Percentile or other interval method is named in the manifest and validated in §14; no automatic guarantee of nominal coverage is claimed.

### 5.3 Equilibration and adaptive selection

Each new or changed state has explicit equilibration treatment. Seeded starts, CV/envelope updates and rescue events do not become equilibrium data just because the state label is known. Keep calibration/pilot trajectories distinct from production evidence.

Outcome-dependent allocation and stopping can bias finite-sample estimates and coverage. Logging the decision is necessary but not sufficient to repair this. Initial publication-grade results use a fresh fixed-design confirmation phase: freeze states/CVs/envelopes, predeclare its length, equilibrate, and analyze it separately from adaptive discovery. The union remains a secondary diagnostic until its full adaptive policy has demonstrated estimator/interval calibration in closed-loop tests.

## 6. Held-out structural and predictive checks

### 6.1 What is actually predicted

The reverse-logistic model predicts a state label conditional on a configuration. Assess held-out label log score/calibration as a narrow model check; it does not predict whether a missing folded basin exists. Independent structural distribution checks are also required.

For a held-out state with known potential `u_h`, use the training sample mixture and its fitted normalizers to predict supported observables under that state:

```text
W_h,n proportional to exp(-u_h,n) / sum_k N_k*exp(f_k-u_k,n)
```

Compare this prediction with separately measured held-out data under the same Hamiltonian, including uncertainty on both sides. The training-only denominator and counts are fixed before looking at held-out outcomes. This is a reweighted held-out check on observed support, not a generator of new molecular conformations.

For a true posterior predictive structural simulator, specify an additional generative model. It is outside the initial release; do not call resampling old configurations proof of discovery capability. General predictive-check principles are described in [S5].

### 6.2 Splits, regions and discrepancies

Freeze development, calibration and confirmation campaign IDs before fitting. Use independently initialized campaigns where possible; different seed families are deliberately useful stress tests but require their own equilibration assessment. Splitting exchanged replicas or adjacent trajectory halves is not an independent holdout.

Define structural descriptors and regions using development data only, without native supervision. Candidate descriptors include torsion features, nonlocal contact patterns and other existing structural observables. Freeze preprocessing, clustering, region boundaries and observable tolerances. A confirmation configuration outside the training representation is flagged as new support; it is not forced into the nearest familiar cluster. Any region redesign starts a new evaluation version and confirmation split.

Report target-reweighted region probabilities, selected marginal distributions and within-CV-cell structural mixtures. Do not compare raw occupancies from different umbrellas and interpret the difference as disagreement. Use support checks before reweighting between designs.

Declare discrepancy statistics and practical margins in advance. For example, compare a vector of held-out basin populations against simultaneous block-resampling intervals rather than testing many bins independently at 5%. Track seed-family disagreement, between-campaign spread and prediction error after normalization/within-campaign uncertainty. An explicit random-effects model can be evaluated later; extra variance must not become permission to average away non-equilibrium branches.

If every campaign misses the same state, these checks may pass. Reports must retain the statement `coverage conditional on explored support`; no finite dataset produces a universal missing-basin guarantee.

## 7. Importance-weight tail diagnostics

### 7.1 Diagnostic input and outputs

For the pooled fixed-design sample mixture, evaluate log importance ratios `log r_n` from §4.3. Report raw weight ESS `1/sum W_n^2`, maximum weight, top-1% weight share, per-source contribution, and effective independent-block contributions. Weight ESS is not an autocorrelation-corrected trajectory ESS.

For each requested region/observable, report how many blocks and campaigns carry its weighted contribution and perform leave-one-block/campaign-out sensitivity. A globally acceptable weight distribution can still leave a rare region determined by one event.

Add generalized-Pareto upper-tail fits as an exploratory diagnostic following PSIS [S3]. Record the fitting implementation/version, selected tail length, retained sample count, dependence policy, fit warnings and sensitivity across blocks/campaigns. Equal weights produce a benign constant-weight diagnostic; too few distinct tail observations produce `unresolved`, never an invented zero shape parameter.

Do not multiply weights by a second GaMD correction. Do not fit just the boost exponential when umbrella/multistate weights are the actual estimator. For region checks, distinguish the global proposal-to-target ratio from region-conditioned ratios or observable influence values. Conditioning on region membership changes the proposal; naïvely fitting the nonzero subset and applying global thresholds is not justified.

### 7.2 Interpretation and remediation

Standard PSIS thresholds, including sample-size-dependent thresholds described in [S4], are reference annotations only until calibrated for this stratified, correlated and adaptively collected setting. Estimated `k >= 1` is an instability warning, not a claim that a finite molecular partition function is physically divergent. A favorable tail fit cannot detect important configurations that were never observed.

Initial mode is `diagnostic_raw`: preserve the exact estimator weights. A sandbox smoothing comparison may be emitted under a separate estimator ID, with raw-versus-smoothed differences; it must never replace official PMFs, increase eligibility, or trigger stopping. Demonstrating improved numerical stability does not demonstrate unbiased populations.

Candidate remedies are new data: additional fixed-design sampling, verified weak/zero-boost states, bridge rungs, or independent structural seeds with proper equilibration. Choose them at the next epoch through the same feasibility/budget rules, not by changing live weights or envelopes. If no useful supported action exists, return `needs_sampling_design_review`.

## 8. Bayesian sampling design

### 8.1 Objective and action space

Predeclare scientific quantities `Q_j` and tolerances `epsilon_j`. Initial defaults should be user/configuration-selected rather than silently invented; a benchmark fixture can use 0.02 absolute population error and `0.2 kBT` for specified basin free-energy differences. Do not optimize only the normalizers of biased windows as a substitute for the physical observables.

An ideal policy maximizes expected uncertainty reduction per complete action cost:

```text
L(D) = sum_j alpha_j * Var(Q_j | D) / epsilon_j^2
utility(a) = [L(D) - E_Y|D,a L(D union Y)] / cost(a)
```

This is the design objective, not a claim that a calibrated predictive distribution already exists. Quantities lacking valid uncertainty remain unresolved and cannot acquire utility zero merely because a calculation failed. Missing-support exploration is a separate requirement.

Allowed actions are existing valid next-epoch plans: extend a full connected ensemble; prioritize a feasible bridge/rung proposal; or request an independent confirmation/exploration campaign. State-specific top-ups are candidates only when already enabled and scientifically supported by the configured scheduler. Never re-enable `ap_topups` implicitly. Adding states must fit hardware/context limits and the runtime budget. At capacity, report infeasibility; do not evict states solely because their inferred population is small.

The cost includes equilibration, context creation, all-replica work, checkpoint/output drain, analysis overhead and expected hardware occupancy. Compute both aggregate simulated-time charge and predicted GPU/node-hours; the existing aggregate-MD pool remains a hard constraint. A state-level extension that forces an entire synchronized ensemble to run is charged as that ensemble, not as one replica.

### 8.2 Implementable staged acquisition model

**Shadow v1:** list feasible actions, their known costs, support/overlap deficits, and uncertainty diagnostics. Preserve the baseline controller's actual choice. Record predicted rankings only where the design model below is calibrated; otherwise record `utility_unknown`. This stage is useful without pretending BayesMBAR alone generates hypothetical trajectories.

**Calibrated design v2:** implement a Gaussian approximation to a preposterior measurement model. Let `theta` contain the chosen quantities and auxiliary estimable contrasts, in declared coordinates. A feasible future independent block summary is modeled as

```text
y_a = H_a*theta + noise_a,     Cov(noise_a) = R_a
C_after = C - C H_a^T (H_a C H_a^T + R_a)^(-1) H_a C
```

Here `C` is a joint uncertainty covariance validated against block/campaign replication, including configuration-sampling uncertainty for populations. It must not be the free-energy-normalizer posterior covariance padded with deterministic population derivatives. `H_a`, `R_a`, bias checks and the number of genuinely new blocks are learned from independent pilot/confirmation data for the same action class, not invented by assuming every frame is independent or uncertainty always scales as `1/N`.

The update is a declared Gaussian decision approximation, not a new exact molecular posterior. Use its reduction in the `Q` block to rank known actions. Require positive-semidefinite covariances, validated noise floors, explicit cross-covariance, and error bars on predicted gain/cost. Use a conservative gain bound rather than selecting a noisy maximum. Where linearization is poor, probabilities approach 0/1, or an action is outside calibration support, mark its utility unknown and use a pilot/fallback. A hypothetical batch is never entered into the actual likelihood.

The displayed update assumes future measurement noise is conditionally independent of the data already used in `C`. Continuing the same trajectory immediately after a snapshot need not satisfy that assumption. Use validated separated blocks or an augmented model carrying the cross-covariance with the existing data; otherwise that action's gain is unknown. Estimating a small within-batch variance and treating it as independent new information is forbidden.

For a batch of proposed actions, recompute marginal gains with their joint covariance; summing independent utility estimates double-counts shared information. Preserve existing bridge fairness constraints and graph connectivity. A GP extension may model mean forces or action responses, but only with a likelihood derived for the actual biased ensemble. The BUQ precedent [S2] motivates variance-reduction design; its umbrella-integration estimator is not assumed exact for this pipeline.

### 8.3 Exploration protection and policy constraints

Maintain the configured minimum all-state segment and all existing physical/quality constraints. The Bayesian ranker cannot retire states, eliminate low-population regions, strengthen umbrellas, change CVs/envelopes, or stop production in its initial closed-loop release.

Reserve a configurable exploration budget; propose 20% of discretionary action budget as an initial benchmark setting, not a universal optimum. Use it for feasible frontier/bridge/independent-seed probes without requiring high posterior equilibrium probability. Allocate in integer action quanta using a persistent credit ledger so small budgets cannot round exploration to zero forever. Charge every probe, including equilibration. If required baseline/exploration cannot fit, report insufficient budget rather than exceeding it.

Fresh structure proposals must satisfy existing seed geometry/physical checks and equilibration. They provide discovery opportunities, not equilibrium weights. The ranker may prefer improving overlap, but no finite-overlap threshold alone certifies hidden-coordinate mixing.

### 8.4 Epoch transaction and restart

At a committed common boundary: seal the eligible snapshot; compute diagnostics within a bounded budget; generate and validate candidates; choose the next plan; write the complete immutable decision; atomically associate it with the next epoch before execution. Persist snapshot hash, state-registry revision, action list, exact step budgets, model/prior versions, exploration credit, cost predictions and policy RNG state.

Resume reuses that decision and the existing saved schedule verbatim. It never recalculates against a smaller remaining pool. Late analysis results apply only if their input generation and registry revision still match; otherwise they are stale diagnostics. Failed or timed-out analysis invokes the existing deterministic fallback and records the reason. The fallback must not reset a scientific failure or claim convergence.

No helper reads or mutates a live OpenMM Context. Use helper-cpus' shared job budget and bounded memory. Run CPU-only inference in an isolated process with native thread limits and GPU visibility disabled before importing JAX or similar libraries; otherwise an analysis package could helpfully claim the simulation GPUs. If it cannot run within the budget, defer or reduce only a separately labeled diagnostic resolution, never silently change statistical evidence.

## 9. Convergence and stopping semantics

Output independent statuses for `data_integrity`, `equilibrium_eligibility`, `state_identifiability`, `inference_engine`, `dependence`, `weight_stability`, `prior_sensitivity`, `heldout_agreement`, `support`, and `decision_readiness`. Missing evidence is `unresolved`, not `pass`.

The existing adaptive gate decides whether to enter frozen production; this Bayesian module must not turn that into a claim of equilibrium convergence. During initial releases, produce `precision_target_met_on_observed_support` as advisory only when calibrated intervals meet the predeclared tolerances and all other required diagnostics permit it.

Autonomous stopping is deferred until the *entire sequential rule* is calibrated, including repeated looks, adaptive allocation, prior fitting and changing observation schedules. A pointwise 95% interval is not automatically a 95% guarantee at a data-selected stopping time. Retain fixed-length, independent confirmation as the default route. Budgets ending before sufficient evidence yield `budget_exhausted_unresolved`, not a relaxed pass.

For vector populations/PMFs, use predeclared simultaneous uncertainty or an explicitly bounded set of contrasts. A stable PMF offset, a small maximum change between correlated prefixes, or a posterior sampler's R-hat alone cannot satisfy this gate.

## 10. Interfaces, artifacts and implementation tasks

### 10.1 Proposed package and API

```text
gareus/bayes_analysis/
    snapshot.py        # provenance, state canonicalization, fixed u_nk contract
    dependence.py      # masks, synchronized blocks, independent campaigns
    posterior.py       # BayesMBAR adapter, gauge, priors, engine diagnostics
    observables.py     # explicit target weights, f-only sensitivity labels
    resampling.py      # full refits and calibrated block/campaign intervals
    predictive.py      # frozen splits, supported holdout comparisons
    weight_tails.py    # raw concentration and advisory Pareto fits
    design.py          # feasible action ranking and Gaussian design model
    reports.py         # manifests, status fields, machine/human reports
analyze_gareus_bayes.py # standalone entry point
```

Core contracts:

```text
build_snapshot(source, target, split, eligibility) -> BayesSnapshot
fit_state_posterior(snapshot, prior, inference_budget) -> PosteriorResult
estimate_observables(snapshot, posterior, block_plan) -> ObservableReport
check_holdout(training_snapshot, heldout_snapshot, frozen_checks) -> CheckReport
diagnose_weights(snapshot, fitted_state_model) -> WeightReport
rank_actions(snapshot, reports, feasible_plans, calibrated_model) -> DecisionReport
```

The snapshot adapter validates current loaders rather than duplicating force mathematics. Integration into `AdaptiveProductionController` consumes one typed `DecisionReport`; it does not scatter new statistical conditions through the existing long driver. Add config fields through the normal parser/provenance flow and extend aligned-row registries when adding sample arrays.

### 10.2 Proposed configuration

| Setting | Initial value | Contract |
| --- | --- | --- |
| `bayes_mode` | `off` | `off`, `offline`, `shadow`, later `epoch_controller` |
| `bayes_prior` | `flat` | Gauge-fixed; proper/smooth variants sensitivity-only initially |
| `bayes_target` | Explicit physical target manifest | No inference from current window labels |
| `bayes_observables` | Required for design | Frozen definitions, tolerances and weights |
| `bayes_dependence_mode` | `ensemble_blocks` | Campaign hierarchy when available |
| `bayes_bootstrap_replicates` | `1000` offline | Endpoint stability and compute budget reported |
| `bayes_weight_mode` | `diagnostic_raw` | No automatic smoothing |
| `bayes_design_model` | `none` | Calibrated Gaussian model enabled explicitly later |
| `bayes_exploration_fraction` | `0.20` for pilot | Fraction of discretionary budget, with persistent credits |
| `bayes_auto_stop` | `false` | Not supported for initial promotion |
| `bayes_compute_budget` | Explicit job-level budget | Counts against helper allocation; no new independent pool |

Artifacts under a generation/version-specific analysis directory: `snapshot_manifest.json`, `state_table.parquet`, `row_selection.parquet`, `dependence_plan.json`, `posterior_summary.json`, optional bounded posterior draws, `observable_uncertainty.parquet`, `heldout_checks.json`, `weight_diagnostics.json`, `decision.json`, and `report.md`. Avoid cloning the full dense energy matrix into each artifact; use checksummed immutable backing data/chunked access.

Every report names estimator, uncertainty type, support limitations and eligibility. A dashboard must not collapse all statuses to one green convergence badge. Preserve ordinary MBAR outputs for comparison.

### 10.3 Delivery tasks and gates

1. **Snapshot/state audit:** implement canonical identities and target-energy checks; test on valid, changed-CV and changed-envelope fixtures. No inference yet.
2. **Offline state posterior:** flat-prior BayesMBAR, ordinary-MBAR parity, gauge tests, engine diagnostics and prior sensitivity. Explicitly label conditional uncertainty.
3. **Observable uncertainty:** synchronized blocks, complete refits, campaign splits, empty-support handling and single-state counterexample tests.
4. **Holdout/tail reports:** implement unbiased comparisons under declared ensembles, raw-weight concentration, advisory Pareto fits, and failure statuses.
5. **Shadow controller:** emit feasible candidate plans and costs; record baseline choices and prospective gain predictions. No simulation changes.
6. **Calibration study:** validate interval coverage, predictive discrepancy power and utility predictions. Freeze model/thresholds before evaluation.
7. **Opt-in epoch controller:** only calibrated action classes; exact schedule/budget/durability integration and closed-loop benchmarks. Retain frozen confirmation and no automatic stopping.

## 11. Adversarial review and resulting design changes

This section records a self-review of the proposed design against mathematical counterexamples and the pinned code. It is not an independent reviewer sign-off or evidence of production validation.

| ID | Attack/failure | Severity | Resolution and required test |
| --- | --- | --- | --- |
| A01 | Treat posterior draws of state normalizers as full basin-population uncertainty | Critical | Separate uncertainty types; one-state Bernoulli test must retain sampling error |
| A02 | Reused numeric state ID hides a changed potential across rows | Critical | Hash complete Hamiltonians; split states and require cross-evaluation |
| A03 | Remove common boost between states and forget physical target, or remove boost twice | Critical | Explicit target potential and once-only ladder accounting; sign/cancellation fixture |
| A04 | A smooth prior creates a precise bridge between disconnected components | Critical | Likelihood-only identifiability gate; prior sensitivity cannot upgrade eligibility |
| A05 | Count exchanged replicas or resampled frames as independent evidence | Critical | Synchronized ensemble blocks and campaign-level validation |
| A06 | Short trapped trajectories give tiny estimated correlation time | High | Multiple structural observables, longer-block sensitivity and unresolved status |
| A07 | Raw occupancy differences across umbrellas masquerade as seed hysteresis | High | Compare common target or matched-Hamiltonian predictions with support checks |
| A08 | Posterior predictive resampling of existing frames claims to test unseen basins | High | Explicit fixed-support limitation; independent structural probes and unknown-cluster status |
| A09 | Apply standard Pareto thresholds to dependent mixture weights as a theorem | High | Diagnostic-only thresholds; calibrate actual mixture/dependence setting |
| A10 | Smoothed weights erase rare configurations and manufacture convergence | Critical | Raw estimator preserved; smoothing never changes eligibility or stopping |
| A11 | Adaptive stopping/allocation selects favorable fluctuations | Critical | Fixed confirmation first; full-policy calibration before stopping claims |
| A12 | Selecting a CV/cluster/prior on holdout leaks confirmation evidence | High | Frozen development/calibration/confirmation split with version resets |
| A13 | Ranker invents future observations from BayesMBAR's conditional likelihood | High | Explicit predictive measurement model or unknown utility; no fake molecular generator |
| A14 | Acquisition measures window-normalizer precision instead of physical thermodynamics | High | Predeclared physical quantities; configuration uncertainty included |
| A15 | Independent action gains double-count the same overlap information | High | Joint covariance and marginal batch gains |
| A16 | Integer rounding eliminates exploration or overspends the runtime pool | High | Persistent credit ledger and exact feasible integer plan validation |
| A17 | A crash recomputes a new decision and repeats committed sampling | Critical | Decision tied atomically to generation; idempotent budget and schedule replay |
| A18 | Optional JAX inference allocates GPU memory or oversubscribes CPU pools | High | CPU-only isolated process with shared helper budget and resource tests |
| A19 | Nonfinite-energy cleanup creates outcome-dependent missingness | High | Strict eligibility/exclusion audit; no automatic scientific pass after cleanup |
| A20 | All independent-looking campaigns inherit the same missing seed family | High | Lineage provenance; diversified probes; no global completeness claim |
| A21 | Bootstrap silently drops disconnected replicates and reports only successes | Critical | Count failures as unresolved evidence; no conditional-on-success interval |
| A22 | Duplicate physical states introduce artificial uncertain parameters | Medium | Analytic state canonicalization with exact offset constraints |
| A23 | The same snapshots are counted again during sequential posterior updates | Critical | Initial implementation refits immutable cumulative data once; any future incremental update proves disjoint new sample IDs |
| A24 | Conditional posterior diagnostic passes while finite-timestep MD samples the wrong ensemble | Critical | Upstream physical validation remains required; statistical checks cannot certify integrator correctness |

Remaining limits after mitigation: no proof of absent basins; no universal finite-sample calibration of exchange/adaptive data; no validated action-response model for unseen states; no general CE2/tail cure. These are explicit reasons to restrict automation, not omissions to fill with numerical defaults.

## 12. Verification performed for this specification

A small independent NumPy/SciPy algebra fixture was executed during drafting. It used three biased states over four configurations with unequal expected counts (700, 1100, 500), known physical energies and state-specific positive boosts. Fractional counts represented exact equilibrium expectations, so this checks equations rather than statistical coverage.

| Check | Observed result |
| --- | --- |
| Reverse-logistic flat-prior optimum versus analytic state free-energy differences | Maximum error `9.43e-8 kT` |
| Reweighted physical populations versus exact finite-support probabilities | Maximum absolute error `3.43e-9` |
| Subtract common physical energy from both sampled and target potentials | Weight change `0` at displayed precision |
| Deliberately apply an extra origin-state boost correction | Population error about `0.0312`; the wrong estimator is detectably wrong |
| One unbiased state, 100 independent Bernoulli observations with p=0.5 | f-only propagation gives zero width; true population-estimator SD is `0.05` |
| Gaussian covariance update with random positive-definite inputs | Updated covariance and variance reduction were positive semidefinite; minimum updated eigenvalue `0.0401` |

The single-state counterexample is decisive: the original informal idea of converting only BayesMBAR free-energy draws into full population error bars is rejected. The spec now requires separate configuration-sampling uncertainty.

No OpenMM trajectories, actual campaign files, BayesMBAR posterior chains, Pareto fits, interval-coverage experiment, crash/restart test or closed-loop policy benchmark has been run for this document. The table is not a performance result and must not be reported as implementation acceptance.

## 13. Sources and provenance

Repository files in §2 are pinned to [commit 27bb694](https://github.com/sulcjo/atlas-md/tree/27bb694d7db7067b23be792e0bd29fbe92d11bae). Key entry points: [adaptive_production.py](https://github.com/sulcjo/atlas-md/blob/27bb694d7db7067b23be792e0bd29fbe92d11bae/gareus/adaptive_production.py), [analysis package](https://github.com/sulcjo/atlas-md/tree/27bb694d7db7067b23be792e0bd29fbe92d11bae/gareus/mbar_analysis), [subsampling](https://github.com/sulcjo/atlas-md/blob/27bb694d7db7067b23be792e0bd29fbe92d11bae/gareus/mbar_subsample.py), [helper-cpus spec](https://github.com/sulcjo/atlas-md/blob/27bb694d7db7067b23be792e0bd29fbe92d11bae/docs/superpowers/specs/helper-cpus/spec.md).

- **S1:** Xinqiang Ding, Bayesian Multistate Bennett Acceptance Ratio Methods, JCTC (2024), [paper](https://arxiv.org/html/2310.20699), [DOI](https://doi.org/10.1021/acs.jctc.3c01212), [software documentation](https://dinglab.io/BayesMBAR/). Supports the conditional state-free-energy model and flat-prior MAP relationship. The additional uncertainty/eligibility contracts here are this specification's requirements.
- **S2:** Kempkes and Pérez de Alba Ortíz, Bayesian Umbrella Quadrature, [2026 manuscript](https://arxiv.org/html/2601.08783v1), [journal DOI](https://doi.org/10.1021/acs.jctc.6c00120). Precedent for variance-reduction-based umbrella placement; not direct validation for this Hamiltonian ladder.
- **S3:** Vehtari et al., Pareto Smoothed Importance Sampling, JMLR 25(72), 2024, [paper](https://www.jmlr.org/papers/v25/19-556.html). Supports importance-tail diagnostics; application-specific calibration remains required.
- **S4:** Stan/loo, [Pareto-k diagnostic documentation](https://mc-stan.org/loo/reference/pareto-k-diagnostic.html). Supplies conventional reference thresholds, not a guarantee for this application.
- **S5:** Stan, [posterior and prior predictive checks](https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html). General model-checking framework.
- **S6:** PyMBAR, [timeseries documentation](https://pymbar.readthedocs.io/en/master/timeseries.html). Equilibration and statistical-inefficiency utilities; exchange-aware block grouping is an additional design requirement.
- **S7:** Talts et al., [Validating Bayesian Inference Algorithms with Simulation-Based Calibration](https://arxiv.org/abs/1804.06788). Computational calibration is distinct from physical-model validation. Do not apply prior-predictive SBC to an improper flat prior without defining a proper generative test model.

## 14. Final benchmark, acceptance and promotion protocol

### 14.1 Questions and comparisons

Measure whether uncertainty statements are calibrated, whether diagnostics detect consequential failures, and whether the controller reaches predeclared thermodynamic accuracy with less total computation. Smaller reported error bars are not success by themselves.

| Arm | Purpose |
| --- | --- |
| A | Current production/analysis/controller, recorded unchanged |
| B | A plus offline/shadow Bayesian and weight diagnostics | Is analysis correct and affordable without affecting MD? |
| C | B plus calibrated epoch action ranking | Does sampling allocation improve actual error per cost? |
| D | C without exploration reserve, synthetic tests only | Detect self-confirming exploitation failure |
| E | C without dependence correction, synthetic tests only | Detect false precision from correlated evidence |
| F | Smooth-prior sensitivity arm | Measure prior-driven bias and hidden-barrier failure |

Ablations D–F never provide a shortcut around production correctness gates. Preserve the same physical model, resource envelope, accounting, output cadence and final-confirmation requirements for A–C.

### 14.2 Tier 1: mathematical and adversarial fixtures

Implement exact enumerable systems, harmonic umbrellas and multimodal low-dimensional potentials with independently computable equilibrium probabilities. Include unequal counts, rare basins, known shared energy shifts, duplicate states, zero-sample targets and changed envelope/CV state identities.

Construct two basins indistinguishable in the chosen CV pair but separated in a hidden coordinate. Run both discoverable and deliberately trapped versions. The latter must not pass global-equilibrium claims merely because posterior chains mix. Add disconnected and nearly disconnected overlap graphs, an unseen narrow basin erased by a smooth prior, bounded but extremely concentrated importance weights, and correlated replicas with controlled exchange.

Test exact target weights, MAP parity, gauge invariance, proper handling of NPT common-term cancellation, and the single-state uncertainty counterexample. Inject shuffled state labels, repeated frame IDs, missing raw channels, corrupted assignment revisions, origin-versus-destination labeling errors and duplicate boost correction. Required outcome is either the correct result or a specific blocking status.

For proper finite-dimensional generative test models, use simulation-based calibration to test the inference engine. Separately assess fixed-truth estimator/interval coverage; these answer different questions. Include output quantities, not only free-energy parameters, in calibration.

### 14.3 Tier 2: empirical coverage and failure detection

Use at least 200 independent synthetic repetitions per predeclared scenario, more if needed for the chosen binomial precision. Report bias, RMSE, interval width, nominal 50/80/95% coverage, and binomial confidence intervals. For 95% intervals, proposed promotion requires the lower one-sided 95% confidence bound on empirical coverage to be at least 90%, plus absence of systematic bias exceeding the scientific tolerance. Do not aggregate easy and failing scenarios into one favorable average.

For adversarial invalid scenarios, predeclare detectable errors and require at least 90% detection with confidence bounds, while measuring false alarms on valid cases. Fundamentally invisible missing support cannot be assigned an impossible detection target; the acceptance criterion there is that the software does not claim certified global completeness.

Repeat the full adaptive policy, including window selection, region/prior fitting rules, resource constraints, stopping attempts and confirmation. Testing a fixed schedule chosen after looking at truth is not a closed-loop benchmark. Keep synthetic truth out of the controller; it is available only to evaluation.

### 14.4 Tier 3: molecular pilot and held-out confirmation

Use the user's representative chignolin system and one held-out peptide/configuration. Freeze force field, temperature/pressure, timestep, CV/boost definitions and structural-observable protocol per comparison. Establish a reference from substantially larger, independent, diversified campaigns with its own uncertainty; label any comparison against that reference as reference-relative, not exact truth.

Run A–C with independent campaign repetitions and counterbalanced scheduling. Use at least six held-out paired campaign comparisons for an initial performance decision; do not treat 236 replicas in one exchange ensemble as 236 repeats. Different numerical trajectories are expected. Match budget and starting-ensemble policies, and include all exploration, equilibration and discarded adaptive work in cost.

The primary endpoint is total allocated GPU-hours/node-hours to achieve the predeclared thermodynamic tolerance and pass fixed independent confirmation. Also report absolute error against the reference, interval coverage where assessable, held-out structural discrepancies, target/block ESS, support discovery, durable aggregate ns/day, analysis CPU-hours, memory, context pressure and durable-output overhead.

Record runs that never meet accuracy within budget as censored failures; never average only successful runs. Report success probability at fixed budgets and restricted mean cost/time to the endpoint over the predeclared budget. A controller that declares success earlier while increasing physical error fails.

### 14.5 Systems, durability and resource tests

During B/C, saturate output queues, deny helper tokens, interrupt inference, kill the process between decision persistence and epoch start, and resume from normal entry points. Require one canonical decision/schedule, exact budget charging, no duplicated committed observations, no live Context access from helpers, bounded memory and no implicit CUDA allocation by analysis workers.

For B, require unchanged scientific event ordering/RNG consumption and baseline estimates when the feature is off. Proposed overhead gate for routine shadow use is below 2% loss of durable MD throughput under the allocated helper budget; otherwise run it offline. Compare only runs with the same durability contract.

### 14.6 Final promotion criteria and report

Promote the offline sidecar only after state/target correctness, dependency-aware uncertainty, unsupported-input refusal and clear uncertainty labeling pass. Promote an action class to the controller only after its prospective gain estimates are calibrated on held-out actions and the full closed-loop policy passes scientific and systems gates.

Proposed benefit threshold for default controller use: at least 10% lower median total compute cost to the validated accuracy endpoint, with a positive independent paired confidence interval for the improvement where the endpoint is observed, and no reduced fixed-budget success probability or systematic error regression. Report censored outcomes explicitly; use a predeclared survival/restricted-mean analysis when they prevent an ordinary paired comparison. These thresholds are engineering decisions, not expected gains.

The benchmark report ends with raw campaign manifests/results; numerical and coverage tests; failure-detection rates and blind spots; ordinary-MBAR versus Bayesian comparisons; prior/dependence sensitivity; held-out checks; action-cost/gain prediction calibration; complete runtime/MD budget reconciliation; crash/resource outcomes; and a decision for each component: **promote offline**, **shadow only**, **opt-in controller**, **inconclusive**, or **reject**. Automatic convergence stopping remains outside initial promotion.
