# ATLaS-MD equilibrium CV selection: implementation plan for AI agents

Version 1.0 · 2026-09-20  
Repository: sulcjo/ATLaS-MD  
Verified baseline: cdee6cc87ac540b0bac9b230f389c77d70383faf (main)  
Status: implementation plan; no implementation, simulation, or repository test execution is claimed here.

## 1. Build decision

Build a small, strict equilibrium-comparison pipeline first. Reuse the repository's production driver, fixed-state correctness APIs, Parquet transaction machinery, and MBAR solver. Its first experiment compares the existing contact coordinate with each of six newly fitted quadratic-residual torsion PCs, plus a contact-only baseline. Every force-valid component receives an actual sampling pilot. A later milestone selects both coordinates.

The objective is accurate equilibrium distributions and thermodynamics for the declared molecular Hamiltonian. Preserving physical kinetics, physical transition rates, and physical relaxation times is irrelevant. Correlation in the accelerated sampler matters only because it affects statistical uncertainty and computational efficiency.

The software must be able to finish successfully with an inconclusive scientific result. A small confidence interval does not establish accuracy. No finite diagnostic panel establishes global convergence of an arbitrary molecular distribution.

This plan resolves implementation choices left open in the original specification and incorporates its adversarial review. Where they conflict, use this plan for implementation. Keep the original specification and review as historical design evidence; do not silently rewrite them.

### Deliverable boundaries

| Milestone | Deliverable | Permitted claim |
|---|---|---|
| M0 — trustworthy measurements | Frozen state definitions, complete observations, deployable residual CVs, strict MBAR export | The software evaluates the declared sampling Hamiltonians consistently |
| M1 — validated comparison engine | Dependence-aware uncertainty, explicit decisions, controls, CLI and resumable runner | The comparison procedure passes its recorded control suite |
| M2 — fixed-primary experiment | Seven-arm prospective screen and fresh confirmation | Best observed supported companion to this specific contact CV, or insufficient evidence |
| M3 — joint-pair experiment | Both coordinate slots selectable from an explicit finite dictionary | Best observed supported tested pair/protocol, or insufficient evidence |

Statistically established efficiency superiority is an optional extension. M2 and M3 must report `advantage: NOT_TESTED` unless a separately validated comparison method is enabled. This avoids making a weakly powered 20% performance claim a prerequisite for useful equilibrium work.

## 2. Repository evidence and integration map

All existing paths below were checked at the pinned commit. New paths elsewhere in this document are proposals. Function names are better anchors than line numbers.

| Existing code | Observed behavior | Implementation consequence |
|---|---|---|
| `gareus/correctness/state_identity.py` | `make_state_definition`, `freeze_snapshot`, `write_frozen_snapshot`, and `validate_fixed_state_segments` already encode immutable CV/state definitions; same fixed table required across selected segments | Wire these into selector-enabled production. Do not invent another state-hashing format |
| `gareus/production.py::run_gareus`; `gareus/store.py::WindowSnapshot.snapshot` | Production calls the older snapshot writer, which records windows and CV type names without the embedded strict state definition | An early integration task must connect the existing strict API to the real writer |
| `gareus/correctness/export.py::build_export_arrays` | Strict fixed-state export; explicit immutable/quiescent boundary; rejects incomplete cross-state energies; checks recorded lambdas exactly against state definitions | Build a narrow sample-view adapter. Preserve rejection semantics |
| `gareus/store.py::ParquetSampleWriter` | Atomic manifested chunks; scalar physics fields currently use float32 | Selector observation mode needs float64 for CVs, raw boost energies, and lambda. In particular, float32 storage of a non-binary-exact lambda must not break the strict equality check |
| `gareus/parquet_manifest.py` | Manifest is authoritative; unlisted Parquet files are ignored; crash-safe compaction support | Put selector observations in the same sample rows and transaction, avoiding a second independently committed feature stream |
| `gareus/production.py::run_gareus` nested `sample` | Samples carry the live assignment before exchange; torsion observations currently use `DihedralObsBuffer` separately and may trigger another positions read | Capture selector features from the same frame as scalar observations, before exchange, with exact row identity |
| `gareus/tica.py::DihedralObsBuffer` | Writes per-replica NPZ arrays; no complete campaign/segment identity in those arrays; inactive CV2 may be NaN | Keep legacy tICA behavior compatible; do not use these legacy arrays as the selector's authoritative observations |
| `gareus/tica.py::compute_bootstrap_torsion_pca` | `component` is a count of leading PCs combined, not an individual PC index; its residualization differs from the handoff | New model kind and explicit one-based component index. Never redefine the existing argument |
| `smoke/scripts/cv_hunt.py` | Quadratic residualization and six individual components; sine block then cosine block | Reimplement the mathematical contract with stored coefficients and explicit feature IDs; old scores alone are insufficient |
| `gareus/tica.py::backbone_dihedral_features` | Runtime feature order interleaves sine/cosine | Use this ordering as canonical; convert legacy ordering explicitly |
| `gareus/production.py::_add_weighted_trig_torsion_force` | Uses `sin(-theta)`/`cos(-theta)` to match the stored dihedral convention | Reuse that convention and verify against NumPy and finite differences |
| `gareus/production.py::add_secondary_structure_cv_force`; `gareus/cv.py::secondary_structure_score_from_positions_nm` | Runtime force and numeric scorer currently handle existing secondary modes | Add a distinct residual model mode to both and to resume/reprojection paths |
| `gareus/query.py::reconstruct_bias_matrix`; `gareus/correctness/bias.py` | Canonical strict umbrella reconstruction, with kcal-to-kJ conversion | Delegate to these; selector must not own a duplicate bias equation |
| `gareus/mbar_analysis/ladder.py`; `estimators.py` | Canonical GaMD ladder contribution; extra exponential/cumulant methods excluded under ladder | Apply boost reweighting exactly once |
| `gareus/mbar_analysis/solvers.py` | `solve_mbar` and `logw_from_fk` already available | Reuse them behind a stable selector adapter; pin backend and solver settings |
| `gareus/correctness/sampling_policy.py` | Only `phase_kind=production` is equilibrium-analysis eligible | A screening study's fixed measurement phase is physically `production`; store its study role separately as `screen`. Do not label it `pilot` and weaken the existing eligibility rule |
| `gareus/windows.py::load_explicit_2d_window_csv` | Explicit two-axis window rows supported | Generate frozen tables; use manual production after preparation |
| `gareus/cli.py` | Manual mode and explicit window tables are supported; adaptive paths are separate | Invoke the existing CLI through a generated run configuration; keep adaptation out of measured phases |
| `gareus/swarm/` | Seed exploration, frozen envelope export, gates and ladder design exist | Reuse preparation outputs through explicit adapters; a seed bank is not an equilibrium reference |
| `gareus/synth/landscapes.py`, `sampler.py`, `sampler_langevin.py`, `oracle.py` | Known-distribution infrastructure exists | Extend it with comparison controls, including dependent sampling; exact IID sampling alone cannot validate time-series uncertainty |
| `.github/workflows/ci.yml` | Fast job currently runs three exchange test files; docs job builds MkDocs | Add targeted selector CPU tests; do not assume the existing CI already runs all new tests |
| `pyproject.toml` | NumPy core dependency; optional analysis, OpenMM and dev extras; Python >=3.10 | Keep selector imports lightweight and optional dependencies lazy |
| `.gitignore` | `*.md` ignored, with `docs/atlas-md/**` allowed | Put tracked plan/derivation/runbook under `docs/atlas-md/`; verify Git actually tracks them |

Sources: [fixed-state APIs](https://github.com/sulcjo/ATLaS-MD/blob/cdee6cc87ac540b0bac9b230f389c77d70383faf/gareus/correctness/state_identity.py), [strict export](https://github.com/sulcjo/ATLaS-MD/blob/cdee6cc87ac540b0bac9b230f389c77d70383faf/gareus/correctness/export.py), [production](https://github.com/sulcjo/ATLaS-MD/blob/cdee6cc87ac540b0bac9b230f389c77d70383faf/gareus/production.py), [sample writer](https://github.com/sulcjo/ATLaS-MD/blob/cdee6cc87ac540b0bac9b230f389c77d70383faf/gareus/store.py), [existing component behavior](https://github.com/sulcjo/ATLaS-MD/blob/cdee6cc87ac540b0bac9b230f389c77d70383faf/gareus/tica.py).

The inspection establishes integration requirements, not that a historical production result has been proven wrong. Existing test comments and CLAUDE.md contain historical statements; code and current tests take precedence when they disagree.

## 3. Non-negotiable scientific contracts

1. **One target per comparison.** Record sequence, termini/protonation, topology and atom order, force-field files, physical System serialization, constraints, masses, temperature, ensemble, pressure or fixed box, and periodic-boundary conventions. For NPT the instantaneous box is an observation, not part of immutable state identity. Numerical settings belong in protocol provenance.
2. **Frozen measurement.** CV coefficients, umbrella parameters, boost envelope, exchange policy, and integrator settings are fixed during each scored campaign. Preparation and discarded equilibration are distinct from measurement. An extension using a changed Hamiltonian starts a new experiment/protocol.
3. **Canonical reweighting.** At a common physical target, use the repository's reduced umbrella-plus-ladder bias and target bias zero. Different temperatures/pressures/physical Hamiltonians are unsupported in v1. Do not add a second GaMD correction.
4. **Exact observation association.** Never forward-fill a CV, attach a nearby trajectory frame, or reconstruct historical state parameters from current config. Missing authoritative data makes that campaign ineligible.
5. **One synchronized ensemble process.** v1 uncertainty accepts exactly one observation from each of K fixed sampling states at each retained reporting tick, collected before exchange. Resume boundaries, missing ticks, or unequal state occupancy are explicit cases, not silent flattening.
6. **Native-blind prospective selection.** Generate a new generic `broad` discovery bank or another explicitly declared native-blind discovery source. No native RMSD, native contact labels, folded references, historical winner labels, or fold-specific GENPEPT preset enters fitting, ranking, or stopping. Historical PC2 may be replayed only as a separate exploratory arm.
7. **Precision and accuracy remain separate.** Supported competing protocols must agree within declared observable tolerances before scoped equilibrium confirmation. Agreement is evidence, not proof that all protocols escaped common trapping.
8. **Abstention is a valid result.** Integrity failure, unresolved correlation, insufficient support, protocol disagreement, budget exhaustion, and unresolved ranking are separate machine-readable outcomes.
9. **No artificial orthogonality requirement.** Zero covariance and full local gradient rank are not thermodynamic admissibility conditions. Diagnose geometry, including manifold embeddings; do not reject the `(sin(theta), cos(theta))` control as useless.
10. **No frame-count evidence inflation.** Kish weight ESS and maximum per-frame weight are diagnostics. They are not independent sample counts or hard support certificates.

## 4. Minimal architecture and interfaces

Create modules when their task needs them. Avoid a plugin framework, a new MD engine, a general symbolic compiler, or a replacement MBAR solver.

| Proposed module | Responsibility | Must not do |
|---|---|---|
| `gareus/cv_selection/contracts.py` | Versioned models, plans, result enums, readiness validation | Launch jobs or infer missing scientific settings |
| `gareus/cv_selection/models.py` | Fit/load/evaluate six individual residual components and feature mappings | Change legacy PCA semantics or fit from confirmation data |
| `gareus/cv_selection/runtime.py` | Narrow force/evaluation integration for frozen models | Implement exchange, GaMD, or barostat algorithms |
| `gareus/cv_selection/data.py` | Immutable sample-view audit and strict export adapter | Repair missing values with interpolation or silently pool protocols |
| `gareus/cv_selection/estimate.py` | Existing MBAR solver adapter and target-observable estimates | Apply independent-frame uncertainty by default |
| `gareus/cv_selection/uncertainty.py` | Joint-ensemble influence series, batch covariance, reference resampling | Certify stationarity from a short correlation estimate |
| `gareus/cv_selection/evaluation.py` | Frozen observable panel, diagnostics, comparison and deterministic decision | Use candidate-specific evaluation panels |
| `gareus/cv_selection/trials.py` | Layout compilation, resource accounting, launch/resume ledger | Alter a frozen protocol in response to a favorable result |
| `gareus/cv_selection/cli.py` and `__main__.py` | Command dispatch, reports, exit codes | Hide launch behind an analysis or plan command |

This is nine focused modules, introduced incrementally. Keep shared physics and persistence in existing packages. Later joint-pair support extends models/runtime/trials; split a file only when its responsibilities actually require it.

### Proposed Python seams

These are new APIs, not claims about existing functions.

~~~python
validate_protocol(protocol, artifacts, *, stage) -> ReadinessReport
fit_residual_components(features, primary_values, discovery_manifest) -> CandidateSet
evaluate_model(model, features, primary_values) -> ndarray
compile_secondary_force(openmm, system, model, primary_definition) -> ForceMetadata
load_campaign_view(campaign_path, measurement_plan) -> CampaignView
estimate_campaign(view, panel, solver_spec) -> CampaignEstimate
estimate_joint_uncertainty(view, fit, observables, uncertainty_spec) -> UncertaintyResult
compare_protocols(estimates, evaluation_spec) -> Decision
compile_trial_plan(protocol, candidates, preparation) -> TrialPlan
execute_trial(trial_plan, trial_id, *, resume=False) -> TrialStatus
~~~

Use dataclasses plus explicit strict validation, NumPy arrays for numerical data, and existing strict JSON utilities. No Pydantic dependency is needed. Unknown schema versions, unknown fields, nonfinite persisted numbers, and invalid enum values fail. Structured results contain reason codes and evidence paths, not just strings.

### Artifact ownership

A study directory contains immutable input artifacts and append-only run records:

~~~text
protocol.json
input_manifest.json
feature_schema.json
candidate_set.json
observable_panel.json
preparation/frozen_envelope.json
plans/screen.json
plans/confirm.json
readiness/screen.json
readiness/confirm.json
calibration/certificate.json
trials/<trial_id>/effective_config.json
trials/<trial_id>/run/                 # existing production outputs
trials/<trial_id>/measurement.json
analysis/<input_signature>/campaigns/<trial_id>.json
analysis/<input_signature>/decision.json
analysis/<input_signature>/report.md
~~~

Paths are references; hashes establish identity. Compute each artifact digest over the existing canonical JSON encoding of its scientific payload, excluding its own digest/signature field; put provenance locations outside the embedded CV definition because the strict state API forbids path-based CV identity. Changing any input creates a new analysis signature. Reports may summarize a run; they cannot change its eligibility. A partial analysis must never leave an old successful decision looking current.

## 5. Exact data and model contracts

### 5.1 Observation schema

Add an opt-in selector schema to `ParquetSampleWriter`; keep legacy callers valid. The opt-in is fixed for a segment. Selection features and scalar energies share each Parquet row and commit, so a crash cannot independently advance one stream.

| Field | Type/meaning |
|---|---|
| `campaign_id` | Immutable independent-run identity, also recorded in run metadata |
| `segment_id` | Existing segment identity, supplied explicitly to the selector view |
| `replica` | Physical walker identifier; not the current state identifier |
| `step` | Existing integer step semantics; do not relabel as globally unique |
| `ensemble_tick` | Monotone reporting tick within a continuity chain |
| `continuity_id` | Identifies uninterrupted trajectory history; restored exactly or starts a new chain |
| `window_id` | Actual pre-exchange generating-state ID |
| `state_definition_sha256` | Link to the existing canonical fixed-state snapshot; may be immutable Parquet metadata rather than repeated text |
| `cv1`, `cv2` | float64 values of deployed models, even where the corresponding stiffness is zero |
| `torsion_features` | Fixed-width list of float64 values with feature-schema hash |
| `rg_nm`, `end_to_end_nm` | Declared structural observables, with exact atom definitions |
| `box_vectors_nm` | Nine float64 values for NPT geometry/auditing; optional only for explicitly nonperiodic controls |
| raw energy channels, boosts, lambda | float64; preserve existing units and canonical names |

The unique row key is `(campaign_id, segment_id, replica, step)`. In addition, `(campaign_id, continuity_id, ensemble_tick, replica)` must be unique. Restarting from an earlier checkpoint must not retain the abandoned future tail as a second trajectory. Pin the selected segments and measurement boundary before reading.

Only every declared selector reporting interval needs the feature columns; v1 selects that synchronized grid and requires complete rows there. Off-grid scalar rows may remain for existing outputs. Encode observation presence explicitly; null off-grid features are not eligible selection samples. Recompute `N_k` from the selected grid.

Require all K replicas and a permutation of all K state IDs at each eligible tick. v1 rejects incomplete ticks anywhere inside the declared measurement interval; it does not condition the sample set on which values happened to survive. A trailing partial tick from interruption may be excluded at the common committed boundary and recorded. Resume may concatenate only checkpoint-verified continuous chains. Otherwise analyze chains separately and return unsupported for the single-chain covariance mode.

Compute torsions and contact values with the repository's declared PBC convention. Test wrapped versus equivalent unwrapped peptide coordinates; use molecule-consistent coordinates for Rg/end-to-end. Do not switch a runtime minimum-image definition to an offline whole-molecule definition without changing model identity.

### 5.2 Frozen residual model

Canonical feature vector uses ordered phi then psi torsions, with interleaved sine/cosine for each torsion, matching `backbone_dihedral_features`. Store feature IDs including atom quadruplets, residue/atom identities, trig function, and dihedral sign convention. A matching vector length is insufficient.

Fit on the discovery set only, with explicit uniform structural weights for the initial bank-based model. These weights do not purport to be equilibrium weights.

1. Sort discovery rows by immutable row ID; reject duplicate identities and nonfinite input.
2. Standardize primary contact value using stored discovery mean and population standard deviation: `a=(c-mu_c)/sigma_c`; reject `sigma_c <= 1e-12`.
3. Solve least squares `X ~= D B` with `D=[1,a,a^2]`, NumPy `lstsq(rcond=1e-12)`, requiring rank three. Store all coefficients, feature ordering, and scaling.
4. Compute `R=X-D B`. Center R explicitly, store its tiny/nonzero numerical mean, then use full deterministic SVD with descending singular values.
5. For j=1..6, retain an individual right singular vector. Choose sign so its largest-magnitude coefficient is positive; tie by feature order. Store singular values and library versions. If an eigenvalue tie makes a subspace basis unstable, flag it; frozen coefficients, not the label PCj, define the subsequent experiment.
6. Define `z_j=(v_j dot (X-D B-mean_R)-mu_j)/sigma_j`, with stored discovery mean and population SD. Reject only numerical zero variance, invalid geometry/forces, or unsupported model—not low explained variance.

The model JSON contains the complete coefficients, all scaling factors, one-based `component_index`, training-row digest, target/topology identity, feature-schema digest, and `kind=quadratic-residual-torsion-pc-v1`. Historical imports require an explicit feature permutation and complete model coefficients; scores-only NPZ data are not a deployable model.

Force evaluation must include the derivative of the primary-dependent subtraction:

$$
\nabla z_j={1\over\sigma_j}\left[\sum_m v_{jm}\nabla X_m
-\left(v_j\!\cdot B_1+2a\,v_j\!\cdot B_2\right){\nabla c\over\sigma_c}\right].
$$

OpenMM automatic differentiation is suitable if the expression includes both the torsion projection and the actual primary contact CV. Do not approximate this by a torsion-only projection or reuse an OpenMM Force object owned by another parent. Aggregate weighted trig terms to respect CustomCVForce's variable limit. Preserve umbrella force groups and the Pep-GaMD exclusion of umbrella energy from physical boost targets.

### 5.3 Target and state hashing

Use `correctness.state_identity` unmodified where possible. Its physical-system hash must refer to the unbiased physical system and the declared particle/topology/measure definition, not a biased System with arm-specific umbrellas. Hash constraints and masses consistently. Integrator settings and observation cadence belong in the protocol hash as well.

Build strict snapshots only after final window/start-state reconciliation. In selector mode, dropped states, changed coefficients, or a changed frozen envelope abort the planned trial before measurement. No auto-drop repair may quietly change an arm's planned state set.

For the one-CV arm, `cv2` in the physical state definition may be null. Nevertheless store the full torsion features so all candidate projections remain measurable offline. For a two-CV arm with a zero-stiffness secondary state, CV2 still exists and is measured.

## 6. Statistical implementation contract

### 6.1 MBAR adapter

Analyze each independent campaign separately in v1; do not pool all campaigns first and erase between-run disagreement. Call the strict export core with a pinned eligible view and a factory constructing the envelope from the embedded frozen definition. Then call the existing `solve_mbar`, using explicit `backend=anderson`, `tol=1e-10`, and `maxiter=10000` for the reference implementation. Faster backends may be added after equivalence/performance checks and must be recorded.

Treat nonconvergence, nonfinite weights, missing states, and numerical non-identifiability as failures. Verify the returned solution with a gauge-fixed residual check in the adapter; do not depend only on a generic success flag. Compute normalized target log weights with the existing solver helper. The physical target has zero bias under the supported common-base convention.

Retain full chronological synchronized data. Do not independently decorrelate each window and then feed the resulting irregular subset to the joint-ensemble uncertainty estimator. Existing strict subsampling tools can provide diagnostics and cross-checks.

### 6.2 Joint influence covariance: bounded v1 scope

Use joint-state uncertainty, not independent-state error bars. The primary source explicitly covers joint replica-exchange sampling; its assumptions still require suitable stationary sampling and resolved dependence. [Li et al., main Section IV and Supplement S2.2](https://arxiv.org/html/2203.01227v1)

Implement the following estimating equations for exactly one walker in each of K sampling states at each tick, hence `N_k=T` and `N=K*T`. They also specify the normalization that an implementing agent must derive and test.

Let

$$
D_n=\sum_kN_k\exp(f_k-u_{nk}),\quad
q_{nk}=N_k\exp(f_k-u_{nk})/D_n,\quad h_n=N/D_n.
$$

Fix `f_0=0` and omit its equation. For tick t, containing K samples, use

$$
g_{t,j}=\sum_{r=1}^Kq_{tr,j}-1\quad(j=1,\ldots,K-1),
$$

$$
g_{t,A}=K^{-1}\sum_{r=1}^Kh_{tr}(A_{tr}-\mu_A).
$$

An overall constant rescaling of h cancels; evaluate it in log space to avoid overflow. Hold the chosen numerical scale constant when checking the Jacobian by finite differences. Use the gauge-fixed Jacobian `J=mean_t(dg_t/dtheta)` for `theta=(f_1,...,f_{K-1},mu_1,...,mu_M)`. Implement analytical derivatives and verify them independently by finite differences. The influence sequence is `psi_t=-solve(J,g_t)`. Never invert J explicitly or add an undocumented ridge to make a disconnected system look estimable. The initial numerical-identifiability guard rejects a gauge-fixed free-energy Jacobian with singular-value ratio <=1e-12; record its spectrum and distinguish numerical failure from physical support diagnostics. Calibration must exercise this guard.

For batch length b, split the influence sequence into a complete prefix of A nonoverlapping batches. Estimate

$$
\widehat\Omega_b={b\over A-1}\sum_{a=1}^{A}
(\bar\psi_a-\bar\psi)(\bar\psi_a-\bar\psi)^T,
\qquad \widehat{\mathrm{Cov}}_b(\widehat\theta)=\widehat\Omega_b/T.
$$

Record the omitted covariance-estimation tail; point estimation still uses the declared full measurement set. This is an asymptotic estimator, not an exact finite-sample confidence statement.

Initial engineering policy `joint-batch-v1`:

- Candidate batch lengths: `b0=ceil(sqrt(T))`, `2*b0`, `4*b0` in ticks.
- Require at least 20 complete batches at each length and a discarded covariance tail <=5% at each length; otherwise return `UNRESOLVED_CORRELATION`.
- Use the maximum variance across these lengths for each observable; do not assemble a purported covariance matrix by elementwise maxima. Preserve each complete covariance matrix separately.
- Flag variance instability if the largest/smallest strictly positive variance exceeds 2 for a required observable; treat zero variance without a valid support argument as unresolved.
- Reference check: synchronized moving-block resampling, whole ensemble per tick, with MBAR refitted per replicate. Blocks never cross unverified continuity boundaries. Use 199 resamples for selected calibration cases, not thousands of refits on every production campaign.
- A supported campaign needs a valid influence result and the calibrated sensitivity checks, not merely `T >= some minimum`.

These numbers are proposed algorithm parameters to test in T08, not already demonstrated universal thresholds. Calibration may reject this policy; scientific readiness remains false until the final parameterized policy passes the controls. Do not tune it after seeing which chignolin candidate wins.

For support sensitivity, refit after deleting the five largest-influence nonoverlapping time blocks (rank by maximum primary normalized influence; deterministic ties by time), plus first/last blocks if different. Also refit on regular thinning factors 2 and 4. Recompute counts and MBAR normalization each time. A deletion or thinning shift greater than an observable's declared tolerance, or a thinning variance ratio outside [0.5, 2], marks sensitivity unresolved. If a thinned sequence is too short for the same dependence checks, report that limitation; do not claim a passed cadence check. Compare at the same physical elapsed duration and report the different recording intervals. These are calibrated robustness diagnostics, not independent new data or universal bounds.

Keep peak memory bounded: accumulate J and influence projections in chunks; store only the observable influence series and required diagnostics when possible. K=96, N=10^6 costs approximately 768 MB for one float64 N-by-K array before solver copies; budget those copies explicitly.

### 6.3 Across-campaign estimates and intervals

Require equal planned cost and equal measurement policy across the R independent campaigns being combined. Use the arithmetic mean of campaign estimates; do not inverse-variance weight apparently precise trapped campaigns more heavily.

For observable a, use

$$
\bar\mu_a=R^{-1}\sum_r\widehat\mu_{ra},\quad
v_a=\max\left\{R^{-2}\sum_r\widehat v_{ra},\ s_a^2/R\right\}.
$$

Use complete planned campaigns; do not selectively omit a disagreeing campaign or reuse its seed until it becomes favorable. A failed campaign leaves the planned comparison incomplete unless the predeclared replacement rule uses a fresh campaign ID and retains the failure in the report. For a family of F reported primary arm-observable means, the initial conservative interval is `mean +/- t(R-1, 1-alpha/(2F))*sqrt(v)`, with `alpha=0.05`. Freeze F before confirmation. Label the interval method and its calibrated/asymptotic status. R=3 screening estimates are exploratory; start confirmation design evaluation at R=6 per compared arm, then use control/runtime evidence to set the final R and duration. Neither number is a convergence guarantee.

Intervals for individual campaigns and first/second halves are diagnostic families, calculated with their own recorded multiplicity correction and dependence estimate. An estimable difference exceeding its practical tolerance even after interval uncertainty is a conflict. An unresolvable half-run uncertainty is an unresolved temporal diagnostic, not evidence of stationarity.

### 6.4 Cross-protocol agreement

For two arm estimates and their simultaneous primary intervals, form a conservative difference interval

`[mean_A-mean_B-(h_A+h_B), mean_A-mean_B+(h_A+h_B)]`.

For practical agreement tolerance delta:

- Entire interval inside `[-delta,+delta]`: `AGREEMENT_SUPPORTED` for that observable.
- Entire interval above +delta or below -delta: `PROTOCOL_DISAGREEMENT`.
- Otherwise: `AGREEMENT_UNRESOLVED`.

Require supported agreement across the frozen primary panel for scoped confirmation of the selected candidate against the baseline. A precise candidate with an under-resolved baseline can be provisionally useful, but cannot claim confirmed cross-protocol agreement. A baseline is never assumed to be truth. Resolve disagreement with new exploration, never by choosing the narrowest interval.

### 6.5 Region populations and free energies

Every requested region has `RESOLVED_POPULATION`, `MASS_BOUNDED_SMALL`, or `UNRESOLVED_SUPPORT`. The v1 molecular implementation supports the first and third. The second is available only for a separately justified bound with recorded assumptions; the initial exact zero-count binomial bound is limited to IID target controls. Never substitute Kish ESS into it for weighted MD.

For a population ratio free energy, transform simultaneous positive population intervals conservatively:

`DeltaF/kT in [-log(U_B/L_A), -log(L_B/U_A)]`.

If a necessary lower bound is zero, report an unbounded/one-sided result and `FREE_ENERGY_UNRESOLVED`, not a clipped finite number. Store unbounded endpoints with explicit endpoint-kind fields rather than invalid JSON infinity. Passing an absolute population tolerance does not imply useful log-probability accuracy.

## 7. Evaluation, ranking, and readiness policies

### 7.1 Frozen panel

The compact initial primary panel has eight CDF probabilities (indicator convention: value <= threshold): Rg below discovery quantiles 0.2, 0.4, 0.6, 0.8, and end-to-end distance below the same four quantiles. Atom sets and quantile algorithm (`numpy.quantile`, linear interpolation) are frozen. Initial absolute half-width tolerance is 0.02 and cross-protocol practical difference tolerance is 0.02 for each probability. These are explicit proposed study tolerances, not estimates of achievable accuracy.

Define heavy-atom Rg by the unweighted centroid and mean squared heavy-atom distance, and end-to-end distance by the first and last peptide alpha carbons; freeze exact atom IDs. Define histogram edges as 20 discovery quantile bins with explicit underflow/overflow bins; collapsing edges cause preparation failure rather than outcome-dependent merging. If thresholds collapse or the intended thermodynamic question requires different quantities, fail panel preparation and create a revised discovery-stage panel before any screening. Do not repair the panel using arm outcomes. Optional scientifically important basin/free-energy observables must be defined and assigned tolerances before the study; absent user-provided basins, do not invent a native basin.

The broad diagnostic panel contains all backbone sine/cosine means, contact/Rg/end-to-end marginal histograms, and a joint structural partition. This supplements the primary panel so matching two marginal distributions alone cannot masquerade as distributional agreement.

Use 16 frozen structural centers selected by deterministic farthest-point traversal on discovery features: all torsion sin/cos terms, contact, Rg, and end-to-end. Center scalar features at discovery means and divide by discovery population SD. Center the torsion feature block at its discovery mean and divide the whole block by the square root of its discovery mean squared Euclidean norm; keep all sine/cosine features. Reject zero scale. This keeps the number of torsions from arbitrarily dominating. First center is the smallest immutable discovery row ID; tie distances by row ID. Assign every configuration to its nearest center. Record radii as the within-cell 99th percentile discovery distances; a point outside its assigned radius is novel. Zero-radius cells require exact matches. This is a diagnostic geometric partition, not an equilibrium basin definition.

Discovery cells are not all mandatory-visit cells. Report their estimated mass/support, including unresolved cells. A supported cross-protocol difference beyond 0.02 in a joint-cell population or bounded diagnostic mean (normalize sine/cosine to [0,1]) triggers a distributional conflict. Use a separate recorded Bonferroni family for these diagnostic comparisons. If the lower confidence bound on novel mass exceeds 0.02, return `PANEL_REVISION_REQUIRED`; if its bound is unavailable, record `NOVELTY_UNRESOLVED`. A revised panel starts a new study and requires fresh confirmation.

Support rules for the initial implementation are deliberately scoped. A required primary probability must have a nonzero finite uncertainty estimate, contributions from its event and complement in at least two independent campaigns, resolved dependence, and passed block/cadence sensitivity. Otherwise its status is unresolved; a separately valid small-mass bound can later replace the event-visit condition. These conditions are diagnostics for the declared panel, not a proof against common trapping. User-declared free-energy regions must additionally meet their own population and log-ratio requirements.

Unvisited diagnostic partition cells and zero observed novelty do not automatically fail the primary panel; report their unresolved mass and the resulting limit on distributional coverage. Supported diagnostic conflicts do block confirmation. Significant positive novelty requires panel revision as above. Thus `SUPPORTED_ON_DECLARED_PANEL` can coexist with an explicit list of unresolved diagnostic regions; it must never be rendered as exhaustive coverage. Diagnostic uncertainty that prevents assessing a specific observed discrepancy is a blocker, not grounds for dismissing that discrepancy.

This panel is deliberately finite. The decision report must list unassessed and unresolved support rather than claim that zero observed novelty proves exhaustive coverage.

### 7.2 Deterministic result model

Use separate fields, not one overloaded pass/fail score:

| Field | Values |
|---|---|
| `integrity` | `PASS`, `FAIL` |
| `precision` | `MET`, `NOT_MET`, `UNRESOLVED` |
| `dependence` | `SUPPORTED`, `UNRESOLVED_CORRELATION` |
| `reproducibility` | `NO_CONFLICT_DETECTED`, `CONFLICT`, `UNRESOLVED` |
| `cross_protocol` | `AGREEMENT_SUPPORTED`, `PROTOCOL_DISAGREEMENT`, `AGREEMENT_UNRESOLVED`, `NOT_COMPARED` |
| `support` | `SUPPORTED_ON_DECLARED_PANEL`, `LIMITED`, `UNRESOLVED` |
| `advantage` | `NOT_TESTED`, `SUPPORTED`, `UNRESOLVED` |
| `decision` | `INVALID_INPUT`, `INSUFFICIENT_EVIDENCE`, `PROVISIONAL_CHOICE`, `CONFIRMED_FOR_DECLARED_PANEL`, `PANEL_REVISION_REQUIRED` |

All outcomes include `reasons[]`, `unresolved_regions[]`, `tested_protocol_ids[]`, and input hashes. `CONFIRMED_FOR_DECLARED_PANEL` does not mean globally correct thermodynamics or a statistically superior CV pair.

Decision order:

1. Reject invalid inputs and ineligible phases before calculating scores.
2. Mark per-arm estimates/dependence/support. Do not replace failures with finite sentinel scores.
3. Any credible supported cross-protocol/distributional conflict blocks confirmation for the comparison as a whole. Preserve valid estimates in the report.
4. During screening, rank eligible arms by `L=max_a(h_a/epsilon_a)^2` at the matched measured cost. This is conditional precision, not unknown bias/MSE. Arms with unresolved support stay unranked.
5. Use a practical tie band of 5% in L. Prefer the contact-only baseline within the band; otherwise fewer active CV terms, then lexicographic immutable protocol ID. Store the exact tie rule.
6. Choose at most one candidate for fresh confirmation against the baseline. If the baseline is provisionally best, retain it and report that a two-CV gain was not established; a two-CV output is not mandatory.
7. Confirmation uses a frozen chosen model/layout and fresh campaigns. Precision met, no blocking diagnostics, adequate declared support, and supported primary agreement permit `CONFIRMED_FOR_DECLARED_PANEL`. Anything less receives a narrower outcome.
8. A candidate failing confirmation cannot be replaced using the same confirmation data as a new selection set. A new comparison has a new manifest and fresh confirmation.

### 7.3 Readiness has explicit levels

`build-ready` means all schemas/algorithms for a task are specified. `engineering-ready` permits bounded smoke/throughput runs. `screen-ready` additionally requires validated controls, frozen discovery models/panel, target identity, complete windows/rungs, burn-in and measured budget. `confirm-ready` additionally requires a frozen candidate, fresh run identities, declared interval family and confirmation allocation.

A readiness report enumerates missing requirements. A molecular duration, burn-in length, or boost envelope cannot be sensibly invented in this document. The engineering/calibration tasks produce those concrete artifacts. Until then dry-run planning works and scientific launch is rejected with `PROTOCOL_NOT_READY`. This is an implementation dependency, not a request to the user for arbitrary defaults.

## 8. Trial design and resource policy

### Fixed-primary experiment

Seven arms: `contact-only`, `contact+residual-pc01`, ..., `contact+residual-pc06`. New discovery coefficients are shared and frozen before screen. The original contact definition is copied by complete parameters from the chosen target config, not reconstructed from its comments or a remembered label. The inspected chignolin_8 leaf values include normalized heavy-atom contacts, sequence separation >=4, r0=12 A and beta=3/A; its older comments contain obsolete settings. Record effective parser output.

Initial layout policy for engineering/calibration is `grid24-v1`: 24 primary centers for baseline; six primary by four secondary centers for each two-CV arm. Four common frozen lambda rungs give 96 states. Use explicit CSVs and one replica per state. This simple layout replaces the original more elaborate bridge/axis/patch allocation for the first implementation. It is a tested sampling design, not a claim of optimal geometry.

Centers are discovery quantiles uniformly spaced from 0.02 to 0.98, calculated separately per deployed coordinate. Duplicate centers cause an explicit layout failure. For each active axis, define local spacing as the smaller adjacent positive center gap, endpoints using the sole neighbor. Set initial `sigma=spacing/1.5`, then `k=RT/sigma^2` in kcal/mol per squared deployed-coordinate unit. Bounds `k_min`/`k_max` are required preparation parameters; record any clamp and actual resulting width. Do not claim a flat-PMF overlap heuristic certifies molecular overlap. A protocol with poor measured connectivity is not cleared by predicted geometry.

The common four-rung list comes from validated preparation and includes lambda=0; do not force the maximum to one. If preparation does not provide four admissible rungs, emit a new versioned layout/budget proposal within 64–128 replicas; do not silently change state count. Preserve the actual boost formula, including its behavior outside the fitted energy range. No force-scaling clamp or boost modification is part of this plan.

Run preparation through the existing system/swarm/envelope machinery. Launch frozen measurement through `window_mode=manual` with `windows_2d_csv`, fixed model, fixed setup/envelope, and selector observation mode. Disable adaptive CV updates, window addition/deletion, forced rescue/reseeding and envelope refits in measurement. Burn-in is a fixed predeclared interval established by setup/calibration; failure of later diagnostics does not authorize cherry-picking a new cutoff.

### Costs and execution

The target node has 4 L40S GPUs and 192 CPU threads. Initially run one multi-replica campaign at a time across the node; do not launch seven 96-replica campaigns concurrently. Keep replica/OpenMM thread allocation under the existing CPU-budget resolver; cap analysis BLAS threads separately and record settings. Optional concurrency changes require a measured throughput comparison.

Measure wall time, reserved GPU-hours, CPU analysis time, peak RAM, bytes written, setup/burn-in cost and retained ticks separately. GPU cost is reserved GPU count times elapsed hours, not replica count times wall time. Report marginal sampling cost and fully amortized study cost separately.

No scientific GPU-hour total is asserted before benchmarking. The earlier 288-hour illustration used a different arm count and does not apply automatically to seven arms. The budget compiler calculates:

`shared_preparation + 7*R_screen*C_screen + 2*R_confirm*C_confirm + contingency`.

When only baseline is retained, confirmation design changes explicitly rather than charging imaginary second-arm jobs. Screening is exploratory; avoid treating more short runs as a free substitute for sufficiently long trajectories. Stop at the declared fixed budget, checkpoint cleanly, analyze the common complete prefix, and report insufficient evidence when needed. No sequential significance peeking in v1.

## 9. Agent work packages

Each package is a bounded PR or a small explicit PR series. An agent receives one package plus its listed contracts and prerequisite outputs. Do not ask an agent to “implement the whole spec.” Existing modules with many call sites have one designated integration owner at a time.

### T00 — Baseline, contracts and test entry point

**Depends on:** nothing. **Owns:** new `contracts.py`, package skeleton, `pyproject.toml` test/dependency registration, proposed `tests/test_cv_selection_contracts.py`, tracked developer plan.

**Read first:** this plan; `pyproject.toml`; `.github/workflows/ci.yml`; `.gitignore`; `correctness/_io.py`; `correctness/state_identity.py`; `config.py`.

**Implement:** strict versioned dataclasses/JSON for CandidateSet, FeatureSchema, ObservablePanel, ProtocolSpec, TrialPlan, ReadinessReport and Decision. Separate study role from physical sampling phase. Define enums and required hashes/units. Add a fixtures directory under `tests/fixtures/cv_selection/` with tiny generated data, not molecular run dumps. Use existing canonical JSON/hash helpers.

**Acceptance:** unknown fields/versions, NaNs, duplicate JSON keys, missing policies, wrong feature identity, and changed immutable artifacts fail with stable reason codes. Imports work with NumPy-only core installation. Record pre-existing failures in relevant regression tests at the actual working commit; no historical “2147 passed” claim substitutes for a current run.

**Do not:** implement scientific defaults by guessing missing duration/envelope, or edit old PCA behavior.

### T01 — Connect frozen state definitions to production

**Depends on:** T00. **Owns:** narrow hooks in `production.py`, `store.py::WindowSnapshot`, `provenance.py`; tests `test_cv_selection_state_wiring.py`.

**Implement:** build selector-mode strict state definitions from the actual finalized physical system, CV definitions, window table and embedded boost envelope; write immutable snapshots through `correctness.state_identity`. Preserve existing compatibility fields. Restore and compare definitions before resumed measurement; a changed model, lambda, topology, temperature, pressure, force constant or envelope is fatal. Record engine/environment/integrator settings separately. Use production eligibility only for fixed measurements.

**Acceptance:** drive the real hook from an instrumented production harness; different source paths with identical embedded definitions agree, identical source paths with changed coefficients disagree. A state permutation preserves correct ID-to-column mapping. A resume attempts to change the envelope and is rejected before sampling. Ordinary legacy runs retain their existing behavior.

**Regression focus:** `tests/test_stage_phase_identity.py`, `tests/test_resume_secondary_cv_reconciliation.py`, `tests/test_segment_finalization.py`, `tests/test_lambda_ladder_states.py`.

**Do not:** replace canonical state identity with a path hash, or infer missing old coefficients from today's files.

### T02 — Atomic complete observations and strict sample view

**Depends on:** T00, T01. **Owns:** selector-mode sample schema in `store.py`, sampling hook in `production.py`, new `data.py`, tests `test_cv_selection_observations.py` and `test_cv_selection_sample_view.py`.

**Implement:** schema in Section 5.1, same-frame feature extraction, explicit selector tick cadence, float64 physics fields, pre-exchange labels, pinned manifest view, exact uniqueness and synchronized occupancy checks. Reuse positions already fetched by `sample` where possible. Buffer feature rows with their scalar rows and use existing manifest transactions/compaction. Ensure existing query projections tolerate the extra fields. Preserve legacy `DihedralObsBuffer` for other workflows.

**Acceptance:** a behavioral test runs two walkers through sample → accepted/rejected exchange → next sample and reads back manifested Parquet, proving features, window labels and energies refer to the same frames. Tests cover crash before manifest publication, compaction without double counting, resume from an earlier checkpoint, duplicate row IDs, missing interior tick, and trailing partial tick. A non-binary lambda such as 0.3 survives round-trip exact comparison with its frozen float64 value. Required values cannot be forward-filled.

**Regression focus:** `test_store.py`, `test_parquet_manifest_transactions.py`, `test_query_concat_schema_evolution.py`, `test_sample_before_exchange_ordering.py`, `test_resume_round_trip.py`.

**Do not:** count structural observations using a separate file-order join or weaken strict export to accept rounded lambdas. Migrating old data is a separate, explicitly scoped operation.

### T03 — Individual quadratic-residual candidate models

**Depends on:** T00. **Owns:** new `models.py`, tests `test_cv_selection_models.py`, feature/model fixtures.

**Implement:** complete Section 5.2 algorithm; native-blind input manifest validation; six independent component artifacts; explicit historical feature permutation when sufficient source data exists; model round-trip and stable digest.

**Acceptance:** a constructed `X=D B + noise` case removes quadratic dependence in fitted residuals; NumPy reload projection agrees to tight float64 tolerance; the six models are individual SVD directions rather than variance-weighted mixtures. Scrambling feature IDs fails even at the same vector width. Historical sine-block/cosine-block data produce matching projections only after the declared permutation. Zero variance and rank-deficient regression fail explicitly. Nearly tied components are flagged without secretly rotating a frozen basis.

**Regression focus:** `test_bootstrap_torsion_cv.py`, `test_tica.py`, `test_genpept_cv2_overlay.py`.

**Do not:** alter `compute_bootstrap_torsion_pca(component=...)`, rank candidates by native overlap, or use low explained variance as an exclusion.

### T04 — Residual CV force, runtime evaluation and resume

**Depends on:** T01, T03. **Owns:** new `runtime.py`; narrow dispatch in `cv.py`, `production.py`, `cli.py`; extension to `mbar_analysis/cv2_reprojection.py`; tests `test_cv_selection_forces.py`, `test_cv_selection_runtime.py`.

**Implement:** new secondary mode `residual-torsion-pc` with a frozen model artifact argument; same NumPy/OpenMM/reprojection expression; primary-dependent chain-rule term; correct trig sign; aggregated torsion forces; metadata/resume validation; evaluation even at k2=0. Reuse primary contact definition and force construction through a small factory rather than copying its equation into multiple files. Persist the model contents in state identity.

**Acceptance:** on several nonsingular peptide-like configurations, NumPy CV matches the OpenMM CV; umbrella energy matches canonical reconstruction; analytic OpenMM force matches central finite differences, including perturbations that predominantly alter contact CV1. Test PBC-equivalent configurations, lambda=0/nonzero, inactive restraints and serialized/reloaded models. A mutation omitting the CV1 derivative or changing sine sign must fail. Explicitly check physical boost energy excludes both umbrella force groups. Exercise actual OpenMM CPU/Reference, not only fake force objects. For trajectory-based checks, account for the documented first-step integrator behavior and assert that the measured atoms actually moved; a no-motion comparison is not force validation.

**Regression focus:** `test_secondary_cv_restraint_sign_convention.py`, `test_bootstrap_torsion_cv.py`, `test_cv2_reprojection.py`, `test_pep_gamd_bias_force_groups.py`, `test_fast_cv_path.py`.

**Do not:** rewrite unrelated secondary-structure modes, add a kinetic optimization objective, or nest shared Force objects illegally.

### T05 — Strict MBAR estimation adapter

**Depends on:** T01, T02, T03, T04. **Owns:** new `estimate.py`, data/export glue; tests `test_cv_selection_estimation.py`.

**Implement:** immutable view → strict fixed-state export → existing MBAR solver → normalized target estimates. Verify target identity, counts, gauge-fixed residuals and state/sample correspondence. Keep campaigns separate. Export hashes cover actual arrays and pinned boundaries. Extend the selector analysis signature with the complete feature/observable arrays and their definitions; the existing fixed-state export signature alone does not cover a newly added observable column. Evaluate all declared observables on exactly the same retained rows.

**Acceptance:** tiny known-distribution controls recover target means and populations; state/row permutation and adding a common per-sample reduced potential leave target results invariant. Wrong stiffness units, doubled boost, missing raw energy, altered state definition and stale cache are detected or produce failing oracle results. Runtime bias energy and reconstructed all-state energies agree on actual sampled configurations. A caller cannot turn failed normalization into a score.

**Regression focus:** `test_physics_oracle.py`, `test_query_reconstruct_bias_matrix_nan_guard.py`, `test_lambda_ladder_mbar.py`, `test_estimator_excludes_cumulant_under_ladder.py`, `test_mbar_analysis_solvers_module.py`.

**Do not:** write a selector-specific GaMD formula or pool incompatible fixed-state tables through a fabricated row-dependent matrix.

### T06 — Dependence-aware uncertainty

**Depends on:** T02, T05. **Owns:** new `uncertainty.py`, tracked derivation `docs/atlas-md/developer/cv-selection-uncertainty.md`, tests `test_cv_selection_uncertainty.py`.

**Implement:** estimating equations, gauge-fixed Jacobian, observable influence series, batch policy, failure reasons, independent-campaign aggregation, and synchronized block-refit reference from Section 6. Provide deterministic RNG seeds. Return covariance assumptions and batch sensitivity, not just stderr.

**Acceptance:** analytical Jacobian matches finite differences; IID equal-state controls agree with direct replicate variance; a known correlated joint process shows enlarged uncertainty; changing replica labels without changing ensemble data leaves estimates unchanged. A disconnected overlap problem is unresolved rather than regularized into confidence. Denser recording of the same continuous process cannot manufacture a stronger support conclusion. Unequal occupancy and missing interior ticks reject the v1 joint mode. Benchmark memory and wall time against the block-refit reference.

**Reference tests:** compare against an independently generated control distribution and empirical repeated-campaign variance; do not validate only against a second function sharing the same derivative code.

**Do not:** assume replica walkers are independent trajectories or report a correlation time for physical dynamics.

### T07 — Observable panel and honest decision engine

**Depends on:** T00, T03, T05, T06. **Owns:** new `evaluation.py`, tests `test_cv_selection_evaluation.py` and `test_cv_selection_decision.py`.

**Implement:** fixed primary/broad panels; deterministic structural partition; mass/free-energy status; simultaneous intervals; explicit cross-protocol equivalence; ranking/ties/abstention from Section 7. Structured evidence explains every decision. The confirmation evaluator accepts a pre-frozen candidate, not an arbitrary new winner.

**Acceptance:** precise biased arm versus accurate noisier arm cannot win confirmation merely on precision. Conflicting otherwise-supported protocols yield `PROTOCOL_DISAGREEMENT`. Wide overlapping intervals yield unresolved agreement, not supported agreement. Zero unsupported population never yields zero uncertainty or a finite free energy. Independent implementations supplied the same result fixture produce identical decisions and tie breaks. Parity distributions with identical one/two-variable marginals are separated by the joint diagnostic fixture. Constant observables without justified support do not generate automatic precision passes.

**Do not:** require visits to every bank-derived cell, make per-frame maximum weight a hard certificate, or erase a conflict by dropping an inconvenient observable.

### T08 — Known-distribution calibration and adversarial controls

**Depends on:** T04, T05, T06, T07. **Owns:** additions under `gareus/synth/`, tests `test_cv_selection_controls.py`, generated calibration reports and small fixtures.

**Implement:** controls listed in Section 10; run the full comparison code, including its failure outcomes. Separate calibration seeds from validation seeds. Exact finite-state enumerations or converged quadrature define truth; document reference numerical error. Add dependent joint-state chains and at least one actual OpenMM force-to-observation-to-MBAR control. Existing exact IID samplers are useful for algebra only.

**Acceptance:** deterministic mutation cases all fail as intended. For stochastic release calibration, use 1,000 independent validation replicates per core case, fixed seed list, nominal 95% primary interval family. Report a Wilson interval for empirical coverage and false confirmation; proposed gate: coverage lower 95% bound >=0.90 and false-confirmation upper 95% bound <=0.05. Include a well-mixed positive control with decision-success lower 95% bound >=0.80, so always-abstain does not pass. Require bias/error below each case's specified tolerance, not coverage alone. Computationally expensive real-MD cases may use smaller separately reported replication and cannot replace the cheap repeated validation.

**Outputs:** `certificate.json` with exact algorithm/config/code hashes (the relevant numerical/decision implementation and parameters, plus repository SHA for provenance), validation seed digest, applicable data shape/sampling assumptions, observed metrics, runtime, and pass/fail. The same declared method must be used in prospective analysis. Certificate scope does not claim all molecular dynamics satisfy the tested assumptions.

**Do not:** adjust thresholds on held-out validation seeds, use the oracle to rank production candidates, or call an exact sampler a test of barrier crossing.

### T09 — Frozen layouts, budget compiler and trial executor

**Depends on:** T00, T01, T02, T03, T04. **Owns:** new `trials.py`; minimal existing CLI/config integration; tests `test_cv_selection_trials.py`.

**Implement:** deterministic seven-arm generation, grid24-v1, preparation artifact loading, stage-specific readiness, equal-cost budgets, resource allocation, generated effective configs, subprocess launch through existing `gareus` entry point, ledger, checkpoint/resume and completed-run discovery. A plan command never launches. A run command names exactly one compiled plan/trial set.

**Acceptance:** all seven arms receive nonzero planned pilot budget; state count and GPU/CPU allocation obey bounds; recorded leaf config equals actual parser-resolved settings; changing a model/envelope/window invalidates resume. Crashes produce interrupted status without duplicate rows or repeated charges for already committed work. Budget exhaustion is deterministic and cannot silently add sampling. An engineering smoke may run before T08; scientific launch requires T08's accepted certificate and a complete preparation plan.

**Regression focus:** `test_explicit_window_table_lambda.py`, `test_max_total_windows.py`, `test_cpu_budget.py`, `test_config_profiles.py`, `test_swarm_restart_safety.py`, `test_resume_round_trip.py`.

**Do not:** make adaptive-production obey fixed measurements by monkeypatching its policy; generate an explicit manual measurement configuration instead.

### T10 — CLI, reports and end-to-end release gate

**Depends on:** T05, T07, T08, T09. **Owns:** `cli.py`, `__main__.py`, console script, `.github/workflows/ci.yml`, tracked runbook and example study config; tests `test_cv_selection_cli.py`, `test_cv_selection_e2e.py`.

**Implement:** commands in Section 11, Markdown/JSON reports generated from the same result object, and an end-to-end test from tiny discovery/control inputs to final decision. Add CPU-only correctness CI and separately marked OpenMM/slow jobs. Keep existing console scripts and imports working.

**Acceptance:** clean temp-directory execution emits all required artifacts; rerunning the same analysis is idempotent; a stale result is not reused after an input change; the prospective path never invokes historical native-label analysis, and inputs declared native-derived are rejected. This validates declared provenance; it cannot detect an arbitrarily mislabeled source file. The CLI returns success for a valid inconclusive analysis while recording its scientific status. Missing dependencies, invalid inputs and runtime failure use distinct nonzero exits. Documentation commands correspond to implemented parsers.

**Do not:** claim molecular selection succeeded merely because the smoke pipeline exits zero.

### T11 — Engineering benchmark and fixed-primary molecular study

**Depends on:** T10. **Type:** experiment/runbook task, separate from a code PR. **Owns:** study manifests, measured cost report, screen and confirmation outputs.

**Implement/execute:** prepare the declared chignolin system and new native-blind discovery bank; split discovery, screening-start and confirmation-start provenance; fit six models; freeze panel; measure throughput and storage; validate numerical timestep sensitivity on a bounded target-observable panel; finalize burn-in, retained duration, R and total cap from that evidence. Use generic preparation rather than a fold-specific preset. Seed IDs and integrator/exchange/barostat RNG seeds are recorded independently.

Run every valid component and baseline through the frozen screen. Confirm at most one selected candidate against baseline on fresh independent campaigns. Use the same initial-diversity policy across arms; independent momenta alone are insufficient protection against shared trapped starting structures. Shared frozen discovery coefficients/envelope are allowed, but independence claims apply to measurement randomness and declared starting-state allocation conditional on those artifacts.

**Acceptance:** a complete report with target/force/numerical checks, measured budget, per-arm estimates/statuses, conflicts/unresolved regions, provisional/confirmed decision and deployable model/window files when supported. Insufficient evidence is a legitimate experiment result, with a concrete reason and required next experiment. No specific PC is an acceptance requirement.

**Do not:** use the old 5.351 microsecond campaign as numerical equilibrium truth, extrapolate variance as 1/cost without evidence, or turn a failed candidate into a silently retuned protocol.

### T12 — Select both coordinate slots

**Depends on:** T10 and a completed T11 end-to-end study demonstrating valid data/force/estimation behavior. T11 may be scientifically inconclusive; it need not show that two CVs beat one. **Owns:** extension to models/runtime/trials, tests `test_cv_selection_joint_pairs.py`.

**Implement:** first joint dictionary is explicit and bounded: the declared contact coordinate, heavy-atom Rg, declared end-to-end distance, and six individual plain torsion PCs fitted on discovery data. Nine primitives yield 36 unordered pairs. Residualized pair-conditioned models remain separate declared protocols; do not mix their identities with plain PCs.

Generalize both runtime slots with a small registry of these concrete model kinds. Validate gradients, units, PBC and all-state reconstruction for each. Use a symmetric five-by-five quantile layout, four frozen rungs, 100 replicas; this avoids hidden role preference from the six-by-four MVP layout. Each force-valid pair receives a real pilot if the study claims exhaustive comparison over that dictionary. If the budget cannot cover 36 pilots, return an explicit smaller declared search-space plan or insufficient budget; do not quietly discard low-variance pairs with a structural surrogate. A later prescreen is allowed only after recall against actual sampling utility is validated.

**Acceptance:** primary/secondary role swaps under the symmetric policy produce the same restraint family after remapping; circular manifold embedding survives geometry validation; collinearity is reported rather than universally vetoed; no transform/whitening changes a bias without changing protocol identity. The result scopes itself to tested pairs and protocols.

**Do not:** report the fixed-primary milestone as completing this task or promise a globally optimal two-dimensional reaction coordinate.

### T13 — Optional confirmed efficiency advantage

**Depends on:** T08, T10; enable only if the scientific use needs an advantage claim. **Owns:** comparison extension and a new calibration certificate.

**Implement:** compare a frozen chosen candidate with baseline on fresh confirmation campaigns. Specify a versioned hierarchical resampling method: independently resample campaign IDs within each arm, then synchronized time blocks within selected campaigns, refit MBAR, and recompute the same aggregate loss. Use 1,999 replicates and frozen block policy as an initial algorithm, not a validity assumption. Require a declared practical loss ratio (initial proposal 0.8) and one-sided upper 95% ratio bound <1. Keep advantage separate from equilibrium adequacy.

**Acceptance:** held-out repeated experiments calibrate the complete method, including selection/confirmation separation, null false superiority and power at the claimed effect. Require false superiority upper 95% bound <=0.05 and power lower 95% bound >=0.80 at the declared budget/effect before enabling the claim. Test with heterogeneity, long correlation and near-zero losses; undefined ratios remain unresolved. If calibration fails, leave `advantage: NOT_TESTED`/`UNRESOLVED` and report needed cost; do not loosen the claim to obtain a winner.

**Do not:** assume four campaigns establish a 20% variance improvement or use the review's simple F-test benchmark as the power calculation for this full algorithm.

## 10. Validation matrix

The controls below target concrete scientific failure modes, not implementation-shaped tests.

| Control | Ground truth / construction | Required observation |
|---|---|---|
| IID multistate harmonic/finite-state target | Analytic or exact enumeration | Correct estimates, units, normalization and uncertainty in the simplest supported case |
| Joint replica-exchange finite chain | Exact stationary distribution and controllable persistence | Whole-ensemble covariance matches empirical repeated-campaign error |
| Hidden barrier/biased starting basin | Known important missing mass | Narrow within-basin uncertainty cannot confirm equilibrium accuracy |
| Two precise conflicting protocols | Artificial known population offset | Cross-protocol conflict blocks confirmation |
| Low-variance useful coordinate | Known sampling gain despite poor structural variance rank | Candidate is piloted; no structural veto |
| Circle embedding | Uniform angle with sin/cos coordinates | Valid bias and representation despite rank-one tangent space |
| Rare IID target region | Known tiny mass and zero visits | Valid IID upper bound can resolve small absolute mass |
| Rare weighted correlated region | Same zero observation without valid bound | `UNRESOLVED_SUPPORT`; no ESS-binomial shortcut |
| Parity joint distribution | Equal lower-order marginals, disjoint joint support | Broad multivariate diagnostic detects the discrepancy |
| Recording-cadence change | Same underlying correlated path sampled more densely | No fabricated information gain or new support certificate |
| Heavy-weight episode | Most target contribution from one correlated episode | Influence/leave-block-out sensitivity exposes uncertainty |
| Residual contact dependence | Nonzero quadratic subtraction | Omitting chain-rule term fails force verification |
| Runtime/export energy identity | Actual OpenMM configurations and frozen state table | Forces, reported CVs, raw energies and reconstructed bias agree |
| Pre/post-exchange label mutation | Real recorder and accepted/rejected swap | Wrong labels fail behavioral test |
| Crash, resume and compaction | Faults injected at manifest/checkpoint boundaries | No duplicated measurements, stale state identity or invented continuity |
| Integrator timestep sensitivity | Same target and protocol at dt and dt/2 | Numerical error assessed in target observables; no kinetics gate |

Before a slow test is adopted, measure its runtime and make its role explicit. Fast CI runs deterministic controls and small statistical smoke checks. The 1,000-replicate calibration suite is a reproducible release/experiment gate, not a per-edit requirement.

Existing regression anchors to retain include the exchange-kernel tests, NPT acceptance/ownership/transaction tests, all three thermodynamic-validity oracle files, ladder-MBAR tests, and persistence/resume tests. Existing real-MD oracles do not by themselves exercise the complete production sampling closure; T02/T10 must close that integration gap behaviorally.

## 11. Proposed command workflow

These commands are to be implemented. They do not work at the pinned baseline.

~~~bash
python -m gareus.cv_selection validate --protocol study/protocol.json --stage build
python -m gareus.cv_selection prepare-models --protocol study/protocol.json
python -m gareus.cv_selection prepare-panel --protocol study/protocol.json
python -m gareus.cv_selection calibrate --suite controls-v1 --out study/calibration
python -m gareus.cv_selection plan --protocol study/protocol.json --stage engineering
python -m gareus.cv_selection run --plan study/plans/engineering.json
python -m gareus.cv_selection plan --protocol study/protocol.json --stage screen
python -m gareus.cv_selection run --plan study/plans/screen.json
python -m gareus.cv_selection analyze --plan study/plans/screen.json
python -m gareus.cv_selection freeze-confirmation --study study
python -m gareus.cv_selection run --plan study/plans/confirm.json
python -m gareus.cv_selection analyze --plan study/plans/confirm.json
~~~

Console alias: `gareus-select-cvs`. It calls the same module entry point. Add `--resume` to `run`; add `--dry-run` to display compiled commands and resource bounds. `freeze-confirmation` succeeds only with a valid eligible provisional choice and a fully resolved confirmation budget; it cannot manufacture the required fresh inputs.

Exit codes: 0 = command completed, including a valid inconclusive scientific decision; 2 = invalid input/protocol; 3 = unavailable dependency/environment; 4 = execution failure/interruption; 5 = scientific launch refused because readiness is incomplete. Machine callers inspect `decision.json` for scientific success.

A generated example protocol has unresolved fields as null and starts `screen-ready=false`; it must not look launch-ready. The compiler lists the exact artifact-producing task for every missing item. Distinguish this from fully resolved tiny synthetic fixtures, which must run end to end in CI.

## 12. Scheduling and handoff rules for agents

### Dependency order

| Wave | Work | Scheduling rule |
|---|---|---|
| A | T00 | Establish contracts first |
| B | T01 and T03 | Independent files; can run concurrently if delegation is explicitly authorized |
| C | T02, then production integration part of T04 | Both touch production/store; serialize those edits through one owner. Pure model/force code can be developed separately |
| D | T05 and independent parts of T09 | No inference work before exact data/physics contracts exist |
| E | T06, then T07 | Freeze statistical APIs and reason codes before CLI/reporting |
| F | T08; finish T09 | Controls decide scientific readiness; runner may already support engineering smoke |
| G | T10 | One integrated vertical slice and CI gate |
| H | T11 | Measured experiment, not assumed success |
| I | T12 | Joint-pair capability and new study; T13 only if an advantage claim is needed |

Do not use separate agents to edit `production.py`, `store.py`, `cli.py`, or shared schemas concurrently without an explicit integration owner. A plan for agents does not itself require multi-agent execution.

### Copyable task prompt

~~~text
Implement task Txx from docs/atlas-md/developer/equilibrium-cv-implementation-plan.md.
Base your branch on the designated integration commit; first report any baseline drift.
Read the task's listed source files and contracts. Preserve existing public behavior.
Own only the task's listed modules/tests plus the smallest necessary integration edits.
Implement the stated acceptance cases using the real code path where required.
Do not change physics, statistical thresholds, candidate scope, or eligibility semantics
just to make a test pass. If a required assumption fails, produce the specified failure
result and evidence. Keep unrelated fixes separate.
Run targeted regressions and the task's new tests; report exact commands and outcomes.
Deliver a reviewable diff, schema/API changes, tests, remaining limitations, and handoff
notes. Do not claim a scientific result from a software smoke test.
~~~

The plan file should be copied into the allowed tracked docs path in T00. Do not create a repository-wide AGENTS.md merely to duplicate this task plan.

### Per-task completion record

Each PR/handoff records: task ID; actual base SHA; changed paths; interfaces changed; accepted input/output example; acceptance cases exercised; regression commands and counts; tests skipped with reason; unresolved issues; and next task readiness. An implementing agent may propose a contract revision in a separate documented change; it may not silently relax a gate.

Task-sized branches should follow `cv-select/t00-contracts`, `cv-select/t01-state-wiring`, etc. Verify `git status` and existing user changes before editing. At a newer main, inspect changes in the integration map and rerun affected baseline tests; do not reset work back to this SHA or assume the old map still fits.

### Useful existing verification commands

Run from a real checkout, not this conversation's partial source snapshot. Choose the extras required by the task.

~~~bash
python -m pip install -e '.[dev,analysis,config]'
gareus-test-run --check-deps --skip-if-missing
python -m pytest -q tests/test_exchange_kernel_exact.py tests/test_gibbs_walk.py tests/test_exchange_fixes.py
python -m pytest -q tests/test_store.py tests/test_parquet_manifest_transactions.py tests/test_sample_before_exchange_ordering.py
python -m pytest -q tests/test_bootstrap_torsion_cv.py tests/test_secondary_cv_restraint_sign_convention.py tests/test_cv2_reprojection.py
python -m pytest -q tests/test_lambda_ladder_mbar.py tests/test_estimator_excludes_cumulant_under_ladder.py tests/test_query_reconstruct_bias_matrix_nan_guard.py
git diff --check
~~~

OpenMM tasks also need `.[openmm]` and the repository's actual GaMD runtime dependency/environment. The dependency-check command is diagnostic only; a skipped dependency check cannot satisfy an OpenMM/GaMD release gate. T00 records the installation route and versions; the current project extras do not imply every external GaMD integration is installed. Do not hide unavailable scientific dependencies behind skipped release gates. Run the new task tests and affected existing tests; reserve the broader suite for integration or a concrete regression risk.

## 13. Definition of done and review traceability

Software is complete for M1 when a clean, reproducible tiny study goes through actual recording, strict export, estimation, uncertainty, decision, report and resume behavior; all required controls pass; and a molecular plan can be compiled with honest readiness errors. M2 is complete when the fixed-primary experiment is reported, even if inconclusive. M3 requires actual two-slot support and a separately scoped joint study.

| Adversarial finding | Resolution in this plan | Primary owner |
|---|---|---|
| R1: precision mistaken for accuracy | Separate statuses; binding cross-protocol and joint-distribution conflict; scoped confirmation | T07 |
| R2: small replication cannot support promised improvement | No default superiority claim; measured confirmation design; optional calibrated advantage | T08, T11, T13 |
| R3: costly/underspecified uncertainty | Explicit joint estimating equations and covariance; selected block-refit checks | T06 |
| R4: structural shortlist misses useful coordinates | Pilot all six MVP components; no unvalidated joint prescreen | T09, T12 |
| R5: rank hard veto rejects valid embeddings | Geometry diagnostic only; circle acceptance case | T04, T08, T12 |
| R6: mandatory visits reject negligible mass | Three population statuses; valid bounds only; unresolved free energy allowed | T07 |
| R7: per-frame caps reward dense recording | Dependence-aware influence and episode/block sensitivity | T06, T08 |
| R8: whitening/order changes restraint family | Simple scaling; explicit layouts; symmetric joint policy; transforms change identity | T03, T09, T12 |
| R9: unspecified consequential choices | Concrete v1 algorithms plus staged machine-checked readiness and calibration artifacts | T00–T10 |
| R10: broad framework before first valid experiment | Incremental vertical slice; reuse existing correctness/production machinery | All tasks |

The next actionable step is **T00**, followed by **T01 and T03**. The first high-value executable milestone is **T01–T05: a frozen residual CV that is forced correctly, recorded with exact identity, and reweighted through the existing strict path**. That work remains useful even if the subsequent comparison cannot identify a better sampler.

## 14. Method references and evidence limits

- [Shirts and Chodera, MBAR](https://arxiv.org/abs/0801.1426): equilibrium estimator basis. Correct algebra does not repair missing relevant support.
- [Li et al., MBAR asymptotic error including jointly sampled states](https://arxiv.org/html/2203.01227v1): methodological basis for the dependence-aware implementation; the estimating-equation normalization and code still require independent validation.
- [PyMBAR MBAR API](https://pymbar.readthedocs.io/en/stable/mbar.html): weight-based effective sample number is a concentration diagnostic, not an automatic temporal-information count.
- [Pinned repository](https://github.com/sulcjo/ATLaS-MD/tree/cdee6cc87ac540b0bac9b230f389c77d70383faf): implementation evidence. Only targeted source inspection was performed for this plan; no full runtime audit or repository test run was performed here.

Numerical thresholds introduced here are explicit proposed policies with calibration gates. Force correctness, data integrity and exact schema semantics are hard implementation requirements. Molecular convergence, useful correlation resolution and statistical decision power must be established by the planned experiments, and may remain unresolved at the available budget.
