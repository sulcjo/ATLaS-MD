# Side-chain-aware z3: implementation and verification plan

Status: ready for implementation after task-level regression checks; no implementation or simulation validation claimed.
Date: 2026-10-10. Target: PR #141, branch `feat/cvaux-stage-a`.
Baseline inspected: `fd5be0fdc284384a4bd5b1f6600a50994bf466eb` (unchanged at plan preparation).
Specification: [side-chain z3 extension](../specs/2026-10-10-cvaux-sidechain-z3-extension-spec.md).
Scope precedence: [B/C/D repair plan](2026-10-10-cvaux-repair-bcd-plan.md) continues to govern the earlier repair package.

## 1. Binding amendments from adversarial verification

This plan refines the extension specification in five places. Implement these decisions rather than reproducing the current global backbone search under a new feature name.

| Finding | Required resolution | Tasks |
|---|---|---|
| Opposite local chi/packing relationships cancel in the global discriminant | Fit and score new-family candidates within training-defined conditioning neighbourhoods; deploy a fixed, globally evaluable torsional function | T05, T06 |
| Candidate source partition may differ from placement partition | Persist source partition identity, group IDs and neighbourhood; use these same labels for candidate-specific placement | T05, T07 |
| Runtime/storage assume one backbone quadruplet per feature | Compile v2 features to a canonical unique-torsion basis and weighted harmonic terms shared by all consumers | T02–T04 |
| Solute trajectory indices are assumed to equal production indices | Explicit per-phase atom mapping in discovery, XTC backfill and final-PDB evaluation | T01, T04 |
| No frames inside the reconstruction check's two-width region produces zero error | Return unavailable coverage, report full joined-sample energy discrepancies and refuse to treat an empty check as passed | T08 |

Counterexample A: two conditioning regions a in {-1,+1}, chi predictor x in {-1,+1}, packing label y = 1[a*x > 0]. Current global L1 fit has zero coefficients and current additive `info_gain` gives zero; each regional classifier gives perfect held-out classification. This is an algorithmic regression fixture, not molecular evidence.

Counterexample B: one worker with centre 0 and width sigma=sqrt(RT/k), recorded z=3*sigma and backfilled z=4*sigma. Current `worker_energy_error_kt` reports 0 with no qualifying frames although the actual energy discrepancy is 3.5 kT.

## 2. Non-negotiable scope and workflow

- One admitted z3 model per campaign, the current worker reserve and lambda=0 workers. No additional umbrella grid, REST2, worker retirement or simultaneous model collection.
- `--ap-aux-feature-space backbone|sidechain|mixed|auto`, default backbone. Backbone-only search retains its prior code path and random draws. Aux-off paths remain unchanged. The reconstruction empty-coverage repair is an intentional correction on affected aux paths, not an opt-out compatibility promise.
- Preserve `--ap-aux-validation required|off`, the unvalidated stiffness cap, per-carrier burn-in, log-space exchange, reconstruction refusals and heuristic crosscheck wording. Mathematical/schema checks remain active in both validation modes.
- New-family candidates use local fitting. Legacy backbone candidates stay global for compatibility; auto compares both and records which search regime generated each candidate.
- No native reference, folded label or post-hoc residue selection enters discovery. A native readout is permitted only in a separately declared benchmark.
- Add failing targeted regressions before changing the relevant implementation. Do not run the whole suite by default, turn failures into skips or weaken numerical tolerances to make tests pass.
- Each task is a reviewable commit on the same PR branch. Check its current head before writing; preserve concurrent changes. Do not overwrite the earlier repair work.
- Do not assume GPU or OpenMM availability. Report unavailable execution honestly; CPU mocks do not certify physical force or NPT behaviour.

## 3. Task sequence

### T00 — Pin baseline and define opt-in policy

Files: `gareus/cli.py`, `gareus/adaptive/aux_discovery/settings.py`, `gareus/adaptive_production.py`, discovery replay CLI and help text.

1. Record baseline commit and focused-test results. Inspect repository instructions before implementation.
2. Add the four-value feature-space option. Reject use without aux discovery; default/missing historical fields resolve to backbone.
3. Freeze feature space in campaign options alongside the existing validation mode. Refuse a different mode on resume, including before the first admission, unless an existing supported settings-reset workflow explicitly starts a new campaign.
4. Add extension-only settings for dictionary/schema versions, local-support minima, conditioning neighbourhood policy and feature-variance floor. No new RNG calls on the backbone-only path.
5. Default local-support minima to the existing placement minima: 100 training and 50 holdout rows. Use the existing co-occurrence share rule for each class, not an invented folding-specific threshold. These are heuristic support filters, not independent-sample counts.

Acceptance: new option parses; absent option preserves pinned old behaviour; incompatible resume fails before MD; sidechain with no chi support returns `no_sidechain_features`; auto without chi delegates exactly to backbone search.

### T01 — Typed chi dictionary and explicit atom mapping

New modules proposed: `gareus/auxiliary_cv/sidechain_dictionary.py`, `gareus/auxiliary_cv/atom_mapping.py`.
Touch: `adaptive/aux_discovery/descriptors.py`, `frames.py`.

Define immutable records for torsion family, residue/template variant, chi index, chain/residue identity, ordered quadruplets, symmetry orbit, harmonic and sign. Initial coverage is chi1/chi2 of explicitly enumerated standard residue templates and supported protonation aliases. Publish the supported/excluded table in code and tests. Ala/Gly have none; proline/ring-constrained chi and unsupported modifications are excluded with reasons. Missing atoms in a supported template are reported, never inferred from position in an array.

Use explicit writer atom-index metadata if available. Otherwise resolve an unambiguous mapping using full atom/residue/chain identity and connectivity. Duplicate or ambiguous matches refuse the mapping; never fall back to positional indices. Persist the mapping for each phase because a campaign-level assumption is insufficient. Distinguish full-system indices in the frozen model from trajectory-local indices used to evaluate a frame.

For symmetry, enumerate a small validated template orbit rather than a general graph-automorphism engine. Validate its compatibility with the parameterised System when preparing admission; topology alone cannot establish equality of force-field parameters. Unsupported or asymmetric cases omit the affected symmetry feature, with a reason. Obtain the relevant System through existing setup artifacts; do not have frame workers create CUDA contexts. Cache this check per system/dictionary version.

Acceptance tests: representative branched, aromatic, charged and protonated residues; duplicate residue numbering across chains; non-prefix solute; reordered trajectory atoms; unsupported residue; equivalent atom relabelling. All supported templates need positive fixtures, not only chignolin residues.

### T02 — v2 model schema and canonical compiled representation

Files: `auxiliary_cv/model.py`, `features.py`; new `auxiliary_cv/compiled_features.py` if appropriate. Leave shared CV-selection v1 semantics unchanged.

Introduce an aux-specific feature-schema v2, under `atlas-aux-cv-model-v2`. Every feature carries ordered unique orbit quadruplets, sin/cos, harmonic in {1,2}, sign in {-1,+1}, family and residue identity. A one-member orbit is an ordinary torsion. All numeric/index fields have strict type/range validation; booleans are not indices or coefficients. Reject duplicate orbit members and malformed/empty orbits.

Provide one pure `compile_features(model)` operation returning:

- deterministically ordered distinct full-system quadruplets;
- a sparse feature-to-torsion term list `(feature_index, torsion_index, trig, signed_harmonic, orbit_weight)`;
- active torsion indices derived from exactly nonzero feature coefficients;
- canonical feature/basis identity metadata.

For each primitive, orbit weights equal 1/orbit_size. Canonicalise orbit order, combine exact duplicate terms deterministically, and retain a documented feature order. Do not threshold a small nonzero fitted coefficient to zero silently. Any deliberate sparsification precedes scoring and freezing.

Keep v1 serialization and hashes unchanged. The v1 compilation adapter must reproduce old ordering, values and force expressions. Implement version dispatch through the aux model facade; audit all callers of `feature.atom_indices`, `unique_torsions`, `_feature_atoms` and `feature_schema.width`.

Acceptance: v1 roundtrip/hash pins; v2 deterministic identity; reordered equivalent orbit inputs canonicalise identically; invalid features refuse; zero-coefficient degenerate torsions remain excluded from active geometry requirements.

### T03 — Forces, gradients and real force-side observation

Files: `auxiliary_cv/{features,evaluate,force,runtime}.py`, including `aux_sub_cv_spec`, `_check_aux_sub_cvs`, topology binding and observation helpers.

Evaluate each feature as the average of trig(m*sign*theta) over its orbit. Use the compiled basis in position and stored-angle evaluators. The derivative includes both m*sign and orbit weight. Sum the full projection before applying its harmonic umbrella; never sum independent per-feature umbrellas.

Compile OpenMM subforces using the same term representation, grouping by trig and signed harmonic without collisions. Force inspection must compare harmonic expressions, quadruplets and weights. Runtime z continues to come from the actual replica's aux force; offline positions provide the independent comparison.

Preserve the imaging convention. Detect inconsistent molecular imaging rather than silently using minimum image in one evaluator. Map topology/active-torsion checks over every orbit member. No discontinuous representative sorting or angle averaging.

Acceptance: finite-difference forces for ordinary chi, symmetric harmonic-2 and mixed backbone/chi models; symmetry-invariant energies and permutation-covariant forces; wrap continuity; inactive states; fast/slow CV1 paths. Extend existing aux force/observation/parity suites with new-family fixtures.

### T04 — Frame features, sample basis and model emission

Files: `adaptive/aux_discovery/{descriptors,frames,z3_search}.py`, `auxiliary_cv/{sample_schema,runtime_definition,runtime_io}.py`, affected store readers/writers.

Keep backbone, sidechain and mixed feature views separate and labelled. Raw angles are distinct from sin/cos/orbit feature columns. Train-only variance filtering and normalisation must be persisted; do not divide nearly cancelled symmetry features by tiny scales.

For v2 phases, build the stored unique-angle basis as the existing backbone basis plus missing quadruplets required by the admitted model, in canonical order. Store ordinary OpenMM angles, not harmonics disguised as angles. Add an explicit v2 sample payload discriminator and teach readers both versions; v1 payloads remain unchanged. Mapping from stored angles to each model's compiled basis is mandatory.

Replace backbone-specific emission with feature-view-aware emission. Fold feature means/stds into coefficients and offset; freeze the projection scale. Verify `fit_z == exported_model_z` on held-out coordinates before placement/admission. Missing angle columns or model/basis mismatch refuse reconstruction.

Acceptance: selected chi and every symmetry orbit member survive write/read; no fixed 36-column assumption; mixed projection matches fitting; full-system and reordered-solute coordinates agree; old payloads remain readable. Audit `tor_###` consumers by symbol search rather than trusting this file list to be exhaustive.

### T05 — Local candidates with stable partition identities

Files: `adaptive/aux_discovery/{partitions,z3_search,pipeline}.py`.

Introduce candidate provenance containing source partition hash/k, original group IDs, feature family, training-defined neighbourhood ID/bounds, feature view, normalisation and selected coefficients. Group integers have meaning only within their partition identity.

Initial neighbourhoods are the already frozen conditioning bins; do not add a second adaptive spatial partition. Coarse bins may miss finer structure, which is a reported limitation. Enumerate neighbourhoods and co-occurring group pairs from training data; use holdout for score/confirmation, not to create a new region. Apply existing absent/constant-conditioning handling.

Fit sidechain and mixed pairwise discriminants only on training samples of their group pair inside the region. Preserve the current block-aware fitting structure; carrier IDs do not imply independent trajectories. Compute a pairwise held-out log-likelihood improvement over a local class-prior baseline fitted on training rows. Class priors use training counts with explicit Laplace pseudocount 1. Require both classes and the support/share rules in train and holdout; insufficient support produces a named rejection.

For cross-region ranking, define `score = local_gain * training_pair_region_fraction`, with fraction measured against the common eligible training frame population. This intentionally balances local predictability against observed prevalence; it is not an equilibrium population estimate. Recompute all label-dependent fractions in each null search. Record local gain and prevalence separately.

In auto mode, retain legacy backbone candidate generation but rescore its candidates on the same eligible region/pair tasks for comparison. Do not compare a local binary score numerically with the existing global multiclass information gain. Backbone-only mode retains its old scoring. This avoids changing the established baseline while making auto comparisons meaningful.

The exported z is a fixed function everywhere; region membership never enters forces. Test counterexample A and verify the same frozen z can be evaluated outside its fit region. Also test no-hidden-structure and insufficient-local-support cases.

### T06 — Complete-search null and deterministic selection

Files: `z3_search.py`, `pipeline.py`, attempt reporting.

One null replicate uses one scrambled label array per partition, shared across all feature families. Repeat every label-dependent step of T05: regional support, eligible pairs, local fits, regularisation, score and winner selection. Fixed training CV-bin boundaries remain fixed. Do not reuse real-data eligible pair lists when claiming to repeat the complete enlarged search.

Compare the real maximum score with the null maxima using the existing number of null searches. Retain trapped-lineage refusal and attempt history. Repeat deduplication identically in observed/null paths. Fix seeds and canonical tie breakers; ties use fewer nonzero coefficients then canonical candidate identity. Do not choose by rounded report values.

Circular shifts can disrupt conditioning relationships as well as predictor-label relationships. Keep the result explicitly heuristic, not a conditional-randomisation p-value. Include null controls with labels determined by CVs, heterogeneous umbrella-state occupancy and no additional chi information. Unexpected admission in these controls blocks release pending diagnosis; do not tune a threshold against the folding benchmark.

Acceptance: spy/counter tests demonstrate all families/regions enter each replicate; same permutation shared across families; constant-label carriers produce unavailable null evidence; repeated runs deterministic. Preserve backbone-only RNG behaviour and original tests.

### T07 — Candidate-consistent placement and admission

Files: `pipeline.py`, `placement.py`, `adaptive/aux_admission_io.py`, frozen partition/admission records and worker diagnostics.

Use the winner's source partition labels for placement and its frozen evaluation partition for subsequent candidate-specific readouts. Persist partition identity, group pair and region alongside model identity in admission. Do not substitute the largest passing k. If retaining a separate general reporting partition, namespace it explicitly.

Parent eligibility requires training support for both winning populations within the winning region and the existing active/ordinary/lambda=0 conditions. Compute parent umbrella forecasts from the parent's full eligible training distribution, not a region-truncated distribution pretending to represent the complete parent ensemble. Region checks select parents; they do not truncate Hamiltonians or exchanges.

Reuse the stable overlap forecast, observed quantile centres, spring/worker caps and holdout placement confirmation. If the winning model has no admissible placements, return `no_worker`; do not search runner-up models post hoc unless that extra selection stage is first incorporated into the documented null procedure.

Acceptance: winner from k=2 with a separate passing k=4 uses k=2 labels throughout; correct local parent included, unrelated parent excluded; no placement returns no admission; worker states carry complete biases and unchanged reserve semantics.

### T08 — Backfill mapping and evidence-bearing reconstruction checks

Files: `adaptive/aux_backfill.py`, admission/pooling callers, `mbar_analysis/loaders_union_parquet.py` as necessary.

Replace prefix assumptions in `check_solute_indices`, `frame_z` and `_final_pdb_z` for v2 models with T01's explicit per-phase map. Evaluate only mapped model atoms; do not create uninitialised pseudo-full-system coordinates. Keep step alignment, checksums, sample identities and missing-frame refusals.

Change reconstruction diagnostics to report, per worker: joined count, finite count, in-region count, maximum error inside the existing region, and maximum error over all joined frames. Zero in-region count gives `insufficient_worker_coverage` and no numeric zero error. Empty/invalid evidence cannot yield `ok=true`.

For v2 model checks, require nonempty in-region evidence and apply the existing energy tolerance to all joined finite samples as a conservative empirical rule. This is stricter than the previous region-only heuristic but does not claim an error bound on historical frames that were never compared. Retain the configured tolerance and report it; do not call a 1 kT threshold high-precision thermodynamic certification. If quantisation fails the check, refuse pooling and report the limitation; do not quietly loosen tolerance or drop bad samples.

The empty-coverage correction also applies to v1 checks. Preserve other v1 behaviour unless separately reviewed. Missing/failed post-admission checks block pooling in both validation modes. Historical XTC reconstruction remains subject to its actual precision; float64 evaluation cannot restore lost coordinate information.

Acceptance: counterexample B cannot pass; reordered-solute backfill matches direct evaluation; no-frame/missing-orbit/NaN cases refuse; matching controls pass; both union paths agree on observation keys and reconstructed energies. Add an adversarial test with errors only outside the two-width region for v2.

### T09 — Validation coverage, equilibrium integration and resume

Files: `adaptive/aux_discovery/validation.py`, admission, `auxiliary_cv/checkpoint.py`, parity scripts and targeted integration tests.

Extend evidence coverage with explicit model-schema/feature-family/harmonic support and a description of the tested normalisation/spring regime. Required mode must not accept backbone-only evidence as chi/harmonic-2 evidence. Off mode retains current admission policy and cap while reporting absent evidence. Neither mode bypasses model/force consistency or reconstruction checks.

Test actual cross-state energies for ordinary and chi workers and inclusion of the aux force exactly once, outside Pep-GaMD channels. Exercise the existing detailed-balance regression with the new energies. Run a controlled NPT distribution test using the actual barostat target and new bias; common-pV exchange cancellation does not replace this test.

Exercise pull/seeding, carrier-based burn-in transported through exchanges, sample writing, union analysis and checkpoint/resume. A v2 checkpoint resumes the same model and compiled basis; mismatched harmonic/orbit/mapping or changed feature-space options refuse. Keep existing one-model and population-freeze policies.

Acceptance: CPU end-to-end discovery or deterministic candidate injection reaches force -> samples -> MBAR; a separate test covers real synthetic discovery. GPU parity and finite-timestep validation are recorded when run, not inferred from CPU results. Same k across models is not equivalent stiffness.

### T10 — Benchmark, documentation and release review

Compare backbone-only against auto using equal worker budgets and total GPU-hours, including discovery/backfill overhead. Use independent repeated campaigns and preregister benchmark observables. Compare equilibrium folded-population/free-energy uncertainty and between-run agreement; also record overlap/ESS, structural coverage, worker utilisation and failed/no-admission outcomes.

Keep generation native-blind. Do not optimise search settings against the evaluation-only folded criterion. Distinguish correctness, empirical sampling improvement and inconclusive evidence. A first folding event or a successful classifier is not sufficient.

Update help text, supported-residue coverage, model/sample schemas and the spec with T01–T09 decisions. Record actual targeted test commands/results and unavailable platforms. The change is implementation-complete after the real end-to-end path and compatibility checks pass; a performance claim additionally needs benchmark evidence.

## 4. Dependency order and stopping rules

Execute T00 -> T01 -> T02 -> T03 -> T04 -> T05 -> T06 -> T07 -> T08 -> T09 -> T10. The small empty-coverage regression/fix in T08 can be committed earlier; mapping integration still depends on T01/T04.

Do not proceed to production admission if force/energy parity, sample mapping or partition identity is unresolved. No chi features, no informative null, inadequate local support, or no useful placement are normal no-admission outcomes. An unsupported chemistry case should not disable valid backbone sampling.

The plan deliberately does not promise a universal side-chain reaction coordinate. A linear periodic projection may still miss cooperative nonlinear barriers, and one admitted model may address only one local obstruction. Those are scientific limitations to measure, not reasons to relax equilibrium bookkeeping.
