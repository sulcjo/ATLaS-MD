# CVaux repair implementation plan

Status: ready for implementation; this plan does not implement or certify the repairs.

Baseline: `f6f44b03e9f0720d6cee92643cc91e67c68d0c7f`. PR #141 and `feat/cvaux-stage-a` were rechecked at this revision on 2026-10-10. Reconcile any later diff before starting a task; do not overwrite intervening fixes.

Binding design: [CVaux correctness repair specification](../specs/2026-10-10-cvaux-correctness-repair-spec.md). F01–F13 below refer to that document. Proposed new module, schema and test names in this plan are implementation targets, not claims that they already exist.

## 1. Deliverable and execution rules

Deliver a sequence of reviewable commits, each containing its implementation, focused tests and a short behavior/migration note. Preserve the existing scientific Hamiltonian and common-T,p exchange cancellation. Do not expand this work into kinetics, MTS, force-field support, native-structure supervision or a replacement discovery algorithm.

Use the order below as the default. Dependencies are explicit; tasks without a dependency may be developed independently, but integration edits to `production.py`, `adaptive_production.py`, `store.py`, the union loaders and report entry points must be serialized. The plan does not require agent delegation.

- First make each reported failure reproducible; then change the production function the regression exercises.
- Do not rerun all real-MD tests after every small edit. Run the task's tests and directly affected contracts; use the broad suite at integration checkpoints.
- Never turn a required test into a skip, a failed scientific check into a warning-only PASS, or an invalid energy into zero to make a fixture pass.
- Keep existing numerical behavior where valid. Record intentional differences in proposal algorithms, schemas, sample eligibility and report semantics.
- No GPU availability means GPU evidence remains unavailable. A CPU smoke test is not equilibrium distribution validation.

## 2. Dependency order

| Task | Scope | Depends on | Completion artifact |
|---|---|---|---|
| T00 | Baseline, regression fixtures, valid Pep-GaMD smoke setup | — | Reproducible CPU environment and failing-case tests |
| T01 | Shared contracts, schema versions and policy decisions | T00 | Pure validators and migration fixtures |
| T02 | Checked auxiliary arithmetic and stable placement | T00 | F04/F08 regressions pass |
| T03 | One-dimensional/degenerate discovery conditioning | T00 | F05 discovery path passes |
| T04 | Independent force-side parity | T01, T02 | CPU force/z/energy parity, GPU harness ready |
| T05 | Log-space Gibbs and versioned event records | T01, T02 | Exact exchange tests and ledger round trips |
| T06 | Pending proposals and applicable validation certificates | T01, T02, T03, T04 | No unvalidated proposal can become active |
| T07 | Population burn-in transaction and resume | T01, T06 | Complete interval evidence; no transient leakage |
| T08 | Strict reconstruction, post-admission parity and pool selection | T01, T04, T07 | Both union paths select the same valid observations |
| T09 | Observation identity and lambda precision migration | T01 | Strict exporter and mixed-schema reads pass |
| T10 | Aligned volume storage and NPT thermodynamic reporting | T01, T09 | Correct pV observables and historical fallback |
| T11 | Phase-local worker diagnostics and lineage reporting | T07, T08 | No synthetic cross-phase transitions |
| T12 | Bounded-memory discovery and crosscheck | T03, T08 | Chunked/dense parity and resource contract |
| T13 | Statistical crosscheck and discovery limitation statuses | T08, T11, T12 | Honest pass/fail/inconclusive semantics |
| T14 | Compatibility audit, CI and integrated CPU tests | T02–T13 | Machine-readable audit and green required CPU jobs |
| T15 | GPU, finite-timestep, NPT and cost evidence | T14 | Applicable immutable validation records |

The priority path is T00 → T01/T02 → T04/T06 → T07 → T08. Earlier commits may merge as repairs, but certification stays blocked until every applicable required gate is satisfied.

## 3. T00 — Establish a reproducible baseline

**Existing files:** `tests/conftest.py`, `tests/test_aux_parity_report.py`, `tests/test_aux_admission_e2e.py`, `pyproject.toml`.

1. Record the source SHA, Python, NumPy, SciPy, scikit-learn, OpenMM, PyMBAR, Arrow/DuckDB, MDTraj and exact GaMD source revision in a baseline result manifest. Reuse the reviewed environment when possible. The prior measured suite was 916 passed, 2 failed, 2 skipped, using OpenMM 8.5.1; this is provenance, not a result for future changes.
2. Fix the Pep-GaMD test configuration by explicitly setting preparation and cumulative stage lengths consistent with the averaging interval. Inspect the actual resolved integrator arguments; do not merely increase runtime until the error disappears. Keep the upstream stage validation enabled.
3. Replace the exact-zero parity assertion with the existing double-precision contract and retain a deliberately significant discrepancy that fails.
4. Add focused regressions for the six new review counterexamples before fixing them. Use existing auxiliary fixtures where possible; include the production entry point in each regression.
5. Add a validation constraints file, proposed `tests/constraints/cvaux-validation.txt`, recording the tested dependency set. Do not arbitrarily pin all package users to the validation environment. Ensure test extras explicitly cover scikit-learn/MDTraj and the tested GaMD dependency; the current `dev` extra alone is insufficient for the full suite.

**Acceptance:** the two baseline fixture defects are understood and corrected, and each new regression fails for its intended scientific/contract reason on the unrepaired implementation. Conventional and Pep-GaMD smoke fixtures must assert that the intended integrator, ensemble and force groups were actually used. Record expected failing regressions during development; do not leave an unexplained red suite in a completed integration commit.

## 4. T01 — Create one policy and contract layer

**New modules:** `gareus/adaptive/aux_contracts.py`, `gareus/adaptive/aux_eligibility.py`. **Existing:** `aux_admission_io.py`, `aux_pooling.py`, `kernel_identity.py`, `correctness/_io.py`, `correctness/state_identity.py`.

Implement pure, IO-independent validation and decisions first. Suggested API surface:

```python
validate_validation_record(record, required_identity) -> ValidationDecision
validate_admission_record(record, registry, required_identity) -> AdmissionRecord
decide_phase_eligibility(phase, admission, equilibration, reconstruction, policy) -> PhaseDecision
build_pool_selection(phases, observation_index, decisions) -> PoolSelection
```

`PhaseDecision` carries an explicit eligibility status, reason codes, evidence references and boundary. `PoolSelection` carries selected observation identities plus excluded phases/counts; do not reduce it to an unauditable boolean. Put typed finite-number/integer/hash validation in this layer or reuse existing strict helpers. Reject booleans as numeric identities, masked identifiers, fractional IDs and nonfinite parameters before coercion.

Define and test schema constants for validation v2, admission v2, population equilibration v1 and reconstruction certificates v1. Distinguish semantic content digests from byte hashes of source artifacts. Use existing atomic JSON/fsync utilities rather than adding a new inconsistent writer. Do not inject these new fields into legacy physical-state hashes unless that schema is explicitly versioned.

Define error behavior once:

- Before admission, unavailable candidate evidence produces `pending_validation`; ordinary sampling may continue.
- Invalid active Hamiltonian/force/checkpoint identity blocks affected MD before another step.
- Failed historical reconstruction blocks the affected equilibrium analysis; otherwise-valid ongoing MD need not stop.
- Unfinished burn-in blocks eligibility, not storage of engineering observations.
- Post-admission integrity errors must not be swallowed by the discovery hook's broad exception handler.

**Tests:** proposed `test_aux_repair_contracts.py`; v1/v2 fixtures, unknown schemas, malformed mappings, NaN/Infinity, duplicate IDs, tampered hashes, and the difference between missing evidence and conflicting evidence.

## 5. T02 — Repair the numerical kernels

**Files:** `auxiliary_cv/{model,runtime,force}.py`, `adaptive/aux_pooling.py`, `adaptive/aux_discovery/placement.py`, `correctness/bias.py` where helper reuse is appropriate.

1. Extract a checked active auxiliary-term evaluator with explicit broadcasting/shape rules. Validate subtraction, squaring, multiplication, kcal→kJ conversion and reduced-energy conversion. Return exact inactive zeros without evaluating unused centers. Use one implementation where contracts coincide; retain different runtime/export error types at their boundaries.
2. Validate all converted Context parameters before the first `setParameter` call. Do not leave a partial assignment after a predictable numeric validation failure.
3. Add early model-value range checks and preserve explicit degeneracy detection. Do not silently regularize torsion geometry or promise that observation-time checks protect every intervening MD step.
4. Change `_forecast` to `logsumexp`/`expit` formulas from F04, retaining existing selection thresholds and documented ranking/rounding policy.

**Tests:** `test_aux_cv_runtime.py`, `test_aux_cv_bias.py`, `test_aux_discovery_placement.py`, plus numeric-boundary cases. Assert center `1e200` and conversion overflow refuse. For `du=[800,1000,...]`, N=100, assert O≈0.009900990099 rather than 1. Test finite common-shift invariance and identical-state O=0.5.

**Done:** no runtime/union path bypasses the checked arithmetic, and ordinary-range reference fixtures still agree within their declared tolerances.

## 6. T03 — Make discovery conditioning explicit

**Files:** `adaptive/aux_discovery/{partitions,frames,pipeline,z3_search,settings}.py`.

Introduce a versioned `ConditioningTransform` holding declared/selected dimensions, units, train-only means/scales, constancy thresholds and bin edges. Pass it through fitting, frozen prediction, information-gain calculations and redundancy guards. Replace hard-coded two-axis bin counts. Keep descriptor preprocessing and conditioning-coordinate preprocessing distinct.

An absent CV2 is valid one-dimensional input; an unexpected NaN in a declared active coordinate is invalid data. Constant dimensions are dropped using a documented scale-aware rule. No usable dimension returns `insufficient_conditioning_evidence`. All-constant descriptors and insufficient training rows have explicit outcomes. Frozen legacy partitions may retain a legacy reader; do not reinterpret their arrays as a new transform without schema evidence.

**Tests:** existing partition/pipeline/z3 suites plus a real discovery call with absent CV2, constant CV1 but useful CV2, tied quantiles, constant descriptors, small variance and malformed holdout input. Assert no training statistic is recomputed using holdout data.

## 7. T04 — Obtain the auxiliary force's own value

**Files:** `auxiliary_cv/{runtime,runtime_io,parity_report}.py`, `production.py`, GPU parity runbook/scripts.

Resolve the actual `ATLaSAuxCVUmbrella` force separately for each cloned production System/Context. Add a force-side observer independent of `_use_fast_cv_path`, which is a CV1/CV2 implementation choice. Preserve the full projection offset/scale when deriving z from the auxiliary sub-CVs. Compare it with the independent position/torsion evaluator, including pure-offset models and inactive-state observations.

Use the same observer in production parity reports; no NumPy-vs-NumPy fallback may claim force parity. Record the evaluation source in diagnostic evidence. Keep the existing precision-specific reduced-energy tolerances.

**Tests:** `test_aux_runtime_io.py`, `test_aux_cv_runtime.py`, `test_aux_parity_report.py`. Deliberately perturb the force-side value while leaving the position evaluator unchanged: both distance-primary and contact-primary paths must detect the disagreement. Check cloned Contexts, force groups, state reassignment and checkpoint round trips.

**Done:** CPU/Reference checks pass and the GPU harness uses the same observer. GPU result remains pending until T15.

## 8. T05 — Change Gibbs and ledger semantics together

**Files:** `production.py` (`_gibbs_window_proposal_distribution`, `GibbsProposal`, `gibbs_propose_one_replica`, acceptance path), `auxiliary_cv/runtime_io.py`, `store.py`, exchange schema/replay validators, `kernel_identity.py` and checkpoint compatibility code as required.

Carry authoritative `log_q_forward`, `log_q_reverse` and `log_p_accept`; remove proposal-log clipping. Compute the reverse distribution using the existing hypothetical post-swap assignment. Keep exponentiated probabilities only for the candidate sampler or presentation, never to reconstruct MH log ratios. Define stay/no-candidate sentinels and zero-uniform handling explicitly.

Introduce a proposal algorithm identifier, proposed `gibbs_softmax_log_v2`. Writers store authoritative log values directly. Audit callers using positional `GibbsProposal` construction, `p_override`, logging and RNG order. Choose one acceptance draw and persist the resulting decision; do not recompute an independent acceptance decision for the ledger.

Support historical proposal records under their original version. The current `ledger.py` primarily replays assignments: preserve that function and add/version a separate numerical decision audit rather than claiming assignment replay verifies proposal probabilities. If numeric replay needs the relevant bias matrix and RNG evidence, explicitly require it or report that verification unavailable.

**Tests:** `test_gibbs_walk.py`, `test_exchange_kernel_exact.py`, `test_exchange_fixes.py`, `test_aux_cv_exchange.py`, `test_aux_ledger.py`. Use the F09 harmonic counterexample and exact small-permutation enumeration through the production function. Assert acceptance≈0.26894142137 and finite logged reverse probability even when its exponential underflows. Test old/new record loading, checkpoint/resume and ordinary-range behavior.

**Done:** proposal generation, acceptance, event writer and compatibility reader land in the same integration change. Never ship only the helper fix while `aux_event_fields` still logs rounded probabilities.

## 9. T06 — Persist proposals, validate applicability, then admit

**Files:** `adaptive/aux_discovery/validation.py`, `adaptive/aux_admission_io.py`, `adaptive_production.py`, `cli.py`, `provenance.py`; proposed `adaptive/aux_proposals.py`.

Refactor `run_epoch_aux_discovery` into explicit phases: discover → persist pending proposal → validate evidence → recheck live applicability → prepare admission actions → publish active admission transaction. `aux_pending/<digest>` must never satisfy `inject_aux_phase_args` or `aux_admission_allows_pooling`.

Store immutable model/partition/worker proposal artifacts and training/holdout boundaries. Reuse a pending proposal by identity instead of silently rediscovering a new model behind an old certificate. After the epoch's other actions, recheck parent activity, lambda, budget and worker limits. Reject stale proposals with a recorded reason.

Implement per-check applicability: finite-timestep evidence must cover the actual model/worker/mass/integrator combination; cost evidence may cover an explicit population/feature-count envelope. A source commit string or a maximum spring constant alone is insufficient. `physical_system_check=not_checked` cannot authorize admission.

If the applier accepts only a subset of proposed workers, verify that the certificate explicitly covers that subset and resulting population before publishing its resolved digest. Otherwise leave admission uncommitted and report the discrepancy. Do not reconcile the admission record by silently rewriting its certified scientific identity.

**Tests:** extend admission hook/applier/registry/fix-wave suites. Cover pending resume, stale source/model/system, malformed validation JSON, a documentation-only commit with unchanged scientific digest, parent retirement after discovery, budget changes and partial refusals. No case lacking applicable evidence may create an active auxiliary force.

## 10. T07 — Make burn-in a durable population transaction

**Files:** `adaptive/aux_admission_io.py`, `adaptive_production.py` numbered/scheduled/final phase setup, `store.py` snapshots, checkpoint completion hooks; shared T01 policy.

Create a population-equilibration record at admission. Persist explicit phase IDs, carrier population, planned per-carrier MD duration and durable completed boundaries. Mark affected segments as equilibration before sampling. Remove unconditional `aux_equilibrium_eligible=True` for these segments.

Advance completion only after checkpoint/segment evidence is durable. On resume, reconstruct progress from those boundaries, not from output file presence. A partial epoch/final phase under an exhausted MD budget remains unfinished. Convergence cannot bypass an outstanding population equilibration requirement.

Retain raw burn-in samples for engineering checks. The query layer's equilibrium filtering must not make those samples inaccessible to a deliberately engineering-only force/backfill diagnostic. Conversely, that diagnostic reader must not promote its rows into the equilibrium pool.

**Tests:** extend `test_aux_admission_e2e.py` and crash/resume fixtures. Inject kills around admission publication, registry save, segment close and burn-in completion. Test the configuration moving from worker to ordinary state, scheduled subphases and shortened final phases. No completion event may be fabricated or double applied.

## 11. T08 — Unify pool selection and enforce reconstruction

**Files:** `adaptive/aux_pooling.py`, `adaptive/aux_backfill.py`, `auxiliary_cv/offline.py`, `kernel_identity.py`, `adaptive_production.py::build_union_state_mbar_inputs`, `mbar_analysis/loaders_union_parquet.py`, `loaders.py`, `query.py` integration; proposed `adaptive/aux_reconstruction.py`.

1. Build a metadata/observation inventory with explicit phase, segment, origin state, replica and authoritative step identities. Validate origin maps and common physical/T/P compatibility before taking advantage of common-energy cancellation.
2. Invoke T01 eligibility once per phase/population and return the selection/audit to both union consumers. Remove the duplicated worker-only burn-in loops and any ad hoc admission-file-exists authorization.
3. For post-admission rows, require runtime z plus stored torsions and perform strict per-segment parity. Do not fall back to XTC when a mandatory runtime column is missing.
4. For historical rows, read a reconstruction certificate and verify source/backfill hashes and observation alignment. Separate certificate integrity from the method establishing an error bound.
5. Implement explicit modes: all-phases certified analysis refuses on required missing evidence; certified-subset analysis excludes whole uncertified phases and records them; engineering analysis propagates an uncertified status into filenames/metadata/reports. Never silently switch modes.
6. Recompute counts after selection. Handle states without retained samples and disconnected support explicitly; do not invent finite normalizations to rescue an unidentifiable result.

**Historical certification scope:** first ship the refusal/selection/certificate machinery. It is correct to leave compressed historical phases uncertified if no validated error bound exists. Do not label random coordinate perturbations or the single post-admission spot check a certificate. A rigorous quantization-bound implementation or sufficiently justified reference-based method is an additional acceptance gate for enabling historical certified pooling, not something to approximate merely to make the old fixture pass. Collect high-precision torsion observations prospectively where useful; they cannot be retroactively invented for old runs.

**Tests:** union fixture and builder/loader suites. The `spot_check.ok=false, max_energy_err_kt=100` fixture must refuse in all-phases mode. Test modified backfill, lost certificate, incomplete join, stale final PDB, missing runtime z, mixed precision/schema, changed physical identity and all-samples-excluded. Compare selected observation keys, origin counts and energy matrices between both union paths. Compare matrices only after aligning keys, not assuming row order.

## 12. T09 — Fix export identity and lambda precision

**Files:** `correctness/export.py`, `auxiliary_cv/offline.py`, `store.py`, `query.py`, Parquet manifest/sample schema support.

Require strict observation keys before export. Reuse duplicate detection but remove missing-key bypasses at the strict API boundary. Resolve resume supersession before the exporter; conflicting duplicates must refuse rather than be averaged or silently discarded. Segment identity alone cannot distinguish two copies of the same observation.

Introduce explicit scalar-storage metadata for new samples, including float64 lambda. Legacy reads retain the original per-segment Arrow dtype before DuckDB concatenation/upcasting; compare legacy float32 observations to correctly rounded frozen values, then reconstruct with the frozen float64 Hamiltonian. Mixed legacy/new segments remain distinguishable. Unknown precision requires a specific compatibility failure rather than a guessed tolerance.

Opening an existing nonempty float32 segment must not append a different writer schema to it. Resume starts a compatible/versioned segment through the existing segment machinery, or refuses before registration if it cannot do so safely. Apply the same rule to T10 volume additions.

**Tests:** `test_aux_export.py`, store/query/manifest tests and strict export fixtures. Include duplicates with different labels, absent/masked/fractional/overflow keys, lambda 0.1 round trip, genuinely wrong lambda, adjacent representable rungs, and mixed-schema resume. Verify no hash or `N_k` is computed from an invalid observation view.

## 13. T10 — Store volume at the observation boundary and repair thermo

**Files:** `production.py` sample logging, `store.py`, `query.py`, `mbar_analysis/{data,loaders,loaders_union_parquet,thermo}.py`, NPZ/export adapters and report consumers as applicable.

Add optional `volume_nm3` to the sample API and `Data`. Compute positive finite cell volume from the box vectors at the same pre-exchange, post-integration observation boundary as the energy. Reuse an already acquired State/box when possible, but never use a cached box from before an accepted barostat move. Validate chronology using existing NPT driver scheduling tests.

Preserve optional-volume alignment in `_OPTIONAL_PER_SAMPLE_FIELDS`, `clean`, `_masked_data`, stride/burn-in selection, concatenation and exports. Old samples get missing volume, not zero. Carry frozen ensemble and common pressure identity explicitly into thermo analysis.

Add physical U, pV and H observables to `compute_thermo`, with paired bootstrap resampling and the F11 units/conventions. Audit `_channels`/`own_state_umbrella_kj` so auxiliary/GaMD terms are removed exactly once. Update labels and flattened/report outputs together. Without aligned volume/pressure, NPT delta_H and its derived entropy are unavailable; delta_U remains available.

**Tests:** `test_thermo_decomposition.py`, NPT sample-ordering and store tests, Data alignment tests. Assert equal-volume limit, identical-U/different-volume toy, `1 bar * 1 nm3 = 0.0602214076 kJ/mol`, triclinic boxes, missing volume, zero/negative volume refusal and a volume move immediately before the sample.

## 14. T11 — Repair descriptive diagnostics

**Files:** `adaptive/aux_worker_table.py`, `mbar_analysis/data.py` source metadata, `analyze_gareus_mbar.py`, `gareus_report.py`.

Group episodes by phase-local trajectory identity, after validated resume supersession; never interleave equal local steps from different epochs. Preserve gaps and explicit continuation provenance. Report unique carrier instances separately from known independent ancestry. Do not infer independence from renamed phase/replica strings.

Only compute return-label diagnostics when the frozen evaluation partition and matching frames are supplied. Otherwise report unavailable with a reason. Empty/no-evidence diagnostics cannot produce an affirmative worker-health PASS.

**Tests:** `test_aux_worker_table.py`, analyzer/report tests. The two constant-state phases from the review produce no transition, not two entries/one exit. Include reused IDs, branched seeds, missing frames, real within-phase crossings and checkpoint continuations.

## 15. T12 — Bound memory before adding bootstrap cost

**Files:** `adaptive/aux_discovery/frames.py`, `mbar_analysis/crosscheck.py`, `mbar_analysis/storage.py`, `adaptive/mbar_solve.py` and selected solver backend as necessary.

Use metadata-first deterministic frame selection before descriptor construction and chunked XTC reads. Version the frame selection policy and record selected keys. Do not turn the intended global sampling measure into an unweighted per-state quota. Resume/worker-count changes must not alter the selected observation set.

Replace `_target_logw` full matrix temporaries with `row_blocks`. Materialize ordinary-subset matrices into managed disk-backed storage, and audit the chosen `solve_rows` backend: a memmap input is not a memory guarantee if the solver immediately makes a dense copy. Reuse existing storage cleanup utilities; clean temporary matrices on success, failure and cancellation.

Use a configured memory budget/block size rather than an arbitrary universal RAM requirement. Account separately for O(N) weights/index arrays, O(K) state arrays and O(block_rows*K) scratch. Disk IO and elapsed time belong in the benchmark too.

**Tests:** small dense/chunked parity for f, log weights and PMFs; deterministic selected frame keys; allocation guards against full N*K temporaries; subprocess peak-RSS measurements with a declared budget; temp-file cleanup after injected failure. Coordinate descriptor O(N_residue^2) scaling with issue #143 without changing its feature definition in this repair.

## 16. T13 — Make the statistical verdict explicit

**Files:** `mbar_analysis/crosscheck.py`, existing uncertainty/bootstrap utilities, `analyze_gareus_mbar.py`, `gareus_report.py`, `adaptive/aux_discovery/{z3_search,pipeline}.py`.

First relabel the raw-bin-count crosscheck as heuristic. Introduce explicit `pass`, `fail`, `inconclusive`, `unavailable` and `heuristic` outcomes; no unknown outcome falls through to PASS. Integrity failures from T01/T08 always take precedence.

Then implement the paired population-block bootstrap specified in F12. Resample synchronized time blocks across interacting carriers, separately within fixed-Hamiltonian phases; refit both MBAR estimates on the same resample. Use fixed bins, a declared common reference region and a simultaneous discrepancy criterion. Report block construction/length, replicate count, solver failures and effective support. Missing structural overlap, failed solves or insufficient independent blocks produce inconclusive results; do not quietly remove failed bootstrap draws.

A PASS requires the discrepancy uncertainty bound to fit within the chosen practical tolerance, not merely an interval containing zero. Shared-bias agreement remains a limitation printed with the result. Bootstrap computation may be optional, but omitting it cannot produce a statistical PASS from the heuristic fallback.

Detect the constant-within-lineage null limitation and record `null_uninformative_trapped_lineages` where appropriate. Preserve conservative non-admission; changing the null distribution is outside this plan. Record repeated attempts/holdout reuse without claiming campaign-level false-discovery control.

**Tests:** consistent independent toy, deliberately missing auxiliary term, correlated trajectories, concentrated importance weights, insufficient support, paired-resampling identity, failed solver, simultaneous-band behavior and trapped-lineage labels. Validate coverage/false-positive behavior with repeated synthetic experiments before asserting statistical calibration.

## 17. T14 — Integrate audit and CI

**Proposed new CLI/module:** `python -m gareus.adaptive.aux_audit RUN_DIR --out OUTSIDE_RUN_DIR`, implemented in `adaptive/aux_audit.py`. **Other files:** `.github/workflows/ci.yml`, validation constraints, integration tests and handoff docs.

The audit is read-only and returns machine-readable statuses for each F ID, certificate identities, selected/excluded phases, schema compatibility and missing evidence. It must not fabricate certificates, rewrite old manifests or silently recalculate a campaign under different eligibility rules. Audit v1 campaigns conservatively and explain the available engineering/certified-subset/all-phases routes.

Extend CI beyond its current three exchange test files. Keep an inexpensive core job; add a dependency-complete CPU auxiliary/analysis job and an explicitly selected real-MD job. Mandatory dependencies failing to install must fail that job rather than cause import-skipped success. Publish test/evidence manifests without embedding credentials or machine-private data.

Integration checks include:

- Legacy ordinary and ladder behavior, with documented intentional differences from log-space exchange and schema upgrades.
- One-dimensional real discovery into pending proposal, applicable evidence, active admission, full population burn-in and retained production.
- Crash/resume at every admission/burn-in publication boundary.
- Identical selection/energies in driver and analyzer union paths.
- Old/new sample/event schema mixing, strict export and NPT thermo fallback.
- Default refusal of failed or absent required reconstruction evidence.

Run the broad focused suite at this point with bounded CPU threading. The test inventory includes `tests/test_aux_*.py`, `tests/test_npt_*.py`, `tests/test_contact_pca_strata.py`, exact exchange/Gibbs/fix suites, thermo decomposition and directly affected store/query/data tests. Use the installed validation environment's Python; do not hard-code a transient `/tmp` virtualenv in CI.

**Done:** required CPU jobs pass, expected optional skips are listed, the compatibility audit produces deterministic results, and every F ID maps to implementation/test commits. A GPU-less CI badge remains insufficient for T15.

## 18. T15 — Produce production-applicable evidence

Run this on an isolated validation campaign, with source and dependency identities frozen. Do not borrow a production campaign's completion markers or alter its data to obtain a passing certificate.

1. **GPU parity:** actual auxiliary force value/energy/gradient versus independent evaluator, slow and fast primary CV paths, supported precision modes, state changes and checkpoint round trip.
2. **Finite timestep:** actual mass/HMR/constraint configuration and candidate worker envelope; smaller-step reference; independent repeats and uncertainty for structural populations, restrained-coordinate distributions and relevant energy/volume observables. Freeze acceptance tolerances before seeing results.
3. **NPT:** controlled target with measurable coordinate-volume coupling, correct full potential in trial acceptance, and volume/structural distribution agreement. Include a deliberate missing-auxiliary-energy fault that the harness must detect. Pure torsion translation invariance alone does not exercise this gate.
4. **Scale/cost:** intended population, including 236 contexts if that is the certified deployment, on the intended GPU class. Record aggregate throughput, wall/GPU hours, peak GPU/RAM/scratch use, startup, discovery, backfill and analysis overhead. Report physical validity and affordability separately.
5. Generate validation v2 only from the immutable result artifacts and applicability checks. If no candidate exists yet, validate reusable components and record candidate-dependent evidence as pending; do not manufacture a model-independent pass.

**Completion:** the exact deployment configuration is covered by valid evidence. A source/model/integrator change invalidating that coverage requires revalidation, not manual editing of a pass string.

## 19. Release and review checklist

Each completed task's review states: behavior changed; functions/schemas affected; focused tests run; migration implications; and unresolved evidence. The final repair report must distinguish code-complete from production-validated.

The project can ship conservative repairs before historical compressed-data certification or calibrated bootstrap evidence exists, provided those capabilities remain explicitly unavailable/refused. It cannot advertise the missing capability as repaired or use a fallback to authorize an equilibrium estimate.

Do not roll back an integrity failure by re-enabling the old permissive pooling path. Preserve raw data, report the failure and use an explicitly valid subset or an engineering-only analysis. Keep the report's existing CAUTION ceiling until the diagnostics audit and applicable validation requirements are actually satisfied.

Suggested first implementation batch: T00–T02, followed by T04 and T05. These establish trustworthy numerical kernels and records before the admission/eligibility integration. The full critical path then continues through T06–T08 and T14–T15, with the other tasks integrated according to the dependency table.
