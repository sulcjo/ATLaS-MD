# CVaux correctness repairs: admission, exchange, reconstruction and equilibrium analysis

Status: proposed implementation specification; no fixes or production certification implied.

Repository: `sulcjo/ATLaS-MD`, PR #141. Baseline: `f6f44b03e9f0720d6cee92643cc91e67c68d0c7f`.
On 2026-10-10 the PR head, `feat/cvaux-stage-a`, and `feat/cvaux-adaptive` all resolve to this revision. The latest update folded the already-reviewed adaptive implementation into PR #141; it did not repair the findings below.

This document amends the 2026-10-07 auxiliary-state specification and the 2026-10-09 adaptive-discovery design where they conflict with these requirements. In particular, worker-only burn-in exclusion and warning-only reconstruction checks are replaced. Existing U1–U13 choices otherwise remain: opt-in discovery, no mandatory sham population, frozen admitted model, dedicated worker reserve, and unrestricted exchange.

## 1. Objective and boundaries

Recover defensible equilibrium configurational distributions and thermodynamic surfaces under common-temperature, common-pressure NPT sampling. Correct kinetics, extra umbrella dimensions, MTS, a new force field, and deployment are outside this repair.

The Hamiltonian remains

\[
U_s(x,V)=U_0(x,V)+G_s(x,V)+B_{1,s}(x,V)+B_{2,s}(x,V)+A_s(x),
\qquad A_s=\tfrac12 k_s[z(x)-c_s]^2.
\]

At common temperature, pressure, physical Hamiltonian and configurational measure, state-label swaps at fixed configurations/volumes cancel the common physical and pressure terms. Do not add an extra pressure term to that exchange difference. Include the auxiliary potential exactly once in dynamics, exchange, volume-move energy differences and cross-state reconstruction. Keep it outside the GaMD boost as currently intended.

Distinguish three properties throughout reports and schemas:

1. **Numerical integrity:** evaluations, identities and stored data are internally consistent.
2. **Protocol eligibility:** the declared equilibration and validation protocol completed.
3. **Statistical adequacy:** the retained data support the requested equilibrium estimate and uncertainty.

A completed protocol is not proof of stationarity or ergodicity. A green unit-test suite is not a finite-timestep or GPU validation result.

## 2. Work packages and priority

| ID | Priority | Required repair | Primary code |
|---|---|---|---|
| F01 | P1 | Population-wide post-admission burn-in | `adaptive/aux_admission_io.py`, `adaptive_production.py`, `adaptive/aux_pooling.py`, `mbar_analysis/loaders_union_parquet.py` |
| F02 | P1 | Enforced reconstruction eligibility; independent stored-feature parity | `adaptive/aux_backfill.py`, `adaptive/aux_pooling.py`, `kernel_identity.py`, `auxiliary_cv/offline.py` |
| F03 | P1 | Validation applicability and finite parameter contracts | `adaptive/aux_discovery/validation.py`, `settings.py`, `aux_admission_io.py` |
| F04 | P2 | Stable placement overlap calculation | `adaptive/aux_discovery/placement.py` |
| F05 | P2 | Absent, constant and nonfinite conditioning coordinates | `adaptive/aux_discovery/partitions.py`, `frames.py`, `pipeline.py` |
| F06 | P2 | Correct phase-local worker diagnostics | `adaptive/aux_worker_table.py`, analyzer call site |
| F07 | P1 | Independent force-side runtime parity | `auxiliary_cv/runtime_io.py`, `runtime.py`, `parity_report.py`, `production.py` |
| F08 | P2 | Finite auxiliary arithmetic and explicit singularity handling | `auxiliary_cv/model.py`, `runtime.py`, `force.py`, `adaptive/aux_pooling.py` |
| F09 | P2 | Log-space Gibbs proposal/MH calculation and faithful ledger | `production.py`, `auxiliary_cv/runtime_io.py`, exchange replay and schemas |
| F10 | P2 | Strict export observation identity and lambda storage precision | `correctness/export.py`, `auxiliary_cv/offline.py`, `store.py` |
| F11 | P2 | Correct NPT thermodynamic labels and volume support | `mbar_analysis/thermo.py`, sample/data schemas and loaders |
| F12 | P2 | Honest uncertainty and bounded-memory diagnostics | `mbar_analysis/crosscheck.py`, `adaptive/aux_discovery/frames.py`, analyzer/report |
| F13 | P2 | Executable validation fixtures and release evidence | `tests/conftest.py`, auxiliary/NPT/exchange tests, CI and runbook |

P1 items block certification of affected auxiliary equilibrium production. P2 findings remain required repairs; their impact ranges from loss of exploration to incorrect reporting or failed export. Fixes touching legacy exchange or analysis paths require explicit regression coverage rather than a claim that flag-off behavior is universally unchanged.

## 3. F01 — Equilibrate the interacting population

### Decision

Use **population-wide exclusion** as the initial implementation. Do not introduce a separate quarantined exchange kernel in this patch. All configurations in the population interacting with a newly admitted worker are ineligible during the declared post-admission equilibration interval, regardless of their current state label or original carrier.

The existing next-numbered-epoch convention may define the minimum interval, but an epoch label alone is insufficient evidence that the interval completed. Freeze an explicit minimum MD duration per participating carrier, resolved from the campaign policy before the phase starts. If budget exhaustion prevents completion, the affected population remains ineligible; a short final phase must not silently clear the requirement. The minimum duration is a declared protocol parameter to validate, not a universal physical equilibration time.

### Required behavior

- Create an admission transaction ID and a persisted population-equilibration record containing participating states/carriers, phase IDs, planned MD duration, completed MD boundaries and completion status.
- Mark every phase/segment in that interval as equilibration. `inject_aux_phase_args` must not unconditionally declare it equilibrium eligible.
- Both union builders apply the same selection implementation. No state-specific exception retains ordinary rows from an excluded population interval.
- Exchange transfers configurations and therefore transfers the initialization problem. Filtering only the original worker carrier is also insufficient once configurations interact.
- Scheduled subphases belonging to the affected interval must all be accounted for. Preserve the existing segment/checkpoint boundaries; do not infer completion from a directory name or the presence of final PDBs.
- Resume restores progress from durable checkpoint/segment evidence. A crash cannot advance the eligibility state. Commit completion atomically after the required MD boundary is durable.
- Every subsequent admission resets the requirement for its interacting population. Starting new ordinary states has its own existing equilibration requirements; this repair does not certify those automatically.
- Preserve excluded observations for engineering diagnostics, but keep them out of equilibrium `N_k`, MBAR solves and uncertainty calculations.
- Record exclusions by phase, state and reason, plus the retained population definition.

### Acceptance

Construct an exchange trajectory in which an unequilibrated worker configuration moves into an ordinary state. Both observations must be excluded. Test scheduled phases, premature budget exhaustion, final-phase entry, and crash/resume immediately before and after the completion record. Driver-built and analyzer-built pools must select identical observation identities.

Add a small exact-state counterexample: an ordinary two-state distribution is uniform, the auxiliary state favors one configuration, and the worker starts in the other. One valid MH swap can change the ordinary marginal away from equilibrium. The eligibility policy must exclude that entire transient; passing the MH detailed-balance test is not an acceptable substitute.

## 4. F02 — Reconstruction must govern pooling

### Separate post-admission and historical data

**Post-admission:** require the declared torsion basis, each required recorded `aux_z` column, per-segment runtime precision, frozen model identity and origin-state identity. Recompute z from stored torsions, compare recorded z using the existing precision-specific reduced-energy parity contract, and reconstruct every active auxiliary column. Missing recorded z is an integrity failure here, not authorization to fall back to historical backfill.

**Pre-admission:** absence of recorded z is legitimate. Reconstruct from source coordinates with an explicit reconstruction certificate. Require complete observation joins, source identities, coordinate format/precision, topology mapping, model hash, worker Hamiltonian digest, source-data boundary/hash, reconstruction implementation identity and validation status. Verify the stored backfill hash when reading it; merely storing a hash in the admission record is insufficient.

Compressed XTC and decimal PDB coordinates are approximations to the integrated coordinates. A model hash and a successful join do not establish energy accuracy. The existing post-admission spot check is useful evidence but is not, by itself, a bound for all historical geometries or another storage format.

### Numerical criterion

Evaluate reconstruction errors in reduced auxiliary energies:

\[
e_{ns}=\beta\left|\tfrac12 k_s(\tilde z_n-c_s)^2-\tfrac12 k_s(z_n-c_s)^2\right|.
\]

Where source coordinate quantization has a defensible error envelope, obtain a conservative z/energy bound over that envelope, including conditioning near torsion singularities. A local first-order derivative estimate or random perturbations alone must not be called a rigorous bound. If no reliable bound/reference exists, report the historical reconstruction as uncertified.

Define a versioned historical-reconstruction policy. Proposed strict default: maximum certified cross-state error of **0.05 kT** for every retained observation and active worker, with finite values throughout. This is an engineering energy-error budget, not a theorem that the final PMF error is below 0.05 kT; require end-to-end PMF sensitivity validation as well. The former 1 kT spot-check threshold remains diagnostic only and cannot authorize strict pooling. Do not restrict certification to samples within two worker widths unless omitted contributions have a demonstrated bound.

### Failure policy

- A failed, stale, incomplete or missing required certificate excludes the affected **whole source phase** from an explicitly requested certified-subset analysis, or refuses the requested all-phases analysis. Default all-phases behavior is refusal.
- Do not silently delete only frames with large reconstruction errors: conformation-dependent deletion can bias the distribution.
- An engineering override may expose approximate data only with an explicit non-equilibrium-certified status that propagates into every output. It cannot silently feed the standard equilibrium result or receive PASS.
- A failed `spot_check` blocks reliance on that reconstruction method until resolved. It must never be ignored by either union path.
- Certificate invalidation follows any change to source data, model, applicable worker parameters, topology or reconstruction implementation.
- Final-PDB fallback requires evidence connecting the PDB to the exact final observation/checkpoint. Matching replica/window names is not a time identity.

### Acceptance

The reviewed fixture with `spot_check.ok=false` and `max_energy_err_kt=100` must refuse all-phases equilibrium pooling. Add missing-z, model mismatch, modified backfill, duplicate join key, missing frame, stale final PDB, near-singular torsion and mixed-coordinate-format cases. A separately requested certified-subset result must enumerate excluded phases and recompute `N_k`.

## 5. F03 — Applicable validation, not a pass-string file

Introduce `atlas-aux-validation-v2` and an admission schema referencing its content digest. Validate types explicitly: reject booleans as numeric parameters, nonfinite values, nonpositive temperature/timestep/stiffness bounds, malformed check mappings, missing evidence and unknown schema versions. Parse-time, resume-time and admission-time validation use the same contracts.

A certificate identifies:

- Scientific implementation identity: commit plus relevant source digest; dependency versions, including OpenMM and the GaMD implementation.
- Physical-system and topology identities; force field, masses/HMR, constraints, ensemble, T/P and barostat configuration.
- Integrator settings and timestep; platform, precision and applicable hardware class.
- Exact candidate model and worker-parameter digest, or an explicitly tested model-family bound that covers them. A scalar `k3_max` without control of the model derivatives is insufficient.
- Separate evidence for force/energy parity, finite-timestep distribution convergence, NPT controlled distributions, and production-scale resource cost. Do not conflate cost with physical validity.
- Machine-readable results, sample counts, uncertainty/tolerances, seeds, commands and evidence content hashes. Require nonempty results, not only `checks: pass`.

Use existing `physical_system_sha256` where supported. `physical_system_check.status=not_checked` must block admission requiring that identity. Unsupported force types continue to refuse; canonical hashing for new force fields is a separate feature.

### Candidate-specific validation workflow

The fitted model is unavailable before discovery. Persist a proposal under `aux_pending/<proposal_digest>/` containing the candidate model, worker table, settings, training/holdout boundaries and discovery report. This is not an admission and must not inject forces or enable pooling. Report `pending_validation` while ordinary sampling may continue.

Validation consumes that immutable proposal. On a later boundary, the driver may admit it only if the evidence matches and its parents, resource limits and campaign settings remain applicable after other actions. Revalidate eligibility; do not silently replace the model while retaining the certificate. An invalidated pending proposal gets a recorded rejection reason. Only successful admission publishes the active `aux_admission.json` transaction.

Existing v1 records are evidence pointers, not automatic v2 passes. No fabricated migration of missing identities or results. Provide a read-only compatibility audit explaining exactly what new evidence is needed.

### Acceptance

Reject NaN timestep, infinite stiffness, stale source digest, wrong physical system, changed masses/constraints, model normalization change, wrong platform precision and absent evidence. A documentation-only commit may reuse evidence only when the declared scientific digests and all applicable identities are unchanged. Crash/restart while pending must never inject an auxiliary state.

## 6. F04 — Stable forecast overlap

In `_forecast`, validate shapes, finite positive RT, finite z/c/k and valid labels/lineage indices. Calculate

\[
\Delta f=\log N-\operatorname{logsumexp}(-\Delta u),\quad
\log w_n=-\Delta u_n-\operatorname{logsumexp}(-\Delta u),\quad
O=N^{-1}\sum_n\operatorname{expit}(\Delta f-\Delta u_n).
\]

Use float64 accumulation. Refuse nonfinite energy construction before these operations. Do not repair overflow by clipping energies. Keep existing selection thresholds and historical four-decimal gate/ranking policy in this repair; document intentional departures from old numerical fixtures only where the old arithmetic was wrong.

Regression: `du=[800,1000,...,1000]` with 100 entries must give overlap `0.009900990099...`, not 1. With the reviewed label arrangement, held-out net benefit is positive but overlap must still reject admission. Test invariance to adding a finite common constant to `du`, equal-energy overlap 0.5, extreme separation, and zero-variance parent handling. An overlap slightly above its exact 0.5 maximum may only be tolerated at a declared floating-point rounding scale, never at order-one magnitude.

## 7. F05 — Conditioning coordinate contract

Resolve available CV dimensions from metadata, not by silently treating arbitrary NaNs as absence. On training data only, classify each declared coordinate as usable, constant within an explicit scale-aware tolerance, or invalid. Unexpected nonfinite values are invalid input; an absent CV2 is a supported one-dimensional case.

Persist selected dimensions, units, training means/scales, constancy thresholds and bin edges in a versioned frozen partition. Use float64 preprocessing. Apply the identical transform to holdout and future data; never refit scales or bin edges on holdout.

Support a one-dimensional conditioning space. With no informative coordinates, return a specific `insufficient_conditioning_evidence` status in the initial repair rather than crashing or inventing a dimension. Replace hard-coded `s_bins ** 2` and two-axis indexing with the actual conditioning-bin count. Update z3 redundancy/correlation guards to use the same selected dimensions, so dropping an unavailable coordinate does not trigger a later mandatory-correlation failure. Handle tied quantiles without phantom populated bins. Keep descriptor-family filtering explicit, including the case where every descriptor is constant.

Tests: absent CV2, constant CV2, tiny finite variance, one unexpected NaN, all descriptors constant, too few training rows, tied quantiles, and train/holdout schema mismatch. The production discovery path—not only an injected result—must exercise the one-dimensional case.

## 8. F06 — Diagnostic trajectories and genealogy

Worker entry/exit episodes are phase-local: group by `(campaign_id, phase_id, replica_id)`, resolve resume supersession first, and sort by the phase's authoritative step clock. Do not create transitions across phase boundaries or silently bridge missing intervals. Use explicit continuation/genealogy records only when they demonstrate that a cross-phase trajectory is continuous.

Distinguish physical carrier IDs, phase-local carrier instances and independent ancestral lineages. Report the first two descriptively; do not label either count as independent evidence. For discovery bootstraps, preserve known shared ancestry across phases/seeds. If ancestry is unknown, state that the phase-replica bootstrap is an approximation rather than claiming 20 independent lineages from 20 renamed pieces of a chain.

Wire frozen-partition return-label diagnostics only when their required frames are loaded and validated. Otherwise retain a specific unavailable reason. No fabricated success from an empty worker table or missing overlap evidence.

Regression: phase A `[ordinary,ordinary]`, phase B `[worker,worker]`, reused replica 0, local steps `[1,2]` in both phases must not produce the reviewed spurious two entries/one exit. Test resumed segments, reused replica IDs, scheduled subphases and known branched seed ancestry.

## 9. F07 — Force-side parity independent of CV1/CV2

Always obtain the auxiliary runtime value from the actual auxiliary force attached to each simulation Context, regardless of the primary/secondary CV fast-path choice. Obtain the comparison value independently from positions/stored torsions. An unavailable force-side evaluator is an explicit failure/unavailable result, never a substitution of the same NumPy evaluator on both sides.

Resolve force handles per cloned System/Context. Do not assume that the setup force object belongs to all production contexts. Preserve the complete scalar projection, offset and scale when recovering z from its force sub-CVs. Pure-offset models need explicit coverage too.

Require parity evidence for distance-primary slow path and contact-primary fast path, active/inactive worker states, lambda-zero and boosted states, NVT/NPT, and checkpoint round trips. Use the existing documented precision-specific reduced-energy tolerances; replacing an exact-zero assertion with the appropriate tolerance is valid, weakening tolerances to accommodate a defective evaluator is not.

GPU evidence must run on the actual supported GPU backends/precision modes. CPU tests cannot set GPU status to pass. No automatic production launch belongs to this repair.

## 10. F08 — Finite arithmetic and singularities

Validate numeric input types, shapes and finiteness at model/state boundaries. Evaluate active auxiliary terms under explicit overflow/invalid checking and check the assembled output after subtraction, squaring, multiplication, unit conversion and beta scaling. Reject invalid energies with model/state/carrier/step provenance before exchange or pooling. Never replace a nonfinite term with zero or quietly remove its state column.

For `k=0`, construct exact zero contributions without evaluating unused centers or hazardous arithmetic. Before setting Context parameters, validate every converted value; `k * 4.184` must be finite before either parameter is changed. Prevalidate whole assignments to avoid leaving a partly changed target after a predictable validation failure.

Use conservative model-value bounds where possible (`|offset| + sum|coefficients|`, divided by scale) to catch impossible numeric ranges early. Such bounds do not bound torsion derivatives near collinearity. Keep explicit geometric degeneracy checks at setup, seeds, observations and checkpoint boundaries; reject a nonfinite integrated state and prevent its output from being certified. Do not claim observation-time checks prevent every between-observation integration failure.

Any future regularization of torsion singularities changes the CV Hamiltonian and requires a new model identity and matched runtime/offline evaluation. It is not an invisible numerical patch.

Tests: finite center near `1e200`, converted stiffness overflow, tiny positive model scale, malformed array dimensions, masked/NaN/infinite coordinates, degenerate torsions, all-inactive states, and valid large finite values. Reuse one checked auxiliary-energy helper in adaptive pooling and strict/runtime assembly where their contracts coincide.

## 11. F09 — Keep Gibbs probabilities in log space

For each valid candidate, compute `log_q = -beta*delta - logsumexp(-beta*delta)` without the current +/-745 clipping. Retain log probabilities through proposal selection, reverse-proposal evaluation and MH correction. The reverse proposal remains evaluated against the hypothetical post-swap holder assignment.

\[
\log\alpha=\min\{0,-\beta\Delta U+\log q_{reverse}-\log q_{forward}\}.
\]

An exponentiated probability used by a sampling API or displayed in a report is not the authoritative value for acceptance. In particular, an underflowed reverse probability must not replace its finite log probability. Define zero-uniform handling explicitly if drawing acceptance in log space. Validate finite reduced energies before forming proposals; nonfinite auxiliary arithmetic follows F08, not candidate masking as a recovery strategy.

Extend `GibbsProposal` to carry authoritative log probabilities. `runtime_io` must write those values directly instead of taking logs of rounded probabilities. Version the proposal algorithm in exchange/kernel metadata and update replay accordingly. Preserve replay support for historical records under their recorded algorithm; do not pretend a historical `-inf` field contains a recoverable finite log probability. Historical and new records need an explicit compatibility assessment, not a blanket assertion of bitwise equality.

Regression: with beta=1, centers `c=[sqrt(10),0,0.1]`, configurations `z=[0,sqrt(10),-0.1]`, harmonic matrix `u[s,r]=50*(c[s]-z[r])**2`, identity assignment and carrier 0 proposing state 1, acceptance must be approximately `0.26894142137` instead of zero. Add exact small-permutation stationary-distribution/detailed-balance tests through the production proposal function, ordinary-range compatibility and ledger replay of the extreme case.

## 12. F10 — Export identity and storage precision

Strict export requires validated observation identity. Within one phase, duplicates are determined by `(replica, absolute_step)` after authoritative resume supersession; across phases include the phase identity. Segment ID is provenance, not a way to make two copies of one observation distinct. Require appropriate integer types/ranges and nonmissing keys before conversion. Do not use `astype(int)` to turn fractional identifiers into valid observations.

Share duplicate detection between strict export and auxiliary pooling. The existing helper's early return when identity columns are missing must not bypass the export contract. Do not silently deduplicate at the exporter; the caller must resolve conflicting segment history explicitly. Compute counts and export hashes only after a validated selection.

Write lambda as float64 for new sample schemas. For legacy float32 columns, compare against the correctly rounded frozen lambda in the declared storage dtype, then use the frozen float64 value for Hamiltonian reconstruction. Do not apply a loose global `isclose` that could conflate nearby distinct rungs. Preserve storage-precision metadata through query/loading; upcasting alone loses the evidence needed to choose a comparison rule.

Tests: duplicate identical/conflicting rows, missing identity, fractional/overflow IDs, equal steps in different legitimate phases, lambda 0/1/0.1, a genuinely wrong lambda and adjacent distinct representable rungs. Adding a duplicate must refuse, not change `N_k` from `[4,4]` to `[5,4]`.

## 13. F11 — NPT enthalpy and entropy

At common temperature, configurational basins defined independently of momenta and the same degrees of freedom, the kinetic contribution cancels between basins. Report

\[
\Delta H=\Delta\langle U_{physical}\rangle+p\,\Delta\langle V\rangle,
\qquad -T\Delta S=\Delta G-\Delta H.
\]

Keep removal of umbrella/GaMD bias from recorded potential energies consistent with the actual Hamiltonian. Add per-observation volume in nm^3, measured at the same configuration boundary as the energy, and carry it through sample storage, loaders, `Data` slicing/cleaning and bootstrap resampling. Require known common pressure in bar. Conversion: `p_bar * V_nm3 * 0.0602214076` is molar pV in kJ/mol. Bootstrap U and V together using the same weights and dependence blocks.

For old NPT data lacking reliable aligned volumes, report internal-energy differences as `delta_U`; mark enthalpy and its derived entropy unavailable. Do not silently label delta_U as delta_H or call `delta_G-delta_U` an NPT entropy contribution. An approximate neglect of pDeltaV, if exposed at all, must be explicitly requested and labeled approximate. Distinguish NVT Helmholtz and NPT Gibbs decomposition in output metadata.

Tests: equal-volume limit; a two-basin toy with identical U but known different volume; unit conversion; row slicing; bootstrap alignment; missing volume/pressure; and correct labels for historical output.

## 14. F12 — Statistical claims, discovery limitations and memory

### Ordinary/all-state crosscheck

The existing raw-count expression is a heuristic, not a statistical uncertainty estimate for correlated, unequally weighted MBAR samples. Preserve it only under an explicitly heuristic status. For a statistical verdict, use a paired dependence-aware block/bootstrap procedure: resample the same population blocks for both estimates, refit both MBAR problems, use common bin edges and a common declared free-energy alignment, and compare their difference distribution. Preserve correlations among replica-exchange carriers within the chosen time blocks; independent per-row resampling is not sufficient. Report low effective support as inconclusive.

Separate practical accuracy tolerance from uncertainty. A precision-based PASS requires an uncertainty bound on the discrepancy that lies within the declared tolerance over the supported domain; an interval merely containing zero is insufficient. Handle multiple bins through a simultaneous criterion, such as a bootstrap maximum-deviation band. A confidently excessive discrepancy is FAIL; insufficient precision/support is inconclusive. Neither agreement nor PASS can override failed integrity or burn-in gates, and shared bias can make both estimates agree incorrectly.

### Discovery interpretation

Keep the current null gate conservative during the correctness repair. Detect constant-within-lineage labels explicitly and report `null_uninformative_trapped_lineages` when shifts cannot provide a meaningful discrimination test. Do not weaken the threshold, treat ties as passes, or inject native labels to make this case admit workers. A replacement discovery test is a separately versioned methodological change requiring null controls and metastable synthetic positives.

Record repeated discovery attempts and held-out reuse. Twenty null searches and repeated boundaries do not establish campaign-wide false-discovery control. Report predictive/placement scores as heuristics unless a sequential testing procedure is explicitly implemented and validated. Benchmark benefit using equilibrium surface uncertainty and basin population reproducibility per aggregate GPU-hour, including discovery/backfill/analysis overhead; replica-label mixing alone is not the target.

### Resource contracts

Compute MBAR log denominators in row blocks, and allow the ordinary-state subset to remain disk-backed. Avoid unconditional full `N x K` intermediate arrays in the crosscheck and its repeated solves. One float64 matrix with 10 million rows and 236 columns is 18.88 GB before temporaries.

Apply the discovery frame budget before allocating descriptors for the whole campaign. Use a deterministic, metadata-first selection followed by chunked trajectory reads, with recorded selected observation identities and a versioned selection policy. Preserve the intended sampling measure; changing a global uniform sample into an unweighted per-state quota is a statistical behavior change, not a free optimization. Coordinate streaming work with issue #143 without making a new descriptor design a prerequisite for core correctness.

Acceptance: chunked/dense agreement on small data; deterministic selection across resume/worker counts; bounded block allocations as N grows; and a measured production-scale memory/time report. Explicit resource exhaustion must never silently skip a required validation and retain PASS.

## 15. F13 — Tests and evidence

Reviewed baseline evidence: OpenMM 8.5.1, CPU/Reference; focused auxiliary, NPT and contact-PCA tests produced **916 passed, 2 failed, 2 skipped**. These are baseline results, not results for the repairs.

Repair the two failures deliberately:

- `test_aux_parity_report.py`: replace exact-zero floating-point equality with the documented numerical tolerance and retain a test that rejects physically significant disagreement. The observed baseline discrepancy was `1.1102230246251565e-16`.
- `tests/conftest.py` Pep-GaMD fixture: explicitly configure valid preparation and cumulative stage lengths, with averaging-window compatibility. Its 100-step stages currently conflict with 5000-step preparation defaults under the tested upstream GaMD version. Assert that the intended boosted/NPT/admission paths actually ran; do not swallow initialization failures or substitute a mock integrator.

The tiny GA end-to-end fixture deliberately replaces discovery. Keep it for plumbing/resume tests, and add a separate real-discovery synthetic/peptide fixture that exercises partitioning, conditioning, the null gate and proposal persistence. Do not interpret short real-MD smoke runs as equilibrium distribution validation.

Required evidence before certification:

1. Focused regression tests for F01–F12 and unchanged mathematical identities.
2. CPU real-MD admission and checkpoint/crash/resume through ordinary and Pep-GaMD paths.
3. Independent GPU force/energy/z parity on supported precision modes and checkpoint round trips.
4. Controlled common-T,p NPT distribution tests, including volume and a CV/volume-coupled case; verify the auxiliary contribution participates in volume acceptance rather than relying only on a translation-invariant torsion example.
5. Finite-timestep comparisons against a smaller-step reference, with uncertainty on relevant equilibrium observables and prescribed tolerances, at the actual masses/constraints and certified model/worker envelope.
6. A 236-context cost/memory test, or the explicitly certified intended production population, recorded separately from physics validity.

CI must actually select the auxiliary and affected legacy exchange/export/analysis tests. Record optional-backend skips as missing evidence for that backend, not passes. Freeze exact dependency versions and commands for validation; the upstream GaMD branch name alone is not a reproducible dependency pin.

## 16. Implementation sequence and completion criteria

**Wave A — enforce eligibility:** F01–F03 and F07. Implement common validators/selection, pending proposals, certificate identities and atomic resume transitions before expanding analysis access. Existing uncertified campaigns receive an audit and explicit eligibility status; never rewrite their historical evidence to claim a pass.

**Wave B — numerical kernels:** F04, F05, F08, F09. Keep each repair independently testable. Version proposal/frozen-partition changes and document expected non-bitwise differences. Do not duplicate physics formulas in tests as the sole correctness oracle; use analytic cases and exact enumeration.

**Wave C — export and interpretation:** F06, F10–F12. Update both union paths, all per-sample slicing, report schemas and compatibility readers together. New volume collection must precede any claim of NPT enthalpy support for new runs.

**Wave D — validation:** F13, revised runbook and a read-only compatibility audit for existing campaigns. Re-run evidence invalidated by scientific code/model changes. A revision is ready for certified auxiliary production only when applicable integrity, burn-in, reconstruction, GPU, timestep and NPT gates pass; costs and statistical support remain separately visible.

Required handoff artifacts: implementation commits; tests linked to F IDs; schema migration notes; baseline-to-new behavior table; validation manifest with immutable evidence hashes; and a concise list of unavailable results. This document specifies the repairs; implementation and production validation remain outstanding.

## References

- Shirts and Chodera, *Statistically optimal analysis of samples from multiple equilibrium states*, J. Chem. Phys. 129, 124105 (2008): <https://arxiv.org/abs/0801.1426>.
- PyMBAR timeseries documentation, equilibration and correlated observations: <https://pymbar.readthedocs.io/en/latest/timeseries.html>.
- OpenMM standard-force/barostat theory: <https://docs.openmm.org/latest/userguide/theory/02_standard_forces.html>. Pin the exact runtime documentation/version in validation evidence.
- Existing repository specs: `2026-10-07-auxiliary-cv-gibbs-production-spec.md`, `2026-10-07-cvaux-gpu-parity-runbook.md`, `2026-10-09-cvaux-adaptive-discovery-design.md`.
