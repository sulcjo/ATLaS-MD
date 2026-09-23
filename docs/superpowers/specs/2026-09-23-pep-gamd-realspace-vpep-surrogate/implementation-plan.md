# Pep-GaMD real-space surrogate: implementation handoff

Date: 2026-09-23  
Status: proposed implementation plan; no implementation or tests executed for this handoff.  
Repository baseline: `sulcjo/atlas-md`, `main` at `fa42346b8f18f459e1de93c264d60824f6cd665d` (PR #93 merge; same commit deployed on aurum2 `DEPLOYED_COMMIT`). Originally written against `dc9cb297cd17f9405ce296b796a99d3f1d6e708a`; revised 2026-09-23 for the shared contact-sum CV force layout, see §2.1.  
Canonical specification: `docs/superpowers/specs/2026-09-23-pep-gamd-realspace-vpep-surrogate/spec.md`, including its new §14. The earlier flat specification path has moved.  
Suggested repository destination: `docs/superpowers/specs/2026-09-23-pep-gamd-realspace-vpep-surrogate/implementation-plan.md`.

## 1. Outcome and scope

Implement an opt-in `pep-gamd-lower-dual-rs` kernel using a peptide-focused real-space measurement coordinate. Keep the physical PME Hamiltonian unchanged, remove the water-only auxiliary PME from RS contexts, and preserve a single consistent dependent-dual bias in integration, exchange, NPT, recording, and MBAR.

Separate **software acceptance**, **experimental pilot eligibility**, and **campaign promotion**. Passing unit tests is not evidence of improved exploration, converged thermodynamics, or GPU speedup. Implementation must not start or resume chignolin_9 automatically.

Initial supported target: fixed-charge, PME, rigid TIP3P solvated peptide with supported harmonic bonds/angles and peptide torsions. Support S1-direct for offline comparison and S1-smooth for gated pilots. Defer S2/S3 until evidence requires them; do not advertise unimplemented flavours. No MTS, timestep increase, force-field change, CV redesign, or new cumulant approximation.

Preserve exact-kernel force arithmetic and normal finite-input results. Shared fail-closed safeguards may intentionally reject runs previously allowed through silent fallbacks; document these separately from the numerical kernel change. Do not promise bit-identical trajectories between different Langevin algorithms.

## 2. Baseline findings that affect the work

These are code-reading findings at the pinned baseline, not test results.

| Location | Existing behaviour | Required consequence |
|---|---|---|
| `gareus/pep_gamd.py` | Exact-only type, named auxiliary discovery, `E0-E1+E2`, exact NPT adapter | Introduce family/role dispatch; preserve the exact implementation |
| `production._fetch_v_pep_v_dih` | Directly calls `peptide_essential_energy_kj`; shared by sample/exchange | Change this helper, not just calibration helpers |
| `production.make_cmd_integrator` | Excludes group 1 only when the exact auxiliary is found | Recognize either measurement partition |
| `swarm.members` and `swarm.driver` | Build exact partition; members measure exact energy | Make both setup and measurement kernel-aware |
| `swarm.envelope.write_envelope_setup_dir`; `swarm.analyze` sidecar | Hard-coded exact boost type | Carry explicit kernel/coordinate identity through the whole swarm |
| `production.load_reusable_shared_gamd_setup` | Loads/copies globals without RS identity validation | Validate before copying files or touching a Context |
| `kernel_identity_for_run` versus `production.verify_kernel_identity_on_resume` | Digest includes boost type; resume guard compares only exchange/CV versions | Digest existence alone does not block exact↔RS resumes; enforce comparison |
| `store.SampleWriter` | Raw boost energies and lambda written as float32 | Float64 for new raw-energy/lambda columns; test actual round trips |
| `swarm.ladder_design` | Uses `lambda * DeltaV_max` to design rungs and reweight CV2 | Use full dependent-dual `DeltaV_lambda` for RS ladder design |
| `swarm.analyze` | Reuses `selection` for CV selection and seed selection; missing member status defaults to success | Fix explicit §14 campaign blockers |
| `swarm.epoch0._artefact_digests` | Hashes explicit files, but directory contents only through `final_survivor_seeds.csv` | Envelope JSON is already explicitly listed and hashed here; do not repeat the board's unsupported claim that it is absent. Seed-bank coverage still needs repair |
| `production.sample` | Reads native boost globals after propagation and possible volume moves | Characterize coordinate timing before adding equality checks |

The exact integrator already reads bias groups directly rather than recovering them through all-group force subtraction. Benchmark the actual baseline, not an obsolete implementation.

### 2.1 Baseline change: shared contact-sum CV force (PR #93, `fa42346`)

PR #93 does not touch `gareus/pep_gamd.py`, groups 0/1/2, or the Pep-GaMD boost algebra. It changes which bias groups exist, and that feeds the integrator's per-DOF bias variables, checkpoint compatibility and the benchmark baseline.

| Location at `fa42346` | Behaviour | Consequence for RS |
|---|---|---|
| `production.cv_force_layout` / `add_umbrella_cv_forces` | CV1 nonlocal-contacts + CV2 `residual-torsion-pc` → `SHARED_CONTACT_LAYOUT`: CV1 umbrella `0.5*k*((res_contacts/contact_norm)-r0)^2` appended to the CV2 CustomCVForce in group 29; group 31 empty. Otherwise `SPLIT_CV_LAYOUT` | Three reachable bias-group shapes: shared `(29,)`, split 2-D `(29, 31)`, CV1-only `(31,)`. chignolin_9 uses `cv2: auto` with `fallback: cv1_only`, so its shape is decided by the swarm, not the config. Every RS integrator/NPT/audit test runs over all three |
| `pep_gamd.pep_gamd_bias_force_groups` | Bias groups = occupied groups minus {0,1,2}, by construction | RS inherits this only if **all four** RS measurement components sit in group 1. An RS component parked anywhere else would silently become a "bias group" and be applied at full force. T01 validator enforces this |
| `production._restore_secondary_cv_args_from_metadata` | Sets `args.cv_force_layout` from `secondary_cv_metadata["cv_force_layout"]`; absent record → split (pre-PR campaign) | Layout is now Hamiltonian-construction identity. T08 guard compares it alongside RS descriptor |
| `production.reusable_checkpoint_matches_bias_groups` | Shared-GaMD `.chk` loaded only when recorded `bias_force_groups` equal current; unrecorded → not loaded, globals copy only | Precedent for T08. It covers only the binary checkpoint: `load_reusable_shared_gamd_setup` still copies files and returns globals before any identity check |
| `production.fast_cv_force_indices` / `observe_fast_path` | Shared layout: CV1 read from the `contact_sum` sub-CV of the CV2 force; both observers point at one force | Unaffected by RS (`_fetch_v_pep_v_dih` reads groups 1/2 only), but RS wiring tests must run on a shared-layout system |
| `benchmarks/c8_integ_bench4.py` arm `P5`, `BIAS=split\|shared`; job 2608721 | 236 contexts, 59/GPU, MPS on, PME stream disabled: split 2286/2294, shared 2630/2618 ns/day/node (+14.6 %); 1 context/GPU +8.1 %. Bias energy/forces equal on Reference (dE 0, max\|dF\| 1e-14) | Exact-kernel control for T11 = `P5 BIAS=shared` (or the layout the pilot actually uses). The ~2307 figure is split-layout history |

Tests at baseline: `tests/test_shared_contact_cv_force.py`, plus the Pep-GaMD suites below.

## 3. Frozen mathematical and data contracts

### 3.1 Force and coordinate definitions

For RS, group 1 contains only the surrogate `S`; group 2 contains the physical dihedral channel `D`. Define:

```text
R = S + D                         # recorded v_pep_kj_mol
bD = g(D; lambda*k0D, envelopeD)
bT = g(R + bD; lambda*k0T, envelopeT)
Delta = bD + bT
Ustar = Uphys + W + Delta
Fstar = Fphys+bias - (1-FSF_T)*F_S - (1-FSF_T*FSF_D)*F_D
```

Use the actual upstream branch/small-range rules, not merely an unguarded quadratic. Never replace `Delta(lambda)` with `lambda*Delta(1)`. Inactive channels have their effective scaling factor equal to one.

`F_S` acts on every atom participating in S, including solvent/ions: do not zero its solvent components. “Peptide-focused” describes the selected interactions, not a permission to violate the energy gradient by applying only half the pair force.

Group 1 is excluded from bare integration force and ordinary physical-plus-restraint energy. It may be read explicitly for measurement. `physical_potential_energy_kj` currently includes restraints outside group 1; retain/document this legacy reporting convention rather than silently changing it. NPT still separates physical and bias contributions for reporting.

### 3.2 Descriptor and frozen-envelope identity

Proposed import-light `gareus/pep_gamd_rs.py` contains a frozen `RSSurrogateConfig` and the builder; OpenMM imports occur inside runtime construction functions. `pep_gamd.py` keeps dispatch, integration and NPT contracts, avoiding cyclic module-level imports.

An RS coordinate descriptor must include:

- descriptor/schema version and flavour (`s1-direct`, `s1-smooth`);
- actual numeric alpha, cutoff, switching distance, electrostatic constant and expression version;
- mixing/LJ/switching/LRC policies, exception PBC policy, bonded selection policy;
- sorted peptide-atom-index digest;
- physical parameter/topology fingerprint sufficient to detect atom reordering, changed exceptions, or changed charges/LJ;
- canonical component-definition digest, including exclusions and interaction groups.

Use canonical JSON with normalized units and finite numbers. Hash coordinate identity separately from the frozen envelope payload and state-table identity; these have different meanings. Add descriptor digest to RS kernel identity only, retaining old exact identity serialization for valid legacy exact runs. Do not introduce a redundant second string duplicating the RS boost type.

Persist identity in swarm plan/member metadata, envelope JSON, run manifest, window/segment metadata, and checkpoint compatibility data. Restore through serialized force definitions plus persisted descriptor; do not depend on Python attributes surviving XML serialization. An already-built RS partition is idempotent only when its components match the requested descriptor and physical system. Missing, duplicated, renamed, altered or partial components fail.

### 3.3 Supported construction, explicitly bounded

For the first implementation, require one physical `NonbondedForce` with ordinary fixed parameters and supported PME settings. Refuse parameter offsets, unsupported nonbonded methods/LJPME, unsupported custom physical terms, and systems whose torsion partition cannot be identified as the intended peptide channel. Do not silently reinterpret these inputs.

Use disjoint interaction groups peptide×peptide and peptide×nonpeptide, including ions in nonpeptide. All System particles must have surrogate particle parameters. Mirror the full physical exception exclusion list in the custom nonbonded force (including nonpeptide exceptions, which have no contributing pairs here) to support platform exclusion compatibility; restore only nonzero peptide-involving exceptions. Independently enumerate the contributing pair set in tests.

Exceptions use their actual charge product, sigma and epsilon with full unscreened exception Coulomb/LJ, not erfc screening or the ordinary-pair switch. Copy the physical exception periodic-boundary setting. Reject unsupported cross-boundary covalent/bonded terms in this first scope rather than inventing a peptide selection rule.

Copy peptide-internal harmonic bonds/angles only; keep all physical originals unchanged. Retain zero-term components if needed for a uniform four-component inventory. Constraints/masses/virtual sites remain physical and unchanged. Audit rigid-solvent bonded terms and the NPT molecule partition before/after adding measurement forces.

S1-direct ordinary pairs: `C*q_i*q_j*erfc(alpha*r)/r` plus physical pair LJ and physical LJ switching if enabled. It is an offline diagnostic, not a production default.

S1-smooth proposed first concrete flavour: multiply the unswitched ordinary-pair Coulomb+LJ sum by a quintic switch `p(t)=1-10t^3+15t^4-6t^5`, with `t=clamp((r-r_on)/(r_c-r_on),0,1)`. Require `0 < r_on < r_c`; initial benchmark candidate `r_on=0.9*r_c`, recorded explicitly, not an immutable scientific optimum. Disable any additional built-in switch to avoid double switching. This deliberately changes the surrogate LJ near the cutoff, not physical LJ.

Disable surrogate long-range correction. Physical PME and its dispersion correction remain unchanged. Resolve the surrogate alpha once from explicit physical parameters or the existing tolerance formula; do not modify the physical force's PME parameters just to construct the surrogate. Freeze alpha and distances under NPT; evaluate S at the trial box using the same descriptor. The descriptor is not recalculated as volume fluctuates.

## 4. Ordered implementation tasks

Each task starts with a failing regression, ends with focused checks and a short evidence record. Names marked “new” are proposed APIs/files, not existing repository features. Use the sequence below; do not expose a usable RS CLI route until all propagation/measurement/NPT/persistence paths work.

### T00 — Baseline characterization and acceptance fixtures

**Files:** existing `tests/pep_gamd_fixture.py`, `tests/test_pep_gamd_boost.py`, `tests/test_pep_gamd_wiring.py`, `tests/test_pep_gamd_bias_force_groups.py`, `tests/test_gamd_frozen_envelope_stage5.py`, `tests/test_shared_contact_cv_force.py`; new `tests/test_pep_gamd_rs_platform_contract.py`.

1. Record commit, Python/OpenMM/gamd-openmm versions, CUDA precision/device and available test dependencies. Read any applicable repository instructions at implementation time.
2. Run existing exact-kernel tests before edits; distinguish skips, dependency failures and real baseline failures.
3. On a tiny independent-force fixture, prove bare `f`/`energy` respect the integration mask while explicit `f1`/`energy1` remain readable. Use separate computations for each force/energy read.
4. Prove the same on Reference and supported CUDA versions; test masked cMD and a dummy native volume controller only where its acceptance is supposed to match exposed physical energy.
5. Characterize native boost-global timing after a warmed step, accepted/rejected external volume move, and label swap. Inspect the installed upstream instruction sequence.
6. Parameterize the acceptance fixture over the three bias-group layouts of §2.1 (`(29,)`, `(29, 31)`, `(31,)`), with a cheap stand-in CV where the full residual-torsion-pc model is too heavy; later tasks reuse this parameterization.
7. Add an independent physical-System clone for comparisons; zero-boost trajectory equality must use the same update algorithm/noise/constraints, not compare GaMD Langevin to LangevinMiddle and demand equality.

**Exit:** mask semantics, timestep phase and deterministic test oracle established. Failure of explicit excluded-group reads blocks RS; do not enable all-group integration as a workaround.

### T01 — Configuration, inventory and identity contract

**Files:** new `gareus/pep_gamd_rs.py`; `gareus/pep_gamd.py`; new `tests/test_pep_gamd_rs_identity.py`.

Proposed public interfaces:

```python
resolve_rs_config(args, system) -> RSSurrogateConfig
ensure_pep_gamd_rs_partition(system, peptide_atoms, flavour, *, config=None) -> dict
find_pep_gamd_measurement_forces(system) -> tuple
validate_pep_gamd_partition(system, *, expected_kind, descriptor=None) -> None
```

Keep `find_aux_force` exact-only for existing exact-specific callers. Add explicit `is_exact_pep_gamd`/`is_rs_pep_gamd`; make `is_pep_gamd` family membership only when all generic callers have been audited. Treat names as role identifiers plus structural validation, not trust in any arbitrary Force carrying a prefix.

Validator rejects exact+RS coexistence, unrelated group-1 occupants, any RS component outside group 1 (it would otherwise be counted by `pep_gamd_bias_force_groups` as a bias group and applied unscaled), malformed RS sets and mismatched descriptor. Exact builder must reject RS as well as RS builder rejecting exact. Preserve reserved groups and existing bias-group complement.

**Exit:** pure configuration/hash tests and invalid-inventory tests pass without importing OpenMM for pure tests.

### T02 — Real-space measurement builder

**Files:** new `gareus/pep_gamd_rs.py`; `pep_gamd.assign_pep_gamd_force_groups`; new `tests/test_pep_gamd_rs_partition.py`, `tests/test_pep_gamd_rs_energy.py`.

Build exactly named group-1 components: `PepGaMDRSNonbonded`, `PepGaMDRSExceptions`, `PepGaMDRSBonds`, `PepGaMDRSAngles`. Stage components and validate all inputs before mutating the System. Assign physical groups exactly as the existing contract requires; recognize RS copies before physical class classification.

Test independently enumerated ordinary pairs, zero exclusions, scaled/nonstandard exceptions, particle permutations, peptide-solvent forces, PBC boundary crossings, constrained solvent, XML round-trip and idempotence. Check component sum against group-1 energy and finite differences on peptide and solvent atoms. Sweep the smooth switching region and both boundaries. Physical potential/forces, masses, constraints, box and molecule connectivity must match the unmodified physical model.

Run a CUDA construction/energy smoke early: interaction groups/exclusions may behave correctly on Reference yet fail or perform poorly on the actual backend.

**Exit:** both S1 definitions validated; S1-direct remains diagnostic-only.

### T03 — RS integrator, dependency ordering and diagnostic witness

**Files:** `gareus/pep_gamd.py`; new `tests/test_pep_gamd_rs_integrator.py`.

Add lazily built `PepGaMDRSLowerDualIntegrator`; reuse upstream lower-dual stages but do not rewrite the exact subclass's numerical instructions. Set `TOTAL_ENERGY_PLUS_GROUPS={1,2}`, `TOTAL_ENERGY_MINUS_GROUPS={}` and the integration mask excluding 1 before Context construction.

Energy setup reads E1 and E2 into separate globals. Force update separately copies `f1` and `f2`, then uses one bare `f` expression with stored arrays. Preserve thermostat, constraints, RNG consumption and first-step behaviour. Conventional stages use masked bare `f`.

Add a last-force-evaluation witness containing raw R/D, both native boost components, effective FSFs, stage, channel parameters and lambda-related k0 values captured at the same force evaluation. It must retain values before position/box/label changes and must not overwrite upstream working globals for reporting.

Tests cover all three bias-group layouts of §2.1 (per-DOF bias-variable count must equal `len(pep_gamd_bias_force_groups(system))`), both channels active, either inactive, lambda=0, lambda=0.5/1, umbrella blindness, calibration→production, out-of-envelope values and actual nonzero motion. Compare against an independent finite-difference Ustar oracle; test the force effect on solvent too. Add mutations for missing D in R, wrong F_S sign, omitted bias, doubled surrogate and incorrect channel order.

Energy reconstruction agreement alone is insufficient: the native audit and its independent reconstruction could agree while the integrator applies an incorrect force. Require two force-level oracles in addition to energy checks: (a) central finite differences of the independently evaluated Ustar for representative peptide and solvent coordinates, and (b) a deterministic warmed one-step comparison against an independently assembled reference integrator using identical positions, velocities, constraints, timestep, random seed/noise path and update ordering. Include nonzero dual boost and mutations that preserve reported energies while corrupting the applied F_S/F_D coefficients; those mutations must fail.

**Guard caveat:** the inherited hard `E+b < threshold` and small-range guards can introduce non-smooth branch boundaries outside the normal calibrated domain. Before any pilot eligibility, enumerate every reachable guard surface and test one-sided energy limits, analytic versus finite-difference forces, and short deterministic trajectories crossing it. An unobserved crossing is not evidence that the boundary is safe. Preserve existing semantics for equivalence, but fail the software/pilot gate if continuity or force consistency is unacceptable anywhere in the declared supported domain; any guard redesign requires a separately versioned method change propagated to every consumer.

**Exit:** algebra and phase-aligned witness proven. Do not infer one-PME wall-time cost solely from source expressions.

### T04 — Generic measurement and runtime construction

**Files:** `gareus/pep_gamd.py`, `gareus/production.py`, `gareus/swarm/members.py`, `gareus/swarm/driver.py`, relevant system setup/validation callers discovered by symbol search; new `tests/test_pep_gamd_rs_wiring.py`.

- Generalize physical-energy and cMD exclusion discovery.
- Dispatch RS from `build_pep_gamd_integrator`/`make_gamd_integrator`; retain exact dispatch unchanged.
- Give `_fetch_v_pep_v_dih` an explicit boost-coordinate definition or args-derived group tuple; all four sample/exchange fast/slow call sites must use it.
- Make `swarm.members.measure_frame` use generic raw Total measurement, and both swarm partition construction sites dispatch by intended campaign kernel even though swarm propagation is unboosted.
- Thread identity through `_gamd_boost_group_targets`, cMD recon, shared setup and any serialized-System reconstruction path.
- Build order: the RS partition and `add_umbrella_cv_forces` may run in either order; after both, `verify_pep_gamd_bias_force_groups` must hold and the layout recorded in `secondary_cv_metadata` must match the forces actually present.
- Keep the single-dihedral path valid: absent Total energy may remain NaN there, not in a dual RS path.
- Update CSV/NPZ metadata descriptions that currently say physical-minus-water-only.

For initial RS support, require a frozen envelope for production (fresh unboosted swarm/recon may create it). No mid-production envelope adaptation. Measure raw coordinates at lambda=0 as well as boosted states. Do not silently disable measurement just because a ladder object was not initialized; either construct the supported state definition or reject that RS mode explicitly.

**Exit:** cMD/recon/swarm/production construction and observation tests prove R=S+D at every entry point.

### T05 — RS NPT target and transaction checks

**Files:** `gareus/pep_gamd.py`, `gareus/npt.py`, `gareus/npt_driver.py`, relevant `system_setup.py` ownership checks; new `tests/test_npt_rs_adapter.py`.

Add `PepGamdRSLowerDualNptTargetAdapter`, with a distinct adapter ID and validated descriptor. Generalize force roles to measurement (mapping existing auxiliary diagnostics compatibly); measurement energy never enters physical/bias totals directly. Conventional adapters must exclude RS group 1; stock lower-dihedral adapters must reject either foreign measurement partition.

Freeze live channel parameters once per trial. Evaluate S,D,physical,bias independently at both endpoints, apply dependent boost exactly once, and leave controller PV/Jacobian/centroid/RNG/scheduling logic unchanged. Compare molecule partition before/after adding surrogate forces so measurement bonds cannot silently change NPT Jacobian ownership.

Invalid current-state required energies raise. Nonfinite trial energy rejects and restores the exact prior state with diagnostic counters; never replace it with zero. Accepted volume moves invalidate observable caches even at an unchanged integration step. Test a trial that moves a pair through the surrogate switch region.

Add RS to `SUPPORTED_BIASED_MC_BOOST_TYPES` and `LADDER_BOOST_TYPES` together only once this adapter is tested. Keep biased MC required for boosted NPT.

**Exit:** stage-aware boost parity, old/trial independent oracle, zero-boost NPT and controller checkpoint round trips pass.

### T06 — Fail-closed boost pricing and phase-correct runtime checks

**Files:** `gareus/production.py`, `gareus/pep_gamd.py`, `gareus/correctness/bias.py`, `gareus/mbar_analysis/ladder.py`; new `tests/test_rs_runtime_hamiltonian_echo.py`, extend missing-coordinate tests.

Implement two separate checks at every RS production sample:

1. **Applied last-kick witness:** compare captured native bD+bT and FSFs against independently reconstructed values using the captured R,D, parameters and stage from that same kick. Verify expected production stage and frozen channel parameters. This catches the chignolin_8 inactive-stage failure without comparing different configurations.
2. **Current-frame pricing:** evaluate the current-frame boost from live frozen parameters through the independent NPT target calculation and compare with the sampling-state entry of the exchange/MBAR boost matrix from current R,D and campaign envelope. Read on the Context-owning worker; include current box and assignment/lambda phase.

The current-frame check must not masquerade as a new force application. Native boost globals are last-evaluation values until proven otherwise. Do not refresh them by stepping MD, setting dt=0, or restoring checkpoints inside ordinary sampling. These actions risk modifying counters, RNG, constraints or calibration.

Define one explicit transaction/phase contract for propagation, sampling, exchange, external volume moves and checkpointing. Every persisted observation must state whether it represents the pre-swap or post-swap assignment, and its coordinates, box, state ID, lambda, R/D values, boost matrix and witness phase must all refer to that same logical state. Test `sample→exchange`, `exchange→sample`, accepted/rejected `volume-move→sample`, and checkpoint/resume at each boundary, including a swap immediately adjacent to a volume move. Resume must reproduce the same next logical transaction and must never pair pre-swap energies with post-swap labels.

**Necessary refinement of spec §14 F1/F4:** record last-kick native diagnostics with an explicit phase; record frame-aligned bias from the validated current-frame evaluator. Do not compare stale native globals to the new frame and fail valid runs, or relabel stale native output as the frame's boost. Document this refinement in the spec as part of implementation. If the installed native reader can supply frame-aligned components without mutation, a tested explicit refresh API may replace the two-phase arrangement; it is not assumed available.

For RS native audit components, use known version-tested channel keys and fail if missing, malformed, nonfinite or duplicated. Never use `infer_gamd_boost_kj_from_globals` as an RS fallback. At lambda=0, zero boost is legitimate; at lambda>0, zero may also be legitimate outside thresholds, so do not require all sampled boosts to be positive.

Unified required-input policy: runtime exchange raises on nonfinite required coordinates/parameters; MBAR marks/excludes invalid observations with counts; NPT handles trial failures transactionally as above. A coordinate for a genuinely inactive axis/channel is not required. Derive the secondary-axis active predicate from one state-table definition for both sample and exchange. No zero-imputation of an active-axis NaN.

Use reference-unit tolerance 1e-6 kJ/mol where measured achievable; define tighter/looser backend tolerances from controlled tests, with separate absolute and relative terms and a dimensionless beta*error ceiling. Do not silently loosen tolerances on mismatch. A failed audit prevents sample publication and further exchange, records context/phase/parameters, and aborts safely.

Under the shared layout the umbrella energy W (CV1 and CV2 terms) lives in one force in group 29; umbrella-blindness and wrong-state-label mutations must be run there as well as on split and CV1-only systems.

**Exit:** tests detect stale globals, wrong stage, wrong envelope, wrong lambda, missing D, reader failure and NaN. Valid accepted-volume-move and post-swap samples must pass.

### T07 — Swarm identities, sidecar fixes and dependent ladder design

**Files:** `gareus/swarm/{driver,members,envelope,analyze,epoch0,ladder_design,gates}.py`; extend existing swarm tests and add `tests/test_swarm_rs_handoff.py`.

Remove hard-coded exact type in envelope writer and sidecar. Make boost type/descriptor explicit arguments, reserving identity keys so generic `meta` cannot override them. Require consistent member identity before pooling; missing or incompatible status/descriptor in new RS members is an error, never presumed exact or successful.

Rename CV-selection result and seed-selection result to distinct variables. Test manual CV2 retained, explicit none retained, automatic pair and automatic 1D fallback. Reject incomplete members from calibration with an explicit gate outcome.

Replace the linear-lambda assumption for the RS path with a callback/matrix `Delta(lambda,n)=pep_gamd_boost_kj(R_n,D_n,lambda,env)` using raw arrays. Compute stabilized provisional rung weights from `-beta*Delta(lambda,n)` and rung increments from `Delta(lambda_b,n)-Delta(lambda_a,n)`. Apply the same exact lambda dependence to `reweighted_cv2_centers`. Label all unboosted-swarm predictions as provisional: pooling nonequilibrated seeds is not an equilibrium acceptance estimate.

Do not force the last rung to 1 without checking the final edge. If the rung budget cannot support the requested endpoint, return a failed/inconclusive ladder proposal; do not silently change endpoint/envelope. Test analytic nonlinear dual examples, degenerate energy traces, tiny ESS and an impossible final edge. Use independent pilot overlap to confirm the actual ladder.

The existing `1-lambda*k0max` FSF report is only an in-envelope channel estimate, not a global RS effective-force bound. Report actual FSFs/effective forces on pilot frames, including below-Vmin configurations. Preserve configured safety floors; do not invent force clipping. Any reduced boost budget requires a newly frozen candidate envelope before production.

Repair seed-bank digest coverage by hashing a deterministic manifest of the actual exported files and their content hashes; inspect the exporter layout before choosing names. Retain the already-present envelope JSON hash, add descriptor/state-table binding, and prove mutations are detected rather than blindly implementing the review's directory claim.

**Exit:** end-to-end RS swarm→envelope→sidecar→production dry handoff succeeds and exact/RS mixing fails.

### T08 — Persistence precision and unconditional resume compatibility

**Files:** `gareus/store.py`, `gareus/logger.py`, `gareus/production.py`, `gareus/kernel_identity.py`, `gareus/provenance.py`, `gareus/checkpoints.py`, `gareus/query.py`, relevant MBAR loaders and adaptive-production envelope-copy paths; new `tests/test_rs_recording_roundtrip.py`, `tests/test_rs_resume_identity.py`.

Write new raw R/D and lambda columns as float64 in Parquet; preserve float64 NPZ and round-trip-safe CSV/JSON formatting. Do not downgrade scientific inputs through `analysis_array_dtype`. Keep legacy reads supported without claiming their old float32 data meets new tolerance. Test mixed-schema read promotion/masks and non-binary lambda values such as 0.3.

At every reusable-envelope load, validate boost type, RS descriptor digest, exact channel values/envelope digest, units and finite numbers before copying or returning globals. Today `load_reusable_shared_gamd_setup` calls `_copy_shared_gamd_setup_files` and writes the local payload before any check, and `reusable_checkpoint_matches_bias_groups` gates only the later `.chk` load. Generalize that helper into one identity matcher (bias groups, `cv_force_layout`, boost type, RS descriptor digest, envelope digest), record all of these at export next to the existing `bias_force_groups`, and call it before the copy. An export lacking any RS field is refused for RS, not treated as legacy. Missing RS identity is a hard failure; legacy exact compatibility must be explicit. Binary shared-setup reuse cannot cross coordinate definitions.

Extend the actual resume guard (`verify_kernel_identity_on_resume`) to compare kernel and Hamiltonian-defining fields, not only exchange/CV versions: boost type, RS descriptor digest and `cv_force_layout`. The layout is already restored from `secondary_cv_metadata` by `_restore_secondary_cv_args_from_metadata`, but nothing compares it with what the rebuilt System actually contains; test that a shared-layout RS campaign rebuilt as split (or vice versa) is refused before checkpoint load. Validate original ordered state table, state IDs, lambda assignments, descriptor and frozen envelope before any checkpoint load or output append. An intentional adaptive grid extension must use the existing explicit new-epoch/segment workflow with old-state identity preserved; an accidental reordered/relabelled table is not a resume.

Test exact→RS, RS→exact, shared↔split layout, RS flavour/switch/alpha/cutoff/atom-set changes, modified envelope, missing descriptor, changed state assignments, unchanged restart, fresh-process serialization and context checkpoint restoration. Test no partial output writes on refusal. For new RS analysis, require authoritative lambda metadata and consistent kernel/envelope across sources; do not silently infer it through nanmedian. Exact and RS campaigns cannot be merged using one R value per frame, since cross-pricing would require both coordinates.

**Exit:** saved-frame reconstruction meets tolerance and all incompatible resume/analysis combinations fail before mutation.

### T09 — CLI, documentation and end-to-end thermodynamic tests

**Files:** `gareus/{cli,config,helptext}.py`, guide/reference documentation, new RS example config; existing thermodynamic-validity, ladder MBAR and exchange tests plus new `tests/test_thermodynamic_validity_rs.py`.

Expose the opt-in boost type only now. Proposed options: `--pep-gamd-rs-flavour s1-smooth`, `--pep-gamd-rs-switch-distance-nm VALUE`; cutoff/alpha are resolved from the physical system in v1, not separately adjustable. Reject RS-only flags on exact runs. Ordinary production refuses diagnostic S1-direct. Unsupported backend/system combinations fail preflight. Preserve existing defaults.

Extend a known-answer smooth toy system with a surrogate deliberately different from its physical potential. Run actual RS propagation and existing exchange→recording→MBAR pipeline against independent quadrature/reference probabilities. Include both boost channels, multiple lambdas and umbrella states. Add NPT coupled-target validation with explicitly box-dependent S and the existing volume controller.

Test lambda=0 anchor, duplicate-rung consistency, and sensitivity mutations: sign reversal, direct S contamination, dropped D, double lambda, wrong envelope, stale applied phase, wrong state label and omitted measurement energy from the boost. Predefine statistical tolerances and independent seeds; do not enlarge them until a wrong implementation passes. Run smaller-timestep controls: exact bias pricing and Metropolis volume acceptance do not eliminate Langevin discretization bias.

Treat surrogate-versus-exact correlation strictly as an efficiency/suitability screen. It is never thermodynamic acceptance evidence, even when correlation is near unity. Thermodynamic acceptance requires the independent known-answer distribution tests, lambda=0 anchor, exchange/MBAR consistency and convergence/support diagnostics above.

**Exit:** fast suite, full existing regression suite and slow thermodynamic tests pass in the declared environment. A skipped CUDA or slow-MD test remains an open gate.

### T10 — Offline surrogate gate and reproducible frame manifest

**New tool:** `tools/pep_gamd_rs_offline_gate.py`; new tests for metric edge cases. Place generated reports outside source inputs, under a new experiment directory.

Accept serialized physical System, topology, complete coordinates plus per-frame box vectors, and frame manifest containing source/run/window/block IDs. Use separate exact and RS evaluation contexts; never place both measurement partitions in one System. Validate identical physical model and evaluate identical atom ordering/boxes.

Chignolin_8 data may provide geometric diversity but must not be called an equilibrium reference given known historical pipeline problems. Solute-only seed PDBs cannot reconstruct S including peptide-solvent energy; use documented solvated/grafted and relaxed configurations or retained full-system frames, labelled accordingly.

Compute the spec's energy correlations/scale ratios and peptide-force cosine/norm metrics for B_exact=E0-E1 versus S. Add all-atom force errors, worst outliers and structural-stratum diagnostics. Undefined correlation from zero variance and near-zero force norms are inconclusive, not pass. Use block-aware intervals and a held-out frame split for candidate selection.

Preserve stated prefilters: pooled r≥0.90, per-populated-window r≥0.80, sigma ratio 0.8–1.25, median cosine≥0.90, p10≥0.70, all finite, no norm collapse. Define population/coverage sufficiency before inspecting results. A failed prefilter cannot automatically promote; broader science-based exceptions require an explicit reviewed experiment, resolving the spec's tension between strict prefilters and efficiency-based selection.

**Exit:** JSON plus readable report, hashes, candidate descriptors and explicit pass/fail/inconclusive decision. No candidate passing means retain exact or reconsider the design, not lower gates silently.

### T11 — Boosted pilot, GPU benchmark and release gate

**Files:** extend or wrap the shipped `benchmarks/c8_integ_bench4.py` (successor of `c8_integ_bench3.py`; arm `P5` = real Pep-GaMD integrator in stage 5 with real CV forces, `BIAS=none|split|merged|shared`) and `benchmarks/c8_sharedcv.sh`; retain archived historical logs unchanged; new `tools/pep_gamd_rs_pilot_report.py` if existing reporting cannot express required comparisons.

Parameterize hard-coded run paths/arm selection; use the real integrator, warm-up, fresh appropriate envelope, stage 5 and the T06 audit. Do not use the branch-free historical harness arm as a correctness oracle. Record envelope/descriptor/source/device/version hashes.

Benchmark exact versus RS with/without real CV forces at one context/GPU, 16/GPU and the documented high-occupancy 48/59/GPU regimes where resources permit. Use four L40S/192 CPU threads as the primary node target. Recorded 236-context MPS results with PME stream disabled: about 2307 ns/day/node (job 2580889) and 2286/2294 (job 2608721) for the split layout, 2630/2618 for the shared layout (job 2608721). These are historical context, not the RS control measurement: the exact control is re-measured in the same job as RS, with the CV layout the pilot actually uses (shared `(29,)` for a residual-torsion-pc CV2, CV1-only otherwise). The shared layout removed the duplicated contact sum; the second PME is untouched, so any RS gain is measured on top of that baseline. Use equal replica count, precision, timestep, output/exchange/barostat cadences and MPS settings, with repeated timings and compilation excluded. Measure audit overhead explicitly. Count actual force/kernel work rather than promising one PME evaluation from an algebraic expression.

Pilot compact/intermediate/extended label-free seed strata, using independent seeds and newly fitted RS envelopes. Compare against exact and lambda=0 controls at matched scientific conditions; report per-step and per-wall-time performance. Include finite/constraint stability, force/guard crossings, boost/FSF distributions, actual ladder overlap, block-aware target ESS, zero-rung/full-ladder agreement and retention of compact seed regions. Use coverage as an efficiency diagnostic, not a native-structure selection criterion. No Gaussianity/CE2 assumption is needed by exact ladder MBAR; anharmonicity remains diagnostic.

Promotion requires reproducible improvement in useful sampling per wall time with no validated thermodynamic regression, not merely faster steps. A short pilot cannot prove folding convergence. Keep the spec's publication-validation gates distinct: comparable physical-model reference, independent convergence, identical-rung tests and connected-support window-removal sensitivity. Removing essential connecting windows is not a valid invariance test. A published FES with different force field/water/temperature is a qualitative comparator, not an exact oracle.

**Exit:** written promotion decision. Software may merge as experimental before real-system promotion; never mark scientific gates passed because hardware or trajectories were unavailable.

## 5. Commit sequence and acceptance evidence

| Commit unit | Tasks | Required evidence |
|---|---|---|
| 1. Baseline and contracts | T00–T01 | Existing exact tests; mask/timing probe; identity fixtures |
| 2. Measurement kernel | T02 | Independent pair/exception/bonded/PBC oracle; XML; CUDA smoke |
| 3. RS integration | T03 | Warmed finite-difference force oracle and mutations |
| 4. Runtime/NPT wiring | T04–T05 | All constructors, cMD exclusions, trial-box parity |
| 5. Runtime safeguards | T06 | Phase-correct witnesses, strict reader, missing-data tests |
| 6. Swarm handoff | T07 | Identity propagation, manual CV2, nonlinear ladder tests |
| 7. Durable scientific identity | T08 | Float64 round trips, restart refusals, strict analysis identity |
| 8. Experimental public route | T09 | CLI preflight and complete end-to-end regression |
| 9. Validation utilities | T10–T11 | Reproducible manifests/reports; no fabricated benchmark claims |

Every commit record should list changed contracts, test command, platform, pass/fail/skip counts, and remaining blockers. Intermediate commits must not expose a partially supported production kernel. Update the folder README and spec status only to the state actually achieved.

Suggested test commands after proposed test files exist:

```bash
python -m pytest -q tests/test_pep_gamd_boost.py tests/test_pep_gamd_wiring.py tests/test_pep_gamd_bias_force_groups.py tests/test_gamd_frozen_envelope_stage5.py tests/test_shared_contact_cv_force.py
python -m pytest -q tests/test_pep_gamd_rs_platform_contract.py tests/test_pep_gamd_rs_identity.py tests/test_pep_gamd_rs_partition.py tests/test_pep_gamd_rs_energy.py tests/test_pep_gamd_rs_integrator.py tests/test_pep_gamd_rs_wiring.py
python -m pytest -q tests/test_npt_rs_adapter.py tests/test_rs_runtime_hamiltonian_echo.py tests/test_swarm_rs_handoff.py tests/test_rs_recording_roundtrip.py tests/test_rs_resume_identity.py
python -m pytest -q tests/ -m 'not slow'
python -m pytest -q tests/test_thermodynamic_validity_rs.py
python -m pytest -q tests/ -m slow
```

Run in the configured research environment with gamd-openmm installed; it is not included in the shown `pyproject.toml` extras. Do not launch cluster jobs or install/upgrade research dependencies without the user's established authorization. Record known external-fixture requirements rather than bypassing tests.

Repository-wide final sweep (review every hit; historical docs/exact-only tests are legitimate):

```bash
rg -n 'PepGaMDWaterOnlyNonbonded|find_aux_force|ensure_pep_gamd_partition|peptide_essential_energy_kj' gareus tests
rg -n 'pep-gamd-lower-dual|LADDER_BOOST_TYPES|SUPPORTED_BIASED_MC_BOOST_TYPES' gareus tests
rg -n 'v_pep|v_dih|float32|infer_gamd_boost|load_reusable_shared_gamd_setup' gareus
rg -n 'nanmedian|state_gamd_lambdas|kernel_identity|shared_gamd_setup|artefact_digests' gareus
rg -n 'cv_force_layout|SHARED_CONTACT_LAYOUT|bias_force_groups|reusable_checkpoint_matches' gareus tests
```

## 6. Stop conditions and rollback

- Stop for a mask/force mismatch, failure of either independent force oracle, ambiguous unsupported force, incomplete identity, nonfinite required live value, mismatched envelope, phase-incorrect comparison, transaction-label mismatch, any unvalidated or inconsistent reachable guard boundary, incompatible resume, or disconnected requested ladder.
- Never fix these by including measurement in the physical force, replacing NaN by zero, skipping the NPT adapter, borrowing the exact envelope, clamping FSFs ad hoc, or falling back to a guessed boost.
- Roll back an RS campaign by ending it and starting a separately identified exact campaign with an exact envelope. Coordinate reuse requires its normal equilibration; no RS/exact checkpoint or sample append.
- Preserve existing production campaigns and historical result logs. No in-place migration of chignolin_8.
- No performance or equilibrium result is claimed by this document. It is a build-and-validation plan, not completed validation.

## 7. Source anchors

- [Current specification and §14](https://github.com/sulcjo/atlas-md/blob/fa42346b8f18f459e1de93c264d60824f6cd665d/docs/superpowers/specs/2026-09-23-pep-gamd-realspace-vpep-surrogate/spec.md)
- [Pep-GaMD implementation](https://github.com/sulcjo/atlas-md/blob/fa42346b8f18f459e1de93c264d60824f6cd665d/gareus/pep_gamd.py)
- [Production, measurement, resume and sampling](https://github.com/sulcjo/atlas-md/blob/fa42346b8f18f459e1de93c264d60824f6cd665d/gareus/production.py)
- [Swarm ladder design](https://github.com/sulcjo/atlas-md/blob/fa42346b8f18f459e1de93c264d60824f6cd665d/gareus/swarm/ladder_design.py)
- [Swarm epoch-0 identities](https://github.com/sulcjo/atlas-md/blob/fa42346b8f18f459e1de93c264d60824f6cd665d/gareus/swarm/epoch0.py)
- [Parquet writer](https://github.com/sulcjo/atlas-md/blob/fa42346b8f18f459e1de93c264d60824f6cd665d/gareus/store.py)
- [Kernel identity](https://github.com/sulcjo/atlas-md/blob/fa42346b8f18f459e1de93c264d60824f6cd665d/gareus/kernel_identity.py)
- [Historical production-stage benchmark](https://github.com/sulcjo/atlas-md/blob/fa42346b8f18f459e1de93c264d60824f6cd665d/docs/superpowers/specs/2026-09-23-pep-gamd-realspace-vpep-surrogate/benchmarks/c8_integ_bench3.py)
- [Current benchmark harness, split/shared CV arms](https://github.com/sulcjo/atlas-md/blob/fa42346b8f18f459e1de93c264d60824f6cd665d/docs/superpowers/specs/2026-09-23-pep-gamd-realspace-vpep-surrogate/benchmarks/c8_integ_bench4.py)
- [Shared contact-sum benchmark log, job 2608721](https://github.com/sulcjo/atlas-md/blob/fa42346b8f18f459e1de93c264d60824f6cd665d/docs/superpowers/specs/2026-09-23-pep-gamd-realspace-vpep-surrogate/benchmarks/results/c8_sharedcv_2608721.log)
- [Shared-layout tests](https://github.com/sulcjo/atlas-md/blob/fa42346b8f18f459e1de93c264d60824f6cd665d/tests/test_shared_contact_cv_force.py)
- [Historical MPS log](https://github.com/sulcjo/atlas-md/blob/fa42346b8f18f459e1de93c264d60824f6cd665d/docs/superpowers/specs/2026-09-23-pep-gamd-realspace-vpep-surrogate/benchmarks/results/c8_mps59_2580889.log)

API references checked during planning: [OpenMM CustomIntegrator](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomIntegrator.html) for persistent globals and separate group reads; [CustomNonbondedForce](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomNonbondedForce.html) for interaction groups, exclusions and switching; [NonbondedForce](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.NonbondedForce.html) for physical parameters and exception settings. These latest pages identify a development version; installed-version runtime tests in T00/T02, not the latest-page label, determine support.
