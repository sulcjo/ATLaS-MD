# ATLaS-MD auxiliary CV states in Gibbs exchange production

Detailed design and revised chignolin pilot, version 1.0, 7 October 2026.

Status: proposed implementation specification. The exchange architecture follows the user clarification of 7 October 2026. Numerical pilot thresholds below are proposed policy values, not empirically validated guarantees. They must be frozen with the analysis before efficacy runs begin.

Repository baseline: `sulcjo/ATLaS-MD`, main commit `c56b4b333a9aedfa1a4b4da62716a0620b6f26c8`. Intended handoff: implementation and scientific review by the ATLaS-MD maintainers. Suggested repository destination: `docs/superpowers/specs/2026-10-07-auxiliary-cv-gibbs-production-spec.md`.

## 1 Purpose and governing decision

Add a small number of thermodynamic states carrying an auxiliary collective-variable restraint to the existing ATLaS-MD replica population. Spawn their configurations using the ordinary preparation and admission machinery, then admit them to ordinary production. All occupied states participate in the existing unrestricted, Metropolis-Hastings-corrected Gibbs-walk exchange mechanism.

An auxiliary state is a Hamiltonian, not a permanently attached worker trajectory. Its spawning parent records where its initial configuration and baseline restraints came from. It does not constrain future exchange partners. A configuration may enter an auxiliary state from any ordinary state, visit other auxiliary states, and return to any ordinary state.

The physical objective is more efficient recovery of equilibrium structural distributions. Physical kinetics are not a target. Sampler transport metrics below describe the enhanced sampler and must never be reported as physical transition rates.

The two existing umbrella coordinates and their model definitions remain frozen within a phase. This change adds selected Hamiltonians with an auxiliary restraint; it does not create a Cartesian CV1 × CV2 × CVaux umbrella grid. CV1 and CV2 restraint parameters can differ between states and change when a configuration changes its assigned state. Their coordinate values are evaluated from that configuration.

### 1.1 Changes to the previous pilot

This specification supersedes the parent-only exchange topology, direct-parent-return primary endpoint, associated exposure forecasts, and decision table in `2026-10-07-c10-z3-worker-pilot.md` v2. It preserves the following intentions:

- Initially four auxiliary states at lambda zero, using one frozen torsional model.
- Matched additional sham states, identical in baseline restraints to the auxiliary states but with auxiliary strength zero.
- Automatic placement with frozen inputs and no native structure in selection or promotion.
- Separate runs from the live chignolin campaign and independent confirmation before adoption.
- Complete Hamiltonian accounting, strict missing-data handling, and explicit thermodynamic eligibility.

The earlier review's demand to implement parent-only Gibbs proposals is withdrawn. Unrestricted proposals are the desired behaviour. Geometry-graph problems remain relevant to diagnostics and any future neighbour-only mode, but are not a reason to restrict this Gibbs pilot.

### 1.2 Scope

MVP supports common temperature, common pressure when NPT, one physical force field, one frozen boost envelope per phase, one auxiliary model, and at most one active auxiliary restraint per state. The manifest reserves a model registry so a later phase can contain several local models, but MVP must reject more than one distinct active auxiliary model unless that extension passes its own tests. A future limit of four models is not a claim of current support.

Different force fields, temperatures, continuously changing CVs, adaptive biases during retained production, coordinate-dependent state eligibility, and uncorrected state insertion/removal during a production segment are out of scope. No MTS or integrator change is introduced.

## 2 State and configuration semantics

Use separate identities for these objects:

| Object | Meaning | Persists through exchange |
|---|---|---|
| Configuration carrier | Coordinates, velocities, box and continuous dynamical history owned by a simulation Context | Yes; its assigned state may change |
| Thermodynamic state | Frozen potential definition and thermodynamic parameters | Yes; a different carrier may occupy it |
| State instance | A unique occupied slot in the production population | Yes, even if another slot has the same Hamiltonian |
| Hamiltonian hash | Canonical identity of the potential and ensemble | Shared by physically identical shams and parents |
| Model hash | Exact CV function, feature schema, atom map, units and coefficients | Fixed within a phase |
| Spawn parent | Source state/configuration used for initial preparation | Provenance only |

Maintain exactly one carrier per occupied state instance and a bijective assignment map. Preserve the current assignment-swap implementation rather than introducing coordinate copying at exchange. If another implementation swaps coordinates, its carrier identities must follow coordinates, velocities and box consistently.

An ordinary configuration has a numerical auxiliary-CV value whenever the active model is evaluable. It is not assigned a constant CVaux merely because its current restraint strength is zero. A zero-force state has no preferred auxiliary centre.

## 3 Hamiltonian and force contract

For state s, configuration x and volume V, define

$$
U_s(x,V)=U_0(x,V)+G_s(x,V)+B_{1,s}(x,V)+B_{2,s}(x,V)+A_s(x,V).
$$

Here U0 is the common physical potential, Gs is the exact existing frozen GaMD or Pep-GaMD boost for that state's lambda, and B1/B2 are the existing umbrella terms. Do not replace the repository's boost evaluator by an assumed linear scaling with lambda.

For the auxiliary term,

$$
A_s(x,V)=\begin{cases}
0,&k_{a,s}=0,\\
\frac{1}{2}k_{a,s}[z_{m_s}(x,V)-c_{a,s}]^2,&k_{a,s}>0.
\end{cases}
$$

The auxiliary force is

$$
\mathbf F_{a,s}=-k_{a,s}(z_{m_s}-c_{a,s})\nabla_x z_{m_s}.
$$

Energies use kJ/mol inside OpenMM and the exchange kernel. User-facing state tables may use kcal/mol; perform exactly one explicit conversion by 4.184. Dimensionless z implies k in energy per z squared. Reduced energies use beta = 1/(RT) in the matching molar units. For a harmonic restraint of nominal width sigma, k = RT/sigma squared.

### 3.1 Zero strength

For an inactive restraint, branch directly to an exact zero contribution. Do not calculate `0 * (NaN - centre)**2`. Inactive auxiliary metadata canonicalises to no active model, centre zero and strength zero for Hamiltonian identity. Diagnostic model evaluation is specified by the phase registry, separately from whether the current state feels that model.

Missing, negative or nonfinite force constants are errors, not synonyms for no pull. Positive strength requires a known model and finite centre. Any configuration participating in unrestricted exchanges must supply the coordinates needed by every active model in that population, even while it occupies an ordinary state.

### 3.2 Initial torsional model

Use the frozen c10 candidate only after the original complete artifact is recovered and hashed. Its proposed form is

$$
z(x)=b+\sum_j[a_j\sin\theta_j(x)+d_j\cos\theta_j(x)].
$$

The prior pilot describes 18 backbone torsions, 36 sine/cosine features, 35 nonzero coefficients, and a frozen scale of 3.181. Those values are historical inputs, not independently reproduced measurements. The actual coefficient file, topology mapping and schema are authoritative; do not reconstruct the model from prose or its abbreviated hash.

Record atom quadruplets, angle sign, feature ordering, terminal handling, periodic imaging, offsets, normalisation and coefficient units. If porting negates an angle, negate its sine coefficient and leave its cosine coefficient unchanged. Do not refit, renormalise or flip signs using pilot outcomes.

### 3.3 OpenMM deployment

Install the same supported auxiliary model capability in every carrier Context before production. State assignment activates the corresponding centre and force constant; an ordinary state deactivates it. This avoids rebuilding a Context when a configuration enters an auxiliary state.

For a scalar that is a sum of features, implement the square of the complete sum. A collection of independently squared torsion restraints is a different potential because it loses cross terms. Use a validated CustomCVForce or equivalent composition whose total energy is exactly A_s.

Update auxiliary parameters together with CV1, CV2 and boost parameters in the existing state-assignment operation. The exact complete target tuple must be set, including resetting auxiliary strength to zero when leaving an active state. Invalidate cached state-dependent values afterward.

Allocate force groups through the existing force-group audit; do not assume an unused fixed group number. Auxiliary forces must enter the integrator once at full strength and remain outside the energy groups defining the boost. Audit the actual integrator expression: a force-group label alone is not proof of correct inclusion. No independent clipping of auxiliary forces is allowed.

If z is undefined for a degenerate geometry, fail admission or fail the affected production segment with diagnostics. Do not silently return zero or remove candidate states from a proposal based on an evaluation failure.

### 3.4 NPT and finite timestep

The barostat's trial acceptance must include the change of the same full U_s used by dynamics, including A_s, at the trial coordinates and box. Reuse the established volume proposal and Jacobian; this feature adds an energy term, not a new volume formula. Do not assume an internal coordinate is invariant under the actual volume move.

Correct exchange and volume acceptance do not prove finite-timestep configurational accuracy. Validate strong auxiliary restraints at the intended timestep against a shorter-step reference on a controlled model and representative peptide probes. Preserve the existing production timestep until that comparison passes.

## 4 Gibbs exchange

At an exchange barrier, freeze coordinates and boxes for the entire exchange sweep and evaluate the state-by-carrier matrix

$$
E_{s,r}=G_s(x_r,V_r)+B_{1,s}(x_r,V_r)+B_{2,s}(x_r,V_r)+A_s(x_r,V_r).
$$

Runtime shape is K states × K carriers. Offline MBAR shape is N observations × K states; assert the transpose convention at the interface. At common temperature and pressure, the physical potential and pV are configuration-dependent but state-independent, so they cancel from a state-assignment swap. They must not be omitted for extensions where this common-base assumption fails.

For carriers i and j in states a and b,

$$
\Delta E=E_{b,i}+E_{a,j}-E_{a,i}-E_{b,j}.
$$

For a nonuniform swap proposal q, accept with

$$
\alpha=\min\{1,\exp(-\beta\Delta E)q(\pi'\to\pi)/q(\pi\to\pi')\}.
$$

Pi is the complete assignment permutation. Reverse proposal probabilities must be evaluated using the hypothetical post-swap holder map. Keep the existing stay option and Metropolis-Hastings correction in `gibbs_propose_one_replica` and `_gibbs_mh_acceptance_probability`. A heat-bath-like proposal over transpositions is not generally an exact full conditional over assignment permutations, so calling the method Gibbs does not justify deleting this correction.

### 4.1 Proposal eligibility

Every occupied state is a candidate for each selected carrier, with the existing numerical proposal convention and MH correction. There is no parent-only mask, geometry-neighbour mask, auxiliary-region hard wall or low-overlap exclusion in MVP Gibbs production. Poor overlap should make moves unlikely through the energy calculation, not through an unaccounted state-dependent filter.

The placement region describes where a bias was designed and initially placed. It does not define where z is allowed to be evaluated. The deployed model must have a globally defined potential over valid configurations admitted to the run.

Geometry graphs may still be generated for diagnostics. Do not let those graphs change Gibbs eligibility. A later restricted-exchange mode requires a separately derived proposal rule and detailed-balance tests.

### 4.2 Matrix assembly and updates

Evaluate z for every carrier, once per distinct active model per exchange barrier. Assemble all A_s columns/rows from those values and the state parameters. Ordinary states contribute zero auxiliary energy, but ordinary carriers contribute nonzero z values to active-state cross-evaluations.

During a sweep, coordinates stay fixed while assignments change. The complete E matrix remains valid; the holder and assignment maps must update after every accepted swap before the next proposal and reverse-proposal evaluation. Two state-parameter updates and the map change form one logical transaction. No integration or reporting may observe a half-updated swap. On a runtime error, abort the segment and recover from a committed checkpoint rather than continuing with inconsistent maps.

After the sweep, MD resumes under the newly assigned complete Hamiltonians. Never carry an auxiliary force with a configuration merely because it entered production through an auxiliary spawning route.

### 4.3 Duplicate states and sampling measure

Distinct state instances may share a Hamiltonian. They remain separate exchange slots. An unrestrained sham and its parent can have identical energies but different provenance and instance IDs. Extra duplicate slots deliberately allocate more sampling to that Hamiltonian; the matched-sham experiment controls for that allocation relative to active auxiliary bias.

Do not promise that every proposed parent–sham swap accepts under every proposal scheme: identical energies make the energy difference zero, but the applicable MH proposal ratio must still be evaluated. Report observed proposal and acceptance statistics.

For analysis, either keep duplicate instances as separate origins or combine exactly identical Hamiltonians while adding their sample counts. Record the mapping and verify that both approaches recover the same target weights. Never merge states merely because CV1/CV2 centres match.

## 5 State schema and provenance

Introduce a versioned extension of the existing canonical state schema. Proposed field names below are interfaces to implement, not claims that current main already accepts them.

| Field | Requirement |
|---|---|
| `state_instance_id` | Unique stable ID within the regime; mapped explicitly to runtime window index |
| `hamiltonian_sha256` | Hash of canonical physical state definition, excluding spawn provenance and slot ID |
| `center1`, `k1`, `center2`, `k2`, `gamd_lambda` | Existing state-specific parameters, preserved exactly |
| `aux_model_sha256` | Active model hash; null for an inactive canonical term |
| `aux_center` | Finite centre for positive strength; canonical zero when inactive |
| `aux_k_kcal_mol` | Explicit nonnegative strength in documented z units |
| `state_role` | `ordinary`, `auxiliary` or `sham`; diagnostic/protocol metadata, not a substitute for energy fields |
| `spawn_parent_state_id` | Provenance only; never consumed by exchange eligibility |
| `spawn_source_observation` | Run, segment, carrier, state, checkpoint and exact source step |
| `matched_additional_slot_id` | Stable W/B pairing of each proposed auxiliary state and its sham |

The phase manifest embeds or content-addresses all models, the physical-system identity, topology/atom-map hash, CV1/CV2 definitions, boost envelope, state table and ensemble settings. Canonical model payloads must include schemas and units. Hash coefficients and normalization, not a mutable filename.

Current `correctness/state_identity.py` rejects unknown window physics fields. Extend that whitelist and its canonicalisation deliberately; do not bypass the validator. Both state snapshot hashes and restart comparisons must cover the new fields. The phase identity includes the full registry of diagnostic models, even where an individual inactive Hamiltonian does not.

Legacy phases with no auxiliary feature retain their established definition and exchange-energy version. A new auxiliary-capable phase uses a new explicit energy/schema version, proposed `state_bias_matrix_v3_aux`. A sham arm running the same capability and diagnostic model may use this version with all strengths zero; this is more informative than requiring the old version string. Separate zero-feature regression tests establish compatibility with v2.

## 6 Spawning and admission

Auxiliary state creation occurs at a phase boundary, or in a new plain-run fork. Do not resize the occupied permutation while a retained production segment is running. A future live insertion kernel would require an additional statistical-mechanical design and is not part of this work.

The admission sequence is:

1. Freeze the proposed full next state table, model registry and sampling budget.
2. Resolve each original state's latest valid endpoint with the existing checkpoint-manifest and epoch-window-map logic. Do not use file modification times or glob order.
3. Create each additional carrier from a recorded source configuration. A parent is a convenient source, not an exchange attachment.
4. Apply ordinary minimisation, safety checks, constraint handling and initialisation. If needed, ramp the auxiliary restraint during a preparation segment only.
5. Freeze all target parameters and run explicit equilibration under the resulting regime. Exchange may operate during this excluded stage.
6. Audit exact Context parameters against the complete state table, assignments and model hashes.
7. Seal preparation/equilibration outputs, checkpoint all carriers and RNGs, then begin a new fixed-state production segment.

One Context is required per occupied state instance under the current architecture. Keep the existing GPU chunking and ownership rules. Do not allocate an independent scheduler or coordinate-copy loop for auxiliary carriers.

Use distinct integrator and barostat streams within a run, including additional carriers. The per-replica integrator-seed fix already merged into the pinned baseline must remain active. Paired W/B runs may share a documented seed mapping; independent seed pairs must have independent mappings. Preserve checkpointed RNG state on resume rather than reseeding it.

The original c10 proposal's 0.2 ns discard is a starting preparation allocation, not proof of stationarity. Thermodynamic eligibility also requires the prespecified stability and burn-in sensitivity checks in Section 13. Copying or rescuing coordinates after production starts invalidates continuous-carrier path accounting and the affected equilibrium eligibility unless handled by a separately approved protocol.

Keep experimental purpose separate from sampling eligibility. The strict exporter currently permits equilibrium eligibility only for `phase_kind = production`. A fixed, equilibrated retained segment of the pilot therefore has production phase kind plus separate experiment metadata identifying it as a pilot; preparation, pulling and unresolved equilibration remain ineligible. Do not relabel an unequilibrated segment merely to satisfy the exporter.

## 7 Observation and exchange records

At every thermodynamic sample, record enough information to evaluate every Hamiltonian in the intended analysis union. At every exchange barrier, record enough information to reconstruct the entire accepted assignment sequence.

The observation key is `(run_id, segment_id, carrier_id, absolute_step, observation_phase)`. Use a separate exchange attempt sequence number for multiple attempts at one step. Existing logs are sampled before exchange; preserve and document this ordering. Do not join rows on floating-point time or on step alone.

Required sample fields include state instance ID, CV1, CV2, model-indexed z, boost inputs, box/volume where needed, model/phase identities, and source eligibility. Use float64 for auxiliary values and offline cross-energy assembly initially. Any later lower-precision storage requires a reduced-energy error validation over the strongest allowed bias.

For future torsional models, retain the complete ordered torsion feature basis at every eligible thermodynamic sample, or matching lossless coordinates that pass reconstruction parity. A scalar z is enough to evaluate its frozen model but does not support arbitrary future models. Peptide coordinates cannot support later solvent-dependent models without solvent data.

Exchange records must retain: exact step and within-step order; selected carrier; old states and carrier pair; proposed states; stay/propose outcome; accepted/rejected status; delta energy; forward/reverse proposal probabilities or reproducible log values; energy/schema version; and assignment checksums at checkpoints. Preserve useful existing fields rather than replacing them with a second independent ledger.

The evaluation trajectory used for structural labels must contain frames on the declared exchange-boundary grid, associated with pre-exchange assignments and the matching coordinates. Align the reporting cadence with the exchange interval or write an explicit boundary stream. Do not interpolate a structural frame to an exchange timestamp.

Runtime force calculations remain authoritative for online dynamics. Offline feature computation must use the same angle, atom-map, imaging and unit conventions. Missing or nonfinite model values in a live unrestricted exchange matrix are fatal; offline missing rows require explicit exclusion and an eligibility audit.

## 8 MBAR and thermodynamic pooling

For this common-temperature, common-pressure, common-base-potential design, the reduced bias matrix is

$$
u^{\mathrm{bias}}_{ns}=\beta[G_s(x_n)+B_{1,s}(x_n)+B_{2,s}(x_n)+A_s(x_n)].
$$

Subtracting the common per-configuration physical and pV contribution is a valid gauge choice only if it is applied to every state and the physical target. In this representation the unboosted, unrestrained physical target has zero reduced bias. Do not combine a bias-only sampled-state matrix with a full-energy physical-target column.

Use exact auxiliary bias energies in MBAR; no auxiliary cumulant approximation is needed. This addition does not otherwise change the established treatment of GaMD or Pep-GaMD in the pipeline. The cross-state matrix, origin counts, eligibility mask and target definition must remain aligned after any filtering.

An ordinary-origin sample must be evaluated under active auxiliary states. Recording z only while a carrier occupies an auxiliary state is insufficient and blocks pooled analysis.

Historical c10 data without the necessary features are not automatically poolable with the new states. The default pilot union contains only complete, eligible pilot phases. Historical data remain available for separate analysis and provenance. They can enter a combined union only after exact-frame feature reconstruction and energy-parity validation; do not declare them irretrievably unusable if such coordinates exist, and do not silently fabricate missing features.

An exclusion fraction below 0.1 percent is only an audit threshold, not proof that missingness is harmless. Production should produce zero avoidable missing auxiliary observations. Report exclusions by origin state, structural group, z range, time block and carrier. State-dependent or structure-dependent failure can bias results even when numerically rare.

Zero-lambda samples still need auxiliary reweighting when their state has positive auxiliary strength. The lambda-zero versus full-ladder diagnostic must use complete reduced biases on both sides. Define a separate ordinary-lambda-zero crosscheck if the intention is a no-auxiliary reference. Never equate lambda zero with unbiased physical sampling.

## 9 Checkpoint and resume

The committed checkpoint bundle must bind the complete state table, model payloads and hashes, auxiliary Context parameters, assignments, all carrier checkpoints, random streams, phase identity, and the last committed sample/exchange boundaries.

On resume, rebuild the same model capability, load checkpoints, then verify observed Context parameters and assignments against the manifest. Missing models, changed coefficients, topology mismatch, unknown schema versions or incomplete carrier sets block resume. A requested change of model or state population begins a new phase; it is not a resume.

Crash tests must cover interruption before, during and after an accepted swap and during file flush. On recovery, truncate uncommitted tails through the existing segment machinery. Do not repair disagreement by guessing a state from its CV centres: duplicate Hamiltonians and auxiliary differences make such inference ambiguous.

## 10 Runtime cost and budgets

With K carriers and M distinct models, evaluate M × K auxiliary coordinates per exchange barrier and assemble the corresponding K × K bias entries. MVP has M = 1. Reuse z values; never launch one OpenMM energy evaluation for every state–carrier pair in routine production.

Per-step auxiliary force overhead may be present even in zero-strength states if the engine still evaluates the model. Measure this rather than assuming a zero coefficient eliminates all cost. Both arms use the same diagnostic feature recording, sampling cadence and hardware allocation. Record any unavoidable difference in their force graph.

Benchmark the actual multi-Context GPU workload, not just one isolated Context. Report node-hours, preparation hours, retained simulation time, aggregate ns/day, exchange/reporting cost and peak memory. Cap total states and node-hours before scheduling. No performance gain is asserted before measurement.

Adding auxiliary states creates an experimental compute allocation. Matched shams measure the bias benefit beyond adding equivalent extra carriers. A later adoption benchmark must also compare against the best ordinary allocation under the same node-hour budget, because the sham allocation itself may not be optimal.

## 11 Revised chignolin screening experiment

### 11.1 Arms and frozen inputs

Arm W contains the frozen 228 ordinary state instances and m additional active auxiliary states. Arm B contains those same ordinary instances and m matched sham instances. The existing candidate placement has m = 4, giving 232 instances per arm. MVP requires exactly four recovered and validated candidate rows for this historical pilot; if fewer are available, issue a new preregistration with the new replica count, budget and precision design before launch. Do not silently reuse the four-state power assumptions.

All added W states have lambda zero for this pilot. This is a pilot placement decision; the state schema and energy machinery support other lambdas after their force and exchange tests pass. Original states retain their existing lambda values. Shams copy the baseline CV1/CV2/lambda parameters of their matched W state and use zero auxiliary strength. W and B both permit unrestricted Gibbs-walk exchanges among all occupied states.

Historical placement results to recover from the authoritative full-precision artifact are:

| Spawn parent | Approximate auxiliary centre | Approximate k in kcal/mol per z squared |
|---|---:|---:|
| 120 | +1.49 | 1.82 |
| 117 | -1.92 | 1.20 |
| 9 | +1.59 | 2.40 |
| 198 | +1.29 | 2.16 |

These rounded numbers are for orientation, not executable parameters. Baseline restraints must come from the frozen state registry matched by exact identity, not from rounded copies of the earlier spec's table. Freeze the placement code, complete candidate ranking, selected rows, feature/evaluation models and data manifests. Missing inputs block this particular pilot; they do not block building and testing the generic machinery.

Retain the old model as a screening candidate without treating its forecast utility as established dynamical benefit. Forecast overlap and static label redistribution are proposals to test. The historical circular-shift null needs a documented series construction and exchangeability assumptions; a large shift alone does not prove an independent null. The historical results do not acquire confirmatory status by being written into a new spec.

### 11.2 Runs and starting structures

Run the screen on a separate fork of the current c10 endpoint set. Resolve each original state's latest endpoint through the committed assignment manifest and epoch window map. Every additional W/B slot uses its matched recorded source. W may require an auxiliary pull; B receives comparable preparation time under its own fixed sham Hamiltonian. Record preparation histories separately.

Use three independent seed pairs for the revised screen. Within a pair, W and B share the source configurations and documented seed mapping; seeds differ across pairs. Seed pairing is common-random-number experimental bookkeeping, not a claim that trajectories remain matched after exchange histories diverge. Both arms use the same code commit, physical force field, temperature, boost envelope, timestep and exchange policy.

Starting structures for confirmation must come from fresh independently evolved endpoints where feasible. A different phase of the same ancestral campaign is a sensitivity check, not proof of independent initial conditions. Report that dependence explicitly if it cannot be removed.

### 11.3 Feasibility before efficacy

First run the controlled physics tests and a separately labelled short engineering run. Use it to establish stable execution, full matrix parity, storage correctness, actual exchange exposure, approximate correlation scales and runtime cost. Do not use its W/B efficacy difference to change the endpoint or select a model and then include its observations in the efficacy test.

The original 3 ns retained length may be used as an engineering starting budget. It is not an adequate fixed statistical design merely because the population has 232 states. At 0.5 ns blocks it gives six non-overlapping blocks per run.

For efficacy, preregister a fixed core duration T after engineering calibration. Proposed minimum is at least 12 usable non-overlapping ensemble time blocks per run, with block duration at least 0.5 ns and at least three times the largest reliably estimated relevant correlation time. Relevant signals include structural labels, auxiliary occupancy episodes and return-count fluctuations. This is a practical floor, not a theorem guaranteeing valid confidence intervals. If correlations cannot be bounded, declare the precision design unresolved and budget longer runs or independent starts before efficacy launch.

Freeze the final core duration, block length, seed count, maximum total node-hours and analysis method before W/B efficacy outcomes are inspected. Use one terminal efficacy analysis. A run extension after that analysis is a new registered study unless a valid sequential analysis was already specified.

### 11.4 Equal time and compute

Match physical retained durations and state counts in the screen to make local transport rates interpretable, and measure the actual node-hours consumed. Do not assert that node-hours are simultaneously identical. The performance endpoint below uses the measured cost, so auxiliary overhead is included.

Charge preparation, integration, exchange, required recording and mandatory analysis consistently. Report production-only and total charged cost separately. Adoption uses total charged cost; asymptotic production cost can be a secondary projection, labelled as such.

For budgeting, the integrated length is `2 * J * (228 + m) * (preparation_ns + core_ns + guard_ns)`, plus any separate pulling and engineering runs. With three seed pairs, four additional states, 0.2 ns preparation, a 12 ns illustrative core and a 0.546 ns guard, this is approximately 17.74 microseconds. At the historical 3.6 microseconds per node-day it would require about 4.9 node-days before added force overhead and analysis. This is an illustration, not a selected run length or current throughput promise. Freeze a feasible budget after engineering calibration; the smaller engineering run remains useful even if a powered efficacy study is deferred.

## 12 Structural return endpoint

### 12.1 Ordinary and additional state sets

Let O denote the 228 original state instances. Let A denote the m additional instances: active auxiliary slots in W and sham slots in B. Define membership by frozen instance IDs, not by current numerical CV values or Hamiltonian equality. A sham is in A for exposure accounting even though its Hamiltonian matches an O state.

An episode starts when a carrier crosses from O into A and subsequently experiences positive MD time in A. The carrier can move among A states. The episode ends at its first subsequent crossing into any O state. No relationship to the spawning parent is required. A trajectory that begins the analysis core in A has a left-censored episode; report it but do not invent its entry label.

Several accepted swaps can occur at the same step in Gibbs-walk. Reconstruct their exact order, but state visits with zero MD residence are not bias exposure. For episode time accounting, use the assignment held during each following MD interval; collapse within-barrier zero-time round trips. Keep their raw exchange counts as a plumbing diagnostic.

### 12.2 Frozen structural labels

Use the complete frozen contact/H-bond evaluation pipeline from the historical pilot, if recoverable, to label exact-grid configurations. The torsional auxiliary model and evaluation features remain distinct. Freeze preprocessing, residualisation, topology, feature ordering and clustering, including the training set if the transform requires it. Neither arm may refit labels.

Record raw contact/H-bond descriptors as well as labels. A label is an operational structural partition, not automatically a metastable basin. Residualised labels can change when the conditioning CVs change; report actual raw-descriptor changes so apparent transport can be inspected for this failure mode. Any replacement evaluation partition requires a new preregistration.

### 12.3 A successful return

Proposed operational defaults use the 10.5 ps exchange-boundary grid of the historical c10 run. Verify that the new run actually uses this grid; otherwise translate all durations into a newly frozen integer-step schedule.

1. At entry, define the pre-label from the five boundary frames ending at entry. At least four must have the same evaluation label. The four intervening MD intervals must have been in O. If not, the episode has no qualified pre-label and cannot contribute a successful return; retain it in entry/failure accounting.
2. Permit up to 48 exchange intervals, 504 ps, from entry to first exit into O. This deadline is a proposed screening choice and must be frozen before efficacy. A later return remains diagnostically useful but is not a primary success.
3. At exit, inspect the five boundary frames starting at exit. The next four MD intervals, totalling 42 ps, must all be in O. Require at least four of these five labels to agree, and require the exit frame itself to have that post-label.
4. The post-label must differ from the qualified pre-label. Re-entry into A during the 42 ps confirmation interval makes the episode a primary non-success rather than censoring it.

This defines short persistence after return, not a claim of long-term basin commitment. Report sensitivity to longer persistence windows as prespecified secondary analyses; do not select whichever window makes W win.

Freeze a core interval [t0, t1] for eligible episode entries. Run both arms for an additional fixed 52 exchange intervals, 546 ps, beyond t1 so every core entry can be assessed through the maximum episode and confirmation window. Do not count guard-period entries in the primary numerator. Charge the guard period to compute cost. Episodes unfinished by their deadline are primary non-successes and reported as such; do not omit them selectively.

An infrastructure failure that destroys needed observations is different from a scientifically unsuccessful episode. It invalidates the affected run or requires a prespecified complete-core truncation common to the matched pair; it must never be converted to a negative structural result.

### 12.4 Primary quantities

For arm a of seed pair j, let C_aj be the number of successful core-entry episodes, T_j the core duration and m the fixed number of additional slots. Define

$$
R_{aj}=C_{aj}/(mT_j),\qquad Q_{aj}=C_{aj}/H_{aj}.
$$

R is useful returns per additional-slot nanosecond. Because exactly m carriers occupy A throughout production, mT is its total occupancy exposure. Longer excursions consume this time without automatically increasing event counts, unlike a per-excursion probability. Q is useful returns per charged node-hour, H, for the whole run under the common charging policy.

The paired contrasts are equal-weight means over seed pairs:

$$
\Delta_R=J^{-1}\sum_j(R_{Wj}-R_{Bj}),\qquad
\Delta_Q=J^{-1}\sum_j(Q_{Wj}-Q_{Bj}).
$$

Also report ratios of the arm means when the B mean is positive. Do not average unstable per-pair ratios. If B has zero successful returns, report ratios as undefined and use the difference intervals; do not introduce an arbitrary pseudocount to obtain a finite ratio.

R measures a local output of the additional-state subsystem; it is not diluted by unrelated ordinary-state transitions. Q measures the cost of producing that output. Neither alone proves global thermodynamic efficiency. A later adoption benchmark must assess physical-target estimator precision or reproducibility at equal total compute.

All entries, including unqualified pre-labels, deadline failures, re-entries and returns to the original label, are reported. There is no imputation of zero W/B effect for missing strata. Stratified results by source state/label, destination state/label and additional slot are descriptive unless support and weights were frozen in advance.

### 12.5 Supporting transport requirements

For a screen pass, successful W returns must include both directions of at least one pair of evaluation labels, with at least five successful events per direction in at least two seed pairs. This is a proposed minimum guard against one-way displacement; counts are correlated and do not replace uncertainty estimates.

Report the full directed label transition matrix and the distribution of raw contact/H-bond changes. Report which ordinary states receive returns and the survival of the new label after 0.1, 0.25 and 0.5 ns wherever the recorded follow-up permits. These are sampler diagnostics, not unbiased kinetics.

For each additional slot, report proposals, acceptances, positive-residence entries/exits, carrier diversity, duration distribution, deadline-failure fraction and pairwise MBAR overlap against ordinary states. The pilot no longer requires 200 direct-parent transitions. High acceptance or forced z switching never substitutes for successful structural returns.

Proposed exposure floors for interpreting a negative transport result are at least 100 positive-residence A entries per run, at least 50 with a qualified pre-label, at least 10 entries to each additional slot, and at least two qualified pre-labels with 20 entries each across the W seed pairs. These count thresholds are feasibility defaults to freeze or revise before efficacy, not independent-sample counts or replacements for interval precision. Failure of these floors means insufficient exposure. With the floors satisfied, failure of the bidirectional-success criterion is a transport failure, so it is not automatically hidden by the insufficient-evidence branch of the decision table.

A proposed overlap floor of 0.15 uses the pairwise MBAR overlap scale whose maximum for identical distributions is 0.5. Each auxiliary slot should have at least one supported ordinary-state partner at that scale; its best partner need not be its spawn parent. Lack of measurable overlap is inconclusive; well-measured overlap below the floor is a placement/transport failure, not an energy-correctness failure. Do not use diluted entries of the full 232-state overlap matrix for this threshold.

## 13 Statistical and thermodynamic decisions

### 13.1 Uncertainty procedure

Use seed pairs as the highest-level independent replication units. Within each run, use synchronized ensemble time blocks: one block contains the complete ensemble over that physical-time interval. Never resample carriers as though independent.

Extract and score episodes on the original continuous trajectories before resampling. Associate each complete outcome with its entry-time block. Do not concatenate trajectory blocks and run the episode parser across artificial boundaries. Block choice must account for the episode deadline plus confirmation interval and measured serial dependence; enlarge the nominal 0.5 ns minimum where those dependencies require it. The chosen length must leave the preregistered minimum number of usable blocks.

For the proposed hierarchical bootstrap, draw seed-pair IDs with replacement, then draw synchronized time blocks within each selected run. The entire ensemble block is the resampling object. For matched runs, resample matched time-block indices together to preserve the chosen pairing. Recompute counts and contrasts on each replicate. Each resampled run has the same number of equal-length core blocks as its original, so mT stays fixed. Hold that run's measured total H fixed for its conditional Q calculation; between-run cost variability enters through seed-pair replication. Preserve multiplicity when a seed pair or block is drawn more than once. Use at least 10,000 cheap return-count bootstrap draws, recording the seed. This is a proposed screening approximation and must pass coverage simulations before a confirmatory confidence claim.

Thermodynamic bootstrap requires new MBAR solves and has a separate prespecified computational budget, initially 500 replicates with an interval-stability check. Do not run 10,000 full MBAR solves by inheriting the cheap event-bootstrap setting. If 500 solves cannot stabilise an equivalence verdict, report the verdict as unresolved or preregister a larger analysis budget; never select a convenient random seed.

Report per-pair effects and intervals, pooled 90 percent intervals for the screen, the number of effective time blocks, and between-pair spread. With three seed pairs, the between-run tail distribution remains weakly identified. The screen is exploratory even if its interval is positive. A bootstrap cannot create independent experiments from one correlated history.

For confirmation, preregister at least five fresh seed pairs, final run lengths, a 95 percent interval method and a fixed terminal analysis. This is a stronger proposed design than the old three-pair confirmation; its actual precision must be checked against screening variability. If the required precision exceeds the available compute budget, do not label a smaller study confirmatory by changing terminology.

### 13.2 Efficacy and precision rules

Proposed screening success requires all of the following:

- Lower 90 percent confidence bounds for both Delta_R and Delta_Q exceed zero.
- The point improvement Delta_R is at least the frozen practical threshold delta_R = 0.10 R_ref, or the absolute threshold frozen for a near-zero reference. Ratios of observed arm means are descriptive, not a second changing practical threshold.
- At least two of three seed pairs have positive R and Q point differences, and no pair has a 90 percent upper confidence bound below minus 0.10 R_ref or minus 0.10 Q_ref for its respective paired difference.
- Bidirectional return transport and ordinary-partner overlap requirements pass.
- No unresolved implementation or thermodynamic-consistency gate remains.

The 10 percent reference-rate threshold is a proposed minimum practically useful improvement, not a prediction. Before launch define R_ref and Q_ref from independent engineering/sham data and require the 90 percent half-widths of Delta_R and Delta_Q to be at most 0.10 times their respective reference rates for a negative efficacy conclusion. The primary practical requirement for Q is a positive gain, delta_Q = 0, since its separate confidence bound already requires a demonstrated cost benefit. If reference rates cannot be estimated, resolve the duration/power design first; never use the observed efficacy contrast to manufacture a favourable precision threshold.

For zero-baseline counts, use absolute contrasts and prespecified absolute precision targets derived from feasibility. A positive noisy point estimate is not a pass. An effect below the practical target can be classified as insufficient practical gain only when its uncertainty excludes the target; otherwise it remains inconclusive.

Confirmation repeats the frozen endpoint and cost policy on new data, using the preregistered 95 percent criteria and practical threshold. Do not pool screening observations into the confirmatory verdict. Statistical success authorises the next bounded deployment stage, not a claim of improved folding or converged global equilibrium.

### 13.3 Thermodynamic consistency

Keep the original proposed margins as screening tolerances: 0.3 kT weighted RMS for CV1 and z PMF differences and 0.03 absolute difference per evaluation-group population. They are engineering policy values requiring justification, not universal accuracy standards.

Compare common physical-target estimates from complete bias matrices. Define bin edges and alignment weights before unblinding. Track both common-support disagreement and probability mass outside common support. Never drop newly discovered W-only regions and then call the remaining PMFs equivalent. Unsupported regions, unstable target ESS, or disagreement between starts make the gate inconclusive.

Use 90 percent confidence intervals for screening equivalence: signed population-difference intervals must lie inside plus/minus the margin; the upper bound for nonnegative PMF RMS must lie below its margin. A lower RMS bound above the margin or a population interval wholly beyond one margin indicates disagreement. Otherwise the result is inconclusive. Re-solve the required MBAR problems within bootstrap replicates; a fixed-weight histogram bootstrap understates estimator uncertainty.

Differences between W and B do not by themselves prove thermodynamic harm. W could relax an inherited trap faster, B could relax faster, or either implementation could be wrong. Classify such results as `thermodynamic_inconsistency` until force parity, cross-energy parity, stationarity and start-dependence checks identify the cause. A controlled-model distribution failure is an implementation/physics failure.

Require retained-window sensitivity analyses after additional leading discard, time-block trends, between-pair population comparisons and model-support diagnostics. Prespecify the discard variants; do not select the one producing agreement. A common apparent plateau in both arms is not proof of global convergence.

### 13.4 Exhaustive decision order

Evaluate the following ordered rules. Every gate has pass, fail or unavailable status; unavailable must not be coerced to pass. Preserve secondary failure reasons even when an earlier rule determines the headline result.

| Priority | Result | Condition and action |
|---:|---|---|
| 1 | `invalid` | Any state, force, exchange, barostat, identity, restart or observation integrity failure. Repair machinery; no efficacy interpretation. |
| 2 | `blocked_inputs` | Required frozen model, state table, evaluation artifact or analysis definition absent. Do not launch or issue a scientific verdict. |
| 3 | `thermodynamic_inconsistency` | A well-measured physical-target disagreement exceeds the margin. Diagnose equilibration versus implementation; no promotion. |
| 4 | `retune_transport` | Placement has well-measured inadequate ordinary-state overlap or repeated support/admission failure under correct machinery. New placement and new preregistration; keep this trial as a failure of that deployment. |
| 5 | `insufficient_evidence` | A mandatory gate is unavailable, thermodynamic equivalence is inconclusive, correlation/precision requirements fail, or the exposure floors in Section 12.5 fail. No automatic extension after the final look. |
| 6 | `screen_pass` | All screening efficacy, practical-gain, transport and consistency requirements pass. Proceed only to fresh confirmation. |
| 7 | `insufficient_practical_gain` | Adequate precision gives an upper Delta_R bound below delta_R or an upper Delta_Q bound at or below zero. Do not promote this deployment. |
| 8 | `transport_not_demonstrated` | Counts and precision are adequate but the prespecified bidirectional criterion fails. Do not promote; diagnose trapping or one-way displacement. |
| 9 | `insufficient_evidence` | All remaining combinations, including intervals crossing a required threshold. Record the precise unresolved conditions. |

The analysis function must return exactly one headline result for every valid combination, with a machine-readable list of all failed and unavailable gates. Test the table exhaustively. None of these results licenses the general conclusion that a CV can never be useful under another placement or strength.

## 14 Confirmation and deployment

A successful confirmation permits a bounded new adaptive-production regime, with the same state semantics and complete recording. It does not retroactively make historical incomplete data eligible. Start with the validated lambda-zero auxiliary placement; other rungs require their own full force/energy and transport validation.

The adaptive controller may propose additions at a phase boundary using existing discovery/placement evidence. A request contains explicit baseline restraints, an auxiliary model hash, centre, strength, source provenance and allowed lambda placement. Proposed `rungs` accepts an explicit list; defaulting to every rung is not allowed without an explicit policy choice and budget calculation. This pilot uses `[0.0]`.

Resolve the new state population under the existing replica and node-hour cap. Record whether ordinary allocations are retained, replaced or reduced. Re-equilibrate after a population/model change. Every production segment keeps an immutable state registry. No dynamic retuning of auxiliary centres or coefficients occurs inside retained segments.

For eventual adoption, compare physical-target estimator uncertainty or between-campaign reproducibility at equal total compute, with the best supported ordinary-state allocation as baseline. Report reference-free structural coverage and start dependence. Native RMSD and folded fractions are post-decision readouts only and never guide model selection, placement or promotion.

## 15 Repository implementation map

The following entry points were inspected at the pinned commit. Exact call chains must be rechecked if implementation starts from a newer main. New auxiliary module names below are proposed.

| Area | Current entry points | Required change |
|---|---|---|
| State loading | `gareus/windows.py` explicit-window loader | Parse and validate auxiliary columns; preserve state-instance identity and duplicates |
| State identity | `gareus/correctness/state_identity.py`, `normalize_windows` in `correctness/bias.py` | Version schema, canonicalise inactive terms, hash active model/parameters, update validators |
| Model runtime | Proposed `gareus/auxiliary_cv/model.py` and `features.py` | Frozen model loading, topology/schema checks, online/offline parity |
| Force deployment | `gareus/forces.py`, production setup, Pep-GaMD integration | Build exact scalar-square force, audit force groups and full-strength inclusion |
| State application | Existing `set_window` call chain reached by production swaps | Apply and clear all auxiliary parameters in every state transition and on resume |
| Exchange matrix | `_current_exchange_arrays` and matrix assembly in `gareus/production.py` | Add full active-state auxiliary cross-energies for every carrier |
| Gibbs kernel | `_gibbs_window_proposal_distribution`, `gibbs_propose_one_replica` | Preserve unrestricted candidates and reverse-proposal correction; consume complete matrix |
| Storage | `gareus/store.py`, sample/exchange writer call sites | Versioned auxiliary/features records, boundary observations and ordered events |
| Offline energies | `gareus/correctness/bias.py`, MBAR loaders/ladder pipeline | Evaluate model-indexed auxiliary terms with strict completeness |
| Diagnostics | `mbar_analysis/ladder_overlap.py`, `crosscheck.py`, report consumers | Distinguish auxiliary state role from lambda; report pairwise-scale overlaps and truthful unavailable states |
| NPT | `gareus/npt.py`, `npt_driver.py` and energy callbacks | Include auxiliary energy at current and proposed volume; preserve Context ownership |
| Resume | `gareus/checkpoints.py`, production checkpoint helpers, segment registry | Bind models, state table, assignment and RNG state atomically |
| Pilot analysis | Proposed `tools/analyze_auxiliary_pilot.py` | Episode parser, fixed-horizon outcomes, cost rates, uncertainty and total decision function |
| Adaptive entry | `gareus/adaptive_production.py` and admission/continuation helpers | Boundary-only additions with explicit lambda list and resource caps |

Current `ParquetSampleWriter.write_sample` has fixed fields and does not accept model-indexed auxiliary values. Merely adding a CSV column to the explicit table cannot complete this feature. Current strict bias reconstruction returns after zero-lambda handling; auxiliary terms must be included before any such early return so lambda-zero auxiliary states are not silently unrestrained in analysis.

Geometry consumers identify states using restrained CV1/CV2 centres and lambda. They can confuse a duplicate or auxiliary state with an ordinary counterpart. Audit `other_rung_same_centre`, top-up partners, ladder grouping and continuation maps. MVP may explicitly exclude auxiliary instances from a diagnostic that cannot yet represent them, but the report must say unavailable and the pilot must not treat a skipped mandatory gate as a pass.

## 16 Implementation stages and release gates

### Stage A Immutable definitions and exact evaluators

Implement the model registry, state schema extension, canonical identity and shared auxiliary energy evaluator. Add analytic gradients and an OpenMM force expression. Deliver a pure offline reconstruction path and an independently evaluated numerical reference. Exit only after zero-strength, topology, sign, units and gradient tests pass.

### Stage B Production Hamiltonians and unrestricted exchange

Wire the exact term into every Context, complete state application, exchange matrices, sampling and NPT. Preserve the existing Gibbs proposal semantics. Add canonical direct-energy tests and exact small-permutation invariance tests. No chignolin efficacy run is authorised by this stage alone.

### Stage C Storage and resume

Add complete thermodynamic observations, ordered event identities, model-aware manifests and restart binding. Verify interrupted/restarted trajectories and analysis against uninterrupted controls. Add strict MBAR pooling and legacy compatibility tests. Exit only after no active auxiliary term can disappear through a writer, loader, early return or state-hash path.

### Stage D Pilot admission and analysis

Recover and freeze the actual c10 artifacts. Implement matched additional slots through ordinary admission. Implement the endpoint and decision function before efficacy data are generated. Validate the parser against synthetic event traces and adversarial stochastic examples. Run engineering probes to freeze a feasible statistical and resource budget.

### Stage E Screening and independent confirmation

Execute preregistered runs and publish complete outputs, including unsuccessful episodes and all gate statuses. A screen pass advances to the separate confirmation. A confirmation pass advances to a bounded adaptive integration proposal and equal-compute thermodynamic benchmark.

### Stage F Optional regional models

Only after the single-model implementation is validated, support more than one auxiliary model per phase. Every carrier must evaluate every active model needed for unrestricted cross-state proposals; record model-indexed values and all required feature bases. Different regions may use different models without a Cartesian grid. Runtime and analysis must reject an unknown model rather than substitute the current state's scalar z.

## 17 Required validation cases

Tests below address silent physics and interpretation failures, rather than merely mirroring the implementation.

| Test | Required outcome |
|---|---|
| Zero auxiliary strength on every state | Legacy forces, cross-energies, sampled target and existing output behaviour agree within declared precision; extra diagnostic fields do not change physics |
| Active to ordinary to active assignment | Auxiliary force is cleared and reinstated exactly; no previous state's centre/strength remains |
| Scalar sum versus sum of squares | Analytic mixed-feature example confirms cross terms of the intended squared scalar |
| Angle sign and wrapping | Online/offline z agrees on positive/negative angles and across periodic boundaries |
| Model parity on c10 frames | At least 1,000 representative frames across all available phases and all four source parents; max absolute z error at most 1e-5, plus energy-scaled tolerance below |
| Gradient check | Auxiliary forces agree with finite differences at ordinary and extreme supported z values; use combined absolute/relative force tolerances fixed for platform precision |
| Force-group composition | Physical, boost and auxiliary forces reconstruct the intended total; auxiliary term is neither boosted nor counted twice |
| Direct cross-energy check | All four swap energies from assembled matrix agree with direct Context evaluations after complete state assignment |
| Reduced auxiliary-energy precision | Proposed FP64 oracle tolerance 1e-6 in reduced auxiliary energy; production mixed-precision tolerance initially 1e-4, validated empirically before enabling that platform |
| Ordinary-origin samples | Their auxiliary cross-energies are present and correct despite current auxiliary strength zero |
| Lambda-zero early return | Active auxiliary terms survive offline reconstruction when all sampled lambdas are zero |
| Exact Gibbs permutations | Enumerate 3 or 4 state assignment permutations with ordinary, auxiliary and duplicate states; each single-carrier MH kernel satisfies detailed balance, and the composed sweep preserves the target distribution |
| Global candidates | A worker exchanges with a non-parent ordinary state under the complete energy matrix; no neighbour mask is imposed |
| Duplicate Hamiltonians | Separate-slot and correctly merged MBAR analyses agree; state instances remain distinguishable in exchange logs |
| NPT trial | Direct energy difference includes A at the trial geometry; controlled distribution agrees with a trusted reference |
| Missing feature or changed model | Resume/analysis fails closed; never zero-fills an active cross-energy |
| Multiple swaps at one timestamp | Episode parser preserves order and counts zero MD exposure correctly |
| Auxiliary to auxiliary to ordinary | One episode; destination can be any ordinary state |
| Re-entry during confirmation | Primary non-success, not dropped or credited twice |
| Long residence with unchanged dynamics | A synthetic two-state process cannot gain primary rate merely from higher per-episode change probability |
| Initial auxiliary occupant or late unfinished visit | Correct left-censor/deadline accounting and unchanged declared exposure denominator |
| Bootstrap repeated draws | Repeated seed pairs and blocks retain their multiplicity; no set-based deduplication |
| Exhaustive decision combinations | Exactly one result for every gate combination, with all reasons retained |
| Two-dimensional hidden-barrier model | Recover known equilibrium populations while separately measuring useful return rates and their cost |
| Fast spectator and scalar-degenerate models | Coordinate motion/classification alone cannot satisfy structural transport plus thermodynamic validation |

For full-potential direct checks, do not subtract two large noisy energies and interpret cancellation error as auxiliary physics failure. Compare isolated auxiliary terms and complete exchange differences with precision-aware tolerances. A tolerance may be loosened only after a documented numerical-error study, not to make a failing model pass.

The stationary assignment law in the exact permutation test is proportional to exp[-beta times the sum of state energies over carriers]. Individual reversible updates can compose to a stationary sweep that is not itself reversible; test sweep invariance rather than imposing the wrong detailed-balance criterion on a deterministic composition.

## 18 Deliverables and reproducibility

The implementation handoff must include:

- Versioned model and state schemas with migration and strict validation.
- Full immutable model payloads and feature schemas; no abbreviated hashes in executable manifests.
- W/B state tables, source resolver output, seed mappings, run definitions and per-Context parameter audit.
- Shared runtime/offline bias construction and direct-energy verification reports.
- Exact Gibbs-permutation, force, NPT and controlled-distribution tests.
- Complete sampling, exchange and checkpoint schemas and reconstruction examples.
- Frozen evaluation model, placement code/results, episode analysis and decision function.
- `pilot_prereg.json` containing all hashes, duration/cost limits, primary endpoints, confidence method, practical thresholds, precision targets, block rule and confirmatory separation.
- Engineering benchmark, screening report, machine-readable gate results and any later independent confirmation report.

Keep code and compact scientific metadata in the repository. Large trajectories or sensitive/expensive raw artifacts may be in durable external storage, but a manifest must identify exact immutable objects, checksums and retrieval instructions. A gitignored local path alone is not a reproducible dependency.

Before efficacy launch, the preregistration validator must reject unset values, unknown state IDs, incomplete model payloads, mismatched arm state counts, budgets inconsistent with guard periods, insufficient planned blocks, or an analysis script whose hash differs from the frozen version.

## 19 Acceptance checklist

The generic auxiliary-state feature is ready for experimental production only when the Hamiltonian, exchange, storage, NPT, restart and controlled-target tests pass. The chignolin pilot is ready only when its full historical inputs are recovered and its prospective analysis and budget are frozen. Auxiliary sampling is ready for bounded adoption only after independent confirmation and a supported assessment of equilibrium estimator performance at equal cost.

The essential invariant is simple: a configuration always evolves under the complete Hamiltonian of its current assigned state, and every proposed reassignment is evaluated using those same complete Hamiltonians. Spawning history never replaces energy evaluation or imposes an exchange restriction.

## 20 Sources and scope of evidence

Repository sources are pinned to `c56b4b333a9aedfa1a4b4da62716a0620b6f26c8`:

- [Original c10 pilot v2](https://github.com/sulcjo/ATLaS-MD/blob/c56b4b333a9aedfa1a4b4da62716a0620b6f26c8/docs/superpowers/specs/2026-10-07-c10-z3-worker-pilot.md).
- [Adaptive auxiliary CV architecture](https://github.com/sulcjo/ATLaS-MD/blob/c56b4b333a9aedfa1a4b4da62716a0620b6f26c8/docs/superpowers/specs/2026-09-28-adaptive-auxiliary-cv-ladder-spec.md).
- [Production and Gibbs exchange](https://github.com/sulcjo/ATLaS-MD/blob/c56b4b333a9aedfa1a4b4da62716a0620b6f26c8/gareus/production.py), [window loading and graphs](https://github.com/sulcjo/ATLaS-MD/blob/c56b4b333a9aedfa1a4b4da62716a0620b6f26c8/gareus/windows.py).
- [Canonical state identity](https://github.com/sulcjo/ATLaS-MD/blob/c56b4b333a9aedfa1a4b4da62716a0620b6f26c8/gareus/correctness/state_identity.py), [strict bias reconstruction](https://github.com/sulcjo/ATLaS-MD/blob/c56b4b333a9aedfa1a4b4da62716a0620b6f26c8/gareus/correctness/bias.py), [sample and exchange storage](https://github.com/sulcjo/ATLaS-MD/blob/c56b4b333a9aedfa1a4b4da62716a0620b6f26c8/gareus/store.py).
- [Pre-exchange observation ordering tests](https://github.com/sulcjo/ATLaS-MD/blob/c56b4b333a9aedfa1a4b4da62716a0620b6f26c8/tests/test_sample_before_exchange_ordering.py).
- Shirts and Chodera, *Statistically optimal analysis of samples from multiple equilibrium states*, J. Chem. Phys. 129, 124105 (2008), [doi:10.1063/1.2978177](https://doi.org/10.1063/1.2978177). Supports equilibrium reweighting and the need to handle correlated observations; it does not validate this pilot's endpoint or thresholds.
- Chodera, *A simple method for automated equilibration detection in molecular simulations*, J. Chem. Theory Comput. 12, 1799–1805 (2016), [doi:10.1021/acs.jctc.5b00784](https://doi.org/10.1021/acs.jctc.5b00784). Supports explicit treatment of initial relaxation; no automatic detector guarantees escape from unseen metastable states.

The separate `2026-10-07-z3-auxiliary-umbrella-design.md` dependency named by the old pilot was not retrievable at its stated branch during this review. This specification makes the required contracts explicit and does not treat that missing dependency as implemented. Historical placement values and partition statistics require their original artifacts before reproduction or execution. No peptide efficacy, new implementation, or production speedup is claimed by this design.
