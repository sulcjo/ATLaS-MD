# Structural guidance for efficient conformational exploration

Date: 2026-09-27  
Status: proposed specification; adversarial code review complete, implementation and MD validation pending  
Reviewed repository baseline: `9aa536cc5ef8f07eebe6218d762a496afd01d913`  
Scope: equilibrium thermodynamics and conformational coverage at fixed total computational cost.

## 1. Goal and scientific boundaries

Improve the standard GENPEPT -> swarm -> CV selection -> adaptive epochs -> optional current-regime top-ups -> MBAR workflow without adding umbrella dimensions. Preserve the two spatial CVs and existing Pep-GaMD lambda ladder.

Add structural observations and diagnostics that distinguish molecular families hidden by the current CV projection. Use these first for offline reports and shadow recommendations, then for validated discovery seed selection and recommendations to the existing controller.

Success means more accurate and reproducible equilibrium observables per total wall-clock/GPU cost. Correct physical kinetics is not an objective. Structural transitions are evidence about mixing, not physical rate estimates. Neither cluster counts nor replica round trips certify equilibrium or complete exploration.

Non-goals for the first release:

- No new bias force, extra umbrella axis, replacement exchange kernel, or automatic CV refit.
- No replacement of top-up continuation states with novel conformers.
- No second scheduler competing with existing allocation and budget accounting.
- No native structure, native-contact target, folded labels, or post hoc native-RMSD selection.
- No REST2 implementation or persistent-homology force.
- No declaration that helper-cpus or Bayesian specifications are already implemented.

## 2. Existing architecture and incremental changes

Paths below describe the reviewed baseline; recheck interfaces before implementation.

| Existing component | Present behaviour | Proposed extension |
| --- | --- | --- |
| `gareus/swarm/stratify.py` | CV1/Rg/end-to-end strata and balanced seed draws | Structural diversity within existing strata |
| `gareus/swarm/seeds.py` | Nearby-CV1 frames with preference for distinct swarm members | Residue-labelled fingerprint diversity and original-seed ancestry |
| `gareus/swarm/members.py` | Scalar traces, torsion features, sparse seed structures | Reuse synchronized snapshots for additional descriptors |
| `gareus/cv_selection/select_pair.py` | Residual CV2 selection, structural partitions, information gain, predictability and coupling checks | Independent structural validation of the selected pair |
| `gareus/cv_selection/coverage.py` | Supported CV1 intervals, including rare regions | Preserve those protections; add a separate structural inventory |
| `gareus/production.py` | Fast CV observations, optional coordinate observations/trajectories, state-label exchange | Sparse immutable diagnostic snapshots without extra per-step force work |
| `gareus/store.py` | Scalar samples, exchange journals and segment provenance | Companion structural tables with verified frame joins |
| `gareus/adaptive/union_diagnostics.py` | MBAR uncertainty/overlap and local reduced-bias split-half tests | Structural population and seed-dependence diagnostics |
| `gareus/adaptive/topup_allocator.py` | Deficit-driven, wall-costed allocation with spatial/rung partners | Consume separately typed structural recommendations after validation |
| `gareus/topup_seeding.py` and production top-up seeding | Parent positions/velocities/box restoration and restraint checks | Preserve continuation semantics |
| `gareus/mbar_analysis/ladder.py` | Common ladder-energy reconstruction | Reuse target weights; never apply boost correction twice |

The allocator's existing `structural_edges` means persistent thermodynamic-state overlap problems. It does not mean molecular contact topology. New records must use distinct names such as `conformation_family` and `hidden_family_split`.

Related designs:

- [Bayesian inference and sampling design](../bayesian-sampling/spec.md)
- [Helper CPUs and durability](../helper-cpus/spec.md)
- [Effective top-ups](../2026-09-24-effective-topups-design.md)

## 3. Structural observations

### 3.1 Initial descriptor panel

Use deterministic, versioned, native-blind features:

1. Residue-labelled nonlocal contact fingerprints, with declared atom selection, sequence exclusions and smooth distance definition.
2. Backbone hydrogen-bond patterns with declared donor/acceptor and geometric criteria.
3. Backbone sin/cos torsions and turn descriptors.
4. Optional selected side-chain torsions and peptide-bond omega descriptors where supported by the actual topology.

Retain residue identity and signed torsion information. A scalar contact count or unlabelled graph summary can merge different arrangements. Descriptor availability must be explicit: backbone-only GENPEPT seeds cannot silently supply side-chain features. Structural comparison across preparation stages must use compatible features or separate schemas.

Hydration descriptors are deferred. Solute-only trajectories do not contain the needed solvent information.

### 3.2 Snapshot and provenance contract

Each observation identifies the campaign, segment, checkpoint/output generation where available, replica/walker, integer integration step, state assignment at the observation event, Hamiltonian identity, topology/atom mapping, descriptor version and source lineage.

Document whether an observation is before or after exchange at the same step. Resolve its state using the matching assignment revision/event order, never the latest assignment. Replica IDs and state IDs are different objects.

Persist coordinates or sufficient descriptor inputs with the periodic box and atom selection when needed. Reconstruct a whole peptide consistently under PBC before intramolecular geometry calculations; declare distance conventions.

The structural table is a companion to scalar samples. Join using verified identities, never row number or nearest timestamp. Structural snapshots may have a different cadence from energy samples. If reweighting is requested, require the matching reduced-energy information and valid counts/weights for the actual selected dataset; do not blindly reuse full-table normalization on an arbitrary subset.

Resume/replay must not duplicate accepted observations. Observations beyond the valid checkpoint boundary are excluded consistently with scalar samples. Missing, stale, corrupt or ambiguous observations produce an explicit unresolved diagnostic, not a successful convergence status.

### 3.3 Performance and helper ownership

Production's fast CV path avoids coordinate copies. Additional CPU capacity does not remove GPU synchronization or transfer cost.

Start with sparse peptide snapshots and reuse existing reads where event timing agrees. Simulation workers own OpenMM contexts; helpers receive immutable arrays or sealed artifacts and must not concurrently read a context. Bound queues, memory and helper concurrency; preserve CPU capacity for simulation workers.

Record snapshot extraction time, transfer cost, helper CPU time, queue lag, missing diagnostic observations and throughput impact. A declared overload policy may omit diagnostic snapshots and record the gaps; it must never silently omit scientific samples. Decisions requiring incomplete observations abstain.

Until an implemented durable snapshot interface is verified, use offline analysis of sealed exports. Do not infer live transaction safety from a specification alone.

## 4. Structural inventory and diagnostics

Build contact/turn families from development data using a deterministic, versioned representation and metric. Control feature scaling and redundant-feature domination. Freeze the evaluation panel and partitions before confirmation.

Report:

- Structural families within CV neighbourhoods.
- Population estimates under a declared common target when eligible.
- Dependence on original seed ancestry.
- Family visits and persistent changes along replica coordinate histories.
- Independent block/campaign contributions, unknown support and out-of-distribution frames.
- Sensitivity to partition resolution and observation cadence.

A family is an operational partition, not automatically a metastable state. A constant family trace does not establish zero autocorrelation. Unseen families have unknown mass, not zero inferred uncertainty.

The present reduced-bias drift and overlap checks can miss a family split when both families have similar CVs and boost energy channels. This new layer complements those checks and cannot override their failures.

### 4.1 Correct comparisons under the lambda ladder

For a pure umbrella depending only on z, conditioning on exact z removes the umbrella term. With the additional boost:

```text
p_k(x | z) proportional to p_0(x | z) * exp[-beta * DeltaV_lambda_k(x)]
```

Different lambda states can therefore legitimately have different family mixtures at the same CV values. Finite CV bins also retain umbrella variation within the bin.

Compare independent data under the same complete Hamiltonian, or reweight to a common target with support and dependence checks. Never interpret raw across-lambda occupancy differences as trapping. Use the existing exact ladder accounting, without a second GaMD correction.

State-label exchange leaves replica coordinates unchanged. Count a conformational transition from molecular features along the coordinate history, not from state-label changes. Split traces at resets, reseeding and discontinuous reconstruction. Do not interpret these accelerated-history counts as unbiased kinetics.

### 4.2 Dependence and uncertainty

Exchange-connected replicas are not independent campaigns. Different swarm members can share seed ancestry. Preserve ancestry and synchronized ensemble blocks; follow the Bayesian specification's dependence-aware uncertainty design where applicable.

Disagreement is diagnostic evidence, not proof of its cause. Possible causes include insufficient sampling, inadequate support, descriptor ambiguity and legitimate Hamiltonian differences. Reports distinguish these cases and abstain when they cannot.

## 5. Diversity-aware discovery seeding

Extend existing strata instead of replacing them with a global diversity objective:

1. Keep the current supported-region inventory and rare-region protections.
2. Within a stratum/target neighbourhood, identify compatible structural families.
3. Prefer representatives with distinct original ancestry and structural fingerprints.
4. Apply existing physical preparation and quality checks.
5. Re-evaluate diversity after grafting, relaxation and pulling.
6. Record collapsed families, preparation failures and regions without usable representatives.

Do not rank diversity solely by global RMSD, extension, rarity or low potential energy. These can favour extended/strained structures, misrepresent free energy, or suppress rare compact alternatives. No native supervision is introduced. Equal seed quotas correct the experimental design, not equilibrium probabilities.

Retain baseline exploration allocation so the selector cannot permanently starve poorly represented regions. Quotas, distance thresholds and budget fractions are development parameters to calibrate; this specification does not invent universal values.

## 6. Scheduling and phase semantics

Initial output is shadow-only. Structural diagnostics emit typed recommendations to the existing controller, with evidence, uncertainty, scope and expected cost:

- `continue_existing_state`
- `investigate_hidden_family_split`
- `propose_discovery_branch`
- `retain_unresolved_coverage`

The controller remains the sole owner of budgets and persisted decisions. Any accepted action must be idempotent on resume and respect spatial/rung partner and final-reserve policies.

Do not insert novelty into the existing sigma model as if it predicted uncertainty reduction. The allocator's inverse-square-root approximation does not estimate the time to discover a missing basin. Repeated lack of structural improvement should lead to reassessment or unresolved status, with bounded expenditure.

### 6.1 Top-ups

Top-ups continue parent exported positions, velocities and box under matching state/restraint semantics. Preserve existing fail-closed mapping and seed checks.

A novel representative cannot silently replace a top-up start. A discovery branch or new production initialization has its own provenance, equilibration treatment and budget charge. Exploratory branch data are not automatically eligible equilibrium MBAR evidence.

### 6.2 Confirmation

Freeze CVs, Hamiltonians, evaluation definitions and confirmation length before the primary confirmation phase. Outcome-dependent exploration and stopping require separate calibration; decision logging alone does not repair selection bias.

Use fresh, appropriately equilibrated fixed-design confirmation evidence for the primary comparison. Pooling arbitrary equal numbers of trapped families does not determine their relative equilibrium masses. MBAR cannot restore absent structural support.

## 7. CV and Hamiltonian changes

The selected pair already uses structural partitions and residual-component scoring. First test its hidden-family failures using independent descriptors.

If needed, compare alternative CV2 candidates in separate matched-cost pilots. Judge equilibrium observable error and between-campaign agreement per cost, with force stability and evaluation overhead. Preserve native-blind selection and freeze the deployed pair. Changed CV definitions require new state identities and valid cross-evaluation; never relabel old scalar samples with a new model.

Existing tICA code avoids cross-file lag pairs. Replica histories still experience changing Hamiltonians; equilibrium frame weights do not automatically recover unbiased kinetic correlations. Kinetic optimality is not the acceptance objective here.

Persistent homology is an optional offline ablation after contact fingerprints. Require incremental diagnostic or sampling benefit at measured cost. Distance-only topology lacks chirality and geometric loops are not necessarily knots or slow basins.

REST2 is deferred to a separate specification. It requires a different Hamiltonian decomposition and corresponding force, exchange, NPT, checkpoint and analysis validation. Existing Pep-GaMD energy channels are not a drop-in representation of REST2.

## 8. Delivery sequence and promotion gates

1. **Offline observation adapter:** exact joins, PBC/topology conventions, replay exclusions, schema and coverage reports.
2. **Offline structural diagnostics:** frozen descriptor panel, conditional comparisons, lineage-aware checks and adversarial fixtures.
3. **Discovery seeding extension:** within-stratum diversity and post-preparation verification; matched-budget ablation.
4. **Shadow controller integration:** recommendations and cost accounting without runtime actions.
5. **Optional controlled actions:** only after benchmark improvement and calibrated failure behaviour; separate discovery branches retain separate semantics.

Prefer small dedicated structural-analysis modules and pure decision helpers over expanding production/adaptive monoliths. Exact module names and CLI/config fields belong in the implementation plan.

Existing tests such as `tests/test_topup_seeding.py`, `tests/test_topup_allocator.py`, `tests/test_swarm_seeds.py`, `tests/test_swarm_stratify.py`, `tests/test_residual_kernel_resume.py` and `tests/test_fast_cv_path.py` are regression anchors, not proof of the proposed functionality.

## 9. Adversarial review rulings

| Failure mode | Required protection |
| --- | --- |
| Current CV/energy diagnostics pass while a structural family is absent | Report seed dependence or unresolved support; never certify complete exploration |
| Across-lambda mixtures differ legitimately | Compare a common target or identical Hamiltonians |
| Multiple descendants masquerade as independent evidence | Track original ancestry and exchange coupling |
| Diversity favours extended states | Select within existing strata and preserve rare compact support |
| Preparation erases diversity | Verify after preparation, not only in the bank |
| New conformers enter a top-up | Reject substitution; separate branch/initialization semantics |
| More descriptors negate fast observation performance | Sparse immutable snapshots and measured total overhead |
| A helper uses stale or mismatched frames | Exact event identities, sealed inputs and abstention |
| A second scheduler overspends or replays actions | One existing controller and budget ledger |
| Novelty count improves while thermodynamics worsens | Independent frozen equilibrium evaluation panel |
| New CVs invalidate historical cross-energies | New state identity and explicit evaluability checks |
| An apparent topology signal is only descriptor artefact | Resolution/cadence/PBC controls and simpler-feature ablations |

These are design findings from source inspection, not claims that current MD correctness or the new method has been experimentally validated.

## 10. Benchmark and acceptance

### 10.1 Adversarial synthetic cases

Use known equilibrium targets where possible:

| Case | Required result |
| --- | --- |
| Two families with nearly identical CVs/boost channels and poor interconversion | Existing scalar checks may pass; structural layer reports unresolved family sampling |
| Equilibrated states with different lambda-dependent family populations | No false trapping claim from raw occupancy differences |
| Many clones/descendants of one seed | No independent-campaign claim or artificial confidence gain |
| Rare compact family among many extended structures | Family remains eligible for discovery; no implicit zero mass |
| Seed families collapse during pulling | Final-start diversity report records the collapse |
| Restart/replay and mismatched frame/energy cadence | No duplicate or approximate joins; ambiguity abstains |
| PBC-wrapped peptide and permuted atom mapping | Invariant descriptors under valid remapping; invalid mapping fails |
| Rapid feature flicker without persistent rearrangement | No inflated claim of independent basin transitions |
| Helper backlog or missing snapshots | Explicit incomplete diagnostic and preserved simulation records |

### 10.2 Real-workload ablation

Use the target four-L40S, 192-logical-CPU allocation with the same context cap and physical model. Pin input structures, configuration, software and hardware metadata. Repeat across independently initialized campaigns, deliberately varying seed families and RNG streams.

Compare incremental arms:

A. Current workflow.  
B. Structural diagnostics only.  
C. B plus within-stratum diversity selection.  
D. C plus validated structural allocation/discovery policy.  
E. Optional persistent-homology descriptors against C/D, only after simpler features establish value.

Count GENPEPT/swarm, preparation, equilibration, production, analysis and helper overhead in total cost. Record throughput as a secondary metric. Keep umbrella dimensionality fixed and account for contexts diverted to discovery.

Primary endpoints are predeclared equilibrium observable error against a justified reference where available, calibrated uncertainty, and agreement across independent campaigns at equal total cost. Without a trustworthy reference, report reproducibility/precision improvement separately from unproven accuracy.

Secondary endpoints include time to independently revisit structural families, seed dependence within comparable ensembles, target-weight support, effective independent-block contributions, and end-to-end throughput.

Freeze evaluation descriptors independently of the scheduler and assess clustering-resolution sensitivity. Do not let an arm claim success merely by subdividing its own clusters.

Specify practical improvement and overhead margins before confirmation, based on development variance and scientific tolerances. Promote only if the method improves the primary cost/quality objective without degrading coverage, existing thermodynamic gates or restart correctness. Report all failed/abstaining campaigns. No speedup, complete-coverage guarantee or universal default is claimed before these benchmarks.
