# Side-chain-aware local basin discovery and generated z3

Status: proposed addition to PR #141; specification only, not implemented or validated.
Date: 2026-10-10.
Reviewed branch: `feat/cvaux-stage-a`, commit `a3da4eed26e13fc45874d7719f6028f562313aff`.

## 1. Decision and scope

Extend adaptive CVaux discovery so a generated scalar z3 can distinguish side-chain packing populations that co-occur at similar CV1/CV2 values. Keep the existing two umbrella coordinates, one admitted auxiliary model per campaign, at most the configured number of workers, and unrestricted exchange with ordinary states. This is not a third Cartesian umbrella grid and does not add REST2 or change Pep-GaMD.

Compare three feature families against the same contact/H-bond partition: backbone-only, side-chain-only, and mixed backbone/side-chain. Select one winning model. A side-chain model replaces the backbone-only choice when selected; simultaneous independently biased backbone and side-chain models are out of scope.

Discovery is native-blind: no native contacts, RMSD, folded labels, or residue selection based on the known chignolin fold. Equilibrium populations and free energies remain the objective; kinetics are not claimed. Better classification is evidence of unresolved structure, not proof of faster transitions, ergodicity, or faster folding.

Related documents:

- [Adaptive discovery design](2026-10-09-cvaux-adaptive-discovery-design.md), especially U12/U13 and the one-model rule.
- [Correctness repair specification](2026-10-10-cvaux-correctness-repair-spec.md).
- [B/C/D repair scope](../plans/2026-10-10-cvaux-repair-bcd-plan.md), which takes precedence over the broader repair specification.

This extension retains the current B/C/D rulings. It does not reinstate excluded whole-population burn-in, reconstruction certificates, paired-bootstrap gates, volume storage, or a full validation-certificate identity system. It retains the current `--ap-aux-validation required|off` semantics and the unvalidated stiffness cap.

## 2. Existing implementation and integration boundaries

At the reviewed commit:

| Component | Present behaviour | Extension |
|---|---|---|
| `adaptive/aux_discovery/descriptors.py` | Phi/psi sin/cos predictors; residue heavy-atom contacts and backbone H-bond descriptors | Add a topology-bound chi dictionary and separate predictor-family indices |
| `frames.py` | Phase-aligned solute frames, sample keys, selected conditioning coordinates | Carry the extended feature dictionary and preserve sample alignment |
| `partitions.py`, `pipeline.py` | One partition of CV-residualised contacts/H-bonds; candidate sources from passing, triggered k | Preserve labels independent of torsional predictors; require local co-occurrence for new-family candidates |
| `z3_search.py` | L1 torsion discriminants; backbone-specific model emission | Family-aware search, emission and joint best-of-search null |
| `placement.py` | Stable overlap forecast; parent-specific centres and strengths | Reuse placement and limits for the selected z3 |
| `auxiliary_cv/{model,features,evaluate,force}.py` | Frozen first-harmonic torsion projection | Versioned harmonic/orbit features for side-chain symmetry |
| `adaptive/aux_admission_io.py`, `aux_backfill.py`, `aux_pooling.py` | Admission, reconstruction checks, carrier burn-in and pooling | Support new model semantics without weakening current checks |

Paths in this table are under `gareus/`. The current force can already evaluate general atom quadruplets, but descriptor discovery and model emission are backbone-specific. Therefore this is not merely appending chi columns to a matrix.

## 3. Configuration and compatibility

Introduce `--ap-aux-feature-space backbone|sidechain|mixed|auto`, default `backbone`. In `auto`, search all three families; `mixed` permits both families but does not force both to have nonzero coefficients. Report the selected nonzero support honestly, including a mixed fit that reduces to backbone-only.

The option requires aux discovery. Freeze it in campaign options, with missing historical fields interpreted as `backbone`. Refuse changes on resume rather than silently regenerating an admitted model. Freeze chi dictionary version, symmetry policy, feature preprocessing and candidate enumeration in discovery settings/artifacts. Reject unknown values.

`backbone` follows the existing code, model schema, feature order and RNG sequence. Flag-off paths remain unchanged. `auto` with no eligible chi features reports that fact and runs the original backbone search without duplicate family candidates. `sidechain` with no eligible chi features admits nothing with reason `no_sidechain_features`.

Do not activate this extension implicitly in existing campaign configurations. Existing reserve size, maximum replica count, lambda=0 worker restriction, frozen worker policy and one-admission rule continue to apply.

## 4. Chi dictionary, topology and symmetry

Initial coverage is chi1 and chi2 of supported standard amino-acid residue templates and explicitly supported protonation variants. Do not add chi3/chi4 in this change. Alanine and glycine have no eligible heavy-atom chi. Exclude proline/ring-constrained chi from the initial dictionary; direct rotation is coupled to ring geometry. Unsupported modifications are reported and excluded before fitting, not guessed from atom order.

For each eligible torsion store: chain/residue identity including insertion identity when present, topology residue index, residue/template variant, chi index, ordered atom names and full-system indices, solute-to-system mapping, sign convention, harmonic and symmetry orbit. Verify bonded connectivity and four distinct integer indices. Do not assume that solute atoms form an unchanged prefix of the solvated topology. Bind the emitted model to the existing topology identity mechanism.

Use a curated, versioned residue dictionary with explicit treatment of atom equivalence. Candidate equivalences must preserve chemical connectivity, protonation and the actual force-field parameterisation. Similar names alone are insufficient. Aromatic branch labels and equivalent carboxylate oxygens must not define separate physical basins merely because labels were exchanged. A protonated carboxyl group is not interchangeable with a deprotonated one.

For unambiguous ordinary torsions use sin(theta), cos(theta). For a validated symmetry orbit G of atom-quadruplet permutations, define a primitive

    f(x) = (1 / |G|) sum[g in G] trig(m * sign * theta_g(x))

where trig is sin or cos, sign is +1 or -1 and harmonic m is 1 or 2. Use m=2 for the supported twofold symmetric terminal/ring torsions, and m=1 for ordinary torsions. The explicit orbit average supplies exact permutation invariance; simply asserting theta -> theta+pi is not an exact guarantee for distorted molecular geometry. Enumerate and validate the orbit once, never choose a representative by coordinate-dependent sorting.

Average features, not angles. Test invariance under relabelling and force covariance under the same atom permutation. If an orbit feature cancels or has negligible training variance, remove it as uninformative. Do not amplify numerical residue by dividing by its tiny standard deviation. If a supported symmetry cannot be established, omit that torsion with a recorded reason; other eligible features remain usable. No methyl-hydrogen rotations in the initial dictionary.

## 5. Frozen model and numerical semantics

Introduce `atlas-aux-cv-model-v2` for the extended feature representation and an aux-specific versioned feature schema. Keep v1 readers/writers and v1 hashes unchanged. Do not silently reinterpret shared CV-selection v1 features or change existing CV2 identities. Version-dispatch through the aux model facade; legacy backbone discovery continues emitting v1.

The model is

    z3(x) = [offset + sum_j coefficient_j * f_j(x)] / scale
    B_i(x) = 0.5 * k_i * [z3(x) - centre_i]^2

with a frozen positive scale and finite coefficients. z3 and its centre are dimensionless; k is energy per z3 squared. z3 is not an angle: do not wrap z3 or its umbrella displacement. Fold all training feature centring/scaling into the exported coefficients/offset. Use no runtime classifier, conditional fit, contact label, or CV1/CV2 bin lookup to evaluate z3.

Model identity includes all energy-defining feature fields: ordered quadruplet orbits, harmonic, trig, sign, weights, scale, offset and imaging convention. Hash the canonical representation using existing helpers. Provenance records dictionary version, candidate family, training phases, feature exclusions and search policy.

Extend both OpenMM and independent offline evaluation. Compile each orbit term into torsional subforces; multiply force derivatives by harmonic m and divide orbit contributions by orbit size. Preserve cross terms by squaring the complete scalar projection. Runtime observation must read the actual replica force, as in the current F07 repair.

Use float64 for fitting, model parameters, z and reconstructed bias energies. Validate finite inputs, nonzero scale and finite resulting energies through existing aux checks. Absent chi features are schema-level absence, not NaN padding. Invalid values in a declared active feature invalidate the relevant discovery attempt or production operation; never replace them by zero. Handle zero-strength states using existing inactive-state semantics.

Retain `periodic_imaging: none` and require the same whole-molecule geometry convention in training, runtime and backfill. Do not introduce minimum-image torsions in only one evaluator. A wrapped/split molecule that violates this convention must not silently yield a different z. Collinear/zero-length torsion geometry remains invalid; no artificial angle regularisation in this extension.

## 6. Local basin discovery and fitting

Preserve U13: build population labels only from the existing residualised contact/H-bond descriptors, never from chi/backbone predictor columns. These descriptors are separate observables, not statistically independent evidence. They may miss rotamers that have no resolved packing consequence; such rotamers need not receive workers. A new contact descriptor family is not required here.

Use the repaired conditioning contract for absent/constant CV2, invalid active coordinates and tied bins. Train all preprocessing on training data. For sidechain/mixed candidates, require the existing co-occurrence rule to identify at least one supported conditioning neighbourhood containing both populations; lineage separation alone does not justify a local side-chain worker. Record those neighbourhoods for placement.

Maintain phase-aware samples and current train/holdout split. Locality describes candidate discovery and parent placement, not a piecewise force: z3 is globally evaluable, including outside the neighbourhood used to fit it. Do not switch its coefficients when a replica crosses a bin boundary.

For each enabled family, search the same passing/triggered partition sources, eligible group pairs and frozen L1 regularisation grid. Apply training-only variance filtering, standardisation and the existing redundancy checks. Feature order and seeds must be deterministic. Deduplicate identical effective candidates, particularly when mixed fits reduce to a single family.

Select by existing held-out conditional information gain, with fewer nonzero coefficients and then canonical candidate identity as deterministic tie breakers. Compare against the winning backbone-only candidate and report both scores. Larger chi variance alone is not a selection criterion. A mixed model need not win merely because it has more available features.

Lambda=0 frames are still umbrella-biased. Within finite CV neighbourhoods, raw group frequencies are not unbiased equilibrium populations. Treat discovery scores as proposal heuristics, retain parent-aware placement, and reserve equilibrium probability claims for the existing MBAR analysis. Do not reinterpret these scores as barrier heights.

## 7. Null search and selection evidence

For `auto`, every scrambled-label replicate must repeat the entire enlarged search: all enabled feature families, partition sources, eligible pairs, regularisation choices and best-candidate selection. Compare the observed winner with the null winner across that complete search. Three independent family tests with an uncorrected winner are not acceptable.

Preserve the existing phase-local scrambling contract, `null_uninformative_trapped_lineages` refusal and discovery-attempt history. Holdout information used for candidate ranking is selection evidence, not an untouched final confirmation set. With 20 null searches, report a descriptive best-of-null result and its resolution; do not claim a calibrated campaign-wide significance level, especially across repeated attempts. The complete-search null is required for fair family comparison but does not cure every dependence or repeated-testing issue.

Add noise controls with fast/free chi motion, torsions decoupled from packing labels, and trapped label-constant carriers. Scrambling must preserve the sin/cos pairing and the internal geometry of a feature frame; destroy the predictor-label association without manufacturing invalid angular features.

## 8. Placement, admission and equilibrium bookkeeping

Reuse the repaired log-space placement forecast and existing quantile/width proposals, overlap, effective-sample and stiffness gates. For a sidechain/mixed winner, restrict eligible parents to those with training support in the recorded co-occurrence neighbourhoods and both relevant populations represented under the existing share rule. Holdout confirmation still applies. No extrapolated z centres beyond the configured observed quantile proposals.

Workers inherit the parent's CV1/CV2 umbrella and receive the selected z3 centre/strength. They remain lambda=0, within the existing reserve, and frozen after admission. Ordinary states carry zero aux strength. All eligible state pairs use the current complete-energy exchange machinery; locality does not restrict exchanges.

For state i, retain the reduced potential conceptually as

    u_i(x,V) = beta * [U_phys(x,V) + boost_i(x,V)
                       + W1_i(x) + W2_i(x) + B_i(x) + p*V]

using the actual dependent Pep-GaMD boost evaluator. Common physical/pV terms may cancel in equal-T/equal-p configuration exchanges, but the barostat must target the complete active Hamiltonian. Adding chi features introduces no new chi-space Jacobian: simulation still samples Cartesian configurations, and chi only defines an added potential.

Preserve existing carrier-based burn-in from actual seeding/pull evidence, including exchange transport and continuation/resume rules. Do not equate removal of a fixed burn-in interval with proof of equilibrium. Preserve the current heuristic crosscheck wording; ordinary-only agreement is not proof of correctness or independence.

Pre-admission samples require complete model-specific z backfill from available coordinates, even though no chi columns were recorded originally. Post-admission model-active phases require recorded z and the appropriate torsion basis. Extend column-to-feature metadata to cover harmonic/orbit features unambiguously; do not assume a fixed backbone width or equate feature count with distinct torsion count. Both union paths must select the same observation keys and reconstruct identical energies.

New-family models require their own force/evaluator tests: earlier backbone-only GPU results do not establish chi/harmonic parity. In `required` validation mode, check the declared sidechain/mixed feature coverage of evidence using a small explicit coverage field, not a new certificate system. Missing coverage is unavailable evidence. In `off` mode retain current admission semantics and k cap; report unvalidated status without fabricating a PASS. Both modes retain mathematical/schema, reconstruction and finite-value checks.

The same numeric k cap is not the same physical stiffness for different projections: forces depend on the gradient of z, and local curvature includes both its gradient and Hessian. Harmonic m=2 and a small normalisation scale can increase stiffness. Finite-timestep evidence must therefore exercise the proposed feature/normalisation regime and springs; a backbone result at the same k alone is insufficient. In validation-off mode expose this limitation explicitly rather than relabelling the cap as a stability guarantee.

## 9. Acceptance tests and scientific benchmark

| Test | Required outcome |
|---|---|
| Topology dictionary | Expected chi1/chi2 for supported residues; none for Ala/Gly; explicit Pro/unsupported exclusions; correct chain and atom mapping |
| Symmetry | Equivalent-atom relabelling leaves z/energy invariant and permutes forces correctly; asymmetric protonation is not falsely merged |
| Angular evaluation | Wrap continuity; m=1/m=2 CPU/offline/OpenMM agreement; finite-difference forces for mixed orbit models |
| Numerical cases | Constant/empty chi basis, tiny scale, invalid active values and degenerate geometries give explicit outcomes, never plausible zero energies |
| Conditional synthetic data | Planted chi-dependent packing groups at overlapping CVs are recoverable; CV-only differences do not qualify as hidden side-chain basins |
| Selection/null | All enabled families appear in each null search; duplicate candidates do not gain extra chances; decoupled/noise controls reject appropriately |
| Force/parity | Fast and slow CV1 paths read real force-side z; CPU and available GPU precision paths tested for new primitives |
| Exchange/NPT | Cross-state energies match direct Context energies; existing detailed-balance tests include chi workers; controlled distribution test includes active chi bias and volume moves |
| Reconstruction | Historical chi backfill agrees with recorded post-admission z within existing tolerances; missing/failed evidence refuses pooling |
| Lifecycle | Admission, pull, carrier exclusion, checkpoint/resume and frozen option checks work end to end for the new model |
| Compatibility | Backbone-only and aux-off identities, results and RNG behaviour remain unchanged on pinned fixtures |

Use targeted tests following the repository repair workflow; do not require unrelated whole-suite runs for this addition.

For scientific evaluation, compare backbone-only CVaux against auto CVaux at equal total GPU-hours and equal worker budgets with independent campaign repeats. Freeze benchmark readouts beforehand. Report folded-population/free-energy uncertainty and independent-run agreement, contact-population coverage, reweighting overlap/ESS, and discovery/backfill/runtime cost. A native structure may define an evaluation-only chignolin readout, never feed discovery or candidate tuning.

Repeated structural transitions and chi diversity are supporting diagnostics. Neither first-folding time nor worker occupancy alone demonstrates improved thermodynamic efficiency. An uninformative result is acceptable and must not trigger post-hoc threshold changes disguised as validation.

## 10. Delivery sequence and completion criteria

1. Add opt-in settings, topology dictionary and family-aware frame schema; preserve legacy behaviour.
2. Implement/version frozen harmonic-orbit model evaluation and force compilation, with symmetry and force tests.
3. Extend candidate search, emission and complete-search null; add synthetic positive/negative controls.
4. Integrate local parent eligibility, admission, new-model coverage reporting, storage/backfill and resume.
5. Run focused end-to-end correctness checks and the equal-cost sampling benchmark; report correctness and sampling benefit separately.

Implementation is complete when a sidechain or mixed model can pass the actual discovery-to-exchange-to-MBAR path, while backbone-only campaigns remain compatible. A claim of improved folding efficiency additionally requires the benchmark evidence. This specification alone authorises no claim that folded chignolin will be sampled faster.
