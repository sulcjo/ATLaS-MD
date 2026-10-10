# Local GLM handoff: side-chain CVaux

Target: PR #141, `feat/cvaux-stage-a`. Work started from `4bcf845`.
Purpose: finish integration in small tasks using the supplied, tested numerical kernels.
Read this file first. The [detailed plan](2026-10-10-cvaux-sidechain-z3-implementation-plan.md) and [specification](../specs/2026-10-10-cvaux-sidechain-z3-extension-spec.md) explain the science. Do not implement everything in one prompt.

## What is already done

| File | Implemented | Still needs integration |
|---|---|---|
| `gareus/auxiliary_cv/sidechain_core.py` | Immutable periodic/orbit primitives, canonical terms, angle/position evaluator, analytic Cartesian gradient, explicit atom remapping, OpenMM force compiler | Production model schema, chemistry dictionary, observer/state/checkpoint adapters |
| `gareus/adaptive/aux_discovery/local_search_core.py` | Regional L1 fitting, separate fit/tune/holdout, stable likelihoods, comparable scores, frozen exported projection, shared-shift best-of-search null across a partition matrix | Real frame/partition adapters, CV redundancy guards, production reporting and placement |
| `gareus/adaptive/aux_backfill.py` | Empty worker coverage no longer reports zero error; full-frame discrepancy is reported; invalid inputs refused | v2 per-phase atom mapping and the stricter v2 all-frame verdict |
| `gareus/adaptive/aux_admission_io.py` | Warning reports the unavailable-evidence status instead of None | v2 admission integration |

The kernels are not imported by production automatically. Side-chain discovery is NOT enabled by this commit. The backfill coverage correction is an intentional live bug fix for existing aux campaigns.

## Copy this into the local model's first prompt

```text
Work on the existing PR141 branch feat/cvaux-stage-a.
Read docs/superpowers/plans/2026-10-10-cvaux-glm-handoff.md.
Implement ONLY the next unfinished task card. Inspect git status first.
Reuse the supplied sidechain_core and local_search_core kernels.
Do not change the backbone-only/default path, its hashes, or RNG draws.
Do not change exchange acceptance, Pep-GaMD, timestep or barostat algorithms.
Do not weaken a failed test or replace failed/missing data with zero.
Write a regression, make the smallest change, run the card's tests.
End with changed files, test results, and the next card number.
Do not claim the whole feature works until the final end-to-end card passes.
```

Run one card per model session if context is limited. Keep a short local progress note with completed card numbers, exact commit and test result. Read only the files named by the current card plus their direct callers. Never infer completion from the existence of a file.

## Fixed decisions: do not redesign these

1. Default `backbone` uses the old pipeline unchanged. New modes are `sidechain`, `mixed`, `auto` and are explicit opt-ins.
2. **Simplification to the earlier plan:** auto uses the supplied LOCAL fitting/scoring for all enabled families, including backbone. It does not mix old global multiclass scores with local binary scores. This supersedes T05's global-backbone-rescoring adapter in auto only. Backbone-only mode remains unchanged.
3. One admitted model and the existing worker reserve; no new CV grid, additional temperature ladder or simultaneous models.
4. Fit locally, evaluate globally: region membership never enters the force expression.
5. Candidate partition identity + region + group pair travel together into placement. Never substitute the largest-k partition.
6. No native/folded reference in discovery. Keep `--ap-aux-validation required|off`; off does not disable schema, force or reconstruction checks.
7. The supplied core serialization (`atlas-aux-projection-core-v1`) is NOT a production AuxModel or checkpoint payload. Wrap it inside a validated v2 model; do not hand it directly to existing v1 readers.
8. A supplied symmetry orbit is an input assumption. The kernel averages it exactly but cannot prove chemical or force-field equivalence.

## Exact kernel APIs

```python
from gareus.auxiliary_cv.sidechain_core import Primitive, Projection, atom_map

# Twofold equivalent terminal torsion; orbit validated by the dictionary adapter.
primitive = Primitive(((0, 1, 2, 3), (0, 1, 2, 4)), 'cos', harmonic=2, sign=1)
p = Projection((primitive,), (0.7,), offset=0.2, scale=1.3)
z = p.values(xyz_nm)                 # shape (frames,), or one frame
z, dz_dx = p.value_gradient(xyz_nm)  # exactly one configuration
theta_order = p.quads                # canonical ACTIVE unique quadruplets
z = p.from_angles(theta)             # theta columns must match p.quads
force = p.build_force(openmm, center=0., k_kj=2.0, force_group=3)
# Production still chooses a free group and supplies its observer/identity facade.
mapping = atom_map(production_atom_keys, trajectory_atom_keys)
trajectory_projection = p.remap(mapping)
z_xtc = trajectory_projection.values(trajectory_xyz_nm)
```

`Projection` coefficients and offset are BEFORE division by `scale`. When exporting a fitted candidate:

```python
p = Projection(tuple(dictionary_primitives), tuple(candidate.coefficients),
               offset=candidate.offset, scale=candidate.scale)
```

Do not divide coefficients by scale a second time. Do not wrap z as an angle. Energy is `0.5*k*(z-centre)**2`; convert kcal to kJ exactly once at the force boundary. Core forces use globals `aux_k`, `aux_c` and name `ATLaSAuxCVUmbrella`, but production force inspection must be adapted to their new sub-CV expressions/names.

```python
from gareus.adaptive.aux_discovery.local_search_core import fit_local_candidates, compare_null

candidates = fit_local_candidates(
    X, labels, region_ids, split_ids, partition_id=partition_sha,
    families={'backbone': backbone_cols, 'sidechain': chi_cols, 'mixed': all_cols},
    c_grid=settings.l1_c_grid, min_train=100, min_holdout=50)
# split_ids: 0=fit, 1=tune, 2=holdout. No random frame splitting.
# Each candidate: partition_id, region, pair, family, coefficients, offset,
# scale, local_gain, prevalence, score, values(X), identity.

def complete_search(label_matrix):
    found = []
    for column, partition_sha in enumerate(partition_ids):
        found.extend(fit_local_candidates(
            X, label_matrix[:, column], region_ids, split_ids,
            partition_id=partition_sha, families=enabled_families,
            c_grid=settings.l1_c_grid))
    return apply_frozen_redundancy_guards(found)  # same function for observed/null

gate = compare_null(label_matrix, phase_local_carrier_ids, steps,
                    complete_search, n_null=settings.n_null_z3, seed=frozen_seed)
```

Use the same masks, preprocessing policy and guards in every callback invocation. `compare_null` shifts the label matrix jointly, preserving relationships between partitions. It is a heuristic best-of-search comparison, not a calibrated p-value. Constant-label carriers give an unavailable null. Candidate.score is local held-out gain times training-region/pair prevalence, not an equilibrium basin probability.

## Card 1 — Add configuration, no scientific changes

Edit only `gareus/cli.py`, `adaptive/aux_discovery/settings.py`, driver option wiring, and a small option test.
Add the four-value flag, default backbone. Freeze it per campaign; missing legacy field means backbone. Reject mismatched resume. Do not invoke either new core module on the default path.

Check: existing settings/CLI tests and a new default/invalid/resume test. Done when the flag round-trips through settings without changing the old search result. Commit separately.

## Card 2 — Build feature dictionary and per-phase atom maps

Add `auxiliary_cv/sidechain_dictionary.py` and `atom_mapping.py`; extend discovery descriptors. Start with explicit standard residue templates and chi1/chi2. No proline ring chi, chi3/chi4, methyl-H rotations or guessed nonstandard residues. Ala/Gly have no chi. Include template/protonation and exclusion reasons.

Create stable atom keys containing chain, residue/insertion identity and atom name; verify connectivity and map uniqueness. Use the supplied `atom_map`; never assume solute is a prefix. Dictionary order is frozen. One Primitive per sin/cos/harmonic/orbit feature. For symmetry cases, inspect the parameterised System once and record the validated orbit; ordinary topology names alone are insufficient. If equivalence is not supported, exclude that torsion explicitly. Do not assign harmonic=2 to every chi2: it applies only to the declared twofold-symmetry cases.

Check: independent fixtures for supported residue templates, equivalent atom permutation, protonation asymmetry, duplicate residue numbers across chains and reordered/non-prefix solute. Do not proceed without these tests. This card is chemistry-sensitive; report unsupported cases instead of improvising.

## Card 3 — Wrap the kernel in production v2 model identity

Edit `auxiliary_cv/model.py`; add an aux-specific v2 schema facade. Store/bind the primitive list, topology identity, coefficients, offset, scale, dictionary version and imaging policy. Delegate numerical work to Projection. Existing v1 parser/serializer/hash remains untouched.

Audit all direct `.atom_indices` consumers and dispatch by version. Core `p.quads` includes only active terms; store a separate complete dictionary when inactive-feature provenance is needed. Do not pretend feature width equals unique-angle count. Restrict production models to a nonconstant supported projection even though the numerical core permits a constant projection for tests/inactive expressions.

Check: v1 hash pins, v2 roundtrip, unsupported schema, unknown fields and model/atom mismatch. Done when a v2 model evaluates through the facade but no new campaign has been enabled.

## Card 4 — Force observer, angle storage and resume adapters

Edit `auxiliary_cv/{force,runtime,sample_schema,runtime_definition,checkpoint}.py` and direct consumers.
Reuse `Projection.build_force`; select its free group through the existing allocator. Add a v2 observer that sums the actual sub-CV values, adds offset, then divides by scale. Validate sub-CV expressions and weights against the compiled projection, including harmonics/orbit averaging. Do not call the position evaluator as the force observer.

Write a v2 sample payload with ordered raw-angle quadruplets and model identity; preserve v1 payloads. Store the union of required angles with an explicit column map. Checkpoint/resume binds the same model and basis.

Check: run `tests/test_aux_sidechain_core.py`, then existing aux observation/runtime/sample/checkpoint tests and new v2 roundtrips. Actual OpenMM Reference comparisons are required; do not replace them by fake force objects.

## Card 5 — Connect local search using existing frozen partitions

Edit `adaptive/aux_discovery/{frames,pipeline,z3_search}.py`. X is the feature matrix evaluated from the exact same Primitive dictionary used by the model facade. Keep contact/H-bond partition labels independent of torsion predictor columns.

Use the partition's frozen CV bin IDs as regions. Epoch holdout rows get split=2; divide training carriers into deterministic fit/tune groups using the current carrier grouping, never random individual frames. Include phase in identity and ensure a shuffle block does not cross split boundaries. Drop no invalid active observations silently.

Call `fit_local_candidates` through one complete-search wrapper for all source partitions/families, and call `compare_null` once on the joint label matrix. Apply active-CV redundancy checks identically inside the observed/null wrapper. Use all candidates returned, then deterministic score/size/identity ranking. If auto has no chi columns, delegate to the original backbone path exactly.

Check: `tests/test_aux_local_search_core.py`, plus a real frame-table adapter fixture demonstrating opposite local relationships, no invented holdout region, CV-only negative control, constant chi and no informative null.

## Card 6 — Place workers using the winning partition

Edit `pipeline.py`, `placement.py`, `aux_admission_io.py`. Bind winner.partition_id, region and pair in the frozen admission record. Pass THAT partition's labels to placement; never largest-k labels. Parent local-support check uses both winning groups. The forecast still uses the parent's complete eligible distribution, not only the region subset.

One model, existing caps and reserve. No admissible placements means no_worker. Do not try runner-up models without including that extra selection procedure in the null definition. Emit Projection from the candidate with the exact API above; compare fitted and emitted z before admission.

Check: winner from k=2 while k=4 also passes; identity survives admission; wrong parent rejected; no_worker cleanly returned; nonzero stiffness and units correct. Existing ordinary-state behaviour stays pinned.

## Card 7 — Finish v2 backfill and validation coverage

Use `Projection.remap` in `aux_backfill` XTC and final-PDB paths for v2. Keep step evidence, sample keys, hashes and no-fallback rules. The empty-coverage fix is already supplied: do not undo None/status handling.

For v2, require nonempty relevant coverage and apply the existing energy tolerance to the newly reported all-frame discrepancy. Preserve the existing v1 covered-frame verdict. Add feature/harmonic coverage to required validation records; backbone-only evidence does not cover chi harmonic-2. Keep validation-off policy while reporting absent physical evidence.

Check: reordered atom maps, missing orbit atom, empty coverage, error only outside two widths, checksum mismatch and both union selection paths. Run existing backfill/reconstruction suites as well as the new coverage tests.

## Card 8 — End-to-end and handover

Exercise actual discovery -> admission -> force observation -> sample storage -> backfill -> union MBAR with a small deterministic fixture. Separately test force parity, cross-state energies, pull/carrier burn-in, crash/resume, and NPT volume moves under an active chi bias. Preserve the current exchange and barostat algorithms; integration tests verify their inputs.

Record CPU/GPU/timestep checks actually executed. No GPU access means GPU validation remains pending, not passed. Compare auto/backbone with equal GPU-hours and worker budgets before claiming better folding efficiency. Update status in this file and the detailed plan only after checks pass.

## Test commands and delivered evidence

Core and live backfill regression tests:

```bash
python -m pytest tests/test_aux_sidechain_core.py tests/test_aux_local_search_core.py tests/test_aux_backfill_coverage.py tests/test_aux_backfill.py tests/test_aux_reconstruction_pooling.py -q
```

The preparation run uses Python 3.12, OpenMM 8.6.1 on Reference, NumPy/scikit-learn from the task environment and isolated pytest/Arrow/DuckDB dependencies. Result: **80 passed, 0 skipped, 5 warnings**. OpenMM force comparison, symmetry/gradient tests, regional counterexample and existing backfill/union regression tests were executed. The five warnings are synthetic union fixtures using their existing default-temperature fallback.

Not executed here: CUDA/OpenCL, production NPT distribution validation, long-timestep stability, real chi campaign or performance benchmark. The new kernels remain unwired until the cards are completed.
