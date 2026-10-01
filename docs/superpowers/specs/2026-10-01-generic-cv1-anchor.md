# Generic CV1 anchor for the automatic residual CV2 (2026-10-01)

Status: phase 1 (framework + end-to-end distance) in implementation on
`feat/generic-cv1-anchor`; phase 2 (residue contact-map component) specified, not built.

## Why

`cv2: auto` residualises backbone torsions against CV1 and picks a slow, multimodal,
reproducible residual direction as CV2. Every stage hard-codes CV1 = the heavy-atom
nonlocal contact fraction: the swarm driver refuses anything else (`swarm/driver.py:197`),
the anchor is built from the trace `cv1` column (`swarm/analyze.py:403`), only the contact
kind is deployable (`cv_selection/anchor.py:7`), the pair model binds a contact-pair digest,
and the production force builds the CV2 on a contact sum (`production.py:539-552`).

The 2026-10-01 CV pair search on chignolin_10's swarm
(`~/.claude/jobs/6bfa4c5c/tmp/modality/pair_search*.json`) found the contact fraction a poor
coverage coordinate (held-out CA-cluster information 0.01 +- 0.04 nats) and fast (rho 0.80
at 200 ps); end-to-end distance covers more geometry (0.13) but is fastest (0.74); the slow
tICA mode of the residue contact map was best (0.25, rho 0.97). The user asked for one
generic mechanism, so that CV2 can be residualised against any of these.

## The abstraction

Every supported CV1 is a weighted sum of a per-atom-pair function, in the CV1's own units:

    c = scale * sum_p w_p f(r_p)        (r in nm, as OpenMM computes it)

| kind | pairs | f(r) | w | scale | units | runtime CV1 umbrella |
|---|---|---|---|---|---|---|
| `nonlocal-contact-fraction` (today) | heavy pairs, min separation | rational/sigmoid switch | 1 | 1/norm | dimensionless | contact umbrella (exists) |
| `end-to-end-distance` (phase 1) | one atom pair (terminal CA by default) | r | 1 | 10 (nm -> A) | angstrom | distance umbrella (exists) |
| `contact-map-component` (phase 2) | heavy pairs of selected residue pairs | switch | tICA loading | 1 | dimensionless | new |

The residual coordinate already has the form `c = anchor_sub_cv / norm`
(`residual_runtime.CompiledResidualComponent`), so the CV2 math is unchanged: the anchor
kind only decides how the anchor sub-CV is built and evaluated (`norm` = 1/scale; 0.1 for
end-to-end distance in A). One anchor spec module owns, per kind: the sub-CV force builder,
the positions evaluator, the swarm-trace column, the units and k bounds, and the
deployment binding.

## Phase 1 requirements (end-to-end distance)

1. **Units.** The anchor is fitted on `e2e_nm * 10` (A); `anchor_mean`/`anchor_std` are in A.
   All k values for this anchor are kcal/mol/A^2: selection's `k1_kcal_reference` (the
   coupling check), `n_resolvable_windows`, the CV1 ladder k bounds (today the contact-scaled
   `contact_adaptive_max/min_k_kcal` 1200/5 at analyze.py:517-518), the coupling gate. The
   [0, 1] clips in `ladder_design` apply only to the contact fraction.
2. **One distance.** The residual force's anchor sub-CV and the CV1 distance umbrella use the
   same atoms (`cv.choose_cv_atoms`, terminal CA by default; the swarm's `e2e_nm` is the same
   pair) and the same periodicity (non-periodic `CustomBondForce`, as `forces.add_umbrella_force`).
3. **Refuse mismatches.** Production refuses a pair model whose anchor kind or binding differs
   from the run's CV1, both directions (contact model on a distance run and the reverse).
   `apply_epoch0_sidecar` writes `cvs.cv1` from the analysis' anchor, so a mismatching YAML is
   caught at the first job.
4. **Deployment binding.** New optional pair-model deployment field `anchor_binding_sha256`
   (kind-specific digest: the atom pair for distance). `contact_pair_list_sha256` stays the
   contact kind's binding and is never reused. Existing chignolin_9/10 pair models and
   candidate sets must validate to the same digests (regression test). The residual
   evaluator version is bumped only if the compiled record format changes.
5. **Swarm side.** The analyze-only path on a reused round 0 needs no contact CV1 (round 0
   already stratified on contacts). `_library_cv1_for_round0` uses the anchor's column of
   `seed_descriptors.csv` (`e2e_nm` exists). Gates and the CV1 ladder read the anchor values.
   The sidecar writes the anchor's `cv1`. Fresh swarm MD with a distance CV1 (stratify on
   contacts regardless) is out of scope for phase 1 and still refused, with a clear message.
6. **Positions evaluators and seeding.** `cv.residual_cv2_from_positions_nm` gains the
   distance anchor; seed scoring, the seeding preflight and the start-quality gate work in A
   (the existing distance-mode code); extension-seeding restraint checks, slow-mode reseed,
   fast-path sub-CV roles, CV1-free (X7) windows and the analyzer's 2-D bias reconstruction
   are checked for contact assumptions.
7. **Tests.** Force-vs-evaluator oracle (energy and z) on the small fixture for the distance
   anchor; contacts path byte-identical (existing pair-model/candidate-set digests, force
   expression); a scratch `--swarm-stage analyze` replay on chignolin_10's swarm with
   `--cv1 distance` reporting the CV2 pick and layout.

## Phase 2 (contact-map component) -- open decisions

- The 2026-10-01 scores used a min-distance switch per residue pair; production needs a
  smooth definition (summed heavy-atom switches per residue pair). That is a different
  coordinate: re-run the pair search on the production definition before building.
- Coordinates exist only for the 20 ps PDB frames (4,872 rows of 48,024): either record
  per-frame residue contact counts in future swarms, or select CV2 on PDB rows only.
- CV1 becomes a fitted artifact (tICA of the contact map): it needs freezing, a digest and a
  reproducibility gate like the pair model, plus its own CV1 umbrella force (weighted contact
  sum) and window design.
