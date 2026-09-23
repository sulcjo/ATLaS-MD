# Spec: real-space surrogate for the Pep-GaMD peptide energy V_pep

- **Status:** reviewed design revision; implementation-ready after the mandatory gates below (not implemented)
- **Date:** 2026-09-23
- **Code base reviewed:** ATLaS-MD / gareus v0.8.2, `main` @ dfeb8f983904948ddeeb0f8fa898f4cc9387ecf2
- **Target campaign:** chignolin_9 (new boost kernel, new swarm envelope; never a resume of an
  existing exact-Pep-GaMD campaign)
- **Author:** ATLaS-MD development session; revised after independent review against `main`

---

## 1. Problem

`PepGaMDLowerDualIntegrator` (`gareus/pep_gamd.py`) boosts only the peptide. The current exact
implementation obtains the peptide-focused potential by subtracting a water-only copy of the
physical nonbonded force:

| Force group | Current exact-Pep-GaMD contents | Energy |
|---|---|---|
| 0 | physical `NonbondedForce` (full PME), `HarmonicBondForce`, `HarmonicAngleForce`, `CMMotionRemover` | E0 |
| 1 | auxiliary `NonbondedForce` `PepGaMDWaterOnlyNonbonded`: physical copy with peptide q, epsilon and peptide exceptions zeroed, PME parameters pinned to group 0 | E1 |
| 2 | `PeriodicTorsionForce`, `CMAPTorsionForce` | E2 |
| 29, 31 | secondary-CV restraint, primary contact umbrella (never boosted) | bias |

```text
V_pep_exact = E0 - E1 + E2
```

Every boosted MD step evaluates two full PME forces, groups 0 and 1, in separate passes. Measured on
the real chignolin_8 system (about 21k atoms, TIP3P, cutoff 0.8 nm, Ewald tolerance 1e-4), one
context per NVIDIA L40S, no contention (jobs 2577764 and 2579032):

| Step | steps/s | vs Langevin |
|---|---:|---:|
| Langevin, one force evaluation | 4,170 | 1.00 |
| Langevin + auxiliary PME, still one evaluation | 3,030 | 0.73 |
| Pep-GaMD, per-group evaluations | 1,978 | 0.47 |
| Pep-GaMD + production CV forces | 1,461 | 0.35 |

The water-only PME costs about 27% by itself. Its separate pass, together with the remaining
per-group reads, costs about another 35%. The peptide has 138 atoms and the solvent about 21,000, so
a second full-system PME is being spent to recover the energy of a small atom subset.

The proposal is to replace that second PME, only for a new Pep-GaMD kernel, by a cheap real-space
surrogate that remains a **measurement channel**, never a physical force.

## 2. Correctness argument

### 2.1 The boost coordinate need not equal the exact peptide PME energy

GaMD equilibrium reweighting is valid for any deterministic differentiable boost

```text
DeltaV(x) = g(S(x))
```

provided that:

1. the propagated force is exactly `-grad[U_phys(x) + U_bias(x) + DeltaV(x)]`; and
2. exchange and MBAR reconstruct exactly the same `DeltaV` from the same recorded raw boost
   coordinates and the same frozen envelope.

Therefore the Total-channel coordinate does not need to equal `E0 - E1 + E2`. The surrogate only
needs to be a useful smooth coordinate for lowering the barriers of interest. Its agreement with the
old exact target controls **sampling efficiency**, not the formal equilibrium target.

For NPT the same statement is box-dependent:

```text
DeltaV = DeltaV(x, B)
```

The application-controlled biased-MC barostat must evaluate the same surrogate at both old and trial
boxes and use

```text
U*(x, B) = U_phys(x, B) + U_bias(x, B) + DeltaV(x, B)
```

for the Metropolis volume-move energy. The surrogate measurement energy itself is never added to
`U*`.

### 2.2 Dependent dual ordering is unchanged

The dihedral channel is evaluated first. Its boost enters the Total-channel argument exactly as in
the current dependent-dual implementation. The raw per-frame Total coordinate is the energy before
that dependent dihedral boost is added.

This ordering is load-bearing for the lambda ladder and MBAR reconstruction.

## 3. Mandatory force-group architecture

### 3.1 Reuse force group 1 as the universal Pep-GaMD measurement channel

Do **not** allocate the surrogate to a new force group.

The current repository defines `pep_gamd_bias_force_groups(system)` as every occupied group outside
`{0, 1, 2}`. A surrogate placed in a new group would therefore be classified as an ordinary bias
force and physically applied in addition to being used as a boost coordinate.

Instead preserve one stable invariant:

> **Force group 1 is the Pep-GaMD measurement channel. It is never part of the physical
> Hamiltonian and is excluded from ordinary integration.**

The two Pep-GaMD kernels are mutually exclusive:

| Boost kernel | Group 1 contents | Total raw coordinate |
|---|---|---|
| `pep-gamd-lower-dual` | exact water-only PME auxiliary | `E0 - E1 + E2` |
| `pep-gamd-lower-dual-rs` | real-space surrogate force set | `E1 + E2 = S + E2` |

An exact auxiliary and an RS surrogate must never coexist in the same `System`. Construction must
fail loudly if both are detected.

Group 1 may contain multiple OpenMM Force objects for the RS kernel. `energy1` and `f1` then
naturally mean the sum of all surrogate components.

### 3.2 Required RS force names

Use stable names so every code path can identify measurement forces by role rather than by Force
class:

- `PepGaMDRSNonbonded`
- `PepGaMDRSExceptions`
- `PepGaMDRSBonds`
- `PepGaMDRSAngles`

All live in force group 1. Optional future components must use the `PepGaMDRS` prefix and group 1.

Generalize the existing exact-only helper logic to recognize a Pep-GaMD measurement channel, rather
than assuming that the only nonphysical measuring force is `PepGaMDWaterOnlyNonbonded`.

### 3.3 Physical and bias semantics

For both exact and RS Pep-GaMD:

- physical energy excludes group 1;
- conventional MD integration excludes group 1;
- group 2 remains physical peptide torsion energy;
- umbrella and secondary-CV forces remain outside groups 0, 1 and 2;
- `pep_gamd_bias_force_groups` can retain its current complement-of-{0,1,2} definition.

This preserves the existing group ontology and avoids a repo-wide new-group migration.

## 4. Proposed RS energy and force

### 4.1 Energy definition

Let group 1 contain the surrogate energy `S(x)` and group 2 contain `E2(x)`.

```text
S(x)       = E1(x)                       # surrogate peptide-focused non-torsion energy
V_D(x)     = E2(x)                       # Dihedral raw coordinate
V_T_raw(x) = S(x) + E2(x)                # raw Total coordinate recorded as v_pep_kj_mol

dV_D = 1/2 k_D (Eth_D - E2)^2            if the lower-bound GaMD guard is active
Y     = S + E2 + dV_D
dV_T = 1/2 k_T (Eth_T - Y)^2             if the lower-bound GaMD guard is active

FSF_D = 1 - k_D (Eth_D - E2)
FSF_T = 1 - k_T (Eth_T - Y)

U* = U_phys + U_bias + dV_D + dV_T
```

The exact gamd-openmm small-range guards and threshold guards remain authoritative; the equations
above show the dependency structure, not a replacement implementation of those guards.

### 4.2 Force algebra

Let `f1 = -grad S` and `f2 = -grad E2`. Then

```text
-grad U*
    = f_phys + f_bias
      - (1 - FSF_D) f2
      - (1 - FSF_T) (f1 + FSF_D f2)

    = f_phys + f_bias
      - (1 - FSF_T) f1
      - (1 - FSF_T FSF_D) f2
```

This is the force that must be propagated.

### 4.3 One-PME implementation

Set the integrator's integration-force-group mask to all physical and bias groups **except group 1**.
OpenMM's `CustomIntegrator` bare `f`/`energy` use that integration-force-group mask, while explicit
`f1`/`energy1` request group 1 directly. This behavior has been checked against the current OpenMM
implementation and must still be locked by a repository test.

The boosted update becomes

```text
F = f - (1 - FSF_T) f1 - (1 - FSF_T FSF_D) f2
```

where bare `f` already equals `f_phys + f_bias`.

Required per-step reads:

| Read | Contents | Expected cost |
|---|---|---|
| `f` | physical + bias, group 1 excluded | one full physical PME pass |
| `f1` and `energy1` | surrogate components only | cheap real-space peptide-focused pass |
| `f2` and `energy2` | dihedrals | cheap |

The conventional-MD stages can use bare `f` directly with the same group-1 exclusion.

There is no fallback form that intentionally includes group 1 in bare `f`. A failure of the
integration-force-group invariant is a hard error/test failure, not a production alternative.

## 5. Surrogate construction

### 5.1 Pair set

The surrogate nonbonded pair set contains every pair with at least one peptide atom:

- peptide-peptide; and
- peptide-solvent.

Water-water and other nonpeptide-nonpeptide pairs are absent.

Use `CustomNonbondedForce` interaction groups:

```text
{peptide} x {peptide}
{peptide} x {nonpeptide}
```

with the nonpeptide set explicitly disjoint from the peptide set. Tests must verify that no pair is
counted twice.

Use periodic cutoff semantics appropriate to the physical system.

### 5.2 Exceptions: mirror the physical NonbondedForce exactly

Do not infer 1-2, 1-3 or 1-4 structure from topology.

Iterate the physical `NonbondedForce` exception table. For every exception involving at least one
peptide atom:

1. add the pair as an exclusion to the surrogate `CustomNonbondedForce`;
2. if the physical exception has nonzero Coulomb and/or LJ energy, recreate that exception exactly
   in `PepGaMDRSExceptions` using its actual `chargeProd`, `sigma` and `epsilon`.

This prevents double counting and preserves arbitrary force-field exception scaling.

### 5.3 Bonded terms

`S` also includes peptide-associated physical `HarmonicBondForce` and `HarmonicAngleForce` terms
that are part of the intended non-torsion peptide target. Copy only the selected terms into group 1;
do not move or modify the physical originals.

For the chignolin_9 TIP3P target, rigid water is expected not to contribute physical harmonic
bond/angle energy. Confirm this on the built `System` and record the result. If a future solvent
model has energetic bonded terms, they must not enter the RS surrogate merely because they happen to
share a physical force group.

Torsions remain exclusively in group 2 and are not duplicated into `S`.

### 5.4 Candidate electrostatics

| ID | Electrostatics | LJ | Purpose |
|---|---|---|---|
| **S1-direct** | PME direct-space term `q_i q_j erfc(alpha r)/r` with the physical PME alpha and cutoff | physical pair LJ form | closest diagnostic approximation to the existing exact target |
| **S1-smooth** | smoothly switched/force-shifted version of the PME direct-space term | physical pair LJ form with compatible cutoff handling | preferred production candidate if it preserves correlation and improves force continuity |
| S2 | DSF/Wolf | same physical LJ family | robust smooth fallback |
| S3 | reaction field, e.g. `eps_rf = 78.5` | same physical LJ family | simple reference/fallback |

For this application, smoothness of `S` and `grad S` at the cutoff is more important than literal
identity to the direct-space PME term, because `S` is a boost coordinate, not part of
`U_phys`.

Chignolin (GYDPETGTWG) has charged termini plus D3 and E5 side chains, so removal of reciprocal-space
electrostatics may matter. This is an efficiency question and must be measured rather than assumed.

## 6. Offline surrogate gate

Use existing chignolin_8 production frames spanning the umbrella windows plus swarm/seed-bank frames.
On identical coordinates evaluate:

```text
B_exact = E0 - E1_exact
S       = candidate surrogate group-1 energy
```

Compare non-torsion components here. For a full Total-channel comparison use `B_exact + E2` versus
`S + E2`.

Minimum prefilter:

| Metric | Gate | Rationale |
|---|---:|---|
| pooled Pearson r, `S` vs `B_exact` | >= 0.90 | fluctuation tracking |
| per-window Pearson r | >= 0.80 in every materially populated window | avoid a locally broken target |
| per-window `sigma(S)/sigma(B_exact)` | 0.8-1.25 | protects GaMD envelope scale |
| peptide-atom force cosine, `f_S` vs exact non-torsion target force | median >= 0.90 | direction of barrier flattening |
| force cosine lower tail | p10 >= 0.70 | prevents a good median hiding bad regions |
| peptide-atom force-norm ratio | report median and p10-p90; no near-zero-collapse mode | cosine alone cannot detect a vanishing surrogate |
| finite energy/force | 100% finite | basic gate |

These are **prefilters**, not the final promotion criterion. A surrogate that is thermodynamically
valid and samples better must not be rejected solely because it is not numerically identical to the
old target.

If no candidate clears the prefilter, retain exact Pep-GaMD.

## 7. Short boosted pilot gate

After the offline gate, run a short representative boosted pilot over compact, intermediate and
extended regions using a newly calibrated RS envelope.

Required diagnostics:

- no NaN/Inf or constraint instability;
- cutoff/stability test, including an NVE or deterministic force-continuity control against the
  physical-system baseline;
- distribution of `dV_D`, `dV_T` and total `dV`;
- existing GaMD anharmonicity diagnostics;
- fraction of frames with each boost channel active;
- lambda-rung overlap and effective sample-size diagnostics;
- exchange acceptance / round trips when a ladder is used;
- structural-space coverage and transitions as **efficiency** metrics;
- no systematic disappearance of compact/folded-like seed-bank regions relative to the exact
  control.

Promotion is based on the combined thermodynamic and sampling diagnostics, not transition count
alone.

## 8. Recording and reweighting semantics

### 8.1 Preserve `v_pep_kj_mol` as the raw Total-channel coordinate

For the RS kernel record

```text
v_pep_kj_mol = S + E2
v_dih_kj_mol = E2
```

This preserves the existing semantic contract consumed by `pep_gamd_boost_kj`: `v_pep_kj_mol` is
the raw Total-channel energy before the dependent dihedral boost is added.

Do **not** replace `v_pep_kj_mol` with bare `S`. Doing so would silently remove `E2` from the
Total-channel reconstruction.

Optionally add

```text
v_s_kj_mol = S
```

as a diagnostic column. It is not required for MBAR if `v_pep_kj_mol` and `v_dih_kj_mol` are
present.

### 8.2 Do not evaluate the exact auxiliary PME during RS production

Running the old exact water-only PME merely to record exact and surrogate energies would reintroduce
the cost this design removes.

Exact-versus-surrogate comparisons belong in offline analysis contexts over saved frames.

### 8.3 Frozen envelope

The RS envelope is calibrated from RS raw coordinates and frozen for the campaign exactly like the
current ladder envelope. Every exchange/MBAR consumer must reconstruct the boost from the recorded
`v_pep_kj_mol`, `v_dih_kj_mol` and that RS envelope.

No consumer may infer the boost from physical potential energy or from the old exact-Pep-GaMD
definition.

## 9. Repository implementation plan

### 9.1 Boost type and dispatch

Add

```text
pep-gamd-lower-dual-rs
```

rather than `pep-gamd-rs-lower-dual`. After the existing `pep-gamd-` prefix is stripped, the current
calibration dispatcher still sees a name beginning with `lower`.

Even with this compatible name, add an explicit regression test for threshold dispatch.

Generalize the strict exact-only checks:

- `is_pep_gamd(args)` becomes membership in the Pep-GaMD family;
- add an exact-vs-RS discriminator;
- add the RS type to `LADDER_BOOST_TYPES`;
- keep `SUPPORTED_BIASED_MC_BOOST_TYPES` synchronized;
- update CLI/help/preflight allow-lists.

The stock `pep-gamd-lower-dual` path remains bit-identical.

### 9.2 Partition builder

Add `ensure_pep_gamd_rs_partition(system, peptide_atoms, flavour)`.

It must:

1. fail if the exact water-only auxiliary already exists;
2. be idempotent for an already-built matching RS partition;
3. construct all RS components in force group 1;
4. assign the ordinary physical forces to groups 0/2 exactly as today;
5. leave umbrella/secondary forces outside 0/1/2;
6. expose enough metadata/name checks to distinguish surrogate flavour and prevent an accidental
   S1/S2/S3 resume mismatch.

Generalize `assign_pep_gamd_force_groups` so recognized RS measurement forces are legal in group 1,
while unrelated `Custom*` forces in groups 0-2 are still rejected.

### 9.3 Integrator

Add an RS lower-dependent-dual integrator or parameterize the current subclass without changing the
exact path.

RS `_setup_energy_values`:

```text
PepE1 = energy1
PepE2 = energy2

StartingPotentialEnergy_Dihedral = PepE2
StartingPotentialEnergy_Total    = PepE1 + PepE2
```

RS boosted force update:

```text
PepF1 = f1
PepF2 = f2
v += fscale * (f - (1-FSF_T)*PepF1 - (1-FSF_T*FSF_D)*PepF2) / m
```

The integration-force-group mask excludes group 1.

Conventional stages use the same mask and bare `f`, with no surrogate force applied.

### 9.4 Generic Total-channel helpers

`total_energy_groups_for_args(args)` must return:

```text
exact Pep-GaMD: plus={0,2}, minus={1}
RS Pep-GaMD:    plus={1,2}, minus={}
stock modes:    existing behavior
```

Then `boost_target_energy_kj` can remain the shared calibration/recon primitive.

Replace exact-only uses of `peptide_essential_energy_kj` in generic production/swarm paths with a
boost-type-aware Total-channel helper. In particular, the swarm frame measurement must record
`S+E2` for RS, not call the exact `E0-E1+E2` helper.

### 9.5 Physical energy and conventional integration

Generalize every helper that currently detects only `PepGaMDWaterOnlyNonbonded`:

- `physical_energy_groups_for_args`;
- `physical_potential_energy_kj`;
- `make_cmd_integrator(..., system=...)`;
- measurement-force discovery used by validation.

For either Pep-GaMD kernel, group 1 is excluded from physical energy and from conventional
integration.

### 9.6 Lambda ladder and exchange

No change to the dependent-dual closed-form mathematics is required.

All exchange and MBAR paths keep consuming:

```text
v_pep_kj_mol = raw Total coordinate
v_dih_kj_mol = raw Dihedral coordinate
frozen envelope
state lambda
```

The RS kernel merely changes how `v_pep_kj_mol` is measured.

The existing one-place ladder reconstruction through `apply_ladder_boost_to_u` remains the desired
architecture.

### 9.7 NPT biased-MC adapter

Add a dedicated RS target adapter.

The current exact adapter cannot be reused unchanged because its Total coordinate is
`E0-E1+E2` and its force-role classifier recognizes only the exact water-only auxiliary.

For RS, classify the group-1 surrogate components explicitly as `measurement`, not `bias`.

Endpoint evaluation:

```text
physical = energy of all physical groups, excluding group 1
bias     = umbrella/secondary/nonphysical bias groups, excluding group 1
S        = energy1
E2       = energy2

b_D = lower_bound_boost(E2)
b_T = lower_bound_boost(S + E2 + b_D)

boost     = b_D + b_T
effective = physical + bias + boost
```

The surrogate/measurement energy is diagnostic only and is never added directly to `effective`.

Evaluate this same definition at both old and trial box/coordinates. This is mandatory for exact NPT
sampling.

### 9.8 Kernel and resume identity

`kernel_identity_for_run` already hashes `gamd_boost_type`. Therefore the distinct RS boost type
already yields a different kernel digest.

Do not add a redundant second identity string solely for RS.

Add tests that:

- exact and RS kernel digests differ;
- a resume cannot append RS samples to an exact segment or vice versa;
- surrogate flavour metadata is frozen so S1/S2/S3 cannot be silently mixed if they share the same
  top-level boost type.

## 10. Test matrix

### 10.1 Builder and group invariants

- exact and RS partitions are mutually exclusive;
- repeated RS partition construction is idempotent;
- every RS component is group 1;
- no unrelated custom force is allowed in groups 0-2;
- `pep_gamd_bias_force_groups` excludes group 1;
- physical-energy helpers exclude group 1;
- cMD integration excludes group 1.

### 10.2 Surrogate energy construction

On a small solvated peptide fixture:

- interaction-group pair accounting has no duplicates;
- all physical peptide-involving exceptions are excluded from the custom nonbonded term;
- nonzero physical exceptions are reproduced by `PepGaMDRSExceptions`;
- copied bond/angle terms match the selected physical terms;
- group-1 energy equals the explicit component sum.

### 10.3 Force algebra

With both boost channels live (`0 < FSF < 1`):

- finite-difference `-grad U*` agrees with the implemented effective force within the established
  Reference-platform tolerance;
- a warmed deterministic step moves atoms, so the gamd-openmm first-step inertness cannot make the
  test vacuous;
- an independent algebra oracle is used rather than the integrator's own helper.

### 10.4 Zero-boost identities

At `k0_Total = k0_Dihedral = 0`:

- RS integrator trajectory matches the physical system on the Reference platform after the required
  warm step;
- physical potential is unchanged by presence of the surrogate measurement forces;
- biased-MC `U*` equals physical + umbrella/secondary bias and does not contain `S`.

### 10.5 Recording and closed-form reconstruction

For every tested frame:

```text
recorded v_pep == energy1 + energy2
recorded v_dih == energy2
pep_gamd_boost_kj(recorded values, lambda, envelope)
    == integrator-applied boost
```

within 1e-6 kJ/mol or the tighter established numerical tolerance.

If `v_s_kj_mol` is stored, verify `v_s == energy1`.

### 10.6 Exchange and MBAR

- RS exchange state-bias matrix uses the same reconstructed boost as the integrator;
- the existing thermodynamic-validity/reweighting oracle is extended with an RS boost;
- recovered equilibrium FES/distributions match the known physical target within uncertainty;
- lambda=0 remains a true zero-boost state.

### 10.7 NPT

- old/trial endpoint `U*` matches an independent RS oracle;
- group-1 measurement energy is never counted directly in `U*`;
- volume scaling updates the surrogate through the trial context;
- zero-boost NPT reproduces the physical biased-MC target;
- `SUPPORTED_BIASED_MC_BOOST_TYPES == LADDER_BOOST_TYPES` or the existing synchronization invariant
  remains satisfied.

### 10.8 Resume and provenance

- exact -> RS resume fails;
- RS -> exact resume fails;
- RS flavour mismatch fails;
- new campaign with RS succeeds with a fresh envelope.

## 11. Performance gate

Measure with the existing `c8_integ_bench*.py` harness:

1. exact Pep-GaMD, no CV forces;
2. RS Pep-GaMD, no CV forces;
3. exact production CV forces;
4. RS production CV forces;
5. single context per L40S;
6. production-like MPS occupancy, including the current high-context-per-GPU regime.

The expected upper-bound trajectory is from about 1,978 steps/s toward the one-PME regime represented
by about 3,030 steps/s before production CV cost. A roughly 1.5x gain is plausible from deleting the
second full PME; larger gains are possible if the new update also eliminates unnecessary full-group
passes.

Do not claim the gain until measured. The physical CV forces are unchanged by this proposal.

## 12. Main risks and safeguards

- **Surrogate misses long-range peptide physics.** This reduces acceleration quality, not formal
  reweighting validity. Guard with offline correlation/force tests and a boosted pilot.
- **Cutoff discontinuity.** A cheap surrogate with a bad force discontinuity can destabilize the
  boosted dynamics. Prefer a smooth candidate or reject it at the force/NVE gate.
- **Measurement force accidentally becomes physical.** Group-1 invariant, physical-energy tests and
  NPT tests are mandatory.
- **Bare S recorded as v_pep.** This would omit E2 from the dependent Total channel. Lock the
  `v_pep = S+E2` invariant in tests.
- **Exact-only helper survives in an RS path.** Generalize calibration, swarm, physical-energy,
  cMD and NPT measurement discovery; search the repo for `E0-E1` assumptions before merge.
- **Surrogate flavour changes mid-campaign.** Freeze flavour metadata and refuse resume mismatch.
- **Non-Gaussian dV.** Re-run the existing anharmonicity diagnostics on the RS envelope.
- **Folded-region exploration is weakened.** Include compact/folded-like seed-bank regions in both
  the offline and boosted pilot gates; do not judge only on pooled statistics.

## 13. Decisions from review

1. **Validity:** accepted. An exact peptide PME decomposition is not required for equilibrium
   correctness if the propagated and reconstructed boost are identical.
2. **Force group:** resolved. Reuse group 1 as the measurement channel; do not allocate a new
   surrogate group.
3. **Boost name:** resolved. Use `pep-gamd-lower-dual-rs` so the present lower-bound dispatch remains
   compatible.
4. **Recording:** resolved. Keep `v_pep_kj_mol = S + E2`. Optional `v_s_kj_mol = S` is diagnostic.
5. **NPT:** resolved. Add an RS-specific effective-potential adapter and classify group 1 as
   measurement, never bias.
6. **S1 choice:** start with S1-direct as the diagnostic closest-to-exact candidate and evaluate a
   smooth S1 variant in parallel; promote the cheapest smooth candidate that clears the gates.
7. **Acceptance:** correlation thresholds are prefilters. Final promotion uses short boosted
   thermodynamic/sampling diagnostics.
8. **Exact comparison in production:** rejected. Do not keep the second PME alive merely for
   diagnostics; compare exact and surrogate offline on saved frames.
9. **Kernel identity:** the existing digest already includes `gamd_boost_type`. Add resume/flavour
   tests rather than a redundant identity field.

## 14. Whole-codebase board review (2026-09-23) and what it means for this kernel

A seven-board adversarial review ran over `main` @ dfeb8f9 from 00:59 to 04:35. Each board had five
judges (glm, kimi, deepseek-thinking, mini, thinker), a debate round, a chair and a math/physics
veto. Six boards covered code subsystems and one covered the method. The consolidated report is
`bigboard-review/README.md` (this folder); each board's judgment, summary and
full transcript are in the sub-folders next to it. Every headline claim below was re-checked against the code or the OpenMM runtime.

### 14.1 Verdicts

| Board | Subsystem | Result |
|---|---|---|
| b1 | Pep-GaMD integrator, boost hand-off, pricing | ACCEPT-WITH-CHANGES (75), split 3-2 |
| b2 | Replica exchange | ACCEPT-WITH-CHANGES (74), split 3-2 |
| b3 | NPT biased-MC barostat | REJECT (78), split 3-2; its main pillar is refuted (§14.3) |
| b4 | Analysis / MBAR | the chair's output was truncated; judges 2 REJECT / 2 ACCEPT-WITH-CHANGES / 1 n/a |
| b5 | Swarm, envelope, sidecar hand-off | REJECT (80), split 4-1 |
| b6 | Checkpoint / resume / recalibration | the chair's output was truncated; judges 4 of 5 REJECT |
| b7 | Method as a whole | SOUND WITH CONDITIONS, value SIMPLIFY (74), split 4-1 |

Every board agreed that the core mathematics is correct:
- the Pep-GaMD force algebra is the exact gradient of the boosted potential;
- `_channel_boost` reproduces the gamd-openmm lower-bound kernel;
- the gibbs-walk Metropolis-Hastings kernel is in detailed balance;
- MBAR over (window, rung) states with a λ=0 anchor is exact.

The failures sit in **hand-offs and silent fallbacks around that core**, which is the chignolin_8
stage-2 class.

### 14.2 Verified findings this kernel must not inherit

| # | Finding | Location | Requirement for `pep-gamd-lower-dual-rs` |
|---|---|---|---|
| F1 | No runtime check that the applied boost equals the priced boost. `sample()` holds both and never compares them. | `gareus/production.py`, sample path | **Mandatory.** On every sample, assert that the integrator's boost global equals `pep_gamd_boost_kj(v_pep = S+E2, v_dih, λ, env)` within tolerance, and fail loudly on a mismatch. This is the runtime twin of the §10 consistency test. The RS kernel changes exactly the quantity this check guards. |
| F2 | NaN becomes zero, in exchange only. `umbrella_bias_matrix_kcal` zeroes a non-finite secondary displacement, and `_channel_boost` maps a NaN energy to zero boost. MBAR propagates NaN for the same frame. | `production.py` L914-916; `pep_gamd.py` `_channel_boost` | One shared missing-data policy for exchange, NPT U* and MBAR. A non-finite `S` must raise or exclude the frame everywhere, never price as 0. |
| F3 | The swarm step overwrites the CV-selection result. `selection` is reassigned to the seed selection (never `None`), so the 1-D sidecar branch always writes `cv2: "none"`. | `gareus/swarm/analyze.py` L601 / L715 / L740 | Fix before chignolin_9's swarm, otherwise a hand-configured CV2 silently becomes 1-D. It does not affect `cv2: auto` pair layouts. |
| F4 | Native boost reader swallows exceptions; the fallback `infer_gamd_boost_kj_from_globals` returns only the last "total"-like global, dropping the dihedral channel. | `production.py` L344-365, L382-409 | Recorded boost columns for the RS kernel must come from the native reader or fail; no heuristic fallback. |

### 14.3 Reported by the board, not yet re-checked

- **b5:** the epoch-0 marker does not digest `shared_gamd_setup_globals.json` or the seed bank, so a
  changed frozen envelope passes the gate. Directly relevant: this kernel's envelope differs from
  the exact one, so the marker must bind the envelope digest and the surrogate flavour.
- **b5:** the FSF floor is reported but not gated. The forced top rung (`lambdas[-1] = 1.0`) is
  never re-checked for adjacent-rung acceptance.
- **b5:** `done.get("status", "ok")` pools members without a status as successful.
- **b6:** resume can relabel states without validating the window/λ table.
- **b4:** λ per state is inferred by `nanmedian` rather than read from one authoritative source.
- **b2:** `sample()` and `_current_exchange_arrays()` use different predicates for the secondary
  axis.

**Refuted:**
- **b3's main pillar**, that `Simulation.currentStep` does not advance on `integrator.step()`, is
  false. It is a property returning `context.getStepCount()` in OpenMM 8.5.1 and 8.3.1; tested.
- **b7's dissent**, that the barostat omits the PV term, is false. `npt.py:274-276` is
  `-β(ΔU* + PΔV) + N_mol ln(V_new/V_old)`.

### 14.4 Method-level verdict (b7), as it bears on this spec

**SOUND WITH CONDITIONS / SIMPLIFY.** The board regards the Pep-GaMD + λ-ladder + custom barostat +
adaptive-epoch stack as unearned complexity for chignolin. It holds that no chignolin FES may be
quoted until these validation gates pass:
- the overlap matrix and ESS;
- identical-rung consistency;
- PMF invariance under window removal;
- agreement with a ≥100 μs unbiased reference or the published FES;
- a known-answer benchmark in CI.

Consequence for this spec: the RS kernel is a speed optimisation of the boosted path. It should be
built only if the boosted path is kept at all. Its promotion gate (§7, §11) should include the b7
validation gates, not only throughput and correlation. One correction to b7: its "native NPT"
recommendation applies only to an unboosted method. Under any GaMD boost, the biased-MC barostat
is required.
