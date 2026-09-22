# Spec: real-space surrogate for the Pep-GaMD peptide energy V_pep

- **Status:** proposal for review (not implemented)
- **Date:** 2026-09-23
- **Code base:** ATLaS-MD / gareus v0.8.2, `main` @ c73224f and later
- **Target campaign:** chignolin_9 (new kernel identity, new swarm envelope; never a resume of an
  existing campaign)
- **Author:** ATLaS-MD development session, for review by Astra

---

## 1. Problem

`PepGaMDLowerDualIntegrator` (`gareus/pep_gamd.py`) boosts only the peptide. It gets the peptide's
nonbonded energy by subtracting a water-only copy of the nonbonded force:

| Force group | Contents | Energy |
|---|---|---|
| 0 | physical `NonbondedForce` (full PME), `HarmonicBondForce`, `HarmonicAngleForce`, `CMMotionRemover` | E0 |
| 1 | auxiliary `NonbondedForce` `PepGaMDWaterOnlyNonbonded`: physical copy with peptide q, epsilon and peptide exceptions zeroed, PME parameters pinned to group 0 | E1 |
| 2 | `PeriodicTorsionForce`, `CMAPTorsionForce` | E2 |
| 29, 31 | secondary-CV restraint, primary contact umbrella (never boosted) | bias |

```
V_pep = E0 - E1 + E2
```

Every MD step evaluates **two full PME forces** (groups 0 and 1) in **separate passes**, because an
OpenMM `CustomIntegrator` computation step may read only one force group.

Measured on the real chignolin_8 system (21k atoms, TIP3P, cutoff 0.8 nm, Ewald tolerance 1e-4),
one context per NVIDIA L40S, no contention (jobs 2577764, 2579032):

| Step | steps/s | vs Langevin |
|---|---|---|
| Langevin, one force evaluation | 4,170 | 1.00 |
| Langevin + auxiliary PME, still one evaluation | 3,030 | 0.73 |
| Pep-GaMD, per-group evaluations | 1,978 | 0.47 |
| Pep-GaMD + production CV forces | 1,461 | 0.35 |

The water-only PME costs about 27 % by itself. Its separate pass, together with the other per-group
reads, costs about another 35 %. The peptide has 138 atoms and the solvent about 21,000, so a second
full-system PME is being spent to recover the energy of a 138-atom subset.

## 2. Correctness argument: the boost need not use the exact peptide energy

GaMD reweighting is exact for **any** boost `dV(x) = g(S(x))`, where `S` is any smooth function of
the coordinates, provided two conditions hold:

1. **The applied force is the exact negative gradient of `U(x) + dV(x)`.** The only thing that
   matters is that the sampled distribution is proportional to `exp[-beta (U + dV)]`.
2. **MBAR / the lambda ladder reweights with exactly the `dV` that was applied**, reconstructed from
   the same recorded `S` value per frame with the same frozen envelope. That is the job of
   `apply_ladder_boost_to_u` and `pep_gamd_boost_kj` today.

`V_pep` therefore only has to **target** the degrees of freedom whose barriers we want to lower. It
does not have to equal the PME peptide energy. How close the surrogate is to the exact `V_pep`
determines **efficiency** (how well the boost flattens the relevant barriers, and how Gaussian dV
stays), not **validity**. This is the central claim for review.

## 3. Proposed energy and force

### 3.1 Surrogate

Define the Total-channel argument with a surrogate `S(x)` in place of `E0 - E1`:

```
S(x)  = E_S(x)                      surrogate peptide nonbonded + peptide bonded energy
Y(x)  = S(x) + E2(x) + dV_D(E2)      Total-channel energy (dual dependent, as today)
dV_D  = 1/2 k_D (Eth_D - E2)^2       if E2 < Eth_D
dV_T  = 1/2 k_T (Eth_T - Y)^2        if Y  < Eth_T
FSF_D = 1 - k_D (Eth_D - E2),   FSF_T = 1 - k_T (Eth_T - Y)   (1 when above threshold)
U*    = U_phys + U_bias + dV_D(E2) + dV_T(Y)
```

The surrogate force lives in its own force group `s`. It contributes **no physical force**: it
exists only to compute `S` and `f_S = -grad S`.

### 3.2 Force algebra (derived, to be checked independently)

```
-grad U* = f_phys + f_bias - (1 - FSF_D) f2 - (1 - FSF_T) (f_S + FSF_D f2)
         = f_phys + f_bias - (1 - FSF_T) f_S - (1 - FSF_T FSF_D) f2
```

Sanity check against today's integrator. Substituting the exact decomposition `f_S = f0 - f1` and
`f_phys = f0 + f2` (the auxiliary force is not physical) recovers the current applied force
`FSF_T (f0 - f1) + FSF_T FSF_D f2 + f1 + f_bias`. So the surrogate is a strict generalisation.

### 3.3 Per-step reads (cost)

| Read | Contents | Cost |
|---|---|---|
| `f` (all integrated groups), `energy` not needed | physical + bias, **one** PME | full pass |
| `f_s`, `energy_s` | surrogate, real-space only, peptide pairs only | cheap |
| `f2`, `energy2` | dihedrals | cheap |

`f` must **exclude** group `s`. This is to be verified: whether `setIntegrationForceGroups` restricts
`f` / `energy` inside a `CustomIntegrator`. If it does not, use `f - f_s`, which costs no extra pass
because `f_s` is read anyway. The applied force is then:

```
F = (f - f_s) - (1 - FSF_T) f_s - (1 - FSF_T FSF_D) f2      if f includes group s
```

That is one PME per step instead of two, and one full pass plus two cheap ones instead of four or
more passes.

## 4. Surrogate candidates

All candidates cover the same pair set: every pair with **at least one peptide atom**
(peptide-peptide and peptide-solvent). Water-water pairs are never included, which is exactly the
set `E0 - E1` measures. Implement them with `CustomNonbondedForce` interaction groups
`{pep} x {pep}` and `{pep} x {solvent}`, as two groups so no pair is counted twice. Mirror the
physical exclusions (1-2, 1-3). Add the peptide 1-4 scaled pairs as a `CustomBondForce` over the
physical exception list, and copies of the peptide's `HarmonicBond` / `HarmonicAngle` terms. The
bonded copies keep `S` covering what `E0` covered for the peptide. Rigid TIP3P water carries no
bond or angle terms, which is to be confirmed on the built system.

| ID | Electrostatics | LJ | Relation to exact E0 - E1 | Notes |
|---|---|---|---|---|
| **S1 (recommended)** | PME **real-space term only**: `q_i q_j erfc(alpha r)/r`, same alpha and cutoff as the physical PME | same as physical (cutoff 0.8 nm, same switch or dispersion treatment, minus the long-range correction) | differs by the reciprocal-space and self terms for peptide-involving pairs, which vary slowly with peptide conformation | closest to exact; alpha taken from `pme_parameters_from_tolerance` |
| S2 | damped shifted force (DSF / Wolf) | same | smooth at cutoff by construction | fallback if S1 shows cutoff artefacts |
| S3 | reaction field (eps_rf = 78.5) | same | classic, cruder | simplest reference |

Chignolin (GYDPETGTWG) carries charged termini plus D3 and E5 side chains. The long-range part that
S1 drops is the part most sensitive to that charge distribution, which is why the correlation test
in §5 is mandatory rather than optional.

## 5. Acceptance criteria before building (offline, no MD)

Use existing frames: chignolin_8 production (plain umbrella sampling over 62 windows, a broad
conformational spread) and the swarm frames. For each frame compute the exact `E0 - E1`, and each
surrogate `S` on the same coordinates (Reference or CUDA platform, single context).

| Metric | Threshold (proposed) | Why |
|---|---|---|
| Pearson r of `S` vs `E0 - E1`, pooled and per window | >= 0.90 pooled, >= 0.80 in every window | the boost acts on fluctuations |
| sigma of `S` / sigma of `E0 - E1`, per window | 0.8-1.25 | sigma_V sets k0 and the Gaussianity of dV |
| force cosine similarity `f_S` vs `(f0 - f1)` on peptide atoms | median >= 0.90 | the boost flattens along `f_S` |
| energy drift / NaN in a 1 ns NVE test with the surrogate boost live | none beyond the physical system's own drift | smoothness at the cutoff |

Pick the cheapest surrogate that passes. If none pass, stay on the exact auxiliary PME; the
method is unchanged and only the speed is lost.

## 6. Implementation outline

1. New boost type `pep-gamd-rs-lower-dual`. The stock `pep-gamd-lower-dual` stays bit-identical.
2. `ensure_pep_gamd_rs_partition(system, peptide_atoms, flavour)`: adds the surrogate forces in a
   dedicated group, never 0..2, and never counted by `pep_gamd_bias_force_groups`. It is idempotent
   and found by name, like the auxiliary force today.
3. Integrator subclass: overrides `_setup_energy_values` (`Y` from `energy_s + energy2`) and
   `_add_gamd_update_step` (§3.2). The conventional-MD stages exclude the surrogate force.
4. **Everything that prices the boost switches to the same `S`:**
   - the recorded per-frame value, which replaces `v_pep_kj_mol` or is stored as a new
     `v_s_kj_mol` column;
   - the exchange state-bias matrix;
   - the NPT biased-MC adapter's U*;
   - `pep_gamd_boost_kj`;
   - swarm envelope calibration.

   Physical-energy reports (`potential_kj_mol`, energy decomposition) exclude group `s`.
5. **New kernel identity string** (F04): the loader refuses to mix surrogate and exact segments.
6. The stage-5 seeding and `verify_gamd_production_stage5` from c73224f apply unchanged.

## 7. Tests

- **Force algebra:** on the Pep-GaMD test fixture with the Reference platform, the applied force
  equals the finite-difference `-grad U*` to 1e-6 relative, with the boost live (0 < FSF < 1 on both
  channels).
- **k0 = 0 identity:** bit-identical trajectory to the physical system without the surrogate force,
  after one warm step (gamd-openmm's first step is inert).
- **Consistency:** the value that MBAR reconstructs (`pep_gamd_boost_kj` from the recorded `S`)
  equals the integrator's own boost global within 1e-6 kJ/mol on every frame.
- **Reweighting oracle:** one existing thermodynamic-validity oracle, extended with a surrogate
  boost. The recovered FES must match ground truth as the exact-boost case does.
- **Exclusion:** physical energy and pressure/barostat U* are unchanged by the surrogate force's
  presence at k0 = 0.

## 8. Expected gain and how to measure

Upper bound per context: from 1,978 steps/s (exact, no CV forces) toward the 3,030 of Langevin plus
one extra PME, i.e. about 1.5x from removing the second PME. Collapsing four or more passes into one
full pass plus two cheap ones may add more, up to about 2x. The CV forces (26 %) are untouched by
this change. Measure with the existing `c8_integ_bench*.py` harness: single context, and 59
contexts per GPU under MPS.

## 9. Risks

- **A weaker boost target.** If `S` misses the long-range part that couples to the peptide's slow
  motions, the boost flattens less of the relevant landscape. §5 guards this.
- **dV non-Gaussianity.** A different `S` gives a different dV distribution, so the anharmonicity
  diagnostics must be re-checked on the first epoch.
- **Hidden energy consumers.** Any code path that still reads `E0 - E1` as the boost argument
  silently mis-prices the boost. The consistency test in §7 and the kernel-identity refusal guard
  this.
- **OpenMM detail.** Whether `f` inside a `CustomIntegrator` honours `setIntegrationForceGroups`
  decides between the two forms in §3.3; verify first.

## 10. Questions for review

1. Is the §2 validity argument complete for the dual dependent scheme with the lambda ladder, and
   for the biased-MC NPT barostat's U*?
2. Is S1 (PME real space only) the right first choice over DSF / reaction field for a peptide with
   net charges?
3. Are the §5 thresholds appropriate, or should acceptance be judged on boost efficiency directly,
   e.g. transition counts in a short boosted pilot?
4. Recording: replace `v_pep_kj_mol` or add `v_s_kj_mol` alongside it (both are cheap)? Recording
   both would allow a later exact-vs-surrogate comparison on the same frames.
