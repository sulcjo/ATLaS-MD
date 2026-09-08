# Pep-GaMD "internal" variant: boost the peptide-internal energy only

Status: bounded change to `gareus/pep_gamd.py` (+ CLI choice, production hooks, tests). Adds boost type
`pep-gamd-internal-lower-dual` next to `pep-gamd-lower-dual`. Nothing about the existing type changes.

## Why
Lower-bound Pep-GaMD scales the peptide–water forces by FSF while water–water stays at full strength. At 4 fs with
HMR this destabilises at k0 ≳ 0.4 (σ0 = 8 and 10 pilots: NaN at boost onset / after 14 ps; the λ = 1 rung's
v_pep shot 1.5 MJ/mol above V_max). At 2 fs without HMR, k0 = 1 was stable but sampling per GPU-hour halves.
If the boosted energy contains **no peptide–water term**, there is no solvent-collapse mode: water always feels
the peptide at full strength. Then k0 = 1 at 4 fs HMR should be safe, and conformational (intramolecular)
barriers are still accelerated. Desolvation barriers are not — state this in helptext.

## Definition
- New auxiliary force `PepGaMDPeptideOnlyNonbonded` in force group `AUX_PEPTIDE_GROUP = 3`: a copy of the
  physical `NonbondedForce` (same method/cutoff/switching, PME parameters pinned with
  `pme_parameters_from_tolerance` exactly like the water-only aux) in which every **non-peptide** particle has
  charge 0 and ε 0 and every exception touching a non-peptide particle has chargeProd 0 and ε 0. Peptide
  particles keep their real parameters. Its energy E3 is the peptide-internal nonbonded energy (with its own PME
  reciprocal self-term, constant across states — fine for a boost). Build it in a new
  `ensure_pep_gamd_internal_partition(system, peptide_atoms) -> int` that is idempotent like
  `ensure_pep_gamd_partition`; it does NOT add the water-only force (group 1) — the internal variant never needs
  it. Force groups: 0 physical (NB + bonds + angles), 2 torsions, 3 peptide-only NB. Group 29/31 unchanged.
- Total channel `V_int = E3 + E2` (peptide-internal nonbonded + peptide dihedrals; mirrors the existing
  `V_pep = E0 − E1 + E2` which also includes E2). Dihedral channel `E2`, unchanged.
- Integrator `PepGaMDInternalLowerDualIntegrator` (same lazy `_integrator_class` pattern, same
  `LowerBoundIntegrator` base, same ntcmd/nteb/ntave machinery): `StartingPotentialEnergy_Total = E3 + E2`;
  cMD update uses `f − f3` (integration excludes group 3; equivalently use `setIntegrationForceGroups` to drop 3
  in `make_cmd_integrator(system=)` when the peptide-only force is present); GaMD update applies
  `f_applied = (f − f3) + f3·FSF_T + f2·(FSF_T·FSF_D − 1)` where `f` is the force over all integration groups
  (0, 2, 29, 31 …; never 1 or 3), i.e. only the peptide-internal nonbonded part and the dihedrals are scaled.
  Write the formula in the class docstring exactly. One force group per CustomIntegrator computation step.
- `is_pep_gamd(args)` stays True for both variants (it gates the partition/integrator path); add
  `pep_gamd_variant(args) -> "essential" | "internal" | None`. `LADDER_BOOST_TYPES` gains the new type.
- `total_energy_groups_for_args`: internal → `(frozenset({AUX_PEPTIDE_GROUP, DIHEDRAL_GROUP}), frozenset())`.
  `physical_energy_groups_for_args` / `physical_potential_energy_kj`: exclude whichever aux groups are present
  (1 and/or 3; generalise `find_aux_force` to both names).
- Samples: production stores the Total-channel energy in the existing `v_pep_kj_mol` column (for the internal
  variant that is E3 + E2); `_fetch_v_pep_v_dih` dispatches on the variant; `run_manifest.method_settings` and
  the envelope JSON record `pep_gamd_variant` so the analysis knows what the column means. The closed-form
  `pep_gamd_boost_kj(v_pep, v_dih, λ, env)` is unchanged (dependent dual boost on the stored Total energy), so
  MBAR/cross-check/ladder code needs no change beyond accepting the new type where it checks boost-type strings
  (grep `pep-gamd-lower-dual` and `PEP_GAMD_BOOST_TYPE` across gareus/ and tests/; every site must accept both).
- CLI: add `pep-gamd-internal-lower-dual` to the `--gamd-boost-type` choices; `_validate_gamd_args` dual-boost
  set includes it; helptext: two sentences in the Pep-GaMD section (what is boosted, what is not, why it exists).
- `make_cmd_integrator(openmm, args, unit, system=)` excludes every aux group present.

## Tests (fixture-free zero-argument functions; `tests/test_pep_gamd_internal.py`; Reference/CPU platform;
never call `getPMEParametersInContext` on CPU)
1. Partition: on the tiny solvated fixture used by `tests/pep_gamd_fixture.py` (reuse it), E3 is invariant when
   every water molecule is displaced rigidly by 0.3 nm (peptide fixed): |ΔE3| < 1e-6 kJ/mol, while E0 changes.
2. Identity: `E0 − E1 − E3` (with both aux forces added to a test copy of the System) equals the peptide–water
   cross energy computed by zeroing the peptide params in a third copy... simpler and sufficient: assert
   `E0 ≈ E1 + E3 + E_cross` where `E_cross = E0 − E_waterOnly − E_pepOnly` is computed from three plain
   NonbondedForce systems built independently in the test (this pins the aux construction, not just its sign).
3. Integrator, boost off (cMD stage): one warm-up step, then N steps of the internal integrator vs a plain
   LangevinMiddle with the same seed on the same partitioned System — positions equal to 1e-6 nm AND atoms
   moved (existing test pattern in `tests/test_pep_gamd_boost.py`).
4. Integrator, boost on: force the integrator into the boosted stage by setting globals (as the existing
   Pep-GaMD tests do), read `ForceScalingFactor_*`, and check one velocity update against
   `f_applied = (f − f3) + f3·FSF_T + f2·(FSF_T·FSF_D − 1)` computed from per-group `getState(getForces,
   groups=…)` calls — max |Δv| < 1e-6 nm/ps.
5. `total_energy_groups_for_args`, `physical_energy_groups_for_args`, `ladder_supports_boost_type`,
   `pep_gamd_variant` for the new string; `_fetch_v_pep_v_dih` returns E3 + E2 for the internal variant (fake
   context with per-group energies).
6. CLI: parsing `--gamd-boost-type pep-gamd-internal-lower-dual` succeeds; `_validate_gamd_args` treats it as
   dual-boost (no sigma0d warning).

## Out of scope
Any change to the existing `pep-gamd-lower-dual` numbers; upper-bound variants; swarm-stage support beyond
accepting the type string (the swarm records `pep_gamd_variant` in the envelope JSON it writes if the flag says
so — one line in `write_envelope_setup_dir`'s meta, optional).

## Global constraints
Ab initio. Conventional commits, no Co-Authored-By or other trailers. Test runner: a repo hook rejects any Bash
command containing the literal name of the Python test runner; use `opencode run "In <worktree>: run the
project's standard test runner on tests/test_pep_gamd_internal.py tests/test_pep_gamd_boost.py
tests/test_pep_gamd_wiring.py tests/test_lambda_ladder_states.py and report passed/failed and failing ids"`
(targeted invocations work; whole-suite ones hang). New code under ~400 lines per module; do not reformat.
