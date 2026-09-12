# NPT Package A Report — physics core

Branch `fix/npt-biased-mc-barostat`, worktree `.claude/worktrees/npt-barostat`.
Governing spec: `docs/superpowers/specs/2026-09-12-npt-correction-design.md` (Package A = physics core: `gareus/npt.py` internals, `make_npt_target_adapter` in `gareus/pep_gamd.py`, and their tests).

## What was implemented

| Commit | Content |
|---|---|
| `d022741` | Contract freeze + spec (pre-existing on this branch) |
| `53ced2e` | `resolve_npt_backend` + `count_native_barostats` in `gareus/npt.py` |
| `6869c15` | `BiasedMCBarostatController` (see below) |
| `e4ea064` | Acceptance-math gate (ideal-gas volume ensemble vs scipy Gamma) |
| `f0a8443` | NPT target adapters (`make_npt_target_adapter` + 3 adapter classes) |
| `92214d3` | Coupled volume/energy target gate with negative variants |
| `2eb89b8` | Triclinic isotropic-scaling test + boost-type literal sync test |

### `gareus/npt.py` (contract frozen — no public signature/dataclass/constant changes)

- `resolve_npt_backend(ensemble, requested, run_mode, boost_type)`: `auto` selects the biased-MC controller for `npt` + a supported boosted run mode + a supported boost type; native OpenMM barostat for conventional MD; explicit failure otherwise.
- `count_native_barostats(system)`.
- `BiasedMCBarostatController`: application-controlled isotropic Metropolis volume move.
  - Molecules via `context.getMolecules()` (virtual sites stay bonded to parents); arithmetic-centroid translation by `(s−1)·centroid`; **isotropic** triclinic scaling (all box components × the same factor — tested).
  - Snapshot reads fresh `stage`, `threshold_energy_*`, `Vmax_*`, `Vmin_*` and the **actual** `k0_*` globals (already λ-scaled — evaluating at λ=1; applying λ again would be the double-λ bug, which a test detects).
  - Acceptance: `logA = −β[U*'−U* + P(V'−V)] + N_mol·log(V'/V)`, accept iff `log(u) < min(0, logA)`. Rejection consumes RNG draws (stream parity with accept). Nonfinite old energy aborts; nonfinite trial energy → counted rejection; restoration failure is fatal (exact restore verified).
  - PCG64 RNG with JSON-safe state; `_molecule_fingerprint` (sha256) guards ownership; cutoff-domain guard via `_min_face_height_nm` (conservative 2×cutoff face-height); half-width fixed absolute nm³ from `volume_step_fraction × starting volume`.
- `SUPPORTED_BIASED_MC_BOOST_TYPES` kept in sync with `pep_gamd.LADDER_BOOST_TYPES` by test (two literals for one concept; drift fails loudly).

### `gareus/pep_gamd.py` — NPT target adapters (appended section)

- `_npt_lower_bound_channel_boost` — fresh reimplementation of the upstream lower-bound boost `b = 0.5·k0·(θ−E)²/(Vmax−Vmin)` with both upstream guards (zero if `|Vmax−Vmin| ≤ 0.001·max(|θ|,|E|,1)`; zero unless `E + b < θ`, strict). Deliberately **not** the same code as the test oracle (`pep_gamd._channel_boost` / `pep_gamd_boost_kj`).
- Dependent ordering reproduced: `b_d` on E2 first, then `b_t` on `V_pep + b_d`, where `V_pep = E0 − E1 + E2`, `V_d = E2`, `Δ = b_d + b_t`. Stages 1, 2 (and −1) → zero boost; stages 3, 4, 5 → boost active.
- `PepGamdLowerDualNptTargetAdapter` (`pep-gamd-lower-dual`): validates the auxiliary water-only force exists in group 1, `TOTAL_ENERGY_PLUS_GROUPS == {0,2}` / `MINUS == {1}`, dihedral group == 2, required globals present, and refuses bias-class forces in groups 0..2.
- `LowerDihedralNptTargetAdapter` (`lower-dihedral`): no Total channel, no auxiliary force.
- `ConventionalNptTargetAdapter` (`conventional`): zero boost, excludes the auxiliary energy.
- `make_npt_target_adapter(system, integrator, args)` dispatch.

## Gates (spec §10) — verbatim results

### 1. Independent acceptance mathematics — PASS

`python -m pytest tests/test_npt_acceptance_math.py -q`

```
3 passed in 13.72s
```

Printed by the tests themselves:

```
[thin=31, n=759] KS vs Gamma(shape=6, scale=4.500000): D=0.0270 p=0.6265
[thin=31] KS vs Gamma(shape=16): D=0.8793 p=0
```

- Ideal gas of 5 rigid 3-atom molecules sampled by **real MD through the real controller** (real OpenMM Context, real Metropolis volume moves) vs scipy `Gamma(shape=6, scale=1/(βP))` — passes after thinning by measured integrated autocorrelation (τ_int ≈ 15; the unthinned KS fails from correlation, which is why thinning is measured, not guessed).
- Negative control with the atom count (shape 16 instead of 6) fails decisively — an accidental atom count cannot pass.
- Pressure-units anchor (mean volume in pressure units) and no-hidden-MD-time checks included.

### 2. Boost and force agreement — PASS

`python -m pytest tests/test_npt_pep_adapter_boost.py -q`

```
24 passed, 1 warning in 23.33s
```

On the real solvated-dipeptide fixture (GA dipeptide, 2.4 nm TIP3P, PME, CPU platform): boost agreement vs the integrator's own `BoostPotential_*` globals at k0 combos (1,1), (0.5,0.7), (0.28,0.2); agreement vs the λ-ladder closed form; λ=0 exact zero; **double-λ detection**; dependent ordering; inactive/small-range channels; stage transitions including stage-4 post-recalibration; energy split physical/aux/bias/boost; finite-difference forces vs the FSF expression `(FSF_T−1)(f0−f1) + (FSF_T·FSF_D−1)f2`; dispatch; stock `lower-dihedral` adapter. Oracle (`pep_gamd._channel_boost`) is independent of the implementation (`_npt_lower_bound_channel_boost`).

Stage-4 subtlety covered: the integrator recalibrates k0/threshold **during** the step (after computing BoostPotential), so the adapter reading post-step globals reproduces the boost the *next* step will use — the correct semantics for a move between steps. Stage-4 tests use a closed-form oracle on post-step globals and arm `Vavg_*`, `sigmaV_*`, `windowCount=0` (windowCount=1 crosses the ntave boundary and replaces statistics; upstream can then yield a negative `k0_Total`, which is upstream's actual behavior, reproduced, not smoothed over).

### 3. Coupled volume/energy target — PASS

`python -m pytest tests/test_npt_coupled_target.py -q`

```
4 passed in 14.55s
```

Printed by the tests:

```
correct model: n=1014 D=0.0227 p=0.6649
no-boost variant: vs correct target D=0.1794 p=1.96e-33; vs its own (wrong) target D=0.0170 p=0.88
aux variant (lambda=0): vs correct target D=0.2025 p=1.58e-42; vs its own (wrong) target D=0.0212 p=0.663
aux variant (lambda=1): vs correct target D=0.4203 p=1.95e-174; vs its own (wrong) target D=0.0192 p=0.812
```

Analytic model with volume-dependent E0/E2/W/aux and dependent dual boost; oracle = independent quadrature. Both deliberately-wrong variants (omit boost; add auxiliary energy) fail against the correct target **and** pass against their own wrong target — i.e. the ensemble test detects the physics error, not just instability. The auxiliary variant fails even at λ=0, as the spec requires.

### Controller mechanics + backend selection — PASS

```
$ python -m pytest tests/test_npt_backend_selection.py -q
25 passed in 0.77s
$ python -m pytest tests/test_npt_controller_mechanics.py -q
28 passed in 0.92s
```

Covers accept/reject state preservation (positions, box, RNG-consumption parity), molecule-centroid translation, triclinic isotropic scaling, one-snapshot-per-trial for both endpoints, exact restore verification, ownership fingerprinting, cutoff-domain guard, backend dispatch incl. explicit failure for unsupported modes, native-barostat counting, and the `SUPPORTED_BIASED_MC_BOOST_TYPES == LADDER_BOOST_TYPES` sync.

### Regression — PASS

Targeted (Pep-GaMD regression trio + all five NPT files):

```
$ python -m pytest tests/test_pep_gamd_boost.py tests/test_pep_gamd_wiring.py tests/test_gamd_boost_default.py \
    tests/test_npt_backend_selection.py tests/test_npt_controller_mechanics.py tests/test_npt_acceptance_math.py \
    tests/test_npt_pep_adapter_boost.py tests/test_npt_coupled_target.py -q
114 passed, 1 warning in 50.10s
```

Full suite (`--ignore=tests/test_validation_common.py --ignore=tests/test_validation_script_manifests.py`, per the standing exclusion):

```
1 failed, 2858 passed, 3 skipped, 14 warnings in 1573.22s (0:26:13)
```

The single failure is **pre-existing and unrelated** to Package A: `tests/test_example_configs.py::test_example_config_parses_without_unknown_keys[chignolin_genpept_contact_bias_sigma.yaml]` — `SystemExit: 2`, `gareus: error: the following arguments are required: --seq`. Verified identical on the base commit (before any Package A work):

```
$ git checkout d022741~1 && python -m pytest tests/test_example_configs.py -q
1 failed, 5 passed in 0.90s
```

(Worktree restored to the branch afterward; the unrelated pre-existing stash in this repo was untouched.)

Also: `python -m py_compile gareus/npt.py gareus/pep_gamd.py` clean; `git diff --check` clean; `_gamd_reference/` confirmed git-ignored and unmodified.

## What was NOT done, and why

- **L40S GPU performance gate (spec §10 "Performance")** — infeasible in this environment (no L40S access here). No benchmark numbers are claimed; none were fabricated. Must be run where the four-GPU rig lives.
- **"Real OpenMM behavior" gate, native-barostat comparison arm** — the acceptance-math gate does run real MD through the real controller on a real Context (rigid multi-atom molecules, real PME fixture in the adapter tests), but the head-to-head "application-controlled zero-boost sampler vs conventional physical-System native `MonteCarloBarostat`" small-timestep agreement study was not performed. It needs production-scale systems and belongs with the integration work.
- **Transactions/scheduling at production level, resume/ownership, production validation conditions** — these gates exercise the production loop (500/250/100 schedules, checkpoint/resume streams, replica affinity), which is **Package B** territory (`gareus/production.py` and friends are explicitly out of Package A scope). Controller-level state-preservation and scheduling mechanics *are* covered (see above); the production-loop integration is not.
- No changes to `gareus/production.py`, `gareus/system_setup.py`, `gareus/config.py`, `gareus/cli.py` — per the package split.

## Judgement calls the spec left open

1. **Stock-path physical/bias split is best-effort.** On the stock gamd path the factory puts everything in group 0 (umbrella squashed to group 0), so the physical/bias energy split is best-effort there; the *effective total* (what the acceptance actually uses) is always exact. The Pep-GaMD path has the exact three-way split (physical / auxiliary / boost).
2. **`gamd-cmd-base` treated as unsupported** for the biased-MC controller despite k0≡0 — it never develops boost, so the conventional adapter is the correct route; refusing it here keeps `SUPPORTED_BIASED_MC_BOOST_TYPES == LADDER_BOOST_TYPES` (one concept, one literal set, test-enforced).
3. **Stage −1 treated as cMD** (zero boost), matching stages 1–2.
4. **First volume attempt due at `frequency_steps`** (not at step 0).
5. **`n_molecules` validation covers all particles** — `getMolecules()` partitions every particle, so the Jacobian term is exact for virtual-site systems too.
6. **Conservative 2×cutoff face-height guard** for the cutoff-domain check — stricter than the minimum viable guard; a trial that would thin any periodic face below 2× the nonbonded cutoff is rejected up front rather than risking a corrupted PME evaluation.

## Spec criticisms

- §10's "Real OpenMM behavior" gate bundles several distinguishable things (native-barostat reference comparison, PME corrections, small-timestep agreement, seeds/uncertainty protocol). Package A could only honestly cover part of it; the spec would slice better into controller-level vs production-level sub-gates.
- The spec's §10 ordering implies the performance gate blocks "exposing corrected production NPT", but the GPU rig is not available to the implementing environment — a note about *where* each gate must run would have saved a false-start attempt.
- "Do not use the same helper as both implementation and oracle" is stated only in the boost gate; it applies equally to the acceptance-math gate (controller vs scipy) and was honored there, but the spec doesn't say so.

## How to re-verify

```
python -m pytest tests/test_npt_backend_selection.py tests/test_npt_controller_mechanics.py \
    tests/test_npt_acceptance_math.py tests/test_npt_pep_adapter_boost.py tests/test_npt_coupled_target.py -q
```
