# Thermodynamic repair — task log

Authority: `atlas_md_thermodynamic_repair_spec.md` v1.0 and
`atlas_md_thermodynamic_repair_implementation_plan.md` v1.0 (2026-09-21), both derived from
the implementation adversarial review (pinned `5a9b380`) and re-review (pinned `25cc63a`).
Scientific objective: equilibrium populations and thermodynamics; kinetics irrelevant.

Milestones: M1 = F01–F04 (kernel/initialization repair), M2 = F05–F08 (exploration and
selection repair), M3 = F09 (equilibrium evaluation). M1 is not convergence; M2 is not
confirmation.

## P00 — baseline (2026-09-21)

Starting commit `25cc63aa88acedce75b50017250e05a2b94ad564` (`main`).

Environment, local workstation: Python 3.14.5, NumPy 2.3.5, OpenMM 8.5.1, platforms
Reference / CPU / OpenCL. Aurum2 production env (`conda-envs/calc`): Python 3.9.23,
NumPy 1.24.2, OpenMM 8.3.1, CUDA (L40S). **All new code must run on Python 3.9** (no
`match`, no runtime `X | Y` unions, `from __future__ import annotations` for PEP 604 hints).

Independent re-verification of the review findings against `25cc63a` before any edit:

| Finding | Check | Result |
|---|---|---|
| I01 | `_ss_scalar_from_sub_cv_values` on the real chignolin_8 force's sub-CVs `[1.2884, -0.0001, -0.0793, -0.3324, 736.7403]` | recorded z2 **0.6442** vs true 0.695 (force and positions evaluator agree with each other) |
| N01 | peptide x = 0, 0.5; water O at 0.05, 0.45 | both waters pushed to x = 0.25, O–O **0.000 nm** |
| N02 | peptide x = 0, 0.3; solvent at 0.15 | returns after 3 rounds at 0.182 nm (< 0.22), no status field |
| I02 on real data | chignolin_8 `ladder_design.json` | autotune did **not** fire (`autotuned: None`, centres 0.087–0.915, library q99 0.893) — I02 did not bite this run; I03 does (joint 16×4 grid, no unrestrained state) |

No production sample was ever written by the affected kernel: both chignolin_8 attempts
died before MD (gate failure, then the sidecar handover). The cancelled third job
(2555675) never started.

Baseline suites (plan §24), run detached at `25cc63a` on the local workstation:

| Suite | Result |
|---|---|
| `tests/test_cv_selection_*.py tests/test_epoch0_sidecar_handover.py tests/test_graft_solvent_clash.py` | 207 passed, 1 warning (Biopython import location), 21.8 s |
| `tests/test_exchange_kernel_exact.py tests/test_sample_before_exchange_ordering.py tests/test_npt_production_integration.py tests/test_npt_pep_adapter_boost.py` | 70 passed, 10 warnings, 21.8 s |

The full suite's seven pre-existing failures (docs site-name casing, an example config without
`--seq`, four `beta must be positive` mutation cases) are unchanged from the record in
`equilibrium-cv-task-log.md` and are not part of this repair.

Regression fixtures: `tests/fixtures/thermodynamic_repair/` (see its README).

## F01 (P01 + P02) — residual fast-path dispatch, shared observation/bias seam, kernel identity

Finding I01. Changes (`gareus/production.py`, `gareus/kernel_identity.py`, `gareus/provenance.py`):

* `_ss_scalar_from_sub_cv_values` is an explicit registry (`FAST_SCALAR_MODES`): every mode
  validates its sub-CV count, non-finite inputs raise, an unknown mode raises instead of falling
  through. `residual_scalar_from_sub_cvs` evaluates the frozen force expression from the
  builder's own record of ordered sub-CV roles (`subcv_roles`) and constants
  (`residual_scalar`: K0, K1, K2, sigma_j, norm, mu_c, sigma_c, degree, transform, clamp).
* `fast_cv_path_supported(metadata)` gates activation by mode registry and, for residual mode,
  by `cv_evaluator_version == residual_full_expression_v1`; otherwise the positions evaluator
  is used (and says so).
* `observe_fast_path(ctx, primary_force, ss_force, args, metadata)` and
  `umbrella_bias_matrix_kcal(...)` are module-level and are what BOTH the sample writer and
  the exchange kernel now call; the exchange path gained the non-finite secondary-CV guard
  the sample path already had (a NaN used to freeze that replica's exchanges).
* Kernel identity: `RESIDUAL_EVALUATOR_VERSION = residual_full_expression_v1`,
  `EXCHANGE_ENERGY_VERSION = state_bias_matrix_v2`, recorded in the force metadata and in
  `run_manifest.method_settings`. `_restore_secondary_cv_args_from_metadata` restores the three
  residual artifact paths on resume (the review noted they were never restored) and refuses a
  residual checkpoint whose metadata lacks the current evaluator version.

Regression (tests/test_residual_cv_production_paths.py, 11 tests, Reference platform): the
review fixture reproduces z = 0.8396162908 (faulty 0.0676117650) and E = 1.4559287064
(faulty 0.2700214588) kJ/mol; context energy == positions CV == fast CV == shared matrix at
CV atol 1e-10 / energy atol 1e-6 for degree 1, degree 2 inside/outside the clamp, a missing
psi block, unequal k, negative centres and a k = 0 state; neighbour-swap Delta and Gibbs-walk
normalised probabilities agree with independently set context energies; the NPT adapter's
`bias_kj_mol` moves by exactly the umbrella energy on the solvated dipeptide with the Pep-GaMD
integrator; the mandatory mutation (restoring the two-term average) fails the energy check.

Batches: 113 passed (new file + exchange kernel, ordering, NPT integration/adapter, forces,
e2e, analyze). Driver paths exercised: the module-level functions the closures call; the
closures themselves were inspected, not executed under MD in this batch (the CPU integration
run of the whole sequence is P13). H0 can ship.

## F03 (P05 + P06) — collision-aware solvent repair, transactional graft

Findings N01–N03. `gareus/solvent_repair.py` (new, NumPy only): `repair_solvent_clashes` moves
whole solvent molecules rigidly, never the peptide, and for every candidate move checks the
peptide–solvent clearance (0.22 nm heavy–heavy / 0.16 nm with H) AND an emergency
solvent–solvent clearance (0.16 / 0.10 nm) against every other molecule under the periodic box
(reduced triclinic validated; `safe_radius_nm` bounds where the sequential minimum image is
exact and the repair refuses cells whose radius is below its search cutoff). Directions: radial
escape plus 26 fixed directions; six bounded translation lengths; a molecule that cannot be freed
alone is moved with up to three neighbours; exact coincidence gets a seeded direction. RESOLVED
is declared only by a final independent validation pass (`n_peptide_solvent_clashes`,
`n_solvent_solvent_clashes`, minimum clearance per pair class); otherwise UNRESOLVED with a
reason, or INVALID_INPUT. The cell list is checked against the brute-force oracle.
`displace_clashing_solvent_nm` is a wrapper that now returns a status.

`graft_conformer_into_context` captures the caller's unwrapped positions first and restores
them on every failed path; UNRESOLVED/INVALID_INPUT is a failed graft; after minimisation every
position, force component and the energy must be finite BEFORE the |F|max bound (unchanged
convention: max absolute Cartesian component, 1e5 kJ/mol/nm) is applied — the old guard
accepted a NaN force maximum (N03). Virtual sites are recomputed and excluded from clearance.

Regression: `tests/test_solvent_repair_geometry.py` (two waters 0.40 nm apart between peptide
atoms 0.50 nm apart no longer coincide — O–O ≥ 0.16 nm, brute-force (0, 0); the two-obstacle
fixture resolves past the pair or reports UNRESOLVED; periodic-wall and skewed-cell cases;
INVALID_INPUT cases; cell list vs brute force; virtual sites ride along), and
`tests/test_graft_transaction.py` (real solvated GA-dipeptide graft with a water 0.01 nm inside
the peptide: RESOLVED, E ≈ −20,275 kJ/mol, |F|max 2,320; forced UNRESOLVED rolls the context
back exactly; NaN and Inf injected separately into positions, forces and energy each reject with
their own reason; a failed attempt followed by a healthy one matches a clean context bitwise on
the Reference platform).

## F02 (P03 + P04) — one frozen residual coordinate, versioned artifacts

Finding I04. `gareus/cv_selection/residual_runtime.py::CompiledResidualComponent` is the single
definition: `evaluate_features`, `evaluate_subcvs`, `openmm_expression`, `anchor_partials`
(dz/dc, d²z/dc² inside the clip, zero outside, kinks flagged) and a JSON record that is exactly
the force metadata's `residual_scalar`. `ResidualFit` carries its declared `transform`; degree-2
fits freeze the clip bounds from the training measure (weighted 0.5 % / 99.5 % quantiles)
BEFORE regressing on `[1, T(a), T(a)²]`, so the fitted and deployed coordinates are one function
everywhere; `evaluate_component` lost its `clamp=` override and delegates to the compiled
object; the force builder, the fast-path scalar, the positions evaluator and reprojection all
consume it. Certificate v2 certifies `cov_transformed_anchor_weighted` (exact) and reports
`cov_raw_anchor_weighted`.

Schemas `atlas-cv-selection-candidate-set-v2` / `pair-model-v2` add `basis_transform`,
`design_measure`, `deployment` bindings (topology / physical system / contact-pair digests +
`deployable`) and cross-artifact validation (`validate_pair_semantics`: anchor equality,
selected component present, degree/transform agreement, shared regression parameters, feature
layout compiled by the builder). v1 artifacts are read in explicit legacy mode as the OLD force
definition (bytes and digests untouched, certificate tagged legacy) and refused for deployment
unless `--legacy-model-policy allow-v1`; placeholder bindings can never be deployable.

Regression: `tests/test_residual_coordinate_stage_consistency.py` (independent oracle; tail rows
identical across stages; OpenMM expression == evaluator; finite-difference partials; the
quadratic-curvature fixture yields 4), `tests/test_residual_model_v2.py` (round trip; five
digest-valid semantic mutations rejected; certificate semantics; deployability; legacy v1;
unsupported feature layouts refused).

## F04 (P07) — kernel identity, sample eligibility, marker digests, restart checks

`gareus/kernel_identity.py::kernel_identity_for_run` is recorded in every
`windows/<segment>.json`; `classify_segment_kernel` yields verified / affected / unknown /
not_applicable. `gareus.query.load_samples` — the loader every analysis path goes through —
excludes affected and unknown residual segments by default (conventional and legacy-CV
segments keep their own rules), reports them in `LAST_ELIGIBILITY_REPORT`, and never deletes or
relabels (`include_ineligible=True` reads everything). Epoch-0 completion with `cv2: auto`
requires the sidecar (and the three pair artifacts for a selected pair) and records every
artefact's content digest plus the selection outcome; a present-but-changed artefact blocks a
later job (a missing one keeps the old recoverable re-analyse path). `verify_kernel_identity_on_resume`
refuses a resume whose manifest records a different exchange/evaluator version, or a
residual-CV campaign whose manifest predates kernel recording. Tests:
`tests/test_residual_kernel_resume.py` (incl. a fresh-interpreter check of the epoch-0 verdicts).
Limitation: the real chain restart under MD is exercised only through these seams, not a GPU
job.

## F05 (P08 + P09) — region inventory, exploration-preserving layout

Findings I02, I03. `gareus/cv_selection/coverage.py::build_region_inventory` splits the
discovery CV1 sample into supported intervals (gap > 3 window widths at k_max), keeps rare
regions, counts invalid rows and failed seed preparations separately, and gives each region a
representative that has an exported seed (else marks it unresolved). `region_centres` places
centres inside supported intervals only. `swarm.ladder_design.design_exploration_layout`
reserves one k1 = k2 = 0 spatial state (a real λ = 0 unbiased target) at EVERY cap, then region
representatives, then axis/joint states; 96/4 → 23 others, 256/4 → 63 others (16×4 = 64 joint
cells therefore give the sparse layout, not the old all-joint grid); INSUFFICIENT_REGION_COVERAGE
when the mandatory stacks do not fit. `layout_plan.json` (roles, coverage, mandatory ids,
unresolved) travels beside the physics rows; `coverage_design` gate fails on unresolved regions.
Zero stiffness is written exactly with finite placeholder centres. Mandatory states are refused
by `retire_state`, skipped by `apply_actions` (retire/split), never dropped by the reachability
filter (which also keeps unrestrained windows) or the post-pull auto-drop. The endpoint tuner
survives as a diagnostic only (`legacy_autotune`, with a warning when it would have truncated).
Tests: `tests/test_exploration_layout_invariants.py`, `tests/test_exploration_layout_adaptive_guards.py`.
Not done here: 1-D (cv1-only) ladders keep their previous design; predicting sparse-layout
connectivity before writing the table remains open (campaign-end connected-components check).


## Completion record — M1 + F05 (2026-09-21)

Commits on `main` (pushed 2026-09-21 21:5x CEST): `6daa088` P00 fixtures/log, `0b967c6` F01+F02+F04,
`7e98fc0` F03, `3237298` F05, `ccd73c7` task log + internal 0.9.0 version bump (superseded before tagging by the official v0.8.2 release). Full test suite at `ccd73c7`
(`-q tests/`, two slow smoke items deselected): **3374 passed, 2 skipped, 7 failed**; the 7 failures
are the P00 baseline's pre-existing items (`test_atlas_md_docs` ×2 site_name casing,
`test_example_configs` chignolin_genpept_contact_bias_sigma.yaml without `--seq`, and the four
`beta must be positive` mutation-battery cases in `test_thermodynamic_validity_2d_rough` /
`_real_md`), identified by collection index from the progress dots because the detached run ended
before printing its summary. Targeted batch of the 47 touched/new test files: 835 passed.

Deployment: both aurum2 trees (`/home/sulcjo/2026_peptide_sampler`, `/home/sulcjo/gareus`) rsynced
from the tracked file list, package digest identical to local main (`d0a9a066…`), `compileall` clean
under `conda-envs/calc` (Python 3.9.23), `DEPLOYED_COMMIT = ccd73c7`.

chignolin_8 recovery without re-running MD: the swarm round (186 members) and `swarm/system` were kept;
the v1 analysis and the pool bookkeeping were set aside; `analyze_swarm_stage` re-run on the login
node with the deployed yaml (`swarm_n_windows 15`, `max_replicas 256`) → all six gates pass
(`coverage`, `envelope_stability`, `ladder_ess`, `graft`, `pair`, `coverage_design`), pair =
component 3, degree 1 / identity basis, deployable with real topology / physical-system / contact-pair
digests, one CV1 region [0.04, 0.968], layout PROPOSED joint 62 spatial × 4 rungs = 248 states with
the k1 = k2 = 0 stack (rows 0–3) and 4 region representatives mandatory. CPU pre-flight on the swarm's
equilibrated frame: fast-path z = positions-evaluator z = 0.695036 (the value the I01 finding showed
the old fast path mis-recorded as 0.6442), and umbrella energy on a restrained row equals
0.5·k2·(z − c)² to 1e-6 (9.969554 kJ/mol). Job 2558439 submitted; the job re-analyses (deterministic),
writes `epoch0_complete.json`, charges the swarm to the pool and starts production on the new ladder.

Not done in this pass: F06–F08 (M2), F09 (M3); see the plan for their gates.

## Follow-up F04b — the sidecar's frozen GaMD envelope never reached production (2026-09-22)

Found on the first job of the repaired chignolin_8 (2558439): `ladder_run_args.yaml` names
`gamd.shared_gamd_setup_dir` (the swarm's frozen Pep-GaMD envelope, 16 integrator globals, no
checkpoint) but `apply_epoch0_sidecar` applied only the CV block, and
`run_adaptive_production_auto_loop` had already resolved `shared_gamd_setup_dir` to an empty
`adaptive_production/global_shared_gamd_setup` before the epoch-0 hook ran. The first worker
therefore ran its own 1,010,000-step shared GaMD setup and was about to run the per-state recon
(~6.75 ns x 248 states ~ 1.7 us of a 6 us pool), re-deriving the envelope the frozen-envelope rule
says is measured once; a second job would have loaded that recalibration. The job was cancelled at
2h11m (seeding done, no production MD).

Fix: `swarm.epoch0._apply_sidecar_gamd_envelope` points `shared_gamd_setup_dir` at the swarm
envelope whenever nobody chose one (empty, or the driver's auto-resolved campaign default); an
explicit user directory is kept and the decision recorded; a sidecar naming a directory without
`shared_gamd_setup_globals.json` raises instead of recalibrating silently.
`adaptive_production._seed_global_shared_gamd_from_envelope` copies the envelope into the
campaign-global directory on the first job (never overwriting an existing one), because the hook
does not run once a state registry exists and a worker that reuses a setup never exports one.
Production's reuse path (`load_reusable_shared_gamd_setup`) accepts the 16-global payload, skips
`run_shared_gamd_setup_article_a` and `apply_joint_envelope_gamd_calibration`, and applies the
globals to every integrator; the epoch-0 recalibration is already a hard no-op under a lambda
ladder. Tests: `tests/test_epoch0_sidecar_gamd_envelope.py` (6). Still unapplied from the sidecar,
on purpose: the `starting_structures` pull knobs (its 150,000 pull steps would multiply seeding
time ~15x); the run's own pull settings produced 0 bad / 155 warn starting structures.

## Follow-up — 248 replica contexts on one node: descriptor hang and MPS ceiling (2026-09-22)

chignolin_8 job 2560085 (the first job past the envelope fix) hung at replica 223/248 for 1h41m and
timed out with no checkpoint and no chain resubmission: the hung CUDA call never returned, so the
launcher's TERM handling and cleanup never ran. Reproduced without seeding
(`~/gareus/chignolin/aurum_ctx_test.py`, jobs 2562717 / 2563011 / 2563506 / 2563549): SLURM
propagates the login soft `nofile` limit of 1024 into jobs, the CUDA driver raises it only to 4096,
and each context costs ~18 descriptors, so the 225th context blocks inside `Context()`. With
`ulimit -n 65536` MPS builds 240 contexts and fails at ~241 (`The requested CUDA device could not be
loaded`; about 60 clients per L40S is the ceiling), while without MPS all 248 contexts build in 106 s
and step. Throughput (3000 steps x 3 fs per replica, all replicas stepping concurrently, upper
bound): 64 contexts with MPS 10,940 ns/day, 192 with MPS 7,494 ns/day, 248 without MPS 3,195 ns/day
aggregate.

Decision: keep the user's 248-state joint design and run without MPS. Launcher changes:
`ulimit -n 65536` before python, no MPS daemon, `--cuda-use-blocking-sync false
--cuda-deterministic-forces false` in place of `--cuda-mps` (the flag only set those two soft
defaults). The alternative (cap `max_replicas` at 192 with MPS, 2.3x the aggregate throughput)
changes the state-space design and is left to the user. Job 2563608 launched with this
launcher; `smoke/config/chignolin_8.sh` mirrors it. Known gap: a python process hung in a CUDA call
never returns the walltime TERM, so the chain does not resubmit; the descriptor fix removes the
known hang, but the launcher still has no hang watchdog. A wrongly patched relaunch (2563578, MPS
still on) was cancelled before it left seeding.
