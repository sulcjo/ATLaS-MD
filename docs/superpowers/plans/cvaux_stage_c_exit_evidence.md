# CVaux Stage C exit evidence (Task 14, CPU)

These are the clean runs of 2026-10-09 at commit `fceca0e`. Scratch dir: `S=/home/sulcjo/.claude/jobs/969c720f/tmp/cvaux_stage_c_e2e`. Nothing is under `RUNS/`.

## SPEC DEVIATION: the gate passes structurally only, not bitwise and not within any tolerance

The spec gate compares a restarted run with its uninterrupted control. It expects equal ledgers and `max_abs_dz = max_abs_dtorsion = 0` on CPU. That comparison passes in none of the configurations below.

- **Across run directories (the brief's `compare_runs(control, resumed)`).**
  - `ok = False` everywhere: ledgers unequal, `(step, replica, window)` key sets unequal.
  - `max_abs_dz` is 1.75 (A), 1.94 (B), 1.77 (C) and 1.98 (D).
  - `max_abs_dtorsion` is 2.14 (A), 2.65 (B), 2.49 (C) and 3.08 (D) rad.
  - Cause: two fresh builds with `--seed 7` already differ before MD. `01_solvated_start.pdb` differs in hydrogen placement, so separately built dirs can never agree (F2).
- **In the same directory (controller ruling 2).** The crashed parent's rows past the rollback step serve as the uninterrupted control for that interval. They are compared against the resumed child's rows at the same steps:

  | Config | Interval (absolute) | Rows | cv1 max\|d\| (Å) | aux z max\|d\| | torsion max\|d\| (rad) | potential max\|d\| (kJ/mol) | Exchange decisions equal |
  |---|---|---|---|---|---|---|---|
  | A | 2000–3500 | 24/24 | 0.164 | 1.48 | 1.72 | 399 | yes (12/12) |
  | B | 2000–3500 | 24/24 | 0.185 | 1.00 | 1.46 | 433 | no |
  | C | 16000–17500 | 24/24 | 0.197 | 0.93 | 1.76 | 404 | no |
  | D | 2000–3500 | 24/24 | 0.154 | 1.56 | 1.80 | 324 | yes (12/12) |

  - Divergence starts at the first sample after the resume (250 steps), so no tolerance short of "anything" passes.
- **Why: F3, a legacy residual, not fixed.** A Context-level probe (`ckpt_determinism.py`, no gareus code) shows two things:
  1. On OpenMM Reference, `loadCheckpoint` into a fresh Context reproduces continuation exactly (max |dx| = 0 nm), with or without a `MonteCarloBarostat`.
  2. On CPU, OpenMM itself is not reproducible: 1.9e-3 nm without and 7.5e-4 nm with a barostat, after 250 steps.

  A gareus aux crash/resume on Reference still diverged (cv1 0.069 Å at the first sample; 0.158 Å max over 2000–3500). It did so even with the fresh System built from the same PDB as the resume (the PDB-reload experiment, since reverted). So the residual divergence comes from how gareus rebuilds the resumed run, not from OpenMM's checkpoint.
  - It could not be shown separately for a legacy run. The crash hook is aux-gated (ruling 19), and a graceful SIGTERM checkpoints at the stop step, which leaves no rolled-back rows to compare.
  - Suspects to investigate (not done): the resumed Context build (PME grid derived from the creation-time box, the NPT/barostat controller state), and post-load re-applies.

## What passes: the structural gate

Every check below holds for A, B, C, D, `RUN_A_R` (SIGTERM) and `RUN_A_F4` (refuse → fix → resume), each against its config's control:
- 96/96 sample rows;
- 0 duplicate `(step, replica)` keys;
- `(step, replica)` key set equal to the control's;
- complete ledger (12 exchange steps x 4 events, equal to nrep, `exchange_max_pairs_per_interval` 0);
- 0 duplicate `(step, attempt_seq)`;
- ledger steps equal to the control's;
- one `state_definition_sha256` across parent and child snapshot;
- finite `u_nk` (96, 4) from `load_parquet(allow_ineligible_aux_segments=True)`.

## Re-seal scope (ruling B4)

The re-seal is auxiliary-only. The legacy double count after an exception crash and a resume is a known residual, left for a separate PR.

Parent `end_step`, from `segments.json`:

| Run | After crash | After resume |
|---|---|---|
| A, B, D | 3500 | 2000 |
| C (calib 14000) | 17500 = 3500 + calib | 16000 = 2000 + calib |
| `RUN_A_R` (graceful) | 3250 | 3250, nothing to cut |

The child starts at the checkpoint step every time.

**F4.** A refused resume no longer leaves an orphan segment, and the re-seal walks back to the checkpoint's own segment. In `RUN_A_F4`, a resume with `--temperature-k 301` was refused with `state_definition_sha256 changed ... first differing fields: state.temperature_k: 300.0 -> 301.0`. Its `segments.json` sha was identical before and after (`08787da55c5eb7dd`), and `windows/` still held only `seg_001.json`. The following plain resume completed with 0 duplicate keys and a complete ledger.

## Fresh vs resume identities (same in the parent and child snapshot of every config)

| Config | state_definition | physical_system | CV1 def | CV2 def | boost envelope | ensemble / box |
|---|---|---|---|---|---|---|
| A | 98cdd2e662044b2f | c0ee726fb6750b40 | af5b3f5c9a6dad2d | (none) | (none) | NPT |
| B | 5eecb61d0efe3c62 | c0ee726fb6750b40 | af5b3f5c9a6dad2d | d3828c267f781087 | (none) | NPT |
| C | 18f4542b0e527bce | c0ee726fb6750b40 | af5b3f5c9a6dad2d | b41079f7fd26db4c | bf017a8969fdd7b1 | NPT |
| D | 4bd419c61af07637 | c0ee726fb6750b40 | af5b3f5c9a6dad2d | (none) | (none) | NVT, fixed box 2.7513893556571216 nm cubic |

- The topology identities are the same in all configs, both read from `01_solvated_start.pdb`:
  - `canonical_topology_sha256` = `645bffede48ae7ee3b511aa1e179d67ff3b313c374efaf0a065819e65e1a3cbf`, which equals the kernel `aux_topology_sha256` and the model binding.
  - `topology_identity_sha256` = `a13d139017d29df60f8636761420c025b54b5d4548153d8eac34e547b2462a61`, which equals the checkpoint's `aux.topology_sha256`.
- The kernel `aux_model_sha256` is `d7285ee10503ad0d…`.
- The physical hash is now order-insensitive (F1 fix), so it is the same for fresh and resumed Systems in every config, and the same across builds.

## Other carry-overs
- **NPZ validators on every aux run** (control and resumed, all configs):
  - `analysis_metadata_validation.json` and `legacy_1d_mbar_validation.json` are `skipped`;
  - `final_report.json.gamd_reweighting_diagnostics` is `skipped`, with reason "auxiliary run: the legacy analysis_arrays.npz export is refused …";
  - `final_report.json` status is `ok`.
- **Non-aux control `RUN_P`.** It loads 0 `gareus.auxiliary_cv` modules (`-X importtime`). Samples have 12 columns and exchanges 7, with no aux or torsion column. The checkpoint manifest and `windows/seg_001.json` have no aux key. `run_manifest.resolved_args` carries only the default-valued provenance keys `aux_cv_model: None`, `aux_equilibrium_eligible: False` and `aux_phase_kind: pilot` (ruling 17).
- **main()-level refusals.**
  - `RUN_A_X_ref` (aux campaign, no model): "resume it with the same --aux-cv-model". No segment was added.
  - `RUN_P_ref` (non-aux campaign, with model): "ran without auxiliary-CV states". The manifest does NOT gain `aux_cv_model_sha256`, which is the refused-resume stamping ruling (supersedes the brief). No segment was added.
- **Config C.** `calib_steps` = 14000 is a multiple of `--traj-interval` 500, so no rerun with another interval was needed.
