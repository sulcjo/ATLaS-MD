# CVaux Stage C exit evidence (Task 14, CPU)

Scratch dir: `S=/home/sulcjo/.claude/jobs/969c720f/tmp/cvaux_stage_c_e2e`. Nothing is under `RUNS/`.

## Which commit each result comes from
- **Shipping commit `cbd26c7` (fix round 3).** Configs A (`RUN_A3_*`) and C (`RUN_C3_*`), plus the real refuse → fix → resume (`RUN_A3_F4`), were re-run here. Every structural check below holds for all three:
  - parent `end_step`: A 3500 → 2000, C 17500 → 16000;
  - the resumed `seg_002` has `parent_segment_id` `seg_001`;
  - 96/96 rows, 0 duplicate keys, key set equal to the control's;
  - ledger complete (12 x 4), 0 duplicate `(step, attempt_seq)`;
  - one `state_definition_sha256`, finite `u_nk`;
  - physical hash `c0ee726fb6750b40` parent = child.
  - `RUN_A3_F4`: `--temperature-k 301` was refused ("state.temperature_k: 300.0 -> 301.0"). `segments.json` was unchanged (`3cdf3518bb818daf`), only `seg_001.json` existed, and the fixed resume passed.
- **Predates the resume reorder (clean round, `fceca0e`).** Configs B and D, the graceful SIGTERM run `RUN_A_R`, and the first A/C/`RUN_A_F4` runs. Their structural results are below. They were not re-run at `cbd26c7`.
- **Round-2 diagnostics (`78243b2`).** `DIAG_A` and `DIAG_B`.

## SPEC DEVIATION: with default (concurrent) replica stepping the gate passes structurally only

The spec gate compares a restarted run with its uninterrupted control. It expects equal ledgers and `max_abs_dz = max_abs_dtorsion = 0` on CPU.

**What was measured with the default concurrent stepping (configs A–D, CPU):**
- **Across run directories** (the brief's `compare_runs(control, resumed)`): `ok = False` everywhere, with ledgers and `(step, replica, window)` key sets unequal.
  - `max_abs_dz`: 1.75 (A), 1.94 (B), 1.77 (C), 1.98 (D).
  - `max_abs_dtorsion`: 2.14 (A), 2.65 (B), 2.49 (C), 3.08 (D) rad.
  - Two fresh builds with `--seed 7` already differ before MD (hydrogen placement in `01_solvated_start.pdb`). So this cross-directory comparison cannot be met by any resume (F2).
- **In the same directory** (controller ruling 2): the crashed parent's rows past the rollback step against the resumed child's rows at the same steps.

  | Config | Interval (absolute) | Rows | cv1 max\|d\| (Å) | aux z max\|d\| | torsion max\|d\| (rad) | potential max\|d\| (kJ/mol) | Exchange decisions equal |
  |---|---|---|---|---|---|---|---|
  | A | 2000–3500 | 24/24 | 0.164 | 1.48 | 1.72 | 399 | yes (12/12) |
  | B | 2000–3500 | 24/24 | 0.185 | 1.00 | 1.46 | 433 | no |
  | C | 16000–17500 | 24/24 | 0.197 | 0.93 | 1.76 | 404 | no |
  | D | 2000–3500 | 24/24 | 0.154 | 1.56 | 1.80 | 324 | yes (12/12) |

  The first compared sample (250 steps after the resume) already differs. No tolerance was evaluated; none was pre-registered.

## F3: diagnostics consistent with concurrent replica stepping as the cause (hypothesis; Reference, code 78243b2)

**Measured: a non-aux concurrent run diverges after resume, and one aux run with serialised stepping resumes bitwise. Both are consistent with the hypothesis that concurrent replica stepping causes the divergence; neither proves it.**

1. **(a) Non-aux control, concurrent stepping.** Run `DIAG_A`: config F with `W0.csv`, Reference, `--production-steps 4000`, `--flush-every-log`.
   - SIGKILL at production step 3500. The legacy `_parent_was_running` seal then cut the parent to 2000 at the plain `--resume`.
   - Child vs parent rows over 2250–3500: **not bitwise**. cv1 max |d| per sample is 0.163 / 0.047 / 0.142 / 0.084 / 0.036 / 0.087 Å; potential differs by 229–402 kJ/mol. The exchange decisions over the 8 overlapping events were equal.
   - So the divergence exists without any auxiliary code.
2. **(b) Aux config D, serialised stepping.** Run `DIAG_B`: config D, Reference, `--active-replicas-per-gpu 1`. Reference replicas share one admission queue, so at most one replica steps at a time, in 50-step turns. It used `--production-steps 4000`, crashed at 3500 and resumed.
   - Child vs parent over 2250–3500: **bitwise equal in every column** (window_id, cv1, cv2, potential, boosts, v_pep, v_dih, gamd_lambda, tor_000, tor_001, aux_z_00).
   - All 12 exchange events match: equal decisions and `assignment_sha256_after`, delta_e |d| = 0.
   - The structural invariants also hold: 64 rows, 0 duplicate keys, ledger complete (8 steps x 4), one `state_definition_sha256`.
3. **Earlier probe** (`ckpt_determinism.py`, OpenMM only): on Reference, `loadCheckpoint` into a fresh Context reproduces continuation exactly for a single Context. On CPU, OpenMM itself was not reproducible after 250 steps (7.5e-4 to 1.9e-3 nm), so a CPU bitwise gate is not attainable with OpenMM CPU.

**Interpretation, with its limits.** The two diagnostics differ in more than the stepping mode, so they do not isolate it. Confounds between `DIAG_A` (diverges) and `DIAG_B` (bitwise):
- **Configuration:** F with the `W0` table (4 ordinary windows) vs D (aux slot table, NVT production).
- **Auxiliary code:** non-aux vs aux (aux force, aux sample/event writers, `prepare_aux_resume` / re-seal path).
- **Stepping machinery:** default `pool.map` stepping (all replicas at once) vs the admission dispatcher with one active replica and 50-step turns.
- **Interruption:** SIGKILL with `--flush-every-log` vs the aux-only exception hook at 3500.

The round-1 Reference aux run `F1_X` (config A, concurrent stepping, with an experimental code change since reverted) also diverged. That fits the same hypothesis, but it is a different config and code. "Bitwise for an aux run" rests on a single run (`DIAG_B`). The suspected mechanism (shared random-number state across concurrently stepped Reference replicas) was not instrumented. A discriminating follow-up would be config D on Reference with and without `--active-replicas-per-gpu 1`, everything else identical.

**Consequence.** One aux run with serialised stepping resumed bitwise on Reference. With the default concurrent stepping, only the structural gate below passed.

## What passes: the structural gate

Every check below holds for A, B, C, D, `RUN_A_R` (SIGTERM) and `RUN_A_F4` (refuse → fix → resume) at `fceca0e`, and for `RUN_A3_X`, `RUN_A3_F4` and `RUN_C3_X` at the shipping commit `cbd26c7`, each against its config's control:
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

**F4.** Every Context-free refusal runs before the resumed segment is registered, and touches nothing (round 2: `prepare_aux_resume` is read-only). The Context-dependent refusals inside the checkpoint load run after registration, but before the re-seal. On such a refusal, the empty new segment is discarded: its registry entry, its empty data dirs and its snapshot are removed. That is a rollback of the job's own writes, not an absence of writes. The re-seal walks back over orphans to the checkpoint's own segment, and the resumed segment's parent is that segment (round 2). This is measured only for a Context-free refusal (below); the in-load discard path is covered by unit tests, not by a real run. In `RUN_A_F4`, a resume with `--temperature-k 301` was refused with `state_definition_sha256 changed ... first differing fields: state.temperature_k: 300.0 -> 301.0`. Its `segments.json` sha was identical before and after (`08787da55c5eb7dd`), and `windows/` still held only `seg_001.json`. The following plain resume completed with 0 duplicate keys and a complete ledger.

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
