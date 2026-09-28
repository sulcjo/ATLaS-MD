# Output retention and compression (spec, 2026-09-28)

Status: implemented 2026-09-28 (plan docs/superpowers/plans/2026-09-28-output-retention.md); roll-out on chignolin_9 pending. Replaces the sketch in `docs/atlas-md/developer/topups-todo.md` T7.
Example run: chignolin_9 (236 states, 7 phases with checkpoints), measured 28 Sep 2026 on the live aurum2 run and
its local mirror. Raw measurements: `inventory_report.md` (session scratch; key numbers are copied below).

## 1. Goal and hard rules

Cut a campaign's disk footprint by an order of magnitude without losing anything that resume, the adaptive driver,
or downstream analysis reads.

- **Never change sampled data.** Parquet samples/exchanges, trajectories and every file an analysis reads stay
  byte-identical.
- **Never break resume.** A resumed job must find exactly what it finds today: the newest checkpoint generation of
  its phase, every phase's root checkpoint manifest, the solvated topology, the state registry.
- **Never break continuation.** `final_window_states/` and `final_pdbs/` stay wherever a later top-up or extension
  round can still seed from them.
- **Opt-out, not opt-in, for the lossless parts; opt-in for anything that deletes data a human might want.**

## 2. Where the bytes are (chignolin_9, aurum2, 28 Sep 15:33)

688 GB apparent, 459 GB on disk (the aurum2 filesystem compresses transparently, ~1.40x on average).

| Output | Apparent | Share | Notes |
| --- | --- | --- | --- |
| Checkpoint generations (`*/checkpoints/generations/<id>/replica_*.chk`) | 633.4 GB | 92.0 % | 424 generations x 236 x 6.88 MB (~1.625 GB each, incl. a ~1.3 MB per-generation `manifest.json`); `final/baseline` alone 213 generations = 346 GB |
| `progress.jsonl` | 23.7 GB | 3.4 % | one file; 99.47 % of its bytes are `distances` events (34,716 lines, avg 678 kB) |
| PDB sets | 23.4 GB | 3.4 % | `final_pdbs` 8.2, `filtered_seed_bank` 7.1, `seed_bank_*` 5.7, `us_starting_structures` 2.0, swarm frames 0.4 |
| Trajectories (`.xtc`) | 4.4 GB | 0.6 % | |
| npz (`analysis_arrays`, `adaptive_union_mbar`, ...) | 1.7 GB | 0.3 % | |
| `final_window_states/` | 0.7 GB | 0.1 % | `final_extension_001` only |
| Parquet samples + exchanges | 0.27 GB | 0.04 % | |
| Everything else | ~0.2 GB | | csv/json/md/yaml |

Compressibility of copies (zstd 1.5.7 on the local machine; ratio = original/compressed):

| Payload | zstd -3 | zstd -19 --long | gzip -6 | aurum2 FS alone |
| --- | --- | --- | --- | --- |
| `.chk` (raw OpenMM binary state) | 1.48x | 1.48x | 1.47x | 1.36x |
| `progress.jsonl` (1 GB slice) | 12.3x | 21.8x | 7.4x | 3.36x |
| PDB directory (tar) | - | 8.6x | - | 2.26x |
| `final_window_states/` (tar) | 2.6x | 3.2x | 2.7x | 1.54x |
| `.xtc` | 1.11x | 1.13x | 1.11x | 0.98x |
| `.parquet` | 1.00x | 1.01x | 1.00x | 1.00x |

aurum2 has no `zstd` binary on the login node, but the calc env's Python has `zstandard` 0.23.0.

## 3. Who reads what (code map, `file:line` as of cdaaeba)

| Output | Readers | Needs |
| --- | --- | --- |
| Checkpoint generations | resume `load_production_checkpoint` (`production.py:5357` -> `checkpoint_store.read_validated_generation` `:249`); availability probe `production_checkpoint_available` (`checkpoints.py:57` -> `require_available` `checkpoint_store.py:326`), called for every phase the driver visits (`adaptive_production.py:6586, 6735, 6809, 7818, 8402, 8510`); scratch sync `copy_committed_generation` (`checkpoint_store.py:336`, via `repo_adapters.py:56`) | **only the generation the root manifest names**. No fallback to an older one: a corrupt current generation raises. `require_available` validates it, so the newest generation of even a finished phase must stay (or the phase needs an explicit "sealed" state, §4.1b). `previous_generation_id` is written (`checkpoint_store.py:206`) and read by nothing. No garbage collection exists (`checkpoint_store.py:8-10`). |
| Root `production_checkpoint_manifest.json` of every phase | runtime-pool reconciliation `stage_delivered_md_steps` (`adaptive_production.py:5609`, globs every phase), `_segment_checkpoint_prod_done` (`:6298`), segment reseal (`:6977`), extension seeding `_parent_restraints` (`extension_seeding.py:113`) | JSON only, every phase, forever |
| `progress.jsonl` `distances` events | `gareus_monitor.py` only: `parse_distances_samples` (`:1222`) over a tail read that grows until each window has enough boost samples (`:1310-1333`); ETA history `ns_history` tails 500 lines (`:2299`) | the recent tail only; nothing reads old events. Emitted by `DistanceLogger.log` (`logger.py:1670-1693`) on every sample (250 steps) regardless of `--distance-output-mode` |
| `progress.jsonl` other events | monitor tail (ETA, pool, boost) | recent tail |
| `lifecycle.jsonl` | nothing (state comes from `state_registry.json`) | - |
| Parquet `samples/`, `exchanges/` | `query.load_samples`/`load_exchanges` (`query.py:286, 312`) for the driver's diagnostics and all MBAR analysis | every segment, forever |
| `analysis_arrays.npz`, `adaptive_union_mbar.npz` | analysis, diagnostics, quality gate | regenerable from Parquet (`analysis._ensure_analysis_arrays_npz`, `correctness/export.py:63`) |
| Trajectories | adaptive seed extraction `tica.resolve_seed_frame_pdb` (`tica.py:1328`, all fragments of an epoch), tICA/CV discovery, and standalone analyses (Rg/PCA/chignolin FES, CV2 reprojection, thermo frames, audits). Not read by resume | every fragment, forever |
| `tica_obs/*.npz` | tICA refit; ground truth for seed-frame extraction (`tica.py:1356`) | forever |
| `seed_bank_epoch_NNN/`, `seed_bank_final/` | `_discover_latest_seed_bank` on resume (`adaptive_production.py:1267`, load-bearing: see the chignolin_6 note in its docstring) | newest usable bank; older banks only as fallback candidates |
| `filtered_seed_bank/` (per segment) | seeding of that segment, at segment start only (`adaptive_production.py:6866-6870`) | a subset copy of a seed bank, regenerable; only needed until the segment has seeded |
| `final_window_states/` | top-up seeding (`topup_seeding.py`), extension continuation (`extension_seeding.resolve_seed_sources`, newest parent per state across all parents) | every parent a later round may still seed from; not regenerable (velocities) |
| `final_pdbs/` | extension continuation fallback (`extension_seeding.py:140`), energy decomposition fallback | as above |
| `us_starting_structures/` | the US pull of the same invocation (`seeding.py:1328`); deleted by the pilot-round cleanup (`adaptive_feedback.py:2815`) | write-then-consume; not re-read on resume (a resumed phase skips pulling) |
| `01_solvated_start.pdb`, `03_npt_equilibrated_state.xml`, `run_manifest.json`, `gareus_metadata.json`, `state_registry.json`, `epoch_window_map.csv`, window CSVs, `shared_gamd_setup_globals.json` | resume, driver, analysis | small; always keep |

## 4. Design

### 4.1 R1: checkpoint generation retention (the only change that matters: -92 %)

**a) Keep the newest N generations per phase, delete the rest.** New flag `--checkpoint-keep-generations N`
(YAML `checkpoint_keep_generations`). **Opt-in:** the default is `0`, which keeps every generation (today's
behaviour); a campaign enables pruning by setting it, and the recommended value is **4** (the current generation
plus three fallbacks a human can recover from by hand).

- Where: in `checkpoint_store`, a `prune_generations(out_dir, keep)` run by `publish_generation` right after the
  root manifest has been replaced and fsynced (`checkpoint_store.py:223`), under the same publication lock.
  Also run by `copy_committed_generation` on the destination after a successful copy (scratch -> main), so the main
  tree does not keep accumulating the generations scratch prunes.
- What it keeps (as implemented): the `keep` generations reached by following each generation manifest's
  `previous_generation_id` back from the root manifest's `generation_id`, never ordered by directory mtime (a copy
  can scramble mtimes). If that chain breaks before `keep` readable generations (an unreadable or missing manifest
  in the middle, or the main copy of a `--scratchdir` run lacking a generation scratch never synced), the remaining
  slots are filled from readable generations with `absolute_step` <= the root's, newest step first (tie-break
  `created_unix_time` when present), so older valid fallbacks are not all deleted. Any generation off the chain
  with `absolute_step` >= the root's (an unpublished orphan) is kept. Never delete a directory whose
  `manifest.json` cannot be read (it is reported as skipped), and never touch `.staging-*` directories here
  (publish already cleans those).
- Deletion renames a complete, superseded directory to `generations/.deleting-<id>-<nonce>` and then `rmtree`s it,
  after the new root manifest is durable. A crash mid-delete leaves a dot-name no reader or counter enumerates; the
  next prune sweeps any `.deleting-*` leftover first (one it cannot remove is reported as skipped and never blocks
  the prune). A crash during pruning leaves extra old generations, never a missing current one.
- An in-run prune failure (inside `publish_generation` or `copy_committed_generation`) prints a WARNING and
  returns normally: the new checkpoint is already committed.
- Nothing reads older generations (§3), so resume behaviour is unchanged. Chignolin_9 at 4: 424 generations -> 28,
  633 GB -> ~46 GB (apparent).

**b) Optional, later: seal finished phases.** A finished phase's newest generation (1.6 GB) is still validated by
`require_available` whenever the driver probes the phase. Deleting it would need a new status (`sealed_v2`: root
manifest kept, payloads removed, `require_available` returns True and `load_production_checkpoint` refuses with a
clear error). Saves ~1.6 GB per finished phase (8-11 GB on chignolin_9). Not in the first cut: small gain for a
change to the resume state machine.

**c) Retro-pruning existing runs.** `python -m gareus.retention prune-checkpoints <run_dir> [--keep 4] [--apply]`:
walks every `checkpoints/` under the run, applies the same rule (`prune_generations`), prints per-phase bytes freed,
and refuses to touch a phase whose `.gareus_run.lock` names a live process. That live-lock check is host-local
(`os.kill` on this host); on a cluster, run it when no job of the campaign is running. A dry run by default;
deleting needs `--apply`. On an existing campaign with many generations, retro-prune first before turning on
`--checkpoint-keep-generations`: the first in-run prune otherwise deletes them inline under the checkpoint lock.

### 4.2 R2: move `distances` events out of `progress.jsonl` (-23 GB, and growing ~5 GB/day on chignolin_9)

The `distances` event is a full per-replica dump every 250 steps, full float64 precision, verbose keys repeated for
every replica. The monitor needs only the recent tail; the same per-replica values are in the Parquet samples
(to be verified field by field, task T2.1).

- `DistanceLogger.log` writes `distances` events to their own file `live_distances.jsonl` instead of
  `progress.jsonl`. Small events (`progress`, `run_start`, `run_complete`, ...) stay in `progress.jsonl`
  (124 MB over chignolin_9's whole life).
- `live_distances.jsonl` is a two-file ring: when it passes `--live-distances-max-mb` (default 256), it is renamed to
  `live_distances.1.jsonl` (replacing the previous one) and a new file starts. At most ~512 MB on disk per phase.
- `gareus_monitor.py` reads `live_distances.jsonl`, then `live_distances.1.jsonl` if it needs more depth, then
  falls back to `progress.jsonl` for runs written before this change.
- The exchange dashboard (~0.5 MB at 236 states) goes into every 20th ring line only, so a 1 MB ring tail misses it
  about half the time (20 slim lines are ~1.1 MB). `DistanceLogger` therefore also atomically replaces
  `<phase>/live_dashboard.json` (newest dashboard with `wall_time_s` and `step`) whenever it puts a dashboard in the
  ring; the monitor takes the dashboard and its age in seconds from that file when `progress.jsonl` has none,
  falling back to the ring scan.
- The ring is monitor-only output and must never stop production: if creating or writing the ring (or
  `live_dashboard.json`) raises (ENOSPC, stale NFS handle, failed rotate, a non-JSON value), `DistanceLogger` prints
  one WARNING, drops the ring, and emits the legacy full `distances` event to `progress.jsonl` for the rest of the
  run.
- Dashboard history restore is unaffected (it reads CSV/Parquet, `logger.py:166-181`).
- Existing runs: `python -m gareus.retention split-progress <progress.jsonl> [--apply] [--force]` rewrites
  `progress.jsonl`, replacing each `distances` line with its `distances_summary` (streaming, atomic replace). There
  is no keep-tail option: the old per-sample dumps are not moved into a ring. Opt-in, because it deletes data (the
  old per-sample CV dumps) that nothing reads but a human might. Stopped campaigns only: a live job's open append
  handle would keep writing to the replaced file's old inode and lose every later line, so it refuses
  (`run_locked`, exit 1) whenever any `.gareus_run.lock` exists at or below the file's directory (no PID check, the
  job may run on another host) unless `--force`.

### 4.3 R3: PDB sets (-7 to -21 GB)

- **a) `filtered_seed_bank/` as hard links** (-7.1 GB on chignolin_9). `filter_seed_bank_for_state_ids`
  (`adaptive_production.py:1369`) copies a subset of a seed bank's PDBs; write hard links instead (`os.link`,
  falling back to copy across filesystems). Readers see the same file names and bytes. Same for any other
  seed-bank-to-seed-bank copy found during implementation.
- **b) `us_starting_structures/` of finished adaptive-production phases** (2.0 GB): consumed by the same
  invocation's pull only. Delete once the phase's production has published its first checkpoint, under an opt-in
  flag `--prune-us-starting-structures` (default off; they are a debugging aid when a pull goes wrong). Verify no
  other reader first (T3.2).
- **c) Cold archive of finished campaigns** (§4.4) covers `final_pdbs/` and old seed banks; they stay uncompressed
  while any later round may seed from them.

### 4.4 R4: cold archive for a finished campaign (opt-in tool)

`python -m gareus.retention archive <run_dir> [--apply] [--force]` and
`python -m gareus.retention restore <run_dir> [--apply]`, for a campaign whose driver reports `completed` and that
will not be extended soon:

- packs each PDB directory and each `final_window_states/` into `<dir>.tar.zst` (Python `zstandard`, level 19,
  without long-distance matching), with a sidecar listing member names, sizes and sha256; removes the originals
  only after the archive verifies;
- `restore` unpacks everything back to the original paths byte for byte (checked against the sidecar), so a later
  `--extend` works unchanged after a restore;
- leaves Parquet, trajectories, npz, checkpoints (after R1), manifests and all small files untouched.

Chignolin_9 after R1-R3: PDB sets 16.3 (after R3a) -> ~1.9 GB, `final_window_states/` 0.74 -> 0.23 GB.

### 4.5 Not changed

- **Checkpoint compression.** `.chk` compresses 1.48x and the aurum2 filesystem already gets 1.36x for free; after
  R1 the checkpoints are ~46 GB, so the remaining win is ~4 GB for added resume latency. Not worth it.
- **Parquet and trajectories.** Already compressed (1.00-1.13x); they are the analysis inputs.
- **npz caches** (1.7 GB): regenerable but small; the archive tool may offer `--drop-caches` later.

## 5. Expected result (chignolin_9, apparent size, aurum2)

| Step | Size | Saving |
| --- | --- | --- |
| Today | 688 GB | |
| R1 keep 4 generations per phase (opt-in) | ~100 GB | -588 GB |
| R2 distances out of `progress.jsonl` (+ one-off split) | ~77 GB | -23 GB |
| R3a hard-linked filtered seed banks | ~70 GB | -7 GB |
| R4 cold archive (PDBs, final_window_states) | ~55 GB | -15 GB |

The two live phases grow about 1.6 GB per checkpoint generation today; with R1 that growth stops at two
generations per phase, so the saving increases for as long as the campaign runs.

## 6. Tasks

1. **R1** `prune_generations` in `checkpoint_store` + wiring in `publish_generation` and
   `copy_committed_generation` + CLI/YAML flag + `python -m gareus.retention prune-checkpoints`.
   Tests: default `0` deletes nothing; keeps exactly the current + N-1 newest along the `previous_generation_id`
   chain (step-sorted fill when it breaks); never deletes the root manifest's
   generation, even with scrambled mtimes; crash injected between root replace and prune leaves a resumable
   phase; prune after `copy_committed_generation` on the destination; `keep=0` deletes nothing; unreadable
   generation manifest is skipped with a WARNING; resume after pruning loads bit-identical state.
2. **R2** T2.1: check that every field of a `distances` row that anything reads (the monitor's CV1/CV2/bias/boost/
   window/replica, `logger.py:1640-1665`) exists in the Parquet samples, or is derivable, and write the mapping into
   the spec. Then the `live_distances.jsonl` ring, monitor reader with fallback, and the one-off splitter.
   Tests: ring size bound; monitor reads new, rotated and legacy layouts; splitter keeps every non-`distances` line
   byte for byte and in order.
3. **R3** hard links in `filter_seed_bank_for_state_ids`; T3.2 audit of `us_starting_structures` readers before the
   opt-in prune flag. Tests: seeding from a hard-linked bank is identical; cross-device fallback copies.
4. **R4** archive/restore tool. Tests: round trip byte-identical; refuses a campaign that is not `completed`;
   refuses to delete originals if verification fails.
5. **Roll-out on chignolin_9:** R1 retro-prune as a dry run, then `--apply` between two jobs of the chain
   (job held, like a deploy); record bytes before/after. The local mirror keeps old generations unless its sync
   runs with `--delete` on `checkpoints/`; prune it with the same tool.

## 7. Decisions (user, 28 Sep 2026)

1. `checkpoint_keep_generations`: recommended value **4** (current + three fallbacks).
2. R2: one `live_distances.jsonl` ring **per phase directory**.
3. R3b (`--prune-us-starting-structures`) and R4 (cold archive): **opt-in**, as proposed.
4. R1 ships **opt-in** (default `0` = keep every generation); a campaign turns it on explicitly. R2 and R3a are
   lossless for every reader and ship on by default.
