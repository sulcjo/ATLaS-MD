# Continue existing states between epochs (`--ap-continue-states`)

Status: implemented on `feat/continue-existing-states` (2026-10-01). Off by default.

## Why

On chignolin_9, every numbered epoch re-grafted all 236 windows from the seed bank and re-pulled them.
That had two costs:

- Continuous trajectories lasted only about 14 ns, while the D3/P4 turn dihedrals decorrelate much more
  slowly (their autocorrelation is still 0.93 at 10 ns).
- The fold populations stayed locked to the seeds. Analysis: `docs/_local_docs/c9_convergence/`.

## Behaviour

When the flag is on, every non-top-up segment of a numbered epoch ≥ 1 and of the final phase receives
`continue_parent_dirs`: the segments of all earlier phases, oldest first, that left end states. Each window
is then handled as follows:

- **Existing state:** if its state has an end state in a parent, it continues from the newest one (the same
  resolver as frozen-final extensions).
  - From an exported `final_window_states/` State it gets exact positions, velocities and box, and the CVs
    are checked after `set_window`.
  - From `final_pdbs/` it gets positions and box; constraints are re-applied and velocities redrawn fresh.
  - The parent's restraint for that state must match, or it raises `SeedMismatchError`, and the whole
    phase falls back to the old full pull.
- **New state:** if its state has no earlier end state (add, insert, split child, respring, new rung or
  respaced rung), it is seeded and pulled as before.
  - Only those windows go through `generate_us_starting_states_by_pulling`, and the results are merged back
    with `merge_partial_pull`.
  - `us_starting_structures_subset.json` records which windows were pulled.
  - Slow-mode reseeding (X3) is skipped in this mode, because it re-seeds existing windows.
  - The auto-drop fraction cap applies to the pulled subset.

After an auto-drop, `drop_bad_us_windows_and_rebuild` returns `keep_indices`, and the boxes and seed maps
are re-indexed with it. Top-ups, extensions and epoch 0 are unchanged.

## Verification

- `tests/test_continue_existing_states.py`: parent ordering, the driver helper, partial continuation with
  missing states, refusal on a restraint mismatch, the merge with drop mapping, and the CLI flag.
- Related suites pass: 2,324 tests.
- Real-data dry run: c9 `epoch_002/baseline` and `final/baseline` resolve 236/236 windows from their
  parents' final PDBs (restraints verified, 0 to pull).

## Adversarial review fixes

- **HIGH (verified): stale chain-end PDBs.** `final_pdbs/` accumulates one file per replica per job, and the
  PDB picker took the last file in name order. That was the wrong (stale) fork for 152–208 of 236 windows per
  chignolin_9 phase, and it also affected frozen-final extensions: `final_extension_002` was seeded from forks
  for 208/236 windows.
  - `_chain_end_pdb_by_window` now takes `replica_r_window_{assignments[r]}` from the segment's last checkpoint
    manifest, falling back to the newest file by mtime. It picks the true chain end for 236/236 windows in
    every c9 phase.
- **Exported States no longer get the CV check.** A continued State crosses a phase boundary, where the CV
  definition may legitimately change (tICA refit, CV2 switch). It keeps its velocities but skips the
  same-phase `assert_seed_matches`; the restraint check still applies.
- **Auto-drop cap over the whole phase.** The cap is now measured against the phase's full window count
  (`total_windows`), not the pulled subset.
- **Non-scheduled final phase.** It honours the flag too.
- **Subset report numbering.** The pulled subset's report rows are renumbered to full window indices
  (`remap_subset_window_indices`).
- **X3 warning.** Combining the flag with X3 prints a warning (X3's reseeds of existing windows are never
  applied).
- **Provenance.** The flag is recorded in the run manifest.

Known limitations:
- The explicit-2D seed-reachability pre-filter still runs over all windows before continuation is decided.
  It could drop an existing state that has no GENPEPT seed nearby; it never fired on c9 and is a no-op with
  a large `us_seed_preflight_max_score`.
- Continued windows skip the start-quality gate (they start from their own restraint's end state).
