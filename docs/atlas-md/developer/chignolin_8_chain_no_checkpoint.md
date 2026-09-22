# chignolin_8: the chain resubmits but never checkpoints — every job restarts epoch 0 at step 0

Found 2026-09-22 11:0x while checking job 2566295. Status: diagnosed, launcher fix not yet applied.

## Symptom

The chain looks healthy: job 2563608 ran 07:13:13 -> 11:03:09, printed
`[gareus] Resubmitting chignolin_8 with --resume`, and job 2566295 started immediately on d094 and
reported `epoch 0 already complete; resuming at the ladder it designed`, 248 windows loaded.

It is not healthy. `adaptive_production/epoch_000/checkpoints/` does not exist, `segments.json` still
holds `seg_001` with `status: "running"`, `start_step: null`, and job 2566295 is re-running the
2h11 US pull (`Generating US starting structures with GENPEPT seeding: 90 conformers, 248 windows`)
instead of resuming replica state.

## Mechanism (proven, not inferred)

1. `#SBATCH --signal=B:TERM@600` sends TERM to the **batch shell** at 11:03:04 (10 min before the
   11:13:13 walltime). `trap request_graceful_stop TERM USR1` forwards it to the python PID.
2. A trapped signal makes bash's `wait` **return immediately**, even though python is still alive.
   The launcher's last three lines are:
   ```bash
   wait "${GAREUS_PID}"
   GAREUS_EXIT=$?
   set -e
   exit "${GAREUS_EXIT}"
   ```
   so the script exits 143 about a second later.
3. SLURM tears the job down when the batch script exits. `chignolin_8_2563608.err`:
   `*** JOB 2563608 ON d093 CANCELLED AT 2026-09-22T11:03:08 DUE to SIGNAL Terminated ***` —
   **4 seconds** after the TERM, and 10 minutes before walltime. Nothing external killed it; the
   script exited voluntarily and took python with it.
4. python never reached `save_production_checkpoint` (`gareus/production.py:8062`, the
   `_graceful_shutdown.is_set()` branch). Neither its handler print
   (`SIGTERM received — will checkpoint and exit after the current MD chunk`, `gareus/lifecycle.py:34`)
   nor `Graceful shutdown: saving checkpoint at step ...` appears in the .log or the .err.

The periodic checkpoint cannot cover for this: `checkpoint_interval: 250000` and the measured
throughput is 50,500 steps per 95 min of production, i.e. the first periodic checkpoint needs ~7.8 h
of production inside a 4 h walltime. It has never fired and never will at these settings.

## Consequence

- `prod_done` resets to 0 every job. Epoch 0's target is 540,322 steps; the per-job production
  ceiling is ~50,500. **Epoch 0 can never complete.** The chain runs to its `MAX_RESUBMITS` of 50.
- Duty cycle is ~40%: 2h11 of each 3h50 job goes to redoing the US pull. With a checkpoint this is
  skipped outright — `gareus/production.py:6288` prints
  `[resume] Production checkpoint manifest found; skipping window generation, US pulling, and shared GaMD setup.`
  So the one fix recovers both the lost steps and the wasted 55%.

## Fix

In `~/gareus/chignolin/chignolin_8.sh`, re-wait until python has actually exited:

```bash
wait "${GAREUS_PID}"; GAREUS_EXIT=$?
while kill -0 "${GAREUS_PID}" 2>/dev/null; do
    wait "${GAREUS_PID}"; GAREUS_EXIT=$?
done
```

Edit by temp-file + `mv` (inode replacement): job 2566295 has that file open and SLURM re-reads
`${RUN_SCRIPT}` at the next `sbatch`. No `scancel` is needed — `cleanup` resubmits at the end of the
current job (~15:03) and SLURM snapshots the script at submission time, so the edit lands on the
next cycle without re-tripping the double-submit hazard.

Open questions, not changed here:
- Is `@600` enough grace to serialize 248 OpenMM States? If not, `--signal=B:TERM@1200`.
- `checkpoint_interval: 250000` is unreachable at this walltime; ~20-25k would protect against
  hangs that never deliver a TERM. It lives in `${CONFIG}`, so changing it mid-campaign touches
  provenance/resume validation — a question for the user, not an edit.
- `chignolin_8_2563608.err` also carries
  `[ATLaS-MD] dashboard render failed (RuntimeError: dictionary changed size during iteration)` —
  separate, cosmetic, suppressed after the first occurrence.
- When 2566295 enters production (~13:15), check whether it seals `seg_001` `abandoned` and opens
  `seg_002` or writes into the same segment with overlapping `first_step`/`last_step`.
