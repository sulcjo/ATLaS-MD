# Replica admission cap and MPS thread share — design

Date: 2026-09-26
Status: design approved in conversation; amended 2026-09-27 after spec review (manifest per job, MPS timing,
turn-end deadlines, hang semantics, non-GPU queue key); amended again 2026-09-27 for the small board's seven
conditions (one manifest representation, total callback, drain protocol and lock rule, pool exclusivity,
four test gaps, real entry chain, single-context phases). Board transcript: job tmp
`board_admission/transcript.md`, verdict ACCEPT-WITH-CHANGES 82/100, one REJECT dissent (§11)
Scope: performance-upgrades P3 (bounded active replicas per GPU) and P4 (MPS active-thread percentage),
production delivery. Parent spec: `docs/superpowers/specs/performance-upgrades/spec.md` §6–7.

## 1. Why

At 236 resident contexts (59 per L40S, the chignolin_9 layout) under MPS, production submits every
replica's step to its own thread at once. Each thread keeps a CPU core busy while its GPU work is queued,
so 236 threads contend for 192 cores and MPS interleaves 59 clients per GPU.

Measured in the benchmark harness (`docs/superpowers/specs/performance-upgrades/benchmarks/`, jobs 2664328
and 2665264, chignolin_8 system, production force layout, Pep-GaMD stage 5 with the boost live, two to four
60-s runs per cell):

| Configuration | node ns/day | vs today | host CPU cores |
| --- | --- | --- | --- |
| all 59 admitted, MPS default (today) | 2,617 | – | ~175 |
| 8 per GPU, MPS default | 4,102 | +57 % | ~31 |
| 8 per GPU, MPS 25 % | 4,558 | +74 % | ~30 |
| 6 per GPU, MPS 25 % | 4,566 | +74 % | ~23 |
| 8 per GPU, MPS 25 %, 50-step turns | 4,646 | +78 % | ~31 |
| 4 per GPU, MPS 25 % | 3,952 | +51 % | ~16 |
| 2 per GPU, MPS 25 % | 2,354 | −10 % | ~8 |

Full table, figure and caveats: Section 6.3 of `ATLaS-MD_PepGaMD_integrator_v0.8.2_2026-09-22.docx`. The
harness runs MD only; production adds sampling, exchange and checkpoints at common boundaries, so the gain
must be re-measured in the real loop (§9).

## 2. Goal and success criteria

Deliver the cap and the MPS share to production as opt-in settings, then enable them on chignolin_9 at a
clean restart.

Success:

- With the defaults, production behaves exactly as today (same code path, not a limit of "infinity").
- With `active_replicas_per_gpu: 8`, `active_replica_turn_steps: 50`, `cuda_mps_active_thread_percentage: 25`,
  chignolin_9's real production ns/day rises by most of the harness gain.
- No stalls, deadlocks, lost steps or changed boundary sequence; failures surface as today's single
  exception with diagnostics.

Out of scope: an automatic default (a follow-up spec, once production confirms the gain — decision "C",
2026-09-26); capping Context reads (`_fetch_state`, `_fetch_exchange_state`); setup, US-pull and GaMD
recon phases; CPU affinity (P5); removing reporting barriers (P6).

## 3. Configuration

Three settings, each a YAML key and a CLI flag (same name, dashes for underscores), validated in
`parse_args`, and recorded in `run_manifest.json` for every job (see "Recording" below):

| Key | Values | Default | Meaning |
| --- | --- | --- | --- |
| `active_replicas_per_gpu` | `all` or integer ≥ 1 | `all` | At most this many replicas per GPU advance at once during production stepping. |
| `active_replica_turn_steps` | integer ≥ 1 | `50` | Steps a replica runs per admitted turn before rejoining the back of its GPU's queue. Ignored when the cap is `all`. |
| `cuda_mps_active_thread_percentage` | `inherit` or integer 1–100 | `inherit` | Process-wide MPS active-thread percentage, applied at startup (§5). |

Validation rejects 0, negatives, non-integers and percentages outside 1–100 with a clear message. A cap
above a GPU's resident count is capped to that count; the effective per-GPU values are printed at production
start.

The cap is not part of any checkpoint and changes no recorded number, so it may differ between the jobs of
one campaign (for example, off before a clean restart, on after it).

**Recording.** `provenance._method_settings` alone is not enough: on a resume,
`ensure_run_manifest_initialized` finds a complete manifest and leaves it untouched, so `method_settings`
keeps the values of the campaign's first job. chignolin_9 would switch the cap on at a resume, and the
manifest would still say `all`/`inherit`. Each job therefore records its own values through
`update_run_manifest`, which runs on every job:

- `method_settings["replica_admission"]` — the current job's values, overwritten each job:
  `{"active_replicas_per_gpu": ..., "active_replica_turn_steps": ..., "cuda_mps_active_thread_percentage":
  {"requested": ..., "inherited_env": ...}, "effective_per_queue": {queue_key: limit}}`.
- `replica_admission_history` — a list with one entry per `run_gareus` call, i.e. per phase output directory
  (a job spanning two phases writes one entry into each phase's manifest; UTC time, `SLURM_JOB_ID` if set, and
  the same dict), appended read-modify-write (`_deep_update` replaces lists, so the patch carries the whole
  list). Safe without a file lock because one job process is the manifest's only writer while it runs.

This nested dict is the **only** representation. `_method_settings` does not gain three flat keys; it
emits the same `replica_admission` dict (built by one shared function, `admission_manifest_record(args,
effective_per_queue=None)`), so a fresh campaign's first manifest already holds it and every later job
overwrites the whole dict. `effective_per_queue` is `null` until production has built its replicas, then
patched. Every current reader of `method_settings` looks up explicit keys (checked 2026-09-27), so a record
that differs between jobs cannot trip a resume guard; a future whole-dict comparison must exclude it. Two copies of the same setting in one file would let one go stale on a resume — the bug this
section exists to fix — so the config test asserts the flat keys are absent (§7).

Readers (dashboard, `gareus_report.py`, provenance summaries) read `method_settings["replica_admission"]`,
never the history list and never `resolved_args` (a first-job snapshot that still holds the flat values), and treat an absent key as a pre-change manifest meaning `all`/`inherit`.

## 4. Dispatcher

New module `gareus/replica_admission.py`, one class:

```python
class AdmissionDispatcher:
    def __init__(self, pool, gpu_of_replica: Sequence[str], limit: int, turn_steps: int): ...
    def run(self, fn: Callable[[int, int], None], nsteps: int) -> None:
        """Advance every replica by exactly nsteps, in turns of at most turn_steps.

        fn(replica_index, n) runs on the replica's own executor thread (pool.submit(i, ...)).
        Returns when every replica has completed nsteps. Per GPU, at most `limit` calls of fn are
        in flight at any moment."""
```

Behaviour, carried over from the benchmark harness (`p3_admission_bench.py`) where it was measured:

- One FIFO queue of replica indices per GPU, filled in replica-index order. At most `limit` replicas per GPU
  are admitted; admission happens under a lock, submission to `pool` happens after the lock is released (a
  done-callback on an already-finished future runs in the submitting thread and would re-enter the lock).
- A replica runs `min(turn_steps, remaining)` steps per turn; when the turn completes, it rejoins the back of
  its GPU's queue if it still has steps left. The last turn of a call may be shorter.
- At most one admitted turn per replica at a time, so a replica's calls never overlap and keep its thread
  affinity (`_ReplicaAffinityExecutor` is unchanged).
- No busy waiting and no thread blocked on a semaphore: `run` waits on one `threading.Event`.

**Protocol.** Shared state (per-GPU queues, per-replica remaining steps, per-GPU in-flight counts, a
total in-flight count, `failed`, `first_exc`) is touched only under one plain `threading.Lock`. A done-
callback runs in the worker thread when the turn finishes, or synchronously in the submitting thread if the
future had already finished when the callback was attached; both cases go through the same code:

1. `_on_done(i, fut)`, under the lock: decrement replica `i`'s GPU in-flight count and the total; if
   `fut` raised and `failed` is not yet set, set `failed` and store that exception as `first_exc` (a later
   exception is kept only in a list for the log). If not `failed`: subtract the turn's steps from `i`'s
   remaining and, if steps remain, append `i` to the back of its GPU's queue. Then, still under the lock,
   pop the turns to launch: if not `failed`, fill every GPU's free slots from its queue head; if `failed`,
   pop nothing and clear every queue. If the total in-flight count is 0 and no turns were popped, set the
   Event. Release the lock.
2. Outside the lock, submit each popped turn (`pool.submit(i, fn, i, n)`) and attach `_on_done` with
   `add_done_callback`. A submit that raises is fed straight back into `_on_done` as a failed turn, so its
   in-flight count is returned.
3. `run` seeds the queues, performs one fill-and-submit as in steps 1–2, then calls `Event.wait()`.
   After it returns: if `first_exc` is set, clear the state and raise it; otherwise every replica has
   remaining 0 (asserted).

Rules the implementation must keep, each pinned by a test (§7):

- **Lock discipline.** The lock is never held across `pool.submit`, `add_done_callback`, `fn` or
  `Event.wait()`. It is a plain `Lock`, not an `RLock`: re-entry from the synchronous callback path would
  then corrupt state instead of deadlocking loudly, and releasing before submit makes re-entry impossible.
- **Total callback.** `_on_done`'s whole body is wrapped; an unexpected internal error is recorded as a
  failure (if none is recorded yet), the in-flight counts are still decremented in a `finally`, and the
  Event is still set when nothing remains in flight. This matters because `concurrent.futures` catches and
  only logs an exception raised inside a done-callback: an unguarded error there would lose the completion
  signal and leave `run` waiting forever with one log line as the only trace.
- **Drain.** Once `failed` is set no turn is re-queued, and no turn is popped, on any GPU; a turn already
  popped but not yet submitted when `failed` is set is skipped by `_submit_one`'s check, except in the
  unavoidable microsecond window between that check and `pool.submit` (the check cannot be atomic with a
  submit made outside the lock), where it may still start. Its result is then discarded and its slot
  released like any other. Turns already running finish; `run` raises `first_exc` only when the total in-flight count is 0. `run` therefore never returns
  or raises while any replica's `fn` is still executing, and a dispatcher that raised is reusable (the next
  `run` starts from clean state).
- **Pool exclusivity.** While `run` is active, only the dispatcher submits to the pool. This holds in
  production: `run_gareus` uses `_sim_pool` only from the coordinator thread (`production.py` step,
  fetch, swap-apply and rescue sites are all sequential calls), and the dispatcher's own callbacks submit
  only replica turns. The in-flight accounting and the idle-pool guarantee depend on it; `run` asserts it
  cheaply by refusing re-entry (a `running` flag, set and cleared under the lock, raises if `run` is called
  while already running).

The idle-pool guarantee is new: today's `_ReplicaAffinityExecutor.map` returns from `f.result()` at the
first failed replica in index order while later replicas may still be stepping, so `step_all`'s NaN
diagnostics can read Contexts that are still in use. With the dispatcher they never do. The
`dispatcher=None` path keeps today's behaviour unchanged, including this.

`gpu_of_replica` is taken from the `DeviceIndex` that `replica_platform_properties`
(`gareus/system_setup.py`) assigns when production builds each replica's context (`production.py`, replica
construction loop), so the cap follows the real placement for round-robin and `--replica-device-map` alike.
The queue key is the `DeviceIndex` string as given. In `single-context-split` mode every context gets the
same multi-device string (for example `"0,1,2,3"`), so all replicas share one queue — correct, since each
context spans every GPU. Non-GPU platforms (CPU, Reference) return no `DeviceIndex`; all replicas then share
one queue under the key `"shared"`, and the cap limits total concurrency (useful for tests). The same key
names appear in `effective_per_queue` (§3).

**A hung replica still hangs the run.** "Turns already running are allowed to finish" means one turn that
never returns blocks `run` forever. Today's `pool.map` join behaves the same way, so this is not a
regression; the dispatcher adds no timeout, and the existing job-level watchdogs (SLURM time limit, the
dashboard's stalled-progress warning) remain the only detection.

## 5. MPS thread share

Applied as the first action of `gareus.cli.main` after `parse_args`, before any OpenMM platform or context
exists (the setup, US-pull and production phases all run in this one process, and the MPS client reads the
variable when CUDA first initialises). Timing checked 2026-09-27 on the real entry chain: `python -m
gareus` runs `gareus.__main__` → `gareus.core.main`, which imports `gareus.cli` lazily; after importing
`gareus.core` and `gareus.cli` and running `parse_args`, neither `openmm` nor `gamd` is in `sys.modules`.
So at the apply point no OpenMM platform exists. The apply function asserts this: if `openmm` is already in
`sys.modules` when a percentage is requested, it exits with an error rather than set a variable that may be
read too late. The guard only sees OpenMM; CUDA initialised by some other import (none today) would pass it
undetected, which the spec accepts. A subprocess test pins the fact on the real entry chain (§7).

- `inherit`: do nothing; record the inherited `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` (or "unset") in the
  manifest.
- integer N: if the environment already holds a different value, exit with an error naming both values
  (no silent override); otherwise set `os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(N)` and record
  it.
- If N is set but `CUDA_MPS_PIPE_DIRECTORY` is absent, print a warning that MPS does not appear to be
  running and the setting has no effect.

MPS does not report the effective share back to the client, so the manifest records the requested value
and labels it "requested". The setting applies to every context the process opens, including setup and
US-pull contexts; up to 28 pull contexts at 25 % each still cover the GPUs.

**Single-context phases.** Setup (minimisation, NVT/NPT equilibration), the shared GaMD setup and the
recon phase run one or a few contexts per GPU; at 25 % each such context can use at most a quarter of its
GPU's SMs. Unmeasured bound: those phases can run up to ~4x slower, since they are compute-bound on one
context. They do not re-run on a production resume, so the chignolin_9 switch (§9, a resume in the final
phase) does not pay this; a fresh campaign with the setting on does. The rollout records these phases'
wall time from the job log when a fresh campaign first uses the setting; measuring them now is out of
scope. §9 step 3's production ns/day comparison cannot see this cost.

**Daemon side.** A default percentage can also be set on the MPS control daemon
(`set_default_active_thread_percentage`, or the variable exported to `nvidia-cuda-mps-control -d`), and
the client's request then interacts with that limit. The chignolin_9 launcher (`smoke/config/chignolin_9.sh`)
starts the daemon with neither (checked 2026-09-27), so today only the client value applies. The benchmark
jobs (2664328, 2665264) set the client variable the same way, so the measured gain is for this
configuration. The rollout (§9 step 2) requires the launcher to keep the daemon default unset; if a future
launcher sets one, the benchmark does not cover that configuration. How a client request above a daemon
default is resolved has not been checked against NVIDIA's documentation and is not relied on.

## 6. Integration in `step_all`

`step_all(nsteps)` (`production.py`, a closure inside `run_gareus`) keeps its structure, including the
`production_safe_chunk_steps` split, which still happens first. Its inner "advance every replica by `sub`
steps" moves into a module-level function in `gareus/replica_admission.py`, so tests can call it without
running `run_gareus`:

```python
def advance_replicas(pool, drivers, nsteps: int, dispatcher: Optional[AdmissionDispatcher]) -> None:
    """dispatcher None: today's pool.map over every driver's advance(nsteps), unchanged.
    Otherwise dispatcher.run(lambda i, n: drivers[i].advance(n), nsteps)."""
```

`step_all` calls `advance_replicas(_sim_pool, drivers, sub, dispatcher)`. The dispatcher is built once,
after replica construction, from the effective per-GPU limits; it is `None` when the cap is `all`.

Invariants (unchanged from today):

- `step_all(chunk)` returns only when every replica has advanced exactly `chunk` steps.
- Exchange, sampling, logging and checkpoints happen at the same common boundaries, in the same order;
  the production loop's boundary computation is untouched.
- Each replica's own NPT volume moves and trajectory frames fire inside its `driver.advance`, at the same
  local steps: `ReplicaStepDriver.advance` (`gareus/npt_driver.py`) integrates to each local deadline
  (volume move, then that state's reports) within the requested span, so splitting one call into several
  shorter ones visits the same events in the same order. A deadline exactly on a turn's end is serviced
  inside that turn's call: `advance` admits deadlines with `due <= end` (inclusive), integrates to `end`,
  then runs the due volume move and the reports of the post-move state before returning, and each moves
  its own next deadline past `end`. The next turn therefore neither repeats nor skips it (checked in the
  code 2026-09-27; the drivers test in §7 pins it).
- A `production_safe_chunk_steps` sub-call shorter than `turn_steps` simply makes every turn in that
  sub-call at most the sub-call's length; turn length never crosses a sub-call boundary, because each
  sub-call is one `dispatcher.run`.
- Random-number consumption per replica is unchanged (each replica's integrator and NPT RNG advance only
  with its own steps); the global exchange RNG is consumed only at boundaries, by the coordinator.

`step_all`'s error handling is unchanged: an exception from `dispatcher.run` is wrapped in the same
`RuntimeError` with NaN diagnostics. `completed` substep accounting stays per `safe_chunk` sub-call.

## 7. Tests

All targeted, CPU-only (run only the files below, per the targeted-tests rule).

`tests/test_replica_admission.py` — dispatcher with fake replicas (a sleep or no-op `fn`, recording
steps and an independent per-GPU in-flight counter):

- every replica advances exactly `nsteps` per `run`, over several consecutive runs;
- per-GPU in-flight count never exceeds `min(limit, resident)`;
- uneven GPU populations (59/59/59/58) and a single queue (no `DeviceIndex`);
- limit 1; limit above resident count; turn length that does not divide `nsteps` (last turn shorter);
  `nsteps < turn_steps`;
- instant `fn` (exercises the already-finished-future callback path, no deadlock);
- a failure in one replica mid-run: `run` raises that exception, no hang, nothing in flight afterwards,
  and a subsequent `run` on the same dispatcher works;
- drain across GPUs: one replica on GPU 0 fails while long turns run on GPUs 1–3; `run` does not raise
  until those turns finish (the fake `fn` records exit times; the raise must come after the last one), and
  no turn starts after the failure on any GPU;
- two near-simultaneous failures (two replicas released by one `threading.Barrier`, both raising): `run`
  raises exactly one of them, as `first_exc`, and the other is logged; no hang;
- total callback: an internal error injected into `_on_done` (monkeypatched bookkeeping helper that
  raises once) makes `run` raise within a timeout instead of hanging;
- lock discipline: the test replaces the dispatcher's lock with a recording proxy that tracks the owning
  thread, and asserts the lock is not held by the calling thread inside the fake `fn`, inside a wrapper
  around `pool.submit`, and inside a patched `Event.wait` (`lock.locked()` alone cannot tell whose lock
  it is);
- re-entry: calling `run` from inside `fn` raises;
- randomized stress (seeded, a few hundred iterations, CPU only, seconds): 200–300 replicas over 1–4
  queues with uneven populations, random limit, turn length and `nsteps`, `fn` sleeping a random 0–2 ms or
  returning instantly; assert every replica completes exactly `nsteps` per run, the per-queue in-flight
  maximum never exceeds `min(limit, resident)` and is reached when there is enough work, and no run takes
  longer than a generous timeout.

`tests/test_replica_admission_drivers.py` — `advance_replicas` over real `ReplicaStepDriver`s wrapping
small OpenMM Simulations (`tests/pep_gamd_fixture.build_small_simulation`, Reference or CPU platform, 4
replicas on 2 fake GPUs), each with a fake controller and a fake reporter that record the local steps at
which they fire (the pattern of `tests/test_npt_driver_scheduling.py`):

- over a sequence of calls shaped like production's boundaries (for example 400, 250, 150, 400 steps),
  vs `dispatcher=None`: identical per-replica step counters and identical recorded volume-move and report
  steps, in the same order — for cap 1 with 50- and 30-step turns, and for the production shape of a cap
  below the resident count with many replicas interleaving (8 small-OpenMM replicas on 2 queues, cap 2,
  so 2 of the 4 replicas per queue wait at any moment), plus cap ≥ resident;
- a deadline exactly on a turn boundary is serviced once;
- finite energies afterwards.

Trajectories are not compared bit for bit (thread scheduling is not deterministic on the CPU platform).
`step_all` itself stays a thin wrapper and is covered by the chignolin_9 production check (§9).

Defaults: with `active_replicas_per_gpu: all` the dispatcher factory returns `None`, and
`advance_replicas(pool, drivers, n, None)` calls `pool.map` exactly once with every driver (a recording
fake pool asserts the call and that `submit` is never used).

`tests/test_replica_admission_config.py` — parsing and validation of the three keys (YAML and CLI),
manifest recording on a fresh run AND on a resume against an already-complete manifest (the resume must
overwrite `method_settings["replica_admission"]` and append one `replica_admission_history` entry), no flat
`active_replicas_per_gpu`/`active_replica_turn_steps`/`cuda_mps_active_thread_percentage` keys anywhere
under `method_settings`, the MPS conflict error, the "MPS not running" warning, the `openmm`-already-imported
error, and — in a subprocess — the real entry chain: import `gareus.core` and `gareus.cli`, run
`parse_args` with a percentage, apply the MPS setting, and assert `openmm` and `gamd` are absent from
`sys.modules` and `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` is set.

## 8. Files

- New: `gareus/replica_admission.py` (`AdmissionDispatcher`, `advance_replicas`), the three test files above.
- Changed: `gareus/cli.py` (flags, validation, MPS apply in `main`), `gareus/provenance.py`
  (`admission_manifest_record`, used by `_method_settings` and the per-job patch), `gareus/production.py` (dispatcher build after replica construction; `step_all`
  branch), `gareus/helptext.py` (short entry under the performance/MPS topic).
- Launcher: `smoke/config/chignolin_9.yaml` gains the three keys only when the chignolin_9 switch is made
  (§9), not in this change.

## 9. Rollout

1. Merge with defaults (`all`, `inherit`); deploy to both aurum2 trees as usual (whole package, between
   jobs, never mid-job).
2. Enable on chignolin_9 at a clean restart: YAML `active_replicas_per_gpu: 8`,
   `active_replica_turn_steps: 50`, `cuda_mps_active_thread_percentage: 25`. The launcher already starts
   MPS; it must not export a conflicting percentage.
3. Compare the dashboard's node ns/day for the first full job against the preceding job at the same phase.
   Record the result in the integrator report's Section 6.3 and in
   `docs/atlas-md/developer/gpu-throughput-benchmark-todo.md`.
4. Follow-up spec (not this change): an automatic default (for example 8 per GPU when MPS is on and more
   than 16 replicas share a GPU), only if step 3 confirms the gain.

## 10. Risks

- **Production gain smaller than the harness gain.** Boundaries (exchange every 400 steps, sampling every
  250) still join all replicas; with a cap, the last admitted turns at each boundary run with a partly
  empty GPU. 50-step turns keep that tail short; step 3 measures it.
- **Context reads stay uncapped.** `_fetch_state` and `_fetch_exchange_state` still issue 236 concurrent
  `getState` calls at boundaries. They are short; if profiling in step 3 shows them material, capping them
  is a separate change.
- **Straggler replica.** A slow replica (for example a costly NPT move) delays its GPU's queue by at most
  one turn per boundary, the same as today's join.
- **Hung replica.** Blocks the run forever, as today (§4); no new detection.
- **MPS daemon default.** Unset in today's launcher; a launcher that sets one runs outside the measured
  configuration (§5).
- **Single-context phases at 25 %.** Up to ~4x slower setup/recon on a fresh campaign; not paid by the
  chignolin_9 resume (§5).

## 11. Review record

Small board, 2026-09-27 (kimi, mini; thinker failed with HTTP 500 twice and gave no position; chair glm):
ACCEPT-WITH-CHANGES, 82/100. Its seven conditions are folded in above: one manifest representation (§3),
total callback, drain and lock rule, pool exclusivity (§4), the added tests (§7), the real entry chain
(§5, §7), and single-context phases plus absent-key tolerance (§3, §5).

Dissent (mini, REJECT, 92), not adopted, with the reason:

- *Callbacks run in the worker thread and deadlock on the lock.* A callback attached to a pending
  future runs in the worker thread; one attached to an already-finished future runs synchronously in the
  attaching thread. The protocol (§4) never holds the lock while attaching or submitting, so neither case
  can re-enter or deadlock; the lock-discipline test pins it.
- *A replica is re-queued across `production_safe_chunk_steps` sub-calls, adding a turn.* Each sub-call
  is one `run`, which returns only when every replica's remaining count is 0 (asserted), so nothing carries
  into the next `run`. A turn is `min(turn_steps, remaining)` inside one `run`; a 60 + 40 split gives
  50 + 10 then 40 steps, exactly the requested totals, and the driver visits the same deadlines (§6).
- *The client value is silently capped by the daemon default; query the daemon.* Not verified against
  NVIDIA's documentation, and the launcher sets no daemon default (§5); the control is the §9 step 2
  launcher requirement, not a runtime query.
