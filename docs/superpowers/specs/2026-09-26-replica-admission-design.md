# Replica admission cap and MPS thread share — design

Date: 2026-09-26
Status: design approved in conversation; written spec awaiting review
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
`parse_args`, and recorded in `run_manifest.json` via `provenance._method_settings`:

| Key | Values | Default | Meaning |
| --- | --- | --- | --- |
| `active_replicas_per_gpu` | `all` or integer ≥ 1 | `all` | At most this many replicas per GPU advance at once during production stepping. |
| `active_replica_turn_steps` | integer ≥ 1 | `50` | Steps a replica runs per admitted turn before rejoining the back of its GPU's queue. Ignored when the cap is `all`. |
| `cuda_mps_active_thread_percentage` | `inherit` or integer 1–100 | `inherit` | Process-wide MPS active-thread percentage, applied at startup (§5). |

Validation rejects 0, negatives, non-integers and percentages outside 1–100 with a clear message. A cap
above a GPU's resident count is capped to that count; the effective per-GPU values are printed at production
start and recorded in the manifest (`method_settings["active_replicas_per_gpu_effective"]`, a dict GPU →
limit).

The cap is not part of any checkpoint and changes no recorded number, so it may differ between the jobs of
one campaign (for example, off before a clean restart, on after it).

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
- Failure: on the first exception, every replica still queued is dropped, turns already running are allowed
  to finish, and `run` re-raises the first exception only after nothing is in flight. The pool is therefore
  idle when `step_all`'s diagnostics read Contexts.
- No busy waiting and no thread blocked on a semaphore: `run` waits on one `threading.Event`.

`gpu_of_replica` is taken from the `DeviceIndex` that `replica_platform_properties` assigns when production
builds each replica's context (`production.py`, replica construction loop), so the cap follows the real
placement for round-robin and `--replica-device-map` alike. Non-GPU platforms (CPU, Reference) have no
`DeviceIndex`; all replicas then share one queue, and the cap limits total concurrency (useful for tests).

## 5. MPS thread share

Applied as the first action of `gareus.cli.main` after `parse_args`, before any OpenMM platform or context
exists (the setup, US-pull and production phases all run in this one process, and the MPS client reads the
variable when CUDA first initialises):

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
  shorter ones visits the same events in the same order; a deadline that falls exactly on a turn's end is
  serviced at the end of that turn, on the same state.
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
  and a subsequent `run` on the same dispatcher works.

`tests/test_replica_admission_drivers.py` — `advance_replicas` over real `ReplicaStepDriver`s wrapping
small OpenMM Simulations (`tests/pep_gamd_fixture.build_small_simulation`, Reference or CPU platform, 4
replicas on 2 fake GPUs), each with a fake controller and a fake reporter that record the local steps at
which they fire (the pattern of `tests/test_npt_driver_scheduling.py`):

- over a sequence of calls shaped like production's boundaries (for example 400, 250, 150, 400 steps),
  cap 1 with 50-step turns and 30-step turns vs `dispatcher=None`: identical per-replica step counters and
  identical recorded volume-move and report steps, in the same order;
- a deadline exactly on a turn boundary is serviced once;
- finite energies afterwards.

Trajectories are not compared bit for bit (thread scheduling is not deterministic on the CPU platform).
`step_all` itself stays a thin wrapper and is covered by the chignolin_9 production check (§9).

`tests/test_replica_admission_config.py` — parsing and validation of the three keys (YAML and CLI),
manifest recording, the MPS conflict error, and the "MPS not running" warning.

## 8. Files

- New: `gareus/replica_admission.py` (`AdmissionDispatcher`, `advance_replicas`), the three test files above.
- Changed: `gareus/cli.py` (flags, validation, MPS apply in `main`), `gareus/provenance.py`
  (`_method_settings` keys), `gareus/production.py` (dispatcher build after replica construction; `step_all`
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
