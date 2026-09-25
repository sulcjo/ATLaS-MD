# helper-cpus: durable asynchronous output and helper CPU allocation

Date: 2026-09-24  
Status: revised after adversarial design review (revision 1.1); implementation and runtime validation pending  
Repository baseline: `sulcjo/ATLaS-MD`, `main`, `4fe6a1484acb2516d1fb9ccba0ab262e955adfe8` (v0.8.3)  
Repository destination: `docs/superpowers/specs/helper-cpus/spec.md`

## 1. Objective and scope

Remove avoidable output work from the MD critical path while preserving a provable relationship between restart state, scientific records, replica assignments, and simulated time. Use automatically budgeted helper CPUs without starving CUDA submission threads.

Target workload: 236 production replicas, 59 per L40S across four GPUs, 192 allocated logical CPUs, MPS enabled, shared contact-sum CV force, active frozen-envelope stage-5 Pep-GaMD. Support smaller allocations, changing replica populations between epochs, CPU execution, and operation without local scratch.

Priority order is durability, scientific bookkeeping, bounded resource use, then throughput. No change to the Hamiltonian, timestep, random-number consumption, force precision, exchange algorithm, volume-move schedule, frame cadence, or thermodynamic eligibility rules is authorized by this optimization. Background workers must never control physics through their completion order.

Deliver in stages: durable output protocol first; asynchronous Parquet/checkpoint publication second; CPU autoallocation third; asynchronous trajectory encoding fourth. Synchronous trajectory rotation/indexing and campaign-level accounting are prerequisites of stage one, not deferred features. A synchronous execution mode must implement the SAME protocol, so scheduling can be A/B tested independently of correctness. Scratch mirroring and exclusive helper-CPU reservation are optional later extensions; neither is needed to ship the initial direct-persistent-output implementation.

## 2. Repository findings and integration points

All paths below refer to the pinned baseline, not assumptions about an installed cluster version.

| Existing component | Verified behavior | Required change |
|---|---|---|
| `gareus/production.py: save_production_checkpoint` | Captures integrator globals, binary context checkpoint and NPT controller state on the owning replica worker; immediately waits on each submitted task, serializing replicas; then calls `publish_generation` synchronously | Separate capture from publication; immutable indexed capture results; bounded cross-GPU capture concurrency |
| `gareus/correctness/checkpoint_store.py` | Immutable checkpoint generations, checksums, staging, file/directory synchronization and atomic root-manifest publication | Extend to reference an immutable output catalog, controller/bookkeeping state and output cursors as one restart transaction |
| `gareus/store.py` | Parquet flush converts buffers, compresses with Zstd, writes, synchronizes, hashes and updates manifests in the calling path | Detach batches; ordered asynchronous persistence; distinguish flush submission from durable completion |
| `gareus/store.py: _consolidate` (both writers) | `close()` consolidates Parquet then unlinks original chunks | Disable destructive close-time consolidation for v3; otherwise old committed catalogs lose their dependencies |
| `gareus/adaptive_production.py: AdaptiveRuntimePool, reconcile_runtime_pool_with_delivered_md` | Mutable pool charges are flushed separately; reconciliation only increases usage | Add canonical campaign-level span accounting and atomic epoch handoff; legacy monotonic reconciliation cannot be the v3 authority |
| `gareus/parquet_manifest.py` | Per-stream manifests govern Parquet files | Add immutable catalog snapshots for checkpoint bundles; mutable latest manifests cannot define an old generation |
| `gareus/store.py: SegmentRegistry, finalize_segment` | Segment status and end-step filtering support restart; completion depends on successful writer closure | Make segment closure part of committed metadata; end-step filtering alone is insufficient for the new protocol |
| `gareus/npt_driver.py: _emit_report, advance` | Gets state and invokes reporters on propagation path; reports after due volume moves | Keep capture/order there; background workers receive detached frame data only |
| `gareus/production.py: sample, attempt_exchanges` | Logging precedes exchanges at a common boundary; shared CV/bias computations are already reused | Preserve event ordering and freeze assignment/state metadata when observations are captured |
| `gareus/logger.py` | Dashboard already has a single background render executor and skips display frames when busy | Reuse and budget it; audit nested mutable data in snapshots; do not add a second renderer |
| `gareus/progress.py` | Rates use segment-local progress; aggregate totals multiply a common step/time by replica count | Add authoritative per-replica/epoch accounting and monotonic wall-clock timing |
| `gareus/system_setup.py: resolve_cpu_threads_for_replicas` | Divides an explicit CPU-platform budget across replicas | Keep separate from CUDA-host/helper allocation; do not infer free CPUs from this setting |
| `gareus/correctness/repo_adapters.py: sync_run_tree_quiescent` | Explicitly requires paused/flushed writers; is not a full output transaction | Never run it concurrently with mutable writers; add immutable bundle replication instead |

The inspected main has advanced beyond the earlier `ea002f29` discussion. The dashboard is already partially asynchronous. The real-space peptide-energy surrogate is outside this proposal.

## 3. Non-negotiable invariants

1. **Single owner of each live context.** All stepping, state reads, checkpoint creation, parameter updates and controller access execute on that context's established worker. Helpers receive bytes, numeric arrays and frozen metadata, never `Simulation`, `Context`, integrator or controller objects.
2. **One ensemble boundary.** A production checkpoint contains every expected live replica at the declared boundary, after all events required at that boundary. A partial replica vector is never published as a usable ensemble checkpoint.
3. **State and outputs commit together.** A published restart generation references the exact complete set of required samples, exchange events and enabled trajectory frames through that boundary. Queued, renamed or locally visible files alone do not establish durability.
4. **Snapshot identity is assigned at capture.** A writer must not consult the current replica assignment, step, box, epoch or timestep after capture.
5. **Exactly once in the canonical history.** Physical orphan files may exist after a crash. Accepted logical observations/events are neither duplicated nor silently omitted.
6. **No progress by submission.** Progress advances when integration actually finishes. Durable progress advances only when the authoritative generation commits. Output completion order never advances simulation time.
7. **No unbounded backlog.** Bound all queued, in-flight, copied and codec-workspace memory. Scientific data are never dropped on overload. Backpressure is preferable to a scientifically incomplete run.
8. **Visible errors.** Every scientific writer future is observed. Failure stops further advancement at a safe boundary, preserves the last valid commit, and produces a non-success status. Dashboard failure alone may remain nonfatal.
9. **Restart does not replay boundary events.** Store next-due cursors and event phase explicitly, including exchange attempt/RNG state and reporter schedules.
10. **Performance does not relax durability.** Async and sync modes use identical file synchronization, hashes, output cadence and retention policy.

## 4. Record identity, event order and time

### 4.1 Identity schema

Introduce a versioned `RecordHeader` carried by scalar batches, trajectory frame indexes and exchange records. Persist schemas in catalogs, including units and exact integer types.

| Field | Meaning |
|---|---|
| `campaign_uuid` | Persistent campaign identity |
| `epoch_id`, `segment_uuid` | Immutable physics/layout epoch and unique execution segment; new UUID after resume |
| `parent_generation_id` | Committed restart state from which this segment descends |
| `replica_uid` | Persistent walker/context lineage identifier; not a mutable window assignment or reusable array index |
| `state_id`, `assignment_revision` | Thermodynamic state and version of assignment at observation time |
| `topology_id`, `kernel_identity`, `window_snapshot_id` | Exact definitions needed to interpret coordinates, boosts and state energies |
| `stream_id`, `stream_seq` | Stream identity and contiguous sequence number assigned by producer |
| `production_step`, `absolute_step` | Integer step within the epoch's production and the explicitly defined legacy absolute-step convention |
| `event_phase`, `event_ordinal` | Within-step event ordering and stable ordinal for multiple events |
| `time_origin`, `timestep_schedule_id` | Exact physical-time reconstruction |
| `observation_id` | Shared identity for scalar and coordinate observations of the same configuration, when both were captured |

Physical record key: `(campaign_uuid, segment_uuid, stream_id, stream_seq)`. Logical observation key includes epoch, replica, production step, event phase and ordinal. A new segment UUID does not make an abandoned continuation scientifically distinct: lineage/cutoff rules select the canonical branch before duplicate checks.

Also assign an explicit `branch_id`: restart from the current head continues a branch; explicit rollback to an older generation forks it. A catalog view selects exactly one lineage. Walkers cloned into a new epoch have new replica UIDs plus a parent seed reference; inherited coordinates do not re-credit the parent's elapsed production time. Define `stream_id` deterministically from epoch, replica (or coordinator), record kind and schema; `stream_seq` is allocated by that stream's producer, never by a shared append completion order.

Per-stream sequence cursors continue from the chosen generation. Use unsigned 64-bit integers where appropriate; do not represent IDs or steps as float32. Labels assigned at encode/write time are forbidden. Include a coordinate/state revision if multiple snapshots at the same step can differ because of exchange or another state mutation. Independent runs/epochs never merge merely because they have matching numerical steps.

### 4.2 Ordering at coincident deadlines

Preserve the existing physics and sampling order:

1. Integrate a replica to its next due step.
2. Complete its due NPT move, including rejection/rollback, before capturing a due frame.
3. Capture due trajectory/scalar observations with pre-exchange assignments; explicitly label this phase.
4. At the ensemble barrier, perform all due exchange attempts, update assignments/parameters and advance exchange RNG and attempt counters in deterministic coordinator order.
5. Complete any existing permitted boundary state mutations and invalidate dependent caches. This proposal does not endorse rescue moves or change their scientific eligibility; enabled mutations must be recorded and included in checkpoint state.
6. Advance next-log, next-exchange, next-checkpoint and per-reporter cursors exactly once. Capture a post-events checkpoint and its bookkeeping at the same boundary.

Checkpoint assignment state may therefore differ from the assignment on a sample at the same integer step. This is correct and must be explicit, not repaired by relabeling old samples. Frames that do not coincide with ensemble log boundaries retain their own exact identities. Do not synthesize scalar observations just to match a trajectory.

At terminal production steps, retain the configured baseline event order, including whether a due terminal exchange occurs. Persist this policy. Restart begins after the saved event phase; it must not produce another scientific initial frame at the same boundary. An optional restart diagnostic frame uses a separate diagnostic stream.

### 4.3 Exact simulated time

Use integer steps plus a versioned rational timestep schedule as the authoritative representation. Parse decimal timesteps as decimal/rational values, not cumulative binary floating-point additions. For replica r over accepted production spans j:

`T_r(fs) = sum_j(delta_steps[r,j] * dt_fs[j])`

`aggregate_ns = sum_r T_r(fs) / 1_000_000`

Store a separate physical time origin and account for setup phases with their own timestep schedules if absolute physical time is required. Never infer physical time from GaMD's stage-machine `stepCount`: that counter may be deliberately seeded into stage 5. Existing `calib_steps + prod_done` is an indexing convention, not proof that every prior step used the production timestep. Cross-check OpenMM time against the schedule within numerical tolerance without replacing the exact schedule with the floating-point value.

Maintain separate counters:

- **Live completed production time:** current branch's successfully completed integration, per replica.
- **Committed production time:** canonical spans covered by a durable generation.
- **Executed work:** includes subsequently discarded/replayed computation; never presented as retained trajectory time. After an abrupt crash, any unjournaled executed tail is unknown rather than invented.
- **Frames/events captured, persisted and committed:** separate counts per stream; include expected counts and contiguous cursors.

Aggregate production time sums contributions over epochs and changing populations. It is not the newest replica count times a global step. Replica exchange changes state occupancy, not walker elapsed time. Budget consumption based on retained scientific time must use canonical committed spans on resume; optional hardware-cost budgets use a distinct work ledger.

At an unequal-progress observation, publish the actual per-replica vector and the minimum synchronized step; never multiply the fastest replica's step by N. A checkpoint still requires a common production boundary within that ensemble.

Use `monotonic_ns()` for in-process durations and UTC timestamps only for provenance. Restart starts a new wall-time segment. Rates exclude old simulated progress, count checkpoint drains in end-to-end timing, and distinguish live from durable throughput.

## 5. Durable checkpoint/output transaction

### 5.1 Bundle and state machine

Define a new checkpoint manifest schema `gareus_production_checkpoint_v3`, with required capability `coupled_output_catalog_v1`. Keep the existing authoritative root path `checkpoints/production_checkpoint_manifest.json`. Baseline code must refuse this unsupported schema; do not disguise new semantics as optional v2 fields that older readers silently ignore.

This root is authoritative for a standalone ensemble. For an adaptive multi-epoch campaign, section 5.5 defines a single campaign authority that pins the chosen ensemble generation(s); an epoch root alone cannot certify campaign progress. Establish a durable generation G0 with an empty production-output prefix and fully prepared initial state before the first production step. With periodic checkpoints disabled, retain G0 and still require a final commit; a crash before the final commit retains no later production output.

States:

`CAPTURING -> CAPTURED -> OUTPUTS_DURABLE -> GENERATION_STAGED -> COMMITTED`

Any pre-commit state may fail. Only `COMMITTED` is eligible for automatic resume and default scientific analysis.

A bundle includes:

- All replica binary checkpoints and readable integrator globals, indexed by replica UID.
- NPT controller state/RNG, coordinator/exchange RNG, assignments, counters, next-due cursors, and every non-context state that affects the next transition (including optional rescue/adaptation counters).
- Exact step/time vector, enabled output-stream roster and schedule definitions.
- Immutable window/kernel/topology snapshots, segment lineage and status.
- An immutable output catalog naming files by safe relative path with byte size, SHA-256, schema, record count, sequence ranges and relevant step bounds.
- Per-stream committed cursor and expected-event ledger; no use of max step alone as proof of completeness.
- Parent generation identity/hash and persistent commit sequence.

Catalogs may reference immutable prior catalogs to avoid copying all history on each checkpoint. The entire catalog dependency closure is retained and validated. Keep catalog objects independent of disposable checkpoint binary generations: a provenance link to an old checkpoint does not by itself require retaining all of its binary payloads forever. Mutable `segments.json`, window files, or latest Parquet manifests are compatibility views, never the authority for a committed bundle.

### 5.2 Capture and publication protocol

1. Finish the existing ensemble boundary. Freeze roster, assignments, RNG states, counters, metadata and schedule cursors with deep/immutable copies. Scalar sample production for the boundary is complete before sealing outputs.
2. Issue `seal_through(cursor)` to each enabled scientific stream. Detach partial buffers; rotate trajectory chunks. Writers subsequently place newer data into different chunks. A sealed file may not straddle this boundary with future records.
3. Capture each replica on its owning worker. Return `(replica_uid, checkpoint_bytes, globals, controller_state, step, time)` as an explicit result. Gather by UID, validate uniqueness/completeness and steps. Never share append-order payload lists.
4. Initially allow one concurrent capture per GPU, so four GPUs can capture in parallel without a 59-way readback burst. CPU contexts use a separately bounded budget. Serial capture remains a supported fallback. No replica starts stepping again until the complete bundle has been captured and validated.
5. Enqueue the frozen bundle into the single ordered checkpoint publisher. Resume MD once capture and queue admission succeed; disk persistence need not hold the MD barrier.
6. The publisher waits for all required output acknowledgments through the sealed cursors. Validate actual counts, contiguous sequences and schedule coverage, including empty streams. Finish immutable catalog, metadata and checkpoint payloads in a unique staging directory.
7. Synchronize every required file and directory, including externally located output chunks and their parent directories. Hash the finalized bytes. Publish the immutable generation directory and synchronize its parent. Atomically replace the root manifest on the same filesystem and synchronize its parent directory last, retaining the existing checkpoint-store durability mechanics.
8. Only then emit the durable acknowledgment and advance committed progress. Materialize legacy views from that committed snapshot afterward; if interrupted, rebuild them from the root/catalog on recovery.

In adaptive campaigns, the acknowledgment in step 8 waits for the campaign-head commit in section 5.5. The publisher checks its expected parent generation and writer-session ID under the publication lock before advancing any head. A stale retry may confirm the same committed generation, but may never replace a newer head. Canonical v3 data live in a separate namespace from mutable legacy output. Compatibility views are non-authoritative and new readers must never fall back to them after a v3 integrity error; an old analysis binary is unsupported for a live v3 campaign. Offer only explicit, immutable, generation-labeled legacy exports for old analysis tools.

No other code path may independently advance the root, promote a sample manifest to canonical history, or mark a segment complete. Completion of a chunk writer is not completion of a checkpoint.

### 5.3 What durability means

Guarantees depend on the storage system honoring synchronization and atomic-rename semantics. Rename alone protects visibility, not power-loss persistence. Use existing explicit file/directory synchronization and surface errors, including ENOSPC, quota, EIO and synchronization failure. Document any filesystem without the required guarantees; refuse strict mode rather than quietly ignoring failed synchronization.

A crash before root publication leaves the old generation authoritative. A crash around atomic replacement can expose old or new; recovery validates whichever root is present. If replacement succeeded but directory synchronization failed, durability is uncertain: report failure and do not claim either that the new generation committed or that the old root necessarily remains.

Node-local scratch protects against process failure but not loss of the node. Expose distinct `local_committed_generation` and `durable_destination_generation`. When a persistent mirror is configured as the durability target, dashboard “durable” means destination acknowledgment. Direct persistent output is the default when no scratch is configured.

Retention default: keep at least the two newest fully committed generations and all referenced output ancestors. Never prune a generation in use by a reader, publisher or mirror. Garbage collection traces references; it must not delete a shared old chunk just because its original generation is old. Unreferenced staging/orphan files are quarantined and cleaned only under exclusive recovery/maintenance control.

The current `ParquetSampleWriter.close()` and `ParquetExchangeWriter.close()` call `_consolidate()`, which deletes source chunks. For v3, close means seal/drain only. Initially disable compaction and automated garbage collection entirely. Later compaction must publish a new immutable catalog, preserve logical record IDs and record-order semantics, and retain all old files reachable from retained or pinned views. Same-step exchange ordering uses the event ordinal, not an unstable sort by step. Pin acquisition and GC selection must share a lock or equivalent race-free protocol; checking for a pin before a reader creates it is insufficient.

### 5.4 Exact output coverage

For every stream, record the starting cursor, contiguous committed cursor, expected schedule and count. Check no duplicates or gaps in sequence. A count and max-step alone do not prove the records are the right ones: verify record IDs against due steps/event phases, assignment revisions and producer batch inventories.

For periodic reports in `(S0,S1]` with an unchanged global period d, the expected count is `floor(S1/d)-floor(S0/d)`. Real implementation uses the saved next-due cursor to handle phase offsets, irregular terminal reports and cadence changes. Exchange streams count actual attempts, including rejected attempts, rather than guessing from sampling cadence. Enabled trajectory streams are mandatory dependencies even when scalar frames are more frequent. Explicitly disabled streams have recorded status and no fabricated obligations.

Publisher wait dependencies must be acyclic: output workers durably acknowledge sealed chunks without waiting for the root publisher; publisher then commits their catalog. Give the publisher a dedicated service lane or nonblocking coordinator state machine so it cannot occupy the only worker needed to satisfy its own fence.

### 5.5 Campaign accounting and epoch transitions

An epoch-level checkpoint does not atomically update `adaptive_runtime_pool.json`, adaptive decisions or the next epoch's roster. In an adaptive campaign, introduce one authoritative `adaptive_campaign_commit.json` pointing to an immutable campaign snapshot. It records the selected generation for each retained epoch, exact committed production spans, stream/catalog roots, pool policy, and the current transition/roster state. Publish child immutable generations first and the campaign head last. Per-epoch roots and mutable runtime-pool reports are derived views for these campaigns. Readers and restart follow the campaign head once, rather than independently sampling several latest epoch heads.

At every candidate checkpoint, derive retained time from disjoint span IDs `(branch, epoch, replica_uid, start_step, end_step, timestep_schedule)`; intervals use a documented half-open convention. Validate no overlap, then add only the newly committed suffix. Do not add a cumulative `prod_done` value on every checkpoint. Reconstruct `used_ns` from the canonical span set, not from replayed `consume()` calls. The existing reconciliation rule that only raises usage is retained for legacy campaigns only. Exported pool reports include the authoritative campaign generation and can be regenerated idempotently after a crash.

Keep the current budget's phase-inclusion policy explicit: seed/pull/calibration work, production retained time and hardware work are different quantities. This optimization must not silently change which phases the user's pool charges. For included non-production work, define its own committed phase spans. Before concurrent work starts, the coordinator records/reserves its worst-case permitted budget so multiple ensembles cannot each spend the same remaining allocation; outstanding reservations and completed-but-uncommitted work reduce admission headroom. Release or convert reservations exactly once on commit or recovery. Never reserve future CPU time by inventing completed MD time.

Epoch transition is a committed state machine: `EPOCH_RUNNING -> EPOCH_CLOSED -> NEXT_EPOCH_PREPARED -> NEXT_EPOCH_RUNNING`. Commit closure before selecting/releasing old state. Capture any adaptive decision inputs, decision/RNG state, new immutable roster and budget reservation in the prepared state before new propagation. Crash recovery repeats neither the pool charge nor an already committed stochastic selection. Kill tests must cover each head publication and the gap between child generation creation and campaign publication; unreferenced child generations remain provisional. The initial implementation may enforce one active production ensemble per campaign while replicas remain parallel; multi-ensemble admission is enabled only after reservation tests pass.

### 5.6 Publication cost and validation scope

Validation of a new commit reads/decodes newly sealed output and verifies its inherited catalog identities against pinned committed parents. Do not rehash or re-decode the entire accumulated trajectory on every checkpoint: that creates approximately quadratic cumulative I/O. Maintain per-stream exact rolling counts, cursor and catalog-delta hashes. Full dependency verification remains required on cold strict recovery and is measured separately; live readers can reuse validation for already pinned immutable objects within their process.

Compare decoded new frame headers/sidecars against an independently generated due-event schedule in tests. A producer that forgets both a frame and its expected-count increment must not pass merely because its own two counters agree. Metadata and output-directory creation must durably synchronize the newly created parent chain, not only the leaf directory.

## 6. Asynchronous writer architecture

### 6.1 Services and ownership

Add small modules with explicit interfaces, suggested names:

- `gareus/output_records.py`: immutable headers, frame packets, counters and time schedules.
- `gareus/async_output.py`: bounded batch queues, ordered stream writers, durable futures and errors.
- `gareus/checkpoint_pipeline.py`: capture, fences and commit orchestration; use checkpoint_store for durable publication primitives.
- `gareus/output_catalog.py`: immutable output catalogs and recovery validation.
- `gareus/helper_cpus.py`: allocation discovery, budgets and safe worker placement.

Suggested contracts:

```python
submit_batch(batch) -> Ticket                 # admission only
seal_through(stream_cursors) -> DurableFence   # detached final batch + exact coverage
capture_ensemble(boundary) -> FrozenBundle    # owning context workers only
publish_bundle(bundle, fence) -> CommitFuture # acknowledged only after durable root
drain(deadline) -> DrainResult                # no swallowed errors
```

Submission order defines each stream's sequence. Completion order may differ across streams but cannot change output order or assignment labels. One actor owns each output file and chunk counter. A bounded pool may multiplex many replica streams; do not create 236 new helper processes.

### 6.2 Parquet and trajectory handling

Detach sample/exchange column buffers by swapping ownership, not by clearing the list a worker is using. Capture all mutable nested metadata before submission. Native compression and hashing may use threads; Python-heavy transformation/encoding may use spawned helper processes after measurement. Never fork a live CUDA process or initialize OpenMM/CUDA in helpers. Limit native library thread pools per helper.

Trajectory workers receive detached positions, box, time, step, atom selection, topology identity and state metadata. Do not call an ordinary `reporter.report(sim, state)` later with the live sim: reporters can read mutable step/topology/context information. Implement a packet-based encoding adapter and test each supported format.

Rotate XTC/DCD into immutable per-replica chunks at checkpoint seals or an earlier size limit. Store an exact sidecar mapping every frame to its record header and frame ordinal. The sidecar is authoritative for integer identity/time; the format's floating time may be rounded. Validate frame count and ordering by decoding the finalized file. Hash both files and publish neither as complete without the other. Never append to an already committed trajectory chunk; continuation opens a new chunk. Frame arrival order across replicas need not be globally sorted.

Trajectory offloading stays disabled for an unsupported reporter until its packet adapter passes tests. Do not silently omit that stream from the checkpoint fence. Initial implementation may keep encoding synchronous while still publishing immutable chunks through the new catalog.

The existing renderer may coalesce obsolete display updates. Deep-copy nested values it reads, including histories, or provide an immutable render model; its current shallow snapshots are not a general asynchronous-data contract. Scientific output must not share the dashboard's drop-on-busy policy.

### 6.3 Backpressure and errors

Use byte limits as well as item counts; include serializing/IPC copies, active codec buffers and checkpoint payloads. Default output-buffer budget: the smaller of 1 GiB and 1% of effective job memory limit, leaving a separately checked checkpoint-capture allowance. Determine the limit from allocation/cgroup constraints rather than host RAM alone. Profile actual checkpoint size before enabling asynchronous capture.

Reserve worst-case packet/codec bytes before the context owner extracts a frame, not after allocating 236 simultaneous snapshots. Memory credits belong to buffers until the final consumer releases them. A record larger than the configured queue limit takes an explicit bounded synchronous/streaming path, rather than waiting forever for impossible credits. Keep separate reserved capacity for seal/fence/control messages. A producer may not hold a stream/publication lock while waiting for memory that only the writer can release. Early batch flush is allowed to free credits and must preserve the same output schedule.

Allow at most one checkpoint bundle in flight. At the next checkpoint boundary, wait for it to commit before capturing another. This bounds outstanding bundles and the amount of further MD admitted, and prevents stale publication from overwriting a newer generation. It does not bound wall-clock durability lag during a storage stall; report that lag and apply the stop/error policy explicitly. Do not silently skip checkpoint requests. On insufficient memory, use synchronous durable publication with bounded staging rather than losing records or claiming unbounded “free RAM.” Implement an equivalent consistent-boundary streaming capture fallback if a whole ensemble cannot fit in memory.

Full output queues apply backpressure to the appropriate producer. No spin polling. Expose queue bytes, oldest packet age and blocked time. Drain with bounded-time waits that keep processing stop/error signals. A failing writer poisons its stream and prevents any dependent generation from committing; fail the run visibly. Worker retry uses the same ticket/record identity and verifies existing immutable artifacts instead of making duplicate records.

## 7. Automatic assignment of helper CPUs

### 7.1 Discover the actual allocation

Do not use host CPU count or `192 - replica_count`. A replica is not a reserved CPU and GPU submission threads consume variable fractions of CPUs. One job-level allocator owns the helper budget across all ensembles, epoch workers, renderers and publishers in that job; individual replicas or ensembles must not each independently claim H CPUs. Separate scheduler jobs retain their own OS-enforced allocations; this feature must not repin another job.

Build an `AllocationSnapshot` from:

1. Save the launching thread's allowed mask before any application pinning. Intersect that mask with the effective cgroup cpuset and any verified scheduler mask. Do not union masks of differently bound replica threads to grant helpers extra CPUs. A broader job-level mask may only be used when independently verified as authorized; otherwise retain the conservative launch mask. Record per-thread restrictions where they differ.
2. Effective CPU quota, including restrictive ancestor cgroups; a quota limits CPU-time capacity but does not identify CPU IDs.
3. Explicit user total budget, where provided; scheduler counts are conservative caps/cross-checks and never permission to widen an OS mask.
4. Topology: NUMA node, physical core and SMT sibling group for each allowed logical CPU.

`B = min(number_of_allowed_logical_CPUs, effective_quota_CPUs, explicit_job_cap)` with unavailable terms omitted and diagnostics recorded. Fractional quotas remain fractional. Distinguish logical CPUs from physical cores in reports. `OPENMM_CPU_THREADS`, BLAS thread settings and CPU-platform replica budgets are not evidence of spare CUDA-host capacity.

### 7.2 Conservative automatic budget

Default `--helper-cpus auto` is opportunistic shared scheduling, not exclusive CPU reservation. Start with no additional CPU-heavy helpers; synchronous mode establishes a baseline. After 30 seconds of representative steady propagation, collect non-helper job CPU demand in 1-second windows, including measured MPS-server CPU activity attributable to the job where available. Mark incomplete attribution. Exclude startup compilation and retain statistics for both normal propagation and output windows.

Let D95 be the 95th percentile CPU equivalents consumed by non-helper work. Let `R = min(B, max(2, ceil(0.02*B)))` be safety headroom. A conservative candidate is:

`H = min(user_helper_cap, 4, floor(max(0, B - D95 - R)))`

H is the budget for simultaneous helper CPU work, not the number of sleeping I/O threads. Existing render work and native codec threads count against it. The output writer and checkpoint publisher may be separate sleeping service threads, but CPU-heavy work must acquire helper tokens; publisher waits must release tokens.

**Progress exception:** H limits optional overlap with propagation, not permission to finish essential output. If H becomes zero, stop admitting optional asynchronous work, pause producers at a safe scheduling point, and let already-owning writers drain with one service permit while propagation is paused. Account that CPU use explicitly; never exceed the job allocation. Do not reduce the effective permit count below active work or hand an open stream to another writer concurrently. Once drained, use synchronous output. Test the H=1 to H=0 transition with full queues and an awaiting checkpoint publisher. Under fractional quotas the service still executes subject to the OS quota; H=0 does not mean the job has no permission to execute any instructions.

For B=192 and D95 approximately 179, R=4 and H is capped at 4. The historical 179 value is an average benchmark observation, not an actual measured D95 or a promise that four CPUs will be available. If demand reaches the cap, H can be zero; sync persistence remains functional. Storage-latency overlap may still be useful with a single waiting I/O lane, provided it is measured not to harm MD.

Re-evaluate at safe epoch boundaries or at most every 60 seconds. Require three favorable windows before growing by one CPU; shrink at the next drained queue boundary when headroom disappears or sustained CPU throttling rises. Do not recreate a pool during active writes. Replica roster changes trigger rediscovery. Missing usage information defaults to no extra CPU-heavy workers with a visible reason. Explicit user helper counts override the heuristic but cannot exceed allocation limits.

D95 measures CPU time received, not unmet CPU demand. A helper that starves MD can make measured non-helper CPU use fall, falsely suggesting more headroom. Retain the helper-disabled reference demand for the current workload and use the larger of reference and current estimates. Never increase H when MD propagation rate degrades, CPU-pressure/throttling grows, or producer backpressure is suppressing measured demand. Growth requires an output-free propagation canary consistent with the baseline; otherwise freeze or reduce H. Whole-job usage from cgroups must not be added again to process/MPS usage already included in it. Independent or ancestor-sharing jobs consume quota too; unknown shared-quota contention prevents optimistic growth. Helper pinning/concurrency tokens do not themselves enforce an OS CPU-time quota.

On ordinary GIL-enabled CPython, a spare logical CPU does not allow another Python thread to execute Python bookkeeping simultaneously. Measure GIL contention or compare isolated spawned helpers before attributing a benefit to CPU allocation. JSON transformations, Python-side frame work and the existing renderer may contend with the coordinator even if they run on different CPUs. Prefer native work that releases the GIL or measured process isolation; do not assume a free-threaded Python build.

### 7.3 Placement and implementation limits

Auto/shared mode places helpers only within the allowed mask, using measured sustained spare capacity and topology as preferences. Prefer unused physical cores over busy SMT siblings when evidence exists. Do not claim those CPUs are exclusive or infer freedom from a single idle sample. Spread substantial encoding work without concentrating it on one GPU's NUMA-local submission CPUs; record effective affinity, not just requested affinity.

Optional `--helper-affinity reserved` is an explicit startup-only policy. Partition helper and simulation CPU masks before CUDA/OpenMM context and native-worker creation; budget complete SMT sibling groups where practical and report the actual rounded reservation. Respect pre-existing scheduler bindings; fail or fall back visibly if all participating threads cannot be placed. Do not silently steal CPUs by repinning a live simulation or a shared MPS server. CPU-platform thread budgets must be recalculated against the reduced simulation allocation in reserved mode.

Spawn helper processes with a fresh interpreter, no CUDA imports, explicit affinity and one-thread native pools unless a token budget allows more. Thread helpers must avoid global Arrow/BLAS configuration changes that alter existing simulation libraries; process isolation is preferred when such settings are necessary. Audit nested compression pools and dashboard work to avoid multiplying H by hidden native thread counts.

## 8. Resume, termination and persistent mirroring

### 8.1 Recovery

Acquire the existing exclusive run lock before recovery/publication. Read and validate the authoritative root once, pin the referenced generation/catalog closure, verify hashes, schemas, topology/kernel compatibility, all replica identities and every required output cursor before mutating a live context.

Unpublished output beyond the committed cursors is excluded from default analysis and the resumed lineage even if its files are intact. Keep it quarantined for diagnostics. Start a new execution segment with `parent_generation_id`; restored reporter/stream cursors prevent duplicate boundary frames. Reconstruct registry and window views from committed metadata. Glob enumeration or “latest modification time” must never discover scientific history.

Use separate explicit live/provisional views for dashboards. All scientific readers must respect catalogs, including scalar query, MBAR loaders, frame matching, thermodynamic decomposition and trajectory concatenation. Migration is incomplete until every such path stops treating any visible frame as canonical.

Corrupt latest committed generation is a hard validation failure. Older-generation recovery is explicit, names the discarded interval and records a lineage branch; do not silently fall back. Legacy v1/v2 checkpoints remain supported under their existing validation policy. Resuming them creates a clearly labeled legacy parent and a v3 child; do not retroactively assert exact frame coverage that legacy files cannot prove.

### 8.2 Clean stop and walltime

Signal handlers only set stop flags. At the next safe boundary, complete pending event ordering, finish an existing publication, capture the final boundary if needed, seal and drain scientific writers, publish the terminal generation, and wait for the configured durability-target acknowledgment. Mark completion only after the final transaction succeeds. Runtime cadence/cursors and final segment status are inside that transaction.

Respect the scheduler's known grace deadline. Estimate required drain time from recent p95 capture/persistence times plus margin and request stop early where the launcher supports it. If time runs out, preserve the last committed generation, emit its identity and lost-tail bounds, and exit non-success; do not claim the latest live step was saved. An uninterruptible filesystem call may exceed the deadline, so hard-kill recovery must remain safe independently of graceful draining.

### 8.3 Scratch-to-persistent replication

Use a new immutable-bundle mirror; do not background the existing quiescent tree copier. Pin a committed source generation and its full catalog closure, copy to destination staging, verify hashes, synchronize destination data and directories, then atomically publish destination root last. Concurrent source writers operate only on new chunks. Preserve generation order and ancestor closure; retain source dependencies until destination acknowledgment. Apply bounded queues and bandwidth/concurrency limits to mirroring as well.

On node loss, only a destination-committed generation is guaranteed available. Report live/local/destination times separately. Initial deployment without scratch uses a single durable destination and avoids this extra mechanism.

## 9. Configuration and observability

Proposed CLI/config additions; names are subject to the repository's normal config mapping, semantics are normative:

| Option | Initial behavior |
|---|---|
| `--async-output off\|on` | Off until promotion gates pass; both modes use v3 protocol |
| `--helper-cpus auto\|N` | Auto budget above; N includes all CPU-heavy output/render helpers |
| `--helper-max-cpus N` | Default 4 |
| `--helper-affinity shared\|reserved` | Shared default; reserved startup-only |
| `--output-buffer-mib auto\|N` | Enforce memory-accounted admission |
| `--checkpoint-capture-concurrency auto\|N` | Auto one active capture per GPU, with CPU fallback |
| `--async-trajectories off\|on` | Off until packet encoders pass coverage tests |
| `--durability-target output\|mirror` | Output default; mirror requires configured persistent destination |

Preserve existing checkpoint/report/exchange intervals. Do not add an option that claims durability while skipping fsync. All enabled scientific streams are required by default; display-only progress/TUI files are explicitly optional. A small explicit audit event stream for replica roster/time changes is required and belongs in the catalog.

Publish `helper_cpu_assignment.json` with allocation evidence, masks, quota, topology, D95 windows, H, actual worker/native-thread counts and reasons for fallback. Persist policy changes as diagnostics, not dynamics events.

Telemetry includes live/common/committed step vectors, local/destination generation IDs, production-time totals, per-stream expected/captured/persisted/committed counts, oldest queue age, bytes in flight, capture pause, encode/write/hash/fsync times, publisher wait, durability lag, backpressure time, helper CPU use and errors. Do not sum overlapping helper durations into wall time; measure the actual critical-path pause separately.

## 10. Implementation sequence and review gates

1. **Protocol first:** record identities, exact time ledger, catalogs and v3 manifest, synchronous publication, resume/query integration. Establish reference fixtures before adding concurrency.
   This includes synchronous trajectory chunk rotation/indexing, disabling destructive close-time consolidation, G0, and adaptive campaign-head/accounting changes. Test an actual sync v3 run before proceeding; merely adding fields to an epoch checkpoint is not completion of this stage.
2. **Checkpoint split:** immutable indexed capture; concurrent capture across GPUs; durable futures; root still published in order. Audit every mutable coordinator/controller counter used after restart.
3. **Parquet offload:** detachable buffers, ordered writers, fences, bounded queues, error propagation. Make flush-at-checkpoint await the fence in the publisher rather than in MD, without weakening its guarantee.
4. **Helper allocation:** allocation discovery, telemetry, conservative token budget, native-thread accounting, shared/reserved affinity modes.
5. **Trajectory packets:** chunk rotation/index, codec adapters, exact sample/frame join and reader support. Keep synchronous fallback for unsupported reporters.
6. **Lifecycle integration:** signals, final generation, mirror, retention and recovery. Update adaptive epoch transitions, all query/analysis consumers and launcher status interpretation.
7. **Promotion:** pass correctness/fault gates, then the final benchmark protocol below. No default enablement based on the earlier integrator-only harness.

Suggested new tests: `test_output_catalog_transaction.py`, `test_async_output_ordering.py`, `test_checkpoint_output_fence.py`, `test_frame_time_ledger.py`, `test_helper_cpu_allocation.py`, `test_async_output_crash_recovery.py`. Extend existing checkpoint, segment, trajectory, replica-affinity and union-loader tests rather than duplicating their assertions.

Acceptance includes deterministic synthetic scheduling tests for deliberately out-of-order workers, unequal replica progress, coincident deadlines, zero observations, non-divisible terminal steps, timestep/roster changes and repeated resumes. Test actual public reader paths, not only the writer's internal counters.

## 11. Failure cases that must be demonstrated

| Injected condition | Required outcome |
|---|---|
| Slow replica / out-of-order checkpoint completion | Correct UID mapping; no mixed-step ensemble |
| Assignment changes after frame capture but before encoding | Original state label and box/time remain intact |
| Volume move accepted/rejected at frame step | Frame represents the post-decision state exactly once |
| Chunk count correct but one frame duplicated and another missing | Coverage/identity validation rejects it |
| Disk full, quota, EIO, fsync failure, writer process death | No dependent commit; visible run failure; prior committed closure retained |
| SIGKILL during capture, chunk write, hash, generation rename or root update | Old or new complete bundle recovered; no hybrid accepted |
| Root rename succeeds, final directory fsync errors | No successful durability acknowledgment; recovery validates actual on-disk state |
| Crash after durable commit but before coordinator acknowledgment | Recovery recognizes committed generation without duplicating data |
| Output ahead of checkpoint / abandoned resumed branch | Ahead/abandoned records excluded from default scientific history |
| Queue saturation and exhausted memory budget | Bounded backpressure or synchronous fallback; no silent drops |
| Queue/publisher single-worker starvation | No cyclic wait or deadlock |
| Wall-clock jump / restart / changed replica count | Exact retained time and valid segment-local rate |
| Restricted affinity, fractional quota, SMT, missing telemetry | Allocation respects constraints and reports conservative fallback |
| Mirror slow or failed / source GC attempt | Destination root remains valid; required source files retained |
| Corrupt referenced frame index or old ancestor chunk | Full bundle validation fails before loading contexts |
| Existing writer close/consolidation after a v3 commit | Every retained generation remains readable; no referenced source chunk is deleted |
| Crash between epoch checkpoint, runtime-pool report and campaign-head update | One canonical retained-span total; no double charge, forgotten charge or uncommitted epoch selection |
| H shrinks to zero with full output queues | Essential drain completes under paused propagation; no token starvation |
| Packet exceeds queue limit / all producers reserve memory concurrently | Explicit bounded fallback or early backpressure before allocation; no infinite wait or memory explosion |
| Helpers steal CPU and apparent non-helper usage decreases | Allocator does not interpret starvation as new spare capacity |
| Crash before first periodic checkpoint | G0 is the only canonical state; no fabricated production history |
| Stale publisher retry / delayed mirror from an older writer session | Authoritative head never moves backward or across an unapproved branch |
| Repeated commits over increasing retained history | Per-commit validation does not reread the complete historical payload |

Fault injection of process death cannot by itself prove power-loss behavior. Test crash recovery at every publication hook and separately document filesystem durability assumptions. Verify stale/uncommitted directory visibility never becomes a reader fallback.

### 11.1 Adversarial review record (revision 1.1)

Verdict on the original draft: **sound direction, not yet an implementation-ready durability contract**. The following are design/integration findings, not claims that existing synchronous production is broken. No new implementation, cluster benchmark or crash test was executed during this review. Evidence consists of source inspection and explicit counterexample schedules. Corrections above and below are normative; they still require implementation tests.

| ID / severity | Counterexample or evidence | Resolution |
|---|---|---|
| R1 / blocking | G1 references Parquet chunks; existing `close()` consolidates and deletes those chunks; G1 then fails validation | Disable destructive consolidation for v3; later use reference-aware compaction/GC |
| R2 / blocking | Epoch output commits, pool report or epoch decision does not; resume consumes or reconciles a different budget/history | One adaptive campaign head, immutable span accounting, committed epoch transitions and reservations |
| R3 / high | Queued writer needs a helper token; H is set to zero; coordinator waits for writer before switching to sync | Explicit paused-propagation drain permit; no reduction below active work |
| R4 / high | Every producer allocates a large frame before enqueue, or one packet exceeds the byte ceiling; memory exceeds limit or admission never succeeds | Reserve memory before extraction; bounded oversized-record path and separate control credits |
| R5 / high | Helper steals submission CPU; measured MD CPU demand falls; allocator raises H again | Baseline reference demand, throughput/pressure veto, conservative masks, no double-counted CPU telemetry |
| R6 / high | Stage-one v3 needs durable trajectory prefixes but chunk rotation/indexing is deferred to stage four | Synchronous trajectory transaction support is now a stage-one prerequisite; only asynchronous encoding is deferred |
| R7 / high | Runtime interprets old readers/manifests or an uncommitted next epoch as canonical | Separate v3 namespace, explicit campaign authority and fixed-generation legacy exports |
| R8 / medium | Every commit rehashes/redecodes all history; cost grows with campaign length | Incremental commit validation, full cold-recovery audit, history-size benchmark |
| R9 / medium | Three paired runs and correlated checkpoint cycles are treated as sufficient distribution-free 95% evidence; noisy p95 is a hard gate | Three pairs are screening only; independent-run confirmatory design and explicit tail uncertainty |
| R10 / medium | “Free CPU” threads execute Python bookkeeping under one GIL; work still blocks coordinator bytecode | GIL-aware attribution, native operations or tested spawned-process isolation |

Remaining limitations: storage durability depends on the filesystem contract; old legacy records cannot gain exact provenance retroactively; GPU checkpoint compatibility remains version/platform dependent; performance benefit and CPU-allocation stability remain unmeasured. Coupling every enabled trajectory stream to restart means a failed trajectory writer deliberately prevents a newer checkpoint commit. That is a chosen strict policy with a liveness cost, not a guarantee that newer restart state survives all output failures. A future restart-only emergency save must use a distinct noncanonical status and explicit missing-output ledger; it must never masquerade as a complete scientific bundle.

## 12. Sources and baseline evidence

Repository code links are pinned to the reviewed commit:

- [Production, snapshot and checkpoint orchestration](https://github.com/sulcjo/ATLaS-MD/blob/4fe6a1484acb2516d1fb9ccba0ab262e955adfe8/gareus/production.py)
- [Checkpoint generation store](https://github.com/sulcjo/ATLaS-MD/blob/4fe6a1484acb2516d1fb9ccba0ab262e955adfe8/gareus/correctness/checkpoint_store.py)
- [Filesystem durability primitives](https://github.com/sulcjo/ATLaS-MD/blob/4fe6a1484acb2516d1fb9ccba0ab262e955adfe8/gareus/correctness/_io.py)
- [Scalar writers and segment registry](https://github.com/sulcjo/ATLaS-MD/blob/4fe6a1484acb2516d1fb9ccba0ab262e955adfe8/gareus/store.py)
- [Parquet manifests](https://github.com/sulcjo/ATLaS-MD/blob/4fe6a1484acb2516d1fb9ccba0ab262e955adfe8/gareus/parquet_manifest.py)
- [NPT driver and report ordering](https://github.com/sulcjo/ATLaS-MD/blob/4fe6a1484acb2516d1fb9ccba0ab262e955adfe8/gareus/npt_driver.py)
- [Existing background dashboard rendering](https://github.com/sulcjo/ATLaS-MD/blob/4fe6a1484acb2516d1fb9ccba0ab262e955adfe8/gareus/logger.py)
- [Progress calculations](https://github.com/sulcjo/ATLaS-MD/blob/4fe6a1484acb2516d1fb9ccba0ab262e955adfe8/gareus/progress.py)
- [Quiescent scratch copier](https://github.com/sulcjo/ATLaS-MD/blob/4fe6a1484acb2516d1fb9ccba0ab262e955adfe8/gareus/correctness/repo_adapters.py)
- [Adaptive runtime pool and reconciliation](https://github.com/sulcjo/ATLaS-MD/blob/4fe6a1484acb2516d1fb9ccba0ab262e955adfe8/gareus/adaptive_production.py)
- [Query/legacy data discovery](https://github.com/sulcjo/ATLaS-MD/blob/4fe6a1484acb2516d1fb9ccba0ab262e955adfe8/gareus/query.py)
- [Python threading/GIL documentation](https://docs.python.org/3/library/threading.html) distinguishes I/O overlap from CPU parallelism on standard CPython.
- [Python affinity API](https://docs.python.org/3/library/os.html#os.sched_getaffinity) and [Linux cgroup constraints](https://docs.kernel.org/admin-guide/cgroup-v2.html) inform allocation discovery; these do not guarantee an exclusive CPU reservation.
- [OpenMM Context API](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.Context.html) documents checkpoint/state interfaces and checkpoint compatibility limitations.

## 13. Final benchmark and promotion protocol

### 13.1 Questions to answer

Does offloading increase durable aggregate production throughput, reduce checkpoint/report pauses, and preserve exact record coverage and restart behavior? Does automatic helper allocation avoid starving CUDA submission? Distinguish savings from asynchronous persistence, concurrent capture and trajectory encoding.

Historical reference only: the shared-contact integrator harness measured 2,630/2,618 ns/day/node at 236 replicas under MPS, with approximately 179 CPU equivalents used. It lacks the full output/publication workload and is not the control arm for this change. [Raw historical log](https://github.com/sulcjo/ATLaS-MD/blob/4fe6a1484acb2516d1fb9ccba0ab262e955adfe8/docs/superpowers/specs/2026-09-23-pep-gamd-realspace-vpep-surrogate/benchmarks/results/c8_sharedcv_2608721.log)

### 13.2 Fixed workload

Use the same node class, CPU allocation, four L40S, MPS, 236 replicas, physics inputs, frozen envelope, shared CV layout, precision and seeds. Record exact ATLaS-MD/OpenMM/CUDA/driver versions, GPU clocks/power, topology, masks, storage type and full config. Start each arm from the same validated checkpoint in a separate output tree. Verify stage 5 and a nonzero actually applied boost on appropriate replicas; a config label alone is insufficient.

Keep production timestep (currently configured 3.5 fs), 250-step scalar/trajectory cadence, 400-step exchange cadence, 200-step NPT attempts and 20,000-step checkpoints if confirmed in the selected fixture. Resolve values from the actual config and record them; never silently force these numbers onto a changed campaign. Match atom selection and enabled output formats. Maintain identical sync/hash/durability policy between protocol-comparable arms.

### 13.3 Arms

| Arm | Purpose |
|---|---|
| A0: pinned existing production | Measure current user-visible baseline; report its weaker cross-output transaction semantics separately |
| A1: v3 synchronous, serial capture | Correctness-equivalent control; isolates cost of the stronger durability protocol |
| B: A1 + bounded capture concurrency | Quantify capture pause reduction across GPUs |
| C: B + asynchronous Parquet/checkpoint persistence, helper auto | Primary candidate |
| D: C + packet-based trajectory encoding | Optional candidate after encoder tests |
| E: best candidate with fixed H=1,2,4 | Calibration of the auto allocator; no need to run all after a clear failing gate |

Compare C/D primarily against A1, and report net change against A0 honestly. Hold capture mode constant when isolating writer speedup. Do not attribute gains from changed frame cadence, omitted fsync, disabled trajectories or a different Hamiltonian to offloading.

### 13.4 Execution length and measurement

First run short functional tests with 4–8 replicas and accelerated checkpoint cadence. Then use paired alternating A1/candidate runs on the full node. Warm up for at least 30 seconds and until CPU autoallocation has stabilized; use a warmed shared checkpoint to exclude context construction from steady-state results.

For full production-cadence performance, first collect three paired screening runs with at least five checkpoint cycles per run, including final output drain and durable destination acknowledgment. This may reject an ineffective change but is not sufficient for a distribution-free 95% superiority claim. Predeclare a confirmatory set of at least six independent paired runs for the selected candidate; do not select the winning candidate and reuse its tuning measurements as independent confirmation. At the historical roughly 37 steps/s/replica, five 20,000-step cycles are about 45 minutes per arm. Six pairs are about nine node-hours, excluding startup; obtain inconclusive/opt-in results if that measurement budget is unavailable rather than weakening the evidence label.

Randomize or balance AB/BA order. Treat each paired run, not each correlated checkpoint cycle, as the independent unit. Predeclare the estimator (paired throughput ratio and its median), a valid paired interval/test, the sample count and stopping rule. Report interval method and attainable coverage; with only three pairs, even all-positive differences cannot meet a two-sided 5% exact sign test. Avoid a narrow bootstrap interval from treating within-run cycles as independent replicates. Use low-overhead phase timers and a short targeted profiler trace only where attribution remains unclear. Store all cycle/run values and uncertainty; do not hide outliers or choose the best run.

Primary metric:

`durable_node_ns_per_day = newly_committed_aggregate_production_ns / end_to_end_wall_seconds * 86400`

Wall seconds span first measured propagation through last durable output acknowledgment. Report separate startup, graceful-stop and mirror costs. Collect per-replica progress spread, p50/p95 capture pause, output stalls, commit latency, queue wait, durability lag, helper/non-helper CPU, RSS/peak owned-buffer bytes, cgroup throttling, context switches, GPU utilization/power and storage bandwidth. CPU and GPU utilization alone are not throughput evidence.

Include two scaling/stall checks: (1) replay identical new output against increasing retained catalog sizes to detect cumulative rehash/decode or metadata growth; report cold strict-recovery time separately; (2) inject a bounded storage stall and H=0 transition under the full replica load, then verify that memory remains bounded, progress resumes and all records reconcile. Keep injected-stall runs separate from clean speedup estimates. Measure producer extraction/copy/IPC time explicitly: moving compression to a process is not useful if serialization costs more than it saves. A warm process is not equivalent to restoring a checkpoint; warm and log each newly created process/context before measurement.

### 13.5 Correctness and crash benchmark

Run deterministic synthetic/reference fixtures to compare exact event identities, counts, assignment revisions, RNG/bookkeeping restoration and time ledgers against uninterrupted execution. Force reorder/delay helpers and exercise all section 11 publication hooks. Decode trajectories and query through real analysis loaders. Verify every committed generation contains the exact scheduled output prefix, with zero duplicate/missing records and zero accepted records from abandoned tails.

For real CUDA/MPS runs, compare captured payload/metadata identity where applicable and physics invariants after resume: positions/velocities at capture, box, integrator globals, controller state, assignments, schedules, boost accounting and energy within the established precision tolerance. Do not require bitwise equality of long chaotic trajectories between independently scheduled nondeterministic GPU runs. Asynchronous encoding must nevertheless encode the exact frozen source packet, within the chosen lossy trajectory format's existing precision.

Kill/restart tests cover each publication phase and repeated resumes at coincident log/exchange/NPT/checkpoint boundaries. Inject writer errors and storage delays; verify graceful-stop behavior with a short scheduler grace period. Test default persistent output first; add scratch/mirror node-loss simulation only when that mode is implemented. No benchmark pass may waive a durability or bookkeeping failure.

### 13.6 Promotion gates and result artifact

Hard correctness gates: zero accepted partial/mixed generations; exact identity/coverage/time reconciliation; no future/abandoned frames in canonical analysis; no swallowed scientific-writer error; bounded memory and no deadlock; allocation remains within permitted resources; successful resume from the committed bundle using normal entry points.

Proposed performance gate for default enablement: median durable throughput improves by at least 3% against A1 with a positive paired 95% confidence interval from the independent confirmatory design, and no reproducible regression larger than 2% against A0. Treat these as engineering thresholds, not predicted gains. Require a meaningful checkpoint-pause reduction (target at least 25% median) to justify concurrent capture; if capture is negligible, omit it. Report p95 with the raw pause sample count and uncertainty; do not use a p95 estimate from 15 pauses as a decisive gate. A tail claim requires a separately adequate, predeclared measurement budget. Auto allocation should be within 3% of the best validated fixed small helper budget on a held-out workload. If intervals are inconclusive, keep opt-in and report that result rather than extending tests indefinitely. A durability-only improvement can be promoted separately with its cost disclosed.

The benchmark report must end with: complete config/commit manifests, per-arm durable throughput and uncertainty, pause/lag/memory tables, correctness/fault-test results, helper CPU assignments, storage assumptions, and one decision: promote, opt-in only, or reject. No speedup is claimed until these measurements exist.
