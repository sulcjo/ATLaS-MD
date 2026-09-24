# ATLaS-MD: seven further performance upgrades

Date: 2026-09-24  
Status: implementation specification; runtime validation and performance measurements pending  
Repository baseline: `sulcjo/atlas-md`, commit `36ebbd9e1fbb20e5fde0e51c30aa3d9c46531a33`  
Suggested repository destination: `docs/superpowers/specs/performance-upgrades/spec.md`

## 1. Objective and scope

Improve aggregate, durably recorded production throughput for the current target: 236 replicas, nominally 59 resident replicas per L40S, four GPUs, 192 allocated logical CPUs, MPS, shared contact CV, and active frozen-envelope stage-5 Pep-GaMD with the exact auxiliary-PME energy decomposition.

This document specifies all seven proposed upgrades. It extends [helper-cpus](https://github.com/sulcjo/atlas-md/blob/36ebbd9e1fbb20e5fde0e51c30aa3d9c46531a33/docs/superpowers/specs/helper-cpus/spec.md), whose durability, output identity, exact time ledger, recovery, and CPU-budget contracts remain authoritative. That document is a specification, not evidence that its implementation already exists.

The seven changes preserve the scientific model: no timestep, precision, force cutoff, contact definition, random-number schedule, boost envelope, exchange eligibility, volume-move cadence, trajectory cadence, or production-budget change. They introduce neither multiple-time-step integration nor a surrogate for auxiliary PME. Throughput gains below are hypotheses to measure, not forecasts.

| ID | Upgrade | Main opportunity | Initial delivery |
| --- | --- | --- | --- |
| P1 | Fuse weighted torsion sub-CVs | Reduce repeated torsion evaluation and sub-CV overhead | Opt-in force construction |
| P2 | Reuse identical energy observations | Remove duplicate physical-energy queries | Exact-layout observation adapter |
| P3 | Bound concurrently advancing replicas per GPU | Reduce oversubscription while retaining all thermodynamic states | Fixed configurable admission limit |
| P4 | Tune MPS active-thread percentage | Improve GPU resource sharing for this process/context topology | Offline, startup-only tuning |
| P5 | Place owner threads and memory near their GPUs | Reduce remote CPU/memory traffic and migration | Topology-aware startup placement |
| P6 | Remove passive reporting barriers | Let a replica proceed after its own scheduled observation | Fixed-boundary event scheduler |
| P7 | Native combined CUDA CV force | Avoid the remaining nested CV overhead | Experimental plugin, gated by profiling |

Priority: implement P1 and the local portion of P2 first; instrument and tune P3–P5 next; deliver P6 after helper-cpus has passed its durability gates; prototype P7 only if the combined CV remains a material cost. Measure marginal improvements against the already accepted combination, not only against the original baseline.

## 2. Verified integration points

The following are observations of the pinned source, not assumptions about deployed binaries.

| Source | Current behavior | Required seam |
| --- | --- | --- |
| `gareus/production.py:_add_residual_torsion_cv_force` | Up to four weighted sine/cosine torsion child forces, plus the shared contact sum | P1 builder and layout metadata; P7 provider |
| `gareus/production.py:_add_weighted_trig_torsion_force` | Uses `sin(-theta)` and `cos(-theta)` | Preserve the stored feature convention exactly |
| `gareus/cv_selection/residual_runtime.py` | Compiles one residual definition; evaluates sub-CVs by semantic role | Reuse coefficients and role metadata; do not fork the residual definition |
| `gareus/production.py:sample`, `_fetch_v_pep_v_dih`, `_current_exchange_arrays` | Reporting can request physical energy, then request it again for the peptide energy | P2 typed immutable observations |
| `gareus/production.py:_ReplicaAffinityExecutor` | A stable single-worker executor per replica | Preserve context ownership; add admission before submission |
| `gareus/production.py:step_all` and production event loop | All-replica joins at logging and other outer boundaries | P3 admission; P6 classification of passive versus scientific boundaries |
| `gareus/npt_driver.py:advance` | Advances to local report/NPT events and emits reports after the volume decision | Preserve local event order and report state |
| `gareus/npt.py`, `gareus/pep_gamd.py` | NPT proposals require fresh energies; exact boost partition has distinct force groups | Observation invalidation and P7 force-group validation |
| `gareus/system_setup.py` | CPU-platform thread budgeting and CUDA property handling | Keep CPU-platform budgets distinct from GPU owner/helper allocation |

Source: [production.py](https://github.com/sulcjo/atlas-md/blob/36ebbd9e1fbb20e5fde0e51c30aa3d9c46531a33/gareus/production.py), [residual_runtime.py](https://github.com/sulcjo/atlas-md/blob/36ebbd9e1fbb20e5fde0e51c30aa3d9c46531a33/gareus/cv_selection/residual_runtime.py), [npt_driver.py](https://github.com/sulcjo/atlas-md/blob/36ebbd9e1fbb20e5fde0e51c30aa3d9c46531a33/gareus/npt_driver.py), [npt.py](https://github.com/sulcjo/atlas-md/blob/36ebbd9e1fbb20e5fde0e51c30aa3d9c46531a33/gareus/npt.py), [pep_gamd.py](https://github.com/sulcjo/atlas-md/blob/36ebbd9e1fbb20e5fde0e51c30aa3d9c46531a33/gareus/pep_gamd.py), [system_setup.py](https://github.com/sulcjo/atlas-md/blob/36ebbd9e1fbb20e5fde0e51c30aa3d9c46531a33/gareus/system_setup.py).

## 3. Shared contracts

### 3.1 Scientific identity and restart compatibility

Record two separate identities: the semantic model identity, covering the Hamiltonian/CV/boost definition, and the implementation identity, covering force construction, native backend, layout version, precision and relevant binary versions. Equivalent formulas do not imply interchangeable binary checkpoints.

Retain the legacy force builders. An old checkpoint resumes using its recorded construction. Missing legacy metadata is interpreted through an explicit versioned compatibility rule, never the current default. Initially refuse resume-time switching among legacy, fused, and native force layouts. A future migration must be an explicit, separately validated restart mode; loading positions and velocities alone does not preserve stochastic integrator state.

Scheduling/placement settings may change on a clean restart only where the deployed checkpoint contract permits it, with the change recorded. No mid-run context migration, force replacement, or live MPS repartitioning is part of this scope.

### 3.2 State identity and observations

Define an immutable observation identity containing at least campaign/branch, epoch, replica UID, integer integration step, coordinate/box revision, context-parameter revision, assignment revision, envelope/kernel identity and observation schema. Store the exact requested group set, quantities, units, precision and position-wrapping convention with its payload.

Any integration, coordinate/box update, parameter update, exchange assignment, accepted or rejected volume proposal, rollback, checkpoint load, rescue, or envelope transition invalidates the appropriate cached view. Initial implementation may conservatively invalidate every view on every mutation. Revision increments happen on the owning thread through audited mutation entry points. An unaudited mutation path must disable cross-call reuse.

The same step number is insufficient for reuse. A rejected NPT proposal can restore coordinates at the same step; an exchange can change bias parameters without advancing time. An observation captured before such an event must not become a post-event observation by relabeling it.

### 3.3 Ownership, accounting and liveness

Every Context and integrator remains owned by the same replica thread throughout its life. Helpers receive frozen arrays, scalars or checkpoint bytes and never call Context methods. Central coordination owns cross-replica exchange decisions and global RNG consumption in canonical order.

Use helper-cpus identities and integer-step time accounting. Concurrently progressing replicas have a step vector; there is no truthful single live step until a common boundary. Aggregate time is the sum of eligible per-replica spans, not replica count times the fastest replica's step. Only the atomically committed generation contributes to durable production time.

GPU admission tokens, helper CPU tokens and queue capacity are separate resources. Specify acquisition order and test bounded-queue liveness. A worker must not hold a GPU slot while waiting for output that requires another GPU-owner task. Publication waits release helper compute tokens. Shutdown and zero-helper-budget drains follow helper-cpus' paused-propagation exception.

## 4. P1 — fused weighted torsion sub-CV

### 4.1 Construction

Replace the four child forces with one `CustomTorsionForce` whose expression is:

```text
a*sin(-theta) + b*cos(-theta)
```

Both `a` and `b` are per-torsion parameters. Append phi torsions first and psi torsions second, matching the compiled feature order. For phi index `i`, take `a=v[2*i]`, `b=v[2*i+1]`. For psi index `j`, take `a=v[2*n_phi+2*j]`, `b=v[2*n_phi+2*j+1]`. Validate exact feature width before construction; never allow truncating `zip` to hide a mismatch.

Name the new scalar `sum_weighted_torsions` and give it the existing `torsion_sum` semantic role. Pass that single name to `CompiledResidualComponent.openmm_expression`. Keep the same contact child, normalizations, polynomial coefficients, clipping transform, umbrella parameters and outer force group. Do not create a second contact sum in the shared layout.

**The minus sign is mandatory.** Stored features use the negated OpenMM torsion convention. Using `a*sin(theta)` with unchanged weights would change the CV and the force. Preserve the current per-force PBC settings; correcting a possible geometry convention is a separate scientific change.

Retain duplicate torsion entries as additive terms, including a quadruplet appearing in both feature blocks. Empty phi or psi blocks and zero weights are valid where the existing runtime accepts them. For an entirely empty accepted torsion feature set, preserve the reference zero-sum behavior without inventing a dummy physical interaction.

OpenMM supports custom torsion expressions with per-torsion parameters and explicit PBC settings. [CustomTorsionForce API](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomTorsionForce.html).

### 4.2 Integration and acceptance

Introduce a force-construction version independent of `shared` versus `split` contact layout. Audit all consumers that enumerate sub-CVs, serialize layout metadata, use `getCollectiveVariableValues`, rescore seeds, run pulling/preparation, or reconstruct CVs offline. Use roles, not a hard-coded expectation of four names. Fresh force builders across these paths must resolve the same campaign policy.

Compare legacy and fused CV values, isolated umbrella energies, per-atom forces, and parameter derivatives wherever requested. Fixtures cover unequal phi/psi counts, all sign combinations, zero weights, duplicate torsions, both residual degrees, clamp interior/exterior/boundaries, zero umbrella stiffness and nonzero contact coupling. Independently compare to positions-based reprojection and finite differences, including angles near ±π.

Floating-point reduction order changes: promise numerical equivalence within declared precision tolerances, not bitwise trajectory identity. Do not assess correctness by requiring chaotic trajectories to remain identical. Run production/NPT/exchange and checkpoint round-trip tests for each construction separately.

Profile force evaluation and full production. Count actual kernels/sub-CV operations where tooling allows; a smaller Python force object count alone is not a performance result. Promote only after the final benchmark gates.

## 5. P2 — eliminate redundant energy queries

### 5.1 First delivery: reuse inside one observation

Add a small observation adapter, for example `gareus/observations.py`, with explicit requested quantities and force-group provenance. In the exact auxiliary-PME path:

```text
Ephys = E({0, 2})
Vdih  = E({2})
Vpep  = Ephys - E({1})
```

When reporting has already requested that exact physical group set from the same unmodified Context state, pass its value to the peptide-energy calculation. The current reporting pattern can then use three energy queries instead of four: physical, dihedral and auxiliary. This count excludes CV observations and any separate reporter/NPT queries; verify it in an instrumented fixture.

Do not assume a union-group request exposes its individual components. Do not subtract auxiliary energy from a reported quantity that includes extra physical or bias groups. Require equality of the group set and the applicable energy-definition version; otherwise request the needed groups separately. The upcoming real-space auxiliary implementation, if introduced, needs its own explicit energy contract.

Preserve existing branches: no active envelope makes no peptide/dihedral request; dihedral-only boost requests its required component without auxiliary work; disabled potential-energy reporting remains disabled unless an internal boost calculation independently needs that energy. Audit conventional MD, stock GaMD, exact Pep-GaMD and non-fast CV paths explicitly. Unsupported combinations keep the reference reader rather than guessing.

Do not substitute integrator globals for current-state energy without proof of the exact position and phase those globals represent. A value calculated before the last integration update is not the energy at the final reported coordinates.

### 5.2 Second delivery: shared immutable observation views

After the revision system in §3.2 exists, permit a sample and exchange observation at the same state to share exact scalar values. Separate acquiring raw state from computing vectorized bias matrices. Preserve the current vectorized cross-window calculation; do not reintroduce per-window Context requests.

Initially leave NPT proposal energy acquisition uncached. Only add NPT reuse in a separate change with proposal/rollback revision tests and unchanged trial-energy semantics. Never reuse a pre-trial value as a proposal value, even at the same integration step.

Bound cache size to the observations still needed by current events/output, not a trajectory history. Payloads already handed to helpers remain immutable. P6's later arrival of another replica's sample cannot overwrite the earlier step's observation.

### 5.3 Acceptance

Instrument state calls by purpose, flags and group set. On frozen states, compare every returned quantity against the unoptimized reader. Assert expected query reductions in the concrete supported layout and no extra reads when channels are disabled. Inject same-step parameter updates, NPT acceptance/rejection, rollback, checkpoint load and envelope changes; stale reuse must be impossible. Include a physical-group-set mismatch fixture that deliberately refuses the optimization.

Benchmark realistic reporting cadences with and without potential-energy output. Report saved calls and measured critical-path time separately: an avoided API call need not save a complete force calculation on every OpenMM version.

## 6. P3 — bounded active replicas per GPU

### 6.1 Meaning of the limit

`active_replicas_per_gpu=A` limits simultaneous admitted replica-owner operations that advance or observe GPU state. It does not reduce the number of resident replicas, remove thermodynamic windows, or change exchange participants. All 59 replicas remain resident per GPU in the target case; GPU memory usage and CUDA context limits therefore do not automatically decrease.

Initial candidates are `A=8,16,32,59`. Interpret `all` as the assigned resident count. An explicit positive limit greater than the resident count is capped and reported. Reject zero or negative values. Support uneven GPU populations and record each GPU's effective limit.

### 6.2 Dispatcher

Keep `_ReplicaAffinityExecutor` ownership. Add central per-GPU ready queues and admit an operation before submitting it to its replica's single-worker executor. Do not submit 236 blocking semaphore waiters and depend on worker scheduling to escape them. Allow at most one admitted operation per replica.

Use stable round-robin admission with persistent replica UIDs, not elapsed-time ranking. Once admitted, call the existing driver to its requested boundary. The operation includes its scheduled local NPT work and Context reads. Release its token on normal completion or failure; account for any backend work that can outlive a host return so the limit has a documented, measured meaning. Do not add a device-wide synchronization after every chunk merely to count occupancy: first establish the supported backend's completion behavior and choose a per-operation completion mechanism if necessary.

In the first implementation, retain all existing outer event boundaries and safe-chunk barriers. P3 changes admission only. Each replica still completes the same number of integration steps before the same exchange. The final partial batch is mandatory. Replica count, RNG ownership, event count and exchange order remain unchanged.

Initialization, warmup and checkpoint capture use separately documented admission policies; they must not accidentally bypass GPU memory or host-memory bounds. Checkpoint capture may share the dispatcher but cannot begin until its common-boundary precondition holds.

### 6.3 Failure and acceptance

Cancel not-yet-started work after the first fatal error, wait for already-running owner operations to reach a safe termination point, and retain a per-replica progress vector for diagnostics. Do not publish a mixed-step checkpoint as a valid generation. Recover from the last committed generation under helper-cpus.

Test 1, uneven counts, `A=1`, `A=resident_count`, and non-divisors such as 59 replicas with `A=16`. Instrument simultaneous admitted work and owner thread IDs. Force one replica to be slow, one to fail, and one queue to stall; verify no starvation, token leak or deadlock. Replay deterministic event/RNG fixtures and compare canonical event logs against unrestricted admission.

Benchmark aggregate completion time to fixed common boundaries. A faster active subset with slower completion of all 236 replicas is a regression. Include per-replica waiting time, spread of progress, CPU demand and GPU utilization, but select by durable node throughput.

## 7. P4 — MPS active-thread tuning

### 7.1 Startup-only policy

Add a benchmark launcher setting for `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`, initially uniform across the process, set before launching the Python/CUDA client. Sweep `100`, `50`, and `25`, plus the actual inherited deployment setting if different. Record the inherited setting separately; do not silently redefine baseline as 100.

ATLaS-MD's many Contexts in one process are not equivalent to one independent process per simulation. NVIDIA documents process-level caps applying to the process's client contexts, with separate opt-in rules for per-context partitioning. Existing contexts do not acquire a new cap from a later environment change. Therefore this delivery forbids modifying process environment variables from parallel context-creation threads. [NVIDIA MPS environment-variable reference](https://docs.nvidia.com/deploy/mps/latest/appendix-environment-variables.html).

Do not derive a production cap from `100/59` or `200/59`. The useful cap is empirical for this workload and topology. Caps are not exclusive reservations of a disjoint SM partition per replica.

### 7.2 Verification and isolation

Record GPU UUIDs, driver/CUDA/OpenMM versions, MPS version, daemon/server/namespace policy where applicable, client process count and actual CUDA contexts per GPU, including nested contexts. The count of replicas must not be mislabeled as the count of CUDA contexts. Detect whether server policy constrains or overrides the requested client setting. Where supported, query the effective multiprocessor allocation from the relevant current context through a minimal diagnostic; otherwise label the effective allocation unverified.

Launch each arm as a fresh process with an explicit environment manifest. Keep `CUDA_DEVICE_MAX_CONNECTIONS`, precision, blocking-sync behavior, force layout and helper policy fixed. Leave shared MPS servers and other jobs untouched. A requested setting that cannot be verified or is overridden is not a valid independent tuning arm; report or skip it.

Test P4 first at the baseline admission setting, then cross the leading cap candidates with P3's leading admission candidates. Neither optimum should be assumed independent of the other. Finish the selected combination in a fresh held-out benchmark. Do not implement online cap adaptation.

### 7.3 Acceptance

The launcher rejects malformed percentages before starting CUDA, emits requested/effective settings, and produces separate manifests for each arm. Verify that a new process is used for each setting and that no daemon-global mutation occurs. Resume a compatible checkpoint in each supported arm and validate normal output/accounting. A cap that accelerates a small fixture but slows the full 236-replica ensemble remains opt-in or is rejected.

## 8. P5 — GPU-local CPU and memory placement

### 8.1 One allocator for the job

Extend helper-cpus' job-level allocation discovery; do not create a second allocator that independently claims all 192 threads. Preserve its original authorized launch mask, effective cpuset, CPU quota, safety headroom and helper-token budget. OS thread ownership in `_ReplicaAffinityExecutor` is not CPU affinity.

Discover GPU UUID/PCI identity and its available NUMA relationship, CPU socket/core/SMT topology, and allowed memory nodes. Intersect candidates with permitted CPUs and memory nodes. Never assume GPU index equals socket or NUMA index. Multiple GPUs may share a node; all placement decisions must consider the total owner population mapped there.

Initial policy `gpu-local` binds owner threads to a permitted node-local CPU mask before Context creation and major native allocations. Prefer an eligible core set over pinning 59 owner threads to 59 invented exclusive cores. More owner threads than CPUs may share the same mask. Allocate across GPUs sharing a node to avoid assigning every GPU a misleading exclusive copy of the same cores.

Offer `none` as the baseline/default until validated. Unknown topology, NUMA node `-1`, or an empty local intersection yields a visible fallback to the authorized mask, or an explicit strict-mode startup error. Never broaden a scheduler restriction to obtain locality.

### 8.2 First touch and helper interaction

Create and touch owner-specific host buffers on their owner threads after placement. Existing parent-thread allocations do not become local merely because a later worker is pinned. Use per-thread/per-allocation memory policy only if supported and measured; avoid process-wide binding to one GPU's node in a process serving all four GPUs. Do not add privileged memory migration.

Helpers use the same allocator's shared or explicitly reserved policy. Shared mode does not claim exclusive cores. Reserved mode, if implemented under helper-cpus, partitions masks before Context/native-worker creation. One budget includes codecs, renderers and hidden native worker pools; topology policy must not multiply the helper count by GPU count. Keep the coordinator able to drain output and dispatch work.

Linux distinguishes task and address-range memory policies, and allowed node restrictions still apply. Use that distinction when implementing allocation placement; affinity alone is not evidence of memory locality. [Linux NUMA memory-policy documentation](https://www.kernel.org/doc/html/latest/admin-guide/mm/numa_memory_policy.html).

### 8.3 Acceptance

Test restricted masks, SMT-only fragments, asymmetric NUMA topology, two GPUs on one node, unknown topology, cpuset changes, and a quota smaller than the mask size. Record requested and actual thread affinity after native initialization; audit spawned library threads instead of assuming inheritance solves every case. Verify no CPU outside the allocation is used by an application-created placement decision.

Benchmark `none`, local owner placement, and local placement plus measured first-touch changes as separate arms. Record remote/local memory evidence where available, CPU migrations, context switches, quota throttling, owner/helper utilization and durable throughput. Re-evaluate helper headroom after a placement change. A lower owner CPU measurement caused by starvation must not trigger more helpers.

## 9. P6 — remove passive reporting barriers

### 9.1 Eligibility and boundaries

This is an event-scheduler change, not merely moving the existing `sample()` call to a helper. It depends on immutable observations, helper-cpus' bounded queues and durable output protocol, and a complete inventory of report consumers.

Classify every event as local passive observation or global scientific/control boundary. Exchange, checkpoint capture, epoch transitions, envelope/recalibration updates, rescue decisions, roster changes and any report-driven stopping/adaptation decision remain barriers. Preserve configured safety-chunk boundaries unless their semantics are separately proven local. If a report feeds a decision, its step remains a boundary even if its file writing is asynchronous.

Initially enable the mode only for a frozen production configuration whose passive events are certified by the inventory. Unsupported adaptive or report-driven rescue paths use the legacy scheduler, with a recorded reason. Do not infer passivity solely from the name of a logging function.

### 9.2 Execution model

From common boundary `b`, compute the next global boundary `B`. Each replica advances through its own exact local event steps, stopping for its required CV/energy/frame capture, then becomes ready to continue toward `B` without waiting for another replica's passive report. No replica may cross `B` until the coordinator completes the boundary action.

If P3 is enabled, use bounded owner tasks that end at the next local event or a configured scheduling quantum; completion requeues the replica fairly. Capacity reservation occurs before expensive snapshot extraction. If output capacity is unavailable, yield admission so another eligible task or the output drain can run. Never wait with a GPU token held for queue capacity that requires a queued owner task to free it.

Capture scalars, optional positions and assignment/envelope metadata on the owner thread at the exact scheduled step. Store an immutable packet keyed by step and replica UID. Cross-replica tables may be assembled later in canonical order, preserving vectorized calculations and the appropriate frozen window/envelope version. A packet's assignment comes from capture time, never from the coordinator's current mapping at write time.

At coincident deadlines, preserve the existing scientific order: complete the step and due NPT decision; capture pre-exchange observations under the old assignment; perform exchange/control actions in the reference order; capture the checkpoint under the resulting state. Preserve the repository's terminal-exchange policy. The durable catalog can therefore include pre-exchange frames while the checkpoint contains post-exchange assignments, linked by their revisions.

Example with the target cadences: from common step 400, each replica may capture its step-500 observation independently and proceed through its due local step-600 NPT event toward the step-800 exchange boundary. Its step-750 observation also occurs before 800. It cannot run past 800 while another replica is still below 800. A slow replica changes wall time, never which replica participates in the exchange.

### 9.3 Failure, stop and time semantics

Expose live per-replica steps, minimum common progress, and committed generation progress separately. Use monotonic wall time for durations; compute simulation time from integer steps and the epoch's exact timestep representation. Do not report the maximum live step as completed ensemble time.

On clean stop, reach the next permissible common stop boundary under the existing stop policy, drain/capture/commit, then exit. If the deadline or failure prevents that, retain the last committed generation and mark later output as uncommitted. Do not manufacture a synchronized checkpoint from replicas at different steps. Walltime-driven termination must never select a faster subset of replicas for scientific analysis.

Bound packet storage by bytes and enforce helper-cpus' oversized synchronous fallback. Out-of-order arrival must not imply dropped frames, skipped samples, or unbounded buffering while waiting for one slow replica. Global assembly tracks expected identities, not merely a packet count. Writer failure is fatal for required scientific output.

### 9.4 Acceptance

Compare exact event identities and ordering against the legacy scheduler using deterministic simulated drivers: unequal replica speeds, coincident NPT/log/exchange/checkpoint events, non-dividing cadences, final partial intervals, queue saturation and failure between arrivals. Check assignment revisions, time totals, report counts, exchange RNG consumption and final replica steps exactly.

Use the normal loaders and resume path to verify that every committed generation has its exact output prefix. Crash at each capture/publication boundary; delayed packets from an abandoned branch must never enter the recovered canonical data. Verify the passive-mode eligibility checker refuses report-driven rescue/adaptation cases.

Benchmark against a helper-cpus-enabled scheduler that still has global logging barriers. This isolates P6's scheduling benefit from asynchronous output's benefit. An additional run without helper-cpus is useful for context but cannot establish P6's marginal gain.

## 10. P7 — native combined CUDA CV force

### 10.1 Scope and go/no-go

Implement an experimental native force for the existing shared contact/residual-torsion umbrella. It replaces that CV force, not physical PME, auxiliary PME, the integrator or boost application. First profile P1 plus accepted observation changes. If CV evaluation is no longer a material fraction of production time, stop this project at the prototype gate.

Provide a public force API, serialization, a Reference implementation and CUDA kernels using the deployed OpenMM plugin interfaces. Pin supported OpenMM/CUDA versions and test loading against each shipped combination. A plugin has both public API and platform-kernel integration obligations. [OpenMM plugin guide](https://docs.openmm.org/8.0.0/developerguide/03_writing_plugins.html).

### 10.2 Mathematical contract

Let each contact have its existing weight `w_p`, distance `r_p`, switching distance `r_s` and slope `beta`. Let `N` be the existing contact normalization. Define:

```text
c = sum_p w_p * 0.5 * (1 - tanh(0.5*beta*(r_p-r_s)))
S = sum_i [a_i*sin(-theta_i) + b_i*cos(-theta_i)]
u = (c/N - mu_c) / sigma_c
t = T(u)                       # identity or the compiled hard clip
z = (S - K0 - K1*t - K2*t*t) / sigma_z
U = 0.5*k_c*(c/N-c0)^2 + 0.5*k_z*(z-z0)^2
```

All symbols come from the existing frozen runtime or current window parameters. `K1` and `K2` are polynomial coefficients, distinct from umbrella stiffnesses `k_c` and `k_z`. Parameter validation requires positive `N`, `sigma_c`, `sigma_z`, valid clip bounds and finite coefficients. Initial native scope requires shared-layout normalization identities to match the validated reference contract; do not silently conflate distinct recorded normalizations.

The chain rule is:

```text
dU/dS = k_z*(z-z0)/sigma_z
dU/dc = k_c*(c/N-c0)/N
         - [k_z*(z-z0)/sigma_z] * (K1+2*K2*t) * T'(u)/(N*sigma_c)
F = -(dU/dc)*grad(c) - (dU/dS)*grad(S)
```

For each contact switch, `df/dr = -beta/4 * sech(0.5*beta*(r-r_s))^2`. For the OpenMM torsion variable, the derivative of `a*sin(-theta)+b*cos(-theta)` is `-a*cos(theta)-b*sin(theta)`. Dropping the contact contribution from residualization or reversing this sine convention changes the sampled distribution.

Inside a hard-clip interval `T'=1`; outside it `T'=0`. At the endpoints, match the deployed reference expression's derivative convention; characterize this explicitly rather than claiming differentiability. Test one-sided limits and exact representable boundary cases. Do not smooth clipping as a performance optimization.

### 10.3 GPU implementation requirements

Compute contact/torsion contributions, reduce `c` and `S`, then apply the chain-rule force coefficients entirely on device during force evaluation. A multi-kernel implementation is acceptable; there is no promise of a single launch. Host scalar observation may synchronize only when requested and must carry the observation identity.

Match all current periodic/nonperiodic force flags and triclinic geometry behavior. Do not introduce a cutoff/neighbor-list omission into the smooth contact sum. Preserve duplicate pairs, weights and torsions. Support atom reordering with correct index remapping, virtual-site force redistribution through OpenMM's normal path, and the relevant molecule-information contract.

Honor force-group selection and independent energy/force request flags. The native force remains in the same bias group and outside the auxiliary-PME and physical-group definitions. Energy-only NPT trial calls and force-only integration calls must both work. Return the same CV values even when umbrella stiffness is zero. Support every global-parameter update used by window swaps and requested parameter derivatives; reject unsupported capabilities explicitly at startup.

Use the platform's force-accumulation conventions and appropriate reduction precision. Expose deterministic-force compatibility honestly; an unsupported requested mode is a startup error. No per-step host allocation, Context creation, or host reduction is allowed in the production force path. Size and validate scratch buffers at initialization and bounded update points.

### 10.4 Integration and compatibility

Introduce a backend-neutral CV observation interface returning semantic `contact_sum` and `torsion_sum` quantities. Adapt fast reporting, pulling, seed validation and offline comparison through that interface; do not require the native force to impersonate `CustomCVForce`. Continue using `CompiledResidualComponent` as the coefficient/schema authority.

Serialize all definitions and record plugin build/backend/layout versions. Checkpoint restore requires a matching supported implementation. Initial backend selection is explicit at campaign creation; missing binaries or incompatible layouts fail before production. No silent mid-campaign fallback to a different force construction.

### 10.5 Acceptance

Use three independently informative references: legacy CustomCVForce, P1's fused force, and finite differences of an independently evaluated scalar energy. Compare CV values, isolated bias energies, per-atom forces and parameter derivatives on a frozen conformation corpus covering all supported branches. Include contact transition regions, clip boundaries, torsion wrap, nonuniform weights, changed windows, atom reordering, periodic boxes and volume proposals.

Set precision-specific error tolerances before evaluating candidate results. Report absolute errors and normalized RMS force error against the isolated CV-force scale, with a floor for near-zero forces. Calibrate the mixed-precision tolerance from repeat/reference-platform behavior and finite-difference convergence; do not widen it to admit a failing candidate. Use a displacement-step sweep for finite differences, not one arbitrary step.

Then validate boost partitions, NPT acceptance calculations on matched proposals, exchange bias matrices, normal checkpoint restore and output reconstruction. Exact long-run trajectory equality is not an acceptance requirement; equilibrium observables, reweighting consistency and event semantics are. Compare distributions with independent replicas/runs and autocorrelation-aware uncertainty, not a short trajectory's matching mean.

## 11. Configuration, delivery and adversarial review gates

### 11.1 Proposed controls

Names below are proposed interfaces, not claims that flags already exist. Resolve through the repository's normal configuration/CLI system and persist the effective values in the run manifest.

| Control | Initial default | Meaning |
| --- | --- | --- |
| `torsion_subcv_layout: legacy\|fused` | `legacy` | P1 force construction; frozen per campaign |
| `energy_observation_reuse: off\|local\|revisioned` | `off` during validation | P2 scope; local first |
| `active_replicas_per_gpu: all\|N` | `all` | P3 admission; per-GPU effective values recorded |
| `mps_active_thread_percentage: inherit\|1..100` | `inherit` | P4 launcher-only setting |
| `gpu_cpu_affinity: none\|gpu-local` | `none` | P5 startup placement |
| `gpu_cpu_affinity_strict: bool` | `false` | Fail instead of fallback when requested locality is unavailable |
| `production_report_scheduler: barrier\|local` | `barrier` | P6; local requires eligibility checks |
| `cv_force_backend: openmm\|native-cuda` | `openmm` | P7; explicit experimental backend |

For `native-cuda`, the torsion-sub-CV layout option is not operative; reject an explicitly conflicting layout request and record the native construction. Never accept two settings while silently honoring only one. Unsupported backend/platform combinations fail before running production.

### 11.2 Delivery order

1. Add manifests, narrowly scoped timers/query counters, observation types and force-layout identity. Keep reference behavior.
2. Deliver P1 and P2 local reuse independently, then together. Validate all scientific construction paths and legacy resume.
3. Deliver P3 admission and P4 launcher experiments; select a measured combination.
4. Deliver P5 using the helper-cpus allocator contract. Benchmark affinity independently before combining it with selected scheduling settings.
5. Complete helper-cpus durability implementation and gates if still pending. Add revisioned observations and P6 only afterward.
6. Profile the accepted stack and decide whether to fund P7. Deliver native backend behind an explicit experimental flag; compare with the accepted fused OpenMM backend.

Use separate reviewable changes for each upgrade. Keep per-feature rollback switches, but never use a force-layout switch to bypass checkpoint compatibility. No performance configuration becomes the default merely because it won a tuning sweep.

### 11.3 Adversarial cases required before promotion

| Failure hypothesis | Required proof |
| --- | --- |
| Fusion accidentally flips the sine sign | Negated-feature oracle and force comparison detect it |
| A changed sub-CV count breaks resume or fast observation | Versioned builders and role-based consumer tests |
| A same-step NPT rollback or swap permits stale energy | Revision invalidation tests with deliberately different values |
| Extra physical force groups corrupt peptide subtraction | Exact-set guard refuses incompatible reuse |
| 59 replicas with a limit of 16 lose the last batch | Exact final-step and event-count reconciliation |
| Admission or helper queues deadlock under backpressure | Forced slow writers, exhausted tokens and error/stop drains terminate |
| MPS setting is ignored or races Context creation | Fresh-process arms and effective-setting evidence |
| CPU placement escapes the allocation or starves helpers | Restricted-mask/quota tests and actual affinity manifest |
| Fast replicas contribute extra frames/time or get different exchange participation | Step-vector accounting and fixed-boundary event oracle |
| Late packets acquire the post-exchange state label | Immutable capture-time assignment revisions |
| A crash leaves a checkpoint and output from different boundaries | helper-cpus transactional recovery through normal loaders |
| Logging triggers rescue, making barrier removal change science | Dependency inventory and explicit ineligibility/fallback |
| Native force omits the residual contact chain rule | Independent finite differences with nonzero K1/K2 |
| Native forces match but NPT energy-only queries fail | Group/flag matrix and matched volume-proposal tests |
| A large speedup comes from less scientific work | Equal steps, required records, exchange/NPT events and committed coverage |

## 12. Final benchmark and promotion protocol

### 12.1 Fixed workload and manifests

Use the user's actual production system and frozen initial states on the target four-L40S allocation. Start from 236 replicas/59 per GPU with the current shared CV and exact auxiliary PME. Freeze topology, force field, masses, timestep, temperature, pressure, boost envelopes, window assignments, random seeds, precision and output settings for each paired comparison.

The checked-in [chignolin_9 configuration](https://github.com/sulcjo/atlas-md/blob/36ebbd9e1fbb20e5fde0e51c30aa3d9c46531a33/smoke/config/chignolin_9.yaml) supplies an initial cadence fixture: 3.5 fs timestep, scalar/trajectory intervals of 250 steps, exchange every 400, NPT every 200, checkpoint every 20,000. Thus 250 steps is 0.875 ps. Verify the resolved runtime configuration rather than copying comments or assuming this fixture is the deployed workload. Keep potential-energy reporting enabled for the representative P2 arm.

Record git/submodule commits, package/plugin builds, GPU UUIDs, driver/CUDA/OpenMM/MPS versions, requested/effective MPS settings, all Context counts, force layout/backend, RNG manifests, allocation/quota/topology evidence, helper policy, storage filesystem/durability target and resolved configuration. Preserve unrelated performance controls at their baseline values.

Use a smaller deterministic/reference fixture for exact event tests and a production conformation corpus for numerical tests. Neither substitutes for the full-node throughput workload. Include one held-out workload/configuration for portability, such as a different valid window population or contact/torsion size; do not extrapolate its result to all proteins.

For final durable comparisons, establish an A1 reference implementing helper-cpus' transactional protocol synchronously, with P1–P7 off. Compare asynchronous or scheduling candidates under the same output coverage and durability contract. Keep the original pinned implementation as A0 for operational context, but if it cannot certify the new durable-output contract, label its rate completed-work throughput and do not equate it with the primary durable metric. Early P1–P5 measurements before A1 exists are provisional; their final promotion requires a like-for-like durable comparison.

### 12.2 Arms and interactions

| Stage | Arms | Question |
| --- | --- | --- |
| Baseline | A0 pinned run; instrumented A0; A1 synchronous durable reference with P1–P7 off | What do instrumentation and the common durability contract cost? |
| P1 | Legacy versus fused | Does sub-CV fusion save full-production time? |
| P2 | Off versus local; later revisioned | Does query reuse save time at actual output cadence? |
| P3 | `all`, 32, 16, 8 active replicas/GPU | Which limit minimizes all-replica completion time? |
| P4 | Inherited, 100, 50, 25, deduplicated | Which verified cap helps this process topology? |
| P3×P4 | Cross the leading two admission limits with leading two caps | Is the combined choice better? |
| P5 | None; owner affinity; affinity plus first-touch changes | Does locality add benefit under the selected GPU schedule? |
| P6 | Helper-cpus with barrier scheduler versus identical local scheduler | What is the marginal benefit of removing passive barriers? |
| P7 | Accepted fused OpenMM stack versus native backend | Is plugin complexity justified after simpler improvements? |
| Final | A1 durable reference; accepted stack; selected feature ablations; A0 separately labeled | What gain survives independent confirmation, and what caused it? |

Deduplicate equivalent arms. Use short kernel/integrator-only measurements to diagnose, not promote. Keep unrelated options and helper behavior fixed when isolating a feature. For the final realistic stack use the selected production helper policy and record its actual assignments; tuning may change CPU demand, so verify headroom again.

### 12.3 Timing and statistics

Predeclare the benchmark budget and decision rule. Use three independent paired screening runs per candidate that reaches full-production screening, each spanning at least five checkpoint cycles. Shorter preliminary runs may reject obviously poor settings but cannot establish durable-throughput superiority. Randomize or counterbalance A/B order and avoid simultaneous competing arms on the same GPUs.

After selecting candidates, run at least six new independent confirmatory pairs for the final choice. Do not reuse the tuning runs as independent confirmation. Pair common initial states/seeds where compatible; for a changed force layout construct matching validated starting states instead of loading an incompatible binary checkpoint. Use separate RNG seeds across independent pairs. Cycles or replicas within one coupled ensemble are not independent benchmark replicates.

For pair `i`, define `d_i = log(Tcandidate_i / Treference_i)`. Report each ratio, the median ratio, and an explicitly named paired 95% interval. A conservative distribution-free option is an order-statistic interval for the median of independent `d_i`, using binomial coverage and transforming back; with only six pairs, `[min(d_i), max(d_i)]` has 96.875% coverage for a continuous median and can be very wide. If the interval is inconclusive, label the result inconclusive or opt-in. Do not describe an unspecified bootstrap over checkpoint cycles as independent 95% evidence.

At historical approximately 37 steps/s/replica, five 20,000-step cycles would be about 45 minutes per arm; this is a planning estimate, not a measurement from this work. Report startup and warmup separately. Include all required output draining and final durable acknowledgment in the end-to-end measurement. Stop optional testing when the predeclared gates resolve the decision; report remaining uncertainty if the budget is insufficient.

### 12.4 Primary and diagnostic metrics

```text
durable_node_ns_per_day =
    newly_committed_eligible_aggregate_production_ns
    / measured_end_to_end_wall_seconds * 86400
```

The numerator excludes equilibration, replayed/abandoned work, duplicate epoch spans and uncommitted tails. Count all replicas' actual eligible spans exactly once. The denominator runs from the first measured propagation through the final durable output acknowledgment. If scratch mirroring is used, the declared durable destination determines completion.

Also record common-boundary latency; per-replica progress/wait distribution; reporting and checkpoint capture pauses; output drain and commit latency; GPU-owner admission wait; queue/backpressure bytes and time; cache hit/miss reasons; energy-query counts by groups/purpose; CV/kernel timings; CPU demand, migrations and throttling; helper assignments; peak host/device memory; GPU utilization and power; and errors/fallbacks. Timers must distinguish overlap from critical-path elapsed time. Do not sum overlapping worker durations into an alleged wall-time saving.

For tail metrics, publish raw sample counts and uncertainty. Five checkpoints in one run cannot support a reliable p99 claim. Only make a tail-latency promotion claim with a separately sufficient, predeclared sample budget.

### 12.5 Correctness, recovery and scientific gates

Every candidate must pass numerical/model equivalence for its supported scope, exact event/identity/time accounting, owner-thread checks, allocation bounds, and normal resume. P3/P6 additionally pass slow/failing-replica and bounded-queue liveness tests. P1/P7 pass legacy checkpoint compatibility and explicit incompatible-layout rejection. P2 passes mutation invalidation and force-group mismatch cases. P4/P5 pass requested-versus-effective configuration validation.

For the integrated stack, run helper-cpus' publication fault matrix, forced output reorder/delay, disk failure and interrupted final drain. Through the real loaders, require zero accepted partial generations, duplicate or missing committed records, future/abandoned frames, or mislabeled assignment revisions. Demonstrate checkpoint and output recovery together, not merely successful Context deserialization.

Run matched-state force/energy checks before equilibrium comparisons. For changes that alter floating-point accumulation, assess relevant CV distributions, boost/reweighting quantities and equilibrium observables with autocorrelation-aware independent-run uncertainty. Predeclare scientifically meaningful equivalence margins; an underpowered nonsignificant difference is not evidence of equivalence. Report inability to establish equivalence as an unresolved gate.

### 12.6 Promotion decision and benchmark deliverable

Proposed engineering gate for default performance enablement: at least 3% median durable-throughput improvement against the appropriate current accepted reference, a positive lower bound from the independent paired interval, and no reproducible regression above 2% on the held-out supported workload. These are decision thresholds, not predicted gains. A useful feature with a narrower scope can remain opt-in with that scope documented.

P7 additionally needs a predeclared maintenance-value gate: propose at least 5% median durable-throughput improvement over the accepted fused OpenMM stack, with the same positive-interval and correctness requirements. If it only beats the old unfused implementation, it has not justified replacement of the simpler accepted backend.

The final report must include complete manifests; raw per-run throughput/paired ratios and interval method; an equal-work reconciliation table; numerical and thermodynamic validation; fault/recovery outcomes; resource/pause/memory results; requested/effective placement and MPS settings; feature ablations; known unsupported paths; and one decision for each P1–P7 and the combined stack: **promote**, **opt-in only**, **inconclusive**, or **reject**. No speedup is claimed until this report exists.
