# TODO — MPS / CUDA throughput benchmarking and tune-ups (opened 2026-09-22)

Not started. Nothing here blocks chignolin_8; all of it applies to a **future** campaign or to a
config change on a later resubmission. Owner: unassigned.

## Why this exists

The decision "248 states without MPS vs fewer states with MPS" was made on benchmark numbers that
production does not reproduce:

| configuration | benchmark (ns/day, aggregate) | measured in production |
|---|---|---|
| 64 contexts, MPS | 10,940 | — |
| 192 contexts, MPS | 7,494 | — |
| 248 contexts, no MPS | 3,195 | **570** (job 2563608: 50,500 steps in 95 min) |

Production is **5.6x below its own benchmark**, so context scheduling is demonstrably not what is
rate-limiting the real run. Any MPS/state-count redesign argued from the benchmark column alone is
arguing from a regime the production loop never reaches. Close that gap first.

Benchmark harness already exists and prints ns/day: `~/gareus/chignolin/aurum_ctx_test.py N`
(round-robin over `TEST_DEVICES`, one thread per replica, one open file per replica, real
chignolin_8 `base_system.xml`). It builds **bare replica contexts only** — no setup context, no
pull sims.

## T1 — RESOLVED 2026-09-22 14:40: the gap is the no-MPS time-slicing regime, not the pipeline

Measured on job 2567463 (d094, 248 replicas, production from 13:52:22; all numbers re-derived from
the node, the 20k-step checkpoint and `progress.jsonl`, not from the dashboard):

- **Not a regression from the launcher/config update.** Job 2563608 (before the `wait` fix,
  `checkpoint_interval` 250000, pull 10000) ran 8.86 steps/s/replica; 2567463 runs 9.7 (median of
  296 production records). The 20k checkpoint had not even fired when the rate was first measured.
- **GPU**: 4 x L40S at 97-99 % "utilisation" but only 160 W of 350 W and 25 % memory-bus
  utilisation — a kernel is always resident, the SMs are mostly empty. That is what 62 contexts
  time-slicing one GPU without MPS looks like: kernels from different contexts never overlap.
  Assignment is balanced (31.1 GB x 3, 31.5 GB on GPU 0 which also hosts the setup context).
- **CPU**: the process holds **7,474 threads** and burns **165-190 of 192 cores** continuously.
  248 replica threads spin-wait (`UseBlockingSync=false`, inherited from the old `--cuda-mps` soft
  default) on a 192-core node: hot threads show ~52k involuntary vs ~12k voluntary context switches
  — the scheduler is preempting them. Affinity is unpinned (0-191); GPUs sit on NUMA nodes
  3/9/15/21 of 24. ~7,000 of the threads are idle and of unidentified origin (not a cost, noted).
- **Barostat** (`biased_mc`, every 200 steps): 100 attempts/replica in 2,075 s of production;
  `evaluate_s` median 29.7 s + `read_s` 4.7 s = **34 s/replica = 1.7 % of wall**. 27 % acceptance.
  Each U* evaluation is 5 group-energy `getState` reads, two evaluations plus a positions read and a
  restore per attempt (~12 syncs). Per-sync cost measured from it: **47-59 ms** — one full
  62-context scheduler round.
- **Exchanges** (`gibbs-walk`, every 400 steps): 12,030 attempts / 20k steps, 16.9 % accepted.
  Rung (lambda) swaps dw1/dw2/dw3 accept 20.6 / 6.6 / 14.9 %; CV-neighbour swaps dw4/dw8/dw16 accept
  67 / 41 / 68 %. 341 of 674 pairs with >=5 attempts never accepted. The stop-the-world barrier
  costs nothing measurable: over 60 s at 200 ms sampling, GPU util < 50 % in 1.6-2.3 % of samples,
  longest dip 0.4 s. (`report_interval` 2500 vs the observed 250-step cadence is
  `distance_output_interval`/`traj_interval` by design, and irrelevant at this barrier cost.)
- **Why the Langevin benchmark under-predicted the MPS penalty by 2x**: `aurum_ctx_test.py` stepped
  a plain `LangevinMiddleIntegrator`. The production Pep-GaMD `CustomIntegrator` reads three energy
  groups into host-evaluated globals (`PepE0/1/2`, `gareus/pep_gamd.py` `_setup_energy_values`) and
  four force groups (`f0`, `f1` = second PME, `f2`, bias) **every step**, so every step carries >=2
  host<->device round trips. Without MPS each round trip waits one scheduler round (~50 ms measured
  above); 2 x 50 ms ~= the observed 103 ms/step. Under MPS a round trip is kernel latency.
- **The real-workload MPS number the decision needed** (same integrator, dt, cadences, node class;
  `progress.jsonl`, median over all production records at that replica count):

  | run | replicas | contexts/GPU | ns/day/replica | node aggregate ns/day |
  |---|---|---|---|---|
  | chignolin_7, MPS | 16 | 4 | 41.2 | 659 |
  | chignolin_7, MPS | 32 | 8 | 105.4 | **3,373** |
  | chignolin_7, MPS | 64 | 16 | 49.9 | **3,194** |
  | chignolin_8, no MPS | 248 | 62 | 2.9 | **727** |

  The 248-state no-MPS design costs **4.4-4.6x node throughput** against the MPS regime this
  integrator was already measured in. (The 16-replica row is below the 32-replica row; whatever
  that phase was doing, it is not the regime of interest and was not investigated.)

Consequences for the items below: T2 is no longer optional — it is the only unknown left (does MPS
hold its ~3,200 ns/day at 59 contexts/GPU, or does time-slicing overhead return?). T5's
`UseBlockingSync=true` is the one single-variable A/B that can run on the *current* design at its
next resubmission: it frees ~180 cores of spin and the involuntary-preemption penalty; expected
effect is second-order next to MPS, but it is free to measure and it becomes first-order once a
236-context MPS run puts 236 spinning threads back on 192 cores.

## T2 — measure the missing MPS points

`aurum_ctx_test.py` has only ever been run at 64, 192, 240 and 248. Throughput was recorded at 64
and 192; the 240 run only established that contexts *build*. Needed, MPS on, ~10 min each:

- `aurum_ctx_test.py 236`
- `aurum_ctx_test.py 240`

At ~60 clients/GPU, MPS time-slicing may eat most of the nominal gain — that is the number the
state-count decision actually needs and nobody has it.

## T3 — if T2 says MPS is worth it: land on 236, not 240

- 248 = **62 centres x 4 rungs** (`windows_lambda_ladder.csv`: 4 distinct `gamd_lambda` values,
  62 rows each; 16 distinct CV1 x 5 distinct CV2, sparsely filled).
- `swarm/analysis/ladder_design.json` already says `n_states: 60`, `n_windows: 15` — the 62 comes
  from the **joint CV2 layout** expanding those 15 windows, not from the ladder requesting 62. The
  clean lever is regenerating the CV2 layout to land on 60 centres, **not** deleting 2 rows from
  the CSV. Which centres to drop is a physics call; note one of the 62 is the deliberate zero-k
  unrestrained exploration stack from F05, so that one is not a free deletion.
- **240 has zero headroom.** The ceiling is ~60 *per GPU* and 240 round-robins to exactly
  60/60/60/60, but production also holds a setup context on GPU 0
  (`[setup] OpenMM setup platform: CUDA {... 'DeviceIndex': '0' ...}`) and, during seeding, 28 pull
  sims on the same GPU. 61 clients on GPU 0 fails exactly as the 241st context did
  (`The requested CUDA device could not be loaded`). Target **236 = 59 centres x 4 rungs**.

## T4 — multi-GPU US pull

The umbrella pull runs **all workers on GPU 0** while GPUs 1-3 idle: `gareus/seeding.py:1319` takes
`device_tokens` from the *setup* platform props, and `setup_platform_and_properties`
(`gareus/system_setup.py:318`) deliberately uses only the first entry of `--device-index`. Measured
cost: 10,000 steps x 248 windows = 2h11 end-to-end, which is why `us_pull_steps_per_window` is
capped at 12,500 (a longer pull risks not finishing inside the 3h50 TERM, and an unfinished pull is
redone from scratch by the next job).

`--setup-device-index 0,1,2,3` would round-robin the pull workers over 4 GPUs (each
`_make_pull_sim` gets a single token, so each pull sim stays single-GPU) — but it also makes every
*other* setup context a 4-GPU context, which is untested here and probably slower for a 21k-atom
system. Needs a scoped test before use.

## T5 — `UseBlockingSync=true` A/B: MEASURED 2026-09-22, no gain (-2 to -4 %)

Single-variable test on the live campaign: job 2567463 (`UseBlockingSync=false`, spin) vs its
successor 2575924 (`true`, everything else identical, resumed from the 54,400-step checkpoint on the
same node d094).

| | spin (2567463) | blocking (2575924) |
|---|---|---|
| first 20,000 production steps, wall clock | 2,075 s -> **9.64** steps/s/rep | 2,161 s -> **9.25** |
| XTC frame count, whole run so far | 9.75 (median of 326 records) | 9.56 (83 frames, 36 min) |
| XTC frame count, steady window | 9.75 | 9.44 (17:51-18:03) |
| cores busy (of 192) | 165-190 | 156-173 |
| GPU power (of 350 W) | 160 W | 173-181 W |
| hottest thread, voluntary / involuntary switches | 12k / 52k | 1.1M / 158k |

The switch did what it says on the tin -- threads now sleep and wake (1,100 wake-ups/s each)
instead of spinning and being preempted -- and the GPUs drew 11 % more power, but the step rate
did not move (slightly down, consistent with the wake-up latency now sitting on every one of the
>=2 host round trips per step). **CPU oversubscription was not the limiter.** The ~0.9 core per
replica thread persists under blocking sync, so it is not spin either; it is the host side of the
CustomIntegrator step (kernel launches, per-step global evaluation, energy downloads) times 62
contexts per GPU waiting on one another. That leaves the time-slicing regime itself (T2/T3, MPS)
as the only lever with a measured payoff (c7: 3,200-3,400 ns/day/node vs 727 here).

Flag left at `true` in `~/gareus/chignolin/chignolin_8.sh` (line 242) and the repo mirror: neutral
on throughput, ~15 fewer cores burnt. Revert to `false` if an exact baseline configuration is
wanted for a later comparison; nothing else depends on it.

Remaining, never A/B'd on this system: `--precision mixed`, `--cuda-disable-pme-stream true`,
`--cuda-deterministic-forces false`.

## Related, already fixed — do not re-diagnose

- Descriptor hang at replica 225 (soft `nofile` 1024, CUDA raises only to 4096, ~18 fds/context):
  fixed by `ulimit -n 65536`. Not an MPS problem.
- Chain never checkpointed because bash's `wait` returns early on a trapped signal: fixed
  2026-09-22, see `chignolin_8_chain_no_checkpoint.md`. First real test is job 2567463's TERM.
