# TODO — MPS / CUDA throughput benchmarking and tune-ups (opened 2026-09-22)

## DECIDED 2026-09-22: chignolin_9 runs 236 states under MPS (59 contexts/GPU)

User decision after T6 (MPS cannot be bypassed at 248 contexts). Checklist for chignolin_9:
- [x] **0 (user, 2026-09-23): MBAR connectivity of chignolin_8's sparse pseudo-2D US layout -- DONE.**
      Verdict (commits e3b5786, dc9cb29; `RUNS/chignolin_8_US/ANALYSIS_SUMMARY.md`): connected and
      reweightable (MBAR converged 43 iters, 87% ESS, 481k samples), native hairpin not sampled
      (dG_fold ~ +5.3 kcal/mol), rung distortion persists. Sparse layout is fit to carry chignolin_9.
      Original note: Once
      chignolin_8 has stopped (step 485,200), analyse it as umbrella sampling and check that every
      window is connected and reweightable: (a) rung consistency first -- the 4 rungs per centre must
      give coinciding CV1/CV2 distributions (exchanges used a phantom boost); if they do, collapse to
      62 US states with lambda ignored and boost terms excluded; (b) joint (CV1,CV2) overlap matrix and
      the connected-components check on the thresholded overlap graph -- a disconnected state set is a
      FAIL (the CV1-marginal overlap is blind on this layout: many centres share a CV1 value);
      (c) per-state self-bias (~1 kT expected), MBAR convergence, ESS per state, and whether the sparse
      CV2 spacing (spacing/sigma) leaves gaps. Result decides whether the swarm-designed sparse layout
      is fit to carry chignolin_9's 59-centre design.
- [x] **GaMD stage fix: implemented 2026-09-22 (commit on main after 0401199; seed_frozen_envelope_stage5 +
      verify_gamd_production_stage5).** DEPLOYED to aurum2 on 2026-09-23 after chignolin_8 stopped (both trees,
      DEPLOYED_COMMIT dc9cb29, md5-verified); confirmed live in job 2580889 (stage 5, FSF < 1).
      Original note: chignolin_8 production never left gamd stage 2 (unboosted; see
      run-registry.md). Frozen-envelope production must start in stage 5; add a test asserting FSF < 1
      on a lambda = 1 replica after one warm step. Nothing else on this list matters without it.
- [x] T2 MEASURED 2026-09-23 (job 2580889, see T9): MPS at 59/GPU = 2,307 ns/day/node, 2.75x no-MPS.
      Original note: MPS throughput at 59 contexts/GPU with the REAL Pep-GaMD integrator (fix the
      c8_integ_bench.py P-arm segfault at high context counts, or measure via a short real run).
- [x] State space regen done 2026-09-23: `smoke/config/chignolin_9.yaml` (max_replicas 236, otherwise
      identical to chignolin_8.yaml) + `python -m gareus --config smoke/config/chignolin_9.yaml --out
      RUNS/chignolin_9 --swarm-stage analyze --max-replicas 236 --platform CPU`, reusing chignolin_8's
      real round_000 unbiased-swarm data (symlinked, same peptide/library, frozen-envelope rule) rather
      than re-running MD. Result verified in `RUNS/chignolin_9/swarm/analysis/layout_plan.json`:
      spatial_states=59, cap_spatial=59, kind=sparse (1 unrestrained anchor + 1 region representative +
      18 axis states + 39 joint diagonal-band cells) x 4 rungs = 236 states; gate status=pass; F05
      zero-k unrestrained stack confirmed present (states 0-3, k1=k2=0). windows_lambda_ladder.csv is
      237 lines (236 rows + header) -- a fresh design, not chignolin_8's 249-line CSV with 3 rows cut.
      CAVEAT: this local run had no GENPEPT seed_conformers_dir on disk, so cv_selection_report.json
      records genpept_preset="unknown" instead of chignolin_8's real "chignolin" preset -- cosmetic for
      the layout itself (window centres are identical, verified against chignolin_8's own
      ladder_design.json) but re-run on aurum2 (where the real seed dir exists) before treating
      cv_pair_model.json/cv_selection_report.json as final, not just windows_lambda_ladder.csv.
      NOT YET DONE: deploying this swarm/ tree (round_000 + system + this analysis output) into
      chignolin_9's actual aurum2 out dir so its epoch 0 is skipped rather than re-running 186x1ns of
      unbiased MD -- `epoch0_complete.json` (written by the production driver, not `--swarm-stage
      analyze`) does not exist yet; production would still treat epoch 0 as pending on a fresh launch.
- [x] Launcher applied 2026-09-23: `smoke/config/chignolin_9.sh` (MPS daemon on, `--cuda-mps`,
      `ulimit -n 65536` kept, wait-loop kept, `--us-pull-device-index 0,1,2,3` kept). NOT verified:
      whether the up-to-28 `--us-pull-workers` pull contexts (round-robin over 4 GPUs, ~7/GPU) fully
      close before the 59/GPU production contexts open -- T2's 2,307 ns/day/node measured
      production-only, never pull+production concurrently under MPS -- re-check before a real
      launch. `--cuda-disable-pme-stream true` is confirmed right under MPS (2,307 disabled vs
      2,220 enabled, job 2580889; 2,290 disabled again in job 2608721). `chignolin_9.yaml` does not
      exist yet (state-space item below).
- [x] #4 PME-stream verdict (T8): `--cuda-disable-pme-stream false` = +8-11 % vs blocking-sync
      baseline, keep it -- but re-check under MPS, where the flag was originally set for stability.
- [x] Candidate (T7): compute the 1,256-pair contact sum once per step -- CV2 (residual vs CV1)
      currently re-evaluates it; the two CV forces together cost 26 % per context.
      **IMPLEMENTED 2026-09-23 (branch feat/shared-contact-cv-force):** `gareus/production.py`
      `cv_force_layout` / `add_umbrella_cv_forces` build SHARED_CONTACT_LAYOUT for every new
      residual-torsion-pc-over-contacts campaign (chignolin_9 gets it automatically); resumed
      campaigns rebuild the layout recorded in their secondary_cv_metadata (absent = split).
      **MEASURED 2026-09-23 (job 2608721, d098, `c8_integ_bench4.py` + `c8_sharedcv.sh`): +14.6 %
      node ns/day at the chignolin_9 layout -- worth building.** "shared" = the residual-torsion-pc
      CustomCVForce (group 29) with `+ 0.5*k*((res_contacts/contact_norm)-r0)^2` appended and the CV1
      force removed; same parameter names (k, r0, contact_norm) as `forces.add_contact_umbrella_force`.
      Equivalence vs split at k=1000, ss_k=20: dE = 0, max|dF| 4.8e-7 (CUDA mixed) / 2.3e-10 (double)
      kJ/mol/nm; CV1 term is 9.33 kJ/mol of the 10.89 total, so the check has teeth.
      | real integrator, stage 5 | split (29+31) | shared (29) | gain |
      |---|---|---|---|
      | no MPS, 1 ctx/GPU (2 repeats) | 1,666 / 1,635 steps/s | 1,783 / 1,784 | +8.1 % |
      | MPS, 236 ctx (59/GPU), PME stream off | 2,286 / 2,294 ns/day/node | 2,630 / 2,618 | **+14.6 %** |
      Repeats agree within 0.5 %. CPU load unchanged (~179 cores). Not yet implemented in production:
      needs the fast CV path (`production.py` ~6958, identifies the two CustomCVForces by force group)
      and the CV1 value readout re-keyed to the shared force, and changes the kernel identity -> fresh
      campaign only (chignolin_9 qualifies).
- [ ] **Integrator lever 1 (user, 2026-09-22): replace the water-only auxiliary PME with a cheap
      real-space approximation of the peptide energy.** The boost may be any function of the
      coordinates provided the applied force is its exact gradient and MBAR reweights with the same
      recorded dV, so V_pep only needs to target the right DOFs, not equal the PME peptide energy.
      Candidate: CustomNonbondedForce with interaction groups (138 peptide atoms x environment),
      cutoff / reaction field, in place of group 1's PME. Removes the second PME and one pass:
      single-context Pep-GaMD 1,978 steps/s vs 3,030 (Langevin + 2 PME) / 4,170 (Langevin) -> up to
      ~2x per context. Before building: correlate approximate vs exact V_pep (and its fluctuation
      sigma_V) on chignolin_8 swarm/production frames. Consequences: new kernel identity, new swarm
      envelope calibration (Vmax/Vmin/sigma_V all change) -- fits a fresh campaign, not a resume.

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

## T4 — multi-GPU US pull: IMPLEMENTED 2026-09-22 (`ac549a5`), takes effect at the next phase start

`--us-pull-device-index 0,1,2,3` (new flag; `gareus/seeding.py:us_pull_device_tokens`) makes the
28 pull workers round-robin over all four GPUs, each worker still a single-GPU context. Default
(flag absent) is byte-for-byte the old behaviour. Not `--setup-device-index 0,1,2,3`: that would
make every setup context (minimisation, NPT equilibration, shared GaMD setup) a multi-GPU context.

Expected: 2h11 -> ~35 min per phase start if the pull is GPU-bound. Zero effect on steps/s.
First real measurement will be the `epoch_001` (or `final`) pull -- the epoch_000 pull is done and
checkpointed, so the current chain never re-runs it. Launcher line: `smoke/config/chignolin_8.sh`,
mirrors `~/gareus/chignolin/chignolin_8.sh`. Test: `tests/test_us_pull_device_index.py`.

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

`--cuda-disable-pme-stream true -> false` **queued as the next single-variable A/B** (launcher patched
18:50 for the resubmission after job 2575924; baseline for it is 2575924's 9.25-9.56 steps/s/replica,
blocking sync on in both). Rationale: the separate PME stream was disabled for MPS stability and MPS
is off, so PME may overlap the rest of the force evaluation inside each context.
Still never A/B'd: `--precision mixed`, `--cuda-deterministic-forces false`.

## Related, already fixed — do not re-diagnose

- Descriptor hang at replica 225 (soft `nofile` 1024, CUDA raises only to 4096, ~18 fds/context):
  fixed by `ulimit -n 65536`. Not an MPS problem.
- Chain never checkpointed because bash's `wait` returns early on a trapped signal: fixed
  2026-09-22, see `chignolin_8_chain_no_checkpoint.md`. First real test is job 2567463's TERM.

## T6 — can MPS be bypassed with 248 contexts on 4 GPUs? MEASURED 2026-09-22 (job 2577764): no

`~/gareus/chignolin/c8_integ_bench.py` on the real chignolin_8 system, no MPS, one thread per
context, 250-step chunks, dt 3.5 fs (steps/s per replica; node ns/day):

| arm | 1 context/GPU (N=4) | 248 contexts | kept under time-slicing |
|---|---|---|---|
| L   Langevin | 4,170 (5,045 node) | 50.6 (3,793 node) | 75 % |
| L2  Langevin + 2nd PME | 3,030 (3,665) | 38.9 (2,918) | 80 % |
| P   real PepGaMDLowerDualIntegrator, stage-2 globals | 1,978 (2,393) | **segfault after build** | — |
| B   branch-free, same force/energy group reads as P | 2,010 (2,431) | 17.0 (1,275) | 52 % |
| production (P + CV forces + gareus loop) | — | 9.75 (727) | — |

- **The `if` blocks are not the cost.** Single-context P and B are identical (1,978 vs 2,010). The
  hypothesis that host-evaluated conditionals caused the time-slicing penalty is refuted.
- The Pep-GaMD step costs **1.5x the GPU time of L2** even alone (per-group force/energy reads:
  f0, f1, f2, energy0/1/2 instead of one evaluation), and those extra launches are what time-slicing
  punishes: B keeps 52 % of its single-context rate at 248 contexts, L keeps 75 %.
- **Hard ceiling without MPS**: time-slicing can never beat one context per GPU, so this integrator
  tops out at ~2,400 ns/day/node however the contexts are arranged -- and reaches 1,275 at 248.
  Under MPS, chignolin_7 reached 3,200-3,400 with the same integrator: concurrent kernels from
  several 21k-atom contexts fill SMs that a single context leaves idle. No arrangement of contexts
  without MPS can do that.
- Remaining no-MPS headroom over production: at most 17.0 / 9.75 = 1.7x, and only if the whole
  gap to B is removable. What fills that gap (P's own penalty at 248, the contact/secondary CV
  forces, the gareus loop) is unmeasured because P segfaulted at 248 in the benchmark (production
  builds and runs the same integrator at 248, so this is a benchmark-harness difference, not
  chased).

## T7 — lever 2 (merge bias forces into one group): MEASURED 2026-09-22 (job 2579032), ~2 %, not worth it

Real production CV forces (contact umbrella, 1,256 pairs, group 31; residual-torsion-pc CV2,
group 29) added to the benchmark system (`c8_integ_bench2.py`, `BIAS=split|merged`):

| | split 29 + 31 (production) | merged into 29 |
|---|---|---|
| real integrator, 1 context/GPU (2 repeats) | 1,461 / 1,450 steps/s | 1,481 / 1,479 (+1.8 %) |
| real integrator, 248 contexts | 10.34 | 10.52 (+1.7 %) |
| branch-free, 248 contexts | harness error (B reads both groups in one step) | 10.61 |

Not implemented: +1.7 % does not pay for re-keying the fast CV path (`production.py:6958`
identifies the two CustomCVForces by force group).

What this run corrected in T6:
- **The CV forces cost 26 % per context** (real integrator 1,978 -> 1,461 steps/s). T6's
  branch-free arm had no CV forces, so its "1.7x headroom over production" was an artefact.
  With the real CV forces, branch-free and real integrator are equal at 248 (10.61 vs 10.52).
- **Remaining gap to production is ~6 %** (10.34 in the benchmark vs 9.75 in production): that is
  all the gareus loop (exchange, sampling, barostat, reporters) costs. There is no large
  non-MPS headroom left anywhere in the step.
- The real integrator did not segfault at 248 this time; T6's crash was transient.
- Next per-context lever: the CV2 force (residual against CV1) re-evaluates the same 1,256-pair
  contact sum as the CV1 umbrella every step. Evaluating it once (one CustomCVForce carrying both
  restraints, sharing the contact sub-CV) could recover up to roughly half of the 26 %. Changes the
  force layout -> chignolin_9 candidate, alongside integrator lever 1.

## T8 — `--cuda-disable-pme-stream false`: MEASURED 2026-09-22 (job 2579059), +8-11 %, kept

Same node (d094), same launcher except this one flag; blocking sync on in both.

| | PME stream disabled (2575924) | PME stream enabled (2579059) |
|---|---|---|
| first 20,000 production steps | 2,161 s -> 9.25 steps/s/rep | 1,943 s -> **10.29** (+11 %) |
| steady window (XTC frames) | 9.44 | **10.48** (+11 %) |
| whole run so far | 9.56 | 10.39 (+9 %) |

Against the original spin-sync configuration (2567463, 9.64-9.75) the net gain is +6-8 %.
Production start taken from the resume XTC's birth time (21:40:36); checkpoint 205,200 written
22:12:59. The separate PME stream lets reciprocal-space work overlap direct-space work inside each
context, which time-slicing does not undo. Launcher keeps `false`. Under MPS (chignolin_9) the flag
was historically `true` for stability -- A/B it again there rather than carrying it over blindly.

## T9 — MPS at 59 contexts/GPU, real integrator in stage 5: MEASURED 2026-09-23 (job 2580889)

Real Pep-GaMD integrator + real CV forces, seeded into stage 5 with the frozen swarm envelope at
lambda = 1; the boost was live in every arm (stage 5, FSF_Total 0.94-0.98, FSF_Dihedral 0.70-0.85).

| MPS | contexts (per GPU) | PME stream | steps/s/rep | ns/day/rep | node ns/day | CPU cores/rep |
|---|---|---|---|---|---|---|
| on | 236 (59) | disabled | **32.3** | **9.8** | **2,307** | 0.75 |
| on | 236 (59) | enabled | 31.1 | 9.4 | 2,220 | 0.75 |
| on | 192 (48) | enabled | 37.6 | 11.4 | 2,182 | 0.94 |
| on | 64 (16) | enabled | 163.0 | 49.3 | 3,154 | 0.99 |
| off | 236 (59) | enabled | 11.8 | 3.6 | 840 | 0.77 |

- **MPS at 236 contexts: 2.75x the no-MPS rate** (2,307 vs 840 ns/day/node); per replica 9.8 vs
  3.1-3.6 ns/day. Less than the 3-4.5x extrapolated from chignolin_7: the node aggregate falls from
  3,154 at 16/GPU to ~2,200-2,300 at 48-59/GPU. At 236 contexts ~177 of 192 cores are busy, so
  host CPU (one thread per context) is a likely co-limit under MPS at this count.
- **PME stream under MPS: keep it DISABLED** (2,307 vs 2,220, -4 % when enabled) -- the opposite of
  the no-MPS result (T8, +8-11 %). chignolin_9's launcher must set `--cuda-disable-pme-stream true`.
- The 64-context MPS point reproduces chignolin_7's production (3,154 vs 3,194-3,373 ns/day).
- Stage-5 seeding works on the cluster (boost live in every arm).
