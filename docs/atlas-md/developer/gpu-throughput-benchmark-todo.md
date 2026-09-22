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

## T1 — close the 570 vs 3,195 gap (do this first; costs nothing)

The benchmark excluded exchange attempts, reporting, the barostat and trajectory I/O. Find which of
those dominates before buying throughput elsewhere.

Live lead, not yet chased: `chignolin_8.yaml` sets `report_interval: 2500`, but job 2563608's
parquet manifest shows an effective **250**-step cadence — 50,500 steps produced 50,000 rows across
248 replicas, i.e. ~202 reports per replica. `traj_interval` is also 250. Project memory
(`project_md_perf_levers`) records that `report_interval` bounds the step chunk, so a 10x
discrepancy here is a plausible first-order cost. Confirm which interval actually drives the chunk
size, then A/B it.

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

## T5 — other CUDA knobs never A/B'd on this system

Currently passed: `--precision mixed --cuda-disable-pme-stream true --cuda-use-blocking-sync false
--cuda-deterministic-forces false`. `--cuda-mps` was dropped (it only set the latter two). None of
these has been measured against its alternative at 248 contexts on L40S.

## Related, already fixed — do not re-diagnose

- Descriptor hang at replica 225 (soft `nofile` 1024, CUDA raises only to 4096, ~18 fds/context):
  fixed by `ulimit -n 65536`. Not an MPS problem.
- Chain never checkpointed because bash's `wait` returns early on a trapped signal: fixed
  2026-09-22, see `chignolin_8_chain_no_checkpoint.md`. First real test is job 2567463's TERM.
