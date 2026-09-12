# NPT Package B Report — production integration

Branch `fix/npt-production-integration`, worktree `.claude/worktrees/npt-integration`.
Governing spec: `docs/superpowers/specs/2026-09-12-npt-correction-design.md` (Package B = §9's `gareus/production.py` integration, `gareus/system_setup.py` ownership/preflight, `gareus/config.py`/`gareus/cli.py` options, and the new stepping/report driver; `gareus/npt.py` itself is merged Package A, untouched here).

## What was implemented

| Commit | Content |
|---|---|
| `f23613a` | `gareus/npt_driver.py`: `ReplicaStepDriver` + `NptRunContext` (+ `tests/test_npt_driver_scheduling.py`) |
| `d06f55f` | Barostat ownership resolution, physical preflight, CLI flags (+ `tests/test_npt_ownership.py`, `tests/test_npt_cli_args.py`) |
| `063062a` | All production stepping paths routed through the driver + ownership; checkpoint NPT block; legacy rejection |
| `b950b81` | Retire the package-1 interim shim (dead after PR #88); smoke-test skip retargeted to package 2 |
| `8f884c8` | `tests/test_npt_production_integration.py` — 19 production-level gates over the REAL controller |
| (this commit) | This report + `.gitignore` negation |

### `gareus/npt_driver.py` (new module)

- **`ReplicaStepDriver`** — per-replica stepping/report driver. Subdivides an MD chunk at the replica's *own* barostat and reporter deadlines; no cross-replica barrier per volume attempt (§7). Enforces the coincident-endpoint order **integrate → finish due volume move → emit reports of that same post-move state**. Reporters are detached from `sim.reporters` and driven through the public `describeNextReport()`/`report()` interface with a fresh post-move `State` — never `Simulation.step()`'s built-in dispatch, which would emit pre-move frames at the same step as post-move scalars (§7). Scheduling uses the controller's *persisted next-due integration step*, not report or exchange counts. A stuck controller schedule fails loudly instead of looping. `advance(..., reports=False)` supports rollback probes. The controller-less path matches plain `Simulation.step` frame-for-frame.
- **`NptRunContext`** — carries the resolved `BarostatOwnership` plus the effective-potential adapter; `initialize_controller` gives every context its own independently seeded controller (never clones one calibration RNG stream into every replica, §6). Seeding helpers for shared GaMD setup, recon windows, replicas and the joint check run.
- **`NPT_REPORT_PHASE_NOTE` / `NPT_CHECKPOINT_PHASE_NOTE`** — label metadata pinning the phase semantics: the step-500 frame is post-volume-move/pre-exchange; the checkpoint is post-exchange (§7).

### `gareus/system_setup.py`

- **`resolve_barostat_ownership`** — the single decision point, running *before any Context is created from a System*. Delegates the `auto`/`native`/`biased_mc` dispatch (including the loud failures for explicit-native-with-boost and unsupported boosted modes) to the frozen `gareus.npt.resolve_npt_backend` contract. NVT needs no resolver and rejects an explicit backend. `biased_mc` Systems are built **without** the native `MonteCarloBarostat` — removal-before-Context-creation by construction (§6).
- **`preflight_barostat_ownership`** — runs on the fully-assembled System before any Context exists: no anisotropic/membrane/flexible barostat family; exactly one volume controller for `native` (zero for `biased_mc`, whose controller is attached per replica later); a periodic box; no immobile non-virtual-site particle for `biased_mc`; boosted dynamics refused under `native`.
- The interim `NotImplementedError`→`RuntimeError` translation added in `063062a` (written while package 1 was pending) became dead code when PR #88 merged; `b950b81` removes it — the resolver now delegates to the contract directly.

### `gareus/cli.py` / `gareus/config.py` / `gareus/provenance.py`

- `--npt-barostat-backend {auto,native,biased_mc}` (default `auto`) and `--barostat-volume-step-fraction` (default `0.01`), with argument-level validation: explicit backend under NVT, `native` with a boosted run mode, and non-finite or out-of-(0,1) fractions all fail at parse time. Both keys are recorded in the run manifest's `method_settings` (a resume must not silently change who owns volume moves) and appear in the YAML config template.

### `gareus/production.py`

- `run_gareus` resolves barostat ownership *before* the base System is built, runs the physical preflight on the assembled System, and threads an `NptRunContext` through every context-advancing path: shared GaMD setup, multiwindow reconnaissance (per-window controllers with independent seeds), the 50-step joint-calibration check run, the production probe (reports suppressed, controller state rolled back), replica construction (controller init on each replica's own worker; reporters registered with the driver, never `sim.reporters`), `step_all`, accepted window swaps and stuck-replica rescue (context writes routed through the affinity pool; CV cache invalidated), and checkpoint save/load (per-replica controller `state_dict` captured/restored on its own worker; manifest carries the NPT block — backend, frequency, T/P, adapter ids, per-controller RNG algorithm/state, molecule-partition fingerprint, counters, next-due step, schema version; legacy boosted-NPT checkpoints rejected with a specific explanation, §8).
- Swarm `ensure_system` resolves ownership (run_mode `cmd` → native) and preflights before serialising `base_system.xml`.
- `gareus_metadata.json` / `umbrella_pymbar_metadata.json` record the resolved backend, the preflight report and the energy-column semantics (§7's "raw Context potential must not be presented as physical or effective energy when the auxiliary force exists").
- **`_resolve_npt_adapter`** — the single package-2 seam: when `biased_mc` is resolved, `pep_gamd.build_effective_potential_adapter(args)` must exist or the run fails here, loudly, never by silently falling back to the wrong acceptance energy.

## Gates (spec §10) — verbatim results

Package B's core gates are **Transactions and scheduling** and **Resume and ownership**; the integration tests additionally exercise the transaction half of **Real OpenMM behavior** (real Contexts, real controller, real DCD reporters on the deterministic Reference platform) and the probe-rollback part of **Production validation conditions**. The physics oracles (acceptance mathematics, boost/force agreement, coupled volume/energy target) are Package A's and were not re-run.

### Transactions and scheduling — PASS

`tests/test_npt_driver_scheduling.py` (commit `f23613a`, contract-honoring fake controller + real DCDReporter) and `tests/test_npt_production_integration.py` (REAL `BiasedMCBarostatController` on real tiny OpenMM Contexts):

- 500/250/100 and mutually incommensurate strides with exact due steps and **no hidden extra MD steps**;
- coincident-endpoint frames provably **post-move** (step-500 DCD unit cell equals the post-move box and differs from the pre-move one — the test refuses to pass vacuously);
- rejected trials leave positions/box/velocities/time/step count unchanged while still consuming draws and advancing the schedule; accepted trials scale the box by exactly `s=(V'/V)^{1/3}`, translate the molecule by `(s−1)·Rm`, preserve internal geometry and do not rethermalize velocities;
- cache invalidated after every attempted move (no stale exchange cache); exactly-one-volume-controller asserted by preflight.

### Resume and ownership — PASS

- Checkpoint round-trip through the production save/load functions onto **fresh Contexts**: restored controllers continue the saved streams exactly (identical event sequences, counters, next-due steps, RNG state) and the resumed MD trajectory is bit-identical to the uninterrupted one on this platform.
- Replica affinity extended to volume moves and reports: every real volume-move energy evaluation ran on one of the replicas' pinned workers, never on the dispatching thread (journal-verified against the affinity pool's own worker idents).
- Legacy boosted-NPT checkpoint rejected with a specific explanation; legacy-NVT checkpoint fresh-inits controllers; backend / adapter-id / temperature / pressure / atom-count mismatches all rejected; the NVT resume never grows a volume controller.
- Uninterrupted-vs-checkpoint/resumed proposal streams compared with no extra move on resume.

### Production probe (validation-conditions fragment) — PASS

The 50-step probe suppresses reports and rolls back **both** the Context and the controller state; a mid-transaction volume-move failure (exploding adapter at the proposed endpoint) still restores the pre-trial state, rolls the probe back and surfaces loudly in `production_probe_report.json` with the physical-energy source presented explicitly.

### Verbatim results (recorded in the pre-commit run; this handoff's finalisation did not re-run them)

`tests/test_npt_production_integration.py` alone:

```
19 passed
```

Full suite (`tests/`, same standing exclusions as Package A's run):

```
1 failed, 2910 passed, 4 skipped
```

The single failure is **pre-existing and unrelated** to this work: `tests/test_example_configs.py::test_example_config_parses_without_unknown_keys[chignolin_genpept_contact_bias_sigma.yaml]` — `SystemExit: 2`, `gareus: error: the following arguments are required: --seq`; introduced in commit `9e72a22` (2026-09-03) and independently confirmed identical on the base commit by Package A. The 4 skips include the slow boosted-NPT end-to-end smoke test, which self-heals once package 2 lands (see below).

## What was NOT done, and why

- **L40S GPU performance gate (§10 "Performance")** — cannot be run in this environment (no L40S access). No benchmark numbers are claimed; none were fabricated. Must be run where the four-GPU rig lives, after package 2 lands (there is nothing meaningful to benchmark until the corrected boosted-NPT path actually runs).
- **Package 2 — the stage-aware effective-potential adapter (`gareus/pep_gamd.py`)** — explicitly out of scope and explicitly off-limits in this finalisation. Until it lands, `pep_gamd.build_effective_potential_adapter` does not exist and every boosted-NPT run fails loudly at `production._resolve_npt_adapter`; the slow end-to-end smoke test skips on exactly that condition. **Interface note for package 2:** Package A shipped `make_npt_target_adapter(system, integrator, args)`; the production seam expects a per-run factory `build_effective_potential_adapter(args)` returning the adapter object `NptRunContext` drives (`snapshot`/`evaluate` → `EnergyBreakdown`). Bridging those two shapes is package 2's first job.
- **Production-scale "Real OpenMM behavior" arm** — the integration tests run real Contexts and the real controller, but on tiny deterministic Reference-platform systems. The head-to-head comparison against a conventional physical-System native `MonteCarloBarostat` at production scale, the small-timestep→4 fs agreement study, and the PME/volume-dependent-correction verification all need the complete corrected chain (package 2) and production-scale systems.
- **Full "Production validation conditions"** (freeze envelopes, disable noncanonical rescue, equilibrium distribution comparison) — a campaign-level exercise requiring the complete corrected chain; only the probe-rollback fragment is gated here.
- **Legacy checkpoint migration** (§8's "separately requested migration" extracting positions/velocities/box from a legacy System) — not implemented; only the mandated rejection-with-explanation is.

## Judgement calls the spec left open

1. **Adapter substitution in tests.** The only substituted piece in the integration tests is the package-2 adapter, and it goes through the same `NptRunContext` seam production code itself uses (never monkeypatching internals). Three deterministic stand-ins: physical-energy (raw Context potential — valid for the auxiliary-force-free test System), reject-any-change (U* = 1e12 off the start volume, so rejection never relies on chance) and exploding (exercises restore-then-abort).
2. **Sequential replica advancement for exact resume comparisons.** Measured on this machine (OpenMM 8.5.1, Reference platform): the stochastic-integrator noise draws come from **one stream shared across every Reference-platform Context in the process**, in `step()`-call granularity — two replicas stepping concurrently interleave nondeterministically. Exact uninterrupted-vs-resumed comparisons therefore advance replicas sequentially (deterministic draw order) and run the uninterrupted continuation before the fresh resumed Contexts are constructed. Documented in the round-trip test's platform note.
3. **Removal-before-Context-creation implemented structurally.** The native barostat is simply never added to `biased_mc` Systems (the ownership decision precedes System construction) rather than being deleted from an already-built System — same guarantee, no mutation path to get wrong.
4. **Legacy-NVT checkpoints fresh-init controllers.** The spec pins only the boosted-NPT rejection; a legacy NVT (or otherwise controller-less) checkpoint starts fresh controller streams on resume rather than being rejected.
5. **Probe rollback replaces `driver.controller`** with the restored controller (the advanced object is deliberately discarded) — the test reads the driver's controller, and the semantics are documented there.
6. **The smoke-test skip is existence-keyed**, on `hasattr(pep_gamd, "build_effective_potential_adapter")` — no version sniffing, and it disappears by itself when package 2 lands.
7. **Reporting mechanism.** §7 mandates reporting *after* the volume move but not how; the driver detaches reporters from `sim.reporters` and dispatches them itself via the public reporter interface with a fresh post-move `State`, rather than relying on `Simulation.step()`'s dispatch order or a barostat-first reporter trick (both of which the spec warns about).

## Spec / cartography findings

- **§10's "use deterministic platforms for strict comparisons" is necessary but not sufficient.** On the Reference platform, stochastic-integrator noise comes from one stream shared across *all* Contexts in the process at `step()`-call granularity, so concurrent replicas interleave nondeterministically even on the deterministic platform. Exact resume comparisons need sequential advancement (or a noise-stream discipline the spec does not mention). This cost real debugging time and should be a named caveat in the spec's resume gate.
- **The A/B package boundary was under-specified.** §9's table lists "Stage-aware supported boost adapter" for `gareus/pep_gamd.py` but never names the run-facing factory or its shape. Package A shipped `make_npt_target_adapter(system, integrator, args)`; production needs `build_effective_potential_adapter(args)`. The bridging seam (`production._resolve_npt_adapter`) had to be invented here, and until package 2 lands the two halves do not compose — correctly loud, but the spec's "land internal work in that order" sequencing did not anticipate that package B's end-to-end path cannot run at all until the bridge exists.
- **§7's reporter-ordering concern is avoidable by construction, not just workable-around.** The spec frames "Simulation may assemble a shared State before reporter callbacks execute" as a hazard to schedule around; the shipped design sidesteps it entirely by never using `Simulation.step()`'s reporter dispatch for controller-carrying replicas. A spec revision could state the requirement as "reporters must be driver-dispatched", which is simpler to test.
- No factual errors were found in the spec's §4–§8 mechanics as implemented; the acceptance/Jacobian/report-order contracts matched Package A's controller and this package's driver without forced reinterpretations.

## How to re-verify

```
python -m pytest tests/test_npt_ownership.py tests/test_npt_cli_args.py \
    tests/test_npt_driver_scheduling.py tests/test_npt_production_integration.py -q
```

(The full-suite line above used the repo's standing exclusions; the single failure is the pre-existing `--seq` config-parse issue, unrelated to NPT.)
