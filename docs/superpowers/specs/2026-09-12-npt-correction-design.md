# ATLaS-MD: NPT correction specification

Status: proposed design for review; no implementation changes made.

Reviewed repository: `sulcjo/ATLaS-MD`, commit `a792cee281cef87af798102665549c91ce47919f`. Scope: correct volume sampling for the existing GaMD/Pep-GaMD workflow, retaining OpenMM CUDA, HMR/4 fs, 64–128 replicas on four L40S GPUs, exchange every 500 steps and output every 250 steps. No MTS or kinetic fidelity requirement.

## 1. Decision

Introduce an application-controlled, isotropic Metropolis volume move for supported boosted simulations. Its acceptance energy must include the actual GaMD boost and all physical restraints, and exclude the auxiliary water-only energy. Run it on each replica's existing context-owning worker. Keep the native OpenMM barostat for conventional MD contexts whose exposed energy equals their propagated potential.

This corrects the barostat's target distribution. It does **not** remove finite-timestep bias from the existing Langevin/GaMD propagation. Equilibrium fidelity at 4 fs must still be checked against a smaller timestep; exact sampling at arbitrary timestep would require a separate integrator correction such as an appropriately designed HMC scheme.

## 2. Evidence and failure mechanism

The following are source-inspection findings, not runtime reproductions:

| Location | Relevant behavior |
| --- | --- |
| `gareus/system_setup.py:create_system` | Adds `MonteCarloBarostat` when requested. |
| `gareus/pep_gamd.py:ensure_pep_gamd_partition` | Adds a second, water-only nonbonded force for measuring the peptide contribution. |
| `PepGaMDLowerDualIntegrator` | Explicitly combines force groups to exclude the auxiliary force and apply the dependent dual boost. This transformed potential is not exposed as the Context's ordinary potential energy. |
| `gareus/production.py:make_cmd_integrator` | Excludes auxiliary group 1 from conventional integration when present. That fixes this auxiliary contribution for that path, but does not supply a missing GaMD boost. |
| Shared calibration, reconnaissance and production constructors | Can construct boosted contexts from a base System containing a native barostat. All these paths need coverage. |
| Production stepping/reporting/checkpointing | Uses replica worker affinity, attached Simulation reporters, cached observations, label exchange and binary checkpoints. A volume move must participate in these mechanisms. |

OpenMM's native implementation evaluates Context potential energy using the integrator's integration force groups. The Pep-GaMD force transformation is implemented in the integrator, whereas the auxiliary force remains in the System. Consequently, the current boosted path can accept volume changes using physical energy plus auxiliary energy, without the boost. Setting integration force groups alone cannot express the nonlinear dependent dual boost. See the [native barostat implementation](https://github.com/openmm/openmm/blob/master/openmmapi/src/MonteCarloBarostatImpl.cpp).

Even lambda zero is affected if the auxiliary energy enters acceptance. For nonzero lambda, omitting the auxiliary group still leaves the missing boost. Upstream GaMD also reads its working energies before `addUpdateContextState`; an internal barostat can therefore change the configuration after those reads. The proposed external move completes before the next integration step begins.

## 3. Target energy and state contract

For window/label a, define

$$
U_a^*(x,B)=U_{\mathrm{phys}}(x,B)+W_a(x,B)+\Delta_a(x,B).
$$

Here B is the box matrix, W contains every actual umbrella, secondary-CV and other unscaled bias, and Delta is the boost actually applied by the integrator. Auxiliary bookkeeping forces contribute only to constructing boost inputs, never as additional physical energy.

For the current Pep partition:

$$
V_{\mathrm{pep}}=E_0-E_1+E_2,\qquad V_d=E_2,
$$

$$
b_d=b(V_d;\theta_d,k_{0d}),\quad
b_t=b(V_{\mathrm{pep}}+b_d;\theta_t,k_{0t}),\quad
\Delta=b_d+b_t.
$$

E0 and E2 are physical groups; E1 is the auxiliary water-only group. The remaining physical/bias forces are included once, unscaled. Implement an explicit force inventory: do not equate the helper `total_energy_groups()` with total physical energy; it describes the integrator's boost channel.

The channel function must match the supported integrator's threshold, small-range and branch guards, including the repository's dependent ordering. Lambda scales each channel's k0; it is **not** a final linear multiplier on the completed dual boost.

For each attempted move, snapshot the current stage, current channel parameters and current window parameters. Use this immutable snapshot for both old and proposed energies. Read the *actual* k0 values from the integrator; if constructing an envelope from those values, evaluate it at lambda one to avoid applying lambda twice. Do not use cached integrator energy or boost globals to evaluate proposed coordinates.

In the integrator's conventional-MD stages, Delta is zero. In frozen production stages, evaluate the frozen boost. Calibration may update parameters between moves, but not during a move; adaptive calibration is not stationary ensemble sampling and remains excluded from production analysis. Unsupported stages or boost implementations fail before production.

## 4. Volume proposal and acceptance

Use molecule translations, preserving every molecule's internal geometry. Match OpenMM's molecule partition, including constraints and virtual-site connectivity, rather than residue counts or a water-name heuristic.

For volume V = det(B), propose

$$
V'=V+\delta,\quad\delta\sim\mathrm{Uniform}[-a,a],\quad
s=(V'/V)^{1/3},\quad B'=sB.
$$

For molecule m with consistently represented center Rm, translate all its atoms by `(s-1)*Rm`. Internal vectors, constraints and velocities remain unchanged. Whole-molecule periodic imaging may be used; independent atom wrapping must not tear molecules apart. Establish and test a reversible center/imaging convention. Recompute virtual sites when required. Do not project scaled atomic coordinates back onto constraints: that is a different proposal with an unaccounted Jacobian.

The absolute proposal half-width a is fixed during production. An initial configuration option sets it to a fraction of that replica's starting volume, then stores the resulting value in nm³. It must not silently become a fraction of the *current* volume.

Accept using

$$
\log A=-\beta[U_a^*(x',B')-U_a^*(x,B)+P(V'-V)]
+N_{\mathrm{mol}}\log(V'/V),
$$

and `log(u) < min(0, logA)`. Use beta = 1/(R T) with molar energies. The pressure conversion is `1 bar nm³ = 0.0602214076 kJ/mol`.

The Jacobian counts translated molecules, including individual ions. It does not count atoms or independent constrained degrees of freedom. This symmetric proposal in V has no additional +1 exponent; a proposal symmetric in log(V) would require a different rule. These conventions follow the molecular volume-move construction in the [OpenMM barostat theory](https://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#montecarlobarostat).

Reject nonpositive volumes and boxes outside OpenMM's permitted periodic/cutoff domain without redrawing. Support fixed-shape isotropic scaling of valid OpenMM boxes, including triclinic boxes, with determinant and face-height checks. Do not support anisotropic or membrane barostats in this change. Reject unsupported molecule connectivity, immobile particles or external volume-changing machinery explicitly rather than applying an unverified Jacobian.

## 5. Trial transaction and energy evaluation

Each replica owns a controller, RNG and proposal schedule. A trial runs entirely on its context-owning worker:

1. Snapshot raw coordinates, box and target parameters. Draw the proposal and acceptance uniform from the replica's barostat RNG.
2. Evaluate current U* from fresh force-group energies.
3. Set proposed box and molecule-translated positions, update virtual sites, and evaluate proposed U* under the same parameter snapshot.
4. Accept or restore the original box and positions. Never step MD, update calibration statistics, rethermalize velocities or change labels during a trial.
5. Invalidate all configuration-dependent observation caches after acceptance. Conservatively invalidate after any trial until restoration/cache behavior is proven. Return move statistics and any reusable *current-state* energy observation.

Both energy evaluations include PME reciprocal contributions and every enabled volume-dependent correction, including dispersion corrections. The auxiliary and physical PME parameterization remains matched as established by the current partition code. Do not retune alpha, grids, cutoffs or force parameters as part of a trial, or evaluate the two endpoints with different numerical Hamiltonians.

Expected geometric invalidity rejects cheaply. A finite old energy and nonfinite trial energy may reject with a counted numerical-rejection reason. Nonfinite old energy aborts. Unexpected evaluation errors restore the original state and then abort with context; they must not become silently accepted or silently repeated rejections. Restoration failure is fatal. Rejection consumes the barostat random draws but leaves physical state, MD time and integration step count unchanged.

The first implementation uses Python coordinate transfers and existing Context energy calls. This keeps the energy definition inspectable and avoids a compiled plugin. It is a correctness-first baseline; overhead must be measured on the intended replica density.

## 6. Backend selection and coverage

Proposed option: `npt_barostat_backend = auto | native | biased_mc`.

| Configuration | auto behavior |
| --- | --- |
| NVT | No barostat or volume scheduling. |
| Conventional MD with validated physical integration groups | Native OpenMM barostat. |
| Supported Pep lower-dual or supported lower-dihedral GaMD | Application-controlled biased MC. |
| Other boosted modes lacking a validated target adapter | Clear preflight error for NPT, including mode and missing support. |

Explicit native + boosted dynamics must fail rather than preserve the known mismatch. Explicit biased_mc can be used with zero boost for reference comparisons. Remove the native barostat from application-controlled Systems **before Context creation**. Assert that exactly one volume controller exists. Do not change the requested ensemble or silently switch to NVT.

Retain existing pressure, temperature and barostat-frequency settings, including the existing production override semantics. Proposed additional option: `barostat_volume_step_fraction = 0.01`, converted once to fixed absolute width. No automatic production-width adaptation in the initial implementation. Log resolved backend, pressure, temperature, frequency, molecule count, width, stage adapter and numerical settings.

Apply the controller to shared GaMD setup, multiwindow reconnaissance, production, resumed production and any validation/probe path that advances these contexts. Temporary check runs that are intended to roll back must also roll back controller state. Ordinary physical NPT preparation can retain its native barostat. Calibration must supply stage-specific target snapshots and independent controller initialization; never clone a calibration RNG stream into every replica.

## 7. Scheduling, output and exchange

Preserve the existing one-worker-per-replica ownership. Each worker subdivides its assigned MD chunk at local barostat and reporter deadlines. Do not add a cross-replica barrier for each volume attempt. Keep the existing outer synchronization for sampling/exchange/checkpoints.

At a coincident endpoint, the required order is:

1. Integrate to the endpoint.
2. Finish the due volume move.
3. Emit due trajectory frames and scalar observations of that same state.
4. Attempt label exchange using fresh current-state observations.
5. Save a due checkpoint after exchange.

Current attached Simulation reporters run inside `Simulation.step()`. Adding a barostat callback after `step()` would therefore create pre-move trajectories and post-move scalar records at the same step. The implementation must explicitly schedule reporting after the volume move: register supported reporters with the new stepping driver, advance with automatic reporter dispatch disabled, and call reporters with a fresh post-move State. Preserve reporting requirements for positions, velocities, energies and periodic imaging; test the installed OpenMM reporter interface. Do not rely on a barostat reporter inserted first, because Simulation may assemble a shared State before reporter callbacks execute.

Energy columns must distinguish physical U, boost, umbrella and effective U*. A generic StateDataReporter's raw Context potential must not be presented as physical or effective energy when the auxiliary force exists.

For the user's 500/250 strides and a barostat frequency of 100, the step-500 frame is post-volume-move and pre-exchange; the checkpoint is post-exchange. Label metadata must record this distinction. Barostat scheduling uses a persisted next-due integration step, not report count or exchange count.

Existing exchanges swap labels, not boxes or coordinates. At common temperature and pressure, Uphys, pV and the common volume measure cancel from the swap ratio; keep the existing umbrella-plus-boost structure, with fresh energies and the correct current labels. Rebuild/invalidate target snapshots after each label change. Common-temperature/common-pressure MBAR comparisons likewise need no new state-dependent pV term, but sample volumes and provenance must be retained. Different-temperature or different-pressure exchange is outside scope.

## 8. Checkpoints and compatibility

Extend the versioned checkpoint manifest with per-replica backend, molecule-partition fingerprint, RNG algorithm/state, fixed proposal width, counters, last/next due step and controller schema version. Save these atomically with the corresponding binary Context checkpoint and existing label/integrator metadata.

Resume must restore the exact schedule and random stream without attempting an extra move. Validate topology, target adapter, T/P and backend compatibility before accepting a checkpoint. Record relevant dependency versions, repository revision and frozen-envelope identity.

Removing a native barostat changes the System. Do not promise binary compatibility with old boosted-NPT checkpoints. Strict resume must reject an incompatible legacy checkpoint with a specific explanation. A separately requested migration can extract positions, velocities and box using the legacy System and start a new, explicitly identified equilibration/production segment. Historical frames must not be relabeled as corrected samples or repaired with a simple pV reweighting.

## 9. Implementation work packages

| Area | Planned work |
| --- | --- |
| New `gareus/npt.py` | Target snapshot/evaluator interface, molecule proposal, acceptance transaction, RNG/state serialization and controller. |
| `gareus/pep_gamd.py` | Stage-aware supported boost adapter; explicit separation of physical energy, auxiliary energy and boost inputs; verify consistency with actual propagated forces. |
| `gareus/system_setup.py` | Choose barostat ownership before constructing Contexts; physical-force and molecule preflight checks. |
| `gareus/production.py` | Integrate local stepping/report driver into calibration, reconnaissance, probes, production and resume; cache invalidation; checkpoint metadata. |
| `gareus/config.py`, `gareus/cli.py` | Backend and fixed-width options, validation, resolved configuration output. |
| Tests and documentation | Independent ensemble tests, scheduling/restart coverage, compatibility notes and measured overhead. |

Land internal work in that order, but do not expose corrected production NPT until the integration and acceptance gates below pass. No change to MTS, HMR masses, timestep, replica count or exchange/output strides is required by this design.

## 10. Acceptance gates

**Independent acceptance mathematics.** Verify pressure units, reverse/forward log ratios, zero-boost limit and molecule Jacobian against hand-derived cases. A noninteracting molecular volume sampler has `p(V) ∝ V^Nmol exp(-beta P V)`, hence Gamma shape Nmol+1 and scale 1/(beta P), for unbounded positive V. Use independent quadrature or the corresponding truncated distribution if imposing bounds. Include rigid multi-atom molecules to make an accidental atom count fail.

**Boost and force agreement.** On frozen representative configurations, compare independently computed boost energies with the integrator's channel definitions and finite-difference forces, away from branch boundaries. Cover lambda zero, intermediate lambda, lambda one, active/inactive channels, dependent ordering, and stage transitions. A test must detect double application of lambda. Do not use the same helper as both implementation and oracle.

**Coupled volume/energy target.** Use a small analytic model with a volume-dependent bias and independent reference integration. Deliberately wrong variants that omit the boost or add the auxiliary energy must fail the ensemble test; the latter must fail even at lambda zero. This tests the failure being fixed, rather than only sampling stable-looking volumes.

**Real OpenMM behavior.** Compare the application-controlled zero-boost sampler with a conventional physical-System native-barostat reference. Include PME water/ions, a constrained peptide, restraints and virtual sites where supported. Verify trial energy components and volume-dependent corrections against independent evaluations. Establish small-timestep agreement before the 4 fs comparison. Use independent seeds and uncertainty estimates accounting for autocorrelation; predeclare tolerances and sample lengths before evaluating results.

**Transactions and scheduling.** Accepted/rejected/invalid/error moves must preserve the specified state. Check constraints, velocities, time, step count and restored energies within declared platform tolerances. Exercise 500/250/100 and mutually incommensurate schedules; compare frame, scalar, label and checkpoint phase. No hidden extra MD steps, stale exchange cache or duplicate volume controller is allowed.

**Resume and ownership.** Verify uninterrupted versus checkpoint/resumed proposal streams and event sequences. Use deterministic platforms for strict comparisons; do not require cross-platform CUDA bitwise identity. Extend replica-affinity tests to volume moves and reports. Verify all calibration paths and disabled/NVT paths.

**Production validation conditions.** Freeze envelopes and sampling-state definitions; disable noncanonical stuck-replica rescue and other adaptation when testing equilibrium distributions. This barostat correction does not validate those separate mechanisms. Report timestep sensitivity separately from volume-move correctness.

**Performance.** Benchmark 64 and 128 replicas on four L40S GPUs with approximately 30k atoms each, the existing 192-core budget and requested strides. Report aggregate throughput, time spent in volume trials/energy reads/transfers/reporting, acceptance and volume autocorrelation. Compare controlled on/off runs for overhead, but do not treat the existing incorrect boosted-NPT sampler as an equilibrium reference. No speedup claim or arbitrary overhead guarantee is made before measurement.

## 11. Completion definition and present status

The fix is complete when every supported boosted-NPT path uses the validated target, the mathematical and OpenMM gates pass, output/restart semantics are consistent, and measured overhead is documented. Unsupported modes must fail explicitly. NVT and conventional MD behavior must remain covered.

Present status: specification only. Repository source and upstream implementation were inspected. OpenMM is unavailable in this review runtime, so no molecular-dynamics tests or GPU benchmarks have been run. Implementation has not begun.

## Source references

- [Reviewed Pep-GaMD implementation](https://github.com/sulcjo/ATLaS-MD/blob/a792cee281cef87af798102665549c91ce47919f/gareus/pep_gamd.py)
- [Reviewed production implementation](https://github.com/sulcjo/ATLaS-MD/blob/a792cee281cef87af798102665549c91ce47919f/gareus/production.py)
- [Reviewed System construction](https://github.com/sulcjo/ATLaS-MD/blob/a792cee281cef87af798102665549c91ce47919f/gareus/system_setup.py)
- [Upstream GaMD stage integrator](https://github.com/MiaoLab20/gamd-openmm/blob/master/gamd/stage_integrator.py) and [Langevin base](https://github.com/MiaoLab20/gamd-openmm/blob/master/gamd/langevin/base_integrator.py)
- [OpenMM MC barostat source](https://github.com/openmm/openmm/blob/master/openmmapi/src/MonteCarloBarostatImpl.cpp) and [theory](https://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#montecarlobarostat)

Repository findings are pinned to the revision above. Upstream references describe the sources inspected; implementation must pin and test the actual installed OpenMM and gamd-openmm versions.
