"""Production-side stepping/report driver for the NPT correction.

The frozen barostat contract lives in :mod:`gareus.npt` (controller internals
owned by that package).  This module owns everything *around* the controller on
the production side:

* :class:`ReplicaStepDriver` -- the per-replica local stepping driver.  It
  subdivides an MD chunk at the replica's own barostat and reporter deadlines
  and enforces the spec's coincident-endpoint order:

      integrate -> finish due volume move -> emit due reports of that SAME
      post-move state

  Attached ``Simulation`` reporters fire inside ``Simulation.step()``, which is
  exactly the defect this driver exists to remove: with a volume move appended
  after ``step()`` the same step number would carry a pre-move trajectory frame
  and a post-move scalar record.  Reporters are therefore *detached* from
  ``sim.reporters`` and registered here instead; the driver calls the public
  ``describeNextReport()`` to learn each reporter's interval/requirements,
  advances with ``integrator.step()`` (so no automatic reporter dispatch can
  fire), performs the due volume move, builds a FRESH ``State`` and calls the
  public ``report(simulation, state)``.

* :class:`NptRunContext` -- the run-wide bundle (resolved ownership, adapter,
  seeds) shared by the five production-side paths that advance boosted
  contexts: shared GaMD setup, multi-window reconnaissance, fresh production,
  resumed production and the validation/probe path.

Scheduling uses the controller's persisted next-due INTEGRATION step
(``next_due_step``), never a report or exchange count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import gareus.npt as npt

__all__ = [
    "ReplicaStepDriver",
    "NptRunContext",
    "NPT_REPORT_PHASE_NOTE",
    "NPT_CHECKPOINT_PHASE_NOTE",
    "describe_reporter",
]

# Phase notes recorded in run metadata so downstream consumers know exactly
# what a step-stamped artifact saw (spec section 7: for the 500/250/100
# strides, the step-500 frame is post-volume-move and pre-exchange; the
# checkpoint is post-exchange).
NPT_REPORT_PHASE_NOTE = (
    "trajectory frames and scalar samples stamped with step S reflect the state "
    "AFTER any volume move due at S and BEFORE any label exchange at S"
)
NPT_CHECKPOINT_PHASE_NOTE = (
    "checkpoints stamped with step S are saved AFTER the label exchange at S"
)

# getState()-legal State property flags.  A reporter's ``include`` description
# may be a dict {name: bool} (OpenMM >= 8.1 for some reporters) or a list of
# names (installed OpenMM 8.5.1's DCD/XTC reporters); anything outside this set
# is not a State property and is ignored, matching Simulation._generate_reports
# which passes the names straight through to getState().
_STATE_INCLUDE_KEYS = ("positions", "velocities", "forces", "energy")


def describe_reporter(reporter, sim) -> dict:
    """Call the PUBLIC ``describeNextReport`` and normalize its description.

    Raises ``TypeError`` for reporters still using the pre-8.0 tuple protocol:
    only reporters describing themselves in the installed OpenMM's dict form
    are supported by the driver, and an unsupported one must fail loudly
    rather than silently stop producing frames.
    """
    desc = reporter.describeNextReport(sim)
    if not isinstance(desc, dict) or "steps" not in desc:
        raise TypeError(
            f"reporter {reporter!r} describeNextReport() did not return the dict "
            "form {steps, periodic, include} required by the NPT stepping driver; "
            "it cannot be scheduled post-volume-move"
        )
    return desc


def _include_kwargs(include) -> dict:
    """Convert an ``include`` description into getState() keyword arguments."""
    kwargs: dict[str, bool] = {}
    if not include:
        return kwargs
    if isinstance(include, dict):
        items = include.items()
    else:
        items = ((name, True) for name in include)
    for name, flag in items:
        if name in _STATE_INCLUDE_KEYS and flag:
            kwargs[str(name)] = True
    return kwargs


class ReplicaStepDriver:
    """One replica's local stepping/report driver.

    Owns a single OpenMM ``Simulation``, its optional
    :class:`gareus.npt.BiasedMCBarostatController`, and every reporter
    registered with it.  ``advance(nsteps)`` integrates exactly ``nsteps`` MD
    steps, subdividing at the replica's own deadlines:

    * a volume move whenever ``controller.next_due_step`` falls inside the
      chunk (attempted on whatever thread called ``advance`` -- production
      always dispatches this per replica onto its context-owning worker);
    * a report whenever a registered reporter's absolute due step falls
      inside the chunk.

    At a coincident deadline the order is: integrate to the deadline, finish
    the due volume move, then emit the due reports of that same post-move
    state.  A volume move never advances the integration step count, so the
    number of MD steps actually integrated is exactly ``nsteps``.

    ``on_volume_move`` (if given) is invoked after EVERY attempted move --
    accepted or rejected -- so callers can conservatively invalidate
    configuration-dependent observation caches.
    """

    def __init__(
        self,
        sim,
        controller: Any = None,
        on_volume_move: Optional[Callable[[Any], None]] = None,
        label: str = "",
    ):
        self.sim = sim
        self.controller = controller
        self.on_volume_move = on_volume_move
        self.label = str(label)
        self._reporters: list[dict] = []

    # ------------------------------------------------------------------ state

    @property
    def current_step(self) -> int:
        return int(self.sim.currentStep)

    @property
    def n_registered_reporters(self) -> int:
        return len(self._reporters)

    # -------------------------------------------------------------- reporters

    def register_reporter(self, reporter) -> None:
        """Detach-and-register: the driver becomes this reporter's scheduler.

        The reporter must NOT also sit on ``sim.reporters`` -- Simulation would
        fire it inside ``step()`` (pre-volume-move) and the frames would be
        emitted twice.
        """
        if any(entry["reporter"] is reporter for entry in self._reporters):
            return
        desc = describe_reporter(reporter, self.sim)
        steps = int(desc.get("steps", 0) or 0)
        due = self.current_step + (steps if steps > 0 else 0)
        self._reporters.append({"reporter": reporter, "due": due})

    def _emit_report(self, entry: dict) -> None:
        reporter = entry["reporter"]
        # Describe at the emission step: the include/periodic requirements of
        # this report, then a fresh post-report State for the callback.
        desc = describe_reporter(reporter, self.sim)
        periodic = desc.get("periodic")
        if periodic is None:
            periodic = bool(self.sim.system.usesPeriodicBoundaryConditions())
        include_kwargs = _include_kwargs(desc.get("include"))
        groups = self.sim.context.getIntegrator().getIntegrationForceGroups()
        state = self.sim.context.getState(
            groups=groups,
            enforcePeriodicBox=bool(periodic),
            parameters=True,
            **include_kwargs,
        )
        reporter.report(self.sim, state)
        nxt = describe_reporter(reporter, self.sim)
        steps = int(nxt.get("steps", 0) or 0)
        # max(1, ...) so a reporter that keeps describing itself as due "now"
        # cannot livelock the driver.
        entry["due"] = self.current_step + max(1, steps)

    # ---------------------------------------------------------------- driver

    def _next_controller_due(self) -> Optional[int]:
        if self.controller is None:
            return None
        return int(self.controller.next_due_step)

    def advance(self, nsteps: int, *, reports: bool = True) -> None:
        """Advance exactly ``nsteps`` MD steps, servicing local deadlines.

        ``reports=False`` is used by rollback probes: the context (and any
        volume moves attempted) is rolled back afterwards, so trajectory
        frames of the transient probe states must not be emitted.
        """
        nsteps = int(nsteps)
        if nsteps <= 0:
            return
        end = self.current_step + nsteps
        while True:
            cur = self.current_step
            if cur >= end:
                return
            target = end
            ctrl_due = self._next_controller_due()
            if ctrl_due is not None and ctrl_due <= end:
                target = min(target, ctrl_due)
            if reports:
                for entry in self._reporters:
                    if entry["due"] <= end:
                        target = min(target, int(entry["due"]))
            if target < cur:
                # A deadline in the past (e.g. a restored schedule) is
                # serviced now rather than blocking progress.
                target = cur
            delta = target - cur
            if delta > 0:
                self.sim.integrator.step(delta)
            now = self.current_step
            # 1) finish the due volume move ...
            if ctrl_due is not None and int(self.controller.next_due_step) <= now:
                result = self.controller.attempt_due(now)
                if int(self.controller.next_due_step) <= now:
                    raise RuntimeError(
                        "BiasedMCBarostatController.attempt_due() did not advance "
                        f"next_due_step past {now} (still {self.controller.next_due_step}); "
                        "the NPT stepping driver refuses to loop on a stuck schedule"
                    )
                if self.on_volume_move is not None:
                    self.on_volume_move(result)
            # 2) ... then reports of that SAME post-move state.
            if reports:
                for entry in self._reporters:
                    if entry["due"] <= now:
                        self._emit_report(entry)
            if delta <= 0:
                # No MD progress was made this iteration; without a deadline
                # firing this cannot make progress and must not spin.
                progressed = (
                    self._next_controller_due() is not None
                    or any(entry["due"] > now for entry in self._reporters)
                )
                if not progressed:
                    return


@dataclass
class _OwnershipView:
    """The subset of system_setup.BarostatOwnership the driver needs.

    Kept structural (not imported) so the driver module has no dependency on
    system_setup and can be unit-tested with plain fakes.
    """

    backend: str
    pressure_bar: float
    temperature_k: float
    barostat_frequency: int
    volume_step_fraction: float


# Disjoint seed ranges so no two controllers in one campaign ever share a
# random stream (spec section 6: "never clone a calibration RNG stream into
# every replica").  All offsets are added to the user's --seed.
_SHARED_SETUP_SEED_OFFSET = 61_001
_RECON_WINDOW_SEED_OFFSET = 62_000
_REPLICA_SEED_OFFSET = 63_000
_CHECK_RUN_SEED_OFFSET = 64_001


def npt_seed_shared_setup(args) -> int:
    return int(getattr(args, "seed", 0) or 0) + _SHARED_SETUP_SEED_OFFSET


def npt_seed_recon_window(args, window_index: int) -> int:
    return int(getattr(args, "seed", 0) or 0) + _RECON_WINDOW_SEED_OFFSET + int(window_index)


def npt_seed_replica(args, replica_index: int) -> int:
    return int(getattr(args, "seed", 0) or 0) + _REPLICA_SEED_OFFSET + int(replica_index)


def npt_seed_check_run(args) -> int:
    return int(getattr(args, "seed", 0) or 0) + _CHECK_RUN_SEED_OFFSET


class NptRunContext:
    """Run-wide NPT bundle shared by every path that advances a boosted context.

    Built once in ``run_gareus`` (and once per swarm/system build) right after
    the barostat ownership is resolved and before the production base system
    is created.  ``ownership`` is any object exposing the
    :class:`_OwnershipView` fields (system_setup's ``BarostatOwnership``).
    """

    def __init__(self, ownership, adapter: Any = None):
        self.ownership = ownership
        self.adapter = adapter

    @property
    def backend(self) -> str:
        return str(self.ownership.backend)

    @property
    def needs_controller(self) -> bool:
        return self.backend == "biased_mc"

    def initialize_controller(self, context, seed: int):
        if not self.needs_controller:
            raise RuntimeError(
                f"NptRunContext.initialize_controller called for backend {self.backend!r}; "
                "only biased_mc runs own a BiasedMCBarostatController"
            )
        if self.adapter is None:
            raise RuntimeError(
                "biased_mc NPT requires the stage-aware effective-potential adapter; "
                "none was provided"
            )
        # A lazily-resolved adapter (production._LazyNptAdapter) builds its
        # real self here: the controller's state_dict must carry the real
        # adapter_id from the very first checkpoint, not the empty pre-build
        # placeholder.
        _ensure = getattr(self.adapter, "ensure_built", None)
        if _ensure is not None:
            _ensure(context)
        return npt.BiasedMCBarostatController.initialize(
            context=context,
            adapter=self.adapter,
            pressure_bar=float(self.ownership.pressure_bar),
            temperature_k=float(self.ownership.temperature_k),
            frequency_steps=int(self.ownership.barostat_frequency),
            volume_step_fraction=float(self.ownership.volume_step_fraction),
            seed=int(seed),
        )

    def restore_controller(self, context, state):
        if not self.needs_controller:
            raise RuntimeError(
                f"NptRunContext.restore_controller called for backend {self.backend!r}"
            )
        # Build the lazily-resolved adapter before BiasedMCBarostatController
        # .restore compares its adapter_id against the checkpoint's.
        _ensure = getattr(self.adapter, "ensure_built", None)
        if _ensure is not None:
            _ensure(context)
        return npt.BiasedMCBarostatController.restore(
            context=context,
            adapter=self.adapter,
            state=state,
            expected_pressure_bar=float(self.ownership.pressure_bar),
            expected_temperature_k=float(self.ownership.temperature_k),
        )

    def make_driver(
        self,
        sim,
        controller: Any = None,
        on_volume_move: Optional[Callable[[Any], None]] = None,
        label: str = "",
    ) -> ReplicaStepDriver:
        """Uniform driver factory: every backend gets a driver, only
        biased_mc attaches a controller."""
        return ReplicaStepDriver(sim, controller=controller, on_volume_move=on_volume_move, label=label)

    def manifest_block(
        self,
        *,
        controllers: Optional[list] = None,
        n_atoms: Optional[list] = None,
    ) -> dict:
        """The checkpoint manifest's per-replica NPT block.

        ``controllers`` is the per-replica ``controller.state_dict()`` list
        captured on each replica's own worker together with its binary Context
        checkpoint.  The contract puts the molecule-partition fingerprint, RNG
        algorithm/state, fixed width, counters, last/next due step and
        controller schema version inside that dict.
        """
        block = {
            "backend": self.backend,
            "pressure_bar": float(self.ownership.pressure_bar),
            "temperature_k": float(self.ownership.temperature_k),
            "frequency_steps": int(self.ownership.barostat_frequency),
            "volume_step_fraction": float(self.ownership.volume_step_fraction),
            "adapter_ids": [str(getattr(self.adapter, "adapter_id", ""))] if self.adapter is not None else [],
            "report_phase": NPT_REPORT_PHASE_NOTE,
            "checkpoint_phase": NPT_CHECKPOINT_PHASE_NOTE,
        }
        if n_atoms is not None:
            block["n_atoms"] = [int(x) for x in n_atoms]
        if controllers is not None:
            block["controllers"] = controllers
        return block
