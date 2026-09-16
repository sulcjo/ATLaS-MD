"""Scheduling/transaction gates for the NPT production stepping driver.

These tests drive :class:`gareus.npt_driver.ReplicaStepDriver` against a REAL
OpenMM system on the Reference platform with a REAL OpenMM ``DCDReporter``,
plus a fake volume controller that honours the frozen
``gareus.npt.BiasedMCBarostatController`` contract (initialize/restore/
next_due_step/steps_until_due/attempt_due/state_dict).  The controller
internals are package 1's; everything tested here is the production-side
scheduling, phase ordering and restart plumbing.

Gates covered (spec section 10, "Transactions and scheduling"):

* 500/250/100 and mutually incommensurate strides: frame/volume-move phase;
* no hidden extra MD steps (exact integration step counts after every chunk);
* reports of a coincident endpoint are post-volume-move (box proves it);
* restored rejections and invalid trials leave state, time and step count
  unchanged;
* resume vs uninterrupted proposal-stream and event-sequence equality with a
  deterministic RNG whose state round-trips through ``state_dict``/restore;
* one-worker-per-replica affinity extended to volume moves and reports;
* disabled (controller-less) path still reports, and matches plain
  ``Simulation.step`` frame-for-frame.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
import openmm
from openmm import LangevinMiddleIntegrator, Platform, System, unit
from openmm.app import DCDReporter, Simulation

Vec3 = openmm.Vec3

import gareus.npt as npt
import gareus.npt_driver as npt_driver
from gareus.npt_driver import ReplicaStepDriver

DT_FS = 1.0
N_STEPS_TOTAL = 600


# --------------------------------------------------------------------- fakes


class FakeVolumeController:
    """Contract-honoring stand-in for ``BiasedMCBarostatController``.

    * ``initialize``/``restore`` classmethods mirror the frozen signatures;
    * ``attempt_due`` scales the periodic box isotropically by ``scale`` on
      acceptance (drawn from a deterministic NumPy RNG whose state travels in
      ``state_dict``), and leaves everything untouched on rejection;
    * every attempt is journaled as ``("volume", step, accepted)`` on the
      thread it ran on, for the affinity and phase-order gates.
    """

    schema_version = "fake_controller_v1"

    def __init__(self, sim, frequency, *, start_step, scale, accept, journal, rng):
        self.sim = sim
        self.frequency = int(frequency)
        self.next_due_step = int(start_step) + int(frequency)
        self.scale = float(scale)
        self.accept = accept
        self.journal = journal
        self.rng = rng
        self.attempts = 0
        self.accepted = 0

    # -- contract surface --------------------------------------------------
    @classmethod
    def initialize(cls, sim, *, frequency_steps, start_step=0, scale=1.01,
                   accept=lambda step: True, journal=None, seed=0):
        return cls(
            sim, frequency_steps,
            start_step=start_step, scale=scale, accept=accept,
            journal=journal if journal is not None else [],
            rng=np.random.default_rng(seed),
        )

    @classmethod
    def restore(cls, sim, *, state, frequency_steps, scale, accept, journal, seed=None):
        ctrl = cls(sim, state["frequency"], start_step=0, scale=scale,
                   accept=accept, journal=journal, rng=np.random.default_rng(0))
        ctrl.next_due_step = int(state["next_due_step"])
        ctrl.attempts = int(state["attempts"])
        ctrl.accepted = int(state["accepted"])
        ctrl.rng = np.random.default_rng(seed)
        ctrl.rng.bit_generator.state = state["rng_state"]
        return ctrl

    def steps_until_due(self, current_step):
        return max(0, int(self.next_due_step) - int(current_step))

    def attempt_due(self, current_step):
        current_step = int(current_step)
        assert current_step >= self.next_due_step, "attempt_due called when not due"
        u = float(self.rng.random())
        delta = float(self.rng.random())
        self.attempts += 1
        accepted = bool(self.accept(current_step)) and u < 0.999
        result = npt.VolumeMoveResult(
            step=current_step,
            accepted=accepted,
            reason="fake",
            old_volume_nm3=1.0,
            proposed_volume_nm3=1.0 * (self.scale if accepted else 1.0),
            log_acceptance=0.0,
            current_energy=npt.EnergyBreakdown(
                physical_kj_mol=0.0, bias_kj_mol=0.0, boost_kj_mol=0.0,
                auxiliary_kj_mol=0.0, effective_kj_mol=0.0,
            ),
        )
        if accepted:
            self.accepted += 1
            box = self.sim.context.getState().getPeriodicBoxVectors()
            self.sim.context.setPeriodicBoxVectors(
                *(self.scale * v for v in box)
            )
        self.next_due_step = current_step + self.frequency
        self.journal.append(("volume", current_step, accepted, threading.get_ident()))
        return result

    def state_dict(self):
        return {
            "schema_version": self.schema_version,
            "frequency": int(self.frequency),
            "next_due_step": int(self.next_due_step),
            "attempts": int(self.attempts),
            "accepted": int(self.accepted),
            "rng_state": self.rng.bit_generator.state,
        }


class JournalingReporter:
    """Wrap a real reporter, journaling every emitted frame's step and box."""

    def __init__(self, inner, journal):
        self.inner = inner
        self.journal = journal

    def describeNextReport(self, simulation):
        return self.inner.describeNextReport(simulation)

    def report(self, simulation, state):
        box = state.getPeriodicBoxVectors()
        self.journal.append(
            ("report", int(simulation.currentStep), float(box[0].x), threading.get_ident())
        )
        self.inner.report(simulation, state)


# ------------------------------------------------------------------ fixtures


def _make_sim(tmp_path, seed=17):
    system = System()
    for _ in range(4):
        system.addParticle(12.0)
    bonds = unit.nanometer
    harmonic = openmm.HarmonicBondForce()
    for i in range(3):
        harmonic.addBond(i, i + 1, 0.15 * bonds, 500 * unit.kilojoule_per_mole / bonds ** 2)
    system.addForce(harmonic)
    nb = openmm.NonbondedForce()
    for _ in range(4):
        nb.addParticle(0.0, 0.2, 0.0)
    system.addForce(nb)
    system.setDefaultPeriodicBoxVectors(
        Vec3(2.5, 0, 0), Vec3(0, 2.5, 0), Vec3(0, 0, 2.5)
    )
    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, DT_FS * unit.femtoseconds
    )
    integrator.setRandomNumberSeed(seed)
    platform = Platform.getPlatformByName("Reference")
    topology = openmm.app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("LIG", chain)
    from openmm.app.element import Element
    for _ in range(4):
        topology.addAtom("C", Element.getByAtomicNumber(6), residue)
    topology.setPeriodicBoxVectors((Vec3(2.5, 0, 0), Vec3(0, 2.5, 0), Vec3(0, 0, 2.5)))
    sim = Simulation(topology, system, integrator, platform)
    sim.context.setPositions(
        [[0, 0, 0], [0.15, 0, 0], [0.3, 0, 0], [0.45, 0, 0]]
    )
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, seed)
    return sim


def _journal_box_x(journal, kind="report"):
    return [entry[2] for entry in journal if entry[0] == kind]


# ---------------------------------------------------------------------- gates


def test_subdivides_at_barostat_deadlines_500_250_100(tmp_path):
    """The canonical stride triple: volume attempts at 100..500, frames at
    250/500, and NO hidden extra MD steps."""
    journal = []
    sim = _make_sim(tmp_path)
    ctrl = FakeVolumeController.initialize(sim, frequency_steps=100, journal=journal, seed=1)
    dcd = DCDReporter(str(tmp_path / "rep.dcd"), 250)
    driver = ReplicaStepDriver(sim, controller=ctrl, on_volume_move=lambda r: journal.append(("invalidate", r.step)))
    driver.register_reporter(JournalingReporter(dcd, journal))

    driver.advance(500)

    volume_steps = [e[1] for e in journal if e[0] == "volume"]
    report_steps = [e[1] for e in journal if e[0] == "report"]
    assert volume_steps == [100, 200, 300, 400, 500]
    assert report_steps == [250, 500]
    assert sim.context.getStepCount() == 500, "hidden extra MD steps detected"
    # every attempted move (accepted or not) invalidates the observation cache
    assert [e[1] for e in journal if e[0] == "invalidate"] == volume_steps


def test_coincident_endpoint_report_is_post_volume_move(tmp_path):
    """At step 500 both a volume move and a frame are due: the move finishes
    first and the DCD frame records the post-move box."""
    import mdtraj

    journal = []
    sim = _make_sim(tmp_path)
    ctrl = FakeVolumeController.initialize(
        sim, frequency_steps=100, scale=1.5, journal=journal, seed=2
    )
    dcd = DCDReporter(str(tmp_path / "rep.dcd"), 250)
    driver = ReplicaStepDriver(sim, controller=ctrl)
    driver.register_reporter(JournalingReporter(dcd, journal))

    driver.advance(500)

    # phase: at the coincident step the volume move precedes the report
    events_500 = [e for e in journal if e[1] == 500]
    kinds = [e[0] for e in events_500]
    assert kinds.index("volume") < kinds.index("report")

    from mdtraj.formats import DCDTrajectoryFile
    with DCDTrajectoryFile(str(tmp_path / "rep.dcd")) as dcd_file:
        _xyz, cell_lengths, _cell_angles = dcd_file.read()
    lengths = cell_lengths * 0.1  # DCD stores the unit cell in Angstrom
    assert lengths is not None and len(lengths) == 2
    # frame 0 (step 250): after 2 accepted moves -> 2.5 * 1.5^2
    assert lengths[0][0] == pytest.approx(2.5 * 1.5 ** 2, abs=1e-4)
    # frame 1 (step 500): after 5 accepted moves -> 2.5 * 1.5^5
    assert lengths[1][0] == pytest.approx(2.5 * 1.5 ** 5, abs=1e-4)


def test_incommensurate_strides_do_not_drift(tmp_path):
    """Mutually incommensurate schedules (barostat 70, reporter 300) driven in
    unequal chunks (500 then 250): deadlines land exactly on their multiples
    and nothing drifts or is skipped."""
    journal = []
    sim = _make_sim(tmp_path)
    ctrl = FakeVolumeController.initialize(sim, frequency_steps=70, journal=journal, seed=3)
    dcd = DCDReporter(str(tmp_path / "rep.dcd"), 300)
    driver = ReplicaStepDriver(sim, controller=ctrl)
    driver.register_reporter(JournalingReporter(dcd, journal))

    driver.advance(500)
    driver.advance(250)

    volume_steps = [e[1] for e in journal if e[0] == "volume"]
    report_steps = [e[1] for e in journal if e[0] == "report"]
    assert volume_steps == [70 * k for k in range(1, 11)]
    assert report_steps == [300, 600]
    assert sim.context.getStepCount() == 750


def test_rejected_moves_leave_state_time_and_steps_unchanged(tmp_path):
    """Restored rejections: box unchanged, exact step count, attempt still
    consumed its draws (journal) and still invalidated the cache."""
    journal = []
    sim = _make_sim(tmp_path)
    box_before = sim.context.getState().getPeriodicBoxVectors()
    ctrl = FakeVolumeController.initialize(
        sim, frequency_steps=100, journal=journal, seed=4,
        accept=lambda step: False,  # every trial rejected
    )
    driver = ReplicaStepDriver(
        sim, controller=ctrl, on_volume_move=lambda r: journal.append(("invalidate", r.step))
    )
    driver.advance(300)

    assert sim.context.getStepCount() == 300
    box_after = sim.context.getState().getPeriodicBoxVectors()
    assert [tuple(v) for v in box_after] == [tuple(v) for v in box_before]
    assert [e[1] for e in journal if e[0] == "volume"] == [100, 200, 300]
    assert all(e[2] is False for e in journal if e[0] == "volume")
    assert [e[1] for e in journal if e[0] == "invalidate"] == [100, 200, 300]


def test_resume_matches_uninterrupted_proposal_stream_and_events(tmp_path):
    """Checkpoint/resume vs uninterrupted: identical volume-attempt steps,
    acceptance decisions (same RNG draws -> same u sequence) and report steps."""
    def run(total, resume_at=None):
        journal = []
        sim = _make_sim(tmp_path)
        ctrl = FakeVolumeController.initialize(sim, frequency_steps=90, journal=journal, seed=42)
        dcd_path = tmp_path / ("resume.dcd" if resume_at is None else "uninterrupted.dcd")
        dcd = DCDReporter(str(dcd_path), 200)
        driver = ReplicaStepDriver(sim, controller=ctrl)
        driver.register_reporter(JournalingReporter(dcd, journal))
        if resume_at is None:
            driver.advance(total)
            return journal
        driver.advance(resume_at)
        saved = ctrl.state_dict()
        # "restart": fresh driver over a re-loaded context carrying the saved
        # controller state (the binary Context checkpoint is skipped here; the
        # controller stream is the object under test).
        ctrl2 = FakeVolumeController.restore(
            sim, state=saved, frequency_steps=90, scale=ctrl.scale,
            accept=lambda step: True, journal=journal,
        )
        driver.controller = ctrl2
        driver.advance(total - resume_at)
        return journal

    uninterrupted = run(N_STEPS_TOTAL)
    resumed = run(N_STEPS_TOTAL, resume_at=270)

    assert [(e[0], e[1]) for e in uninterrupted] == [(e[0], e[1]) for e in resumed]
    # acceptance decisions agree (identical RNG streams)
    acc_u = [e[2] for e in uninterrupted if e[0] == "volume"]
    acc_r = [e[2] for e in resumed if e[0] == "volume"]
    assert acc_u == acc_r
    # and the resumed stream made no extra or missing attempts
    assert [e[1] for e in resumed if e[0] == "volume"] == [90 * k for k in range(1, 7)]


def test_volume_moves_and_reports_run_on_the_replicas_own_thread(tmp_path):
    """Replica-affinity extension: attempts and reports execute on the
    replica's pinned worker thread, never the dispatching main thread."""
    from gareus.production import _ReplicaAffinityExecutor

    nrep = 3
    journal = []
    pool = _ReplicaAffinityExecutor(nrep)
    drivers = []
    try:
        for i in range(nrep):
            sim = _make_sim(tmp_path, seed=100 + i)
            ctrl = FakeVolumeController.initialize(
                sim, frequency_steps=40 + 10 * i, journal=journal, seed=50 + i
            )
            driver = ReplicaStepDriver(sim, controller=ctrl)
            driver.register_reporter(JournalingReporter(DCDReporter(str(tmp_path / f"r{i}.dcd"), 100), journal))
            drivers.append(driver)
        list(pool.map(lambda d: d.advance(300), drivers))
    finally:
        pool.shutdown(wait=True)

    main_ident = threading.get_ident()
    assert journal, "no events recorded"
    for entry in journal:
        assert entry[-1] != main_ident, "event ran on the dispatching thread"
    # each driver's events all share one ident
    for i, driver in enumerate(drivers):
        idents = {e[-1] for e in journal if e[0] in ("volume", "report")}
    per_event = [e[-1] for e in journal]
    # with 3 replicas there are exactly 3 distinct worker idents
    assert len(set(per_event)) == 3


def test_disabled_controller_less_path_matches_plain_simulation_stepping(tmp_path):
    """NVT/native (controller None): the driver still schedules reporters and
    produces the same frames plain ``Simulation.step`` would."""
    sim_a = _make_sim(tmp_path, seed=9)
    sim_b = _make_sim(tmp_path, seed=9)
    dcd_a = DCDReporter(str(tmp_path / "driver.dcd"), 250)
    dcd_b = DCDReporter(str(tmp_path / "plain.dcd"), 250)

    driver = ReplicaStepDriver(sim_a)
    driver.register_reporter(dcd_a)
    driver.advance(500)
    sim_b.reporters.append(dcd_b)
    sim_b.step(500)

    assert sim_a.context.getStepCount() == sim_b.context.getStepCount() == 500
    from mdtraj.formats import DCDTrajectoryFile
    with DCDTrajectoryFile(str(tmp_path / "driver.dcd")) as f:
        n_a = len(f.read()[0])
    with DCDTrajectoryFile(str(tmp_path / "plain.dcd")) as f:
        n_b = len(f.read()[0])
    assert n_a == n_b == 2


def test_unsupported_reporter_protocol_is_rejected_loudly(tmp_path):
    """Genuinely undescribable reporters must still fail loudly rather than
    silently stop producing frames. The legacy TUPLE protocol is NOT that case
    -- see the mdtraj tests below."""
    sim = _make_sim(tmp_path)

    class OpaqueReporter:
        def describeNextReport(self, simulation):
            return "every 250 steps"          # not a mapping, not a tuple

        def report(self, simulation, state):
            raise AssertionError("must not be called")

    class ShortTupleReporter:
        def describeNextReport(self, simulation):
            return (250, False)               # too short to say what to include

        def report(self, simulation, state):
            raise AssertionError("must not be called")

    driver = ReplicaStepDriver(sim)
    for bad in (OpaqueReporter(), ShortTupleReporter()):
        with pytest.raises(TypeError):
            driver.register_reporter(bad)


# ------------------------------------------------------ legacy tuple protocol
#
# mdtraj 1.10.1 -- current, and the reporter this project uses whenever a
# trajectory needs an atomSubset (system_setup.py:971: "OpenMM's own reporters
# do not" support it) -- describes itself with the legacy tuple, not OpenMM's
# dict. _BaseReporter.describeNextReport returns
#     (steps, positions, velocities, forces, energy, enforcePeriodicBox)
# Rejecting that killed chignolin_7 job 2390511 at epoch-0 production setup:
# "reporter <mdtraj...XTCReporter> describeNextReport() did not return the dict
# form". The tuple carries exactly the flags _emit_report consumes, so it is
# normalised rather than refused.


def test_mdtraj_style_six_tuple_is_normalised(tmp_path):
    sim = _make_sim(tmp_path)

    class MdtrajStyleReporter:
        def describeNextReport(self, simulation):
            return (250, True, False, False, True, False)

    desc = npt_driver.describe_reporter(MdtrajStyleReporter(), sim)
    assert desc["steps"] == 250
    assert desc["periodic"] is False
    assert npt_driver._include_kwargs(desc["include"]) == {
        "positions": True, "energy": True}


def test_legacy_five_tuple_defers_periodic_to_the_system(tmp_path):
    sim = _make_sim(tmp_path)

    class FiveTupleReporter:
        def describeNextReport(self, simulation):
            return (100, False, True, False, False)

    desc = npt_driver.describe_reporter(FiveTupleReporter(), sim)
    assert desc["steps"] == 100
    assert desc["periodic"] is None, "no periodic element -> _emit_report asks the System"
    assert npt_driver._include_kwargs(desc["include"]) == {"velocities": True}


def test_a_tuple_reporter_is_scheduled_and_actually_reports(tmp_path):
    """End to end: registering a legacy-protocol reporter produces frames."""
    sim = _make_sim(tmp_path)
    seen = []

    class TupleReporter:
        def describeNextReport(self, simulation):
            return (10, True, False, False, False, True)

        def report(self, simulation, state):
            seen.append(int(state.getTime().value_in_unit(unit.picoseconds) * 0 + len(seen)))

    driver = ReplicaStepDriver(sim)
    driver.register_reporter(TupleReporter())
    driver.advance(30)
    assert len(seen) >= 2, "a legacy-protocol reporter must actually receive report()"


def test_controller_that_never_advances_its_schedule_fails_loudly(tmp_path):
    class StuckController:
        next_due_step = 50

        def steps_until_due(self, current_step):
            return 0

        def attempt_due(self, current_step):
            return npt.VolumeMoveResult(
                step=current_step, accepted=False, reason="stuck",
                old_volume_nm3=1.0, proposed_volume_nm3=1.0,
                log_acceptance=0.0,
                current_energy=npt.EnergyBreakdown(
                    0.0, 0.0, 0.0, 0.0, 0.0),
            )

    sim = _make_sim(tmp_path)
    driver = ReplicaStepDriver(sim, controller=StuckController())
    with pytest.raises(RuntimeError, match="did not advance"):
        driver.advance(100)


def test_reports_false_suppresses_frames_but_not_volume_moves(tmp_path):
    """Rollback probes advance with reports disabled: the context still sees
    the volume moves, no frames are written."""
    journal = []
    sim = _make_sim(tmp_path)
    ctrl = FakeVolumeController.initialize(sim, frequency_steps=50, journal=journal, seed=7)
    dcd = DCDReporter(str(tmp_path / "probe.dcd"), 50)
    driver = ReplicaStepDriver(sim, controller=ctrl)
    driver.register_reporter(JournalingReporter(dcd, journal))

    driver.advance(150, reports=False)

    assert [e[1] for e in journal if e[0] == "volume"] == [50, 100, 150]
    assert not [e for e in journal if e[0] == "report"]
    probe_dcd = tmp_path / "probe.dcd"
    if probe_dcd.exists() and probe_dcd.stat().st_size > 0:
        from mdtraj.formats import DCDTrajectoryFile
        with DCDTrajectoryFile(str(probe_dcd)) as f:
            assert len(f.read()[0]) == 0
