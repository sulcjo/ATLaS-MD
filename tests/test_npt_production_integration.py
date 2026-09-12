"""Production-integration gates for the NPT correction (package B).

These tests exercise the production.py wiring laid over the REAL
``gareus.npt.BiasedMCBarostatController`` from merged package A: checkpoint
save/load (controller state captured per replica on its own worker),
checkpoint compatibility validation (including the legacy boosted-NPT
rejection), the rollback probe, and the resolved-barostat metadata
presentation -- against REAL tiny OpenMM Contexts on the Reference platform
(fully deterministic: fixed seeds, fixed platform).

The only substituted piece is the *effective-potential adapter* (package 2's
half of the correction, not yet landed): a deterministic physical-energy
adapter stands in through the same ``NptRunContext`` seam production code
itself uses.  The adapter is deliberately inspectable so the transaction gates
below can force deterministic accept/reject outcomes.
"""

from __future__ import annotations

import json
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import openmm
from openmm import LangevinMiddleIntegrator, Platform, System, unit
from openmm.app import DCDReporter, Simulation

Vec3 = openmm.Vec3

import gareus.npt as npt
import gareus.production as production
from gareus.npt_driver import NptRunContext, ReplicaStepDriver
from gareus.system_setup import BarostatOwnership
from gareus.windows import set_window


# --------------------------------------------------------------------- adapter


class PhysicalEnergyAdapter:
    """Deterministic stand-in for package 2's effective-potential adapter.

    U* is the raw Context potential over all force groups -- a valid physical
    energy for the tiny test System, which carries no auxiliary force.  The
    optional ``journal`` records the thread every energy evaluation ran on,
    which is how the replica-affinity gates observe real volume moves.
    """

    adapter_id = "physical_reference_v1"

    def __init__(self, journal=None):
        self.journal = journal

    def snapshot(self, context, integrator):
        return None

    def evaluate(self, context, snapshot):
        if self.journal is not None:
            self.journal.append(("evaluate", threading.get_ident()))
        pe = float(context.getState(getEnergy=True).getPotentialEnergy()
                   .value_in_unit(unit.kilojoule_per_mole))
        return npt.EnergyBreakdown(physical_kj_mol=pe, bias_kj_mol=0.0,
                                   boost_kj_mol=0.0, auxiliary_kj_mol=0.0,
                                   effective_kj_mol=pe)


class RejectAnyChangeAdapter(PhysicalEnergyAdapter):
    """U* = 0 at the starting volume, 1e12 kJ/mol anywhere else.

    Every proposal (expansion or contraction) is deterministically REJECTED:
    log A is dominated by -beta * 1e12.  Used to gate the rejection
    transaction invariants without relying on chance."""

    adapter_id = "reject_any_change_v1"

    def __init__(self, journal=None):
        super().__init__(journal)
        self.start_volume_nm3 = None

    def snapshot(self, context, integrator):
        # Capture the reference volume once per trial; both endpoints of the
        # trial must be evaluated under the SAME parameter snapshot.
        box = npt._box_matrix_nm(context)
        if self.start_volume_nm3 is None:
            self.start_volume_nm3 = npt._box_volume_nm3(box)
        return self.start_volume_nm3

    def evaluate(self, context, snapshot):
        if self.journal is not None:
            self.journal.append(("evaluate", threading.get_ident()))
        volume = npt._box_volume_nm3(npt._box_matrix_nm(context))
        eff = 0.0 if volume == snapshot else 1.0e12
        return npt.EnergyBreakdown(physical_kj_mol=eff, bias_kj_mol=0.0,
                                   boost_kj_mol=0.0, auxiliary_kj_mol=0.0,
                                   effective_kj_mol=eff)


class ExplodingAdapter(PhysicalEnergyAdapter):
    """Raises on the Nth evaluate call (default: the proposed endpoint of the
    first trial), exercising the real controller's restore-then-abort path."""

    adapter_id = "exploding_v1"

    def __init__(self, explode_at_call=2):
        super().__init__()
        self.calls = 0
        self.explode_at_call = int(explode_at_call)

    def evaluate(self, context, snapshot):
        self.calls += 1
        if self.calls >= self.explode_at_call:
            raise RuntimeError("boom: proposed-endpoint evaluation failed")
        return super().evaluate(context, snapshot)


# ------------------------------------------------------------------- fixtures


def _make_ownership(backend="biased_mc"):
    return BarostatOwnership(
        backend, requested="auto", ensemble="npt", run_mode="gamd",
        boost_type="pep-gamd-lower-dual", barostat_frequency=25,
        pressure_bar=1.0, temperature_k=300.0, volume_step_fraction=0.01,
    )


def _make_runtime(backend="biased_mc", journal=None):
    return NptRunContext(ownership=_make_ownership(backend),
                         adapter=PhysicalEnergyAdapter(journal))


def _make_sim(seed=17):
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
    # umbrella-shaped force so set_window()'s r0/k parameter writes are valid
    umbrella = openmm.CustomCompoundBondForce(2, "0.5*k*(distance(p1,p2)-r0)^2")
    umbrella.addGlobalParameter("k", 0.0)
    umbrella.addGlobalParameter("r0", 0.0)
    umbrella.addBond([0, 3], [])
    system.addForce(umbrella)
    system.setDefaultPeriodicBoxVectors(Vec3(2.5, 0, 0), Vec3(0, 2.5, 0), Vec3(0, 0, 2.5))
    integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 1.0 * unit.femtoseconds)
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
    sim.context.setPositions([[0, 0, 0], [0.15, 0, 0], [0.3, 0, 0], [0.45, 0, 0]])
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, seed)
    return sim


def _make_args(**over):
    ns = SimpleNamespace(
        production_probe_steps=60,
        production_probe_warn_only=False,
        temperature_k=300.0,
        seed=2026,
        gamd_boost_type="",
        run_mode="cmd",
        exchange_stats=None,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


class _NoPool:
    """pool stand-in: run everything inline on the calling thread."""

    def submit(self, r, fn, *a, **kw):
        fut = _InlineFuture(fn, a, kw)
        return fut

    def map(self, fn, items):
        return [fn(i) for i in items]


class _InlineFuture:
    def __init__(self, fn, args, kwargs):
        self._fn, self._args, self._kwargs = fn, args, kwargs

    def result(self):
        return self._fn(*self._args, **self._kwargs)


class JournalingReporter:
    """Wrap a real reporter, journaling every emitted frame's step and box."""

    def __init__(self, inner, journal):
        self.inner = inner
        self.journal = journal

    def describeNextReport(self, simulation):
        return self.inner.describeNextReport(simulation)

    def report(self, simulation, state):
        box = state.getPeriodicBoxVectors()
        self.journal.append(("report", int(simulation.currentStep),
                             float(box[0].x), threading.get_ident()))
        self.inner.report(simulation, state)


def _box_volume(sim) -> float:
    return npt._box_volume_nm3(npt._box_matrix_nm(sim.context))


# ------------------------------------------------- real-controller transactions


def test_rejected_volume_moves_preserve_state_and_consume_draws():
    """Spec section 5 rejection gate against the REAL controller: a rejected
    trial leaves positions, box, velocities, MD time and step count untouched,
    still consumes its random draws and still advances its schedule."""
    sim = _make_sim()
    adapter = RejectAnyChangeAdapter()
    ctrl = npt.BiasedMCBarostatController.initialize(
        sim.context, adapter, pressure_bar=1.0, temperature_k=300.0,
        frequency_steps=25, volume_step_fraction=0.01, seed=7)
    sim.integrator.step(25)

    pre = sim.context.getState(getPositions=True, getVelocities=True)
    pre_pos = np.asarray(pre.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    pre_vel = np.asarray(pre.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond))
    pre_box = [tuple(v) for v in sim.context.getState().getPeriodicBoxVectors()]
    pre_time_ps = sim.context.getState().getTime().value_in_unit(unit.picosecond)
    pre_rng = ctrl.state_dict()["rng"]

    res = ctrl.attempt_due(25)

    assert res.accepted is False
    post = sim.context.getState(getPositions=True, getVelocities=True)
    assert np.array_equal(np.asarray(post.getPositions(asNumpy=True).value_in_unit(unit.nanometer)),
                          pre_pos), "rejected trial moved positions"
    assert np.array_equal(np.asarray(post.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)),
                          pre_vel), "rejected trial changed velocities"
    assert [tuple(v) for v in sim.context.getState().getPeriodicBoxVectors()] == pre_box
    assert sim.context.getStepCount() == 25, "a rejected trial must not integrate"
    assert sim.context.getState().getTime().value_in_unit(unit.picosecond) == pre_time_ps
    counters = ctrl.state_dict()["counters"]
    assert counters["attempted"] == 1 and counters["rejected"] == 1 and counters["accepted"] == 0
    assert ctrl.next_due_step == 50, "rejection still advances the due schedule"
    assert ctrl.state_dict()["rng"] != pre_rng, "rejection must still consume the draws"


def test_accepted_volume_move_translates_the_molecule_and_preserves_geometry():
    """Spec section 4 gate against the REAL controller: an accepted move scales
    the box by exactly s=(V'/V)^(1/3), translates the whole molecule by
    (s-1)*Rm (internal geometry, velocities, time and step count unchanged)."""
    sim = _make_sim(seed=30)
    ctrl = npt.BiasedMCBarostatController.initialize(
        sim.context, PhysicalEnergyAdapter(), pressure_bar=0.0, temperature_k=300.0,
        frequency_steps=100, volume_step_fraction=0.01, seed=200)
    sim.integrator.step(100)

    pre = sim.context.getState(getPositions=True, getVelocities=True)
    pre_pos = np.asarray(pre.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    pre_vel = np.asarray(pre.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond))
    pre_box = npt._box_matrix_nm(sim.context)
    pre_time_ps = sim.context.getState().getTime().value_in_unit(unit.picosecond)

    res = ctrl.attempt_due(100)

    assert res.accepted, "the flat-adapter P=0 trial must accept deterministically"
    counters = ctrl.state_dict()["counters"]
    assert counters["accepted"] == 1
    assert sim.context.getStepCount() == 100, "a volume move must not advance MD time"
    assert sim.context.getState().getTime().value_in_unit(unit.picosecond) == pytest.approx(pre_time_ps, abs=1e-12)

    post = sim.context.getState(getPositions=True, getVelocities=True)
    post_pos = np.asarray(post.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    post_box = npt._box_matrix_nm(sim.context)
    s = (res.proposed_volume_nm3 / res.old_volume_nm3) ** (1.0 / 3.0)
    assert np.allclose(post_box, s * pre_box, rtol=0, atol=1e-12)
    # single molecule spanning all four particles: x' = x + (s-1)*mean(x)
    expected = pre_pos + (s - 1.0) * pre_pos.mean(axis=0)
    assert np.allclose(post_pos, expected, rtol=0, atol=1e-9)
    # internal geometry preserved (bond lengths unchanged)
    for i in range(3):
        d_pre = float(np.linalg.norm(pre_pos[i + 1] - pre_pos[i]))
        d_post = float(np.linalg.norm(post_pos[i + 1] - post_pos[i]))
        assert d_post == pytest.approx(d_pre, abs=1e-9)
    assert np.array_equal(np.asarray(post.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)),
                          pre_vel), "accepted move must not rethermalize velocities"


# ------------------------------------------------------------- 500/250/100 etc.


def test_canonical_strides_with_real_controller(tmp_path):
    """The user's stride triple (exchange 500 / output 250 / barostat 100)
    against the REAL controller: volume attempts at 100..500, frames at
    250/500, exact MD step count, and a cache-invalidation event after every
    attempted move (the no-stale-exchange-cache mechanism)."""
    journal = []
    sim = _make_sim()
    ctrl = npt.BiasedMCBarostatController.initialize(
        sim.context, PhysicalEnergyAdapter(), pressure_bar=1.0, temperature_k=300.0,
        frequency_steps=100, volume_step_fraction=0.01, seed=200)
    driver = ReplicaStepDriver(sim, controller=ctrl,
                               on_volume_move=lambda r: journal.append(("invalidate", r.step, r.accepted)))
    driver.register_reporter(JournalingReporter(DCDReporter(str(tmp_path / "rep.dcd"), 250), journal))

    driver.advance(500)

    assert [e[1] for e in journal if e[0] == "invalidate"] == [100, 200, 300, 400, 500]
    assert [e[1] for e in journal if e[0] == "report"] == [250, 500]
    assert sim.context.getStepCount() == 500, "hidden extra MD steps detected"
    assert ctrl.state_dict()["counters"]["attempted"] == 5
    assert ctrl.state_dict()["next_due_step"] == 600


def test_incommensurate_strides_with_real_controller_do_not_drift(tmp_path):
    """Mutually incommensurate strides (barostat 70, reporter 300) driven in
    unequal chunks (500 then 250): every deadline lands exactly on its own
    multiple, nothing drifts, no extra MD steps."""
    journal = []
    sim = _make_sim()
    ctrl = npt.BiasedMCBarostatController.initialize(
        sim.context, PhysicalEnergyAdapter(), pressure_bar=1.0, temperature_k=300.0,
        frequency_steps=70, volume_step_fraction=0.01, seed=201)
    driver = ReplicaStepDriver(sim, controller=ctrl,
                               on_volume_move=lambda r: journal.append(("invalidate", r.step, r.accepted)))
    driver.register_reporter(JournalingReporter(DCDReporter(str(tmp_path / "rep.dcd"), 300), journal))

    driver.advance(500)
    driver.advance(250)

    assert [e[1] for e in journal if e[0] == "invalidate"] == [70 * k for k in range(1, 11)]
    assert [e[1] for e in journal if e[0] == "report"] == [300, 600]
    assert sim.context.getStepCount() == 750


# ------------------------------------------------- coincident-endpoint phase gate


def test_coincident_endpoint_frame_scalar_label_checkpoint_phase(tmp_path):
    """Spec section 7 order gate at the production-function level, with the
    REAL controller and real DCD reporters.  At step 500 (exchange stride 500
    / output stride 250 / barostat 100), for each replica:

    1. the DCD frame stamped 500 records the POST-move box (all five volume
       moves accept deterministically here -- flat adapter, one molecule --
       so the frame's unit cell differs from the pre-move one);
    2. the scalar observation of the SAME post-move state agrees with that
       frame (both precede the label exchange);
    3. the label exchange swaps umbrella labels (set_window on both replicas);
    4. save_production_checkpoint then records the post-exchange assignments
       and the post-move volume-move schedule (next_due 600), and the
       manifest's phase notes pin this semantics down.

    Replicas are advanced SEQUENTIALLY (deterministic draw order on the
    Reference platform; see the round-trip test's comment) so the acceptance
    pattern can be asserted exactly rather than probabilistically.
    """
    nrep = 2
    journal = []
    sims, drivers = [], []
    for i in range(nrep):
        sim = _make_sim(seed=30 + i)
        ctrl = npt.BiasedMCBarostatController.initialize(
            sim.context, PhysicalEnergyAdapter(), pressure_bar=1.0, temperature_k=300.0,
            frequency_steps=100, volume_step_fraction=0.01, seed=200 + i)
        driver = ReplicaStepDriver(
            sim, controller=ctrl,
            on_volume_move=lambda r: journal.append(("invalidate", r.step, r.accepted)),
            label=f"replica_{i:03d}")
        driver.register_reporter(
            JournalingReporter(DCDReporter(str(tmp_path / f"replica_{i:03d}.dcd"), 250), journal))
        sims.append(sim)
        drivers.append(driver)

    assignments = [0, 1]
    pre_boxes = [npt._box_matrix_nm(sim.context) for sim in sims]
    for driver in drivers:  # sequential: deterministic Reference-platform draw order
        driver.advance(500)

    # (1)+(2): frame and scalar phases agree on the post-move state.
    from mdtraj.formats import DCDTrajectoryFile
    for i, driver in enumerate(drivers):
        assert driver.sim.context.getStepCount() == 500, "hidden extra MD steps detected"
        box_x = float(driver.sim.context.getState().getPeriodicBoxVectors()[0].x)
        # scalar observation of the post-move, pre-exchange state
        scalar_box_x = box_x
        with DCDTrajectoryFile(str(tmp_path / f"replica_{i:03d}.dcd")) as f:
            _xyz, cell_lengths, _cell_angles = f.read()
        assert len(cell_lengths) == 2
        frame_box_x = float(cell_lengths[1][0]) * 0.1  # DCD stores Angstrom
        assert frame_box_x == pytest.approx(scalar_box_x, abs=1e-4), (
            "the step-500 DCD frame must record the post-volume-move box"
        )
        counters = driver.controller.state_dict()["counters"]
        assert counters == {"attempted": 5, "accepted": 5, "rejected": 0,
                            "invalid_geometry": 0, "nonfinite_trial_energy": 0}
        assert frame_box_x != pytest.approx(float(pre_boxes[i][0][0]), abs=1e-6), (
            "vacuous phase gate: the step-500 move did not change the box"
        )

    # every attempted move invalidated the observation cache (no stale
    # exchange cache may survive a volume move)
    assert sorted(e[1] for e in journal if e[0] == "invalidate") == \
        sorted([100, 200, 300, 400, 500] * nrep)
    assert all(e[2] for e in journal if e[0] == "invalidate")

    # (3): label exchange at the coincident endpoint (labels, not boxes).
    assignments = [1, 0]
    centers = [np.array([0.4])] * 2
    ks = [10.0, 10.0]
    for i, driver in enumerate(drivers):
        set_window(driver.sim.context, centers, ks, assignments[i])

    # (4): the checkpoint is saved AFTER the exchange.
    production.save_production_checkpoint(
        tmp_path, sims, assignments, 500, 500, 0, 0, 1000, 500,
        {"attempts": 0, "accepted": 0, "pairs": {}}, np.random.default_rng(1),
        drivers=drivers, pool=None,
        npt_runtime=NptRunContext(ownership=_make_ownership(), adapter=PhysicalEnergyAdapter()),
    )
    manifest = json.loads(
        (tmp_path / "checkpoints" / "production_checkpoint_manifest.json").read_text()
    )
    assert manifest["assignments"] == [1, 0], "checkpoint must record post-exchange labels"
    npt_block = manifest["npt"]
    assert [c["next_due_step"] for c in npt_block["controllers"]] == [600, 600], (
        "checkpoint must record the post-move volume schedule"
    )
    assert [c["counters"]["attempted"] for c in npt_block["controllers"]] == [5, 5]
    assert "AFTER any volume move due at S" in npt_block["report_phase"]
    assert "AFTER the label exchange" in npt_block["checkpoint_phase"]


# ------------------------------------------------------------- checkpoints


def test_checkpoint_round_trip_restores_controller_streams(tmp_path):
    """save -> load through the production functions: the manifest carries the
    per-replica controller state, load restores it onto fresh drivers over
    FRESH Contexts, and the resumed proposal stream / event sequence / MD
    trajectory is EXACTLY the uninterrupted one.

    Platform note (measured on this machine, OpenMM 8.5.1 Reference platform):
    the stochastic-integrator noise draws come from one stream shared across
    every Reference-platform Context in the process, in step()-call
    granularity -- so two replicas stepping CONCURRENTLY interleave
    nondeterministically.  Binary checkpoints capture and restore that stream,
    which is what makes exact resume comparisons possible at all, but the
    comparison below therefore advances replicas SEQUENTIALLY (deterministic
    draw order), and runs the uninterrupted continuation BEFORE the fresh
    resumed Contexts are constructed (their construction also draws from the
    shared stream; the load then rewinds it to the checkpointed state).  The
    first advance and the checkpoint save/load still run through the real
    replica-affinity pool.
    """
    evaluate_journal: list = []
    npt_runtime = _make_runtime(journal=evaluate_journal)
    nrep = 2
    pool = production._ReplicaAffinityExecutor(nrep)
    sims, drivers, events = [], [], []
    try:
        for i in range(nrep):
            sim = _make_sim(seed=30 + i)
            ctrl = npt_runtime.initialize_controller(sim.context, seed=100 + i)
            ev = []
            drivers.append(ReplicaStepDriver(
                sim, controller=ctrl,
                on_volume_move=lambda r, _ev=ev: _ev.append((r.step, r.accepted, r.log_acceptance)),
                label=f"replica_{i:03d}"))
            events.append(ev)
            sims.append(sim)
        # advance so the controllers hold nonzero counters / RNG progress
        list(pool.map(lambda d: d.advance(60), drivers))
        saved_states = [d.controller.state_dict() for d in drivers]
        assert all(s["counters"]["attempted"] > 0 for s in saved_states)

        production.save_production_checkpoint(
            tmp_path, sims, list(range(nrep)), 60, 100, 0, 0, 70, 60,
            {"attempts": 0, "accepted": 0, "pairs": {}}, np.random.default_rng(1),
            drivers=drivers, pool=pool, npt_runtime=npt_runtime,
        )
        manifest = json.loads(
            (tmp_path / "checkpoints" / "production_checkpoint_manifest.json").read_text()
        )
        assert manifest["npt"]["backend"] == "biased_mc"
        assert manifest["npt"]["frequency_steps"] == 25
        assert len(manifest["npt"]["controllers"]) == nrep
        assert all(c is not None for c in manifest["npt"]["controllers"])
        assert all(c["schema_version"] == 1 for c in manifest["npt"]["controllers"])
        assert all(c["rng"]["algorithm"] == "PCG64" for c in manifest["npt"]["controllers"])
        assert all(c["molecule_partition_fingerprint"] for c in manifest["npt"]["controllers"])
        assert manifest["npt"]["n_atoms"] == [s.system.getNumParticles() for s in sims]
        assert manifest["npt"]["adapter_ids"] == ["physical_reference_v1"]
        n_tail = [len(ev) for ev in events]

        # affinity gate: every real volume-move energy evaluation ran on one
        # of the replicas' pinned workers, never on the dispatching thread
        worker_idents = {pool.submit(i, threading.get_ident).result() for i in range(nrep)}
        eval_idents = {e[1] for e in evaluate_journal}
        assert eval_idents, "no volume-move energy evaluations were recorded"
        assert eval_idents <= worker_idents
        assert threading.get_ident() not in eval_idents
        assert eval_idents == worker_idents

        # the UNINTERRUPTED continuation runs first (see the platform note):
        # same 40-step chunk the resumed side will run.  The load path applies
        # set_window to the restored replicas, so the uninterrupted side must
        # run the identical Hamiltonian.
        centers = [np.array([0.4])] * nrep
        ks = [10.0] * nrep
        for i, driver in enumerate(drivers):
            pool.submit(i, set_window, driver.sim.context, centers, ks, i).result()
        for i, driver in enumerate(drivers):
            pool.submit(i, driver.advance, 40).result()

        # fresh Contexts + fresh drivers (controller None), then load: the
        # restored controllers must continue the saved streams exactly.
        sims2 = [_make_sim(seed=30 + i) for i in range(nrep)]
        drivers2, events2 = [], []
        for i, sim in enumerate(sims2):
            ev = []
            drivers2.append(ReplicaStepDriver(
                sim, controller=None,
                on_volume_move=lambda r, _ev=ev: _ev.append((r.step, r.accepted, r.log_acceptance)),
                label=f"replica_{i:03d}"))
            events2.append(ev)
        rng2 = np.random.default_rng(2)
        loaded = production.load_production_checkpoint(
            tmp_path, sims2, centers, ks, rng2,
            drivers=drivers2, pool=pool, npt_runtime=npt_runtime, args=_make_args(),
        )
        assert loaded is not None
        for i, d in enumerate(drivers2):
            assert d.controller is not None
            got = d.controller.state_dict()
            assert got["counters"] == saved_states[i]["counters"]
            assert got["next_due_step"] == saved_states[i]["next_due_step"]
            assert got["rng"] == saved_states[i]["rng"]

        # ... and the resumed side replays the same 40 steps.
        for i, driver in enumerate(drivers2):
            pool.submit(i, driver.advance, 40).result()
        for i in range(nrep):
            assert events[i][n_tail[i]:] == events2[i], (
                f"replica {i}: resumed volume-move event sequence diverged from "
                "the uninterrupted one"
            )
            assert drivers[i].controller.next_due_step == drivers2[i].controller.next_due_step
            assert (drivers[i].controller.state_dict()["counters"]
                    == drivers2[i].controller.state_dict()["counters"])
            # the resumed MD trajectory itself is bit-identical on this platform
            pa = sims[i].context.getState(getPositions=True).getPositions(asNumpy=True)
            pb = sims2[i].context.getState(getPositions=True).getPositions(asNumpy=True)
            assert np.array_equal(np.asarray(pa), np.asarray(pb))
    finally:
        pool.shutdown(wait=True)


def test_biased_mc_checkpoint_refuses_without_controllers(tmp_path):
    npt_runtime = _make_runtime()
    sim = _make_sim()
    driver = ReplicaStepDriver(sim, controller=None, label="replica_000")
    with pytest.raises(RuntimeError, match="no replica carries a volume controller"):
        production.save_production_checkpoint(
            tmp_path, [sim], [0], 0, 0, 0, 0, 0, 0,
            {"attempts": 0}, np.random.default_rng(1),
            drivers=[driver], pool=None, npt_runtime=npt_runtime,
        )


def test_nvt_backend_checkpoint_round_trip_needs_no_controllers(tmp_path):
    """Disabled/NVT path: a backend-'none' runtime saves and loads checkpoints
    without any volume controller and never initializes one on resume."""
    npt_runtime = _make_runtime(backend="none")
    sim = _make_sim()
    driver = ReplicaStepDriver(sim, controller=None, label="replica_000")
    production.save_production_checkpoint(
        tmp_path, [sim], [0], 40, 40, 0, 0, 50, 40,
        {"attempts": 0}, np.random.default_rng(1),
        drivers=[driver], pool=None, npt_runtime=npt_runtime,
    )
    manifest = json.loads(
        (tmp_path / "checkpoints" / "production_checkpoint_manifest.json").read_text()
    )
    assert manifest["npt"]["backend"] == "none"
    loaded = production.load_production_checkpoint(
        tmp_path, [sim], [np.array([0.4])], [10.0], np.random.default_rng(2),
        drivers=[driver], pool=None, npt_runtime=npt_runtime, args=_make_args(),
    )
    assert loaded is not None
    assert driver.controller is None, "an NVT resume must not grow a volume controller"


# ------------------------------------------- checkpoint compatibility gates


def _write_legacy_run(tmp_path, *, ensemble, gamd):
    meta = {"production_ensemble": ensemble, "gamd_enabled": gamd}
    (tmp_path / "gareus_metadata.json").write_text(json.dumps(meta))


def test_legacy_boosted_npt_checkpoint_is_rejected_with_explanation(tmp_path):
    npt_runtime = _make_runtime()
    sim = _make_sim()
    _write_legacy_run(tmp_path, ensemble="npt", gamd=True)
    manifest = {"prod_done": 10}
    with pytest.raises(RuntimeError, match="LEGACY boosted-NPT"):
        production._validate_npt_checkpoint_compatibility(tmp_path, manifest, npt_runtime, [sim])


def test_legacy_nvt_checkpoint_fresh_inits_controllers(tmp_path):
    npt_runtime = _make_runtime()
    sim = _make_sim()
    _write_legacy_run(tmp_path, ensemble="nvt", gamd=True)
    mode = production._validate_npt_checkpoint_compatibility(tmp_path, {}, npt_runtime, [sim])
    assert mode == "fresh_init"


def test_backend_mismatch_between_checkpoint_and_run_is_rejected(tmp_path):
    npt_runtime = _make_runtime()
    sim = _make_sim()
    manifest = {"npt": {"backend": "native"}}
    with pytest.raises(RuntimeError, match="npt backend 'native'"):
        production._validate_npt_checkpoint_compatibility(tmp_path, manifest, npt_runtime, [sim])


def test_adapter_mismatch_is_rejected(tmp_path):
    npt_runtime = _make_runtime()
    sim = _make_sim()
    manifest = {"npt": {
        "backend": "biased_mc", "adapter_ids": ["some_other_adapter"],
        "controllers": [{}], "temperature_k": 300.0, "pressure_bar": 1.0,
        "n_atoms": [sim.system.getNumParticles()],
    }}
    with pytest.raises(RuntimeError, match="adapter id"):
        production._validate_npt_checkpoint_compatibility(tmp_path, manifest, npt_runtime, [sim])


def test_temperature_pressure_and_atom_mismatches_are_rejected(tmp_path):
    npt_runtime = _make_runtime()
    sim = _make_sim()
    base = {
        "backend": "biased_mc", "adapter_ids": ["physical_reference_v1"],
        "controllers": [{}], "n_atoms": [sim.system.getNumParticles()],
        "temperature_k": 300.0, "pressure_bar": 1.0,
    }
    for key, value, pattern in [
        ("temperature_k", 310.0, "temperature"),
        ("pressure_bar", 2.0, "pressure"),
        ("n_atoms", [999], "particles"),
    ]:
        manifest = {"npt": dict(base, **{key: value})}
        with pytest.raises(RuntimeError, match=pattern):
            production._validate_npt_checkpoint_compatibility(tmp_path, manifest, npt_runtime, [sim])


def test_current_generation_matching_manifest_restores(tmp_path):
    npt_runtime = _make_runtime()
    sim = _make_sim()
    manifest = {"npt": {
        "backend": "biased_mc", "adapter_ids": ["physical_reference_v1"],
        "controllers": [{"schema_version": 1}],
        "temperature_k": 300.0, "pressure_bar": 1.0,
        "n_atoms": [sim.system.getNumParticles()],
    }}
    assert production._validate_npt_checkpoint_compatibility(tmp_path, manifest, npt_runtime, [sim]) == "restored"


# -------------------------------------------------------------- probe gates


def test_probe_rolls_back_context_and_controller_state(tmp_path):
    npt_runtime = _make_runtime()
    sim = _make_sim()
    ctrl = npt_runtime.initialize_controller(sim.context, seed=7)
    driver = ReplicaStepDriver(sim, controller=ctrl, label="replica_000")
    args = _make_args(production_probe_steps=50)

    pre_state = sim.context.getState(getPositions=True)
    pre_positions = pre_state.getPositions(asNumpy=True)
    pre_ctrl = ctrl.state_dict()

    report = production.run_production_probe(
        args, tmp_path, [sim], [0], [np.array([0.4])], [10.0],
        {"mode": "distance", "cv_atom1": 0, "cv_atom2": 3}, 0, 3, unit,
        {}, drivers=[driver], pool=_NoPool(), npt_runtime=npt_runtime,
    )

    assert report["ok"], report.get("failures")
    # the probe report presents the PHYSICAL energy source explicitly
    assert report["replicas"][0]["potential_source"] == "physical_force_groups"
    # context rolled back: same step count as before the probe
    assert sim.context.getStepCount() == 0
    post_positions = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
    assert np.allclose(np.asarray(pre_positions.value_in_unit(unit.nanometer)),
                       np.asarray(post_positions.value_in_unit(unit.nanometer)))
    # controller rolled back: the probe replaces driver.controller with the
    # restored one, so read the driver's (the original object was advanced
    # and deliberately discarded)
    post_ctrl = driver.controller.state_dict()
    assert post_ctrl["counters"] == pre_ctrl["counters"]
    assert post_ctrl["next_due_step"] == pre_ctrl["next_due_step"]
    assert post_ctrl["rng"] == pre_ctrl["rng"]
    assert (tmp_path / "production_probe_report.json").exists()


def test_probe_failure_surfaces_and_reports_source(tmp_path):
    """A volume-move trial that fails mid-transaction: the REAL controller
    restores the pre-trial state before aborting, the probe still rolls the
    context and controller back, and the failure surfaces loudly."""
    runtime = NptRunContext(ownership=_make_ownership(), adapter=ExplodingAdapter(explode_at_call=2))
    sim = _make_sim()
    ctrl = runtime.initialize_controller(sim.context, seed=11)
    driver = ReplicaStepDriver(sim, controller=ctrl, label="replica_000")
    args = _make_args(production_probe_steps=50)
    pre_ctrl = ctrl.state_dict()
    pre_positions = np.asarray(
        sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer))

    with pytest.raises(RuntimeError, match="Production probe failed"):
        production.run_production_probe(
            args, tmp_path, [sim], [0], [np.array([0.4])], [10.0],
            {"mode": "distance", "cv_atom1": 0, "cv_atom2": 3}, 0, 3, unit,
            {}, drivers=[driver], pool=_NoPool(), npt_runtime=runtime,
        )
    report = json.loads((tmp_path / "production_probe_report.json").read_text())
    assert report["ok"] is False
    assert "volume-move trial failed" in report["exception"]
    # the context and the controller are rolled back despite the failure
    assert sim.context.getStepCount() == 0
    post_positions = np.asarray(
        sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    assert np.array_equal(post_positions, pre_positions)
    post_ctrl = driver.controller.state_dict()
    assert post_ctrl["counters"] == pre_ctrl["counters"]
    assert post_ctrl["next_due_step"] == pre_ctrl["next_due_step"]
    assert post_ctrl["rng"] == pre_ctrl["rng"]


# ------------------------------------------------- resolver / metadata gates


def test_production_barostat_description_reflects_resolved_backend():
    args_biased = _make_args()
    args_biased.npt_barostat_backend = "biased_mc"
    args_biased.production_ensemble = "npt"
    args_biased.production_barostat_frequency = 25
    desc = production._production_barostat_description(args_biased)
    assert "BiasedMCBarostatController" in desc and "25" in desc

    args_native = _make_args()
    args_native.npt_barostat_backend = "auto"
    args_native.production_ensemble = "npt"
    assert "MonteCarloBarostat" in production._production_barostat_description(args_native)

    args_nvt = _make_args()
    args_nvt.production_ensemble = "nvt"
    assert production._production_barostat_description(args_nvt) == "none"


def test_resolver_real_dispatch_supported_and_unsupported_modes():
    """The REAL gareus.npt.resolve_npt_backend behind resolve_barostat_ownership
    (package 1 is merged): supported Pep-GaMD under NPT resolves to the
    biased-MC backend, conventional MD to native, and every forbidden
    combination fails loudly."""
    from gareus.system_setup import resolve_barostat_ownership

    args = _make_args(production_ensemble="npt", run_mode="gamd")
    args.gamd_boost_type = "pep-gamd-lower-dual"
    args.npt_barostat_backend = "auto"
    args.production_barostat_frequency = 25
    args.barostat_frequency = 25
    args.pressure_bar = 1.0
    args.temperature_k = 300.0
    args.barostat_volume_step_fraction = 0.01
    ownership = resolve_barostat_ownership(args, ensemble="npt", run_mode="gamd",
                                           boost_type="pep-gamd-lower-dual")
    assert ownership.backend == "biased_mc"
    assert ownership.include_native_barostat is False

    args_cmd = _make_args(production_ensemble="npt", run_mode="cmd")
    args_cmd.gamd_boost_type = ""
    ownership = resolve_barostat_ownership(args_cmd, ensemble="npt", run_mode="cmd", boost_type="")
    assert ownership.backend == "native"
    assert ownership.include_native_barostat is True

    with pytest.raises(ValueError, match="native is invalid for boosted"):
        args_native_boost = _resolver_args("pep-gamd-lower-dual")
        args_native_boost.npt_barostat_backend = "native"
        resolve_barostat_ownership(args_native_boost, ensemble="npt", run_mode="gamd",
                                   boost_type="pep-gamd-lower-dual")

    with pytest.raises(ValueError, match="no validated target adapter"):
        resolve_barostat_ownership(
            _resolver_args("upper-total"), ensemble="npt", run_mode="gamd",
            boost_type="upper-total")


def _resolver_args(boost_type):
    args = _make_args(production_ensemble="npt", run_mode="gamd")
    args.gamd_boost_type = boost_type
    args.npt_barostat_backend = "auto"
    args.production_barostat_frequency = 25
    args.barostat_frequency = 25
    args.pressure_bar = 1.0
    args.temperature_k = 300.0
    args.barostat_volume_step_fraction = 0.01
    return args


def test_adapter_seam_fails_loudly_while_package2_is_missing(monkeypatch):
    # pep_gamd.build_effective_potential_adapter does not exist yet: the seam
    # must refuse rather than run volume moves against the wrong energy.
    import gareus.pep_gamd as pep_gamd
    monkeypatch.delattr(pep_gamd, "build_effective_potential_adapter", raising=False)
    args = _make_args()
    with pytest.raises(RuntimeError, match="package 2"):
        production._resolve_npt_adapter(args)
