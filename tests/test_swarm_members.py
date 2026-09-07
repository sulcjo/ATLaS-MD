"""One swarm member's bookkeeping loop and the measured-frame contract (Task 5).

The MD-heavy parts of ``run_member`` (grafting, OpenMM Simulation construction) are not
exercised here -- ``run_member_loop`` takes injected ``step_fn``/``measure_fn``/
``write_frame_fn`` callables precisely so the frame/CSV/seed-frame bookkeeping is testable
without any MD, and ``measure_frame`` is exercised on a tiny real OpenMM system (Reference
platform, no PME-context calls on CPU).
"""
import csv
import json
import pathlib
import tempfile
import time
import types


def test_run_member_loop_writes_expected_frames_and_seed_pdbs():
    from gareus.swarm.members import run_member_loop, TRACE_COLUMNS
    tmp = pathlib.Path(tempfile.mkdtemp())
    stepped = []
    frames_written = []
    state = {"t": 0}

    def step_fn(n):
        stepped.append(n)
        state["t"] += n

    def measure_fn():
        return {"cv1": 0.3, "rg_nm": 0.6, "e2e_nm": 1.0, "v_pep_kj": -50.0 - state["t"], "v_dih_kj": 20.0, "potential_kj": -1e4}

    def write_frame_fn(i):
        frames_written.append(i)

    summary = run_member_loop(
        n_equil_steps=100, n_prod_steps=1000, steps_per_frame=100, seed_frame_every=5,
        step_fn=step_fn, measure_fn=measure_fn, write_frame_fn=write_frame_fn, trace_path=tmp / "trace.csv",
    )
    rows = list(csv.DictReader((tmp / "trace.csv").open()))
    assert list(rows[0].keys()) == TRACE_COLUMNS
    assert len(rows) == 10 and summary["n_frames"] == 10
    assert stepped[0] == 100 and sum(stepped) == 1100  # equilibration first, never in the trace
    assert float(rows[0]["v_pep_kj"]) == -50.0 - 200.0  # first trace row is after equil + first prod chunk
    assert frames_written == [4, 9]  # every 5th frame (0-based index 4, 9)
    assert float(rows[-1]["t_ps"]) > float(rows[0]["t_ps"])
    csv_frame_indices = {int(r["frame"]) for r in rows}
    assert set(frames_written) <= csv_frame_indices  # Task 4 joins frames/*.pdb to trace.csv by this index


def test_run_member_loop_rejects_non_divisible_chunking():
    from gareus.swarm.members import run_member_loop
    try:
        run_member_loop(
            n_equil_steps=0, n_prod_steps=1001, steps_per_frame=100, seed_frame_every=1,
            step_fn=lambda n: None, measure_fn=lambda: {}, write_frame_fn=lambda i: None,
            trace_path=pathlib.Path(tempfile.mkdtemp()) / "t.csv",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for n_prod_steps not divisible by steps_per_frame")


def test_measure_frame_returns_all_energy_and_geometry_keys_on_tiny_real_system():
    # Reference platform, alanine-dipeptide-like tiny system from the repo's Pep-GaMD test fixture.
    from pep_gamd_fixture import tiny_solvated_system
    from gareus.swarm.members import measure_frame, TRACE_COLUMNS
    openmm, app, unit, topology, system, positions = tiny_solvated_system()
    from gareus.pep_gamd import ensure_pep_gamd_partition
    from gareus.cv import solute_atom_indices, build_nonlocal_contact_pairs, peptide_residues, find_atom_in_residue
    ensure_pep_gamd_partition(system, solute_atom_indices(topology))
    args = types.SimpleNamespace(
        contact_atom_selection="heavy", contact_scheme="residue-balanced", contact_min_sequence_separation=1,
        contact_r0_a=4.5, contact_beta_a_inv=6.0, contact_normalize=True, contact_pair_warning_threshold=5000,
    )
    pairs = build_nonlocal_contact_pairs(topology, args)
    ca = [find_atom_in_residue(r, "CA") for r in peptide_residues(topology)]
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(positions)
    import numpy as np
    pos_nm = np.asarray(ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    row = measure_frame(ctx, system, unit, pos_nm, pairs, ca, args)
    assert set(row) == set(TRACE_COLUMNS) - {"frame", "t_ps"}
    assert all(np.isfinite(float(row[k])) for k in row)


def test_member_done_false_when_no_done_json():
    from gareus.swarm.members import member_done
    tmp = pathlib.Path(tempfile.mkdtemp())
    assert member_done(tmp) is False


def test_member_done_true_when_done_json_parses():
    from gareus.swarm.members import member_done
    tmp = pathlib.Path(tempfile.mkdtemp())
    (tmp / "done.json").write_text(json.dumps({"member_id": 0, "n_frames": 5}))
    assert member_done(tmp) is True


def test_member_done_false_when_done_json_unparseable():
    from gareus.swarm.members import member_done
    tmp = pathlib.Path(tempfile.mkdtemp())
    (tmp / "done.json").write_text("{not json")
    assert member_done(tmp) is False


def test_run_member_raises_without_a_seed_conformer():
    from gareus.swarm.members import run_member
    tmp = pathlib.Path(tempfile.mkdtemp())
    try:
        run_member(
            types.SimpleNamespace(), {"velocity_seed": 0}, tmp,
            openmm=None, app=None, unit=None, topology=None, base_system_xml="<System/>",
            equil_state=None, conformer=None, platform=None, props=None, contact_pairs=[],
        )
    except ValueError as exc:
        assert "seed conformer" in str(exc)
    else:
        raise AssertionError("expected ValueError when conformer is None")


def test_run_loop_recording_failure_writes_md_failed_done_and_never_raises():
    """A mid-loop exception (e.g. run_steps_safely raising on a NaN coordinate) is a failed
    member, not an uncaught exception: recorded with frames_written == the number of trace
    rows already flushed before the crash, never re-raised."""
    from gareus.swarm.members import _run_loop_recording_failure
    tmp = pathlib.Path(tempfile.mkdtemp())
    calls = {"n": 0}

    def step_fn(n):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("Particle coordinate is NaN")

    def measure_fn():
        return {"cv1": 0.1, "rg_nm": 0.5, "e2e_nm": 0.9, "v_pep_kj": -10.0, "v_dih_kj": 5.0, "potential_kj": -100.0}

    result = _run_loop_recording_failure(
        n_equil_steps=0, n_prod_steps=500, steps_per_frame=100, seed_frame_every=1,
        step_fn=step_fn, measure_fn=measure_fn, write_frame_fn=lambda i: None,
        trace_path=tmp / "trace.csv", timestep_ps=0.004,
        member_dir=tmp, member_id=7, crash_label="swarm", t0=time.time(),
    )
    assert result["failed"] is True
    done = result["done"]
    assert done["status"] == "md_failed"
    assert "NaN" in done["error"]
    assert done["frames_written"] == 2
    assert done["n_frames"] == 2
    assert done["member_id"] == 7
    assert done["graft_fallback"] is False
    assert done["crash_pdb"] is None  # no CRASH_*.pdb was left behind by this fake step_fn

    # done.json round-trips exactly as run_member would write/read it.
    done_path = tmp / "done.json"
    with done_path.open("w") as f:
        json.dump(done, f)
    with done_path.open() as f:
        reloaded = json.load(f)
    assert reloaded["status"] == "md_failed"
    assert reloaded["frames_written"] == 2

    from gareus.swarm.members import member_done
    assert member_done(tmp) is True


def test_run_loop_recording_failure_picks_the_highest_step_crash_pdb():
    """CRASH_<label>_before_nan_step_<n>.pdb filenames are not zero-padded, so the highest
    step number must be picked numerically, not by a lexical sort/glob order."""
    from gareus.swarm.members import _run_loop_recording_failure
    tmp = pathlib.Path(tempfile.mkdtemp())
    (tmp / "CRASH_swarm_before_nan_step_20.pdb").write_text("REMARK fake\n")
    (tmp / "CRASH_swarm_before_nan_step_100.pdb").write_text("REMARK fake\n")

    def step_fn(n):
        raise RuntimeError("Particle coordinate is NaN")

    result = _run_loop_recording_failure(
        n_equil_steps=0, n_prod_steps=100, steps_per_frame=100, seed_frame_every=1,
        step_fn=step_fn, measure_fn=lambda: {}, write_frame_fn=lambda i: None,
        trace_path=tmp / "trace.csv", timestep_ps=0.004,
        member_dir=tmp, member_id=1, crash_label="swarm", t0=time.time(),
    )
    assert result["done"]["crash_pdb"] == str(tmp / "CRASH_swarm_before_nan_step_100.pdb")


def test_run_loop_recording_failure_returns_success_when_nothing_raises():
    from gareus.swarm.members import _run_loop_recording_failure
    tmp = pathlib.Path(tempfile.mkdtemp())
    result = _run_loop_recording_failure(
        n_equil_steps=0, n_prod_steps=200, steps_per_frame=100, seed_frame_every=1,
        step_fn=lambda n: None,
        measure_fn=lambda: {"cv1": 0.0, "rg_nm": 0.0, "e2e_nm": 0.0, "v_pep_kj": 0.0, "v_dih_kj": 0.0, "potential_kj": 0.0},
        write_frame_fn=lambda i: None,
        trace_path=tmp / "trace.csv", timestep_ps=0.004,
        member_dir=tmp, member_id=3, crash_label="swarm", t0=time.time(),
    )
    assert result["failed"] is False
    assert result["loop_summary"]["n_frames"] == 2
