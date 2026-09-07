"""One swarm member's bookkeeping loop and the measured-frame contract (Task 5).

The MD-heavy parts of ``run_member`` (grafting, OpenMM Simulation construction) are not
exercised here -- ``run_member_loop`` takes injected ``step_fn``/``measure_fn``/
``write_frame_fn`` callables precisely so the frame/CSV/seed-frame bookkeeping is testable
without any MD, and ``measure_frame`` is exercised on a tiny real OpenMM system (Reference
platform, no PME-context calls on CPU).
"""
import csv
import pathlib
import tempfile
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
    import json
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
