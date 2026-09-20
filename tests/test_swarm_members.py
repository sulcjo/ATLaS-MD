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

import pytest


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


def test_default_graft_minimize_iters_is_500():
    """100 iterations (the old default) left 18/99 real swarm members NaN 0-3s into
    equilibration (2026-09-08); the umbrella-seeding graft caller already uses 1000 via
    --us-pull-minimize-iterations. This is the smaller of the two fixes in this task."""
    from gareus.swarm.members import _DEFAULT_GRAFT_MINIMIZE_ITERS
    assert _DEFAULT_GRAFT_MINIMIZE_ITERS == 500


def test_run_member_passes_graft_minimize_iters_through_to_the_graft_call():
    """run_member must read args.swarm_graft_minimize_iters (falling back to the module
    default when the attribute is absent) and forward it verbatim to
    graft_conformer_into_context -- the value used to be unreachable from config or the
    command line because gareus/cli.py had no matching flag at all. Every other OpenMM/MD
    call this function makes before the graft is mocked out; the graft itself is made to
    report a fallback so run_member returns immediately afterwards without needing any
    further MD/stepping infrastructure."""
    from unittest.mock import MagicMock, patch
    from gareus.swarm import members as members_mod

    def _run_member_with_args(args):
        captured = {}

        def fake_graft(sim, topology, conformer, cv_atom1, cv_atom2, temperature_k, unit,
                       minimize_iters=None, seed=None):
            captured["minimize_iters"] = minimize_iters
            return {"fallback": True, "fallback_reason": "short-circuited for the unit test"}

        openmm = MagicMock()
        app = MagicMock()
        unit = MagicMock()
        equil_state = MagicMock()
        equil_state.getPeriodicBoxVectors.return_value = (1.0, 2.0, 3.0)
        tmp = pathlib.Path(tempfile.mkdtemp())
        member_dir = tmp / "member_0000"

        with patch.object(members_mod, "solute_atom_indices", return_value=[0, 1, 2]), \
             patch.object(members_mod, "ensure_pep_gamd_partition"), \
             patch.object(members_mod, "make_cmd_integrator", return_value=(MagicMock(), None)), \
             patch.object(members_mod, "_terminal_ca_atoms", return_value=(0, 1)), \
             patch.object(members_mod, "graft_conformer_into_context", side_effect=fake_graft):
            done = members_mod.run_member(
                args, {"velocity_seed": 5, "member_id": 0}, member_dir,
                openmm=openmm, app=app, unit=unit, topology=object(),
                base_system_xml="<System/>", equil_state=equil_state,
                conformer={"positions_nm": None, "pdb_path": "x"},
                platform=None, props=None, contact_pairs=[],
            )
        assert done["status"] == "graft_failed"
        return captured["minimize_iters"]

    default_args = types.SimpleNamespace(temperature_k=300.0)
    assert _run_member_with_args(default_args) == 500

    overridden_args = types.SimpleNamespace(temperature_k=300.0, swarm_graft_minimize_iters=777)
    assert _run_member_with_args(overridden_args) == 777


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


@pytest.mark.slow
def test_tiny_real_swarm_member_moves_atoms_and_writes_trace_frames_and_done_slow():
    """Task 11 SLOW e2e: run one real swarm member end to end through ``run_member`` --
    graft, minimize, thermalize, unbiased Langevin MD, per-frame trace, seed-frame PDBs,
    ``last_frame.pdb``, ``done.json`` -- on the repo's cached GA-dipeptide-in-TIP3P fixture
    (``tests/pep_gamd_fixture.py``), CPU platform (no ``getPMEParametersInContext`` call
    anywhere here -- the Pep-GaMD partition pins PME analytically already).

    The "seed conformer" is a second, genuinely different GA backbone (phi=-120,
    psi=130 vs the fixture's phi=-60, psi=-45), built and protonated the exact same way
    the fixture builds its own peptide (build_peptide_pdb -> Modeller.addHydrogens with
    the same forcefield/pH) so it has the same atom count and per-atom name order and
    takes graft_conformer_into_context's real "wholesale" code path (Kabsch-align on Ca,
    overwrite, minimize clashes, re-thermalize) rather than a no-op identity graft --
    the atom-name-order match is asserted, not assumed. No native reference or
    folded-state label is used anywhere (ab initio, per global-constraints.md): the two
    backbones are just two arbitrary (phi, psi) choices, not a folded/native structure.

    Marked with the repo's ``slow`` marker (pyproject.toml) so a real pytest run can
    deselect it with ``-m "not slow"``; runs unconditionally otherwise (the fixture-free
    fallback runner ignores markers entirely and just calls the function, same as
    ``tests/test_package_smoke.py``'s ``test_tiny_lambda_ladder_run_completes_end_to_end_slow``)
    -- there is no env-var gate, so every run of this file actually executes the MD.
    """
    import numpy as np

    from pep_gamd_fixture import solvated_dipeptide
    from gareus.imports import import_openmm
    from gareus.cv import build_nonlocal_contact_pairs
    from gareus.swarm.members import run_member, TRACE_COLUMNS
    from gareus.system_setup import build_peptide_pdb, make_forcefield

    fx = solvated_dipeptide()  # cached: GA dipeptide, 2.4 nm TIP3P box, briefly minimized
    openmm, app, unit = import_openmm()

    args = types.SimpleNamespace(
        nonbonded_cutoff_nm=0.9, ewald_error_tolerance=0.0005, hmr=False, hydrogen_mass_amu=0.0,
        temperature_k=300.0, friction_per_ps=1.0, timestep_fs=2.0, run_mode="cmd", seed=1234,
        contact_atom_selection="heavy", contact_scheme="residue-balanced", contact_min_sequence_separation=1,
        contact_r0_a=4.5, contact_beta_a_inv=6.0, contact_normalize=True, contact_pair_warning_threshold=5000,
        swarm_seed_ns=0.002, swarm_equil_ps=0.4, swarm_output_interval_ps=0.2,
        swarm_seed_frame_interval_ps=0.4, swarm_graft_minimize_iters=200,
    )
    # 2 fs timestep: 0.4 ps equil = 200 discarded steps; 2 ps production at 0.2 ps/frame =
    # 100 steps/frame x 10 frames; seed-frame export every 0.4 ps = every 2nd frame.
    # 200 minimize iterations: a genuinely different conformer has real clashes after
    # grafting into the solvated box, unlike a self-graft's no-op overlap.

    base_system_xml = openmm.XmlSerializer.serialize(fx["system"])  # fresh copy per member, no partition yet
    platform = openmm.Platform.getPlatformByName("CPU")
    props = {}

    scratch_integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    scratch_ctx = openmm.Context(
        openmm.XmlSerializer.deserialize(base_system_xml), scratch_integrator, platform, props,
    )
    scratch_ctx.setPositions(fx["positions"])
    equil_state = scratch_ctx.getState(getPositions=True)
    del scratch_ctx, scratch_integrator

    pep_idx = list(fx["peptide"])  # ascending atom indices, same order run_member's graft expects

    # A second, differently-folded GA conformer (phi=-120, psi=130 vs the fixture's
    # phi=-60, psi=-45) built and protonated the exact same way the fixture builds its own
    # peptide, so the "wholesale" graft path (same atom count, no name map) is real, not
    # a self-graft identity no-op. No native reference or folded-state label -- both
    # backbones are arbitrary (phi, psi) choices.
    seed_dir = pathlib.Path(tempfile.mkdtemp())
    seed_pdb_path = build_peptide_pdb("GA", seed_dir / "seed.pdb", phi_deg=-120.0, psi_deg=130.0)
    seed_raw = app.PDBFile(str(seed_pdb_path))
    ff = make_forcefield(app, "tip3p")
    seed_modeller = app.Modeller(seed_raw.topology, seed_raw.positions)
    seed_modeller.addHydrogens(ff, pH=7.0)

    fixture_pep_names = [a.name for a in fx["topology"].atoms() if a.index in set(pep_idx)]
    seed_names = [a.name for a in seed_modeller.topology.atoms()]
    assert seed_names == fixture_pep_names, (
        "seed conformer atom names/order do not match the real topology's peptide atoms "
        "-- the wholesale graft path (same atom count, no index map) requires this"
    )

    seed_pos_nm = np.asarray(seed_modeller.positions.value_in_unit(unit.nanometer))
    conformer = {"positions_nm": seed_pos_nm, "pdb_path": str(seed_pdb_path)}

    contact_pairs = build_nonlocal_contact_pairs(fx["topology"], args)

    tmp = pathlib.Path(tempfile.mkdtemp())
    member_dir = tmp / "member_0000"

    t0 = time.time()
    done = run_member(
        args, {"velocity_seed": 777, "member_id": 0}, member_dir,
        openmm=openmm, app=app, unit=unit, topology=fx["topology"], base_system_xml=base_system_xml,
        equil_state=equil_state, conformer=conformer, platform=platform, props=props,
        contact_pairs=contact_pairs, progress=None,
    )
    wall_s = time.time() - t0
    print(f"test_tiny_real_swarm_member ... wall_s={wall_s:.1f}")
    assert wall_s < 120.0

    assert done["status"] == "ok", done
    assert (member_dir / "done.json").exists()
    reloaded_done = json.loads((member_dir / "done.json").read_text())
    assert reloaded_done["status"] == "ok"

    # The graft actually did something (a real wholesale graft, not a no-op self-graft).
    graft = done["graft"]
    assert graft["graft_mode"] == "wholesale"
    assert graft["n_grafted_atoms"] == len(pep_idx)
    assert np.isfinite(graft["ca_rmsd_A"])

    trace_rows = list(csv.DictReader((member_dir / "trace.csv").open()))
    assert trace_rows, "trace.csv has no rows"
    assert list(trace_rows[0].keys()) == TRACE_COLUMNS
    for row in trace_rows:
        assert np.isfinite(float(row["v_pep_kj"]))
        assert np.isfinite(float(row["v_dih_kj"]))

    frame_files = sorted((member_dir / "frames").glob("frame_*.pdb"))
    assert frame_files, "no seed-frame PDBs were written"
    last_frame_path = member_dir / "last_frame.pdb"
    assert last_frame_path.exists()

    def _positions_nm(pdb_path):
        return np.asarray(
            app.PDBFile(str(pdb_path)).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        )

    # frames/frame_*.pdb are peptide-only (write_solute_only_pdb); last_frame.pdb is the
    # full system (write_state_pdb, peptide + water) -- restrict it to the same peptide
    # atoms (ascending, same order as the solute-only PDBs) for a like-for-like compare.
    first_positions = _positions_nm(frame_files[0])
    last_positions_full = _positions_nm(last_frame_path)
    last_positions = last_positions_full[pep_idx, :]
    assert first_positions.shape == last_positions.shape, (
        "first seed-frame PDB and the peptide subset of last_frame.pdb have different atom counts"
    )
    max_disp_nm = float(np.max(np.linalg.norm(last_positions - first_positions, axis=1)))
    print(f"test_tiny_real_swarm_member ... max_disp_nm={max_disp_nm:.4f}")
    assert max_disp_nm > 1e-3, (
        f"atoms did not move between the first written frame and the last frame "
        f"(max per-atom displacement {max_disp_nm:.6f} nm <= 1e-3 nm threshold)"
    )


# ---------------------------------------------------------------------------
# Task 1: canonical torsion features recorded alongside every trace row
# ---------------------------------------------------------------------------

def test_member_loop_writes_torsion_features_aligned_with_trace(tmp_path):
    import numpy as np
    from gareus.swarm.members import run_member_loop

    calls = {"n": 0}

    def measure_fn():
        calls["n"] += 1
        return {"cv1": 0.1 * calls["n"], "rg_nm": 1.0, "e2e_nm": 2.0,
                "v_pep_kj": -1.0, "v_dih_kj": 1.0, "potential_kj": -5.0}

    def feature_fn():
        return np.array([np.sin(calls["n"]), np.cos(calls["n"]), 0.0, 1.0])

    trace = tmp_path / "trace.csv"
    feats = tmp_path / "torsion_features.npy"
    run_member_loop(n_equil_steps=0, n_prod_steps=6, steps_per_frame=2, seed_frame_every=100,
                    step_fn=lambda n: None, measure_fn=measure_fn, write_frame_fn=lambda i: None,
                    trace_path=trace, timestep_ps=0.002, feature_fn=feature_fn,
                    features_path=feats)
    rows = trace.read_text().strip().splitlines()[1:]
    stored = np.load(feats)
    assert stored.shape == (len(rows), 4)
    assert np.isclose(stored[1, 0], np.sin(2.0))      # measured in the same call as trace row 1
    assert not list(tmp_path.glob("*.tmp*"))          # atomic publish left no temp file behind


def test_member_loop_without_feature_fn_writes_no_feature_file(tmp_path):
    from gareus.swarm.members import run_member_loop

    trace = tmp_path / "trace.csv"
    run_member_loop(n_equil_steps=0, n_prod_steps=2, steps_per_frame=1, seed_frame_every=100,
                    step_fn=lambda n: None,
                    measure_fn=lambda: {"cv1": 0.0, "rg_nm": 1.0, "e2e_nm": 1.0,
                                        "v_pep_kj": 0.0, "v_dih_kj": 0.0, "potential_kj": 0.0},
                    write_frame_fn=lambda i: None, trace_path=trace, timestep_ps=0.002)
    assert not (tmp_path / "torsion_features.npy").exists()


def test_a_crash_mid_loop_leaves_no_feature_file_and_is_recorded_as_md_failed(tmp_path):
    import time
    from gareus.swarm.members import _run_loop_recording_failure

    calls = {"n": 0}

    def step_fn(n):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("NaN coordinate")

    result = _run_loop_recording_failure(
        n_equil_steps=0, n_prod_steps=6, steps_per_frame=1, seed_frame_every=100,
        step_fn=step_fn,
        measure_fn=lambda: {"cv1": 0.0, "rg_nm": 1.0, "e2e_nm": 1.0,
                            "v_pep_kj": 0.0, "v_dih_kj": 0.0, "potential_kj": 0.0},
        write_frame_fn=lambda i: None, trace_path=tmp_path / "trace.csv", timestep_ps=0.002,
        member_dir=tmp_path, member_id=7, crash_label="swarm", t0=time.time(),
        feature_fn=lambda: [0.0, 1.0], features_path=tmp_path / "torsion_features.npy")
    assert result["failed"] and result["done"]["status"] == "md_failed"
    assert not (tmp_path / "torsion_features.npy").exists()   # a partial member has no features
