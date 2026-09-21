"""The graft must not leave solvent inside the placed peptide (chignolin_8 swarm, 2026-09-21).

`graft_conformer_into_context` overwrote every peptide atom with the compact seed but left
the water box untouched. A water oxygen 0.098 A from Trp9 CZ2 gave E = 3.6e18 kJ/mol after
"minimisation" (L-BFGS moved nothing), the graft passed its only check (positions not NaN)
and MD NaN'd at step 50 -- six of six replicates of one seed, emptying a stratification cell
and failing epoch 0's coverage gate. The healthy control seed's worst overlap was 0.149 A,
which the minimiser still resolves: whether a member survives was a lottery on where the
solvent happened to sit.
"""
import numpy as np
import pytest


def _cubic(L):
    return np.diag([L, L, L]).astype(float)


def test_overlapping_water_is_pushed_out_rigidly_and_the_rest_is_untouched():
    from gareus.seeding import displace_clashing_solvent_nm
    pep = np.array([[1.0, 1.0, 1.0], [1.15, 1.0, 1.0], [1.0, 1.15, 1.0]])
    water = np.array([[1.005, 1.0, 1.0], [1.005, 1.0957, 1.0], [1.09, 0.98, 1.0]])   # O on top of pep[0]
    far = np.array([[2.5, 2.5, 2.5], [2.5, 2.6, 2.5], [2.59, 2.47, 2.5]])
    pos = np.vstack([pep, water, far])
    heavy = np.array([True, True, True, True, False, False, True, False, False])
    out, stats = displace_clashing_solvent_nm(
        pos, _cubic(5.0), np.arange(3), [np.arange(3, 6), np.arange(6, 9)], heavy)
    assert stats["n_groups_moved"] == 1 and stats["n_rounds"] >= 1
    np.testing.assert_array_equal(out[:3], pep)                     # peptide never moves
    np.testing.assert_array_equal(out[6:], far)                     # non-clashing water bitwise untouched
    # rigid: intramolecular geometry preserved
    d_before = np.linalg.norm(water[1:] - water[0], axis=1)
    d_after = np.linalg.norm(out[4:6] - out[3], axis=1)
    np.testing.assert_allclose(d_after, d_before, atol=1e-12)
    # clear of the peptide at the requested radii
    r = np.linalg.norm(out[3:6][:, None, :] - out[:3][None, :, :], axis=-1)
    assert r[0].min() >= 0.22 - 1e-9                                # O vs heavy
    assert r[1:].min() >= 0.16 - 1e-9                               # H vs heavy


def test_minimum_image_clash_across_the_box_boundary_is_seen_and_resolved():
    from gareus.seeding import displace_clashing_solvent_nm
    L = 3.0
    pep = np.array([[0.02, 1.5, 1.5]])
    water = np.array([[2.99, 1.5, 1.5], [2.99, 1.5957, 1.5], [3.08 - L, 1.47, 1.5]])  # 0.03 nm away through the wall
    pos = np.vstack([pep, water])
    out, stats = displace_clashing_solvent_nm(pos, _cubic(L), np.array([0]), [np.arange(1, 4)],
                                              np.array([True, True, False, False]))
    assert stats["n_groups_moved"] == 1
    d = out[1] - out[0]
    d -= L * np.round(d / L)
    assert np.linalg.norm(d) >= 0.22 - 1e-9


def test_no_clash_means_no_change_and_no_rounds():
    from gareus.seeding import displace_clashing_solvent_nm
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0957, 0.0], [1.09, -0.03, 0.0]])
    out, stats = displace_clashing_solvent_nm(pos, _cubic(4.0), np.array([0]), [np.arange(1, 4)],
                                              np.array([True, True, False, False]))
    np.testing.assert_array_equal(out, pos)
    assert stats["n_groups_moved"] == 0


def test_a_coincident_atom_still_gets_a_direction_and_is_moved():
    from gareus.seeding import displace_clashing_solvent_nm
    pos = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0957, 1.0], [1.09, 0.97, 1.0]])  # exact coincidence
    out, stats = displace_clashing_solvent_nm(pos, _cubic(4.0), np.array([0]), [np.arange(1, 4)],
                                              np.array([True, True, False, False]), seed=7)
    assert stats["n_groups_moved"] == 1
    assert np.linalg.norm(out[1] - out[0]) >= 0.22 - 1e-9


def test_solvent_groups_from_topology_cover_every_non_peptide_residue(tmp_path):
    openmm = pytest.importorskip("openmm")
    from openmm import app
    from pep_gamd_fixture import solvated_dipeptide
    from gareus.seeding import solvent_groups_from_topology
    fx = solvated_dipeptide()
    topology = fx["topology"]
    groups, heavy = solvent_groups_from_topology(topology)
    n_pep = sum(1 for a in topology.atoms() if a.residue.name not in ("HOH", "WAT", "NA", "CL", "K", "Na+", "Cl-"))
    covered = sum(len(g) for g in groups)
    assert covered == topology.getNumAtoms() - n_pep
    assert heavy.shape == (topology.getNumAtoms(),) and heavy.dtype == bool
    assert all(len(g) in (1, 3, 4, 5) for g in groups)             # ions, TIP3P, 4- and 5-site waters


def test_graft_survives_a_water_placed_inside_the_peptide_and_refuses_a_blown_up_minimisation(monkeypatch):
    openmm = pytest.importorskip("openmm")
    from openmm import app, unit
    from pep_gamd_fixture import solvated_dipeptide, _fresh_system
    import gareus.seeding as S
    from gareus.cv import peptide_residues
    fx = solvated_dipeptide()
    topology = fx["topology"]
    system = _fresh_system()
    integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.5 * unit.femtoseconds)
    sim = app.Simulation(topology, system, integrator, openmm.Platform.getPlatformByName("CPU"))
    pos = np.asarray(fx["positions"].value_in_unit(unit.nanometer), dtype=float)
    sim.context.setPositions(pos * unit.nanometer)
    pep_atoms = [a.index for r in peptide_residues(topology) for a in r.atoms()]
    # Build a conformer = the peptide's own coordinates (so the graft is the identity) and park a
    # water oxygen 0.01 nm from a peptide heavy atom: the chignolin_8 failure in miniature.
    conformer = {"positions_nm": pos[pep_atoms], "pdb_path": "identity",
                 "topology_to_conformer_atom_index": {int(t): i for i, t in enumerate(pep_atoms)}}
    waters = [r for r in topology.residues() if r.name in ("HOH", "WAT")]
    w = [a.index for a in waters[0].atoms()]
    target = pep_atoms[len(pep_atoms) // 2]
    shift = (pos[target] + np.array([0.01, 0.0, 0.0])) - pos[w[0]]
    clashed = pos.copy(); clashed[w] += shift
    sim.context.setPositions(clashed * unit.nanometer)
    ca = [a.index for a in topology.atoms() if a.name == "CA"]
    st = S.graft_conformer_into_context(sim, topology, conformer, ca[0], ca[-1], 300.0, unit,
                                        minimize_iters=200, seed=1)
    assert st.get("fallback") is False, st
    assert st["n_solvent_groups_displaced"] >= 1
    state = sim.context.getState(getEnergy=True, getForces=True)
    e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    fmax = np.abs(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)).max()
    assert np.isfinite(e) and e < 0 and fmax < S._GRAFT_MAX_FORCE_KJ_MOL_NM

    # And if minimisation cannot fix a graft, that is a failed graft -- not a successful one:
    # leave the clash in place (repair reports RESOLVED without moving anything) and skip
    # minimisation, so the accepted-state force bound is what has to catch it.
    import gareus.solvent_repair as R
    monkeypatch.setattr(S, "_minimize_energy", lambda sim, **kw: None)

    def no_repair(positions_nm, *a, **k):
        return R.RepairResult(R.STATUS_RESOLVED, "resolved", np.array(positions_nm, dtype=float))
    monkeypatch.setattr(R, "repair_solvent_clashes", no_repair)
    sim.context.setPositions(clashed * unit.nanometer)
    st2 = S.graft_conformer_into_context(sim, topology, conformer, ca[0], ca[-1], 300.0, unit,
                                         minimize_iters=1, seed=1)
    assert st2.get("fallback") is True and st2["fallback_reason"] == "minimization_blowup"
