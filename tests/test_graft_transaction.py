"""F03/P06: graft preparation is transactional and fails closed (spec F03; findings N01-N03)."""
from __future__ import annotations

import types

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402

import gareus.seeding as S  # noqa: E402
from gareus.cv import peptide_residues  # noqa: E402
from pep_gamd_fixture import _fresh_system, solvated_dipeptide  # noqa: E402


def _sim_and_identity_conformer(clash_offset_nm=0.01, platform="CPU"):
    fx = solvated_dipeptide()
    topology = fx["topology"]
    system = _fresh_system()
    integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.5 * unit.femtoseconds)
    sim = app.Simulation(topology, system, integrator, openmm.Platform.getPlatformByName(platform))
    pos = np.asarray(fx["positions"].value_in_unit(unit.nanometer), dtype=float)
    pep_atoms = [a.index for r in peptide_residues(topology) for a in r.atoms()]
    conformer = {"positions_nm": pos[pep_atoms], "pdb_path": "identity", "seed_id": "seed_identity",
                 "topology_to_conformer_atom_index": {int(t): i for i, t in enumerate(pep_atoms)}}
    waters = [r for r in topology.residues() if r.name in ("HOH", "WAT")]
    w = [a.index for a in waters[0].atoms()]
    target = pep_atoms[len(pep_atoms) // 2]
    clashed = pos.copy()
    clashed[w] += (pos[target] + np.array([clash_offset_nm, 0.0, 0.0])) - pos[w[0]]
    sim.context.setPositions(clashed * unit.nanometer)
    ca = [a.index for a in topology.atoms() if a.name == "CA"]
    return sim, topology, conformer, ca, clashed


def _positions(sim):
    return np.asarray(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer))


def test_real_solvated_graft_with_a_water_inside_the_peptide_resolves_and_is_finite():
    sim, topology, conformer, ca, _ = _sim_and_identity_conformer()
    st = S.graft_conformer_into_context(sim, topology, conformer, ca[0], ca[-1], 300.0, unit, minimize_iters=200, seed=1)
    assert st.get("fallback") is False, st
    rep = st["solvent_repair"]
    assert rep["status"] == "RESOLVED" and rep["n_groups_moved"] >= 1 and rep["seed_id"] == "seed_identity"
    assert rep["n_solvent_solvent_clashes"] == 0 and rep["n_peptide_solvent_clashes"] == 0
    state = sim.context.getState(getEnergy=True, getForces=True)
    e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f = np.asarray(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    assert np.isfinite(e) and e < 0 and np.isfinite(f).all() and np.abs(f).max() < S._GRAFT_MAX_FORCE_KJ_MOL_NM


def test_unresolved_repair_is_a_failed_graft_and_the_context_is_restored(monkeypatch):
    sim, topology, conformer, ca, clashed = _sim_and_identity_conformer()
    import gareus.solvent_repair as R
    real = R.repair_solvent_clashes

    def unresolved(*a, **k):
        res = real(*a, **k)
        res.status = R.STATUS_UNRESOLVED
        res.reason_code = "exhausted:test"
        return res
    monkeypatch.setattr(R, "repair_solvent_clashes", unresolved)
    st = S.graft_conformer_into_context(sim, topology, conformer, ca[0], ca[-1], 300.0, unit, minimize_iters=5, seed=1)
    assert st["fallback"] is True and st["fallback_reason"].startswith("solvent_repair_unresolved")
    assert st["solvent_repair"]["reason_code"] == "exhausted:test"
    np.testing.assert_allclose(_positions(sim), clashed, atol=1e-6)     # nothing of the candidate remains


_REAL_ACCEPTED_STATE_CHECK = S._accepted_state_check


@pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=["nan", "inf"])
@pytest.mark.parametrize("quantity", ["positions", "forces", "energy"])
def test_injected_nan_and_inf_in_each_accepted_state_quantity_rejects_and_rolls_back(monkeypatch, quantity, bad):
    for _ in (0,):
        sim, topology, conformer, ca, clashed = _sim_and_identity_conformer()
        real_check = _REAL_ACCEPTED_STATE_CHECK        # captured once at import, never the patched one

        def poisoned(min_state, unit_):
            out = real_check(min_state, unit_)
            # re-run the validation on poisoned copies of the accepted-state arrays
            pos = np.array(out["positions_nm"], copy=True)
            forces = np.zeros_like(pos) + 1.0
            energy = out["energy_kj_mol"]
            if quantity == "positions":
                pos[3, 1] = bad
            elif quantity == "forces":
                forces[5, 2] = bad
            else:
                energy = bad

            class _State:
                def getPositions(self, asNumpy=True):
                    return pos * unit_.nanometer

                def getForces(self, asNumpy=True):
                    return forces * (unit_.kilojoule_per_mole / unit_.nanometer)

                def getPotentialEnergy(self):
                    return energy * unit_.kilojoule_per_mole
            return real_check(_State(), unit_)
        monkeypatch.setattr(S, "_accepted_state_check", poisoned)
        st = S.graft_conformer_into_context(sim, topology, conformer, ca[0], ca[-1], 300.0, unit, minimize_iters=5, seed=1)
        assert st["fallback"] is True, (quantity, bad, st)
        assert st["fallback_reason"] == f"minimization_nonfinite_{quantity}"
        np.testing.assert_allclose(_positions(sim), clashed, atol=1e-6)


def test_a_finite_force_above_the_bound_is_a_blowup_by_its_documented_convention():
    class _State:
        def getPositions(self, asNumpy=True):
            return np.zeros((2, 3)) * unit.nanometer

        def getForces(self, asNumpy=True):
            f = np.zeros((2, 3)); f[1, 0] = 2.0 * S._GRAFT_MAX_FORCE_KJ_MOL_NM
            return f * (unit.kilojoule_per_mole / unit.nanometer)

        def getPotentialEnergy(self):
            return -1.0 * unit.kilojoule_per_mole
    out = S._accepted_state_check(_State(), unit)
    assert out["fallback_reason"] == "minimization_blowup"
    assert out["max_force_kj_mol_nm"] == pytest.approx(2.0 * S._GRAFT_MAX_FORCE_KJ_MOL_NM)


def test_failed_then_healthy_preparation_matches_a_clean_context(monkeypatch):
    # attempt 1 fails (forced UNRESOLVED) and must not perturb attempt 2 on the same context.
    # Reference platform: the CPU minimiser's threaded reductions differ between contexts by
    # ~1e-3 nm, which would mask (or fake) a transactional defect at this tolerance.
    sim, topology, conformer, ca, clashed = _sim_and_identity_conformer(platform="Reference")
    import gareus.solvent_repair as R
    real = R.repair_solvent_clashes
    calls = {"n": 0}

    def flaky(*a, **k):
        res = real(*a, **k)
        calls["n"] += 1
        if calls["n"] == 1:
            res.status = R.STATUS_UNRESOLVED
        return res
    monkeypatch.setattr(R, "repair_solvent_clashes", flaky)
    first = S.graft_conformer_into_context(sim, topology, conformer, ca[0], ca[-1], 300.0, unit, minimize_iters=50, seed=2)
    assert first["fallback"] is True
    second = S.graft_conformer_into_context(sim, topology, conformer, ca[0], ca[-1], 300.0, unit, minimize_iters=50, seed=2)
    assert second["fallback"] is False
    after_retry = _positions(sim)
    # clean context, same seed and budget
    sim2, topology2, conformer2, ca2, _ = _sim_and_identity_conformer(platform="Reference")
    monkeypatch.setattr(R, "repair_solvent_clashes", real)
    clean = S.graft_conformer_into_context(sim2, topology2, conformer2, ca2[0], ca2[-1], 300.0, unit, minimize_iters=50, seed=2)
    assert clean["fallback"] is False
    np.testing.assert_allclose(after_retry, _positions(sim2), atol=1e-6)
    assert second["solvent_repair"]["n_groups_moved"] == clean["solvent_repair"]["n_groups_moved"]


def test_wrapper_reports_status_for_the_two_review_fixtures():
    from fixtures.thermodynamic_repair import oscillating_displacement_case, solvent_collision_case
    c = solvent_collision_case()
    out, stats = S.displace_clashing_solvent_nm(c.positions_nm, c.box_nm, c.peptide, c.groups, c.heavy)
    assert stats["status"] == "RESOLVED"
    assert np.linalg.norm(out[2] - out[5]) >= 0.16 - 1e-9
    o = oscillating_displacement_case()
    out, stats = S.displace_clashing_solvent_nm(o.positions_nm, o.box_nm, o.peptide, o.groups, o.heavy)
    assert stats["status"] in ("RESOLVED", "UNRESOLVED")
    if stats["status"] == "UNRESOLVED":
        assert stats["n_peptide_solvent_clashes"] >= 1
