"""Task 6: the residual-torsion-pc runtime force is the numeric evaluator, chain rule included.

ss_k reaches the Context already in kJ/mol/CV^2 (production.py converts before
set_window; set_window writes it verbatim). The force expression therefore
contains NO 4.184, and these tests set ss_k in kJ.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

from gareus.cv import contact_normalization_denominator, residual_cv2_from_positions_nm  # noqa: E402
from gareus.cv_selection.models import (PairModelRuntime, ResidualFit,  # noqa: E402
                                        contact_pair_list_digest)
from gareus.production import (_add_residual_torsion_cv_force,  # noqa: E402
                               _ss_scalar_from_sub_cv_values)

KJ_PER_KCAL = 4.184


def _peptide_like(n_atoms=16, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.cumsum(rng.normal(scale=0.12, size=(n_atoms, 3)), axis=0) + 2.0
    phi = [(0, 1, 2, 3), (4, 5, 6, 7)]
    psi = [(8, 9, 10, 11), (12, 13, 14, 15)]
    contacts = [(0, 15, 1.0), (1, 14, 1.0), (2, 13, 1.0), (3, 12, 1.0)]
    return pos, phi, psi, contacts


class _Args:
    contact_r0_a = 12.0
    contact_beta_a_inv = 3.0
    contact_normalize = True
    contact_min_sequence_separation = 4
    contact_atom_selection = "heavy"
    contact_scheme = "atom-pairs"


def _definition(contacts, args=_Args()):
    return {"r0_angstrom": 12.0, "beta_per_angstrom": 3.0, "min_sequence_separation": 4,
            "atom_selection": "heavy", "pair_rule": "explicit-test-pairs", "normalize": True,
            "pair_list_sha256": contact_pair_list_digest(contacts),
            "norm": contact_normalization_denominator(contacts, args)}


def _runtime(phi, psi, contacts, d=8, degree=2, seed=1):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(3, d)) * 0.3
    if degree == 1:
        B[2] = 0.0
    V = np.linalg.qr(rng.normal(size=(d, d)))[0].T
    fit = ResidualFit(B, rng.normal(size=d) * 0.1, np.linspace(2, 0.5, d), V, 0.05, 0.02,
                      (-3.0, 3.0), np.zeros(d), np.ones(d), degree)
    return PairModelRuntime(fit, 2, "nonlocal-contact-fraction", _definition(contacts), "f" * 64,
                            tuple(phi) + tuple(psi))


def _energy_forces(pos, phi, psi, contacts, runtime, ss_k_kj, c2, *, return_subcvs=False):
    system = openmm.System()
    for _ in range(len(pos)):
        system.addParticle(12.0)
    meta = _add_residual_torsion_cv_force(openmm, system, phi, psi, contacts, runtime, _Args(),
                                          force_group=29)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pos * unit.nanometer)
    ctx.setParameter("ss_k", ss_k_kj)
    ctx.setParameter("ss0", c2)
    st = ctx.getState(getEnergy=True, getForces=True, groups={29})
    energy = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = np.asarray(st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    if return_subcvs:
        subcvs = tuple(float(x) for x in system.getForce(0).getCollectiveVariableValues(ctx))
        return energy, forces, meta, subcvs
    return energy, forces, meta


@pytest.mark.parametrize("degree", [1, 2])
def test_umbrella_energy_matches_the_numpy_evaluator_with_ss_k_in_kJ(degree):
    pos, phi, psi, contacts = _peptide_like()
    rt = _runtime(phi, psi, contacts, degree=degree)
    k2_kcal, c2 = 40.0, 0.3
    e, _, _ = _energy_forces(pos, phi, psi, contacts, rt, k2_kcal * KJ_PER_KCAL, c2)
    z2 = residual_cv2_from_positions_nm(pos, rt, phi, psi, contacts)
    expected = 0.5 * k2_kcal * KJ_PER_KCAL * (z2 - c2) ** 2
    assert abs(e - expected) <= 1e-6 * max(1.0, abs(expected))


def test_forces_match_finite_differences_including_the_chain_rule_term():
    pos, phi, psi, contacts = _peptide_like(seed=5)
    rt = _runtime(phi, psi, contacts, degree=2)
    _, f, _ = _energy_forces(pos, phi, psi, contacts, rt, 40.0 * KJ_PER_KCAL, 0.3)
    h = 1e-6
    for atom in (0, 3, 9, 15):                      # atoms in torsions and in contacts
        for ax in range(3):
            p = pos.copy(); p[atom, ax] += h
            m = pos.copy(); m[atom, ax] -= h
            ep = _energy_forces(p, phi, psi, contacts, rt, 40.0 * KJ_PER_KCAL, 0.3)[0]
            em = _energy_forces(m, phi, psi, contacts, rt, 40.0 * KJ_PER_KCAL, 0.3)[0]
            fd = -(ep - em) / (2 * h)
            assert abs(f[atom, ax] - fd) <= 1e-4 * max(1.0, abs(fd)), (atom, ax, f[atom, ax], fd)


def test_dropping_the_anchor_term_changes_the_energy():
    """The mutation this task exists to catch: a torsion-only projection is a different CV."""
    pos, phi, psi, contacts = _peptide_like(seed=7)
    rt = _runtime(phi, psi, contacts, degree=2)
    e_full = _energy_forces(pos, phi, psi, contacts, rt, 40.0 * KJ_PER_KCAL, 0.3)[0]
    torsion_only = ResidualFit(np.zeros_like(rt.fit.coefficients), rt.fit.residual_mean,
                               rt.fit.singular_values, rt.fit.right_vectors, rt.fit.anchor_mean,
                               rt.fit.anchor_std, rt.fit.anchor_clamp, rt.fit.projection_mean,
                               rt.fit.projection_std, 2)
    mut = PairModelRuntime(torsion_only, 2, rt.anchor_kind, rt.anchor_definition, rt.pair_sha256,
                           rt.feature_atoms)
    e_mut = _energy_forces(pos, phi, psi, contacts, mut, 40.0 * KJ_PER_KCAL, 0.3)[0]
    assert abs(e_full - e_mut) > 1e-3


def test_degree_two_anchor_is_clamped_and_the_clamp_is_recorded():
    pos, phi, psi, contacts = _peptide_like(seed=8)
    rt = _runtime(phi, psi, contacts, degree=2)
    e, _, meta = _energy_forces(pos, phi, psi, contacts, rt, 40.0 * KJ_PER_KCAL, 0.3)
    assert meta["anchor_clamp"] == list(rt.fit.anchor_clamp)
    assert np.isfinite(e) and np.isfinite(residual_cv2_from_positions_nm(pos, rt, phi, psi, contacts))


def test_a_topology_whose_torsions_differ_from_the_schema_is_refused():
    pos, phi, psi, contacts = _peptide_like(seed=9)
    rt = _runtime(phi, psi, contacts)
    swapped = [phi[1], phi[0]]                      # same width, permuted
    system = openmm.System()
    for _ in range(len(pos)):
        system.addParticle(12.0)
    with pytest.raises(RuntimeError, match="feature schema"):
        _add_residual_torsion_cv_force(openmm, system, swapped, psi, contacts, rt, _Args(), force_group=29)


def test_a_production_run_with_a_different_contact_definition_is_refused():
    pos, phi, psi, contacts = _peptide_like(seed=10)
    rt = _runtime(phi, psi, contacts)

    class Other(_Args):
        contact_r0_a = 10.0

    system = openmm.System()
    for _ in range(len(pos)):
        system.addParticle(12.0)
    with pytest.raises(RuntimeError, match="anchor"):
        _add_residual_torsion_cv_force(openmm, system, phi, psi, contacts, rt, Other(), force_group=29)


def test_a_different_contact_pair_list_is_refused_even_with_identical_parameters():
    pos, phi, psi, contacts = _peptide_like(seed=13)
    rt = _runtime(phi, psi, contacts)
    other_pairs = contacts[:-1] + [(4, 11, 1.0)]
    system = openmm.System()
    for _ in range(len(pos)):
        system.addParticle(12.0)
    with pytest.raises(RuntimeError, match="anchor"):
        _add_residual_torsion_cv_force(openmm, system, phi, psi, other_pairs, rt, _Args(), force_group=29)


def test_reconstructed_bias_equals_the_bias_the_context_actually_applied():
    """u_nk rebuilt from recorded (cv1, cv2) and the CSV's kcal stiffness must equal the umbrella
    energy OpenMM evaluated. A unit slip anywhere (a stray 4.184, a kJ CSV, a different norm)
    fails this; the energy-vs-evaluator test above cannot see it."""
    from gareus.cv import nonlocal_contact_cv_from_positions_nm
    from gareus.query import reconstruct_bias_matrix

    pos, phi, psi, contacts = _peptide_like(seed=12)
    rt = _runtime(phi, psi, contacts, degree=1)
    k2_kcal, c2 = 30.0, -0.4
    e_kj, _, _ = _energy_forces(pos, phi, psi, contacts, rt, k2_kcal * KJ_PER_KCAL, c2)
    z1 = nonlocal_contact_cv_from_positions_nm(pos, contacts, _Args())
    z2 = residual_cv2_from_positions_nm(pos, rt, phi, psi, contacts)
    beta = 1.0 / (0.0083144626 * 300.0)            # mol/kJ, as reconstruct_bias_matrix expects
    windows = [{"window_id": 0, "center1": z1, "k1": 0.0, "center2": c2, "k2": k2_kcal, "gamd_lambda": 0.0}]
    u = reconstruct_bias_matrix(np.array([z1]), np.array([z2]), windows, beta)
    assert abs(u[0, 0] / beta - e_kj) <= 1e-6 * max(1.0, e_kj)


def test_zero_stiffness_state_still_exposes_the_cv_value():
    pos, phi, psi, contacts = _peptide_like(seed=11)
    rt = _runtime(phi, psi, contacts)
    e, f, meta = _energy_forces(pos, phi, psi, contacts, rt, 0.0, 0.0)
    assert e == 0.0 and np.allclose(f, 0.0)
    assert np.isfinite(residual_cv2_from_positions_nm(pos, rt, phi, psi, contacts))
    assert meta["mode"] == "residual-torsion-pc" and meta["pair_model_sha256"] == "f" * 64
    assert "_runtime" in meta and "runtime" not in meta and "_args" not in meta


def test_the_scorer_reproduces_the_force_from_metadata_alone():
    """secondary_structure_score_from_positions_nm sees only the (JSON-safe + _runtime) metadata."""
    from gareus.cv import secondary_structure_score_from_positions_nm

    pos, phi, psi, contacts = _peptide_like(seed=14)
    rt = _runtime(phi, psi, contacts)
    _, _, meta = _energy_forces(pos, phi, psi, contacts, rt, 1.0, 0.0)
    scored = secondary_structure_score_from_positions_nm(pos, meta)
    assert np.isclose(scored, residual_cv2_from_positions_nm(pos, rt, phi, psi, contacts))


@pytest.mark.parametrize("degree", [1, 2])
def test_fast_subcv_reconstruction_matches_force_and_numpy_oracles(degree):
    """Production sampling/exchange must use the exact CV that OpenMM biases."""
    pos, phi, psi, contacts = _peptide_like(seed=21 + degree)
    rt = _runtime(phi, psi, contacts, degree=degree)
    k2_kcal, c2 = 37.0, -0.25
    e_kj, _, meta, subcvs = _energy_forces(
        pos, phi, psi, contacts, rt, k2_kcal * KJ_PER_KCAL, c2, return_subcvs=True)

    resumed_meta = json.loads(json.dumps({key: value for key, value in meta.items()
                                          if not key.startswith("_")}))
    fast = _ss_scalar_from_sub_cv_values(subcvs, resumed_meta)
    oracle = residual_cv2_from_positions_nm(pos, rt, phi, psi, contacts)
    reconstructed_energy = 0.5 * k2_kcal * KJ_PER_KCAL * (fast - c2) ** 2

    assert np.isclose(fast, oracle, rtol=1e-12, atol=1e-12)
    assert np.isclose(reconstructed_energy, e_kj, rtol=1e-12, atol=1e-12)


def test_residual_fast_path_refuses_legacy_or_incomplete_metadata():
    """Fail closed instead of silently interpreting residual children as alpha/beta scores."""
    pos, phi, psi, contacts = _peptide_like(seed=24)
    rt = _runtime(phi, psi, contacts, degree=2)
    _, _, meta, subcvs = _energy_forces(
        pos, phi, psi, contacts, rt, 1.0, 0.0, return_subcvs=True)

    legacy = dict(meta)
    legacy.pop("scalar_reconstruction")
    with pytest.raises(RuntimeError, match="lacks scalar_reconstruction"):
        _ss_scalar_from_sub_cv_values(subcvs, legacy)

    malformed = dict(meta)
    malformed["scalar_reconstruction"] = dict(meta["scalar_reconstruction"])
    malformed["scalar_reconstruction"]["sub_cv_names"] = ["sum_sin_phi", "res_contacts"]
    with pytest.raises(RuntimeError, match="count"):
        _ss_scalar_from_sub_cv_values(subcvs, malformed)
