"""Generic CV1 anchor for the automatic residual CV2 (spec 2026-10-01-generic-cv1-anchor), phase 1.

The residual CV2 is fitted against the run's CV1 ("anchor"). The contact fraction keeps every
byte; the end-to-end (terminal CA--CA) distance in A is the second anchor kind: its sub-CV is
``r`` (nm) with norm 0.1, the force is the positions evaluator, and a pair model never deploys
on a run whose CV1 is a different kind or atom pair.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gareus.cv_selection import anchor_spec as AS
from gareus.cv_selection.models import PairModelRuntime, ResidualFit
from gareus.cv_selection.pair_model import _parse_deployment

KJ_PER_KCAL = 4.184


# ---- anchor_spec -----------------------------------------------------------------------------

def _args(**kw):
    base = dict(primary_cv="distance", cv_mode="terminal-ca", cv_atom1=None, cv_atom2=None, cv1_k_min=0.0,
                cv1_k_max=0.0, contact_adaptive_min_k_kcal=5.0, contact_adaptive_max_k_kcal=1200.0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_kind_follows_the_run_cv1_and_refuses_other_atom_pairs():
    assert AS.anchor_kind_for_args(_args(primary_cv="contacts")) == AS.KIND_CONTACT
    assert AS.anchor_kind_for_args(_args()) == AS.KIND_E2E
    with pytest.raises(ValueError):
        AS.anchor_kind_for_args(_args(cv_atom1="1:CA", cv_atom2="5:CA"))
    with pytest.raises(ValueError):
        AS.anchor_kind_for_args(_args(cv_mode="terminal-n-c"))


def test_trace_values_units_and_k_bounds():
    tr = {"cv1": np.array([0.2, 0.5]), "e2e_nm": np.array([1.0, 1.5])}
    assert np.allclose(AS.trace_values(AS.KIND_CONTACT, tr), [0.2, 0.5])
    assert np.allclose(AS.trace_values(AS.KIND_E2E, tr), [10.0, 15.0])            # A
    assert AS.descriptor_value(AS.KIND_E2E, {"cv1": "0.3", "e2e_nm": "0.9"}) == pytest.approx(9.0)
    assert AS.units(AS.KIND_E2E) == "angstrom" and AS.units(AS.KIND_CONTACT) == "dimensionless"
    assert AS.k_bounds(AS.KIND_CONTACT, _args()) == (5.0, 1200.0)                   # unchanged
    assert AS.k_bounds(AS.KIND_E2E, _args()) == (AS.E2E_DEFAULT_K_MIN, AS.E2E_DEFAULT_K_MAX)
    assert AS.k_bounds(AS.KIND_E2E, _args(cv1_k_min=0.2, cv1_k_max=4.0)) == (0.2, 4.0)
    assert AS.value_bounds(AS.KIND_CONTACT) == (0.0, 1.0) and AS.value_bounds(AS.KIND_E2E) == (0.0, None)


def test_binding_digest_is_stable_and_atom_specific():
    d1, d2 = AS.e2e_definition(4, 130), AS.e2e_definition(4, 131)
    assert AS.binding_digest(AS.KIND_E2E, d1) == AS.binding_digest(AS.KIND_E2E, dict(d1))
    assert AS.binding_digest(AS.KIND_E2E, d1) != AS.binding_digest(AS.KIND_E2E, d2)
    assert AS.binding_digest(AS.KIND_CONTACT, {"norm": 1.0}) is None


# ---- pair-model deployment -------------------------------------------------------------------

H = "a" * 64


def test_contact_deployment_is_unchanged():
    raw = {"topology_sha256": H, "physical_system_sha256": H, "contact_pair_list_sha256": H, "deployable": True}
    assert _parse_deployment(raw) == raw
    with pytest.raises(Exception):
        _parse_deployment({**raw, "contact_pair_list_sha256": None})


def test_anchor_binding_stands_in_for_the_contact_pair_digest():
    raw = {"topology_sha256": H, "physical_system_sha256": H, "contact_pair_list_sha256": None,
           "anchor_binding_sha256": "b" * 64, "deployable": True}
    out = _parse_deployment(raw)
    assert out["anchor_binding_sha256"] == "b" * 64 and out["contact_pair_list_sha256"] is None
    with pytest.raises(Exception):
        _parse_deployment({**raw, "anchor_binding_sha256": None})
    with pytest.raises(Exception):
        _parse_deployment({**raw, "unknown": 1})


def test_selection_binding_uses_the_anchor_field(monkeypatch):
    from gareus.cv_selection import select_pair as sp
    assert sp._units(AS.KIND_E2E) == "angstrom" and sp._units(AS.KIND_CONTACT) == "dimensionless"


# ---- force vs evaluator for the distance anchor ----------------------------------------------

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

from gareus.cv import residual_cv2_from_positions_nm, secondary_structure_score_from_positions_nm  # noqa: E402
from gareus.production import _add_residual_torsion_cv_force  # noqa: E402


def _peptide_like(n_atoms=16, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.cumsum(rng.normal(scale=0.12, size=(n_atoms, 3)), axis=0) + 2.0
    phi = [(0, 1, 2, 3), (4, 5, 6, 7)]
    psi = [(8, 9, 10, 11), (12, 13, 14, 15)]
    return pos, phi, psi


def _e2e_runtime(phi, psi, d=8, degree=2, seed=1, atoms=(0, 15)):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(3, d)) * 0.3
    if degree == 1:
        B[2] = 0.0
    V = np.linalg.qr(rng.normal(size=(d, d)))[0].T
    # anchor mean/std in A: the fixture's end-to-end distance is a few A
    fit = ResidualFit(B, rng.normal(size=d) * 0.1, np.linspace(2, 0.5, d), V, 6.0, 2.0,
                      (-3.0, 3.0), np.zeros(d), np.ones(d), degree)
    return PairModelRuntime(fit, 2, AS.KIND_E2E, AS.e2e_definition(*atoms), "e" * 64, tuple(phi) + tuple(psi))


def _energy_forces(pos, phi, psi, pairs, runtime, ss_k_kj, c2, args=None):
    system = openmm.System()
    for _ in range(len(pos)):
        system.addParticle(12.0)
    meta = _add_residual_torsion_cv_force(openmm, system, phi, psi, pairs, runtime, args or _args(),
                                          force_group=29)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pos * unit.nanometer)
    ctx.setParameter("ss_k", ss_k_kj)
    ctx.setParameter("ss0", c2)
    st = ctx.getState(getEnergy=True, getForces=True, groups={29})
    e = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f = np.asarray(st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    return e, f, meta


@pytest.mark.parametrize("degree", [1, 2])
def test_distance_anchor_force_energy_matches_the_evaluator(degree):
    pos, phi, psi = _peptide_like()
    rt = _e2e_runtime(phi, psi, degree=degree)
    pairs = [(0, 15, 1.0)]
    k2_kcal, c2 = 40.0, 0.3
    e, _, meta = _energy_forces(pos, phi, psi, pairs, rt, k2_kcal * KJ_PER_KCAL, c2)
    z2 = residual_cv2_from_positions_nm(pos, rt, phi, psi, pairs)
    assert abs(e - 0.5 * k2_kcal * KJ_PER_KCAL * (z2 - c2) ** 2) <= 1e-6 * max(1.0, abs(e))
    # the anchor is 10 x the CA--CA distance in nm
    assert AS.value_from_positions(AS.KIND_E2E, pos, pairs, rt) == pytest.approx(
        10 * np.linalg.norm(pos[15] - pos[0]))
    assert meta["anchor_kind"] == AS.KIND_E2E and meta["contact_pairs"] == []
    assert meta["anchor_pairs"] == [[0, 15, 1.0]]
    assert np.isclose(secondary_structure_score_from_positions_nm(pos, meta), z2)   # metadata alone


def test_distance_anchor_forces_match_finite_differences_including_the_chain_rule():
    pos, phi, psi = _peptide_like(seed=5)
    rt = _e2e_runtime(phi, psi, degree=2)
    pairs = [(0, 15, 1.0)]
    _, f, _ = _energy_forces(pos, phi, psi, pairs, rt, 40.0 * KJ_PER_KCAL, 0.3)
    h = 1e-6
    for atom in (0, 3, 9, 15):                 # anchor atoms 0 and 15 carry the chain-rule term
        for ax in range(3):
            p = pos.copy(); p[atom, ax] += h
            m = pos.copy(); m[atom, ax] -= h
            fd = -(_energy_forces(p, phi, psi, pairs, rt, 40.0 * KJ_PER_KCAL, 0.3)[0]
                   - _energy_forces(m, phi, psi, pairs, rt, 40.0 * KJ_PER_KCAL, 0.3)[0]) / (2 * h)
            assert abs(f[atom, ax] - fd) <= 1e-4 * max(1.0, abs(fd)), (atom, ax, f[atom, ax], fd)


def test_the_anchor_term_matters_for_the_distance_anchor():
    pos, phi, psi = _peptide_like(seed=7)
    rt = _e2e_runtime(phi, psi, degree=1)
    pairs = [(0, 15, 1.0)]
    moved = pos.copy(); moved[15] += 0.3 * (pos[15] - pos[0]) / np.linalg.norm(pos[15] - pos[0])
    z_a = residual_cv2_from_positions_nm(pos, rt, phi, psi, pairs)
    z_b = residual_cv2_from_positions_nm(moved, rt, phi, psi, pairs)   # torsions of atoms 12-15 move too,
    assert z_a != z_b                                                  # but the anchor shift is 3 A


def test_a_distance_model_refuses_a_contact_run_and_a_different_atom_pair():
    pos, phi, psi = _peptide_like()
    rt = _e2e_runtime(phi, psi)
    with pytest.raises(RuntimeError, match="anchor kind mismatch"):
        _energy_forces(pos, phi, psi, [(0, 15, 1.0)], rt, 1.0, 0.0, args=_args(primary_cv="contacts"))
    with pytest.raises(RuntimeError, match="atom pair mismatch"):
        _energy_forces(pos, phi, psi, [(0, 14, 1.0)], rt, 1.0, 0.0)


def test_a_contact_model_refuses_a_distance_run():
    from test_cv_selection_forces import _peptide_like as contact_fixture, _runtime as contact_runtime
    pos, phi, psi, contacts = contact_fixture()
    rt = contact_runtime(phi, psi, contacts)
    with pytest.raises(RuntimeError, match="anchor kind mismatch"):
        _energy_forces(pos, phi, psi, contacts, rt, 1.0, 0.0, args=_args())


def test_a_distance_anchor_cannot_carry_the_cv1_umbrella():
    pos, phi, psi = _peptide_like()
    rt = _e2e_runtime(phi, psi)
    system = openmm.System()
    for _ in range(len(pos)):
        system.addParticle(12.0)
    with pytest.raises(ValueError, match="carry the CV1 umbrella"):
        _add_residual_torsion_cv_force(openmm, system, phi, psi, [(0, 15, 1.0)], rt, _args(), force_group=29,
                                       carry_primary_umbrella=True)
