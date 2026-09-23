"""One CustomCVForce carries both umbrellas when CV2 is residual-torsion-pc over a contact CV1.

The residual CV2 force already holds a private copy of the 1,256-pair contact sum
(``res_contacts``). The shared layout appends the CV1 umbrella to that force instead of adding
a second CustomCVForce with its own copy, so the sum is evaluated once per step. Measured on
chignolin_8's system: +14.6 % node ns/day at 236 contexts under MPS (job 2608721).

The bias must be the same function of the coordinates as the split pair, the fast CV path must
read CV1 from the shared force's contact sub-CV (not sub-CV [0], which is a torsion sum), and a
resumed campaign must rebuild the layout its checkpoints were written with.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

from fixtures.thermodynamic_repair import residual_fast_path_case  # noqa: E402
import gareus.production as P  # noqa: E402
from gareus.forces import add_contact_umbrella_force  # noqa: E402

E_ATOL, E_RTOL = 1e-6, 1e-8
F_ATOL = 1e-6
CV_ATOL = 1e-10
WINDOW = {"k": 500.0, "r0": 0.35, "ss_k": 12.0, "ss0": 0.2}
CONTACT_PRIMARY = {"primary_cv": "nonlocal-contacts", "contact_pairs": []}


def _context(system, positions_nm):
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(positions_nm * unit.nanometer)
    return ctx


def _split(case):
    system = openmm.System()
    for _ in range(case.n_particles):
        system.addParticle(case.particle_mass)
    meta = P._add_residual_torsion_cv_force(openmm, system, case.phi, case.psi, case.pairs, case.runtime,
                                            case.args, force_group=29)
    ss_force = system.getForce(system.getNumForces() - 1)
    primary = add_contact_umbrella_force(openmm, system, case.pairs, case.args, force_group=31)
    return system, ss_force, primary, meta


def _shared(case):
    system = openmm.System()
    for _ in range(case.n_particles):
        system.addParticle(case.particle_mass)
    meta = P._add_residual_torsion_cv_force(openmm, system, case.phi, case.psi, case.pairs, case.runtime,
                                            case.args, force_group=29, carry_primary_umbrella=True)
    return system, system.getForce(system.getNumForces() - 1), meta


def _bias(ctx, groups, window=WINDOW):
    for name, value in window.items():
        ctx.setParameter(name, float(value))
    st = ctx.getState(getEnergy=True, getForces=True, groups=set(groups))
    return (st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            np.asarray(st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)))


@pytest.mark.parametrize("which", ["positions_nm", "positions2_nm"])
def test_shared_force_reproduces_the_split_pair_energy_and_forces(which):
    case = residual_fast_path_case()
    pos = getattr(case, which)
    split_sys, _, _, _ = _split(case)
    shared_sys, _, _ = _shared(case)
    e_split, f_split = _bias(_context(split_sys, pos), {29, 31})
    e_shared, f_shared = _bias(_context(shared_sys, pos), {29})
    assert e_shared == pytest.approx(e_split, abs=E_ATOL, rel=E_RTOL)
    np.testing.assert_allclose(f_shared, f_split, atol=F_ATOL)


def test_the_equivalence_check_sees_the_cv1_umbrella():
    """Teeth: the CV1 term is a real, nonzero part of the compared energy."""
    case = residual_fast_path_case()
    ctx = _context(_shared(case)[0], case.positions_nm)
    e_on, _ = _bias(ctx, {29})
    e_off, _ = _bias(ctx, {29}, {**WINDOW, "k": 0.0})
    split_ctx = _context(_split(case)[0], case.positions_nm)
    e_cv1_only, _ = _bias(split_ctx, {31})
    assert e_cv1_only > 1e-3
    assert e_on - e_off == pytest.approx(e_cv1_only, abs=E_ATOL, rel=E_RTOL)


def test_shared_force_carries_the_cv1_parameters_under_their_production_names():
    case = residual_fast_path_case()
    _, force, meta = _shared(case)
    names = {force.getGlobalParameterName(i) for i in range(force.getNumGlobalParameters())}
    assert {"k", "r0", "contact_norm", "ss_k", "ss0"} <= names
    assert meta["cv_force_layout"] == P.SHARED_CONTACT_LAYOUT
    assert _split(case)[3]["cv_force_layout"] == P.SPLIT_CV_LAYOUT


@pytest.mark.parametrize("which", ["positions_nm", "positions2_nm"])
def test_fast_path_reads_cv1_from_the_shared_contact_subcv(which):
    case = residual_fast_path_case()
    pos = getattr(case, which)
    split_sys, split_ss, split_primary, split_meta = _split(case)
    shared_sys, shared_force, shared_meta = _shared(case)
    split_ctx, shared_ctx = _context(split_sys, pos), _context(shared_sys, pos)
    cv_split, ss_split = P.observe_fast_path(split_ctx, split_primary, split_ss, case.args, split_meta)
    cv_shared, ss_shared = P.observe_fast_path(shared_ctx, shared_force, shared_force, case.args, shared_meta)
    assert cv_shared == pytest.approx(cv_split, abs=CV_ATOL)
    assert ss_shared == pytest.approx(ss_split, abs=CV_ATOL)
    # Mutation: the split layout's "sub-CV [0] is the contact sum" rule is wrong here --
    # sub-CV [0] of the shared force is a torsion sum.
    first = shared_force.getCollectiveVariableValues(shared_ctx)[0]
    norm = float(shared_ctx.getParameter("contact_norm"))
    assert abs(first / norm - cv_split) > 1e-3


def test_fast_cv_force_indices_point_both_observers_at_the_shared_force():
    case = residual_fast_path_case()
    split_sys, _, _, split_meta = _split(case)
    shared_sys, _, shared_meta = _shared(case)
    n = split_sys.getNumForces()
    assert P.fast_cv_force_indices(split_sys, 31, 29, split_meta) == (n - 1, n - 2)
    idx = shared_sys.getNumForces() - 1
    assert P.fast_cv_force_indices(shared_sys, 31, 29, shared_meta) == (idx, idx)


@pytest.mark.parametrize("primary, secondary, enabled, layout_attr, expected", [
    (CONTACT_PRIMARY, "residual-torsion-pc", True, None, "shared"),
    (CONTACT_PRIMARY, "residual-torsion-pc", True, "split", "split"),
    ({"primary_cv": "distance", "cv_atom1": 0, "cv_atom2": 1}, "residual-torsion-pc", True, None, "split"),
    (CONTACT_PRIMARY, "tica-linear", True, None, "split"),
    (CONTACT_PRIMARY, "residual-torsion-pc", False, None, "split"),
])
def test_layout_is_shared_only_for_residual_cv2_over_a_contact_cv1(primary, secondary, enabled, layout_attr, expected):
    args = types.SimpleNamespace(secondary_cv=secondary)
    if layout_attr is not None:
        args.cv_force_layout = layout_attr
    want = P.SHARED_CONTACT_LAYOUT if expected == "shared" else P.SPLIT_CV_LAYOUT
    assert P.cv_force_layout(primary, args, secondary_enabled=enabled) == want


@pytest.mark.parametrize("recorded, expected", [
    ("shared_contact_subcv", "shared"),
    ("split", "split"),
    (None, "split"),          # checkpoints written before the shared layout existed
])
def test_resume_rebuilds_the_layout_the_checkpoint_was_written_with(tmp_path, recorded, expected):
    paths = {}
    for key in ("pair_model_path", "candidate_set_path", "feature_schema_path"):
        p = tmp_path / f"{key}.json"
        p.write_text("{}")
        paths[key] = str(p)
    meta = {"enabled": True, "mode": "residual-torsion-pc",
            "cv_evaluator_version": P.RESIDUAL_EVALUATOR_VERSION, **paths}
    if recorded is not None:
        meta["cv_force_layout"] = recorded
    args = types.SimpleNamespace()
    P._restore_secondary_cv_args_from_metadata(args, meta, out_dir=tmp_path)
    assert args.cv_force_layout == expected


def test_fast_indices_ignore_a_shared_layout_record_when_cv2_is_disabled():
    case = residual_fast_path_case()
    split_sys, _, _, split_meta = _split(case)
    n = split_sys.getNumForces()
    stale = {**split_meta, "enabled": False, "cv_force_layout": P.SHARED_CONTACT_LAYOUT}
    assert P.fast_cv_force_indices(split_sys, 31, 29, stale) == (n - 1, n - 2)


@pytest.mark.parametrize("recorded, current, expected", [
    ([29, 31], (29, 31), True),
    ([29], (29,), True),
    ([29, 31], (29,), False),     # exported from a split context, loading into a shared one
    ([29], (29, 31), False),
    (None, (29, 31), False),      # export predates the record: layout unknown, do not load
])
def test_reused_gamd_context_checkpoint_loads_only_into_matching_bias_groups(recorded, current, expected):
    assert P.reusable_checkpoint_matches_bias_groups(recorded, current) is expected


@pytest.mark.parametrize("field, value, message", [
    ("contact_r0_a", float("inf"), "contact-r0-a"),
    ("contact_beta_a_inv", float("nan"), "contact-beta-a-inv"),
    ("contact_beta_a_inv", -3.0, "contact-beta-a-inv"),
])
def test_both_contact_sum_copies_validate_their_switching_constants(field, value, message):
    from gareus.forces import contact_switch_constants_nm
    args = types.SimpleNamespace(contact_r0_a=12.0, contact_beta_a_inv=3.0)
    assert contact_switch_constants_nm(args) == pytest.approx((1.2, 30.0))
    setattr(args, field, value)
    with pytest.raises(ValueError, match=message):
        contact_switch_constants_nm(args)
