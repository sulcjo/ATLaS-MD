"""F01 acceptance: the coordinate production records and exchanges on IS the one the force applies.

Review finding I01: `_ss_scalar_from_sub_cv_values` had no residual-torsion-pc branch and fell
through to the legacy two-term average, so the sample writer and the exchange kernel used a
different coordinate from the CustomCVForce (review fixture: z 0.8396 vs 0.0676; umbrella
1.4559 vs 0.2700 kJ/mol). These tests build real OpenMM Reference contexts and call the same
module-level observation and bias-assembly functions the production closures now call.

Tolerances (spec F01, Reference platform): CV atol 1e-10 / rtol 1e-9; energy atol 1e-6 kJ/mol
/ rtol 1e-8.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

from fixtures.thermodynamic_repair import (REVIEW_CONTEXT_ENERGY_KJ, REVIEW_FAULTY_ENERGY_KJ,  # noqa: E402
                                           REVIEW_FAULTY_Z, REVIEW_TRUE_Z, residual_degree2_case,
                                           residual_fast_path_case)
import gareus.production as P  # noqa: E402
from gareus.cv import residual_cv2_from_positions_nm  # noqa: E402
from gareus.kernel_identity import RESIDUAL_EVALUATOR_VERSION  # noqa: E402

CV_ATOL, CV_RTOL = 1e-10, 1e-9
E_ATOL, E_RTOL = 1e-6, 1e-8
BETA = 1.0 / (0.008314462618 * 300.0)


def _legacy_two_term_average(sub_cv_values, metadata):
    """The pre-repair fall-through, kept only as the mandatory mutation."""
    arr = np.asarray(sub_cv_values, dtype=np.float64)
    return float(0.5 * (arr[0] + arr[1]))


def _build(case):
    """System + Reference context carrying the residual umbrella (group 29) and a primary
    contact CustomCVForce (group 31) whose sub-variable is the raw contact sum."""
    system = openmm.System()
    for _ in range(case.n_particles):
        system.addParticle(case.particle_mass)
    meta = P._add_residual_torsion_cv_force(openmm, system, case.phi, case.psi, case.pairs, case.runtime,
                                            case.args, force_group=29)
    ss_force = system.getForce(system.getNumForces() - 1)
    r0_nm, beta_nm = float(case.args.contact_r0_a) * 0.1, float(case.args.contact_beta_a_inv) * 10.0
    contact = openmm.CustomBondForce(f"contact_weight*0.5*(1-tanh(0.5*{beta_nm:.17g}*(r-{r0_nm:.17g})))")
    contact.addPerBondParameter("contact_weight")
    for a, b, w in case.pairs:
        contact.addBond(int(a), int(b), [float(w)])
    primary = openmm.CustomCVForce("0*contact_norm*contact_sum")
    primary.addGlobalParameter("contact_norm", float(case.runtime.anchor_definition["norm"]))
    primary.addCollectiveVariable("contact_sum", contact)
    primary.setForceGroup(31)
    system.addForce(primary)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(case.positions_nm * unit.nanometer)
    return system, ctx, ss_force, primary, meta


def _umbrella_energy_kj(ctx, k_kj, center):
    ctx.setParameter("ss_k", float(k_kj))
    ctx.setParameter("ss0", float(center))
    return float(ctx.getState(getEnergy=True, groups={29}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))


def test_review_fixture_fast_scalar_equals_positions_evaluator_and_context_energy():
    case = residual_fast_path_case()
    system, ctx, ss_force, primary, meta = _build(case)
    sub = ss_force.getCollectiveVariableValues(ctx)
    z_fast = P._ss_scalar_from_sub_cv_values(sub, meta)
    z_pos = residual_cv2_from_positions_nm(case.positions_nm, case.runtime, case.phi, case.psi, case.pairs)
    assert z_fast == pytest.approx(REVIEW_TRUE_Z, abs=1e-9)
    assert z_pos == pytest.approx(z_fast, abs=CV_ATOL, rel=CV_RTOL)
    e_ctx = _umbrella_energy_kj(ctx, case.k_kj, case.center)
    assert e_ctx == pytest.approx(REVIEW_CONTEXT_ENERGY_KJ, abs=1e-8)
    assert 0.5 * case.k_kj * (z_fast - case.center) ** 2 == pytest.approx(e_ctx, abs=E_ATOL, rel=E_RTOL)
    # the shared observation function is what production calls
    cv, ss = P.observe_fast_path(ctx, primary, ss_force, case.args, meta)
    assert ss == pytest.approx(z_fast, abs=CV_ATOL)
    assert cv == pytest.approx(sub[-1] / case.runtime.anchor_definition["norm"], abs=CV_ATOL)


def test_mandatory_mutation_the_legacy_two_term_average_fails_the_energy_check(monkeypatch):
    case = residual_fast_path_case()
    system, ctx, ss_force, primary, meta = _build(case)
    sub = ss_force.getCollectiveVariableValues(ctx)
    z_bad = _legacy_two_term_average(sub, meta)
    assert z_bad == pytest.approx(REVIEW_FAULTY_Z, abs=1e-9)          # the fixture has teeth
    e_ctx = _umbrella_energy_kj(ctx, case.k_kj, case.center)
    assert 0.5 * case.k_kj * (z_bad - case.center) ** 2 == pytest.approx(REVIEW_FAULTY_ENERGY_KJ, abs=1e-8)
    assert abs(0.5 * case.k_kj * (z_bad - case.center) ** 2 - e_ctx) > 1.0
    # restoring the fall-through in the dispatch makes the production observation disagree
    monkeypatch.setattr(P, "_ss_scalar_from_sub_cv_values", _legacy_two_term_average)
    _, ss_mutated = P.observe_fast_path(ctx, primary, ss_force, case.args, meta)
    assert abs(0.5 * case.k_kj * (ss_mutated - case.center) ** 2 - e_ctx) > 1.0


@pytest.mark.parametrize("states", [
    # (center, k_kJ): unequal k, negative and positive centres, an unrestrained state
    [(0.3, 10.0), (-0.5, 25.0), (0.9, 0.0), (0.0, 3.5)],
])
def test_shared_bias_assembly_matches_independent_context_energies_swaps_and_gibbs(states):
    case = residual_fast_path_case()
    system, ctx, ss_force, primary, meta = _build(case)
    configs = [case.positions_nm, case.positions2_nm]
    # independent oracle: set each state's parameters on the context and read the energy
    e_ctx = np.zeros((len(states), len(configs)))
    z_fast = np.zeros(len(configs))
    for r, pos in enumerate(configs):
        ctx.setPositions(pos * unit.nanometer)
        for k, (c, kk) in enumerate(states):
            e_ctx[k, r] = _umbrella_energy_kj(ctx, kk, c)
        _, z_fast[r] = P.observe_fast_path(ctx, primary, ss_force, case.args, meta)
        assert z_fast[r] == pytest.approx(
            residual_cv2_from_positions_nm(pos, case.runtime, case.phi, case.psi, case.pairs), abs=CV_ATOL, rel=CV_RTOL)
    centers = np.array([c for c, _ in states])
    ks_kcal = np.array([kk for _, kk in states]) / 4.184
    zeros = np.zeros(len(states))
    dist_kcal, ss_kcal = P.umbrella_bias_matrix_kcal(np.zeros(len(configs)), z_fast, zeros, zeros, centers, ks_kcal)
    _, bias_kj = P.assemble_bias_matrices(dist_kcal, ss_kcal, np.zeros_like(dist_kcal))
    assert bias_kj.shape == (len(states), len(configs))                 # [state, replica]
    np.testing.assert_allclose(bias_kj, e_ctx, atol=E_ATOL, rtol=E_RTOL)
    # neighbour swap: B_j(x) + B_i(y) - B_i(x) - B_j(y) from the matrix vs from context energies
    for i in range(len(states)):
        for j in range(len(states)):
            d_matrix = bias_kj[j, 0] + bias_kj[i, 1] - bias_kj[i, 0] - bias_kj[j, 1]
            d_ctx = e_ctx[j, 0] + e_ctx[i, 1] - e_ctx[i, 0] - e_ctx[j, 1]
            assert d_matrix == pytest.approx(d_ctx, abs=E_ATOL, rel=E_RTOL)
    # Gibbs walk: normalised conditional probabilities of each configuration over states
    for r in range(len(configs)):
        w_matrix = np.exp(-BETA * (bias_kj[:, r] - bias_kj[:, r].min()))
        w_ctx = np.exp(-BETA * (e_ctx[:, r] - e_ctx[:, r].min()))
        np.testing.assert_allclose(w_matrix / w_matrix.sum(), w_ctx / w_ctx.sum(), atol=1e-9, rtol=1e-8)


def test_degree_two_clamp_inside_and_outside_matches_force_and_positions():
    case = residual_degree2_case(clamp=(-1.0, 1.0))
    system, ctx, ss_force, primary, meta = _build(case)
    assert meta["residual_scalar"]["transform"] == "hard_clip"
    seen_outside = False
    for scale in (1.0, 0.7, 1.4):                        # squeeze/stretch the chain to move the anchor
        pos = (case.positions_nm - case.positions_nm.mean(0)) * scale + case.positions_nm.mean(0)
        ctx.setPositions(pos * unit.nanometer)
        sub = ss_force.getCollectiveVariableValues(ctx)
        a = (sub[-1] / meta["residual_scalar"]["norm"] - meta["residual_scalar"]["anchor_mean"]) / meta["residual_scalar"]["anchor_std"]
        seen_outside = seen_outside or not (-1.0 <= a <= 1.0)
        z_fast = P._ss_scalar_from_sub_cv_values(sub, meta)
        z_pos = residual_cv2_from_positions_nm(pos, case.runtime, case.phi, case.psi, case.pairs)
        assert z_fast == pytest.approx(z_pos, abs=CV_ATOL, rel=CV_RTOL)
        e_ctx = _umbrella_energy_kj(ctx, 7.0, -0.4)
        assert 0.5 * 7.0 * (z_fast + 0.4) ** 2 == pytest.approx(e_ctx, abs=E_ATOL, rel=E_RTOL)
    assert seen_outside, "fixture never left the clamp; widen the scale sweep"


def test_missing_psi_block_is_supported_and_consistent():
    case = residual_fast_path_case()
    case.psi = []
    from gareus.cv_selection.models import PairModelRuntime, ResidualFit
    v = case.fit.right_vectors[:, :4]
    fit = ResidualFit(case.fit.coefficients[:, :4], np.zeros(4), np.array([1.0]), v, 0.5, 0.2, (-3.0, 3.0),
                      np.array([0.0]), np.array([0.7]), 1)
    case.fit = fit
    case.runtime = PairModelRuntime(fit, 1, case.runtime.anchor_kind, case.runtime.anchor_definition, "d" * 64,
                                    tuple(case.phi))
    system, ctx, ss_force, primary, meta = _build(case)
    assert [r["name"] for r in meta["subcv_roles"]] == ["sum_sin_phi", "sum_cos_phi", "res_contacts"]
    z_fast = P._ss_scalar_from_sub_cv_values(ss_force.getCollectiveVariableValues(ctx), meta)
    z_pos = residual_cv2_from_positions_nm(case.positions_nm, case.runtime, case.phi, [], case.pairs)
    assert z_fast == pytest.approx(z_pos, abs=CV_ATOL, rel=CV_RTOL)
    e_ctx = _umbrella_energy_kj(ctx, 4.0, 0.1)
    assert 0.5 * 4.0 * (z_fast - 0.1) ** 2 == pytest.approx(e_ctx, abs=E_ATOL, rel=E_RTOL)


def test_dispatch_refuses_unknown_modes_bad_counts_and_nonfinite_values():
    case = residual_fast_path_case()
    system, ctx, ss_force, primary, meta = _build(case)
    sub = list(ss_force.getCollectiveVariableValues(ctx))
    with pytest.raises(ValueError, match="no fast-path scalar"):
        P._ss_scalar_from_sub_cv_values(sub, {"mode": "bogus-mode", "enabled": True})
    with pytest.raises(ValueError, match="sub-CVs, metadata declares"):
        P._ss_scalar_from_sub_cv_values(sub[:-1], meta)
    with pytest.raises(ValueError, match="not a finite vector"):
        P._ss_scalar_from_sub_cv_values(sub[:-1] + [float("nan")], meta)
    with pytest.raises(ValueError, match="expects 2 sub-CVs"):
        P._ss_scalar_from_sub_cv_values([0.1, 0.2, 0.3], {"mode": "alpha", "enabled": True})
    with pytest.raises(ValueError, match="expects 4 sub-CVs"):
        P._ss_scalar_from_sub_cv_values([0.1, 0.2], {"mode": "alpha-coil-beta", "enabled": True})
    stripped = {k: v for k, v in meta.items() if k not in ("subcv_roles", "residual_scalar")}
    with pytest.raises(ValueError, match="no subcv_roles/residual_scalar"):
        P._ss_scalar_from_sub_cv_values(sub, stripped)
    # legacy modes keep their formula, by name only
    assert P._ss_scalar_from_sub_cv_values([0.2, 0.4], {"mode": "alpha", "enabled": True}) == pytest.approx(0.3)


def test_fast_path_activation_is_an_allowlist_not_a_capability_probe():
    case = residual_fast_path_case()
    system, ctx, ss_force, primary, meta = _build(case)
    assert P.fast_cv_path_supported(meta) is True
    assert P.fast_cv_path_supported({"enabled": False}) is True
    assert P.fast_cv_path_supported({"enabled": True, "mode": "alpha"}) is True
    assert P.fast_cv_path_supported({"enabled": True, "mode": "residual-torsion-pc"}) is False      # affected kernel's metadata
    stale = dict(meta); stale["cv_evaluator_version"] = "something_else"
    assert P.fast_cv_path_supported(stale) is False
    assert P.fast_cv_path_supported({"enabled": True, "mode": "never-heard-of-it"}) is False


def test_metadata_survives_json_and_declares_the_evaluator_version():
    from gareus.io import _json_ready
    case = residual_fast_path_case()
    system, ctx, ss_force, primary, meta = _build(case)
    assert meta["cv_evaluator_version"] == RESIDUAL_EVALUATOR_VERSION
    plain = _json_ready(meta)
    assert "_runtime" not in plain and plain["subcv_roles"] == meta["subcv_roles"]
    z_from_json = P._ss_scalar_from_sub_cv_values(ss_force.getCollectiveVariableValues(ctx), plain)
    assert z_from_json == pytest.approx(REVIEW_TRUE_Z, abs=1e-9)


def test_resume_restores_residual_artifact_paths_and_refuses_the_affected_kernel(tmp_path):
    import types
    for name in ("m.json", "c.json", "f.json"):
        (tmp_path / name).write_text("{}")
    meta = {"enabled": True, "mode": "residual-torsion-pc", "cv_evaluator_version": RESIDUAL_EVALUATOR_VERSION,
            "pair_model_path": "m.json", "candidate_set_path": "c.json", "feature_schema_path": "f.json"}
    args = types.SimpleNamespace(secondary_cv="auto")
    P._restore_secondary_cv_args_from_metadata(args, meta, out_dir=tmp_path)
    assert args.secondary_cv == "residual-torsion-pc"
    assert args.secondary_cv_model == str(tmp_path / "m.json")
    assert args.secondary_cv_feature_schema == str(tmp_path / "f.json")
    affected = dict(meta); affected.pop("cv_evaluator_version")
    with pytest.raises(RuntimeError, match="unverified CV evaluator"):
        P._restore_secondary_cv_args_from_metadata(types.SimpleNamespace(), affected, out_dir=tmp_path)
    missing = dict(meta); missing["pair_model_path"] = "gone.json"
    with pytest.raises(RuntimeError, match="missing artifact"):
        P._restore_secondary_cv_args_from_metadata(types.SimpleNamespace(), missing, out_dir=tmp_path)


def test_npt_effective_potential_adapter_sees_exactly_the_residual_umbrella():
    """Spec F01 acceptance item 6: the U* adapter's bias term changes by the same umbrella
    energy the force applies and the fast path reconstructs (solvated GA dipeptide, Pep-GaMD
    lower-dual integrator, Reference platform)."""
    from pep_gamd_fixture import solvated_dipeptide, _fresh_system
    from gareus import pep_gamd
    from gareus.cv import secondary_structure_torsions
    from gareus.cv_selection.models import PairModelRuntime, ResidualFit, contact_pair_list_digest
    import types
    fx = solvated_dipeptide()
    system = _fresh_system()
    topology = fx["topology"]
    phi, psi = secondary_structure_torsions(topology)
    pep = fx["peptide"]
    pairs = [(int(pep[0]), int(pep[-1]), 1.0)]
    width = 2 * len(phi) + 2 * len(psi)
    rng = np.random.default_rng(11)
    v = rng.normal(size=(1, width)); v /= np.linalg.norm(v)
    fit = ResidualFit(np.vstack([np.full(width, 0.05), np.full(width, 0.02), np.zeros(width)]), np.zeros(width),
                      np.array([1.0]), v, 0.3, 0.1, (-3.0, 3.0), np.array([0.0]), np.array([0.5]), 1)
    args = types.SimpleNamespace(contact_r0_a=12.0, contact_beta_a_inv=3.0, contact_normalize=True,
                                 contact_min_sequence_separation=4, contact_atom_selection="heavy",
                                 run_mode="gamd", gamd_boost_type="pep-gamd-lower-dual")
    definition = {"r0_angstrom": 12.0, "beta_per_angstrom": 3.0, "min_sequence_separation": 4,
                  "atom_selection": "heavy", "normalize": True,
                  "pair_list_sha256": contact_pair_list_digest(pairs), "norm": 1.0}
    runtime = PairModelRuntime(fit, 1, "nonlocal-contact-fraction", definition, "a" * 64, tuple(phi + psi))
    meta = P._add_residual_torsion_cv_force(openmm, system, phi, psi, pairs, runtime, args, force_group=29)
    ss_force = system.getForce(system.getNumForces() - 1)
    pep_gamd.ensure_pep_gamd_partition(system, pep)
    integ = pep_gamd.PepGaMDLowerDualIntegrator(
        pep_gamd.DIHEDRAL_GROUP, bias_force_groups=pep_gamd.pep_gamd_bias_force_groups(system),
        dt=0.002 * unit.picoseconds, ntcmdprep=2, ntcmd=4, ntebprep=2, nteb=4, nstlim=100, ntave=2,
        sigma0p=6.0 * unit.kilocalories_per_mole, sigma0d=6.0 * unit.kilocalories_per_mole,
        collision_rate=1.0 / unit.picoseconds, temperature=300.0 * unit.kelvin)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(fx["positions"])
    adapter = pep_gamd.make_npt_target_adapter(system, integ, args)
    pos = np.asarray(fx["positions"].value_in_unit(unit.nanometer), dtype=float)
    z_pos = residual_cv2_from_positions_nm(pos, runtime, phi, psi, pairs)
    z_fast = P._ss_scalar_from_sub_cv_values(ss_force.getCollectiveVariableValues(ctx), meta)
    assert z_fast == pytest.approx(z_pos, abs=CV_ATOL, rel=CV_RTOL)
    ctx.setParameter("ss_k", 0.0)
    base = adapter.evaluate(ctx, adapter.snapshot(ctx, integ))
    for k_kj, center in ((12.0, z_pos + 0.7), (3.0, -0.25)):
        ctx.setParameter("ss_k", k_kj); ctx.setParameter("ss0", center)
        now = adapter.evaluate(ctx, adapter.snapshot(ctx, integ))
        expected = 0.5 * k_kj * (z_fast - center) ** 2
        assert now.bias_kj_mol - base.bias_kj_mol == pytest.approx(expected, abs=E_ATOL, rel=E_RTOL)
        assert now.effective_kj_mol - base.effective_kj_mol == pytest.approx(expected, abs=E_ATOL, rel=E_RTOL)
        assert now.physical_kj_mol == pytest.approx(base.physical_kj_mol, abs=E_ATOL)


def test_nan_secondary_value_zeroes_that_axis_for_both_paths_instead_of_freezing_exchange():
    centers = np.array([0.0, 1.0]); ks = np.array([2.0, 2.0])
    dist, ss = P.umbrella_bias_matrix_kcal(np.array([0.1, 0.2]), np.array([0.5, float("nan")]),
                                           np.array([0.1, 0.1]), np.array([1.0, 1.0]), centers, ks)
    assert np.isfinite(ss).all() and ss[:, 1].tolist() == [0.0, 0.0]
    assert ss[0, 0] == pytest.approx(0.5 * 2.0 * 0.25) and ss[1, 0] == pytest.approx(0.5 * 2.0 * 0.25)
