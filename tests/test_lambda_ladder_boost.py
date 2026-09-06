import json, types
import numpy as np
from gareus.imports import import_openmm
from pep_gamd_fixture import solvated_dipeptide, _fresh_system


def _env():
    from gareus.pep_gamd import PepGamdEnvelope
    return PepGamdEnvelope(vmax_total=50.0, vmin_total=-50.0, threshold_total=50.0, k0max_total=0.8,
                           vmax_dih=50.0, vmin_dih=-50.0, threshold_dih=50.0, k0max_dih=0.6)


def test_boost_is_zero_at_lambda_zero_and_monotone_in_lambda():
    from gareus.pep_gamd import pep_gamd_boost_kj
    env = _env()
    assert pep_gamd_boost_kj(10.0, 5.0, 0.0, env) == 0.0
    b = [pep_gamd_boost_kj(10.0, 5.0, lam, env) for lam in (0.1, 0.25, 0.5, 1.0)]
    assert all(x > 0 for x in b) and b == sorted(b)


def test_boost_vanishes_above_threshold():
    from gareus.pep_gamd import pep_gamd_boost_kj
    env = _env()
    assert pep_gamd_boost_kj(60.0, 60.0, 1.0, env) == 0.0   # both channels above E = Vmax


def test_boost_matrix_shape_and_rows():
    from gareus.pep_gamd import pep_gamd_boost_kj, pep_gamd_boost_matrix_kj
    env = _env()
    v_pep = np.array([10.0, 20.0, 30.0]); v_dih = np.array([5.0, 6.0, 7.0]); lams = np.array([0.0, 0.5, 1.0])
    M = pep_gamd_boost_matrix_kj(v_pep, v_dih, lams, env)
    assert M.shape == (3, 3)
    assert np.allclose(M[0], 0.0)
    assert np.isclose(M[2, 1], pep_gamd_boost_kj(20.0, 6.0, 1.0, env))


def test_envelope_from_integrator_globals_roundtrip():
    from gareus.pep_gamd import PepGamdEnvelope
    g = {"Vmax_Total": 50.0, "Vmin_Total": -50.0, "threshold_energy_Total": 50.0, "k0_Total": 0.8,
         "Vmax_Dihedral": 40.0, "Vmin_Dihedral": -40.0, "threshold_energy_Dihedral": 40.0, "k0_Dihedral": 0.6}
    env = PepGamdEnvelope.from_integrator_globals(g)
    assert (env.vmax_total, env.k0max_dih, env.threshold_dih) == (50.0, 0.6, 40.0)


def test_closed_form_matches_the_integrator_at_lambda_one_and_half():
    """Oracle: the integrator computes BoostPotential_* every step. Force a production stage
    with a known envelope and compare."""
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    fx = solvated_dipeptide()
    for lam in (1.0, 0.5):
        system = _fresh_system()
        pep_gamd.ensure_pep_gamd_partition(system, fx["peptide"])
        integ = pep_gamd.PepGaMDLowerDualIntegrator(
            pep_gamd.DIHEDRAL_GROUP, dt=0.002 * unit.picoseconds, ntcmdprep=2, ntcmd=4, ntebprep=2, nteb=4,
            nstlim=100, ntave=2, sigma0p=6.0 * unit.kilocalories_per_mole, sigma0d=6.0 * unit.kilocalories_per_mole,
            collision_rate=1.0 / unit.picoseconds, temperature=300.0 * unit.kelvin)
        ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
        ctx.setPositions(fx["positions"]); ctx.setVelocitiesToTemperature(300.0 * unit.kelvin, 3)
        integ.step(1)                                    # first step of a fresh Context is a no-op move
        ctx.setPositions(fx["positions"])
        v_pep = pep_gamd.peptide_essential_energy_kj(ctx, unit)
        v_dih = ctx.getState(getEnergy=True, groups={pep_gamd.DIHEDRAL_GROUP}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        env = pep_gamd.PepGamdEnvelope(vmax_total=v_pep + 200.0, vmin_total=v_pep - 200.0, threshold_total=v_pep + 200.0, k0max_total=0.8,
                                       vmax_dih=v_dih + 100.0, vmin_dih=v_dih - 100.0, threshold_dih=v_dih + 100.0, k0max_dih=0.6)
        for k, v in {"stepCount": 50, "stage": 5,
                     "k0_Total": lam * env.k0max_total, "Vmax_Total": env.vmax_total, "Vmin_Total": env.vmin_total, "threshold_energy_Total": env.threshold_total,
                     "k0_Dihedral": lam * env.k0max_dih, "Vmax_Dihedral": env.vmax_dih, "Vmin_Dihedral": env.vmin_dih, "threshold_energy_Dihedral": env.threshold_dih}.items():
            integ.setGlobalVariableByName(k, v)
        integ.step(1)
        got = integ.getGlobalVariableByName("BoostPotential_Total") + integ.getGlobalVariableByName("BoostPotential_Dihedral")
        want = pep_gamd.pep_gamd_boost_kj(v_pep, v_dih, lam, env)
        assert abs(got - want) < 1e-3, (lam, got, want)


def test_set_replica_lambda_scales_both_k0_globals():
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    fx = solvated_dipeptide(); system = _fresh_system()
    pep_gamd.ensure_pep_gamd_partition(system, fx["peptide"])
    integ = pep_gamd.PepGaMDLowerDualIntegrator(
        pep_gamd.DIHEDRAL_GROUP, dt=0.002 * unit.picoseconds, ntcmdprep=2, ntcmd=4, ntebprep=2, nteb=4, nstlim=100, ntave=2,
        sigma0p=6.0 * unit.kilocalories_per_mole, sigma0d=6.0 * unit.kilocalories_per_mole,
        collision_rate=1.0 / unit.picoseconds, temperature=300.0 * unit.kelvin)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    k0max = {"Total": 0.8, "Dihedral": 0.6}
    pep_gamd.set_replica_lambda(integ, 0.25, k0max)
    assert abs(integ.getGlobalVariableByName("k0_Total") - 0.2) < 1e-12
    assert abs(integ.getGlobalVariableByName("k0_Dihedral") - 0.15) < 1e-12
    pep_gamd.set_replica_lambda(integ, 0.0, k0max)
    assert integ.getGlobalVariableByName("k0_Total") == 0.0 and integ.getGlobalVariableByName("k0_Dihedral") == 0.0


def test_k0max_from_globals_reads_both_channels():
    from gareus.pep_gamd import k0max_from_globals
    assert k0max_from_globals({"k0_Total": 0.8, "k0_Dihedral": 0.6, "Vmax_Total": 1.0}) == {"Total": 0.8, "Dihedral": 0.6}


def test_set_replica_lambda_rejects_out_of_range():
    from gareus import pep_gamd
    try:
        pep_gamd.set_replica_lambda(types.SimpleNamespace(setGlobalVariableByName=lambda *a: None), 1.2, {"Total": 1.0, "Dihedral": 1.0})
    except ValueError:
        pass
    else:
        raise AssertionError("λ > 1 must be rejected")
