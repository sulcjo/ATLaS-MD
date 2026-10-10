import numpy as np
import pytest
import types


@pytest.mark.parametrize("platform", ["Reference", "CPU"])
def test_active_sidechain_bias_is_in_npt_trial_energy(platform):
    import openmm
    from openmm import unit

    from test_aux_sidechain_dictionary import residue_fixture
    from test_pep_gamd_boost import _gamd_kwargs
    from gareus import npt, pep_gamd
    from gareus.auxiliary_cv.evaluate import aux_energy_kj
    from gareus.auxiliary_cv.force import set_aux_parameters
    from gareus.auxiliary_cv.runtime import add_aux_cv_force
    from gareus.auxiliary_cv.runtime import aux_bias_matrix_kcal
    from gareus.auxiliary_cv.sidechain_dictionary import build_sidechain_dictionary
    from gareus.auxiliary_cv.sidechain_model import SidechainModel
    from gareus.auxiliary_cv.state_table import AuxStateTable

    topology, source_system = residue_fixture("PHE", symmetric=True)
    dictionary = build_sidechain_dictionary(topology, source_system)
    assert dictionary.primitives
    coeff = np.zeros(len(dictionary.primitives))
    coeff[0] = 1.0
    model = SidechainModel.from_dictionary(dictionary, coeff, topology=topology)
    positions = np.random.default_rng(401).normal(scale=0.2, size=(source_system.getNumParticles(), 3))
    z = float(model.values(positions)[0])
    center, k_kcal = z + 0.35, 2.0
    table = AuxStateTable(model, (center,), (k_kcal,), (None,))
    cross_state = AuxStateTable(model, (0.0, center, center - 0.2), (0.0, k_kcal, 0.5 * k_kcal),
                                (None, None, None))
    matrix = aux_bias_matrix_kcal(np.array([z]), cross_state)
    assert matrix.shape == (3, 1)
    assert matrix[0, 0] == 0.0
    np.testing.assert_allclose(matrix[1:, 0], [
        0.5 * k_kcal * (z - center) ** 2,
        0.25 * k_kcal * (z - (center - 0.2)) ** 2,
    ], rtol=1e-13, atol=1e-13)

    system = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(source_system))
    pep_gamd.ensure_pep_gamd_partition(system, range(system.getNumParticles()))
    runtime = add_aux_cv_force(openmm, system, table,
                               types.SimpleNamespace(umbrella_force_group=31, secondary_cv_force_group=29))
    integrator = pep_gamd.PepGaMDLowerDualIntegrator(
        pep_gamd.DIHEDRAL_GROUP,
        bias_force_groups=pep_gamd.pep_gamd_bias_force_groups(system),
        **_gamd_kwargs(unit),
    )
    adapter = pep_gamd.PepGamdLowerDualNptTargetAdapter(system, integrator)
    assert runtime.info.force_group in adapter._bias_groups
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName(platform))
    context.setPositions(positions * unit.nanometer)
    set_aux_parameters(context, runtime.info, center=center, k_kcal=k_kcal)
    snapshot = adapter.snapshot(context, integrator)

    before = adapter.evaluate(context, snapshot)
    expected = float(aux_energy_kj(np.array([z]), center, k_kcal)[0])
    assert before.bias_kj_mol == pytest.approx(expected, abs=1e-9)

    molecules = npt._molecules_from_context(context)
    molecule_ids, molecule_sizes = npt._molecule_index_arrays(molecules, system.getNumParticles())
    trial = npt._scale_about_molecule_centroids(positions, molecule_ids, molecule_sizes, 0.015)
    context.setPositions(trial * unit.nanometer)
    moved = adapter.evaluate(context, snapshot)
    z_trial = float(model.values(trial)[0])
    assert z_trial == pytest.approx(z, abs=1e-10)
    assert moved.bias_kj_mol == pytest.approx(float(aux_energy_kj(np.array([z_trial]), center, k_kcal)[0]),
                                               abs=1e-9)

    torsion_atom = model.projection.quads[0][0]
    perturbed = trial.copy()
    perturbed[torsion_atom] += np.array([0.02, -0.015, 0.01])
    context.setPositions(perturbed * unit.nanometer)
    changed = adapter.evaluate(context, snapshot)
    z_changed = float(model.values(perturbed)[0])
    assert abs(z_changed - z_trial) > 1e-5
    assert changed.bias_kj_mol == pytest.approx(
        float(aux_energy_kj(np.array([z_changed]), center, k_kcal)[0]), abs=1e-9)
    del context, integrator
