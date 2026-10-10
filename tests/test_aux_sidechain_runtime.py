from __future__ import annotations

import numpy as np
import pytest

from gareus.auxiliary_cv.force import aux_sub_cv_spec, build_aux_force
from gareus.auxiliary_cv.runtime import (AuxRuntime, aux_z_from_force, check_aux_force,
                                         check_aux_geometry)
from gareus.auxiliary_cv.sidechain_model import SidechainModel
from gareus.auxiliary_cv.state_table import AuxStateTable


def model_fixture():
    from test_aux_sidechain_dictionary import residue_fixture
    from gareus.auxiliary_cv.sidechain_dictionary import build_sidechain_dictionary
    topology, system = residue_fixture('PHE', symmetric=True)
    dictionary = build_sidechain_dictionary(topology, system)
    model = SidechainModel.from_dictionary(dictionary, [.7,0.,0.,.2], topology=topology)
    return topology, system, model


def test_v2_sample_basis_maps_complete_orbits_and_observes_projection():
    from gareus.auxiliary_cv.sample_schema import (AUX_SAMPLES_SCHEMA_V2, AuxSampleSchema,
                                                   build_sample_schema, observe_carrier)
    from gareus.auxiliary_cv.sidechain_dictionary import build_sidechain_dictionary
    topology, system, _ = model_fixture()
    dictionary = build_sidechain_dictionary(topology, system)
    model = SidechainModel.from_dictionary(dictionary, [0.,0.,0.,.2], topology=topology)
    schema = build_sample_schema(topology, [], [model])
    assert schema.schema == AUX_SAMPLES_SCHEMA_V2
    assert len(schema.torsion_quads) == 3
    assert len(model.projection.quads) == 2
    restored = AuxSampleSchema.from_payload(schema.to_payload())
    assert restored == schema
    positions = np.random.default_rng(91).normal(size=(topology.getNumAtoms(),3))
    observed = observe_carrier(positions, schema, {model.model_sha256: model})
    assert observed.torsions.shape == (3,)
    assert observed.z[0] == pytest.approx(model.values(positions)[0], abs=1e-12)


def test_v2_sample_angle_map_tamper_refused():
    from gareus.auxiliary_cv.sample_schema import AuxSampleSchema, build_sample_schema
    from gareus.auxiliary_cv.sample_schema import _basis_sha_v2
    topology, _, model = model_fixture()
    schema = build_sample_schema(topology, [], [model])
    raw = schema.to_payload()
    raw['model_angle_maps'][0]['feature_orbit_columns'][0][0] = 1
    maps = list(schema.model_angle_maps[0])
    maps[0] = (1,)
    raw['basis_sha256'] = _basis_sha_v2(schema.torsion_quads, schema.torsion_labels,
                                        schema.model_shas, (tuple(maps),))
    with pytest.raises(ValueError, match='column map'):
        AuxSampleSchema.from_payload(raw).model_basis_index(model)


def test_v2_force_compiler_builds_orbit_weighted_harmonic_subcvs():
    import openmm
    topology, system, model = model_fixture()
    spec = aux_sub_cv_spec(model)
    assert {name for name, _, _ in spec} == {'sc_cos_pos2', 'sc_sin_pos1'}
    by_name = {name: (expr, terms) for name, expr, terms in spec}
    assert by_name['sc_cos_pos2'][0] == 'w*cos(2*theta)'
    assert len(by_name['sc_cos_pos2'][1]) == 2
    assert [weight for _, weight in by_name['sc_cos_pos2'][1]] == [.1, .1]
    force, info = build_aux_force(openmm, model, force_group=30)
    system.addForce(force)
    table = AuxStateTable(model, (0.,), (1.,), (None,))
    runtime = AuxRuntime(table, info, system.getNumForces()-1, model.topology_sha256)
    check_aux_force(force, runtime)


def test_v2_force_observer_matches_projection_on_reference():
    import openmm
    from openmm import unit
    topology, source, model = model_fixture()
    system = openmm.System()
    for i in range(source.getNumParticles()):
        system.addParticle(source.getParticleMass(i))
    force, info = build_aux_force(openmm, model, force_group=30)
    system.addForce(force)
    integrator = openmm.VerletIntegrator(.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName('Reference'))
    positions = np.random.default_rng(46).normal(size=(system.getNumParticles(),3))
    context.setPositions(positions * unit.nanometer)
    context.setParameter(info.global_k, 4.184 * 1.7)
    context.setParameter(info.global_c, .31)
    table = AuxStateTable(model, (.31,), (1.7,), (None,))
    runtime = AuxRuntime(table, info, 0, model.topology_sha256)
    z_force = aux_z_from_force(context, force, runtime)
    assert z_force == pytest.approx(model.values(positions)[0], abs=1e-12)
    energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert energy == pytest.approx(.5 * 4.184 * 1.7 * (z_force - .31)**2, abs=1e-10)
    actual_force = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer)
    _, dz = model.value_gradient(positions)
    expected_force = -(4.184 * 1.7) * (z_force - .31) * dz
    np.testing.assert_allclose(actual_force, expected_force, rtol=2e-9, atol=2e-8)
    check_aux_geometry(positions, runtime)
    del context, integrator
