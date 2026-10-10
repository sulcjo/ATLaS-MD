from __future__ import annotations

import numpy as np
import pytest
from openmm import app

from gareus.auxiliary_cv.sidechain_model import SidechainModel
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.atom_mapping import _digest
from gareus.correctness._io import digest, json_bytes


def payload():
    atom_keys = ([["A", "1", "", "SER", name, element] for name, element in
                 [('N','N'),('CA','C'),('CB','C'),('OG','O')]] +
                 [["A", "2", "", "SER", name, element] for name, element in
                  [('N','N'),('CA','C'),('CB','C'),('OG','O')]])
    bonds = [[0,1],[1,2],[2,3],[4,5],[5,6],[6,7]]
    features = []
    for base, rid, residue_index in ((0, '1', 0), (4, '2', 1)):
        for trig in ('sin', 'cos'):
            features.append({'name': f'chi1_A:{rid}_SER_{trig}',
                             'orbit': [[base,base+1,base+2,base+3]], 'trig': trig,
                             'harmonic': 1, 'sign': 1, 'family': 'sidechain', 'chain_id': 'A',
                             'residue_id': rid, 'insertion_code': '', 'residue_index': residue_index,
                             'residue_name': 'SER', 'template': 'SER', 'chi_index': 1})
    return {
        'schema': 'atlas-aux-cv-model-v2',
        'dictionary_version': 'atlas-aux-sidechain-chi12-v1',
        'topology_sha256': _digest({'atom_keys': atom_keys, 'bonds': bonds}),
        'system_sha256': 'b' * 64,
        'atom_keys': atom_keys, 'bonds': bonds,
        'features': features,
        'coefficients': [0.75, 0.0, 0.0, 0.0], 'offset': 0.2, 'scale': 1.3,
        'periodic_imaging': 'none', 'units': 'dimensionless',
        'dictionary_exclusions': [], 'provenance': {'training_phase': 'epoch-0'},
    }


def test_v2_roundtrip_identity_and_kernel_evaluation():
    model = SidechainModel.from_mapping(payload())
    restored = SidechainModel.from_mapping(model.to_mapping())
    assert restored == model
    assert restored.model_sha256 == model.model_sha256
    assert model.width == 4
    assert model.projection.quads == ((0,1,2,3),)
    assert (4,5,6,7) not in model.projection.quads
    xyz = np.random.default_rng(11).normal(size=(len(model.atom_keys), 3))
    assert model.values(xyz) == pytest.approx(model.projection.values(xyz))
    assert model.values(xyz) != pytest.approx(model.offset / model.scale)


@pytest.mark.parametrize('mutate', [
    lambda p: p.update(schema='future'),
    lambda p: p.update(extra=1),
    lambda p: p.update(scale=True),
    lambda p: p.update(coefficients=[0.0, 0.0, 0.0, 0.0]),
    lambda p: p.update(periodic_imaging='minimum-image'),
])
def test_v2_rejects_bad_identity_and_unsupported_inputs(mutate):
    raw = payload()
    mutate(raw)
    with pytest.raises(ValueError):
        SidechainModel.from_mapping(raw)


def test_v2_rejects_bad_feature_orbit_and_atom_identity():
    raw = payload()
    raw['features'][0]['orbit'][0][0] = len(raw['atom_keys'])
    with pytest.raises(ValueError):
        SidechainModel.from_mapping(raw)
    raw = payload()
    raw['features'][0]['orbit'].append(list(raw['features'][0]['orbit'][0]))
    with pytest.raises(ValueError):
        SidechainModel.from_mapping(raw)


def test_aux_model_facade_dispatches_v2_and_v1_hash_pin_stays_fixed():
    from aux_cv_fixture import model_payload
    v1 = AuxModel.from_mapping(model_payload([(0, 1, 2, 3), (1, 2, 3, 4)],
                                             [0.5, -0.25, 0.0, 1.0], offset=0.3, scale=3.181))
    assert v1.model_sha256 == 'df70d034cd6f461dfff91351d727005254b8e4fc47b8c4b6103f2aa386c06324'
    assert digest(json_bytes(v1.to_mapping())) == 'ad4f1b6ff4294627773bb514bf5959c72fdfa6aa4e319ed7b3a0cdc5b65c5896'
    assert v1.feature_schema.sha256 == '27c02dc975c46dfbee94d8080d52cf2e37a8b7124db021815205fa9bf274ff51'
    model = SidechainModel.from_mapping(payload())
    assert isinstance(AuxModel.from_mapping(payload()), SidechainModel)
    assert model.model_sha256 == AuxModel.from_mapping(payload()).model_sha256


def test_topology_identity_binds_atom_keys_and_connectivity():
    top = app.Topology()
    chain = top.addChain('A')
    for resid in ('1', '2'):
        residue = top.addResidue('SER', chain, resid)
        atoms = [top.addAtom(name, element, residue) for name, element in
                 [('N', app.element.nitrogen), ('CA', app.element.carbon),
                  ('CB', app.element.carbon), ('OG', app.element.oxygen)]]
        for a, b in zip(atoms, atoms[1:]):
            top.addBond(a, b)
    raw = payload()
    model = SidechainModel.from_mapping(raw)
    model.validate_topology(top)
    top._bonds.pop()
    with pytest.raises(ValueError, match='topology'):
        model.validate_topology(top)


def test_model_emission_freezes_dictionary_and_roundtrips_file(tmp_path):
    from test_aux_sidechain_dictionary import residue_fixture
    from gareus.auxiliary_cv.sidechain_dictionary import build_sidechain_dictionary
    topology, system = residue_fixture('SER')
    dictionary = build_sidechain_dictionary(topology, system)
    model = SidechainModel.from_dictionary(dictionary, [0.4, 0.0], topology=topology,
                                           provenance={'family': 'sidechain'})
    assert model.dictionary_version == dictionary.version
    assert model.features[1].name.endswith('_cos')
    assert model.active_feature_indices == (0,)
    path = tmp_path / 'model-v2.json'
    model.write(path)
    assert SidechainModel.load(path) == model
    model.validate_topology(topology)


def test_model_rejects_feature_atom_identity_mismatch():
    raw = payload()
    raw['atom_keys'][3][4] = 'XX'
    raw['topology_sha256'] = _digest({'atom_keys': raw['atom_keys'], 'bonds': raw['bonds']})
    with pytest.raises(ValueError, match='atom names'):
        SidechainModel.from_mapping(raw)


def test_v2_backbone_feature_uses_same_compiled_projection():
    keys = [['A','0','','ALA','C','C'], ['A','1','','SER','N','N'],
            ['A','1','','SER','CA','C'], ['A','1','','SER','C','C'],
            ['A','2','','GLY','N','N']]
    bonds = [[0,1],[1,2],[2,3],[3,4]]
    raw = {'schema': 'atlas-aux-cv-model-v2', 'dictionary_version': 'atlas-aux-sidechain-chi12-v1',
           'topology_sha256': _digest({'atom_keys': keys, 'bonds': bonds}), 'system_sha256': '',
           'atom_keys': keys, 'bonds': bonds,
           'features': [{'name': 'phi_SER1', 'orbit': [[0,1,2,3]], 'trig': 'sin',
                         'harmonic': 1, 'sign': 1, 'family': 'backbone', 'chain_id': 'A',
                         'residue_id': '1', 'insertion_code': '', 'residue_index': 1,
                         'residue_name': 'SER', 'template': 'SER', 'chi_index': 0}],
           'coefficients': [0.2], 'offset': 0.0, 'scale': 1.0, 'periodic_imaging': 'none',
           'units': 'dimensionless', 'dictionary_exclusions': []}
    model = SidechainModel.from_mapping(raw)
    assert model.projection.quads == ((0,1,2,3),)
