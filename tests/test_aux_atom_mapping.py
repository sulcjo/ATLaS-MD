from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from openmm import app

from gareus.auxiliary_cv import atom_mapping
from gareus.auxiliary_cv.sidechain_core import Primitive, Projection


def make_topology(order=None, insertion=''):
    top = app.Topology()
    atoms = {}
    water = top.addResidue('HOH', top.addChain('W'), '1')
    atoms['water'] = top.addAtom('O', app.element.oxygen, water)
    for chain in ('A', 'B'):
        residue = top.addResidue('SER', top.addChain(chain), '1', insertionCode=insertion)
        for name in ('N', 'CA', 'CB', 'OG'):
            atoms[chain + name] = top.addAtom(name, app.element.oxygen if name == 'OG' else app.element.carbon, residue)
        for a, b in zip(('N', 'CA', 'CB'), ('CA', 'CB', 'OG')):
            top.addBond(atoms[chain + a], atoms[chain + b])
    if order is None:
        return top
    target = app.Topology()
    residues = {}
    out = {}
    for key in order:
        atom = atoms[key]
        source = atom.residue
        if source.chain.id not in residues:
            residues[source.chain.id] = target.addResidue(source.name, target.addChain(source.chain.id),
                                                         source.id, insertionCode=source.insertionCode)
        out[key] = target.addAtom(atom.name, atom.element, residues[source.chain.id])
    for a, b in top.bonds():
        ia = next(k for k,v in atoms.items() if v == a)
        ib = next(k for k,v in atoms.items() if v == b)
        if ia in out and ib in out:
            target.addBond(out[ia], out[ib])
    return target


def test_nonprefix_reordered_phase_mapping_and_projection():
    production = make_topology()
    order = ['BOG', 'BCB', 'BCA', 'BN', 'AOG', 'ACB', 'ACA', 'AN']
    trajectory = make_topology(order)
    phase = atom_mapping.build_phase_atom_map(production, trajectory, 'phase-2')
    assert phase.production_to_trajectory == (None, 7, 6, 5, 4, 3, 2, 1, 0)
    p = Projection((Primitive(((1,2,3,4),), 'sin'),), (.7,))
    xyz = np.random.default_rng(301).normal(size=(9,3))
    y = xyz[[8,7,6,5,4,3,2,1]]
    assert p.values(xyz) == pytest.approx(p.remap(phase.production_to_trajectory).values(y))
    restored = atom_mapping.PhaseAtomMap.from_mapping(json.loads(json.dumps(phase.to_mapping())))
    assert restored == phase
    restored.verify(production, trajectory, phase_id='phase-2')
    with pytest.raises(ValueError, match='phase'):
        restored.verify(production, trajectory, phase_id='other')


def test_persistence_tamper_and_writer_metadata():
    production = make_topology()
    trajectory = make_topology(['AN','ACA','ACB','AOG'])
    phase = atom_mapping.build_phase_atom_map(production, trajectory, 'p', writer_indices=(1,2,3,4))
    raw = copy.deepcopy(phase.to_mapping()); raw['production_to_trajectory'][1] = 3
    with pytest.raises(ValueError):
        atom_mapping.PhaseAtomMap.from_mapping(raw)
    with pytest.raises(ValueError, match='writer'):
        atom_mapping.build_phase_atom_map(production, trajectory, 'p', writer_indices=(4,3,2,1))
    with pytest.raises(ValueError, match='missing'):
        Projection((Primitive(((5,6,7,8),), 'cos'),), (1.,)).remap(phase.production_to_trajectory)


def test_connectivity_mismatch_and_phase_identity():
    production = make_topology()
    trajectory = make_topology(['AN','ACA','ACB','AOG'])
    trajectory._bonds.pop()
    with pytest.raises(ValueError, match='connectivity'):
        atom_mapping.build_phase_atom_map(production, trajectory, 'p')
    with pytest.raises(ValueError):
        atom_mapping.build_phase_atom_map(production, make_topology(['AN','ACA','ACB','AOG'], insertion='X'), 'p')


@pytest.mark.parametrize('collision', ['chain', 'insertion', 'atom'])
def test_identity_collisions_refused(collision):
    top = make_topology()
    if collision == 'chain':
        list(top.chains())[2].id = 'A'
    elif collision == 'atom':
        list(top.atoms())[2].name = 'N'
    else:
        chain = list(top.chains())[2]
        r = top.addResidue('SER', chain, '1', insertionCode='')
        top.addAtom('N', app.element.carbon, r)
    with pytest.raises(ValueError, match='identity|ambiguous'):
        atom_mapping.build_phase_atom_map(top, top, 'p')


def test_distinct_insertions_with_same_residue_number_map_uniquely():
    topology = app.Topology(); chain = topology.addChain('A')
    for code in ('A', 'B'):
        residue = topology.addResidue('GLY', chain, '9', insertionCode=code)
        topology.addAtom('CA', app.element.carbon, residue)
    phase = atom_mapping.build_phase_atom_map(topology, topology, 'p')
    assert phase.production_to_trajectory == (0,1)
    assert phase.production_atom_keys[0][2] == 'A'
    assert phase.production_atom_keys[1][2] == 'B'


def test_recomputed_checksum_still_refuses_inconsistent_mapping():
    phase = atom_mapping.build_phase_atom_map(make_topology(), make_topology(), 'p')
    raw = phase.to_mapping(); raw['production_to_trajectory'][1] = 2
    raw['sha256'] = atom_mapping._digest({k:v for k,v in raw.items() if k != 'sha256'})
    with pytest.raises(ValueError, match='inconsistent'):
        atom_mapping.PhaseAtomMap.from_mapping(raw)


def test_real_mdtraj_conversion_refuses_lost_insertion_identity():
    import mdtraj as md
    production = make_topology(insertion='X')
    trajectory = md.Topology.from_openmm(production)
    with pytest.raises(ValueError, match='missing|identity'):
        atom_mapping.build_phase_atom_map(production, trajectory, 'p')
    production = make_topology()
    trajectory = md.Topology.from_openmm(production)
    phase = atom_mapping.build_phase_atom_map(production, trajectory, 'p')
    assert phase.production_to_trajectory == tuple(range(9))
