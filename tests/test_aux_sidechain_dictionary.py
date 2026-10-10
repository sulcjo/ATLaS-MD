from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import openmm
from openmm import app, unit

from gareus.auxiliary_cv import sidechain_dictionary as dictionary
from gareus.auxiliary_cv.sidechain_core import Projection

EXPECTED = {
    'ALA': (), 'GLY': (), 'ARG': ('CG', 'CD'), 'ASN': ('CG', 'OD1'),
    'ASP': ('CG', 'OD1'), 'ASH': ('CG', 'OD1'), 'CYS': ('SG',),
    'CYX': ('SG',), 'GLN': ('CG', 'CD'), 'GLU': ('CG', 'CD'),
    'GLH': ('CG', 'CD'), 'HID': ('CG', 'ND1'), 'HIE': ('CG', 'ND1'),
    'HIP': ('CG', 'ND1'), 'HIS': ('CG', 'ND1'), 'ILE': ('CG1', 'CD1'),
    'LEU': ('CG', 'CD1'), 'LYS': ('CG', 'CD'), 'LYN': ('CG', 'CD'),
    'MET': ('CG', 'SD'), 'PHE': ('CG', 'CD1'), 'SER': ('OG',),
    'THR': ('OG1',), 'TRP': ('CG', 'CD1'), 'TYR': ('CG', 'CD1'),
    'VAL': ('CG1',),
}


def residue_fixture(name, symmetric=False):
    ff = app.ForceField('amber14/protein.ff14SB.xml')
    source = 'HID' if name == 'HIS' else name
    template = ff._templates[source]
    topology = app.Topology()
    residue = topology.addResidue(name, topology.addChain('A'), '10', insertionCode='B')
    atoms = [topology.addAtom(a.name, a.element, residue) for a in template.atoms]
    for a, b in template.bonds:
        topology.addBond(atoms[a], atoms[b])
    residue_templates = {residue: source}
    if name == 'CYX':
        partner = topology.addResidue('CYX', residue.chain, '11')
        other = [topology.addAtom(a.name,a.element,partner) for a in template.atoms]
        for a,b in template.bonds:
            topology.addBond(other[a],other[b])
        sulfur = next(i for i,a in enumerate(template.atoms) if a.name == 'SG')
        topology.addBond(atoms[sulfur],other[sulfur])
        residue_templates[partner] = 'CYX'
    system = ff.createSystem(topology, ignoreExternalBonds=True, residueTemplates=residue_templates)
    if symmetric:
        by_name = {a.name: a.index for a in atoms}
        pairs = [('OD1', 'OD2')] if name == 'ASP' else [
            ('CD1', 'CD2'), ('CE1', 'CE2'), ('HD1', 'HD2'), ('HE1', 'HE2')]
        perm = list(range(len(atoms)))
        for a, b in pairs:
            perm[by_name[a]], perm[by_name[b]] = by_name[b], by_name[a]
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                terms = [force.getExceptionParameters(i) for i in range(force.getNumExceptions())]
                by_pair = {tuple(sorted(t[:2])): (i,t) for i,t in enumerate(terms)}
                for i, term in enumerate(terms):
                    pair = tuple(sorted(perm[j] for j in term[:2]))
                    j, other = by_pair[pair]
                    force.setExceptionParameters(j, *other[:2], *term[2:])
                    force.setExceptionParameters(i, *term)
            if isinstance(force, openmm.PeriodicTorsionForce):
                terms = [force.getTorsionParameters(i) for i in range(force.getNumTorsions())]
                for term in terms:
                    force.addTorsion(*(perm[i] for i in term[:4]), *term[4:])
    return topology, system


@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_every_explicit_template(name):
    topology, system = residue_fixture(name, symmetric=name in ('ASP', 'PHE', 'TYR'))
    result = dictionary.build_sidechain_dictionary(topology, system)
    expected = (len(EXPECTED[name]) - (1 if name in ('VAL','LEU') else 0)) * (2 if name == 'CYX' else 1)
    assert len(result.torsions) == expected
    assert result.version == dictionary.DICTIONARY_VERSION
    assert all(t.family == 'sidechain' and t.sign == 1 for t in result.torsions)
    assert all(t.chain_id == 'A' and t.residue_id == '10' and t.insertion_code == 'B'
               and t.residue_index == 0 for t in result.torsions if t.residue_index == 0)
    for torsion in result.torsions:
        assert torsion.atom_names[-1] == EXPECTED[name][torsion.chi_index - 1]
        assert tuple(list(topology.atoms())[i].name for i in torsion.orbit[0]) == torsion.atom_names
        assert torsion.harmonic == (2 if name in ('ASP', 'PHE', 'TYR') and torsion.chi_index == 2 else 1)
    assert len(result.primitives) == expected * 2
    if name in ('VAL', 'LEU'):
        assert 'methyl' in result.exclusions[-1].reason
    if name == 'HIS':
        assert result.torsions[0].template == 'HID'


@pytest.mark.parametrize('name', ['ASP', 'PHE', 'TYR'])
def test_actual_amber_improper_equivalence_is_not_assumed(name):
    topology, system = residue_fixture(name)
    result = dictionary.build_sidechain_dictionary(topology, system)
    assert [t.chi_index for t in result.torsions] == [1]
    assert 'PeriodicTorsionForce' in result.exclusions[-1].reason


@pytest.mark.parametrize('name', ['ASP', 'PHE', 'TYR'])
def test_validated_orbit_label_invariance_and_gradient_covariance(name):
    topology, system = residue_fixture(name, symmetric=True)
    result = dictionary.build_sidechain_dictionary(topology, system)
    chi2 = result.torsions[1]
    p = Projection(result.primitives, tuple(np.linspace(.2, .8, len(result.primitives))))
    xyz = np.random.default_rng(402).normal(size=(system.getNumParticles(), 3))
    perm = chi2.permutation
    value, gradient = p.value_gradient(xyz)
    perm_value, perm_gradient = p.value_gradient(xyz[list(perm)])
    assert perm_value == pytest.approx(value, abs=1e-13)
    np.testing.assert_allclose(perm_gradient, gradient[list(perm)], atol=1e-12)


@pytest.mark.parametrize('kind', ['mass', 'charge', 'exception', 'bond', 'angle', 'torsion',
                                 'constraint', 'unknown', 'offset', 'virtual'])
def test_parameter_asymmetry_or_missing_evidence_excludes_only_chi2(kind):
    topology, system = residue_fixture('PHE', symmetric=True)
    names = {a.name: a.index for a in topology.atoms()}
    a, b, c, d = (names[n] for n in ('CD1', 'CG', 'CB', 'CA'))
    nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    if kind == 'mass':
        system.setParticleMass(a, 13)
    elif kind == 'charge':
        q, sig, eps = nb.getParticleParameters(a)
        nb.setParticleParameters(a, q + .1 * unit.elementary_charge, sig, eps)
    elif kind == 'exception':
        nb.addException(a, d, .1, .3, .2, replace=True)
    elif kind == 'bond':
        f = openmm.HarmonicBondForce(); f.addBond(a, b, .1, 99); system.addForce(f)
    elif kind == 'angle':
        f = openmm.HarmonicAngleForce(); f.addAngle(a, b, c, .5, 99); system.addForce(f)
    elif kind == 'torsion':
        f = openmm.PeriodicTorsionForce(); f.addTorsion(a, b, c, d, 2, .2, 99); system.addForce(f)
    elif kind == 'constraint':
        system.addConstraint(a, b, .1)
    elif kind == 'unknown':
        system.addForce(openmm.CustomExternalForce('0'))
    elif kind == 'offset':
        nb.addGlobalParameter('lambda', 0); nb.addParticleParameterOffset('lambda', a, .1, 0, 0)
    else:
        system.setVirtualSite(a, openmm.TwoParticleAverageSite(b, c, .5, .5))
    result = dictionary.build_sidechain_dictionary(topology, system)
    assert [t.chi_index for t in result.torsions] == [1]
    assert result.exclusions and result.exclusions[-1].reason


def test_missing_system_missing_orbit_and_bond_refuse_symmetry():
    topology, system = residue_fixture('PHE', symmetric=True)
    assert 'System' in dictionary.build_sidechain_dictionary(topology).exclusions[-1].reason
    next(a for a in topology.atoms() if a.name == 'HE2').name = 'X'
    assert 'missing' in dictionary.build_sidechain_dictionary(topology, system).exclusions[-1].reason
    topology, system = residue_fixture('ASH')
    topology._bonds = [b for b in topology.bonds() if {b[0].name, b[1].name} != {'CG', 'OD1'}]
    result = dictionary.build_sidechain_dictionary(topology, system)
    assert not result.torsions
    assert 'connectivity' in result.exclusions[-1].reason


def test_protonated_carboxylate_and_his_trp_are_ordinary():
    for name in ('ASH', 'GLH', 'HID', 'HIE', 'HIP', 'TRP'):
        topology, system = residue_fixture(name)
        assert all(t.harmonic == 1 and len(t.orbit) == 1
                   for t in dictionary.build_sidechain_dictionary(topology, system).torsions)
    topology, system = residue_fixture('ASH')
    next(topology.residues()).name = 'ASP'
    result = dictionary.build_sidechain_dictionary(topology, system)
    assert not result.torsions
    assert 'protonation' in result.exclusions[-1].reason


def test_explicit_unknown_and_proline_exclusions():
    topology, system = residue_fixture('PRO')
    assert 'ring' in dictionary.build_sidechain_dictionary(topology, system).exclusions[0].reason
    next(topology.residues()).name = 'MSE'
    assert 'unsupported' in dictionary.build_sidechain_dictionary(topology, system).exclusions[0].reason


def test_templates_match_portable_installed_reference():
    root = Path(app.__file__).parent / 'data'
    ff_templates = ET.parse(root / 'amber14/protein.ff14SB.xml').findall('Residues/Residue')
    lookup = {r.attrib['name']: r for r in ff_templates}
    assert (root / 'residues.xml').is_file()
    for name, endpoints in EXPECTED.items():
        residue = lookup['HID' if name == 'HIS' else name]
        bonds = {frozenset((b.attrib['atomName1'], b.attrib['atomName2'])) for b in residue.findall('Bond')}
        for chi, end in enumerate(endpoints, 1):
            q = ('N', 'CA', 'CB', end) if chi == 1 else ('CA', 'CB', 'CG1' if name == 'ILE' else 'CG', end)
            assert all(frozenset(pair) in bonds for pair in zip(q, q[1:]))


def test_optin_descriptor_adapter_uses_kernel_and_validates_phase_identity():
    from gareus.adaptive.aux_discovery import descriptors
    from gareus.auxiliary_cv.atom_mapping import build_phase_atom_map
    topology, system = residue_fixture('SER')
    result = descriptors.sidechain_descriptor_definition(topology, system)
    xyz = np.random.default_rng(102).normal(size=(3, system.getNumParticles(), 3))
    out = descriptors.evaluate_sidechain_descriptors(xyz, result)
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out[:,0], Projection((result.primitives[0],), (1.,)).values(xyz))
    next(topology.residues()).insertionCode = ''
    result = descriptors.sidechain_descriptor_definition(topology, system)
    trajectory = app.Topology()
    residue = trajectory.addResidue('SER', trajectory.addChain('A'), '10')
    atoms = list(topology.atoms())
    order = tuple(reversed(range(len(atoms))))
    mapped = {i:trajectory.addAtom(atoms[i].name,atoms[i].element,residue) for i in order}
    for a,b in topology.bonds():
        trajectory.addBond(mapped[a.index],mapped[b.index])
    phase = build_phase_atom_map(topology, trajectory, 'p')
    mapped_xyz = xyz[:,order]
    np.testing.assert_allclose(descriptors.evaluate_sidechain_descriptors(mapped_xyz,result,phase), out)
    with pytest.raises(ValueError, match='atom|coordinates'):
        descriptors.evaluate_sidechain_descriptors(mapped_xyz[:,:-1],result)
    next(topology.residues()).id = 'other'
    wrong = build_phase_atom_map(topology, topology, 'p')
    with pytest.raises(ValueError, match='identity'):
        descriptors.evaluate_sidechain_descriptors(xyz,result,wrong)


@pytest.mark.parametrize('source,label', [('ASP','ASH'), ('ASH','ASP'), ('GLU','GLH'),
    ('GLH','GLU'), ('HID','HIE'), ('HIP','HID'), ('LYS','LYN'), ('LYN','LYS'), ('CYS','CYX')])
def test_inconsistent_protonation_labels_are_excluded(source,label):
    topology, system = residue_fixture(source)
    next(topology.residues()).name = label
    result = dictionary.build_sidechain_dictionary(topology, system)
    assert not result.torsions
    assert 'protonation' in result.exclusions[0].reason


@pytest.mark.parametrize('kind', ['wrong_element','duplicate_name','unbonded_his_proton'])
def test_element_and_atom_name_evidence_is_required(kind):
    topology, system = residue_fixture('HID' if kind == 'unbonded_his_proton' else 'SER')
    residue = next(topology.residues())
    if kind == 'wrong_element':
        next(a for a in topology.atoms() if a.name == 'OG').element = app.element.hydrogen
    elif kind == 'duplicate_name':
        topology.addAtom('OG', app.element.hydrogen, residue)
    else:
        residue.name = 'HIS'
        topology._bonds = [b for b in topology.bonds() if {b[0].name,b[1].name} != {'ND1','HD1'}]
    if kind == 'duplicate_name':
        with pytest.raises(ValueError, match='identity'):
            dictionary.build_sidechain_dictionary(topology, system)
    else:
        result = dictionary.build_sidechain_dictionary(topology, system)
        assert not result.torsions and result.exclusions


@pytest.mark.parametrize('kind', ['orphan_his','orphan_asp','extra_heavy','ring_break','crosslink','missing_bonded_evidence'])
def test_modified_template_or_missing_parameter_evidence_is_excluded(kind):
    name = 'HIE' if kind == 'orphan_his' else 'ASP' if kind == 'orphan_asp' else 'PHE' if kind in ('ring_break','missing_bonded_evidence') else 'SER'
    topology, system = residue_fixture(name, symmetric=name in ('ASP','PHE'))
    residue = next(topology.residues()); names = {a.name:a for a in residue.atoms()}
    if kind == 'orphan_his':
        residue.name = 'HIS'; topology.addAtom('HD1', app.element.hydrogen,residue)
    elif kind == 'orphan_asp':
        topology.addAtom('HD2', app.element.hydrogen,residue)
    elif kind == 'extra_heavy':
        atom = topology.addAtom('CX', app.element.carbon,residue); topology.addBond(names['CA'],atom)
    elif kind == 'ring_break':
        topology._bonds = [b for b in topology.bonds() if set((b[0].name,b[1].name)) not in
                           ({'CD1','CE1'},{'CD2','CE2'})]
    elif kind == 'crosslink':
        other = topology.addResidue('SER',topology.addChain('B'),'11')
        atom = topology.addAtom('CB',app.element.carbon,other); topology.addBond(names['OG'],atom)
    else:
        for i in reversed(range(system.getNumForces())):
            if isinstance(system.getForce(i),(openmm.HarmonicBondForce,openmm.HarmonicAngleForce,openmm.PeriodicTorsionForce)):
                system.removeForce(i)
    result = dictionary.build_sidechain_dictionary(topology,system)
    assert not result.torsions if kind != 'missing_bonded_evidence' else [t.chi_index for t in result.torsions] == [1]
    assert result.exclusions


def test_dictionary_order_and_system_evidence_are_frozen():
    from dataclasses import FrozenInstanceError
    topology, system = residue_fixture('PHE', symmetric=True)
    first = dictionary.build_sidechain_dictionary(topology,system)
    second = dictionary.build_sidechain_dictionary(topology,system)
    assert first == second
    assert first.feature_labels == ('chi1_A:10:B_PHE_sin','chi1_A:10:B_PHE_cos',
                                    'chi2_A:10:B_PHE_sin','chi2_A:10:B_PHE_cos')
    assert [(p.trig,p.harmonic) for p in first.primitives] == [('sin',1),('cos',1),('sin',2),('cos',2)]
    with pytest.raises(FrozenInstanceError):
        first.version = 'changed'
    with pytest.raises(TypeError):
        dictionary.TEMPLATES['MSE'] = ('CG','SD')
    index = next(a.index for a in topology.atoms() if a.name == 'CD1')
    system.setParticleMass(index,13)
    changed = dictionary.build_sidechain_dictionary(topology,system)
    assert changed.system_digest != first.system_digest
    assert len(changed.torsions) == 1


def test_full_heavy_graph_matches_both_installed_sources():
    root = Path(app.__file__).parent / 'data'
    amber = {r.attrib['name']:r for r in ET.parse(root / 'amber14/protein.ff14SB.xml').findall('Residues/Residue')}
    residues = {r.attrib['name']:r for r in ET.parse(root / 'residues.xml').findall('Residue')}
    aliases = {'ASH':'ASP','CYX':'CYS','GLH':'GLU','HID':'HIS','HIE':'HIS','HIP':'HIS','LYN':'LYS'}
    for name,bonds in dictionary.HEAVY_BONDS.items():
        ff_bonds = {frozenset((b.attrib['atomName1'],b.attrib['atomName2'])) for b in amber[name].findall('Bond')}
        topology_bonds = {frozenset((b.attrib['from'],b.attrib['to'])) for b in residues[aliases.get(name,name)].findall('Bond')}
        assert {frozenset(b) for b in bonds} <= ff_bonds & topology_bonds
