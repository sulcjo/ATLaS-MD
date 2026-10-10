"""Curated chi1/chi2 templates; chemical symmetry requires exact System evidence."""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from types import MappingProxyType

from .atom_mapping import _digest, _items, topology_metadata, topology_digest
from .sidechain_core import Primitive

DICTIONARY_VERSION = 'atlas-aux-sidechain-chi12-v1'
TEMPLATES = MappingProxyType({
    'ALA': (), 'GLY': (), 'ARG': ('CG', 'CD'), 'ASN': ('CG', 'OD1'),
    'ASP': ('CG', 'OD1'), 'ASH': ('CG', 'OD1'), 'CYS': ('SG',), 'CYX': ('SG',),
    'GLN': ('CG', 'CD'), 'GLU': ('CG', 'CD'), 'GLH': ('CG', 'CD'),
    'HID': ('CG', 'ND1'), 'HIE': ('CG', 'ND1'), 'HIP': ('CG', 'ND1'),
    'ILE': ('CG1', 'CD1'), 'LEU': ('CG', 'CD1'), 'LYS': ('CG', 'CD'),
    'LYN': ('CG', 'CD'), 'MET': ('CG', 'SD'), 'PHE': ('CG', 'CD1'),
    'SER': ('OG',), 'THR': ('OG1',), 'TRP': ('CG', 'CD1'),
    'TYR': ('CG', 'CD1'), 'VAL': ('CG1',),
})
SUPPORTED_ALIASES = MappingProxyType({'HIS': ('HID', 'HIE', 'HIP')})
EXCLUDED_TEMPLATES = MappingProxyType({'PRO': 'proline ring-constrained chi unsupported'})
SYMMETRY_PAIRS = MappingProxyType({
    'ASP': (('OD1', 'OD2'),),
    'PHE': (('CD1', 'CD2'), ('CE1', 'CE2'), ('HD1', 'HD2'), ('HE1', 'HE2')),
    'TYR': (('CD1', 'CD2'), ('CE1', 'CE2'), ('HD1', 'HD2'), ('HE1', 'HE2')),
})


HEAVY_BONDS = MappingProxyType({
    'ALA': (('CA', 'CB'),),
    'GLY': (),
    'ARG': (('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'NE'), ('NE', 'CZ'), ('CZ', 'NH1'), ('CZ', 'NH2')),
    'ASN': (('CA', 'CB'), ('CB', 'CG'), ('CG', 'OD1'), ('CG', 'ND2')),
    'ASP': (('CA', 'CB'), ('CB', 'CG'), ('CG', 'OD1'), ('CG', 'OD2')),
    'ASH': (('CA', 'CB'), ('CB', 'CG'), ('CG', 'OD1'), ('CG', 'OD2')),
    'CYS': (('CA', 'CB'), ('CB', 'SG')),
    'CYX': (('CA', 'CB'), ('CB', 'SG')),
    'GLN': (('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'OE1'), ('CD', 'NE2')),
    'GLU': (('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'OE1'), ('CD', 'OE2')),
    'GLH': (('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'OE1'), ('CD', 'OE2')),
    'HID': (('CA', 'CB'), ('CB', 'CG'), ('CG', 'ND1'), ('CG', 'CD2'), ('ND1', 'CE1'), ('CE1', 'NE2'), ('NE2', 'CD2')),
    'HIE': (('CA', 'CB'), ('CB', 'CG'), ('CG', 'ND1'), ('CG', 'CD2'), ('ND1', 'CE1'), ('CE1', 'NE2'), ('NE2', 'CD2')),
    'HIP': (('CA', 'CB'), ('CB', 'CG'), ('CG', 'ND1'), ('CG', 'CD2'), ('ND1', 'CE1'), ('CE1', 'NE2'), ('NE2', 'CD2')),
    'ILE': (('CA', 'CB'), ('CB', 'CG2'), ('CB', 'CG1'), ('CG1', 'CD1')),
    'LEU': (('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2')),
    'LYS': (('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'CE'), ('CE', 'NZ')),
    'LYN': (('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'CE'), ('CE', 'NZ')),
    'MET': (('CA', 'CB'), ('CB', 'CG'), ('CG', 'SD'), ('SD', 'CE')),
    'PHE': (('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2'), ('CD1', 'CE1'), ('CE1', 'CZ'), ('CZ', 'CE2'), ('CE2', 'CD2')),
    'SER': (('CA', 'CB'), ('CB', 'OG')),
    'THR': (('CA', 'CB'), ('CB', 'CG2'), ('CB', 'OG1')),
    'TRP': (('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2'), ('CD1', 'NE1'), ('NE1', 'CE2'), ('CE2', 'CZ2'), ('CE2', 'CD2'), ('CZ2', 'CH2'), ('CH2', 'CZ3'), ('CZ3', 'CE3'), ('CE3', 'CD2')),
    'TYR': (('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2'), ('CD1', 'CE1'), ('CE1', 'CZ'), ('CZ', 'OH'), ('CZ', 'CE2'), ('CE2', 'CD2')),
    'VAL': (('CA', 'CB'), ('CB', 'CG1'), ('CB', 'CG2')),
})
PROTONATION = MappingProxyType({
    'ASP': (('OD1', ()), ('OD2', ())),
    'ASH': (('OD1', ()), ('OD2', ('HD2',))),
    'GLU': (('OE1', ()), ('OE2', ())),
    'GLH': (('OE1', ()), ('OE2', ('HE2',))),
    'CYS': (('SG', ('HG',)),),
    'CYX': (('SG', ()),),
    'HID': (('ND1', ('HD1',)), ('NE2', ())),
    'HIE': (('ND1', ()), ('NE2', ('HE2',))),
    'HIP': (('ND1', ('HD1',)), ('NE2', ('HE2',))),
    'LYS': (('NZ', ('HZ1', 'HZ2', 'HZ3')),),
    'LYN': (('NZ', ('HZ2', 'HZ3')),),
})
PROTON_GROUPS = (('ASP','ASH'), ('GLU','GLH'), ('CYS','CYX'),
                 ('HID','HIE','HIP'), ('LYS','LYN'))
_EQUIVALENCE_CACHE = OrderedDict()


@dataclass(frozen=True)
class ChiTorsion:
    chain_id: str
    residue_id: str
    insertion_code: str
    residue_index: int
    residue_name: str
    template: str
    chi_index: int
    atom_names: tuple
    orbit: tuple
    harmonic: int
    permutation: tuple = ()
    sign: int = 1
    family: str = 'sidechain'

    @property
    def primitives(self):
        return tuple(Primitive(self.orbit, trig, self.harmonic, self.sign) for trig in ('sin', 'cos'))


@dataclass(frozen=True)
class ChiExclusion:
    chain_id: str
    residue_id: str
    insertion_code: str
    residue_index: int
    residue_name: str
    template: str
    chi_index: int
    reason: str


@dataclass(frozen=True)
class SidechainDictionary:
    torsions: tuple
    exclusions: tuple
    atom_keys: tuple
    topology_digest: str
    system_digest: str
    version: str = DICTIONARY_VERSION

    @property
    def primitives(self):
        return tuple(p for t in self.torsions for p in t.primitives)

    @property
    def feature_labels(self):
        return tuple(f'chi{t.chi_index}_{t.chain_id}:{t.residue_id}:{t.insertion_code}_{t.template}_{trig}'
                     for t in self.torsions for trig in ('sin', 'cos'))


def _scalar(value):
    return float(value._value) if hasattr(value, '_value') else float(value)


def _system_evidence(system, count):
    if system is None:
        return None, '', 'parameterised System is required for symmetry'
    import openmm
    serialized = openmm.XmlSerializer.serialize(system)
    digest = _digest({'system_xml': serialized})
    if system.getNumParticles() != count:
        return None, digest, 'System particle count differs from topology'
    if any(system.isVirtualSite(i) for i in range(count)):
        return None, digest, 'virtual sites have unsupported symmetry evidence'
    constraints = tuple(tuple(system.getConstraintParameters(i)) for i in range(system.getNumConstraints()))
    forces = []
    for force in system.getForces():
        family = type(force).__name__
        if family == 'NonbondedForce':
            if force.getNumParticleParameterOffsets() or force.getNumExceptionParameterOffsets():
                return None, digest, 'NonbondedForce offsets have unsupported symmetry evidence'
            particles = tuple(tuple(_scalar(v) for v in force.getParticleParameters(i))
                              for i in range(force.getNumParticles()))
            exceptions = tuple(tuple(force.getExceptionParameters(i)) for i in range(force.getNumExceptions()))
            forces.append((family, particles, exceptions))
        elif family in ('HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce'):
            getter, number = {'HarmonicBondForce': ('getBondParameters','getNumBonds'),
                              'HarmonicAngleForce': ('getAngleParameters','getNumAngles'),
                              'PeriodicTorsionForce': ('getTorsionParameters','getNumTorsions')}[family]
            terms = tuple(tuple(getattr(force, getter)(i)) for i in range(getattr(force, number)()))
            forces.append((family, terms))
        elif family != 'CMMotionRemover':
            return None, digest, f'unsupported System force family {family}'
    required = {'NonbondedForce','HarmonicBondForce','HarmonicAngleForce','PeriodicTorsionForce'}
    if required - {f[0] for f in forces}:
        return None, digest, 'missing supported bonded/nonbonded System equivalence evidence'
    return (tuple(_scalar(system.getParticleMass(i)) for i in range(count)), constraints, tuple(forces)), digest, ''


def _canonical_terms(terms, width, permutation):
    rows = []
    for term in terms:
        atoms = tuple(permutation[int(i)] for i in term[:width])
        atoms = min(atoms, atoms[::-1])
        rows.append(atoms + tuple(_scalar(v) for v in term[width:]))
    return Counter(rows)


def _validate_symmetry(permutation, bonds, evidence, error):
    if error:
        return error
    if {tuple(sorted((permutation[a], permutation[b]))) for a,b in bonds} != set(bonds):
        return 'symmetry permutation does not preserve topology connectivity/protonation'
    masses, constraints, forces = evidence
    if tuple(masses[i] for i in permutation) != masses:
        return 'symmetry permutation changes System particle masses'
    identity = tuple(range(len(permutation)))
    if _canonical_terms(constraints, 2, identity) != _canonical_terms(constraints, 2, permutation):
        return 'symmetry permutation changes System constraints'
    for force in forces:
        family = force[0]
        if family == 'NonbondedForce':
            if tuple(force[1][i] for i in permutation) != force[1]:
                return 'symmetry permutation changes NonbondedForce particle parameters'
            terms, width = force[2], 2
        else:
            terms, width = force[1], {'HarmonicBondForce':2, 'HarmonicAngleForce':3,
                                     'PeriodicTorsionForce':4}[family]
        if _canonical_terms(terms, width, identity) != _canonical_terms(terms, width, permutation):
            return f'no sufficient ordered-term equivalence proof for {family}'
    return ''


def _element(name):
    if name.startswith('H'):
        return 'H'
    return name[0]


def _template_error(template, names, keys, bonds):
    for group in PROTON_GROUPS:
        if template not in group:
            continue
        possible = {n for variant in group for _,signature in PROTONATION[variant] for n in signature}
        expected = {n for _,signature in PROTONATION[template] for n in signature}
        if (possible & set(names)) != expected:
            return 'template protonation atom signature mismatch'
    for heteroatom, expected in PROTONATION.get(template, ()):
        if heteroatom not in names:
            return 'template protonation evidence missing heteroatom'
        index = names[heteroatom]
        attached = {keys[b if a == index else a][4] for a,b in bonds
                    if index in (a,b) and keys[b if a == index else a][-1] == 'H'}
        if attached != set(expected) or any(n not in names or keys[names[n]][-1] != 'H'
                                          for n in expected):
            return 'template protonation signature/connectivity mismatch'
        if any(n in names and tuple(sorted((index,names[n]))) not in bonds
               for n in expected):
            return 'template protonation bond mismatch'
    heavy_bonds = HEAVY_BONDS[template]
    heavy_names = {n for pair in heavy_bonds for n in pair}
    allowed = heavy_names | {'N','CA','C','O','OXT'}
    if any(n not in allowed and keys[i][-1] != 'H' for n,i in names.items()):
        return 'unsupported extra heavy atom in standard template'
    if any(n not in names for n in heavy_names):
        return 'missing required heavy template atom'
    if any(keys[names[n]][-1] != _element(n) for n in heavy_names):
        return 'template heavy atom element mismatch'
    expected = {tuple(sorted((names[a],names[b]))) for a,b in heavy_bonds}
    selected = {names[n] for n in heavy_names if n != 'CA'}
    actual = {pair for pair in bonds if any(i in selected for i in pair)
              and all(keys[i][-1] != 'H' for i in pair)
              and all(keys[i][:3] == keys[next(iter(names.values()))][:3] for i in pair)}
    if actual != expected:
        return 'template heavy bonded connectivity mismatch'
    external = [(i,j) for a,b in bonds for i,j in ((a,b),(b,a)) if i in selected
                and keys[j][:3] != keys[i][:3]]
    if template == 'CYX':
        sg = names['SG']
        if len(external) != 1 or external[0][0] != sg or keys[external[0][1]][3:] != ('CYX','SG','S'):
            return 'CYX template requires an explicit disulfide bond'
    elif external:
        return 'unsupported sidechain crosslink in standard template'
    return ''


def build_sidechain_dictionary(topology, system=None):
    keys, bonds = topology_metadata(topology)
    bond_set = set(bonds)
    evidence, system_digest, evidence_error = _system_evidence(system, len(keys))
    topology_sha = topology_digest(topology)
    torsions, exclusions = [], []
    for residue in _items(topology, 'residues'):
        atoms = _items(residue, 'atoms')
        names = {a.name:a.index for a in atoms}
        if not atoms:
            continue
        chain, rid, insertion = keys[atoms[0].index][:3]
        name, template = str(residue.name), str(residue.name)
        if name == 'HIS':
            hd = 'HD1' in names and 'ND1' in names and tuple(sorted((names['HD1'],names['ND1']))) in bond_set
            he = 'HE2' in names and 'NE2' in names and tuple(sorted((names['HE2'],names['NE2']))) in bond_set
            template = 'HIP' if hd and he else 'HID' if hd else 'HIE' if he else ''
        identity = (chain, rid, insertion, int(residue.index), name, template)
        if template not in TEMPLATES:
            reason = EXCLUDED_TEMPLATES.get(name, 'unsupported residue/template or ambiguous HIS protonation')
            exclusions.append(ChiExclusion(*identity, 0, reason))
            continue
        error = _template_error(template, names, keys, bond_set)
        if error:
            exclusions.append(ChiExclusion(*identity, 0, error))
            continue
        for chi, end in enumerate(TEMPLATES[template], 1):
            if template == 'VAL' or (template == 'LEU' and chi == 2):
                exclusions.append(ChiExclusion(*identity, chi, 'equivalent methyl branch orbit unsupported'))
                continue
            quad_names = ('N','CA','CB',end) if chi == 1 else (
                'CA','CB','CG1' if template == 'ILE' else 'CG',end)
            if any(n not in names for n in quad_names):
                exclusions.append(ChiExclusion(*identity, chi, 'missing required chi atom'))
                continue
            quad = tuple(names[n] for n in quad_names)
            if any(keys[i][-1] != _element(n) for i,n in zip(quad,quad_names)):
                exclusions.append(ChiExclusion(*identity, chi, 'chi atom element mismatch'))
                continue
            if any(tuple(sorted(pair)) not in bond_set for pair in zip(quad, quad[1:])):
                exclusions.append(ChiExclusion(*identity, chi, 'chi bonded connectivity mismatch'))
                continue
            orbit, harmonic, permutation = (quad,), 1, ()
            if chi == 2 and template in SYMMETRY_PAIRS:
                pairs = SYMMETRY_PAIRS[template]
                required = {n for pair in pairs for n in pair}
                if template in ('PHE', 'TYR'):
                    required.update(('CG', 'CZ', 'HZ') if template == 'PHE' else ('CG', 'CZ', 'OH', 'HH'))
                if required - set(names):
                    exclusions.append(ChiExclusion(*identity, chi, 'missing required symmetry orbit atom'))
                    continue
                permutation = list(range(len(keys)))
                for a,b in pairs:
                    permutation[names[a]], permutation[names[b]] = names[b], names[a]
                permutation = tuple(permutation)
                if any(keys[names[n]][-1] != _element(n) for n in required):
                    exclusions.append(ChiExclusion(*identity, chi, 'symmetry orbit atom element mismatch'))
                    continue
                cache_key = (DICTIONARY_VERSION, topology_sha, system_digest,
                             tuple((i,j) for i,j in enumerate(permutation) if i != j))
                if cache_key not in _EQUIVALENCE_CACHE:
                    _EQUIVALENCE_CACHE[cache_key] = _validate_symmetry(permutation,bonds,evidence,evidence_error)
                    if len(_EQUIVALENCE_CACHE) > 128:
                        _EQUIVALENCE_CACHE.popitem(last=False)
                error = _EQUIVALENCE_CACHE[cache_key]
                if error:
                    exclusions.append(ChiExclusion(*identity, chi, error))
                    continue
                orbit = tuple(sorted((quad, tuple(permutation[a] for a in quad))))
                harmonic = 2
            torsions.append(ChiTorsion(*identity, chi, quad_names, orbit, harmonic, permutation))
    return SidechainDictionary(tuple(torsions), tuple(exclusions), keys,
                               topology_sha, system_digest)
