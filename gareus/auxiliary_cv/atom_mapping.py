"""Frozen atom identity/connectivity maps for explicitly selected trajectory phases."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral

from .sidechain_core import atom_map

MAP_SCHEMA = 'atlas-aux-phase-atom-map-v1'


def _items(topology, name):
    value = getattr(topology, name)
    return tuple(value() if callable(value) else value)


def _digest(raw):
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def topology_metadata(topology):
    atoms = _items(topology, 'atoms')
    residues = _items(topology, 'residues')
    chains = _items(topology, 'chains')
    chain_ids = tuple(str(getattr(c, 'id', getattr(c, 'chain_id', '')) or '') for c in chains)
    if len(set(chain_ids)) != len(chain_ids):
        raise ValueError('ambiguous chain identity')
    residue_keys = {}
    for residue in residues:
        cid = chain_ids[chains.index(residue.chain)]
        rid = str(getattr(residue, 'id', getattr(residue, 'resSeq', '')))
        insertion = str(getattr(residue, 'insertionCode', '') or '')
        key = (cid, rid, insertion)
        if key in residue_keys.values():
            raise ValueError('duplicate residue identity')
        residue_keys[residue] = key
    keys = tuple(residue_keys[a.residue] + (str(a.residue.name), str(a.name),
                  str(a.element.symbol) if a.element is not None else '') for a in atoms)
    if any(a.index != i for i, a in enumerate(atoms)):
        raise ValueError('atom indices must match topology atom enumeration')
    if len({k[:5] for k in keys}) != len(keys):
        raise ValueError('duplicate atom identity')
    edges = tuple(sorted(tuple(sorted((int(a.index), int(b.index))))
                         for a, b in _items(topology, 'bonds')))
    if len(set(edges)) != len(edges) or any(a == b for a,b in edges):
        raise ValueError('invalid topology connectivity')
    return keys, edges


def topology_digest(topology):
    keys, bonds = topology_metadata(topology)
    return _digest({'atom_keys': keys, 'bonds': bonds})


@dataclass(frozen=True)
class PhaseAtomMap:
    phase_id: str
    production_atom_keys: tuple
    trajectory_atom_keys: tuple
    production_bonds: tuple
    trajectory_bonds: tuple
    production_to_trajectory: tuple

    def __post_init__(self):
        if not isinstance(self.phase_id, str) or not self.phase_id.strip():
            raise ValueError('phase identity must be nonempty')
        for field in ('production_atom_keys', 'trajectory_atom_keys'):
            keys = tuple(tuple(k) for k in getattr(self, field))
            if any(len(k) != 6 or any(not isinstance(v, str) for v in k) for k in keys):
                raise ValueError('invalid atom identity')
            if len({k[:5] for k in keys}) != len(keys):
                raise ValueError('duplicate atom identity')
            residue_names = {}
            for k in keys:
                if k[:3] in residue_names and residue_names[k[:3]] != k[3]:
                    raise ValueError('ambiguous residue identity')
                residue_names[k[:3]] = k[3]
            object.__setattr__(self, field, keys)
        for field, count in (('production_bonds', len(self.production_atom_keys)),
                             ('trajectory_bonds', len(self.trajectory_atom_keys))):
            bonds = tuple(tuple(b) for b in getattr(self, field))
            if any(len(b) != 2 or any(isinstance(i, bool) or not isinstance(i, Integral)
                                     or not 0 <= i < count for i in b) or b[0] >= b[1] for b in bonds):
                raise ValueError('invalid connectivity indices')
            if tuple(sorted(set(bonds))) != bonds:
                raise ValueError('noncanonical connectivity')
            object.__setattr__(self, field, bonds)
        mapping = tuple(self.production_to_trajectory)
        expected = atom_map(self.production_atom_keys, self.trajectory_atom_keys)
        if any(i is not None and (isinstance(i, bool) or not isinstance(i, Integral)) for i in mapping):
            raise ValueError('invalid mapping indices')
        if mapping != expected or set(self.trajectory_atom_keys) - set(self.production_atom_keys):
            raise ValueError('missing or inconsistent atom identity in phase mapping')
        mapped_bonds = tuple(sorted(tuple(sorted((mapping[a], mapping[b])))
                                   for a,b in self.production_bonds
                                   if mapping[a] is not None and mapping[b] is not None))
        if mapped_bonds != self.trajectory_bonds:
            raise ValueError('phase connectivity differs from production topology')
        object.__setattr__(self, 'production_to_trajectory', mapping)

    def _payload(self):
        return {'schema': MAP_SCHEMA, 'phase_id': self.phase_id,
                'production_atom_keys': [list(k) for k in self.production_atom_keys],
                'trajectory_atom_keys': [list(k) for k in self.trajectory_atom_keys],
                'production_bonds': [list(b) for b in self.production_bonds],
                'trajectory_bonds': [list(b) for b in self.trajectory_bonds],
                'production_to_trajectory': list(self.production_to_trajectory)}

    @property
    def digest(self):
        return _digest(self._payload())

    def to_mapping(self):
        return dict(self._payload(), sha256=self.digest)

    @classmethod
    def from_mapping(cls, raw):
        fields = {'schema', 'phase_id', 'production_atom_keys', 'trajectory_atom_keys',
                  'production_bonds', 'trajectory_bonds', 'production_to_trajectory', 'sha256'}
        if set(raw) != fields or raw['schema'] != MAP_SCHEMA:
            raise ValueError('unsupported phase atom map payload')
        if _digest({k:v for k,v in raw.items() if k != 'sha256'}) != raw['sha256']:
            raise ValueError('phase mapping checksum mismatch')
        return cls(**{k:v for k,v in raw.items() if k not in ('schema', 'sha256')})

    def verify(self, production_topology, trajectory_topology, phase_id):
        if self.phase_id != phase_id:
            raise ValueError('phase identity mismatch')
        expected = build_phase_atom_map(production_topology, trajectory_topology, phase_id)
        if self != expected:
            raise ValueError('phase atom mapping metadata mismatch')


def build_phase_atom_map(production_topology, trajectory_topology, phase_id, writer_indices=None):
    production_keys, production_bonds = topology_metadata(production_topology)
    trajectory_keys, trajectory_bonds = topology_metadata(trajectory_topology)
    mapping = atom_map(production_keys, trajectory_keys)
    if writer_indices is not None:
        indices = tuple(writer_indices)
        if len(indices) != len(trajectory_keys) or any(
                isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < len(production_keys)
                for i in indices) or len(set(indices)) != len(indices):
            raise ValueError('invalid writer atom-index metadata')
        if tuple(production_keys[i] for i in indices) != trajectory_keys:
            raise ValueError('writer atom-index metadata contradicts atom identity')
    return PhaseAtomMap(phase_id, production_keys, trajectory_keys,
                        production_bonds, trajectory_bonds, mapping)
