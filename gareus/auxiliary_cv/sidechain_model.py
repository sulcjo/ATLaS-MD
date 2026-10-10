"""Frozen production identity facade for orbit-aware auxiliary projections."""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping

from ..correctness._io import IntegrityError, digest, json_bytes, json_loads
from .atom_mapping import topology_metadata
from .sidechain_core import Primitive, Projection

SIDECHAIN_MODEL_SCHEMA = 'atlas-aux-cv-model-v2'
SUPPORTED_UNITS = frozenset({'dimensionless'})
SUPPORTED_PERIODIC_IMAGING = frozenset({'none'})
_MODEL_FIELDS = {'schema', 'dictionary_version', 'topology_sha256', 'system_sha256', 'atom_keys',
                 'bonds', 'features', 'coefficients', 'offset', 'scale', 'periodic_imaging', 'units',
                 'dictionary_exclusions'}
_OPTIONAL_FIELDS = {'model_sha256', 'label', 'provenance'}
_FEATURE_FIELDS = {'name', 'orbit', 'trig', 'harmonic', 'sign', 'family', 'chain_id',
                   'residue_id', 'insertion_code', 'residue_index', 'residue_name',
                   'template', 'chi_index'}


class SidechainModelError(IntegrityError):
    """A v2 payload does not describe one finite, bound projection."""


def _sha(value: Any, label: str, *, optional: bool = False) -> str:
    if optional and value == '':
        return ''
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise SidechainModelError(f'{label} must be a lowercase SHA-256 digest')
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise SidechainModelError(f'{label} must be a finite real number')
    return float(value) + 0.0


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise SidechainModelError(f'{label} must be an integer >= {minimum}')
    return int(value)


@dataclass(frozen=True)
class SidechainFeature:
    name: str
    orbit: tuple[tuple[int, int, int, int], ...]
    trig: str
    harmonic: int
    sign: int
    family: str
    chain_id: str
    residue_id: str
    insertion_code: str
    residue_index: int
    residue_name: str
    template: str
    chi_index: int

    @property
    def primitive(self) -> Primitive:
        return Primitive(self.orbit, self.trig, self.harmonic, self.sign)

    def to_mapping(self) -> dict[str, Any]:
        return {'name': self.name, 'orbit': [list(q) for q in self.orbit], 'trig': self.trig,
                'harmonic': self.harmonic, 'sign': self.sign, 'family': self.family,
                'chain_id': self.chain_id, 'residue_id': self.residue_id,
                'insertion_code': self.insertion_code, 'residue_index': self.residue_index,
                'residue_name': self.residue_name, 'template': self.template,
                'chi_index': self.chi_index}


def _parse_feature(raw: Any, position: int, atom_count: int) -> SidechainFeature:
    if not isinstance(raw, dict) or set(raw) != _FEATURE_FIELDS:
        raise SidechainModelError(f'features[{position}] has missing or unknown fields')
    text_fields = ('name', 'family', 'chain_id', 'residue_id', 'insertion_code',
                   'residue_name', 'template')
    if any(not isinstance(raw[k], str) for k in text_fields) or not raw['name']:
        raise SidechainModelError(f'features[{position}] has invalid identity text')
    if raw['family'] not in ('backbone', 'sidechain'):
        raise SidechainModelError(f'features[{position}].family is unsupported')
    if raw['trig'] not in ('sin', 'cos'):
        raise SidechainModelError(f'features[{position}].trig must be sin or cos')
    harmonic = _integer(raw['harmonic'], f'features[{position}].harmonic', minimum=1)
    if harmonic not in (1, 2):
        raise SidechainModelError(f'features[{position}].harmonic must be 1 or 2')
    sign = raw['sign']
    if isinstance(sign, bool) or not isinstance(sign, Integral) or sign not in (-1, 1):
        raise SidechainModelError(f'features[{position}].sign must be -1 or +1')
    residue_index = _integer(raw['residue_index'], f'features[{position}].residue_index')
    chi_index = _integer(raw['chi_index'], f'features[{position}].chi_index')
    if (raw['family'] == 'sidechain' and chi_index < 1) or (raw['family'] == 'backbone' and chi_index != 0):
        raise SidechainModelError(f'features[{position}].chi_index does not match feature family')
    orbit_raw = raw['orbit']
    if not isinstance(orbit_raw, list) or not orbit_raw:
        raise SidechainModelError(f'features[{position}].orbit must be nonempty')
    orbit = []
    for member in orbit_raw:
        if (not isinstance(member, list) or len(member) != 4
                or any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 or i >= atom_count
                       for i in member)
                or len(set(member)) != 4):
            raise SidechainModelError(f'features[{position}].orbit has invalid atom indices')
        orbit.append(tuple(int(i) for i in member))
    if len(set(orbit)) != len(orbit):
        raise SidechainModelError(f'features[{position}].orbit contains duplicate members')
    primitive = Primitive(tuple(orbit), raw['trig'], harmonic, int(sign))
    return SidechainFeature(raw['name'], primitive.orbit, raw['trig'], harmonic, int(sign),
                            raw['family'], raw['chain_id'], raw['residue_id'], raw['insertion_code'],
                            residue_index, raw['residue_name'], raw['template'], chi_index)


def _atom_keys(raw: Any) -> tuple[tuple[str, ...], ...]:
    if not isinstance(raw, list) or not raw:
        raise SidechainModelError('atom_keys must be a nonempty ordered list')
    keys = []
    for i, key in enumerate(raw):
        if not isinstance(key, list) or len(key) != 6 or any(not isinstance(v, str) for v in key):
            raise SidechainModelError(f'atom_keys[{i}] must contain six identity strings')
        keys.append(tuple(key))
    if len(set(keys)) != len(keys):
        raise SidechainModelError('atom_keys contain duplicate atom identities')
    residues = {}
    atom_names = set()
    for key in keys:
        residue_key = key[:3]
        if residue_key in residues and residues[residue_key] != key[3]:
            raise SidechainModelError('atom_keys contain conflicting residue names')
        residues[residue_key] = key[3]
        atom_key = key[:3] + (key[4],)
        if atom_key in atom_names:
            raise SidechainModelError('atom_keys repeat an atom name within one residue')
        atom_names.add(atom_key)
    return tuple(keys)


def _bonds(raw: Any, atom_count: int) -> tuple[tuple[int, int], ...]:
    if not isinstance(raw, list):
        raise SidechainModelError('bonds must be an ordered list')
    out = []
    for edge in raw:
        if (not isinstance(edge, list) or len(edge) != 2
                or any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 or i >= atom_count
                       for i in edge)
                or edge[0] >= edge[1]):
            raise SidechainModelError('bonds contain invalid atom indices')
        out.append((int(edge[0]), int(edge[1])))
    if out != sorted(set(out)):
        raise SidechainModelError('bonds must be unique and canonically ordered')
    return tuple(out)


def _validate_feature_binding(feature: SidechainFeature, atom_keys, bonds) -> None:
    edge_set = set(bonds)
    if feature.family == 'backbone':
        if feature.harmonic != 1 or len(feature.orbit) != 1 or feature.chi_index != 0:
            raise SidechainModelError(f'{feature.name}: backbone feature must be one ordinary torsion')
        torsion = feature.name.split('_', 1)[0]
        expected = {'phi': ('C', 'N', 'CA', 'C'), 'psi': ('N', 'CA', 'C', 'N')}.get(torsion)
        if expected is None:
            raise SidechainModelError(f'{feature.name}: backbone feature must identify phi or psi')
        quad = feature.orbit[0]
        keys = [atom_keys[i] for i in quad]
        if tuple(k[4] for k in keys) != expected:
            raise SidechainModelError(f'{feature.name}: atom names do not match backbone torsion')
        residues = [k[:3] for k in keys]
        order = {}
        for key in atom_keys:
            order.setdefault(key[:3], len(order))
        central_slots = (1, 2, 3) if torsion == 'phi' else (0, 1, 2)
        central = residues[central_slots[0]]
        if any(residues[i] != central for i in central_slots):
            raise SidechainModelError(f'{feature.name}: central backbone atoms span residues')
        terminal = residues[0] if torsion == 'phi' else residues[3]
        if central[0] != feature.chain_id or central[1] != feature.residue_id or central[2] != feature.insertion_code:
            raise SidechainModelError(f'{feature.name}: declared residue identity differs from backbone atoms')
        if order.get(central) != feature.residue_index or order.get(terminal) != feature.residue_index + (-1 if torsion == 'phi' else 1):
            raise SidechainModelError(f'{feature.name}: backbone atoms are not on adjacent residues')
        if feature.residue_name != keys[central_slots[0]][3]:
            raise SidechainModelError(f'{feature.name}: declared residue name differs from backbone atoms')
        if feature.template not in (feature.residue_name, 'BACKBONE'):
            raise SidechainModelError(f'{feature.name}: template differs from backbone residue identity')
        if any(tuple(sorted((a, b))) not in edge_set for a, b in zip(quad, quad[1:])):
            raise SidechainModelError(f'{feature.name}: backbone torsion is not bonded')
        return
    from .sidechain_dictionary import SYMMETRY_PAIRS, TEMPLATES
    ends = TEMPLATES.get(feature.template)
    if ends is None or feature.chi_index > len(ends):
        raise SidechainModelError(f'{feature.name}: chi is outside curated template')
    if feature.residue_name not in (feature.template, 'HIS' if feature.template in ('HID','HIE','HIP') else ''):
        raise SidechainModelError(f'{feature.name}: residue and template names are inconsistent')
    symmetric_case = feature.chi_index == 2 and feature.template in SYMMETRY_PAIRS
    if symmetric_case and (feature.harmonic != 2 or len(feature.orbit) != 2):
        raise SidechainModelError(f'{feature.name}: curated symmetric chi2 requires validated harmonic-2 orbit')
    if feature.harmonic == 2:
        if not symmetric_case:
            raise SidechainModelError(f'{feature.name}: harmonic-2 orbit is not a curated twofold case')
    elif len(feature.orbit) != 1:
        raise SidechainModelError(f'{feature.name}: ordinary harmonic-1 feature must have one orbit member')
    expected = (('N', 'CA', 'CB', ends[feature.chi_index - 1]) if feature.chi_index == 1 else
                ('CA', 'CB', 'CG1' if feature.template == 'ILE' else 'CG',
                 ends[feature.chi_index - 1]))
    residue_key = (feature.chain_id, feature.residue_id, feature.insertion_code)
    for quad in feature.orbit:
        keys = [atom_keys[i] for i in quad]
        if any(k[:3] != residue_key or k[3] != feature.residue_name for k in keys):
            raise SidechainModelError(f'{feature.name}: orbit atoms differ from declared residue identity')
        names = tuple(k[4] for k in keys)
        if names[0:3] != expected[0:3] or names[3] not in (expected[3],):
            symmetry = dict(SYMMETRY_PAIRS.get(feature.template, ()))
            counterpart = symmetry.get(expected[3])
            if names[0:3] != expected[0:3] or names[3] != counterpart:
                raise SidechainModelError(f'{feature.name}: orbit atom names disagree with curated chi')
        if any(tuple(sorted((a, b))) not in edge_set for a, b in zip(quad, quad[1:])):
            raise SidechainModelError(f'{feature.name}: orbit does not follow frozen bonded connectivity')


@dataclass(frozen=True)
class SidechainModel:
    dictionary_version: str
    topology_sha256: str
    system_sha256: str
    atom_keys: tuple[tuple[str, ...], ...]
    bonds: tuple[tuple[int, int], ...]
    features: tuple[SidechainFeature, ...]
    coefficients: tuple[float, ...]
    offset: float
    scale: float
    periodic_imaging: str
    units: str
    dictionary_exclusions_json: str
    label: str
    provenance_json: str
    model_sha256: str

    @property
    def projection(self) -> Projection:
        return Projection(tuple(f.primitive for f in self.features), self.coefficients,
                          self.offset, self.scale)

    @property
    def width(self) -> int:
        return len(self.features)

    @property
    def schema_version(self) -> int:
        return 2

    @property
    def active_feature_indices(self) -> tuple[int, ...]:
        return tuple(i for i, c in enumerate(self.coefficients) if c != 0.0)

    @property
    def dictionary_exclusions(self) -> tuple[dict[str, Any], ...]:
        return tuple(json_loads(self.dictionary_exclusions_json))

    def values(self, xyz_nm):
        return self.projection.values(xyz_nm)

    def value_gradient(self, xyz_nm):
        return self.projection.value_gradient(xyz_nm)

    @staticmethod
    def _identity(*, dictionary_version, topology_sha256, system_sha256, atom_keys, bonds,
                   features, coefficients, offset, scale, periodic_imaging, units,
                   dictionary_exclusions):
        return {'schema': SIDECHAIN_MODEL_SCHEMA, 'dictionary_version': dictionary_version,
                'topology_sha256': topology_sha256, 'system_sha256': system_sha256,
                'atom_keys': [list(k) for k in atom_keys],
                'bonds': [list(edge) for edge in bonds],
                'features': [f.to_mapping() for f in features],
                'coefficients': list(coefficients), 'offset': offset, 'scale': scale,
                'periodic_imaging': periodic_imaging, 'units': units,
                'dictionary_exclusions': list(dictionary_exclusions)}

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> 'SidechainModel':
        try:
            data = json_loads(json_bytes(dict(raw)))
        except (TypeError, ValueError) as exc:
            raise SidechainModelError(f'invalid v2 model JSON: {exc}') from exc
        unknown = sorted(set(data) - _MODEL_FIELDS - _OPTIONAL_FIELDS)
        missing = sorted(_MODEL_FIELDS - set(data))
        if unknown or missing:
            raise SidechainModelError(f'v2 model fields invalid; missing={missing}, unknown={unknown}')
        if data['schema'] != SIDECHAIN_MODEL_SCHEMA:
            raise SidechainModelError(f'unsupported model schema {data["schema"]!r}')
        from .sidechain_dictionary import DICTIONARY_VERSION
        if data['dictionary_version'] != DICTIONARY_VERSION:
            raise SidechainModelError(f'unsupported sidechain dictionary {data["dictionary_version"]!r}')
        for key in ('dictionary_version',):
            if not isinstance(data[key], str) or not data[key]:
                raise SidechainModelError(f'{key} must be nonempty text')
        topology = _sha(data['topology_sha256'], 'topology_sha256')
        system = _sha(data['system_sha256'], 'system_sha256', optional=True)
        keys = _atom_keys(data['atom_keys'])
        bonds = _bonds(data['bonds'], len(keys))
        from .atom_mapping import _digest
        if _digest({'atom_keys': keys, 'bonds': bonds}) != topology:
            raise SidechainModelError('topology_sha256 does not match frozen atom keys and connectivity')
        feature_rows = data['features']
        if not isinstance(feature_rows, list) or not feature_rows:
            raise SidechainModelError('v2 model needs at least one declared feature')
        features = tuple(_parse_feature(row, i, len(keys)) for i, row in enumerate(feature_rows))
        residue_order = {}
        for key in keys:
            residue_order.setdefault(key[:3], len(residue_order))
        for feature in features:
            if feature.family == 'sidechain' and residue_order.get(
                    (feature.chain_id, feature.residue_id, feature.insertion_code)) != feature.residue_index:
                raise SidechainModelError(f'{feature.name}: residue_index differs from frozen atom ordering')
            _validate_feature_binding(feature, keys, bonds)
        names = [f.name for f in features]
        if len(set(names)) != len(names):
            raise SidechainModelError('feature names must be unique')
        coeff_raw = data['coefficients']
        if not isinstance(coeff_raw, list) or len(coeff_raw) != len(features):
            raise SidechainModelError('one coefficient is required per declared feature')
        coeffs = tuple(_finite(c, f'coefficients[{i}]') for i, c in enumerate(coeff_raw))
        terms = {}
        for feature, coefficient in zip(features, coeffs):
            if coefficient == 0.0:
                continue
            for quad in feature.orbit:
                key = (quad, feature.trig, feature.sign * feature.harmonic)
                terms[key] = terms.get(key, 0.0) + coefficient / len(feature.orbit)
        if not any(value != 0.0 for value in terms.values()):
            raise SidechainModelError('v2 production model must have nonconstant active projection')
        offset, scale = _finite(data['offset'], 'offset'), _finite(data['scale'], 'scale')
        if scale <= 0.0:
            raise SidechainModelError('scale must be positive')
        if not isinstance(data['periodic_imaging'], str) or data['periodic_imaging'] not in SUPPORTED_PERIODIC_IMAGING:
            raise SidechainModelError('unsupported periodic_imaging; only whole-molecule none is supported')
        if not isinstance(data['units'], str) or data['units'] not in SUPPORTED_UNITS:
            raise SidechainModelError('unsupported units; z must be dimensionless')
        exclusions = data['dictionary_exclusions']
        if not isinstance(exclusions, list) or any(not isinstance(x, dict) for x in exclusions):
            raise SidechainModelError('dictionary_exclusions must be a list of records')
        label = data.get('label', '')
        provenance = data.get('provenance', {})
        if not isinstance(label, str) or not isinstance(provenance, dict):
            raise SidechainModelError('label must be text and provenance must be an object')
        ident = cls._identity(dictionary_version=data['dictionary_version'], topology_sha256=topology,
                              system_sha256=system, atom_keys=keys, bonds=bonds, features=features,
                              coefficients=coeffs, offset=offset, scale=scale,
                              periodic_imaging=data['periodic_imaging'], units=data['units'],
                              dictionary_exclusions=exclusions)
        sha = digest(json_bytes(ident))
        claimed = data.get('model_sha256')
        if claimed is not None and claimed != sha:
            raise SidechainModelError(f'claimed model hash {claimed!r} differs from content hash {sha}')
        return cls(data['dictionary_version'], topology, system, keys, bonds, features, coeffs, offset, scale,
                   data['periodic_imaging'], data['units'], json_bytes(exclusions).decode(), label,
                   json_bytes(provenance).decode(), sha)

    @classmethod
    def from_dictionary(cls, dictionary, coefficients, *, topology, offset=0.0, scale=1.0,
                        periodic_imaging='none', units='dimensionless', provenance=None):
        atom_keys, bonds = topology_metadata(topology)
        if atom_keys != dictionary.atom_keys or cls._topology_digest(atom_keys, bonds) != dictionary.topology_digest:
            raise SidechainModelError('dictionary identity does not match supplied topology')
        features = []
        for torsion in dictionary.torsions:
            for primitive, name in zip(torsion.primitives,
                                        (f'chi{torsion.chi_index}_{torsion.chain_id}:{torsion.residue_id}:'
                                         f'{torsion.insertion_code}_{torsion.template}_{trig}'
                                         for trig in ('sin', 'cos'))):
                features.append({'name': name, 'orbit': [list(q) for q in primitive.orbit],
                                 'trig': primitive.trig, 'harmonic': primitive.harmonic,
                                 'sign': primitive.sign, 'family': torsion.family,
                                 'chain_id': torsion.chain_id, 'residue_id': torsion.residue_id,
                                 'insertion_code': torsion.insertion_code,
                                 'residue_index': torsion.residue_index, 'residue_name': torsion.residue_name,
                                 'template': torsion.template, 'chi_index': torsion.chi_index})
        raw = {'schema': SIDECHAIN_MODEL_SCHEMA, 'dictionary_version': dictionary.version,
               'topology_sha256': dictionary.topology_digest, 'system_sha256': dictionary.system_digest,
               'atom_keys': [list(k) for k in dictionary.atom_keys],
               'bonds': [list(edge) for edge in bonds], 'features': features,
               'coefficients': list(coefficients), 'offset': offset, 'scale': scale,
               'periodic_imaging': periodic_imaging, 'units': units,
               'dictionary_exclusions': [vars(x) for x in dictionary.exclusions],
               'provenance': dict(provenance or {})}
        return cls.from_mapping(raw)

    def identity_mapping(self) -> dict[str, Any]:
        return {**self._identity(dictionary_version=self.dictionary_version,
                                 topology_sha256=self.topology_sha256, system_sha256=self.system_sha256,
                                 atom_keys=self.atom_keys, bonds=self.bonds, features=self.features,
                                 coefficients=self.coefficients, offset=self.offset, scale=self.scale,
                                 periodic_imaging=self.periodic_imaging, units=self.units,
                                 dictionary_exclusions=self.dictionary_exclusions),
                'model_sha256': self.model_sha256}

    def to_mapping(self) -> dict[str, Any]:
        return {**self.identity_mapping(), 'label': self.label,
                'provenance': json_loads(self.provenance_json)}

    def validate_topology(self, topology) -> None:
        keys, bonds = topology_metadata(topology)
        actual = self._topology_digest(keys, bonds)
        if keys != self.atom_keys or bonds != self.bonds or actual != self.topology_sha256:
            raise SidechainModelError(f'model atom identity/connectivity differs from topology {actual}')

    @classmethod
    def load(cls, path) -> 'SidechainModel':
        from pathlib import Path
        return cls.from_mapping(json_loads(Path(path).read_bytes()))

    def write(self, path) -> None:
        from pathlib import Path
        from ..correctness._io import atomic_bytes
        atomic_bytes(Path(path), json_bytes(self.to_mapping()))

    @staticmethod
    def _topology_digest(keys, bonds):
        from .atom_mapping import _digest
        return _digest({'atom_keys': keys, 'bonds': bonds})
