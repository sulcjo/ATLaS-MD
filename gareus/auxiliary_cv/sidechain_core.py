"""Tested numerical kernel for future v2 auxiliary models; not wired into production.

An orbit is supplied by a validated chemistry dictionary, NOT inferred here.
This module guarantees permutation averaging, not chemical equivalence. Units:
positions nm, angles radians, projection dimensionless, build_force k in kJ/mol.
The core payload is deliberately not an AuxModel deployment/checkpoint schema.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral, Real

import numpy as np

from .features import openmm_dihedrals
from .evaluate import _dtheta_dx


def _real(x, name):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) or not np.isfinite(x):
        raise ValueError(f'{name} must be a finite real number')
    return float(x) + 0.0


def _integer(x, name):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) or x < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(x)


@dataclass(frozen=True)
class Primitive:
    orbit: tuple[tuple[int, int, int, int], ...]
    trig: str
    harmonic: int = 1
    sign: int = 1

    def __post_init__(self):
        orbit = tuple(tuple(_integer(a, 'atom index') for a in q) for q in self.orbit)
        if not orbit or any(len(q) != 4 or len(set(q)) != 4 for q in orbit):
            raise ValueError('orbit needs quadruplets of four distinct atom indices')
        if len(set(orbit)) != len(orbit):
            raise ValueError('duplicate orbit member')
        if self.trig not in ('sin', 'cos'):
            raise ValueError('trig must be sin or cos')
        if _integer(self.harmonic, 'harmonic') not in (1, 2):
            raise ValueError('harmonic must be 1 or 2')
        if isinstance(self.sign, (bool, np.bool_)) or not isinstance(self.sign, Integral) or self.sign not in (-1, 1):
            raise ValueError('sign must be -1 or +1')
        object.__setattr__(self, 'orbit', tuple(sorted(orbit)))
        object.__setattr__(self, 'harmonic', int(self.harmonic))
        object.__setattr__(self, 'sign', int(self.sign))


def atom_map(production_keys, trajectory_keys):
    """Production index -> trajectory index or None; keys must be unique stable identities.

    Caller validates chemistry/connectivity and builds keys including chain,
    residue/insertion identity and atom name. Missing solvent atoms are allowed;
    Projection.remap refuses missing model atoms. No positional fallback.
    """
    prod, traj = tuple(production_keys), tuple(trajectory_keys)
    if len(set(prod)) != len(prod) or len(set(traj)) != len(traj):
        raise ValueError('duplicate atom identity; mapping is ambiguous')
    where = {key: i for i, key in enumerate(traj)}
    return tuple(where.get(key) for key in prod)


@dataclass(frozen=True)
class Projection:
    primitives: tuple[Primitive, ...]
    coefficients: tuple[float, ...]
    offset: float = 0.0
    scale: float = 1.0

    def __post_init__(self):
        ps = tuple(self.primitives)
        cs = tuple(_real(c, 'coefficient') for c in self.coefficients)
        if len(ps) != len(cs) or any(not isinstance(p, Primitive) for p in ps):
            raise ValueError('one Primitive per coefficient is required')
        scale = _real(self.scale, 'scale')
        if scale <= 0:
            raise ValueError('scale must be positive')
        object.__setattr__(self, 'primitives', ps)
        object.__setattr__(self, 'coefficients', cs)
        object.__setattr__(self, 'offset', _real(self.offset, 'offset'))
        object.__setattr__(self, 'scale', scale)

    def to_mapping(self):
        return {'schema': 'atlas-aux-projection-core-v1',
                'primitives': [{'orbit': [list(q) for q in p.orbit], 'trig': p.trig,
                                'harmonic': p.harmonic, 'sign': p.sign} for p in self.primitives],
                'coefficients': list(self.coefficients), 'offset': self.offset, 'scale': self.scale}

    @classmethod
    def from_mapping(cls, raw):
        keys = {'schema', 'primitives', 'coefficients', 'offset', 'scale'}
        if set(raw) != keys or raw['schema'] != 'atlas-aux-projection-core-v1':
            raise ValueError('unsupported projection core payload')
        return cls(tuple(Primitive(**p) for p in raw['primitives']), tuple(raw['coefficients']),
                   raw['offset'], raw['scale'])

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.to_mapping(), sort_keys=True, separators=(',', ':'),
                                         allow_nan=False).encode()).hexdigest()

    def terms(self):
        """Canonical active (quad, trig, signed harmonic, coefficient/orbit_size) terms.

        Both numerical and force evaluators consume these exact terms. Do not
        merge nearly cancelling terms or discard small coefficients by tolerance.
        """
        out = []
        for p, c in zip(self.primitives, self.coefficients):
            if c != 0.0:
                out.extend((q, p.trig, p.sign*p.harmonic, c/len(p.orbit)) for q in p.orbit)
        return tuple(sorted(out))

    @property
    def quads(self):
        return tuple(sorted({t[0] for t in self.terms()}))

    def _angles(self, xyz):
        x = np.asarray(xyz, dtype=np.float64)
        if x.ndim == 2:
            x = x[None]
        if x.ndim != 3 or x.shape[2] != 3:
            raise ValueError('positions must have shape (frames, atoms, 3) or (atoms, 3)')
        quads = self.quads
        if quads and max(max(q) for q in quads) >= x.shape[1]:
            raise ValueError('model atom outside supplied coordinates; apply explicit atom mapping')
        angles = openmm_dihedrals(x, quads) if quads else np.empty((len(x), 0))
        if not np.all(np.isfinite(angles)):
            raise ValueError('nonfinite or degenerate active torsion geometry')
        return x, angles

    def from_angles(self, theta):
        """Evaluate angles in self.quads order, refusing missing/nonfinite columns."""
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim != 2 or theta.shape[1] != len(self.quads) or not np.isfinite(theta).all():
            raise ValueError('angle basis must be finite and match projection.quads')
        where = {q:i for i,q in enumerate(self.quads)}
        value = np.full(len(theta), self.offset)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            for q, trig, m, c in self.terms():
                value += c * (np.sin if trig == 'sin' else np.cos)(m * theta[:, where[q]])
            value /= self.scale
        if not np.isfinite(value).all():
            raise ValueError('projection overflow')
        return value

    def values(self, xyz):
        return self.from_angles(self._angles(xyz)[1])

    def value_gradient(self, xyz):
        x, angles = self._angles(xyz)
        if len(x) != 1:
            raise ValueError('gradient requires one configuration')
        where = {q:i for i,q in enumerate(self.quads)}
        grad = np.zeros_like(x[0])
        for q, trig, m, c in self.terms():
            theta = angles[0, where[q]]
            derivative = m * (np.cos(m*theta) if trig == 'sin' else -np.sin(m*theta))
            grad[list(q)] += (c/self.scale) * derivative * _dtheta_dx(x[0, list(q)])
        if not np.isfinite(grad).all():
            raise ValueError('nonfinite projection gradient')
        return float(self.from_angles(angles)[0]), grad

    def remap(self, production_to_trajectory):
        mapping = tuple(production_to_trajectory)
        present = [_integer(i, 'mapped index') for i in mapping if i is not None]
        if len(set(present)) != len(present):
            raise ValueError('atom mapping must be injective')
        ps = []
        for p in self.primitives:
            if any(a >= len(mapping) or mapping[a] is None for q in p.orbit for a in q):
                raise ValueError('missing model atom in trajectory mapping')
            ps.append(Primitive(tuple(tuple(mapping[a] for a in q) for q in p.orbit),
                                p.trig, p.harmonic, p.sign))
        return Projection(tuple(ps), self.coefficients, self.offset, self.scale)

    def build_force(self, openmm, *, center, k_kj, force_group=3):
        """Reference compiler; production facade must still supply identity/observer wiring."""
        k = _real(k_kj, 'k_kj')
        if k < 0 or _integer(force_group, 'force_group') > 31:
            raise ValueError('invalid force strength/group')
        center = 0. if k == 0 else _real(center, 'center')
        grouped = {}
        for q, trig, m, c in self.terms():
            grouped.setdefault((trig, m), []).append((q, c))
        force = openmm.CustomCVForce('0')
        names = []
        for (trig, m), terms in sorted(grouped.items()):
            name = f'sc_{trig}_{"neg" if m < 0 else "pos"}{abs(m)}'
            sub = openmm.CustomTorsionForce(f'w*{trig}({m}*theta)')
            sub.addPerTorsionParameter('w')
            sub.setUsesPeriodicBoundaryConditions(False)
            for q, c in terms:
                sub.addTorsion(*q, [c])
            force.addCollectiveVariable(name, sub)
            names.append(name)
        expr = f'(({self.offset:.17g})' + ''.join(f'+{n}' for n in names) + f')/({self.scale:.17g})'
        force.setEnergyFunction(f'select(aux_k,0.5*aux_k*(z-aux_c)^2,0);z={expr}')
        force.addGlobalParameter('aux_k', k)
        force.addGlobalParameter('aux_c', center)
        force.setName('ATLaSAuxCVUmbrella')
        force.setForceGroup(force_group)
        return force
