"""One compiled residual coordinate for every stage (repair spec F02; finding I04).

Fitting, candidate scoring, layout design, seed scoring, the OpenMM force expression, the
fast-path scalar, the positions evaluator and offline reprojection used to carry their own
copies of "the" residual coordinate, and disagreed for degree 2: selection and design
evaluated the unbounded polynomial while the force and the runtime evaluators clamped the
anchor. `CompiledResidualComponent` is the single frozen definition all of them consume.

    a = (c - mu_c) / sigma_c          c = contact_sum / norm
    t = T(a)                          identity, or hard_clip(lo, hi) for degree-2 models
    z = (v . f - K0 - K1 t - K2 t^2) / sigma_j
    K0 = v . (B0 + mean_R) + mu_j,  K1 = v . B1,  K2 = v . B2

The methods share frozen parameters; the tests check them against an independently written
oracle, so a shared implementation mistake is detectable.

NumPy only. Python 3.9 compatible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

TRANSFORM_IDENTITY = "identity"
TRANSFORM_HARD_CLIP = "hard_clip"
TRANSFORMS = frozenset({TRANSFORM_IDENTITY, TRANSFORM_HARD_CLIP})

#: Feature layout the force builder and the feature extractor agree on: phi block then psi
#: block, each torsion as (sin, cos), dihedral sign "negated" (OpenMM's theta is minus the
#: extractor's dihedral). Anything else is refused at compile time, never hashed and shipped.
SUPPORTED_FEATURE_LAYOUT = "phi-block-then-psi-block/sin-cos-interleaved/negated"

ROLE_TORSION_SUM = "torsion_sum"
ROLE_CONTACT_SUM = "contact_sum"


def _f(value) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"non-finite compiled parameter {value!r}")
    return out


@dataclass(frozen=True)
class CompiledResidualComponent:
    """Scalar coefficients of one residual direction plus its feature weights. Immutable."""

    component_index: int
    feature_weights: Tuple[float, ...]          # v, in schema feature order
    k0: float
    k1: float
    k2: float
    sigma_j: float
    anchor_mean: float
    anchor_std: float
    norm: float                                 # contact normalisation denominator
    transform: str
    clamp_lo: float
    clamp_hi: float
    degree: int
    feature_layout: str = SUPPORTED_FEATURE_LAYOUT

    def __post_init__(self):
        if self.transform not in TRANSFORMS:
            raise ValueError(f"unknown basis transform {self.transform!r}")
        if self.degree not in (1, 2):
            raise ValueError("degree must be 1 or 2")
        if self.degree == 1 and self.k2 != 0.0:
            raise ValueError("a degree-1 component cannot carry a quadratic coefficient")
        if self.degree == 1 and self.transform != TRANSFORM_IDENTITY:
            raise ValueError("degree-1 components use the identity transform")
        if self.degree == 2 and self.transform != TRANSFORM_HARD_CLIP:
            raise ValueError("degree-2 components use the hard_clip transform in this version")
        if not self.clamp_lo < self.clamp_hi:
            raise ValueError("clamp_lo must be < clamp_hi")
        for name in ("sigma_j", "anchor_std", "norm"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be positive")
        if self.feature_layout != SUPPORTED_FEATURE_LAYOUT:
            raise ValueError(f"unsupported feature layout {self.feature_layout!r}")

    # ----------------------------------------------------------------------------- basis
    @property
    def width(self) -> int:
        return len(self.feature_weights)

    def standardised_anchor(self, anchor_values) -> np.ndarray:
        return (np.asarray(anchor_values, dtype=np.float64) - self.anchor_mean) / self.anchor_std

    def transform_anchor(self, a_std) -> np.ndarray:
        a_std = np.asarray(a_std, dtype=np.float64)
        if self.transform == TRANSFORM_HARD_CLIP:
            return np.clip(a_std, self.clamp_lo, self.clamp_hi)
        return a_std

    # ----------------------------------------------------------------------- evaluators
    def evaluate_features(self, features, anchor_values) -> np.ndarray:
        """z for every row of ``features`` (n, width) at contact CV values ``anchor_values`` (n,)."""
        X = np.asarray(features, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.width:
            raise ValueError(f"features have width {X.shape[1]}, component has {self.width}")
        a = np.asarray(anchor_values, dtype=np.float64).reshape(-1)
        if a.shape[0] != X.shape[0]:
            raise ValueError("one anchor value per feature row is required")
        t = self.transform_anchor(self.standardised_anchor(a))
        proj = X @ np.asarray(self.feature_weights, dtype=np.float64)
        return (proj - self.k0 - self.k1 * t - self.k2 * t * t) / self.sigma_j

    def evaluate_subcvs(self, named_values: Mapping[str, float], roles: Sequence[Mapping[str, str]]) -> float:
        """z from a force's cached sub-variables: the torsion sums and the raw contact sum."""
        torsion_sum = 0.0
        contact = None
        for role in roles:
            name, kind = str(role["name"]), str(role["role"])
            value = float(named_values[name])
            if not np.isfinite(value):
                raise ValueError(f"sub-CV {name!r} is not finite")
            if kind == ROLE_TORSION_SUM:
                torsion_sum += value
            elif kind == ROLE_CONTACT_SUM:
                if contact is not None:
                    raise ValueError("more than one contact_sum role")
                contact = value
            else:
                raise ValueError(f"unknown sub-CV role {kind!r}")
        if contact is None:
            raise ValueError("no contact_sum role")
        t = float(self.transform_anchor(self.standardised_anchor(contact / self.norm)))
        return float((torsion_sum - self.k0 - self.k1 * t - self.k2 * t * t) / self.sigma_j)

    def openmm_expression(self, torsion_sum_names: Sequence[str], contact_name: str) -> str:
        """The z expression over named CustomCVForce sub-variables (kJ-neutral: no unit factor)."""
        if not torsion_sum_names:
            raise ValueError("at least one torsion-sum sub-variable is required")
        a_raw = f"((({contact_name})/{self.norm:.17g}) - {self.anchor_mean:.17g})/{self.anchor_std:.17g}"
        if self.transform == TRANSFORM_HARD_CLIP:
            t = f"min({self.clamp_hi:.17g}, max({self.clamp_lo:.17g}, {a_raw}))"
        else:
            t = a_raw
        return (f"(({' + '.join(torsion_sum_names)}) - {self.k0:.17g} - {self.k1:.17g}*({t}) - "
                f"{self.k2:.17g}*({t})^2)/{self.sigma_j:.17g}")

    def anchor_partials(self, features, anchor_values, *, kink_tol: float = 1e-9):
        """(z, dz/dc, d2z/dc2, kink_mask) holding the torsion projection fixed.

        c is the raw contact CV. Inside the clip the transform is the identity in a, so
        dz/dc = -(K1 + 2 K2 t) / (sigma_j sigma_c) and d2z/dc2 = -2 K2 / (sigma_j sigma_c^2);
        outside it both vanish. At the clip boundary the derivative is discontinuous: those
        rows are flagged, not differentiated across (spec F06).
        """
        z = self.evaluate_features(features, anchor_values)
        a = self.standardised_anchor(np.asarray(anchor_values, dtype=np.float64).reshape(-1))
        t = self.transform_anchor(a)
        inside = np.ones_like(a, dtype=bool)
        kink = np.zeros_like(a, dtype=bool)
        if self.transform == TRANSFORM_HARD_CLIP:
            inside = (a > self.clamp_lo) & (a < self.clamp_hi)
            kink = (np.abs(a - self.clamp_lo) <= kink_tol) | (np.abs(a - self.clamp_hi) <= kink_tol)
        dz_dc = np.where(inside, -(self.k1 + 2.0 * self.k2 * t) / (self.sigma_j * self.anchor_std), 0.0)
        d2z_dc2 = np.where(inside, -2.0 * self.k2 / (self.sigma_j * self.anchor_std ** 2), 0.0)
        return z, dz_dc, d2z_dc2, kink

    # ------------------------------------------------------------------------- identity
    def as_record(self) -> Dict[str, object]:
        """JSON-ready scalar record (what the force builder stores as ``residual_scalar``)."""
        return {
            "component_index": int(self.component_index), "K0": self.k0, "K1": self.k1, "K2": self.k2,
            "sigma_j": self.sigma_j, "norm": self.norm, "anchor_mean": self.anchor_mean,
            "anchor_std": self.anchor_std, "degree": int(self.degree), "transform": self.transform,
            "clamp_lo": self.clamp_lo, "clamp_hi": self.clamp_hi, "feature_layout": self.feature_layout,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object], feature_weights: Sequence[float]) -> "CompiledResidualComponent":
        return cls(
            component_index=int(record.get("component_index", 0)),
            feature_weights=tuple(float(w) for w in feature_weights),
            k0=_f(record["K0"]), k1=_f(record["K1"]), k2=_f(record["K2"]), sigma_j=_f(record["sigma_j"]),
            anchor_mean=_f(record["anchor_mean"]), anchor_std=_f(record["anchor_std"]), norm=_f(record["norm"]),
            transform=str(record["transform"]), clamp_lo=_f(record["clamp_lo"]), clamp_hi=_f(record["clamp_hi"]),
            degree=int(record["degree"]), feature_layout=str(record.get("feature_layout", SUPPORTED_FEATURE_LAYOUT)),
        )


def compile_component(fit, j: int, *, norm: float) -> CompiledResidualComponent:
    """Compile component ``j`` (1-based) of a :class:`~gareus.cv_selection.models.ResidualFit`."""
    j = int(j)
    if not 1 <= j <= fit.n_components:
        raise ValueError(f"component {j} out of range 1..{fit.n_components}")
    v = np.asarray(fit.right_vectors[j - 1], dtype=np.float64)
    B = np.asarray(fit.coefficients, dtype=np.float64)
    k0 = float(v @ (B[0] + np.asarray(fit.residual_mean, dtype=np.float64)) + float(fit.projection_mean[j - 1]))
    k1 = float(v @ B[1])
    k2 = float(v @ B[2]) if fit.degree == 2 else 0.0
    if fit.degree == 1 and np.any(B[2] != 0.0):
        raise ValueError("degree-1 fit carries a nonzero quadratic row")
    transform = getattr(fit, "transform", None) or (TRANSFORM_HARD_CLIP if fit.degree == 2 else TRANSFORM_IDENTITY)
    lo, hi = (float(fit.anchor_clamp[0]), float(fit.anchor_clamp[1]))
    return CompiledResidualComponent(
        component_index=j, feature_weights=tuple(float(x) for x in v), k0=k0, k1=k1, k2=k2,
        sigma_j=float(fit.projection_std[j - 1]), anchor_mean=float(fit.anchor_mean), anchor_std=float(fit.anchor_std),
        norm=float(norm), transform=str(transform), clamp_lo=lo, clamp_hi=hi, degree=int(fit.degree),
    )


def check_feature_schema_layout(schema) -> Tuple[int, int]:
    """Assert the schema is in the supported layout and return (n_phi, n_psi).

    phi block first, then psi block; within a torsion sin precedes cos; every torsion has
    both; the dihedral sign convention is 'negated'. Anything else is refused (spec F02).
    """
    features = list(schema.features)
    n = len(features)
    if n % 2:
        raise ValueError("feature schema width must be even (sin, cos per torsion)")
    n_phi = 0
    n_psi = 0
    seen_psi = False
    for k in range(0, n, 2):
        s, c = features[k], features[k + 1]
        if s.trig != "sin" or c.trig != "cos":
            raise ValueError(f"features {k},{k + 1}: expected (sin, cos), got ({s.trig}, {c.trig})")
        if s.torsion_name != c.torsion_name or tuple(s.atom_indices) != tuple(c.atom_indices):
            raise ValueError(f"features {k},{k + 1} do not describe the same torsion")
        if s.dihedral_sign_convention != "negated" or c.dihedral_sign_convention != "negated":
            raise ValueError("only the 'negated' dihedral sign convention is compiled")
        block = s.torsion_name.split("-")[0]
        if block == "phi":
            if seen_psi:
                raise ValueError("phi torsion after the psi block: unsupported feature order")
            n_phi += 1
        elif block == "psi":
            seen_psi = True
            n_psi += 1
        else:
            raise ValueError(f"unknown torsion block {block!r}")
    return n_phi, n_psi
