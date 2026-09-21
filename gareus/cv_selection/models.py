"""Residual torsion components: phi(x) minus its polynomial dependence on the anchor.

For canonical torsion features ``X`` (rows = frames) and a scalar anchor ``a``
(the contact CV), fit by weighted least squares

    m(a) = B0 + B1 a_std [+ B2 a_std^2],      a_std = (a - mu_c) / sigma_c

and diagonalise the weighted covariance of the residual ``R = X - m(a)``. Each
right singular vector ``v_j`` defines one candidate coordinate

    z2^(j) = ( v_j . (R - mean_R) - mu_j ) / sigma_j .

``sum_i w_i z2_i a_std_i == 0`` holds *exactly* because the residual of a
weighted OLS fit with an intercept is w-orthogonal to every regressor column,
and ``a_std`` is a regressor column. The identity is in the WEIGHTED inner
product; the plain ``np.cov`` is not zero under non-uniform weights and must
never be used as the certificate.

Runtime consequence (plan section 5.2): the CV2 umbrella must differentiate
the whole expression, including ``-(v.B1 + 2 a v.B2) grad a / sigma_c``. That
term also makes the CV2 restraint push on the anchor; its induced curvature is
:func:`coupling_curvature_kcal` and is a deployability criterion, because it
varies by orders of magnitude between otherwise interchangeable components.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from ..correctness._io import digest, json_bytes
from . import contracts as C

#: Two adjacent singular values closer than this fraction of the leading one
#: mark an unstable basis: the frozen vector, not the label "PC j", is identity.
_TIE_FRACTION = 1e-6

#: Training quantiles that bound the anchor at runtime for degree-2 models.
_CLAMP_QUANTILES = (0.005, 0.995)


@dataclass(frozen=True)
class ResidualFit:
    """Everything needed to evaluate every component, in float64."""

    coefficients: np.ndarray        # (3, d): rows B0, B1, B2 (B2 all-zero for degree 1)
    residual_mean: np.ndarray       # (d,)
    singular_values: np.ndarray     # (k,)
    right_vectors: np.ndarray       # (k, d), row j-1 is v_j, sign-fixed
    anchor_mean: float
    anchor_std: float
    anchor_clamp: tuple             # (lo, hi) in a_std units, training quantiles
    projection_mean: np.ndarray     # (k,)
    projection_std: np.ndarray      # (k,)
    degree: int
    #: Declared basis transform T applied to the standardised anchor everywhere: "identity"
    #: (degree 1) or "hard_clip" to ``anchor_clamp`` (degree 2). Fitting, scoring, design, the
    #: force and every evaluator use the same T (spec F02; finding I04).
    transform: str = "identity"

    def __post_init__(self):
        expected = "hard_clip" if int(self.degree) == 2 else "identity"
        if self.transform != expected:
            object.__setattr__(self, "transform", expected)

    @property
    def n_components(self) -> int:
        return int(self.right_vectors.shape[0])

    @property
    def width(self) -> int:
        return int(self.right_vectors.shape[1])


def _design(a_std: np.ndarray, degree: int) -> np.ndarray:
    columns = [np.ones_like(a_std), a_std]
    if degree == 2:
        columns.append(a_std ** 2)
    return np.column_stack(columns)


def _normalised_weights(weights, n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be a finite nonnegative vector of length n with positive sum")
    return w / w.sum()


def fit_residual_components(X, anchor, *, degree: int = 1,
                            n_components: int = C.MAX_COMPONENT_INDEX,
                            weights=None) -> ResidualFit:
    X = np.asarray(X, dtype=np.float64)
    a = np.asarray(anchor, dtype=np.float64)
    if degree not in (1, 2):
        raise ValueError("degree must be 1 or 2")
    if X.ndim != 2 or a.shape != (X.shape[0],):
        raise ValueError("X must be (n, d) and anchor (n,)")
    if not (np.isfinite(X).all() and np.isfinite(a).all()):
        raise ValueError("non-finite input")
    n, d = X.shape
    w = _normalised_weights(weights, n)
    mu_c = float(np.sum(w * a))
    sd_c = float(np.sqrt(np.sum(w * (a - mu_c) ** 2)))
    if sd_c <= 1e-12:
        raise ValueError("anchor has zero variance; a constant anchor defines no coordinate")
    a_std = (a - mu_c) / sd_c
    # Degree 2 regresses on the CLIPPED anchor: the bounds are frozen from the training
    # measure before the fit, so the fitted coordinate and the deployed one are the same
    # function everywhere, tails included. Degree 1 is the identity transform.
    clamp = (float(_weighted_quantile(a_std, w, _CLAMP_QUANTILES[0])),
             float(_weighted_quantile(a_std, w, _CLAMP_QUANTILES[1])))
    if not clamp[0] < clamp[1]:
        raise ValueError("anchor clip bounds collapsed; the anchor has no spread on the training measure")
    t = np.clip(a_std, *clamp) if degree == 2 else a_std
    design = _design(t, degree)
    sqrt_w = np.sqrt(w)[:, None]
    B, _, rank, _ = np.linalg.lstsq(design * sqrt_w, X * sqrt_w, rcond=1e-12)
    if rank < design.shape[1]:
        raise ValueError(f"design matrix rank {rank} < {design.shape[1]}; residualisation is undefined")
    if degree == 1:
        B = np.vstack([B, np.zeros((1, d))])
    residual = X - _design(t, 2) @ B
    mean_R = np.sum(w[:, None] * residual, axis=0)
    _, singular, Vt = np.linalg.svd((residual - mean_R) * sqrt_w, full_matrices=False)
    k = min(int(n_components), Vt.shape[0])
    V = Vt[:k].copy()
    for row in V:
        pivot = row[np.argmax(np.abs(row))]
        row *= np.sign(pivot) if pivot != 0.0 else 1.0
    scores = (residual - mean_R) @ V.T
    mu = np.sum(w[:, None] * scores, axis=0)
    sd = np.sqrt(np.sum(w[:, None] * (scores - mu) ** 2, axis=0))
    if np.any(sd <= 1e-12):
        raise ValueError("a component has zero variance on the training data")
    return ResidualFit(B, mean_R, singular[:k], V, mu_c, sd_c, clamp, mu, sd, int(degree),
                       "hard_clip" if degree == 2 else "identity")


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="stable")
    cdf = np.cumsum(weights[order])
    cdf = cdf / cdf[-1]
    return float(values[order][int(np.searchsorted(cdf, q, side="left").clip(0, values.size - 1))])


def standardised_anchor(fit: ResidualFit, anchor) -> np.ndarray:
    """T(a): the fit's declared transform of the standardised anchor -- never a caller's choice."""
    a_std = (np.asarray(anchor, dtype=np.float64) - fit.anchor_mean) / fit.anchor_std
    return np.clip(a_std, *fit.anchor_clamp) if fit.transform == "hard_clip" else a_std


def evaluate_component(fit: ResidualFit, j: int, X, anchor) -> np.ndarray:
    """Standardised ``z2^(j)`` for every row of ``X`` -- the compiled coordinate, nothing else.

    Delegates to :func:`gareus.cv_selection.residual_runtime.compile_component` so selection,
    design, seed scoring, the force builder, the fast-path scalar, the positions evaluator and
    reprojection all evaluate one definition (spec F02). ``norm`` does not enter here because
    ``anchor`` is already the normalised contact CV.
    """
    from .residual_runtime import compile_component

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != fit.width:
        raise ValueError(f"X must be (n, {fit.width})")
    return compile_component(fit, int(j), norm=1.0).evaluate_features(X, anchor)


def coupling_curvature_kcal(fit: ResidualFit, j: int, k2_kcal: float) -> float:
    """Curvature the CV2 umbrella induces along the anchor, d²U₂/dc² at a_std = 0.

    With z2 = (v.phi - K0 - K1 a - K2 a²)/sigma_j and a = (c - mu_c)/sigma_c,
    dz2/dc = -(K1 + 2 K2 a)/(sigma_j sigma_c), so the harmonic CV2 restraint
    contributes k2 (dz2/dc)² of curvature along c. Reported at a = 0.
    """
    v = fit.right_vectors[int(j) - 1]
    slope = float(v @ fit.coefficients[1])
    return float(k2_kcal) * slope ** 2 / (float(fit.projection_std[int(j) - 1]) ** 2
                                          * float(fit.anchor_std) ** 2)


def _tie_flags(singular: np.ndarray) -> list[bool]:
    if singular.size == 0:
        return []
    scale = max(float(singular[0]), 1e-300)
    flags = []
    for i, s in enumerate(singular):
        near = []
        if i > 0:
            near.append(abs(float(s) - float(singular[i - 1])))
        if i + 1 < singular.size:
            near.append(abs(float(s) - float(singular[i + 1])))
        flags.append(bool(near) and min(near) < _TIE_FRACTION * scale)
    return flags


def to_candidate_set(fit: ResidualFit, feature_schema: C.FeatureSchema,
                     primary_definition: Mapping[str, Any], physical_system_sha256: str,
                     training_rows_sha256: str, library_versions: Mapping[str, str],
                     *, design_measure: str = "unspecified") -> C.CandidateSet:
    """Freeze the fit as the v2 contract artifact every later stage reads (declared basis)."""
    if fit.width != feature_schema.width:
        raise ValueError(f"fit width {fit.width} != feature schema width {feature_schema.width}")
    ties = _tie_flags(fit.singular_values)
    components = []
    for j in range(1, fit.n_components + 1):
        components.append({
            "component_index": j,
            "singular_value": float(fit.singular_values[j - 1]),
            "eigenvalue_tie_flagged": ties[j - 1],
            "right_singular_vector": fit.right_vectors[j - 1].tolist(),
            "residual_mean": fit.residual_mean.tolist(),
            "regression_coefficients": [row.tolist() for row in fit.coefficients],
            "primary_mean": float(fit.anchor_mean),
            "primary_std": float(fit.anchor_std),
            "anchor_clamp": [float(fit.anchor_clamp[0]), float(fit.anchor_clamp[1])],
            "projection_mean": float(fit.projection_mean[j - 1]),
            "projection_std": float(fit.projection_std[j - 1]),
        })
    return C.CandidateSet.from_mapping({
        "schema": C.CANDIDATE_SET_VERSION_V2,
        "kind": C.CANDIDATE_KIND_QUADRATIC_RESIDUAL,
        "feature_schema_sha256": feature_schema.sha256,
        "physical_system_sha256": physical_system_sha256,
        "training_rows_sha256": training_rows_sha256,
        "library_versions": dict(library_versions),
        "primary_definition": dict(primary_definition),
        "components": components,
        "basis_transform": {"kind": fit.transform, "lo": float(fit.anchor_clamp[0]), "hi": float(fit.anchor_clamp[1])},
        "design_measure": str(design_measure),
    })


def from_candidate_set(candidates: C.CandidateSet) -> ResidualFit:
    """Invert :func:`to_candidate_set` exactly (the JSON holds float64 verbatim)."""
    comps = sorted(candidates.components, key=lambda c: c.component_index)
    if [c.component_index for c in comps] != list(range(1, len(comps) + 1)):
        raise ValueError("candidate set components must be 1..k without gaps")
    first = comps[0]
    B = np.asarray(first.regression_coefficients, dtype=np.float64)
    degree = 2 if np.any(B[2] != 0.0) else 1
    return ResidualFit(
        coefficients=B,
        residual_mean=np.asarray(first.residual_mean, dtype=np.float64),
        singular_values=np.asarray([c.singular_value for c in comps], dtype=np.float64),
        right_vectors=np.asarray([c.right_singular_vector for c in comps], dtype=np.float64),
        anchor_mean=float(first.primary_mean),
        anchor_std=float(first.primary_std),
        anchor_clamp=(float(first.anchor_clamp[0]), float(first.anchor_clamp[1])),
        projection_mean=np.asarray([c.projection_mean for c in comps], dtype=np.float64),
        projection_std=np.asarray([c.projection_std for c in comps], dtype=np.float64),
        degree=degree,
        transform=str(candidates.basis_transform["kind"]),
    )


# ---------------------------------------------------------------------------
# Runtime: the frozen pair as production loads it
# ---------------------------------------------------------------------------

#: Anchor-definition keys and the ``args`` attribute each must equal at deploy time.
_ANCHOR_ARG_KEYS = (
    ("r0_angstrom", "contact_r0_a"),
    ("beta_per_angstrom", "contact_beta_a_inv"),
    ("min_sequence_separation", "contact_min_sequence_separation"),
    ("atom_selection", "contact_atom_selection"),
    ("normalize", "contact_normalize"),
)


def validate_pair_semantics(pair, candidates: C.CandidateSet, schema: C.FeatureSchema) -> None:
    """Cross-artifact consistency a digest cannot express (spec F02 'artifact semantics')."""
    from .residual_runtime import check_feature_schema_layout

    if dict(pair.anchor) != dict(candidates.primary_definition):
        raise RuntimeError("pair model anchor differs from the candidate set's primary definition; "
                           "the pair was not selected from these candidates")
    indices = {c.component_index for c in candidates.components}
    if int(pair.selected_component_index) not in indices:
        raise RuntimeError(f"pair model selects component {pair.selected_component_index}, which the "
                           f"candidate set does not contain ({sorted(indices)})")
    if int(pair.degree) != int(candidates.degree):
        raise RuntimeError(f"pair model declares degree {pair.degree} but the candidate coefficients imply "
                           f"degree {candidates.degree}")
    if pair.basis_transform is not None and dict(pair.basis_transform) != dict(candidates.basis_transform):
        raise RuntimeError(f"pair model basis transform {pair.basis_transform} differs from the candidate "
                           f"set's {candidates.basis_transform}")
    n_phi, n_psi = check_feature_schema_layout(schema)
    if 2 * (n_phi + n_psi) != candidates.width:
        raise RuntimeError("feature schema torsion count does not match the candidate width")


def contact_pair_list_digest(contact_pairs) -> str:
    """Digest of the realised contact pair list, order-independent.

    Binding the pair list, not just its parameters, is what stops a model
    fitted on one contact definition from being deployed against another --
    and is the one place a native contact map could otherwise slip in.
    """
    rows = sorted([int(p[0]), int(p[1]), float(p[2]) if len(p) > 2 else 1.0] for p in contact_pairs)
    return digest(json_bytes(rows))


def _same(expected: Any, got: Any) -> bool:
    if isinstance(expected, bool) or isinstance(got, bool):
        return bool(expected) == bool(got)
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        return bool(np.isclose(float(expected), float(got), rtol=1e-9, atol=0.0))
    return expected == got


@dataclass(frozen=True)
class PairModelRuntime:
    """One selected component plus everything needed to force and evaluate it."""

    fit: ResidualFit
    j: int
    anchor_kind: str
    anchor_definition: dict
    pair_sha256: str
    feature_atoms: tuple            # one quadruplet per torsion: phi block then psi block
    legacy: bool = False            # v1 artifacts, read in explicit legacy mode
    deployable: bool = True

    @classmethod
    def load(cls, pair_model_path, candidate_set_path, feature_schema_path, *,
             require_deployable: bool = True, allow_legacy_v1: bool = False) -> "PairModelRuntime":
        """Load the three artifacts, hold them to each other's digests AND to each other's
        semantics (spec F02): the pair's anchor is the candidates' primary definition, the
        selected component exists, degree and basis transform agree, the feature layout is one
        the force builder compiles, and -- for deployment -- the bindings are real.
        """
        from .pair_model import PairModel
        from .residual_runtime import check_feature_schema_layout

        pair = PairModel.from_json_bytes(Path(pair_model_path).read_bytes())
        candidates = C.CandidateSet.from_json_bytes(Path(candidate_set_path).read_bytes())
        schema = C.FeatureSchema.from_json_bytes(Path(feature_schema_path).read_bytes())
        if pair.candidate_set_sha256 != candidates.sha256:
            raise RuntimeError(f"pair model binds candidate set {pair.candidate_set_sha256} but "
                               f"{candidate_set_path} hashes to {candidates.sha256}")
        if pair.feature_schema_sha256 != schema.sha256:
            raise RuntimeError(f"pair model binds feature schema {pair.feature_schema_sha256} but "
                               f"{feature_schema_path} hashes to {schema.sha256}")
        C.require_feature_binding(candidates, schema)
        validate_pair_semantics(pair, candidates, schema)
        if pair.legacy or candidates.legacy:
            if not allow_legacy_v1:
                raise RuntimeError(
                    "legacy v1 residual artifacts: their degree-2 certificate describes the unclipped "
                    "training calculation while the deployed coordinate was clipped (review I04). Reading "
                    "them reproduces the OLD force definition exactly; pass allow_legacy_v1=True "
                    "(--legacy-model-policy allow-v1) to deploy them knowingly, or re-run the swarm "
                    "analysis to produce v2 artifacts")
        if require_deployable and not pair.deployable:
            raise RuntimeError(
                "pair model is not deployable: it carries no real topology/system/contact-pair bindings "
                f"(deployment={pair.deployment}); a discovery-only artifact cannot restrain production")
        atoms = tuple(tuple(int(i) for i in f.atom_indices) for f in schema.features if f.trig == "sin")
        return cls(from_candidate_set(candidates), int(pair.selected_component_index),
                   str(pair.anchor["kind"]), dict(pair.anchor["definition"]), pair.sha256, atoms,
                   legacy=bool(pair.legacy or candidates.legacy), deployable=bool(pair.deployable))

    def contact_args(self) -> SimpleNamespace:
        """The contact parameters as an ``args``-shaped object, from the frozen definition."""
        d = self.anchor_definition
        return SimpleNamespace(
            contact_r0_a=float(d["r0_angstrom"]),
            contact_beta_a_inv=float(d["beta_per_angstrom"]),
            contact_normalize=bool(d.get("normalize", True)),
            contact_min_sequence_separation=int(d.get("min_sequence_separation", 0)),
            contact_atom_selection=str(d.get("atom_selection", "heavy")),
        )

    def check_topology(self, phi_torsions, psi_torsions) -> None:
        """The production topology must yield the exact torsion quadruplets the model was fitted on."""
        got = tuple(tuple(int(i) for i in q) for q in list(phi_torsions) + list(psi_torsions))
        if len(got) != len(self.feature_atoms):
            raise RuntimeError(f"feature schema defines {len(self.feature_atoms)} torsions, topology "
                               f"yields {len(got)}")
        for k, (expected, actual) in enumerate(zip(self.feature_atoms, got)):
            if tuple(expected) != actual:
                raise RuntimeError(f"feature schema torsion {k} is atoms {tuple(expected)}, topology "
                                   f"yields {actual}; same width, different coordinate")

    def check_anchor(self, args, contact_pairs) -> None:
        """Every frozen anchor parameter must equal the run's, including the pair list itself."""
        from ..cv import contact_normalization_denominator

        d = self.anchor_definition
        for key, attr in _ANCHOR_ARG_KEYS:
            if key in d and not _same(d[key], getattr(args, attr, None)):
                raise RuntimeError(f"anchor definition mismatch: model {key}={d[key]!r}, run "
                                   f"{attr}={getattr(args, attr, None)!r}")
        if "norm" in d:
            live = contact_normalization_denominator(list(contact_pairs), args)
            if not _same(d["norm"], live):
                raise RuntimeError(f"anchor normalisation mismatch: model {d['norm']}, run {live}")
        if "pair_list_sha256" in d:
            live_digest = contact_pair_list_digest(contact_pairs)
            if d["pair_list_sha256"] != live_digest:
                raise RuntimeError("anchor contact pair list differs from the one the model was "
                                   f"fitted on ({d['pair_list_sha256'][:12]} vs {live_digest[:12]})")
