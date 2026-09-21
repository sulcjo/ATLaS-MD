"""Names for the numerical kernels that define a scientific segment (repair spec F01/F04).

A changed CV evaluator or exchange-energy assembly is a different kernel: samples recorded
under one cannot be appended to under another. These identifiers describe implementation
semantics, not git SHAs; bump them when the arithmetic changes, never for a refactor that
leaves every recorded number identical.

Kept import-light on purpose: ``production.py`` and ``provenance.py`` both import it.
"""
from __future__ import annotations

#: Full-expression residual scalar reconstruction from the ordered sub-CV roles recorded by
#: the force builder (all torsion sums, contact normalisation, anchor polynomial with its
#: declared transform, offset and scale). Before this version the fast path fell through to
#: the legacy two-term average and recorded a different coordinate from the one the force
#: applied (review finding I01). Absent/other values are the affected kernel.
RESIDUAL_EVALUATOR_VERSION = "residual_full_expression_v1"

#: One shared assembly of the [state, replica] umbrella matrices for sampling and exchange,
#: with the non-finite secondary-CV guard applied to both.
EXCHANGE_ENERGY_VERSION = "state_bias_matrix_v2"

#: Secondary-CV modes whose scalar can be reconstructed from CustomCVForce sub-variables.
#: Anything else must take the positions-based evaluator; an unknown mode raises.
FAST_SCALAR_MODES = frozenset({
    "alpha", "beta", "custom", "alpha-coil-beta", "rama-map",
    "tica-linear", "torsion-pca", "residual-torsion-pc",
})

#: Legacy modes whose scalar is the mean of exactly two sub-CVs (phi score, psi score).
LEGACY_TWO_TERM_MODES = frozenset({"alpha", "beta", "custom"})

RESIDUAL_MODE = "residual-torsion-pc"

#: Sample eligibility for validated equilibrium claims (spec F04).
ELIGIBLE_VERIFIED = "verified"          # kernel recorded and equal to the current one
ELIGIBLE_AFFECTED = "affected"          # recorded kernel is a known-wrong one
ELIGIBLE_UNKNOWN = "unknown"            # residual mode but no kernel record (pre-F01 segments)
ELIGIBLE_NOT_APPLICABLE = "not_applicable"   # no residual CV: the F01 defect cannot have touched it


def kernel_identity_for_run(args, secondary_cv_metadata=None) -> dict:
    """The numerical kernel of a segment about to be written, as JSON-ready fields.

    Physical-state identity (windows, envelope) lives in the state table and snapshots;
    this is the *arithmetic* that turns a configuration into recorded coordinates and
    exchange energies. Bound into every window snapshot so a later reader can tell a
    verified segment from an affected or unknown one without guessing from dates.
    """
    import hashlib
    import json

    meta = dict(secondary_cv_metadata or {})
    mode = str(meta.get("mode") or getattr(args, "secondary_cv", "none") or "none")
    residual = bool(meta.get("enabled")) and mode == RESIDUAL_MODE
    identity = {
        "kernel_identity_version": "kernel_identity_v1",
        "secondary_cv_mode": mode if meta.get("enabled") else "none",
        "cv_evaluator_version": (str(meta.get("cv_evaluator_version")) if residual and meta.get("cv_evaluator_version")
                                 else (None if not residual else "affected_pre_f01")),
        "exchange_energy_version": EXCHANGE_ENERGY_VERSION,
        "pair_model_sha256": meta.get("pair_model_sha256") if residual else None,
        "production_ensemble": str(getattr(args, "production_ensemble", "") or ""),
        "gamd_boost_type": str(getattr(args, "gamd_boost_type", "") or ""),
    }
    body = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    identity["digest"] = hashlib.sha256(body).hexdigest()
    return identity


def classify_segment_kernel(window_snapshot: dict) -> tuple:
    """(eligibility, reason) of one segment from its ``windows/<segment>.json`` payload.

    Non-residual segments are not affected by the residual fast-path defect and stay
    eligible under their own documented rules; a residual segment is verified only when its
    recorded evaluator and exchange versions equal the current ones.
    """
    snap = dict(window_snapshot or {})
    cv2 = str(snap.get("cv2_type") or "none")
    identity = snap.get("kernel_identity") or {}
    if cv2 != RESIDUAL_MODE:
        return ELIGIBLE_NOT_APPLICABLE, "secondary CV is not residual-torsion-pc"
    if not identity:
        return ELIGIBLE_UNKNOWN, "residual segment without a kernel_identity record (written before F01)"
    ev = identity.get("cv_evaluator_version")
    ex = identity.get("exchange_energy_version")
    if ev == RESIDUAL_EVALUATOR_VERSION and ex == EXCHANGE_ENERGY_VERSION:
        return ELIGIBLE_VERIFIED, "kernel matches the current evaluator and exchange assembly"
    if ev in (None, "affected_pre_f01"):
        return ELIGIBLE_AFFECTED, "recorded coordinates came from the two-term fall-through (review I01)"
    return ELIGIBLE_UNKNOWN, f"kernel {ev!r}/{ex!r} is not the current {RESIDUAL_EVALUATOR_VERSION!r}/{EXCHANGE_ENERGY_VERSION!r}"
