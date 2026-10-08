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

#: Auxiliary-CV segment whose samples carry no z / torsion features (Stage B engineering runs):
#: its bias cannot be reconstructed for any analysis, whatever its CV2 mode.
ELIGIBLE_AUX_UNPERSISTED = "aux_unpersisted"

#: The same shared assembly plus the exact auxiliary-CV restraint term evaluated for every
#: carrier under every active state (spec 2026-10-07 auxiliary CV, Section 5). Used whenever an
#: auxiliary model is configured, including sham arms whose strengths are all zero.
EXCHANGE_ENERGY_VERSION_AUX = "state_bias_matrix_v3_aux"

#: Payload schema a Stage C samples manifest records when every row stores the auxiliary z and the
#: full torsion basis (gareus.auxiliary_cv.sample_schema.AUX_SAMPLES_SCHEMA; kept here import-light).
AUX_SAMPLES_PAYLOAD_SCHEMA = "atlas-aux-samples-v1"


def raw_sample_payload_schema(run_dir, segment_id) -> "dict | None":
    """``payload_schema`` of ``samples/<segment>/`` as raw JSON; never validates files, never raises."""
    import json
    from pathlib import Path
    from .parquet_manifest import manifest_path
    try:
        raw = json.loads(manifest_path(Path(run_dir) / "samples" / str(segment_id)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    payload = raw.get("payload_schema") if isinstance(raw, dict) else None
    return payload if isinstance(payload, dict) else None


class AuxPoolingRefused(RuntimeError):
    """An auxiliary-CV run reached a pooling path that cannot reconstruct its bias (CVaux Stage C)."""


#: Window-snapshot globs per depth: the run itself, then adaptive phases and their top-ups
#: (adaptive_production/epoch_NNN/windows, adaptive_production/epoch_NNN/topup_*/windows). Never rglob:
#: a union build runs every epoch over trees holding checkpoint generations and trajectories.
_SNAPSHOT_GLOBS = ("windows/*.json", "*/windows/*.json", "*/*/windows/*.json")


def _positive(value) -> bool:
    try:
        return float(value or 0.0) > 0.0
    except (TypeError, ValueError):
        return False


def snapshot_has_aux(payload) -> bool:
    """A windows/<segment>.json payload that carries auxiliary states (kernel identity, frozen v2
    aux_models, or any row with auxiliary fields: Stage B D5a rows included)."""
    if not isinstance(payload, dict):
        return False
    if is_aux_kernel_record(payload.get("kernel_identity")):
        return True
    state = payload.get("state_definition")
    state = state if isinstance(state, dict) else {}
    if state.get("aux_models"):
        return True
    rows = list(payload.get("windows") or []) + list(state.get("windows") or [])
    return any(isinstance(r, dict) and ("aux_model_sha256" in r or _positive(r.get("aux_k"))) for r in rows)


def aux_snapshot_hits(root, *, depth: int = 0) -> list:
    """Snapshot paths with auxiliary states under fixed-depth globs (read-only; unreadable files skipped)."""
    import json
    from pathlib import Path
    root = Path(root)
    hits = []
    for pattern in _SNAPSHOT_GLOBS[: int(depth) + 1]:
        for path in sorted(root.glob(pattern)):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if snapshot_has_aux(payload):
                hits.append(str(path))
    return hits


def run_has_aux(prod) -> bool:
    """Auxiliary evidence for ONE run directory (read-only JSON; legacy output unchanged)."""
    import json
    from pathlib import Path
    prod = Path(prod)
    try:
        manifest = json.loads((prod / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    if isinstance(manifest, dict) and is_aux_kernel_record(manifest.get("method_settings")):
        return True
    if aux_snapshot_hits(prod, depth=0):
        return True
    samples = prod / "samples"
    if samples.is_dir():
        for seg_dir in sorted(p for p in samples.iterdir() if p.is_dir()):
            if (raw_sample_payload_schema(prod, seg_dir.name) or {}).get("schema") == AUX_SAMPLES_PAYLOAD_SCHEMA:
                return True
    return False


def refuse_aux_snapshots(root, *, where: str, depth: int = 2) -> None:
    """Raise AuxPoolingRefused when any window snapshot under ``root`` (fixed depth) carries auxiliary states."""
    hits = aux_snapshot_hits(root, depth=depth)
    if hits:
        raise AuxPoolingRefused(f"{where}: auxiliary states pool only through the strict fixed-state exporter or "
                                f"load_parquet in the MVP (spec Sections 8/14); found {hits[:3]}")


def exchange_energy_version_for_args(args) -> str:
    return EXCHANGE_ENERGY_VERSION_AUX if getattr(args, "aux_cv_model", None) else EXCHANGE_ENERGY_VERSION


def is_aux_kernel_record(record) -> bool:
    """True when a kernel_identity / method_settings record describes an auxiliary-CV run.

    One predicate for the eligibility classifier, the Stage B resume refusal and the MBAR loaders'
    guard: the v3_aux exchange version, or any recorded auxiliary model digest.
    """
    if not isinstance(record, dict):
        return False
    return bool(record.get("exchange_energy_version") == EXCHANGE_ENERGY_VERSION_AUX
                or record.get("aux_model_sha256") or record.get("aux_cv_model_sha256"))


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
        "exchange_energy_version": exchange_energy_version_for_args(args),
        "pair_model_sha256": meta.get("pair_model_sha256") if residual else None,
        "production_ensemble": str(getattr(args, "production_ensemble", "") or ""),
        "gamd_boost_type": str(getattr(args, "gamd_boost_type", "") or ""),
    }
    if getattr(args, "aux_cv_model", None):
        runtime = getattr(args, "_aux_runtime", None)
        if runtime is not None:
            identity["aux_model_sha256"] = str(runtime.info.model_sha256)
            if getattr(runtime, "topology_sha256", None):
                identity["aux_topology_sha256"] = str(runtime.topology_sha256)
        else:
            from .auxiliary_cv.model import AuxModel   # lazy: keep this module import-light
            identity["aux_model_sha256"] = AuxModel.load(args.aux_cv_model).model_sha256
    body = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    identity["digest"] = hashlib.sha256(body).hexdigest()
    return identity


def classify_segment_kernel(window_snapshot: dict, *, sample_payload_schema: "dict | None" = None) -> tuple:
    """(eligibility, reason) of one segment from its ``windows/<segment>.json`` payload.

    Non-residual segments are not affected by the residual fast-path defect and stay
    eligible under their own documented rules; a residual segment is verified only when its
    recorded evaluator and exchange versions equal the current ones. An auxiliary-CV segment is
    verified only when its samples manifest records the atlas-aux-samples-v1 payload and its frozen
    snapshot and that payload name the kernel's model (``sample_payload_schema``, Stage C).
    """
    snap = dict(window_snapshot or {})
    cv2 = str(snap.get("cv2_type") or "none")
    identity = snap.get("kernel_identity") or {}
    if is_aux_kernel_record(identity):
        sha = identity.get("aux_model_sha256")
        payload = dict(sample_payload_schema or {})
        state = snap.get("state_definition") or {}
        if payload.get("schema") != AUX_SAMPLES_PAYLOAD_SCHEMA:
            return ELIGIBLE_AUX_UNPERSISTED, ("auxiliary-CV segment whose samples carry no atlas-aux-samples-v1 "
                                              "payload (Stage B engineering run): its bias cannot be reconstructed")
        if not sha or sha not in (state.get("aux_models") or {}) or sha not in (payload.get("model_shas") or []):
            return ELIGIBLE_AUX_UNPERSISTED, ("auxiliary model of the kernel identity is not bound by the frozen "
                                              "snapshot and the sample schema")
        if cv2 != RESIDUAL_MODE:
            return ELIGIBLE_VERIFIED, "auxiliary features recorded (atlas-aux-samples-v1) with a frozen state table"
    elif cv2 != RESIDUAL_MODE:
        return ELIGIBLE_NOT_APPLICABLE, "secondary CV is not residual-torsion-pc"
    if not identity:
        return ELIGIBLE_UNKNOWN, "residual segment without a kernel_identity record (written before F01)"
    ev = identity.get("cv_evaluator_version")
    ex = identity.get("exchange_energy_version")
    if ev == RESIDUAL_EVALUATOR_VERSION and ex in (EXCHANGE_ENERGY_VERSION, EXCHANGE_ENERGY_VERSION_AUX):
        return ELIGIBLE_VERIFIED, "kernel matches the current evaluator and exchange assembly"
    if ev in (None, "affected_pre_f01"):
        return ELIGIBLE_AFFECTED, "recorded coordinates came from the two-term fall-through (review I01)"
    return ELIGIBLE_UNKNOWN, f"kernel {ev!r}/{ex!r} is not the current {RESIDUAL_EVALUATOR_VERSION!r}/{EXCHANGE_ENERGY_VERSION!r}"
