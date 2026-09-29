"""Frozen CV2 pair model for the adaptive driver (spec P5) and the adaptive coupling gate (3.4).

The driver never rebuilds a force; it needs the frozen pair only to ask how much a proposed
k2 stiffens CV1 (:mod:`gareus.cv_selection.coupling`). :func:`load_driver_pair` resolves the
three artifact paths from the campaign arguments (set by the config or the swarm sidecar,
which every job re-applies), falling back to the campaign manifest's recorded paths on a
resume, loads them through :meth:`PairModelRuntime.load` (digest- and semantics-checked,
real bindings required) and checks the pair digest against the manifest's
``cv_pair_model_sha256`` when one is recorded. Only a ``residual-torsion-pc`` campaign with a
bound, verified model gets a pair; everything else -- another CV2 type, a missing path, an
unreadable or mismatched artifact -- is ``NA`` with a reason, and NA blocks nothing. This
module never raises into the epoch loop: a completed MD epoch is never lost to the gate.

:class:`AdaptiveCouplingGate` is what the weak-edge proposer consults for every adaptive k2.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..cv_selection.coupling import (MAX_COUPLING_FRACTION, STATUS_NA, STATUS_OK, CouplingWindow,
                                     largest_passing_k2)

RESIDUAL_MODE = "residual-torsion-pc"
PATH_ATTRS = ("secondary_cv_model", "secondary_cv_candidate_set", "secondary_cv_feature_schema")
GATE_REPORT_NAME = "cv2_coupling_gate.json"


@dataclass(frozen=True)
class DriverPair:
    """The frozen pair as the driver sees it: a fit and component, or NA with a reason."""

    status: str                       # "bound" or "NA"
    reason: str = ""
    fit: Any = None
    j: Optional[int] = None
    pair_sha256: Optional[str] = None
    paths: Tuple[Optional[str], ...] = ()
    paths_source: str = ""

    @property
    def bound(self) -> bool:
        return self.status == "bound"

    def as_record(self) -> Dict[str, Any]:
        return {"status": self.status, "reason": self.reason, "component_index": self.j,
                "pair_model_sha256": self.pair_sha256, "paths": list(self.paths),
                "paths_source": self.paths_source}


def _read_settings(path: Path) -> Dict[str, Any]:
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return dict(doc.get("method_settings") or {}) if isinstance(doc, dict) else {}


def _manifest_settings(out_dir) -> Dict[str, Any]:
    """The first recorded ``method_settings`` naming the pair: the campaign root, then the phases.

    A swarm campaign's root manifest is written from the job's args before the epoch-0 sidecar
    is applied, so it can lack the pair; every phase manifest records it (``provenance``) with
    its ``cv_pair_model_sha256``. Phases are read oldest first (epoch_000 is the pair's first use).
    """
    if out_dir is None:
        return {}
    root = Path(out_dir)
    candidates = [root / "run_manifest.json"]
    ap_dir = root / "adaptive_production"
    candidates += sorted(ap_dir.glob("epoch_*/run_manifest.json")) + sorted(ap_dir.glob("epoch_*/*/run_manifest.json"))
    first: Dict[str, Any] = {}
    for path in candidates:
        settings = _read_settings(path)
        if not first:
            first = settings
        if all(settings.get(a) for a in PATH_ATTRS):
            return settings
    return first


def load_driver_pair(args, out_dir=None) -> DriverPair:
    """Resolve and verify the campaign's frozen pair; NA (never an exception) otherwise."""
    mode = str(getattr(args, "secondary_cv", "") or "")
    if mode != RESIDUAL_MODE:
        return DriverPair(STATUS_NA, f"cv2 is {mode or 'none'!r}, not {RESIDUAL_MODE}")
    settings = _manifest_settings(out_dir)
    paths = tuple(str(getattr(args, a, "") or "") or None for a in PATH_ATTRS)
    source = "args"
    if not all(paths):
        recorded = tuple(str(settings.get(a) or "") or None for a in PATH_ATTRS)
        if all(recorded):
            paths, source = recorded, "run_manifest"
    if not all(paths):
        missing = [a for a, p in zip(PATH_ATTRS, paths) if not p]
        return DriverPair(STATUS_NA, f"no bound pair model ({', '.join(missing)} unset)", paths=paths)
    try:
        from ..cv_selection.models import PairModelRuntime

        policy = str(getattr(args, "legacy_model_policy", "refuse") or "refuse")
        runtime = PairModelRuntime.load(*paths, require_deployable=True, allow_legacy_v1=(policy == "allow-v1"))
    except Exception as exc:          # NA, loudly; the production loader refuses bad artifacts itself
        return DriverPair(STATUS_NA, f"pair model not verified: {type(exc).__name__}: {exc}",
                          paths=paths, paths_source=source)
    recorded_sha = settings.get("cv_pair_model_sha256")
    if recorded_sha and str(recorded_sha) != runtime.pair_sha256:
        return DriverPair(STATUS_NA, f"pair model digest {runtime.pair_sha256[:12]} differs from the campaign "
                          f"manifest's {str(recorded_sha)[:12]}", paths=paths, paths_source=source,
                          pair_sha256=runtime.pair_sha256)
    return DriverPair("bound", "", runtime.fit, int(runtime.j), runtime.pair_sha256, paths, source)


@dataclass
class AdaptiveCouplingGate:
    """Spec 3.4 on adaptive k2: lower above ``max_fraction``, refuse below ``k_min``.

    ``gate`` returns ``(k2, note)`` with ``k2 = None`` meaning "do not create this state";
    ``note`` is a reason fragment (or None) for the state's reason string. Every decision is
    kept in ``records`` for ``cv2_coupling_gate.json``.
    """

    pair: DriverPair
    max_fraction: float = MAX_COUPLING_FRACTION
    k_min: float = 0.0
    temperature_k: float = 300.0
    records: List[Dict[str, Any]] = field(default_factory=list)

    def gate(self, primary_center, primary_k, secondary_center, secondary_k, *,
             context: str) -> Tuple[Optional[float], Optional[str]]:
        if secondary_k is None:
            return None, None                                    # CV1-only state: nothing to gate
        k2 = float(secondary_k)
        rec: Dict[str, Any] = {"context": context, "k2_proposed": k2}
        if not self.pair.bound:
            self.records.append({**rec, "status": STATUS_NA, "reason": self.pair.reason, "k2": k2})
            return k2, None
        if not (math.isfinite(k2) and k2 > 0.0):
            self.records.append({**rec, "status": "no_cv2_restraint", "k2": k2})
            return k2, None
        window = CouplingWindow(primary_center=float(primary_center), primary_k=float(primary_k or 0.0),
                                secondary_center=float(secondary_center or 0.0),
                                temperature_k=float(self.temperature_k))
        try:
            out = largest_passing_k2(self.pair.fit, self.pair.j, k2, window, max_fraction=self.max_fraction)
        except Exception as exc:        # never lose an epoch to the gate
            self.records.append({**rec, "status": STATUS_NA, "reason": f"{type(exc).__name__}: {exc}", "k2": k2})
            return k2, None
        before = out["before"]
        if before["status"] != STATUS_OK:
            self.records.append({**rec, "status": STATUS_NA, "reason": before.get("reason"), "k2": k2})
            return k2, None
        rec.update({"fraction_before": before["fraction"], "fraction_after": out["after"]["fraction"],
                    "anchor_clamped": before.get("anchor_clamped"), "mode": before.get("mode")})
        if not out["lowered"]:
            self.records.append({**rec, "status": "pass", "k2": k2})
            return k2, None
        k_new = float(out["k2"])
        if k_new < float(self.k_min or 0.0):
            self.records.append({**rec, "status": "refused_below_k_min", "k2": k_new, "k_min": float(self.k_min)})
            return None, (f"coupling gate: largest passing k2 {k_new:.4g} < cv2_k_min {float(self.k_min):.4g} "
                          f"(fraction {before['fraction']:.3f} at k2={k2:.4g}); state not created")
        self.records.append({**rec, "status": "lowered", "k2": k_new})
        return k_new, (f"secondary_k lowered {k2:.4g} -> {k_new:.4g} by the CV2 coupling gate "
                       f"(CV1 curvature fraction {before['fraction']:.3f} > {self.max_fraction})")

    def report(self) -> Dict[str, Any]:
        # ``counts`` are the proposers' decisions; the applier (spec P2) re-gates every child it
        # validates, with an ``apply: `` context, and those are counted apart so a proposed and
        # then applied state is not counted twice.
        counts: Dict[str, int] = {}
        apply_counts: Dict[str, int] = {}
        for r in self.records:
            target = apply_counts if str(r.get("context", "")).startswith("apply: ") else counts
            target[r["status"]] = target.get(r["status"], 0) + 1
        return {"schema_version": "cv2_coupling_gate_v1", "pair": self.pair.as_record(),
                "max_coupling_fraction": float(self.max_fraction), "cv2_k_min": float(self.k_min or 0.0),
                "temperature_k": float(self.temperature_k), "counts": counts, "apply_counts": apply_counts,
                "decisions": list(self.records)}


def gate_from_args(args, out_dir, policy, *, temperature_k: float) -> Optional[AdaptiveCouplingGate]:
    """The gate for this epoch, or None when ``policy.cv2_coupling_gate`` is off."""
    if not bool(getattr(policy, "cv2_coupling_gate", False)):
        return None
    pair = load_driver_pair(args, out_dir)
    if not pair.bound:
        print(f"    CV2 coupling gate: NA ({pair.reason}); adaptive k2 is not gated")
    try:
        k_min = float(getattr(args, "cv2_k_min", 0.0) or 0.0)
    except (TypeError, ValueError):
        k_min = 0.0
    return AdaptiveCouplingGate(pair, max_fraction=float(getattr(policy, "max_coupling_fraction",
                                                                   MAX_COUPLING_FRACTION)),
                                k_min=k_min, temperature_k=float(temperature_k))
