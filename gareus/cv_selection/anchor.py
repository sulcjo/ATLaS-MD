from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from . import contracts as C
from ..swarm.ladder_design import n_resolvable_windows

DEPLOYABLE_ANCHOR_KINDS = frozenset({"nonlocal-contact-fraction"})


@dataclass(frozen=True)
class AnchorCandidate:
    kind: str
    definition: dict
    values: np.ndarray


@dataclass(frozen=True)
class AnchorScore:
    kind: str
    definition: dict
    n_resolvable: int
    coverage_range: float
    dynamic_range: float
    forceable: bool
    deployable: bool
    reasons: tuple


def score_anchor(cand, *, temperature_k, k_max_kcal, min_windows, min_dynamic_range=0.0,
                 overlap_sigma=1.5):
    """Deployability of one anchor candidate on the swarm's own frames.

    ``n_resolvable`` is the shipped, unit-dependent count the ladder actually
    uses; for the configured dimensionless contact fraction it is the right
    gate. ``dynamic_range = range / sigma`` is unit-invariant and is REPORTED,
    not gated by default: it is 3.46 for any uniform distribution, so a gate
    at 6 would reject the best-covered anchor imaginable. Kinds without a
    landed runtime force are scored but can never be deployable.
    """
    if cand.kind not in C.NATIVE_BLIND_CV_KINDS:
        raise ValueError(f"{cand.kind!r} is not in the native-blind dictionary")
    v = np.asarray(cand.values, dtype=np.float64); v = v[np.isfinite(v)]
    reasons = []
    forceable = cand.kind in DEPLOYABLE_ANCHOR_KINDS
    if not forceable:
        reasons.append(f"{cand.kind}: no runtime force for this anchor kind yet")
    if v.size < 2:
        return AnchorScore(cand.kind, dict(cand.definition), 0, 0.0, 0.0, forceable, False,
                           tuple(reasons + ["fewer than two finite values"]))
    rng_ = float(v.max() - v.min()); sd = float(v.std())
    dyn = rng_ / sd if sd > 0 else 0.0
    n_res = n_resolvable_windows(rng_, temperature_k, k_max_kcal=k_max_kcal, overlap_sigma=overlap_sigma)
    if n_res < min_windows:
        reasons.append(f"{n_res} resolvable windows < required {min_windows} at k_max={k_max_kcal}")
    if dyn < min_dynamic_range:
        reasons.append(f"dynamic range {dyn:.2f} sigma < {min_dynamic_range}")
    return AnchorScore(cand.kind, dict(cand.definition), n_res, rng_, dyn, forceable,
                       forceable and not reasons, tuple(reasons))


def rank_anchors(candidates, **kw):
    scored = [(-(s.dynamic_range), i, s) for i, s in
              enumerate(score_anchor(c, **kw) for c in candidates)]
    return [s for _, _, s in sorted(scored, key=lambda t: t[:2])]
