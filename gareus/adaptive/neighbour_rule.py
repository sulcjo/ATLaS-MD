"""One per-pair restraint-width neighbour rule (spec P7a, Section 3.0).

Which umbrella states are spatial neighbours of which: the rule 3.1's edge
metric measures (every same-rung pair within the radius), and the rule P7b will
switch the exchange graph (`windows.build_explicit_2d_neighbor_edges`), the
top-up partners (`layout_neighbours.spatial_neighbour_pairs`) and
`ladder_overlap` to. Nothing calls it yet; this module changes no behaviour.

Distance
--------
On each CV axis a harmonic window of force constant k (kcal/mol/CV^2) samples,
to first order, a Gaussian of width sigma_w = sqrt(RT/k) around its centre
(`math_helpers.restraint_sigma`, RT at the run temperature). For a pair (a, b)
the axis is normalised by the PAIR'S OWN scale

    s_ab = sqrt(sigma_a^2 + sigma_b^2),

the width of the difference x_a - x_b of two independent draws, one from each
window. |c_a - c_b| / s_ab is then the separation of the two distributions in
units of their combined spread, which is what their overlap mainly depends on.
For equal widths the overlap is a function of that ratio alone; for unequal
widths there is an extra width-ratio factor (1D Bhattacharyya coefficient
BC = sqrt(2 sigma_a sigma_b / (sigma_a^2 + sigma_b^2)) exp(-d^2 / 4)) that can
only LOWER the overlap at a given d, so the equal-width numbers below are an
upper bound and a radius set from them errs toward inclusion (chignolin_9's
most unequal CV1 pair, k 186 vs 520, has a prefactor ~0.94). A global median
width (today's `layout_neighbours`) instead makes a stiff window (small
sigma_w) look NEARER than it is and a soft one FARTHER, and the arithmetic mean
of the widths, or the larger one, is not the width of the difference. The
per-axis ratios combine in quadrature:

    d_ab = sqrt(sum over used axes of ((c_a - c_b) / s_ab)^2).

Axis cases:

  * restrained on both states: as above. A restrained axis whose k is not
    recorded (None) has no known width, so the pair is unmeasurable (inf)
    unless the caller passes ``fallback_sigma`` (P7b: ``fallback_axis_sigmas``
    = the run's default k, else the median recorded width, else the median
    centre spacing; the caller records which);
  * restrained on neither: the axis is left out. An unrestrained axis carries a
    placeholder centre, and placeholders are never identity (spec P6);
  * restrained on exactly one state (only reachable with
    ``same_pattern_only=False``, i.e. 3.1's spanning edges from axis states to
    their nearest restrained neighbours): the free state's width on that axis
    is the POOLED SAMPLED sd of the axis, and its position is its own SAMPLED
    MEAN on that axis; both are passed in. Without either the axis cannot be
    measured and the distance is inf (never the placeholder centre).

Restraint pattern follows `adaptive_production._axis_restrained` /
`_restraint_pattern` exactly (copied, not imported, so this module does not
pull in the driver; `tests/test_neighbour_rule.py` pins the two to agree): k
None (not recorded) counts as restrained, k <= 0 / NaN / non-numeric does not,
and a missing CV2 centre means CV2 is unrestrained. Note that
`layout_neighbours._restrained_mask` differs on a NEGATIVE k (it counts as
restrained there); P7b's switch resolves that in favour of P6.

Neighbour criterion
-------------------
(a, b) are neighbours iff: they are distinct, on the same rung (lambda equal
to 1e-6; an unknown rung pairs only with an unknown rung), at least one axis is
restrained on each, their restraint patterns are identical (unless
``same_pattern_only=False``), and d_ab <= ``radius``.

Radius. For two equal-width Gaussians at normalised distance d, the pairwise
MBAR overlap int p_a p_b / (p_a + p_b) (the 0..0.5 scale of
`gareus.mbar_analysis.ladder.pairwise_state_overlap` and of the 0.15 / 0.25
thresholds) is 0.325 at d = 1, 0.206 at 1.5, 0.116 at 2, 0.058 at 2.5 and
0.026 at 3 (the overlap COEFFICIENT 2 Phi(-d / sqrt 2) is a different scale:
0.48 / 0.29 / 0.16 / 0.077 / 0.034). The 0.15 floor sits near d = 1.8.
`DEFAULT_RADIUS` = 2.5 measures every pair out to a Gaussian-model overlap of
~0.06, i.e. comfortably past the floor, because sampled widths exceed sigma_w
wherever the free energy is flat and the Gaussian model then under-predicts
overlap; on chignolin_9's lambda = 0 layout it gives 235 edges, matching spec
3.1's "K ~ 250". The radius is a parameter.

Scale invariance: rescaling a CV by s (centres x s, k x 1/s^2, pooled sd and
sampled means x s) leaves every d_ab unchanged; so does T x c with k x c.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

from gareus.math_helpers import restraint_sigma

DEFAULT_RADIUS = 2.5
RUNG_DECIMALS = 6            # same rung grouping as layout_neighbours / the 2D window map

PooledSd = Tuple[Optional[float], Optional[float]]


def axis_restrained(k: Optional[float]) -> bool:
    """k None = not recorded, which every writer means as "the run's default restraint".

    Same semantics as `adaptive_production._axis_restrained`.
    """
    if k is None:
        return True
    try:
        k = float(k)
    except (TypeError, ValueError):
        return True
    return bool(math.isfinite(k) and k > 0.0)


def restraint_pattern(primary_k: Optional[float], secondary_k: Optional[float],
                      secondary_center: Optional[float]) -> Tuple[bool, bool]:
    """(CV1 restrained, CV2 restrained). No CV2 centre means no CV2 restraint.

    Same semantics as `adaptive_production._restraint_pattern`.
    """
    return (axis_restrained(primary_k), secondary_center is not None and axis_restrained(secondary_k))


def _finite(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


@dataclass(frozen=True)
class NeighbourPoint:
    """One umbrella state as the neighbour rule sees it.

    ``sampled_mean`` = (CV1, CV2) sampled means, read only for an axis this
    state does not restrain when it is paired with a state that does.
    """

    primary_center: float
    primary_k: Optional[float]
    secondary_center: Optional[float] = None
    secondary_k: Optional[float] = None
    rung: Optional[float] = None
    sampled_mean: Tuple[Optional[float], Optional[float]] = (None, None)

    @classmethod
    def from_state(cls, state: Any, sampled_mean: Tuple[Optional[float], Optional[float]] = (None, None)
                   ) -> "NeighbourPoint":
        """From a `WindowState` (or anything with its attribute names)."""
        return cls(
            primary_center=float(state.primary_center),
            primary_k=getattr(state, "primary_k", None),
            secondary_center=getattr(state, "secondary_center", None),
            secondary_k=getattr(state, "secondary_k", None),
            rung=getattr(state, "gamd_lambda", None),
            sampled_mean=sampled_mean,
        )

    @property
    def pattern(self) -> Tuple[bool, bool]:
        return restraint_pattern(self.primary_k, self.secondary_k, self.secondary_center)

    def centre(self, axis: int) -> Optional[float]:
        return _finite(self.primary_center if axis == 0 else self.secondary_center)

    def k(self, axis: int) -> Optional[float]:
        return self.primary_k if axis == 0 else self.secondary_k

    def rung_key(self) -> Optional[float]:
        lam = _finite(self.rung)
        return None if lam is None else round(lam, RUNG_DECIMALS)


def _restrained_width(k: Optional[float], temperature_k: float) -> Optional[float]:
    """sigma_w of a restrained axis; None when k is not recorded (no width known)."""
    value = _finite(k)
    if value is None or value <= 0.0:
        return None
    sigma = restraint_sigma(value, temperature_k)
    return sigma if math.isfinite(sigma) and sigma > 0.0 else None


def _width_or_fallback(k: Optional[float], temperature_k: float, fallback: Optional[float]) -> Optional[float]:
    """sigma_w of a restrained axis; ``fallback`` (a finite positive width) when k is not recorded."""
    width = _restrained_width(k, temperature_k)
    if width is not None:
        return width
    fb = _finite(fallback)
    return fb if fb is not None and fb > 0.0 else None


def axis_separation(a: NeighbourPoint, b: NeighbourPoint, axis: int, temperature_k: float,
                    pooled_sd: Optional[float] = None, fallback_sigma: Optional[float] = None) -> Optional[float]:
    """|delta| / s_ab on one axis; None if the axis is left out; inf if unmeasurable.

    ``fallback_sigma`` is the width used for a restrained axis whose k is not recorded
    (``fallback_axis_sigmas``); without it such an axis is unmeasurable (inf), as in P7a.
    """
    on_a, on_b = a.pattern[axis], b.pattern[axis]
    if not (on_a or on_b):
        return None
    if on_a and on_b:
        ca, cb = a.centre(axis), b.centre(axis)
        sa = _width_or_fallback(a.k(axis), temperature_k, fallback_sigma)
        sb = _width_or_fallback(b.k(axis), temperature_k, fallback_sigma)
        if ca is None or cb is None or sa is None or sb is None:
            return math.inf
        return abs(ca - cb) / math.sqrt(sa * sa + sb * sb)
    fixed, free = (a, b) if on_a else (b, a)
    c_fixed, s_fixed = fixed.centre(axis), _width_or_fallback(fixed.k(axis), temperature_k, fallback_sigma)
    m_free, sd_free = _finite(free.sampled_mean[axis]), _finite(pooled_sd)
    if c_fixed is None or s_fixed is None or m_free is None or sd_free is None or sd_free <= 0.0:
        return math.inf
    return abs(c_fixed - m_free) / math.sqrt(s_fixed * s_fixed + sd_free * sd_free)


def pair_distance(a: NeighbourPoint, b: NeighbourPoint, temperature_k: float,
                  pooled_sd: PooledSd = (None, None), fallback_sigma: PooledSd = (None, None)) -> float:
    """Normalised distance d_ab (symmetric). inf when no axis is usable or one is unmeasurable."""
    total, used = 0.0, 0
    for axis in (0, 1):
        sep = axis_separation(a, b, axis, temperature_k, pooled_sd[axis], fallback_sigma[axis])
        if sep is None:
            continue
        if not math.isfinite(sep):
            return math.inf
        total += sep * sep
        used += 1
    return math.sqrt(total) if used else math.inf


def is_neighbour(a: NeighbourPoint, b: NeighbourPoint, temperature_k: float,
                 radius: float = DEFAULT_RADIUS, pooled_sd: PooledSd = (None, None),
                 same_pattern_only: bool = True, fallback_sigma: PooledSd = (None, None)) -> bool:
    """The P7a criterion (module docstring). ``a is b`` is never a neighbour of itself."""
    if a is b or a.rung_key() != b.rung_key():
        return False
    pa, pb = a.pattern, b.pattern
    if not any(pa) or not any(pb):
        return False
    if same_pattern_only and pa != pb:
        return False
    return pair_distance(a, b, temperature_k, pooled_sd, fallback_sigma) <= float(radius)


def neighbour_pairs(points: Sequence[NeighbourPoint], temperature_k: float,
                    radius: float = DEFAULT_RADIUS, pooled_sd: PooledSd = (None, None),
                    same_pattern_only: bool = True, fallback_sigma: PooledSd = (None, None)
                    ) -> list[Tuple[int, int]]:
    """Sorted index pairs (i, j), i < j, that are neighbours."""
    out = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if is_neighbour(points[i], points[j], temperature_k, radius, pooled_sd, same_pattern_only,
                            fallback_sigma):
                out.append((i, j))
    return out


def fallback_axis_sigmas(points: Sequence[NeighbourPoint], temperature_k: float,
                         default_k: Tuple[Optional[float], Optional[float]] = (None, None)
                         ) -> Tuple[Tuple[Optional[float], Optional[float]], Tuple[str, str]]:
    """Per axis, the width a restrained-but-k-unrecorded state is given, and where it came from.

    Resolves P7a's open point for old registries (k None = restrained, width unknown). In
    order: the run's default k (``default_k``, e.g. ``--k`` / ``--secondary-cv-k``) ->
    ``"run_default_k"``; else the median sigma_w over the states whose k IS recorded ->
    ``"median_restraint_width"``; else the median gap between distinct restrained centres on
    that axis (the same last resort as ``layout_neighbours._axis_scale``) ->
    ``"median_centre_spacing"``. An axis no state restrains -> (None, ``"unrestrained"``); a
    restrained axis where none of these exists -> (None, ``"unmeasurable"``). Callers record
    the source next to the graph they build.
    """
    sig: list = []
    src: list = []
    for axis in (0, 1):
        on = [p for p in points if p.pattern[axis]]
        if not on:
            sig.append(None); src.append("unrestrained")
            continue
        width = _restrained_width(default_k[axis], temperature_k)
        if width is not None:
            sig.append(width); src.append("run_default_k")
            continue
        known = [w for w in (_restrained_width(p.k(axis), temperature_k) for p in on) if w is not None]
        if known:
            sig.append(float(statistics.median(known))); src.append("median_restraint_width")
            continue
        centres = sorted({round(c, RUNG_DECIMALS) for c in (p.centre(axis) for p in on) if c is not None})
        gaps = [b - a for a, b in zip(centres[:-1], centres[1:]) if b - a > 0.0]
        if gaps:
            sig.append(float(statistics.median(gaps))); src.append("median_centre_spacing")
        else:
            sig.append(None); src.append("unmeasurable")
    return (sig[0], sig[1]), (src[0], src[1])


def neighbour_lists(n: int, pairs: Iterable[Tuple[int, int]]) -> dict[int, list[int]]:
    """index -> sorted neighbour indices, every index 0..n-1 present (empty = isolated)."""
    out: dict[int, set] = {i: set() for i in range(n)}
    for i, j in pairs:
        out[int(i)].add(int(j))
        out[int(j)].add(int(i))
    return {i: sorted(v) for i, v in out.items()}


def components(n: int, pairs: Iterable[Tuple[int, int]]) -> list[list[int]]:
    """Connected components over 0..n-1, each sorted, ordered by smallest member."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


__all__ = ["DEFAULT_RADIUS", "NeighbourPoint", "axis_restrained", "axis_separation", "components",
           "fallback_axis_sigmas",
           "is_neighbour", "neighbour_lists", "neighbour_pairs", "pair_distance", "restraint_pattern"]
