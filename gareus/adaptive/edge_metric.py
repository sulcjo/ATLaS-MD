"""One edge metric: two-state MBAR overlap for spatial edges (spec 3.1).

Behind ``--ap-edge-metric`` (``AdaptiveDecisionPolicy.edge_metric``). With the
default ``"marginal"`` nothing here runs and every diagnostics key is what it
was; with ``"pairwise-mbar"`` every collector (flat epoch, segmented epoch,
final combined) grades its spatial edges on the symmetrised pairwise MBAR
overlap ``sqrt(O_ab O_ba)`` -- the 0..0.5 scale of
``gareus.mbar_analysis.ladder.pairwise_state_overlap`` and of the rung
thresholds -- and the weak-edge predicate reads that number.

Estimator (pre-union)
---------------------
For an edge (a, b) on ONE rung, the Pep-GaMD boost is the same function of
configuration in both states (same lambda, same frozen envelope) and cancels,
so the reduced-energy difference of a sample x is the two umbrella terms alone:

    u_k(x) = beta * sum over the axes state k restrains of 0.5 * k_k,ax * (cv_ax(x) - c_k,ax)^2

with the P4 restraint record (kcal/mol per CV^2, ``reconstruct_bias_matrix``'s
form) and beta = 1 / (R T) in mol/kcal. An axis a state does not restrain adds
nothing to its u (never 0 * NaN, never a placeholder centre); a restrained axis
whose k is not recorded (P6: None = restrained, width unknown) makes the edge
unmeasurable. A sample without CV2 is dropped FROM THIS EDGE only when one of
the two states restrains CV2 (a missing CV2 is never an on-target sample).

Both states' samples are pooled and the two-state MBAR (= BAR) equation

    sum_n 1 / (N_b + N_a exp(Delta_n - df)) = 1,    Delta_n = u_b(x_n) - u_a(x_n)

is solved for df = f_b - f_a in float64 with log-sum-exp (monotone in df,
bracketed, ``brentq``). The overlap is then ``pairwise_state_overlap`` itself
with f = (0, df): one overlap formula for the pre-union and post-union values.
Samples are the P4 bounded constant-stride subsample (the sidecar
``<json stem>_paired_cv.npz``), so the live collector and an offline replay of
the NPZ give identical numbers; N_a, N_b are the subsample sizes (recorded).

Uncertainty and sufficiency
---------------------------
Each state's own work series w = u_other - u_self (the quantity BAR averages)
is blocked Flyvbjerg-Petersen style WITHIN each source (``source_index``): at
block size b = 1, 2, 4, ... the statistical inefficiency
g(b) = b var(block means) / var(w), with error g sqrt(2 / (n_blocks - 1)); the
first level whose successor is not higher by more than its error is the
plateau (else the last level with >= ``MIN_BLOCKS`` blocks, flagged
``plateau: False`` -- a lower bound). tau = (g - 1) / 2 in subsample frames,
N_eff = n / g. An edge is graded only if both states have
N_eff >= ``min_edge_neff`` (200); below it it is "unmeasured" and never weak.
The decision uses a moving-block bootstrap (block length >= 2 tau, i.e.
ceil(g), blocks never straddle a source), df re-solved per replicate, seeded
from the edge's state ids: an edge is weak only if the UPPER 90 % quantile is
below the threshold (``min_rung_overlap``), i.e. confidently below it. The
point estimate and both quantiles are recorded. (Spec Section 10 item 6 writes
"lower 90 % bound"; 3.1 writes "upper 90 % bound ... weak only if confidently
below threshold". Only the upper bound means "confidently weak"; that is what
is implemented.) The bootstrap does not account for adaptive design choices;
its q10-q90 width matched the spread over repeated independent datasets for iid
samples (0.011 vs 0.012 at N = 500 per state), not checked for correlated ones.
At N_eff ~ 500 that width is ~0.01, so the "confidently below" guard moves few
decisions: the (uncalibrated, spec T2) 0.15 threshold does most of the work.

Graph
-----
Nodes are the states on the representative rung (lambda = 0 if present, else
the lowest; the same rule as ``build_geometry_edges``). Edges:

* ``neighbour``: every pair within the P7a restraint-width radius
  (``neighbour_rule.neighbour_pairs``, same restraint pattern only);
* ``spanning``: every axis state (k1 = 0 or k2 = 0) and every anchor (no axis
  restrained) gets an edge to its ``SPANNING_NEIGHBOURS`` nearest states of a
  DIFFERENT pattern that restrain at least one axis, by the P7a distance with
  the free axis at the state's own sampled mean and the pooled sampled sd
  (root-mean of the sampled variances of the states that do not restrain the
  axis). Without these the same-pattern graph is 4 components per rung on
  chignolin_9 (anchor, CV1-only, CV2-only, 2D) by construction.

The collector's own geometry edges (``primary_chain``, ``nearest_2d``) are
graded too; graph edges they do not already contain are appended.

Which edges can be WEAK (what the gate counts and the bridge proposer acts on):
only the collector's geometry edges between two states of the same restraint
pattern. A ``neighbour`` edge is graded but never weak: the P7a radius reaches
past the threshold on purpose (Gaussian-model overlap ~0.06 at 2.5), so a
second or third neighbour reads "below 0.15" without being a gap (chignolin_9
epoch_002: 86 such edges), and midpoint-bridging them would add states between
states that already have neighbours in between. A cross-pattern edge (every
``spanning`` edge, and any cross-pattern geometry edge) is never weak either: a
midpoint between two patterns lands on a placeholder coordinate. The whole
graph decides connectivity instead (``edge_metric.components``): its verdict
keeps every graded edge that is not confidently below threshold, so an
unmeasured edge never splits a component. Acting on a split is spec 3.3 R1's
job (not here): a warning is recorded. Rung edges are untouched
(they stay on the union ``mbar_overlap``; the boost does not cancel across
rungs and the P4 subsample carries no V_pep/V_dih).

Post-union, a spatial edge that the union solve scored (``mbar_overlap``,
written by ``_apply_union_edge_overlap`` with the union f_k and full n_k) is
judged on that point value instead (no bootstrap bound exists for it).

Everything here is diagnostics: any failure is caught and reported as
``edge_metric.status == "error"`` and every edge stays unmeasured (never weak);
it never raises into a collector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from gareus.adaptive.neighbour_rule import (
    DEFAULT_RADIUS,
    NeighbourPoint,
    components as graph_components,
    neighbour_pairs,
    pair_distance,
)
from gareus.units import K_B_KJ_PER_MOL_K, KJ_PER_KCAL

EDGE_METRICS = ("marginal", "pairwise-mbar")
DEFAULT_EDGE_METRIC = "marginal"
PAIRWISE_MBAR = "pairwise-mbar"
DEFAULT_MIN_EDGE_NEFF = 200.0
N_BOOTSTRAP = 200
BOOTSTRAP_QUANTILES = (0.10, 0.90)
MIN_BLOCKS = 16
SPANNING_NEIGHBOURS = 2
GRAPH_EDGE_TYPES = ("neighbour", "spanning")
METHOD = "two_state_bar_stride_subsample_v1"
_BOOT_SEED = 3101


def _finite(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def beta_mol_per_kcal(temperature_k: float) -> float:
    """1 / (R T) in mol/kcal."""
    return KJ_PER_KCAL / (K_B_KJ_PER_MOL_K * float(temperature_k))


def _axis_restrained(k: Any) -> bool:
    """Same rule as P6 / P4: None = restrained (default k), <= 0 / NaN = not."""
    if k is None:
        return True
    try:
        k = float(k)
    except (TypeError, ValueError):
        return True
    return bool(math.isfinite(k) and k > 0.0)


@dataclass(frozen=True)
class Restraint:
    """One state's umbrella, from the P4 ``paired_cv["restraint"]`` record."""

    primary_center: Optional[float]
    primary_k: Optional[float]
    secondary_center: Optional[float] = None
    secondary_k: Optional[float] = None
    gamd_lambda: Optional[float] = 0.0

    @classmethod
    def from_record(cls, rec: Mapping[str, Any]) -> "Restraint":
        return cls(primary_center=rec.get("primary_center"), primary_k=rec.get("primary_k"),
                   secondary_center=rec.get("secondary_center"), secondary_k=rec.get("secondary_k"),
                   gamd_lambda=rec.get("gamd_lambda"))

    @property
    def pattern(self) -> Tuple[bool, bool]:
        return (_axis_restrained(self.primary_k),
                self.secondary_center is not None and _axis_restrained(self.secondary_k))

    def known(self) -> bool:
        """Every restrained axis has a finite centre and a recorded k."""
        on1, on2 = self.pattern
        if on1 and (_finite(self.primary_k) is None or _finite(self.primary_center) is None):
            return False
        if on2 and (_finite(self.secondary_k) is None or _finite(self.secondary_center) is None):
            return False
        return True

    def reduced_bias(self, cv1: np.ndarray, cv2: np.ndarray, beta: float) -> np.ndarray:
        """beta * U on each sample; NaN where a restrained axis' CV is missing."""
        on1, on2 = self.pattern
        cv1 = np.asarray(cv1, dtype=np.float64)
        u = np.zeros(cv1.shape, dtype=np.float64)
        if on1:
            u = u + 0.5 * float(self.primary_k) * (cv1 - float(self.primary_center)) ** 2
        if on2:
            cv2 = np.asarray(cv2, dtype=np.float64)
            u = u + 0.5 * float(self.secondary_k) * (cv2 - float(self.secondary_center)) ** 2
        return float(beta) * u


# ---- two-state solve ---------------------------------------------------------------

def _bar_h(df: float, delta: np.ndarray, log_na: float, log_nb: float) -> float:
    """log sum_n 1 / (N_b + N_a exp(Delta_n - df)); zero at the two-state MBAR root."""
    x = -np.logaddexp(log_nb, log_na + delta - df)
    m = float(np.max(x))
    return m + math.log(float(np.sum(np.exp(x - m))))


def solve_two_state_df(delta: np.ndarray, n_a: int, n_b: int, guess: Optional[float] = None) -> float:
    """df = f_b - f_a (kT) from the pooled samples' Delta = u_b - u_a (two-state MBAR/BAR)."""
    from scipy.optimize import brentq  # noqa: PLC0415
    delta = np.asarray(delta, dtype=np.float64)
    log_na, log_nb = math.log(n_a), math.log(n_b)
    if guess is not None and math.isfinite(guess):
        # Newton on h (monotone, h' = sum t (1 - N_b t) / sum t); a bootstrap replicate
        # starts next to the point estimate and converges in a few steps.
        df = float(guess)
        for _ in range(20):
            x = -np.logaddexp(log_nb, log_na + delta - df)          # log t_n
            m = float(np.max(x))
            t = np.exp(x - m)
            st = float(np.sum(t))
            h = m + math.log(st)
            slope = float(np.sum(t * (1.0 - n_b * np.exp(x)))) / st
            if not (slope > 0.0 and math.isfinite(h)):
                break
            step = h / slope
            df -= step
            if abs(step) < 1.0e-10:
                return df
    pad = 40.0 + math.log(float(n_a + n_b))
    lo_all, hi_all = float(np.min(delta)) - pad, float(np.max(delta)) + pad
    if guess is not None and math.isfinite(guess):
        lo, hi = guess - 5.0, guess + 5.0
        for _ in range(8):
            if _bar_h(lo, delta, log_na, log_nb) < 0.0 < _bar_h(hi, delta, log_na, log_nb):
                break
            lo, hi = guess - 2.0 * (guess - lo), guess + 2.0 * (hi - guess)
        else:
            lo, hi = lo_all, hi_all
        lo, hi = max(lo, lo_all), min(hi, hi_all)
    else:
        lo, hi = lo_all, hi_all
    return float(brentq(_bar_h, lo, hi, args=(delta, log_na, log_nb), xtol=1.0e-10, rtol=1.0e-12))


def two_state_overlap(delta_a: np.ndarray, delta_b: np.ndarray,
                      guess: Optional[float] = None) -> Tuple[float, float]:
    """(sqrt(O_ab O_ba), df) from Delta = u_b - u_a on a's and on b's samples.

    The overlap is ``pairwise_state_overlap`` with f = (0, df); a row's two
    reduced energies are passed as (0, Delta), which leaves every weight
    unchanged (a common per-row shift cancels in W_nk).
    """
    from gareus.mbar_analysis.ladder import pairwise_state_overlap  # noqa: PLC0415
    delta_a = np.asarray(delta_a, dtype=np.float64)
    delta_b = np.asarray(delta_b, dtype=np.float64)
    n_a, n_b = int(delta_a.size), int(delta_b.size)
    delta = np.concatenate([delta_a, delta_b])
    df = solve_two_state_df(delta, n_a, n_b, guess)
    u = np.column_stack([np.zeros_like(delta), delta])
    window = np.concatenate([np.zeros(n_a, dtype=np.int64), np.ones(n_b, dtype=np.int64)])
    ov = pairwise_state_overlap(u, window, np.array([0.0, df]), np.array([n_a, n_b], dtype=float), 0, 1)
    return float(ov), df


# ---- blocking and bootstrap --------------------------------------------------------

def _segments(source_index: Optional[np.ndarray], n: int) -> List[Tuple[int, int]]:
    """Contiguous [start, stop) runs of one source (the P4 NPZ keeps sources contiguous)."""
    if n == 0:
        return []
    if source_index is None:
        return [(0, n)]
    src = np.asarray(source_index)
    cuts = np.flatnonzero(src[1:] != src[:-1]) + 1
    bounds = [0, *cuts.tolist(), n]
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def blocking_inefficiency(x: np.ndarray, source_index: Optional[np.ndarray] = None,
                          min_blocks: int = MIN_BLOCKS) -> Dict[str, Any]:
    """Flyvbjerg-Petersen blocking of ``x`` within each source.

    Returns ``{"g", "g_err", "tau", "tau_err", "n", "n_eff", "block_size", "plateau", "levels"}``;
    ``g`` is the statistical inefficiency (1 + 2 tau), floored at 1.
    """
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    out: Dict[str, Any] = {"g": 1.0, "g_err": 0.0, "tau": 0.0, "tau_err": 0.0, "n": n,
                           "n_eff": float(n), "block_size": 1, "plateau": True, "levels": 0}
    if n < 2:
        out["plateau"] = False
        return out
    mean = float(np.mean(x))
    var0 = float(np.mean((x - mean) ** 2))
    if not var0 > 0.0:
        return out
    segs = _segments(source_index, n)
    levels: List[Tuple[int, float, float]] = []
    b = 1
    while True:
        means = []
        for s0, s1 in segs:
            m = (s1 - s0) // b
            if m:
                means.append(x[s0:s0 + m * b].reshape(m, b).mean(axis=1))
        nb = int(sum(v.size for v in means))
        if nb < int(min_blocks):
            break
        bm = np.concatenate(means)
        g = b * float(np.var(bm, ddof=1)) / var0
        levels.append((b, g, g * math.sqrt(2.0 / (nb - 1))))
        b *= 2
    if not levels:
        out["plateau"] = False
        return out
    chosen, plateau = levels[-1], False
    for cur, nxt in zip(levels[:-1], levels[1:]):
        if nxt[1] - cur[1] <= nxt[2]:
            chosen, plateau = cur, True
            break
    g = max(1.0, chosen[1])
    g_err = chosen[2] if chosen[1] >= 1.0 else 0.0
    out.update(g=g, g_err=float(g_err), tau=0.5 * (g - 1.0), tau_err=0.5 * float(g_err),
               n_eff=n / g, block_size=int(chosen[0]), plateau=bool(plateau), levels=len(levels))
    return out


def _block_starts(segs: Sequence[Tuple[int, int]], length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Moving-block starts and lengths that never straddle a source."""
    starts, lengths = [], []
    for s0, s1 in segs:
        if s1 - s0 <= length:
            starts.append(np.array([s0])); lengths.append(np.array([s1 - s0]))
        else:
            st = np.arange(s0, s1 - length + 1)
            starts.append(st); lengths.append(np.full(st.size, length))
    return np.concatenate(starts), np.concatenate(lengths)


def _resample(n: int, segs: Sequence[Tuple[int, int]], length: int, rng: np.random.Generator) -> np.ndarray:
    """One moving-block bootstrap draw of ``n`` indices (blocks of ``length``, within sources)."""
    starts, lengths = _block_starts(segs, max(1, int(length)))
    n_blocks = int(math.ceil(n / float(max(1, int(lengths.min()))))) + 1
    k = rng.integers(starts.size, size=n_blocks)
    offsets = np.arange(int(lengths.max()))
    idx = starts[k][:, None] + offsets[None, :]
    return idx[offsets[None, :] < lengths[k][:, None]][:n]


# ---- one edge ----------------------------------------------------------------------

@dataclass
class StateSamples:
    state_id: int
    restraint: Restraint
    cv1: np.ndarray
    cv2: np.ndarray
    source_index: Optional[np.ndarray] = None


def _unmeasured(reason: str, **extra: Any) -> Dict[str, Any]:
    out = {"status": "unmeasured", "reason": reason, "overlap": None, "overlap_lower": None,
           "overlap_upper": None, "method": METHOD}
    out.update(extra)
    return out


def evaluate_edge(a: StateSamples, b: StateSamples, beta: float, *, min_neff: float = DEFAULT_MIN_EDGE_NEFF,
                  n_bootstrap: int = N_BOOTSTRAP, seed: int = _BOOT_SEED) -> Dict[str, Any]:
    """The two-state MBAR overlap of one edge, its blocking and bootstrap bounds."""
    pattern_pair = "same" if a.restraint.pattern == b.restraint.pattern else "cross"
    base = {"pattern_pair": pattern_pair}
    if not (a.restraint.known() and b.restraint.known()):
        return _unmeasured("restraint_k_unknown", **base)
    if not any(a.restraint.pattern) and not any(b.restraint.pattern):
        return _unmeasured("no_restrained_axis", **base)
    sides = []
    for own, other in ((a, b), (b, a)):
        u_self = own.restraint.reduced_bias(own.cv1, own.cv2, beta)
        u_other = other.restraint.reduced_bias(own.cv1, own.cv2, beta)
        keep = np.isfinite(u_self) & np.isfinite(u_other)
        src = None if own.source_index is None else np.asarray(own.source_index)[keep]
        sides.append((u_other[keep] - u_self[keep], src, int((~keep).sum())))
    (w_a, src_a, drop_a), (w_b, src_b, drop_b) = sides
    # Delta = u_b - u_a on every sample: +w on a's samples, -w on b's.
    delta_a, delta_b = w_a, -w_b
    base.update(n=[int(delta_a.size), int(delta_b.size)], n_dropped=[drop_a, drop_b])
    if delta_a.size < 2 or delta_b.size < 2:
        return _unmeasured("too_few_samples", **base)
    blk_a = blocking_inefficiency(w_a, src_a)
    blk_b = blocking_inefficiency(w_b, src_b)
    base.update(n_eff=[blk_a["n_eff"], blk_b["n_eff"]], tau=[blk_a["tau"], blk_b["tau"]],
                tau_err=[blk_a["tau_err"], blk_b["tau_err"]], g=[blk_a["g"], blk_b["g"]],
                tau_plateau=[blk_a["plateau"], blk_b["plateau"]])
    point, df = two_state_overlap(delta_a, delta_b)
    base.update(overlap_point=point, delta_f_kT=df)
    if min(blk_a["n_eff"], blk_b["n_eff"]) < float(min_neff):
        return _unmeasured("low_neff", **base)
    rng = np.random.default_rng([int(seed), int(a.state_id), int(b.state_id)])
    segs_a, segs_b = _segments(src_a, delta_a.size), _segments(src_b, delta_b.size)
    len_a, len_b = int(math.ceil(blk_a["g"])), int(math.ceil(blk_b["g"]))
    boot = []
    for _ in range(int(n_bootstrap)):
        ia = _resample(delta_a.size, segs_a, len_a, rng)
        ib = _resample(delta_b.size, segs_b, len_b, rng)
        boot.append(two_state_overlap(delta_a[ia], delta_b[ib], guess=df)[0])
    lo, hi = (float(q) for q in np.quantile(np.asarray(boot), BOOTSTRAP_QUANTILES))
    base.update(status="ok", reason=None, overlap=point, overlap_lower=lo, overlap_upper=hi,
                bootstrap={"n": int(n_bootstrap), "quantiles": list(BOOTSTRAP_QUANTILES),
                           "block_length": [len_a, len_b]}, method=METHOD)
    return base


# ---- graph -------------------------------------------------------------------------

def representative_rung(lambdas: Iterable[Optional[float]]) -> Optional[float]:
    vals = sorted({round(float(v or 0.0), 6) for v in lambdas})
    if not vals:
        return None
    return 0.0 if any(abs(v) <= 1.0e-9 for v in vals) else vals[0]


def pooled_free_axis_sd(nodes: Sequence[Tuple[Restraint, Tuple[Optional[float], Optional[float]]]]
                        ) -> Tuple[Optional[float], Optional[float]]:
    """Per axis: sqrt(mean sampled variance) over the states that do NOT restrain it."""
    out: List[Optional[float]] = []
    for axis in (0, 1):
        free = [v[axis] for r, v in nodes if not r.pattern[axis] and _finite(v[axis]) is not None]
        out.append(math.sqrt(float(np.mean(free))) if free else None)
    return out[0], out[1]


def build_edge_graph(points: Sequence[NeighbourPoint], temperature_k: float, *,
                     radius: float = DEFAULT_RADIUS, pooled_sd: Tuple[Optional[float], Optional[float]] = (None, None),
                     n_spanning: int = SPANNING_NEIGHBOURS) -> List[Tuple[int, int, str, float]]:
    """(i, j, kind, P7a distance), i < j, over ``points`` (all on one rung)."""
    edges: Dict[Tuple[int, int], Tuple[int, int, str, float]] = {}
    for i, j in neighbour_pairs(points, temperature_k, radius, pooled_sd, same_pattern_only=True):
        edges[(i, j)] = (i, j, "neighbour", pair_distance(points[i], points[j], temperature_k, pooled_sd))
    for i, p in enumerate(points):
        if p.pattern == (True, True):
            continue
        cands = []
        for j, q in enumerate(points):
            if j == i or q.pattern == p.pattern or not any(q.pattern):
                continue
            d = pair_distance(p, q, temperature_k, pooled_sd)
            if math.isfinite(d):
                cands.append((d, j))
        for d, j in sorted(cands)[:int(n_spanning)]:
            key = (min(i, j), max(i, j))
            edges.setdefault(key, (key[0], key[1], "spanning", float(d)))
    return [edges[k] for k in sorted(edges)]


# ---- the weak predicate ------------------------------------------------------------

def edge_below_threshold(edge: Mapping[str, Any], threshold: float) -> bool:
    """Confidently below ``threshold``: the union value if scored, else the bootstrap q90."""
    pm = edge.get("pairwise_mbar") or {}
    union = _finite(edge.get("mbar_overlap"))
    if union is not None:
        return union < float(threshold)
    if pm.get("status") != "ok":
        return False
    upper = _finite(pm.get("overlap_upper"))
    return upper is not None and upper < float(threshold)


def edge_is_weak_pairwise(edge: Mapping[str, Any], threshold: float) -> bool:
    """Spatial-edge weak rule under ``pairwise-mbar``.

    Weak-eligible are only the collector's own geometry edges (the proposers'
    bridge set) between states of one restraint pattern. The 3.1 graph edges
    (``neighbour``: any pair within the P7a radius, which reaches past the
    threshold on purpose -- a second or third neighbour is "below 0.15" without
    being a gap; ``spanning``) are measured and decide connectivity, never
    bridging. Unmeasured and cross-pattern edges are never weak.
    """
    pm = edge.get("pairwise_mbar") or {}
    if str(edge.get("edge_type")) in GRAPH_EDGE_TYPES or pm.get("pattern_pair") == "cross":
        return False
    return edge_below_threshold(edge, threshold)


def edge_sort_overlap(edge: Mapping[str, Any], policy: Any) -> Optional[float]:
    """The value weak edges are ordered by (worst first): the metric the policy grades on."""
    if str(getattr(policy, "edge_metric", DEFAULT_EDGE_METRIC)) != PAIRWISE_MBAR:
        return edge.get("overlap")
    union = _finite(edge.get("mbar_overlap"))
    if union is not None:
        return union
    return _finite((edge.get("pairwise_mbar") or {}).get("overlap"))


# ---- collector hook ----------------------------------------------------------------

def resolve_temperature_k(run_dir: Path, max_depth: int = 2) -> Optional[float]:
    """The run temperature from ``run_dir`` or, breadth-first, its sub-run directories."""
    from gareus.io import resolve_run_temperature_k  # noqa: PLC0415
    level = [Path(run_dir)]
    for _ in range(int(max_depth) + 1):
        nxt = []
        for d in level:
            try:
                t = resolve_run_temperature_k(d)
            except Exception:
                t = None
            if t is not None:
                return float(t)
            try:
                nxt.extend(sorted(p for p in d.iterdir() if p.is_dir()))
            except OSError:
                pass
        level = nxt
    return None


def _state_nodes(payload: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in payload.get("states", []) or []:
        pc = row.get("paired_cv") or {}
        rec = pc.get("restraint")
        if not rec:
            continue
        sid = int(row.get("state_id"))
        m1, m2 = pc.get("cv1") or {}, pc.get("cv2") or {}
        out[sid] = {"restraint": Restraint.from_record(rec), "window": row.get("epoch_window", -1),
                    "mean": (_finite(m1.get("mean")), _finite(m2.get("mean"))),
                    "var": (_finite(m1.get("var")), _finite(m2.get("var"))),
                    "n_pairs": int(pc.get("n_pairs", 0) or 0)}
    return out


def _new_edge(si: int, sj: int, kind: str, dist: float, nodes: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    def _w(s: int) -> int:
        w = nodes.get(s, {}).get("window", -1)
        return -1 if w is None else int(w)
    return {"state_i": int(si), "state_j": int(sj), "window_i": _w(si), "window_j": _w(sj),
            "edge_type": kind, "normalized_distance": float(dist), "overlap": None, "mbar_overlap": None,
            "exchange_attempts": 0, "exchange_accepted": 0, "exchange_acceptance": None, "warnings": [],
            "overlap_joint_2d": None, "overlap_joint_2d_reason": "edge_metric_graph_edge"}


def _components_summary(ids: Sequence[int], graded: Sequence[Tuple[int, int, Optional[float], bool]],
                        threshold: float, n_k: Sequence[int]) -> Dict[str, Any]:
    """Connectivity of the representative rung over every graded spatial edge.

    ``graded`` rows are (i, j, value, confidently_below): value = union overlap
    or bootstrap q90 (None when unmeasured). The verdict ``n_components`` keeps
    every edge that is NOT confidently below the threshold, so an unmeasured
    edge inside a component is never read as a disconnection (spec 10 item 2);
    ``n_measured_components`` keeps only measured edges at or above it (via
    ``overlap_components``) and ``n_graph_components`` every graded edge.
    """
    idx = {s: i for i, s in enumerate(ids)}
    rows = [(idx[a], idx[b], v, below) for a, b, v, below in graded if a in idx and b in idx]
    graph = graph_components(len(ids), [(i, j) for i, j, _v, _b in rows])
    verdict = graph_components(len(ids), [(i, j) for i, j, _v, below in rows if not below])
    out: Dict[str, Any] = {
        "n_components": len(verdict), "components": [[int(ids[i]) for i in c] for c in verdict],
        "components_rule": "graded spatial edges not confidently below threshold (unmeasured kept)",
        "n_graph_components": len(graph)}
    try:
        from gareus.mbar_analysis.pmf import overlap_components  # noqa: PLC0415
        mat = np.full((len(ids), len(ids)), np.nan)
        for i, j, v, _b in rows:
            if v is not None:
                mat[i, j] = mat[j, i] = float(v)
        comp = overlap_components(mat, float(threshold), n_k=np.asarray(n_k, dtype=float))
        out["n_measured_components"] = int(comp.get("n_components"))
        out["measured_component_sizes"] = sorted((len(c) for c in comp.get("components", [])), reverse=True)
    except Exception as exc:  # the verdict above stays
        out["measured_components_error"] = f"{type(exc).__name__}: {exc}"
    return out


def attach_edge_metric(payload: Dict[str, Any], policy: Any, run_dir: Path, *,
                       temperature_k: Optional[float] = None,
                       n_bootstrap: int = N_BOOTSTRAP) -> Dict[str, Any]:
    """Grade the payload's spatial edges (and the 3.1 graph) under ``policy.edge_metric``.

    ``marginal`` (the default): returns the payload untouched, no key added.
    Otherwise adds ``pairwise_mbar`` to every non-rung edge, appends the graph
    edges not already present, and writes ``payload["edge_metric"]``. Never raises.
    """
    metric = str(getattr(policy, "edge_metric", DEFAULT_EDGE_METRIC) or DEFAULT_EDGE_METRIC)
    if metric == DEFAULT_EDGE_METRIC:
        return payload
    threshold = float(getattr(policy, "min_rung_overlap", 0.15))
    min_neff = float(getattr(policy, "min_edge_neff", DEFAULT_MIN_EDGE_NEFF))
    record: Dict[str, Any] = {
        "metric": metric, "status": "ok", "error": None, "method": METHOD,
        "threshold": threshold, "threshold_field": "min_rung_overlap",
        "decision": f"weak iff same-pattern, measured and bootstrap q{int(BOOTSTRAP_QUANTILES[1] * 100)} "
                    "< threshold (union mbar_overlap point value when present)",
        "min_edge_neff": min_neff, "radius": DEFAULT_RADIUS, "spanning_neighbours": SPANNING_NEIGHBOURS,
        "n_bootstrap": int(n_bootstrap), "temperature_k": None, "warnings": [],
        # Graded at collection time. A union solve applied later (``_apply_union_edge_overlap``)
        # writes ``mbar_overlap``, which the weak predicate then reads instead; the counts,
        # warnings and components here are not refreshed.
        "stage": "pre_union",
    }
    edges = payload.setdefault("edges", [])
    try:
        if metric != PAIRWISE_MBAR:
            raise ValueError(f"unknown edge metric {metric!r} (choices {EDGE_METRICS})")
        pc = payload.get("paired_cv") or {}
        if pc.get("status") != "ok" or not pc.get("npz"):
            raise RuntimeError(f"paired-CV data unavailable ({pc.get('error') or 'no paired_cv record'})")
        temp = temperature_k if temperature_k is not None else resolve_temperature_k(run_dir)
        if temp is None:
            raise RuntimeError("run temperature could not be resolved")
        record["temperature_k"] = float(temp)
        beta = beta_mol_per_kcal(float(temp))
        from gareus.adaptive.paired_cv import load_paired_subsamples  # noqa: PLC0415
        sub = load_paired_subsamples(Path(pc["npz"]))
        nodes = _state_nodes(payload)
        rep = representative_rung(n["restraint"].gamd_lambda for n in nodes.values())
        rung_ids = sorted(s for s, n in nodes.items()
                          if rep is not None and round(float(n["restraint"].gamd_lambda or 0.0), 6) == rep)
        points = [NeighbourPoint(primary_center=float(nodes[s]["restraint"].primary_center),
                                 primary_k=nodes[s]["restraint"].primary_k,
                                 secondary_center=nodes[s]["restraint"].secondary_center,
                                 secondary_k=nodes[s]["restraint"].secondary_k,
                                 rung=nodes[s]["restraint"].gamd_lambda, sampled_mean=nodes[s]["mean"])
                  for s in rung_ids]
        pooled = pooled_free_axis_sd([(nodes[s]["restraint"], nodes[s]["var"]) for s in rung_ids])
        graph = [(rung_ids[i], rung_ids[j], kind, d)
                 for i, j, kind, d in build_edge_graph(points, float(temp), pooled_sd=pooled)]
        present = {(min(int(e["state_i"]), int(e["state_j"])), max(int(e["state_i"]), int(e["state_j"])))
                   for e in edges}
        graph_kind = {(si, sj): kind for si, sj, kind, _d in graph}
        added = 0
        for si, sj, kind, d in graph:
            if (si, sj) not in present:
                edges.append(_new_edge(si, sj, kind, d, nodes))
                present.add((si, sj))
                added += 1

        def _samples(sid: int) -> Optional[StateSamples]:
            if sid not in nodes or sid not in sub:
                return None
            s = sub[sid]
            return StateSamples(sid, nodes[sid]["restraint"], s["cv1"], s["cv2"], s["source_index"])

        counts: Dict[str, int] = {}
        graded: List[Tuple[int, int, Optional[float]]] = []
        n_weak = n_unmeasured = 0
        for edge in edges:
            if str(edge.get("edge_type")) == "rung":
                continue
            si, sj = sorted((int(edge["state_i"]), int(edge["state_j"])))
            a, b = _samples(si), _samples(sj)
            if a is None or b is None:
                res = _unmeasured("no_samples")
            else:
                try:
                    res = evaluate_edge(a, b, beta, min_neff=min_neff, n_bootstrap=n_bootstrap)
                except Exception as exc:  # one bad edge stays unmeasured
                    res = _unmeasured("error", error=f"{type(exc).__name__}: {exc}")
            res["graph_kind"] = graph_kind.get((si, sj))   # None: a collector edge outside the 3.1 graph
            edge["pairwise_mbar"] = res
            kind = str(edge.get("edge_type"))
            counts[kind] = counts.get(kind, 0) + 1
            warnings = edge.setdefault("warnings", [])
            if edge_is_weak_pairwise(edge, threshold):
                n_weak += 1
                if "low_pairwise_mbar_overlap" not in warnings:
                    warnings.append("low_pairwise_mbar_overlap")
            elif res.get("status") != "ok" and _finite(edge.get("mbar_overlap")) is None:
                n_unmeasured += 1
                if "pairwise_mbar_unmeasured" not in warnings:
                    warnings.append("pairwise_mbar_unmeasured")
            union = _finite(edge.get("mbar_overlap"))
            below = edge_below_threshold(edge, threshold)
            res["below_threshold"] = bool(below)
            graded.append((si, sj, union if union is not None else
                           (res.get("overlap_upper") if res.get("status") == "ok" else None), below))
        record.update(representative_rung=rep, n_graph_nodes=len(rung_ids), n_graph_edges=len(graph),
                      n_edges_added=added, n_edges_graded_by_type=counts, n_weak=n_weak,
                      n_unmeasured=n_unmeasured, pooled_free_axis_sd=list(pooled),
                      n_below_threshold=sum(1 for g in graded if g[3]))
        record["components"] = _components_summary(rung_ids, graded, threshold,
                                                    [nodes[s]["n_pairs"] for s in rung_ids])
        if record["components"]["n_components"] > 1:
            record["warnings"].append(
                f"spatial overlap graph splits into {record['components']['n_components']} components "
                f"at {threshold:g} (no action here; spec 3.3 R1)")
    except Exception as exc:  # diagnostics never kill a completed epoch
        record.update(status="error", error=f"{type(exc).__name__}: {exc}")
        for edge in edges:
            if str(edge.get("edge_type")) != "rung" and "pairwise_mbar" not in edge:
                edge["pairwise_mbar"] = _unmeasured("edge_metric_error")
    payload["edge_metric"] = record
    return payload


__all__ = ["DEFAULT_EDGE_METRIC", "DEFAULT_MIN_EDGE_NEFF", "EDGE_METRICS", "PAIRWISE_MBAR", "Restraint",
           "StateSamples", "attach_edge_metric", "beta_mol_per_kcal", "blocking_inefficiency",
           "build_edge_graph", "edge_is_weak_pairwise", "edge_sort_overlap", "evaluate_edge",
           "pooled_free_axis_sd", "representative_rung", "resolve_temperature_k", "solve_two_state_df",
           "two_state_overlap"]
