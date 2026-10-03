"""CV2 resolution actions R1-R3 (spec 3.3), behind ``--ap-cv2-resolution`` (off by default).

Spec: docs/superpowers/specs/2026-09-29-adaptive-cv2-resolution-design.md, Section 3.3.
Pure decision logic (NumPy only); disk I/O, the union solve for R2 and the epoch-loop
entry point are in ``cv2_resolution_io``, the R2 coverage math in ``cv2_coverage``.

Eligibility (structural): a state takes part only if it restrains BOTH axes above their
floors -- k1 > max(0, --cv1-k-min) and k2 > max(0, --cv2-k-min) with a CV2 centre. A
floored spring is effectively no restraint, so the spec's "k2 > 0" is read as "k2 above the
cv2_k_min floor". Anchors and axis states (k1 = 0 or k2 = 0, ``_is_anchor_or_axis_state``)
fail this by construction. Only states on the representative rung (lambda = 0 if present)
are analysed; an action names a centre and the applier replicates it onto every rung.

R1 CV2-gap bridge -- the collector's same-pattern geometry edges between two eligible states
whose restraint centres differ MAINLY IN CV2: with the P7a per-axis normalised distances
d_ax = |c_a - c_b| / sqrt(sigma_a^2 + sigma_b^2), sigma = sqrt(RT/k), the edge is CV2-mainly
iff d2^2 >= MAINLY_CV2_SHARE (1/2) x (d1^2 + d2^2), i.e. d2 >= d1. Under the 3.1 metric
(``--ap-edge-metric pairwise-mbar``; without it R1 is ``unavailable``):
  * structural: confidently below threshold AND its ends lie in different 3.1 components
    (``edge_metric.components``) -> bridge now (one per component pair, the closest edge);
  * weak: ``edge_is_weak_pairwise`` inside one component -> bridge;
  * unmeasured: extend both endpoints; bridge once it has been unmeasured-or-weak in
    UNMEASURED_WAIT_EPOCHS + 1 consecutive epochs (first sighting + 2 epochs of sampling).
    An ``extend`` action is a lifecycle record only: the epoch schedule gives every state a
    uniform share (top-ups, when on, allocate by their own deficits), so "extend sampling
    first" means the edge waits through two ordinary epochs of sampling.
  The per-edge history is keyed by (min, max) state id and stores the SET of epochs per
  status, so re-proposing an epoch (a kill before the P3 ledger) never double-counts.

R3 mode resolution -- a state whose CV2 subsample (P4 NPZ) has two accepted mixture modes
passing ``mode_pair_resolvable`` (depth >= 1 kT, both >= 10 %) AND at least
``refine_min_transitions`` core-to-core crossings (``count_transitions``; which crossings is
``refine_transition_count``: replica-path by default, see cv2_resolution_rules._transitions).
Mixture "members" for production windows are time blocks: each state's subsample is cut into
MEMBER_BLOCKS contiguous blocks, never across a source, so a mode must be revisited in >= 8
blocks to be accepted. Bimodal without transitions is flagged ``trapped_or_orthogonal`` and
nothing is inserted. Otherwise the two children at the mode means (parent kept) are planned;
with ``refine_r3_mode`` "insert" they become an ``insert`` action, with "flag" (the default
since T2: inserts gave no PMF benefit at matched budget and the 4 x cap refused every genuine
candidate) the candidate is ``flagged`` with reason ``r3_flag_only``, keeps the would-be
children and springs in ``proposal`` and ``metrics.would_be`` = {decision proposed|refused,
refusal}, and never becomes an action, draws on the budget or blocks convergence. A would-be
spring-cap refusal stays ``refused`` in either mode (nothing to insert).

Springs (every new window): target sampled sigma from the spacing (R3: 2 delta / 1.5 with the
children at +/- delta; R1/R2: the new centre's distance to its neighbour / 1.5), never below
``refine_min_sigma``; F''_est from the window's own samples UNDER ITS OWN SPRING:
F'' = max(0, estimate_f2(...) - k2_own), where ``estimate_f2`` is the sampled precision
RT/var shrunk in PRECISION space toward the pooled one (RT (w/var_c + (1 - w)/var_pool),
w = n/(n + 8); R1/R2 use the window's own variance, w = 1); for biased samples that precision
is k2 + F'', hence the subtraction. k2 = ``shape_rule_k2`` (width rule raised to the
mean-compression floor) in [cv2_k_min, min(cv2_k_max, 4 x parent k2)], then the 3.4 coupling
gate when configured. Refused: k2 at or below the floor (``k2_at_floor``) and a cap or gate
that holds k2 below the compression floor (``k2_capped_below_compression``). Sd/sigma_w,
Sarle bimodality and curvature are reported, never trigger.

Budget: resolution actions draw only on the P1 reserve (``reserve_allowances``), in priority
order R1 structural > R1 weak > R1 after the wait > R2 > R3, costing one centre (R1/R2, n
states on an n-rung ladder) or two (R3, 2n). Without a reserve every proposal is refused
(``no_reserve``; the driver hook prints one WARNING per campaign job); a dry run can pass
``ignore_budget``. Budget refusals are recorded and CAUTION-graded (3.7) but never block the
convergence gate: only funded (``proposed``) work does.

R3 records the test that decided every candidate in ``metrics.r3_gate`` (``R3_GATES``) with
the measured values in ``metrics.r3_gate_values``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from gareus.adaptive.cv2_shape import (DEFAULT_MIN_MEAN_COMPRESSION, DEFAULT_MIN_MODE_MEMBERS,
                                       compression_floor_k2, estimate_f2, fit_cv2_mixture, mode_depth,
                                       mode_pair_resolvable, predicted_sampled_sigma, shape_rule_k2)
from gareus.adaptive.edge_metric import edge_below_threshold, edge_is_weak_pairwise
from gareus.swarm.ladder_design import R_KCAL_MOL_K

SCHEMA_VERSION = "cv2_resolution_report_v3"
REPORT_NAME = "cv2_resolution_report.json"
HISTORY_NAME = "cv2_resolution_history.json"
HISTORY_SCHEMA = "cv2_resolution_history_v1"
METADATA_KEY = "cv2_resolution"
SOURCE = "adaptive_production_cv2_resolution"
MAINLY_CV2_SHARE = 0.5
UNMEASURED_WAIT_EPOCHS = 2
R3_MIN_DEPTH_KT = 1.0
R3_MIN_WEIGHT = 0.10
MEMBER_BLOCKS = 32
SPACING_SIGMA = 1.5
MAX_K2_GROWTH = 4.0
CORE_HALF_SD = 0.5
# R2 bootstrap blocks (cv2_coverage._block_ids): each state's rows are cut, within each sample
# source, into blocks of ceil(BOOT_BLOCK_G_MULTIPLE x g) rows, g = the state's statistical
# inefficiency (effective_samples.pooled_inefficiency, max over CV1/CV2), and never fewer than
# BOOT_MIN_BLOCKS blocks per state (the guard is recorded when it binds).
BOOT_BLOCK_G_MULTIPLE = 5.0
BOOT_MIN_BLOCKS = 5
NON_BRIDGE_EDGE_TYPES = ("rung", "neighbour", "spanning", "pattern_link")
PRIORITY = ("R1:structural", "R1:weak", "R1:unmeasured", "R2", "R3")
BURN_IN_NOTE = ("standard: the US pull is not written as samples, per-state burnin_steps 0, "
                "union MBAR per-state equilibration (t0) detection")
# Knob defaults. Only refine_budget_fraction (0.5) and refine_protect_epochs (2) are spec
# values; refine_transition_count, refine_r3_mode, coverage_count and refine_pmf_sigma_kT
# follow the T2 calibration (t2_synthetic.md 9, 9.9); the rest are uncalibrated choices.
DEFAULTS = {"coverage_min_windows": 2.0, "refine_min_transitions": 10, "refine_pmf_sigma_kT": 0.25,
            "refine_budget_fraction": 0.5, "refine_protect_epochs": 2, "refine_min_sigma": 0.1,
            "refine_transition_count": "replica-path", "refine_r3_mode": "flag",
            "coverage_count": "same-column", "coverage_bootstrap": "fixed-f", "cv2_bridge_sets": False}
# R3 crossing counts (``count_transitions`` over different runs; see cv2_resolution_rules._transitions).
TRANSITION_COUNTS = ("replica", "replica-path", "state-series")
# R3 modes: "flag" records the would-be children and never emits an insert; "insert" acts.
R3_MODES = ("flag", "insert")
# R2 contributor count: "same-column" counts only centres of the interval's own CV1 column.
COVERAGE_COUNTS = ("any", "same-column")
# R2 bootstrap: "fixed-f" resamples the weights at the point MBAR f; "resolve-f" re-solves the
# lambda = 0 MBAR per replicate (cv2_coverage.resolve_f_sigma).
COVERAGE_BOOTSTRAPS = ("fixed-f", "resolve-f")
BUDGET_REFUSALS = ("no_reserve", "resolution_budget")
R3_FLAG_ONLY = "r3_flag_only"
# The test that decided an R3 candidate (metrics["r3_gate"]), in the order they are applied.
# Mixture gates (mode_analysis / r3_mode_gate): single_component (the fit has < 2 components),
# member_support (< 2 components with >= min_mode_members member blocks), no_density_minimum
# (no accepted pair has a density minimum between its means), depth_below_1kT (the deepest
# accepted pair is shallower than R3_MIN_DEPTH_KT), mode_weight_below_10pct (a pair deep
# enough has a mode below R3_MIN_WEIGHT); then too_few_samples (skipped before the fit),
# no_replica_series / transitions_below_min (flagged), passed (proposed or refused; a refused
# candidate also carries its refusal code).
R3_GATES = ("too_few_samples", "single_component", "member_support", "no_density_minimum", "depth_below_1kT",
            "mode_weight_below_10pct", "no_replica_series", "transitions_below_min", "passed")


@dataclass(frozen=True)
class ResolutionSettings:
    coverage_min_windows: float = DEFAULTS["coverage_min_windows"]
    refine_min_transitions: int = DEFAULTS["refine_min_transitions"]
    refine_pmf_sigma_kT: float = DEFAULTS["refine_pmf_sigma_kT"]
    refine_budget_fraction: float = DEFAULTS["refine_budget_fraction"]
    refine_protect_epochs: int = DEFAULTS["refine_protect_epochs"]
    refine_min_sigma: float = DEFAULTS["refine_min_sigma"]
    refine_transition_count: str = DEFAULTS["refine_transition_count"]
    refine_r3_mode: str = DEFAULTS["refine_r3_mode"]
    coverage_count: str = DEFAULTS["coverage_count"]
    coverage_bootstrap: str = DEFAULTS["coverage_bootstrap"]
    # R1 bridges a CV2 gap with the whole bridge set of its layout column (spec 2026-10-03-cv2-bridge-sets).
    cv2_bridge_sets: bool = DEFAULTS["cv2_bridge_sets"]
    temperature_k: float = 300.0
    k1_min: float = 0.0
    k2_min: float = 0.0
    k2_max: Optional[float] = None
    threshold: float = 0.15

    def __post_init__(self) -> None:
        for name, allowed in (("refine_transition_count", TRANSITION_COUNTS), ("refine_r3_mode", R3_MODES),
                              ("coverage_count", COVERAGE_COUNTS), ("coverage_bootstrap", COVERAGE_BOOTSTRAPS)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")

    @classmethod
    def from_policy(cls, policy: Any, **extra: Any) -> "ResolutionSettings":
        vals = {k: getattr(policy, k, v) for k, v in DEFAULTS.items()}
        vals["threshold"] = float(getattr(policy, "min_rung_overlap", 0.15))
        return cls(**vals, **extra)

    @property
    def rt(self) -> float:
        return R_KCAL_MOL_K * float(self.temperature_k)

    def as_record(self) -> Dict[str, Any]:
        rec = {k: getattr(self, k) for k in self.__dataclass_fields__}
        rec.update(mainly_cv2_share=MAINLY_CV2_SHARE, unmeasured_wait_epochs=UNMEASURED_WAIT_EPOCHS,
                   r3_min_depth_kT=R3_MIN_DEPTH_KT, r3_min_weight=R3_MIN_WEIGHT, member_blocks=MEMBER_BLOCKS,
                   min_mode_members=DEFAULT_MIN_MODE_MEMBERS, spacing_sigma=SPACING_SIGMA,
                   min_mean_compression=DEFAULT_MIN_MEAN_COMPRESSION,
                   max_k2_growth=MAX_K2_GROWTH, core_half_sd=CORE_HALF_SD,
                   boot_block_g_multiple=BOOT_BLOCK_G_MULTIPLE, boot_min_blocks=BOOT_MIN_BLOCKS,
                   uncalibrated=["refine_min_transitions", "refine_min_sigma"])
        return rec


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


@dataclass(frozen=True)
class StateView:
    """One registry state plus its sampled CV2 moments (P4 ``paired_cv``)."""
    state_id: int
    c1: float
    k1: Optional[float]
    c2: Optional[float]
    k2: Optional[float]
    lam: float
    created_epoch: int = 0
    mean2: Optional[float] = None
    var2: Optional[float] = None
    n_pairs: int = 0
    moments2: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def sigma_w(self, axis: int, rt: float) -> Optional[float]:
        k = _finite(self.k1 if axis == 1 else self.k2)
        return math.sqrt(rt / k) if k is not None and k > 0 else None


def state_views(registry: Any, payload: Mapping[str, Any]) -> Dict[int, StateView]:
    """Active registry states joined to the payload's P4 moments (missing -> None)."""
    rows = {int(r.get("state_id")): r for r in payload.get("states", []) or []}
    out: Dict[int, StateView] = {}
    for s in registry.active_states():
        pc = (rows.get(int(s.state_id)) or {}).get("paired_cv") or {}
        m2 = pc.get("cv2") or {}
        out[int(s.state_id)] = StateView(
            int(s.state_id), float(s.primary_center), _finite(s.primary_k), _finite(s.secondary_center),
            _finite(s.secondary_k), float(s.gamd_lambda or 0.0), int(s.created_epoch or 0),
            _finite(m2.get("mean")), _finite(m2.get("var")), int(pc.get("n_pairs", 0) or 0),
            dict(m2), dict(s.metadata or {}))
    return out


def representative_ids(views: Mapping[int, StateView]) -> List[int]:
    lams = sorted({round(v.lam, 6) for v in views.values()})
    if not lams:
        return []
    rep = 0.0 if any(abs(x) <= 1e-9 for x in lams) else lams[0]
    return sorted(s for s, v in views.items() if abs(round(v.lam, 6) - rep) <= 1e-9)


def eligibility(view: StateView, settings: ResolutionSettings) -> Tuple[bool, str]:
    if view.k1 is None or view.k1 <= max(0.0, float(settings.k1_min)):
        return False, "cv1_unrestrained_or_at_floor"
    if view.c2 is None or view.k2 is None or view.k2 <= max(0.0, float(settings.k2_min)):
        return False, "cv2_unrestrained_or_at_floor"
    return True, "eligible"


def is_protected(view: StateView, epoch: int, settings: ResolutionSettings) -> bool:
    """A state an R action created is left alone for ``refine_protect_epochs`` epochs."""
    if METADATA_KEY not in (view.metadata or {}):
        return False
    return int(epoch) < int(view.created_epoch) + int(settings.refine_protect_epochs)


# ---- springs -----------------------------------------------------------------------------

def f2_under_bias(var: Optional[float], k2_own: Optional[float], temperature_k: float, *,
                  pooled_var: float = float("nan"), n_members: float = 1.0e9) -> float:
    """Landscape curvature from samples taken under spring ``k2_own``: the (precision-space
    shrunk, ``estimate_f2``) sampled precision RT/var minus k2_own, floored at 0."""
    if var is None or not (var > 0):
        return 0.0
    return max(0.0, estimate_f2(var, pooled_var, n_members, temperature_k) - float(k2_own or 0.0))


def child_spring(sigma_target: float, f2: float, parent_k2: float, settings: ResolutionSettings) -> Dict[str, Any]:
    """k2 for one new window from the shape rule (width rule raised to the mean-compression
    floor k2 >= F'' c/(1 - c), then clipped to [k_min, k_max]). Refusals: ``k2_at_floor`` (at
    or below cv2_k_min) and ``k2_capped_below_compression`` (the cap, 4 x parent k2 or
    cv2_k_max, holds k2 below the compression floor: the window mean would not reach even
    ``min_mean_compression`` of the way to its centre)."""
    sigma = max(float(sigma_target), float(settings.refine_min_sigma))
    k_min = max(0.0, float(settings.k2_min))
    k_max = MAX_K2_GROWTH * float(parent_k2)
    if settings.k2_max is not None and float(settings.k2_max) > 0:
        k_max = min(k_max, float(settings.k2_max))
    k2 = shape_rule_k2(sigma, f2, settings.temperature_k, k_min, max(k_min, k_max))
    floor_c = float(compression_floor_k2(f2, DEFAULT_MIN_MEAN_COMPRESSION))
    width_k2 = settings.rt / sigma ** 2 - max(0.0, float(f2))
    compression = k2 / (k2 + max(0.0, float(f2))) if k2 + max(0.0, float(f2)) > 0 else 1.0
    out = {"sigma_target": float(sigma_target), "sigma_used": sigma, "f2_est": float(f2), "k2": float(k2),
           "k2_cap": float(k_max), "predicted_sampled_sigma": predicted_sampled_sigma(k2, f2, settings.temperature_k),
           "mean_compression": float(compression), "min_mean_compression": DEFAULT_MIN_MEAN_COMPRESSION,
           "compression_floor_k2": floor_c,
           "at_compression_floor": bool(floor_c > width_k2 and abs(k2 - floor_c) <= 1e-9 * max(1.0, floor_c)),
           "refusal": None}
    if k2 <= k_min * (1.0 + 1e-12):
        out["refusal"] = "k2_at_floor"
    elif k2 < floor_c * (1.0 - 1e-9):
        out["refusal"] = "k2_capped_below_compression"
    return out


def gate_child(child: Dict[str, Any], c1: float, k1: float, c2: float, gate: Any, context: str,
               settings: ResolutionSettings) -> Dict[str, Any]:
    """Apply the 3.4 coupling gate (None = off) to a planned child; refusal below the floor."""
    if gate is None or child.get("refusal"):
        return child
    k2, note = gate.gate(c1, k1, c2, child["k2"], context=context)
    if k2 is None:
        return {**child, "refusal": "below_k_min", "gate_note": note}
    f2 = max(0.0, float(child["f2_est"]))
    out = {**child, "k2": float(k2), "gate_note": note, "mean_compression": float(k2) / (float(k2) + f2) if float(k2) + f2 > 0 else 1.0,
           "predicted_sampled_sigma": predicted_sampled_sigma(float(k2), f2, settings.temperature_k)}
    if float(k2) <= max(0.0, float(settings.k2_min)):
        out["refusal"] = "k2_at_floor"
    elif float(k2) < float(child.get("compression_floor_k2", 0.0)) * (1.0 - 1e-9):
        out["refusal"] = "k2_capped_below_compression"
    return out


# ---- R3 ----------------------------------------------------------------------------------

def member_blocks(source_index: Optional[np.ndarray], n: int, n_blocks: int = MEMBER_BLOCKS) -> np.ndarray:
    """Contiguous time blocks of one state's subsample, never across a source."""
    src = np.zeros(n, dtype=np.int64) if source_index is None else np.asarray(source_index, dtype=np.int64)
    length = max(1, int(math.ceil(n / float(n_blocks))))
    pos = np.zeros(n, dtype=np.int64)
    for s in np.unique(src):
        idx = np.flatnonzero(src == s)
        pos[idx] = np.arange(idx.size)
    return src * (10 ** 7) + pos // length


def sarle_bimodality(moments: Mapping[str, Any]) -> Optional[float]:
    """Sarle's coefficient (skew^2 + 1) / (kurt_excess + 3 (n-1)^2 / ((n-2)(n-3))); > 5/9 bimodal-ish."""
    n, sk, ku = moments.get("n"), _finite(moments.get("skewness")), _finite(moments.get("kurtosis_excess"))
    if sk is None or ku is None or not n or int(n) < 4:
        return None
    n = int(n)
    return (sk * sk + 1.0) / (ku + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def _pair_record(i: int, j: int, d: Mapping[str, Any]) -> Dict[str, Any]:
    return {"a": int(i), "b": int(j), "depth_kT": float(d["depth_kT"]), "bimodal": bool(d["bimodal"]),
            "weight_a": float(d["weight_a"]), "weight_b": float(d["weight_b"]),
            "mean_a": float(d["mean_a"]), "mean_b": float(d["mean_b"])}


def r3_mode_gate(components: Sequence[Any]) -> Dict[str, Any]:
    """Which mixture test decides R3 for these components (``R3_GATES``, applied in order),
    with every measured value: {gate, pair (i, j) | None, depth (mode_depth dict) | None,
    values {n_components, n_accepted, component_members, component_weights, min_mode_members,
    min_depth_kT, min_weight, pairs [every accepted pair], deepest_pair}}. ``passed`` names
    the deepest resolvable pair."""
    comps = list(components)
    acc = [i for i, c in enumerate(comps) if c.accepted]
    values: Dict[str, Any] = {"n_components": len(comps), "n_accepted": len(acc),
                              "component_members": [int(c.n_members) for c in comps],
                              "component_weights": [float(c.weight) for c in comps],
                              "min_mode_members": DEFAULT_MIN_MODE_MEMBERS, "min_depth_kT": R3_MIN_DEPTH_KT,
                              "min_weight": R3_MIN_WEIGHT, "pairs": [], "deepest_pair": None}
    if len(comps) < 2:
        return {"gate": "single_component", "pair": None, "depth": None, "values": values}
    if len(acc) < 2:
        return {"gate": "member_support", "pair": None, "depth": None, "values": values}
    best, deepest = None, None
    for ii, i in enumerate(acc):
        for j in acc[ii + 1:]:
            d = mode_depth(comps, i, j)
            values["pairs"].append(_pair_record(i, j, d))
            if deepest is None or d["depth_kT"] > deepest[2]["depth_kT"]:
                deepest = (i, j, d)
            if mode_pair_resolvable(d, min_depth_kT=R3_MIN_DEPTH_KT, min_weight=R3_MIN_WEIGHT):
                if best is None or d["depth_kT"] > best[2]["depth_kT"]:
                    best = (i, j, d)
    values["deepest_pair"] = _pair_record(*deepest)
    if best is not None:
        return {"gate": "passed", "pair": (best[0], best[1]), "depth": best[2], "values": values}
    if not any(p["bimodal"] for p in values["pairs"]):
        gate = "no_density_minimum"
    elif deepest[2]["depth_kT"] < R3_MIN_DEPTH_KT:
        gate = "depth_below_1kT"
    else:
        gate = "mode_weight_below_10pct"
    return {"gate": gate, "pair": None, "depth": None, "values": values}


def mode_analysis(cv2: np.ndarray, source_index: Optional[np.ndarray], *, seed: int = 0) -> Dict[str, Any]:
    """Mixture fit of one window's CV2 subsample, its deepest resolvable mode pair, and the
    mixture gate that decided (``r3_mode_gate``)."""
    z = np.asarray(cv2, dtype=float)
    fit = fit_cv2_mixture(z, member_blocks(source_index, z.size), max_components=3,
                          min_mode_members=DEFAULT_MIN_MODE_MEMBERS, seed=seed)
    comps = list(fit.components)
    g = r3_mode_gate(comps)
    out = {"fit": fit.as_record(), "bimodal": g["pair"] is not None, "pair": None, "depth": None,
           "gate": g["gate"], "gate_values": g["values"]}
    if g["pair"] is not None:
        i, j = g["pair"]
        lo, hi = sorted((comps[i], comps[j]), key=lambda c: c.mean)
        out.update(pair=[lo.as_record(), hi.as_record()], depth=g["depth"])
    return out


def core_bounds(lo: Mapping[str, Any], hi: Mapping[str, Any], barrier: float) -> Tuple[float, float]:
    """Core edges: barrier -/+ CORE_HALF_SD x the narrower mode's sd, never past a mode mean."""
    h = CORE_HALF_SD * min(float(lo["sd"]), float(hi["sd"]))
    upper_a = max(barrier - h, 0.5 * (float(lo["mean"]) + barrier))
    lower_b = min(barrier + h, 0.5 * (float(hi["mean"]) + barrier))
    return float(upper_a), float(lower_b)


def count_transitions(runs: Iterable[np.ndarray], upper_a: float, lower_b: float) -> int:
    """Core-to-core crossings: each run is labelled A (<= upper_a) / B (>= lower_b), samples
    in between are ignored, and every change of label within a run counts once."""
    total = 0
    for run in runs:
        z = np.asarray(run, dtype=float)
        lab = np.where(z <= upper_a, 0, np.where(z >= lower_b, 1, -1))
        lab = lab[lab >= 0]
        if lab.size > 1:
            total += int(np.count_nonzero(np.diff(lab)))
    return total


# ---- R1 ----------------------------------------------------------------------------------

def axis_distances(a: StateView, b: StateView, rt: float) -> Tuple[Optional[float], Optional[float]]:
    """P7a per-axis normalised distances |dc| / sqrt(sigma_a^2 + sigma_b^2) (None if unknown)."""
    out = []
    for axis, ca, cb in ((1, a.c1, b.c1), (2, a.c2, b.c2)):
        sa, sb = a.sigma_w(axis, rt), b.sigma_w(axis, rt)
        if sa is None or sb is None or ca is None or cb is None:
            out.append(None)
        else:
            out.append(abs(float(ca) - float(cb)) / math.sqrt(sa * sa + sb * sb))
    return out[0], out[1]


def mainly_cv2(d1: Optional[float], d2: Optional[float]) -> bool:
    if d1 is None or d2 is None:
        return False
    total = d1 * d1 + d2 * d2
    return total > 0 and d2 * d2 >= MAINLY_CV2_SHARE * total


def edge_key(i: int, j: int) -> str:
    a, b = sorted((int(i), int(j)))
    return f"{a}-{b}"


def classify_edge(edge: Mapping[str, Any], comp_of: Mapping[int, int], threshold: float) -> str:
    si, sj = int(edge["state_i"]), int(edge["state_j"])
    split = si in comp_of and sj in comp_of and comp_of[si] != comp_of[sj]
    if edge_below_threshold(edge, threshold) and split:
        return "structural"
    if edge_is_weak_pairwise(edge, threshold):
        return "weak"
    pm = edge.get("pairwise_mbar") or {}
    if pm.get("status") != "ok" and _finite(edge.get("mbar_overlap")) is None:
        return "unmeasured"
    return "ok"


def update_history(history: Mapping[str, Any], epoch: int, statuses: Mapping[str, str],
                   active_ids: Iterable[int]) -> Dict[str, Any]:
    """A new history with this epoch's edge statuses (sets of epochs per status, per edge).

    Recording the same epoch again replaces it (idempotent); an edge with a retired end is
    dropped; an edge seen "ok" keeps its record (the consecutive count resets on it)."""
    alive = {int(s) for s in active_ids}
    edges: Dict[str, Dict[str, List[int]]] = {}
    for key, rec in (history.get("edges") or {}).items():
        a, b = (int(x) for x in key.split("-"))
        if a in alive and b in alive:
            edges[key] = {st: sorted({int(e) for e in eps if int(e) != int(epoch)}) for st, eps in rec.items()}
    for key, status in statuses.items():
        rec = edges.setdefault(key, {})
        rec[status] = sorted(set(rec.get(status, [])) | {int(epoch)})
    return {"schema_version": HISTORY_SCHEMA, "edges": edges}


def consecutive_bad(history: Mapping[str, Any], key: str, epoch: int) -> int:
    """Consecutive epochs ending at ``epoch`` in which the edge was unmeasured or weak."""
    rec = (history.get("edges") or {}).get(key) or {}
    bad = set(rec.get("unmeasured", [])) | set(rec.get("weak", [])) | set(rec.get("structural", []))
    n, e = 0, int(epoch)
    while e in bad:
        n, e = n + 1, e - 1
    return n


def bridge_child(a: StateView, b: StateView, settings: ResolutionSettings) -> Dict[str, Any]:
    """The R1 bridge: at the midpoint of the endpoints' SAMPLED CV2 means (nominal CV1)."""
    m_a = a.mean2 if a.mean2 is not None else a.c2
    m_b = b.mean2 if b.mean2 is not None else b.c2
    c2 = 0.5 * (float(m_a) + float(m_b))
    delta = 0.5 * abs(float(m_b) - float(m_a))
    f2 = max(f2_under_bias(a.var2, a.k2, settings.temperature_k), f2_under_bias(b.var2, b.k2, settings.temperature_k))
    spring = child_spring(delta / SPACING_SIGMA if delta > 0 else settings.refine_min_sigma, f2,
                          max(float(a.k2), float(b.k2)), settings)
    parent = a if abs(float(m_a) - c2) <= abs(float(m_b) - c2) else b
    return {"parent_state_id": int(parent.state_id), "primary_center": 0.5 * (a.c1 + b.c1),
            "primary_k": 0.5 * (float(a.k1) + float(b.k1)), "secondary_center": c2, "centred_on": "sampled_means",
            **spring}


# ---- actions and budget ------------------------------------------------------------------

def child_params(child: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    return (float(child["primary_center"]), float(child["primary_k"]), float(child["secondary_center"]),
            float(child["k2"]))


def action_metadata(rule: str, cls: Optional[str], epoch: int, parent: int) -> Dict[str, Any]:
    return {METADATA_KEY: {"rule": rule, "class": cls, "proposed_epoch": int(epoch),
                           "seed_source_state_id": int(parent), "burn_in": BURN_IN_NOTE}}


def candidate_action(cand: Mapping[str, Any], epoch: int) -> Optional[Tuple]:
    prop = cand.get("proposal") or {}
    if not prop:
        return None
    parent, rule = int(prop["parent_state_id"]), str(cand["rule"])
    meta = action_metadata(rule, cand.get("class"), epoch, parent)
    reason = f"cv2_resolution {rule}{'/' + cand['class'] if cand.get('class') else ''}: {cand['reason']}"
    kids = [child_params(c) for c in prop["children"]]
    if rule == "R3" or len(kids) > 1:          # R3 children, or an R1 bridge set: one atomic insert
        return ("insert", parent, kids, reason, meta)
    return ("add", parent, kids[0], reason, meta)


def priority_of(cand: Mapping[str, Any]) -> int:
    key = f"{cand['rule']}:{cand.get('class')}" if cand["rule"] == "R1" else str(cand["rule"])
    return PRIORITY.index(key) if key in PRIORITY else len(PRIORITY)


def allocate(cands: Sequence[Dict[str, Any]], allowance: Any, n_rungs: int, *,
             ignore_budget: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fund proposed candidates in priority order from the resolution slots; refuse the rest."""
    pending = sorted([c for c in cands if c["decision"] == "proposed"], key=priority_of)
    rest = [c for c in cands if c["decision"] != "proposed"]
    mode = "ignored" if ignore_budget else ("reserve" if getattr(allowance, "governed", False) else "no_reserve")
    slots = None if ignore_budget or allowance is None else int(allowance.resolution_slots)
    spent, out = 0, []
    for c in pending:
        n_kids = len((c.get("proposal") or {}).get("children") or []) or 1
        cost = int(n_rungs) * (2 if c["rule"] == "R3" else n_kids)
        c = {**c, "cost_states": cost}
        if mode == "no_reserve":
            detail = getattr(allowance, "reason", "no_reserve")
            c.update(decision="refused", reason=f"no_reserve ({detail}): {c['reason']}", refusal="no_reserve")
        elif slots is not None and spent + cost > slots:
            c.update(decision="refused", refusal="resolution_budget",
                     reason=f"resolution_budget: {cost} states > {slots - spent} left; {c['reason']}")
        else:
            spent += cost
        out.append(c)
    record = {"mode": mode, "n_rungs": int(n_rungs), "resolution_slots": slots, "spent_states": spent,
              "allowance": None if allowance is None else {
                  k: getattr(allowance, k) for k in ("governed", "reason", "free_slots", "add_rung_slots",
                                                     "resolution_slots", "n_rungs", "add_rung_centres",
                                                     "resolution_centres")}}
    return out + rest, record


def summarise(cands: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, Dict[str, int]] = {}
    for c in cands:
        by_rule = counts.setdefault(str(c["rule"]), {})
        by_rule[str(c["decision"])] = by_rule.get(str(c["decision"]), 0) + 1
    # Only funded work blocks convergence. A budget refusal (no_reserve, resolution_budget)
    # repeats every epoch while the reserve is missing or spent, so counting it would pin the
    # gate at "continue" for the rest of the campaign; it is recorded (n_refused_budget) and
    # CAUTION-graded by the 3.7 row instead.
    blocking = sum(1 for c in cands if c["decision"] == "proposed")
    kids = [k for c in cands if c["decision"] == "proposed" for k in (c.get("proposal") or {}).get("children", [])]
    return {"counts": counts, "n_proposed": sum(1 for c in cands if c["decision"] == "proposed"),
            "n_windows_at_compression_floor": sum(1 for k in kids if k.get("at_compression_floor")),
            "n_blocking": int(blocking),
            "n_refused_budget": sum(1 for c in cands if c.get("refusal") in BUDGET_REFUSALS),
            "n_refused_spring_cap": sum(1 for c in cands if c.get("refusal") == "k2_capped_below_compression"),
            "n_trapped_or_orthogonal": sum(1 for c in cands if (c.get("metrics") or {}).get("trapped_or_orthogonal")),
            "n_r3_flag_only": sum(1 for c in cands if is_flag_only(c))}


def is_flag_only(cand: Mapping[str, Any]) -> bool:
    """An R3 candidate that would have been inserted but ``refine_r3_mode`` is "flag"."""
    return (str(cand.get("rule")) == "R3" and cand.get("decision") == "flagged"
            and str(cand.get("reason", "")).startswith(R3_FLAG_ONLY))


def new_candidate(rule: str, kind: str, state_ids: Sequence[int], decision: str, reason: str, **extra: Any
                  ) -> Dict[str, Any]:
    return {"rule": rule, "kind": kind, "state_ids": [int(s) for s in state_ids], "class": extra.pop("cls", None),
            "decision": decision, "reason": reason, "refusal": extra.pop("refusal", None),
            "metrics": extra.pop("metrics", {}), "proposal": extra.pop("proposal", None),
            "cost_states": 0, **extra}


__all__ = ["BUDGET_REFUSALS", "COVERAGE_COUNTS", "DEFAULTS", "R3_FLAG_ONLY", "R3_MODES", "TRANSITION_COUNTS",
           "is_flag_only", "HISTORY_NAME", "METADATA_KEY", "R3_GATES", "REPORT_NAME", "SCHEMA_VERSION",
           "SOURCE", "r3_mode_gate",
           "ResolutionSettings", "StateView", "allocate", "axis_distances", "bridge_child", "candidate_action",
           "child_spring", "classify_edge", "consecutive_bad", "core_bounds", "count_transitions", "edge_key",
           "eligibility", "f2_under_bias", "gate_child", "is_protected", "mainly_cv2", "member_blocks",
           "mode_analysis", "new_candidate", "representative_ids", "sarle_bimodality", "state_views",
           "summarise", "update_history"]
