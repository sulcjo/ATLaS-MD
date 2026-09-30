"""Spring re-derivation from production samples ("respring"), behind ``--ap-cv2-respring`` (off).

Why: the 3.2 shape layout sets every CV2 spring from the swarm's design-measure curvature,
k2 = max(RT/sigma^2 - F''_est, F''_est) (the mean-compression floor at c = 0.5). On chignolin_9
the production window-local F'' differs from the swarm's by a factor ~1.6 either way (t3
retrospective 10.5), so the realised mean compression k2/(k2 + F''_prod) is 0.35-0.73 instead
of 0.5. After a numbered epoch this module re-derives each CV2-restrained window's spring from
its OWN samples and, where the window is confidently under-compressed, proposes ONE atomic
``respring`` action per centre: a new centre (same c1, k1, c2 on every rung, new k2) plus the
retirement of every rung of the old one. A state_id's Hamiltonian never changes; the old
states stay in the registry (``usable_for_mbar`` is kept on retirement), so their samples stay
in the union MBAR with the params their own phases' ``epoch_window_map.csv`` recorded.

Measurement (representative rung, lambda = 0 if present; the boost reshapes the landscape at
lambda > 0, and the new k2 is replicated onto every rung as every centre is):
  * var = the P4 ``paired_cv.cv2.var`` over ALL the phase's pooled pairs (full series, not the
    subsample; population variance, CV2 marginal -- the same variance 3.3's
    ``f2_under_bias`` reads). The CV1-conditional variance var2 - cov^2/var1 is reported as
    ``f2_prod_conditional`` (t3 10.5 used it; |corr| <= 0.24 on c9, so they agree within ~11 %).
  * F''_prod = RT/var - k2 (samples taken under the window's own spring: sampled precision is
    k2 + F''), and c_real = k2/(k2 + F''_prod) = k2 var / RT, i.e. var / sigma_w^2. c_real is
    linear in var, so a var interval maps straight onto it; F''_prod <= 0 is c_real >= 1.
  * interval: block bootstrap of var on the P4 subsample, blocks of
    ceil(BOOT_BLOCK_G_MULTIPLE x g_variance / stride) subsample rows, never across a sample
    source (the minimum-block guard included: too few source-local blocks = no interval,
    skipped ``too_few_source_blocks``), each replicate redrawing every source's own blocks
    (stratified). The bootstrap's var_b/var_sub quantiles scale the full-series var
    (BOOT_QUANTILES 5-95 %). n_eff = n / g_variance must reach ``respring_min_neff``.
    g_variance (raw sample spacing, report schema v2; v1's ``g`` was the CV2 series' g) =
    max(g_cv2, g_q): g_q = the subsample's inefficiency of the variance observable
    q = (z - mean z)^2 (one mean over the sample) x stride, g_cv2 = the X5 inefficiency of the
    CV2 series (``effective_samples``; else the subsample's own x stride). A variance is a second
    moment: a slowly switching amplitude gives g_q >> g_z (synthetic: 275 vs 1.05), while for a
    Gaussian series g_q <= g_z, so the max never trusts the variance more than the series. No g_q
    (estimate failed) = skipped ``variance_inefficiency_unavailable``, never g = 1.
    Known bias (safe direction): when the P4 stride exceeds g the bootstrap sees about
    n / stride nearly independent rows, not n_eff, so the interval is too wide by up to
    sqrt(n_eff stride / n) (chignolin_9 final-combined: stride 20, median ~1.7x) and fewer
    windows trigger; point estimates are unaffected.

Decision (per candidate state, ``decision`` / ``reason``):
  * skipped -- ``not_representative_rung`` (never reported per state), ``cv2_unrestrained``
    (k2 <= max(0, cv2_k_min) or no CV2 centre: anchors and CV1-only axis states),
    ``mandatory`` (the centre holds a mandatory exploration state: it cannot be retired),
    ``already_respringed`` (a respring child is never re-sprung: one correction per lineage,
    so a noisy estimate cannot oscillate), ``protected`` (created by a 3.3 action within
    ``refine_protect_epochs``), ``touched_by_other_action`` (another action this epoch retires,
    splits or inserts at this centre -- respring never overrides another proposer; with R3 in
    its default flag mode no insert exists), ``no_samples`` / ``no_subsample`` /
    ``too_few_source_blocks`` / ``variance_inefficiency_unavailable`` (no interval or no
    variance g, never act) and ``too_few_effective_samples``.
    CV1-free windows (k1 = 0 with k2 > floor: the X7 mode-axis windows and CV2-only axis
    states) ARE candidates: their CV2 spring is what the shape rule set. Their var is the CV2
    marginal over a free CV1, so F''_prod is the marginal curvature.
  * no_action -- ``f2_nonpositive`` (sampled var >= RT/k2: flat or concave locally, c_real >= 1;
    the spring alone sets the width, nothing to correct) or ``compression_reached`` (the upper
    end of the c_real interval reaches min_mean_compression - respring_tolerance). A window
    whose whole interval lies above OVER_STIFF_COMPRESSION is flagged ``over_compressed`` in
    its metrics (stiffer than needed, narrower sampling, less overlap) but never acted on:
    softening a window moves its mean away from its centre, which the layout did not ask for.
  * triggered (the WHOLE c_real interval below min_mean_compression - respring_tolerance):
    k2' = ``shape_rule_k2`` (sigma_target, F''_prod point, T, cv2_k_min, cv2_k_max), the width
    rule raised to the compression floor. sigma_target = the layout's
    ``cv2_shape.sigma_w_target`` from ``layout_plan.json`` when recorded, else the window's own
    sampled sd. With the sampled sd the width rule returns k2 (RT/var - F'' = k2), so k2' =
    max(k2, F''_prod) = F''_prod for every triggered window (predicted c = 0.5 exactly); a
    uniform layout (c9) gives the same. The prediction assumes F'' does not change as the window
    narrows (locally harmonic). Then the 3.4 coupling gate when configured. Refused:
    ``k2_capped_below_compression`` (cv2_k_max or the gate holds k2' below the floor),
    ``below_k_min`` (gate), ``no_change`` (|k2' - k2| <= respring_k2_rtol x k2).
  * deferred (refusal ``per_epoch_cap``) -- triggered windows past the per-epoch cap
    max(1, floor(respring_max_fraction x centres)); the most under-compressed (lowest c_real
    upper bound) go first. They are re-measured next epoch.
``summary.n_unresolved`` = triggered windows not proposed (refused or deferred), plus the
applier's refusals after the apply; 3.7 grades it CAUTION. Respring never blocks the
convergence gate: it corrects an existing window's spring, it adds no resolution, and every
remaining epoch and the final phase sample the new window regardless.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from gareus.adaptive.cv2_shape import (DEFAULT_MIN_MEAN_COMPRESSION, compression_floor_k2,
                                       predicted_sampled_sigma, shape_rule_k2)
from gareus.adaptive.effective_samples import contiguous_runs, pooled_inefficiency
from gareus.swarm.ladder_design import R_KCAL_MOL_K

SCHEMA_VERSION = "cv2_respring_report_v2"
REPORT_NAME = "cv2_respring_report.json"
ACTION = "respring"
RULE = "respring"
METADATA_KEY = "cv2_resolution"          # shared with 3.3: seeding + protection read this key
SOURCE = "adaptive_production_cv2_respring"
REASON_PREFIX = "cv2_respring"
BURN_IN_NOTE = ("standard: the US pull is not written as samples, per-state burnin_steps 0, "
                "union MBAR per-state equilibration (t0) detection")
BOOT_REPS = 200
BOOT_QUANTILES = (0.05, 0.95)
BOOT_BLOCK_G_MULTIPLE = 5.0
BOOT_MIN_BLOCKS = 5
OVER_STIFF_COMPRESSION = 0.9
DEFAULTS = {"respring_min_neff": 200.0, "respring_tolerance": 0.05, "respring_max_fraction": 0.25,
            "respring_k2_rtol": 0.10}
DECISIONS = ("proposed", "no_action", "skipped", "refused", "deferred")


@dataclass(frozen=True)
class RespringSettings:
    respring_min_neff: float = DEFAULTS["respring_min_neff"]
    respring_tolerance: float = DEFAULTS["respring_tolerance"]
    respring_max_fraction: float = DEFAULTS["respring_max_fraction"]
    respring_k2_rtol: float = DEFAULTS["respring_k2_rtol"]
    refine_protect_epochs: int = 2
    min_mean_compression: float = DEFAULT_MIN_MEAN_COMPRESSION
    temperature_k: float = 300.0
    k2_min: float = 0.0
    k2_max: Optional[float] = None
    sigma_target: Optional[float] = None
    sigma_target_source: str = "sampled_sd"

    def __post_init__(self) -> None:
        validate_knobs(self.respring_min_neff, self.respring_tolerance, self.respring_max_fraction,
                       self.respring_k2_rtol)
        if not 0.0 < float(self.min_mean_compression) < 1.0:
            raise ValueError(f"min_mean_compression must be in (0, 1), got {self.min_mean_compression!r}")

    @classmethod
    def from_policy(cls, policy: Any, **extra: Any) -> "RespringSettings":
        vals = {k: getattr(policy, k, v) for k, v in DEFAULTS.items()}
        vals["refine_protect_epochs"] = int(getattr(policy, "refine_protect_epochs", 2))
        return cls(**vals, **extra)

    @property
    def rt(self) -> float:
        return R_KCAL_MOL_K * float(self.temperature_k)

    @property
    def trigger_below(self) -> float:
        return float(self.min_mean_compression) - float(self.respring_tolerance)

    def as_record(self) -> Dict[str, Any]:
        rec = {k: getattr(self, k) for k in self.__dataclass_fields__}
        rec.update(trigger_below=self.trigger_below, boot_reps=BOOT_REPS, boot_quantiles=list(BOOT_QUANTILES),
                   boot_block_g_multiple=BOOT_BLOCK_G_MULTIPLE, boot_min_blocks=BOOT_MIN_BLOCKS,
                   over_stiff_compression=OVER_STIFF_COMPRESSION,
                   uncalibrated=["respring_min_neff", "respring_tolerance", "respring_max_fraction",
                                 "respring_k2_rtol"])
        return rec


def validate_knobs(min_neff: Any, tolerance: Any, max_fraction: Any, k2_rtol: Any) -> None:
    """Range checks shared with ``AdaptiveDecisionPolicy.__post_init__`` (fails at policy build)."""
    checks = (("respring_min_neff", min_neff, lambda v: v >= 0.0, ">= 0"),
              ("respring_tolerance", tolerance, lambda v: 0.0 <= v < 0.5, "in [0, 0.5)"),
              ("respring_max_fraction", max_fraction, lambda v: 0.0 <= v <= 1.0, "in [0, 1]"),
              ("respring_k2_rtol", k2_rtol, lambda v: 0.0 < v < 1.0, "in (0, 1)"))
    for name, value, ok, text in checks:
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number {text}, got {value!r}") from None
        if not (math.isfinite(v) and ok(v)):
            raise ValueError(f"{name} must be {text}, got {value!r}")


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ---- measurement -------------------------------------------------------------------------

def compression_from_var(var: float, k2: float, rt: float) -> float:
    """c_real = k2/(k2 + F''_prod) with F''_prod = RT/var - k2, i.e. k2 var / RT."""
    return float(k2) * float(var) / float(rt)


def f2_from_var(var: float, k2: float, rt: float) -> float:
    """F''_prod = RT/var - k2 (samples taken under spring k2)."""
    return float(rt) / float(var) - float(k2)


def block_ids(n: int, source_index: Optional[np.ndarray], block_len: int) -> np.ndarray:
    """Contiguous blocks of ``block_len`` rows, restarting at every sample source."""
    src = np.zeros(int(n), dtype=np.int64) if source_index is None else np.asarray(source_index, dtype=np.int64)
    pos = np.zeros(int(n), dtype=np.int64)
    for s in np.unique(src):
        idx = np.flatnonzero(src == s)
        pos[idx] = np.arange(idx.size)
    _u, ids = np.unique(np.stack([src, pos // max(1, int(block_len))], axis=1), axis=0, return_inverse=True)
    return np.asarray(ids, dtype=np.int64).ravel()


def _series_inefficiency(x: np.ndarray, source_index: Optional[np.ndarray],
                         step: Optional[np.ndarray]) -> Tuple[Optional[float], str]:
    """(g in subsample spacing, status) of one observable; runs never cross a source or a step gap.

    ``status`` is the estimator's (ok | no_samples | too_short | zero_variance | failed); g is
    None unless ok -- a failed estimate is never read as g = 1.
    """
    x = np.asarray(x, dtype=float)
    src = np.zeros(x.size, dtype=np.int64) if source_index is None else np.asarray(source_index, dtype=np.int64)
    runs = []
    for s in np.unique(src):
        idx = np.flatnonzero(src == s)
        if step is not None and np.all(np.asarray(step)[idx] >= 0):
            runs.extend(contiguous_runs(np.asarray(step, dtype=np.int64)[idx], x[idx]))
        else:
            runs.append(x[idx])
    est = pooled_inefficiency(runs, min_segment=10)
    return (float(est.g), "ok") if est.status == "ok" else (None, str(est.status))


def subsample_inefficiency(z: np.ndarray, source_index: Optional[np.ndarray],
                           step: Optional[np.ndarray]) -> Optional[float]:
    """g of the CV2 subsample itself (subsample spacing); None on failure."""
    return _series_inefficiency(z, source_index, step)[0]


def variance_observable(z: np.ndarray) -> np.ndarray:
    """q = (z - mean(z))^2 with ONE mean over the measured sample (population-variance summand).

    One fixed mean keeps between-source mean differences inside the variance, as ``np.var(z)``
    does; centring per source would measure a different (within-source) variance.
    """
    z0 = np.asarray(z, dtype=float)
    return (z0 - float(np.mean(z0))) ** 2


def variance_inefficiency(z: np.ndarray, source_index: Optional[np.ndarray],
                          step: Optional[np.ndarray]) -> Tuple[Optional[float], str]:
    """g (subsample spacing) of the variance observable q; the variance's own correlation time.

    For a Gaussian series rho_q(t) = rho_z(t)^2, so g_q <= g_z; a slowly switching amplitude
    (two CV2 modes of different width, rare hops) gives g_q >> g_z.
    """
    return _series_inefficiency(variance_observable(z), source_index, step)


def bootstrap_var_ratio(z: np.ndarray, source_index: Optional[np.ndarray], block_len: int, *,
                        reps: int = BOOT_REPS, seed: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Source-stratified block-bootstrap distribution of var_b / var(z) (population variance).

    Blocks never cross a sample source, the minimum-block guard included: with fewer than
    BOOT_MIN_BLOCKS blocks at ``block_len`` the length drops to floor(n / BOOT_MIN_BLOCKS)
    (still source-local; ``source_boundary_guard`` recorded); still fewer = no distribution,
    ``bootstrap_status`` ``too_few_source_blocks``. Each replicate redraws every source's own
    blocks with replacement, as many as that source has, so each source keeps its share.
    """
    z = np.asarray(z, dtype=float)
    src = np.zeros(z.size, dtype=np.int64) if source_index is None else np.asarray(source_index, dtype=np.int64)
    info: Dict[str, Any] = {"block_len_requested": int(block_len), "source_boundary_guard": False,
                            "n_sources": int(np.unique(src).size)}
    ids = block_ids(z.size, src, block_len)
    n_blocks = int(ids.max()) + 1 if ids.size else 0
    if n_blocks < BOOT_MIN_BLOCKS:
        block_len = max(1, z.size // BOOT_MIN_BLOCKS)
        ids = block_ids(z.size, src, block_len)
        n_blocks = int(ids.max()) + 1 if ids.size else 0
        info["source_boundary_guard"] = True
    info.update(block_len=int(block_len), n_blocks=n_blocks)
    v0 = float(np.var(z))
    if n_blocks < BOOT_MIN_BLOCKS:
        info["bootstrap_status"] = "too_few_source_blocks"
        return np.array([]), info
    if not v0 > 0:
        info["bootstrap_status"] = "zero_variance"
        return np.array([]), info
    members = [np.flatnonzero(ids == b) for b in range(n_blocks)]
    block_src = np.array([int(src[m[0]]) for m in members])
    strata = [np.flatnonzero(block_src == s) for s in np.unique(block_src)]
    rng = np.random.default_rng(int(seed))
    out = np.empty(int(reps))
    for r in range(int(reps)):
        pick = np.concatenate([rng.choice(bl, size=bl.size, replace=True) for bl in strata])
        out[r] = float(np.var(z[np.concatenate([members[b] for b in pick])])) / v0
    info["bootstrap_status"] = "ok"
    return out, info


def _combine_g(g_cv2: Optional[float], g_q: Optional[float]) -> Tuple[Optional[float], str]:
    """g_variance = max(g_cv2, g_q) (raw spacing): never less conservative than the CV2 series' g.

    Needs g_q; g_cv2 alone is not an estimate of the variance's correlation time.
    """
    if g_q is None:
        return None, "none"
    if g_cv2 is not None and g_cv2 > g_q:
        return float(g_cv2), "cv2"
    return float(g_q), "variance"


def measure_window(*, var: float, n: int, k2: float, rt: float, z: Optional[np.ndarray],
                   source_index: Optional[np.ndarray] = None, step: Optional[np.ndarray] = None,
                   stride: int = 1, g: Optional[float] = None, g_source: str = "x5", seed: int = 0,
                   cond_var: Optional[float] = None) -> Dict[str, Any]:
    """F''_prod and c_real of one window, with block-bootstrap intervals (see module docstring).

    ``g`` is the CV2 series' inefficiency in raw sample spacing (X5), audit and lower bound only;
    the decision uses ``g_variance`` = max(g_cv2, g_q) with g_q from the variance observable.
    """
    stride = max(1, int(stride or 1))
    m: Dict[str, Any] = {"n": int(n), "var": float(var), "sd": math.sqrt(var), "sigma_w2": math.sqrt(rt / k2),
                         "subsample_stride": stride, "f2_prod": f2_from_var(var, k2, rt),
                         "c_real": compression_from_var(var, k2, rt)}
    m["f2_prod_conditional"] = f2_from_var(cond_var, k2, rt) if cond_var and cond_var > 0 else None
    empty = dict(g_variance=None, g_variance_status="no_subsample", g_variance_source="none",
                 n_eff=None, n_eff_variance=None, c_interval=None, f2_interval=None, bootstrap=None,
                 bootstrap_status="no_subsample")
    if z is None or np.asarray(z).size < 2 * BOOT_MIN_BLOCKS:
        m.update(g_cv2=g, g_cv2_source=g_source, g_q=None, **empty)
        return m
    z = np.asarray(z, dtype=float)
    if g is None or not (g > 0):
        g_sub = subsample_inefficiency(z, source_index, step)
        g, g_source = (None, "none") if g_sub is None else (g_sub * stride, "subsample")
    gq_sub, gq_status = variance_inefficiency(z, source_index, step)
    g_q = None if gq_sub is None else gq_sub * stride
    g_var, g_var_source = _combine_g(g, g_q)
    m.update(g_cv2=g, g_cv2_source=g_source, g_q=g_q, g_q_status=gq_status)
    if g_var is None:
        m.update({**empty, "g_variance_status": f"variance_inefficiency_{gq_status}",
                  "bootstrap_status": "not_run"})
        return m
    n_eff = float(n) / max(1.0, g_var)
    block = int(math.ceil(BOOT_BLOCK_G_MULTIPLE * g_var / stride))
    ratio, info = bootstrap_var_ratio(z, source_index, block, seed=seed)
    m.update(g_variance=g_var, g_variance_status="ok", g_variance_source=g_var_source, n_eff=n_eff,
             n_eff_variance=n_eff, bootstrap=info, bootstrap_status=info["bootstrap_status"])
    if ratio.size == 0:
        m.update(c_interval=None, f2_interval=None)
        return m
    lo, hi = (float(np.quantile(ratio, q)) for q in BOOT_QUANTILES)
    v_lo, v_hi = var * lo, var * hi
    m["var_interval"] = [v_lo, v_hi]
    m["c_interval"] = [compression_from_var(v_lo, k2, rt), compression_from_var(v_hi, k2, rt)]
    m["f2_interval"] = [f2_from_var(v_hi, k2, rt), f2_from_var(v_lo, k2, rt)]
    m["over_compressed"] = bool(m["c_interval"][0] > OVER_STIFF_COMPRESSION)
    return m


# ---- springs -----------------------------------------------------------------------------

def plan_spring(m: Mapping[str, Any], k2: float, settings: RespringSettings) -> Dict[str, Any]:
    """k2' from the shape rule on F''_prod (point), with cap/floor refusals (no gate here)."""
    f2 = max(0.0, float(m["f2_prod"]))
    sigma = settings.sigma_target if settings.sigma_target else float(m["sd"])
    src = settings.sigma_target_source if settings.sigma_target else "sampled_sd"
    k_min = max(0.0, float(settings.k2_min))
    k_max = float(settings.k2_max) if settings.k2_max is not None and float(settings.k2_max) > 0 else math.inf
    k2_new = shape_rule_k2(sigma, f2, settings.temperature_k, k_min, max(k_min, k_max),
                           min_mean_compression=settings.min_mean_compression)
    floor = float(compression_floor_k2(f2, settings.min_mean_compression))
    return _spring_record(k2_new, k2, f2, m, settings, sigma_target=float(sigma), sigma_target_source=src,
                          k2_cap=None if math.isinf(k_max) else k_max, compression_floor_k2=floor)


def _spring_record(k2_new: float, k2: float, f2: float, m: Mapping[str, Any], settings: RespringSettings,
                   **extra: Any) -> Dict[str, Any]:
    f2_iv = m.get("f2_interval")
    pred_iv = None
    if f2_iv:
        pred_iv = sorted(k2_new / (k2_new + max(0.0, float(x))) for x in f2_iv)
    rec = {"k2_old": float(k2), "k2_new": float(k2_new), "k2_ratio": float(k2_new) / float(k2),
           "predicted_c": float(k2_new) / (float(k2_new) + f2) if k2_new + f2 > 0 else 1.0,
           "predicted_c_interval": pred_iv,
           "predicted_sampled_sigma": predicted_sampled_sigma(k2_new, f2, settings.temperature_k),
           "min_mean_compression": float(settings.min_mean_compression), "refusal": None, **extra}
    floor = float(rec.get("compression_floor_k2") or 0.0)
    if k2_new < floor * (1.0 - 1e-9):
        rec["refusal"] = "k2_capped_below_compression"
    elif abs(k2_new - float(k2)) <= float(settings.respring_k2_rtol) * float(k2):
        rec["refusal"] = "no_change"
    return rec


def gate_spring(rec: Dict[str, Any], c1: float, k1: float, c2: float, gate: Any, f2: float,
                settings: RespringSettings) -> Dict[str, Any]:
    """The 3.4 coupling gate (None = off) on a planned respring; re-checks the refusals."""
    if gate is None or rec.get("refusal"):
        return rec
    k2g, note = gate.gate(c1, k1, c2, rec["k2_new"], context=f"respring at ({c1:.6g}, {c2:.6g})")
    if k2g is None:
        return {**rec, "refusal": "below_k_min", "gate_note": note}
    if float(k2g) == float(rec["k2_new"]):
        return {**rec, "gate_note": note}
    return {**_spring_record(float(k2g), rec["k2_old"], f2, {"f2_interval": None}, settings,
                             **{k: rec[k] for k in ("sigma_target", "sigma_target_source", "k2_cap",
                                                    "compression_floor_k2")}),
            "gate_note": note, "k2_before_gate": rec["k2_new"]}


# ---- candidates and actions --------------------------------------------------------------

def _candidate(view: Mapping[str, Any], decision: str, reason: str, **extra: Any) -> Dict[str, Any]:
    return {"state_id": int(view["state_id"]), "centre_state_ids": list(view.get("members", [])),
            "c1": view["c1"], "k1": view["k1"], "c2": view["c2"], "k2": view["k2"], "lambda": view["lam"],
            "decision": decision, "reason": reason, "refusal": extra.pop("refusal", None),
            "triggered": bool(extra.pop("triggered", False)), "metrics": extra.pop("metrics", {}),
            "proposal": extra.pop("proposal", None), **extra}


def structural_skip(view: Mapping[str, Any], settings: RespringSettings, epoch: int,
                    touched: Iterable[int]) -> Optional[str]:
    """Why a state is not a candidate before any measurement (None = candidate)."""
    k2 = _finite(view.get("k2"))
    if view.get("c2") is None or k2 is None or k2 <= max(0.0, float(settings.k2_min)):
        return "cv2_unrestrained"
    if view.get("mandatory"):
        return "mandatory"
    meta = (view.get("metadata") or {}).get(METADATA_KEY)
    if isinstance(meta, Mapping) and meta.get("rule") == RULE:
        return "already_respringed"
    if meta is not None and int(epoch) < int(view.get("created_epoch", 0)) + int(settings.refine_protect_epochs):
        return "protected"
    if set(int(s) for s in view.get("members", [])) & set(int(t) for t in touched):
        return "touched_by_other_action"
    return None


def evaluate(view: Mapping[str, Any], m: Optional[Mapping[str, Any]], settings: RespringSettings, *,
             epoch: int, touched: Iterable[int], gate: Any = None) -> Dict[str, Any]:
    """One candidate record (decision before the per-epoch cap)."""
    skip = structural_skip(view, settings, epoch, touched)
    if skip is not None:
        return _candidate(view, "skipped", skip)
    if m is None:
        return _candidate(view, "skipped", "no_samples")
    if str(m.get("g_variance_status") or "").startswith("variance_inefficiency_"):
        return _candidate(view, "skipped", "variance_inefficiency_unavailable", metrics=dict(m))
    if m.get("c_interval") is None:
        reason = ("too_few_source_blocks" if m.get("bootstrap_status") == "too_few_source_blocks"
                  else "no_subsample")
        return _candidate(view, "skipped", reason, metrics=dict(m))
    if m.get("n_eff") is None or float(m["n_eff"]) < float(settings.respring_min_neff):
        return _candidate(view, "skipped", "too_few_effective_samples", metrics=dict(m))
    if float(m["f2_prod"]) <= 0.0:
        return _candidate(view, "no_action", "f2_nonpositive", metrics=dict(m))
    if float(m["c_interval"][1]) >= settings.trigger_below:
        return _candidate(view, "no_action", "compression_reached", metrics=dict(m))
    rec = plan_spring(m, float(view["k2"]), settings)
    rec = gate_spring(rec, float(view["c1"]), float(view["k1"] or 0.0), float(view["c2"]), gate,
                      max(0.0, float(m["f2_prod"])), settings)
    if rec.get("refusal"):
        return _candidate(view, "refused", f"{rec['refusal']}: c_real {m['c_real']:.3f} "
                          f"[{m['c_interval'][0]:.3f}, {m['c_interval'][1]:.3f}]", refusal=rec["refusal"],
                          triggered=True, metrics=dict(m), proposal=rec)
    return _candidate(view, "proposed", f"under-compressed: c_real {m['c_real']:.3f} "
                      f"[{m['c_interval'][0]:.3f}, {m['c_interval'][1]:.3f}] < {settings.trigger_below:.3f}; "
                      f"k2 {float(view['k2']):.4g} -> {rec['k2_new']:.4g}", triggered=True, metrics=dict(m),
                      proposal=rec)


def apply_cap(cands: Sequence[Dict[str, Any]], n_centres: int, settings: RespringSettings
              ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Keep the most under-compressed proposals up to the per-epoch cap; defer the rest."""
    frac = float(settings.respring_max_fraction)
    limit = 0 if frac <= 0 else max(1, int(math.floor(frac * int(n_centres))))
    props = sorted([c for c in cands if c["decision"] == "proposed"],
                   key=lambda c: (c["metrics"]["c_interval"][1], c["state_id"]))
    keep = {c["state_id"] for c in props[:limit]}
    out = []
    for c in cands:
        if c["decision"] == "proposed" and c["state_id"] not in keep:
            c = {**c, "decision": "deferred", "refusal": "per_epoch_cap",
                 "reason": f"per_epoch_cap ({limit} of {n_centres} centres): {c['reason']}"}
        out.append(c)
    return out, {"max_fraction": frac, "n_centres": int(n_centres), "limit": limit,
                 "n_over_cap": max(0, len(props) - limit)}


def action_of(cand: Mapping[str, Any], epoch: int) -> Tuple:
    """("respring", old_state_id, (c1, k1, c2, k2'), reason, metadata)."""
    p = cand["proposal"]
    meta = {METADATA_KEY: {"rule": RULE, "class": None, "proposed_epoch": int(epoch),
                           "seed_source_state_id": int(cand["state_id"]), "burn_in": BURN_IN_NOTE,
                           "respring": {"k2_old": p["k2_old"], "f2_prod": cand["metrics"]["f2_prod"],
                                        "c_real": cand["metrics"]["c_real"], "predicted_c": p["predicted_c"]}}}
    return (ACTION, int(cand["state_id"]),
            (float(cand["c1"]), float(cand["k1"] or 0.0), float(cand["c2"]), float(p["k2_new"])),
            f"{REASON_PREFIX}: {cand['reason']}", meta)


def summarise(cands: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_dec: Dict[str, int] = {}
    by_reason: Dict[str, int] = {}
    for c in cands:
        by_dec[c["decision"]] = by_dec.get(c["decision"], 0) + 1
        key = c.get("refusal") or str(c["reason"]).split(":")[0]
        by_reason[key] = by_reason.get(key, 0) + 1
    meas = [c for c in cands if (c.get("metrics") or {}).get("c_interval") is not None]
    cr = np.array([c["metrics"]["c_real"] for c in meas], dtype=float)
    trig = [c for c in cands if c.get("triggered")]
    return {"n_candidates": len(cands), "n_measured": len(meas), "n_triggered": len(trig),
            "n_proposed": by_dec.get("proposed", 0),
            "n_unresolved": sum(1 for c in trig if c["decision"] != "proposed"),
            "n_over_compressed": sum(1 for c in meas if c["metrics"].get("over_compressed")),
            "decisions": by_dec, "reasons": by_reason,
            "c_real_quantiles": None if not cr.size else
            {q: float(np.quantile(cr, float(q))) for q in ("0.1", "0.5", "0.9")}}


__all__ = ["ACTION", "BOOT_QUANTILES", "DEFAULTS", "METADATA_KEY", "REPORT_NAME", "RULE", "SCHEMA_VERSION",
           "RespringSettings", "action_of", "apply_cap", "block_ids", "bootstrap_var_ratio",
           "compression_from_var", "evaluate", "f2_from_var", "gate_spring", "measure_window", "plan_spring",
           "structural_skip", "subsample_inefficiency", "summarise", "validate_knobs",
           "variance_inefficiency", "variance_observable"]
