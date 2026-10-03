"""CV2 window design from the local free-energy shape (spec 3.2; reused by 3.3 R1/R3).

The swarm is a DESIGN MEASURE (short seeded runs from a stratified library), not an
equilibrium conditional p(CV2 | CV1). Nothing here estimates a free energy for quoting: the
mixture only locates candidate structure and bounds how finely CV2 can usefully be resolved
(a mode the swarm saw as narrow gets windows no finer than its own width). Spec 3.3 R2 is the
swarm-independent fallback when the design measure misses structure.

Pieces (pure, NumPy only):

* ``fit_cv2_mixture`` -- a 1-D Gaussian mixture of one CV1 window's CV2 samples (weights = the
  CV1 kernel of that window). BIC picks the component count; member-blocked cross-validation
  (folds never split a swarm member) picks the variance regularisation; a component is
  accepted only when at least ``min_mode_members`` independent members support it.
* ``mode_depth`` / ``mode_pair_resolvable`` -- free-energy depth (kT) between two modes of a
  mixture and R3's "depth >= 1 kT, both modes >= 10 %" test.
* ``estimate_f2`` -- F''_est = RT x precision, the component precision (1/var) shrunk toward
  the pooled region precision with weight n/(n + 8) (precision space, not the spec's variance
  space); floored at 0. Callers pass the component's ``variance_curvature`` (the fit
  re-converged at the smallest regularisation, ``curvature_variance``), never the regularised
  ``variance``: reg 3e-2 x pooled 0.55 made a var-0.041 mode read 0.070 (F'' ~40 % low).
* ``shape_rule_k2`` -- k2 = RT/sigma_w^2 - F''_est, raised to the mean-compression floor
  (k2 >= F''_est x c/(1 - c), c = ``min_mean_compression``, default 0.5 -> k2 >= F''_est),
  floored at cv2_k_min and clamped at cv2_k_max (the CV1 curvature design rule of ``ladder_design.cv1_force_constants_from_curvature``
  applied to CV2). The coupling gate (3.4) is the caller's job.
* ``place_cv2_centres`` -- a mandatory centre at every accepted mode, then steps of
  ``spacing_sigma`` (1.5) x the SMALLEST predicted sampled sigma over [z, z + delta] (never the
  value at z), inside the outer envelope (today's rung-reweighted CV2 support).

Units: CV2 in its own (standardised) units; F'' and k2 in kcal/mol per CV^2, as the deployed
k2; RT from ``ladder_design.R_KCAL_MOL_K``; depth in kT.

Mean-compression floor: the width rule alone designs the sampled WIDTH only. Where the
landscape is already as narrow as the target (F''_est >= RT/sigma_w^2; chignolin_9's swarm:
49 of 63 centres) it asks for no spring, and a window's sampled mean follows the mode, not
its centre (means compressed by k2/(k2 + F'')). The floor keeps every window's mean at least
``min_mean_compression`` of the way to its centre; the price is a narrower sampled width
there (sqrt(RT/(2 F'')) at c = 0.5), so the placement steps more finely. Per centre the
compression is reported as ``mean_compression``; ``n_at_compression_floor`` counts centres
where the floor binds and ``n_at_k_floor`` those left at cv2_k_min (nominally restrained).
``min_mean_compression = 0`` restores the plain width rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gareus.swarm.ladder_design import R_KCAL_MOL_K

DEFAULT_MIN_MODE_MEMBERS = 8
DEFAULT_PRIOR_MEMBERS = 8           # shrinkage weight n / (n + 8) toward the pooled precision
DEFAULT_SPACING_SIGMA = 1.5         # neighbour spacing in predicted sampled sigma (spec 3.2)
DEFAULT_REG_GRID = (1e-4, 1e-3, 1e-2, 3e-2)   # variance regularisation, x pooled variance
DEFAULT_MIN_MEMBER_WEIGHT = 3.0     # kernel-weighted frames a member needs to support a mode
DEFAULT_MIN_MEMBER_FRACTION = 0.2   # ...and this share of its own kernel-weighted frames
DEFAULT_MAX_FRAMES = 5000           # deterministic stride subsample (swarm frames are ps apart)
DEFAULT_MODE_MERGE_SIGMA = 0.5      # accepted modes closer than this x sampled sigma are one mode
DEFAULT_MIN_MEAN_COMPRESSION = 0.5  # k2/(k2 + F'') floor: window means move >= half-way to their centre
CURVATURE_VARIANCE_FLOOR = 1e-4     # x pooled variance: the de-regularised variance's floor (= the smallest reg)
_LOG_2PI = math.log(2.0 * math.pi)


def curvature_variance(variance: float, reg_variance: float, pooled_variance: float,
                       polished: Optional[float] = None) -> float:
    """The variance F'' (= RT/var) uses for one component: ``polished`` when given (the
    component's variance after EM re-converges from the fit at the grid's smallest
    regularisation, ``_polish_variances``), else ``variance - reg_variance``; never below
    ``CURVATURE_VARIANCE_FLOOR`` x the pooled variance (a collapsed component) nor above
    ``variance``. The EM adds ``reg_variance`` = reg x pooled variance to every component;
    that stabilises the fit but is not the mode's width. Subtracting it alone leaves part of
    the bias (the inflated width also pulled in the neighbour's tail: var 0.041, reg 3e-2 x
    pooled 0.55 -> 0.070 fitted, 0.053 subtracted, 0.043 polished)."""
    v, r = float(variance), max(0.0, float(reg_variance))
    floor = CURVATURE_VARIANCE_FLOOR * float(pooled_variance) if math.isfinite(float(pooled_variance)) else 0.0
    base = float(polished) if polished is not None and math.isfinite(float(polished)) else v - r
    return float(min(v, max(base, floor)))


@dataclass(frozen=True)
class MixtureComponent:
    """One mixture component. ``variance`` is the regularised EM variance (density, BIC,
    responsibilities, ``mode_depth``); ``variance_curvature`` = ``variance`` - ``reg_variance``
    (``curvature_variance``) is what every F'' estimate uses. Built without them (an old
    record, a hand-made component) both default to "no regularisation"."""
    mean: float
    variance: float
    weight: float
    n_members: int
    accepted: bool
    reason: str
    reg_variance: float = 0.0
    variance_curvature: Optional[float] = None

    def __post_init__(self):
        if self.variance_curvature is None:
            object.__setattr__(self, "variance_curvature", float(self.variance))

    def as_record(self) -> dict:
        return {"mean": self.mean, "variance": self.variance, "sd": math.sqrt(max(self.variance, 0.0)),
                "weight": self.weight, "n_members": int(self.n_members), "accepted": bool(self.accepted),
                "reason": self.reason, "reg_variance": self.reg_variance,
                "variance_curvature": self.variance_curvature}


@dataclass(frozen=True)
class CV2MixtureFit:
    components: Tuple[MixtureComponent, ...]
    n_components: int
    bic: Dict[int, float]
    regularisation: float
    cv_scores: Dict[float, float]
    cv_folds: int
    n_frames: int
    n_members: int
    weight_total: float
    pooled_mean: float
    pooled_variance: float
    min_mode_members: int
    reasons: Tuple[str, ...] = ()

    @property
    def accepted_components(self) -> Tuple[MixtureComponent, ...]:
        return tuple(c for c in self.components if c.accepted)

    def as_record(self) -> dict:
        return {"n_components": self.n_components, "components": [c.as_record() for c in self.components],
                "bic": {str(k): v for k, v in sorted(self.bic.items())}, "regularisation": self.regularisation,
                "cv_scores": {repr(k): v for k, v in sorted(self.cv_scores.items())}, "cv_folds": self.cv_folds,
                "n_frames": self.n_frames, "n_members": self.n_members, "weight_total": self.weight_total,
                "pooled_mean": self.pooled_mean, "pooled_variance": self.pooled_variance,
                "min_mode_members": self.min_mode_members, "reasons": list(self.reasons)}


# --- weighted 1-D Gaussian mixture ------------------------------------------------------

def _log_components(z: np.ndarray, means, variances, pis) -> np.ndarray:
    m, v, p = (np.asarray(a, dtype=float)[None, :] for a in (means, variances, pis))
    with np.errstate(divide="ignore"):
        return np.log(p) - 0.5 * (_LOG_2PI + np.log(v)) - 0.5 * (z[:, None] - m) ** 2 / v


def _logsumexp_rows(a: np.ndarray) -> np.ndarray:
    top = np.max(a, axis=1)
    return top + np.log(np.sum(np.exp(a - top[:, None]), axis=1))


def mixture_logpdf(z, components: Sequence[MixtureComponent]) -> np.ndarray:
    """log sum_k w_k N(z; mean_k, var_k) over the given components (weights used as given)."""
    x = np.atleast_1d(np.asarray(z, dtype=float))
    comps = list(components)
    return _logsumexp_rows(_log_components(x, [c.mean for c in comps], [c.variance for c in comps],
                                           [c.weight for c in comps]))


def _em(z, w, params, reg_var: float, max_iter: int, tol: float):
    """Weighted EM from ``params`` = (means, variances, pis); returns (params, weighted loglik)."""
    means, var, pis = (np.array(a, dtype=float) for a in params)
    total = float(w.sum())
    ll_prev = -np.inf
    for _ in range(int(max_iter)):
        logp = _log_components(z, means, var, pis)
        lse = _logsumexp_rows(logp)
        ll = float(np.dot(w, lse))
        wr = w[:, None] * np.exp(logp - lse[:, None])
        nk = np.maximum(wr.sum(axis=0), 1e-12 * total)
        pis = nk / nk.sum()
        means = (wr * z[:, None]).sum(axis=0) / nk
        var = (wr * (z[:, None] - means[None, :]) ** 2).sum(axis=0) / nk + reg_var
        if abs(ll - ll_prev) <= tol * max(1.0, abs(ll)):
            break
        ll_prev = ll
    return (means, var, pis), _weighted_loglik(z, w, (means, var, pis))


def _polish_variances(z, w, params, pooled_var: float, max_iter: int, tol: float) -> List[Optional[float]]:
    """Per component (same order as ``params``): the variance after EM re-converges from the
    regularised fit with regularisation CURVATURE_VARIANCE_FLOOR x pooled (then removed), or
    None where the polished component is not the same mode (its mean moved by more than one
    regularised sd, or its weight halved) -- the caller then subtracts the regularisation."""
    reg_c = CURVATURE_VARIANCE_FLOOR * float(pooled_var)
    (m2, v2, p2), _ = _em(z, w, params, reg_c, max_iter, tol)
    out: List[Optional[float]] = []
    for m, v, p, mm, vv, pp in zip(*params, m2, v2, p2):
        same = abs(float(mm) - float(m)) <= math.sqrt(max(float(v), 0.0)) and float(pp) >= 0.5 * float(p)
        out.append(float(vv) - reg_c if same and math.isfinite(float(vv)) else None)
    return out


def _weighted_loglik(z, w, params) -> float:
    return float(np.dot(w, _logsumexp_rows(_log_components(z, *params))))


def _inits(z, w, k: int, pooled_var: float, n_init: int, rng) -> List[tuple]:
    order = np.argsort(z)
    cdf = np.cumsum(w[order]) / w.sum()
    quantile_means = np.interp((np.arange(k) + 0.5) / k, cdf, z[order])
    out = [(quantile_means, np.full(k, pooled_var / k ** 2), np.full(k, 1.0 / k))]
    p = w / w.sum()
    for _ in range(max(0, int(n_init) - 1)):
        out.append((np.sort(rng.choice(z, size=k, replace=False, p=p)), np.full(k, pooled_var / k ** 2),
                    np.full(k, 1.0 / k)))
    return out


def _best_fit(z, w, inits, reg_var, max_iter, tol):
    best = None
    for init in inits:
        params, ll = _em(z, w, init, reg_var, max_iter, tol)
        if best is None or ll > best[1]:
            best = (params, ll)
    return best


def _member_folds(member_ids: np.ndarray, n_folds: int) -> List[np.ndarray]:
    """Test masks; folds never split a member (member m goes to fold rank(m) % n_folds)."""
    uniq = np.unique(member_ids)
    if uniq.size < 2:
        return []
    k = min(int(n_folds), int(uniq.size))
    fold_of = {m: i % k for i, m in enumerate(uniq.tolist())}
    labels = np.array([fold_of[m] for m in member_ids.tolist()])
    return [labels == f for f in range(k)]


def _cv_regularisation(z, w, member_ids, start, pooled_var, reg_grid, n_folds, max_iter, tol):
    """Member-blocked held-out log-likelihood per regularisation; (best reg, scores, n folds)."""
    folds = _member_folds(member_ids, n_folds)
    if not folds:
        return float(reg_grid[len(reg_grid) // 2]), {}, 0
    scores: Dict[float, float] = {}
    for reg in reg_grid:
        vals = []
        for test in folds:
            train = ~test
            if w[train].sum() <= 0 or w[test].sum() <= 0:
                continue
            params, _ = _em(z[train], w[train], start, float(reg) * pooled_var, max_iter, tol)
            vals.append(_weighted_loglik(z[test], w[test], params) / float(w[test].sum()))
        if vals:
            scores[float(reg)] = float(np.mean(vals))
    if not scores:
        return float(reg_grid[len(reg_grid) // 2]), {}, len(folds)
    return max(scores, key=scores.get), scores, len(folds)


def _member_support(z, w, member_ids, params, min_member_weight: float, min_member_fraction: float) -> List[int]:
    """Members supporting each component: responsibility-weighted frames >= ``min_member_weight``
    AND >= ``min_member_fraction`` of the member's own weighted frames (a member whose tail
    brushes a mode is not independent evidence for it)."""
    logp = _log_components(z, *params)
    resp = w[:, None] * np.exp(logp - _logsumexp_rows(logp)[:, None])
    uniq, inverse = np.unique(member_ids, return_inverse=True)
    per_member = np.zeros((uniq.size, resp.shape[1]))
    np.add.at(per_member, inverse, resp)
    own = per_member.sum(axis=1)
    ok = (per_member >= float(min_member_weight)) & (per_member >= float(min_member_fraction) * own[:, None])
    return [int(np.sum(ok[:, k])) for k in range(resp.shape[1])]


def _prepare(cv2, member_ids, weights, max_frames: int):
    z = np.asarray(cv2, dtype=float).reshape(-1)
    ids = np.asarray(member_ids).reshape(-1)
    w = np.ones_like(z) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    if not (z.shape == ids.shape == w.shape):
        raise ValueError("cv2, member_ids and weights must have the same length")
    keep = np.isfinite(z) & np.isfinite(w) & (w > 0)
    z, ids, w = z[keep], ids[keep], w[keep]
    if max_frames and z.size > int(max_frames):
        idx = np.linspace(0, z.size - 1, int(max_frames)).astype(int)   # deterministic stride
        z, ids, w = z[idx], ids[idx], w[idx]
    return z, ids, w


def _empty_fit(z, ids, w, min_mode_members, reason) -> CV2MixtureFit:
    total = float(w.sum()) if w.size else 0.0
    mean = float(np.dot(w, z) / total) if total > 0 else float("nan")
    var = float(np.dot(w, (z - mean) ** 2) / total) if total > 0 else float("nan")
    return CV2MixtureFit((), 0, {}, float("nan"), {}, 0, int(z.size), int(np.unique(ids).size), total, mean, var,
                         int(min_mode_members), (reason,))


def fit_cv2_mixture(cv2, member_ids, weights=None, *, max_components: int = 3,
                    min_mode_members: int = DEFAULT_MIN_MODE_MEMBERS,
                    min_member_weight: float = DEFAULT_MIN_MEMBER_WEIGHT,
                    min_member_fraction: float = DEFAULT_MIN_MEMBER_FRACTION,
                    reg_grid: Sequence[float] = DEFAULT_REG_GRID, n_folds: int = 4, n_init: int = 4,
                    max_iter: int = 300, tol: float = 1e-8, max_frames: int = DEFAULT_MAX_FRAMES,
                    seed: int = 0) -> CV2MixtureFit:
    """Gaussian mixture of one CV1 window's CV2 design-measure samples (see module docstring).

    ``weights`` are per-frame CV1-kernel weights (1 = every frame counts fully). For each
    component count K in 1..max_components the regularisation (``reg`` x pooled variance added
    to every component variance; recorded per component as ``reg_variance``; ``variance_curvature``,
    the variance F'' uses, re-converges the fit without it, ``curvature_variance``) is chosen
    by member-blocked cross-validation, the model is
    refitted on all frames and scored by weighted BIC (n = sum of weights, 3K - 1 parameters);
    the lowest BIC wins. A member supports a component when its responsibility-weighted frames
    sum to >= ``min_member_weight`` and to >= ``min_member_fraction`` of its own weighted
    frames; a component is accepted with >= ``min_mode_members`` supporting members.
    BIC counts frames, which are correlated within a member, so it over-splits rather than
    under-splits; the member-support rule is the guard. Deterministic for a given ``seed``.
    """
    z, ids, w = _prepare(cv2, member_ids, weights, max_frames)
    if z.size < 3 or w.sum() <= 0:
        return _empty_fit(z, ids, w, min_mode_members, "too_few_frames")
    total = float(w.sum())
    pooled_mean = float(np.dot(w, z) / total)
    pooled_var = float(np.dot(w, (z - pooled_mean) ** 2) / total)
    if not (pooled_var > 0.0 and math.isfinite(pooled_var)):
        return _empty_fit(z, ids, w, min_mode_members, "degenerate_variance")
    rng = np.random.default_rng(int(seed))
    fits: Dict[int, tuple] = {}
    for k in range(1, int(max_components) + 1):
        if k > np.unique(z).size:
            break
        start, _ = _best_fit(z, w, _inits(z, w, k, pooled_var, n_init, rng), float(reg_grid[0]) * pooled_var,
                             max_iter, tol)
        reg, scores, folds = _cv_regularisation(z, w, ids, start, pooled_var, reg_grid, n_folds, max_iter, tol)
        params, ll = _em(z, w, start, reg * pooled_var, max_iter, tol)
        fits[k] = (params, ll, reg, scores, folds, -2.0 * ll + (3 * k - 1) * math.log(total))
    best_k = min(fits, key=lambda k: fits[k][5])
    params, _ll, reg, scores, folds, _bic = fits[best_k]
    support = _member_support(z, w, ids, params, min_member_weight, min_member_fraction)
    comps = []
    reg_var = float(reg) * pooled_var
    polished = _polish_variances(z, w, params, pooled_var, max_iter, tol)
    for mean, var, pi, n_mem, pol in sorted(zip(*params, support, polished), key=lambda t: t[0]):
        ok = n_mem >= int(min_mode_members)
        comps.append(MixtureComponent(float(mean), float(var), float(pi), int(n_mem), bool(ok),
                                      "accepted" if ok else f"min_mode_members: {n_mem} < {int(min_mode_members)}",
                                      reg_var, curvature_variance(float(var), reg_var, pooled_var, pol)))
    return CV2MixtureFit(tuple(comps), int(best_k), {k: float(v[5]) for k, v in fits.items()}, float(reg),
                         dict(scores), int(folds), int(z.size), int(np.unique(ids).size), total, pooled_mean,
                         pooled_var, int(min_mode_members))


# --- mode depth (R3) --------------------------------------------------------------------

def mode_depth(components: Sequence[MixtureComponent], a: int, b: int, *, n_grid: int = 2049) -> dict:
    """Free-energy depth (kT) of the shallower of modes ``a`` and ``b`` below the barrier between them.

    F/kT = -ln p(z) of the mixture of ``components`` (weights as given). The barrier is the
    density minimum strictly between the two means; if the density has no interior minimum
    there (a shoulder, or two merged components), depth is 0 and ``bimodal`` False.
    ``weight_a``/``weight_b`` are the components' own mixture weights."""
    ca, cb = components[a], components[b]
    lo, hi = (ca, cb) if ca.mean <= cb.mean else (cb, ca)
    out = {"mean_a": ca.mean, "mean_b": cb.mean, "weight_a": ca.weight, "weight_b": cb.weight,
           "depth_kT": 0.0, "barrier_z": None, "bimodal": False}
    if hi.mean - lo.mean <= 0.0:
        return out
    pad_lo, pad_hi = 0.5 * math.sqrt(max(lo.variance, 0.0)), 0.5 * math.sqrt(max(hi.variance, 0.0))
    grid = np.linspace(lo.mean - pad_lo, hi.mean + pad_hi, int(n_grid))
    lp = mixture_logpdf(grid, components)
    between = np.where((grid >= lo.mean) & (grid <= hi.mean))[0]
    i = int(between[np.argmin(lp[between])])
    if i in (int(between[0]), int(between[-1])):
        return out
    depth = float(min(lp[: i + 1].max(), lp[i:].max()) - lp[i])
    out.update({"depth_kT": depth, "barrier_z": float(grid[i]), "bimodal": depth > 0.0})
    return out


def mode_pair_resolvable(depth: dict, *, min_depth_kT: float = 1.0, min_weight: float = 0.10) -> bool:
    """Spec 3.3 R3's mixture test: bimodal, depth >= 1 kT and both modes >= 10 %."""
    return (bool(depth.get("bimodal")) and float(depth.get("depth_kT", 0.0)) >= float(min_depth_kT)
            and min(float(depth["weight_a"]), float(depth["weight_b"])) >= float(min_weight))


# --- curvature and the shape rule -------------------------------------------------------

def estimate_f2(component_variance: float, pooled_variance: float, n_members: int, temperature_k: float, *,
                prior_members: float = DEFAULT_PRIOR_MEMBERS) -> float:
    """F''_est = RT x prec_s (kcal/mol per CV^2), prec_s = w / var_c + (1 - w) / var_pool with
    w = n/(n + prior_members): the PRECISION is shrunk toward the pooled one (user decision
    2026-09-30; the spec's variance-space form let the broad pooled variance dominate and
    widened narrow modes: 12 members, sd 0.15 -> sampled sd ~0.4-0.56). F'' is linear in the
    precision, so this is shrinkage of the curvature itself. A non-finite pooled variance
    leaves the component's own; no usable variance gives 0 (no curvature is assumed)."""
    vc, vp = float(component_variance), float(pooled_variance)
    if not (math.isfinite(vc) and vc > 0.0):
        return 0.0
    n = max(0.0, float(n_members))
    pool_ok = math.isfinite(vp) and vp > 0.0
    w = n / (n + float(prior_members)) if pool_ok else 1.0
    prec = w / vc + ((1.0 - w) / vp if pool_ok else 0.0)
    if not (math.isfinite(prec) and prec > 0.0):
        return 0.0
    return max(0.0, R_KCAL_MOL_K * float(temperature_k) * prec)


def compression_floor_k2(f2_est, min_mean_compression: float = DEFAULT_MIN_MEAN_COMPRESSION):
    """Smallest k2 with k2/(k2 + F''_est) >= c: F''_est x c/(1 - c) (0 where F''_est <= 0)."""
    c = float(min_mean_compression)
    if not 0.0 <= c < 1.0:
        raise ValueError(f"min_mean_compression must be in [0, 1), got {min_mean_compression!r}")
    f2 = np.maximum(np.nan_to_num(np.asarray(f2_est, dtype=float), nan=0.0), 0.0)
    return f2 * (c / (1.0 - c))


def shape_rule_k2(sigma_w_target: float, f2_est: float, temperature_k: float, k_min: float, k_max: float, *,
                  min_mean_compression: float = DEFAULT_MIN_MEAN_COMPRESSION) -> float:
    """k2 = max(RT/sigma_w^2 - F''_est, F''_est c/(1 - c)) (kcal/mol per CV^2), floored at
    ``k_min``, clamped at ``k_max`` (the clamp wins over the compression floor)."""
    sigma = float(sigma_w_target)
    if not (math.isfinite(sigma) and sigma > 0.0):
        raise ValueError(f"sigma_w_target must be finite and > 0, got {sigma_w_target!r}")
    f2 = float(f2_est) if math.isfinite(float(f2_est)) else 0.0
    raw = max(R_KCAL_MOL_K * float(temperature_k) / sigma ** 2 - f2,
              float(compression_floor_k2(f2, min_mean_compression)))
    return float(min(float(k_max), max(float(k_min), raw)))


def predicted_sampled_sigma(k2: float, f2_est: float, temperature_k: float) -> float:
    """sqrt(RT / (k2 + F''_est)): the sampled CV2 width of a window in a locally harmonic well."""
    total = float(k2) + max(0.0, float(f2_est))
    return math.sqrt(R_KCAL_MOL_K * float(temperature_k) / total) if total > 0.0 else math.inf


# --- placement --------------------------------------------------------------------------

@dataclass(frozen=True)
class CV2Placement:
    centres: Tuple[float, ...]
    k2: Tuple[float, ...]
    f2: Tuple[float, ...]
    sampled_sigma: Tuple[float, ...]
    mean_compression: Tuple[float, ...]
    kinds: Tuple[str, ...]                   # "mode" | "fill" | "edge" | "start"
    envelope: Tuple[float, float]
    sigma_w_target: float
    spacing_sigma: float
    n_at_k_floor: int
    dropped_modes: Tuple[dict, ...] = field(default_factory=tuple)
    n_at_compression_floor: int = 0
    min_mean_compression: float = DEFAULT_MIN_MEAN_COMPRESSION

    def as_record(self) -> dict:
        return {"centres": list(self.centres), "k2": list(self.k2), "f2": list(self.f2),
                "sampled_sigma": list(self.sampled_sigma), "mean_compression": list(self.mean_compression),
                "kinds": list(self.kinds), "envelope": list(self.envelope), "sigma_w_target": self.sigma_w_target,
                "spacing_sigma": self.spacing_sigma, "n_at_k_floor": self.n_at_k_floor,
                "dropped_modes": list(self.dropped_modes), "n_at_compression_floor": self.n_at_compression_floor,
                "min_mean_compression": self.min_mean_compression}


class _ShapeModel:
    """F''(z) from the dominant accepted component at z (pooled when none), and sigma_s(z)."""

    def __init__(self, fit: CV2MixtureFit, temperature_k, sigma_w_target, k_min, k_max, prior_members,
                 min_mean_compression=DEFAULT_MIN_MEAN_COMPRESSION):
        self.t, self.sigma_t, self.k_min, self.k_max = float(temperature_k), float(sigma_w_target), k_min, k_max
        self.min_c = float(min_mean_compression)
        self.comps = list(fit.accepted_components)
        # F'' from the de-regularised variance; the argmax dominance in f2() keeps the density's.
        self.f2_comp = np.array([estimate_f2(c.variance_curvature, fit.pooled_variance, c.n_members, temperature_k,
                                             prior_members=prior_members) for c in self.comps])
        self.f2_pool = estimate_f2(fit.pooled_variance, fit.pooled_variance, fit.n_members, temperature_k,
                                   prior_members=prior_members)

    def f2(self, z) -> np.ndarray:
        x = np.atleast_1d(np.asarray(z, dtype=float))
        if not self.comps:
            return np.full(x.shape, self.f2_pool)
        logp = _log_components(x, [c.mean for c in self.comps], [c.variance for c in self.comps],
                               [c.weight for c in self.comps])
        return self.f2_comp[np.argmax(logp, axis=1)]

    def width_k2(self, z) -> np.ndarray:
        return R_KCAL_MOL_K * self.t / self.sigma_t ** 2 - self.f2(z)

    def k2(self, z) -> np.ndarray:
        raw = np.maximum(self.width_k2(z), compression_floor_k2(self.f2(z), self.min_c))
        return np.clip(raw, self.k_min, self.k_max)

    def sigma(self, z) -> np.ndarray:
        total = self.k2(z) + np.maximum(self.f2(z), 0.0)
        with np.errstate(divide="ignore"):
            return np.where(total > 0, np.sqrt(R_KCAL_MOL_K * self.t / np.where(total > 0, total, 1.0)), np.inf)


def _step(model: _ShapeModel, z: float, direction: int, spacing: float, reach: float, n_grid: int) -> float:
    """A step delta with delta <= spacing x min sigma_s over [z, z + direction*delta].

    Iterates delta <- spacing x min sigma_s over the current step, from spacing x sigma_s(z)
    downward, and stops at the first delta that satisfies the rule. That delta is feasible
    but can undershoot the largest feasible (fixed-point) step: conservative -- a centre too
    many, never a gap too wide. (After 64 iterations the last value is returned unchecked.)"""
    d = min(spacing * float(model.sigma(z)[0]), reach)
    for _ in range(64):
        pts = z + direction * np.linspace(0.0, d, int(n_grid))
        new = min(spacing * float(np.min(model.sigma(pts))), reach)
        if new >= d * (1.0 - 1e-12):
            break
        d = new
    return d


def _walk(model, start, stop, direction, spacing, edge_fraction, n_grid, tol, to_edge, limit):
    """Centres after ``start`` towards ``stop``: fills, then (``to_edge``) the envelope edge if
    it is more than ``edge_fraction`` of a step beyond the last centre."""
    out: List[Tuple[float, str]] = []
    z = float(start)
    while True:
        d = _step(model, z, direction, spacing, abs(stop - z) + 1.0, n_grid)
        nxt = z + direction * d
        if direction * (stop - nxt) <= tol:
            if to_edge and abs(stop - z) > edge_fraction * d:
                out.append((float(stop), "edge"))
            return out
        z = nxt
        out.append((z, "fill"))
        if len(out) > limit:
            raise ValueError(f"CV2 placement exceeded {limit} centres; sigma_w_target or k2 bounds are degenerate")


def place_cv2_centres(fit: CV2MixtureFit, envelope: Tuple[float, float], *, sigma_w_target: float,
                      temperature_k: float, k_min: float, k_max: float,
                      spacing_sigma: float = DEFAULT_SPACING_SIGMA, prior_members: float = DEFAULT_PRIOR_MEMBERS,
                      edge_fraction: float = 0.5, mode_merge_sigma: float = DEFAULT_MODE_MERGE_SIGMA,
                      min_mean_compression: float = DEFAULT_MIN_MEAN_COMPRESSION,
                      n_grid: int = 33, max_centres: int = 500) -> CV2Placement:
    """CV2 centres for one CV1 window (spec 3.2): mandatory centres at accepted mode means
    inside ``envelope``, filled by greedy steps of ``spacing_sigma`` x min sigma_s over the step
    (so every adjacent gap is <= 1.5 x the smallest predicted sampled sigma between the pair).
    ``sigma_w_target`` is the base design width (the uniform layout's spacing / overlap_sigma);
    a single Gaussian whose F'' is at most RT/(2 sigma_w^2) therefore reproduces the uniform grid
    (with the default mean-compression floor; below RT/sigma_w^2 with it off).
    With no accepted mode the walk starts at the pooled mean. Accepted modes closer than
    ``mode_merge_sigma`` x their predicted sampled sigma to a heavier one (BIC splitting one
    non-Gaussian bump) are one mode (``dropped_modes``, reason "merged"). k2, F'', sigma_s and
    the mean compression k2/(k2 + F'') are reported per centre."""
    lo, hi = float(min(envelope)), float(max(envelope))
    model = _ShapeModel(fit, temperature_k, sigma_w_target, float(k_min), float(k_max), prior_members,
                        min_mean_compression)
    tol = 1e-9 * max(hi - lo, 1e-12)
    anchors, dropped = [], []
    for c in sorted(fit.accepted_components, key=lambda c: -c.weight):
        near = [a for a, _ in anchors if abs(a - c.mean) < float(mode_merge_sigma) * float(model.sigma(c.mean)[0])]
        if not lo - tol <= c.mean <= hi + tol:
            dropped.append({"mean": c.mean, "reason": "outside_envelope"})
        elif near:
            dropped.append({"mean": c.mean, "reason": f"merged: within {mode_merge_sigma} sampled sigma of {near[0]}"})
        else:
            anchors.append((float(c.mean), "mode"))
    if not anchors:
        anchors = [(float(min(max(fit.pooled_mean, lo), hi)) if math.isfinite(fit.pooled_mean) else 0.5 * (lo + hi),
                    "start")]
    anchors = sorted(dict(anchors).items())
    walk = dict(spacing=float(spacing_sigma), edge_fraction=float(edge_fraction), n_grid=n_grid, tol=tol,
                limit=int(max_centres))
    placed = list(reversed(_walk(model, anchors[0][0], lo, -1, to_edge=True, **walk)))
    for i, (a, kind) in enumerate(anchors):
        placed.append((a, kind))
        last = i + 1 == len(anchors)
        placed.extend(_walk(model, a, hi if last else anchors[i + 1][0], +1, to_edge=last, **walk))
    z = np.array([p[0] for p in placed])
    k2, f2, sig = model.k2(z), model.f2(z), model.sigma(z)
    floor = int(np.sum(k2 <= float(k_min) * (1.0 + 1e-12)))
    width = model.width_k2(z)
    c_floor = int(np.sum((width < compression_floor_k2(f2, min_mean_compression)) & (k2 > float(k_min))))
    return CV2Placement(tuple(float(x) for x in z), tuple(float(x) for x in k2), tuple(float(x) for x in f2),
                        tuple(float(x) for x in sig), tuple(float(a / (a + max(b, 0.0))) if a + max(b, 0.0) > 0
                                                            else 1.0 for a, b in zip(k2, f2)),
                        tuple(p[1] for p in placed), (lo, hi), float(sigma_w_target), float(spacing_sigma), floor,
                        tuple(dropped), c_floor, float(min_mean_compression))


# --- bridge sets (spec 2026-10-03-cv2-bridge-sets-design) --------------------------------

def fit_from_record(record: Dict[str, Any]) -> CV2MixtureFit:
    """``CV2MixtureFit`` rebuilt from its ``as_record()`` (a layout plan's ``fit``)."""
    comps = tuple(MixtureComponent(float(c["mean"]), float(c["variance"]), float(c["weight"]), int(c["n_members"]),
                                   bool(c["accepted"]), str(c.get("reason", "")), float(c.get("reg_variance", 0.0)),
                                   None if c.get("variance_curvature") is None else float(c["variance_curvature"]))
                  for c in record.get("components", []))
    return CV2MixtureFit(comps, int(record.get("n_components", len(comps))),
                         {int(k): float(v) for k, v in (record.get("bic") or {}).items()},
                         float(record.get("regularisation", 0.0)),
                         {float(k): float(v) for k, v in (record.get("cv_scores") or {}).items()},
                         int(record.get("cv_folds", 0)), int(record.get("n_frames", 0)), int(record.get("n_members", 0)),
                         float(record.get("weight_total", 0.0)), float(record.get("pooled_mean", float("nan"))),
                         float(record.get("pooled_variance", float("nan"))), int(record.get("min_mode_members", 0)),
                         tuple(record.get("reasons") or ()))


def bridge_fills(fit: CV2MixtureFit, z_lo: float, z_hi: float, *, sigma_w_target: float, temperature_k: float,
                 k_min: float, k_max: float, spacing_sigma: float = DEFAULT_SPACING_SIGMA,
                 prior_members: float = DEFAULT_PRIOR_MEMBERS,
                 min_mean_compression: float = DEFAULT_MIN_MEAN_COMPRESSION, n_grid: int = 33,
                 max_centres: int = 500) -> Dict[str, List[float]]:
    """The fills strictly between ``z_lo`` and ``z_hi`` by ``place_cv2_centres``' own step rule
    and springs (the walk between two anchors): {centres, k2, f2, sampled_sigma}."""
    lo, hi = sorted((float(z_lo), float(z_hi)))
    model = _ShapeModel(fit, temperature_k, sigma_w_target, float(k_min), float(k_max), prior_members,
                        min_mean_compression)
    tol = 1e-9 * max(hi - lo, 1e-12)
    walk = _walk(model, lo, hi, +1, float(spacing_sigma), 0.5, n_grid, tol, False, int(max_centres))
    z = np.array([w[0] for w in walk], dtype=float)
    if z.size == 0:
        return {"centres": [], "k2": [], "f2": [], "sampled_sigma": []}
    return {"centres": [float(x) for x in z], "k2": [float(x) for x in model.k2(z)],
            "f2": [float(x) for x in model.f2(z)], "sampled_sigma": [float(x) for x in model.sigma(z)]}


def _window_log_densities(fit: CV2MixtureFit, centres, k2s, temperature_k: float, n_grid: int):
    """(grid z, dz, [log p_k normalised on the grid]) for windows on the fit's profile."""
    c = np.asarray(centres, dtype=float)
    k = np.asarray(k2s, dtype=float)
    rt = R_KCAL_MOL_K * float(temperature_k)
    comps = list(fit.components)
    sds = [math.sqrt(max(cm.variance, 0.0)) for cm in comps]
    sig_k = math.sqrt(rt / max(float(np.min(k[k > 0])) if np.any(k > 0) else rt, 1e-12))
    lo = min(float(c.min()), *[cm.mean - 8.0 * sd for cm, sd in zip(comps, sds)]) - 6.0 * sig_k
    hi = max(float(c.max()), *[cm.mean + 8.0 * sd for cm, sd in zip(comps, sds)]) + 6.0 * sig_k
    z = np.linspace(lo, hi, int(n_grid))
    dz = z[1] - z[0]
    logp = mixture_logpdf(z, comps)
    out = []
    for ci, ki in zip(c, k):
        lq = logp - 0.5 * float(ki) * (z - float(ci)) ** 2 / rt
        lq = lq - lq.max()
        out.append(lq - math.log(float(np.exp(lq).sum()) * dz))
    return z, dz, out


def predicted_overlaps(fit: CV2MixtureFit, centres: Sequence[float], k2s: Sequence[float], temperature_k: float,
                       n_grid: int = 4001) -> List[float]:
    """Adjacent-window overlaps predicted on the fit's profile F(z) = -RT ln p_mix(z) (every
    component): window k samples p_k ~ exp(-(F + k2 (z - c)^2 / 2) / RT); overlap(a, b) =
    sum p_a p_b / (p_a + p_b) dz, the two-state MBAR overlap scale (0..0.5) for umbrella-only bias
    differences. An UPPER bound on what sampling realises: it is an equilibrium quantity on the
    swarm's design measure and cannot see a window trapped in one of two basins it straddles
    (``internal_barriers_kT``), nor a swarm profile that differs from production."""
    if len(centres) < 2:
        return []
    _z, dz, logs = _window_log_densities(fit, centres, k2s, temperature_k, n_grid)
    dens = [np.exp(lq) for lq in logs]
    out = []
    for pa, pb in zip(dens[:-1], dens[1:]):
        s = pa + pb
        m = s > 0
        out.append(float(np.sum(pa[m] * pb[m] / s[m]) * dz))
    return out


def internal_barriers_kT(fit: CV2MixtureFit, centres: Sequence[float], k2s: Sequence[float], temperature_k: float,
                         n_grid: int = 4001, min_peak_fraction: float = 0.01) -> List[float]:
    """Per window, the barrier (kT) inside its own predicted density p_k on the design profile: the
    lower of the two highest peaks minus the deepest point between them (0 for a unimodal window).
    Peaks under ``min_peak_fraction`` of the top peak are ignored. A window with a barrier of a few kT
    straddles two basins and samples one of them for long stretches -- overlap cannot show that."""
    if not len(centres):
        return []
    _z, _dz, logs = _window_log_densities(fit, centres, k2s, temperature_k, n_grid)
    floor = math.log(float(min_peak_fraction))
    out = []
    for lq in logs:
        top = float(lq.max())
        peaks = [i for i in range(1, lq.size - 1) if lq[i] >= lq[i - 1] and lq[i] > lq[i + 1] and lq[i] >= top + floor]
        if len(peaks) < 2:
            out.append(0.0)
            continue
        best = 0.0
        for i, j in zip(peaks[:-1], peaks[1:]):
            valley = float(lq[i:j + 1].min())
            best = max(best, min(float(lq[i]), float(lq[j])) - valley)
        out.append(best)
    return out


__all__ = ["CURVATURE_VARIANCE_FLOOR", "bridge_fills", "fit_from_record", "internal_barriers_kT",
           "predicted_overlaps", "CV2MixtureFit", "CV2Placement", "DEFAULT_MIN_MEAN_COMPRESSION", "curvature_variance", "DEFAULT_MIN_MODE_MEMBERS", "DEFAULT_PRIOR_MEMBERS",
           "DEFAULT_SPACING_SIGMA", "MixtureComponent", "estimate_f2", "fit_cv2_mixture", "mixture_logpdf", "compression_floor_k2",
           "mode_depth", "mode_pair_resolvable", "place_cv2_centres", "predicted_sampled_sigma", "shape_rule_k2"]
