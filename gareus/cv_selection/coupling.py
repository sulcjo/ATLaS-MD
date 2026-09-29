"""Coupling gate: how much a residual CV2 umbrella stiffens CV1 in one window (spec 3.4).

A residual coordinate depends on the anchor c (the contact CV1) through its regression rows,

    z = (v.f - K0 - K1 t - K2 t^2) / sigma_j,     t = T((c - mu_c) / sigma_c),

so the CV2 umbrella U2 = k2 (z - z0)^2 / 2 also curves CV1:

    d2U2/dc2 = k2 [ (dz/dc)^2 + (z - z0) d2z/dc2 ],
    dz/dc    = -(K1 + 2 K2 t) / (sigma_j sigma_c),   d2z/dc2 = -2 K2 / (sigma_j sigma_c^2).

``cv2_coupling_fraction`` is the maximum of that curvature over the window's box
(c in c1 +/- 2 sigma_w1, |z - z0| <= 2 sigma_w2) divided by the designed CV1 curvature
RT / sigma_w1^2 = k1 + F''_1. (K1 + 2 K2 t)^2 is a convex quadratic in t, so its box maximum
sits at an end of the (clipped) t interval; the second term does not depend on t and peaks
at 2 sigma_w2 |d2z/dc2|. No grid is needed. The coefficients come from
:func:`~gareus.cv_selection.residual_runtime.compile_component`, the one compiled definition
the force and every evaluator use (the same regression rows as
:func:`~gareus.cv_selection.models.coupling_curvature_kcal`, which stays the selection
certificate's number at a = 0).

Anchor values outside the fit's training clamp are evaluated AT the clamp (the one-sided
derivative, not the zero a hard clip gives beyond it) and the result records that it was
clamped. sigma_w2 = sqrt(RT / k2), so the gated curvature is k2 A + 2 B sqrt(RT k2): monotone
in k2, and the largest passing k2 solves a quadratic in sqrt(k2) (:func:`largest_passing_k2`).

States with k1 = 0 (CV1 unrestrained) have no designed CV1 curvature. For them the result
bounds the predicted CV1 mean shift under the CV2 tilt k2 (z - z0) dz/dc (linear response,
shift = tilt sigma_ref^2 / RT, reported in units of sigma_ref) and the width change, with
sigma_ref the sampled/pooled CV1 sd the caller supplies; without it the result is NA.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union

from .residual_runtime import compile_component

LOG = logging.getLogger(__name__)

#: kcal/mol/K (same constant as gareus.swarm.ladder_design).
R_KCAL_MOL_K = 1.987204e-3
#: Spec 3.4: the CV2 umbrella may add at most this fraction of the designed CV1 curvature.
MAX_COUPLING_FRACTION = 0.25
#: The window box half-widths, in sigma_w.
BOX_SIGMA = 2.0

STATUS_OK = "ok"
STATUS_NA = "NA"
MODE_K1 = "k1_restrained"
MODE_K1_ZERO = "k1_unrestrained"


@dataclass(frozen=True)
class CouplingWindow:
    """One window's restraint, as the gate sees it (CV1 in the anchor's own units).

    ``cv1_curvature_kcal`` is F''_1,est at the centre (0 when unknown: the adaptive side has
    no estimate, and ignoring a positive F'' only makes the gate stricter). ``sigma_w1``
    overrides the designed width; for a k1 = 0 window it is the sampled/pooled CV1 sd and is
    required.
    """

    primary_center: float
    primary_k: float
    secondary_center: float = 0.0
    temperature_k: float = 300.0
    cv1_curvature_kcal: float = 0.0
    sigma_w1: Optional[float] = None

    @classmethod
    def coerce(cls, window: Union["CouplingWindow", Mapping[str, Any]]) -> "CouplingWindow":
        if isinstance(window, cls):
            return window
        w = dict(window)
        return cls(primary_center=float(w["primary_center"]), primary_k=float(w.get("primary_k") or 0.0),
                   secondary_center=float(w.get("secondary_center") or 0.0),
                   temperature_k=float(w.get("temperature_k", 300.0)),
                   cv1_curvature_kcal=float(w.get("cv1_curvature_kcal") or 0.0),
                   sigma_w1=None if w.get("sigma_w1") is None else float(w["sigma_w1"]))


def _terms(comp, window: CouplingWindow, sigma_w1: float) -> Dict[str, Any]:
    """Box terms that do not depend on k2: A = max (dz/dc)^2, B = |d2z/dc2|, clamp record."""
    a_lo = (window.primary_center - BOX_SIGMA * sigma_w1 - comp.anchor_mean) / comp.anchor_std
    a_hi = (window.primary_center + BOX_SIGMA * sigma_w1 - comp.anchor_mean) / comp.anchor_std
    lo, hi = float(comp.clamp_lo), float(comp.clamp_hi)
    clamped = bool(a_lo < lo or a_hi > hi)
    t_ends = (min(max(a_lo, lo), hi), min(max(a_hi, lo), hi))
    scale = comp.sigma_j * comp.anchor_std
    dz = [-(comp.k1 + 2.0 * comp.k2 * t) / scale for t in t_ends]
    d2 = -2.0 * comp.k2 / (comp.sigma_j * comp.anchor_std ** 2)
    if clamped:
        LOG.info("cv2 coupling: window c1=%.6g box a_std [%.4g, %.4g] leaves the training clamp "
                 "[%.4g, %.4g]; evaluated at the clamp", window.primary_center, a_lo, a_hi, lo, hi)
    return {"A": max(d * d for d in dz), "B": abs(d2), "dz_dc_max": max(abs(d) for d in dz),
            "d2z_dc2": d2, "anchor_box_std": [a_lo, a_hi], "anchor_clamp": [lo, hi],
            "anchor_clamped": clamped}


def _sigma_w1(window: CouplingWindow, rt: float) -> tuple:
    """(sigma_w1, designed CV1 curvature, note) or (None, None, reason)."""
    if window.sigma_w1 is not None:
        s = float(window.sigma_w1)
        if not (math.isfinite(s) and s > 0.0):
            return None, None, f"sigma_w1 {s!r} is not a positive width"
        return s, rt / s ** 2, "given"
    k1 = float(window.primary_k)
    if not (math.isfinite(k1) and k1 > 0.0):
        return None, None, "k1 = 0 window without a sampled CV1 sd"
    curv = k1 + float(window.cv1_curvature_kcal or 0.0)
    note = "k1 + F''"
    if not (math.isfinite(curv) and curv > 0.0):
        LOG.warning("cv2 coupling: k1 + F'' = %.4g <= 0 at c1=%.6g; using k1 alone", curv, window.primary_center)
        curv, note = k1, "k1 (k1 + F'' <= 0)"
    return math.sqrt(rt / curv), curv, note


def _evaluate(terms: Dict[str, Any], k2: float, rt: float, sigma_w1: float, designed: float,
              k1_zero: bool) -> Dict[str, Any]:
    sigma_w2 = math.sqrt(rt / k2)
    curvature = k2 * (terms["A"] + BOX_SIGMA * sigma_w2 * terms["B"])
    ratio = curvature / designed
    out = {"curvature_kcal": curvature, "sigma_w2": sigma_w2, "curvature_ratio": ratio}
    if k1_zero:
        tilt = k2 * BOX_SIGMA * sigma_w2 * terms["dz_dc_max"]
        shift = tilt * sigma_w1 ** 2 / rt
        out.update({"tilt_kcal_per_cv": tilt, "mean_shift_cv": shift, "mean_shift_sigma": shift / sigma_w1,
                    "width_change_fraction": 1.0 - 1.0 / math.sqrt(1.0 + max(ratio, 0.0))})
        out["fraction"] = max(ratio, shift / sigma_w1)
    else:
        out["fraction"] = ratio
    return out


def cv2_coupling_fraction(fit, j: int, k2: float, window) -> Dict[str, Any]:
    """Maximum CV2-induced CV1 curvature over the window box, over the designed CV1 curvature.

    ``fit`` is a :class:`~gareus.cv_selection.models.ResidualFit`, ``j`` the 1-based component,
    ``k2`` in kcal/mol per z^2, ``window`` a :class:`CouplingWindow` or mapping with its fields.
    Returns a JSON-ready record; ``status`` is ``"NA"`` (with ``reason``) when the window has no
    usable CV1 width, else ``"ok"`` with ``fraction``. For k1 = 0 windows ``fraction`` is the
    larger of the curvature ratio and the mean shift in sigma_ref (spec 3.4 gives no separate
    threshold for the shift, so both are held to the same one).
    """
    w = CouplingWindow.coerce(window)
    rt = R_KCAL_MOL_K * float(w.temperature_k)
    k2 = float(k2)
    k1_zero = not (math.isfinite(float(w.primary_k)) and float(w.primary_k) > 0.0)
    base = {"k2": k2, "mode": MODE_K1_ZERO if k1_zero else MODE_K1, "component_index": int(j),
            "primary_center": float(w.primary_center), "primary_k": float(w.primary_k)}
    sigma_w1, designed, note = _sigma_w1(w, rt)
    if sigma_w1 is None:
        return {**base, "status": STATUS_NA, "reason": note, "fraction": None}
    if not (math.isfinite(k2) and k2 > 0.0):
        return {**base, "status": STATUS_OK, "fraction": 0.0, "curvature_kcal": 0.0,
                "sigma_w1": sigma_w1, "designed_cv1_curvature_kcal": designed, "sigma_w1_source": note}
    comp = compile_component(fit, int(j), norm=1.0)
    terms = _terms(comp, w, sigma_w1)
    return {**base, "status": STATUS_OK, "sigma_w1": sigma_w1, "designed_cv1_curvature_kcal": designed,
            "sigma_w1_source": note, "dz_dc_max": terms["dz_dc_max"], "d2z_dc2": terms["d2z_dc2"],
            "anchor_box_std": terms["anchor_box_std"], "anchor_clamp": terms["anchor_clamp"],
            "anchor_clamped": terms["anchor_clamped"],
            **_evaluate(terms, k2, rt, sigma_w1, designed, k1_zero)}


def _largest_root(a: float, b: float, c: float) -> float:
    """Largest s >= 0 with a s^2 + b s <= c (a, b >= 0, c > 0)."""
    if a <= 0.0:
        return math.inf if b <= 0.0 else c / b
    return (-b + math.sqrt(b * b + 4.0 * a * c)) / (2.0 * a)


def largest_passing_k2(fit, j: int, k2: float, window, *,
                       max_fraction: float = MAX_COUPLING_FRACTION) -> Dict[str, Any]:
    """Gate one proposed k2: ``{"k2": passing value, "lowered": bool, "before": ..., "after": ...}``.

    ``k2`` is returned unchanged when it passes (or the gate is NA); otherwise the largest k2
    whose fraction is ``max_fraction``. Solved in closed form in s = sqrt(k2): the curvature
    ratio is (A s^2 + 2 B sqrt(RT) s) / D and, for k1 = 0, the shift is 2 sqrt(RT) dz s sigma^2
    / (RT sigma) -- both monotone in s.
    """
    before = cv2_coupling_fraction(fit, j, k2, window)
    if before["status"] != STATUS_OK or before["fraction"] is None or before["fraction"] <= max_fraction:
        return {"k2": float(k2), "lowered": False, "before": before, "after": before}
    w = CouplingWindow.coerce(window)
    rt = R_KCAL_MOL_K * float(w.temperature_k)
    sigma_w1, designed = before["sigma_w1"], before["designed_cv1_curvature_kcal"]
    terms = _terms(compile_component(fit, int(j), norm=1.0), w, sigma_w1)
    f = float(max_fraction)
    s = _largest_root(terms["A"], BOX_SIGMA * math.sqrt(rt) * terms["B"], f * designed)
    if before["mode"] == MODE_K1_ZERO and terms["dz_dc_max"] > 0.0:
        # mean shift / sigma_ref = 2 sqrt(RT) dz s sigma_ref / RT <= f
        s = min(s, f * math.sqrt(rt) / (BOX_SIGMA * terms["dz_dc_max"] * sigma_w1))
    k2_new = min(float(k2), s * s)
    after = cv2_coupling_fraction(fit, j, k2_new, window)
    return {"k2": k2_new, "lowered": True, "before": before, "after": after}


def gate_layout_rows(fit, j: int, rows, cells, cv1_curvature_kcal, *, temperature_k: float,
                     pooled_cv1_sd: Optional[float], max_fraction: float = MAX_COUPLING_FRACTION,
                     k_min: float = 0.0):
    """Spec 3.4 at the swarm layout: ``(gated_rows, report)``; the input rows are not mutated.

    ``rows`` are :func:`gareus.swarm.ladder_design.layout_rows` rows (center1, k1, center2, k2),
    parallel to ``cells`` ((i1, i2), None for an unrestrained axis). A CV1-restrained cell uses
    k1 + F''(its centre) as the designed CV1 curvature; a CV1-unrestrained cell uses the pooled
    CV1 sd. A cell whose passing k2 falls below ``k_min`` keeps its design k2 and is listed
    under ``below_k_min`` (the report is then ``ok: False``): a layout cell cannot be removed
    here without desynchronising the plan's roles and mandatory ids.
    """
    gated, decisions, below = [], [], []
    for index, (row, cell) in enumerate(zip(rows, cells)):
        new = dict(row)
        gated.append(new)
        k2 = float(row.get("k2") or 0.0)
        if k2 <= 0.0:
            continue
        i1 = cell[0]
        window = CouplingWindow(
            primary_center=float(row["center1"]), primary_k=float(row.get("k1") or 0.0),
            secondary_center=float(row.get("center2") or 0.0), temperature_k=float(temperature_k),
            cv1_curvature_kcal=float(cv1_curvature_kcal[i1]) if i1 is not None else 0.0,
            sigma_w1=None if i1 is not None else pooled_cv1_sd)
        out = largest_passing_k2(fit, j, k2, window, max_fraction=max_fraction)
        before = out["before"]
        rec = {"row": index, "cell": [cell[0], cell[1]], "k2_design": k2, "status": before["status"],
               "fraction_before": before.get("fraction"), "fraction_after": out["after"].get("fraction"),
               "mode": before.get("mode"), "anchor_clamped": before.get("anchor_clamped"),
               "reason": before.get("reason")}
        if out["lowered"] and float(out["k2"]) < float(k_min):
            rec.update({"k2": k2, "k2_passing": float(out["k2"]), "decision": "below_k_min"})
            below.append(index)
        elif out["lowered"]:
            new["k2"] = float(out["k2"])
            rec.update({"k2": new["k2"], "decision": "lowered"})
        else:
            rec.update({"k2": k2, "decision": "pass" if before["status"] == STATUS_OK else "NA"})
        decisions.append(rec)
    fractions = [r["fraction_before"] for r in decisions if r["fraction_before"] is not None]
    reasons = ([f"cv2 coupling: {len(below)} layout cell(s) need k2 below cv2_k_min={k_min:g} to keep the "
                f"CV2-induced CV1 curvature <= {max_fraction} of the designed CV1 curvature (rows {below})"]
               if below else [])
    report = {"ok": not below, "reasons": reasons, "max_coupling_fraction": float(max_fraction),
              "cv2_k_min": float(k_min), "component_index": int(j),
              "n_lowered": sum(1 for r in decisions if r["decision"] == "lowered"),
              "n_below_k_min": len(below), "n_na": sum(1 for r in decisions if r["decision"] == "NA"),
              "fraction_max": max(fractions) if fractions else None,
              "gated_k2": [float(r.get("k2") or 0.0) for r in gated], "decisions": decisions}
    return gated, report
