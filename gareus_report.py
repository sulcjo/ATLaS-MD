"""Human-facing result-health verdict + warning triage for GaREUS PMF analysis.

PRESENTATION-ONLY. This module computes no new physics. It re-reads the numbers
that ``analyze_gareus_mbar.analyze()`` already produced (MBAR convergence, base
ESS, neighbour overlap, GaMD boost anharmonicity, PMF-convergence flag) and
turns them into:

  * a composite PASS / CAUTION / FAIL verdict an operator can read at a glance,
  * per-check status lines that say *why*,
  * warning severity classification + de-duplication (runs routinely emit dozens
    of near-identical warnings; a flat dump buries the one that matters).

Thresholds mirror the analysis's own warning thresholds (``min_neighbor_overlap``
0.30, GaMD exponential-ESS 0.05, anharmonicity 1.0, boost std 6.0 kcal/mol) so
the verdict never disagrees with the warnings the script already prints. Where a
gradation (caution vs fail) is not fixed by an existing threshold it is a
labelled heuristic, documented at the constant.

Scientific nuance baked in: a near-zero GaMD *exponential* reweighting ESS is
EXPECTED whenever ``gamd_cumulant2``/``cumulant3`` is the selected estimator
(cumulant expansion is used precisely because exponential reweighting collapses).
So exponential-ESS never fails the GaMD check and its warning is downgraded to
INFO; the cumulant-validity signal is the boost anharmonicity score.
"""
from __future__ import annotations

import re
import sys
from typing import Any, Optional

# --- status vocabulary -------------------------------------------------------
PASS, CAUTION, FAIL, NA = "pass", "caution", "fail", "na"
_STATUS_RANK = {FAIL: 3, CAUTION: 2, PASS: 1, NA: 0}
_OVERALL = {3: "FAIL", 2: "CAUTION", 1: "PASS", 0: "UNKNOWN"}

# --- thresholds (mirror analyze_gareus_mbar warning thresholds) --------------
ESS_FRAC_CAUTION = 0.05   # matches GaMD-exp ESS warning line; heuristic for base ESS
ESS_FRAC_FAIL = 0.02      # heuristic: below this the estimator is effectively unweighted
ANHARM_CAUTION = 1.0      # matches "anharmonicity score is high (>1.0)" warning
ANHARM_FAIL = 2.0         # heuristic: strongly non-Gaussian boost -> cumulant unreliable
BOOST_STD_CAUTION = 6.0   # matches "boost std is large (>6 kcal/mol)" warning
OVERLAP_FAIL_FRACTION = 0.5   # heuristic: worst < 0.5*target overlap == broken bridge

_GAMD_METHODS = ("gamd_cumulant2", "gamd_cumulant3", "gamd_exponential")

# --- terminal glyphs / colours ----------------------------------------------
_GLYPH = {PASS: "✓", CAUTION: "⚠", FAIL: "✗", NA: "·"}
_ANSI = {PASS: "32", CAUTION: "33", FAIL: "31", NA: "90"}
_OVERALL_ANSI = {"PASS": "32", "CAUTION": "33", "FAIL": "31", "UNKNOWN": "90"}
_SEV_ANSI = {"CRITICAL": "31", "HIGH": "31", "MEDIUM": "33", "INFO": "90"}


# ===========================================================================
# helpers
# ===========================================================================
def _num(x) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # drop NaN


def _worse(a: str, b: str) -> str:
    return a if _STATUS_RANK.get(a, 0) >= _STATUS_RANK.get(b, 0) else b


def _human_count(n: Optional[float]) -> str:
    if n is None:
        return "?"
    n = float(n)
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}k"
    return f"{n:.0f}"


# ===========================================================================
# verdict
# ===========================================================================
def build_health_verdict(s: dict, min_neighbor_overlap: float = 0.30) -> dict:
    """Derive a PASS/CAUTION/FAIL verdict from an analysis summary dict ``s``.

    Never raises on partial/old-format summaries; missing inputs yield ``na``
    checks that do not affect the overall verdict.
    """
    s = s or {}
    checks: list[dict] = []
    checks.append(_check_mbar(s))
    checks.append(_check_zero_windows(s))
    checks.append(_check_base_ess(s))
    checks.append(_check_overlap(s, float(min_neighbor_overlap)))
    checks.append(_check_gamd(s))
    checks.append(_check_pmf_convergence(s))

    worst = NA
    for c in checks:
        worst = _worse(worst, c["status"])
    overall = _OVERALL[_STATUS_RANK[worst]]

    return {
        "overall": overall,
        "checks": checks,
        "headline": _headline(s),
    }


def _headline(s: dict) -> list[dict]:
    mbar = s.get("mbar", {}) or {}
    ns = _num(s.get("n_samples"))
    base_ess = _num(mbar.get("base_ess"))
    frac = (base_ess / ns) if (base_ess is not None and ns) else None
    hl = [
        {"label": "selected PMF", "value": s.get("selected_unbiased_method", "?")},
        {"label": "samples/windows",
         "value": f"{_human_count(ns)} / {s.get('n_windows', '?')}"},
        {"label": "PMF span",
         "value": (f"{_num(s.get('pmf_span_kcal_mol')):.2f} kcal/mol"
                   if _num(s.get("pmf_span_kcal_mol")) is not None else "?")},
        {"label": "base ESS",
         "value": (f"{frac*100:.1f}% ({_human_count(base_ess)})"
                   if frac is not None else "?")},
    ]
    return hl


def _check_mbar(s: dict) -> dict:
    mbar = s.get("mbar")
    if not isinstance(mbar, dict) or "converged" not in mbar:
        return {"name": "MBAR convergence", "status": NA, "detail": "no MBAR result"}
    md = _num(mbar.get("max_delta"))
    md_txt = f" (max_delta {md:.1e})" if md is not None else ""
    if mbar.get("converged"):
        return {"name": "MBAR convergence", "status": PASS,
                "detail": f"converged{md_txt}"}
    return {"name": "MBAR convergence", "status": FAIL,
            "detail": f"did NOT converge{md_txt}"}


def _check_zero_windows(s: dict) -> dict:
    n_k = (s.get("mbar", {}) or {}).get("n_k")
    if not isinstance(n_k, (list, tuple)) or not n_k:
        return {"name": "Window sampling", "status": NA, "detail": "per-window counts unavailable"}
    empty = [i for i, n in enumerate(n_k) if (_num(n) or 0) <= 0]
    if empty:
        show = ", ".join(str(i) for i in empty[:8]) + (" …" if len(empty) > 8 else "")
        return {"name": "Window sampling", "status": FAIL,
                "detail": f"{len(empty)} window(s) with zero samples: {show}"}
    return {"name": "Window sampling", "status": PASS,
            "detail": f"all {len(n_k)} windows populated"}


def _check_base_ess(s: dict) -> dict:
    ns = _num(s.get("n_samples"))
    base_ess = _num((s.get("mbar", {}) or {}).get("base_ess"))
    if ns is None or not ns or base_ess is None:
        return {"name": "Sampling (base ESS)", "status": NA, "detail": "ESS unavailable"}
    frac = base_ess / ns
    detail = f"{frac*100:.1f}% of {_human_count(ns)} samples ({_human_count(base_ess)} eff.)"
    if frac < ESS_FRAC_FAIL:
        st = FAIL
    elif frac < ESS_FRAC_CAUTION:
        st = CAUTION
    else:
        st = PASS
    return {"name": "Sampling (base ESS)", "status": st, "detail": detail, "metric": frac}


def _check_overlap(s: dict, thr: float) -> dict:
    neigh = s.get("neighbor_overlap")
    if not isinstance(neigh, (list, tuple)) or not neigh:
        return {"name": "Window overlap", "status": NA, "detail": "no neighbour overlap"}
    vals = [(_num(x) if _num(x) is not None else 1.0) for x in neigh]
    worst = min(vals)
    wi = vals.index(worst)
    detail = f"worst {worst:.3f} (pair {wi}-{wi+1}); target ≥{thr:.2f}"
    if worst < OVERLAP_FAIL_FRACTION * thr:
        st = FAIL
    elif worst < thr:
        st = CAUTION
    else:
        st = PASS
    return {"name": "Window overlap", "status": st, "detail": detail, "metric": worst}


def _check_gamd(s: dict) -> dict:
    selected = str(s.get("selected_unbiased_method", ""))
    boost = s.get("boost", {}) or {}
    name = "GaMD reweighting"
    if selected not in _GAMD_METHODS or not boost.get("available"):
        return {"name": name, "status": NA,
                "detail": "umbrella-only PMF (no GaMD boost applied)"}
    anh = _num(boost.get("anharmonicity_score"))
    std = _num(boost.get("std_kcal_mol"))
    is_cumulant = selected in ("gamd_cumulant2", "gamd_cumulant3")
    bits = []
    st = PASS
    if anh is not None:
        bits.append(f"anharmonicity {anh:.2f}")
        if anh > ANHARM_FAIL:
            st = _worse(st, FAIL)
        elif anh > ANHARM_CAUTION:
            st = _worse(st, CAUTION)
    if std is not None:
        bits.append(f"boost σ {std:.1f} kcal/mol")
        if std > BOOST_STD_CAUTION:
            st = _worse(st, CAUTION)
    # exponential-ESS: informational for cumulant estimators (expected to be ~0)
    frac = _num(boost.get("boost_reweight_ess_fraction"))
    if frac is not None:
        if is_cumulant:
            bits.append(f"exp-reweight ESS {frac*100:.3f}% (expected low)")
        else:
            bits.append(f"exp-reweight ESS {frac*100:.3f}%")
            if frac < ESS_FRAC_FAIL:
                st = _worse(st, FAIL)
            elif frac < ESS_FRAC_CAUTION:
                st = _worse(st, CAUTION)
    label = ("cumulant2 validity" if selected == "gamd_cumulant2"
             else "cumulant3 validity" if selected == "gamd_cumulant3"
             else "exponential reweighting")
    return {"name": name, "status": st,
            "detail": f"{label}: " + ", ".join(bits) if bits else label}


def _check_pmf_convergence(s: dict) -> dict:
    conv = s.get("convergence", {}) or {}
    name = "PMF convergence"
    if not conv.get("enabled"):
        return {"name": name, "status": NA, "detail": conv.get("reason", "not run")}
    summ = conv.get("summary") or {}
    cb = summ.get("converged_bool")
    if cb is None:
        return {"name": name, "status": NA, "detail": "convergence summary unavailable"}
    js = _num(summ.get("tail_median_JS"))
    rm = _num(summ.get("tail_median_RMSE_F_kcal_mol"))
    tail = []
    if js is not None:
        tail.append(f"tail JS {js:.3g}")
    if rm is not None:
        tail.append(f"tail RMSE {rm:.2f} kcal/mol")
    tail_txt = (" (" + ", ".join(tail) + ")") if tail else ""
    if int(cb) == 1:
        return {"name": name, "status": PASS, "detail": f"converged by JS/RMSE tail test{tail_txt}"}
    return {"name": name, "status": CAUTION,
            "detail": f"NOT converged by JS/RMSE tail test{tail_txt}"}


# ===========================================================================
# warning triage
# ===========================================================================
# (regex, severity). First match wins; unmatched -> MEDIUM.
_WARN_RULES: list[tuple[re.Pattern, str]] = [
    # INFO: expected / informational, not a defect
    (re.compile(r"exponential reweighting ESS is very low", re.I), "INFO"),
    (re.compile(r"\bApplied analysis stride\b", re.I), "INFO"),
    (re.compile(r"\bAuto-selected\b", re.I), "INFO"),
    (re.compile(r"\breconstructed\b", re.I), "INFO"),
    (re.compile(r"kept \d+/\d+ samples", re.I), "INFO"),
    # CRITICAL: result may be invalid
    (re.compile(r"did not (fully )?converge", re.I), "CRITICAL"),
    (re.compile(r"[Zz]ero production samples", re.I), "CRITICAL"),
    (re.compile(r"\bESS[^.]*\b(is\s+)?(effectively\s+)?zero\b", re.I), "CRITICAL"),
    # HIGH: quality concern that biases the PMF
    (re.compile(r"[Ww]eak neighbor CV overlap", re.I), "HIGH"),
    (re.compile(r"anharmonicity score is high", re.I), "HIGH"),
    (re.compile(r"boost std is large", re.I), "HIGH"),
    (re.compile(r"umbrella-only", re.I), "HIGH"),
    (re.compile(r"cumulant reweighting may be unreliable", re.I), "HIGH"),
]

_SEV_RANK = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "INFO": 3}


_EXP_ESS_RX = re.compile(r"exponential reweighting ESS is very low", re.I)


def _severity_of(w: str, selected: Optional[str] = None) -> str:
    # When exponential reweighting is the *selected* estimator, a low exponential
    # ESS is a genuine quality problem, not the expected-and-ignored case that
    # holds for cumulant estimators — so don't downgrade it to INFO.
    if selected == "gamd_exponential" and _EXP_ESS_RX.search(w):
        return "HIGH"
    for rx, sev in _WARN_RULES:
        if rx.search(w):
            return sev
    return "MEDIUM"


def _dedup_key(w: str) -> str:
    """Collapse warnings differing only in numbers/ids so duplicates group."""
    k = re.sub(r"[-+]?\d[\d.,eE+_-]*", "#", w)   # numbers -> #
    k = re.sub(r"\s+", " ", k).strip()
    return k


def classify_warnings(warnings, selected: Optional[str] = None) -> list[dict]:
    """Group warnings by (normalized text), tag severity, sort severe-first.

    Returns list of ``{severity, count, representative}`` dicts. Order preserves
    first-seen among equal (severity, -count). ``selected`` is the chosen PMF
    estimator; it only affects the exponential-ESS downgrade (see ``_severity_of``).
    """
    if not warnings:
        return []
    groups: dict[str, dict] = {}
    order: list[str] = []
    for w in warnings:
        w = str(w)
        key = _dedup_key(w)
        g = groups.get(key)
        if g is None:
            groups[key] = {"severity": _severity_of(w, selected), "count": 1, "representative": w}
            order.append(key)
        else:
            g["count"] += 1
    out = [groups[k] for k in order]
    out.sort(key=lambda g: (_SEV_RANK.get(g["severity"], 2), -g["count"]))
    return out


def _severity_tally(groups) -> str:
    tally: dict[str, int] = {}
    for g in groups or []:
        tally[g["severity"]] = tally.get(g["severity"], 0) + 1
    parts = [f"{tally[s]} {s}" for s in ("CRITICAL", "HIGH", "MEDIUM", "INFO") if tally.get(s)]
    return ", ".join(parts) if parts else "none"


# ===========================================================================
# rendering
# ===========================================================================
def _c(txt: str, code: str, use_color: bool) -> str:
    return f"\x1b[{code}m{txt}\x1b[0m" if use_color else txt


def render_verdict_terminal(verdict: Optional[dict], groups, use_color: Optional[bool] = None) -> str:
    """Render a compact end-screen health block. Safe on None/degraded input."""
    if use_color is None:
        use_color = bool(getattr(sys.stdout, "isatty", lambda: False)())
    v = verdict or {"overall": "UNKNOWN", "checks": [], "headline": []}
    overall = v.get("overall", "UNKNOWN")
    width = 60
    head = f" RESULT HEALTH: {overall} "
    bar = _c(head.center(width, "="), _OVERALL_ANSI.get(overall, "90"), use_color)
    lines = ["", bar]
    for c in v.get("checks", []):
        st = c.get("status", NA)
        glyph = _c(_GLYPH.get(st, "?"), _ANSI.get(st, "90"), use_color)
        lines.append(f"  {glyph} {c.get('name',''):<22} {c.get('detail','')}")
    if groups:
        lines.append("-" * width)
        lines.append(f"  Warnings: {_severity_tally(groups)}")
        shown = [g for g in groups if g["severity"] in ("CRITICAL", "HIGH")][:6]
        for g in shown:
            sev = _c(f"{g['severity']:<8}", _SEV_ANSI.get(g["severity"], "0"), use_color)
            cnt = f" (×{g['count']})" if g["count"] > 1 else ""
            lines.append(f"   {sev}{_truncate(g['representative'], width-14)}{cnt}")
        n_lower = sum(1 for g in groups if g["severity"] in ("MEDIUM", "INFO"))
        if n_lower:
            lines.append(_c(f"   … {n_lower} more MEDIUM/INFO group(s) — see pmf_summary.md",
                            "90", use_color))
    lines.append("=" * width)
    return "\n".join(lines)


def _truncate(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[: max(1, n - 1)] + "…"


def render_verdict_md(verdict: Optional[dict], groups) -> list[str]:
    """Render the '## Result health' Markdown section as a list of lines."""
    v = verdict or {}
    overall = v.get("overall", "UNKNOWN")
    badge = {"PASS": "\U0001f7e2 PASS", "CAUTION": "\U0001f7e1 CAUTION",
             "FAIL": "\U0001f534 FAIL", "UNKNOWN": "⚪ UNKNOWN"}.get(overall, overall)
    lines = ["## Result health", "", f"**Overall: {badge}**", ""]
    hl = v.get("headline") or []
    if hl:
        lines.append(" · ".join(f"{h['label']}: **{h['value']}**" for h in hl))
        lines.append("")
    lines += ["| Check | Status | Detail |", "|---|---|---|"]
    _md_status = {PASS: "✓ pass", CAUTION: "⚠ caution", FAIL: "✗ FAIL", NA: "· n/a"}
    for c in v.get("checks", []):
        st = _md_status.get(c.get("status", NA), c.get("status", ""))
        detail = str(c.get("detail", "")).replace("|", "\\|")
        lines.append(f"| {c.get('name','')} | {st} | {detail} |")
    lines.append("")
    if groups:
        lines += [f"### Warnings by severity ({_severity_tally(groups)})", ""]
        for g in groups:
            cnt = f" _(×{g['count']})_" if g["count"] > 1 else ""
            lines.append(f"- **{g['severity']}** — {g['representative']}{cnt}")
        lines.append("")
    return lines
