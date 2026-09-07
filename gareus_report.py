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

# Joint (CV1, CV2) overlap has its own target, published per-run in
# ``s['joint_overlap']['threshold']``; this is only the fallback for a summary
# that carries joint numbers without one. It is NOT ``min_neighbor_overlap``:
# joint overlap is bounded above by the CV1 marginal (refining a histogram can
# only reduce sum_c min(p_c, q_c)) and deflates further at small per-state N, so
# grading it against a marginal-calibrated 0.30 would fail 2D runs that are
# perfectly well resolved. The default is that same per-axis quality applied to
# both axes, i.e. ``min_neighbor_overlap ** 2``.
JOINT_OVERLAP_THRESHOLD_EXPONENT = 2

# Names for the two overlap spaces, published as `metric_space` beside each
# check's `metric`. MIRRORS gareus.mbar_analysis.pmf.OVERLAP_SPACE_MARGINAL /
# _JOINT -- pmf.py is the source of truth (it stamps the same strings into
# pmf_summary.json's `overlap_space` and each overlap CSV's header), and these
# are duplicated rather than imported for the same reason SELF_BIAS_* above is:
# this module must keep working standalone against an old pmf_summary.json.
# Same vocabulary on purpose, so a consumer can compare a check's metric_space
# with the summary's own overlap_space without a translation table.
OVERLAP_SPACE_MARGINAL = "cv1_marginal"
OVERLAP_SPACE_JOINT = "cv1_cv2_joint"

# Per-state self-bias (each state's own samples scored in its own restraint),
# in kT. MIRRORS gareus.mbar_analysis.pmf.SELF_BIAS_MEDIAN_WARN_KT /
# _P90_WARN_KT -- pmf.py is the source of truth; these are duplicated rather
# than imported so this module keeps working standalone against an old
# pmf_summary.json (its whole contract; see the module docstring).
#
# ~1 kT is the only physical answer for a correctly-mapped 2-DOF harmonic, at
# any temperature, CV or window spacing. chignolin_6's 24 healthy states
# measured 0.5-1.4 kT median / 1.7-4.9 kT p90; its three corrupted ones measured
# 63.6 / 1.1 / 132.9 median and 80.7 / 652.8 / 221.8 p90.
SELF_BIAS_MEDIAN_FAIL_KT = 10.0
SELF_BIAS_P90_FAIL_KT = 50.0
# Caution band: deliberately loose. The real run's worst *healthy* state was
# 3.4 kT median (state 20 -- which was in fact a mis-mapped phantom, so this
# band is set above it knowingly: a near-neighbour mis-mapping is invisible to
# this check by construction, see self_bias_diagnostics' own docstring). Set
# tighter than that and stiff restraints against a steric wall would start
# failing healthy runs, which is the mistake this whole review round is about.
SELF_BIAS_MEDIAN_CAUTION_KT = 5.0

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


def _iter_numbers(seq) -> list:
    """``seq`` as a plain list, for any array-like that is not a string/mapping.

    Summary values reach this module either as numpy arrays (called in-process
    from analyze()) or as JSON lists (re-run standalone against a written
    pmf_summary.json); both must work, and anything else must degrade to empty
    rather than raise.
    """
    if seq is None or isinstance(seq, (str, bytes, dict)):
        return []
    try:
        return list(seq)
    except TypeError:
        return []


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
    # Separate check, not folded into _check_overlap: this one grades a property
    # of the whole overlap GRAPH (is the set bridged at all), which no pairwise
    # worst-overlap number can express -- and `metric` there is a scalar
    # overlap, not a component count.
    checks.append(_check_overlap_connectivity(s))
    checks.append(_check_mapping(s))
    checks.append(_check_gamd(s))
    checks.append(_check_pmf_convergence(s))
    checks.append(_check_ladder_crosscheck(s))

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


def _worst_of(rows) -> Optional[tuple]:
    """Worst ``(overlap, i, j)`` over a list of CV-space nearest-neighbour rows.

    Returns None for any absent/old/malformed list so every caller's fallback
    stays reachable -- this module's contract is never to raise on a partial
    summary.
    """
    if not isinstance(rows, (list, tuple)) or not rows:
        return None
    best = None
    for r in rows:
        if not isinstance(r, dict):
            continue
        ov = _num(r.get("overlap"))
        if ov is None:
            continue
        i, j = r.get("window"), r.get("neighbor")
        if i is None or j is None:
            continue
        if best is None or ov < best[0]:
            best = (ov, int(i), int(j))
    return best


def _cv_space_worst(s: dict) -> Optional[tuple]:
    """Worst (overlap, i, j) over each state's CV-space nearest-neighbour pair,
    in the CV1-MARGINAL space.

    ``cv_space_neighbor_overlap`` is produced by
    ``gareus.mbar_analysis.pmf.cv_space_neighbor_overlap`` and is the pairing an
    operator actually cares about: each window against its nearest neighbour in
    (primary, secondary) restraint-centre space, sigma-scaled.
    """
    return _worst_of(s.get("cv_space_neighbor_overlap"))


def _status_for_overlap(worst: float, thr: float) -> str:
    if worst < OVERLAP_FAIL_FRACTION * thr:
        return FAIL
    if worst < thr:
        return CAUTION
    return PASS


def _index_adjacent_worst(arr) -> Optional[tuple]:
    """Worst ``(overlap, i, i+1)`` over a consecutive-INDEX overlap array.

    Non-numeric entries count as 1.0 rather than being dropped, so the returned
    position still indexes the caller's own array (dropping them would shift
    every reported window id after the first bad entry).
    """
    vals = _iter_numbers(arr)
    if not vals:
        return None
    nums = [(_num(x) if _num(x) is not None else 1.0) for x in vals]
    worst = min(nums)
    wi = nums.index(worst)
    return (float(worst), wi, wi + 1)


def _check_overlap(s: dict, thr: float) -> dict:
    """Window overlap, graded in every pairing and space the summary carries --
    the WORST status wins.

    Three independent corrections to what this check used to do, all from
    docs/chignolin_6_low_ess_root_cause.md:

    1. Pair windows by CV-space adjacency, not index adjacency. On the real run
       this check reported "worst 0.013 (pair 23-24)" and sent the whole
       investigation at state 24 -- a well-overlapped, innocent window that
       merely happened to be indexed next to 23; the two are nowhere near each
       other in CV space. Index adjacency is only meaningful for a monotone 1D
       ladder.

    2. Grade the JOINT (CV1, CV2) overlap when the analysis computed one. The
       marginal number is what let that run pass this check while 20 of its 27
       states were essentially disjoint along the axis it cannot see.

    3. Keep grading the index-adjacent array as well (round-3 review). Fix 1
       originally REPLACED the index grading with the CV-space grading, which
       made the index branch dead code for every new analysis
       (_report_summary_fields always emits cv_space_neighbor_overlap, and
       cv_space_neighbor_overlap() is non-empty for any run with >=2 populated
       windows) -- so a broken umbrella ladder could start reading PASS on
       unchanged data. Correcting the pairing must not cost the number the old
       pairing did catch: both are computed, both are graded, both appear in
       `detail`, and the CV-space pair remains the one named as physically
       adjacent.

    Every number is graded against its own threshold and the worse status wins.
    That asymmetry is deliberate and necessary in each direction: joint overlap
    is bounded above by the marginal, so a joint pass does not imply a marginal
    pass (0.20 joint clears 0.09 while 0.25 marginal fails 0.30); and a
    nearest-neighbour pairing only ever examines one edge per state, so a
    passing CV-space pair says nothing about a pair the pairing never compared.
    Adding a number here can never weaken this check.

    Note what this check still CANNOT do, however many pairings it grades:
    certify that the window set is bridged. That is a property of the whole
    overlap graph, not of any pair -- see _check_overlap_connectivity below.
    """
    joint = s.get("joint_overlap")
    joint = joint if isinstance(joint, dict) else {}
    jworst = _worst_of(joint.get("cv_space_neighbor_overlap")) if joint.get("available") else None
    if jworst is None and joint.get("available"):
        # A joint block with only the index-adjacent array (older/partial
        # summary): still better than ignoring the joint space entirely.
        jworst = _index_adjacent_worst(joint.get("neighbor_overlap"))

    cv_worst = _cv_space_worst(s)
    iworst = _index_adjacent_worst(s.get("neighbor_overlap"))
    if cv_worst is None and iworst is None and jworst is None:
        return {"name": "Window overlap", "status": NA, "detail": "no neighbour overlap"}

    st = NA
    metric = None
    # Set in the same statement as `metric`, never separately: the whole point
    # of this sibling is that it cannot disagree with which number won.
    metric_space = None
    parts: list[str] = []
    if cv_worst is not None:
        worst, i, j = cv_worst
        st = _worse(st, _status_for_overlap(worst, thr))
        metric, metric_space = worst, OVERLAP_SPACE_MARGINAL
        parts.append(f"worst {worst:.3f} CV1-marginal (windows {i}-{j}, CV-space nearest "
                     f"neighbours)")
    if iworst is not None:
        worst, i, j = iworst
        st = _worse(st, _status_for_overlap(worst, thr))
        if metric is None:
            metric, metric_space = worst, OVERLAP_SPACE_MARGINAL
        # Spelled out deliberately: the old wording ("pair 23-24") reads as a
        # physical adjacency claim it cannot support, which is exactly what
        # misdirected the chignolin_6 investigation. It is still graded (see
        # correction 3 above) -- just never presented as the adjacent pair.
        parts.append(f"worst {worst:.3f} CV1-marginal index-adjacent (windows {i}-{j} -- "
                     f"may not be neighbours in CV space)")
    detail = "; ".join(parts)
    if detail:
        detail += f"; target ≥{thr:.2f}"

    if jworst is not None:
        jthr = _num(joint.get("threshold"))
        if jthr is None or jthr <= 0.0:
            jthr = float(thr) ** JOINT_OVERLAP_THRESHOLD_EXPONENT
        jw, ji, jj = jworst
        jst = _status_for_overlap(jw, jthr)
        # Lead with the joint number: it is the one that sees both axes. The
        # marginal value for the SAME pair is quoted beside it, because the gap
        # between the two is the whole chignolin_6 illusion in one line (0.97
        # marginal, 0.02 joint) and an operator who sees only the joint number
        # has no way to tell a genuinely 2D-under-resolved run from one whose
        # primary ladder is broken too.
        same_pair = next((r for r in (s.get("cv_space_neighbor_overlap") or [])
                          if isinstance(r, dict) and r.get("window") == ji
                          and r.get("neighbor") == jj), None)
        marg_txt = ""
        if isinstance(same_pair, dict) and _num(same_pair.get("overlap")) is not None:
            marg_txt = f", CV1-marginal for that pair {_num(same_pair['overlap']):.2f}"
        jdetail = (f"worst {jw:.3f} joint (CV1,CV2) (windows {ji}-{jj}, CV-space nearest "
                   f"neighbours{marg_txt}); target ≥{jthr:.3f} (= {thr:.2f} per axis)")
        detail = f"{jdetail}; {detail}" if detail else jdetail
        st = _worse(st, jst)
        metric, metric_space = jw, OVERLAP_SPACE_JOINT
    out = {"name": "Window overlap", "status": st, "detail": detail}
    if metric is not None:
        # Convention when several numbers are graded: `metric` is the JOINT one
        # when it exists (the only one that sees both axes), else the CV-space
        # marginal, else the index-adjacent one -- while `status` is the worst
        # over all of them and may therefore have come from a different one.
        # Every number is always spelled out in `detail`; nothing in the repo
        # consumes `metric` programmatically today, so it stays a single scalar
        # rather than growing into a per-space dict.
        #
        # But WHICH of those three won is not inferable from the scalar, and the
        # two spaces are the same shape and scale with different thresholds:
        # 0.20 is a healthy joint overlap and a failing marginal one. An
        # external consumer grading `metric` against --min-neighbor-overlap
        # (a marginal-calibrated threshold, see JOINT_OVERLAP_THRESHOLD_EXPONENT)
        # would silently misread a joint value as a marginal one, so every
        # `metric` now ships the space it was measured in beside it.
        out["metric"] = metric
        out["metric_space"] = metric_space
    return out


def _check_overlap_connectivity(s: dict) -> dict:
    """Is the umbrella window set BRIDGED? The question no worst-pair number can
    answer, in either space.

    MBAR determines free energies only up to one additive constant per connected
    component of the overlap graph: two blocks of states that never exchange
    samples carry no information about each other, so the offset between them is
    fixed by nothing in the data and every cross-block free-energy difference --
    the PMF span included -- is arbitrary. That is chignolin_6's actual failure
    mode (its posterior put 94% of its weight on one state), and it is invisible
    to `_check_overlap` above by construction: a set can split in two while
    every graded pair clears the threshold.

    ``s['overlap_connectivity']`` is computed by
    ``gareus.mbar_analysis.pmf.overlap_components`` in both spaces, each graded
    at its own threshold, and the worse status wins here for the same reason it
    does above -- neither space's connectivity implies the other's (joint
    overlap is bounded above by the marginal, while the joint threshold is
    BELOW the marginal one). On the real run the marginal graph is fully
    connected at 0.30 while the joint graph splits into 3 components at 0.09, so
    grading only the always-available marginal would be silent on the very run
    this exists for.

    A split is graded on the MOST ISOLATED component's best escape route
    (``worst_component_best_cross_overlap`` -- its strongest overlap with
    anything outside it), through the same ``_status_for_overlap`` calibration
    every other overlap number here uses, and never better than CAUTION since a
    split is a split:

      * essentially no overlap out of some component (below
        ``OVERLAP_FAIL_FRACTION`` of the target) -> FAIL. That block's free
        energy relative to the rest really is undetermined. chignolin_6's
        isolated state 20 measures 0.0048 to anything outside its component.
      * a weak-but-real link that merely misses the target -> CAUTION, i.e. the
        same severity the pairwise check already gives that pair, plus the
        sharper news that it is the ONLY bridge. Its state 18 measures 0.0648
        against a 0.09 target.

    The most isolated component, not the strongest link across the split: on the
    real run those give different answers (0.0048 vs 0.0648) and grading on the
    latter would let state 18's weak-but-real link talk state 20's isolation
    down to a CAUTION -- which is what the first version of this code did.

    Grading a bare component count FAIL would instead flip every run with one
    slightly-under-target link -- already a CAUTION on the pairwise check --
    into a FAIL on unchanged data, while asserting in the detail text that the
    free energies are undetermined when they demonstrably are not. An unknown
    cut strength (older summary, or an all-non-finite cut) grades FAIL: a split
    whose severity cannot be measured is not evidence of a mild one.

    A space whose matrix was not computed (``--no-joint-overlap``, or too few
    samples carrying a secondary CV) reports ``na`` with its reason rather than
    being silently treated as connected. Absent entirely (an old
    pmf_summary.json), the whole check is ``na`` -- this module must keep working
    against summaries written before it existed.
    """
    conn = s.get("overlap_connectivity")
    if not isinstance(conn, dict) or not conn:
        return {"name": "Overlap connectivity", "status": NA,
                "detail": "overlap connectivity not computed by this analysis"}
    st = NA
    parts: list[str] = []
    n_worst = None
    # Joint first: it is the space that sees both axes, so it leads the detail
    # string exactly as it does in _check_overlap.
    for key, label in (("joint", "joint (CV1,CV2)"), ("marginal", "CV1-marginal")):
        block = conn.get(key)
        if not isinstance(block, dict):
            continue
        if not block.get("available", True):
            # Reported, not silently skipped: an unmeasured space leaves this
            # check blind on that axis, and on the real run the joint axis is
            # the ONLY one that showed the split. Contributes no status (na),
            # so it neither passes nor fails the run on its own.
            reason = str(block.get("reason") or "not computed")
            parts.append(f"{label} not measured ({reason})")
            continue
        n = _num(block.get("n_components"))
        if n is None or n < 1:
            # 0 graded states: nothing to connect. The zero-sample-window check
            # owns that condition.
            parts.append(f"{label} has no populated windows to grade")
            continue
        n = int(n)
        thr = _num(block.get("threshold"))
        thr_txt = f" at ≥{thr:.3f}" if thr is not None else ""
        if n > 1:
            best = _num(block.get("worst_component_best_cross_overlap"))
            iso = block.get("most_isolated_component")
            # Never better than CAUTION (a split is a split, whatever the
            # summary claims the best link was), and FAIL when the cut is
            # essentially empty or its strength is unknown.
            bst = (_worse(CAUTION, _status_for_overlap(best, thr))
                   if (best is not None and thr is not None) else FAIL)
            st = _worse(st, bst)
            n_worst = n if n_worst is None else max(n_worst, n)
            iso_txt = ""
            if isinstance(iso, (list, tuple)) and iso:
                iso_txt = f" [{','.join(str(x) for x in list(iso)[:6])}]"
            best_txt = (f", most isolated component{iso_txt} links out at only {best:.4f}"
                        if best is not None else ", isolation depth unknown")
            parts.append(f"{label} SPLIT into {n} components{thr_txt}: "
                         f"{_components_text(block)}{best_txt}")
        else:
            st = _worse(st, PASS)
            parts.append(f"{label} connected (1 component{thr_txt})")
    if not parts:
        return {"name": "Overlap connectivity", "status": NA,
                "detail": "overlap connectivity not computed by this analysis"}
    out = {"name": "Overlap connectivity", "status": st, "detail": "; ".join(parts)}
    if n_worst is not None:
        out["metric"] = n_worst
    return out


def _components_text(block: dict, max_components: int = 4, max_states: int = 6) -> str:
    """``[18] 3.4k, [20] 6.0k, [0,1,2,...] 117.0k`` for one graded space.

    Deliberately NOT shared with gareus.mbar_analysis.pmf.overlap_component_text:
    this module must keep working standalone against an old pmf_summary.json
    with no gareus package importable (its whole contract -- see the module
    docstring and SELF_BIAS_MEDIAN_FAIL_KT), and the two renderings differ on
    purpose anyway -- the warning line spells counts out in full, a terminal
    check row uses this module's own abbreviated _human_count.

    The per-component sample counts are quoted, not just the component count:
    they are what lets an operator tell a small-N joint-histogram artefact
    (chignolin_6's isolated states hold 3.4k and 6.0k samples of 126k) from a
    physically unbridged region. Truncated so a shattered K=364 geometry cannot
    render a multi-kilobyte terminal line; the full lists stay in
    pmf_summary.json.
    """
    comps = block.get("components")
    comps = list(comps) if isinstance(comps, (list, tuple)) else []
    samples = block.get("component_samples")
    samples = list(samples) if isinstance(samples, (list, tuple)) else []
    parts = []
    for n, c in enumerate(comps[:max_components]):
        ids = list(c) if isinstance(c, (list, tuple)) else [c]
        shown = ",".join(str(x) for x in ids[:max_states])
        if len(ids) > max_states:
            shown += f",+{len(ids) - max_states}"
        txt = f"[{shown}]"
        if n < len(samples) and _num(samples[n]) is not None:
            txt += f" {_human_count(_num(samples[n]))}"
        parts.append(txt)
    if len(comps) > max_components:
        parts.append(f"+{len(comps) - max_components} more")
    return ", ".join(parts)


# Warnings that mean this analysis's samples are knowingly attributed to the
# wrong umbrella states. Matched against s['warnings'] by _check_mapping below.
# Wording comes from gareus/mbar_analysis/loaders_adaptive.py.
_STALE_MAP_OVERRIDE_RX = re.compile(
    r"\[stale window map\][^\n]*(Loaded with the STALE map anyway|"
    r"GAREUS_ALLOW_STALE_WINDOW_MAP is set)", re.I)
_STALE_MAP_ANY_RX = re.compile(r"\[stale window map\]", re.I)
# The equal-count membership failure: the map has the right NUMBER of rows but
# lists a different window set than the phase's own surviving-window table.
# Matched on the note's own claim rather than on its tag, so the sibling
# "could not cross-check" note (same tag, weaker fact) can never reach the FAIL
# below -- a check that did not run is not a detected fault.
_MAP_MEMBERSHIP_RX = re.compile(
    r"\[window map check\][^\n]*DIFFERENT window set of the same size", re.I)


def _check_mapping(s: dict) -> dict:
    """Is each sample attributed to the umbrella state it was actually generated
    in? The check the chignolin_6 campaign did not have.

    Three independent inputs, in order of authority:

    1. The stale-``epoch_window_map.csv`` loader guard's own notes. The
       ``GAREUS_ALLOW_STALE_WINDOW_MAP`` escape hatch loads a run whose samples
       are provably attributed to the wrong states; before this check existed
       that produced nothing but a MEDIUM-by-default warning, so the resulting
       pmf_summary.json could read PASS. An operator who forgets the variable is
       exported must not be able to publish such a run as healthy.

    1b. The equal-count *membership* note from the same guard, which fires when
       a phase's map and its own post-drop surviving-window table describe
       different window sets of the same size. Deterministic bookkeeping (the
       two artifacts are written from the same in-memory arrays by the process
       that runs the phase), so it is graded with the escape hatch rather than
       with the physical diagnostics below -- but the note itself cannot say
       WHICH of the two records is stale, and the detail text does not pretend
       otherwise.

    2. ``s['self_bias']`` -- each state's own samples scored in its own restraint
       (``gareus.mbar_analysis.pmf.self_bias_diagnostics``). ~1 kT is the only
       physical answer for a 2-DOF harmonic, so >10 kT is unambiguous evidence
       that the sample-to-state mapping or the window parameters are wrong. This
       is a one-way test: passing it does NOT prove the mapping is right (a
       mis-mapping between two nearby windows stays inside the band -- on the
       real run a wholly phantom state measured a benign 3.4 kT).

    Reads the MAIN report's self_bias only. ``epoch_000_report``'s own copy is
    published beside it but does not gate the verdict, matching every other
    headline number in this summary (pmf_span, base ESS, boost) -- a deliberate
    convention, not an oversight.
    """
    name = "Sample-to-state mapping"
    warns = s.get("warnings")
    warns = [str(w) for w in warns] if isinstance(warns, (list, tuple)) else []
    for w in warns:
        if _STALE_MAP_OVERRIDE_RX.search(w):
            return {"name": name, "status": FAIL,
                    "detail": ("loaded with a STALE epoch_window_map.csv because "
                               "GAREUS_ALLOW_STALE_WINDOW_MAP is set: samples are attributed to "
                               "the wrong umbrella states and every free energy here is invalid")}
    for w in warns:
        if _MAP_MEMBERSHIP_RX.search(w):
            return {"name": name, "status": FAIL,
                    "detail": ("a phase's epoch_window_map.csv lists a DIFFERENT window set of the "
                               "same size than the phase's own surviving-window table: one of the "
                               "two records is stale and the artifacts do not say which, so that "
                               "phase's samples may be attributed to the wrong umbrella states")}
    repaired = any(_STALE_MAP_ANY_RX.search(w) for w in warns)

    sb = s.get("self_bias")
    # Accept BOTH shapes this can arrive in: the in-memory dict of numpy arrays
    # (build_health_verdict called from inside analyze()) and the JSON-round-
    # tripped dict of plain lists with null where NaN was (this module re-run
    # standalone against a pmf_summary.json). A list/tuple-only isinstance test
    # silently returns "unavailable" for the in-process call -- i.e. for every
    # real run -- which is precisely the class of gap this review round is about.
    med = (sb or {}).get("median_kT") if isinstance(sb, dict) else None
    p90 = (sb or {}).get("p90_kT") if isinstance(sb, dict) else None
    worst_med = worst_p90 = None
    worst_state = None
    for k, v in enumerate(_iter_numbers(med)):
        f = _num(v)
        if f is not None and (worst_med is None or f > worst_med):
            worst_med, worst_state = f, k
    for v in _iter_numbers(p90):
        f = _num(v)
        if f is not None and (worst_p90 is None or f > worst_p90):
            worst_p90 = f

    if worst_med is None and worst_p90 is None:
        if repaired:
            return {"name": name, "status": CAUTION,
                    "detail": ("a phase's stale epoch_window_map.csv was repaired in memory for "
                               "this analysis; the run data on disk is still stale")}
        return {"name": name, "status": NA, "detail": "per-state self-bias unavailable"}

    bits = []
    if worst_med is not None:
        bits.append(f"worst own-restraint self-bias {worst_med:.1f} kT median (state {worst_state})")
    if worst_p90 is not None:
        bits.append(f"p90 {worst_p90:.1f} kT")
    # "~1 kT expected" is the physical claim; "not a proof of correct mapping" is
    # the honest bound on it and belongs in the operator-visible detail, not only
    # in this function's docstring: on the real run a WHOLLY phantom state (every
    # one of its 861,547 samples actually belonged to its neighbour) measured a
    # benign 3.4 kT median, because the two states' secondary centres were only
    # ~5-6 kT apart. A pass here rules out gross mis-mapping, nothing more.
    detail = (", ".join(bits)
              + "; ~1 kT expected for a 2-DOF harmonic (a pass does not prove the mapping is "
                "right -- a mis-mapping between two nearby windows stays inside this band)")
    st = PASS
    if (worst_med is not None and worst_med > SELF_BIAS_MEDIAN_FAIL_KT) or \
       (worst_p90 is not None and worst_p90 > SELF_BIAS_P90_FAIL_KT):
        st = FAIL
        detail += " -- the sample-to-state mapping or those windows' parameters are WRONG"
    elif worst_med is not None and worst_med > SELF_BIAS_MEDIAN_CAUTION_KT:
        st = CAUTION
    if repaired:
        st = _worse(st, CAUTION)
        detail += "; a phase's stale epoch_window_map.csv was repaired in memory for this analysis"
    return {"name": name, "status": st, "detail": detail}


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


def _check_ladder_crosscheck(s: dict) -> dict:
    """Hard-FAIL companion to the "λ-ladder cross-check FAILED" CRITICAL
    warning (see _WARN_RULES below) -- same convention this file already
    uses for the stale-window-map escape hatch and the split overlap graph:
    a CRITICAL warning must be backed by a dedicated check that can move
    `overall`, not left to sit inside an otherwise-PASS verdict. `"skipped"`
    (no λ=0 states, or too few comparable bins) and an absent block (not a
    ladder run at all) both grade NA -- neither is a detected fault.
    """
    name = "λ-ladder cross-check"
    lcc = s.get("ladder_crosscheck")
    if not isinstance(lcc, dict) or "status" not in lcc:
        return {"name": name, "status": NA, "detail": "not a λ-ladder run"}
    status = lcc.get("status")
    diff = _num(lcc.get("max_abs_diff_kcal"))
    tol = _num(lcc.get("tolerance_kcal"))
    nbins = lcc.get("n_bins_compared")
    if status == "fail":
        detail = (f"λ=0-only PMF disagrees with the full-ladder PMF by {diff:.3f} kcal/mol "
                  f"(tolerance {tol:.3f})" if diff is not None and tol is not None
                  else "λ=0-only PMF disagrees with the full-ladder PMF")
        return {"name": name, "status": FAIL, "detail": detail}
    if status == "pass":
        detail = (f"agrees within {diff:.3f} kcal/mol (tolerance {tol:.3f}"
                  + (f", {nbins} bins compared" if nbins is not None else "") + ")"
                  if diff is not None and tol is not None else "agrees")
        return {"name": name, "status": PASS, "detail": detail}
    if status == "skipped":
        return {"name": name, "status": NA, "detail": lcc.get("reason", "skipped")}
    return {"name": name, "status": NA, "detail": f"unrecognized ladder_crosscheck status {status!r}"}


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
    # The operator asked for the CV1 marginal only; not a defect. Must precede
    # the "was NOT computed" HIGH rule below -- first match wins.
    (re.compile(r"[Jj]oint \(CV1, CV2\) window overlap is disabled", re.I), "INFO"),
    # CRITICAL: result may be invalid
    (re.compile(r"did not (fully )?converge", re.I), "CRITICAL"),
    (re.compile(r"[Zz]ero production samples", re.I), "CRITICAL"),
    (re.compile(r"\bESS[^.]*\b(is\s+)?(effectively\s+)?zero\b", re.I), "CRITICAL"),
    # The GAREUS_ALLOW_STALE_WINDOW_MAP escape hatch: this run's samples are
    # provably attributed to the wrong umbrella states, so every free energy in
    # it is invalid -- strictly worse than any "quality concern". Must precede
    # the general [stale window map] rule below (first match wins), and is
    # backed by a hard FAIL in _check_mapping so it cannot sit inside a PASS
    # verdict either.
    (re.compile(r"\[stale window map\][^\n]*(Loaded with the STALE map anyway|"
                r"GAREUS_ALLOW_STALE_WINDOW_MAP is set)", re.I), "CRITICAL"),
    # The lambda-ladder quoting gate failed: the lambda=0-only PMF (plain
    # umbrella sampling, reweighted with the shared global f_k) disagrees
    # with the full-ladder PMF outside tolerance. That is the built-in check
    # that the ladder boost's MBAR reweighting is right at all -- a failure
    # here means every PMF from this run may be wrong, not just this one.
    (re.compile(r"λ-ladder cross-check FAILED", re.I), "CRITICAL"),
    # HIGH: quality concern that biases the PMF
    (re.compile(r"[Ww]eak neighbor CV overlap", re.I), "HIGH"),
    # Mapping-sanity detectors added 2026-08-25 (see
    # docs/chignolin_6_low_ess_root_cause.md). An unmatched warning silently
    # defaults to MEDIUM, which would bury the one warning that says the PMF is
    # attributing samples to the wrong umbrella state.
    (re.compile(r"[Ii]mplausible self-bias", re.I), "HIGH"),
    # The equal-count membership check (loaders_adaptive.py's
    # _consistent_map_membership_notes). Its "could not run" variant must be
    # matched FIRST and graded lower on purpose: an absent check is not a
    # detected fault, and grading the two the same would teach an operator to
    # read the loud one as routine. Neither variant defaults to MEDIUM by
    # accident any more -- the loud one says a phase's own two records of which
    # windows it ran contradict each other, which is this branch's headline
    # failure mode and is backed by a hard FAIL in _check_mapping.
    (re.compile(r"\[window map check\][^\n]*Could not cross-check", re.I), "MEDIUM"),
    (re.compile(r"\[window map check\]", re.I), "HIGH"),
    (re.compile(r"[Ww]eak CV-space nearest-neighbour overlap", re.I), "HIGH"),
    (re.compile(r"[Ww]eak joint \(CV1, CV2\) nearest-neighbour overlap", re.I), "HIGH"),
    # A split overlap graph: MBAR never determined the offset between the
    # components, so every cross-component free-energy difference in the run is
    # arbitrary. HIGH rather than CRITICAL on the same reasoning this file
    # already applies to [cv2 regime change] -- WITHIN a component the free
    # energies stay valid, and the dedicated FAIL in _check_overlap_connectivity
    # is what moves the verdict. It reaches the terminal health block either way.
    (re.compile(r"overlap graph is DISCONNECTED", re.I), "HIGH"),
    # A 2D run whose joint overlap could not be computed is blind on exactly the
    # axis that failed on chignolin_6. Not a PMF bias in itself, but the missing
    # diagnostic is why that failure went unnoticed for a whole campaign, so it
    # is graded with the overlap warnings rather than left to default MEDIUM.
    (re.compile(r"[Jj]oint \(CV1, CV2\) window overlap was NOT computed", re.I), "HIGH"),
    # A repaired-in-memory stale window map: THIS analysis is correct (the
    # mapping was rebuilt from the phase's own surviving window table), but the
    # run data on disk is still stale and every earlier analysis of the affected
    # phases is invalid. Loud, not fatal -- deliberately one band below the
    # escape-hatch rule above.
    (re.compile(r"\[stale window map\]", re.I), "HIGH"),
    # A mid-campaign secondary-CV redefinition: state k is not one Hamiltonian
    # across the switch, so the single pooled MBAR solve is formally invalid
    # across the regime boundary (the note's own words). Triaged on purpose
    # rather than defaulting to MEDIUM; not CRITICAL because within-regime free
    # energies stay valid and CV2-facing plots are already split per regime.
    (re.compile(r"\[cv2 regime change\]", re.I), "HIGH"),
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
