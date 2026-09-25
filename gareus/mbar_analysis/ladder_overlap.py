"""Overlap along each axis of a (CV1 centre, λ rung) state grid.

A ladder run is two-dimensional even when cv2 is "none": states differ by CV1
centre and by rung. Collapsing both into one CV1-marginal number reports
whichever axis happens to dominate and hides the other -- chignolin_7 epoch 0
measured 0.93-0.98 along λ and 0.35-0.69 across CV1 boundaries, and only the
second is a problem.
"""
from __future__ import annotations

import logging

import numpy as np

from gareus.mbar_analysis.ladder import symmetric_state_overlap
from gareus.mbar_analysis.pmf import overlap_components

_TOL = 1.0e-9
# One warning per process for the full-matrix (pair_overlap=None) path; reset in tests.
_FULL_MATRIX_WARNED = False

# Threshold for the SYMMETRISED PAIRWISE MBAR STATE-OVERLAP metric this module
# grades -- NOT for a CV histogram-intersection overlap, and NOT for a raw
# entry of the full-union ``mbar_state_overlap`` matrix either (see
# ``ladder_overlap_by_axis``'s ``pair_overlap`` argument below). The CV
# histogram-intersection quantity is a different scale entirely and not
# interchangeable: gareus.mbar_analysis.pmf says so explicitly of
# overlap_matrix.csv vs --min-neighbor-overlap, and this module used to
# default to that CV1-marginal target (0.30), against which no pair of
# chignolin_7's 64 states cleared on either axis while the CV1-marginal check
# on the same axis simultaneously passed at 0.491/connected.
#
# 0.15 is the project's own number for this metric, not a new one:
# AdaptivePolicy.min_rung_overlap (gareus/adaptive_production.py) gates rung
# edges at 0.15 with target_rung_overlap 0.25, taken from the S3 pilot's
# adjacent-rung entries 0.298/0.250/0.240/0.273 (docs/superpowers/specs/
# 2026-09-07-adaptive-ladder-rungs-design.md:18). Using it here makes the
# analysis report agree with the driver gate on the same measurement.
#
# That pilot was a 5-state ladder read off its FULL 5-state overlap matrix,
# which is itself diluted; 0.15 is applied to the PAIRWISE metric this module
# grades but has NOT been re-measured on it. For scale only: RUNS/chignolin_7's
# 64-state union (16 CV1 centres x 4 rungs) measures a pairwise median of
# 0.258 across its 48 adjacent-rung edges (0/48 below 0.15) when each edge is
# scored with ``gareus.mbar_analysis.ladder.pairwise_state_overlap`` (union
# f_k held fixed, only the pair's own samples in the denominator), against a
# full-union median of only 0.089 (38/48 below 0.15) on the SAME edges --
# diluted by roughly how many other states share each edge's region, not a
# real overlap difference -- which is why ``ladder_overlap_by_axis`` should
# be given a pairwise ``pair_overlap`` for its neighbour-edge/threshold rows.
#
# Caveat, deliberately not encoded: that calibration is a RUNG calibration,
# measured on a 1-D 5-state single-centre ladder. The cv1_direction rows reuse
# it as the nearest calibration for the same metric -- far better than
# borrowing a different metric's target, but still a reuse across axes (rung
# vs CV1), independent of the pairwise-vs-full-matrix fix above.
LADDER_STATE_OVERLAP_MIN = 0.15

_EMPTY_AXIS = {"pairs": [], "worst": None, "worst_pair": None, "n_pairs": 0,
               "n_components": 0, "connected": None, "expected_components": None}


def _empty_axes() -> dict:
    return {"lambda_direction": dict(_EMPTY_AXIS), "cv1_direction": dict(_EMPTY_AXIS)}


def _summarise(pairs):
    if not pairs:
        return dict(_EMPTY_AXIS)
    worst = min(pairs, key=lambda p: p[2])
    return {
        "pairs": [(int(a), int(b), float(v)) for a, b, v in pairs],
        "worst": float(worst[2]),
        "worst_pair": (int(worst[0]), int(worst[1])),
        "n_pairs": len(pairs),
    }


def _eligible_mask(K: int, n_k) -> np.ndarray:
    if n_k is not None:
        arr = np.asarray(n_k).ravel()
        if arr.size >= K:
            return arr[:K] > 0
    return np.ones(K, dtype=bool)


def _axis_connectivity(K: int, pairs, group_values: np.ndarray,
                        eligible: np.ndarray, thr: float, n_k) -> dict:
    """Connected-components verdict for one axis, using ONLY that axis' own
    already-symmetrised pairs as edges -- never the other axis' pairs, never
    the raw asymmetric ``overlap[i, j]``.

    A lambda-direction pair always shares a CV1 centre; a CV1-direction pair
    always shares a rung -- so an edge never crosses groups, and a connected
    component can therefore never span two groups either. The best
    achievable outcome is exactly one component per eligible group, NOT one
    component overall: with more than one CV1 centre (the normal case --
    chignolin_7 epoch 0 has many), the lambda-direction graph is a forest of
    per-centre chains and can never collapse to a single component no matter
    how good the overlap is. Grading ``n_components == 1`` would FAIL every
    such run trivially, which defeats the check. ``connected`` is instead
    ``n_components == (number of distinct eligible groups)``: true iff every
    group's own chain is internally bridged.

    ``pairs == []`` (no adjacent pair with a differing λ/centre exists at
    all -- e.g. a single-rung ladder, or a non-ladder-shaped input) grades
    NA, matching the pairwise check's own "no adjacent pairs on this axis"
    NA for the same axis, rather than the vacuous FAIL a bare component count
    over singleton nodes would otherwise report.
    """
    if not pairs:
        return {"n_components": 0, "connected": None, "expected_components": None}
    mat = np.zeros((K, K), dtype=np.float64)
    for a, b, v in pairs:
        mat[a, b] = mat[b, a] = v
    np.fill_diagonal(mat, 1.0)
    comp = overlap_components(mat, float(thr), n_k)
    n_components = int(comp.get("n_components", 0) or 0)
    if n_components == 0:
        return {"n_components": 0, "connected": None, "expected_components": None}
    excluded = {int(i) for i in (comp.get("excluded_unsampled_states") or [])}
    expected = len({float(np.round(group_values[i], 9))
                    for i in range(K) if eligible[i] and i not in excluded})
    expected = max(expected, 1)
    return {"n_components": n_components, "connected": bool(n_components == expected),
            "expected_components": expected}


def _warn_full_matrix_once() -> None:
    global _FULL_MATRIX_WARNED
    if _FULL_MATRIX_WARNED:
        return
    _FULL_MATRIX_WARNED = True
    logging.warning(
        "ladder_overlap_by_axis: no pair_overlap given, grading FULL-matrix overlap entries against "
        "the pairwise-scale threshold %.2f; the full matrix is diluted on a large union (chignolin_7: "
        "median 0.089 full vs 0.258 pairwise). Pass pair_overlap (pairwise_state_overlap) for a real "
        "verdict.", LADDER_STATE_OVERLAP_MIN)


def ladder_overlap_by_axis(overlap, state_lambdas, centers,
                            thr: float = LADDER_STATE_OVERLAP_MIN, n_k=None,
                            pair_overlap=None):
    """Split neighbour overlaps into the λ direction and the CV1 direction,
    and grade each axis' own bridging.

    λ-direction pairs share a CV1 centre and are adjacent in sorted λ.
    CV1-direction pairs share a rung and are adjacent in sorted centre.

    Each pair's value comes from ``pair_overlap(a, b)`` if given, else from
    ``symmetric_state_overlap(overlap, a, b)`` -- i.e. ``sqrt(O_ab * O_ba)`` --
    never the raw ``overlap[a, b]``. ``O`` is asymmetric whenever the two
    states' sample counts differ (unequal ``n_k`` is the normal case under
    adaptive extension, not the exception; see
    ``gareus.mbar_analysis.ladder.symmetric_state_overlap``), so the raw
    entry would make a reported "worst" pair depend on which state happened
    to come first in the pair -- not on anything physical.

    ``pair_overlap``, when given, is a ``(a, b) -> Optional[float]`` callable
    used INSTEAD of indexing ``overlap`` (which may then be ``None``) --
    normally a closure over
    ``gareus.mbar_analysis.ladder.pairwise_state_overlap(u_nk, window, f_k,
    n_k, a, b)``. This module's neighbour-edge/threshold rows are meant to be
    graded on the PAIRWISE metric, not a raw entry of the full-union
    ``mbar_state_overlap`` matrix, which dilutes an edge's overlap by roughly
    how many OTHER states share its region (see ``LADDER_STATE_OVERLAP_MIN``'s
    module-level comment for the chignolin_7 numbers). ``overlap`` stays a
    real, separate parameter -- not merely kept for backward compatibility --
    because nothing in this function actually needs the FULL K x K matrix:
    every value it reads is one ``(a, b)`` pair the axis-splitting loops
    below already identified, and the connectivity check
    (``_axis_connectivity``) is built from that same already-symmetrised
    pair list, never from ``overlap`` itself. Omitting ``pair_overlap``
    (the default) keeps the original full-matrix-indexing behaviour, still
    exercised directly by this module's own unit tests, which construct
    ``overlap`` by hand rather than from real per-sample MBAR data.

    Returns ``(summary, warnings)``. ``summary`` has one key per axis
    (``lambda_direction``, ``cv1_direction``), each carrying the pairwise
    diagnostic (``pairs``, ``worst``, ``worst_pair``, ``n_pairs``) and the
    connectivity diagnostic (``n_components``, ``connected``,
    ``expected_components`` -- see ``_axis_connectivity``). ``warnings`` is a
    list of human-readable strings, empty on the ordinary path -- same
    degrade-with-a-named-warning convention as
    ``gareus.mbar_analysis.boost_report.gamd_boost_by_rung_report``, rather
    than raising or leaking a bare exception to the caller's generic
    ``except``.

    Degrades to an all-NA/empty summary (never raises) when:

    - ``state_lambdas is None`` -- reachable: the CONTRADICTION case in
      ``gareus.mbar_analysis.crosscheck`` (``gamd_ladder`` asserted while a
      loader dropped ``state_lambdas``) leaves exactly this on ``d``.
    - ``state_lambdas``/``centers`` disagree in length, or (when
      ``pair_overlap`` is not given) ``overlap``'s shape disagrees with
      either -- also independently reachable: ``gareus.mbar_analysis.data``'s
      window reader can silently drop a state with a blank centre, making
      ``len(centers) < K`` true on its own.
    """
    if state_lambdas is None:
        return _empty_axes(), [
            "ladder-overlap axis report skipped: state_lambdas is None "
            "(a loader dropped per-state lambda -- see the CONTRADICTION "
            "case in gareus.mbar_analysis.crosscheck)"
        ]

    lam = np.asarray(state_lambdas, dtype=np.float64)
    cen = np.asarray(centers, dtype=np.float64)
    K = int(lam.shape[0])
    if pair_overlap is None:
        _warn_full_matrix_once()
        overlap = np.asarray(overlap, dtype=np.float64)
        overlap_ok = overlap.ndim == 2 and overlap.shape[0] == overlap.shape[1] and overlap.shape[0] == K
        if not overlap_ok or cen.shape[0] != K:
            return _empty_axes(), [
                "ladder-overlap axis report skipped: array length mismatch "
                f"(overlap {overlap.shape}, state_lambdas {lam.shape}, centers {cen.shape})"
            ]
        pair_overlap = lambda a, b: symmetric_state_overlap(overlap, a, b)  # noqa: E731
    elif cen.shape[0] != K:
        return _empty_axes(), [
            "ladder-overlap axis report skipped: array length mismatch "
            f"(state_lambdas {lam.shape}, centers {cen.shape})"
        ]

    lam_pairs, cv1_pairs = [], []
    for value in np.unique(np.round(cen, 9)):
        idx = np.flatnonzero(np.abs(cen - value) < _TOL)
        order = idx[np.argsort(lam[idx])]
        for a, b in zip(order[:-1], order[1:]):
            if abs(lam[b] - lam[a]) > _TOL:
                v = pair_overlap(int(a), int(b))
                if v is not None:
                    lam_pairs.append((a, b, v))
    for value in np.unique(np.round(lam, 9)):
        idx = np.flatnonzero(np.abs(lam - value) < _TOL)
        order = idx[np.argsort(cen[idx])]
        for a, b in zip(order[:-1], order[1:]):
            if abs(cen[b] - cen[a]) > _TOL:
                v = pair_overlap(int(a), int(b))
                if v is not None:
                    cv1_pairs.append((a, b, v))

    eligible = _eligible_mask(K, n_k)
    lam_summary = _summarise(lam_pairs)
    cv1_summary = _summarise(cv1_pairs)
    lam_summary.update(_axis_connectivity(K, lam_pairs, cen, eligible, thr, n_k))
    cv1_summary.update(_axis_connectivity(K, cv1_pairs, lam, eligible, thr, n_k))
    return {"lambda_direction": lam_summary, "cv1_direction": cv1_summary}, []


def ladder_overlap_health_checks(lo: dict, thr: float) -> list:
    """Build the RESULT-HEALTH rows for one ``ladder_overlap_by_axis()``
    summary: one worst-pair-overlap row and one connectivity row per axis (4
    rows total).

    Isolated from the report-writer wiring in analyze_gareus_mbar.py so it is
    directly unit-testable -- same pattern as
    ``gareus.mbar_analysis.boost_report.gamd_boost_by_rung_report``. The
    caller is responsible for appending these onto ``s['health']['checks']``
    and then recomputing ``s['health']['overall']`` (via
    ``gareus_report.overall_from_checks``) so the banner can never disagree
    with a row it is now showing.

    The connectivity row's severity mirrors
    ``gareus_report._check_overlap_connectivity`` deliberately, not a bare
    component count: a split is graded on the axis' own weakest link
    (``ax['worst']`` -- already the minimum symmetrised overlap among that
    axis' pairs, the same number the pairwise row above already reports),
    never better than CAUTION (a split is a split) and FAIL only when that
    link is below ``OVERLAP_FAIL_FRACTION`` of the target -- otherwise a
    single pair sitting just under threshold (already a CAUTION on the
    pairwise row) would double as a hard FAIL on the connectivity row for
    the exact same, mildly-under-target reason. An unknown weakest link
    (``ax['worst'] is None`` while still split -- not reachable through
    ``ladder_overlap_by_axis`` today, since ``connected`` is only ever False
    when at least one real pair exists, but guarded for robustness) grades
    FAIL: a split whose severity cannot be measured is not evidence of a
    mild one, same convention as the module it mirrors.
    """
    from gareus_report import PASS, CAUTION, FAIL, NA, OVERLAP_FAIL_FRACTION
    checks = []
    for axis_key, label in (("lambda_direction", "Overlap along λ"),
                             ("cv1_direction", "Overlap across CV1")):
        ax = lo[axis_key]
        if ax.get("worst") is None:
            status, detail = NA, "no adjacent pairs on this axis"
        else:
            w = ax["worst"]; a, b = ax["worst_pair"]
            status = FAIL if w < OVERLAP_FAIL_FRACTION * thr else (CAUTION if w < thr else PASS)
            detail = f"worst {w:.3f} (states {a}-{b})"
        checks.append({"name": label, "status": status, "detail": detail})
    for axis_key, label in (("lambda_direction", "Connectivity along λ"),
                            ("cv1_direction", "Connectivity across CV1")):
        ax = lo[axis_key]
        n_comp = ax.get("n_components") or 0
        connected = ax.get("connected")
        if n_comp == 0:
            status, detail = NA, "no states graded on this axis"
        elif connected:
            status = PASS
            detail = f"{n_comp} group(s) each internally bridged"
        else:
            exp = ax.get("expected_components")
            w = ax.get("worst")
            if w is None:
                status = FAIL
                detail = f"split into {n_comp} components (expected {exp}); cut-edge overlap unknown"
            else:
                status = FAIL if w < OVERLAP_FAIL_FRACTION * thr else CAUTION
                detail = f"split into {n_comp} components (expected {exp}); weakest link {w:.3f}"
        checks.append({"name": label, "status": status, "detail": detail})
    return checks
