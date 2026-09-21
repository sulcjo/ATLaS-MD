"""S1 gating: coverage, envelope stability, ladder ESS, and graft failure.

Decision points before production: every round must pass all gates or extend swarm members.
All gates are ab initio (no reference structure); graft_gate enforces the seed-library
prerequisite constraint (spec §2, item 5a).

ladder_ess_gate is ADVISORY, not blocking (measured 2026-09-08): a 174-member swarm round's
ladder_ess_gate failed, predicting the top rungs unusable (reweighted ESS 34.1 and 19.2
against a floor of 50, extrapolated_from_rung=5). A 7-rung production probe run at those
exact rungs then measured neighbour overlap 0.919-0.939 across all six gaps (the validated
pilot's own overlap was 0.809-0.902) and passed its lambda=0 cross-check at 0.282 kcal/mol
against a 0.5 tolerance. The gate asks whether the UNBIASED swarm can predict a rung by a
single reweighting jump from lambda=0; production never does that -- it runs MD at every
rung and couples them by exchange. design_lambda_ladder's own docstring already says
extrapolated rungs "must be confirmed by the S3 stage". So ladder_ess_gate keeps ok=True
always and reports its diagnostics (per-rung ESS, extrapolated_from_rung) as warnings
instead of reasons; evaluate_gates surfaces those warnings at the top level so they still
reach swarm_gate.json and the report even though they no longer fail the round.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from gareus.swarm.envelope import pool_member_envelopes


def coverage_gate(plan_rows: list[dict], done_ids: set[int], *, min_done_fraction: float = 0.9) -> dict:
    """Every cell must have ≥1 done member; ≥ min_done_fraction of all members done.

    Args:
        plan_rows: list of dicts with keys member_id, cell_id, etc.
        done_ids: set of member_ids that completed successfully
        min_done_fraction: minimum fraction of members that must be done

    Returns:
        dict with keys "ok" (bool) and "reasons" (list of str)
    """
    reasons: List[str] = []

    # Extract unique cells and track which are empty
    cells = {}
    for row in plan_rows:
        cell_id = row["cell_id"]
        member_id = row["member_id"]
        if cell_id not in cells:
            cells[cell_id] = []
        cells[cell_id].append(member_id)

    # Check that every cell has at least one done member
    empty_cells = []
    for cell_id, members in cells.items():
        if not any(m in done_ids for m in members):
            empty_cells.append(cell_id)

    if empty_cells:
        reasons.append(f"cells with no done members: {empty_cells}")

    # Check that >= min_done_fraction of all members are done
    total_members = len(plan_rows)
    n_done = len(done_ids)
    fraction_done = n_done / total_members if total_members > 0 else 0.0

    if fraction_done < min_done_fraction:
        reasons.append(
            f"only {n_done}/{total_members} members done (fraction {fraction_done:.3f} < min {min_done_fraction:.3f})"
        )

    ok = len(reasons) == 0
    return {"ok": ok, "reasons": reasons}


def envelope_stability_gate(
    traces: Dict[int, Dict[str, np.ndarray]], discard: int, *, sigma_rel_tol: float = 0.10, extrema_sigma_tol: float = 1.0
) -> dict:
    """Odd vs even member halves must agree on σ_V and extrema within tolerance.

    Pools odd and even member ids separately and compares:
    - sigmav relative difference
    - |Vmax_a - Vmax_b| and |Vmin_a - Vmin_b| in units of pooled sigmav

    A gate that cannot be evaluated (empty half, pooling error) FAILS CLOSED: appends a reason
    and returns ok=False.

    Args:
        traces: Dict[member_id, {"v_pep_kj": array, "v_dih_kj": array}]
        discard: frames to discard before pooling
        sigma_rel_tol: maximum allowed relative difference in sigma
        extrema_sigma_tol: maximum allowed extrema difference in units of sigma

    Returns:
        dict with keys "ok" (bool) and "reasons" (list of str)
    """
    reasons: List[str] = []

    # Split members into odd and even
    odd_members = {m: tr for m, tr in traces.items() if m % 2 == 1}
    even_members = {m: tr for m, tr in traces.items() if m % 2 == 0}

    # Pool each half
    for grp_name, (odd_traces, even_traces) in [
        ("Total", (odd_members, even_members)),
        ("Dihedral", (odd_members, even_members)),
    ]:
        # Check if either half is empty (fail-closed)
        if not odd_traces or not even_traces:
            reasons.append(
                f"{grp_name}: cannot evaluate stability (odd half has {len(odd_traces)} members, "
                f"even half has {len(even_traces)} members)"
            )
            continue

        # Try to pool both halves (fail-closed on pooling error)
        try:
            env_odd = pool_member_envelopes(odd_traces, discard)[grp_name]
            env_even = pool_member_envelopes(even_traces, discard)[grp_name]
        except (ValueError, KeyError) as exc:
            reasons.append(f"{grp_name}: pooling failed: {exc}")
            continue

        # Compare sigma relative difference
        sigma_mean = (env_odd.sigmav + env_even.sigmav) / 2.0
        if sigma_mean > 0:
            sigma_rel_diff = abs(env_odd.sigmav - env_even.sigmav) / sigma_mean
            if sigma_rel_diff > sigma_rel_tol:
                reasons.append(
                    f"{grp_name}: σ_V relative difference {sigma_rel_diff:.3f} > {sigma_rel_tol:.3f} "
                    f"(odd={env_odd.sigmav:.3f}, even={env_even.sigmav:.3f})"
                )

        # Compare extrema in units of pooled sigma
        pooled_sigma = (env_odd.sigmav + env_even.sigmav) / 2.0
        if pooled_sigma > 0:
            vmax_diff = abs(env_odd.vmax - env_even.vmax) / pooled_sigma
            vmin_diff = abs(env_odd.vmin - env_even.vmin) / pooled_sigma

            if vmax_diff > extrema_sigma_tol:
                reasons.append(
                    f"{grp_name}: Vmax difference {vmax_diff:.3f} σ > {extrema_sigma_tol:.3f} σ "
                    f"(odd={env_odd.vmax:.1f}, even={env_even.vmax:.1f})"
                )

            if vmin_diff > extrema_sigma_tol:
                reasons.append(
                    f"{grp_name}: Vmin difference {vmin_diff:.3f} σ > {extrema_sigma_tol:.3f} σ "
                    f"(odd={env_odd.vmin:.1f}, even={env_even.vmin:.1f})"
                )

    ok = len(reasons) == 0
    return {"ok": ok, "reasons": reasons}


def ladder_ess_gate(ladder: dict, *, ess_floor: int = 50) -> dict:
    """Report (never fail on) extrapolated rungs and sub-floor ESS -- ADVISORY (see the
    module docstring for the 2026-09-08 measurement that justifies this).

    Args:
        ladder: dict returned by design_lambda_ladder with keys lambdas, ess_per_rung, extrapolated_from_rung
        ess_floor: minimum effective sample size

    Returns:
        dict with keys "ok" (always True), "reasons" (always empty -- kept for schema
        parity with the other gates), and "warnings" (list of str: the same diagnostics
        this gate used to fail the round on)
    """
    warnings: List[str] = []

    # Check extrapolation
    if ladder.get("extrapolated_from_rung") is not None:
        rung = ladder["extrapolated_from_rung"]
        warnings.append(f"ladder is extrapolated from rung {rung}; rungs {rung} onward cannot be trusted")

    # Check ESS floor
    ess_per_rung = ladder.get("ess_per_rung", [])
    for i, ess in enumerate(ess_per_rung):
        if ess < ess_floor:
            warnings.append(f"rung {i}: ESS {ess:.1f} < floor {ess_floor}")

    return {"ok": True, "reasons": [], "warnings": warnings}


def graft_gate(done_summaries: list[dict], *, max_fallback_fraction: float = 0.10) -> dict:
    """Fraction of members with status != "ok" must be <= max_fallback_fraction.

    Args:
        done_summaries: list of done.json dicts, each with a "status" field
        max_fallback_fraction: maximum fraction of members allowed to fail (graft or MD)

    Returns:
        dict with keys "ok" (bool), "reasons" (list of str), and counts: n_graft_failed, n_md_failed, failed_fraction
    """
    reasons: List[str] = []

    n_graft_failed = 0
    n_md_failed = 0
    n_total = len(done_summaries)

    for summary in done_summaries:
        status = summary.get("status", "unknown")
        if status == "graft_failed":
            n_graft_failed += 1
        elif status == "md_failed":
            n_md_failed += 1

    failed_fraction = (n_graft_failed + n_md_failed) / n_total if n_total > 0 else 0.0

    if failed_fraction > max_fallback_fraction:
        reasons.append(
            f"member failure fraction {failed_fraction:.3f} > {max_fallback_fraction:.3f} "
            f"({n_graft_failed} graft_failed, {n_md_failed} md_failed out of {n_total})"
        )

    ok = len(reasons) == 0
    return {
        "ok": ok,
        "reasons": reasons,
        "n_graft_failed": n_graft_failed,
        "n_md_failed": n_md_failed,
        "failed_fraction": failed_fraction,
    }


def pair_gate(selection: dict, *, fallback: str) -> dict:
    """Did automatic CV2 selection leave the run with a deployable pair, or an honest 1-D fallback?

    ``pair``: ok. ``cv1_only`` (a deployable anchor, but no component passed) is ok only
    when the operator declared ``fallback = "cv1_only"``. ``no_deployable_anchor`` is never
    ok: running the CV the selector just certified as resolving too few windows is the
    exact failure the design names as the binding constraint, and no fallback buys it.
    """
    status = selection.get("status")
    ok = status == "pair" or (status == "cv1_only" and fallback == "cv1_only")
    if status == "no_deployable_anchor":
        reasons = [f"the configured anchor is not deployable ({'; '.join(selection.get('anchor_reasons', []))}); "
                   "no fallback runs an anchor the selector rejected"]
    elif not ok:
        reasons = [f"cv pair selection returned {status!r} with fallback={fallback!r}"]
    else:
        reasons = []
    note = None
    if ok and status == "cv1_only":
        note = f"automatic CV2 selection found no deployable component ({selection.get('selection_reason')}); running CV1 alone as configured"
    return {"ok": ok, "status": status, "fallback": fallback, "reasons": reasons,
            "warnings": [note] if note else []}


def coverage_design_gate(layout_plan: dict) -> dict:
    """The layout must be PROPOSED (mandatory stacks fit the cap) and leave no discovered
    region unresolved (spec F05). A rare region is a reason to fail, never to delete."""
    status = str((layout_plan or {}).get("status", ""))
    unresolved = list((layout_plan or {}).get("unresolved") or [])
    coverage = (layout_plan or {}).get("region_coverage") or {}
    uncovered = sorted(rid for rid, info in coverage.items() if info.get("unresolved"))
    reasons = []
    if status != "PROPOSED":
        reasons.append(f"layout status {status or 'absent'}")
    if unresolved:
        reasons.append("regions without an eligible seed: " + "; ".join(unresolved))
    if uncovered:
        reasons.append("regions without a design representative: " + ", ".join(uncovered))
    return {"ok": not reasons, "status": status, "reasons": reasons, "warnings": [],
            "mandatory_state_ids": list((layout_plan or {}).get("mandatory_state_ids") or [])}


def evaluate_gates(
    plan_rows: list[dict],
    done_ids: set[int],
    traces: Dict[int, Dict[str, np.ndarray]],
    discard: int,
    ladder: dict,
    done_summaries: list[dict],
    *,
    min_done_fraction: float = 0.9,
    sigma_rel_tol: float = 0.10,
    extrema_sigma_tol: float = 1.0,
    ladder_ess_floor: int = 50,
    max_graft_fallback_fraction: float = 0.10,
    selection: dict | None = None,
    pair_fallback: str = "cv1_only",
    layout_plan: dict | None = None,
) -> dict:
    """Evaluate all gates; return aggregate status and reasons.

    Args:
        plan_rows: list of plan members
        done_ids: set of member_ids that completed successfully
        traces: Dict[member_id, {"v_pep_kj": array, "v_dih_kj": array}]
        discard: frames to discard before pooling
        ladder: dict returned by design_lambda_ladder
        done_summaries: list of done.json dicts
        min_done_fraction: threshold for coverage_gate
        sigma_rel_tol: threshold for envelope_stability_gate
        extrema_sigma_tol: threshold for envelope_stability_gate
        ladder_ess_floor: threshold for ladder_ess_gate
        max_graft_fallback_fraction: threshold for graft_gate

    Returns:
        dict with keys "status" ("pass"|"fail"), "gates" (dict of gate results),
        "reasons" (list of reasons from failed gates), and "warnings" (list of
        non-blocking diagnostics from any gate, e.g. ladder_ess_gate's advisory
        ESS/extrapolation findings -- collected regardless of that gate's own ok
        status, so nothing is silently lost).
    """
    gates = {
        "coverage": coverage_gate(plan_rows, done_ids, min_done_fraction=min_done_fraction),
        "envelope_stability": envelope_stability_gate(traces, discard, sigma_rel_tol=sigma_rel_tol, extrema_sigma_tol=extrema_sigma_tol),
        "ladder_ess": ladder_ess_gate(ladder, ess_floor=ladder_ess_floor),
        "graft": graft_gate(done_summaries, max_fallback_fraction=max_graft_fallback_fraction),
    }
    if selection is not None:
        gates["pair"] = pair_gate(selection, fallback=pair_fallback)
    if layout_plan is not None:
        gates["coverage_design"] = coverage_design_gate(layout_plan)

    # Aggregate reasons from all failed gates
    all_reasons = []
    all_warnings = []
    for gate_name, gate_result in gates.items():
        if not gate_result.get("ok", True):
            all_reasons.extend(gate_result.get("reasons", []))
        all_warnings.extend(gate_result.get("warnings", []))

    status = "pass" if all(g.get("ok", True) for g in gates.values()) else "fail"

    return {
        "status": status,
        "gates": gates,
        "reasons": all_reasons,
        "warnings": all_warnings,
    }


def extension_plan(plan_meta: dict, gate: dict) -> dict:
    """Determine how many extra replicates to run based on gate failures.

    Rules:
    - Stability failure: double R (extra_replicates_per_cell)
    - Coverage failure: +1
    - Cap at 4× base_replicates_per_cell (or 4× current_r if base not present)

    ladder_ess is advisory (see the module docstring): evaluate_gates's real output never
    sets ladder_ess["ok"] to False, so it never reaches the extension logic below in
    practice; the check is kept only for a hand-built gate dict that still sets it.

    Args:
        plan_meta: dict with keys "replicates_per_cell" and optional "base_replicates_per_cell"
        gate: dict returned by evaluate_gates with keys "status", "gates"

    Returns:
        dict with keys "extra_replicates_per_cell" (int) and "reason" (str)
    """
    current_r = plan_meta.get("replicates_per_cell", 1)
    base_r = plan_meta.get("base_replicates_per_cell", current_r)
    gates = gate.get("gates", {})

    extra = 0
    reasons = []

    # Check stability/ESS failures
    if not gates.get("envelope_stability", {}).get("ok", True):
        extra = current_r  # double
        reasons.append("envelope stability failed")

    if not gates.get("ladder_ess", {}).get("ok", True):
        extra = max(extra, current_r)  # double if not already doubled
        reasons.append("ladder ESS failed")

    # Check coverage failure
    if not gates.get("coverage", {}).get("ok", True):
        extra += 1
        reasons.append("coverage failed")

    # Cap at 4× base_replicates_per_cell
    max_r = 4 * int(base_r)
    if current_r + extra > max_r:
        extra = max_r - current_r

    reason_str = "; ".join(reasons) if reasons else "all gates passed"

    return {
        "extra_replicates_per_cell": extra,
        "reason": reason_str,
    }
