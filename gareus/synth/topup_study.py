"""Synthetic A/B study of effective top-ups (spec 6.1-6.2): uniform extension vs deficit-driven top-ups.

No MD. Each campaign drives the REAL top-up allocator (``plan_topup``), calibration
(``update_after_topup``), throughput model (``wall_hours``), layout graphs
(``build_geometry_edges``, ``layout_neighbours``) and union-MBAR diagnostics
(``union_diagnostics_from_npz``) over an analytic 2D landscape with a lambda-rung ladder.

Model (all choices fixed up front, identical for both arms):

* Ladder: a uniform 2D grid over 5-95 % of each CV axis, k = 50 kBT/CV^2 on both axes
  (the harness's reference spring, ``ess.K_REF``), centres 2 restraint widths apart,
  crossed with rungs lambda in {0, 1/3, 2/3, 1}. Every state samples the density
  exp(-(F + U_w + dV_lambda)) (``sampler.boost_dv``); its reduced potential in the union
  NPZ is U_w + dV_lambda (F is common and cancels).
* Mixing: tau from ``ess.tau_int`` with ``n_partners`` = the window's spatial + rung
  partners present in its segment. A segment advances all its windows by the same
  length L; each gains L / (report_interval * (1 + 2 tau)) decorrelated samples.
* Draws are common random numbers: window w's i-th decorrelated sample is the same in
  both arms (blocked per-window streams), so arms differ only in how many each state gets.
* Cost: ``wall_hours`` with the policy's default throughput table, 4 GPUs, 4 fs,
  report interval 5000 steps (the driver's defaults); kT = 0.596 kcal/mol (300 K).
* Budget: the campaign's wall hours are split evenly over 3 epochs. Arm A ("uniform")
  runs every epoch as one all-state segment. Arm B ("topup") runs a (1 - cap) baseline,
  then the allocator's single top-up with budget cap * epoch hours (as
  ``_topup_plan_for_phase``); unused hours roll into the next epoch (the live pool),
  and both arms spend what is left after epoch 3 in one final all-state segment, so the
  arms use equal modelled wall hours up to one 1000-step all-state quantum (the driver's
  ``_quantized_extra_steps``). Arm B carries the calibration state (correction, edge
  attempts, warm-start f) across epochs like the driver.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .ess import tau_int
from .landscapes import LANDSCAPES, Landscape
from .topup_model import (KT_KCAL, K_WINDOW, N_EPOCHS, N_GPUS, REPORT_INTERVAL, RUNGS, TIMESTEP_FS,  # noqa: F401
                          _Campaign, _diagnose, _ladder, _layout, _partners_in, _pmf_rmse,
                          _reduced_potentials, _sample_segment, _Streams, _write_union_npz)
from .topup_regime import DEFICIT_BAND, MIN_ROWS, bridge_budget, calibrate_regime

HETEROGENEOUS = ("rugged-2d", "gated-barrier", "slow-cv2-double-branch")
HOMOGENEOUS = ("harmonic-bowl",)
BRIDGE_LANDSCAPE = "gated-barrier"
SEGMENT_QUANTUM = 1000


@dataclass(frozen=True)
class ArmResult:
    max_sigma: float
    pmf_rmse: float
    hours_used: float
    topup_md_fraction: float
    structural_routed: tuple
    n_states: int = 0
    n_sigma_nan: int = 0
    epochs: tuple = ()
    deficit_fraction: float = math.nan     # states with sigma above target in the final union solve
    median_rows: float = math.nan          # median decorrelated rows per state over the campaign
    zero_rows_epoch0: int = 0              # states with no row when epoch 0's diagnostics would run
    zero_rows_final: int = 0
    routed_coords: tuple = ()              # structural edges keyed by ((c1, c2, lam), (c1, c2, lam))
    topped_then_routed: int = 0            # edges topped as noise in an earlier epoch that end structural
    g_ratio: float = math.nan              # mean g(patch) / g(all-state segment) over topped-up states



def _plan_record(plan, diag_after) -> dict:
    rec = {"reason": plan.reason, "steps": int(plan.steps), "cost_hours": float(plan.cost_hours),
           "patch": len(plan.state_ids), "deficits": list(plan.deficit_state_ids),
           "partners": list(plan.partner_state_ids), "structural_edges": [list(e) for e in plan.structural_edges],
           "weak_edges_topped": [list(e) for e in plan.weak_edges_topped],
           "sigma_before": {str(k): v for k, v in plan.sigma_before.items() if k in plan.deficit_state_ids},
           "predicted_sigma": {str(k): v for k, v in plan.predicted_sigma.items() if k in plan.deficit_state_ids}}
    if diag_after is not None:
        rec["realised_sigma"] = {str(k): diag_after.sigma_kcal.get(k) for k in plan.deficit_state_ids}
    return rec


def run_arm(landscape: Landscape, *, arm: str, seed: int, wall_hours_budget: float,
            remove_bridge: bool = False, n_epochs: int = N_EPOCHS) -> ArmResult:
    """One synthetic campaign; ``arm`` is ``"uniform"`` (A) or ``"topup"`` (B)."""
    from ..adaptive.throughput import wall_hours
    from ..adaptive.topup_allocator import plan_topup
    from ..adaptive.topup_state import update_after_topup
    if arm not in ("uniform", "topup"):
        raise ValueError(f"unknown arm {arm!r}")
    windows = _ladder(landscape, remove_bridge)
    layout = _layout(windows)
    edges, neighbours, rung_partners, policy = layout
    table, cap = policy.topup_throughput_table, float(policy.topup_max_fraction)
    ids = list(range(len(windows)))
    per_step_all = wall_hours(1, len(ids), TIMESTEP_FS, N_GPUS, table)
    full_np = _partners_in(ids, neighbours, rung_partners)
    camp, streams = _Campaign(len(ids)), _Streams(landscape, windows, seed)
    state = {"correction": {}, "edge_attempts": {}, "f_kT": {}, "wall_time": []}
    carry, structural, epochs = 0.0, set(), []
    topped, g_ratios, zero0 = set(), [], None
    tau_g = {w: 1.0 + 2.0 * tau_int(landscape, windows[w], n_partners=full_np[w]) for w in ids}

    def all_state_segment(hours: float) -> float:
        # floor to the driver's segment quantum (_quantized_extra_steps: 1000 steps), never overspending
        steps = int(hours / per_step_all // SEGMENT_QUANTUM) * SEGMENT_QUANTUM
        return _sample_segment(landscape, windows, ids, steps, streams, full_np, camp, table) if steps else 0.0

    with tempfile.TemporaryDirectory() as tmp:
        for e in range(n_epochs):
            hours = wall_hours_budget / n_epochs + carry
            spent = all_state_segment(hours if arm == "uniform" else (1.0 - cap) * hours)
            rec = {"epoch": e, "hours": hours, "baseline_hours": spent}
            if zero0 is None:
                zero0 = int(np.sum(camp.kept == 0))
            if arm == "topup":
                diag = _diagnose(tmp, landscape, camp, windows, layout, state["f_kT"])
                plan = plan_topup(diag, state_ids_in_order=ids, neighbours=neighbours, rung_partners=rung_partners,
                                  policy=policy, report_interval=REPORT_INTERVAL, timestep_fs=TIMESTEP_FS,
                                  n_gpus=N_GPUS, budget_hours=cap * hours, correction=state["correction"],
                                  edge_attempts=state["edge_attempts"])
                if diag is not None:
                    state["f_kT"] = dict(diag.f_kT)
                    rec["sigma_baseline"] = _sigma_summary(diag)
                structural.update(tuple(x) for x in plan.structural_edges)
                last_structural = {tuple(x) for x in plan.structural_edges}
                after = None
                if plan.reason == "planned":
                    np_patch = _partners_in(plan.state_ids, neighbours, rung_partners)
                    g_ratios += [(1.0 + 2.0 * tau_int(landscape, windows[w], n_partners=np_patch[w])) / tau_g[w]
                                 for w in plan.state_ids]
                    h = _sample_segment(landscape, windows, plan.state_ids, plan.steps, streams, np_patch, camp, table)
                    spent += h
                    camp.topup_hours += h
                    after = _diagnose(tmp, landscape, camp, windows, layout, state["f_kT"])
                    if after is not None:
                        state = update_after_topup(state, plan, after.sigma_kcal)
                        state["f_kT"] = dict(after.f_kT)
                rec["plan"] = _plan_record(plan, after)
                rec["topped_earlier_now_structural"] = len(topped & last_structural)
                topped.update(tuple(x) for x in plan.weak_edges_topped)
            carry = hours - spent
            epochs.append(rec)
        all_state_segment(carry)
        final = _diagnose(tmp, landscape, camp, windows, layout, state["f_kT"])
    if final is None:
        raise RuntimeError("final union-MBAR solve failed; the campaign has no result")
    sig = np.array([final.sigma_kcal.get(k, math.nan) for k in ids], dtype=float)
    if np.all(~np.isfinite(sig)):
        raise RuntimeError("final union-MBAR solve gave no finite sigma")
    target = float(policy.topup_target_sigma)

    def coord(w: int) -> tuple:
        return (round(windows[w].center1, 6), round(windows[w].center2, 6), round(windows[w].lam, 6))

    return ArmResult(max_sigma=float(np.nanmax(sig)), pmf_rmse=_pmf_rmse(landscape, windows, camp, final.f_kT),
                     hours_used=float(camp.hours), topup_md_fraction=float(camp.topup_hours / camp.hours),
                     structural_routed=tuple(sorted(structural)), n_states=len(ids),
                     n_sigma_nan=int(np.sum(~np.isfinite(sig))), epochs=tuple(epochs),
                     deficit_fraction=float(np.mean(np.nan_to_num(sig, nan=np.inf) > target)),
                     median_rows=float(np.median(camp.kept)), zero_rows_epoch0=int(zero0 or 0),
                     zero_rows_final=int(np.sum(camp.kept == 0)),
                     routed_coords=tuple(sorted(tuple(sorted((coord(a), coord(b)))) for a, b in structural)),
                     topped_then_routed=int(epochs[-1].get("topped_earlier_now_structural", 0)) if epochs else 0,
                     g_ratio=float(np.mean(g_ratios)) if g_ratios else math.nan)


def _sigma_summary(diag) -> dict:
    s = np.array([v for v in diag.sigma_kcal.values() if math.isfinite(v)])
    return {"median": float(np.median(s)), "max": float(s.max()), "n_above_0.10": int(np.sum(s > 0.10)),
            "n": int(s.size), "n_unconverged": len(diag.unconverged)}


def _one_sided_p(b: Sequence[float], a: Sequence[float]) -> float:
    """Wilcoxon signed-rank, H1: b < a (paired over seeds)."""
    from scipy.stats import wilcoxon
    d = np.asarray(b, float) - np.asarray(a, float)
    if not np.any(d != 0):
        return 1.0
    return float(wilcoxon(b, a, alternative="less").pvalue)


WIN_MARGIN = 0.01            # a "win" needs B more than 1 % below A (ruling 24)
# Frozen by `calibrate_regime` (arm A only, CAL_SEEDS) before any arm B ran; ``None`` = no
# targetable regime (at >= 40 rows/state no state of arm A is above target on any landscape).
# ROWS_FLOOR_HOURS: the smallest calibrated budget meeting the rows floor; the bridge scenario and
# the informational (excluded) A/B comparison run there. ``--recalibrate`` reruns the calibration.
FROZEN_BUDGET_HOURS: Dict[str, Optional[float]] = {
    "rugged-2d": None, "gated-barrier": None, "slow-cv2-double-branch": None, "harmonic-bowl": None,
}
ROWS_FLOOR_HOURS = {
    "rugged-2d": 2.5234375, "gated-barrier": 2.4921875, "slow-cv2-double-branch": 2.4765625,
    "harmonic-bowl": 1.97265625,
}


def wins(b: Sequence[float], a: Sequence[float]) -> int:
    return int(sum(1 for x, y in zip(b, a) if x < (1.0 - WIN_MARGIN) * y))


def _med(rs: List[ArmResult], key: str) -> float:
    return float(np.nanmedian([getattr(r, key) for r in rs]))


def _landscape_summary(name: str, a: List[ArmResult], b: List[ArmResult], cap: float):
    """(summary, per-seed detail) for one landscape at its frozen budget."""
    ma = {k: _med(a, k) for k in ("max_sigma", "pmf_rmse")}
    mb = {k: _med(b, k) for k in ("max_sigma", "pmf_rmse")}
    sa, sb = [r.max_sigma for r in a], [r.max_sigma for r in b]
    ra, rb = [r.pmf_rmse for r in a], [r.pmf_rmse for r in b]
    out = {"kind": "homogeneous" if name in HOMOGENEOUS else "heterogeneous", "n_states": a[0].n_states,
           "median_uniform": ma, "median_topup": mb, "p_max_sigma": _one_sided_p(sb, sa),
           "p_pmf_rmse": _one_sided_p(rb, ra), "wins_max_sigma": wins(sb, sa), "wins_pmf_rmse": wins(rb, ra),
           "max_topup_md_fraction": float(max(r.topup_md_fraction for r in b)),
           "max_rel_hours_gap": float(max(abs(x.hours_used - y.hours_used) / x.hours_used for x, y in zip(a, b))),
           "deficit_fraction_uniform": _med(a, "deficit_fraction"), "median_rows_uniform": _med(a, "median_rows"),
           "median_rows_topup": _med(b, "median_rows"),
           "zero_row_states_epoch0": {"uniform": int(max(r.zero_rows_epoch0 for r in a)),
                                      "topup": int(max(r.zero_rows_epoch0 for r in b))},
           "zero_row_states_final": {"uniform": int(max(r.zero_rows_final for r in a)),
                                     "topup": int(max(r.zero_rows_final for r in b))},
           "g_ratio_patch_over_all_state": _med(b, "g_ratio"),
           "topped_then_routed_median": _med(b, "topped_then_routed")}
    rel = {k: (mb[k] - ma[k]) / ma[k] for k in ma}
    out["rel_diff"] = rel
    if out["kind"] == "homogeneous":
        within = out["max_topup_md_fraction"] <= cap + 1e-9
        out["pass_max_sigma"] = bool(abs(rel["max_sigma"]) < 0.05 and within)
        out["pass_pmf_rmse"] = bool(abs(rel["pmf_rmse"]) < 0.05 and within)
    else:
        for k, p_key in (("max_sigma", "p_max_sigma"), ("pmf_rmse", "p_pmf_rmse")):
            out[f"pass_{k}"] = bool(mb[k] < (1.0 - WIN_MARGIN) * ma[k] and out[p_key] < 0.05)
    out["pass"] = bool(out["pass_max_sigma"] and out["pass_pmf_rmse"])
    detail = [{"seed": s, "uniform": [x.max_sigma, x.pmf_rmse], "topup": [y.max_sigma, y.pmf_rmse],
               "topup_md_fraction": y.topup_md_fraction, "epochs": list(y.epochs)}
              for s, (x, y) in enumerate(zip(a, b))]
    return out, detail


def removed_pass_centres(landscape: Landscape) -> set:
    full = {(round(w.center1, 6), round(w.center2, 6)) for w in _ladder(landscape)}
    return full - {(round(w.center1, 6), round(w.center2, 6)) for w in _ladder(landscape, True)}


def flanking_edges(routed_coords, removed) -> set:
    """Routed coordinate edges that span a removed centre along one axis (same column or row, either side)."""
    out = set()
    for a, b in routed_coords:
        for r in removed:
            same_col = a[0] == b[0] == r[0] and min(a[1], b[1]) < r[1] < max(a[1], b[1])
            same_row = a[1] == b[1] == r[1] and min(a[0], b[0]) < r[0] < max(a[0], b[0])
            if same_col or same_row:
                out.add((a, b))
    return out


def _bridge_summary(n_seeds: int, budget: float) -> dict:
    """Ruling 27: edges keyed by (centre, lambda); flanking edges routed with the gap, absent intact."""
    ls = LANDSCAPES[BRIDGE_LANDSCAPE]
    removed = removed_pass_centres(ls)
    rows = []
    for s in range(n_seeds):
        cut = run_arm(ls, arm="topup", seed=s, wall_hours_budget=budget, remove_bridge=True)
        whole = run_arm(ls, arm="topup", seed=s, wall_hours_budget=budget)
        flank = flanking_edges(cut.routed_coords, removed)
        rows.append({"seed": s, "flanking_routed_with_gap": sorted(flank),
                     "flanking_routed_intact": sorted(flank & set(whole.routed_coords)),
                     "n_routed_with_gap": len(cut.routed_coords), "n_routed_intact": len(whole.routed_coords),
                     "topped_then_routed_with_gap": cut.topped_then_routed})
    ok = [bool(r["flanking_routed_with_gap"]) and not r["flanking_routed_intact"] for r in rows]
    return {"landscape": BRIDGE_LANDSCAPE, "budget_hours": budget, "n_seeds": n_seeds,
            "removed_centres": sorted(removed), "seeds_passing": int(sum(ok)),
            "topped_then_routed_with_gap": [r["topped_then_routed_with_gap"] for r in rows],
            "per_seed": rows, "pass": bool(all(ok))}


def _write(out: Path, doc: dict, detail: Optional[dict] = None) -> None:
    import gzip
    out.mkdir(parents=True, exist_ok=True)
    (out / "topup_study.json").write_text(json.dumps(doc, indent=1, default=str))
    if detail is not None:
        with gzip.open(out / "topup_study_detail.json.gz", "wt") as fh:
            json.dump(detail, fh, default=str)


def run_study(landscapes: Sequence[str], *, n_seeds: int = 20, out: Path,
              budgets: Optional[Dict[str, Optional[float]]] = None, bridge: bool = False) -> dict:
    """Both arms x ``n_seeds`` per landscape at a frozen budget; writes ``<out>/topup_study.json``
    (summaries) and ``topup_study_detail.json.gz`` (per seed / epoch), returns the summary.

    Budgets not given are calibrated from arm A alone (``calibrate_regime``) and written to the
    JSON before any arm B campaign runs. A ``None`` budget = no targetable regime: that landscape
    is reported as such and left out of pass/fail. The bridge scenario, which is not an A/B
    comparison, then runs at the rows-floor budget from the calibration trace.
    """
    from ..adaptive_production import AdaptiveDecisionPolicy
    policy = AdaptiveDecisionPolicy(topups_enabled=True)
    cap, target = float(policy.topup_max_fraction), float(policy.topup_target_sigma)
    out = Path(out)
    names = list(landscapes) + ([BRIDGE_LANDSCAPE] if bridge and BRIDGE_LANDSCAPE not in landscapes else [])
    frozen, calibration = dict(budgets or {}), {}
    floors = dict(ROWS_FLOOR_HOURS if budgets else {})
    for name in names:
        if name not in frozen:
            frozen[name], calibration[name] = calibrate_regime(LANDSCAPES[name], run_arm)
            floors[name] = bridge_budget(calibration[name])

    def floor_hours(name: str) -> float:
        return floors[name]
    meta = {"n_seeds": n_seeds, "n_epochs": N_EPOCHS, "n_gpus": N_GPUS, "report_interval": REPORT_INTERVAL,
            "timestep_fs": TIMESTEP_FS, "kt_kcal": KT_KCAL, "rungs": list(RUNGS), "k_window_kT": K_WINDOW,
            "target_sigma_kcal": target, "deficit_band": list(DEFICIT_BAND), "min_rows": MIN_ROWS,
            "win_margin": WIN_MARGIN, "frozen_budget_hours": frozen, "rows_floor_hours": floors,
            "regime_calibration_arm_A_only": calibration}
    _write(out, {"meta": meta, "landscapes": {}})          # frozen before any arm B runs
    summary: Dict[str, dict] = {}
    detail: Dict[str, list] = {}
    for name in landscapes:
        ls, h = LANDSCAPES[name], frozen[name]
        status = "tested"
        if h is None:        # excluded from pass/fail; A/B still run at the rows floor, for information
            status, h = "no targetable regime", floor_hours(name)
        a = [run_arm(ls, arm="uniform", seed=s, wall_hours_budget=h) for s in range(n_seeds)]
        b = [run_arm(ls, arm="topup", seed=s, wall_hours_budget=h) for s in range(n_seeds)]
        summ, detail[name] = _landscape_summary(name, a, b, cap)
        summary[name] = {"budget_hours": h, "status": status, **summ}
        if status != "tested":
            summary[name]["pass"] = summary[name]["pass_max_sigma"] = summary[name]["pass_pmf_rmse"] = None
    if bridge:
        h = frozen[BRIDGE_LANDSCAPE] or floor_hours(BRIDGE_LANDSCAPE)
        bs = _bridge_summary(n_seeds, h)
        detail["structural_gap"] = bs.pop("per_seed")
        summary["structural_gap"] = bs
    _write(out, {"meta": meta, "landscapes": summary}, detail)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--landscapes", nargs="+", default=list(HETEROGENEOUS + HOMOGENEOUS), choices=sorted(LANDSCAPES))
    p.add_argument("--recalibrate", action="store_true", help="ignore FROZEN_BUDGET_HOURS; calibrate from arm A")
    p.add_argument("--no-bridge", action="store_true", help="skip the structural-gap scenario")
    a = p.parse_args(argv)
    summary = run_study(a.landscapes, n_seeds=a.n_seeds, out=a.out, bridge=not a.no_bridge,
                        budgets=None if a.recalibrate else FROZEN_BUDGET_HOURS)
    for name, s in summary.items():
        if name == "structural_gap":
            print(f"{name} ({s['budget_hours']:.3f} h): flanking edges routed with the gap and absent intact "
                  f"in {s['seeds_passing']}/{s['n_seeds']} seeds; pass={s['pass']}")
            continue
        tag = "" if s["status"] == "tested" else f" [{s['status']}: excluded, informational]"
        print(f"{name}{tag} ({s['kind']}, {s['budget_hours']:.3f} h): max sigma A {s['median_uniform']['max_sigma']:.4f} "
              f"B {s['median_topup']['max_sigma']:.4f} (p={s['p_max_sigma']:.3g}, wins {s['wins_max_sigma']}) "
              f"pass={s['pass_max_sigma']}; PMF RMSE A {s['median_uniform']['pmf_rmse']:.4f} "
              f"B {s['median_topup']['pmf_rmse']:.4f} (p={s['p_pmf_rmse']:.3g}, wins {s['wins_pmf_rmse']}) "
              f"pass={s['pass_pmf_rmse']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
