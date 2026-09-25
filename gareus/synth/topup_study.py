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

from .landscapes import LANDSCAPES, Landscape
from .topup_model import (KT_KCAL, K_WINDOW, N_EPOCHS, N_GPUS, REPORT_INTERVAL, RUNGS, TIMESTEP_FS,  # noqa: F401
                          _Campaign, _diagnose, _ladder, _layout, _partners_in, _pmf_rmse,
                          _reduced_potentials, _sample_segment, _Streams, _write_union_npz)

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

    def all_state_segment(hours: float) -> float:
        # floor to the driver's segment quantum (_quantized_extra_steps: 1000 steps), never overspending
        steps = int(hours / per_step_all // SEGMENT_QUANTUM) * SEGMENT_QUANTUM
        return _sample_segment(landscape, windows, ids, steps, streams, full_np, camp, table) if steps else 0.0

    with tempfile.TemporaryDirectory() as tmp:
        for e in range(n_epochs):
            hours = wall_hours_budget / n_epochs + carry
            spent = all_state_segment(hours if arm == "uniform" else (1.0 - cap) * hours)
            rec = {"epoch": e, "hours": hours, "baseline_hours": spent}
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
                after = None
                if plan.reason == "planned":
                    np_patch = _partners_in(plan.state_ids, neighbours, rung_partners)
                    h = _sample_segment(landscape, windows, plan.state_ids, plan.steps, streams, np_patch, camp, table)
                    spent += h
                    camp.topup_hours += h
                    after = _diagnose(tmp, landscape, camp, windows, layout, state["f_kT"])
                    if after is not None:
                        state = update_after_topup(state, plan, after.sigma_kcal)
                        state["f_kT"] = dict(after.f_kT)
                rec["plan"] = _plan_record(plan, after)
            carry = hours - spent
            epochs.append(rec)
        all_state_segment(carry)
        final = _diagnose(tmp, landscape, camp, windows, layout, state["f_kT"])
    if final is None:
        raise RuntimeError("final union-MBAR solve failed; the campaign has no result")
    sig = np.array([final.sigma_kcal.get(k, math.nan) for k in ids], dtype=float)
    if np.all(~np.isfinite(sig)):
        raise RuntimeError("final union-MBAR solve gave no finite sigma")
    return ArmResult(max_sigma=float(np.nanmax(sig)), pmf_rmse=_pmf_rmse(landscape, windows, camp, final.f_kT),
                     hours_used=float(camp.hours), topup_md_fraction=float(camp.topup_hours / camp.hours),
                     structural_routed=tuple(sorted(structural)), n_states=len(ids),
                     n_sigma_nan=int(np.sum(~np.isfinite(sig))), epochs=tuple(epochs))


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


WIN_MARGIN = 0.01          # a "win" needs B more than 1 % below A (ruling 24)
SIGMA_BAND = (1.5, 2.0)    # arm A's median max sigma, in units of the top-up target, at the frozen budget
CAL_SEEDS = (100, 101, 102)  # calibration seeds, disjoint from the study's seeds 0..n-1
# Frozen by `calibrate_budget` (arm A only, CAL_SEEDS) before any arm B ran; recorded with the
# calibration trace in topup_study.json (``--recalibrate`` reproduces these exactly).
FROZEN_BUDGET_HOURS = {
    "rugged-2d": 0.3270618273826279,
    "gated-barrier": 0.7874790695726125,
    "slow-cv2-double-branch": 0.3360239363932954,
    "harmonic-bowl": 0.267254855639931,
}


def wins(b: Sequence[float], a: Sequence[float]) -> int:
    return int(sum(1 for x, y in zip(b, a) if x < (1.0 - WIN_MARGIN) * y))


def calibrate_budget(landscape: Landscape, *, target_sigma: float, band=SIGMA_BAND, seeds=CAL_SEEDS,
                     start_hours: float = 5.0, max_iter: int = 8):
    """Wall-hour budget at which arm A's median max sigma sits in ``band`` x target (arm A only).

    sigma ~ 1/sqrt(hours), so each step rescales hours by (median / (mid * target))^2.
    Returns (hours, trace).
    """
    hours, mid, trace = float(start_hours), 0.5 * (band[0] + band[1]), []
    for _ in range(max_iter):
        med = float(np.median([run_arm(landscape, arm="uniform", seed=s, wall_hours_budget=hours).max_sigma
                               for s in seeds]))
        trace.append({"hours": hours, "median_max_sigma": med, "ratio": med / target_sigma})
        if band[0] <= med / target_sigma <= band[1]:
            return hours, trace
        hours *= (med / (mid * target_sigma)) ** 2
    raise RuntimeError(f"budget calibration did not converge for {landscape.name}: {trace}")


def _landscape_summary(name: str, a: List[ArmResult], b: List[ArmResult], cap: float) -> dict:
    ma = {k: float(np.median([getattr(r, k) for r in a])) for k in ("max_sigma", "pmf_rmse")}
    mb = {k: float(np.median([getattr(r, k) for r in b])) for k in ("max_sigma", "pmf_rmse")}
    sa, sb = [r.max_sigma for r in a], [r.max_sigma for r in b]
    ra, rb = [r.pmf_rmse for r in a], [r.pmf_rmse for r in b]
    out = {"kind": "homogeneous" if name in HOMOGENEOUS else "heterogeneous", "n_states": a[0].n_states,
           "median_uniform": ma, "median_topup": mb, "p_max_sigma": _one_sided_p(sb, sa),
           "p_pmf_rmse": _one_sided_p(rb, ra), "wins_max_sigma": wins(sb, sa), "wins_pmf_rmse": wins(rb, ra),
           "max_topup_md_fraction": float(max(r.topup_md_fraction for r in b)),
           "hours_uniform": [r.hours_used for r in a], "hours_topup": [r.hours_used for r in b],
           "per_seed": [{"seed": s, "uniform": [x.max_sigma, x.pmf_rmse], "topup": [y.max_sigma, y.pmf_rmse],
                         "topup_md_fraction": y.topup_md_fraction, "epochs": list(y.epochs)}
                        for s, (x, y) in enumerate(zip(a, b))]}
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
    return out


def _bridge_summary(n_seeds: int, budget: float) -> dict:
    """Two-sided: an edge is routed structural with the gap, and none is routed without it."""
    ls = LANDSCAPES[BRIDGE_LANDSCAPE]
    cut = [run_arm(ls, arm="topup", seed=s, wall_hours_budget=budget, remove_bridge=True) for s in range(n_seeds)]
    whole = [run_arm(ls, arm="topup", seed=s, wall_hours_budget=budget) for s in range(n_seeds)]
    topped = sum(1 for r in cut for e in r.epochs if "plan" in e
                 and set(map(tuple, e["plan"]["weak_edges_topped"])) & set(map(tuple, e["plan"]["structural_edges"])))
    return {"landscape": BRIDGE_LANDSCAPE, "budget_hours": budget, "n_seeds": n_seeds,
            "seeds_routed_with_gap": sum(bool(r.structural_routed) for r in cut),
            "seeds_routed_without_gap": sum(bool(r.structural_routed) for r in whole),
            "routed_edges_with_gap": sorted({e for r in cut for e in r.structural_routed}),
            "routed_edges_without_gap": sorted({e for r in whole for e in r.structural_routed}),
            "epochs_topping_a_routed_edge": topped,
            "pass": bool(all(r.structural_routed for r in cut) and not any(r.structural_routed for r in whole)
                         and topped == 0)}


def _write(out: Path, doc: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "topup_study.json").write_text(json.dumps(doc, indent=1, default=str))


def run_study(landscapes: Sequence[str], *, n_seeds: int = 20, out: Path,
              budgets: Optional[Dict[str, float]] = None, bridge: bool = False) -> dict:
    """Both arms x ``n_seeds`` per landscape; writes ``<out>/topup_study.json`` and returns the summary.

    Budgets not given are calibrated from arm A alone (``calibrate_budget``) and written to
    the JSON before any arm B campaign runs.
    """
    from ..adaptive_production import AdaptiveDecisionPolicy
    policy = AdaptiveDecisionPolicy(topups_enabled=True)
    cap, target = float(policy.topup_max_fraction), float(policy.topup_target_sigma)
    out = Path(out)
    names = list(landscapes) + ([BRIDGE_LANDSCAPE] if bridge and BRIDGE_LANDSCAPE not in landscapes else [])
    frozen, calibration = dict(budgets or {}), {}
    for name in names:
        if name not in frozen:
            frozen[name], calibration[name] = calibrate_budget(LANDSCAPES[name], target_sigma=target)
    meta = {"n_seeds": n_seeds, "n_epochs": N_EPOCHS, "n_gpus": N_GPUS, "report_interval": REPORT_INTERVAL,
            "timestep_fs": TIMESTEP_FS, "kt_kcal": KT_KCAL, "rungs": list(RUNGS), "k_window_kT": K_WINDOW,
            "target_sigma_kcal": target, "sigma_band_x_target": list(SIGMA_BAND), "win_margin": WIN_MARGIN,
            "frozen_budget_hours": frozen, "budget_calibration_arm_A_only": calibration}
    _write(out, {"meta": meta, "landscapes": {}})          # frozen before any arm B runs
    summary: Dict[str, dict] = {}
    for name in landscapes:
        ls, h = LANDSCAPES[name], frozen[name]
        a = [run_arm(ls, arm="uniform", seed=s, wall_hours_budget=h) for s in range(n_seeds)]
        b = [run_arm(ls, arm="topup", seed=s, wall_hours_budget=h) for s in range(n_seeds)]
        summary[name] = {"budget_hours": h, **_landscape_summary(name, a, b, cap)}
    if bridge:
        summary["structural_gap"] = _bridge_summary(n_seeds, frozen[BRIDGE_LANDSCAPE])
    _write(out, {"meta": meta, "landscapes": summary})
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--landscapes", nargs="+", default=list(HETEROGENEOUS + HOMOGENEOUS), choices=sorted(LANDSCAPES))
    p.add_argument("--recalibrate", action="store_true", help="ignore FROZEN_BUDGET_HOURS and calibrate from arm A")
    p.add_argument("--no-bridge", action="store_true", help="skip the structural-gap scenario")
    a = p.parse_args(argv)
    summary = run_study(a.landscapes, n_seeds=a.n_seeds, out=a.out, bridge=not a.no_bridge,
                        budgets=None if a.recalibrate else FROZEN_BUDGET_HOURS)
    for name, s in summary.items():
        if name == "structural_gap":
            print(f"{name}: routed {s['seeds_routed_with_gap']}/{s['n_seeds']} with gap, "
                  f"{s['seeds_routed_without_gap']}/{s['n_seeds']} without; pass={s['pass']}")
            continue
        print(f"{name} ({s['kind']}, {s['budget_hours']:.3f} h): max sigma A {s['median_uniform']['max_sigma']:.4f} "
              f"B {s['median_topup']['max_sigma']:.4f} (p={s['p_max_sigma']:.3g}, wins {s['wins_max_sigma']}) "
              f"pass={s['pass_max_sigma']}; PMF RMSE A {s['median_uniform']['pmf_rmse']:.4f} "
              f"B {s['median_topup']['pmf_rmse']:.4f} (p={s['p_pmf_rmse']:.3g}, wins {s['wins_pmf_rmse']}) "
              f"pass={s['pass_pmf_rmse']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
