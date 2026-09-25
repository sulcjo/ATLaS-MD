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
  arms use equal modelled wall hours up to one report interval. Arm B carries the
  calibration state (correction, edge attempts, warm-start f) across epochs like the driver.
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
from .oracle import LOW_F_THRESHOLD_KBT, reference_pmf
from .sampler import BIAS, Window, boost_dv, sample_window_exact

RUNGS = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
K_WINDOW = 50.0
SPACING_SIGMA = 2.0
EDGE_MARGIN = 0.05
N_EPOCHS = 3
REPORT_INTERVAL = 5000
TIMESTEP_FS = 4.0
N_GPUS = 4
TEMPERATURE_K = 300.0
KT_KCAL = 0.0019872041 * TEMPERATURE_K
BLOCK = 256
HETEROGENEOUS = ("rugged-2d", "gated-barrier", "slow-cv2-double-branch")
HOMOGENEOUS = ("harmonic-bowl",)
BRIDGE_LANDSCAPE = "gated-barrier"
PMF_BINS = 60


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


def _axis(lo: float, hi: float) -> np.ndarray:
    a, b = lo + EDGE_MARGIN * (hi - lo), hi - EDGE_MARGIN * (hi - lo)
    n = int(round((b - a) / (SPACING_SIGMA / math.sqrt(K_WINDOW)))) + 1
    return np.linspace(a, b, max(2, n))


def _ladder(landscape: Landscape, remove_bridge: bool = False) -> List[Window]:
    """Centres x rungs, ordered centre-major. ``remove_bridge`` drops the gated pass.

    The bridge is the CV2 row nearest gated-barrier's low-F pass (cv2 = 0.2), removed in
    the two CV1 columns flanking the ridge (cv1 = 0.5) on every rung: the cross-ridge
    edge then has weak but nonzero overlap (the ridge's own CV2 flanks still leak).
    """
    c1s, c2s = _axis(*landscape.cv1_bounds), _axis(*landscape.cv2_bounds)
    drop = set()
    if remove_bridge:
        row = float(c2s[np.argmin(np.abs(c2s - 0.2))])
        cols = sorted(c1s, key=lambda c: abs(c - 0.5))[:2]
        drop = {(float(c), row) for c in cols}
    return [Window(float(c1), K_WINDOW, float(c2), K_WINDOW, lam=float(lam))
            for c1 in c1s for c2 in c2s if (float(c1), float(c2)) not in drop for lam in RUNGS]


def _layout(windows: Sequence[Window]):
    """Edges, same-rung neighbours and rung partners, built exactly as the driver builds them."""
    from ..adaptive_production import AdaptiveDecisionPolicy, WindowStateRegistry, build_geometry_edges
    from ..layout_neighbours import other_rung_same_centre, same_rung_neighbours, spatial_neighbour_pairs
    registry = WindowStateRegistry()
    for w in windows:   # the driver's k is kcal/mol/CV^2
        registry.add_state(primary_center=w.center1, primary_k=w.k1 * KT_KCAL, secondary_center=w.center2,
                           secondary_k=w.k2 * KT_KCAL, gamd_lambda=w.lam, source="synthetic")
    policy = AdaptiveDecisionPolicy(topups_enabled=True)
    edges = [(a, b) for a, b, _t, _d in build_geometry_edges(registry, policy)]
    c1 = [w.center1 for w in windows]; c2 = [w.center2 for w in windows]; lam = [w.lam for w in windows]
    pairs = spatial_neighbour_pairs(c1, c2, lam, [w.k1 * KT_KCAL for w in windows],
                                    [w.k2 * KT_KCAL for w in windows], TEMPERATURE_K)
    nb, rp = same_rung_neighbours(pairs, lam), other_rung_same_centre(c1, c2, lam)
    n = len(windows)
    return edges, {w: nb.get(w, []) for w in range(n)}, {w: rp.get(w, []) for w in range(n)}, policy


class _Streams:
    """Per-window i.i.d. draw streams, blocked so sample i never depends on how draws were chunked."""

    def __init__(self, landscape: Landscape, windows: Sequence[Window], seed: int):
        self._ls, self._w, self._seed, self._blocks = landscape, windows, int(seed), {}

    def take(self, w: int, start: int, n: int) -> np.ndarray:
        first, last = start // BLOCK, (start + n - 1) // BLOCK
        for b in range(first, last + 1):
            if (w, b) not in self._blocks:
                rng = np.random.default_rng([self._seed, w, b])
                self._blocks[(w, b)] = sample_window_exact(self._ls, self._w[w], BLOCK, rng=rng)
        pooled = np.concatenate([self._blocks[(w, b)] for b in range(first, last + 1)])
        off = start - first * BLOCK
        return pooled[off:off + n]


class _Campaign:
    def __init__(self, n_states: int):
        self.steps = np.zeros(n_states)
        self.eff = np.zeros(n_states)
        self.kept = np.zeros(n_states, dtype=np.int64)
        self.rows_w: List[np.ndarray] = []
        self.rows_x: List[np.ndarray] = []
        self.hours = 0.0
        self.topup_hours = 0.0


def _partners_in(members, neighbours, rung_partners) -> Dict[int, int]:
    mset = set(members)
    return {w: len((set(neighbours.get(w, [])) | set(rung_partners.get(w, []))) & mset) for w in members}


def _sample_segment(landscape, windows, members, length, streams, n_partners, camp, table) -> float:
    """One lockstep segment: every member advances ``length`` steps; returns its wall hours."""
    from ..adaptive.throughput import wall_hours
    for w in members:
        g = 1.0 + 2.0 * tau_int(landscape, windows[w], n_partners=n_partners[w])
        camp.steps[w] += length
        camp.eff[w] += length / (REPORT_INTERVAL * g)
        new = int(math.floor(camp.eff[w] + 1e-9)) - int(camp.kept[w])
        if new > 0:
            camp.rows_x.append(streams.take(w, int(camp.kept[w]), new))
            camp.rows_w.append(np.full(new, w, dtype=np.int64))
            camp.kept[w] += new
    hours = wall_hours(int(length), len(members), TIMESTEP_FS, N_GPUS, table)
    camp.hours += hours
    return hours


def _reduced_potentials(landscape, windows, x: np.ndarray) -> np.ndarray:
    u = np.empty((x.shape[0], len(windows)))
    for k, win in enumerate(windows):
        u[:, k] = BIAS(win, x[:, 0], x[:, 1]) + boost_dv(landscape, win, x[:, 0], x[:, 1])
    return u


def _write_union_npz(path: Path, landscape, camp: _Campaign, windows) -> dict:
    """The union NPZ format read by ``union_diagnostics_from_npz``; returns subsample counts."""
    x = np.concatenate(camp.rows_x) if camp.rows_x else np.empty((0, 2))
    w = np.concatenate(camp.rows_w) if camp.rows_w else np.empty(0, dtype=np.int64)
    np.savez(path, umbrella_reduced_bias_nk=_reduced_potentials(landscape, windows, x),
             state_ids=np.arange(len(windows)), sampled_state_ids=w, cv1=x[:, 0], cv2=x[:, 1])
    counts = {}
    for k in range(len(windows)):
        raw = int(camp.steps[k] // REPORT_INTERVAL)
        g = raw / camp.eff[k] if camp.eff[k] > 0 else 1.0
        counts[str(k)] = {"raw": raw, "t0": 0, "kept": int(camp.kept[k]), "g": float(g), "status": "subsampled"}
    return counts


def _diagnose(tmp: Path, landscape, camp, windows, layout, f_init):
    """The driver's ``_phase_union_diagnostics`` call: sigma against same-rung neighbours, rung fallback."""
    from ..adaptive.union_diagnostics import union_diagnostics_from_npz
    edges, neighbours, rung_partners, policy = layout
    path = Path(tmp) / "union.npz"
    counts = _write_union_npz(path, landscape, camp, windows)
    sigma_nb = {w: neighbours.get(w) or rung_partners.get(w, []) for w in range(len(windows))}
    return union_diagnostics_from_npz(path, edges, kt_kcal=KT_KCAL, subsample_counts=counts,
                                      min_effect_kcal=float(policy.topup_min_effect), f_init=f_init or None,
                                      sigma_neighbours=sigma_nb)


def _pmf_rmse(landscape, windows, camp, f_kT: Dict[int, float]) -> float:
    """Low-F-weighted RMSE (kBT) of the union-MBAR CV1 PMF vs the reference PMF."""
    from scipy.special import logsumexp
    x = np.concatenate(camp.rows_x)
    n_k = np.bincount(np.concatenate(camp.rows_w), minlength=len(windows))
    act = [k for k in range(len(windows)) if n_k[k] > 0 and k in f_kT]
    u = _reduced_potentials(landscape, [windows[k] for k in act], x)
    f = np.array([f_kT[k] for k in act])
    log_w = -logsumexp(f[None, :] - u, b=n_k[act][None, :], axis=1)
    bins = np.linspace(*landscape.cv1_bounds, PMF_BINS + 1)
    dens, _ = np.histogram(x[:, 0], bins=bins, weights=np.exp(log_w - log_w.max()))
    raw, _ = np.histogram(x[:, 0], bins=bins)
    xr, pr = reference_pmf(landscape, axis="cv1")
    ref = np.interp(0.5 * (bins[:-1] + bins[1:]), xr, pr)
    use = (raw > 0) & (dens > 0) & (ref <= LOW_F_THRESHOLD_KBT)
    diff = -np.log(dens[use]) - ref[use]
    wgt = np.exp(-ref[use]); wgt /= wgt.sum()
    diff -= np.sum(wgt * diff)
    return float(np.sqrt(np.sum(wgt * diff ** 2)))


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
        steps = int(hours / per_step_all // REPORT_INTERVAL) * REPORT_INTERVAL
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


def _landscape_summary(name: str, a: List[ArmResult], b: List[ArmResult], cap: float) -> dict:
    ma = {k: float(np.median([getattr(r, k) for r in a])) for k in ("max_sigma", "pmf_rmse")}
    mb = {k: float(np.median([getattr(r, k) for r in b])) for k in ("max_sigma", "pmf_rmse")}
    p_sig = _one_sided_p([r.max_sigma for r in b], [r.max_sigma for r in a])
    p_rmse = _one_sided_p([r.pmf_rmse for r in b], [r.pmf_rmse for r in a])
    out = {"kind": "homogeneous" if name in HOMOGENEOUS else "heterogeneous", "n_states": a[0].n_states,
           "median_uniform": ma, "median_topup": mb, "p_max_sigma": p_sig, "p_pmf_rmse": p_rmse,
           "max_topup_md_fraction": float(max(r.topup_md_fraction for r in b)),
           "hours_uniform": [r.hours_used for r in a], "hours_topup": [r.hours_used for r in b],
           "per_seed": [{"seed": s, "uniform": [ra.max_sigma, ra.pmf_rmse], "topup": [rb.max_sigma, rb.pmf_rmse],
                         "topup_md_fraction": rb.topup_md_fraction, "epochs": list(rb.epochs)}
                        for s, (ra, rb) in enumerate(zip(a, b))]}
    if out["kind"] == "homogeneous":
        rel = abs(mb["max_sigma"] - ma["max_sigma"]) / ma["max_sigma"]
        out["rel_diff_max_sigma"] = rel
        out["pass"] = bool(rel < 0.05 and out["max_topup_md_fraction"] <= cap + 1e-9)
    else:
        out["pass"] = bool(mb["max_sigma"] < ma["max_sigma"] and mb["pmf_rmse"] < ma["pmf_rmse"] and p_sig < 0.05)
        out["pass_strict_both_p"] = bool(out["pass"] and p_rmse < 0.05)
    return out


def _bridge_summary(n_seeds: int, budget: float) -> dict:
    ls = LANDSCAPES[BRIDGE_LANDSCAPE]
    cut = [run_arm(ls, arm="topup", seed=s, wall_hours_budget=budget, remove_bridge=True) for s in range(n_seeds)]
    whole = [run_arm(ls, arm="topup", seed=s, wall_hours_budget=budget) for s in range(n_seeds)]
    topped = sum(1 for r in cut for e in r.epochs if "plan" in e
                 and set(map(tuple, e["plan"]["weak_edges_topped"])) & set(map(tuple, e["plan"]["structural_edges"])))
    return {"landscape": BRIDGE_LANDSCAPE, "seeds_routed_with_gap": sum(bool(r.structural_routed) for r in cut),
            "seeds_routed_without_gap": sum(bool(r.structural_routed) for r in whole),
            "routed_edges_with_gap": sorted({e for r in cut for e in r.structural_routed}),
            "routed_edges_without_gap": sorted({e for r in whole for e in r.structural_routed}),
            "epochs_topping_a_routed_edge": topped, "n_seeds": n_seeds,
            "pass": bool(all(r.structural_routed for r in cut) and topped == 0)}


def run_study(landscapes: Sequence[str], *, n_seeds: int = 20, out: Path, wall_hours_budget: float = 5.0,
              bridge: bool = False) -> dict:
    """Both arms x ``n_seeds`` per landscape; writes ``<out>/topup_study.json`` and returns the summary."""
    from ..adaptive_production import AdaptiveDecisionPolicy
    cap = float(AdaptiveDecisionPolicy(topups_enabled=True).topup_max_fraction)
    summary: Dict[str, dict] = {}
    for name in landscapes:
        ls = LANDSCAPES[name]
        a = [run_arm(ls, arm="uniform", seed=s, wall_hours_budget=wall_hours_budget) for s in range(n_seeds)]
        b = [run_arm(ls, arm="topup", seed=s, wall_hours_budget=wall_hours_budget) for s in range(n_seeds)]
        summary[name] = _landscape_summary(name, a, b, cap)
    if bridge:
        summary["structural_gap"] = _bridge_summary(n_seeds, wall_hours_budget)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"n_seeds": n_seeds, "wall_hours_budget": wall_hours_budget, "n_epochs": N_EPOCHS, "n_gpus": N_GPUS,
            "report_interval": REPORT_INTERVAL, "timestep_fs": TIMESTEP_FS, "kt_kcal": KT_KCAL,
            "rungs": list(RUNGS), "k_window_kT": K_WINDOW}
    (out / "topup_study.json").write_text(json.dumps({"meta": meta, "landscapes": summary}, indent=1, default=str))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--budget-hours", type=float, default=5.0)
    p.add_argument("--landscapes", nargs="+", default=list(HETEROGENEOUS + HOMOGENEOUS), choices=sorted(LANDSCAPES))
    p.add_argument("--no-bridge", action="store_true", help="skip the structural-gap scenario")
    a = p.parse_args(argv)
    summary = run_study(a.landscapes, n_seeds=a.n_seeds, out=a.out, wall_hours_budget=a.budget_hours,
                        bridge=not a.no_bridge)
    for name, s in summary.items():
        if name == "structural_gap":
            print(f"{name}: routed {s['seeds_routed_with_gap']}/{s['n_seeds']} with gap, "
                  f"{s['seeds_routed_without_gap']}/{s['n_seeds']} without; pass={s['pass']}")
            continue
        print(f"{name} ({s['kind']}): max sigma A {s['median_uniform']['max_sigma']:.4f} B "
              f"{s['median_topup']['max_sigma']:.4f} (p={s['p_max_sigma']:.3g}); PMF RMSE A "
              f"{s['median_uniform']['pmf_rmse']:.4f} B {s['median_topup']['pmf_rmse']:.4f} "
              f"(p={s['p_pmf_rmse']:.3g}); pass={s['pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
