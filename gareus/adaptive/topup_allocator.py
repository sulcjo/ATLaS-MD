"""Deficit-driven, wall-hour-costed top-up plan (spec 4.2-4.3, revision 2, rulings 8-10). Pure function; no I/O."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .throughput import wall_hours

MAX_STEP_MULTIPLE = 4
CORRECTION_BOUNDS = (0.1, 2.0)


@dataclass(frozen=True)
class TopupPlan:
    state_ids: tuple = ()
    steps: int = 0
    deficit_state_ids: tuple = ()
    partner_state_ids: tuple = ()
    structural_edges: tuple = ()
    weak_edges_topped: tuple = ()
    predicted_sigma: dict = field(default_factory=dict)
    sigma_before: dict = field(default_factory=dict)
    cost_hours: float = 0.0
    reason: str = "healthy"


def _ok(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _clamp(c) -> float:
    lo, hi = CORRECTION_BOUNDS
    return min(hi, max(lo, float(c))) if _ok(c) else 1.0


def _round_up(steps: float, interval: int) -> int:
    return int(math.ceil(max(0.0, steps) / interval - 1e-9) * interval)


def plan_topup(diag, *, state_ids_in_order: Sequence[int], neighbours: Dict[int, List[int]],
               rung_partners: Dict[int, List[int]], policy, report_interval: int, timestep_fs: float,
               n_gpus: int, budget_hours: float, correction: Optional[Dict[int, float]] = None,
               edge_attempts: Optional[Dict[Tuple[int, int], int]] = None) -> TopupPlan:
    if diag is None:
        return TopupPlan(reason="no_diagnostics")
    interval = max(1, int(report_interval))
    target = float(policy.topup_target_sigma)
    corr = {s: _clamp((correction or {}).get(s, 1.0)) for s in state_ids_in_order}
    attempts = edge_attempts or {}
    sigma = {s: float(diag.sigma_kcal.get(s, math.nan)) for s in state_ids_in_order}
    n_eff = {s: int(diag.n_k.get(s, 0)) for s in state_ids_in_order}
    g = {s: max(1.0, float(diag.inefficiency.get(s, 1.0))) for s in state_ids_in_order}
    sampled = {s for s in state_ids_in_order if n_eff[s] > 0 and _ok(sigma[s])}

    deficits = {s for s in sampled if sigma[s] > target or s in diag.unconverged}
    initial = frozenset(deficits)
    structural, noise_edges = [], []
    for (a, b), ov in sorted(diag.edge_overlap.items()):
        if not _ok(ov) or ov >= float(policy.topup_weak_overlap):
            continue
        tried_out = attempts.get((a, b), 0) >= int(policy.topup_max_edge_attempts)
        if (a in initial or b in initial) and not tried_out:
            noise_edges.append((a, b))
        else:
            structural.append((a, b))
    deficits.update(x for (a, b) in noise_edges for x in (a, b) if x in sampled)
    if not deficits:
        return TopupPlan(structural_edges=tuple(structural), reason="healthy")

    def predicted(s: int, L: int) -> float:
        n = max(1e-9, float(n_eff[s]))
        return sigma[s] * math.sqrt(n / (n + corr[s] * L / (interval * g[s])))

    def required(s: int) -> int:
        # noise-edge endpoints and unconverged states below target both use σ/√2 (double their data)
        eff_target = target if sigma[s] > target else sigma[s] / math.sqrt(2.0)
        extra_eff = n_eff[s] * ((sigma[s] / eff_target) ** 2 - 1.0)
        need = _round_up(extra_eff * interval * g[s] / corr[s], interval)
        cap = _round_up(MAX_STEP_MULTIPLE * n_eff[s] * interval * g[s], interval)
        return max(interval, min(need, cap))

    def pick(pool: List[int]) -> int:
        return max(pool, key=lambda x: (sigma[x], -x))

    chosen: set = set()
    for s in sorted(deficits):
        same = [x for x in neighbours.get(s, []) if x in sampled]
        rung = [x for x in rung_partners.get(s, []) if x in sampled]
        if not any(x in deficits or x in chosen for x in same):
            pool = same or [x for x in rung if x not in deficits]
            if pool:
                chosen.add(pick(pool))
        if rung and not any(x in deficits or x in chosen for x in rung):
            chosen.add(pick(rung))
    partners = chosen - deficits
    patch = deficits | partners

    table = policy.topup_throughput_table
    per_step_hours = wall_hours(interval, len(patch), timestep_fs, n_gpus, table) / interval
    need_L = max(required(s) for s in deficits)
    budget_max = int(budget_hours / per_step_hours // interval) * interval if per_step_hours > 0 else 0
    L = min(need_L, budget_max)          # bring the worst deficit to target, capped by the budget
    if L < interval:
        return TopupPlan(structural_edges=tuple(structural), reason="cap_too_small")
    pred = {s: predicted(s, L) for s in patch}
    if not all(_ok(v) for v in pred.values()):
        return TopupPlan(structural_edges=tuple(structural), reason="cap_too_small")
    cost = per_step_hours * L
    return TopupPlan(state_ids=tuple(sorted(patch)), steps=int(L), deficit_state_ids=tuple(sorted(deficits)),
                     partner_state_ids=tuple(sorted(partners)), structural_edges=tuple(structural),
                     weak_edges_topped=tuple(noise_edges), predicted_sigma=pred,
                     sigma_before={s: sigma[s] for s in patch}, cost_hours=float(cost), reason="planned")
