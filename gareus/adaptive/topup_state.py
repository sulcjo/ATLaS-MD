"""Top-up plan persistence (resume-stable) and campaign-level top-up state, written atomically."""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .topup_allocator import TopupPlan

PLAN_NAME = "topup_plan.json"
OVERLAP_NAME = "topup_union_overlap.json"
STATE_NAME = "topup_state.json"
CORRECTION_BOUNDS = (0.1, 2.0)


def _clean(v):
    return None if isinstance(v, float) and not math.isfinite(v) else v


def _atomic_write(path: Path, obj: Any) -> Path:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    os.replace(tmp, path)
    return path


def _num(v) -> float:
    return math.nan if v is None else float(v)


def save_plan(epoch_dir, plan: TopupPlan) -> Path:
    d = asdict(plan)
    for key in ("predicted_sigma", "sigma_before", "sample_scale_steps"):
        d[key] = {str(k): _clean(float(v)) for k, v in getattr(plan, key).items()}
    d["cost_hours"] = _clean(float(plan.cost_hours))
    d["structural_edges"] = [list(e) for e in plan.structural_edges]
    d["weak_edges_topped"] = [list(e) for e in plan.weak_edges_topped]
    return _atomic_write(Path(epoch_dir) / PLAN_NAME, d)


def load_plan(epoch_dir) -> Optional[TopupPlan]:
    path = Path(epoch_dir) / PLAN_NAME
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    for key in ("state_ids", "deficit_state_ids", "partner_state_ids"):
        d[key] = tuple(int(x) for x in d.get(key, ()))
    for key in ("structural_edges", "weak_edges_topped"):
        d[key] = tuple(tuple(int(x) for x in e) for e in d.get(key, ()))
    for key in ("predicted_sigma", "sigma_before", "sample_scale_steps"):
        d[key] = {int(k): _num(v) for k, v in d.get(key, {}).items()}
    d["cost_hours"] = _num(d.get("cost_hours"))
    return TopupPlan(**d)


def load_state(adaptive_dir) -> Dict[str, Any]:
    path = Path(adaptive_dir) / STATE_NAME
    if not path.exists():
        return {"correction": {}, "edge_attempts": {}, "f_kT": {}, "wall_time": []}
    d = json.loads(path.read_text())
    return {
        "correction": {int(k): float(v) for k, v in d.get("correction", {}).items()},
        "edge_attempts": {tuple(int(x) for x in k.split("-")): int(v) for k, v in d.get("edge_attempts", {}).items()},
        # save_state writes a non-finite f as null; restore it as nan, not float(None).
        "f_kT": {int(k): _num(v) for k, v in d.get("f_kT", {}).items()},
        "wall_time": list(d.get("wall_time", [])),
    }


def save_state(adaptive_dir, state: Dict[str, Any]) -> Path:
    d = {
        "correction": {str(k): v for k, v in state["correction"].items()},
        "edge_attempts": {f"{a}-{b}": v for (a, b), v in state["edge_attempts"].items()},
        "f_kT": {str(k): _clean(float(v)) for k, v in state["f_kT"].items()},
        "wall_time": state["wall_time"],
    }
    return _atomic_write(Path(adaptive_dir) / STATE_NAME, d)


def update_after_topup(state: Dict[str, Any], plan: TopupPlan, realised_sigma: Dict[int, float]) -> Dict[str, Any]:
    lo, hi = CORRECTION_BOUNDS
    new = {"correction": dict(state["correction"]), "edge_attempts": dict(state["edge_attempts"]),
           "f_kT": dict(state["f_kT"]), "wall_time": list(state["wall_time"])}
    for s in plan.deficit_state_ids:
        before, pred, real = plan.sigma_before.get(s), plan.predicted_sigma.get(s), realised_sigma.get(s)
        if not all(isinstance(v, float) and math.isfinite(v) for v in (before, pred, real)) or before <= pred:
            continue
        ratio = (before ** 2 - real ** 2) / (before ** 2 - pred ** 2)
        c = min(hi, max(lo, 0.7 * new["correction"].get(s, 1.0) + 0.3 * ratio))
        if ratio < 0.5:
            c = max(lo, 0.5 * c)
        new["correction"][s] = c
    for edge in plan.weak_edges_topped:
        key = (int(edge[0]), int(edge[1]))
        new["edge_attempts"][key] = new["edge_attempts"].get(key, 0) + 1
    return new


def save_union_overlap(epoch_dir, edge_overlap: Dict) -> Path:
    """The phase's latest measured union edge overlaps, beside its plan (resume-stable)."""
    d = {f"{int(a)}-{int(b)}": _clean(float(v)) for (a, b), v in edge_overlap.items()}
    return _atomic_write(Path(epoch_dir) / OVERLAP_NAME, d)


def load_union_overlap(epoch_dir) -> Optional[Dict]:
    path = Path(epoch_dir) / OVERLAP_NAME
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        a, b = (int(x) for x in k.split("-"))
        out[(a, b)] = float(v)
    return out
