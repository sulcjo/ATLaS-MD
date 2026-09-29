"""``layout_plan.json`` -- the swarm layout's companion artifact -- read in one place (spec P7b).

The swarm stage writes the file beside ``windows_lambda_ladder.csv``
(``gareus.swarm.ladder_design.layout_plan_record``). It carries per-state roles, mandatory
state ids and region coverage -- metadata, never physics (spec F05). Two schemas exist:

* v1 (``"version": "layout_plan_v1"``, chignolin_8/9): ``states`` lists every state with
  ``state_id``, ``spatial_index``, ``cell``, ``role``, ``gamd_lambda``, ``mandatory`` and the
  physics row's ``center1``/``k1``/``center2``/``k2`` (an unrestrained axis: k = 0 and a
  finite placeholder centre). No per-state region; the region of a CV1 centre is in
  ``region_inventory.region_of_centre``.
* v2 (``"version": "layout_plan_v2"``, ``"schema_version": 2``): the same, plus per state
  ``region`` (the discovered CV1 region its CV1 centre represents; None when CV1 is
  unrestrained) and ``restrained`` ([CV1, CV2] booleans, the P6 pattern), so a consumer never
  has to infer identity from a placeholder coordinate.

``read_layout_plan`` returns one normalised form for both; v1 files keep working (region is
derived from ``region_of_centre`` when that is present, restrained from k > 0).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

LAYOUT_PLAN_FILENAME = "layout_plan.json"
LAYOUT_PLAN_V1 = "layout_plan_v1"
LAYOUT_PLAN_V2 = "layout_plan_v2"
SCHEMA_VERSIONS = {LAYOUT_PLAN_V1: 1, LAYOUT_PLAN_V2: 2}


def _finite(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _restrained(k: Any) -> bool:
    """P6: restrained iff k is a finite number > 0 (the physics rows always record k)."""
    value = _finite(k)
    return value is not None and value > 0.0


@dataclass(frozen=True)
class LayoutPlanState:
    state_id: int
    role: Optional[str]
    mandatory: bool
    region: Optional[str]
    center1: Optional[float]
    k1: Optional[float]
    center2: Optional[float]
    k2: Optional[float]
    gamd_lambda: Optional[float]
    restrained: Tuple[bool, bool]
    spatial_index: Optional[int] = None


def schema_version(plan: Mapping[str, Any]) -> int:
    """1 or 2; an unknown or missing version string reads as 1 (the only earlier writer)."""
    explicit = plan.get("schema_version")
    if isinstance(explicit, int) and explicit in (1, 2):
        return explicit
    return SCHEMA_VERSIONS.get(str(plan.get("version")), 1)


def plan_states(plan: Mapping[str, Any]) -> List[LayoutPlanState]:
    """Every state of a v1 or v2 plan, normalised (v1 region from ``region_of_centre``)."""
    version = schema_version(plan)
    region_of_centre: Sequence[Any] = ((plan.get("region_inventory") or {}).get("region_of_centre") or [])
    out: List[LayoutPlanState] = []
    for rec in plan.get("states") or []:
        cell = rec.get("cell") or [None, None]
        k1, k2 = _finite(rec.get("k1")), _finite(rec.get("k2"))
        if version >= 2 and isinstance(rec.get("restrained"), (list, tuple)) and len(rec["restrained"]) == 2:
            restrained = (bool(rec["restrained"][0]), bool(rec["restrained"][1]))
        else:
            restrained = (_restrained(k1), _restrained(k2))
        if version >= 2:
            region = rec.get("region")
        else:
            i1 = cell[0] if isinstance(cell, (list, tuple)) and cell else None
            region = (region_of_centre[int(i1)] if i1 is not None and restrained[0]
                      and 0 <= int(i1) < len(region_of_centre) else None)
        out.append(LayoutPlanState(
            state_id=int(rec["state_id"]), role=rec.get("role"), mandatory=bool(rec.get("mandatory")),
            region=None if region is None else str(region),
            center1=_finite(rec.get("center1")), k1=k1, center2=_finite(rec.get("center2")), k2=k2,
            gamd_lambda=_finite(rec.get("gamd_lambda")), restrained=restrained,
            spatial_index=None if rec.get("spatial_index") is None else int(rec["spatial_index"])))
    return out


def read_layout_plan(path: Path) -> Tuple[dict, List[LayoutPlanState]]:
    """(raw plan, normalised states). Raises OSError / ValueError / KeyError / TypeError on a bad file."""
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise TypeError(f"{path}: layout plan is not a JSON object")
    return plan, plan_states(plan)


def layout_plan_beside(table_path: Path) -> Optional[Path]:
    """The companion ``layout_plan.json`` next to a window table, if there is one."""
    candidate = Path(table_path).parent / LAYOUT_PLAN_FILENAME
    return candidate if candidate.exists() else None


__all__ = ["LAYOUT_PLAN_FILENAME", "LAYOUT_PLAN_V1", "LAYOUT_PLAN_V2", "LayoutPlanState", "layout_plan_beside",
           "plan_states", "read_layout_plan", "schema_version"]
