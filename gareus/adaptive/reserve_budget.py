"""Adaptive headroom (spec P1): the swarm layout's reserve and its per-epoch split.

``--swarm-adaptive-reserve-fraction f`` makes the swarm layout leave ``floor(f * max_replicas)``
replicas unfilled and record that in ``layout_plan.json`` (``adaptive_reserve``, read with
``gareus.layout_plan.adaptive_reserve``). The adaptive side spends the free slots of the LIVE
registry (``max_replicas - active states``), per epoch:

* ``add_rung`` may take at most ``ADD_RUNG_SHARE`` (1/3) of the free slots;
* resolution actions (spec 3.3 R1-R3) at most ``RESOLUTION_SHARE`` (1/2) of what remains after
  add_rung took its share (``add_rung_taken``; its full allowance when not given).

Actions cost whole centres (one state per rung), so the allowances are also given in centres.
Nothing here is wired into an action proposer yet: it is the pure rule 3.3 will call. Note
that without that wiring the existing adaptive adds (weak-edge bridges, add_rung) spend a
reserve like any other free slot, and that one new rung costs one state per centre (59 on a
chignolin_9-sized layout), far above 1/3 of a 35-slot reserve -- the split effectively keeps
add_rung out of a small reserve.

NumPy-free; Python 3.9 compatible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

ADD_RUNG_SHARE = 1.0 / 3.0
RESOLUTION_SHARE = 0.5


def validate_reserve_fraction(fraction: Any) -> float:
    """0 <= f < 1 (f = 1 would leave no state at all); raises ValueError otherwise."""
    value = float(fraction)
    if not math.isfinite(value) or value < 0.0 or value >= 1.0:
        raise ValueError(f"adaptive reserve fraction must be in [0, 1), got {fraction!r}")
    return value


def reserved_replicas(max_replicas: int, fraction: float) -> int:
    """Replicas the layout leaves unfilled: floor(f * max_replicas); 0 for an unlimited cap (0)."""
    cap = int(max_replicas or 0)
    if cap <= 0:
        return 0
    return int(math.floor(validate_reserve_fraction(fraction) * cap))


def layout_reserve_record(fraction: float, max_replicas: int, n_rungs: int, *, fill_cap_spatial: int,
                          spatial_states: int) -> dict:
    """The ``adaptive_reserve`` record of a layout plan (spec P1).

    ``free_slots`` is what the layout actually left free (the reserve plus rounding to whole
    rung stacks); ``reserve_shortfall`` is how much of the request the mandatory stacks ate
    (the reserve never makes a feasible layout insufficient)."""
    requested = reserved_replicas(max_replicas, fraction)
    granted = int(spatial_states) * int(n_rungs)
    free = int(max_replicas) - granted
    return {"fraction": float(fraction), "max_replicas": int(max_replicas), "n_rungs": int(n_rungs),
            "reserved_replicas_requested": requested, "cap_spatial_full": int(max_replicas) // int(n_rungs),
            "fill_cap_spatial": int(fill_cap_spatial), "granted_states": granted, "free_slots": free,
            "reserve_shortfall": max(0, requested - free),
            "add_rung_share": ADD_RUNG_SHARE, "resolution_share": RESOLUTION_SHARE}


@dataclass(frozen=True)
class ReserveAllowance:
    """Per-epoch allowances in states (``*_slots``) and whole centres (``*_centres``).

    ``governed`` is False when no reserve applies: ``reason`` "unlimited" (max_replicas 0) or
    "no_reserve"; then ``add_rung_slots`` is None (today's budget rule applies) and resolution
    actions get nothing (they may only draw on a reserve). "cap_changed" = the live cap differs
    from the one the layout recorded (the live one is used)."""
    governed: bool
    reason: str
    free_slots: Optional[int]
    add_rung_slots: Optional[int]
    resolution_slots: int
    n_rungs: Optional[int] = None
    add_rung_centres: Optional[int] = None
    resolution_centres: Optional[int] = None


def _centres(slots: Optional[int], n_rungs: Optional[int]) -> Optional[int]:
    if slots is None or not n_rungs or int(n_rungs) <= 0:
        return None
    return int(slots) // int(n_rungs)


def reserve_allowances(max_replicas: int, n_active_states: int, reserve: Optional[Mapping[str, Any]], *,
                       n_rungs: Optional[int] = None, add_rung_taken: Optional[int] = None) -> ReserveAllowance:
    """Split the live free slots (``max_replicas - n_active_states``) for one epoch (spec P1/3.3)."""
    cap = int(max_replicas or 0)
    if cap <= 0:
        return ReserveAllowance(False, "unlimited", None, None, 0, n_rungs)
    fraction = float((reserve or {}).get("fraction") or 0.0)
    if reserve is None or fraction <= 0.0:
        return ReserveAllowance(False, "no_reserve", max(0, cap - int(n_active_states)), None, 0, n_rungs)
    recorded = (reserve or {}).get("max_replicas")
    reason = "cap_changed" if recorded is not None and int(recorded) != cap else "reserve"
    free = max(0, cap - int(n_active_states))
    add_rung = int(math.floor(ADD_RUNG_SHARE * free))
    taken = add_rung if add_rung_taken is None else min(max(0, int(add_rung_taken)), add_rung)
    resolution = int(math.floor(RESOLUTION_SHARE * (free - taken)))
    return ReserveAllowance(True, reason, free, add_rung, resolution, n_rungs,
                            _centres(add_rung, n_rungs), _centres(resolution, n_rungs))


__all__ = ["ADD_RUNG_SHARE", "RESOLUTION_SHARE", "ReserveAllowance", "layout_reserve_record",
           "reserve_allowances", "reserved_replicas", "validate_reserve_fraction"]
