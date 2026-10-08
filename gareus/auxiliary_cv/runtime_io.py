# gareus/auxiliary_cv/runtime_io.py
"""Small, testable glue between run_gareus and the auxiliary storage layer (CVaux Stage C).

Only imported on the auxiliary path (``--aux-cv-model``).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..correctness._io import IntegrityError


def check_exchange_boundary_alignment(*, exchange_interval: int, distance_interval: int, traj_interval: int,
                                      calib_steps: int) -> list[str]:
    """Every exchange boundary must be a sample boundary (and a frame boundary when frames are written).

    Takes the RESOLVED intervals: ``distance_interval`` after run_gareus' defaulting, ``traj_interval`` =
    ``effective_traj_interval``. Reporters fire on the absolute step grid, so ``calib_steps`` must be a
    multiple of ``traj_interval`` too.
    """
    ex = int(exchange_interval)
    if int(distance_interval) <= 0 or ex % int(distance_interval) != 0:
        raise IntegrityError(f"auxiliary runs need distance_interval ({distance_interval}, resolved) to divide "
                             f"exchange_interval ({ex}) so samples exist on every exchange boundary (spec Section 7)")
    if int(traj_interval) <= 0:
        return ["trajectory frames are off: no structural frames on the exchange-boundary grid (Stage D prerequisite)"]
    if ex % int(traj_interval) != 0:
        raise IntegrityError(f"auxiliary runs need traj_interval ({traj_interval}) to divide exchange_interval ({ex})")
    if int(calib_steps) % int(traj_interval) != 0:
        raise IntegrityError(f"auxiliary runs need calib_steps ({calib_steps}) to be a multiple of traj_interval "
                             f"({traj_interval}): reporters fire on the absolute step grid")
    return []


class ExchangeEventCounter:
    """attempt_seq: the order of exchange events within one absolute step (0, 1, ...)."""

    def __init__(self) -> None:
        self._step = None
        self._seq = 0

    def next(self, step: int) -> int:
        if step != self._step:
            self._step, self._seq = step, 0
        seq = self._seq
        self._seq += 1
        return seq


def _log(q: float) -> float:
    return math.log(q) if q > 0 else float("-inf")


def aux_event_fields(prop) -> dict[str, float]:
    """The proposal fields of a Gibbs move (no ``selected_replica``); log q is -inf for q <= 0."""
    return {"log_q_forward": _log(float(prop.q_forward)), "log_q_reverse": _log(float(prop.q_reverse)),
            "p_accept": float(prop.pacc)}


def event_from_gibbs(prop, *, step: int, seq: int, selected_replica: int, accepted: bool, energy_version: str,
                     assignments_after: Sequence[int]) -> dict[str, Any]:
    """``write_event`` keywords for a Gibbs decision that changes nothing (``stay`` / ``no_candidates``)."""
    kind = "no_candidates" if prop.no_candidates else ("stay" if prop.stayed else "swap")
    return dict(step=int(step), attempt_seq=int(seq), selected_replica=int(selected_replica),
                replica_i=int(selected_replica), replica_j=int(selected_replica),
                window_i=int(prop.current_window),
                window_j=int(prop.proposed_window if kind == "swap" else prop.current_window),
                kind=kind, delta_e_kj=float(prop.delta_kj), accepted=bool(accepted),
                energy_version=str(energy_version), assignments_after=[int(x) for x in assignments_after],
                **aux_event_fields(prop))


def check_runtime_parity(runtime_z, positions_z, *, beta: float, k_max_kcal: float, centers, tolerance: float) -> None:
    """The exchange-matrix z vs the z of the stored torsions, as a reduced-energy bound.

    With no active state it returns first and never raises (ruling M5: a sham-only population records).
    """
    if float(k_max_kcal) <= 0:
        return
    from .offline import parity_violation
    from .runtime import AuxObservationError
    runtime_z = np.asarray(runtime_z, dtype=np.float64)
    positions_z = np.asarray(positions_z, dtype=np.float64)
    if np.any(np.isfinite(runtime_z) != np.isfinite(positions_z)):
        raise AuxObservationError("runtime/positions aux z parity: finite mismatch with an active auxiliary state")
    du = parity_violation(runtime_z, positions_z, beta=beta, k_max_kcal=k_max_kcal, centers=centers)
    if du.size and np.nanmax(du) > float(tolerance):
        raise AuxObservationError(f"runtime/positions aux z parity: reduced disagreement {np.nanmax(du):.3g} > "
                                  f"{tolerance:g}")


def context_box_nm(context, unit) -> list:
    """One Context's periodic box vectors in nm (an OpenMM State always carries the box)."""
    return np.asarray(context.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
                      dtype=np.float64).tolist()


def check_fixed_box(boxes, fixed_box, *, label: str) -> None:
    """An NVT state definition has ONE fixed box: every replica's box must equal it exactly."""
    want = np.asarray(fixed_box, dtype=np.float64)
    for r, box in enumerate(boxes):
        if not np.array_equal(np.asarray(box, dtype=np.float64), want):
            raise IntegrityError(f"{label}: replica {r} box {np.asarray(box).tolist()} != the fixed NVT box "
                                 f"{want.tolist()}; an NVT auxiliary state definition needs one fixed box")


def make_aux_record_observer(runtime, *, use_fast_path: bool, fast_forces, unit, schema, models):
    """Per-sample observation of an auxiliary run.

    ONE positions read per carrier gives the geometry check (only with an active state), the stored
    torsion basis and the positions z used for runtime parity. The runtime exchange z is computed as
    Stage B's ``make_aux_z_observer`` does: on the fast path the force's own value, on the slow path
    these positions (ruling M4: no second positions read).
    """
    from .runtime import check_aux_geometry, observe_aux_z
    from .sample_schema import observe_carrier
    any_active = any(float(k) > 0.0 for k in runtime.table.k_kcal)

    def observe(r, sim):
        pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        if any_active:
            check_aux_geometry(pos, runtime, replica=r)
        if use_fast_path:
            z_runtime = observe_aux_z(sim.context, runtime, force=fast_forces[r])
        else:
            z_runtime = observe_aux_z(sim.context, runtime, positions_nm=pos)
        obs = observe_carrier(pos, schema, models)
        return float(z_runtime), obs.torsions, float(obs.z[0])

    return observe


def data_boundary(out_dir, segment_id: str) -> dict[str, dict[str, int]]:
    """Committed Parquet generation and row count of one segment's samples and exchanges.

    A manifest exists only after the first non-empty flush (e.g. SIGTERM before any sample, ruling M1):
    a missing one is generation 0 with 0 rows.
    """
    from ..parquet_manifest import load_manifest
    out = {}
    for kind in ("samples", "exchanges"):
        seg_dir = Path(out_dir) / kind / str(segment_id)
        m = (load_manifest(seg_dir, expected_kind=kind) if seg_dir.is_dir() else None) or {"generation": 0,
                                                                                          "n_rows": 0}
        out[kind] = {"generation": int(m["generation"]), "n_rows": int(m["n_rows"])}
    return out


def anchor_ledger_events(out_dir, manifest: Mapping[str, Any]) -> dict:
    """The exchange events of the checkpoint's ledger-anchor segment only (carry-over 2).

    ``replay_assignments`` filters by step alone, so events of any other segment (an earlier run, a
    restarted child at the same steps) must never reach it. Rows after the checkpoint step are left in:
    the replay's ``up_to_step`` drops them.
    """
    from ..query import load_exchanges
    segment_id = str(manifest["aux"]["ledger_anchor"]["segment_id"])
    return load_exchanges(Path(out_dir), segment_ids=[segment_id])
