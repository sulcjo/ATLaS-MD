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


#: Mirrors gareus.production.GIBBS_PROPOSAL_ALGORITHM_V2 (pinned equal by test; production is not imported here).
GIBBS_PROPOSAL_ALGORITHM_V2 = "gibbs_softmax_log_v2"


def aux_event_fields(prop) -> dict[str, Any]:
    """The proposal fields of a Gibbs move (no ``selected_replica``), written straight from the proposal.

    Auxiliary runs record only ``gibbs_softmax_log_v2`` proposals (F09): the log q's and log alpha are the
    proposal's own authoritative logs (finite even when their exponentials underflow), never logs of
    rounded probabilities. A proposal from another algorithm is refused rather than logged as -inf.
    """
    algorithm = getattr(prop, "proposal_algorithm", None)
    if algorithm != GIBBS_PROPOSAL_ALGORITHM_V2:
        raise ValueError(f"auxiliary exchange ledger records only {GIBBS_PROPOSAL_ALGORITHM_V2} Gibbs proposals, "
                         f"got proposal_algorithm {algorithm!r}")
    log_p_accept = float(prop.log_p_accept)
    return {"log_q_forward": float(prop.log_q_forward), "log_q_reverse": float(prop.log_q_reverse),
            "p_accept": math.exp(log_p_accept) if not math.isnan(log_p_accept) else float("nan"),
            "log_p_accept": log_p_accept, "proposal_algorithm": GIBBS_PROPOSAL_ALGORITHM_V2}


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
    both = np.isfinite(runtime_z) & np.isfinite(positions_z)
    if not np.any(both):
        # Final fix wave I4b: with an active state, a sample with no finite comparison cannot vouch for parity.
        raise AuxObservationError("runtime/positions aux z parity: no finite comparison (every carrier's z is "
                                  "non-finite) with an active auxiliary state")
    du = parity_violation(runtime_z[both], positions_z[both], beta=beta, k_max_kcal=k_max_kcal, centers=centers)
    worst = float(np.max(du))
    if not np.isfinite(worst) or worst > float(tolerance):
        raise AuxObservationError(f"runtime/positions aux z parity ({AUX_Z_SOURCE} vs {AUX_Z_REFERENCE}): reduced "
                                  f"disagreement {worst:.3g} > {tolerance:g}")


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


#: Where the stored runtime z (``aux_z_00``) and its parity reference come from (F07).
AUX_Z_SOURCE = "force"
AUX_Z_REFERENCE = "positions"


def make_aux_record_observer(runtime, *, aux_forces, unit, schema, models):
    """Per-sample observation of an auxiliary run.

    The runtime z is always the aux force's own value on the replica's Context (F07; ``AUX_Z_SOURCE``),
    whichever CV1/CV2 path the run uses. ONE positions read per carrier gives the geometry check (only
    with an active state), the stored torsion basis and the independent positions z (``AUX_Z_REFERENCE``)
    that runtime parity compares it with.
    """
    from .runtime import aux_z_from_force, check_aux_geometry, checked_aux_forces
    from .sample_schema import observe_carrier
    any_active = any(float(k) > 0.0 for k in runtime.table.k_kcal)
    forces = checked_aux_forces(runtime, aux_forces)

    def observe(r, sim):
        pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        if any_active:
            check_aux_geometry(pos, runtime, replica=r)
        z_runtime = aux_z_from_force(sim.context, forces[r], runtime, replica=r)
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


def committed_rows_up_to(out_dir, kind: str, segment_id: str, step: int) -> int:
    """Committed (manifest-listed) rows of one segment with ``step <= step`` -- the step column only.

    Read straight from the manifest's files (no eligibility filter: pilot segments count too).
    """
    from ..parquet_manifest import committed_files
    seg_dir = Path(out_dir) / kind / str(segment_id)
    files = committed_files(seg_dir, expected_kind=kind) if seg_dir.is_dir() else None
    if not files:
        return 0
    import pyarrow.parquet as pq
    n = 0
    for path in files:
        steps = np.asarray(pq.read_table(path, columns=["step"]).column("step").to_numpy(zero_copy_only=False))
        n += int(np.count_nonzero(steps.astype(np.int64) <= int(step)))
    return n


def check_data_boundary(out_dir, manifest: Mapping[str, Any]) -> None:
    """The checkpoint segment's committed rows at or before the checkpoint step must reach the counts its
    ``data_boundary`` recorded at save time (spec Section 9; final fix wave I1).

    Fewer rows means the samples/exchanges on disk are older than the checkpoint (a restored or rolled-back
    data directory, a lost flush): resuming would leave a hole the checkpoint claims is filled.
    """
    block = manifest["aux"]
    boundary = block.get("data_boundary")
    if not isinstance(boundary, Mapping):
        raise IntegrityError("resume refused: the auxiliary checkpoint records no data_boundary")
    seg, step = str(block["segment_id"]), int(manifest.get("absolute_step", 0))
    for kind in ("samples", "exchanges"):
        want = int(boundary[kind]["n_rows"])
        have = committed_rows_up_to(out_dir, kind, seg, step)
        if have < want:
            raise IntegrityError(
                f"resume refused: segment {seg} holds {have} committed {kind} rows at or before the checkpoint step "
                f"{step}, but the checkpoint's data_boundary recorded {want}; the {kind} on disk are older than "
                "the checkpoint")


def anchor_ledger_events(out_dir, manifest: Mapping[str, Any]) -> dict:
    """The exchange events of the checkpoint's ledger-anchor segment only (carry-over 2).

    ``replay_assignments`` filters by step alone, so events of any other segment (an earlier run, a
    restarted child at the same steps) must never reach it. Rows after the checkpoint step are left in:
    the replay's ``up_to_step`` drops them.
    """
    from ..query import load_exchanges
    segment_id = str(manifest["aux"]["ledger_anchor"]["segment_id"])
    return load_exchanges(Path(out_dir), segment_ids=[segment_id])


def prepare_aux_resume(out_dir, registry, manifest: Mapping[str, Any], *, state_definition, force_info,
                       topology_sha256: str, kernel_identity_digest: str) -> dict[str, Any]:
    """Every auxiliary resume refusal that needs no Context, run BEFORE the resumed segment is registered.

    Task 14 F4 (fix round 2): read-only -- it never changes the registry. Order: static checkpoint binding
    (``verify_aux_resume_static``), the ledger replay from the anchor segment, the duplicate-event check on
    the registry view the re-seal will leave. The checkpoint's data_boundary is enforced right after the
    static binding (final fix wave I1). Returns the re-seal plan: ``run_gareus`` registers the resumed
    segment under ``parent_segment_id`` (the checkpoint's segment, never an orphan) and applies
    ``store.reseal_chain_for_resume`` only after the Context-dependent checks in the checkpoint load passed.
    """
    from .checkpoint import verify_aux_ledger, verify_aux_resume_static
    from .ledger import refuse_duplicate_event_keys
    verify_aux_resume_static(manifest, aux_enabled=True, state_definition=state_definition, force_info=force_info,
                             topology_sha256=topology_sha256, kernel_identity_digest=kernel_identity_digest)
    check_data_boundary(out_dir, manifest)
    verify_aux_ledger(manifest, anchor_ledger_events(out_dir, manifest))
    ckpt_seg, ckpt_step = str(manifest["aux"]["segment_id"]), int(manifest.get("absolute_step", 0))
    refuse_duplicate_event_keys(_ledger_after_reseal(out_dir, registry, ckpt_seg, ckpt_step))
    return {"checkpoint_segment_id": ckpt_seg, "checkpoint_step": ckpt_step, "parent_segment_id": ckpt_seg}


def _ledger_after_reseal(out_dir, registry, ckpt_seg: str, ckpt_step: int) -> dict:
    """(step, attempt_seq, segment_id) of every exchange event the registry will pool once the re-seal ran.

    Read segment by segment (explicit ids bypass the loader's status filter), with the cut the re-seal will
    apply: segments after the checkpoint's are dropped (abandoned), the checkpoint's own is cut to the
    checkpoint step, earlier ones keep their committed boundary.
    """
    from ..query import load_exchanges
    segs = registry.all_segments()
    ids = [s["segment_id"] for s in segs]
    if ckpt_seg not in ids:
        raise IntegrityError(f"resume refused: the checkpoint's segment {ckpt_seg!r} is not in the segment registry {ids}")
    cols: dict[str, list] = {"step": [], "attempt_seq": [], "segment_id": []}
    for seg in segs[:ids.index(ckpt_seg) + 1]:
        sid, status, end = seg["segment_id"], seg.get("status"), seg.get("end_step")
        if sid == ckpt_seg:
            cut = None if status == "complete" else (ckpt_step if end is None else min(int(end), ckpt_step))
        elif status == "complete":
            cut = None
        elif status == "interrupted" and end is not None and int(end) >= 0:
            cut = int(end)
        else:
            continue                                   # running non-last / abandoned: never pooled
        ev = load_exchanges(Path(out_dir), segment_ids=[sid])
        if not ev or "step" not in ev or "attempt_seq" not in ev or len(ev["step"]) == 0:
            continue
        step = np.asarray(np.ma.getdata(ev["step"])).astype(np.int64)
        keep = np.ones(step.size, dtype=bool) if cut is None else step <= cut
        cols["step"].append(step[keep])
        cols["attempt_seq"].append(ev["attempt_seq"][keep])
        cols["segment_id"].append(np.full(int(keep.sum()), sid, dtype=object))
    if not cols["step"]:
        return {}
    return {"step": np.concatenate(cols["step"]), "attempt_seq": np.ma.concatenate(cols["attempt_seq"]),
            "segment_id": np.concatenate(cols["segment_id"])}


def refuse_aux_resume_without_checkpoint(out_dir) -> None:
    """An auxiliary --resume that finds no production checkpoint while the registry holds committed data.

    Task 13 minor 7. Restarting at step 0 would write a second copy of every (step, replica) key the
    committed segment already holds; sealing that segment abandoned would silently drop committed data.
    Refused at start instead (consistent with the re-seal, which never discards rows at or before a
    checkpoint). Segments without rows (a job killed during setup, a refused resume) do not count.
    """
    from ..store import SegmentRegistry
    for seg in SegmentRegistry(Path(out_dir)).all_segments():
        if seg.get("status") == "abandoned":
            continue
        rows = data_boundary(out_dir, seg["segment_id"])
        if rows["samples"]["n_rows"] or rows["exchanges"]["n_rows"]:
            raise IntegrityError(
                f"resume refused: no production checkpoint, but segment {seg['segment_id']} ({seg.get('status')}, "
                f"end_step {seg.get('end_step')}) holds committed auxiliary data; restarting at step 0 would pool "
                "its steps twice. Restore the checkpoint, or start a new run directory.")


def discard_refused_segment(out_dir, registry, segment_id: str) -> bool:
    """Roll back the segment a refused auxiliary resume registered (Task 14 F4): no side effects remain.

    Only the newest segment, and only when it holds no committed row, is removed: its registry entry, its
    (empty) sample and exchange directories and its window snapshot. Returns whether it was removed.
    """
    import shutil
    latest = registry.get_latest_segment()
    if latest is None or latest["segment_id"] != segment_id:
        return False
    rows = data_boundary(out_dir, segment_id)
    if rows["samples"]["n_rows"] or rows["exchanges"]["n_rows"]:
        return False
    if not registry.discard_latest_segment(segment_id):
        return False
    for kind in ("samples", "exchanges"):
        shutil.rmtree(Path(out_dir) / kind / str(segment_id), ignore_errors=True)
    (Path(out_dir) / "windows" / f"{segment_id}.json").unlink(missing_ok=True)
    return True
