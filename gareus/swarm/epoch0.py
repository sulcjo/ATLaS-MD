"""Epoch 0 -- the unbiased swarm -- as a resumable phase of one self-contained run.

The swarm is not a separate job with its own script and config: it is the first
epoch of the production run, and it is long enough that a walltime stop lands in
the middle of it routinely. The job that picks the campaign up afterwards has no
memory of what the previous one did, so this module answers, from the filesystem
alone, the only question that matters at startup: what is left to do?

Getting that answer wrong is costly in both directions. Re-running a finished
epoch spends budget that was already spent. Proceeding from a half-finished one
is worse and quieter: the ladder would be designed from whichever subset of the
swarm happened to finish before the kill, and nothing downstream would report it.

The states are derived, never stored -- except for one marker, written last:

    not_started       no plan on disk yet                 -> run members
    members_pending   some planned members lack done.json -> run members
    members_complete  every member accounted for          -> analyse
    complete          marker + artefacts + gate pass      -> go to epoch 1
    blocked           the gate recorded a failure         -> stop

``complete`` deliberately requires the marker AND the artefacts it claims. A
marker is only ever written after them (``mark_epoch0_complete`` refuses
otherwise), so a marker whose artefacts are missing does not mean "finished" --
it means something removed them afterwards, and the honest response is to
analyse again rather than to hand production a ladder that is not there.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from gareus.io import read_json_file, write_json
from gareus.swarm.driver import round_dir, swarm_root
from gareus.swarm.members import member_done

EPOCH0_MARKER_NAME = "epoch0_complete.json"

EPOCH0_RUN_MEMBERS = "run_members"
EPOCH0_ANALYZE = "analyze"
EPOCH0_PROCEED = "proceed"
EPOCH0_STOP = "stop"

#: What production actually consumes. The marker is only meaningful while all
#: three are on disk, so they are checked together rather than trusted.
REQUIRED_ARTEFACTS = (
    "windows_lambda_ladder.csv",
    "shared_gamd_setup/shared_gamd_setup_globals.json",
    "seed_bank",
)


def analysis_dir(out_dir) -> Path:
    return swarm_root(Path(out_dir)) / "analysis"


def missing_artefacts(out_dir) -> List[str]:
    an = analysis_dir(out_dir)
    return [rel for rel in REQUIRED_ARTEFACTS if not (an / rel).exists()]


def _planned_member_ids(rd: Path) -> List[int]:
    """Member ids from the frozen plan, or [] when no complete plan exists.

    ``plan_meta.json`` is written after ``plan.csv`` and is the completion
    marker for the pair, so a plan without it is treated as absent.
    """
    if not (rd / "plan_meta.json").exists() or not (rd / "plan.csv").exists():
        return []
    ids: List[int] = []
    with (rd / "plan.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                ids.append(int(row["member_id"]))
            except (KeyError, TypeError, ValueError):
                continue
    return ids


def mark_epoch0_complete(out_dir, *, n_members: int, ns_charged: float) -> Path:
    """Record that epoch 0 finished, after checking that it really did.

    Written last, and only once the artefacts exist: the marker is a claim about
    other files, so it must never be able to outlive them.
    """
    missing = missing_artefacts(out_dir)
    if missing:
        raise FileNotFoundError(
            "refusing to mark epoch 0 complete; missing swarm artefacts: " + ", ".join(missing)
        )
    path = analysis_dir(out_dir) / EPOCH0_MARKER_NAME
    write_json(path, {
        "n_members": int(n_members),
        "ns_charged": float(ns_charged),
        "artefacts": list(REQUIRED_ARTEFACTS),
    })
    return path


def epoch0_status(out_dir, round_index: int = 0) -> Dict[str, Any]:
    """Decide, from disk alone, what a starting or restarting job should do."""
    out_dir = Path(out_dir)
    rd = round_dir(out_dir, round_index)
    an = analysis_dir(out_dir)

    planned = _planned_member_ids(rd)
    done_ids: List[int] = []
    n_failed = 0
    for member_id in planned:
        member_dir = rd / f"member_{member_id:04d}"
        if not member_done(member_dir):
            continue
        done_ids.append(member_id)
        summary = read_json_file(member_dir / "done.json", {}) or {}
        if str(summary.get("status", "ok")) != "ok":
            n_failed += 1

    finished = set(done_ids)
    pending = [m for m in planned if m not in finished]

    gate = read_json_file(an / "swarm_gate.json", None) or {}
    gate_status = str(gate.get("status", "")) or None
    marker = read_json_file(an / EPOCH0_MARKER_NAME, None) or {}
    absent = missing_artefacts(out_dir)

    status: Dict[str, Any] = {
        "round": int(round_index),
        "n_planned": len(planned),
        "n_done": len(done_ids),
        "n_failed": n_failed,
        "pending_member_ids": pending,
        "gate_status": gate_status,
        "missing_artefacts": absent,
        "ns_charged": float(marker.get("ns_charged", 0.0)),
        "reasons": [],
    }

    if gate_status == "fail":
        # A recorded failure is the round's own verdict on itself. Re-running
        # members would not change it; only a new round can.
        status.update(state="blocked", next_action=EPOCH0_STOP)
        status["reasons"].append("swarm gate recorded status=fail")
    elif marker and not absent and gate_status in ("pass", "ok"):
        status.update(state="complete", next_action=EPOCH0_PROCEED)
    elif marker and absent:
        status.update(state="members_complete", next_action=EPOCH0_ANALYZE)
        status["reasons"].append(
            "epoch-0 marker present but its artefacts are missing: " + ", ".join(absent)
        )
    elif not planned:
        status.update(state="not_started", next_action=EPOCH0_RUN_MEMBERS)
    elif pending:
        status.update(state="members_pending", next_action=EPOCH0_RUN_MEMBERS)
    else:
        status.update(state="members_complete", next_action=EPOCH0_ANALYZE)
    return status
