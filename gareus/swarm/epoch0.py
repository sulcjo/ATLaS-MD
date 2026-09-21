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
from typing import Any, Dict, List, Optional

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


#: Artefacts an automatic-CV2 epoch 0 must also leave behind (spec F04).
AUTO_CV_ARTEFACTS = ("ladder_run_args.yaml",)
PAIR_ARTEFACTS = ("cv_pair_model.json", "cv_candidate_set.json", "cv_feature_schema.json")


def _artefact_digests(out_dir, names) -> Dict[str, str]:
    from gareus.correctness._io import file_digest

    an = analysis_dir(out_dir)
    out: Dict[str, str] = {}
    for rel in names:
        p = an / rel
        if p.is_file():
            out[rel] = file_digest(p)
        elif p.is_dir():
            csv_path = p / "final_survivor_seeds.csv"
            if csv_path.is_file():
                out[rel + "/final_survivor_seeds.csv"] = file_digest(csv_path)
    return out


def mark_epoch0_complete(out_dir, *, n_members: int, ns_charged: float,
                         auto_cv: bool = False, selection_status: Optional[str] = None) -> Path:
    """Record that epoch 0 finished, after checking that it really did.

    Written last, and only once the artefacts exist: the marker is a claim about
    other files, so it must never be able to outlive them. With automatic CV2 the
    sidecar is required too, and a selected pair requires its three frozen artifacts.
    The marker records every artefact's content digest and the selection outcome, so a
    later job can tell that the files it is about to trust are the ones the gate passed.
    """
    required = list(REQUIRED_ARTEFACTS)
    if auto_cv:
        required += list(AUTO_CV_ARTEFACTS)
        if selection_status == "pair":
            required += list(PAIR_ARTEFACTS)
    an = analysis_dir(out_dir)
    missing = [rel for rel in required if not (an / rel).exists()]
    if missing:
        raise FileNotFoundError(
            "refusing to mark epoch 0 complete; missing swarm artefacts: " + ", ".join(missing)
        )
    path = an / EPOCH0_MARKER_NAME
    write_json(path, {
        "n_members": int(n_members),
        "ns_charged": float(ns_charged),
        "artefacts": required,
        "artefact_digests": _artefact_digests(out_dir, required),
        "auto_cv": bool(auto_cv),
        "selection_status": selection_status,
    })
    return path


def changed_artefacts(out_dir, marker: Dict[str, Any]) -> List[str]:
    """Artefacts whose current content digest differs from the one the marker recorded."""
    recorded = dict(marker.get("artefact_digests") or {})
    if not recorded:
        return []
    current = _artefact_digests(out_dir, marker.get("artefacts") or [])
    # A MISSING artefact is the old recoverable case (re-analyse); only a present-but-different
    # file is a changed artefact that blocks production from trusting the marker.
    return sorted(rel for rel, dig in recorded.items() if rel in current and current[rel] != dig)


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

    changed = changed_artefacts(out_dir, marker) if marker else []
    status["changed_artefacts"] = changed
    if gate_status == "fail":
        # A recorded failure is the round's own verdict on itself. Re-running
        # members would not change it; only a new round can.
        status.update(state="blocked", next_action=EPOCH0_STOP)
        status["reasons"].append("swarm gate recorded status=fail")
    elif marker and changed:
        # The files production is about to trust are not the ones the gate passed.
        status.update(state="blocked", next_action=EPOCH0_STOP)
        status["reasons"].append("epoch-0 artefacts changed after completion: " + ", ".join(changed))
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


class Epoch0GateFailure(RuntimeError):
    """The swarm round cannot hand off, so the campaign must not continue.

    Raised rather than returned because there is no useful degraded mode: every
    later epoch samples the ladder this round was supposed to design.
    """


#: Each pass performs one action and re-derives the state, so a cold start needs
#: two (run members, then analyse) and a resume needs one. The bound exists only
#: so a state that fails to advance stops instead of spinning.
_MAX_EPOCH0_PASSES = 4


SIDECAR_NAME = "ladder_run_args.yaml"
_SIDECAR_PATH_KEYS = ("secondary_cv_model", "secondary_cv_candidate_set", "secondary_cv_feature_schema")


def apply_epoch0_sidecar(args, out_dir) -> Dict[str, Any]:
    """Apply what epoch 0 decided about the CVs to the campaign arguments, in place.

    The swarm hands production a window table AND a sidecar (``ladder_run_args.yaml``):
    the resolved ``cvs`` (``auto`` -> the selected ``residual-torsion-pc``, or ``none`` on
    the cv1-only fallback), the three frozen pair-model paths and ``tica_switch_cv2``.
    The driver used to take only the table, so ``args.secondary_cv`` stayed ``"auto"``
    into production and the loader rejected the residual centres against ``[0, 1]``
    (chignolin_8, job 2553783). Idempotent: every job in the chain calls this.
    Returns the fields applied (empty when there is no sidecar or it names no CVs)."""
    from gareus.config import _load_config_file

    path = analysis_dir(out_dir) / SIDECAR_NAME
    if not path.exists():
        return {}
    side = _load_config_file(path) or {}
    applied: Dict[str, Any] = {}
    cvs = side.get("cvs") or {}
    cv2 = cvs.get("cv2")
    if cv2 not in (None, ""):
        args.secondary_cv = str(cv2)
        applied["secondary_cv"] = str(cv2)
    for key in _SIDECAR_PATH_KEYS:
        if side.get(key) not in (None, ""):
            setattr(args, key, str(side[key]))
            applied[key] = str(side[key])
    if "tica_switch_cv2" in side:
        args.tica_switch_cv2 = bool(side["tica_switch_cv2"])
        applied["tica_switch_cv2"] = bool(side["tica_switch_cv2"])
    applied.update(_apply_sidecar_gamd_envelope(args, side, path))
    if str(getattr(args, "secondary_cv", "")) == "auto":
        raise RuntimeError(
            f"epoch 0 sidecar {path} does not resolve cv2 'auto' (cvs.cv2 missing); the swarm "
            "analysis must be re-run before production can build its restraints"
        )
    return applied


def _apply_sidecar_gamd_envelope(args, side: Dict[str, Any], sidecar_path: Path) -> Dict[str, Any]:
    """Point production at the swarm's frozen GaMD envelope (``gamd.shared_gamd_setup_dir``).

    The envelope is measured once by epoch 0 and never re-derived; production reuses it through
    ``load_reusable_shared_gamd_setup`` and skips its own shared setup and per-state recon. The
    sidecar wins only where nobody chose a directory: an empty ``shared_gamd_setup_dir`` or one the
    adaptive driver auto-resolved to its campaign default (``_global_shared_gamd_setup_dir``). An
    explicit user directory is left alone and the decision is recorded in the returned mapping.
    A sidecar that names a directory without a globals file is an error, not a silent
    recalibration (job 2558439 recalibrated silently for exactly this class of gap)."""
    gamd = side.get("gamd") or {}
    shared_dir = gamd.get("shared_gamd_setup_dir")
    if shared_dir in (None, ""):
        return {}
    shared_dir = str(shared_dir)
    globals_path = Path(shared_dir) / "shared_gamd_setup_globals.json"
    if not globals_path.exists():
        raise RuntimeError(
            f"epoch 0 sidecar {sidecar_path} names gamd.shared_gamd_setup_dir={shared_dir} but "
            f"{globals_path.name} is missing there; the frozen envelope must exist before production "
            "(re-run the swarm analysis)"
        )
    current = str(getattr(args, "shared_gamd_setup_dir", "") or "").strip()
    auto_dir = str(getattr(args, "_global_shared_gamd_setup_dir", "") or "").strip()
    if current and current != auto_dir and Path(current).resolve() != Path(shared_dir).resolve():
        return {"shared_gamd_setup_dir_kept_explicit": current}
    args.shared_gamd_setup_dir = shared_dir
    return {"shared_gamd_setup_dir": shared_dir}


def run_or_resume_epoch0(
    args,
    out_dir,
    *,
    progress=None,
    run_members=None,
    analyze=None,
    charge_ns=None,
    round_index: int = None,
) -> Path:
    """Drive epoch 0 to completion and return the ladder production will sample.

    Safe to call at the start of every job in the chain: the work already done
    is read off the disk, so a finished swarm costs nothing and a half-finished
    one continues from where it stopped. ``run_members`` and ``analyze`` are
    injectable so the decisions can be tested without running MD.

    ``charge_ns`` is called at most once per campaign, with the nanoseconds the
    swarm consumed. It fires on the pass that completes the round and never on a
    resume, because by then the marker already records the cost.
    """
    out_dir = Path(out_dir)
    if round_index is None:
        round_index = int(getattr(args, "swarm_round", 0) or 0)
    if run_members is None:
        from gareus.swarm.driver import run_swarm_stage as run_members
    if analyze is None:
        from gareus.swarm.analyze import analyze_swarm_stage as analyze

    ladder = analysis_dir(out_dir) / "windows_lambda_ladder.csv"

    for _ in range(_MAX_EPOCH0_PASSES):
        status = epoch0_status(out_dir, round_index)
        action = status["next_action"]

        if action == EPOCH0_STOP:
            raise Epoch0GateFailure(
                "swarm epoch 0 gate failed: " + "; ".join(status["reasons"] or ["no reason recorded"])
            )

        if action == EPOCH0_PROCEED:
            return ladder

        if action == EPOCH0_RUN_MEMBERS:
            # Resumable in its own right: members with a done.json are skipped
            # inside the driver, so this both starts and continues a round.
            run_members(args, out_dir, progress)
            continue

        report = analyze(out_dir, args) or {}
        if str(report.get("status", "")) not in ("ok", "pass"):
            raise Epoch0GateFailure(
                "swarm epoch 0 analysis did not pass: "
                + str(report.get("status", "unknown"))
                + (("; " + "; ".join(str(r) for r in report.get("reasons", []))) if report.get("reasons") else "")
            )

        settled = epoch0_status(out_dir, round_index)
        ns = float(settled["n_done"]) * float(getattr(args, "swarm_seed_ns", 1.0) or 0.0)
        # Marker last, and only now: analyze has written the artefacts it names.
        auto_cv = str(getattr(args, "secondary_cv", "") or "") == "auto"
        selection_status = (report.get("cv_selection") or {}).get("status") if isinstance(report, dict) else None
        mark_epoch0_complete(out_dir, n_members=settled["n_done"], ns_charged=ns,
                             auto_cv=auto_cv, selection_status=selection_status)
        if charge_ns is not None:
            charge_ns(ns)
        return ladder

    raise RuntimeError(
        f"swarm epoch 0 did not reach a terminal state in {_MAX_EPOCH0_PASSES} passes "
        f"(last state {epoch0_status(out_dir, round_index)['state']!r})"
    )


def epoch0_seed_bank(out_dir) -> Path:
    """Where epoch 0 exports the per-window seed bank."""
    return analysis_dir(out_dir) / "seed_bank"


def epoch0_seed_bank_if_present(out_dir) -> Path:
    """The swarm's seed bank, or None when the swarm has not exported one yet.

    ``seed_conformers_dir`` carries two different meanings across one campaign:
    the GENPEPT library the swarm grafts its members from, and the bank the swarm
    exports for the umbrella windows. Production must switch to the latter once
    epoch 0 has produced it -- otherwise every window would be seeded from generic
    library conformers instead of structures the swarm actually visited there.
    """
    bank = epoch0_seed_bank(out_dir)
    return bank if bank.exists() else None


#: The swarm's own precondition (see run_swarm_stage): members are grafted from a
#: GENPEPT library, and the library is identified by this index.
SEED_LIBRARY_INDEX = "final_survivor_seeds.csv"


def epoch0_is_available(args) -> bool:
    """Whether this campaign has a swarm to run at all.

    "No window table was handed in" is not sufficient to conclude "run a swarm":
    plenty of adaptive-production runs legitimately start without one and take
    their states from a registry, an epoch directory, or a resume. Engaging the
    swarm for those turns a working run into a SystemExit about a missing seed
    library.

    So the test is the swarm's own precondition, checked before it is called
    rather than raised from inside it: a seed library with a non-empty index. A
    seed bank exported by a previous campaign has no index and correctly does not
    qualify.
    """
    raw = getattr(args, "seed_conformers_dir", None)
    if not raw:
        return False
    index = Path(str(raw)) / SEED_LIBRARY_INDEX
    try:
        return index.is_file() and index.stat().st_size > 0
    except OSError:
        return False
