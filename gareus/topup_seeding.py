"""Continue each top-up window's chain from its parent segment's final State (spec 4.4, revision 2)."""
from __future__ import annotations

import csv
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

DIR_NAME = "final_window_states"


class SeedMismatchError(RuntimeError):
    """Raised when a top-up segment cannot safely seed a window from a parent export.

    ``window``/``state_id`` are set whenever the failure is tied to a specific
    window/state pair (every raise site that knows them passes them through), so a
    caller can identify exactly which one without re-parsing the message text. Both
    default to ``None`` for a segment-wide failure (e.g. no ``epoch_window_map.csv``
    at all) that is not about one particular window.
    """

    def __init__(self, message: str, window: Optional[int] = None, state_id: Optional[int] = None):
        super().__init__(message)
        self.window = window
        self.state_id = state_id


@dataclass(frozen=True)
class SeedState:
    positions: Any
    velocities: Any
    box: Any
    cv1: float
    cv2: float
    source: str
    state_id: Optional[int] = None
    # The restraint this seed was recorded under (spec revision 2, ruling 15) -- lets a
    # later top-up verify a loaded seed really belongs to the window it's being placed
    # into, not just that its CVs happen to fall near that window's target. None on a
    # legacy export (written before this field existed) or for the secondary terms on a
    # CV1-only run (no secondary restraint at all).
    primary_center: Optional[float] = None
    primary_k: Optional[float] = None
    secondary_center: Optional[float] = None
    secondary_k: Optional[float] = None


def _int_or_none(value) -> Optional[int]:
    # Local, deliberately-duplicated copy of gareus.production's helper of the same
    # name: production.py imports this module, so importing back from production would
    # be circular. gareus/tica.py's _read_csv_dicts_local follows the same pattern.
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def _optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _deserialize_state(text: str):
    from openmm import XmlSerializer  # noqa: PLC0415
    st = XmlSerializer.deserialize(text)
    return st.getPositions(), st.getVelocities(), st.getPeriodicBoxVectors()


def state_id_of_window_from_epoch_map(out_dir) -> Dict[int, int]:
    """Window -> state_id for a segment, from its own ``epoch_window_map.csv``.

    Identity mapping when the file is absent/empty -- the same fallback used
    throughout the adaptive-production driver for this file. A top-up's own seeding
    path must NOT rely on this identity fallback when the file is genuinely absent
    (spec revision 2, ruling 15); that caller checks the file's existence itself
    before calling this and fails closed instead.
    """
    out_dir = Path(out_dir)
    path = out_dir / "epoch_window_map.csv"
    out: Dict[int, int] = {}
    if not path.exists():
        return out
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        w = _int_or_none(row.get("epoch_window"))
        sid = _int_or_none(row.get("state_id"))
        if w is not None and sid is not None:
            out[w] = sid
    return out


def topup_parent_dirs_by_creation_order(out_dir) -> List[Path]:
    """This top-up's candidate parent dirs (baseline + every sibling topup_*), oldest
    first -- ``load_seed_index``'s "later parents override earlier ones" then makes the
    most recently exported parent win, not merely the alphabetically-last one.

    Ordered by the max ``export_seq`` (a ``time.time_ns()`` stamp written into every
    record of an export, spec revision 2 ruling 16) found in each parent's own
    ``final_window_states/index.json``, NOT by directory name and not primarily by
    mtime. ``topup_<idx>_<extra_steps>`` directory names are not chronological: the
    numeric suffix is that segment's own remaining-step duration, which *shrinks*
    across successive quality-gate extension rounds -- CLAUDE.md's 2026-08-05
    ``plot_adaptive_diagnostics.py`` fix documents the identical footgun on a real run
    (``topup_001_34794000`` created first, ``...25733000`` second, ``...18937000``
    last; name-ascending sort is exactly the reverse of creation order). A name-sorted
    parent list would let a stale, earlier-created topup's export silently outrank a
    later, more-authoritative one, and ``assert_seed_matches``/
    ``assert_seed_restraint_matches`` cannot catch that on their own -- they only ever
    check a State against its own recorded values, never against which export is
    actually the most recent.

    A parent whose index has no ``export_seq`` at all (a legacy export from before
    ruling 16) falls back to that index file's own mtime, converted to the same
    nanosecond-epoch scale as ``export_seq`` so the two remain comparable even when a
    campaign mixes old and new exports. A parent with no export yet sorts first of
    all (an unexported parent can never override one that has a real export).
    """
    out_dir = Path(out_dir)
    # Directories only: the phase dir also holds ``topup_plan.json`` and each
    # segment's ``topup_*_windows.csv``, which match the glob but are no parent.
    candidates = [
        d for d in [out_dir.parent / "baseline", *(p for p in out_dir.parent.glob("topup_*") if p.is_dir())]
        if d != out_dir
    ]

    def _order_key(d: Path) -> float:
        # A corrupt/unreadable index here only affects sort position (it sorts
        # first, like an unexported parent) -- it does NOT decide whether this
        # parent's states are usable. load_seed_index (ruling 19) is the single
        # place that raises on a corrupt index, the moment it is actually asked to
        # resolve a state from it; ordering happens earlier and must stay
        # exception-free so a merely-cosmetic ordering pass can't itself crash a
        # segment that never needed the broken parent's states at all.
        idx = d / DIR_NAME / "index.json"
        try:
            record = json.loads(idx.read_text())
        except Exception:
            return -1.0
        if not isinstance(record, dict):
            return -1.0
        seqs = [
            v.get("export_seq") for v in record.values()
            if isinstance(v, dict) and isinstance(v.get("export_seq"), (int, float))
        ]
        if seqs:
            return float(max(seqs))
        try:
            return float(idx.stat().st_mtime) * 1e9  # seconds -> ns scale, for mixed old/new exports
        except OSError:
            return -1.0

    return sorted(candidates, key=_order_key)


def should_export_final_window_states(args) -> bool:
    """True only for a top-ups-on adaptive-production segment.

    ``final_window_states/`` exists solely to seed a later top-up, and costs
    ~3 MB per window State at 19k atoms.  ``_adaptive_phase_info`` is set by
    the adaptive-production driver on every epoch/final segment it launches
    (``is_adaptive_epoch``); a standalone run, a pilot, or any segment of a
    top-ups-off campaign writes nothing.
    """
    info = getattr(args, "_adaptive_phase_info", None) or {}
    return bool(getattr(args, "adaptive_production_topups", False)) and bool(info.get("is_adaptive_epoch"))


def export_final_window_states(out_dir, sims: Sequence, assignments: Sequence[int],
                               state_id_of_window: Dict[int, int],
                               cv_of_replica: Callable[[int], tuple],
                               centers_nm: Sequence[float], ks_kj_nm2: Sequence[float],
                               secondary_centers: Optional[Sequence[float]] = None,
                               secondary_ks_kj: Optional[Sequence[float]] = None) -> Path:
    """Write each window's current State, its restraint parameters, and its CVs.

    ``centers_nm``/``ks_kj_nm2``/``secondary_centers``/``secondary_ks_kj`` must be the
    exact same arrays :func:`gareus.windows.set_window` was called with for these
    replicas -- recording them lets a later top-up verify (spec revision 2, ruling 15)
    that a loaded seed really belongs to the window it is being seeded into. Both the
    State XML and the index are written via a tmp-file-then-replace, so a crash mid-write
    can never leave a truncated file behind for a later top-up to trip over (ruling 14).
    """
    from openmm import XmlSerializer  # noqa: PLC0415
    d = Path(out_dir) / DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    centers_list = list(centers_nm)
    k_list = list(ks_kj_nm2)
    sec_c_list = list(secondary_centers) if secondary_centers is not None else None
    sec_k_list = list(secondary_ks_kj) if secondary_ks_kj is not None else None
    export_seq = time.time_ns()
    index = {}
    for r, sim in enumerate(sims):
        w = int(assignments[r])
        sid = int(state_id_of_window.get(w, w))
        st = sim.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=False)
        tmp_state = d / f"state_{sid}.xml.tmp"
        tmp_state.write_text(XmlSerializer.serialize(st))
        tmp_state.replace(d / f"state_{sid}.xml")
        cv1, cv2 = cv_of_replica(r)
        index[str(sid)] = {
            "window": w,
            "cv1": float(cv1),
            "cv2": float(cv2),
            "primary_center": float(centers_list[w]) if w < len(centers_list) else None,
            "primary_k": float(k_list[w]) if w < len(k_list) else None,
            "secondary_center": (float(sec_c_list[w])
                                 if sec_c_list is not None and w < len(sec_c_list) else None),
            "secondary_k": (float(sec_k_list[w])
                            if sec_k_list is not None and w < len(sec_k_list) else None),
            "export_seq": export_seq,
        }
    tmp = d / "index.json.tmp"
    tmp.write_text(json.dumps(index, indent=2))
    tmp.replace(d / "index.json")
    return d


def load_seed_index(parent_dirs: Iterable) -> Dict[int, str]:
    """Window/state -> parent dir map, "later parents override earlier ones".

    A corrupt or unreadable ``index.json`` in ANY candidate parent raises
    ``SeedMismatchError`` naming that parent (ruling 19), rather than silently
    skipping it the way a per-state problem does (ruling 14). Skipping a whole
    corrupt parent would let an OLDER parent's copy of the same state silently win
    the "later parents override earlier ones" resolution -- restarting that
    window's chain from an earlier point instead of failing loudly, which is
    exactly the kind of silent-regression bug this whole seeding mechanism exists
    to prevent. A problem confined to one state's own record or XML (the index
    itself parses fine) is unaffected and still degrades to "treated as missing"
    via :func:`load_seed_states`.
    """
    out: Dict[int, str] = {}
    for parent in parent_dirs:                       # later parents override earlier ones
        idx = Path(parent) / DIR_NAME / "index.json"
        if not idx.exists():
            continue
        try:
            record = json.loads(idx.read_text())
        except Exception as exc:
            raise SeedMismatchError(
                f"top-up seed index {idx} (parent {parent}) is unreadable ({exc}); refusing to silently "
                "skip this parent -- that could let an older parent's copy of the same state win and "
                "restart its chain from an earlier point"
            )
        if not isinstance(record, dict):
            raise SeedMismatchError(
                f"top-up seed index {idx} (parent {parent}) does not hold a JSON object; refusing to "
                "silently skip this parent"
            )
        for key in record:
            try:
                sid = int(key)
            except (TypeError, ValueError):
                continue  # a non-integer sibling key, not a per-state record
            if (Path(parent) / DIR_NAME / f"state_{sid}.xml").exists():
                out[sid] = str(parent)
    return out


def load_seed_states(parent_dirs: Iterable, state_ids: Iterable[int]) -> Dict[int, SeedState]:
    """Load each requested state's seed, skipping (with a logged warning) any entry
    whose record is corrupt/incomplete or whose State XML fails to deserialize --
    never raising a JSON/parse/OpenMM exception out of this function (ruling 14). A
    skipped state is simply absent from the returned dict; the caller's own
    missing-seed handling (``SeedMismatchError``) is what turns that into a hard
    failure, so a corrupt export degrades a top-up the same way a missing one does,
    rather than crashing the whole segment on an unrelated exception type.
    """
    parents = list(parent_dirs)
    where = load_seed_index(parents)
    out: Dict[int, SeedState] = {}
    for raw_sid in state_ids:
        sid = int(raw_sid)
        parent = where.get(sid)
        if parent is None:
            continue
        d = Path(parent) / DIR_NAME
        try:
            index_data = json.loads((d / "index.json").read_text())
            rec = index_data[str(sid)]
            cv1 = float(rec["cv1"])
            cv2 = float(rec["cv2"])
        except Exception as exc:
            logger.warning("Top-up seed record for state %s in %s is unreadable/incomplete (%s); "
                          "treating as missing", sid, parent, exc)
            continue
        try:
            pos, vel, box = _deserialize_state((d / f"state_{sid}.xml").read_text())
        except Exception as exc:
            logger.warning("Top-up seed XML for state %s in %s failed to deserialize (%s); "
                          "treating as missing", sid, parent, exc)
            continue
        out[sid] = SeedState(
            positions=pos, velocities=vel, box=box, cv1=cv1, cv2=cv2, source=str(parent),
            state_id=sid,
            primary_center=_optional_float(rec.get("primary_center")),
            primary_k=_optional_float(rec.get("primary_k")),
            secondary_center=_optional_float(rec.get("secondary_center")),
            secondary_k=_optional_float(rec.get("secondary_k")),
        )
    return out


def assert_seed_matches(seed: SeedState, cv1_now: float, cv2_now: float, tol: float = 1e-3,
                        window: Optional[int] = None) -> None:
    """CV1 must always be finite and match; CV2 is compared only when both sides are finite.

    CV2 is legitimately NaN for a CV1-only run (no secondary CV configured) on both the
    recorded seed and the freshly-evaluated context, so that comparison is skipped rather
    than raised. CV1 is never expected to be NaN -- a NaN there means the loaded/seeded
    state is physically broken (e.g. blown-up positions), which must raise loudly rather
    than be silently accepted by NaN comparison semantics.
    """
    if not math.isfinite(cv1_now) or not math.isfinite(seed.cv1):
        raise SeedMismatchError(
            f"seeded state from {seed.source} (window {window}, state {seed.state_id}) does not "
            f"reproduce its recorded CVs (cv1 {cv1_now!r} vs {seed.cv1!r} -- non-finite CV1)",
            window=window, state_id=seed.state_id)
    cv1_mismatch = abs(cv1_now - seed.cv1) > tol
    cv2_comparable = math.isfinite(cv2_now) and math.isfinite(seed.cv2)
    cv2_mismatch = cv2_comparable and abs(cv2_now - seed.cv2) > tol
    if cv1_mismatch or cv2_mismatch:
        raise SeedMismatchError(
            f"seeded state from {seed.source} (window {window}, state {seed.state_id}) does not "
            f"reproduce its recorded CVs (cv1 {cv1_now:.6f} vs {seed.cv1:.6f}, "
            f"cv2 {cv2_now:.6f} vs {seed.cv2:.6f})",
            window=window, state_id=seed.state_id)


def assert_seed_restraint_matches(
    seed: SeedState, window: int,
    primary_center: Optional[float], primary_k: Optional[float],
    secondary_center: Optional[float], secondary_k: Optional[float],
    rel_tol: float = 1e-6, abs_tol: float = 1e-9,
) -> None:
    """The seed's own recorded restraint must match this window's current restraint
    (spec revision 2, ruling 15) -- a safety net against index misalignment (e.g. a
    scrambled ``epoch_window_map.csv``) that a CV-value check alone cannot catch,
    since a window's CV target and a *different* window's CV target can be close by
    construction (adjacent umbrella windows).

    The primary restraint is always required on both sides; a missing field there
    fails closed. The secondary restraint is allowed to be absent (``None``/NaN) on
    both sides at once (a CV1-only run has no secondary restraint at all) -- but a
    mismatch where exactly one side has it and the other does not, or where both have
    it but the values differ, still raises.
    """
    def _fail(reason: str) -> None:
        raise SeedMismatchError(
            f"seed for window {window} (state {seed.state_id}, from {seed.source}) {reason}",
            window=window, state_id=seed.state_id,
        )

    if seed.primary_center is None or seed.primary_k is None:
        _fail("has no recorded primary restraint (primary_center/primary_k missing)")
    if primary_center is None or primary_k is None:
        _fail("this window's own primary restraint is undefined -- cannot verify the seed")
    if not math.isclose(seed.primary_center, primary_center, rel_tol=rel_tol, abs_tol=abs_tol):
        _fail(f"primary_center mismatch (seed {seed.primary_center!r} vs window {primary_center!r})")
    if not math.isclose(seed.primary_k, primary_k, rel_tol=rel_tol, abs_tol=abs_tol):
        _fail(f"primary_k mismatch (seed {seed.primary_k!r} vs window {primary_k!r})")

    def _missing(x: Optional[float]) -> bool:
        return x is None or not math.isfinite(x)

    seed_sc, seed_sk = seed.secondary_center, seed.secondary_k
    if _missing(seed_sc) != _missing(secondary_center):
        _fail(f"secondary_center presence mismatch (seed {seed_sc!r} vs window {secondary_center!r})")
    if not _missing(seed_sc) and not math.isclose(seed_sc, secondary_center, rel_tol=rel_tol, abs_tol=abs_tol):
        _fail(f"secondary_center mismatch (seed {seed_sc!r} vs window {secondary_center!r})")
    if _missing(seed_sk) != _missing(secondary_k):
        _fail(f"secondary_k presence mismatch (seed {seed_sk!r} vs window {secondary_k!r})")
    if not _missing(seed_sk) and not math.isclose(seed_sk, secondary_k, rel_tol=rel_tol, abs_tol=abs_tol):
        _fail(f"secondary_k mismatch (seed {seed_sk!r} vs window {secondary_k!r})")
