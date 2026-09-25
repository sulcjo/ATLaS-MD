"""Continue each top-up window's chain from its parent segment's final State (spec 4.4, revision 2)."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Sequence

DIR_NAME = "final_window_states"


class SeedMismatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class SeedState:
    positions: Any
    velocities: Any
    box: Any
    cv1: float
    cv2: float
    source: str


def _deserialize_state(text: str):
    from openmm import XmlSerializer  # noqa: PLC0415
    st = XmlSerializer.deserialize(text)
    return st.getPositions(), st.getVelocities(), st.getPeriodicBoxVectors()


def export_final_window_states(out_dir, sims: Sequence, assignments: Sequence[int],
                               state_id_of_window: Dict[int, int],
                               cv_of_replica: Callable[[int], tuple]) -> Path:
    from openmm import XmlSerializer  # noqa: PLC0415
    d = Path(out_dir) / DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    index = {}
    for r, sim in enumerate(sims):
        w = int(assignments[r])
        sid = int(state_id_of_window.get(w, w))
        st = sim.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=False)
        (d / f"state_{sid}.xml").write_text(XmlSerializer.serialize(st))
        cv1, cv2 = cv_of_replica(r)
        index[str(sid)] = {"window": w, "cv1": float(cv1), "cv2": float(cv2)}
    tmp = d / "index.json.tmp"
    tmp.write_text(json.dumps(index, indent=2))
    tmp.replace(d / "index.json")
    return d


def load_seed_index(parent_dirs: Iterable) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for parent in parent_dirs:                       # later parents override earlier ones
        idx = Path(parent) / DIR_NAME / "index.json"
        if not idx.exists():
            continue
        for key in json.loads(idx.read_text()):
            if (Path(parent) / DIR_NAME / f"state_{int(key)}.xml").exists():
                out[int(key)] = str(parent)
    return out


def load_seed_states(parent_dirs: Iterable, state_ids: Iterable[int]) -> Dict[int, SeedState]:
    parents = list(parent_dirs)
    where = load_seed_index(parents)
    out: Dict[int, SeedState] = {}
    for sid in state_ids:
        parent = where.get(int(sid))
        if parent is None:
            continue
        d = Path(parent) / DIR_NAME
        rec = json.loads((d / "index.json").read_text())[str(int(sid))]
        pos, vel, box = _deserialize_state((d / f"state_{int(sid)}.xml").read_text())
        out[int(sid)] = SeedState(pos, vel, box, float(rec["cv1"]), float(rec["cv2"]), str(parent))
    return out


def assert_seed_matches(seed: SeedState, cv1_now: float, cv2_now: float, tol: float = 1e-3) -> None:
    """CV1 must always be finite and match; CV2 is compared only when both sides are finite.

    CV2 is legitimately NaN for a CV1-only run (no secondary CV configured) on both the
    recorded seed and the freshly-evaluated context, so that comparison is skipped rather
    than raised. CV1 is never expected to be NaN -- a NaN there means the loaded/seeded
    state is physically broken (e.g. blown-up positions), which must raise loudly rather
    than be silently accepted by NaN comparison semantics.
    """
    if not math.isfinite(cv1_now) or not math.isfinite(seed.cv1):
        raise SeedMismatchError(
            f"seeded state from {seed.source} does not reproduce its recorded CVs "
            f"(cv1 {cv1_now!r} vs {seed.cv1!r} -- non-finite CV1)")
    cv1_mismatch = abs(cv1_now - seed.cv1) > tol
    cv2_comparable = math.isfinite(cv2_now) and math.isfinite(seed.cv2)
    cv2_mismatch = cv2_comparable and abs(cv2_now - seed.cv2) > tol
    if cv1_mismatch or cv2_mismatch:
        raise SeedMismatchError(
            f"seeded state from {seed.source} does not reproduce its recorded CVs "
            f"(cv1 {cv1_now:.6f} vs {seed.cv1:.6f}, cv2 {cv2_now:.6f} vs {seed.cv2:.6f})")
