"""X8 discovery census: how many new structural states each campaign phase finds.

Reference-free: no native or folded structure is used anywhere. Two state definitions:

1. **Backbone basin string.** Each core residue gets a Ramachandran basin letter from its
   (phi, psi), angles wrapped to [-180, 180):

   - ``A`` alpha-R: phi < 0 and -120 <= psi < 50
   - ``B`` beta:    phi < -90 and psi outside [-120, 50)
   - ``P`` PPII:    -90 <= phi < 0 and psi outside [-120, 50)
   - ``L`` alpha-L: phi >= 0 and -90 <= psi < 90
   - ``O`` other:   phi >= 0 and psi outside [-90, 90)

   The partition is complete (every angle pair has exactly one letter). Core residues are
   those with both phi and psi defined (the terminal residues are excluded). A frame's state
   is the string over the core residues, in sequence order (chignolin: 8 letters).

2. **C-alpha cluster.** Greedy leader clustering at ``cutoff_nm`` (default 0.2 nm = 2 A)
   C-alpha RMSD after optimal superposition, over all C-alpha atoms. Frames are fed in
   campaign order (phase, then time, then replica); a frame becomes a new leader iff it is
   farther than the cutoff from every earlier leader, so a cluster's first-seen phase is its
   leader's phase. Deterministic. Clustering uses a subsample (``cluster_stride``).

Per phase: states seen for the first time (``new_raw``), and a transient-robust count
(``new_populated``: the phase in which a state's cumulative frame count first reaches
``min_frames``). Rates are new states per 100 ns of aggregate replica time, where time is
all frames present in the phase (before any stride) times the phase's frame interval.
Frames from boosted rungs are included: trajectories are written per replica, not per state.

Saturation (``saturation``): on the populated count of the last epoch group, the rate in
that group vs the previous one, the first one and the campaign mean. Verdict ``saturated``
if the last group found no new populated state or its rate is below ``SATURATION_RATIO``
(0.1) of the campaign mean, else ``discovering``. The threshold is a heuristic.

Diagnostics only: nothing here changes an adaptive decision.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = "discovery_census_v1"
BASIN_LETTERS = "ABPLO"
BASIN_NAMES = {"A": "alpha-R", "B": "beta", "P": "PPII", "L": "alpha-L", "O": "other"}
BASIN_RULES = {
    "A": "phi < 0 and -120 <= psi < 50",
    "B": "phi < -90 and not (-120 <= psi < 50)",
    "P": "-90 <= phi < 0 and not (-120 <= psi < 50)",
    "L": "phi >= 0 and -90 <= psi < 90",
    "O": "phi >= 0 and not (-90 <= psi < 90)",
}
DEFAULT_CUTOFF_NM = 0.2
DEFAULT_MIN_FRAMES = 5
DEFAULT_MAX_CLUSTER_FRAMES = 100_000
SATURATION_RATIO = 0.1

_REPLICA_RE = re.compile(r"replica_(\d+)(?:_resume_from_(\d+))?\.(xtc|dcd)$")
_TOPUP_RE = re.compile(r"topup_(\d+)")


# --------------------------------------------------------------------------- basins

def _wrap_deg(a) -> np.ndarray:
    return (np.asarray(a, dtype=np.float64) + 180.0) % 360.0 - 180.0


def basin_codes(phi_deg, psi_deg) -> np.ndarray:
    """Basin index into ``BASIN_LETTERS`` for each (phi, psi) in degrees (any shape)."""
    phi, psi = _wrap_deg(phi_deg), _wrap_deg(psi_deg)
    alpha_r_psi = (psi >= -120.0) & (psi < 50.0)
    out = np.full(phi.shape, 4, dtype=np.uint8)                       # O
    neg = phi < 0.0
    out[neg & alpha_r_psi] = 0                                        # A
    out[neg & ~alpha_r_psi & (phi < -90.0)] = 1                       # B
    out[neg & ~alpha_r_psi & (phi >= -90.0)] = 2                      # P
    out[~neg & (psi >= -90.0) & (psi < 90.0)] = 3                     # L
    return out


def encode_basin_strings(codes: np.ndarray) -> np.ndarray:
    """(n_frames, n_core) basin codes -> one int64 id per frame (base-5, first residue most significant)."""
    codes = np.asarray(codes, dtype=np.int64)
    if codes.ndim != 2:
        raise ValueError("codes must be (n_frames, n_core)")
    weights = len(BASIN_LETTERS) ** np.arange(codes.shape[1] - 1, -1, -1, dtype=np.int64)
    return codes @ weights


def decode_basin_string(state_id: int, n_core: int) -> str:
    base = len(BASIN_LETTERS)
    letters = []
    for _ in range(n_core):
        state_id, r = divmod(int(state_id), base)
        letters.append(BASIN_LETTERS[r])
    return "".join(reversed(letters))


# --------------------------------------------------------------------------- clustering

def _rmsd_to_one(xyz: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """RMSD (nm) of every frame of centred ``xyz`` (n, a, 3) to centred ``ref`` (a, 3)."""
    try:
        import mdtraj as md
    except ImportError:                                               # numpy Kabsch fallback
        h = np.einsum("nai,aj->nij", xyz, ref)
        u, s, vt = np.linalg.svd(h)
        s[:, -1] *= np.sign(np.linalg.det(u @ vt))
        g = np.einsum("nai,nai->n", xyz, xyz) + float(np.sum(ref * ref))
        return np.sqrt(np.clip((g - 2.0 * s.sum(axis=1)) / xyz.shape[1], 0.0, None))
    top = _point_topology(xyz.shape[1])
    target = md.Trajectory(xyz.astype(np.float32), top)
    reference = md.Trajectory(ref[None].astype(np.float32), top)
    return md.rmsd(target, reference, frame=0).astype(np.float64)


_POINT_TOPS: Dict[int, object] = {}


def _point_topology(n_atoms: int):
    import mdtraj as md
    if n_atoms not in _POINT_TOPS:
        top = md.Topology()
        res = top.add_residue("X", top.add_chain())
        for _ in range(n_atoms):
            top.add_atom("CA", md.element.carbon, res)
        _POINT_TOPS[n_atoms] = top
    return _POINT_TOPS[n_atoms]


def leader_cluster(xyz_nm: np.ndarray, cutoff_nm: float = DEFAULT_CUTOFF_NM) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy leader clustering in the given frame order.

    Returns ``(leader_frame_indices, assignment)``; ``assignment[i]`` is the index (into the
    leader list) of the first leader within ``cutoff_nm`` (<=) of frame i. Implemented as a
    sweep (take the first unassigned frame as leader, absorb every unassigned frame within
    the cutoff): identical leaders and assignments to the frame-by-frame algorithm, since a
    frame is only ever absorbed by a leader that precedes it.
    """
    xyz = np.asarray(xyz_nm, dtype=np.float64)
    n = len(xyz)
    assign = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return np.zeros(0, dtype=np.int64), assign
    xyz = xyz - xyz.mean(axis=1, keepdims=True)
    remaining = np.arange(n)
    leaders: List[int] = []
    while remaining.size:
        k = int(remaining[0])
        d = _rmsd_to_one(xyz[remaining], xyz[k])
        d[0] = 0.0
        inside = d <= cutoff_nm
        assign[remaining[inside]] = len(leaders)
        leaders.append(k)
        remaining = remaining[~inside]
    return np.asarray(leaders, dtype=np.int64), assign


# --------------------------------------------------------------------------- census

def first_seen_census(labels: Sequence[str], state_ids: Sequence[np.ndarray], ns: Sequence[float],
                      min_frames: int = 1) -> List[dict]:
    """Per-phase first-seen counts, in the given (chronological) phase order."""
    min_frames = max(1, int(min_frames))
    cum: Dict[int, int] = {}
    seen_raw: set = set()
    populated: set = set()
    rows = []
    for label, ids, t in zip(labels, state_ids, ns):
        u, c = np.unique(np.asarray(ids, dtype=np.int64), return_counts=True)
        new_raw = new_pop = 0
        for s, k in zip(u.tolist(), c.tolist()):
            if s not in seen_raw:
                seen_raw.add(s)
                new_raw += 1
            total = cum.get(s, 0) + k
            cum[s] = total
            if total >= min_frames and s not in populated:
                populated.add(s)
                new_pop += 1
        t = float(t)
        rows.append({
            "phase": label, "frames": int(len(ids)), "ns": t, "distinct_in_phase": int(len(u)),
            "new_raw": new_raw, "cumulative_raw": len(seen_raw),
            "new_populated": new_pop, "cumulative_populated": len(populated),
            "new_raw_per_100ns": (100.0 * new_raw / t) if t > 0 else None,
            "new_populated_per_100ns": (100.0 * new_pop / t) if t > 0 else None,
        })
    return rows


def saturation(rows: Sequence[dict], key: str = "populated") -> dict:
    """Last-row discovery rate vs the previous row, the first row and the campaign mean."""
    new_key = f"new_{key}"
    out = {"count": key, "rule": (f"saturated if the last row's new_{key} is 0 or its rate per 100 ns "
                                  f"is < {SATURATION_RATIO} x the campaign-mean rate; else discovering")}
    if len(rows) < 2:
        return {**out, "verdict": "insufficient_phases"}

    def rate(r):
        return 100.0 * r[new_key] / r["ns"] if r["ns"] > 0 else None

    def ratio(a, b):
        return (a / b) if (a is not None and b not in (None, 0)) else None

    total_ns = sum(r["ns"] for r in rows)
    mean = 100.0 * sum(r[new_key] for r in rows) / total_ns if total_ns > 0 else None
    last, prev, first = rate(rows[-1]), rate(rows[-2]), rate(rows[0])
    cum_last = rows[-1].get(f"cumulative_{key}") or 0
    vs_mean = ratio(last, mean)
    saturated = rows[-1][new_key] == 0 or (vs_mean is not None and vs_mean < SATURATION_RATIO)
    return {**out,
            "last_row": rows[-1]["phase"], "last_new": rows[-1][new_key],
            "last_rate_per_100ns": last, "previous_rate_per_100ns": prev,
            "first_rate_per_100ns": first, "campaign_mean_rate_per_100ns": mean,
            "last_vs_previous": ratio(last, prev), "last_vs_first": ratio(last, first),
            "last_vs_campaign_mean": vs_mean,
            "last_fraction_of_cumulative": (rows[-1][new_key] / cum_last) if cum_last else None,
            "verdict": "saturated" if saturated else "discovering"}


# --------------------------------------------------------------------------- phases

def _group_of(label: str) -> str:
    return label.split("/", 1)[0]


def _group_sort_key(name: str):
    m = re.fullmatch(r"epoch_(\d+)", name)
    if m:
        return (0, int(m.group(1)))
    if name == "final":
        return (1, 0)
    m = re.fullmatch(r"final_extension_(\d+)", name)
    if m:
        return (2, int(m.group(1)))
    return (3, name)


def ordered_phases(adaptive_dir) -> Tuple[List[Tuple[str, Path]], List[dict]]:
    """``([(label, phase_dir)], skipped)`` in campaign order.

    Phases are the ones ``ladder_adapt.phase_dirs`` finds (Parquet samples plus a window map).
    Epoch groups: epoch_NNN by number, then final, then final_extension_NNN. Within a group:
    the flat phase or ``baseline`` first, then ``topup_*`` by ordinal with directory mtime as
    the tie-break (a topup name's step suffix is not chronological). Epoch/final directories
    that hold no phase are returned in ``skipped``.
    """
    from gareus.adaptive.ladder_adapt import phase_dirs
    ad = Path(adaptive_dir)
    found = [(str(Path(p).relative_to(ad)), Path(p)) for p, _ in phase_dirs(ad)]

    def within(item):
        label, path = item
        sub = label.split("/", 1)[1] if "/" in label else ""
        if sub in ("", "baseline"):
            return (0, 0, 0.0, sub)
        m = _TOPUP_RE.match(sub)
        return (1, int(m.group(1)) if m else 10**9, path.stat().st_mtime, sub)

    found.sort(key=lambda it: (_group_sort_key(_group_of(it[0])), within(it)))
    groups = {_group_of(l) for l, _ in found}
    skipped = [{"phase": d.name, "reason": "no phase with samples and a window map"}
               for d in sorted(ad.iterdir(), key=lambda p: _group_sort_key(p.name))
               if d.is_dir() and re.fullmatch(r"epoch_\d+|final|final_extension_\d+", d.name)
               and d.name not in groups]
    return found, skipped


def frame_interval_ps(phase_dir: Path) -> Optional[float]:
    """``traj_interval * timestep_fs / 1000`` from the phase's run_manifest.json, else None."""
    try:
        a = json.loads((Path(phase_dir) / "run_manifest.json").read_text()).get("resolved_args", {})
        v = float(a["traj_interval"]) * float(a["timestep_fs"]) / 1000.0
        return v if v > 0 else None
    except Exception:
        return None


def _find_topology(phase_dir: Path, adaptive_dir: Path) -> Optional[Path]:
    for cand in (phase_dir / "solute_only.pdb", phase_dir.parent / "solute_only.pdb"):
        if cand.exists():
            return cand
    hits = sorted(adaptive_dir.glob("epoch_*/solute_only.pdb")) + sorted(adaptive_dir.glob("*/*/solute_only.pdb"))
    return hits[0] if hits else None


def backbone_selection(top_path: Path) -> dict:
    """Core-residue torsion quadruplets (paired by the shared CA) and C-alpha indices."""
    import mdtraj as md
    ref = md.load(str(top_path))
    phi_idx, _ = md.compute_phi(ref)
    psi_idx, _ = md.compute_psi(ref)
    phi_by_ca = {int(q[2]): q for q in phi_idx}                       # C(i-1) N CA C
    psi_by_ca = {int(q[1]): q for q in psi_idx}                       # N CA C N(i+1)
    core_ca = sorted(set(phi_by_ca) & set(psi_by_ca))
    if not core_ca:
        raise ValueError(f"no residue with both phi and psi in {top_path}")
    ca = [a.index for a in ref.topology.atoms if a.name == "CA"]
    phi_q = np.array([phi_by_ca[c] for c in core_ca])
    psi_q = np.array([psi_by_ca[c] for c in core_ca])
    subset = np.array(sorted(set(phi_q.ravel()) | set(psi_q.ravel()) | set(ca)))
    remap = {int(a): i for i, a in enumerate(subset)}
    return {
        "subset": subset,
        "phi": np.vectorize(remap.get)(phi_q), "psi": np.vectorize(remap.get)(psi_q),
        "ca": np.array([remap[a] for a in ca]),
        "core_residues": [f"{ref.topology.atom(c).residue.name}{ref.topology.atom(c).residue.resSeq}"
                          for c in core_ca],
        "n_atoms": ref.n_atoms,
    }


def _read_one(task) -> dict:
    """Read one trajectory file: all frame times, basin ids at basin_stride, CA at cluster_stride."""
    path, top_path, sel, basin_stride, cluster_stride = task
    out = {"path": str(path)}
    if Path(path).stat().st_size == 0:
        return {**out, "error": "empty"}
    try:
        import mdtraj as md
        tr = md.load(str(path), top=str(top_path), atom_indices=sel["subset"])
    except Exception as exc:                                         # malformed / truncated
        return {**out, "error": f"unreadable: {type(exc).__name__}: {exc}"}
    if tr.n_frames == 0:
        return {**out, "error": "no frames"}
    b = np.arange(0, tr.n_frames, basin_stride)
    c = np.arange(0, tr.n_frames, cluster_stride)
    phi = np.degrees(md.compute_dihedrals(tr[b], sel["phi"]))
    psi = np.degrees(md.compute_dihedrals(tr[b], sel["psi"]))
    return {**out, "time": np.asarray(tr.time, dtype=np.float64),
            "basin_idx": b, "basin_ids": encode_basin_strings(basin_codes(phi, psi)),
            "ca_idx": c, "ca_xyz": tr.xyz[c][:, sel["ca"]].astype(np.float32)}


def _trajectory_files(phase_dir: Path) -> List[Tuple[int, int, Path]]:
    files = []
    for p in sorted((phase_dir / "replica_trajectories").glob("replica_*")):
        m = _REPLICA_RE.search(p.name)
        if m:
            files.append((int(m.group(1)), int(m.group(2) or 0), p))
    return sorted(files)


def load_phase(phase_dir: Path, top_path: Path, sel: dict, basin_stride: int, cluster_stride: int,
               workers: int = 1, pool=None) -> dict:
    """Frames of one phase, resume overlaps removed (a file's frames at or after the next
    segment's first time are superseded by that segment)."""
    files = _trajectory_files(phase_dir)
    tasks = [(p, top_path, sel, basin_stride, cluster_stride) for _, _, p in files]
    results = list(pool.map(_read_one, tasks)) if pool is not None else [_read_one(t) for t in tasks]
    good = [(rep, res) for (rep, _, _), res in zip(files, results) if "error" not in res]
    skipped = [{"path": r["path"], "reason": r["error"]} for r in results if "error" in r]
    n_frames, basin, ca, ca_t, ca_rep, dts = 0, [], [], [], [], []
    for i, (rep, res) in enumerate(good):
        t = res["time"]
        nxt = next((r for rp, r in good[i + 1:] if rp == rep), None)
        keep = t < nxt["time"][0] if nxt is not None else np.ones(len(t), dtype=bool)
        n_frames += int(keep.sum())
        if len(t) > 1:
            dts.append(float(np.median(np.diff(t))))
        basin.append(res["basin_ids"][keep[res["basin_idx"]]])
        kc = keep[res["ca_idx"]]
        ca.append(res["ca_xyz"][kc])
        ca_t.append(t[res["ca_idx"]][kc])
        ca_rep.append(np.full(int(kc.sum()), rep))
    interval = frame_interval_ps(phase_dir)
    source = "run_manifest"
    if interval is None:
        interval, source = (float(np.median(dts)) if dts else 0.0), "xtc_time"
    n_ca = len(sel["ca"])
    ca_xyz = np.concatenate(ca) if ca else np.zeros((0, n_ca, 3), np.float32)
    ca_time = np.concatenate(ca_t) if ca_t else np.zeros(0)
    ca_replica = np.concatenate(ca_rep) if ca_rep else np.zeros(0, int)
    order = np.lexsort((ca_replica, ca_time))                          # time, then replica
    return {"n_frames": n_frames, "ns": n_frames * interval / 1000.0,
            "frame_interval_ps": interval, "frame_interval_source": source,
            "basin_ids": np.concatenate(basin) if basin else np.zeros(0, np.int64),
            "ca_xyz": ca_xyz[order], "files_read": len(good), "files_skipped": skipped}


def estimate_total_frames(phases: Sequence[Tuple[str, Path]], top_path: Path) -> int:
    """Total frames estimated from file sizes, calibrated on the largest readable file."""
    import mdtraj as md
    files = [p for _, d in phases for _, _, p in _trajectory_files(d) if p.stat().st_size > 0]
    if not files:
        return 0
    total_bytes = sum(p.stat().st_size for p in files)
    for cal in sorted(files, key=lambda p: (-p.stat().st_size, str(p)))[:3]:
        try:
            with md.formats.XTCTrajectoryFile(str(cal)) as fh:
                n = len(fh)
            if n > 0:
                return int(round(total_bytes * n / cal.stat().st_size))
        except Exception:
            continue
    return 0


def run_census(adaptive_dir, *, basin_stride: int = 1, cluster_stride: Optional[int] = None,
               max_cluster_frames: int = DEFAULT_MAX_CLUSTER_FRAMES,
               cutoff_nm: float = DEFAULT_CUTOFF_NM, min_frames: int = DEFAULT_MIN_FRAMES,
               workers: int = 1, phases: Optional[Sequence[str]] = None) -> dict:
    """Discovery census over an adaptive_production directory (read-only)."""
    t0 = time.time()
    ad = Path(adaptive_dir)
    found, skipped_phases = ordered_phases(ad)
    if phases is not None:
        found = [(l, p) for l, p in found if any(l == n or l.startswith(n.rstrip("/") + "/") for n in phases)]
    if not found:
        raise ValueError(f"no campaign phases with samples under {ad}")
    top_path = _find_topology(found[0][1], ad)
    if top_path is None:
        raise FileNotFoundError(f"no solute_only.pdb under {ad}")
    sel = backbone_selection(top_path)
    basin_stride = max(1, int(basin_stride))
    est = None
    if cluster_stride is None:
        est = estimate_total_frames(found, top_path)
        cluster_stride = max(1, -(-est // max(1, int(max_cluster_frames))))
    cluster_stride = max(1, int(cluster_stride))

    per_phase = []
    pool = ProcessPoolExecutor(max_workers=int(workers)) if int(workers) > 1 else None
    try:
        for label, path in found:
            tp = _find_topology(path, ad) or top_path
            data = load_phase(path, tp, sel, basin_stride, cluster_stride, pool=pool)
            per_phase.append((label, path, data))
    finally:
        if pool is not None:
            pool.shutdown()

    labels = [l for l, _, _ in per_phase]
    ns = [d["ns"] for _, _, d in per_phase]
    groups = []
    for l in labels:
        if _group_of(l) not in groups:
            groups.append(_group_of(l))

    def by_group(arrays):
        return [np.concatenate([a for l, a in zip(labels, arrays) if _group_of(l) == g]) for g in groups]
    g_ns = [sum(t for l, t in zip(labels, ns) if _group_of(l) == g) for g in groups]

    # basin strings
    basin_ids = [d["basin_ids"] for _, _, d in per_phase]
    b_rows = first_seen_census(labels, basin_ids, ns, min_frames)
    b_groups = first_seen_census(groups, by_group(basin_ids), g_ns, min_frames)
    n_core = len(sel["core_residues"])
    seen, first = set(), {}
    for l, ids in zip(labels, basin_ids):
        new = sorted(set(np.unique(ids).tolist()) - seen)
        seen.update(new)
        first[l] = [decode_basin_string(s, n_core) for s in new]

    # C-alpha clusters: one global leader clustering in campaign order
    lens = [len(d["ca_xyz"]) for _, _, d in per_phase]
    all_ca = np.concatenate([d["ca_xyz"] for _, _, d in per_phase]) if sum(lens) else np.zeros((0, len(sel["ca"]), 3))
    t_cl = time.time()
    leaders, assign = leader_cluster(all_ca, cutoff_nm)
    t_cl = time.time() - t_cl
    splits = np.split(assign, np.cumsum(lens)[:-1])
    c_rows = first_seen_census(labels, splits, ns, min_frames)
    c_groups = first_seen_census(groups, by_group(splits), g_ns, min_frames)

    files_skipped = [s for _, _, d in per_phase for s in d["files_skipped"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "adaptive_dir": str(ad),
        "reference_free": True,
        "definitions": {
            "basin_strings": {"letters": BASIN_NAMES, "rules_deg": BASIN_RULES,
                              "angle_wrap": "[-180, 180)", "core_residues": sel["core_residues"],
                              "core_rule": "residues with both phi and psi defined",
                              "frame_stride": basin_stride},
            "ca_clusters": {"method": "greedy leader, campaign order (phase, time, replica)",
                            "cutoff_nm": float(cutoff_nm), "atoms": "all C-alpha", "n_atoms": int(len(sel["ca"])),
                            "superposition": "optimal (Kabsch/QCP)", "frame_stride": cluster_stride,
                            "estimated_total_frames": est, "max_cluster_frames": int(max_cluster_frames),
                            "frames_clustered": int(len(all_ca)), "leaders": int(len(leaders)),
                            "clustering_seconds": round(t_cl, 2)},
            "min_frames": int(min_frames),
            "time_basis": "aggregate replica time: all frames in the phase (before stride), resume "
                          "overlaps removed, x frame interval; boosted-rung frames included",
            "saturation_ratio": SATURATION_RATIO,
        },
        "phase_order": labels,
        "phases_meta": [{"phase": l, "n_frames": d["n_frames"], "ns": d["ns"],
                         "frame_interval_ps": d["frame_interval_ps"],
                         "frame_interval_source": d["frame_interval_source"], "files_read": d["files_read"],
                         "files_skipped": len(d["files_skipped"])} for l, _, d in per_phase],
        "skipped_phases": skipped_phases,
        "files": {"skipped": files_skipped},
        "basin_strings": {"phases": b_rows, "epochs": b_groups, "states_first_seen": first,
                          "saturation": saturation(b_groups, "populated"),
                          "saturation_raw": saturation(b_groups, "raw")},
        "ca_clusters": {"phases": c_rows, "epochs": c_groups,
                        "saturation": saturation(c_groups, "populated"),
                        "saturation_raw": saturation(c_groups, "raw")},
        "topology": str(top_path),
        "runtime_seconds": round(time.time() - t0, 2),
    }


# --------------------------------------------------------------------------- output

CSV_FIELDS = ["definition", "level", "phase", "frames", "ns", "distinct_in_phase", "new_raw",
              "cumulative_raw", "new_populated", "cumulative_populated", "new_raw_per_100ns",
              "new_populated_per_100ns"]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_outputs(result: dict, out_dir, *, plot: bool = True, stem: str = "discovery_census") -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": str(out / f"{stem}.json"), "csv": str(out / f"{stem}.csv")}
    tmp = out / f".{stem}.json.tmp"
    tmp.write_text(json.dumps(result, indent=2, default=_json_default))
    tmp.replace(paths["json"])
    with open(paths["csv"], "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for definition in ("basin_strings", "ca_clusters"):
            for level in ("phases", "epochs"):
                for r in result[definition][level]:
                    w.writerow({"definition": definition, "level": level[:-1], **r})
    if plot:
        png = _plot(result, out / f"{stem}.png")
        if png:
            paths["png"] = png
    return paths


def _plot(result: dict, path: Path) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    try:
        from gareus.mbar_analysis.plotstyle import OKABE_ITO_LINES, style_line_axes
        colors = [c if isinstance(c, str) else c[0] for c in OKABE_ITO_LINES]
    except Exception:
        style_line_axes, colors = None, ["#0072B2", "#D55E00"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    titles = {"basin_strings": "Core backbone basin strings",
              "ca_clusters": f"C-alpha clusters ({result['definitions']['ca_clusters']['cutoff_nm'] * 10:g} A)"}
    m = result["definitions"]["min_frames"]
    for ax, key in zip(axes, ("basin_strings", "ca_clusters")):
        rows = result[key]["phases"]
        x = np.cumsum([r["ns"] for r in rows])
        ax.plot(x, [r["cumulative_raw"] for r in rows], marker="o", color=colors[0], label="first seen")
        ax.plot(x, [r["cumulative_populated"] for r in rows], marker="s", color=colors[1 % len(colors)],
                label=f"populated (>= {m} frames)")
        for xi, r in zip(x, rows):
            ax.axvline(xi, color="#BBBBBB", lw=0.5, zorder=0)
        ax.set_title(f"{titles[key]}: {result[key]['saturation'].get('verdict')}")
        if style_line_axes is not None:
            style_line_axes(ax, xlabel="cumulative aggregate replica time (ns)", ylabel="cumulative states")
        else:
            ax.set_xlabel("cumulative aggregate replica time (ns)")
            ax.set_ylabel("cumulative states")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def census_for_epoch(adaptive_dir, epoch_dir, **kwargs) -> Optional[str]:
    """Driver hook: census over the phases so far, written to ``epoch_dir/discovery_census.json``.

    Never raises: diagnostics must not end an epoch.
    """
    try:
        result = run_census(adaptive_dir, **kwargs)
        return write_outputs(result, epoch_dir, plot=False)["json"]
    except Exception as exc:
        print(f"WARNING: discovery census failed for {epoch_dir}: {type(exc).__name__}: {exc}")
        return None


def _summary_lines(result: dict) -> List[str]:
    lines = []
    for key in ("basin_strings", "ca_clusters"):
        lines.append(f"{key}:")
        for r in result[key]["epochs"]:
            lines.append(f"  {r['phase']:<22} {r['ns']:10.1f} ns  new {r['new_raw']:6d} (pop {r['new_populated']:6d})"
                         f"  cum {r['cumulative_raw']:6d} (pop {r['cumulative_populated']:6d})")
        s = result[key]["saturation"]
        lines.append(f"  verdict: {s.get('verdict')}  last/mean {s.get('last_vs_campaign_mean')}"
                     f"  last/prev {s.get('last_vs_previous')}")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m gareus.adaptive.discovery_census",
                                description="Reference-free discovery census per campaign phase (read-only).")
    p.add_argument("adaptive_dir", type=Path)
    p.add_argument("--out", type=Path, default=None, help="output directory (default: adaptive_dir)")
    p.add_argument("--basin-stride", type=int, default=1, help="frame stride for basin strings")
    p.add_argument("--cluster-stride", type=int, default=None,
                   help="frame stride for C-alpha clustering (default: from --max-cluster-frames)")
    p.add_argument("--max-cluster-frames", type=int, default=DEFAULT_MAX_CLUSTER_FRAMES)
    p.add_argument("--cluster-cutoff-nm", type=float, default=DEFAULT_CUTOFF_NM)
    p.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES,
                   help="frames (at the stride) for a state to count as populated")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--phases", nargs="*", default=None)
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args(argv)
    result = run_census(a.adaptive_dir, basin_stride=a.basin_stride, cluster_stride=a.cluster_stride,
                        max_cluster_frames=a.max_cluster_frames, cutoff_nm=a.cluster_cutoff_nm,
                        min_frames=a.min_frames, workers=a.workers, phases=a.phases)
    paths = write_outputs(result, a.out or a.adaptive_dir, plot=not a.no_plot)
    print("\n".join(_summary_lines(result)))
    print(f"runtime {result['runtime_seconds']} s; wrote {', '.join(paths.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
