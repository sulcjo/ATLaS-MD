"""X3 slow-mode reseeding: phase-directory I/O, the epoch-boundary runner and a read-only replay.

Frame table. The hidden mode is fitted on the epoch's own production frames: backbone torsion
features in the frozen feature schema's order, the recorded CV1/CV2, the trajectory a frame
belongs to (lagged pairs are formed within one trajectory file only) and its state.

* Primary source: ``replica_trajectories/*.xtc`` (solute-only frames) joined on
  (replica, step) with the phase's Parquet samples for CV1, CV2 and the local window, mapped
  to state_id through the phase's own ``epoch_window_map.csv``. On chignolin_9 the
  recomputed CV1 matches the recorded one to 1.5e-4 and CV2 to 6e-3 (XTC precision), which
  pins the sign convention, the atom indexing and the step alignment together.
* Alternative: ``tica_obs/dihedral_obs_*.npz`` (exact positions; needs ``--tica-obs-interval``).

End states. Every ``final_pdbs/*.pdb`` of the epoch's phases (one per window per segment
end, including interrupted ones); CVs are recomputed from each file's positions, never taken
from the window it was written under.

Nothing here raises into the epoch loop: :func:`run_epoch_slow_mode_reseed` records any
failure as status "error", removes any override so seeding proceeds exactly as without X3,
and returns.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from . import slow_mode_reseed as X3

_XTC_RE = re.compile(r"^replica_(\d+)(?:_resume_from_(\d+))?\.xtc$")
_PDB_RE = re.compile(r"^replica_(\d+)_window_(\d+)\.pdb$")
_ONE_LETTER = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
               "GLY": "G", "HIS": "H", "HIE": "H", "HID": "H", "HIP": "H", "ILE": "I", "LEU": "L",
               "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
               "TYR": "Y", "VAL": "V"}
#: Default frame stride when reading trajectories (every 4th frame: 3.5 ps on chignolin_9).
DEFAULT_STRIDE = 4


# ---------------------------------------------------------------------------------------
# Geometry (vectorised twins of gareus.tica._dihedral_rad / backbone_dihedral_features and
# gareus.cv.nonlocal_contact_cv_from_positions_nm; tests hold them equal)
# ---------------------------------------------------------------------------------------

def torsion_features(xyz_nm, quads) -> np.ndarray:
    """(n_frames, 2 * n_torsions) sin/cos features, torsion order as given."""
    xyz = np.asarray(xyz_nm, dtype=np.float64)
    if xyz.ndim == 2:
        xyz = xyz[None]
    q = np.asarray(quads, dtype=np.int64)
    p0, p1, p2, p3 = (xyz[:, q[:, k], :] for k in range(4))
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    b2n = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-30)
    m1 = np.cross(n1, b2n)
    ang = np.arctan2(np.sum(m1 * n2, axis=-1), np.sum(n1 * n2, axis=-1))
    out = np.empty((xyz.shape[0], 2 * q.shape[0]))
    out[:, 0::2], out[:, 1::2] = np.sin(ang), np.cos(ang)
    return out


def contact_cv(xyz_nm, pairs, contact_args) -> np.ndarray:
    """Smooth nonlocal contact CV for each frame (same switch and normalisation as gareus.cv)."""
    from gareus.cv import contact_normalization_denominator

    xyz = np.asarray(xyz_nm, dtype=np.float64)
    if xyz.ndim == 2:
        xyz = xyz[None]
    P = np.asarray([(int(p[0]), int(p[1])) for p in pairs], dtype=np.int64)
    w = np.asarray([float(p[2]) if len(p) > 2 else 1.0 for p in pairs])
    r = np.linalg.norm(xyz[:, P[:, 0]] - xyz[:, P[:, 1]], axis=-1)
    x = float(contact_args.contact_beta_a_inv) * 10.0 * (r - float(contact_args.contact_r0_a) * 0.1)
    total = (0.5 * (1.0 - np.tanh(0.5 * x))) @ w
    if bool(getattr(contact_args, "contact_normalize", True)):
        total = total / float(contact_normalization_denominator(list(pairs), contact_args))
    return total


# ---------------------------------------------------------------------------------------
# Deployed pair and its context
# ---------------------------------------------------------------------------------------

@dataclass
class PairContext:
    runtime: Any                 # PairModelRuntime
    quads: list                  # torsion quadruplets in feature order
    torsion_labels: list         # e.g. "psi(D3)"
    solute_names: list           # (resname, atom name) of the solute atoms, in order
    contact_pairs: list
    contact_args: Any


def load_pair_context(pair_paths: Sequence, solute_pdb) -> PairContext:
    """Load the frozen pair and rebuild the contact list, held to the pair's pair-list digest."""
    from openmm import app  # noqa: PLC0415

    from gareus.cv import build_nonlocal_contact_pairs
    from gareus.cv_selection.models import PairModelRuntime, contact_pair_list_digest

    rt = PairModelRuntime.load(*[Path(p) for p in pair_paths], require_deployable=False, allow_legacy_v1=True)
    topology = app.PDBFile(str(solute_pdb)).topology
    atoms = list(topology.atoms())
    from gareus.cv_selection.anchor_spec import is_contact
    if is_contact(rt.anchor_kind):
        cargs = rt.contact_args()
        rule = str(rt.anchor_definition.get("pair_rule", "atom-pairs"))
        cargs.contact_scheme = rule.split(":")[0] or "atom-pairs"
        cargs.contact_pair_warning_threshold = 10 ** 9
        pairs = build_nonlocal_contact_pairs(topology, cargs)
        want = rt.anchor_definition.get("pair_list_sha256")
        if want and contact_pair_list_digest(pairs) != want:
            raise RuntimeError("contact pair list rebuilt from the solute topology does not match the pair model's")
    elif rt.anchor_kind == "contact-map-component":
        cargs = None                      # composite: anchor_spec.value_from_positions reads the model
        pairs = []
    else:                                 # distance anchor: the model's own atom pair
        cargs = None
        pairs = [(int(rt.anchor_definition["atom1"]), int(rt.anchor_definition["atom2"]), 1.0)]
    schema = json.loads(Path(pair_paths[2]).read_text())
    labels = []
    for f in schema["features"]:
        if f["trig"] != "sin":
            continue
        ca = next((atoms[i] for i in f["atom_indices"] if atoms[i].name == "CA"), atoms[f["atom_indices"][1]])
        res = ca.residue
        kind = str(f["torsion_name"]).split("-")[0]
        labels.append(f"{kind}({_ONE_LETTER.get(res.name, res.name)}{int(res.id)})")
    return PairContext(rt, [list(q) for q in rt.feature_atoms], labels,
                       [(a.residue.name, a.name) for a in atoms], pairs, cargs)


def pair_paths_from_args(args) -> Optional[tuple]:
    paths = tuple(getattr(args, k, None) for k in
                  ("secondary_cv_model", "secondary_cv_candidate_set", "secondary_cv_feature_schema"))
    if any(p is None or not str(p) or not Path(p).exists() for p in paths):
        return None
    return paths


# ---------------------------------------------------------------------------------------
# Frame table
# ---------------------------------------------------------------------------------------

@dataclass
class FrameTable:
    X: np.ndarray
    cv1: np.ndarray
    cv2: np.ndarray
    member: np.ndarray           # int code: one per trajectory file
    frame_index: np.ndarray      # MD step within the phase
    state_id: np.ndarray
    lag_steps: int
    lag_ps: float
    sources: list


def _samples(phase: Path):
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(phase / "samples" / "seg_*" / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"{phase}: no Parquet samples")
    cols = ["step", "replica", "window_id", "cv1", "cv2"]
    tabs = [pq.read_table(f, columns=cols) for f in files]
    return {c: np.concatenate([np.asarray(t.column(c).to_numpy(zero_copy_only=False), dtype=np.float64)
                               for t in tabs]) for c in cols}


def _state_of_window(phase: Path) -> Dict[int, int]:
    from gareus.topup_seeding import state_id_of_window_from_epoch_map

    m = state_id_of_window_from_epoch_map(phase)
    if not m:
        raise FileNotFoundError(f"{phase}: no epoch_window_map.csv (never mapped by identity)")
    return m


def _frames_from_xtc(phase: Path, ctx: PairContext, stride: int, lag_ps: float, member0: int):
    from mdtraj.formats import XTCTrajectoryFile

    files = sorted(p for p in (phase / "replica_trajectories").glob("*.xtc") if _XTC_RE.match(p.name))
    if not files:
        return None
    s = _samples(phase)
    span = int(s["step"].max()) + 1
    key = s["replica"].astype(np.int64) * span + s["step"].astype(np.int64)
    order = np.argsort(key, kind="stable")
    skey = key[order]
    smap = _state_of_window(phase)
    n_sol = len(ctx.solute_names)
    X, c1, c2, mem, fr, st = [], [], [], [], [], []
    spf = dt = None
    for k, f in enumerate(files):
        with XTCTrajectoryFile(str(f)) as fh:
            xyz, t, steps, _box = fh.read(stride=int(stride))
        if xyz.shape[0] < 2 or xyz.shape[1] < n_sol:
            continue
        if spf is None:
            spf, dt = int(steps[1] - steps[0]) // int(stride), float(t[1] - t[0]) / int(stride)
        replica = int(_XTC_RE.match(f.name).group(1))
        want = replica * span + steps.astype(np.int64)
        pos = np.searchsorted(skey, want)
        pos = np.minimum(pos, skey.size - 1)
        hit = skey[pos] == want
        if not hit.any():
            continue
        rows = order[pos[hit]]
        states = np.asarray([smap.get(int(w), -1) for w in s["window_id"][rows]], dtype=np.int64)
        ok = states >= 0
        X.append(torsion_features(xyz[hit][ok, :n_sol], ctx.quads))
        c1.append(s["cv1"][rows][ok]); c2.append(s["cv2"][rows][ok])
        mem.append(np.full(int(ok.sum()), member0 + k, dtype=np.int64))
        fr.append(steps[hit][ok].astype(np.int64)); st.append(states[ok])
    if not X:
        return None
    lag_frames = max(1, int(round(lag_ps / (dt * stride))))
    return (np.vstack(X), np.concatenate(c1), np.concatenate(c2), np.concatenate(mem),
            np.concatenate(fr), np.concatenate(st), lag_frames * stride * spf, lag_frames * stride * dt,
            {"phase": str(phase), "source": "replica_trajectories", "n_files": len(files), "stride": int(stride),
             "frame_dt_ps": dt * stride})


def _frames_from_tica_obs(phase: Path, ctx: PairContext, lag_ps: float, timestep_fs: float, member0: int):
    from gareus.tica import load_epoch_dihedral_obs

    if not list((phase / "tica_obs").glob("dihedral_obs_*.npz")):
        return None
    X, window, cv1, cv2, _seg, steps, replica = load_epoch_dihedral_obs(phase)
    smap = _state_of_window(phase)
    states = np.asarray([smap.get(int(w), -1) for w in window], dtype=np.int64)
    ok = (states >= 0) & np.isfinite(cv1) & (np.asarray(steps) >= 0)
    steps = np.asarray(steps, dtype=np.int64)
    d = np.diff(np.sort(np.unique(steps[ok])))
    interval = int(d[d > 0].min()) if (d > 0).any() else 1
    lag_intervals = max(1, int(round(lag_ps * 1000.0 / (timestep_fs * interval))))
    lag_steps = lag_intervals * interval
    return (np.asarray(X)[ok], np.asarray(cv1)[ok], np.asarray(cv2)[ok],
            member0 + np.asarray(replica, dtype=np.int64)[ok], steps[ok], states[ok],
            lag_steps, lag_steps * timestep_fs / 1000.0,
            {"phase": str(phase), "source": "tica_obs", "obs_interval_steps": interval})


def load_frame_table(phase_dirs: Sequence, ctx: PairContext, *, lag_ps: float = X3.DEFAULT_LAG_PS,
                     stride: int = DEFAULT_STRIDE, timestep_fs: Optional[float] = None) -> FrameTable:
    parts, lags = [], set()
    for i, phase in enumerate(Path(p) for p in phase_dirs):
        member0 = (i + 1) * 1_000_000
        got = None
        if timestep_fs and (phase / "tica_obs").is_dir():
            got = _frames_from_tica_obs(phase, ctx, lag_ps, float(timestep_fs), member0)
        if got is None:
            got = _frames_from_xtc(phase, ctx, stride, lag_ps, member0)
        if got is None:
            continue
        parts.append(got)
        lags.add(int(got[6]))
    if not parts:
        raise FileNotFoundError("no phase has trajectory frames (replica_trajectories/*.xtc or tica_obs)")
    if len(lags) != 1:
        raise ValueError(f"phases disagree on the lag in steps ({sorted(lags)}); refusing to pool")
    cat = [np.concatenate([p[k] for p in parts]) if k else np.vstack([p[0] for p in parts]) for k in range(6)]
    return FrameTable(*cat, lag_steps=parts[0][6], lag_ps=float(parts[0][7]), sources=[p[8] for p in parts])


# ---------------------------------------------------------------------------------------
# End-state candidates
# ---------------------------------------------------------------------------------------

def read_solute_positions(path, solute_names) -> Optional[np.ndarray]:
    """First len(solute_names) atoms of a PDB, in nm; None if their names do not match."""
    n = len(solute_names)
    xyz, k = np.empty((n, 3)), 0
    with open(path) as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if (line[17:20].strip(), line[12:16].strip()) != tuple(solute_names[k]):
                return None
            xyz[k] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            k += 1
            if k == n:
                return xyz * 0.1
    return None


def measure_structures(paths: Sequence, ctx: PairContext, mode: "X3.HiddenMode") -> Dict[str, Dict[str, Any]]:
    """path -> cv1, cv2, hidden_z, side, measured from the file's own positions."""
    from gareus.cv_selection.models import evaluate_component

    out: Dict[str, Dict[str, Any]] = {}
    for p in paths:
        pos = read_solute_positions(p, ctx.solute_names)
        if pos is None:
            continue
        feats = torsion_features(pos, ctx.quads)
        if ctx.contact_args is None:      # distance (A) / contact-map anchor, as anchor_spec.value_from_positions
            from gareus.cv_selection.anchor_spec import value_from_positions
            c1 = np.atleast_1d(value_from_positions(ctx.runtime.anchor_kind, np.asarray(pos)[0] if np.ndim(pos) == 3
                                                    else pos, ctx.contact_pairs, ctx.runtime))
        else:
            c1 = contact_cv(pos, ctx.contact_pairs, ctx.contact_args)
        c2 = float(evaluate_component(ctx.runtime.fit, ctx.runtime.j, feats, c1)[0])
        z = float(mode.scores(feats, c1)[0])
        out[str(p)] = {"cv1": float(c1[0]), "cv2": c2, "hidden_z": z, "side": int(mode.side([z])[0])}
    return out


def end_state_candidates(phase_dirs: Sequence, ctx: PairContext, mode: "X3.HiddenMode") -> List[Dict[str, Any]]:
    paths, meta = [], {}
    for phase in (Path(p) for p in phase_dirs):
        pdb_dir = phase / "final_pdbs"
        if not pdb_dir.is_dir():
            continue
        smap = _state_of_window(phase)
        for f in sorted(pdb_dir.iterdir()):
            m = _PDB_RE.match(f.name)
            if m is None or int(m.group(2)) not in smap:
                continue
            paths.append(f)
            meta[str(f)] = {"source_state_id": int(smap[int(m.group(2))]), "source_phase": str(phase)}
    measured = measure_structures(paths, ctx, mode)
    return [{"path": p, **meta[p], **v} for p, v in measured.items()]


# ---------------------------------------------------------------------------------------
# The epoch-boundary runner
# ---------------------------------------------------------------------------------------

def _targets(states) -> List[Dict[str, Any]]:
    return [{"state_id": int(s.state_id), "primary_center": float(s.primary_center),
             "primary_k": float(s.primary_k),
             "secondary_center": None if s.secondary_center is None else float(s.secondary_center),
             "secondary_k": None if s.secondary_k is None else float(s.secondary_k),
             "gamd_lambda": float(s.gamd_lambda)} for s in states]


def _default_seed_paths(seed_bank_dir) -> Dict[int, str]:
    """target state -> the seed the bank's state-aware assignment names for it. Without an
    override, seeding re-ranks the whole filtered pool by CV distance, so the structure a
    window is actually grafted from may be another state's seed; "assigned" is a proxy."""
    import csv

    if seed_bank_dir is None:
        return {}
    path = Path(seed_bank_dir) / "state_aware_seed_assignments.csv"
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            p = Path(row.get("seed_pdb_path", ""))
            p = p if p.is_absolute() else Path(seed_bank_dir) / p
            if row.get("target_state_id") not in ("", None) and p.exists():
                out[int(float(row["target_state_id"]))] = str(p)
    return out


BURN_IN_NOTE = (
    "Every numbered epoch and the final phase re-graft and re-pull EVERY window from a seed-bank "
    "structure (generate_us_starting_states_by_pulling); X3 changes which structure a reseeded "
    "window is grafted from, not whether its segment starts from a fresh graft. Discards that "
    "apply to it are exactly those that apply to every other window: the restrained US pull "
    "(--us-pull-steps-per-window) runs before the first production sample and is never written; "
    "the per-state registry burnin_steps (default 0) is applied by the union Parquet loader; the "
    "driver's union MBAR detects one equilibration t0 per state on the state's pooled trace "
    "(mbar_subsample.equilibrated_subsample), which trims the start of that trace only, not each "
    "epoch's; analyze_gareus_mbar.py --skip-first-n-frames trims per replica. MBAR is unchanged.")


def _window_rows(targets, occ, plan, default_measured, default_paths, one_sided_max):
    chosen = {int(r["target_state_id"]): r for r in plan["reseeded"]}
    rows = []
    for t in targets:
        sid = int(t["state_id"])
        o = occ.get(sid, {})
        d = default_measured.get(default_paths.get(sid, ""), {})
        c = chosen.get(sid)
        rows.append({"state_id": sid, "gamd_lambda": t["gamd_lambda"], "n_frames": o.get("n_frames", 0),
                     "frac_high": o.get("frac_high"), "minority_fraction": o.get("minority_fraction"),
                     "minority_side": o.get("minority_side"),
                     "one_sided": bool(o) and o.get("n_frames", 0) >= X3.MIN_FRAMES_PER_STATE
                     and float(o.get("minority_fraction", 1.0)) <= one_sided_max,
                     "assigned_seed_side": d.get("side"), "reseeded": c is not None,
                     "start_side_after": c["seed_side"] if c else d.get("side")})
    return rows


def _hidden_mode_report(mode, table, ctx, occ_all):
    from gareus.cv_selection.slowness import anchor_cells, conditional_autocorrelation, max_bimodality_at_fixed_anchor

    z = mode.scores(table.X, table.cv1)
    cells = anchor_cells(table.cv1, 8)
    ev = float(mode.eigenvalue)
    v = np.asarray(ctx.runtime.fit.right_vectors[ctx.runtime.j - 1])
    return {
        "definition": "leading conditional tICA mode of the torsion residual after CV1 (the frozen fit's "
                      "regression) and the deployed CV2 direction are removed; reference-free",
        "deployed_cv2_component": int(ctx.runtime.j),
        "deployed_cv2_family": str(ctx.runtime.fit.families[ctx.runtime.j - 1]),
        "lag_steps": int(table.lag_steps), "lag_ps": float(table.lag_ps),
        "tica_eigenvalue": ev,
        "implied_timescale_ps": float(-table.lag_ps / math.log(ev)) if 0.0 < ev < 1.0 else None,
        "conditional_autocorrelation_at_lag": float(conditional_autocorrelation(
            z, cells, table.member, table.frame_index, table.lag_steps)),
        "max_bimodality_at_fixed_cv1": float(max_bimodality_at_fixed_anchor(z, cells)),
        "abs_cos_to_deployed_cv2": float(abs(mode.direction @ v)),
        "split": float(mode.split), "split_method": mode.split_method,
        "pooled_frac_high": float(np.mean(mode.side(z))),
        "loadings": X3.torsion_loadings(mode.direction, ctx.torsion_labels),
        "direction": [float(x) for x in mode.direction],
    }


def seed_preflight(targets, seed_args=None) -> Dict[str, Any]:
    """The seed preflight generate_us_starting_states_by_pulling applies to this window set
    (same spacing helper over the same centres, same defaults and the same ``or`` fallbacks)."""
    from gareus.seeding import _finite_spacing_scale

    a = seed_args if seed_args is not None else argparse.Namespace()
    mode = str(getattr(a, "seed_selection_mode", "auto") or "auto").strip().lower().replace("_", "-")
    weight = max(0.0, float(getattr(a, "seed_secondary_weight", 1.0) or 0.0))
    secondary = [t["secondary_center"] for t in targets if t.get("secondary_center") is not None]
    return {"max_score": float(getattr(a, "us_seed_preflight_max_score", 1.2) or 1.2),
            "primary_scale": _finite_spacing_scale([t["primary_center"] for t in targets], fallback=0.2),
            "secondary_scale": _finite_spacing_scale(secondary, fallback=0.25) if secondary else None,
            "secondary_weight": weight if mode in ("auto", "active-cv") else 0.0}


def plan_epoch(phase_dirs, targets, ctx: PairContext, *, fraction: float, temperature_k: float,
               seed_bank_dir=None, lag_ps: float = X3.DEFAULT_LAG_PS, stride: int = DEFAULT_STRIDE,
               timestep_fs: Optional[float] = None, seed_args=None) -> Dict[str, Any]:
    """Fit, measure and plan (no writes). Returns the report body.

    ``seed_args`` (the campaign args, or anything carrying ``us_seed_preflight_max_score``,
    ``seed_secondary_weight``, ``seed_selection_mode``) sets the seeding preflight the
    chosen seeds must pass; ``None`` uses the CLI defaults.
    """
    table = load_frame_table(phase_dirs, ctx, lag_ps=lag_ps, stride=stride, timestep_fs=timestep_fs)
    mode = X3.fit_hidden_mode(ctx.runtime.fit, ctx.runtime.j, table.X, table.cv1, table.member,
                              table.frame_index, lag=table.lag_steps)
    occ = X3.side_occupancy(mode.side(mode.scores(table.X, table.cv1)), table.state_id)
    candidates = end_state_candidates(phase_dirs, ctx, mode)
    rt_kcal = X3.R_KCAL * float(temperature_k)
    default_paths = _default_seed_paths(seed_bank_dir)
    default_measured = measure_structures(sorted(set(default_paths.values())), ctx, mode)
    assigned_sides = {sid: default_measured[p]["side"] for sid, p in default_paths.items() if p in default_measured}
    plan = X3.plan_reseed(targets, occ, candidates, fraction=fraction, rt_kcal=rt_kcal,
                          assigned_sides=assigned_sides, preflight=seed_preflight(targets, seed_args))
    lam = {int(t["state_id"]): t["gamd_lambda"] for t in targets}
    for r in plan["reseeded"]:
        src = r.get("seed_source_state_id")
        r["seed_from_other_rung"] = bool(src is not None and src in lam and lam[src] != r["target_gamd_lambda"])
    windows = _window_rows(targets, occ, plan, default_measured, default_paths, X3.ONE_SIDED_MINORITY_MAX)
    resd = [w for w in windows if w["reseeded"]]
    summary = {
        "n_windows": plan["n_windows"], "n_one_sided": plan["n_one_sided"], "n_target": plan["n_target"],
        "n_reseeded": plan["n_reseeded"],
        "n_seed_from_other_state": sum(1 for r in plan["reseeded"] if r["seed_from_other_state"]),
        "n_seed_from_other_rung": sum(1 for r in plan["reseeded"] if r["seed_from_other_rung"]),
        "reseeded_assigned_seed_on_minority_side": sum(1 for w in resd if w["assigned_seed_side"] == w["minority_side"]),
        "windows_assigned_seed_on_minority_side": sum(1 for w in windows if w["minority_side"] is not None
                                                        and w["assigned_seed_side"] == w["minority_side"]),
        "windows_start_on_minority_side_after_x3": sum(1 for w in windows if w["minority_side"] is not None
                                                       and w["start_side_after"] == w["minority_side"]),
        "assigned_seed_note": "assigned = the seed the bank's state-aware assignment names; without X3 "
                              "seeding re-ranks the filtered pool by CV distance, so the grafted "
                              "structure may differ. seed_from_other_rung = source state has another lambda",
        "n_candidates": len(candidates),
        "candidates_per_side": {str(s): sum(1 for c in candidates if c["side"] == s) for s in (0, 1)},
    }
    return {"hidden_mode": _hidden_mode_report(mode, table, ctx, occ),
            "data": {"n_frames": int(table.X.shape[0]), "n_trajectories": int(np.unique(table.member).size),
                     "sources": table.sources, "temperature_k": float(temperature_k)},
            "summary": summary, "rules": plan["rules"], "reseeded": plan["reseeded"],
            "skipped": plan["skipped"], "windows": windows, "burn_in": BURN_IN_NOTE}


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def _is_propagated_bank(seed_bank_dir, epoch_dir) -> bool:
    """A bank the adaptive driver itself wrote (``adaptive_production/seed_bank_*``)."""
    bank = Path(seed_bank_dir)
    return bank.name.startswith("seed_bank_") and bank.resolve().parent == Path(epoch_dir).resolve().parent


def run_epoch_slow_mode_reseed(*, fraction: float, epoch_dir, phase_dirs, states, seed_bank_dir, args
                               ) -> Dict[str, Any]:
    """Plan X3 for the next epoch and inject it into ``seed_bank_dir``. Never raises."""
    report: Dict[str, Any] = {"schema_version": "slow_mode_reseed_v1", "fraction": float(fraction),
                              "epoch_dir": str(epoch_dir), "seed_bank_dir": str(seed_bank_dir or "")}
    if seed_bank_dir is not None and not _is_propagated_bank(seed_bank_dir, epoch_dir):
        seed_bank_dir = None      # never write into a user's own conformer library
    try:
        if seed_bank_dir is not None:
            X3.clear_overrides(seed_bank_dir)          # a stale override never outlives its epoch
        paths = pair_paths_from_args(args)
        phases = [Path(p) for p in phase_dirs if Path(p).is_dir()]
        solute = next((p / "solute_only.pdb" for p in phases if (p / "solute_only.pdb").exists()), None)
        if seed_bank_dir is None:
            report.update(status="skipped", reason="no propagated seed bank for the next epoch")
        elif paths is None:
            report.update(status="skipped", reason="no frozen residual pair (--secondary-cv-model/"
                          "--secondary-cv-candidate-set/--secondary-cv-feature-schema)")
        elif solute is None:
            report.update(status="skipped", reason="no solute_only.pdb in the epoch's phases")
        else:
            ctx = load_pair_context(paths, solute)
            body = plan_epoch(phases, _targets(states), ctx, fraction=float(fraction),
                              temperature_k=float(getattr(args, "temperature_k", 300.0) or 300.0),
                              seed_bank_dir=seed_bank_dir, seed_args=args,
                              timestep_fs=float(getattr(args, "timestep_fs", 0.0) or 0.0) or None)
            report.update(body)
            if body["reseeded"]:
                report["override_dir"] = str(X3.write_overrides(seed_bank_dir, body["reseeded"]))
            report["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 -- X3 must never stop an epoch
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
        try:
            if seed_bank_dir is not None:
                X3.clear_overrides(seed_bank_dir)
        except Exception:  # noqa: BLE001
            pass
        print(f"WARNING [slow-mode reseed]: {report['error']}; the next epoch seeds as without X3")
    try:
        _write_report(Path(epoch_dir) / X3.REPORT_NAME, report)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING [slow-mode reseed]: could not write {X3.REPORT_NAME} ({exc})")
    s = report.get("summary") or {}
    print(f"    Slow-mode reseed: status {report['status']}; {s.get('n_reseeded', 0)}/{s.get('n_windows', 0)} "
          f"window(s) reseeded ({s.get('n_one_sided', 0)} one-sided)")
    return report


# ---------------------------------------------------------------------------------------
# Read-only replay:  python -m gareus.adaptive.slow_mode_reseed_io <adaptive_dir> ...
# ---------------------------------------------------------------------------------------

def _registry_targets(adaptive_dir: Path, phases: Sequence[Path]) -> List[Dict[str, Any]]:
    """The states the last phase ran, with their restraints, from the registry."""
    reg = json.loads((adaptive_dir / "state_registry.json").read_text())
    rows = reg.get("states", reg) if isinstance(reg, dict) else reg
    by_id = {int(r["state_id"]): r for r in (rows.values() if isinstance(rows, dict) else rows)}
    sids = sorted(set(_state_of_window(phases[-1]).values()))

    def _f(v):
        return None if v in (None, "", "None") else float(v)

    return [{"state_id": s, "primary_center": float(by_id[s]["primary_center"]),
             "primary_k": float(by_id[s]["primary_k"]), "secondary_center": _f(by_id[s].get("secondary_center")),
             "secondary_k": _f(by_id[s].get("secondary_k")),
             "gamd_lambda": float(by_id[s].get("gamd_lambda") or 0.0)} for s in sids]


def _campaign_args(adaptive_dir: Path):
    """The campaign's recorded args (``run_args.json`` beside ``adaptive_production``), if any."""
    path = Path(adaptive_dir).parent / "run_args.json"
    try:
        return argparse.Namespace(**json.loads(path.read_text()))
    except (OSError, ValueError, TypeError):
        return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Read-only X3 replay on a finished campaign; writes only --out.")
    p.add_argument("adaptive_dir", type=Path)
    p.add_argument("--phases", nargs="+", required=True, help="phase dirs relative to adaptive_dir")
    p.add_argument("--pair-dir", type=Path, required=True, help="dir holding cv_pair_model/candidate_set/feature_schema")
    p.add_argument("--fraction", type=float, default=0.25)
    p.add_argument("--temperature-k", type=float, default=300.0)
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    p.add_argument("--lag-ps", type=float, default=X3.DEFAULT_LAG_PS)
    p.add_argument("--seed-bank", type=Path, default=None,
                   help="the next epoch's seed bank, read only (for the default-seed before/after view)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    phases = [a.adaptive_dir / ph for ph in a.phases]
    paths = tuple(a.pair_dir / n for n in ("cv_pair_model.json", "cv_candidate_set.json", "cv_feature_schema.json"))
    ctx = load_pair_context(paths, phases[0] / "solute_only.pdb")
    body = plan_epoch(phases, _registry_targets(a.adaptive_dir, phases), ctx, fraction=a.fraction,
                      temperature_k=a.temperature_k, lag_ps=a.lag_ps, stride=a.stride,
                      seed_bank_dir=a.seed_bank, seed_args=_campaign_args(a.adaptive_dir))
    a.out.mkdir(parents=True, exist_ok=True)
    _write_report(a.out / X3.REPORT_NAME, {"schema_version": "slow_mode_reseed_v1", "status": "replay", **body})
    hm, s = body["hidden_mode"], body["summary"]
    print(json.dumps({"loadings_top5": hm["loadings"][:5], "eigenvalue": hm["tica_eigenvalue"],
                      "autocorr": hm["conditional_autocorrelation_at_lag"], "split": [hm["split"], hm["split_method"]],
                      "summary": s}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
