"""Campaign frames for aux discovery: each phase's solute XTCs joined on (replica, step) to its Parquet
samples (deduplicated, last segment wins), state from the phase's epoch_window_map.csv, lambda = 0 only."""
from __future__ import annotations

import multiprocessing
import re
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from gareus.adaptive import discovery_census as census
from gareus.topup_seeding import state_id_of_window_from_epoch_map
from .descriptors import DescriptorDefinition, descriptor_definition, evaluate_descriptors

_EPOCH_RE = re.compile(r"^epoch_(\d{3})(?:/|$)")
_ARRAY_FIELDS = ("phase", "epoch", "replica", "step", "state_id", "lam", "cv1", "cv2",
                 "tors", "tors_theta_iupac", "hc", "hb", "basin")
_DESC_FIELDS = ("tors", "tors_theta_iupac", "hc", "hb", "basin")


_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _single_thread_worker():
    for var in _THREAD_VARS:
        os.environ[var] = "1"


def phase_epoch(label: str) -> Optional[int]:
    """``epoch_002/baseline`` -> 2, ``final*`` -> None."""
    m = _EPOCH_RE.match(str(label))
    return int(m.group(1)) if m else None


@dataclass(frozen=True)
class FrameTable:
    phase: np.ndarray
    epoch: np.ndarray
    replica: np.ndarray
    step: np.ndarray
    state_id: np.ndarray
    lam: np.ndarray
    cv1: np.ndarray
    cv2: np.ndarray
    tors: np.ndarray
    tors_theta_iupac: np.ndarray
    hc: np.ndarray
    hb: np.ndarray
    basin: np.ndarray
    definition: DescriptorDefinition
    sources: List[dict] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.step.size)

    @property
    def lineage(self) -> np.ndarray:
        return np.char.add(np.char.add(self.phase.astype(str), ":"), self.replica.astype(str))

    def take(self, idx: np.ndarray) -> "FrameTable":
        arrays = {k: getattr(self, k)[idx] for k in _ARRAY_FIELDS}
        return FrameTable(**arrays, definition=self.definition, sources=self.sources)


def load_phase_samples(phase_dir: Path) -> pd.DataFrame:
    """replica, step, window_id, cv1, cv2, gamd_lambda; one row per (replica, step), last segment wins."""
    import duckdb
    q = (f"select replica, step, window_id, cv1, cv2, gamd_lambda, filename from "
         f"read_parquet('{Path(phase_dir)}/samples/*/*.parquet', filename=true, union_by_name=true)")
    df = duckdb.connect().execute(q).df()
    df["seg"] = [int(m.group(1)) if (m := re.search(r"seg_(\d+)", f)) else 0 for f in df["filename"]]
    df = df.sort_values(["replica", "step", "seg"], kind="stable").drop_duplicates(["replica", "step"], keep="last")
    return df.drop(columns=["filename", "seg"]).reset_index(drop=True)


def _read_xtc(path: Path):
    """(xyz nm (F,N,3), steps int64) from one read; XTC ``step`` is the authority for steps."""
    import mdtraj as md
    with md.formats.XTCTrajectoryFile(str(path), "r") as fh:
        xyz, _time, step, _box = fh.read()
    return np.asarray(xyz, dtype=np.float32), np.asarray(step, dtype=np.int64)


def _phase_frames(task):
    """One phase -> (definition, list of (frame DataFrame, descriptor dict)). Module level for spawn."""
    import mdtraj as md
    label, phase_dir, adaptive_dir, definition, registry_lambda, stride_steps = task
    phase_dir, adaptive_dir = Path(phase_dir), Path(adaptive_dir)
    top_path = census._find_topology(phase_dir, adaptive_dir)
    if top_path is None:
        raise FileNotFoundError(f"{phase_dir}: no solute_only.pdb")
    own = descriptor_definition(md.load_topology(str(top_path)))
    if definition is None:
        definition = own
    elif own.schema_sha256 != definition.schema_sha256:
        raise ValueError(f"phase {label}: descriptor schema {own.schema_sha256[:12]} differs from the "
                         f"campaign's {definition.schema_sha256[:12]} (topology {top_path})")
    wmap = state_id_of_window_from_epoch_map(phase_dir)
    if not wmap:
        raise RuntimeError(f"{phase_dir}: no epoch_window_map.csv")
    samples = load_phase_samples(phase_dir)
    n_before = len(samples)
    samples["state_id"] = samples["window_id"].map(lambda w: wmap.get(int(w), -1)).astype(np.int64)
    samples["lam_reg"] = samples["state_id"].map(lambda s: registry_lambda.get(int(s), np.nan))
    n_unmapped = int(n_before - ((samples["state_id"] >= 0) & samples["lam_reg"].notna()).sum())
    samples = samples[(samples["state_id"] >= 0) & (samples["lam_reg"].abs() < 1e-12)]
    samples = samples.drop(columns=["lam_reg"])
    files = sorted(census._trajectory_files(phase_dir), key=lambda t: (t[0], t[1] or 0, str(t[2])))
    blocks = []
    anchors: Dict[int, int] = {}                     # one stride anchor per replica per phase
    for i, (replica, _start, path) in enumerate(files):
        if Path(path).stat().st_size == 0:
            continue
        nxt = next((s for r, s, _ in files[i + 1:] if r == replica), None)
        try:
            xyz, steps = _read_xtc(path)
        except Exception as exc:
            raise RuntimeError(f"cannot read trajectory {path}: {type(exc).__name__}: {exc}") from exc
        if steps.size == 0:
            continue
        keep = np.ones(steps.size, bool) if nxt is None else steps < int(nxt)    # later resume file wins
        if not keep.any():
            continue
        kept = np.nonzero(keep)[0]
        anchor = anchors.setdefault(int(replica), int(steps[kept][0]))   # earliest file's first kept step
        kept = kept[((steps[kept] - anchor) % int(stride_steps)) == 0]
        frames = pd.DataFrame({"replica": int(replica), "step": steps[kept], "pos": kept})
        joined = frames.merge(samples, on=["replica", "step"], how="inner")      # XTC-only frames skipped
        if joined.empty:
            continue
        desc = evaluate_descriptors(xyz[joined["pos"].to_numpy()], definition)
        blocks.append((joined.drop(columns=["pos"]).reset_index(drop=True), desc))
    return definition, blocks, n_unmapped


def build_frame_table(adaptive_dir: Path, *, epochs: Iterable[int], registry_lambda: Dict[int, float],
                      stride_steps: int, max_frames: int, seed: int, workers: int = 1) -> FrameTable:
    """lambda = 0 frames of the numbered epochs ``epochs`` (lambda from ``registry_lambda``, state -> lambda),
    strided on steps within each trajectory file, then at most ``max_frames`` drawn uniformly at random."""
    adaptive_dir = Path(adaptive_dir)
    root = adaptive_dir if adaptive_dir.name == "adaptive_production" else adaptive_dir / "adaptive_production"
    wanted = {int(e) for e in epochs}
    phases, _skipped = census.ordered_phases(root)
    todo = [(label, Path(pd_)) for label, pd_ in phases if phase_epoch(label) in wanted]
    definition = None
    if todo:
        top0 = census._find_topology(todo[0][1], root)
        if top0 is None:
            raise FileNotFoundError(f"{todo[0][1]}: no solute_only.pdb")
        import mdtraj as md
        definition = descriptor_definition(md.load_topology(str(top0)))
    if int(workers) > 1 and len(todo) > 1:
        # spawn, never fork: the driver process may hold CUDA contexts (as discovery_census.run_census)
        for var in _THREAD_VARS:  # spawned workers inherit this; avoids BLAS/OpenMP oversubscription
            os.environ[var] = "1"
        with ProcessPoolExecutor(max_workers=int(workers), mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_single_thread_worker) as pool:
            results = list(pool.map(_phase_frames, [(l, p, root, definition, registry_lambda, stride_steps)
                                                   for l, p in todo]))
    else:
        results = []
        for l, p in todo:
            results.append(_phase_frames((l, p, root, definition, registry_lambda, stride_steps)))
    cols = {k: [] for k in _ARRAY_FIELDS}
    sources = []
    for (label, phase_dir), (_def, blocks, n_unmapped) in zip(todo, results):
        n_phase = 0
        for sub, desc in blocks:
            n = len(sub)
            n_phase += n
            cols["phase"].append(np.full(n, label, dtype=object))
            cols["epoch"].append(np.full(n, phase_epoch(label), dtype=np.int64))
            cols["replica"].append(sub["replica"].to_numpy(np.int64))
            cols["step"].append(sub["step"].to_numpy(np.int64))
            cols["state_id"].append(sub["state_id"].to_numpy(np.int64))
            cols["lam"].append(np.zeros(n))
            cols["cv1"].append(sub["cv1"].to_numpy(np.float32))
            cols["cv2"].append(sub["cv2"].to_numpy(np.float32))
            for k in _DESC_FIELDS:
                cols[k].append(desc[k])
        sources.append({"phase": label, "dir": str(phase_dir), "n_frames": int(n_phase), "n_dropped_unmapped": n_unmapped})
    if definition is None or not cols["step"]:
        raise RuntimeError(f"no lambda = 0 frames in epochs {sorted(wanted)} under {root}")
    arrays = {k: np.concatenate(v) for k, v in cols.items()}
    arrays["phase"] = arrays["phase"].astype(str)
    ft = FrameTable(**arrays, definition=definition, sources=sources)
    if ft.n > int(max_frames):
        rng = np.random.default_rng(int(seed))
        ft = ft.take(np.sort(rng.choice(ft.n, size=int(max_frames), replace=False)))
    return ft
