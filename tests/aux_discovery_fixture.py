"""Shared synthetic campaign phases for aux-discovery tests (Task 6, reused by later tasks)."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd

PDB = Path(__file__).parent / "data" / "chignolin_solute.pdb"
DT_PS = 0.0035


def make_phase(root: Path, name: str, steps_by_file, lam_by_window, cv_by_step, *,
               sample_step_offset: int = 0) -> Path:
    """``root/adaptive_production/<name>``: solute PDB, XTCs (one per entry of ``steps_by_file``:
    ``fname -> (replica, window, steps)``), one Parquet segment, segments.json and epoch_window_map.csv
    (window w -> state 100 + w). ``sample_step_offset`` > 0 = a self-calibrated GaMD phase: samples log
    ``calib_steps + prod_done`` while the XTC carries production-relative steps; gareus_metadata.json
    records ``shared_gamd_calibration_steps`` as production writes it."""
    ph = root / "adaptive_production" / name
    (ph / "replica_trajectories").mkdir(parents=True)
    (ph / "samples" / "seg_000").mkdir(parents=True)
    t = md.load(str(PDB))
    shutil.copy(PDB, ph / "solute_only.pdb")
    rows = []
    for fname, (replica, window, steps) in steps_by_file.items():
        xyz = np.repeat(t.xyz, len(steps), axis=0)
        with md.formats.XTCTrajectoryFile(str(ph / "replica_trajectories" / fname), "w") as fh:
            fh.write(xyz, time=np.asarray(steps) * DT_PS, step=np.asarray(steps, dtype=np.int32))
        for s in steps:
            rows.append({"step": s + int(sample_step_offset), "replica": replica, "window_id": window, "cv1": cv_by_step(s + int(sample_step_offset)),
                         "cv2": 0.0, "gamd_lambda": lam_by_window[window]})
    pd.DataFrame(rows).drop_duplicates(["replica", "step"]).to_parquet(ph / "samples" / "seg_000" / "data.parquet")
    (ph / "segments.json").write_text(json.dumps([{"segment": 0}]))
    if sample_step_offset:
        (ph / "gareus_metadata.json").write_text(json.dumps({"shared_gamd_calibration_steps": int(sample_step_offset)}))
    with (ph / "epoch_window_map.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch_window", "state_id"])
        for win in lam_by_window:
            w.writerow([win, 100 + win])
    return ph


def frozen_partition(seed: int = 0):
    """A real ``FrozenPartition`` fitted on a planted two-state synthetic set (picklable, ``to_file`` works).
    For plumbing tests whose campaign is too small for a partition fit of its own."""
    from gareus.adaptive.aux_discovery import partitions as P
    from gareus.adaptive.aux_discovery.settings import AuxDiscoverySettings
    rng = np.random.default_rng(seed)
    n_lin, per = 40, 60
    n = n_lin * per
    lineage = np.repeat([f"p:{i}" for i in range(n_lin)], per)
    step = np.tile(np.arange(per) * 3000, n_lin)
    cv = rng.normal(size=(n, 2)).astype(np.float32)
    state = np.repeat(rng.integers(0, 2, n_lin), per)
    X = np.hstack([rng.normal(size=(n, 6)) + 3.0 * state[:, None], rng.normal(size=(n, 6))])
    train = np.repeat(np.arange(n_lin) < 28, per)
    res = P.fit_partition(X, ["a"] * 6 + ["b"] * 6, cv, train, ~train, lineage, step,
                          AuxDiscoverySettings(k_max=3), seed=seed)
    assert res.frozen is not None, res.status
    return res.frozen
