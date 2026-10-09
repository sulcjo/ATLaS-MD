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


def make_phase(root: Path, name: str, steps_by_file, lam_by_window, cv_by_step) -> Path:
    """``root/adaptive_production/<name>``: solute PDB, XTCs (one per entry of ``steps_by_file``:
    ``fname -> (replica, window, steps)``), one Parquet segment, segments.json and epoch_window_map.csv
    (window w -> state 100 + w)."""
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
            rows.append({"step": s, "replica": replica, "window_id": window, "cv1": cv_by_step(s),
                         "cv2": 0.0, "gamd_lambda": lam_by_window[window]})
    pd.DataFrame(rows).drop_duplicates(["replica", "step"]).to_parquet(ph / "samples" / "seg_000" / "data.parquet")
    (ph / "segments.json").write_text(json.dumps([{"segment": 0}]))
    with (ph / "epoch_window_map.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch_window", "state_id"])
        for win in lam_by_window:
            w.writerow([win, 100 + win])
    return ph
