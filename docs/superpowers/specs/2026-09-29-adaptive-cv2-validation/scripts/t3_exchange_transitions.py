"""T3 task 2 (b): realised replica transitions between states, per phase, read-only.

Usage: python t3_exchange_transitions.py <adaptive_dir> <out.json> <phase> [<phase> ...]

Reads every ``<phase>[/sub-run]`` exchange table through ``gareus.query.load_exchanges``
(committed files only), additionally drops exact duplicate rows (step, replica_i, replica_j,
window_i, window_j) as a guard, maps windows to state ids through that run
directory's own ``epoch_window_map.csv`` (refuses a run dir holding an
``epoch_window_map_rewrites.json``: none exist in c7/c9), and counts per unordered state pair:
attempts, accepted (= realised transitions of one replica each way), and the per-replica
simulated time of the run (step span x timestep_fs from run_manifest.json).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ad = Path(sys.argv[1])
out_path = Path(sys.argv[2])
phases = sys.argv[3:]


def run_dirs(phase_dir: Path):
    if (phase_dir / "exchanges").is_dir():
        yield phase_dir
    for sub in sorted(p for p in phase_dir.iterdir() if p.is_dir()):
        if (sub / "exchanges").is_dir() and (sub / "epoch_window_map.csv").exists():
            yield sub


def window_map(d: Path):
    if (d / "epoch_window_map_rewrites.json").exists():
        raise SystemExit(f"{d} has a map rewrite ledger; handle it explicitly")
    with (d / "epoch_window_map.csv").open() as fh:
        return {int(r["epoch_window"]): int(r["state_id"]) for r in csv.DictReader(fh)}


result = {}
for phase in phases:
    pdir = ad / phase
    tot = {}
    runs = []
    for d in run_dirs(pdir):
        wm = window_map(d)
        man = json.loads((d / "run_manifest.json").read_text())
        dt_fs = float(man["resolved_args"]["timestep_fs"])
        # The canonical loader: committed files per parquet_manifest.json, segments.json status
        # and end_step filtering (a raw glob double-counts consolidated chunks: 18-35 % of rows
        # in several c7/c9 segments are such copies).
        from gareus.query import load_exchanges
        import pandas as pd
        data = load_exchanges(d)
        if not data or len(data.get("step", [])) == 0:
            continue
        df = pd.DataFrame({k: np.asarray(data[k]) for k in ("step", "replica_i", "replica_j", "window_i",
                                                             "window_j", "accepted")})
        n_raw = len(df)
        df = df.drop_duplicates(subset=["step", "replica_i", "replica_j", "window_i", "window_j"])
        steps = df["step"].to_numpy()
        span_ns = float(steps.max() - steps.min()) * dt_fs * 1e-6
        si = np.array([wm.get(int(w), -1) for w in df["window_i"].to_numpy()])
        sj = np.array([wm.get(int(w), -1) for w in df["window_j"].to_numpy()])
        acc = df["accepted"].to_numpy().astype(bool)
        lo, hi = np.minimum(si, sj), np.maximum(si, sj)
        key = lo.astype(np.int64) * 100000 + hi
        uk, inv = np.unique(key, return_inverse=True)
        att = np.bincount(inv)
        accn = np.bincount(inv, weights=acc)
        for k, a, c in zip(uk.tolist(), att.tolist(), accn.tolist()):
            t = tot.setdefault(k, [0, 0])
            t[0] += a
            t[1] += int(c)
        runs.append(dict(run=str(d.relative_to(ad)), rows=n_raw, dedup_rows=len(df), span_ns=span_ns,
                         unmapped=int(((si < 0) | (sj < 0)).sum())))
    result[phase] = dict(runs=runs, span_ns=sum(r["span_ns"] for r in runs),
                         pairs={f"{k // 100000}-{k % 100000}": v for k, v in tot.items()})
    print(phase, runs)
out_path.write_text(json.dumps(result))
