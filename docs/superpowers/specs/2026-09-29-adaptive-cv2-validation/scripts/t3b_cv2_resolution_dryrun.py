"""Read-only, cap-ignoring R1-R3 dry run on a replayed chignolin_9 payload.

Usage: python t3b_cv2_resolution_dryrun.py <adaptive_dir> <payload.json> <phase|final_combined> <out.json>
       [--union NPZ] [--transition-count replica|replica-path|state-series] [--r3-mode flag|insert] [--coverage-count any|same-column]
Nothing under RUNS is written (the history is kept in memory, the report goes to out.json).
"""
import json
import sys
import time
from argparse import Namespace
from collections import Counter
from pathlib import Path

import gareus.adaptive_production as ap
from gareus.adaptive import cv2_resolution_io as cio

adaptive_dir, payload_path, phase, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
union_npz = Path(sys.argv[sys.argv.index("--union") + 1]) if "--union" in sys.argv else None
diag = json.loads(payload_path.read_text())
reg = ap.WindowStateRegistry.load(adaptive_dir)
count = sys.argv[sys.argv.index("--transition-count") + 1] if "--transition-count" in sys.argv else "replica-path"
r3_mode = sys.argv[sys.argv.index("--r3-mode") + 1] if "--r3-mode" in sys.argv else "flag"
cov_count = sys.argv[sys.argv.index("--coverage-count") + 1] if "--coverage-count" in sys.argv else "same-column"
policy = ap.AdaptiveDecisionPolicy(cv2_resolution=True, edge_metric="pairwise-mbar", max_replicas_budget=236,
                                   refine_transition_count=count, refine_r3_mode=r3_mode,
                                   coverage_count=cov_count)
args = Namespace(temperature_k=300.0, cv2_k_min=0.0, cv2_k_max=1000.0, cv1_k_min=0.0)
settings = cio.settings_from(policy, args)
if phase == "final_combined":
    sources = ap._final_sample_dirs(adaptive_dir)
else:
    sources = ap._sample_sources_from_run_root(phase, adaptive_dir / phase)
runs_for = cio.parquet_runs_provider(sources, lambda d: ap._load_epoch_window_map(d, reg))
t0 = time.time()
union = None
if union_npz is not None:
    union = cio.union_coverage(union_npz, cio._column_views(reg, diag), settings, epoch=2, max_gb=16.0)
t_union = time.time() - t0
actions, report, _hist = cio.propose_cv2_resolution(
    reg, diag, [], settings, policy, epoch=2, history={}, subsamples=cio._subsamples(diag), runs_for=runs_for,
    union=union, union_reason="no union for this phase (top-ups were off)", reserve=None, gate=None,
    ignore_budget=True)
report["stage"] = "replay"
report["wall_s"] = {"total": time.time() - t0, "union": t_union}
out.write_text(json.dumps(report, indent=1, default=str))
counts = Counter((c["rule"], c.get("class"), c["decision"], (c.get("refusal") or c["reason"].split(":")[0])[:60])
                 for c in report["candidates"])
print(json.dumps({"phase": phase, "rules": {k: {kk: v.get(kk) for kk in ("status", "reason", "classes",
                                                                           "n_candidate_edges", "skipped_edges",
                                                                           "n_components", "n_rows")}
                                             for k, v in report["rules"].items()},
                  "summary": report["summary"], "wall_s": report["wall_s"],
                  "actions": Counter(a[0] for a in actions)}, indent=1, default=str))
for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(v, k)
