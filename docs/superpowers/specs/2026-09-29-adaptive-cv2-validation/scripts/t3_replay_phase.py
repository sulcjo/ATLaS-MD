"""T3 read-only replay: run one phase's diagnostics collector with ``edge_metric="pairwise-mbar"``.

Usage: python t3_replay_phase.py <adaptive_dir> <phase|final_combined> <out_dir>

Every file the collector would write under RUNS/ (``write_json``, ``write_text_atomic``,
the P4 paired-CV NPZ sidecar) is redirected into ``out_dir`` (path relative to the
adaptive dir, "/" -> "__"). The registry is loaded once (both campaigns kept every
state from epoch 0 on; only ``extend`` actions were ever applied). Verify afterwards
with ``find RUNS/<campaign> -newer <stamp>`` that nothing under RUNS changed.
"""
import json
import resource
import sys
import time
from pathlib import Path

import gareus.adaptive_production as ap

adaptive_dir = Path(sys.argv[1]).resolve()
phase = sys.argv[2]
out = Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
RUNS = str(adaptive_dir.parents[1])


def _redirect(path) -> Path:
    p = Path(path)
    if str(p.resolve()).startswith(RUNS):
        return out / str(p.resolve().relative_to(adaptive_dir)).replace("/", "__")
    return p


_real_write = ap.write_json
_real_text = ap.write_text_atomic
_real_attach = ap.attach_paired_cv
ap.write_json = lambda path, payload, *a, **k: _real_write(_redirect(path), payload, *a, **k)
ap.write_text_atomic = lambda path, text, *a, **k: _real_text(_redirect(path), text, *a, **k)
ap.attach_paired_cv = lambda payload, collector, json_path: _real_attach(payload, collector, _redirect(json_path))

reg = ap.WindowStateRegistry.load(adaptive_dir)
policy = ap.AdaptiveDecisionPolicy(edge_metric="pairwise-mbar")
t0 = time.time()
if phase == "final_combined":
    payload = ap.collect_final_combined_diagnostics(adaptive_dir, reg, policy=policy)
else:
    payload = ap.collect_segmented_epoch_diagnostics(adaptive_dir / phase, reg, policy)
wall = time.time() - t0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
target = out / f"t3_{phase.replace('/', '__')}_payload.json"
target.write_text(json.dumps(ap._json_ready(payload) if hasattr(ap, "_json_ready") else payload, default=float))
rec = payload.get("edge_metric", {})
print(json.dumps({"phase": phase, "wall_s": round(wall, 1), "maxrss_gb": round(rss, 2),
                  "status": rec.get("status"), "error": rec.get("error"),
                  "n_weak": rec.get("n_weak"), "n_unmeasured": rec.get("n_unmeasured"),
                  "graded": rec.get("n_edges_graded_by_type"),
                  "components": {k: v for k, v in (rec.get("components") or {}).items() if k != "components"},
                  "paired_npz": (payload.get("paired_cv") or {}).get("npz"), "payload": str(target)}))
