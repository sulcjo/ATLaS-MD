"""Read-only: the harness's candidate "replica-path" crossing count on chignolin_9 epoch_002.

For each R3 candidate with core bounds in the dry-run report, count core-to-core label changes
between consecutive visits of the SAME replica to the state (joined across its absences, never
across a sample source or segment, since a new phase re-grafts and re-pulls every window), next
to the shipped replica-residence and state-series counts (which this script recomputes as a
check on its own bookkeeping).

python t3c_replica_path_c9.py <adaptive_dir> <dry_report.json> <phase> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path('/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler')
sys.path.insert(0, str(REPO))
import gareus.adaptive_production as ap  # noqa: E402
from gareus.adaptive import cv2_resolution as cr  # noqa: E402
from gareus.adaptive.effective_samples import contiguous_runs  # noqa: E402
from gareus.query import load_samples  # noqa: E402


def labels(z, bounds):
    ua, lb = bounds
    lab = np.where(z <= ua, 0, np.where(z >= lb, 1, -1))
    return lab[lab >= 0]


def main(adaptive_dir, report_path, phase, out_path):
    adaptive_dir = Path(adaptive_dir)
    rep = json.load(open(report_path))
    cands = {c['state_ids'][0]: c for c in rep['candidates']
             if c['rule'] == 'R3' and (c.get('metrics') or {}).get('core_bounds')}
    reg = ap.WindowStateRegistry.load(adaptive_dir)
    sources = ap._sample_sources_from_run_root(phase, adaptive_dir / phase)
    res = {s: {"replica_residence": 0, "state_series": 0, "replica_path": 0, "n_rows": 0, "n_replicas": set(),
               "mean_residence_samples": []} for s in cands}
    for label, sample_dir in sources:
        data = load_samples(Path(sample_dir))
        wmap = ap._load_epoch_window_map(Path(sample_dir), reg)
        steps = np.asarray(data['step'], dtype=np.int64)
        win = np.asarray(data['window_id'], dtype=np.int64)
        seg = np.asarray(data.get('segment_id') if data.get('segment_id') is not None else np.zeros(steps.size)).astype(str)
        repl = np.asarray(data['replica'], dtype=np.int64)
        cv2 = np.asarray(np.ma.asarray(data['cv2']).astype(float).filled(np.nan), dtype=float)
        state = np.asarray([int(wmap.get(int(w), -1)) for w in win], dtype=np.int64)
        for sid, c in cands.items():
            b = c['metrics']['core_bounds']
            r = res[sid]
            for g in np.unique(seg[state == sid]):
                idx = np.flatnonzero((state == sid) & (seg == g))
                idx = idx[np.argsort(steps[idx], kind='stable')]
                r['n_rows'] += idx.size
                stride = None
                for run in contiguous_runs(steps[idx], cv2[idx]):
                    lab = labels(run, b)
                    r['state_series'] += int(np.count_nonzero(np.diff(lab))) if lab.size > 1 else 0
                d = np.diff(steps[idx]); d = d[d > 0]
                if d.size:
                    vals, cnt = np.unique(d, return_counts=True)
                    stride = int(vals[np.argmax(cnt)])
                for rr in np.unique(repl[idx]):
                    ridx = idx[repl[idx] == rr]
                    r['n_replicas'].add(int(rr))
                    runs = contiguous_runs(steps[ridx], cv2[ridx], stride=stride)
                    r['mean_residence_samples'].extend(len(x) for x in runs)
                    for run in runs:
                        lab = labels(run, b)
                        r['replica_residence'] += int(np.count_nonzero(np.diff(lab))) if lab.size > 1 else 0
                    zz = cv2[ridx]
                    lab = labels(zz[np.isfinite(zz)], b)
                    r['replica_path'] += int(np.count_nonzero(np.diff(lab))) if lab.size > 1 else 0
        print(f'{label}: done', flush=True)
    out = {}
    for sid, r in res.items():
        m = cands[sid]['metrics']
        out[str(sid)] = {"decision": cands[sid]['decision'], "gate": m.get('r3_gate'), "core_bounds": m['core_bounds'],
                         "report_transitions": m.get('transitions'),
                         "report_state_series": m.get('transitions_state_series'),
                         "replica_residence": r['replica_residence'], "state_series": r['state_series'],
                         "replica_path": r['replica_path'], "n_rows": r['n_rows'], "n_replicas": len(r['n_replicas']),
                         "mean_residence_samples": float(np.mean(r['mean_residence_samples'])) if r['mean_residence_samples'] else None}
        print(sid, out[str(sid)])
    json.dump({"phase": phase, "report": str(report_path), "states": out}, open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:5])
