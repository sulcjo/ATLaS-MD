#!/usr/bin/env python3
"""Extract folding/unfolding pathway trajectories from committed Poincaré crossing events.

Usage (simplest):
    python extract_poincare_pathways.py RUNS/chignolin_2d_run2_66ns/

With explicit crossing CSVs (e.g. from /tmp):
    python extract_poincare_pathways.py RUNS/chignolin_2d_run2_66ns/ \\
        --fold-csv /tmp/poincare_full_test/poincare_fold_crossings.csv \\
        --unfold-csv /tmp/poincare_full_test/poincare_unfold_crossings.csv

Control output size:
    python extract_poincare_pathways.py RUNS/.../ --max-frames 300 --top-n 10
    python extract_poincare_pathways.py RUNS/.../ --window-only --context-frames 50

Modes
-----
Default (full pathway):  extract the entire U→F or F→U trajectory segment.
  Long events are sub-sampled to --max-frames.

--window-only:  extract only ±context_frames around the threshold crossing point.
  Use when you want a compact "transition state ensemble" movie rather than the
  full folding trajectory. Much faster and smaller output.

Outputs  (<run_dir>/poincare_pathways/ by default)
-------
  folding_route_A/pathway_001.xtc   + _frame0.pdb (topology for VMD/PyMOL)
  folding_route_B/pathway_001.xtc
  unfolding/pathway_001.xtc
  all_folding_route_A.xtc           concatenated ensemble (all Route A events)
  all_folding_route_B.xtc
  all_unfolding.xtc
  pathway_summary.csv
  README.txt

Loading in VMD:
  mol new folding_route_A/pathway_001_frame0.pdb
  mol addfile folding_route_A/pathway_001.xtc type xtc waitfor all
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('run_dir', nargs='?', default='.',
                   help='GAREUS run directory  [default: current dir]')
    p.add_argument('--fold-csv', default=None,
                   help='poincare_fold_crossings.csv  [auto-detect]')
    p.add_argument('--unfold-csv', default=None,
                   help='poincare_unfold_crossings.csv  [auto-detect]')
    p.add_argument('--topology', default=None,
                   help='Topology PDB  [auto-detect]')
    p.add_argument('--traj-dir', default=None,
                   help='replica_trajectories dir  [auto-detect]')
    p.add_argument('--out-dir', default=None,
                   help='Output directory  [<run_dir>/poincare_pathways]')
    p.add_argument('--route-split', type=float, default=None,
                   help='CV2 cutpoint: Route A below, Route B above  [auto]')
    p.add_argument('--top-n', type=int, default=20,
                   help='Max pathways per category  [20]')
    p.add_argument('--max-frames', type=int, default=500,
                   help='Cap extracted pathway length; longer paths are sub-sampled  [500]')
    p.add_argument('--window-only', action='store_true',
                   help='Extract only ±context_frames around the crossing (compact movies)')
    p.add_argument('--context-frames', type=int, default=25,
                   help='Frames of context to add before/after each crossing  [25]')
    p.add_argument('--step-per-frame', type=int, default=None,
                   help='Simulation steps per trajectory frame  [auto-detect, default 50]')
    p.add_argument('--no-protein-only', action='store_true',
                   help='Keep all atoms including solvent  [default: protein only]')
    p.add_argument('--no-concat', action='store_true',
                   help='Skip writing concatenated ensemble trajectories')
    p.add_argument('--min-frames', type=int, default=3,
                   help='Skip pathways shorter than this many frames  [3]')
    return p.parse_args()


# ── Auto-detection helpers ────────────────────────────────────────────────────

def _find_production_dir(run_dir: Path) -> Path:
    for name in ('final_production', 'production', 'prod'):
        p = run_dir / name
        if p.is_dir():
            return p
    for p in sorted(run_dir.iterdir()):
        if p.is_dir() and any(p.rglob('*.xtc')):
            return p
    return run_dir


def _find_traj_dir(prod_dir: Path) -> Path:
    for name in ('replica_trajectories', 'trajectories', 'trajs'):
        p = prod_dir / name
        if p.is_dir() and any(p.glob('*.xtc')):
            return p
    if any(prod_dir.glob('replica_*.xtc')):
        return prod_dir
    return prod_dir


def _find_topology(run_dir: Path, prod_dir: Path) -> Path:
    candidates = [
        prod_dir / 'shared_gamd_setup_final.pdb',
        prod_dir / 'topology.pdb',
    ] + sorted(run_dir.glob('02_npt_equilibrated.pdb')) + sorted(run_dir.glob('*.pdb'))
    for p in candidates:
        if p.exists() and 'CRASH' not in p.name:
            return p
    raise FileNotFoundError(f'No topology PDB in {run_dir} or {prod_dir}')


def _find_crossing_csvs(run_dir: Path, prod_dir: Path):
    for base_dir in [run_dir, prod_dir, prod_dir / 'analysis']:
        fold = base_dir / 'poincare_fold_crossings.csv'
        unfold = base_dir / 'poincare_unfold_crossings.csv'
        if fold.exists() and unfold.exists():
            return fold, unfold
    # deep search
    for fold in run_dir.rglob('poincare_fold_crossings.csv'):
        unfold = fold.parent / 'poincare_unfold_crossings.csv'
        if unfold.exists():
            return fold, unfold
    raise FileNotFoundError(
        'Cannot find poincare_fold_crossings.csv / poincare_unfold_crossings.csv.\n'
        'Run analyze_gareus_mbar.py first, then pass --fold-csv and --unfold-csv.'
    )


def _read_step_per_frame(prod_dir: Path) -> int:
    for name in ('effective_config.json', 'run_args.json'):
        p = prod_dir / name
        if not p.exists():
            continue
        try:
            cfg = json.loads(p.read_text())
            for loc in (cfg, cfg.get('args', {}), cfg.get('output', {})):
                if isinstance(loc, dict) and loc.get('traj_interval'):
                    return int(loc['traj_interval'])
        except Exception:
            pass
    try:
        import yaml
        p = prod_dir / 'effective_config.yaml'
        if p.exists():
            cfg = yaml.safe_load(p.read_text())
            for loc in (cfg, cfg.get('args', {}), cfg.get('output', {})):
                if isinstance(loc, dict) and loc.get('traj_interval'):
                    return int(loc['traj_interval'])
    except Exception:
        pass
    return 50


# ── XTC segment helpers ───────────────────────────────────────────────────────

def _find_segments(traj_dir: Path, rep: int) -> list:
    """Return [(resume_start_step, Path), ...] sorted chronologically."""
    base_path = None
    for name in [f'replica_{rep:03d}.xtc', f'replica_{rep}.xtc']:
        p = traj_dir / name
        if p.exists() and p.stat().st_size > 100:
            base_path = p; break
    resume_re = re.compile(rf'^replica_{rep:03d}_resume_from_(\d+)\.xtc$')
    resumes = []
    for f in sorted(traj_dir.iterdir()):
        m = resume_re.match(f.name)
        if m and f.stat().st_size > 100:
            resumes.append((int(m.group(1)), f))
    resumes.sort(key=lambda x: x[0])
    return ([(0, base_path)] if base_path else []) + resumes


def _frames_in_segment(s_start: int, s_end: int,
                        resume_start: int, n_frames: int, spf: int):
    """Return (lo, hi) inclusive local frame indices, or None if no overlap.
    Frame i: abs_step = resume_start + (i+1)*spf
    """
    seg_first = resume_start + spf
    seg_last  = resume_start + n_frames * spf
    if s_start > seg_last or s_end < seg_first:
        return None
    lo = max(0, int(np.floor((s_start - resume_start) / spf)) - 1)
    hi = min(n_frames - 1, int(np.ceil((s_end - resume_start) / spf)) - 1)
    return (lo, hi) if hi >= lo else None


# ── Crossing CSV ──────────────────────────────────────────────────────────────

def load_crossings(csv_path: Path) -> list:
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({'replica': int(r['replica']),
                         'step': int(r['step']),
                         'cv2': float(r['cv2'])})
    return rows


# ── Route split detection ─────────────────────────────────────────────────────

def detect_route_split(fold_crossings, fold_csv: Path, prod_dir: Path, run_dir: Path) -> float:
    """Load from poincare_map_summary.json (preferred) or recompute with full-range KDE."""
    for summary_path in [
        fold_csv.parent / 'poincare_map_summary.json',
        prod_dir / 'poincare_map_summary.json',
        run_dir / 'poincare_map_summary.json',
    ]:
        if summary_path.exists():
            try:
                s = json.loads(summary_path.read_text())
                peaks = s.get('fold_peaks', [])
                if len(peaks) >= 2:
                    split = float((peaks[0]['cv2'] + peaks[1]['cv2']) / 2)
                    print(f'Route split: {split:.4f}  '
                          f'(poincare_map_summary peaks {peaks[0]["cv2"]:.3f}, {peaks[1]["cv2"]:.3f})')
                    return split
            except Exception:
                pass

    cv2_vals = np.array([e['cv2'] for e in fold_crossings])
    try:
        from scipy.stats import gaussian_kde
        from scipy.signal import argrelmax
        kde = gaussian_kde(cv2_vals, bw_method=0.12)
        xg = np.linspace(-1.0, 1.0, 400)
        dens = kde(xg)
        pks = argrelmax(dens, order=10)[0]
        peaks_kd = sorted([(xg[i], dens[i]) for i in pks], key=lambda t: -t[1])
        if len(peaks_kd) >= 2:
            split = float((peaks_kd[0][0] + peaks_kd[1][0]) / 2)
            print(f'Route split: {split:.4f}  (KDE peaks {peaks_kd[0][0]:.3f}, {peaks_kd[1][0]:.3f})')
            return split
    except Exception:
        pass
    split = float(np.median(cv2_vals))
    print(f'Route split: {split:.4f}  (median fallback)')
    return split


# ── Pair matching ─────────────────────────────────────────────────────────────

def match_pairs(fold_crossings, unfold_crossings, route_split: float):
    """Match each fold crossing to the nearest preceding unfold in the same replica."""
    unfold_by_rep = defaultdict(list)
    for e in unfold_crossings:
        unfold_by_rep[e['replica']].append(e)
    fold_by_rep = defaultdict(list)
    for e in fold_crossings:
        fold_by_rep[e['replica']].append(e)

    fold_pairs, unfold_pairs = [], []

    for rep, fevents in fold_by_rep.items():
        uevents = sorted(unfold_by_rep[rep], key=lambda x: x['step'])
        for fe in sorted(fevents, key=lambda x: x['step']):
            preceding_u = [u for u in uevents if u['step'] < fe['step']]
            if not preceding_u:
                continue
            ue = max(preceding_u, key=lambda x: x['step'])
            route = 'B' if fe['cv2'] >= route_split else 'A'
            fold_pairs.append({
                'type': f'folding_route_{route}', 'route': route, 'replica': rep,
                's_start': ue['step'], 's_end': fe['step'],
                'cv2_start': ue['cv2'], 'cv2_end': fe['cv2'],
                'duration_steps': fe['step'] - ue['step'],
                'crossing_step': fe['step'],  # the fold threshold crossing frame
            })

    for rep, uevents in unfold_by_rep.items():
        fevents = sorted(fold_by_rep[rep], key=lambda x: x['step'])
        for ue in sorted(uevents, key=lambda x: x['step']):
            preceding_f = [f for f in fevents if f['step'] < ue['step']]
            if not preceding_f:
                continue
            fe = max(preceding_f, key=lambda x: x['step'])
            route = 'B' if fe['cv2'] >= route_split else 'A'
            unfold_pairs.append({
                'type': 'unfolding', 'route': route, 'replica': rep,
                's_start': fe['step'], 's_end': ue['step'],
                'cv2_start': fe['cv2'], 'cv2_end': ue['cv2'],
                'duration_steps': ue['step'] - fe['step'],
                'crossing_step': ue['step'],
            })

    return fold_pairs, unfold_pairs


# ── Batched trajectory extraction ─────────────────────────────────────────────

def _get_n_frames(xtc_path: Path) -> int:
    """Count frames in an XTC without loading atom positions."""
    try:
        import mdtraj.formats as mf
        with mf.XTCTrajectoryFile(str(xtc_path)) as xf:
            return len(xf)
    except Exception:
        return -1


def extract_all_pathways(pairs, traj_dir: Path, topology_path: Path,
                          spf: int, context: int, max_frames: int,
                          window_only: bool, protein_only: bool,
                          min_frames: int) -> list:
    """Extract trajectories for a list of pathway pairs.

    Batches loads by (replica, segment_file) so each XTC is loaded at most once.
    Returns list of (pair_dict, mdtraj.Trajectory or None).
    """
    import mdtraj as md

    top = md.load_topology(str(topology_path))
    if protein_only:
        try:
            atom_sel = top.select('protein')
        except Exception:
            atom_sel = None
    else:
        atom_sel = None

    # Build per-replica segment list
    rep_ids = list({p['replica'] for p in pairs})
    rep_segments = {r: _find_segments(traj_dir, r) for r in rep_ids}

    # Determine step range for each pair
    def _step_range(pair):
        if window_only:
            cx = pair['crossing_step']
            return cx - context * spf, cx + context * spf
        else:
            return pair['s_start'] - context * spf, pair['s_end'] + context * spf

    # Group (pair_index, segment_path) → list of frame ranges needed
    # Structure: seg_path → [(pair_idx, lo, hi), ...]
    seg_to_ranges = defaultdict(list)
    pair_step_ranges = [_step_range(p) for p in pairs]
    seg_n_frames_cache = {}

    for pi, pair in enumerate(pairs):
        rep = pair['replica']
        s_start, s_end = pair_step_ranges[pi]
        for resume_start, seg_path in rep_segments[rep]:
            # Fast n_frames lookup
            if seg_path not in seg_n_frames_cache:
                nf = _get_n_frames(seg_path)
                seg_n_frames_cache[seg_path] = nf
            nf = seg_n_frames_cache[seg_path]
            if nf <= 0:
                continue
            fr = _frames_in_segment(s_start, s_end, resume_start, nf, spf)
            if fr is not None:
                seg_to_ranges[seg_path].append((pi, fr[0], fr[1], resume_start))

    # Load each segment once; slice out all needed frame ranges
    pair_pieces = defaultdict(list)  # pair_idx → [traj_piece, ...]

    all_seg_paths = sorted(seg_to_ranges.keys(), key=lambda p: str(p))
    for seg_path in all_seg_paths:
        ranges = seg_to_ranges[seg_path]
        if not ranges:
            continue
        try:
            t_full = md.load(str(seg_path), top=str(topology_path))
        except Exception as e:
            print(f'  Warning: cannot load {seg_path.name}: {e}')
            continue
        if atom_sel is not None:
            t_full = t_full.atom_slice(atom_sel)
        for pi, lo, hi, _ in ranges:
            piece = t_full[lo:hi + 1]
            if piece.n_frames > 0:
                pair_pieces[pi].append(piece)

    # Assemble and sub-sample
    results = []
    for pi, pair in enumerate(pairs):
        pieces = pair_pieces.get(pi, [])
        if not pieces:
            results.append((pair, None))
            continue
        try:
            if len(pieces) == 1:
                traj = pieces[0]
            else:
                traj = md.join(pieces)
        except Exception as e:
            print(f'  Warning: join failed for pair {pi}: {e}')
            results.append((pair, None))
            continue
        if traj.n_frames < min_frames:
            results.append((pair, None))
            continue
        # Sub-sample if longer than max_frames
        if traj.n_frames > max_frames:
            stride = max(1, traj.n_frames // max_frames)
            traj = traj[::stride]
        results.append((pair, traj))

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    prod_dir = _find_production_dir(run_dir)
    traj_dir = Path(args.traj_dir) if args.traj_dir else _find_traj_dir(prod_dir)
    topology = Path(args.topology) if args.topology else _find_topology(run_dir, prod_dir)
    spf = args.step_per_frame or _read_step_per_frame(prod_dir)
    timestep_fs = 4.0  # assumed; 4 fs with HMR

    print(f'Run dir:    {run_dir}')
    print(f'Traj dir:   {traj_dir}')
    print(f'Topology:   {topology}')
    print(f'Step/frame: {spf}  ({spf * timestep_fs / 1000:.3f} ps/frame)')
    print(f'Mode:       {"window-only ±" + str(args.context_frames) + " frames" if args.window_only else "full U→F/F→U pathway"}')
    print(f'Max frames: {args.max_frames} per pathway (sub-sampled if longer)')

    # Locate crossing CSVs
    if args.fold_csv and args.unfold_csv:
        fold_csv = Path(args.fold_csv)
        unfold_csv = Path(args.unfold_csv)
    else:
        try:
            fold_csv, unfold_csv = _find_crossing_csvs(run_dir, prod_dir)
        except FileNotFoundError as e:
            sys.exit(f'Error: {e}')

    print(f'\nFold CSV:   {fold_csv}')
    print(f'Unfold CSV: {unfold_csv}')

    fold_crossings = load_crossings(fold_csv)
    unfold_crossings = load_crossings(unfold_csv)
    print(f'Fold crossings:   {len(fold_crossings)}')
    print(f'Unfold crossings: {len(unfold_crossings)}')

    if not fold_crossings or not unfold_crossings:
        sys.exit('No crossings found. Run analyze_gareus_mbar.py first.')

    # Route split
    if args.route_split is not None:
        route_split = args.route_split
        print(f'Route split: {route_split:.4f}  (user-specified)')
    else:
        route_split = detect_route_split(fold_crossings, fold_csv, prod_dir, run_dir)

    print(f'  → Route A: CV2 < {route_split:.4f}  |  Route B: CV2 ≥ {route_split:.4f}')

    fold_pairs, unfold_pairs = match_pairs(fold_crossings, unfold_crossings, route_split)
    route_A = [p for p in fold_pairs if p['route'] == 'A']
    route_B = [p for p in fold_pairs if p['route'] == 'B']

    print(f'\nMatched pairs:')
    print(f'  Folding Route A:  {len(route_A):4d}  '
          f'(median dur {np.median([p["duration_steps"] for p in route_A]) * timestep_fs / 1000:.1f} ps)'
          if route_A else f'  Folding Route A:  {len(route_A):4d}')
    print(f'  Folding Route B:  {len(route_B):4d}  '
          f'(median dur {np.median([p["duration_steps"] for p in route_B]) * timestep_fs / 1000:.1f} ps)'
          if route_B else f'  Folding Route B:  {len(route_B):4d}')
    print(f'  Unfolding:        {len(unfold_pairs):4d}  '
          f'(median dur {np.median([p["duration_steps"] for p in unfold_pairs]) * timestep_fs / 1000:.1f} ps)'
          if unfold_pairs else f'  Unfolding:        {len(unfold_pairs):4d}')

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / 'poincare_pathways'
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in ('folding_route_A', 'folding_route_B', 'unfolding'):
        (out_dir / d).mkdir(exist_ok=True)

    import mdtraj as md

    categories = [
        ('folding_route_A', route_A,     out_dir / 'folding_route_A'),
        ('folding_route_B', route_B,     out_dir / 'folding_route_B'),
        ('unfolding',       unfold_pairs, out_dir / 'unfolding'),
    ]

    summary_rows = []
    concat_trajs = {}

    for cat_name, pairs, cat_dir in categories:
        if not pairs:
            print(f'\n[{cat_name}] No pairs.')
            concat_trajs[cat_name] = []
            continue

        # Sort: shortest first (most committed transitions)
        top_pairs = sorted(pairs, key=lambda p: p['duration_steps'])[:args.top_n]
        print(f'\n[{cat_name}] Extracting top {len(top_pairs)} / {len(pairs)} '
              f'(shortest-first) ...')

        extracted = extract_all_pathways(
            top_pairs, traj_dir, topology, spf,
            args.context_frames, args.max_frames, args.window_only,
            not args.no_protein_only, args.min_frames
        )

        cat_trajs = []
        for idx, (pair, traj) in enumerate(extracted, 1):
            dur_ps = pair['duration_steps'] * timestep_fs / 1000.0
            if traj is None:
                print(f'  [{idx:03d}] rep {pair["replica"]:02d} '
                      f'{pair["s_start"]}→{pair["s_end"]}  SKIPPED')
                continue
            out_xtc = cat_dir / f'pathway_{idx:03d}.xtc'
            out_pdb = cat_dir / f'pathway_{idx:03d}_frame0.pdb'
            try:
                traj.save_xtc(str(out_xtc))
                traj[0].save_pdb(str(out_pdb))
            except Exception as e:
                print(f'  [{idx:03d}] save failed: {e}'); continue

            print(f'  [{idx:03d}] rep {pair["replica"]:02d}  '
                  f'cv2 {pair["cv2_start"]:+.3f}→{pair["cv2_end"]:+.3f}  '
                  f'{dur_ps:.0f} ps  {traj.n_frames} frames  → {out_xtc.name}')

            cat_trajs.append(traj)
            summary_rows.append({
                'category': cat_name, 'index': idx,
                'replica': pair['replica'],
                's_start': pair['s_start'], 's_end': pair['s_end'],
                'duration_ps': f'{dur_ps:.1f}',
                'n_frames': traj.n_frames,
                'cv2_start': f'{pair["cv2_start"]:.4f}',
                'cv2_end': f'{pair["cv2_end"]:.4f}',
                'route': pair.get('route', ''),
                'xtc': str(out_xtc.relative_to(out_dir)),
                'topology_pdb': str(out_pdb.relative_to(out_dir)),
            })

        concat_trajs[cat_name] = cat_trajs

    # Concatenated ensemble trajectories
    if not args.no_concat:
        print('\nWriting concatenated ensemble trajectories ...')
        for cat_name, trajs in concat_trajs.items():
            if not trajs:
                continue
            try:
                combined = md.join(trajs)
                out_xtc = out_dir / f'all_{cat_name}.xtc'
                out_pdb = out_dir / f'all_{cat_name}_topology.pdb'
                combined.save_xtc(str(out_xtc))
                combined[0].save_pdb(str(out_pdb))
                print(f'  {out_xtc.name}  ({combined.n_frames} frames from {len(trajs)} events)')
            except Exception as e:
                print(f'  {cat_name}: concat failed: {e}')

    # Summary CSV
    if summary_rows:
        summary_csv = out_dir / 'pathway_summary.csv'
        with open(summary_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader(); w.writerows(summary_rows)
        print(f'\nSummary: {summary_csv}  ({len(summary_rows)} pathways)')

    # README
    n_A = sum(1 for r in summary_rows if r['category'] == 'folding_route_A')
    n_B = sum(1 for r in summary_rows if r['category'] == 'folding_route_B')
    n_U = sum(1 for r in summary_rows if r['category'] == 'unfolding')
    cv2_A_mean = (np.mean([float(r['cv2_end']) for r in summary_rows if r['category'] == 'folding_route_A'])
                  if n_A else float('nan'))
    (out_dir / 'README.txt').write_text(f"""Poincaré Pathway Trajectories
==============================
Run:          {run_dir}
Topology:     {topology}
Route split:  CV2 = {route_split:.4f}
              Route A (CV2 < split): dominant folding route
              Route B (CV2 ≥ split): minor folding route
Step/frame:   {spf} steps  ({spf * timestep_fs / 1000:.3f} ps/frame)
Mode:         {"window-only ±" + str(args.context_frames) + " frames around crossing" if args.window_only else "full U→F/F→U trajectory (sub-sampled to " + str(args.max_frames) + " frames max)"}

Extracted
---------
  folding_route_A/   {n_A} trajectories  (CV2 ≈ {cv2_A_mean:.3f} at fold crossing)
  folding_route_B/   {n_B} trajectories
  unfolding/         {n_U} trajectories
  Sorted shortest-first (most committed transitions first).

Loading in VMD
--------------
  mol new folding_route_A/pathway_001_frame0.pdb
  mol addfile folding_route_A/pathway_001.xtc type xtc waitfor all

Ensemble (all events back-to-back):
  mol new all_folding_route_A_topology.pdb
  mol addfile all_folding_route_A.xtc type xtc waitfor all

Loading in PyMOL
----------------
  load folding_route_A/pathway_001_frame0.pdb, traj
  load_traj folding_route_A/pathway_001.xtc, traj

MDTraj
------
  import mdtraj as md
  t = md.load('folding_route_A/pathway_001.xtc',
              top='folding_route_A/pathway_001_frame0.pdb')
  print(t)  # n_frames, n_atoms, time range

Caveats
-------
- REUS: replicas exchange umbrella potentials (not coordinates). Atom positions
  are physically continuous. Pathways are real trajectories under biased sampling.
- The umbrella bias is present; these are NOT unbiased folding events.
- Recurrence times are REUS-biased; do not interpret as physical folding rates.
- Route B has very few events; statistics are indicative only.
- Sub-sampling is applied when pathway > {args.max_frames} frames.
""")
    print(f'Output dir: {out_dir}')


if __name__ == '__main__':
    main()
