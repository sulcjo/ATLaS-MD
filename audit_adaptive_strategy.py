#!/usr/bin/env python3
"""Adaptive sampling strategy audit for GAREUS runs.

Answers: Do epochs/topups help? What is the baseline vs topup efficiency ratio?
Which run parts bought the most new CV coverage per ns?

Usage:
    python audit_adaptive_strategy.py RUNS/run_name/adaptive_production/ [--out DIR] [--bins N]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

TIMESTEP_FS = 4.0
SAVE_INTERVAL_STEPS = 50
NS_PER_SAMPLE = TIMESTEP_FS * SAVE_INTERVAL_STEPS / 1e6  # 0.0002 ns


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class RunPart:
    label: str
    epoch: int           # -1 = final
    part_type: str       # "flat" | "baseline" | "topup" | "final_baseline" | "final_topup"
    topup_round: int     # 0 if not topup
    topup_steps: int     # 0 if not topup
    samples_dir: Path
    epoch_dir: Path
    # Populated after loading
    n_samples: int = 0
    ns: float = 0.0
    per_window_samples: dict = field(default_factory=dict)
    cv1: Optional[np.ndarray] = None
    cv2: Optional[np.ndarray] = None
    window_ids: Optional[np.ndarray] = None
    boost: Optional[np.ndarray] = None
    new_bins: int = 0
    cumulative_bins: int = 0
    exchange_acceptance: float = float('nan')
    loaded: bool = False


# ─── Discovery ───────────────────────────────────────────────────────────────

def _topup_key(name: str) -> tuple[int, int]:
    """topup_003_1532000 → (3, 1532000)"""
    parts = name.split('_')
    try:
        return int(parts[1]), int(parts[2])
    except (IndexError, ValueError):
        return (0, 0)


def discover_run_parts(adaptive_dir: Path) -> list[RunPart]:
    parts: list[RunPart] = []

    # epoch_000: flat run with samples/ directly
    ep0 = adaptive_dir / 'epoch_000'
    if (ep0 / 'samples').exists():
        parts.append(RunPart(
            label='ep000/flat', epoch=0, part_type='flat',
            topup_round=0, topup_steps=0,
            samples_dir=ep0, epoch_dir=ep0,
        ))

    # epoch_001 .. epoch_NNN
    for ep_dir in sorted(adaptive_dir.glob('epoch_???')):
        if ep_dir.name == 'epoch_000':
            continue
        ep_num = int(ep_dir.name.split('_')[1])

        baseline = ep_dir / 'baseline'
        if baseline.exists() and (baseline / 'samples').exists():
            parts.append(RunPart(
                label=f'ep{ep_num:03d}/baseline', epoch=ep_num, part_type='baseline',
                topup_round=0, topup_steps=0,
                samples_dir=baseline, epoch_dir=ep_dir,
            ))

        topups = sorted(
            [d for d in ep_dir.iterdir() if d.is_dir() and d.name.startswith('topup_') and (d / 'samples').exists()],
            key=lambda d: _topup_key(d.name),
        )
        for td in topups:
            rnd, steps = _topup_key(td.name)
            parts.append(RunPart(
                label=f'ep{ep_num:03d}/{td.name}', epoch=ep_num, part_type='topup',
                topup_round=rnd, topup_steps=steps,
                samples_dir=td, epoch_dir=ep_dir,
            ))

    # final/
    final_dir = adaptive_dir / 'final'
    if final_dir.exists():
        baseline = final_dir / 'baseline'
        if baseline.exists() and (baseline / 'samples').exists():
            parts.append(RunPart(
                label='final/baseline', epoch=-1, part_type='final_baseline',
                topup_round=0, topup_steps=0,
                samples_dir=baseline, epoch_dir=final_dir,
            ))
        topups = sorted(
            [d for d in final_dir.iterdir() if d.is_dir() and d.name.startswith('topup_') and (d / 'samples').exists()],
            key=lambda d: _topup_key(d.name),
        )
        for td in topups:
            rnd, steps = _topup_key(td.name)
            parts.append(RunPart(
                label=f'final/{td.name}', epoch=-1, part_type='final_topup',
                topup_round=rnd, topup_steps=steps,
                samples_dir=td, epoch_dir=final_dir,
            ))

    return parts


# ─── Loading ─────────────────────────────────────────────────────────────────

def _load_part_task(part: RunPart, n_threads: int) -> RunPart:
    import duckdb
    sdir = part.samples_dir / 'samples'
    if not sdir.exists():
        return part

    files = sorted(str(f) for f in sdir.rglob('*.parquet'))
    if not files:
        return part

    conn = duckdb.connect()
    if n_threads > 0:
        conn.execute(f'SET threads={n_threads}')

    r = conn.execute(
        'SELECT window_id, cv1, cv2, gamd_boost_total FROM read_parquet(?)',
        [files],
    ).fetchnumpy()
    conn.close()

    if not r or 'cv1' not in r or len(r['cv1']) == 0:
        return part

    part.window_ids = r['window_id'].astype(np.int32)
    part.cv1 = r['cv1'].astype(np.float32)
    part.cv2 = r['cv2'].astype(np.float32) if 'cv2' in r else None
    part.boost = r['gamd_boost_total'].astype(np.float32) if 'gamd_boost_total' in r else None
    part.n_samples = len(part.cv1)
    part.ns = part.n_samples * NS_PER_SAMPLE

    for wid in np.unique(part.window_ids):
        part.per_window_samples[int(wid)] = int(np.sum(part.window_ids == wid))

    part.loaded = True
    return part


def load_all_parts(parts: list[RunPart], n_workers: int = 8) -> list[RunPart]:
    n_threads = max(1, min(8, 64 // max(1, n_workers)))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        result = list(pool.map(lambda p: _load_part_task(p, n_threads), parts))
    print(f'  [load] {len(result)} parts in {time.time()-t0:.1f}s')
    return result


# ─── Coverage tracking ────────────────────────────────────────────────────────

def compute_coverage(parts: list[RunPart], bins: int = 30) -> None:
    seen: set[int] = set()
    cv1_min, cv1_max = 0.0, 1.0
    cv2_min, cv2_max = -1.0, 1.0

    for p in parts:
        if not p.loaded:
            p.new_bins = 0
            p.cumulative_bins = len(seen)
            continue
        # 2D bin indices packed as single int
        b1 = np.clip(((p.cv1 - cv1_min) / (cv1_max - cv1_min) * bins).astype(np.int32), 0, bins - 1)
        if p.cv2 is not None:
            b2 = np.clip(((p.cv2 - cv2_min) / (cv2_max - cv2_min) * bins).astype(np.int32), 0, bins - 1)
        else:
            b2 = np.zeros(len(b1), dtype=np.int32)
        ids = set(map(int, b1 * bins + b2))
        new = ids - seen
        seen |= ids
        p.new_bins = len(new)
        p.cumulative_bins = len(seen)


# ─── Metadata loading ─────────────────────────────────────────────────────────

def _rjson(path: Path, default=None):
    try:
        return json.loads(path.read_text()) if path.exists() else default
    except Exception:
        return default


def load_exchange_acceptance(part: RunPart) -> float:
    report = _rjson(part.samples_dir / 'exchange_tuning_report.json')
    if report and 'acceptance_fraction' in report:
        return float(report['acceptance_fraction'])
    return float('nan')


def load_epoch_metadata(adaptive_dir: Path) -> dict:
    """Load per-epoch: actions, convergence gate, diagnostics."""
    meta = {}
    for ep_dir in sorted(adaptive_dir.glob('epoch_???')):
        ep = int(ep_dir.name.split('_')[1])
        actions = _rjson(ep_dir / 'adaptive_epoch_actions.json', {})
        gate = _rjson(ep_dir / 'adaptive_convergence_gate.json', {})
        meta[ep] = {
            'action_counts': actions.get('action_counts', {}),
            'weak_edges': len(gate.get('weak_edges', [])),
            'low_sample_states': len(gate.get('low_sample_states', [])),
            'high_boost_states': len(gate.get('high_boost_states', [])),
            'status': gate.get('status', '?'),
            'stop_adaptive': gate.get('stop_adaptive', False),
        }
    return meta


def load_all_exchange(parts: list[RunPart]) -> None:
    for p in parts:
        p.exchange_acceptance = load_exchange_acceptance(p)


# ─── Analysis & report ───────────────────────────────────────────────────────

def per_window_matrix(parts: list[RunPart]) -> tuple[np.ndarray, list[int], list[str]]:
    all_windows = sorted({w for p in parts for w in p.per_window_samples})
    labels = [p.label for p in parts]
    mat = np.zeros((len(all_windows), len(parts)), dtype=np.int32)
    for j, p in enumerate(parts):
        for i, w in enumerate(all_windows):
            mat[i, j] = p.per_window_samples.get(w, 0)
    return mat, all_windows, labels


def per_window_boost(parts: list[RunPart]) -> tuple[np.ndarray, list[int]]:
    """Mean boost std per window per part, shape (n_windows, n_parts)."""
    all_windows = sorted({w for p in parts for w in p.per_window_samples})
    mat = np.full((len(all_windows), len(parts)), np.nan)
    for j, p in enumerate(parts):
        if not p.loaded or p.boost is None:
            continue
        for i, w in enumerate(all_windows):
            mask = p.window_ids == w
            if mask.sum() > 2:
                mat[i, j] = float(np.std(p.boost[mask]))
    return mat, all_windows


def print_summary_table(parts: list[RunPart]) -> None:
    total_ns = sum(p.ns for p in parts)
    header = f"{'Label':<30} {'Type':<14} {'Samples':>9} {'ns':>8} {'New bins':>9} {'Cum bins':>9} {'Eff (bins/ns)':>14} {'Exch acc':>9}"
    print(header)
    print('-' * len(header))
    for p in parts:
        eff = p.new_bins / p.ns if p.ns > 0 else 0.0
        exch = f'{p.exchange_acceptance:.3f}' if np.isfinite(p.exchange_acceptance) else '   N/A'
        print(f'{p.label:<30} {p.part_type:<14} {p.n_samples:>9,} {p.ns:>8.2f} {p.new_bins:>9,} {p.cumulative_bins:>9,} {eff:>14.1f} {exch:>9}')
    print(f"\nTotal ns: {total_ns:.2f}")
    print(f"Final unique 2D bins: {max((p.cumulative_bins for p in parts), default=0)}")


def print_epoch_meta_table(meta: dict) -> None:
    if not meta:
        return
    print(f"\n{'Epoch':<7} {'Extend':>7} {'Retire':>7} {'Add':>7} {'Weak edges':>11} {'Low-sample':>11} {'High-boost':>11} {'Status':<10}")
    print('-' * 70)
    for ep in sorted(meta):
        m = meta[ep]
        ac = m['action_counts']
        print(f"{ep:<7} {ac.get('extend',0):>7} {ac.get('retire',0):>7} {ac.get('add',0):>7} "
              f"{m['weak_edges']:>11} {m['low_sample_states']:>11} {m['high_boost_states']:>11} "
              f"{m['status']:<10}")


def ratio_analysis(parts: list[RunPart]) -> None:
    by_type: dict[str, list[RunPart]] = defaultdict(list)
    for p in parts:
        by_type[p.part_type].append(p)

    print('\n─── Efficiency ratios (bins per ns) by part type ───')
    for ptype, ps in sorted(by_type.items()):
        total_ns = sum(p.ns for p in ps)
        total_bins = sum(p.new_bins for p in ps)
        eff = total_bins / total_ns if total_ns > 0 else 0.0
        print(f'  {ptype:<16}: {len(ps):>3} parts, {total_ns:>8.2f} ns, {total_bins:>7} new bins, {eff:>8.1f} bins/ns')

    # Baseline vs topup comparison (adaptive epochs only)
    b_parts = [p for p in parts if p.part_type == 'baseline']
    t_parts = [p for p in parts if p.part_type == 'topup']
    if b_parts and t_parts:
        b_eff = sum(p.new_bins for p in b_parts) / max(sum(p.ns for p in b_parts), 1e-9)
        t_eff = sum(p.new_bins for p in t_parts) / max(sum(p.ns for p in t_parts), 1e-9)
        ratio = t_eff / b_eff if b_eff > 0 else float('nan')
        print(f'\n  Topup/baseline efficiency ratio: {ratio:.2f}  (>1 = topups more efficient than baselines)')

    # Final vs adaptive epochs
    f_parts = [p for p in parts if p.part_type in ('final_baseline', 'final_topup')]
    a_parts = [p for p in parts if p.part_type in ('baseline', 'topup', 'flat')]
    if f_parts and a_parts:
        f_eff = sum(p.new_bins for p in f_parts) / max(sum(p.ns for p in f_parts), 1e-9)
        a_eff = sum(p.new_bins for p in a_parts) / max(sum(p.ns for p in a_parts), 1e-9)
        ratio = f_eff / a_eff if a_eff > 0 else float('nan')
        print(f'  Final/adaptive efficiency ratio: {ratio:.2f}  (<1 = adaptive added novel coverage; >1 = final explored more new space)')


# ─── Plotting ────────────────────────────────────────────────────────────────

COLORS = {
    'flat': '#888888',
    'baseline': '#2196F3',
    'topup': '#FF9800',
    'final_baseline': '#4CAF50',
    'final_topup': '#8BC34A',
}
TYPE_ORDER = ['flat', 'baseline', 'topup', 'final_baseline', 'final_topup']


def plot_all(parts: list[RunPart], meta: dict, out_dir: Path, bins: int) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import Patch
    except ImportError:
        print('[plots] matplotlib not available, skipping plots')
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [p.label for p in parts]
    x = np.arange(len(parts))
    colors = [COLORS.get(p.part_type, '#aaaaaa') for p in parts]

    fig = plt.figure(figsize=(22, 20))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── (0,0) Budget allocation: ns per part stacked by type ─────────────────
    ax = fig.add_subplot(gs[0, 0])
    ns_arr = np.array([p.ns for p in parts])
    ax.bar(x, ns_arr, color=colors)
    ax.set_title('ns per run part')
    ax.set_ylabel('ns')
    ax.set_xticks(x[::max(1, len(parts)//12)])
    ax.set_xticklabels([labels[i].split('/')[-1] for i in x[::max(1, len(parts)//12)]], rotation=45, ha='right', fontsize=7)
    legend_patches = [Patch(color=COLORS[t], label=t) for t in TYPE_ORDER if any(p.part_type == t for p in parts)]
    ax.legend(handles=legend_patches, fontsize=7)

    # ── (0,1) Cumulative coverage curve ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    cum_ns = np.cumsum(ns_arr)
    cum_bins = np.array([p.cumulative_bins for p in parts])
    ax.plot(cum_ns, cum_bins, 'k-', linewidth=2)
    for ptype, col in COLORS.items():
        mask = [i for i, p in enumerate(parts) if p.part_type == ptype]
        if mask:
            ax.scatter(cum_ns[mask], cum_bins[mask], c=col, s=30, zorder=5, label=ptype)
    ax.set_title(f'Cumulative 2D coverage ({bins}×{bins} grid)')
    ax.set_xlabel('Cumulative ns')
    ax.set_ylabel('Unique CV bins')
    ax.legend(fontsize=7)

    # ── (0,2) Coverage efficiency: new bins per ns ────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    eff = np.array([p.new_bins / p.ns if p.ns > 0 else 0.0 for p in parts])
    ax.bar(x, eff, color=colors)
    ax.set_title('Coverage efficiency (new bins / ns)')
    ax.set_ylabel('bins / ns')
    ax.set_xticks(x[::max(1, len(parts)//12)])
    ax.set_xticklabels([labels[i].split('/')[-1] for i in x[::max(1, len(parts)//12)]], rotation=45, ha='right', fontsize=7)

    # ── (1,0) Exchange acceptance ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    exc = np.array([p.exchange_acceptance for p in parts])
    valid = np.isfinite(exc)
    if valid.any():
        ax.bar(x[valid], exc[valid], color=np.array(colors)[valid])
        ax.axhline(0.08, color='red', linestyle='--', linewidth=1, label='min threshold (0.08)')
        ax.axhline(0.25, color='green', linestyle='--', linewidth=1, label='target (0.25)')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
    ax.set_title('Exchange acceptance per run part')
    ax.set_ylabel('Acceptance fraction')
    ax.set_xticks(x[::max(1, len(parts)//12)])
    ax.set_xticklabels([labels[i].split('/')[-1] for i in x[::max(1, len(parts)//12)]], rotation=45, ha='right', fontsize=7)

    # ── (1,1) Weak edges per epoch ────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    if meta:
        eps = sorted(meta)
        we = [meta[e]['weak_edges'] for e in eps]
        ls = [meta[e]['low_sample_states'] for e in eps]
        ax.plot(eps, we, 'o-', color='#E53935', label='weak edges')
        ax.plot(eps, ls, 's--', color='#FB8C00', label='low-sample states')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
    ax.set_title('Convergence gate: weak edges & low-sample states')

    # ── (1,2) Adaptive actions per epoch ──────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    if meta:
        eps = sorted(meta)
        actions = ['extend', 'retire', 'add', 'split']
        action_colors = ['#1565C0', '#B71C1C', '#2E7D32', '#6A1B9A']
        bottom = np.zeros(len(eps))
        for act, col in zip(actions, action_colors):
            vals = np.array([meta[e]['action_counts'].get(act, 0) for e in eps], dtype=float)
            if vals.sum() > 0:
                ax.bar(eps, vals, bottom=bottom, color=col, label=act)
                bottom += vals
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Action count')
        ax.legend(fontsize=7)
    ax.set_title('Adaptive actions per epoch')

    # ── (2,0) Per-window sample count heatmap ────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    mat, all_windows, part_labels = per_window_matrix(parts)
    if mat.size > 0:
        im = ax.imshow(mat, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_yticks(range(len(all_windows)))
        ax.set_yticklabels([str(w) for w in all_windows], fontsize=6)
        ax.set_xticks(range(0, len(parts), max(1, len(parts)//10)))
        ax.set_xticklabels([part_labels[i].split('/')[-1] for i in range(0, len(parts), max(1, len(parts)//10))],
                           rotation=45, ha='right', fontsize=6)
        plt.colorbar(im, ax=ax, shrink=0.8).set_label('Samples', fontsize=7)
    ax.set_title('Per-window sample counts')
    ax.set_xlabel('Run part')
    ax.set_ylabel('Window ID')

    # ── (2,1) GaMD boost std per window per part ─────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    bmat, bwin = per_window_boost(parts)
    if bmat.size > 0:
        im = ax.imshow(bmat, aspect='auto', cmap='plasma', interpolation='nearest')
        ax.set_yticks(range(len(bwin)))
        ax.set_yticklabels([str(w) for w in bwin], fontsize=6)
        ax.set_xticks(range(0, len(parts), max(1, len(parts)//10)))
        ax.set_xticklabels([labels[i].split('/')[-1] for i in range(0, len(parts), max(1, len(parts)//10))],
                           rotation=45, ha='right', fontsize=6)
        plt.colorbar(im, ax=ax, shrink=0.8).set_label('Boost std (kcal/mol)', fontsize=7)
    ax.set_title('GaMD boost σ per window (kcal/mol)')
    ax.set_xlabel('Run part')
    ax.set_ylabel('Window ID')

    # ── (2,2) ns budget breakdown pie ────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    by_type: dict[str, float] = defaultdict(float)
    for p in parts:
        by_type[p.part_type] += p.ns
    type_labels = [t for t in TYPE_ORDER if by_type.get(t, 0) > 0]
    type_ns = [by_type[t] for t in type_labels]
    type_cols = [COLORS[t] for t in type_labels]
    wedges, texts, autotexts = ax.pie(type_ns, labels=type_labels, colors=type_cols,
                                       autopct='%1.1f%%', startangle=90)
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title('ns budget by part type')

    fig.suptitle('GAREUS Adaptive Strategy Audit', fontsize=14, fontweight='bold', y=0.98)
    out_path = out_dir / 'adaptive_audit.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plots] saved → {out_path}')

    # 2nd figure: coverage heatmap (cv1 × cv2 per part type)
    _plot_coverage_heatmaps(parts, bins, out_dir)


def _plot_coverage_heatmaps(parts: list[RunPart], bins: int, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    cv1_min, cv1_max = 0.0, 1.0
    cv2_min, cv2_max = -1.0, 1.0

    by_type: dict[str, list[RunPart]] = defaultdict(list)
    for p in parts:
        by_type[p.part_type].append(p)

    types_with_data = [t for t in TYPE_ORDER if by_type.get(t) and any(p.loaded for p in by_type[t])]
    if not types_with_data:
        return

    fig, axes = plt.subplots(1, len(types_with_data), figsize=(5 * len(types_with_data), 4), squeeze=False)
    vmax = None
    grids = []
    for t in types_with_data:
        g = np.zeros((bins, bins), dtype=np.int32)
        for p in by_type[t]:
            if not p.loaded:
                continue
            b1 = np.clip(((p.cv1 - cv1_min) / (cv1_max - cv1_min) * bins).astype(np.int32), 0, bins - 1)
            cv2 = p.cv2 if p.cv2 is not None else np.zeros(len(b1))
            b2 = np.clip(((cv2 - cv2_min) / (cv2_max - cv2_min) * bins).astype(np.int32), 0, bins - 1)
            np.add.at(g, (b1, b2), 1)
        grids.append(g)
        if vmax is None or g.max() > vmax:
            vmax = max(int(g.max()), 1)

    for i, (t, g) in enumerate(zip(types_with_data, grids)):
        ax = axes[0][i]
        im = ax.imshow(np.log1p(g).T, origin='lower', cmap='hot', aspect='auto',
                       extent=[cv1_min, cv1_max, cv2_min, cv2_max])
        ax.set_title(f'{t}\n(log1p counts)', fontsize=9)
        ax.set_xlabel('CV1 (contact fraction)')
        ax.set_ylabel('CV2 (Rama angle)')
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('CV coverage heatmaps by part type', fontsize=12)
    out_path = out_dir / 'adaptive_audit_coverage_heatmaps.png'
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plots] coverage heatmaps → {out_path}')


# ─── JSON report ─────────────────────────────────────────────────────────────

def write_json_report(parts: list[RunPart], meta: dict, out_dir: Path) -> None:
    by_type: dict[str, dict] = defaultdict(lambda: {'n_parts': 0, 'total_ns': 0.0, 'total_new_bins': 0, 'total_samples': 0})
    for p in parts:
        d = by_type[p.part_type]
        d['n_parts'] += 1
        d['total_ns'] += p.ns
        d['total_new_bins'] += p.new_bins
        d['total_samples'] += p.n_samples

    for t, d in by_type.items():
        d['efficiency_bins_per_ns'] = d['total_new_bins'] / d['total_ns'] if d['total_ns'] > 0 else 0.0

    report = {
        'n_run_parts': len(parts),
        'total_ns': sum(p.ns for p in parts),
        'final_unique_bins': max((p.cumulative_bins for p in parts), default=0),
        'by_type': dict(by_type),
        'epoch_meta': meta,
        'parts': [
            {
                'label': p.label,
                'epoch': p.epoch,
                'type': p.part_type,
                'n_samples': p.n_samples,
                'ns': round(p.ns, 4),
                'new_bins': p.new_bins,
                'cumulative_bins': p.cumulative_bins,
                'efficiency': round(p.new_bins / p.ns, 2) if p.ns > 0 else 0.0,
                'exchange_acceptance': None if not np.isfinite(p.exchange_acceptance) else round(p.exchange_acceptance, 4),
            }
            for p in parts
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'adaptive_audit.json'
    out_path.write_text(json.dumps(report, indent=2))
    print(f'  [report] → {out_path}')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('adaptive_dir', help='Path to adaptive_production/ directory')
    ap.add_argument('--out', default='adaptive_audit', help='Output directory (default: adaptive_audit)')
    ap.add_argument('--bins', type=int, default=30, help='CV grid bins per axis (default: 30)')
    ap.add_argument('--workers', type=int, default=8, help='Parallel DuckDB workers (default: 8)')
    ap.add_argument('--no-plots', action='store_true', help='Skip matplotlib plots')
    args = ap.parse_args()

    adaptive_dir = Path(args.adaptive_dir).resolve()
    if not adaptive_dir.exists():
        sys.exit(f'ERROR: {adaptive_dir} not found')

    out_dir = Path(args.out).resolve()

    print(f'Auditing: {adaptive_dir}')

    print('[1] Discovering run parts...')
    parts = discover_run_parts(adaptive_dir)
    print(f'  Found {len(parts)} run parts')

    print('[2] Loading samples...')
    parts = load_all_parts(parts, n_workers=args.workers)
    loaded = sum(1 for p in parts if p.loaded)
    total_samples = sum(p.n_samples for p in parts)
    print(f'  Loaded {loaded}/{len(parts)} parts, {total_samples:,} total samples')

    print('[3] Computing 2D CV coverage...')
    compute_coverage(parts, bins=args.bins)

    print('[4] Loading exchange data...')
    load_all_exchange(parts)

    print('[5] Loading epoch metadata...')
    meta = load_epoch_metadata(adaptive_dir)

    print('\n═══ Run-part summary ═══')
    print_summary_table(parts)
    print_epoch_meta_table(meta)
    ratio_analysis(parts)

    print('\n[6] Writing JSON report...')
    write_json_report(parts, meta, out_dir)

    if not args.no_plots:
        print('[7] Plotting...')
        plot_all(parts, meta, out_dir, bins=args.bins)

    print('\nDone.')


if __name__ == '__main__':
    main()
