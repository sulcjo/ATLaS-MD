#!/usr/bin/env python3
"""Ramachandran convergence audit — per-residue deep focus.

Metrics per residue, cumulative across run-parts:
  - JS divergence        (raw prefix vs MBAR reference)
  - Overlap coefficient  sum(min(P_prefix, P_ref))  — missing-basin indicator
  - Missing mass         1 - overlap
  - PMF RMSE             in low-FES bins (F_ref < 5 kBT)
  - Basin populations    alpha_R / beta / PPII / alpha_L / other
                         (Gly: wider alpha_L allowed; Pro: phi restricted)

Figures:
  1. 2D Ramachandran evolution   (residues × timepoints + reference)
  2. Per-residue JS convergence lines
  3. Per-residue overlap convergence lines
  4. Difference maps             P_prefix_final − P_ref  per residue
  5. Basin population heatmap    basin × run-part × residue
  6. Per-residue summary barplot (JS, overlap, missing_mass, RMSE at final)

Reuses _feature_cache from audit_adaptive_convergence.py if present.

Usage:
    python audit_rama.py RUNS/run/adaptive_production/ \\
        --reference-dir ANALYSIS_OUT_DIR \\
        [--cache-dir PATH] [--out PATH] [--stride N] [--workers N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from scipy.spatial.distance import jensenshannon

KB_KCAL = 0.001987204
T_K = 300.0
KBT = KB_KCAL * T_K
TIMESTEP_FS = 4.0
SAVE_STEPS = 50
NS_PER_SAMPLE = TIMESTEP_FS * SAVE_STEPS / 1e6
RAMA_BINS = 72  # 5° bins, matches analyze_gareus_mbar reference

# ── Ramachandran basin definitions (degrees) ─────────────────────────────────
# General residues (Lovell 2003 / Richardson lab, broadened for MD)
BASINS_GENERAL = {
    'alpha_R': dict(phi=(-160, 0), psi=(-80, 50)),
    'beta':    [dict(phi=(-180, -60), psi=(100, 180)),
                dict(phi=(-180, -60), psi=(-180, -140))],
    'ppii':    dict(phi=(-90, -40), psi=(110, 180)),
    'alpha_L': dict(phi=(20, 140), psi=(-60, 120)),
}
# Gly: positive phi allowed in alpha_L region (type II' turns)
BASINS_GLY = dict(BASINS_GENERAL)
BASINS_GLY['alpha_L'] = dict(phi=(-180, 180), psi=(-180, 180))  # placeholder; wider
BASINS_GLY['alpha_L_pos'] = dict(phi=(20, 160), psi=(-120, 130))

BASIN_ORDER = ['alpha_R', 'beta', 'ppii', 'alpha_L', 'other']
BASIN_COLORS = {
    'alpha_R': '#E53935', 'beta': '#1E88E5', 'ppii': '#43A047',
    'alpha_L': '#FB8C00', 'other': '#9E9E9E',
}


def _in_box(phi_deg: np.ndarray, psi_deg: np.ndarray, box: dict) -> np.ndarray:
    phi_lo, phi_hi = box['phi']
    psi_lo, psi_hi = box['psi']
    mask = ((phi_deg >= phi_lo) & (phi_deg <= phi_hi) &
            (psi_deg >= psi_lo) & (psi_deg <= psi_hi))
    return mask


def classify_basin(phi_deg: np.ndarray, psi_deg: np.ndarray,
                   is_gly: bool = False) -> np.ndarray:
    """Returns int array: 0=alpha_R 1=beta 2=ppii 3=alpha_L 4=other."""
    n = len(phi_deg)
    result = np.full(n, 4, dtype=np.int8)  # other

    basins = BASINS_GLY if is_gly else BASINS_GENERAL
    for bi, basin in enumerate(['alpha_R', 'beta', 'ppii', 'alpha_L']):
        defn = basins.get(basin)
        if defn is None:
            continue
        if isinstance(defn, list):
            mask = np.zeros(n, dtype=bool)
            for box in defn:
                mask |= _in_box(phi_deg, psi_deg, box)
        else:
            mask = _in_box(phi_deg, psi_deg, defn)
        result[mask & (result == 4)] = bi  # first matching basin wins

    return result


# ── Run-part discovery ────────────────────────────────────────────────────────

def _topup_key(name: str) -> tuple[int, int]:
    parts = name.split('_')
    try:
        return int(parts[1]), int(parts[2])
    except (IndexError, ValueError):
        return (0, 0)


def discover_run_parts(adaptive_dir: Path) -> list[dict]:
    parts: list[dict] = []
    ep0 = adaptive_dir / 'epoch_000'
    if (ep0 / 'replica_trajectories').exists():
        parts.append(dict(label='ep000/flat', epoch=0, part_type='flat',
                          traj_dir=ep0 / 'replica_trajectories', epoch_dir=ep0))

    for ep_dir in sorted(adaptive_dir.glob('epoch_???')):
        if ep_dir.name == 'epoch_000':
            continue
        ep_num = int(ep_dir.name.split('_')[1])
        subs = ['baseline'] + sorted(
            [d.name for d in ep_dir.iterdir() if d.is_dir() and d.name.startswith('topup_')],
            key=_topup_key)
        for sub in subs:
            traj_d = ep_dir / sub / 'replica_trajectories'
            if traj_d.exists() and any(traj_d.glob('*.xtc')):
                ptype = 'baseline' if sub == 'baseline' else 'topup'
                parts.append(dict(label=f'ep{ep_num:03d}/{sub}', epoch=ep_num,
                                  part_type=ptype, traj_dir=traj_d,
                                  epoch_dir=ep_dir))

    final_dir = adaptive_dir / 'final'
    if final_dir.exists():
        subs = ['baseline'] + sorted(
            [d.name for d in final_dir.iterdir() if d.is_dir() and d.name.startswith('topup_')],
            key=_topup_key)
        for sub in subs:
            traj_d = final_dir / sub / 'replica_trajectories'
            if traj_d.exists() and any(traj_d.glob('*.xtc')):
                ptype = 'final_baseline' if sub == 'baseline' else 'final_topup'
                parts.append(dict(label=f'final/{sub}', epoch=-1, part_type=ptype,
                                  traj_dir=traj_d, epoch_dir=final_dir))
    return parts


# ── Residue identification ────────────────────────────────────────────────────

def get_rama_residues(topology_pdb: Path) -> list[dict]:
    """Return list of residues with both phi AND psi (can form 2D Rama plot)."""
    try:
        import mdtraj as md
    except ImportError:
        return []
    t = md.load(str(topology_pdb))
    prot = t.atom_slice(t.topology.select('protein'))
    res_list = list(prot.topology.residues)
    # phi at residues [1..n], psi at residues [0..n-1]
    # Both phi and psi: residues [1..n-1] (0-indexed)
    # phi_idx: index into phi array (0-based) = residue_index - 1
    # psi_idx: index into psi array (0-based) = residue_index
    result = []
    sequence = [r.name for r in res_list]
    for ri, res in enumerate(res_list[1:-1]):  # skip N-term (no phi) and C-term (no psi)
        actual_ri = ri + 1  # position in full residue list
        phi_idx = actual_ri - 1   # phi array: starts at res index 1
        psi_idx = actual_ri       # psi array: starts at res index 0
        next_res = sequence[actual_ri + 1] if actual_ri + 1 < len(sequence) else ''
        result.append(dict(
            name=f'{res.name}{res.resSeq}',
            resname=res.name,
            phi_idx=phi_idx,
            psi_idx=psi_idx,
            is_gly=(res.name == 'GLY'),
            is_pro=(res.name == 'PRO'),
            is_prepro=(next_res == 'PRO'),
        ))
    return result


# ── Feature cache loading ─────────────────────────────────────────────────────

def _sincos(a: np.ndarray) -> np.ndarray:
    return np.column_stack([np.sin(a), np.cos(a)])


def _extract_features_traj(traj_dir: Path, topology_pdb: Path,
                            stride: int = 10) -> Optional[dict]:
    try:
        import mdtraj as md
    except ImportError:
        return None
    xtcs = sorted(traj_dir.glob('*.xtc'))
    if not xtcs:
        return None
    phi_chunks, psi_chunks, chi1_chunks = [], [], []
    for xtc in xtcs:
        try:
            t = md.load(str(xtc), top=str(topology_pdb), stride=stride)
            prot = t.atom_slice(t.topology.select('protein'))
            _, phi = md.compute_phi(prot)
            _, psi = md.compute_psi(prot)
            _, chi1 = md.compute_chi1(prot)
            if phi.shape[0] == 0:
                continue
            phi_chunks.append(phi.astype(np.float32))
            psi_chunks.append(psi.astype(np.float32))
            chi1_chunks.append(chi1.astype(np.float32))
        except Exception:
            continue
    if not phi_chunks:
        return None
    phi_all = np.vstack(phi_chunks)
    psi_all = np.vstack(psi_chunks)
    chi1_all = np.vstack(chi1_chunks)
    feat_all = np.hstack([_sincos(phi_all), _sincos(psi_all)]).astype(np.float32)
    return dict(features=feat_all, phi=phi_all, psi=psi_all, chi1=chi1_all,
                n_frames=len(phi_all))


def _load_or_extract(part: dict, cache_dir: Path, topology_pdb: Path,
                     stride: int) -> Optional[dict]:
    safe_label = part['label'].replace('/', '_')
    cache_path = cache_dir / f'{safe_label}.npz'
    if cache_path.exists():
        try:
            d = np.load(cache_path, allow_pickle=False)
            return {k: d[k] for k in d.files}
        except Exception:
            pass
    result = _extract_features_traj(part['traj_dir'], topology_pdb, stride)
    if result is None:
        return None
    np.savez_compressed(str(cache_path), **result)
    return result


def load_all_features(parts: list[dict], topology_pdb: Path, cache_dir: Path,
                      stride: int = 10, workers: int = 8) -> list[Optional[dict]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def _task(p: dict) -> Optional[dict]:
        return _load_or_extract(p, cache_dir, topology_pdb, stride)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_task, parts))

    loaded = sum(1 for r in results if r)
    total_frames = sum(r['n_frames'] for r in results if r)
    print(f'  [features] {loaded}/{len(parts)} parts, {total_frames:,} frames '
          f'(stride={stride}) in {time.time()-t0:.1f}s')
    return results


# ── Reference loading ─────────────────────────────────────────────────────────

def load_rama_reference(reference_dir: Path) -> dict:
    """Returns dict: residue_label -> {pmf, prob, counts, phi_edges, psi_edges}."""
    rama_dir = reference_dir / 'extra_observable_pmfs' / 'ramachandran_2d_fes'
    if not rama_dir.exists():
        return {}
    ref = {}
    for npz in sorted(rama_dir.glob('*_2d_fes.npz')):
        name = npz.stem.replace('_2d_fes', '')
        d = np.load(npz, allow_pickle=False)
        ref[name] = dict(pmf=d['pmf_kcal_mol'], prob=d['probability'],
                         counts=d['counts'], phi_edges=d['phi_edges_deg'],
                         psi_edges=d['psi_edges_deg'])
    return ref


# ── Metric computation ────────────────────────────────────────────────────────

def _hist2d(phi_deg: np.ndarray, psi_deg: np.ndarray,
            bins: int = RAMA_BINS) -> np.ndarray:
    edges = np.linspace(-180, 180, bins + 1)
    h, _, _ = np.histogram2d(phi_deg, psi_deg, bins=[edges, edges])
    return h.astype(np.int64)


def _normalize(hist: np.ndarray, pseudo: float = 0.5) -> np.ndarray:
    p = hist.astype(np.float64) + pseudo
    return p / p.sum()


def _js(p: np.ndarray, q: np.ndarray) -> float:
    return float(jensenshannon(p.ravel(), q.ravel()) ** 2)


def _overlap(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.minimum(p.ravel(), q.ravel())))


def _pmf_rmse(hist_prefix: np.ndarray, pmf_ref: np.ndarray,
              fes_cutoff_kbt: float = 5.0) -> float:
    p = _normalize(hist_prefix)
    pmf_p = -KBT * np.log(p)
    pmf_p -= pmf_p.min()
    ref = pmf_ref.copy()
    mask = (ref <= fes_cutoff_kbt * KBT * 1000) & np.isfinite(pmf_p) & np.isfinite(ref)
    if mask.sum() < 4:
        return float('nan')
    r = ref[mask]; pr = pmf_p[mask]
    r -= r.min(); pr -= pr.min()
    return float(np.sqrt(np.mean((pr - r) ** 2)))


def compute_convergence(parts: list[dict],
                        all_features: list[Optional[dict]],
                        rama_residues: list[dict],
                        rama_ref: dict) -> dict:
    """Return convergence arrays keyed by metric and residue."""
    n_parts = len(parts)
    n_res = len(rama_residues)
    ref_keys = sorted(rama_ref.keys())

    # Cumulative histogram per residue
    cum_hist = np.zeros((n_res, RAMA_BINS, RAMA_BINS), dtype=np.int64)
    # Per-part cumulative histograms (snapshots) for evolution plots
    snap_interval = max(1, n_parts // 5)
    snap_indices = sorted(set([0, snap_interval, 2*snap_interval,
                                3*snap_interval, 4*snap_interval, n_parts-1]))

    # Output arrays
    js_arr       = np.full((n_parts, n_res), np.nan)
    overlap_arr  = np.full((n_parts, n_res), np.nan)
    missing_arr  = np.full((n_parts, n_res), np.nan)
    rmse_arr     = np.full((n_parts, n_res), np.nan)
    basin_arr    = np.zeros((n_parts, n_res, len(BASIN_ORDER)), dtype=np.float32)
    cum_ns       = np.zeros(n_parts)
    cum_snaps: dict[int, np.ndarray] = {}  # snap_idx -> (n_res, BINS, BINS)

    for pi, (part, feats) in enumerate(zip(parts, all_features)):
        if pi > 0:
            cum_ns[pi] = cum_ns[pi-1]
        if feats is None:
            if pi in snap_indices:
                cum_snaps[pi] = cum_hist.copy()
            continue

        phi_rad = feats['phi']  # (N, n_phi_res)
        psi_rad = feats['psi']
        n_frames = phi_rad.shape[0]
        cum_ns[pi] += n_frames * NS_PER_SAMPLE * 10  # stride correction

        for ri, rinfo in enumerate(rama_residues):
            phi_col = rinfo['phi_idx']
            psi_col = rinfo['psi_idx']
            if phi_col >= phi_rad.shape[1] or psi_col >= psi_rad.shape[1]:
                continue
            phi_deg = np.degrees(phi_rad[:, phi_col])
            psi_deg = np.degrees(psi_rad[:, psi_col])

            # Update cumulative histogram
            cum_hist[ri] += _hist2d(phi_deg, psi_deg)

            # Basin populations for this part
            labels = classify_basin(phi_deg, psi_deg, is_gly=rinfo['is_gly'])
            for bi in range(len(BASIN_ORDER)):
                basin_arr[pi, ri, bi] = float((labels == bi).sum()) / max(1, n_frames)

        # Metrics vs reference
        for ri, (rinfo, rkey) in enumerate(zip(rama_residues, ref_keys)):
            if rkey not in rama_ref or cum_hist[ri].sum() < 10:
                continue
            ref_d = rama_ref[rkey]
            p_cum = _normalize(cum_hist[ri])
            p_ref = ref_d['prob'].copy()
            if p_ref.sum() == 0:
                continue
            p_ref = (p_ref + 0.5) / (p_ref + 0.5).sum()

            js_arr[pi, ri] = _js(p_cum, p_ref)
            ov = _overlap(p_cum, p_ref)
            overlap_arr[pi, ri] = ov
            missing_arr[pi, ri] = 1.0 - ov
            rmse_arr[pi, ri] = _pmf_rmse(cum_hist[ri], ref_d['pmf'])

        if pi in snap_indices:
            cum_snaps[pi] = cum_hist.copy()

    # Last snapshot must always be present
    if n_parts - 1 not in cum_snaps:
        cum_snaps[n_parts - 1] = cum_hist.copy()

    return dict(js=js_arr, overlap=overlap_arr, missing=missing_arr,
                rmse=rmse_arr, basin=basin_arr, cum_ns=cum_ns,
                snap_indices=sorted(cum_snaps.keys()),
                cum_snaps=cum_snaps)


# ── Plotting ──────────────────────────────────────────────────────────────────

PTYPE_COLORS = {
    'flat': '#888888', 'baseline': '#2196F3', 'topup': '#FF9800',
    'final_baseline': '#4CAF50', 'final_topup': '#8BC34A',
}


def _fig1_evolution(rama_residues: list[dict], snap_indices: list[int],
                    cum_snaps: dict, cum_ns: np.ndarray,
                    rama_ref: dict, ref_keys: list[str],
                    out_dir: Path) -> None:
    """2D Ramachandran density evolution: residues (rows) × timepoints+ref (cols)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    n_res = len(rama_residues)
    n_snap = len(snap_indices)
    n_cols = n_snap + 1  # +1 for reference

    fig, axes = plt.subplots(n_res, n_cols,
                             figsize=(n_cols * 2.5, n_res * 2.5),
                             squeeze=False)
    edges = np.linspace(-180, 180, RAMA_BINS + 1)
    phi_centers = 0.5 * (edges[:-1] + edges[1:])

    for ri, rinfo in enumerate(rama_residues):
        for ci, si in enumerate(snap_indices):
            ax = axes[ri, ci]
            hist = cum_snaps[si][ri].astype(float)
            if hist.sum() > 0:
                p = hist / hist.sum()
                p[p == 0] = np.nan
                im = ax.pcolormesh(edges, edges, p.T,
                                   cmap='Blues', norm=LogNorm(vmin=1e-4, vmax=p[np.isfinite(p)].max()))
            ax.axhline(0, color='gray', lw=0.4); ax.axvline(0, color='gray', lw=0.4)
            ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ns_label = f'{cum_ns[si]:.0f} ns'
                ax.set_title(ns_label, fontsize=8)
            if ci == 0:
                ax.set_ylabel(rinfo['name'], fontsize=8, rotation=0,
                              labelpad=45, va='center')

        # Reference column
        ax = axes[ri, n_snap]
        rkey = ref_keys[ri] if ri < len(ref_keys) else None
        if rkey and rkey in rama_ref:
            p_ref = rama_ref[rkey]['prob']
            p_ref = p_ref.copy().astype(float)
            p_ref[p_ref == 0] = np.nan
            ax.pcolormesh(edges, edges, p_ref.T,
                          cmap='Reds', norm=LogNorm(vmin=1e-4, vmax=np.nanmax(p_ref)))
        ax.axhline(0, color='gray', lw=0.4); ax.axvline(0, color='gray', lw=0.4)
        ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title('MBAR ref', fontsize=8, color='darkred')

    # Axis labels for bottom row
    for ci in range(n_cols):
        axes[-1, ci].set_xlabel('φ', fontsize=7)
    for ri in range(n_res):
        axes[ri, 0].set_yticks([-120, 0, 120])
        axes[ri, 0].set_yticklabels(['-120', '0', '120'], fontsize=6)
        axes[ri, 0].set_xticks([-120, 0, 120])
        axes[ri, 0].set_xticklabels(['-120', '0', '120'], fontsize=6)

    fig.suptitle('Ramachandran density evolution (blue=raw prefix, red=MBAR reference)',
                 fontsize=11, y=1.001)
    fig.tight_layout()
    p = out_dir / 'rama_fig1_evolution.png'
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  [fig1] → {p}')


def _fig2_js_lines(rama_residues: list[dict], conv: dict,
                   parts: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cum_ns = conv['cum_ns']
    js = conv['js']
    n_res = len(rama_residues)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    cmap = plt.get_cmap('tab10', max(n_res, 1))
    for ri, rinfo in enumerate(rama_residues):
        valid = np.isfinite(js[:, ri])
        if not valid.any():
            continue
        ax1.plot(cum_ns[valid], js[valid, ri], '-o', markersize=2,
                 color=cmap(ri), label=rinfo['name'], alpha=0.8)

    # Mean + max band
    js_mean = np.nanmean(js, axis=1)
    js_max  = np.nanmax(js, axis=1)
    valid_m = np.isfinite(js_mean)
    if valid_m.any():
        ax1.fill_between(cum_ns[valid_m], js_mean[valid_m], js_max[valid_m],
                         alpha=0.15, color='black', label='max band')
        ax1.plot(cum_ns[valid_m], js_mean[valid_m], 'k-', linewidth=2, label='mean')
    ax1.axhline(0.05, color='orange', ls='--', lw=1, label='JS=0.05')
    ax1.axhline(0.01, color='green',  ls='--', lw=1, label='JS=0.01')
    ax1.set_xlabel('Cumulative ns'); ax1.set_ylabel('JS divergence (raw prefix vs MBAR ref)')
    ax1.set_title('Per-residue JS divergence convergence')
    ax1.legend(fontsize=7, ncol=2)
    ax1.set_ylim(bottom=0)

    # Missing mass
    missing = conv['missing']
    for ri, rinfo in enumerate(rama_residues):
        valid = np.isfinite(missing[:, ri])
        if not valid.any():
            continue
        ax2.plot(cum_ns[valid], missing[valid, ri] * 100, '-o', markersize=2,
                 color=cmap(ri), label=rinfo['name'], alpha=0.8)
    ax2.axhline(5, color='orange', ls='--', lw=1, label='5% missing')
    ax2.axhline(2, color='green',  ls='--', lw=1, label='2% missing')
    ax2.set_xlabel('Cumulative ns'); ax2.set_ylabel('Missing reference mass (%)')
    ax2.set_title('Per-residue missing mass  (1 − overlap)')
    ax2.legend(fontsize=7, ncol=2)
    ax2.set_ylim(bottom=0)

    fig.tight_layout()
    p = out_dir / 'rama_fig2_js_missing.png'
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  [fig2] → {p}')


def _fig3_overlap_rmse(rama_residues: list[dict], conv: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cum_ns = conv['cum_ns']
    overlap = conv['overlap']
    rmse    = conv['rmse']
    n_res = len(rama_residues)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.get_cmap('tab10', max(n_res, 1))

    for ri, rinfo in enumerate(rama_residues):
        valid = np.isfinite(overlap[:, ri])
        if not valid.any():
            continue
        ax1.plot(cum_ns[valid], overlap[valid, ri], '-o', markersize=2,
                 color=cmap(ri), label=rinfo['name'], alpha=0.8)
    ax1.axhline(0.90, color='green',  ls='--', lw=1, label='overlap=0.90')
    ax1.axhline(0.85, color='orange', ls='--', lw=1, label='overlap=0.85')
    ax1.set_xlabel('Cumulative ns'); ax1.set_ylabel('Overlap coefficient')
    ax1.set_title('Per-residue overlap  Σmin(P_prefix, P_ref)')
    ax1.legend(fontsize=7, ncol=2)
    ax1.set_ylim(0, 1)

    for ri, rinfo in enumerate(rama_residues):
        valid = np.isfinite(rmse[:, ri])
        if not valid.any():
            continue
        ax2.plot(cum_ns[valid], rmse[valid, ri], '-o', markersize=2,
                 color=cmap(ri), label=rinfo['name'], alpha=0.8)
    ax2.axhline(0.5, color='green',  ls='--', lw=1, label='0.5 kcal/mol')
    ax2.axhline(1.0, color='orange', ls='--', lw=1, label='1.0 kcal/mol')
    ax2.set_xlabel('Cumulative ns'); ax2.set_ylabel('PMF RMSE (kcal/mol, low-FES bins)')
    ax2.set_title('Per-residue PMF RMSE  (raw prefix vs MBAR ref, F_ref<5kBT)')
    ax2.legend(fontsize=7, ncol=2)
    ax2.set_ylim(bottom=0)

    fig.tight_layout()
    p = out_dir / 'rama_fig3_overlap_rmse.png'
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  [fig3] → {p}')


def _fig4_difference_maps(rama_residues: list[dict], conv: dict,
                           rama_ref: dict, ref_keys: list[str],
                           out_dir: Path) -> None:
    """P_final_prefix − P_ref difference maps per residue."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_res = len(rama_residues)
    last_snap = conv['cum_snaps'][conv['snap_indices'][-1]]
    edges = np.linspace(-180, 180, RAMA_BINS + 1)

    ncols = min(4, n_res)
    nrows = (n_res + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5),
                             squeeze=False)
    axes_flat = axes.ravel()

    for ri, rinfo in enumerate(rama_residues):
        ax = axes_flat[ri]
        rkey = ref_keys[ri] if ri < len(ref_keys) else None
        if rkey and rkey in rama_ref:
            p_cum = _normalize(last_snap[ri])
            p_ref = rama_ref[rkey]['prob'].astype(float)
            if p_ref.sum() > 0:
                p_ref = (p_ref + 0.5) / (p_ref + 0.5).sum()
            diff = p_cum - p_ref
            vmax = max(abs(diff).max(), 1e-6)
            im = ax.pcolormesh(edges, edges, diff.T,
                               cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            plt.colorbar(im, ax=ax, shrink=0.8).set_label('ΔP', fontsize=7)
        ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
        ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
        ax.set_xlabel('φ', fontsize=8); ax.set_ylabel('ψ', fontsize=8)
        ax.set_title(f'{rinfo["name"]}\n(blue=over, red=under)', fontsize=8)

    for ax in axes_flat[n_res:]:
        ax.set_visible(False)

    fig.suptitle('Ramachandran difference maps: P_prefix_final − P_ref\n'
                 '(blue=oversampled, red=undersampled)', fontsize=10)
    fig.tight_layout()
    p = out_dir / 'rama_fig4_difference_maps.png'
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  [fig4] → {p}')


def _fig5_basin_heatmap(rama_residues: list[dict], parts: list[dict],
                         conv: dict, out_dir: Path) -> None:
    """Basin population stacked area per residue, one subplot each."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    basin_arr = conv['basin']  # (n_parts, n_res, n_basins)
    cum_ns    = conv['cum_ns']
    n_res = len(rama_residues)

    ncols = min(4, n_res)
    nrows = (n_res + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3),
                             squeeze=False)
    axes_flat = axes.ravel()

    for ri, rinfo in enumerate(rama_residues):
        ax = axes_flat[ri]
        # Build cumulative fraction for stacked area
        fracs = basin_arr[:, ri, :]  # (n_parts, n_basins)
        # Normalize so they sum to 1
        row_sum = fracs.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        fracs_norm = fracs / row_sum
        # Stacked area
        bottoms = np.zeros(len(parts))
        for bi, bname in enumerate(BASIN_ORDER):
            ax.fill_between(cum_ns, bottoms, bottoms + fracs_norm[:, bi],
                            alpha=0.85, color=BASIN_COLORS[bname], label=bname if ri == 0 else '')
            bottoms += fracs_norm[:, bi]
        ax.set_xlim(cum_ns[0], cum_ns[-1])
        ax.set_ylim(0, 1)
        ax.set_title(rinfo['name'], fontsize=9)
        ax.set_xlabel('ns', fontsize=7)
        ax.set_ylabel('Basin frac', fontsize=7)

    for ax in axes_flat[n_res:]:
        ax.set_visible(False)

    # Single legend
    handles = [plt.Rectangle((0,0),1,1, color=BASIN_COLORS[b]) for b in BASIN_ORDER]
    fig.legend(handles, BASIN_ORDER, loc='lower right', fontsize=8)
    fig.suptitle('Ramachandran basin populations per residue', fontsize=10)
    fig.tight_layout()
    p = out_dir / 'rama_fig5_basin_populations.png'
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  [fig5] → {p}')


def _fig6_summary_bar(rama_residues: list[dict], conv: dict, out_dir: Path) -> None:
    """Final-timepoint per-residue summary: JS, overlap, missing, RMSE."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    last = -1  # last part index
    res_names = [r['name'] for r in rama_residues]
    n = len(res_names)
    x = np.arange(n)

    js_final      = conv['js'][last]
    overlap_final = conv['overlap'][last]
    missing_final = conv['missing'][last]
    rmse_final    = conv['rmse'][last]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    kw = dict(edgecolor='white', linewidth=0.5)

    axes[0,0].bar(x, js_final, color='#E53935', **kw)
    axes[0,0].axhline(0.05, color='orange', ls='--', lw=1)
    axes[0,0].axhline(0.01, color='green',  ls='--', lw=1)
    axes[0,0].set_title('JS divergence (final prefix vs MBAR ref)')
    axes[0,0].set_ylabel('JS'); _set_res_xticks(axes[0,0], x, res_names)

    axes[0,1].bar(x, overlap_final, color='#1E88E5', **kw)
    axes[0,1].axhline(0.90, color='green',  ls='--', lw=1, label='0.90')
    axes[0,1].axhline(0.85, color='orange', ls='--', lw=1, label='0.85')
    axes[0,1].set_title('Overlap coefficient (final prefix vs MBAR ref)')
    axes[0,1].set_ylabel('Overlap'); axes[0,1].set_ylim(0, 1)
    _set_res_xticks(axes[0,1], x, res_names)

    axes[1,0].bar(x, missing_final * 100, color='#FB8C00', **kw)
    axes[1,0].axhline(5, color='orange', ls='--', lw=1, label='5%')
    axes[1,0].axhline(2, color='green',  ls='--', lw=1, label='2%')
    axes[1,0].set_title('Missing reference mass % (final)')
    axes[1,0].set_ylabel('%'); _set_res_xticks(axes[1,0], x, res_names)

    axes[1,1].bar(x, rmse_final, color='#43A047', **kw)
    axes[1,1].axhline(0.5, color='green',  ls='--', lw=1)
    axes[1,1].axhline(1.0, color='orange', ls='--', lw=1)
    axes[1,1].set_title('PMF RMSE kcal/mol (low-FES bins, final)')
    axes[1,1].set_ylabel('kcal/mol'); _set_res_xticks(axes[1,1], x, res_names)

    fig.suptitle('Final-timepoint Ramachandran convergence summary', fontsize=11)
    fig.tight_layout()
    p = out_dir / 'rama_fig6_summary.png'
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  [fig6] → {p}')


def _set_res_xticks(ax, x: np.ndarray, names: list[str]) -> None:
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)


# ── JSON report ───────────────────────────────────────────────────────────────

def write_report(parts: list[dict], rama_residues: list[dict],
                 conv: dict, out_dir: Path) -> None:
    rows = []
    for pi, part in enumerate(parts):
        row = dict(label=part['label'], part_type=part['part_type'],
                   cum_ns=round(float(conv['cum_ns'][pi]), 3))
        for ri, rinfo in enumerate(rama_residues):
            prefix = rinfo['name']
            row[f'{prefix}_js']      = _nan_round(conv['js'][pi, ri], 5)
            row[f'{prefix}_overlap'] = _nan_round(conv['overlap'][pi, ri], 4)
            row[f'{prefix}_missing'] = _nan_round(conv['missing'][pi, ri], 4)
            row[f'{prefix}_rmse']    = _nan_round(conv['rmse'][pi, ri], 4)
        rows.append(row)

    # Summary: final timepoint, aggregated
    last = len(parts) - 1
    summary = dict(
        total_ns=round(float(conv['cum_ns'][-1]), 2),
        n_residues=len(rama_residues),
        final_js_mean=_nan_round(float(np.nanmean(conv['js'][last])), 5),
        final_js_max=_nan_round(float(np.nanmax(conv['js'][last][np.isfinite(conv['js'][last])])), 5)
            if np.any(np.isfinite(conv['js'][last])) else None,
        final_overlap_mean=_nan_round(float(np.nanmean(conv['overlap'][last])), 4),
        final_missing_mean=_nan_round(float(np.nanmean(conv['missing'][last])), 4),
        final_rmse_mean=_nan_round(float(np.nanmean(conv['rmse'][last])), 4),
        residues={rinfo['name']: dict(
            js_final=_nan_round(float(conv['js'][last, ri]), 5),
            overlap_final=_nan_round(float(conv['overlap'][last, ri]), 4),
            missing_final=_nan_round(float(conv['missing'][last, ri]), 4),
            rmse_final=_nan_round(float(conv['rmse'][last, ri]), 4),
        ) for ri, rinfo in enumerate(rama_residues)},
    )

    report = dict(summary=summary, per_part=rows)
    p = out_dir / 'rama_report.json'
    p.write_text(json.dumps(report, indent=2))
    print(f'  [report] → {p}')


def _nan_round(v, decimals: int = 4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    return round(float(v), decimals)


# ── Main ──────────────────────────────────────────────────────────────────────

def find_topology(adaptive_dir: Path) -> Optional[Path]:
    run_dir = adaptive_dir.parent
    for name in ['01_solvated_start.pdb', '02_npt_equilibrated.pdb', '00_built_peptide.pdb']:
        p = run_dir / name
        if p.exists():
            return p
    for p in sorted(run_dir.rglob('*.pdb')):
        return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('adaptive_dir')
    ap.add_argument('--reference-dir', required=True,
                    help='analyze_gareus_mbar.py output dir (Ramachandran NPZs)')
    ap.add_argument('--out', default='adaptive_rama_audit')
    ap.add_argument('--cache-dir', default=None,
                    help='Feature cache dir (default: <out>/_feature_cache). '
                         'Pass convergence audit cache dir to reuse it.')
    ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()

    adaptive_dir = Path(args.adaptive_dir).resolve()
    out_dir = Path(args.out).resolve()
    ref_dir = Path(args.reference_dir).resolve()

    if args.cache_dir:
        cache_dir = Path(args.cache_dir).resolve()
    else:
        cache_dir = out_dir / '_feature_cache'

    out_dir.mkdir(parents=True, exist_ok=True)

    # Topology
    topology_pdb = find_topology(adaptive_dir)
    if topology_pdb is None:
        sys.exit('ERROR: no topology PDB found')
    print(f'Topology: {topology_pdb}')

    # Reference
    print('[1] Loading MBAR Ramachandran reference...')
    rama_ref = load_rama_reference(ref_dir)
    if not rama_ref:
        sys.exit(f'ERROR: no Ramachandran NPZs found in {ref_dir}/extra_observable_pmfs/ramachandran_2d_fes/')
    ref_keys = sorted(rama_ref.keys())
    print(f'  {len(ref_keys)} reference residues: {ref_keys}')

    # Residue identification
    print('[2] Identifying backbone residues...')
    rama_residues = get_rama_residues(topology_pdb)
    print(f'  {len(rama_residues)} residues with phi+psi: '
          f'{[r["name"] for r in rama_residues]}')

    # Align residue count to reference
    n_use = min(len(rama_residues), len(ref_keys))
    if n_use < len(rama_residues):
        print(f'  NOTE: using first {n_use} residues to match reference')
        rama_residues = rama_residues[:n_use]

    # Run-part discovery
    print('[3] Discovering run parts...')
    parts = discover_run_parts(adaptive_dir)
    print(f'  {len(parts)} run parts')

    # Feature loading
    print('[4] Loading dihedral features...')
    all_features = load_all_features(parts, topology_pdb, cache_dir,
                                     stride=args.stride, workers=args.workers)

    # Convergence metrics
    print('[5] Computing convergence metrics...')
    conv = compute_convergence(parts, all_features, rama_residues, rama_ref)

    # Summary
    last = len(parts) - 1
    print('\n══ Final-timepoint Ramachandran convergence ══')
    hdr = f"{'Residue':<12} {'JS':>8} {'Overlap':>8} {'Missing%':>10} {'RMSE':>8}"
    print(hdr); print('─' * len(hdr))
    for ri, rinfo in enumerate(rama_residues):
        js_v   = conv['js'][last, ri]
        ov_v   = conv['overlap'][last, ri]
        mis_v  = conv['missing'][last, ri]
        rmse_v = conv['rmse'][last, ri]
        js_s   = f'{js_v:.4f}'   if np.isfinite(js_v)   else '  N/A'
        ov_s   = f'{ov_v:.4f}'   if np.isfinite(ov_v)   else '  N/A'
        mis_s  = f'{mis_v*100:.2f}%' if np.isfinite(mis_v) else '  N/A'
        rmse_s = f'{rmse_v:.3f}' if np.isfinite(rmse_v) else '  N/A'
        print(f'{rinfo["name"]:<12} {js_s:>8} {ov_s:>8} {mis_s:>10} {rmse_s:>8}')
    print(f'\nMean JS: {np.nanmean(conv["js"][last]):.4f}  |  '
          f'Max JS: {np.nanmax(conv["js"][last][np.isfinite(conv["js"][last])]):.4f}  |  '
          f'Mean overlap: {np.nanmean(conv["overlap"][last]):.4f}')

    # Report
    print('\n[6] Writing report...')
    write_report(parts, rama_residues, conv, out_dir)

    # Plots
    if not args.no_plots:
        print('[7] Plotting...')
        snap_indices = conv['snap_indices']
        cum_snaps    = conv['cum_snaps']
        cum_ns       = conv['cum_ns']

        try:
            _fig1_evolution(rama_residues, snap_indices, cum_snaps, cum_ns,
                            rama_ref, ref_keys, out_dir)
        except Exception as e:
            print(f'  [fig1] FAILED: {e}')
        try:
            _fig2_js_lines(rama_residues, conv, parts, out_dir)
        except Exception as e:
            print(f'  [fig2] FAILED: {e}')
        try:
            _fig3_overlap_rmse(rama_residues, conv, out_dir)
        except Exception as e:
            print(f'  [fig3] FAILED: {e}')
        try:
            _fig4_difference_maps(rama_residues, conv, rama_ref, ref_keys, out_dir)
        except Exception as e:
            print(f'  [fig4] FAILED: {e}')
        try:
            _fig5_basin_heatmap(rama_residues, parts, conv, out_dir)
        except Exception as e:
            print(f'  [fig5] FAILED: {e}')
        try:
            _fig6_summary_bar(rama_residues, conv, out_dir)
        except Exception as e:
            print(f'  [fig6] FAILED: {e}')

    print('\nDone.')


if __name__ == '__main__':
    main()
