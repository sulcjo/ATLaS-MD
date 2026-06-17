#!/usr/bin/env python3
"""Adaptive sampling convergence audit — structural & thermodynamic lanes.

TWO LANES (kept strictly separate per GPT/research advice):
  Exploration lane  — raw histograms / structural clusters / UMAP
                      "Did we visit new conformational territory?"
  Thermodynamic lane — cumulative raw histogram RMSE/JS vs MBAR-reweighted
                       reference from analyze_gareus_mbar.py output
                       "Did free-energy estimates converge to the final run?"

Usage:
    python audit_adaptive_convergence.py RUNS/run/adaptive_production/ \\
        [--reference-dir ANALYSIS_OUT_DIR] \\
        [--out AUDIT_DIR] \\
        [--stride N] [--n-clusters K] [--workers N] [--no-plots]

    --reference-dir   path to analyze_gareus_mbar.py output (for thermodynamic lane)
    --stride          XTC stride for subsampling (default 10)
    --n-clusters      codebook size for MiniBatchKMeans (default 80)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.distance import jensenshannon

KB_KCAL = 0.001987204  # kcal/(mol·K)
T_K = 300.0
KBT = KB_KCAL * T_K
TIMESTEP_FS = 4.0
SAVE_STEPS = 50
NS_PER_SAMPLE = TIMESTEP_FS * SAVE_STEPS / 1e6


# ─── Run-part discovery (reuse audit_adaptive_strategy logic) ────────────────

def _topup_key(name: str) -> tuple[int, int]:
    parts = name.split('_')
    try:
        return int(parts[1]), int(parts[2])
    except (IndexError, ValueError):
        return (0, 0)


def discover_run_parts(adaptive_dir: Path) -> list[dict]:
    """Return ordered list of {label, epoch, part_type, traj_dir, epoch_dir}."""
    parts: list[dict] = []

    ep0 = adaptive_dir / 'epoch_000'
    if (ep0 / 'replica_trajectories').exists():
        parts.append(dict(label='ep000/flat', epoch=0, part_type='flat',
                          traj_dir=ep0 / 'replica_trajectories', epoch_dir=ep0))

    for ep_dir in sorted(adaptive_dir.glob('epoch_???')):
        if ep_dir.name == 'epoch_000':
            continue
        ep_num = int(ep_dir.name.split('_')[1])
        for sub in ['baseline'] + sorted(
            [d.name for d in ep_dir.iterdir() if d.is_dir() and d.name.startswith('topup_')],
            key=_topup_key,
        ):
            traj_d = ep_dir / sub / 'replica_trajectories'
            if traj_d.exists() and any(traj_d.glob('*.xtc')):
                ptype = 'baseline' if sub == 'baseline' else 'topup'
                parts.append(dict(label=f'ep{ep_num:03d}/{sub}', epoch=ep_num,
                                  part_type=ptype, traj_dir=traj_d,
                                  epoch_dir=ep_dir))

    final_dir = adaptive_dir / 'final'
    if final_dir.exists():
        for sub in ['baseline'] + sorted(
            [d.name for d in final_dir.iterdir() if d.is_dir() and d.name.startswith('topup_')],
            key=_topup_key,
        ):
            traj_d = final_dir / sub / 'replica_trajectories'
            if traj_d.exists() and any(traj_d.glob('*.xtc')):
                ptype = 'final_baseline' if sub == 'baseline' else 'final_topup'
                parts.append(dict(label=f'final/{sub}', epoch=-1, part_type=ptype,
                                  traj_dir=traj_d, epoch_dir=final_dir))

    return parts


# ─── Feature extraction (XTC → sin/cos dihedral features) ───────────────────

def _sincos(angles_rad: np.ndarray) -> np.ndarray:
    return np.column_stack([np.sin(angles_rad), np.cos(angles_rad)])


def _extract_features_traj(traj_dir: Path, topology_pdb: Path,
                            stride: int = 10) -> Optional[dict]:
    """Load all replica XTCs from traj_dir, return feature dict."""
    try:
        import mdtraj as md
    except ImportError:
        return None

    xtcs = sorted(traj_dir.glob('*.xtc'))
    if not xtcs:
        return None

    chunks: list[np.ndarray] = []
    phi_chunks: list[np.ndarray] = []
    psi_chunks: list[np.ndarray] = []
    chi1_chunks: list[np.ndarray] = []

    for xtc in xtcs:
        try:
            t = md.load(str(xtc), top=str(topology_pdb), stride=stride)
            prot = t.atom_slice(t.topology.select('protein'))
            _, phi = md.compute_phi(prot)
            _, psi = md.compute_psi(prot)
            _, chi1 = md.compute_chi1(prot)
            N = phi.shape[0]
            if N == 0:
                continue
            # sin/cos features: (N, 2*n_phi + 2*n_psi)
            feat = np.hstack([_sincos(phi), _sincos(psi)]).astype(np.float32)
            chunks.append(feat)
            phi_chunks.append(phi.astype(np.float32))
            psi_chunks.append(psi.astype(np.float32))
            chi1_chunks.append(chi1.astype(np.float32))
        except Exception:
            continue

    if not chunks:
        return None

    feat_all = np.vstack(chunks)
    phi_all = np.vstack(phi_chunks)
    psi_all = np.vstack(psi_chunks)
    chi1_all = np.vstack(chi1_chunks)

    return dict(features=feat_all, phi=phi_all, psi=psi_all, chi1=chi1_all,
                n_frames=len(feat_all))


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


# ─── Global codebook (PCA + KMeans + UMAP) ──────────────────────────────────

def build_codebook(all_features: list[Optional[dict]], n_clusters: int = 80,
                   n_pca: int = 10, umap_n_landmarks: int = 50_000
                   ) -> tuple[object, object, object, np.ndarray]:
    """Returns (pca, kmeans, umap_model, landmark_umap_coords)."""
    from sklearn.decomposition import IncrementalPCA
    from sklearn.cluster import MiniBatchKMeans

    valid = [f['features'] for f in all_features if f is not None]
    if not valid:
        raise RuntimeError('No feature data loaded')

    all_feat = np.vstack(valid)
    N = len(all_feat)
    print(f'  [codebook] fitting PCA({n_pca}) on {N:,} frames...')

    pca = IncrementalPCA(n_components=n_pca, batch_size=max(10 * n_pca, 10_000))
    pca.fit(all_feat)
    pca_all = pca.transform(all_feat)
    explained = np.cumsum(pca.explained_variance_ratio_)
    print(f'  [codebook] PCA variance explained: {explained[min(n_pca-1, len(explained)-1)]*100:.1f}%')

    print(f'  [codebook] fitting KMeans(k={n_clusters})...')
    km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=min(100_000, N),
                         n_init=3, random_state=42)
    km.fit(pca_all)

    umap_model = None
    umap_coords = np.zeros((min(N, umap_n_landmarks), 2))
    try:
        import umap
        idx = np.random.RandomState(42).choice(N, size=min(N, umap_n_landmarks), replace=False)
        idx.sort()
        landmarks = pca_all[idx]
        print(f'  [codebook] UMAP on {len(idx):,} landmarks...')
        umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                                random_state=42, verbose=False)
        umap_coords = umap_model.fit_transform(landmarks)
        print('  [codebook] UMAP done')
    except ImportError:
        print('  [codebook] umap not available, skipping UMAP embedding')

    return pca, km, umap_model, umap_coords


# ─── Per-part metrics ────────────────────────────────────────────────────────

@dataclass
class PartMetrics:
    label: str
    epoch: int
    part_type: str
    n_frames: int = 0
    ns: float = 0.0
    # Exploration lane
    cluster_ids: Optional[np.ndarray] = None
    new_clusters: int = 0
    cumulative_cluster_frac: float = 0.0
    phi_raw: Optional[np.ndarray] = None   # (n_res, 72, 72) - 5deg bins
    psi_raw: Optional[np.ndarray] = None
    chi1_raw: Optional[np.ndarray] = None  # (n_chi1_res, 36) - 10deg bins
    # Thermodynamic lane (filled later)
    rama_js_vs_ref: Optional[np.ndarray] = None   # per residue
    rama_rmse_vs_ref: Optional[np.ndarray] = None  # per residue
    chi1_js_vs_ref: float = float('nan')
    chi1_rmse_vs_ref: float = float('nan')


def assign_clusters(feats: dict, pca, kmeans) -> np.ndarray:
    pca_feat = pca.transform(feats['features'])
    return kmeans.predict(pca_feat).astype(np.int32)


def compute_per_part_metrics(parts: list[dict], all_features: list[Optional[dict]],
                              pca, kmeans, n_clusters: int,
                              phi_bins: int = 72) -> list[PartMetrics]:
    phi_edges = np.linspace(-np.pi, np.pi, phi_bins + 1)
    seen_clusters: set[int] = set()
    metrics: list[PartMetrics] = []

    for part, feats in zip(parts, all_features):
        m = PartMetrics(label=part['label'], epoch=part['epoch'],
                        part_type=part['part_type'])
        if feats is None:
            metrics.append(m)
            continue

        m.n_frames = feats['n_frames']
        m.ns = m.n_frames * NS_PER_SAMPLE * 10  # stride=10 correction

        # Cluster assignment
        ids = assign_clusters(feats, pca, kmeans)
        m.cluster_ids = ids
        unique = set(ids.tolist())
        new = unique - seen_clusters
        seen_clusters |= unique
        m.new_clusters = len(new)
        m.cumulative_cluster_frac = len(seen_clusters) / n_clusters

        # Raw Ramachandran histograms (per residue)
        phi_rad = feats['phi']  # (N, n_res)
        psi_rad = feats['psi']
        n_res = phi_rad.shape[1]
        rama_hist = np.zeros((n_res, phi_bins, phi_bins), dtype=np.int64)
        for ri in range(n_res):
            h, _, _ = np.histogram2d(
                np.degrees(phi_rad[:, ri]), np.degrees(psi_rad[:, ri]),
                bins=[np.linspace(-180, 180, phi_bins + 1),
                      np.linspace(-180, 180, phi_bins + 1)],
            )
            rama_hist[ri] = h.astype(np.int64)
        m.phi_raw = rama_hist

        # Chi1 histogram (per chi1 residue)
        chi1_rad = feats['chi1']  # (N, n_chi1)
        chi1_bins = 36  # 10°
        chi1_hist = np.zeros((chi1_rad.shape[1], chi1_bins), dtype=np.int64)
        chi1_edges = np.linspace(-180, 180, chi1_bins + 1)
        for ci in range(chi1_rad.shape[1]):
            h, _ = np.histogram(np.degrees(chi1_rad[:, ci]), bins=chi1_edges)
            chi1_hist[ci] = h
        m.chi1_raw = chi1_hist

        metrics.append(m)

    return metrics


# ─── Thermodynamic lane: RMSE/JS vs MBAR reference ──────────────────────────

def _pmf_from_hist(hist: np.ndarray, kbt: float = KBT,
                   pseudocount: float = 0.5) -> np.ndarray:
    """Raw PMF in kcal/mol from count histogram. Pseudocount prevents -inf."""
    p = (hist.astype(np.float64) + pseudocount)
    p /= p.sum()
    pmf = -kbt * np.log(p)
    pmf -= pmf.min()
    return pmf


def _js_2d(h1: np.ndarray, h2: np.ndarray) -> float:
    """JS divergence between two 2D count histograms (flattened)."""
    p = h1.ravel().astype(np.float64) + 0.5
    q = h2.ravel().astype(np.float64) + 0.5
    p /= p.sum(); q /= q.sum()
    return float(jensenshannon(p, q) ** 2)


def _pmf_rmse(pmf_prefix: np.ndarray, pmf_ref: np.ndarray,
              min_ref_counts: int = 5, ref_counts: Optional[np.ndarray] = None
              ) -> float:
    """RMSE of PMF values (kcal/mol) over bins with sufficient reference data."""
    if ref_counts is not None:
        mask = (ref_counts.ravel() >= min_ref_counts) & np.isfinite(pmf_prefix.ravel()) & np.isfinite(pmf_ref.ravel())
    else:
        mask = np.isfinite(pmf_prefix.ravel()) & np.isfinite(pmf_ref.ravel())
    p = pmf_prefix.ravel()[mask]
    r = pmf_ref.ravel()[mask]
    if len(p) < 4:
        return float('nan')
    # Zero-reference to minimum
    p = p - p.min()
    r = r - r.min()
    return float(np.sqrt(np.mean((p - r) ** 2)))


def load_reference_npz(reference_dir: Path) -> dict:
    """Load Ramachandran and CV PMF reference NPZs."""
    ref = {}

    # Ramachandran per residue
    rama_dir = reference_dir / 'extra_observable_pmfs' / 'ramachandran_2d_fes'
    if rama_dir.exists():
        ref['rama'] = {}
        for npz in sorted(rama_dir.glob('*_2d_fes.npz')):
            name = npz.stem.replace('_2d_fes', '')  # e.g. rama_001_TYR2
            d = np.load(npz, allow_pickle=False)
            ref['rama'][name] = {
                'pmf': d['pmf_kcal_mol'],
                'prob': d['probability'],
                'counts': d['counts'],
                'phi_edges': d['phi_edges_deg'],
                'psi_edges': d['psi_edges_deg'],
            }

    # CV1/CV2 PMF
    cv_npz = reference_dir / 'cv1_cv2_2d_fes_selected.npz'
    if cv_npz.exists():
        d = np.load(cv_npz, allow_pickle=False)
        ref['cv'] = {
            'pmf': d['pmf_kcal_mol'],
            'prob': d['probability'],
            'counts': d['counts'],
            'cv_edges': d['cv_edges_A'],
            'cv2_edges': d['cv2_edges_A'],
        }

    return ref


def add_thermodynamic_metrics(metrics: list[PartMetrics],
                               reference: dict) -> None:
    """Compute cumulative prefix RMSE/JS vs MBAR reference per residue.

    Uses cumulative histograms (prefix of all run-parts up to and including
    each part) so convergence is monotonically increasing with data.
    WARNING: comparing raw prefix histogram vs MBAR-reweighted reference.
    This is the 'thermodynamic lane' — label caveats in plots.
    """
    if not reference:
        return

    rama_ref = reference.get('rama', {})
    if not rama_ref:
        return

    # Build cumulative phi/psi count histograms aligned to reference 72-bin grid
    ref_keys = sorted(rama_ref.keys())
    n_res = len(ref_keys)
    rama_bins = 72  # match reference

    cumulative_hist = np.zeros((n_res, rama_bins, rama_bins), dtype=np.int64)

    for m in metrics:
        if m.phi_raw is None:
            m.rama_js_vs_ref = np.full(n_res, float('nan'))
            m.rama_rmse_vs_ref = np.full(n_res, float('nan'))
            continue

        # Accumulate (phi_raw is (n_part_res, 72, 72) — same grid)
        n_part_res = m.phi_raw.shape[0]
        use_res = min(n_res, n_part_res)
        cumulative_hist[:use_res] += m.phi_raw[:use_res]

        js_per_res = np.full(n_res, float('nan'))
        rmse_per_res = np.full(n_res, float('nan'))

        for ri, key in enumerate(ref_keys):
            if ri >= n_part_res:
                continue
            ref_d = rama_ref[key]
            cum_h = cumulative_hist[ri]
            if cum_h.sum() < 10:
                continue
            js_per_res[ri] = _js_2d(cum_h, ref_d['counts'])
            pmf_cum = _pmf_from_hist(cum_h)
            pmf_ref = ref_d['pmf'].copy()
            pmf_ref[~np.isfinite(pmf_ref)] = np.nanmax(pmf_ref[np.isfinite(pmf_ref)]) + 5.0
            rmse_per_res[ri] = _pmf_rmse(pmf_cum, pmf_ref, ref_counts=ref_d['counts'])

        m.rama_js_vs_ref = js_per_res
        m.rama_rmse_vs_ref = rmse_per_res


# ─── UMAP per-part coverage ──────────────────────────────────────────────────

def compute_umap_grid_coverage(parts: list[dict], all_features: list[Optional[dict]],
                                pca, umap_model, grid_bins: int = 40
                                ) -> list[Optional[np.ndarray]]:
    """Project each part's features to UMAP space, compute grid occupancy."""
    if umap_model is None:
        return [None] * len(parts)

    results = []
    for feats in all_features:
        if feats is None:
            results.append(None)
            continue
        try:
            pca_feat = pca.transform(feats['features'])
            umap_coords = umap_model.transform(pca_feat)
            results.append(umap_coords.astype(np.float32))
        except Exception:
            results.append(None)
    return results


# ─── Summary table & report ──────────────────────────────────────────────────

def print_summary(metrics: list[PartMetrics], n_clusters: int) -> None:
    hdr = (f"{'Label':<30} {'Type':<14} {'Frames':>8} {'ns':>7} "
           f"{'New clust':>10} {'Cum clust%':>10} "
           f"{'JS_rama':>9} {'RMSE_rama':>10}")
    print(hdr)
    print('-' * len(hdr))
    for m in metrics:
        js_s = f'{np.nanmean(m.rama_js_vs_ref):.4f}' if m.rama_js_vs_ref is not None else '   N/A'
        rmse_s = f'{np.nanmean(m.rama_rmse_vs_ref):.3f}' if m.rama_rmse_vs_ref is not None else '   N/A'
        print(f'{m.label:<30} {m.part_type:<14} {m.n_frames:>8,} {m.ns:>7.2f} '
              f'{m.new_clusters:>10} {m.cumulative_cluster_frac*100:>9.1f}% '
              f'{js_s:>9} {rmse_s:>10}')
    final_m = [m for m in metrics if m.cumulative_cluster_frac > 0]
    if final_m:
        print(f'\nFinal codebook coverage: {final_m[-1].cumulative_cluster_frac*100:.1f}% of {n_clusters} clusters')


def write_report(metrics: list[PartMetrics], out_dir: Path, n_clusters: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in metrics:
        row = dict(
            label=m.label, epoch=m.epoch, part_type=m.part_type,
            n_frames=m.n_frames, ns=round(m.ns, 4),
            new_clusters=m.new_clusters,
            cumulative_cluster_frac=round(m.cumulative_cluster_frac, 4),
            rama_js_mean=round(float(np.nanmean(m.rama_js_vs_ref)), 5) if m.rama_js_vs_ref is not None else None,
            rama_js_max=round(float(np.nanmax(m.rama_js_vs_ref[np.isfinite(m.rama_js_vs_ref)])), 5)
                if m.rama_js_vs_ref is not None and np.any(np.isfinite(m.rama_js_vs_ref)) else None,
            rama_rmse_mean=round(float(np.nanmean(m.rama_rmse_vs_ref)), 4) if m.rama_rmse_vs_ref is not None else None,
        )
        rows.append(row)
    report = dict(n_clusters=n_clusters, parts=rows)
    p = out_dir / 'convergence_report.json'
    p.write_text(json.dumps(report, indent=2))
    print(f'  [report] → {p}')


# ─── Plots ───────────────────────────────────────────────────────────────────

PTYPE_COLORS = {
    'flat': '#888888', 'baseline': '#2196F3', 'topup': '#FF9800',
    'final_baseline': '#4CAF50', 'final_topup': '#8BC34A',
}
EPOCH_CMAP = 'plasma'


def plot_all(parts: list[dict], metrics: list[PartMetrics],
             umap_results: list, all_features: list[Optional[dict]],
             pca, umap_model, n_clusters: int, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import Patch
    except ImportError:
        print('  [plots] matplotlib not available')
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = [PTYPE_COLORS.get(p['part_type'], '#aaaaaa') for p in parts]
    x = np.arange(len(metrics))

    fig = plt.figure(figsize=(22, 18))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── (0,0) New codebook clusters per part ─────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(x, [m.new_clusters for m in metrics], color=colors)
    ax.set_title('New codebook clusters per run part\n(EXPLORATION LANE)')
    ax.set_ylabel(f'New clusters / {n_clusters} total')
    ax.set_xticks(x[::max(1, len(x)//10)])
    ax.set_xticklabels([metrics[i].label.split('/')[-1] for i in x[::max(1, len(x)//10)]],
                       rotation=45, ha='right', fontsize=7)
    legend_patches = [Patch(color=PTYPE_COLORS[t], label=t)
                      for t in ['flat','baseline','topup','final_baseline','final_topup']
                      if any(p['part_type']==t for p in parts)]
    ax.legend(handles=legend_patches, fontsize=7)

    # ── (0,1) Cumulative codebook coverage ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    cum_ns = np.cumsum([m.ns for m in metrics])
    cum_frac = [m.cumulative_cluster_frac * 100 for m in metrics]
    ax.plot(cum_ns, cum_frac, 'k-', linewidth=2)
    for ptype, col in PTYPE_COLORS.items():
        idx = [i for i, p in enumerate(parts) if p['part_type'] == ptype]
        if idx:
            ax.scatter(cum_ns[idx], [cum_frac[i] for i in idx], c=col, s=30,
                       zorder=5, label=ptype)
    ax.set_xlabel('Cumulative ns'); ax.set_ylabel('Codebook coverage (%)')
    ax.set_title('Cumulative structural codebook coverage\n(EXPLORATION LANE)')
    ax.legend(fontsize=7)

    # ── (0,2) Ramachandran JS divergence vs MBAR reference ──────────────────
    ax = fig.add_subplot(gs[0, 2])
    js_means = [np.nanmean(m.rama_js_vs_ref) if m.rama_js_vs_ref is not None else np.nan
                for m in metrics]
    js_maxs = [np.nanmax(m.rama_js_vs_ref[np.isfinite(m.rama_js_vs_ref)])
               if m.rama_js_vs_ref is not None and np.any(np.isfinite(m.rama_js_vs_ref)) else np.nan
               for m in metrics]
    valid = np.isfinite(js_means)
    if valid.any():
        ax.fill_between(cum_ns[valid], np.array(js_maxs)[valid], alpha=0.2, color='#E53935',
                        label='max per residue')
        ax.plot(cum_ns[valid], np.array(js_means)[valid], 'o-', color='#E53935', markersize=4,
                label='mean per residue')
        ax.axhline(0.05, color='orange', linestyle='--', linewidth=1, label='JS=0.05 (rough)')
        ax.axhline(0.01, color='green', linestyle='--', linewidth=1, label='JS=0.01 (converged)')
        ax.legend(fontsize=7)
    ax.set_xlabel('Cumulative ns'); ax.set_ylabel('JS divergence')
    ax.set_title('Ramachandran JS divergence vs MBAR ref\n(THERMODYNAMIC LANE — raw prefix vs MBAR)')
    ax.set_ylim(bottom=0)

    # ── (1,0) Ramachandran RMSE per residue heatmap ──────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    n_valid_metrics = [m for m in metrics if m.rama_rmse_vs_ref is not None
                       and np.any(np.isfinite(m.rama_rmse_vs_ref))]
    if n_valid_metrics:
        n_res = len(n_valid_metrics[0].rama_rmse_vs_ref)
        rmse_mat = np.full((n_res, len(n_valid_metrics)), np.nan)
        for j, m in enumerate(n_valid_metrics):
            rmse_mat[:, j] = m.rama_rmse_vs_ref
        im = ax.imshow(rmse_mat, aspect='auto', cmap='hot_r', interpolation='nearest',
                       vmin=0)
        plt.colorbar(im, ax=ax, shrink=0.8).set_label('RMSE (kcal/mol)', fontsize=7)
        ax.set_xlabel('Run part (cumulative)')
        ax.set_ylabel('Residue index')
        ax.set_title('Ramachandran RMSE vs MBAR ref\n(per residue × epoch)')
    else:
        ax.text(0.5, 0.5, 'No thermodynamic reference available',
                ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.set_title('Ramachandran RMSE (requires --reference-dir)')

    # ── (1,1) UMAP structural map ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    if umap_model is not None and any(u is not None for u in umap_results):
        # Sample landmarks from each part, colored by epoch
        all_umap_pts = []
        all_umap_epochs = []
        for m, feats, pca_model in zip(metrics, all_features, [pca]*len(metrics)):
            if feats is None:
                continue
            try:
                pca_feat = pca.transform(feats['features'])
                coords = umap_model.transform(pca_feat[::10])  # thin for plotting
                all_umap_pts.append(coords)
                all_umap_epochs.extend([m.epoch] * len(coords))
            except Exception:
                pass
        if all_umap_pts:
            pts = np.vstack(all_umap_pts)
            eps_arr = np.array(all_umap_epochs)
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=eps_arr, cmap=EPOCH_CMAP,
                            s=1, alpha=0.3, rasterized=True)
            plt.colorbar(sc, ax=ax).set_label('Epoch', fontsize=7)
            ax.set_title('UMAP structural map (colored by epoch)\n(EXPLORATION LANE — raw dihedrals)')
    else:
        ax.text(0.5, 0.5, 'UMAP not available\n(install umap-learn)',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('UMAP structural map')
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')

    # ── (1,2) Cluster assignment heatmap (window × epoch) ────────────────────
    ax = fig.add_subplot(gs[1, 2])
    # Cluster occupancy per run-part: fraction of n_clusters occupied
    occ_fracs = []
    for m in metrics:
        if m.cluster_ids is not None:
            occ_fracs.append(len(np.unique(m.cluster_ids)) / n_clusters)
        else:
            occ_fracs.append(0.0)
    ax.bar(x, [o * 100 for o in occ_fracs], color=colors)
    ax.set_title(f'Codebook occupancy per run part\n(% of {n_clusters} clusters occupied)')
    ax.set_ylabel('Occupancy (%)')
    ax.set_xticks(x[::max(1, len(x)//10)])
    ax.set_xticklabels([metrics[i].label.split('/')[-1] for i in x[::max(1, len(x)//10)]],
                       rotation=45, ha='right', fontsize=7)

    # ── (2,0) PCA variance explained ─────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    if hasattr(pca, 'explained_variance_ratio_'):
        cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
        ax.plot(range(1, len(cumvar) + 1), cumvar, 'o-', color='#1565C0')
        ax.axhline(90, color='red', linestyle='--', linewidth=1, label='90%')
        ax.set_xlabel('PCA component'); ax.set_ylabel('Cumulative variance (%)')
        ax.legend(fontsize=8)
    ax.set_title('PCA variance explained\n(backbone dihedral features)')

    # ── (2,1) New clusters vs cumulative ns (efficiency) ─────────────────────
    ax = fig.add_subplot(gs[2, 1])
    ns_arr = np.array([m.ns for m in metrics])
    new_clust = np.array([m.new_clusters for m in metrics], dtype=float)
    eff = np.where(ns_arr > 0, new_clust / ns_arr, 0.0)
    ax.bar(x, eff, color=colors)
    ax.set_title('New clusters per ns\n(EXPLORATION LANE efficiency)')
    ax.set_ylabel('New clusters / ns')
    ax.set_xticks(x[::max(1, len(x)//10)])
    ax.set_xticklabels([metrics[i].label.split('/')[-1] for i in x[::max(1, len(x)//10)]],
                       rotation=45, ha='right', fontsize=7)

    # ── (2,2) Cumulative RMSE convergence ────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    rmse_means = [np.nanmean(m.rama_rmse_vs_ref) if m.rama_rmse_vs_ref is not None else np.nan
                  for m in metrics]
    valid2 = np.isfinite(rmse_means)
    if valid2.any():
        ax.plot(cum_ns[valid2], np.array(rmse_means)[valid2], 'o-', color='#1565C0',
                markersize=4, label='Ramachandran RMSE mean')
        ax.axhline(0.1, color='green', linestyle='--', linewidth=1, label='0.1 kcal/mol target')
        ax.set_xlabel('Cumulative ns'); ax.set_ylabel('RMSE (kcal/mol)')
        ax.legend(fontsize=7)
    ax.set_title('PMF RMSE convergence vs MBAR ref\n(THERMODYNAMIC LANE)')

    fig.suptitle('GAREUS Adaptive Convergence Audit\n(Two-lane: Exploration | Thermodynamic)',
                 fontsize=13, fontweight='bold', y=0.98)
    out_path = out_dir / 'convergence_audit.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plots] → {out_path}')

    # Separate per-residue JS heatmap
    _plot_per_residue_js(metrics, cum_ns, out_dir)


def _plot_per_residue_js(metrics: list[PartMetrics], cum_ns: np.ndarray,
                          out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    valid_m = [(i, m) for i, m in enumerate(metrics)
               if m.rama_js_vs_ref is not None and np.any(np.isfinite(m.rama_js_vs_ref))]
    if not valid_m:
        return

    n_res = len(valid_m[0][1].rama_js_vs_ref)
    js_mat = np.full((n_res, len(valid_m)), np.nan)
    x_ns = []
    for j, (i, m) in enumerate(valid_m):
        js_mat[:, j] = m.rama_js_vs_ref
        x_ns.append(cum_ns[i])

    fig, ax = plt.subplots(figsize=(max(8, len(valid_m) * 0.4), 5))
    im = ax.imshow(js_mat, aspect='auto', cmap='RdYlGn_r', interpolation='nearest',
                   vmin=0, vmax=0.3)
    ax.set_yticks(range(n_res))
    ax.set_yticklabels([f'res{r}' for r in range(n_res)], fontsize=7)
    ax.set_xticks(range(0, len(x_ns), max(1, len(x_ns)//12)))
    ax.set_xticklabels([f'{x_ns[i]:.0f}ns' for i in range(0, len(x_ns), max(1, len(x_ns)//12))],
                       rotation=45, ha='right', fontsize=7)
    plt.colorbar(im, ax=ax).set_label('JS divergence (raw vs MBAR ref)', fontsize=8)
    ax.set_title('Per-residue Ramachandran JS divergence vs MBAR reference\n'
                 '(THERMODYNAMIC LANE — green=converged, red=not converged)')
    ax.set_xlabel('Cumulative ns'); ax.set_ylabel('Residue')
    fig.tight_layout()
    out_path = out_dir / 'per_residue_js_heatmap.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plots] per-residue JS → {out_path}')


# ─── Main ─────────────────────────────────────────────────────────────────────

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
    ap.add_argument('--reference-dir', default=None,
                    help='analyze_gareus_mbar.py output dir for thermodynamic lane')
    ap.add_argument('--out', default='adaptive_convergence_audit')
    ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--n-clusters', type=int, default=80)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--no-umap', action='store_true', help='Skip UMAP (saves time)')
    args = ap.parse_args()

    adaptive_dir = Path(args.adaptive_dir).resolve()
    out_dir = Path(args.out).resolve()
    cache_dir = out_dir / '_feature_cache'

    print(f'Auditing: {adaptive_dir}')

    # Topology
    topology_pdb = find_topology(adaptive_dir)
    if topology_pdb is None:
        sys.exit('ERROR: no topology PDB found')
    print(f'  Topology: {topology_pdb}')

    print('[1] Discovering run parts...')
    parts = discover_run_parts(adaptive_dir)
    print(f'  Found {len(parts)} run parts')

    print('[2] Extracting dihedral features (cached)...')
    all_features = load_all_features(parts, topology_pdb, cache_dir,
                                     stride=args.stride, workers=args.workers)

    print('[3] Building global structural codebook (PCA + KMeans + UMAP)...')
    skip_umap = args.no_umap
    try:
        pca, kmeans, umap_model, umap_landmarks = build_codebook(
            all_features, n_clusters=args.n_clusters,
            umap_n_landmarks=50_000 if not skip_umap else 0,
        )
    except Exception as e:
        sys.exit(f'ERROR building codebook: {e}')

    print('[4] Computing per-part metrics...')
    metrics = compute_per_part_metrics(parts, all_features, pca, kmeans,
                                       args.n_clusters)

    print('[5] UMAP projections...')
    umap_results = compute_umap_grid_coverage(parts, all_features, pca, umap_model)

    print('[6] Thermodynamic lane...')
    reference = {}
    if args.reference_dir:
        ref_path = Path(args.reference_dir)
        if ref_path.exists():
            reference = load_reference_npz(ref_path)
            n_rama = len(reference.get('rama', {}))
            print(f'  Loaded {n_rama} residue Ramachandran references + '
                  f'{"CV PMF" if "cv" in reference else "no CV PMF"}')
        else:
            print(f'  WARNING: reference dir not found: {ref_path}')
    else:
        print('  Skipped (no --reference-dir provided)')

    add_thermodynamic_metrics(metrics, reference)

    print('\n═══ Convergence summary ═══')
    print_summary(metrics, args.n_clusters)

    print('\n[7] Writing report...')
    write_report(metrics, out_dir, args.n_clusters)

    if not args.no_plots:
        print('[8] Plotting...')
        plot_all(parts, metrics, umap_results, all_features, pca,
                 umap_model, args.n_clusters, out_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
