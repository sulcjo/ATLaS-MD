"""Matplotlib rendering + smoothing/label helpers for the MBAR/PMF analysis
report. Relocated verbatim from analyze_gareus_mbar.py (Plan A6a) -- no
logic changes, only module location.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from gareus.mbar_analysis import plotstyle as ps
from gareus.mbar_analysis.data import Data
from gareus.units import KJ_PER_KCAL

def _secondary_cv_label(meta: dict) -> str:
    """Extract human-readable secondary CV label from run metadata."""
    raw = (meta or {}).get('secondary_cv', {})
    if isinstance(raw, str):
        low = raw.lower()
        if 'rama' in low: return 'Ramachandran CV'
        if 'torsion-pca' in low or 'bootstrap' in low: return 'Bootstrap torsion PC1'
        if 'tica' in low: return 'tIC1 torsion CV'
        return f'Secondary CV ({raw})'
    sec = raw or {}
    label = sec.get('label', '')
    mode = sec.get('mode', '')
    if label: return label
    if mode == 'torsion-pca': return 'Bootstrap torsion PC1'
    if mode == 'tica-linear': return 'tIC1 torsion CV'
    if mode: return f'Secondary CV ({mode})'
    return 'Secondary CV'

def _secondary_cv_regions(meta: dict) -> list:
    """Return [{value, label}] region annotations for Ramachandran-style secondary CVs."""
    raw = (meta or {}).get('secondary_cv', {})
    sec = raw if isinstance(raw, dict) else {}
    return [{'value': float(r['value']), 'label': r.get('label', r.get('name', ''))}
            for r in sec.get('regions', []) if 'value' in r]


def _regime_slug(regime: str) -> str:
    if not regime:
        return 'unknown'
    from analyze_gareus_mbar import _slug
    return _slug(regime)
def _primary_cv_label(meta: dict) -> str:
    label = (meta or {}).get('primary_cv_label', '')
    if label: return label
    mode = (meta or {}).get('primary_cv', '')
    if mode == 'nonlocal-contacts': return 'nonlocal contact fraction'
    return 'CV distance'

def _primary_cv_units(meta: dict) -> str:
    units = (meta or {}).get('primary_cv_units', '')
    if units: return units
    mode = (meta or {}).get('primary_cv', '')
    if mode == 'nonlocal-contacts': return 'dimensionless'
    return 'A'

def _primary_cv_axis_label(meta: dict) -> str:
    label = _primary_cv_label(meta)
    units = _primary_cv_units(meta)
    if units == 'dimensionless': return label
    return f'{label} ({units})'
def _smooth_masked_grid(grid, sigma=1.0):
    grid=np.asarray(grid,dtype=np.float64)
    finite=np.isfinite(grid)
    if not np.any(finite):
        return grid
    if sigma is None or float(sigma) <= 0:
        return grid.copy()
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        return grid.copy()
    vals=np.where(finite, grid, 0.0)
    mask=finite.astype(np.float64)
    sval=gaussian_filter(vals*mask, sigma=float(sigma), mode='nearest')
    smask=gaussian_filter(mask, sigma=float(sigma), mode='nearest')
    out=np.full_like(grid, np.nan, dtype=np.float64)
    good=smask>1e-8
    out[good]=sval[good]/smask[good]
    return out

def _smooth_pmf_1d(pmf, sigma):
    """Gaussian smooth a 1D PMF array for plotting; NaN values preserved."""
    if sigma is None or float(sigma) <= 0:
        return np.asarray(pmf, dtype=float)
    try:
        from scipy.ndimage import gaussian_filter1d
    except Exception:
        return np.asarray(pmf, dtype=float)
    pmf = np.asarray(pmf, dtype=float)
    finite = np.isfinite(pmf)
    if not np.any(finite):
        return pmf.copy()
    filled = pmf.copy()
    idx = np.where(finite)[0]
    for i in np.where(~finite)[0]:
        filled[i] = pmf[idx[np.argmin(np.abs(idx - i))]]
    smoothed = gaussian_filter1d(filled, sigma=float(sigma), mode='nearest')
    out = pmf.copy()
    out[finite] = smoothed[finite]
    return out

def _eff_smooth(args, attr: str) -> float:
    """Return effective smooth sigma: --smooth-sigma master overrides specific attr."""
    master = float(getattr(args, 'smooth_sigma', 0.0) or 0.0)
    if master > 0:
        return master
    return float(getattr(args, attr, 0.0) or 0.0)

FES_PLOT_VMAX_VALUES = (2.0, 5.0, 10.0, 20.0, None)

def _fes_range_tag(vmax) -> str:
    if vmax is None:
        return '0_all'
    v=float(vmax)
    if abs(v-round(v)) < 1.0e-9:
        return f'0_{int(round(v))}'
    return '0_' + str(v).replace('.', 'p')

def _fes_range_label(vmax, actual_max: float) -> str:
    if vmax is None:
        if math.isfinite(float(actual_max)):
            return f'0-all kcal/mol (max {float(actual_max):.2f})'
        return '0-all kcal/mol'
    return f'0-{float(vmax):g} kcal/mol'

def _fes_variant_path(base_path, vmax) -> Path:
    base=Path(base_path)
    suffix=base.suffix or '.png'
    return base.with_name(f'{base.stem}_{_fes_range_tag(vmax)}{suffix}')

def _plot_2d_fes_range(
    F, xedges, yedges, xc, yc, out_png, title, xlabel, ylabel, warnings,
    *, smooth_sigma=1.0, range_vmax=None, figsize=(8.8,6.6), dpi=220, cmap_name='viridis', contour=True,
    y_annotation_lines=None
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        warnings.append(f'matplotlib unavailable; no 2D FES plot written: {e}')
        return False
    F=np.asarray(F,dtype=np.float64)
    xedges=np.asarray(xedges,dtype=np.float64)
    yedges=np.asarray(yedges,dtype=np.float64)
    xc=np.asarray(xc,dtype=np.float64)
    yc=np.asarray(yc,dtype=np.float64)
    Fplot=_smooth_masked_grid(F, sigma=smooth_sigma)
    finite=np.isfinite(Fplot)
    if not np.any(finite):
        warnings.append(f'{title} has no finite bins; skipped 2D FES heatmap plot.')
        return False
    actual_max=float(np.nanmax(Fplot[finite]))
    if not np.isfinite(actual_max) or actual_max <= 0:
        actual_max=1.0
    if range_vmax is None:
        vmax=max(1.0, actual_max)
        extend='neither'
    else:
        vmax=max(1.0e-12, float(range_vmax))
        extend='max' if actual_max > vmax else 'neither'
    fig,ax=plt.subplots(figsize=figsize)
    cmap=plt.get_cmap(cmap_name).copy() if hasattr(plt.get_cmap(cmap_name),'copy') else plt.get_cmap(cmap_name)
    try:
        cmap.set_bad(color='white', alpha=0.0)
    except Exception:
        pass
    im=ax.imshow(Fplot.T, origin='lower', extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]], aspect='auto', interpolation='bicubic', cmap=cmap, vmin=0.0, vmax=vmax)
    if contour:
        try:
            X,Y=np.meshgrid(xc,yc,indexing='ij')
            contour_source=np.where(np.isfinite(Fplot),Fplot,np.nan)
            levels=np.linspace(0.0, vmax, 10)
            if np.count_nonzero(np.isfinite(contour_source)) >= 9 and len(levels) > 2:
                cs=ax.contour(X,Y,contour_source,levels=levels[1:],colors='white',linewidths=0.7,alpha=0.75)
                ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
        except Exception:
            pass
    finite_raw=np.isfinite(F)
    if np.any(finite_raw):
        min_idx=np.unravel_index(np.nanargmin(np.where(finite_raw,F,np.inf)),F.shape)
        ax.plot([xc[min_idx[0]]],[yc[min_idx[1]]],marker='*',markersize=11,markeredgecolor='black',markerfacecolor='gold',zorder=5)
    if y_annotation_lines:
        for ann in y_annotation_lines:
            v = float(ann.get('value', float('nan')))
            lbl = str(ann.get('label', ''))
            if not (np.isfinite(v) and yedges[0] <= v <= yedges[-1]):
                continue
            ax.axhline(v, color='white', linewidth=0.9, linestyle='--', alpha=0.7)
            ax.text(xedges[0], v, f' {lbl}', color='white', fontsize=7, va='bottom', ha='left',
                    bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.35, lw=0))
    cbar=fig.colorbar(im,ax=ax,extend=extend)
    cbar.set_label('Free energy (kcal/mol, minimum shifted to 0)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title} [{_fes_range_label(range_vmax, actual_max)}]')
    ax.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(out_png,dpi=dpi)
    plt.close(fig)
    return True

def _plot_2d_fes_multirange(
    F, xedges, yedges, xc, yc, out_png, title, xlabel, ylabel, warnings,
    *, smooth_sigma=1.0, figsize=(8.8,6.6), dpi=220, cmap_name='viridis', contour=True,
    y_annotation_lines=None
) -> dict[str,str]:
    files={}
    first_vmax=FES_PLOT_VMAX_VALUES[0]
    first_path=None
    for vmax in FES_PLOT_VMAX_VALUES:
        tag=_fes_range_tag(vmax)
        path=_fes_variant_path(out_png, vmax)
        ok=_plot_2d_fes_range(F,xedges,yedges,xc,yc,path,title,xlabel,ylabel,warnings,smooth_sigma=smooth_sigma,range_vmax=vmax,figsize=figsize,dpi=dpi,cmap_name=cmap_name,contour=contour,y_annotation_lines=y_annotation_lines)
        if ok:
            files[tag]=str(path)
            if vmax == first_vmax:
                first_path=path
    if first_path is not None:
        # The "main"/untagged file is byte-for-byte identical to the first
        # tagged range variant (confirmed via SHA256) -- copy it instead of
        # re-running the whole render (figure/contour/colorbar/PNG encode)
        # a second time from scratch.
        shutil.copyfile(first_path, out_png)
        files['main']=str(out_png)
    return files

def plot_2d_fes(fes, method, out_png, title, warnings, smooth_sigma=1.0):
    F=np.asarray(fes['pmf'],dtype=np.float64)
    xedges=np.asarray(fes['cv_edges_A'],dtype=np.float64)
    yedges=np.asarray(fes['rg_edges_A'],dtype=np.float64)
    xc=np.asarray(fes['cv_A'],dtype=np.float64)
    yc=np.asarray(fes['rg_A'],dtype=np.float64)
    return _plot_2d_fes_multirange(F,xedges,yedges,xc,yc,out_png,title,'CV distance (A)','Rg (A)',warnings,smooth_sigma=smooth_sigma,figsize=(8.8,6.6),dpi=220,cmap_name='viridis',contour=True)

def plot_cv1_cv2_2d_fes(fes, method, out_png, title, warnings, smooth_sigma=1.0, cv2_label='Secondary CV', cv1_label='CV distance (A)', regions=None):
    F=np.asarray(fes['pmf'],dtype=np.float64)
    xedges=np.asarray(fes['cv_edges_A'],dtype=np.float64)
    yedges=np.asarray(fes['rg_edges_A'],dtype=np.float64)
    xc=np.asarray(fes['cv_A'],dtype=np.float64)
    yc=np.asarray(fes['rg_A'],dtype=np.float64)
    return _plot_2d_fes_multirange(F,xedges,yedges,xc,yc,out_png,title,cv1_label,cv2_label,warnings,smooth_sigma=smooth_sigma,figsize=(8.8,6.6),dpi=220,cmap_name='viridis',contour=True,y_annotation_lines=regions)
def plot_rg_outputs(d: Data, rg_pmfs: dict, selected: str, out: Path, warnings: list[str], smooth_sigma: float = 0.0, args=None):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        warnings.append(f'matplotlib unavailable; no Rg PNG plots written: {e}')
        return
    import gareus.mbar_analysis.plotstyle as ps
    fig,ax=plt.subplots(figsize=(8,5))
    for i,(name,p) in enumerate(_visible_pmfs(rg_pmfs, selected, args).items()):
        pmf_plot=_smooth_pmf_1d(p['pmf'],smooth_sigma); m=np.isfinite(pmf_plot)
        if np.any(m): ps.plot_method_curve(ax,p['cv_A'][m],pmf_plot[m],name,selected,idx=i)
    ps.style_line_axes(ax,xlabel='Rg (Å)',ylabel='PMF (kcal/mol, shifted)',title='Radius of gyration PMF estimates'); fig.tight_layout(); fig.savefig(out/'rg_pmf_all_methods.png',dpi=200); plt.close(fig)
    rg=np.asarray(d.rg_A,dtype=float); mask=np.isfinite(rg)&np.isfinite(d.cv)
    if np.count_nonzero(mask)>5:
        fig,ax=plt.subplots(figsize=(6,5))
        hb=ax.hexbin(d.cv[mask],rg[mask],gridsize=45,mincnt=1)
        ax.set_xlabel(_primary_cv_axis_label(d.meta)); ax.set_ylabel('Rg (A)'); ax.set_title('Sampled CV-Rg coverage')
        fig.colorbar(hb,ax=ax,label='sample count'); fig.tight_layout(); fig.savefig(out/'rg_vs_cv_sampled.png',dpi=200); plt.close(fig)
def plot_pca_2d_fes(fes: dict, method: str, out_png: Path, title: str, warnings: list[str], smooth_sigma: float = 1.0) -> dict[str,str]:
    F=np.asarray(fes['pmf'],dtype=np.float64)
    xedges=np.asarray(fes['cv_edges_A'],dtype=np.float64)
    yedges=np.asarray(fes['rg_edges_A'],dtype=np.float64)
    xc=np.asarray(fes['cv_A'],dtype=np.float64)
    yc=np.asarray(fes['rg_A'],dtype=np.float64)
    return _plot_2d_fes_multirange(F,xedges,yedges,xc,yc,out_png,title,'PCA1 (A)','PCA2 (A)',warnings,smooth_sigma=smooth_sigma,figsize=(8.8,6.6),dpi=220,cmap_name='viridis',contour=True)
_OPT_IN_GAMD_METHODS = {'gamd_exponential': 'plot_gamd_exponential', 'gamd_cumulant3': 'plot_gamd_cumulant3'}

def _want_gamd_method(method: str, chosen: str, args) -> bool:
    """gamd_exponential/gamd_cumulant3 are opt-in (--plot-gamd-exponential/--plot-gamd-cumulant3);
    umbrella_only and gamd_cumulant2 are always shown. The currently chosen/selected method is
    always shown even if it is one of the opt-in ones (explicit --selected-method forces it)."""
    flag = _OPT_IN_GAMD_METHODS.get(method)
    return flag is None or bool(getattr(args, flag, False)) or method == chosen

def _visible_pmfs(pmfs: dict, chosen: str, args) -> dict:
    """Filter a {method: pmf} dict down to the methods that should be drawn in a multi-method
    comparison plot, per _want_gamd_method."""
    return {name: p for name, p in pmfs.items() if _want_gamd_method(name, chosen, args)}
def _per_window_gamd_boost_stats(window, comb_kcal, kbt_kcal, K, dih_kcal=None):
    """Per-window GaMD-boost statistics used by plot_gamd_boost.

    Computed via a single stable sort + per-window contiguous slice, shared
    across every statistic, instead of re-deriving the O(N) `window==k`
    boolean mask separately per statistic (perf audit: ~31.5s at
    N=8.18M/K=364 with the old per-stat masking vs. 0.76s with this
    sort-once approach; independently re-measured here at N=2M/K=364:
    7.28s -> 0.38s, ~19x). Output is bit-identical, not just close -- a
    stable sort preserves each window's samples in their original relative
    order, so every slice here is element-for-element identical to the
    equivalent `field[window==k]` (verified directly with
    `np.array_equal`, including the empty-window case; see
    tests/test_perf_plotting_redundancy.py).

    Returns a dict with keys means_comb, stds_comb, varbdv, skew, kurt,
    anharmonicity, groups (list of K per-window finite-filtered arrays), and
    -- only when `dih_kcal` is given -- means_dih, frac_dih.
    """
    try:
        from gareus.mbar_analysis.pmf import _window_moments
    except ImportError:
        from analyze_gareus_mbar import _window_moments
    window = np.asarray(window)
    comb_kcal = np.asarray(comb_kcal)
    has_dih = dih_kcal is not None
    wins = np.arange(K)
    order = np.argsort(window, kind='stable')
    window_sorted = window[order]
    starts = np.searchsorted(window_sorted, wins, side='left')
    ends = np.searchsorted(window_sorted, wins, side='right')
    comb_sorted = comb_kcal[order]
    win_means_comb = np.full(K, np.nan)
    win_stds_comb = np.full(K, np.nan)
    win_varbdv = np.full(K, np.nan)
    win_skew = np.full(K, np.nan)
    win_kurt = np.full(K, np.nan)
    win_anharmonicity = np.full(K, np.nan)
    win_groups = [comb_sorted[0:0] for _ in range(K)]
    if has_dih:
        dih_kcal = np.asarray(dih_kcal)
        dih_sorted = dih_kcal[order]
        _c_pos = comb_kcal.copy(); _c_pos[_c_pos <= 0] = np.nan
        cpos_sorted = _c_pos[order]
        win_means_dih = np.full(K, np.nan)
        win_frac_dih = np.full(K, np.nan)
    for k in range(K):
        lo, hi = starts[k], ends[k]
        if hi <= lo:
            continue
        seg_comb = comb_sorted[lo:hi]
        win_means_comb[k] = float(np.nanmean(seg_comb))
        win_stds_comb[k] = float(np.nanstd(seg_comb))
        win_varbdv[k] = float(np.var(seg_comb / kbt_kcal))
        win_groups[k] = seg_comb[np.isfinite(seg_comb)]
        win_skew[k], win_kurt[k], win_anharmonicity[k] = _window_moments(seg_comb)
        if has_dih:
            seg_dih = dih_sorted[lo:hi]
            win_means_dih[k] = float(np.nanmean(seg_dih))
            win_frac_dih[k] = float(np.nanmedian(seg_dih / cpos_sorted[lo:hi]))
    out = dict(means_comb=win_means_comb, stds_comb=win_stds_comb, varbdv=win_varbdv,
               skew=win_skew, kurt=win_kurt, anharmonicity=win_anharmonicity, groups=win_groups)
    if has_dih:
        out['means_dih'] = win_means_dih
        out['frac_dih'] = win_frac_dih
    return out


def plot_gamd_boost(d, out, warnings):
    """Write gamd_boost_diagnostics.png, gamd_dv_distribution_per_window.png,
    gamd_reweight_quality.png, and gamd_cumulant_quality.png to out/."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        warnings.append(f'matplotlib unavailable; GaMD boost plots skipped: {e}'); return
    boost_comb = d.boost_kj
    if not (np.isfinite(boost_comb).sum() > 10 and np.nanstd(boost_comb) > 1e-12):
        return
    kj_kcal = 1.0 / KJ_PER_KCAL
    kbt_kcal = 1.0 / (d.beta * KJ_PER_KCAL)
    comb_kcal = boost_comb * kj_kcal
    has_dih = (d.boost_dih_kj is not None and np.isfinite(d.boost_dih_kj).sum() > 10)
    dih_kcal = d.boost_dih_kj * kj_kcal if has_dih else None
    tot_kcal = (comb_kcal - dih_kcal) if has_dih else None
    K = d.u_nk.shape[1]
    wins = np.arange(K)

    stats = _per_window_gamd_boost_stats(d.window, comb_kcal, kbt_kcal, K, dih_kcal)
    win_means_comb = stats['means_comb']
    win_stds_comb = stats['stds_comb']
    win_varbdv = stats['varbdv']
    win_skew = stats['skew']
    win_kurt = stats['kurt']
    win_anharmonicity = stats['anharmonicity']
    win_groups = stats['groups']
    if has_dih:
        win_means_dih = stats['means_dih']
        win_frac_dih = stats['frac_dih']
        win_means_tot = win_means_comb - win_means_dih
    n_panels = 3 if has_dih else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5), constrained_layout=True)
    ax = axes[0]
    bins_hist = np.linspace(0, float(np.nanpercentile(comb_kcal, 99.5)), 80)
    ax.hist(np.clip(comb_kcal, 0, None), bins=bins_hist, color='#2ecc71', alpha=0.5, label='combined', density=True)
    if has_dih:
        ax.hist(np.clip(dih_kcal, 0, None), bins=bins_hist, color='#e05c5c', alpha=0.6, label='dihedral', density=True)
        ax.hist(np.clip(tot_kcal, 0, None), bins=bins_hist, color='#5c82e0', alpha=0.45, label='total-PE', density=True)
    ax.axvline(kbt_kcal, color='k', ls='--', lw=0.9, label='kT')
    ax.set_xlabel('GaMD boost ΔV (kcal/mol)'); ax.set_ylabel('density')
    ax.set_title('GaMD boost distribution'); ax.legend(fontsize=8)
    ax = axes[1]
    if has_dih:
        ax.bar(wins, win_means_dih, label='dihedral', color='#e05c5c', alpha=0.8)
        ax.bar(wins, win_means_tot, bottom=win_means_dih, label='total-PE', color='#5c82e0', alpha=0.8)
    else:
        ax.bar(wins, win_means_comb, label='combined', color='#2ecc71', alpha=0.8)
    ax.errorbar(wins, win_means_comb, yerr=win_stds_comb, fmt='none', color='k', capsize=3)
    ax.set_xlabel('window index'); ax.set_ylabel('mean boost (kcal/mol)')
    ax.set_title('Mean boost per window  (error bars = ±σ)'); ax.legend(fontsize=8)
    if has_dih:
        ax = axes[2]
        ax.bar(wins, win_frac_dih, color='#9b59b6', alpha=0.85)
        med_frac = float(np.nanmedian(win_frac_dih))
        ax.axhline(med_frac, color='k', ls='--', lw=1.2, label=f'median={med_frac:.3f}')
        ax.set_ylim(0, 1); ax.set_xlabel('window index')
        ax.set_ylabel('dihedral / combined (median)'); ax.set_title('Dihedral fraction per window')
        ax.legend(fontsize=8)
    fig.savefig(out / 'gamd_boost_diagnostics.png', dpi=200, bbox_inches='tight'); plt.close(fig)

    valid_wins = [k for k in range(K) if win_groups[k].size >= 4]
    if valid_wins:
        fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(valid_wins)), 4.5), constrained_layout=True)
        vp = ax.violinplot([win_groups[k] for k in valid_wins], positions=valid_wins, widths=0.8,
                            showmeans=True, showextrema=True)
        for body in vp['bodies']:
            body.set_facecolor('#2ecc71'); body.set_alpha(0.55)
        for part in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
            if part in vp: vp[part].set_color('#1e8449')
        ax.axhline(kbt_kcal, color='k', ls='--', lw=0.9, label='kT')
        ax.set_xlabel('window index'); ax.set_ylabel('GaMD boost ΔV (kcal/mol)')
        ax.set_title('ΔV distribution per window (combined boost)')
        ax.legend(fontsize=8)
        fig.savefig(out / 'gamd_dv_distribution_per_window.png', dpi=200, bbox_inches='tight'); plt.close(fig)
    else:
        warnings.append('GaMD ΔV per-window distribution plot skipped: no window has >=4 finite boost samples')

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.bar(wins, win_varbdv, color='#e08c5c', alpha=0.85)
    for thresh, color, lbl in [(1.0, 'green', 'var=1'), (5.0, 'orange', 'var=5'), (10.0, 'red', 'var=10')]:
        ax.axhline(thresh, color=color, ls='--', lw=1, label=lbl)
    ax.set_xlabel('window index'); ax.set_ylabel('var(β·ΔV_combined)')
    ax.set_title('GaMD reweighting quality per window  [↑ = worse ESS]'); ax.legend(fontsize=8)
    fig.savefig(out / 'gamd_reweight_quality.png', dpi=200, bbox_inches='tight'); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    # Panel 1: skewness
    ax = axes[0]
    bar_colors_skew = ['#fa8072' if s >= 0 else '#87ceeb' for s in win_skew]
    ax.bar(wins, win_skew, color=bar_colors_skew, alpha=0.85)
    for thresh, color in [(0.5, 'orange'), (-0.5, 'orange'), (1.0, 'red'), (-1.0, 'red')]:
        ax.axhline(thresh, color=color, ls='--', lw=1.0, label=f'{thresh:+.1f}' if thresh > 0 else None)
    ax.set_xlabel('window index'); ax.set_ylabel('skewness')
    ax.set_title('Boost skewness per window\n(pre-smoothing, raw ΔV)')
    handles = [plt.Line2D([0], [0], color='orange', ls='--', lw=1, label='±0.5 (marginal)'),
               plt.Line2D([0], [0], color='red',    ls='--', lw=1, label='±1.0 (poor)')]
    ax.legend(handles=handles, fontsize=8)
    # Panel 2: excess kurtosis
    ax = axes[1]
    ax.bar(wins, win_kurt, color='#8ecae6', alpha=0.85)
    for thresh, color in [(1.0, 'orange'), (-1.0, 'orange'), (2.0, 'red'), (-2.0, 'red')]:
        ax.axhline(thresh, color=color, ls='--', lw=1.0)
    ax.set_xlabel('window index'); ax.set_ylabel('excess kurtosis')
    ax.set_title('Boost excess kurtosis per window\n(pre-smoothing, raw ΔV)')
    handles = [plt.Line2D([0], [0], color='orange', ls='--', lw=1, label='±1.0 (marginal)'),
               plt.Line2D([0], [0], color='red',    ls='--', lw=1, label='±2.0 (poor)')]
    ax.legend(handles=handles, fontsize=8)
    # Panel 3: anharmonicity
    ax = axes[2]
    bar_colors_anh = ['#2ecc71' if v < 0.3 else ('#e08c2e' if v < 1.0 else '#e05c5c') for v in win_anharmonicity]
    ax.bar(wins, win_anharmonicity, color=bar_colors_anh, alpha=0.85)
    ax.axhline(0.3, color='green', ls='--', lw=1.0, label='0.3 (marginal)')
    ax.axhline(1.0, color='red',   ls='--', lw=1.0, label='1.0 (poor)')
    ax.set_xlabel('window index'); ax.set_ylabel('anharmonicity score')
    ax.set_title('Cumulant2 validity score per window\n(0: Gaussian; >1: poor cumulant2)')
    import matplotlib.patches as _mpatch
    handles = [_mpatch.Patch(color='#2ecc71', alpha=0.85, label='<0.3 (good)'),
               _mpatch.Patch(color='#e08c2e', alpha=0.85, label='0.3-1.0 (marginal)'),
               _mpatch.Patch(color='#e05c5c', alpha=0.85, label='>1.0 (poor)'),
               plt.Line2D([0], [0], color='green', ls='--', lw=1, label='0.3 threshold'),
               plt.Line2D([0], [0], color='red',   ls='--', lw=1, label='1.0 threshold')]
    ax.legend(handles=handles, fontsize=8)
    fig.savefig(out / 'gamd_cumulant_quality.png', dpi=200, bbox_inches='tight'); plt.close(fig)

def plot_outputs(d,pmfs,selected,O,out,warnings,smooth_sigma=0.0,args=None):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        warnings.append(f'matplotlib unavailable; no PNG plots written: {e}'); return
    import gareus.mbar_analysis.plotstyle as ps
    _cvlab=_primary_cv_axis_label(d.meta)
    fig,ax=plt.subplots(figsize=(8,5))
    for i,(name,p) in enumerate(_visible_pmfs(pmfs, selected, args).items()):
        pmf_plot=_smooth_pmf_1d(p['pmf'],smooth_sigma); m=np.isfinite(pmf_plot)
        if np.any(m): ps.plot_method_curve(ax,p['cv_A'][m],pmf_plot[m],name,selected,idx=i)
    ps.style_line_axes(ax,xlabel=_cvlab,ylabel='PMF (kcal/mol, shifted)',title='ATLaS-MD PMF estimates'); fig.tight_layout(); fig.savefig(out/'pmf_all_methods.png',dpi=200); plt.close(fig)
    p=pmfs[selected]; pmf_plot=_smooth_pmf_1d(p['pmf'],smooth_sigma); fig,ax=plt.subplots(figsize=(8,5)); m=np.isfinite(pmf_plot); _selc,_=ps.method_style(selected); ax.plot(p['cv_A'][m],pmf_plot[m],linewidth=2.6,color=_selc); ps.annotate_minimum(ax,p['cv_A'][m],pmf_plot[m]); ps.style_line_axes(ax,xlabel=_cvlab,ylabel='PMF (kcal/mol, shifted)',title=f'Selected unbiased PMF: {ps.pretty_method(selected)}',legend=False); fig.tight_layout(); fig.savefig(out/'pmf_unbiased.png',dpi=200); plt.close(fig)
    counts=np.bincount(d.window[(d.window>=0)&(d.window<d.u_nk.shape[1])],minlength=d.u_nk.shape[1]); fig,ax=plt.subplots(figsize=(8,4)); ax.bar(np.arange(counts.size),counts,color=ps.BAR_COLOR); ps.style_line_axes(ax,xlabel='window',ylabel='samples',title='Samples per umbrella window',legend=False); fig.tight_layout(); fig.savefig(out/'window_sample_counts.png',dpi=200); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(O,origin='lower',vmin=0,vmax=1,aspect='auto'); ax.set_xlabel('window'); ax.set_ylabel('window'); ax.set_title('CV1 marginal histogram overlap'); fig.colorbar(im,ax=ax,label='overlap'); fig.tight_layout(); fig.savefig(out/'overlap_matrix_cv1_hist.png',dpi=200); plt.close(fig)
    plot_gamd_boost(d, out, warnings)


def plot_ladder_crosscheck(cross: dict, out: Path, cv_label: str = 'CV') -> Optional[str]:
    """Two-curve overlay for the lambda-ladder quoting gate: the full-ladder
    PMF (every sample, global f_k) against the lambda=0-only PMF (same
    global f_k, only the plain-umbrella rungs' own samples -- see
    gareus.mbar_analysis.crosscheck.ladder_crosscheck). Only called when
    that check actually ran (status 'pass' or 'fail', never 'skipped').

    The lambda=0 curve reuses plotstyle's 'umbrella_only' colour -- it IS a
    plain umbrella PMF -- so it reads as the same "honest unbiased
    reference" hue used everywhere else in this report; the full-ladder
    curve gets the other strong, already-registered hue ('gamd_cumulant2')
    so the two are distinguishable under the same colourblind-safe palette
    without inventing a third fixed slot for a curve pair that only ever
    appears in this one plot.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    pmf_full = cross.get('pmf_full') or {}
    pmf_lam0 = cross.get('pmf_lambda0') or {}
    if not pmf_full or not pmf_lam0:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    full_color, _ = ps.method_style('gamd_cumulant2')
    lam0_color, _ = ps.method_style('umbrella_only')
    F = np.asarray(pmf_full['pmf'], dtype=float); mF = np.isfinite(F)
    L = np.asarray(pmf_lam0['pmf'], dtype=float); mL = np.isfinite(L)
    if np.any(mF):
        ax.plot(np.asarray(pmf_full['cv_A'])[mF], F[mF], color=full_color,
                linewidth=2.6, label='full ladder (all samples)')
    if np.any(mL):
        ax.plot(np.asarray(pmf_lam0['cv_A'])[mL], L[mL], color=lam0_color,
                linewidth=2.2, linestyle='--', label='λ=0 only (umbrella)')
    status = cross.get('status', 'unknown')
    diff = cross.get('max_abs_diff_kcal'); tol = cross.get('tolerance_kcal')
    title = f'λ-ladder cross-check: {status}'
    if diff is not None and tol is not None:
        title += f' (max |Δ| {diff:.2f} / tol {tol:.2f} kcal/mol)'
    ps.style_line_axes(ax, xlabel=cv_label, ylabel='PMF (kcal/mol, shifted)', title=title)
    fig.tight_layout()
    path = out / 'pmf_ladder_crosscheck.png'
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)
