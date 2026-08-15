"""Convergence-sweep figure and report writers. Relocated verbatim from
analyze_gareus_mbar.py (Plan A6a) -- no logic changes, only module
location. write_convergence_report (the dead CV1-legacy sibling of
write_observable_convergence_report) was deleted, not relocated -- it had
zero callers anywhere in the repo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from gareus.mbar_analysis.plotting import _smooth_pmf_1d

def _add_epoch_annotations_to_axes(axes_list: list, epoch_annotations: list) -> None:
    """Add vertical lines + labels for epoch/topup boundaries to a list of matplotlib Axes."""
    if not epoch_annotations:
        return
    for ax in axes_list:
        ylim = ax.get_ylim()
        for i, (label, frac, color) in enumerate(epoch_annotations):
            ax.axvline(frac, color=color, lw=0.9, ls='--', alpha=0.7)
            # Label above the plot: alternate y to avoid overlap; clamp x so the
            # right-edge (frac≈1.0) label stays inside the axes instead of clipping.
            y_pos = 0.97 - 0.09 * (i % 4)
            tx = min(frac + 0.005, 0.985)
            ax.text(tx, y_pos, label, transform=ax.get_xaxis_transform(),
                    fontsize=5.5, color=color, va='top', rotation=90, alpha=0.85)


def _add_aggregate_ns_secondary_axis(ax, x_frac: np.ndarray, agg_ns: np.ndarray) -> None:
    """Add cumulative aggregate simulation time (ns) as a secondary top x-axis."""
    if agg_ns is None or len(agg_ns) < 2:
        return
    x_f = np.asarray(x_frac, dtype=np.float64)
    a_n = np.asarray(agg_ns, dtype=np.float64)
    valid = np.isfinite(x_f) & np.isfinite(a_n)
    x_f, a_n = x_f[valid], a_n[valid]
    if len(x_f) < 2 or not np.all(np.diff(a_n) >= 0):
        return
    try:
        import matplotlib.ticker as mticker
        fwd = lambda f, _x=x_f, _a=a_n: np.interp(np.asarray(f, float), _x, _a)
        inv = lambda n, _x=x_f, _a=a_n: np.interp(np.asarray(n, float), _a, _x)
        ax2 = ax.secondary_xaxis('top', functions=(fwd, inv))
        ax2.set_xlabel('aggregate simulation (ns)', fontsize=7.5)
        total_ns = float(a_n[-1])
        if total_ns < 1.0:
            fmt = mticker.FuncFormatter(lambda v, _: f'{v * 1000:.0f}ps')
        elif total_ns < 10.0:
            fmt = mticker.FuncFormatter(lambda v, _: f'{v:.2f}ns')
        else:
            fmt = mticker.FuncFormatter(lambda v, _: f'{v:.1f}ns')
        ax2.xaxis.set_major_formatter(fmt)
        ax2.tick_params(labelsize=6.5)
    except Exception:
        pass


def _write_epoch_ess_plot(conv_rows: list, out: Path, *, ea: list = None,
                          aggregate_ns: Optional[np.ndarray] = None) -> Optional[str]:
    """Plot MBAR vs GaMD-reweight effective-sample-size fraction across epochs.

    MBAR ESS (umbrella debiasing) typically grows as epochs accumulate; the GaMD
    exponential-reweighting ESS (umbrella + boost) usually stays near zero — so the
    two on one log-y axis make the boost-reweighting collapse legible at a glance.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    rows = [r for r in conv_rows if 'frac_total' in r
            and ('mbar_ess_frac' in r or 'gamd_reweight_ess_frac' in r)]
    if not rows:
        return None
    x = np.asarray([r['frac_total'] for r in rows], dtype=float)
    mbar = np.asarray([r.get('mbar_ess_frac', np.nan) for r in rows], dtype=float)
    gamd = np.asarray([r.get('gamd_reweight_ess_frac', np.nan) for r in rows], dtype=float)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    if np.any(np.isfinite(mbar) & (mbar > 0)):
        ax.plot(x, mbar, 'o-', color='#377eb8', lw=1.4, ms=4,
                label='MBAR ESS / N  (umbrella debias)')
    if np.any(np.isfinite(gamd) & (gamd > 0)):
        ax.plot(x, gamd, 's-', color='#e41a1c', lw=1.4, ms=4,
                label='GaMD reweight ESS / N  (incl. boost)')
    ax.set_yscale('log')
    ax.set_xlabel('fraction of production samples')
    ax.set_ylabel('effective sample size fraction (log)')
    ax.set_title('Effective sample size vs accumulated sampling')
    ax.axhline(0.05, color='gray', ls=':', lw=0.8, alpha=0.7)
    ax.text(0.02, 0.052, '5% ESS floor', fontsize=6, color='gray', va='bottom',
            transform=ax.get_yaxis_transform())
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8, loc='best')
    _add_epoch_annotations_to_axes([ax], ea or [])
    _add_aggregate_ns_secondary_axis(ax, x, aggregate_ns)
    path = out / 'ess_vs_timepoints.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return str(path)


def write_convergence_plots(conv_rows: list[dict], pmf_rows: list[dict], summary_rows: list[dict], out: Path, args, warnings: list[str], *, epoch_annotations: list = None, aggregate_ns: Optional[np.ndarray] = None) -> list[str]:
    paths=[]
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        warnings.append(f'matplotlib unavailable; convergence plots skipped: {e}')
        return paths
    out.mkdir(parents=True,exist_ok=True)
    if not conv_rows:
        return paths
    ea = epoch_annotations or []
    x=np.asarray([r['frac_total'] for r in conv_rows],dtype=float)
    step=np.asarray([r['checkpoint_step'] for r in conv_rows],dtype=float)
    js=np.asarray([r.get('JS',np.nan) for r in conv_rows],dtype=float)
    rmse=np.asarray([r.get('RMSE_F_kcal_mol',np.nan) for r in conv_rows],dtype=float)
    djs=np.asarray([r.get('delta_JS',np.nan) for r in conv_rows],dtype=float)
    drmse=np.asarray([r.get('delta_RMSE_F_kcal_mol',np.nan) for r in conv_rows],dtype=float)
    berr=np.asarray([r.get('barrier_error_kcal_mol',np.nan) for r in conv_rows],dtype=float)
    occ=np.asarray([r.get('occupied_bins_frac_ref',np.nan) for r in conv_rows],dtype=float)
    newbins=np.asarray([r.get('new_bins_discovered',np.nan) for r in conv_rows],dtype=float)

    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    axes[0].plot(x,js,marker='o',lw=1.8)
    axes[0].axhline(float(args.convergence_js_threshold),ls='--',lw=0.9,alpha=0.6)
    axes[0].set_xlabel('fraction of production samples'); axes[0].set_ylabel('JS divergence vs final'); axes[0].set_title('PMF JS convergence')
    axes[0].grid(True,alpha=0.25)
    axes[1].plot(x,rmse,marker='o',lw=1.8)
    axes[1].axhline(float(args.convergence_rmse_threshold),ls='--',lw=0.9,alpha=0.6)
    axes[1].set_xlabel('fraction of production samples'); axes[1].set_ylabel('PMF RMSE vs final (kcal/mol)'); axes[1].set_title('PMF RMSE convergence')
    axes[1].grid(True,alpha=0.25)
    _add_epoch_annotations_to_axes(list(axes), ea)
    if aggregate_ns is not None:
        for _ax in axes: _add_aggregate_ns_secondary_axis(_ax, x, aggregate_ns)
    path=out/'js_rmse_vs_timepoints.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); paths.append(str(path))

    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    axes[0].plot(x,djs,marker='o',lw=1.8)
    axes[0].set_xlabel('fraction of production samples'); axes[0].set_ylabel('delta JS from previous timepoint'); axes[0].set_title('Consecutive-checkpoint JS change')
    axes[0].grid(True,alpha=0.25)
    axes[1].plot(x,drmse,marker='o',lw=1.8)
    axes[1].set_xlabel('fraction of production samples'); axes[1].set_ylabel('delta RMSE from previous (kcal/mol)'); axes[1].set_title('Consecutive-checkpoint PMF change')
    axes[1].grid(True,alpha=0.25)
    _add_epoch_annotations_to_axes(list(axes), ea)
    if aggregate_ns is not None:
        for _ax in axes: _add_aggregate_ns_secondary_axis(_ax, x, aggregate_ns)
    path=out/'delta_js_rmse_vs_timepoints.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); paths.append(str(path))

    if np.any(np.isfinite(berr)):
        fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
        ax.plot(x,berr,marker='o',lw=1.8)
        ax.set_xlabel('fraction of production samples'); ax.set_ylabel('barrier error vs final (kcal/mol)'); ax.set_title('Barrier-height convergence')
        ax.grid(True,alpha=0.25)
        _add_epoch_annotations_to_axes([ax], ea)
        if aggregate_ns is not None: _add_aggregate_ns_secondary_axis(ax, x, aggregate_ns)
        path=out/'barrier_error_vs_timepoints.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); paths.append(str(path))

    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    axes[0].plot(x,occ,marker='o',lw=1.8)
    axes[0].set_ylim(-0.03,1.03); axes[0].set_xlabel('fraction of production samples'); axes[0].set_ylabel('occupied-bin fraction of final'); axes[0].set_title('Coverage saturation')
    axes[0].grid(True,alpha=0.25)
    axes[1].step(x,newbins,where='post',lw=1.8)
    axes[1].set_xlabel('fraction of production samples'); axes[1].set_ylabel('new bins since previous timepoint'); axes[1].set_title('New CV-bin discovery')
    axes[1].grid(True,alpha=0.25)
    _add_epoch_annotations_to_axes(list(axes), ea)
    if aggregate_ns is not None:
        for _ax in axes: _add_aggregate_ns_secondary_axis(_ax, x, aggregate_ns)
    path=out/'coverage_saturation_vs_timepoints.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); paths.append(str(path))

    # scorecard: same spirit as the XVG convergence tool, but single-system.
    metrics=[('tail_median_JS','tail median JS'),('tail_median_RMSE_F_kcal_mol','tail median RMSE'),('tail_median_delta_JS','tail median dJS'),('tail_median_delta_RMSE_F_kcal_mol','tail median dRMSE')]
    if summary_rows:
        vals=np.asarray([[float(summary_rows[0].get(k,np.nan)) for k,_ in metrics]],dtype=float)
        fig,axes=plt.subplots(1,len(metrics),figsize=(max(8,2.0*len(metrics)),2.8),constrained_layout=True)
        if len(metrics)==1: axes=[axes]
        for i,(k,label) in enumerate(metrics):
            ax=axes[i]
            v=vals[:,i:i+1]
            vmax=float(np.nanmax(v)) if np.any(np.isfinite(v)) else 1.0
            im=ax.imshow(np.ma.masked_invalid(v),aspect='auto',vmin=0.0,vmax=max(vmax,1e-9),cmap='RdYlGn_r')
            ax.set_title(label,fontsize=9); ax.set_xticks([]); ax.set_yticks([0]); ax.set_yticklabels(['current'])
            if np.isfinite(vals[0,i]): ax.text(0,0,f'{vals[0,i]:.3g}',ha='center',va='center',fontsize=8)
            fig.colorbar(im,ax=ax,fraction=0.08,pad=0.03)
        fig.suptitle('Convergence scorecard',fontsize=10)
        path=out/'convergence_scorecard.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); paths.append(str(path))

    # milestone timeline for single system
    if summary_rows:
        events=[('first_frac_JS_lt_threshold','JS < thr','o'),('first_frac_RMSE_lt_threshold','RMSE < thr','s'),('first_frac_delta_JS_lt_threshold','delta JS stable','^'),('first_frac_delta_RMSE_lt_threshold','delta RMSE stable','D')]
        fig,ax=plt.subplots(figsize=(8,3.5),constrained_layout=True)
        y=0
        for i,(key,label,marker) in enumerate(events):
            v=summary_rows[0].get(key,np.nan)
            try: v=float(v)
            except Exception: v=np.nan
            if np.isfinite(v): ax.scatter([v],[y],marker=marker,s=80,label=label)
        ax.set_xlim(-0.02,1.05); ax.set_yticks([0]); ax.set_yticklabels(['current'])
        ax.set_xlabel('fraction of production samples'); ax.set_title('Convergence milestone timeline')
        for xv in (0.5,0.8,1.0): ax.axvline(xv,ls=':',lw=0.8,alpha=0.5)
        _add_epoch_annotations_to_axes([ax], ea)
        ax.grid(True,axis='x',alpha=0.25)
        if ax.get_legend_handles_labels()[0]: ax.legend(frameon=False,fontsize=8,loc='lower right')
        path=out/'milestone_timeline.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); paths.append(str(path))

    # PMF overlays at checkpoints
    if pmf_rows:
        fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
        for row in pmf_rows:
            cv=np.asarray(row['cv_A'],dtype=float); F=np.asarray(row['pmf_kcal_mol'],dtype=float)
            m=np.isfinite(F)
            if np.any(m):
                alpha=0.35+0.55*float(row.get('frac_total',1.0))
                ax.plot(cv[m],F[m],lw=1.0,alpha=alpha,label=f"{100*row.get('frac_total',1.0):.0f}%")
        ax.set_xlabel('CV distance (A)'); ax.set_ylabel('PMF (kcal/mol, shifted)'); ax.set_title('PMF convergence overlay')
        if len(pmf_rows)<=12: ax.legend(frameon=False,fontsize=7,ncol=2)
        ax.grid(True,alpha=0.2)
        path=out/'pmf_convergence_overlay.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); paths.append(str(path))
    (out/'plot_index.txt').write_text('\n'.join(paths)+'\n')
    return paths


def write_observable_convergence_plots(conv_rows: list[dict], pmf_rows: list[dict], summary_rows: list[dict], out: Path, args, warnings: list[str], *, prefix: str, metric_label: str, x_label: str, smooth_sigma: float = 0.0, epoch_annotations: list = None, aggregate_ns: Optional[np.ndarray] = None) -> list[str]:
    """Generic convergence plots for any scalar observable PMF."""
    paths=[]
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        warnings.append(f'matplotlib unavailable; {metric_label} convergence plots skipped: {e}')
        return paths
    out.mkdir(parents=True,exist_ok=True)
    if not conv_rows:
        return paths
    ea = epoch_annotations or []
    x=np.asarray([r.get('frac_total',np.nan) for r in conv_rows],dtype=float)
    js=np.asarray([r.get('JS',np.nan) for r in conv_rows],dtype=float)
    rmse=np.asarray([r.get('RMSE_F_kcal_mol',np.nan) for r in conv_rows],dtype=float)
    djs=np.asarray([r.get('delta_JS',np.nan) for r in conv_rows],dtype=float)
    drmse=np.asarray([r.get('delta_RMSE_F_kcal_mol',np.nan) for r in conv_rows],dtype=float)
    berr=np.asarray([r.get('barrier_error_kcal_mol',np.nan) for r in conv_rows],dtype=float)
    occ=np.asarray([r.get('occupied_bins_frac_ref',np.nan) for r in conv_rows],dtype=float)
    newbins=np.asarray([r.get('new_bins_discovered',np.nan) for r in conv_rows],dtype=float)

    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    axes[0].plot(x,js,marker='o',lw=1.8)
    axes[0].axhline(float(args.convergence_js_threshold),ls='--',lw=0.9,alpha=0.6)
    axes[0].set_xlabel('fraction of production samples'); axes[0].set_ylabel('JS divergence vs final'); axes[0].set_title(f'{metric_label} PMF JS convergence')
    axes[0].grid(True,alpha=0.25)
    axes[1].plot(x,rmse,marker='o',lw=1.8)
    axes[1].axhline(float(args.convergence_rmse_threshold),ls='--',lw=0.9,alpha=0.6)
    axes[1].set_xlabel('fraction of production samples'); axes[1].set_ylabel('PMF RMSE vs final (kcal/mol)'); axes[1].set_title(f'{metric_label} PMF RMSE convergence')
    axes[1].grid(True,alpha=0.25)
    _add_epoch_annotations_to_axes(list(axes), ea)
    if aggregate_ns is not None:
        for _ax in axes: _add_aggregate_ns_secondary_axis(_ax, x, aggregate_ns)
    path=out/f'{prefix}_js_rmse_vs_timepoints.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); paths.append(str(path))

    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    axes[0].plot(x,djs,marker='o',lw=1.8)
    axes[0].set_xlabel('fraction of production samples'); axes[0].set_ylabel('delta JS from previous timepoint'); axes[0].set_title(f'{metric_label} consecutive-checkpoint JS change')
    axes[0].grid(True,alpha=0.25)
    axes[1].plot(x,drmse,marker='o',lw=1.8)
    axes[1].set_xlabel('fraction of production samples'); axes[1].set_ylabel('delta RMSE from previous (kcal/mol)'); axes[1].set_title(f'{metric_label} consecutive-checkpoint PMF change')
    axes[1].grid(True,alpha=0.25)
    _add_epoch_annotations_to_axes(list(axes), ea)
    if aggregate_ns is not None:
        for _ax in axes: _add_aggregate_ns_secondary_axis(_ax, x, aggregate_ns)
    path=out/f'{prefix}_delta_js_rmse_vs_timepoints.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); paths.append(str(path))

    if np.any(np.isfinite(berr)):
        fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
        ax.plot(x,berr,marker='o',lw=1.8)
        ax.set_xlabel('fraction of production samples'); ax.set_ylabel('barrier error vs final (kcal/mol)'); ax.set_title(f'{metric_label} barrier-height convergence')
        ax.grid(True,alpha=0.25)
        _add_epoch_annotations_to_axes([ax], ea)
        if aggregate_ns is not None: _add_aggregate_ns_secondary_axis(ax, x, aggregate_ns)
        path=out/f'{prefix}_barrier_error_vs_timepoints.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); paths.append(str(path))

    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    axes[0].plot(x,occ,marker='o',lw=1.8)
    axes[0].set_ylim(-0.03,1.03); axes[0].set_xlabel('fraction of production samples'); axes[0].set_ylabel('occupied-bin fraction of final'); axes[0].set_title(f'{metric_label} coverage saturation')
    axes[0].grid(True,alpha=0.25)
    axes[1].step(x,newbins,where='post',lw=1.8)
    axes[1].set_xlabel('fraction of production samples'); axes[1].set_ylabel('new bins since previous timepoint'); axes[1].set_title(f'{metric_label} new-bin discovery')
    axes[1].grid(True,alpha=0.25)
    _add_epoch_annotations_to_axes(list(axes), ea)
    if aggregate_ns is not None:
        for _ax in axes: _add_aggregate_ns_secondary_axis(_ax, x, aggregate_ns)
    path=out/f'{prefix}_coverage_saturation_vs_timepoints.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); paths.append(str(path))

    if summary_rows:
        metrics=[('tail_median_JS','tail median JS'),('tail_median_RMSE_F_kcal_mol','tail median RMSE'),('tail_median_delta_JS','tail median dJS'),('tail_median_delta_RMSE_F_kcal_mol','tail median dRMSE')]
        vals=np.asarray([[float(summary_rows[0].get(k,np.nan)) for k,_ in metrics]],dtype=float)
        fig,axes=plt.subplots(1,len(metrics),figsize=(max(8,2.0*len(metrics)),2.8),constrained_layout=True)
        if len(metrics)==1: axes=[axes]
        for i,(k,label) in enumerate(metrics):
            ax=axes[i]
            v=vals[:,i:i+1]
            vmax=float(np.nanmax(v)) if np.any(np.isfinite(v)) else 1.0
            im=ax.imshow(np.ma.masked_invalid(v),aspect='auto',vmin=0.0,vmax=max(vmax,1e-9),cmap='RdYlGn_r')
            ax.set_title(label,fontsize=9); ax.set_xticks([]); ax.set_yticks([0]); ax.set_yticklabels(['current'])
            if np.isfinite(vals[0,i]): ax.text(0,0,f'{vals[0,i]:.3g}',ha='center',va='center',fontsize=8)
            fig.colorbar(im,ax=ax,fraction=0.08,pad=0.03)
        fig.suptitle(f'{metric_label} convergence scorecard',fontsize=10)
        path=out/f'{prefix}_convergence_scorecard.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); paths.append(str(path))

        events=[('first_frac_JS_lt_threshold','JS < thr','o'),('first_frac_RMSE_lt_threshold','RMSE < thr','s'),('first_frac_delta_JS_lt_threshold','delta JS stable','^'),('first_frac_delta_RMSE_lt_threshold','delta RMSE stable','D')]
        fig,ax=plt.subplots(figsize=(8,3.5),constrained_layout=True)
        y=0
        for key,label,marker in events:
            try: v=float(summary_rows[0].get(key,np.nan))
            except Exception: v=np.nan
            if np.isfinite(v): ax.scatter([v],[y],marker=marker,s=80,label=label)
        ax.set_xlim(-0.02,1.05); ax.set_yticks([0]); ax.set_yticklabels(['current'])
        ax.set_xlabel('fraction of production samples'); ax.set_title(f'{metric_label} convergence milestone timeline')
        for xv in (0.5,0.8,1.0): ax.axvline(xv,ls=':',lw=0.8,alpha=0.5)
        _add_epoch_annotations_to_axes([ax], ea)
        ax.grid(True,axis='x',alpha=0.25)
        if ax.get_legend_handles_labels()[0]: ax.legend(frameon=False,fontsize=8,loc='lower right')
        path=out/f'{prefix}_milestone_timeline.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); paths.append(str(path))

    if pmf_rows:
        fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
        for row in pmf_rows:
            xvals=np.asarray(row['x'],dtype=float); F=_smooth_pmf_1d(np.asarray(row['pmf_kcal_mol'],dtype=float),smooth_sigma)
            m=np.isfinite(F)
            if np.any(m):
                alpha=0.35+0.55*float(row.get('frac_total',1.0))
                ax.plot(xvals[m],F[m],lw=1.0,alpha=alpha,label=f"{100*row.get('frac_total',1.0):.0f}%")
        ax.set_xlabel(x_label); ax.set_ylabel('PMF (kcal/mol, shifted)'); ax.set_title(f'{metric_label} PMF convergence overlay')
        if len(pmf_rows)<=12: ax.legend(frameon=False,fontsize=7,ncol=2)
        ax.grid(True,alpha=0.2)
        path=out/f'{prefix}_pmf_convergence_overlay.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); paths.append(str(path))
    (out/'plot_index.txt').write_text('\n'.join(paths)+'\n')
    return paths


def _plot_basin_population_convergence(basin_pop_rows: list, basins: list, final_pmf: dict, out: Path, file_prefix: str, cv_label: str, warnings: list, smooth_sigma: float = 0.0) -> list:
    paths: list = []
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except Exception as e:
        warnings.append(f'matplotlib unavailable; basin population plots skipped: {e}'); return paths
    if not basin_pop_rows or not basins:
        return paths
    n_basins = len(basins)
    colors = [cm.tab10(i % 10) for i in range(n_basins)]
    # population vs convergence fraction
    fig, ax = plt.subplots(figsize=(7, 4))
    for b in basins:
        bid = b['basin_id']
        xs = [r['frac_total'] for r in basin_pop_rows if r['basin_id'] == bid]
        ys = [r['population'] for r in basin_pop_rows if r['basin_id'] == bid]
        ax.plot(xs, ys, marker='o', ms=3, lw=1.5, color=colors[bid % 10], label=f"basin {bid} ({b['center_cv_A']:.2f})")
    ax.set_xlabel('fraction of production samples'); ax.set_ylabel('basin population (integrated P)')
    ax.set_title('Basin population convergence'); ax.legend(frameon=False, fontsize=8, ncol=2); ax.grid(True, alpha=0.2)
    p = out / f'{file_prefix}_basin_populations.png'; fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig); paths.append(str(p))
    # final PMF with basin regions shaded
    cv = np.asarray(final_pmf['cv_A'], dtype=float); F = _smooth_pmf_1d(np.asarray(final_pmf['pmf'], dtype=float), smooth_sigma)
    finite_F = F[np.isfinite(F)]
    F_shifted = F - (float(np.min(finite_F)) if finite_F.size else 0.0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cv, F_shifted, 'k-', lw=1.5, label='PMF (final)')
    for b in basins:
        bid = b['basin_id']
        seg_cv = cv[b['left_bin']:b['right_bin']+1]; seg_F = F_shifted[b['left_bin']:b['right_bin']+1]
        ax.fill_between(seg_cv, 0, np.where(np.isfinite(seg_F), seg_F, 0), alpha=0.25, color=colors[bid % 10], label=f"basin {bid}")
        ax.axvline(b['center_cv_A'], ls='--', lw=0.8, color=colors[bid % 10])
    ax.set_xlabel(cv_label); ax.set_ylabel('PMF (kcal/mol, shifted)'); ax.set_title('Identified PMF basins')
    ax.legend(frameon=False, fontsize=8, ncol=2); ax.grid(True, alpha=0.2)
    p = out / f'{file_prefix}_basin_map.png'; fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig); paths.append(str(p))
    return paths


def write_observable_convergence_report(path: Path, row: dict, conv_rows: list[dict], warnings: list[str], metric_label: str) -> None:
    lines=[f'# {metric_label} PMF convergence report','']
    lines.append(f"Timepoints: **{int(row.get('n_checkpoints',0))}**")
    lines.append(f"Converged by JS/RMSE thresholds: **{'YES' if row.get('converged_bool') else 'NO'}**")
    lines.append('')
    lines.append(f"- JS threshold: `{row.get('js_threshold')}`")
    lines.append(f"- RMSE threshold: `{row.get('rmse_threshold_kcal_mol')}` kcal/mol")
    lines.append(f"- First JS below threshold: `{row.get('first_frac_JS_lt_threshold')}` fraction")
    lines.append(f"- First RMSE below threshold: `{row.get('first_frac_RMSE_lt_threshold')}` fraction")
    lines.append(f"- Tail median JS: `{row.get('tail_median_JS')}`")
    lines.append(f"- Tail median RMSE: `{row.get('tail_median_RMSE_F_kcal_mol')}` kcal/mol")
    lines.append('')
    if conv_rows:
        last=conv_rows[-1]
        lines.append('## Last checkpoint')
        lines.append(f"- step: `{last.get('checkpoint_step')}`")
        lines.append(f"- samples: `{last.get('n_samples')}`")
        lines.append(f"- JS: `{last.get('JS')}`")
        lines.append(f"- RMSE: `{last.get('RMSE_F_kcal_mol')}` kcal/mol")
        lines.append(f"- occupied bins: `{last.get('occupied_bins')}/{last.get('occupied_bins_ref')}`")
        lines.append('')
    lines.append('## Warnings')
    if warnings:
        lines.extend([f'- {w}' for w in warnings])
    else:
        lines.append('- No convergence-specific warnings.')
    path.write_text('\n'.join(lines)+'\n')
