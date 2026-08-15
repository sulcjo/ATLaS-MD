"""CSV/NPZ serialization of already-built PMF/FES dicts, plus the generic
_write_csv_rows list-of-dicts writer used by Plans A6b/A6c/A6d. Relocated
verbatim from analyze_gareus_mbar.py (Plan A6a) -- no logic changes, only
module location.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np

from gareus.mbar_analysis.plotting import (
    _smooth_pmf_1d, _plot_2d_fes_multirange, _want_gamd_method, _visible_pmfs,
)

def write_2d_fes_csv(path, fes, method):
    path.parent.mkdir(parents=True, exist_ok=True)
    x=np.asarray(fes['cv_A'],dtype=np.float64)
    y=np.asarray(fes['rg_A'],dtype=np.float64)
    prob=np.asarray(fes['prob'],dtype=np.float64)
    pmf=np.asarray(fes['pmf'],dtype=np.float64)
    counts=np.asarray(fes['counts'])
    with path.open('w', newline='') as f:
        wr=csv.DictWriter(f, fieldnames=['method','cv_bin','rg_bin','cv_A','rg_A','probability','pmf_kcal_mol','counts'])
        wr.writeheader()
        for i,xc in enumerate(x):
            for j,yc in enumerate(y):
                wr.writerow({
                    'method':method,
                    'cv_bin':i,
                    'rg_bin':j,
                    'cv_A':float(xc),
                    'rg_A':float(yc),
                    'probability':float(prob[i,j]) if np.isfinite(prob[i,j]) else '',
                    'pmf_kcal_mol':float(pmf[i,j]) if np.isfinite(pmf[i,j]) else '',
                    'counts':int(counts[i,j]),
                })

def write_2d_fes_npz(path, fes, method):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        method=np.asarray([method]),
        cv_A=np.asarray(fes['cv_A'],dtype=np.float64),
        rg_A=np.asarray(fes['rg_A'],dtype=np.float64),
        cv_edges_A=np.asarray(fes['cv_edges_A'],dtype=np.float64),
        rg_edges_A=np.asarray(fes['rg_edges_A'],dtype=np.float64),
        probability=np.asarray(fes['prob'],dtype=np.float64),
        pmf_kcal_mol=np.asarray(fes['pmf'],dtype=np.float64),
        counts=np.asarray(fes['counts'],dtype=np.int64),
    )

def write_cv1_cv2_2d_fes_csv(path, fes, method):
    path.parent.mkdir(parents=True, exist_ok=True)
    x=np.asarray(fes['cv_A'],dtype=np.float64)
    y=np.asarray(fes['rg_A'],dtype=np.float64)
    prob=np.asarray(fes['prob'],dtype=np.float64)
    pmf=np.asarray(fes['pmf'],dtype=np.float64)
    counts=np.asarray(fes['counts'])
    with path.open('w', newline='') as f:
        wr=csv.DictWriter(f, fieldnames=['method','cv_bin','cv2_bin','cv_A','cv2_A','probability','pmf_kcal_mol','counts'])
        wr.writeheader()
        for i,xc in enumerate(x):
            for j,yc in enumerate(y):
                wr.writerow({'method':method,'cv_bin':i,'cv2_bin':j,'cv_A':float(xc),'cv2_A':float(yc),'probability':float(prob[i,j]) if np.isfinite(prob[i,j]) else '','pmf_kcal_mol':float(pmf[i,j]) if np.isfinite(pmf[i,j]) else '','counts':int(counts[i,j])})

def write_cv1_cv2_2d_fes_npz(path, fes, method):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        method=np.asarray([method]),
        cv_A=np.asarray(fes['cv_A'],dtype=np.float64),
        cv2_A=np.asarray(fes['rg_A'],dtype=np.float64),
        cv_edges_A=np.asarray(fes['cv_edges_A'],dtype=np.float64),
        cv2_edges_A=np.asarray(fes['rg_edges_A'],dtype=np.float64),
        probability=np.asarray(fes['prob'],dtype=np.float64),
        pmf_kcal_mol=np.asarray(fes['pmf'],dtype=np.float64),
        counts=np.asarray(fes['counts'],dtype=np.int64),
    )

def _write_csv_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text('')
        return
    fields=[]
    seen=set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fields.append(k); seen.add(k)
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

def write_rg_pmf(path,pmf,method,extra=None):
    extra=extra or {}; path.parent.mkdir(parents=True,exist_ok=True)
    fields=['method','bin','rg_A','probability','pmf_kcal_mol','counts']+list(extra.keys())
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fields); wr.writeheader()
        for i,x in enumerate(pmf['cv_A']):
            row={'method':method,'bin':i,'rg_A':float(x),'probability':float(pmf['prob'][i]),'pmf_kcal_mol':float(pmf['pmf'][i]) if np.isfinite(pmf['pmf'][i]) else '', 'counts':int(pmf['counts'][i])}
            for k,a in extra.items(): row[k]=float(a[i]) if np.isfinite(a[i]) else ''
            wr.writerow(row)

def write_rg_all(path,pmfs):
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=['method','bin','rg_A','probability','pmf_kcal_mol','counts']); wr.writeheader()
        for name,p in pmfs.items():
            for i,x in enumerate(p['cv_A']):
                wr.writerow({'method':name,'bin':i,'rg_A':float(x),'probability':float(p['prob'][i]),'pmf_kcal_mol':float(p['pmf'][i]) if np.isfinite(p['pmf'][i]) else '', 'counts':int(p['counts'][i])})

def write_pca_2d_fes_csv(path: Path, fes: dict, method: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x=np.asarray(fes['cv_A'],dtype=np.float64)
    y=np.asarray(fes['rg_A'],dtype=np.float64)
    prob=np.asarray(fes['prob'],dtype=np.float64)
    pmf=np.asarray(fes['pmf'],dtype=np.float64)
    counts=np.asarray(fes['counts'])
    with path.open('w', newline='') as f:
        wr=csv.DictWriter(f, fieldnames=['method','pca1_bin','pca2_bin','pca1_A','pca2_A','probability','pmf_kcal_mol','counts'])
        wr.writeheader()
        for i,xc in enumerate(x):
            for j,yc in enumerate(y):
                wr.writerow({'method':method,'pca1_bin':i,'pca2_bin':j,'pca1_A':float(xc),'pca2_A':float(yc),'probability':float(prob[i,j]) if np.isfinite(prob[i,j]) else '', 'pmf_kcal_mol':float(pmf[i,j]) if np.isfinite(pmf[i,j]) else '', 'counts':int(counts[i,j])})

def write_pca_2d_fes_npz(path: Path, fes: dict, method: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, method=np.asarray([method]), pca1_A=np.asarray(fes['cv_A'],dtype=np.float64), pca2_A=np.asarray(fes['rg_A'],dtype=np.float64), pca1_edges_A=np.asarray(fes['cv_edges_A'],dtype=np.float64), pca2_edges_A=np.asarray(fes['rg_edges_A'],dtype=np.float64), probability=np.asarray(fes['prob'],dtype=np.float64), pmf_kcal_mol=np.asarray(fes['pmf'],dtype=np.float64), counts=np.asarray(fes['counts'],dtype=np.int64))

def _write_scalar_pmfs(out_dir: Path, prefix: str, label: str, xlabel: str, pmfs: dict, selected_method: str, warnings: list[str], smooth_sigma: float = 0.0, args=None) -> dict:
    out_dir.mkdir(parents=True,exist_ok=True)
    all_path=out_dir/f'{prefix}_pmf_all_methods.csv'
    selected_path=out_dir/f'{prefix}_pmf_unbiased.csv'
    fields=['method','bin','x','probability','pmf_kcal_mol','counts']
    with all_path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fields); wr.writeheader()
        for method,p in pmfs.items():
            for i,x in enumerate(p['x']):
                wr.writerow({'method':method,'bin':i,'x':float(x),'probability':float(p['prob'][i]) if np.isfinite(p['prob'][i]) else '', 'pmf_kcal_mol':float(p['pmf'][i]) if np.isfinite(p['pmf'][i]) else '', 'counts':int(p['counts'][i])})
    sel=pmfs.get(selected_method,pmfs.get('umbrella_only'))
    with selected_path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fields); wr.writeheader()
        for i,x in enumerate(sel['x']):
            wr.writerow({'method':selected_method,'bin':i,'x':float(x),'probability':float(sel['prob'][i]) if np.isfinite(sel['prob'][i]) else '', 'pmf_kcal_mol':float(sel['pmf'][i]) if np.isfinite(sel['pmf'][i]) else '', 'counts':int(sel['counts'][i])})
    png=out_dir/f'{prefix}_pmf.png'
    try:
        import matplotlib.pyplot as plt
        import gareus.mbar_analysis.plotstyle as ps
        fig,ax=plt.subplots(figsize=(8,5))
        for i,(method,p) in enumerate(_visible_pmfs(pmfs, selected_method, args).items()):
            pmf_plot=_smooth_pmf_1d(p['pmf'],smooth_sigma); m=np.isfinite(pmf_plot)
            if np.any(m):
                ps.plot_method_curve(ax,p['x'][m],pmf_plot[m],method,selected_method,idx=i)
        ps.style_line_axes(ax,xlabel=xlabel,ylabel='PMF (kcal/mol, shifted)',title=label)
        fig.tight_layout(); fig.savefig(png,dpi=200); plt.close(fig)
    except Exception as exc:
        warnings.append(f'Could not plot {label}: {exc}')
    finite=sel['pmf'][np.isfinite(sel['pmf'])]
    min_x=None; span=float('nan')
    if finite.size:
        mi=int(np.nanargmin(sel['pmf'])); min_x=float(sel['x'][mi]); span=float(np.nanmax(finite)-np.nanmin(finite))
    return {'selected_method':selected_method,'minimum_x':min_x,'span_kcal_mol':span,'files':{f'{prefix}_pmf_unbiased_csv':str(selected_path),f'{prefix}_pmf_all_methods_csv':str(all_path),f'{prefix}_pmf_png':str(png)}}

def _write_generic_2d_fes(out_dir: Path, prefix: str, title: str, xlabel: str, ylabel: str, pmfs2d: dict, selected_method: str, warnings: list[str], x_field: str='x', y_field: str='y', x_unit: str='', y_unit: str='', smooth_sigma: float = 1.0, args=None) -> dict:
    out_dir.mkdir(parents=True,exist_ok=True)
    sel=pmfs2d.get(selected_method,pmfs2d.get('umbrella_only'))
    csv_path=out_dir/f'{prefix}_2d_fes.csv'
    npz_path=out_dir/f'{prefix}_2d_fes.npz'
    png_path=out_dir/f'{prefix}_2d_fes.png'
    with csv_path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=['method',f'{x_field}_bin',f'{y_field}_bin',x_field,y_field,'probability','pmf_kcal_mol','counts'])
        wr.writeheader()
        for i,x in enumerate(sel['x']):
            for j,y in enumerate(sel['y']):
                wr.writerow({'method':selected_method,f'{x_field}_bin':i,f'{y_field}_bin':j,x_field:float(x),y_field:float(y),'probability':float(sel['prob'][i,j]) if np.isfinite(sel['prob'][i,j]) else '', 'pmf_kcal_mol':float(sel['pmf'][i,j]) if np.isfinite(sel['pmf'][i,j]) else '', 'counts':int(sel['counts'][i,j])})
    np.savez_compressed(npz_path, method=np.asarray([selected_method]), x=sel['x'], y=sel['y'], xedges=sel['xedges'], yedges=sel['yedges'], probability=sel['prob'], pmf_kcal_mol=sel['pmf'], counts=sel['counts'], x_label=np.asarray([xlabel]), y_label=np.asarray([ylabel]), x_field=np.asarray([x_field]), y_field=np.asarray([y_field]), x_unit=np.asarray([x_unit]), y_unit=np.asarray([y_unit]))
    plot_files=_plot_2d_fes_multirange(sel['pmf'],sel['xedges'],sel['yedges'],sel['x'],sel['y'],png_path,title,xlabel,ylabel,warnings,smooth_sigma=smooth_sigma,figsize=(8.0,6.2),dpi=210,cmap_name='viridis',contour=True)
    finite=sel['pmf'][np.isfinite(sel['pmf'])]
    min_x=min_y=None; span=float('nan')
    if finite.size:
        mi=np.unravel_index(np.nanargmin(np.where(np.isfinite(sel['pmf']),sel['pmf'],np.inf)),sel['pmf'].shape)
        min_x=float(sel['x'][mi[0]]); min_y=float(sel['y'][mi[1]]); span=float(np.nanmax(finite)-np.nanmin(finite))
    files={'csv':str(csv_path),'npz':str(npz_path),'png':str(png_path)}
    files.update({f'png_{k}':v for k,v in (plot_files or {}).items()})
    for cum_method in ('gamd_cumulant2','gamd_cumulant3'):
        if not _want_gamd_method(cum_method, selected_method, args):
            continue
        sel_cum=pmfs2d.get(cum_method)
        if sel_cum is None:
            continue
        cum_tag=cum_method.replace('gamd_','')
        csv_path_cum=out_dir/f'{prefix}_2d_fes_{cum_tag}.csv'
        npz_path_cum=out_dir/f'{prefix}_2d_fes_{cum_tag}.npz'
        png_path_cum=out_dir/f'{prefix}_2d_fes_{cum_tag}.png'
        with csv_path_cum.open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=['method',f'{x_field}_bin',f'{y_field}_bin',x_field,y_field,'probability','pmf_kcal_mol','counts'])
            wr.writeheader()
            for i,x in enumerate(sel_cum['x']):
                for j,y in enumerate(sel_cum['y']):
                    wr.writerow({'method':cum_method,f'{x_field}_bin':i,f'{y_field}_bin':j,x_field:float(x),y_field:float(y),'probability':float(sel_cum['prob'][i,j]) if np.isfinite(sel_cum['prob'][i,j]) else '', 'pmf_kcal_mol':float(sel_cum['pmf'][i,j]) if np.isfinite(sel_cum['pmf'][i,j]) else '', 'counts':int(sel_cum['counts'][i,j])})
        np.savez_compressed(npz_path_cum, method=np.asarray([cum_method]), x=sel_cum['x'], y=sel_cum['y'], xedges=sel_cum['xedges'], yedges=sel_cum['yedges'], probability=sel_cum['prob'], pmf_kcal_mol=sel_cum['pmf'], counts=sel_cum['counts'], x_label=np.asarray([xlabel]), y_label=np.asarray([ylabel]), x_field=np.asarray([x_field]), y_field=np.asarray([y_field]), x_unit=np.asarray([x_unit]), y_unit=np.asarray([y_unit]))
        plot_files_cum=_plot_2d_fes_multirange(sel_cum['pmf'],sel_cum['xedges'],sel_cum['yedges'],sel_cum['x'],sel_cum['y'],png_path_cum,title.rsplit(' (',1)[0]+f' ({cum_method})',xlabel,ylabel,warnings,smooth_sigma=smooth_sigma,figsize=(8.0,6.2),dpi=210,cmap_name='viridis',contour=True)
        files[f'{cum_tag}_csv']=str(csv_path_cum); files[f'{cum_tag}_npz']=str(npz_path_cum); files[f'{cum_tag}_png']=str(png_path_cum)
        files.update({f'{cum_tag}_png_{k}':v for k,v in (plot_files_cum or {}).items()})
    return {'selected_method':selected_method,'minimum_x':min_x,'minimum_y':min_y,'span_kcal_mol':span,'plot_ranges_kcal_mol':['0-2','0-5','0-10','0-20','0-all'],'files':files,'x_label':xlabel,'y_label':ylabel,'x_field':x_field,'y_field':y_field}

def _write_rama_2d(out_dir: Path, residue_label: str, pmfs2d: dict, selected_method: str, warnings: list[str], smooth_sigma: float = 1.0, args=None) -> dict:
    out_dir.mkdir(parents=True,exist_ok=True)
    from analyze_gareus_mbar import _slug
    slug=_slug(residue_label)
    sel=pmfs2d.get(selected_method,pmfs2d.get('umbrella_only'))
    csv_path=out_dir/f'rama_{slug}_2d_fes.csv'
    npz_path=out_dir/f'rama_{slug}_2d_fes.npz'
    png_path=out_dir/f'rama_{slug}_2d_fes.png'
    with csv_path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=['method','phi_bin','psi_bin','phi_deg','psi_deg','probability','pmf_kcal_mol','counts'])
        wr.writeheader()
        for i,x in enumerate(sel['x']):
            for j,y in enumerate(sel['y']):
                wr.writerow({'method':selected_method,'phi_bin':i,'psi_bin':j,'phi_deg':float(x),'psi_deg':float(y),'probability':float(sel['prob'][i,j]) if np.isfinite(sel['prob'][i,j]) else '', 'pmf_kcal_mol':float(sel['pmf'][i,j]) if np.isfinite(sel['pmf'][i,j]) else '', 'counts':int(sel['counts'][i,j])})
    np.savez_compressed(npz_path, method=np.asarray([selected_method]), phi_deg=sel['x'], psi_deg=sel['y'], phi_edges_deg=sel['xedges'], psi_edges_deg=sel['yedges'], probability=sel['prob'], pmf_kcal_mol=sel['pmf'], counts=sel['counts'])
    plot_files=_plot_2d_fes_multirange(sel['pmf'],sel['xedges'],sel['yedges'],sel['x'],sel['y'],png_path,f'Ramachandran FES {residue_label} ({selected_method})','phi (deg)','psi (deg)',warnings,smooth_sigma=smooth_sigma,figsize=(6.6,5.8),dpi=200,cmap_name='viridis',contour=True)
    finite=sel['pmf'][np.isfinite(sel['pmf'])]
    min_phi=min_psi=None; span=float('nan')
    if finite.size:
        mi=np.unravel_index(np.nanargmin(np.where(np.isfinite(sel['pmf']),sel['pmf'],np.inf)),sel['pmf'].shape)
        min_phi=float(sel['x'][mi[0]]); min_psi=float(sel['y'][mi[1]]); span=float(np.nanmax(finite)-np.nanmin(finite))
    files={'csv':str(csv_path),'npz':str(npz_path),'png':str(png_path)}
    files.update({f'png_{k}':v for k,v in plot_files.items()})
    for cum_method in ('gamd_cumulant2','gamd_cumulant3'):
        if not _want_gamd_method(cum_method, selected_method, args):
            continue
        sel_cum=pmfs2d.get(cum_method)
        if sel_cum is None:
            continue
        cum_tag=cum_method.replace('gamd_','')
        csv_path_cum=out_dir/f'rama_{slug}_2d_fes_{cum_tag}.csv'
        npz_path_cum=out_dir/f'rama_{slug}_2d_fes_{cum_tag}.npz'
        png_path_cum=out_dir/f'rama_{slug}_2d_fes_{cum_tag}.png'
        with csv_path_cum.open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=['method','phi_bin','psi_bin','phi_deg','psi_deg','probability','pmf_kcal_mol','counts'])
            wr.writeheader()
            for i,x in enumerate(sel_cum['x']):
                for j,y in enumerate(sel_cum['y']):
                    wr.writerow({'method':cum_method,'phi_bin':i,'psi_bin':j,'phi_deg':float(x),'psi_deg':float(y),'probability':float(sel_cum['prob'][i,j]) if np.isfinite(sel_cum['prob'][i,j]) else '', 'pmf_kcal_mol':float(sel_cum['pmf'][i,j]) if np.isfinite(sel_cum['pmf'][i,j]) else '', 'counts':int(sel_cum['counts'][i,j])})
        np.savez_compressed(npz_path_cum, method=np.asarray([cum_method]), phi_deg=sel_cum['x'], psi_deg=sel_cum['y'], phi_edges_deg=sel_cum['xedges'], psi_edges_deg=sel_cum['yedges'], probability=sel_cum['prob'], pmf_kcal_mol=sel_cum['pmf'], counts=sel_cum['counts'])
        plot_files_cum=_plot_2d_fes_multirange(sel_cum['pmf'],sel_cum['xedges'],sel_cum['yedges'],sel_cum['x'],sel_cum['y'],png_path_cum,f'Ramachandran FES {residue_label} ({cum_method})','phi (deg)','psi (deg)',warnings,smooth_sigma=smooth_sigma,figsize=(6.6,5.8),dpi=200,cmap_name='viridis',contour=True)
        files[f'{cum_tag}_csv']=str(csv_path_cum); files[f'{cum_tag}_npz']=str(npz_path_cum); files[f'{cum_tag}_png']=str(png_path_cum)
        files.update({f'{cum_tag}_png_{k}':v for k,v in (plot_files_cum or {}).items()})
    return {'residue':residue_label,'selected_method':selected_method,'minimum_phi_deg':min_phi,'minimum_psi_deg':min_psi,'span_kcal_mol':span,'plot_ranges_kcal_mol':['0-2','0-5','0-10','0-20','0-all'],'files':files}

def write_pmf(path,pmf,method,extra=None):
    extra=extra or {}; path.parent.mkdir(parents=True,exist_ok=True)
    fields=['method','bin','cv_A','probability','pmf_kcal_mol','counts']+list(extra.keys())
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fields); wr.writeheader()
        for i,x in enumerate(pmf['cv_A']):
            row={'method':method,'bin':i,'cv_A':float(x),'probability':float(pmf['prob'][i]),'pmf_kcal_mol':float(pmf['pmf'][i]) if np.isfinite(pmf['pmf'][i]) else '', 'counts':int(pmf['counts'][i])}
            for k,a in extra.items(): row[k]=float(a[i]) if np.isfinite(a[i]) else ''
            wr.writerow(row)

def write_all(path,pmfs):
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=['method','bin','cv_A','probability','pmf_kcal_mol','counts']); wr.writeheader()
        for name,p in pmfs.items():
            for i,x in enumerate(p['cv_A']):
                wr.writerow({'method':name,'bin':i,'cv_A':float(x),'probability':float(p['prob'][i]),'pmf_kcal_mol':float(p['pmf'][i]) if np.isfinite(p['pmf'][i]) else '', 'counts':int(p['counts'][i])})

def write_cv2_pmf(path, pmf, method, extra=None):
    extra=extra or {}; path.parent.mkdir(parents=True, exist_ok=True)
    fields=['method','bin','cv2_A','probability','pmf_kcal_mol','counts']+list(extra.keys())
    with path.open('w', newline='') as f:
        wr=csv.DictWriter(f, fieldnames=fields); wr.writeheader()
        for i,x in enumerate(pmf['cv_A']):
            row={'method':method,'bin':i,'cv2_A':float(x),'probability':float(pmf['prob'][i]),'pmf_kcal_mol':float(pmf['pmf'][i]) if np.isfinite(pmf['pmf'][i]) else '','counts':int(pmf['counts'][i])}
            for k,a in extra.items(): row[k]=float(a[i]) if np.isfinite(a[i]) else ''
            wr.writerow(row)
