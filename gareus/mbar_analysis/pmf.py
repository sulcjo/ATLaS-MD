"""PMF construction: weighted histograms, GaMD cumulant expansion (2nd/3rd
order reweighting correction), boost-distribution statistics, and the
fixed-f_k block-bootstrap uncertainty machinery.

Relocated verbatim from analyze_gareus_mbar.py by Plan A4 of the
mbar-analysis-modularization sequence (see
docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a4-design.md).
This is a byte-faithful move of already-audited numerical code -- no
formula, guard, or default was changed. run_pmf_and_gamd_boost_report,
run_secondary_cv_analyses, and analyze_secondary_cv_pmf (added by this same
plan's later tasks) and their _bridge()-resolved dependencies on the
not-yet-migrated remainder of analyze_gareus_mbar.py are appended below this
header by those tasks; see _bridge()'s own docstring for why a plain
`import analyze_gareus_mbar` is not used anywhere in this module.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from gareus.mbar_analysis.data import Data


def _bridge() -> Any:
    """Resolve the not-yet-migrated remainder of analyze_gareus_mbar.py.

    Temporary, explicit strangler-fig bridge (see
    docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a1-design.md
    and this module's own -a4-design.md): functions below call back into
    analyze_gareus_mbar.py for names that belong to a different,
    not-yet-executed plan in this sequence (data loading/masking -> A2, MBAR
    solving -> A3, histogram-overlap diagnostics -> A5, writers/plotting/
    2D-FES-family/convergence -> A6) or that no plan in the A1-A6
    decomposition explicitly owns (_eff_smooth, _smooth_pmf_1d,
    _secondary_cv_label, _secondary_cv_regions, _visible_pmfs -- see the
    design doc's Out Of Scope).

    A plain `import analyze_gareus_mbar` is NOT safe here: running
    `python analyze_gareus_mbar.py <run_dir>` (the still-current, unchanged
    invocation) loads that file as `__main__`, not as a module named
    `analyze_gareus_mbar` -- `sys.modules` has no entry under that name in
    that case. A bare import would then load a SECOND, independent copy of
    the whole script, with its own never-`parse_args`-mutated globals
    (DEFAULT_MBAR_BACKEND, SAMBAR_*, MBAR_ANDERSON_HISTORY) -- silently
    diverging from whatever the user actually passed on the command line.
    This resolver prefers whichever copy is already loaded, `__main__`
    included, and only falls back to a fresh import (which Python then
    caches under the real module name for any subsequent call) when neither
    is present -- e.g. a test or notebook that imports
    `gareus.mbar_analysis.pmf` directly without ever loading the script.
    """
    mod = sys.modules.get('analyze_gareus_mbar')
    if mod is not None:
        return mod
    main_mod = sys.modules.get('__main__')
    if str(getattr(main_mod, '__file__', '')).endswith('analyze_gareus_mbar.py'):
        return main_mod
    import analyze_gareus_mbar as mod
    return mod


# --- Relocated verbatim from analyze_gareus_mbar.py (lines 554-1020 at the
# commit this plan was written against -- see Task 1, Step 1 for how to
# re-derive the range if it has shifted). ---
def make_bins(cv,bins,lo,hi):
    lo=float(np.nanmin(cv) if lo is None else lo); hi=float(np.nanmax(cv) if hi is None else hi)
    pad=0.02*max(1.0,hi-lo)
    if lo==hi: hi=lo+1.0
    return np.linspace(lo-pad if lo is None else lo, hi+pad if hi is None else hi, bins+1)

def _bin_indices(values, edges):
    """0-based bin index of each value against `edges`, using the same
    half-open-on-the-left/closed-on-the-right convention as np.histogram
    (the last bin includes its right edge). Values outside
    [edges[0], edges[-1]] land on an out-of-range index (-1 or
    len(edges)-1); callers must mask those out via
    ``(bi >= 0) & (bi < len(edges) - 1)`` before using bi to index a
    length-(len(edges)-1) array. Verified (see tests) to reproduce
    np.histogram's own bin assignment bit-for-bit, so
    ``np.bincount(bi[inrange], minlength=B)`` on unweighted data is
    interchangeable with ``np.histogram(values, bins=edges)[0]``.
    """
    B=len(edges)-1
    bi=np.searchsorted(edges,values,side='right')-1
    bi[values==edges[-1]]=B-1
    return bi

def pmf_from_weights(cv,w,bins,kbt_kcal):
    cv=np.asarray(cv,dtype=np.float64); w=np.asarray(w,dtype=np.float64)
    prob,edges=np.histogram(cv,bins=bins,weights=w)
    # counts must reflect samples that actually contribute to `prob` (finite,
    # positive weight) -- a bin populated only by zero/NaN-weight samples has
    # true reweighted probability 0 and must NOT read as "occupied" to
    # consumers like the occupied_bins convergence diagnostic. Reuse `edges`
    # (not `bins`) so counts stays aligned with prob even if bins was an int.
    valid=np.isfinite(w)&(w>0)
    B=len(edges)-1
    cv_valid=cv[valid]
    bi=_bin_indices(cv_valid,edges)
    inrange=(bi>=0)&(bi<B)
    counts=np.bincount(bi[inrange],minlength=B)
    prob=np.asarray(prob,float)
    if np.sum(prob)>0: prob/=np.sum(prob)
    with np.errstate(divide='ignore',invalid='ignore'): F=-kbt_kcal*np.log(prob)
    mask=np.isfinite(F)
    if np.any(mask): F-=np.nanmin(F[mask])
    return {'cv_A':0.5*(edges[:-1]+edges[1:]),'prob':prob,'pmf':F,'counts':counts.astype(int)}

def _cumulant_shared_stats(cv,base_w,boost,bins):
    """The O(N) work shared by the order-2 and order-3 cumulant expansions:
    histogram/bin-edges, per-bin unweighted counts, bin-index assignment,
    and the per-bin boost mean/variance (order=2's full computation).
    order=3 adds exactly one more O(N) bincount (kappa3) on top of this;
    nothing in this shared stage depends on which order is requested.
    """
    cv=np.asarray(cv,dtype=np.float64)
    base_w=np.asarray(base_w,dtype=np.float64)
    boost=np.asarray(boost,dtype=np.float64)
    p0,edges=np.histogram(cv,bins=bins,weights=base_w)
    centers=0.5*(edges[:-1]+edges[1:])
    B=centers.size
    bi=_bin_indices(cv,edges)
    inrange=(bi>=0)&(bi<B)
    counts=np.bincount(bi[inrange],minlength=B)
    good=inrange&np.isfinite(boost)&np.isfinite(base_w)&(base_w>0)
    mean=np.full(B,np.nan,dtype=np.float64)
    var=np.full(B,np.nan,dtype=np.float64)
    nz=np.zeros(B,dtype=bool)
    idx=w=x=dx=None
    sw=np.zeros(B,dtype=np.float64)
    if np.any(good):
        idx=bi[good].astype(np.int64,copy=False)
        w=base_w[good]
        x=boost[good]
        sw=np.bincount(idx,weights=w,minlength=B).astype(np.float64)
        sx=np.bincount(idx,weights=w*x,minlength=B).astype(np.float64)
        nz=sw>0
        mean[nz]=sx[nz]/sw[nz]
        dx=x-mean[idx]
        sdx2=np.bincount(idx,weights=w*dx*dx,minlength=B).astype(np.float64)
        var[nz]=np.maximum(0.0,sdx2[nz]/sw[nz])
    return {'edges':edges,'centers':centers,'B':B,'p0':p0,'counts':counts,
            'nz':nz,'mean':mean,'var':var,'idx':idx,'w':w,'x':x,'dx':dx,'sw':sw}


def _cumulant_from_shared(shared,beta,kbt_kcal,order,smooth_logfac_sigma=0.0):
    """Finish an order-2 or order-3 cumulant PMF from `_cumulant_shared_stats`
    output. order=2 keeps the mean+variance terms (Gaussian/CE2
    approximation). order=3 adds the beta^3/6 * kappa3 term, where kappa3
    is the per-bin third cumulant (= third central moment) of the boost.
    kappa3/var are computed via a two-pass mean-centered accumulation
    rather than raw moments, since raw <x^3>-3<x^2><x>+2<x>^3
    catastrophically cancels when the boost mean (O(10-200) kJ/mol)
    dominates its spread.
    """
    B=shared['B']; nz=shared['nz']; mean=shared['mean']; var=shared['var']
    p0=shared['p0']; counts=shared['counts']; centers=shared['centers']
    kappa3=np.full(B,np.nan,dtype=np.float64)
    logfac=np.zeros(B,dtype=np.float64)
    if np.any(nz):
        logfac[nz]=beta*mean[nz]+0.5*beta*beta*var[nz]
        if order==3:
            idx=shared['idx']; w=shared['w']; dx=shared['dx']; sw=shared['sw']
            sdx3=np.bincount(idx,weights=w*dx*dx*dx,minlength=B).astype(np.float64)
            kappa3[nz]=sdx3[nz]/sw[nz]
            logfac[nz]+=(beta**3/6.0)*kappa3[nz]
    # A bin with real weighted samples (p0>0) but zero samples with a finite
    # boost has an UNKNOWN GaMD correction -- flag NaN rather than silently
    # falling back to logfac=0 (which would look like "no correction needed"
    # and reproduce the raw/unbiased-looking p0 value for that bin). Bins
    # with no samples at all (p0==0) keep logfac=0 so p=0*exp(0)=0 there,
    # matching the existing "no samples" -> F=inf -> excluded-by-isfinite
    # convention used throughout this file.
    logfac[(~nz)&(p0>0)]=np.nan
    if smooth_logfac_sigma and float(smooth_logfac_sigma) > 0:
        try:
            from scipy.ndimage import gaussian_filter1d
            logfac=gaussian_filter1d(logfac,sigma=float(smooth_logfac_sigma),mode='nearest')
        except Exception:
            pass
    p=p0*np.exp(np.clip(logfac,-700,700))
    ps=float(np.nansum(p))
    if ps>0: p/=ps
    with np.errstate(divide='ignore',invalid='ignore'):
        F=-kbt_kcal*np.log(p)
    mask=np.isfinite(F)
    if np.any(mask): F-=np.nanmin(F[mask])
    return {'cv_A':centers.copy(),'prob':p,'pmf':F,'counts':counts.astype(int)}, {'boost_mean_kj':mean.copy(),'boost_var_kj2':var.copy(),'boost_kappa3_kj3':kappa3,'log_reweight_factor':logfac}


def _cumulant_expansion(cv,base_w,boost,bins,beta,kbt_kcal,order=2,smooth_logfac_sigma=0.0):
    """Cumulant GaMD reweighting, vectorized by CV bin.

    order=2 keeps the mean+variance terms (Gaussian/CE2 approximation).
    order=3 adds the beta^3/6 * kappa3 term, where kappa3 is the per-bin
    third cumulant (= third central moment) of the boost. See
    `_cumulant_from_shared` for the reweighting-factor math and
    `_cumulant_shared_stats` for the shared histogram/mean/variance pass.
    """
    if order not in (2,3):
        raise ValueError(f"cumulant expansion order must be 2 or 3, got {order}")
    shared=_cumulant_shared_stats(cv,base_w,boost,bins)
    return _cumulant_from_shared(shared,beta,kbt_kcal,order,smooth_logfac_sigma=smooth_logfac_sigma)


def _cumulant_expansion_both(cv,base_w,boost,bins,beta,kbt_kcal,smooth_logfac_sigma=0.0):
    """Compute the order-2 AND order-3 cumulant PMFs from a single shared
    pass, for callers that need both (as every current caller of
    cumulant2+cumulant3 back-to-back does). Equivalent to calling
    `_cumulant_expansion(order=2)` then `_cumulant_expansion(order=3)`
    separately -- same bit-for-bit numbers -- but the shared O(N)
    histogram/bin-assignment/mean/variance work (the expensive part) is
    only performed once instead of twice.

    Returns ((pmf2, diag2), (pmf3, diag3)).
    """
    shared=_cumulant_shared_stats(cv,base_w,boost,bins)
    result2=_cumulant_from_shared(shared,beta,kbt_kcal,2,smooth_logfac_sigma=smooth_logfac_sigma)
    result3=_cumulant_from_shared(shared,beta,kbt_kcal,3,smooth_logfac_sigma=smooth_logfac_sigma)
    return result2, result3


def cumulant2(cv,base_w,boost,bins,beta,kbt_kcal,smooth_logfac_sigma=0.0):
    """Second-order cumulant GaMD reweighting, vectorized by CV bin."""
    return _cumulant_expansion(cv,base_w,boost,bins,beta,kbt_kcal,order=2,smooth_logfac_sigma=smooth_logfac_sigma)


def cumulant3(cv,base_w,boost,bins,beta,kbt_kcal,smooth_logfac_sigma=0.0):
    """Third-order cumulant GaMD reweighting (adds the beta^3/6 * kappa3 term)."""
    return _cumulant_expansion(cv,base_w,boost,bins,beta,kbt_kcal,order=3,smooth_logfac_sigma=smooth_logfac_sigma)


def pmf2d_from_weights(x,y,w,xbins,ybins,kbt_kcal):
    x=np.asarray(x,dtype=np.float64)
    y=np.asarray(y,dtype=np.float64)
    w=np.asarray(w,dtype=np.float64)
    prob,xedges,yedges=np.histogram2d(x,y,bins=[xbins,ybins],weights=w)
    counts,_,_=np.histogram2d(x,y,bins=[xbins,ybins])
    prob=np.asarray(prob,dtype=np.float64)
    if np.sum(prob)>0:
        prob/=np.sum(prob)
    with np.errstate(divide='ignore', invalid='ignore'):
        F=-kbt_kcal*np.log(prob)
    mask=np.isfinite(F)
    if np.any(mask):
        F-=np.nanmin(F[mask])
    return {
        'cv_A':0.5*(xedges[:-1]+xedges[1:]),
        'rg_A':0.5*(yedges[:-1]+yedges[1:]),
        'cv_edges_A':np.asarray(xedges,dtype=np.float64),
        'rg_edges_A':np.asarray(yedges,dtype=np.float64),
        'prob':prob,
        'pmf':F,
        'counts':counts.astype(int),
    }

def _cumulant_shared_stats_2d(x,y,base_w,boost,xbins,ybins):
    """2D counterpart of `_cumulant_shared_stats`; see that docstring."""
    x=np.asarray(x,dtype=np.float64)
    y=np.asarray(y,dtype=np.float64)
    base_w=np.asarray(base_w,dtype=np.float64)
    boost=np.asarray(boost,dtype=np.float64)
    p0,xedges,yedges=np.histogram2d(x,y,bins=[xbins,ybins],weights=base_w)
    xc=0.5*(xedges[:-1]+xedges[1:])
    yc=0.5*(yedges[:-1]+yedges[1:])
    Bx=len(xc); By=len(yc)
    xi=_bin_indices(x,xedges)
    yi=_bin_indices(y,yedges)
    inrange=(xi>=0)&(xi<Bx)&(yi>=0)&(yi<By)
    idx_all=xi[inrange].astype(np.int64,copy=False)*By+yi[inrange].astype(np.int64,copy=False)
    counts=np.bincount(idx_all,minlength=Bx*By).reshape(Bx,By)
    good=inrange&np.isfinite(boost)&np.isfinite(base_w)&(base_w>0)
    mean=np.full((Bx,By),np.nan,dtype=np.float64)
    var=np.full((Bx,By),np.nan,dtype=np.float64)
    nz=np.zeros((Bx,By),dtype=bool)
    idx=w=b=db=None
    sw=np.zeros((Bx,By),dtype=np.float64)
    if np.any(good):
        xf=xi[good].astype(np.int64,copy=False)
        yf=yi[good].astype(np.int64,copy=False)
        idx=xf*By+yf
        w=base_w[good]
        b=boost[good]
        sw=np.bincount(idx,weights=w,minlength=Bx*By).astype(np.float64).reshape(Bx,By)
        sb=np.bincount(idx,weights=w*b,minlength=Bx*By).astype(np.float64).reshape(Bx,By)
        nz=sw>0
        mean[nz]=sb[nz]/sw[nz]
        flat_mean=mean.reshape(-1)
        db=b-flat_mean[idx]
        sdb2=np.bincount(idx,weights=w*db*db,minlength=Bx*By).astype(np.float64).reshape(Bx,By)
        var[nz]=np.maximum(0.0,sdb2[nz]/sw[nz])
    return {'xc':xc,'yc':yc,'xedges':xedges,'yedges':yedges,'Bx':Bx,'By':By,
            'p0':p0,'counts':counts,'nz':nz,'mean':mean,'var':var,
            'idx':idx,'w':w,'b':b,'db':db,'sw':sw}


def _cumulant_from_shared_2d(shared,beta,kbt_kcal,order,smooth_logfac_sigma=0.0):
    """2D counterpart of `_cumulant_from_shared`; see that docstring."""
    Bx=shared['Bx']; By=shared['By']; nz=shared['nz']
    mean=shared['mean']; var=shared['var']; p0=shared['p0']; counts=shared['counts']
    xc=shared['xc']; yc=shared['yc']; xedges=shared['xedges']; yedges=shared['yedges']
    kappa3=np.full((Bx,By),np.nan,dtype=np.float64)
    logfac=np.zeros((Bx,By),dtype=np.float64)
    if np.any(nz):
        logfac[nz]=beta*mean[nz]+0.5*beta*beta*var[nz]
        if order==3:
            idx=shared['idx']; w=shared['w']; db=shared['db']; sw=shared['sw']
            sdb3=np.bincount(idx,weights=w*db*db*db,minlength=Bx*By).astype(np.float64).reshape(Bx,By)
            kappa3[nz]=sdb3[nz]/sw[nz]
            logfac[nz]+=(beta**3/6.0)*kappa3[nz]
    # See _cumulant_from_shared: a bin with real weighted samples (p0>0) but
    # zero finite-boost samples gets an unknown (NaN) correction, not a
    # silent logfac=0 fallback. Genuinely empty bins (p0==0) stay at
    # logfac=0 -> p=0, unchanged.
    logfac[(~nz)&(p0>0)]=np.nan
    if smooth_logfac_sigma and float(smooth_logfac_sigma) > 0:
        try:
            from scipy.ndimage import gaussian_filter
            logfac=gaussian_filter(logfac,sigma=float(smooth_logfac_sigma),mode='nearest')
        except Exception:
            pass
    p=p0*np.exp(np.clip(logfac,-700,700))
    ps=float(np.nansum(p))
    if ps>0:
        p/=ps
    with np.errstate(divide='ignore', invalid='ignore'):
        F=-kbt_kcal*np.log(p)
    mask=np.isfinite(F)
    if np.any(mask):
        F-=np.nanmin(F[mask])
    return {
        'cv_A':xc.copy(),
        'rg_A':yc.copy(),
        'cv_edges_A':np.array(xedges,dtype=np.float64),
        'rg_edges_A':np.array(yedges,dtype=np.float64),
        'prob':p,
        'pmf':F,
        'counts':counts.astype(int),
    }, {
        'boost_mean_kj':mean.copy(),
        'boost_var_kj2':var.copy(),
        'boost_kappa3_kj3':kappa3,
        'log_reweight_factor':logfac,
    }


def _cumulant_expansion_2d(x,y,base_w,boost,xbins,ybins,beta,kbt_kcal,order=2,smooth_logfac_sigma=0.0):
    """2D counterpart of _cumulant_expansion; see that docstring for order/kappa3 notes."""
    if order not in (2,3):
        raise ValueError(f"cumulant expansion order must be 2 or 3, got {order}")
    shared=_cumulant_shared_stats_2d(x,y,base_w,boost,xbins,ybins)
    return _cumulant_from_shared_2d(shared,beta,kbt_kcal,order,smooth_logfac_sigma=smooth_logfac_sigma)


def _cumulant_expansion_2d_both(x,y,base_w,boost,xbins,ybins,beta,kbt_kcal,smooth_logfac_sigma=0.0):
    """2D counterpart of `_cumulant_expansion_both`; see that docstring.
    Returns ((fes2, diag2), (fes3, diag3))."""
    shared=_cumulant_shared_stats_2d(x,y,base_w,boost,xbins,ybins)
    result2=_cumulant_from_shared_2d(shared,beta,kbt_kcal,2,smooth_logfac_sigma=smooth_logfac_sigma)
    result3=_cumulant_from_shared_2d(shared,beta,kbt_kcal,3,smooth_logfac_sigma=smooth_logfac_sigma)
    return result2, result3


def _bootstrap_pmf_uncertainty_1d(cv, logw, boost, bins, beta, kbt_kcal,
                                   window, block_ids, selected_method,
                                   main_pmf, n_boot, rng, smooth_logfac_sigma=0.0):
    """Fixed-f_k block bootstrap uncertainty for a 1D PMF (see
    docs/superpowers/specs/2026-08-13-pmf-bootstrap-uncertainty-design.md).

    For each window, resamples that window's own blocks (see
    `_sample_block_ids`) with replacement -- same number of blocks drawn as
    the window originally had, different composition -- reusing each drawn
    sample's already-computed per-sample log-weight `logw` (f_k stays
    fixed; no MBAR re-solve). Rebuilds `selected_method`'s PMF on the
    resampled set for each of `n_boot` replicates, re-anchors every
    replicate to read exactly 0 at the SAME bin `main_pmf` uses as its own
    minimum (never the replicate's own minimum -- anchoring to each
    replicate's own minimum would artificially erase uncertainty exactly at
    that bin and distort every other bin's uncertainty relative to it), and
    returns the per-bin std across replicates.

    `selected_method` must be one of 'umbrella_only', 'gamd_exponential',
    'gamd_cumulant2', 'gamd_cumulant3' -- the same keys as the `pmfs` dict
    built in `run_pmf_and_gamd_boost_report`.

    `smooth_logfac_sigma`: forwarded to `_cumulant_expansion` for the
    'gamd_cumulant2'/'gamd_cumulant3' branches only (the other two branches
    don't smooth). Callers must pass the SAME value used to build `main_pmf`
    -- otherwise the replicates describe a different (differently-smoothed)
    curve than the one `pmf_std` is meant to describe the uncertainty of.

    Returns {'pmf_std': ndarray (len(bins)-1,),
             'blocks_per_window': ndarray (K,),
             'low_block_windows': list[int]} -- windows with fewer than 3
    blocks, whose contribution to the estimate is unreliable.
    """
    _agm = _bridge()
    cv = np.asarray(cv, dtype=np.float64)
    logw = np.asarray(logw, dtype=np.float64)
    boost = np.asarray(boost, dtype=np.float64)
    window = np.asarray(window)
    block_ids = np.asarray(block_ids)
    K = int(np.max(window)) + 1 if window.size else 0
    minidx = int(np.nanargmin(main_pmf['pmf']))

    window_block_map = []
    blocks_per_window = np.zeros(K, dtype=np.int64)
    for k in range(K):
        idx_k = np.where(window == k)[0]
        blocks_k = block_ids[idx_k]
        uniq = np.unique(blocks_k)
        blocks_per_window[k] = uniq.size
        grouped = {b: idx_k[blocks_k == b] for b in uniq}
        window_block_map.append((uniq, grouped))
    low_block_windows = [k for k in range(K) if 0 < blocks_per_window[k] < 3]

    n_bins = len(bins) - 1
    reps = np.full((n_boot, n_bins), np.nan, dtype=np.float64)
    for b in range(n_boot):
        resampled_parts = []
        for k in range(K):
            uniq, grouped = window_block_map[k]
            if uniq.size == 0:
                continue
            chosen = rng.choice(uniq, size=uniq.size, replace=True)
            for blk in chosen:
                resampled_parts.append(grouped[blk])
        if not resampled_parts:
            continue
        resampled_idx = np.concatenate(resampled_parts)
        cv_b = cv[resampled_idx]
        logw_b = logw[resampled_idx]
        boost_b = boost[resampled_idx]
        if selected_method == 'umbrella_only':
            w_b = _agm.norm_logw(logw_b)
            rep_pmf = pmf_from_weights(cv_b, w_b, bins, kbt_kcal)
        elif selected_method == 'gamd_exponential':
            w_b = _agm.norm_logw(logw_b + beta * boost_b)
            rep_pmf = pmf_from_weights(cv_b, w_b, bins, kbt_kcal)
        elif selected_method in ('gamd_cumulant2', 'gamd_cumulant3'):
            base_w_b = _agm.norm_logw(logw_b)
            order = 2 if selected_method == 'gamd_cumulant2' else 3
            rep_pmf, _ = _cumulant_expansion(cv_b, base_w_b, boost_b, bins, beta, kbt_kcal, order=order, smooth_logfac_sigma=smooth_logfac_sigma)
        else:
            raise ValueError(f"unknown selected_method {selected_method!r}")
        rep_arr = np.asarray(rep_pmf['pmf'], dtype=np.float64)
        anchor = rep_arr[minidx]
        reps[b] = rep_arr - anchor

    with np.errstate(invalid='ignore'):
        pmf_std = np.nanstd(reps, axis=0)
    return {'pmf_std': pmf_std, 'blocks_per_window': blocks_per_window, 'low_block_windows': low_block_windows}


def _bootstrap_pmf_uncertainty_2d(x, y, logw, boost, xbins, ybins, beta, kbt_kcal,
                                   window, block_ids, selected_method,
                                   main_fes, n_boot, rng, smooth_logfac_sigma=0.0):
    """2D counterpart of `_bootstrap_pmf_uncertainty_1d`; see that
    docstring (including the `smooth_logfac_sigma` note). Iterates
    windows/blocks identically, so with the same `rng` state and the same
    `window`/`block_ids`, it draws the exact same resampled index sets per
    replicate as the 1D helper -- verified by the degenerate-y collapse
    test.
    """
    _agm = _bridge()
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    logw = np.asarray(logw, dtype=np.float64)
    boost = np.asarray(boost, dtype=np.float64)
    window = np.asarray(window)
    block_ids = np.asarray(block_ids)
    K = int(np.max(window)) + 1 if window.size else 0
    main_arr = np.asarray(main_fes['pmf'], dtype=np.float64)
    minidx_flat = int(np.nanargmin(main_arr.ravel()))
    minidx = np.unravel_index(minidx_flat, main_arr.shape)

    window_block_map = []
    blocks_per_window = np.zeros(K, dtype=np.int64)
    for k in range(K):
        idx_k = np.where(window == k)[0]
        blocks_k = block_ids[idx_k]
        uniq = np.unique(blocks_k)
        blocks_per_window[k] = uniq.size
        grouped = {b: idx_k[blocks_k == b] for b in uniq}
        window_block_map.append((uniq, grouped))
    low_block_windows = [k for k in range(K) if 0 < blocks_per_window[k] < 3]

    Bx, By = main_arr.shape
    reps = np.full((n_boot, Bx, By), np.nan, dtype=np.float64)
    for b in range(n_boot):
        resampled_parts = []
        for k in range(K):
            uniq, grouped = window_block_map[k]
            if uniq.size == 0:
                continue
            chosen = rng.choice(uniq, size=uniq.size, replace=True)
            for blk in chosen:
                resampled_parts.append(grouped[blk])
        if not resampled_parts:
            continue
        resampled_idx = np.concatenate(resampled_parts)
        x_b = x[resampled_idx]
        y_b = y[resampled_idx]
        logw_b = logw[resampled_idx]
        boost_b = boost[resampled_idx]
        if selected_method == 'umbrella_only':
            w_b = _agm.norm_logw(logw_b)
            rep_fes = pmf2d_from_weights(x_b, y_b, w_b, xbins, ybins, kbt_kcal)
        elif selected_method == 'gamd_exponential':
            w_b = _agm.norm_logw(logw_b + beta * boost_b)
            rep_fes = pmf2d_from_weights(x_b, y_b, w_b, xbins, ybins, kbt_kcal)
        elif selected_method in ('gamd_cumulant2', 'gamd_cumulant3'):
            base_w_b = _agm.norm_logw(logw_b)
            order = 2 if selected_method == 'gamd_cumulant2' else 3
            rep_fes, _ = _cumulant_expansion_2d(x_b, y_b, base_w_b, boost_b, xbins, ybins, beta, kbt_kcal, order=order, smooth_logfac_sigma=smooth_logfac_sigma)
        else:
            raise ValueError(f"unknown selected_method {selected_method!r}")
        rep_arr = np.asarray(rep_fes['pmf'], dtype=np.float64)
        anchor = rep_arr[minidx]
        reps[b] = rep_arr - anchor

    with np.errstate(invalid='ignore'):
        pmf_std = np.nanstd(reps, axis=0)
    return {'pmf_std': pmf_std, 'blocks_per_window': blocks_per_window, 'low_block_windows': low_block_windows}


def cumulant2_2d(x,y,base_w,boost,xbins,ybins,beta,kbt_kcal,smooth_logfac_sigma=0.0):
    return _cumulant_expansion_2d(x,y,base_w,boost,xbins,ybins,beta,kbt_kcal,order=2,smooth_logfac_sigma=smooth_logfac_sigma)


def cumulant3_2d(x,y,base_w,boost,xbins,ybins,beta,kbt_kcal,smooth_logfac_sigma=0.0):
    return _cumulant_expansion_2d(x,y,base_w,boost,xbins,ybins,beta,kbt_kcal,order=3,smooth_logfac_sigma=smooth_logfac_sigma)

def _window_cv_mean_std(window: np.ndarray, cv: np.ndarray, K: int) -> tuple:
    """Vectorized per-window sample count / mean / std of a CV array.

    Same bincount-over-linear-index idiom as overlap_matrix above, extracted
    so window_diagnostics.csv's writer (run_pmf_and_gamd_boost_report) avoids
    an O(N*K) Python loop that rebuilt a fresh boolean mask (``cv[window ==
    k]``) once per window.

    Two-pass form (mean first, then mean of squared deviations from that
    mean) mirrors np.std's own algorithm, unlike a raw-moment
    ``sum(x**2)/n - mean**2`` formulation, which can lose several digits to
    cancellation for CV values with a large mean and small within-window
    spread -- this keeps results numerically equivalent to (though not
    always bit-identical with) the original per-window np.mean/np.std.

    Returns ``(n_k, mean_per_window, std_per_window)``, each length K:
    - ``n_k``: per-window sample count (same as
      ``np.bincount(window, minlength=K)`` restricted to valid window
      indices in [0, K), matching the original loop's implicit assumption
      that every ``k`` iterated over ``range(K)`` is itself a valid index).
    - ``mean_per_window``/``std_per_window``: NaN for any window with zero
      samples (callers should treat that the same as the original's empty
      ``cv[window == k]`` slice, i.e. write '' rather than the NaN literal);
      a window containing any non-finite (e.g. NaN) cv sample also gets a
      NaN mean/std for that whole window, matching np.mean/np.std's own
      NaN-propagation behavior on such a slice.
    """
    window=np.asarray(window,dtype=np.int64)
    cv=np.asarray(cv,dtype=np.float64)
    valid=(window>=0)&(window<K)
    w_valid=window[valid]
    cv_valid=cv[valid]
    n_k=np.bincount(w_valid,minlength=K)
    has_samples=n_k>0
    mean=np.full(K,np.nan)
    sum_per_window=np.bincount(w_valid,weights=cv_valid,minlength=K)
    mean[has_samples]=sum_per_window[has_samples]/n_k[has_samples]
    dev_sq=(cv_valid-mean[w_valid])**2
    sumsq_dev=np.bincount(w_valid,weights=dev_sq,minlength=K)
    std=np.full(K,np.nan)
    std[has_samples]=np.sqrt(sumsq_dev[has_samples]/n_k[has_samples])
    return n_k,mean,std

def boost_stats(boost,beta):
    _agm = _bridge()
    b=boost[np.isfinite(boost)]
    if b.size==0: return {'available':False}
    kc=b/_agm.KJ_PER_KCAL; out={'available':True,'n':int(b.size),'mean_kcal_mol':float(np.mean(kc)),'std_kcal_mol':float(np.std(kc)),'min_kcal_mol':float(np.min(kc)),'max_kcal_mol':float(np.max(kc))}
    if b.size>=3 and np.std(b)>0:
        z=(b-np.mean(b))/np.std(b); out['skew']=float(np.mean(z**3)); out['excess_kurtosis']=float(np.mean(z**4)-3.0); out['anharmonicity_score']=float(math.sqrt(out['skew']**2+0.25*out['excess_kurtosis']**2))
    else: out.update({'skew':None,'excess_kurtosis':None,'anharmonicity_score':None})
    w=_agm.norm_logw(beta*b); out['boost_reweight_ess']=float(_agm.ess(w)); out['boost_reweight_ess_fraction']=float(out['boost_reweight_ess']/b.size)
    return out

def _window_moments(a):
    """Skewness/excess-kurtosis/anharmonicity of one window's finite boost samples."""
    a = a[np.isfinite(a)]
    if a.size < 4:
        return np.nan, np.nan, np.nan
    mu, sigma = np.mean(a), np.std(a)
    if sigma < 1e-12:
        return 0.0, 0.0, 0.0
    z = (a - mu) / sigma
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4) - 3.0)
    anharmonicity = float(np.sqrt(skew**2 + 0.25 * kurt**2))
    return skew, kurt, anharmonicity

def run_pmf_and_gamd_boost_report(d: 'Data', args, logw: np.ndarray, bins: np.ndarray,
                                   kbt_kcal: float, out: Path, warnings: list,
                                   progress: Optional['Progress'], warning_prefix: str = '',
                                   extra_pmfs: Optional[dict] = None,
                                   precomputed_base_w: Optional[np.ndarray] = None) -> dict:
    """Core PMF (4 methods) + GaMD boost diagnostics for one Data.

    Extracted from analyze() so the identical formula can run twice against
    different sample subsets sharing the same global MBAR solve (see the
    epoch_000/rest split in analyze()) instead of only ever pooling every
    epoch into one report. ``logw`` is the per-sample MBAR log-weight for
    THIS subset -- renormalized internally, same pattern as
    ``analyze_secondary_cv_pmf``. When ``d`` is a subset of the full
    population (e.g. the epoch_000/rest split), callers must pass
    ``_subset_logw_from_global_fk(d, f_k_global)`` here rather than a naive
    ``m['logw'][mask]`` slice of the global logw -- the naive slice only
    renormalizes for the subset's overall size, not for different states
    losing different fractions of their samples to the split, which
    produces a real direction-consistent tilt in the resulting PMF. When
    ``d`` is the full population, plain ``m['logw']`` is already correct.

    ``precomputed_base_w``: optional escape hatch for the common no-split
    case, where analyze() has already computed ``norm_logw(m['logw'])`` for
    the exact same full-population ``logw`` passed in here -- recomputing it
    is a deterministic no-op that still costs a real logsumexp/exp pass over
    every sample. Callers must only pass this when ``logw`` is that SAME
    full-population array (unmodified); for any genuinely different
    population (e.g. an epoch_000/rest subset with its own renormalized
    logw), pass ``None`` so it's computed fresh here instead of silently
    reusing a value for the wrong population. Copied defensively on the way
    in: analyze() keeps using its own ``base_w`` after this call returns
    (Rg analysis, ``base_ess`` in pmf_summary.json), so nothing this
    function or its callees (pmf_from_weights/cumulant2/cumulant3, all
    read-only on this array today) do to the local name here can ever reach
    back and corrupt that array -- a guarantee worth the one extra O(N)
    copy, still far cheaper than the norm_logw() this parameter exists to
    skip (isfinite mask + logsumexp + exp over every sample).
    """
    _agm = _bridge()
    K = d.u_nk.shape[1]
    N = len(d.cv)
    if precomputed_base_w is not None:
        base_w = np.array(precomputed_base_w, dtype=np.float64, copy=True)
    else:
        base_w = _agm.norm_logw(np.asarray(logw, dtype=np.float64))
    umbrella = pmf_from_weights(d.cv, base_w, bins, kbt_kcal)
    bs = boost_stats(d.boost_kj, d.beta)
    boost_ok = bool(bs.get('available')) and np.nanstd(d.boost_kj) > 1e-12
    if boost_ok:
        exp_w = _agm.norm_logw(logw + d.beta * d.boost_kj)
        exp_pmf = pmf_from_weights(d.cv, exp_w, bins, kbt_kcal)
        (cum_pmf, cdiag), (cum3_pmf, cdiag3) = _cumulant_expansion_both(d.cv, base_w, d.boost_kj, bins, d.beta, kbt_kcal, smooth_logfac_sigma=_agm._eff_smooth(args, 'gamd_smooth_sigma'))
        selected = 'gamd_cumulant2'
        # A bin can have real samples (counts>0) but zero with a finite GaMD
        # boost -- e.g. a whole segment/epoch missing gamd_boost_total_kj_mol
        # dominating that CV bin. _cumulant_expansion flags this NaN rather
        # than silently reproducing the unbiased value; surface it here since
        # this is the main choke point with a `warnings` list in scope.
        nan_bins2 = np.isnan(cdiag['log_reweight_factor']) & (np.asarray(cum_pmf['counts']) > 0)
        if np.any(nan_bins2):
            warnings.append(f"{warning_prefix}GaMD cumulant2 correction is undefined (NaN) for {int(np.sum(nan_bins2))} CV bin(s) with samples but no finite boost values; those pmf_gamd_cumulant2 bins are NaN.")
        nan_bins3 = np.isnan(cdiag3['log_reweight_factor']) & (np.asarray(cum3_pmf['counts']) > 0)
        if np.any(nan_bins3):
            warnings.append(f"{warning_prefix}GaMD cumulant3 correction is undefined (NaN) for {int(np.sum(nan_bins3))} CV bin(s) with samples but no finite boost values; those pmf_gamd_cumulant3 bins are NaN.")
        e = _agm.ess(exp_w)
        if e / max(1, N) < 0.05:
            warnings.append(f'{warning_prefix}GaMD exponential reweighting ESS is very low: {e:.1f}/{N}')
        if bs.get('std_kcal_mol', 0) > 6.0:
            warnings.append(f"{warning_prefix}GaMD boost std is large ({bs['std_kcal_mol']:.2f} kcal/mol); cumulant reweighting may be unreliable")
        if bs.get('anharmonicity_score') is not None and bs['anharmonicity_score'] > 1.0:
            warnings.append(f"{warning_prefix}GaMD boost anharmonicity score is high ({bs['anharmonicity_score']:.2f})")
    else:
        exp_pmf = umbrella; cum_pmf = umbrella; cum3_pmf = umbrella; selected = 'umbrella_only'
        cdiag = {'boost_mean_kj': np.full(args.bins, np.nan), 'boost_var_kj2': np.full(args.bins, np.nan)}; cdiag3 = cdiag
        warnings.append(f'{warning_prefix}No finite variable GaMD boosts found; selected PMF is umbrella-only unbiased.')
    pmfs = {'umbrella_only': umbrella, 'gamd_exponential': exp_pmf, 'gamd_cumulant2': cum_pmf, 'gamd_cumulant3': cum3_pmf}
    if extra_pmfs:
        pmfs.update(extra_pmfs)
    _force_method = str(getattr(args, 'selected_method', 'auto') or 'auto')
    if _force_method != 'auto' and _force_method in pmfs:
        if _force_method in ('gamd_exponential', 'gamd_cumulant2', 'gamd_cumulant3') and not boost_ok:
            warnings.append(f'{warning_prefix}--selected-method {_force_method} requested but no usable GaMD boost; it equals umbrella-only here.')
        selected = _force_method
    O = _agm.overlap_matrix(d.cv, d.window, bins, K)
    neigh = [float(O[i, i + 1]) for i in range(K - 1)]
    bad = [i for i, x in enumerate(neigh) if x < args.min_neighbor_overlap]
    if bad:
        warnings.append(f'{warning_prefix}Weak neighbor CV overlap below %.2f for pairs: ' % args.min_neighbor_overlap + ', '.join(f'{i}-{i + 1} ({neigh[i]:.2f})' for i in bad))
    sel = pmfs[selected]
    finite = sel['pmf'][np.isfinite(sel['pmf'])]
    span = float(np.max(finite) - np.min(finite)) if finite.size else float('nan')
    minidx = int(np.nanargmin(sel['pmf'])) if finite.size else -1
    _sel_diag = {'gamd_cumulant2': cdiag, 'gamd_cumulant3': cdiag3}.get(selected, cdiag)
    pmf_uncertainty_std = None
    _uncertainty_extra = {}
    if getattr(args, 'pmf_uncertainty', False) and minidx >= 0:
        # minidx<0 means sel['pmf'] has no finite bin at all (see the
        # finite/minidx guard two lines above) -- _bootstrap_pmf_uncertainty_1d's
        # own np.nanargmin on an all-NaN main_pmf['pmf'] would raise, so skip
        # the uncertainty computation the same way pmf_minimum_cv_A is
        # already silently skipped for this degenerate case.
        block_ids = _agm._sample_block_ids(d)
        boot_rng = np.random.default_rng(int(getattr(args, 'pmf_uncertainty_seed', 0)))
        n_boot = int(getattr(args, 'pmf_uncertainty_n_boot', 100))
        # `selected` can be force-overridden to a gamd_* name via
        # --selected-method even when boost_ok is False (sel is then really
        # umbrella-only, an all-NaN boost array under the hood) -- dispatch
        # the bootstrap on what `sel` actually IS, not on the possibly-forced
        # label, so it doesn't try to rebuild gamd-style replicates from an
        # all-NaN boost.
        _boot_method = selected if boost_ok else 'umbrella_only'
        boot_result = _bootstrap_pmf_uncertainty_1d(
            d.cv, logw, d.boost_kj, bins, d.beta, kbt_kcal, d.window, block_ids,
            _boot_method, sel, n_boot, boot_rng,
            smooth_logfac_sigma=_agm._eff_smooth(args, 'gamd_smooth_sigma'),
        )
        pmf_uncertainty_std = boot_result['pmf_std']
        _uncertainty_extra = {'pmf_std_kcal_mol': pmf_uncertainty_std}
        if boot_result['low_block_windows']:
            warnings.append(f"{warning_prefix}PMF uncertainty is unreliable near windows {boot_result['low_block_windows']} (fewer than 3 independent trajectory blocks).")
    _agm.write_pmf(out / 'pmf_unbiased.csv', sel, selected, {'boost_mean_kj_mol': _sel_diag.get('boost_mean_kj', np.full(args.bins, np.nan)), 'boost_var_kj2_mol2': _sel_diag.get('boost_var_kj2', np.full(args.bins, np.nan)), **_uncertainty_extra})
    _agm.write_pmf(out / 'pmf_umbrella_only.csv', umbrella, 'umbrella_only')
    _agm.write_pmf(out / 'pmf_gamd_exponential.csv', exp_pmf, 'gamd_exponential')
    _agm.write_pmf(out / 'pmf_gamd_cumulant2.csv', cum_pmf, 'gamd_cumulant2', {'boost_mean_kj_mol': cdiag.get('boost_mean_kj', np.full(args.bins, np.nan)), 'boost_var_kj2_mol2': cdiag.get('boost_var_kj2', np.full(args.bins, np.nan))})
    _agm.write_pmf(out / 'pmf_gamd_cumulant3.csv', cum3_pmf, 'gamd_cumulant3', {'boost_mean_kj_mol': cdiag3.get('boost_mean_kj', np.full(args.bins, np.nan)), 'boost_var_kj2_mol2': cdiag3.get('boost_var_kj2', np.full(args.bins, np.nan))})
    _agm.write_all(out / 'pmf_all_methods.csv', pmfs)
    with (out / 'overlap_matrix.csv').open('w', newline='') as f:
        wr = csv.writer(f); wr.writerow(['window'] + list(range(K))); [wr.writerow([i] + [float(x) for x in O[i]]) for i in range(K)]
    n_k_local, _mean_per_window, _std_per_window = _window_cv_mean_std(d.window, d.cv, K)
    _has_samples = n_k_local > 0
    with (out / 'window_diagnostics.csv').open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['window', 'center_A', 'k_kcal_mol_A2', 'samples', 'cv_mean_A', 'cv_std_A', 'overlap_left', 'overlap_right']); wr.writeheader()
        for k in range(K):
            wr.writerow({'window': k, 'center_A': float(d.centers[k]) if k < d.centers.size and np.isfinite(d.centers[k]) else '', 'k_kcal_mol_A2': float(d.k_kcal[k]) if k < d.k_kcal.size and np.isfinite(d.k_kcal[k]) else '', 'samples': int(n_k_local[k]), 'cv_mean_A': float(_mean_per_window[k]) if _has_samples[k] else '', 'cv_std_A': float(_std_per_window[k]) if _has_samples[k] else '', 'overlap_left': float(O[k - 1, k]) if k > 0 else '', 'overlap_right': float(O[k, k + 1]) if k + 1 < K else ''})
    if progress is not None:
        progress.bar('analysis stages', 5, 6, 'plotting PNG outputs', force=True)
    _agm.plot_outputs(d, pmfs, selected, O, out, warnings, smooth_sigma=_agm._eff_smooth(args, 'pmf_smooth_sigma'), args=args)
    return {
        'pmfs': pmfs, 'selected': selected, 'boost_ok': boost_ok, 'boost': bs,
        'pmf_span_kcal_mol': span, 'pmf_minimum_cv_A': float(sel['cv_A'][minidx]) if minidx >= 0 else None,
        'neighbor_overlap': neigh, 'n_samples': N, 'O': O,
        'pmf_uncertainty_std': pmf_uncertainty_std,
        'files': {
            'pmf_unbiased_csv': str(out / 'pmf_unbiased.csv'), 'pmf_all_methods_csv': str(out / 'pmf_all_methods.csv'),
            'pmf_umbrella_only_csv': str(out / 'pmf_umbrella_only.csv'), 'pmf_gamd_exponential_csv': str(out / 'pmf_gamd_exponential.csv'),
            'pmf_gamd_cumulant2_csv': str(out / 'pmf_gamd_cumulant2.csv'), 'pmf_gamd_cumulant3_csv': str(out / 'pmf_gamd_cumulant3.csv'),
            'overlap_matrix_csv': str(out / 'overlap_matrix.csv'), 'window_diagnostics_csv': str(out / 'window_diagnostics.csv'),
        },
    }
