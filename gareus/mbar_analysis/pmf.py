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

import math
import sys
from typing import Any, Optional

import numpy as np


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

