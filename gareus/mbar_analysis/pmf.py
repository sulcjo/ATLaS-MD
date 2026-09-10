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
# Imported DIRECTLY, not resolved through _bridge(): make_overlap_bins reads no
# parse_args()-mutated global, and _bridge() has a known branch-order defect
# (see CLAUDE.md) that can hand back a second, never-parse_args()'d script copy.
# analyze_gareus_mbar.py's own solvers import list does not re-export this name
# either, so _agm.make_overlap_bins does not exist.
from gareus.mbar_analysis.solvers import make_overlap_bins
from gareus.mbar_analysis.estimators import ladder_excluded_methods, choose_site_method


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

# A truncated cumulant expansion approximates ln<exp(beta*dV)> by its first
# few cumulants. That is only meaningful if successive terms SHRINK. The ratio
# below is the largest |next term| / |previous term| still treated as a
# converging series; above it the truncation error is comparable to (or larger
# than) the term retained, so the "correction" is not an approximation of
# anything and the estimator must not be used.
#
# For a Gaussian boost of width sigma the terms scale as (beta*sigma)^n/n!, so
# the ratio is ~beta*sigma/n: the series only converges usefully while the
# boost is at most a couple of kT wide. Measured on a real 29M-sample
# chignolin run with beta*sigma = 3.57, the terms were 8.81, 6.38 and 9.57 kT
# -- flat, not shrinking -- while the shipped default silently selected the
# 2nd-order result.
CUMULANT_CONVERGENCE_MAX_RATIO = 0.5


def cumulant_series_verdict(diag3: dict, counts=None) -> dict:
    """Is the cumulant expansion converging on this data?

    Takes the ORDER-3 diagnostics (which carry all three term magnitudes) and
    returns the typical size of each term in kT plus a boolean verdict. Bins
    are weighted by sample count so that a handful of near-empty bins cannot
    condemn (or rescue) an otherwise well-behaved run.

    A verdict of False does not mean the PMF is slightly off. It means the
    truncation error is the same size as the terms kept, so neither the 2nd-
    nor the 3rd-order curve estimates the unbiased free energy.
    """
    def _typ(key):
        v=np.asarray(diag3.get(key), dtype=np.float64) if diag3.get(key) is not None else None
        if v is None or v.size==0:
            return float('nan')
        w=np.ones_like(v) if counts is None else np.asarray(counts,dtype=np.float64)
        m=np.isfinite(v)&np.isfinite(w)&(w>0)
        if not np.any(m):
            return float('nan')
        return float(np.average(np.abs(v[m]), weights=w[m]))
    t1,t2,t3=_typ('cumulant_term1_kT'),_typ('cumulant_term2_kT'),_typ('cumulant_term3_kT')
    r21 = t2/t1 if (np.isfinite(t1) and t1>0) else float('nan')
    r32 = t3/t2 if (np.isfinite(t2) and t2>0) else float('nan')
    # The verdict rests on term3/term2, NOT on term2/term1.
    #
    # term1 = beta*<dV> is the MEAN boost. A GaMD envelope can sit at any
    # offset, and a constant offset shifts every bin's log-weight equally,
    # cancelling in the PMF -- it changes no free-energy difference. Including
    # it would let a large boost mean make the ratio look healthy no matter how
    # badly the expansion behaves. The terms that shape the PMF are the central
    # cumulants from order 2 up, and those scale as (beta*sigma)^n/n!.
    #
    # Width alone does NOT disqualify the expansion: for a Gaussian dV every
    # cumulant above the second is exactly zero, so the series terminates at
    # order 2 and CE2 is exact however wide the boost. It is non-Gaussianity
    # that breaks the truncation, which is precisely what term3/term2 measures.
    #
    # With only order-2 diagnostics kappa3 is unavailable, so Gaussianity
    # cannot be checked at all. There the fallback is deliberately
    # conservative and keys on the width (term2 = 0.5*(beta*sigma)^2, so
    # beta*sigma = sqrt(2*term2)): a narrow boost cannot go far wrong whatever
    # its shape, while a wide one is only safe if it happens to be Gaussian --
    # which is exactly what cannot be verified without kappa3. Callers that
    # have order-3 diagnostics (the report path does) never take this branch.
    t4=_typ('cumulant_term4_kT')
    r42 = t4/t2 if (np.isfinite(t2) and t2>0) else float('nan')
    # term3/term2 alone is necessary but NOT sufficient: an odd cumulant can
    # vanish by symmetry while the even ones keep growing, so a symmetric-but-
    # heavy-tailed boost would pass a third-order test and still diverge.
    # Require the fourth order to be controlled too, when it is available.
    if np.isfinite(r32):
        checks=[r32]+([r42] if np.isfinite(r42) else [])
        converging = bool(all(r <= CUMULANT_CONVERGENCE_MAX_RATIO for r in checks))
        basis = 'ratio_3_over_2 and ratio_4_over_2' if np.isfinite(r42) else 'ratio_3_over_2'
        governing = max(checks)
    elif np.isfinite(t2) and t2 > 0:
        beta_sigma = float(np.sqrt(2.0*t2))
        converging = bool(beta_sigma <= 1.0)
        basis = 'beta_sigma (no third cumulant available)'
        governing = beta_sigma
    else:
        converging, basis, governing = False, 'no usable terms', float('nan')
    # Between-bin scatter of the neglected term. This is what actually governs
    # whether the truncation error cancels in a free-energy DIFFERENCE: a
    # neglected term that is large but CONSTANT across CV cancels exactly when
    # two bins are subtracted, while its variation does not. Reporting only the
    # magnitude would invite the (false) claim that a big term is harmless
    # because it is "uniform".
    def _scatter(key):
        v=diag3.get(key)
        if v is None: return float('nan')
        v=np.asarray(v,dtype=np.float64)
        w=np.ones_like(v) if counts is None else np.asarray(counts,dtype=np.float64)
        m=np.isfinite(v)&np.isfinite(w)&(w>0)
        if m.sum()<2: return float('nan')
        mu=np.average(v[m],weights=w[m])
        return float(np.sqrt(np.average((v[m]-mu)**2,weights=w[m])))
    return {'term1_kT':t1,'term2_kT':t2,'term3_kT':t3,'term4_kT':t4,
            'term1_between_bin_kT':_scatter('cumulant_term1_kT'),
            'term3_between_bin_kT':_scatter('cumulant_term3_kT'),
            'term4_between_bin_kT':_scatter('cumulant_term4_kT'),
            'ratio_2_over_1':r21,'ratio_3_over_2':r32,'ratio_4_over_2':r42,
            'beta_sigma':float(np.sqrt(2.0*t2)) if (np.isfinite(t2) and t2>0) else float('nan'),
            'basis':basis,'max_ratio':governing,
            'threshold':CUMULANT_CONVERGENCE_MAX_RATIO,
            'converging':converging}


# Exponential reweighting of the MBAR umbrella weights is not an approximation.
# This package calibrates ONE shared GaMD setup per campaign, so dV is the same
# function of configuration for every state: reweighting by exp(beta*dV) is
# algebraically identical to putting dV in the reduced potential and adding an
# unbiased target state at u=0, because a per-row constant cancels from the f_k
# solve and survives only in the target weight. gamd_exponential IS the exact
# estimator; CE2 is the approximation to it.
#
# What makes CE2 preferable on a WIDE boost is variance, not correctness: the
# exact estimator's ESS falls off like exp(-(beta*sigma)^2). At chignolin_6's
# beta*sigma = 3.57 that is 3.0e-6 of N, which estimates nothing. At the
# sigma0 = 1.0 kcal/mol rebuild, beta*sigma = 1.43 and ESS/N = 0.13, where
# taking CE2's truncation bias buys nothing.
#
# The fraction matches the pre-existing low-ESS warning threshold in
# run_pmf_and_gamd_boost_report, so a run can never be told its ESS is fine and
# still be handed the biased estimator. The absolute floor exists because a
# healthy FRACTION of a tiny sample is still a tiny sample.
EXACT_REWEIGHT_MIN_ESS_FRACTION = 0.05
EXACT_REWEIGHT_MIN_ESS = 100.0


def select_unbiased_method(exp_ess, n_samples,
                           min_ess_fraction=EXACT_REWEIGHT_MIN_ESS_FRACTION,
                           min_ess=EXACT_REWEIGHT_MIN_ESS, *,
                           gamd_ladder: bool = False):
    """Choose between the exact estimator and the second-order cumulant one.

    Returns ``(method, reason)`` where ``method`` is ``'gamd_exponential'`` or
    ``'gamd_cumulant2'`` -- or, when ``gamd_ladder`` is true, unconditionally
    ``'umbrella_only'``.

    Gated on the EXPONENTIAL estimator's own effective sample size, deliberately
    not on the cumulant series' convergence verdict. An earlier revision switched
    estimator when the cumulant series failed to converge and was reverted,
    correctly: that condition fires precisely when the boost is wide, which is
    precisely when the exponential estimator is worthless. Gating on the exact
    estimator's own ESS means the switch can only happen when the exact answer is
    actually affordable.

    ``gamd_ladder=True`` means the sample's own ``u_nk`` already carries the
    closed-form Pep-GaMD boost for every state (see ``reconstruct_bias_matrix``
    and ``build_union_state_mbar_inputs``), so MBAR's own reweighting is exact
    and neither the exponential-reweighting nor the cumulant-expansion
    approximation is needed or wanted -- 'umbrella_only' here means "no
    additional correction on top of u_nk", not "the boost was discarded".
    """
    if gamd_ladder:
        return 'umbrella_only', 'λ ladder: boost is inside u_nk, MBAR is exact; no cumulant'
    try:
        ess = float(exp_ess)
        n = int(n_samples)
    except (TypeError, ValueError):
        return 'gamd_cumulant2', 'exponential ESS unavailable; CE2 selected'
    if n <= 0:
        return 'gamd_cumulant2', 'no samples; CE2 selected'
    if not np.isfinite(ess) or ess < 0.0:
        return 'gamd_cumulant2', f'exponential ESS is not finite ({exp_ess!r}); CE2 selected'
    frac = ess / float(n)
    if frac >= float(min_ess_fraction) and ess >= float(min_ess):
        return ('gamd_exponential',
                f'exponential reweighting is exact and affordable here '
                f'(ESS {ess:.1f}/{n} = {frac:.3f} >= {float(min_ess_fraction):.3f} '
                f'and >= {float(min_ess):.0f}); selected over CE2, which carries a '
                f'truncation bias')
    return ('gamd_cumulant2',
            f'exponential reweighting is exact but unaffordable here '
            f'(ESS {ess:.1f}/{n} = {frac:.3g}; need fraction >= '
            f'{float(min_ess_fraction):.3f} and count >= {float(min_ess):.0f}), '
            f'so CE2 is selected despite its truncation bias')


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


def robust_pmf_reference(F, logfac=None, counts=None, mad_k: float = 4.0):
    """Index of the bin the PMF should be zeroed at, and why.

    ``F -= F.min()`` makes the single lowest bin the reference for every other
    value. That is fine when the minimum is a real basin and catastrophic when
    it is a reweighting outlier: on the motivating run one bin's GaMD
    correction sat ~28 sigma (autocorrelation-corrected) above its neighbours',
    took ten times their probability, became the minimum, and thereby shifted
    the whole curve -- moving the reported free-energy minimum by a full bin
    relative to the unweighted umbrella estimate.

    So the reference is taken over bins that are not reweighting outliers:
    those whose ``logfac`` lies within ``mad_k`` median-absolute-deviations of
    the median. MAD rather than a standard deviation precisely because the
    contaminating bins are the ones we must not let set the scale. Bins with no
    samples are never eligible.

    Excluding a bin from being the REFERENCE does not remove it from the PMF --
    its value is still reported, and still relative to the same zero. This
    chooses where zero sits; it does not smooth, mask or reweight anything.

    Returns ``(index, info)``; ``index`` is None when nothing is eligible, in
    which case the caller should fall back to the plain minimum.
    """
    F=np.asarray(F,dtype=np.float64)
    ok=np.isfinite(F)
    if counts is not None:
        ok&=np.asarray(counts,dtype=np.float64)>0
    info={'mad_k':float(mad_k),'n_finite':int(np.count_nonzero(np.isfinite(F)))}
    if not np.any(ok):
        info['reason']='no finite bins'
        return None, info
    eligible=ok.copy()
    if logfac is not None:
        lf=np.asarray(logfac,dtype=np.float64)
        m=np.isfinite(lf)&ok
        if np.count_nonzero(m)>=3:
            med=float(np.median(lf[m]))
            mad=float(np.median(np.abs(lf[m]-med)))
            if mad>0:
                dev=np.full(lf.shape,np.inf); dev[m]=np.abs(lf[m]-med)/(1.4826*mad)
                eligible=ok&(dev<=mad_k)
                info.update({'logfac_median':med,'logfac_mad':mad,
                             'n_outliers_excluded':int(np.count_nonzero(ok&~eligible))})
    if not np.any(eligible):
        info['reason']='every bin is a reweighting outlier; using plain minimum'
        return int(np.nanargmin(np.where(ok,F,np.inf))), info
    idx=int(np.nanargmin(np.where(eligible,F,np.inf)))
    plain=int(np.nanargmin(np.where(ok,F,np.inf)))
    info['reference_bin']=idx
    info['plain_minimum_bin']=plain
    info['differs_from_plain_minimum']=bool(idx!=plain)
    if idx!=plain:
        info['shift_kcal_mol']=float(F[idx]-F[plain])
    return idx, info


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
    kappa4=np.full(B,np.nan,dtype=np.float64)
    logfac=np.zeros(B,dtype=np.float64)
    if np.any(nz):
        logfac[nz]=beta*mean[nz]+0.5*beta*beta*var[nz]
        if order==3:
            idx=shared['idx']; w=shared['w']; dx=shared['dx']; sw=shared['sw']
            sdx3=np.bincount(idx,weights=w*dx*dx*dx,minlength=B).astype(np.float64)
            kappa3[nz]=sdx3[nz]/sw[nz]
            logfac[nz]+=(beta**3/6.0)*kappa3[nz]
            # Fourth cumulant, for the convergence diagnostic only -- it is NOT
            # added to logfac (that would be a CE4 estimator, which this code
            # does not offer). term3/term2 alone is necessary but not
            # sufficient: an odd cumulant can vanish by symmetry while the even
            # ones keep growing, so a series can look convergent at third order
            # and still diverge. kappa4 = m4 - 3*var^2 (excess kurtosis form).
            sdx4=np.bincount(idx,weights=w*dx*dx*dx*dx,minlength=B).astype(np.float64)
            m4=np.full(B,np.nan); m4[nz]=sdx4[nz]/sw[nz]
            kappa4[nz]=m4[nz]-3.0*var[nz]*var[nz]
    # A bin with real weighted samples (p0>0) but zero samples with a finite
    # boost has an UNKNOWN GaMD correction -- flag NaN rather than silently
    # falling back to logfac=0 (which would look like "no correction needed"
    # and reproduce the raw/unbiased-looking p0 value for that bin). Bins
    # with no samples at all (p0==0) keep logfac=0 so p=0*exp(0)=0 there,
    # matching the existing "no samples" -> F=inf -> excluded-by-isfinite
    # convention used throughout this file.
    logfac[(~nz)&(p0>0)]=np.nan
    # Per-order cumulant term magnitudes, kept for the convergence check that
    # decides whether this estimator may be used at all (see
    # `cumulant_series_verdict`). A truncated cumulant expansion is only
    # meaningful when successive terms SHRINK; when they do not, the truncation
    # error exceeds the term retained and the result is not an approximation of
    # anything.
    t1=np.full(B,np.nan); t2=np.full(B,np.nan); t3=np.full(B,np.nan); t4=np.full(B,np.nan)
    if np.any(nz):
        t1[nz]=beta*mean[nz]
        t2[nz]=0.5*beta*beta*var[nz]
        if order==3 and np.any(np.isfinite(kappa3)):
            fin=nz&np.isfinite(kappa3)
            t3[fin]=(beta**3/6.0)*kappa3[fin]
            fin4=nz&np.isfinite(kappa4)
            t4[fin4]=(beta**4/24.0)*kappa4[fin4]
    lf_finite=logfac[np.isfinite(logfac)]
    lf_scatter=float(np.std(np.diff(lf_finite))) if lf_finite.size>2 else 0.0
    sigma_req=float(smooth_logfac_sigma or 0.0)
    if sigma_req>0:
        try:
            from scipy.ndimage import gaussian_filter1d
            finite=np.isfinite(logfac)
            if np.any(finite):
                # Interpolate across undefined bins before filtering: running a
                # Gaussian filter over NaN would spread it across the array.
                filled=np.interp(np.arange(B),np.arange(B)[finite],logfac[finite])
                sm=gaussian_filter1d(filled,sigma=sigma_req,mode='nearest')
                logfac=np.where(finite,sm,np.nan)
        except Exception:
            pass
    p=p0*np.exp(np.clip(logfac,-700,700))
    ps=float(np.nansum(p))
    if ps>0: p/=ps
    with np.errstate(divide='ignore',invalid='ignore'):
        F=-kbt_kcal*np.log(p)
    ref_idx, ref_info = robust_pmf_reference(F, logfac=logfac, counts=counts)
    if ref_idx is not None:
        F=F-F[ref_idx]
    else:
        mask=np.isfinite(F)
        if np.any(mask): F-=np.nanmin(F[mask])
    return {'cv_A':centers.copy(),'prob':p,'pmf':F,'counts':counts.astype(int)}, {
        'pmf_reference':ref_info,
        'boost_mean_kj':mean.copy(),'boost_var_kj2':var.copy(),'boost_kappa3_kj3':kappa3,
        'log_reweight_factor':logfac,
        # Provenance for the reweighting factor, so a consumer can tell a
        # trustworthy cumulant PMF from a noise-dominated one without
        # re-deriving it from the boost moments.
        'logfac_scatter_kT':lf_scatter,
        'logfac_smooth_sigma_requested':sigma_req,
        'boost_kappa4_kj4':kappa4,
        'cumulant_term1_kT':t1,'cumulant_term2_kT':t2,'cumulant_term3_kT':t3,
        'cumulant_term4_kT':t4,
    }


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

# ===========================================================================
# Mapping-sanity diagnostics (added 2026-08-25)
#
# All four of the helpers below came out of one root-cause investigation:
# docs/chignolin_6_low_ess_root_cause.md. On that real run
# --us-auto-drop-bad-windows pruned bad umbrella windows post-pull and
# gareus/production.py renumbered the survivors 0..N-1, but each phase's
# epoch_window_map.csv -- written earlier by the registry as an IDENTITY map
# over all 27 nominally-active states -- was never rewritten. ~1.8M samples
# (15% of the campaign) were therefore attributed to the WRONG umbrella state:
# one state was scored against a sign-flipped secondary centre (2,012 kT of
# fabricated bias), one was never sampled at all (a phantom duplicate of its
# neighbour), and one pooled three different Hamiltonians. The pipeline
# produced a complete, plausible-looking PMF and not a single existing check
# fired.
#
# Nothing here changes a PMF number. These are detectors for that failure
# class, which the analysis was structurally blind to.
# ===========================================================================

# Source of truth for these column spellings is gareus/mbar_analysis/loaders.py's
# own secondary-CV lookup inside load_csv (the sec_centers/sec_ks pair). They are
# duplicated rather than imported so this module stays importable on its own
# (test_pmf_module_is_importable_standalone) without dragging in the loader chain
# (DuckDB/Parquet) that only the loaders themselves need. If a loader grows a new
# spelling, add it in BOTH places.
_SECONDARY_CENTER_KEYS = ('secondary_cv_center', 'secondary_center', 'secondary',
                          'ss0', 'secondary_cv_target')
_SECONDARY_K_KEYS = ('secondary_cv_k_kcal_mol', 'secondary_k_kcal_mol',
                     'secondary_cv_k_kcal', 'ss_k_kcal_mol', 'secondary_k')

# Self-bias warning thresholds, in kT.
#
# u_nk is ALREADY reduced -- every loader multiplies its harmonic sum by
# beta*KJ_PER_KCAL before storing it (loaders.py's `scale`, loaders_adaptive.py's
# same) -- so u_nk[n, k] is a dimensionless kT and these numbers are directly
# comparable with the table in the root-cause doc's section 3.
#
# A correctly-mapped harmonic restraint puts its own samples at 0.5 kT per
# restrained degree of freedom by equipartition, i.e. ~1 kT for the 2-DOF
# (primary + secondary) restraints this pipeline uses; the reduced self-bias is
# then Exp(1)-distributed, median ln2 = 0.69, p90 2.30. The real run's 24 healthy
# states measured 0.5-1.4 kT median and 1.7-4.9 kT p90. The three corrupted ones
# measured 63.6 / 1.1 / 132.9 median and 80.7 / 652.8 / 221.8 p90 -- more than an
# order of magnitude clear of the healthy band, which is why loose thresholds are
# enough and why they do not need per-run tuning. 10 kT sits ~2x above the worst
# healthy p90; the separate 50 kT p90 trigger exists for the pooled-Hamiltonian
# shape (that run's state 22), whose median is textbook because only a minority
# tail of its slot is fabricated.
SELF_BIAS_MEDIAN_WARN_KT = 10.0
SELF_BIAS_P90_WARN_KT = 50.0

# Fraction of samples that must carry a finite secondary CV before the JOINT
# (cv1, cv2) overlap matrix is allowed to be the reported, health-gating one.
#
# A run can legitimately have cv2 null for a large block of its samples -- a
# mid-campaign secondary-CV regime switch (torsion-pca -> tica-linear) leaves an
# early epoch with a different projection or none at all. The joint matrix
# computed on a minority of the population looks exactly as authoritative as the
# marginal one while describing a different population, so below this fraction it
# is treated as diagnostic-only: the marginal matrix is reported instead and the
# reason is stated in a warning. Between this and _WARN, the joint matrix is used
# but flagged. Thresholds named per overlap_matrix()'s own docstring guidance.
OVERLAP_CV2_MIN_RETAINED_FRACTION = 0.50
OVERLAP_CV2_WARN_RETAINED_FRACTION = 0.95

# Names for the two overlap spaces. Stamped into pmf_summary.json
# (``overlap_space`` / ``joint_overlap['space']``) and into the header comment
# of each overlap matrix CSV, so a number on disk always says which space it
# is in. Before this existed, a joint number and a marginal number were
# indistinguishable once written out, which made two runs analysed either side
# of a code change silently incomparable.
OVERLAP_SPACE_MARGINAL = 'cv1_marginal'
OVERLAP_SPACE_JOINT = 'cv1_cv2_joint'


def _secondary_window_params(meta: dict, K: int) -> tuple:
    """``(secondary_center, secondary_k)``, length K, from the loader's own
    per-window row list (``meta['umbrella_window_rows']``).

    Every loader that builds a ``Data`` stashes its window table there, but they
    disagree on the column spelling (union-Parquet's registry rows use
    ``secondary_center``/``secondary_k``; load_csv's umbrella_windows.csv and the
    multi-round augmentation path use ``secondary_cv_center``/
    ``secondary_cv_k_kcal_mol``), hence the alias tuples above.

    This is ONE snapshot of the window table, applied to every epoch: it is not
    the per-epoch native window params the union loader reconstructs u_nk (and
    therefore self_bias) from, and on a run that recentred its windows
    mid-campaign the two legitimately differ. The columns this feeds are named
    ``*_window_table`` for that reason; see the window_diagnostics.csv writer.

    Absent, blank, unparseable or missing rows yield NaN rather than 0.0 -- the
    caller must be able to tell "this window has no secondary restraint" from
    "this window is restrained at 0.0", and a 0.0 default would silently claim
    the latter. (Note ``loaders.py`` itself defaults secondary_k to 0.0 for its
    own bias-matrix construction, where 0.0 correctly means "no term"; here the
    value is displayed to an operator, so absence must stay visibly absent.)

    Known gap, not fixable from this module: ``loaders_adaptive.py``'s
    ``load_epoch_csv_adaptive`` writes ``umbrella_window_rows`` with only
    ``center_A``/``k_kcal_mol_A2``, discarding the ``sec_centers``/``sec_ks`` it
    computed locally -- runs loaded through that legacy epoch-CSV path get blank
    secondary columns here even though the values existed.
    """
    rows = (meta or {}).get('umbrella_window_rows') or []
    c2 = np.full(int(K), np.nan)
    k2 = np.full(int(K), np.nan)
    for k in range(min(int(K), len(rows))):
        row = rows[k] if isinstance(rows[k], dict) else {}
        for dest, keys in ((c2, _SECONDARY_CENTER_KEYS), (k2, _SECONDARY_K_KEYS)):
            for key in keys:
                raw = row.get(key, '')
                if raw in (None, '', 'nan', 'None'):
                    continue
                try:
                    dest[k] = float(raw)
                except (TypeError, ValueError):
                    pass
                break
    return c2, k2


def self_bias_diagnostics(u_nk: np.ndarray, window: np.ndarray, K: int) -> dict:
    """Per-state distribution of each state's OWN samples' bias in its OWN
    restraint -- the diagonal ``u_nk[n, window[n]]``, grouped by state.

    This is the cheapest possible test of the sample-to-state mapping, and the
    one check that would have caught the whole chignolin_6 root cause on its own
    (see the module comment above and SELF_BIAS_MEDIAN_WARN_KT for the real
    numbers). Physically it must be ~1 kT for a 2-DOF harmonic regardless of the
    CV, the window spacing, the temperature or the system -- there is no run for
    which 60 kT is a legitimate value, so a violation is unambiguous.

    Deliberately NOT a complete detector, and the caller must say so: a
    mis-mapping between two *nearby* windows stays under threshold. On the real
    run, state 20 was a pure phantom (every one of its 861,547 samples actually
    belonged to state 21) yet measured a benign 3.4 kT median / 6.7 kT p90,
    because those two states' secondary centres were only ~5-6 kT apart. Absence
    of this warning is therefore not proof the mapping is right.

    Returns a dict of length-K arrays: ``n`` (finite own-bias samples per state),
    ``median_kT``, ``p90_kT``, ``max_kT``, ``frac_above_100_kT``. Every
    statistic is NaN for a state with no samples (matching
    ``_window_cv_mean_std``'s convention, so the CSV writer blanks it the same
    way).
    """
    u_nk = np.asarray(u_nk)
    window = np.asarray(window, dtype=np.int64)
    K = int(K)
    n_k = np.zeros(K, dtype=np.int64)
    med = np.full(K, np.nan); p90 = np.full(K, np.nan)
    mx = np.full(K, np.nan); frac100 = np.full(K, np.nan)
    if window.size == 0 or K <= 0 or u_nk.ndim != 2 or u_nk.shape[1] < K:
        return {'n': n_k, 'median_kT': med, 'p90_kT': p90, 'max_kT': mx,
                'frac_above_100_kT': frac100}
    rows = np.flatnonzero((window >= 0) & (window < K))
    if rows.size:
        u_self = np.asarray(u_nk[rows, window[rows]], dtype=np.float64)
        # A non-finite own-bias entry is an already-excluded sample (clean()
        # drops rows with any non-finite u_nk, but a subset Data or a
        # partially-reconstructed matrix can still carry them). Dropping them
        # here rather than letting np.median propagate NaN matters: a single NaN
        # would otherwise silence this check for that whole state.
        keep = np.isfinite(u_self)
        rows = rows[keep]; u_self = u_self[keep]
    if rows.size:
        # One stable sort, then contiguous per-state slices. The obvious
        # alternative (``u_self[window[rows] == k]`` inside a loop over K) is the
        # O(N*K) boolean-mask-per-window pattern _window_cv_mean_std was
        # extracted to eliminate, and this runs on every analysis over the full
        # multi-million-row population. np.median/np.percentile have no
        # by-group form, so the per-state loop over slices stays.
        w_rows = window[rows]
        order = np.argsort(w_rows, kind='stable')
        w_sorted = w_rows[order]
        u_sorted = u_self[order]
        bounds = np.searchsorted(w_sorted, np.arange(K + 1))
        for k in range(K):
            a, b = int(bounds[k]), int(bounds[k + 1])
            if b <= a:
                continue
            blk = u_sorted[a:b]
            n_k[k] = blk.size
            med[k] = float(np.median(blk))
            p90[k] = float(np.percentile(blk, 90.0))
            mx[k] = float(np.max(blk))
            frac100[k] = float(np.count_nonzero(blk > 100.0) / blk.size)
    return {'n': n_k, 'median_kT': med, 'p90_kT': p90, 'max_kT': mx,
            'frac_above_100_kT': frac100}


def self_bias_warning_lines(sb: dict, warning_prefix: str = '',
                            median_thr: float = SELF_BIAS_MEDIAN_WARN_KT,
                            p90_thr: float = SELF_BIAS_P90_WARN_KT) -> list:
    """HIGH-severity warning text (zero or one line) for self_bias_diagnostics.

    One line listing every offending state rather than one line per state, so
    gareus_report.py's de-duplicator does not collapse a 3-state failure into a
    single representative and hide which states they were.

    The wording is load-bearing in two ways: gareus_report.py's ``_WARN_RULES``
    matches "Implausible self-bias" to classify this HIGH (an unmatched warning
    silently defaults to MEDIUM), and it names the mapping explicitly rather
    than describing a symptom -- the whole point is that an operator reading it
    goes and checks epoch_window_map.csv instead of blaming the sampling.
    """
    med = np.asarray(sb.get('median_kT'), dtype=np.float64).ravel()
    p90 = np.asarray(sb.get('p90_kT'), dtype=np.float64).ravel()
    bad = [k for k in range(med.size)
           if (np.isfinite(med[k]) and med[k] > median_thr)
           or (k < p90.size and np.isfinite(p90[k]) and p90[k] > p90_thr)]
    if not bad:
        return []
    detail = ', '.join(
        f'state {k} (median {med[k]:.1f} kT, p90 {p90[k]:.1f} kT)' for k in bad)
    return [
        f'{warning_prefix}Implausible self-bias in {len(bad)} umbrella state(s): {detail}. '
        f'A correctly-mapped harmonic restraint puts its own samples at ~1 kT '
        f'(trigger: median >{median_thr:.0f} kT or p90 >{p90_thr:.0f} kT), so the '
        f'sample-to-state mapping or the window parameters for those states are WRONG '
        f'and the PMF from them is not trustworthy. Check each phase\'s '
        f'epoch_window_map.csv against the window table the sampler actually ran '
        f'(umbrella_explicit_windows.csv). See docs/chignolin_6_low_ess_root_cause.md.']


def cv_space_neighbor_overlap(O: np.ndarray, centers: np.ndarray, k_kcal: np.ndarray,
                              centers2: np.ndarray, k2: np.ndarray, kbt_kcal: float,
                              n_k: Optional[np.ndarray] = None) -> list:
    """Each populated state's overlap with its nearest neighbour in RESTRAINT
    CENTRE space, rather than with the state that happens to sit beside it in
    the index ordering.

    Why: the chignolin_6 health check reported "worst 0.013 (pair 23-24); target
    >=0.30", which sent the investigation at state 24 -- an innocent, well-
    overlapped late-born bridge window that merely happened to be indexed next
    to 23. States 23 and 24 are nowhere near each other in CV space. Index
    adjacency is only meaningful for a 1D monotone ladder; the moment states are
    added adaptively, or a second CV exists, the index order is bookkeeping.

    Distance is measured in units of the states' own restraint widths
    (sigma ~ sqrt(kT/k), averaged over the pair per axis) and combined in
    quadrature, NOT as a raw Euclidean distance on the centres. Raw Euclidean
    would be dominated by whichever axis has the larger numeric range -- on the
    real run the primary contact CV spanned 0->0.176 while the secondary spanned
    ~3.8, so a raw metric would have been a pure CV2 nearest-neighbour search by
    accident rather than by design. Sigma units are also what the root-cause
    doc's "healthy umbrella overlap wants spacing/sigma ~1-1.5" language means.

    Axes are dropped, not defaulted, when they carry no restraint: a state pair
    where either side has a non-finite secondary centre or a non-positive
    secondary k contributes 0 to the secondary term, since an unrestrained axis
    says nothing about how far apart two windows were driven. With no usable
    secondary restraint anywhere this reduces to a primary-only nearest
    neighbour, which for a monotone 1D ladder reproduces index adjacency exactly.

    States with zero samples are excluded as both source and neighbour: their
    overlap row is identically zero, so including them would peg the reported
    worst-overlap at 0.000 for a condition the dedicated zero-sample-window
    check already reports, drowning out any real overlap problem.

    Ties resolve to the lowest index so the reported pair is stable across
    re-runs (the real run has near-coincident states, where ties are routine).

    Returns ``[{'window': i, 'neighbor': j, 'overlap': float,
    'centre_distance_sigma': float}, ...]`` in ascending window order, empty if
    fewer than two states are eligible.
    """
    O = np.asarray(O, dtype=np.float64)
    K = int(O.shape[0]) if O.ndim == 2 else 0
    if K < 2:
        return []
    c1 = np.asarray(centers, dtype=np.float64).ravel()
    kk1 = np.asarray(k_kcal, dtype=np.float64).ravel()
    c2 = np.asarray(centers2, dtype=np.float64).ravel() if centers2 is not None else np.full(K, np.nan)
    kk2 = np.asarray(k2, dtype=np.float64).ravel() if k2 is not None else np.full(K, np.nan)
    for name, arr in (('c1', c1), ('kk1', kk1), ('c2', c2), ('kk2', kk2)):
        if arr.size < K:
            pad = np.full(K, np.nan); pad[:arr.size] = arr
            if name == 'c1': c1 = pad
            elif name == 'kk1': kk1 = pad
            elif name == 'c2': c2 = pad
            else: kk2 = pad

    def _sigma(k_arr):
        """sqrt(kT/k) per state, with a single shared fallback width where k is
        unusable -- a missing force constant must not make a state infinitely
        far from (or infinitely close to) everything else."""
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.sqrt(float(kbt_kcal) / np.asarray(k_arr, dtype=np.float64))
        ok = np.isfinite(s) & (s > 0.0)
        if not np.all(ok):
            s = s.copy()
            s[~ok] = float(np.median(s[ok])) if np.any(ok) else 1.0
        return s

    s1 = _sigma(kk1)
    has_secondary = np.isfinite(c2) & np.isfinite(kk2) & (kk2 > 0.0)
    s2 = _sigma(np.where(has_secondary, kk2, np.nan))

    eligible = np.isfinite(c1)
    if n_k is not None:
        nk = np.asarray(n_k).ravel()
        if nk.size >= K:
            eligible = eligible & (nk[:K] > 0)
    idx = np.flatnonzero(eligible)
    if idx.size < 2:
        return []

    # Pairwise sigma-scaled centre distance, restricted to eligible states.
    ci = c1[idx]; si = s1[idx]
    d1 = np.abs(ci[:, None] - ci[None, :]) / (0.5 * (si[:, None] + si[None, :]))
    both2 = has_secondary[idx][:, None] & has_secondary[idx][None, :]
    if np.any(both2):
        cj = c2[idx]; sj = s2[idx]
        with np.errstate(invalid='ignore'):
            d2 = np.abs(cj[:, None] - cj[None, :]) / (0.5 * (sj[:, None] + sj[None, :]))
        d2 = np.where(both2 & np.isfinite(d2), d2, 0.0)
    else:
        d2 = np.zeros_like(d1)
    dist = np.sqrt(d1 ** 2 + d2 ** 2)
    np.fill_diagonal(dist, np.inf)

    rows = []
    for a, i in enumerate(idx):
        b = int(np.argmin(dist[a]))          # first minimum -> lowest index on ties
        j = int(idx[b])
        rows.append({'window': int(i), 'neighbor': j, 'overlap': float(O[i, j]),
                     'centre_distance_sigma': float(dist[a, b])})
    return rows


def overlap_components(O: np.ndarray, thr: float,
                       n_k: Optional[np.ndarray] = None) -> dict:
    """Connected components of the umbrella-overlap GRAPH: one edge wherever
    ``O[i, j] >= thr``.

    Why a graph and not another worst-pair number. Every overlap diagnostic
    above this one is pairwise -- worst index-adjacent pair, worst CV-space
    nearest-neighbour pair -- and no pairwise statistic can express the failure
    that actually happened on chignolin_6. MBAR determines free energies only up
    to one additive constant PER CONNECTED COMPONENT: two sets of states that
    never exchange samples share no information, so the offset between them is
    unconstrained by the data and the solver settles it arbitrarily. That run's
    posterior put 94% of its weight on a single state for exactly this reason
    (docs/chignolin_6_low_ess_root_cause.md, section 1). A set can be split in
    two while EVERY graded pair clears the threshold -- two well-overlapped
    pairs that do not touch each other is enough -- so this has to be measured
    on the whole matrix, not inferred from the pairing.

    Related but distinct from the nearest-neighbour pairing above: a
    nearest-neighbour pairing over K states contributes at most K edges, so it
    is a spanning forest at best and cannot certify bridging even in principle.
    (Measured on the real run's 27-state matrix: 19 unique pairs. Counting
    components of THAT graph is a property of the pairing construction, not of
    the run -- 19 edges over 27 nodes must split into 8 pieces by arithmetic.
    The number below is computed from the full matrix instead.)

    ``thr`` is the caller's own per-space overlap target: the marginal matrix is
    graded at ``--min-neighbor-overlap`` and the joint one at
    ``--min-joint-neighbor-overlap``, since joint overlap is bounded above by
    the marginal and neither space's connectivity implies the other's. On the
    real run the CV1-marginal graph is FULLY CONNECTED at 0.30 while the joint
    graph splits into 3 components at 0.09 -- i.e. computing this on the
    always-available marginal matrix alone would re-commit, inside the fix, the
    marginal blindness the fix exists for.

    States with zero samples are excluded from the graph entirely (reported
    under ``excluded_unsampled_states``): their overlap row is identically zero,
    so each would be its own component and every run with one unsampled window
    would trivially "disconnect" -- a condition the dedicated zero-sample-window
    check already owns. Same convention and same rationale as
    ``cv_space_neighbor_overlap``.

    Non-finite entries are never edges (``nan >= thr`` is False), and the matrix
    is symmetrised when tested (``O[i, j]`` or ``O[j, i]``) so an asymmetric
    input cannot lose a real edge.

    Returns ``{'threshold', 'n_states_graded', 'n_components', 'components',
    'component_samples', 'excluded_unsampled_states',
    'component_best_cross_overlap', 'worst_component_best_cross_overlap',
    'most_isolated_component'}``. ``components`` is a
    list of ascending state-id lists, ordered by lowest member, so the output is
    stable across re-runs; ``component_samples`` is the matching total sample
    count per component (empty when no ``n_k`` was supplied) -- that number is
    what lets an operator tell small-N histogram deflation from a real physical
    gap. ``n_components == 0`` means fewer than one state was eligible, i.e.
    "nothing to grade", not "connected".

    HOW BADLY split is measured too, not just whether. For each component,
    ``component_best_cross_overlap`` gives its strongest overlap with anything
    outside it -- its best escape route -- and
    ``worst_component_best_cross_overlap`` is the smallest of those, belonging
    to ``most_isolated_component`` (ties resolve to the lowest-indexed
    component, the same determinism ``components``' own ordering has, so two
    equally-isolated blocks name one of them stably rather than arbitrarily).
    That distinguishes two very different
    situations a bare component count reports identically: a block that shares
    essentially no samples with anything, whose relative free energy is
    undetermined outright (chignolin_6's isolated state 20 measures 0.0048 to
    anything outside its own component, and 0.00075 to the main block), from a
    block whose only link is a weak-but-real pair that merely misses the target
    (its state 18 measures 0.0648 against a 0.09 target). The first is a hard
    failure; the second is the pairwise CAUTION the overlap check already
    reports, plus the sharper news that the weak pair is the ONLY bridge.

    The MINIMUM over components is the grading number, deliberately, not the
    maximum over cut edges: on the real run those differ and only the minimum is
    right. Measured on that npz, its three components' best escape routes are
    0.06481 / 0.06481 / 0.00480, so a max-over-cut-edges statistic reports
    0.06481 and grades the whole split on state 18's weak-but-real link,
    silently rescuing state 20 -- which shares essentially nothing with anything
    -- from the FAIL it has earned. That is not hypothetical: the first version
    of this code took the max and graded the real run CAUTION. The grading
    itself lives in gareus_report._check_overlap_connectivity; this function
    only measures.
    """
    O = np.asarray(O, dtype=np.float64)
    K = int(O.shape[0]) if (O.ndim == 2 and O.shape[0] == O.shape[1]) else 0
    thr = float(thr)
    eligible = np.ones(K, dtype=bool)
    excluded: list = []
    nk = None
    if n_k is not None:
        arr = np.asarray(n_k).ravel()
        if arr.size >= K:
            nk = arr
            eligible = arr[:K] > 0
            excluded = [int(i) for i in np.flatnonzero(~eligible)]
    idx = np.flatnonzero(eligible)
    out = {'threshold': thr, 'n_states_graded': int(idx.size), 'n_components': 0,
           'components': [], 'component_samples': [], 'excluded_unsampled_states': excluded,
           'component_best_cross_overlap': [], 'worst_component_best_cross_overlap': None,
           'most_isolated_component': None}
    if idx.size == 0:
        return out

    parent = {int(i): int(i) for i in idx}

    def _find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    if idx.size > 1:
        with np.errstate(invalid='ignore'):
            adj = np.isfinite(O) & (O >= thr)
        sub = adj[np.ix_(idx, idx)]
        sub = sub | sub.T
        ii, jj = np.nonzero(np.triu(sub, 1))
        for a, b in zip(ii, jj):
            ra, rb = _find(int(idx[a])), _find(int(idx[b]))
            if ra != rb:
                parent[rb] = ra

    groups: dict = {}
    for i in idx:
        groups.setdefault(_find(int(i)), []).append(int(i))
    comps = sorted((sorted(v) for v in groups.values()), key=lambda c: c[0])
    out['n_components'] = len(comps)
    out['components'] = comps
    if nk is not None:
        out['component_samples'] = [int(sum(int(nk[x]) for x in c)) for c in comps]
    if len(comps) > 1:
        # Per-component best escape route, then the worst of them (see the
        # docstring: the minimum over components, NOT the maximum over cut
        # edges). Every one of these is below `thr` by construction -- that is
        # what made it a split -- so what they add is HOW far below.
        #
        # Done as one O(K^2) masked reduction rather than a loop over component
        # pairs, so a pathologically shattered K=364 geometry stays cheap. The
        # matrix is symmetrised first, so an asymmetric input cannot lose a
        # link. Non-finite entries are ignored, and a component with no finite
        # cross entry at all stays None rather than becoming 0.0 -- absence must
        # not read as a measured zero.
        labels = np.full(K, -1, dtype=np.int64)
        for ci, c in enumerate(comps):
            labels[c] = ci
        sub = O[np.ix_(idx, idx)]
        sub = np.maximum(sub, sub.T)
        lab = labels[idx]
        cross = np.where(np.isfinite(sub) & (lab[:, None] != lab[None, :]), sub, -np.inf)
        per_state_best = cross.max(axis=1)
        per_component = []
        for ci in range(len(comps)):
            rows = per_state_best[lab == ci]
            v = float(rows.max()) if rows.size else float('-inf')
            per_component.append(v if np.isfinite(v) else None)
        out['component_best_cross_overlap'] = per_component
        finite = [v for v in per_component if v is not None]
        if finite:
            worst = min(finite)
            out['worst_component_best_cross_overlap'] = worst
            out['most_isolated_component'] = list(comps[per_component.index(worst)])
    return out


def overlap_component_text(conn: dict, max_components: int = 6,
                           max_states: int = 8) -> str:
    """``[18] (3,410 samples), [20] (6,030 samples), ...`` for one
    ``overlap_components`` result.

    Truncated on both axes because a genuinely shattered K=364 geometry would
    otherwise render an unreadable multi-kilobyte warning line; the counts are
    always published in full under ``overlap_connectivity`` in
    pmf_summary.json.
    """
    comps = list(conn.get('components') or [])
    samples = list(conn.get('component_samples') or [])
    parts = []
    for n, c in enumerate(comps[:int(max_components)]):
        shown = ', '.join(str(x) for x in c[:int(max_states)])
        if len(c) > int(max_states):
            shown += f', ... +{len(c) - int(max_states)} more'
        txt = f'[{shown}]'
        if n < len(samples):
            txt += f' ({samples[n]:,} samples)'
        parts.append(txt)
    if len(comps) > int(max_components):
        parts.append(f'... +{len(comps) - int(max_components)} more components')
    return ', '.join(parts)


def overlap_connectivity_warning_lines(conn: dict, space_label: str,
                                       warning_prefix: str = '') -> list:
    """HIGH-severity warning text (zero or one line) for a split overlap graph.

    Wording is load-bearing: gareus_report.py's ``_WARN_RULES`` matches
    "overlap graph is DISCONNECTED" to classify this HIGH (an unmatched warning
    silently defaults to MEDIUM, i.e. hidden from the terminal health block),
    and the per-component sample counts are in the line itself because that is
    the number that distinguishes small-N joint-histogram deflation from a real
    physical gap.
    """
    if not conn.get('available', True):
        return []
    n = int(conn.get('n_components') or 0)
    if n <= 1:
        return []
    best = conn.get('worst_component_best_cross_overlap')
    iso = conn.get('most_isolated_component')
    best_txt = ''
    if best is not None:
        iso_txt = (f' ({overlap_component_text({"components": [iso]})})'
                   if iso else '')
        best_txt = (f'The most isolated component{iso_txt} has at most '
                    f'{float(best):.4f} overlap with anything outside it. ')
    return [
        f'{warning_prefix}Umbrella overlap graph is DISCONNECTED in the {space_label} space: '
        f'{n} components at overlap >= {float(conn.get("threshold", 0.0)):.3f} -- '
        f'{overlap_component_text(conn)}. {best_txt}MBAR fixes free energies only up to one '
        f'additive constant per connected component, so every cross-component free-energy '
        f'difference (the PMF span included) rests on nothing but that one link; on chignolin_6 '
        f'the equivalent number was 0.0008 and it left f_k unconstrained, parking 94% of the '
        f'posterior on a single state. No worst-pair overlap number can show this -- every graded '
        f'pair can clear the threshold while the set stays split. Closing it needs bridging '
        f'windows or softer restraints between the components, i.e. new sampling, not '
        f're-analysis. See docs/chignolin_6_low_ess_root_cause.md.']


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
        # CE2 remains the selected estimator; the convergence check is
        # REPORTED, not acted on.
        #
        # Not acted on because the alternatives are worse, not because the
        # diagnostic is cosmetic. Exponential reweighting is formally exact but
        # its ESS falls off like exp(-(beta*sigma)^2): at the boost width that
        # triggers this warning (beta*sigma ~ 3.6) that is ~1e-6 of N -- of
        # order a hundred effective samples out of tens of millions, which
        # estimates nothing. umbrella_only is well defined but discards the
        # GaMD correction outright. So a non-converging series does not mean
        # "use something else"; it means the truncation error is real and must
        # be quoted alongside the curve.
        _cum_verdict = cumulant_series_verdict(cdiag3, counts=cum3_pmf.get('counts'))
        # computed here rather than further down because the estimator choice
        # depends on it
        e = _agm.ess(exp_w)
        selected, _sel_reason = select_unbiased_method(e, N, gamd_ladder=bool(d.meta.get("gamd_ladder", False)))
        warnings.append(f'{warning_prefix}Unbiased estimator: {_sel_reason}.')
        if not _cum_verdict['converging'] and selected == 'gamd_cumulant2':
            warnings.append(
                f"{warning_prefix}GaMD cumulant expansion is NOT converging on this boost "
                f"(typical terms: 1st {_cum_verdict['term1_kT']:.2f} kT, 2nd "
                f"{_cum_verdict['term2_kT']:.2f} kT, 3rd {_cum_verdict['term3_kT']:.2f} kT, 4th "
                f"{_cum_verdict['term4_kT']:.2f} kT; term3/term2 = {_cum_verdict['ratio_3_over_2']:.2f}, "
                f"term4/term2 = {_cum_verdict['ratio_4_over_2']:.2f}, threshold "
                f"{_cum_verdict['threshold']:.2f}; beta*sigma = {_cum_verdict['beta_sigma']:.2f}). "
                f"The boost is too non-Gaussian for the truncation: the first neglected term is "
                f"comparable to those kept, so gamd_cumulant2 carries a systematic error of order "
                f"{_cum_verdict['term3_kT']:.1f} kT on ABSOLUTE free energies, and gamd_cumulant3 "
                f"is not a refinement of it. CE2 is still selected -- exponential reweighting's ESS "
                f"falls off like exp(-(beta*sigma)^2) and is useless at this width, and "
                f"umbrella_only discards the GaMD correction entirely. "
                f"Free-energy DIFFERENCES are better determined than absolute values, but NOT "
                f"exactly: only the bin-INDEPENDENT part of a neglected term cancels on "
                f"subtraction. The residuals that survive are the between-bin scatters -- 1st "
                f"{_cum_verdict['term1_between_bin_kT']:.2f} kT, 3rd "
                f"{_cum_verdict['term3_between_bin_kT']:.2f} kT, 4th "
                f"{_cum_verdict['term4_between_bin_kT']:.2f} kT -- so quote differences with at "
                f"least that uncertainty, and note these are computed per bin from correlated "
                f"samples, so their effective sample size is well below the raw counts. "
                f"beta*sigma = {_cum_verdict['beta_sigma']:.2f} is outside the regime where "
                f"second-order cumulant reweighting is validated; for quantitative PMF work rerun "
                f"with a narrower GaMD envelope (beta*sigma <~ 2), which fixes this at the source "
                f"rather than at a higher cumulant order.")
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
        # The reweighting factor's own bin-to-bin noise, reported but NOT
        # smoothed away. p = p0*exp(logfac), so a scatter of s means adjacent
        # bins' probabilities differ by e^s from estimator noise alone, and a
        # single excursion can make a bin the PMF minimum and re-reference the
        # whole curve. Smoothing it is available (--gamd-smooth-sigma) but is
        # deliberately not automatic: where the scatter is real structure
        # rather than noise, smoothing erases signal, and where the cumulant
        # series is not converging (checked above) a smooth curve is a tidier
        # wrong answer, not a better one.
        _sc = float(cdiag.get('logfac_scatter_kT', 0.0) or 0.0)
        if _sc > 0.35 and not float(cdiag.get('logfac_smooth_sigma_requested', 0.0) or 0.0):
            warnings.append(
                f"{warning_prefix}GaMD cumulant reweighting factor has a bin-to-bin scatter of "
                f"{_sc:.2f} kT, so adjacent bins' probabilities carry up to "
                f"e^{_sc:.2f}={np.exp(_sc):.1f}x of estimator noise; a single excursion can place "
                f"the PMF minimum on a fluctuation and re-reference the curve. Check whether the "
                f"excursions are statistically significant (compare each bin's variance against "
                f"its neighbours' using its own sample count) before trusting or smoothing them: "
                f"--gamd-smooth-sigma exists but will erase real structure if the variation is real.")
        if e / max(1, N) < EXACT_REWEIGHT_MIN_ESS_FRACTION:
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
    # --- window overlap ----------------------------------------------------
    # FIX A5 (docs/chignolin_6_low_ess_root_cause.md, secondary finding #1):
    # the overlap diagnostic was the primary-CV MARGINAL only. On that run 20 of
    # 27 states shared just two primary centres and differed only in the
    # secondary CV, where centre spacing / sigma was 2.6-9.3 -- essentially
    # disjoint -- yet the marginal diagnostic reported 0.6-0.99 and the health
    # check declared window overlap fine. It was structurally unable to see the
    # axis that had failed.
    #
    # The joint (CV1, CV2) overlap therefore has to exist. It is published
    # ALONGSIDE the marginal, never in place of it, and this is not tidiness:
    #
    #  * Every historical key/column/file keeps carrying the marginal number.
    #    `neighbor_overlap`, `overlap_matrix.csv`, `overlap_matrix.png` and
    #    window_diagnostics.csv's overlap_left/overlap_right have meant "CV1
    #    marginal" for the whole life of this pipeline; silently switching the
    #    *space* of an existing key makes two runs analysed either side of the
    #    change incomparable with nothing on disk to tell them apart.
    #  * Joint overlap is bounded above by the CV1 marginal (a refinement of the
    #    histogram can only reduce sum_c min(p_c, q_c)) and deflates further at
    #    small per-state N. So --min-neighbor-overlap 0.30, calibrated against
    #    marginal numbers, is simply not a threshold for it: reusing it would
    #    have flipped already-published 2D runs' PASS/CAUTION/FAIL verdicts on
    #    data that had not changed. The joint number gets its own threshold
    #    (see _joint_threshold below) and its own distinctly-worded warning.
    #
    # "Is this a 2D run" cannot be a `d.cv2 is None` test: Data.cv2 is
    # non-Optional and is a NaN-filled column for a CV1-only run (CLAUDE.md,
    # 2026-08-15). It also cannot be "does cv2 have finite values", since cv2 is
    # sometimes recorded as a plain observable with no restraint on it at all --
    # overlapping in an unbiased axis would report a gap the sampler was never
    # asked to bridge. Both conditions must hold: real cv2 samples AND at least
    # one window carrying a real secondary restraint.
    _sec_centers, _sec_ks = _secondary_window_params(d.meta, K)
    _n_total = int(d.cv.size)
    _cv2_finite_frac = (float(np.count_nonzero(np.isfinite(d.cv2))) / _n_total) if _n_total else 0.0
    _has_secondary_restraint = bool(np.any(np.isfinite(_sec_centers) & np.isfinite(_sec_ks) & (_sec_ks > 0.0)))
    _is_2d_run = _cv2_finite_frac > 0.0 and _has_secondary_restraint

    # The reported matrix: the CV1 marginal, via the untouched 4-argument call,
    # so it is bit-identical to what every previous release produced.
    O = _agm.overlap_matrix(d.cv, d.window, bins, K)

    # Should the joint matrix be built at all, and why not if not? The reason
    # string is published (joint_overlap['reason']) rather than only warned
    # about, so an absent joint number is never just an absence.
    _joint_disabled = bool(getattr(args, 'no_joint_overlap', False))
    _bins2 = None
    _joint_reason = ''
    if _joint_disabled:
        _joint_reason = 'disabled by --no-joint-overlap'
    elif not _is_2d_run:
        _joint_reason = ('this run has no biased second axis: no window carries a finite positive '
                         'secondary restraint constant, and/or no sample has a finite secondary CV')
    elif _cv2_finite_frac < OVERLAP_CV2_MIN_RETAINED_FRACTION:
        # Gate on the cv2-specific retention, not on overlap_matrix's combined
        # retained_fraction: that number also drops samples outside the PRIMARY
        # bin range, so it can fall below the threshold for reasons that have
        # nothing to do with the secondary CV and skip the joint matrix for the
        # wrong reason.
        _joint_reason = (f'only {_cv2_finite_frac*100:.1f}% of samples carry a finite secondary CV '
                         f'(below {OVERLAP_CV2_MIN_RETAINED_FRACTION*100:.0f}%), so a joint matrix '
                         f'would describe a minority of the population while looking exactly as '
                         f'authoritative as the marginal one')
    else:
        _bins2 = make_overlap_bins(d.cv2)
        if _bins2 is None:
            # make_overlap_bins returned None: fewer than two distinct finite
            # cv2 values, i.e. a secondary axis with no information to overlap
            # on.
            _joint_reason = ('the secondary CV has fewer than two distinct finite values, so there '
                             'is no secondary range to overlap in')

    _ostats: dict = {}
    O_joint = None
    if _bins2 is not None:
        O_joint = _agm.overlap_matrix(d.cv, d.window, bins, K, cv2=d.cv2, bins2=_bins2,
                                      stats=_ostats)
        if int(_ostats.get('dim', 1)) != 2:
            # Defensive: overlap_matrix falls back to the marginal on a wholly
            # non-finite cv2. The retention gate above already excludes that,
            # but never publish a matrix under the joint label without the
            # function itself confirming which space it computed.
            O_joint = None
            _joint_reason = 'overlap_matrix fell back to the CV1 marginal (no finite secondary-CV sample)'

    # The joint threshold. Default: the marginal target applied independently on
    # each axis (thr ** dim), i.e. "as well resolved on both axes as
    # --min-neighbor-overlap asks for on one". 0.30 marginal corresponds to
    # centre spacing ~2.1 sigma on a Gaussian pair, so 0.09 joint asks for that
    # same per-axis quality in 2D -- a derived number, not the marginal number
    # reused. Overridable outright with --min-joint-neighbor-overlap.
    _marg_thr = float(getattr(args, 'min_neighbor_overlap', 0.30))
    _joint_thr_override = getattr(args, 'min_joint_neighbor_overlap', None)
    _joint_thr = (float(_joint_thr_override) if _joint_thr_override is not None
                  else _marg_thr ** 2)

    if _is_2d_run and O_joint is None and not _joint_disabled:
        warnings.append(
            f'{warning_prefix}Joint (CV1, CV2) window overlap was NOT computed: {_joint_reason}. '
            f'The reported window overlap is the CV1 marginal, which CANNOT see a secondary-CV gap '
            f'-- the exact blindness that let chignolin_6 report 0.6-0.99 overlap for essentially '
            f'disjoint states. See docs/chignolin_6_low_ess_root_cause.md.')
    elif _is_2d_run and _joint_disabled:
        warnings.append(
            f'{warning_prefix}Joint (CV1, CV2) window overlap is disabled by --no-joint-overlap; the '
            f'reported overlap is the CV1 marginal, which cannot see a secondary-CV gap.')
    elif O_joint is not None and _cv2_finite_frac < OVERLAP_CV2_WARN_RETAINED_FRACTION:
        # Deliberately left to gareus_report's MEDIUM default: the joint number
        # IS usable here (>= half the population), this is a caveat on its
        # coverage, not a defect that biases the PMF.
        warnings.append(
            f'{warning_prefix}Joint (CV1, CV2) window overlap was computed from '
            f'{_cv2_finite_frac*100:.1f}% of samples; the rest have no finite secondary CV (e.g. a '
            f'mid-campaign secondary-CV regime switch) and are excluded, never binned as 0.0.')

    n_k_local, _mean_per_window, _std_per_window = _window_cv_mean_std(d.window, d.cv, K)
    _has_samples = n_k_local > 0

    # Consecutive-index neighbour overlap, CV1 marginal. KEPT as-is (same key,
    # same formula, same threshold, same wording): other code and every
    # historical pmf_summary.json read it, and re-analysing an old run must not
    # invent new warnings. It is no longer the headline number -- see the
    # CV-space block below for why.
    neigh = [float(O[i, i + 1]) for i in range(K - 1)]
    bad = [i for i, x in enumerate(neigh) if x < args.min_neighbor_overlap]
    if bad:
        warnings.append(f'{warning_prefix}Weak neighbor CV overlap below %.2f for pairs: ' % args.min_neighbor_overlap + ', '.join(f'{i}-{i + 1} ({neigh[i]:.2f})' for i in bad))

    # FIX A6: the same worst-overlap question asked of each state's true nearest
    # neighbour in (primary, secondary) restraint-centre space. The index-order
    # version above reported "worst 0.013 (pair 23-24)" on the real run and sent
    # the investigation at state 24, which was innocent and well overlapped --
    # 23 and 24 are simply adjacent in the index, not in CV space.
    def _weak_pairs(rows, thr):
        """Unique {(lo, hi): overlap} pairs below ``thr``, ascending by index."""
        return sorted({(min(r['window'], r['neighbor']), max(r['window'], r['neighbor'])): r['overlap']
                       for r in rows if r['overlap'] < thr}.items())

    cv_neigh = cv_space_neighbor_overlap(O, d.centers, d.k_kcal, _sec_centers, _sec_ks,
                                         kbt_kcal, n_k=n_k_local)
    _cv_neigh_by_window = {int(r['window']): r for r in cv_neigh}
    _cv_bad_pairs = _weak_pairs(cv_neigh, _marg_thr)
    if _cv_bad_pairs:
        warnings.append(
            f'{warning_prefix}Weak CV-space nearest-neighbour overlap below '
            f'%.2f (CV1 marginal) for pairs: ' % _marg_thr
            + ', '.join(f'{i}-{j} ({x:.2f})' for (i, j), x in _cv_bad_pairs)
            + '. These are nearest neighbours in restraint-centre space (sigma units), not index '
              'neighbours; a weak pair here is a genuinely unbridged gap.')

    # The joint block: same two pairings (index-adjacent and CV-space nearest
    # neighbour) recomputed on the joint matrix, published under their own key
    # and gated by their own threshold.
    joint_info: dict = {'available': False, 'reason': _joint_reason,
                        'space': OVERLAP_SPACE_JOINT, 'threshold': _joint_thr,
                        'secondary_cv_finite_fraction': _cv2_finite_frac}
    cv_neigh_joint: list = []
    if O_joint is not None:
        cv_neigh_joint = cv_space_neighbor_overlap(O_joint, d.centers, d.k_kcal, _sec_centers,
                                                  _sec_ks, kbt_kcal, n_k=n_k_local)
        _joint_bad_pairs = _weak_pairs(cv_neigh_joint, _joint_thr)
        if _joint_bad_pairs:
            warnings.append(
                f'{warning_prefix}Weak joint (CV1, CV2) nearest-neighbour overlap below '
                f'%.3f for pairs: ' % _joint_thr
                + ', '.join(f'{i}-{j} ({x:.3f})' for (i, j), x in _joint_bad_pairs)
                + f'. Measured in the full (CV1, CV2) histogram, so unlike the CV1-marginal number '
                  f'this sees a secondary-CV gap; the target is --min-neighbor-overlap '
                  f'({_marg_thr:.2f}) applied on both axes, NOT the marginal target itself.')
        _worst_joint = min(cv_neigh_joint, key=lambda r: r['overlap']) if cv_neigh_joint else None
        joint_info.update({
            'available': True, 'reason': '', 'dim': int(_ostats.get('dim', 2)),
            'neighbor_overlap': [float(O_joint[i, i + 1]) for i in range(K - 1)],
            'cv_space_neighbor_overlap': cv_neigh_joint,
            'worst_cv_space_pair': (dict(_worst_joint) if _worst_joint else None),
            'weak_cv_space_pairs': [{'window': i, 'neighbor': j, 'overlap': float(x)}
                                    for (i, j), x in _joint_bad_pairs],
            'stats': dict(_ostats),
            'files': {'overlap_matrix_joint_csv': str(out / 'overlap_matrix_joint.csv')},
        })
    _joint_by_window = {int(r['window']): r for r in cv_neigh_joint}

    # ROUND-3 FINDING 1, deeper half: grade CONNECTIVITY, not only the worst
    # pair. Every overlap number above this point is pairwise, and a pairwise
    # statistic structurally cannot see the failure this whole investigation is
    # about -- MBAR fixes free energies only up to one additive constant per
    # connected component of the overlap graph, so a split set has undetermined
    # offsets between its blocks no matter how good each graded pair looks. See
    # overlap_components() for the full argument and for the real-run numbers.
    #
    # Computed in BOTH spaces, each against its own threshold, because neither
    # implies the other: joint overlap is bounded above by the marginal, while
    # the joint THRESHOLD (0.09 by default) is below the marginal one (0.30).
    # On chignolin_6's real 27-state matrices the marginal graph is fully
    # connected at 0.30 and the joint graph splits into 3 components at 0.09
    # ([18] 3,410 samples | [20] 6,030 | the other 25 states 117,030), so a
    # marginal-only connectivity check would be silent on the very run it was
    # written for.
    _conn_marginal = overlap_components(O, _marg_thr, n_k=n_k_local)
    _conn_marginal.update({'space': OVERLAP_SPACE_MARGINAL, 'available': True, 'reason': ''})
    if O_joint is not None:
        _conn_joint = overlap_components(O_joint, _joint_thr, n_k=n_k_local)
        _conn_joint.update({'space': OVERLAP_SPACE_JOINT, 'available': True, 'reason': ''})
    else:
        # Never hand back only the marginal's clean bill of health: an absent
        # joint component count means the connectivity is blind on exactly the
        # axis that failed on the real run, so it carries the reason with it
        # (the same string joint_overlap['reason'] publishes) and
        # gareus_report's check reports it as unknown rather than as connected.
        _conn_joint = {'space': OVERLAP_SPACE_JOINT, 'available': False,
                       'reason': (_joint_reason
                                  or 'the joint (CV1, CV2) overlap matrix was not computed'),
                       'threshold': _joint_thr, 'n_states_graded': 0, 'n_components': 0,
                       'components': [], 'component_samples': [],
                       'excluded_unsampled_states': []}
    connectivity = {'marginal': _conn_marginal, 'joint': _conn_joint}
    warnings.extend(overlap_connectivity_warning_lines(
        _conn_marginal, 'CV1-marginal', warning_prefix))
    warnings.extend(overlap_connectivity_warning_lines(
        _conn_joint, 'joint (CV1, CV2)', warning_prefix))
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
    # Both matrices carry a leading `#` comment line naming the space their
    # numbers are in. This is the only thing on disk that distinguishes a
    # CV1-marginal overlap from a joint one -- they are the same shape, the same
    # scale and the same file name otherwise. Readers must therefore pass
    # comment='#' (pandas) or rely on the default `#` skipping (np.loadtxt); the
    # `window,0,1,...` header row itself is unchanged and still the second line.
    def _write_overlap_csv(path: Path, mat, stamp: str):
        with path.open('w', newline='') as f:
            f.write(f'# {stamp}\n')
            wr = csv.writer(f)
            wr.writerow(['window'] + list(range(K)))
            for i in range(K):
                wr.writerow([i] + [float(x) for x in mat[i]])

    _write_overlap_csv(
        out / 'overlap_matrix.csv', O,
        f'overlap space: CV1 marginal ({len(bins) - 1} primary bins); pairwise '
        f'histogram-intersection overlap of the primary CV only. This CANNOT see a '
        f'secondary-CV gap: for the joint (CV1, CV2) overlap see overlap_matrix_joint.csv '
        f'(written only when this run has a biased second axis). '
        f'docs/chignolin_6_low_ess_root_cause.md')
    if O_joint is not None:
        _write_overlap_csv(
            out / 'overlap_matrix_joint.csv', O_joint,
            f'overlap space: joint (CV1, CV2) 2D histogram '
            f'({_ostats.get("n_bins_primary", len(bins) - 1)} x '
            f'{_ostats.get("n_bins_secondary", 0)} bins), computed from '
            f'{_cv2_finite_frac*100:.1f}% of samples (the rest have no finite secondary CV and are '
            f'excluded, never binned as 0.0). Bounded above by the CV1 marginal in '
            f'overlap_matrix.csv, so it is NOT comparable with --min-neighbor-overlap '
            f'({_marg_thr:.2f}); its target is {_joint_thr:.3f}. '
            f'docs/chignolin_6_low_ess_root_cause.md')
    # FIX A4: per-state self-bias -- each state's own samples scored in its own
    # restraint. See self_bias_diagnostics() for why ~1 kT is the only physical
    # answer and why this is the one check that would have caught the whole
    # chignolin_6 mis-mapping on its own.
    self_bias = self_bias_diagnostics(d.u_nk, d.window, K)
    warnings.extend(self_bias_warning_lines(self_bias, warning_prefix))

    # FIX A12: observed per-window secondary CV. Computed over the cv2-FINITE
    # rows only -- _window_cv_mean_std propagates NaN across a whole window if
    # any sample in it is non-finite, which for a partially-null cv2 column
    # (routine after a secondary-CV regime switch) would blank every window.
    _cv2_finite_mask = np.isfinite(d.cv2)
    _n_k_cv2, _cv2_mean_per_window, _cv2_std_per_window = _window_cv_mean_std(
        d.window[_cv2_finite_mask], d.cv2[_cv2_finite_mask], K)
    _has_cv2 = _n_k_cv2 > 0

    def _f(arr, k, ok=True):
        """CSV cell: the float, or '' for absent/non-finite (the writer's
        existing convention -- never the string 'nan')."""
        return float(arr[k]) if (ok and k < len(arr) and np.isfinite(arr[k])) else ''

    with (out / 'window_diagnostics.csv').open('w', newline='') as f:
        # New columns (2026-08-25): the SECONDARY restraint and its observed
        # spread (A12 -- the axis that actually failed on chignolin_6 was
        # completely invisible to the operator here), per-state self-bias (A4),
        # and each window's CV-space nearest neighbour with that pair's overlap
        # in BOTH spaces (A6).
        #
        # secondary_center_window_table / secondary_k_kcal_mol_window_table name
        # their provenance in the header rather than leaving it to be guessed.
        # They are the loader's own per-window table (_secondary_window_params
        # -> meta['umbrella_window_rows']: the union-Parquet loader's
        # final_registry_used_for_mbar.csv, load_csv's umbrella_windows.csv, and
        # blank for the legacy epoch-CSV path), i.e. ONE snapshot applied to
        # every epoch. self_bias_median_kT beside them is computed from each
        # epoch's OWN native window params (loaders_union_parquet's per-epoch
        # blocks), and cv2_mean/cv2_std are observed. On a run that recentred
        # its windows mid-campaign -- the tICA CV2 switch overwrites
        # primary/secondary centres in place, which is exactly what the
        # 2026-08-04 per-epoch-bias fix exists for -- the snapshot columns and
        # the observed/native ones legitimately disagree, and that disagreement
        # is a recentring, NOT the mapping fault this table was added to expose.
        # The historical center_A / k_kcal_mol_A2 columns come from the same
        # snapshot (d.centers / d.k_kcal are built from those same rows) and
        # keep their long-standing names for backward compatibility; only the
        # new columns get to say so in the header.
        #
        # overlap_left/overlap_right keep their historical meaning exactly:
        # index-adjacent, CV1 marginal. Every new overlap column names its own
        # space instead, because marginal and joint overlaps are the same shape
        # and scale and are otherwise indistinguishable once written out --
        # cv_space_overlap_joint is blank for a run with no biased second axis
        # (or with --no-joint-overlap).
        wr = csv.DictWriter(f, fieldnames=[
            'window', 'center_A', 'k_kcal_mol_A2',
            'secondary_center_window_table', 'secondary_k_kcal_mol_window_table',
            'samples', 'cv_mean_A', 'cv_std_A', 'cv2_mean', 'cv2_std',
            'self_bias_median_kT', 'self_bias_p90_kT',
            'overlap_left', 'overlap_right', 'cv_space_neighbor',
            'cv_space_overlap_marginal', 'cv_space_overlap_joint'])
        wr.writeheader()
        for k in range(K):
            _cvn = _cv_neigh_by_window.get(k)
            wr.writerow({'window': k,
                         'center_A': _f(d.centers, k),
                         'k_kcal_mol_A2': _f(d.k_kcal, k),
                         'secondary_center_window_table': _f(_sec_centers, k),
                         'secondary_k_kcal_mol_window_table': _f(_sec_ks, k),
                         'samples': int(n_k_local[k]),
                         'cv_mean_A': _f(_mean_per_window, k, _has_samples[k]),
                         'cv_std_A': _f(_std_per_window, k, _has_samples[k]),
                         'cv2_mean': _f(_cv2_mean_per_window, k, _has_cv2[k]),
                         'cv2_std': _f(_cv2_std_per_window, k, _has_cv2[k]),
                         'self_bias_median_kT': _f(self_bias['median_kT'], k),
                         'self_bias_p90_kT': _f(self_bias['p90_kT'], k),
                         'overlap_left': float(O[k - 1, k]) if k > 0 else '',
                         'overlap_right': float(O[k, k + 1]) if k + 1 < K else '',
                         'cv_space_neighbor': _cvn['neighbor'] if _cvn else '',
                         'cv_space_overlap_marginal': f"{_cvn['overlap']:.6f}" if _cvn else '',
                         'cv_space_overlap_joint': (f"{_joint_by_window[k]['overlap']:.6f}"
                                                    if k in _joint_by_window else '')})
    if progress is not None:
        progress.bar('analysis stages', 5, 6, 'plotting PNG outputs', force=True)
    _agm.plot_outputs(d, pmfs, selected, O, out, warnings, smooth_sigma=_agm._eff_smooth(args, 'pmf_smooth_sigma'), args=args)
    return {
        'pmfs': pmfs, 'selected': selected, 'boost_ok': boost_ok, 'boost': bs,
        'pmf_span_kcal_mol': span, 'pmf_minimum_cv_A': float(sel['cv_A'][minidx]) if minidx >= 0 else None,
        # 'neighbor_overlap' and 'O' are the historical consecutive-INDEX array
        # and matrix, both CV1-MARGINAL, unchanged in space, formula and
        # threshold (see the overlap block above for why that continuity is
        # load-bearing). 'overlap_space' stamps that fact into
        # pmf_summary.json so it is never again inferable only from which
        # commit the analysis ran at.
        #
        # 'cv_space_neighbor_overlap' is the marginal matrix re-paired by true
        # CV-space adjacency; 'joint_overlap' carries the (CV1, CV2) numbers
        # with their own threshold. gareus_report.build_health_verdict consumes
        # both and takes the worse -- joint overlap is bounded above by the
        # marginal, so neither implies the other.
        #
        # EVERY key below must be propagated into pmf_summary.json by
        # analyze_gareus_mbar._report_summary_fields(); round 1 added them here
        # and never edited analyze()'s summary literal, so the health verdict
        # silently kept using the index-adjacency fallback on real runs.
        'neighbor_overlap': neigh, 'n_samples': N, 'O': O,
        'overlap_space': OVERLAP_SPACE_MARGINAL,
        'cv_space_neighbor_overlap': cv_neigh,
        'self_bias': self_bias,
        'joint_overlap': joint_info,
        # Connectivity of the whole overlap graph, both spaces (see the
        # overlap_components() call above). This is the only key here that
        # can express "MBAR has no information linking these two blocks of
        # states"; every other overlap key is pairwise and cannot.
        'overlap_connectivity': connectivity,
        'secondary_cv_finite_fraction': _cv2_finite_frac,
        'pmf_uncertainty_std': pmf_uncertainty_std,
        'files': {
            'pmf_unbiased_csv': str(out / 'pmf_unbiased.csv'), 'pmf_all_methods_csv': str(out / 'pmf_all_methods.csv'),
            'pmf_umbrella_only_csv': str(out / 'pmf_umbrella_only.csv'), 'pmf_gamd_exponential_csv': str(out / 'pmf_gamd_exponential.csv'),
            'pmf_gamd_cumulant2_csv': str(out / 'pmf_gamd_cumulant2.csv'), 'pmf_gamd_cumulant3_csv': str(out / 'pmf_gamd_cumulant3.csv'),
            'overlap_matrix_csv': str(out / 'overlap_matrix.csv'), 'window_diagnostics_csv': str(out / 'window_diagnostics.csv'),
            **(joint_info.get('files') or {}),
        },
    }


def _regime_independent_logw(d_regime: 'Data', args, _agm) -> tuple:
    """Solve MBAR for ONE secondary-CV regime, using only that regime's rows.

    Returns ``(logw, source)``, or ``(None, reason)`` when the regime has no
    usable estimator.

    Why a fresh solve rather than reweighting the regime out of the pooled
    ``f_k``: across a CV2 regime change the pooled solve is not a single
    Hamiltonian. Every row block is evaluated against its own regime's cv2,
    so a state's column mixes two different bias definitions, and states
    created after the switch have no meaningful value at all on pre-switch
    rows. ``f_k`` from that solve is therefore not a property of one
    Hamiltonian's population, which is precisely the assumption
    ``_subset_logw_from_global_fk`` documents and relies on.

    Restricting the ROWS to one regime also drops the COLUMNS of states that
    hold no samples there, because ``active = where(n_k > 0)`` -- so states
    created after the switch leave the denominator instead of contributing a
    value for a state that never existed on these rows. That column dropping
    is standard MBAR behaviour, not something this function adds: pymbar omits
    zero-sample states from the linear system too. The correctness gain here
    is the CONSISTENT ``u_nk`` evaluation (every row and column in one CV2
    definition); the pruning is a welcome consequence of partitioning, not the
    reason it is right.

    A regime is only ~6% of samples in the motivating run, so a solve that
    does not converge is a real outcome, not a theoretical one. It is
    reported as unavailable; callers must not substitute the pooled f_k.
    """
    try:
        mb = _agm.solve_mbar(
            d_regime.u_nk, d_regime.window,
            tol=float(getattr(args, 'mbar_tol', 1e-10)),
            backend=getattr(args, 'mbar_backend', 'auto'),
            threads=getattr(args, 'mbar_threads', 0),
            progress=None,
        )
    except Exception as exc:  # a thin regime can fail outright
        return None, f'unavailable:solve_failed:{type(exc).__name__}'
    if not mb.get('converged'):
        return None, f"unavailable:not_converged:max_delta={float(mb.get('max_delta', float('nan'))):.3e}"
    logw = np.asarray(mb['logw'], dtype=np.float64)
    if not np.any(np.isfinite(logw)):
        return None, 'unavailable:no_finite_logw'
    return logw, 'per_regime_solve'


def run_secondary_cv_analyses(d: 'Data', args, base_logw: np.ndarray, selected: str,
                               boost_ok: bool, kbt_kcal: float, out: Path,
                               warnings: list, progress: Optional['Progress'],
                               f_k_global: Optional[np.ndarray] = None) -> tuple:
    """Secondary-CV PMF + CV1xCV2 2D FES, split by secondary-CV regime when the
    run's CV2 definition changed mid-campaign (see
    ``_secondary_cv_epoch_regime_masks``). Single-regime runs (the common
    case) are entirely unaffected: this degrades to the plain unmodified
    calls, writing to the same paths as before.

    When multiple regimes exist, each regime gets its OWN independent MBAR
    solve over only its own rows (``_regime_independent_logw``). Neither
    ``f_k_global`` nor ``base_logw`` is used to build a regime's weights.

    This replaces reweighting each regime out of the pooled ``f_k``. That
    approach fixed the binning axis but not the estimator: across a CV2
    regime change the pooled solve splices two Hamiltonians into every
    column, so its ``f_k`` is not the whole-population property that
    ``_subset_logw_from_global_fk`` assumes, and the non-dominant regime's
    PMF came out as one regime's ``f_k`` applied to the other regime's
    ``u_nk``. On the motivating run the two CV2 definitions' weight vectors
    have cosine similarity 0.141 (81.9 degrees apart) in the shared 36-dim
    torsion feature space, so the substitution replaces the coordinate with a
    nearly orthogonal one rather than perturbing it. (A Pearson r of +0.158
    between the two scalar CVs was quoted previously; that measures linear
    association of CV VALUES, not the geometry of the projections, and is the
    wrong statistic for this claim even though it lands on a similar number.)

    ``f_k_global`` is retained in the signature for callers that still pass
    it, and is deliberately unused: there is no correct way to derive a
    regime's weights from a cross-regime solve. A regime whose own solve
    does not converge is reported unavailable rather than falling back.

    Every returned info dict carries ``secondary_cv_logw_source``, so a
    downstream consumer can tell a regime-internal estimate from an absent
    one without inferring it from the file layout.

    Returns ``(secondary_cv_pmf_info, cv1_cv2_fes_info)`` for the *dominant*
    regime (or the only regime, if there's just one) -- same shape/keys
    downstream code already expects. When multiple regimes exist, each gets
    its own analysis written under ``out/secondary_cv_regime_<type>/``, and
    both dominant-regime info dicts additionally carry a ``regime_breakdown``
    key with every regime's own info (including the dominant one).
    """
    _agm = _bridge()
    regimes = _agm._secondary_cv_epoch_regime_masks(d, warnings=warnings)
    if regimes is None:
        pmf_info = analyze_secondary_cv_pmf(d, args, base_logw, selected, boost_ok, kbt_kcal, out, warnings, progress)
        fes_info = (_agm.analyze_cv1_cv2_2d_fes(d, args, base_logw, selected, boost_ok, kbt_kcal, out, warnings, progress)
                    if isinstance(pmf_info, dict) and pmf_info.get('available')
                    else {'available': False, 'reason': 'Secondary CV PMF unavailable'})
        return pmf_info, fes_info

    breakdown: dict = {}
    dominant_pmf_info = dominant_fes_info = None
    for regime, (mask, is_dominant) in regimes.items():
        _orig_secondary_cv = d.meta.get('secondary_cv')
        if isinstance(_orig_secondary_cv, dict):
            _regime_secondary_cv = dict(_orig_secondary_cv)
            _regime_secondary_cv['mode'] = regime
        else:
            _regime_secondary_cv = regime
        regime_meta = dict(d.meta); regime_meta['secondary_cv'] = _regime_secondary_cv
        d_regime = _agm._masked_data(d, mask, meta_override=regime_meta)
        base_logw_regime, logw_source = _regime_independent_logw(d_regime, args, _agm)
        regime_out = out if is_dominant else out / f'secondary_cv_regime_{_agm._regime_slug(regime)}'
        if base_logw_regime is None:
            # No usable estimator for this regime. Emit NOTHING rather than
            # falling back to the pooled f_k: across a regime change that
            # fallback is the defect this function exists to avoid, and a
            # silently wrong curve is worse than an absent one.
            reason = (f'No regime-internal MBAR estimator for secondary-CV regime '
                      f'{regime!r} ({logw_source}); refusing to reweight it with the '
                      f'pooled f_k, whose state free energies are not valid across a '
                      f'CV2 regime change.')
            warnings.append(reason)
            pmf_info = {'available': False, 'reason': reason,
                        'secondary_cv_logw_source': logw_source}
            fes_info = {'available': False, 'reason': reason,
                        'secondary_cv_logw_source': logw_source}
        else:
            regime_out.mkdir(parents=True, exist_ok=True)
            pmf_info = analyze_secondary_cv_pmf(d_regime, args, base_logw_regime, selected, boost_ok, kbt_kcal, regime_out, warnings, progress)
            fes_info = (_agm.analyze_cv1_cv2_2d_fes(d_regime, args, base_logw_regime, selected, boost_ok, kbt_kcal, regime_out, warnings, progress)
                        if isinstance(pmf_info, dict) and pmf_info.get('available')
                        else {'available': False, 'reason': 'Secondary CV PMF unavailable'})
            for _info in (pmf_info, fes_info):
                if isinstance(_info, dict):
                    _info['secondary_cv_logw_source'] = logw_source
        # Store shallow copies in the breakdown, not the live dicts -- the
        # dominant regime's own pmf_info/fes_info get a 'regime_breakdown' key
        # added to them below, and aliasing the same object here would nest
        # that dict inside itself (a real circular reference JSON serialization
        # rejects; caught by an actual end-to-end run against chignolin_5).
        breakdown[regime] = {
            'is_dominant': is_dominant, 'n_samples': int(np.count_nonzero(mask)),
            'secondary_cv_logw_source': logw_source,
            'secondary_cv_pmf': dict(pmf_info) if isinstance(pmf_info, dict) else pmf_info,
            'cv1_cv2_2d_fes': dict(fes_info) if isinstance(fes_info, dict) else fes_info,
        }
        if is_dominant:
            dominant_pmf_info, dominant_fes_info = pmf_info, fes_info

    if isinstance(dominant_pmf_info, dict):
        dominant_pmf_info['regime_breakdown'] = breakdown
    if isinstance(dominant_fes_info, dict):
        dominant_fes_info['regime_breakdown'] = breakdown
    return dominant_pmf_info, dominant_fes_info


def analyze_secondary_cv_pmf(d: Data, args, base_logw: np.ndarray, selected: str, boost_ok: bool, kbt_kcal: float, out: Path, warnings: list[str], progress: Optional['Progress']) -> dict:
    """1D PMF along the secondary collective variable (cv2_A from samples.csv)."""
    _agm = _bridge()
    cv2=np.asarray(d.cv2, dtype=np.float64)
    mask=np.isfinite(cv2) & np.isfinite(base_logw)
    if np.count_nonzero(mask) < max(20, d.u_nk.shape[1]):
        return {'available': False, 'reason': 'Too few finite secondary CV (cv2_A) samples', 'n_finite': int(np.count_nonzero(mask))}
    cv2_sel=cv2[mask]
    boost_sel=d.boost_kj[mask]
    base_logw_sel=np.asarray(base_logw, dtype=np.float64)[mask]
    base_w=_agm.norm_logw(base_logw_sel)
    bins_n=int(getattr(args,'cv2_bins',None) or args.bins)
    bins=make_bins(cv2_sel, bins_n, getattr(args,'cv2_min',None), getattr(args,'cv2_max',None))
    umbrella=pmf_from_weights(cv2_sel, base_w, bins, kbt_kcal)
    if boost_ok and np.isfinite(boost_sel).sum()>10 and np.nanstd(boost_sel)>1e-12:
        exp_w=_agm.norm_logw(base_logw_sel + d.beta*boost_sel)
        exp_pmf=pmf_from_weights(cv2_sel, exp_w, bins, kbt_kcal)
        (cum_pmf,cdiag),(cum3_pmf,cdiag3)=_cumulant_expansion_both(cv2_sel, base_w, boost_sel, bins, d.beta, kbt_kcal, smooth_logfac_sigma=_agm._eff_smooth(args,'gamd_smooth_sigma'))
        chosen=choose_site_method(selected, ladder_excluded_methods(bool(d.meta.get('gamd_ladder', False))))
    else:
        exp_pmf=umbrella; cum_pmf=umbrella; cum3_pmf=umbrella
        cdiag={'boost_mean_kj':np.full(len(bins)-1,np.nan),'boost_var_kj2':np.full(len(bins)-1,np.nan)}
        cdiag3=cdiag
        chosen='umbrella_only'
    pmfs={'umbrella_only':umbrella,'gamd_exponential':exp_pmf,'gamd_cumulant2':cum_pmf,'gamd_cumulant3':cum3_pmf}
    chosen_pmf=pmfs.get(chosen, umbrella)
    _agm.write_cv2_pmf(out/'cv2_pmf_unbiased.csv', chosen_pmf, chosen)
    _agm.write_cv2_pmf(out/'cv2_pmf_umbrella_only.csv', umbrella, 'umbrella_only')
    _agm.write_cv2_pmf(out/'cv2_pmf_gamd_exponential.csv', exp_pmf, 'gamd_exponential')
    _agm.write_cv2_pmf(out/'cv2_pmf_gamd_cumulant2.csv', cum_pmf, 'gamd_cumulant2')
    _agm.write_cv2_pmf(out/'cv2_pmf_gamd_cumulant3.csv', cum3_pmf, 'gamd_cumulant3')
    plot_file=None
    cv2_label=_agm._secondary_cv_label(d.meta)
    regions=_agm._secondary_cv_regions(d.meta)
    try:
        import matplotlib.pyplot as plt
        import gareus.mbar_analysis.plotstyle as ps
        fig,ax=plt.subplots(figsize=(8,5))
        _cv2_smooth=_agm._eff_smooth(args,'pmf_smooth_sigma')
        for i,(name,p) in enumerate(_agm._visible_pmfs(pmfs, chosen, args).items()):
            pmf_plot=_agm._smooth_pmf_1d(p['pmf'],_cv2_smooth); m=np.isfinite(pmf_plot)
            if np.any(m): ps.plot_method_curve(ax,p['cv_A'][m],pmf_plot[m],name,chosen,idx=i)
        for reg in regions:
            v=float(reg.get('value',float('nan'))); lbl=str(reg.get('label',''))
            if np.isfinite(v):
                ax.axvline(v, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
                ax.text(v, ax.get_ylim()[1] if ax.get_ylim()[1] != ax.get_ylim()[0] else 0, lbl,
                        rotation=90, va='top', ha='right', fontsize=7, color='gray')
        ps.style_line_axes(ax,xlabel=cv2_label,ylabel='PMF (kcal/mol, shifted)',title=f'{cv2_label} PMF ({ps.pretty_method(chosen)})')
        fig.tight_layout(); fig.savefig(out/'cv2_pmf_unbiased.png',dpi=200); plt.close(fig)
        plot_file=str(out/'cv2_pmf_unbiased.png')
    except Exception as e:
        warnings.append(f'Secondary CV PMF plot failed: {e}')
    finite=np.isfinite(chosen_pmf['pmf'])
    span=float(np.nanmax(chosen_pmf['pmf'][finite])-np.nanmin(chosen_pmf['pmf'][finite])) if np.any(finite) else float('nan')
    cv2_conv=_agm.run_observable_pmf_convergence(
        d,args,d.cv2,bins,chosen,chosen_pmf,out,
        metric_name='secondary_cv',metric_label=cv2_label,x_label=cv2_label,
        out_dir_name='cv2_convergence',file_prefix='cv2',
        legacy_total_names=False,basin_tracking=True,progress=progress,
    )
    info={'available':True,'selected_unbiased_method':chosen,'n_samples':int(np.count_nonzero(mask)),'bins':int(len(bins)-1),'pmf_span_kcal_mol':span,'convergence':cv2_conv,'files':{'cv2_pmf_unbiased_csv':str(out/'cv2_pmf_unbiased.csv'),'cv2_pmf_umbrella_only_csv':str(out/'cv2_pmf_umbrella_only.csv'),'cv2_pmf_gamd_exponential_csv':str(out/'cv2_pmf_gamd_exponential.csv'),'cv2_pmf_gamd_cumulant2_csv':str(out/'cv2_pmf_gamd_cumulant2.csv'),'cv2_pmf_gamd_cumulant3_csv':str(out/'cv2_pmf_gamd_cumulant3.csv')}}
    if plot_file: info['files']['cv2_pmf_png']=plot_file
    if isinstance(cv2_conv,dict) and cv2_conv.get('files'):
        info['files'].update({k:v for k,v in cv2_conv['files'].items()})
    _agm.wjson(out/'cv2_pmf_summary.json', info)
    return info
