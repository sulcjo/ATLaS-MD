"""MBAR self-consistent solver family, relocated from analyze_gareus_mbar.py.

Includes the backend dispatcher (solve_mbar), four solver backends
(solve_mbar_numba, solve_mbar_numba_anderson, solve_mbar_sambar +
solve_mbar_sambar_warmstart, solve_mbar_lbfgs), their shared
logsumexp/Anderson-mixing/overlap/subset-reweight helpers, the two
@njit-decorated hot-loop kernels, and the MBAR configuration constants
(DEFAULT_MBAR_BACKEND, SAMBAR_*, MBAR_ANDERSON_HISTORY).

analyze_gareus_mbar.py's parse_args() applies --sambar-*/--mbar-backend/
--mbar-anderson-history CLI overrides by mutating THIS module's globals()
directly (gareus.mbar_analysis.solvers.SAMBAR_EPOCHS = ...), not its own --
every solve_mbar*/logsumexp* function here resolves these names as bare
globals against the module they are defined in (this one), so the override
must target this module's namespace to actually take effect.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:
    from numba import njit, prange, set_num_threads, get_num_threads
    NUMBA_AVAILABLE = True
except Exception:  # numba is optional; NumPy backend remains the safe fallback.
    NUMBA_AVAILABLE = False
    njit = None
    prange = range
    set_num_threads = None
    get_num_threads = None

try:
    from scipy.optimize import minimize as _scipy_minimize
    SCIPY_AVAILABLE = True
except Exception:
    _scipy_minimize = None
    SCIPY_AVAILABLE = False

# -----------------------------------------------------------------------------
# MBAR solver configuration defaults
#
# The following module-level variables control advanced MBAR solver behavior.
# They are exposed as command‑line options in analyze_gareus_mbar.py and can
# be overridden at runtime -- see that module's parse_args(), which mutates
# THIS module's globals() directly after argparse runs. See the argparse
# section in analyze_gareus_mbar.py for the corresponding --sambar-* and
# --mbar-anderson-history flags.  If these defaults are changed here,
# remember to update the help text and parser default values accordingly.

# Default MBAR backend.  "sambar" uses a stochastic warm‑start followed by
# deterministic polishing (usually L-BFGS).  Other choices include
# "numba", "numpy", "anderson", "numba-anderson", "lbfgs", etc.  The
# solve_mbar() function dispatches to the appropriate solver based on this
# string.  Changing this value here affects only the fallback when the
# command line does not specify --mbar-backend; the CLI parser overrides
# this default.
DEFAULT_MBAR_BACKEND = 'sambar'

# Stochastic SAMBAR warm‑start parameters.  The SAMBAR algorithm performs
# several epochs of mini‑batch MBAR fixed‑point updates to produce a good
# initial estimate for the free energy offsets f_k.  These parameters
# control the stochastic batching and learning rate.  See
# solve_mbar_sambar_warmstart() for details.
SAMBAR_EPOCHS = 30
SAMBAR_INITIAL_BATCH_SIZE = 1024
SAMBAR_BATCH_PATIENCE = 5
SAMBAR_SEED = 12345
SAMBAR_LR_SCALE = 1.0
SAMBAR_DELTA_F_MAX = 10.0

# Which deterministic backend to use to polish the SAMBAR warm‑start.  This
# should be one of the accepted backends for solve_mbar(): 'lbfgs',
# 'numba', 'numba-anderson', 'anderson', or 'numpy'.  It must not be
# 'sambar' to avoid infinite recursion.  See the CLI --sambar-polish-backend.
SAMBAR_POLISH_BACKEND = 'numba-anderson'

# Anderson/DIIS mixing history length for the Numba‑accelerated solver.
# A larger history can accelerate convergence but increases memory and
# susceptibility to ill‑conditioning.  See solve_mbar_numba_anderson().
MBAR_ANDERSON_HISTORY = 5


def logsumexp(a,axis=None):
    """Small dependency-free logsumexp.

    The analysis data are cleaned before use, so the hot MBAR path can avoid
    the slower nan-aware reductions.  This fallback still tolerates non-finite
    values outside the hot path.
    """
    a=np.asarray(a,dtype=np.float64)
    if axis is None:
        finite=np.isfinite(a)
        if not np.any(finite): return float('-inf')
        x=a[finite]; m=float(np.max(x)); return float(m+np.log(np.sum(np.exp(x-m))))
    finite=np.isfinite(a)
    safe=np.where(finite,a,-np.inf)
    m=np.max(safe,axis=axis,keepdims=True)
    all_bad=~np.isfinite(m)
    shifted=np.exp(safe-m)
    shifted=np.where(np.isfinite(shifted),shifted,0.0)
    out=m+np.log(np.sum(shifted,axis=axis,keepdims=True))
    out=np.where(all_bad,-np.inf,out)
    return np.squeeze(out,axis=axis)

def logsumexp_axis1_finite(a):
    a=np.asarray(a,dtype=np.float64)
    m=np.max(a,axis=1)
    return m+np.log(np.sum(np.exp(a-m[:,None]),axis=1))

def logsumexp_axis0_finite(a):
    a=np.asarray(a,dtype=np.float64)
    m=np.max(a,axis=0)
    return m+np.log(np.sum(np.exp(a-m[None,:]),axis=0))

def norm_logw(lw):
    lw=np.asarray(lw,float); out=np.zeros_like(lw); mask=np.isfinite(lw)
    if not np.any(mask): return out
    x=lw[mask]; x=x-logsumexp(x); out[mask]=np.exp(x); return out

if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_mbar_update(u, logn, f, ld, nf):
        N=u.shape[0]
        K=u.shape[1]
        for n in prange(N):
            m=-1.0e300
            for k in range(K):
                v=logn[k]+f[k]-u[n,k]
                if v>m:
                    m=v
            ss=0.0
            for k in range(K):
                ss+=math.exp(logn[k]+f[k]-u[n,k]-m)
            ld[n]=m+math.log(ss)
        for k in prange(K):
            m=-1.0e300
            for n in range(N):
                v=-u[n,k]-ld[n]
                if v>m:
                    m=v
            ss=0.0
            for n in range(N):
                ss+=math.exp(-u[n,k]-ld[n]-m)
            nf[k]=-(m+math.log(ss))

    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_mbar_logdenom(u, logn, f, ld):
        N=u.shape[0]
        K=u.shape[1]
        for n in prange(N):
            m=-1.0e300
            for k in range(K):
                v=logn[k]+f[k]-u[n,k]
                if v>m:
                    m=v
            ss=0.0
            for k in range(K):
                ss+=math.exp(logn[k]+f[k]-u[n,k]-m)
            ld[n]=m+math.log(ss)
else:
    _numba_mbar_update = None
    _numba_mbar_logdenom = None


def _anderson_step(F_hist: list, G_hist: list, m: int = 5) -> np.ndarray:
    """Anderson/DIIS mixing step.

    Given history lists F_hist (input iterates) and G_hist (fixed-point outputs),
    return the next Anderson-mixed iterate.  Falls back to the last G value when
    the system is ill-conditioned or history is length-1.

    The constrained LS problem min||Σ c_i r_i||² s.t. Σ c_i=1 is solved by
    translating the constraint: c_0 = 1 - Σ c_red, leading to
    dR @ c_red = -r_0 where dR[:,j] = r_{j+1} - r_0.
    """
    mk = min(len(F_hist), m)
    Rk = np.array(G_hist[-mk:]) - np.array(F_hist[-mk:])   # (mk, Ka)
    Gk = np.array(G_hist[-mk:])
    if mk == 1:
        return Gk[0].copy()
    r0 = Rk[0]
    dR = (Rk[1:] - r0).T                                     # (Ka, mk-1)
    try:
        c_red, _, _, _ = np.linalg.lstsq(dR, -r0, rcond=None)
    except Exception:
        return Gk[-1].copy()
    c0 = 1.0 - float(np.sum(c_red))
    coeffs = np.concatenate([[c0], c_red])
    if np.any(np.abs(coeffs) > 1e3) or not np.all(np.isfinite(coeffs)):
        return Gk[-1].copy()
    return coeffs @ Gk


def solve_mbar_numba(u_nk, window, tol=1e-10, maxiter=10000, progress: Optional['Progress'] = None, threads: int = 0, f_init: Optional[np.ndarray] = None):
    """Parallel MBAR fixed-point solve using optional Numba kernels.

    This accelerates the two hot reductions in each iteration:
      logsum_k N_k exp(f_k-u_nk) for every sample, and
      logsum_n exp(-u_nk-logdenom_n) for every state.
    It falls back before import-time if numba is unavailable.
    """
    if not NUMBA_AVAILABLE or _numba_mbar_update is None:
        raise RuntimeError('numba backend requested but numba is not available')
    if int(threads or 0) > 0 and set_num_threads is not None:
        set_num_threads(int(threads))
    used_threads = int(get_num_threads()) if get_num_threads is not None else None
    u_nk=np.asarray(u_nk,dtype=np.float64,order='C')
    window=np.asarray(window,dtype=np.int64)
    N,K=u_nk.shape
    nk=np.bincount(window[(window>=0)&(window<K)],minlength=K).astype(np.float64)
    active=np.where(nk>0)[0]
    if active.size==0:
        raise ValueError('no samples assigned to any state')
    # active.size==K (a size-K subset of np.where's size-K domain) implies
    # active==arange(K) exactly, i.e. every window has samples -- the common
    # case. u_nk is already float64/C-contiguous from the np.asarray call
    # above, so u_nk[:, active] would just be a full copy of u_nk itself;
    # skip it and use u_nk directly instead of paying for that copy.
    u = u_nk if active.size==K else np.ascontiguousarray(u_nk[:,active],dtype=np.float64)
    n=nk[active]
    logn=np.log(n)
    f=np.zeros(active.size,dtype=np.float64)
    if f_init is not None:
        fi=np.asarray(f_init,dtype=np.float64)
        if fi.size==K:
            for _i,_a in enumerate(active):
                if _a<fi.size and np.isfinite(fi[_a]):
                    f[_i]=fi[_a]
            f-=f[0]
    nf=np.zeros_like(f)
    ld=np.empty(N,dtype=np.float64)
    conv=False
    md=float('inf')
    # Compile before the timed/status loop so the first real iteration does not
    # look like a mysterious MBAR coma. Yes, JIT compilation has theatre.
    if progress is not None:
        progress.step('MBAR backend', f'numba parallel backend; threads={used_threads if used_threads is not None else "auto"}; compiling kernels')
    _numba_mbar_update(u,logn,f,ld,nf)
    for it in range(1,maxiter+1):
        if progress is not None and (it == 1 or it % 25 == 0):
            progress.bar('MBAR iterations', it, maxiter, f'numba delta {md:.2e}')
        _numba_mbar_update(u,logn,f,ld,nf)
        nf-=nf[0]
        md=float(np.max(np.abs(nf-f)))
        f,nf=nf,f
        if md<tol:
            conv=True
            break
    if progress is not None:
        progress.bar('MBAR iterations', 1, 1, f'backend=numba converged={conv} iter={it} delta={md:.2e}', force=True)
    _numba_mbar_logdenom(u,logn,f,ld)
    lw=-ld
    lw-=logsumexp(lw)
    fall=np.full(K,np.nan,dtype=np.float64)
    fall[active]=f
    return {'f_k':fall,'n_k':nk,'active':active,'logw':lw,'converged':conv,'iterations':it,'max_delta':md,'backend':'numba','threads':used_threads}


def solve_mbar_numba_anderson(u_nk, window, tol: float = 1e-10, maxiter: int = 10000,
                              progress: Optional['Progress'] = None, threads: int = 0,
                              f_init: Optional[np.ndarray] = None,
                              history: int = MBAR_ANDERSON_HISTORY) -> dict:
    """Parallel MBAR fixed‑point solve using Numba kernels with Anderson/DIIS mixing.

    This solver accelerates the plain Numba fixed‑point iterations by mixing the
    last few iterates using Anderson acceleration (also known as DIIS).  The
    parameter ``history`` controls how many previous iterates are retained for
    the least‑squares mixing; larger values can improve convergence but are
    more memory intensive and may become ill‑conditioned for noisy problems.

    Parameters
    ----------
    u_nk : array_like, shape (N, K)
        Reduced bias energies for every sample and every window (column order
        corresponds to thermodynamic states).  Only entries for ``active``
        windows are used; others are ignored.
    window : array_like, shape (N,)
        Index of the thermodynamic state for each sample.  States with no
        assigned samples are considered inactive and ignored.
    tol : float, optional
        Convergence threshold on the maximum change of f_k between iterations.
    maxiter : int, optional
        Maximum number of fixed‑point iterations to perform.
    progress : Progress, optional
        Progress bar object for interactive status updates.
    threads : int, optional
        Number of Numba threads to use; 0 leaves the default unchanged.
    f_init : array_like, optional
        Optional initial guess for f_k over all K states; values for inactive
        states are ignored.  When provided, the initial gauge is removed so
        that f_k[0] = 0.
    history : int, optional
        Number of past iterates to retain for Anderson mixing.  Defaults to
        ``MBAR_ANDERSON_HISTORY``.

    Returns
    -------
    result : dict
        Dictionary with keys: 'f_k', 'n_k', 'active', 'logw', 'converged',
        'iterations', 'max_delta', 'backend', and 'threads'.
    """
    if not NUMBA_AVAILABLE or _numba_mbar_update is None:
        raise RuntimeError('numba-anderson backend requested but numba is not available')
    # Configure Numba threads if requested
    if int(threads or 0) > 0 and set_num_threads is not None:
        set_num_threads(int(threads))
    used_threads = int(get_num_threads()) if get_num_threads is not None else None

    # Convert inputs to contiguous arrays
    u_nk = np.asarray(u_nk, dtype=np.float64, order='C')
    window = np.asarray(window, dtype=np.int64)
    N, K = u_nk.shape
    # Sample counts per state and active state indices
    nk = np.bincount(window[(window >= 0) & (window < K)], minlength=K).astype(np.float64)
    active = np.where(nk > 0)[0]
    if active.size == 0:
        raise ValueError('no samples assigned to any state')
    # Restrict bias energies and counts to active states. active.size==K
    # (every window has samples, the common case) implies active==arange(K)
    # exactly; u_nk is already float64/C-contiguous, so u_nk[:, active] would
    # just copy u_nk itself -- skip that copy and use u_nk directly.
    u = u_nk if active.size == K else np.ascontiguousarray(u_nk[:, active], dtype=np.float64)
    n = nk[active]
    logn = np.log(n)
    Ka = active.size
    # Initial f values on active states
    f = np.zeros(Ka, dtype=np.float64)
    if f_init is not None:
        fi = np.asarray(f_init, dtype=np.float64)
        if fi.size == K:
            # transfer initial guess for active windows
            for _i, _a in enumerate(active):
                if _a < fi.size and np.isfinite(fi[_a]):
                    f[_i] = fi[_a]
            f -= f[0]
    # Work arrays for Numba update
    nf = np.zeros_like(f)
    ld = np.empty(N, dtype=np.float64)
    # Prepare history lists for Anderson mixing
    F_hist: list = []
    G_hist: list = []
    conv = False
    md = float('inf')
    # Compile kernels before timing loop
    if progress is not None:
        progress.step('MBAR backend', f'numba-anderson backend; threads={used_threads if used_threads is not None else "auto"}; compiling kernels')
    _numba_mbar_update(u, logn, f, ld, nf)
    # Main fixed‑point iteration with Anderson mixing
    for it in range(1, maxiter + 1):
        # Update status every 25 iterations or on the first iteration
        if progress is not None and (it == 1 or it % 25 == 0):
            progress.bar('MBAR iterations', it, maxiter, f'numba-anderson delta {md:.2e}')
        # Compute next iterate via Numba update
        _numba_mbar_update(u, logn, f, ld, nf)
        # Re‑gauge nf so nf[0] = 0
        nf -= nf[0]
        # Change magnitude before mixing
        md = float(np.max(np.abs(nf - f)))
        # Append to history
        F_hist.append(f.copy())
        G_hist.append(nf.copy())
        if history is not None and history > 0:
            # Trim history to specified length
            while len(F_hist) > history:
                F_hist.pop(0); G_hist.pop(0)
        # Perform Anderson mixing when history has more than one element
        if len(F_hist) > 1:
            try:
                f_new = _anderson_step(F_hist, G_hist, m=history if history is not None else 5)
                # Remove gauge
                f_new -= f_new[0]
                f = f_new
            except Exception:
                # Fallback to nf if mixing fails
                f = nf.copy()
        else:
            # For the first iteration, no mixing
            f = nf.copy()
        # Convergence check on the un‑mixed delta
        if md < tol:
            conv = True
            break
    # Compute final log weights and fill full f_k array
    if progress is not None:
        progress.bar('MBAR iterations', 1, 1, f'backend=numba-anderson converged={conv} iter={it} delta={md:.2e}', force=True)
    # Compute log denominators and weights using Numba kernel
    _numba_mbar_logdenom(u, logn, f, ld)
    lw = -ld
    lw -= logsumexp(lw)
    fall = np.full(K, np.nan, dtype=np.float64)
    fall[active] = f
    return {'f_k': fall, 'n_k': nk, 'active': active, 'logw': lw, 'converged': conv,
            'iterations': it, 'max_delta': md, 'backend': 'numba-anderson', 'threads': used_threads}


def solve_mbar_sambar_warmstart(u_nk, window,
                                epochs: int = None,
                                initial_batch_size: int = None,
                                batch_patience: int = None,
                                seed: Optional[int] = None,
                                lr_scale: float = None,
                                delta_f_max: float = None,
                                progress: Optional['Progress'] = None,
                                f_init: Optional[np.ndarray] = None) -> np.ndarray:
    """Stochastic SAMBAR mini‑batch warm‑start for MBAR.

    This function performs a number of epochs of mini‑batch fixed‑point updates
    to provide a good initial estimate for the free energy offsets f_k.  The
    mini‑batch size starts at ``initial_batch_size`` and is doubled every
    ``batch_patience`` epochs until it reaches the full data size.  Within
    each epoch a subset of samples is randomly drawn (with replacement) and
    used to approximate the MBAR fixed‑point update.  A simple learning
    rate proportional to ``sqrt(batch_size / N)`` is applied to the update.
    Large free energy changes are clipped to ``delta_f_max`` to prevent
    divergence.  The result is an initial f_k that can be passed to a
    deterministic solver for polishing.

    Parameters
    ----------
    u_nk : array_like, shape (N, K)
        Reduced bias energies for every sample and window.
    window : array_like, shape (N,)
        Index of the thermodynamic state for each sample.
    epochs : int, optional
        Number of mini‑batch epochs.  Defaults to ``SAMBAR_EPOCHS``.
    initial_batch_size : int, optional
        Starting mini‑batch size.  Defaults to ``SAMBAR_INITIAL_BATCH_SIZE``.
    batch_patience : int, optional
        Number of epochs at a fixed batch size before doubling it.  Defaults
        to ``SAMBAR_BATCH_PATIENCE``.
    seed : int, optional
        Random seed for reproducible batching.  Defaults to ``SAMBAR_SEED``.
    lr_scale : float, optional
        Global learning rate scale.  Defaults to ``SAMBAR_LR_SCALE``.
    delta_f_max : float, optional
        Maximum absolute change applied to f_k per epoch.  Defaults to
        ``SAMBAR_DELTA_F_MAX``.
    progress : Progress, optional
        Progress object for status updates.
    f_init : array_like, optional
        Optional initial guess for f_k over all K states; values for inactive
        states are ignored.

    Returns
    -------
    f_full : ndarray, shape (K,)
        Full array of length K with warm‑started f_k values.  Inactive
        windows are filled with NaNs.
    """
    # Use module‑level defaults if parameters are None
    if epochs is None:
        epochs = SAMBAR_EPOCHS
    if initial_batch_size is None:
        initial_batch_size = SAMBAR_INITIAL_BATCH_SIZE
    if batch_patience is None:
        batch_patience = SAMBAR_BATCH_PATIENCE
    if seed is None:
        seed = SAMBAR_SEED
    if lr_scale is None:
        lr_scale = SAMBAR_LR_SCALE
    if delta_f_max is None:
        delta_f_max = SAMBAR_DELTA_F_MAX
    # Convert inputs
    u_nk = np.asarray(u_nk, dtype=np.float64)
    window = np.asarray(window, dtype=np.int64)
    N, K = u_nk.shape
    # Sample counts per state and active windows
    nk = np.bincount(window[(window >= 0) & (window < K)], minlength=K).astype(np.float64)
    active = np.where(nk > 0)[0]
    if active.size == 0:
        raise ValueError('no samples assigned to any state')
    # active.size==K (every window has samples, the common case) implies
    # active==arange(K) exactly, so u_nk[:, active] would just copy u_nk
    # itself; skip that copy and use u_nk directly.
    u = u_nk if active.size == K else np.ascontiguousarray(u_nk[:, active], dtype=np.float64)
    n = nk[active]
    logn = np.log(n)
    Ka = active.size
    # Initial f on active states
    f = np.zeros(Ka, dtype=np.float64)
    if f_init is not None:
        fi = np.asarray(f_init, dtype=np.float64)
        if fi.size == K:
            for _i, _a in enumerate(active):
                if _a < fi.size and np.isfinite(fi[_a]):
                    f[_i] = fi[_a]
            f -= f[0]
    # Random number generator
    rng = np.random.default_rng(seed)
    batch_size = int(initial_batch_size)
    patience_counter = 0
    # Work arrays for update on subsets
    # Note: We allocate new arrays each epoch because subset shapes vary.
    for epoch in range(int(epochs)):
        if progress is not None and (epoch == 0 or epoch % 10 == 0):
            progress.step('MBAR backend', f'SAMBAR warm‑start epoch {epoch + 1}/{epochs}, batch_size={batch_size}')
        # Determine indices for current batch; sample with replacement when necessary
        if batch_size >= N:
            # Use all samples
            idx = np.arange(N)
        else:
            idx = rng.integers(0, N, batch_size)
        # Compute log denominators for the batch
        tmp = logn[None, :] + f[None, :] - u[idx, :]
        # logsumexp along axis=1
        ld = logsumexp_axis1_finite(tmp)
        # Compute new f_k estimate from the batch
        tmp2 = -u[idx, :] - ld[:, None]
        nf = -logsumexp_axis0_finite(tmp2)
        nf -= nf[0]
        # Update step
        delta = nf - f
        # Compute learning rate based on relative batch size
        lr = float(lr_scale) * math.sqrt(float(len(idx)) / float(N)) if N > 0 else float(lr_scale)
        # Clip free energy changes to prevent divergence
        if delta_f_max is not None and float(delta_f_max) > 0:
            delta = np.clip(delta, -float(delta_f_max), float(delta_f_max))
        f = f + lr * delta
        f -= f[0]
        # Update batch size after patience epochs
        patience_counter += 1
        if patience_counter >= int(batch_patience) and batch_size < N:
            new_size = min(int(batch_size * 2), N)
            if new_size > batch_size:
                batch_size = new_size
                patience_counter = 0
    # Fill full K array with NaNs and assign active f values
    f_full = np.full(K, np.nan, dtype=np.float64)
    f_full[active] = f
    return f_full


def solve_mbar_sambar(u_nk, window, tol: float = 1e-10, maxiter: int = 10000,
                      progress: Optional['Progress'] = None, threads: int = 0,
                      f_init: Optional[np.ndarray] = None,
                      polish_backend: Optional[str] = None,
                      epochs: Optional[int] = None,
                      initial_batch_size: Optional[int] = None,
                      batch_patience: Optional[int] = None,
                      seed: Optional[int] = None,
                      lr_scale: Optional[float] = None,
                      delta_f_max: Optional[float] = None) -> dict:
    """Full SAMBAR MBAR solver: stochastic warm‑start followed by deterministic polish.

    This solver first calls ``solve_mbar_sambar_warmstart`` to obtain an
    approximate free energy vector ``f_k`` using stochastic mini‑batch
    updates.  It then refines this initial guess using a deterministic MBAR
    solver (e.g., L-BFGS or Numba) specified by ``polish_backend``.  The
    final log weights and statistics are those of the deterministic solve.

    Parameters
    ----------
    u_nk, window, tol, maxiter, progress, threads : see ``solve_mbar``
    f_init : array_like, optional
        Optional additional initial guess passed to the warm‑start.  Values
        for inactive states are ignored.  If provided, they override the
        default zero initialisation.
    polish_backend : str, optional
        Backend string for the deterministic polish.  When None, uses
        ``SAMBAR_POLISH_BACKEND``.

    Returns
    -------
    result : dict
        MBAR solution dictionary as returned by ``solve_mbar`` for the
        polishing backend.  The 'backend' field reflects the polishing
        backend rather than 'sambar'.
    """
    if polish_backend is None or not polish_backend:
        polish_backend = SAMBAR_POLISH_BACKEND
    # Allow callers to use cheaper SAMBAR settings for repeated convergence
    # solves without changing the full-production MBAR defaults.
    epochs = SAMBAR_EPOCHS if epochs is None else int(epochs)
    initial_batch_size = SAMBAR_INITIAL_BATCH_SIZE if initial_batch_size is None else int(initial_batch_size)
    batch_patience = SAMBAR_BATCH_PATIENCE if batch_patience is None else int(batch_patience)
    seed = SAMBAR_SEED if seed is None else int(seed)
    lr_scale = SAMBAR_LR_SCALE if lr_scale is None else float(lr_scale)
    delta_f_max = SAMBAR_DELTA_F_MAX if delta_f_max is None else float(delta_f_max)
    # Warm-start: compute f_init across all K states (including NaNs for inactive)
    if progress is not None:
        progress.step('MBAR backend', f'sambar warm-start epochs={epochs}')
    warm_f = solve_mbar_sambar_warmstart(u_nk, window,
                                         epochs=epochs,
                                         initial_batch_size=initial_batch_size,
                                         batch_patience=batch_patience,
                                         seed=seed,
                                         lr_scale=lr_scale,
                                         delta_f_max=delta_f_max,
                                         progress=progress,
                                         f_init=f_init)
    # Deterministic polish using specified backend
    if progress is not None:
        progress.step('MBAR backend', f'sambar polish ({polish_backend})')
    # Avoid recursion if polish_backend == 'sambar'
    if str(polish_backend).lower() == 'sambar':
        raise RuntimeError('sambar backend cannot polish another sambar solve')
    res = solve_mbar(u_nk, window, tol=tol, maxiter=maxiter, progress=progress,
                     backend=polish_backend, threads=threads, f_init=warm_f)
    res['sambar_warmstart_epochs'] = int(epochs)
    res['sambar_initial_batch_size'] = int(initial_batch_size)
    res['sambar_batch_patience'] = int(batch_patience)
    res['sambar_polish_backend'] = str(polish_backend)
    return res

def solve_mbar_lbfgs(u_nk, window, tol=1e-10, maxiter=10000, progress: Optional['Progress'] = None, f_init: Optional[np.ndarray] = None):
    """MBAR via L-BFGS-B on the negated log-likelihood.

    The MBAR log-likelihood L(f) = Σ_k N_k f_k - Σ_n log Σ_k N_k exp(f_k-u_nk)
    is concave, so we minimize -L.  f[0] is fixed to 0 (gauge); scipy optimizes
    f_red = f[1:] (Ka-1 free parameters).

    The gradient costs exactly one fixed-point iteration:
      ∂L/∂f_k = N_k - Σ_n exp(logn_k + f_k - u_nk - ld_n)
    which drops out for free from the log-denominator already needed for -L.

    tol maps to gtol (gradient ∞-norm) in scipy.  This is NOT identical to the
    fixed-point residual tol used by the other backends — L-BFGS typically
    converges in O(10-100) gradient evaluations vs O(100-10000) SCI iterations.
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError('lbfgs backend requires scipy')
    u_nk = np.asarray(u_nk, dtype=np.float64, order='C')
    window = np.asarray(window, dtype=np.int64)
    N, K = u_nk.shape
    nk = np.bincount(window[(window >= 0) & (window < K)], minlength=K).astype(np.float64)
    active = np.where(nk > 0)[0]
    if active.size == 0:
        raise ValueError('no samples assigned to any state')
    Ka = active.size
    # active.size==K (every window has samples, the common case) implies
    # active==arange(K) exactly, so u_nk[:, active] would just copy u_nk
    # itself; skip that copy and use u_nk directly.
    u = u_nk if active.size == K else np.ascontiguousarray(u_nk[:, active], dtype=np.float64)
    n = nk[active]
    logn = np.log(n)

    f0 = np.zeros(Ka, dtype=np.float64)
    if f_init is not None:
        fi = np.asarray(f_init, dtype=np.float64)
        if fi.size == K:
            for _i, _a in enumerate(active):
                if _a < fi.size and np.isfinite(fi[_a]):
                    f0[_i] = fi[_a]
            f0 -= f0[0]

    tmp = np.empty((N, Ka), dtype=np.float64)

    def neg_loglik_and_grad(f_red):
        f = np.empty(Ka, dtype=np.float64)
        f[0] = 0.0
        f[1:] = f_red
        # log denominator: log Σ_k N_k exp(f_k - u_nk) for each sample
        np.add(logn[None, :] + f[None, :], -u, out=tmp)
        ld = logsumexp_axis1_finite(tmp)
        # -L = -(dot(n,f) - sum(ld))
        neg_L = -(float(np.dot(n, f)) - float(np.sum(ld)))
        # gradient of L w.r.t. f: N_k - Σ_n exp(logn_k + f_k - u_nk - ld_n)
        tmp2 = tmp - ld[:, None]
        grad_L = n - np.sum(np.exp(tmp2), axis=0)
        # negate and drop f[0] component (fixed gauge)
        neg_grad_red = -grad_L[1:]
        return neg_L, neg_grad_red

    if progress is not None:
        progress.step('MBAR backend', 'L-BFGS-B (scipy)')

    result = _scipy_minimize(
        neg_loglik_and_grad,
        f0[1:],
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': maxiter, 'gtol': tol, 'ftol': 0.0},
    )

    f = np.empty(Ka, dtype=np.float64)
    f[0] = 0.0
    f[1:] = result.x
    # scipy L-BFGS-B status codes: 0 = converged (gtol/ftol satisfied), 1 =
    # iteration/function-eval limit reached (NOT converged), 2 = other
    # abnormal termination.  Only status 0 (or a scipy-reported success, kept
    # for forward compatibility) counts as converged here -- status 1 must
    # never be treated as convergence, or a run that merely hit maxiter gets
    # silently reported as fully converged.
    conv = result.success or result.status == 0
    it = int(result.nit)
    grad_norm = float(np.max(np.abs(result.jac))) if result.jac is not None else float('nan')

    # Final log-denominator and weights
    np.add(logn[None, :] + f[None, :], -u, out=tmp)
    ld = logsumexp_axis1_finite(tmp)
    lw = -ld
    lw -= logsumexp(lw)

    fall = np.full(K, np.nan, dtype=np.float64)
    fall[active] = f

    if progress is not None:
        progress.bar('MBAR iterations', 1, 1, f'backend=lbfgs converged={conv} iter={it} grad_norm={grad_norm:.2e}', force=True)

    return {'f_k': fall, 'n_k': nk, 'active': active, 'logw': lw,
            'converged': conv, 'iterations': it, 'max_delta': grad_norm, 'backend': 'lbfgs', 'threads': None}


def solve_mbar(u_nk, window, tol=1e-10, maxiter=10000, progress: Optional['Progress'] = None, backend: str = 'auto', threads: int = 0, f_init: Optional[np.ndarray] = None,
               sambar_epochs: Optional[int] = None, sambar_initial_batch_size: Optional[int] = None,
               sambar_batch_patience: Optional[int] = None, sambar_seed: Optional[int] = None,
               sambar_lr_scale: Optional[float] = None, sambar_delta_f_max: Optional[float] = None,
               sambar_polish_backend: Optional[str] = None):
    """Solve MBAR self-consistency with selectable backends.

    Supported backends include:

      lbfgs          – L-BFGS-B on the negated MBAR log-likelihood (requires SciPy).  Usually converges in O(10–100) gradient evaluations; ``tol`` maps to the gradient ∞-norm.
      numba          – Parallel fixed‑point iterations via Numba JIT kernels.
      numba-anderson – Same as ``numba`` but with Anderson/DIIS history mixing.  Typically reduces the number of iterations by an order of magnitude with minimal overhead.
      numba-diis     – Alias for ``numba-anderson``.
      anderson       – Pure NumPy fixed‑point with Anderson/DIIS mixing (~5–20× fewer iterations than plain NumPy).
      numpy          – Plain NumPy fixed‑point iteration (baseline, no extra deps).
      sambar         – Stochastic SAMBAR warm‑start followed by deterministic polish (see ``--sambar-*`` options).
      auto           – Chooses a backend based on problem size and dependencies: L-BFGS for moderate problems when SciPy is available, Numba for very large problems when Numba is available, and Anderson otherwise.

    """
    # Normalize backend string; use module default when None or empty
    if backend is None or backend == '':
        backend = DEFAULT_MBAR_BACKEND
    backend = str(backend).lower()
    try:
        problem_size = int(np.asarray(u_nk).shape[0]) * int(np.asarray(u_nk).shape[1])
    except Exception:
        problem_size = 0

    # lbfgs: O(10-100) gradient evaluations; preferred for moderate problem sizes
    # Handle special backends first
    # SAMBAR: stochastic warm‑start then deterministic polish
    if backend == 'sambar':
        return solve_mbar_sambar(u_nk, window, tol=tol, maxiter=maxiter, progress=progress, threads=threads, f_init=f_init,
                                 polish_backend=sambar_polish_backend, epochs=sambar_epochs,
                                 initial_batch_size=sambar_initial_batch_size, batch_patience=sambar_batch_patience,
                                 seed=sambar_seed, lr_scale=sambar_lr_scale, delta_f_max=sambar_delta_f_max)
    # Numba with Anderson/DIIS mixing
    if backend in ('numba-anderson', 'numba-diis'):
        res = solve_mbar_numba_anderson(u_nk, window, tol=tol, maxiter=maxiter, progress=progress, threads=threads, f_init=f_init, history=MBAR_ANDERSON_HISTORY)
        # Tag backend exactly as requested (useful when alias 'numba-diis' is used)
        res['backend'] = backend
        return res

    # Continue with standard backend selection logic
    use_lbfgs = (backend == 'lbfgs') or (backend == 'auto' and SCIPY_AVAILABLE and problem_size < 1_000_000)
    if use_lbfgs:
        try:
            return solve_mbar_lbfgs(u_nk, window, tol=tol, maxiter=maxiter, progress=progress, f_init=f_init)
        except Exception as exc:
            if backend == 'lbfgs':
                raise
            if progress is not None:
                progress.step('MBAR backend', f'lbfgs failed ({exc}); trying next backend')

    # numba: parallel JIT kernels; pays off for very large problems
    use_numba = (backend == 'numba') or (backend == 'auto' and NUMBA_AVAILABLE and problem_size >= 200_000)
    if use_numba:
        try:
            return solve_mbar_numba(u_nk, window, tol=tol, maxiter=maxiter, progress=progress, threads=threads, f_init=f_init)
        except Exception as exc:
            if backend == 'numba':
                raise
            if progress is not None:
                progress.step('MBAR backend', f'numba unavailable/failed ({exc}); falling back to anderson/numpy')

    # anderson or plain numpy fixed-point
    use_anderson = backend in ('anderson', 'auto')
    if progress is not None:
        progress.step('MBAR backend', f'{"anderson" if use_anderson else "numpy"} vectorized backend')
    u_nk=np.asarray(u_nk,dtype=np.float64,order='C')
    window=np.asarray(window,dtype=np.int64)
    N,K=u_nk.shape
    nk=np.bincount(window[(window>=0)&(window<K)],minlength=K).astype(np.float64)
    active=np.where(nk>0)[0]
    if active.size==0: raise ValueError('no samples assigned to any state')
    # active.size==K (every window has samples, the common case) implies
    # active==arange(K) exactly, so u_nk[:, active] would just copy u_nk
    # itself; skip that copy and use u_nk directly.
    u = u_nk if active.size==K else np.ascontiguousarray(u_nk[:,active],dtype=np.float64)
    n=nk[active]
    logn=np.log(n)
    f=np.zeros(active.size,dtype=np.float64)
    if f_init is not None:
        fi=np.asarray(f_init,dtype=np.float64)
        if fi.size==K:
            for _i,_a in enumerate(active):
                if _a<fi.size and np.isfinite(fi[_a]):
                    f[_i]=fi[_a]
            f-=f[0]
    conv=False
    md=float('inf')
    tmp=np.empty_like(u)
    F_hist: list = []
    G_hist: list = []
    for it in range(1,maxiter+1):
        if progress is not None and (it == 1 or it % 25 == 0):
            progress.bar('MBAR iterations', it, maxiter, f'delta {md:.2e}')
        # log denominator for each sample: log sum_k N_k exp(f_k-u_nk)
        np.subtract(logn[None,:]+f[None,:],u,out=tmp)
        ld=logsumexp_axis1_finite(tmp)
        # new f_k = -log sum_n exp(-u_nk - ld_n), shifted to f_0=0
        np.negative(u,out=tmp)
        tmp-=ld[:,None]
        nf=-logsumexp_axis0_finite(tmp)
        nf-=nf[0]
        md=float(np.max(np.abs(nf-f)))
        if use_anderson:
            F_hist.append(f.copy())
            G_hist.append(nf.copy())
            if len(F_hist) > 5:
                F_hist.pop(0); G_hist.pop(0)
            f = _anderson_step(F_hist, G_hist)
            f -= f[0]
        else:
            f=nf
        if md<tol:
            conv=True
            break
    bname = 'anderson' if use_anderson else 'numpy'
    if progress is not None:
        progress.bar('MBAR iterations', 1, 1, f'backend={bname} converged={conv} iter={it} delta={md:.2e}', force=True)
    np.subtract(logn[None,:]+f[None,:],u,out=tmp)
    ld=logsumexp_axis1_finite(tmp)
    lw=-ld
    lw-=logsumexp(lw)
    fall=np.full(K,np.nan,dtype=np.float64)
    fall[active]=f
    return {'f_k':fall,'n_k':nk,'active':active,'logw':lw,'converged':conv,'iterations':it,'max_delta':md,'backend':bname,'threads':None}


def overlap_matrix(cv,window,bins,K):
    """Window-window histogram overlap with vectorized histogram assembly."""
    cv=np.asarray(cv,dtype=np.float64)
    window=np.asarray(window,dtype=np.int64)
    B=len(bins)-1
    H=np.zeros((K,B),dtype=np.float64)
    if cv.size and K>0 and B>0:
        bi=np.searchsorted(bins,cv,side='right')-1
        bi[cv==bins[-1]]=B-1
        mask=(window>=0)&(window<K)&(bi>=0)&(bi<B)
        if np.any(mask):
            linear=window[mask]*B+bi[mask]
            H=np.bincount(linear,minlength=K*B).reshape(K,B).astype(np.float64)
            row_sums=H.sum(axis=1)
            nz=row_sums>0
            H[nz]/=row_sums[nz,None]
    return np.minimum(H[:,None,:],H[None,:,:]).sum(axis=2)


def _subset_logw_from_global_fk(d_subset: 'Data', f_k_global: np.ndarray) -> np.ndarray:
    """Correct per-sample MBAR log-weights for a SUBSET of the full sample
    population (e.g. an epoch_000/rest split, or a secondary-CV regime
    split), reusing the already-solved GLOBAL free energies ``f_k_global``
    but the subset's own per-state sample counts ``N_k^subset`` in the MBAR
    self-consistency denominator:

        logw_S[n] = -logsumexp_k( log(N_k^subset[k]) + f_k[k] - u_nk[n, k] )

    Naively slicing the GLOBAL logw to a subset and renormalizing (``
    norm_logw(logw[mask])``) only corrects for the subset's overall size --
    it silently keeps using N_k^GLOBAL inside every sample's denominator,
    which is wrong whenever different states lose different *fractions* of
    their samples to the exclusion (e.g. one window losing 80% of its
    samples to an epoch_000 exclusion while another loses 10%). That
    produces a real, direction-consistent tilt across the CV axis.

    This recomputes just the denominator with the subset's own N_k while
    still reusing f_k_global -- state free energies are a property of the
    whole population and don't need re-solving for a subset reweight. It is
    an approximation (not a from-scratch MBAR resolve of the subset alone),
    but a substantially better one than the naive mask-and-renormalize.
    States with zero subset representation (N_k^subset == 0) are dropped
    from the denominator sum entirely (equivalent to log(0) = -inf), and
    states with a non-finite global f_k (never solved -- zero global
    samples) are dropped the same way for safety, though a subset can never
    contain samples from a state absent at the global level.

    Degenerate case: when d_subset covers the ENTIRE global population, this
    reduces to f_k_global's own logw exactly (same N_k, same f_k, same u_nk
    used to derive it in the first place).
    """
    K = int(f_k_global.size)
    u_nk = np.asarray(d_subset.u_nk, dtype=np.float64)
    window = np.asarray(d_subset.window, dtype=np.int64)
    f_k_global = np.asarray(f_k_global, dtype=np.float64)
    n_k_subset = np.bincount(window[(window >= 0) & (window < K)], minlength=K).astype(np.float64)
    active = np.where((n_k_subset > 0) & np.isfinite(f_k_global[:K]))[0]
    if active.size == 0:
        return np.full(u_nk.shape[0], -np.inf, dtype=np.float64)
    log_n = np.log(n_k_subset[active])
    f_active = f_k_global[active]
    if active.size == K and u_nk.shape[1] == K:
        # active is every column 0..K-1 (a size-K subset of the size-K
        # np.where domain must BE the whole domain) AND u_nk has exactly K
        # columns, so u_nk[:, active] would just be a full copy of u_nk
        # itself. Skip the copy. (Guarding on u_nk.shape[1] too, not just
        # active.size==K, matters here specifically because K comes from
        # f_k_global.size rather than from u_nk.shape as in the solve_mbar*
        # backends below -- the two are not structurally guaranteed equal at
        # this call site the way they are there.)
        tmp = log_n[None, :] + f_active[None, :] - u_nk
    else:
        tmp = log_n[None, :] + f_active[None, :] - u_nk[:, active]
    ld = logsumexp_axis1_finite(tmp)
    logw_s = -ld
    logw_s -= logsumexp(logw_s)
    return logw_s


__all__ = [
    "NUMBA_AVAILABLE", "SCIPY_AVAILABLE",
    "DEFAULT_MBAR_BACKEND", "SAMBAR_EPOCHS", "SAMBAR_INITIAL_BATCH_SIZE",
    "SAMBAR_BATCH_PATIENCE", "SAMBAR_SEED", "SAMBAR_LR_SCALE",
    "SAMBAR_DELTA_F_MAX", "SAMBAR_POLISH_BACKEND", "MBAR_ANDERSON_HISTORY",
    "logsumexp", "logsumexp_axis1_finite", "logsumexp_axis0_finite",
    "norm_logw", "solve_mbar_numba", "solve_mbar_numba_anderson",
    "solve_mbar_sambar_warmstart", "solve_mbar_sambar", "solve_mbar_lbfgs",
    "solve_mbar", "overlap_matrix", "_subset_logw_from_global_fk",
]
