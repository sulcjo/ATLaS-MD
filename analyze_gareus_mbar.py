#!/usr/bin/env python3
from __future__ import annotations
import os as _os_env
_ne_cap = int(_os_env.environ.get('NUMEXPR_MAX_THREADS', 64))
_ne_num = int(_os_env.environ.get('NUMEXPR_NUM_THREADS', _os_env.cpu_count() or _ne_cap))
_thread_cap = str(min(_ne_num, _ne_cap))
_os_env.environ['NUMEXPR_NUM_THREADS'] = _thread_cap  # force-cap even if already set
_os_env.environ.setdefault('NUMBA_NUM_THREADS', _thread_cap)
del _os_env, _ne_cap, _ne_num, _thread_cap
import argparse, csv, hashlib, json, math, os, re, shutil, sys, time
from pathlib import Path
from typing import Optional
import numpy as np
from gareus.units import KJ_PER_KCAL, K_B_KJ_PER_MOL_K
from gareus.mbar_analysis.bias import (
    _compute_u_nk_analytical,
    _parse_epoch_window_map_native_params,
    _epoch_bias_param_vectors,
    _reconstruct_union_bias_block,
)
from gareus.mbar_analysis.solvers import (
    NUMBA_AVAILABLE, SCIPY_AVAILABLE,
    DEFAULT_MBAR_BACKEND, SAMBAR_EPOCHS, SAMBAR_INITIAL_BATCH_SIZE,
    SAMBAR_BATCH_PATIENCE, SAMBAR_SEED, SAMBAR_LR_SCALE, SAMBAR_DELTA_F_MAX,
    SAMBAR_POLISH_BACKEND, MBAR_ANDERSON_HISTORY,
    logsumexp, logsumexp_axis1_finite, logsumexp_axis0_finite,
    norm_logw, solve_mbar_numba, solve_mbar_numba_anderson,
    solve_mbar_sambar_warmstart, solve_mbar_sambar, solve_mbar_lbfgs,
    solve_mbar, overlap_matrix, _subset_logw_from_global_fk,
)
import gareus.mbar_analysis.solvers as _mbar_solvers
from gareus.mbar_analysis.data import (
    Data, rjson, wjson, read_windows, jvec, infer_temp_beta,
    clean, _masked_data, _apply_analysis_stride, _filter_epoch_source,
    _sample_block_ids, _skip_first_n_frames, _epoch_dir_index,
    _epoch_number_for_run_dir, _epoch_run_manifest_secondary_cv_type,
    _epoch_zero_split_masks, _secondary_cv_epoch_regime_masks,
)
from gareus.mbar_analysis.loaders_adaptive import (
    _find_adaptive_epoch_dirs, _find_selfcontained_epoch_dirs,
    _find_adaptive_epoch_csv_sources, _has_epoch_csv_layout,
    _find_adaptive_final_run_dirs, _find_gareus_round_dirs,
    _vectorized_map_lookup, _vectorized_map_lookup_or_self, _vectorized_map_index,
    load_epoch_csv_adaptive, load_union_npz, _load_round_raw,
    _build_union_window_table, _round_window_to_union_map,
    _augment_with_adaptive_rounds,
)
from gareus.mbar_analysis.loaders_union_parquet import (
    _is_usable_for_mbar, _merge_missing_usable_states, load_parquet_adaptive_union,
)
from gareus.mbar_analysis.loaders import (
    _npz_sample_count_open, _npz_window_count_open, _Arrays, _ANALYSIS_VECTOR_KEYS,
    _discover_analysis_chunk_paths, _append_npz_arrays, _load_merged_arrays, load_npz,
    _load_secondary_cv_from_csv, _csv_row_count_fast, _npz_sample_count,
    _analysis_binary_sample_count, _window_float_array, load_csv,
    _parquet_sample_count, load_parquet, prod_dir_of, load_data,
)

try:
    from scipy.signal import find_peaks as _scipy_find_peaks
    SCIPY_SIGNAL_AVAILABLE = True
except Exception:
    _scipy_find_peaks = None
    SCIPY_SIGNAL_AVAILABLE = False

class Progress:
    """Small stderr progress bar for long analysis steps.

    It uses carriage-return redraws on interactive terminals and concise status
    lines otherwise, so batch logs stay readable.
    """
    def __init__(self):
        self.enabled = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self.last_draw = 0.0
        self.finished_line = True

    def bar(self, label: str, current: int, total: int, msg: str = "", force: bool = False):
        total = max(1, int(total))
        current = max(0, min(int(current), total))
        now = time.time()
        if not force and (now - self.last_draw) < 0.12:
            return
        self.last_draw = now
        frac = current / total
        if not self.enabled:
            if force:
                print(f"[{label}] {current}/{total} {100.0*frac:5.1f}% {msg}", file=sys.stderr, flush=True)
            return
        width = 32
        filled = int(round(frac * width))
        bar = "#" * filled + "-" * (width - filled)
        text = f"[{label:<18}] [{bar}] {current:>6}/{total:<6} {100.0*frac:5.1f}%"
        if msg:
            text += f" | {msg}"
        cols = 120
        try:
            import shutil
            cols = shutil.get_terminal_size((120, 24)).columns
        except Exception:
            pass
        if len(text) > cols - 2:
            text = text[: max(20, cols - 5)] + "..."
        sys.stderr.write("\r" + text + " " * max(0, cols - len(text) - 1))
        sys.stderr.flush()
        self.finished_line = False

    def step(self, label: str, msg: str = ""):
        if not self.enabled:
            print(f"[{label}] {msg}", file=sys.stderr, flush=True)
            return
        if not self.finished_line:
            sys.stderr.write("\n")
            self.finished_line = True
        print(f"[{label}] {msg}", file=sys.stderr, flush=True)

    def done(self, label: str = "done", msg: str = ""):
        self.bar(label, 1, 1, msg, force=True)
        if self.enabled and not self.finished_line:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self.finished_line = True


from gareus.mbar_analysis.plotting import (
    _secondary_cv_label, _secondary_cv_regions, _regime_slug,
    _primary_cv_label, _primary_cv_units, _primary_cv_axis_label,
)


def run_secondary_cv_analyses(d: 'Data', args, base_logw: np.ndarray, selected: str,
                               boost_ok: bool, kbt_kcal: float, out: Path,
                               warnings: list, progress: Optional['Progress'],
                               f_k_global: Optional[np.ndarray] = None) -> tuple:
    """Secondary-CV PMF + CV1xCV2 2D FES, split by secondary-CV regime when the
    run's CV2 definition changed mid-campaign (see
    ``_secondary_cv_epoch_regime_masks``). Single-regime runs (the common
    case) are entirely unaffected: this degrades to the plain unmodified
    calls, writing to the same paths as before.

    ``f_k_global`` should be the GLOBAL MBAR solve's ``f_k`` (``m['f_k']``)
    when available. When multiple regimes exist, each regime's own logw is
    recomputed from ``f_k_global`` and that regime's own per-state sample
    counts (``_subset_logw_from_global_fk``) rather than naively slicing
    ``base_logw`` to the regime's mask and renormalizing -- the naive slice
    only corrects for the regime's overall size, not for different states
    losing different *fractions* of their samples to the regime split. When
    ``f_k_global`` is not supplied (e.g. older/direct callers), this falls
    back to the previous naive mask-and-renormalize behavior.

    Returns ``(secondary_cv_pmf_info, cv1_cv2_fes_info)`` for the *dominant*
    regime (or the only regime, if there's just one) -- same shape/keys
    downstream code already expects. When multiple regimes exist, each gets
    its own analysis written under ``out/secondary_cv_regime_<type>/``, and
    both dominant-regime info dicts additionally carry a ``regime_breakdown``
    key with every regime's own info (including the dominant one).
    """
    regimes = _secondary_cv_epoch_regime_masks(d, warnings=warnings)
    if regimes is None:
        pmf_info = analyze_secondary_cv_pmf(d, args, base_logw, selected, boost_ok, kbt_kcal, out, warnings, progress)
        fes_info = (analyze_cv1_cv2_2d_fes(d, args, base_logw, selected, boost_ok, kbt_kcal, out, warnings, progress)
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
        d_regime = _masked_data(d, mask, meta_override=regime_meta)
        if f_k_global is not None:
            base_logw_regime = _subset_logw_from_global_fk(d_regime, f_k_global)
        else:
            base_logw_regime = np.asarray(base_logw, dtype=np.float64)[mask]
        regime_out = out if is_dominant else out / f'secondary_cv_regime_{_regime_slug(regime)}'
        regime_out.mkdir(parents=True, exist_ok=True)
        pmf_info = analyze_secondary_cv_pmf(d_regime, args, base_logw_regime, selected, boost_ok, kbt_kcal, regime_out, warnings, progress)
        fes_info = (analyze_cv1_cv2_2d_fes(d_regime, args, base_logw_regime, selected, boost_ok, kbt_kcal, regime_out, warnings, progress)
                    if isinstance(pmf_info, dict) and pmf_info.get('available')
                    else {'available': False, 'reason': 'Secondary CV PMF unavailable'})
        # Store shallow copies in the breakdown, not the live dicts -- the
        # dominant regime's own pmf_info/fes_info get a 'regime_breakdown' key
        # added to them below, and aliasing the same object here would nest
        # that dict inside itself (a real circular reference JSON serialization
        # rejects; caught by an actual end-to-end run against chignolin_5).
        breakdown[regime] = {
            'is_dominant': is_dominant, 'n_samples': int(np.count_nonzero(mask)),
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


def _poincare_primary_cv_supported(meta: dict) -> bool:
    meta = meta or {}
    mode = str(meta.get('primary_cv', '') or '').lower()
    label = str(meta.get('primary_cv_label', '') or '').lower()
    units = str(meta.get('primary_cv_units', '') or '').lower()
    return mode == 'nonlocal-contacts' or ('contact' in label and units in {'', 'dimensionless'})


_MERGED_TRAJ_STEP_STRIDE = 10_000_000_000  # must match _prepare_adaptive_merged_traj_dir


def _adjusted_steps_for_merged_traj(d: Data, spf: int = 50) -> Optional[np.ndarray]:
    """Return sample steps corrected for merged-dir resume offsets.

    In GAREUS, absolute sample steps include GaMD calibration steps (e.g. step 160050
    for calib=160000, report_interval=50).  Trajectory frame 0 of each epoch starts at
    relative step spf (step 1 within that epoch's production).  We subtract the
    calibration offset so relative steps are in [spf, n_frames*spf], then add the
    per-epoch STEP_STRIDE so each epoch maps to a non-overlapping window.
    """
    epoch_src = d.meta.get('_epoch_source')
    if not epoch_src:
        return None
    src = np.asarray(epoch_src, dtype=np.int64)
    if src.size != d.step.size:
        return None
    spf = max(1, int(spf))
    # Calibration offset: first production step corresponds to relative step = spf.
    # Subtracting it makes all epoch step ranges start at ~spf regardless of calib length.
    calib_offset = max(0, int(d.step.min()) - spf)
    relative = d.step.astype(np.int64) - calib_offset
    return relative + src * _MERGED_TRAJ_STEP_STRIDE


def _read_traj_interval_from_epoch_dirs(d: Data) -> int:
    """Read traj_interval from epoch run dirs when prod_dir has no effective_config."""
    for run_dir in d.meta.get('adaptive_epoch_run_dirs', []):
        spf = _read_traj_interval(Path(run_dir))
        if spf > 0:
            return spf
    return 500  # fallback default


def _get_adaptive_epoch_traj_dirs(d: Data) -> list:
    """Return [(epoch_idx, traj_dir)] for epochs that have non-empty replica_trajectories/."""
    run_dirs = d.meta.get('adaptive_epoch_run_dirs', [])
    result = []
    for i, run_dir in enumerate(run_dirs):
        traj_dir = Path(run_dir) / 'replica_trajectories'
        if traj_dir.is_dir() and any(traj_dir.iterdir()):
            result.append((i, traj_dir))
    return result


_MERGED_TRAJ_RESUME_RE = re.compile(r'^(.*)_resume_from_(\d+)$')


_MERGED_TRAJ_DIR_CACHE: dict = {}


def _prepare_adaptive_merged_traj_dir(d: Data, args=None) -> Optional[Path]:
    """Create/update a merged replica_trajectories/ in adaptive_production/ using symlinks.

    Each phase's own trajectory segments are linked so that
    _find_all_replica_trajectory_segments treats them as resume segments in
    chronological phase order, each keyed by a globally-unique
    ``resume_from_{epoch_idx * STEP_STRIDE + local_resume_start}`` offset --
    matching _adjusted_steps_for_merged_traj's own ``+ src * STEP_STRIDE``
    per-phase window. Empty (0-byte) files are skipped.

    A phase can itself have been interrupted and resumed multiple times
    within its own directory (SIGTERM + auto-resume writes a fresh
    ``replica_NNN_resume_from_<local_step>`` file each time, independent of
    which phase this is) -- the previous version of this function collapsed
    every one of a phase's OWN internal resume segments onto a single link
    name derived from ``epoch_idx`` alone, so only the first-processed
    segment per (phase, replica) survived; every other internal resume
    segment for that phase/replica was silently dropped before ever
    reaching the analysis. Confirmed on a real run (chignolin_5,
    epoch_001/baseline): 4 real trajectory segments per replica (1 base + 3
    internal resumes) collapsed to 1 merged file, discarding roughly 3/4 of
    that phase's coordinate data -- the dominant reason chignolin-FES/Rg/PCA
    only matched 856,655 of 3,348,831 available merged-directory frames
    (25.6%) even though the true per-segment step ranges are fully
    self-consistent and non-overlapping. Fixed by keying each segment's
    link by its own *local* resume_start (parsed from its original
    filename, 0 for a phase's true base file) folded into the same
    global-offset scheme, so every segment gets a distinct name.

    Memoized per-process by ``d.prod_dir`` (this function is called from
    several independent analysis functions within a single ``analyze()``
    run, all against the same prod_dir -- only the first call actually
    rebuilds; later calls reuse the cached result). Rebuilds are done into a
    fresh temp directory and atomically swapped into place via ``os.replace``
    rather than ``shutil.rmtree``-then-repopulate-in-place, so a concurrent
    reader (e.g. a second ``analyze_gareus_mbar.py`` process analyzing a
    different ``--epoch`` subset of the same run) never observes a
    partially-built directory.
    """
    epoch_traj = _get_adaptive_epoch_traj_dirs(d)
    if not epoch_traj:
        return None
    merged = d.prod_dir / '_merged_replica_trajectories'
    cache_key = str(merged)
    if cache_key in _MERGED_TRAJ_DIR_CACHE:
        return _MERGED_TRAJ_DIR_CACHE[cache_key]

    STEP_STRIDE = _MERGED_TRAJ_STEP_STRIDE
    TRAJ_EXTS = {'.xtc', '.dcd', '.nc', '.trr'}
    building = d.prod_dir / f'_merged_replica_trajectories.building-{os.getpid()}'
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    for epoch_idx, traj_dir in epoch_traj:
        for f in sorted(traj_dir.iterdir()):
            if f.suffix not in TRAJ_EXTS:
                continue
            if f.stat().st_size == 0:
                continue  # skip empty/unwritten trajectory files
            stem = f.stem  # e.g. "replica_000" or "replica_000_resume_from_3736350"
            m = _MERGED_TRAJ_RESUME_RE.match(stem)
            base_stem, local_resume = (m.group(1), int(m.group(2))) if m else (stem, 0)
            global_resume = epoch_idx * STEP_STRIDE + local_resume
            link = (building / f'{base_stem}{f.suffix}' if global_resume == 0
                    else building / f'{base_stem}_resume_from_{global_resume}{f.suffix}')
            if not link.exists():
                try:
                    link.symlink_to(f.resolve())
                except Exception:
                    pass

    if not any(building.iterdir()):
        shutil.rmtree(building, ignore_errors=True)
        _MERGED_TRAJ_DIR_CACHE[cache_key] = None
        return None

    stale = d.prod_dir / f'_merged_replica_trajectories.stale-{os.getpid()}'
    if merged.exists():
        os.replace(merged, stale)
    os.replace(building, merged)
    if stale.exists():
        shutil.rmtree(stale, ignore_errors=True)

    _MERGED_TRAJ_DIR_CACHE[cache_key] = merged
    return merged


def run_epoch_pmf_convergence(d: Data, args, bins: np.ndarray, selected: str,
                               final_pmf: dict, out_base: Path,
                               progress: Optional[Progress] = None,
                               f_init_hint: Optional[np.ndarray] = None) -> dict:
    """PMF convergence by cumulative epoch addition for adaptive-production runs.

    Iterates [epoch_0], [epoch_0+1], ..., [all epochs].  Each prefix runs
    MBAR on the pooled samples from those epochs and records PMF + JS/RMSE
    vs the full-run final PMF.
    """
    if bool(getattr(args, 'no_convergence', False)):
        return {'enabled': False, 'metric': 'epoch_convergence', 'reason': 'disabled'}
    epoch_src = d.meta.get('_epoch_source')
    if not epoch_src:
        return {'enabled': False, 'metric': 'epoch_convergence', 'reason': 'no epoch tracking'}
    epoch_src = np.asarray(epoch_src, dtype=np.int32)
    if epoch_src.size != d.cv.size:
        return {'enabled': False, 'metric': 'epoch_convergence', 'reason': 'epoch_source size mismatch'}
    n_epochs = int(epoch_src.max()) + 1
    if n_epochs < 2:
        return {'enabled': False, 'metric': 'epoch_convergence', 'reason': f'only {n_epochs} epoch'}
    run_dirs = d.meta.get('adaptive_epoch_run_dirs', [])
    out = out_base / 'epoch_convergence'
    out.mkdir(parents=True, exist_ok=True)
    ref_prob = pmf_probability(final_pmf)
    ref_F = np.asarray(final_pmf['pmf'], dtype=np.float64)
    ref_counts = np.asarray(final_pmf.get('counts', np.zeros_like(ref_prob)), dtype=float)
    ref_occ = int(np.count_nonzero(ref_counts > 0))
    K = d.u_nk.shape[1]
    kbt_kcal = (1.0 / d.beta) / KJ_PER_KCAL
    conv_backend = str(getattr(args, 'convergence_mbar_backend', 'sambar') or 'sambar')
    conv_sambar_epochs = int(getattr(args, 'convergence_sambar_epochs', 10) or 10)
    conv_sambar_batch = int(getattr(args, 'convergence_sambar_initial_batch_size', SAMBAR_INITIAL_BATCH_SIZE) or SAMBAR_INITIAL_BATCH_SIZE)
    conv_sambar_patience = int(getattr(args, 'convergence_sambar_batch_patience', 2) or 2)
    conv_sambar_seed = int(getattr(args, 'convergence_sambar_seed', SAMBAR_SEED) or SAMBAR_SEED)
    conv_sambar_lr = float(getattr(args, 'convergence_sambar_lr_scale', SAMBAR_LR_SCALE) or SAMBAR_LR_SCALE)
    conv_sambar_delta = float(getattr(args, 'convergence_sambar_delta_f_max', SAMBAR_DELTA_F_MAX) or SAMBAR_DELTA_F_MAX)
    conv_sambar_polish = str(getattr(args, 'convergence_sambar_polish_backend', SAMBAR_POLISH_BACKEND) or SAMBAR_POLISH_BACKEND)
    conv_mbar_tol = float(getattr(args, 'convergence_mbar_tol', 1e-6) or 1e-6)
    conv_mbar_maxiter = int(getattr(args, 'convergence_mbar_maxiter', 2000) or 2000)
    prev_f_k = np.asarray(f_init_hint, dtype=np.float64) if f_init_hint is not None else None
    prev_prob = prev_F = prev_counts = None
    conv_rows: list = []; pmf_rows: list = []
    if progress is not None:
        progress.step('epoch convergence', f'{n_epochs} epochs; backend={conv_backend}')
    for n_ep in range(1, n_epochs + 1):
        mask = epoch_src < n_ep
        n = int(np.count_nonzero(mask))
        if n < max(5, K):
            continue
        if progress is not None:
            progress.bar('epoch_convergence', n_ep, n_epochs,
                         f'epochs 0-{n_ep-1}: {n} samples')
        sub_u = d.u_nk[mask]; sub_w = d.window[mask]
        sub_cv = d.cv[mask]; sub_boost = d.boost_kj[mask]
        try:
            mb = solve_mbar(sub_u, sub_w,
                            tol=conv_mbar_tol, maxiter=conv_mbar_maxiter,
                            progress=None, backend=conv_backend,
                            threads=getattr(args, 'mbar_threads', 0),
                            f_init=prev_f_k,
                            sambar_epochs=conv_sambar_epochs,
                            sambar_initial_batch_size=min(n, conv_sambar_batch),
                            sambar_batch_patience=conv_sambar_patience,
                            sambar_seed=conv_sambar_seed,
                            sambar_lr_scale=conv_sambar_lr,
                            sambar_delta_f_max=conv_sambar_delta,
                            sambar_polish_backend=conv_sambar_polish)
            prev_f_k = mb.get('f_k')
            # ESS diagnostics: MBAR umbrella-debiasing ESS, plus the GaMD
            # exponential-reweighting ESS (umbrella + boost).  The latter exposes
            # the boost-reweighting collapse — it stays tiny no matter how many
            # epochs accumulate, while MBAR ESS grows with samples.
            _logw = np.asarray(mb['logw'], dtype=np.float64)
            mbar_ess = float(ess(norm_logw(_logw)))
            gamd_reweight_ess = float('nan')
            if np.any(np.isfinite(sub_boost)) and float(np.nanstd(sub_boost)) > 1e-12:
                gamd_reweight_ess = float(ess(norm_logw(_logw + d.beta * sub_boost)))
            pmf, _diag, used_method = _observable_pmf_from_logw(
                sub_cv, mb['logw'], sub_boost, bins, selected,
                d.beta, kbt_kcal,
                smooth_logfac_sigma=_eff_smooth(args, 'gamd_smooth_sigma'))
            prob = pmf_probability(pmf)
            F = np.asarray(pmf['pmf'], dtype=float)
            counts = np.asarray(pmf['counts'], dtype=float)
            js_vs_final = js_divergence_1d(prob, ref_prob)
            rmse_vs_final = pmf_rmse_1d(F, ref_F, prob, ref_prob)
            delta_js = js_divergence_1d(prob, prev_prob) if prev_prob is not None else np.nan
            delta_rmse = pmf_rmse_1d(F, prev_F, prob, prev_prob) if prev_F is not None else np.nan
            epoch_label = str(run_dirs[n_ep - 1]) if n_ep - 1 < len(run_dirs) else f'epoch_{n_ep-1}'
            row = {
                'variant': 'epoch', 'metric': 'cv_distance',
                'selected_method': used_method,
                'checkpoint_index': n_ep, 'checkpoint_step': n_ep,
                'n_epochs_included': n_ep, 'epoch_label': epoch_label,
                'n_samples': n, 'n_samples_total': int(d.cv.size),
                'frac_total': float(n / max(1, d.cv.size)),
                'JS': float(js_vs_final), 'RMSE_F_kcal_mol': float(rmse_vs_final),
                'barrier_error_kcal_mol': barrier_error_1d(F, ref_F, prob, ref_prob),
                'delta_JS': float(delta_js), 'delta_RMSE_F_kcal_mol': float(delta_rmse),
                'occupied_bins': int(np.count_nonzero(counts > 0)),
                'occupied_bins_ref': ref_occ,
                'occupied_bins_frac_ref': float(np.count_nonzero(counts > 0) / max(1, ref_occ)),
                'mbar_converged': int(bool(mb.get('converged'))),
                'mbar_iterations': int(mb.get('iterations', 0)),
                'mbar_max_delta': float(mb.get('max_delta', np.nan)),
                'mbar_backend': str(mb.get('backend', 'unknown')),
                'mbar_ess': mbar_ess,
                'mbar_ess_frac': float(mbar_ess / max(1, n)),
                'gamd_reweight_ess': gamd_reweight_ess,
                'gamd_reweight_ess_frac': (float(gamd_reweight_ess / max(1, n))
                                           if math.isfinite(gamd_reweight_ess) else float('nan')),
            }
            conv_rows.append(row)
            pmf_rows.append({'checkpoint_index': n_ep, 'checkpoint_step': n_ep,
                              'frac_total': row['frac_total'],
                              'x': pmf['cv_A'].copy(), 'pmf_kcal_mol': F.copy(),
                              'probability': prob.copy(), 'counts': counts.astype(int).copy()})
            prev_prob = prob; prev_F = F; prev_counts = counts
        except Exception as exc:
            conv_rows.append({'variant': 'epoch', 'metric': 'cv_distance',
                               'checkpoint_index': n_ep, 'n_epochs_included': n_ep,
                               'n_samples': n, 'error': str(exc)})
    if not conv_rows:
        return {'enabled': False, 'metric': 'epoch_convergence', 'reason': 'no valid epochs'}
    _write_csv_rows(out / 'epoch_pmf_convergence.csv', conv_rows)
    flat = []
    for p in pmf_rows:
        for i, x in enumerate(p['x']):
            flat.append({'variant': 'epoch', 'metric': 'cv_distance',
                         'checkpoint_index': p['checkpoint_index'],
                         'n_epochs': p['checkpoint_index'],
                         'frac_total': p['frac_total'],
                         'bin': i, 'x': float(x),
                         'F_kcal_mol': float(p['pmf_kcal_mol'][i]) if np.isfinite(p['pmf_kcal_mol'][i]) else '',
                         'P': float(p['probability'][i]),
                         'count': int(p['counts'][i])})
    _write_csv_rows(out / 'epoch_pmf_by_epoch.csv', flat)
    # Per-source pooling table: confirms every epoch + final source fed the PMF.
    pooling = _epoch_source_pooling_table(d)
    if pooling:
        _write_csv_rows(out / 'epoch_source_pooling.csv', pooling)
    # Flag planned epoch_NNN dirs that produced no loaded samples so they don't
    # read as silently dropped.
    skipped_epochs = _skipped_empty_epochs(d)
    if progress is not None:
        progress.step('epoch pooling',
                      f'{len(pooling)} sources pooled (incl. final)'
                      + (f'; empty epochs skipped: {", ".join(skipped_epochs)}' if skipped_epochs else ''))
        progress.bar('epoch_convergence', 1, 1, 'writing epoch convergence outputs', force=True)
    finite_rows = [r for r in conv_rows if 'JS' in r and math.isfinite(float(r.get('JS', math.nan)))]
    summary = {}
    if finite_rows:
        arr_js = np.asarray([r['JS'] for r in finite_rows], float)
        arr_rmse = np.asarray([r['RMSE_F_kcal_mol'] for r in finite_rows], float)
        js_thr = float(getattr(args, 'convergence_js_threshold', 0.01))
        rmse_thr = float(getattr(args, 'convergence_rmse_threshold', 0.1))
        _last = finite_rows[-1]
        summary = {
            'final_JS': float(arr_js[-1]) if arr_js.size else float('nan'),
            'final_RMSE_F_kcal_mol': float(arr_rmse[-1]) if arr_rmse.size else float('nan'),
            'converged_JS': bool(arr_js[-1] < js_thr) if arr_js.size else False,
            'converged_RMSE': bool(arr_rmse[-1] < rmse_thr) if arr_rmse.size else False,
            'final_mbar_ess': float(_last.get('mbar_ess', float('nan'))),
            'final_mbar_ess_frac': float(_last.get('mbar_ess_frac', float('nan'))),
            'final_gamd_reweight_ess': float(_last.get('gamd_reweight_ess', float('nan'))),
            'final_gamd_reweight_ess_frac': float(_last.get('gamd_reweight_ess_frac', float('nan'))),
        }
        summary['converged'] = summary['converged_JS'] and summary['converged_RMSE']
    _ea_ec = _epoch_source_annotations(d)
    _agg_ec = None
    try:
        _fracs_ec = np.asarray([r['frac_total'] for r in conv_rows], dtype=float)
        _ts_ec = float(d.meta.get('timestep_fs', 4.0) or 4.0)
        _agg_ec = _aggregate_ns_for_fracs(d, _fracs_ec, _ts_ec)
        write_convergence_plots(conv_rows, pmf_rows, [summary] if summary else [],
                                out, args, warnings=[], epoch_annotations=_ea_ec,
                                aggregate_ns=_agg_ec)
    except Exception:
        pass
    try:
        _write_epoch_ess_plot(conv_rows, out, ea=_ea_ec, aggregate_ns=_agg_ec)
    except Exception:
        pass
    wjson(out / 'epoch_convergence_summary.json',
          {'n_epochs': n_epochs, 'n_rows': len(conv_rows), 'summary': summary,
           'n_sources_pooled': len(pooling), 'sources': [p['label'] for p in pooling],
           'skipped_empty_epochs': skipped_epochs, 'source_pooling': pooling,
           'output_dir': str(out)})
    return {'enabled': True, 'metric': 'epoch_convergence',
            'n_epochs': n_epochs, 'n_rows': len(conv_rows),
            'summary': summary, 'n_sources_pooled': len(pooling),
            'sources': [p['label'] for p in pooling], 'skipped_empty_epochs': skipped_epochs,
            'output_dir': str(out)}


def ess(w):
    w=np.asarray(w,float); w=w[np.isfinite(w)&(w>=0)]
    if w.size==0: return 0.0
    s1=float(np.sum(w)); s2=float(np.sum(w*w)); return 0.0 if s2<=0 else s1*s1/s2

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
            w_b = norm_logw(logw_b)
            rep_pmf = pmf_from_weights(cv_b, w_b, bins, kbt_kcal)
        elif selected_method == 'gamd_exponential':
            w_b = norm_logw(logw_b + beta * boost_b)
            rep_pmf = pmf_from_weights(cv_b, w_b, bins, kbt_kcal)
        elif selected_method in ('gamd_cumulant2', 'gamd_cumulant3'):
            base_w_b = norm_logw(logw_b)
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
            w_b = norm_logw(logw_b)
            rep_fes = pmf2d_from_weights(x_b, y_b, w_b, xbins, ybins, kbt_kcal)
        elif selected_method == 'gamd_exponential':
            w_b = norm_logw(logw_b + beta * boost_b)
            rep_fes = pmf2d_from_weights(x_b, y_b, w_b, xbins, ybins, kbt_kcal)
        elif selected_method in ('gamd_cumulant2', 'gamd_cumulant3'):
            base_w_b = norm_logw(logw_b)
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

from gareus.mbar_analysis.writers import (
    write_2d_fes_csv, write_2d_fes_npz, write_cv1_cv2_2d_fes_csv, write_cv1_cv2_2d_fes_npz,
    write_pca_2d_fes_csv, write_pca_2d_fes_npz, _write_scalar_pmfs, _write_generic_2d_fes,
    _write_rama_2d, write_rg_pmf, write_rg_all, write_pmf, write_all, write_cv2_pmf,
    _write_csv_rows,
)


from gareus.mbar_analysis.plotting import (
    _smooth_masked_grid, _smooth_pmf_1d, _eff_smooth, FES_PLOT_VMAX_VALUES,
    _fes_range_tag, _fes_range_label, _fes_variant_path, _plot_2d_fes_range,
    _plot_2d_fes_multirange, plot_2d_fes, plot_cv1_cv2_2d_fes, plot_pca_2d_fes,
    _per_window_gamd_boost_stats, plot_gamd_boost, plot_outputs,
    plot_rg_outputs, _OPT_IN_GAMD_METHODS, _want_gamd_method, _visible_pmfs,
)


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


def pmf_probability(pmf: dict) -> np.ndarray:
    prob=np.asarray(pmf.get('prob',[]),dtype=np.float64)
    total=float(np.nansum(prob))
    if total>0:
        prob=prob/total
    return np.where(np.isfinite(prob)&(prob>=0),prob,0.0)

def js_divergence_1d(P,Q):
    P=pmf_probability({'prob':P})
    Q=pmf_probability({'prob':Q})
    if P.size!=Q.size:
        n=min(P.size,Q.size); P=P[:n]; Q=Q[:n]
    M=0.5*(P+Q)
    with np.errstate(divide='ignore',invalid='ignore'):
        a=np.where(P>0,P*np.log(P/np.maximum(M,1e-300)),0.0)
        b=np.where(Q>0,Q*np.log(Q/np.maximum(M,1e-300)),0.0)
    return float(0.5*(np.sum(a)+np.sum(b)))

def pmf_rmse_1d(F,Fref,P,Pref,min_prob=0.0):
    F=np.asarray(F,dtype=np.float64); Fref=np.asarray(Fref,dtype=np.float64)
    P=np.asarray(P,dtype=np.float64); Pref=np.asarray(Pref,dtype=np.float64)
    n=min(F.size,Fref.size,P.size,Pref.size)
    if n<=0: return float('nan')
    F=F[:n]; Fref=Fref[:n]; P=P[:n]; Pref=Pref[:n]
    mask=np.isfinite(F)&np.isfinite(Fref)&(P>float(min_prob))&(Pref>float(min_prob))
    if not np.any(mask): return float('nan')
    d=F[mask]-Fref[mask]
    return float(np.sqrt(np.mean(d*d)))

def barrier_error_1d(F,Fref,P,Pref):
    F=np.asarray(F,dtype=np.float64); Fref=np.asarray(Fref,dtype=np.float64)
    P=np.asarray(P,dtype=np.float64); Pref=np.asarray(Pref,dtype=np.float64)
    n=min(F.size,Fref.size,P.size,Pref.size)
    if n<=0: return float('nan')
    mask=np.isfinite(F[:n])&np.isfinite(Fref[:n])&(P[:n]>0)&(Pref[:n]>0)
    if not np.any(mask): return float('nan')
    return float(abs(np.nanmax(F[:n][mask])-np.nanmax(Fref[:n][mask])))

def identify_basins_1d(cv_A: np.ndarray, F: np.ndarray, min_depth_kcal: float = 0.5) -> list:
    """Find basins in a 1D PMF and partition the CV axis into basin domains.

    Returns a list of dicts (sorted by CV position):
      basin_id, center_cv_A, min_F_kcal, prominence_kcal,
      left_bin, right_bin (inclusive bin indices into cv_A/F),
      left_cv_A, right_cv_A.

    Basin boundaries sit at the local maximum between adjacent minima so that
    every bin belongs to exactly one basin and populations sum to 1.
    """
    cv_A = np.asarray(cv_A, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    n = len(F)
    if n < 3:
        return []
    F_safe = np.where(np.isfinite(F), F, np.inf)
    neg_F = np.where(np.isfinite(F_safe), -F_safe, -np.inf)
    minima_idx: list = []; prominences: list = []
    if SCIPY_SIGNAL_AVAILABLE and _scipy_find_peaks is not None:
        try:
            peaks, props = _scipy_find_peaks(neg_F, prominence=min_depth_kcal)
            minima_idx = list(peaks); prominences = list(props['prominences'])
        except Exception:
            pass
    if not minima_idx:
        finite_mask = np.isfinite(F)
        for i in range(1, n - 1):
            if not finite_mask[i]: continue
            left_ok = finite_mask[:i]; right_ok = finite_mask[i+1:]
            if not (left_ok.any() and right_ok.any()): continue
            if F[i] >= F[i-1] or F[i] >= F[i+1]: continue
            lmax = float(np.max(F[:i][left_ok])); rmax = float(np.max(F[i+1:][right_ok]))
            prom = min(lmax, rmax) - F[i]
            if prom >= min_depth_kcal:
                minima_idx.append(i); prominences.append(float(prom))
    if not minima_idx:
        finite_idx = np.where(np.isfinite(F))[0]
        if len(finite_idx) == 0: return []
        gmin = int(finite_idx[np.argmin(F[finite_idx])])
        minima_idx = [gmin]; prominences = [0.0]
    order = np.argsort(minima_idx)
    minima_idx = [minima_idx[i] for i in order]; prominences = [prominences[i] for i in order]
    # barrier_bins[i] = argmax between minima_idx[i] and minima_idx[i+1]
    barrier_bins = []
    for i in range(len(minima_idx) - 1):
        lo, hi = minima_idx[i], minima_idx[i+1]
        barrier_bins.append(lo + int(np.argmax(F_safe[lo:hi+1])))
    # Partition: basin i covers [left_edges[i], right_edges[i]] inclusive.
    # Barrier bin belongs to the left basin so the union is [0, n-1] without gaps or overlap.
    left_edges  = [0] + [b + 1 for b in barrier_bins]
    right_edges = barrier_bins + [n - 1]
    basins = []
    for i, mi in enumerate(minima_idx):
        lb, rb = left_edges[i], right_edges[i]
        basins.append({'basin_id': i, 'center_cv_A': float(cv_A[mi]), 'min_F_kcal': float(F[mi]) if np.isfinite(F[mi]) else float('nan'), 'prominence_kcal': float(prominences[i]), 'left_bin': int(lb), 'right_bin': int(rb), 'left_cv_A': float(cv_A[lb]), 'right_cv_A': float(cv_A[rb])})
    return basins


def _compute_basin_populations(prob: np.ndarray, basins: list) -> list:
    prob = np.asarray(prob, dtype=np.float64)
    return [float(np.sum(prob[b['left_bin']:b['right_bin']+1])) for b in basins]


def _short_source_label(rd) -> str:
    """Short display label for one adaptive epoch / baseline / topup / final source dir.

    examples: adaptive_production/epoch_000          -> 'epoch_000'
              adaptive_production/epoch_001/baseline  -> 'epoch_001/baseline'
              adaptive_production/final/topup_001_658000 -> 'final/topup_001@658k'
    """
    import re as _re
    p = Path(str(rd))
    name = p.name
    parent = p.parent.name
    label = name if parent == 'adaptive_production' else f'{parent}/{name}'
    # Shorten topup step suffix: topup_001_658000 -> topup_001@658k
    return _re.sub(r'topup_(\d+)_(\d+)', lambda m: f'topup_{m.group(1)}@{int(m.group(2))//1000}k', label)


def _epoch_source_annotations(d: 'Data') -> list:
    """Return [(short_label, frac_end, color), ...] for convergence-plot boundary markers.

    Each source (epoch / baseline / topup / final) contributes a contiguous block
    of the pooled samples in source order.  The epoch-convergence x-axis is
    ``frac_total = cum_samples(sources 0..k) / N``, so source k's right-edge boundary
    is ``cum_count(src <= k) / N``.

    This is independent of the per-epoch-LOCAL step counter (which resets each
    epoch): the previous implementation sorted by ``d.step`` alone, which pushed the
    longest-running epoch (largest local step) to the far right and short late
    top-ups to the left — i.e. the markers came out reversed relative to the curve.
    """
    src_meta = d.meta.get('_epoch_source')
    if not src_meta or len(src_meta) != d.step.size:
        return []
    src = np.asarray(src_meta, dtype=np.int64)
    N = src.size
    run_dirs = d.meta.get('adaptive_epoch_run_dirs', [])
    if N == 0 or not run_dirs:
        return []
    src_labels = [_short_source_label(rd) for rd in run_dirs]

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999']
    counts = np.bincount(src, minlength=len(src_labels))
    cum = np.cumsum(counts)
    result = []
    for s_idx, label in enumerate(src_labels):
        if s_idx >= counts.size or counts[s_idx] == 0:
            continue
        frac_end = float(cum[s_idx]) / max(1, N)  # right edge of this source in source order
        result.append((label, frac_end, colors[s_idx % len(colors)]))
    result.sort(key=lambda x: x[1])
    return result


def _epoch_source_pooling_table(d: 'Data', timestep_fs: float = 4.0) -> list:
    """Per-source pooling summary for the MBAR union that was actually built.

    Confirms which epoch / baseline / topup / final sources were pooled into the
    PMF and how much each contributed.  Built from ``d.meta['_epoch_source']`` (the
    real loaded union) rather than a disk rescan, so it reflects the PMF that was
    constructed.  Returns [] when epoch-source metadata is unavailable.
    """
    src_meta = d.meta.get('_epoch_source')
    if not src_meta or len(src_meta) != d.step.size:
        return []
    src = np.asarray(src_meta, dtype=np.int64)
    N = src.size
    run_dirs = d.meta.get('adaptive_epoch_run_dirs', [])
    if N == 0 or not run_dirs:
        return []
    labels = [_short_source_label(rd) for rd in run_dirs]
    ts = float(d.meta.get('timestep_fs', timestep_fs) or timestep_fs)
    agg = _epoch_source_aggregate_ns_info(d, ts)
    cum_ns = agg[2] if agg is not None else None
    rows = []
    cum_n = 0
    for s_idx in range(len(labels)):
        mask = src == s_idx
        n = int(mask.sum())
        if n == 0:
            continue
        cum_n += n
        step_s = d.step[mask]
        rows.append({
            'src_idx': s_idx,
            'label': labels[s_idx],
            'n_samples': n,
            'frac_total': float(n / N),
            'cum_frac': float(cum_n / N),
            'n_windows': int(np.unique(d.window[mask]).size),
            'step_min': int(step_s.min()) if step_s.size else 0,
            'step_max': int(step_s.max()) if step_s.size else 0,
            'cum_aggregate_ns': (float(cum_ns[s_idx + 1])
                                 if cum_ns is not None and s_idx + 1 < len(cum_ns) else float('nan')),
        })
    return rows


def _skipped_empty_epochs(d: 'Data') -> list:
    """epoch_NNN dirs on disk that contributed no loaded samples to the union.

    e.g. an aborted/promoted epoch that holds only epoch_window_map.csv.  Listing
    these makes clear they were intentionally skipped, not silently dropped.
    """
    run_dirs = d.meta.get('adaptive_epoch_run_dirs', [])
    if not run_dirs:
        return []
    try:
        ap = d.prod_dir if d.prod_dir.name == 'adaptive_production' else d.prod_dir / 'adaptive_production'
        loaded_tops = {_short_source_label(rd).split('/')[0] for rd in run_dirs}
        return [ed.name for ed in sorted(ap.glob('epoch_[0-9][0-9][0-9]'))
                if ed.is_dir() and ed.name not in loaded_tops]
    except Exception:
        return []


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


def _epoch_source_aggregate_ns_info(d: 'Data', timestep_fs: float = 4.0):
    """Per-source (n_wins, step_range, cumulative_ns) triple for aggregate cost computation.

    Returns None when _epoch_source metadata is absent or mismatched.
    Aggregate ns for source k = n_windows_k × (max_step_k − min_step_k) × timestep_fs / 1e6.
    """
    src_meta = d.meta.get('_epoch_source')
    if not src_meta or len(src_meta) != d.step.size:
        return None
    src = np.asarray(src_meta, dtype=np.int64)
    n_sources = int(src.max()) + 1
    n_wins: list = []
    step_ranges: list = []
    for k in range(n_sources):
        mask = src == k
        if not mask.any():
            n_wins.append(1)
            step_ranges.append((0, 0))
        else:
            n_wins.append(max(1, int(len(np.unique(d.window[mask])))))
            s = d.step[mask]
            step_ranges.append((int(s.min()), int(s.max())))
    source_ns = [n_wins[k] * (step_ranges[k][1] - step_ranges[k][0]) * timestep_fs / 1e6
                 for k in range(n_sources)]
    cumulative = [0.0] + list(np.cumsum(source_ns).tolist())
    return n_wins, step_ranges, cumulative


def _aggregate_ns_for_fracs(d: 'Data', fracs: np.ndarray, timestep_fs: float = 4.0) -> Optional[np.ndarray]:
    """Map fraction-of-samples values to cumulative aggregate simulation time (ns).

    Cost = n_active_windows × sim_steps × timestep (REUS parallel replicas all run
    simultaneously, so GPU cost scales with window count, not just wall-clock steps).
    Samples are ordered by (src_idx, step) = chronological order across epochs.
    Returns None when epoch-source metadata is unavailable.
    """
    info = _epoch_source_aggregate_ns_info(d, timestep_fs)
    if info is None:
        return None
    n_wins, step_ranges, cumulative = info
    src = np.asarray(d.meta['_epoch_source'], dtype=np.int64)
    N = d.step.size
    adj_step = src * int(1e10) + d.step.astype(np.int64)
    sort_idx = np.argsort(adj_step, kind='stable')
    src_sorted = src[sort_idx]
    step_sorted = d.step[sort_idx]
    completed = np.array([cumulative[k] for k in src_sorted], dtype=np.float64)
    step_min = np.array([step_ranges[k][0] for k in src_sorted], dtype=np.float64)
    n_wins_arr = np.array([n_wins[k] for k in src_sorted], dtype=np.float64)
    partial = n_wins_arr * np.maximum(0.0, step_sorted.astype(np.float64) - step_min) * timestep_fs / 1e6
    cum_agg_ns = completed + partial
    sample_fracs = (np.arange(N, dtype=np.float64) + 1.0) / N
    return np.interp(np.asarray(fracs, dtype=np.float64), sample_fracs, cum_agg_ns)


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


def checkpoint_steps_from_data(step: np.ndarray, n_timepoints: int) -> np.ndarray:
    steps=np.asarray(step,dtype=np.int64)
    steps=steps[np.isfinite(steps)]
    if steps.size==0:
        return np.arange(1,max(2,int(n_timepoints))+1,dtype=np.int64)
    uniq=np.unique(steps)
    if uniq.size<=int(n_timepoints):
        return uniq
    idx=np.rint(np.linspace(0,uniq.size-1,int(n_timepoints))).astype(int)
    idx=np.unique(np.clip(idx,0,uniq.size-1))
    if idx[-1] != uniq.size-1:
        idx=np.unique(np.concatenate([idx,[uniq.size-1]]))
    return uniq[idx]

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

def write_convergence_report(path: Path, row: dict, conv_rows: list[dict], warnings: list[str]) -> None:
    lines=['# PMF convergence report','']
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


def _observable_pmf_from_logw(values: np.ndarray, logw: np.ndarray, boost: np.ndarray, bins: np.ndarray, selected: str, beta: float, kbt_kcal: float, smooth_logfac_sigma: float = 0.0):
    """Build an unbiased 1D PMF for any scalar observable from MBAR log weights.

    The observable can be CV distance, Rg, SASA, contacts, helicity, or any
    future per-sample scalar.  Umbrella unbiasing comes from MBAR log weights;
    optional GaMD reweighting uses the same exponential/cumulant choices as the
    main CV analysis.
    """
    values=np.asarray(values,dtype=np.float64)
    logw=np.asarray(logw,dtype=np.float64)
    boost=np.asarray(boost,dtype=np.float64)
    selected=str(selected or 'umbrella_only')
    base_w=norm_logw(logw)
    boost_ok=np.isfinite(boost).sum()>10 and np.nanstd(boost)>1.0e-12
    if selected=='gamd_cumulant3' and boost_ok:
        pmf,diag=cumulant3(values,base_w,boost,bins,beta,kbt_kcal,smooth_logfac_sigma=smooth_logfac_sigma)
        return pmf, diag, 'gamd_cumulant3'
    if selected=='gamd_cumulant2' and boost_ok:
        pmf,diag=cumulant2(values,base_w,boost,bins,beta,kbt_kcal,smooth_logfac_sigma=smooth_logfac_sigma)
        return pmf, diag, 'gamd_cumulant2'
    if selected=='gamd_exponential' and boost_ok:
        ew=norm_logw(logw+beta*boost)
        return pmf_from_weights(values,ew,bins,kbt_kcal), {}, 'gamd_exponential'
    return pmf_from_weights(values,base_w,bins,kbt_kcal), {}, 'umbrella_only'


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


def _get_convergence_mbar_cache(args) -> dict:
    """Return the per-run cache for prefix MBAR solves.

    Convergence for CV, Rg, cv2, SASA, contacts, etc. often uses the same
    prefix sample mask. MBAR weights depend only on u_nk/window plus that mask,
    not on which scalar observable is histogrammed, so those solves can be
    safely reused within one analysis run.
    """
    cache = getattr(args, '_convergence_mbar_cache', None)
    if cache is None:
        cache = {}
        try:
            setattr(args, '_convergence_mbar_cache', cache)
        except Exception:
            return {}
    return cache


def _convergence_mask_digest(mask: np.ndarray) -> str:
    mask = np.asarray(mask, dtype=np.bool_)
    packed = np.packbits(mask)
    h = hashlib.blake2b(digest_size=12)
    h.update(np.asarray(mask.shape, dtype=np.int64).tobytes())
    h.update(packed.tobytes())
    return h.hexdigest()


def _convergence_mbar_cache_key(d: Data, args, mask: np.ndarray, checkpoint_step: int, conv_backend: str,
                                conv_mbar_maxiter: int, conv_sambar_epochs: int, conv_sambar_batch: int,
                                conv_sambar_patience: int, conv_sambar_seed: int, conv_sambar_lr: float,
                                conv_sambar_delta: float, conv_sambar_polish: str) -> tuple:
    return (
        str(d.prod_dir),
        tuple(np.asarray(d.u_nk).shape),
        int(checkpoint_step),
        _convergence_mask_digest(mask),
        str(conv_backend),
        float(getattr(args, 'convergence_mbar_tol', 1e-6)),
        int(conv_mbar_maxiter),
        int(getattr(args, 'mbar_threads', 0) or 0),
        int(getattr(args, 'mbar_anderson_history', MBAR_ANDERSON_HISTORY) or MBAR_ANDERSON_HISTORY),
        int(conv_sambar_epochs),
        int(conv_sambar_batch),
        int(conv_sambar_patience),
        int(conv_sambar_seed),
        float(conv_sambar_lr),
        float(conv_sambar_delta),
        str(conv_sambar_polish),
    )


def _get_checkpoint_steps_cache(args) -> dict:
    """Return the per-run cache for checkpoint_steps_from_data results.

    Mirrors _get_convergence_mbar_cache below it. run_observable_pmf_convergence
    is called ~20+ times per analyze() run (main CV1, Rg, secondary CV2,
    per-residue phi/psi, SASA, contact-count family, secondary-structure
    fractions), and most of those calls share both the same production `d`
    (only the main CV1 call uses a genuinely different, possibly-masked
    population) and, for the trajectory-derived observables, the same
    finite-sample mask: which samples have a real trajectory-derived value is
    a property of which frames were scanned, not of which per-frame
    observable is being histogrammed from them (see
    analyze_extra_observable_pmfs -- phi/psi/SASA/contacts/secondary-structure
    are all filled from the same per-chunk `sample_idx`). checkpoint_steps_
    from_data's np.unique(step) is then identical work repeated many times.
    Keyed on (production dir, population size, a content digest of the
    finite-sample mask, n_timepoints) -- content-based, like
    _convergence_mbar_cache_key, not object identity, since callers filter a
    fresh boolean-indexed copy of `step` on every call, so a naive id()-keyed
    cache would simply never hit. That key alone isn't a airtight guarantee
    two different Data populations can never coincide on it (e.g. two
    same-size masked subsets of the same run), so each cache entry also
    stores the exact `d.step` array the result was computed from; the call
    site (run_observable_pmf_convergence) requires `is` identity against it
    before trusting a hit, making a wrong reuse structurally impossible
    regardless of key collisions.
    """
    cache = getattr(args, '_checkpoint_steps_cache', None)
    if cache is None:
        cache = {}
        try:
            setattr(args, '_checkpoint_steps_cache', cache)
        except Exception:
            return {}
    return cache


def _checkpoint_steps_cache_key(d: Data, finite_global: np.ndarray, n_timepoints: int) -> tuple:
    return (
        str(d.prod_dir),
        int(np.asarray(d.step).shape[0]),
        _convergence_mask_digest(finite_global),
        int(n_timepoints),
    )


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


def run_observable_pmf_convergence(
    d: Data,
    args,
    values: np.ndarray,
    bins: np.ndarray,
    selected: str,
    final_pmf: dict,
    out_base: Path,
    *,
    metric_name: str,
    metric_label: str,
    x_label: str,
    out_dir_name: str,
    file_prefix: str,
    legacy_total_names: bool = False,
    basin_tracking: bool = False,
    progress: Optional[Progress] = None,
    f_init_hint: Optional[np.ndarray] = None,
) -> dict:
    """Recompute MBAR/reweighted PMFs over prefix timepoints for any scalar observable.

    This is the generic convergence engine.  The only observable-specific inputs
    are the per-sample scalar values, bin edges, names/labels, and the final PMF
    used as the full-production reference.
    """
    if bool(getattr(args,'no_convergence',False)):
        return {'enabled':False,'metric':metric_name,'reason':'disabled'}
    values=np.asarray(values,dtype=np.float64)
    finite_global=np.isfinite(values)
    if np.count_nonzero(finite_global)<max(5,d.u_nk.shape[1]):
        return {'enabled':False,'metric':metric_name,'reason':'too few finite observable samples','n_finite':int(np.count_nonzero(finite_global))}
    out=out_base/out_dir_name
    out.mkdir(parents=True,exist_ok=True)
    _n_timepoints=int(getattr(args,'convergence_timepoints',10))
    _ckpt_cache=_get_checkpoint_steps_cache(args)
    _ckpt_cache_key=_checkpoint_steps_cache_key(d,finite_global,_n_timepoints)
    _ckpt_hit=_ckpt_cache.get(_ckpt_cache_key)
    # The cache key is content-based (prod_dir/population size/mask digest),
    # not a guarantee that two Data objects sharing those can't coincide
    # (e.g. two same-size masked subsets of the same run). Storing d.step
    # itself alongside the cached result and requiring `is` identity on hit
    # makes a wrong hit structurally impossible: the cached steps are only
    # ever reused for the EXACT step array they were computed from.
    if _ckpt_hit is not None and _ckpt_hit[0] is d.step:
        steps=_ckpt_hit[1]
    else:
        steps=checkpoint_steps_from_data(d.step[finite_global],_n_timepoints)
        _ckpt_cache[_ckpt_cache_key]=(d.step,steps)
    if steps.size<2:
        return {'enabled':False,'metric':metric_name,'reason':'not enough distinct production steps for convergence testing'}
    ref_prob=pmf_probability(final_pmf)
    ref_F=np.asarray(final_pmf['pmf'],dtype=np.float64)
    ref_counts=np.asarray(final_pmf.get('counts',np.zeros_like(ref_prob)),dtype=float)
    ref_occ=int(np.count_nonzero(ref_counts>0))
    conv_rows=[]; pmf_rows=[]; prev_prob=None; prev_F=None; prev_counts=None
    total_samples=int(np.count_nonzero(finite_global))
    kbt_kcal=(1.0/d.beta)/KJ_PER_KCAL
    # Seed first checkpoint with full-run f_k; subsequent checkpoints reuse the
    # previous prefix solution. This makes the cheaper convergence SAMBAR warm-start
    # a nudge rather than a full 50-epoch expedition every time.
    prev_f_k=np.asarray(f_init_hint, dtype=np.float64) if f_init_hint is not None else None
    conv_backend=str(getattr(args,'convergence_mbar_backend','sambar') or 'sambar')
    conv_sambar_epochs=int(getattr(args,'convergence_sambar_epochs',10) or 10)
    conv_sambar_batch=int(getattr(args,'convergence_sambar_initial_batch_size',SAMBAR_INITIAL_BATCH_SIZE) or SAMBAR_INITIAL_BATCH_SIZE)
    conv_sambar_patience=int(getattr(args,'convergence_sambar_batch_patience',2) or 2)
    conv_sambar_seed=int(getattr(args,'convergence_sambar_seed',SAMBAR_SEED) or SAMBAR_SEED)
    conv_sambar_lr=float(getattr(args,'convergence_sambar_lr_scale',SAMBAR_LR_SCALE) or SAMBAR_LR_SCALE)
    conv_sambar_delta=float(getattr(args,'convergence_sambar_delta_f_max',SAMBAR_DELTA_F_MAX) or SAMBAR_DELTA_F_MAX)
    conv_sambar_polish=str(getattr(args,'convergence_sambar_polish_backend',SAMBAR_POLISH_BACKEND) or SAMBAR_POLISH_BACKEND)
    conv_mbar_maxiter=int(getattr(args,'convergence_mbar_maxiter',2000) or 2000)
    cache_enabled=not bool(getattr(args,'no_convergence_mbar_cache',False))
    mbar_cache=_get_convergence_mbar_cache(args) if cache_enabled else {}
    mbar_cache_hits=0
    mbar_cache_misses=0
    if progress is not None:
        cache_note='cache on' if cache_enabled else 'cache off'
        progress.step(f'{metric_name} convergence', f'{steps.size} prefix timepoints; backend={conv_backend}; MBAR tol={float(args.convergence_mbar_tol):.1e}; SAMBAR epochs={conv_sambar_epochs if conv_backend == "sambar" else "n/a"}; {cache_note}')
    for ci,ck in enumerate(steps, start=1):
        mask=(d.step<=ck)&finite_global
        n=int(np.count_nonzero(mask))
        if n<max(2,d.u_nk.shape[1]):
            continue
        if progress is not None:
            progress.bar(f'{metric_name} convergence',ci,steps.size,f'step {int(ck)} samples {n}')
        sub_u=d.u_nk[mask]
        sub_w=d.window[mask]
        sub_x=values[mask]
        sub_boost=d.boost_kj[mask]
        try:
            cache_hit=False
            cache_key=None
            if cache_enabled:
                cache_key=_convergence_mbar_cache_key(d,args,mask,int(ck),conv_backend,conv_mbar_maxiter,
                                                       conv_sambar_epochs,conv_sambar_batch,conv_sambar_patience,
                                                       conv_sambar_seed,conv_sambar_lr,conv_sambar_delta,conv_sambar_polish)
                mb=mbar_cache.get(cache_key)
                cache_hit=mb is not None
            else:
                mb=None
            if mb is None:
                mb=solve_mbar(sub_u,sub_w,tol=float(args.convergence_mbar_tol),maxiter=conv_mbar_maxiter,progress=None,backend=conv_backend,threads=getattr(args,'mbar_threads',0),f_init=prev_f_k,
                              sambar_epochs=conv_sambar_epochs,
                              sambar_initial_batch_size=conv_sambar_batch,
                              sambar_batch_patience=conv_sambar_patience,
                              sambar_seed=conv_sambar_seed,
                              sambar_lr_scale=conv_sambar_lr,
                              sambar_delta_f_max=conv_sambar_delta,
                              sambar_polish_backend=conv_sambar_polish)
                if cache_enabled and cache_key is not None:
                    mbar_cache[cache_key]=mb
                    mbar_cache_misses+=1
            else:
                mbar_cache_hits+=1
            pmf,_diag,used_method=_observable_pmf_from_logw(sub_x,mb['logw'],sub_boost,bins,selected,d.beta,kbt_kcal,smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
            prob=pmf_probability(pmf); F=np.asarray(pmf['pmf'],dtype=float); counts=np.asarray(pmf['counts'],dtype=float)
            delta_js=js_divergence_1d(prob,prev_prob) if prev_prob is not None else np.nan
            delta_rmse=pmf_rmse_1d(F,prev_F,prob,prev_prob) if prev_F is not None else np.nan
            new_bins=int(max(0,np.count_nonzero(counts>0)-(np.count_nonzero(prev_counts>0) if prev_counts is not None else 0)))
            row={
                'variant':'current','metric':metric_name,'selected_method':used_method,'checkpoint_index':ci,
                'checkpoint_step':int(ck),'n_samples':n,'n_samples_total':total_samples,
                'frac_total':float(n/max(1,total_samples)),
                'JS':js_divergence_1d(prob,ref_prob),
                'RMSE_F_kcal_mol':pmf_rmse_1d(F,ref_F,prob,ref_prob,float(getattr(args,'rmse_min_prob',0.0) if hasattr(args,'rmse_min_prob') else 0.0)),
                'barrier_error_kcal_mol':barrier_error_1d(F,ref_F,prob,ref_prob),
                'delta_JS':delta_js,
                'delta_RMSE_F_kcal_mol':delta_rmse,
                'occupied_bins':int(np.count_nonzero(counts>0)),
                'occupied_bins_ref':ref_occ,
                'occupied_bins_frac_ref':float(np.count_nonzero(counts>0)/max(1,ref_occ)),
                'new_bins_discovered':new_bins,
                'mbar_converged':int(bool(mb.get('converged'))),
                'mbar_iterations':int(mb.get('iterations',0)),
                'mbar_max_delta':float(mb.get('max_delta',np.nan)),
                'mbar_backend':str(mb.get('backend','unknown')),
                'mbar_requested_backend':conv_backend,
                'sambar_warmstart_epochs':int(mb.get('sambar_warmstart_epochs',0) or 0),
                'sambar_polish_backend':str(mb.get('sambar_polish_backend','')),
                'mbar_cache_hit':int(bool(cache_hit)),
            }
            conv_rows.append(row)
            pmf_rows.append({'checkpoint_index':ci,'checkpoint_step':int(ck),'frac_total':row['frac_total'],'x':pmf['cv_A'].copy(),'pmf_kcal_mol':F.copy(),'probability':prob.copy(),'counts':counts.astype(int).copy()})
            prev_prob=prob; prev_F=F; prev_counts=counts; prev_f_k=mb['f_k']
        except Exception as exc:
            conv_rows.append({'variant':'current','metric':metric_name,'checkpoint_index':ci,'checkpoint_step':int(ck),'n_samples':n,'n_samples_total':total_samples,'frac_total':float(n/max(1,total_samples)),'error':str(exc)})
    if progress is not None:
        progress.bar(f'{metric_name} convergence',1,1,'writing convergence outputs',force=True)

    conv_name='total_pmf_convergence.csv' if legacy_total_names else f'{file_prefix}_pmf_convergence.csv'
    by_checkpoint_name='total_pmf_by_checkpoint.csv' if legacy_total_names else f'{file_prefix}_pmf_by_checkpoint.csv'
    final_name='total_pmf.csv' if legacy_total_names else f'{file_prefix}_pmf.csv'
    summary_name='total_pmf_summary.csv' if legacy_total_names else f'{file_prefix}_pmf_summary.csv'
    variant_summary_name='total_pmf_convergence_variant_summary.csv' if legacy_total_names else f'{file_prefix}_pmf_convergence_variant_summary.csv'
    report_name='total_pmf_convergence_report.md' if legacy_total_names else f'{file_prefix}_pmf_convergence_report.md'

    _write_csv_rows(out/conv_name,conv_rows)
    flat=[]
    for p in pmf_rows:
        for i,x in enumerate(p['x']):
            flat.append({'variant':'current','metric':metric_name,'checkpoint_index':p['checkpoint_index'],'checkpoint_step':p['checkpoint_step'],'frac_total':p['frac_total'],'bin':i,'x':float(x),'F_kcal_mol':float(p['pmf_kcal_mol'][i]) if np.isfinite(p['pmf_kcal_mol'][i]) else '', 'P':float(p['probability'][i]), 'count':int(p['counts'][i])})
    _write_csv_rows(out/by_checkpoint_name,flat)
    final_rows=[]
    for i,x in enumerate(final_pmf['cv_A']):
        final_rows.append({'variant':'current','metric':metric_name,'x':float(x),'F_kcal_mol':float(ref_F[i]) if np.isfinite(ref_F[i]) else '', 'P':float(ref_prob[i]), 'count':int(ref_counts[i]) if i<ref_counts.size else 0})
    _write_csv_rows(out/final_name,final_rows)

    finite_rows=[r for r in conv_rows if 'JS' in r and np.isfinite(float(r.get('JS',np.nan)))]
    summary={}
    if finite_rows:
        js_thr=float(args.convergence_js_threshold); rmse_thr=float(args.convergence_rmse_threshold)
        arr_js=np.asarray([r['JS'] for r in finite_rows],float); arr_rmse=np.asarray([r['RMSE_F_kcal_mol'] for r in finite_rows],float)
        arr_djs=np.asarray([r.get('delta_JS',np.nan) for r in finite_rows],float); arr_drmse=np.asarray([r.get('delta_RMSE_F_kcal_mol',np.nan) for r in finite_rows],float)
        frac=np.asarray([r['frac_total'] for r in finite_rows],float)
        tail=frac>=(0.75 if len(finite_rows)>=15 else 0.5)
        if not np.any(tail): tail=np.ones_like(frac,dtype=bool)
        def first_frac(mask):
            idx=np.where(mask)[0]
            return float(frac[idx[0]]) if idx.size else np.nan
        summary={'variant':'current','metric':metric_name,'n_checkpoints':len(finite_rows),'js_threshold':js_thr,'rmse_threshold_kcal_mol':rmse_thr,
                 'first_frac_JS_lt_threshold':first_frac(arr_js<js_thr),'first_frac_RMSE_lt_threshold':first_frac(arr_rmse<rmse_thr),
                 'first_frac_delta_JS_lt_threshold':first_frac(arr_djs<js_thr*0.1),'first_frac_delta_RMSE_lt_threshold':first_frac(arr_drmse<rmse_thr*0.1),
                 'tail_median_JS':float(np.nanmedian(arr_js[tail])),'tail_median_RMSE_F_kcal_mol':float(np.nanmedian(arr_rmse[tail])),
                 'tail_median_delta_JS':float(np.nanmedian(arr_djs[tail])),'tail_median_delta_RMSE_F_kcal_mol':float(np.nanmedian(arr_drmse[tail])),
                 'last_eval_JS':float(arr_js[-1]),'last_eval_RMSE_F_kcal_mol':float(arr_rmse[-1]),
                 'last_eval_delta_JS':float(arr_djs[-1]) if np.isfinite(arr_djs[-1]) else np.nan,
                 'last_eval_delta_RMSE_F_kcal_mol':float(arr_drmse[-1]) if np.isfinite(arr_drmse[-1]) else np.nan}
        summary['converged_bool']=int(summary['tail_median_JS']<js_thr and summary['last_eval_JS']<js_thr and summary['tail_median_RMSE_F_kcal_mol']<rmse_thr and summary['last_eval_RMSE_F_kcal_mol']<rmse_thr)
    summary_rows=[summary] if summary else []
    _write_csv_rows(out/summary_name,summary_rows)
    _write_csv_rows(out/variant_summary_name,summary_rows)
    cwarnings=[]
    if finite_rows and not summary.get('converged_bool'):
        cwarnings.append(f'{metric_label} PMF convergence thresholds were not both satisfied in the tail and last checkpoint.')
        _first_rmse_ok=summary.get('first_frac_RMSE_lt_threshold',float('nan'))
        _last_n=finite_rows[-1].get('n_samples_total') or finite_rows[-1].get('n_samples')
        if np.isfinite(_first_rmse_ok) and float(_first_rmse_ok)>0 and _last_n:
            _ext_frac=max(0.0,(1.0/float(_first_rmse_ok))-1.0)
            _ext_n=int(round(_ext_frac*float(_last_n)))
            cwarnings.append(
                f'Extension estimate (linear): ~{_ext_frac*100:.0f}% more samples '
                f'(~{_ext_n:,} additional) based on RMSE first crossing threshold at '
                f'{float(_first_rmse_ok)*100:.1f}% of data. Actual requirement may differ.'
            )
    # No epoch/source boundary markers here: this cost-axis convergence accumulates
    # by raw production step (mask = d.step <= ck), and adaptive sources share the
    # same step_min, so they interleave — there is no single x at which "epoch_000
    # ends".  Source-annotated convergence lives in epoch_convergence/ (chronological
    # source-cumulative x-axis); see run_epoch_pmf_convergence.
    ea = []
    _fracs_ob = np.asarray([r.get('frac_total', np.nan) for r in conv_rows], dtype=float)
    _ts_ob = float(d.meta.get('timestep_fs', 4.0) or 4.0)
    plot_paths=write_observable_convergence_plots(conv_rows,pmf_rows,summary_rows,out,args,cwarnings,prefix=file_prefix,metric_label=metric_label,x_label=x_label,smooth_sigma=_eff_smooth(args,'pmf_smooth_sigma'),epoch_annotations=ea,aggregate_ns=_aggregate_ns_for_fracs(d,_fracs_ob,_ts_ob))
    if summary:
        write_observable_convergence_report(out/report_name,summary,conv_rows,cwarnings,metric_label)
    basin_result: dict = {}
    if basin_tracking and not bool(getattr(args,'no_basin_tracking',False)) and pmf_rows:
        min_depth=float(getattr(args,'basin_min_depth_kcal',0.5))
        basins=identify_basins_1d(np.asarray(final_pmf['cv_A'],dtype=float),ref_F,min_depth_kcal=min_depth)
        if not basins:
            cwarnings.append(f'No basins identified in final {metric_label} PMF with min_depth={min_depth} kcal/mol.')
        elif len(basins)==1:
            cwarnings.append(f'Only one basin identified in final {metric_label} PMF; population is trivially 1.')
        basin_pop_rows=[]
        for prow in pmf_rows:
            pops=_compute_basin_populations(prow['probability'],basins)
            for b,pop in zip(basins,pops):
                basin_pop_rows.append({'basin_id':b['basin_id'],'center_cv_A':b['center_cv_A'],'checkpoint_index':prow['checkpoint_index'],'checkpoint_step':prow['checkpoint_step'],'frac_total':prow['frac_total'],'population':float(pop)})
        _write_csv_rows(out/f'{file_prefix}_basin_populations.csv',basin_pop_rows)
        basin_def_rows=[{k:v for k,v in b.items() if k not in ('left_bin','right_bin')} for b in basins]
        _write_csv_rows(out/f'{file_prefix}_basin_definitions.csv',basin_def_rows)
        bplot_paths=_plot_basin_population_convergence(basin_pop_rows,basins,final_pmf,out,file_prefix,x_label,cwarnings,smooth_sigma=_eff_smooth(args,'pmf_smooth_sigma'))
        bfiles={f'{file_prefix}_basin_definitions_csv':str(out/f'{file_prefix}_basin_definitions.csv'),f'{file_prefix}_basin_populations_csv':str(out/f'{file_prefix}_basin_populations.csv')}
        if len(bplot_paths)>0: bfiles[f'{file_prefix}_basin_populations_png']=bplot_paths[0]
        if len(bplot_paths)>1: bfiles[f'{file_prefix}_basin_map_png']=bplot_paths[1]
        basin_result={'enabled':True,'n_basins':len(basins),'min_depth_kcal':min_depth,'basins':basin_def_rows,'files':bfiles}
    files={'convergence_dir':str(out),f'{file_prefix}_pmf_convergence_csv':str(out/conv_name),f'{file_prefix}_pmf_summary_csv':str(out/summary_name),f'{file_prefix}_pmf_convergence_report_md':str(out/report_name),f'{file_prefix}_convergence_plot_index':str(out/'plot_index.txt')}
    if legacy_total_names:
        files={'convergence_dir':str(out),'total_pmf_convergence_csv':str(out/conv_name),'total_pmf_summary_csv':str(out/summary_name),'total_pmf_convergence_report_md':str(out/report_name),'convergence_plot_index':str(out/'plot_index.txt')}
    if basin_result.get('files'): files.update(basin_result['files'])
    return {'enabled':True,'metric':metric_name,'output_dir':str(out),'n_timepoints_requested':int(args.convergence_timepoints),'n_timepoints_analyzed':len(finite_rows),'mbar_tolerance':float(args.convergence_mbar_tol),'mbar_backend':conv_backend,'sambar_warmstart_epochs':conv_sambar_epochs if conv_backend == 'sambar' else 0,'mbar_cache_hits':int(mbar_cache_hits),'mbar_cache_misses':int(mbar_cache_misses),'summary':summary,'warnings':cwarnings,'basin_tracking':basin_result,'files':files}


def run_pmf_convergence(d: Data, args, bins: np.ndarray, selected: str, final_pmf: dict, out_base: Path, progress: Optional[Progress]=None, f_init_hint: Optional[np.ndarray]=None) -> dict:
    return run_observable_pmf_convergence(
        d,args,d.cv,bins,selected,final_pmf,out_base,
        metric_name='cv_distance',metric_label='CV distance',x_label=_primary_cv_axis_label(d.meta),
        out_dir_name=str(getattr(args,'convergence_dir','convergence')),file_prefix='total',
        legacy_total_names=True,basin_tracking=True,progress=progress,f_init_hint=f_init_hint,
    )

def _find_first_existing_path(candidates) -> Optional[Path]:
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p
    return None


def _find_explicit_topology_path(args, names) -> Optional[Path]:
    for name in names:
        value = getattr(args, name, None)
        if value:
            p = Path(value)
            if p.exists():
                return p
    return None


def _find_default_topology_path(prod: Path) -> Optional[Path]:
    candidates=[
        prod/'solute_only.pdb', prod.parent/'solute_only.pdb',
        prod/'03_npt_equilibrated.pdb', prod.parent/'03_npt_equilibrated.pdb',
        prod/'shared_gamd_setup_final.pdb', prod.parent/'shared_gamd_setup_final.pdb',
        prod/'01_solvated_start.pdb', prod.parent/'01_solvated_start.pdb',
        prod/'02_minimized.pdb', prod.parent/'02_minimized.pdb',
    ]
    return _find_first_existing_path(candidates)


def _find_trajectory_companion_topology_path(run_dirs) -> Optional[Path]:
    candidates = []
    for run_dir in run_dirs or []:
        base = Path(run_dir)
        candidates.append(base/'solute_only.pdb')
        candidates.append(base.parent/'solute_only.pdb')
    return _find_first_existing_path(candidates)


def _find_data_topology_path(d: Data, args, names) -> Optional[Path]:
    explicit = _find_explicit_topology_path(args, names)
    if explicit is not None:
        return explicit
    companion = _find_trajectory_companion_topology_path(d.meta.get('adaptive_epoch_run_dirs', []))
    if companion is not None:
        return companion
    return _find_default_topology_path(d.prod_dir)


def _find_rg_topology_path(prod: Path, args) -> Optional[Path]:
    explicit = _find_explicit_topology_path(args, ('rg_topology',))
    if explicit is not None:
        return explicit
    return _find_default_topology_path(prod)



def _trajectory_extensions(args=None) -> list[str]:
    """Return trajectory suffixes to try for replica_trajectories files.

    The production script can write compressed XTC trajectories via
    --traj-format xtc, while older runs wrote DCD.  Keep analysis format-agnostic
    and prefer explicit user choice when provided.
    """
    fmt = str(getattr(args, 'trajectory_format', 'auto') or 'auto').strip().lower()
    aliases = {
        'auto': ['.xtc', '.dcd'],
        'xtc': ['.xtc'],
        'dcd': ['.dcd'],
    }
    return aliases.get(fmt, ['.xtc', '.dcd'])


def _find_replica_trajectory(traj_dir: Path, rep: int, args=None) -> Optional[Path]:
    """Find the trajectory file for one replica, supporting XTC and DCD outputs."""
    rep = int(rep)
    names = []
    for suffix in _trajectory_extensions(args):
        names.extend([
            f'replica_{rep:03d}{suffix}',
            f'replica_{rep}{suffix}',
            f'r{rep:03d}{suffix}',
            f'r{rep}{suffix}',
        ])
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        path = Path(traj_dir) / name
        if path.exists():
            return path
    return None


def _trajectory_pattern_hint(args=None) -> str:
    exts = ', '.join(_trajectory_extensions(args))
    return f'replica_### trajectory ({exts})'


def _find_all_replica_trajectory_segments(traj_dir: Path, rep: int, args=None) -> list:
    """Return [(resume_start_step, path), ...] for all trajectory segments, sorted chronologically.

    Each OpenMM resume writes a fresh XTC whose internal time resets to 0.2 ps.
    The resume start step is encoded in the filename as *_resume_from_NNNNNNNN.xtc.
    The base file (no _resume_from_) has resume_start=0.
    """
    rep = int(rep)
    result: list = []
    for suffix in _trajectory_extensions(args):
        base_path = None
        for name in [f'replica_{rep:03d}{suffix}', f'replica_{rep}{suffix}',
                     f'r{rep:03d}{suffix}', f'r{rep}{suffix}']:
            p = Path(traj_dir) / name
            if p.exists():
                base_path = p
                break
        resume_re = re.compile(
            rf'^(?:replica_{rep:03d}|r{rep:03d})_resume_from_(\d+){re.escape(suffix)}$'
        )
        resume_segs: list = []
        try:
            for f in sorted(Path(traj_dir).iterdir()):
                m = resume_re.match(f.name)
                if m:
                    resume_segs.append((int(m.group(1)), f))
        except OSError:
            pass
        resume_segs.sort(key=lambda x: x[0])
        segs = ([(0, base_path)] if base_path else []) + resume_segs
        if segs:
            result = segs
            break
    return result


def _read_traj_interval(prod_dir: Path) -> int:
    """Return traj_interval (simulation steps per trajectory frame) from effective_config, default 50."""
    for name in ('effective_config.json', 'run_args.json'):
        p = Path(prod_dir) / name
        if not p.exists():
            continue
        try:
            cfg = json.loads(p.read_text())
            for loc in (cfg, cfg.get('args', {}), cfg.get('output', {})):
                if isinstance(loc, dict) and loc.get('traj_interval'):
                    return int(loc['traj_interval'])
        except Exception:
            pass
    # YAML fallback
    p = Path(prod_dir) / 'effective_config.yaml'
    if p.exists():
        try:
            import yaml as _yaml
            cfg = _yaml.safe_load(p.read_text())
            for loc in (cfg, cfg.get('args', {}), cfg.get('output', {})):
                if isinstance(loc, dict) and loc.get('traj_interval'):
                    return int(loc['traj_interval'])
        except Exception:
            pass
    return 50


def _sample_to_segment_frame(
    sample_steps: np.ndarray,
    resume_start: int,
    n_frames: int,
    step_per_frame: int,
) -> tuple:
    """Map sample absolute steps to local frame indices within one trajectory segment.

    Frame i in a segment with resume_start R and step_per_frame S has absolute step:
        abs_step(i) = R + (i + 1) * S

    Returns (mask, local_frame_indices) where mask selects samples that fall in this segment.
    """
    spf = max(1, int(step_per_frame))
    seg_first = resume_start + spf
    seg_last = resume_start + n_frames * spf
    mask = (sample_steps >= seg_first) & (sample_steps <= seg_last)
    if not np.any(mask):
        return mask, np.empty(0, dtype=np.int64)
    local = (np.round((sample_steps[mask].astype(np.float64) - resume_start) / spf)
             .astype(np.int64) - 1)
    local = np.clip(local, 0, n_frames - 1)
    return mask, local


def _base_segment_resume_start(resume_start: int, use_adj: bool, sample_steps, spf: int) -> int:
    """Anchor a non-merged trajectory segment to the absolute sample-step clock.

    GaMD runs count the (cMD + equilibration) phase in the absolute sample
    ``step`` column (``absolute_step = calib_steps + prod_done``), but the
    production trajectory's frame 0 corresponds to the first *production* step.
    The base segment is recorded with ``resume_start=0`` and resume segments are
    recorded with the *production-relative* ``prod_done`` in their filename
    (``replica_NNN_resume_from_<prod_done>``) -- neither carries the
    ``calib_steps`` offset. Without correction, every absolute sample step falls
    outside a segment's [resume_start+spf, resume_start+n*spf] window and aligns
    to nothing (empty Rg / phi-psi / SASA output for GaMD runs).

    Infer the calibration offset from the data: the first production sample sits
    at ``calib_steps + spf``, so ``calib_steps = min(step) - spf`` (the global
    min is drawn from the base segment, which is sorted first and always present
    when any segment exists). The effective anchor for ANY segment is then
    ``calib_steps + resume_start``: for the base segment (resume_start=0) this is
    just ``calib_steps``; for a resume segment it shifts the production-relative
    filename offset into absolute coordinates.

    No-op where it must be:
      * merged/adaptive trajectory dirs (use_adj=True) -> unchanged,
      * cMD/hmr-cmd runs whose steps are already production-relative
        (calib_steps inferred <= 0) -> unchanged.

    ASSUMPTION: the sample cadence equals the trajectory frame interval
    (``distance_output_interval == traj_interval``). All shipped configs satisfy
    this. If they differ, ``min(step) - spf`` carries an extra
    ``(distance_output_interval - spf)`` term and frames misalign; keep the two
    intervals equal in any GaMD config whose trajectory observables matter.
    """
    if use_adj:
        return int(resume_start)
    s = np.asarray(sample_steps, dtype=np.float64)
    if s.size == 0:
        return int(resume_start)
    calib = int(np.min(s)) - int(spf)
    if calib <= 0:
        return int(resume_start)
    return calib + int(resume_start)


def _saved_frame_index_from_step(step: int, resume_start: int, step_per_frame: int) -> int:
    """Return local trajectory frame index for an absolute saved-frame step."""
    spf = max(1, int(step_per_frame))
    delta = int(step) - int(resume_start)
    if delta <= 0:
        return -1
    return int(round(float(delta) / float(spf))) - 1


def _sample_aligned_trajectory_frames(
    sample_indices: np.ndarray,
    sample_steps: np.ndarray,
    n_frames: int,
    *,
    label: str,
    allow_mismatch: bool,
    warnings: list[str],
    strict_hint: str,
) -> Optional[tuple[np.ndarray, np.ndarray, str]]:
    """Map analysis sample rows to trajectory frame indices.

    Production can write trajectories more frequently than samples.csv/distances.csv
    are logged, e.g. XTC every 50 steps but samples every 500 steps.  The old
    analyzer simply truncated 60000 trajectory frames to ~10000 samples, which
    mapped the wrong coordinates to MBAR weights.  Here we keep all sample rows
    and choose the corresponding trajectory frame by monotonic production step.

    Returns (sample_indices_kept, frame_indices_kept, mode).
    """
    sample_indices=np.asarray(sample_indices,dtype=np.int64)
    sample_steps=np.asarray(sample_steps,dtype=np.float64)
    n_frames=int(n_frames or 0)
    if sample_indices.size == 0 or n_frames <= 0:
        return None
    order=np.argsort(sample_steps, kind='stable')
    sample_indices=sample_indices[order]
    sample_steps=sample_steps[order]
    finite=np.isfinite(sample_steps)
    if not np.all(finite):
        sample_indices=sample_indices[finite]
        sample_steps=sample_steps[finite]
    n_samples=int(sample_indices.size)
    if n_samples <= 0:
        return None
    if n_frames == n_samples:
        return sample_indices, np.arange(n_samples,dtype=np.int64), 'one_to_one'

    msg=f'{label}: {n_frames} trajectory frames vs {n_samples} samples'
    if not allow_mismatch:
        warnings.append(msg + f'; skipped. Set {strict_hint} to align frames by sample step.')
        return None

    if n_samples == 1:
        frame_idx=np.asarray([0],dtype=np.int64)
        warnings.append(msg + '; assigned the first trajectory frame to the only sample.')
        return sample_indices, frame_idx, 'single_sample_first_frame'

    s0=float(sample_steps[0]); s1=float(sample_steps[-1])
    if not np.isfinite(s0) or not np.isfinite(s1) or s1 <= s0:
        # Last-resort proportional mapping by row number.  This is still better
        # than truncating the first chunk of a dense XTC trajectory.
        frame_idx=np.rint(np.linspace(0,n_frames-1,n_samples)).astype(np.int64)
        frame_idx=np.clip(frame_idx,0,n_frames-1)
        warnings.append(msg + '; aligned by row index across the full trajectory span because sample steps were not usable.')
        return sample_indices, frame_idx, 'row_span'

    frac=(sample_steps-s0)/(s1-s0)
    frame_idx=np.rint(frac*(n_frames-1)).astype(np.int64)
    frame_idx=np.clip(frame_idx,0,n_frames-1)
    # Preserve sample order, but make duplicate-frame mappings explicit.
    unique_frames=int(np.unique(frame_idx).size)
    warnings.append(msg + f'; aligned {n_samples} samples to {unique_frames} trajectory frames by sample step across the full trajectory span.')
    return sample_indices, frame_idx, 'step_span'


def _chunk_local_frame_selection(frame_indices: np.ndarray, sample_indices: np.ndarray, frame0: int, n_chunk: int):
    """Return local chunk frame indices and corresponding sample rows."""
    frame_indices=np.asarray(frame_indices,dtype=np.int64)
    sample_indices=np.asarray(sample_indices,dtype=np.int64)
    lo=int(frame0); hi=int(frame0+n_chunk)
    m=(frame_indices>=lo)&(frame_indices<hi)
    if not np.any(m):
        return None, None, None
    local=(frame_indices[m]-lo).astype(np.int64,copy=False)
    return local, sample_indices[m], frame_indices[m]

def _rg_from_segment_chunked(md, seg_path, top, atom_indices, selection: str, chunk_size: int):
    """Compute per-frame Rg (nm) for one trajectory segment via chunked iterload.

    Radius of gyration is a purely per-frame quantity with no cross-frame
    dependency, so there is no need to hold an entire multi-million-frame
    segment in memory at once just to run `md.compute_rg` over it. This reads
    and processes `chunk_size` frames at a time instead, concatenating the
    per-chunk Rg results into the same full-segment array (same order/values)
    a whole-segment `md.load()` + `md.compute_rg()` would produce.

    Returns (rg_nm, n_frames) where `n_frames` is the segment's total frame
    count (sum of per-chunk frame counts), needed by the caller for the same
    sample-to-frame alignment math used before this change.
    """
    rg_chunks: list = []
    n_frames = 0
    if atom_indices is not None:
        for chunk in md.iterload(str(seg_path), top=top, chunk=chunk_size, atom_indices=atom_indices):
            rg_chunks.append(md.compute_rg(chunk))
            n_frames += chunk.n_frames
    else:
        atoms_cache = None
        for chunk in md.iterload(str(seg_path), top=top, chunk=chunk_size):
            if atoms_cache is None:
                atoms_cache = chunk.topology.select(selection)
                if atoms_cache.size == 0:
                    raise ValueError(f'selection {selection!r} matched zero atoms')
            rg_chunks.append(md.compute_rg(chunk.atom_slice(atoms_cache)))
            n_frames += chunk.n_frames
    if not rg_chunks:
        raise ValueError(f'trajectory segment {seg_path} contained zero frames')
    return np.concatenate(rg_chunks), n_frames

def _compute_rg_from_trajectories(d: Data, args, progress: Optional[Progress], warnings: list[str]) -> Optional[np.ndarray]:
    mode=str(getattr(args,'rg_from_trajectories','auto') or 'auto').lower()
    if mode == 'never':
        return None
    traj_dir=d.prod_dir/'replica_trajectories'
    if not traj_dir.exists():
        merged = _prepare_adaptive_merged_traj_dir(d, args)
        if merged is not None:
            traj_dir = merged
        else:
            if mode == 'force': warnings.append(f'Rg trajectory reconstruction requested but {traj_dir} is missing')
            return None
    top_path=_find_data_topology_path(d, args, ('rg_topology',))
    if top_path is None:
        if mode == 'force': warnings.append('Rg trajectory reconstruction requested but no topology PDB was found; use --rg-topology')
        return None
    try:
        import mdtraj as md
    except Exception as exc:
        if mode == 'force' or mode == 'auto':
            warnings.append(f'Rg trajectory reconstruction skipped: mdtraj is unavailable ({exc})')
        return None
    out=np.full(d.cv.shape,np.nan,dtype=np.float64)
    reps=sorted(set(int(x) for x in d.replica if np.isfinite(x)))
    selection=str(getattr(args,'rg_selection','protein and element != H') or 'protein and element != H')
    allow_truncate=bool(getattr(args,'rg_allow_truncate',False))
    if progress is not None:
        progress.step('Rg trajectories', f'using topology {top_path}; selection {selection!r}')
    spf=_read_traj_interval(d.prod_dir)
    if spf <= 0 or spf == 500:
        epoch_spf = _read_traj_interval_from_epoch_dirs(d)
        if epoch_spf > 0:
            spf = epoch_spf
    use_adjusted = (traj_dir != d.prod_dir / 'replica_trajectories')
    adj_steps = _adjusted_steps_for_merged_traj(d, spf) if use_adjusted else None
    # Pre-load topology once — avoids re-parsing PDB on every md.load() call
    try:
        _top_obj = md.load(str(top_path))
        top_topology: object = _top_obj.topology
    except Exception:
        top_topology = str(top_path)
    # Pre-compute atom selection once so every worker/segment reuses it;
    # passing atom_indices to md.load tells the XTC codec to skip non-selected
    # atoms at the byte level — not just a memory filter.
    rg_atoms: object = None
    if hasattr(top_topology, 'select'):
        try:
            rg_atoms = top_topology.select(selection)
            if rg_atoms.size == 0:
                warnings.append(f'Rg atom selection {selection!r} matched zero atoms in topology; falling back to full load')
                rg_atoms = None
        except Exception as exc:
            warnings.append(f'Rg atom pre-selection failed ({exc}); falling back to full load')
            rg_atoms = None
    # Chunk size for streaming segment loads; reuses --pca-chunk-size (default 1000)
    # for consistency with the other chunked-iterload passes in this file (PCA,
    # extra-observable PMFs) rather than introducing a separate Rg-only flag.
    rg_chunk_size = max(1, int(getattr(args, 'pca_chunk_size', 1000) or 1000))

    def _rg_replica_worker(rep):
        local_warns: list = []
        segs=_find_all_replica_trajectory_segments(traj_dir, rep, args)
        if not segs:
            local_warns.append(f'Rg trajectory missing for replica {rep}: {_trajectory_pattern_hint(args)} in {traj_dir}')
            return 0, local_warns
        idx=np.where(d.replica==rep)[0]
        if idx.size==0:
            return 0, local_warns
        order=idx[np.argsort(d.step[idx], kind='stable')]
        steps_for_align = adj_steps[order] if adj_steps is not None else d.step[order]
        rep_assigned=0
        for resume_start, seg_path in segs:
            try:
                rg_nm, n_frames = _rg_from_segment_chunked(md, seg_path, top_topology, rg_atoms, selection, rg_chunk_size)
                rg = rg_nm * 10.0
            except Exception as exc:
                local_warns.append(f'Rg trajectory reconstruction failed for replica {rep} ({seg_path}): {exc}')
                continue
            eff_resume_start=_base_segment_resume_start(resume_start, use_adjusted, steps_for_align, spf)
            mask, local_frames=_sample_to_segment_frame(steps_for_align, eff_resume_start, n_frames, spf)
            if not np.any(mask):
                continue
            out[order[mask]]=rg[local_frames]
            rep_assigned+=int(np.sum(mask))
        return rep_assigned, local_warns

    n_workers=max(1, int(getattr(args,'traj_workers',4) or 4))
    assigned=0
    if n_workers > 1 and len(reps) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=n_workers) as exe:
            futs={exe.submit(_rg_replica_worker, rep): rep for rep in reps}
            for ii,fut in enumerate(as_completed(futs), start=1):
                rep=futs[fut]
                if progress is not None:
                    progress.bar('Rg trajectories', ii, max(1,len(reps)), f'replica {rep}')
                try:
                    n_a, w=fut.result()
                    assigned+=n_a; warnings.extend(w)
                except Exception as exc:
                    warnings.append(f'Rg worker replica {rep} failed: {exc}')
    else:
        for ii,rep in enumerate(reps, start=1):
            if progress is not None:
                progress.bar('Rg trajectories', ii, max(1,len(reps)), f'replica {rep}')
            n_a, w=_rg_replica_worker(rep)
            assigned+=n_a; warnings.extend(w)

    if progress is not None:
        progress.bar('Rg trajectories', 1, 1, f'assigned {assigned}/{d.cv.size} samples', force=True)
    if assigned <= 0:
        return None
    return out

def _weighted_mean_std(values, weights):
    v=np.asarray(values,dtype=np.float64); w=np.asarray(weights,dtype=np.float64)
    mask=np.isfinite(v)&np.isfinite(w)&(w>=0)
    if not np.any(mask): return float('nan'), float('nan')
    v=v[mask]; w=w[mask]
    sw=float(np.sum(w))
    if sw<=0: return float('nan'), float('nan')
    w=w/sw
    mean=float(np.sum(w*v)); var=float(np.sum(w*(v-mean)**2))
    return mean, float(math.sqrt(max(0.0,var)))

def _pmf_distribution_mean_std(pmf: dict):
    x=np.asarray(pmf.get('cv_A',[]),dtype=np.float64)
    p=pmf_probability(pmf)
    if x.size==0 or p.size==0: return float('nan'), float('nan')
    n=min(x.size,p.size); x=x[:n]; p=p[:n]
    mask=np.isfinite(x)&np.isfinite(p)&(p>=0)
    if not np.any(mask): return float('nan'), float('nan')
    x=x[mask]; p=p[mask]
    sp=float(np.sum(p))
    if sp<=0: return float('nan'), float('nan')
    p=p/sp
    mean=float(np.sum(p*x)); var=float(np.sum(p*(x-mean)**2))
    return mean, float(math.sqrt(max(0.0,var)))

def analyze_rg(d: Data, args, m: dict, base_w: np.ndarray, selected: str, boost_ok: bool, kbt_kcal: float, out: Path, warnings: list[str], progress: Optional[Progress]) -> dict:
    rg=np.asarray(d.rg_A,dtype=np.float64)
    initial_rg_count=int(np.count_nonzero(np.isfinite(rg)))
    rg_source='analysis_arrays_or_samples' if initial_rg_count > 0 else 'unavailable'
    if initial_rg_count == 0 and str(getattr(args,'rg_from_trajectories','auto')).lower() != 'never':
        rebuilt=_compute_rg_from_trajectories(d,args,progress,warnings)
        if rebuilt is not None:
            d.rg_A=rebuilt
            rg=rebuilt
            rg_source='trajectory_reconstruction' 
    mask=np.isfinite(rg)
    if np.count_nonzero(mask) < max(5, d.u_nk.shape[1]):
        return {'available':False,'reason':'Rg unavailable or too few finite Rg samples','n_finite':int(np.count_nonzero(mask))}
    if progress is not None: progress.bar('analysis stages', 4, 6, 'analyzing Rg', force=True)
    rg_bins = int(getattr(args,'rg_bins',None) or args.bins)
    bins=make_bins(rg[mask],rg_bins,getattr(args,'rg_min',None),getattr(args,'rg_max',None))
    base_logw=np.asarray(m['logw'],dtype=np.float64)[mask]
    base_w_rg=norm_logw(base_logw)
    rg_sel=rg[mask]
    boost_sel=d.boost_kj[mask]
    umbrella=pmf_from_weights(rg_sel,base_w_rg,bins,kbt_kcal)
    if boost_ok and np.isfinite(boost_sel).sum()>10 and np.nanstd(boost_sel)>1e-12:
        exp_logw=base_logw+d.beta*boost_sel
        exp_w=norm_logw(exp_logw)
        exp_pmf=pmf_from_weights(rg_sel,exp_w,bins,kbt_kcal)
        (cum_pmf,cdiag),(cum3_pmf,cdiag3)=_cumulant_expansion_both(rg_sel,base_w_rg,boost_sel,bins,d.beta,kbt_kcal,smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
        rg_selected=selected if selected in {'gamd_exponential','gamd_cumulant2','gamd_cumulant3'} else 'gamd_cumulant2'
    else:
        exp_w=base_w_rg
        exp_pmf=umbrella; cum_pmf=umbrella; cum3_pmf=umbrella; cdiag={'boost_mean_kj':np.full(len(bins)-1,np.nan),'boost_var_kj2':np.full(len(bins)-1,np.nan)}; cdiag3=cdiag; rg_selected='umbrella_only'
    rg_pmfs={'umbrella_only':umbrella,'gamd_exponential':exp_pmf,'gamd_cumulant2':cum_pmf,'gamd_cumulant3':cum3_pmf}
    sel=rg_pmfs.get(rg_selected, umbrella)
    finite=sel['pmf'][np.isfinite(sel['pmf'])]
    minidx=int(np.nanargmin(sel['pmf'])) if finite.size else -1
    mean_A,std_A=_pmf_distribution_mean_std(sel) if rg_selected in ('gamd_cumulant2','gamd_cumulant3') else _weighted_mean_std(rg_sel, exp_w if rg_selected=='gamd_exponential' else base_w_rg)
    write_rg_pmf(out/'rg_pmf_unbiased.csv',sel,rg_selected,{'boost_mean_kj_mol':cdiag.get('boost_mean_kj',np.full(len(bins)-1,np.nan)),'boost_var_kj2_mol2':cdiag.get('boost_var_kj2',np.full(len(bins)-1,np.nan))})
    write_rg_pmf(out/'rg_pmf_umbrella_only.csv',umbrella,'umbrella_only')
    write_rg_pmf(out/'rg_pmf_gamd_exponential.csv',exp_pmf,'gamd_exponential')
    write_rg_pmf(out/'rg_pmf_gamd_cumulant2.csv',cum_pmf,'gamd_cumulant2',{'boost_mean_kj_mol':cdiag.get('boost_mean_kj',np.full(len(bins)-1,np.nan)),'boost_var_kj2_mol2':cdiag.get('boost_var_kj2',np.full(len(bins)-1,np.nan))})
    write_rg_pmf(out/'rg_pmf_gamd_cumulant3.csv',cum3_pmf,'gamd_cumulant3',{'boost_mean_kj_mol':cdiag3.get('boost_mean_kj',np.full(len(bins)-1,np.nan)),'boost_var_kj2_mol2':cdiag3.get('boost_var_kj2',np.full(len(bins)-1,np.nan))})
    write_rg_all(out/'rg_pmf_all_methods.csv',rg_pmfs)
    sample_rows=[]
    base_w_full=np.full(d.cv.shape,np.nan); base_w_full[mask]=base_w_rg
    exp_w_full=np.full(d.cv.shape,np.nan); exp_w_full[mask]=exp_w
    for i in np.where(mask)[0]:
        sample_rows.append({'step':int(d.step[i]),'replica':int(d.replica[i]),'window':int(d.window[i]),'cv_A':float(d.cv[i]),'rg_A':float(rg[i]),'umbrella_mbar_weight':float(base_w_full[i]),'gamd_exponential_weight':float(exp_w_full[i]) if np.isfinite(exp_w_full[i]) else ''})
    _write_csv_rows(out/'rg_samples_with_weights.csv',sample_rows)
    plot_rg_outputs(d,rg_pmfs,rg_selected,out,warnings,smooth_sigma=_eff_smooth(args,'pmf_smooth_sigma'),args=args)
    span=float(np.max(finite)-np.min(finite)) if finite.size else float('nan')
    rg_conv=run_observable_pmf_convergence(
        d,args,rg,bins,rg_selected,sel,out,
        metric_name='rg',metric_label='Radius of gyration',x_label='Rg (A)',
        out_dir_name=str(getattr(args,'rg_convergence_dir','rg_convergence')),file_prefix='rg',
        legacy_total_names=False,basin_tracking=True,progress=progress,
    )
    rg_info={'available':True,'n_samples':int(np.count_nonzero(mask)),'source':rg_source,'selected_unbiased_method':rg_selected,'mean_A':float(mean_A),'std_A':float(std_A),'rg_min_sample_A':float(np.nanmin(rg[mask])),'rg_max_sample_A':float(np.nanmax(rg[mask])),'pmf_minimum_rg_A':float(sel['cv_A'][minidx]) if minidx>=0 else None,'pmf_span_kcal_mol':span,'convergence':rg_conv,'files':{'rg_pmf_unbiased_csv':str(out/'rg_pmf_unbiased.csv'),'rg_pmf_all_methods_csv':str(out/'rg_pmf_all_methods.csv'),'rg_samples_with_weights_csv':str(out/'rg_samples_with_weights.csv'),'rg_pmf_plot_png':str(out/'rg_pmf_all_methods.png'),'rg_vs_cv_sampled_png':str(out/'rg_vs_cv_sampled.png')}}
    if isinstance(rg_conv,dict) and rg_conv.get('files'):
        rg_info['files'].update({k:v for k,v in rg_conv.get('files',{}).items()})
    wjson(out/'rg_summary.json',rg_info)
    return rg_info


def analyze_distance_rg_2d_fes(d: Data, args, base_logw: np.ndarray, selected: str, boost_ok: bool, kbt_kcal: float, out: Path, warnings: list[str], progress: Optional[Progress]) -> dict:
    rg=np.asarray(d.rg_A, dtype=np.float64)
    mask=np.isfinite(d.cv) & np.isfinite(rg) & np.isfinite(base_logw)
    if np.count_nonzero(mask) < max(20, d.u_nk.shape[1]):
        return {'available': False, 'reason': 'Too few finite paired CV/Rg samples for 2D FES', 'n_finite': int(np.count_nonzero(mask))}
    if progress is not None:
        progress.bar('analysis stages', 4, 6, 'building distance-Rg 2D FES', force=True)
    cv_bins_n=int(getattr(args,'fes2d_cv_bins',None) or args.bins)
    rg_bins_n=int(getattr(args,'fes2d_rg_bins',None) or getattr(args,'rg_bins',None) or args.bins)
    xbins=make_bins(d.cv[mask], cv_bins_n, getattr(args,'cv_min',None), getattr(args,'cv_max',None))
    ybins=make_bins(rg[mask], rg_bins_n, getattr(args,'rg_min',None), getattr(args,'rg_max',None))
    cv_sel=d.cv[mask]
    rg_sel=rg[mask]
    boost_sel=d.boost_kj[mask]
    base_logw_sel=np.asarray(base_logw,dtype=np.float64)[mask]
    base_w=norm_logw(base_logw_sel)
    fes_umbrella=pmf2d_from_weights(cv_sel, rg_sel, base_w, xbins, ybins, kbt_kcal)
    if boost_ok and np.isfinite(boost_sel).sum()>10 and np.nanstd(boost_sel)>1e-12:
        exp_logw=base_logw_sel + d.beta*boost_sel
        exp_w=norm_logw(exp_logw)
        fes_exp=pmf2d_from_weights(cv_sel, rg_sel, exp_w, xbins, ybins, kbt_kcal)
        (fes_cum, cdiag), (fes_cum3, cdiag3) = _cumulant_expansion_2d_both(cv_sel, rg_sel, base_w, boost_sel, xbins, ybins, d.beta, kbt_kcal, smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
        chosen = selected if selected in {'gamd_exponential','gamd_cumulant2','gamd_cumulant3'} else 'gamd_cumulant2'
    else:
        fes_exp=fes_umbrella
        fes_cum=fes_umbrella
        fes_cum3=fes_umbrella
        cdiag={'boost_mean_kj':np.full((len(xbins)-1, len(ybins)-1), np.nan), 'boost_var_kj2':np.full((len(xbins)-1, len(ybins)-1), np.nan)}
        cdiag3=cdiag
        chosen='umbrella_only'
    fes_map={'umbrella_only': fes_umbrella, 'gamd_exponential': fes_exp, 'gamd_cumulant2': fes_cum, 'gamd_cumulant3': fes_cum3}
    chosen_fes=fes_map.get(chosen, fes_umbrella)
    write_2d_fes_csv(out/'distance_rg_2d_fes_selected.csv', chosen_fes, chosen)
    write_2d_fes_npz(out/'distance_rg_2d_fes_selected.npz', chosen_fes, chosen)
    plot_files=plot_2d_fes(chosen_fes, chosen, out/'distance_rg_2d_fes_selected.png', f'Distance vs Rg 2D free-energy surface ({chosen})', warnings, smooth_sigma=float(getattr(args,'fes2d_smooth_sigma',1.0)))
    write_2d_fes_csv(out/'distance_rg_2d_fes_cumulant2.csv', fes_cum, 'gamd_cumulant2')
    write_2d_fes_npz(out/'distance_rg_2d_fes_cumulant2.npz', fes_cum, 'gamd_cumulant2')
    plot_files_c2=plot_2d_fes(fes_cum, 'gamd_cumulant2', out/'distance_rg_2d_fes_cumulant2.png', 'Distance vs Rg 2D free-energy surface (gamd_cumulant2)', warnings, smooth_sigma=float(getattr(args,'fes2d_smooth_sigma',1.0)))
    want_cum3=_want_gamd_method('gamd_cumulant3', chosen, args)
    plot_files_c3={}
    if want_cum3:
        write_2d_fes_csv(out/'distance_rg_2d_fes_cumulant3.csv', fes_cum3, 'gamd_cumulant3')
        write_2d_fes_npz(out/'distance_rg_2d_fes_cumulant3.npz', fes_cum3, 'gamd_cumulant3')
        plot_files_c3=plot_2d_fes(fes_cum3, 'gamd_cumulant3', out/'distance_rg_2d_fes_cumulant3.png', 'Distance vs Rg 2D free-energy surface (gamd_cumulant3)', warnings, smooth_sigma=float(getattr(args,'fes2d_smooth_sigma',1.0)))
    finite=np.isfinite(chosen_fes['pmf'])
    if np.any(finite):
        min_idx=np.unravel_index(np.nanargmin(np.where(finite, chosen_fes['pmf'], np.inf)), chosen_fes['pmf'].shape)
        min_cv=float(chosen_fes['cv_A'][min_idx[0]])
        min_rg=float(chosen_fes['rg_A'][min_idx[1]])
        span=float(np.nanmax(chosen_fes['pmf'][finite]) - np.nanmin(chosen_fes['pmf'][finite]))
    else:
        min_cv=min_rg=span=float('nan')
    info={
        'available': True,
        'selected_unbiased_method': chosen,
        'n_samples': int(np.count_nonzero(mask)),
        'cv_bins': int(len(xbins)-1),
        'rg_bins': int(len(ybins)-1),
        'pmf_minimum_cv_A': min_cv,
        'pmf_minimum_rg_A': min_rg,
        'pmf_span_kcal_mol': span,
        'normalization': 'Minimum finite free energy shifted to 0 kcal/mol',
        'plot_ranges_kcal_mol': ['0-2','0-5','0-10','0-20','0-all'],
        'files': {
            'distance_rg_2d_fes_csv': str(out/'distance_rg_2d_fes_selected.csv'),
            'distance_rg_2d_fes_npz': str(out/'distance_rg_2d_fes_selected.npz'),
            'distance_rg_2d_fes_png': str(out/'distance_rg_2d_fes_selected.png'),
            **{f'distance_rg_2d_fes_png_{k}': v for k, v in (plot_files or {}).items()},
            'distance_rg_2d_fes_cumulant2_csv': str(out/'distance_rg_2d_fes_cumulant2.csv'),
            'distance_rg_2d_fes_cumulant2_npz': str(out/'distance_rg_2d_fes_cumulant2.npz'),
            'distance_rg_2d_fes_cumulant2_png': str(out/'distance_rg_2d_fes_cumulant2.png'),
            **{f'distance_rg_2d_fes_cumulant2_png_{k}': v for k, v in (plot_files_c2 or {}).items()},
            **({
                'distance_rg_2d_fes_cumulant3_csv': str(out/'distance_rg_2d_fes_cumulant3.csv'),
                'distance_rg_2d_fes_cumulant3_npz': str(out/'distance_rg_2d_fes_cumulant3.npz'),
                'distance_rg_2d_fes_cumulant3_png': str(out/'distance_rg_2d_fes_cumulant3.png'),
                **{f'distance_rg_2d_fes_cumulant3_png_{k}': v for k, v in (plot_files_c3 or {}).items()},
            } if want_cum3 else {}),
        },
    }
    wjson(out/'distance_rg_2d_fes_summary.json', info)
    return info


def _find_pca_topology_path(prod: Path, args) -> Optional[Path]:
    explicit = _find_explicit_topology_path(args, ('pca_topology', 'rg_topology'))
    if explicit is not None:
        return explicit
    return _find_default_topology_path(prod)

def _trajectory_frame_count(md, traj_path: Path) -> Optional[int]:
    try:
        fh=md.open(str(traj_path))
        try:
            n=getattr(fh,'n_frames',None)
            if n is None and hasattr(fh,'__len__'):
                n=len(fh)
            return int(n) if n is not None else None
        finally:
            try: fh.close()
            except Exception: pass
    except Exception:
        return None

def _pca_replica_plan(d: Data, args, md, traj_dir: Path, warnings: list[str]) -> list[dict]:
    reps=sorted(set(int(x) for x in d.replica if np.isfinite(x)))
    plan=[]
    spf=_read_traj_interval(d.prod_dir)
    if spf <= 0 or spf == 500:
        epoch_spf = _read_traj_interval_from_epoch_dirs(d)
        if epoch_spf > 0: spf = epoch_spf
    use_adj = (traj_dir != d.prod_dir / 'replica_trajectories')
    adj_steps_all = _adjusted_steps_for_merged_traj(d, spf) if use_adj else None
    for rep in reps:
        segs=_find_all_replica_trajectory_segments(traj_dir, rep, args)
        if not segs:
            warnings.append(f'PCA trajectory missing for replica {rep}: {_trajectory_pattern_hint(args)} in {traj_dir}')
            continue
        idx=np.where(d.replica==rep)[0]
        if idx.size==0:
            continue
        order=idx[np.argsort(d.step[idx], kind='stable')]
        steps_for_align = adj_steps_all[order] if adj_steps_all is not None else d.step[order]
        for resume_start, seg_path in segs:
            n_seg_frames=_trajectory_frame_count(md, seg_path)
            if n_seg_frames is None or n_seg_frames<=0:
                continue
            eff_resume_start=_base_segment_resume_start(resume_start, use_adj, steps_for_align, spf)
            mask, local_frames=_sample_to_segment_frame(steps_for_align, eff_resume_start, n_seg_frames, spf)
            if not np.any(mask):
                continue
            seg_order=order[mask]
            plan.append({'replica':rep,'traj':seg_path,'dcd':seg_path,'order':seg_order,
                         'sample_indices':seg_order,'frame_indices':local_frames,
                         'n_assign':int(seg_order.size),'n_frames':n_seg_frames,
                         'alignment_mode':'step_exact'})
    return plan

def _load_pca_reference(md, plan: list[dict], top_path: Path, selection: str):
    last_exc=None
    for item in plan:
        try:
            first=md.load_frame(str(item.get('traj', item['dcd'])), 0, top=str(top_path))
            atoms=first.topology.select(selection)
            if atoms.size == 0:
                raise ValueError(f'PCA selection {selection!r} matched zero atoms')
            ref=first.atom_slice(atoms)
            return ref, atoms, str(item.get('traj', item['dcd']))
        except Exception as exc:
            last_exc=exc
            continue
    raise RuntimeError(f'Could not load a reference frame for PCA: {last_exc}')

def _aligned_flattened_coords_A(chunk, atoms: np.ndarray, ref, pre_sliced: bool = False):
    sub=chunk if pre_sliced else chunk.atom_slice(atoms)
    sub.superpose(ref, frame=0)
    return sub.xyz.reshape((sub.n_frames, -1)).astype(np.float64, copy=False)*10.0

def _fit_and_project_pca_from_trajectories(d: Data, args, out: Path, progress: Optional[Progress], warnings: list[str]) -> dict:
    mode=str(getattr(args,'pca_fes_from_trajectories','auto') or 'auto').lower()
    if mode == 'never':
        return {'available':False,'reason':'disabled by --pca-fes-from-trajectories never'}
    cache=out/'pca_scores.npz'
    if cache.exists() and not bool(getattr(args,'pca_recompute',False)):
        try:
            z=np.load(cache, allow_pickle=False)
            p1=np.asarray(z['pca1'],dtype=np.float64)
            p2=np.asarray(z['pca2'],dtype=np.float64)
            if p1.shape == d.cv.shape and p2.shape == d.cv.shape:
                info=rjson(out/'pca_scores_metadata.json',{})
                info.update({'available':True,'source':'cache','pca1':p1,'pca2':p2,'files':{'pca_scores_npz':str(cache)}})
                return info
            warnings.append(f'Ignoring PCA cache with wrong shape: {cache}')
        except Exception as exc:
            warnings.append(f'Ignoring unreadable PCA cache {cache}: {exc}')
    traj_dir=d.prod_dir/'replica_trajectories'
    if not traj_dir.exists():
        merged = _prepare_adaptive_merged_traj_dir(d, args)
        if merged is not None:
            traj_dir = merged
        else:
            if mode == 'force': warnings.append(f'PCA requested but {traj_dir} is missing')
            return {'available':False,'reason':f'{traj_dir} is missing'}
    top_path=_find_data_topology_path(d, args, ('pca_topology', 'rg_topology'))
    if top_path is None:
        if mode == 'force': warnings.append('PCA requested but no topology PDB was found; use --pca-topology')
        return {'available':False,'reason':'no topology PDB found; use --pca-topology'}
    try:
        import mdtraj as md
    except Exception as exc:
        if mode in {'auto','force'}:
            warnings.append(f'PCA 2D FES skipped: mdtraj is unavailable ({exc})')
        return {'available':False,'reason':f'mdtraj unavailable: {exc}'}
    plan=_pca_replica_plan(d,args,md,traj_dir,warnings)
    if not plan:
        return {'available':False,'reason':'no usable replica trajectories for PCA'}
    selection=str(getattr(args,'pca_selection','protein and name CA') or 'protein and name CA')
    try:
        ref, atoms, ref_dcd=_load_pca_reference(md,plan,top_path,selection)
    except Exception as exc:
        warnings.append(f'PCA 2D FES skipped: {exc}')
        return {'available':False,'reason':str(exc)}
    # Pre-load topology once for iterload; also pre-select atoms so iterload
    # passes atom_indices and the XTC codec skips non-CA atoms at byte level.
    try:
        _pca_top_obj = md.load(str(top_path))
        _pca_top = _pca_top_obj.topology
    except Exception:
        _pca_top = str(top_path)
    max_fit=max(2,int(getattr(args,'pca_max_fit_frames',50000) or 50000))
    total_assign=sum(int(x['n_assign']) for x in plan)
    user_stride=int(getattr(args,'pca_fit_stride',0) or 0)
    fit_stride=max(1,user_stride if user_stride>0 else int(math.ceil(total_assign/max_fit)))
    chunk_size=max(1,int(getattr(args,'pca_chunk_size',1000) or 1000))
    if progress is not None:
        progress.step('PCA trajectories', f'fit selection {selection!r}; atoms={int(len(atoms))}; fit stride={fit_stride}; max fit frames={max_fit}')
    fit_blocks=[]
    fit_count=0
    for pi,item in enumerate(plan, start=1):
        if fit_count >= max_fit:
            break
        if progress is not None:
            progress.bar('PCA fit pass', pi, max(1,len(plan)), f"replica {item['replica']}")
        frame0=0
        frame_indices=np.asarray(item.get('frame_indices', np.arange(item['n_assign'])),dtype=np.int64)
        sample_indices=np.asarray(item.get('sample_indices', item['order']),dtype=np.int64)
        try:
            for chunk in md.iterload(str(item.get('traj', item['dcd'])), top=_pca_top, chunk=chunk_size, atom_indices=atoms):
                if fit_count >= max_fit:
                    break
                local, _sample_idx, global_frames = _chunk_local_frame_selection(frame_indices, sample_indices, frame0, chunk.n_frames)
                if local is not None and local.size:
                    keep=np.where((global_frames % fit_stride) == 0)[0]
                    if keep.size:
                        coords_all=_aligned_flattened_coords_A(chunk, atoms, ref, pre_sliced=True)
                        coords=coords_all[local[keep]]
                        need=max_fit-fit_count
                        fit_blocks.append(coords[:need].copy())
                        fit_count += int(min(coords.shape[0], need))
                frame0 += chunk.n_frames
                if frame0 > int(np.max(frame_indices)) and fit_count >= max_fit:
                    break
        except Exception as exc:
            warnings.append(f'PCA fit pass failed for replica {item["replica"]} ({item.get("traj", item.get("dcd"))}): {exc}')
    if progress is not None:
        progress.bar('PCA fit pass', 1, 1, f'collected {fit_count} fit frames', force=True)
    if fit_count < 3 or not fit_blocks:
        return {'available':False,'reason':f'too few PCA fit frames collected ({fit_count})'}
    X=np.vstack(fit_blocks)
    mean=X.mean(axis=0)
    Xc=X-mean
    try:
        _u,svals,vt=np.linalg.svd(Xc, full_matrices=False)
    except Exception as exc:
        return {'available':False,'reason':f'PCA SVD failed: {exc}'}
    if vt.shape[0] < 2:
        return {'available':False,'reason':'PCA produced fewer than two components'}
    components=vt[:2].copy()
    evals=(svals*svals)/max(1, X.shape[0]-1)
    total_var=float(np.sum(evals)) if evals.size else float('nan')
    evr=np.asarray(evals[:2]/total_var if total_var>0 else [np.nan,np.nan], dtype=np.float64)
    pca1=np.full(d.cv.shape, np.nan, dtype=np.float32)
    pca2=np.full(d.cv.shape, np.nan, dtype=np.float32)
    assigned=0
    for pi,item in enumerate(plan, start=1):
        if progress is not None:
            progress.bar('PCA project pass', pi, max(1,len(plan)), f"replica {item['replica']}")
        frame0=0
        frame_indices=np.asarray(item.get('frame_indices', np.arange(item['n_assign'])),dtype=np.int64)
        sample_indices=np.asarray(item.get('sample_indices', item['order']),dtype=np.int64)
        try:
            for chunk in md.iterload(str(item.get('traj', item['dcd'])), top=_pca_top, chunk=chunk_size, atom_indices=atoms):
                local, sample_idx, _global_frames = _chunk_local_frame_selection(frame_indices, sample_indices, frame0, chunk.n_frames)
                if local is not None and local.size:
                    coords_all=_aligned_flattened_coords_A(chunk, atoms, ref, pre_sliced=True)
                    scores=(coords_all[local]-mean)@components.T
                    pca1[sample_idx]=scores[:,0].astype(np.float32)
                    pca2[sample_idx]=scores[:,1].astype(np.float32)
                    assigned += int(sample_idx.size)
                frame0 += chunk.n_frames
                if frame0 > int(np.max(frame_indices)):
                    break
        except Exception as exc:
            warnings.append(f'PCA projection failed for replica {item["replica"]} ({item.get("traj", item.get("dcd"))}): {exc}')
    if progress is not None:
        progress.bar('PCA project pass', 1, 1, f'assigned {assigned}/{d.cv.size} samples', force=True)
    info={
        'available': True,
        'source': 'trajectory_reconstruction',
        'topology': str(top_path),
        'reference_dcd': ref_dcd,
        'selection': selection,
        'n_atoms': int(len(atoms)),
        'n_fit_frames': int(fit_count),
        'fit_stride': int(fit_stride),
        'max_fit_frames': int(max_fit),
        'chunk_size': int(chunk_size),
        'n_projected_samples': int(np.count_nonzero(np.isfinite(pca1)&np.isfinite(pca2))),
        'explained_variance_ratio_pc1': float(evr[0]) if evr.size>0 else float('nan'),
        'explained_variance_ratio_pc2': float(evr[1]) if evr.size>1 else float('nan'),
        'pca1': pca1.astype(np.float64),
        'pca2': pca2.astype(np.float64),
    }
    try:
        np.savez_compressed(cache, pca1=pca1, pca2=pca2, mean_A=mean.astype(np.float32), components=components.astype(np.float32), explained_variance_ratio=evr.astype(np.float32), selection=np.asarray([selection]), topology=np.asarray([str(top_path)]))
        meta={k:v for k,v in info.items() if k not in {'pca1','pca2'}}
        meta['files']={'pca_scores_npz':str(cache)}
        wjson(out/'pca_scores_metadata.json',meta)
        info['files']={'pca_scores_npz':str(cache),'pca_scores_metadata_json':str(out/'pca_scores_metadata.json')}
    except Exception as exc:
        warnings.append(f'Could not write PCA score cache: {exc}')
        info['files']={}
    return info

def analyze_pca_2d_fes(d: Data, args, base_logw: np.ndarray, selected: str, boost_ok: bool, kbt_kcal: float, out: Path, warnings: list[str], progress: Optional[Progress]) -> dict:
    score_info=_fit_and_project_pca_from_trajectories(d,args,out,progress,warnings)
    if not isinstance(score_info,dict) or not score_info.get('available'):
        return score_info if isinstance(score_info,dict) else {'available':False,'reason':'PCA score reconstruction failed'}
    pca1=np.asarray(score_info.get('pca1'),dtype=np.float64)
    pca2=np.asarray(score_info.get('pca2'),dtype=np.float64)
    mask=np.isfinite(pca1)&np.isfinite(pca2)&np.isfinite(base_logw)
    if np.count_nonzero(mask) < max(20, d.u_nk.shape[1]):
        return {'available':False,'reason':'Too few finite PCA1/PCA2 samples for 2D FES','n_finite':int(np.count_nonzero(mask))}
    if progress is not None:
        progress.bar('analysis stages',4,6,'building PCA1-PCA2 2D FES',force=True)
    bins_n=int(getattr(args,'pca_bins',None) or args.bins)
    xbins=make_bins(pca1[mask], bins_n, getattr(args,'pca1_min',None), getattr(args,'pca1_max',None))
    ybins=make_bins(pca2[mask], bins_n, getattr(args,'pca2_min',None), getattr(args,'pca2_max',None))
    base_logw_sel=np.asarray(base_logw,dtype=np.float64)[mask]
    base_w=norm_logw(base_logw_sel)
    boost_sel=d.boost_kj[mask]
    x=pca1[mask]
    y=pca2[mask]
    fes_umbrella=pmf2d_from_weights(x,y,base_w,xbins,ybins,kbt_kcal)
    if boost_ok and np.isfinite(boost_sel).sum()>10 and np.nanstd(boost_sel)>1e-12:
        exp_w=norm_logw(base_logw_sel+d.beta*boost_sel)
        fes_exp=pmf2d_from_weights(x,y,exp_w,xbins,ybins,kbt_kcal)
        (fes_cum,_cdiag),(fes_cum3,_cdiag3)=_cumulant_expansion_2d_both(x,y,base_w,boost_sel,xbins,ybins,d.beta,kbt_kcal,smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
        chosen=selected if selected in {'gamd_exponential','gamd_cumulant2','gamd_cumulant3'} else 'gamd_cumulant2'
    else:
        fes_exp=fes_umbrella
        fes_cum=fes_umbrella
        fes_cum3=fes_umbrella
        chosen='umbrella_only'
    fes_map={'umbrella_only':fes_umbrella,'gamd_exponential':fes_exp,'gamd_cumulant2':fes_cum,'gamd_cumulant3':fes_cum3}
    chosen_fes=fes_map.get(chosen,fes_umbrella)
    write_pca_2d_fes_csv(out/'pca_2d_fes_selected.csv',chosen_fes,chosen)
    write_pca_2d_fes_npz(out/'pca_2d_fes_selected.npz',chosen_fes,chosen)
    plot_files=plot_pca_2d_fes(chosen_fes,chosen,out/'pca_2d_fes_selected.png',f'PCA1 vs PCA2 2D free-energy surface ({chosen})',warnings,smooth_sigma=float(getattr(args,'pca_smooth_sigma',1.0)))
    write_pca_2d_fes_csv(out/'pca_2d_fes_cumulant2.csv',fes_cum,'gamd_cumulant2')
    write_pca_2d_fes_npz(out/'pca_2d_fes_cumulant2.npz',fes_cum,'gamd_cumulant2')
    plot_files_c2=plot_pca_2d_fes(fes_cum,'gamd_cumulant2',out/'pca_2d_fes_cumulant2.png','PCA1 vs PCA2 2D free-energy surface (gamd_cumulant2)',warnings,smooth_sigma=float(getattr(args,'pca_smooth_sigma',1.0)))
    want_cum3=_want_gamd_method('gamd_cumulant3', chosen, args)
    plot_files_c3={}
    if want_cum3:
        write_pca_2d_fes_csv(out/'pca_2d_fes_cumulant3.csv',fes_cum3,'gamd_cumulant3')
        write_pca_2d_fes_npz(out/'pca_2d_fes_cumulant3.npz',fes_cum3,'gamd_cumulant3')
        plot_files_c3=plot_pca_2d_fes(fes_cum3,'gamd_cumulant3',out/'pca_2d_fes_cumulant3.png','PCA1 vs PCA2 2D free-energy surface (gamd_cumulant3)',warnings,smooth_sigma=float(getattr(args,'pca_smooth_sigma',1.0)))
    finite=np.isfinite(chosen_fes['pmf'])
    if np.any(finite):
        min_idx=np.unravel_index(np.nanargmin(np.where(finite,chosen_fes['pmf'],np.inf)),chosen_fes['pmf'].shape)
        min_pca1=float(chosen_fes['cv_A'][min_idx[0]])
        min_pca2=float(chosen_fes['rg_A'][min_idx[1]])
        span=float(np.nanmax(chosen_fes['pmf'][finite])-np.nanmin(chosen_fes['pmf'][finite]))
    else:
        min_pca1=min_pca2=span=float('nan')
    files={'pca_2d_fes_csv':str(out/'pca_2d_fes_selected.csv'),'pca_2d_fes_npz':str(out/'pca_2d_fes_selected.npz'),'pca_2d_fes_png':str(out/'pca_2d_fes_selected.png'), **{f'pca_2d_fes_png_{k}': v for k, v in (plot_files or {}).items()},
           'pca_2d_fes_cumulant2_csv':str(out/'pca_2d_fes_cumulant2.csv'),'pca_2d_fes_cumulant2_npz':str(out/'pca_2d_fes_cumulant2.npz'),'pca_2d_fes_cumulant2_png':str(out/'pca_2d_fes_cumulant2.png'), **{f'pca_2d_fes_cumulant2_png_{k}': v for k, v in (plot_files_c2 or {}).items()},
           **({'pca_2d_fes_cumulant3_csv':str(out/'pca_2d_fes_cumulant3.csv'),'pca_2d_fes_cumulant3_npz':str(out/'pca_2d_fes_cumulant3.npz'),'pca_2d_fes_cumulant3_png':str(out/'pca_2d_fes_cumulant3.png'), **{f'pca_2d_fes_cumulant3_png_{k}': v for k, v in (plot_files_c3 or {}).items()}} if want_cum3 else {}),
           'pca_2d_fes_summary_json':str(out/'pca_2d_fes_summary.json')}
    if isinstance(score_info.get('files'),dict):
        files.update(score_info['files'])
    info={
        'available':True,
        'selected_unbiased_method':chosen,
        'n_samples':int(np.count_nonzero(mask)),
        'pca_bins':int(len(xbins)-1),
        'pmf_minimum_pca1_A':min_pca1,
        'pmf_minimum_pca2_A':min_pca2,
        'pmf_span_kcal_mol':span,
        'normalization':'Minimum finite free energy shifted to 0 kcal/mol',
        'selection':score_info.get('selection'),
        'n_atoms':score_info.get('n_atoms'),
        'n_fit_frames':score_info.get('n_fit_frames'),
        'explained_variance_ratio_pc1':score_info.get('explained_variance_ratio_pc1'),
        'explained_variance_ratio_pc2':score_info.get('explained_variance_ratio_pc2'),
        'source':score_info.get('source'),
        'files':files,
    }
    wjson(out/'pca_2d_fes_summary.json',info)
    return info


def _slug(text: str, max_len: int = 80) -> str:
    raw=str(text)
    out=[]
    for ch in raw:
        if ch.isalnum() or ch in {'-','_'}:
            out.append(ch)
        else:
            out.append('_')
    s=''.join(out).strip('_') or 'item'
    while '__' in s:
        s=s.replace('__','_')
    return s[:max_len]

def _md_residue_label(res) -> str:
    name=getattr(res,'name','RES')
    idx=int(getattr(res,'index',0))
    resseq=getattr(res,'resSeq',None)
    if resseq is None:
        return f'{idx:03d}_{name}'
    return f'{idx:03d}_{name}{resseq}'

def _find_extra_topology_path(prod: Path, args) -> Optional[Path]:
    explicit = _find_explicit_topology_path(args, ('extra_topology', 'pca_topology', 'rg_topology'))
    if explicit is not None:
        return explicit
    return _find_default_topology_path(prod)

def _extra_replica_plan(d: Data, args, md, traj_dir: Path, warnings: list[str]) -> list[dict]:
    reps=sorted(set(int(x) for x in d.replica if np.isfinite(x)))
    plan=[]
    spf=_read_traj_interval(d.prod_dir)
    if spf <= 0 or spf == 500:
        epoch_spf = _read_traj_interval_from_epoch_dirs(d)
        if epoch_spf > 0: spf = epoch_spf
    use_adj = (traj_dir != d.prod_dir / 'replica_trajectories')
    adj_steps_all = _adjusted_steps_for_merged_traj(d, spf) if use_adj else None
    for rep in reps:
        segs=_find_all_replica_trajectory_segments(traj_dir, rep, args)
        if not segs:
            warnings.append(f'Extra-observable trajectory missing for replica {rep}: {_trajectory_pattern_hint(args)} in {traj_dir}')
            continue
        idx=np.where(d.replica==rep)[0]
        if idx.size==0:
            continue
        order=idx[np.argsort(d.step[idx], kind='stable')]
        steps_for_align = adj_steps_all[order] if adj_steps_all is not None else d.step[order]
        for resume_start, seg_path in segs:
            n_seg_frames=_trajectory_frame_count(md, seg_path)
            if n_seg_frames is None or n_seg_frames<=0:
                continue
            eff_resume_start=_base_segment_resume_start(resume_start, use_adj, steps_for_align, spf)
            mask, local_frames=_sample_to_segment_frame(steps_for_align, eff_resume_start, n_seg_frames, spf)
            if not np.any(mask):
                continue
            seg_order=order[mask]
            plan.append({'replica':rep,'traj':seg_path,'dcd':seg_path,'order':seg_order,
                         'sample_indices':seg_order,'frame_indices':local_frames,
                         'n_assign':int(seg_order.size),'n_frames':n_seg_frames,
                         'alignment_mode':'step_exact'})
    return plan

def _bin_indices_1d(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    v=np.asarray(values,dtype=np.float64)
    edges=np.asarray(edges,dtype=np.float64)
    idx=np.searchsorted(edges,v,side='right')-1
    if edges.size >= 2:
        idx[v==edges[-1]]=edges.size-2
    good=np.isfinite(v)&(idx>=0)&(idx<edges.size-1)
    return idx.astype(np.int64,copy=False), good

def _make_acc_1d(edges: np.ndarray) -> dict:
    nb=len(edges)-1
    return {'edges':np.asarray(edges,dtype=np.float64),'counts':np.zeros(nb,dtype=np.int64),'umbrella':np.zeros(nb,dtype=np.float64),'exp':np.zeros(nb,dtype=np.float64),'sw':np.zeros(nb,dtype=np.float64),'sx':np.zeros(nb,dtype=np.float64),'sx2':np.zeros(nb,dtype=np.float64),'sx3':np.zeros(nb,dtype=np.float64)}

def _accumulate_1d(acc: dict, values: np.ndarray, sample_idx: np.ndarray, d: Data, base_w_full: np.ndarray, exp_w_full: np.ndarray) -> None:
    idx,good=_bin_indices_1d(values,acc['edges'])
    if not np.any(good):
        return
    ii=idx[good]
    sidx=np.asarray(sample_idx,dtype=np.int64)[good]
    nb=len(acc['edges'])-1
    acc['counts']+=np.bincount(ii,minlength=nb).astype(np.int64)
    bw=np.asarray(base_w_full[sidx],dtype=np.float64)
    ew=np.asarray(exp_w_full[sidx],dtype=np.float64)
    bboost=np.asarray(d.boost_kj[sidx],dtype=np.float64)
    finite_bw=np.isfinite(bw)&(bw>=0)
    if np.any(finite_bw):
        acc['umbrella']+=np.bincount(ii[finite_bw],weights=bw[finite_bw],minlength=nb)
    finite_ew=np.isfinite(ew)&(ew>=0)
    if np.any(finite_ew):
        acc['exp']+=np.bincount(ii[finite_ew],weights=ew[finite_ew],minlength=nb)
    finite_boost=finite_bw&np.isfinite(bboost)
    if np.any(finite_boost):
        acc['sw']+=np.bincount(ii[finite_boost],weights=bw[finite_boost],minlength=nb)
        acc['sx']+=np.bincount(ii[finite_boost],weights=bw[finite_boost]*bboost[finite_boost],minlength=nb)
        acc['sx2']+=np.bincount(ii[finite_boost],weights=bw[finite_boost]*bboost[finite_boost]*bboost[finite_boost],minlength=nb)
        acc['sx3']+=np.bincount(ii[finite_boost],weights=bw[finite_boost]*bboost[finite_boost]*bboost[finite_boost]*bboost[finite_boost],minlength=nb)

def _pmf_from_probability_centers(x: np.ndarray, prob: np.ndarray, counts: np.ndarray, kbt_kcal: float) -> dict:
    p=np.asarray(prob,dtype=np.float64)
    ps=float(np.nansum(p))
    if ps>0:
        p=p/ps
    with np.errstate(divide='ignore',invalid='ignore'):
        F=-float(kbt_kcal)*np.log(p)
    mask=np.isfinite(F)
    if np.any(mask):
        F-=np.nanmin(F[mask])
    return {'x':np.asarray(x,dtype=np.float64),'prob':p,'pmf':F,'counts':np.asarray(counts,dtype=np.int64)}

def _finalize_acc_1d(acc: dict, d: Data, kbt_kcal: float, smooth_logfac_sigma: float = 0.0) -> tuple[dict,dict]:
    edges=np.asarray(acc['edges'],dtype=np.float64)
    x=0.5*(edges[:-1]+edges[1:])
    umbrella=_pmf_from_probability_centers(x,acc['umbrella'],acc['counts'],kbt_kcal)
    exp_pmf=_pmf_from_probability_centers(x,acc['exp'],acc['counts'],kbt_kcal)
    mean=np.full_like(x,np.nan,dtype=np.float64)
    var=np.full_like(x,np.nan,dtype=np.float64)
    kappa3=np.full_like(x,np.nan,dtype=np.float64)
    logfac=np.zeros_like(x,dtype=np.float64)
    logfac3=np.zeros_like(x,dtype=np.float64)
    nz=acc['sw']>0
    if np.any(nz):
        mean[nz]=acc['sx'][nz]/acc['sw'][nz]
        var[nz]=np.maximum(0.0,acc['sx2'][nz]/acc['sw'][nz]-mean[nz]*mean[nz])
        logfac[nz]=d.beta*mean[nz]+0.5*d.beta*d.beta*var[nz]
        # streaming accumulator: mean not known until all chunks are in, so a
        # mean-centered two-pass kappa3 isn't possible here; fall back to the
        # raw-moment identity (same cancellation risk already accepted by var above).
        kappa3[nz]=acc['sx3'][nz]/acc['sw'][nz]-3.0*mean[nz]*(acc['sx2'][nz]/acc['sw'][nz])+2.0*mean[nz]**3
        logfac3[nz]=logfac[nz]+(d.beta**3/6.0)*kappa3[nz]
    # See _cumulant_expansion: a bin with real (umbrella-weighted) samples but
    # zero finite-boost samples has an unknown correction -- NaN, not a
    # silent logfac=0 fallback. Genuinely empty bins (acc['umbrella']==0)
    # stay at logfac=0 -> cum_prob=0, unchanged.
    unknown=(~nz)&(acc['umbrella']>0)
    logfac[unknown]=np.nan
    logfac3[unknown]=np.nan
    if smooth_logfac_sigma and float(smooth_logfac_sigma) > 0:
        try:
            from scipy.ndimage import gaussian_filter1d
            logfac=gaussian_filter1d(logfac,sigma=float(smooth_logfac_sigma),mode='nearest')
            logfac3=gaussian_filter1d(logfac3,sigma=float(smooth_logfac_sigma),mode='nearest')
        except Exception:
            pass
    cum_prob=acc['umbrella']*np.exp(np.clip(logfac,-700,700))
    cum_pmf=_pmf_from_probability_centers(x,cum_prob,acc['counts'],kbt_kcal)
    cum3_prob=acc['umbrella']*np.exp(np.clip(logfac3,-700,700))
    cum3_pmf=_pmf_from_probability_centers(x,cum3_prob,acc['counts'],kbt_kcal)
    return {'umbrella_only':umbrella,'gamd_exponential':exp_pmf,'gamd_cumulant2':cum_pmf,'gamd_cumulant3':cum3_pmf},{'boost_mean_kj':mean,'boost_var_kj2':var,'boost_kappa3_kj3':kappa3,'log_reweight_factor':logfac}

def _make_acc_2d(xedges: np.ndarray, yedges: np.ndarray) -> dict:
    nx=len(xedges)-1; ny=len(yedges)-1
    shape=(nx,ny)
    return {'xedges':np.asarray(xedges,dtype=np.float64),'yedges':np.asarray(yedges,dtype=np.float64),'counts':np.zeros(shape,dtype=np.int64),'umbrella':np.zeros(shape,dtype=np.float64),'exp':np.zeros(shape,dtype=np.float64),'sw':np.zeros(shape,dtype=np.float64),'sx':np.zeros(shape,dtype=np.float64),'sx2':np.zeros(shape,dtype=np.float64),'sx3':np.zeros(shape,dtype=np.float64)}

def _accumulate_2d(acc: dict, xvals: np.ndarray, yvals: np.ndarray, sample_idx: np.ndarray, d: Data, base_w_full: np.ndarray, exp_w_full: np.ndarray) -> None:
    x=np.asarray(xvals,dtype=np.float64); y=np.asarray(yvals,dtype=np.float64)
    xe=acc['xedges']; ye=acc['yedges']
    xi=np.searchsorted(xe,x,side='right')-1; yi=np.searchsorted(ye,y,side='right')-1
    xi[x==xe[-1]]=len(xe)-2; yi[y==ye[-1]]=len(ye)-2
    good=np.isfinite(x)&np.isfinite(y)&(xi>=0)&(xi<len(xe)-1)&(yi>=0)&(yi<len(ye)-1)
    if not np.any(good): return
    nx=len(xe)-1; ny=len(ye)-1
    linear=(xi[good].astype(np.int64)*ny+yi[good].astype(np.int64))
    sidx=np.asarray(sample_idx,dtype=np.int64)[good]
    acc['counts']+=np.bincount(linear,minlength=nx*ny).reshape(nx,ny).astype(np.int64)
    bw=np.asarray(base_w_full[sidx],dtype=np.float64); ew=np.asarray(exp_w_full[sidx],dtype=np.float64); boost=np.asarray(d.boost_kj[sidx],dtype=np.float64)
    finite_bw=np.isfinite(bw)&(bw>=0)
    if np.any(finite_bw): acc['umbrella']+=np.bincount(linear[finite_bw],weights=bw[finite_bw],minlength=nx*ny).reshape(nx,ny)
    finite_ew=np.isfinite(ew)&(ew>=0)
    if np.any(finite_ew): acc['exp']+=np.bincount(linear[finite_ew],weights=ew[finite_ew],minlength=nx*ny).reshape(nx,ny)
    finite_boost=finite_bw&np.isfinite(boost)
    if np.any(finite_boost):
        acc['sw']+=np.bincount(linear[finite_boost],weights=bw[finite_boost],minlength=nx*ny).reshape(nx,ny)
        acc['sx']+=np.bincount(linear[finite_boost],weights=bw[finite_boost]*boost[finite_boost],minlength=nx*ny).reshape(nx,ny)
        acc['sx2']+=np.bincount(linear[finite_boost],weights=bw[finite_boost]*boost[finite_boost]*boost[finite_boost],minlength=nx*ny).reshape(nx,ny)
        acc['sx3']+=np.bincount(linear[finite_boost],weights=bw[finite_boost]*boost[finite_boost]*boost[finite_boost]*boost[finite_boost],minlength=nx*ny).reshape(nx,ny)

def _pmf2d_from_probability(xedges: np.ndarray, yedges: np.ndarray, prob: np.ndarray, counts: np.ndarray, kbt_kcal: float) -> dict:
    p=np.asarray(prob,dtype=np.float64)
    ps=float(np.nansum(p))
    if ps>0: p=p/ps
    with np.errstate(divide='ignore',invalid='ignore'):
        F=-float(kbt_kcal)*np.log(p)
    mask=np.isfinite(F)
    if np.any(mask): F-=np.nanmin(F[mask])
    return {'x':0.5*(xedges[:-1]+xedges[1:]),'y':0.5*(yedges[:-1]+yedges[1:]),'xedges':np.asarray(xedges,dtype=np.float64),'yedges':np.asarray(yedges,dtype=np.float64),'prob':p,'pmf':F,'counts':np.asarray(counts,dtype=np.int64)}

def _finalize_acc_2d(acc: dict, d: Data, kbt_kcal: float, smooth_logfac_sigma: float = 0.0) -> tuple[dict,dict]:
    xe=acc['xedges']; ye=acc['yedges']
    umbrella=_pmf2d_from_probability(xe,ye,acc['umbrella'],acc['counts'],kbt_kcal)
    exp_pmf=_pmf2d_from_probability(xe,ye,acc['exp'],acc['counts'],kbt_kcal)
    mean=np.full(acc['umbrella'].shape,np.nan,dtype=np.float64); var=np.full_like(mean,np.nan); kappa3=np.full_like(mean,np.nan)
    logfac=np.zeros_like(mean); logfac3=np.zeros_like(mean)
    nz=acc['sw']>0
    if np.any(nz):
        mean[nz]=acc['sx'][nz]/acc['sw'][nz]
        var[nz]=np.maximum(0.0,acc['sx2'][nz]/acc['sw'][nz]-mean[nz]*mean[nz])
        logfac[nz]=d.beta*mean[nz]+0.5*d.beta*d.beta*var[nz]
        # streaming accumulator: see _finalize_acc_1d note on raw-moment kappa3.
        kappa3[nz]=acc['sx3'][nz]/acc['sw'][nz]-3.0*mean[nz]*(acc['sx2'][nz]/acc['sw'][nz])+2.0*mean[nz]**3
        logfac3[nz]=logfac[nz]+(d.beta**3/6.0)*kappa3[nz]
    # See _cumulant_expansion / _finalize_acc_1d: a bin with real samples but
    # zero finite-boost samples gets an unknown (NaN) correction rather than
    # a silent logfac=0 fallback.
    unknown=(~nz)&(acc['umbrella']>0)
    logfac[unknown]=np.nan
    logfac3[unknown]=np.nan
    if smooth_logfac_sigma and float(smooth_logfac_sigma) > 0:
        try:
            from scipy.ndimage import gaussian_filter
            logfac=gaussian_filter(logfac,sigma=float(smooth_logfac_sigma),mode='nearest')
            logfac3=gaussian_filter(logfac3,sigma=float(smooth_logfac_sigma),mode='nearest')
        except Exception:
            pass
    cum_prob=acc['umbrella']*np.exp(np.clip(logfac,-700,700))
    cum_pmf=_pmf2d_from_probability(xe,ye,cum_prob,acc['counts'],kbt_kcal)
    cum3_prob=acc['umbrella']*np.exp(np.clip(logfac3,-700,700))
    cum3_pmf=_pmf2d_from_probability(xe,ye,cum3_prob,acc['counts'],kbt_kcal)
    return {'umbrella_only':umbrella,'gamd_exponential':exp_pmf,'gamd_cumulant2':cum_pmf,'gamd_cumulant3':cum3_pmf},{'boost_mean_kj':mean,'boost_var_kj2':var,'boost_kappa3_kj3':kappa3,'log_reweight_factor':logfac}


def _choose_method(selected: str, boost_ok: bool) -> str:
    if boost_ok and selected in {'gamd_exponential','gamd_cumulant2','gamd_cumulant3'}:
        return selected
    if boost_ok and selected not in {'umbrella_only','gamd_exponential','gamd_cumulant2','gamd_cumulant3'}:
        return 'gamd_cumulant2'
    return 'umbrella_only'

def _wrap_degrees(rad_values: np.ndarray) -> np.ndarray:
    deg=np.degrees(np.asarray(rad_values,dtype=np.float64))
    return ((deg+180.0)%360.0)-180.0

def _protein_traj(chunk):
    atoms=chunk.topology.select('protein')
    if atoms.size == 0:
        return None
    return chunk.atom_slice(atoms)

def _sasa_total_A2(md, traj, selection: str, n_sphere_points: int) -> np.ndarray:
    atoms=traj.topology.select(selection)
    if atoms.size == 0:
        atoms=traj.topology.select('protein')
    if atoms.size == 0:
        return np.full(traj.n_frames,np.nan,dtype=np.float64)
    sub=traj.atom_slice(atoms)
    sasa=md.shrake_rupley(sub,mode='atom',n_sphere_points=int(n_sphere_points))
    return np.asarray(np.sum(sasa,axis=1)*100.0,dtype=np.float64)

def _internal_contact_counts(md, traj, cutoff_nm: float, min_seq_sep: int, scheme: str) -> np.ndarray:
    try:
        dist,pairs=md.compute_contacts(traj,contacts='all',scheme=scheme,ignore_nonprotein=True)
    except Exception:
        return np.full(traj.n_frames,np.nan,dtype=np.float64)
    if pairs is None or len(pairs)==0:
        return np.zeros(traj.n_frames,dtype=np.float64)
    pairs=np.asarray(pairs,dtype=np.int64)
    keep=np.abs(pairs[:,0]-pairs[:,1])>=int(min_seq_sep)
    if not np.any(keep):
        return np.zeros(traj.n_frames,dtype=np.float64)
    return np.sum(np.asarray(dist[:,keep])<float(cutoff_nm),axis=1).astype(np.float64)

def _build_peptide_solvent_classification(top) -> dict:
    """Classify a topology's heavy atoms into peptide/water/ion/solvent groups.

    Pure function of the topology alone -- independent of any per-frame
    trajectory data. The topology never changes across the many chunks/segments/
    replicas processed within one `analyze_extra_observable_pmfs` invocation, so
    callers should compute this ONCE and reuse it, instead of re-walking every
    atom in the topology (previously ~10ms) on every one of the thousands of
    per-chunk calls a long trajectory pass makes.
    """
    peptide_heavy=[]; water_heavy=[]; ion_heavy=[]; solvent_heavy=[]
    atom_to_res=np.full(top.n_atoms,-1,dtype=np.int64)
    water_res=set(); ion_res=set(); solvent_res=set()
    for atom in top.atoms:
        elem=getattr(atom,'element',None)
        if elem is None or getattr(elem,'symbol',None) == 'H':
            continue
        ridx=atom.residue.index
        atom_to_res[atom.index]=ridx
        if atom.residue.is_protein:
            peptide_heavy.append(atom.index)
        elif atom.residue.is_water:
            water_heavy.append(atom.index); solvent_heavy.append(atom.index); water_res.add(ridx); solvent_res.add(ridx)
        else:
            ion_heavy.append(atom.index); solvent_heavy.append(atom.index); ion_res.add(ridx); solvent_res.add(ridx)
    return {
        'peptide_heavy': np.asarray(peptide_heavy,dtype=np.int32),
        'solvent_heavy': np.asarray(solvent_heavy,dtype=np.int32),
        'atom_to_res': atom_to_res,
        'water_res': water_res,
        'ion_res': ion_res,
        'solvent_res': solvent_res,
    }

def _peptide_solvent_contact_counts(md, traj, cutoff_nm: float, classification: dict) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    peptide_heavy=classification['peptide_heavy']; solvent_heavy=classification['solvent_heavy']
    atom_to_res=classification['atom_to_res']; water_res=classification['water_res']
    ion_res=classification['ion_res']; solvent_res=classification['solvent_res']
    n=traj.n_frames
    zeros=np.zeros(n,dtype=np.float64)
    if peptide_heavy.size==0 or solvent_heavy.size==0:
        return zeros.copy(), zeros.copy(), zeros.copy()
    try:
        neigh=md.compute_neighbors(traj,float(cutoff_nm),query_indices=peptide_heavy,haystack_indices=solvent_heavy,periodic=True)
    except Exception:
        return np.full(n,np.nan,dtype=np.float64), np.full(n,np.nan,dtype=np.float64), np.full(n,np.nan,dtype=np.float64)
    total=np.zeros(n,dtype=np.float64); water=np.zeros(n,dtype=np.float64); ion=np.zeros(n,dtype=np.float64)
    for i,arr in enumerate(neigh):
        if arr is None or len(arr)==0:
            continue
        residues={int(atom_to_res[int(a)]) for a in np.asarray(arr,dtype=np.int64) if 0 <= int(a) < atom_to_res.size and atom_to_res[int(a)] >= 0}
        total[i]=float(len(residues & solvent_res))
        water[i]=float(len(residues & water_res))
        ion[i]=float(len(residues & ion_res))
    return total, water, ion

def _dssp_valid_counts(arr, axis):
    """Count DSSP H/E/coil occurrences and valid (non-'NA') entries along `axis`.

    mdtraj's `compute_dssp` assigns 'NA' to any topology "residue" lacking a
    full CA/N/C/O backbone quad -- notably capping groups like ACE/NME, which
    this pipeline explicitly supports. 'NA' entries must be excluded from both
    the coil/other count and the denominator used to turn counts into
    fractions/probabilities, otherwise capped termini are silently counted as
    coil and bias helix/strand fractions low.
    """
    valid=(arr != 'NA')
    n_h=np.sum(arr == 'H', axis=axis)
    n_e=np.sum(arr == 'E', axis=axis)
    n_coil=np.sum((arr != 'H') & (arr != 'E') & valid, axis=axis)
    n_valid=np.sum(valid, axis=axis)
    return n_h, n_e, n_coil, n_valid

def _dssp_fraction_arrays(md, traj):
    try:
        ss=md.compute_dssp(traj,simplified=True)
    except Exception:
        return None
    arr=np.asarray(ss)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return None
    n_h,n_e,n_coil,n_valid=_dssp_valid_counts(arr,axis=1)
    denom=np.maximum(n_valid,1)
    helix=(n_h/denom).astype(np.float64)
    strand=(n_e/denom).astype(np.float64)
    coil=(n_coil/denom).astype(np.float64)
    return arr,helix,strand,coil

def _dssp_residue_probability_rows(dssp_labels, dssp_counts, dssp_valid_counts, dssp_total):
    """Build per-residue secondary-structure probability rows.

    `dssp_valid_counts[i]` is the number of frames where residue index `i` was
    not mdtraj's 'NA' code (see `_dssp_valid_counts`). A residue index that is
    'NA' in every frame -- a capped terminus (ACE/NME) or other non-protein
    "residue" -- has no real secondary structure to report; it's excluded
    entirely rather than emitted as a spurious 0/0 probability. Real residues
    use their own valid-frame count as the denominator instead of the total
    frame count, so an occasional NA frame (if any) doesn't dilute them either.
    """
    rows=[]
    for i,lab in enumerate(dssp_labels):
        n_valid_i=int(dssp_valid_counts[i]) if dssp_valid_counts is not None else dssp_total
        if n_valid_i <= 0:
            continue
        rows.append({'residue':lab,'n_frames':n_valid_i,'helix_probability':float(dssp_counts['H'][i]/n_valid_i),'strand_probability':float(dssp_counts['E'][i]/n_valid_i),'coil_other_probability':float(dssp_counts['C'][i]/n_valid_i)})
    return rows

def analyze_extra_observable_pmfs(d: Data, args, base_logw: np.ndarray, selected: str, boost_ok: bool, kbt_kcal: float, out: Path, warnings: list[str], progress: Optional[Progress]) -> dict:
    mode=str(getattr(args,'extra_pmf_from_trajectories','auto') or 'auto').lower()
    if mode == 'never':
        return {'available':False,'reason':'disabled by --extra-pmf-from-trajectories never'}
    traj_dir=d.prod_dir/'replica_trajectories'
    if not traj_dir.exists():
        merged = _prepare_adaptive_merged_traj_dir(d, args)
        if merged is not None:
            traj_dir = merged
        else:
            if mode == 'force': warnings.append(f'Extra observable PMFs requested but {traj_dir} is missing')
            return {'available':False,'reason':f'{traj_dir} is missing'}
    top_path=_find_data_topology_path(d, args, ('extra_topology', 'pca_topology', 'rg_topology'))
    if top_path is None:
        if mode == 'force': warnings.append('Extra observable PMFs requested but no topology PDB was found; use --extra-topology')
        return {'available':False,'reason':'no topology PDB found; use --extra-topology'}
    try:
        import mdtraj as md
    except Exception as exc:
        warnings.append(f'Extra observable PMFs skipped: mdtraj is unavailable ({exc})')
        return {'available':False,'reason':f'mdtraj unavailable: {exc}'}
    plan=_extra_replica_plan(d,args,md,traj_dir,warnings)
    if not plan:
        return {'available':False,'reason':'no usable replica trajectories for extra observable PMFs'}
    out_extra=out/'extra_observable_pmfs'; out_extra.mkdir(parents=True,exist_ok=True)
    chunk_size=max(1,int(getattr(args,'extra_chunk_size',0) or getattr(args,'pca_chunk_size',1000) or 1000))
    # Pre-load topology once and pre-compute a reduced atom selection for the main
    # trajectory pass below: keep every protein atom (any element -- so the default
    # `--sasa-selection protein` and every other protein-only observable see exactly
    # the same protein atom population as before, byte for byte) plus every heavy
    # (non-hydrogen) atom system-wide. This drops only solvent/ion hydrogens -- the
    # vast majority of atoms in an explicit-solvent system, since protein is a tiny
    # fraction of total atom count -- while leaving every downstream consumer in the
    # loop below (phi/psi and DSSP via a protein-only atom_slice, SASA, internal
    # contacts, and _peptide_solvent_contact_counts, which already discards any
    # remaining hydrogens from its own bookkeeping) working on an unchanged atom set.
    # NOTE: excluding ALL hydrogens (including protein ones) was measured to change
    # total protein SASA by ~14% (a real physical difference from removing exposed-H
    # surface area, not float noise) -- unsafe. Restricting to solvent/ion hydrogens
    # only still cuts total atom count ~2-3x for a typical TIP3P-solvated system
    # while being provably SASA/DSSP/phi-psi/contact-count identical.
    extra_top_obj = None
    extra_atom_indices = None
    try:
        extra_top_obj = md.load(str(top_path))
        _extra_top = extra_top_obj.topology
        _extra_protein_idx = _extra_top.select('protein')
        _extra_heavy_idx = _extra_top.select('element != H')
        _extra_keep = np.union1d(_extra_protein_idx, _extra_heavy_idx)
        if 0 < _extra_keep.size < _extra_top.n_atoms:
            extra_atom_indices = _extra_keep
    except Exception as exc:
        warnings.append(f'Extra observable PMF atom pre-selection failed ({exc}); loading full system')
        extra_top_obj = None
        extra_atom_indices = None
    extra_iterload_top = extra_top_obj.topology if extra_top_obj is not None else str(top_path)
    sasa_selection=str(getattr(args,'sasa_selection','protein') or 'protein')
    sasa_points=int(getattr(args,'sasa_n_sphere_points',240) or 240)
    contact_cutoff=float(getattr(args,'contact_cutoff_nm',0.45) or 0.45)
    min_seq_sep=int(getattr(args,'contact_min_sequence_separation',3) or 3)
    contact_scheme=str(getattr(args,'contact_scheme','closest-heavy') or 'closest-heavy')
    # Label preflight: discover phi/psi/DSSP residue labels from 1 frame of first usable
    # replica. Avoids scanning the entire trajectory just for topology-derivable information.
    phi_labels=[]; psi_labels=[]; dssp_labels=[]
    for _item in plan:
        try:
            for _chunk in md.iterload(str(_item.get('traj',_item['dcd'])),top=str(top_path),chunk=1):
                _prot=_protein_traj(_chunk)
                if _prot is None: continue
                try:
                    _pidx,_=md.compute_phi(_prot)
                    phi_labels=[_md_residue_label(_prot.topology.atom(int(q[1])).residue) for q in _pidx]
                except Exception as exc:
                    warnings.append(f'Phi angle label discovery failed: {exc}')
                try:
                    _qidx,_=md.compute_psi(_prot)
                    psi_labels=[_md_residue_label(_prot.topology.atom(int(q[0])).residue) for q in _qidx]
                except Exception as exc:
                    warnings.append(f'Psi angle label discovery failed: {exc}')
                try:
                    _ss=md.compute_dssp(_prot,simplified=True)
                    dssp_labels=[_md_residue_label(r) for r in list(_prot.topology.residues)[:_ss.shape[1]]]
                except Exception:
                    pass
                break
        except Exception:
            pass
        if phi_labels or psi_labels or dssp_labels:
            break
    if progress is not None:
        _atom_note = (f'{extra_atom_indices.size}/{_extra_top.n_atoms} atoms (solvent/ion H excluded)'
                      if extra_atom_indices is not None else 'full system (atom pre-selection unavailable)')
        progress.step('extra PMFs', f'topology {top_path}; chunk={chunk_size}; atoms={_atom_note}; SASA selection={sasa_selection!r}')

    selected_method=_choose_method(selected,boost_ok)
    base_w_full=norm_logw(np.asarray(base_logw,dtype=np.float64))
    if boost_ok:
        exp_w_full=norm_logw(np.asarray(base_logw,dtype=np.float64)+d.beta*np.asarray(d.boost_kj,dtype=np.float64))
    else:
        exp_w_full=base_w_full.copy()
    torsion_bins=int(getattr(args,'torsion_bins',72) or 72)
    torsion_edges=np.linspace(-180.0,180.0,torsion_bins+1)
    scalar_bins=int(getattr(args,'observable_bins',None) or args.bins)
    ss_bins=int(getattr(args,'ss_bins',50) or 50)
    ss_edges=np.linspace(0.0,1.0,ss_bins+1)

    # Fixed-edge accumulators (phi/psi/DSSP/helix/strand/coil): build before main pass
    phi_accs={lab:_make_acc_1d(torsion_edges) for lab in phi_labels}
    psi_accs={lab:_make_acc_1d(torsion_edges) for lab in psi_labels}
    common_labels=[lab for lab in phi_labels if lab in set(psi_labels)]
    phi_pos={lab:i for i,lab in enumerate(phi_labels)}; psi_pos={lab:i for i,lab in enumerate(psi_labels)}
    rama_accs={lab:_make_acc_2d(torsion_edges,torsion_edges) for lab in common_labels}
    ss_accs={
        'helix_fraction':_make_acc_1d(ss_edges),
        'strand_fraction':_make_acc_1d(ss_edges),
        'coil_fraction':_make_acc_1d(ss_edges),
    }
    dssp_counts=None; dssp_valid_counts=None; dssp_total=0
    if dssp_labels:
        dssp_counts={code:np.zeros(len(dssp_labels),dtype=np.int64) for code in ('H','E','C')}
        dssp_valid_counts=np.zeros(len(dssp_labels),dtype=np.int64)

    # Per-sample arrays for convergence + basin tracking.  Indexed by MBAR sample
    # position (same ordering as d.cv / d.step).  NaN = not observed from trajectory.
    _N=len(d.cv)
    phi_samps: dict = {lab: np.full(_N,np.nan,dtype=np.float64) for lab in phi_labels}
    psi_samps: dict = {lab: np.full(_N,np.nan,dtype=np.float64) for lab in psi_labels}
    ss_samps: dict = {k: np.full(_N,np.nan,dtype=np.float64) for k in ('helix_fraction','strand_fraction','coil_fraction')}

    # Single trajectory pass: accumulate fixed-edge observables directly; buffer
    # range-unknown observables (SASA, contacts) for post-pass edge computation.
    # Eliminates original double scan where SASA/contact mdtraj calls were made
    # twice — once for range discovery, once for accumulation.
    buf_scalar=[]  # per-chunk dicts: sasa/cc/solv/wat/ion arrays + sidx
    assigned=0
    # Peptide/solvent atom classification depends only on the (fixed) topology, not
    # on any per-chunk trajectory data -- lazily computed once from the first real
    # chunk's own (possibly atom-index-reduced) topology below and reused for every
    # later call, instead of re-walking every atom on each of the thousands of
    # per-chunk calls a long trajectory pass makes.
    _solv_classification: dict = {}
    for pi,item in enumerate(plan, start=1):
        if progress is not None: progress.bar('extra PMF pass',pi,max(1,len(plan)),f"replica {item['replica']}")
        frame0=0
        frame_indices=np.asarray(item.get('frame_indices', np.arange(item['n_assign'])),dtype=np.int64)
        sample_indices=np.asarray(item.get('sample_indices', item['order']),dtype=np.int64)
        try:
            for chunk in md.iterload(str(item.get('traj', item['dcd'])),top=extra_iterload_top,chunk=chunk_size,atom_indices=extra_atom_indices):
                local, sample_idx, _global_frames = _chunk_local_frame_selection(frame_indices, sample_indices, frame0, chunk.n_frames)
                if local is None or local.size <= 0:
                    frame0+=chunk.n_frames
                    if frame0 > int(np.max(frame_indices)): break
                    continue
                n_use=int(local.size)
                full_use=chunk[local]
                prot=_protein_traj(full_use)
                if prot is None:
                    frame0+=chunk.n_frames; continue
                prot_use=prot
                phi_deg=None; psi_deg=None
                try:
                    _pidx,pa=md.compute_phi(prot_use)
                    phi_deg=_wrap_degrees(pa)
                    for lab,i in phi_pos.items():
                        if i < phi_deg.shape[1]:
                            _accumulate_1d(phi_accs[lab],phi_deg[:,i],sample_idx,d,base_w_full,exp_w_full)
                            phi_samps[lab][sample_idx]=phi_deg[:,i]
                except Exception:
                    phi_deg=None
                try:
                    _qidx,qa=md.compute_psi(prot_use)
                    psi_deg=_wrap_degrees(qa)
                    for lab,i in psi_pos.items():
                        if i < psi_deg.shape[1]:
                            _accumulate_1d(psi_accs[lab],psi_deg[:,i],sample_idx,d,base_w_full,exp_w_full)
                            psi_samps[lab][sample_idx]=psi_deg[:,i]
                except Exception:
                    psi_deg=None
                if phi_deg is not None and psi_deg is not None:
                    for lab in common_labels:
                        pi0=phi_pos[lab]; pj0=psi_pos[lab]
                        if pi0 < phi_deg.shape[1] and pj0 < psi_deg.shape[1]:
                            _accumulate_2d(rama_accs[lab],phi_deg[:,pi0],psi_deg[:,pj0],sample_idx,d,base_w_full,exp_w_full)
                ss_pack=_dssp_fraction_arrays(md,prot_use)
                if ss_pack is not None:
                    ss,helix,strand,coil=ss_pack
                    _accumulate_1d(ss_accs['helix_fraction'],helix,sample_idx,d,base_w_full,exp_w_full)
                    _accumulate_1d(ss_accs['strand_fraction'],strand,sample_idx,d,base_w_full,exp_w_full)
                    _accumulate_1d(ss_accs['coil_fraction'],coil,sample_idx,d,base_w_full,exp_w_full)
                    ss_samps['helix_fraction'][sample_idx]=helix
                    ss_samps['strand_fraction'][sample_idx]=strand
                    ss_samps['coil_fraction'][sample_idx]=coil
                    if dssp_counts is not None and ss.shape[1] == len(dssp_labels):
                        _n_h,_n_e,_n_coil,_n_valid=_dssp_valid_counts(ss,axis=0)
                        dssp_counts['H']+=_n_h.astype(np.int64)
                        dssp_counts['E']+=_n_e.astype(np.int64)
                        dssp_counts['C']+=_n_coil.astype(np.int64)
                        dssp_valid_counts+=_n_valid.astype(np.int64)
                        dssp_total+=int(ss.shape[0])
                cbuf={'sidx':np.asarray(sample_idx,dtype=np.int64).copy()}
                try:
                    cbuf['sasa']=_sasa_total_A2(md,prot_use,sasa_selection,sasa_points)
                except Exception as exc:
                    warnings.append(f'SASA computation failed for replica {item["replica"]}: {exc}')
                    cbuf['sasa']=np.full(n_use,np.nan,dtype=np.float64)
                try:
                    cbuf['cc']=_internal_contact_counts(md,prot_use,contact_cutoff,min_seq_sep,contact_scheme)
                except Exception as exc:
                    warnings.append(f'Contact computation failed for replica {item["replica"]}: {exc}')
                    cbuf['cc']=np.full(n_use,np.nan,dtype=np.float64)
                try:
                    if not _solv_classification:
                        _solv_classification.update(_build_peptide_solvent_classification(full_use.topology))
                    cbuf['solv'],cbuf['wat'],cbuf['ion']=_peptide_solvent_contact_counts(md,full_use,contact_cutoff,_solv_classification)
                except Exception as exc:
                    warnings.append(f'Solvent contact computation failed for replica {item["replica"]}: {exc}')
                    cbuf['solv']=np.full(n_use,np.nan,dtype=np.float64)
                    cbuf['wat']=np.full(n_use,np.nan,dtype=np.float64)
                    cbuf['ion']=np.full(n_use,np.nan,dtype=np.float64)
                buf_scalar.append(cbuf)
                assigned+=int(n_use)
                frame0+=chunk.n_frames
                if frame0 > int(np.max(frame_indices)): break
        except Exception as exc:
            warnings.append(f'Extra observable PMF pass failed for replica {item["replica"]}: {exc}')
    if progress is not None: progress.bar('extra PMF pass',1,1,f'accumulated {assigned} frames',force=True)
    if assigned <= 0:
        return {'available':False,'reason':'no trajectory frames scanned for extra observable PMFs'}

    # Post-pass: compute edges from buffered scalar values, build range-dependent
    # accumulators, then accumulate — no trajectory re-scan required.
    sasa_bins=int(getattr(args,'sasa_bins',None) or scalar_bins)
    contact_bins_arg=getattr(args,'contact_bins',None)
    def _buf_minmax(key):
        arrs=[b[key] for b in buf_scalar if key in b and b[key].size>0]
        if not arrs: return float('nan'),float('nan')
        vals=np.concatenate(arrs); finite=vals[np.isfinite(vals)]
        return (float(finite.min()),float(finite.max())) if finite.size>0 else (float('nan'),float('nan'))
    sasa_min,sasa_max=_buf_minmax('sasa')
    contact_min,contact_max=_buf_minmax('cc')
    solvent_max=_buf_minmax('solv')[1]
    water_max=_buf_minmax('wat')[1]
    ion_max=_buf_minmax('ion')[1]
    if np.isfinite(sasa_min) and np.isfinite(sasa_max) and sasa_max>sasa_min:
        sasa_lo=float(getattr(args,'sasa_min',None) if getattr(args,'sasa_min',None) is not None else sasa_min)
        sasa_hi=float(getattr(args,'sasa_max',None) if getattr(args,'sasa_max',None) is not None else sasa_max)
        if sasa_hi<=sasa_lo: sasa_hi=sasa_lo+1.0
        sasa_edges=np.linspace(sasa_lo,sasa_hi,sasa_bins+1)
    else:
        sasa_edges=None
    if np.isfinite(contact_max):
        cmax=max(1.0,float(contact_max))
        nb=int(contact_bins_arg or min(max(10,int(cmax)+1),max(10,scalar_bins)))
        contact_edges=np.linspace(-0.5,cmax+0.5,nb+1)
    else:
        contact_edges=None
    def _count_edges(max_value):
        if not np.isfinite(max_value): return None
        vmax=max(1.0,float(max_value))
        nb=int(contact_bins_arg or min(max(10,int(vmax)+1),max(10,scalar_bins)))
        return np.linspace(-0.5,vmax+0.5,nb+1)
    solvent_edges=_count_edges(solvent_max)
    water_edges=_count_edges(water_max)
    ion_edges=_count_edges(ion_max)
    scalar_accs={**ss_accs}
    if sasa_edges is not None: scalar_accs['sasa_A2']=_make_acc_1d(sasa_edges)
    if contact_edges is not None: scalar_accs['internal_contacts']=_make_acc_1d(contact_edges)
    if solvent_edges is not None: scalar_accs['peptide_solvent_contacts']=_make_acc_1d(solvent_edges)
    if water_edges is not None: scalar_accs['peptide_water_contacts']=_make_acc_1d(water_edges)
    if ion_edges is not None: scalar_accs['peptide_ion_contacts']=_make_acc_1d(ion_edges)
    contact2d_accs={}
    if contact_edges is not None and solvent_edges is not None: contact2d_accs['intra_vs_solvent_contacts']=_make_acc_2d(contact_edges,solvent_edges)
    if contact_edges is not None and water_edges is not None: contact2d_accs['intra_vs_water_contacts']=_make_acc_2d(contact_edges,water_edges)
    if contact_edges is not None and ion_edges is not None: contact2d_accs['intra_vs_ion_contacts']=_make_acc_2d(contact_edges,ion_edges)
    for cbuf in buf_scalar:
        sidx=cbuf['sidx']
        if 'sasa_A2' in scalar_accs:
            _accumulate_1d(scalar_accs['sasa_A2'],cbuf['sasa'],sidx,d,base_w_full,exp_w_full)
        if 'internal_contacts' in scalar_accs:
            _accumulate_1d(scalar_accs['internal_contacts'],cbuf['cc'],sidx,d,base_w_full,exp_w_full)
        if 'peptide_solvent_contacts' in scalar_accs:
            _accumulate_1d(scalar_accs['peptide_solvent_contacts'],cbuf['solv'],sidx,d,base_w_full,exp_w_full)
        if 'peptide_water_contacts' in scalar_accs:
            _accumulate_1d(scalar_accs['peptide_water_contacts'],cbuf['wat'],sidx,d,base_w_full,exp_w_full)
        if 'peptide_ion_contacts' in scalar_accs:
            _accumulate_1d(scalar_accs['peptide_ion_contacts'],cbuf['ion'],sidx,d,base_w_full,exp_w_full)
        if 'intra_vs_solvent_contacts' in contact2d_accs:
            _accumulate_2d(contact2d_accs['intra_vs_solvent_contacts'],cbuf['cc'],cbuf['solv'],sidx,d,base_w_full,exp_w_full)
        if 'intra_vs_water_contacts' in contact2d_accs:
            _accumulate_2d(contact2d_accs['intra_vs_water_contacts'],cbuf['cc'],cbuf['wat'],sidx,d,base_w_full,exp_w_full)
        if 'intra_vs_ion_contacts' in contact2d_accs:
            _accumulate_2d(contact2d_accs['intra_vs_ion_contacts'],cbuf['cc'],cbuf['ion'],sidx,d,base_w_full,exp_w_full)

    # Per-sample SASA and contact arrays reconstructed from buffered chunk data
    _sasa_samps=np.full(_N,np.nan,dtype=np.float64)
    _cc_samps=np.full(_N,np.nan,dtype=np.float64)
    for _cbuf in buf_scalar:
        _s=_cbuf['sidx']
        if 'sasa' in _cbuf: _sasa_samps[_s]=_cbuf['sasa']
        if 'cc' in _cbuf: _cc_samps[_s]=_cbuf['cc']

    files={}; summary={'available':True,'n_samples':int(assigned),'selected_unbiased_method':selected_method,'topology':str(top_path),'output_dir':str(out_extra),'files':files,'settings':{'sasa_selection':sasa_selection,'sasa_n_sphere_points':sasa_points,'contact_cutoff_nm':contact_cutoff,'contact_min_sequence_separation':min_seq_sep,'contact_scheme':contact_scheme}}
    scalar_summaries={}
    scalar_labels={'sasa_A2':('Solvent-accessible surface area PMF','SASA (A^2)'),'internal_contacts':('Internal contact-count PMF','number of internal contacts'),'peptide_solvent_contacts':('Peptide-solvent contact-count PMF','number of peptide-solvent contacts'),'peptide_water_contacts':('Peptide-water contact-count PMF','number of peptide-water contacts'),'peptide_ion_contacts':('Peptide-ion contact-count PMF','number of peptide-ion contacts'),'helix_fraction':('Helix fraction PMF','helix fraction'),'strand_fraction':('Beta-strand fraction PMF','strand fraction'),'coil_fraction':('Coil/other fraction PMF','coil/other fraction')}
    scalar_pmfs_cache: dict = {}
    for key,acc in scalar_accs.items():
        pmfs,_diag=_finalize_acc_1d(acc,d,kbt_kcal,smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
        scalar_pmfs_cache[key]=(pmfs,np.asarray(acc['edges'],dtype=np.float64))
        lab,xlab=scalar_labels.get(key,(key,key))
        info=_write_scalar_pmfs(out_extra,key,lab,xlab,pmfs,selected_method,warnings,smooth_sigma=_eff_smooth(args,'pmf_smooth_sigma'),args=args)
        scalar_summaries[key]=info
        files.update(info.get('files',{}))
    summary['scalar_pmfs']=scalar_summaries

    contact2d_dir=out_extra/'contact_2d_fes'; contact2d_dir.mkdir(exist_ok=True)
    contact2d_summary={}
    contact2d_meta={
        'intra_vs_solvent_contacts':('intra_vs_solvent_contacts','Internal peptide contacts vs peptide-solvent contacts 2D FES','internal peptide contacts','peptide-solvent contacts','internal_contacts','peptide_solvent_contacts'),
        'intra_vs_water_contacts':('intra_vs_water_contacts','Internal peptide contacts vs peptide-water contacts 2D FES','internal peptide contacts','peptide-water contacts','internal_contacts','peptide_water_contacts'),
        'intra_vs_ion_contacts':('intra_vs_ion_contacts','Internal peptide contacts vs peptide-ion contacts 2D FES','internal peptide contacts','peptide-ion contacts','internal_contacts','peptide_ion_contacts'),
    }
    for key,acc in contact2d_accs.items():
        pmfs2d,_diag=_finalize_acc_2d(acc,d,kbt_kcal,smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
        prefix,title,xlab,ylab,xfield,yfield=contact2d_meta[key]
        info=_write_generic_2d_fes(contact2d_dir,prefix,title+f' ({selected_method})',xlab,ylab,pmfs2d,selected_method,warnings,x_field=xfield,y_field=yfield,smooth_sigma=float(getattr(args,'fes2d_smooth_sigma',1.0)),args=args)
        contact2d_summary[key]=info
        files.update({f'{prefix}_2d_fes_{fk}':fv for fk,fv in info.get('files',{}).items()})
    if contact2d_summary:
        files['contact_2d_fes_dir']=str(contact2d_dir)
    summary['contact_2d_fes']=contact2d_summary

    torsion_dir=out_extra/'torsion_pmfs'; torsion_dir.mkdir(exist_ok=True)
    phi_rows=[]; psi_rows=[]; torsion_info={'phi_residues':len(phi_accs),'psi_residues':len(psi_accs),'ramachandran_residues':len(rama_accs),'ramachandran':[]}
    torsion_pmfs_cache: dict = {}
    for torsion_name,accs,rows in [('phi',phi_accs,phi_rows),('psi',psi_accs,psi_rows)]:
        for lab,acc in accs.items():
            pmfs,_diag=_finalize_acc_1d(acc,d,kbt_kcal,smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
            torsion_pmfs_cache[(torsion_name,lab)]=(pmfs,np.asarray(acc['edges'],dtype=np.float64))
            info=_write_scalar_pmfs(torsion_dir,f'{torsion_name}_{_slug(lab)}',f'{torsion_name.upper()} PMF {lab}',f'{torsion_name} (deg)',pmfs,selected_method,warnings,smooth_sigma=_eff_smooth(args,'pmf_smooth_sigma'),args=args)
            files.update(info.get('files',{}))
            for method,p in pmfs.items():
                for i,x in enumerate(p['x']):
                    rows.append({'torsion':torsion_name,'residue':lab,'method':method,'bin':i,'angle_deg':float(x),'probability':float(p['prob'][i]) if np.isfinite(p['prob'][i]) else '', 'pmf_kcal_mol':float(p['pmf'][i]) if np.isfinite(p['pmf'][i]) else '', 'counts':int(p['counts'][i])})
    if phi_rows: _write_csv_rows(out_extra/'phi_pmf_all_residues.csv',phi_rows); files['phi_pmf_all_residues_csv']=str(out_extra/'phi_pmf_all_residues.csv')
    if psi_rows: _write_csv_rows(out_extra/'psi_pmf_all_residues.csv',psi_rows); files['psi_pmf_all_residues_csv']=str(out_extra/'psi_pmf_all_residues.csv')
    rama_dir=out_extra/'ramachandran_2d_fes'; rama_dir.mkdir(exist_ok=True)
    for lab,acc in rama_accs.items():
        pmfs2d,_diag=_finalize_acc_2d(acc,d,kbt_kcal,smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
        rinfo=_write_rama_2d(rama_dir,lab,pmfs2d,selected_method,warnings,smooth_sigma=float(getattr(args,'fes2d_smooth_sigma',1.0)),args=args)
        torsion_info['ramachandran'].append(rinfo)
    files['ramachandran_2d_fes_dir']=str(rama_dir)
    summary['torsions']=torsion_info

    if dssp_counts is not None and dssp_total>0:
        rows=_dssp_residue_probability_rows(dssp_labels,dssp_counts,dssp_valid_counts,dssp_total)
        ss_path=out_extra/'secondary_structure_residue_probabilities.csv'
        _write_csv_rows(ss_path,rows); files['secondary_structure_residue_probabilities_csv']=str(ss_path)
        summary['secondary_structure_residue_count']=len(rows)
    else:
        summary['secondary_structure_residue_count']=0
    # Convergence + basin tracking for all extra observables
    if not bool(getattr(args,'no_convergence',False)):
        _do_basin=not bool(getattr(args,'no_basin_tracking',False))
        def _to_final_pmf(pmfs_dict,sel_meth):
            p=pmfs_dict.get(sel_meth,pmfs_dict.get('umbrella_only',next(iter(pmfs_dict.values()))))
            return {'cv_A':p['x'],'pmf':p['pmf'],'prob':p['prob'],'counts':p['counts']}
        _scalar_samps_map={'sasa_A2':_sasa_samps,'internal_contacts':_cc_samps,'helix_fraction':ss_samps['helix_fraction'],'strand_fraction':ss_samps['strand_fraction'],'coil_fraction':ss_samps['coil_fraction']}
        _scalar_label_map={'sasa_A2':('SASA','SASA (A^2)'),'internal_contacts':('internal contacts','number of contacts'),'helix_fraction':('helix fraction','helix fraction'),'strand_fraction':('strand fraction','strand fraction'),'coil_fraction':('coil/other fraction','coil/other fraction')}
        _extra_conv={}
        for _key,(_pmfs,_edges) in scalar_pmfs_cache.items():
            _samps=_scalar_samps_map.get(_key)
            if _samps is None or np.count_nonzero(np.isfinite(_samps))<10: continue
            _ml,_xl=_scalar_label_map.get(_key,(_key,_key))
            _conv=run_observable_pmf_convergence(d,args,_samps,_edges,selected,_to_final_pmf(_pmfs,selected_method),out_extra,
                metric_name=f'extra_{_key}',metric_label=_ml,x_label=_xl,
                out_dir_name=f'convergence/{_key}',file_prefix=_key,
                basin_tracking=_do_basin,progress=None)
            _extra_conv[_key]=_conv
            if isinstance(_conv,dict) and _conv.get('files'): files.update(_conv['files'])
        for _tname,_samps_dict in [('phi',phi_samps),('psi',psi_samps)]:
            for _lab,_samps in _samps_dict.items():
                if np.count_nonzero(np.isfinite(_samps))<10: continue
                _ckey=(_tname,_lab)
                if _ckey not in torsion_pmfs_cache: continue
                _pmfs,_edges=torsion_pmfs_cache[_ckey]
                _slug_lab=_slug(_lab)
                _conv=run_observable_pmf_convergence(d,args,_samps,_edges,selected,_to_final_pmf(_pmfs,selected_method),out_extra,
                    metric_name=f'{_tname}_{_slug_lab}',metric_label=f'{_tname.upper()} {_lab}',
                    x_label=f'{_tname} (deg)',
                    out_dir_name=f'convergence/{_tname}_{_slug_lab}',file_prefix=f'{_tname}_{_slug_lab}',
                    basin_tracking=_do_basin,progress=None)
                _extra_conv[f'{_tname}_{_slug_lab}']=_conv
        summary['extra_convergence']={str(k):v for k,v in _extra_conv.items()}

    wjson(out_extra/'extra_observable_pmfs_summary.json',summary)
    files['extra_observable_pmfs_summary_json']=str(out_extra/'extra_observable_pmfs_summary.json')
    return summary


def boost_stats(boost,beta):
    b=boost[np.isfinite(boost)]
    if b.size==0: return {'available':False}
    kc=b/KJ_PER_KCAL; out={'available':True,'n':int(b.size),'mean_kcal_mol':float(np.mean(kc)),'std_kcal_mol':float(np.std(kc)),'min_kcal_mol':float(np.min(kc)),'max_kcal_mol':float(np.max(kc))}
    if b.size>=3 and np.std(b)>0:
        z=(b-np.mean(b))/np.std(b); out['skew']=float(np.mean(z**3)); out['excess_kurtosis']=float(np.mean(z**4)-3.0); out['anharmonicity_score']=float(math.sqrt(out['skew']**2+0.25*out['excess_kurtosis']**2))
    else: out.update({'skew':None,'excess_kurtosis':None,'anharmonicity_score':None})
    w=norm_logw(beta*b); out['boost_reweight_ess']=float(ess(w)); out['boost_reweight_ess_fraction']=float(out['boost_reweight_ess']/b.size)
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



def _render_health_section_md(s):
    """Markdown lines for the result-health verdict (top of pmf_summary.md)."""
    try:
        from gareus_report import render_verdict_md
        return render_verdict_md(s.get('health'), s.get('warnings_grouped'))
    except Exception:
        return []


def _key_diagnostics_md(s):
    """Surface diagnostics that are otherwise buried in sub-directory files:
    conformational basins, Poincaré recurrence routes. Defensive throughout."""
    lines=[]
    try:
        bt=((s.get('convergence') or {}).get('basin_tracking')) or {}
        if bt.get('enabled') and bt.get('n_basins'):
            _cu=s.get('primary_cv_units','A')
            parts=[]
            for b in (bt.get('basins') or [])[:6]:
                lo=b.get('left_cv_A'); hi=b.get('right_cv_A'); ctr=b.get('center_cv_A')
                if lo is not None and hi is not None:
                    parts.append(f"[{float(lo):.2f}–{float(hi):.2f} {_cu}]")
                elif ctr is not None:
                    parts.append(f"~{float(ctr):.2f} {_cu}")
            det=(': '+', '.join(parts)) if parts else ''
            lines.append(f"- **Basins (CV1):** {int(bt['n_basins'])}{det}")
    except Exception:
        pass
    try:
        pm=s.get('poincare_map') or {}
        if pm.get('available'):
            fr=pm.get('fold_routes_summary'); ur=pm.get('unfold_routes_summary')
            frn=pm.get('fold_recurrence_median_ns'); urn=pm.get('unfold_recurrence_median_ns')
            if fr and str(fr)!='unknown':
                extra=f" (median recurrence {float(frn):.1f} ns)" if frn is not None else ""
                lines.append(f"- **Poincaré fold routes:** {fr}{extra}")
            if ur and str(ur)!='unknown':
                extra=f" (median recurrence {float(urn):.1f} ns)" if urn is not None else ""
                lines.append(f"- **Poincaré unfold routes:** {ur}{extra}")
    except Exception:
        pass
    return (['### Key diagnostics','']+lines+['']) if lines else []


def summary_md(path,s):
    _cu=s.get('primary_cv_units','A')
    lines=['# GaREUS PMF analysis summary','',f"Input: `{s['production_dir']}`",f"Samples/windows: **{s['n_samples']} / {s['n_windows']}**",f"Temperature: **{s['temperature_K']:.2f} K**",f"CV range: **{s['cv_min_A']:.3f} - {s['cv_max_A']:.3f} {_cu}**",f"Selected unbiased PMF: **{s['selected_unbiased_method']}**",f"PMF minimum: **{s['pmf_minimum_cv_A']} {_cu}**",f"PMF span: **{s['pmf_span_kcal_mol']:.3f} kcal/mol**",''] + _render_health_section_md(s) + _key_diagnostics_md(s) + ['## MBAR / umbrella diagnostics','',f"Converged: **{s['mbar']['converged']}** after {s['mbar']['iterations']} iterations",f"Backend: **{s['mbar'].get('backend','unknown')}**" + (f" / threads: **{s['mbar'].get('threads')}**" if s['mbar'].get('threads') else ""),f"Base ESS: **{s['mbar']['base_ess']:.1f}** / {s['n_samples']}", '', '## GaMD boost diagnostics','']
    b=s['boost']
    if b.get('available'):
        lines += [f"Boost mean/std: **{b['mean_kcal_mol']:.3f} / {b['std_kcal_mol']:.3f} kcal/mol**",f"Boost range: **{b['min_kcal_mol']:.3f} - {b['max_kcal_mol']:.3f} kcal/mol**",f"Anharmonicity score: **{b.get('anharmonicity_score')}**",f"Boost exponential ESS fraction: **{b.get('boost_reweight_ess_fraction',0):.3f}**"]
    else: lines.append('No finite variable GaMD boosts found; PMF is umbrella-only unbiased.')
    rg=s.get('rg',{}) or {}
    lines += ['', '## Radius of gyration diagnostics', '']
    if rg.get('available'):
        lines += [f"Selected Rg PMF method: **{rg.get('selected_unbiased_method')}**",f"Mean ± std Rg: **{float(rg.get('mean_A',float('nan'))):.3f} ± {float(rg.get('std_A',float('nan'))):.3f} Å**",f"Rg PMF minimum: **{rg.get('pmf_minimum_rg_A')} Å**",f"Rg PMF span: **{float(rg.get('pmf_span_kcal_mol',float('nan'))):.3f} kcal/mol**"]
    else:
        lines.append(f"Rg unavailable: {rg.get('reason','not computed')}")
    fes2d=s.get('distance_rg_2d_fes',{}) or {}
    lines += ['', '## Distance vs Rg 2D FES', '']
    if fes2d.get('available'):
        lines += [f"Selected 2D FES method: **{fes2d.get('selected_unbiased_method')}**",f"2D FES minimum: **distance {fes2d.get('pmf_minimum_cv_A')} Å, Rg {fes2d.get('pmf_minimum_rg_A')} Å**",f"2D FES span: **{float(fes2d.get('pmf_span_kcal_mol',float('nan'))):.3f} kcal/mol**",f"Grid: **{fes2d.get('cv_bins')} × {fes2d.get('rg_bins')}** bins",f"Normalization: **{fes2d.get('normalization','minimum shifted to 0')}**"]
    else:
        lines.append(f"2D FES unavailable: {fes2d.get('reason','not computed')}")
    pca=s.get('pca_2d_fes',{}) or {}
    lines += ['', '## PCA1 vs PCA2 2D FES', '']
    if pca.get('available'):
        lines += [f"Selected PCA 2D FES method: **{pca.get('selected_unbiased_method')}**",f"PCA 2D FES minimum: **PCA1 {pca.get('pmf_minimum_pca1_A')} Å, PCA2 {pca.get('pmf_minimum_pca2_A')} Å**",f"PCA 2D FES span: **{float(pca.get('pmf_span_kcal_mol',float('nan'))):.3f} kcal/mol**",f"Grid: **{pca.get('pca_bins')} × {pca.get('pca_bins')}** bins",f"Explained variance: **PC1 {float(pca.get('explained_variance_ratio_pc1',float('nan'))):.3f}, PC2 {float(pca.get('explained_variance_ratio_pc2',float('nan'))):.3f}**",f"Selection: **{pca.get('selection')}** ({pca.get('n_atoms')} atoms)",f"Normalization: **{pca.get('normalization','minimum shifted to 0')}**"]
    else:
        lines.append(f"PCA 2D FES unavailable: {pca.get('reason','not computed')}")
    extra=s.get('extra_observable_pmfs',{}) or {}
    lines += ['', '## Extra trajectory observable PMFs', '']
    if extra.get('available'):
        scalar=extra.get('scalar_pmfs',{}) or {}
        tors=extra.get('torsions',{}) or {}
        c2d=extra.get('contact_2d_fes',{}) or {}
        lines += [f"Selected method: **{extra.get('selected_unbiased_method')}**",f"Frames accumulated: **{extra.get('n_samples')}**",f"Scalar PMFs: **{', '.join(sorted(scalar.keys())) if scalar else 'none'}**",f"Contact 2D FES: **{', '.join(sorted(c2d.keys())) if c2d else 'none'}**",f"Torsion PMFs: **phi residues {tors.get('phi_residues',0)}, psi residues {tors.get('psi_residues',0)}, Ramachandran 2D residues {tors.get('ramachandran_residues',0)}**",f"Secondary-structure residue probabilities: **{extra.get('secondary_structure_residue_count',0)} residues**"]
    else:
        lines.append(f"Extra observable PMFs unavailable: {extra.get('reason','not computed')}")
    lines += ['', '## Outputs', ''] + [f"- `{k}`: `{v}`" for k,v in s['files'].items()] + ['', '## Warnings', '']
    lines += [f'- {w}' for w in s['warnings']] if s['warnings'] else ['- No major automatic warnings.']
    path.write_text('\n'.join(lines)+'\n')

CHIGNOLIN_FES_PLOT_VMAX_VALUES_KJ = (5.0, 10.0, 20.0, 40.0, None)


def _chignolin_fes_range_label_kj(vmax, actual_max: float) -> str:
    if vmax is None:
        if math.isfinite(float(actual_max)):
            return f'0-all kJ/mol (max {float(actual_max):.2f})'
        return '0-all kJ/mol'
    return f'0-{float(vmax):g} kJ/mol'


def _plot_chignolin_fes_kj_range(
    F, xedges, yedges, xc, yc, out_png, method: str, warnings: list,
    *, range_vmax=None, smooth_sigma=1.0, figsize=(8.8, 6.6), dpi=220
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.append(f"--chignolin_fes: matplotlib unavailable; no plot written: {exc}")
        return False
    F = np.asarray(F, dtype=np.float64)
    xedges = np.asarray(xedges, dtype=np.float64)
    yedges = np.asarray(yedges, dtype=np.float64)
    xc = np.asarray(xc, dtype=np.float64)
    yc = np.asarray(yc, dtype=np.float64)
    Fplot = _smooth_masked_grid(F, sigma=smooth_sigma)
    finite = np.isfinite(Fplot)
    if not np.any(finite):
        warnings.append("--chignolin_fes: no finite bins; plot skipped.")
        return False
    actual_max = float(np.nanmax(Fplot[finite]))
    if not np.isfinite(actual_max) or actual_max <= 0:
        actual_max = 1.0
    if range_vmax is None:
        vmax = max(1.0, actual_max)
        extend = 'neither'
    else:
        vmax = max(1.0e-12, float(range_vmax))
        extend = 'max' if actual_max > vmax else 'neither'
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("viridis").copy() if hasattr(plt.get_cmap("viridis"), "copy") else plt.get_cmap("viridis")
    try:
        cmap.set_bad(color="white", alpha=0.0)
    except Exception:
        pass
    im = ax.imshow(
        Fplot.T, origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto", interpolation="bicubic", cmap=cmap, vmin=0.0, vmax=vmax,
    )
    try:
        X, Y = np.meshgrid(xc, yc, indexing="ij")
        contour_src = np.where(np.isfinite(Fplot), Fplot, np.nan)
        levels = np.linspace(0.0, vmax, 11)
        if np.count_nonzero(np.isfinite(contour_src)) >= 9 and len(levels) > 2:
            cs = ax.contour(X, Y, contour_src, levels=levels[1:], colors="white", linewidths=0.7, alpha=0.75)
            ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")
    except Exception:
        pass
    finite_raw = np.isfinite(F)
    if np.any(finite_raw):
        min_idx = np.unravel_index(np.nanargmin(np.where(finite_raw, F, np.inf)), F.shape)
        ax.plot([xc[min_idx[0]]], [yc[min_idx[1]]], marker="*", markersize=11,
                markeredgecolor="black", markerfacecolor="gold", zorder=5)
    cbar = fig.colorbar(im, ax=ax, extend=extend)
    cbar.set_label("Free energy (kJ/mol, minimum shifted to 0)")
    ax.set_xlabel("dist(Asp3 N – Thr8 O) (Å)")
    ax.set_ylabel("dist(Asp3 N – Gly7 O) (Å)")
    ax.set_title(f"Chignolin contact 2D FES ({method}) [{_chignolin_fes_range_label_kj(range_vmax, actual_max)}]")
    ax.set_facecolor("white")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=dpi)
    plt.close(fig)
    return True


def _plot_chignolin_fes_kj_multirange(fes_kj: dict, out_png: Path, method: str, warnings: list) -> dict[str, str]:
    """Plot chignolin contact 2D FES in kJ/mol for multiple colorbar ranges."""
    F = np.asarray(fes_kj["pmf"], dtype=np.float64)
    xedges = np.asarray(fes_kj["cv_edges_A"], dtype=np.float64)
    yedges = np.asarray(fes_kj["rg_edges_A"], dtype=np.float64)
    xc = np.asarray(fes_kj["cv_A"], dtype=np.float64)
    yc = np.asarray(fes_kj["rg_A"], dtype=np.float64)
    files = {}
    main_vmax = 20.0
    main_path = None
    for vmax in CHIGNOLIN_FES_PLOT_VMAX_VALUES_KJ:
        tag = _fes_range_tag(vmax)
        path = _fes_variant_path(out_png, vmax)
        ok = _plot_chignolin_fes_kj_range(F, xedges, yedges, xc, yc, path, method, warnings, range_vmax=vmax)
        if ok:
            files[tag] = str(path)
            if vmax == main_vmax:
                main_path = path
    if main_path is not None:
        # The "main"/untagged file is byte-for-byte identical to the
        # main_vmax tagged variant (confirmed via SHA256) -- copy it instead
        # of re-running the whole render a second time from scratch.
        shutil.copyfile(main_path, out_png)
        files['main'] = str(out_png)
    return files


def _compute_chignolin_distances(d, args, progress, warnings: list):
    """Load replica DCD trajectories; return per-sample (dist1_A, dist2_A) in Å or (None, None)."""
    traj_dir = d.prod_dir / "replica_trajectories"
    if not traj_dir.exists():
        merged = _prepare_adaptive_merged_traj_dir(d, args)
        if merged is not None:
            traj_dir = merged
        else:
            warnings.append("--chignolin_fes: replica_trajectories/ not found.")
            return None, None
    top_path = _find_data_topology_path(d, args, ('rg_topology',))
    if top_path is None:
        warnings.append("--chignolin_fes: no topology PDB found; use --rg-topology.")
        return None, None
    try:
        import mdtraj as md
    except Exception as exc:
        warnings.append(f"--chignolin_fes: mdtraj unavailable ({exc})")
        return None, None
    try:
        ref = md.load(str(top_path))
        i_asp3_n = ref.topology.select("resSeq 3 and name N")
        i_thr8_o = ref.topology.select("resSeq 8 and name O")
        i_gly7_o = ref.topology.select("resSeq 7 and name O")
        if i_asp3_n.size == 0 or i_thr8_o.size == 0 or i_gly7_o.size == 0:
            raise ValueError(
                f"atom selection returned empty: Asp3N={i_asp3_n}, Thr8O={i_thr8_o}, Gly7O={i_gly7_o}; "
                "ensure topology matches chignolin residue numbering (resSeq 1-10)"
            )
        # Load only the 3 needed atoms — remapped pair indices for the atom-subset trajectory
        needed_atoms = np.array(sorted({int(i_asp3_n[0]), int(i_thr8_o[0]), int(i_gly7_o[0])}), dtype=np.int32)
        _amap = {int(a): i for i, a in enumerate(needed_atoms)}
        pair1 = np.array([[_amap[int(i_asp3_n[0])], _amap[int(i_thr8_o[0])]]], dtype=np.int32)
        pair2 = np.array([[_amap[int(i_asp3_n[0])], _amap[int(i_gly7_o[0])]]], dtype=np.int32)
    except Exception as exc:
        warnings.append(f"--chignolin_fes: atom selection failed: {exc}")
        return None, None
    dist1_out = np.full(d.cv.shape, np.nan, dtype=np.float64)
    dist2_out = np.full(d.cv.shape, np.nan, dtype=np.float64)
    reps = sorted(set(int(x) for x in d.replica if np.isfinite(x)))
    spf = _read_traj_interval(d.prod_dir)
    if spf <= 0 or spf == 500:  # 500 is the fallback default — check epoch dirs too
        epoch_spf = _read_traj_interval_from_epoch_dirs(d)
        if epoch_spf > 0:
            spf = epoch_spf
    # Use epoch-offset-adjusted steps when trajectories come from merged dir
    use_adjusted = (traj_dir != d.prod_dir / 'replica_trajectories')
    adj_steps = _adjusted_steps_for_merged_traj(d, spf) if use_adjusted else None
    # ref.topology reused — avoids re-parsing PDB for every segment
    top_topology = ref.topology

    def _chin_replica_worker(rep):
        local_warns: list = []
        segs = _find_all_replica_trajectory_segments(traj_dir, rep, args)
        if not segs:
            local_warns.append(f"--chignolin_fes: trajectory missing for replica {rep}: {_trajectory_pattern_hint(args)} in {traj_dir}")
            return 0, local_warns
        idx = np.where(d.replica == rep)[0]
        if idx.size == 0:
            return 0, local_warns
        order = idx[np.argsort(d.step[idx], kind="stable")]
        steps_for_align = adj_steps[order] if adj_steps is not None else d.step[order]
        rep_assigned = 0
        for resume_start, seg_path in segs:
            try:
                # atom_indices loads only 3 atoms — XTC codec skips rest at byte level
                traj = md.load(str(seg_path), top=top_topology, atom_indices=needed_atoms)
                d1 = md.compute_distances(traj, pair1).reshape(-1) * 10.0
                d2 = md.compute_distances(traj, pair2).reshape(-1) * 10.0
            except Exception as exc:
                local_warns.append(f"--chignolin_fes: distance failed for replica {rep} ({seg_path}): {exc}")
                continue
            mask, local_frames = _sample_to_segment_frame(steps_for_align, resume_start, traj.n_frames, spf)
            if not np.any(mask):
                continue
            dist1_out[order[mask]] = d1[local_frames]
            dist2_out[order[mask]] = d2[local_frames]
            rep_assigned += int(np.sum(mask))
        return rep_assigned, local_warns

    n_workers = max(1, int(getattr(args, 'traj_workers', 4) or 4))
    assigned = 0
    if n_workers > 1 and len(reps) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=n_workers) as exe:
            futs = {exe.submit(_chin_replica_worker, rep): rep for rep in reps}
            for ii, fut in enumerate(as_completed(futs), start=1):
                rep = futs[fut]
                if progress is not None:
                    progress.bar("chignolin distances", ii, max(1, len(reps)), f"replica {rep}")
                try:
                    n_a, w = fut.result()
                    assigned += n_a; warnings.extend(w)
                except Exception as exc:
                    warnings.append(f"--chignolin_fes: worker replica {rep} failed: {exc}")
    else:
        for ii, rep in enumerate(reps, start=1):
            if progress is not None:
                progress.bar("chignolin distances", ii, max(1, len(reps)), f"replica {rep}")
            n_a, w = _chin_replica_worker(rep)
            assigned += n_a; warnings.extend(w)

    if assigned <= 0:
        warnings.append("--chignolin_fes: no distances assigned; check topology and trajectories.")
        return None, None
    return dist1_out, dist2_out


def analyze_secondary_cv_pmf(d: Data, args, base_logw: np.ndarray, selected: str, boost_ok: bool, kbt_kcal: float, out: Path, warnings: list[str], progress: Optional[Progress]) -> dict:
    """1D PMF along the secondary collective variable (cv2_A from samples.csv)."""
    cv2=np.asarray(d.cv2, dtype=np.float64)
    mask=np.isfinite(cv2) & np.isfinite(base_logw)
    if np.count_nonzero(mask) < max(20, d.u_nk.shape[1]):
        return {'available': False, 'reason': 'Too few finite secondary CV (cv2_A) samples', 'n_finite': int(np.count_nonzero(mask))}
    cv2_sel=cv2[mask]
    boost_sel=d.boost_kj[mask]
    base_logw_sel=np.asarray(base_logw, dtype=np.float64)[mask]
    base_w=norm_logw(base_logw_sel)
    bins_n=int(getattr(args,'cv2_bins',None) or args.bins)
    bins=make_bins(cv2_sel, bins_n, getattr(args,'cv2_min',None), getattr(args,'cv2_max',None))
    umbrella=pmf_from_weights(cv2_sel, base_w, bins, kbt_kcal)
    if boost_ok and np.isfinite(boost_sel).sum()>10 and np.nanstd(boost_sel)>1e-12:
        exp_w=norm_logw(base_logw_sel + d.beta*boost_sel)
        exp_pmf=pmf_from_weights(cv2_sel, exp_w, bins, kbt_kcal)
        (cum_pmf,cdiag),(cum3_pmf,cdiag3)=_cumulant_expansion_both(cv2_sel, base_w, boost_sel, bins, d.beta, kbt_kcal, smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
        chosen=selected if selected in {'gamd_exponential','gamd_cumulant2','gamd_cumulant3'} else 'gamd_cumulant2'
    else:
        exp_pmf=umbrella; cum_pmf=umbrella; cum3_pmf=umbrella
        cdiag={'boost_mean_kj':np.full(len(bins)-1,np.nan),'boost_var_kj2':np.full(len(bins)-1,np.nan)}
        cdiag3=cdiag
        chosen='umbrella_only'
    pmfs={'umbrella_only':umbrella,'gamd_exponential':exp_pmf,'gamd_cumulant2':cum_pmf,'gamd_cumulant3':cum3_pmf}
    chosen_pmf=pmfs.get(chosen, umbrella)
    write_cv2_pmf(out/'cv2_pmf_unbiased.csv', chosen_pmf, chosen)
    write_cv2_pmf(out/'cv2_pmf_umbrella_only.csv', umbrella, 'umbrella_only')
    write_cv2_pmf(out/'cv2_pmf_gamd_exponential.csv', exp_pmf, 'gamd_exponential')
    write_cv2_pmf(out/'cv2_pmf_gamd_cumulant2.csv', cum_pmf, 'gamd_cumulant2')
    write_cv2_pmf(out/'cv2_pmf_gamd_cumulant3.csv', cum3_pmf, 'gamd_cumulant3')
    plot_file=None
    cv2_label=_secondary_cv_label(d.meta)
    regions=_secondary_cv_regions(d.meta)
    try:
        import matplotlib.pyplot as plt
        import gareus.mbar_analysis.plotstyle as ps
        fig,ax=plt.subplots(figsize=(8,5))
        _cv2_smooth=_eff_smooth(args,'pmf_smooth_sigma')
        for i,(name,p) in enumerate(_visible_pmfs(pmfs, chosen, args).items()):
            pmf_plot=_smooth_pmf_1d(p['pmf'],_cv2_smooth); m=np.isfinite(pmf_plot)
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
    cv2_conv=run_observable_pmf_convergence(
        d,args,d.cv2,bins,chosen,chosen_pmf,out,
        metric_name='secondary_cv',metric_label=cv2_label,x_label=cv2_label,
        out_dir_name='cv2_convergence',file_prefix='cv2',
        legacy_total_names=False,basin_tracking=True,progress=progress,
    )
    info={'available':True,'selected_unbiased_method':chosen,'n_samples':int(np.count_nonzero(mask)),'bins':int(len(bins)-1),'pmf_span_kcal_mol':span,'convergence':cv2_conv,'files':{'cv2_pmf_unbiased_csv':str(out/'cv2_pmf_unbiased.csv'),'cv2_pmf_umbrella_only_csv':str(out/'cv2_pmf_umbrella_only.csv'),'cv2_pmf_gamd_exponential_csv':str(out/'cv2_pmf_gamd_exponential.csv'),'cv2_pmf_gamd_cumulant2_csv':str(out/'cv2_pmf_gamd_cumulant2.csv'),'cv2_pmf_gamd_cumulant3_csv':str(out/'cv2_pmf_gamd_cumulant3.csv')}}
    if plot_file: info['files']['cv2_pmf_png']=plot_file
    if isinstance(cv2_conv,dict) and cv2_conv.get('files'):
        info['files'].update({k:v for k,v in cv2_conv['files'].items()})
    wjson(out/'cv2_pmf_summary.json', info)
    return info


def analyze_cv1_cv2_2d_fes(d: Data, args, base_logw: np.ndarray, selected: str, boost_ok: bool, kbt_kcal: float, out: Path, warnings: list[str], progress: Optional[Progress]) -> dict:
    """2D FES with CV1 (umbrella distance) on X and secondary CV (cv2_A) on Y."""
    cv2=np.asarray(d.cv2, dtype=np.float64)
    mask=np.isfinite(d.cv) & np.isfinite(cv2) & np.isfinite(base_logw)
    if np.count_nonzero(mask) < max(20, d.u_nk.shape[1]):
        return {'available': False, 'reason': 'Too few finite paired CV/CV2 samples for 2D FES', 'n_finite': int(np.count_nonzero(mask))}
    cv_bins_n=int(getattr(args,'fes2d_cv_bins',None) or args.bins)
    cv2_bins_n=int(getattr(args,'fes2d_cv2_bins',None) or getattr(args,'cv2_bins',None) or args.bins)
    xbins=make_bins(d.cv[mask], cv_bins_n, getattr(args,'cv_min',None), getattr(args,'cv_max',None))
    ybins=make_bins(cv2[mask], cv2_bins_n, getattr(args,'cv2_min',None), getattr(args,'cv2_max',None))
    cv_sel=d.cv[mask]; cv2_sel=cv2[mask]; boost_sel=d.boost_kj[mask]
    base_logw_sel=np.asarray(base_logw, dtype=np.float64)[mask]
    base_w=norm_logw(base_logw_sel)
    fes_umbrella=pmf2d_from_weights(cv_sel, cv2_sel, base_w, xbins, ybins, kbt_kcal)
    if boost_ok and np.isfinite(boost_sel).sum()>10 and np.nanstd(boost_sel)>1e-12:
        exp_logw=base_logw_sel + d.beta*boost_sel
        exp_w=norm_logw(exp_logw)
        fes_exp=pmf2d_from_weights(cv_sel, cv2_sel, exp_w, xbins, ybins, kbt_kcal)
        (fes_cum,cdiag),(fes_cum3,cdiag3)=_cumulant_expansion_2d_both(cv_sel, cv2_sel, base_w, boost_sel, xbins, ybins, d.beta, kbt_kcal, smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
        chosen=selected if selected in {'gamd_exponential','gamd_cumulant2','gamd_cumulant3'} else 'gamd_cumulant2'
    else:
        fes_exp=fes_umbrella; fes_cum=fes_umbrella; fes_cum3=fes_umbrella
        cdiag={'boost_mean_kj':np.full((len(xbins)-1,len(ybins)-1),np.nan),'boost_var_kj2':np.full((len(xbins)-1,len(ybins)-1),np.nan)}
        cdiag3=cdiag
        chosen='umbrella_only'
    chosen_fes=({'umbrella_only':fes_umbrella,'gamd_exponential':fes_exp,'gamd_cumulant2':fes_cum,'gamd_cumulant3':fes_cum3}).get(chosen,fes_umbrella)
    smooth_sigma=float(getattr(args,'fes2d_cv2_smooth_sigma',1.0))
    cv2_label=_secondary_cv_label(d.meta)
    regions=_secondary_cv_regions(d.meta)
    write_cv1_cv2_2d_fes_csv(out/'cv1_cv2_2d_fes_selected.csv', chosen_fes, chosen)
    write_cv1_cv2_2d_fes_npz(out/'cv1_cv2_2d_fes_selected.npz', chosen_fes, chosen)
    plot_files=plot_cv1_cv2_2d_fes(chosen_fes, chosen, out/'cv1_cv2_2d_fes_selected.png', f'CV1 vs {cv2_label} 2D FES ({chosen})', warnings, smooth_sigma=smooth_sigma, cv2_label=cv2_label, cv1_label=_primary_cv_axis_label(d.meta), regions=regions or None)
    write_cv1_cv2_2d_fes_csv(out/'cv1_cv2_2d_fes_cumulant2.csv', fes_cum, 'gamd_cumulant2')
    write_cv1_cv2_2d_fes_npz(out/'cv1_cv2_2d_fes_cumulant2.npz', fes_cum, 'gamd_cumulant2')
    plot_files_c2=plot_cv1_cv2_2d_fes(fes_cum, 'gamd_cumulant2', out/'cv1_cv2_2d_fes_cumulant2.png', f'CV1 vs {cv2_label} 2D FES (gamd_cumulant2)', warnings, smooth_sigma=smooth_sigma, cv2_label=cv2_label, cv1_label=_primary_cv_axis_label(d.meta), regions=regions or None)
    want_cum3=_want_gamd_method('gamd_cumulant3', chosen, args)
    plot_files_c3={}
    if want_cum3:
        write_cv1_cv2_2d_fes_csv(out/'cv1_cv2_2d_fes_cumulant3.csv', fes_cum3, 'gamd_cumulant3')
        write_cv1_cv2_2d_fes_npz(out/'cv1_cv2_2d_fes_cumulant3.npz', fes_cum3, 'gamd_cumulant3')
        plot_files_c3=plot_cv1_cv2_2d_fes(fes_cum3, 'gamd_cumulant3', out/'cv1_cv2_2d_fes_cumulant3.png', f'CV1 vs {cv2_label} 2D FES (gamd_cumulant3)', warnings, smooth_sigma=smooth_sigma, cv2_label=cv2_label, cv1_label=_primary_cv_axis_label(d.meta), regions=regions or None)
    finite=np.isfinite(chosen_fes['pmf'])
    if np.any(finite):
        min_idx=np.unravel_index(np.nanargmin(np.where(finite,chosen_fes['pmf'],np.inf)),chosen_fes['pmf'].shape)
        min_cv=float(chosen_fes['cv_A'][min_idx[0]]); min_cv2=float(chosen_fes['rg_A'][min_idx[1]])
        span=float(np.nanmax(chosen_fes['pmf'][finite])-np.nanmin(chosen_fes['pmf'][finite]))
    else:
        min_cv=min_cv2=span=float('nan')
    info={'available':True,'selected_unbiased_method':chosen,'n_samples':int(np.count_nonzero(mask)),'cv_bins':int(len(xbins)-1),'cv2_bins':int(len(ybins)-1),'pmf_minimum_cv_A':min_cv,'pmf_minimum_cv2_A':min_cv2,'pmf_span_kcal_mol':span,'normalization':'Minimum finite free energy shifted to 0 kcal/mol','files':{
        'cv1_cv2_2d_fes_csv':str(out/'cv1_cv2_2d_fes_selected.csv'),'cv1_cv2_2d_fes_npz':str(out/'cv1_cv2_2d_fes_selected.npz'),'cv1_cv2_2d_fes_png':str(out/'cv1_cv2_2d_fes_selected.png'),**{f'cv1_cv2_2d_fes_png_{k}':v for k,v in (plot_files or {}).items()},
        'cv1_cv2_2d_fes_cumulant2_csv':str(out/'cv1_cv2_2d_fes_cumulant2.csv'),'cv1_cv2_2d_fes_cumulant2_npz':str(out/'cv1_cv2_2d_fes_cumulant2.npz'),'cv1_cv2_2d_fes_cumulant2_png':str(out/'cv1_cv2_2d_fes_cumulant2.png'),**{f'cv1_cv2_2d_fes_cumulant2_png_{k}':v for k,v in (plot_files_c2 or {}).items()},
        **({'cv1_cv2_2d_fes_cumulant3_csv':str(out/'cv1_cv2_2d_fes_cumulant3.csv'),'cv1_cv2_2d_fes_cumulant3_npz':str(out/'cv1_cv2_2d_fes_cumulant3.npz'),'cv1_cv2_2d_fes_cumulant3_png':str(out/'cv1_cv2_2d_fes_cumulant3.png'),**{f'cv1_cv2_2d_fes_cumulant3_png_{k}':v for k,v in (plot_files_c3 or {}).items()}} if want_cum3 else {}),
    }}
    wjson(out/'cv1_cv2_2d_fes_summary.json', info)
    return info


def analyze_poincare_map(d: Data, args, base_logw: np.ndarray, selected: str, boost_ok: bool, kbt_kcal: float, out: Path, warnings: list[str], progress: Optional[Progress]) -> dict:
    """Poincaré return map analysis: backbone CV2 geometry at consecutive CV1 threshold crossings.

    Defines two Poincaré sections in CV1 (primary, contact-fraction) space:
      Σ_fold:   CV1 crosses c_fold upward   → entering folded/high-contact region
      Σ_unfold: CV1 crosses c_unfold downward → entering unfolded/extended region

    At each crossing, records CV2 (rama-map backbone score). Plots CV2_n vs CV2_{n+1}
    (consecutive crossings) to reveal pathway memory and distinct folding/unfolding routes.
    """
    if getattr(args, 'no_poincare_map', False):
        return {'available': False, 'reason': 'disabled via --no-poincare-map'}
    cv1 = np.asarray(d.cv, dtype=np.float64)
    cv2_raw = np.asarray(d.cv2, dtype=np.float64)
    # If no secondary CV, use CV1 itself as the observable recorded at crossings
    # (1-D first-return map: where in CV1 space does the trajectory land at each threshold crossing?)
    cv2_is_self = np.sum(np.isfinite(cv2_raw)) < 100
    cv2 = cv1.copy() if cv2_is_self else cv2_raw
    replica = np.asarray(d.replica, dtype=np.int32)
    step = np.asarray(d.step, dtype=np.int64) if d.step is not None and len(d.step) == len(cv1) else np.arange(len(cv1), dtype=np.int64)
    mask_valid = np.isfinite(cv1) & np.isfinite(cv2)
    if np.count_nonzero(mask_valid) < 100:
        return {'available': False, 'reason': 'Too few finite CV1 samples', 'n_finite': int(np.count_nonzero(np.isfinite(cv1)))}

    # Threshold selection
    c_fold_arg = getattr(args, 'poincare_fold_threshold', None)
    c_unfold_arg = getattr(args, 'poincare_unfold_threshold', None)
    cv1_finite = cv1[mask_valid]
    if c_fold_arg is not None:
        c_fold = float(c_fold_arg)
    else:
        # auto: use p90 of the biased distribution as fold entry threshold
        c_fold = float(np.percentile(cv1_finite, 90))
    if c_unfold_arg is not None:
        c_unfold = float(c_unfold_arg)
    else:
        c_unfold = float(np.percentile(cv1_finite, 5))

    min_seg = int(getattr(args, 'poincare_min_segment', 5))
    min_dwell = int(getattr(args, 'poincare_min_dwell', 50))
    timestep_fs = float(d.meta.get('timestep_fs', 4.0))
    # stride: infer from step differences per replica; fall back to 50 steps
    rep_ids = np.unique(replica)
    stride_ns = None
    for rid in rep_ids[:3]:
        rs = np.sort(step[replica == rid])
        if len(rs) > 2:
            med_stride = float(np.median(np.diff(rs)))
            if med_stride > 0:
                stride_ns = med_stride * timestep_fs * 1e-6
                break
    if stride_ns is None:
        stride_ns = 50 * timestep_fs * 1e-6

    def _find_crossings(cv1_r, cv2_r, threshold, direction):
        if direction > 0:
            cross = (cv1_r[:-1] < threshold) & (cv1_r[1:] >= threshold)
        else:
            cross = (cv1_r[:-1] > threshold) & (cv1_r[1:] <= threshold)
        idx = np.where(cross)[0] + 1
        # Dwell (commitment) filter: crossing only counts if CV1 stays on the
        # committed side for min_dwell consecutive frames after the crossing.
        # Eliminates threshold-bounce artifacts without removing genuine events.
        if min_dwell > 1 and len(idx) > 0:
            committed = []
            for ci in idx:
                end = min(ci + min_dwell, len(cv1_r))
                if direction > 0:
                    if np.all(cv1_r[ci:end] >= threshold):
                        committed.append(ci)
                else:
                    if np.all(cv1_r[ci:end] <= threshold):
                        committed.append(ci)
            idx = np.array(committed, dtype=np.int64) if committed else np.array([], dtype=np.int64)
        if len(idx) > 1:
            keep = np.concatenate([[True], np.diff(idx) >= min_seg])
            idx = idx[keep]
        ivs = np.diff(idx) * stride_ns if len(idx) > 1 else np.array([], dtype=np.float64)
        return cv2_r[idx], ivs, idx

    fold_cv2_per_rep, unfold_cv2_per_rep = [], []
    fold_iv_all, unfold_iv_all = [], []
    fold_step_per_rep, fold_rep_id_per_rep = [], []

    for rid in rep_ids:
        rmask = (replica == rid) & mask_valid
        if not np.any(rmask):
            continue
        ord_idx = np.argsort(step[rmask])
        cv1_r = cv1[rmask][ord_idx]
        cv2_r = cv2[rmask][ord_idx]
        step_r = step[rmask][ord_idx]
        cv2_f, iv_f, idx_f = _find_crossings(cv1_r, cv2_r, c_fold, +1)
        cv2_u, iv_u, idx_u = _find_crossings(cv1_r, cv2_r, c_unfold, -1)
        fold_cv2_per_rep.append(cv2_f)
        unfold_cv2_per_rep.append(cv2_u)
        fold_iv_all.append(iv_f)
        unfold_iv_all.append(iv_u)
        fold_step_per_rep.append(step_r[idx_f] if len(idx_f) else np.array([], dtype=np.int64))
        fold_rep_id_per_rep.append(np.full(len(idx_f), rid, dtype=np.int32))

    cv2_fold = np.concatenate(fold_cv2_per_rep) if fold_cv2_per_rep else np.array([])
    cv2_unfold = np.concatenate(unfold_cv2_per_rep) if unfold_cv2_per_rep else np.array([])
    iv_fold = np.concatenate(fold_iv_all) if fold_iv_all else np.array([])
    iv_unfold = np.concatenate(unfold_iv_all) if unfold_iv_all else np.array([])

    def _return_pairs(cv2_list):
        xs, ys = [], []
        for arr in cv2_list:
            if len(arr) > 1:
                xs.append(arr[:-1]); ys.append(arr[1:])
        return (np.concatenate(xs), np.concatenate(ys)) if xs else (np.array([]), np.array([]))

    x_fold, y_fold = _return_pairs(fold_cv2_per_rep)
    x_unfold, y_unfold = _return_pairs(unfold_cv2_per_rep)

    # Basin definitions from metadata (rama-map) or defaults
    regions = _secondary_cv_regions(d.meta)
    cv2_label = _secondary_cv_label(d.meta)
    default_basins = [
        {'name': 'β/extended', 'cv2': -1.0,    'color': '#2c7bb6'},
        {'name': 'PPII/coil',  'cv2': -1/3,    'color': '#74add1'},
        {'name': 'right-α',   'cv2': +1/3,    'color': '#f46d43'},
        {'name': 'left-α',    'cv2': +1.0,    'color': '#d73027'},
    ]
    basins = default_basins
    if regions:
        try:
            basins = [{'name': r.get('label', r.get('name', '')),
                       'cv2': float(r['value']),
                       'color': default_basins[i % 4]['color']}
                      for i, r in enumerate(regions) if 'value' in r]
        except Exception:
            basins = default_basins

    def _nearest_basin(cv2_val):
        return min(basins, key=lambda b: abs(b['cv2'] - cv2_val))

    # CSV output
    files: dict = {}
    try:
        fold_rows = []
        for rep_arr, step_arr, cv2_arr in zip(fold_rep_id_per_rep, fold_step_per_rep, fold_cv2_per_rep):
            for ri, si, c2 in zip(rep_arr, step_arr, cv2_arr):
                fold_rows.append({'replica': int(ri), 'step': int(si), 'cv2': float(c2)})
        _write_csv_rows(out / 'poincare_fold_crossings.csv', fold_rows)
        files['poincare_fold_crossings_csv'] = str(out / 'poincare_fold_crossings.csv')
    except Exception as e:
        warnings.append(f'Poincaré fold crossings CSV failed: {e}')

    unfold_rows = []
    try:
        unf_steps_rep = []
        for rid in rep_ids:
            rmask = (replica == rid) & mask_valid
            if not np.any(rmask):
                continue
            ord_idx = np.argsort(step[rmask])
            cv1_r = cv1[rmask][ord_idx]; cv2_r = cv2[rmask][ord_idx]; step_r = step[rmask][ord_idx]
            _, _, idx_u = _find_crossings(cv1_r, cv2_r, c_unfold, -1)
            for si, c2 in zip(step_r[idx_u] if len(idx_u) else [], cv2_r[idx_u] if len(idx_u) else []):
                unfold_rows.append({'replica': int(rid), 'step': int(si), 'cv2': float(c2)})
        _write_csv_rows(out / 'poincare_unfold_crossings.csv', unfold_rows)
        files['poincare_unfold_crossings_csv'] = str(out / 'poincare_unfold_crossings.csv')
    except Exception as e:
        warnings.append(f'Poincaré unfold crossings CSV failed: {e}')

    # KDE peak detection
    fold_peaks, unfold_peaks = [], []
    try:
        from scipy.stats import gaussian_kde
        from scipy.signal import argrelmax as _argrelmax
        def _kde_peaks(data, bw=0.12, n=400):
            if len(data) < 10:
                return []
            kde = gaussian_kde(data, bw_method=bw)
            lo = float(np.nanpercentile(data, 1)); hi = float(np.nanpercentile(data, 99))
            pad = max((hi - lo) * 0.1, 1e-6)
            xg = np.linspace(lo - pad, hi + pad, n)
            dens = kde(xg)
            pk = _argrelmax(dens, order=10)[0]
            return sorted([(float(xg[i]), float(dens[i])) for i in pk], key=lambda t: -t[1])
        fold_peaks = _kde_peaks(cv2_fold) if len(cv2_fold) >= 10 else []
        unfold_peaks = _kde_peaks(cv2_unfold) if len(cv2_unfold) >= 10 else []
    except Exception:
        pass

    # Build plain-language summary
    fold_route_str = 'unknown'
    if len(fold_peaks) >= 2:
        b1 = _nearest_basin(fold_peaks[0][0]); b2 = _nearest_basin(fold_peaks[1][0])
        fold_route_str = (f"2 routes: '{b1['name']}' (CV2≈{fold_peaks[0][0]:+.2f}) "
                          f"and '{b2['name']}' (CV2≈{fold_peaks[1][0]:+.2f})")
    elif len(fold_peaks) == 1:
        b1 = _nearest_basin(fold_peaks[0][0])
        fold_route_str = f"1 route: '{b1['name']}' (CV2≈{fold_peaks[0][0]:+.2f})"
    unfold_route_str = 'unknown'
    if len(unfold_peaks) >= 1:
        bu = _nearest_basin(unfold_peaks[0][0])
        unfold_route_str = f"1 route: '{bu['name']}' (CV2≈{unfold_peaks[0][0]:+.2f})"

    # Annotated plot
    # Dynamic CV2 plot range (data-driven; not hardcoded [-1,1])
    cv2_all = np.concatenate([cv2_fold, cv2_unfold]) if (len(cv2_fold) + len(cv2_unfold)) > 0 else cv2[mask_valid]
    if len(cv2_all) > 0:
        cv2_lo = float(np.nanpercentile(cv2_all, 1)); cv2_hi = float(np.nanpercentile(cv2_all, 99))
    else:
        cv2_lo = float(np.nanpercentile(cv2[mask_valid], 1)); cv2_hi = float(np.nanpercentile(cv2[mask_valid], 99))
    cv2_pad = max((cv2_hi - cv2_lo) * 0.08, 0.05)
    cv2_lo -= cv2_pad; cv2_hi += cv2_pad
    # Section names: "fold/unfold" for contact CVs, "high/low" for others
    is_contact_cv = _poincare_primary_cv_supported(d.meta)
    sname_high = 'fold'  if is_contact_cv else 'high'
    sname_low  = 'unfold' if is_contact_cv else 'low'
    cv1_xlabel = _primary_cv_axis_label(d.meta)
    # Whether to draw rama-map basin shading (only when cv2 has known region annotations)
    use_basin_shading = bool(regions) and not cv2_is_self

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as mgrid
        FOLD_C = '#c0392b'; UNFOLD_C = '#1a6b3c'; BG = '#f9f9f9'; PBG = '#ffffff'

        fig = plt.figure(figsize=(18, 13))
        fig.patch.set_facecolor(BG)
        gs = mgrid.GridSpec(3, 3, figure=fig, height_ratios=[1.1, 2.4, 0.9],
                            hspace=0.5, wspace=0.38, left=0.07, right=0.97, top=0.93, bottom=0.04)

        def _pbg(ax):
            ax.set_facecolor(PBG)
            for sp in ax.spines.values(): sp.set_edgecolor('#cccccc')

        def _blines(ax, ori='v'):
            if not use_basin_shading: return
            for b in basins:
                fn = ax.axvline if ori == 'v' else ax.axhline
                fn(b['cv2'], color=b['color'], lw=0.9, ls='--', alpha=0.45)

        def _bbg(ax, alpha=0.07):
            if not use_basin_shading: return
            edges = sorted(set([cv2_lo] + [round((basins[i]['cv2'] + basins[i+1]['cv2'])/2, 4) for i in range(len(basins)-1)] + [cv2_hi]))
            for i, b in enumerate(basins):
                lo = edges[i] if i < len(edges) else cv2_lo
                hi = edges[i+1] if i+1 < len(edges) else cv2_hi
                ax.axvspan(lo, hi, color=b['color'], alpha=alpha, zorder=0)
                ax.axhspan(lo, hi, color=b['color'], alpha=alpha, zorder=0)

        # Row 0: CV1 distribution
        ax0 = fig.add_subplot(gs[0, :])
        _pbg(ax0)
        cv1_fin = cv1[mask_valid]
        ax0.hist(cv1_fin, bins=300, density=True, color='#7f8c8d', alpha=0.6, linewidth=0)
        ax0.axvspan(0, c_unfold, color=UNFOLD_C, alpha=0.10)
        ax0.axvspan(c_fold, float(np.nanmax(cv1_fin)) * 1.05, color=FOLD_C, alpha=0.10)
        ax0.axvline(c_unfold, color=UNFOLD_C, lw=2.2, ls='--',
                    label=f'Σ_{sname_low}  CV1={c_unfold:.3f}  (downward crossings counted here)')
        ax0.axvline(c_fold, color=FOLD_C, lw=2.2, ls='--',
                    label=f'Σ_{sname_high}   CV1={c_fold:.3f}   (upward crossings counted here)')
        ax0.set_xlabel(f'CV1: {cv1_xlabel}', fontsize=10)
        ax0.set_ylabel('Density (REUS-biased)', fontsize=9)
        ax0.set_title('CV1 distribution — dashed lines are the two Poincaré sections\n'
                      'Each time the trajectory crosses a dashed line, CV2 (backbone geometry) is recorded',
                      fontsize=10, fontweight='bold')
        ax0.legend(fontsize=9, loc='upper right')
        ax0.set_xlim(-0.005, float(np.nanmax(cv1_fin)) * 1.08)

        # Row 1 col 0: basin legend (or CV info when no basin regions)
        ax_leg = fig.add_subplot(gs[1, 0])
        ax_leg.set_facecolor(PBG); ax_leg.axis('off')
        if use_basin_shading and basins:
            ax_leg.set_title('CV2 reference\nregions', fontsize=10, fontweight='bold')
            for i, b in enumerate(basins):
                y = 3.5 - i * (3.5 / max(1, len(basins) - 1)) if len(basins) > 1 else 1.75
                ax_leg.add_patch(plt.matplotlib.patches.FancyBboxPatch(
                    (-1.2, y - 0.3), 2.4, 0.6, boxstyle='round,pad=0.05',
                    fc=b['color'], alpha=0.15, ec=b['color'], lw=1.2))
                ax_leg.text(-1.1, y, b['name'], fontsize=10, va='center',
                            fontweight='bold', color=b['color'])
                ax_leg.text(0.5, y, f"CV2={b['cv2']:+.3f}", fontsize=9, va='center', color=b['color'])
            ax_leg.set_xlim(-1.3, 1.3); ax_leg.set_ylim(-0.5, 4.5)
        else:
            ax_leg.set_title('Poincaré sections', fontsize=10, fontweight='bold')
            info_txt = (
                f'CV1 = {cv1_xlabel}\n\n'
                f'Σ_{sname_high}: CV1 ↑ {c_fold:.4g}\n'
                f'(upward crossings)\n\n'
                f'Σ_{sname_low}: CV1 ↓ {c_unfold:.4g}\n'
                f'(downward crossings)\n\n'
                + (f'CV2 = {cv2_label}' if not cv2_is_self else 'CV2 = CV1\n(no secondary CV;\n1-D return map)')
            )
            ax_leg.text(0.5, 0.5, info_txt, transform=ax_leg.transAxes,
                        fontsize=9, va='center', ha='center',
                        bbox=dict(boxstyle='round,pad=0.5', fc='#f0f4ff', ec='#3498db', lw=1.2))

        # Row 1 col 1: Σ_high/fold return map
        ax_f = fig.add_subplot(gs[1, 1]); _pbg(ax_f); _bbg(ax_f)
        if len(x_fold) > 0:
            h, xe, ye = np.histogram2d(x_fold, y_fold, bins=65, range=[[cv2_lo,cv2_hi],[cv2_lo,cv2_hi]])
            ax_f.imshow(np.log1p(h).T, origin='lower', extent=[cv2_lo,cv2_hi,cv2_lo,cv2_hi],
                        aspect='auto', cmap='Reds', alpha=0.85, interpolation='bilinear')
            ax_f.scatter(x_fold, y_fold, s=2, alpha=0.07, color='#7b241c', rasterized=True, zorder=2)
        ax_f.plot([cv2_lo,cv2_hi],[cv2_lo,cv2_hi],'k--',lw=1,alpha=0.45,zorder=3,label='diagonal: perfect memory')
        _blines(ax_f,'v'); _blines(ax_f,'h')
        ax_f.set_xlim(cv2_lo,cv2_hi); ax_f.set_ylim(cv2_lo,cv2_hi)
        _cv2_obs_label = cv2_label if not cv2_is_self else cv1_xlabel
        ax_f.set_xlabel(f'{_cv2_obs_label} at {sname_high} event #n', fontsize=9)
        ax_f.set_ylabel(f'{_cv2_obs_label} at {sname_high} event #n+1', fontsize=9)
        ax_f.set_title(f'Σ_{sname_high} return map  ({len(x_fold):,} pairs)\n'
                       f'Each dot = two consecutive times trajectory\nreached CV1 > {c_fold:.3g}',
                       fontsize=9, fontweight='bold')
        for cv2_p, dens_p in fold_peaks[:2]:
            if dens_p > 0.2:
                nb = _nearest_basin(cv2_p)
                ax_f.annotate(f"≈{nb['name']}\n{cv2_p:+.2f}",
                              xy=(cv2_p, cv2_p), xytext=(cv2_p - 0.4, cv2_p + 0.3),
                              fontsize=7.5, color=FOLD_C, fontweight='bold',
                              arrowprops=dict(arrowstyle='->', color=FOLD_C, lw=1),
                              bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=FOLD_C, alpha=0.85), zorder=5)
        ax_f.text(-0.97, 0.96, 'Tight cluster on diagonal\n= same backbone geometry\nevent after event\n(pathway memory)',
                  fontsize=7, va='top', style='italic',
                  bbox=dict(boxstyle='round,pad=0.25', fc='#fff9e6', ec='#ccaa00', alpha=0.9),
                  transform=ax_f.transData, zorder=6)
        ax_f.legend(fontsize=7, loc='lower right')

        # Row 1 col 2: Σ_low/unfold return map
        ax_u = fig.add_subplot(gs[1, 2]); _pbg(ax_u); _bbg(ax_u)
        if len(x_unfold) > 0:
            h2, _, _ = np.histogram2d(x_unfold, y_unfold, bins=65, range=[[cv2_lo,cv2_hi],[cv2_lo,cv2_hi]])
            ax_u.imshow(np.log1p(h2).T, origin='lower', extent=[cv2_lo,cv2_hi,cv2_lo,cv2_hi],
                        aspect='auto', cmap='Greens', alpha=0.85, interpolation='bilinear')
            ax_u.scatter(x_unfold, y_unfold, s=2, alpha=0.07, color='#145a32', rasterized=True, zorder=2)
        ax_u.plot([cv2_lo,cv2_hi],[cv2_lo,cv2_hi],'k--',lw=1,alpha=0.45,zorder=3)
        _blines(ax_u,'v'); _blines(ax_u,'h')
        ax_u.set_xlim(cv2_lo,cv2_hi); ax_u.set_ylim(cv2_lo,cv2_hi)
        ax_u.set_xlabel(f'{_cv2_obs_label} at {sname_low} event #n', fontsize=9)
        ax_u.set_ylabel(f'{_cv2_obs_label} at {sname_low} event #n+1', fontsize=9)
        ax_u.set_title(f'Σ_{sname_low} return map  ({len(x_unfold):,} pairs)\n'
                       f'Each dot = two consecutive times trajectory\nreached CV1 < {c_unfold:.3g}',
                       fontsize=9, fontweight='bold')
        for cv2_p, dens_p in unfold_peaks[:1]:
            if dens_p > 0.2:
                nb = _nearest_basin(cv2_p)
                ax_u.annotate(f"≈{nb['name']}\n{cv2_p:+.2f}",
                              xy=(cv2_p, cv2_p), xytext=(cv2_p + 0.1, cv2_p - 0.4),
                              fontsize=7.5, color=UNFOLD_C, fontweight='bold',
                              arrowprops=dict(arrowstyle='->', color=UNFOLD_C, lw=1),
                              bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=UNFOLD_C, alpha=0.85), zorder=5)
        ax_u.text(-0.97, 0.96, 'Single cluster on diagonal\n= one stereotyped backbone\nunfolding route',
                  fontsize=7, va='top', style='italic',
                  bbox=dict(boxstyle='round,pad=0.25', fc='#e8f8ee', ec=UNFOLD_C, alpha=0.9),
                  transform=ax_u.transData, zorder=6)

        # Row 2: CV2 marginals (col 0+1) and summary (col 2)
        ax_hist = fig.add_subplot(gs[2, :2]); _pbg(ax_hist)
        bins_cv2 = np.linspace(cv2_lo, cv2_hi, 55)
        if use_basin_shading and basins:
            basin_edges_h = sorted(set([cv2_lo] + [round((basins[i]['cv2'] + basins[i+1]['cv2'])/2, 4) for i in range(len(basins)-1)] + [cv2_hi]))
            for i, b in enumerate(basins):
                lo2 = basin_edges_h[i] if i < len(basin_edges_h) else cv2_lo
                hi2 = basin_edges_h[i+1] if i+1 < len(basin_edges_h) else cv2_hi
                ax_hist.axvspan(lo2, hi2, color=b['color'], alpha=0.07)
        if len(cv2_fold) > 0:
            ax_hist.hist(cv2_fold, bins=bins_cv2, density=True, alpha=0.65, color=FOLD_C,
                         label=f'at Σ_{sname_high} crossings (n={len(cv2_fold):,})')
        if len(cv2_unfold) > 0:
            ax_hist.hist(cv2_unfold, bins=bins_cv2, density=True, alpha=0.65, color=UNFOLD_C,
                         label=f'at Σ_{sname_low} crossings (n={len(cv2_unfold):,})')
        if use_basin_shading and basins:
            for b in basins:
                ax_hist.axvline(b['cv2'], color=b['color'], lw=1.3, ls='--', alpha=0.6)
                ax_hist.text(b['cv2'], 0, b['name'], rotation=90, ha='center', va='bottom',
                             fontsize=6.5, color=b['color'], fontweight='bold',
                             transform=ax_hist.get_xaxis_transform())
        ax_hist.set_xlabel(f'{_cv2_obs_label}  (value at crossing)', fontsize=9)
        ax_hist.set_ylabel('Density', fontsize=9)
        ax_hist.set_title(f'Observable distribution at each crossing type', fontsize=9, fontweight='bold')
        ax_hist.set_xlim(cv2_lo, cv2_hi); ax_hist.legend(fontsize=8)

        ax_sum = fig.add_subplot(gs[2, 2]); ax_sum.axis('off')
        med_f_ps = float(np.median(iv_fold) * 1000) if len(iv_fold) > 0 else float('nan')
        med_u_ps = float(np.median(iv_unfold) * 1000) if len(iv_unfold) > 0 else float('nan')
        dwell_ps = min_dwell * stride_ns * 1000
        summary_txt = (
            'KEY FINDINGS\n\n'
            f'Σ_{sname_high} routes:  {fold_route_str}\n'
            f'Σ_{sname_low} route: {unfold_route_str}\n\n'
            f'Σ_{sname_high} crossings: {len(cv2_fold):,}  (median {med_f_ps:.1f} ps apart)\n'
            f'Σ_{sname_low} crossings:  {len(cv2_unfold):,}  (median {med_u_ps:.1f} ps apart)\n\n'
            f'Dwell filter: {min_dwell} frames = {dwell_ps:.1f} ps\n'
            '(only committed entries counted)\n\n'
            'Return map on diagonal\n→ strong pathway memory\n\n'
            'Note: recurrence times reflect\nREUS bias, not physical rates'
        )
        ax_sum.text(0.04, 0.96, summary_txt, transform=ax_sum.transAxes,
                    fontsize=8.5, va='top', ha='left', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', fc='#f0f4ff', ec='#3498db', lw=1.3))

        n_rep = len(rep_ids)
        fig.suptitle(
            f'Poincaré map — pathway analysis via CV1 threshold crossings\n'
            f'{n_rep} replicas  |  {int(np.count_nonzero(mask_valid)):,} frames  |  '
            f'Σ_{sname_high} CV1↑{c_fold:.4g}  |  Σ_{sname_low} CV1↓{c_unfold:.4g}',
            fontsize=11, fontweight='bold', y=0.97)
        fig.savefig(out / 'poincare_map.png', dpi=150, bbox_inches='tight', facecolor=BG)
        plt.close(fig)
        files['poincare_map_png'] = str(out / 'poincare_map.png')
    except Exception as e:
        warnings.append(f'Poincaré map plot failed: {e}')

    info = {
        'available': True,
        'c_fold': c_fold,
        'c_unfold': c_unfold,
        'min_dwell_frames': min_dwell,
        'min_dwell_ps': float(min_dwell * stride_ns * 1000),
        'n_fold_crossings': int(len(cv2_fold)),
        'n_unfold_crossings': int(len(cv2_unfold)),
        'n_fold_return_pairs': int(len(x_fold)),
        'n_unfold_return_pairs': int(len(x_unfold)),
        'fold_recurrence_median_ns': float(np.median(iv_fold)) if len(iv_fold) > 0 else None,
        'unfold_recurrence_median_ns': float(np.median(iv_unfold)) if len(iv_unfold) > 0 else None,
        'fold_peaks': [{'cv2': float(c), 'density': float(r), 'nearest_basin': _nearest_basin(c)['name']} for c, r in fold_peaks[:4]],
        'unfold_peaks': [{'cv2': float(c), 'density': float(r), 'nearest_basin': _nearest_basin(c)['name']} for c, r in unfold_peaks[:4]],
        'fold_routes_summary': fold_route_str,
        'unfold_routes_summary': unfold_route_str,
        'files': files,
    }
    wjson(out / 'poincare_map_summary.json', info)
    files['poincare_map_summary_json'] = str(out / 'poincare_map_summary.json')
    return info


def _poincare_torsions_chunked(md, seg_path, top, atom_indices, chunk_size: int, needed_frames):
    """Compute phi/psi (radians) at specific known frame indices via chunked iterload.

    Poincare crossing-event analysis only needs a handful of specific frames per
    segment (the committed fold/unfold crossing events), not the whole segment.
    This reads `chunk_size` frames at a time and computes phi/psi immediately for
    any needed frame found in that chunk, discarding the chunk (and any larger
    per-frame Trajectory it would otherwise pin in memory) right away -- only the
    small per-frame angle arrays are retained, never a whole chunk's coordinates.

    `needed_frames` are 0-based local frame indices within this segment. Returns
    {frame_idx: (phi_rad, psi_rad)} for every requested index actually found in
    the segment; missing/out-of-range indices are simply absent from the result,
    matching the original whole-load-then-index behavior this replaces.
    """
    needed = {int(fi) for fi in needed_frames if int(fi) >= 0}
    result: dict = {}
    if not needed:
        return result
    frame0 = 0
    for chunk in md.iterload(str(seg_path), top=top, chunk=chunk_size, atom_indices=atom_indices):
        chunk_n = chunk.n_frames
        local_hits = [fi - frame0 for fi in needed if frame0 <= fi < frame0 + chunk_n]
        if local_hits:
            _, phi_all = md.compute_phi(chunk)
            _, psi_all = md.compute_psi(chunk)
            for local_idx in local_hits:
                fi = frame0 + local_idx
                result[fi] = (phi_all[local_idx].copy(), psi_all[local_idx].copy())
                needed.discard(fi)
        frame0 += chunk_n
        if not needed:
            break
    return result

def analyze_poincare_residue_torsions(d: Data, args, out: Path, poincare_info: dict, warnings: list, progress: Optional[Progress]) -> dict:
    """Per-residue backbone torsion (phi/psi) analysis at committed Poincare fold/unfold crossing frames.

    Loads the actual trajectory frames at each crossing step, computes phi/psi angles with MDTraj,
    and produces a Ramachandran-per-residue figure coloured by folding route (A vs B split on CV2).
    """
    # --- Guard clauses ---
    if getattr(args, 'no_poincare_residue_torsions', False):
        return {'available': False, 'reason': 'disabled via --no-poincare-residue-torsions'}
    if not isinstance(poincare_info, dict) or not poincare_info.get('available'):
        return {'available': False, 'reason': 'poincare_info not available'}
    if int(poincare_info.get('n_fold_crossings', 0)) < 5:
        return {'available': False, 'reason': f'Too few fold crossings ({poincare_info.get("n_fold_crossings",0)}) for torsion analysis'}

    try:
        import mdtraj as md
    except ImportError:
        return {'available': False, 'reason': 'MDTraj not available'}

    top_path = _find_data_topology_path(d, args, ('rg_topology',))
    if top_path is None:
        return {'available': False, 'reason': 'topology PDB not found; use --rg-topology'}

    fold_csv = out / 'poincare_fold_crossings.csv'
    unfold_csv = out / 'poincare_unfold_crossings.csv'
    if not fold_csv.exists():
        return {'available': False, 'reason': f'poincare_fold_crossings.csv not found at {fold_csv}'}
    if not unfold_csv.exists():
        return {'available': False, 'reason': f'poincare_unfold_crossings.csv not found at {unfold_csv}'}

    # --- Load crossing CSVs ---
    def _load_crossing_csv(path: Path) -> list:
        rows = []
        try:
            with path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        rows.append({'replica': int(row['replica']), 'step': int(row['step']), 'cv2': float(row['cv2'])})
                    except (KeyError, ValueError):
                        pass
        except Exception:
            pass
        return rows

    fold_rows = _load_crossing_csv(fold_csv)
    unfold_rows = _load_crossing_csv(unfold_csv)

    # --- Route split CV2 cutpoint ---
    route_split = getattr(args, 'poincare_route_split', None)
    if route_split is not None:
        route_split = float(route_split)
    else:
        peaks = poincare_info.get('fold_peaks', [])
        if len(peaks) >= 2:
            # Sort by density descending, take top 2, midpoint of their cv2 values
            top2 = sorted(peaks, key=lambda p: -p['density'])[:2]
            route_split = float((top2[0]['cv2'] + top2[1]['cv2']) / 2.0)
        else:
            route_split = 0.0

    # Split fold events into Route A (cv2 < route_split) and Route B (cv2 >= route_split)
    fold_A = [r for r in fold_rows if r['cv2'] < route_split]
    fold_B = [r for r in fold_rows if r['cv2'] >= route_split]

    # --- Prepare trajectory loading ---
    traj_dir = d.prod_dir / 'replica_trajectories'
    if not traj_dir.exists():
        merged = _prepare_adaptive_merged_traj_dir(d, args)
        if merged is not None:
            traj_dir = merged
        else:
            return {'available': False, 'reason': f'replica_trajectories directory not found at {traj_dir}'}

    traj_interval = _read_traj_interval(d.prod_dir)
    if traj_interval <= 0:
        traj_interval = 50  # fallback

    # Load protein atom indices once from topology
    try:
        _top_traj = md.load(str(top_path))
        protein_indices = _top_traj.topology.select('protein')
        if protein_indices.size == 0:
            return {'available': False, 'reason': 'topology protein selection returned zero atoms'}
        protein_top = _top_traj.atom_slice(protein_indices).topology
        # Get residue names for subplots (from protein-sliced topology, not full system)
        residue_names = [f'{res.name.capitalize()}{res.resSeq}' for res in protein_top.residues]
        n_residues = protein_top.n_residues
    except Exception as exc:
        return {'available': False, 'reason': f'topology loading failed: {exc}'}

    # Determine phi/psi residue mapping from a small test frame.
    # compute_phi returns N-1 angles (no phi for first residue),
    # compute_psi returns N-1 angles (no psi for last residue).
    # Map each angle column to its residue index via MDTraj atom indices.
    try:
        test_frame = md.load_frame(str(top_path), 0, top=str(top_path))
        test_prot = test_frame.atom_slice(protein_indices)
        phi_atom_indices, _ = md.compute_phi(test_prot)
        psi_atom_indices, _ = md.compute_psi(test_prot)
        # phi: [C(i-1), N(i), CA(i), C(i)] — residue i is the owner of N(i) = atom index 1
        phi_col_to_resid = [protein_top.atom(int(a[1])).residue.index for a in phi_atom_indices]
        # psi: [N(i), CA(i), C(i), N(i+1)] — residue i is the owner of N(i) = atom index 0
        psi_col_to_resid = [protein_top.atom(int(a[0])).residue.index for a in psi_atom_indices]
    except Exception as exc:
        return {'available': False, 'reason': f'phi/psi residue mapping failed: {exc}'}

    # --- Batch frame loading by (replica, segment_path) ---
    all_crossings = (
        [('fold_A', r) for r in fold_A] +
        [('fold_B', r) for r in fold_B] +
        [('unfold', r) for r in unfold_rows]
    )

    if not all_crossings:
        return {'available': False, 'reason': 'no crossing events to analyze'}

    # Build map: replica -> segments list
    replica_segments: dict = {}
    all_reps = set(r['replica'] for _, r in all_crossings)
    for rep_id in all_reps:
        segs = _find_all_replica_trajectory_segments(traj_dir, rep_id, args)
        if segs:
            replica_segments[rep_id] = segs  # [(start_step, path), ...]

    def _find_segment_for_step(segs, step):
        """Return (seg_start, seg_path, frame_idx) for the segment containing `step`."""
        for i, (seg_start, seg_path) in enumerate(segs):
            next_start = segs[i + 1][0] if i + 1 < len(segs) else float('inf')
            if seg_start < step <= next_start:
                frame_idx = _saved_frame_index_from_step(step, seg_start, traj_interval)
                return seg_start, seg_path, frame_idx
        return None, None, -1

    # Group crossings by (replica, segment_path) to batch-load each file once
    from collections import defaultdict
    seg_batches: dict = defaultdict(list)
    skipped = 0
    for event_type, crossing in all_crossings:
        rep_id = crossing['replica']
        segs = replica_segments.get(rep_id)
        if not segs:
            skipped += 1
            continue
        seg_start, seg_path, frame_idx = _find_segment_for_step(segs, crossing['step'])
        if seg_path is None or frame_idx < 0:
            skipped += 1
            continue
        seg_batches[(rep_id, str(seg_path))].append((event_type, crossing, frame_idx))

    if not seg_batches:
        return {'available': False, 'reason': f'no trajectory segments found for any crossing replica (skipped {skipped})'}

    # Load each segment once (via chunked iterload -- only the specific frames
    # needed are ever computed/retained, never a whole segment at once), extract
    # needed frames.
    # results: list of (event_type, crossing_dict, phi_deg_aligned, psi_deg_aligned)
    results: list = []
    poincare_chunk_size = max(1, int(getattr(args, 'pca_chunk_size', 1000) or 1000))

    for (rep_id, seg_path_str), batch_items in seg_batches.items():
        seg_path = Path(seg_path_str)
        needed_frames = {int(frame_idx) for _et, _c, frame_idx in batch_items}
        try:
            angles_by_frame = _poincare_torsions_chunked(md, seg_path, str(top_path), protein_indices, poincare_chunk_size, needed_frames)
        except Exception as exc:
            warnings.append(f'Poincare torsions: failed to load segment {seg_path.name} for replica {rep_id}: {exc}')
            continue

        for event_type, crossing, frame_idx in batch_items:
            hit = angles_by_frame.get(int(frame_idx))
            if hit is None:
                continue
            try:
                phi_rad, psi_rad = hit
                phi_deg = np.degrees(phi_rad)  # shape (n_phi_cols,)
                psi_deg = np.degrees(psi_rad)  # shape (n_psi_cols,)
                # Scatter into per-residue aligned arrays (terminals stay NaN)
                phi_aligned = np.full(n_residues, np.nan)
                psi_aligned = np.full(n_residues, np.nan)
                for k, ri in enumerate(phi_col_to_resid):
                    if k < len(phi_deg) and 0 <= ri < n_residues:
                        phi_aligned[ri] = phi_deg[k]
                for k, ri in enumerate(psi_col_to_resid):
                    if k < len(psi_deg) and 0 <= ri < n_residues:
                        psi_aligned[ri] = psi_deg[k]
                results.append((event_type, crossing, phi_aligned, psi_aligned))
            except Exception:
                continue

    if not results:
        return {'available': False, 'reason': 'no frames successfully loaded from any crossing event'}

    # --- Build per-route arrays ---
    def _collect(event_tag):
        rows_et = [(c, ph, ps) for (et, c, ph, ps) in results if et == event_tag]
        if not rows_et:
            return np.empty((0, n_residues)), np.empty((0, n_residues)), []
        phi_arr = np.stack([r[1] for r in rows_et])
        psi_arr = np.stack([r[2] for r in rows_et])
        crossings_list = [r[0] for r in rows_et]
        return phi_arr, psi_arr, crossings_list

    fold_A_phi, fold_A_psi, fold_A_crossings = _collect('fold_A')
    fold_B_phi, fold_B_psi, fold_B_crossings = _collect('fold_B')
    unfold_phi, unfold_psi, unfold_crossings = _collect('unfold')

    n_A = len(fold_A_crossings)
    n_B = len(fold_B_crossings)
    n_unfold_cnt = len(unfold_crossings)
    n_fold = n_A + n_B

    # --- Save CSV ---
    csv_rows = []
    for et, c_list, ph_arr, ps_arr in [('fold_A', fold_A_crossings, fold_A_phi, fold_A_psi),
                                        ('fold_B', fold_B_crossings, fold_B_phi, fold_B_psi),
                                        ('unfold', unfold_crossings, unfold_phi, unfold_psi)]:
        for i, c in enumerate(c_list):
            row: dict = {'event_type': et, 'replica': c['replica'], 'step': c['step'], 'cv2': c['cv2']}
            if i < len(ph_arr):
                for ri in range(n_residues):
                    row[f'phi_{ri}'] = float(ph_arr[i, ri]) if np.isfinite(ph_arr[i, ri]) else ''
                    row[f'psi_{ri}'] = float(ps_arr[i, ri]) if np.isfinite(ps_arr[i, ri]) else ''
            csv_rows.append(row)
    try:
        _write_csv_rows(out / 'poincare_residue_torsions.csv', csv_rows)
    except Exception as exc:
        warnings.append(f'Poincare torsions CSV write failed: {exc}')

    # --- Compute mean/std differences for bar charts ---
    def _mean_std(arr):
        """Per-residue mean and std, ignoring NaN."""
        if arr.shape[0] == 0:
            return np.full(n_residues, np.nan), np.full(n_residues, np.nan)
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

    phi_A_mean, phi_A_std = _mean_std(fold_A_phi)
    phi_B_mean, phi_B_std = _mean_std(fold_B_phi)
    psi_A_mean, psi_A_std = _mean_std(fold_A_psi)
    psi_B_mean, psi_B_std = _mean_std(fold_B_psi)

    dphi = phi_B_mean - phi_A_mean  # Route B minus Route A
    dpsi = psi_B_mean - psi_A_mean
    # Propagated std (assuming independence)
    dphi_err = np.sqrt(np.where(np.isfinite(phi_A_std), phi_A_std**2, 0.0) +
                       np.where(np.isfinite(phi_B_std), phi_B_std**2, 0.0))
    dpsi_err = np.sqrt(np.where(np.isfinite(psi_A_std), psi_A_std**2, 0.0) +
                       np.where(np.isfinite(psi_B_std), psi_B_std**2, 0.0))

    # Most different residue
    def _most_diff_res(darr):
        abs_d = np.abs(darr)
        if not np.any(np.isfinite(abs_d)):
            return 'unknown'
        return residue_names[int(np.nanargmax(abs_d))]

    most_diff_psi = _most_diff_res(dpsi)
    most_diff_phi = _most_diff_res(dphi)

    # Nearest basin helper using poincare_info fold_peaks
    def _nb(cv2_val):
        if not isinstance(poincare_info.get('fold_peaks'), list) or not poincare_info['fold_peaks']:
            return 'unknown'
        peaks = poincare_info['fold_peaks']
        return min(peaks, key=lambda p: abs(p['cv2'] - cv2_val)).get('nearest_basin', 'unknown')

    dwell_ps = float(poincare_info.get('min_dwell_ps', 0.0))

    # --- Figure ---
    png_path = None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D

        fig = plt.figure(figsize=(22, 14))
        fig.patch.set_facecolor('#f0f0f0')

        # Outer grid: 2 rows, ratio [2.2, 1]
        outer_gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2.2, 1.0], hspace=0.35)

        # Top sub-grid: 2 rows x 5 cols for Ramachandran subplots
        top_gs = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=outer_gs[0], hspace=0.45, wspace=0.35)

        # Bottom sub-grid: 1 row x 5 cols; panels span (0-1), (2-3), (4)
        bot_gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer_gs[1], wspace=0.4)

        # Ramachandran basin background: (phi_min, phi_max, psi_min, psi_max, color)
        rama_basins = [
            (-160, -90, 100, 180, '#d0e8ff'),   # beta/extended — pale blue
            (-100, -50, 120, 180, '#ccf5f5'),    # PPII — pale cyan
            (-90,  -30, -80,   0, '#ffe8cc'),    # right-alpha — pale orange
            ( 30,   90,  10,  70, '#ffd5d5'),    # left-alpha — pale red
        ]

        # Proxy handles for legend — always show all three routes regardless
        # of which residue subplot is drawn first (N-terminal lacks phi).
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                   markersize=6, alpha=0.7, label='Unfold'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=6, alpha=0.7, label=f'Route A (CV2<{route_split:.2f})'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                   markersize=6, alpha=0.7, label=f'Route B (CV2≥{route_split:.2f})'),
        ]
        legend_placed = False

        n_subplots = min(n_residues, 10)  # 2x5 = max 10 subplots
        for idx in range(n_subplots):
            row_i, col_i = divmod(idx, 5)
            ax = fig.add_subplot(top_gs[row_i, col_i])
            ax.set_facecolor('white')
            for pb_phi0, pb_phi1, pb_psi0, pb_psi1, pb_col in rama_basins:
                ax.add_patch(mpatches.Rectangle(
                    (pb_phi0, pb_psi0), pb_phi1 - pb_phi0, pb_psi1 - pb_psi0,
                    color=pb_col, zorder=0, transform=ax.transData))
            ax.axhline(0, color='gray', lw=0.5, zorder=1)
            ax.axvline(0, color='gray', lw=0.5, zorder=1)

            phi_A_i = fold_A_phi[:, idx] if fold_A_phi.shape[0] > 0 else np.array([])
            psi_A_i = fold_A_psi[:, idx] if fold_A_psi.shape[0] > 0 else np.array([])
            phi_B_i = fold_B_phi[:, idx] if fold_B_phi.shape[0] > 0 else np.array([])
            psi_B_i = fold_B_psi[:, idx] if fold_B_psi.shape[0] > 0 else np.array([])
            phi_U_i = unfold_phi[:, idx] if unfold_phi.shape[0] > 0 else np.array([])
            psi_U_i = unfold_psi[:, idx] if unfold_psi.shape[0] > 0 else np.array([])

            valid_A = np.isfinite(phi_A_i) & np.isfinite(psi_A_i)
            valid_B = np.isfinite(phi_B_i) & np.isfinite(psi_B_i)
            valid_U = np.isfinite(phi_U_i) & np.isfinite(psi_U_i)

            if np.any(valid_U):
                ax.scatter(phi_U_i[valid_U], psi_U_i[valid_U],
                           c='green', alpha=0.3, s=12, zorder=3)
            if np.any(valid_A):
                ax.scatter(phi_A_i[valid_A], psi_A_i[valid_A],
                           c='red', alpha=0.5, s=20, zorder=4)
            if np.any(valid_B):
                ax.scatter(phi_B_i[valid_B], psi_B_i[valid_B],
                           c='blue', alpha=0.5, s=20, zorder=5)

            # Place legend on first subplot that has any data, using proxy handles
            if not legend_placed and (np.any(valid_A) or np.any(valid_B) or np.any(valid_U)):
                ax.legend(handles=legend_handles, fontsize=5,
                          loc='upper right', framealpha=0.7)
                legend_placed = True

            ax.set_xlim(-180, 180)
            ax.set_ylim(-180, 180)
            ax.set_title(residue_names[idx] if idx < len(residue_names) else f'Res{idx}', fontsize=8)
            ax.tick_params(labelsize=6)
            ax.set_xlabel('φ (°)', fontsize=6)
            ax.set_ylabel('ψ (°)', fontsize=6)

        # --- Bottom panels ---
        x_pos = np.arange(n_residues)

        # Panel A: Delta-psi bar chart (cols 0-1)
        ax_psi = fig.add_subplot(bot_gs[0, 0:2])
        colors_psi = ['red' if (np.isfinite(v) and v > 0) else 'blue' for v in dpsi]
        ax_psi.bar(x_pos, np.where(np.isfinite(dpsi), dpsi, 0.0), color=colors_psi, alpha=0.75, zorder=2)
        ax_psi.errorbar(x_pos, np.where(np.isfinite(dpsi), dpsi, 0.0),
                        yerr=np.where(np.isfinite(dpsi_err), dpsi_err, 0.0),
                        fmt='none', color='black', capsize=3, lw=1, zorder=3)
        ax_psi.axhline(0, color='black', lw=1.0)
        ax_psi.set_xticks(x_pos)
        ax_psi.set_xticklabels(residue_names, rotation=45, ha='right', fontsize=7)
        ax_psi.set_ylabel('Δψ (°)', fontsize=9)
        ax_psi.set_title('Which residues drive the two folding routes? (ψ difference)\nRoute B − Route A', fontsize=9)
        ax_psi.set_facecolor('white')

        # Panel B: Delta-phi bar chart (cols 2-3)
        ax_phi = fig.add_subplot(bot_gs[0, 2:4])
        colors_phi = ['red' if (np.isfinite(v) and v > 0) else 'blue' for v in dphi]
        ax_phi.bar(x_pos, np.where(np.isfinite(dphi), dphi, 0.0), color=colors_phi, alpha=0.75, zorder=2)
        ax_phi.errorbar(x_pos, np.where(np.isfinite(dphi), dphi, 0.0),
                        yerr=np.where(np.isfinite(dphi_err), dphi_err, 0.0),
                        fmt='none', color='black', capsize=3, lw=1, zorder=3)
        ax_phi.axhline(0, color='black', lw=1.0)
        ax_phi.set_xticks(x_pos)
        ax_phi.set_xticklabels(residue_names, rotation=45, ha='right', fontsize=7)
        ax_phi.set_ylabel('Δφ (°)', fontsize=9)
        ax_phi.set_title('φ difference: Route B − Route A', fontsize=9)
        ax_phi.set_facecolor('white')

        # Panel C: summary text box (col 4)
        ax_txt = fig.add_subplot(bot_gs[0, 4])
        ax_txt.axis('off')
        mean_cv2_A = float(np.mean([c['cv2'] for c in fold_A_crossings])) if fold_A_crossings else float('nan')
        mean_cv2_B = float(np.mean([c['cv2'] for c in fold_B_crossings])) if fold_B_crossings else float('nan')
        mean_cv2_U = float(np.mean([c['cv2'] for c in unfold_crossings])) if unfold_crossings else float('nan')
        nb_A = _nb(mean_cv2_A) if np.isfinite(mean_cv2_A) else 'unknown'
        nb_B = _nb(mean_cv2_B) if np.isfinite(mean_cv2_B) else 'unknown'
        nb_U = _nb(mean_cv2_U) if np.isfinite(mean_cv2_U) else 'unknown'
        summary_text = (
            f'Route A (CV2 < {route_split:.2f})\n'
            f'  n = {n_A}\n'
            f'  nearest basin: {nb_A}\n'
            f'  mean CV2: {mean_cv2_A:.3f}\n\n'
            f'Route B (CV2 >= {route_split:.2f})\n'
            f'  n = {n_B}\n'
            f'  nearest basin: {nb_B}\n'
            f'  mean CV2: {mean_cv2_B:.3f}\n\n'
            f'Unfold\n'
            f'  n = {n_unfold_cnt}\n'
            f'  nearest basin: {nb_U}\n'
            f'  mean CV2: {mean_cv2_U:.3f}\n\n'
            f'Most diff. residue\n'
            f'  psi: {most_diff_psi}\n'
            f'  phi: {most_diff_phi}\n\n'
            f'Route split CV2: {route_split:.3f}'
        )
        ax_txt.text(0.05, 0.95, summary_text, transform=ax_txt.transAxes,
                    fontsize=8, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        fig.suptitle(
            f'Per-residue backbone geometry at committed Poincaré crossing events\n'
            f'{n_fold} fold (A:{n_A}, B:{n_B}) + {n_unfold_cnt} unfold committed crossings'
            f'  |  dwell≥{dwell_ps:.0f} ps',
            fontsize=12, fontweight='bold'
        )

        png_path = out / 'poincare_residue_torsions.png'
        fig.savefig(str(png_path), dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
    except Exception as exc:
        warnings.append(f'Poincare residue torsions plot failed: {exc}')
        png_path = None

    files: dict = {}
    if png_path is not None and Path(png_path).exists():
        files['poincare_residue_torsions_png'] = str(png_path)
    csv_out = out / 'poincare_residue_torsions.csv'
    if csv_out.exists():
        files['poincare_residue_torsions_csv'] = str(csv_out)

    return {
        'available': True,
        'route_split_cv2': route_split,
        'n_route_A': n_A,
        'n_route_B': n_B,
        'n_unfold': n_unfold_cnt,
        'most_different_residue_psi': most_diff_psi,
        'most_different_residue_phi': most_diff_phi,
        'files': files,
    }


def analyze_chignolin_fes(d, args, base_logw: np.ndarray, selected: str, boost_ok: bool, kbt_kcal: float, out: Path, warnings: list, progress) -> dict:
    """2D FES: X=dist(Asp3N-Thr8O), Y=dist(Asp3N-Gly7O), energy in kJ/mol, 0-20 kJ/mol range."""
    if not getattr(args, "chignolin_fes", False):
        return {"available": False, "reason": "not requested"}
    dist1, dist2 = _compute_chignolin_distances(d, args, progress, warnings)
    if dist1 is None:
        return {"available": False, "reason": "distance computation failed; see warnings"}
    mask = np.isfinite(d.cv) & np.isfinite(dist1) & np.isfinite(dist2) & np.isfinite(base_logw)
    if np.count_nonzero(mask) < max(20, d.u_nk.shape[1]):
        return {"available": False, "reason": "too few finite samples", "n_finite": int(np.count_nonzero(mask))}
    n_bins = int(getattr(args, "bins", 80))
    xbins = make_bins(dist1[mask], n_bins, None, None)
    ybins = make_bins(dist2[mask], n_bins, None, None)
    x_sel = dist1[mask]; y_sel = dist2[mask]
    boost_sel = d.boost_kj[mask]
    logw_sel = np.asarray(base_logw, dtype=np.float64)[mask]
    base_w = norm_logw(logw_sel)
    fes_umbrella = pmf2d_from_weights(x_sel, y_sel, base_w, xbins, ybins, kbt_kcal)
    if boost_ok and np.isfinite(boost_sel).sum() > 10 and np.nanstd(boost_sel) > 1e-12:
        exp_w = norm_logw(logw_sel + d.beta * boost_sel)
        fes_exp = pmf2d_from_weights(x_sel, y_sel, exp_w, xbins, ybins, kbt_kcal)
        (fes_cum, _), (fes_cum3, _) = _cumulant_expansion_2d_both(x_sel, y_sel, base_w, boost_sel, xbins, ybins, d.beta, kbt_kcal, smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
        chosen = selected if selected in {"gamd_exponential", "gamd_cumulant2", "gamd_cumulant3"} else "gamd_cumulant2"
    else:
        fes_exp = fes_cum = fes_cum3 = fes_umbrella
        chosen = "umbrella_only"
    chosen_fes = {"umbrella_only": fes_umbrella, "gamd_exponential": fes_exp, "gamd_cumulant2": fes_cum, "gamd_cumulant3": fes_cum3}.get(chosen, fes_umbrella)
    # kcal/mol CSVs (standard format); kJ/mol plot
    write_2d_fes_csv(out / "chignolin_fes_selected.csv", chosen_fes, chosen)
    write_2d_fes_npz(out / "chignolin_fes_selected.npz", chosen_fes, chosen)
    fes_kj = dict(chosen_fes); fes_kj["pmf"] = chosen_fes["pmf"] * KJ_PER_KCAL
    plot_files = _plot_chignolin_fes_kj_multirange(fes_kj, out / "chignolin_fes.png", chosen, warnings)
    write_2d_fes_csv(out / "chignolin_fes_cumulant2.csv", fes_cum, "gamd_cumulant2")
    write_2d_fes_npz(out / "chignolin_fes_cumulant2.npz", fes_cum, "gamd_cumulant2")
    fes_kj_cum2 = dict(fes_cum); fes_kj_cum2["pmf"] = fes_cum["pmf"] * KJ_PER_KCAL
    plot_files_c2 = _plot_chignolin_fes_kj_multirange(fes_kj_cum2, out / "chignolin_fes_cumulant2.png", "gamd_cumulant2", warnings)
    write_2d_fes_csv(out / "chignolin_fes_cumulant3.csv", fes_cum3, "gamd_cumulant3")
    write_2d_fes_npz(out / "chignolin_fes_cumulant3.npz", fes_cum3, "gamd_cumulant3")
    fes_kj_cum3 = dict(fes_cum3); fes_kj_cum3["pmf"] = fes_cum3["pmf"] * KJ_PER_KCAL
    plot_files_c3 = _plot_chignolin_fes_kj_multirange(fes_kj_cum3, out / "chignolin_fes_cumulant3.png", "gamd_cumulant3", warnings)
    finite = np.isfinite(chosen_fes["pmf"])
    if np.any(finite):
        min_idx = np.unravel_index(np.nanargmin(np.where(finite, chosen_fes["pmf"], np.inf)), chosen_fes["pmf"].shape)
        min_x = float(chosen_fes["cv_A"][min_idx[0]])
        min_y = float(chosen_fes["rg_A"][min_idx[1]])
        span_kj = float((np.nanmax(chosen_fes["pmf"][finite]) - np.nanmin(chosen_fes["pmf"][finite])) * KJ_PER_KCAL)
    else:
        min_x = min_y = span_kj = float("nan")
    info = {
        "available": True,
        "selected_unbiased_method": chosen,
        "n_samples": int(np.count_nonzero(mask)),
        "x_label": "dist(Asp3N-Thr8O) (A)",
        "y_label": "dist(Asp3N-Gly7O) (A)",
        "pmf_minimum_dist1_A": min_x,
        "pmf_minimum_dist2_A": min_y,
        "pmf_span_kj_mol": span_kj,
        "plot_ranges_kj_mol": ["0-5", "0-10", "0-20", "0-40", "0-all"],
        "files": {
            "chignolin_fes_csv": str(out / "chignolin_fes_selected.csv"),
            "chignolin_fes_npz": str(out / "chignolin_fes_selected.npz"),
            "chignolin_fes_png": str(out / "chignolin_fes.png"),
            **({f"chignolin_fes_png_{k}": v for k, v in plot_files.items()} if plot_files else {}),
            "chignolin_fes_cumulant2_csv": str(out / "chignolin_fes_cumulant2.csv"),
            "chignolin_fes_cumulant2_npz": str(out / "chignolin_fes_cumulant2.npz"),
            **({f"chignolin_fes_cumulant2_png_{k}": v for k, v in plot_files_c2.items()} if plot_files_c2 else {}),
            "chignolin_fes_cumulant3_csv": str(out / "chignolin_fes_cumulant3.csv"),
            "chignolin_fes_cumulant3_npz": str(out / "chignolin_fes_cumulant3.npz"),
            **({f"chignolin_fes_cumulant3_png_{k}": v for k, v in plot_files_c3.items()} if plot_files_c3 else {}),
        },
    }
    wjson(out / "chignolin_fes_summary.json", info)
    return info


def _load_epoch_dihedral_features(epoch_dir_str) -> tuple:
    """Return (features array N×D, ok). Loads all replica tica_obs/dihedral_obs_*.npz files
    for one adaptive-production epoch. Shared by tICA-epoch and torsion-PCA-scree analyses."""
    edir = Path(epoch_dir_str) if epoch_dir_str else Path('__nonexistent__')
    tica_dir = edir / 'tica_obs'
    if not tica_dir.is_dir():
        return None, False
    npz_files = sorted(tica_dir.glob('dihedral_obs_*.npz'))
    chunks = []
    for p in npz_files:
        try:
            d = np.load(p)
            if 'features' in d.files:
                chunks.append(d['features'].astype(np.float32))
        except Exception:
            pass
    if not chunks:
        return None, False
    return np.concatenate(chunks, axis=0), True


def _analyze_tica_epochs(prod_dir: Path, out: Path, meta: dict, warn: list) -> dict:
    """Analyze per-epoch tICA CVaux updates: eigenvalue progression, MBAR vs uniform reweighting,
    per-window tIC1 center evolution, and dihedral distribution convergence across epochs."""
    ap = None
    for candidate in (prod_dir / 'adaptive_production', prod_dir.parent / 'adaptive_production'):
        if candidate.is_dir():
            ap = candidate; break
    if ap is None:
        return {'available': False, 'reason': 'adaptive_production/ not found'}

    summary_path = ap / 'adaptive_production_driver_summary.json'
    if not summary_path.exists():
        return {'available': False, 'reason': 'adaptive_production_driver_summary.json not found'}

    try:
        with summary_path.open() as fh:
            driver_summary = json.load(fh)
    except Exception as e:
        return {'available': False, 'reason': f'driver summary read error: {e}'}

    epoch_summaries = driver_summary.get('epoch_summaries', [])
    if not epoch_summaries:
        return {'available': False, 'reason': 'no epoch_summaries in driver summary'}

    tica_records = []
    for es in epoch_summaries:
        tu = es.get('tica_update', {})
        if isinstance(tu, dict) and tu.get('status') == 'updated':
            tica_records.append({
                'epoch': int(es.get('epoch', tu.get('epoch', -1))),
                'epoch_dir': str(es.get('epoch_dir', '')),
                'eigenvalue': float(tu.get('eigenvalue', float('nan'))),
                'n_samples': int(tu.get('n_samples', 0)),
                'mbar_reweighted': bool(tu.get('mbar_reweighted', False)),
                'lag_frames': int(tu.get('lag_frames', 0)),
                'n_windows_updated': int(tu.get('registry_states_updated', len(tu.get('per_window_tic1_centers', {})))),
                'per_window_tic1_centers': {int(k): float(v) for k, v in tu.get('per_window_tic1_centers', {}).items()},
            })

    if not tica_records:
        return {'available': False, 'reason': 'no successful tICA updates in epoch summaries'}

    tica_state_path = ap / 'tica_state.json'

    # Δeigenvalue: change from previous update (nan for first)
    delta_eig = [float('nan')]
    for i in range(1, len(tica_records)):
        delta_eig.append(tica_records[i]['eigenvalue'] - tica_records[i-1]['eigenvalue'])
    for i, r in enumerate(tica_records):
        r['delta_eigenvalue'] = delta_eig[i]

    # --- Dihedral distribution convergence via mean JS divergence -----------------
    # Load sin/cos dihedral features from each epoch's tica_obs/*.npz.
    # Compute cumulative histograms and compare against final (all-epochs) distribution.
    _DBINS = 60
    _DRANGE = (-1.01, 1.01)

    def _feature_histograms(X, n_bins=_DBINS, rng=_DRANGE):
        """Per-feature histogram counts. Returns (D, n_bins) int array."""
        D = X.shape[1]
        counts = np.zeros((D, n_bins), dtype=np.int64)
        for d in range(D):
            counts[d], _ = np.histogram(X[:, d], bins=n_bins, range=rng)
        return counts

    def _mean_js(counts_a, counts_b):
        """Mean Jensen-Shannon divergence (nats) across features. counts: (D, n_bins)."""
        a = counts_a.astype(np.float64); b = counts_b.astype(np.float64)
        a_sum = a.sum(axis=1, keepdims=True); b_sum = b.sum(axis=1, keepdims=True)
        a /= np.where(a_sum > 0, a_sum, 1.0)
        b /= np.where(b_sum > 0, b_sum, 1.0)
        m = 0.5 * (a + b)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_m = np.where(m > 0, np.log(m), 0.0)
            kl_am = np.where(a > 0, a * (np.log(np.where(a > 0, a, 1.0)) - log_m), 0.0)
            kl_bm = np.where(b > 0, b * (np.log(np.where(b > 0, b, 1.0)) - log_m), 0.0)
        return float(np.mean(0.5 * (kl_am + kl_bm).sum(axis=1)))

    epoch_features = []  # per-record: (D, _DBINS) count array or None
    n_features = 0
    for r in tica_records:
        X, ok = _load_epoch_dihedral_features(r['epoch_dir'])
        if ok and X.ndim == 2 and X.shape[1] > 0:
            epoch_features.append(_feature_histograms(X))
            n_features = max(n_features, X.shape[1])
        else:
            epoch_features.append(None)

    # Compute cumulative → final JS (convergence to asymptote) and incremental JS
    dihedral_js_vs_final = []
    dihedral_js_incremental = []
    has_dihedral_data = any(e is not None for e in epoch_features)
    if has_dihedral_data:
        valid_counts = [e for e in epoch_features if e is not None]
        final_counts = sum(valid_counts) if len(valid_counts) > 1 else valid_counts[0]
        cumulative = None
        for i, ec in enumerate(epoch_features):
            if ec is None:
                dihedral_js_vs_final.append(float('nan'))
                dihedral_js_incremental.append(float('nan'))
                continue
            prev_cum = cumulative
            cumulative = ec if cumulative is None else (cumulative + ec)
            dihedral_js_vs_final.append(_mean_js(cumulative, final_counts))
            if prev_cum is not None:
                dihedral_js_incremental.append(_mean_js(prev_cum, cumulative))
            else:
                dihedral_js_incremental.append(float('nan'))
        for i, r in enumerate(tica_records):
            r['dihedral_js_vs_final'] = dihedral_js_vs_final[i]
            r['dihedral_js_incremental'] = dihedral_js_incremental[i]

    tica_out = out / 'tica_epochs'; tica_out.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as e:
        warn.append(f'matplotlib unavailable; tICA epoch plots skipped: {e}')
        return {'available': True, 'n_updates': len(tica_records), 'reason': str(e), 'files': {}}

    generated = {}
    epochs_arr = np.array([r['epoch'] for r in tica_records])
    eigenvalues_arr = np.array([r['eigenvalue'] for r in tica_records])
    delta_eig_arr = np.array([r['delta_eigenvalue'] for r in tica_records])
    n_samples_arr = np.array([r['n_samples'] for r in tica_records])
    mbar_flags = [r['mbar_reweighted'] for r in tica_records]
    colors_mbar = ['#2266cc' if m else '#cc4422' for m in mbar_flags]
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2266cc', markersize=8, label='MBAR-reweighted'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#cc4422', markersize=8, label='Unweighted'),
    ]

    # Plot 1: eigenvalue + Δeigenvalue + training-set size (3-panel)
    try:
        fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        ax1, ax2, ax3 = axes
        ax1.scatter(epochs_arr, eigenvalues_arr, c=colors_mbar, s=60, zorder=5)
        ax1.plot(epochs_arr, eigenvalues_arr, 'k-', alpha=0.4, linewidth=1)
        ax1.legend(handles=legend_handles, fontsize=8)
        ax1.set_ylabel('tIC1 eigenvalue')
        ax1.set_title('tICA CVaux: eigenvalue + model stability per update epoch')
        ax1.grid(True, alpha=0.3)
        # Δeigenvalue: filled area around zero, markers for each update
        valid_delta = np.isfinite(delta_eig_arr)
        if valid_delta.any():
            ax2.axhline(0, color='k', linewidth=0.8, alpha=0.5)
            ax2.bar(epochs_arr[valid_delta], delta_eig_arr[valid_delta],
                    color=[('#2266cc' if m else '#cc4422') for m, v in zip(mbar_flags, valid_delta) if v],
                    alpha=0.7)
            ax2.set_ylabel('Δ eigenvalue')
            ax2.set_title('Eigenvalue change per update (→ 0 = converged model)')
            ax2.grid(True, alpha=0.3, axis='y')
        ax3.bar(epochs_arr, n_samples_arr / 1000, color=colors_mbar, alpha=0.7)
        ax3.set_xlabel('Adaptive epoch')
        ax3.set_ylabel('Training samples (×10³)')
        ax3.set_title('tICA training set size')
        ax3.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        p = tica_out / 'tica_eigenvalue_progression.png'; fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
        generated['tica_eigenvalue_progression_png'] = str(p)
    except Exception as e:
        warn.append(f'tica_eigenvalue_progression.png failed: {e}'); plt.close('all')

    # Plot 2: dihedral distribution convergence (mean JS vs epoch)
    if has_dihedral_data:
        try:
            js_final_arr = np.array([r.get('dihedral_js_vs_final', float('nan')) for r in tica_records])
            js_incr_arr = np.array([r.get('dihedral_js_incremental', float('nan')) for r in tica_records])
            fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
            ax1, ax2 = axes
            # Panel 1: convergence to final distribution
            ax1.plot(epochs_arr, js_final_arr, 'o-', color='#2266cc', linewidth=1.5, markersize=6)
            ax1.fill_between(epochs_arr, 0, js_final_arr, alpha=0.15, color='#2266cc')
            ax1.set_ylabel('Mean JS divergence (nats)')
            ax1.set_title(f'Dihedral convergence vs final distribution  ({n_features} features = {n_features // 2} dihedrals × sin/cos)')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(bottom=0)
            # Panel 2: incremental shift per epoch
            valid = np.isfinite(js_incr_arr)
            if valid.any():
                ax2.bar(epochs_arr[valid], js_incr_arr[valid], color='#884499', alpha=0.7)
                ax2.set_ylabel('Incremental JS (nats)')
                ax2.set_title('Per-epoch dihedral distribution shift (→ 0 = diminishing returns)')
                ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_xlabel('Adaptive epoch')
            fig.tight_layout()
            p = tica_out / 'tica_dihedral_convergence.png'; fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
            generated['tica_dihedral_convergence_png'] = str(p)
        except Exception as e:
            warn.append(f'tica_dihedral_convergence.png failed: {e}'); plt.close('all')

    # Plot 3: per-window tIC1 center heatmap (row=epoch, col=window) + Δ from first epoch
    try:
        all_windows = sorted({w for r in tica_records for w in r['per_window_tic1_centers']})
        if all_windows:
            n_ep = len(tica_records); n_w = len(all_windows)
            heatmap = np.full((n_ep, n_w), np.nan)
            for i, r in enumerate(tica_records):
                for j, w in enumerate(all_windows):
                    v = r['per_window_tic1_centers'].get(w)
                    if v is not None:
                        heatmap[i, j] = v
            finite_vals = heatmap[np.isfinite(heatmap)]
            if finite_vals.size > 0:
                # Main heatmap + Δ from epoch-0 side by side
                n_plots = 2 if n_ep > 1 else 1
                fig, axes = plt.subplots(1, n_plots, figsize=(max(6, n_w * 0.25 + 2) * n_plots, max(4, n_ep * 0.4 + 1.5)))
                if n_plots == 1:
                    axes = [axes]
                vabs = max(abs(float(finite_vals.min())), abs(float(finite_vals.max())), 1e-9)
                ylabels = [f'Ep{r["epoch"]} n_win={r["n_windows_updated"]}' for r in tica_records]
                im0 = axes[0].imshow(heatmap, aspect='auto', cmap='RdBu_r', vmin=-vabs, vmax=vabs,
                                     origin='lower', interpolation='nearest')
                axes[0].set_yticks(range(n_ep)); axes[0].set_yticklabels(ylabels, fontsize=7)
                axes[0].set_xlabel('Window index'); axes[0].set_ylabel('tICA update epoch')
                axes[0].set_title('Per-window tIC1 center')
                plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='tIC1 center')
                if n_plots > 1:
                    delta_map = heatmap - heatmap[0:1, :]  # change from first epoch
                    dabs = float(np.nanmax(np.abs(delta_map[1:]))) if n_ep > 1 else 1e-9
                    dabs = max(dabs, 1e-9)
                    im1 = axes[1].imshow(delta_map, aspect='auto', cmap='PiYG', vmin=-dabs, vmax=dabs,
                                         origin='lower', interpolation='nearest')
                    axes[1].set_yticks(range(n_ep)); axes[1].set_yticklabels(ylabels, fontsize=7)
                    axes[1].set_xlabel('Window index')
                    axes[1].set_title('Δ tIC1 center from epoch 0')
                    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Δ tIC1 center')
                fig.tight_layout()
                p = tica_out / 'tica_window_center_heatmap.png'; fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
                generated['tica_window_center_heatmap_png'] = str(p)
    except Exception as e:
        warn.append(f'tica_window_center_heatmap.png failed: {e}'); plt.close('all')

    # CSV summary (enhanced)
    try:
        csv_path = tica_out / 'tica_epoch_summary.csv'
        fields = ['epoch', 'eigenvalue', 'delta_eigenvalue', 'n_samples', 'n_windows_updated',
                  'mbar_reweighted', 'lag_frames', 'dihedral_js_vs_final', 'dihedral_js_incremental']
        with csv_path.open('w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
            wr.writeheader()
            for r in tica_records:
                row = {k: r.get(k, '') for k in fields}
                wr.writerow(row)
        generated['tica_epoch_summary_csv'] = str(csv_path)
    except Exception as e:
        warn.append(f'tica_epoch_summary.csv failed: {e}')

    return {
        'available': True,
        'n_updates': len(tica_records),
        'n_mbar_reweighted': sum(1 for r in tica_records if r['mbar_reweighted']),
        'n_dihedral_features': n_features,
        'latest_eigenvalue': float(tica_records[-1]['eigenvalue']) if tica_records else None,
        'latest_delta_eigenvalue': float(tica_records[-1]['delta_eigenvalue']) if tica_records else None,
        'latest_epoch': int(tica_records[-1]['epoch']) if tica_records else None,
        'dihedral_js_vs_final': [r.get('dihedral_js_vs_final', float('nan')) for r in tica_records],
        'adaptive_dir': str(ap),
        'tica_state_file': str(tica_state_path) if tica_state_path.exists() else None,
        'files': generated,
    }


def _analyze_torsion_pca_scree(prod_dir: Path, out: Path, meta: dict, args, warn: list) -> dict:
    """Full-spectrum scree analysis of the backbone torsion-PCA feature space (sin/cos phi/psi),
    computed from one adaptive-production epoch's real ``tica_obs/dihedral_obs_*.npz`` samples
    (default epoch 0). Complements the single PC1 used as CV2: shows the whole eigenvalue
    spectrum plus, per component, the top loading torsions (which residue's backbone motion
    that component actually encodes) -- both known to matter for judging whether torsion-PCA
    or a switch to tICA is the better CV2 (see docs/... torsion-PCA scree discussion).
    """
    if getattr(args, 'no_torsion_pca_scree', False):
        return {'available': False, 'reason': 'disabled via --no-torsion-pca-scree'}

    ap = None
    for candidate in (prod_dir / 'adaptive_production', prod_dir.parent / 'adaptive_production'):
        if candidate.is_dir():
            ap = candidate; break
    if ap is None:
        return {'available': False, 'reason': 'adaptive_production/ not found'}

    epoch_idx = int(getattr(args, 'torsion_pca_scree_epoch', 0) or 0)
    epoch_dir = ap / f'epoch_{epoch_idx:03d}'
    if not epoch_dir.is_dir():
        return {'available': False, 'reason': f'{epoch_dir} not found'}

    X, ok = _load_epoch_dihedral_features(str(epoch_dir))
    if not ok or X is None or X.shape[0] < 2:
        return {'available': False, 'reason': f'no usable tica_obs dihedral features in {epoch_dir}'}

    X = X.astype(np.float64)
    n_samples, n_features = X.shape
    if n_features < 2 or n_features % 2 != 0:
        return {'available': False, 'reason': f'unexpected feature count ({n_features}); expected sin/cos pairs'}
    n_torsions_total = n_features // 2
    n_phi = n_torsions_total // 2  # gareus backbone torsion CVs always pair n_phi == n_psi

    mean = X.mean(axis=0)
    Xc = X - mean
    try:
        _, singular_values, vt = np.linalg.svd(Xc, full_matrices=False)
    except Exception as e:
        warn.append(f'torsion_pca_scree SVD failed: {e}')
        return {'available': False, 'reason': f'SVD failed: {e}'}
    variances = (singular_values ** 2) / max(1, n_samples - 1)
    total_variance = float(np.sum(variances))
    if total_variance <= 1e-12:
        return {'available': False, 'reason': 'zero variance in dihedral features'}
    evr = variances / total_variance
    cum_evr = np.cumsum(evr)
    n_components = len(evr)

    # Residue-based torsion labels when the sequence length matches n_phi+1 residues;
    # otherwise fall back to generic phi_i/psi_i (still numerically correct, just less readable).
    seq = str(meta.get('sequence', '') or '')
    labels = None
    if len(seq) == n_phi + 1:
        try:
            labels = [f'phi-{seq[i + 1]}{i + 2}' for i in range(n_phi)] + \
                     [f'psi-{seq[i]}{i + 1}' for i in range(n_phi)]
        except IndexError:
            labels = None
    if labels is None:
        labels = [f'phi_{i + 1}' for i in range(n_phi)] + [f'psi_{i + 1}' for i in range(n_phi)]

    def _top_loadings(pc_idx: int, k: int = 3):
        loadings = np.sqrt(vt[pc_idx, 0::2] ** 2 + vt[pc_idx, 1::2] ** 2)  # per-torsion sin/cos pair magnitude
        order = np.argsort(-loadings)[:k]
        return [(labels[i], float(loadings[i])) for i in order]

    scree_out = out / 'torsion_pca_scree'
    scree_out.mkdir(parents=True, exist_ok=True)
    generated = {}

    try:
        np.savez_compressed(
            scree_out / 'torsion_pca_scree_data.npz',
            mean=mean.astype(np.float32), components=vt.astype(np.float32),
            eigenvalues=variances.astype(np.float64), explained_variance_ratio=evr.astype(np.float64),
            cumulative_evr=cum_evr.astype(np.float64), labels=np.asarray(labels), n_samples=n_samples,
            epoch=epoch_idx,
        )
        generated['torsion_pca_scree_data_npz'] = str(scree_out / 'torsion_pca_scree_data.npz')
    except Exception as e:
        warn.append(f'torsion_pca_scree_data.npz failed: {e}')

    try:
        csv_path = scree_out / 'torsion_pca_scree_table.csv'
        fields = ['pc', 'eigenvalue', 'explained_variance_ratio_pct', 'cumulative_evr_pct',
                  'top1_torsion', 'top1_loading', 'top2_torsion', 'top2_loading', 'top3_torsion', 'top3_loading']
        with csv_path.open('w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            wr.writeheader()
            for i in range(n_components):
                top = _top_loadings(i, k=3)
                row = {'pc': i + 1, 'eigenvalue': float(variances[i]),
                       'explained_variance_ratio_pct': float(evr[i] * 100.0),
                       'cumulative_evr_pct': float(cum_evr[i] * 100.0)}
                for j in range(3):
                    tors, load = top[j] if j < len(top) else ('', float('nan'))
                    row[f'top{j + 1}_torsion'] = tors
                    row[f'top{j + 1}_loading'] = load
                wr.writerow(row)
        generated['torsion_pca_scree_table_csv'] = str(csv_path)
    except Exception as e:
        warn.append(f'torsion_pca_scree_table.csv failed: {e}')

    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        idx = np.arange(1, n_components + 1)
        ax1.bar(idx, evr * 100.0, color=['#2a78d6' if i == 0 else '#b9b7ad' for i in range(n_components)])
        ax1.set_ylabel('Explained variance (%)')
        ax1.set_title(f'Torsion-PCA scree — epoch {epoch_idx} ({n_samples} samples, {n_torsions_total} torsions)')
        ax1.grid(True, alpha=0.3, axis='y')
        ax2.plot(idx, cum_evr * 100.0, 'o-', color='#1baf7a', markersize=4, linewidth=1.5)
        for ref in (50, 80):
            ax2.axhline(ref, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        ax2.set_ylabel('Cumulative variance (%)')
        ax2.set_xlabel('Principal component')
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        p = scree_out / 'torsion_pca_scree.png'
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
        generated['torsion_pca_scree_png'] = str(p)
    except Exception as e:
        warn.append(f'torsion_pca_scree.png failed: {e}')

    return {
        'available': True,
        'epoch': epoch_idx,
        'epoch_dir': str(epoch_dir),
        'n_samples': int(n_samples),
        'n_torsions': int(n_torsions_total),
        'n_components': int(n_components),
        'pc1_explained_variance_ratio': float(evr[0]),
        'top10_explained_variance_ratio': [float(x) for x in evr[:10]],
        'top10_cumulative_evr': [float(x) for x in cum_evr[:10]],
        'top10_top_loadings': [_top_loadings(i, k=1)[0][0] for i in range(min(10, n_components))],
        'files': generated,
    }


def _analyze_epoch_cv_exploration(prod_dir: Path, out: Path, meta: dict, warn: list) -> dict:
    for candidate in (prod_dir / 'adaptive_production', prod_dir.parent / 'adaptive_production'):
        if candidate.is_dir():
            ap = candidate; break
    else:
        return {'available': False, 'reason': 'adaptive_production/ not found'}

    def _load_subrun_cv(subdir: Path):
        npz = subdir / 'analysis_arrays.npz'
        if npz.exists():
            f = np.load(npz)
            cv = np.asarray(f['cv_A'], dtype=np.float64)
            sec = np.asarray(f['secondary_cv'], dtype=np.float64) if 'secondary_cv' in f.files else np.full(len(cv), np.nan)
            return cv, sec
        chunk_dir = subdir / 'analysis_chunks'
        if chunk_dir.is_dir():
            chunks = sorted(chunk_dir.glob('chunk_*.npz'))
            if chunks:
                cv_p, sec_p = [], []
                for cp in chunks:
                    try:
                        f = np.load(cp)
                        cv_p.append(np.asarray(f['cv_A'], dtype=np.float64))
                        sec_p.append(np.asarray(f['secondary_cv'], dtype=np.float64) if 'secondary_cv' in f.files else np.full(len(f['cv_A']), np.nan))
                    except Exception: pass
                if cv_p: return np.concatenate(cv_p), np.concatenate(sec_p)
        return None, None

    # Load per-epoch CV samples.
    # Epoch 0: single-level layout (epoch_000/ has NPZ directly).
    # Epoch 1+: sub-run layout — each sub-run dir (baseline, topup_*) holds its own NPZ/chunks.
    epochs = []
    for epoch_dir in sorted(ap.iterdir()):
        if not epoch_dir.is_dir() or not epoch_dir.name.startswith('epoch_'): continue
        try: epoch_idx = int(epoch_dir.name.rsplit('_', 1)[-1])
        except ValueError: continue
        try:
            cv, sec = _load_subrun_cv(epoch_dir)
            if cv is None:
                # sub-run layout: aggregate baseline + topup_* dirs
                cv_parts, sec_parts = [], []
                for sub in sorted(epoch_dir.iterdir()):
                    if not sub.is_dir(): continue
                    a, b = _load_subrun_cv(sub)
                    if a is not None: cv_parts.append(a); sec_parts.append(b)
                if not cv_parts: continue
                cv = np.concatenate(cv_parts); sec = np.concatenate(sec_parts)
            epochs.append({'epoch_idx': epoch_idx, 'cv_A': cv, 'secondary_cv': sec, 'n_samples': len(cv)})
        except Exception as e:
            warn.append(f'Epoch {epoch_idx} load error: {e}'); continue

    if not epochs:
        return {'available': False, 'reason': 'no epoch data found in adaptive_production/'}

    # Also load final-phase samples for the growth plot (may not exist yet during a run).
    final_entry = None
    final_dir = ap / 'final'
    if final_dir.is_dir():
        try:
            cv_f, sec_f = _load_subrun_cv(final_dir)
            if cv_f is None:
                cv_parts, sec_parts = [], []
                for sub in sorted(final_dir.iterdir()):
                    if not sub.is_dir(): continue
                    a, b = _load_subrun_cv(sub)
                    if a is not None: cv_parts.append(a); sec_parts.append(b)
                if cv_parts: cv_f = np.concatenate(cv_parts); sec_f = np.concatenate(sec_parts)
            if cv_f is not None:
                final_entry = {'label': 'Final', 'cv_A': cv_f, 'secondary_cv': sec_f, 'n_samples': len(cv_f)}
        except Exception: pass

    all_sec = np.concatenate([e['secondary_cv'] for e in epochs])
    is_2d = not np.all(np.isnan(all_sec))

    # Window centers from state_registry.csv: single source covering all epochs including epoch 0.
    # registry_by_epoch[epoch_idx] = (primary_centers, secondary_centers) of states BORN in that epoch.
    registry_by_epoch: dict[int, tuple] = {}
    state_reg = ap / 'state_registry.csv'
    if state_reg.exists():
        try:
            with state_reg.open() as fh:
                by_epoch: dict[int, tuple[list, list]] = {}
                for row in csv.DictReader(fh):
                    try:
                        eidx = int(row['created_epoch'])
                        pc = float(row['primary_center']); sc = float(row.get('secondary_center', 'nan'))
                        by_epoch.setdefault(eidx, ([], []))
                        by_epoch[eidx][0].append(pc); by_epoch[eidx][1].append(sc)
                    except (ValueError, KeyError): pass
            registry_by_epoch = {k: (np.array(v[0]), np.array(v[1])) for k, v in by_epoch.items()}
        except Exception as e:
            warn.append(f'state_registry.csv read error: {e}')

    cv1_label = _primary_cv_axis_label(meta)
    _sec_raw = (meta or {}).get('secondary_cv', {})
    cv2_label = (_secondary_cv_label(meta) if isinstance(_sec_raw, dict) else ('Ramachandran CV2' if isinstance(_sec_raw, str) and 'rama' in _sec_raw.lower() else 'Secondary CV')) if is_2d else 'CV2'

    all_cv = np.concatenate([e['cv_A'] for e in epochs])
    valid_cv = all_cv[np.isfinite(all_cv)]
    if valid_cv.size == 0:
        return {'available': False, 'reason': 'all cv_A samples are NaN'}
    cv_min, cv_max = float(valid_cv.min()), float(valid_cv.max())
    if is_2d:
        valid_sec = all_sec[np.isfinite(all_sec)]
        sec_min = float(valid_sec.min()) if valid_sec.size else -1.0
        sec_max = float(valid_sec.max()) if valid_sec.size else 1.0

    ep_out = out / 'epoch_cv_exploration'; ep_out.mkdir(parents=True, exist_ok=True)
    try: import matplotlib.pyplot as plt
    except Exception as e:
        warn.append(f'matplotlib unavailable; epoch plots skipped: {e}')
        return {'available': False, 'reason': str(e)}

    n = len(epochs)
    cmap = plt.colormaps['tab10' if n <= 10 else 'viridis']
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
    generated = {}

    # Plot 1: per-epoch grid — hexbin (2D) or histogram (1D) with window overlays.
    try:
        ncols = min(n, 5); nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows), squeeze=False)
        for pi, epoch in enumerate(epochs):
            r, c = divmod(pi, ncols); ax = axes[r][c]
            cv_A = epoch['cv_A']; sec = epoch['secondary_cv']
            if is_2d:
                mask = np.isfinite(cv_A) & np.isfinite(sec)
                if mask.sum() > 1:
                    ax.hexbin(cv_A[mask], sec[mask], gridsize=25, cmap='hot_r', mincnt=1, extent=[cv_min, cv_max, sec_min, sec_max])
                ew = registry_by_epoch.get(epoch['epoch_idx'])
                if ew is not None:
                    wmask = np.isfinite(ew[0]) & np.isfinite(ew[1])
                    ax.scatter(ew[0][wmask], ew[1][wmask], c='white', edgecolors='black', s=25, zorder=5, linewidths=0.6, marker='D')
                ax.set_xlim(cv_min, cv_max); ax.set_ylim(sec_min, sec_max)
                if r == nrows - 1 or pi >= n - ncols: ax.set_xlabel(cv1_label, fontsize=8)
                if c == 0: ax.set_ylabel(cv2_label, fontsize=8)
            else:
                valid = cv_A[np.isfinite(cv_A)]
                if valid.size: ax.hist(valid, bins=30, range=(cv_min, cv_max), color=colors[pi], alpha=0.7)
                ax.set_xlim(cv_min, cv_max)
                if r == nrows - 1 or pi >= n - ncols: ax.set_xlabel(cv1_label, fontsize=8)
            ax.set_title(f'Epoch {epoch["epoch_idx"]}  ({epoch["n_samples"]:,} samples)', fontsize=8)
            ax.tick_params(labelsize=7)
        for pi in range(n, nrows * ncols):
            r, c = divmod(pi, ncols); axes[r][c].set_visible(False)
        fig.tight_layout()
        p = ep_out / 'epoch_cv_grid.png'; fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
        generated['epoch_cv_grid_png'] = str(p)
    except Exception as e:
        warn.append(f'epoch_cv_grid.png failed: {e}'); plt.close('all')

    # Plot 2: cumulative — all epochs overlaid, colored by epoch.
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, epoch in enumerate(epochs):
            cv_A = epoch['cv_A']; sec = epoch['secondary_cv']; color = colors[i]; lbl = f'Epoch {epoch["epoch_idx"]}'
            if is_2d:
                mask = np.isfinite(cv_A) & np.isfinite(sec); x, y = cv_A[mask], sec[mask]
                if x.size > 5000:
                    rng = np.random.default_rng(42); idx = rng.choice(x.size, 5000, replace=False); x, y = x[idx], y[idx]
                ax.scatter(x, y, c=[color], alpha=0.3, s=1, label=lbl, rasterized=True)
            else:
                valid = cv_A[np.isfinite(cv_A)]
                if valid.size: ax.hist(valid, bins=40, range=(cv_min, cv_max), color=color, alpha=0.5, label=lbl, density=True)
        ax.set_xlabel(cv1_label)
        if is_2d: ax.set_ylabel(cv2_label)
        ax.legend(fontsize=8, markerscale=5 if is_2d else 1)
        ax.set_title('CV exploration — all epochs overlaid')
        fig.tight_layout(); p = ep_out / 'epoch_cv_cumulative.png'; fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
        generated['epoch_cv_cumulative_png'] = str(p)
    except Exception as e:
        warn.append(f'epoch_cv_cumulative.png failed: {e}'); plt.close('all')

    # Plot 3: window center evolution — states colored by birth epoch (from state_registry).
    if registry_by_epoch and is_2d:
        try:
            all_born_epochs = sorted(registry_by_epoch.keys())
            ne = len(all_born_epochs); wcmap = plt.colormaps['tab10' if ne <= 10 else 'viridis']
            wcolors = {e: wcmap(i / max(ne - 1, 1)) for i, e in enumerate(all_born_epochs)}
            fig, ax = plt.subplots(figsize=(6, 5))
            for eidx in all_born_epochs:
                pc, sc = registry_by_epoch[eidx]; wmask = np.isfinite(pc) & np.isfinite(sc)
                n_states = wmask.sum()
                ax.scatter(pc[wmask], sc[wmask], c=[wcolors[eidx]], s=50, label=f'Epoch {eidx} ({n_states} states)', zorder=3, alpha=0.9, edgecolors='black', linewidths=0.5, marker='D')
            ax.set_xlabel(cv1_label); ax.set_ylabel(cv2_label)
            ax.set_title('Window states — colored by birth epoch')
            ax.legend(fontsize=8)
            fig.tight_layout(); p = ep_out / 'epoch_window_evolution.png'; fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
            generated['epoch_window_evolution_png'] = str(p)
        except Exception as e:
            warn.append(f'epoch_window_evolution.png failed: {e}'); plt.close('all')

    # Plot 4: cumulative space growth — panels show CV1×CV2 density after each epoch,
    # plus a final-phase panel.  Fixed color scale across all panels so coverage
    # accumulation is directly comparable.
    if is_2d:
        try:
            GBINS = 40
            panels = []
            cum_cv, cum_sec = np.empty(0), np.empty(0)
            for epoch in epochs:
                mask = np.isfinite(epoch['cv_A']) & np.isfinite(epoch['secondary_cv'])
                cum_cv = np.concatenate([cum_cv, epoch['cv_A'][mask]])
                cum_sec = np.concatenate([cum_sec, epoch['secondary_cv'][mask]])
                h, xe, ye = np.histogram2d(cum_cv, cum_sec, bins=GBINS,
                                           range=[[cv_min, cv_max], [sec_min, sec_max]])
                panels.append({'label': f'Up to epoch {epoch["epoch_idx"]}  ({len(cum_cv):,} samples)', 'h': h, 'xe': xe, 'ye': ye})
            if final_entry is not None:
                fmask = np.isfinite(final_entry['cv_A']) & np.isfinite(final_entry['secondary_cv'])
                if fmask.sum() > 0:
                    hf, xef, yef = np.histogram2d(
                        final_entry['cv_A'][fmask], final_entry['secondary_cv'][fmask],
                        bins=GBINS, range=[[cv_min, cv_max], [sec_min, sec_max]])
                    panels.append({'label': f'Final phase  ({fmask.sum():,} samples)', 'h': hf, 'xe': xef, 'ye': yef, 'is_final': True})

            if panels:
                global_vmax = max(float(p['h'].max()) for p in panels)
                global_vmax = max(global_vmax, 1.0)
                ncols = min(len(panels), 4); nrows = math.ceil(len(panels) / ncols)
                fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.4 * nrows), squeeze=False)
                for pi, panel in enumerate(panels):
                    r, c = divmod(pi, ncols); ax = axes[r][c]
                    h = panel['h']; xe = panel['xe']; ye = panel['ye']
                    # mask empty bins so they show as background, not zero-color
                    hm = np.ma.masked_where(h == 0, h)
                    xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
                    im = ax.pcolormesh(xc, yc, hm.T, cmap='hot_r', vmin=1, vmax=global_vmax, shading='nearest')
                    is_final = panel.get('is_final', False)
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#2266cc' if is_final else 'black')
                        spine.set_linewidth(2.0 if is_final else 0.8)
                    ax.set_title(panel['label'], fontsize=8, color='#2266cc' if is_final else 'black')
                    ax.set_xlabel(cv1_label, fontsize=8)
                    ax.set_ylabel(cv2_label, fontsize=8)
                    ax.tick_params(labelsize=7)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
                for pi in range(len(panels), nrows * ncols):
                    r, c = divmod(pi, ncols); axes[r][c].set_visible(False)
                fig.suptitle('Exploration space growth by epoch (CV1 × CV2 sample density)', fontsize=10, y=1.01)
                fig.tight_layout()
                p = ep_out / 'epoch_cv_space_growth.png'
                fig.savefig(p, dpi=160, bbox_inches='tight'); plt.close(fig)
                generated['epoch_cv_space_growth_png'] = str(p)
        except Exception as e:
            warn.append(f'epoch_cv_space_growth.png failed: {e}'); plt.close('all')

    return {'available': True, 'n_epochs': n, 'is_2d': is_2d, 'adaptive_dir': str(ap), 'files': generated}


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
    K = d.u_nk.shape[1]
    N = len(d.cv)
    if precomputed_base_w is not None:
        base_w = np.array(precomputed_base_w, dtype=np.float64, copy=True)
    else:
        base_w = norm_logw(np.asarray(logw, dtype=np.float64))
    umbrella = pmf_from_weights(d.cv, base_w, bins, kbt_kcal)
    bs = boost_stats(d.boost_kj, d.beta)
    boost_ok = bool(bs.get('available')) and np.nanstd(d.boost_kj) > 1e-12
    if boost_ok:
        exp_w = norm_logw(logw + d.beta * d.boost_kj)
        exp_pmf = pmf_from_weights(d.cv, exp_w, bins, kbt_kcal)
        (cum_pmf, cdiag), (cum3_pmf, cdiag3) = _cumulant_expansion_both(d.cv, base_w, d.boost_kj, bins, d.beta, kbt_kcal, smooth_logfac_sigma=_eff_smooth(args, 'gamd_smooth_sigma'))
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
        e = ess(exp_w)
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
    O = overlap_matrix(d.cv, d.window, bins, K)
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
        block_ids = _sample_block_ids(d)
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
            smooth_logfac_sigma=_eff_smooth(args, 'gamd_smooth_sigma'),
        )
        pmf_uncertainty_std = boot_result['pmf_std']
        _uncertainty_extra = {'pmf_std_kcal_mol': pmf_uncertainty_std}
        if boot_result['low_block_windows']:
            warnings.append(f"{warning_prefix}PMF uncertainty is unreliable near windows {boot_result['low_block_windows']} (fewer than 3 independent trajectory blocks).")
    write_pmf(out / 'pmf_unbiased.csv', sel, selected, {'boost_mean_kj_mol': _sel_diag.get('boost_mean_kj', np.full(args.bins, np.nan)), 'boost_var_kj2_mol2': _sel_diag.get('boost_var_kj2', np.full(args.bins, np.nan)), **_uncertainty_extra})
    write_pmf(out / 'pmf_umbrella_only.csv', umbrella, 'umbrella_only')
    write_pmf(out / 'pmf_gamd_exponential.csv', exp_pmf, 'gamd_exponential')
    write_pmf(out / 'pmf_gamd_cumulant2.csv', cum_pmf, 'gamd_cumulant2', {'boost_mean_kj_mol': cdiag.get('boost_mean_kj', np.full(args.bins, np.nan)), 'boost_var_kj2_mol2': cdiag.get('boost_var_kj2', np.full(args.bins, np.nan))})
    write_pmf(out / 'pmf_gamd_cumulant3.csv', cum3_pmf, 'gamd_cumulant3', {'boost_mean_kj_mol': cdiag3.get('boost_mean_kj', np.full(args.bins, np.nan)), 'boost_var_kj2_mol2': cdiag3.get('boost_var_kj2', np.full(args.bins, np.nan))})
    write_all(out / 'pmf_all_methods.csv', pmfs)
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
    plot_outputs(d, pmfs, selected, O, out, warnings, smooth_sigma=_eff_smooth(args, 'pmf_smooth_sigma'), args=args)
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


def analyze(d,args, progress: Optional[Progress] = None):
    out=d.out_dir; out.mkdir(parents=True,exist_ok=True); N,K=d.u_nk.shape; kbt_kj=1.0/d.beta; kbt_kcal=kbt_kj/KJ_PER_KCAL; warn=list(d.meta.get('load_notes',[]))
    if progress is not None: progress.step('analysis', f'loaded {N} samples across {K} windows from {d.source}')
    m=solve_mbar(d.u_nk,d.window, tol=float(getattr(args,'mbar_tol',1e-10)), progress=progress, backend=getattr(args,'mbar_backend','auto'), threads=getattr(args,'mbar_threads',0)); base_w=norm_logw(m['logw'])
    if progress is not None: progress.bar('analysis stages', 1, 6, 'MBAR solved', force=True)
    if not m['converged']: warn.append(f"MBAR solver did not fully converge: max_delta={m['max_delta']:.3e}")
    zero=np.where(m['n_k']==0)[0].tolist()
    if zero: warn.append(f'Zero production samples for windows: {zero}')
    bins=make_bins(d.cv,args.bins,args.cv_min,args.cv_max)
    if progress is not None: progress.bar('analysis stages', 2, 6, 'building PMFs', force=True)
    logw=np.asarray(m['logw'],dtype=np.float64)

    # Never pool epoch_000 into the main PMF/GaMD-boost report: it runs under
    # a different GaMD envelope (the shared-envelope recalibration fires from
    # epoch 0's own sampling and at most once, gareus/adaptive_production.py's
    # _maybe_recalibrate_gamd_boost) and often a different secondary-CV
    # definition than every later epoch (see run_secondary_cv_analyses above).
    # When a split is available, epoch_000 gets its own report under
    # epoch_000_separate/ and the "main" report -- the one at the normal
    # paths, feeding pmf_summary.json's headline numbers -- covers only
    # epoch_001+ (and final). Both reuse this same global MBAR solve (m/logw)
    # rather than re-solving -- only which samples get binned differs.
    #
    # Each subset's logw is recomputed via _subset_logw_from_global_fk (reuses
    # the global f_k, but the SUBSET's own per-state N_k in the MBAR
    # self-consistency denominator) rather than naively sliced from the global
    # logw and renormalized. The naive slice only corrects for the subset's
    # overall size, not for different states losing different *fractions* of
    # their samples to the split -- that mismatch produces a real,
    # direction-consistent tilt in the resulting PMF (confirmed on real data:
    # ~0.11 kcal/mol shift in pmf_span_kcal_mol on a ~13 kcal/mol span).
    epoch0_split=_epoch_zero_split_masks(d)
    epoch0_report_info=None
    if epoch0_split is not None:
        mask0,mask_rest=epoch0_split
        d_epoch0=_masked_data(d,mask0)
        out_epoch0=out/'epoch_000_separate'; out_epoch0.mkdir(parents=True,exist_ok=True)
        logw_epoch0=_subset_logw_from_global_fk(d_epoch0,m['f_k'])
        epoch0_report_info=run_pmf_and_gamd_boost_report(d_epoch0,args,logw_epoch0,bins,kbt_kcal,out_epoch0,warn,progress,warning_prefix='[epoch_000 report] ')
        d_main=_masked_data(d,mask_rest); logw_main=_subset_logw_from_global_fk(d_main,m['f_k'])
        main_precomputed_base_w=None
    else:
        d_main=d; logw_main=logw
        # No split: logw_main is the exact same full-population array as
        # m['logw'] used to compute base_w above, so norm_logw(logw_main)
        # would just recompute an identical value. Thread it through instead.
        main_precomputed_base_w=base_w

    if progress is not None: progress.bar('analysis stages', 3, 6, 'overlap diagnostics', force=True)
    main_report_info=run_pmf_and_gamd_boost_report(d_main,args,logw_main,bins,kbt_kcal,out,warn,progress,precomputed_base_w=main_precomputed_base_w)
    pmfs=main_report_info['pmfs']; selected=main_report_info['selected']; boost_ok=main_report_info['boost_ok']
    bs=main_report_info['boost']; O=main_report_info['O']; neigh=main_report_info['neighbor_overlap']
    span=main_report_info['pmf_span_kcal_mol']; sel=pmfs[selected]

    if progress is not None: progress.bar('analysis stages', 4, 6, 'writing CSV outputs', force=True)
    rg_info=analyze_rg(d,args,m,base_w,selected,boost_ok,kbt_kcal,out,warn,progress)
    fes2d_info=analyze_distance_rg_2d_fes(d,args,logw,selected,boost_ok,kbt_kcal,out,warn,progress) if isinstance(rg_info,dict) and rg_info.get('available') else {'available':False,'reason':'Rg analysis unavailable'}
    pca2d_info=analyze_pca_2d_fes(d,args,logw,selected,boost_ok,kbt_kcal,out,warn,progress)
    extra_obs_info=analyze_extra_observable_pmfs(d,args,logw,selected,boost_ok,kbt_kcal,out,warn,progress)
    chignolin_fes_info=analyze_chignolin_fes(d,args,logw,selected,boost_ok,kbt_kcal,out,warn,progress)
    secondary_cv_pmf_info,cv1_cv2_fes_info=run_secondary_cv_analyses(d,args,logw,selected,boost_ok,kbt_kcal,out,warn,progress,f_k_global=m['f_k'])
    poincare_info=analyze_poincare_map(d,args,logw,selected,boost_ok,kbt_kcal,out,warn,progress)
    poincare_torsions_info=analyze_poincare_residue_torsions(d,args,out,poincare_info,warn,progress)
    epoch_cv_info=_analyze_epoch_cv_exploration(d.prod_dir,out,d.meta,warn)
    tica_epoch_info=_analyze_tica_epochs(d.prod_dir,out,d.meta,warn)
    torsion_pca_scree_info=_analyze_torsion_pca_scree(d.prod_dir,out,d.meta,args,warn)
    # sel (the reference PMF) was built from d_main -- epoch_000 excluded when
    # a split exists (see above). The convergence checks must be run against
    # that SAME population, not the full d: otherwise even the 100%-of-data
    # checkpoint can never match a reference it structurally can't reach,
    # forcing a spurious converged=False purely from population mismatch
    # (confirmed: this propagated into gareus_report.py's top-level
    # PASS/CAUTION/FAIL verdict for every multi-epoch adaptive-production
    # run). When no split exists, d_main is d itself, so this is unchanged.
    conv_info=run_pmf_convergence(d_main,args,bins,selected,sel,out,progress=progress,f_init_hint=m.get('f_k'))
    epoch_conv_info={}
    if d_main.meta.get('_epoch_source'):
        epoch_conv_info=run_epoch_pmf_convergence(d_main,args,bins,selected,sel,out,progress=progress,f_init_hint=m.get('f_k'))
    s={'production_dir':str(d.prod_dir),'output_dir':str(out),'source':d.source,'n_samples':int(N),'n_windows':int(K),'temperature_K':float(d.temp),'beta_1_over_kj_mol':float(d.beta),'cv_min_A':float(np.nanmin(d.cv)),'cv_max_A':float(np.nanmax(d.cv)),'primary_cv_units':_primary_cv_units(d.meta),'primary_cv_axis_label':_primary_cv_axis_label(d.meta),'selected_unbiased_method':selected,'pmf_span_kcal_mol':span,'pmf_minimum_cv_A':main_report_info['pmf_minimum_cv_A'],'mbar':{'converged':bool(m['converged']),'iterations':int(m['iterations']),'max_delta':float(m['max_delta']),'backend':m.get('backend','unknown'),'threads':m.get('threads',None),'active_states':[int(x) for x in m['active']],'n_k':[int(x) for x in m['n_k']],'base_ess':float(ess(base_w))},'boost':bs,'neighbor_overlap':neigh,'warnings':warn,'files':{**main_report_info['files'],'summary_md':str(out/'pmf_summary.md'),'summary_json':str(out/'pmf_summary.json')}}
    if epoch0_report_info is not None:
        s['epoch_000_report']={'available':True,'reason':'epoch_000 excluded from the main PMF/GaMD-boost report above; this covers epoch_000 only',
                               'n_samples':epoch0_report_info['n_samples'],'selected_unbiased_method':epoch0_report_info['selected'],
                               'pmf_span_kcal_mol':epoch0_report_info['pmf_span_kcal_mol'],'pmf_minimum_cv_A':epoch0_report_info['pmf_minimum_cv_A'],
                               'boost':epoch0_report_info['boost'],'neighbor_overlap':epoch0_report_info['neighbor_overlap'],
                               'files':epoch0_report_info['files']}
        s['files'].update({f'epoch_000_{k}':v for k,v in epoch0_report_info['files'].items()})
    else:
        s['epoch_000_report']={'available':False,'reason':'no epoch_000/rest split available (single-epoch run or non-adaptive-production source)'}
    s['rg']=rg_info
    s['distance_rg_2d_fes']=fes2d_info
    s['pca_2d_fes']=pca2d_info
    s['extra_observable_pmfs']=extra_obs_info
    s['chignolin_fes']=chignolin_fes_info
    s['secondary_cv_pmf']=secondary_cv_pmf_info
    s['poincare_map']=poincare_info
    s['poincare_residue_torsions']=poincare_torsions_info
    s['cv1_cv2_2d_fes']=cv1_cv2_fes_info
    s['epoch_convergence']=epoch_conv_info
    s['epoch_cv_exploration']=epoch_cv_info
    s['tica_epochs']=tica_epoch_info
    s['torsion_pca_scree']=torsion_pca_scree_info
    s['convergence']=conv_info
    for _info in (rg_info,fes2d_info,pca2d_info,extra_obs_info,chignolin_fes_info,
                  poincare_info,poincare_torsions_info,secondary_cv_pmf_info,
                  cv1_cv2_fes_info,epoch_cv_info,tica_epoch_info,torsion_pca_scree_info,conv_info):
        if isinstance(_info,dict) and _info.get('files'):
            s['files'].update(_info['files'])
    # Presentation-only result-health verdict + warning triage (derived from the
    # numbers already in `s`; computes no new physics). Guarded so a verdict
    # edge-case never aborts an otherwise-complete analysis run.
    try:
        from gareus_report import build_health_verdict, classify_warnings
        s['health']=build_health_verdict(s,float(getattr(args,'min_neighbor_overlap',0.30)))
        s['warnings_grouped']=classify_warnings(s.get('warnings',[]),s.get('selected_unbiased_method'))
    except Exception as _hv_exc:
        s['health']={'overall':'UNKNOWN','checks':[],'error':str(_hv_exc)}
        s.setdefault('warnings_grouped',[])
    wjson(out/'pmf_summary.json',s); summary_md(out/'pmf_summary.md',s)
    if progress is not None: progress.bar('analysis stages', 6, 6, 'summary written', force=True)
    return s

def parse_args(argv=None):
    p=argparse.ArgumentParser(description='GaREUS MBAR/PMF analysis. By default, this runs the full analysis suite: main CV PMF, convergence plots, Rg, distance-Rg 2D FES, PCA1-PCA2 FES, phi/psi/Ramachandran, SASA, secondary-structure fractions, and internal-contact PMFs whenever trajectories/topology are available.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('input', help='Run directory or final_production directory')
    p.add_argument('--epoch', dest='epochs', type=int, action='append', default=None,
                   metavar='N', help='Pool only adaptive-production epoch N; repeat to select multiple epochs. '
                   'The final/ phase (baseline + topup_*) is always included regardless of this filter.')
    p.add_argument('--out', default=None, help='PMF analysis output directory; default: <final_production>/pmf_analysis')
    p.add_argument('--analysis-source', choices=['auto','parquet','npz','csv'], default='auto', help='Analysis input source. auto prefers Parquet chunks (new format) then npz then csv; parquet reads segments.json + samples/*.parquet directly (new gareus package format); npz uses analysis_arrays.npz/analysis_chunks; csv uses samples.csv.')
    p.add_argument('--bins', type=int, default=60)
    p.add_argument('--cv-min', type=float, default=None)
    p.add_argument('--cv-max', type=float, default=None)
    p.add_argument('--min-neighbor-overlap', type=float, default=0.30)
    p.add_argument('--pmf-uncertainty', action='store_true', help='Compute per-bin statistical uncertainty (std, kcal/mol) for the selected/headline PMF via a fixed-f_k block bootstrap (blocks = one replica within one epoch/phase). Off by default: adds real compute cost (roughly n_boot resample-and-rebuild passes) with no MBAR re-solve.')
    p.add_argument('--pmf-uncertainty-n-boot', type=int, default=100, help='Number of block-bootstrap replicates for --pmf-uncertainty. Higher is more precise but slower; below ~20 the uncertainty-of-the-uncertainty is itself noisy.')
    p.add_argument('--pmf-uncertainty-seed', type=int, default=0, help='Seed for --pmf-uncertainty block resampling, for reproducible error bars across runs.')
    p.add_argument('--traj-workers', type=int, default=8, help='Number of parallel worker threads for trajectory-derived observables (Rg, chignolin FES). Each thread loads one replica\'s segments concurrently. mdtraj releases the GIL during XTC/DCD reads so true parallelism is achieved. Set to 1 to disable threading.')
    p.add_argument('--duckdb-threads', type=int, default=0, help='Total DuckDB threads distributed across parallel parquet loaders. 0=auto (min(cpu_count, NUMEXPR_MAX_THREADS, 64)). Divide by --load-workers to get per-connection thread count.')
    p.add_argument('--load-workers', type=int, default=8, help='Number of parallel epoch-dir workers for adaptive parquet loading. Each opens its own DuckDB connection with (--duckdb-threads / --load-workers) threads. Set to 1 to disable parallelism.')
    # MBAR solver backend selection.  Choices include auto, explicit deterministic
    # solvers (lbfgs, numpy, anderson), Numba variants (numba, numba-anderson,
    # numba-diis), and sambar which performs a stochastic warm‑start before a
    # deterministic polish.  The default backend is defined by
    # DEFAULT_MBAR_BACKEND.
    p.add_argument('--mbar-backend',
                   choices=['auto', 'numpy', 'numba', 'lbfgs', 'anderson', 'numba-anderson', 'numba-diis', 'sambar'],
                   default=DEFAULT_MBAR_BACKEND,
                   help='MBAR solver backend. "auto" chooses lbfgs for moderate problem sizes when scipy is available, numba for large problem sizes when numba is available, and anderson otherwise. "sambar" performs a stochastic warm‑start followed by a deterministic polish.')
    p.add_argument('--mbar-threads', type=int, default=0, help='Numba MBAR threads; 0 keeps numba default. Ignored by non-Numba backends.')
    # Anderson/DIIS history for numba-anderson/numba-diis backends
    p.add_argument('--mbar-anderson-history', type=int, default=MBAR_ANDERSON_HISTORY, help='History length for Anderson/DIIS mixing used by the numba-anderson backend.')
    # SAMBAR warm‑start parameters
    p.add_argument('--sambar-epochs', type=int, default=SAMBAR_EPOCHS, help='Number of mini‑batch epochs in the SAMBAR warm‑start.')
    p.add_argument('--sambar-initial-batch-size', type=int, default=SAMBAR_INITIAL_BATCH_SIZE, help='Initial mini‑batch size for the SAMBAR warm‑start.')
    p.add_argument('--sambar-batch-patience', type=int, default=SAMBAR_BATCH_PATIENCE, help='Number of epochs at a fixed batch size before doubling it in the SAMBAR warm‑start.')
    p.add_argument('--sambar-seed', type=int, default=SAMBAR_SEED, help='Random seed for SAMBAR warm‑start batching.')
    p.add_argument('--sambar-lr-scale', type=float, default=SAMBAR_LR_SCALE, help='Learning-rate scale factor for the SAMBAR warm‑start.')
    p.add_argument('--sambar-delta-f-max', type=float, default=SAMBAR_DELTA_F_MAX, help='Maximum absolute change in f_k per SAMBAR warm‑start epoch.')
    p.add_argument('--sambar-polish-backend', choices=['auto', 'lbfgs', 'numba', 'numba-anderson', 'numba-diis', 'anderson', 'numpy'], default=SAMBAR_POLISH_BACKEND, help='Backend used to polish the SAMBAR warm‑start. Must not be "sambar".')
    p.add_argument('--mbar-tol', type=float, default=1e-10, help='Final full-data MBAR fixed-point tolerance.')
    p.add_argument('--trajectory-format', choices=['auto','xtc','dcd'], default='auto', help='Trajectory format to read from replica_trajectories for trajectory-derived observables. auto prefers XTC when present and falls back to DCD.')
    p.add_argument('--convergence-timepoints', type=int, default=10, help='Number of prefix timepoints for PMF convergence testing.')
    p.add_argument('--convergence-mbar-tol', type=float, default=1e-6, help='Looser MBAR tolerance used only for prefix convergence rounds.')
    p.add_argument('--convergence-mbar-maxiter', type=int, default=2000, help='Maximum deterministic MBAR polish iterations used only for prefix convergence rounds.')
    p.add_argument('--convergence-mbar-backend', choices=['auto', 'numpy', 'numba', 'lbfgs', 'anderson', 'numba-anderson', 'numba-diis', 'sambar'], default='sambar', help='MBAR backend used only for prefix convergence rounds. Default keeps SAMBAR for convergence even if the full-run backend is changed.')
    p.add_argument('--convergence-sambar-epochs', type=int, default=10, help='SAMBAR mini-batch warm-start epochs used only for prefix convergence rounds. Default 10 is intentionally lower than --sambar-epochs because checkpoints are already warm-started from the previous prefix.')
    p.add_argument('--convergence-sambar-initial-batch-size', type=int, default=SAMBAR_INITIAL_BATCH_SIZE, help='Initial SAMBAR mini-batch size used only for prefix convergence rounds.')
    p.add_argument('--convergence-sambar-batch-patience', type=int, default=2, help='Number of convergence SAMBAR epochs at a fixed batch size before doubling it.')
    p.add_argument('--convergence-sambar-seed', type=int, default=SAMBAR_SEED, help='Random seed for convergence SAMBAR batching.')
    p.add_argument('--convergence-sambar-lr-scale', type=float, default=SAMBAR_LR_SCALE, help='Learning-rate scale for convergence SAMBAR warm-starts.')
    p.add_argument('--convergence-sambar-delta-f-max', type=float, default=SAMBAR_DELTA_F_MAX, help='Maximum absolute f_k change per convergence SAMBAR epoch.')
    p.add_argument('--convergence-sambar-polish-backend', choices=['auto', 'lbfgs', 'numba', 'numba-anderson', 'numba-diis', 'anderson', 'numpy'], default=SAMBAR_POLISH_BACKEND, help='Deterministic polish backend after convergence SAMBAR warm-start. Must not be sambar.')
    p.add_argument('--convergence-js-threshold', type=float, default=0.01, help='JS threshold for convergence summary.')
    p.add_argument('--convergence-rmse-threshold', type=float, default=0.10, help='PMF RMSE threshold in kcal/mol for convergence summary.')
    p.add_argument('--convergence-dir', default='convergence', help='Subdirectory under output dir for convergence tables/plots.')
    p.add_argument('--selected-method', choices=['auto', 'umbrella_only', 'gamd_exponential', 'gamd_cumulant2', 'gamd_cumulant3'], default='auto', help='Force the selected unbiased estimator written to the *_selected/*_unbiased outputs (incl. Ramachandran 2D FES). Default auto = gamd_cumulant2 when a usable boost is present, else umbrella_only. gamd_cumulant3 adds the beta^3/6*kappa3 (third-cumulant) term on top of gamd_cumulant2, correcting for boost-distribution skew (see gamd_boost_anharmonicity_score); still no exponential/direct reweighting. Use to emit each estimator surface separately for cross-estimator comparison.')
    p.add_argument('--plot-gamd-exponential', action='store_true', help='Include gamd_exponential (direct exponential reweighting) in multi-method comparison plots. Off by default: exponential reweighting has near-zero ESS under typical GaMD boost variance and mostly adds clutter/noise to comparison plots. Always shown regardless of this flag when explicitly forced via --selected-method gamd_exponential.')
    p.add_argument('--plot-gamd-cumulant3', action='store_true', help='Include gamd_cumulant3 (third-cumulant/skew-corrected) in multi-method comparison plots and generate its dedicated 2D FES CSV/NPZ/PNG outputs. Off by default: gamd_cumulant2 is the standard default estimator and cumulant3 is a diagnostic extension. Always generated regardless of this flag when explicitly forced via --selected-method gamd_cumulant3.')
    p.add_argument('--rg-bins', type=int, default=None, help='Rg PMF bins. Defaults to --bins.')
    p.add_argument('--rg-min', type=float, default=None, help='Lower Rg bound in Angstrom for Rg PMF.')
    p.add_argument('--rg-max', type=float, default=None, help='Upper Rg bound in Angstrom for Rg PMF.')
    p.add_argument('--rg-from-trajectories', choices=['auto','never','force'], default='force', help='Reconstruct Rg from replica_trajectories/*.xtc or *.dcd when needed. Default force means full analysis is attempted and missing trajectories/topology/dependencies are reported.')
    p.add_argument('--rg-topology', default=None, help='Topology PDB for post-production Rg reconstruction. If omitted, common production/equilibration PDB paths are searched.')
    p.add_argument('--rg-selection', default='protein and element != H', help='MDTraj atom selection for post-production Rg reconstruction.')
    p.add_argument('--rg-allow-truncate', action='store_true', help='If trajectory frame counts differ from sample counts, align trajectory frames to sample rows by production step instead of skipping.')
    p.add_argument('--rg-convergence-dir', default='rg_convergence', help='Subdirectory under output dir for Rg convergence tables/plots.')
    p.add_argument('--fes2d-cv-bins', type=int, default=None, help='2D distance-vs-Rg FES bins along the CV axis. Defaults to --bins.')
    p.add_argument('--fes2d-rg-bins', type=int, default=None, help='2D distance-vs-Rg FES bins along the Rg axis. Defaults to --rg-bins or --bins.')
    p.add_argument('--smooth-sigma', type=float, default=0.0, help='Master smoothing sigma (bins). Sets both GaMD logfac smoothing (before PMF construction) and 1D PMF plot smoothing to the same value. Overrides --gamd-smooth-sigma and --pmf-smooth-sigma when non-zero. Values 1-3 suppress noisy extremes from sparse bins. 0 = disabled.')
    p.add_argument('--gamd-smooth-sigma', type=float, default=0.0, help='Gaussian smoothing sigma applied to GaMD logfac before PMF construction. Overridden by --smooth-sigma when non-zero.')
    p.add_argument('--pmf-smooth-sigma', type=float, default=0.0, help='Gaussian smoothing sigma for 1D PMF plot curves only (CSV unchanged). Overridden by --smooth-sigma when non-zero.')
    p.add_argument('--fes2d-smooth-sigma', type=float, default=0.0, help='Gaussian smoothing sigma (in grid cells) used only for the plotted interpolated 2D FES heatmap.')
    p.add_argument('--pca-fes-from-trajectories', choices=['auto','never','force'], default='force', help='Build a PCA1-vs-PCA2 2D FES from replica_trajectories/*.xtc or *.dcd using MDTraj. Default force means full analysis is attempted and missing trajectories/topology/dependencies are reported.')
    p.add_argument('--pca-topology', default=None, help='Topology PDB for PCA reconstruction. If omitted, --rg-topology and common production/equilibration PDB paths are searched.')
    p.add_argument('--pca-selection', default='protein and name CA', help='MDTraj atom selection used for alignment and PCA coordinates.')
    p.add_argument('--pca-chunk-size', type=int, default=1000, help='Trajectory chunk size for streaming PCA projection.')
    p.add_argument('--pca-max-fit-frames', type=int, default=100000, help='Maximum number of aligned trajectory frames used to fit PCA components.')
    p.add_argument('--pca-fit-stride', type=int, default=0, help='Frame stride for PCA fitting; 0 chooses a stride from --pca-max-fit-frames.')
    p.add_argument('--pca-allow-truncate', action='store_true', help='If trajectory frame counts differ from sample counts, align PCA trajectory frames to sample rows by production step instead of skipping.')
    p.add_argument('--pca-recompute', action='store_true', help='Recompute PCA scores even if pca_scores.npz already exists in the output directory.')
    p.add_argument('--pca-bins', type=int, default=None, help='Number of bins along both PCA axes for the PCA1-vs-PCA2 2D FES. Defaults to --bins.')
    p.add_argument('--pca1-min', type=float, default=None, help='Lower PCA1 bound in Angstrom for PCA 2D FES.')
    p.add_argument('--pca1-max', type=float, default=None, help='Upper PCA1 bound in Angstrom for PCA 2D FES.')
    p.add_argument('--pca2-min', type=float, default=None, help='Lower PCA2 bound in Angstrom for PCA 2D FES.')
    p.add_argument('--pca2-max', type=float, default=None, help='Upper PCA2 bound in Angstrom for PCA 2D FES.')
    p.add_argument('--pca-smooth-sigma', type=float, default=0.0, help='Gaussian smoothing sigma (in grid cells) used only for the plotted PCA 2D FES heatmap.')
    p.add_argument('--extra-pmf-from-trajectories', choices=['auto','never','force'], default='force', help='Build extra trajectory-derived PMFs by default: phi/psi, Ramachandran 2D FES, SASA, secondary-structure fractions, and internal contact counts. Use never only for a lightweight analysis.')
    p.add_argument('--extra-topology', default=None, help='Topology PDB for extra trajectory observable PMFs. If omitted, PCA/Rg topology search paths are used.')
    p.add_argument('--extra-chunk-size', type=int, default=0, help='Trajectory chunk size for extra observable PMFs. 0 reuses --pca-chunk-size.')
    p.add_argument('--extra-allow-truncate', action='store_true', help='If trajectory frame counts differ from sample counts, align extra-observable trajectory frames to sample rows by production step instead of skipping.')
    p.add_argument('--observable-bins', type=int, default=None, help='Default bin count for scalar extra-observable PMFs. Defaults to --bins.')
    p.add_argument('--torsion-bins', type=int, default=72, help='Number of bins for per-residue phi/psi PMFs and Ramachandran FES axes.')
    p.add_argument('--sasa-bins', type=int, default=None, help='Number of bins for total SASA PMF. Defaults to --observable-bins or --bins.')
    p.add_argument('--sasa-min', type=float, default=None, help='Lower total SASA bound in A^2.')
    p.add_argument('--sasa-max', type=float, default=None, help='Upper total SASA bound in A^2.')
    p.add_argument('--sasa-selection', default='protein', help='MDTraj atom selection used for total SASA. Default computes protein SASA.')
    p.add_argument('--sasa-n-sphere-points', type=int, default=240, help='Sphere points for MDTraj Shrake-Rupley SASA; increase for smoother but slower SASA.')
    p.add_argument('--ss-bins', type=int, default=50, help='Bin count for secondary-structure fraction PMFs.')
    p.add_argument('--contact-bins', type=int, default=None, help='Bin count for internal-contact count PMF. Defaults to an automatic count-based value.')
    p.add_argument('--contact-cutoff-nm', type=float, default=0.45, help='Distance cutoff in nm for internal contacts.')
    p.add_argument('--contact-min-sequence-separation', type=int, default=3, help='Exclude residue pairs closer than this sequence separation for internal contacts.')
    p.add_argument('--contact-scheme', default='closest-heavy', choices=['ca','closest','closest-heavy','sidechain','sidechain-heavy'], help='MDTraj contact scheme for internal contact counts.')
    p.add_argument('--chignolin_fes', action='store_true', help='Compute and plot chignolin native-contact 2D FES: X=dist(Asp3N-Thr8O), Y=dist(Asp3N-Gly7O), plotted in kJ/mol with 0-20 kJ/mol colormap range. Requires replica_trajectories/ XTC or DCD files and a topology PDB (--rg-topology or auto-detected).')
    p.add_argument('--cv2-bins', type=int, default=None, help='Bins for 1D secondary CV PMF. Defaults to --bins.')
    p.add_argument('--cv2-min', type=float, default=None, help='Lower bound for secondary CV PMF axis.')
    p.add_argument('--cv2-max', type=float, default=None, help='Upper bound for secondary CV PMF axis.')
    p.add_argument('--fes2d-cv2-bins', type=int, default=None, help='Bins along secondary CV axis for cv1-vs-cv2 2D FES. Defaults to --cv2-bins or --bins.')
    p.add_argument('--fes2d-cv2-smooth-sigma', type=float, default=0.0, help='Gaussian smoothing sigma (grid cells) for cv1-vs-cv2 2D FES heatmap.')
    p.add_argument('--no-adaptive-rounds', action='store_true', help='Disable automatic augmentation with adaptive_feedback_round_*/ pilot data. By default all pilot rounds are combined with final_production using union-window MBAR.')
    p.add_argument('--no-rg', action='store_true', help='Disable trajectory-based Rg reconstruction/PMF. Convenience alias for --rg-from-trajectories never.')
    p.add_argument('--no-pca-fes', action='store_true', help='Disable PCA1-vs-PCA2 2D FES. Convenience alias for --pca-fes-from-trajectories never.')
    p.add_argument('--no-extra-pmfs', action='store_true', help='Disable phi/psi, Ramachandran, SASA, secondary-structure, and internal-contact PMFs. Convenience alias for --extra-pmf-from-trajectories never.')
    p.add_argument('--no-convergence', action='store_true', help='Skip prefix PMF convergence testing for all scalar observables.')
    p.add_argument('--no-convergence-mbar-cache', action='store_true', help='Disable the in-memory cache that reuses identical prefix MBAR solves across scalar convergence analyses.')
    p.add_argument('--basin-min-depth-kcal', type=float, default=0.3, help='Minimum PMF depth (kcal/mol) for a local minimum to count as a distinct basin in basin-population tracking.')
    p.add_argument('--no-basin-tracking', action='store_true', help='Disable basin population tracking during main CV convergence analysis.')
    p.add_argument('--skip-first-n-frames', type=int, default=0, metavar='N', help='Discard the first N samples from each replica (sorted by production step) before analysis. Useful for equilibration burn-in. Default 0 (keep all).')
    p.add_argument('--analysis-stride','--sample-stride','--frame-skip', dest='analysis_stride', type=int, default=1, metavar='N', help='Keep every Nth saved analysis sample per replica after --skip-first-n-frames. Aliases: --sample-stride and --frame-skip. Default 1 keeps all samples.')
    p.add_argument('--analysis-stride-offset', type=int, default=0, metavar='N', help='Offset within each replica before applying --analysis-stride. Default 0.')
    p.add_argument('--no-poincare-map', action='store_true', help='Disable Poincaré return map analysis (poincare_map.png). Enabled by default when secondary CV (cv2) is available.')
    p.add_argument('--poincare-fold-threshold', type=float, default=None, metavar='CV1', help='CV1 upward-crossing threshold for Poincaré Σ_fold section (entering folded/high-contact region). Default: 90th percentile of CV1 distribution.')
    p.add_argument('--poincare-unfold-threshold', type=float, default=None, metavar='CV1', help='CV1 downward-crossing threshold for Poincaré Σ_unfold section (entering unfolded/extended region). Default: 5th percentile of CV1 distribution.')
    p.add_argument('--poincare-min-segment', type=int, default=5, metavar='N', help='Minimum frames between two counted Poincaré crossings (prevents rapid threshold-bounce double-counting). Default 5.')
    p.add_argument('--poincare-min-dwell', type=int, default=50, metavar='N', help='Committor dwell filter: a crossing only counts if CV1 remains on the committed side for at least N consecutive frames afterward. Filters threshold-bounce artifacts; preserves genuine folding/unfolding events. Default 50 frames (=10 ps at 0.2 ps/frame). Set 0 or 1 to disable.')
    p.add_argument('--no-poincare-residue-torsions', action='store_true', help='Disable per-residue torsion analysis at Poincaré crossing frames.')
    p.add_argument('--poincare-route-split', type=float, default=None, metavar='CV2', help='CV2 cutpoint to split Poincaré folding routes A (below) and B (above). Default: auto-midpoint of the two highest fold CV2 peaks.')
    p.add_argument('--no-adaptive-diag', action='store_true', help='Disable adaptive-production diagnostic plots (epoch/topup phase-space coverage, window layout, topup timeline, overlap). Enabled automatically for adaptive_production runs.')
    p.add_argument('--no-torsion-pca-scree', action='store_true', help='Disable the torsion-PCA scree analysis (full eigenvalue spectrum + per-component torsion loadings, computed from one epoch\'s real tica_obs dihedral samples). Enabled by default when adaptive_production/epoch_NNN/tica_obs/ is present.')
    p.add_argument('--torsion-pca-scree-epoch', type=int, default=0, metavar='N', help='Adaptive-production epoch index to source real dihedral samples from for the torsion-PCA scree analysis. Default 0 (bootstrap/first epoch).')
    p.add_argument('--adaptive-diag-stride', type=int, default=3, metavar='N', help='Sub-sample stride for adaptive diagnostic density maps. Higher = faster but coarser. Default 3.')
    p.add_argument('--no-adaptive-diag-coverage', action='store_true', help='Skip the per-phase 2D density map panel (fig1) from adaptive diagnostics — the slowest panel. Other panels still run.')
    args=p.parse_args(argv)
    if args.epochs is not None and any(epoch < 0 for epoch in args.epochs):
        p.error('--epoch must be non-negative')
    if getattr(args,'no_rg',False):
        args.rg_from_trajectories='never'
    if getattr(args,'no_pca_fes',False):
        args.pca_fes_from_trajectories='never'
    if getattr(args,'no_extra_pmfs',False):
        args.extra_pmf_from_trajectories='never'

    # -------------------------------------------------------------------------
    # Apply command‑line overrides to module‑level MBAR configuration variables.
    # These globals influence the default backend for solve_mbar() and the
    # stochastic SAMBAR warm‑start/Anderson parameters.  They must be
    # updated after parsing so that subsequent calls to solve_mbar() and
    # solve_mbar_sambar() use the user‑supplied values when the backend is
    # unspecified (or "auto").
    try:
        _mbar_solvers.DEFAULT_MBAR_BACKEND = str(getattr(args, 'mbar_backend', _mbar_solvers.DEFAULT_MBAR_BACKEND) or _mbar_solvers.DEFAULT_MBAR_BACKEND)
        _mbar_solvers.SAMBAR_EPOCHS = int(getattr(args, 'sambar_epochs', _mbar_solvers.SAMBAR_EPOCHS))
        _mbar_solvers.SAMBAR_INITIAL_BATCH_SIZE = int(getattr(args, 'sambar_initial_batch_size', _mbar_solvers.SAMBAR_INITIAL_BATCH_SIZE))
        _mbar_solvers.SAMBAR_BATCH_PATIENCE = int(getattr(args, 'sambar_batch_patience', _mbar_solvers.SAMBAR_BATCH_PATIENCE))
        _mbar_solvers.SAMBAR_SEED = int(getattr(args, 'sambar_seed', _mbar_solvers.SAMBAR_SEED))
        _mbar_solvers.SAMBAR_LR_SCALE = float(getattr(args, 'sambar_lr_scale', _mbar_solvers.SAMBAR_LR_SCALE))
        _mbar_solvers.SAMBAR_DELTA_F_MAX = float(getattr(args, 'sambar_delta_f_max', _mbar_solvers.SAMBAR_DELTA_F_MAX))
        _mbar_solvers.SAMBAR_POLISH_BACKEND = str(getattr(args, 'sambar_polish_backend', _mbar_solvers.SAMBAR_POLISH_BACKEND) or _mbar_solvers.SAMBAR_POLISH_BACKEND)
        _mbar_solvers.MBAR_ANDERSON_HISTORY = int(getattr(args, 'mbar_anderson_history', _mbar_solvers.MBAR_ANDERSON_HISTORY))
        # Keep this module's own re-exported copies in sync too, in case any
        # call site reads the local name directly rather than calling
        # solve_mbar()/solve_mbar_sambar() afresh (none identified today, but
        # the assignments are cheap and remove the possibility entirely).
        g = globals()
        g['DEFAULT_MBAR_BACKEND'] = _mbar_solvers.DEFAULT_MBAR_BACKEND
        g['SAMBAR_EPOCHS'] = _mbar_solvers.SAMBAR_EPOCHS
        g['SAMBAR_INITIAL_BATCH_SIZE'] = _mbar_solvers.SAMBAR_INITIAL_BATCH_SIZE
        g['SAMBAR_BATCH_PATIENCE'] = _mbar_solvers.SAMBAR_BATCH_PATIENCE
        g['SAMBAR_SEED'] = _mbar_solvers.SAMBAR_SEED
        g['SAMBAR_LR_SCALE'] = _mbar_solvers.SAMBAR_LR_SCALE
        g['SAMBAR_DELTA_F_MAX'] = _mbar_solvers.SAMBAR_DELTA_F_MAX
        g['SAMBAR_POLISH_BACKEND'] = _mbar_solvers.SAMBAR_POLISH_BACKEND
        g['MBAR_ANDERSON_HISTORY'] = _mbar_solvers.MBAR_ANDERSON_HISTORY
    except Exception:
        # Ignore errors during assignment and retain existing defaults
        pass
    return args

def main(argv=None):
    args=parse_args(argv)
    progress=Progress()
    progress.step('load', 'reading current GaREUS outputs')
    _t0=time.time()
    epoch_ids = set(args.epochs) if args.epochs is not None else None
    d=load_data(Path(args.input), Path(args.out) if args.out else None, args.analysis_source, no_augment=getattr(args,'no_adaptive_rounds',False), n_threads=getattr(args,'duckdb_threads',0), n_workers=getattr(args,'load_workers',8), epoch_ids=epoch_ids)
    if getattr(args,'skip_first_n_frames',0)>0:
        n_before=d.cv.size
        d=_skip_first_n_frames(d,args.skip_first_n_frames)
        print(f'  [skip-first-n-frames] dropped {n_before-d.cv.size} samples ({args.skip_first_n_frames} per replica)')
    if getattr(args,'analysis_stride',1)>1 or getattr(args,'analysis_stride_offset',0)>0:
        n_before=d.cv.size
        d=_apply_analysis_stride(d,args.analysis_stride,args.analysis_stride_offset)
        print(f'  [analysis-stride] kept {d.cv.size}/{n_before} samples (stride={args.analysis_stride}, offset={args.analysis_stride_offset})')
    print(f'  [load] {d.cv.size} samples in {time.time()-_t0:.1f}s')
    progress.done('load', f'{d.cv.size} samples')
    _t1=time.time()
    s=analyze(d,args,progress=progress)
    print(f'  [analyze] total {time.time()-_t1:.1f}s')
    # Adaptive-production diagnostic plots (epoch/topup phase-space, window layout, overlap)
    if getattr(d,'prod_dir',None) is not None and Path(d.prod_dir).name=='adaptive_production' and not getattr(args,'no_adaptive_diag',False):
        try:
            import importlib.util as _ilu
            _diag_script=Path(__file__).parent/'plot_adaptive_diagnostics.py'
            if _diag_script.exists():
                _spec=_ilu.spec_from_file_location('plot_adaptive_diagnostics',_diag_script)
                _diag=_ilu.module_from_spec(_spec); _spec.loader.exec_module(_diag)
                _stride=int(getattr(args,'adaptive_diag_stride',3) or 3)
                _skip_cov=bool(getattr(args,'no_adaptive_diag_coverage',False))
                _diag.run_all(Path(d.prod_dir).parent, d.out_dir, stride=_stride, skip_coverage=_skip_cov)
            else:
                print(f'  [adaptive_diag] plot_adaptive_diagnostics.py not found next to analyze_gareus_mbar.py — skipping')
        except Exception as _exc:
            print(f'  [adaptive_diag] skipped: {_exc}')
    progress.done('complete', 'GaREUS PMF analysis complete')
    print('GaREUS PMF analysis complete')
    print(f"  production dir: {s['production_dir']}")
    print(f"  samples/windows: {s['n_samples']} / {s['n_windows']}")
    _pool = _epoch_source_pooling_table(d)
    if _pool:
        print(f"  adaptive sources pooled into PMF: {len(_pool)} (incl. final) — "
              + ", ".join(f"{p['label']}={p['n_samples']}" for p in _pool))
        _skipped = _skipped_empty_epochs(d)
        if _skipped:
            print(f"    empty epochs skipped (no samples): {', '.join(_skipped)}")
    print(f"  selected PMF: {s['selected_unbiased_method']}")
    print(f"  MBAR backend: {s['mbar'].get('backend','unknown')}")
    print(f"  PMF minimum: {s['pmf_minimum_cv_A']} {s.get('primary_cv_units','A')}")
    print(f"  PMF span: {s['pmf_span_kcal_mol']:.3f} kcal/mol")
    print(f"  output dir: {s['output_dir']}")
    try:
        from gareus_report import render_verdict_terminal
        print(render_verdict_terminal(s.get('health'), s.get('warnings_grouped')))
    except Exception:
        if s['warnings']:
            print('  warnings:')
            for w in s['warnings']: print(f'    - {w}')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
