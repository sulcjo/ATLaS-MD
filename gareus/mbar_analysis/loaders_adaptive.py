"""Data-loading domain: adaptive-production directory discovery, vectorized
index remapping, and the legacy/fallback adaptive-production loaders
(epoch-CSV, legacy union-NPZ, multi-round pilot augmentation).

Relocated verbatim from ``analyze_gareus_mbar.py`` (Plan A2). See
docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a2-design.md
for why this is a separate module from ``loaders_union_parquet.py`` (the
primary, non-legacy adaptive-production loader).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from gareus.units import KJ_PER_KCAL, K_B_KJ_PER_MOL_K
from .data import Data, clean, infer_temp_beta, rjson, read_windows, _epoch_dir_index


def _find_adaptive_epoch_dirs(adaptive_dir: Path, epoch_ids: Optional[set[int]] = None) -> list:
    """Return list of (run_dir, window_map_path) for each epoch/final with Parquet samples."""
    def _parquet_subdir(d: Path, parent_wmap: Path) -> tuple | None:
        """Return (d, wmap) if d has Parquet samples, else None."""
        if not ((d/'samples').is_dir() and (d/'segments.json').exists()):
            return None
        wmap = d/'epoch_window_map.csv'
        if not wmap.exists():
            wmap = parent_wmap
        if wmap.exists():
            return (d, wmap)
        return None

    result = []
    for cand in sorted(adaptive_dir.iterdir()):
        if not cand.is_dir():
            continue
        idx = _epoch_dir_index(cand)
        if epoch_ids is not None and idx is not None and idx not in epoch_ids:
            continue
        # epoch_NNN/ directly holds samples/ (first epoch pattern)
        if (cand/'samples').is_dir() and (cand/'segments.json').exists() and (cand/'epoch_window_map.csv').exists():
            result.append((cand, cand/'epoch_window_map.csv'))
            continue
        # epoch_NNN/{baseline,topup_*}/ sub-dirs (double-adaptive / topup pattern)
        epoch_wmap = cand/'epoch_window_map.csv'
        for sub in sorted(cand.iterdir()):
            if not sub.is_dir():
                continue
            if sub.name == 'baseline' or sub.name.startswith('topup_'):
                entry = _parquet_subdir(sub, epoch_wmap)
                if entry is not None:
                    result.append(entry)
    return result


def _find_selfcontained_epoch_dirs(ap: Path) -> list:
    """Epoch dirs under an interrupted adaptive_production/ that are complete
    single-production runs on their own.

    Used as a resolution fallback when the union scheme (registry +
    epoch_window_map.csv) was never written.  Requires the same artifacts
    load_parquet() needs to succeed: samples/ (Parquet), segments.json, and a
    windows/ snapshot.  Returned sorted so the caller can pick the latest.
    """
    if not ap.is_dir():
        return []
    out = []
    for cand in sorted(ap.glob('epoch_[0-9]*')):
        if not cand.is_dir():
            continue
        if ((cand/'samples').is_dir() and (cand/'segments.json').exists()
                and (cand/'windows').is_dir()):
            out.append(cand)
    return out


def _find_adaptive_epoch_csv_sources(ap: Path, epoch_ids: Optional[set[int]] = None) -> list:
    """Return list of Path for each CSV-bearing dir under adaptive_production/epoch_NNN/."""
    sources = []
    for epoch_dir in sorted(ap.glob('epoch_[0-9][0-9][0-9]')):
        if not epoch_dir.is_dir():
            continue
        if epoch_ids is not None and _epoch_dir_index(epoch_dir) not in epoch_ids:
            continue
        if (epoch_dir / 'samples.csv').exists():
            sources.append(epoch_dir)
        for sub in sorted(epoch_dir.iterdir()):
            if not sub.is_dir():
                continue
            if sub.name == 'baseline' or sub.name.startswith('topup_'):
                if (sub / 'samples.csv').exists():
                    sources.append(sub)
    return sources


def _has_epoch_csv_layout(ap: Path, epoch_ids: Optional[set[int]] = None) -> bool:
    return bool(_find_adaptive_epoch_csv_sources(ap, epoch_ids=epoch_ids))


def _find_adaptive_final_run_dirs(ap_dir: Path) -> list:
    """Return list of Path for each CSV-bearing dir under adaptive_production/final/ and final_extension_*/."""
    sources = []
    for top in sorted(ap_dir.iterdir()):
        if not top.is_dir():
            continue
        if top.name != 'final' and not top.name.startswith('final_extension_'):
            continue
        if (top / 'samples.csv').exists():
            sources.append(top)
        for sub in sorted(top.iterdir()):
            if not sub.is_dir():
                continue
            if sub.name == 'baseline' or sub.name.startswith('topup_'):
                if (sub / 'samples.csv').exists():
                    sources.append(sub)
    return sources


def _find_gareus_round_dirs(run_dir: Path) -> list:
    """Return sorted adaptive_feedback_round_*/ dirs that contain analysis_chunks/*.npz."""
    result = []
    for d in sorted(run_dir.glob('adaptive_feedback_round_*')):
        if d.is_dir() and (d / 'umbrella_windows.csv').exists():
            chunks = d / 'analysis_chunks'
            if chunks.is_dir() and any(chunks.glob('chunk_*.npz')):
                result.append(d)
    return result


# A lookup table is only built when the key range actually needed (mapping
# keys unioned with the array's own value range) stays small -- otherwise a
# sparse key space (e.g. one huge outlier ID) would turn a memory-savings
# fix into a memory blowup. Above this, fall back to the original per-element
# Python-level lookup, which stays correct (just not vectorized) regardless
# of key sparsity.
_VECTORIZED_LOOKUP_MAX_TABLE_SIZE = 10_000_000


def _vectorized_map_lookup(arr: np.ndarray, mapping: dict, default: int, dtype=np.int64) -> np.ndarray:
    """Vectorized equivalent of ``np.array([mapping.get(int(x), default) for x in arr], dtype=dtype)``.

    Builds a small dense lookup table spanning the key range actually needed
    (mapping keys union arr's own value range) and does one fancy-index
    instead of a per-element Python-level ``dict.get`` call. Falls back to
    the exact original comprehension whenever a negative key is involved (a
    map key or an array value), the array is empty, or the key range needed
    is too large to be worth a dense table -- so correctness never depends on
    the LUT approach, only performance does.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        return np.array([], dtype=dtype)
    if not mapping:
        return np.full(arr.shape, default, dtype=dtype)

    def _naive():
        return np.array([mapping.get(int(x), default) for x in arr], dtype=dtype)

    try:
        arr_i64 = arr.astype(np.int64, copy=False)
    except (TypeError, ValueError):
        return _naive()
    map_keys = np.fromiter(mapping.keys(), dtype=np.int64, count=len(mapping))
    if arr_i64.min() < 0 or map_keys.min() < 0:
        return _naive()
    hi = max(int(arr_i64.max()), int(map_keys.max()))
    if hi + 1 > _VECTORIZED_LOOKUP_MAX_TABLE_SIZE:
        return _naive()
    lut = np.full(hi + 1, default, dtype=np.int64)
    map_vals = np.fromiter(mapping.values(), dtype=np.int64, count=len(mapping))
    lut[map_keys] = map_vals
    return lut[arr_i64].astype(dtype, copy=False)


def _vectorized_map_lookup_or_self(arr: np.ndarray, mapping: dict, dtype=np.int64) -> np.ndarray:
    """Vectorized equivalent of ``np.array([mapping.get(int(x), int(x)) for x in arr], dtype=dtype)``.

    Same LUT strategy as ``_vectorized_map_lookup``, but the default for a
    key absent from ``mapping`` is the key itself (an identity fallback)
    rather than a fixed sentinel.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        return np.array([], dtype=dtype)
    if not mapping:
        return arr.astype(dtype, copy=True)

    def _naive():
        return np.array([mapping.get(int(x), int(x)) for x in arr], dtype=dtype)

    try:
        arr_i64 = arr.astype(np.int64, copy=False)
    except (TypeError, ValueError):
        return _naive()
    map_keys = np.fromiter(mapping.keys(), dtype=np.int64, count=len(mapping))
    if arr_i64.min() < 0 or map_keys.min() < 0:
        return _naive()
    hi = max(int(arr_i64.max()), int(map_keys.max()))
    if hi + 1 > _VECTORIZED_LOOKUP_MAX_TABLE_SIZE:
        return _naive()
    lut = np.arange(hi + 1, dtype=np.int64)
    map_vals = np.fromiter(mapping.values(), dtype=np.int64, count=len(mapping))
    lut[map_keys] = map_vals
    return lut[arr_i64].astype(dtype, copy=False)


def _vectorized_map_index(arr: np.ndarray, mapping: dict, dtype=np.int64) -> np.ndarray:
    """Vectorized equivalent of ``np.array([mapping[int(x)] for x in arr], dtype=dtype)``.

    Unlike ``_vectorized_map_lookup`` there is no default: a value in ``arr``
    absent from ``mapping`` raises ``KeyError``, matching plain ``dict[key]``
    subscripting semantics exactly (including on an empty mapping).
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        return np.array([], dtype=dtype)

    def _naive():
        return np.array([mapping[int(x)] for x in arr], dtype=dtype)

    if not mapping:
        return _naive()  # raises KeyError on the first element, same as dict[key]
    try:
        arr_i64 = arr.astype(np.int64, copy=False)
    except (TypeError, ValueError):
        return _naive()
    map_keys = np.fromiter(mapping.keys(), dtype=np.int64, count=len(mapping))
    if arr_i64.min() < 0 or map_keys.min() < 0:
        return _naive()
    hi = max(int(arr_i64.max()), int(map_keys.max()))
    if hi + 1 > _VECTORIZED_LOOKUP_MAX_TABLE_SIZE:
        return _naive()
    present = np.zeros(hi + 1, dtype=bool)
    present[map_keys] = True
    missing = ~present[arr_i64]
    if np.any(missing):
        raise KeyError(int(arr_i64[missing][0]))
    lut = np.zeros(hi + 1, dtype=np.int64)
    map_vals = np.fromiter(mapping.values(), dtype=np.int64, count=len(mapping))
    lut[map_keys] = map_vals
    return lut[arr_i64].astype(dtype, copy=False)


def load_epoch_csv_adaptive(ap: Path, epoch_ids: Optional[set[int]] = None) -> Data:
    """Load all epoch/baseline/topup samples.csv files, union windows, rebuild N×K bias matrix."""
    sources = _find_adaptive_epoch_csv_sources(ap, epoch_ids=epoch_ids)
    if not sources:
        selected = f' for requested epoch(s) {sorted(epoch_ids)}' if epoch_ids is not None else ''
        raise FileNotFoundError(f'No epoch CSV sources found under {ap}{selected}')
    root = ap.parent
    meta: dict = {}
    meta.update(rjson(root / 'run_args.json', {}))
    meta.update(rjson(root / 'umbrella_pymbar_metadata.json', {}))
    meta['source'] = 'epoch_csv_adaptive'

    all_rows: list = []
    for src_idx, src in enumerate(sources):
        with (src / 'samples.csv').open(newline='') as fh:
            for row in csv.DictReader(fh):
                row['_src_idx'] = src_idx
                all_rows.append(row)

    if not all_rows:
        raise ValueError(f'No sample rows found across {len(sources)} epoch CSV sources')

    # Collect union window states keyed by (c_prim, k_prim, c_sec, k_sec)
    # Use 8-sig-fig rounding to handle float noise
    def _fkey(v, default=0.0):
        try:
            x = float(v)
            return round(x, 8) if math.isfinite(x) else default
        except Exception:
            return default

    state_key_to_idx: dict = {}
    state_list: list = []  # list of (c_prim, k_prim, c_sec, k_sec)

    def _row_state_key(row):
        cp = _fkey(row.get('primary_cv_center', row.get('center_A', row.get('center', ''))))
        kp = _fkey(row.get('primary_cv_k', row.get('k_kcal_mol_A2', row.get('k', ''))), 0.0)
        cs = _fkey(row.get('secondary_cv_center', ''), float('nan'))
        ks_ = _fkey(row.get('secondary_cv_k_kcal_mol', row.get('secondary_cv_k_kcal', '')), 0.0)
        return (cp, kp, cs, ks_)

    for row in all_rows:
        key = _row_state_key(row)
        if key not in state_key_to_idx:
            state_key_to_idx[key] = len(state_list)
            state_list.append(key)

    K = len(state_list)
    centers = np.asarray([s[0] for s in state_list], dtype=np.float64)
    k_kcal = np.asarray([s[1] for s in state_list], dtype=np.float64)
    sec_centers = np.asarray([s[2] for s in state_list], dtype=np.float64)
    sec_ks = np.asarray([s[3] for s in state_list], dtype=np.float64)
    has_secondary = np.any(np.isfinite(sec_centers) & (sec_ks != 0.0))

    # Extract sample arrays
    cv_list, cv2_list, rg_list, win_list, rep_list, step_list, boost_list, pot_list = [], [], [], [], [], [], [], []
    beta_sample = []
    src_idx_list = []
    for row in all_rows:
        cv_val = row.get('cv_A', row.get('primary_cv_value', ''))
        try:
            cv_list.append(float(cv_val))
        except Exception:
            continue
        try: cv2_list.append(float(row.get('secondary_cv', '') or 'nan'))
        except Exception: cv2_list.append(float('nan'))
        try: rg_list.append(float(row.get('rg_A', '') or row.get('radius_gyration_A', '') or 'nan'))
        except Exception: rg_list.append(float('nan'))
        key = _row_state_key(row)
        win_list.append(state_key_to_idx[key])
        try: rep_list.append(int(float(row.get('replica', 0) or 0)))
        except Exception: rep_list.append(0)
        try: step_list.append(int(float(row.get('step', len(step_list)) or len(step_list))))
        except Exception: step_list.append(len(step_list))
        try: boost_list.append(float(row.get('gamd_boost_total_kj_mol', '') or 'nan'))
        except Exception: boost_list.append(float('nan'))
        try: pot_list.append(float(row.get('potential_kj_mol', '') or 'nan'))
        except Exception: pot_list.append(float('nan'))
        try: beta_sample.append(float(row.get('beta_1_over_kJ_mol', '') or 'nan'))
        except Exception: beta_sample.append(float('nan'))
        src_idx_list.append(int(row.get('_src_idx', 0)))

    cv = np.asarray(cv_list, dtype=np.float64)
    cv2 = np.asarray(cv2_list, dtype=np.float64)
    rg = np.asarray(rg_list, dtype=np.float64)
    window = np.asarray(win_list, dtype=int)
    replica = np.asarray(rep_list, dtype=int)
    step = np.asarray(step_list, dtype=int)
    boost = np.asarray(boost_list, dtype=np.float64)
    pot = np.asarray(pot_list, dtype=np.float64)
    beta_arr = np.asarray(beta_sample, dtype=np.float64)
    finite_betas = beta_arr[np.isfinite(beta_arr)]
    if finite_betas.size:
        beta = float(finite_betas[0])
        temp = 1.0 / (K_B_KJ_PER_MOL_K * beta)
    else:
        temp, beta = infer_temp_beta(root, meta, None)

    # Reconstruct full N×K bias matrix analytically
    scale = beta * KJ_PER_KCAL
    u_nk = np.empty((cv.size, K), dtype=np.float64)
    for k in range(K):
        total = 0.5 * float(k_kcal[k]) * (cv - float(centers[k])) ** 2
        if has_secondary and np.isfinite(sec_centers[k]) and float(sec_ks[k]) != 0.0:
            total = total + 0.5 * float(sec_ks[k]) * (cv2 - float(sec_centers[k])) ** 2
        u_nk[:, k] = scale * total

    meta['load_notes'] = [f'Loaded {cv.size} samples from {len(sources)} epoch CSV sources; union {K} windows.']
    meta['umbrella_window_rows'] = [
        {'center_A': str(centers[i]), 'k_kcal_mol_A2': str(k_kcal[i])} for i in range(K)
    ]
    meta['adaptive_epoch_run_dirs'] = [str(s) for s in sources]
    meta['_epoch_source'] = src_idx_list
    src_str = f'{sources[0]}/samples.csv ... {sources[-1]}/samples.csv'
    return clean(Data(root, root / 'pmf_analysis', cv, cv2, rg, window, replica, step, u_nk, centers, k_kcal, beta, temp, boost, pot, src_str, meta))


def load_union_npz(ap_dir: Path) -> Data:
    """Load MBAR inputs from adaptive_union_mbar.npz (adaptive-production runs where final_production/ not yet complete)."""
    npz_path = ap_dir / 'adaptive_union_mbar.npz'
    jmeta = rjson(ap_dir / 'adaptive_union_mbar.json', {})
    root = ap_dir.parent
    with np.load(npz_path, allow_pickle=False) as f:
        cv = np.asarray(f['cv_A'], dtype=np.float64)
        cv2 = np.asarray(f['secondary_cv'], dtype=np.float64)
        window = np.asarray(f['sampled_state_ids'], dtype=int)
        u_nk = np.asarray(f['umbrella_reduced_bias_nk'], dtype=np.float64)
        centers = np.asarray(f['primary_centers'], dtype=np.float64)
        k_kcal = np.asarray(f['primary_k'], dtype=np.float64)
    rg = np.full(cv.shape, np.nan, dtype=np.float64)
    replica = window.copy()
    step = np.arange(cv.size, dtype=int)
    boost_kj = np.zeros(cv.size, dtype=np.float64)
    meta: dict = {}
    meta.update(rjson(root / 'run_args.json', {}))
    meta.update(rjson(root / 'umbrella_pymbar_metadata.json', {}))
    meta.update(jmeta)
    meta['source'] = 'adaptive_union_mbar'
    beta_kj = float(jmeta.get('beta_1_over_kJ_mol', 0.0))
    if beta_kj > 0:
        temp = 1.0 / (K_B_KJ_PER_MOL_K * beta_kj)
        beta = beta_kj
    else:
        temp, beta = infer_temp_beta(root, meta, None)
    state_reg = ap_dir / 'state_registry.csv'
    if state_reg.exists():
        try:
            with state_reg.open() as fh:
                meta['umbrella_window_rows'] = list(csv.DictReader(fh))
        except Exception:
            pass
    # Read per-sample step, boost, replica, and source info from companion CSV when present.
    samples_csv = ap_dir / 'adaptive_union_mbar.samples.csv'
    run_dirs = _find_adaptive_final_run_dirs(ap_dir)
    if samples_csv.exists() and samples_csv.stat().st_size > 0:
        try:
            steps_list, boost_list, src_list = [], [], []
            src_label_list: list = []
            src_dir_to_idx: dict = {}
            unique_src_labels: list = []
            with samples_csv.open(newline='') as fh:
                for row in csv.DictReader(fh):
                    try: steps_list.append(int(float(row.get('step', 0) or 0)))
                    except Exception: steps_list.append(len(steps_list))
                    try: boost_list.append(float(row.get('gamd_boost_total_kj_mol', '') or 'nan'))
                    except Exception: boost_list.append(float('nan'))
                    src_label = row.get('source', '')
                    if src_label not in src_dir_to_idx:
                        src_dir_to_idx[src_label] = len(unique_src_labels)
                        unique_src_labels.append(src_label)
                    src_list.append(src_dir_to_idx[src_label])
                    src_label_list.append(src_label)
            if len(steps_list) == cv.size:
                step = np.asarray(steps_list, dtype=int)
                boost_kj = np.asarray(boost_list, dtype=np.float64)
                meta['_epoch_source'] = src_list
                # Build (source_label, step, epoch_window) → hardware replica lookup
                # from per-source samples.csv files so trajectory frame matching works.
                label_to_path: dict = {}
                for rd in run_dirs:
                    # Derive source label: relative path from adaptive_production root (e.g. "final/baseline")
                    try:
                        rel = Path(rd).relative_to(ap_dir)
                        label_to_path[str(rel)] = Path(rd)
                    except Exception:
                        pass
                rep_lookup: dict = {}  # (src_label, step, epoch_window) → replica
                src_win_list_for_lookup = []
                for i, row in enumerate([]):
                    pass  # placeholder; we re-read the union CSV below for window column
                # Re-read union CSV for sampled_epoch_window; annotate with replica from source CSVs
                epoch_win_list = []
                try:
                    with samples_csv.open(newline='') as fh:
                        for row in csv.DictReader(fh):
                            try: epoch_win_list.append(int(float(row.get('sampled_epoch_window', 0) or 0)))
                            except Exception: epoch_win_list.append(0)
                except Exception:
                    epoch_win_list = [0] * len(steps_list)
                for src_label, src_path in label_to_path.items():
                    src_csv = src_path / 'samples.csv'
                    if not src_csv.exists():
                        continue
                    try:
                        with src_csv.open(newline='') as fh:
                            for row in csv.DictReader(fh):
                                try:
                                    s = int(float(row.get('step', 0) or 0))
                                    w = int(float(row.get('window', 0) or 0))
                                    r = int(float(row.get('replica', 0) or 0))
                                    rep_lookup[(src_label, s, w)] = r
                                except Exception:
                                    pass
                    except Exception:
                        pass
                rep_list = []
                for i, (lbl, s, w) in enumerate(zip(src_label_list, steps_list, epoch_win_list)):
                    rep_list.append(rep_lookup.get((lbl, s, w), w))
                replica = np.asarray(rep_list, dtype=int)
        except Exception:
            pass
    # Populate adaptive_epoch_run_dirs so _prepare_adaptive_merged_traj_dir can find trajectories.
    if run_dirs:
        meta['adaptive_epoch_run_dirs'] = [str(d) for d in run_dirs]
    return clean(Data(root, root / 'pmf_analysis', cv, cv2, rg, window, replica, step, u_nk, centers, k_kcal, beta, temp, boost_kj, None, str(npz_path), meta))


def _load_round_raw(round_dir: Path) -> Optional[dict]:
    """Load per-sample arrays from all NPZ chunks in a round dir."""
    chunks_dir = round_dir / 'analysis_chunks'
    bufs: dict = {k: [] for k in ('cv_A', 'secondary_cv', 'step', 'replica', 'window',
                                   'gamd_boost_total_kj_mol', 'potential_kj_mol')}
    for chunk_path in sorted(chunks_dir.glob('chunk_*.npz')):
        try:
            with np.load(chunk_path, allow_pickle=False) as f:
                n = len(f['cv_A'])
                if n == 0:
                    continue
                bufs['cv_A'].append(np.asarray(f['cv_A'], float))
                for key in ('secondary_cv', 'cv2_A'):
                    if key in f.files:
                        bufs['secondary_cv'].append(np.asarray(f[key], float))
                        break
                else:
                    bufs['secondary_cv'].append(np.full(n, np.nan))
                bufs['step'].append(np.asarray(f['step'], int) if 'step' in f.files else np.arange(n, dtype=int))
                bufs['replica'].append(np.asarray(f['replica'], int) if 'replica' in f.files else np.zeros(n, int))
                bufs['window'].append(np.asarray(f['window'], int) if 'window' in f.files else np.zeros(n, int))
                for key in ('gamd_boost_total_kj_mol', 'gamd_boost_kj_mol'):
                    if key in f.files:
                        bufs['gamd_boost_total_kj_mol'].append(np.asarray(f[key], float))
                        break
                else:
                    bufs['gamd_boost_total_kj_mol'].append(np.full(n, np.nan))
                bufs['potential_kj_mol'].append(np.asarray(f['potential_kj_mol'], float) if 'potential_kj_mol' in f.files else np.full(n, np.nan))
        except Exception:
            pass
    if not bufs['cv_A']:
        return None
    return {k: np.concatenate(v) for k, v in bufs.items()}


def _build_union_window_table(all_dirs: list, tol: float = 5e-4) -> list:
    """Deduplicate windows across all round dirs + final_production.

    Returns list of dicts sorted by (primary_center, secondary_cv_center).
    Tolerance tol is used to merge numerically identical centers.
    """
    seen: dict = {}
    for rdir in all_dirs:
        wcsv = rdir / 'umbrella_windows.csv'
        if not wcsv.exists():
            continue
        _, _, rows = read_windows(wcsv)
        for row in rows:
            pc = float(row.get('primary_center', row.get('center_A', 0)))
            sc = float(row.get('secondary_cv_center', 0))
            pk = float(row.get('primary_k', row.get('k_kcal_mol_A2', 0)))
            sk = float(row.get('secondary_cv_k_kcal_mol', 0))
            key = (round(pc / tol), round(sc / tol))
            if key not in seen:
                seen[key] = {'primary_center': pc, 'secondary_cv_center': sc,
                             'primary_k_kcal': pk, 'secondary_k_kcal': sk}
    return sorted(seen.values(), key=lambda w: (w['primary_center'], w['secondary_cv_center']))


def _round_window_to_union_map(round_dir: Path, union_windows: list, tol: float = 5e-4) -> dict:
    """Map per-round local window index → union window index."""
    _, _, rows = read_windows(round_dir / 'umbrella_windows.csv')
    mapping: dict = {}
    for row in rows:
        w_local = int(float(row.get('window', 0)))
        pc = float(row.get('primary_center', row.get('center_A', 0)))
        sc = float(row.get('secondary_cv_center', 0))
        for k_union, uw in enumerate(union_windows):
            if abs(uw['primary_center'] - pc) < tol and abs(uw['secondary_cv_center'] - sc) < tol:
                mapping[w_local] = k_union
                break
    return mapping


def _augment_with_adaptive_rounds(d: Data, run_dir: Path) -> Data:
    """Combine final_production Data with samples from adaptive_feedback_round_* dirs.

    Builds the union window set across all rounds, recomputes u_nk analytically
    for every sample, and returns a new Data with all samples concatenated.
    """
    round_dirs = _find_gareus_round_dirs(run_dir)
    if not round_dirs:
        return d

    # NOTE (Plan A2 -> Plan A3 handoff): _compute_u_nk_analytical is one of
    # four bias-reconstruction functions deliberately left behind in
    # analyze_gareus_mbar.py for Plan A3's audited reconciliation with
    # gareus.query.reconstruct_bias_matrix (see this plan's design spec).
    # This import MUST stay function-body-local (lazy): a module-top import
    # here would create a real circular ImportError, since
    # analyze_gareus_mbar.py itself imports Data/load_data/etc. from this
    # subpackage near its own top, before _compute_u_nk_analytical is
    # defined further down in that file. When Plan A3 relocates that
    # function, update the module path in this one line.
    from analyze_gareus_mbar import _compute_u_nk_analytical

    all_dirs = round_dirs + [d.prod_dir]
    union_windows = _build_union_window_table(all_dirs)
    K_union = len(union_windows)

    round_data = []
    for rdir in round_dirs:
        raw = _load_round_raw(rdir)
        if raw is None or raw['cv_A'].size == 0:
            continue
        w2u = _round_window_to_union_map(rdir, union_windows)
        raw['window_union'] = _vectorized_map_lookup(raw['window'], w2u, default=0, dtype=int)
        round_data.append(raw)

    if not round_data:
        return d

    final_w2u = _round_window_to_union_map(d.prod_dir, union_windows)
    final_window_union = _vectorized_map_lookup_or_self(d.window, final_w2u, dtype=int)

    cv1_all = np.concatenate([d.cv] + [r['cv_A'] for r in round_data])
    cv2_all = np.concatenate([d.cv2] + [r['secondary_cv'] for r in round_data])
    rg_all = np.concatenate([d.rg_A] + [np.full(r['cv_A'].size, np.nan) for r in round_data])
    win_all = np.concatenate([final_window_union] + [r['window_union'] for r in round_data])
    rep_all = np.concatenate([d.replica] + [r['replica'] for r in round_data])
    step_all = np.concatenate([d.step] + [r['step'] for r in round_data])
    boost_all = np.concatenate([d.boost_kj] + [r['gamd_boost_total_kj_mol'] for r in round_data])
    pot_parts = [d.potential_kj if d.potential_kj is not None else np.full(d.cv.size, np.nan)]
    pot_parts += [r['potential_kj_mol'] for r in round_data]
    pot_all = np.concatenate(pot_parts)

    u_all = _compute_u_nk_analytical(cv1_all, cv2_all, union_windows, d.beta)

    centers_union = np.array([w['primary_center'] for w in union_windows], float)
    ks_union = np.array([w['primary_k_kcal'] for w in union_windows], float)

    n_round_samples = sum(r['cv_A'].size for r in round_data)
    meta = dict(d.meta)
    meta['load_notes'] = list(meta.get('load_notes') or []) + [
        f'Multi-round augmentation: {len(round_dirs)} adaptive_feedback_round_* dirs, '
        f'+{n_round_samples} pilot samples ({cv1_all.size} total). '
        f'Union {K_union} windows (final_production had {d.u_nk.shape[1]}).'
    ]
    meta['umbrella_window_rows'] = [
        {'center_A': str(w['primary_center']), 'k_kcal_mol_A2': str(w['primary_k_kcal']),
         'primary_center': str(w['primary_center']),
         'secondary_cv_center': str(w['secondary_cv_center']),
         'secondary_cv_k_kcal_mol': str(w['secondary_k_kcal'])}
        for w in union_windows
    ]
    meta['adaptive_round_dirs'] = [str(r) for r in round_dirs]

    return clean(Data(
        d.prod_dir, d.out_dir,
        cv1_all, cv2_all, rg_all, win_all, rep_all, step_all,
        u_all, centers_union, ks_union,
        d.beta, d.temp, boost_all, pot_all,
        d.source + f'+{len(round_dirs)}rounds',
        meta,
    ))
