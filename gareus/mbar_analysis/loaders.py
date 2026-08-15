"""Data-loading domain: single-run NPZ/CSV/Parquet loaders, run-directory
resolution (``prod_dir_of``), and the top-level format-auto-detection
dispatcher (``load_data``).

Relocated verbatim from ``analyze_gareus_mbar.py`` (Plan A2). This is the top
of the ``gareus.mbar_analysis`` data-loading dependency graph -- the only
module here that needs to know about every other loader module, since
``load_data``'s job is inherently to compare across all of them. See
docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a2-design.md.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

from gareus.units import KJ_PER_KCAL
from .data import Data, clean, infer_temp_beta, rjson, read_windows, jvec
from .loaders_adaptive import (
    _find_adaptive_epoch_dirs, _find_selfcontained_epoch_dirs,
    _find_adaptive_epoch_csv_sources, _has_epoch_csv_layout,
    load_epoch_csv_adaptive, load_union_npz,
    _find_gareus_round_dirs, _augment_with_adaptive_rounds,
)
from .loaders_union_parquet import load_parquet_adaptive_union


class _Arrays(dict):
    """Dict wrapper that exposes a .files attribute for drop-in NpzFile compatibility."""
    @property
    def files(self) -> list[str]:
        return list(self.keys())


_ANALYSIS_VECTOR_KEYS = {
    'step', 'replica', 'window', 'cv_A', 'cv2_A', 'secondary_cv',
    'rg_A', 'radius_gyration_A', 'radius_of_gyration_A',
    'secondary_cv_center', 'secondary_cv_k_kcal_mol',
    'center_A', 'k_kcal_mol_A2',
    'potential_kj_mol', 'potential_energy_kj_mol',
    'gamd_boost_total_kj_mol', 'gamd_boost_kj_mol', 'boost_kj_mol',
    'temperature_K', 'temperature_k', 'beta_1_over_kJ_mol',
    'beta_1_over_kj_mol', 'beta',
    'umbrella_reduced_bias_nk', 'umbrella_reduced_bias_kn',
}


def _npz_sample_count_open(f) -> Optional[int]:
    for key in ('step', 'cv_A', 'window', 'replica'):
        if key in f.files:
            arr = f[key]
            if getattr(arr, 'ndim', 0) >= 1:
                return int(arr.shape[0])
    return None


def _npz_window_count_open(f, n_samples: Optional[int]) -> Optional[int]:
    for key in ('umbrella_reduced_bias_nk', 'umbrella_bias_kcal_mol_nk', 'umbrella_bias_kj_mol_nk'):
        if key in f.files:
            arr = f[key]
            if getattr(arr, 'ndim', 0) == 2:
                return int(arr.shape[1])
    if 'umbrella_reduced_bias_kn' in f.files and n_samples is not None:
        arr = f['umbrella_reduced_bias_kn']
        if getattr(arr, 'ndim', 0) == 2:
            if arr.shape[0] == n_samples:
                return int(arr.shape[1])
            if arr.shape[1] == n_samples:
                return int(arr.shape[0])
    return None


def _discover_analysis_chunk_paths(prod: Path, manifest: Optional[dict] = None, notes: Optional[list[str]] = None) -> list[Path]:
    chunk_dir = prod / 'analysis_chunks'
    found: dict[str, Path] = {}
    manifest_count = 0
    if isinstance(manifest, dict):
        for chunk in manifest.get('chunks', []) or []:
            name = Path(str(chunk.get('path', ''))).name
            if not name:
                continue
            manifest_count += 1
            p = chunk_dir / name
            if p.exists():
                found[p.name] = p
            elif notes is not None:
                notes.append(f'analysis_chunks: missing {p.name}; skipped.')
    if chunk_dir.exists():
        for p in sorted(chunk_dir.glob('*.npz')):
            found.setdefault(p.name, p)
    paths = [found[name] for name in sorted(found)]
    if notes is not None and manifest_count and len(paths) > manifest_count:
        notes.append(f'analysis_chunks: manifest lists {manifest_count} chunk(s), discovered {len(paths)} chunk file(s); loading discovered files.')
    return paths


def _append_npz_arrays(merged: dict[str, list], f, source: str, expected_k: Optional[int], notes: list[str]) -> tuple[Optional[int], Optional[int], bool]:
    n_samples = _npz_sample_count_open(f)
    file_k = _npz_window_count_open(f, n_samples)
    if expected_k is not None and file_k is not None and file_k != expected_k:
        notes.append(f'analysis_chunks: {source} has {file_k} windows vs {expected_k}; skipped.')
        return n_samples, file_k, False

    has_nk = 'umbrella_reduced_bias_nk' in f.files
    for key in f.files:
        if key not in _ANALYSIS_VECTOR_KEYS:
            continue
        arr = np.asarray(f[key])
        out_key = key

        if key == 'umbrella_reduced_bias_kn':
            if has_nk:
                continue
            if n_samples is None or arr.ndim != 2:
                notes.append(f'analysis_chunks: {source} has unusable umbrella_reduced_bias_kn shape {arr.shape}; skipped.')
                continue
            if arr.shape[0] == n_samples:
                out_key = 'umbrella_reduced_bias_nk'
            elif arr.shape[1] == n_samples:
                arr = arr.T
                out_key = 'umbrella_reduced_bias_nk'
            else:
                notes.append(f'analysis_chunks: {source} has umbrella_reduced_bias_kn shape {arr.shape} for {n_samples} samples; skipped.')
                continue
        elif n_samples is not None and arr.ndim >= 1 and arr.shape[0] != n_samples:
            notes.append(f'analysis_chunks: {source} key {key} shape {arr.shape} is not sample-major for {n_samples} samples; skipped.')
            continue

        existing = merged.get(out_key)
        if existing:
            ref = existing[0]
            if arr.ndim != ref.ndim or arr.shape[1:] != ref.shape[1:]:
                notes.append(f'analysis_chunks: {source} key {out_key} shape {arr.shape} incompatible with existing {ref.shape}; skipped.')
                continue
        merged.setdefault(out_key, []).append(arr)
    return n_samples, file_k, True


def _load_merged_arrays(prod: Path) -> tuple['_Arrays', list[str]]:
    """Load analysis_arrays.npz plus any chunk files, tolerating stale manifests."""
    merged: dict[str, list] = {}
    notes: list[str] = []
    expected_k: Optional[int] = None

    npz_path = prod / 'analysis_arrays.npz'
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as f:
            n_samples, file_k, loaded = _append_npz_arrays(merged, f, npz_path.name, expected_k, notes)
            if loaded and file_k is not None:
                expected_k = file_k

    manifest_path = prod / 'analysis_chunks_manifest.json'
    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:
            notes.append(f'Could not read analysis_chunks_manifest.json: {exc}')
    chunk_paths = _discover_analysis_chunk_paths(prod, manifest, notes)
    added = 0
    for p in chunk_paths:
        try:
            with np.load(p, allow_pickle=False) as f:
                n_samples, file_k, loaded = _append_npz_arrays(merged, f, p.name, expected_k, notes)
                if loaded:
                    added += 1
                    if expected_k is None and file_k is not None:
                        expected_k = file_k
        except Exception as exc:
            notes.append(f'analysis_chunks: could not read {p.name}: {exc}')
    if added:
        notes.append(f'Merged {added} analysis chunk file(s) with analysis_arrays.npz.')

    if not merged:
        return {}, notes

    result: dict = {k: np.concatenate(vs, axis=0) for k, vs in merged.items() if vs}

    # Deduplicate by (step, replica) to handle any NPZ/chunk overlap
    if 'step' in result and 'replica' in result:
        n = int(result['step'].size)
        _rep_max = int(result['replica'].max()) + 1 if n > 0 else 1
        combined = result['step'].astype(np.int64) * _rep_max + result['replica'].astype(np.int64)
        _, first_idx = np.unique(combined, return_index=True)
        keep = np.zeros(n, dtype=bool)
        keep[first_idx] = True
        if not np.all(keep):
            result = {k: v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n else v
                      for k, v in result.items()}
            notes.append(f'Removed {int((~keep).sum())} duplicate samples from NPZ/chunk overlap.')

    return _Arrays(result), notes


def _load_secondary_cv_from_csv(samples_csv: Path, expected_size: int) -> np.ndarray:
    """Load secondary_cv column from samples.csv; returns all-NaN if unavailable or size mismatch."""
    result = []
    try:
        with samples_csv.open(newline='') as f:
            for row in csv.DictReader(f):
                try:
                    result.append(float(row.get('secondary_cv','') or row.get('cv2_A','') or row.get('cv2','')))
                except Exception:
                    result.append(float('nan'))
    except Exception:
        pass
    arr = np.asarray(result, float)
    return arr if arr.size == expected_size else np.full(expected_size, np.nan)


def load_npz(prod: Path) -> Data:
    arr, load_notes = _load_merged_arrays(prod)
    if not arr:
        raise FileNotFoundError(f'No samples in {prod}/analysis_arrays.npz or analysis_chunks/')
    meta=rjson(prod/'analysis_arrays_metadata.json',{}); meta.update(rjson(prod/'umbrella_pymbar_metadata.json',{}))
    if load_notes:
        meta.setdefault('load_notes', []).extend(load_notes)
    cv=np.asarray(arr['cv_A'],float)
    cv2=np.full(cv.shape,np.nan,dtype=float)
    for _cv2_key in ('cv2_A', 'secondary_cv'):
        if _cv2_key in arr.files:
            cv2=np.asarray(arr[_cv2_key],float)
            break
    if not np.any(np.isfinite(cv2)) and (prod/'samples.csv').exists():
        cv2=_load_secondary_cv_from_csv(prod/'samples.csv', cv.size)
    rg=np.full(cv.shape,np.nan,dtype=float)
    for name in ('rg_A','radius_gyration_A','radius_of_gyration_A'):
        if name in arr.files:
            rg=np.asarray(arr[name],float)
            break
    window=np.asarray(arr['window'],int) if 'window' in arr.files else np.zeros(cv.size,int)
    replica=np.asarray(arr['replica'],int) if 'replica' in arr.files else np.zeros(cv.size,int)
    step=np.asarray(arr['step'],int) if 'step' in arr.files else np.arange(cv.size,dtype=int)
    if 'umbrella_reduced_bias_nk' in arr.files: u=np.asarray(arr['umbrella_reduced_bias_nk'],float)
    elif 'umbrella_reduced_bias_kn' in arr.files: u=np.asarray(arr['umbrella_reduced_bias_kn'],float).T
    else: raise KeyError('analysis_arrays.npz lacks umbrella_reduced_bias_nk/kn')
    centers,ks,rows=read_windows(prod/'umbrella_windows.csv')
    if centers.size==0 and 'center_A' in arr.files:
        centers=np.full(u.shape[1],np.nan); ca=np.asarray(arr['center_A'],float)
        for k in range(u.shape[1]):
            vals=ca[window==k]
            if vals.size: centers[k]=float(np.nanmedian(vals))
    if centers.size==0: centers=np.arange(u.shape[1],dtype=float)
    if ks.size==0 and 'k_kcal_mol_A2' in arr.files:
        ks=np.full(u.shape[1],np.nan); ka=np.asarray(arr['k_kcal_mol_A2'],float)
        for k in range(u.shape[1]):
            vals=ka[window==k]
            if vals.size: ks[k]=float(np.nanmedian(vals))
    if ks.size==0: ks=np.full(u.shape[1],np.nan)
    temp,beta=infer_temp_beta(prod,meta,arr)
    boost=np.full(cv.shape,np.nan)
    for name in ('gamd_boost_total_kj_mol','gamd_boost_kj_mol','boost_kj_mol'):
        if name in arr.files: boost=np.asarray(arr[name],float); break
    pot=None
    for name in ('potential_kj_mol','potential_energy_kj_mol'):
        if name in arr.files: pot=np.asarray(arr[name],float); break
    meta['umbrella_window_rows']=rows
    return clean(Data(prod,prod/'pmf_analysis',cv,cv2,rg,window,replica,step,u,centers,ks,beta,temp,boost,pot,str(prod/'analysis_arrays.npz'),meta))


def _csv_row_count_fast(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            n += block.count(b'\n')
    return max(0, n - 1)


def _npz_sample_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with np.load(path, allow_pickle=False) as f:
            n = _npz_sample_count_open(f)
            return int(n or 0)
    except Exception:
        return 0


def _analysis_binary_sample_count(prod: Path) -> int:
    total = _npz_sample_count(prod / 'analysis_arrays.npz')
    for p in _discover_analysis_chunk_paths(prod):
        total += _npz_sample_count(p)
    return int(total)


def _window_float_array(rows: list[dict], keys: tuple[str, ...], default: float) -> np.ndarray:
    vals = []
    for row in rows:
        raw = ''
        for key in keys:
            raw = row.get(key, '')
            if raw not in (None, '', 'nan', 'None'):
                break
        try:
            vals.append(float(raw))
        except Exception:
            vals.append(float(default))
    return np.asarray(vals, dtype=np.float64)


def load_csv(prod: Path, load_notes: Optional[list[str]] = None) -> Data:
    meta=rjson(prod/'umbrella_pymbar_metadata.json',{})
    if load_notes:
        meta.setdefault('load_notes', []).extend(load_notes)
    centers,ks,rows=read_windows(prod/'umbrella_windows.csv')
    temp,beta=infer_temp_beta(prod,meta,None)
    cv=[]; cv2=[]; rg=[]; win=[]; rep=[]; step=[]; boost=[]; boost_dih=[]; pot=[]; urows=[]
    vector_keys = (
        'umbrella_reduced_bias_all_windows_json', 'umbrella_reduced_bias_all_windows',
        'umbrella_bias_all_windows_kj_mol_json', 'umbrella_bias_all_windows_kj_mol',
    )
    with (prod/'samples.csv').open(newline='') as f:
        reader=csv.DictReader(f)
        has_vectors=any(k in (reader.fieldnames or []) for k in vector_keys)
        has_components='gamd_boost_components_kj_mol_json' in (reader.fieldnames or [])
        for row in reader:
            try: c=float(row['cv_A'])
            except Exception: continue
            cv.append(c)
            try: cv2.append(float(row.get('secondary_cv','') or row.get('cv2_A','') or row.get('cv2','')))
            except Exception: cv2.append(float('nan'))
            try: rg.append(float(row.get('rg_A','') or row.get('radius_gyration_A','') or row.get('radius_of_gyration_A','')))
            except Exception: rg.append(float('nan'))
            win.append(int(float(row.get('window',0) or 0))); rep.append(int(float(row.get('replica',0) or 0))); step.append(int(float(row.get('step',len(step)) or len(step))))
            try: boost.append(float(row.get('gamd_boost_total_kj_mol','') or row.get('gamd_boost_kj_mol','')))
            except Exception: boost.append(float('nan'))
            if has_components:
                try:
                    _comps=json.loads(row.get('gamd_boost_components_kj_mol_json') or '{}')
                    _dih=next((v for k,v in _comps.items() if 'dihedral' in k.lower() or 'torsion' in k.lower()),None)
                    boost_dih.append(float(_dih) if _dih is not None else float('nan'))
                except Exception: boost_dih.append(float('nan'))
            else: boost_dih.append(float('nan'))
            try: pot.append(float(row.get('potential_kj_mol','')))
            except Exception: pot.append(float('nan'))
            if has_vectors:
                vec=[]
                for key in ('umbrella_reduced_bias_all_windows_json','umbrella_reduced_bias_all_windows'):
                    if row.get(key): vec=jvec(row[key]); break
                if not vec:
                    for key in ('umbrella_bias_all_windows_kj_mol_json','umbrella_bias_all_windows_kj_mol'):
                        if row.get(key): vec=[beta*x for x in jvec(row[key])]; break
                urows.append(vec)
    cv=np.asarray(cv,float); cv2=np.asarray(cv2,float); rg=np.asarray(rg,float); win=np.asarray(win,int); rep=np.asarray(rep,int); step=np.asarray(step,int); boost=np.asarray(boost,float); boost_dih=np.asarray(boost_dih,float); pot=np.asarray(pot,float)
    if any(len(v)>0 for v in urows):
        K=max(len(v) for v in urows); u=np.full((len(urows),K),np.nan)
        for i,v in enumerate(urows):
            if v: u[i,:len(v)]=np.asarray(v,float)
    else:
        if centers.size==0 or ks.size==0: raise ValueError('samples.csv lacks all-window biases and umbrella_windows.csv is incomplete')
        if centers.size != ks.size:
            raise ValueError('umbrella_windows.csv has inconsistent center/k columns')
        K=int(centers.size)
        sec_centers=_window_float_array(rows, ('secondary_cv_center','secondary_center','secondary','ss0','secondary_cv_target'), np.nan)
        sec_ks=_window_float_array(rows, ('secondary_cv_k_kcal_mol','secondary_k_kcal_mol','secondary_cv_k_kcal','ss_k_kcal_mol','secondary_k'), 0.0)
        has_secondary=(sec_centers.size==K and sec_ks.size==K and np.any(np.isfinite(sec_centers) & np.isfinite(sec_ks) & (np.abs(sec_ks)>0)))
        scale=beta*KJ_PER_KCAL
        u=np.empty((cv.size,K),dtype=np.float64)
        for k in range(K):
            total=0.5*float(ks[k])*(cv-float(centers[k]))**2
            if has_secondary and np.isfinite(sec_centers[k]) and np.isfinite(sec_ks[k]) and float(sec_ks[k]) != 0.0:
                total=total + 0.5*float(sec_ks[k])*(cv2-float(sec_centers[k]))**2
            u[:,k]=scale*total
        if has_secondary:
            meta.setdefault('load_notes', []).append('samples.csv lacks all-window bias vectors; reconstructed full primary+secondary umbrella bias matrix from umbrella_windows.csv.')
        else:
            meta.setdefault('load_notes', []).append('samples.csv lacks all-window bias vectors; reconstructed full primary umbrella bias matrix from umbrella_windows.csv.')
    if centers.size==0: centers=np.arange(u.shape[1],dtype=float)
    if ks.size==0: ks=np.full(u.shape[1],np.nan)
    meta['umbrella_window_rows']=rows
    _boost_dih_arg = boost_dih if np.any(np.isfinite(boost_dih)) else None
    return clean(Data(prod,prod/'pmf_analysis',cv,cv2,rg,win,rep,step,u,centers,ks,beta,temp,boost,pot,str(prod/'samples.csv'),meta,boost_dih_kj=_boost_dih_arg))


def _parquet_sample_count(prod: Path) -> int:
    """Count rows across Parquet sample chunks without loading full data."""
    try:
        import duckdb
        files = sorted((prod / 'samples').glob('**/*.parquet'))
        if not files: return 0
        conn = duckdb.connect()
        n = conn.execute("SELECT count(*) FROM read_parquet(?)", [[str(f) for f in files]]).fetchone()
        conn.close()
        return int(n[0]) if n else 0
    except Exception:
        return 0


def load_parquet(prod: Path) -> Data:
    """Load MBAR inputs from new Parquet sample format (gareus >= 2026.05 package).

    Reads samples/{seg_id}/chunk_*.parquet via gareus.query.load_samples(),
    reconstructs the full N×K reduced-bias matrix analytically from stored
    CV values and window definitions, and returns a Data object compatible
    with all downstream MBAR/PMF analysis functions.
    """
    try:
        from gareus.query import load_samples, load_windows, reconstruct_bias_matrix
    except ImportError as exc:
        raise ImportError(
            f'gareus package required for Parquet loading: {exc}. '
            'Run from the gareus project directory or install with pip install -e .'
        ) from exc

    samples = load_samples(prod)
    if not samples or 'cv1' not in samples:
        raise FileNotFoundError(f'No Parquet sample data found in {prod}/samples/')

    windows = load_windows(prod)
    if not windows:
        raise FileNotFoundError(f'No window snapshot found in {prod}/windows/')

    meta = rjson(prod / 'umbrella_pymbar_metadata.json', {})
    meta.update(rjson(prod / 'gareus_metadata.json', {}))
    temp, beta = infer_temp_beta(prod, meta)

    cv       = samples['cv1'].astype(np.float64)
    cv2_raw  = samples.get('cv2')
    cv2      = cv2_raw.astype(np.float64) if cv2_raw is not None else np.full(cv.shape, np.nan)
    # window/replica are stored on-disk as uint16 (see gareus/store.py's Parquet
    # schema); downstream consumers already defensively re-cast to int64 before
    # use, so keep them narrow here rather than widening to int32 for no reason.
    window   = samples['window_id'].astype(np.int16)
    step     = samples['step'].astype(np.int64)
    replica  = samples['replica'].astype(np.int16) if 'replica' in samples else np.zeros(cv.shape, dtype=np.int16)
    boost_raw      = samples.get('gamd_boost_total')
    boost          = boost_raw.astype(np.float64) if boost_raw is not None else np.full(cv.shape, np.nan)
    boost_dih_raw  = samples.get('gamd_boost_dihedral')
    boost_dih      = boost_dih_raw.astype(np.float64) if boost_dih_raw is not None else np.full(cv.shape, np.nan)
    pot_raw  = samples.get('potential')
    potential= pot_raw.astype(np.float64) if pot_raw is not None else None

    # Reconstruct full N×K dimensionless reduced-bias matrix
    cv2_for_nk = cv2_raw.astype(np.float64) if cv2_raw is not None else None
    u_nk = reconstruct_bias_matrix(cv, cv2_for_nk, windows, beta)

    centers = np.array([float(w['center1']) for w in windows])
    k_kcal  = np.array([float(w['k1'])      for w in windows])

    rows = []
    wcsv = prod / 'umbrella_windows.csv'
    if wcsv.exists():
        with wcsv.open(newline='') as f:
            rows = list(csv.DictReader(f))
    meta['umbrella_window_rows'] = rows
    meta['parquet_windows']      = windows

    _boost_dih_arg = boost_dih if np.any(np.isfinite(boost_dih)) else None
    return clean(Data(
        prod_dir=prod, out_dir=prod / 'pmf_analysis',
        cv=cv, cv2=cv2, rg_A=np.full(cv.shape, np.nan),
        window=window, replica=replica, step=step,
        u_nk=u_nk, centers=centers, k_kcal=k_kcal,
        beta=beta, temp=temp, boost_kj=boost, potential_kj=potential,
        source=str(prod / 'samples'), meta=meta,
        boost_dih_kj=_boost_dih_arg,
    ))


def prod_dir_of(path: Path) -> Path:
    path=Path(path).resolve()
    def _has_parquet(p: Path) -> bool:
        return (p/'segments.json').exists() and (p/'samples').is_dir()
    def _has_data(p: Path) -> bool:
        return (p/'analysis_arrays.npz').exists() or (p/'samples.csv').exists() or (p/'analysis_chunks_manifest.json').exists()
    def _has_adaptive_parquet(p: Path) -> bool:
        ap = p/'adaptive_production'
        if not ap.is_dir():
            return False
        has_registry = (ap/'final_registry_used_for_mbar.csv').exists() or (ap/'state_registry.csv').exists()
        has_parquet = any(ap.glob('*/samples/**/*.parquet'))
        if has_registry and has_parquet:
            return True
        return bool(_find_adaptive_epoch_dirs(ap))
    if _has_parquet(path) or _has_data(path): return path
    fp=path/'final_production'
    if _has_parquet(fp) or _has_data(fp): return fp
    ap=path/'adaptive_production'
    if (ap/'adaptive_union_mbar.npz').exists(): return ap
    if _has_adaptive_parquet(path): return ap
    if _has_epoch_csv_layout(ap): return ap
    # Interrupted adaptive_production run: union-MBAR artifacts (registry,
    # epoch_window_map.csv) are only written once an epoch finalizes.  If the
    # driver was interrupted mid-epoch, the completed epoch dir is still a
    # self-contained single-production run (samples/ + segments.json + windows/)
    # and can be analyzed on its own.  Resolve to the latest such epoch; it goes
    # through the standard load_parquet path (not union MBAR, which needs the
    # registry that was never written).  `path` itself may already be the
    # adaptive_production dir when the user points at it directly.
    for base in (ap, path):
        solo = _find_selfcontained_epoch_dirs(base)
        if solo:
            return solo[-1]
    raise FileNotFoundError(
        f'No analyzable samples found for {path}. Checked {path}/ and {fp}/ '
        f'(Parquet samples/, analysis_arrays.npz, samples.csv) and {ap}/ '
        f'(adaptive_union_mbar.npz, epoch Parquet/CSV, self-contained epoch dirs). '
        f'If this is an interrupted adaptive_production run, point directly at a '
        f'completed epoch dir, e.g. {ap/"epoch_000"}.'
    )


def load_data(inp: Path, out: Optional[Path], source: str = 'auto', no_augment: bool = False,
              n_threads: int = 0, n_workers: int = 4,
              epoch_ids: Optional[set[int]] = None) -> Data:
    prod=prod_dir_of(inp)
    # Adaptive-production: prefer new Parquet epoch data, fall back to legacy NPZ.
    if prod.name == 'adaptive_production':
        has_registry = (prod / 'final_registry_used_for_mbar.csv').exists() or (prod / 'state_registry.csv').exists()
        has_epoch_parquet = has_registry and any(prod.glob('*/samples/**/*.parquet'))
        if not has_epoch_parquet:
            has_epoch_parquet = bool(_find_adaptive_epoch_dirs(prod, epoch_ids=epoch_ids))
        if has_epoch_parquet:
            d = load_parquet_adaptive_union(
                prod, n_threads=n_threads, n_workers=n_workers, epoch_ids=epoch_ids)
        elif (prod / 'adaptive_union_mbar.npz').exists():
            if epoch_ids is not None:
                raise ValueError(
                    '--epoch requires per-epoch Parquet or CSV inputs; '
                    'adaptive_union_mbar.npz cannot be subset safely.')
            d = load_union_npz(prod)
        elif _has_epoch_csv_layout(prod, epoch_ids=epoch_ids):
            d = load_epoch_csv_adaptive(prod, epoch_ids=epoch_ids)
        else:
            selected = f' for requested epoch(s) {sorted(epoch_ids)}' if epoch_ids is not None else ''
            raise FileNotFoundError(
                'adaptive_production/ has neither epoch Parquet data nor '
                f'adaptive_union_mbar.npz nor epoch CSV layout in {prod}{selected}')
        if out is not None: d.out_dir = Path(out)
        return d
    requested=str(source or 'auto').strip().lower()
    has_parquet=(prod/'segments.json').exists() and (prod/'samples').is_dir()
    has_npz=(prod/'analysis_arrays.npz').exists() or (prod/'analysis_chunks_manifest.json').exists() or (prod/'analysis_chunks').exists()
    has_csv=(prod/'samples.csv').exists()
    notes: list[str] = []
    if requested == 'parquet':
        if not has_parquet:
            raise FileNotFoundError(f'No segments.json + samples/ in {prod}')
        d = load_parquet(prod)
    elif requested == 'npz':
        if not has_npz:
            raise FileNotFoundError(f'No analysis_arrays.npz or analysis_chunks/ in {prod}')
        d = load_npz(prod)
    elif requested == 'csv':
        if not has_csv:
            raise FileNotFoundError(f'No samples.csv in {prod}')
        d = load_csv(prod, notes)
    elif requested == 'auto':
        if has_parquet and not has_npz and not has_csv:
            d = load_parquet(prod)
        elif has_parquet and (has_npz or has_csv):
            parquet_n = _parquet_sample_count(prod)
            npz_n = _analysis_binary_sample_count(prod) if has_npz else 0
            csv_n = _csv_row_count_fast(prod/'samples.csv') if has_csv else 0
            best = max(parquet_n, npz_n, csv_n)
            if parquet_n >= best:
                d = load_parquet(prod)
                notes.append(f'Auto-selected Parquet ({parquet_n} rows vs npz~{npz_n} csv~{csv_n}).')
            elif npz_n >= csv_n:
                d = load_npz(prod)
            else:
                d = load_csv(prod, notes)
        elif has_npz and has_csv:
            csv_n=_csv_row_count_fast(prod/'samples.csv')
            bin_n=_analysis_binary_sample_count(prod)
            if csv_n > bin_n:
                notes.append(f'Auto-selected samples.csv ({csv_n} rows vs ~{bin_n} binary).')
                d = load_csv(prod, notes)
            else:
                d = load_npz(prod)
        elif has_npz:
            d = load_npz(prod)
        elif has_csv:
            d = load_csv(prod, notes)
        else:
            raise FileNotFoundError(f'No Parquet samples/, analysis_arrays.npz, analysis_chunks/, or samples.csv in {prod}')
    else:
        raise ValueError(f'Unknown analysis source {source!r}; use auto, parquet, npz, or csv')
    if out is not None: d.out_dir=Path(out)
    if not no_augment:
        run_dir = prod.parent if prod.name == 'final_production' else prod
        if _find_gareus_round_dirs(run_dir):
            d = _augment_with_adaptive_rounds(d, run_dir)
            if out is not None: d.out_dir = Path(out)
    return d
