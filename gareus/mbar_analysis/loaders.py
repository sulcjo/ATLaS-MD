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
from .data import Data, clean, infer_temp_beta, rjson, read_windows, jvec, _fill_masked_nan
from .loaders_adaptive import (
    _find_adaptive_epoch_dirs, _find_selfcontained_epoch_dirs,
    _find_adaptive_epoch_csv_sources, _has_epoch_csv_layout,
    load_epoch_csv_adaptive, load_union_npz,
    _find_gareus_round_dirs, _augment_with_adaptive_rounds,
    _PHANTOM_EPOCH_WINDOW_BASE, _STALE_WINDOW_MAP_DOC,
    _STALE_WINDOW_MAP_OVERRIDE_ENV, _phase_label,
    _phase_recorded_window_count, _read_epoch_window_map_rows,
    _validate_and_repair_epoch_window_map,
    window_map_note_reports_a_fault, window_map_note_rewrote_rows,
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
    cv=[]; cv2=[]; rg=[]; win=[]; rep=[]; step=[]; boost=[]; boost_dih=[]; pot=[]; urows=[]; vpep=[]; vdih=[]; lam=[]
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
            try: vpep.append(float(row.get('v_pep_kj_mol','')))
            except Exception: vpep.append(float('nan'))
            try: vdih.append(float(row.get('v_dih_kj_mol','')))
            except Exception: vdih.append(float('nan'))
            try: lam.append(float(row.get('gamd_lambda','') or 0.0))
            except Exception: lam.append(float('nan'))
            if has_vectors:
                vec=[]
                for key in ('umbrella_reduced_bias_all_windows_json','umbrella_reduced_bias_all_windows'):
                    if row.get(key): vec=jvec(row[key]); break
                if not vec:
                    for key in ('umbrella_bias_all_windows_kj_mol_json','umbrella_bias_all_windows_kj_mol'):
                        if row.get(key): vec=[beta*x for x in jvec(row[key])]; break
                urows.append(vec)
    cv=np.asarray(cv,float); cv2=np.asarray(cv2,float); rg=np.asarray(rg,float); win=np.asarray(win,int); rep=np.asarray(rep,int); step=np.asarray(step,int); boost=np.asarray(boost,float); boost_dih=np.asarray(boost_dih,float); pot=np.asarray(pot,float); vpep=np.asarray(vpep,float); vdih=np.asarray(vdih,float); lam=np.asarray(lam,float)
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
    # Per-state (per-window) λ, derived from the per-sample gamd_lambda column
    # by grouping on the sample's own window index -- umbrella_windows.csv (the
    # `rows` above) does not itself carry gamd_lambda, mirroring load_parquet's
    # windows/<segment>.json gap. Every replica sampling a given window under
    # an active ladder is assigned that window's own fixed rung, so nanmedian
    # per group is a robust reduction; a window with no finite samples (or no
    # samples at all) defaults to 0.0, the documented "ladder inactive" value.
    K=int(u.shape[1])
    state_lambdas=np.zeros(K,dtype=np.float64)
    if np.any(np.isfinite(lam)):
        for k in range(K):
            grp=lam[(win==k)&np.isfinite(lam)]
            if grp.size:
                state_lambdas[k]=float(np.nanmedian(grp))
    # `u` above (either branch) is pure umbrella -- the ladder boost is never
    # embedded in the stored per-sample vectors or in the analytic
    # reconstruction, so it must be added here explicitly, under the same
    # "never silently reweight without the term" invariant as
    # reconstruct_bias_matrix/build_union_state_mbar_inputs (see their
    # docstrings): reporting meta['gamd_ladder']=True while `u` secretly
    # omitted the term would make select_unbiased_method skip the GaMD
    # correction and silently discard real physics.
    if np.any(state_lambdas>0.0):
        if not np.any(np.isfinite(vpep)):
            raise ValueError(
                f'{prod}: windows carry gamd_lambda > 0 (derived from the per-sample gamd_lambda '
                f'column) but no sample has a finite v_pep_kj_mol; the λ-ladder cannot be '
                f'reweighted without the raw channel energies'
            )
        if not np.any(np.isfinite(vdih)):
            raise ValueError(
                f'{prod}: windows carry gamd_lambda > 0 but no sample has a finite v_dih_kj_mol '
                f'(v_pep_kj_mol is present); the λ-ladder cannot be reweighted without the raw '
                f'channel energies'
            )
        envelope_path=prod/'shared_gamd_setup_globals.json'
        if not envelope_path.exists():
            raise ValueError(
                f'{prod}: windows carry gamd_lambda > 0 but the frozen GaMD envelope '
                f'{envelope_path} does not exist; v_pep/v_dih cannot be reweighted under the '
                f'ladder without it'
            )
        from gareus.pep_gamd import PepGamdEnvelope, pep_gamd_boost_matrix_kj
        envelope=PepGamdEnvelope.from_json(envelope_path)
        u = u + beta*pep_gamd_boost_matrix_kj(vpep, vdih, state_lambdas, envelope).T
    meta['gamd_ladder']=bool(np.any(state_lambdas>0.0))
    _boost_dih_arg = boost_dih if np.any(np.isfinite(boost_dih)) else None
    return clean(Data(prod,prod/'pmf_analysis',cv,cv2,rg,win,rep,step,u,centers,ks,beta,temp,boost,pot,str(prod/'samples.csv'),meta,boost_dih_kj=_boost_dih_arg,v_pep_kj=vpep,v_dih_kj=vdih,state_lambdas=state_lambdas))


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
    cv2      = _fill_masked_nan(cv2_raw) if cv2_raw is not None else np.full(cv.shape, np.nan)
    # window/replica are stored on-disk as uint16 (see gareus/store.py's Parquet
    # schema); downstream consumers already defensively re-cast to int64 before
    # use, so keep them narrow here rather than widening to int32 for no reason.
    window   = samples['window_id'].astype(np.int16)
    step     = samples['step'].astype(np.int64)
    replica  = samples['replica'].astype(np.int16) if 'replica' in samples else np.zeros(cv.shape, dtype=np.int16)
    # gamd_boost_total/gamd_boost_dihedral are real SQL NULLs for every sample
    # of a non-GaMD/plain-umbrella run (gareus/production.py's
    # extract_gamd_boost_kj returns None for a non-GaMD integrator) -- route
    # through _fill_masked_nan like cv2 above, not a bare .astype(), so
    # Data.boost_kj/boost_dih_kj are always plain NaN-filled float64 arrays,
    # never a live MaskedArray reaching downstream mask-unaware consumers
    # (e.g. analyze_gareus_mbar.py's np.nanstd(d.boost_kj)/boost_stats()).
    # potential isn't proven reachable-null at the current writer, but there's
    # no cost to guarding it the same way, and casing 2 of 3 near-identical
    # columns differently would be worse style than treating all 3 alike.
    boost_raw      = samples.get('gamd_boost_total')
    boost          = _fill_masked_nan(boost_raw) if boost_raw is not None else np.full(cv.shape, np.nan)
    boost_dih_raw  = samples.get('gamd_boost_dihedral')
    boost_dih      = _fill_masked_nan(boost_dih_raw) if boost_dih_raw is not None else np.full(cv.shape, np.nan)
    pot_raw  = samples.get('potential')
    potential= _fill_masked_nan(pot_raw) if pot_raw is not None else None
    # λ-ladder columns: real SQL NULLs on any segment predating the schema
    # addition, or on a run where the ladder was never active -- same
    # _fill_masked_nan treatment as gamd_boost_total/gamd_boost_dihedral above.
    v_pep_raw = samples.get('v_pep_kj_mol')
    v_pep     = _fill_masked_nan(v_pep_raw) if v_pep_raw is not None else np.full(cv.shape, np.nan)
    v_dih_raw = samples.get('v_dih_kj_mol')
    v_dih     = _fill_masked_nan(v_dih_raw) if v_dih_raw is not None else np.full(cv.shape, np.nan)
    lambda_raw    = samples.get('gamd_lambda')
    lambda_sample = _fill_masked_nan(lambda_raw) if lambda_raw is not None else np.full(cv.shape, np.nan)

    centers = np.array([float(w['center1']) for w in windows])
    k_kcal  = np.array([float(w['k1'])      for w in windows])
    # Per-state (per-window) λ: windows/<segment>.json does not itself carry
    # a "gamd_lambda" key today, so it is derived from the per-sample column
    # by grouping on window_id -- every replica sampling a given window under
    # an active ladder is assigned that window's own fixed rung, so the
    # per-window values are constant modulo NaN noise; nanmedian is robust to
    # the rare stale/missing row. A window with zero matching finite samples
    # (e.g. never sampled) defaults to 0.0, matching the "ladder inactive"
    # convention documented on Data.state_lambdas.
    state_lambdas = np.zeros(len(windows), dtype=np.float64)
    if np.any(np.isfinite(lambda_sample)):
        window_i64 = window.astype(np.int64)
        for i, w in enumerate(windows):
            explicit = w.get('gamd_lambda')
            if explicit is not None:
                state_lambdas[i] = float(explicit)
                continue
            wid = int(w.get('window_id', i))
            grp = lambda_sample[(window_i64 == wid) & np.isfinite(lambda_sample)]
            if grp.size:
                state_lambdas[i] = float(np.nanmedian(grp))

    # Reconstruct full N×K dimensionless reduced-bias matrix. When any derived
    # state carries a nonzero rung, the ladder boost must actually be embedded
    # here -- reporting meta['gamd_ladder']=True while u_nk secretly omitted
    # the term would make select_unbiased_method skip the GaMD correction
    # entirely (see its docstring) and silently discard real physics, so this
    # loader is held to the same "never silently reweight without the term"
    # invariant as reconstruct_bias_matrix itself (see its docstring) and
    # build_union_state_mbar_inputs.
    cv2_for_nk = _fill_masked_nan(cv2_raw) if cv2_raw is not None else None
    windows_for_nk = windows
    if np.any(state_lambdas > 0.0):
        if not np.any(np.isfinite(v_pep)):
            raise ValueError(
                f'{prod}: windows carry gamd_lambda > 0 (derived from the per-sample gamd_lambda '
                f'column) but no sample has a finite v_pep_kj_mol; the λ-ladder cannot be '
                f'reweighted without the raw channel energies'
            )
        if not np.any(np.isfinite(v_dih)):
            raise ValueError(
                f'{prod}: windows carry gamd_lambda > 0 but no sample has a finite v_dih_kj_mol '
                f'(v_pep_kj_mol is present); the λ-ladder cannot be reweighted without the raw '
                f'channel energies'
            )
        envelope_path = prod / 'shared_gamd_setup_globals.json'
        if not envelope_path.exists():
            raise ValueError(
                f'{prod}: windows carry gamd_lambda > 0 but the frozen GaMD envelope '
                f'{envelope_path} does not exist; v_pep/v_dih cannot be reweighted under the '
                f'ladder without it'
            )
        from gareus.pep_gamd import PepGamdEnvelope
        envelope = PepGamdEnvelope.from_json(envelope_path)
        windows_for_nk = [dict(w, gamd_lambda=float(state_lambdas[i])) for i, w in enumerate(windows)]
        u_nk = reconstruct_bias_matrix(cv, cv2_for_nk, windows_for_nk, beta, v_pep=v_pep, v_dih=v_dih, envelope=envelope)
    else:
        u_nk = reconstruct_bias_matrix(cv, cv2_for_nk, windows_for_nk, beta)

    rows = []
    wcsv = prod / 'umbrella_windows.csv'
    if wcsv.exists():
        with wcsv.open(newline='') as f:
            rows = list(csv.DictReader(f))
    meta['umbrella_window_rows'] = rows
    meta['parquet_windows']      = windows
    meta['gamd_ladder']          = bool(np.any(state_lambdas > 0.0))

    _boost_dih_arg = boost_dih if np.any(np.isfinite(boost_dih)) else None
    return clean(Data(
        prod_dir=prod, out_dir=prod / 'pmf_analysis',
        cv=cv, cv2=cv2, rg_A=np.full(cv.shape, np.nan),
        window=window, replica=replica, step=step,
        u_nk=u_nk, centers=centers, k_kcal=k_kcal,
        beta=beta, temp=temp, boost_kj=boost, potential_kj=potential,
        source=str(prod / 'samples'), meta=meta,
        boost_dih_kj=_boost_dih_arg,
        v_pep_kj=v_pep, v_dih_kj=v_dih, state_lambdas=state_lambdas,
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


# ---------------------------------------------------------------------------
# Stale `epoch_window_map.csv` guard -- shared entry points for the consumers
# that are NOT the union-Parquet loader
# ---------------------------------------------------------------------------
#
# `_validate_and_repair_epoch_window_map` (loaders_adaptive.py) is the single
# implementation of the auto-drop staleness check; loaders_union_parquet.py
# calls it per epoch. Three other consumers read the same file raw, so the
# same mis-attribution bug (~15% of a real 12.1M-sample campaign, see
# docs/chignolin_6_low_ess_root_cause.md) reached them untouched:
#
#   * compare_gareus_runs.py's `load_npz_adaptive_union` -- a parallel union
#     implementation for legacy NPZ-only adaptive runs. It produces comparison
#     PMF numbers, so it calls the guard directly and fails closed exactly
#     like the Parquet loader does.
#   * `load_union_npz` -- reads `adaptive_union_mbar.npz`, whose per-sample
#     state attribution the *driver* baked in at run time from these same
#     maps. It cannot be re-derived from the npz, so the only honest check is
#     of the maps it was built from: `check_union_npz_window_map_provenance`.
#   * plot_adaptive_diagnostics.py -- figures, not free energies. It gets
#     `figure_epoch_window_map_rows`, which repairs where it can and annotates
#     loudly where it cannot, instead of refusing to draw.
#
# These wrappers exist so those callers share one call into the guard and one
# statement of the policy, rather than each growing its own variant.


def _stale_window_map_override_enabled() -> bool:
    """Whether ``GAREUS_ALLOW_STALE_WINDOW_MAP`` downgrades a refusal to a warning.

    Read here rather than left to the guard because the union-npz check below
    has to make ONE decision over many phases: with the override set the guard
    stops raising and reports through its notes instead, so a wrapper that
    treated "any note" as fatal would invert the override's documented job.
    """
    return os.environ.get(_STALE_WINDOW_MAP_OVERRIDE_ENV, '').strip().lower() in ('1', 'true', 'yes')


def _row_epoch_window(row: dict) -> int:
    """A map row's local window index, or -1 when it has no parsable one."""
    try:
        return int(row['epoch_window'])
    except (KeyError, TypeError, ValueError):
        return -1


def _reachable_state_by_window(rows: list) -> dict:
    """``{local_window: state_id}`` over the rows a sample can actually reach.

    Phantom rows (a repair parks the dropped states' rows past
    `_PHANTOM_EPOCH_WINDOW_BASE`) are excluded, so this is directly comparable
    between a stale row list and its repair: same keys iff the repair moved
    nothing.
    """
    out = {}
    for r in rows:
        ew = _row_epoch_window(r)
        if 0 <= ew < _PHANTOM_EPOCH_WINDOW_BASE:
            try:
                out[ew] = int(r['state_id'])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def figure_epoch_window_map_rows(phase_dir, wmap_rows: list,
                                 window_ids=None, cv2=None) -> tuple:
    """``(rows, note)`` for a *figure* consumer of one phase's window map.

    Same guard, deliberately non-fatal policy: a diagnostic plotter that
    refuses to draw when a run looks odd is useless exactly when it is
    reached for, so an unrepairable map is still drawn -- from the stale rows
    -- and the caller gets a note to print and to stamp on the figure. `note`
    is None when the guard found nothing wrong with the map (the overwhelming
    majority of phases) and, per the paragraph below, also when all it had to
    report is that a cross-check could not run; the rows then come back
    untouched. It is NOT None for anything the guard actually found.

    Unlike the MBAR path, the returned rows are the SURVIVORS ONLY. A repair
    keeps the dropped states' rows, parked at `_PHANTOM_EPOCH_WINDOW_BASE`+,
    because MBAR still needs their epoch-native window params to back those
    states' bias columns when it evaluates *other* epochs' samples against
    them. A figure has no such consumer and would instead draw one window
    ellipse per row -- re-adding the never-run windows the repair just took
    out. Same returned list, two consumers wanting opposite halves of it.

    A note saying only that a cross-check could NOT RUN comes back as `None`
    here, deliberately, and this is the one consumer where that is right. The
    plotter renders whatever it is handed under one fixed heading
    (`plot_adaptive_diagnostics._annotate_map_warnings` stamps "STALE
    epoch_window_map.csv" on the figure), so passing it a note about an absent
    check would print a claim of staleness this guard did not make -- on a
    figure, where there is no room for the qualification. The same note still
    reaches the MBAR and run-summary consumers, whose surfaces carry its own
    wording and its own severity.
    """
    try:
        rows, notes = _validate_and_repair_epoch_window_map(
            phase_dir, list(wmap_rows), window_ids=window_ids, cv2=cv2)
    except ValueError as exc:
        return list(wmap_rows), (
            f'[stale window map] {exc} FIGURES BELOW ARE DRAWN FROM THE STALE MAP: every '
            f'per-state sample count and window label for {_phase_label(phase_dir)} may belong '
            f'to a different state.')
    # Both decisions are taken on the note OBJECTS, before any copy of their
    # text: `kind` does not survive an f-string, and a note that loses it grades
    # as a fault (see WindowMapNote). Notes arrive one at a time today; the
    # `any`/list forms are so a second one could never silently change which
    # branch is taken.
    faults = [n for n in notes if window_map_note_reports_a_fault(n)]
    if not faults:
        return list(rows), None
    if not any(window_map_note_rewrote_rows(n) for n in notes):
        # A detected fault the guard could not repair (or an override load): the
        # rows are the map's own, unrenumbered, so there is no phantom half to
        # take out and no contiguity to protect. Drawing them is what this
        # consumer is for; the note says they may be wrong.
        return list(rows), faults[0]
    # The guard rewrote the list, so every row that belongs to a real local
    # window carries a freshly renumbered 0..N-1 `epoch_window`. A row that
    # still fails to parse (-1) is therefore one the repair could not place,
    # and it must not travel into a frame whose index column is otherwise
    # contiguous -- hence the lower bound as well as the phantom upper bound.
    survivors = [r for r in rows
                 if 0 <= _row_epoch_window(r) < _PHANTOM_EPOCH_WINDOW_BASE]
    # The ONE returned note has to be the one that describes what happened to
    # the returned rows, because that is the only thing the caller has left to
    # ask: `plot_adaptive_diagnostics._phase_window_map` decides whether to
    # follow the repair by testing `window_map_note_rewrote_rows` on it. Notes
    # arrive one at a time today, so this picks the same object `faults[0]`
    # does -- it is here for the same reason as the `any` above, so that a
    # second note could never make the plotter draw the UNREPAIRED frame under
    # a note that says REPAIRED IN MEMORY.
    return survivors, next((n for n in faults if window_map_note_rewrote_rows(n)), faults[0])


def _union_npz_phase_dirs_with_maps(ap_dir: Path) -> list:
    """The phase dirs whose ``epoch_window_map.csv`` the driver read when it
    built ``adaptive_union_mbar.npz``.

    Mirrors `_epoch_sample_sources` (gareus/adaptive_production.py): the union
    builder walks ``final`` and ``final_extension_NNN`` -- plus every
    ``epoch_NNN`` when it was built with ``include_epochs`` -- taking each run
    root and its ``baseline``/``topup_*`` children, and resolves every sample's
    state through that directory's own map. ``adaptive_union_mbar.json``
    records which epoch policy was used, so a stale numbered epoch does not
    condemn an npz that never read it; the policy defaults to the inclusive
    (checks more) side when the json is missing or silent. Pilot run roots can
    live outside `ap_dir` entirely and are not covered here.
    """
    ap_dir = Path(ap_dir)
    include_epochs = bool(rjson(ap_dir / 'adaptive_union_mbar.json', {}).get('include_epochs', True))
    roots = []
    if include_epochs:
        roots.extend(sorted(ap_dir.glob('epoch_[0-9][0-9][0-9]')))
    roots.append(ap_dir / 'final')
    roots.extend(sorted(ap_dir.glob('final_extension_[0-9][0-9][0-9]')))
    out = []
    for root in roots:
        if not root.is_dir():
            continue
        for cand in [root] + sorted(c for c in root.iterdir()
                                    if c.is_dir() and (c.name == 'baseline'
                                                       or c.name.startswith('topup_'))):
            if (cand / 'epoch_window_map.csv').exists():
                out.append(cand)
    return out


def check_union_npz_window_map_provenance(ap_dir: Path) -> list:
    """Notes on whether ``adaptive_union_mbar.npz``'s state attribution is sound.

    The npz stores `sampled_state_ids` -- the mapping already applied -- and
    not the local window indices it was applied to, so nothing about the
    mapping can be recomputed, let alone repaired, from the npz itself. What
    *can* be checked is the evidence the driver used: each phase's own
    `epoch_window_map.csv`, run through the same guard the Parquet loader
    uses. A phase the guard would repair or refuse is a phase whose map was
    stale while the driver was reading it, so the npz's baked-in attribution
    for that phase's samples is wrong on disk.

    Returns notes for the caller to print and to carry into the run summary:

    * ``[]`` -- every phase map checked out. Untouched behaviour.
    * an ``[unverified window map]`` note -- some or all of the phases record
      nothing about how many windows they really ran, so the check has nothing
      to compare against there. Reported rather than passed over silently:
      "not checkable" and "checked, fine" are different claims, and this
      artifact predates the guard entirely.
    * any ``[window map check] Could not cross-check ...`` note the guard
      produced, passed straight through. Same principle one level down: the
      phase's row count checked out and its *membership* could not be
      cross-checked, which is neither a clean bill of health nor a fault. This
      used to land in the stale bucket below and RAISE -- refusing to load a
      healthy run on the strength of an absent check, the exact inversion the
      note kinds exist to prevent (see `window_map_note_reports_a_fault`).
    Raises ValueError on any stale map, because the npz cannot be repaired the
    way per-epoch Parquet can -- re-analysing from the epoch data is the fix.
    ``GAREUS_ALLOW_STALE_WINDOW_MAP=1`` downgrades that to a note (inspection
    only; the free energies are then wrong by construction).

    A stale map does not always mis-attribute samples: when the phantom rows
    all sit *past* the phase's last real window, local indices 0..N-1 name the
    same states before and after the repair (real case: RUNS/chignolin_5's
    final/baseline -- 30 rows for 29 windows, the single phantom row last).
    That is reported, because the severity differs, but it is NOT an exemption.
    It is reported only for a phase the guard actually REPAIRED, since it is a
    statement about the repair: a detected fault the guard could not repair has
    no "after" to compare against, and the message says that instead.
    The reason is the half of the damage this check cannot see: a window
    dropped post-pull was never retired from the state registry either, and
    `build_union_state_mbar_inputs` builds a column over every registry state
    marked usable -- so the never-run window can still have entered the solve
    as its own state, with an N_k this function has no way to attribute to a
    phase. Passing such an npz on the strength of a check that does not cover
    that would be exactly the silent-wrong-number trade this guard exists to
    refuse.
    """
    ap_dir = Path(ap_dir)
    override = _stale_window_map_override_enabled()
    phase_dirs = _union_npz_phase_dirs_with_maps(ap_dir)
    stale: list = []
    verified: list = []
    unverifiable: list = []
    could_not_check: list = []
    for phase_dir in phase_dirs:
        rows = _read_epoch_window_map_rows(phase_dir)
        if not rows:
            unverifiable.append(_phase_label(phase_dir))
            continue
        try:
            # No `window_ids`/`cv2`: the npz pools every phase's samples into
            # one flat array with the local indices already resolved away, so
            # this phase's own sample columns are not recoverable here. The
            # row-count check against the phase's recorded window count still
            # has teeth -- 126 of the 152 real phase maps in this repo's RUNS/
            # tree carry that record.
            repaired, notes = _validate_and_repair_epoch_window_map(phase_dir, rows)
        except ValueError as exc:
            stale.append(f'{_phase_label(phase_dir)}: {exc}')
            continue
        # On the note objects, before anything copies their text -- `kind` does
        # not survive an f-string, and the `stale.append` below makes exactly
        # such a copy.
        faults = [n for n in notes if window_map_note_reports_a_fault(n)]
        if not faults:
            could_not_check.extend(notes)
            if _phase_recorded_window_count(phase_dir) is None:
                unverifiable.append(_phase_label(phase_dir))
            else:
                verified.append(_phase_label(phase_dir))
            continue
        # Stale on disk, which is what the driver read. Fatal either way; the
        # qualifier below only sizes the damage in the message.
        #
        # Sizing it is a claim about a REPAIR, so it may only be made when
        # there was one -- and that is a structural question the note answers
        # directly. Both of the things that used to answer it here were proxies
        # for it, and both were wrong in the same direction:
        #
        #   * `before == after`. For a fault the guard detects but cannot
        #     repair (MAP_NOTE_FAULT_UNREPAIRED -- an equal-count map listing a
        #     genuinely different window set) it hands back the caller's own
        #     rows, so of course they name the same states: the comparison is
        #     true VACUOUSLY, and the message then told the operator that this
        #     phase's "per-sample attribution is unaffected" when its mapping
        #     is not known at all. A reassuring sentence inside a refusal is the
        #     worst possible place for one.
        #   * `not override`. The override is consulted only on the refusal
        #     path (`_fail`); BOTH repair paths -- the row-count
        #     survivors+phantoms rebuild and the equal-count permutation
        #     reorder -- run regardless of it, because it never suppresses a
        #     repair that is possible. So it withheld a qualifier that was true
        #     on a repairable phase read under the override.
        #
        # Taken on the note OBJECTS, before `stale.append` makes the first copy
        # of their text: `kind` does not survive an f-string. These are the
        # guard's own objects, uncopied, so the kind is always there; the
        # unlabelled fallback is a safety net that grades as not-rewritten,
        # which declines to size the damage rather than mis-sizing it.
        rewrote = any(window_map_note_rewrote_rows(n) for n in notes)
        if not rewrote:
            qualifier = (' (no corrected mapping could be derived for this phase, so which of its '
                         'local windows name the wrong state -- and how many samples that moves -- '
                         'cannot be sized from these artifacts)')
        else:
            # `after` is non-empty for every repair the guard makes (survivors
            # are renumbered from 0), but it is spelled out rather than relied
            # on: an empty mapping would make `all()` vacuously true and put
            # back exactly the false reassurance this branch exists to remove.
            before = _reachable_state_by_window(rows)
            after = _reachable_state_by_window(repaired)
            unshifted = bool(after) and all(before.get(ew) == sid for ew, sid in after.items())
            qualifier = ''
            if unshifted:
                qualifier = (' (this phase\'s own local windows name the same states before and '
                             'after the repair, so its per-sample attribution is unaffected -- but '
                             'a window dropped post-pull was never retired from the state registry '
                             'either, so the never-run window may still have entered the npz\'s '
                             'solve as its own state, which cannot be checked from the maps)')
        stale.append(f'{_phase_label(phase_dir)}: {faults[0]}{qualifier}')

    if stale:
        detail = (
            f'adaptive_union_mbar.npz in {ap_dir} was built in-run by the driver, which resolved '
            f'every sample\'s umbrella state through each phase\'s own epoch_window_map.csv. '
            f'{len(stale)} of those {len(phase_dirs)} map(s) is/are stale (the '
            f'`--us-auto-drop-bad-windows` renumbering bug): {" | ".join(stale)}')
        remedy = (
            'The npz stores only the already-resolved sampled_state_ids, not the local window '
            'indices they came from, so this CANNOT be repaired in memory the way the per-epoch '
            'Parquet/CSV path is -- re-analyse from the epoch data instead. '
            f'See {_STALE_WINDOW_MAP_DOC}.')
        if override:
            return [f'[stale window map] {detail}. Loaded anyway because '
                    f'{_STALE_WINDOW_MAP_OVERRIDE_ENV} is set: these phases\' samples are '
                    f'attributed to the wrong umbrella states and every free energy derived from '
                    f'them is invalid. {remedy}']
        raise ValueError(
            f'{detail}. Loading it would report free energies built on samples attributed to the '
            f'wrong umbrella states -- refusing to load. {remedy} Set '
            f'{_STALE_WINDOW_MAP_OVERRIDE_ENV}=1 to load the mis-attributed npz regardless '
            f'(for inspection only).')

    notes = []
    if unverifiable:
        shown = unverifiable[:4]
        listed = ', '.join(shown)
        if len(unverifiable) > len(shown):
            listed += f' (+{len(unverifiable) - len(shown)} more)'
        notes.append(
            f'[unverified window map] adaptive_union_mbar.npz\'s per-sample state attribution '
            f'was baked in at run time from {len(phase_dirs)} phase window map(s); '
            f'{len(unverifiable)} of them record nothing about how many windows the phase '
            f'really ran ({listed}), so whether the `--us-auto-drop-bad-windows` staleness bug '
            f'affected this artifact cannot be checked'
            + (f' ({len(verified)} other phase(s) did check out). ' if verified else '. ')
            + f'If this run used that flag, treat these numbers as unverified and re-analyse '
              f'from the per-epoch data, which is checked and repaired in memory. '
              f'See {_STALE_WINDOW_MAP_DOC}.')
    # Verbatim, not summarised: each already names its phase and says exactly
    # what could not be compared against what.
    notes.extend(could_not_check)
    return notes


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
            # The npz's local-window -> state mapping was applied by the driver
            # while the run was live and is not recoverable from the file, so
            # unlike the Parquet path there is nothing here to repair. Check
            # the maps it was built from instead: refuse on a provably stale
            # one, say so when provenance cannot be established, stay silent
            # otherwise. Notes also ride into meta['load_notes'] so they reach
            # pmf_summary.json rather than only the terminal.
            prov_notes = check_union_npz_window_map_provenance(prod)
            d = load_union_npz(prod)
            for _n in prov_notes:
                print(f'    {_n}')
            if prov_notes:
                d.meta['load_notes'] = list(d.meta.get('load_notes') or []) + prov_notes
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
