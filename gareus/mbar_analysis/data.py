"""Data-loading domain: the ``Data`` dataclass and its lifecycle helpers.

Relocated verbatim from ``analyze_gareus_mbar.py`` (Plan A2 of the
MBAR-analysis modularization sequence; see
docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a2-design.md).
No behavior change from the original script versions.
"""
from __future__ import annotations

import csv
import json
import math
import re
import warnings as _warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from gareus.units import KJ_PER_KCAL, K_B_KJ_PER_MOL_K


def rjson(path: Path, default=None):
    try:
        return json.loads(path.read_text()) if path.exists() else ({} if default is None else default)
    except Exception:
        return {} if default is None else default


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x=float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def wjson(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default))


def read_windows(path: Path):
    centers=[]; ks=[]; rows=[]
    if not path.exists(): return np.array([]), np.array([]), rows
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
            c=row.get('center_A') or row.get('center') or row.get('center_a')
            k=row.get('k_kcal_mol_A2') or row.get('k_kcal_A2') or row.get('k')
            if c not in (None,''): centers.append(float(c))
            if k not in (None,''): ks.append(float(k))
    return np.asarray(centers,float), np.asarray(ks,float), rows


def jvec(txt):
    if txt is None or txt=='': return []
    v=json.loads(txt)
    if isinstance(v,dict): return [float(v[k]) for k in sorted(v, key=lambda x:int(x) if str(x).isdigit() else str(x))]
    return [float(x) for x in v]


@dataclass
class Data:
    prod_dir: Path
    out_dir: Path
    cv: np.ndarray
    cv2: np.ndarray
    rg_A: np.ndarray
    window: np.ndarray
    replica: np.ndarray
    step: np.ndarray
    u_nk: np.ndarray
    centers: np.ndarray
    k_kcal: np.ndarray
    beta: float
    temp: float
    boost_kj: np.ndarray
    potential_kj: Optional[np.ndarray]
    source: str
    meta: dict[str,Any]
    boost_dih_kj: Optional[np.ndarray] = None  # dihedral-only component of GaMD boost


def infer_temp_beta(prod: Path, meta: dict, arrays=None):
    if arrays is not None:
        for name in ('beta_1_over_kJ_mol','beta_1_over_kj_mol','beta'):
            if name in arrays.files:
                a=np.asarray(arrays[name],float)
                if a.size and np.isfinite(a.flat[0]) and a.flat[0]>0:
                    beta=float(a.flat[0]); return 1.0/(K_B_KJ_PER_MOL_K*beta), beta
        for name in ('temperature_K','temperature_k'):
            if name in arrays.files:
                a=np.asarray(arrays[name],float)
                if a.size and np.isfinite(np.nanmedian(a)):
                    t=float(np.nanmedian(a)); return t, 1.0/(K_B_KJ_PER_MOL_K*t)
    if meta is None:
        meta = {}
    for key in ('temperature_K','temperature_k','temperature'):
        if key in meta:
            try:
                t=float(meta[key]);
                if t>0: return t, 1.0/(K_B_KJ_PER_MOL_K*t)
            except Exception: pass
    # Top-level lookup above misses run_manifest.json-shaped meta dicts, where the
    # real value lives nested under 'method_settings'/'resolved_args' (see
    # gareus/provenance.py's _method_settings/_public_args). Check those too before
    # falling through to the less-reliable run_args.json file search below.
    for key_path in (
        ('method_settings', 'temperature_k'), ('method_settings', 'temperature_K'),
        ('resolved_args', 'temperature_k'), ('resolved_args', 'temperature_K'),
    ):
        val = meta
        for k in key_path:
            val = val.get(k) if isinstance(val, dict) else None
            if val is None:
                break
        if val is not None:
            try:
                t = float(val)
                if t > 0:
                    return t, 1.0/(K_B_KJ_PER_MOL_K*t)
            except (TypeError, ValueError):
                pass
    # Shallowest-first: prod itself, then one level up (final_production/epoch_NNN
    # style callers), then two levels up (covers epoch_dir -> adaptive_production ->
    # run_root, where the real run_args.json lives at the run root).
    for p in (prod/'run_args.json', prod.parent/'run_args.json', prod.parent.parent/'run_args.json'):
        m=rjson(p,{})
        for key in ('temperature_k','temperature_K','temperature'):
            if key in m:
                t=float(m[key]); return t, 1.0/(K_B_KJ_PER_MOL_K*t)
    t=300.0
    msg=(f'infer_temp_beta: could not resolve run temperature for {prod} from arrays, '
         f'meta, or run_args.json (checked {prod}, {prod.parent}, {prod.parent.parent}); '
         f'falling back to default {t:.1f} K. MBAR reduced-bias energies will be wrong '
         f'if the real run temperature differs.')
    if isinstance(meta, dict):
        meta.setdefault('load_notes', []).append(msg)
    _warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return t, 1.0/(K_B_KJ_PER_MOL_K*t)


def clean(d: Data) -> Data:
    if d.u_nk.shape[0] != d.cv.size: raise ValueError('u_nk/sample count mismatch')
    if d.boost_kj.shape != d.cv.shape: d.boost_kj=np.full(d.cv.shape,np.nan)
    if d.cv2.shape != d.cv.shape: d.cv2=np.full(d.cv.shape,np.nan)
    if d.rg_A.shape != d.cv.shape: d.rg_A=np.full(d.cv.shape,np.nan)
    mask=np.isfinite(d.cv) & np.all(np.isfinite(d.u_nk),axis=1)
    if mask.all():
        # Nothing to filter: boolean fancy indexing always copies in NumPy,
        # even when the mask keeps every element, so skip the copies below
        # entirely in the common (fully-finite) case. The boost_dih_kj shape
        # normalization just below is independent of sample finiteness (it
        # only depends on whether the array's own length already matches the
        # sample count) and must still run regardless of this fast path.
        if d.boost_dih_kj is not None and d.boost_dih_kj.size!=mask.size: d.boost_dih_kj=None
        return d
    d.cv=d.cv[mask]; d.cv2=d.cv2[mask]; d.rg_A=d.rg_A[mask]; d.window=d.window[mask]; d.replica=d.replica[mask]; d.step=d.step[mask]; d.u_nk=d.u_nk[mask]; d.boost_kj=d.boost_kj[mask]
    if d.potential_kj is not None and d.potential_kj.size==mask.size: d.potential_kj=d.potential_kj[mask]
    if d.boost_dih_kj is not None and d.boost_dih_kj.size==mask.size: d.boost_dih_kj=d.boost_dih_kj[mask]
    elif d.boost_dih_kj is not None: d.boost_dih_kj=None
    return d


def _masked_data(d: 'Data', mask: np.ndarray, meta_override: Optional[dict] = None) -> 'Data':
    """Row-slice a Data by a boolean sample mask.

    K-length/scalar fields (window-space arrays, beta, temp, ...) are shared
    with the original -- only per-sample arrays (including meta['_epoch_source'],
    when present) are sliced. Used to run the existing single-regime analysis
    functions unmodified against a subset of samples (one secondary-CV regime,
    or the epoch_000/rest split, at a time).
    """
    def _sl(arr):
        return arr[mask] if arr is not None else None
    meta_out = dict(meta_override if meta_override is not None else d.meta)
    _epoch_src_meta = meta_out.get('_epoch_source')
    if _epoch_src_meta is not None and len(_epoch_src_meta) == mask.size:
        meta_out['_epoch_source'] = np.asarray(_epoch_src_meta)[mask].tolist()
    return Data(
        prod_dir=d.prod_dir, out_dir=d.out_dir,
        cv=_sl(d.cv), cv2=_sl(d.cv2), rg_A=_sl(d.rg_A),
        window=_sl(d.window), replica=_sl(d.replica), step=_sl(d.step),
        u_nk=d.u_nk[mask] if d.u_nk is not None else None,
        centers=d.centers, k_kcal=d.k_kcal,
        beta=d.beta, temp=d.temp,
        boost_kj=_sl(d.boost_kj),
        potential_kj=_sl(d.potential_kj),
        source=d.source, meta=meta_out,
        boost_dih_kj=_sl(d.boost_dih_kj),
    )


def _filter_epoch_source(d: Data, keep: np.ndarray) -> None:
    """Filter d.meta['_epoch_source'] in-place to match the keep mask."""
    src = d.meta.get('_epoch_source')
    if src is not None and len(src) == keep.size:
        arr = np.asarray(src, dtype=np.int64)
        d.meta['_epoch_source'] = arr[keep].tolist()


def _sample_block_ids(d: 'Data') -> np.ndarray:
    """Per-sample block id for the fixed-f_k block bootstrap (see
    docs/superpowers/specs/2026-08-13-pmf-bootstrap-uncertainty-design.md):
    one block = one (epoch_source, replica) pair, matching one replica's
    samples within one epoch/phase -- treating the same replica index reused
    in a later epoch as a NEW block, since adaptive-production runs don't
    guarantee trajectory continuity across epoch boundaries. Falls back to
    `replica` alone when `d.meta['_epoch_source']` is absent (single-source/
    non-adaptive-production runs).

    Returns a compact 0..M-1 int64 array, same length as d.replica.
    """
    replica = np.asarray(d.replica)
    epoch_src = d.meta.get('_epoch_source')
    if epoch_src is not None and len(epoch_src) != replica.size:
        # Stale relative to replica/cv/etc. -- clean() row-filters per-sample
        # arrays by a finiteness mask but doesn't sync this metadata list, so
        # it can be longer than replica.size on any path where clean() ever
        # dropped a sample. Same defensive length check every other consumer
        # of _epoch_source in this file already applies; fall back to
        # replica-only blocks rather than let np.stack raise below.
        epoch_src = None
    if epoch_src is None:
        keys = replica.reshape(-1, 1)
    else:
        keys = np.stack([np.asarray(epoch_src), replica], axis=1)
    _, block_ids = np.unique(keys, axis=0, return_inverse=True)
    return block_ids.reshape(-1).astype(np.int64)


def _skip_first_n_frames(d: Data, n: int) -> Data:
    """Drop first n samples per replica (sorted by step) for equilibration burn-in."""
    keep = np.ones(d.cv.size, dtype=bool)
    for rep in np.unique(d.replica):
        idx = np.where(d.replica == rep)[0]
        order = np.argsort(d.step[idx])
        keep[idx[order[:min(n, idx.size)]]] = False
    d.cv=d.cv[keep]; d.cv2=d.cv2[keep]; d.rg_A=d.rg_A[keep]
    d.window=d.window[keep]; d.replica=d.replica[keep]; d.step=d.step[keep]
    d.u_nk=d.u_nk[keep]; d.boost_kj=d.boost_kj[keep]
    if d.potential_kj is not None and d.potential_kj.size==keep.size:
        d.potential_kj=d.potential_kj[keep]
    if d.boost_dih_kj is not None and d.boost_dih_kj.size==keep.size:
        d.boost_dih_kj=d.boost_dih_kj[keep]
    _filter_epoch_source(d, keep)
    return d


def _apply_analysis_stride(d: Data, stride: int, offset: int = 0) -> Data:
    """Keep every Nth saved analysis sample per replica after any burn-in cut.

    This is a data-level stride: all per-sample arrays stay aligned, and every
    downstream MBAR/FES/convergence calculation sees the same subsampled
    data.  The stride is applied independently within each replica after sorting
    by production step, so replica time traces remain internally consistent.
    """
    stride=max(1,int(stride or 1))
    offset=max(0,int(offset or 0))
    if stride <= 1 and offset <= 0:
        return d
    keep=np.zeros(d.cv.size,dtype=bool)
    for rep in np.unique(d.replica):
        idx=np.where(d.replica==rep)[0]
        if idx.size == 0:
            continue
        order=idx[np.argsort(d.step[idx],kind='stable')]
        if offset < order.size:
            keep[order[offset::stride]]=True
    if not np.any(keep):
        raise ValueError(f'analysis stride/offset kept zero samples: stride={stride}, offset={offset}')
    before=int(d.cv.size)
    d.cv=d.cv[keep]; d.cv2=d.cv2[keep]; d.rg_A=d.rg_A[keep]
    d.window=d.window[keep]; d.replica=d.replica[keep]; d.step=d.step[keep]
    d.u_nk=d.u_nk[keep]; d.boost_kj=d.boost_kj[keep]
    if d.potential_kj is not None and d.potential_kj.size==keep.size:
        d.potential_kj=d.potential_kj[keep]
    if d.boost_dih_kj is not None and d.boost_dih_kj.size==keep.size:
        d.boost_dih_kj=d.boost_dih_kj[keep]
    _filter_epoch_source(d, keep)
    d.meta.setdefault('load_notes',[]).append(f'Applied analysis stride {stride} with offset {offset}: kept {int(d.cv.size)}/{before} samples.')
    d.meta['analysis_stride']=int(stride)
    d.meta['analysis_stride_offset']=int(offset)
    return d


def _epoch_dir_index(path: Path) -> Optional[int]:
    """Return ``epoch_NNN`` index, or ``None`` for non-epoch directories."""
    match = re.fullmatch(r'epoch_(\d+)', path.name)
    return int(match.group(1)) if match else None


def _epoch_number_for_run_dir(run_dir: Path) -> Optional[int]:
    """Literal epoch_NNN number for a run_dir, checking both the flat
    (epoch_NNN/) and baseline/topup_* sub-run (epoch_NNN/{baseline,topup_*}/)
    layouts. None for non-numbered dirs (e.g. final/*).
    """
    idx = _epoch_dir_index(run_dir)
    if idx is not None:
        return idx
    return _epoch_dir_index(run_dir.parent)


def _epoch_run_manifest_secondary_cv_type(run_dir) -> str:
    """Return ``resolved_args.secondary_cv`` from one epoch/phase run_dir's own
    ``run_manifest.json`` (e.g. ``"torsion-pca"`` or ``"tica-linear"``), or
    ``''`` if unavailable. This is that specific epoch's own recorded config,
    not whatever a state's row in the live registry says today.
    """
    manifest = rjson(Path(run_dir) / 'run_manifest.json', {})
    return str((manifest.get('resolved_args') or {}).get('secondary_cv') or '')


def _epoch_zero_split_masks(d: 'Data') -> Optional[tuple]:
    """(mask_epoch0, mask_rest) for a Data with real epoch_000 samples alongside
    later-epoch samples, else None.

    epoch_000 is systematically different from later epochs in ways that make
    pooling it into the main PMF/GaMD-boost report misleading, not just
    inconsistent style: the GaMD shared-envelope recalibration
    (``_maybe_recalibrate_gamd_boost``, ``gareus/adaptive_production.py``)
    recalibrates the boost envelope from epoch 0's own real sampling and fires
    at most once, so epoch 0 runs under a *different* GaMD envelope than every
    later epoch; and the tICA CV2 auto-switch typically also fires after
    epoch 0. Callers should treat the whole Data as one report (unchanged)
    when this returns None -- e.g. non-adaptive-production sources, or a run
    with only epoch_000 and nothing else to compare it against.
    """
    epoch_src = d.meta.get('_epoch_source')
    run_dirs = d.meta.get('adaptive_epoch_run_dirs')
    if not epoch_src or not run_dirs:
        return None
    epoch_src = np.asarray(epoch_src, dtype=np.int64)
    if epoch_src.size != len(d.cv) or int(epoch_src.max()) >= len(run_dirs):
        return None
    epoch_numbers = []
    for rd in run_dirs:
        n = _epoch_number_for_run_dir(Path(rd))
        epoch_numbers.append(-1 if n is None else n)
    epoch_numbers = np.asarray(epoch_numbers, dtype=np.int64)
    sample_epoch_numbers = epoch_numbers[epoch_src]
    mask0 = sample_epoch_numbers == 0
    mask_rest = ~mask0
    if not np.any(mask0) or not np.any(mask_rest):
        return None
    return mask0, mask_rest


def _secondary_cv_epoch_regime_masks(d: 'Data', warnings: Optional[list] = None) -> Optional[dict]:
    """Group this Data's samples by which secondary-CV *type* was actually
    active when each one was sampled, using every epoch/phase run_dir's own
    ``run_manifest.json`` (``d.meta['adaptive_epoch_run_dirs']``, indexed by
    ``d.meta['_epoch_source']``).

    Two different secondary-CV modes (e.g. torsion-pca vs tica-linear across
    the tICA CV2 auto-switch, ``gareus/adaptive_production.py``) are not a
    recentering of one coordinate -- they are different linear projections of
    the same raw torsion features, i.e. genuinely different order parameters.
    Pooling their raw cv2 values into one axis for a combined PMF/2D-FES
    conflates two different physical quantities. This does *not* affect
    MBAR's f_k/weights themselves (those are already correct after the
    per-epoch-native bias fix in ``load_parquet_adaptive_union``) -- only
    which samples' cv2 values get binned together for CV2-facing plots.

    Returns ``None`` when there is only one regime (the overwhelming
    majority of runs) -- callers should fall back to the existing
    single-pass analysis, unchanged. Otherwise returns
    ``{regime_type: (mask, is_dominant)}``, where exactly one regime is
    "dominant": the one containing the *last* epoch/phase, matching the
    "last phase wins" convention already used elsewhere in this pipeline
    for CV2 labeling. Any epoch/phase whose own ``run_manifest.json`` is
    unreadable is folded into the dominant regime (with a warning) rather
    than silently dropping its samples from every regime-specific plot.
    """
    epoch_src = d.meta.get('_epoch_source')
    run_dirs = d.meta.get('adaptive_epoch_run_dirs')
    if not epoch_src or not run_dirs:
        return None
    epoch_src = np.asarray(epoch_src, dtype=np.int64)
    if epoch_src.size != len(d.cv2) or int(epoch_src.max()) >= len(run_dirs):
        return None
    regime_by_epoch = [_epoch_run_manifest_secondary_cv_type(rd) for rd in run_dirs]
    distinct = sorted({r for r in regime_by_epoch if r})
    if len(distinct) < 2:
        return None

    def _run_dir_mtime(rd):
        try:
            return Path(rd).stat().st_mtime
        except OSError:
            return -1.0

    chronological = sorted(range(len(run_dirs)), key=lambda i: _run_dir_mtime(run_dirs[i]))
    dominant = next((regime_by_epoch[i] for i in reversed(chronological) if regime_by_epoch[i]), distinct[-1])
    unresolved = [i for i, r in enumerate(regime_by_epoch) if not r]
    if unresolved and warnings is not None:
        warnings.append(
            f'{len(unresolved)} adaptive-production epoch/phase run_dir(s) had no '
            f"readable secondary_cv type in run_manifest.json; folded into the "
            f"dominant regime ({dominant!r}) for the per-regime CV2 breakdown."
        )
    out: dict = {}
    for regime in distinct:
        idxs = [i for i, r in enumerate(regime_by_epoch) if r == regime]
        if regime == dominant:
            idxs = idxs + unresolved
        out[regime] = (np.isin(epoch_src, idxs), regime == dominant)
    return out
