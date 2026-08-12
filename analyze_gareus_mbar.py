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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
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

try:
    from scipy.signal import find_peaks as _scipy_find_peaks
    SCIPY_SIGNAL_AVAILABLE = True
except Exception:
    _scipy_find_peaks = None
    SCIPY_SIGNAL_AVAILABLE = False

K_B_KJ_PER_MOL_K=0.00831446261815324
KJ_PER_KCAL=4.184

# -----------------------------------------------------------------------------
# MBAR solver configuration defaults
#
# The following module-level variables control advanced MBAR solver behavior.
# They are exposed as command‑line options later and can be overridden at
# runtime.  See the argparse section near the bottom of this file for the
# corresponding --sambar-* and --mbar-anderson-history flags.  If these
# defaults are changed here, remember to update the help text and parser
# default values accordingly.

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
    for key in ('temperature_K','temperature_k','temperature'):
        if key in meta:
            try:
                t=float(meta[key]);
                if t>0: return t, 1.0/(K_B_KJ_PER_MOL_K*t)
            except Exception: pass
    for p in (prod/'run_args.json', prod.parent/'run_args.json'):
        m=rjson(p,{})
        for key in ('temperature_k','temperature_K','temperature'):
            if key in m:
                t=float(m[key]); return t, 1.0/(K_B_KJ_PER_MOL_K*t)
    t=300.0; return t, 1.0/(K_B_KJ_PER_MOL_K*t)

def jvec(txt):
    if txt is None or txt=='': return []
    v=json.loads(txt)
    if isinstance(v,dict): return [float(v[k]) for k in sorted(v, key=lambda x:int(x) if str(x).isdigit() else str(x))]
    return [float(x) for x in v]

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

def _secondary_cv_label(meta: dict) -> str:
    """Extract human-readable secondary CV label from run metadata."""
    raw = (meta or {}).get('secondary_cv', {})
    if isinstance(raw, str):
        low = raw.lower()
        if 'rama' in low: return 'Ramachandran CV'
        if 'torsion-pca' in low or 'bootstrap' in low: return 'Bootstrap torsion PC1'
        if 'tica' in low: return 'tIC1 torsion CV'
        return f'Secondary CV ({raw})'
    sec = raw or {}
    label = sec.get('label', '')
    mode = sec.get('mode', '')
    if label: return label
    if mode == 'torsion-pca': return 'Bootstrap torsion PC1'
    if mode == 'tica-linear': return 'tIC1 torsion CV'
    if mode: return f'Secondary CV ({mode})'
    return 'Secondary CV'

def _secondary_cv_regions(meta: dict) -> list:
    """Return [{value, label}] region annotations for Ramachandran-style secondary CVs."""
    raw = (meta or {}).get('secondary_cv', {})
    sec = raw if isinstance(raw, dict) else {}
    return [{'value': float(r['value']), 'label': r.get('label', r.get('name', ''))}
            for r in sec.get('regions', []) if 'value' in r]


def _epoch_run_manifest_secondary_cv_type(run_dir) -> str:
    """Return ``resolved_args.secondary_cv`` from one epoch/phase run_dir's own
    ``run_manifest.json`` (e.g. ``"torsion-pca"`` or ``"tica-linear"``), or
    ``''`` if unavailable. This is that specific epoch's own recorded config,
    not whatever a state's row in the live registry says today.
    """
    manifest = rjson(Path(run_dir) / 'run_manifest.json', {})
    return str((manifest.get('resolved_args') or {}).get('secondary_cv') or '')


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
    dominant = next((r for r in reversed(regime_by_epoch) if r), distinct[-1])
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


def _masked_data(d: 'Data', mask: np.ndarray, meta_override: Optional[dict] = None) -> 'Data':
    """Row-slice a Data by a boolean sample mask.

    K-length/scalar fields (window-space arrays, beta, temp, ...) are shared
    with the original -- only per-sample arrays are sliced. Used to run the
    existing single-regime analysis functions unmodified against a subset of
    samples (one secondary-CV regime at a time).
    """
    def _sl(arr):
        return arr[mask] if arr is not None else None
    return Data(
        prod_dir=d.prod_dir, out_dir=d.out_dir,
        cv=_sl(d.cv), cv2=_sl(d.cv2), rg_A=_sl(d.rg_A),
        window=_sl(d.window), replica=_sl(d.replica), step=_sl(d.step),
        u_nk=d.u_nk[mask] if d.u_nk is not None else None,
        centers=d.centers, k_kcal=d.k_kcal,
        beta=d.beta, temp=d.temp,
        boost_kj=_sl(d.boost_kj),
        potential_kj=_sl(d.potential_kj),
        source=d.source, meta=(d.meta if meta_override is None else meta_override),
        boost_dih_kj=_sl(d.boost_dih_kj),
    )


def _regime_slug(regime: str) -> str:
    return ''.join(c if (c.isalnum() or c in '-_') else '_' for c in regime) or 'unknown'


def run_secondary_cv_analyses(d: 'Data', args, base_logw: np.ndarray, selected: str,
                               boost_ok: bool, kbt_kcal: float, out: Path,
                               warnings: list, progress: Optional['Progress']) -> tuple:
    """Secondary-CV PMF + CV1xCV2 2D FES, split by secondary-CV regime when the
    run's CV2 definition changed mid-campaign (see
    ``_secondary_cv_epoch_regime_masks``). Single-regime runs (the common
    case) are entirely unaffected: this degrades to the plain unmodified
    calls, writing to the same paths as before.

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
        regime_meta = dict(d.meta); regime_meta['secondary_cv'] = regime
        d_regime = _masked_data(d, mask, meta_override=regime_meta)
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

def _primary_cv_label(meta: dict) -> str:
    label = (meta or {}).get('primary_cv_label', '')
    if label: return label
    mode = (meta or {}).get('primary_cv', '')
    if mode == 'nonlocal-contacts': return 'nonlocal contact fraction'
    return 'CV distance'

def _primary_cv_units(meta: dict) -> str:
    units = (meta or {}).get('primary_cv_units', '')
    if units: return units
    mode = (meta or {}).get('primary_cv', '')
    if mode == 'nonlocal-contacts': return 'dimensionless'
    return 'A'

def _primary_cv_axis_label(meta: dict) -> str:
    label = _primary_cv_label(meta)
    units = _primary_cv_units(meta)
    if units == 'dimensionless': return label
    return f'{label} ({units})'

def _poincare_primary_cv_supported(meta: dict) -> bool:
    meta = meta or {}
    mode = str(meta.get('primary_cv', '') or '').lower()
    label = str(meta.get('primary_cv_label', '') or '').lower()
    units = str(meta.get('primary_cv_units', '') or '').lower()
    return mode == 'nonlocal-contacts' or ('contact' in label and units in {'', 'dimensionless'})

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
        if epoch_ids is not None and _epoch_dir_index(cand) not in epoch_ids:
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


# ---------------------------------------------------------------------------
# Multi-round (adaptive_feedback_round_*) support
# ---------------------------------------------------------------------------

def _find_gareus_round_dirs(run_dir: Path) -> list:
    """Return sorted adaptive_feedback_round_*/ dirs that contain analysis_chunks/*.npz."""
    result = []
    for d in sorted(run_dir.glob('adaptive_feedback_round_*')):
        if d.is_dir() and (d / 'umbrella_windows.csv').exists():
            chunks = d / 'analysis_chunks'
            if chunks.is_dir() and any(chunks.glob('chunk_*.npz')):
                result.append(d)
    return result


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


def _compute_u_nk_analytical(cv1: np.ndarray, cv2: np.ndarray,
                              union_windows: list, beta: float) -> np.ndarray:
    """Compute N×K_union reduced bias matrix analytically from CV values."""
    K = len(union_windows)
    scale = beta * KJ_PER_KCAL
    u = np.empty((cv1.size, K), dtype=np.float64)
    for k, w in enumerate(union_windows):
        dc1 = cv1 - w['primary_center']
        dc2 = cv2 - w['secondary_cv_center']
        bias_kcal = 0.5 * w['primary_k_kcal'] * dc1 ** 2 + 0.5 * w['secondary_k_kcal'] * dc2 ** 2
        u[:, k] = scale * bias_kcal
    return u


def _augment_with_adaptive_rounds(d: Data, run_dir: Path) -> Data:
    """Combine final_production Data with samples from adaptive_feedback_round_* dirs.

    Builds the union window set across all rounds, recomputes u_nk analytically
    for every sample, and returns a new Data with all samples concatenated.
    """
    round_dirs = _find_gareus_round_dirs(run_dir)
    if not round_dirs:
        return d

    all_dirs = round_dirs + [d.prod_dir]
    union_windows = _build_union_window_table(all_dirs)
    K_union = len(union_windows)

    round_data = []
    for rdir in round_dirs:
        raw = _load_round_raw(rdir)
        if raw is None or raw['cv_A'].size == 0:
            continue
        w2u = _round_window_to_union_map(rdir, union_windows)
        raw['window_union'] = np.array([w2u.get(int(w), 0) for w in raw['window']], dtype=int)
        round_data.append(raw)

    if not round_data:
        return d

    final_w2u = _round_window_to_union_map(d.prod_dir, union_windows)
    final_window_union = np.array([final_w2u.get(int(w), int(w)) for w in d.window], dtype=int)

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


def _is_usable_for_mbar(row: dict) -> bool:
    return str(row.get('usable_for_mbar', '')).strip().lower() in ('true', '1', 'yes')


def _merge_missing_usable_states(primary_rows: list, live_rows: list) -> list:
    """Merge usable states present in `live_rows` but absent from `primary_rows`.

    `final_registry_used_for_mbar.csv` is written once, when a run first enters
    its final phase, and is *not* regenerated if the run is later resumed and the
    epoch loop adds more states (e.g. adaptive splits) before re-entering final
    phase. `state_registry.csv` keeps growing as the live source of truth. A
    state_id referenced by a later epoch's samples but missing from the frozen
    snapshot must not be silently excluded (that would drop real samples and bias
    the recovered free energies) — so any usable state_id absent from
    `primary_rows` is appended here, sourced from `live_rows`.
    """
    seen = {int(r['state_id']) for r in primary_rows}
    merged = list(primary_rows)
    for r in live_rows:
        sid = int(r['state_id'])
        if sid in seen:
            continue
        if _is_usable_for_mbar(r):
            merged.append(r)
            seen.add(sid)
    return merged


def _parse_epoch_window_map_native_params(rows: list) -> dict:
    """Return ``{state_id: {primary_center, primary_k, secondary_center, secondary_k}}``.

    These are the window parameters that were *actually in effect* for that
    specific epoch, as recorded in that epoch's own ``epoch_window_map.csv``
    snapshot at the time it ran — as opposed to whatever a state's row in the
    live/final registry says today. A state's secondary (or even primary)
    center/k can be recentered by later epochs (e.g. the tICA CV2 auto-switch
    in ``gareus/adaptive_production.py``, which overwrites
    ``state.secondary_center`` in place); this dict preserves the
    epoch-specific ground truth so bias energies for that epoch's own samples
    can be reconstructed against what was really applied, not what the state
    looks like now.
    """
    def _f(row: dict, key: str, default: float) -> float:
        v = row.get(key, '')
        if v in ('', 'None', 'nan', None):
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    out: dict = {}
    for r in rows:
        if 'state_id' not in r:
            continue
        sid = int(r['state_id'])
        out[sid] = {
            'primary_center': _f(r, 'primary_center', float('nan')),
            'primary_k': _f(r, 'primary_k', float('nan')),
            'secondary_center': _f(r, 'secondary_center', float('nan')),
            'secondary_k': _f(r, 'secondary_k', 0.0),
        }
    return out


def _epoch_bias_param_vectors(native_params: dict, state_ids: list,
                               global_primary_centers: np.ndarray, global_primary_ks: np.ndarray,
                               global_sec_centers: np.ndarray, global_sec_ks: np.ndarray) -> tuple:
    """Per-epoch (primary_center, primary_k, secondary_center, secondary_k) vectors.

    Starts from the global (final-registry) arrays — the existing fallback
    behavior — then overrides entries for any state_id this epoch's own
    ``epoch_window_map.csv`` snapshot actually covers. States created in a
    later epoch (absent from this epoch's snapshot) keep the global fallback,
    which is correct: they didn't exist yet, so there is no "native" value to
    prefer, and no sample from this epoch can be assigned to them anyway.
    """
    pc = global_primary_centers.copy()
    pk = global_primary_ks.copy()
    sc = global_sec_centers.copy()
    sk = global_sec_ks.copy()
    for k, sid in enumerate(state_ids):
        row = native_params.get(sid)
        if row is None:
            continue
        if math.isfinite(row['primary_center']):
            pc[k] = row['primary_center']
        if math.isfinite(row['primary_k']):
            pk[k] = row['primary_k']
        if math.isfinite(row['secondary_center']):
            sc[k] = row['secondary_center']
        if math.isfinite(row['secondary_k']):
            sk[k] = row['secondary_k']
    return pc, pk, sc, sk


def _reconstruct_union_bias_block(cv: np.ndarray, cv2: np.ndarray, beta: float,
                                   primary_centers: np.ndarray, primary_ks: np.ndarray,
                                   sec_centers: np.ndarray, sec_ks: np.ndarray) -> np.ndarray:
    """Build one epoch-block's N x K reduced-bias-energy matrix.

    Pure function (no I/O) so the bias math can be unit-tested independently
    of Parquet/DuckDB loading. Same formula as the legacy single-snapshot
    reconstruction, just parameterized so callers can pass per-epoch-native
    center/k vectors instead of one static array shared across every epoch.
    """
    n = len(cv)
    k_count = len(primary_centers)
    u = np.zeros((n, k_count), dtype=np.float64)
    for k in range(k_count):
        d1 = cv - primary_centers[k]
        u[:, k] = beta * 4.184 * 0.5 * primary_ks[k] * d1 * d1
        if math.isfinite(sec_centers[k]) and sec_ks[k] > 0:
            d2 = np.where(np.isfinite(cv2), cv2 - sec_centers[k], 0.0)
            u[:, k] += beta * 4.184 * 0.5 * sec_ks[k] * d2 * d2
    return u


def _load_epoch_task(epoch_dir: Path, wmap_path: Path, n_threads: int) -> tuple:
    """Load one epoch's samples, window map, and native bias params (runs in a thread)."""
    from gareus.query import load_samples
    with wmap_path.open(newline='') as f:
        wmap_rows = list(csv.DictReader(f))
    wmap = {int(r['epoch_window']): int(r['state_id']) for r in wmap_rows}
    native_params = _parse_epoch_window_map_native_params(wmap_rows)
    samples = load_samples(epoch_dir, n_threads=n_threads)
    ep_meta = rjson(epoch_dir / 'umbrella_pymbar_metadata.json', {})
    return samples, wmap, ep_meta, native_params


def load_parquet_adaptive_union(adaptive_dir: Path, n_threads: int = 0, n_workers: int = 4,
                                epoch_ids: Optional[set[int]] = None) -> Data:
    """Load MBAR inputs from adaptive-production Parquet epoch data.

    Pools samples from all epoch run directories, remaps per-epoch window IDs to
    global state IDs via epoch_window_map.csv, and builds the union N×K bias
    matrix against the full registry of usable states.

    n_threads: DuckDB threads per connection (0=auto, capped at min(cpu_count,64))
    n_workers: parallel epoch-dir workers; each opens its own DuckDB connection
    """
    try:
        from gareus.query import load_samples  # noqa: F401 – used in _load_epoch_task
    except ImportError as exc:
        raise ImportError(f'gareus package required for Parquet loading: {exc}') from exc
    from concurrent.futures import ThreadPoolExecutor

    adaptive_dir = Path(adaptive_dir)
    registry_csv = adaptive_dir / 'final_registry_used_for_mbar.csv'
    if not registry_csv.exists():
        fallback = adaptive_dir / 'state_registry.csv'
        if fallback.exists():
            registry_csv = fallback
        else:
            raise FileNotFoundError(f'No final_registry_used_for_mbar.csv or state_registry.csv in {adaptive_dir}')

    with registry_csv.open(newline='') as f:
        reg_rows = [r for r in csv.DictReader(f) if _is_usable_for_mbar(r)]

    # `final_registry_used_for_mbar.csv` is a one-time snapshot taken when the
    # run first entered its final phase; if the run was later resumed and the
    # epoch loop added more states before re-entering final phase, this file
    # goes stale relative to the live `state_registry.csv`. Merge in any usable
    # states the snapshot is missing so their samples aren't silently dropped.
    live_registry_csv = adaptive_dir / 'state_registry.csv'
    if registry_csv.name != live_registry_csv.name and live_registry_csv.exists():
        with live_registry_csv.open(newline='') as f:
            live_rows = list(csv.DictReader(f))
        n_before = len(reg_rows)
        reg_rows = _merge_missing_usable_states(reg_rows, live_rows)
        n_added = len(reg_rows) - n_before
        if n_added:
            print(f'    [registry merge] {registry_csv.name} was missing {n_added} usable '
                  f'state(s) present in state_registry.csv (added after the final-phase '
                  f'snapshot was taken); merging them in so their samples are included')

    if not reg_rows:
        raise ValueError(f'No usable states in {registry_csv}')
    reg_rows.sort(key=lambda r: int(r['state_id']))

    state_ids = [int(r['state_id']) for r in reg_rows]
    state_id_to_k = {sid: k for k, sid in enumerate(state_ids)}
    K = len(state_ids)
    # Per-state burnin thresholds (steps within an epoch to discard for equilibration).
    # Currently 0 for all states by default; respected when explicitly set.
    burnin_by_k = np.array([int(r.get('burnin_steps') or 0) for r in reg_rows], dtype=np.int64)
    primary_centers = np.array([float(r['primary_center']) for r in reg_rows])
    primary_ks     = np.array([float(r['primary_k'])      for r in reg_rows])
    sec_centers    = np.array([float(r['secondary_center']) if r.get('secondary_center', '') not in ('', 'None', 'nan') else np.nan for r in reg_rows])
    sec_ks         = np.array([float(r['secondary_k'])      if r.get('secondary_k', '')      not in ('', 'None', 'nan') else 0.0   for r in reg_rows])

    epoch_dirs = _find_adaptive_epoch_dirs(adaptive_dir, epoch_ids=epoch_ids)
    if not epoch_dirs:
        selected = f' for requested epoch(s) {sorted(epoch_ids)}' if epoch_ids is not None else ''
        raise FileNotFoundError(f'No epoch Parquet data found in {adaptive_dir}{selected}')

    all_cv = []; all_cv2 = []; all_window = []; all_step = []
    all_replica = []; all_boost = []; all_boost_dih = []; all_potential = []; all_epoch_src = []
    all_unk_blocks = []
    beta = float('nan')
    meta: dict = rjson(adaptive_dir.parent / 'run_manifest.json', {})

    # Compute per-connection thread budget: distribute n_threads across n_workers.
    _cpu_cap = min(os.cpu_count() or 64, int(os.environ.get('NUMEXPR_MAX_THREADS', 64)))
    _total_threads = n_threads if n_threads > 0 else _cpu_cap
    _n_workers = min(len(epoch_dirs), max(1, n_workers))
    _threads_per_conn = max(1, _total_threads // _n_workers)

    # Load all epochs in parallel (I/O bound); post-process sequentially (order-dependent).
    with ThreadPoolExecutor(max_workers=_n_workers) as _pool:
        epoch_loaded = list(_pool.map(
            lambda _ewt: _load_epoch_task(_ewt[0], _ewt[1], _ewt[2]),
            [(ed, wp, _threads_per_conn) for ed, wp in epoch_dirs],
        ))

    # Resolve beta once, up front, using the same fallback order as before (first
    # epoch whose metadata yields it, else a top-level adaptive_dir inference).
    # Must be fixed *before* any per-epoch bias block is built below, since every
    # block needs the same beta.
    for (samples, _wmap, ep_meta, _native_params), (epoch_dir, _) in zip(epoch_loaded, epoch_dirs):
        if math.isfinite(beta):
            break
        if not samples or 'cv1' not in samples or len(samples['cv1']) == 0:
            continue
        b = float(ep_meta.get('beta_1_over_kJ_mol') or 0.0)
        beta = b if b > 0 else infer_temp_beta(epoch_dir, ep_meta)[1]
    if not math.isfinite(beta):
        _, beta = infer_temp_beta(adaptive_dir, meta)

    for (samples, wmap, ep_meta, native_params), (epoch_dir, _) in zip(epoch_loaded, epoch_dirs):
        if not samples or 'cv1' not in samples or len(samples['cv1']) == 0:
            continue
        raw_w = samples['window_id'].astype(np.int32)
        remapped = np.array([wmap.get(int(w), -1) for w in raw_w], dtype=np.int32)
        valid = remapped >= 0
        if not np.any(valid):
            continue
        cv_epoch = samples['cv1'].astype(np.float64)[valid]
        all_cv.append(cv_epoch)
        cv2_raw = samples.get('cv2')
        cv2_epoch = cv2_raw.astype(np.float64)[valid] if cv2_raw is not None else np.full(valid.sum(), np.nan)
        all_cv2.append(cv2_epoch)
        all_window.append(np.array([state_id_to_k[int(s)] for s in remapped[valid]], dtype=np.int32))
        all_step.append(samples['step'].astype(np.int64)[valid])
        rep = samples['replica'].astype(np.int32) if 'replica' in samples else np.zeros(int(valid.sum()), np.int32)
        all_replica.append(rep[valid])
        boost_raw = samples.get('gamd_boost_total')
        all_boost.append(boost_raw.astype(np.float64)[valid] if boost_raw is not None else np.full(valid.sum(), np.nan))
        boost_dih_raw = samples.get('gamd_boost_dihedral')
        all_boost_dih.append(boost_dih_raw.astype(np.float64)[valid] if boost_dih_raw is not None else np.full(valid.sum(), np.nan))
        pot_raw = samples.get('potential')
        all_potential.append(pot_raw.astype(np.float64)[valid] if pot_raw is not None else np.full(valid.sum(), np.nan))
        all_epoch_src.append(np.full(int(valid.sum()), len(all_cv) - 1, dtype=np.int32))
        # Bias energies for THIS epoch's samples must use the window params that
        # were actually in effect during this epoch (native_params), not whatever
        # a state's row in the live/final registry says today — that snapshot can
        # be stale for any state recentered by a later epoch (e.g. the tICA CV2
        # auto-switch overwrites secondary_center in place; see CLAUDE.md).
        pc_e, pk_e, sc_e, sk_e = _epoch_bias_param_vectors(
            native_params, state_ids, primary_centers, primary_ks, sec_centers, sec_ks)
        all_unk_blocks.append(_reconstruct_union_bias_block(cv_epoch, cv2_epoch, beta, pc_e, pk_e, sc_e, sk_e))

    if not all_cv:
        raise ValueError(f'No valid samples after window remapping in {adaptive_dir}')

    cv      = np.concatenate(all_cv)
    cv2     = np.concatenate(all_cv2)
    window  = np.concatenate(all_window)
    step    = np.concatenate(all_step)
    replica = np.concatenate(all_replica)
    boost     = np.concatenate(all_boost)
    boost_dih = np.concatenate(all_boost_dih)
    pot_arr = np.concatenate(all_potential)
    potential = pot_arr if np.any(np.isfinite(pot_arr)) else None
    u_nk    = np.concatenate(all_unk_blocks, axis=0)

    epoch_src = np.concatenate(all_epoch_src) if all_epoch_src else np.zeros(len(cv), dtype=np.int32)

    # Drop per-state burnin frames: for each sample, compare its epoch-local step
    # against the burnin threshold for the state it was collected in.
    if np.any(burnin_by_k > 0):
        keep = step >= burnin_by_k[window]
        n_dropped = int((~keep).sum())
        if n_dropped > 0:
            print(f'    [burnin filter] dropped {n_dropped}/{len(cv)} samples ({100*n_dropped/len(cv):.1f}%) from pre-equilibration steps')
        cv = cv[keep]; cv2 = cv2[keep]; window = window[keep]
        step = step[keep]; replica = replica[keep]; boost = boost[keep]
        epoch_src = epoch_src[keep]
        pot_arr = pot_arr[keep]
        potential = pot_arr if np.any(np.isfinite(pot_arr)) else None
        u_nk = u_nk[keep]

    temp = 1.0 / (K_B_KJ_PER_MOL_K * beta)

    meta_out = dict(meta)
    meta_out.update({'temperature_K': temp, 'beta_1_over_kJ_mol': beta,
                     'adaptive_union_states': K, 'adaptive_union_epochs': len(epoch_dirs),
                     'umbrella_window_rows': list(reg_rows),
                     '_epoch_source': epoch_src.tolist(),
                     'adaptive_epoch_run_dirs': [str(ed) for ed, _ in epoch_dirs]})

    _boost_dih_arg = boost_dih if np.any(np.isfinite(boost_dih)) else None
    return clean(Data(
        prod_dir=adaptive_dir, out_dir=adaptive_dir / 'pmf_analysis',
        cv=cv, cv2=cv2, rg_A=np.full(cv.shape, np.nan),
        window=window, replica=replica, step=step,
        u_nk=u_nk, centers=primary_centers, k_kcal=primary_ks,
        beta=beta, temp=temp, boost_kj=boost, potential_kj=potential,
        source=str(registry_csv), meta=meta_out,
        boost_dih_kj=_boost_dih_arg,
    ))


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

    Rebuilds from scratch each call (removes any previous merged dir first)
    so a stale run from before this fix -- or from a run whose phase list
    changed -- can never leave incorrect leftover links in place.
    """
    epoch_traj = _get_adaptive_epoch_traj_dirs(d)
    if not epoch_traj:
        return None
    merged = d.prod_dir / '_merged_replica_trajectories'
    if merged.exists():
        shutil.rmtree(merged)
    merged.mkdir(exist_ok=True)
    STEP_STRIDE = _MERGED_TRAJ_STEP_STRIDE
    TRAJ_EXTS = {'.xtc', '.dcd', '.nc', '.trr'}
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
            link = (merged / f'{base_stem}{f.suffix}' if global_resume == 0
                    else merged / f'{base_stem}_resume_from_{global_resume}{f.suffix}')
            if not link.exists():
                try:
                    link.symlink_to(f.resolve())
                except Exception:
                    pass
    return merged if any(merged.iterdir()) else None


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
    window   = samples['window_id'].astype(np.int32)
    step     = samples['step'].astype(np.int64)
    replica  = samples['replica'].astype(np.int32) if 'replica' in samples else np.zeros(cv.shape, dtype=np.int32)
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


def clean(d: Data) -> Data:
    if d.u_nk.shape[0] != d.cv.size: raise ValueError('u_nk/sample count mismatch')
    if d.boost_kj.shape != d.cv.shape: d.boost_kj=np.full(d.cv.shape,np.nan)
    if d.cv2.shape != d.cv.shape: d.cv2=np.full(d.cv.shape,np.nan)
    if d.rg_A.shape != d.cv.shape: d.rg_A=np.full(d.cv.shape,np.nan)
    mask=np.isfinite(d.cv) & np.all(np.isfinite(d.u_nk),axis=1)
    d.cv=d.cv[mask]; d.cv2=d.cv2[mask]; d.rg_A=d.rg_A[mask]; d.window=d.window[mask]; d.replica=d.replica[mask]; d.step=d.step[mask]; d.u_nk=d.u_nk[mask]; d.boost_kj=d.boost_kj[mask]
    if d.potential_kj is not None and d.potential_kj.size==mask.size: d.potential_kj=d.potential_kj[mask]
    if d.boost_dih_kj is not None and d.boost_dih_kj.size==mask.size: d.boost_dih_kj=d.boost_dih_kj[mask]
    elif d.boost_dih_kj is not None: d.boost_dih_kj=None
    return d

def _filter_epoch_source(d: Data, keep: np.ndarray) -> None:
    """Filter d.meta['_epoch_source'] in-place to match the keep mask."""
    src = d.meta.get('_epoch_source')
    if src is not None and len(src) == keep.size:
        arr = np.asarray(src, dtype=np.int64)
        d.meta['_epoch_source'] = arr[keep].tolist()


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
    downstream MBAR/dTRAM/FES/convergence calculation sees the same subsampled
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

def ess(w):
    w=np.asarray(w,float); w=w[np.isfinite(w)&(w>=0)]
    if w.size==0: return 0.0
    s1=float(np.sum(w)); s2=float(np.sum(w*w)); return 0.0 if s2<=0 else s1*s1/s2

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


def solve_mbar_numba(u_nk, window, tol=1e-10, maxiter=10000, progress: Optional[Progress] = None, threads: int = 0, f_init: Optional[np.ndarray] = None):
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
    u=np.ascontiguousarray(u_nk[:,active],dtype=np.float64)
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
                              progress: Optional[Progress] = None, threads: int = 0,
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
    # Restrict bias energies and counts to active states
    u = np.ascontiguousarray(u_nk[:, active], dtype=np.float64)
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
                                progress: Optional[Progress] = None,
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
    u = np.ascontiguousarray(u_nk[:, active], dtype=np.float64)
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
                      progress: Optional[Progress] = None, threads: int = 0,
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

def solve_mbar_lbfgs(u_nk, window, tol=1e-10, maxiter=10000, progress: Optional[Progress] = None, f_init: Optional[np.ndarray] = None):
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
    u = np.ascontiguousarray(u_nk[:, active], dtype=np.float64)
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
    conv = result.success or (result.status in (0, 1))
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


def solve_mbar(u_nk, window, tol=1e-10, maxiter=10000, progress: Optional[Progress] = None, backend: str = 'auto', threads: int = 0, f_init: Optional[np.ndarray] = None,
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
    u=np.ascontiguousarray(u_nk[:,active],dtype=np.float64)
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

def make_bins(cv,bins,lo,hi):
    lo=float(np.nanmin(cv) if lo is None else lo); hi=float(np.nanmax(cv) if hi is None else hi)
    pad=0.02*max(1.0,hi-lo)
    if lo==hi: hi=lo+1.0
    return np.linspace(lo-pad if lo is None else lo, hi+pad if hi is None else hi, bins+1)

def pmf_from_weights(cv,w,bins,kbt_kcal):
    prob,edges=np.histogram(cv,bins=bins,weights=w); counts,_=np.histogram(cv,bins=bins); prob=np.asarray(prob,float)
    if np.sum(prob)>0: prob/=np.sum(prob)
    with np.errstate(divide='ignore',invalid='ignore'): F=-kbt_kcal*np.log(prob)
    mask=np.isfinite(F)
    if np.any(mask): F-=np.nanmin(F[mask])
    return {'cv_A':0.5*(edges[:-1]+edges[1:]),'prob':prob,'pmf':F,'counts':counts.astype(int)}

def _cumulant_expansion(cv,base_w,boost,bins,beta,kbt_kcal,order=2,smooth_logfac_sigma=0.0):
    """Cumulant GaMD reweighting, vectorized by CV bin.

    order=2 keeps the mean+variance terms (Gaussian/CE2 approximation).
    order=3 adds the beta^3/6 * kappa3 term, where kappa3 is the per-bin
    third cumulant (= third central moment) of the boost. kappa3/var are
    computed via a two-pass mean-centered accumulation rather than raw
    moments, since raw <x^3>-3<x^2><x>+2<x>^3 catastrophically cancels
    when the boost mean (O(10-200) kJ/mol) dominates its spread.
    """
    if order not in (2,3):
        raise ValueError(f"cumulant expansion order must be 2 or 3, got {order}")
    cv=np.asarray(cv,dtype=np.float64)
    base_w=np.asarray(base_w,dtype=np.float64)
    boost=np.asarray(boost,dtype=np.float64)
    p0,edges=np.histogram(cv,bins=bins,weights=base_w)
    counts,_=np.histogram(cv,bins=bins)
    centers=0.5*(edges[:-1]+edges[1:])
    B=centers.size
    bi=np.searchsorted(edges,cv,side='right')-1
    bi[cv==edges[-1]]=B-1
    good=(bi>=0)&(bi<B)&np.isfinite(boost)&np.isfinite(base_w)&(base_w>0)
    mean=np.full(B,np.nan,dtype=np.float64)
    var=np.full(B,np.nan,dtype=np.float64)
    kappa3=np.full(B,np.nan,dtype=np.float64)
    logfac=np.zeros(B,dtype=np.float64)
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
        logfac[nz]=beta*mean[nz]+0.5*beta*beta*var[nz]
        if order==3:
            sdx3=np.bincount(idx,weights=w*dx*dx*dx,minlength=B).astype(np.float64)
            kappa3[nz]=sdx3[nz]/sw[nz]
            logfac[nz]+=(beta**3/6.0)*kappa3[nz]
    if smooth_logfac_sigma and float(smooth_logfac_sigma) > 0:
        try:
            from scipy.ndimage import gaussian_filter1d
            logfac=gaussian_filter1d(logfac,sigma=float(smooth_logfac_sigma),mode='nearest')
        except Exception:
            pass
    p=p0*np.exp(np.clip(logfac,-700,700))
    ps=float(np.sum(p))
    if ps>0: p/=ps
    with np.errstate(divide='ignore',invalid='ignore'):
        F=-kbt_kcal*np.log(p)
    mask=np.isfinite(F)
    if np.any(mask): F-=np.nanmin(F[mask])
    return {'cv_A':centers,'prob':p,'pmf':F,'counts':counts.astype(int)}, {'boost_mean_kj':mean,'boost_var_kj2':var,'boost_kappa3_kj3':kappa3,'log_reweight_factor':logfac}


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

def _cumulant_expansion_2d(x,y,base_w,boost,xbins,ybins,beta,kbt_kcal,order=2,smooth_logfac_sigma=0.0):
    """2D counterpart of _cumulant_expansion; see that docstring for order/kappa3 notes."""
    if order not in (2,3):
        raise ValueError(f"cumulant expansion order must be 2 or 3, got {order}")
    x=np.asarray(x,dtype=np.float64)
    y=np.asarray(y,dtype=np.float64)
    base_w=np.asarray(base_w,dtype=np.float64)
    boost=np.asarray(boost,dtype=np.float64)
    p0,xedges,yedges=np.histogram2d(x,y,bins=[xbins,ybins],weights=base_w)
    counts,_,_=np.histogram2d(x,y,bins=[xbins,ybins])
    xc=0.5*(xedges[:-1]+xedges[1:])
    yc=0.5*(yedges[:-1]+yedges[1:])
    Bx=len(xc); By=len(yc)
    xi=np.searchsorted(xedges,x,side='right')-1
    yi=np.searchsorted(yedges,y,side='right')-1
    xi[x==xedges[-1]]=Bx-1
    yi[y==yedges[-1]]=By-1
    good=(xi>=0)&(xi<Bx)&(yi>=0)&(yi<By)&np.isfinite(boost)&np.isfinite(base_w)&(base_w>0)
    mean=np.full((Bx,By),np.nan,dtype=np.float64)
    var=np.full((Bx,By),np.nan,dtype=np.float64)
    kappa3=np.full((Bx,By),np.nan,dtype=np.float64)
    logfac=np.zeros((Bx,By),dtype=np.float64)
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
        logfac[nz]=beta*mean[nz]+0.5*beta*beta*var[nz]
        if order==3:
            sdb3=np.bincount(idx,weights=w*db*db*db,minlength=Bx*By).astype(np.float64).reshape(Bx,By)
            kappa3[nz]=sdb3[nz]/sw[nz]
            logfac[nz]+=(beta**3/6.0)*kappa3[nz]
    if smooth_logfac_sigma and float(smooth_logfac_sigma) > 0:
        try:
            from scipy.ndimage import gaussian_filter
            logfac=gaussian_filter(logfac,sigma=float(smooth_logfac_sigma),mode='nearest')
        except Exception:
            pass
    p=p0*np.exp(np.clip(logfac,-700,700))
    ps=float(np.sum(p))
    if ps>0:
        p/=ps
    with np.errstate(divide='ignore', invalid='ignore'):
        F=-kbt_kcal*np.log(p)
    mask=np.isfinite(F)
    if np.any(mask):
        F-=np.nanmin(F[mask])
    return {
        'cv_A':xc,
        'rg_A':yc,
        'cv_edges_A':np.asarray(xedges,dtype=np.float64),
        'rg_edges_A':np.asarray(yedges,dtype=np.float64),
        'prob':p,
        'pmf':F,
        'counts':counts.astype(int),
    }, {
        'boost_mean_kj':mean,
        'boost_var_kj2':var,
        'boost_kappa3_kj3':kappa3,
        'log_reweight_factor':logfac,
    }


def cumulant2_2d(x,y,base_w,boost,xbins,ybins,beta,kbt_kcal,smooth_logfac_sigma=0.0):
    return _cumulant_expansion_2d(x,y,base_w,boost,xbins,ybins,beta,kbt_kcal,order=2,smooth_logfac_sigma=smooth_logfac_sigma)


def cumulant3_2d(x,y,base_w,boost,xbins,ybins,beta,kbt_kcal,smooth_logfac_sigma=0.0):
    return _cumulant_expansion_2d(x,y,base_w,boost,xbins,ybins,beta,kbt_kcal,order=3,smooth_logfac_sigma=smooth_logfac_sigma)

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

def _smooth_masked_grid(grid, sigma=1.0):
    grid=np.asarray(grid,dtype=np.float64)
    finite=np.isfinite(grid)
    if not np.any(finite):
        return grid
    if sigma is None or float(sigma) <= 0:
        return grid.copy()
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        return grid.copy()
    vals=np.where(finite, grid, 0.0)
    mask=finite.astype(np.float64)
    sval=gaussian_filter(vals*mask, sigma=float(sigma), mode='nearest')
    smask=gaussian_filter(mask, sigma=float(sigma), mode='nearest')
    out=np.full_like(grid, np.nan, dtype=np.float64)
    good=smask>1e-8
    out[good]=sval[good]/smask[good]
    return out

def _smooth_pmf_1d(pmf, sigma):
    """Gaussian smooth a 1D PMF array for plotting; NaN values preserved."""
    if sigma is None or float(sigma) <= 0:
        return np.asarray(pmf, dtype=float)
    try:
        from scipy.ndimage import gaussian_filter1d
    except Exception:
        return np.asarray(pmf, dtype=float)
    pmf = np.asarray(pmf, dtype=float)
    finite = np.isfinite(pmf)
    if not np.any(finite):
        return pmf.copy()
    filled = pmf.copy()
    idx = np.where(finite)[0]
    for i in np.where(~finite)[0]:
        filled[i] = pmf[idx[np.argmin(np.abs(idx - i))]]
    smoothed = gaussian_filter1d(filled, sigma=float(sigma), mode='nearest')
    out = pmf.copy()
    out[finite] = smoothed[finite]
    return out

def _eff_smooth(args, attr: str) -> float:
    """Return effective smooth sigma: --smooth-sigma master overrides specific attr."""
    master = float(getattr(args, 'smooth_sigma', 0.0) or 0.0)
    if master > 0:
        return master
    return float(getattr(args, attr, 0.0) or 0.0)

FES_PLOT_VMAX_VALUES = (2.0, 5.0, 10.0, 20.0, None)

def _fes_range_tag(vmax) -> str:
    if vmax is None:
        return '0_all'
    v=float(vmax)
    if abs(v-round(v)) < 1.0e-9:
        return f'0_{int(round(v))}'
    return '0_' + str(v).replace('.', 'p')

def _fes_range_label(vmax, actual_max: float) -> str:
    if vmax is None:
        if math.isfinite(float(actual_max)):
            return f'0-all kcal/mol (max {float(actual_max):.2f})'
        return '0-all kcal/mol'
    return f'0-{float(vmax):g} kcal/mol'

def _fes_variant_path(base_path, vmax) -> Path:
    base=Path(base_path)
    suffix=base.suffix or '.png'
    return base.with_name(f'{base.stem}_{_fes_range_tag(vmax)}{suffix}')

def _plot_2d_fes_range(
    F, xedges, yedges, xc, yc, out_png, title, xlabel, ylabel, warnings,
    *, smooth_sigma=1.0, range_vmax=None, figsize=(8.8,6.6), dpi=220, cmap_name='viridis', contour=True,
    y_annotation_lines=None
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        warnings.append(f'matplotlib unavailable; no 2D FES plot written: {e}')
        return False
    F=np.asarray(F,dtype=np.float64)
    xedges=np.asarray(xedges,dtype=np.float64)
    yedges=np.asarray(yedges,dtype=np.float64)
    xc=np.asarray(xc,dtype=np.float64)
    yc=np.asarray(yc,dtype=np.float64)
    Fplot=_smooth_masked_grid(F, sigma=smooth_sigma)
    finite=np.isfinite(Fplot)
    if not np.any(finite):
        warnings.append(f'{title} has no finite bins; skipped 2D FES heatmap plot.')
        return False
    actual_max=float(np.nanmax(Fplot[finite]))
    if not np.isfinite(actual_max) or actual_max <= 0:
        actual_max=1.0
    if range_vmax is None:
        vmax=max(1.0, actual_max)
        extend='neither'
    else:
        vmax=max(1.0e-12, float(range_vmax))
        extend='max' if actual_max > vmax else 'neither'
    fig,ax=plt.subplots(figsize=figsize)
    cmap=plt.get_cmap(cmap_name).copy() if hasattr(plt.get_cmap(cmap_name),'copy') else plt.get_cmap(cmap_name)
    try:
        cmap.set_bad(color='white', alpha=0.0)
    except Exception:
        pass
    im=ax.imshow(Fplot.T, origin='lower', extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]], aspect='auto', interpolation='bicubic', cmap=cmap, vmin=0.0, vmax=vmax)
    if contour:
        try:
            X,Y=np.meshgrid(xc,yc,indexing='ij')
            contour_source=np.where(np.isfinite(Fplot),Fplot,np.nan)
            levels=np.linspace(0.0, vmax, 10)
            if np.count_nonzero(np.isfinite(contour_source)) >= 9 and len(levels) > 2:
                cs=ax.contour(X,Y,contour_source,levels=levels[1:],colors='white',linewidths=0.7,alpha=0.75)
                ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
        except Exception:
            pass
    finite_raw=np.isfinite(F)
    if np.any(finite_raw):
        min_idx=np.unravel_index(np.nanargmin(np.where(finite_raw,F,np.inf)),F.shape)
        ax.plot([xc[min_idx[0]]],[yc[min_idx[1]]],marker='*',markersize=11,markeredgecolor='black',markerfacecolor='gold',zorder=5)
    if y_annotation_lines:
        for ann in y_annotation_lines:
            v = float(ann.get('value', float('nan')))
            lbl = str(ann.get('label', ''))
            if not (np.isfinite(v) and yedges[0] <= v <= yedges[-1]):
                continue
            ax.axhline(v, color='white', linewidth=0.9, linestyle='--', alpha=0.7)
            ax.text(xedges[0], v, f' {lbl}', color='white', fontsize=7, va='bottom', ha='left',
                    bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.35, lw=0))
    cbar=fig.colorbar(im,ax=ax,extend=extend)
    cbar.set_label('Free energy (kcal/mol, minimum shifted to 0)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title} [{_fes_range_label(range_vmax, actual_max)}]')
    ax.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(out_png,dpi=dpi)
    plt.close(fig)
    return True

def _plot_2d_fes_multirange(
    F, xedges, yedges, xc, yc, out_png, title, xlabel, ylabel, warnings,
    *, smooth_sigma=1.0, figsize=(8.8,6.6), dpi=220, cmap_name='viridis', contour=True,
    y_annotation_lines=None
) -> dict[str,str]:
    files={}
    first_vmax=FES_PLOT_VMAX_VALUES[0]
    first_ok=False
    for vmax in FES_PLOT_VMAX_VALUES:
        tag=_fes_range_tag(vmax)
        path=_fes_variant_path(out_png, vmax)
        ok=_plot_2d_fes_range(F,xedges,yedges,xc,yc,path,title,xlabel,ylabel,warnings,smooth_sigma=smooth_sigma,range_vmax=vmax,figsize=figsize,dpi=dpi,cmap_name=cmap_name,contour=contour,y_annotation_lines=y_annotation_lines)
        if ok:
            files[tag]=str(path)
            if vmax == first_vmax:
                first_ok=True
    if first_ok:
        _plot_2d_fes_range(F,xedges,yedges,xc,yc,out_png,title,xlabel,ylabel,warnings,smooth_sigma=smooth_sigma,range_vmax=first_vmax,figsize=figsize,dpi=dpi,cmap_name=cmap_name,contour=contour,y_annotation_lines=y_annotation_lines)
        files['main']=str(out_png)
    return files

def plot_2d_fes(fes, method, out_png, title, warnings, smooth_sigma=1.0):
    F=np.asarray(fes['pmf'],dtype=np.float64)
    xedges=np.asarray(fes['cv_edges_A'],dtype=np.float64)
    yedges=np.asarray(fes['rg_edges_A'],dtype=np.float64)
    xc=np.asarray(fes['cv_A'],dtype=np.float64)
    yc=np.asarray(fes['rg_A'],dtype=np.float64)
    return _plot_2d_fes_multirange(F,xedges,yedges,xc,yc,out_png,title,'CV distance (A)','Rg (A)',warnings,smooth_sigma=smooth_sigma,figsize=(8.8,6.6),dpi=220,cmap_name='viridis',contour=True)

def plot_cv1_cv2_2d_fes(fes, method, out_png, title, warnings, smooth_sigma=1.0, cv2_label='Secondary CV', cv1_label='CV distance (A)', regions=None):
    F=np.asarray(fes['pmf'],dtype=np.float64)
    xedges=np.asarray(fes['cv_edges_A'],dtype=np.float64)
    yedges=np.asarray(fes['rg_edges_A'],dtype=np.float64)
    xc=np.asarray(fes['cv_A'],dtype=np.float64)
    yc=np.asarray(fes['rg_A'],dtype=np.float64)
    return _plot_2d_fes_multirange(F,xedges,yedges,xc,yc,out_png,title,cv1_label,cv2_label,warnings,smooth_sigma=smooth_sigma,figsize=(8.8,6.6),dpi=220,cmap_name='viridis',contour=True,y_annotation_lines=regions)

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
    steps=checkpoint_steps_from_data(d.step[finite_global],int(getattr(args,'convergence_timepoints',10)))
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
                if rg_atoms is not None:
                    traj=md.load(str(seg_path), top=top_topology, atom_indices=rg_atoms)
                    rg=md.compute_rg(traj)*10.0
                else:
                    traj=md.load(str(seg_path), top=top_topology)
                    _atoms=traj.topology.select(selection)
                    if _atoms.size == 0:
                        raise ValueError(f'selection {selection!r} matched zero atoms')
                    rg=md.compute_rg(traj.atom_slice(_atoms))*10.0
            except Exception as exc:
                local_warns.append(f'Rg trajectory reconstruction failed for replica {rep} ({seg_path}): {exc}')
                continue
            eff_resume_start=_base_segment_resume_start(resume_start, use_adjusted, steps_for_align, spf)
            mask, local_frames=_sample_to_segment_frame(steps_for_align, eff_resume_start, traj.n_frames, spf)
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

def plot_rg_outputs(d: Data, rg_pmfs: dict, selected: str, out: Path, warnings: list[str], smooth_sigma: float = 0.0, args=None):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        warnings.append(f'matplotlib unavailable; no Rg PNG plots written: {e}')
        return
    import gareus_plotstyle as ps
    fig,ax=plt.subplots(figsize=(8,5))
    for i,(name,p) in enumerate(_visible_pmfs(rg_pmfs, selected, args).items()):
        pmf_plot=_smooth_pmf_1d(p['pmf'],smooth_sigma); m=np.isfinite(pmf_plot)
        if np.any(m): ps.plot_method_curve(ax,p['cv_A'][m],pmf_plot[m],name,selected,idx=i)
    ps.style_line_axes(ax,xlabel='Rg (Å)',ylabel='PMF (kcal/mol, shifted)',title='Radius of gyration PMF estimates'); fig.tight_layout(); fig.savefig(out/'rg_pmf_all_methods.png',dpi=200); plt.close(fig)
    rg=np.asarray(d.rg_A,dtype=float); mask=np.isfinite(rg)&np.isfinite(d.cv)
    if np.count_nonzero(mask)>5:
        fig,ax=plt.subplots(figsize=(6,5))
        hb=ax.hexbin(d.cv[mask],rg[mask],gridsize=45,mincnt=1)
        ax.set_xlabel(_primary_cv_axis_label(d.meta)); ax.set_ylabel('Rg (A)'); ax.set_title('Sampled CV-Rg coverage')
        fig.colorbar(hb,ax=ax,label='sample count'); fig.tight_layout(); fig.savefig(out/'rg_vs_cv_sampled.png',dpi=200); plt.close(fig)

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
        cum_pmf,cdiag=cumulant2(rg_sel,base_w_rg,boost_sel,bins,d.beta,kbt_kcal,smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
        cum3_pmf,cdiag3=cumulant3(rg_sel,base_w_rg,boost_sel,bins,d.beta,kbt_kcal,smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
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
        fes_cum, cdiag = cumulant2_2d(cv_sel, rg_sel, base_w, boost_sel, xbins, ybins, d.beta, kbt_kcal, smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
        fes_cum3, cdiag3 = cumulant3_2d(cv_sel, rg_sel, base_w, boost_sel, xbins, ybins, d.beta, kbt_kcal, smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
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
    dtram2d_info=run_dtram_2d_fes(d,args,d.cv,rg,np.asarray(base_logw,dtype=np.float64),xbins,ybins,kbt_kcal,chosen_fes,chosen,'distance_rg_2d_fes','Distance vs Rg 2D free-energy surface',_primary_cv_axis_label(d.meta),'Rg (A)',out,warnings,progress,mbar_result=None)
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
    if isinstance(dtram2d_info,dict) and dtram2d_info.get('available'):
        info['dtram']=_dtram_public_info(dtram2d_info)
        if isinstance(dtram2d_info.get('files'),dict):
            info['files'].update({k:v for k,v in dtram2d_info.get('files',{}).items()})
    elif bool(getattr(args,'dtram_2d_fes',False)):
        info['dtram']=_dtram_public_info(dtram2d_info)
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

def plot_pca_2d_fes(fes: dict, method: str, out_png: Path, title: str, warnings: list[str], smooth_sigma: float = 1.0) -> dict[str,str]:
    F=np.asarray(fes['pmf'],dtype=np.float64)
    xedges=np.asarray(fes['cv_edges_A'],dtype=np.float64)
    yedges=np.asarray(fes['rg_edges_A'],dtype=np.float64)
    xc=np.asarray(fes['cv_A'],dtype=np.float64)
    yc=np.asarray(fes['rg_A'],dtype=np.float64)
    return _plot_2d_fes_multirange(F,xedges,yedges,xc,yc,out_png,title,'PCA1 (A)','PCA2 (A)',warnings,smooth_sigma=smooth_sigma,figsize=(8.8,6.6),dpi=220,cmap_name='viridis',contour=True)

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
        fes_cum,_cdiag=cumulant2_2d(x,y,base_w,boost_sel,xbins,ybins,d.beta,kbt_kcal,smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
        fes_cum3,_cdiag3=cumulant3_2d(x,y,base_w,boost_sel,xbins,ybins,d.beta,kbt_kcal,smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
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
    dtram2d_info=run_dtram_2d_fes(d,args,pca1,pca2,np.asarray(base_logw,dtype=np.float64),xbins,ybins,kbt_kcal,chosen_fes,chosen,'pca_2d_fes','PCA1 vs PCA2 2D free-energy surface','PCA1 (A)','PCA2 (A)',out,warnings,progress,mbar_result=None)
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
    if isinstance(dtram2d_info,dict) and dtram2d_info.get('available'):
        info['dtram']=_dtram_public_info(dtram2d_info)
        if isinstance(dtram2d_info.get('files'),dict):
            info['files'].update({k:v for k,v in dtram2d_info.get('files',{}).items()})
    elif bool(getattr(args,'dtram_2d_fes',False)):
        info['dtram']=_dtram_public_info(dtram2d_info)
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

_OPT_IN_GAMD_METHODS = {'gamd_exponential': 'plot_gamd_exponential', 'gamd_cumulant3': 'plot_gamd_cumulant3'}

def _want_gamd_method(method: str, chosen: str, args) -> bool:
    """gamd_exponential/gamd_cumulant3 are opt-in (--plot-gamd-exponential/--plot-gamd-cumulant3);
    umbrella_only and gamd_cumulant2 are always shown. The currently chosen/selected method is
    always shown even if it is one of the opt-in ones (explicit --selected-method forces it)."""
    flag = _OPT_IN_GAMD_METHODS.get(method)
    return flag is None or bool(getattr(args, flag, False)) or method == chosen

def _visible_pmfs(pmfs: dict, chosen: str, args) -> dict:
    """Filter a {method: pmf} dict down to the methods that should be drawn in a multi-method
    comparison plot, per _want_gamd_method."""
    return {name: p for name, p in pmfs.items() if _want_gamd_method(name, chosen, args)}

def _choose_method(selected: str, boost_ok: bool) -> str:
    if boost_ok and selected in {'gamd_exponential','gamd_cumulant2','gamd_cumulant3'}:
        return selected
    if boost_ok and selected not in {'umbrella_only','gamd_exponential','gamd_cumulant2','gamd_cumulant3'}:
        return 'gamd_cumulant2'
    return 'umbrella_only'

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
        import gareus_plotstyle as ps
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

def _peptide_solvent_contact_counts(md, traj, cutoff_nm: float) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    top=traj.topology
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
    n=traj.n_frames
    zeros=np.zeros(n,dtype=np.float64)
    if len(peptide_heavy)==0 or len(solvent_heavy)==0:
        return zeros.copy(), zeros.copy(), zeros.copy()
    try:
        neigh=md.compute_neighbors(traj,float(cutoff_nm),query_indices=np.asarray(peptide_heavy,dtype=np.int32),haystack_indices=np.asarray(solvent_heavy,dtype=np.int32),periodic=True)
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

def _dssp_fraction_arrays(md, traj):
    try:
        ss=md.compute_dssp(traj,simplified=True)
    except Exception:
        return None
    arr=np.asarray(ss)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return None
    helix=np.mean(arr=='H',axis=1).astype(np.float64)
    strand=np.mean(arr=='E',axis=1).astype(np.float64)
    coil=np.mean((arr!='H')&(arr!='E'),axis=1).astype(np.float64)
    return arr,helix,strand,coil

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
        progress.step('extra PMFs', f'topology {top_path}; chunk={chunk_size}; SASA selection={sasa_selection!r}')

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
    dssp_counts=None; dssp_total=0
    if dssp_labels:
        dssp_counts={code:np.zeros(len(dssp_labels),dtype=np.int64) for code in ('H','E','C')}

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
    for pi,item in enumerate(plan, start=1):
        if progress is not None: progress.bar('extra PMF pass',pi,max(1,len(plan)),f"replica {item['replica']}")
        frame0=0
        frame_indices=np.asarray(item.get('frame_indices', np.arange(item['n_assign'])),dtype=np.int64)
        sample_indices=np.asarray(item.get('sample_indices', item['order']),dtype=np.int64)
        try:
            for chunk in md.iterload(str(item.get('traj', item['dcd'])),top=str(top_path),chunk=chunk_size):
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
                        dssp_counts['H']+=np.sum(ss=='H',axis=0).astype(np.int64)
                        dssp_counts['E']+=np.sum(ss=='E',axis=0).astype(np.int64)
                        dssp_counts['C']+=np.sum((ss!='H')&(ss!='E'),axis=0).astype(np.int64)
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
                    cbuf['solv'],cbuf['wat'],cbuf['ion']=_peptide_solvent_contact_counts(md,full_use,contact_cutoff)
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
        rows=[]
        for i,lab in enumerate(dssp_labels):
            rows.append({'residue':lab,'n_frames':int(dssp_total),'helix_probability':float(dssp_counts['H'][i]/dssp_total),'strand_probability':float(dssp_counts['E'][i]/dssp_total),'coil_other_probability':float(dssp_counts['C'][i]/dssp_total)})
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


# -----------------------------------------------------------------------------
# Discrete TRAM / dTRAM estimator
#
# This is a self-contained discrete TRAM maximum-likelihood implementation for
# CV-bin microstates.  It uses transition count matrices C[k, i, j] and the
# thermodynamic bias b[i, k] to estimate an unbiased discrete stationary
# distribution over microstates.  For each thermodynamic state k, the biased
# stationary distribution is constrained as
#
#     mu_k(i) proportional to pi(i) * exp(-b[i, k])
#
# and the transition matrix is profiled out under detailed balance with respect
# to mu_k.  The outer optimizer varies pi(i).  This is intentionally separate
# from the MBAR/GaMD path so existing outputs remain unchanged unless --dtram is
# requested.

def _dtram_logsumexp_vec(x: np.ndarray) -> float:
    x=np.asarray(x,dtype=np.float64)
    if x.size == 0:
        return float('-inf')
    m=float(np.max(x))
    if not np.isfinite(m):
        return m
    return float(m+np.log(np.sum(np.exp(x-m))))


def _dtram_softmax(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,dtype=np.float64)
    lse=_dtram_logsumexp_vec(x)
    if not np.isfinite(lse):
        return np.full(x.shape,1.0/max(1,x.size),dtype=np.float64)
    return np.exp(x-lse)


def _dtram_discrete_states_1d(x: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return integer microstate labels for 1D bins and a finite/in-range mask."""
    x=np.asarray(x,dtype=np.float64)
    edges=np.asarray(edges,dtype=np.float64)
    states=np.full(x.shape,-1,dtype=np.int64)
    if edges.size < 2:
        return states, np.zeros(x.shape,dtype=bool)
    idx=np.searchsorted(edges,x,side='right')-1
    idx[x==edges[-1]]=edges.size-2
    good=np.isfinite(x)&(idx>=0)&(idx<edges.size-1)
    states[good]=idx[good].astype(np.int64,copy=False)
    return states, good


def _dtram_build_transition_counts(d: Data, states: np.ndarray, n_states: int, lag: int = 1,
                                   mode: str = 'same-window') -> tuple[np.ndarray, dict]:
    """Build per-window transition count matrices from ordered replica traces.

    Parameters
    ----------
    d : Data
        Loaded analysis data.  The current implementation uses ``replica`` as
        the continuous trajectory identity because that is the only portable
        identity exposed by the analyzer cache.
    states : ndarray, shape (N,)
        Discrete microstate labels, or -1 for invalid/out-of-range samples.
    n_states : int
        Number of discrete microstates.
    lag : int
        Lag in saved analysis samples, not raw MD steps.
    mode : {'same-window', 'start-window'}
        ``same-window`` counts only pairs whose start and end samples have the
        same umbrella/window label.  ``start-window`` assigns the transition to
        the starting window.  For replica-exchange trajectories, same-window is
        the conservative default.
    """
    lag=max(1,int(lag or 1))
    K=int(d.u_nk.shape[1])
    C=np.zeros((K,int(n_states),int(n_states)),dtype=np.float64)
    n_pairs=0; n_used=0; n_bad_state=0; n_window_skipped=0
    states=np.asarray(states,dtype=np.int64)
    mode=str(mode or 'same-window').lower()
    reps=np.unique(d.replica[np.isfinite(d.replica)])
    for rep in reps:
        idx=np.where(d.replica==rep)[0]
        if idx.size <= lag:
            continue
        idx=idx[np.argsort(d.step[idx],kind='stable')]
        a=idx[:-lag]; b=idx[lag:]
        n_pairs += int(a.size)
        si=states[a]; sj=states[b]
        good=(si>=0)&(si<n_states)&(sj>=0)&(sj<n_states)
        n_bad_state += int(np.count_nonzero(~good))
        if not np.any(good):
            continue
        a=a[good]; b=b[good]; si=si[good]; sj=sj[good]
        wi=np.asarray(d.window[a],dtype=np.int64)
        wj=np.asarray(d.window[b],dtype=np.int64)
        if mode == 'same-window':
            wgood=(wi==wj)&(wi>=0)&(wi<K)
            n_window_skipped += int(np.count_nonzero(~wgood))
            wi=wi[wgood]; si=si[wgood]; sj=sj[wgood]
        elif mode == 'start-window':
            wgood=(wi>=0)&(wi<K)
            n_window_skipped += int(np.count_nonzero(~wgood))
            wi=wi[wgood]; si=si[wgood]; sj=sj[wgood]
        else:
            raise ValueError(f'Unknown dTRAM transition mode {mode!r}')
        if wi.size == 0:
            continue
        lin=wi*(n_states*n_states)+si*n_states+sj
        C += np.bincount(lin,minlength=K*n_states*n_states).reshape(K,n_states,n_states)
        n_used += int(wi.size)
    diag={
        'lag_samples':int(lag),
        'transition_mode':mode,
        'candidate_pairs':int(n_pairs),
        'used_transitions':int(n_used),
        'bad_state_pairs':int(n_bad_state),
        'window_skipped_pairs':int(n_window_skipped),
        'nonzero_count_entries':int(np.count_nonzero(C)),
    }
    return C, diag


def _dtram_microstate_bias_from_samples(d: Data, states: np.ndarray, n_states: int,
                                        include_gamd: bool = True) -> tuple[np.ndarray, dict]:
    """Average reduced bias u_nk over samples in each discrete microstate.

    The umbrella part is the existing all-window reduced bias matrix.  When
    GaMD is included, the recorded per-frame boost is added to every window
    column as beta*DeltaV(frame), matching globally shared GaMD boost parameters.
    Missing microstates are filled by nearest observed microstate in CV-bin
    index space so the optimizer has finite bias values for all retained states.
    """
    states=np.asarray(states,dtype=np.int64)
    K=int(d.u_nk.shape[1])
    M=int(n_states)
    b=np.full((M,K),np.nan,dtype=np.float64)
    counts=np.zeros(M,dtype=np.int64)
    total_bias=np.asarray(d.u_nk,dtype=np.float64).copy()
    used_gamd=False
    if bool(include_gamd) and d.boost_kj.shape == d.cv.shape and np.isfinite(d.boost_kj).sum() > 10 and np.nanstd(d.boost_kj)>1.0e-12:
        total_bias=total_bias + d.beta*np.asarray(d.boost_kj,dtype=np.float64)[:,None]
        used_gamd=True
    for i in range(M):
        mask=states==i
        counts[i]=int(np.count_nonzero(mask))
        if counts[i] > 0:
            with np.errstate(invalid='ignore'):
                b[i,:]=np.nanmean(total_bias[mask,:],axis=0)
    observed=np.where(np.all(np.isfinite(b),axis=1))[0]
    if observed.size == 0:
        raise ValueError('dTRAM could not build any finite microstate bias rows')
    missing=np.where(~np.all(np.isfinite(b),axis=1))[0]
    for i in missing:
        j=int(observed[np.argmin(np.abs(observed-i))])
        b[i,:]=b[j,:]
    diag={'include_gamd':bool(include_gamd),'used_gamd':bool(used_gamd),'observed_microstates':int(observed.size),'filled_missing_microstates':int(missing.size),'microstate_sample_counts':[int(x) for x in counts]}
    return b, diag


def _dtram_reversible_profile_loglik_grad(C: np.ndarray, pi: np.ndarray,
                                          maxiter: int = 1000, tol: float = 1e-10) -> tuple[float, np.ndarray, dict]:
    """Profile reversible transition likelihood at fixed stationary pi.

    Solves for symmetric equilibrium fluxes q_ij with row sums pi_i and returns
    the maximized transition log likelihood plus dL/dpi.  The implementation is
    based on the standard fixed-stationary reversible MLE KKT equations:

        q_ii = c_ii / lambda_i
        q_ij = (c_ij + c_ji) / (lambda_i + lambda_j), i != j

    with lambda chosen so sum_j q_ij = pi_i.
    """
    C=np.asarray(C,dtype=np.float64)
    pi=np.asarray(pi,dtype=np.float64)
    M=int(C.shape[0])
    if M == 0 or float(np.sum(C)) <= 0:
        return 0.0, np.zeros(M,dtype=np.float64), {'iterations':0,'max_residual':0.0}
    if M == 1:
        return 0.0, np.zeros(1,dtype=np.float64), {'iterations':0,'max_residual':0.0}
    pi=np.clip(pi,1.0e-300,1.0)
    pi=pi/np.sum(pi)
    D=C+C.T
    np.fill_diagonal(D,0.0)
    cii=np.diag(C).copy()
    nout=np.sum(C,axis=1)
    # Rows with no observations do not constrain the likelihood.  They still
    # consume stationary probability, but their derivative contribution is zero.
    observed=(nout+np.sum(C,axis=0))>0
    if np.count_nonzero(observed) <= 1:
        return 0.0, np.zeros(M,dtype=np.float64), {'iterations':0,'max_residual':0.0}
    lam=np.maximum(nout+np.sum(C,axis=0)+1.0,1.0)/np.maximum(pi,1.0e-300)
    lam=np.asarray(lam,dtype=np.float64)
    max_res=float('inf')
    for it in range(1,int(maxiter)+1):
        denom=lam[:,None]+lam[None,:]
        off=np.divide(D,denom,out=np.zeros_like(D),where=denom>0)
        row=np.sum(off,axis=1)+np.divide(cii,lam,out=np.zeros_like(cii),where=lam>0)
        ratio=np.ones_like(lam)
        m=observed
        ratio[m]=row[m]/np.maximum(pi[m],1.0e-300)
        max_res=float(np.max(np.abs(ratio[m]-1.0))) if np.any(m) else 0.0
        if max_res < float(tol):
            break
        # If row is too large, increase lambda; if too small, decrease.  Clip
        # the multiplicative correction to keep pathological sparse windows sane.
        corr=np.clip(ratio,0.2,5.0)
        lam[m]*=corr[m]
        lam=np.clip(lam,1.0e-300,1.0e300)
    denom=lam[:,None]+lam[None,:]
    q=np.divide(D,denom,out=np.zeros_like(D),where=denom>0)
    diag_q=np.divide(cii,lam,out=np.zeros_like(cii),where=lam>0)
    np.fill_diagonal(q,diag_q)
    # Small correction to enforce row sums numerically for rows with observations.
    row=np.sum(q,axis=1)
    scale=np.ones(M,dtype=np.float64)
    m=observed & (row>0)
    scale[m]=pi[m]/row[m]
    q=(q*scale[:,None]+q*scale[None,:])*0.5
    loglik=0.0
    nz=C>0
    if np.any(nz):
        p=np.divide(q,pi[:,None],out=np.zeros_like(q),where=pi[:,None]>0)
        loglik=float(np.sum(C[nz]*np.log(np.maximum(p[nz],1.0e-300))))
    grad=np.zeros(M,dtype=np.float64)
    grad[observed]=lam[observed]-nout[observed]/np.maximum(pi[observed],1.0e-300)
    return loglik, grad, {'iterations':int(it),'max_residual':float(max_res)}


def solve_dtram(C_kij: np.ndarray, bias_ik: np.ndarray, *, maxiter: int = 200,
                tol: float = 1e-7, inner_maxiter: int = 1000,
                inner_tol: float = 1e-10, initial_prob: Optional[np.ndarray] = None,
                progress: Optional[Progress] = None) -> dict:
    """Solve discrete TRAM by direct profile-likelihood optimization.

    Parameters
    ----------
    C_kij : ndarray, shape (K, M, M)
        Transition counts for each thermodynamic state/window.
    bias_ik : ndarray, shape (M, K)
        Reduced thermodynamic bias for each discrete microstate and window.
    initial_prob : ndarray, optional
        Initial unbiased microstate probability guess.
    """
    if not SCIPY_AVAILABLE or _scipy_minimize is None:
        raise RuntimeError('dTRAM requires scipy.optimize.minimize')
    C=np.asarray(C_kij,dtype=np.float64)
    b=np.asarray(bias_ik,dtype=np.float64)
    if C.ndim != 3:
        raise ValueError('C_kij must have shape (K,M,M)')
    K,M,M2=C.shape
    if M != M2:
        raise ValueError('C_kij must be square in microstate dimensions')
    if b.shape != (M,K):
        raise ValueError(f'bias_ik shape {b.shape} does not match expected {(M,K)}')
    micro_counts=np.sum(C,axis=(0,2))+np.sum(C,axis=(0,1))
    active=np.where((micro_counts>0)&np.all(np.isfinite(b),axis=1))[0]
    if active.size < 2:
        raise ValueError('dTRAM needs at least two active microstates with transitions')
    C=C[:,active[:,None],active[None,:]]
    b=b[active,:]
    M=int(active.size)
    if initial_prob is not None:
        p0=np.asarray(initial_prob,dtype=np.float64)
        p0=p0[active] if p0.size >= int(np.max(active))+1 else np.full(M,1.0/M)
        p0=np.where(np.isfinite(p0)&(p0>0),p0,0.0)
        if np.sum(p0) <= 0:
            p0=np.full(M,1.0/M)
        else:
            p0=p0/np.sum(p0)
    else:
        vc=np.sum(C,axis=(0,2))+1.0
        p0=vc/np.sum(vc)
    eta0=np.log(np.maximum(p0,1.0e-300)); eta0-=eta0[0]
    cache={'n_eval':0,'inner_max_residual':0.0,'inner_max_iterations':0,'history':[]}

    active_by_k=[]
    for k in range(K):
        obs=(np.sum(C[k],axis=1)+np.sum(C[k],axis=0))>0
        active_by_k.append(np.where(obs)[0])

    def objective(red):
        eta=np.empty(M,dtype=np.float64); eta[0]=0.0; eta[1:]=np.asarray(red,dtype=np.float64)
        total_ll=0.0
        grad_eta=np.zeros(M,dtype=np.float64)
        for k in range(K):
            A=active_by_k[k]
            if A.size < 2 or float(np.sum(C[k])) <= 0:
                continue
            mu=_dtram_softmax(eta-b[:,k])
            zA=float(np.sum(mu[A]))
            if zA <= 0 or not np.isfinite(zA):
                continue
            piA=mu[A]/zA
            ll,gA,idiag=_dtram_reversible_profile_loglik_grad(C[k][np.ix_(A,A)],piA,maxiter=inner_maxiter,tol=inner_tol)
            total_ll += ll
            cache['inner_max_residual']=max(float(cache.get('inner_max_residual',0.0)),float(idiag.get('max_residual',0.0)))
            cache['inner_max_iterations']=max(int(cache.get('inner_max_iterations',0)),int(idiag.get('iterations',0)))
            h=np.zeros(M,dtype=np.float64)
            centered_gA=gA-float(np.sum(gA*piA))
            h[A]=centered_gA/zA
            hbar=float(np.sum(h*mu))
            grad_eta += mu*(h-hbar)
        red_grad=-grad_eta[1:].copy()
        cache['n_eval'] += 1
        cache['history'].append({
            'evaluation':int(cache['n_eval']),
            'negative_log_likelihood':-float(total_ll),
            'gradient_norm':float(np.max(np.abs(red_grad))) if red_grad.size else 0.0,
            'inner_max_residual':float(cache.get('inner_max_residual',np.nan)),
            'inner_max_iterations':int(cache.get('inner_max_iterations',0)),
        })
        return -float(total_ll), red_grad

    if progress is not None:
        progress.step('dTRAM', f'optimizing {M} active microstates across {K} windows')
    res=_scipy_minimize(objective, eta0[1:], method='L-BFGS-B', jac=True,
                        options={'maxiter':int(maxiter),'gtol':float(tol),'ftol':1.0e-12,'maxls':50})
    eta=np.empty(M,dtype=np.float64); eta[0]=0.0; eta[1:]=res.x
    logp_active=eta-_dtram_logsumexp_vec(eta)
    p_active=np.exp(logp_active)
    p_full=np.zeros(int(bias_ik.shape[0]),dtype=np.float64)
    p_full[active]=p_active
    logp_full=np.full(int(bias_ik.shape[0]),-np.inf,dtype=np.float64)
    logp_full[active]=logp_active
    return {
        'probability':p_full,
        'log_probability':logp_full,
        'active_microstates':[int(x) for x in active],
        'converged':bool(res.success),
        'message':str(res.message),
        'iterations':int(getattr(res,'nit',0)),
        'objective':float(getattr(res,'fun',np.nan)),
        'gradient_norm':float(np.max(np.abs(getattr(res,'jac',np.asarray([np.nan]))))) if getattr(res,'jac',None) is not None else float('nan'),
        'function_evaluations':int(cache.get('n_eval',0)),
        'inner_max_residual':float(cache.get('inner_max_residual',np.nan)),
        'inner_max_iterations':int(cache.get('inner_max_iterations',0)),
        'history':list(cache.get('history',[])),
    }


def dtram_pmf_from_probability(prob: np.ndarray, edges: np.ndarray, counts: np.ndarray, kbt_kcal: float) -> dict:
    prob=np.asarray(prob,dtype=np.float64)
    counts=np.asarray(counts,dtype=np.int64)
    p=np.where(np.isfinite(prob)&(prob>=0),prob,0.0)
    ps=float(np.sum(p))
    if ps > 0:
        p=p/ps
    with np.errstate(divide='ignore',invalid='ignore'):
        F=-float(kbt_kcal)*np.log(p)
    finite=np.isfinite(F)
    if np.any(finite):
        F-=np.nanmin(F[finite])
    centers=0.5*(np.asarray(edges[:-1],dtype=np.float64)+np.asarray(edges[1:],dtype=np.float64))
    return {'cv_A':centers,'prob':p,'pmf':F,'counts':counts.astype(int)}


def _parse_dtram_lag_scan(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        raw_items=value
    else:
        txt=str(value).strip()
        if not txt:
            return []
        raw_items=txt.replace(';',',').split(',')
    lags=[]
    for item in raw_items:
        try:
            lag=int(float(str(item).strip()))
        except Exception:
            continue
        if lag > 0 and lag not in lags:
            lags.append(lag)
    return lags


def _dtram_window_microstate_counts(d: Data, states: np.ndarray, n_states: int) -> np.ndarray:
    K=int(d.u_nk.shape[1])
    occ=np.zeros((K,int(n_states)),dtype=np.int64)
    st=np.asarray(states,dtype=np.int64)
    win=np.asarray(d.window,dtype=np.int64)
    good=(st>=0)&(st<int(n_states))&(win>=0)&(win<K)
    if np.any(good):
        lin=win[good]*int(n_states)+st[good]
        occ=np.bincount(lin,minlength=K*int(n_states)).reshape(K,int(n_states)).astype(np.int64)
    return occ


def _dtram_connectivity_diagnostics(C: np.ndarray) -> dict:
    C=np.asarray(C,dtype=np.float64)
    if C.ndim != 3:
        return {'component_count':0,'component_sizes':[],'largest_component_size':0,'disconnected':False}
    M=int(C.shape[1])
    W=np.sum(C,axis=0)
    support=(W+W.T)>0
    node_weight=np.sum(W,axis=0)+np.sum(W,axis=1)
    active=[int(x) for x in np.where(node_weight>0)[0]]
    labels=np.full(M,-1,dtype=np.int64)
    components=[]
    for start in active:
        if labels[start] >= 0:
            continue
        cid=len(components)
        stack=[start]
        labels[start]=cid
        comp=[]
        while stack:
            i=stack.pop()
            comp.append(int(i))
            neigh=np.where(support[i])[0]
            for j in neigh:
                j=int(j)
                if labels[j] < 0:
                    labels[j]=cid
                    stack.append(j)
        components.append(sorted(comp))
    sizes=[len(c) for c in components]
    return {
        'component_count':int(len(components)),
        'component_sizes':[int(x) for x in sizes],
        'largest_component_size':int(max(sizes) if sizes else 0),
        'active_microstates_by_transition':int(len(active)),
        'disconnected':bool(len(components)>1),
        'component_labels':[int(x) for x in labels],
        'components':components,
    }


def _dtram_subset_data(d: Data, mask: np.ndarray) -> Data:
    mask=np.asarray(mask,dtype=bool)
    pot=None
    if d.potential_kj is not None and np.asarray(d.potential_kj).shape == d.cv.shape:
        pot=np.asarray(d.potential_kj)[mask]
    return Data(
        d.prod_dir,d.out_dir,
        np.asarray(d.cv)[mask],np.asarray(d.cv2)[mask],np.asarray(d.rg_A)[mask],
        np.asarray(d.window)[mask],np.asarray(d.replica)[mask],np.asarray(d.step)[mask],
        np.asarray(d.u_nk)[mask],d.centers,d.k_kcal,d.beta,d.temp,
        np.asarray(d.boost_kj)[mask],pot,d.source,d.meta,
    )


def _dtram_solve_from_states(d: Data, args, states: np.ndarray, M: int, bins: np.ndarray,
                             kbt_kcal: float, use_gamd: bool, *, lag: int,
                             progress: Optional[Progress] = None,
                             initial_prob: Optional[np.ndarray] = None) -> dict:
    C,cdiag=_dtram_build_transition_counts(d,states,int(M),lag=int(lag),mode=str(getattr(args,'dtram_transition_mode','same-window')))
    cdiag.update(_dtram_connectivity_diagnostics(C))
    if cdiag.get('used_transitions',0) < max(10,int(M)):
        raise ValueError(f'only {cdiag.get("used_transitions",0)} usable transitions for {int(M)} microstates')
    bias,bdiag=_dtram_microstate_bias_from_samples(d,states,int(M),include_gamd=use_gamd)
    counts=np.bincount(np.asarray(states,dtype=np.int64)[np.asarray(states,dtype=np.int64)>=0],minlength=int(M)).astype(np.int64)
    if initial_prob is None:
        initial=(counts.astype(np.float64)+1.0)
        initial/=np.sum(initial)
    else:
        initial=np.asarray(initial_prob,dtype=np.float64)
    sol=solve_dtram(C,bias,maxiter=int(getattr(args,'dtram_maxiter',200) or 200),tol=float(getattr(args,'dtram_tol',1.0e-7) or 1.0e-7),inner_maxiter=int(getattr(args,'dtram_inner_maxiter',1000) or 1000),inner_tol=float(getattr(args,'dtram_inner_tol',1.0e-10) or 1.0e-10),initial_prob=initial,progress=progress)
    method='dtram_gamd' if bdiag.get('used_gamd') else 'dtram_umbrella'
    pmf=dtram_pmf_from_probability(sol['probability'],bins,counts,kbt_kcal)
    return {
        'available':True,'method':method,'pmf':pmf,
        'transition_diagnostics':cdiag,'bias_diagnostics':bdiag,'solver':sol,
        '_C':C,'_bias':bias,'_counts':counts,'_states':np.asarray(states,dtype=np.int64),
        '_use_gamd':bool(use_gamd),'_lag_samples':int(lag),
    }


def _dtram_public_info(obj):
    if isinstance(obj, dict):
        out={}
        for k,v in obj.items():
            if str(k).startswith('_') or k == 'pmf':
                continue
            out[k]=_dtram_public_info(v)
        return out
    if isinstance(obj, list):
        return [_dtram_public_info(v) for v in obj]
    if isinstance(obj, tuple):
        return [_dtram_public_info(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x=float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _write_dtram_lag_sensitivity(d: Data, args, bins: np.ndarray, kbt_kcal: float,
                                  dtram_info: dict, out: Path, warnings: list[str],
                                  progress: Optional[Progress]) -> dict:
    lags=_parse_dtram_lag_scan(getattr(args,'dtram_lag_scan',''))
    base_lag=int(dtram_info.get('_lag_samples',getattr(args,'dtram_lag_samples',1) or 1))
    if not lags:
        return {}
    if base_lag not in lags:
        lags=[base_lag]+lags
    states=np.asarray(dtram_info.get('_states'),dtype=np.int64)
    M=int(len(bins)-1)
    use_gamd=bool(dtram_info.get('_use_gamd',False))
    ref=dtram_info.get('pmf',{})
    ref_prob=pmf_probability(ref)
    ref_F=np.asarray(ref.get('pmf',[]),dtype=np.float64)
    rows=[]
    pmf_by_lag=[]
    if progress is not None:
        progress.step('dTRAM lag scan', f'{len(lags)} lag value(s): {",".join(str(x) for x in lags)}')
    for ii,lag in enumerate(lags, start=1):
        if progress is not None:
            progress.bar('dTRAM lag scan',ii,len(lags),f'lag {lag}')
        try:
            if int(lag) == base_lag:
                res=dtram_info
            else:
                res=_dtram_solve_from_states(d,args,states,M,bins,kbt_kcal,use_gamd,lag=int(lag),progress=None,initial_prob=np.asarray(dtram_info.get('solver',{}).get('probability',None)) if isinstance(dtram_info.get('solver'),dict) and dtram_info.get('solver',{}).get('probability',None) is not None else None)
            pmf=res['pmf']; prob=pmf_probability(pmf); F=np.asarray(pmf['pmf'],dtype=np.float64)
            trans=res.get('transition_diagnostics',{}) or {}; sol=res.get('solver',{}) or {}
            row={'lag_samples':int(lag),'used_transitions':int(trans.get('used_transitions',0) or 0),'candidate_pairs':int(trans.get('candidate_pairs',0) or 0),'component_count':int(trans.get('component_count',0) or 0),'largest_component_size':int(trans.get('largest_component_size',0) or 0),'active_microstates':int(len(sol.get('active_microstates',[]) or [])),'JS_vs_baseline':js_divergence_1d(prob,ref_prob),'RMSE_F_vs_baseline_kcal_mol':pmf_rmse_1d(F,ref_F,prob,ref_prob),'solver_converged':int(bool(sol.get('converged'))),'solver_iterations':int(sol.get('iterations',0) or 0),'solver_gradient_norm':float(sol.get('gradient_norm',np.nan))}
            rows.append(row)
            pmf_by_lag.append((int(lag),pmf))
        except Exception as exc:
            rows.append({'lag_samples':int(lag),'error':str(exc)})
            warnings.append(f'dTRAM lag scan failed at lag {lag}: {exc}')
    if progress is not None:
        progress.bar('dTRAM lag scan',1,1,'writing lag sensitivity outputs',force=True)
    csv_path=out/'dtram_lag_sensitivity.csv'
    _write_csv_rows(csv_path,rows)
    files={'dtram_lag_sensitivity_csv':str(csv_path)}
    try:
        import matplotlib.pyplot as plt
        good=[r for r in rows if not r.get('error')]
        if good:
            x=np.asarray([r['lag_samples'] for r in good],dtype=float)
            js=np.asarray([r.get('JS_vs_baseline',np.nan) for r in good],dtype=float)
            rmse=np.asarray([r.get('RMSE_F_vs_baseline_kcal_mol',np.nan) for r in good],dtype=float)
            used=np.asarray([r.get('used_transitions',np.nan) for r in good],dtype=float)
            comps=np.asarray([r.get('component_count',np.nan) for r in good],dtype=float)
            fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
            axes[0].plot(x,js,marker='o',label='JS')
            axes[0].set_xlabel('dTRAM lag (saved samples)'); axes[0].set_ylabel('JS vs baseline'); axes[0].set_title('Lag sensitivity: probability')
            axes[0].grid(True,alpha=0.25)
            axes[1].plot(x,rmse,marker='o',label='RMSE')
            axes[1].set_xlabel('dTRAM lag (saved samples)'); axes[1].set_ylabel('PMF RMSE vs baseline (kcal/mol)'); axes[1].set_title('Lag sensitivity: PMF shape')
            axes[1].grid(True,alpha=0.25)
            axes[2].plot(x,used,marker='o',label='transitions')
            ax2=axes[2].twinx(); ax2.plot(x,comps,marker='s',linestyle='--',label='components')
            axes[2].set_xlabel('dTRAM lag (saved samples)'); axes[2].set_ylabel('usable transitions'); ax2.set_ylabel('connected components')
            axes[2].set_title('Lag sensitivity: data support'); axes[2].grid(True,alpha=0.25)
            path=out/'dtram_lag_sensitivity.png'
            fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); files['dtram_lag_sensitivity_png']=str(path)
            if pmf_by_lag:
                fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
                for lag,pmf in pmf_by_lag:
                    F=_smooth_pmf_1d(np.asarray(pmf['pmf'],dtype=float),_eff_smooth(args,'pmf_smooth_sigma'))
                    m=np.isfinite(F)
                    if np.any(m): ax.plot(np.asarray(pmf['cv_A'])[m],F[m],lw=1.2,label=f'lag {lag}')
                ax.set_xlabel(_primary_cv_axis_label(d.meta)); ax.set_ylabel('PMF (kcal/mol, shifted)'); ax.set_title('dTRAM PMF overlay across lag scan')
                if len(pmf_by_lag)<=12: ax.legend(frameon=False,fontsize=8,ncol=2)
                ax.grid(True,alpha=0.2)
                path=out/'dtram_lag_pmf_overlay.png'
                fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); files['dtram_lag_pmf_overlay_png']=str(path)
    except Exception as exc:
        warnings.append(f'dTRAM lag sensitivity plot skipped: {exc}')
    return files


def _write_dtram_vs_mbar_convergence(d: Data, args, bins: np.ndarray, kbt_kcal: float,
                                      dtram_info: dict, pmfs: dict, selected: str,
                                      out: Path, warnings: list[str], progress: Optional[Progress],
                                      mbar_result: Optional[dict]) -> dict:
    if bool(getattr(args,'no_convergence',False)):
        return {}
    steps=checkpoint_steps_from_data(d.step,int(getattr(args,'convergence_timepoints',10)))
    if steps.size < 2:
        return {}
    states_full=np.asarray(dtram_info.get('_states'),dtype=np.int64)
    M=int(len(bins)-1)
    use_gamd=bool(dtram_info.get('_use_gamd',False))
    base_lag=int(dtram_info.get('_lag_samples',getattr(args,'dtram_lag_samples',1) or 1))
    ref_dt=dtram_info.get('pmf',{})
    ref_dt_prob=pmf_probability(ref_dt); ref_dt_F=np.asarray(ref_dt.get('pmf',[]),dtype=float)
    ref_mb=pmfs.get(selected)
    if ref_mb is None:
        return {}
    ref_mb_prob=pmf_probability(ref_mb); ref_mb_F=np.asarray(ref_mb.get('pmf',[]),dtype=float)
    rows=[]
    prev_f_k=np.asarray(mbar_result.get('f_k'),dtype=np.float64) if isinstance(mbar_result,dict) and mbar_result.get('f_k') is not None else None
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
    total=int(d.cv.size)
    if progress is not None:
        progress.step('dTRAM vs MBAR', f'{steps.size} prefix timepoints')
    for ci,ck in enumerate(steps,start=1):
        mask=(d.step<=ck)
        n=int(np.count_nonzero(mask))
        frac=float(n/max(1,total))
        if n < max(2,d.u_nk.shape[1]):
            continue
        if progress is not None:
            progress.bar('dTRAM vs MBAR',ci,steps.size,f'step {int(ck)} samples {n}')
        try:
            cache_hit=False; cache_key=None; mb=None
            if cache_enabled:
                cache_key=_convergence_mbar_cache_key(d,args,mask,int(ck),conv_backend,conv_mbar_maxiter,conv_sambar_epochs,conv_sambar_batch,conv_sambar_patience,conv_sambar_seed,conv_sambar_lr,conv_sambar_delta,conv_sambar_polish)
                mb=mbar_cache.get(cache_key); cache_hit=mb is not None
            if mb is None:
                mb=solve_mbar(d.u_nk[mask],d.window[mask],tol=float(args.convergence_mbar_tol),maxiter=conv_mbar_maxiter,progress=None,backend=conv_backend,threads=getattr(args,'mbar_threads',0),f_init=prev_f_k,sambar_epochs=conv_sambar_epochs,sambar_initial_batch_size=conv_sambar_batch,sambar_batch_patience=conv_sambar_patience,sambar_seed=conv_sambar_seed,sambar_lr_scale=conv_sambar_lr,sambar_delta_f_max=conv_sambar_delta,sambar_polish_backend=conv_sambar_polish)
                if cache_enabled and cache_key is not None:
                    mbar_cache[cache_key]=mb
            pmf,_diag,used_method=_observable_pmf_from_logw(d.cv[mask],mb['logw'],d.boost_kj[mask],bins,selected,d.beta,kbt_kcal,smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
            prob=pmf_probability(pmf); F=np.asarray(pmf['pmf'],dtype=float)
            rows.append({'method':'mbar_selected','selected_method':used_method,'checkpoint_index':ci,'checkpoint_step':int(ck),'n_samples':n,'frac_total':frac,'JS':js_divergence_1d(prob,ref_mb_prob),'RMSE_F_kcal_mol':pmf_rmse_1d(F,ref_mb_F,prob,ref_mb_prob,float(getattr(args,'rmse_min_prob',0.0) if hasattr(args,'rmse_min_prob') else 0.0)),'converged':int(bool(mb.get('converged'))),'iterations':int(mb.get('iterations',0) or 0),'max_delta':float(mb.get('max_delta',np.nan)),'cache_hit':int(bool(cache_hit))})
            prev_f_k=mb.get('f_k')
        except Exception as exc:
            rows.append({'method':'mbar_selected','checkpoint_index':ci,'checkpoint_step':int(ck),'n_samples':n,'frac_total':frac,'error':str(exc)})
        try:
            sub=_dtram_subset_data(d,mask)
            res=_dtram_solve_from_states(sub,args,states_full[mask],M,bins,kbt_kcal,use_gamd,lag=base_lag,progress=None,initial_prob=np.asarray(dtram_info.get('solver',{}).get('probability',None)) if isinstance(dtram_info.get('solver'),dict) and dtram_info.get('solver',{}).get('probability',None) is not None else None)
            pmf=res['pmf']; prob=pmf_probability(pmf); F=np.asarray(pmf['pmf'],dtype=float)
            trans=res.get('transition_diagnostics',{}) or {}; sol=res.get('solver',{}) or {}
            rows.append({'method':'dtram','selected_method':res.get('method','dtram'),'checkpoint_index':ci,'checkpoint_step':int(ck),'n_samples':n,'frac_total':frac,'JS':js_divergence_1d(prob,ref_dt_prob),'RMSE_F_kcal_mol':pmf_rmse_1d(F,ref_dt_F,prob,ref_dt_prob,float(getattr(args,'rmse_min_prob',0.0) if hasattr(args,'rmse_min_prob') else 0.0)),'used_transitions':int(trans.get('used_transitions',0) or 0),'component_count':int(trans.get('component_count',0) or 0),'converged':int(bool(sol.get('converged'))),'iterations':int(sol.get('iterations',0) or 0),'max_delta':float(sol.get('gradient_norm',np.nan))})
        except Exception as exc:
            rows.append({'method':'dtram','checkpoint_index':ci,'checkpoint_step':int(ck),'n_samples':n,'frac_total':frac,'error':str(exc)})
    if progress is not None:
        progress.bar('dTRAM vs MBAR',1,1,'writing convergence comparison',force=True)
    csv_path=out/'dtram_vs_mbar_convergence.csv'
    _write_csv_rows(csv_path,rows)
    files={'dtram_vs_mbar_convergence_csv':str(csv_path)}
    try:
        import matplotlib.pyplot as plt
        good=[r for r in rows if not r.get('error') and 'JS' in r]
        if good:
            fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
            for method in ['mbar_selected','dtram']:
                rr=[r for r in good if r.get('method')==method]
                if not rr:
                    continue
                x=np.asarray([r['frac_total'] for r in rr],dtype=float)
                js=np.asarray([r.get('JS',np.nan) for r in rr],dtype=float)
                rmse=np.asarray([r.get('RMSE_F_kcal_mol',np.nan) for r in rr],dtype=float)
                label='MBAR selected' if method == 'mbar_selected' else 'dTRAM'
                axes[0].plot(x,js,marker='o',label=label)
                axes[1].plot(x,rmse,marker='o',label=label)
            axes[0].set_xlabel('fraction of production samples'); axes[0].set_ylabel('JS vs final same-method PMF'); axes[0].set_title('Convergence-through-time: probability')
            axes[1].set_xlabel('fraction of production samples'); axes[1].set_ylabel('PMF RMSE vs final (kcal/mol)'); axes[1].set_title('Convergence-through-time: free energy')
            for ax in axes:
                ax.grid(True,alpha=0.25); ax.legend(frameon=False)
            path=out/'dtram_vs_mbar_convergence.png'
            fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); files['dtram_vs_mbar_convergence_png']=str(path)
    except Exception as exc:
        warnings.append(f'dTRAM vs MBAR convergence plot skipped: {exc}')
    return files


def write_dtram_diagnostic_outputs(d: Data, args, bins: np.ndarray, kbt_kcal: float,
                                   dtram_info: dict, pmfs: dict, selected: str,
                                   out: Path, warnings: list[str], progress: Optional[Progress],
                                   mbar_result: Optional[dict] = None,
                                   full_dataset_note: str = '') -> dict:
    files={}
    if not isinstance(dtram_info,dict) or not dtram_info.get('available'):
        return files
    C=dtram_info.get('_C')
    states=dtram_info.get('_states')
    if C is None or states is None:
        return files
    C=np.asarray(C,dtype=np.float64)
    states=np.asarray(states,dtype=np.int64)
    M=int(len(bins)-1)
    centers=0.5*(np.asarray(bins[:-1],dtype=float)+np.asarray(bins[1:],dtype=float))
    try:
        import matplotlib.pyplot as plt
        method=dtram_info.get('method','dtram')
        # PMF comparison
        fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
        for name,p in _visible_pmfs(pmfs, selected, args).items():
            if not isinstance(p,dict) or 'pmf' not in p:
                continue
            F=_smooth_pmf_1d(np.asarray(p['pmf'],dtype=float),_eff_smooth(args,'pmf_smooth_sigma'))
            m=np.isfinite(F)
            if np.any(m):
                lw=2.6 if name in {selected,method} else 1.1
                ax.plot(np.asarray(p['cv_A'])[m],F[m],lw=lw,label=name)
        ax.set_xlabel(_primary_cv_axis_label(d.meta)); ax.set_ylabel('PMF (kcal/mol, shifted)'); ax.set_title('dTRAM PMF comparison' + full_dataset_note)
        ax.legend(frameon=False,fontsize=8); ax.grid(True,alpha=0.2)
        path=out/'dtram_pmf_comparison.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); files['dtram_pmf_comparison_png']=str(path)
        # Delta vs selected baseline
        base=pmfs.get(selected); dt=dtram_info.get('pmf')
        if isinstance(base,dict) and isinstance(dt,dict):
            Fb=np.asarray(base.get('pmf',[]),dtype=float); Fd=np.asarray(dt.get('pmf',[]),dtype=float)
            n=min(Fb.size,Fd.size,centers.size)
            m=np.isfinite(Fb[:n])&np.isfinite(Fd[:n])
            if np.any(m):
                delta=Fd[:n]-Fb[:n]
                delta=delta-float(np.nanmedian(delta[m]))
                fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
                ax.axhline(0.0,lw=0.8,ls='--',alpha=0.6)
                ax.plot(centers[:n][m],delta[m],marker='o',ms=3,lw=1.5)
                ax.set_xlabel(_primary_cv_axis_label(d.meta)); ax.set_ylabel('dTRAM - selected PMF (kcal/mol, median-aligned)'); ax.set_title('dTRAM delta PMF vs baseline')
                ax.grid(True,alpha=0.25)
                path=out/'dtram_delta_pmf_vs_baseline.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); files['dtram_delta_pmf_vs_baseline_png']=str(path)
        # Transition matrix
        T=np.sum(C,axis=0)
        fig,ax=plt.subplots(figsize=(6.5,5.5),constrained_layout=True)
        im=ax.imshow(np.log10(T+1.0),origin='lower',aspect='auto')
        ax.set_xlabel('to microstate bin'); ax.set_ylabel('from microstate bin'); ax.set_title('dTRAM transition matrix (sum over windows)' + full_dataset_note)
        fig.colorbar(im,ax=ax,label='log10(count + 1)')
        path=out/'dtram_transition_matrix.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); files['dtram_transition_matrix_png']=str(path)
        # Occupancy matrix
        occ=_dtram_window_microstate_counts(d,states,M)
        fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
        im=ax.imshow(np.log10(occ+1.0),origin='lower',aspect='auto')
        ax.set_xlabel('microstate bin'); ax.set_ylabel('window'); ax.set_title('dTRAM occupancy: window x microstate' + full_dataset_note)
        fig.colorbar(im,ax=ax,label='log10(samples + 1)')
        path=out/'dtram_occupancy_window_microstate.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); files['dtram_occupancy_window_microstate_png']=str(path)
        # Transitions per window
        wtrans=np.sum(C,axis=(1,2))
        fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
        ax.bar(np.arange(wtrans.size),wtrans)
        ax.set_xlabel('window'); ax.set_ylabel('usable transitions'); ax.set_title('dTRAM transition counts per window')
        ax.grid(True,axis='y',alpha=0.25)
        path=out/'dtram_window_transition_counts.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); files['dtram_window_transition_counts_png']=str(path)
        # Solver convergence history
        hist=(dtram_info.get('solver',{}) or {}).get('history',[]) or []
        if hist:
            x=np.asarray([h.get('evaluation',i+1) for i,h in enumerate(hist)],dtype=float)
            obj=np.asarray([h.get('negative_log_likelihood',np.nan) for h in hist],dtype=float)
            grad=np.asarray([h.get('gradient_norm',np.nan) for h in hist],dtype=float)
            fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
            axes[0].plot(x,obj,lw=1.5); axes[0].set_xlabel('objective evaluation'); axes[0].set_ylabel('negative log likelihood'); axes[0].set_title('dTRAM optimizer objective'); axes[0].grid(True,alpha=0.25)
            axes[1].semilogy(x,np.maximum(grad,1.0e-300),lw=1.5); axes[1].set_xlabel('objective evaluation'); axes[1].set_ylabel('gradient norm'); axes[1].set_title('dTRAM optimizer gradient'); axes[1].grid(True,alpha=0.25)
            path=out/'dtram_solver_convergence.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); files['dtram_solver_convergence_png']=str(path)
        # Connectivity graph
        conn=_dtram_connectivity_diagnostics(C)
        labels=np.asarray(conn.get('component_labels',[-1]*M),dtype=int)
        W=np.sum(C,axis=0); U=W+W.T; np.fill_diagonal(U,0.0)
        edges=np.argwhere(np.triu(U,k=1)>0)
        weights=np.asarray([U[i,j] for i,j in edges],dtype=float) if edges.size else np.asarray([],dtype=float)
        if weights.size > 2000:
            keep=np.argsort(weights)[-2000:]
            edges=edges[keep]; weights=weights[keep]
        fig,ax=plt.subplots(figsize=(10,4.8),constrained_layout=True)
        if edges.size:
            wmax=float(np.max(weights)) if np.any(np.isfinite(weights)) else 1.0
            for (i,j),w in zip(edges,weights):
                yy0=labels[i] if labels[i] >= 0 else 0
                yy1=labels[j] if labels[j] >= 0 else 0
                lw=0.3+2.2*math.sqrt(float(w)/max(wmax,1.0e-300))
                ax.plot([centers[i],centers[j]],[yy0,yy1],alpha=0.20,lw=lw)
        active=np.where((np.sum(W,axis=0)+np.sum(W,axis=1))>0)[0]
        y=np.asarray([labels[i] if labels[i] >= 0 else 0 for i in active],dtype=float)
        sizes=20+80*np.sqrt((np.sum(W,axis=0)[active]+np.sum(W,axis=1)[active])/max(1.0,float(np.max(np.sum(W,axis=0)[active]+np.sum(W,axis=1)[active])))) if active.size else []
        if active.size:
            ax.scatter(centers[active],y,s=sizes,zorder=3)
        ax.set_xlabel(_primary_cv_axis_label(d.meta)); ax.set_ylabel('connected component'); ax.set_title(f'dTRAM connectivity graph: {conn.get("component_count",0)} component(s)')
        ax.grid(True,alpha=0.2)
        path=out/'dtram_connectivity_graph.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); files['dtram_connectivity_graph_png']=str(path)
    except Exception as exc:
        warnings.append(f'dTRAM diagnostic plots skipped: {exc}')
    files.update(_write_dtram_lag_sensitivity(d,args,bins,kbt_kcal,dtram_info,out,warnings,progress))
    files.update(_write_dtram_vs_mbar_convergence(d,args,bins,kbt_kcal,dtram_info,pmfs,selected,out,warnings,progress,mbar_result))
    return files


def _dtram_discrete_states_2d(x: np.ndarray, y: np.ndarray, xedges: np.ndarray, yedges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return linear 2D microstate labels and a finite/in-range mask."""
    x=np.asarray(x,dtype=np.float64)
    y=np.asarray(y,dtype=np.float64)
    xedges=np.asarray(xedges,dtype=np.float64)
    yedges=np.asarray(yedges,dtype=np.float64)
    states=np.full(x.shape,-1,dtype=np.int64)
    if xedges.size < 2 or yedges.size < 2 or x.shape != y.shape:
        return states, np.zeros(x.shape,dtype=bool)
    nx=int(xedges.size-1); ny=int(yedges.size-1)
    xi=np.searchsorted(xedges,x,side='right')-1
    yi=np.searchsorted(yedges,y,side='right')-1
    xi[x==xedges[-1]]=nx-1
    yi[y==yedges[-1]]=ny-1
    good=np.isfinite(x)&np.isfinite(y)&(xi>=0)&(xi<nx)&(yi>=0)&(yi<ny)
    states[good]=(xi[good].astype(np.int64,copy=False)*ny + yi[good].astype(np.int64,copy=False))
    return states, good


def dtram_2d_fes_from_probability(prob: np.ndarray, xedges: np.ndarray, yedges: np.ndarray, counts: np.ndarray, kbt_kcal: float) -> dict:
    """Convert flat dTRAM probabilities over a 2D grid into the standard 2D FES dict."""
    xedges=np.asarray(xedges,dtype=np.float64)
    yedges=np.asarray(yedges,dtype=np.float64)
    nx=int(xedges.size-1); ny=int(yedges.size-1)
    M=nx*ny
    p=np.asarray(prob,dtype=np.float64).reshape(-1)[:M]
    c=np.asarray(counts,dtype=np.int64).reshape(-1)[:M]
    if p.size < M:
        p=np.pad(p,(0,M-p.size),constant_values=0.0)
    if c.size < M:
        c=np.pad(c,(0,M-c.size),constant_values=0)
    p=np.where(np.isfinite(p)&(p>=0),p,0.0)
    ps=float(np.sum(p))
    if ps > 0:
        p=p/ps
    p2=p.reshape(nx,ny)
    c2=c.reshape(nx,ny)
    with np.errstate(divide='ignore',invalid='ignore'):
        F=-float(kbt_kcal)*np.log(p2)
    finite=np.isfinite(F)
    if np.any(finite):
        F-=np.nanmin(F[finite])
    return {
        'cv_A':0.5*(xedges[:-1]+xedges[1:]),
        'rg_A':0.5*(yedges[:-1]+yedges[1:]),
        'cv_edges_A':xedges,
        'rg_edges_A':yedges,
        'prob':p2,
        'pmf':F,
        'counts':c2.astype(int),
    }


def _pmf_probability_flat(obj: dict) -> np.ndarray:
    prob=np.asarray((obj or {}).get('prob',[]),dtype=np.float64).reshape(-1)
    prob=np.where(np.isfinite(prob)&(prob>=0),prob,0.0)
    s=float(np.sum(prob))
    return prob/s if s>0 else prob


def _pmf_rmse_flat(F, Fref, P, Pref, min_prob=0.0) -> float:
    return pmf_rmse_1d(np.asarray(F,dtype=np.float64).reshape(-1),np.asarray(Fref,dtype=np.float64).reshape(-1),np.asarray(P,dtype=np.float64).reshape(-1),np.asarray(Pref,dtype=np.float64).reshape(-1),min_prob=min_prob)


def _selected_2d_fes_from_logw(x: np.ndarray, y: np.ndarray, logw: np.ndarray, boost: np.ndarray,
                               xedges: np.ndarray, yedges: np.ndarray, selected: str,
                               beta: float, kbt_kcal: float, args) -> tuple[dict, str]:
    """Build the selected MBAR/GaMD 2D FES for a given subset and fixed grid."""
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    logw=np.asarray(logw,dtype=np.float64); boost=np.asarray(boost,dtype=np.float64)
    base_w=norm_logw(logw)
    boost_ok=np.isfinite(boost).sum()>10 and np.nanstd(boost)>1.0e-12
    selected=str(selected or 'umbrella_only')
    if selected == 'gamd_exponential' and boost_ok:
        return pmf2d_from_weights(x,y,norm_logw(logw+beta*boost),xedges,yedges,kbt_kcal), 'gamd_exponential'
    if selected == 'gamd_cumulant3' and boost_ok:
        fes,_diag=cumulant3_2d(x,y,base_w,boost,xedges,yedges,beta,kbt_kcal,smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
        return fes, 'gamd_cumulant3'
    if selected == 'gamd_cumulant2' and boost_ok:
        fes,_diag=cumulant2_2d(x,y,base_w,boost,xedges,yedges,beta,kbt_kcal,smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
        return fes, 'gamd_cumulant2'
    return pmf2d_from_weights(x,y,base_w,xedges,yedges,kbt_kcal), 'umbrella_only'


def _dtram_2d_solve_from_states(d: Data, args, states: np.ndarray, M: int,
                                xedges: np.ndarray, yedges: np.ndarray,
                                kbt_kcal: float, use_gamd: bool, *, lag: int,
                                progress: Optional[Progress] = None,
                                initial_prob: Optional[np.ndarray] = None) -> dict:
    C,cdiag=_dtram_build_transition_counts(d,states,int(M),lag=int(lag),mode=str(getattr(args,'dtram_transition_mode','same-window')))
    cdiag.update(_dtram_connectivity_diagnostics(C))
    counts=np.bincount(np.asarray(states,dtype=np.int64)[np.asarray(states,dtype=np.int64)>=0],minlength=int(M)).astype(np.int64)
    active_counts=int(np.count_nonzero(counts>0))
    min_used=max(10,min(200,max(2,active_counts)))
    if cdiag.get('used_transitions',0) < min_used:
        raise ValueError(f'only {cdiag.get("used_transitions",0)} usable transitions for {int(M)} 2D microstates ({active_counts} occupied)')
    bias,bdiag=_dtram_microstate_bias_from_samples(d,states,int(M),include_gamd=use_gamd)
    if initial_prob is None:
        initial=(counts.astype(np.float64)+1.0)
        initial/=np.sum(initial)
    else:
        initial=np.asarray(initial_prob,dtype=np.float64)
    sol=solve_dtram(C,bias,maxiter=int(getattr(args,'dtram_maxiter',200) or 200),tol=float(getattr(args,'dtram_tol',1.0e-7) or 1.0e-7),inner_maxiter=int(getattr(args,'dtram_inner_maxiter',1000) or 1000),inner_tol=float(getattr(args,'dtram_inner_tol',1.0e-10) or 1.0e-10),initial_prob=initial,progress=progress)
    method='dtram_gamd' if bdiag.get('used_gamd') else 'dtram_umbrella'
    fes=dtram_2d_fes_from_probability(sol['probability'],xedges,yedges,counts,kbt_kcal)
    return {
        'available':True,'method':method,'fes':fes,
        'transition_diagnostics':cdiag,'bias_diagnostics':bdiag,'solver':sol,
        '_C':C,'_bias':bias,'_counts':counts,'_states':np.asarray(states,dtype=np.int64),
        '_xedges':np.asarray(xedges,dtype=np.float64),'_yedges':np.asarray(yedges,dtype=np.float64),
        '_use_gamd':bool(use_gamd),'_lag_samples':int(lag),'_M':int(M),
    }


def _write_dtram_2d_lag_sensitivity(d: Data, args, xedges: np.ndarray, yedges: np.ndarray, kbt_kcal: float,
                                     dtram_info: dict, prefix: str, out: Path, warnings: list[str],
                                     progress: Optional[Progress]) -> dict:
    lags=_parse_dtram_lag_scan(getattr(args,'dtram_lag_scan',''))
    base_lag=int(dtram_info.get('_lag_samples',getattr(args,'dtram_lag_samples',1) or 1))
    if not lags:
        return {}
    if base_lag not in lags:
        lags=[base_lag]+lags
    states=np.asarray(dtram_info.get('_states'),dtype=np.int64)
    M=int(dtram_info.get('_M',int((len(xedges)-1)*(len(yedges)-1))))
    use_gamd=bool(dtram_info.get('_use_gamd',False))
    ref=dtram_info.get('fes',{})
    ref_prob=_pmf_probability_flat(ref)
    ref_F=np.asarray(ref.get('pmf',[]),dtype=np.float64).reshape(-1)
    rows=[]
    if progress is not None:
        progress.step(f'{prefix} dTRAM lag scan', f'{len(lags)} lag value(s): {",".join(str(x) for x in lags)}')
    for ii,lag in enumerate(lags,start=1):
        if progress is not None:
            progress.bar(f'{prefix} dTRAM lag scan',ii,len(lags),f'lag {lag}')
        try:
            if int(lag)==base_lag:
                res=dtram_info
            else:
                init=None
                if isinstance(dtram_info.get('solver'),dict) and dtram_info['solver'].get('probability') is not None:
                    init=np.asarray(dtram_info['solver']['probability'],dtype=np.float64)
                res=_dtram_2d_solve_from_states(d,args,states,M,xedges,yedges,kbt_kcal,use_gamd,lag=int(lag),progress=None,initial_prob=init)
            fes=res.get('fes',{})
            prob=_pmf_probability_flat(fes); F=np.asarray(fes.get('pmf',[]),dtype=np.float64).reshape(-1)
            trans=res.get('transition_diagnostics',{}) or {}; sol=res.get('solver',{}) or {}
            rows.append({'lag_samples':int(lag),'method':res.get('method','dtram'),'JS_vs_base_lag':js_divergence_1d(prob,ref_prob),'RMSE_F_kcal_mol_vs_base_lag':_pmf_rmse_flat(F,ref_F,prob,ref_prob,float(getattr(args,'rmse_min_prob',0.0) if hasattr(args,'rmse_min_prob') else 0.0)),'used_transitions':int(trans.get('used_transitions',0) or 0),'component_count':int(trans.get('component_count',0) or 0),'largest_component_size':int(trans.get('largest_component_size',0) or 0),'converged':int(bool(sol.get('converged'))),'iterations':int(sol.get('iterations',0) or 0),'gradient_norm':float(sol.get('gradient_norm',np.nan))})
        except Exception as exc:
            rows.append({'lag_samples':int(lag),'error':str(exc)})
    csv_path=out/f'{prefix}_dtram_lag_sensitivity.csv'
    _write_csv_rows(csv_path,rows)
    files={f'{prefix}_dtram_lag_sensitivity_csv':str(csv_path)}
    try:
        import matplotlib.pyplot as plt
        good=[r for r in rows if not r.get('error')]
        if good:
            x=np.asarray([r['lag_samples'] for r in good],dtype=float)
            js=np.asarray([r.get('JS_vs_base_lag',np.nan) for r in good],dtype=float)
            rmse=np.asarray([r.get('RMSE_F_kcal_mol_vs_base_lag',np.nan) for r in good],dtype=float)
            used=np.asarray([r.get('used_transitions',np.nan) for r in good],dtype=float)
            comps=np.asarray([r.get('component_count',np.nan) for r in good],dtype=float)
            fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
            axes[0].plot(x,js,marker='o'); axes[0].set_xlabel('dTRAM lag (saved samples)'); axes[0].set_ylabel('JS vs base lag'); axes[0].set_title('2D dTRAM lag sensitivity: probability'); axes[0].grid(True,alpha=0.25)
            axes[1].plot(x,rmse,marker='o'); axes[1].set_xlabel('dTRAM lag (saved samples)'); axes[1].set_ylabel('PMF RMSE vs base lag (kcal/mol)'); axes[1].set_title('2D dTRAM lag sensitivity: FES'); axes[1].grid(True,alpha=0.25)
            axes[2].plot(x,used,marker='o',label='transitions'); ax2=axes[2].twinx(); ax2.plot(x,comps,marker='s',linestyle='--',label='components')
            axes[2].set_xlabel('dTRAM lag (saved samples)'); axes[2].set_ylabel('usable transitions'); ax2.set_ylabel('connected components'); axes[2].set_title('2D dTRAM data support'); axes[2].grid(True,alpha=0.25)
            path=out/f'{prefix}_dtram_lag_sensitivity.png'
            fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); files[f'{prefix}_dtram_lag_sensitivity_png']=str(path)
    except Exception as exc:
        warnings.append(f'{prefix} 2D dTRAM lag sensitivity plot skipped: {exc}')
    return files


def _write_dtram_2d_vs_mbar_convergence(d: Data, args, x: np.ndarray, y: np.ndarray, base_logw_final: np.ndarray,
                                         xedges: np.ndarray, yedges: np.ndarray, kbt_kcal: float,
                                         dtram_info: dict, baseline_fes: dict, selected: str,
                                         prefix: str, out: Path, warnings: list[str], progress: Optional[Progress],
                                         mbar_result: Optional[dict]) -> dict:
    if bool(getattr(args,'no_convergence',False)):
        return {}
    valid=np.isfinite(x)&np.isfinite(y)&np.isfinite(base_logw_final)
    steps=checkpoint_steps_from_data(d.step[valid],int(getattr(args,'convergence_timepoints',10)))
    if steps.size < 2:
        return {}
    states_full=np.asarray(dtram_info.get('_states'),dtype=np.int64)
    M=int(dtram_info.get('_M',int((len(xedges)-1)*(len(yedges)-1))))
    use_gamd=bool(dtram_info.get('_use_gamd',False))
    base_lag=int(dtram_info.get('_lag_samples',getattr(args,'dtram_lag_samples',1) or 1))
    ref_dt=dtram_info.get('fes',{})
    ref_dt_prob=_pmf_probability_flat(ref_dt); ref_dt_F=np.asarray(ref_dt.get('pmf',[]),dtype=float).reshape(-1)
    ref_mb=baseline_fes or {}
    ref_mb_prob=_pmf_probability_flat(ref_mb); ref_mb_F=np.asarray(ref_mb.get('pmf',[]),dtype=float).reshape(-1)
    rows=[]
    prev_f_k=np.asarray(mbar_result.get('f_k'),dtype=np.float64) if isinstance(mbar_result,dict) and mbar_result.get('f_k') is not None else None
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
    total=int(np.count_nonzero(valid))
    if progress is not None:
        progress.step(f'{prefix} dTRAM vs MBAR', f'{steps.size} prefix timepoints')
    for ci,ck in enumerate(steps,start=1):
        mask=(d.step<=ck)&valid
        n=int(np.count_nonzero(mask)); frac=float(n/max(1,total))
        if n < max(2,d.u_nk.shape[1]):
            continue
        if progress is not None:
            progress.bar(f'{prefix} dTRAM vs MBAR',ci,steps.size,f'step {int(ck)} samples {n}')
        try:
            cache_hit=False; cache_key=None; mb=None
            if cache_enabled:
                cache_key=_convergence_mbar_cache_key(d,args,mask,int(ck),conv_backend,conv_mbar_maxiter,conv_sambar_epochs,conv_sambar_batch,conv_sambar_patience,conv_sambar_seed,conv_sambar_lr,conv_sambar_delta,conv_sambar_polish)
                mb=mbar_cache.get(cache_key); cache_hit=mb is not None
            if mb is None:
                mb=solve_mbar(d.u_nk[mask],d.window[mask],tol=float(args.convergence_mbar_tol),maxiter=conv_mbar_maxiter,progress=None,backend=conv_backend,threads=getattr(args,'mbar_threads',0),f_init=prev_f_k,sambar_epochs=conv_sambar_epochs,sambar_initial_batch_size=conv_sambar_batch,sambar_batch_patience=conv_sambar_patience,sambar_seed=conv_sambar_seed,sambar_lr_scale=conv_sambar_lr,sambar_delta_f_max=conv_sambar_delta,sambar_polish_backend=conv_sambar_polish)
                if cache_enabled and cache_key is not None:
                    mbar_cache[cache_key]=mb
            fes,used_method=_selected_2d_fes_from_logw(x[mask],y[mask],mb['logw'],d.boost_kj[mask],xedges,yedges,selected,d.beta,kbt_kcal,args)
            prob=_pmf_probability_flat(fes); F=np.asarray(fes.get('pmf',[]),dtype=float).reshape(-1)
            rows.append({'method':'mbar_selected','selected_method':used_method,'checkpoint_index':ci,'checkpoint_step':int(ck),'n_samples':n,'frac_total':frac,'JS':js_divergence_1d(prob,ref_mb_prob),'RMSE_F_kcal_mol':_pmf_rmse_flat(F,ref_mb_F,prob,ref_mb_prob,float(getattr(args,'rmse_min_prob',0.0) if hasattr(args,'rmse_min_prob') else 0.0)),'converged':int(bool(mb.get('converged'))),'iterations':int(mb.get('iterations',0) or 0),'max_delta':float(mb.get('max_delta',np.nan)),'cache_hit':int(bool(cache_hit))})
            prev_f_k=mb.get('f_k')
        except Exception as exc:
            rows.append({'method':'mbar_selected','checkpoint_index':ci,'checkpoint_step':int(ck),'n_samples':n,'frac_total':frac,'error':str(exc)})
        try:
            sub=_dtram_subset_data(d,mask)
            init=None
            if isinstance(dtram_info.get('solver'),dict) and dtram_info['solver'].get('probability') is not None:
                init=np.asarray(dtram_info['solver']['probability'],dtype=np.float64)
            res=_dtram_2d_solve_from_states(sub,args,states_full[mask],M,xedges,yedges,kbt_kcal,use_gamd,lag=base_lag,progress=None,initial_prob=init)
            fes=res['fes']; prob=_pmf_probability_flat(fes); F=np.asarray(fes.get('pmf',[]),dtype=float).reshape(-1)
            trans=res.get('transition_diagnostics',{}) or {}; sol=res.get('solver',{}) or {}
            rows.append({'method':'dtram','selected_method':res.get('method','dtram'),'checkpoint_index':ci,'checkpoint_step':int(ck),'n_samples':n,'frac_total':frac,'JS':js_divergence_1d(prob,ref_dt_prob),'RMSE_F_kcal_mol':_pmf_rmse_flat(F,ref_dt_F,prob,ref_dt_prob,float(getattr(args,'rmse_min_prob',0.0) if hasattr(args,'rmse_min_prob') else 0.0)),'used_transitions':int(trans.get('used_transitions',0) or 0),'component_count':int(trans.get('component_count',0) or 0),'converged':int(bool(sol.get('converged'))),'iterations':int(sol.get('iterations',0) or 0),'max_delta':float(sol.get('gradient_norm',np.nan))})
        except Exception as exc:
            rows.append({'method':'dtram','checkpoint_index':ci,'checkpoint_step':int(ck),'n_samples':n,'frac_total':frac,'error':str(exc)})
    csv_path=out/f'{prefix}_dtram_vs_mbar_convergence.csv'
    _write_csv_rows(csv_path,rows)
    files={f'{prefix}_dtram_vs_mbar_convergence_csv':str(csv_path)}
    try:
        import matplotlib.pyplot as plt
        good=[r for r in rows if not r.get('error') and 'JS' in r]
        if good:
            fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
            for method in ['mbar_selected','dtram']:
                rr=[r for r in good if r.get('method')==method]
                if not rr:
                    continue
                xx=np.asarray([r['frac_total'] for r in rr],dtype=float)
                js=np.asarray([r.get('JS',np.nan) for r in rr],dtype=float)
                rmse=np.asarray([r.get('RMSE_F_kcal_mol',np.nan) for r in rr],dtype=float)
                label='MBAR selected 2D' if method == 'mbar_selected' else 'dTRAM 2D'
                axes[0].plot(xx,js,marker='o',label=label)
                axes[1].plot(xx,rmse,marker='o',label=label)
            axes[0].set_xlabel('fraction of production samples'); axes[0].set_ylabel('JS vs final same-method FES'); axes[0].set_title('2D convergence-through-time: probability')
            axes[1].set_xlabel('fraction of production samples'); axes[1].set_ylabel('FES RMSE vs final (kcal/mol)'); axes[1].set_title('2D convergence-through-time: free energy')
            for ax in axes:
                ax.grid(True,alpha=0.25); ax.legend(frameon=False)
            path=out/f'{prefix}_dtram_vs_mbar_convergence.png'
            fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); files[f'{prefix}_dtram_vs_mbar_convergence_png']=str(path)
    except Exception as exc:
        warnings.append(f'{prefix} 2D dTRAM vs MBAR convergence plot skipped: {exc}')
    return files


def _write_dtram_2d_diagnostic_outputs(d: Data, args, x: np.ndarray, y: np.ndarray, base_logw: np.ndarray,
                                       xedges: np.ndarray, yedges: np.ndarray, kbt_kcal: float,
                                       dtram_info: dict, baseline_fes: dict, selected: str,
                                       prefix: str, title: str, xlabel: str, ylabel: str,
                                       out: Path, warnings: list[str], progress: Optional[Progress],
                                       mbar_result: Optional[dict]) -> dict:
    files={}
    if not isinstance(dtram_info,dict) or not dtram_info.get('available'):
        return files
    fes=dtram_info.get('fes',{})
    method=dtram_info.get('method','dtram')
    C=np.asarray(dtram_info.get('_C'),dtype=np.float64)
    states=np.asarray(dtram_info.get('_states'),dtype=np.int64)
    xedges=np.asarray(xedges,dtype=np.float64); yedges=np.asarray(yedges,dtype=np.float64)
    xc=0.5*(xedges[:-1]+xedges[1:]); yc=0.5*(yedges[:-1]+yedges[1:])
    nx=int(len(xc)); ny=int(len(yc)); M=nx*ny
    csv_path=out/f'{prefix}_dtram.csv'; npz_path=out/f'{prefix}_dtram.npz'; png_path=out/f'{prefix}_dtram.png'
    write_2d_fes_csv(csv_path,fes,method)
    write_2d_fes_npz(npz_path,fes,method)
    plot_files=_plot_2d_fes_multirange(fes['pmf'],xedges,yedges,xc,yc,png_path,f'{title} ({method})',xlabel,ylabel,warnings,smooth_sigma=float(getattr(args,'fes2d_smooth_sigma',0.0) or 0.0),figsize=(8.8,6.6),dpi=220,cmap_name='viridis',contour=True)
    files.update({f'{prefix}_dtram_csv':str(csv_path),f'{prefix}_dtram_npz':str(npz_path),f'{prefix}_dtram_png':str(png_path)})
    files.update({f'{prefix}_dtram_png_{k}':v for k,v in (plot_files or {}).items()})
    try:
        import matplotlib.pyplot as plt
        Fd=np.asarray(fes.get('pmf',[]),dtype=float)
        Fb=np.asarray((baseline_fes or {}).get('pmf',[]),dtype=float)
        # FES comparison and delta heatmaps
        if Fd.shape == Fb.shape and Fd.size:
            finite=np.isfinite(Fd)&np.isfinite(Fb)
            delta=Fd-Fb
            if np.any(finite):
                delta=delta-float(np.nanmedian(delta[finite]))
            vmax=max(1.0,float(np.nanpercentile(np.concatenate([Fd[np.isfinite(Fd)],Fb[np.isfinite(Fb)]]) if np.any(np.isfinite(Fd)) or np.any(np.isfinite(Fb)) else np.asarray([1.0]),95)))
            fig,axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
            for ax,F,ttl in [(axes[0],Fb,'selected MBAR/GaMD'),(axes[1],Fd,'2D dTRAM')]:
                im=ax.imshow(F.T,origin='lower',extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],aspect='auto',vmin=0.0,vmax=vmax)
                ax.set_title(ttl); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); fig.colorbar(im,ax=ax,label='kcal/mol')
            dmax=float(np.nanpercentile(np.abs(delta[finite]),95)) if np.any(finite) else 1.0
            im=axes[2].imshow(delta.T,origin='lower',extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],aspect='auto',vmin=-max(dmax,1e-9),vmax=max(dmax,1e-9),cmap='coolwarm')
            axes[2].set_title('dTRAM - selected (median aligned)'); axes[2].set_xlabel(xlabel); axes[2].set_ylabel(ylabel); fig.colorbar(im,ax=axes[2],label='kcal/mol')
            path=out/f'{prefix}_dtram_pmf_comparison.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); files[f'{prefix}_dtram_pmf_comparison_png']=str(path)
            fig,ax=plt.subplots(figsize=(7.2,5.8),constrained_layout=True)
            im=ax.imshow(delta.T,origin='lower',extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],aspect='auto',vmin=-max(dmax,1e-9),vmax=max(dmax,1e-9),cmap='coolwarm')
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(f'{title}: dTRAM - selected FES')
            fig.colorbar(im,ax=ax,label='kcal/mol, median aligned')
            path=out/f'{prefix}_dtram_delta_pmf_vs_baseline.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); files[f'{prefix}_dtram_delta_pmf_vs_baseline_png']=str(path)
        # Transition matrix
        T=np.sum(C,axis=0)
        fig,ax=plt.subplots(figsize=(6.5,5.5),constrained_layout=True)
        im=ax.imshow(np.log10(T+1.0),origin='lower',aspect='auto')
        ax.set_xlabel('to 2D microstate'); ax.set_ylabel('from 2D microstate'); ax.set_title(f'{prefix} dTRAM transition matrix')
        fig.colorbar(im,ax=ax,label='log10(count + 1)')
        path=out/f'{prefix}_dtram_transition_matrix.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); files[f'{prefix}_dtram_transition_matrix_png']=str(path)
        # Occupancy matrix
        occ=_dtram_window_microstate_counts(d,states,M)
        fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
        im=ax.imshow(np.log10(occ+1.0),origin='lower',aspect='auto')
        ax.set_xlabel('2D microstate'); ax.set_ylabel('window'); ax.set_title(f'{prefix} dTRAM occupancy: window x 2D microstate')
        fig.colorbar(im,ax=ax,label='log10(samples + 1)')
        path=out/f'{prefix}_dtram_occupancy_window_microstate.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); files[f'{prefix}_dtram_occupancy_window_microstate_png']=str(path)
        # Transitions per window
        wtrans=np.sum(C,axis=(1,2))
        fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
        ax.bar(np.arange(wtrans.size),wtrans); ax.set_xlabel('window'); ax.set_ylabel('usable transitions'); ax.set_title(f'{prefix} dTRAM transition counts per window'); ax.grid(True,axis='y',alpha=0.25)
        path=out/f'{prefix}_dtram_window_transition_counts.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); files[f'{prefix}_dtram_window_transition_counts_png']=str(path)
        # Solver convergence
        hist=(dtram_info.get('solver',{}) or {}).get('history',[]) or []
        if hist:
            xx=np.asarray([h.get('evaluation',i+1) for i,h in enumerate(hist)],dtype=float)
            obj=np.asarray([h.get('negative_log_likelihood',np.nan) for h in hist],dtype=float)
            grad=np.asarray([h.get('gradient_norm',np.nan) for h in hist],dtype=float)
            fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
            axes[0].plot(xx,obj,lw=1.5); axes[0].set_xlabel('objective evaluation'); axes[0].set_ylabel('negative log likelihood'); axes[0].set_title(f'{prefix} dTRAM optimizer objective'); axes[0].grid(True,alpha=0.25)
            axes[1].semilogy(xx,np.maximum(grad,1.0e-300),lw=1.5); axes[1].set_xlabel('objective evaluation'); axes[1].set_ylabel('gradient norm'); axes[1].set_title(f'{prefix} dTRAM optimizer gradient'); axes[1].grid(True,alpha=0.25)
            path=out/f'{prefix}_dtram_solver_convergence.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); files[f'{prefix}_dtram_solver_convergence_png']=str(path)
        # Connectivity in physical 2D coordinates
        conn=_dtram_connectivity_diagnostics(C)
        labels=np.asarray(conn.get('component_labels',[-1]*M),dtype=int)
        W=np.sum(C,axis=0); U=W+W.T; np.fill_diagonal(U,0.0)
        edges=np.argwhere(np.triu(U,k=1)>0)
        weights=np.asarray([U[i,j] for i,j in edges],dtype=float) if edges.size else np.asarray([],dtype=float)
        if weights.size > 2500:
            keep=np.argsort(weights)[-2500:]
            edges=edges[keep]; weights=weights[keep]
        fig,ax=plt.subplots(figsize=(7.2,6.2),constrained_layout=True)
        if edges.size:
            wmax=float(np.max(weights)) if np.any(np.isfinite(weights)) else 1.0
            for (i,j),ww in zip(edges,weights):
                ix0=int(i)//ny; iy0=int(i)%ny; ix1=int(j)//ny; iy1=int(j)%ny
                lw=0.2+1.8*math.sqrt(float(ww)/max(wmax,1.0e-300))
                ax.plot([xc[ix0],xc[ix1]],[yc[iy0],yc[iy1]],alpha=0.16,lw=lw)
        active=np.where((np.sum(W,axis=0)+np.sum(W,axis=1))>0)[0]
        if active.size:
            ax.scatter(xc[active//ny],yc[active%ny],c=labels[active],s=16,zorder=3)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(f'{prefix} dTRAM connectivity: {conn.get("component_count",0)} component(s)')
        ax.grid(True,alpha=0.2)
        path=out/f'{prefix}_dtram_connectivity_graph.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig); files[f'{prefix}_dtram_connectivity_graph_png']=str(path)
    except Exception as exc:
        warnings.append(f'{prefix} 2D dTRAM diagnostic plots skipped: {exc}')
    files.update(_write_dtram_2d_lag_sensitivity(d,args,xedges,yedges,kbt_kcal,dtram_info,prefix,out,warnings,progress))
    files.update(_write_dtram_2d_vs_mbar_convergence(d,args,np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64),np.asarray(base_logw,dtype=np.float64),xedges,yedges,kbt_kcal,dtram_info,baseline_fes,selected,prefix,out,warnings,progress,mbar_result))
    return files


def run_dtram_2d_fes(d: Data, args, x: np.ndarray, y: np.ndarray, base_logw: np.ndarray,
                     xedges: np.ndarray, yedges: np.ndarray, kbt_kcal: float,
                     baseline_fes: dict, selected: str, prefix: str, title: str,
                     xlabel: str, ylabel: str, out: Path, warnings: list[str],
                     progress: Optional[Progress], mbar_result: Optional[dict] = None) -> dict:
    """Optional 2D dTRAM FES and diagnostic plot suite for an existing 2D grid."""
    if not (bool(getattr(args,'dtram',False)) and bool(getattr(args,'dtram_2d_fes',False))):
        return {'available':False,'reason':'disabled'}
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    states,good=_dtram_discrete_states_2d(x,y,xedges,yedges)
    nx=int(len(xedges)-1); ny=int(len(yedges)-1); M=nx*ny
    if M < 2:
        return {'available':False,'reason':'fewer than two 2D dTRAM microstates'}
    max_m=int(getattr(args,'dtram_2d_max_microstates',0) or getattr(args,'dtram_max_microstates',200) or 200)
    if M > max_m:
        msg=f'{prefix} 2D dTRAM skipped: {M} microstates ({nx}x{ny}) exceeds --dtram-2d-max-microstates={max_m}. Reduce 2D bins or increase the limit.'
        warnings.append(msg)
        return {'available':False,'reason':msg,'microstates':int(M),'x_bins':int(nx),'y_bins':int(ny)}
    if np.count_nonzero(good) < max(20,d.u_nk.shape[1]):
        return {'available':False,'reason':'too few finite/in-range samples for 2D dTRAM','n_finite':int(np.count_nonzero(good))}
    include_gamd=str(getattr(args,'dtram_gamd','auto') or 'auto').lower()
    boost_available=(np.isfinite(d.boost_kj).sum()>10 and np.nanstd(d.boost_kj)>1.0e-12)
    use_gamd=(include_gamd == 'on') or (include_gamd == 'auto' and boost_available)
    try:
        res=_dtram_2d_solve_from_states(d,args,states,M,xedges,yedges,kbt_kcal,use_gamd,lag=int(getattr(args,'dtram_lag_samples',1) or 1),progress=progress)
        sol=res.get('solver',{}) or {}; trans=res.get('transition_diagnostics',{}) or {}
        if trans.get('disconnected'):
            warnings.append(f'{prefix} 2D dTRAM graph disconnected: {trans.get("component_count")} connected components; largest has {trans.get("largest_component_size")} microstates.')
        if not sol.get('converged'):
            warnings.append(f'{prefix} 2D dTRAM optimizer did not report convergence: {sol.get("message")}; gradient_norm={sol.get("gradient_norm"):.3e}')
        files=_write_dtram_2d_diagnostic_outputs(d,args,x,y,base_logw,xedges,yedges,kbt_kcal,res,baseline_fes,selected,prefix,title,xlabel,ylabel,out,warnings,progress,mbar_result)
        if files:
            res.setdefault('files',{}).update(files)
        res.setdefault('files',{})[f'{prefix}_dtram_summary_json']=str(out/f'{prefix}_dtram_summary.json')
        public=_dtram_public_info(res)
        wjson(out/f'{prefix}_dtram_summary.json',public)
        return res
    except Exception as exc:
        msg=f'{prefix} 2D dTRAM failed: {exc}'
        warnings.append(msg)
        return {'available':False,'reason':msg}

def run_dtram_cv_pmf(d: Data, args, bins: np.ndarray, kbt_kcal: float,
                     warnings: list[str], progress: Optional[Progress]) -> dict:
    """Run optional normal dTRAM over the main 1D CV bins."""
    if not bool(getattr(args,'dtram',False)):
        return {'available':False,'reason':'disabled'}
    states,good=_dtram_discrete_states_1d(d.cv,bins)
    M=int(len(bins)-1)
    if M < 2:
        return {'available':False,'reason':'fewer than two dTRAM microstates'}
    max_m=int(getattr(args,'dtram_max_microstates',200) or 200)
    if M > max_m:
        msg=f'dTRAM skipped: {M} microstates exceeds --dtram-max-microstates={max_m}. Reduce --bins or increase the limit.'
        warnings.append(msg)
        return {'available':False,'reason':msg}
    include_gamd=str(getattr(args,'dtram_gamd','auto') or 'auto').lower()
    boost_available=(np.isfinite(d.boost_kj).sum()>10 and np.nanstd(d.boost_kj)>1.0e-12)
    use_gamd=(include_gamd == 'on') or (include_gamd == 'auto' and boost_available)
    try:
        res=_dtram_solve_from_states(d,args,states,M,bins,kbt_kcal,use_gamd,lag=int(getattr(args,'dtram_lag_samples',1) or 1),progress=progress)
        sol=res.get('solver',{}) or {}
        trans=res.get('transition_diagnostics',{}) or {}
        if trans.get('disconnected'):
            warnings.append(f'dTRAM graph disconnected: {trans.get("component_count")} connected components; largest has {trans.get("largest_component_size")} microstates.')
        if not sol.get('converged'):
            warnings.append(f'dTRAM optimizer did not report convergence: {sol.get("message")}; gradient_norm={sol.get("gradient_norm"):.3e}')
        return res
    except Exception as exc:
        msg=f'dTRAM failed: {exc}'
        warnings.append(msg)
        return {'available':False,'reason':msg}

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

def boost_stats(boost,beta):
    b=boost[np.isfinite(boost)]
    if b.size==0: return {'available':False}
    kc=b/KJ_PER_KCAL; out={'available':True,'n':int(b.size),'mean_kcal_mol':float(np.mean(kc)),'std_kcal_mol':float(np.std(kc)),'min_kcal_mol':float(np.min(kc)),'max_kcal_mol':float(np.max(kc))}
    if b.size>=3 and np.std(b)>0:
        z=(b-np.mean(b))/np.std(b); out['skew']=float(np.mean(z**3)); out['excess_kurtosis']=float(np.mean(z**4)-3.0); out['anharmonicity_score']=float(math.sqrt(out['skew']**2+0.25*out['excess_kurtosis']**2))
    else: out.update({'skew':None,'excess_kurtosis':None,'anharmonicity_score':None})
    w=norm_logw(beta*b); out['boost_reweight_ess']=float(ess(w)); out['boost_reweight_ess_fraction']=float(out['boost_reweight_ess']/b.size)
    return out

def plot_gamd_boost(d, out, warnings):
    """Write gamd_boost_diagnostics.png, gamd_dv_distribution_per_window.png,
    gamd_reweight_quality.png, and gamd_cumulant_quality.png to out/."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        warnings.append(f'matplotlib unavailable; GaMD boost plots skipped: {e}'); return
    boost_comb = d.boost_kj
    if not (np.isfinite(boost_comb).sum() > 10 and np.nanstd(boost_comb) > 1e-12):
        return
    kj_kcal = 1.0 / KJ_PER_KCAL
    kbt_kcal = 1.0 / (d.beta * KJ_PER_KCAL)
    comb_kcal = boost_comb * kj_kcal
    has_dih = (d.boost_dih_kj is not None and np.isfinite(d.boost_dih_kj).sum() > 10)
    dih_kcal = d.boost_dih_kj * kj_kcal if has_dih else None
    tot_kcal = (comb_kcal - dih_kcal) if has_dih else None
    K = d.u_nk.shape[1]
    wins = np.arange(K)
    win_means_comb = np.array([float(np.nanmean(comb_kcal[d.window == k])) if np.any(d.window == k) else np.nan for k in range(K)])
    win_stds_comb  = np.array([float(np.nanstd(comb_kcal[d.window == k]))  if np.any(d.window == k) else np.nan for k in range(K)])
    win_varbdv     = np.array([float(np.var(comb_kcal[d.window == k] / kbt_kcal)) if np.any(d.window == k) else np.nan for k in range(K)])
    if has_dih:
        win_means_dih = np.array([float(np.nanmean(dih_kcal[d.window == k])) if np.any(d.window == k) else np.nan for k in range(K)])
        win_means_tot = win_means_comb - win_means_dih
        _c_pos = comb_kcal.copy(); _c_pos[_c_pos <= 0] = np.nan
        win_frac_dih = np.array([float(np.nanmedian(dih_kcal[d.window == k] / _c_pos[d.window == k])) if np.any(d.window == k) else np.nan for k in range(K)])
    n_panels = 3 if has_dih else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5), constrained_layout=True)
    ax = axes[0]
    bins_hist = np.linspace(0, float(np.nanpercentile(comb_kcal, 99.5)), 80)
    ax.hist(np.clip(comb_kcal, 0, None), bins=bins_hist, color='#2ecc71', alpha=0.5, label='combined', density=True)
    if has_dih:
        ax.hist(np.clip(dih_kcal, 0, None), bins=bins_hist, color='#e05c5c', alpha=0.6, label='dihedral', density=True)
        ax.hist(np.clip(tot_kcal, 0, None), bins=bins_hist, color='#5c82e0', alpha=0.45, label='total-PE', density=True)
    ax.axvline(kbt_kcal, color='k', ls='--', lw=0.9, label='kT')
    ax.set_xlabel('GaMD boost ΔV (kcal/mol)'); ax.set_ylabel('density')
    ax.set_title('GaMD boost distribution'); ax.legend(fontsize=8)
    ax = axes[1]
    if has_dih:
        ax.bar(wins, win_means_dih, label='dihedral', color='#e05c5c', alpha=0.8)
        ax.bar(wins, win_means_tot, bottom=win_means_dih, label='total-PE', color='#5c82e0', alpha=0.8)
    else:
        ax.bar(wins, win_means_comb, label='combined', color='#2ecc71', alpha=0.8)
    ax.errorbar(wins, win_means_comb, yerr=win_stds_comb, fmt='none', color='k', capsize=3)
    ax.set_xlabel('window index'); ax.set_ylabel('mean boost (kcal/mol)')
    ax.set_title('Mean boost per window  (error bars = ±σ)'); ax.legend(fontsize=8)
    if has_dih:
        ax = axes[2]
        ax.bar(wins, win_frac_dih, color='#9b59b6', alpha=0.85)
        med_frac = float(np.nanmedian(win_frac_dih))
        ax.axhline(med_frac, color='k', ls='--', lw=1.2, label=f'median={med_frac:.3f}')
        ax.set_ylim(0, 1); ax.set_xlabel('window index')
        ax.set_ylabel('dihedral / combined (median)'); ax.set_title('Dihedral fraction per window')
        ax.legend(fontsize=8)
    fig.savefig(out / 'gamd_boost_diagnostics.png', dpi=200, bbox_inches='tight'); plt.close(fig)

    win_groups = [comb_kcal[d.window == k] for k in range(K)]
    win_groups = [g[np.isfinite(g)] for g in win_groups]
    valid_wins = [k for k in range(K) if win_groups[k].size >= 4]
    if valid_wins:
        fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(valid_wins)), 4.5), constrained_layout=True)
        vp = ax.violinplot([win_groups[k] for k in valid_wins], positions=valid_wins, widths=0.8,
                            showmeans=True, showextrema=True)
        for body in vp['bodies']:
            body.set_facecolor('#2ecc71'); body.set_alpha(0.55)
        for part in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
            if part in vp: vp[part].set_color('#1e8449')
        ax.axhline(kbt_kcal, color='k', ls='--', lw=0.9, label='kT')
        ax.set_xlabel('window index'); ax.set_ylabel('GaMD boost ΔV (kcal/mol)')
        ax.set_title('ΔV distribution per window (combined boost)')
        ax.legend(fontsize=8)
        fig.savefig(out / 'gamd_dv_distribution_per_window.png', dpi=200, bbox_inches='tight'); plt.close(fig)
    else:
        warnings.append('GaMD ΔV per-window distribution plot skipped: no window has >=4 finite boost samples')

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.bar(wins, win_varbdv, color='#e08c5c', alpha=0.85)
    for thresh, color, lbl in [(1.0, 'green', 'var=1'), (5.0, 'orange', 'var=5'), (10.0, 'red', 'var=10')]:
        ax.axhline(thresh, color=color, ls='--', lw=1, label=lbl)
    ax.set_xlabel('window index'); ax.set_ylabel('var(β·ΔV_combined)')
    ax.set_title('GaMD reweighting quality per window  [↑ = worse ESS]'); ax.legend(fontsize=8)
    fig.savefig(out / 'gamd_reweight_quality.png', dpi=200, bbox_inches='tight'); plt.close(fig)

    def _window_moments(arr_kcal, win_mask):
        a = arr_kcal[win_mask]
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

    win_skew         = np.array([_window_moments(comb_kcal, d.window == k)[0] for k in range(K)])
    win_kurt         = np.array([_window_moments(comb_kcal, d.window == k)[1] for k in range(K)])
    win_anharmonicity = np.array([_window_moments(comb_kcal, d.window == k)[2] for k in range(K)])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    # Panel 1: skewness
    ax = axes[0]
    bar_colors_skew = ['#fa8072' if s >= 0 else '#87ceeb' for s in win_skew]
    ax.bar(wins, win_skew, color=bar_colors_skew, alpha=0.85)
    for thresh, color in [(0.5, 'orange'), (-0.5, 'orange'), (1.0, 'red'), (-1.0, 'red')]:
        ax.axhline(thresh, color=color, ls='--', lw=1.0, label=f'{thresh:+.1f}' if thresh > 0 else None)
    ax.set_xlabel('window index'); ax.set_ylabel('skewness')
    ax.set_title('Boost skewness per window\n(pre-smoothing, raw ΔV)')
    handles = [plt.Line2D([0], [0], color='orange', ls='--', lw=1, label='±0.5 (marginal)'),
               plt.Line2D([0], [0], color='red',    ls='--', lw=1, label='±1.0 (poor)')]
    ax.legend(handles=handles, fontsize=8)
    # Panel 2: excess kurtosis
    ax = axes[1]
    ax.bar(wins, win_kurt, color='#8ecae6', alpha=0.85)
    for thresh, color in [(1.0, 'orange'), (-1.0, 'orange'), (2.0, 'red'), (-2.0, 'red')]:
        ax.axhline(thresh, color=color, ls='--', lw=1.0)
    ax.set_xlabel('window index'); ax.set_ylabel('excess kurtosis')
    ax.set_title('Boost excess kurtosis per window\n(pre-smoothing, raw ΔV)')
    handles = [plt.Line2D([0], [0], color='orange', ls='--', lw=1, label='±1.0 (marginal)'),
               plt.Line2D([0], [0], color='red',    ls='--', lw=1, label='±2.0 (poor)')]
    ax.legend(handles=handles, fontsize=8)
    # Panel 3: anharmonicity
    ax = axes[2]
    bar_colors_anh = ['#2ecc71' if v < 0.3 else ('#e08c2e' if v < 1.0 else '#e05c5c') for v in win_anharmonicity]
    ax.bar(wins, win_anharmonicity, color=bar_colors_anh, alpha=0.85)
    ax.axhline(0.3, color='green', ls='--', lw=1.0, label='0.3 (marginal)')
    ax.axhline(1.0, color='red',   ls='--', lw=1.0, label='1.0 (poor)')
    ax.set_xlabel('window index'); ax.set_ylabel('anharmonicity score')
    ax.set_title('Cumulant2 validity score per window\n(0: Gaussian; >1: poor cumulant2)')
    import matplotlib.patches as _mpatch
    handles = [_mpatch.Patch(color='#2ecc71', alpha=0.85, label='<0.3 (good)'),
               _mpatch.Patch(color='#e08c2e', alpha=0.85, label='0.3-1.0 (marginal)'),
               _mpatch.Patch(color='#e05c5c', alpha=0.85, label='>1.0 (poor)'),
               plt.Line2D([0], [0], color='green', ls='--', lw=1, label='0.3 threshold'),
               plt.Line2D([0], [0], color='red',   ls='--', lw=1, label='1.0 threshold')]
    ax.legend(handles=handles, fontsize=8)
    fig.savefig(out / 'gamd_cumulant_quality.png', dpi=200, bbox_inches='tight'); plt.close(fig)

def plot_outputs(d,pmfs,selected,O,out,warnings,smooth_sigma=0.0,args=None):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        warnings.append(f'matplotlib unavailable; no PNG plots written: {e}'); return
    import gareus_plotstyle as ps
    _cvlab=_primary_cv_axis_label(d.meta)
    fig,ax=plt.subplots(figsize=(8,5))
    for i,(name,p) in enumerate(_visible_pmfs(pmfs, selected, args).items()):
        pmf_plot=_smooth_pmf_1d(p['pmf'],smooth_sigma); m=np.isfinite(pmf_plot)
        if np.any(m): ps.plot_method_curve(ax,p['cv_A'][m],pmf_plot[m],name,selected,idx=i)
    ps.style_line_axes(ax,xlabel=_cvlab,ylabel='PMF (kcal/mol, shifted)',title='GaREUS PMF estimates'); fig.tight_layout(); fig.savefig(out/'pmf_all_methods.png',dpi=200); plt.close(fig)
    p=pmfs[selected]; pmf_plot=_smooth_pmf_1d(p['pmf'],smooth_sigma); fig,ax=plt.subplots(figsize=(8,5)); m=np.isfinite(pmf_plot); _selc,_=ps.method_style(selected); ax.plot(p['cv_A'][m],pmf_plot[m],linewidth=2.6,color=_selc); ps.annotate_minimum(ax,p['cv_A'][m],pmf_plot[m]); ps.style_line_axes(ax,xlabel=_cvlab,ylabel='PMF (kcal/mol, shifted)',title=f'Selected unbiased PMF: {ps.pretty_method(selected)}',legend=False); fig.tight_layout(); fig.savefig(out/'pmf_unbiased.png',dpi=200); plt.close(fig)
    counts=np.bincount(d.window[(d.window>=0)&(d.window<d.u_nk.shape[1])],minlength=d.u_nk.shape[1]); fig,ax=plt.subplots(figsize=(8,4)); ax.bar(np.arange(counts.size),counts,color=ps.BAR_COLOR); ps.style_line_axes(ax,xlabel='window',ylabel='samples',title='Samples per umbrella window',legend=False); fig.tight_layout(); fig.savefig(out/'window_sample_counts.png',dpi=200); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(O,origin='lower',vmin=0,vmax=1,aspect='auto'); ax.set_xlabel('window'); ax.set_ylabel('window'); ax.set_title('CV histogram overlap'); fig.colorbar(im,ax=ax,label='overlap'); fig.tight_layout(); fig.savefig(out/'overlap_matrix.png',dpi=200); plt.close(fig)
    plot_gamd_boost(d, out, warnings)

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
    dt=s.get('dtram',{}) or {}
    lines += ['', '## dTRAM diagnostics', '']
    if dt.get('available'):
        sol=dt.get('solver',{}) or {}; trans=dt.get('transition_diagnostics',{}) or {}; bd=dt.get('bias_diagnostics',{}) or {}
        lines += [f"Method: **{dt.get('method')}**",f"Optimizer converged: **{sol.get('converged')}** after {sol.get('iterations')} iterations",f"Usable transitions: **{trans.get('used_transitions')}** at lag {trans.get('lag_samples')} sample(s)",f"Active microstates: **{len(sol.get('active_microstates',[]) or [])}**",f"Connectivity components: **{trans.get('component_count','n/a')}** (largest {trans.get('largest_component_size','n/a')})",f"GaMD included in dTRAM bias: **{bd.get('used_gamd')}**"]
    else:
        lines.append(f"dTRAM unavailable: {dt.get('reason','not computed')}")
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
    main_ok = False
    for vmax in CHIGNOLIN_FES_PLOT_VMAX_VALUES_KJ:
        tag = _fes_range_tag(vmax)
        path = _fes_variant_path(out_png, vmax)
        ok = _plot_chignolin_fes_kj_range(F, xedges, yedges, xc, yc, path, method, warnings, range_vmax=vmax)
        if ok:
            files[tag] = str(path)
            if vmax == main_vmax:
                main_ok = True
    if main_ok:
        _plot_chignolin_fes_kj_range(F, xedges, yedges, xc, yc, out_png, method, warnings, range_vmax=main_vmax)
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
        cum_pmf,cdiag=cumulant2(cv2_sel, base_w, boost_sel, bins, d.beta, kbt_kcal, smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
        cum3_pmf,cdiag3=cumulant3(cv2_sel, base_w, boost_sel, bins, d.beta, kbt_kcal, smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
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
        import gareus_plotstyle as ps
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
        fes_cum,cdiag=cumulant2_2d(cv_sel, cv2_sel, base_w, boost_sel, xbins, ybins, d.beta, kbt_kcal, smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
        fes_cum3,cdiag3=cumulant3_2d(cv_sel, cv2_sel, base_w, boost_sel, xbins, ybins, d.beta, kbt_kcal, smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
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
    dtram2d_info=run_dtram_2d_fes(d,args,d.cv,cv2,np.asarray(base_logw,dtype=np.float64),xbins,ybins,kbt_kcal,chosen_fes,chosen,'cv1_cv2_2d_fes',f'CV1 vs {cv2_label} 2D FES',_primary_cv_axis_label(d.meta),cv2_label,out,warnings,progress,mbar_result=None)
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
    if isinstance(dtram2d_info,dict) and dtram2d_info.get('available'):
        info['dtram']=_dtram_public_info(dtram2d_info)
        if isinstance(dtram2d_info.get('files'),dict):
            info['files'].update({k:v for k,v in dtram2d_info.get('files',{}).items()})
    elif bool(getattr(args,'dtram_2d_fes',False)):
        info['dtram']=_dtram_public_info(dtram2d_info)
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

    # Load each segment once, extract needed frames
    # results: list of (event_type, crossing_dict, phi_deg_aligned, psi_deg_aligned)
    results: list = []

    for (rep_id, seg_path_str), batch_items in seg_batches.items():
        seg_path = Path(seg_path_str)
        try:
            traj = md.load(str(seg_path), top=str(top_path), atom_indices=protein_indices)
        except Exception as exc:
            warnings.append(f'Poincare torsions: failed to load segment {seg_path.name} for replica {rep_id}: {exc}')
            continue

        for event_type, crossing, frame_idx in batch_items:
            if frame_idx < 0 or frame_idx >= traj.n_frames:
                continue
            try:
                frame = traj[frame_idx]
                _, phi_rad = md.compute_phi(frame)  # shape (1, n_phi_cols)
                _, psi_rad = md.compute_psi(frame)  # shape (1, n_psi_cols)
                phi_deg = np.degrees(phi_rad[0])    # shape (n_phi_cols,)
                psi_deg = np.degrees(psi_rad[0])    # shape (n_psi_cols,)
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
        fes_cum, _ = cumulant2_2d(x_sel, y_sel, base_w, boost_sel, xbins, ybins, d.beta, kbt_kcal, smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
        fes_cum3, _ = cumulant3_2d(x_sel, y_sel, base_w, boost_sel, xbins, ybins, d.beta, kbt_kcal, smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))
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
    dtram2d_info=run_dtram_2d_fes(d,args,dist1,dist2,np.asarray(base_logw,dtype=np.float64),xbins,ybins,kbt_kcal,chosen_fes,chosen,'chignolin_fes','Chignolin native-contact 2D FES','dist(Asp3N-Thr8O) (A)','dist(Asp3N-Gly7O) (A)',out,warnings,progress,mbar_result=None)
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
    if isinstance(dtram2d_info,dict) and dtram2d_info.get('available'):
        info['dtram']=_dtram_public_info(dtram2d_info)
        if isinstance(dtram2d_info.get('files'),dict):
            info['files'].update({k:v for k,v in dtram2d_info.get('files',{}).items()})
    elif bool(getattr(args,'dtram_2d_fes',False)):
        info['dtram']=_dtram_public_info(dtram2d_info)
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


def run_pmf_and_gamd_boost_report(d: 'Data', args, logw: np.ndarray, bins: np.ndarray,
                                   kbt_kcal: float, out: Path, warnings: list,
                                   progress: Optional['Progress'], warning_prefix: str = '',
                                   extra_pmfs: Optional[dict] = None) -> dict:
    """Core PMF (4 methods) + GaMD boost diagnostics for one Data.

    Extracted from analyze() so the identical formula can run twice against
    different sample subsets sharing the same global MBAR solve (see the
    epoch_000/rest split in analyze()) instead of only ever pooling every
    epoch into one report. ``logw`` is the *raw* per-sample MBAR log-weight
    (``m['logw']``, masked to this subset if applicable) -- renormalized
    internally, same pattern as ``analyze_secondary_cv_pmf``.
    """
    K = d.u_nk.shape[1]
    N = len(d.cv)
    base_w = norm_logw(np.asarray(logw, dtype=np.float64))
    umbrella = pmf_from_weights(d.cv, base_w, bins, kbt_kcal)
    bs = boost_stats(d.boost_kj, d.beta)
    boost_ok = bool(bs.get('available')) and np.nanstd(d.boost_kj) > 1e-12
    if boost_ok:
        exp_w = norm_logw(logw + d.beta * d.boost_kj)
        exp_pmf = pmf_from_weights(d.cv, exp_w, bins, kbt_kcal)
        cum_pmf, cdiag = cumulant2(d.cv, base_w, d.boost_kj, bins, d.beta, kbt_kcal, smooth_logfac_sigma=_eff_smooth(args, 'gamd_smooth_sigma'))
        cum3_pmf, cdiag3 = cumulant3(d.cv, base_w, d.boost_kj, bins, d.beta, kbt_kcal, smooth_logfac_sigma=_eff_smooth(args, 'gamd_smooth_sigma'))
        selected = 'gamd_cumulant2'
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
    write_pmf(out / 'pmf_unbiased.csv', sel, selected, {'boost_mean_kj_mol': _sel_diag.get('boost_mean_kj', np.full(args.bins, np.nan)), 'boost_var_kj2_mol2': _sel_diag.get('boost_var_kj2', np.full(args.bins, np.nan))})
    write_pmf(out / 'pmf_umbrella_only.csv', umbrella, 'umbrella_only')
    write_pmf(out / 'pmf_gamd_exponential.csv', exp_pmf, 'gamd_exponential')
    write_pmf(out / 'pmf_gamd_cumulant2.csv', cum_pmf, 'gamd_cumulant2', {'boost_mean_kj_mol': cdiag.get('boost_mean_kj', np.full(args.bins, np.nan)), 'boost_var_kj2_mol2': cdiag.get('boost_var_kj2', np.full(args.bins, np.nan))})
    write_pmf(out / 'pmf_gamd_cumulant3.csv', cum3_pmf, 'gamd_cumulant3', {'boost_mean_kj_mol': cdiag3.get('boost_mean_kj', np.full(args.bins, np.nan)), 'boost_var_kj2_mol2': cdiag3.get('boost_var_kj2', np.full(args.bins, np.nan))})
    write_all(out / 'pmf_all_methods.csv', pmfs)
    with (out / 'overlap_matrix.csv').open('w', newline='') as f:
        wr = csv.writer(f); wr.writerow(['window'] + list(range(K))); [wr.writerow([i] + [float(x) for x in O[i]]) for i in range(K)]
    n_k_local = np.bincount(d.window[(d.window >= 0) & (d.window < K)], minlength=K)
    with (out / 'window_diagnostics.csv').open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['window', 'center_A', 'k_kcal_mol_A2', 'samples', 'cv_mean_A', 'cv_std_A', 'overlap_left', 'overlap_right']); wr.writeheader()
        for k in range(K):
            vals = d.cv[d.window == k]
            wr.writerow({'window': k, 'center_A': float(d.centers[k]) if k < d.centers.size and np.isfinite(d.centers[k]) else '', 'k_kcal_mol_A2': float(d.k_kcal[k]) if k < d.k_kcal.size and np.isfinite(d.k_kcal[k]) else '', 'samples': int(n_k_local[k]), 'cv_mean_A': float(np.mean(vals)) if vals.size else '', 'cv_std_A': float(np.std(vals)) if vals.size else '', 'overlap_left': float(O[k - 1, k]) if k > 0 else '', 'overlap_right': float(O[k, k + 1]) if k + 1 < K else ''})
    if progress is not None:
        progress.bar('analysis stages', 5, 6, 'plotting PNG outputs', force=True)
    plot_outputs(d, pmfs, selected, O, out, warnings, smooth_sigma=_eff_smooth(args, 'pmf_smooth_sigma'), args=args)
    return {
        'pmfs': pmfs, 'selected': selected, 'boost_ok': boost_ok, 'boost': bs,
        'pmf_span_kcal_mol': span, 'pmf_minimum_cv_A': float(sel['cv_A'][minidx]) if minidx >= 0 else None,
        'neighbor_overlap': neigh, 'n_samples': N, 'O': O,
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

    dtram_info=run_dtram_cv_pmf(d,args,bins,kbt_kcal,warn,progress)
    extra_pmfs={dtram_info.get('method','dtram'):dtram_info['pmf']} if isinstance(dtram_info,dict) and dtram_info.get('available') else None

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
    epoch0_split=_epoch_zero_split_masks(d)
    epoch0_report_info=None
    if epoch0_split is not None:
        mask0,mask_rest=epoch0_split
        d_epoch0=_masked_data(d,mask0)
        out_epoch0=out/'epoch_000_separate'; out_epoch0.mkdir(parents=True,exist_ok=True)
        epoch0_report_info=run_pmf_and_gamd_boost_report(d_epoch0,args,logw[mask0],bins,kbt_kcal,out_epoch0,warn,progress,warning_prefix='[epoch_000 report] ')
        d_main=_masked_data(d,mask_rest); logw_main=logw[mask_rest]
    else:
        d_main=d; logw_main=logw

    if progress is not None: progress.bar('analysis stages', 3, 6, 'overlap diagnostics', force=True)
    main_report_info=run_pmf_and_gamd_boost_report(d_main,args,logw_main,bins,kbt_kcal,out,warn,progress,extra_pmfs=extra_pmfs)
    pmfs=main_report_info['pmfs']; selected=main_report_info['selected']; boost_ok=main_report_info['boost_ok']
    bs=main_report_info['boost']; O=main_report_info['O']; neigh=main_report_info['neighbor_overlap']
    span=main_report_info['pmf_span_kcal_mol']; sel=pmfs[selected]

    if isinstance(dtram_info,dict) and dtram_info.get('available'):
        write_pmf(out/'pmf_dtram.csv',dtram_info['pmf'],dtram_info.get('method','dtram'))
        dtram_info.setdefault('files',{})['pmf_dtram_csv']=str(out/'pmf_dtram.csv')
        if epoch0_split is not None:
            dtram_full_dataset_note=' [full dataset incl. epoch_000]'
            warn.append('[dtram] dtram_pmf_comparison.png / occupancy / transition-matrix panels are computed from the full dataset (including epoch_000); the rest of this report (pmfs, window_diagnostics.csv) excludes epoch_000 -- see epoch_000_report.')
        else:
            dtram_full_dataset_note=''
        dtram_plot_files=write_dtram_diagnostic_outputs(d,args,bins,kbt_kcal,dtram_info,pmfs,selected,out,warn,progress,mbar_result=m,full_dataset_note=dtram_full_dataset_note)
        if dtram_plot_files:
            dtram_info.setdefault('files',{}).update(dtram_plot_files)
        dtram_info.setdefault('files',{})['dtram_summary_json']=str(out/'dtram_summary.json')
        wjson(out/'dtram_summary.json',_dtram_public_info(dtram_info))

    if progress is not None: progress.bar('analysis stages', 4, 6, 'writing CSV outputs', force=True)
    rg_info=analyze_rg(d,args,m,base_w,selected,boost_ok,kbt_kcal,out,warn,progress)
    fes2d_info=analyze_distance_rg_2d_fes(d,args,logw,selected,boost_ok,kbt_kcal,out,warn,progress) if isinstance(rg_info,dict) and rg_info.get('available') else {'available':False,'reason':'Rg analysis unavailable'}
    pca2d_info=analyze_pca_2d_fes(d,args,logw,selected,boost_ok,kbt_kcal,out,warn,progress)
    extra_obs_info=analyze_extra_observable_pmfs(d,args,logw,selected,boost_ok,kbt_kcal,out,warn,progress)
    chignolin_fes_info=analyze_chignolin_fes(d,args,logw,selected,boost_ok,kbt_kcal,out,warn,progress)
    secondary_cv_pmf_info,cv1_cv2_fes_info=run_secondary_cv_analyses(d,args,logw,selected,boost_ok,kbt_kcal,out,warn,progress)
    poincare_info=analyze_poincare_map(d,args,logw,selected,boost_ok,kbt_kcal,out,warn,progress)
    poincare_torsions_info=analyze_poincare_residue_torsions(d,args,out,poincare_info,warn,progress)
    epoch_cv_info=_analyze_epoch_cv_exploration(d.prod_dir,out,d.meta,warn)
    tica_epoch_info=_analyze_tica_epochs(d.prod_dir,out,d.meta,warn)
    torsion_pca_scree_info=_analyze_torsion_pca_scree(d.prod_dir,out,d.meta,args,warn)
    conv_info=run_pmf_convergence(d,args,bins,selected,sel,out,progress=progress,f_init_hint=m.get('f_k'))
    epoch_conv_info={}
    if d.meta.get('_epoch_source'):
        epoch_conv_info=run_epoch_pmf_convergence(d,args,bins,selected,sel,out,progress=progress,f_init_hint=m.get('f_k'))
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
    s['dtram']=_dtram_public_info(dtram_info)
    s['convergence']=conv_info
    for _info in (rg_info,fes2d_info,pca2d_info,extra_obs_info,chignolin_fes_info,
                  poincare_info,poincare_torsions_info,secondary_cv_pmf_info,
                  cv1_cv2_fes_info,epoch_cv_info,tica_epoch_info,torsion_pca_scree_info,dtram_info,conv_info):
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
                   metavar='N', help='Pool only adaptive-production epoch N; repeat to select multiple epochs.')
    p.add_argument('--out', default=None, help='PMF analysis output directory; default: <final_production>/pmf_analysis')
    p.add_argument('--analysis-source', choices=['auto','parquet','npz','csv'], default='auto', help='Analysis input source. auto prefers Parquet chunks (new format) then npz then csv; parquet reads segments.json + samples/*.parquet directly (new gareus package format); npz uses analysis_arrays.npz/analysis_chunks; csv uses samples.csv.')
    p.add_argument('--bins', type=int, default=60)
    p.add_argument('--cv-min', type=float, default=None)
    p.add_argument('--cv-max', type=float, default=None)
    p.add_argument('--min-neighbor-overlap', type=float, default=0.30)
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
    p.add_argument('--dtram', action='store_true', help='Compute an experimental normal discrete TRAM PMF over the main CV bins using per-window transition counts and thermodynamic bias constraints. Does not replace the default selected MBAR/GaMD PMF.')
    p.add_argument('--dtram-2d-fes', action='store_true', help='Also compute experimental 2D dTRAM FES diagnostics for available 2D surfaces (distance-Rg, PCA1-PCA2, CV1-CV2, and chignolin when enabled). Requires --dtram.')
    p.add_argument('--dtram-2d-max-microstates', type=int, default=900, help='Safety limit for 2D dTRAM microstates. For a 30x30 grid this is 900. Increase only if you expect the sparse dTRAM solve to be affordable.')
    p.add_argument('--dtram-lag-samples', type=int, default=1, help='dTRAM lag time in saved analysis samples, not MD steps.')
    p.add_argument('--dtram-lag-scan', default='', help='Comma-separated dTRAM lag values to scan for lag-sensitivity diagnostics, e.g. 1,2,5,10.')
    p.add_argument('--dtram-transition-mode', choices=['same-window','start-window'], default='same-window', help='How to assign transition pairs to thermodynamic states. same-window is conservative for exchange simulations; start-window assigns by the starting sample window.')
    p.add_argument('--dtram-gamd', choices=['auto','on','off'], default='auto', help='Include per-frame GaMD boost in the dTRAM thermodynamic bias. auto enables this when finite boosts are present.')
    p.add_argument('--dtram-max-microstates', type=int, default=200, help='Safety limit for dTRAM microstates. The main 1D CV uses --bins microstates.')
    p.add_argument('--dtram-maxiter', type=int, default=200, help='Maximum outer L-BFGS iterations for dTRAM.')
    p.add_argument('--dtram-tol', type=float, default=1e-7, help='Outer dTRAM optimizer gradient tolerance.')
    p.add_argument('--dtram-inner-maxiter', type=int, default=1000, help='Maximum fixed-stationary reversible MLE iterations inside each dTRAM objective evaluation.')
    p.add_argument('--dtram-inner-tol', type=float, default=1e-10, help='Inner reversible MLE residual tolerance for dTRAM.')
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
        # Update module-level configuration variables via globals() to avoid global declarations
        g = globals()
        g['DEFAULT_MBAR_BACKEND'] = str(getattr(args, 'mbar_backend', g.get('DEFAULT_MBAR_BACKEND')) or g.get('DEFAULT_MBAR_BACKEND'))
        g['SAMBAR_EPOCHS'] = int(getattr(args, 'sambar_epochs', g.get('SAMBAR_EPOCHS')))
        g['SAMBAR_INITIAL_BATCH_SIZE'] = int(getattr(args, 'sambar_initial_batch_size', g.get('SAMBAR_INITIAL_BATCH_SIZE')))
        g['SAMBAR_BATCH_PATIENCE'] = int(getattr(args, 'sambar_batch_patience', g.get('SAMBAR_BATCH_PATIENCE')))
        g['SAMBAR_SEED'] = int(getattr(args, 'sambar_seed', g.get('SAMBAR_SEED')))
        g['SAMBAR_LR_SCALE'] = float(getattr(args, 'sambar_lr_scale', g.get('SAMBAR_LR_SCALE')))
        g['SAMBAR_DELTA_F_MAX'] = float(getattr(args, 'sambar_delta_f_max', g.get('SAMBAR_DELTA_F_MAX')))
        # Ensure polish backend is a string
        g['SAMBAR_POLISH_BACKEND'] = str(getattr(args, 'sambar_polish_backend', g.get('SAMBAR_POLISH_BACKEND')) or g.get('SAMBAR_POLISH_BACKEND'))
        g['MBAR_ANDERSON_HISTORY'] = int(getattr(args, 'mbar_anderson_history', g.get('MBAR_ANDERSON_HISTORY')))
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
